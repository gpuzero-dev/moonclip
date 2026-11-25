//! How often to save between two checkpoints, chosen from what saving costs
//! (GPU-517).
//!
//! A save at every step is what lets a run resume from any step, and it is
//! worth that while a save is cheap. When the state changes everywhere at
//! every step - Adam's moments do - a delta weighs what a full does: on
//! deepseek-1b, 9 GB a step, which no rented node's disk holds for a window
//! and no link carries. Until GPU-517 the caller decided with fixed rules
//! looking into the store from outside: weigh the first deltas, watch the
//! bucket fall behind, guess from the optimizer's size, or a number of steps
//! between deltas picked by hand. Every one ended as all or nothing.
//!
//! The caller now offers every step and asks [`Cadence::wants`] whether to
//! save it. A checkpoint - a step that is a multiple of the window - is always
//! saved; between two, a step is saved once `every` steps have passed since
//! the last save. `every` follows the measurement, the way Ravex's H follows
//! the network in its outer loop: at each checkpoint the window that just
//! closed says how much of its wall clock the writer and the upload were
//! busy, and `every` moves so that the busier of the two sits at
//! [`BUSY_SHARE`] of the window. The fewer the steps between saves the less a
//! resume loses, so `every` comes down whenever the cost leaves room, to a
//! save at every step if it is cheap enough.
//!
//! Nor below what the window can hold: at most `fulls` fulls' worth of bytes
//! (`max_full_snapshots` of the store), judged from the size of the last
//! window's saves. Time alone took a state whose every save is a full down to
//! a full at every step on a fast disk and link - a window of fulls on the node
//! and in the bucket. A state whose deltas are light is not held back by it.
//!
//! It starts at the window, one save per checkpoint, and comes down from
//! there: a first window of saves at every step was the 18 GB GPU-516 found
//! in the bucket before anything had been measured.
//!
//! The same measurement decides between deltas and fulls. The first delta
//! written is weighed against its base: past [`DELTA_WORTH`] the store goes
//! on with fulls, which cost the same and need no base; under it the next
//! save is a full whose bytes stay in memory as the base of the deltas after
//! it. Until then no base is held in memory at all: holding one is a copy of
//! the whole state, 11 GB on deepseek-1b, for deltas that may never pay.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The share of a window's wall clock the writer, or the upload, may be busy.
///
/// Both run beside the training, so a share under one costs the training
/// nothing; at one the next save waits for the writer, or the bucket falls
/// further behind at every window. Half leaves room for a window slower than
/// the last. Chosen, not measured.
pub(crate) const BUSY_SHARE: f64 = 0.5;

/// How far one window may move `every`, up or down, once the first window
/// has been measured: one window's timing is one noisy sample, and a link
/// that stalls once should not double the gap for good. The first
/// measurement moves it as far as it says, or a run whose saves are cheap
/// would take seven windows to come down from a window of 100 to a save at
/// every step.
pub(crate) const MAX_STEP_FACTOR: f64 = 2.0;

/// A change smaller than this fraction of `every` is not made: the cost of a
/// window varies by more than that from one to the next.
pub(crate) const STEP_DEADBAND: f64 = 0.1;

/// The share of its base a delta may weigh and still be worth writing.
///
/// Past it a delta costs nearly what a full does, and a full needs no base -
/// neither on disk, where retention keeps the base as long as a delta needs
/// it, nor in memory. The same line GPU-418 drew in Ravex.
pub(crate) const DELTA_WORTH: f64 = 0.5;

/// Whether deltas pay for this state, as the first one written said.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Deltas {
    /// No delta written yet: no base is held in memory for one.
    Unknown,
    /// Light enough: deltas from here, against a base held in memory.
    Pay,
    /// As heavy as fulls: fulls from here.
    DontPay,
}

#[derive(Debug)]
struct State {
    /// Steps between two saves inside a window.
    every: u64,
    /// Whether a window has been measured yet; see [`MAX_STEP_FACTOR`].
    measured: bool,
    /// The last step [`Cadence::wants`] said yes to.
    last: Option<u64>,
    /// Where the window being measured began: its step, and when.
    opened: Option<(u64, Instant)>,
    /// Saves accepted in that window.
    saves: u32,
    /// What the writer spent on them so far.
    busy: Duration,
    /// When the save being written started, while one is.
    writing_since: Option<Instant>,
    deltas: Deltas,
    /// The next save is a full, to be held in memory as the base: deltas
    /// were just found to pay, and the base they were weighed against was
    /// read off the disk.
    rebase: bool,
    /// Compressed bytes the saves written in the window being measured
    /// took, and how many they were.
    bytes: u64,
    stored: u32,
    /// The size of the last full written.
    full: u64,
}

/// The cadence of one store. See the module documentation.
#[derive(Debug)]
pub(crate) struct Cadence {
    window: u64,
    /// The most a window may hold, in fulls of this state; see
    /// [`Cadence::new`].
    fulls: u64,
    state: Mutex<State>,
}

impl Cadence {
    /// A window of `window` steps, which may hold up to `fulls` fulls' worth
    /// of bytes. Time alone let a state whose every save is a full come down
    /// to a full at every step on a fast disk and link: a window of them,
    /// which on deepseek-1b would be 190 GB on the node and in the bucket.
    /// The bound is in fulls of the state being saved, so a state whose
    /// deltas are light is not held back by it.
    pub(crate) fn new(window: u64, fulls: u64) -> Self {
        let window = window.max(1);
        Cadence {
            window,
            fulls: fulls.max(1),
            state: Mutex::new(State {
                every: window,
                measured: false,
                last: None,
                opened: None,
                saves: 0,
                busy: Duration::ZERO,
                writing_since: None,
                deltas: Deltas::Unknown,
                rebase: false,
                bytes: 0,
                stored: 0,
                full: 0,
            }),
        }
    }

    /// Steps between two saves inside a window, as it stands.
    pub(crate) fn every(&self) -> u64 {
        self.state.lock().unwrap().every
    }

    /// Whether to save `step`. Asked for every step, checkpoints included: a
    /// checkpoint closes the window being measured, and is always saved.
    ///
    /// `upload` is how long the bucket took over the last checkpoint's
    /// upload, or has taken so far if it is still going; None without one.
    pub(crate) fn wants(&self, step: u64, upload: Option<Duration>) -> bool {
        self.wants_at(step, upload, Instant::now())
    }

    fn wants_at(&self, step: u64, upload: Option<Duration>, now: Instant) -> bool {
        let mut s = self.state.lock().unwrap();
        let checkpoint = step % self.window == 0;
        if checkpoint {
            self.close(&mut s, step, upload, now);
        }
        let due = checkpoint || s.last.is_some_and(|last| step >= last.saturating_add(s.every));
        if due {
            s.last = Some(step);
            s.saves += 1;
        }
        due
    }

    /// Measure the window that ends at `step` and open the next.
    fn close(&self, s: &mut State, step: u64, upload: Option<Duration>, now: Instant) {
        let Some((start_step, start)) = s.opened.replace((step, now)) else {
            return;
        };
        // A write still going belongs to both windows: the part so far to
        // this one, the rest to the next.
        let mut busy = std::mem::take(&mut s.busy);
        if let Some(since) = s.writing_since.as_mut() {
            busy += now.saturating_duration_since(*since);
            *since = now;
        }
        let saves = std::mem::take(&mut s.saves);
        let bytes = std::mem::take(&mut s.bytes);
        let stored = std::mem::take(&mut s.stored);
        let wall = now.saturating_duration_since(start);
        // Nothing written, or no time: nothing measured, nothing to move.
        if saves == 0 || step <= start_step || wall.is_zero() {
            return;
        }
        let spent = busy.max(upload.unwrap_or(Duration::ZERO));
        let share = spent.as_secs_f64() / wall.as_secs_f64();
        let timed = next_every(s.every, share, self.window, s.measured);
        let floor = bytes_floor(self.window, self.fulls, bytes, stored, s.full);
        let every = timed.max(floor);
        if every != s.every && crate::remote_sync::verbose_logging() {
            let why = if floor > timed {
                format!(
                    "a window of saves the size of the last ones would hold more than {} fulls",
                    self.fulls
                )
            } else {
                format!("the last window's writes and upload were busy {:.0}% of it", 100.0 * share)
            };
            eprintln!("[Moonclip] A save every {every} steps from step {step} (was {}): {why}.", s.every);
        }
        s.every = every;
        s.measured = true;
    }

    /// The caller's thread spent `took` copying a save's state before
    /// handing it over. Counted with the writer's time: the training loop
    /// waits for it, so it is a cost of the cadence like the write.
    pub(crate) fn copied(&self, took: Duration) {
        self.state.lock().unwrap().busy += took;
    }

    /// A save has started being written.
    pub(crate) fn writing(&self, at: Instant) {
        self.state.lock().unwrap().writing_since = Some(at);
    }

    /// The save being written is done.
    pub(crate) fn written(&self, at: Instant) {
        let mut s = self.state.lock().unwrap();
        if let Some(since) = s.writing_since.take() {
            s.busy += at.saturating_duration_since(since);
        }
    }

    /// Whether the next save must be a full whatever the retention says.
    pub(crate) fn wants_full(&self) -> bool {
        let s = self.state.lock().unwrap();
        s.rebase || s.deltas == Deltas::DontPay
    }

    /// Whether a full just written is worth holding in memory as a base.
    pub(crate) fn keeps_base(&self) -> bool {
        self.state.lock().unwrap().deltas == Deltas::Pay
    }

    /// A full of `bytes` compressed bytes was written: any rebase asked for
    /// is done.
    pub(crate) fn full_written(&self, bytes: u64) {
        let mut s = self.state.lock().unwrap();
        s.rebase = false;
        s.full = bytes;
        s.bytes += bytes;
        s.stored += 1;
    }

    /// A delta of `delta` compressed bytes was written against a base of
    /// `base`. Only the first one is weighed; see the module documentation.
    pub(crate) fn delta_written(&self, delta: u64, base: u64) {
        let mut s = self.state.lock().unwrap();
        s.bytes += delta;
        s.stored += 1;
        if s.full == 0 {
            s.full = base;
        }
        if s.deltas != Deltas::Unknown || base == 0 {
            return;
        }
        let share = delta as f64 / base as f64;
        if share < DELTA_WORTH {
            s.deltas = Deltas::Pay;
            s.rebase = true;
        } else {
            s.deltas = Deltas::DontPay;
            eprintln!(
                "[Moonclip] A delta weighs {:.0}% of a full ({} against {}), so this store \
                 saves fulls from here. Nothing is wrong: a full costs the same and needs \
                 no base.",
                100.0 * share,
                human(delta),
                human(base)
            );
        }
    }
}

/// The fewest steps between saves at which a window of saves the size of the
/// last window's holds at most `fulls` fulls: `window / every` saves of
/// `bytes / stored` each against `fulls * full`. 1 when nothing was measured.
pub(crate) fn bytes_floor(window: u64, fulls: u64, bytes: u64, stored: u32, full: u64) -> u64 {
    if stored == 0 || full == 0 {
        return 1;
    }
    let per_save = bytes as f64 / stored as f64;
    let floor = (window as f64 * per_save / (fulls as f64 * full as f64)).ceil();
    (floor as u64).clamp(1, window)
}

/// Bytes as a person reads them, for the log.
fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// The `every` at which the busier of writer and upload sits at
/// [`BUSY_SHARE`] of a window, given that it was busy `share` of the last one
/// at `current`.
///
/// What a window costs is its saves, `window / every` of them, so the share
/// scales with `1 / every` and the answer is `current * share / BUSY_SHARE`;
/// then bounded by [`MAX_STEP_FACTOR`] once a window has been measured, by 1
/// and the window, and left alone inside [`STEP_DEADBAND`].
pub(crate) fn next_every(current: u64, share: f64, window: u64, measured: bool) -> u64 {
    if !share.is_finite() || share < 0.0 {
        return current;
    }
    let now = current as f64;
    let mut wanted = now * share / BUSY_SHARE;
    if measured {
        wanted = wanted.clamp(now / MAX_STEP_FACTOR, now * MAX_STEP_FACTOR);
    }
    wanted = wanted.clamp(1.0, window as f64);
    if measured && (wanted - now).abs() < STEP_DEADBAND * now {
        return current;
    }
    (wanted.round() as u64).clamp(1, window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_measurement_moves_every_as_far_as_it_says() {
        // Busy 1% of the window at a save per window: a save at every step.
        assert_eq!(next_every(100, 0.01, 100, false), 2);
        assert_eq!(next_every(100, 0.001, 100, false), 1);
        // Busy past the share at one save per window: it stays there.
        assert_eq!(next_every(100, 3.0, 100, false), 100);
    }

    #[test]
    fn later_windows_move_it_at_most_twice_and_not_by_noise() {
        assert_eq!(next_every(10, 0.01, 100, true), 5);
        assert_eq!(next_every(10, 5.0, 100, true), 20);
        assert_eq!(next_every(10, 0.52, 100, true), 10);
        assert_eq!(next_every(80, 5.0, 100, true), 100);
        assert_eq!(next_every(1, 0.0, 100, true), 1);
    }

    #[test]
    fn sizes_are_said_in_the_unit_that_fits() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(30 << 10), "30.0 KiB");
        assert_eq!(human(9 << 30), "9.0 GiB");
    }

    #[test]
    fn the_window_holds_no_more_than_its_fulls() {
        // A full a save, cheap: twenty of them in a window of twenty would be
        // twenty fulls, and five are allowed, so a save every four steps.
        assert_eq!(bytes_floor(20, 5, 10 * 100, 10, 100), 4);
        // Deltas a hundredth of a full: no bound short of a save per step.
        assert_eq!(bytes_floor(20, 5, 100 + 19, 20, 100), 1);
        assert_eq!(bytes_floor(20, 5, 0, 0, 0), 1);
        assert_eq!(bytes_floor(20, 1, 10_000, 1, 100), 20);
    }

    #[test]
    fn a_state_saved_in_fulls_stops_at_the_bound_however_cheap() {
        let cadence = Cadence::new(20, 5);
        let mut clock = Instant::now();
        for step in 1..=100u64 {
            clock += Duration::from_millis(100);
            if cadence.wants_at(step, None, clock) {
                cadence.writing(clock);
                cadence.written(clock + Duration::from_millis(1));
                cadence.full_written(1 << 20);
            }
        }
        assert_eq!(cadence.every(), 4);
    }

    #[test]
    fn a_share_that_measured_nothing_leaves_it_alone() {
        assert_eq!(next_every(10, f64::NAN, 100, true), 10);
        assert_eq!(next_every(10, -1.0, 100, true), 10);
    }

    /// Offer steps `from..=to` at one step a `step_time`, each accepted save
    /// written in `write`, and return the accepted steps.
    fn drive(
        cadence: &Cadence,
        clock: &mut Instant,
        steps: std::ops::RangeInclusive<u64>,
        step_time: Duration,
        write: Duration,
    ) -> Vec<u64> {
        let mut saved = Vec::new();
        for step in steps {
            *clock += step_time;
            if cadence.wants_at(step, None, *clock) {
                saved.push(step);
                cadence.writing(*clock);
                cadence.written(*clock + write);
            }
        }
        saved
    }

    #[test]
    fn cheap_saves_come_down_to_every_step_after_one_window() {
        let cadence = Cadence::new(10, 1000);
        let mut clock = Instant::now();
        let ms = Duration::from_millis;
        // First window: nothing before the first checkpoint.
        assert_eq!(drive(&cadence, &mut clock, 1..=10, ms(100), ms(1)), vec![10]);
        // Second: still one save, which is then measured at 1% busy.
        assert_eq!(drive(&cadence, &mut clock, 11..=20, ms(100), ms(1)), vec![20]);
        assert_eq!(cadence.every(), 1);
        assert_eq!(drive(&cadence, &mut clock, 21..=24, ms(100), ms(1)), vec![21, 22, 23, 24]);
    }

    #[test]
    fn saves_heavier_than_the_window_stay_at_one_per_checkpoint() {
        let cadence = Cadence::new(10, 1000);
        let mut clock = Instant::now();
        let s = Duration::from_secs;
        drive(&cadence, &mut clock, 1..=20, s(1), s(30));
        assert_eq!(cadence.every(), 10);
        assert_eq!(drive(&cadence, &mut clock, 21..=30, s(1), s(30)), vec![30]);
    }

    #[test]
    fn a_slow_upload_holds_the_gap_up_where_the_writer_would_not() {
        let cadence = Cadence::new(10, 1000);
        let mut clock = Instant::now();
        let s = Duration::from_secs;
        drive(&cadence, &mut clock, 1..=19, s(1), Duration::ZERO);
        clock += s(1);
        // The writer was idle; the bucket took the whole window.
        assert!(cadence.wants_at(20, Some(s(10)), clock));
        assert_eq!(cadence.every(), 10);
    }

    #[test]
    fn a_heavy_delta_means_fulls_and_a_light_one_a_rebase() {
        let heavy = Cadence::new(10, 1000);
        assert!(!heavy.wants_full() && !heavy.keeps_base());
        heavy.delta_written(90, 100);
        assert!(heavy.wants_full() && !heavy.keeps_base());

        let light = Cadence::new(10, 1000);
        light.delta_written(5, 100);
        assert!(light.wants_full() && light.keeps_base());
        light.full_written(100);
        assert!(!light.wants_full() && light.keeps_base());
        // Weighed once: a later heavy delta does not undo it.
        light.delta_written(90, 100);
        assert!(!light.wants_full());
    }

    /// Whatever the timings - steps of 0 to 50 ms, writes of nothing to two
    /// seconds, an upload or none - every checkpoint is saved, the gap stays
    /// in [1, window], a save between checkpoints is at least the gap after
    /// the last, and once measured the gap never moves by more than twice.
    #[test]
    fn its_rules_hold_whatever_the_timings() {
        for seed in 1..300u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut next = |below: u64| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % below
            };
            let window = 1 + next(40);
            let cadence = Cadence::new(window, 1000);
            let mut clock = Instant::now();
            let mut last_saved: Option<u64> = None;
            for step in 1..=window * 15 {
                clock += Duration::from_millis(next(50));
                let upload = (next(3) == 0).then(|| Duration::from_millis(next(3000)));
                let (before, measured) = {
                    let s = cadence.state.lock().unwrap();
                    (s.every, s.measured)
                };
                let saved = cadence.wants_at(step, upload, clock);
                let every = cadence.every();
                assert!((1..=window).contains(&every), "seed {seed}: every {every} of {window}");
                if measured && every != before {
                    assert!(every * 2 >= before && every <= before * 2, "seed {seed}: {before} -> {every}");
                }
                if step % window == 0 {
                    assert!(saved, "seed {seed}: checkpoint {step} not saved");
                } else if saved {
                    let last = last_saved.expect("a save between checkpoints before any");
                    assert!(step - last >= before.min(every), "seed {seed}: {last} then {step}");
                }
                if saved {
                    last_saved = Some(step);
                    cadence.writing(clock);
                    cadence.written(clock + Duration::from_millis(next(2000)));
                }
            }
        }
    }

    #[test]
    fn a_resume_mid_window_waits_for_the_next_checkpoint() {
        let cadence = Cadence::new(10, 1000);
        let mut clock = Instant::now();
        let ms = Duration::from_millis;
        assert_eq!(drive(&cadence, &mut clock, 15..=21, ms(10), ms(1)), vec![20]);
    }
}
