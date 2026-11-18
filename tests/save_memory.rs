//! How much memory a save and a load hold, counted by the allocator (GPU-443).
//!
//! A 1B-parameter run with its Adam state was killed by the kernel on a node
//! with room for several copies of that state. The save path held, at once:
//! the shadow copy of the next save taken while the previous save was still
//! being written, that save's own copy, the whole state compressed before a
//! byte reached the file, and the base kept for deltas - which a run saving a
//! full at every step kept too, for a delta that could never come. A load
//! held the compressed pack, its base's pack and everything decompressed out
//! of them.
//!
//! The numbers here are live heap bytes, counted by a global allocator that
//! only this test binary has, not the process's resident set: they say what
//! Moonclip keeps alive, independent of what the system allocator does with
//! freed memory (that half is glibc's thresholds, and the Python test measures it).
//! One test function runs the cases in turn, because the counter is global and
//! parallel tests would count each other's buffers.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use moonclip::coordinator::{Coordinator, CoordinatorConfig};
use moonclip::manifest::{CompressionAlgo, RetentionPolicy};
use moonclip::storage::{LocalStorage, StorageBackend};
use moonclip::tensor::TensorData;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            grew(new_size);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const MIB: usize = 1 << 20;
/// 128 MiB in 4 MiB tensors: twice the save path's 64 MiB batch, so the
/// streaming is exercised and a whole-state copy stands out from the bounded
/// working buffers.
const TENSORS: usize = 32;
const TENSOR_BYTES: usize = 4 * MIB;
const STATE: usize = TENSORS * TENSOR_BYTES;

/// Threads in Moonclip's pool for this test. Each tensor in flight has
/// working buffers a few times its size, so the working set grows with the
/// threads up to a batch's worth, whatever the state. Four keeps it the same
/// on every runner, which is what lets the bound below be tight.
const THREADS: &str = "4";

/// What a save or a load may hold beyond the whole-state copies it is
/// allowed: one 64 MiB batch of blobs (`SAVE_BATCH_BYTES`) and the working
/// buffers of four 4 MiB tensors in flight. Under one more copy of the state,
/// so a whole extra copy cannot hide in it.
const BOUNDED: usize = 64 * MIB + 48 * MIB;

/// Float32 noise that a step nudges, as training does: barely compressible,
/// and a delta between steps pays off.
fn state() -> Vec<TensorData> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..TENSORS)
        .map(|i| {
            let mut data = vec![0u8; TENSOR_BYTES];
            for chunk in data.chunks_exact_mut(4) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                chunk.copy_from_slice(&(((s >> 40) as f32 / 1.0e5) - 5.0).to_le_bytes());
            }
            TensorData {
                name: format!("ravex/models/t{i:03}"),
                shape: vec![TENSOR_BYTES / 4],
                dtype: "float32".into(),
                data,
            }
        })
        .collect()
}

fn nudge(state: &mut [TensorData]) {
    for t in state {
        for chunk in t.data.chunks_exact_mut(4).step_by(3) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap()) + 1e-3;
            chunk.copy_from_slice(&v.to_le_bytes());
        }
    }
}

fn coordinator(dir: &std::path::Path, full_every: u64, keep_base: bool) -> Coordinator {
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir).unwrap());
    Coordinator::new(
        storage,
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 10,
                max_deltas_per_full: 1000,
                full_snapshot_every_steps: full_every,
                max_total_snapshots: Some(10),
            },
            async_save: true,
            keep_base_in_memory: keep_base,
            ..Default::default()
        },
    )
    .unwrap()
}

struct Measured {
    /// Highest live heap over the caller's own state, during the saves.
    peak: usize,
    /// Live heap over the caller's state once each save had drained.
    held: Vec<usize>,
}

/// Save the caller's state `saves` times the way the Python binding does:
/// wait for the previous save, take the shadow copy, hand it over.
fn run_saves(coord: &Coordinator, live: &mut [TensorData], saves: u64) -> Measured {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let mut held = Vec::new();
    for step in 1..=saves {
        nudge(live);
        coord.wait_for_previous_save();
        let shadow = live.to_vec();
        coord.save(step, shadow, HashMap::new()).unwrap();
        coord.flush().unwrap();
        held.push(LIVE.load(Ordering::Relaxed).saturating_sub(baseline));
    }
    // The flushes above make the saves drain one by one; the peak that
    // matters is the one an un-flushed run reaches, so repeat without them.
    for step in saves + 1..=2 * saves {
        nudge(live);
        coord.wait_for_previous_save();
        let shadow = live.to_vec();
        coord.save(step, shadow, HashMap::new()).unwrap();
    }
    coord.flush().unwrap();
    held.push(LIVE.load(Ordering::Relaxed).saturating_sub(baseline));
    Measured {
        peak: PEAK.load(Ordering::Relaxed) - baseline,
        held,
    }
}

fn x(bytes: usize) -> String {
    format!("{:.2}x the state", bytes as f64 / STATE as f64)
}

#[test]
fn saves_and_loads_hold_one_copy_of_the_state_and_keep_nothing_they_were_not_asked_to() {
    // Before anything touches the pool, which reads it once.
    std::env::set_var("MOONCLIP_THREADS", THREADS);
    let mut live = state();

    // ── A full at every step: the shadow copy, and nothing kept after. ──
    {
        let dir = tempfile::tempdir().unwrap();
        let coord = coordinator(dir.path(), 1, true);
        let m = run_saves(&coord, &mut live, 2);
        eprintln!("full every step: peak {}, held {:?}", x(m.peak), m.held.iter().map(|&b| x(b)).collect::<Vec<_>>());
        assert!(
            m.peak <= STATE + BOUNDED,
            "a full save peaked at {} over the caller's state: one shadow copy \
             and bounded buffers are {} at most",
            x(m.peak),
            x(STATE + BOUNDED)
        );
        for (i, &h) in m.held.iter().enumerate() {
            assert!(
                h <= 8 * MIB,
                "after save {} a run saving only fulls still held {}: there is \
                 no delta for a kept base to serve",
                i + 1,
                x(h)
            );
        }
        drop(coord);
    }

    // ── Deltas with the base kept: the base, plus one shadow copy. ──
    let dir = tempfile::tempdir().unwrap();
    {
        let coord = coordinator(dir.path(), 1000, true);
        let m = run_saves(&coord, &mut live, 2);
        eprintln!("deltas, base kept: peak {}, held {:?}", x(m.peak), m.held.iter().map(|&b| x(b)).collect::<Vec<_>>());
        assert!(
            m.peak <= 2 * STATE + BOUNDED,
            "a delta save with its base kept peaked at {}: the base and one \
             shadow copy and bounded buffers are {} at most",
            x(m.peak),
            x(2 * STATE + BOUNDED)
        );
        for (i, &h) in m.held.iter().enumerate() {
            assert!(
                h <= STATE + 8 * MIB,
                "after save {} {} was still held: the kept base is the state, \
                 once, and it must not grow from save to save",
                i + 1,
                x(h)
            );
        }
        drop(coord);
    }

    // ── Deltas without it: one shadow copy, nothing after. ──
    {
        let dir = tempfile::tempdir().unwrap();
        let coord = coordinator(dir.path(), 1000, false);
        let m = run_saves(&coord, &mut live, 2);
        eprintln!("deltas, no base: peak {}, held {:?}", x(m.peak), m.held.iter().map(|&b| x(b)).collect::<Vec<_>>());
        assert!(m.peak <= STATE + BOUNDED, "peaked at {}", x(m.peak));
        assert!(m.held.iter().all(|&h| h <= 8 * MIB), "held {:?}", m.held);
        drop(coord);
    }

    // ── A load of the newest delta from the store written above. ──
    drop(live);
    let coord = coordinator(dir.path(), 1000, false);
    let newest = coord.list_snapshots().into_iter().max_by_key(|s| s.step).unwrap();
    assert!(newest.is_delta, "the case wants a delta to load");
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let loaded = coord.load(newest.id).unwrap();
    let peak = PEAK.load(Ordering::Relaxed) - baseline;
    eprintln!("load of a delta: peak {}", x(peak));
    assert_eq!(loaded.len(), TENSORS);
    assert!(
        peak <= STATE + BOUNDED,
        "loading a delta peaked at {}: the result and bounded buffers are {} \
         at most - not the packs whole beside it",
        x(peak),
        x(STATE + BOUNDED)
    );
}
