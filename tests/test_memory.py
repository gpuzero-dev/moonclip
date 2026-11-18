"""
Resident memory of saves and loads through the Python binding (GPU-443).

``tests/save_memory.rs`` counts the bytes Moonclip keeps alive. This counts
what the kernel sees, which is what an out-of-memory kill is decided on: the
live buffers plus whatever the allocator keeps after they are freed. Before
GPU-443, four full saves of a state left the process 2.3x the state larger
than it started, growing with every save, and peaked at 4.4x; a load peaked
at 2.4x the snapshot.

Each measurement runs in a fresh interpreter, because the high-water mark it
reads (``ru_maxrss``) is the process's whole life: in the pytest process it
would be whatever the largest earlier test reached. Linux only - that is
where the numbers are defined and where the runs that were killed were.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc and ru_maxrss in kilobytes"
)

PYTHON_DIR = os.path.join(os.path.dirname(__file__), "..", "python")

# Small enough to be quick, large enough that a whole extra copy of the state
# stands well clear of the working buffers a save is allowed (a 64 MiB batch).
STATE_MB = 256

_PRELUDE = textwrap.dedent(
    """
    import os, resource, sys, time
    sys.path.insert(0, {python_dir!r})
    import torch
    import moonclip

    def rss_mb():
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0

    def peak_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def manager(root, **extra):
        return moonclip.MoonclipManager(
            storage_root=root, compression_level=1, max_total_snapshots=10,
            max_full_snapshots=10, async_save=True, world_size=1, rank=0, **extra)
    """
)


def _run(body: str, root: str) -> dict:
    code = _PRELUDE.format(python_dir=PYTHON_DIR) + textwrap.dedent(body)
    # glibc's thresholds held where they start, as Ravex holds them for a run:
    # otherwise the first large free raises them and freed buffers stay with
    # the process, which is the allocator's doing, not Moonclip's.
    env = dict(os.environ, MOONCLIP_FSYNC="0", STATE_MB=str(STATE_MB), STORE=root,
               MALLOC_MMAP_THRESHOLD_="131072", MALLOC_TRIM_THRESHOLD_="131072")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return dict(line.split("=", 1) for line in out.stdout.split() if "=" in line)


_SAVES = """
    state = {"t%03d" % i: torch.randn(4 * 2**20) for i in range(int(os.environ["STATE_MB"]) // 16)}
    mgr = manager(os.environ["STORE"], full_every_steps=1)
    start_peak, start = peak_mb(), rss_mb()
    ends = []
    for step in range(1, 5):
        for value in state.values():
            value.add_(1e-3)
        tensors, _ = moonclip.flatten_state_dict({"ravex": state}, "", as_tensors=True)
        mgr.save_tensors(step=step, tensors=tensors, metadata={})
        mgr.flush()
        ends.append(rss_mb() - start)
    print("peak=%f" % (peak_mb() - start_peak))
    print("first=%f" % ends[0])
    print("last=%f" % ends[-1])
"""


def test_full_saves_hold_one_shadow_copy_and_give_it_back(tmp_path):
    got = {k: float(v) / STATE_MB for k, v in _run(_SAVES, str(tmp_path)).items()}
    # One shadow copy of the state, and bounded buffers besides it.
    assert got["peak"] < 1.6, f"a full save peaked at {got['peak']:.2f}x the state"
    # A run that saves only fulls has no use for a kept base, and freed
    # buffers must go back to the system rather than accumulate.
    assert got["first"] < 0.2, f"{got['first']:.2f}x the state still resident after one save"
    assert got["last"] < 0.2, f"{got['last']:.2f}x the state still resident after four saves"


_LOAD = """
    mgr = manager(os.environ["STORE"])
    snap = max(mgr.list_snapshots(), key=lambda s: s["step"])
    start_peak, start = peak_mb(), rss_mb()
    loaded = mgr.load(snap["id"])
    print("peak=%f" % (peak_mb() - start_peak))
    print("held=%f" % (rss_mb() - start))
    print("size=%f" % (sum(len(v) for v in loaded.values()) / 2**20))
"""


def test_a_load_holds_the_snapshot_once(tmp_path):
    _run(_SAVES, str(tmp_path))
    got = _run(_LOAD, str(tmp_path))
    size = float(got["size"])
    assert abs(size - STATE_MB) < 1, got
    peak, held = float(got["peak"]) / size, float(got["held"]) / size
    # The bytearrays handed back are the snapshot, once; the load may hold
    # bounded buffers besides them, not the packs whole.
    assert peak < 1.6, f"a load peaked at {peak:.2f}x the snapshot"
    assert held < 1.2, f"{held:.2f}x the snapshot resident after the load returned"
