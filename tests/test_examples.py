"""The README's quickstart and the examples, run the way a reader runs them (GPU-295).

Each starts as its own process from an empty directory, with the Moonclip this
test session imports. Before this nothing ran them, and they had drifted: the
README's snippet used a `model` and a `train_step` it never defined, and
`examples/ddp_integration.py` stopped at the topology check 0.0.9 added. A
change that breaks one of them now breaks here first.

Not here: `transformer_training.py`, minutes on a CPU, and `s3_training.py`,
which needs a bucket.
"""

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="the examples train a PyTorch model")

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(script, cwd):
    env = {
        **os.environ,
        # The Moonclip this session imports, whichever way it got onto the path.
        "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(free_port()),
    }
    result = subprocess.run(
        [sys.executable, str(script)], cwd=cwd, env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr[-2000:])
    return result.stdout


def quickstart() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", readme, re.S)
    assert match, "the README has no Python block"
    return match.group(1)


def test_the_readme_quickstart_runs_and_resumes(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(quickstart(), encoding="utf-8")

    first = run(script, tmp_path)
    second = run(script, tmp_path)

    assert first.splitlines()[0] == "starting at step 0"
    assert second.splitlines()[0] == "starting at step 4501", "the second run did not resume"


def test_ddp_integration(tmp_path):
    assert "All weights match" in run(EXAMPLES / "ddp_integration.py", tmp_path)


def test_model_integration(tmp_path):
    out = run(EXAMPLES / "model_integration.py", tmp_path)
    assert "Moonclip Checkpoint Stats" in out


def test_transformer_implementation(tmp_path):
    out = run(EXAMPLES / "transformer_implementation.py", tmp_path)
    units = {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}

    def size(label):
        match = re.search(label + r":\s+([\d.]+) (\w+)", out)
        assert match, out[-1500:]
        return float(match.group(1)) * units[match.group(2)]

    # What the stats printed before GPU-295 got wrong: a "naive cost" of
    # saving everything in full that came out below the raw size.
    assert size("On disk") <= size("Raw size")
