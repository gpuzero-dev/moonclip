"""
How long a save holds the training loop, and how much it writes: torch.save,
torch.distributed.checkpoint (DCP) and Moonclip, on the same model.

    python examples/compare_savers.py                 # ~50M parameters, CPU
    python examples/compare_savers.py --width 2048 --layers 8 --device cuda

A small transformer-shaped stack of linear layers trained with AdamW, so that
every parameter and both of Adam's moments change at every step: the least
favourable case for Moonclip, which skips what did not change. Each saver
saves every `--every` steps for `--steps` steps, into a directory of its own;
torch.save and DCP keep only their latest checkpoint, as a run short of disk
would.

"Blocked" is what the training loop waits for: the whole save for torch.save
and DCP, the copy of the tensors for Moonclip, whose hashing, delta and write
run in the background (its `flush()` at the end is timed apart). "Written per
save" is the size of one checkpoint for torch.save and DCP, and for Moonclip
its store's size at the end divided by the saves: the first is a full
snapshot and the rest are deltas against it.

Printed as a Markdown table, with the machine it ran on, so a number in the
README always says where it came from.
"""

import argparse
import os
import platform
import shutil
import tempfile
import time

import torch
import torch.distributed.checkpoint as dcp
from moonclip import CheckpointManager


def build(width, layers, device):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        *[torch.nn.Sequential(torch.nn.Linear(width, 4 * width), torch.nn.GELU(), torch.nn.Linear(4 * width, width)) for _ in range(layers)]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optimizer


def size_of(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def run(saver, args, root):
    model, optimizer = build(args.width, args.layers, args.device)
    directory = os.path.join(root, saver)
    os.makedirs(directory)
    mgr = CheckpointManager(directory) if saver == "moonclip" else None
    blocked = []
    written = []
    for step in range(1, args.steps + 1):
        x = torch.randn(8, args.width, device=args.device)
        loss = model(x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % args.every:
            continue
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if saver == "torch.save":
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(directory, f"step_{step}.pt"),
            )
        elif saver == "dcp":
            dcp.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                checkpoint_id=os.path.join(directory, f"step_{step}"),
            )
        else:
            mgr.save(step=step, model=model, optimizer=optimizer)
        blocked.append(time.perf_counter() - started)
        if mgr is None:
            # Only the latest kept, and its size is what this save wrote.
            for name in os.listdir(directory):
                if name != (f"step_{step}.pt" if saver == "torch.save" else f"step_{step}"):
                    path = os.path.join(directory, name)
                    shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            written.append(size_of(directory))
    flushed = 0.0
    if mgr is not None:
        started = time.perf_counter()
        mgr.flush()
        flushed = time.perf_counter() - started
        written = [size_of(directory) / len(blocked)]
    parameters = sum(p.numel() for p in model.parameters())
    return parameters, blocked, flushed, sum(written) / len(written)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # Not the stale-copy-of-a-gigabyte trap: each saver starts from its own
    # empty directory, removed at the end.
    root = tempfile.mkdtemp(prefix="compare_savers_")
    try:
        rows = []
        for saver in ("torch.save", "dcp", "moonclip"):
            parameters, blocked, flushed, size = run(saver, args, root)
            rows.append((saver, blocked, flushed, size))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    machine = torch.cuda.get_device_name() if args.device == "cuda" else (platform.processor() or platform.machine())
    print(
        f"{parameters / 1e6:.0f}M parameters + AdamW, fp32, {args.device} ({machine}, {os.cpu_count()} cores), "
        f"torch {torch.__version__}; a save every {args.every} steps, {args.steps // args.every} saves\n"
    )
    print("| saver | blocked per save (median) | flush at the end | written per save |")
    print("|---|---|---|---|")
    for saver, blocked, flushed, size in rows:
        median = sorted(blocked)[len(blocked) // 2]
        print(
            f"| {saver} | {median * 1000:,.0f} ms | {flushed * 1000:,.0f} ms | {size / 2**20:,.0f} MiB |"
            if saver == "moonclip"
            else f"| {saver} | {median * 1000:,.0f} ms | - | {size / 2**20:,.0f} MiB |"
        )


if __name__ == "__main__":
    main()
