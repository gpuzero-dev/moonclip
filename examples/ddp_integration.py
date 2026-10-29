"""
Example: Moonclip with PyTorch Distributed Data Parallel (DDP).

Spawns 2 processes that share one store: trains a toy model a step, saves
it as one snapshot with a shard per rank, trains on, and loads it back.

Usage:
    python examples/ddp_integration.py

Note: Uses mp.spawn (no torchrun needed). Works on all platforms.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from moonclip import CheckpointManager


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.net(x)


def run_worker(rank, world_size, storage_root):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    # A port already chosen wins, so two runs side by side (CI) do not collide.
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = ToyModel()
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # The topology is stated, not detected: since 0.0.9 a manager under a
    # multi-rank launcher refuses to guess whether the ranks share this store
    # or each keep their own. Here they share it, as rank 0 and rank 1 of 2.
    manager = CheckpointManager(
        storage_root=storage_root,
        compression_level=3,
        max_full_snapshots=3,
        world_size=world_size,
        rank=rank,
    )
    print(f"[Rank {rank}] world_size={manager.world_size}, rank={manager.rank}")

    # ─── Train one step ──────────────────────────────────────────────
    inputs = torch.randn(8, 128)
    loss = ddp_model(inputs).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    original_weights = {n: p.clone() for n, p in model.named_parameters()}
    print(f"[Rank {rank}] Step 1 done, loss={loss.item():.4f}")

    # ─── Save ────────────────────────────────────────────────────────
    snap_id = manager.save(
        step=1,
        model=ddp_model,
        optimizer=optimizer,
        metadata={"loss": f"{loss.item():.4f}"},
    )
    print(f"[Rank {rank}] Saved {snap_id[:8]}...")

    # ─── Mutate (simulate more training) ─────────────────────────────
    loss = ddp_model(inputs).sum()
    loss.backward()
    optimizer.step()

    for n, p in model.named_parameters():
        assert not torch.equal(p.data, original_weights[n]), "Weights should have changed"
    print(f"[Rank {rank}] Mutated weights in step 2")

    # ─── Resume ──────────────────────────────────────────────────────
    # The same manager that saved: every rank reads the snapshot rank 0
    # finalized. A restarted process would build a new manager the same way
    # and call the same method.
    manager.load_latest(model=ddp_model, optimizer=optimizer)
    print(f"[Rank {rank}] Resumed from checkpoint")

    # ─── Verify ──────────────────────────────────────────────────────
    for n, p in model.named_parameters():
        assert torch.equal(p.data, original_weights[n]), f"[Rank {rank}] Mismatch in {n}!"
    print(f"[Rank {rank}] ✓ All weights match!")

    dist.destroy_process_group()


def main():
    world_size = 2
    with tempfile.TemporaryDirectory() as storage_root:
        print(f"Storage: {storage_root}")
        mp.spawn( # type: ignore
            run_worker,
            args=(world_size, storage_root),
            nprocs=world_size,
            join=True,
        )
    print("✓ DDP integration test passed!")


if __name__ == "__main__":
    main()
