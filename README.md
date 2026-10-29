# Moonclip

A checkpoint engine for ML training, written in Rust with Python bindings. It
compares each tensor with the previous checkpoint, skips the ones that did not
change, stores the others as a compressed XOR delta, and writes in the
background, so a save holds the training loop only for the copy of the
tensors.

On dense pre-training, the least favourable case because Adam changes every
parameter at every step, that measured 1.6-3.0× faster than `torch.save` and
1.9× smaller. Fine-tuning, LoRA and adapters leave most tensors unchanged
between checkpoints and should gain more; that has not been measured.
The design follows [DECK (Meta, PVLDB 2025)](https://doi.org/10.14778/3750601.3750621).

```python
import torch
from moonclip import CheckpointManager

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)

mgr = CheckpointManager("./checkpoints")
start_step = mgr.resume(model=model, optimizer=optimizer, scheduler=scheduler)
print(f"starting at step {start_step}")

for step in range(start_step, 5_000):
    x = torch.randn(32, 16)
    loss = torch.nn.functional.mse_loss(model(x), x.sum(dim=1, keepdim=True))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 500 == 0:
        mgr.save(step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                 metadata={"loss": f"{loss.item():.4f}"})
        print(f"step {step}  loss {loss.item():.4f}")

mgr.flush()
```

It runs as it is, on a CPU, in a few seconds. Stop it with Ctrl-C part way
and run it again: it starts after the last step it saved.

`resume()` loads the latest checkpoint and returns the step after it, or 0
when there is none. `save()` returns once the tensors are copied; hashing,
delta detection, compression and the write run on a background thread. Call
`flush()` when the checkpoint has to be on disk, which means `fsync`ed; loads
and `list_snapshots()` wait for pending saves on their own, and
`async_save=False` makes every save synchronous.

## Install

```bash
pip install moonclip
```

Wheels exist for Linux x86_64 (manylinux_2_28), CPython 3.11-3.14, because
that is where training runs. On Python 3.10 or older `pip` may install 0.0.8
or earlier instead of refusing, since those said they ran on 3.9: check
`python --version`, or ask for `"moonclip>=0.3"`, which turns that into an
error naming the Python version. On any other platform `pip` finds no wheel and
the package has to be built, which needs a [Rust toolchain](https://rustup.rs/):

```bash
pip install git+https://github.com/gpuzero-dev/moonclip.git
```

`import moonclip` does not import torch: the Rust engine stores named tensors
and knows nothing about PyTorch, and `CheckpointManager`, the layer that knows
what a `state_dict` is, is loaded the first time it is used.

## What a checkpoint holds

- **Deltas.** An unchanged tensor costs no I/O. A changed one is stored as an
  XOR delta against the previous checkpoint when that is smaller than storing
  it whole (`delta_max_ratio`), and a full snapshot is taken periodically
  (`full_every_steps`). A background thread merges delta chains so that a load
  does not have to replay a long one.
- **Compression.** zstd, with large tensors split into frames compressed on
  every core.
- **Precision.** `save_dtype="bf16"` (or `fp16`) casts floating-point tensors
  before compression and casts them back on load; a mapping applies a
  different dtype per name pattern. `fp8` quantizes to one byte per element
  against a per-tensor scale, which is for archived copies rather than
  checkpoints you resume from. Tensors that already are float8 are stored
  bit-exact.
- **Integrity.** Every tensor carries an xxHash3-128 checksum that is checked
  on read.
- **Retention.** `max_full_snapshots` and `max_total_snapshots` bound what is
  kept on disk.

## S3

The local directory stays the primary copy, so a save is as fast as the disk;
a bucket (S3, R2, MinIO or any S3-compatible server) receives it in batches in
the background:

```python
mgr = CheckpointManager(
    "./checkpoints",
    s3_bucket="my-bucket",
    s3_endpoint="https://<account>.r2.cloudflarestorage.com",  # omit for AWS
    s3_path_style=True,
    sync_every_n_saves=5,
)
```

**Credentials**, in this order, the first that is there:

1. `s3_access_key` and `s3_secret_key` (and `s3_session_token`) passed in.
2. `MOONCLIP_S3_ACCESS_KEY`, `MOONCLIP_S3_SECRET_KEY` and, for temporary
   credentials, `MOONCLIP_S3_SESSION_TOKEN`; then the `AWS_*` names every S3
   tool reads. Each pair is taken whole, never half from one set of names and
   half from the other.
3. A profile in the shared credentials file: `AWS_PROFILE`, else `default`, in
   `AWS_SHARED_CREDENTIALS_FILE`, else `~/.aws/credentials`. Its keys only: a
   profile in `~/.aws/config` that assumes a role or uses SSO is not followed.
4. The machine's role, with no key anywhere: a web identity
   (`AWS_WEB_IDENTITY_TOKEN_FILE` and `AWS_ROLE_ARN`, as EKS sets them), the
   container's credentials (ECS, EKS Pod Identity), or the EC2 instance
   profile through IMDSv2. These expire, and are fetched again before they
   do, so a run longer than their life keeps writing.
   `AWS_EC2_METADATA_DISABLED=true` skips the instance metadata service.

Keep keys out of the command line, where `ps` and the shell's history keep
them: the environment, a profile or a role. Objects are written without
per-request encryption headers, so the bucket's default encryption (SSE-S3 or
SSE-KMS) is what applies.

**Requests.** A sync uploads each file the bucket does not have yet, one
`PUT` each, and lists the prefix once. With Ravex's default of a delta at
every step, a sync every 100 steps of a small model is in the order of 200
`PUT`s: on AWS, about a tenth of a cent. `sync_every_n_saves` trades that
against how much a lost machine costs. The syncs say nothing unless
`MOONCLIP_LOG=debug`; failures are always printed.

`restore_from_remote()` brings a store with an empty local directory back from
the bucket.

**Trust.** Restoring a checkpoint unpickles part of it: optimizer and
scheduler state, and the metadata of each state dict. Anyone who can write to
the directory or the bucket can therefore run code in the process that
restores from it, as with `torch.load(..., weights_only=False)`. Give write
access to a checkpoint bucket only to what you would let run code on your
training machines, and do not restore checkpoints from a source you do not
trust.

`moonclip.Bucket` is the same client for plain files: whole objects by path
(`get`, `put`, `list`), with no manifest and no tensors, for whatever has to
sit next to the checkpoints.

## More than one rank

Say which topology the manager is for. Under `torchrun`, `"auto"` takes it
from the launcher:

```python
mgr = CheckpointManager("./checkpoints", world_size="auto", rank="auto")
```

The ranks then share one store, through the `create_snapshot` / `save_rank` /
`finalize_snapshot` flow. If each rank writes to a directory of its own
instead, every one of them is single-rank and says so:

```python
mgr = CheckpointManager(f"./checkpoints/rank_{rank}", world_size=1, rank=0)
```

Leaving the topology unsaid under a multi-rank launcher is an error that names
both numbers. Adopting the launcher's variables silently made the same code
save under `python train.py` and refuse under `torchrun`, decided by nothing
the code could see. A single process needs no configuration.

## Environment

| | |
|---|---|
| `MOONCLIP_THREADS` | Size of Moonclip's own thread pool. Default: the cores divided by `LOCAL_WORLD_SIZE`, so that the ranks on one node share the machine instead of each claiming all of it. |
| `MOONCLIP_PROFILE=1` | A per-phase breakdown of each save on stderr, and a line whenever a save waited for the previous one. |
| `MOONCLIP_FSYNC=0` | Skips the `fsync` before a pack is renamed into place. Faster, but a power loss mid-save can leave a pack of the right length full of zeros that the manifest vouches for. |
| `MOONCLIP_NO_PINNED_STAGING=1` | Copies GPU tensors through ordinary memory instead of pinned buffers, for a box where host RAM is the constraint. |
| `MOONCLIP_LOG=debug` | Also logs the syncs that went well, and the snapshots a restore left out because a sync had not carried them whole yet; errors are logged either way. |

## Moonclip and Ravex

Moonclip is the storage layer under [Ravex](https://github.com/gpuzero-dev/ravex),
and the line between them is that saving files is Moonclip's and the run is
Ravex's. Moonclip decides how a checkpoint becomes files and where they go,
bucket keys included. Ravex decides when to save and what, puts the result
back into a live training process, and coordinates one or more machines; it
carries no storage code of its own. To Moonclip a rank is only a number on a
snapshot. Moonclip works on its own as well, as in the examples above.

## Development

```bash
cargo test                                      # Rust tests, no Python needed
maturin develop --release && pytest tests/ -v   # Python and PyTorch tests
cargo clippy --all-targets -- -D warnings       # what CI gates on
```

`rust-toolchain.toml` pins the same compiler as CI, so that a red clippy here
means a red gate there. It is not the MSRV, which is `rust-version` in
`Cargo.toml`. The crate is deliberately not `cargo fmt`-clean: the gate is
clippy. What changed between versions is in the [CHANGELOG](CHANGELOG.md), and
runnable examples (a plain model, a transformer, DDP, S3) are in `examples/`.

## License

Apache-2.0 — [GPU Zero](https://gpuzero.dev)
