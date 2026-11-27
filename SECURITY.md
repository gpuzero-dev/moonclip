# Security

## Reporting a vulnerability

Please do not open a public issue. Report it privately, either through
GitHub's [private vulnerability reporting](https://github.com/gpuzero-dev/moonclip/security/advisories/new)
or by email to vecchionejonathan@gmail.com.

Say what is affected, how to reproduce it, and what an attacker gains. You
will get an answer within a week, a fix or a plan for one, and credit in the
release notes if you want it.

## Supported versions

Only the latest release receives fixes. Moonclip is 0.x, and a fix
is released as a new version, never backported.

## What to know before you rely on it

- **Restoring builds plain data only (GPU-538).** The non-tensor part of a
  checkpoint (optimizer and scheduler scalars, the structure of each state
  dict, RNG state) is written with `pickle`, and read back by an unpickler
  that builds only the primitives pickle encodes directly, `OrderedDict`,
  `set`, `frozenset`, `complex`, torch's `Size`, `device` and dtypes, and
  NumPy arrays and dtypes (for NumPy's RNG state). Anything else fails the
  restore with `moonclip.pytorch.UnsafeCheckpoint`, naming the entry and what
  it held; there is no unrestricted fallback. A `<prefix>._blob` entry, the
  whole-state-dict pickle the first versions wrote, is refused the same way.
  Write access to the store still decides what a resume trains on, so give
  it only to what you trust with your runs.
- Bucket credentials are read from the environment, a profile in
  `~/.aws/credentials` or the machine's role, and are never written to the
  store or to a log. Keep keys out of command lines.
- Every tensor carries a checksum that is verified on read: that detects
  corruption, not tampering.
