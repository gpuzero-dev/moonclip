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

- **Restoring runs code from the checkpoint.** Part of a checkpoint
  (optimizer and scheduler state, the metadata of each state dict) is
  unpickled. Whoever can write to the store's directory or bucket can
  therefore run code in the process that restores from it. Give write access
  only to what you would let run code on your training machines.
- Bucket credentials are read from the environment, a profile in
  `~/.aws/credentials` or the machine's role, and are never written to the
  store or to a log. Keep keys out of command lines.
- Every tensor carries a checksum that is verified on read: that detects
  corruption, not tampering.
