# MPS autodiff publication batch

The user explicitly authorized committing and pushing the completed MPS work.
This batch contains the adaptive JAX QR policy, Torch FX-compatible QR VJP,
traceable MPS norm and canonicalization handling, zero-cutoff gate compression,
backend conversion helpers, focused regressions, and their API/evidence notes.
The baseline is `6de7551` on `develop`.

Validation reuses the unchanged implementation's 295 focused Pepsy tests in
[the captured-VJP record](2026-10-08-mps-captured-vjp-workflow.md), plus the
downstream Gaugy MPS checks. Fresh `python -m ruff check src tests` and
`git diff --check` passed before staging. No new numerical changes were made
for publication; no full-suite or GPU success is claimed.

Earlier statements that this work was uncommitted describe their historical
checkpoints. This record accompanies the user-authorized publication batch;
Gaugy's dependent engine changes are published after the Pepsy support.
