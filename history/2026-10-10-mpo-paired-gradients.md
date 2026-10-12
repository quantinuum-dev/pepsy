# 2026-10-10 — Paired MPO derivatives for Gaugy

- Scope: fix the remaining MPO derivative failures reported to the user.
- Branch / baseline: `develop` at `b3e7985`; changes are working-tree edits
  at entry creation. Publication is recorded in the final addendum.
- Existing PEPS/ALS implementation, tests and documentation are separate work
  and are preserved. Only this task's changelog entry belongs to this batch.

## What changed

Dense direct MPO replay has an explicit paired-factor path, scoped chart-error
capture, canonical-center reuse, and a narrow bypass of Quimb's QR shortcut
that retains the actual compression cap. See the
[numerical record](../docs/development/notes/2026-10-10-mpo-paired-gradients.md)
for contracts, upstream audit and downstream evidence.

## Validation

- Initial focused projector/MPO/API/layout suite: 220 passed, 12.11 s.
- Final focused suite including scoped error capture: 221 passed, 12.26 s.
  Includes CPU and CUDA, untruncated dense derivatives and real cap-2
  truncation checks. These runs overlap; they are not additive.
- `python -m ruff check src tests` and `git diff --check` passed.
- Local Markdown links passed.
- Full development run, `JAX_PLATFORMS=cpu python -m pytest -q -o addopts=''`:
  **8,581 passed, 10 skipped, 2,118 warnings**, 2,060.61 s; exit status zero.
  BLAS/OpenMP thread counts were one. This run started during development
  before the final capture/metadata refinements and spans a changing workspace,
  including unrelated PEPS work. It is not a full-suite validation of an
  unchanged final commit. The focused checks above cover this task's final
  implementation. CUDA was available and exercised by the focused tests;
  JAX used CPU in the full run.

## Limits

The compatibility default is unchanged. Projector replay requires supported,
numerically resolved charts and rejects others. The downstream bounded exact
trace fixes the recorded untruncated cases; it does not silently substitute
an exact objective for a truncated one. No native-array algorithm changed.

## Publication

Implementation commit `3afa925` was pushed to `origin/develop`. The downstream
Gaugy change was committed and pushed as `3cfcbd3`. The existing PEPS/ALS
working-tree changes, including their separate changelog entry, were excluded.
