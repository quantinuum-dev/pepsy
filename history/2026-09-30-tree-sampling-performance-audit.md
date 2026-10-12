# 2026-09-30 — TreeSampler performance and canonicalization audit

- Scope: review correctness and efficiency for a 5×6 system, chi=256,
  8,192 samples, chunk size 2,048, native CuPy; compare MPS reuse and center
  metadata handling.
- Branch / baseline: `develop`, `414798e`.
- Commit status: uncommitted audit records only; no source edits, commits,
  publication, or production-job changes. Unrelated PEPS working-tree edits
  were preserved.

Canonical-region reuse works: root-centered refresh performs no gauging,
off-root refresh touches only the center-to-root path, and sampling performs
no canonicalization. CuPy einsum already dispatches binary contractions to
matrix multiplication. Tree transfer environments are rebuilt per chunk,
unlike MPS cached right environments, but transfer application is the main
measured cost. Full quartic environment caching is not memory-safe at chi=256.

A synthetic 30-site, three-child-root tree with actual chi=256, complex128,
and the requested sample/chunk counts took 39.88 s on an RTX A5000;
32.03 s was first-child transfer application and 0.87 s its construction.
This is an instrumented single measurement, not the user's production state.
See [detailed evidence, limitations, and optimization candidates](../docs/development/notes/2026-09-30-tree-sampling-performance-audit.md).

Focused sampler suite: **67 passed, 1 skipped** (two GPUs required).
Independent large-tree probability scoring agreed to 2.17e-14 relative error.
Full-package validation was not run; no implementation changed. The measured
results support an algorithmic contraction bottleneck, not a demonstrated
Born-probability or canonical-center bug.

An isolated exact shared-density prototype passed 12 small NumPy/Torch
seeded-parity and dense Born checks, then reduced the same synthetic CuPy
case to 29.50 s (26% less elapsed time in a single comparison). Large-case
independent scoring agreed to 1.20e-14 relative error. It is not integrated
or validated as a complete replacement. An optional, much more expensive
binary-root benchmark was interrupted; no finished result is claimed.

Local documentation links and whitespace checks passed. The patch helper
failed on a later update with the environment's mountinfo sandbox error;
scoped exact replacements completed these audit-only updates after both
apply_patch entry points were attempted.
