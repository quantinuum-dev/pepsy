# 2026-09-30 — Integrate shared-density tree sampling

- Scope: user authorized the exact shared-density optimization following the
  [sampler audit](2026-09-30-tree-sampling-performance-audit.md).
- Branch / baseline: `develop`, `414798e`.
- Commit status: working-tree edits only; nothing staged, committed, or
  published. Existing PEPS changes were preserved.

TreeSampler now shares incoming densities before physical conditioning
distinguishes shots, and contracts singleton transfers without constructing
the quartic environment. Both chunked and unchunked paths use the change.
Canonical-center handling, RNG ordering, backend/device ownership, and
refresh semantics are preserved. Updated API documentation, changelog, and
independent dense-reference/work-sharing/gradient tests.

[Implementation details and validation](../docs/development/notes/2026-09-30-tree-shared-density.md):
151 initial sampler/readout checks passed, with one two-GPU skip; 19 final
updated/new regression checks passed (18 overlap the first selection).
Ruff, local links, and whitespace checks passed. Full-package tests were not
run. Two new regressions fail against the saved baseline for the intended
repeated-work reasons.

On the same synthetic 30-site chi=256 CuPy complex128 tree, 8,192 samples in
chunks of 2,048 took 39.87 s before and 28.90 s after (one warmed comparison).
All configurations matched exactly and probabilities agreed to 4.62e-14
relative difference. Production-state performance remains unverified.

The required patch helper was attempted, but tracked-file edits failed with
the environment's mountinfo sandbox error; exact scoped replacements were
used to complete the authorized changes. No installed libraries or sibling
repositories were modified.
