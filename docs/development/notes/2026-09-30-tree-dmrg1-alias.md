# 2026-09-30 — Tree DMRG1 becomes the one-site DMRG alias

The user requested identical TreeOptimizer `dmrg`/`dmrg1` behavior, a warning
to use `dmrg`, no notebook CPU thread cap, and package defaults in roughening.

## Implemented behavior

`dmrg1` retains its requested-name metadata but now uses generic DMRG's block
size and adaptive policy. The implicit two-node growth phase and special
minimum-three-iteration requirement were removed. Both names default to
one-site refinement of the initialized guess. Explicit FIT controls, including
larger block sizes, behave identically. `dmrg2`/`dmrg3` retain their schedules.
TreePepsOptimizer and standalone TreeFIT retain their separate behavior.

Selecting the legacy name at construction or explicit replay emits a
`FutureWarning` recommending `dmrg`. Copying or replaying without a mode
override does not repeat it. The notebook uses `threads=None`; saved outputs
are preserved. Roughening applies TreeOptimizer's defaults to both names:
block size 1, 4 iterations, patience 1, automatic initialization/cutoff/rtol,
and unchanged ambient CPU threads. Explicit CLI overrides remain authoritative.

## Compatibility and validation

The installed Quimb, Autoray, Cotengra and Symmray versions are unchanged from
the [one-site-default audit](2026-09-30-tree-dmrg-one-site-default.md#compatibility-audit).
Rechecked `TreeFIT.run_gate` and its block/adaptive arguments. Reused that
active-task upstream audit; no dependency, tensor kernel, or dispatch changes.
Classification: **adopt** the existing generic-DMRG schedule for the deprecated
name; **defer** unrelated numerical/backend work.

- New NumPy/Torch CPU comparisons verify equal states, traces, iteration
  counts, convergence reasons, and canonical centers for constructor/run/copy
  entry points, one/four iterations, and one/two-node explicit block settings.
- Full tree-domain plus public-API/layout selection: **1533 passed,
  101 skipped, 1 failed**. Only the known installed-distribution metadata
  mismatch fails (installed 0.4.0 versus project 0.5.0).
- Full examples benchmark suite: **522 passed, 6 skipped, 4 failed**. Failures
  are the previously reproduced bubble `fit`/`dmrg`/removed-MPS-`dmrg1`
  compatibility cases. Final defaults/override/sweep checks: **16 passed**.
- Package Ruff, changed examples Ruff, notebook schema/code validation, and
  whitespace checks pass. Notebook outputs match the pre-edit snapshot.
- Validation used the shared Python 3.12 environment, local package source,
  CUDA hidden, and one OMP/OpenBLAS thread. No new GPU validation or production
  runs. Logs: `/tmp/tree-dmrg-alias-suite.log`,
  `/tmp/roughening-tree-alias-suite.log`, and
  `/tmp/roughening-tree-alias-final-controls.log`.
