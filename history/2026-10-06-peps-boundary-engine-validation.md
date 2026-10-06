# 2026-10-06 — Real Quimb MPS / DMRG / DMRG2 validation

- Scope: test and report on all three PEPS sweep boundary configurations.
- Branch / baseline: `develop`, `8f7c896`.
- Commit status: new uncommitted regression tests and review notes only;
  pre-existing working-tree changes preserved. Nothing staged or published.

Added `tests/test_peps_boundary_engine_numerics.py`: real 3x3 entangled
two-RZZ optimization with the default NLopt maxeval=50 and four round trips
per axis, plus exact-reference 4x4 norms/fidelities in both directions.
All nine new cases passed in separate optimization/metric selections;
40 existing relevant tests also passed. Ruff and diff checks passed.

Additional probes exercised six complete optimization runs, eighteen
boundary-cap/direction metric cases, and 42 local objective/gradient checks.
All engines converged to dense-reference metrics at sufficient boundary cap.
Quimb needed a larger cap than FIT for comparable accuracy in the tested
4x4 fixture; do not infer a universal ranking from this observation.

Detailed fixtures, measured errors, timings, and limitations are in the
[validation report](../docs/development/notes/2026-10-06-peps-boundary-engine-validation.md).
Previously reported sweep failure-handling issues remain open. No production
fixes, full suite, GPU validation, or broad complex64 comparison in this task.
