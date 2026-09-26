# 2026-09-26 — First roughening PEPS engine

- Scope: implement the first PEPS path from the agreed overall workflow;
  user-selected dt and selected observation times remain separate.
- Branch / baseline: Pepsy `develop` / `80f451a`; examples `main` / `301159a`.
- Commit status: all changes are working-tree edits, unstaged and uncommitted.
  Nothing published. Existing unrelated edits and production jobs preserved.

## Implemented

- New example-side `magnetization/engines/peps.py`, selected by the existing
  roughening frontend with `--mode peps`. Reuses physical lattice mapping,
  wall product vectors, and ordered Strang gates. Uses public `ps_to_peps`
  geometry, initial SU gauges, and `gate_simple` with renorm=False.
- Independent PEPS D (`--chi`, frontend default 4), boundary chi, and evolution/
  boundary cutoff controls. NumPy/Torch support, including tested CUDA.
- Private gauge-absorbed snapshots for normalized local Z and imbalance.
  Selected global norm-squared contractions and BP/loop-cluster corrections
  default to in-range t=5,6,7,8,9,10. Nonconverged BP is reported and suppresses
  the fixed-point correction. No contraction-value cache crosses snapshots.
- Save targets split steps; gates are rebuilt for actual intervals. Nominal
  dt/depth determine final time, and actual steps/times are saved. Per-step
  manifest progress and atomic per-observation/aggregate output are supported.
- Existing single/batch/failure lifecycle dispatches PEPS. The angle-sweep
  launcher understands final physical time and refuses changed PEPS controls
  when reusing outputs. Default PEPS sweep paths have a distinct suffix.
- Updated runner/API-output documentation, experiment indexes, and nested
  AGENTS guidance. No Pepsy package numerical source changed.

## Validation

- Full benchmark suite: **393 passed, 65 warnings**, 88.00 s;
  `/tmp/peps-benchmark-full.log`. This includes the new nine PEPS test cases
  and existing MPS/tree/exact/effective/storage behavior. The warning count
  matches the preceding storage task; no full Pepsy suite is claimed.
- Existing Pepsy simple-update domain suite: **10 passed**.
- New tests compare independent dense evolution, wall orientation/mapping,
  2x2 and 2x3 lattices, local Z, imbalance, norm rescaling, truncation, gauge/
  tensor isolation and continued evolution, nonconverged BP status, actual
  save intervals, NumPy/Torch CPU subprocess outputs, and two-angle sweeps.
- Separate CUDA 2x2 probes preserved dtype/device: complex128 state error
  6.28e-16; complex64 error 5.43e-7. Boundary and full-plaquette loop norm
  squared matched dense contraction. Complex64 used BP tolerance 1e-5.
- Ruff passed for Pepsy `src tests` and the benchmark implementation/tests.
  Documentation local links and both repositories' `git diff --check` passed.
- Initial development checks exposed an inherited default shot-retention
  setting and Path serialization; both were fixed. A test-only completion
  helper typo was corrected before the passing focused/full checks.

The final small argument-validation, progress-reporting, and sweep-control
refinements were additionally checked with the affected PEPS, sweep, launcher,
and entrypoint tests: **44 passed**, 46.90 s.

## Limitations and evidence

- First implementation: open lattices with both dimensions >=2; contracted
  local Z/imbalance only. No shot sampling, entropy, energy, higher moments,
  PBC, serialized-state resumption, or PEPS disk cache. Explicit sampling/
  entropy requests are rejected; metadata reports zero samples and no entropy.
- No production PEPS run launched. Larger-lattice accuracy, boundary/loop
  convergence, and performance remain unmeasured. User-chosen save targets
  change step subdivision; comparisons must use actual recorded intervals.
- [Runner controls/schema](../../pepsy_examples/experiments/mps_magnetization/benchmark/magnetization/README.md#peps-simple-update)
- [Upstream audit and numerical policy](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/development/notes/peps_simple_update.md)
- [Earlier design investigation](2026-09-26-peps-simple-update-design.md)
