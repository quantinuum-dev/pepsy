# 2026-11-17 — Optional random ALS restarts

- Scope: user-requested SVD-plus-random ALS trials for loop-cluster updates
  and `PepsOptimizer` two-site full update.
- Branch / baseline: `develop`, `328979f`.
- Commit status: implementation, tests, documentation and the
  [earlier ALS audit](2026-11-17-eli-fu-als-audit.md) are included in the local
  feature commit requested after final review. No publication requested.

## Implementation

`solve_reduced_als(..., random_restarts=1, restart_seed=7)` fits the usual
SVD/supplied start plus one random factor pair. Zero is the default and
retains the original solver path. Each trial shares the problem, environment,
rank, solver controls and iteration cap. Actual finite weighted residuals
select the result; ties retain the first trial. Random draws reuse the
existing backend random-array helper and match each initial factor's norm.
The seed is local to each fit; global random streams are unchanged.

The shared `_als_restarts.py` helper validates controls, generates random
starts and attaches scalar per-trial metadata under
`solution.solver_report['restarts']`. Solution fields describe the winner;
the metadata records all costs, available convergence and total iterations.
Loop-cluster trackers retain a detached scalar snapshot as `als_restarts`.

`PreparedLoopClusterUpdate` exposes both controls, including in `.info`.
Sequential, fixed-batch and automatic-batch paths use them. Every restart
keeps the batched solve kernels; winners are selected independently per
member, without rebuilding numerical environments or changing gate commits.

`PepsOptimizer(..., full_update_kwargs={'random_restarts': 1,
'restart_seed': 7})` passes controls into reduced full-update ALS and records
`optimizer_result['als_restarts']`. Exact-SVD shortcuts still bypass fitting.
Existing SU/supplied starts remain trial zero, and the outer SVD fallback
remains active. Full-tensor and L-BFGS modes reject this ALS-only option;
native Symmray and autodiff reduced routes reject it before dense conversion.

Updated the BP/PEPS API guides, constructor documentation, changelog and
cluster status ledger. Numerical code and inputs frozen into existing Eli
campaigns were not modified or relaunched.

## Dependency audit

Installed environment: Python 3.12, Quimb 1.15.0, Autoray 0.8.4,
Cotengra 0.8.4.dev8+g646239731, Symmray 0.5.1.dev2+g20035beb7,
Torch 2.14.1+cu126. Inspected the installed public Quimb ALS signature and
dispatch, the official Quimb changelog, Autoray repository, Cotengra docs
and changelog, and Symmray repository. The Symmray array-documentation page
was unavailable through browsing; installed source/repository remains the
compatibility reference. No dependency or installed-library changes.

Classification: **adopt** existing native ALS and random-array helpers;
reuse the existing legacy Autoray backend-string namespace and unhashable
CuPy-device **compatibility shims** in mixed-backend test setup, initializing
the device shim before the string-key shim. This environment has an older Autoray
than the declared >=0.9 dependency. The shim is capability-gated and is not
a new numerical route. Test contractions use Cotengra's explicit implementation
as in the production campaign.

## Validation

New regressions check an actual random-trial win against independently
solved candidates, native dtype/device, RNG/input preservation, deterministic
seeds, supplied starts, invalid options, exact dense PEPS/PEPO fidelity,
ordinary/prepared replay equivalence, per-member batch selection, tracker
snapshot isolation and optional compiled Torch kernels. Batch gate/gauge
gradients with restarts are compared against sequential replay and finite
differences under both native and stabilized linear algebra.

- Focused numerical selection: **568 passed**, 43 pre-existing warnings,
  81.95 s. It covers `test_als_restarts`, BP reduced/Torch ALS,
  prepared/joint/batched cluster replay, PEPS/PEPO full updates, paired and
  scheduling reuse, pair L-BFGS and CuPy full update.
- Final restart file after adding explicit CuPy cases:
  `pytest -q -o addopts='' tests/test_als_restarts.py` → **29 passed**, 6.88 s.
  CPU, Torch CUDA and CuPy complex64/complex128 paths execute on this machine.
- Batch restart gradient selection: **28 passed, 36 deselected**, 5.10 s,
  before the final extra compiled/CuPy cases; those gradients also pass in
  the broader selection above.
- Public API/layout checks: **55 passed**, two existing warnings, 6.13 s.
- `ruff check src tests`, diff whitespace and changed documentation's local
  links passed. The full repository suite was not run.

The first broader run stopped at a legacy CuPy namespace-cache failure
before ALS (364 passed). An isolated `git archive` of unchanged baseline
`328979f` under `/tmp/pepsy-als-baseline-20261117` also failed the same
accelerator test in backend dispatch. Completing the existing compatibility
setup in the new test fixture allowed the full focused selection to pass;
no unrelated package or installed-library fixes were introduced.

Final pre-commit review reran the numerical selection together with public
API/layout checks: **627 passed**, 45 existing warnings, 87.93 s. Ruff,
whitespace and documentation links passed again. Explicit zero/one/two-restart
edge checks passed; clarified that unmonitored iteration totals can be `None`.
