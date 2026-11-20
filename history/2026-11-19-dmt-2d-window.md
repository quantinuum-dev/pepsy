# 2026-11-19 — 6×6 real-Pauli interval DMT prototype

- Scope: improve mapped 2D DMT performance using existing Pepsy/Quimb
  machinery, test 6×6 and multiple chi values; finish after user interruption.
- Branch / baseline: `develop`, `04a712435391712b47ad42777bdb4829d9dd7519`.
- Status: uncommitted. Previous 4×4 working-tree work is preserved. No commit,
  push, package upgrade, installed-library edit or public numerical default
  change was performed for this task.

## Implemented

The shared private workspace gains owned metadata copying and interval
transfer-MPO application, exact local QR preparation and affected-bond DMT
splits. Gauge transport retains valid center metadata after a failed QR.
Ordinary MPO replay uses the shared copy helper. The public gate APIs and
their supported locality remain unchanged.

[The research example](../examples/dmt_2d_window.py) constructs a 36-site
rank-two initial mixture and exact real rank-four ZZ transfer MPOs without
global statevectors/densities. It compares full/interval sweeps, gate order,
optimized seeds with zero local-ALS sweeps, float64/complex128 and a Torch
control. Checkpoints, raw local spectra and trace failures remain visible.

New interval compression performs fresh scalar target/output trace checks,
independent of cached protected maps; `warn` continues and records finite
failures. Full panel scans remain optional. Added exact small-system,
actual-loss, complex, Torch, canonical-gauge and failure-policy regressions.

## Evidence and limits

[Scientific record, plot and machine-readable artifacts](../docs/development/notes/2026-11-19-dmt-2d-window.md):
16 retained 6×6 cases, 45 readouts, 66 completed timed steps, plus four
20-step exact-reference 3×3 cases and separate audits/profiling. Comparable
pilot medians: full χ96 14.74 s, interval χ96 8.82 s, optimized seeds χ96
7.68 s, complex interval χ96 24.92 s, interval χ128 19.56 s. Apple
Accelerate thread limits were unset in these pilots; `threadpoolctl` did
not control that BLAS. Later VECLIB-limited timings are a separate group.

The bounded χ96 continuation reaches t=1.9 (19 steps, 250.07 s update
time, budget checked between steps), trace defect 7.45e-11. χ96/128 RDM
difference at t=0.5 is 0.03042: no convergence certificate. The ZZ-only
finite-neighborhood oracle gives χ96 error 0.08927 at t=0.5. Small 3×3
full/interval/seed local errors at t=2 are 0.10790/0.10863/0.10868;
global minimum eigenvalues are negative. Real Pauli cores enforce
Hermiticity, not positivity.

Actual lossy 6×6 product-mixture compression preserves all 25 plaquettes
within 3.32e-15; a long-gate current-target audit after t=1.9 is within
1.17e-14. Native QR/SVD take 79% of a 13.97 s profiled timestep; the
new scalar trace checks take 0.165 s (1.2%).

**Unresolved:** χ192 produces large trace/local-density defects, broken
canonical isometries and native process faults. Full sweeps, batched Quimb
decompositions, Torch float64, thread limiting and alternate region selection
do not establish a fix. No particular upstream library is proven responsible.
The new guard catches finite failures, not process-level native faults.
Keep χ192 out of accuracy references. Temporary reproducer scripts and
captured arrays remain under `/tmp/dmt6*`; durable summary excludes large
core dumps. This prototype is not promoted to a public nonlocal optimizer.

## Validation

The final regression pass has **301 passing tests**, including all 22 new
interval tests and the interrupted-gauge-move regression, with eight existing
Torch rank-policy/Quimb warnings. Ruff, the independent mathematical probe,
skill catalog, local Markdown links and whitespace checks pass. All scheduled
benchmark/validation jobs have ended; native-fault cases retain their last
completed checkpoints. No full-suite, GPU, native loopy PEPO, or general
constrained global variational solve is claimed. Earlier unrelated baseline
metadata/backend gradient failures were not relabeled as passing.
