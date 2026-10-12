# 2026-10-10 — One-site ALS within PEPS row/column sweeps

- Scope: add ALS as an alternative to joint strip L-BFGS, with several inner
  forward/backward passes per active column, retaining the outer sweep.
- Branch / baseline: `develop`, `b3e7985`; clean initial Pepsy working tree.
- Commit status: working-tree changes only; nothing staged, committed or pushed.

## Implemented

`PepsOptimizer(mode="sweep", optimizer="als")` and
`SweepOptimizer.optimize_axis(solver="als")` select fixed-rank one-site
positive-support solves. `optimizer_options["n_round_trips"]` defaults to two;
each trip visits all strip tensors forward then backward. Outer sweep budgets,
boundary caps, target construction and gradient-solver defaults are unchanged.

The new [local solver](../src/pepsy/optimizers/sweep/_als.py) consumes existing
norm/overlap boundary stores. It reuses the established `StripSweepCursor`
and native `solve_positive` kernel, carries network exponents explicitly,
and checks normalized overlap before accepting each site. Only active tensor
data change. Dense NumPy/Torch/CuPy remain native; unsupported Symmray/JAX
inputs fail explicitly. The default virtual matrix dimension guard is 1024.
Driver summaries retain ALS counts without keeping extra tensor networks.

See the [public options](../docs/api/optimizers/peps.md#one-site-als-inside-each-row-or-column)
and [standalone sweep guide](../docs/api/optimizers/sweep.md#als-within-a-slice).

## Validation

- New [ALS suite](../tests/test_peps_sweep_als.py): **38 passed**, including
  real/complex single/double precision, NumPy/Torch CPU, Torch CUDA/CuPy,
  both axes and boundary engines, independent dense least-squares and
  statevector references, rank-deficient metrics, nonunit target norms,
  large/opposite network exponents, locality/metadata, rollback and guards.
  The driver is exercised with fixed and adaptive boundary policies.
- Broader PEPS/sweep/refinement/API regression selection: **441 passed**,
  54 warnings, 101.48 seconds. It covered the new suite, sweep safeguards and
  performance, boundary-engine numerics, PEPS driver/batching, strip/layer
  refinement, public API, package layout and module extraction. The final
  38-test run above also checked the added compact driver diagnostics.
- Ruff `src tests` and `git diff --check` passed; the full repository suite
  was not run. Existing Quimb unspecified split-policy warning remains.

## Upstream check

Installed: Quimb `1.15.1.dev90+g6a3906cbe`, Autoray `0.11.1.dev14+g014a3f69a`,
Cotengra `0.8.3.dev8+g8954240f2`, Cotengrust `0.2.1`,
Symmray `0.4.1.dev15+g0374aaa3c`, NumPy `2.5.2`, Torch `2.6.0+cu124`.
Inspected current `tensor_network_fit_als` and `PepsOptimizer` signatures.
Reviewed [Quimb release notes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[release notes](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray source](https://github.com/jcmgray/symmray).
The requested Symmray abelian-array documentation page was unavailable.
Classification: adopt existing public contraction APIs and native Pepsy
positive-support kernel; defer symmetry support. No upstream patches or
dependency changes were needed.

## Limits

Acceptance is against the fixed approximate boundary environment, not an
exact whole-state reference. Dense local eigensolves are not matrix-free;
interior virtual matrix dimensions scale as D^4. No new 4×4 long-time
trajectory, performance ranking against L-BFGS, or symmetry support is claimed.
The earlier notebook ALS experiment remains a separate exact-contraction
benchmark; its saved trajectories were not changed or rerun.
