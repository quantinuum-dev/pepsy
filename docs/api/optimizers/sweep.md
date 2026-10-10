# `pepsy.optimizers.sweep.optimizer`

`SweepOptimizer` supports two boundary-environment engines:

- `boundary_engine="dmrg"` uses Pepsy `BdyMPS` plus `CompBdy`.
- `boundary_engine="quimb-mps"` uses Quimb MPS environments and scalar
  `contract_boundary(...)` contractions. During a half-sweep it builds the
  opposite-side environments once, then advances the moving boundary one row
  or column at a time.
- `boundary_engine="auto"` keeps dense inputs on `dmrg` and routes
  Symmray-looking inputs to `quimb-mps`.

Torch-backed Symmray blocks use the Torch autograd local solver. NumPy-backed
Symmray blocks retain the finite-difference fallback.

## ALS within a slice

Dense inputs also support `optimizer="als"` in `set_optimize_kwargs`, or
`solver="als"` in `optimize_axis`. The outer boundary sweep is unchanged;
each local solve instead visits the individual site tensors forward and
backward at fixed bond dimensions.

```python
sweeper.set_optimize_kwargs(
    optimizer="als",
    optimizer_options={"n_round_trips": 2, "rtol": 0.},
    axes=("y", "x"),
    n_round_trips=1,
)
result = sweeper.run()
```

Inner `n_round_trips` defaults to two and is independent of the outer count.
Each inner round trip visits every tensor twice. `rtol=1e-9` stops on relative
infidelity improvement after a complete round trip; `0`/`None` disables it.
The default `linear_solver="dense-cg"` builds the active site's norm matrix
from cached strip environments, Hermitianizes it as `(N + N.H) / 2`, and
solves `N A = b` iteratively with Jacobi-preconditioned CG. It performs no
SVD, eigendecomposition, or direct factorization. Norm scoring uses
`Re(A.H N A) = A.H ((N + N.H)/2) A`, consistently with the local solve:
small imaginary residuals from truncated boundary environments do not reject
an otherwise valid strip. Nonfinite or nonpositive Hermitian norms still
reject it. The next site uses the
updated past and cached future; reversal rebuilds directional caches.
Nonfinite or unconverged solves are rejected. NumPy, Torch and CuPy stay
native; Symmray is rejected explicitly. Single-precision objective
contractions accumulate in double precision on the same device to reduce
cancellation; owned tensors and solver outputs keep their requested dtype.

Explicit `linear_solver="cg"` instead applies the Hermitian norm through
tensor contractions and uses Jacobi-preconditioned conjugate gradients, with
no full norm matrix or eigendecomposition. Environment halves are contracted
separately; a trial tensor is absorbed before joining them. Small cursor/scalar
kernels use optimal paths to avoid a hidden dense-metric contraction.

`cg_maxiter=200` bounds the iteration count. `cg_rtol=None` selects a
relative true-residual tolerance of 1e-8 in double or 1e-5 in single precision.
`cg_shift=0.` leaves the
Hermitian metric unshifted; a positive value adds that fraction of the largest
absolute diagonal entry to the diagonal. `None` selects 1e-10 in double or
1e-5 in single precision. Hermitianization alone does not ensure positivity.
Nonpositive curvature or failure to reach the residual tolerance rejects the
site update. The default `cg_fallback=False` never falls back to a direct factorization.

Explicit `linear_solver="dense"` uses a direct linear solve without
diagonalization. `dense_shift=0.` leaves its matrix unshifted; a positive
value adds relative diagonal regularization. Explicit `linear_solver="pinv"`
uses an eigendecomposition and positive-support pseudoinverse, with dtype-aware
cutoff `rcond=None`. `max_matrix_size=1024` limits matrix dimensions for all
assembled-matrix methods, including `dense-lbfgs`; `None` removes that limit.
It does not limit matrix-free solves. Explicit `cg_fallback=True` permits a
direct dense fallback only within this guard, restoring D^12 work.

For fixed physical dimension and target bond dimensions O(D), the default's
D^4-by-D^4 matrix costs O(chi^3 D^4 + chi^2 D^8) to construct. Hermitianization
costs O(D^8), and k dense CG iterations cost O(k D^8). Thus at chi=D^2 the
default local update is O(D^12 + k D^8); it is the construction, not the
iterative solve, that introduces D^12.

Matrix-free CG instead costs O(k D^10) at chi=D^2; a strip of length L with
r inner round trips costs O(r L k D^10), plus outer boundary construction.
Cached environment halves and contraction workspace can still require
O(D^8) memory. Direct factorization and spectral pseudoinverse alternatives
also have O(D^12) solve cost. These are
arithmetic counts, not a guarantee that CG is faster at small D. See the
[audit and timings](../../development/notes/2026-10-10-sweep-als-costs.md).

Local results identify `solver="als"`, completed `n_round_trips`, individual
`local_solves` (site, direction, solver, acceptance and rejection reason),
and directional `environment_reuse`. Histories retain the raw accepted local
infidelities. ALS uses strict dtype-roundoff validity checks and rejects
worsening overlaps, independently of the gradient solver's broader diagnostic
clipping allowance. These guarantees refer to the fixed approximate boundary
environment, not an exact whole-state fidelity.
Direct and CG records include the true relative residual; CG also records
iterations, while the explicit pseudoinverse records retained rank.
Failed iterative solves retain `cg_failure`, and an
explicitly enabled fallback records `fallback_from="cg"`. Its
`relative_residual` describes the final direct solution; when available,
`cg_relative_residual` retains the failed iterative solve's residual.

### L-BFGS for each site

Within `optimizer="als"`, select `linear_solver="dense-lbfgs"` to replace
the default CG solve with L-BFGS while retaining the explicit Hermitian N.
Select `linear_solver="lbfgs"` for the same one-site optimization using
matrix-free environment contractions. Both preserve the inner forward/backward
schedule, cached neighbors, exterior tensors, and overlap acceptance.
The default remains `"dense-cg"`.

The local objective is `Re(A.H N_H A) - 2 Re(A.H b)`, with analytic gradient
`2 (N_H A - b)` represented by separate real/imaginary coordinates. This
path does not use finite differences or require an autodiff graph, including
for NumPy/CuPy. SciPy controls L-BFGS on the host: site parameter and gradient
vectors are transferred for Torch/CuPy, with a warning; environment arrays,
contractions, dtype, and device stay unchanged. SciPy is imported lazily.

Controls are `lbfgs_maxiter=100`, `lbfgs_history=10`, `lbfgs_maxls=20`, and
`lbfgs_rtol=None` (true-residual target 1e-6 double / 1e-4 single). L-BFGS
returns its best finite evaluated quadratic candidate when its budget or
line search stops; it need not have converged. The original overlap still
decides whether that tensor is accepted. Site records distinguish
`converged`, `relative_residual`, `termination_reason`, `lbfgs_iterations`,
and `lbfgs_evaluations`. No fallback or spectral projection is implicit.
CG/direct-specific shift controls do not modify the L-BFGS objective.

At chi=D^2, q matrix-free function/gradient evaluations cost O(q D^10);
explicit-matrix L-BFGS costs O(D^12 + q D^8), including assembly. History
algebra is lower order for fixed history size. Line searches contribute to q;
there is no general speed guarantee over CG for this quadratic problem.

## Boundary and diagnostic controls

For dense DMRG environments, `fit_mode="two-site"` starts new boundaries at
bond 1 and lets native pair SVDs grow them to the requested `chi`. The
optimizer stores requested chi separately from the current warm-state rank,
so `normalize()`, `infidelity()`, and `set_target(...)` do not accidentally
turn bond 1 into the future accuracy cap. A tuple `chi=(chi_norm, chi_overlap)`
keeps the two caps independent during local sweeps.

The same adaptive schedule is available through `fit_mode="eff"`: set
`fit_block_size=2` or `3` and `fit_adaptive_sweeps=2` for two initial block-SVD
sweeps followed by one-site refinement. `fit_rtol`, `fit_min_iter`, and
`fit_patience` are optional; leaving `fit_rtol=None` keeps fixed-sweep behavior.
When enabled for the `eff` path, stopping waits for at least two completed
sweeps and compares consecutive retained-center norms. The default boundary
sequence is `RL`: left-to-right followed by right-to-left.

`SweepOptimizer` uses the same layer and diagnostic controls as the public
boundary functions. `fit_layer_mode="joint"` is the default and is required
for FIT/DMRG modes. Direct Quimb modes can use
`fit_layer_mode="sequential"` with an explicit `layer_tags` absorption order.
`fit_timing=True` enables FIT timing diagnostics, and
`fit_timing_sync_device=True` adds accelerator synchronization only when
timing is enabled. These options can be supplied at construction or through
`normalize_kwargs` / `infidelity(...)` per-call overrides.
With `renormalize_state=True`, constructor normalization uses
`normalize_kwargs["chi"]` when supplied, otherwise the environment `chi`.
An explicit `renormalize_kwargs["chi"]` takes precedence for that initial call.
The latest metric diagnostics are available as `optimizer.fit_diagnostics`,
and sweep runs also return them under the `fit_diagnostics` result key.

Fidelity diagnostics use Autoray `clip(real(value), 0, 1)` on their array
backend. The differentiable local objective remains unclipped to preserve
gradients. `evaluation_negative_tol=1e-3` allows approximate diagnostic losses
outside [0,1] by up to 0.001: a warning is emitted once per sweep run and their
values are clipped for display and continuation. The historical
option name covers both ends of the interval. The existing `1e-10`
roundoff allowance is a lower bound; zero restores that older allowance.
Raw local losses remain in `raw_loss_initial`, `raw_loss_final`, and solver
`history`; whole-sweep raw values remain in `loss_before` and `raw_loss_after`.
This is a continuation policy, not an accuracy bound. Negative values below
the ordinary `-1e-10` roundoff allowance cannot become winning zero-loss states
or trigger early convergence. A small negative initial estimate continues
through the requested sweeps instead of skipping refinement. Compare raw
losses while increasing normalization and
norm/overlap boundary caps, with exact small-system references where practical.
PepsOptimizer's compact `clipped_loss_records` preserve initial/final local
excursions on both sides of [0,1].

An initial infidelity estimate that is nonfinite or outside [0,1] beyond the allowance stops
cleanup without changing the warm start. The result reports `success=False`,
`converged=False`, and `termination_reason="invalid_initial_loss"`; the raw
estimate remains in `loss_before`, while `loss_after` and `best_loss` are
`None`. These safeguards do not increase boundary caps or FIT budgets.

Local solver output is checked before slice writeback. Nonfinite parameters,
nonfinite losses, or losses outside [0,1] beyond the allowance are rejected with a warning and
`invalid_loss=True`. The previous slice is retained. `loss_final` describes
that retained slice; `candidate_loss` and `rejection_reason` describe the
rejected result. Boundary accuracy remains the caller's responsibility.

Every local result records `update_applied`. Skipped initial objectives retain
`raw_loss_initial` and `rejection_reason="invalid_initial_loss"`; rejected
candidates also retain their raw loss. Global results count
`applied_local_updates` and `invalid_local_updates`. If every attempted update
was rejected, `success=False` and
`termination_reason="no_valid_local_updates"`; the PEPS driver retains its
compressed warm start and does not label that step optimized. Partial sweeps
report `partial_local_updates`, without claiming convergence. An explicitly
requested zero-cycle run remains a successful no-op.
Compact PEPS records retain `invalid_loss_records` with slice coordinates,
raw values, rejection reasons, and writeback status.

Sweep setup uses lazy boundary containers for direct compression, including
after restoring the best state. Boundary values are constructed when needed;
this does not reuse stale contraction values after a tensor update.

Local objectives cache contraction paths within each fixed slice environment
by ordered indices and shapes. Tensor values and gradients are recomputed.
`cache_contraction_paths=False` disables this cache. Explicit optimizer/tree
objects retain their original handling, including sliced contractions.
By default, row/column norm and overlap objectives use Pepsy's reusable
Cotengra `build_optimizer(progbar=False)` helper (`build_contraction` is its
compatibility alias), independently of the boundary `contraction_opt` preset.
The optimizer is created lazily and reused across slices; `PepsOptimizer`
also shares it across gate batches and successive `run()` calls. Its own
topology/shape search cache remains active independently of the per-slice
string-path cache. Pass `local_contraction_opt` to customize this local search
policy, including explicit optimizer/tree objects. Optional local cost
estimates use the same optimizer as the objective.
The default boundary/diagnostic `contraction_opt=None` also uses this helper
and shares it with local objectives. Boundary FIT inherits that optimizer
unless explicitly overridden, instead of falling back to a string preset.
Contraction-cost estimates are opt-in with constructor
`collect_contraction_metrics=True`; otherwise `flops`/`peak_*` are omitted.
Average boundary-MPS norm reports are also opt-in through
`set_optimize_kwargs(collect_boundary_norms=True)`. Their default `None` values
avoid extra per-slice contractions and unused boundary construction; this does
not disable physical normalization or the norm in the local objective.

An owning driver can use `set_optimize_kwargs(initial_loss=..., compute_final_loss=False)`
when it already measured the initial loss for these exact inputs/policy and
will perform its own final acceptance check. Without overrides, standalone
sweeps retain their initial/final checks. A skipped final check returns
`loss_after=None`; `best_loss` still describes the chosen sweep candidate.
Result fields `initial_loss_reused` and `final_loss_measured` distinguish the
two paths. Round-trip budgets are unchanged.

For objective-only fitting, set `compute_initial_loss=False` and
`compute_final_loss=False`. This skips both separate whole-state diagnostics;
`loss_before` and `loss_after` are `None`, and no initial diagnostic can trigger
early convergence. Local objectives and their safeguards remain active.
An explicitly supplied `initial_loss` still takes precedence, including its
early-exit check. Normalization is controlled independently by `renormalize`.

`SweepOptimizer.infidelity(...)` inherits constructor FIT controls when they
are omitted. Passing `fit_rtol=None` explicitly disables adaptive stopping for
that diagnostic; omitting `fit_rtol` inherits the constructor value. The same
omitted-versus-explicit-`None` rule applies to other FIT controls for which
`None` has an underlying solver meaning.
