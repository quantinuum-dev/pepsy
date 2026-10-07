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
values are clipped for convergence/best-state bookkeeping. The historical
option name covers both ends of the interval. The existing `1e-10`
roundoff allowance is a lower bound; zero restores that older allowance.
Raw local losses remain in `raw_loss_initial`, `raw_loss_final`, and solver
`history`; whole-sweep raw values remain in `loss_before` and `raw_loss_after`.
This is a continuation policy, not an accuracy bound: a negative approximate
loss clipped to zero can satisfy convergence bookkeeping without establishing
an accurate overlap. Compare raw losses while increasing normalization and
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

`SweepOptimizer.infidelity(...)` inherits constructor FIT controls when they
are omitted. Passing `fit_rtol=None` explicitly disables adaptive stopping for
that diagnostic; omitting `fit_rtol` inherits the constructor value. The same
omitted-versus-explicit-`None` rule applies to other FIT controls for which
`None` has an underlying solver meaning.
