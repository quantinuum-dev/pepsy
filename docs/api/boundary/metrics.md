# `pepsy.boundary.metrics`

`peps_norm`, `peps_normalize`, `peps_infidelity`, and `contract_flat` accept
the following boundary-compression modes:

- `fit_mode="direct"`, `"src"`, `"zipup"`, `"sdc"`, or `"dm"`:
  direct Quimb boundary compression.
- `fit_mode="eff"` (also `"dmrg"`): cached one-site FIT sweeps and the
  compatibility default.
- `fit_mode="two-site"`: fixed two-site FIT updates.
- `fit_mode="dmrg2"`: two-site FIT warm-up followed by one-site refinement,
  in the same spirit as a two-site-to-one-site MPS optimizer.
- `fit_mode="global"`: the full-contraction reference fit.

Direct Quimb modes also accept `fit_layer_mode="joint"` (the default) or
`fit_layer_mode="sequential"`. Joint mode compresses all tagged layers in a
boundary slice together. Sequential mode compresses them one at a time in the
order given by `layer_tags`, for example
`layer_tags=("BRA", "PEPO", "KET")`. Sequential mode is available only for
the direct Quimb modes; FIT/DMRG modes always use a joint target.

The layer policy applies to the package `method="dmrg"` path. Quimb's native
`method="mps"` path already handles `layer_tags` itself and rejects
`fit_layer_mode="sequential"` rather than silently ignoring the option.

Selectors are normalized early: `"two_site"` is accepted as an alias for
`"two-site"`, `"one-site"` aliases `"eff"`, and `"dmrg"` aliases the
historical `"eff"` spelling. Unknown values fail before boundary work starts.

`fit_init_strategy="direct"` (or the compatibility alias `"auto"`) preserves
the existing boundary guess. `"guess-direct"`, `"guess-src"`, and
`"guess-sdc"` create disposable Quimb-compressed guesses from each exact
boundary target before FIT; they do not replace the target or mutate reusable
boundaries. These strategies are initialization choices only: the selected
`fit_mode` still controls the final compressor. Native Symmray boundaries
fall back to `direct` with a warning for dense Quimb guesses.
The direct Quimb `fit_mode` values are currently dense-only; use
`fit_mode="dmrg"` or `"dmrg2"` for native Symmray boundaries.

For `fit_mode="eff"`, set `fit_block_size=2` or `3` to use native block-SVD
growth through `FIT.run_eff`. Add `fit_adaptive_sweeps=2` to perform two
block sweeps followed by one-site refinement; omit it to keep fixed block
sweeps. `fit_rtol`, `fit_min_iter`, and `fit_patience` remain optional
convergence controls. When `fit_rtol` is enabled, stopping begins only after
two completed sweeps. The default `fit_block_size=1` path also honors the
alternating `RL` sequence.

Example:

```python
result = pepsy.peps_norm(
    state,
    chi=64,
    method="dmrg",
    fit_mode="dmrg2",
    fit_layer_mode="joint",
    fit_init_strategy="guess-src",
    fit_init_seed=7,
    fit_sweep_sequence="RL",
    cutoff="auto",
    fit_cutoff_mode="auto",
    n_iter=8,
    fit_min_iter=2,
    fit_rtol=1e-8,
    fit_patience=2,
    fit_timing=True,
    return_info=True,
)

print(result.cost)
for fit in result.fit_diagnostics:
    print(fit.boundary_key, fit.iterations, fit.convergence_reason)
```

`chi` is the default two-site SVD cap. Supply `fit_max_bond` to use a
different cap. With `fit_rtol=None`, exactly `n_iter` sweeps run. Reuse a
`BdyMPS` or `{"bdy": BdyMPS}` holder to retain fitted boundary states across
calls; pair updates retain the same fixed-plus-moving environment cache
strategy within each sweep. Newly created two-site boundaries start at bond 1
and grow locally rather than being padded globally to `chi`.

For example, an adaptive full-chain FIT boundary can be requested with:

```python
result = pepsy.peps_norm(
    state,
    chi=64,
    fit_mode="eff",
    fit_block_size=2,
    fit_adaptive_sweeps=2,
    fit_sweep_sequence="RL",
    n_iter=6,
    return_info=True,
)
```

`contract_boundary(...)` always returns `BoundaryContractResult`. The scalar
helpers `peps_norm(...)`, `boundary_norm(...)`, and `contract_flat(...)` keep
returning a scalar by default; pass `return_info=True` to receive the same
structured result. Its `fit_diagnostics` tuple has one `BoundaryFitDiagnostic`
per attempted boundary fit, with the boundary key, actual iteration count,
convergence reason, relative change, final center/direction, and reached bond
dimension. These fields are collected without per-site timing overhead.

Set `fit_timing=True` to additionally populate each diagnostic's
`elapsed_seconds` and detailed two-site `sweep_timings`. On asynchronous
accelerators, `fit_timing_sync_device=True` adds device barriers so those
timings include completed kernels.

`peps_infidelity(...)` always returns its norm and overlap contraction results
under `norm_result`, `norm_target_result`, and `overlap_result` (a supplied
known norm has a corresponding `None`). `peps_fidelity(...)` remains scalar by
default; use `return_info=True` to receive that same dictionary plus the
computed `fidelity`. This makes `fit_timing=True` useful on both helpers.

> API details are maintained as handwritten Markdown in this page.
