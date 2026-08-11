# `pepsy.boundary.metrics`

`peps_norm`, `peps_normalize`, `peps_infidelity`, and `contract_flat` accept
three DMRG/FIT boundary modes:

- `fit_mode="eff"`: cached one-site sweeps and the compatibility default.
- `fit_mode="two-site"`: cached pair updates followed by native SVD splits.
- `fit_mode="global"`: the full-contraction reference fit.

Selectors are normalized early: `"two_site"` is accepted as an alias for
`"two-site"`, and `"one-site"` is a descriptive alias for the historical
`"eff"` spelling. Unknown values fail before boundary work starts.

Example:

```python
result = pepsy.peps_norm(
    state,
    chi=64,
    method="dmrg",
    fit_mode="two-site",
    fit_sweep_sequence="RL",
    cutoff=1e-12,
    fit_cutoff_mode="rsum2",
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
