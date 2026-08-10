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
value = pepsy.peps_norm(
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
)
```

`chi` is the default two-site SVD cap. Supply `fit_max_bond` to use a
different cap. With `fit_rtol=None`, exactly `n_iter` sweeps run. Reuse a
`BdyMPS` or `{"bdy": BdyMPS}` holder to retain fitted boundary states across
calls; pair updates retain the same fixed-plus-moving environment cache
strategy within each sweep. Newly created two-site boundaries start at bond 1
and grow locally rather than being padded globally to `chi`.

> API details are maintained as handwritten Markdown in this page.
