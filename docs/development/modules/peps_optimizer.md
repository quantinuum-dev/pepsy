# pepsy.optimizers.peps

This package owns the PEPS/PEPO gate-stream optimizer.

`PepsOptimizer` is the public entry point. It applies one-site gates directly,
builds exact two-site targets, compresses warm starts to the requested PEPS
bond dimension, and optionally refines those warm starts with either
`SweepOptimizer` or `GlobalOptimizer`.

## Boundary controls

Keep the chi controls separated by job:

- `chi`: virtual bond cap for the optimized PEPS/PEPO.
- `boundary_chi`: sweep/global optimizer environment bond dimension.
- `normalize_chi`: standalone normalization bond dimension.
- `evaluation_chi`: diagnostic infidelity bond dimension.

`boundary_engine` selects the sweep cleanup boundary implementation:

- `"auto"`: dense inputs use Pepsy `BdyMPS`/`CompBdy`; Symmray-looking inputs
  use Quimb MPS boundaries.
- `"dmrg"`: force the Pepsy boundary path.
- `"quimb-mps"`: force Quimb MPS boundaries and default scalar metric
  contractions to `method="mps"`.

`boundary_options` is forwarded to the Quimb MPS boundary store when sweep
cleanup uses that engine. It follows Quimb's environment API: `cutoff` may be
a number or `"auto"`, and `cutoff_mode` is passed to the underlying boundary
SVD through `compress_opts` (`"auto"` resolves to Quimb's `"rsum2"`).

Dense `boundary_engine="dmrg"` runs can opt into Quimb compression, one-site
FIT, fixed two-site FIT, or mixed two-site/one-site FIT boundary updates
through `boundary_kwargs`:

```python
optimizer = PepsOptimizer(
    state,
    gates,
    chi=peps_chi,
    boundary_chi=boundary_chi,
    boundary_kwargs={
        "fit_mode": "dmrg2",
        "fit_init_strategy": "guess-src",
        "fit_init_seed": 7,
        "fit_sweep_sequence": "LR",
        "fit_rtol": 1e-8,
        "fit_min_iter": 2,
        "fit_patience": 2,
        "cutoff": 1e-12,
    },
)
```

The boundary `chi` is used as the native two-site SVD cap unless
`fit_max_bond` is supplied. `fit_mode="dmrg"` aliases the one-site/effective
`"eff"` path; `"dmrg2"` runs two-site warm-up followed by one-site
refinement, while `"two-site"` remains fixed two-site FIT. The Quimb modes
`"direct"`, `"src"`, `"zipup"`, `"sdc"`, and `"dm"` bypass FIT.
`fit_init_strategy` only initializes FIT from a disposable copy
(`"guess-direct"`, `"guess-src"`, or `"guess-sdc"`) and leaves the exact
target and reusable boundary handles unchanged.

## Extraction map

- `optimizer.py`: `PepsOptimizer` and the current orchestration logic.
- `gates.py`: target location for gate routing helpers.
- `warmstart.py`: target location for warm-start construction.
- `routing.py`: target location for sweep/global backend selection.
- `diagnostics.py`: target location for fidelity and progress records.
