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

`SweepOptimizer.infidelity(...)` inherits constructor FIT controls when they
are omitted. Passing `fit_rtol=None` explicitly disables adaptive stopping for
that diagnostic; omitting `fit_rtol` inherits the constructor value. The same
omitted-versus-explicit-`None` rule applies to other FIT controls for which
`None` has an underlying solver meaning.


> API details are maintained as handwritten Markdown in this page.
