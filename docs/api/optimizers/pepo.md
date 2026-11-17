# PEPO density-operator gate replay

`PepoOptimizer` evolves a native two-dimensional PEPO through a stream of
one-site or adjacent two-site gates. Each `(G, where)` means the complete
density-operator action $\rho \mapsto G\rho G^\dagger$; the input PEPO is
copied. Calling `run()` again replays the same stream on the current output.
The first implementation accepts dense arrays; Symmray PEPOs are rejected.

```python
from pepsy.optimizers import PepoOptimizer

opt = PepoOptimizer(rho, [(G, ((0, 0), (0, 1)))], chi=8, mode="svd")
rho_next = opt.run()
```

- `mode="svd"` uses a local split with no persistent environment gauges.
- `mode="simple-update"` carries simple-update bond gauges within the run,
  then absorbs them into the returned physical PEPO. It preserves scale rather
  than normalizing each gate. Gauges are rebuilt for the next `run()`.
- `mode="full-update"` uses the existing two-site `PepsOptimizer` full update,
  with paired upper/lower actions forming one target before compression.
  `boundary_chi` caps the boundary MPS and `fit_mode` selects `"direct"` or
  `"dmrg"`; `boundary_kwargs` and `full_update_kwargs` configure that solver.
  This path currently requires non-differentiable dense complex Torch or CuPy
  tensors, as does the underlying two-site full update. Its objective is
  operator overlap, not exact preservation of reduced density matrices.
- `mode="dmt"` is a **prototype** using exact or boundary-MPS single-layer
  contractions to construct regional maps. It currently requires dense NumPy
  `complex128` tensors, open boundaries and SciPy.

`chi` caps bonds being updated, not unrelated pre-existing bonds. `cutoff`
and `cutoff_mode` default to the existing dtype-aware `"auto"` policy.
`get_step_records()` exposes full-update or DMT records for the latest run;
the local SVD and simple-update modes currently return an empty list. Boundary
caches are reused by the underlying full-update solver within a run, not across
separate runs.

## Experimental regional DMT

```python
opt = PepoOptimizer(rho, gates, chi=4, mode="dmt", regions="L",
                    dmt_tolerance=1e-8, boundary_chi="auto",
                    fit_mode="direct")
rho_next = opt.run()
reports = opt.get_step_records()
```

`regions="L"` selects all three-site L shapes in every orientation. Supply a
sequence of coordinate sequences instead to protect explicit regions. At each
two-site gate, the solver forms the exact gated pair $M$, seeks a rank-$\chi$
pair $A$, and requires $\mathcal T_W(A-M)=0$ to the requested tolerance for
every region $W$. It uses single-layer regional maps, an SVD seed,
nonlinear least-squares feasibility restoration and a guarded local objective
refinement. The problem is nonconvex: it does not certify the global optimum,
and failure to find a feasible pair does not prove that none exists. On
failure, the run raises and retains the input state.

`dmt_refine=False` skips the optional local-objective refinement after a
feasible pair is found; hard regional residual checks remain enabled. This can
avoid a costly dense constrained solve, but may increase the local pair error.
If the feasibility solver stops with a residual above tolerance and has
evaluation budget left, it retries with stricter stopping thresholds. Records
include the retry flag and the number of feasibility evaluations used.

`dmt_refine_method="als"` selects experimental equality-constrained
alternating least squares instead of the default `"slsqp"` refinement. Each
half-sweep uses an SVD to choose an orthonormal factor space and solves a
constrained least-squares problem for the other factor. Only finite updates
that retain all regional constraints and lower the local pair error are
accepted; `dmt_refine_sweeps` caps the number of sweeps (default 4). Records
include the feasible seed error, final pair error and accepted ALS updates.
Neither refinement proves a global optimum, and lowering pair error does not
necessarily reduce error in the full PEPO or a later trajectory. See the
[4×5 comparison](../../development/notes/2026-11-17-pepo-als-investigation.md).

### Weighted regional objective

Keep the `regions` constraints hard and optionally replace the pair error by
an absolute weighted sum of other regional errors:

```python
opt = PepoOptimizer(
    rho, gates, chi=4, mode="dmt", regions="L", boundary_chi=8,
    dmt_refine_method="als",
    dmt_objective_regions=[
        (((0, 0), (0, 1)), 1.0),
        (((1, 1), (1, 2), (2, 1), (2, 2)), 2.0),
    ],
)
```

Each item is `(region, weight)` with a finite nonnegative weight. A mapping
from coordinate tuples to weights is also accepted. Duplicate weights sum;
zero weights are ignored. The objective is the **sum**, with no normalization
by region size or target norm. Protected and weighted regions share maps when
identical. All maps are frozen single-layer contractions in the chosen
environment, and the map budget includes both sets of regions.

The result is a feasible-seed local search; it does not guarantee the global
minimum. ALS uses a constrained nullspace least-squares update and SLSQP uses
the weighted objective with hard constraints. Unmeasured directions can be
poorly controlled, so accepted weighted updates default to a pair-error guard
`dmt_objective_max_pair_growth=2.0`: the final gauge-dependent pair error
must not exceed twice the seed error. Set a different factor (at least one),
or `None` to disable this safeguard. The guard changes the explored feasible
set and does not make the pair error a physical global error.

Records report `seed_objective`, `objective_value`, and `objective_exponent`.
The first two use the PEPO mantissa; multiply by `10**(2*objective_exponent)`
to obtain the corresponding absolute squared cost when representable. See
the [weighted 4×5 investigation](../../development/notes/2026-11-17-pepo-weighted-dmt.md)
for an independent comparison and its limitations.

Reports contain the maximum scaled regional residual, local pair error, and
whether objective refinement improved the feasible seed. Each residual is
scaled by the larger of the target reduced-operator norm and a small map-based
floor. The pair error is gauge-dependent and is not a global PEPO error.
`boundary_chi=None` uses exact contractions. A positive `boundary_chi` uses
Quimb single-layer boundary MPS: `fit_mode="direct"` uses SVD compression;
`fit_mode="dmrg"` uses Quimb's variational 1D FIT. `boundary_kwargs` accepts
`cutoff` (default zero) and `n_iter` for DMRG (default 10). The maps are built
once and held fixed during each pair solve. `boundary_chi="auto"` chooses
`max(4, 2 * chi)` as an initial cap, not an accuracy guarantee.

`dmt_cache=True` reuses regional maps across updates of the same active pair
within a run, while invalidating maps if an updated site lies outside their
active pair. `dmt_cache=False` rebuilds every map. The bounded LRU cache holds
at most `dmt_cache_max_elements` complex entries (default one million); maps
larger than the limit are not retained. Step records expose cumulative hits,
misses, evictions, invalidations and retained size under `map_cache`. A new
`run()` starts with an empty cache, so external changes to `state` between runs
cannot reuse stale maps. Identical regions are deduplicated. This caches whole
maps for repeated pairs; it does not yet share numerical boundary strips
between distinct pairs or regions.
The direct path calls Quimb's `contract_boundary` with its `direct` 1D
compressor, the same compression method selected by Pepsy's boundary fitter.
The feasibility and objective-refinement solvers use analytic real Jacobians
of the complex pair factors; nonfinite targets and maps are rejected.

With finite boundary compression, strict residuals refer to those **approximate
maps**. They are not a certificate of the exact PEPO marginals: independently
check with a larger boundary dimension or exact small-system contractions.
The bounding rectangle containing the active pair and selected region is
left uncompressed, so widely separated regions can limit the speedup.
`dmt_max_nfev` limits solver work, and `dmt_max_map_elements` limits the total
number of explicitly materialized map entries and active open-leg tensors.
This remains a prototype for modest sizes, not a matrix-free large-lattice
algorithm. DMT uses an exact gate target and does not currently apply `cutoff`
to its protected split.

None of the modes guarantees positivity or Hermiticity after a lossy split.
The non-DMT modes also do not guarantee preservation of trace or local densities.
