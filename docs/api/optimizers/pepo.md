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
