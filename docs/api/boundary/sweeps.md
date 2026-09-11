# `pepsy.boundary.sweeps`

`CompBdy(..., fit_mode="dmrg2")` fits the complete boundary interval with
`FIT.run_gate(block_size=2)`, using two-site warm-up sweeps followed by
one-site refinement. The legacy `fit_mode="two-site"` remains fixed
two-site FIT. The direct modes `"direct"`, `"src"`, `"zipup"`, `"sdc"`,
and `"dm"` use Quimb boundary compression without FIT. Configure FIT modes
with:

- `fit_max_bond`: required for rank growth beyond the current boundary bond;
  omission safely caps direct `CompBdy` use at the current bond.
- `fit_init_strategy="guess-direct"`, `"guess-src"`, or `"guess-sdc"`:
  initialize FIT from a disposable compression of the exact boundary target;
  `"direct"` is the default and `"auto"` is its compatibility alias.
- `fit_init_seed`: seed for the disposable SRC guess.
- `fit_sweep_sequence="RL"`: alternating sweep directions.
- `fit_cutoff` and `fit_cutoff_mode`: native SVD truncation policy;
  `"auto"` selects a dtype-aware cutoff and the standard `"rsum2"` mode.
- `fit_min_iter`, `fit_rtol`, and `fit_patience`: adaptive stopping policy.
- `fit_timing=True`: include elapsed and per-sweep/site timing records in each
  public `BoundaryFitDiagnostic`; add `fit_timing_sync_device=True` only when
  kernel-complete accelerator profiling is required.

The implementation builds the fixed environment once per sweep and updates
the moving environment after each pair. Thus it does not turn a linear cached
boundary sweep into a full environment rebuild at every bond.

`CompBdy.fit_diagnostics` is reset by each public run/move call. Convergence
metadata is always cheap and available; detailed timers are opt-in.

> API details are maintained as handwritten Markdown in this page.
