# `pepsy.boundary.sweeps`

`CompBdy(..., fit_mode="two-site")` fits the complete boundary interval with
`FIT.run_gate(block_size=2)`. Configure it with:

- `fit_max_bond`: required for rank growth beyond the current boundary bond;
  omission safely caps direct `CompBdy` use at the current bond.
- `fit_sweep_sequence="RL"`: alternating sweep directions.
- `fit_cutoff` and `fit_cutoff_mode`: native SVD truncation policy.
- `fit_min_iter`, `fit_rtol`, and `fit_patience`: adaptive stopping policy.

The implementation builds the fixed environment once per sweep and updates
the moving environment after each pair. Thus it does not turn a linear cached
boundary sweep into a full environment rebuild at every bond.

> API details are maintained as handwritten Markdown in this page.
