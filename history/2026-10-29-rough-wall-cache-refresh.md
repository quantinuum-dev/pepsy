# 2026-10-29 — Complete saved roughening wall caches

- Scope: fill missing $D_{\rm raw}$ and $L_{\max}$ points in the sibling
  `rough_exact_.ipynb` and `rough_exact_mps_.ipynb` analysis notebooks.
- Baselines: Pepsy `develop` at `e658bed`; examples `main` at `58e3f8d`.
  Both checkouts already contained unrelated working-tree edits.
- Commit status: uncommitted; nothing staged, committed, or published.

The existing exact cache builder computed the two missing 5×6, $dt=0.4$
cases at offsets $5(0.533/6)$ and $0.533$, reusing 110 entries. All 112
configured exact cases now have means and standard errors from 8192 retained
$t=6$ Z shots. The existing MPS cache already covered all 35 finalized cases
at χ=32,64,128,256,512. The inspected χ=1024 sweep had no finalized cases.
No evolution was launched or modified.

Refreshed the cache-status output and both wall figures in
[rough_exact_](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/rough_exact_.ipynb).
The corresponding [MPS notebook](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/rough_exact_mps_.ipynb)
outputs matched the new execution exactly and needed no rewrite. All notebook
code and unrelated outputs were preserved. JSON caches remain Git-ignored
under `benchmark/store/analysis_cache/rough_exact_wall_metrics/` and
`rough_exact_mps_wall_metrics/`.

Validation: temporary notebook copies executed the loading and wall-plot
cells; assertions checked all seven finite points in every plotted series,
112 exact caches, and 35 MPS caches. All three figures were visually checked.
A second exact-builder run computed zero and reused all 112 entries.
Notebook schema validation and `git diff --check` passed. No numerical
implementation changed; no full numerical suite was run.
