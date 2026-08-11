# `pepsy.fitting.local`

`FIT(target, p=guess, ...)` variationally fits an open-boundary MPS or MPO
guess to a target tensor network. There are three sweep entry points:

- `run()` is the simple full-contraction reference.
- `run_eff()` is the cached one-site full-chain solver used by the default
  boundary/sampling path.
- `run_gate()` is the cached active-window solver used by MPS/MPO circuit FIT
  and by `fit_mode="two-site"` boundary contraction over the full interval.

For circuit compression, set `range_int=(xmin, xmax)` and use:

```python
fit = pepsy.FIT(
    target,
    p=current,
    range_int=(xmin, xmax),
    cutoffs=1e-12,
    environment_strategy="auto",
    # Keep the safe default for caller-owned targets. MpsOptimizer transfers
    # its fresh disposable target with copy_target=False.
    copy_target=True,
)
fit.run_gate(
    n_iter=4,
    block_size=2,
    sweep_sequence="RL",
    max_bond=chi,
    cutoff=1e-12,
    cutoff_mode="rsum2",
    single_pair_fast_path=True,
    collect_split_diagnostics=False,
)
compressed = fit.p
```

`block_size=2` is recommended. Each update forms the two-site wavefunction
with two outer virtual legs and both physical groups, then calls Quimb
`Tensor.split` across the middle bond. This permits active rank growth and
dispatches natively for dense NumPy/Torch/CuPy and Symmray U1/U1xU1 fermionic
arrays. `block_size=1` retains the fixed-rank compatibility update.

For an interval containing exactly one neighboring pair,
`single_pair_fast_path=True` marks structural convergence after one update:
the effective tensor and its SVD solve the entire active problem, so another
sweep only rebuilds the same environments. The default is `False` on direct
`FIT.run_gate` calls to preserve fixed-sweep compatibility; MpsOptimizer opts
in by default. `collect_split_diagnostics=False` omits per-SVD truncation
dictionaries when only the fitted state and retained norm are needed.

`sweep_sequence` uses Quimb direction names: `"R"` is left-to-right, `"L"` is
right-to-left, and `"RL"` alternates. `environment_strategy="auto"` selects
`"mps-direct"` for an ordinary dense one-tensor-per-site target and otherwise
uses the general native-safe `"generic"` route. The two strategies implement
the same objective for targets supported by both; the explicit setting is
mainly useful for profiling and regression comparison.

With `timing=True`, `get_timing()` returns completed and failed partial sweep
records, including direction, block size, environment strategy, pair/site
times, and convergence status. Pair records break out effective contraction,
SVD, writeback/norm, and moving-environment time. Add
`timing_sync_device=True` for device-complete Torch CUDA, CuPy, or JAX timing;
normal runs never pay for these synchronization barriers. FIT also exposes
`final_center_site`, `final_norm`, `final_direction`, and
`convergence_reason`, allowing an optimizer to reuse the known canonical
center without a redundant sweep.

Those adaptive and detailed timing fields belong to `run_gate()`. The legacy
`run()` and `run_eff()` solvers remain fixed-sweep numerical paths and are not
silently changed by the gate-window controls. PEPS boundary results describe
them as `convergence_reason="fixed_sweeps"` and can collect one coarse elapsed
time per boundary fit without altering either solver's update sequence.
