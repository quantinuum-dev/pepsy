# `pepsy.fitting.local`

`FIT(target, p=guess, ...)` variationally fits an open-boundary MPS or MPO
guess to a target tensor network. There are three sweep entry points:

- `run()` is the simple full-contraction reference.
- `run_eff()` is the cached full-chain solver used by the default
  boundary/sampling path; it defaults to one-site updates and also supports
  opt-in native two- and three-site updates.
- `run_gate()` is the cached active-window solver used by MPS/MPO circuit FIT
  and by `fit_mode="two-site"` boundary contraction over the full interval.

Choose the entry point by the scope of the fit:

| Entry point | Scope | Typical consumer |
| --- | --- | --- |
| `run()` | Full chain, simple reference contractions | Debugging and compatibility checks |
| `run_eff()` | Full chain, cached one-, two-, or three-site environments | Boundary and sampling workflows |
| `run_gate()` | `range_int` only, one-, two-, or three-site updates | MPS/MPO circuit compression |

The distinction is deliberate: `run_eff()` must not replace `run_gate()` for a
local gate target, because refitting sites outside the active interval changes
the circuit-compression algorithm. The FIT implementation follows the same
high-level order in each path: own and validate the target/state, prepare
effective environments, update the requested sites, then record optional
fidelity or timing diagnostics.

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

`block_size=2` is recommended for the usual DMRG compression. Each update
forms the two-site wavefunction with two outer virtual legs and both physical
groups, then calls Quimb `Tensor.split` across the middle bond. This permits
active rank growth and dispatches natively for dense NumPy/Torch/CuPy and
Symmray U1/U1xU1 fermionic arrays. `block_size=3` forms a three-site
wavefunction and performs two direction-aware native SVD splits, while
`block_size=1` retains the fixed-rank compatibility update. Three-site FIT is
useful when a larger local window is worth the extra SVD cost; it is not a
dense `from_dense` conversion.

The same native block updates are available for the full-chain path:

```python
fit.run_eff(
    n_iter=4,
    block_size=3,
    sweep_sequence="RL",
    max_bond=chi,
    cutoff=1e-12,
)
```

`run_eff(block_size=1)` remains the default compatibility path. The block-2
and block-3 variants visit the complete chain, reuse cached environments, and
grow only bonds reached by their native SVD splits. They always perform the
requested fixed sweep sequence; adaptive `rtol` stopping and detailed timing
remain controls of `run_gate()`.

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

When consecutive sweeps reverse direction, `run_gate()` reuses the canonical
form produced by the preceding block update instead of repeating the boundary
canonicalization pass. The first sweep and consecutive same-direction sweeps
still prepare their required gauge explicitly.

With `timing=True`, `get_timing()` returns completed and failed partial sweep
records, including a `timing_schema` version, direction, block size, active
window size, update count, environment strategy, block/site times, and
convergence status. Every update has the same four stage fields:
`effective_seconds`, `svd_seconds`, `writeback_seconds`, and
`environment_seconds`; one-site updates report `svd_seconds=0.0`. The sweep
record aggregates each stage as well as `elapsed_seconds`, making one-, two-,
and three-site runs directly comparable in benchmark output. Block records
break out effective contraction, SVD, writeback/norm, and moving-environment
time. Add
`timing_sync_device=True` for device-complete Torch CUDA, CuPy, or JAX timing;
normal runs never pay for these synchronization barriers. FIT also exposes
`final_center_site`, `final_norm`, `final_direction`, and
`convergence_reason`, allowing an optimizer to reuse the known canonical
center without a redundant sweep.

Those adaptive stopping and detailed timing fields belong to `run_gate()`. The
`run()` and `run_eff()` solvers remain fixed-sweep numerical paths and are not
silently changed by the gate-window controls. PEPS boundary results describe
them as `convergence_reason="fixed_sweeps"` and can collect one coarse elapsed
time per boundary fit without altering either solver's update sequence.
