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

For two- or three-site FIT on an active window spanning at least three sites,
`final_one_site_sweeps=1` adds a fixed-rank one-site polish pass after the
block sweeps. The pass reuses the canonical window and never touches sites
outside `range_int`; it is skipped for a two-site window. This is an explicit
direct-`FIT.run_gate` control. `MpsOptimizer` uses its separate adaptive
`fit_adaptive_sweeps`/rank-ceiling schedule and does not add this legacy polish
pass automatically.

For direct gate-window fits, `three_site_sweeps=1` (the default) uses one
larger three-site warm-up sweep and then switches to one-site refinement for
any remaining requested sweeps. Set `three_site_sweeps=2` for two directional
warm-up passes. Supplying `adaptive_block_sweeps=N` instead applies the same
minimum block warm-up to two- or three-site FIT. With
`adaptive_until_rank=True`, the block phase continues until all active bonds
reach their physical ceilings; rank stagnation is deliberately not an early
exit. Remaining requested sweeps use one-site FIT. One-site refinement
preserves the bond dimensions opened by the larger block and is cheaper than
repeating the larger SVD block.

The MPS optimizer passes `adaptive_block_sweeps=fit_adaptive_sweeps` and
`adaptive_until_rank=True` for its rank-growing `dmrg`, `dmrg1`, and `dmrg3`
paths. Before constructing a `dmrg1` fit, the optimizer checks the active
attainable bond ceilings: an already-capped window starts with one-site FIT,
while an under-capacity non-adjacent window requires `n_iter >= 3` for two
two-site growth sweeps and at least one one-site refinement sweep. `dmrg2`
uses the configured minimum block warm-up and then refines with one-site FIT.
The direct FIT diagnostics
`adaptive_sweeps_run` and `one_site_sweeps_run` count both scheduled block
sweeps and any explicit `final_one_site_sweeps` polish passes.

For tolerance-controlled `run_gate`, `patience` counts same-phase retained-norm
samples, not norm differences. Thus `patience=2` needs two comparable
one-site samples and stops after their first stable relative change, subject
to `min_iter`. A phase change from block growth to one-site refinement resets
the convergence window.

Ordinary dense arrays and native fermionic Symmray arrays reuse the compatible
partial overlap environments produced by the preceding opposite-direction
sweep. Fermionic FIT keeps the working state conjugated across the complete
sweep sequence, so the reused environments retain one dual-leg convention.
A block sweep retains only the boundaries needed by another reversed sweep of
the same size. If the next reversed sweep changes to one-site refinement, FIT
extends that cache through exactly one terminal tensor after a two-site sweep,
or two terminal tensors after a three-site sweep. Both 2-to-1 and 3-to-1
transitions therefore avoid a complete fixed-side rebuild without constructing
unused terminal environments during block warm-up. Fresh sweeps construct only
the fixed boundaries that their active block can query. Bosonic Symmray arrays
retain their conservative environment rebuild policy.

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
sweep only rebuilds the same environments. That terminal update constructs no
active-window environments; native fermionic outside-window environments stay
intact. The default is `False` on direct
`FIT.run_gate` calls to preserve fixed-sweep compatibility; MpsOptimizer opts
in by default. Consequently, named `dmrg1` and `dmrg2` windows of two sites
perform one two-site update and advance to the next gate without one-site
refinement; `n_iter` and tolerance controls cannot add a second sweep while
the fast path is enabled. `collect_split_diagnostics=False` omits per-SVD
truncation dictionaries when only the fitted state and retained norm are
needed.

`sweep_sequence` uses Quimb direction names: `"R"` is left-to-right, `"L"` is
right-to-left, and `"RL"` alternates. Native fermionic `run_gate` executes the
requested sequence exactly and records the conjugated fitting convention in
`info["fermionic_sweep_sequence"]`. It canonicalizes once around the first
sweep center, contracts the real outside overlap environments rather than
substituting graded boundary identities, applies Symmray's dual-leg phase
correction before each local writeback, and resolves odd dummy-mode global
phases afterward. The physical ket is restored on both success and failure.
The same convention supports block-2/3 native `run_eff` sweeps.

Before entering that conjugated gauge, a native full-chain MPS fit compares
the target and guess virtual charge maps. If the target has sectors absent
from the guess, FIT initializes from a target copy compressed natively with
the requested `max_bond`, `cutoff`, and `cutoff_mode`. This deterministic
target-informed initialization prevents an overlap environment from
projecting every missing sector to zero; it uses neither random noise nor a
dense representation. The decision is recorded under
`info["native_sector_initialization"]`. A partial-window fit cannot replace
its fixed outside boundary safely, so it leaves sector creation to its native
two-/three-site block updates. If the actual effective tensor is nevertheless
empty, FIT raises an explicit disconnected-sector error rather than an
internal Symmray decomposition failure.

`environment_strategy="auto"` selects
`"mps-direct"` for an ordinary dense one-tensor-per-site target,
`"symmray-native"` when all target and fitted tensors are Symmray-backed, and
otherwise uses the general `"generic"` route. Non-fermionic Symmray inputs
use the native blockwise chain product; fermionic Symmray inputs stay on the
resolved native strategy but use Quimb's graph-planned direct tensor
contraction so contraction order, dummy modes, and graded phases remain
authoritative. It dispatches directly on the Symmray arrays and does not build
a temporary TensorNetwork. Neither route densifies the tensor arrays. The
explicit settings are mainly useful for profiling and regression comparison.
For a layered target, FIT resolves each active-window boundary bond by
inspecting tensors on the two neighboring site tags and caches the resulting
index name. It does not rescan the complete target index map during local
environment updates; no tensor data or backend array is copied by this cache.

`cutoff="auto"` chooses `1e-3` for 16-bit data, `1e-6` for 32-bit/complex64
data, and `1e-12` for 64-bit data. Numeric cutoffs retain their explicit
behavior.

When consecutive sweeps reverse direction, `run_gate()` reuses the canonical
form produced by the preceding block update instead of repeating the boundary
canonicalization pass. The first sweep and consecutive same-direction sweeps
still prepare their required gauge explicitly.

With `timing=True`, `get_timing()` returns completed and failed partial sweep
records, including a `timing_schema` version, direction, block size, active
window size, update count, environment strategy, block/site times, and
convergence status. Timing schema 3 adds sweep-level
`canonicalization_seconds`, `sweep_preparation_canonicalization_seconds`,
`fixed_environment_seconds`, `moving_canonicalization_seconds`,
`moving_environment_seconds`, and `sweep_overhead_seconds`. Every update also
reports `effective_seconds`, `svd_seconds`, `writeback_seconds`,
`canonicalization_seconds`, and `moving_environment_seconds`; the legacy
`environment_seconds` remains the complete post-writeback phase, and one-site
updates report `svd_seconds=0.0`. The sweep record aggregates each stage as
well as `elapsed_seconds`, making one-, two-, and three-site runs directly
comparable in benchmark output. Block records break out effective contraction,
SVD, writeback/norm, canonicalization, and moving-environment time. Add
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
