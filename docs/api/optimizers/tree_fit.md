# Tree variational fitting

[Tree API overview](tree.md)

Use `TreeFIT` and the DMRG replay modes to fit the exact layered operator/state target.

Start with the [operator application guide](tree_replay.md) for supported
payloads and the direct, density-matrix, SRC, SDC, and zipup alternatives.

## Tree-native FIT / DMRG

`pepsy.fitting.TreeFIT` is the cached local variational fitting kernel for a
`TreeTensorNetwork`. It has the same separation of target, disposable initial
guess, bounded local updates, ownership controls, and diagnostics as
`pepsy.FIT`, but replaces the chain's left/right environments with one cached
directed overlap message for each tree edge:

```python
from pepsy.fitting import TreeFIT

fit = TreeFIT(target, guess, max_bond=32, cutoffs=1e-12)
fit.run_gate(
    active_nodes,
    n_iter=4,
    block_size=2,       # 1, 2, or 3 connected tree nodes
    sweep_sequence="inward-outward",
)
updated = fit.p
report = fit.fit_diagnostics(overlap=True)
```

Before each local solve, the kernel moves the orthogonality centre along the
unique tree path. Only messages whose branch intersects a changed local block
or centre path are invalidated, so untouched branches retain their cached
entanglement environments. Each missing directed message contracts only its
node's target tensors, fitted bra tensor, and incoming neighbor messages.
An iterative postorder traversal avoids recursive calls on deep trees.
Invalidation follows those dependencies outward; there is no table of full
branch-node sets. Temporary messages carry indices and data without
accumulating branch-wide tags. Numerical messages belong to one TreeFIT
instance and are reused across its sweeps; each TreeOptimizer gate creates
a new fit for its new target. Initial exterior contractions can visit the
whole state even when the operator itself is compact. Subsequent local
updates reuse unaffected messages. The cache holds at most one tensor per
directed tree edge, with tensor sizes determined by the live bond dimensions.
For dense compact circuit targets, an unchanged canonical exterior component
can instead be relabelled to its boundary identity without contracting the
dangling branch; `environment_cache_info()["identity_shortcuts"]` reports
these uses. Native Symmray and fermionic fits retain the graded message
contraction.

Standalone callers who directly edit `fit.p` tensor data must invalidate its
canonical metadata and call `fit.clear_environment_cache()` before resuming.
Raw external edits are not automatically tracked by the FIT message cache.
Construct a new TreeFIT when changing target geometry or connectivity.

`dmrg`, `dmrg1`, `dmrg2`, and `dmrg3` select this
engine in `TreeOptimizer`; `TreePepsOptimizer` accepts the same names. Generic
`dmrg` uses `fit_block_size` (two by default) and its configured adaptive
warm-up. `dmrg1` and `dmrg2` use two-node warm-up blocks, while `dmrg3` uses
three-node warm-up blocks followed by a two-node transition and one-node
refinement. The default `fit_n_iter=4` permits eight directional passes and
reaches refinement: `(2, 2, 1, 1)` for `dmrg2`, `(3, 3, 2, 1)` for `dmrg3`.
Tree warm-up still counts complete iterations, so this is not an identical
sequence of local updates to eight MPS sweeps. Explicit smaller budgets
remain honored. `fit_two_site_transition_sweeps=1` controls the `dmrg3`
transition within that budget; zero restores its previous three-to-one schedule.
Every block-size change resets the tolerance history, and tolerance stopping
cannot skip a pending transition or refinement phase.

`TreeOptimizer(mode="mix")` is a separate preset of the same TreeFIT engine:
apply and directly compress the operator on a private, chi-capped state,
then refine that guess against the separate original layered target using
one-node blocks from the first iteration. It fixes the effective
`fit_init_strategy="guess-direct"`, `fit_block_size=1`, and direct local
compression; it has no larger-block growth warm-up. Stored generic FIT
options are retained for later `run(mode="dmrg")` calls. Conflicting explicit
`compression_mode` values raise. The usual traversal, iteration, and tolerance
controls still apply, and exact one-node regions keep their fast path.

```python
optimizer = TreeOptimizer(gates, state=ttn, chi=64, mode="mix")
```

Mix supports dense and supported even-parity native fermionic trees; odd-parity
TreeFIT restrictions remain unchanged. A failed fit propagates its exception
without committing the temporary guess; unlike MPS mix, this tree preset
does not silently fall back to a direct result. Mix is a TreeOptimizer replay
mode, not a new standalone TreeFIT or TreePepsOptimizer mode.

`fit_sweep_sequence="inward-outward"` is the TreeOptimizer default. Use
`"outward-inward"` to reverse the order. Each iteration includes both passes,
ordered relative to the active region's medial node for branched regions and
explicit depth policies. Auto paths use the endpoint convention described
below. `RL`/`INOUT` and `LR`/`OUTIN` remain compatible aliases
for the two orders. Standalone TreeFIT uses the same names and default;
diagnostics report the normalized `sweep_sequence`.

Local gates in `dmrg`, `dmrg1`, `dmrg2`, `dmrg3`, and `mix` build a compact
`SubTreeMPO`; explicit `apply_sub_mpotree(subtree_operator)` uses the same
FIT route. The exact target has operator tensors only inside that region,
with original site labels and physical input/output indices preserved.

FIT reuses each traversal order within a run. Before a block update, it moves
the canonical center only as far as needed to make the exterior isometric;
an existing center or canonical region contained inside the block needs no
preparatory QR. An unknown gauge is canonicalized toward the block without
collapsing its interior first. Local
factorization still establishes the requested final center, including explicit
endpoint centers for three-node `TreeFIT.fit_block` updates.

`fit_traversal="auto"` is the TreeOptimizer default for `dmrg`, `dmrg1`,
`dmrg2`, `dmrg3`, and `mix`. It inspects the induced active region: paths use
consecutive one-, two-, or three-node windows between endpoints, and branched
regions use depth-first updates around their medial hub. Both visit the same
connected block sets as the explicit `"depth"` and `"depth-first"` policies.
Those explicit policies retain their previous FIT block ordering. Standalone
TreeFIT supports `traversal="auto"` but retains its own `"depth"` default.

These are user-selectable traversal policies, separate from `mode`:

```python
optimizer = TreeOptimizer(
    gates, n=8, mode="dmrg2", fit_traversal="depth-first",
)
optimizer.apply_sub_mpotree(tree_operator)
# fit_traversal="depth" selects ordering by distance from the active hub.
```

| `fit_traversal` | Path-shaped support | Branched multi-site support |
| --- | --- | --- |
| `"auto"` (default) | Endpoint sweeps, including every two-site geodesic | Depth-first sweeps |
| `"depth"` | Explicit distance-from-hub order | Explicit distance-from-hub order |
| `"depth-first"` | Explicit branch-first order | Finish each branch before the next |

The same policies apply to `dmrg1`, `dmrg2`, `dmrg3`, and `mix`, whether the input
is an ordinary gate or an explicit multi-site `sub_mpotree`. A few-body
operator whose physical sites lie on one path still benefits from the
automatic path route. Direct/SRC/SDC/SDCR/zipup keep their own compression sweeps;
`fit_traversal` controls FIT and does not change those algorithms.

Auto freezes a reference path before constructing the guess, beginning at
the endpoint nearest the incoming canonical center (node-id ties; node-id
order if the center is unknown). `inward-outward` visits this path then its
reverse; `outward-inward` reverses the pass order. Each block is factored with
its center at the endpoint in the direction of travel. Reversal changes both
window order and final centers, so adjacent multi-node windows contain the
previous center. The order stays fixed through the 3→2→1 transitions.

Automatic guesses remain SRC for dense trees and direct for native fermionic
trees. Compressed guesses finish at the actual first FIT endpoint, including
when the pass order is reversed. SRC/SDC/SDCR retain the original layered target
and their per-call environment caches. This changes seeded approximations
from the previous hub order while remaining reproducible with the same
state, seed, and options. Finite-bond fidelity and convergence can change;
extra exterior legs can still make multi-node factorization expensive.
Diagnostics report requested `traversal`, `resolved_traversal` (`"path"` or
`"depth-first"` for auto), `path_endpoints`, and the actual final center.

For native Symmray states, `fit_environment_strategy="native-blockwise"`
uses graded blockwise contractions for FIT messages and effective tensors,
avoiding repeated charge-block fusion/unfusion. It retains Quimb/Cotengra
contraction planning, backend/device, and fermionic phases. The default is
`"default"`; the alternative requires native target/state arrays and checks
actual upstream API support. No global dispatch is changed. Standalone
TreeFIT calls this option `environment_strategy`. Performance depends on the
sector sizes and backend; blockwise is not universally faster.

`fit_single_node_fast_path=True` is automatic for a truly one-node active
region. It skips the compressed/random guess and performs one local
projection, preserving the parent RNG sequence. Diagnostics report one
iteration, `block_size_trace=(1,)`, `guess_used=False`, and
`convergence_reason="single_node_exact"`. This remains exact for a local
nonunitary gate and records its resulting norm; it does not depend on
`fit_rtol` or `fit_min_iter`. Set the flag to `False` to restore repeated
sweeps. A multi-node region using one-node updates still needs iteration.
For standalone `TreeFIT.run_gate`, `single_node_fast_path=True` solves the
local least-squares problem with its exterior held fixed; it does not promise
global fidelity one for an arbitrary target. Complete-tree `run`/`run_eff`
retain fixed iteration defaults with `single_node_fast_path=False`.

Small CPU measurements and compatibility probes are recorded in the
[FIT execution review](../../development/notes/tree_fit_execution.md).

The optimizer options `fit_n_iter`, `fit_adaptive_sweeps`,
`fit_two_site_transition_sweeps`, `fit_min_iter`, `fit_rtol`,
`fit_patience`, `fit_sweep_sequence`, `fit_init_strategy`,
`fit_init_rand_strength`, and `fit_init_seed` are forwarded to TreeFIT.
Outside the fixed `mix` preset, `fit_init_strategy="auto"` is the default:
it selects `guess-src` for dense
trees and `guess-direct` for native fermionic trees, whose randomized SRC
compression is unsupported. This preserves the previous dense numerical
policy. Explicit unsupported native `guess-src`/`guess-dm` requests still
raise; `auto` does not change the requested output compression method.
`fit_init_strategy="direct"` keeps the current state as the initial guess;
`"guess-direct"`, `"guess-src"`, `"guess-sdc"`, `"guess-zipup"`, and `"guess-dm"` use a disposable compressed
warm start; `"random"` perturbs only active tensors and `"random_expand"`
also grows active bonds towards the exact target rank, capped by `chi`.
Randomized guesses remain deterministic for a fixed seed. Dense TreeFIT
updates preserve canonical metadata and the represented exponent; no implicit
normalization is performed on a non-unitary target.
`guess-direct` applies the operator to a private copy and compresses it;
it is distinct from the unchanged current-state `direct` initialization.
Native even-parity fermionic projections apply the graded metric correction
on dual open environment legs before local factorization. This preserves an
already representable state at each local update, including one-node refinement.
Odd-parity fermionic FIT remains explicitly unsupported.

Disposable compressed guesses copy the state once through the ordinary state
handoff. They do not clone the parent optimizer's queued gates, replay history,
or diagnostic records. They retain the previous single child-seed draw from
the parent RNG, preserving later measurement sequences; randomized
compression still uses `fit_init_seed`. Public `TreeOptimizer.copy()` preserves independent
histories, replay configuration, and a derived child RNG; it also preserves the
configured `fit_adaptive_sweeps`.

`get_fit_diagnostics()` returns an independent record for the latest completed
FIT update, including its effective `max_bond` and `cutoff`. It returns `None`
outside FIT or after a completed non-FIT update. `fit_diagnostics` retains
historical FIT records; replacing the state through `set_tn` / `set_p` clears
both the history and latest record along with the other update diagnostics.
The record's `split_method` identifies the actual local factorization:
`"direct"` or `"dm"`. Explicit direct/DM compression settings are retained;
SRC/SDC/SDCR settings map to direct local SVD because complementary-environment
compression is a separate whole-subtree algorithm. `fit_init_strategy`
independently determines the disposable guess method.

TreeOptimizer defaults to `fit_rtol="auto"` and `fit_min_iter=2`:

| State precision | `cutoff="auto"` | `fit_rtol="auto"` |
| --- | --- | --- |
| 16-bit | `1e-3` | `1e-3` |
| float32 / complex64 | `1e-6` | `1e-5` |
| float64 / complex128 | `1e-12` | `1e-9` |

The tolerances resolve from the installed state's dtype at construction,
matching MpsOptimizer's numerical policy. `cutoff_mode="auto"` retains
`"rsum2"`. Explicit numbers remain unchanged, and `fit_rtol=None` disables
tolerance stopping. `fit_patience=1` means one stable comparison of two
same-phase norm samples; the counter resets when the block size changes.
`fit_n_iter` remains the iteration budget, and rank-growth/warm-up conditions
still gate early stopping. This criterion measures relative change in the
retained canonical-center norm, not a bound on the global state error.

As with MpsOptimizer's non-unitary policy, automatic tolerance stopping is
disabled during `run(non_unitary=True)`. It is also disabled for updates with
`track_norm=False`, whose target norm is not assumed known. Explicit numeric
`fit_rtol` remains honored in those cases. Optimizer FIT diagnostics report
both `fit_rtol_requested` and the effective `fit_rtol` for each update.
Standalone TreeFIT retains `rtol=None` and its fixed-block defaults. All three
entry points accept `two_site_transition_sweeps=0`; set this to one with
`block_size=3, adaptive_block_sweeps=2` to request the new transition explicitly.

TreeFIT accepts `retag=True` for structural node-tag alignment and
`copy_target=False` for an explicitly disposable target. Its target may be a
fused tree network or a correctly tagged layered tree network. Every target
tensor must belong to exactly one structural node group; local layer bonds
stay inside a group, and one or more inter-group bonds must follow the fitted
tree edges. Ambiguous or untagged layer tensors are rejected rather than
dropped. The separate two-layer path compressor remains the `TreePeps`
`sdc`/`sdcr`/`src`/`zipup` route when that direct Quimb path is desired.

`TreeOptimizer`'s DMRG target is built as a layered operator--state network:
the state and TreeMPO virtual bonds are not fused, and only corresponding
physical input/output legs are connected. This keeps the DMRG target aligned
with the layered FIT representation while leaving the direct TreeMPO route
unchanged.

All ordinary DMRG gate entries now use `apply_sub_mpotree` too. The optimizer
transfers ownership of its disposable layered target to TreeFIT with
`copy_target=False`. `fit_finite_check=False` is the default; enable it to
check active tensor entries once per iteration. `run(finite_check=True)` also
checks active FIT tensors and scans the final state in every replay mode,
including empty streams. This setting is scoped to that replay and inherited
by shot workers unless `run_kwargs` overrides it. Enabled replay warns once
about the optional diagnostic cost; nested FIT calls share that warning.
Standalone TreeFIT uses `finite_check=False` and warns when scans are enabled.
Scalar convergence, scale bookkeeping, and explicit overlap diagnostics remain
independent; the existing scale/zero guards retain their semantics.
Trusted local updates no longer recompute every
outside isometry after each sweep; explicit `state.validate(check_canonical=True)`
remains available. The existing `track_infidelity` default is unchanged.

TreeFIT's `local_fidelity` is the clipped squared ratio of the retained
terminal canonical-centre norm to the fixed target norm, matching MPS FIT's
local norm diagnostic. `local_norm_trace` contains one such centre readout per
completed sweep and `local_norm_stripped_trace` preserves its mantissa and
base-ten exponent. It is distinct from the optimizer's `norm_diagnostics()`
local and cumulative retained-norm proxy, which is computed from
canonical-centre norm ratios and stored in logarithmic form to avoid
underflow. For lazy targets, pass the known exact `target_norm` (a norm or
mantissa/exponent pair) to obtain this normalized ratio without additional
target work. TreeOptimizer supplies the pre-update canonical norm when
`track_norm=True`, the unitary-update contract. For non-unitary updates use
`track_norm=False`; an unknown lazy target norm yields `local_fidelity=None`
while the retained norm and convergence trace remain available.
Routine fitting never contracts `target.norm()` or a doubled target network.
`fit_diagnostics(overlap=True)` separately requests a genuine directional
overlap. If the target norm is unknown, this explicit diagnostic uses a
lossless leaf-to-hub QR pass and one hub norm, never `<target|target>`.
Its directional overlap is reported as `target_fidelity`,
with MPS-compatible `fit_overlap_fidelity` aliases, not as `local_fidelity`.
Thus an MPO/TreeMPO identity normalization scale is not silently treated as
compression fidelity.
Each completed local fit also carries the target's stored exponent into the
fitted state, preserving its represented scale when the guess has a different
exponent.
Optional norm or overlap diagnostic failures are reported in
`fit_overlap_error`; they do not invalidate or prevent installation of a
successful fit.
TreeFIT rejects odd-parity fermionic tensors: its local projection does not
yet preserve their graded signs. Use the native `direct` or `zipup` routes
for those states. This restriction applies to DMRG modes as well.

`TreeOptimizer` accepts Quimb's `cutoff_mode` conventions for every truncating
Tree-edge SVD. Its defaults, `cutoff="auto"` and `cutoff_mode="auto"`, resolve
once at construction from the installed state dtype, using the shared MPS
policy: `1e-3` for 16-bit data, `1e-6` for float32/complex64, and `1e-12` for
float64/complex128. An explicit numeric cutoff, including zero, is preserved.

`cutoff_mode="auto"` (or the compatibility spelling `None`) selects `"rsum2"`
for every tree split, including `mode="dm"`, `mode="tree_mpo_dm"`, and
`compression_mode="dm"`. This matches MPS DM's **numerical criterion**:
tree DM's `svd:eig` kernel truncates singular values `s`, while MPS MPO DM
truncates density-matrix eigenvalues `s**2` with its native `"rsum1"` mode.
Both automatic rules bound the relative discarded squared weight
`sum(s_discarded**2) / sum(s**2)` when the bond cap does not force more loss.
Explicit modes pass through unchanged: tree `"rsum1"` instead bounds
`sum(s_discarded) / sum(s)`, and `"rel"` uses a relative largest-singular-value
threshold. Copies preserve the resolved cutoff and mode.

The lower-level `TreeTensorNetwork.compress_edge_` API retains explicit
numeric defaults (`1e-10` and `"rsum2"`); pass its cutoff controls directly
when using that lower-level interface.
