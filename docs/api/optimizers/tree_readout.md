# Tree readout and diagnostics

[Tree API overview](tree.md)

Measure states, inspect truncation and timing records, and understand the dense and native backend boundaries.

## Readout

`to_dense()` returns the dense statevector in index order `k0, k1, ..., k(n-1)`.
`run(progbar=True)` shows a tqdm replay bar matching the MPS optimizer's core
compression readout: the active mode, exact two-qubit event count `2q`,
cumulative retained-norm fidelity `~F`, and current bond usage `bnd`.
Tree-specific `kq`, `ctrl`, and explicit-operator `mpo` counters are added only
when those event types occur. The bar is display-only and does not replace the
path-level norm ledger or recorded per-edge truncation history. The norm ledger
is the canonical `local_fidelity` / `cumulative_fidelity` diagnostic;
`track_truncation=True` is the more expensive spectrum-attribution diagnostic.
For dense two-level qubit TTNs, `measure(q, outcome=None)` projectively
measures a qubit in the computational basis and returns a bit; `reset(q)`
returns a qubit to `|0>`. Native fermionic TTNs deliberately do not expose
these qubit readouts.
For stream control events, `TreeOptimizer.measure_event`,
`cap_event`, `reset_event`, and `measure_reset_event` build the same tuple forms as
`MpsOptimizer`, including Pauli-basis measurement and reset. Their recorded
results are `(pauli, where, outcome, probability)` in `measurements`.
`cap(q, vec)` contracts and removes one physical site, shifting the remaining labels
above `q` down by one unless stable labels are requested.
For a non-unitary run, `normalize_every=True` (or `normalize_final=True`) keeps
the canonical working tensor numerically normalized and accumulates each
removed base-10 scale in `tn.exponent`; `norm()`, `to_dense()`, copies, and
full contractions continue to represent the original physical scale.
The normalization records expose both the per-event raw scale and the
accumulated exponent. The public `normalize()` method remains a physical
renormalization: it clears that exponent and rescales the represented state to
unit norm. `max_bond()` reports the largest virtual bond. Truncation details are
available through `truncation_report()`, `get_infidelities()`, and
`get_infidelity_samples()` when spectrum tracking is enabled.

## Tree-edge entanglement entropy

A tree has one physical bipartition for every parent-child bond, not one
left/right middle cut. `TreeTensorNetwork.tree_edge_entropies()` measures the
base-2 von Neumann entropy across every such bond:

```python
entropies, edges = opt.tn.tree_edge_entropies(return_edges=True)
# edges[i] == (parent_node, child_node)
single_edge_entropy = opt.entropy(edges[0])
```

The implementation follows Quimb's canonical Schmidt-spectrum approach. It
canonicalizes one private copy around the root, then walks the centre through
the tree and performs one SVD per edge on the centre tensor. Off-centre
isometries alone do not contain the Schmidt weights. The live state and its
canonical centre are unchanged. Dense
Torch/CuPy states stay on their original backend through Autoray linalg, and
native Symmray states use the sector-aware SVD. Only the one-dimensional
singular spectra are reduced to scalar entropy values, so the full statevector
is never formed. `method="eig"` (or `"svd:eig"`) selects the Gram-matrix
variant when that is preferable for the local dimensions.

`TreeOptimizer.entropy(...)`, `TreeOptimizer.tree_edge_entropies(...)`, and
`TreeOptimizer.entanglement_entropy(...)` are thin state-preserving delegates.
These values describe tree-edge bipartitions and should not be interpreted as
an MPS chain-cut profile; each edge is identified by its returned node pair.

## Coefficient-backend feature boundary

`TreeOptimizer` covers the state operations shared with `MpsOptimizer`:
ordinary one-/two-/multi-qubit gates, structured sub-MPOs, Pauli expectation
and projection, measurement, reset, measure-reset, cap, normalization,
copying, canonicalization, layout construction, dense readout, and truncation
diagnostics for dense two-level qubit TTNs. A cap's `absorb` argument is
accepted for stream compatibility. A leaf site absorbs into its unique parent;
a root site is contracted directly without changing the tree edges.
`cap(q, vec)` compacts labels by default; use
`stable_labels=True` (or `compact_labels=False`) to preserve caller-facing
logical IDs across the cap while the internal TTN stays compact.
`TreeOptimizer.qubits`, `logical_order`, `position`, and `logical_site` expose
that mapping. Direct `measure(q, outcome=0|1)` and `reset(q)` use these logical
labels as well. Computational-basis measurement obtains local projected
weights, accepts any positive representable branch, and excludes projection
probability from compression loss. Caps and state replacement synchronize the
root arity with the installed plan so subsequent copies and shots remain valid.
Both operations discard cached gate factorizations tied to the old TreePlan,
including same-size state replacements with a different geometry or site order.
Native fermionic TTNs support native Symmray gates and MPOs, but
the qubit Pauli/measurement/reset helpers intentionally reject them; use the
fermion model's native observable/projector with
`TreeTensorNetwork.local_expectation` instead of silently treating a graded
local space as a qubit.

The shared trajectory runner supports dense-qubit `TreeOptimizer` instances as
well. Independent trajectories can sample Pauli mixtures, depolarizing
channels, and state-dependent Kraus channels; branch probabilities are
evaluated from copied TTNs and selected branches are normalized before replay
continues. Coalesced trajectory replay supports exact branching of mid-circuit
measurement, reset, and measure-reset events through the same paired Born
probabilities used by `measure_pauli`. One-site readout uses local projected
amplitudes; multi-site readout carries a dimension-two parity index through
lossless QR messages on the active subtree. Both probabilities are computed
independently, so a rare branch is not lost by subtracting from one. This
does not form a dense projector, truncate a probability query, or clone the
optimizer and its history. Probabilities ignore the represented exponent;
positive collapsed branches normalize without a fixed probability floor. Use
matrix-valued gate payloads in tree streams (for example `pepsy.h()`), since
textual MPS gate aliases are not normalized by the tree gate parser. Native
fermionic trajectories may use native gates/MPOs, but Pauli/control events
require a model-native observable or projector.

MPS execution modes such as `svd`, `swap`, `perm`, `su`, and `mix` are
chain algorithms and are intentionally not copied into `TreeOptimizer`.
The accepted legacy `mpo` name uses the tree-native route described above.
Tree-native `dmrg`/`dmrg1`/`dmrg2`/`dmrg3` are provided by `TreeFIT` instead.
Tree layout is part of the TTN geometry and is selected with
`tree=`/`layout=` at construction. `StabilizerTreeSimulator` uses
`tree_mpo_direct` or `tree_mpo_dm` for its numerical coefficient updates,
delegating the active-span TreeMPO contraction to this class while keeping
tableau state and stabilizer-specific bookkeeping above it. See the
[StabilizerTreeSimulator API](tree_stabilizer.md) for the supported fixed-basis,
basis-updating, immediate, and deferred magic-injection
Clifford/rotation/measurement paths, bounded dense matrix dispatch, and the
safe MPS naming-compatibility surface.

`TreeOptimizer.run` also exposes the shared shot and MPI entry point. It
creates independent copies from the current tree state, so the parent state,
queued stream, settings, and RNG remain unchanged:

```python
result = optimizer.run(
    shots=1_000_000,
    seed=7,
    mpi=True,
    workers="auto",
    progress="auto",
    retain="none",
)
```

The `strategy="independent"` and `strategy="coalesced"` options follow the
shared runner semantics. `StabilizerTreeSimulator.run` provides the corresponding
API and intentionally supports independent shot distribution only.
`observable`, `chunk_size`, and checkpoint/resume settings require MPI;
supplying them without MPI raises instead of silently using ordinary replay.
Configure FIT controls on the optimizer before starting shots. `run_kwargs`
accepts the child `run` API, not constructor-only `fit_*` arguments.

## Diagnostics

The dominant lever for accuracy at fixed `chi` is the tree structure, so the
finder and optimizer expose diagnostics to choose it:

The same diagnostics are available as a Cotengra-style tent plot.
`TreeLayoutFinder.plot(plan)` is the default tent view (also available as
`plot_tent(plan)`): it keeps the raw graph at the bottom and lifts
the selected hierarchy above its descendant sites: the raw lattice and
optional gate connectivity are gray, while internal tree nodes use a stable order-based
`turbo` palette by default. Incoming edges match their child nodes by default;
pass an explicit `edge_color` for a uniform structural color. Arrows are
disabled by default, matching Cotengra's structural tent view; pass
`show_edge_arrows=True` only when parent-to-child direction is needed.
When the physical background already has its own markers, pass
`show_leaf_nodes=False` to hide the tree's physical leaf circles while keeping
internal tree nodes and hierarchy edges visible. This is useful for a gray
`+`-marked lattice backdrop.
When `site_coords` are supplied, the default tent presentation projects them
with `lattice_skew=0.30` and `lattice_rise=0.18`, and draws gray `+` markers at
the physical sites. Override those values to use a different base projection.
Nearest-neighbor gate edges are not duplicated over the lattice. Supplying
`site_coords={qubit: (x, y)}` places the physical sites on an existing lattice.
It returns `(fig, ax)` and does not mutate the plan or live TTN:

```python
finder = py.TreeLayoutFinder(gates, n=n, objective="congestion")
plan = finder.run()
fig, ax = finder.plot(
    plan,
    site_coords=logical_lattice_coords,
    color_by="scale",
    edge_color=None,
    edge_cmap="GnBu",
    node_cmap="YlOrRd",
    order=True,
    show_edge_arrows=False,
)

# For a live optimizer, the same plot is available without changing its state.
fig, ax = opt.plot_layout(site_coords=logical_lattice_coords)
```

To make the first hierarchy layer easier to see, give the leaf-to-parent
edges a contrasting color while leaving higher layers scale-colored:

```python
fig, ax = finder.plot_tent(
    plan,
    site_coords=logical_lattice_coords,
    color_by="scale",
    leaf_edge_color="#2563eb",
)
```

The default plot is therefore the hierarchy that `TreeLayoutFinder` selected:
one hierarchy edge per parent-child connection, drawn over the physical lattice
and gate connectivity. For a background-free binary check, hide the physical
background with:

```python
fig, ax = finder.plot(
    plan,
    lattice=False,
    show_gate_connectivity=False,
)
```

This leaves only the selected hierarchy. The public tent plot intentionally
does not draw gate-by-gate route overlays.

Use `finder.plot_rubberband(...)` for the same hierarchy in physical-lattice
rubberband form. The optional `viz`
profile provides Matplotlib. The plot uses the same visual
idea as Cotengra's circuit/rubberband views: the source interaction structure
remains visible underneath, and band color can encode either tree scale or
post-order. The default styling is axis-free, following Quimb's schematic
drawings, and the
background lattice is not numbered. Pass `show_axes=True` or
`show_site_labels=True` when those annotations are wanted. A stream-order
colorbar is hidden by default; pass `colorbar=True` when that diagnostic is
wanted.

For a scale-invariant tree view, use `color_by="scale"`:

```python
fig, ax = finder.plot(
    plan,
    site_coords=logical_lattice_coords,
    color_by="scale",
    edge_color=None,
    edge_cmap="GnBu",
    node_cmap="YlOrRd",
)
```

Here leaves are scale zero and nodes use stable colors for their hierarchical
scale. With `edge_color=None`, each incoming hierarchy edge uses the exact
same node-palette color as the node it terminates at, so scale layers remain
easy to follow. Set a literal `edge_color` when a uniform structural edge
color is preferred. The scale colorbar, when enabled, follows `node_cmap`.
Midpoint arrows can show the direction from each parent to its children when
`show_edge_arrows=True`. The mapping is independent of the number or order of gates;
`colorbar=True` then labels tree scale rather than gate-stream order. The plot
has no title by default; pass `show_title=True` if a title is wanted.

For a physical-lattice view closer to Quimb's rubberband drawing, use:

```python
fig, ax = finder.plot_rubberband(
    plan,
    site_coords=logical_lattice_coords,
    color_by="gate",
)

# The live optimizer exposes the same non-mutating view.
fig, ax = opt.plot_rubberband(site_coords=logical_lattice_coords)
```

This keeps the lattice sites and gate connectivity grey and wraps each
non-root tree cluster in a rounded, translucent band. The default is a
Cotengra-style `Spectral` post-order progression, giving each nested band a
distinct color. Use `color_by="scale"` for one stable band color per tree
scale.

- `TreeLayoutFinder.report(plan=None)` summarises the physical-node geodesic
  lengths over the interaction graph (`score`, `max_path`, `mean_path`,
  `weighted_mean_path`) and compares against a balanced index tree
  (`balanced_score`, `score_ratio_vs_balanced`). It also reports
  `edge_loads`, `max_edge_load`, and `peak_bond_growth` for the rank-aware
  congestion estimate.
- `TreeOptimizer.bond_report()` reports the current `max_bond`, `mean_bond`,
  and tensor/bond counts -- bonds pinned at `chi` mean truncation is active.
- `TreeOptimizer.estimate_bonds()` performs the paper's non-mutating dry run:
  it multiplies the operator-Schmidt ranks of gates crossing each tree edge,
  returning the conservative Eq. (4) bound before replay. This is useful for
  choosing `chi`; it deliberately ignores cancellations and can overestimate
  the live dimensions.
- `TreeOptimizer.preflight(...)` turns that bound into explicit resource
  protection. `max_bond`, `max_operator_qubits`, and `max_subtree_nodes` can
  reject a replay with `MemoryError`; the same limits can be passed to the
  constructor for automatic checking before eager replay. The constructor
  defaults to `max_operator_qubits=8` and `max_subtree_nodes=128`; pass `None`
  to disable either guard. Product-Pauli measurements use a factorized parity
  projector and do not materialize a `4**k` dense operator.
- `TreeOptimizer.truncation_report()` exposes the per-edge compression and
  SVD-split history, including before/after bond dimensions. Pass
  `track_truncation=True` to also collect each local full singular spectrum's
  absolute discarded weight and relative discarded fraction. Dense states use
  the global spectrum; native Symmray states compare the full and actually
  retained charge-block spectra using the same sector-aware truncation rule as
  the live update. Spectrum probes are opt-in because they add local SVD work
  per truncation edge. Enabling it emits a one-time warning because the
  diagnostic spectrum probes can add substantial SVD work. It remains
  disabled by default. Lossless zero-cutoff edges that are already within
  their bond cap use QR and do not probe a spectrum even when tracking is on.
  Per-edge retained survival and cumulative infidelity are accumulated in log
  space and exponentiated only for readout, avoiding product underflow on long
  streams.
  The report also contains gate-level `updates`, grouping
  edge events by support and reporting the cumulative relative loss.
- `TreeOptimizer.get_norm_events()` and `TreeOptimizer.norm_diagnostics()`
  expose the separate cheap path-level norm ledger. A Tree event groups the
  complete QR-thread/compression path for one gate or subtree update and
  reports `local_fidelity` plus the log-accumulated `cumulative_fidelity`.
  These are compression fidelities measured from retained norms, not target-
  state overlaps. They are collected independently of
  `track_truncation`; use the latter only when per-edge discarded weight and
  singular-spectrum attribution are required.
  In `norm_diagnostics()`, `norm`/`state_norm` are the live represented Tree
  norm, while `cumulative_norm` is the square-root retained-compression
  proxy. `norm_survival` is an explicit provenance alias for
  `cumulative_fidelity`.
- `TreeOptimizer.convergence_sweep(gates, n, chi_values, ops=...)` replays the
  stream at several `chi` on one fixed tree and returns per-`chi` `max_bond`,
  `norm`, observable `expectations`, `fidelity` against the untruncated state
  (when `2**n <= dense_cap`), and observable `max_drift` between consecutive
  `chi` -- a reference-free convergence signal for large systems. Optional
  observables are evaluated by the underlying `TreeTensorNetwork`; they are
  not a `TreeOptimizer` readout API.

## Native central-edge compression and profiling

For a compression from active endpoint `A` to destination endpoint `B`, the
one-sided native path is valid only when `B` is structurally proven isometric
toward `A` (`can_skip_canonize(A, B, absorb="left")`). It uses

```text
A = Q_A R_A
R_A = U S V†
new_A = Q_A U
new_B = (S V†) B
```

The first QR is lossless and uses `_native_qr_split`; the second factorization
is the only truncating SVD. This avoids SVD'ing the fully fused `A B` tensor,
which can be thousands by thousands at moderate `chi` even when the live
edge is small. The implementation keeps the reduction hint separate from the
destination tensor, uses fresh intermediate bond labels while the old edge is
still present in `R_A`, and restores the original live edge label after the
factors are contracted. `reduced=True` uses the analogous two-sided QR/core
reduction. A positive cutoff never turns this into metadata-only compression.

On the calibrated 6x6 χ=64 complex64 Torch-CPU run (12 threads, 48 gates),
Tree evolution improved from 127.77 s to 6.21 s; MPS took 2.85 s in the same
post-fix run. Thus this fix removes the pathological Tree kernel, but Tree is
still about 2.18x slower for this prefix. The saved profile showed 276 edge
compression events totaling 2.40 s inside 48 update envelopes totaling 6.18 s;
it identified the gate update/threading/contraction path, especially the
central edges, rather than route-length tuning, as the next target.

The update path carries the isometry proof produced by the lossless threading
sweep into the reverse compression sweep, so native edges do not revalidate
the same `left_inds`/charge-map proof at every central edge. Two-site factors
use state-owned unique work labels rather than per-factor UUID allocation;
live routed bonds remain collision-safe across copied states without random
label setup in the hot loop. Native Torch-CPU one-edge contractions use
Symmray's blockwise mode, while CUDA and other backends retain the fused mode.
In multi-site Tree/MPO updates, independent QR messages
landing at the same node are contracted as one batch; dense waves reuse their
worker pool, while native fermionic routing remains serial for Symmray safety.
These changes preserve the complete-gate-before-truncation rule and the
configured χ/cutoff semantics. On repeated warm-cache runs of the same
harness, Tree evolution was 4.90--5.15 s; absolute timings vary with BLAS
thread state, so profile envelopes remain the authoritative comparison.

The remaining two-site update bookkeeping is also shared across arbitrary
gate streams: immutable qubit-support geodesics are cached and re-oriented
against the live centre for each gate, while ordinary one-edge tensor merges
use a direct backend ``tensordot``. Symmray dispatches through its graded
fermionic contraction implementation; unusual hyperedges still use Quimb's
general contraction path. This removes repeated path construction and
contraction-expression setup without changing the routed QR/SVD sequence.

Native complex64 QR also applies a reversible power-of-two scale separately
to small Symmray charge blocks. This avoids a Torch QR failure on finite,
rank-deficient blocks around ``1e-9`` without changing ``Q @ R`` or promoting
the replay to complex128. Native ``reduced="right"`` now has the matching
one-sided endpoint-SVD path; unproven ``right``, ``False``, and ``lazy`` modes
continue to use the conservative complete SVD.

The exact 6x6 ``nsteps=0`` stream (468 gates, χ=64, complex64 Torch CPU,
12 threads, ``track_truncation=False``) subsequently completed without the
previous gate-235 NaN. In one profiled run, MPS evolution took 20.39 s and
Tree evolution 137.11 s; Tree layout planning was a separate 25.79 s, giving
an evolution ratio of 6.72x. This confirms stability, not parity: the remaining
Tree cost is concentrated in the per-gate update envelopes and needs further
backend/profile-guided reduction. The same run's normalized Tree/MPS state
fidelity was 0.590; at χ=64 this measures different truncation histories on
the two geometries, not a QR gauge error. Compare observables or increase χ
when using this number as an accuracy diagnostic.

### QR/hop and bond-growth diagnostics

Construct `TreeOptimizer(..., profile=True)` to split the update envelope into
timed `thread_hop`, `edge_canonize`, and `edge_compress` events. The profile
also records `gate_factorization`, `tensor_absorption`, `center_movement`,
`metadata_path`, and `subtree_hub_merge` phases when those routes are used. The
`thread_hop` events are the exact, lossless QR carry moves; `edge_compress`
events are the truncating SVD work. These timings are nested inside the
per-update envelope and should not be added as independent wall-clock totals;
use `profile_report()["update_seconds"]` as the envelope total. The
`timing_semantics` field records this relationship explicitly. For asynchronous
CuPy or CUDA work, `profile_sync=True` synchronizes the active device at each
phase boundary so phase durations represent device execution; this is a
diagnostic mode and adds synchronization overhead.
With the default `profile=False`, replay does not read update profiling
clocks; `update_history` stores `elapsed_seconds=None`.
Path planning is included in `metadata_path`. For layered direct routing,
`subtree_hub_merge` with `deferred=True` measures queuing the arriving messages;
their actual contraction is recorded later as `tensor_absorption` with
`route="subtreempo_layered"` when that node is ready.
For native Symmray compression, `profile_report()` also returns
`native_compression_routes`: counts of `one_sided_left`, `one_sided_right`, and
`two_sided_reduced` show that the graded reduced-core paths are active, while
`full_svd_fallback` identifies a conservative complete two-node SVD. Route
records have zero duration; use the enclosing `edge_compress` event for timing.

For a dimension-level report, also pass
`track_bond_diagnostics=True`. `bond_diagnostic_report()` then records
the per-update `live_max_bond_before`, `transient_max_bond` during
routing/factorization, and `live_max_bond_after` after the compression sweep.
The former may exceed `chi` by the gate's
operator-Schmidt rank; the latter is the enforced live-state cap. The extra
live maximum scans are opt-in so ordinary replay retains its default cost.

For deterministic small-system fidelity checks, use `norm()` for the
canonical local norm and compare `to_dense()` with an independently replayed
NumPy statevector using a fixed gate stream. This avoids making a restricted
Cotengra overlap path the correctness oracle. A Tree/MPS overlap remains a
useful comparative accuracy diagnostic, but it includes both layouts'
different truncation histories.
