# Tree circuit replay

[Tree API overview](tree.md)

Choose how to apply gates and structured operators, then tune compression and runtime options.

For variational replay, see [TreeFIT and DMRG](tree_fit.md).
For measurement and shot execution, see [readout](tree_readout.md).

## Replay modes

Gates are absorbed into the tree according to the selected optimizer mode:

- **ordinary `apply_gate` entries** in `auto`, `direct`, `dm`, `sdc`,
  `sdc-oversample`, `sdcr`, `sdcr-oversample`, `src`, `src-oversample`, and
  `mpo` are lowered to `SubTreeMPO` with
  `SubTreeMPO.from_gate`,
  factorized on the minimal TreePlan Steiner subtree, and applied with
  `apply_sub_mpotree`. The compression mode controls the tree-edge state
  compression, and gate width no longer causes a dense-route cliff.
- **`tree_mpo_direct` / `tree_mpo_dm`** are explicit names for the same
  TreeMPO path, with direct SVD or density-matrix compression selected by the
  suffix. This route never constructs a chain `sub_mpo`.
- **`submpo`** remains the explicit chain-MPO stream mode. The low-level
  `apply_1q`/`apply_2q` compatibility methods retain their specialized direct
  or chain-MPO kernels when called explicitly; this does not change the
  ordinary `apply_gate` contract.
- **stream events** -- MPS-compatible `measure`, `cap`, `reset`, and
  `measure_reset` entries can be mixed into the stream. Measurements use Pauli
  eigenvalue outcomes (`+1`/`-1`) and are appended to `measurements`;
  explicit `submpo` markers use the same native QR-routing and final subtree
  compression when the payload exposes Quimb's MPO site interface, so
  `to_dense()` is not required; opaque MPO-like payloads fall back to the dense
  recursive subtree-operator path.
  `cap` contracts and removes one physical site, compacts the remaining qubit
  labels above it, and keeps the live tree canonical.

## Multi-qubit / sub-MPO application

`apply_subtree_operator(op, where, *, max_bond=None, cutoff=None,
renormalize=False, track_norm=True)`
applies a general operator on `k >= 1` qubits as a single object, the one-shot
generalisation of the two-qubit gate: a `k`-qubit gate, a multi-site
**non-unitary / Kraus** operator, or a whole **Trotter block**. It is the tree
analogue of a sub-MPO applied over the covering range and then compressed (cf.
Quimb's `MatrixProductState.gate_with_submpo`, which exists for the 1D chain
only). Every mode first factorizes the operator into a compact `SubTreeMPO` on the
**minimal connected subtree** (Steiner subtree) spanning the target physical
nodes, then uses `apply_sub_mpotree`, just like ordinary `apply_gate`.
SRC/SDC/SDCR compute complementary environments from the original
operator/state layers. Zipup streams truncation and DMRG fits that same exact
layered target.
Direct/DM application proceeds recursively from subtree leaves to a hub: each local
state/operator message is losslessly QR-split on one edge and absorbed by its
parent, carrying every still-open operator virtual leg. No dense state tensor
for the whole Steiner subtree is formed. Each routed Q tensor, including native
Symmray graded Q factors, retains its `left_inds` isometry metadata when
available, so canonical recovery recognizes that it already points toward the
hub instead of repeating the same QR. The native predicate additionally
validates charge-map alignment before skipping. Once all MPO factors have
arrived, every
touched edge is compressed once with the configured cap and cutoff. A
zero-cutoff bond within its cap uses lossless QR; positive cutoffs are honored
even below the cap, matching ordinary replay. Each direct/DM truncation sees
the complete operator in an isometric environment.

A compact one-site unitary is absorbed directly in every mode, preserving
the incoming canonical center or region and local isometry metadata. The
current small physical matrix is checked at arithmetic precision, including
even native operators; nonunitary and odd native operators retain the general
route. This exact absorption skips FIT and clears its latest diagnostic record.
Native `compression_mode="dm"` raises before update accounting or state
changes; use `"direct"` for native graded compression.

`op` acts on `len(where)` qubits: an array reshaped to `(2,) * 2k` with output
indices first, `op[o_0..o_{k-1}, i_0..i_{k-1}]` (a `(2**k, 2**k)` matrix is
accepted). It need **not** be unitary; pass `renormalize=True` to renormalise
afterwards (e.g. after a Kraus/projection operator). `max_bond` / `cutoff`
default to the optimizer's `chi` / `cutoff`; explicit values apply only to
this call, including both the initial guess and refinement in DMRG modes.
They do not change the optimizer's defaults. The native TreeMPO route scales
with the operator's spread and factor ranks, using recursive edge messages
rather than one dense state tensor for the whole spanning subtree.

`track_norm=True` records the cheap path-level retained-norm ledger. Set it to
`False` for a known non-unitary operator: its physical norm change is then not
misreported as compression loss. The same keyword is accepted by
`apply_gate`, `apply_1q`, `apply_2q`, `apply_submpo`, and `apply_pauli_sum`.

An explicit MPS-style sub-MPO marker, `("submpo", mpo, where)` (or the
equivalent mapping form), is accepted in a TreeOptimizer stream. Quimb MPOs are
applied natively by carrying their virtual operator bonds through the same
lossless leaf-to-hub QR sweep followed by one subtree compression sweep; bond
estimates use MPO bond dimensions as a conservative Schmidt-rank bound.
Payloads without the required site interface fall back to `mpo.to_dense()`,
which must produce an operator on the declared support.
`TreeOptimizer.submpo_event(...)` builds the tuple form.

For a complete operator already represented as a `TreeMPO`, use
`apply_sub_mpotree(tree_operator, where=None, ...)` or the aliases
`apply_subtreempo` / `apply_sub_tree_mpo` / `apply_subttno`. All refer to the
same implementation. Ordinary gates use `gate -> SubTreeMPO -> apply_sub_mpotree`
in direct, DM, SRC, SDC, zipup, and DMRG modes. Operator bonds follow the
TreePlan geometry; each mode retains its own compression algorithm.
It never extracts a contiguous chain MPO. The operator must use the same plan,
contain one primary network, and either declare all physical sites or declare
exactly its `operator_support` metadata when that known non-identity support is
available. Compact operators also accept their `sites` as the declaration.
They store only their `active_nodes`; exterior identity is implicit. For a
full `TreeMPO`, bond-one identity factors outside a term's active support are still
part of the complete TTNO; the shorter declaration only selects the minimal
Steiner route. Any omitted boundary operator bond must be bond one, otherwise
the application raises instead of silently discarding operator information.
The stream constructor `TreeOptimizer.sub_mpotree_event(tree_operator)` and
the matching `StabilizerTreeSimulator.subtreempo_event(...)` provide the same
native route; `subtreempo_event` and `subttno_event` remain spelling aliases.
The new helper preserves the established `"subtreempo"` wire marker for shared
stream consumers; TreeOptimizer also accepts raw `"sub_mpotree"` markers. Set
`track_norm=False` for a general non-unitary TreeMPO so its physical norm
change is not recorded as compression loss.

The internal Pauli rotation, Pauli-sum, and product-Pauli projector constructors
use a true compact `SubTreeMPO` for Tree evolution. A sparse operator on qubits such as
`(q0, q7)` is represented by MPO tensors only at `q0` and `q7`; identity-only
sites between them are not inserted into a fictitious chain window. The native
Tree MPO router therefore receives the true active support and computes the
minimal Steiner subtree/geodesic before QR routing and compression. The shared
constructors retain their contiguous-window default for the one-dimensional
MPS backend, whose compression domain is a chain interval.

The two explicit two-site families preserve native Symmray gates and their
block-sparse fermionic grading. Ordinary `apply_gate` entries in
`auto`/`direct`/`dm`/`sdc`/`sdc-oversample`/`sdcr`/`sdcr-oversample`/`src`/
`src-oversample`/`mpo` are lowered to a true compact `SubTreeMPO` (a
`TreeMPO` subclass) and contracted through `apply_sub_mpotree` on the active
canonical Steiner region.
`tree_mpo_direct` and `tree_mpo_dm` are explicit names for that same route,
selecting direct SVD or density-matrix compression. The `submpo` mode remains
the explicit chain-MPO stream mode. The low-level `apply_1q`/`apply_2q`
compatibility methods retain their specialized kernels outside FIT, explicit
TreeMPO, and oversampled zipup modes.
DMRG ordinary gates and `apply_subtree_operator` / `apply_sub_mpotree` use FIT;
an explicit structured `apply_submpo` retains its chain-operator routing and
final subtree compression even when the configured mode is DMRG.

For an ordinary gate stream, any of `mode="direct"`, `mode="dm"`,
`mode="sdc"`, `mode="sdc-oversample"`, `mode="sdcr"`,
`mode="sdcr-oversample"`, `mode="src"`, `mode="src-oversample"`, or `mode="mpo"` now
uses the compact `SubTreeMPO` path;
`mode="tree_mpo"` is an alias for `tree_mpo_direct`, hyphenated names are
accepted, and `tree_mpo_dem` is kept as a compatibility spelling for
`tree_mpo_dm`. The combined `tree_mpo_*` names own their compression suffix,
so a conflicting `compression_mode` is rejected.

For ordinary replay, `run(mode=...)` updates the optimizer's selection for
that run, later runs, and copies. Shot replay applies overrides to children.
The old `run(mode="tree")`/`"ttn"` selector is a deprecated no-op retained only
for shared frontends.

Constructor `mode`, deprecated `two_site_mode`, and `run(mode=...)` share
normalization, aliases, and conflict validation. `dm`, `src`, `sdc`,
`sdc-oversample`, `sdcr`, `sdcr-oversample`,
`tree_mpo_direct`, and `tree_mpo_dm` select their named compressor; `zipup`
and `zipup-oversample` (alias `zipup-first`) select streamed direct splits;
`mix` selects direct-guess one-node FIT. These names accept the neutral
`compression_mode="direct"` constructor default or their matching compressor.
Other route selectors (`auto`, `direct`, `mpo`, `submpo`, and DMRG modes)
retain the configured compressor when a run override omits it. To explicitly
switch an existing optimizer to direct compression, use
`run(mode="direct", compression_mode="direct")`.

| Option entry point | Scope |
| --- | --- |
| Constructor `fit_*` controls | Inherited by copies and shot children; unsupported as `run` keywords or inside `run_kwargs` |
| Ordinary `run(mode=..., compression_mode=..., compression_seed=..., max_bond_oversample=..., cutoff_oversample=..., cutoff_mode_oversample=..., track_infidelity=...)` | Persistent after argument/stream validation succeeds |
| Operator `max_bond=...`, `cutoff=...` | One call, including its compressed guess and FIT; `None` inherits optimizer settings |
| Shot top-level replay options | Child overrides; parent state, queue, configuration, and RNG remain unchanged |
| Shot `run_kwargs` | Explicit child replay settings take precedence over corresponding top-level options |

Integer caps, worker counts, and FIT budgets require actual integers rather
than booleans or fractional values. Cutoffs must be finite and nonnegative
(or `"auto"`); cutoff modes accept Quimb's named modes and integer codes 1–6.
Invalid replay configuration or stream labels are rejected before installing
replacement options or a replacement queue. Numerical failures after replay
starts do not promise rollback of preceding events.

For the ordinary TreeMPO route, the decomposition used for state truncation
can be selected independently with `compression_mode="direct"` (SVD) or
`compression_mode="dm"` (Quimb's density-matrix-equivalent `svd:eig`
decomposition on the local fused state/operator compression core).
The latter does not form a global dense state and is currently restricted to
dense tree tensors; native fermionic trees retain their graded direct
compression path. `mode="dm"` is the automatic-routing shorthand for
`mode="auto", compression_mode="dm"`.
For NumPy single-precision arrays, Pepsy probes the installed Quimb eigensolver
once per dtype. If its Numba driver cannot compile, tree edge compression
warns and uses direct SVD with the same dtype, cutoff mode, and bond limit.
Other backends and working eigensolver builds retain the requested method.
The combined `tree_mpo_direct` and `tree_mpo_dm` names select both the true
TreeMPO route and its compression method, so a conflicting explicit
`compression_mode` is rejected.

Dense compression enforces `chi` as a hard cap. Native Symmray SVD retains
its existing global truncation policy: equal singular values at the boundary
can be kept together, so a degenerate cut can exceed the requested `chi`.
`max_bond()` reports the actual retained dimension. Path traversal preserves
this policy; it does not select Symmray's different per-sector eager allocation.

For a path-shaped active subtree, direct and DM prepare the complete operator
losslessly, then compress each edge once from one endpoint to the other.
Exact TreeMPO preparation can peel from both ends to keep QR matrices small;
it then moves the completed center to the compression entry endpoint.
Before routing, direct/DM and SRC/SDC/SDCR reuse a canonical region already inside
the active subtree. Otherwise, a known center moves only to the first subtree
entry. A known multi-node region is peeled only outside its overlap with the
active region; disjoint regions require only the old region and their unique
connector. Unaffected exterior branches and the retained overlap are not
decomposed. A genuinely unknown gauge uses full exterior canonicalization.
Interior preparation
QRs are unnecessary because the routing algorithm replaces those tensors.
They retain the terminal endpoint as the canonical
center, without a return QR walk. This includes two-qubit gates and few-body
gates whose sites lie on one geodesic. The retained endpoint is nearest the
incoming tracked center, with structural node id breaking ties; this routing
order never changes the gate's logical argument order. Off-path branches
remain canonical boundaries with their existing virtual dimensions. Explicit
subtree-operator and sub-MPO paths use the same directional compression.
Branched regions retain their tree sweep. Finite-cap results can change from
the previous interior-hub order.

SRC/SDC/SDCR and their oversampled variants likewise use an endpoint hub on paths, building only the complementary
environments needed for the opposite projection sweep. Zipup uses a directional
path sweep with immediate truncation. These modes keep their distinct algorithms;
the tree is not converted to an MPS.

`mode="zipup"` uses the same `apply_sub_mpotree` entry point, but contracts
one layered operator/state node only when its incoming child messages are
ready. An SVD immediately caps the outgoing state leg at `chi` before the
message reaches its parent. This avoids constructing the fully enlarged
tree before truncation. The unvisited environment is not canonical, so
zipup's intermediate discarded weights are not global error estimates; its
accuracy can differ from `direct` at the same `chi`. It uses direct SVD,
rejects a conflicting `compression_mode`, and leaves a canonical hub.
Dense NumPy/Torch and native fermionic arrays retain their backend and dtype.
Provably lossless zero-cutoff messages use the shared QR policy instead of
SVD. Bond-size history is recorded, but zipup does not report intermediate
discarded spectra as canonical truncation errors, even with
`track_truncation=True`; those error fields remain unavailable.
On a native fermionic tree, an overly small cap can remove every compatible
charge path. Zipup raises before installing such an empty state; increase
`chi` or choose `direct`, whose cuts see the complete operator environment.

`mode="zipup-oversample"` (also `"zipup-first"`) first runs streamed zipup
at an intermediate rank, then directly rounds the active subtree to `chi`
using the final `cutoff` and `cutoff_mode`. No complete uncompressed target
tree is materialized. The intermediate rank defaults to `2 * chi`;
`max_bond_oversample=128` sets an explicit rank, while `2.0` sets a multiplier.
The final cap must be finite, either from `chi` or the per-call `max_bond`.
The first pass uses `cutoff_oversample=0.0` and
`cutoff_mode_oversample="rel"` by default. These controls are independent of
the final cutoff, whose default convention remains `rsum2`.

```python
optimizer = TreeOptimizer(
    gates, state=ttn, chi=64, mode="zipup-oversample",
    max_bond_oversample=2.0,
)
```

Both passes preserve native NumPy/Torch or Symmray arrays and leave the tree
canonical. Native final cuts retain the shared charge-multiplet policy, so
a degenerate multiplet may exceed the nominal cap. Truncation history labels
the intermediate cuts `zipup` and final cuts `zipup_oversample`, including
each pass's own cap and cutoff convention. As with ordinary zipup,
intermediate spectral-error estimates are unavailable; the update's norm
ledger covers the composed operation.

The final round is shared with SRC/SDC/SDCR oversampling. An endpoint-rooted
path is compressed once toward the opposite endpoint and keeps that endpoint
as its canonical center, without return QRs. Branches retain depth-first cut
order and the return moves needed to prepare the next branch. The cuts and
represented result of an individual round are unchanged up to numerical
roundoff; later finite-rank gates can change because their routing direction
depends on the new center. Caps, cutoff conventions, and RNG policy are
unchanged.

For a full `TreeMPO`, the minimal subtree shortcut requires exterior tensors proven to be unit
identities by `TreeMPO.from_gate` or `from_pauli_sum`. Copies, scalar scaling
on the active support, conjugation, and internal backend conversion preserve
that proof. Operator canonicalization, distributed scaling, or exterior
tensor replacement can move physical scale into those tensors. Such operators
use the full tree in every mode, which is conservative and can cost more than
the original local route. Compact `SubTreeMPO` operators have no exterior
layer and never need this proof or fallback. For full operators, the check
compares array references and index metadata without
contraction or backend transfer. After editing array entries directly in place,
call `operator.invalidate_canonical_form()` to invalidate the identity proof
as well as its gauge metadata. A caller-supplied `operator_support` hint alone
does not establish the builder's identity proof.

`compression_mode="src"` contracts product-noise sketches of complementary
branches, caching a directed environment on each tree edge. A second sweep
forms QR projectors using those environments and the original layered target,
then propagates the projected target toward the hub. `compression_seed=...`
makes the sketches reproducible. As in Quimb SRC, the sample count is set by
`chi`; nonzero `cutoff` is ignored with a warning. With `chi=None`, the sample
count uses the largest original cut dimension to retain the full range.
The implementation follows the Khatri–Rao sketches, QR range extraction,
and target projection in [SRC, Algorithm 1](https://arxiv.org/html/2504.06475v2#S3).
The paper defines a chain algorithm; the tree route extends its QB construction
to directed branch environments without claiming the paper proves that
extension. On layered paths, the same seed now matches unmodified Quimb SRC
in both directions. This changes seeded results relative to the earlier
per-node seed-offset implementation.

Only required environments and their dependencies are built. Path targets
therefore use one fixed environment per edge, matching Quimb. Each environment
is released after its last consumer, and projector QR requests only Q.
Numerical environments are cached within one compression call and shared by
its dependent branches. Across calls, a bounded cache (128 plans) reuses only
immutable tree traversal and dependency information, keyed by edge order and
hub. Arrays, dimensions, sketches, and consumption counters are always fresh:
in-place edits, new operators, ranks, seeds, backends, and failed retries cannot
reuse stale numerical environments.
Intermediate contractions drop tags; final tensors retain their local tags.
Only the exterior is canonicalized before SRC/SDC/SDCR; the active tensors are
replaced by the projector sweep without an initial internal center move.

Torch SDCR requires Quimb's dtype-aware randomized split driver (the API with
`noise_dist`). Older drivers can mix real noise with complex Torch tensors;
Pepsy reports that missing capability explicitly. Use SDC/direct compression
or upgrade Quimb for this backend path.

`compression_mode="src-oversample"`, `"sdc-oversample"`, and
`"sdcr-oversample"` are accuracy-oriented opt-in variants. Each first uses
the corresponding successive environment method at an intermediate rank, then
uses direct SVD compression with the requested final cutoff. The default
intermediate rank is Quimb's
`max(round(1.5 * chi), chi + 10)`; set `max_bond_oversample` to an explicit
rank or to a floating-point multiplier of `chi`. Path-shaped sweeps reverse
their first direction so the final direct round sees the opposite environment;
branching regions retain their valid tree peel order. None of these routes
materializes a dense state. The branched-tree behavior is a Pepsy tree
extension of Quimb's chain algorithms.

`compression_mode="sdc"` uses the same successive projection structure with
deterministic low-rank complementary environments. Their factors are computed
using direct truncated SVD, avoiding squared conditioning and a NumPy
complex64 JIT failure in the installed Quimb eigendecomposition driver.
`cutoff`, `cutoff_mode`, and `chi` control those environment factors.

`compression_mode="sdcr"` keeps the same successive environment geometry and
target projection, but computes each low-rank environment factor with
Quimb's static randomized SVD driver (`svd:rand`, with no oversampling or
power iterations by default). `chi` is both the sketch rank and the output
cap; `cutoff` is ignored by this randomized environment step, matching
Quimb. `compression_seed` controls reproducibility. Pepsy explicitly sends
`cutoff=0` and `cutoff_mode="rel"` to randomized environment splits, so
future Quimb releases that reject cumulative cutoff modes remain compatible;
the configured final cutoff is still applied by the final direct round.
`sdc-oversample` instead uses `cutoff_oversample` and
`cutoff_mode_oversample` for its deterministic intermediate factors. These
modes are dense-only and should be benchmarked against `sdc` on representative
branched trees before changing defaults.

These are distinct environment algorithms, not local randomized SVD aliases
of `direct`. On a path they reproduce Quimb's SRC/SDC/SDCR sweeps (SRC/SDCR
comparisons use the corresponding randomized seeds). On a branching tree, each retained node incorporates
its already projected children and a cached complementary environment.
They never materialize the complete operator-applied tree before compression.
The final hub is canonical and retains the projected target norm.

All successive environment methods currently require dense arrays. Native symmetry
trees reject them explicitly; use native `direct` or `zipup`. Edge records
report dimensions, but do not invent discarded spectra or global error bounds
from these approximate environments. The existing TreeFIT
`guess-src`/`guess-sdc` initialization options use their corresponding
environment algorithms; TreeFIT's subsequent variational refinement uses
direct SVD. `sdcr` is not a FIT variant. A Cholesky-based compressor is not
implemented by these modes.

## Performance-oriented defaults and warnings

The default replay configuration is intended for production evolution:
`mode="auto"` uses the true TreeMPO route on every ordinary gate support,
while `threads=1` avoids
oversubscribing the small tree contractions, `subtree_workers=1` keeps the
serial path free of thread-pool overhead, `profile=False` avoids timing overhead, and
`track_truncation=False` avoids full-spectrum diagnostic SVDs, while
`track_bond_diagnostics=False` avoids live-bond scans. `record_history` and
`track_infidelity` retain the established API defaults; the latter enables the
cheap canonical-centre norm ledger and its progress-bar readout. It does not
enable spectrum probes.

`fit_finite_check=False` keeps optional FIT finite scans off. MPI shot replay
also defaults to `collect_diagnostics=False`; pass
`True` explicitly to request rank timing diagnostics.

For Torch, JAX, and CuPy, ordinary replay retains detached norm and
log-fidelity scalars on the array backend. `norm()`, `get_norm_events()`,
`norm_diagnostics()`, diagnostic reports, and progress display are explicit
host readout boundaries. A nonzero extracted `tn.exponent` retains Python
double scale bookkeeping to avoid losing its range on float32-only devices.
Norm tracking remains enabled by default. FIT convergence/local reports,
measurement decisions, native QR safety checks, and upstream truncation rank
selection can still synchronize; disabling optional diagnostics does not
disable these algorithmic decisions.

Warnings are reserved for an actionable behavior change: enabling
`track_truncation=True` emits one diagnostic-performance warning, while
legacy mode selectors emit deprecation warnings. Explicit user stream
backend/device mismatches and incompatible dtypes are errors, not implicit
conversions. Dense and
native paths share the direct one-edge contraction, path-cache, routing, and
proof-reuse optimizations. Only the QR phase safeguard and graded reduced-core
SVD are native Symmray specializations; dense arrays continue through Quimb's
ordinary QR/SVD with the same cutoff, path, and truncation semantics.

`TreeOptimizer.apply_submpo(..., track_norm=True)` is the public form for an explicit MPO of
arbitrary support. It losslessly QR-routes its virtual bonds, then uses its
supplied (or configured) `max_bond` / `cutoff` in one final canonical sweep over
the affected subtree. Existing bonds at or below `max_bond` take a lossless
QR; only bonds expanded past the cap invoke the configured cutoff mode.
The tree backend also exposes numerical Pauli primitives used by a future
stabilizer frontend: `apply_pauli_rotation(...)`, `apply_pauli_sum(...)`,
`expectation_pauli(...)`, `measure_pauli(...)`, and `project_pauli(...)`. These
operate on dense two-level qubit coefficient states and do not require tableau
metadata; they intentionally reject native fermionic TTNs.
`measure_pauli` returns `(outcome, probability)` and accepts an optional
`return_diagnostics=True` flag. `project_pauli` normalizes by default; pass
`renormalize=False` to retain the branch norm. Both APIs can report projection
diagnostics containing the norm ratio and support, spanning-tree, and bond
snapshots before and after the update. The records are also available through
`projection_diagnostics` and `get_projection_diagnostics()`.

## Performance and stability

- **Lossless QR fast paths.** Zero-cutoff splits and edge updates whose
  rank is already within the active bond cap use QR rather than SVD. This
  includes the sibling-leaf split and remains valid for native graded QR.
  A positive cutoff retains the existing rank-revealing compression semantics.
- **Repeated-gate cache.** Direct gate SVDs and MPO factorizations are cached
  by immutable payload identity, backend signature, support, and local
  dimensions. The bounded cache returns fresh-index tensor copies, so it does
  not share mutable network indices with the live state.
- **Subtree and pilot parallelism.** Set `subtree_workers>1` to evaluate
  independent dense leaf-to-hub QR messages in a wave, with deterministic
  merging. Native fermionic routing stays serial because graded Symmray phase
  bookkeeping has not been established as thread-safe. Set `pilot_workers>1`
  for independent layout pilot replays; both options default to one.
- **Sibling fast path.** A two-qubit gate on two leaves that share a parent is
  applied as a single two-site update: the two leaves and their parent are
  contracted into one blob, the gate is applied, and the blob is re-split by
  two truncating SVDs against the (isometric) surrounding tree. This avoids the
  QR bond-threading and double-bond fusion of the general geodesic route and is
  the common case in a locality-aware layout.

- **Thread cap.** Tree tensors are moderate-rank (set by local arity and the
  optional root physical leg, with dimensions bounded by `chi`), so
  multi-threaded BLAS/OpenMP linear algebra is dominated by thread launch and
  synchronisation overhead. `TreeOptimizer` caps threads to `1` around gate
  application and the heavy read-outs by default (`threads=1`), which makes
  replay both markedly faster and stable in wall-clock time; pass
  `threads=None` to leave the ambient thread count untouched (worthwhile only
  in a large-`chi` regime where a single contraction is itself large). Thread
  limiting uses `threadpoolctl` when available and is a no-op otherwise.
- **Lazy canonical centre.** A freshly built product state has every virtual
  bond at dimension 1, so it is already canonical with the root as
  orthogonality centre; `from_plan` records that centre on the network rather
  than recomputing it on the first gate. Native fermionic product trees are
  additionally normalized by their exact graded norm readout.
- **Routed isometry reuse.** Dense geodesic and subtree QR routing retains each
  Q tensor's `left_inds`, allowing later canonical recovery to reuse the proven
  isometry without repeating the decomposition or entering Quimb's dense
  canonicalization kernel. Final path and subtree compression also consults
  that live proof: when the destination-side tensor is already isometric,
  Quimb uses one-sided `reduced="left"` compression and avoids its redundant
  reduction QR; otherwise it falls back to the full two-sided reduction. The
  network derives orientation diagnostics from those tensors; the optimizer
  does not keep a duplicate map. Native fermionic trees keep their separate
  explicit graded QR/SVD path.
- **Regional recovery traversal.** Recovering a single center from a tracked
  canonical region uses a smallest-leaf priority queue. It preserves the
  deterministic QR order while avoiding repeated whole-region scans:
  traversal bookkeeping is O(E + R log R) for R region nodes and E incident
  adjacency entries, excluding tensor operations. Existing `left_inds`
  proofs still turn redundant QR operations into metadata-only moves.
- **State-owned centre.** The orthogonality centre lives on the
  `TreeTensorNetwork` (`orthogonality_center`, an `_EXTRA_PROPS` field), so the
  optimizer and the state cannot disagree and the centre is carried by
  `.copy()`. Incremental moves (`shift_orthogonality_center`) touch only the
  geodesic between old and new centre.
- **Self-healing tid cache.** Node-to-tensor lookups are cached and validated
  against the live tensor map, so the hot path avoids re-scanning tags while
  staying correct when a gate rebuilds a tensor.
- **Resource guards.** `max_intermediate_bond`, `max_operator_qubits`, and
  `max_subtree_nodes` provide preflight and direct-application limits; the
  latter two default to conservative finite values and accept `None` to opt
  out. A dense `k`-qubit operator still has `4**k` payload values, while
  product-Pauli measurement uses a factorized parity projector and recursive
  edge messages.
- **`copy()`.** Returns an independent optimizer that shares the immutable
  `TreePlan` but owns its own tensor network (which carries the tracked
  orthogonality centre), for branching experiments or trial gate sequences.
