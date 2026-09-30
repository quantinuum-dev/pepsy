# Tree states and canonical form

[Tree API overview](tree.md)

Construct a `TreeTensorNetwork`, move its canonical centre, and preserve its backend and gauge metadata.

See [initial-state handoff](tree_layout.md#initial-state-layout-handoff-and-array-backends)
for matching a state to its layout and preparing backend-compatible gates.

## Tree state class

`TreeTensorNetwork` is the tree analogue of Quimb's `MatrixProductState`: a
geometry-owning subclass of Quimb's arbitrary-geometry vector class
`quimb.tensor.TensorNetworkGenVector`. It *is* a Quimb tensor network, so all of
Quimb's arbitrary-geometry methods (`canonize_around`, `canonize_between`,
`compress_between`, `gate_inds`, `to_dense`, `copy`, ...) apply directly; the
class adds the naming and geometry glue on top of a
`TreePlan`:

- every node (leaf **and** internal) is one tensor tagged with the structural
  node tag `node_tag_id.format(nid)` (default `"N{}"`);
- leaf tensors additionally carry the Quimb site tag `site_tag_id.format(q)`
  (default `"I{}"`) and physical index `site_ind_id.format(q)` (default `"k{}"`)
  for qubit `q`; when `plan.root_qubit` is set, the root tensor carries that
  qubit's site tag and physical index as well;
- adjacent nodes share one live virtual bond. Newly constructed edges use the
  deterministic `_tb{lo}_{hi}` name, but Quimb may replace it with a UUID during
  threading or canonicalisation; `TreeTensorNetwork.bond(a, b)` resolves the
  current live index.

Because the geometry (`plan`) and naming live in `_EXTRA_PROPS`, they survive
`.copy()` and every Quimb view, exactly like `site_ind_id` does for an MPS.
Build one with `TreeTensorNetwork.from_plan(plan)` (product `|0...0>`),
`TreeTensorNetwork.from_order(order, structure=...)` (build the plan and the
product state in one step; its default exposes the ternary virtual root), or
`TreeTensorNetwork.rand(plan, D=..., seed=...)`
(a random state, canonicalised around the root by default). `TreeOptimizer`
builds and evolves its state on this class, delegating all node/qubit naming and
geometry queries to it.

`TreeTensorNetwork.local_expectation(op, where, max_bond=None)` has two
backend-specific exact paths. Dense/nonfermionic TTNs move the centre to the
target physical node/subtree, cancel the ordinary isometric exterior, and contract only
the minimal Steiner subtree. Native fermionic TTNs insert the Symmray operator
without densifying it and contract the complete doubled tree, preserving every
graded boundary phase. For native fermionic states, `max_bond` is accepted for
API compatibility but cannot truncate this exact doubled-network contraction.
Observable readout deliberately belongs to the state, not to `TreeOptimizer`;
use `optimizer.tn.local_expectation(...)`.

Readout is gauge-preserving: a dense expectation restores the previously
tracked canonical centre/region, while an unknown dense gauge is evaluated on a
temporary state copy. Native fermionic expectations do not move the gauge.
The repeated normalized native-readout denominator is cached and invalidated by
state mutation, copying, caps, and canonical/gate updates.

`TreeTensorNetwork.local_expectations(terms, optimize=..., normalized=True)`
evaluates many observables at once, where `terms` maps each `where` (an int
site or a tuple of sites) to its operator. It delegates each term to
`local_expectation` with a *shared* `optimize` handle, so a reusable
`pepsy.build_optimizer(...)` caches one contraction path per topology, and it
reuses the memoized graded norm across the batch. Each returned value matches
the corresponding single-term call exactly. For a Hamiltonian-level energy
readout or variational energy optimization, `pepsy.TreeEnergyOptimizer` wraps
this batch path, returns an `EnergyEstimate` mirroring `MpsEnergyOptimizer`,
and exposes the corresponding `make_tn_optimizer` / `optimize` methods.
Optimization updates the tree tensor parameters with Quimb's autodiff
`TNOptimizer` while retaining the exact tree local-expectation objective. For
ordinary readout the native graded norm is memoized. The optimization loss
uses a fresh full doubled-tree denominator because Quimb's direct parameter
injection cannot invalidate that cache; the post-optimization state is marked
non-canonical rather than recanonicalized around an arbitrary centre.

For the package-level product-state constructor, matching `ps_to_mps`, use
`pepsy.ps_to_ttn(n, theta=..., tree=...)`. It builds the requested tree,
initialises every physical site with `[cos(theta), sin(theta)]`, and optionally
expands the virtual bonds with `chi`. Pass `root_qubit=q` to build the plan
directly, or supply a matching root-site `TreePlan` through `tree=`.

For a native Symmray fermionic state, pass a `Fermion` model and occupations:
`pepsy.ps_to_ttn(n, tree=plan, fermion=fermion, occupations=..., chi=1)`.
Physical sites then carry the model's charge/parity sectors, virtual-only
internal nodes are neutral, and every tree edge uses conjugate Symmray virtual
indices.
The constructor selects a definite local Fock basis vector, not a random vector
inside a degenerate charge sector. For spinful `U1`/`Z2`, a scalar occupation
`1` selects the checkerboard `|up>, |down>, ...` representative; pass
`(n_up, n_down)` occupations to choose each spin explicitly. The completed
graded product tree is normalized by an exact graded norm contraction, so its
represented norm is one rather than an arbitrary constructor scalar.
`pepsy.hrs_to_ttn(..., chi=...)` creates the corresponding random symmetric
tree with the requested charge-sector bond dimension and accepts the same
`root_qubit=`, `max_arity=`, and `top_arity=` options. These constructors keep
the Symmray arrays native; they do not materialize dense tensor data.

`pepsy.TreeSampler(state)` samples every registered physical site, including
the optional root site. Its cached canonical arrays use parent, physical, then
child axes, so probabilities and amplitudes retain normal `q0..q(n-1)` order.
For native Abelian or fermionic Symmray trees, pass
`backend="symmray"` (or `backend="native"`) to keep sampling,
amplitudes, probabilities, and edge entropy on block-sparse tensors; inspect
`sampler.physical_code_maps` when the physical basis has degenerate charge
sectors.

`TreeTensorNetwork.show()` prints a top-down ASCII drawing of the tree -- the
tree analogue of a quimb MPS `show()` -- with the root at the top, structural
leaves at the bottom, physical sites labelled by qubit, and every branch
annotated with its current virtual bond dimension
(`ascii_tree()` returns the same drawing as a string).
`TreeOptimizer.show()` delegates to it.

## Canonical state updates

The orthogonality centre is a single node id tracked on the
`TreeTensorNetwork` itself (`orthogonality_center`), so the state -- not any one
driver -- owns the canonical form; it survives `.copy()` and is what
`TreeOptimizer.center` reads. It is moved with
`TreeTensorNetwork.shift_orthogonality_center(node)`, the tree analogue of
Quimb's MPS `shift_orthogonality_center`: the centre is walked to the target
along the unique tree geodesic with a per-edge lossless QR (Quimb
`canonize_between`), touching only the tensors on that path (an O(path length)
move, not O(N)). The move is idempotent when already centred; when the centre is
unknown it is established once with Quimb `canonize_around`. This mirrors the
`info_c["cur_orthog"]` centre tracking of `MpsOptimizer`.
`TreeTensorNetwork.is_canonical_form(center)` verifies the property directly
(every non-centre tensor is an isometry toward the centre) as a diagnostic/test
aid. `TreeOptimizer` mirrors this public surface: `TreeOptimizer.center` (with
the `orthogonality_center` name-parity alias), `shift_orthogonality_center(node)`
and `is_canonical_form(center)` delegate to the state, so the optimizer and its
`TreeTensorNetwork` speak the same canonicalisation vocabulary.
`TreeOptimizer.sync_canonicalization(center=None)` is the explicit recovery
path after lower-level code directly mutates or canonicalizes `opt.tn`; it
rebuilds the state-owned centre before replay resumes. Post-run diagnostics
should normally use `opt.copy()` so the gate-evolution state is not touched.

Direct state users can call `TreeTensorNetwork.compress(max_bond=..., cutoff=...,
center=...)`. This performs one native leaf-to-centre SVD sweep over every
TreePlan edge and leaves the requested node as the validated canonical centre.
The canonicalization phase is QR-only; `max_bond` and `cutoff` control only the
compression phase.

Local isometry orientation also has one owner: each live Quimb tensor carries
its proven `left_inds`, while `TreeTensorNetwork.isometry_direction(node)` and
`isometry_map()` derive read-only node-to-neighbour views from those tensors.
`can_skip_canonize(a, b)` exposes the exact local condition used to avoid an
already-proven QR, and `validate_isometry_metadata()` checks the local
orientations against the tracked canonical region. `TreeOptimizer` delegates
the same four methods without maintaining another mutable map. Native
fermionic edges use this shortcut only when Symmray reports a fermionic array
with aligned charge maps and a complete `left_inds` proof; otherwise they
retain the explicit graded QR path.

`TreeTensorNetwork.validate()` checks the live tensor set, physical legs, tree
edges, and bond ownership against the `TreePlan`; pass
`check_canonical=True` when the metadata alignment and more expensive numerical
isometry check are also desired.
Direct Quimb mutations such as `gate_inds_`, `canonize_between`,
`compress_between`, and `canonize_around_` invalidate the tracked canonical
region. Call `invalidate_canonical_form()` after mutating tensor data directly;
it clears every tensor's `left_inds` proof as well as the native fermionic
norm cache, without modifying tensor data. Clearing only the region is not
sufficient after an in-place array edit: a stale local proof can otherwise
skip a necessary QR. The optimizer's
state-aware wrappers do both automatically and restore the centre only for
operations that prove canonicality is preserved. `TreeOptimizer` also exposes
`sync_canonicalization(center=None)` to explicitly rebuild a single tracked
centre before replay continues; this explicit recovery clears local proofs too.

Native fermionic trees use a separate graded edge path. Centre moves explicitly
QR-split the Symmray tensor and absorb the native carry into the next node;
edge compression uses a reduced graded core whenever the destination endpoint
is already proven isometric: the active endpoint is QR-split first, and only
its `R` factor is sent to the truncating native block SVD. If that proof is
absent, both endpoints are QR-reduced and their contracted core is SVD'd;
only an unrecognised reduction hint forms the complete two-node tensor.
Dense and nonfermionic trees continue to use Quimb's generic
`canonize_between` / `compress_between` wrappers. A graded exterior is not
assumed to be an ordinary Frobenius identity for readout: a known native
fermionic centre uses a one-tensor `TensorNetwork.H` contraction (which applies
the required outer-leg phase flips), while an unknown centre falls back to an
exact complete doubled-network contraction.

## Range / subtree canonicalisation

The single orthogonality centre generalises to a connected **canonical region**
-- the tree analogue of an MPS mixed-canonical range. `canonical_region` is a
frozenset of node ids tracked on the `TreeTensorNetwork` alongside (in fact,
underlying) `orthogonality_center`, which is simply the one-node special case:
when the region spans more than one node `orthogonality_center` honestly reads
`None`. `TreeTensorNetwork.canonize_subtree_(nodes)` gauges every tensor
*outside* a connected subtree to point inward (Quimb `canonize_around` with
`which="any"`), so the whole state norm is carried by the region tensors --
contracting just the region against its graded conjugate reproduces the squared norm,
exactly as the single centre tensor does for a one-node region. Disconnected
`nodes` raise unless `span=True` auto-expands to the minimal connected subtree
that spans them (`subtree_span`). `canonize_around_qubits_(qubits)` is the
qubit-level entry point: it canonicalises around the minimal subtree spanning
those qubits' physical nodes, so the reduced state on a set of qubits is captured by one
subtree. `is_subtree_canonical_form(nodes)` verifies the outside-is-isometric
property directly; `is_canonical_form` is its one-node case. `TreeOptimizer`
mirrors this too: `canonical_region`, `canonize_subtree(nodes, span=...)`,
`canonize_around_qubits(qubits)`, and `is_subtree_canonical_form(nodes)` all
delegate to the state.

## Native fermionic QR stability

Native Symmray tree routes use Pepsy's internal
`TreeTensorNetwork._native_qr_split` policy for every lossless QR gauge move,
including two-qubit path threading, edge canonicalization, lossless path
splits, and sub-MPO message routing. The corresponding network-level subtree
canonicalization uses the same policy through `_native_qr_options()`.

For native block-sparse tensors, the policy passes `stabilized=False` to
Quimb's QR split. Symmray's stabilized QR phase-normalizes each diagonal of
`R`; symmetry can make a diagonal an exact structural zero, so the phase
`0 / |0|` can produce a NaN in `complex64`. Plain QR avoids that undefined
phase while preserving the exact factorization (`Q @ R`) and the tensor's
`left_inds` isometry metadata. This is a gauge choice, not a truncation or a
change to the represented state, and native `complex64` trees therefore do not
need to be promoted to `complex128` as a workaround for this issue.

The safeguard is tensor-aware: dense TTNs retain Quimb's ordinary stabilized
QR convention. It is internal to the tree implementation, so callers do not
need to pass a QR flag. Native truncating compression continues to use the
graded block SVD and the configured `chi`, `cutoff`, and `cutoff_mode`. This
policy is specific to `TreeTensorNetwork` / `TreeOptimizer`; the separate MPS
optimizer implementation is unchanged.
