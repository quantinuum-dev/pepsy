# Plan: TreePeps and tree-embedded PEPS workflows

## Status

Geometry/state slice and the first dense operator slice are implemented,
including `left_inds`-aware canonical movement and compression,
`TreePepo`, and `TreeSubPepo`. Structured operator backends, layout
optimization, and the optimizer family remain planned.

The current implementation provides `TreePepsPlan.from_shape` and
`TreePeps.from_plan` / `TreePeps.rand` for finite open 2D/3D lattices,
coordinate and logical tag/index accessors, spanning-tree validation, exact
dense/norm/local-observable readout, a coordinate schematic via `show`, and
canonicalization, canonical-region movement, `left_inds` isometry metadata,
`info_c` synchronization, and center-oriented edge compression. The broader
workflow below remains the implementation roadmap. The current operator
slice adds exact small dense factorization, product and identity operators,
dense term sums, support/span/attachment metadata, exact tree-bond fusion on
application, expectation values, and optional post-application compression.

This plan defines a new tree-embedded PEPS family for finite 2D and 3D
lattices. It deliberately does not change the existing `TreeTensorNetwork`,
`TreeMPO`, `TreeOptimizer`, or `TreeStabOptimizer` contracts until the shared
tree primitives and representation boundaries are agreed and tested.

## Executive definition

`TreePeps` is a PEPS-like state with one physical tensor at every site of a
coordinate lattice, while its retained virtual-bond graph is a spanning tree
of the original PEPS lattice graph.

Let `G_lat = (V, E_lat)` be the physical lattice graph and let
`T = (V, E_T)` be the selected virtual graph. The core invariant is

```text
E_T is a subset of E_lat
T is connected and acyclic
|E_T| = |V| - 1
degree_T(site) <= 3
```

Every site tensor has one physical index and one virtual index for each
incident edge in `E_T`. The physical index dimension is arbitrary and is not
counted as a virtual degree. Therefore a `TreePeps` tensor has rank at most
four in the first design (`one physical + at most three virtual`), even when
the lattice is embedded in 2D or 3D.

The important distinction is that “2D” or “3D” describes the embedding and
the source lattice, not the contraction topology. The tensor graph is a tree,
so exact tree messages, canonicalization, and edge compression are available.

## Why this is different from dropping bonds from a live PEPS

A generic PEPS cannot usually be converted exactly by deleting selected
bonds. A removed bond can carry correlations that do not factor across the
new tree edge. Removing it is therefore a variational approximation unless
the state is already factorized across that cut or the removed bond has
dimension one.

The initial API must make this distinction explicit:

- Native constructors build a `TreePeps` directly on a selected spanning tree.
- `TreePeps.from_peps(...)` is rejected by default for an entangled full PEPS,
  or requires an explicit approximation method such as a variational fit.
- Exact conversion is allowed only when the removed bonds are provably
  dimension one or the caller supplies a compatible factorized state.
- No constructor silently deletes PEPS bonds and labels the result exact.

## Relationship to `TreeTensorNetwork`

The two representations share the same mathematical tree-contraction ideas,
but their public contracts are different.

| Existing `TreeTensorNetwork` | Proposed `TreePeps` |
| --- | --- |
| Abstract rooted tree for circuit simulation | Spanning tree embedded in a physical lattice |
| Current plans are optimized around physical leaves and an optional physical root | Every lattice site is a physical site, including branch nodes |
| Internal nodes may be structural bond carriers without physical indices | There are no structural-only nodes in the minimum representation |
| Tree coordinates are layout metadata for a circuit tree | Coordinates and lattice-neighbor edges define legal virtual bonds |
| Current `TreeMPO` matches the existing TTN geometry | `TreePepo` matches the lattice-site tree geometry |

If the existing implementation is generalized later, `TreePeps` may become a
specialized all-sites-physical tree state. The first implementation should not
force the current leaf/root assumptions into `TreeTensorNetwork`, because that
would make internal physical sites, site maps, and canonical-region logic
ambiguous. The recommended approach is to extract or share low-level tree
geometry, path, QR, SVD, and diagnostics primitives while keeping the two
state APIs distinct.

## Representation contract

### `TreePepsPlan` / geometry

The plan owns the physical embedding and the selected tree, independently of
tensor data. It should contain:

- site labels and a stable site-to-coordinate map;
- the physical lattice graph, including legal nearest-neighbor edges;
- the selected tree edge set, with a connected/acyclic validation report;
- node-to-site and site-to-node maps, with one physical site per tree node;
- a selectable root and the tree parent/children orientation;
- unique paths, subtree spans, cut edges, and tree distances;
- the requested `max_virtual_degree`, initially defaulting to `3`;
- lattice dimension and boundary metadata for 2D/3D layouts;
- a reproducible layout report containing objective values and constraints.

The physical lattice graph and the virtual tree graph must remain separate
objects. Non-neighboring physical supports are routed through paths in the
virtual tree; they must not create an accidental second virtual edge.

### `TreePeps`

`TreePeps` owns a `TreePepsPlan` and a tensor for every site. Each tensor must
have:

- exactly one physical leg with a stable site label;
- exactly one live virtual leg for each incident tree edge;
- no virtual edge that is absent from the plan;
- compatible backend, dtype, and device metadata;
- optional symmetry, grading, and fermionic leg-order metadata.

The class should expose tree-native methods analogous to the existing tree
state surface:

```text
TreePeps.validate()
TreePeps.norm()
TreePeps.copy()
TreePeps.canonize_to(site_or_node)
TreePeps.canonize_subtree(sites_or_nodes)
TreePeps.shift_orthogonality_center(target)
TreePeps.compress_edge_(edge, ...)
TreePeps.compress_subtree_(nodes, ...)
TreePeps.local_expectation(op, where, ...)
TreePeps.to_dense(...)
```

The precise names can follow the existing `TreeTensorNetwork` API after the
shared state contract is extracted. Physical-site selectors must always use
site labels or coordinates; structural node ids are an implementation detail
of the selected tree plan.

### Bond and rank conventions

The degree limit applies only to virtual tree edges:

```text
TreePeps local rank <= 1 physical + 3 virtual = 4
TreePepo local rank <= 2 physical + 3 virtual = 5
```

The physical index is not a “one-dimensional bond.” It is a site index whose
dimension may be two, four, or another model-specific value. Documentation
should use “physical leg” and “virtual bond” consistently to avoid confusing
the degree constraint with the local Hilbert-space dimension.

## Layout finding

### `TreePepsLayoutFinder`

The layout finder selects a bounded-degree spanning tree from a physical
lattice or an explicit coordinate graph. It should accept:

- `shape=(Lx, Ly)` or `shape=(Lx, Ly, Lz)` for regular lattices;
- explicit coordinates and legal lattice edges for irregular geometries;
- open boundaries first, with periodic source graphs supported only when the
  selected tree is validated against the periodic edge set;
- gate, Hamiltonian, observable, or PEPO support data;
- `max_virtual_degree=3` as the initial hard constraint;
- deterministic seeds and bounded refinement/search budgets.

The interaction supports are not themselves the tree. They are the workload
used to score candidate trees. A nonlocal support is mapped to the unique
tree path or minimal tree span, and the resulting edge traffic contributes to
the objective.

### Layout objective

The first objective should be a documented weighted combination of:

1. weighted tree-path length for two-site and few-site interactions;
2. peak operator-Schmidt load on a tree edge;
3. total routed edge load;
4. estimated local tensor/bond cost;
5. optional geometry penalties for long physical hops.

Operator-Schmidt ranks should be used where available. Counting only the
number of gates is insufficient because a long-range two-site operator and a
rank-one product operator do not impose the same compression burden.

### Search strategy

The finder should use a staged, reproducible search:

1. Build a legal seed tree from the physical lattice graph.
2. Generate bounded-degree candidates using weighted spanning-tree or local
   edge-exchange moves.
3. Score candidates using the workload objective and per-edge traffic.
4. Apply bounded local refinement, such as degree-preserving edge exchanges
   or tree NNI-style moves where the lattice constraints remain valid.
5. Return the selected plan together with a report of rejected constraints,
   degree usage, path statistics, and edge-load hotspots.

The existing `TreeLayoutFinder` has useful interaction scoring and refinement
ideas, but it produces the current circuit-oriented `TreePlan`. Reuse its
scoring helpers where possible; do not silently reinterpret a circuit tree
with structural-only nodes as a `TreePepsPlan`.

## Canonicalization and compression

### Canonical region

Canonicality is state metadata, not a display property. A canonical `TreePeps`
state has a single center site/node or a connected canonical region. Every
tensor outside that region is isometric toward it along the unique tree path.
The physical leg remains part of the active tensor and is never treated as a
gauge bond.

The state should provide the same core guarantees already established for the
tree optimizer:

- moving the center touches only the unique path when the current proof is
  valid;
- a connected region is supported for local ranges and subtree updates;
- unknown or invalid canonical metadata is explicitly invalidated or rebuilt;
- QR is lossless apart from backend numerical behavior;
- scalar normalization and any distributed tensor-network exponent are kept
  separate from physical renormalization;
- native symmetry/fermionic paths use graded QR and preserve leg metadata.

The center must be owned by `TreePeps`, not duplicated in a future optimizer.
Optimizer wrappers should expose read-only delegates just as the current tree
optimizer delegates to `TreeTensorNetwork`.

### Edge compression

Compression is performed on a selected tree edge after canonicalizing toward
that edge. The edge SVD exposes the Schmidt spectrum across the bipartition of
physical sites induced by removing that edge. The API should support:

- `max_bond` and dtype-aware `cutoff="auto"`;
- explicit Quimb cutoff modes;
- exact `chi=None` behavior subject to numerical cutoff;
- per-edge and per-update discarded-weight diagnostics;
- one-sided reductions only when the live isometry metadata proves them;
- native blockwise graded SVD for symmetry/fermionic tensors.

Topology should remain stable by default even when a compressed edge has
dimension one. Optional bond pruning can be added later, but it must update
the plan, tensor indices, layout report, and all cached paths atomically.

### Subtree compression

For a multi-site update, find the minimal tree span of the support, route the
complete operator into that span, and perform one canonical compression sweep
after the complete operator has arrived. Do not truncate each routing hop:
that would make the result depend on the arbitrary path schedule and would
not give every truncation access to the full operator.

This should reuse the existing tree optimizer's exact-threading and
post-arrival-compression policy, generalized so every node on the route can
also carry a physical leg.

## `TreePepo` and `TreeSubPepo`

### `TreePepo`

`TreePepo` is the operator counterpart of `TreePeps`, but it is not a renamed
`TreeMPO` and it should not subclass rectangular `quimb.PEPO`. It should be a
`TensorNetworkGenOperator` carrying the same immutable `TreePepsPlan`:

- one operator tensor per lattice site;
- two physical legs per site, with explicit input/output roles;
- one operator virtual bond per retained tree edge;
- a private operator-bond namespace, distinct from state bonds;
- at most five local legs under the degree-three contract; and
- `plan_signature` metadata sufficient to reject accidental application to a
  differently ordered or differently rooted `TreePeps`.

The input/output convention must be explicit at the `TreePepo` boundary. An
adapter may expose Quimb's upper/lower names, but application code should use
`input_ind(site)` and `output_ind(site)` so that a later convention change
cannot silently reverse an operator. Operator canonical metadata is separate
from state `left_inds`: use `operator_region`, `operator_center`, and
`operator_left_inds` (or an equivalent private metadata layer).

`TreePepo` must be a separate operator network from the state. Expectation
values and applications contract bra, operator, and ket as separate networks;
the operator must not be densified or installed as a state tensor.

The first operator API should support:

- identity and product operators;
- one-site and two-site terms;
- exact operator-Schmidt factorization for small dense local operators;
- structured Pauli and Hamiltonian terms;
- native symmetry and grading metadata when supplied;
- canonicalization and compression of operator bonds independently of state
  bonds, with explicit operator diagnostics.

`TreePepo.apply_to(state)` is the only supported state-application boundary.
It must first validate plan signature, site ordering, physical dimensions,
backend, and dtype. For a full operator, the exact dense-network operation is:

1. contract each operator input leg with the matching `TreePeps` physical leg;
2. retain the operator output leg as the new state physical leg;
3. fuse each pair of parallel state/operator virtual bonds on the same plan
   edge into one live state-tree bond; and
4. canonicalize and compress the resulting `TreePeps` tree according to the
   requested policy.

The fusion step is what preserves the one-tree state invariant. Leaving the
operator bonds as a second graph would produce a general tensor network, not a
`TreePeps`. The original operator and state are never mutated by application
unless `inplace=True` is explicitly requested. Identity/product operators
should use bond-one factors and avoid materializing unnecessary full-lattice
channels.

### `TreeSubPepo`

`TreeSubPepo` is the optimizer-facing, support-aware form of a local operator.
It should be a distinct object that wraps an active operator network rather
than an alias for a full `TreePepo`. It should carry:

- the physical support sites;
- the connected tree span used for routing;
- local operator tensors on the active span;
- private operator bond labels;
- an explicit attachment map for every span boundary edge;
- operator bond estimates and any requested compression policy.

The support and span are different concepts. A two-site operator may have two
physical support sites but must include every tree node along the unique path
between them in its routing span. A disconnected support is therefore not
stored as a disconnected operator network; its minimal connected span is
computed and reported.

The attachment map is topological, not a cache of live tensor indices. It
records `(inside_site, outside_site)` tree edges and resolves the current state
bond only when applying the sub-operator. This remains valid after state bond
renaming or compression. `TreeSubPepo` should expose:

```text
support
span
boundary_edges
plan_signature
operator_bond_dims
apply_to(state, *, inplace=False, max_bond=None, cutoff=...)
expectation(state, *, normalized=True)
```

For `apply_to`, the exact route is deliberately two-stage. First, inject or
fuse the complete operator over its span without truncating intermediate
tree edges. Second, move the `TreePeps` canonical center to the span and run
one inward compression sweep over the affected span/boundary edges. This
ensures every truncation sees the complete multi-site operator and makes the
result independent of the order in which path edges were visited. The
existing `left_inds` metadata selects the shortest valid center move and
one-sided compression path; it must never be inferred from an unvalidated
operator tensor.

The dense fallback must be guarded by a small `max_operator_sites` or
`max_operator_qubits` limit. Large structured operators should remain native
tree networks rather than becoming exponentially large dense arrays.

`TreeSubPepo` should be the object used by optimizers for local gates,
Hamiltonian terms, Kraus operators, and PEPO fragments. It should not be an
alias for the existing `TreeMPO`, whose geometry is currently tied to
`TreeTensorNetwork`.

## Optimizer family

### `TreePepsOptimizer`

`TreePepsOptimizer` should own a `TreePeps` state and provide:

- layout-aware initialization from a lattice and workload;
- exact one-site updates;
- exact factorization and tree routing for two-site gates;
- `TreeSubPepo` application for multi-site gates and Trotter blocks;
- canonical-region maintenance and edge/subtree compression;
- local expectation and energy evaluation through `TreePepo`;
- state replacement and copy semantics with no implicit lossy relayout;
- backend/device/dtype validation at stream boundaries;
- norm, bond, compression, and timing diagnostics.

Its central method should accept a single protocol rather than a collection
of special cases:

```text
apply(TreeSubPepo | TreePepo | one_site_operator, ...)
```

One-site operators can be absorbed directly. Multi-site inputs are normalized
to `TreeSubPepo`, then use exact span injection followed by one post-arrival
compression. A full `TreePepo` may be applied through the same fusion path,
but the optimizer must not silently densify it. State canonical metadata stays
owned by `TreePeps`; the optimizer only requests center moves and consumes the
returned compression report.

The optimizer is not a drop-in alias for `TreeOptimizer`. It can reuse the
same routing, QR, SVD, backend, and diagnostic helpers, but its state and
layout objects must preserve all-sites-physical semantics.

For an entangled `TreePeps`, changing the tree layout is a physical
approximation, not a metadata-only operation. A requested new layout must
either be rejected or require an explicit variational projection. Product
states may be remounted exactly on a new tree, as in the existing tree
optimizer contract.

### `TreePepsStabOptimizer`

The eventual stabilizer optimizer should preserve the STN representation

```text
|psi> = C |nu>
```

where `C` is the Stim tableau frame and `|nu>` is a `TreePeps` coefficient
state. Physical site labels and the tableau's logical-qubit order must be
stable and independent of tree node ids.

This should share the existing stabilizer frame, measurement, injection,
trajectory, and diagnostic semantics, but it should not blindly subclass the
current `TreeStabOptimizer`: that implementation assumes the current
`TreeTensorNetwork` physical-site and layout contract. A shared frame/stream
adapter or composition layer is safer than duplicating the tableau rules or
forcing leaf-only assumptions into the new state.

Required semantics include:

- physical Clifford events update `C` without changing the coefficient tree;
- physical non-Clifford operators are frame-mapped to the coefficient
  `TreePeps` and routed through a `TreeSubPepo`;
- fixed-basis and basis-updating measurements preserve the physical-state
  contract and update the coefficient tree at the selected site/span;
- Born probabilities remain separate from compression-fidelity diagnostics;
- exact cooling and explicit greedy disentangling remain separate policies;
- trajectory-generated operators are converted internally, while user stream
  payloads obey the strict backend contract;
- small cases match Stim and an independent dense reference up to global phase.

The stabilizer optimizer should use the same `TreeSubPepo.apply_to` protocol
as the deterministic optimizer. Clifford-only events update the tableau and
can leave the coefficient tree untouched; non-Clifford or residual operators
are frame-mapped into a `TreeSubPepo` and applied to the coefficient
`TreePeps`. This keeps routing, canonicalization, compression, and operator
bond diagnostics identical across both optimizer families while leaving Born
probabilities and stabilizer-frame bookkeeping outside the tensor compression
report.

## Proposed public surface

The module placement is provisional, but the conceptual API should look like:

```python
geometry = TreePepsGeometry.from_shape(
    shape=(Lx, Ly),                 # or (Lx, Ly, Lz)
    boundary="open",
)

finder = TreePepsLayoutFinder(
    geometry,
    interactions=terms,
    max_virtual_degree=3,
    objective="hybrid",
    seed=0,
)
plan = finder.recommend()

psi = TreePeps.rand(plan, bond_dim=4, seed=0)
op = TreePepo.from_terms(plan, terms)
subop = TreeSubPepo.from_operator(op, support=(site_a, site_b))

optimizer = TreePepsOptimizer(
    state=psi,
    plan=plan,
    chi=64,
)
optimizer.apply(subop)

stab_optimizer = TreePepsStabOptimizer.from_tableau_and_state(
    tableau,
    psi,
    plan=plan,
)
```

The final import path should live under the owning optimizer namespace. A
dedicated `pepsy.optimizers.tree_peps` package is the current recommendation
because it keeps the existing `pepsy.optimizers.tree` circuit-TTN API stable;
public re-exports should be added only after the API and tests settle.

## Implementation phases

### Phase 0 — contract and shared primitives

- Define `TreePepsPlan`, site/coordinate naming, physical-leg naming, and
  validation errors.
- Decide whether common code is extracted into a private tree-network core or
  shared through narrow public helpers.
- Add a compatibility matrix documenting `TreePeps` versus
  `TreeTensorNetwork`, `TreePepo` versus `TreeMPO`, and dense PEPS behavior.
- Add the future skill and API ownership notes only when implementation begins.

Exit criteria: no ambiguity about physical-site placement, degree counting,
source-lattice edges, layout changes, or exact versus approximate conversion.

### Phase 1 — lattice geometry and bounded-degree layout

- Implement finite 2D/3D coordinate graphs and `TreePepsPlan` validation.
- Implement deterministic degree-three spanning-tree seeds and reports.
- Add workload-aware path/span scoring and bounded refinement.
- Validate degree, connectivity, acyclicity, site coverage, and coordinate
  round trips before constructing tensors.

Exit criteria: plans are reproducible and every returned virtual edge is a
legal source-lattice edge.

### Phase 2 — state construction and exact tree contraction

- Implement product and random `TreePeps` constructors.
- Add live tensor/edge validation and site-aware selectors.
- Add exact norm, local expectation, subtree expectation, copy, and dense
  readout for small 2D/3D trees.
- Keep the state canonical around a selected physical root, with no dummy
  structural root.

Exit criteria: uncompressed states agree with independent dense references on
small 2D and 3D lattices, including branch sites with physical legs.

### Phase 3 — canonicalization and compression

- Add center movement, connected canonical regions, lossless QR, and
  canonicality validation.
- Add edge and subtree SVD compression with `chi`, cutoff, cutoff mode, and
  per-edge diagnostics.
- Preserve backend, dtype, device, symmetry, fermionic metadata, and state
  exponent behavior.
- Add exact no-truncation and bounded-`chi` regression tests.

Exit criteria: canonicalization is lossless, compression is localized to the
selected tree span, and every truncation sees the complete active update.

### Phase 4 — PEPO and sub-PEPO operators

- [x] Implement `TreePepo` identity/product/local-term constructors.
- [x] Implement exact small-operator factorization and dense term sums.
- [x] Implement `TreeSubPepo` support/span/attachment metadata.
- [x] Add non-mutating expectation and application paths with separate operator
  and state bond diagnostics.
- Add native structured Pauli/Hamiltonian terms, symmetry/fermionic metadata,
  and span-local post-arrival compression for optimizer updates.

Exit criteria: local, long-path two-site, and multi-site operators match dense
references at `chi=None`; dense fallbacks are bounded and explicit.

### Phase 5 — `TreePepsOptimizer`

- Add one-site, two-site, and multi-site update routing.
- Reuse exact threading followed by one post-arrival compression sweep.
- Add local fitting, energy evaluation, state replacement, copy, and backend
  stream contracts.
- Connect layout reports to optimizer diagnostics without mutating the state
  during candidate scoring.

Exit criteria: gate streams and Hamiltonian terms reproduce dense small-system
results, and entangled state relayouts are rejected unless an explicit fit is
requested.

### Phase 6 — `TreePepsStabOptimizer`

- Adapt the common Stim/tableau frame to all-sites-physical tree states.
- Implement frame-mapped rotations, fixed/basis-updating measurement, reset,
  sampling, and trajectory paths.
- Reuse shared exact cooling, injection, and diagnostics semantics.
- Validate against Stim and dense statevectors on small 2D/3D trees.

Exit criteria: Clifford-only paths remain coefficient-bond preserving, basis
updates preserve the physical state before projection, and bounded compression
diagnostics do not mix Born probabilities with norm loss.

### Phase 7 — symmetry, fermions, and performance

- Add native `U1`, `U1U1`, `Z2`, and fermionic tree states/operators.
- Add graded canonicalization, operator braiding, and charge-aware layout
  costs.
- Add larger 2D/3D benchmarks outside the package's default test loop.
- Compare tree-embedded workflows with dense PEPS and current TTN routes where
  the representations are mathematically equivalent.

## Validation matrix

The implementation must include focused tests for:

- 2D and 3D lattice plan construction, site coverage, and legal tree edges;
- rejection of cycles, disconnected trees, duplicate sites, and degree > 3;
- physical legs on leaves, branch sites, and the selected root;
- exact product-state remounting and rejection of entangled implicit relayout;
- exact norm and local/subtree expectations against dense references;
- center movement, canonical regions, copy metadata, and canonical checks;
- exact edge compression at `chi=None` and controlled truncation at finite `chi`;
- no truncation during operator routing and one post-arrival compression sweep;
- `TreePepo`/`TreeSubPepo` support-span alignment and non-mutating expectation;
- bounded dense-operator fallback and native structured operator paths;
- NumPy/Torch/CuPy/Symmray backend and dtype/device contracts as supported;
- stabilizer frame, measurement, reset, sampling, injection, and trajectory
  agreement with Stim/dense references;
- deterministic layout reports and non-mutating candidate scoring.

The default smoke suite should remain small. Larger lattices, optional
backends, symmetry, 3D stress, and layout-search benchmarks belong in marked
integration or slow coverage.

## Risks and explicit non-goals

- A spanning-tree approximation is not a full PEPS contraction method; it
  cannot represent arbitrary PEPS loop correlations without increasing the
  topology or using an approximation.
- Automatic deletion of virtual bonds from an entangled PEPS is not an exact
  conversion and is not a default operation.
- The first implementation does not support arbitrary loops, a periodic
  tensor-network state after tree selection, or dynamic topology changes
  during ordinary compression.
- `TreePepo` is not a renamed `TreeMPO`; the two operator classes have
  different physical-site and geometry contracts.
- Layout optimization must not silently change an entangled state's tree.
- Native Symmray/fermionic support should follow dense correctness rather than
  introduce a second canonical-center or backend metadata system.
- A degree-three limit may make some small or irregular source graphs
  infeasible. The finder must report infeasibility clearly instead of
  returning a higher-degree tree against the caller's request.

## Documentation and ownership

Implementation should eventually add:

- `docs/api/optimizers/tree_peps.md` for the stable public surface;
- `docs/development/modules/tree_peps.md` for the implementation map;
- focused tests under `tests/test_tree_peps_*.py`;
- a dedicated `.github/skills/tree-peps/SKILL.md` once the subsystem exists;
- changelog entries only when the first public API is implemented.

Until then, this document is the design source of truth and the existing tree
optimizer documentation remains authoritative for `TreeTensorNetwork`,
`TreeMPO`, and `TreeOptimizer`.
