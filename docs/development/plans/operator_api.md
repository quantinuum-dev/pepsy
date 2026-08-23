# Operator and exponential API plan

Status: proposed architecture
Last updated: 2026-08-22
Owners: Pepsy maintainers

This plan organizes the MPO, PEPO, higher-order, cluster-expansion, and
ordered-product work without introducing another competing public API. The
first priority is a stable vocabulary and a clear ownership map. Physical
module moves should come only after that vocabulary has been exercised by
examples and downstream callers.

## 1. The problem we are solving

Pepsy has several useful construction families, but they currently meet at
different abstraction levels:

| Family | Current implementation | What it means |
| --- | --- | --- |
| Higher-order MPO exponential | `MPOBasis`, `FirstDegreeMPO`, `CompiledMPOExp` in `operators.mpo_higher_order` | Approximate `exp(step * H)` using virtual history and Taylor/order controls |
| MPO local cluster expansion | `MPOClusterBasisExpansion` and `MPOGraphClusterBasisExpansion` in `operators.mpo_cluster` | Build connected spatial/graph residuals and assemble them as an MPO |
| Fixed-channel PEPO exponential | `PauliPEPOBasis`, `CompiledPEPOExp` in `operators.pepo_cluster` | Differentiable square-lattice PEPO with value-independent Pauli channels |
| Dense PEPO cluster expansion | `ClusterExpansionPlan` and graph plans in `operators.pepo_cluster` | Factor local connected residuals for a finite model or graph |
| Ordered PEPO products | `PEPOClusterProductExpansion` in `operators.pepo_cluster` | Jointly construct `exp(A) @ exp(B) @ exp(C) @ ...` |
| Ordered MPO products | `MPOClusterFactor` and `MPOClusterBasisExpansion` in `operators.mpo_cluster` | Jointly construct local ordered exponential factors on a chain/graph |
| Native Pauli MPO | `PauliMPO` in `operators.pauli_mpo` | Sparse Pauli-basis operator algebra and conversion to an MPO |

These are not all the same algorithm. In particular:

- Taylor/history order and spatial cluster order are different approximation
  axes.
- `exp(A + B)` is different from `exp(A) @ exp(B)` unless the factors commute.
- A sparse active PEPO, a semantic `FirstDegreeMPO`, and a materialized Quimb
  MPO/PEPO have different responsibilities and should not be silently
  interchanged.
- BP loop-cluster expansion belongs to tensor-network contraction, not to the
  local operator cluster-expansion builders.

The architecture should make those distinctions visible instead of hiding
them behind more convenience functions.

The key ordered-product invariant is the same for MPO and PEPO: for factors
`A`, `B`, and `C`, a local connected support `S` first forms
`exp(A_S) @ exp(B_S) @ exp(C_S)`. Lower connected residuals are subtracted
before the result is assembled into one representation. The implementation
must not materialize three independently truncated full-lattice layers and
multiply those layers as its primary cluster algorithm.

## 2. Canonical conceptual pipeline

Every operator-construction workflow should be explainable as:

```text
operator specification
        ↓
compiled topology / basis
        ↓
selected construction algorithm and order
        ↓
active or semantic result
        ↓
explicit materialization / compression / execution
```

The terms have the following meaning:

1. **Specification** describes local operators, supports, coefficients,
   physical space, lattice/graph geometry, and factor order.
2. **Basis/plan** caches only value-independent structure: site maps, cluster
   shapes, channels, history tables, and embedding maps.
3. **Algorithm** selects `exp`, a connected cluster expansion, or an ordered
   product expansion. It also owns order, rank, cutoff, and autodiff policy.
4. **Result** retains the representation-specific metadata needed by the
   algorithm: `FirstDegreeMPO` for MPO histories and `ActivePEPOBlocks` for
   sparse PEPO sectors.
5. **Materialization** crosses into an ordinary Quimb MPO/PEPO or generic
   tensor network. This boundary must be explicit because compression can
   invalidate semantic history metadata.

## 3. Public API decisions

### 3.1 Vocabulary to standardize

New documentation and examples should use:

- `exp(step, ...)` for the mathematical operator `exp(step * H)`.
- `compile_exp(...)` for reusable value-independent execution structure.
- `step` for the scalar multiplying the operator. Real time is conventionally
  `-1j * time`; imaginary time is a negative real step.
- `order` for the selected Taylor/history or local cluster cutoff, with the
  meaning stated by the owning algorithm.
- `max_bond` for a construction-time/local rank guard.
- `chi` for final numerical MPO/PEPO compression.
- `to_mpo()`, `to_pepo()`, or `to_tensor_network()` for explicit conversion.
- `report` objects for diagnostics, never for hidden global error claims.

The compatibility names `time_evolution`, `evolution_mpo`,
`compile_evolution`, `evaluate`, and `CompiledMPOEvolution` remain supported
but should not appear in new examples.

### 3.2 Keep separate canonical entry points

Do not force all algorithms into one universal `Operator` class yet. The
canonical entry points should remain:

```python
# 1D or snake-ordered higher-order MPO
basis = MPOBasis.from_local_terms(...)
U = basis.exp(step, order=4, mode="optimal")

# Fixed-channel, differentiable square-lattice PEPO
basis = PauliPEPOBasis.compile(..., order=4)
active = basis.exp(step, coefficients=theta)

# Dense finite-model connected PEPO expansion
plan = ClusterExpansionPlan(..., order=4)
active = plan.build(beta=beta, materialize=False)

# Ordered product, with factors in algebraic order
product = PEPOClusterProductExpansion.from_bases((A, B, C), ...)
U = product.compile_exp().exp(step)
```

The common surface is the vocabulary and lifecycle, not identical constructor
signatures. A caller should be able to choose the representation and
algorithm before choosing a backend or compression policy.

### 3.3 Explicitly expose the three different orders

The APIs and reports should distinguish:

- `history_order` — higher-order MPO virtual-history/Taylor construction;
- `cluster_order` — largest connected spatial cluster retained;
- `factor_count` — number of ordered factors in a product such as A, B, C.

Existing `order` arguments remain compatible. New result metadata and new
documentation should identify which meaning applies. A future deprecation
should only happen after callers can inspect the algorithm-specific name.

## 4. Target package structure

The current public namespace `pepsy.operators` is correct. The implementation
files are the part that needs eventual decomposition:

```text
pepsy/operators/
    __init__.py              # small lazy public facade
    terms.py                 # shared support/coefficient term normalization
    spaces.py                # physical dimensions, charges, braiding
    gates.py                 # elementary gates and gate application
    hamiltonians.py          # model-to-MPO helpers
    mpo/
        semantic.py          # FirstDegreeMPO and exact semantic algebra
        exponential.py       # higher-order history algorithms
        basis.py             # MPOBasis and compiled value binding
        cluster.py           # interval and graph MPO cluster expansions
    pepo/
        active.py            # ActivePEPOBlocks and graph active blocks
        basis.py             # PauliPEPOBasis and compiled evaluation
        cluster.py           # dense connected-cluster plans
        product.py            # ordered PEPO factor products
    pauli.py                 # PauliMPO and Pauli decomposition/algebra
    automaton.py             # MPO channel/transition automata
```

This is a target structure, not an immediate file-move instruction. During
the transition, the existing modules can remain the implementation owners
while smaller private modules are extracted behind them. The public imports
must continue to resolve from `pepsy.operators`.

The first public extraction seams now exist as lightweight family facades:
`operators.mpo_higher_order`, `operators.mpo_cluster`, and
`operators.pepo_cluster`. The large `mpo.py` and `cluster.py` files remain
compatibility implementation facades until the next extraction slice.

### Ownership rules

- `terms.py` owns parsing and validation shared by MPO and PEPO inputs.
- `spaces.py` owns physical-space metadata, not exponential algorithms.
- `mpo/semantic.py` owns exact operator algebra and history metadata.
- `mpo/exponential.py` owns Taylor/history construction and its compression
  policies.
- `mpo/cluster.py` owns connected residuals on chain/graph supports.
- `pepo/cluster.py` owns spatial residual factorization and graph PEPOs.
- `pepo/product.py` owns joint local products; it must not multiply separate
  full-lattice factor PEPOs as its primary algorithm.
- `pauli.py` owns Pauli-basis algebra; it may provide adapters to MPO/PEPO
  builders but should not become a second exponential framework.

## 5. Documentation structure

Use three documentation layers consistently:

1. **API guides** in `docs/api/operators/` answer “which entry point do I
   call?” The unified [exponential guide](../../api/operators/exponentials.md)
   is the landing page.
2. **Module maps** in `docs/development/modules/` answer “where is this
   implemented and what invariant does it own?”
3. **Plans and notes** in `docs/development/plans/` and
   `docs/development/notes/` answer “why does it work this way and what is
   next?”

Every new operator feature should add or update:

- one row in the API decision table;
- one short module-map section naming the implementation owner;
- one focused test contract;
- one note if the choice is algorithmically non-obvious.

Avoid duplicating long algorithm explanations between the API guide and the
module map. The API guide should show usage and decisions; the module map
should explain invariants and data flow.

## 6. Roadmap

### Phase 0 — freeze the vocabulary and inventory

Status: complete.

- Keep `exp` and `compile_exp` as the only recommended exponential names.
- Record the distinction between history order, cluster order, and factor
  count.
- Mark every public symbol as canonical, compatibility, or internal.
- Add small end-to-end examples for MPO, fixed-channel PEPO, dense PEPO, and
  ordered products.
- Do not move files or change numerical behavior in this phase. **Completed:**
  the inventory, module map, and four smoke examples are now present.

### Phase 1 — stabilize contracts

Status: in progress.

- Define a small shared protocol for plans/bases: `exp`, `compile_exp`,
  `cache_info`, and cache-clearing methods where applicable.
- Normalize result conversion names and report fields.
- Make sign conventions and approximation semantics part of docstrings and
  metadata, including whether a report is local-factorization diagnostics or
  a global approximation estimate.
- Add public API tests for the canonical imports and compatibility tests for
  the old names.

**Completed first slice:** the public facade separates canonical and
compatibility export groups, and `tests/test_operator_api_contract.py` checks
owners, alias identity, and materialization lifecycle boundaries. The parser
audit found no safe universal term-normalization extraction: MPO parsing is
already shared internally, while PEPO parsing has intentionally different
geometry semantics.

**Completed second slice:** `pepsy.operators.diagnostics.OperatorReportInfo`
and the `.api_info` property provide one summary vocabulary for MPO/PEPO
reports without changing their constructors or detailed fields. The
diagnostics module is the first extracted implementation seam behind the
existing operator facade.

**Completed third slice:** the joint ordered PEPO product implementation now
lives in `operators.pepo_product`. `operators.cluster` re-exports the classes
for compatibility, while `operators.pepo_cluster` is the clean public family
facade. Numerical behavior and the existing `exp(A) @ exp(B) @ exp(C)` tests
are unchanged.

### Phase 2 — extract implementation modules

Status: public facade slice started.

- Keep `operators.mpo_higher_order` limited to the SciPost-style history
  construction and `operators.mpo_cluster` limited to connected MPO
  residuals, including joint ordered local products.
- Keep `operators.pepo_cluster` as the sole named owner for PEPO cluster
  geometry, fixed channels, dense residuals, and joint ordered products.
- Keep the joint ordered-product algorithm in `operators.pepo_product`; the
  facade should re-export it without duplicating the implementation.
- Preserve `operators.mpo` and `operators.cluster` as compatibility facades
  until implementation moves can be made without duplicate algorithms.

- Split `operators.mpo` by responsibility, preserving it as a compatibility
  implementation facade during the transition.
- Split `operators.cluster` into active-block, basis, dense-cluster, and
  product modules.
- Move shared parsing/space logic only after duplicate implementations have
  been removed and focused tests cover the boundary cases.
- Keep lazy imports so optional Torch/JAX/Symmray dependencies do not leak
  into ordinary `import pepsy` or `import pepsy.operators` paths.

### Phase 3 — unify diagnostics and examples

- Give MPO and PEPO reports a common vocabulary for topology, retained rank,
  compression, backend, and approximation mode while retaining
  representation-specific fields.
- Add a small comparison matrix covering exact dense results, MPO, PEPO,
  cluster, and ordered-product results on the same two- and three-site models.
- Put reusable examples under `pepsy/examples/operators/`; keep notebooks in
  sibling example repositories focused on experiments and visualization.

### Phase 4 — only then consider API additions

Candidate additions, in priority order:

1. A shared immutable operator-term specification usable by both MPO and PEPO
   compilers.
2. A common `BuildConfig` only if repeated option normalization is proven to
   be confusing; do not create a large configuration object preemptively.
3. Native symmetry-aware higher-order histories for graded/fermionic MPOs.
4. Environment-aware compression hooks that consume BP/PEPS information,
   while keeping BP itself outside the local operator builders.

The first two are convenience layers. The last two are algorithmic work and
must not be mixed into an API-renaming effort.

## 7. Decisions to preserve

- Compiled objects cache structure, never parameter-dependent tensor values or
  autodiff graphs.
- `FirstDegreeMPO` and `ActivePEPOBlocks` are semantic/active representations;
  Quimb objects are explicit execution/interchange boundaries.
- Numerical compression is separate from analytical history or residual
  compression and reports when semantic metadata is invalidated.
- Ordered products are joint local constructions. They are not silently
  rewritten as `exp(sum)` or as multiplication of independent full-lattice
  approximations.
- Graph cluster geometry is separate from snake-chain geometry.
- BP loop expansions correct contractions/environments; operator cluster
  expansions construct local exponential residuals.
- Top-level `pepsy` exports remain a compatibility facade. New advanced
  symbols should be added to the owning namespace first.

## 8. Immediate next actions

1. Review this plan against the current public API and agree on the canonical
   names.
2. Add a canonical/compatibility/internal inventory to the operator module
   map.
3. Build four minimal examples and run the focused public API, MPO, and PEPO
   tests before any source move.
4. Extract only one low-risk shared component first, preferably term/space
   normalization, and measure import-boundary effects.
5. Revisit the target module tree after that extraction; do not perform a
   broad mechanical split while numerical work is still landing.
