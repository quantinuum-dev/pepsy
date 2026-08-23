# Higher-order MPO

`pepsy.operators.mpo_higher_order` is the public family facade for the
size-extensive higher-order construction in SciPost Phys. 17, 135. The
implementation keeps the paper's virtual-level histories alongside ordinary
local MPO tensors. The semantic history implementation remains in
`pepsy.operators.mpo`; parameterized bases and compiled evaluators are
implemented in `pepsy.operators.mpo_basis`.

This page covers only the paper-style higher-order MPO exponential. Connected
spatial MPO clusters and ordered products of several exponentials belong to
`pepsy.operators.mpo_product`, where `cluster_size` and `factor_count` have
different meanings from history `order`. The historical
`pepsy.operators.mpo_cluster` path is a compatibility facade.

## Public API contract

| API | Responsibility | Accuracy and ownership |
| --- | --- | --- |
| `MPOBasis` / `MPOParameter` | Reuse a parameterized term topology | Structural cache only; each bind creates fresh backend-connected local blocks |
| `MPOBasis.compile_exp` / `CompiledMPOExp` | Reuse coefficient-slot and higher-order execution plans | Value-only evaluator; static banks are cached, coefficient-dependent arrays and autodiff graphs are fresh |
| `MPOProductTerm` | Describe a factorized local product term | Matrix operators or compact Pauli labels; `charge` labels active virtual sectors when symmetry is configured |
| `MPOLocalOperatorTerm` | Describe an arbitrary multi-site local matrix | Exact fixed-rank operator-Schmidt decomposition; coefficient remains a separate differentiable slot |
| `MPOPhysicalSpace` / `MPOBraiding` | Carry local sectors and exchange semantics | Backend-neutral immutable metadata; explicit odd-factor parities determine graded sorting signs |
| `FirstDegreeMPO.from_local_terms` / `.from_pauli_terms` | Build a first-degree Hamiltonian-like MPO | Exact local automaton construction with optional channel sharing; no dense operator |
| `FirstDegreeMPO.product`, `power`, `commutator` | Exact semantic algebra | Returns new objects and retains all virtual paths |
| `FirstDegreeMPO.extensive_exponential` | Apply the paper's Algorithms 1--4 | Local tensor construction; direct Algorithm 3; named `mode` and temporary `max_bond` guard |
| `MPOBasis.compile_cluster_expansion` / `compile_graph_cluster_expansion` | Compatibility adapters to the MPO cluster family | Delegate to `operators.mpo_product`; they do not turn a spatial cluster order into a history order |
| `FirstDegreeMPO.exp` / `MPOBasis.exp` | Build `exp(step * H)` with an explicit scalar step | `chi` is a post-construction MPO cap; real-time uses `step=-1j * tau`; `differentiable=True` selects fixed-rank TT-SVD |
| `FirstDegreeMPO.clear_history_cache` / `MPOBasis.clear_history_cache` | Release reusable higher-order plans | Keeps current tensors and compiled first-degree topology unchanged |
| `FirstDegreeMPO.compress_exact` | Remove provably equivalent history channels | Exact scalar gauge elimination only; optional explicit in-place mutation |
| `FirstDegreeMPO.compress_fixed_rank` | Differentiable numerical compression | Fixed-rank TT-SVD; no value-dependent cutoff, semantic histories are cleared |
| `FirstDegreeMPO.to_mpo` | Interoperate with Quimb | No compression; returns an open-boundary `MatrixProductOperator`, optionally backed by native Symmray blocks |
| `FirstDegreeMPO.compress_numerical` | Apply explicit numerical policy | Delegates SVD/QR to Quimb and returns a separate truncation report |
| `apply_to_mps`, `expectation` | Execute through tensor-network consumers | Delegates to Quimb/Pepsy contraction APIs and does not densify |

`product(kind=...)` uses `kind` as provenance metadata. In particular,
`disjoint_product` does not currently prove that supports are disjoint or
perform connected-term filtering. Keeping that distinction explicit avoids
making the foundational algebra silently depend on a future overlap analysis.

The semantic object is the source of truth for history-aware work. The Quimb
MPO returned by `to_mpo()` is the execution/interchange representation and
has a semantic copy attached as `pepsy_first_degree`.

`from_pauli_terms()` is the compact spin-chain input boundary. A term such as
`((0, 5, 11), "ZXY", coefficient)` places the three non-identity factors at
those zero-based sites and inserts identities in the gaps. The shared
automaton builder uses a prefix trie followed by exact identical-future
merges, so repeated term prefixes and suffixes reuse virtual channels. This
is a structural factorization, not a claim of globally minimal MPO rank;
`share_channels=False` retains the dedicated-channel construction for
diagnostics.

`MPOBasis` stores static channel structure separately from coefficient values.
The shared automaton retains exact prefixes and suffix continuations, while a
term-specific path edge receives the coefficient slot; identical terms share
one slot and sum their coefficients. This keeps rebinding exact even when
different term paths merge before their common suffix. Coefficients can be
resolved individually from `MPOParameter` objects or supplied as one
one-dimensional backend batch through `build(coefficients=...)`. Static local
Pauli matrices and identity rails are promoted to that backend, so term
parameters remain differentiable through the history construction and Quimb
adapter. Backend arrays with functional update semantics, such as JAX arrays,
use replacement updates in Algorithm 3 rather than in-place mutation.

`MPOBasis` is the optimization-facing cache around this input boundary. It
compiles one shared identity-rail automaton and coefficient-slot map, then
assembles only the current slot values in `build(parameters)` or
`build(coefficients=...)`. It deliberately does not cache completed
parameter-value MPOs, because tensor identities are not safe value cache keys
after in-place optimizer updates. Each `FirstDegreeMPO` also caches its raw
reachable history topology and local gather/index execution plan by order.
The symbolic merge schedule is reused, while scalar weights and backend
arrays are rebuilt for each call so parameter and time autodiff graphs remain
fresh.

`compile_exp()` adds the value-only execution boundary for repeated
optimization steps. For ordinary static local operators it compiles each
site into an affine bias/operator bank and evaluates that bank with one
backend contraction per site. Backend-native operators and oversized banks
retain the grouped scatter fallback. This avoids rebuilding the semantic
automaton and first-degree wrapper per call. Algorithms 1--4 then use the
selected dense, structural-sparse, or persistent block-sparse history policy.
`compile_evolution()` and `CompiledMPOEvolution` remain compatibility names for
older callers.

## Multi-site execution order

For a chain with at least two sites, `extensive_exponential` uses one generic
history route for every positive `order`:

1. Walk local virtual transitions from the all-one left boundary and build
   only reachable history channels for the requested order.
2. Optionally add the selected next-order transitions from Algorithm 3.
3. Apply Algorithm 1's factorial-weighted extensive prefactor rewiring.
4. Apply Algorithm 2's exact row/column history compression.
5. Optionally apply Algorithm 4's order-controlled analytical approximation.
6. Contract the all-one boundary histories only after the rewiring passes.

Named policies map to these passes as follows: `mode="base"` selects
Algorithms 1--2, `mode="algorithm4"` selects Algorithms 1, 2, and 4,
`mode="optimal"` selects Algorithms 1--3, and `mode="approximate"` selects
Algorithms 1--4. The fast `algorithm4` policy intentionally omits Algorithm
3's selected next-order replay; `mode="approximate"` retains it. The
`max_bond` guard is checked
while the temporary history table is generated, before exact compression can
remove channels; `on_exceed="raise"` is the safe default.

The order matters. Algorithms 1--4 need both open boundary history tables;
contracting the finite-chain boundary vectors earlier would discard the
channels they operate on. The final boundary contraction removes those
temporary edge states and leaves an ordinary open-boundary MPO.

The reachable-history table can still grow exponentially with the Taylor order
and local MPO bond dimension, although no global dense matrix is formed. The
raw topology and local gather metadata are reusable across parameter bindings
and time steps. History block products are gathered in backend batches rather
than dispatched once per virtual pair. `history_storage="sparse"` skips
structurally invalid local transition products before scattering into dense
virtual tensors. `history_storage="block_sparse"` stores structurally present
operator-valued virtual entries and applies Algorithms 1--4 as sparse row and
column transforms. Symmetry-configured calls select this path automatically;
ordinary cached automaton calls retain the existing structural-sparse default,
and ordinary `cache_history=False` calls retain the compatibility streaming
default unless block-sparse storage is requested explicitly.

The explicit `history_storage="reduced"` executor compiles Algorithms 1--2 as
raw-axis maps, folds Algorithm 3 into those maps, and scatters numerical local
products directly into final sparse virtual tensors. Its cached plan contains
only integer indices and scalar combinatorics. It therefore avoids raw virtual
tensor materialization without retaining backend values or autodiff graphs.

The private `SparseVirtualTensor` representation is not a public tensor API.
It is an execution detail behind `FirstDegreeMPO`: reading `arrays` materializes
dense virtual tensors, while `to_mpo()` directly groups charge-compatible
entries into Symmray sectors. Virtual history charges are recursively combined
under MPO products, so Algorithms 1--4 may permute and merge histories without
losing the total Abelian sector. Compilation validates
`-q_left + q_right + q_upper - q_lower = 0` for every nonzero local block.

The one-site path now evaluates an arbitrary-order local Taylor polynomial.
With `extend=True` or `mode="optimal"`, it evaluates one additional local
Taylor term; there are no non-trivial virtual channels for Algorithm 3 or 4 to
rewire on a one-site chain.

The final MPO bond cap is intentionally separate from the temporary history
guard. `max_bond` limits raw history growth before Algorithms 1--4, while
`chi` is passed to `compress_to_bond` after the analytical construction. The
ordinary Quimb compression path returns a compiled Quimb MPO because numerical
truncation invalidates the semantic histories. The fixed-rank path retains a
semantic wrapper for autodiff but marks `history_valid=False`.

## Deliberate implementation decisions

- Local `Autoray` operations preserve backend, dtype, and device behavior.
- Explicit local loops retain symbolic history and are preferred over generic
  Quimb multiplication in the semantic layer.
- Exact history compression checks both the candidate history and the
  corresponding operator-valued row or column. A history match alone is not
  sufficient for an exact merge.
- Numerical SVD/truncation is explicit through `compress_numerical` and is
  delegated to Quimb. With `estimate_error=True`, the returned report also
  contracts an operator-level Frobenius error through the MPO difference and
  reports its relative value; the default avoids that extra contraction.
- `compress_fixed_rank` uses a backend SVD with a fixed rank cap. Its Torch
  route scopes Pepsy's regularized SVD policy so rank-deficient blocks retain
  finite reverse-mode derivatives; it intentionally clears semantic history
  validity after the numerical sweep.
- Finite Torch and JAX paths use functional tensor updates, so Algorithm 3
  remains compatible with reverse-mode autodiff and JIT tracing. Native
  Symmray compilation supports neutral bosonic `U1`, `Z2`, `U1U1`, and
  `Z2Z2` MPOs with NumPy local blocks. Symmetry names are normalized
  case-insensitively, and `physical_charges` accepts either one charge per
  basis state or an insertion-ordered charge-to-multiplicity mapping.
  `MPOBraiding` already supplies explicit canonicalization signs for odd
  factors; graded fermionic higher-order execution remains disabled until the
  history engine itself carries that sign protocol natively.
- Public algebraic operations are non-mutating. Mutation is available only
  through the named `inplace=True` compression option.

## Future implementation improvements

The dense execution path now uses cached gather metadata, grouped coefficient
contractions, fused virtual transfer maps, raw tensor/batch interfaces for
JAX/Torch compilation, and an explicit direct-reduced history executor.
Native backend blocks for Torch/JAX Symmray, graded fermionic histories,
sector-aware Schmidt decomposition for charged general local matrices,
automatic inference of multi-stage term charges, and automatic storage-policy
crossover selection remain separate work. The maintained
small-system accuracy regression in `tests/test_mpo_benchmarks.py` compares
the finite MPO orders with first-order Trotter and a p=2 two-site cluster
baseline; it is deliberately not a timing harness.
The external timing/memory harness is maintained at
`../pepsy_examples/higher_order_mpo/benchmark.py`.

These are implementation layers, not reasons to widen the current public API.
The existing semantic and Quimb boundaries should remain stable while they are
added.
