# Higher-order MPO

`pepsy.operators.mpo` contains the semantic MPO layer for the size-extensive
higher-order construction in SciPost Phys. 17, 135. The implementation keeps
the paper's virtual-level histories alongside ordinary local MPO tensors.

## Public API contract

| API | Responsibility | Accuracy and ownership |
| --- | --- | --- |
| `MPOBasis` / `MPOParameter` | Reuse a parameterized term topology | Structural cache only; each bind creates fresh backend-connected local blocks |
| `MPOProductTerm` | Describe a factorized local product term | Matrix operators or compact Pauli labels; `charge` is preserved but not interpreted |
| `FirstDegreeMPO.from_local_terms` / `.from_pauli_terms` | Build a first-degree Hamiltonian-like MPO | Exact local automaton construction with optional channel sharing; no dense operator |
| `FirstDegreeMPO.product`, `power`, `commutator` | Exact semantic algebra | Returns new objects and retains all virtual paths |
| `FirstDegreeMPO.extensive_exponential` | Apply the paper's Algorithms 1--4 | Local tensor construction; named `mode` and temporary `max_bond` guard |
| `FirstDegreeMPO.compress_exact` | Remove provably equivalent history channels | Exact scalar gauge elimination only; optional explicit in-place mutation |
| `FirstDegreeMPO.to_mpo` | Interoperate with Quimb | No compression; returns an open-boundary `MatrixProductOperator` |
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

Coefficients are deliberately stored on terminal weighted transitions. When
one is a Torch/JAX scalar, the static local Pauli matrices and identity rails
are promoted to that backend before stacking, so `dt` and term parameters
remain differentiable through the history construction and Quimb adapter.
The same path is used for parameterized observables. Backend arrays with
functional update semantics, such as JAX arrays, use replacement updates in
Algorithm 3 rather than in-place mutation.

`MPOBasis` is the optimization-facing cache around this input boundary. It
compiles one common identity-rail automaton with one coefficient slot per
term, then reweights only the slot transitions in `build(parameters)`. This
avoids rebuilding the graph for every optimizer iteration while leaving the
current backend scalar in the autodiff graph. It deliberately does not cache
completed parameter-value MPOs, because tensor identities are not safe value
cache keys after in-place optimizer updates.

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
Algorithms 1--2, `mode="optimal"` selects Algorithms 1--3, and
`mode="approximate"` selects Algorithms 1--4. The `max_bond` guard is checked
while the temporary history table is generated, before exact compression can
remove channels; `on_exceed="raise"` is the safe default.

The order matters. Algorithms 1--4 need both open boundary history tables;
contracting the finite-chain boundary vectors earlier would discard the
channels they operate on. The final boundary contraction removes those
temporary edge states and leaves an ordinary open-boundary MPO.

The current reachable-history table is still a correctness-first reference
implementation. Its temporary bond dimension can grow exponentially with the
Taylor order and local MPO bond dimension, although no global dense matrix is
formed. Exact history compression reduces redundant channels after the table
is built; reachability removes dead finite-boundary channels but does not change
the worst-case allocation for a fully reachable local graph.

The one-site path is intentionally limited to direct orders one and two. It
provides a simple boundary regression and avoids pretending that the generic
multi-site history convention already covers every one-site edge case.

## Deliberate implementation decisions

- Local `Autoray` operations preserve backend, dtype, and device behavior.
- Explicit local loops retain symbolic history and are preferred over generic
  Quimb multiplication in the semantic layer.
- Exact history compression checks both the candidate history and the
  corresponding operator-valued row or column. A history match alone is not
  sufficient for an exact merge.
- Numerical SVD/truncation is explicit through `compress_numerical` and is
  delegated to Quimb. The returned report records bond dimensions and policy,
  but does not invent a global operator error that Quimb does not provide.
- Fermionic/Symmray compilation is not enabled by this workstream. The
  `charge` and `string_operators` fields preserve construction metadata for a
  future native backend without claiming that it is already supported.
- Public algebraic operations are non-mutating. Mutation is available only
  through the named `inplace=True` compression option.

## Future implementation improvements

The remaining implementation improvements are:

1. Replace the reachable history lists with a sparse channel map or streaming
   local iterator for high orders, while preserving the current
   `MPOLevel.history` data.
2. Add backend-aware batched local products and structural equality/fingerprints
   so exact compression does not repeatedly materialize backend arrays on the
   host.
3. Generate Algorithm 3 transitions directly instead of allocating a complete
   order-plus-one reference table.
4. Generalize the one-site boundary convention to arbitrary order and promote
   that path to the same execution/reporting contract.

These are implementation layers, not reasons to widen the current public API.
The existing semantic and Quimb boundaries should remain stable while they are
added.
