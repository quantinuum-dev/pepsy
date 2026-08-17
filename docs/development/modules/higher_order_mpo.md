# Higher-order MPO

`pepsy.operators.mpo` contains the semantic MPO layer for the size-extensive
higher-order construction in SciPost Phys. 17, 135. The implementation keeps
the paper's virtual-level histories alongside ordinary local MPO tensors.

## Public API contract

| API | Responsibility | Accuracy and ownership |
| --- | --- | --- |
| `MPOProductTerm` | Describe a factorized local product term | Input metadata only; `charge` is preserved but not interpreted |
| `FirstDegreeMPO.from_local_terms` | Build a first-degree Hamiltonian-like MPO | Local automaton construction; no dense operator |
| `FirstDegreeMPO.product`, `power`, `commutator` | Exact semantic algebra | Returns new objects and retains all virtual paths |
| `FirstDegreeMPO.extensive_exponential` | Apply the paper's Algorithms 1--4 | Local tensor construction; `approximate=True` is explicit |
| `FirstDegreeMPO.compress_exact` | Remove provably equivalent history channels | Exact scalar gauge elimination only; optional explicit in-place mutation |
| `FirstDegreeMPO.to_mpo` | Interoperate with Quimb | No compression; returns an ordinary `MatrixProductOperator` |
| `apply_to_mps`, `expectation` | Execute through tensor-network consumers | Delegates to Quimb/Pepsy contraction APIs and does not densify |

`product(kind=...)` uses `kind` as provenance metadata. In particular,
`disjoint_product` does not currently prove that supports are disjoint or
perform connected-term filtering. Keeping that distinction explicit avoids
making the foundational algebra silently depend on a future overlap analysis.

The semantic object is the source of truth for history-aware work. The Quimb
MPO returned by `to_mpo()` is the execution/interchange representation and
has a semantic copy attached as `pepsy_first_degree`.

## Multi-site execution order

For a chain with at least two sites, `extensive_exponential` uses one generic
history route for every positive `order`:

1. Build the Cartesian product of local virtual channels for the requested
   history order.
2. Optionally add the selected next-order transitions from Algorithm 3.
3. Apply Algorithm 1's factorial-weighted extensive prefactor rewiring.
4. Apply Algorithm 2's exact row/column history compression.
5. Optionally apply Algorithm 4's order-controlled analytical approximation.
6. Contract the all-one boundary histories only after the rewiring passes.

The order matters. Algorithms 1--4 need both open boundary history tables;
contracting the finite-chain boundary vectors earlier would discard the
channels they operate on. The final boundary contraction removes those
temporary edge states and leaves an ordinary open-boundary MPO.

The current Cartesian history table is a correctness-first reference
implementation. Its temporary bond dimension grows exponentially with the
Taylor order and local MPO bond dimension, although no global dense matrix is
formed. Exact history compression reduces redundant channels after the table
is built; it does not change the worst-case allocation of the initial table.

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
- Numerical SVD/truncation is not hidden inside this API. Callers can compile
  with `to_mpo()` and then select Quimb's numerical compression policy.
- Fermionic/Symmray compilation is not enabled by this workstream. The
  `charge` and `string_operators` fields preserve construction metadata for a
  future native backend without claiming that it is already supported.
- Public algebraic operations are non-mutating. Mutation is available only
  through the named `inplace=True` compression option.

## Future implementation improvements

The safest order for scaling the implementation is:

1. Replace Cartesian history allocation with a reachable-history iterator or
   sparse channel map, while preserving the current `MPOLevel.history` data.
2. Add backend-aware batched local products and structural equality/fingerprints
   so exact compression does not repeatedly materialize backend arrays on the
   host.
3. Generate Algorithm 3 transitions directly instead of allocating a complete
   order-plus-one reference table.
4. Add a separate numerical compression policy with explicit cutoff, maximum
   bond dimension, and error reporting; keep it separate from Algorithm 4.
5. Add native charge-aware/Symmray compilation and fermionic parity handling
   behind the existing operator/backend namespaces.
6. Generalize the one-site boundary convention to arbitrary order and promote
   that path to the same execution/reporting contract.

These are implementation layers, not reasons to widen the current public API.
The existing semantic and Quimb boundaries should remain stable while they are
added.
