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
| `FirstDegreeMPO.extensive_exponential` | Apply the paper's Algorithms 1--4 | Local tensor construction; direct Algorithm 3; named `mode` and temporary `max_bond` guard |
| `FirstDegreeMPO.compress_exact` | Remove provably equivalent history channels | Exact scalar gauge elimination only; optional explicit in-place mutation |
| `FirstDegreeMPO.compress_fixed_rank` | Differentiable numerical compression | Fixed-rank TT-SVD; no value-dependent cutoff, semantic histories are cleared |
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
reachable history topology by order; exact history merges remain
value-dependent and are not cached.

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

The reachable-history table can still grow exponentially with the Taylor order
and local MPO bond dimension, although no global dense matrix is formed. The
raw topology is reusable across parameter bindings and time steps, while exact
history compression remains a value-dependent pass after the cached table is
assembled. `history_storage="streaming"` (which requires
`cache_history=False`) retains only adjacent cuts while
building the current MPO, and `history_storage="sparse"` additionally skips
structurally invalid local transition products. `cache_history=False` selects
streaming automatically under the default `history_storage="auto"`; no
persistent topology or complete history table is retained after the call.
The final MPO tensors still must exist for the in-place Algorithm 1--4 passes.

The one-site path now evaluates an arbitrary-order local Taylor polynomial.
With `extend=True` or `mode="optimal"`, it evaluates one additional local
Taylor term; there are no non-trivial virtual channels for Algorithm 3 or 4 to
rewire on a one-site chain.

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
  remains compatible with reverse-mode autodiff and JIT tracing. Fermionic/
  Symmray compilation is not enabled by this workstream; `charge` and
  `string_operators` preserve construction metadata for a future native
  backend without claiming that it is already supported.
- Public algebraic operations are non-mutating. Mutation is available only
  through the named `inplace=True` compression option.

## Future implementation improvements

The remaining implementation improvements are native fermionic/Symmray
compilation for the shared topology and fixed-rank compression, plus larger
external timing studies. The maintained small-system accuracy regression in
`tests/test_mpo_benchmarks.py` compares the finite MPO orders with first-order
Trotter and a p=2 two-site cluster baseline; it is deliberately not a timing
harness.

These are implementation layers, not reasons to widen the current public API.
The existing semantic and Quimb boundaries should remain stable while they are
added.
