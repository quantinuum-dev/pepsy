# Higher-order MPO foundation

`pepsy.operators.FirstDegreeMPO` is the semantic layer for the efficient
higher-order MPO construction used in SciPost Phys. 17, 135. It keeps virtual
level histories separate from the compiled Quimb MPO tensors.

## Why this is a separate API

An ordinary Quimb `MatrixProductOperator` is the right object for contraction
and for applying an MPO to an MPS, but it does not retain the paper's virtual
level histories. Those histories are needed to distinguish identity rails,
active operator channels, connected products, and exact equivalent-column
compressions. `FirstDegreeMPO` therefore acts as the semantic construction
layer, while `to_mpo()` remains the compatibility boundary for existing Quimb
and Pepsy code.

The API is intentionally explicit about exact versus numerical work:

- `add`, `scale`, products, commutators, and powers preserve the full operator
  exactly;
- `compress_exact()` only performs history-guided scalar gauge eliminations
  whose operator rows or columns are exactly equal;
- no cutoff is silently applied;
- `extensive_exponential()` exposes the paper's analytical Taylor rewiring,
  while numerical compression remains an explicit Quimb post-processing step;
- `compress_fixed_rank()` provides a fixed-rank TT-SVD path for autodiff,
  while keeping the history-aware and cutoff-based paths separate;
- native Symmray compilation remains separate without changing the ordinary
  MPO contract.

This separation keeps the first implementation compatible with current Pepsy
MPO/MPS workflows and makes the algebra testable against dense operators before
adding backend-specific optimizations.

```python
import numpy as np
from pepsy.operators import FirstDegreeMPO, MPOProductTerm

x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

H = FirstDegreeMPO.from_local_terms(
    4,
    [
        MPOProductTerm((0, 1), (x, x)),
        MPOProductTerm((1, 2), (z, z)),
    ],
)

H2 = H.non_disjoint_product(H)
H2_exact = H2.compress_exact()
mpo = H2_exact.to_mpo()
print(H2_exact.compression_report)
```

For optimization loops, use `MPOBasis` to compile the term topology once and
bind fresh scalar coefficients on every evaluation. `MPOParameter` references
can be named (mapping input) or positional (sequence input):

```python
import torch
from pepsy.operators import MPOBasis, MPOParameter

basis = MPOBasis.from_pauli_terms(
    8,
    [
        ((0, 1), "XX", MPOParameter("J")),
        ((0, 7), "ZZ", MPOParameter("V")),
    ],
)

J = torch.tensor(0.7, requires_grad=True)
V = torch.tensor(0.2, requires_grad=True)
U = basis.evolution_mpo(
    {"J": J, "V": V},
    dt=0.01,
    order=2,
    mode="optimal",
)
```

`evolution_mpo()` and `time_evolution()` use the real-time convention
`exp(-1j * dt * H(parameters))`; use `extensive_exponential()` directly when
an imaginary-time or otherwise signed scalar step is required.

`basis.cache_info` reports the compiled topology, number of bindings, and raw
history orders already generated. The cache intentionally stores structure,
not completed MPO values: caching a Torch/JAX result would risk stale
optimizer values and retain an obsolete autodiff graph. Raw history topology
is cached by Taylor order because it depends only on channels and reachability;
value-dependent Algorithm 2 merges are still recomputed for each build.
`MPOBasis` shares exact prefixes and suffix continuations while retaining
term-specific coefficient slots on a path edge, so its output is directly
compatible with `extensive_exponential()`.

For one-off large-order constructions, disable persistent history caching:

```python
U = basis.evolution_mpo(
    {"J": 0.7, "V": 0.2},
    dt=0.01,
    order=4,
    mode="optimal",
    cache_history=False,
)
```

This keeps the current MPO tensors materialized, but releases the reusable
raw topology after the build. It is useful when orders are not repeated and
the process should not retain all history tables.

For optimizers that already evaluate parameters in a backend batch, pass that
batch directly and avoid per-term coefficient resolution:

```python
coefficients = torch.stack((J, V))
H = basis.build(coefficients=coefficients)
U = basis.evolution_mpo(coefficients=coefficients, dt=0.01, order=2)
```

The current layer supports exact finite open-boundary construction, addition,
scaling, products, commutators, powers, and the paper's conservative exact
history/column compression. `to_mpo()` returns an ordinary Quimb
`MatrixProductOperator` and performs no numerical truncation. The semantic
object remains the source of truth for history-aware operations; the Quimb
object is the execution and interoperability boundary.

The tensor-network-aware time-evolution path is available through
`extensive_exponential(dt, order=N)`. For chains with at least two sites it
walks only raw history channels reachable from the finite all-one left
boundary, applies the paper's Algorithms 1 and 2 as virtual-bond rewiring
operations, and only then contracts the finite all-one boundary vectors.
Every operation is local to MPO tensors; no global dense Hamiltonian or matrix
exponential is formed. The reachable history space can still grow
exponentially with `N`, but it avoids allocating disconnected boundary and
edge channels that cannot contribute to the finite-chain operator.
The one-site path evaluates the direct local Taylor polynomial at arbitrary
positive order. In `mode="optimal"` or with `extend=True`, it includes one
additional local Taylor term; a one-site chain has no non-trivial virtual
history for Algorithm 3 or 4 to rewire.

```python
import quimb.tensor as qtn

U1 = H.extensive_exponential(dt=0.01, order=1)
U2 = H.extensive_exponential(dt=0.01, order=2)
U3 = H.extensive_exponential(dt=0.01, order=3)

# One-site Hamiltonians use the same direct Taylor API at arbitrary order.
single_site = FirstDegreeMPO.from_pauli_terms(1, [((0,), "Z")])
U5 = single_site.extensive_exponential(dt=0.01, order=5)

# Optional paper extensions, kept explicit in the API.
U2_extended = H.extensive_exponential(dt=0.01, order=2, extend=True)
U2_approx = H.extensive_exponential(dt=0.01, order=2, approximate=True)
U2_optimal = H.extensive_exponential(
    dt=0.01,
    order=2,
    mode="optimal",
    max_bond=128,
)

# Quimb remains the compiled interchange boundary.
mpo = U2.to_mpo()

# Tensor-network application and expectation evaluation.
state = qtn.MPS_computational_state("0000")
state_next = U2.apply_to_mps(state, method="direct", cutoff=1e-10)
energy_factor = U2.expectation(state)
```

`extend=True` applies Algorithm 3: selected order `N + 1` local histories are
generated directly from the existing order-`N` transitions and added without
increasing the exact-history bond dimension. The implementation does not
materialize a complete order-`N + 1` MPO just to extract those transitions.
`approximate=True`
applies Algorithm 4 after exact compression. It is an order-controlled
analytical approximation, not a numerical SVD cutoff, and is therefore exposed
as a separate opt-in flag. Numerical Quimb compression remains a separate
post-processing step.

The named `mode` policies are `"base"` (Algorithms 1--2), `"optimal"`
(Algorithms 1--3), and `"approximate"` (Algorithms 1--4). The word
`optimal` refers to the paper's exact extension/compression construction, not
to a globally minimum-bond MPO. `max_bond` limits the temporary history bonds
before exact compression; `on_exceed="raise"` stops safely, while
`on_exceed="warn"` continues with a warning.

History storage is controlled independently with
`history_storage="auto"|"dense"|"sparse"|"streaming"`. The default uses the
cached dense topology for repeated calls, and automatically switches to the
streaming two-cut builder when `cache_history=False`. `"sparse"` also avoids
evaluating structurally impossible local transition products. In all modes the
final local MPO tensors are materialized because Algorithms 1--4 rewrite them;
the streaming mode requires `cache_history=False` and guarantees that earlier
history cuts and dead local products are not retained during construction.
Metadata reports the selected mode and
`history_storage_blocks` gives the structurally stored versus considered local
block counts.

`product(kind=...)` and `disjoint_product` currently label provenance only;
they do not perform support-overlap analysis. See the [development module
map](../../development/modules/higher_order_mpo.md) for the execution order,
design rationale, and prioritized future improvements.

`apply_to_mps` delegates to Quimb's MPO–MPS compression methods, while
`expectation` delegates to Pepsy's normalized MPS contraction helper. These
methods are the intended validation and execution path for larger systems;
dense conversion is only appropriate as a small-system regression oracle.

For direct spin-chain input, `from_pauli_terms()` accepts compact labels and
automatically shares repeated exact prefixes and suffix continuations:

```python
H = FirstDegreeMPO.from_pauli_terms(
    12,
    [
        ((0, 5, 11), "ZXY", 0.7),
        ((0, 5, 11), "ZXZ", 0.3),
    ],
)
```

Sites are zero-based and list the non-identity support; omitted sites receive
identity operators. The sharing is exact and structural, not an SVD or a
numerical approximation. Use `share_channels=False` to retain one dedicated
channel path per term when debugging the unshared representation. The
ordinary `from_local_terms()` constructor accepts the same Pauli strings in
`MPOProductTerm` or mapping inputs.

Coefficients may be scalar tensors from an Autoray-compatible autodiff
backend. Static Pauli matrices are promoted to that backend, so the
coefficient remains in the computation graph. The same constructor can be
used for a parameterized Hamiltonian or observable; pass a backend scalar for
`dt` as well when constructing an evolution MPO:

```python
import torch

theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
H_theta = FirstDegreeMPO.from_pauli_terms(
    8,
    [((0, 7), "ZX", theta)],
)
U_theta = H_theta.extensive_exponential(
    -1j * time,
    order=2,
    mode="optimal",
)

# The compiled MPO still contains Torch tensors connected to theta and time.
loss = U_theta.to_mpo().to_dense().real.sum()
loss.backward()
print(theta.grad, time.grad)
```

For an observable, use the same parameterized term input and call
`observable.expectation(state)`. A NumPy product state is automatically
promoted to the observable backend when needed; for larger differentiable
workloads, keep the state and contraction backend consistent.

For an explicit numerical MPO compression policy, use
`compress_numerical(max_bond=..., cutoff=..., cutoff_mode=...)`. This delegates
the SVD/QR sweep to Quimb, returns a compiled Quimb MPO, and can return an
`MPONumericalCompressionReport` with the requested policy and bond dimensions
before and after compression. Numerical truncation clears the attached
`pepsy_first_degree` semantic object because its original history table no
longer describes the compressed MPO. Set `estimate_error=True` to contract the
operator-level Frobenius norm of `MPO_before - MPO_after` without densifying
the operators; the report exposes both
`operator_frobenius_error` and
`operator_frobenius_relative_error`. This is an explicit contraction because
it can be expensive for large MPOs.

For differentiable numerical compression with a fixed rank, use
`compress_fixed_rank(max_bond=...)`. It performs a TT-SVD sweep with a rank
selected from matrix dimensions and the fixed cap, so it does not branch on a
singular-value cutoff. The Torch path automatically uses Pepsy's regularized
SVD VJP for zero or repeated singular values. The returned
`MPODifferentiableCompressionReport` records the bond dimensions and the
`FirstDegreeMPO` has `metadata["history_valid"] == False`, because numerical
compression changes the analytical history representation.

The current implementation targets finite open-boundary NumPy, Torch, and JAX
Autoray-compatible local blocks, including the optimal extension path under
Torch/JAX autodiff. A small accuracy regression benchmark compares orders
one--three with a first-order Trotter product and a finite two-site cluster
expansion; see `tests/test_mpo_benchmarks.py`. Native fermionic/Symmray
compilation and infinite/unit-cell MPOs remain separate future work.
