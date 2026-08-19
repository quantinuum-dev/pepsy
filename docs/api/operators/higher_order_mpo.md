# Higher-order MPO foundation

`pepsy.operators.FirstDegreeMPO` is the semantic layer for the efficient
higher-order MPO construction used in SciPost Phys. 17, 135. It keeps virtual
level histories separate from the compiled Quimb MPO tensors.

For the short decision guide and canonical MPO/PEPO examples, see the
[unified exponential API](exponentials.md). This page documents the detailed
history, mode, compression, and execution semantics.

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
- `exp()` exposes the operator-valued exponential with an explicit scalar
  step, while `extensive_exponential()` remains the paper-level construction;
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
U = basis.exp(
    -1j * 0.01,
    {"J": J, "V": V},
    order=2,
    mode="algorithm4",
)
```

`exp(step, parameters)` uses `step` as the actual scalar in `exp(step * H)`.
This makes the sign and complex convention explicit: use `step=-1j * tau` for
real time, a negative real `step` for imaginary time, or another scalar for a
custom exponential. `time_evolution` and `evolution_mpo` remain compatibility
aliases for older callers; new code should use `exp`.

The exponential entry points share the same analytical controls: `order`, `mode`,
`max_bond`, `on_exceed`, `cache_history`, and `history_storage`. Use the named
`mode` policies for new code; `extend=True` and `approximate=True` remain
backward-compatible flags on the lower-level construction API. `mode="base"`
is the conservative Algorithms 1--2 path, `mode="algorithm4"` is the fast
Algorithms 1, 2, and 4 path, `mode="optimal"` adds the exact Algorithm-3
extension, and `mode="approximate"` adds Algorithm 4 after that extension.
Legacy flag calls are normalized to these canonical mode names in metadata.

The final numerical bond cap is a separate stage from the temporary history
construction. Pass `chi` to `exp` when the returned MPO should be compressed
after the paper construction:

```python
U_chi, compression_report = basis.exp(
    -1j * 0.01,
    {"J": J, "V": V},
    order=3,
    mode="approximate",
    chi=64,
    cutoff=1.0e-10,
    return_report=True,
)
```

Here `chi` is the final MPO bond cap; it is different from the existing
`max_bond` argument, which only guards the temporary raw-history bonds before
analytical compression. The default `chi` path delegates the numerical sweep
to Quimb and therefore returns an ordinary Quimb `MatrixProductOperator`.
Numerical compression cannot retain the original paper-history metadata.

For parameter optimization, use a fixed-rank differentiable compression:

```python
U_chi, compression_report = basis.exp(
    -1j * 0.01,
    {"J": J, "V": V},
    order=3,
    mode="optimal",
    chi=64,
    differentiable=True,
    return_report=True,
)
```

This returns a `FirstDegreeMPO` with fixed-rank TT-SVD tensors and invalidated
analytical histories. The topology, history, and rewiring plans remain cached;
the parameter-dependent tensor values and compression factors are evaluated on
each call so Torch/JAX autodiff graphs cannot become stale.

The parameterized cache is shared by `exp()` and `compile_exp()`. It caches
structure, not completed parameter-value MPOs. Call
`basis.clear_history_cache()` when a long-running
optimization no longer needs the cached history orders; this leaves the
compiled term topology intact.

`basis.cache_info` reports the compiled topology, number of bindings, and raw
history orders already generated. The cache intentionally stores structure,
not completed MPO values: caching a Torch/JAX result would risk stale
optimizer values and retain an obsolete autodiff graph. Raw history topology
is cached by Taylor order because it depends only on channels and reachability;
the local gather/index plan is cached alongside it. Algorithms 1--4 still
evaluate scalar weights during each numerical pass, so `step` and all
coefficient tensors remain in the current autodiff graph.
`MPOBasis` shares exact prefixes and suffix continuations while retaining
term-specific coefficient slots on a path edge, so its output is directly
compatible with `extensive_exponential()`.

For compiled optimization kernels, use the raw tensor interfaces. They avoid
the semantic-to-Quimb wrapper boundary and return only backend-native arrays:

```python
U_arrays = basis.exp_arrays(
    -1j * time,
    {"J": J, "V": V},
    order=2,
    mode="algorithm4",
)
```

`basis.exp_batch(step, coefficient_batch, ...)` accepts an array with shape
`(batch, number_of_terms)` and returns tensors with a leading batch axis. JAX
and current Torch releases use their native `vmap` implementation when
available; the fallback loop remains autodiff-safe.

For repeated calls with the same higher-order policy, compile the value-only
evaluator explicitly:

```python
compiled = basis.compile_exp(order=3, mode="optimal")
U_arrays = compiled.exp_arrays(
    -1j * time,
    {"J": J, "V": V},
)
```

`CompiledMPOExp` caches slot indices, affine static operator banks, and
higher-order history plans, but never coefficient-dependent tensors. Its raw
array methods avoid rebuilding `MPOAutomaton` and `FirstDegreeMPO` on every
step; coefficient assembly is one backend contraction per site while keeping
Torch/JAX gradients connected. Use `compiled.exp()` when a semantic MPO
wrapper is required. `compile_evolution`, `CompiledMPOEvolution`, `evaluate`,
and `time_evolution_arrays` remain compatibility names.

For one-off large-order constructions, disable persistent history caching:

```python
U = basis.exp(
    -1j * 0.01,
    {"J": 0.7, "V": 0.2},
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
U = basis.exp(-1j * 0.01, coefficients=coefficients, order=2)
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
U2_algorithm4 = H.extensive_exponential(
    dt=0.01,
    order=2,
    mode="algorithm4",
)
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

The named `mode` policies are `"base"` (Algorithms 1--2), `"algorithm4"`
(Algorithms 1, 2, and 4), `"optimal"` (Algorithms 1--3), and
`"approximate"` (Algorithms 1--4). The word `optimal` refers to the paper's
exact extension/compression construction, not to a globally minimum-bond MPO.
`mode="algorithm4"` is the fast explicit Algorithm-4 policy; it omits the
selected next-order replay from Algorithm 3. `mode="approximate"` retains the
full extended approximate construction. `max_bond` limits the temporary
history bonds before exact compression; `on_exceed="raise"` stops safely, while
`on_exceed="warn"` continues with a warning.

History storage is controlled independently with
`history_storage="auto"|"dense"|"sparse"|"streaming"`. The default uses the
cached structural-sparse path for automaton-built MPOs, and automatically
switches to the compatibility streaming path when `cache_history=False`.
`"sparse"` avoids evaluating structurally impossible local transition
products and batches the remaining physical block products. In all modes the
final local MPO tensors are materialized as dense virtual arrays because
Algorithms 1--4 rewrite them; this is not yet MPSKit.jl-style block-sparse
tensor storage. Metadata reports the selected mode and
`history_storage_blocks` gives the structurally stored versus considered local
block counts.

The numerical history pass gathers local products in backend batches. Its
virtual-channel rewiring uses fused transfer contractions for moderate
temporary bonds, with a scatter fallback for unusually large dense maps. This
preserves the exact Algorithms 1--2 result and the order-controlled
Algorithm-4 approximation while avoiding one full tensor copy per channel
merge.

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
the exponential `step` as well when constructing an MPO:

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

`compress_to_bond(chi=...)` is the common explicit wrapper used by the
parameterized evolution API. It selects Quimb cutoff compression by default or
fixed-rank compression when `differentiable=True`; use it when the evolution
MPO has already been constructed and only the final bond policy remains to be
chosen.

The current implementation targets finite open-boundary NumPy, Torch, and JAX
Autoray-compatible local blocks, including the optimal extension path under
Torch/JAX autodiff. A small accuracy regression benchmark compares orders
one--three with a first-order Trotter product and a finite two-site cluster
expansion; see `tests/test_mpo_benchmarks.py`. Native fermionic/Symmray
compilation and infinite/unit-cell MPOs remain separate future work.
