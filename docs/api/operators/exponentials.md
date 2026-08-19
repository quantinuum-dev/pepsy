# Exponential API: MPO and PEPO

This page is the short usage guide for the two higher-order exponential
builders. The rule is simple:

> Use `exp(step, ...)` for the operator exponential. Use `compile_exp(...)`
> when the operator topology is reused. The `step` is the scalar in
> `exp(step * H)`; it is not automatically a physical time.

## Choose the representation

| Goal | Canonical entry point | Result |
| --- | --- | --- |
| One parameterized 1D Hamiltonian | `MPOBasis.from_pauli_terms(...)` or `MPOBasis.from_local_terms(...)` | Reusable `MPOBasis` |
| One MPO exponential | `basis.exp(step, parameters=...)` | Semantic `FirstDegreeMPO` |
| Repeated MPO exponentials | `basis.compile_exp(...).exp(step, ...)` | Cached `CompiledMPOExp` call plus semantic MPO |
| Raw MPO tensors for a compiled kernel | `basis.compile_exp(...).exp_arrays(step, ...)` | Backend-native tensor tuple |
| Quimb MPO interoperability | `semantic_mpo.to_mpo()` | Quimb `MatrixProductOperator` |
| One fixed-channel square-lattice PEPO | `PauliPEPOBasis.compile(...)` | Reusable `PauliPEPOBasis` |
| One PEPO exponential | `basis.exp(step, ...)` | `ActivePEPOBlocks` by default |
| Repeated PEPO exponentials | `basis.compile_exp().exp(step, ...)` | Cached `CompiledPEPOExp` call |
| Dense PEPO materialization | `active_blocks.to_pepo()` or `materialize=True` | Quimb `PEPO` |
| Coefficient-dependent real-time PEPO | `build_real_time_cluster_expansion_pepo(...)` | Quimb `PEPO` or active blocks |
| Fractional-step PEPO composition | `compose_cluster_expansion_pepo(...)` | Quimb `PEPO` |

The MPO and PEPO APIs deliberately have the same top-level vocabulary. They
do not have the same output layout: an MPO is a 1D semantic operator, while a
PEPO is first kept as sparse active virtual-sector blocks.

## MPO: parameterized chain

```python
import torch

from pepsy.operators import MPOBasis, MPOParameter

basis = MPOBasis.from_pauli_terms(
    8,
    [
        ((i, i + 1), "ZZ", MPOParameter("J"))
        for i in range(7)
    ] + [
        ((i,), "X", MPOParameter("h"))
        for i in range(8)
    ],
)

J = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
h = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

# The result is a semantic FirstDegreeMPO. No dense 2**8 x 2**8 matrix is made.
U = basis.exp(
    -1j * tau,
    {"J": J, "h": h},
    order=4,
    mode="optimal",
)
```

Use `coefficients=...` instead of `parameters=...` when the optimizer already
has one backend vector in term order:

```python
theta = torch.stack((J, h))  # one value per compiled term
U = basis.exp(-1j * tau, coefficients=theta, order=4, mode="optimal")
```

`parameters` and `coefficients` are mutually exclusive. A parameter mapping
is resolved through `MPOParameter("name")` references; a coefficient vector
must have exactly `basis.num_terms` entries.

### Repeated MPO evaluations

Compile the order/mode policy once when only coefficients or `step` change:

```python
compiled = basis.compile_exp(order=4, mode="optimal")

U = compiled.exp(-1j * tau, {"J": J, "h": h})
raw_tensors = compiled.exp_arrays(-1j * tau, {"J": J, "h": h})
```

`CompiledMPOExp` caches topology, history plans, coefficient-slot indices,
and static operator banks. It never caches coefficient-dependent arrays or
autodiff graphs. `compiled.exp(...)` returns the semantic MPO; use
`compiled.exp_arrays(...)` only when a numerical kernel needs the raw tensor
tuple.

### MPO step, order, and compression

All exponential methods use the same convention:

```text
real time:       step = -1j * tau
imaginary time:  step = -beta
custom operator: step = any backend scalar
```

`order` is the Taylor/history order. The named policies are:

| `mode` | Construction |
| --- | --- |
| `"base"` | Algorithms 1–2 |
| `"algorithm4"` | Algorithms 1, 2, and 4 |
| `"optimal"` | Algorithms 1–3, including the selected next-order extension |
| `"approximate"` | Algorithms 1–4 |

`max_bond` protects the temporary history construction. `chi` is a separate
final numerical MPO compression cap. With `chi=None`, the result remains a
semantic `FirstDegreeMPO`; with `chi` set, the default result is a Quimb MPO.
Use `differentiable=True` with `chi` for fixed-rank autodiff compression.

## PEPO: fixed Pauli channels on a square lattice

`PauliPEPOBasis` is for a qubit square lattice with translation-invariant
onsite and positive-direction edge Pauli slots. Its virtual channels are
fixed before coefficients are evaluated, so coefficient changes do not cause
rank-changing SVD decisions.

```python
import torch

from pepsy.operators import PauliPEPOBasis

basis = PauliPEPOBasis.compile(
    4,
    4,
    [("onsite", "X"), ("edge", "ZZ")],
    order=4,
    cyclic=True,
    symmetry="C4",
)

theta = torch.tensor(
    [0.5, 1.0],
    dtype=torch.float64,
    requires_grad=True,
)
tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

# Default: keep the sparse active-sector representation.
active = basis.compile_exp().exp(
    -1j * tau,
    coefficients=theta,
)

# Explicit dense/interoperability boundary.
pepo = active.to_pepo()
```

The direct one-off form is `basis.exp(-1j * tau, coefficients=theta)`. The
compiled form returns the same `ActivePEPOBlocks` representation while
reusing the fixed lattice/channel topology. `materialize=True` is equivalent
to calling `.to_pepo()` for a single evaluation.

`ActivePEPOBlocks` contains only stored active sector blocks. It is the normal
autodiff and memory-saving result; dense PEPO tensors should be materialized
only when required by downstream code or a small-system validation.

The fixed Pauli implementation currently supports tree orders 1–4. It does
not perform PEPO × PEPS contractions, expectation values, loop-cluster
corrections, or symmetry-native block-space calculations.

## Dense cluster-expansion convenience API

For a fixed dense local Hamiltonian, use `ClusterExpansionPlan` or
`build_cluster_expansion_pepo`. That family uses the cluster convention
`exp(-beta * H)`:

```python
active = plan.build(beta=0.05, materialize=False)
pepo = active.to_pepo()
```

`ClusterExpansionReport` contains local residual and storage diagnostics. Its
residual norms are local factorization diagnostics, not a global PEPO error
bound. The dense implementation includes recursive generic order-five
through order-nine tree paths; higher orders remain unsupported. Finite dense
model adapters are available through `ClusterModelAdapter` and
`build_model_cluster_expansion_pepo`.

For local coefficient lists and real-time evolution, use
`build_real_time_cluster_expansion_pepo`. It assembles the coefficient-weighted
one- and two-site terms first, then uses the cluster convention with
`beta=1j * time`, targeting `exp(-1j * time * H)` for the summed `H`.
With `fit_method="quimb"`, generic tree residuals use Quimb tree fitting and
generic loop residuals use complex ALS. This numerical path is not
coefficient-differentiable; its local fit diagnostics are returned in
`ClusterExpansionReport`.

## Caching and autodiff contract

- Compiling a basis caches structure only: topology, channels, histories, and
  static operator banks.
- Every `exp` call creates fresh backend-connected values, so Torch/JAX
  gradients see the current coefficients and `step`.
- `clear_history_cache()` releases reusable higher-order plans but leaves the
  compiled term topology intact.
- `to_mpo()` and `to_pepo()` are explicit interoperability/materialization
  boundaries; neither is needed to construct the exponential itself.

## Compatibility names

New code should use `exp`, `compile_exp`, `CompiledMPOExp`, and
`CompiledPEPOExp`. The previous names `time_evolution`, `evolution_mpo`,
`compile_evolution`, `evaluate`, and `CompiledMPOEvolution` remain available
as compatibility shims so existing programs do not break. They are not the
recommended vocabulary for new code.
