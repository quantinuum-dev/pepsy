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
- numerical compression, Taylor coefficient rewiring, and native Symmray
  compilation can be added later without changing the ordinary MPO contract.

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

The current layer supports exact finite open-boundary construction, addition,
scaling, products, commutators, powers, and the paper's conservative exact
history/column compression. `to_mpo()` returns an ordinary Quimb
`MatrixProductOperator` and performs no numerical truncation. The semantic
object remains the source of truth for history-aware operations; the Quimb
object is the execution and interoperability boundary.

The tensor-network-aware time-evolution path is available through
`extensive_exponential(dt, order=N)`. For chains with at least two sites it
constructs the full local history power, applies the paper's Algorithms 1 and
2 as virtual-bond rewiring operations, and only then contracts the finite
all-one boundary vectors. Every operation is local to MPO tensors; no global
dense Hamiltonian or matrix exponential is formed. The raw history space grows
as the local MPO channel space to the power `N`, while exact history
compression removes equivalent channels after that reference table is built.
The one-site path currently supports direct orders one and two.

```python
import quimb.tensor as qtn

U1 = H.extensive_exponential(dt=0.01, order=1)
U2 = H.extensive_exponential(dt=0.01, order=2)
U3 = H.extensive_exponential(dt=0.01, order=3)

# Optional paper extensions, kept explicit in the API.
U2_extended = H.extensive_exponential(dt=0.01, order=2, extend=True)
U2_approx = H.extensive_exponential(dt=0.01, order=2, approximate=True)

# Quimb remains the compiled interchange boundary.
mpo = U2.to_mpo()

# Tensor-network application and expectation evaluation.
state = qtn.MPS_computational_state("0000")
state_next = U2.apply_to_mps(state, method="direct", cutoff=1e-10)
energy_factor = U2.expectation(state)
```

`extend=True` applies Algorithm 3: selected order `N + 1` local histories are
added without increasing the exact-history bond dimension. `approximate=True`
applies Algorithm 4 after exact compression. It is an order-controlled
analytical approximation, not a numerical SVD cutoff, and is therefore exposed
as a separate opt-in flag. Numerical Quimb compression remains a separate
post-processing step.

`product(kind=...)` and `disjoint_product` currently label provenance only;
they do not perform support-overlap analysis. See the [development module
map](../../development/modules/higher_order_mpo.md) for the execution order,
design rationale, and prioritized future improvements.

`apply_to_mps` delegates to Quimb's MPO–MPS compression methods, while
`expectation` delegates to Pepsy's normalized MPS contraction helper. These
methods are the intended validation and execution path for larger systems;
dense conversion is only appropriate as a small-system regression oracle.

The current implementation targets ordinary NumPy/Autoray-compatible local
blocks. Native fermionic/Symmray compilation remains separate and is not
changed by this API.
