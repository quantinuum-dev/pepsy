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
`MatrixProductOperator` and performs no numerical truncation.

The first tensor-network-aware time-evolution path is available through
`extensive_exponential(dt, order=1 or 2)`. It reads the first-degree `I`, `A`,
`B`, `C`, and `D` blocks from the local MPO tensors and assembles the
size-extensive order-one or order-two MPO directly. It never forms a global
dense Hamiltonian or matrix exponential. The resulting internal bond spaces
are the paper's compressed level spaces: `1 + chi` for order one and
`1 + 2 * chi + chi**2` for order two, with `chi` allowed to vary by cut.
This path currently targets ordinary NumPy/Autoray-compatible local blocks;
native fermionic/Symmray compilation remains separate.

```python
U1 = H.extensive_exponential(dt=0.01, order=1)
U2 = H.extensive_exponential(dt=0.01, order=2)

# Quimb remains the compiled interchange boundary.
mpo = U2.to_mpo()
```

General-order Taylor extension, approximate row compression, and native
Symmray compilation are intentionally separate follow-up stages. Keeping these
stages explicit prevents a numerical cutoff from being confused with the
paper's algebraically exact compression.
