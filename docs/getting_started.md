# Getting Started

Build a small PEPS, contract its norm, and inspect the result. For the complete
example in one code block, see [Quickstart](quickstart.md).

## 1. Install

Follow the [installation guide](installation.md) using Python 3.12 or newer.
This example needs only the base package. Run the Python blocks below in order.

## 2. Build a small test network

```python
import quimb.tensor as qtn
from pepsy.boundary import BdyMPS, build_bra_ket, contract_boundary

ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1, dtype="complex128")
ket_tagged, norm = build_bra_ket(ket=ket)
```

## 3. Initialize boundary states

```python
bdy = BdyMPS(
    tn_flat=ket_tagged,  # optional single-layer shape/backend reference
    tn_double=norm,      # BRA--KET target used because flat=False
    chi=32,
    single_layer=False,
)
```

## 4. Contract and inspect diagnostics

```python
res = contract_boundary(
    norm=norm,
    bdy=bdy,
    direction="y",
    n_iter=2,
    track_boundary_fidelity=True,
)

print(res.cost)
print(res.fidel)
```

`contract_boundary` always returns a `BoundaryContractResult` object. Use
`res.cost` and `res.fidel` directly.

Next: see [tutorials](tutorials/index.md) for more complete workflows.
