# Quickstart

After [installation](installation.md), run this complete example to contract
a small PEPS norm. For a step-by-step explanation, use
[Getting started](getting_started.md).

```python
import quimb.tensor as qtn
from pepsy.boundary import BdyMPS, build_bra_ket, contract_boundary

ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1, dtype="complex128")
ket_tagged, norm = build_bra_ket(ket=ket)

bdy = BdyMPS(
    tn_flat=ket_tagged,  # optional single-layer shape/backend reference
    tn_double=norm,      # BRA--KET target used because flat=False
    chi=32,
    single_layer=False,
)

res = contract_boundary(
    norm=norm,
    bdy=bdy,
    direction="y",
    n_iter=2,
    track_boundary_fidelity=True,
)

print("cost:", res.cost)
print("fidelity history:", res.fidel)
```

`contract_boundary` returns `BoundaryContractResult` directly. Use `res.cost`
and `res.fidel` for outputs and diagnostics.
