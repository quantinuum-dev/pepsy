# Getting Started

This page gives the shortest path from install to a first contraction run.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Run these commands from the checkout. For development, use an editable install
with `python -m pip install -e ".[dev]"`. Add only the optional
[installation profiles](installation.md) required by your workflow.

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
