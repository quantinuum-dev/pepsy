# Getting Started

This page gives the shortest path from install to a first contraction run.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e .[torch]
pip install -e .[solvers]
pip install -e .[viz]
```

## 2. Build a small test network

```python
import quimb.tensor as qtn
import pepsy

ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1, dtype="complex128")
ket_tagged, norm = pepsy.build_bra_ket(ket=ket)
```

## 3. Initialize boundary states

```python
bdy = pepsy.BdyMPS(
    tn_flat=ket_tagged,
    tn_double=norm,
    chi=32,
    single_layer=False,
)
```

## 4. Contract and inspect diagnostics

```python
res = pepsy.contract_boundary(
    norm=norm,
    bdy=bdy,
    direction="y",
    n_iter=2,
    track_boundary_fidelity=True,
)

print(res.cost)
print(res.fidel)
```

```{note}
`contract_boundary` always returns a `BoundaryContractResult` object.
Use `res.cost` and `res.fidel` directly.
```

Next: see [tutorials](tutorials/index.md) for more complete workflows.
