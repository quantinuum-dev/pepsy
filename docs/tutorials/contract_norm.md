# Tutorial: Contract a PEPS Norm

This tutorial covers a full `prepare -> boundary init -> sweep contract` pipeline.

## Workflow overview

```text
PEPS ket
  -> build_bra_ket(ket, bra?)
      -> tagged ket + double-layer norm TN
  -> BdyMPS(...)
      -> boundary dictionary mps_b
  -> contract_boundary(norm, bdy, ...)
      -> BoundaryContractResult(cost, fidel, ...)
```

## Step 1: prepare inputs

```python
import pepsy
import quimb.tensor as qtn

ket = qtn.PEPS.rand(Lx=4, Ly=4, bond_dim=2, seed=7, dtype="complex128")
ket_tagged, norm = pepsy.build_bra_ket(ket=ket)
```

## Step 2: initialize boundaries

```python
bdy = pepsy.BdyMPS(
    tn_flat=ket_tagged,
    tn_double=norm,
    chi=64,
    single_layer=False,
)
```

## Step 3: contract

```python
res = pepsy.contract_boundary(
    norm=norm,
    bdy=bdy,
    fit_mode="two-site",
    fit_max_bond=64,
    fit_sweep_sequence="RL",
    fit_cutoff=1e-12,
    fit_rtol=1e-8,
    fit_min_iter=2,
    direction="y",
    n_iter=8,
    max_separation=0,
    track_boundary_fidelity=True,
)

print("cost:", res.cost)
print("fidel entries:", len(res.fidel))
```

## Notes on parameters

- `chi`: higher means potentially better accuracy, higher runtime/memory.
- `n_iter`: more local fit sweeps per boundary update.
- `fit_mode="two-site"`: optimize neighboring boundary tensors and compress
  their middle bond with a native SVD. New boundaries start at bond 1 and grow
  locally up to the cap. The default remains `"eff"`.
- `fit_max_bond`: two-site SVD cap. `peps_norm(..., chi=...)` supplies `chi`
  automatically; direct `contract_boundary(...)` calls can set it explicitly.
- `fit_sweep_sequence`: use `"RL"` to alternate left-to-right and
  right-to-left sweeps.
- `fit_rtol`, `fit_min_iter`, and `fit_patience`: opt into adaptive stopping.
  Leave `fit_rtol=None` to run exactly `n_iter` sweeps.
- `fit_cutoff` and `fit_cutoff_mode`: direct `contract_boundary(...)`
  two-site SVD policy. The higher-level `peps_norm(...)` API uses `cutoff`
  with `fit_cutoff_mode`.
- `direction`: `y`, `y_left`, `y_right`, `x`, `x_left`, `x_right`.
- `max_separation`: currently `0` or `1`.

## Next steps

- See [fidelity diagnostics](fidelity_diagnostics.md) to interpret `res.fidel`.
- See [how-to tuning](../howto/choose_parameters.md) for practical defaults.
