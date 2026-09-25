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
    tn_flat=ket_tagged,  # optional single-layer shape/backend reference
    tn_double=norm,      # BRA--KET target used because flat=False
    chi=64,
    single_layer=False,
)
```

## Step 3: contract

```python
res = pepsy.contract_boundary(
    norm=norm,
    bdy=bdy,
    fit_mode="dmrg2",             # two-site warm-up, then one-site refinement
    fit_init_strategy="guess-src", # disposable SRC warm start
    fit_init_seed=7,
    fit_max_bond=64,
    fit_sweep_sequence="LR",
    fit_cutoff=1e-12,
    fit_rtol=1e-8,
    fit_min_iter=2,
    direction="y",
    n_iter=8,
    max_separation=0,
    track_boundary_fidelity=True,
    fit_timing=True,
)

print("cost:", res.cost)
print("fidel entries:", len(res.fidel))
for fit in res.fit_diagnostics:
    print(
        fit.boundary_key,
        fit.iterations,
        fit.convergence_reason,
        fit.elapsed_seconds,
    )
```

## Notes on parameters

- `chi`: higher means potentially better accuracy, higher runtime/memory.
- `n_iter`: more local fit sweeps per boundary update.
- `fit_mode="two-site"`: fixed two-site FIT updates. `fit_mode="dmrg2"`
  uses two-site warm-up followed by one-site refinement. The direct modes
  `"direct"`, `"src"`, `"zipup"`, `"sdc"`, and `"dm"` use Quimb
  compression without FIT. New two-site boundaries start at bond 1 and grow
  locally up to the cap. `fit_mode="dmrg"` aliases the one-site/effective
  `"eff"` solver; the default remains `"eff"`.
- `fit_init_strategy="guess-direct"`, `"guess-src"`, or `"guess-sdc"`:
  build a disposable compressed guess from the exact boundary target before
  FIT. This does not replace the exact target or mutate the reusable boundary.
  Use `"direct"` to retain the historical initial guess.
- `fit_max_bond`: two-site SVD cap. `peps_norm(..., chi=...)` supplies `chi`
  automatically; direct `contract_boundary(...)` calls can set it explicitly.
- `fit_sweep_sequence`: use `"LR"` for left-to-right then right-to-left local
  sweeps, or `"RL"` for the reverse order.
- `fit_rtol`, `fit_min_iter`, and `fit_patience`: opt into adaptive stopping.
  Leave `fit_rtol=None` to run exactly `n_iter` sweeps.
- `fit_cutoff` and `fit_cutoff_mode`: direct `contract_boundary(...)`
  two-site SVD policy. The higher-level `peps_norm(...)` API uses `cutoff`
  with `fit_cutoff_mode`; `"auto"` selects a dtype-aware cutoff and the
  standard `"rsum2"` mode.
- `fit_timing`: opt into elapsed and detailed two-site sweep timings. Cheap
  iteration and convergence metadata is present in `fit_diagnostics` even
  when timing is disabled.
- `return_info=True`: use this with `peps_norm(...)` or `boundary_norm(...)`
  when the high-level scalar helpers should return `BoundaryContractResult`.
- `direction`: `y`, `y_left`, `y_right`, `x`, `x_left`, `x_right`.
- `max_separation`: currently `0` or `1`.

## Next steps

- See [fidelity diagnostics](fidelity_diagnostics.md) to interpret `res.fidel`.
- See [how-to tuning](../howto/choose_parameters.md) for practical defaults.
