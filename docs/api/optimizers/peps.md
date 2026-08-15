# `pepsy.optimizers.peps`

`PepsOptimizer` has separate chi controls for different jobs:

- `chi` caps the optimized PEPS/PEPO virtual bonds.
- `boundary_chi` controls sweep/global optimizer environments.
- `normalize_chi` controls PEPS normalization contractions.
- `evaluation_chi` controls pre/post infidelity diagnostics used for accepting
  or rejecting a candidate.

Use `evaluation_chi` larger than `boundary_chi` when you want a stricter final
quality check without making every optimization environment more expensive.

`boundary_engine` controls the boundary implementation used when PEPS cleanup
delegates to `SweepOptimizer`. The default, `"auto"`, keeps dense inputs on the
Pepsy `BdyMPS`/`CompBdy` path and routes Symmray-looking inputs to Quimb MPS
boundaries. Use `boundary_engine="quimb-mps"` to force that path, and pass
Quimb environment controls with `boundary_options`.

Use `PepsOptimizer.run(k_2q_batch=N)` to absorb up to `N` sequential two-site
gates, plus intervening one-site gates, into one PEPS target before truncating
to `chi` and optionally running the sweep/global cleanup.

## Two-site boundary FIT

Dense DMRG boundaries can opt into native-SVD two-site updates through
`boundary_kwargs`:

```python
optimizer = pepsy.PepsOptimizer(
    state,
    gates,
    chi=32,
    boundary_chi=(64, 96),
    boundary_engine="dmrg",
    boundary_kwargs={
        "fit_mode": "two-site",
        "fit_sweep_sequence": "RL",
        "fit_rtol": 1e-8,
        "fit_min_iter": 2,
        "fit_patience": 2,
        "cutoff": 1e-12,
    },
)
```

For sweep cleanup, tuple `boundary_chi` values cap the norm and overlap
boundaries independently. Normalization and diagnostic contractions receive
the corresponding scalar `chi`. DMRG two-site boundaries start at bond 1 and
grow through local SVDs instead of global padding. `fit_mode="eff"` remains
the default while two-site accuracy and wall time are workload-dependent.

For the full-chain `eff` solver, `boundary_kwargs` can instead select native
block growth followed by one-site refinement:

```python
boundary_kwargs={
    "fit_mode": "eff",
    "fit_block_size": 2,
    "fit_adaptive_sweeps": 2,
    "fit_sweep_sequence": "RL",
}
```

This schedule is passed consistently to normalization, infidelity, and sweep
boundary initialization. `fit_rtol` is optional and adds one terminal norm
check per completed sweep only when enabled.

## One Torch SVD/QR policy

Torch autodiff through PEPS cleanup uses both SVD and QR. Configure both with
one `TorchLinalgConfig` object instead of registering the individual legacy
helpers separately:

```python
import pepsy

torch_linalg = pepsy.TorchLinalgConfig(
    mode="complex",          # selects complex-safe SVD/QR rules
    stabilized=True,          # finite SVD/QR VJPs for autodiff
    svd_driver="auto",       # CUDA: native Torch's driver selection
    cpu_svd="torch",         # CPU: native Torch LAPACK
    qr_rank_policy="warn",   # warn if a real QR block is rank deficient
    quimb_split_drivers=True, # required for raw Symmray blocks
)

optimizer = pepsy.PepsOptimizer(
    state,
    gates,
    chi=64,
    mode="global",
    torch_linalg_config=torch_linalg,
)
```

`stabilized=True` changes the reverse-mode rule, not the exact forward
factorization: it regularizes singular-gap and QR-pivot terms only where the
ordinary derivative is undefined or ill-conditioned. Use `stabilized=False`
for the fastest native forward/backward path when those gradients are not
needed. For non-approximate speed experiments, use CUDA `svd_driver="gesvdj"`
or CPU `cpu_svd="scipy_gesdd"`; `gesvda` is approximate and requires an
explicit `allow_approximate=True` acknowledgement.

For dense states, `quimb_split_drivers` can remain `False`. For Symmray PEPS,
set it to `True` because Quimb receives raw Torch charge blocks that do not
pass through ordinary Autoray dispatch. `PepsOptimizer` enables this flag
automatically when it detects Symmray blocks, while preserving the other
settings in a supplied policy. `register_torch_svd=False` is retained only as
a compatibility escape hatch for disabling automatic registration.

The lower-level `reg_*_torch` functions and `register_torch_linalg(...)`
remain compatibility APIs; new optimizer code should pass or construct
`TorchLinalgConfig` so SVD and QR cannot silently diverge.

`SimpleUpdateGen` preserves quimb's arbitrary-geometry simple-update sweep and
energy bookkeeping, but routes every gate through `pepsy.gate_simple(...)`.
This lets sequential simple update handle long-range PEPS terms via Pepsy's
SWAP routing. Use `route_opts` for routing controls such as `sequence`,
`path_canonize`, and `path_compress`.

## Important cautions

- A second `run()` call applies the queued gates again to the current state.
  The default `reset_traces=True` resets diagnostics only; use `set_state(...)`
  when you want to replay from a fresh input state.
- If `normalize_chi` or `evaluation_chi` is left unset, standalone
  normalization and infidelity diagnostics use `2 * max(boundary_chi)`.
  This can be more accurate, but it is often the expensive part of a run.
- `accept_if_improved=True` is most consistent with
  `measure_final_infidelity=True`. If final measurement is disabled, the
  fallback optimizer loss can come from the coarser `boundary_chi` environment
  while the pre-check used `evaluation_chi`.
- Sweep mode defaults to Torch Adam for Torch-backed Symmray states and to the
  optional NLopt `LD_LBFGS` solver otherwise. Pass an explicit `optimizer` /
  `sweep_optimize_kwargs` value to override this choice.
- Generated targets are normalized by default before infidelity estimates and
  variational cleanup. Pass `normalize_target=False` only when that
  normalization is handled externally. Passing `normalize_target=None` keeps
  the legacy behavior of following `non_unitary`; this does not implement the
  interval scheduling or norm-proxy machinery available in `MpsOptimizer`.
- Step records and fidelity traces are per measured two-site update. One-site
  gates applied outside a two-site batch are not recorded as separate steps.
  For PEPS lattice gates, prefer coordinate-tuple sites such as
  `((x0, y0), (x1, y1))` over flat integer pairs.
- `SimpleUpdateGen(update="parallel")` currently supports only direct-neighbor
  terms. Long-range routed terms need route-aware layer scheduling and should
  use `update="sequential"` for now.


> API details are maintained as handwritten Markdown in this page.
