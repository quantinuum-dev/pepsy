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
- Sweep mode defaults to the optional NLopt `LD_LBFGS` solver. Install NLopt
  or pass an explicit `optimizer` / `sweep_optimize_kwargs` value when NLopt is
  unavailable.
- `non_unitary=True` normalizes generated targets and candidates; it does not
  implement the interval scheduling or norm-proxy machinery available in
  `MpsOptimizer`.
- Step records and fidelity traces are per measured two-site update. One-site
  gates applied outside a two-site batch are not recorded as separate steps.
  For PEPS lattice gates, prefer coordinate-tuple sites such as
  `((x0, y0), (x1, y1))` over flat integer pairs.
- `SimpleUpdateGen(update="parallel")` currently supports only direct-neighbor
  terms. Long-range routed terms need route-aware layer scheduling and should
  use `update="sequential"` for now.

```{eval-rst}
.. automodule:: pepsy.optimizers.peps.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pepsy.optimizers.peps.simple_update
   :members:
   :undoc-members:
   :show-inheritance:
```
