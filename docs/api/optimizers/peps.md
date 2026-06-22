# `pepsy.optimizers.peps.optimizer`

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

```{eval-rst}
.. automodule:: pepsy.optimizers.peps.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
```
