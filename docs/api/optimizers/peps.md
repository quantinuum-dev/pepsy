# `pepsy.optimizers.peps`

`PepsOptimizer` has separate chi controls for different jobs:

- `chi` caps the optimized PEPS/PEPO virtual bonds.
- `boundary_chi` controls sweep/global optimizer environments.
- `normalize_chi` controls PEPS normalization contractions.
- `evaluation_chi` controls pre/post infidelity diagnostics used for accepting
  or rejecting a candidate.

Use `evaluation_chi` larger than `boundary_chi` when you want a stricter final
quality check without making every optimization environment more expensive.

Use `PepsOptimizer.run(k_2q_batch=N)` to absorb up to `N` sequential two-site
gates, plus intervening one-site gates, into one PEPS target before truncating
to `chi` and optionally running the sweep/global cleanup.

```{eval-rst}
.. automodule:: pepsy.optimizers.peps
   :members:
   :undoc-members:
   :show-inheritance:
```
