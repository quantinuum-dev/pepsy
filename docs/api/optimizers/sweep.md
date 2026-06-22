# `pepsy.optimizers.sweep.optimizer`

`SweepOptimizer` supports two boundary-environment engines:

- `boundary_engine="dmrg"` uses Pepsy `BdyMPS` plus `CompBdy`.
- `boundary_engine="quimb-mps"` uses Quimb MPS environments and scalar
  `contract_boundary(...)` contractions.
- `boundary_engine="auto"` keeps dense inputs on `dmrg` and routes
  Symmray-looking inputs to `quimb-mps`.

```{eval-rst}
.. automodule:: pepsy.optimizers.sweep.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
```
