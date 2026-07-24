# `pepsy.optimizers.sweep.optimizer`

`SweepOptimizer` supports two boundary-environment engines:

- `boundary_engine="dmrg"` uses Pepsy `BdyMPS` plus `CompBdy`.
- `boundary_engine="quimb-mps"` uses Quimb MPS environments and scalar
  `contract_boundary(...)` contractions. During a half-sweep it builds the
  opposite-side environments once, then advances the moving boundary one row
  or column at a time.
- `boundary_engine="auto"` keeps dense inputs on `dmrg` and routes
  Symmray-looking inputs to `quimb-mps`.

Torch-backed Symmray blocks use the Torch autograd local solver. NumPy-backed
Symmray blocks retain the finite-difference fallback.

```{eval-rst}
.. automodule:: pepsy.optimizers.sweep.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
```
