# `pepsy.solvers.gradient`

Use `GradientOptimizer` to optimize a dictionary of parameters against a
scalar loss. Choose a solver through `solver=`, and pass its settings in
`options`.

```python
import torch
from pepsy.solvers import GradientOptimizer

optimizer = GradientOptimizer(
    solver="torch-adam", options={"lr": 0.1}, n_steps=100,
)
result = optimizer.run(
    params_init={"x": torch.tensor([3.0], dtype=torch.float64)},
    loss_fn=lambda params: ((params["x"] - 1.0) ** 2).sum(),
)
print(result.params["x"], result.best_loss)
```

The result contains optimized `params`, loss `history`, `best_loss`,
`final_loss`, `convergence_reason`, and evaluation count `n_evals`.
For the gradient-based `scipy` family, `convergence_reason` contains SciPy's
termination message, identifying convergence, budget exhaustion, or failure.
Pepsy's callback stops retain the labels `"patience"` and `"bad_max"`.
These messages are solver-specific strings, rather than a fixed enumeration.
`optimize_packed_params(...)` provides the function form, returning
`(params, history)`.

| Solver family | Dependencies and behavior |
| --- | --- |
| `torch-*` | Torch; differentiable scalar loss, parameters stay on their device |
| `scipy`, `nlopt` | Torch or JAX plus the selected solver; CPU parameter packing, native device autodiff |
| `jax-*` | JAX and Optax; JAX-compatible scalar loss |
| `fd-*` | Finite differences; see [FDSolver](finite_difference.md) |

`SUPPORTED_SOLVERS` lists canonical solver names. Install the relevant
[optional profile](../../installation.md#optional-features) before using one.

`solver="lbfgs"` or `"scipy-lbfgs"` selects SciPy L-BFGS-B;
`solver="LD_LBFGS"` selects NLopt L-BFGS. Both accept Torch or native JAX
parameters. The callback backend follows the input arrays; loss functions must
use that backend. NumPy inputs retain the Torch default. JAX losses must be
JIT-compatible; neither Torch nor Optax is required for the JAX host-solver path.
JAX parameters must share one device. Floating/complex dtype, shape and device
are preserved, and complex values are packed as real/imaginary coordinates.
SciPy's second-order methods use native JAX Hessians or Hessian-vector products.
Enable JAX x64 before constructing float64/complex128 parameters if needed.

The host solver transfers a packed iterate to the device and reads back the
scalar loss and flat gradient per evaluation. `n_steps` limits SciPy iterations
or NLopt objective evaluations; the budgets have different meanings. Use
`options={"assume_nonnegative": False}` for signed losses, including cluster
infidelity approximations that can be negative. `torch-lbfgs` explicitly selects
Torch's optimizer and accepts only Torch-compatible inputs.
