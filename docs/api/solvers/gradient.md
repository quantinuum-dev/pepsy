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
`optimize_packed_params(...)` provides the function form, returning
`(params, history)`.

| Solver family | Dependencies and behavior |
| --- | --- |
| `torch-*` | Torch; differentiable scalar loss, parameters stay on their device |
| `scipy`, `nlopt` | Torch plus the selected solver; CPU parameter packing |
| `jax-*` | JAX and Optax; JAX-compatible scalar loss |
| `fd-*` | Finite differences; see [FDSolver](finite_difference.md) |

`SUPPORTED_SOLVERS` lists canonical solver names. Install the relevant
[optional profile](../../installation.md#optional-features) before using one.
