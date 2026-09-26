# `pepsy.solvers.finite_difference`

Use the public `FDSolver` interface when the loss can be evaluated but
cannot be differentiated with autograd. It accepts `fd-adam`, `fd-scipy`,
and `fd-nlopt` solver families.

```python
import torch
from pepsy.solvers import FDSolver

optimizer = FDSolver(
    solver="fd-adam",
    options={"lr": 0.1, "fd_method": "central", "fd_eps": 1e-5},
    n_steps=100,
)
result = optimizer.run(
    params_init={"x": torch.tensor([3.0], dtype=torch.float64)},
    loss_fn=lambda params: ((params["x"] - 1.0) ** 2).sum(),
)
print(result.params["x"], result.best_loss)
```

`fd_method` selects `central` or `forward` differences; `fd_eps` controls
the perturbation size. Each gradient estimate requires repeated loss
evaluations, so begin with a small parameter set.

These solvers still use Torch tensors for parameters and loss evaluation,
but do not backpropagate through the loss. SciPy and NLopt variants also
require their respective optional dependencies. Results use the same
[GradSolverResult fields](gradient.md) as `GradientOptimizer`.
