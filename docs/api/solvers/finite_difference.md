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

`fd-adam` evaluates the final update once before selecting the best parameters;
that evaluation is included in `n_evals`. `final_loss` describes the returned
parameters with either value of `restore_best`.

`fd-scipy` uses the [SciPy budget aliases and stopping rules](gradient.md).
Its `n_evals` counts individual loss calls, including finite-difference probes;
SciPy's TNC `maxfun` counts complete objective/gradient estimates instead.
`trust-constr` callbacks are supported. Invalid terminal objectives/gradients
produce an `invalid_objective` warning/status; nonscalar or nonreal losses
raise `ValueError`.

`fd-nlopt` follows the NLopt error and termination policies in the gradient
guide. It counts a final returned-point evaluation when no best-loss cache
applies, and rejects `max_step` instead of altering callback coordinates.
