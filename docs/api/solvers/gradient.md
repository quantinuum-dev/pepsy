# `pepsy.solvers.gradient`

Use `GradientOptimizer` to optimize a dictionary of parameters against a
scalar loss. Choose a solver through `solver=`. `options` is optional: Pepsy
resolves solver aliases, chooses the autodiff backend from the input arrays
for SciPy/NLopt, and applies the selected runner's defaults. Pass only settings
you want to override; callers do not need a solver-specific `if` ladder.

```python
from pepsy.solvers import GradientOptimizer

optimizer = GradientOptimizer(solver="lbfgs", n_steps=100, progress=True)
```

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
JAX/Optax solvers pair each measured loss with the parameters at which it was
evaluated. Their final update receives one extra loss evaluation, included in
`n_evals`, so `final_loss` describes the returned parameters with either value
of `restore_best`. An invalid final update cannot replace a finite best state.
`optimize_packed_params(...)` provides the function form, returning
`(params, history)`.

Solvers preserve the supplied parameter arrays. Torch-backed solves allocate
independent, contiguous parameter storage once per solve on the original
device; trial updates and returned parameters cannot overwrite the inputs or
tensor-network snapshots sharing those inputs. This also applies to NumPy
inputs converted to Torch. JAX follows its immutable-array semantics.

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
Torch's optimizer and accepts only Torch-compatible inputs. Native `torch-*`
solvers reject JAX arrays, native `jax-*` solvers reject Torch tensors, and
mixed Torch/JAX parameter mappings are rejected before conversion or loss
evaluation. NumPy/scalar conversion remains supported. Choose `lbfgs` or
`LD_LBFGS` when the same solver configuration should work with either backend.

## Default options and overrides

- SciPy L-BFGS-B uses Pepsy's `ftol=1e-9`, `gtol=1e-9`, and `maxls=40`.
  Other unspecified L-BFGS-B controls use SciPy's defaults.
- NLopt uses `ftol_rel=1e-9`, `ftol_abs=1e-9`, and `xtol_rel=1e-9`.
- Torch/JAX iterative solvers use Pepsy's default `lr=0.01`, with other
  unspecified algorithm controls supplied by Torch/Optax.

These are numerical defaults, not automatic tuning to a particular objective.
Removing explicit tolerances or learning rates can change a run's stopping
point and convergence history. For example, a deliberately stricter SciPy run
can still request `options={"ftol": 0.0, "gtol": 1e-12}`. NLopt's function and
parameter tolerances are separate controls; they are not interchangeable with
a gradient tolerance. Existing legacy option aliases remain supported.
Constructor options are overridden by matching keys supplied to `run(options=...)`.
