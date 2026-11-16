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
An invalid terminal objective or gradient is reported as `"invalid_objective"`
with a `RuntimeWarning`, rather than successful SciPy convergence. Scalar/real
loss contract violations raise `ValueError`; they are not numerical penalties.
These messages are solver-specific strings, rather than a fixed enumeration.
Torch, JAX/Optax, and finite-difference Adam solvers pair each measured loss
with the parameters at which it was evaluated. First-order solvers' final
updates receive one extra loss evaluation, included in
`n_evals`, so `final_loss` describes the returned parameters with either value
of `restore_best`. An invalid final update cannot replace a finite best state.
Torch terminal evaluations keep autograd enabled for objectives that contain
derivative penalties. Native Torch/JAX solvers raise `FloatingPointError` on
nonfinite gradients before accepting an update; JAX skips updates for nonfinite
losses while applying the existing `bad_max` policy for first-order methods.
Native L-BFGS reuses the terminal line-search value and gradient and stops
immediately on an invalid initial objective or an unsuccessful line search.
`optimize_packed_params(...)` provides the function form, returning
`(params, history)`.

Solvers preserve the supplied parameter arrays. Torch-backed solves allocate
independent, contiguous parameter storage once per solve on the original
device; trial updates and returned parameters cannot overwrite the inputs or
tensor-network snapshots sharing those inputs. This also applies to NumPy
inputs converted to Torch. JAX follows its immutable-array semantics.
Unbounded native Torch solvers retain best-parameter snapshots on the source
device without packing them through NumPy. Explicit Torch bounds still use
host clipping.

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
Native JAX/Optax updates conjugate JAX's complex derivative to obtain the
descent direction. Losses must be scalar and real within absolute imaginary
tolerance `1e-10`, including the final evaluation. Native `jax-*` solvers reject
non-`None` `bounds`, `lower_bounds`, `upper_bounds`, `max_step`, and
`max_step_norm`, and reject `angle_wrap=True`; these controls are unsupported.
For JAX parameters with bounds, choose SciPy L-BFGS-B or NLopt.

For autodiff NLopt solvers (`nlopt` and `LD_*`), `gtol` remains a compatibility
alias for the parameter-step tolerance `xtol_rel`.
Set `options={"gradient_tol": 1e-10}` to add a separate
maximum absolute derivative stopping threshold, measured over packed real
coordinates before gradient clipping. Its result label is `"gradient_tol"`.
The threshold must be finite and nonnegative; omitted or `None` disables it.
Only finite objective/gradient trials qualify, after `min_steps` evaluations
(`min_evals` is an alias). NLopt's budget counts objective evaluations,
including line-search trials. With `restore_best=False`, the returned point
is the trial that met the threshold; the default best-state restoration can
return an earlier point with a lower eligible loss. This is a raw derivative
criterion, not a projected-gradient criterion for constrained optima.

`jax-adam` runs up to 50 Optax updates per JIT-compiled `lax.scan` call by
default, then reports the completed chunk to Python. This avoids synchronizing
the device after every update while preserving each Adam update and the full
per-step loss history. Set `options={"scan_steps": 1}` for per-step reporting,
or choose another positive integer to tune the chunk size. Other first-order JAX
Optax solvers retain one update per call by default; they also accept
`scan_steps` when a multi-step scan is useful.

The host solver transfers a packed iterate to the device and reads back the
scalar loss and flat gradient per evaluation. `n_steps` limits SciPy iterations
(TNC: objective calls) or NLopt objective evaluations; the budgets have
different meanings. SciPy budget aliases take precedence in this order:
`maxiter`, `maxeval`, `its_max`, then `n_steps`. Despite its legacy name,
`maxeval` is an **iteration** alias for SciPy methods other than TNC.
TNC uses `maxfun`, which can be supplied explicitly; L-BFGS-B also accepts
SciPy's separate `maxfun` evaluation cap. Finite-difference perturbation calls
are additional to SciPy's objective-call count. PEPS local sweeps preserve
these explicit limits and retain their default 30 SciPy iterations.
Use
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
- Torch/JAX first-order solvers and Torch L-BFGS use Pepsy's default
  `lr=0.01`, with other unspecified controls supplied by Torch/Optax.

These are numerical defaults, not automatic tuning to a particular objective.
Removing explicit tolerances or learning rates can change a run's stopping
point and convergence history. For example, a deliberately stricter SciPy run
can still request `options={"ftol": 0.0, "gtol": 1e-12}`. NLopt's function and
parameter tolerances are separate controls; they are not interchangeable with
a gradient tolerance. Existing legacy option aliases remain supported.
Constructor options are overridden by matching keys supplied to `run(options=...)`.

NLopt and FD-NLopt propagate nonscalar/nonreal loss errors and report
`invalid_objective` with a warning if no objective/gradient evaluation was
valid. Their normal termination reasons distinguish `success`, `stopval`,
`ftol`, `xtol`, `maxeval`, and `maxtime`; explicit `patience`/`bad_max` stops
retain those labels. When best-state restoration does not supply a cached
loss, the returned point receives a counted final evaluation.

NLopt rejects `max_step`: clipping a trial point inside its callback would
return a value/gradient for a different point from the one requested by the
solver. Use a solver with explicit update clipping when a step-norm limit is
required. Bounds through `lower_bounds`/`upper_bounds` remain supported.

## Native JAX L-BFGS

```python
import jax.numpy as jnp
from pepsy.solvers import GradientOptimizer

result = GradientOptimizer(
    solver="jax-lbfgs", n_steps=100,
    options={"gtol": 1e-6, "memory_size": 10},
).run(
    params_init={"x": jnp.array([3., -2.])},
    loss_fn=lambda p: jnp.sum((p["x"] - .25) ** 2),
)
```

`jax-lbfgs` uses **Optax L-BFGS with zoom line search**. The existing `lbfgs`
alias continues to select SciPy. Parameters, quasi-Newton history, line-search
trials, stopping decisions and best-state tracking stay on the JAX device.
The runner JIT-compiles chunks of `scan_steps=50` updates and transfers only
scalar diagnostics and history at chunk boundaries. `run()` is a synchronous
Python entry point; it is not itself an outer-JIT-compatible or implicitly
differentiable solve. Every call initializes a fresh optimizer state, including
when targets or penalty multipliers change between calls. Compilation is local
to a solve; reuse across separate `run()` calls is not guaranteed.

Complex parameters use real/imaginary JAX coordinates. Objectives receive and
results retain each leaf's original shape, dtype and device, including mixed
real/complex mappings. Internal coordinates and quasi-Newton history use the
common real precision of the parameter leaves and the scalar loss. This keeps
Optax's line-search buffers from rounding a float64 loss to float32, and
supports float32 losses with float64 parameters. Mixed-precision problems may
therefore allocate higher-precision solver state. Enable JAX x64
before creating double-precision arrays. Initial parameters must be finite,
including leaves the objective does not use. Losses must be JAX-compatible real
scalars; the imaginary tolerance `1e-10` is checked inside line-search trials
as well as at the initial point. Nonfinite trial objectives can be rejected by
the search. A finite accepted objective with a nonfinite gradient raises
`FloatingPointError`.

| Option | Meaning/default |
| --- | --- |
| `memory_size` | Positive integer curvature-history length, default 10 |
| `gtol` | Euclidean gradient norm stopping threshold, default `1e-6` |
| `ftol` | Relative successive-loss threshold, default 0 (disabled) |
| `lr` | Optional positive scaling before line search; omitted uses Optax's `None` default |
| `scan_steps` | Positive integer compiled chunk size, default 50 |
| `linesearch_options` | Keyword mapping for `optax.scale_by_zoom_linesearch`; defaults to 20 trials and `initial_guess_strategy="one"` |
| `scale_init_precond` | Forwarded to Optax, default `True` |

`n_steps` and `maxiter`/`maxeval`/`its_max` limit outer iterations, not trial
evaluations. `n_evals` counts the initial objective/gradient and every
line-search trial; accepted values/gradients are cached rather than recomputed.
`history` records each visited pre-update loss, while `final_loss` describes
the returned iterate, including the final accepted update. If several accepted
points share the same best eligible loss, restoration chooses the latest one;
equal losses do not reset `patience`. This retains gradient progress even when
a large constant offset rounds successive loss values to the same number.
Reasons include
`gradient_tol`, `ftol`, `patience`, `maxiter`, `invalid_objective`, and
`linesearch_failed`. A failed search retains a finite nonincreasing candidate
when available and stops; it does not silently continue with bad curvature.
`restore_best`, `min_steps`, `patience`, `min_improve`, `assume_nonnegative`
and `best_neg_tol` apply. An invalid initial objective stops immediately;
`bad_max` and penalty substitution do not restart quasi-Newton history.
Use `assume_nonnegative=False` (the default) for signed augmented-Lagrangian
objectives. Bounds, angle wrapping, gradient clipping and update clipping are
unsupported because they change the line-search problem.

This adds an unconstrained solver for user-supplied objectives. Local-density
constraints are not automatically added to an MPO loss; use
[`compress_mpo_dmt`](../operators/mpo_dmt.md) for direct protected SVD.
