# How-To: Tune Sweep Solvers

Use `SweepOptimizer.optimize_axis(...)` or `optimize_global(...)` with `solver` and `solver_options`.

## Quick comparison

- `solver="torch-adam"`: robust default for noisy or sensitive objectives.
- `solver="scipy"`: robust CPU `L-BFGS-B`, supports bounds.
- `solver="nlopt"`: flexible stopping controls and algorithm variants.

## Recipe: Torch Adam

```python
result = sweeper.optimize_global(
    axes=("y", "x"),
    n_cycles=1,
    n_round_trips=1,
    solver="torch-adam",
    solver_options={
        "n_steps": 60,
        "lr": 1e-2,
        "scheduler": "cosine",
        "clip_grad_norm": 1.0,
    },
    env_n_iter=4,
    progress=True,
)
```

Practical notes:

- This path runs in torch with native autograd tensors.
- For unstable runs, lower `lr` or add gradient clipping.

## Recipe: SciPy L-BFGS-B

```python
result = sweeper.optimize_global(
    axes=("y", "x"),
    n_cycles=1,
    n_round_trips=1,
    solver="scipy",
    solver_options={
        "method": "L-BFGS-B",
        "n_steps": 80,
        "maxiter": 80,
        "ftol": 1e-12,
    },
    env_n_iter=4,
    progress=True,
)
```

Practical notes:

- This backend optimizes on CPU `float64` vectors internally.
- You can set bounds with either `bounds=[(lo, hi), ...]` or `lower_bounds` / `upper_bounds`.

## Recipe: NLopt LBFGS

```python
result = sweeper.optimize_global(
    axes=("y", "x"),
    n_cycles=1,
    n_round_trips=1,
    solver="nlopt",
    solver_options={
        "algorithm": "LD_LBFGS",
        "n_steps": 100,
        "maxeval": 100,
        "ftol_rel": 1e-10,
        "xtol_rel": 1e-10,
    },
    env_n_iter=4,
    progress=True,
)
```

Practical notes:

- `maxeval` is the main NLopt iteration cap.
- If needed, switch `algorithm` to another NLopt gradient method.
- This path is more sensitive to stopping settings; for a robust default start from `scipy`.

## Best-parameter behavior

All sweep solvers now return and apply the **best-loss parameters** seen during the run, not just the last iterate. This is important for non-monotonic trajectories.

## Numerical stability and backend transitions

- External solvers (`scipy`, `nlopt`) flatten params to CPU NumPy `float64`.
- Complex parameters are split into real/imag blocks during optimization and reconstructed afterward.
- Returned parameters are cast back to original tensor dtype/device before being applied to the PEPS state.

This design keeps optimizer interoperability while minimizing conversion-induced drift.
