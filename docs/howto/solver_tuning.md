# How-To: Tune Sweep Solvers

Use `SweepOptimizer.optimize_axis(...)` or `optimize_global(...)` with `solver` and `solver_options`.

## Quick comparison

- `solver="adam"`: best default for noisy or sensitive objectives.
- `solver="lbfgs"`: torch-native quasi-Newton, often faster near convergence.
- `solver="scipy-lbfgs"`: robust CPU `L-BFGS-B`, supports bounds.
- `solver="nlopt-lbfgs"`: flexible stopping controls and algorithm variants.

## Recipe: LBFGS (Torch)

```python
result = sweeper.optimize_global(
    axes=("y", "x"),
    n_cycles=1,
    n_round_trips=1,
    solver="lbfgs",
    solver_options={
        "algorithm": "LBFGS",
        "n_steps": 60,
        "max_iter": 1,      # one inner LBFGS step per outer sweep step
        "history_size": 20,
        "line_search_fn": "strong_wolfe",
        "lr": 1.0,
    },
    env_n_iter=4,
    progress=True,
)
```

Practical notes:

- Keep `max_iter=1` so `n_steps` remains the main outer control knob.
- Start with `lr=1.0`; reduce if updates oscillate.

## Recipe: SciPy L-BFGS-B

```python
result = sweeper.optimize_global(
    axes=("y", "x"),
    n_cycles=1,
    n_round_trips=1,
    solver="scipy-lbfgs",
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
    solver="nlopt-lbfgs",
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
- This path is more sensitive to stopping settings; for a robust default start from `scipy-lbfgs`.

## Best-parameter behavior

All sweep solvers now return and apply the **best-loss parameters** seen during the run, not just the last iterate. This is important for non-monotonic trajectories.

## Numerical stability and backend transitions

- External solvers (`scipy-lbfgs`, `nlopt-lbfgs`) flatten params to CPU NumPy `float64`.
- Complex parameters are split into real/imag blocks during optimization and reconstructed afterward.
- Returned parameters are cast back to original tensor dtype/device before being applied to the PEPS state.

This design keeps optimizer interoperability while minimizing conversion-induced drift.
