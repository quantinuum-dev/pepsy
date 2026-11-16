"""Device-resident Optax L-BFGS with real coordinates and cached line searches."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def run_jax_lbfgs(params_init, loss_fn, *, lr, solver_options, n_steps,
                  log_every, progress, opt_desc, progress_callback):
    """Run compiled chunks; only scalar diagnostics/history cross to the host."""
    from tqdm.auto import tqdm

    from .gradient import _pop_common_controls, _require_jax, _require_optax

    jax, jnp = _require_jax()
    from jax.experimental import checkify

    optax = _require_optax()
    for capability in ("lbfgs", "scale_by_zoom_linesearch", "tree"):
        if not hasattr(optax, capability):
            raise ImportError(f"jax-lbfgs requires Optax with {capability} support")
    if not callable(getattr(optax.tree, "get", None)):
        raise ImportError("jax-lbfgs requires Optax with tree.get support")
    options = dict(solver_options)
    unsupported = [key for key in (
        "bounds", "lower_bounds", "upper_bounds", "max_step", "max_step_norm",
        "clip_grad_norm", "grad_clip_norm",
    ) if options.get(key) is not None]
    if options.get("angle_wrap", False):
        unsupported.append("angle_wrap")
    if unsupported:
        raise ValueError("jax-lbfgs does not support " + ", ".join(unsupported))
    for key in ("bounds", "lower_bounds", "upper_bounds", "max_step", "max_step_norm",
                "clip_grad_norm", "grad_clip_norm"):
        options.pop(key, None)
    controls = _pop_common_controls(
        options, default_steps=n_steps, default_bad_max=1, default_penalty=1e20,
    )
    scan_steps = _positive_integer(options.pop("scan_steps", 50), "scan_steps")
    memory_size = _positive_integer(options.pop("memory_size", 10), "memory_size")
    gtol = float(options.pop("gtol", 1e-6))
    ftol = float(options.pop("ftol", 0.0))
    for name, value in (("gtol", gtol), ("ftol", ftol)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    linesearch_options = dict(options.pop("linesearch_options", {}))
    linesearch_options.setdefault("max_linesearch_steps", 20)
    linesearch_options["max_linesearch_steps"] = _positive_integer(
        linesearch_options["max_linesearch_steps"], "max_linesearch_steps",
    )
    linesearch_options.setdefault("initial_guess_strategy", "one")
    optimizer = optax.lbfgs(
        learning_rate=lr, memory_size=memory_size,
        linesearch=optax.scale_by_zoom_linesearch(**linesearch_options), **options,
    )
    params = {name: jnp.asarray(value) for name, value in params_init.items()}
    if not params:
        return params, [], float("nan"), float("nan"), "empty_params", 0
    if any(not jnp.issubdtype(x.dtype, jnp.inexact) for x in params.values()):
        raise TypeError("jax-lbfgs requires floating-point or complex parameters")
    devices = {device for x in params.values() for device in x.devices()}
    if len(devices) != 1:
        raise ValueError("jax-lbfgs parameters must share one device")

    # A real pytree avoids complex secant/conjugation conventions and retains
    # each leaf's precision without a promoted, host-packed flat vector.
    complex_keys = {name for name, x in params.items() if jnp.iscomplexobj(x)}
    coordinates = {
        name: (x.real, x.imag) if name in complex_keys else (x,)
        for name, x in params.items()
    }

    def unpack(p):
        return {name: jax.lax.complex(*x) if name in complex_keys else x[0]
                for name, x in p.items()}

    def objective(p):
        value = jnp.asarray(loss_fn(unpack(p)))
        if value.ndim != 0:
            raise ValueError("loss_fn must return a scalar")
        # checkify also records violations in rejected line-search trials,
        # without a Python callback or device synchronization per trial.
        checkify.check(jnp.isfinite(value.imag) & (jnp.abs(value.imag) <= 1e-10),
                       "loss_fn returned a complex loss; expected a real scalar")
        return value.real

    value_grad = jax.value_and_grad(objective)

    def checked_error(error):
        message = error.get()
        if message is not None:
            raise ValueError(message)

    error, (value, grad) = jax.jit(checkify.checkify(value_grad))(coordinates)
    checked_error(error)

    def finite_tree(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                                  for x in jax.tree.leaves(tree)]))

    if np.isfinite(float(value)) and not bool(finite_tree(grad)):
        raise FloatingPointError("jax-lbfgs encountered a nonfinite gradient")
    if not np.isfinite(float(value)):
        return params, [float(value)], float("inf"), float(value), "invalid_objective", 1

    def eligible(v):
        return jnp.isfinite(v) & (
            (v >= -controls["best_neg_tol"]) if controls["assume_nonnegative"] else True
        )

    # carry: point, optimizer, cached value/gradient, best point/value,
    # last significant improvement, completed steps, stop code, eval count.
    carry = (coordinates, optimizer.init(coordinates), value, grad, coordinates,
             jnp.where(eligible(value), value, jnp.inf), jnp.int32(0),
             jnp.int32(0), jnp.int32(0), jnp.int32(1))

    def chunk(carry):
        def scan_step(carry, _):
            def perform(c):
                x, state, v, g, best_x, best_v, last, step, _, evaluations = c
                step = step + 1
                gnorm = optax.global_norm(g)
                converged = (gnorm <= gtol) & (step >= controls["min_steps"])

                def update(_):
                    updates, new_state = optimizer.update(
                        g, state, x, value=v, grad=g, value_fn=objective,
                    )
                    new_x = optax.apply_updates(x, updates)
                    # Optax zoom caches the value and derivative at exactly
                    # new_x, including its fallback when Wolfe conditions fail.
                    new_v = optax.tree.get(new_state, "value")
                    new_g = optax.tree.get(new_state, "grad")
                    info = optax.tree.get(new_state, "info")
                    valid_grad = finite_tree(new_g)
                    finite = jnp.isfinite(new_v) & finite_tree(new_x)
                    accept = finite & valid_grad & (new_v <= v)
                    improved = accept & eligible(new_v) & (new_v < best_v)
                    significant = improved & (new_v < best_v - controls["min_improve"])
                    bx = jax.tree.map(lambda a, b: jnp.where(improved, a, b), new_x, best_x)
                    bv = jnp.where(improved, new_v, best_v)
                    last_new = jnp.where(significant, step, last)
                    # A curvature failure can still supply a decreasing point;
                    # retain that point but expose the unsuccessful search.
                    failed = (info.decrease_error > 0) | (info.curvature_error > 0) | ~accept
                    stop = jnp.where(failed, 3, 0)
                    if ftol > 0:
                        close = jnp.abs(v - new_v) <= ftol * jnp.maximum(1., jnp.abs(v))
                        # A zero fallback step after a failed search is not
                        # loss convergence when its derivative is still large.
                        stop = jnp.where(close & accept & ~failed, 2, stop)
                    stop = jnp.where((optax.global_norm(new_g) <= gtol) & accept, 1, stop)
                    stop = jnp.where((step < controls["min_steps"]) & (stop <= 2), 0, stop)
                    if controls["patience"] is not None:
                        stale = (step - last_new >= controls["patience"])
                        stop = jnp.where(stale & (step >= controls["min_steps"]) & (stop == 0), 4, stop)
                    stop = jnp.where(finite & ~valid_grad, 5, stop)
                    xx = jax.tree.map(lambda a, b: jnp.where(accept, a, b), new_x, x)
                    gg = jax.tree.map(lambda a, b: jnp.where(accept, a, b), new_g, g)
                    return (xx, new_state, jnp.where(accept, new_v, v), gg, bx, bv,
                            last_new, step, stop.astype(step.dtype),
                            evaluations + info.num_linesearch_steps.astype(evaluations.dtype))

                out = jax.lax.cond(
                    converged,
                    lambda _: (x, state, v, g, best_x, best_v, last, step, jnp.int32(1), evaluations),
                    update, None,
                )
                return out, (v, jnp.bool_(True))

            return jax.lax.cond(
                (carry[8] == 0) & (carry[7] < controls["max_steps"]), perform,
                lambda c: (c, (c[2], jnp.bool_(False))), carry,
            )

        return jax.lax.scan(scan_step, carry, None, length=scan_steps)

    compiled_chunk = jax.jit(checkify.checkify(chunk))
    history = []
    bar = tqdm(total=controls["max_steps"], desc=opt_desc or "jax-lbfgs",
               disable=not progress, leave=False)
    try:
        while True:
            error, (carry, (values, active)) = compiled_chunk(carry)
            checked_error(error)
            values, active, stop, steps = jax.device_get((values, active, carry[8], carry[7]))
            for measured in values[active]:
                history.append(float(measured))
                if progress_callback is not None and len(history) % log_every == 0:
                    progress_callback(len(history), float(measured))
            bar.update(int(active.sum()))
            if stop or steps >= controls["max_steps"]:
                break
    finally:
        bar.close()
    if stop == 5:
        raise FloatingPointError("jax-lbfgs encountered a nonfinite gradient")
    x, _, value, _, best_x, best_value, *_ = carry
    best_loss, final_loss = float(best_value), float(value)
    if controls["restore_best"] and np.isfinite(best_loss):
        x, final_loss = best_x, best_loss
    reason = {0: "maxiter", 1: "gradient_tol", 2: "ftol",
              3: "linesearch_failed", 4: "patience"}[int(stop)]
    return unpack(x), history, best_loss, final_loss, reason, int(carry[9])
