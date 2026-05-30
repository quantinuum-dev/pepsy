"""Finite-difference solvers extracted from :mod:`pepsy.solvers.gradient`.

This module keeps FD-specific logic (fd-adam, fd-scipy, fd-nlopt)
separate for clarity, while reusing shared packing/utils from
``pepsy.solvers.gradient``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm.auto import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from .gradient import (
    _SCIPY_BOUNDS_METHODS,
    _SCIPY_HESS_METHODS,
    _SCIPY_HESSP_METHODS,
    _accept_as_best,
    _as_nonnegative_float,
    _as_numpy_vector,
    _assign_flat_params,
    _build_param_specs,
    _flatten_params_real_numpy,
    _format_pbar_postfix,
    _iter_steps,
    _pop_common_controls,
    _resolve_bounds_arrays,
    _scalar_real_loss,
    _normalize_scipy_method,
)

def _evaluate_loss_only(
    vector: np.ndarray,
    params_run: dict[str, torch.Tensor],
    specs: list[dict[str, Any]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> float:
    """Evaluate scalar loss at *vector* without autograd/backpropagation."""
    _assign_flat_params(vector, specs)
    with torch.no_grad():
        loss = _scalar_real_loss(loss_fn(params_run))
    return float(loss.detach().cpu())

def _normalize_fd_method(method: Any) -> str:
    """Normalize finite-difference method name."""
    if not isinstance(method, str):
        raise TypeError("fd_method must be a string")
    key = method.strip().lower()
    if key not in {"central", "forward"}:
        raise ValueError("fd_method must be 'central' or 'forward'")
    return key

def _evaluate_fd_gradient(
    vector: np.ndarray,
    *,
    evaluate_loss: Callable[[np.ndarray], float],
    fd_eps: float,
    fd_method: str,
    f0: float | None = None,
    workspace: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``(grad, f0)`` where grad is finite-difference approximation.

    The implementation reuses one (forward) or two (central) work buffers
    instead of allocating fresh copies at every coordinate perturbation.
    This substantially reduces Python/NumPy allocation overhead for high
    dimensional FD problems.
    """
    if workspace is None:
        base = np.array(vector, dtype=np.float64, copy=True)
    else:
        base = workspace.get("base")
        vec = np.asarray(vector, dtype=np.float64)
        if base is None or base.shape != vec.shape:
            base = np.array(vec, dtype=np.float64, copy=True)
            workspace["base"] = base
        else:
            base[...] = vec
    n = int(base.size)
    grad = np.empty(n, dtype=np.float64)

    if f0 is None:
        f0 = float(evaluate_loss(base))

    if fd_method == "forward":
        if workspace is None:
            x_plus = np.array(base, dtype=np.float64, copy=True)
        else:
            x_plus = workspace.get("x_plus")
            if x_plus is None or x_plus.shape != base.shape:
                x_plus = np.array(base, dtype=np.float64, copy=True)
                workspace["x_plus"] = x_plus
            else:
                x_plus[...] = base
        inv_eps = 1.0 / fd_eps
        for i in range(n):
            orig = base[i]
            x_plus[i] += fd_eps
            fp = float(evaluate_loss(x_plus))
            grad[i] = (fp - f0) * inv_eps
            x_plus[i] = orig
        return grad, float(f0)

    if workspace is None:
        x_plus = np.array(base, dtype=np.float64, copy=True)
        x_minus = np.array(base, dtype=np.float64, copy=True)
    else:
        x_plus = workspace.get("x_plus")
        x_minus = workspace.get("x_minus")
        if x_plus is None or x_plus.shape != base.shape:
            x_plus = np.array(base, dtype=np.float64, copy=True)
            workspace["x_plus"] = x_plus
        else:
            x_plus[...] = base
        if x_minus is None or x_minus.shape != base.shape:
            x_minus = np.array(base, dtype=np.float64, copy=True)
            workspace["x_minus"] = x_minus
        else:
            x_minus[...] = base
    inv_2eps = 0.5 / fd_eps
    for i in range(n):
        orig = base[i]
        x_plus[i] += fd_eps
        x_minus[i] -= fd_eps
        fp = float(evaluate_loss(x_plus))
        fm = float(evaluate_loss(x_minus))
        grad[i] = (fp - fm) * inv_2eps
        x_plus[i] = orig
        x_minus[i] = orig
    return grad, float(f0)

def _run_fd_adam(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    lr: float,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    """Run Adam using finite-difference gradients over the packed vector."""
    params_run = dict(items)
    specs = _build_param_specs(items)
    x = _flatten_params_real_numpy(specs)
    if x.size == 0:
        return params_run, [], float("nan"), float("nan"), "empty_params", 0

    options = dict(solver_options)
    # Strip keys handled by other backends.
    for key in (
        "maxeval",
        "method",
        "algorithm",
        "optimizer",
        "ftol",
        "ftol_rel",
        "ftol_abs",
        "gtol",
        "xtol_rel",
        "xtol_abs",
        "stopval",
        "scheduler",
        "eta_min",
        "step_size",
        "gamma",
        "plateau_patience",
        "plateau_factor",
        "plateau_threshold",
        "plateau_cooldown",
        "line_search_max_steps",
        "maxls",
        "ema_alpha",
    ):
        options.pop(key, None)

    controls = _pop_common_controls(
        options,
        default_steps=n_steps,
        default_bad_max=20,
        default_penalty=1e20,
    )

    bounds = options.pop("bounds", None)
    lower = options.pop("lower_bounds", None)
    upper = options.pop("upper_bounds", None)
    grad_clip_norm = options.pop("grad_clip_norm", options.pop("clip_grad_norm", None))
    max_step_norm = options.pop("max_step_norm", options.pop("max_step", None))
    fd_eps = float(options.pop("fd_eps", 1e-5))
    fd_method = _normalize_fd_method(options.pop("fd_method", "central"))
    betas = options.pop("betas", (0.9, 0.999))
    eps_adam = float(options.pop("eps", 1e-8))

    unknown = sorted(options)
    if unknown:
        raise ValueError(
            "Unsupported fd-adam solver_options keys: "
            + ", ".join(unknown)
        )

    if not np.isfinite(fd_eps) or fd_eps <= 0.0:
        raise ValueError("fd_eps must be finite and > 0")
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when set")
    if max_step_norm is not None and float(max_step_norm) <= 0.0:
        raise ValueError("max_step_norm must be > 0 when set")
    if not np.isfinite(eps_adam) or eps_adam <= 0.0:
        raise ValueError("eps must be finite and > 0")
    if (
        not isinstance(betas, (tuple, list))
        or len(betas) != 2
    ):
        raise ValueError("betas must be a tuple/list of length 2")
    beta1 = float(betas[0])
    beta2 = float(betas[1])
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("betas values must be in [0, 1)")

    lower_arr, upper_arr = _resolve_bounds_arrays(
        bounds=bounds,
        lower=lower,
        upper=upper,
        size=int(x.size),
    )
    if lower_arr is not None and upper_arr is not None:
        x = np.clip(x, lower_arr, upper_arr)

    history: list[float] = []
    best_loss = float("inf")
    best_vector = np.array(x, dtype=np.float64, copy=True)
    convergence_reason = "maxiter"
    bad_consecutive = 0
    last_improve_step = 0
    eval_counter = {"value": 0}
    last_step_evals = 0
    fd_workspace: dict[str, np.ndarray] = {}
    need_grad_norm = (grad_clip_norm is not None) or progress

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    step_iter = _iter_steps(controls["max_steps"], progress=progress, opt_desc=opt_desc)

    def _eval_loss(vec: np.ndarray) -> float:
        eval_counter["value"] += 1
        return _evaluate_loss_only(vec, params_run, specs, loss_fn)

    for step in step_iter:
        step_num = step + 1
        try:
            grad, f0 = _evaluate_fd_gradient(
                x,
                evaluate_loss=_eval_loss,
                fd_eps=fd_eps,
                fd_method=fd_method,
                workspace=fd_workspace,
            )
            if not np.isfinite(f0) or not np.isfinite(grad).all():
                raise FloatingPointError("non-finite objective or gradient")
        except (FloatingPointError, RuntimeError, ValueError):
            bad_consecutive += 1
            loss_value = controls["penalty_on_bad"]
            history.append(loss_value)
            ev_this = eval_counter["value"] - last_step_evals
            last_step_evals = eval_counter["value"]
            if progress:
                step_iter.set_postfix(_format_pbar_postfix(
                    step=step_num,
                    evals=eval_counter["value"],
                    ev_per_step=ev_this,
                    gnorm="nan",
                    loss="nan",
                    best=best_loss,
                ))
            if progress_callback is not None and (step_num % log_every == 0):
                progress_callback(step_num, loss_value)
            if (bad_consecutive >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
                convergence_reason = "bad_max"
                break
            continue

        bad_consecutive = 0
        loss_value = float(f0)
        grad_norm = float(np.linalg.norm(grad)) if need_grad_norm else float("nan")
        if grad_clip_norm is not None and grad_norm > float(grad_clip_norm):
            grad = grad * (float(grad_clip_norm) / (grad_norm + 1e-12))

        if (
            _accept_as_best(loss_value, controls)
            and (loss_value + controls["min_improve"] < best_loss)
        ):
            best_loss = loss_value
            best_vector = np.array(x, dtype=np.float64, copy=True)
            last_improve_step = step_num

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** step_num)
        v_hat = v / (1.0 - beta2 ** step_num)
        delta = -lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        if max_step_norm is not None:
            dnorm = float(np.linalg.norm(delta))
            if dnorm > float(max_step_norm):
                delta = delta * (float(max_step_norm) / (dnorm + 1e-12))

        x = x + delta

        if lower_arr is not None and upper_arr is not None:
            x = np.clip(x, lower_arr, upper_arr)

        if controls["angle_wrap"]:
            x = (x + np.pi) % (2.0 * np.pi) - np.pi

        history.append(loss_value)
        ev_this = eval_counter["value"] - last_step_evals
        last_step_evals = eval_counter["value"]
        if progress:
            step_iter.set_postfix(_format_pbar_postfix(
                step=step_num,
                evals=eval_counter["value"],
                ev_per_step=ev_this,
                gnorm=grad_norm,
                loss=loss_value,
                best=best_loss,
            ))
        if progress_callback is not None and (step_num % log_every == 0):
            progress_callback(step_num, loss_value)
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - last_improve_step) >= controls["patience"]:
                convergence_reason = "patience"
                break

    use_best = controls["restore_best"] and np.isfinite(best_loss)
    vector_out = best_vector if use_best else x
    _assign_flat_params(vector_out, specs)

    if not history:
        try:
            history.append(_eval_loss(vector_out))
        except (RuntimeError, ValueError, FloatingPointError):
            history.append(controls["penalty_on_bad"])

    final_loss = best_loss if use_best else history[-1]
    if progress_callback is not None:
        progress_callback(max(1, len(history)), float(final_loss))

    return params_run, history, float(best_loss), float(final_loss), convergence_reason, eval_counter["value"]

def _run_fd_scipy(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    """Run SciPy optimize.minimize using finite-difference gradients."""
    try:
        from scipy import optimize as sp_opt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "solver='fd-scipy' requires SciPy. Install with: pip install scipy"
        ) from exc

    params_run = dict(items)
    specs = _build_param_specs(items)
    x0 = _flatten_params_real_numpy(specs)
    if x0.size == 0:
        return params_run, [], float("nan"), float("nan"), "empty_params", 0

    options = dict(solver_options)
    # maxeval is an nlopt concept; discard it so it doesn't override n_steps for scipy.
    options.pop("maxeval", None)
    controls = _pop_common_controls(
        options,
        default_steps=n_steps,
        default_bad_max=20,
        default_penalty=1e20,
    )
    bounds = options.pop("bounds", None)
    lower = options.pop("lower_bounds", None)
    upper = options.pop("upper_bounds", None)
    method_name = options.pop("method", None)
    algorithm_name = options.pop("algorithm", options.pop("optimizer", None))
    raw_name = algorithm_name or method_name or "L-BFGS-B"
    method = _normalize_scipy_method(raw_name, key="algorithm")
    if method in _SCIPY_HESS_METHODS or method in _SCIPY_HESSP_METHODS:
        raise ValueError(
            f"scipy method {method!r} is not supported for finite-difference solvers. "
            "Use methods such as L-BFGS-B, BFGS, CG, TNC, SLSQP, or trust-constr."
        )

    grad_clip_norm = options.pop("grad_clip_norm", options.pop("clip_grad_norm", None))
    fd_eps = float(options.pop("fd_eps", 1e-5))
    fd_method = _normalize_fd_method(options.pop("fd_method", "central"))
    if not np.isfinite(fd_eps) or fd_eps <= 0.0:
        raise ValueError("fd_eps must be finite and > 0")
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when set")

    if "ftol" in options:
        ftol = _as_nonnegative_float(options.pop("ftol"), key="ftol")
    else:
        ftol = _as_nonnegative_float(options.pop("ftol_rel", 1e-9), key="ftol_rel")

    if "gtol" in options:
        gtol = _as_nonnegative_float(options.pop("gtol"), key="gtol")
    else:
        gtol = _as_nonnegative_float(options.pop("xtol_rel", 1e-9), key="xtol_rel")

    if "line_search_max_steps" in options:
        options["maxls"] = int(options.pop("line_search_max_steps"))
    if "maxls" in options and int(options["maxls"]) <= 0:
        raise ValueError("maxls must be >= 1")

    options["maxiter"] = n_steps
    if method in {"L-BFGS-B", "TNC", "SLSQP"}:
        options.setdefault("ftol", ftol)
    if method == "Newton-CG":
        options.setdefault("xtol", gtol)
    else:
        options.setdefault("gtol", gtol)
    if method in {"L-BFGS-B", "TNC"}:
        options.setdefault("maxls", 40)

    if bounds is None and (lower is not None or upper is not None):
        n_vars = int(x0.size)
        lower_arr = _as_numpy_vector(-np.inf if lower is None else lower, n_vars, key="lower_bounds")
        upper_arr = _as_numpy_vector(np.inf if upper is None else upper, n_vars, key="upper_bounds")
        bounds = list(zip(lower_arr, upper_arr))

    if bounds is not None and method not in _SCIPY_BOUNDS_METHODS:
        warnings.warn(
            f"scipy method {method!r} does not support bounds; ignoring.",
            stacklevel=2,
        )
        bounds = None

    if controls["angle_wrap"] and bounds is None and method in _SCIPY_BOUNDS_METHODS:
        bounds = [(-np.pi, np.pi)] * int(x0.size)

    history: list[float] = []
    step_counter = {"value": 0}
    state = {
        "last_loss": None,
        "last_gnorm": float("nan"),
        "best_loss": float("inf"),
        "best_x": None,
        "last_improve_step": 0,
        "bad_consecutive": 0,
        "convergence_reason": "maxiter",
        "last_step_evals": 0,
    }
    eval_counter = {"value": 0}
    pbar = None
    if progress:
        if opt_desc and method.lower() in opt_desc.lower():
            _pbar_desc = opt_desc
        else:
            _pbar_desc = f"{opt_desc}[{method}]" if opt_desc else method
        pbar = tqdm(
            total=n_steps,
            desc=_pbar_desc,
            leave=False,
            colour="CYAN",
            dynamic_ncols=True,
        )
    fd_workspace: dict[str, np.ndarray] = {}
    need_grad_norm = (pbar is not None) or (grad_clip_norm is not None)

    def _eval_loss(vec: np.ndarray) -> float:
        eval_counter["value"] += 1
        return _evaluate_loss_only(vec, params_run, specs, loss_fn)

    def objective(x):
        x_vec = np.asarray(x, dtype=np.float64)
        try:
            grad_value, f0 = _evaluate_fd_gradient(
                x_vec,
                evaluate_loss=_eval_loss,
                fd_eps=fd_eps,
                fd_method=fd_method,
                workspace=fd_workspace,
            )
            loss_value = float(f0)
            if not np.isfinite(loss_value) or not np.isfinite(grad_value).all():
                raise FloatingPointError("non-finite objective or gradient")
        except (FloatingPointError, RuntimeError, ValueError):
            state["bad_consecutive"] += 1
            loss_value = controls["penalty_on_bad"]
            grad_value = np.zeros_like(x_vec)
        else:
            state["bad_consecutive"] = 0
            grad_norm = float(np.linalg.norm(grad_value)) if need_grad_norm else float("nan")
            state["last_gnorm"] = grad_norm
            if grad_clip_norm is not None:
                clip = float(grad_clip_norm)
                if grad_norm > clip:
                    grad_value = grad_value * (clip / (grad_norm + 1e-12))
            if (
                _accept_as_best(loss_value, controls)
                and (loss_value + controls["min_improve"] < state["best_loss"])
            ):
                state["best_loss"] = loss_value
                state["best_x"] = np.array(x_vec, dtype=np.float64, copy=True)
                state["last_improve_step"] = step_counter["value"] + 1
        state["last_loss"] = float(loss_value)
        if pbar is not None:
            ev_this = eval_counter["value"] - state["last_step_evals"]
            pbar.set_postfix(_format_pbar_postfix(
                step=step_counter["value"] + 1,
                evals=eval_counter["value"],
                ev_per_step=ev_this,
                gnorm=state["last_gnorm"],
                loss=loss_value,
                best=state["best_loss"],
            ))
        return loss_value, grad_value

    def callback(xk):
        step_counter["value"] += 1
        step_num = step_counter["value"]
        loss_value = state["last_loss"]
        if loss_value is None:
            try:
                loss_value = _eval_loss(np.asarray(xk, dtype=np.float64))
            except (RuntimeError, ValueError, FloatingPointError):
                loss_value = controls["penalty_on_bad"]
        history.append(float(loss_value))
        if pbar is not None:
            pbar.update(1)
            ev_this = eval_counter["value"] - state["last_step_evals"]
            state["last_step_evals"] = eval_counter["value"]
            pbar.set_postfix(_format_pbar_postfix(
                step=step_num,
                evals=eval_counter["value"],
                ev_per_step=ev_this,
                gnorm=state["last_gnorm"],
                loss=loss_value,
                best=state["best_loss"],
            ))
        if progress_callback is not None and (
            (step_num % log_every == 0) or (step_num == controls["max_steps"])
        ):
            progress_callback(step_num, float(loss_value))
        if (state["bad_consecutive"] >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
            state["convergence_reason"] = "bad_max"
            raise StopIteration
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - state["last_improve_step"]) >= controls["patience"]:
                state["convergence_reason"] = "patience"
                raise StopIteration

    result = None
    try:
        result = sp_opt.minimize(
            objective,
            x0,
            jac=True,
            method=method,
            bounds=bounds,
            options=options,
            callback=callback,
        )
    except StopIteration:
        pass

    if pbar is not None:
        pbar.close()

    if controls["restore_best"] and state["best_x"] is not None:
        best_x = state["best_x"]
    elif result is not None:
        best_x = np.asarray(result.x, dtype=np.float64)
    elif state["best_x"] is not None:
        best_x = state["best_x"]
    else:
        best_x = np.asarray(x0, dtype=np.float64)
    _assign_flat_params(best_x, specs)

    if not history:
        try:
            history.append(_eval_loss(best_x))
        except (RuntimeError, ValueError, FloatingPointError):
            history.append(controls["penalty_on_bad"])

    best_loss = state["best_loss"]
    if controls["restore_best"] and state["best_x"] is not None:
        final_loss = best_loss
    else:
        final_loss = history[-1] if history else float("nan")
    return params_run, history, best_loss, final_loss, state["convergence_reason"], eval_counter["value"]

def _run_fd_nlopt(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    """Run NLopt using finite-difference gradients."""
    try:
        import nlopt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "solver='fd-nlopt' requires NLopt. Install with: pip install nlopt"
        ) from exc

    params_run = dict(items)
    specs = _build_param_specs(items)
    x0 = _flatten_params_real_numpy(specs)
    if x0.size == 0:
        return params_run, [], float("nan"), float("nan"), "empty_params", 0

    options = dict(solver_options)
    if "algorithm" in options:
        algorithm_name = options.pop("algorithm")
    else:
        algorithm_name = options.pop("optimizer", "LD_LBFGS")
    method_name = options.pop("method", None)
    if method_name is not None and algorithm_name == "LD_LBFGS":
        algorithm_name = method_name

    options.setdefault("assume_nonnegative", True)
    controls = _pop_common_controls(
        options,
        default_steps=n_steps,
        default_bad_max=20,
        default_penalty=1e20,
    )
    maxeval = controls["max_steps"]
    assume_nonnegative = controls["assume_nonnegative"]
    best_neg_tol = controls["best_neg_tol"]
    ftol_rel = _as_nonnegative_float(options.pop("ftol_rel", options.pop("ftol", 1e-9)), key="ftol_rel")
    xtol_rel = _as_nonnegative_float(options.pop("xtol_rel", options.pop("gtol", 1e-9)), key="xtol_rel")
    ftol_abs = _as_nonnegative_float(options.pop("ftol_abs", 1e-9), key="ftol_abs")
    lower = options.pop("lower_bounds", None)
    upper = options.pop("upper_bounds", None)
    max_step = options.pop("max_step", None)
    grad_clip_norm = options.pop("grad_clip_norm", None)
    ema_alpha = options.pop("ema_alpha", None)
    fd_eps = float(options.pop("fd_eps", 1e-5))
    fd_method = _normalize_fd_method(options.pop("fd_method", "central"))

    if not np.isfinite(fd_eps) or fd_eps <= 0.0:
        raise ValueError("fd_eps must be finite and > 0")
    if ema_alpha is not None and not 0.0 < float(ema_alpha) <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1] when set")
    if max_step is not None and float(max_step) <= 0.0:
        raise ValueError("max_step must be > 0 when set")
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when set")

    opt_map = {
        "L_BFGS_B": "LD_LBFGS",
        "L_BFGS": "LD_LBFGS",
        "LBFGS": "LD_LBFGS",
        "LD_LBFGS": "LD_LBFGS",
        "TNEWTON": "LD_TNEWTON",
        "LD_TNEWTON": "LD_TNEWTON",
        "TNEWTON_RESTART": "LD_TNEWTON_RESTART",
        "LD_TNEWTON_RESTART": "LD_TNEWTON_RESTART",
        "TNEWTON_PRECOND": "LD_TNEWTON_PRECOND",
        "LD_TNEWTON_PRECOND": "LD_TNEWTON_PRECOND",
        "TNEWTON_PRECOND_RESTART": "LD_TNEWTON_PRECOND_RESTART",
        "LD_TNEWTON_PRECOND_RESTART": "LD_TNEWTON_PRECOND_RESTART",
        "LD_VAR1": "LD_VAR1",
        "VAR1": "LD_VAR1",
        "LD_VAR2": "LD_VAR2",
        "VAR2": "LD_VAR2",
        "MMA": "LD_MMA",
        "LD_MMA": "LD_MMA",
        "CCSAQ": "LD_CCSAQ",
        "LD_CCSAQ": "LD_CCSAQ",
        "SLSQP": "LD_SLSQP",
        "LD_SLSQP": "LD_SLSQP",
        "COBYLA": "LN_COBYLA",
        "LN_COBYLA": "LN_COBYLA",
    }
    if isinstance(algorithm_name, str):
        normalized = algorithm_name.strip().upper().replace("-", "_")
        normalized = opt_map.get(normalized, normalized)
        if not hasattr(nlopt, normalized):
            raise ValueError(f"Unknown NLopt algorithm: {algorithm_name!r}")
        algorithm = getattr(nlopt, normalized)
    else:
        algorithm = algorithm_name

    opt = nlopt.opt(algorithm, int(x0.size))
    opt.set_maxeval(maxeval)
    opt.set_ftol_rel(ftol_rel)
    opt.set_ftol_abs(ftol_abs)
    opt.set_xtol_rel(xtol_rel)

    if lower is not None:
        opt.set_lower_bounds(_as_numpy_vector(lower, int(x0.size), key="lower_bounds"))
    if upper is not None:
        opt.set_upper_bounds(_as_numpy_vector(upper, int(x0.size), key="upper_bounds"))

    if controls["angle_wrap"] and lower is None and upper is None:
        opt.set_lower_bounds(np.full(int(x0.size), -np.pi))
        opt.set_upper_bounds(np.full(int(x0.size), np.pi))

    _lo = np.asarray(opt.get_lower_bounds(), dtype=np.float64)
    _hi = np.asarray(opt.get_upper_bounds(), dtype=np.float64)
    if _lo.size == x0.size and _hi.size == x0.size:
        x0 = np.clip(x0, _lo, _hi)

    setters = {
        "stopval": opt.set_stopval,
        "ftol_abs": opt.set_ftol_abs,
        "xtol_abs": lambda val: opt.set_xtol_abs(_as_numpy_vector(val, int(x0.size), key="xtol_abs")),
    }
    unknown = set(options) - set(setters)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported nlopt solver_options keys: {keys}")
    for key, value in options.items():
        setters[key](value)

    history: list[float] = []
    eval_state = {
        "best_true": float("inf"),
        "best_x_true": None,
        "best_sel": float("inf"),
        "best_x_sel": None,
        "ema": None,
        "last_gnorm": float("nan"),
        "last_improve_eval": 0,
        "evals": 0,
        "bad_consecutive": 0,
        "prev_x": None,
        "stopped_reason": "maxeval",
        "step": 0,
        "last_loss_evals": 0,
    }
    loss_eval_counter = {"value": 0}
    fd_workspace: dict[str, np.ndarray] = {}

    pbar = None
    pbar_step_size = max(1, maxeval // max(1, n_steps))
    if progress:
        _nlopt_alg_name = algorithm_name if isinstance(algorithm_name, str) else f"nlopt({int(algorithm)})"
        if opt_desc and (
            (isinstance(_nlopt_alg_name, str) and _nlopt_alg_name.lower() in opt_desc.lower())
        ):
            _pbar_desc = opt_desc
        else:
            _pbar_desc = f"{opt_desc}[{_nlopt_alg_name}]" if opt_desc else _nlopt_alg_name
        pbar = tqdm(total=n_steps, desc=_pbar_desc, leave=False, colour="CYAN", dynamic_ncols=True)
    need_grad_norm = (pbar is not None) or (grad_clip_norm is not None)

    def _pbar_advance(evals):
        if pbar is None:
            return 0
        new_pos = min(n_steps, evals // pbar_step_size)
        advance = new_pos - pbar.n
        if advance > 0:
            pbar.update(advance)
        return advance

    def _eval_loss(vec: np.ndarray) -> float:
        loss_eval_counter["value"] += 1
        return _evaluate_loss_only(vec, params_run, specs, loss_fn)

    def objective(x, grad):
        eval_state["evals"] += 1
        x_vec = np.asarray(x, dtype=np.float64)

        if not np.isfinite(x_vec).all():
            if grad.size > 0:
                grad[:] = 0.0
            history.append(controls["penalty_on_bad"])
            eval_state["stopped_reason"] = "nan_x"
            opt.force_stop()
            if pbar is not None:
                advanced = _pbar_advance(eval_state["evals"])
                if advanced > 0:
                    eval_state["step"] += advanced
                pbar.set_postfix(_format_pbar_postfix(
                    step=eval_state["step"],
                    evals=loss_eval_counter["value"],
                    ev_per_step=0,
                    gnorm="nan",
                    loss="nan",
                    best=eval_state["best_true"],
                ))
            return controls["penalty_on_bad"]

        if max_step is not None:
            if eval_state["prev_x"] is not None:
                delta = x_vec - eval_state["prev_x"]
                delta_norm = float(np.linalg.norm(delta))
                step_limit = float(max_step)
                if delta_norm > step_limit:
                    x_vec = eval_state["prev_x"] + delta * (step_limit / (delta_norm + 1e-12))
            eval_state["prev_x"] = np.array(x_vec, dtype=np.float64, copy=True)

        try:
            grad_value, f0 = _evaluate_fd_gradient(
                x_vec,
                evaluate_loss=_eval_loss,
                fd_eps=fd_eps,
                fd_method=fd_method,
                workspace=fd_workspace,
            )
            loss_value = float(f0)
            if not np.isfinite(loss_value) or not np.isfinite(grad_value).all():
                raise FloatingPointError("non-finite objective or gradient")
        except (FloatingPointError, RuntimeError, ValueError):
            eval_state["bad_consecutive"] += 1
            loss_value = controls["penalty_on_bad"]
            if grad.size > 0:
                grad[:] = 0.0
            if (
                eval_state["bad_consecutive"] >= controls["bad_max"]
                and eval_state["evals"] >= controls["min_steps"]
            ):
                eval_state["stopped_reason"] = "bad_max"
                opt.force_stop()
        else:
            eval_state["bad_consecutive"] = 0
            grad_norm = float(np.linalg.norm(grad_value)) if need_grad_norm else float("nan")
            eval_state["last_gnorm"] = grad_norm
            if grad_clip_norm is not None:
                clip = float(grad_clip_norm)
                if grad_norm > clip:
                    grad_value = grad_value * (clip / (grad_norm + 1e-12))
            if grad.size > 0:
                grad[:] = grad_value

            if (not assume_nonnegative) or (loss_value >= -best_neg_tol):
                if loss_value < eval_state["best_true"]:
                    eval_state["best_true"] = loss_value
                    eval_state["best_x_true"] = np.array(x_vec, dtype=np.float64, copy=True)

            if ema_alpha is not None or controls["patience"] is not None:
                selection_value = loss_value
                if ema_alpha is not None:
                    alpha = float(ema_alpha)
                    if eval_state["ema"] is None:
                        eval_state["ema"] = selection_value
                    else:
                        eval_state["ema"] = (1.0 - alpha) * float(eval_state["ema"]) + alpha * selection_value
                    selection_value = float(eval_state["ema"])

                if selection_value + controls["min_improve"] < eval_state["best_sel"]:
                    eval_state["best_sel"] = selection_value
                    eval_state["best_x_sel"] = np.array(x_vec, dtype=np.float64, copy=True)
                    eval_state["last_improve_eval"] = eval_state["evals"]

                if controls["patience"] is not None and eval_state["evals"] >= controls["min_steps"]:
                    if (eval_state["evals"] - eval_state["last_improve_eval"]) >= controls["patience"]:
                        eval_state["stopped_reason"] = "patience"
                        opt.force_stop()

        history.append(float(loss_value))
        step_num = len(history)
        if pbar is not None:
            advanced = _pbar_advance(eval_state["evals"])
            if advanced > 0:
                eval_state["step"] += advanced
            ev_this = loss_eval_counter["value"] - eval_state["last_loss_evals"]
            eval_state["last_loss_evals"] = loss_eval_counter["value"]
            pbar.set_postfix(_format_pbar_postfix(
                step=eval_state["step"] if eval_state["step"] > 0 else 1,
                evals=loss_eval_counter["value"],
                ev_per_step=ev_this,
                gnorm=eval_state.get("last_gnorm", float("nan")),
                loss=loss_value,
                best=eval_state["best_true"],
            ))
        if progress_callback is not None and ((step_num % log_every == 0) or (step_num == maxeval)):
            progress_callback(step_num, float(loss_value))
        return float(loss_value)

    opt.set_min_objective(objective)
    x_opt = None
    try:
        x_opt = opt.optimize(x0)
    except BaseException as exc:  # pylint: disable=broad-except
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            if pbar is not None:
                pbar.close()
            raise
        eval_state["stopped_reason"] = f"nlopt_error:{type(exc).__name__}"
        warnings.warn(
            f"NLopt terminated with {type(exc).__name__}: {exc}. "
            f"Returning best params found after {eval_state['evals']} objective calls.",
            RuntimeWarning,
            stacklevel=2,
        )
    finally:
        if pbar is not None:
            pbar.close()

    if controls["restore_best"] and eval_state["best_x_true"] is not None and np.isfinite(eval_state["best_true"]):
        best_x = eval_state["best_x_true"]
    elif controls["restore_best"] and eval_state["best_x_sel"] is not None:
        best_x = eval_state["best_x_sel"]
    elif x_opt is not None:
        best_x = np.asarray(x_opt, dtype=np.float64)
    else:
        best_x = np.asarray(x0, dtype=np.float64)
    _assign_flat_params(best_x, specs)

    if not history:
        try:
            history.append(_eval_loss(best_x))
        except (RuntimeError, ValueError, FloatingPointError):
            history.append(controls["penalty_on_bad"])

    best_loss = eval_state["best_true"]
    convergence_reason = eval_state["stopped_reason"]
    if controls["restore_best"] and eval_state["best_x_true"] is not None:
        final_loss = best_loss
    else:
        final_loss = history[-1] if history else float("nan")
    return params_run, history, best_loss, final_loss, convergence_reason, loss_eval_counter["value"]
