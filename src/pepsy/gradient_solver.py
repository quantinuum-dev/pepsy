"""Gradient-based parameter solvers used by sweep optimization."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

__all__ = ["SUPPORTED_SOLVERS", "optimize_packed_params"]

SUPPORTED_SOLVERS = (
    "adam",
    "adamw",
    "adagrad",
    "rmsprop",
    "sgd",
    "lbfgs",
    "scipy-lbfgs",
    "nlopt-lbfgs",
)

_TORCH_SOLVERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
    "sgd": torch.optim.SGD,
    "lbfgs": torch.optim.LBFGS,
}

_SOLVER_ALIASES = {
    "torch-adam": "adam",
    "torch_adam": "adam",
    "torch-lbfgs": "lbfgs",
    "torch_lbfgs": "lbfgs",
    "scipy": "scipy-lbfgs",
    "scipy_lbfgs": "scipy-lbfgs",
    "scipy-l-bfgs-b": "scipy-lbfgs",
    "nlopt": "nlopt-lbfgs",
    "nlopt_lbfgs": "nlopt-lbfgs",
}


def _normalize_solver_name(solver: str) -> str:
    if not isinstance(solver, str):
        raise TypeError("solver must be a string")
    normalized = solver.strip().lower()
    normalized = _SOLVER_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_SOLVERS:
        supported = ", ".join(SUPPORTED_SOLVERS)
        raise ValueError(f"Unsupported solver={solver!r}. Supported solvers: {supported}")
    return normalized


def _as_trainable_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if not (tensor.is_floating_point() or tensor.is_complex()):
        tensor = tensor.to(dtype=torch.float64)
    return tensor.requires_grad_(True)


def _scalar_real_loss(loss: Any, imag_tol: float = 1e-10) -> torch.Tensor:
    """Convert scalar loss to a real-valued tensor, with complex safety checks."""
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss)
    if loss.numel() != 1:
        raise ValueError("loss_fn must return a scalar")
    if loss.is_complex():
        imag_mag = float(abs(complex(loss.detach().cpu()).imag))
        if imag_mag > imag_tol:
            raise ValueError(
                f"loss_fn returned complex loss with imag={imag_mag:.3e}; expected near-real scalar."
            )
        loss = loss.real
    return loss


def _param_ordered_items(params_init: Mapping[str, Any]):
    return [(name, _as_trainable_tensor(value)) for name, value in params_init.items()]


def _build_param_specs(items: list[tuple[str, torch.Tensor]]):
    """Create per-parameter metadata with direct tensor refs for fast access."""
    specs = []
    for name, tensor in items:
        data = tensor.detach()
        numel = int(data.numel())
        shape = tuple(data.shape)
        specs.append(
            {
                "name": name,
                "tensor": tensor,
                "shape": shape,
                "numel": numel,
                "dtype": data.dtype,
                "device": tensor.device,
                "is_complex": bool(data.is_complex()),
            }
        )
    return specs


def _flatten_params_real_torch(specs: list[dict[str, Any]]) -> torch.Tensor:
    """Flatten potentially complex tensors to a single CPU float64 tensor."""
    parts: list[torch.Tensor] = []
    for spec in specs:
        data = spec["tensor"].detach()
        if spec["is_complex"]:
            parts.append(data.real.reshape(-1).to(device="cpu", dtype=torch.float64))
            parts.append(data.imag.reshape(-1).to(device="cpu", dtype=torch.float64))
        else:
            parts.append(data.reshape(-1).to(device="cpu", dtype=torch.float64))
    if not parts:
        return torch.empty(0, dtype=torch.float64, device="cpu")
    return torch.cat(parts).detach()


def _flatten_params_real_numpy(specs: list[dict[str, Any]]) -> np.ndarray:
    """Flatten params to NumPy float64 vector (for SciPy/NLopt only)."""
    flat = _flatten_params_real_torch(specs)
    return np.array(flat.numpy(), dtype=np.float64, copy=True)


def _assign_flat_params(vector: np.ndarray | torch.Tensor, specs: list[dict[str, Any]]):
    if isinstance(vector, torch.Tensor):
        vector_t = vector.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    else:
        vector_np = np.asarray(vector, dtype=np.float64)
        if not vector_np.flags.writeable:
            vector_np = np.array(vector_np, dtype=np.float64, copy=True)
        vector_t = torch.from_numpy(vector_np).to(device="cpu", dtype=torch.float64).reshape(-1)
    offset = 0
    for spec in specs:
        tensor = spec["tensor"]
        numel = spec["numel"]
        shape = spec["shape"]
        dtype = spec["dtype"]
        device = spec["device"]
        if spec["is_complex"]:
            real_slice = vector_t[offset : offset + numel].reshape(shape)
            offset += numel
            imag_slice = vector_t[offset : offset + numel].reshape(shape)
            offset += numel
            real_t = real_slice.to(device=device, dtype=torch.float64)
            imag_t = imag_slice.to(device=device, dtype=torch.float64)
            value = torch.complex(real_t, imag_t).to(dtype=dtype)
        else:
            value = vector_t[offset : offset + numel].reshape(shape).to(device=device, dtype=dtype)
            offset += numel
        with torch.no_grad():
            tensor.copy_(value)
        tensor.requires_grad_(True)


def _clone_param_state(params_run: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in params_run.items()}


def _restore_param_state(params_run: dict[str, torch.Tensor], state: Mapping[str, torch.Tensor]) -> None:
    for name, tensor in state.items():
        with torch.no_grad():
            params_run[name].copy_(tensor)
        params_run[name].requires_grad_(True)


def _flatten_grads_real(specs: list[dict[str, Any]]) -> np.ndarray:
    parts: list[torch.Tensor] = []
    for spec in specs:
        numel = spec["numel"]
        grad = spec["tensor"].grad
        if grad is None:
            if spec["is_complex"]:
                parts.append(torch.zeros(numel, dtype=torch.float64, device="cpu"))
                parts.append(torch.zeros(numel, dtype=torch.float64, device="cpu"))
            else:
                parts.append(torch.zeros(numel, dtype=torch.float64, device="cpu"))
            continue
        grad_cpu = grad.detach().to(device="cpu")
        if spec["is_complex"]:
            parts.append(grad_cpu.real.reshape(-1).to(dtype=torch.float64))
            parts.append(grad_cpu.imag.reshape(-1).to(dtype=torch.float64))
        else:
            parts.append(grad_cpu.reshape(-1).to(dtype=torch.float64))
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.array(torch.cat(parts).detach().numpy(), dtype=np.float64, copy=True)


def _evaluate_loss_and_grad(
    vector: np.ndarray,
    params_run: dict[str, torch.Tensor],
    specs: list[dict[str, Any]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
):
    _assign_flat_params(vector, specs)
    for spec in specs:
        spec["tensor"].grad = None
    loss = _scalar_real_loss(loss_fn(params_run))
    loss.backward()
    grad_vec = _flatten_grads_real(specs)
    return float(loss.detach().cpu()), grad_vec


def _as_numpy_vector(values: Any, size: int, *, key: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{key} must be scalar or length-{size} vector.")
    return arr


_SCIPY_METHOD_MAP: dict[str, str] = {
    # L-BFGS-B aliases
    "L_BFGS_B": "L-BFGS-B",
    "L_BFGS": "L-BFGS-B",
    "LBFGS": "L-BFGS-B",
    "LD_LBFGS": "L-BFGS-B",
    # Other gradient-based methods
    "CG": "CG",
    "BFGS": "BFGS",
    "TNC": "TNC",
    "SLSQP": "SLSQP",
    "TRUST_CONSTR": "trust-constr",
    "TRUST_KRYLOV": "trust-krylov",
    "NEWTON_CG": "Newton-CG",
}

_SCIPY_BOUNDS_METHODS: frozenset[str] = frozenset(
    {"L-BFGS-B", "TNC", "SLSQP", "trust-constr"}
)


def _normalize_scipy_method(name: Any, *, key: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{key} must be a string")
    normalized = name.strip().upper().replace("-", "_")
    method = _SCIPY_METHOD_MAP.get(normalized)
    if method is None:
        valid = sorted(_SCIPY_METHOD_MAP.values())
        raise ValueError(
            f"Unknown scipy algorithm {name!r}. "
            f"Supported: {', '.join(dict.fromkeys(valid))}"
        )
    return method


def _as_nonnegative_float(value: Any, *, key: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{key} must be finite and >= 0")
    return out


def _pop_step_count(options: dict[str, Any], *, default: int) -> int:
    if "maxiter" in options:
        steps = int(options.pop("maxiter"))
    elif "maxeval" in options:
        steps = int(options.pop("maxeval"))
    elif "its_max" in options:
        steps = int(options.pop("its_max"))
    else:
        steps = int(default)
    if steps <= 0:
        raise ValueError("step count must be >= 1")
    return steps


def _pop_common_controls(
    options: dict[str, Any],
    *,
    default_steps: int,
    default_bad_max: int,
    default_penalty: float,
):
    max_steps = _pop_step_count(options, default=default_steps)
    patience = options.pop("patience", None)
    min_steps = int(options.pop("min_steps", options.pop("min_evals", 0)))
    min_improve = float(options.pop("min_improve", 0.0))
    restore_best = bool(options.pop("restore_best", True))
    bad_max = int(options.pop("bad_max", default_bad_max))
    penalty_on_bad = float(options.pop("penalty_on_bad", options.pop("penalty_value", default_penalty)))

    if patience is not None and int(patience) < 0:
        raise ValueError("patience must be >= 0 or None")
    if min_steps < 0:
        raise ValueError("min_steps must be >= 0")
    if bad_max <= 0:
        raise ValueError("bad_max must be >= 1")
    if not np.isfinite(penalty_on_bad):
        raise ValueError("penalty_on_bad must be finite")

    return {
        "max_steps": max_steps,
        "patience": None if patience is None else int(patience),
        "min_steps": min_steps,
        "min_improve": min_improve,
        "restore_best": restore_best,
        "bad_max": bad_max,
        "penalty_on_bad": penalty_on_bad,
    }


def _build_torch_scheduler(optimizer, options: dict[str, Any], *, max_steps: int):
    scheduler_name = options.pop("scheduler", None)
    if scheduler_name is None or str(scheduler_name).lower() in {"none", "constant"}:
        return None, "none"

    name = str(scheduler_name).lower()
    eta_min = float(options.pop("eta_min", 1e-8))
    if name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_steps),
            eta_min=eta_min,
        )
        return sched, "cosine"
    if name == "step":
        step_size = int(options.pop("step_size", 50))
        gamma = float(options.pop("gamma", 0.5))
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        return sched, "step"
    if name == "plateau":
        plateau_patience = int(options.pop("plateau_patience", 50))
        plateau_factor = float(options.pop("plateau_factor", 0.5))
        plateau_threshold = float(options.pop("plateau_threshold", 1e-4))
        plateau_cooldown = int(options.pop("plateau_cooldown", 0))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            cooldown=plateau_cooldown,
            min_lr=eta_min,
        )
        return sched, "plateau"
    raise ValueError(
        "scheduler must be one of: none/constant, cosine, step, plateau"
    )


def _step_torch_scheduler(scheduler, kind: str, loss_value: float) -> None:
    if scheduler is None:
        return
    if kind == "plateau":
        scheduler.step(loss_value)
        return
    scheduler.step()


def _iter_steps(
    n_steps: int,
    *,
    show_opt_progress: bool,
    opt_desc: str | None,
):
    iterator = range(n_steps)
    if not show_opt_progress:
        return iterator
    return tqdm(iterator, total=n_steps, desc=opt_desc or "opt", leave=False, colour="CYAN")


def _run_torch_solver(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver_name: str,
    lr: float,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    show_opt_progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    params_run = dict(items)
    options = dict(solver_options)
    controls = _pop_common_controls(
        options,
        default_steps=n_steps,
        default_bad_max=5,
        default_penalty=1e20,
    )
    for key in (
        "method",
        "algorithm",
        "optimizer",
        "ftol",
        "ftol_rel",
        "gtol",
        "xtol_rel",
        "ftol_abs",
        "xtol_abs",
        "stopval",
        "bounds",
        "lower_bounds",
        "upper_bounds",
        "ema_alpha",
        "assume_nonnegative",
        "best_neg_tol",
    ):
        options.pop(key, None)
    grad_clip_norm = options.pop("clip_grad_norm", options.pop("grad_clip_norm", None))
    grad_clip_value = options.pop("grad_clip_value", None)
    max_step_norm = options.pop("max_step_norm", options.pop("max_step", None))
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("clip_grad_norm must be > 0 when set")
    if max_step_norm is not None and float(max_step_norm) <= 0.0:
        raise ValueError("max_step_norm must be > 0 when set")

    scheduler_options = {}
    for key in (
        "scheduler",
        "eta_min",
        "step_size",
        "gamma",
        "plateau_patience",
        "plateau_factor",
        "plateau_threshold",
        "plateau_cooldown",
    ):
        if key in options:
            scheduler_options[key] = options.pop(key)

    optimizer_cls = _TORCH_SOLVERS[solver_name]
    optimizer = optimizer_cls(list(params_run.values()), lr=lr, **options)
    scheduler, scheduler_kind = _build_torch_scheduler(
        optimizer,
        scheduler_options,
        max_steps=controls["max_steps"],
    )

    history: list[float] = []
    best_loss = float("inf")
    best_state = _clone_param_state(params_run)
    step_iter = _iter_steps(
        controls["max_steps"],
        show_opt_progress=show_opt_progress,
        opt_desc=opt_desc,
    )
    bad_consecutive = 0
    last_improve_step = 0

    for step in step_iter:
        optimizer.zero_grad(set_to_none=True)
        loss = _scalar_real_loss(loss_fn(params_run))
        if not torch.isfinite(loss):
            bad_consecutive += 1
            loss_value = controls["penalty_on_bad"]
            history.append(loss_value)
            _step_torch_scheduler(scheduler, scheduler_kind, float("inf"))
            if show_opt_progress:
                step_iter.set_postfix({"loss": loss_value})
            step_num = step + 1
            if progress_callback is not None and (step_num % log_every == 0):
                progress_callback(step_num, loss_value)
            if (bad_consecutive >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
                break
            continue

        bad_consecutive = 0
        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if loss_value + controls["min_improve"] < best_loss:
            best_loss = loss_value
            best_state = _clone_param_state(params_run)
            last_improve_step = step + 1

        if max_step_norm is not None:
            prev_flat = torch.nn.utils.parameters_to_vector(list(params_run.values())).detach().clone()

        loss.backward()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(list(params_run.values()), float(grad_clip_value))
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(params_run.values()), float(grad_clip_norm))
        optimizer.step()

        if max_step_norm is not None:
            with torch.no_grad():
                cur_flat = torch.nn.utils.parameters_to_vector(list(params_run.values()))
                delta = cur_flat - prev_flat
                delta_norm = float(torch.linalg.vector_norm(delta).detach().cpu().item())
                step_max = float(max_step_norm)
                if delta_norm > step_max:
                    clipped = prev_flat + delta * (step_max / (delta_norm + 1e-12))
                    torch.nn.utils.vector_to_parameters(clipped, list(params_run.values()))

        _step_torch_scheduler(scheduler, scheduler_kind, loss_value)

        if show_opt_progress:
            step_iter.set_postfix({"loss": loss_value})
        step_num = step + 1
        if progress_callback is not None and (step_num % log_every == 0):
            progress_callback(step_num, loss_value)
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - last_improve_step) >= controls["patience"]:
                break

    with torch.no_grad():
        final_loss = float(_scalar_real_loss(loss_fn(params_run)).detach().cpu())
    if final_loss + controls["min_improve"] < best_loss:
        best_state = _clone_param_state(params_run)
        best_loss = final_loss
    if history:
        history[-1] = final_loss
    if progress_callback is not None:
        progress_callback(max(1, len(history)), final_loss)

    if controls["restore_best"]:
        _restore_param_state(params_run, best_state)
    return params_run, history


def _run_torch_lbfgs(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    lr: float,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    show_opt_progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    params_run = dict(items)
    options = dict(solver_options)
    controls = _pop_common_controls(
        options,
        default_steps=n_steps,
        default_bad_max=5,
        default_penalty=1e20,
    )
    for key in (
        "method",
        "algorithm",
        "optimizer",
        "ftol",
        "ftol_rel",
        "gtol",
        "xtol_rel",
        "ftol_abs",
        "xtol_abs",
        "stopval",
        "bounds",
        "lower_bounds",
        "upper_bounds",
        "ema_alpha",
        "assume_nonnegative",
        "best_neg_tol",
    ):
        options.pop(key, None)
    grad_clip_norm = options.pop("clip_grad_norm", options.pop("grad_clip_norm", None))
    grad_clip_value = options.pop("grad_clip_value", None)
    max_step_norm = options.pop("max_step_norm", options.pop("max_step", None))
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("clip_grad_norm must be > 0 when set")
    if max_step_norm is not None and float(max_step_norm) <= 0.0:
        raise ValueError("max_step_norm must be > 0 when set")

    scheduler_options = {}
    for key in (
        "scheduler",
        "eta_min",
        "step_size",
        "gamma",
        "plateau_patience",
        "plateau_factor",
        "plateau_threshold",
        "plateau_cooldown",
    ):
        if key in options:
            scheduler_options[key] = options.pop(key)

    # Treat external step count as user-facing iteration count.
    options.setdefault("max_iter", 1)
    optimizer = torch.optim.LBFGS(list(params_run.values()), lr=lr, **options)
    scheduler, scheduler_kind = _build_torch_scheduler(
        optimizer,
        scheduler_options,
        max_steps=controls["max_steps"],
    )

    history: list[float] = []
    best_loss = float("inf")
    best_state = _clone_param_state(params_run)
    step_iter = _iter_steps(
        controls["max_steps"],
        show_opt_progress=show_opt_progress,
        opt_desc=opt_desc,
    )
    bad_consecutive = 0
    last_improve_step = 0

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss_val = _scalar_real_loss(loss_fn(params_run))
        if not torch.isfinite(loss_val):
            return loss_val
        loss_val.backward()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(list(params_run.values()), float(grad_clip_value))
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(params_run.values()), float(grad_clip_norm))
        return loss_val

    for step in step_iter:
        with torch.no_grad():
            loss_before = _scalar_real_loss(loss_fn(params_run))
        if not torch.isfinite(loss_before):
            bad_consecutive += 1
            loss_value = controls["penalty_on_bad"]
            history.append(loss_value)
            _step_torch_scheduler(scheduler, scheduler_kind, float("inf"))
            if show_opt_progress:
                step_iter.set_postfix({"loss": loss_value})
            step_num = step + 1
            if progress_callback is not None and (step_num % log_every == 0):
                progress_callback(step_num, loss_value)
            if (bad_consecutive >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
                break
            continue

        bad_consecutive = 0
        loss_value = float(loss_before.detach().cpu())
        history.append(loss_value)
        if loss_value + controls["min_improve"] < best_loss:
            best_loss = loss_value
            best_state = _clone_param_state(params_run)
            last_improve_step = step + 1

        if max_step_norm is not None:
            prev_flat = torch.nn.utils.parameters_to_vector(list(params_run.values())).detach().clone()

        optimizer.step(closure)

        if max_step_norm is not None:
            with torch.no_grad():
                cur_flat = torch.nn.utils.parameters_to_vector(list(params_run.values()))
                delta = cur_flat - prev_flat
                delta_norm = float(torch.linalg.vector_norm(delta).detach().cpu().item())
                step_max = float(max_step_norm)
                if delta_norm > step_max:
                    clipped = prev_flat + delta * (step_max / (delta_norm + 1e-12))
                    torch.nn.utils.vector_to_parameters(clipped, list(params_run.values()))

        _step_torch_scheduler(scheduler, scheduler_kind, loss_value)

        if show_opt_progress:
            step_iter.set_postfix({"loss": loss_value})
        step_num = step + 1
        if progress_callback is not None and (step_num % log_every == 0):
            progress_callback(step_num, loss_value)
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - last_improve_step) >= controls["patience"]:
                break

    with torch.no_grad():
        final_loss = float(_scalar_real_loss(loss_fn(params_run)).detach().cpu())
    if final_loss + controls["min_improve"] < best_loss:
        best_state = _clone_param_state(params_run)
        best_loss = final_loss
    if history:
        history[-1] = final_loss
    if progress_callback is not None:
        progress_callback(max(1, len(history)), final_loss)

    if controls["restore_best"]:
        _restore_param_state(params_run, best_state)
    return params_run, history


def _run_scipy_lbfgs(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    show_opt_progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    try:
        from scipy import optimize as sp_opt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "solver='scipy-lbfgs' requires SciPy. Install with: pip install scipy"
        ) from exc

    params_run = dict(items)
    specs = _build_param_specs(items)
    x0 = _flatten_params_real_numpy(specs)
    if x0.size == 0:
        return params_run, []

    options = dict(solver_options)
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

    if "ftol" in options:
        ftol = _as_nonnegative_float(options.pop("ftol"), key="ftol")
    else:
        ftol = _as_nonnegative_float(options.pop("ftol_rel", 1e-9), key="ftol_rel")

    if "gtol" in options:
        gtol = _as_nonnegative_float(options.pop("gtol"), key="gtol")
    else:
        gtol = _as_nonnegative_float(options.pop("xtol_rel", 1e-9), key="xtol_rel")

    options.setdefault("maxiter", controls["max_steps"])
    options.setdefault("ftol", ftol)
    options.setdefault("gtol", gtol)

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

    history: list[float] = []
    step_counter = {"value": 0}
    state = {
        "last_loss": None,
        "best_loss": float("inf"),
        "best_x": None,
        "last_improve_step": 0,
        "bad_consecutive": 0,
    }
    pbar = None
    if show_opt_progress:
        pbar = tqdm(
            total=controls["max_steps"],
            desc=opt_desc or "opt",
            leave=False,
            colour="CYAN",
        )

    def objective(x):
        try:
            loss_value, grad_value = _evaluate_loss_and_grad(x, params_run, specs, loss_fn)
            if not np.isfinite(loss_value) or not np.isfinite(grad_value).all():
                raise FloatingPointError("non-finite objective or gradient")
        except (FloatingPointError, RuntimeError, ValueError):
            state["bad_consecutive"] += 1
            loss_value = controls["penalty_on_bad"]
            grad_value = np.zeros_like(np.asarray(x, dtype=np.float64))
        else:
            state["bad_consecutive"] = 0
            if loss_value + controls["min_improve"] < state["best_loss"]:
                state["best_loss"] = loss_value
                state["best_x"] = np.array(x, dtype=np.float64, copy=True)
                state["last_improve_step"] = step_counter["value"] + 1
        state["last_loss"] = float(loss_value)
        return loss_value, grad_value

    def callback(_xk):
        step_counter["value"] += 1
        step_num = step_counter["value"]
        loss_value = state["last_loss"]
        if loss_value is None:
            loss_value, _grad = _evaluate_loss_and_grad(_xk, params_run, specs, loss_fn)
        history.append(float(loss_value))
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"loss": float(loss_value)})
        if progress_callback is not None and (
            (step_num % log_every == 0) or (step_num == controls["max_steps"])
        ):
            progress_callback(step_num, float(loss_value))
        if (state["bad_consecutive"] >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
            raise StopIteration
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - state["last_improve_step"]) >= controls["patience"]:
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
            fallback = float(_scalar_real_loss(loss_fn(params_run)).detach().cpu())
        except (RuntimeError, ValueError, FloatingPointError):
            fallback = controls["penalty_on_bad"]
        history.append(fallback)
    return params_run, history


def _run_nlopt_lbfgs(
    items: list[tuple[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver_options: dict[str, Any],
    n_steps: int,
    log_every: int,
    show_opt_progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    try:
        import nlopt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "solver='nlopt-lbfgs' requires NLopt. Install with: pip install nlopt"
        ) from exc

    params_run = dict(items)
    specs = _build_param_specs(items)
    x0 = _flatten_params_real_numpy(specs)
    if x0.size == 0:
        return params_run, []

    options = dict(solver_options)
    if "algorithm" in options:
        algorithm_name = options.pop("algorithm")
    else:
        algorithm_name = options.pop("optimizer", "LD_LBFGS")
    method_name = options.pop("method", None)
    if method_name is not None:
        normalized = method_name.strip().upper().replace("-", "_")
        lbfgs_aliases = {"L_BFGS", "LBFGS", "LD_LBFGS", "L_BFGS_B"}
        if normalized not in lbfgs_aliases:
            raise ValueError(
                "solver='nlopt-lbfgs' only accepts L-BFGS-B style method aliases."
            )

    controls = _pop_common_controls(
        options,
        default_steps=n_steps,
        default_bad_max=20,
        default_penalty=1e20,
    )
    maxeval = controls["max_steps"]
    ftol_rel = _as_nonnegative_float(options.pop("ftol_rel", options.pop("ftol", 1e-9)), key="ftol_rel")
    xtol_rel = _as_nonnegative_float(options.pop("xtol_rel", options.pop("gtol", 1e-9)), key="xtol_rel")
    lower = options.pop("lower_bounds", None)
    upper = options.pop("upper_bounds", None)
    max_step = options.pop("max_step", None)
    grad_clip_norm = options.pop("grad_clip_norm", None)
    ema_alpha = options.pop("ema_alpha", None)
    assume_nonnegative = bool(options.pop("assume_nonnegative", True))
    best_neg_tol = float(options.pop("best_neg_tol", 1e-12))

    if ema_alpha is not None and not 0.0 < float(ema_alpha) <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1] when set")
    if max_step is not None and float(max_step) <= 0.0:
        raise ValueError("max_step must be > 0 when set")
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when set")

    opt_map = {
        "LBFGS": "LD_LBFGS",
        "LD_VAR2": "LD_VAR2",
        "VAR2": "LD_VAR2",
        "MMA": "LD_MMA",
        "CCSAQ": "LD_CCSAQ",
        "SLSQP": "LD_SLSQP",
        "TNEWTON": "LD_TNEWTON",
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
    opt.set_xtol_rel(xtol_rel)

    if lower is not None:
        opt.set_lower_bounds(_as_numpy_vector(lower, int(x0.size), key="lower_bounds"))
    if upper is not None:
        opt.set_upper_bounds(_as_numpy_vector(upper, int(x0.size), key="upper_bounds"))

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
        "last_improve_eval": 0,
        "evals": 0,
        "bad_consecutive": 0,
        "prev_x": None,
        "stopped_reason": "maxeval",
    }
    pbar = None
    if show_opt_progress:
        pbar = tqdm(total=maxeval, desc=opt_desc or "opt", leave=False, colour="CYAN")

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
                if pbar.n < pbar.total:
                    pbar.update(1)
                pbar.set_postfix({"loss": controls["penalty_on_bad"]})
            return controls["penalty_on_bad"]

        if max_step is not None and eval_state["prev_x"] is not None:
            delta = x_vec - eval_state["prev_x"]
            delta_norm = float(np.linalg.norm(delta))
            step_limit = float(max_step)
            if delta_norm > step_limit:
                x_vec = eval_state["prev_x"] + delta * (step_limit / (delta_norm + 1e-12))
        eval_state["prev_x"] = np.array(x_vec, dtype=np.float64, copy=True)

        try:
            loss_value, grad_value = _evaluate_loss_and_grad(x_vec, params_run, specs, loss_fn)
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
            if grad_clip_norm is not None:
                grad_norm = float(np.linalg.norm(grad_value))
                clip = float(grad_clip_norm)
                if grad_norm > clip:
                    grad_value = grad_value * (clip / (grad_norm + 1e-12))
            if grad.size > 0:
                grad[:] = grad_value

            if (not assume_nonnegative) or (loss_value >= -best_neg_tol):
                if loss_value < eval_state["best_true"]:
                    eval_state["best_true"] = loss_value
                    eval_state["best_x_true"] = np.array(x_vec, dtype=np.float64, copy=True)

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
                if (eval_state["evals"] - eval_state["last_improve_eval"]) > controls["patience"]:
                    eval_state["stopped_reason"] = "patience"
                    opt.force_stop()

        history.append(float(loss_value))
        step_num = len(history)
        if pbar is not None:
            if pbar.n < pbar.total:
                pbar.update(1)
            pbar.set_postfix({"loss": float(loss_value)})
        if progress_callback is not None and ((step_num % log_every == 0) or (step_num == maxeval)):
            progress_callback(step_num, float(loss_value))
        return float(loss_value)

    opt.set_min_objective(objective)
    x_opt = None
    try:
        x_opt = opt.optimize(x0)
    except (RuntimeError, ValueError, FloatingPointError, nlopt.RoundoffLimited,
            nlopt.ForcedStop, nlopt.runtime_error, nlopt.forced_stop) as exc:
        if not history:
            warnings.warn(
                f"NLopt terminated before a valid step ({type(exc).__name__}). "
                "Restoring initial parameters.",
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
            loss_fallback = float(_scalar_real_loss(loss_fn(params_run)).detach().cpu())
        except (RuntimeError, ValueError, FloatingPointError):
            loss_fallback = controls["penalty_on_bad"]
        history.append(loss_fallback)
    return params_run, history


def optimize_packed_params(
    params_init: Mapping[str, Any],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver: str = "adam",
    solver_options: Mapping[str, Any] | None = None,
    n_steps: int = 100,
    log_every: int = 20,
    pbar: bool = False,
    opt_desc: str | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
):
    """Optimize packed tensor params with a selected gradient solver.

    Parameters
    ----------
    solver_options : dict | None
        Backend-specific options. Common keys:

        - ``lr`` (float, default 1e-2): learning rate for torch solvers.
        - ``patience``, ``min_steps``, ``restore_best``, …
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    if log_every <= 0:
        raise ValueError("log_every must be >= 1")
    if solver_options is None:
        options = {}
    elif isinstance(solver_options, Mapping):
        options = dict(solver_options)
    else:
        raise TypeError("solver_options must be a mapping or None")
    lr = float(options.pop("lr", 1e-2))
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and > 0")

    solver_name = _normalize_solver_name(solver)
    items = _param_ordered_items(params_init)

    if solver_name == "lbfgs":
        return _run_torch_lbfgs(
            items,
            loss_fn,
            lr=lr,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            show_opt_progress=pbar,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    if solver_name in _TORCH_SOLVERS:
        return _run_torch_solver(
            items,
            loss_fn,
            solver_name=solver_name,
            lr=lr,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            show_opt_progress=pbar,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    if solver_name == "scipy-lbfgs":
        return _run_scipy_lbfgs(
            items,
            loss_fn,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            show_opt_progress=pbar,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    if solver_name == "nlopt-lbfgs":
        return _run_nlopt_lbfgs(
            items,
            loss_fn,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            show_opt_progress=pbar,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    raise RuntimeError(f"Unhandled solver path for {solver_name!r}")
