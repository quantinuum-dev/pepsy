"""Gradient-based parameter solvers used by sweep optimization."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

__all__ = [
    "SUPPORTED_SOLVERS",
    "GradSolverResult",
    "GradientOptimizer",
    "optimize_packed_params",
]

SUPPORTED_SOLVERS = (
    "torch-adam",
    "torch-lbfgs",
    "torch-adamw",
    "torch-radam",
    "torch-nadam",
    "scipy",
    "nlopt",
)

_TORCH_SOLVERS = {}
if torch is not None:
    _TORCH_SOLVERS = {
        "torch-adam": torch.optim.Adam,
        "torch-lbfgs": torch.optim.LBFGS,
        "torch-adamw": torch.optim.AdamW,
        "torch-radam": torch.optim.RAdam,
        "torch-nadam": torch.optim.NAdam,
    }


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "pepsy.gradient_solver requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )


@dataclass(frozen=True)
class GradSolverResult:
    """Structured optimization result returned by :class:`GradientOptimizer`.

    Fields capture optimized parameters, solver identity, and loss summary.
    """

    params: dict[str, torch.Tensor]
    history: list[float]
    solver: str
    n_steps: int
    best_loss: float
    final_loss: float
    convergence_reason: str = "maxiter"
    n_evals: int = -1

def _normalize_solver_name(solver: str) -> str:
    if not isinstance(solver, str):
        raise TypeError("solver must be a string")
    normalized = solver.strip().lower()
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
    # Ensure contiguous memory layout — torch.optim.LBFGS calls .view(-1) on
    # gradients which fails on non-contiguous (e.g. strided complex) tensors.
    return tensor.contiguous().requires_grad_(True)


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
    """Flatten potentially complex tensors to a single CPU float64 tensor.

    Concatenation is done on the source device first so that only a single
    device→CPU transfer is needed (instead of one per tensor).
    """
    parts: list[torch.Tensor] = []
    for spec in specs:
        data = spec["tensor"].detach()
        if spec["is_complex"]:
            parts.append(data.real.reshape(-1).to(dtype=torch.float64))
            parts.append(data.imag.reshape(-1).to(dtype=torch.float64))
        else:
            parts.append(data.reshape(-1).to(dtype=torch.float64))
    if not parts:
        return torch.empty(0, dtype=torch.float64, device="cpu")
    # Single cat on source device, then one bulk transfer to CPU.
    return torch.cat(parts).detach().to(device="cpu")


def _flatten_params_real_numpy(specs: list[dict[str, Any]]) -> np.ndarray:
    """Flatten params to NumPy float64 vector (for SciPy/NLopt only)."""
    flat = _flatten_params_real_torch(specs)
    return flat.numpy().copy()


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
    device = None
    for spec in specs:
        numel = spec["numel"]
        grad = spec["tensor"].grad
        if grad is None:
            # Use source device for zero-fill so cat stays on one device.
            dev = spec["device"]
            if spec["is_complex"]:
                parts.append(torch.zeros(numel, dtype=torch.float64, device=dev))
                parts.append(torch.zeros(numel, dtype=torch.float64, device=dev))
            else:
                parts.append(torch.zeros(numel, dtype=torch.float64, device=dev))
            continue
        if device is None:
            device = grad.device
        grad_d = grad.detach()
        if spec["is_complex"]:
            parts.append(grad_d.real.reshape(-1).to(dtype=torch.float64))
            parts.append(grad_d.imag.reshape(-1).to(dtype=torch.float64))
        else:
            parts.append(grad_d.reshape(-1).to(dtype=torch.float64))
    if not parts:
        return np.empty(0, dtype=np.float64)
    # Single cat on source device, then one bulk transfer to CPU.
    return torch.cat(parts).detach().to(device="cpu").numpy().copy()


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


def _build_loss_from_flat(
    specs: list[dict[str, Any]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a differentiable scalar function of the flat real parameter vector.

    Unlike :func:`_evaluate_loss_and_grad`, this preserves the autograd graph
    through the flat vector, enabling second-order differentiation.
    Used by :func:`_evaluate_hessian` and :func:`_evaluate_hessp`.
    """
    def _loss(x_vec: torch.Tensor) -> torch.Tensor:
        params_tmp: dict[str, torch.Tensor] = {}
        offset = 0
        for spec in specs:
            name = spec["name"]
            numel = spec["numel"]
            shape = spec["shape"]
            dtype = spec["dtype"]
            device = spec["device"]
            if spec["is_complex"]:
                real_part = x_vec[offset : offset + numel].reshape(shape)
                offset += numel
                imag_part = x_vec[offset : offset + numel].reshape(shape)
                offset += numel
                val = torch.complex(
                    real_part.to(device=device, dtype=torch.float64),
                    imag_part.to(device=device, dtype=torch.float64),
                ).to(dtype=dtype)
            else:
                val = x_vec[offset : offset + numel].reshape(shape).to(device=device, dtype=dtype)
                offset += numel
            params_tmp[name] = val
        return _scalar_real_loss(loss_fn(params_tmp))
    return _loss


def _evaluate_hessian(
    vector: np.ndarray,
    specs: list[dict[str, Any]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> np.ndarray:
    """Compute the full n\u00d7n Hessian matrix via ``torch.autograd.functional.hessian``.

    Cost: O(n) backward passes where n = number of real parameters.
    Only practical for small parameter counts (< ~50).  Used by trust-exact.
    """
    x0 = torch.tensor(vector, dtype=torch.float64)
    loss_from_flat = _build_loss_from_flat(specs, loss_fn)
    H = torch.autograd.functional.hessian(loss_from_flat, x0)
    return np.array(H.detach().cpu().numpy(), dtype=np.float64)


def _evaluate_hessp(
    vector: np.ndarray,
    p: np.ndarray,
    specs: list[dict[str, Any]],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> np.ndarray:
    """Compute the Hessian-vector product H(x)@p via ``torch.autograd.functional.vhp``.

    Cost: O(1) backward passes (one forward + one reverse-mode AD pass).
    Suitable for trust-krylov and Newton-CG — much cheaper than the full Hessian.
    """
    x0 = torch.tensor(vector, dtype=torch.float64)
    p_tensor = torch.tensor(p, dtype=torch.float64)
    loss_from_flat = _build_loss_from_flat(specs, loss_fn)
    _, hvp = torch.autograd.functional.vhp(loss_from_flat, x0, p_tensor)
    return np.array(hvp.detach().cpu().numpy(), dtype=np.float64)


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
    # Hessian-based trust-region
    "TRUST_EXACT": "trust-exact",
    "TRUST_NCG": "trust-ncg",
}

_SCIPY_BOUNDS_METHODS: frozenset[str] = frozenset(
    {"L-BFGS-B", "TNC", "SLSQP", "trust-constr"}
)
# Methods that use a Hessian-vector product callable (hessp): O(1) backward pass per call.
# Much cheaper than a full Hessian; suitable for mid-size problems.
_SCIPY_HESSP_METHODS: frozenset[str] = frozenset({"trust-krylov", "trust-ncg", "Newton-CG"})
# Methods that require the full n×n Hessian matrix: O(n) backward passes per call.
# Only practical when the number of optimised parameters is small (< ~50).
_SCIPY_HESS_METHODS: frozenset[str] = frozenset({"trust-exact"})


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
    angle_wrap = bool(options.pop("angle_wrap", False))

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
        "angle_wrap": angle_wrap,
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
    progress: bool,
    opt_desc: str | None,
):
    iterator = range(n_steps)
    if not progress:
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
    progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    params_run = dict(items)
    specs = _build_param_specs(items)
    options = dict(solver_options)
    # maxeval is an nlopt concept; discard it so it doesn't override n_steps for torch solvers.
    options.pop("maxeval", None)
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
    param_list = list(params_run.values())
    optimizer = optimizer_cls(param_list, lr=lr, **options)

    if isinstance(optimizer, torch.optim.LBFGS):
        # torch.optim.LBFGS._gather_flat_grad uses .view(-1) which fails for
        # complex tensors (stride incompatibility). Patch it to use .reshape(-1).
        def _patched_gather_flat_grad(self_lbfgs):
            views = []
            for group in self_lbfgs.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        view = p.new(p.numel()).zero_()
                    elif p.grad.is_sparse:
                        view = p.grad.to_dense().reshape(-1)
                    else:
                        view = p.grad.reshape(-1)
                    if torch.is_complex(view):
                        view = torch.view_as_real(view).reshape(-1)
                    views.append(view)
            return torch.cat(views, 0)
        import types
        optimizer._gather_flat_grad = types.MethodType(_patched_gather_flat_grad, optimizer)
    scheduler, scheduler_kind = _build_torch_scheduler(
        optimizer,
        scheduler_options,
        max_steps=controls["max_steps"],
    )

    history: list[float] = []
    best_loss = float("inf")
    best_vector = _flatten_params_real_numpy(specs)
    convergence_reason = "maxiter"
    step_iter = _iter_steps(
        controls["max_steps"],
        progress=progress,
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
            if progress:
                step_iter.set_postfix({"||g||": "nan", "loss": "nan", "best": f"{best_loss:.3e}"})
            step_num = step + 1
            if progress_callback is not None and (step_num % log_every == 0):
                progress_callback(step_num, loss_value)
            if (bad_consecutive >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
                convergence_reason = "bad_max"
                break
            continue

        bad_consecutive = 0
        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if loss_value + controls["min_improve"] < best_loss:
            best_loss = loss_value
            best_vector = _flatten_params_real_numpy(specs)
            last_improve_step = step + 1

        if max_step_norm is not None:
            prev_flat = torch.nn.utils.parameters_to_vector(param_list).detach()

        loss.backward()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(param_list, float(grad_clip_value))
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(param_list, float(grad_clip_norm))
        if isinstance(optimizer, torch.optim.LBFGS):
            # LBFGS calls _gather_flat_grad() using the current grads BEFORE calling
            # the closure, so fix contiguity here as well as inside the closure.
            for _p in param_list:
                if isinstance(_p, torch.Tensor) and _p.grad is not None and not _p.grad.is_contiguous():
                    _p.grad.data = _p.grad.data.contiguous()
            def _closure():
                optimizer.zero_grad(set_to_none=True)
                _l = _scalar_real_loss(loss_fn(params_run))
                _l.backward()
                # Also fix inside closure for subsequent line-search evaluations.
                for _p in param_list:
                    if isinstance(_p, torch.Tensor) and _p.grad is not None and not _p.grad.is_contiguous():
                        _p.grad.data = _p.grad.data.contiguous()
                return _l
            optimizer.step(_closure)
        else:
            optimizer.step()

        if max_step_norm is not None:
            with torch.no_grad():
                cur_flat = torch.nn.utils.parameters_to_vector(param_list)
                delta = cur_flat - prev_flat
                delta_norm = float(torch.linalg.vector_norm(delta).detach().cpu().item())
                step_max = float(max_step_norm)
                if delta_norm > step_max:
                    clipped = prev_flat + delta * (step_max / (delta_norm + 1e-12))
                    torch.nn.utils.vector_to_parameters(clipped, param_list)

        _step_torch_scheduler(scheduler, scheduler_kind, loss_value)

        if controls["angle_wrap"]:
            with torch.no_grad():
                _TWO_PI = 2.0 * np.pi
                for spec in specs:
                    if not spec["is_complex"]:
                        t = spec["tensor"]
                        t.data = (t.data + np.pi) % _TWO_PI - np.pi

        if progress:
            _gnorm = float(
                sum(
                    # Use .norm() on the raw (possibly complex) grad — handles real/complex
                    # uniformly without discarding imaginary parts via float cast.
                    float(spec["tensor"].grad.detach().norm().real) ** 2
                    for spec in specs
                    if spec["tensor"].grad is not None
                ) ** 0.5
            )
            step_iter.set_postfix({"||g||": f"{_gnorm:.2e}", "loss": f"{loss_value:.3e}", "best": f"{best_loss:.3e}"})
        step_num = step + 1
        if progress_callback is not None and (step_num % log_every == 0):
            progress_callback(step_num, loss_value)
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - last_improve_step) >= controls["patience"]:
                convergence_reason = "patience"
                break

    if controls["restore_best"]:
        _assign_flat_params(best_vector, specs)
        final_loss = best_loss
    else:
        final_loss = history[-1] if history else float("nan")
    if progress_callback is not None:
        progress_callback(max(1, len(history)), final_loss)

    return params_run, history, best_loss, final_loss, convergence_reason, len(history)


def _run_scipy_lbfgs(
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
    try:
        from scipy import optimize as sp_opt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "solver='scipy' requires SciPy. Install with: pip install scipy"
        ) from exc

    params_run = dict(items)
    specs = _build_param_specs(items)
    x0 = _flatten_params_real_numpy(specs)
    if x0.size == 0:
        return params_run, [], float("nan"), float("nan"), "empty_params"

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
    # Determine whether second-order information is needed for this method.
    _needs_hess = method in _SCIPY_HESS_METHODS    # full n×n Hessian (trust-exact)
    _needs_hessp = method in _SCIPY_HESSP_METHODS  # Hessian-vector product (trust-krylov, Newton-CG)
    if _needs_hess and x0.size > 100:
        warnings.warn(
            f"scipy method {method!r} requires a full {x0.size}×{x0.size} Hessian matrix "
            f"({x0.size} backward passes per step). Consider 'trust-krylov' or "
            "'Newton-CG' with a Hessian-vector product instead.",
            stacklevel=3,
        )
    grad_clip_norm = options.pop("grad_clip_norm", options.pop("clip_grad_norm", None))
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

    # SciPy line-search control alias (mostly useful for L-BFGS-B/TNC).
    if "line_search_max_steps" in options:
        options["maxls"] = int(options.pop("line_search_max_steps"))
    if "maxls" in options and int(options["maxls"]) <= 0:
        raise ValueError("maxls must be >= 1")

    # n_steps is the canonical iteration limit for scipy (= maxiter).
    options["maxiter"] = n_steps
    # ftol is only valid for L-BFGS-B / TNC / SLSQP — Hessian-based methods reject it.
    if not (_needs_hess or _needs_hessp):
        options.setdefault("ftol", ftol)
    # Newton-CG converges on relative step size (xtol); all others use gradient norm (gtol).
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

    # angle_wrap: set bounds to [-π, π] if method supports bounds and none were given.
    if controls["angle_wrap"] and bounds is None and method in _SCIPY_BOUNDS_METHODS:
        n_vars = int(x0.size)
        bounds = [(-np.pi, np.pi)] * n_vars

    history: list[float] = []
    convergence_reason = "maxiter"
    step_counter = {"value": 0}
    state = {
        "last_loss": None,
        "last_gnorm": float("nan"),
        "best_loss": float("inf"),
        "best_x": None,
        "last_improve_step": 0,
        "bad_consecutive": 0,
    }
    pbar = None
    if progress:
        _pbar_desc = f"{opt_desc}[{method}]" if opt_desc else method
        pbar = tqdm(
            total=n_steps,
            desc=_pbar_desc,
            leave=False,
            colour="CYAN",
        )

    eval_counter = {"value": 0}

    def objective(x):
        eval_counter["value"] += 1
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
            if pbar is not None:
                state["last_gnorm"] = float(np.linalg.norm(grad_value))
            if grad_clip_norm is not None:
                grad_norm = float(np.linalg.norm(grad_value))
                clip = float(grad_clip_norm)
                if grad_norm > clip:
                    grad_value = grad_value * (clip / (grad_norm + 1e-12))
            if loss_value + controls["min_improve"] < state["best_loss"]:
                state["best_loss"] = loss_value
                state["best_x"] = np.array(x, dtype=np.float64, copy=True)
                state["last_improve_step"] = step_counter["value"] + 1
        state["last_loss"] = float(loss_value)
        # Update progress bar on every function evaluation (not just callback)
        if pbar is not None:
            pbar.set_postfix({
                "evals": eval_counter["value"],
                "||g||": f"{state['last_gnorm']:.2e}",
                "loss": f"{loss_value:.3e}",
                "best": f"{state['best_loss']:.3e}",
            })
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
            pbar.set_postfix({
                "||g||": f"{state['last_gnorm']:.2e}",
                "loss": f"{loss_value:.3e}",
                "best": f"{state['best_loss']:.3e}",
            })
        if progress_callback is not None and (
            (step_num % log_every == 0) or (step_num == controls["max_steps"])
        ):
            progress_callback(step_num, float(loss_value))
        if (state["bad_consecutive"] >= controls["bad_max"]) and (step_num >= controls["min_steps"]):
            convergence_reason = "bad_max"
            raise StopIteration
        if controls["patience"] is not None and step_num >= controls["min_steps"]:
            if (step_num - state["last_improve_step"]) >= controls["patience"]:
                convergence_reason = "patience"
                raise StopIteration

    hess_fn = None
    hessp_fn = None
    if _needs_hess:
        def hess_fn(x, _specs=specs, _loss_fn=loss_fn):
            return _evaluate_hessian(x, _specs, _loss_fn)
    elif _needs_hessp:
        def hessp_fn(x, p, _specs=specs, _loss_fn=loss_fn):
            return _evaluate_hessp(x, p, _specs, _loss_fn)

    result = None
    try:
        result = sp_opt.minimize(
            objective,
            x0,
            jac=True,
            method=method,
            bounds=bounds,
            hess=hess_fn,
            hessp=hessp_fn,
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

    best_loss = state["best_loss"]
    if controls["restore_best"] and state["best_x"] is not None:
        final_loss = best_loss
    else:
        final_loss = history[-1] if history else float("nan")
    return params_run, history, best_loss, final_loss, convergence_reason, eval_counter["value"]


def _run_nlopt_lbfgs(
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
    try:
        import nlopt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "solver='nlopt' requires NLopt. Install with: pip install nlopt"
        ) from exc

    params_run = dict(items)
    specs = _build_param_specs(items)
    x0 = _flatten_params_real_numpy(specs)
    if x0.size == 0:
        return params_run, [], float("nan"), float("nan"), "empty_params"

    options = dict(solver_options)
    if "algorithm" in options:
        algorithm_name = options.pop("algorithm")
    else:
        algorithm_name = options.pop("optimizer", "LD_LBFGS")
    method_name = options.pop("method", None)
    # method is an alias for algorithm in the nlopt backend; merge into algorithm_name.
    if method_name is not None and algorithm_name == "LD_LBFGS":
        algorithm_name = method_name

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
        # L-BFGS aliases (scipy-style and native)
        "L_BFGS_B": "LD_LBFGS",
        "L_BFGS": "LD_LBFGS",
        "LBFGS": "LD_LBFGS",
        "LD_LBFGS": "LD_LBFGS",
        # Truncated Newton variants
        "TNEWTON": "LD_TNEWTON",
        "LD_TNEWTON": "LD_TNEWTON",
        "TNEWTON_RESTART": "LD_TNEWTON_RESTART",
        "LD_TNEWTON_RESTART": "LD_TNEWTON_RESTART",
        "TNEWTON_PRECOND": "LD_TNEWTON_PRECOND",
        "LD_TNEWTON_PRECOND": "LD_TNEWTON_PRECOND",
        "TNEWTON_PRECOND_RESTART": "LD_TNEWTON_PRECOND_RESTART",
        "LD_TNEWTON_PRECOND_RESTART": "LD_TNEWTON_PRECOND_RESTART",
        # Other gradient-based methods
        "LD_VAR2": "LD_VAR2",
        "VAR2": "LD_VAR2",
        "MMA": "LD_MMA",
        "LD_MMA": "LD_MMA",
        "CCSAQ": "LD_CCSAQ",
        "LD_CCSAQ": "LD_CCSAQ",
        "SLSQP": "LD_SLSQP",
        "LD_SLSQP": "LD_SLSQP",
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

    # angle_wrap: set bounds to [-π, π] if no explicit bounds were given.
    if controls["angle_wrap"] and lower is None and upper is None:
        opt.set_lower_bounds(np.full(int(x0.size), -np.pi))
        opt.set_upper_bounds(np.full(int(x0.size), np.pi))

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
    }
    pbar = None
    # pbar shows n_steps displayed units; each unit covers maxeval/n_steps evaluations
    pbar_step_size = max(1, maxeval // max(1, n_steps))
    if progress:
        _nlopt_alg_name = algorithm_name if isinstance(algorithm_name, str) else f"nlopt({int(algorithm)})"
        _pbar_desc = f"{opt_desc}[{_nlopt_alg_name}]" if opt_desc else _nlopt_alg_name
        pbar = tqdm(total=n_steps, desc=_pbar_desc, leave=False, colour="CYAN")

    def _pbar_advance(evals):
        """Advance pbar proportionally: n_steps units over maxeval total evaluations."""
        if pbar is None:
            return
        new_pos = min(n_steps, evals // pbar_step_size)
        advance = new_pos - pbar.n
        if advance > 0:
            pbar.update(advance)

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
                _pbar_advance(eval_state["evals"])
                pbar.set_postfix({"||g||": "nan", "loss": "nan", "best": f"{eval_state['best_true']:.3e}"})
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
            grad_norm = float(np.linalg.norm(grad_value))
            if pbar is not None:
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
                    if (eval_state["evals"] - eval_state["last_improve_eval"]) > controls["patience"]:
                        eval_state["stopped_reason"] = "patience"
                        opt.force_stop()

        history.append(float(loss_value))
        step_num = len(history)
        if pbar is not None:
            _pbar_advance(eval_state["evals"])
            pbar.set_postfix({
                "||g||": f"{eval_state.get('last_gnorm', float('nan')):.2e}",
                "loss": f"{loss_value:.3e}",
                "best": f"{eval_state['best_true']:.3e}",
            })
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

    best_loss = eval_state["best_true"]
    convergence_reason = eval_state["stopped_reason"]
    if controls["restore_best"] and eval_state["best_x_true"] is not None:
        final_loss = best_loss
    else:
        final_loss = history[-1] if history else float("nan")
    return params_run, history, best_loss, final_loss, convergence_reason, eval_state["evals"]


def _optimize_dispatch(
    params_init: Mapping[str, Any],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver: str,
    solver_options: Mapping[str, Any] | None,
    n_steps: int,
    log_every: int,
    progress: bool,
    opt_desc: str | None,
    progress_callback: Callable[[int, float], None] | None,
):
    """Core backend dispatch used by :class:`GradientOptimizer`."""
    _require_torch()
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

    # lr is only used by torch backends; warn if the user explicitly set it for scipy/nlopt.
    if solver_name not in _TORCH_SOLVERS and "lr" in (solver_options or {}):
        warnings.warn(
            f"'lr' option is ignored by solver={solver_name!r} (torch-only). "
            "Use algorithm-specific options like 'ftol', 'gtol' instead.",
            stacklevel=3,
        )

    items = _param_ordered_items(params_init)
    if solver_name in _TORCH_SOLVERS:
        return _run_torch_solver(
            items,
            loss_fn,
            solver_name=solver_name,
            lr=lr,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            progress=progress,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    if solver_name == "scipy":
        return _run_scipy_lbfgs(
            items,
            loss_fn,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            progress=progress,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    if solver_name == "nlopt":
        return _run_nlopt_lbfgs(
            items,
            loss_fn,
            solver_options=options,
            n_steps=n_steps,
            log_every=log_every,
            progress=progress,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    raise RuntimeError(f"Unhandled solver path for {solver_name!r}")


def optimize_packed_params(
    params_init: Mapping[str, Any],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    solver: str = "scipy",
    solver_options: Mapping[str, Any] | None = None,
    n_steps: int = 100,
    progress: bool = False,
    log_every: int = 1,
    opt_desc: str | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Optimize a dict of packed parameters against a differentiable loss.

    This is the primary entry point used by :class:`pepsy.SweepOptimizer`.
    Parameters may live on any device (CPU or CUDA). ``scipy`` and
    ``nlopt`` flatten params to CPU float64 per evaluation and reconstruct
    them on their original device after each step; ``torch-adam`` keeps
    params on-device throughout.

    Parameters
    ----------
    params_init : dict[str, Any]
        Initial parameter values. Torch tensors are kept on their current
        device; other array types are converted with ``torch.as_tensor``.
    loss_fn : callable
        ``loss_fn(params) -> scalar tensor``.  Must be differentiable when
        using gradient-based solvers.
    solver : str, default="scipy"
        Solver name. Supported values: ``"torch-adam"``, ``"scipy"``,
        and ``"nlopt"``.
    solver_options : dict | None, default=None
        Backend-specific options forwarded to the selected solver.
    n_steps : int, default=100
        Maximum number of optimisation steps / function evaluations.
    progress : bool, default=False
        Show a per-step progress bar with ``||g||``, ``loss``, and ``best``.
    log_every : int, default=1
        Frequency (in steps) for ``progress_callback`` calls.
    opt_desc : str | None, default=None
        Description string for progress bars.
    progress_callback : callable | None, default=None
        ``callback(step, loss_value)`` called every ``log_every`` steps.

    Returns
    -------
    tuple[dict[str, torch.Tensor], list[float]]
        ``(params_opt, history)`` where ``params_opt`` maps parameter names
        to optimised tensors on their original devices (detached, no grad),
        and ``history`` is the per-step loss trace.
    """
    params_opt, history, _best, _final, _reason, _n_evals = _optimize_dispatch(
        params_init,
        loss_fn,
        solver=solver,
        solver_options=solver_options,
        n_steps=n_steps,
        log_every=log_every,
        progress=progress,
        opt_desc=opt_desc,
        progress_callback=progress_callback,
    )
    # Detach returned tensors — optimisation is done, grad tracking not needed.
    params_out = {
        name: t.detach() if isinstance(t, torch.Tensor) else t
        for name, t in params_opt.items()
    }
    return params_out, history


class GradientOptimizer:
    """Simple class wrapper for gradient optimization over packed params.

    The main entry point is :meth:`run`, which accepts ``params_init``,
    ``loss_fn`` and optional ``loss_kwargs``.
    """

    def __init__(
        self,
        *,
        solver: str = "torch-adam",
        options: Mapping[str, Any] | None = None,
        n_steps: int = 100,
        log_every: int = 20,
        progress: bool = False,
        desc: str | None = None,
        verbose: bool = False,
    ):
        self.solver = solver
        self.options = {} if options is None else dict(options)
        self.n_steps = int(n_steps)
        self.log_every = int(log_every)
        self.progress = bool(progress)
        self.desc = desc
        self.verbose = bool(verbose)

    @staticmethod
    def _bind_loss(
        loss_fn: Callable[..., torch.Tensor],
        loss_kwargs: Mapping[str, Any] | None,
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        if loss_kwargs is None:
            return loss_fn
        if not isinstance(loss_kwargs, Mapping):
            raise TypeError("loss_kwargs must be a mapping or None")
        kwargs = dict(loss_kwargs)

        def _wrapped(params):
            return loss_fn(params, **kwargs)

        return _wrapped

    def run(
        self,
        *,
        params_init: Mapping[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_kwargs: Mapping[str, Any] | None = None,
        solver: str | None = None,
        options: Mapping[str, Any] | None = None,
        n_steps: int | None = None,
        log_every: int | None = None,
        progress: bool | None = None,
        desc: str | None = None,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> GradSolverResult:
        """Run optimization and return a structured result."""
        solver_use = self.solver if solver is None else solver
        options_use = dict(self.options)
        if options is not None:
            options_use.update(dict(options))
        n_steps_use = self.n_steps if n_steps is None else int(n_steps)
        log_every_use = self.log_every if log_every is None else int(log_every)
        progress_use = self.progress if progress is None else bool(progress)
        desc_use = self.desc if desc is None else desc

        bound_loss = self._bind_loss(loss_fn, loss_kwargs)
        solver_name = _normalize_solver_name(solver_use)

        if self.verbose:
            print(
                f"[GradientOptimizer] solver={solver_name} "
                f"n_steps={n_steps_use} progress={progress_use}"
            )

        params_opt, history, best_loss, final_loss, convergence_reason, n_evals = _optimize_dispatch(
            params_init,
            bound_loss,
            solver=solver_use,
            solver_options=options_use,
            n_steps=n_steps_use,
            log_every=log_every_use,
            progress=progress_use,
            opt_desc=desc_use,
            progress_callback=progress_callback,
        )
        if not history:
            final_loss = float("nan")
            best_loss = float("nan")

        if self.verbose:
            print(
                f"[GradientOptimizer] done solver={solver_name} "
                f"steps={len(history)} final={final_loss:.6g} best={best_loss:.6g}"
            )

        return GradSolverResult(
            params=params_opt,
            history=list(history),
            solver=solver_name,
            n_steps=len(history),
            best_loss=float(best_loss),
            final_loss=float(final_loss),
            convergence_reason=convergence_reason,
            n_evals=n_evals,
        )
