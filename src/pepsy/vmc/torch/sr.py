"""Stochastic-reconfiguration and log-derivative helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..torch_types import _require_torch
from ._common import _as_long_matrix, _check_nonnegative_int

__all__ = [
    "TorchSRResult",
    "apply_torch_sr_update",
    "solve_torch_sr",
    "torch_log_derivative_matrix",
    "_spring_complement",
]


@dataclass(frozen=True)
class TorchSRResult:
    """Result of a torch stochastic-reconfiguration linear solve."""

    direction: Any
    energy_mean: Any
    energy_variance: Any
    force: Any
    centered_log_derivatives: Any
    method: str
    diag_shift: float
    info: dict[str, Any]


def _torch_model_parameters(model):
    try:
        params = list(model.parameters())
    except AttributeError as exc:
        raise TypeError("model must expose a parameters() method.") from exc
    if not params:
        raise ValueError("model must expose at least one trainable parameter.")
    return params


def _flatten_torch_tensors(tensors, refs):
    torch = _require_torch()
    pieces = []
    for tensor, ref in zip(tensors, refs, strict=True):
        if tensor is None:
            tensor = torch.zeros_like(ref)
        pieces.append(tensor.reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def _log_derivative_denominator(amplitudes, amplitude_floor):
    """Build stable per-sample denominators for log-amplitude derivatives."""
    torch = _require_torch()
    amplitudes = torch.as_tensor(amplitudes)
    amplitude_abs = amplitudes.detach().abs()
    if amplitude_floor is None:
        if bool(torch.any(amplitude_abs == 0)):
            raise ZeroDivisionError(
                "Encountered a zero amplitude while forming log derivatives."
            )
        return amplitudes

    floor = torch.as_tensor(
        amplitude_floor,
        dtype=(
            amplitudes.real.dtype
            if torch.is_complex(amplitudes)
            else amplitudes.dtype
        ),
        device=amplitudes.device,
    )
    if torch.is_complex(amplitudes):
        phase = torch.where(
            amplitude_abs > 0,
            amplitudes / amplitude_abs.to(dtype=amplitudes.dtype),
            torch.ones_like(amplitudes),
        )
        return torch.where(
            amplitude_abs < floor,
            phase * floor.to(dtype=amplitudes.dtype),
            amplitudes,
        )

    sign = torch.where(
        amplitudes.detach() >= 0,
        torch.ones_like(amplitudes),
        -torch.ones_like(amplitudes),
    )
    return torch.where(amplitude_abs < floor, sign * floor, amplitudes)


def _batched_model_log_derivatives(
    model,
    configs,
    *,
    amplitude_floor,
    create_graph,
    complex_parameter_mode,
):
    """Evaluate PEPS log derivatives with one batched Jacobian graph.

    ``TorchPEPSAmplitude`` accepts an explicit parameter tuple, which lets
    ``torch.autograd.functional.jacobian`` differentiate all walker
    amplitudes together without mutating the model's registered parameters.
    The function deliberately targets that parameterized PEPS interface;
    generic amplitude models use the compatibility loop instead.
    """
    torch = _require_torch()
    if not callable(getattr(model, "_params_pytree", None)):
        raise TypeError("model does not expose the functional PEPS parameter API.")
    configs = _as_long_matrix(configs)
    params = tuple(_torch_model_parameters(model))
    if not params:
        raise ValueError("model must expose at least one trainable parameter.")

    mode = str(complex_parameter_mode).replace("_", "-").lower()
    if mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError(
            "complex_parameter_mode must be 'holomorphic' or 'real-imag'."
        )
    real_imag_mode = mode in {"real-imag", "real-imaginary"}

    def amplitudes_with_params(*values):
        amplitudes = torch.as_tensor(model(configs, params=values)).reshape(-1)
        if amplitudes.numel() != configs.shape[0]:
            raise ValueError(
                "model must return one amplitude per configuration row."
            )
        return amplitudes

    amplitudes = amplitudes_with_params(*params)
    denominator = _log_derivative_denominator(amplitudes, amplitude_floor)
    complex_output = torch.is_complex(amplitudes)
    complex_parameters = tuple(torch.is_complex(param) for param in params)
    need_real_and_imag = complex_output and (
        real_imag_mode or not all(complex_parameters)
    )

    if need_real_and_imag:
        jacobian = torch.autograd.functional.jacobian(
            lambda *values: torch.view_as_real(amplitudes_with_params(*values)),
            params,
            create_graph=create_graph,
            strict=False,
            vectorize=True,
        )
        jacobian_real = tuple(value[:, 0, ...] for value in jacobian)
        jacobian_imag = tuple(value[:, 1, ...] for value in jacobian)
    else:
        jacobian_real = torch.autograd.functional.jacobian(
            lambda *values: (
                amplitudes_with_params(*values).real
                if complex_output
                else amplitudes_with_params(*values)
            ),
            params,
            create_graph=create_graph,
            strict=False,
            vectorize=True,
        )
        jacobian_real = tuple(jacobian_real)
        jacobian_imag = (None,) * len(params)

    pieces = []
    for param, real_grad, imag_grad in zip(
        params,
        jacobian_real,
        jacobian_imag,
        strict=True,
    ):
        real_grad = real_grad.reshape(configs.shape[0], -1)
        if complex_output:
            if torch.is_complex(param):
                if real_imag_mode:
                    imag_grad = imag_grad.reshape(configs.shape[0], -1)
                    derivative_real = real_grad.real + 1j * imag_grad.real
                    derivative_imag = real_grad.imag + 1j * imag_grad.imag
                    pieces.append(
                        torch.stack((derivative_real, derivative_imag), dim=-1)
                        .reshape(configs.shape[0], -1)
                    )
                else:
                    # For a holomorphic f(z), autograd's real-output
                    # derivative is conjugated before forming df / f.
                    pieces.append(real_grad.conj())
            else:
                imag_grad = imag_grad.reshape(configs.shape[0], -1)
                pieces.append(real_grad + 1j * imag_grad)
        elif real_imag_mode and torch.is_complex(param):
            pieces.append(
                torch.stack((real_grad.real, real_grad.imag), dim=-1)
                .reshape(configs.shape[0], -1)
            )
        else:
            pieces.append(real_grad)

    result = torch.cat(pieces, dim=1) / denominator.reshape(-1, 1)
    return result if create_graph else result.detach()


def _torch_log_derivative_matrix_loop(
    model,
    configs,
    *,
    amplitude_floor=None,
    create_graph=False,
    complex_parameter_mode="holomorphic",
):
    """Return per-sample log-amplitude derivatives for a torch model.

    The returned matrix has shape ``(n_samples, n_params)`` and entries
    ``d psi(config) / d theta / psi(config)``. Real parameters use the
    ordinary real derivative, while complex parameters use the explicitly
    selected ``complex_parameter_mode``. The default ``"holomorphic"`` mode
    is appropriate for packed PEPS amplitudes, which are holomorphic in their
    complex tensor entries, and returns one complex derivative per complex
    parameter. In ``"real-imag"`` mode, each complex parameter contributes
    two interleaved columns, ``d log(psi) / d Re(theta)`` and
    ``d log(psi) / d Im(theta)``.

    Complex parameters are not treated as real parameters implicitly. The
    holomorphic convention is used by :func:`TorchVMCDriver.step` and by
    :func:`apply_torch_sr_update` for complex PEPS tensors.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    params = _torch_model_parameters(model)
    parameter_mode = str(complex_parameter_mode).replace("_", "-").lower()
    if parameter_mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError(
            "complex_parameter_mode must be 'holomorphic' or 'real-imag'."
        )
    real_imag_mode = parameter_mode in {"real-imag", "real-imaginary"}
    rows = []

    for config in configs:
        amp = model(config.reshape(1, -1))
        amp = torch.as_tensor(amp).reshape(-1)
        if amp.numel() != 1:
            raise ValueError("model(config) must return one amplitude per row.")
        amp = amp[0]
        if not amp.requires_grad:
            raise RuntimeError("model amplitude does not require gradients.")

        amp_abs = amp.detach().abs()
        if amplitude_floor is None:
            if amp_abs.item() == 0:
                raise ZeroDivisionError(
                    "Encountered a zero amplitude while forming log derivatives."
                )
            denom = amp
        else:
            floor = torch.as_tensor(
                amplitude_floor,
                dtype=amp.real.dtype if torch.is_complex(amp) else amp.dtype,
                device=amp.device,
            )
            if torch.is_complex(amp):
                phase = torch.where(
                    amp_abs > 0,
                    amp / amp_abs.to(dtype=amp.dtype),
                    torch.ones_like(amp),
                )
                denom = torch.where(
                    amp_abs < floor,
                    phase * floor.to(dtype=amp.dtype),
                    amp,
                )
            else:
                sign = torch.where(
                    amp.detach() >= 0,
                    torch.ones_like(amp),
                    -torch.ones_like(amp),
                )
                denom = torch.where(amp_abs < floor, sign * floor, amp)

        grad_real = torch.autograd.grad(
            amp.real if torch.is_complex(amp) else amp,
            params,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        if torch.is_complex(amp) and amp.imag.requires_grad:
            grad_imag = torch.autograd.grad(
                amp.imag,
                params,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
        else:
            grad_imag = (None,) * len(params)

        derivative_pieces = []
        for param, real_grad, imag_grad in zip(
            params,
            grad_real,
            grad_imag,
            strict=True,
        ):
            if real_grad is None:
                real_grad = torch.zeros_like(param)
            if torch.is_complex(amp):
                if imag_grad is None:
                    imag_grad = torch.zeros_like(param)
                if torch.is_complex(param) and not real_imag_mode:
                    # For a holomorphic f(z), torch's gradient of Re[f] with
                    # respect to z is conjugate(df / dz).
                    derivative = real_grad.conj()
                elif torch.is_complex(param):
                    # PyTorch encodes the coordinate gradients of a real
                    # scalar with real and imaginary parts of its complex
                    # gradient. Recover both output components explicitly.
                    derivative = (
                        real_grad.real + 1j * imag_grad.real,
                        real_grad.imag + 1j * imag_grad.imag,
                    )
                else:
                    # Real parameters need both output components to recover
                    # the complex derivative along the real parameter axis.
                    derivative = real_grad + 1j * imag_grad
            else:
                if real_imag_mode and torch.is_complex(param):
                    derivative = (real_grad.real, real_grad.imag)
                else:
                    derivative = real_grad
            if isinstance(derivative, tuple):
                derivative_pieces.append(
                    torch.stack(derivative, dim=-1).reshape(-1)
                )
            else:
                derivative_pieces.append(derivative.reshape(-1))

        row = torch.cat(derivative_pieces) / denom
        if not create_graph:
            row = row.detach()
        rows.append(row)

    return torch.stack(rows, dim=0)


def torch_log_derivative_matrix(
    model,
    configs,
    *,
    amplitude_floor=None,
    create_graph=False,
    complex_parameter_mode="holomorphic",
    derivative_backend="auto",
):
    """Return per-sample log-amplitude derivatives for a torch model.

    The returned matrix has shape ``(n_samples, n_params)`` and entries
    ``d psi(config) / d theta / psi(config)``. ``derivative_backend="auto"``
    uses one batched Jacobian for functional PEPS amplitude models and falls
    back to the original per-sample autograd loop for generic models or
    unsupported contraction transformations. Use ``"loop"`` to force the
    compatibility path or ``"batched"`` to require the fast PEPS path.

    Complex parameters use the explicitly selected
    ``complex_parameter_mode``. In ``"holomorphic"`` mode, one complex
    derivative is returned per complex parameter. In ``"real-imag"`` mode,
    each complex parameter contributes interleaved real and imaginary
    coordinate derivatives.
    """
    backend = str(derivative_backend).replace("_", "-").lower()
    if backend not in {"auto", "batched", "loop", "scalar"}:
        raise ValueError(
            "derivative_backend must be 'auto', 'batched', 'loop', or 'scalar'."
        )
    if backend in {"auto", "batched"}:
        try:
            return _batched_model_log_derivatives(
                model,
                configs,
                amplitude_floor=amplitude_floor,
                create_graph=create_graph,
                complex_parameter_mode=complex_parameter_mode,
            )
        except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
            if backend == "batched":
                raise
    return _torch_log_derivative_matrix_loop(
        model,
        configs,
        amplitude_floor=amplitude_floor,
        create_graph=create_graph,
        complex_parameter_mode=complex_parameter_mode,
    )


def _promote_sr_tensors(log_derivatives, local_energies):
    torch = _require_torch()
    log_derivatives = torch.as_tensor(log_derivatives)
    if log_derivatives.ndim != 2:
        raise ValueError("log_derivatives must have shape (n_samples, n_params).")
    if not torch.is_floating_point(log_derivatives) and not torch.is_complex(
        log_derivatives
    ):
        log_derivatives = log_derivatives.to(torch.float64)

    local_energies = torch.as_tensor(local_energies, device=log_derivatives.device)
    if local_energies.ndim != 1:
        local_energies = local_energies.reshape(-1)
    if local_energies.shape[0] != log_derivatives.shape[0]:
        raise ValueError("local_energies must have one entry per sample.")
    if not torch.is_floating_point(local_energies) and not torch.is_complex(
        local_energies
    ):
        local_energies = local_energies.to(log_derivatives.dtype)

    dtype = torch.promote_types(log_derivatives.dtype, local_energies.dtype)
    return log_derivatives.to(dtype), local_energies.to(dtype)


def _resolve_sr_diag_shift(diag_shift, *, step):
    """Resolve a constant or step-indexed SR diagonal shift."""
    value = diag_shift(step) if callable(diag_shift) else diag_shift
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "diag_shift must be a non-negative number or a callable returning one."
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "diag_shift must be a non-negative finite number or schedule value."
        )
    return value


def _torch_solve_linear(matrix, rhs, *, pinv_rtol=None):
    """Solve a Hermitian SR system, falling back to a stable pseudoinverse."""
    torch = _require_torch()
    try:
        factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        if bool(torch.all(info == 0)):
            solution = torch.cholesky_solve(
                rhs.reshape(-1, 1),
                factor,
            ).reshape_as(rhs)
            if bool(torch.isfinite(solution).all()):
                return solution, "cholesky"
    except RuntimeError:
        pass

    pinv_kwargs = {"hermitian": True}
    if pinv_rtol is not None:
        pinv_kwargs["rtol"] = pinv_rtol
    solution = torch.linalg.pinv(matrix, **pinv_kwargs) @ rhs
    if not bool(torch.isfinite(solution).all()):
        raise RuntimeError(
            "The SR pseudoinverse fallback produced non-finite values. "
            "Increase diag_shift or inspect the local-energy samples."
        )
    return solution, "pinv"


def _spring_complement(metric_source, previous_direction, *, pinv_rtol=None):
    """Return the previous SR update outside the current sampled tangent span."""
    torch = _require_torch()
    previous_direction = torch.as_tensor(
        previous_direction,
        dtype=metric_source.dtype,
        device=metric_source.device,
    ).reshape(-1)
    if previous_direction.shape[0] != metric_source.shape[1]:
        raise ValueError(
            "previous_direction must have one entry per SR parameter."
        )
    tangent = (
        metric_source.transpose(0, 1)
        if not torch.is_complex(metric_source)
        else metric_source.conj().transpose(0, 1)
    )
    if tangent.shape[1] == 0:
        return previous_direction
    solve_kwargs = {}
    if pinv_rtol is not None:
        solve_kwargs["rcond"] = pinv_rtol
    coefficients = torch.linalg.lstsq(
        tangent,
        previous_direction,
        **solve_kwargs,
    ).solution
    return previous_direction - tangent @ coefficients


def solve_torch_sr(
    log_derivatives,
    local_energies,
    *,
    sample_weights=None,
    diag_shift=1.0e-4,
    method="auto",
    center=True,
    parameter_mode="holomorphic",
    step=0,
    pinv_rtol=None,
    momentum=None,
    previous_direction=None,
):
    """Solve direct SR or sample-space minSR for a torch VMC batch.

    ``method="direct"`` forms the parameter-space covariance matrix.
    ``method="minsr"`` solves the equivalent sample-space system, which is
    preferable when the number of PEPS parameters is much larger than the
    number of Monte Carlo samples. ``method="auto"`` picks minSR when
    ``n_samples < n_params``. Complex derivatives use the Hermitian covariance
    ``centered.conj().T @ centered`` and return a complex SR direction under
    the holomorphic parameter convention. ``parameter_mode="real-imag"``
    instead solves a real SR system for the explicit real and imaginary
    parameter coordinates returned by :func:`torch_log_derivative_matrix`.
    ``diag_shift`` can be a callable of the non-negative integer ``step``.
    The Hermitian system uses a Cholesky solve when possible and otherwise a
    pseudoinverse fallback. Passing ``momentum`` together with a previous
    direction applies a SPRING-style complement: only the part of the prior
    update outside the current sampled tangent span is retained. Pass
    normalized or relative ``sample_weights`` to form the corresponding
    weighted energy and tangent-space covariances.
    """
    torch = _require_torch()
    step = _check_nonnegative_int("step", step)
    diag_shift = _resolve_sr_diag_shift(diag_shift, step=step)
    if pinv_rtol is not None:
        try:
            pinv_rtol = float(pinv_rtol)
        except (TypeError, ValueError) as exc:
            raise ValueError("pinv_rtol must be a positive finite number or None.") from exc
        if not math.isfinite(pinv_rtol) or pinv_rtol <= 0.0:
            raise ValueError("pinv_rtol must be a positive finite number or None.")
    if momentum is not None:
        try:
            momentum = float(momentum)
        except (TypeError, ValueError) as exc:
            raise ValueError("momentum must be in [0, 1).") from exc
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
    parameter_mode = str(parameter_mode).replace("_", "-").lower()
    if parameter_mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
    real_imag_mode = parameter_mode in {"real-imag", "real-imaginary"}
    log_derivatives, local_energies = _promote_sr_tensors(
        log_derivatives,
        local_energies,
    )
    n_samples, n_params = log_derivatives.shape
    if n_samples == 0 or n_params == 0:
        raise ValueError("SR requires at least one sample and one parameter.")

    method_key = str(method).replace("_", "").replace("-", "").lower()
    if method_key == "auto":
        method_key = "minsr" if n_samples < n_params else "direct"
    if method_key not in {"direct", "sr", "minsr"}:
        raise ValueError("method must be 'auto', 'direct', or 'minsr'.")
    if method_key == "sr":
        method_key = "direct"

    if sample_weights is None:
        weights = torch.full(
            (n_samples,),
            1.0 / n_samples,
            dtype=local_energies.real.dtype,
            device=local_energies.device,
        )
    else:
        weights = torch.as_tensor(sample_weights, device=local_energies.device)
        if weights.ndim != 1 or weights.shape[0] != n_samples:
            raise ValueError("sample_weights must have one entry per sample.")
        if torch.is_complex(weights):
            raise ValueError("sample_weights must be real, finite, and non-negative.")
        if not torch.is_floating_point(weights):
            weights = weights.to(local_energies.real.dtype)
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
            raise ValueError("sample_weights must be real, finite, and non-negative.")
        total_weight = weights.sum()
        if not bool(torch.isfinite(total_weight)) or bool(total_weight <= 0):
            raise ValueError("sample_weights must have a positive finite sum.")
        weights = weights / total_weight

    energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
    energy_residual = local_energies - energy_mean if center else local_energies
    centered = (
        log_derivatives
        - (weights.to(log_derivatives.dtype).reshape(-1, 1) * log_derivatives)
        .sum(dim=0, keepdim=True)
        if center
        else log_derivatives
    )
    sqrt_weights = weights.sqrt().reshape(-1, 1)
    if real_imag_mode:
        # For real coordinates, write C = A + i B and solve with the real
        # design matrix [A; B]. This gives Re(C^H C) and Re(C^H E) while
        # retaining an exact direct/minSR equivalence.
        centered_imag = (
            centered.imag if torch.is_complex(centered) else torch.zeros_like(centered)
        )
        energy_imag = (
            energy_residual.imag
            if torch.is_complex(energy_residual)
            else torch.zeros_like(energy_residual)
        )
        design = torch.cat(
            (sqrt_weights * centered.real, sqrt_weights * centered_imag),
            dim=0,
        )
        solve_energy = torch.cat(
            (sqrt_weights.reshape(-1) * energy_residual.real,
             sqrt_weights.reshape(-1) * energy_imag)
        )
        force = design.transpose(0, 1) @ solve_energy
        metric_source = design
        sr_dtype = design.dtype
    else:
        metric_source = sqrt_weights.to(log_derivatives.dtype) * centered
        solve_energy = (
            sqrt_weights.reshape(-1).to(local_energies.dtype) * energy_residual
        )
        force = metric_source.conj().transpose(0, 1) @ solve_energy
        sr_dtype = log_derivatives.dtype
    shift = torch.as_tensor(
        diag_shift,
        dtype=sr_dtype,
        device=log_derivatives.device,
    )

    if method_key == "direct":
        eye = torch.eye(
            n_params,
            dtype=sr_dtype,
            device=log_derivatives.device,
        )
        if real_imag_mode:
            sr_matrix = metric_source.transpose(0, 1) @ metric_source
        else:
            sr_matrix = metric_source.conj().transpose(0, 1) @ metric_source
        system = sr_matrix + shift * eye
        direction, solver = _torch_solve_linear(
            system,
            force,
            pinv_rtol=pinv_rtol,
        )
        solve_vector = direction
        solve_rhs = force
        matrix_shape = tuple(sr_matrix.shape)
    else:
        n_system = metric_source.shape[0]
        eye = torch.eye(
            n_system,
            dtype=sr_dtype,
            device=log_derivatives.device,
        )
        if real_imag_mode:
            gram = metric_source @ metric_source.transpose(0, 1)
        else:
            gram = metric_source @ metric_source.conj().transpose(0, 1)
        system = gram + shift * eye
        alpha, solver = _torch_solve_linear(
            system,
            solve_energy,
            pinv_rtol=pinv_rtol,
        )
        if real_imag_mode:
            direction = metric_source.transpose(0, 1) @ alpha
        else:
            direction = metric_source.conj().transpose(0, 1) @ alpha
        solve_vector = alpha
        solve_rhs = solve_energy
        matrix_shape = tuple(gram.shape)

    spring_complement_norm = None
    if momentum is not None and momentum > 0.0 and previous_direction is not None:
        spring_complement = _spring_complement(
            metric_source,
            previous_direction,
            pinv_rtol=pinv_rtol,
        )
        direction = direction + momentum * spring_complement
        spring_complement_norm = float(spring_complement.norm().detach().cpu())

    energy_variance = (weights * energy_residual.abs().square()).sum().real
    residual = system @ solve_vector - solve_rhs
    residual_norm = residual.norm()
    rhs_norm = solve_rhs.norm()
    relative_residual = residual_norm / rhs_norm.clamp_min(
        torch.finfo(rhs_norm.real.dtype).tiny
    )
    return TorchSRResult(
        direction=direction,
        energy_mean=energy_mean,
        energy_variance=energy_variance.real,
        force=force,
        centered_log_derivatives=centered,
        method=method_key,
        diag_shift=float(diag_shift),
        info={
            "solver": solver,
            "matrix_shape": matrix_shape,
            "residual_norm": float(residual_norm.detach().cpu()),
            "relative_residual": float(relative_residual.detach().cpu()),
            "step": step,
            "pinv_rtol": pinv_rtol,
            "momentum": momentum,
            "spring_complement_norm": spring_complement_norm,
            "effective_sample_size": float((1.0 / weights.square().sum()).detach().cpu()),
        },
    )


def apply_torch_sr_update(
    model,
    direction,
    *,
    learning_rate=1.0,
    parameter_mode="holomorphic",
):
    """Apply an SR direction in place.

    ``parameter_mode="holomorphic"`` applies one complex direction per
    complex parameter. ``parameter_mode="real-imag"`` consumes interleaved
    real and imaginary coordinate updates for each complex parameter.
    """
    torch = _require_torch()
    params = _torch_model_parameters(model)
    parameter_mode = str(parameter_mode).replace("_", "-").lower()
    if parameter_mode not in {
        "holomorphic",
        "holomorphic-complex",
        "real-imag",
        "real-imaginary",
    }:
        raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
    real_imag_mode = parameter_mode in {"real-imag", "real-imaginary"}
    n_params = sum(
        (2 if real_imag_mode and torch.is_complex(param) else 1) * param.numel()
        for param in params
    )
    direction = torch.as_tensor(direction)
    if direction.numel() != n_params:
        raise ValueError(
            f"direction has {direction.numel()} entries, expected {n_params}."
        )

    offset = 0
    with torch.no_grad():
        for param in params:
            size = param.numel()
            if real_imag_mode and torch.is_complex(param):
                coordinate_updates = direction[
                    offset:offset + 2 * size
                ].reshape(-1, 2)
                real_update = coordinate_updates[:, 0].reshape_as(param.real)
                imag_update = coordinate_updates[:, 1].reshape_as(param.real)
                update = real_update + 1j * imag_update
                offset += 2 * size
            else:
                update = direction[offset:offset + size].reshape_as(param)
                offset += size
            if torch.is_complex(update) and not torch.is_complex(param):
                if update.imag.abs().max().item() > 1.0e-12:
                    raise ValueError(
                        "Cannot apply a complex SR direction to real parameters."
                    )
                update = update.real
            update = update.to(dtype=param.dtype, device=param.device)
            param.sub_(learning_rate * update)
