"""Implicit, fixed-point Torch differentiation for dense FIT updates.

This module is intentionally opt-in. The forward pass converges a one-site
FIT fixed point (after an optional multi-site rank-opening warm-up); backward
solves the linearized fixed-point equation with matrix-free GMRES.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from math import prod
from numbers import Integral

import quimb.tensor as qtn


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _polar_decomposition(torch, matrix):
    scale = matrix.abs().amax()
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    u, singular, vh = torch.linalg.svd(matrix / safe_scale, full_matrices=False)
    resolved = (scale > 0) & torch.all(
        singular > matrix.shape[0] * torch.finfo(singular.dtype).eps * singular[0]
    )
    q = (u @ vh) * torch.where(resolved, 1., torch.nan)
    return q, singular, vh, safe_scale


def _polar_jvp(torch, q, singular, vh, scale, tangent):
    adjoint = lambda x: x.mH
    scaled = tangent / scale
    skew = adjoint(q) @ scaled - adjoint(scaled) @ q
    core = vh @ skew @ adjoint(vh)
    denominator = singular[:, None] + singular[None, :]
    omega = adjoint(vh) @ (core / denominator) @ vh
    return q @ omega


def _polar_unitary(torch, matrix):
    """Polar factor with a repeated-singular-value-safe custom pullback."""
    class Polar(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            q, singular, vh, scale = _polar_decomposition(torch, value)
            ctx.save_for_backward(q, singular, vh, scale, value)
            return q

        @staticmethod
        def backward(ctx, cotangent):
            q, singular, vh, scale, value = ctx.saved_tensors
            with torch.enable_grad():
                tangent = torch.zeros_like(value, requires_grad=True)
                differential = _polar_jvp(
                    torch, q.detach(), singular.detach(), vh.detach(),
                    scale.detach(), tangent,
                )
                gradient, = torch.autograd.grad(
                    differential, tangent, grad_outputs=cotangent,
                    allow_unused=False,
                )
            return gradient

    return Polar.apply(matrix)


def _gmres(torch, matvec, rhs, *, tolerance, max_iterations, restart=24):
    """Small restarted GMRES implementation for a matrix-free real vector."""
    x = torch.zeros_like(rhs)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).tiny)
    restart = min(int(restart), int(rhs.numel()))
    for _ in range(max_iterations):
        residual = rhs - matvec(x)
        beta = torch.linalg.vector_norm(residual)
        if not bool(torch.isfinite(beta)):
            raise RuntimeError("FIT adjoint solve produced a non-finite residual")
        if float(beta / rhs_norm) <= tolerance:
            return x
        basis = [residual / beta]
        hessenberg = torch.zeros(
            (restart + 1, restart), dtype=rhs.dtype, device=rhs.device
        )
        best = x
        for column in range(restart):
            value = matvec(basis[column])
            for row in range(column + 1):
                hessenberg[row, column] = torch.dot(basis[row], value)
                value = value - hessenberg[row, column] * basis[row]
            hessenberg[column + 1, column] = torch.linalg.vector_norm(value)
            if float(hessenberg[column + 1, column]) > 0:
                basis.append(value / hessenberg[column + 1, column])
            target = torch.zeros(column + 2, dtype=rhs.dtype, device=rhs.device)
            target[0] = beta
            coeff = torch.linalg.lstsq(
                hessenberg[:column + 2, :column + 1], target
            ).solution
            best = x + torch.stack(basis[:column + 1], dim=1) @ coeff
            error = torch.linalg.vector_norm(rhs - matvec(best)) / rhs_norm
            if not bool(torch.isfinite(error)):
                raise RuntimeError("FIT adjoint solve produced a non-finite residual")
            if float(error) <= tolerance:
                return best
        x = best
    residual = torch.linalg.vector_norm(rhs - matvec(x)) / rhs_norm
    raise RuntimeError(
        f"FIT adjoint did not converge: relative residual {float(residual):g} > {tolerance:g}"
    )


class TorchFitParameters(dict):
    """Packed fit tensors carrying the skeleton produced by this evaluation."""

    def __init__(self, values, skeleton):
        super().__init__(values)
        self.skeleton = skeleton


@dataclass
class TorchFunctionalFIT:
    """Eager Torch callable around the shared FIT kernel.

    It accepts one dense, open-boundary MPS target and a dense MPS guess.
    The guess is a warm start on the optimized interval; tensors outside the
    interval remain direct identity paths to the input guess.
    """

    target_params: dict
    initial_params: dict
    _target_skeleton: object
    _guess_skeleton: object
    _target_keys: tuple
    _guess_keys: tuple
    _guess_site_keys: tuple
    _window: tuple[int, int]
    _max_bond: int
    _warmup_block_size: int
    _warmup_sweeps: int
    _transition_block_size: int
    _max_iterations: int
    _tolerance: float | None
    _adjoint_tolerance: float
    _adjoint_maxiter: int
    _contraction_opt: object
    _output_skeleton: object = None
    iterations_run: int = 0
    relative_residual: float | None = None
    converged: bool = False

    def __call__(self, target_params=None, guess_params=None):
        return self._eager_call(target_params, guess_params)

    def _eager_call(self, target_params, guess_params):
        import torch

        target = self.target_params if target_params is None else target_params
        guess = self.initial_params if guess_params is None else guess_params
        self._validate_tree(target, "target_params")
        self._validate_tree(guess, "guess_params")
        self._output_skeleton = self._guess_skeleton
        target_leaves = tuple(target[key] for key in self._target_keys)
        guess_leaves = tuple(guess[key] for key in self._guess_keys)
        output = _ImplicitFIT.apply(self, *target_leaves, *guess_leaves)
        if len(self._guess_keys) == 1 and isinstance(output, torch.Tensor):
            output = (output,)
        return TorchFitParameters(
            {key: value for key, value in zip(self._guess_keys, output, strict=True)},
            self._output_skeleton,
        )

    def to_network(self, params):
        skeleton = getattr(params, "skeleton", self._output_skeleton)
        if skeleton is None:
            raise ValueError("call the fit before reconstructing its output")
        return qtn.unpack(params, skeleton)

    def _validate_tree(self, params, name):
        keys = self._target_keys if name == "target_params" else self._guess_keys
        if not isinstance(params, dict) or tuple(sorted(params)) != keys:
            raise ValueError(f"{name} must preserve the packed parameter mapping")
        template = self.target_params if name == "target_params" else self.initial_params
        for key in keys:
            if params[key].shape != template[key].shape:
                raise ValueError(f"{name}[{key}] must preserve its packed shape")
            if params[key].dtype != template[key].dtype:
                raise ValueError(f"{name}[{key}] must preserve its packed dtype")
            if params[key].device != template[key].device:
                raise ValueError(f"{name}[{key}] must preserve its Torch device")

    def _network(self, arrays, skeleton):
        return qtn.unpack(
            {key: value for key, value in zip(self._guess_keys, arrays, strict=True)},
            skeleton,
        )

    def _run_fit(
        self, target_arrays, guess_arrays, *, block_size, sweeps,
        expected_shapes=None,
    ):
        import torch

        from .local import FIT

        target = qtn.unpack(
            {key: value for key, value in zip(self._target_keys, target_arrays, strict=True)},
            self._target_skeleton,
        )
        guess_skeleton = self._output_skeleton or self._guess_skeleton
        guess = qtn.unpack(
            {key: value for key, value in zip(self._guess_keys, guess_arrays, strict=True)},
            guess_skeleton,
        )
        target.distribute_exponent(new_exponent=guess.exponent)
        fit = FIT(
            target, p=guess, cutoffs=0., contraction_opt=self._contraction_opt,
            range_int=(self._window if self._window[0] < self._window[1] else None),
            inplace=True, copy_target=False,
            split_method="svd",
        )
        fit.run_gate(
            n_iter=sweeps, block_size=block_size, max_bond=self._max_bond,
            cutoff=0., rtol=None,
            adaptive_block_sweeps=(sweeps if block_size > 1 else None),
            two_site_transition_sweeps=0, final_one_site_sweeps=0,
            collect_split_diagnostics=False,
        )
        if any(not isinstance(t.data, torch.Tensor) for t in fit.p):
            raise TypeError("Torch implicit FIT requires dense Torch tensor arrays")
        arrays, skeleton = qtn.pack(fit.p)
        actual_shapes = tuple(tuple(arrays[key].shape) for key in self._guess_keys)
        if tuple(sorted(arrays)) != self._guess_keys:
            raise ValueError("Implicit FIT changed its packed tensor keys")
        if expected_shapes is not None and actual_shapes != expected_shapes:
            raise ValueError(
                "Implicit FIT requires fixed packed tensor keys and shapes; "
                f"expected {expected_shapes}, received {actual_shapes}"
            )
        self._output_skeleton = skeleton
        return tuple(arrays[key] for key in self._guess_keys)

    def _decode_active(self, vector, shapes, complex_flags):
        import torch

        arrays = []
        offset = 0
        for shape, is_complex in zip(shapes, complex_flags, strict=True):
            size = prod(shape)
            if is_complex:
                value = torch.view_as_complex(
                    vector[offset:offset + 2 * size].reshape(*shape, 2).contiguous()
                )
                offset += 2 * size
            else:
                value = vector[offset:offset + size].reshape(shape)
                offset += size
            arrays.append(value)
        return arrays

    def _encode_active(self, arrays):
        import torch

        return torch.cat([
            torch.view_as_real(value).reshape(-1) if value.is_complex()
            else value.reshape(-1)
            for value in arrays
        ])

    def _encode_cotangent(self, arrays):
        import torch

        pieces = []
        for value in arrays:
            if value.is_complex():
                pieces.append(torch.stack((value.real, value.imag), dim=-1).reshape(-1))
            else:
                pieces.append(value.reshape(-1))
        return torch.cat(pieces)

    def _align(self, arrays, reference_arrays, skeleton):
        import torch

        network = self._network(arrays, skeleton)
        reference = self._network(reference_arrays, skeleton)
        start, stop = self._window
        for site in range(stop, start, -1):
            tensor, previous = network[site], network[site - 1]
            bond = network.bond(site - 1, site)
            order = (bond, *(ix for ix in tensor.inds if ix != bond))
            data = tensor.transpose(*order).data
            shape = data.shape
            matrix = data.reshape(shape[0], -1)
            reference_matrix = reference[site].transpose(*order).data.reshape(matrix.shape)
            unitary = _polar_unitary(torch, reference_matrix @ matrix.mH)
            tensor.modify(data=(unitary @ matrix).reshape(shape), inds=order)
            tensor.transpose_like_(reference[site])
            left_order = (*(ix for ix in previous.inds if ix != bond), bond)
            left_data = previous.transpose(*left_order).data
            previous.modify(
                data=(left_data.reshape(-1, shape[0]) @ unitary.mH).reshape(left_data.shape),
                inds=left_order,
            )
            previous.transpose_like_(reference[site - 1])
        packed, _ = qtn.pack(network)
        return tuple(packed[key] for key in self._guess_keys)

    def _cycle(self, target_arrays, guess_arrays, *, skeleton, expected_shapes):
        values = self._run_fit(
            target_arrays, guess_arrays, block_size=1, sweeps=2,
            expected_shapes=expected_shapes,
        )
        return self._align(values, guess_arrays, skeleton)


class _ImplicitFIT:
    """Custom reverse-mode boundary for Torch's implicit FIT fixed point."""

    @staticmethod
    def apply(wrapper, *arrays):
        import torch
        from torch.autograd.function import once_differentiable

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, owner, *inputs):
                n_target = len(owner._target_keys)
                target = tuple(inputs[:n_target])
                guess = tuple(inputs[n_target:])
                active_site_keys = set(
                    owner._guess_site_keys[owner._window[0]:owner._window[1] + 1]
                )
                active_positions = [
                    i for i, key in enumerate(owner._guess_keys)
                    if key in active_site_keys
                ]
                if not active_positions:
                    raise ValueError("FIT window contains no packed MPS tensors")
                if owner._warmup_block_size > 1:
                    warm = owner._run_fit(
                        target, guess, block_size=owner._warmup_block_size,
                        sweeps=owner._warmup_sweeps,
                    )
                    guess = warm
                    if owner._transition_block_size > 1:
                        guess = owner._run_fit(
                            target, guess, block_size=owner._transition_block_size,
                            sweeps=1,
                        )
                start_arrays = guess
                active_set = set(active_positions)
                for i, (initial, warmed) in enumerate(
                    zip(inputs[n_target:], start_arrays, strict=True)
                ):
                    if i not in active_set and initial.shape != warmed.shape:
                        raise ValueError(
                            "FIT warm-up changed tensor shapes outside its active window"
                        )
                expected_shapes = tuple(tuple(value.shape) for value in start_arrays)
                active_shapes = tuple(tuple(guess[i].shape) for i in active_positions)
                complex_flags = tuple(guess[i].is_complex() for i in active_positions)
                y = owner._encode_active([guess[i] for i in active_positions])
                tolerance = owner._tolerance
                if tolerance is None:
                    precision = max(
                        torch.finfo(x.real.dtype if x.is_complex() else x.dtype).eps
                        for x in (guess[i] for i in active_positions)
                    )
                    tolerance = max(1e-9, 64 * precision)
                for _ in range(owner._max_iterations):
                    active = owner._decode_active(y, active_shapes, complex_flags)
                    current = list(start_arrays)
                    for pos, value in zip(active_positions, active, strict=True):
                        current[pos] = value
                    updated = owner._cycle(
                        target, tuple(current), skeleton=owner._output_skeleton,
                        expected_shapes=expected_shapes,
                    )
                    next_y = owner._encode_active([updated[i] for i in active_positions])
                    residual = (
                        torch.linalg.vector_norm(next_y - y)
                        / torch.linalg.vector_norm(y).clamp_min(torch.finfo(y.dtype).tiny)
                    )
                    y = next_y
                    owner.iterations_run += 1
                    owner.relative_residual = float(residual)
                    if float(residual) <= tolerance:
                        break
                else:
                    raise RuntimeError(
                        f"FIT fixed point did not converge in {owner._max_iterations} cycles"
                    )
                active = owner._decode_active(y, active_shapes, complex_flags)
                output = [value.clone() for value in start_arrays]
                for pos, value in zip(active_positions, active, strict=True):
                    output[pos] = value
                runtime = copy(owner)
                runtime.target_params = {}
                runtime.initial_params = {}
                runtime._output_skeleton = owner._output_skeleton
                ctx.owner = runtime
                ctx.n_target = n_target
                ctx.active_positions = tuple(active_positions)
                ctx.active_shapes = active_shapes
                ctx.complex_flags = complex_flags
                ctx.expected_shapes = expected_shapes
                ctx.save_for_backward(
                    *(x.detach() for x in target),
                    *(x.detach() for x in start_arrays),
                    y.detach(),
                )
                ctx.output_skeleton = owner._output_skeleton
                owner.converged = True
                return tuple(output)

            @staticmethod
            @once_differentiable
            def backward(ctx, *grad_outputs):
                owner = ctx.owner
                saved = ctx.saved_tensors
                target_vars = tuple(
                    x.detach().requires_grad_(True) for x in saved[:ctx.n_target]
                )
                guess_values = tuple(saved[ctx.n_target:-1])
                y = saved[-1].detach().requires_grad_(True)
                active_set = set(ctx.active_positions)
                outside_positions = tuple(
                    i for i in range(len(guess_values)) if i not in active_set
                )
                outside_vars = tuple(
                    guess_values[i].detach().requires_grad_(True)
                    for i in outside_positions
                )
                active = owner._decode_active(y, ctx.active_shapes, ctx.complex_flags)
                guess = list(guess_values)
                for pos, value in zip(ctx.active_positions, active, strict=True):
                    guess[pos] = value
                for pos, value in zip(outside_positions, outside_vars, strict=True):
                    guess[pos] = value
                with torch.enable_grad():
                    fitted = owner._cycle(
                        target_vars, tuple(guess), skeleton=ctx.output_skeleton,
                        expected_shapes=ctx.expected_shapes,
                    )
                    fitted_active = tuple(fitted[i] for i in ctx.active_positions)
                    fitted_vector = owner._encode_active(fitted_active)
                    output_cotangent = owner._encode_cotangent(
                        tuple(
                            torch.zeros_like(guess_values[i])
                            if grad_outputs[i] is None else grad_outputs[i]
                            for i in ctx.active_positions
                        )
                    )
                    rhs = output_cotangent

                    def matvec(vector):
                        pullback, = torch.autograd.grad(
                            fitted_vector, y, grad_outputs=vector,
                            retain_graph=True, allow_unused=True,
                        )
                        if pullback is None:
                            pullback = torch.zeros_like(vector)
                        return vector - pullback

                    adjoint = _gmres(
                        torch, matvec, rhs, tolerance=owner._adjoint_tolerance,
                        max_iterations=owner._adjoint_maxiter,
                    )
                    external_grads = torch.autograd.grad(
                        fitted_vector, (*target_vars, *outside_vars),
                        grad_outputs=adjoint, retain_graph=False,
                        allow_unused=True,
                    )
                    target_grads = external_grads[:len(target_vars)]
                    outside_fixed_point_grads = external_grads[len(target_vars):]
                guess_grads = []
                outside_gradient_by_position = dict(zip(
                    outside_positions, outside_fixed_point_grads, strict=True
                ))
                for i, original in enumerate(guess_values):
                    if i in active_set:
                        guess_grads.append(None)
                    else:
                        direct_gradient = (
                            torch.zeros_like(original)
                            if grad_outputs[i] is None else grad_outputs[i]
                        )
                        implicit_gradient = outside_gradient_by_position[i]
                        if implicit_gradient is not None:
                            direct_gradient = direct_gradient + implicit_gradient
                        guess_grads.append(direct_gradient)
                return (None, *target_grads, *guess_grads)

        wrapper.iterations_run = 0
        wrapper.relative_residual = None
        wrapper.converged = False
        return Function.apply(wrapper, *arrays)


def make_torch_fit(
    target,
    p,
    *,
    range_int=None,
    max_bond=None,
    warmup_block_size=1,
    warmup_sweeps=2,
    transition_block_size=1,
    max_iterations=100,
    tolerance=None,
    adjoint_tolerance=None,
    adjoint_maxiter=100,
    contraction_opt="greedy",
):
    """Create a dense Torch FIT callable with an implicit fixed-point VJP.

    ``warmup_block_size`` may be one, two, or three. Larger blocks are used
    only to open/expand ranks in the forward warm-up; the implicit derivative
    is taken through the converged one-site refinement. Native Symmray data,
    dynamic-rank derivatives, higher derivatives, and higher-order optimizer
    schedules are not supported.
    """
    import torch

    from ..backends import register_torch_linalg

    if not isinstance(target, qtn.TensorNetwork):
        raise TypeError("target must be a Quimb TensorNetwork")
    if not isinstance(p, qtn.MatrixProductState) or p.cyclic:
        raise ValueError("p must be an open-boundary MPS")
    warmup_block_size = _positive_integer("warmup_block_size", warmup_block_size)
    if warmup_block_size not in (1, 2, 3):
        raise ValueError("warmup_block_size must be 1, 2, or 3")
    warmup_sweeps = _positive_integer("warmup_sweeps", warmup_sweeps)
    transition_block_size = _positive_integer("transition_block_size", transition_block_size)
    if transition_block_size not in (1, 2):
        raise ValueError("transition_block_size must be 1 or 2")
    if transition_block_size > 1 and warmup_block_size != 3:
        raise ValueError("a two-site transition is only used after a three-site warm-up")
    max_iterations = _positive_integer("max_iterations", max_iterations)
    adjoint_maxiter = _positive_integer("adjoint_maxiter", adjoint_maxiter)
    if range_int is None:
        window = (0, int(p.L) - 1)
    else:
        if len(range_int) != 2:
            raise ValueError("range_int must contain inclusive (start, stop) sites")
        window = tuple(map(int, range_int))
        if window[0] < 0 or window[1] < window[0] or window[1] >= p.L:
            raise ValueError("range_int must be a valid inclusive MPS interval")
    if p.L > 1 and window[0] == window[1]:
        raise ValueError("range_int must span at least two sites for a local FIT window")
    max_bond = p.max_bond() if max_bond is None else _positive_integer("max_bond", max_bond)
    if tolerance is not None and (not torch.isfinite(torch.tensor(tolerance)) or tolerance <= 0):
        raise ValueError("tolerance must be positive and finite")
    target_copy, guess_copy = target.copy(), p.copy()
    devices = set()
    for network in (target_copy, guess_copy):
        for tensor in network:
            if not isinstance(tensor.data, torch.Tensor):
                raise TypeError("Torch implicit FIT requires dense Torch tensors")
            if type(tensor.data).__module__.split(".", 1)[0] == "symmray":
                raise NotImplementedError("Native Symmray Torch implicit FIT is unsupported")
            if not (tensor.data.is_floating_point() or tensor.data.is_complex()):
                raise TypeError("Torch implicit FIT requires floating-point or complex tensors")
            devices.add(tensor.data.device)
            tensor.modify(left_inds=None)
    guess_charts = {tensor.data.is_complex() for tensor in guess_copy}
    if len(guess_charts) != 1:
        raise ValueError("The MPS guess must use a uniform real or complex parameter chart")
    if not next(iter(guess_charts)) and any(
        tensor.data.is_complex() for tensor in target_copy
    ):
        raise ValueError("A complex target requires a complex MPS guess chart")
    if len(devices) != 1:
        raise ValueError("All target and guess tensors must be on the same Torch device")
    precision = max(
        torch.finfo(t.data.real.dtype if t.data.is_complex() else t.data.dtype).eps
        for t in guess_copy
    )
    adjoint_tolerance = (
        max(1e-9, 64 * precision)
        if adjoint_tolerance is None
        else float(adjoint_tolerance)
    )
    if not torch.isfinite(torch.tensor(adjoint_tolerance)) or adjoint_tolerance <= 0:
        raise ValueError("adjoint_tolerance must be positive")
    register_torch_linalg(stabilized=True)
    target_params, target_skeleton = qtn.pack(target_copy)
    initial_params, guess_skeleton = qtn.pack(guess_copy)
    target_keys = tuple(sorted(target_params))
    guess_keys = tuple(sorted(initial_params))
    guess_site_keys = tuple(
        next(iter(guess_copy._get_tids_from_tags(guess_copy.site_tag(site))))
        for site in range(int(guess_copy.L))
    )
    wrapper_type = TorchFunctionalFIT
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        # Keep Quimb's Python-driven FIT iteration outside compiled graphs,
        # while retaining the custom autograd.Function edge for callers.
        # Scope the compile boundary to this wrapper type; do not modify the
        # shared class method process-wide.
        wrapper_type = type(
            "CompiledTorchFunctionalFIT",
            (TorchFunctionalFIT,),
            {"__call__": compiler.disable(TorchFunctionalFIT.__call__)},
        )
    wrapper = wrapper_type(
        target_params, initial_params, target_skeleton, guess_skeleton,
        target_keys, guess_keys, guess_site_keys,
        window, int(max_bond), warmup_block_size, warmup_sweeps,
        transition_block_size,
        max_iterations, tolerance, adjoint_tolerance, adjoint_maxiter,
        contraction_opt,
        guess_skeleton,
    )
    return wrapper


__all__ = ["TorchFitParameters", "TorchFunctionalFIT", "make_torch_fit"]
