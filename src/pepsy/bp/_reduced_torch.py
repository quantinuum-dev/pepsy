"""Dense Torch reduced ALS kernels with guarded native solves.

Torch stays optional until this module is used. Compile only the array
contractions: rank choices, conditioning guards and stopping stay observable.
"""

from functools import lru_cache
import math


def _weighted_target(environment, target):
    import torch

    return torch.einsum('...abAB,...apqb->...ApqB', environment, target)


def _left_equations(environment, weighted_target, right):
    import torch

    # Physical indices are multiple RHS columns, never identity-expanded.
    gram = torch.einsum('...abAB,...kqb,...KqB->...AKak', environment, right, right.conj())
    rhs = torch.einsum('...KqB,...ApqB->...AKp', right.conj(), weighted_target)
    size = rhs.shape[-3] * rhs.shape[-2]
    batch = rhs.shape[:-3]
    return gram.reshape(*batch, size, size), rhs.reshape(*batch, size, -1)


def _right_equations(environment, weighted_target, left):
    import torch

    gram = torch.einsum('...abAB,...apk,...ApK->...KBkb', environment, left, left.conj())
    rhs = torch.einsum('...ApK,...ApqB->...KBq', left.conj(), weighted_target)
    size = rhs.shape[-3] * rhs.shape[-2]
    batch = rhs.shape[:-3]
    return gram.reshape(*batch, size, size), rhs.reshape(*batch, size, -1)


def _objective(environment, target, left, right):
    import torch

    theta = torch.einsum('...apk,...kqb->...apqb', left, right)
    delta = theta - target
    weighted_delta = torch.einsum('...abAB,...apqb->...ApqB', environment, delta)
    cost = torch.sum(delta.conj() * weighted_delta, dim=(-4, -3, -2, -1)).real
    return cost, theta, weighted_delta


@lru_cache(maxsize=128)
def _captured_kernel(function, signature):
    """Separate AOT charts for each shape/device/gradient signature.

    Like prepared SU, capture forward and backward together. A per-chart
    cache avoids Dynamo's shared recompile limit on varying patch shapes.
    """
    import torch
    from torch._functorch.aot_autograd import aot_function
    from functorch.compile import make_boxed_func

    def compiler(graph, examples):
        graph = torch.fx.GraphModule(graph, graph.graph)
        return make_boxed_func(graph.forward)

    return aot_function(function, fw_compiler=compiler, bw_compiler=compiler)


def _captured_call(function, *inputs):
    import torch

    # AOT specializes its saved tensors and backward casts to the forward
    # autocast context. Reusing a full-precision chart in AMP can fail only
    # at backward, even when the forward output looks correct.
    autocast = tuple((device, torch.is_autocast_enabled(device),
                      torch.get_autocast_dtype(device))
                     for device in sorted({x.device.type for x in inputs}))
    signature = (torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                 autocast,
                 tuple((tuple(x.shape), x.stride(), x.dtype, x.device,
                        x.requires_grad, x.is_conj(), x.is_neg(), x.is_inference())
                       for x in inputs))
    return _captured_kernel(function, signature)(*inputs)


def _kernels(torch_backend):
    from functools import partial

    functions = (_weighted_target, _left_equations, _right_equations, _objective)
    if torch_backend is None:
        return functions
    if torch_backend != 'aot_eager':
        raise ValueError('torch_backend must be None or aot_eager')
    return tuple(partial(_captured_call, fn) for fn in functions)


def _guarded_solve(gram, rhs, rcond, counts):
    """Use Cholesky only when the reference spectral floor is inactive.

    For A=L L.H, lambda_min(A) >= 1/||L^-1||_F**2 and
    lambda_max(A) <= trace(A). This conservative certificate avoids an
    eigensolve while retaining the existing relative clipping policy.
    """
    import torch

    gram = (gram + gram.mH) * .5
    factor, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    floor = max(rcond, 1e-15)
    eps = torch.finfo(gram.real.dtype).eps
    if int(info.detach()) == 0:
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        inverse = torch.linalg.solve_triangular(factor, identity, upper=False)
        inverse_norm2 = torch.sum(inverse.abs().square())
        trace = torch.trace(gram).real
        solution = torch.cholesky_solve(rhs, factor)
        residual = torch.linalg.vector_norm(gram @ solution - rhs)
        denominator = (torch.linalg.vector_norm(gram) * torch.linalg.vector_norm(solution)
                       + torch.linalg.vector_norm(rhs))
        certificate = trace * inverse_norm2 * floor
        safe = (torch.isfinite(certificate) & (trace > 0) & (certificate < .5)
                & torch.isfinite(solution).all()
                & (residual <= 128 * eps * gram.shape[0] * denominator))
        if bool(safe.detach()):
            counts['cholesky'] += 1
            return solution
    # Singular, indefinite and uncertified cases retain the old positive
    # local eigensolve, including its relative eigenvalue floor.
    values, vectors = torch.linalg.eigh(gram)
    maximum = values.max()
    if not bool((torch.isfinite(values).all() & (maximum > 0)).detach()):
        raise ValueError('local ALS matrix has no finite positive support')
    values = values.clamp_min(maximum * floor)
    counts['eigh'] += 1
    return vectors @ ((vectors.mH @ rhs) / values[:, None])


def solve(problem, *, max_bond, max_iterations, tol, rcond,
          monitor_convergence, initial_factors=None, torch_backend=None,
          measure_fidelity=False):
    """Solve one retained open-environment problem without Quimb TN setup."""
    import torch
    from .reduced_update import (
        ReducedALSSolution, _gauge_reduced_factors, _reduced_als_stop,
        _open_environment_data, _svd_initial_factors,
    )

    if not isinstance(problem.target, torch.Tensor):
        raise TypeError("solver='torch' requires a dense Torch target")
    environment = _open_environment_data(problem.environment)
    if not isinstance(environment, torch.Tensor):
        raise TypeError("solver='torch' requires a dense Torch environment")
    # QR creates transposed/conjugate views. Canonical layouts keep one
    # compiled graph per actual shape/device/gradient chart, rather than
    # recompiling for the initial factors and every sweep's view metadata.
    def canonical(value):
        return value.resolve_conj().contiguous() if torch_backend is not None else value

    environment = canonical(environment)
    target = canonical(problem.target)
    weighted, left_equations, right_equations, objective = _kernels(torch_backend)
    left, right = (_svd_initial_factors(problem.target, max_bond)
                   if initial_factors is None else tuple(x.clone() for x in initial_factors))
    left, right = canonical(left), canonical(right)
    weighted_target = canonical(weighted(environment, target))
    target_norm = float(torch.sum(target.conj() * weighted_target).real.detach())
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError('Torch ALS requires a positive finite target norm')
    initial, _, _ = objective(environment, target, left, right)
    costs = [float(initial.detach())]
    counts = dict(cholesky=0, eigh=0)
    converged, reason = False, 'max_iterations'
    for iteration in range(1, max_iterations + 1):
        gram, rhs = left_equations(environment, weighted_target, right)
        solved = _guarded_solve(gram, rhs, rcond, counts)
        left = canonical(solved.reshape(left.shape[0], left.shape[2], left.shape[1]).permute(0, 2, 1))
        gram, rhs = right_equations(environment, weighted_target, left)
        solved = _guarded_solve(gram, rhs, rcond, counts)
        right = solved.reshape(right.shape[0], right.shape[2], right.shape[1]).permute(0, 2, 1)
        left, right = _gauge_reduced_factors(left, right)
        left, right = canonical(left), canonical(right)
        current, theta, weighted_delta = objective(environment, target, left, right)
        costs.append(float(current.detach()))
        reason = _reduced_als_stop(costs[-2], costs[-1], target_norm, tol)
        if reason is not None:
            converged = reason != 'nonfinite_cost'
            break
    fidelity = None
    fidelity_status = 'not_requested'
    if measure_fidelity:
        from .reduced_update import _validated_local_fidelity

        norm = torch.sum(theta.conj() * (weighted_target + weighted_delta)).real
        overlap = torch.sum(theta.conj() * weighted_target)
        fidelity, fidelity_status = _validated_local_fidelity(
            float(norm.detach()), target_norm, float(overlap.abs().detach()),
            torch.finfo(theta.real.dtype).eps,
        )
    return ReducedALSSolution(
        left=left, right=right, costs=tuple(costs), solver='torch',
        iterations=iteration if monitor_convergence else None,
        converged=converged if monitor_convergence else None,
        termination_reason=(reason or 'max_iterations') if monitor_convergence else None,
        local_fidelity=fidelity,
        local_infidelity=None if fidelity is None else 1-fidelity,
        fidelity_status=fidelity_status,
        solver_report=dict(**counts, torch_backend=torch_backend,
                           environment='open reduced tensor; physical RHS batching'),
        target_norm=target_norm,
    )
