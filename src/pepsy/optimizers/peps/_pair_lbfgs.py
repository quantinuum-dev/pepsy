"""Reduced-pair L-BFGS in a fixed dense positive norm environment."""

import autoray as ar
import numpy as np
from scipy.optimize import minimize

from ...backends import infer_backend_converter_from_sample
from ...bp._backend import einsum, transpose
from ...bp.reduced_update import ReducedALSSolution


def pair_value_gradient(norm, target, left, right):
    """Real squared residual and Euclidean complex gradients of both factors."""
    delta = einsum('apk,kbq->apbq', left, right) - target
    weighted = einsum('abcd,cpdq->apbq', norm, delta)
    value = einsum('apbq,apbq->', delta.conj(), weighted).real
    grad_left = 2 * einsum('apbq,kbq->apk', weighted, right.conj())
    grad_right = 2 * einsum('apk,apbq->kbq', left.conj(), weighted)
    return value, grad_left, grad_right


def solve_pair_lbfgs(norm, target, left, right, *, max_iterations, tol):
    """Keep contractions native; exchange real parameters/gradients with SciPy."""
    shapes = left.shape, right.shape
    left_size = int(np.prod(left.shape))
    convert = infer_backend_converter_from_sample(left)

    def pack(a, b):
        z = np.concatenate((ar.to_numpy(a).ravel(), ar.to_numpy(b).ravel()))
        return np.concatenate((z.real, z.imag)).astype(np.float64)

    def unpack(x):
        re, im = np.split(x, 2)
        z = convert(re + 1j * im)
        return z[:left_size].reshape(shapes[0]), z[left_size:].reshape(shapes[1])

    target_norm = float(einsum('apbq,abcd,cpdq->', target.conj(), norm, target).real)
    costs = []
    best_value, best_x = float('inf'), None

    def objective(x):
        nonlocal best_value, best_x
        value, ga, gb = pair_value_gradient(norm, target, *unpack(x))
        value = float(value) / target_norm
        gradient = pack(ga, gb) / target_norm
        if np.isfinite(value) and np.all(np.isfinite(gradient)) and value < best_value:
            best_value, best_x = value, x.copy()
        return value, gradient

    def callback(intermediate_result):
        costs.append(float(intermediate_result.fun) * target_norm)
        if tol > 0 and intermediate_result.fun <= tol:
            raise StopIteration

    initial = pack(left, right)
    value, _ = objective(initial)
    costs.append(value * target_norm)
    if tol > 0 and value <= tol:
        return ReducedALSSolution(left=left, right=transpose(right, (0, 2, 1)),
                                  costs=tuple(costs), solver='lbfgs', iterations=0,
                                  converged=True, termination_reason='residual_tolerance')
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      callback=callback,
                      options={'maxiter': max_iterations, 'ftol': tol, 'gtol': tol,
                               'maxls': 40})
    left, right = unpack(initial if best_x is None else best_x)
    costs.append(best_value * target_norm)
    residual = tol > 0 and best_value <= tol
    reason = ('residual_tolerance' if residual else
              'lbfgs_tolerance' if result.success else
              'max_iterations' if result.nit >= max_iterations else 'lbfgs_failed')
    return ReducedALSSolution(left=left, right=transpose(right, (0, 2, 1)),
                              costs=tuple(costs), solver='lbfgs', iterations=result.nit,
                              converged=bool(residual or result.success), termination_reason=reason)
