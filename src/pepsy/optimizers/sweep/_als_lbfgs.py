"""One-site quadratic L-BFGS with exact gradients and native norm actions."""

import math

import autoray as ar
import numpy as np

from ...backends.convert import infer_backend_converter_from_sample
from ...bp._backend import all_finite


def solve_lbfgs(action, rhs, initial, *, maxiter, rtol, history, maxls):
    """Minimize Re<x,N_H x> - 2 Re<x,b> without differentiating contractions.

    SciPy owns real host parameter/gradient vectors. Only these vectors move;
    the operator and site-shaped contractions keep the input backend/device.
    Real and imaginary coordinates have gradients 2 Re(N_H x-b) and
    2 Im(N_H x-b), respectively. A finite budget returns the best evaluated
    quadratic candidate; the caller separately checks physical overlap.
    """
    from scipy.optimize import minimize

    convert = infer_backend_converter_from_sample(initial)
    shape = initial.shape
    size = math.prod(shape)
    complex_input = 'complex' in ar.get_dtype_name(initial)
    rhs_norm = float(ar.do('linalg.norm', rhs))
    if not math.isfinite(rhs_norm) or rhs_norm == 0:
        raise ValueError('L-BFGS requires a finite nonzero local RHS')

    def pack(value):
        flat = np.asarray(ar.to_numpy(value)).reshape(-1)
        if complex_input:
            return np.concatenate((flat.real, flat.imag)).astype('float64')
        return flat.astype('float64')

    def unpack(vector):
        value = vector[:size] + 1j * vector[size:] if complex_input else vector
        return convert(value.reshape(shape))

    # Scale the objective uniformly so very small environment mantissas do not
    # look converged to a host optimizer. This does not change its minimizer.
    scale = rhs_norm
    best = {}
    evaluations = 0

    def evaluate(vector):
        nonlocal evaluations
        x = unpack(vector)
        product = action(x)
        residual = product - rhs
        value = float(ar.do('sum', ar.do('conj', x) * (product - 2 * rhs)).real) / scale
        gradient = pack(2 * residual / scale)
        relative = float(ar.do('linalg.norm', residual)) / rhs_norm
        evaluations += 1
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)) or not math.isfinite(relative):
            raise ValueError('nonfinite local L-BFGS objective or gradient')
        if not best or value < best['value']:
            best.update(value=value, vector=vector.copy(), relative=relative)
        if relative <= rtol:
            # Stop on the actual equation residual, including at the warm start.
            best.update(value=value, vector=vector.copy(), relative=relative)
            raise _ResidualConverged
        return value, gradient

    def callback(vector):
        # SciPy's callback is only for accounting; gradients are analytic.
        nonlocal iterations
        iterations += 1

    iterations = 0
    try:
        result = minimize(evaluate, pack(initial), jac=True, method='L-BFGS-B', callback=callback,
                          options=dict(maxiter=maxiter, maxcor=history, maxls=maxls,
                                       maxfun=(maxiter + 1) * maxls, ftol=0., gtol=0.))
        reason = str(result.message)
        iterations = int(result.nit)
    except _ResidualConverged:
        reason = 'relative_residual'
    answer = unpack(best['vector'])
    if not all_finite(answer):
        raise ValueError('nonfinite local L-BFGS solution')
    return answer, dict(solver='lbfgs', matrix_size=int(rhs.shape[0]),
                        lbfgs_iterations=iterations, lbfgs_evaluations=evaluations,
                        relative_residual=best['relative'], converged=best['relative'] <= rtol,
                        termination_reason=reason, optimizer_backend='scipy',
                        gradient='analytic')


class _ResidualConverged(Exception):
    """Stop a local optimization as soon as its true residual is small."""
