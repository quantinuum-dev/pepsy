"""Native matrix-free Hermitian norm actions and preconditioned CG for ALS."""

import itertools
import math

import autoray as ar
import quimb.tensor as qtn

from ...bp._backend import all_finite


class CGFailure(ValueError):
    """An unsuccessful iterative solve, with diagnostics for fallback/rollback."""

    def __init__(self, reason, **report):
        super().__init__(reason)
        self.report = dict(solver='cg', cg_failure=reason, **report)


def split_environment(network, *, optimize):
    """Split the environment before applying trial vectors, never forming N.

    A bulk hole is a four-tensor ring. Contract adjacent pairs independently;
    every subsequent action contracts a trial tensor with one pair FIRST.
    This prevents path planning/constant folding from constructing the full
    D^4-by-D^4 metric. Small edge/corner rings use the same partition rule.
    """
    tensors = network.tensors
    if not tensors:
        return [], network.exponent
    if len(tensors) == 1:
        groups = [tensors]
    else:
        # Only at most four tensors occur in a reduced one-site strip hole.
        # Minimize the largest half's exposed size; ties favor balanced halves.
        choices = []
        for count in range(1, len(tensors)):
            for rest in itertools.combinations(range(1, len(tensors)), count - 1):
                subset = {0, *rest}
                groups = [[t for i, t in enumerate(tensors) if (i in subset) == inside]
                          for inside in (True, False)]
                sizes = []
                for group in groups:
                    tn = qtn.TensorNetwork(group)
                    sizes.append(math.prod(tn.ind_size(i) for i in tn.outer_inds()))
                choices.append((max(sizes), sum(sizes), groups))
        groups = min(choices, key=lambda choice: choice[:2])[2]
    parts = []
    # Strip exponents can be mutable backend scalars. Never update a shared
    # exponent in place while preparing a temporary operator.
    exponent = float(network.exponent)
    for group in groups:
        tn = qtn.TensorNetwork(group)
        part, power = tn.contract(
            all, output_inds=tn.outer_inds(), optimize=optimize,
            strip_exponent=True, preserve_tensor=True,
        )
        parts.append(part)
        exponent += float(power)
    return parts, exponent


def action_path(parts):
    """Absorb the active tensor before joining the environment halves."""
    return ((0, 2), (0, 1)) if len(parts) == 2 else ((0, 1),)


def prepare_operator(network, left, right, sample, *, optimize):
    """Build a Hermitian action and its Jacobi diagonal on the native backend."""
    parts, exponent = split_environment(network, optimize=optimize)
    if not parts:
        diagonal = ar.do('ones_like', sample[:, 0]).real
        return lambda x: x, diagonal, exponent
    reduced = qtn.TensorNetwork(parts)
    diagonal = reduced.reindex(dict(zip(right, left))).contract(
        all, output_inds=left, optimize='greedy', preserve_tensor=True,
    ).data.reshape(-1).real
    if not all_finite(diagonal):
        raise CGFailure('nonfinite_diagonal')

    backend = ar.infer_backend(sample)
    batch = qtn.rand_uuid()
    contractors = {}
    path = action_path(parts)

    def action(x, adjoint=False):
        inputs, outputs = (left, right) if adjoint else (right, left)
        dims = tuple(reduced.ind_size(i) for i in inputs)
        data = (ar.do('conj', x) if adjoint else x).reshape(*dims, x.shape[-1])
        if adjoint not in contractors:
            trial = qtn.Tensor(data, inds=(*inputs, batch))
            contractors[adjoint] = qtn.tensor_contract(
                *parts, trial, output_inds=(*outputs, batch), optimize=path,
                get='expression', constants=tuple(range(len(parts))),
            )
        result = contractors[adjoint](data, backend=backend).reshape(x.shape)
        return ar.do('conj', result) if adjoint else result

    def hermitian_action(x):
        return .5 * (action(x) + action(x, adjoint=True))

    return hermitian_action, diagonal, exponent


def solve_cg(action, diagonal, rhs, initial, *, rtol, maxiter, shift):
    """Jacobi-preconditioned CG for (N_H + shift * scale * I) x = b.

    Physical RHS columns form independent copies of N, solved together with
    a Frobenius inner product. Only scalar convergence decisions leave device.
    A nonpositive curvature or unmet true residual triggers explicit failure.
    """
    scale = float(ar.do('max', ar.do('abs', diagonal)))
    if not math.isfinite(scale) or scale <= 0:
        raise CGFailure('nonpositive_diagonal_scale')
    ridge = shift * scale
    if float(ar.do('min', diagonal)) + ridge <= 0:
        raise CGFailure('nonpositive_diagonal')
    preconditioner = (diagonal + ridge)[:, None]

    def apply(x):
        return action(x) + ridge * x

    def inner(x, y):
        return float(ar.do('sum', ar.do('conj', x) * y).real)

    rhs_norm = float(ar.do('linalg.norm', rhs))
    if not math.isfinite(rhs_norm) or rhs_norm == 0:
        raise CGFailure('invalid_rhs_norm')
    x = initial + ar.do('zeros_like', initial)
    residual = rhs - apply(x)
    relative = float(ar.do('linalg.norm', residual)) / rhs_norm
    z = residual / preconditioner
    direction = z
    rho = inner(residual, z)
    for iteration in range(maxiter + 1):
        if not math.isfinite(relative):
            raise CGFailure('nonfinite_residual', cg_iterations=iteration)
        if relative <= rtol:
            # Recurrence residuals alone can misreport convergence in finite precision.
            residual = rhs - apply(x)
            relative = float(ar.do('linalg.norm', residual)) / rhs_norm
            if relative <= rtol and all_finite(x):
                return x, dict(solver='cg', matrix_size=int(rhs.shape[0]),
                               cg_iterations=iteration, relative_residual=relative,
                               cg_shift=shift, converged=True)
            z = residual / preconditioner
            direction = z
            rho = inner(residual, z)
        if iteration == maxiter:
            break
        product = apply(direction)
        curvature = inner(direction, product)
        if not math.isfinite(curvature) or curvature <= 0 or rho <= 0:
            raise CGFailure('nonpositive_curvature', cg_iterations=iteration)
        step = rho / curvature
        x = x + step * direction
        residual = residual - step * product
        relative = float(ar.do('linalg.norm', residual)) / rhs_norm
        z = residual / preconditioner
        next_rho = inner(residual, z)
        direction = z + (next_rho / rho) * direction
        rho = next_rho
    relative = float(ar.do('linalg.norm', rhs - apply(x))) / rhs_norm
    raise CGFailure('iteration_limit', cg_iterations=maxiter, relative_residual=relative)
