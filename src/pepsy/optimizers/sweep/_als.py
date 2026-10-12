"""Fixed-rank one-site ALS inside an existing row/column sweep environment."""

import math
import warnings
from contextlib import nullcontext
from numbers import Integral, Real

import autoray as ar
import quimb.tensor as qtn

from ...bp._backend import all_finite, dag


def options(values=None):
    """Validate local ALS controls independently of outer sweep budgets."""
    result = dict(n_round_trips=2, rtol=1e-9, rcond=None, max_matrix_size=1024,
                  linear_solver='dense-cg', dense_shift=0., cg_maxiter=200,
                  cg_rtol=None, cg_shift=0., cg_fallback=False,
                  lbfgs_maxiter=100, lbfgs_rtol=None, lbfgs_history=10, lbfgs_maxls=20)
    unknown = set(values or {}) - result.keys()
    if unknown:
        raise ValueError(f"Unknown ALS optimizer_options: {sorted(unknown)}")
    result.update(values or {})
    if result['linear_solver'] not in ('dense-cg', 'dense', 'cg', 'pinv', 'lbfgs', 'dense-lbfgs'):
        raise ValueError("ALS linear_solver must be 'dense-cg', 'dense', 'cg', 'pinv', 'lbfgs', or 'dense-lbfgs'")
    if not isinstance(result['cg_fallback'], bool):
        raise ValueError('ALS cg_fallback must be boolean')
    for key in ('n_round_trips', 'max_matrix_size', 'cg_maxiter',
                'lbfgs_maxiter', 'lbfgs_history', 'lbfgs_maxls'):
        value = result[key]
        if key == 'max_matrix_size' and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'ALS {key} must be a positive integer')
    for key in ('rtol', 'rcond', 'cg_rtol', 'cg_shift', 'dense_shift', 'lbfgs_rtol'):
        value = result[key]
        if value is None:
            if key == 'dense_shift':
                raise ValueError('ALS dense_shift must be finite and nonnegative')
            continue
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f'ALS {key} must be finite and nonnegative')
        if key == 'rcond' and value >= 1:
            raise ValueError('ALS rcond must be less than one')
        if key in ('cg_rtol', 'lbfgs_rtol') and value == 0:
            raise ValueError(f'ALS {key} must be positive')
    return result


def _log_norm(value, tolerance=1e-10):
    mantissa, exponent = value
    mantissa = complex(mantissa)
    if (not math.isfinite(mantissa.real) or not math.isfinite(mantissa.imag)
            or not math.isfinite(float(exponent)) or mantissa.real <= 0
            or abs(mantissa.imag) > tolerance * mantissa.real):
        raise ValueError('ALS requires finite positive, nearly real norms')
    return math.log10(mantissa.real) + float(exponent)


def _hermitian_norm(value):
    """Score with the same Hermitian metric used by every local solver.

    Truncated boundaries can give a complex x.H N x. For N_H=(N+N.H)/2,
    x.H N_H x is exactly Re(x.H N x); an imaginary compression residual
    must not reject the strip before that Hermitian solve is attempted.
    """
    mantissa, exponent = value
    mantissa = complex(mantissa)
    if not math.isfinite(mantissa.real) or not math.isfinite(mantissa.imag):
        raise ValueError('ALS requires a finite norm before Hermitianization')
    return mantissa.real, exponent


def _hole(network, excluded):
    result = qtn.TensorNetwork([t for t in network if all(t is not x for x in excluded)])
    result.exponent = network.exponent
    return result


def _solve_dense(matrix, rhs, *, shift=0.):
    """Hermitianize the explicit metric, then solve directly without eigenvectors."""
    matrix = .5 * (matrix + dag(matrix))
    if shift:
        diagonal = ar.do('diag', matrix)
        scale = float(ar.do('max', ar.do('abs', diagonal)))
        matrix = matrix + ar.do('diag', ar.do('ones_like', diagonal) * (shift * scale))
    try:
        answer = ar.do('linalg.solve', matrix, rhs)
    except Exception as exc:
        if type(exc).__name__ != 'LinAlgError' and not (
                isinstance(exc, RuntimeError) and 'singular' in str(exc).lower()):
            raise
        raise ValueError('singular ALS local equation') from exc
    if not all_finite(answer):
        raise ValueError('nonfinite ALS direct solution')
    rhs_norm = float(ar.do('linalg.norm', rhs))
    residual = float(ar.do('linalg.norm', matrix @ answer - rhs)) / max(rhs_norm, 1e-30)
    single = ar.get_dtype_name(rhs) in ('float32', 'complex64')
    if not math.isfinite(residual) or residual > (1e-4 if single else 1e-8):
        raise ValueError('ALS direct solution failed its residual check')
    return answer, dict(solver='dense', matrix_size=int(matrix.shape[0]),
                        relative_residual=residual, dense_shift=shift)


def _cg_controls(controls, sample):
    single = ar.get_dtype_name(sample) in ('float32', 'complex64')
    return dict(
        rtol=controls['cg_rtol'] if controls['cg_rtol'] is not None else (1e-5 if single else 1e-8),
        maxiter=controls['cg_maxiter'],
        shift=controls['cg_shift'] if controls['cg_shift'] is not None else (1e-5 if single else 1e-10),
    )


def _solve_site(norm, overlap, tag, *, optimize, rcond, controls=None):
    # Spectral solves remain explicit; iterative/local L-BFGS paths share the
    # same one-site norm and overlap environments.
    from ..peps._strip_solve import solve_positive

    ket = norm[tag, '__ALS_KET__']
    bra = norm[tag, '__ALS_BRA__']
    cross_bra = overlap[tag, '__ALS_BRA__']
    physical = tuple(i for i in ket.inds if i in bra.inds)
    right = tuple(i for i in ket.inds if i not in physical)
    left = tuple(i for i in bra.inds if i not in physical)
    size = math.prod(ket.ind_size(i) for i in right)
    controls = options(controls)
    matrix_free = controls['linear_solver'] in ('cg', 'lbfgs')
    lbfgs = controls['linear_solver'] in ('lbfgs', 'dense-lbfgs')
    if lbfgs:
        from ._als_lbfgs import solve_lbfgs
        single = ar.get_dtype_name(ket.data) in ('float32', 'complex64')
        lbfgs_controls = dict(maxiter=controls['lbfgs_maxiter'], history=controls['lbfgs_history'],
                              maxls=controls['lbfgs_maxls'],
                              rtol=controls['lbfgs_rtol'] if controls['lbfgs_rtol'] is not None
                              else (1e-4 if single else 1e-6))
    hole = _hole(norm, (ket, bra))
    if matrix_free:
        from ._als_cg import action_path, split_environment
        target = next(t for t in overlap.select_tensors(tag) if t is not cross_bra)
        parts, exponent = split_environment(_hole(overlap, (cross_bra, target)), optimize=optimize)
        projected = qtn.TensorNetwork([*parts, target])
        projected.exponent = exponent
        rhs_optimize = action_path(parts) if parts else optimize
    else:
        projected = _hole(overlap, (cross_bra,))
        rhs_optimize = optimize
    rhs, rhs_exponent = projected.contract(
        all, output_inds=cross_bra.inds, optimize=rhs_optimize,
        strip_exponent=True, preserve_tensor=True,
    )
    # Cross bra virtual labels can differ from norm bra labels. Their tensor
    # axes follow the same candidate ket, so align by position, not by name.
    cross_left = tuple(cross_bra.inds[ket.inds.index(i)] for i in right)
    cross_physical = tuple(cross_bra.inds[ket.inds.index(i)] for i in physical)
    data = rhs.transpose(*cross_left, *cross_physical).data.reshape(size, -1)
    initial = ket.transpose(*right, *physical).data.reshape(size, -1)
    failure = None
    if matrix_free:
        from ._als_cg import CGFailure, prepare_operator, solve_cg
        try:
            action, diagonal, exponent = prepare_operator(hole, left, right, data, optimize=optimize)
            scaled = data * 10.**float(rhs_exponent - exponent)
            if lbfgs:
                answer, report = solve_lbfgs(action, scaled, initial, **lbfgs_controls)
            else:
                answer, report = solve_cg(
                    action, diagonal, scaled, initial,
                    **_cg_controls(controls, ket.data),
                )
        except CGFailure as exc:
            limit = controls['max_matrix_size']
            if lbfgs or not controls['cg_fallback'] or (limit is not None and size > limit):
                raise
            failure = exc.report
    if not matrix_free or failure is not None:
        matrix, exponent = hole.contract(
            all, output_inds=(*left, *right), optimize=optimize,
            strip_exponent=True, preserve_tensor=True,
        )
        data = data * 10.**float(rhs_exponent - exponent)
        if not all_finite(matrix.data) or not all_finite(data):
            raise ValueError('nonfinite ALS local equation')
        matrix = matrix.data.reshape(size, size)
        if lbfgs:
            matrix = .5 * (matrix + dag(matrix))
            answer, report = solve_lbfgs(lambda x: matrix @ x, data, initial, **lbfgs_controls)
            report['solver'] = 'dense-lbfgs'
        elif controls['linear_solver'] == 'pinv':
            answer, report = solve_positive(matrix, data, rcond=rcond)
        elif controls['linear_solver'] == 'dense-cg':
            from ._als_cg import CGFailure, solve_cg
            matrix = .5 * (matrix + dag(matrix))
            try:
                answer, report = solve_cg(
                    lambda x: matrix @ x, ar.do('diag', matrix).real, data, initial,
                    **_cg_controls(controls, ket.data),
                )
                report['solver'] = 'dense-cg'
            except CGFailure as exc:
                if not controls['cg_fallback']:
                    exc.report['solver'] = 'dense-cg'
                    raise
                failure = exc.report
                answer, report = _solve_dense(matrix, data, shift=controls['dense_shift'])
        else:
            answer, report = _solve_dense(matrix, data, shift=controls['dense_shift'])
        if failure is not None:
            # The final residual describes the returned direct solution, not
            # the failed CG iterate. Retain that earlier residual separately.
            report.update({key: value for key, value in failure.items()
                           if key not in ('solver', 'relative_residual')})
            if 'relative_residual' in failure:
                report['cg_relative_residual'] = failure['relative_residual']
            report['fallback_from'] = 'cg'
    shape = tuple(ket.ind_size(i) for i in (*right, *physical))
    tensor = qtn.Tensor(answer.reshape(shape), inds=(*right, *physical))
    return tensor.transpose(*ket.inds).data, report


def validate_input(owner, controls):
    """Reject unsupported arrays and oversized dense solves before a sweep."""
    if owner._symmray_state:
        raise TypeError('Sweep ALS currently requires dense arrays; use a gradient solver for Symmray')
    backends = {ar.infer_backend(t.data) for t in owner.state}
    if not backends <= {'numpy', 'torch', 'cupy'} or len(backends) != 1:
        raise TypeError('Sweep ALS supports dense NumPy, Torch, or CuPy arrays on one backend')
    outer = set(owner.state.outer_inds())
    sizes = [t.size // math.prod(t.ind_size(i) for i in t.inds if i in outer)
             for t in owner.state]
    limit = controls['max_matrix_size']
    if controls['linear_solver'] not in ('cg', 'lbfgs') and limit is not None and max(sizes) > limit:
        raise ValueError(f'ALS local matrix size {max(sizes)} exceeds max_matrix_size={limit}')
    if controls['linear_solver'] in ('lbfgs', 'dense-lbfgs'):
        if backends != {'numpy'} and not getattr(owner, '_warned_als_lbfgs_host', False):
            warnings.warn('One-site L-BFGS uses SciPy host parameter/gradient vectors; '
                          'environment contractions retain the input backend/device.', stacklevel=3)
            owner._warned_als_lbfgs_host = True


def optimize_slice(owner, index, *, axis, solver_options=None):
    controls = options(solver_options)
    validate_input(owner, controls)
    sample = next(iter(owner.state)).data
    backend = ar.infer_backend(sample)
    context = nullcontext()
    if backend == 'torch':
        import torch
        context = torch.no_grad()
    elif backend == 'cupy':
        context = sample.device
    with context:
        return _optimize_slice(owner, index, axis=axis, controls=controls)


def _optimize_slice(owner, index, *, axis, controls):
    """Fit one strip without changing its exterior or rebuilding boundaries.

    A round trip visits every site forward and backward. Directional cursors
    retain the fixed future and updated past, and are rebuilt on reversal.
    The default constructs and Hermitianizes each site's metric, then solves
    it iteratively. Matrix-free CG, direct factorization, and spectral
    pseudoinverse are alternatives. Acceptance uses the original overlap.
    """
    from ..peps._strip_cursor import StripSweepCursor

    tags = owner._site_tensor_tags(axis, index)
    local = owner.state.select(f'{owner._axis_tag(axis)}{index}').copy()
    # Solver results are new arrays. Detach Torch inputs to avoid retaining a
    # graph through the eigensolves when callers supplied trainable tensors.
    if ar.infer_backend(local[tags[0]].data) == 'torch':
        local.apply_to_arrays(lambda x: x.detach())
    bra = owner._bra_with_reindexed_inner(local)
    cross_bra = owner._overlap_bra(local)
    local.add_tag('__ALS_KET__')
    bra.add_tag('__ALS_BRA__')
    cross_bra.add_tag('__ALS_BRA__')
    target = owner.state_target.select(f'{owner._axis_tag(axis)}{index}')
    right, left = owner._boundary_keys_for_index(index, axis)
    norm = owner._attach_boundaries(local | bra, owner.bdy.mps_b,
                                    right_key=right, left_key=left)
    overlap = owner._attach_boundaries(target | cross_bra, owner.bdy_overlap.mps_b,
                                       right_key=right, left_key=left)
    norm.exponent += owner._local_norm_exponent()
    overlap.exponent += owner._local_overlap_exponent()
    optimize = owner._get_local_contraction_opt()
    along = 'X' if axis == 'y' else 'Y'
    # These kernels have at most six tensors. An optimal small-network path
    # prevents a greedy contraction from forming a dense site metric (D^12
    # work at chi=D^2), even when the user's general optimizer is greedy.
    iterative = controls['linear_solver'] in ('cg', 'dense-cg', 'lbfgs', 'dense-lbfgs')
    scalar_optimize = 'optimal' if iterative else optimize

    def scalar(network):
        if iterative and (network is norm or network is overlap):
            cursor = StripSweepCursor(network, axis=along, length=len(tags), optimize=scalar_optimize)
            cursor.start()
            network = cursor.local(0)
        if single:
            # Objective contractions can cancel strongly after an unprojected
            # solve. Accumulate diagnostics in double precision; owned tensor
            # arrays and solver results retain the requested dtype/device.
            network = network.copy()
            network.apply_to_arrays(lambda x: ar.do(
                'astype', x, 'complex128' if dtype == 'complex64' else 'float64'))
        return network.contract(all, optimize=scalar_optimize, strip_exponent=True)
    report = dict(axis=axis, index=index, solver='als', update_applied=False,
                  history=[], best_history=[], local_solves=[], n_round_trips=0,
                  max_round_trips=controls['n_round_trips'], invalid_loss=False)
    dtype = ar.get_dtype_name(local[tags[0]].data)
    single = dtype in ('float32', 'complex64')
    roundoff = 2e-5 if single else 1e-10
    try:
        log_norm = _log_norm(_hermitian_norm(scalar(norm)), roundoff)
        log_target = _log_norm(owner.target_norm, roundoff)
    except ValueError:
        return {**report, 'loss_initial': float('nan'), 'loss_final': None,
                'loss_best': None, 'invalid_loss': True, 'rejection_reason': 'invalid_initial_loss'}
    norm.exponent -= log_norm
    overlap.exponent -= .5 * (log_norm + log_target)
    rcond = controls['rcond']
    if rcond is None:
        rcond = 1e-6 if single else 1e-12

    def score(aa=norm, ab=overlap):
        n = _hermitian_norm(scalar(aa))
        _log_norm(n, roundoff)
        value = float(1. - owner._scaled_overlap_fidelity(scalar(ab), n, (1., 0.)))
        if not math.isfinite(value) or not -roundoff <= value <= 1.:
            raise ValueError('invalid ALS normalized overlap')
        return value

    try:
        initial = score()
    except ValueError:
        return {**report, 'loss_initial': float('nan'), 'loss_final': None,
                'loss_best': None, 'invalid_loss': True, 'rejection_reason': 'invalid_initial_loss'}
    best = initial
    report.update(loss_initial=max(0., initial), raw_loss_initial=initial,
                  history=[initial], best_history=[max(0., initial)],
                  termination_reason='max_round_trips')
    cursors = [StripSweepCursor(tn, axis=along, length=len(tags), optimize=scalar_optimize)
               for tn in (norm, overlap)]

    def write(tag, data):
        local[tag].modify(data=data)
        norm[tag, '__ALS_KET__'].modify(data=data)
        norm[tag, '__ALS_BRA__'].modify(data=ar.do('conj', data))
        overlap[tag, '__ALS_BRA__'].modify(data=ar.do('conj', data))

    for trip in range(controls['n_round_trips']):
        previous = best
        for reverse in (False, True):
            for cursor in cursors:
                cursor.start(reverse=reverse)
            order = range(len(tags) - 1, -1, -1) if reverse else range(len(tags))
            for position in order:
                tag = tags[position]
                old = local[tag].data
                aa, ab = [cursor.local(position) for cursor in cursors]
                detail = dict(site=tag, round_trip=trip, direction='backward' if reverse else 'forward',
                              accepted=False)
                try:
                    answer, info = _solve_site(aa, ab, tag, optimize=optimize, rcond=rcond,
                                               controls=controls)
                    detail.update(info)
                    write(tag, answer)
                    # The cursor's reduced network owns the same active site
                    # values at entry but can have copied Tensor wrappers.
                    aa[tag, '__ALS_KET__'].modify(data=answer)
                    aa[tag, '__ALS_BRA__'].modify(data=ar.do('conj', answer))
                    ab[tag, '__ALS_BRA__'].modify(data=ar.do('conj', answer))
                    current = score(aa, ab)
                    if current <= best:
                        best = current
                        detail['accepted'] = True
                    else:
                        write(tag, old)
                        detail['rejection_reason'] = 'worse_overlap'
                except (ValueError, OverflowError) as exc:
                    write(tag, old)
                    detail.update(getattr(exc, 'report', {}))
                    detail['rejection_reason'] = 'invalid_candidate'
                report['local_solves'].append(detail)
                report['history'].append(best)
                report['best_history'].append(max(0., best))
        report['n_round_trips'] += 1
        tolerance = controls['rtol']
        if tolerance and max(0., previous - best) <= tolerance * max(abs(previous), 1e-15):
            report['termination_reason'] = 'relative_tolerance'
            break
    # Write only active-site data: no tags, index order, exterior arrays, or
    # PEPS exponent changes. The normalized overlap is scale invariant.
    for tag in tags:
        owner.state[tag].modify(data=local[tag].data)
    owner._maybe_store_best_state(best)
    report.update(loss_final=max(0., best), raw_loss_final=best, loss_best=max(0., best),
                  update_applied=any(s['accepted'] for s in report['local_solves']),
                  environment_reuse=[cursor.report() for cursor in cursors])
    if not report['update_applied']:
        invalid = all(s.get('rejection_reason') == 'invalid_candidate' for s in report['local_solves'])
        report.update(invalid_loss=invalid,
                      rejection_reason='invalid_candidate' if invalid else 'no_improving_site')
    return report
