"""Optional fixed-rank, one-site ALS refinement of a completed PEPS strip."""

import math

import autoray as ar
import quimb.tensor as qtn

from ..._internal.cutoff import resolve_fit_rtol
from ...boundary._reuse import layer_sources
from ...boundary.metrics import build_bra_ket, _as_scaled_scalar
from ...bp._backend import all_finite, dag
from ._boundary_convergence import chi_pair
from ._full_update import boundary_strip
from ._strip_cursor import StripSweepCursor
from ._strip_solve import solve_positive


def local_matrix_size(state, site):
    tensor = state[site]
    return tensor.size // tensor.ind_size(state.site_ind(*site))


def strip_key(where):
    """Return the transverse axis and coordinate of a nearest-neighbor bond."""
    if not isinstance(where, (tuple, list)) or len(where) != 2:
        return None
    if not all(isinstance(site, (tuple, list)) and len(site) == 2 for site in where):
        return None
    a, b = where
    if sum(abs(i - j) for i, j in zip(a, b)) != 1:
        return None
    return ('y', a[1]) if a[1] == b[1] else ('x', a[0])


def _scalar(network, optimize):
    return _as_scaled_scalar(network.contract(all, optimize=optimize, strip_exponent=True))


def _norm_log(value, tolerance):
    mantissa, exponent = value
    value = complex(mantissa)
    if (not math.isfinite(value.real) or not math.isfinite(value.imag)
            or not math.isfinite(exponent) or value.real <= 0
            or abs(value.imag) > tolerance * value.real):
        raise ValueError('strip refinement requires positive finite, nearly real norms')
    return math.log10(value.real) + exponent


def _value(network, optimize):
    mantissa, exponent = _scalar(network, optimize)
    try:
        return complex(mantissa) * 10.**exponent
    except OverflowError:
        return complex(float('inf'))


def refine_strip(state, target, *, key, chi, contraction_opt, boundary_kwargs,
                 sweeps, rtol, rcond=None, boundary=None, overlap_boundary=None,
                 reuse=True, solver='quimb', balance=False,
                 max_matrix_size=1024, target_norm=None):
    """Fit only strip tensors against a fixed exact block target.

    Both objective networks are scaled to unit initial norms before solving;
    their relative exponents are restored in each cached local equation.
    Exterior arrays, tensor metadata, and all bond dimensions stay fixed.
    ``chi`` accepts separate norm and overlap caps, as in boundary calibration.
    """
    norm_chi, overlap_chi = chi_pair(chi)
    axis, coordinate = key
    sites = ([(i, coordinate) for i in range(state.Lx)] if axis == 'y'
             else [(coordinate, j) for j in range(state.Ly)])
    largest = max(local_matrix_size(state, site) for site in sites)
    if max_matrix_size is not None and largest > max_matrix_size:
        return state, {'accepted': False, 'converged': False, 'sweeps': 0,
                       'termination_reason': 'matrix_size_limit', 'max_matrix_size': largest,
                       'matrix_size_limit': max_matrix_size, 'axis': axis,
                       'coordinate': coordinate, 'sites': sites, 'solver': solver,
                       'chi': norm_chi, 'overlap_chi': overlap_chi,
                       'target_max_bond': int(target.max_bond())}, boundary, overlap_boundary
    along = 'X' if axis == 'y' else 'Y'
    sample = state[sites[0]].data
    tolerance = resolve_fit_rtol(rtol, dtype=sample.dtype) or 0.
    roundoff = 2e-5 if ar.get_dtype_name(sample) == 'complex64' else 1e-10
    rcond = rcond or (1e-6 if ar.get_dtype_name(sample) == 'complex64' else 1e-12)
    candidate = state.copy()

    def strip(ket, bra=None, handle=None):
        _, network = build_bra_ket(ket=ket.copy(), bra=bra)
        if bra is not None:
            # Candidate bra labels must match the norm layer even when the
            # exact target has different internal bonds after gate splitting.
            mapping = {}
            inner = set(bra.inner_inds())
            for site in bra.gen_site_coos():
                for original, actual in zip(bra[site].inds, network[bra.site_tag(*site), 'BRA'].inds):
                    if original in inner:
                        mapping[actual] = f'{original}_*'
            network.reindex_(mapping)
            # Checked overlap handles can use different bra labels. Cache
            # signatures must describe the relabeled network actually fitted.
            layer_sources(ket, bra, network)
        result, handle = boundary_strip(network, axis=axis, coordinate=coordinate,
                                        boundary=handle if reuse else None,
                                        chi=norm_chi if bra is None else overlap_chi,
                                        contraction_opt=contraction_opt, boundary_kwargs=boundary_kwargs)
        # Quimb selections omit the full-network exponent; transverse boundary
        # cuts contain only their own compression exponents.
        result.exponent += network.exponent
        return result, handle

    aa, boundary = strip(candidate, handle=boundary)
    ab, overlap_boundary = strip(target, candidate, overlap_boundary)
    bb = strip(target)[0] if target_norm is None else None
    report = {'axis': axis, 'coordinate': coordinate, 'sites': sites,
              'solver': solver, 'max_sweeps': sweeps, 'sweeps': 0,
              'rtol': tolerance, 'accepted': False, 'converged': False,
              'target_max_bond': int(target.max_bond()),
              'chi': norm_chi, 'overlap_chi': overlap_chi,
              'max_matrix_size': largest, 'matrix_size_limit': max_matrix_size,
              'local_solves': [],
              'fidelity_convention': 'fixed boundary estimate against exact strip-block target'}
    try:
        log_aa = _norm_log(_scalar(aa, contraction_opt), roundoff)
        log_bb = _norm_log(_scalar(bb, contraction_opt) if bb is not None else target_norm, roundoff)
    except ValueError:
        report['termination_reason'] = 'invalid_environment'
        return state, report, boundary, overlap_boundary
    aa.exponent -= log_aa
    ab.exponent -= .5 * (log_aa + log_bb)
    del bb
    caches = [StripSweepCursor(network, axis=along, length=len(sites), optimize=contraction_opt)
              for network in (aa, ab)] if reuse else []

    def score():
        norm = _value(aa, contraction_opt)
        overlap = _value(ab, contraction_opt)
        if (not math.isfinite(norm.real) or not math.isfinite(norm.imag) or norm.real <= 0
                or abs(norm.imag) > roundoff * norm.real):
            return None
        try:
            fidelity = abs(overlap)**2 / norm.real
        except OverflowError:
            return None
        cost = norm.real - 2 * overlap.real + 1.
        if (not math.isfinite(cost) or not math.isfinite(fidelity)
                or cost < -roundoff or fidelity > 1. + roundoff):
            return None
        return max(0., cost), max(0., 1. - fidelity)

    initial = score()
    if initial is None:
        report['termination_reason'] = 'invalid_environment'
        return state, report, boundary, overlap_boundary
    best_score = initial
    best = None
    report.update(cost_history=[initial[0]], infidelity_before=initial[1],
                  infidelity_after=initial[1], termination_reason='max_sweeps')
    for sweep in range(sweeps):
        for cache in caches:
            cache.start(reverse=bool(sweep % 2))
        positions = range(len(sites)) if sweep % 2 == 0 else range(len(sites) - 1, -1, -1)
        valid = True
        for position in positions:
            site = sites[position]
            tag = state.site_tag(*site)
            local = []
            for index, network in enumerate((aa, ab)):
                if reuse:
                    reduced = caches[index].local(position)
                else:
                    reduced = network.copy()
                # select() retains virtual tensor views. Keep Quimb's temporary
                # tags/index order and writes local to this one-site solve.
                reduced = reduced.copy()
                reduced.retag_({'KET': '__KET__', 'BRA': '__BRA__'})
                local.append(reduced)
            local_aa, local_ab = local
            ket = local_aa[tag, '__KET__']
            bra = local_aa[tag, '__BRA__']
            ket.add_tag('__VAR0__')
            bra.add_tag('__VAR0__')
            local_ab[tag, '__BRA__'].add_tag('__VAR0__')
            # Precontract and Hermitianize the small local norm. Physical
            # identity legs stay implicit and serve as RHS batch dimensions.
            shared = set(ket.inds) & set(bra.inds)
            left = tuple(i for i in bra.inds if i not in shared)
            right = tuple(i for i in ket.inds if i not in shared)
            env = local_aa.select('__VAR0__', which='!all')
            env.exponent = local_aa.exponent
            matrix, exponent = env.contract(all, output_inds=(*left, *right),
                                             optimize=contraction_opt, strip_exponent=True,
                                             preserve_tensor=True)
            size = math.prod(ket.ind_size(i) for i in right)
            data = matrix.data.reshape(size, size)
            matrix.modify(data=((data + dag(data)) * .5).reshape(matrix.shape))
            overlap_bra = local_ab[tag, '__BRA__']
            rhs = local_ab.select('__VAR0__', which='!all')
            rhs.exponent = local_ab.exponent
            vector, rhs_exponent = rhs.contract(
                all, output_inds=overlap_bra.inds, optimize=contraction_opt,
                strip_exponent=True, preserve_tensor=True,
            )
            # Quimb selects hole networks without their parent exponents.
            # Supply zero-exponent local equations with their relative scale
            # explicitly on b, dividing A and b by the same positive factor.
            try:
                vector.modify(data=vector.data * 10.**float(rhs_exponent - exponent))
            except OverflowError:
                valid = False
                break
            if not all_finite(matrix.data) or not all_finite(vector.data):
                valid = False
                break
            if solver == 'pinv':
                physical = tuple(i for i in overlap_bra.inds if i not in left)
                rhs_data = vector.transpose(*left, *physical).data.reshape(size, -1)
                shape = tuple(ket.ind_size(i) for i in right)
                try:
                    answer, local_report = solve_positive(
                        matrix.data.reshape(size, size), rhs_data, rcond=rcond,
                    )
                except ValueError:
                    valid = False
                    break
                local_report.update(site=site, sweep=sweep)
                report['local_solves'].append(local_report)
                solved = qtn.Tensor(answer.reshape(*shape, *(ket.ind_size(i) for i in physical)),
                                    inds=(*right, *physical))
                ket.modify(data=solved.transpose(*ket.inds).data)
            else:
                local_aa = qtn.TensorNetwork([ket, bra, matrix], virtual=True)
                local_ab = qtn.TensorNetwork([overlap_bra, vector], virtual=True)
                dummy = qtn.TensorNetwork([ket.copy()])
                qtn.tensor_network_fit_als(
                    dummy, dummy, steps=1, tol=0., tnAA=local_aa, tnAB=local_ab,
                    xBB=1., dense_solve=True, solver_dense='eigh', enforce_pos=True,
                    pos_smudge=rcond, contract_optimize=contraction_opt, inplace=False,
                )
            data = ket.transpose(*candidate[site].inds).data
            if not all_finite(data):
                valid = False
                break
            candidate[site].modify(data=data)
            aa[tag, 'KET'].modify(data=data)
            aa[tag, 'BRA'].modify(data=ar.do('conj', data))
            ab[tag, 'BRA'].modify(data=ar.do('conj', data))
        report['sweeps'] += 1
        current = score() if valid else None
        if current is None:
            report['termination_reason'] = 'invalid_candidate'
            break
        report['cost_history'].append(current[0])
        if current[0] < best_score[0] and current[1] <= best_score[1]:
            best, best_score = candidate.copy(), current
        previous = report['cost_history'][-2]
        if tolerance > 0 and (current[0] <= tolerance or abs(current[0] - previous) <= tolerance):
            report['converged'] = True
            report['termination_reason'] = ('residual_tolerance' if current[0] <= tolerance
                                            else 'cost_tolerance')
            break
    report['environment_reuse'] = boundary.mps_b.environment_cache.report()
    report['overlap_environment_reuse'] = overlap_boundary.mps_b.environment_cache.report()
    report['strip_environment_reuse'] = ({'norm': caches[0].report(), 'overlap': caches[1].report()}
                                          if reuse else None)
    if best is not None:
        # The solve used unit initial norms. Restore the target's physical scale.
        best.exponent += .5 * (log_bb - log_aa)
        if balance:
            for site in sites:
                tensor = best[site]
                scale = float(ar.do('max', ar.do('abs', tensor.data)))
                if scale > 0 and math.isfinite(scale):
                    tensor.modify(data=tensor.data / scale)
                    best.exponent += math.log10(scale)
        report.update(accepted=True, infidelity_after=best_score[1], best_cost=best_score[0])
        return best, report, boundary, overlap_boundary
    return state, report, boundary, overlap_boundary
