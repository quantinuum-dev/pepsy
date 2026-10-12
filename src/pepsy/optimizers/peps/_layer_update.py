"""Bounded-layer refinement with a fixed target and full-layer rollback."""

import math

import autoray as ar

from ..._internal.cutoff import resolve_fit_rtol
from ...boundary.metrics import peps_infidelity, peps_norm, _as_scaled_scalar
from ._boundary_convergence import chi_pair
from ._strip_update import refine_strip, local_matrix_size, _norm_log


def refine_layer(state, target, *, chi, contraction_opt, boundary_kwargs,
                 sweeps, rtol, rcond=None, boundary=None, overlap_boundary=None,
                 reuse=True, solver='quimb', balance=False,
                 max_matrix_size=1024, evaluation_chi=None):
    """Alternate rows/columns against one exact window, accepting whole cycles.

    ``sweeps`` counts complete row-and-column cycles, each visiting every site
    twice. Global checks independently rebuild scalar environments; only the
    unchanged target norm is retained between checks at the same cap.
    """
    norm_chi, overlap_chi = chi_pair(chi)
    sample = next(iter(state)).data
    tolerance = resolve_fit_rtol(rtol, dtype=sample.dtype) or 0.
    roundoff = 2e-5 if ar.get_dtype_name(sample) == 'complex64' else 1e-10
    largest = max(local_matrix_size(state, site) for site in state.gen_site_coos())
    report = {'scope': 'layer', 'accepted': False, 'converged': False,
              'solver': solver, 'cycles': [], 'max_sweeps': sweeps, 'sweeps': 0,
              'rtol': tolerance, 'chi': norm_chi, 'overlap_chi': overlap_chi,
              'max_matrix_size': largest, 'matrix_size_limit': max_matrix_size,
              'target_max_bond': int(target.max_bond()),
              'fidelity_convention': 'whole fixed-layer target, independently contracted'}
    if max_matrix_size is not None and largest > max_matrix_size:
        report['termination_reason'] = 'matrix_size_limit'
        return state, report, boundary, overlap_boundary
    check_chi = chi if evaluation_chi is None else evaluation_chi
    metric_options = {k: v for k, v in boundary_kwargs.items()
                      if k not in ('chi', 'norm', 'norm_target', 'balance_bonds')}
    metric_options.update(chi=check_chi, contraction_opt=contraction_opt,
                          strip_exponent=True, progress=False)
    target_norm = None

    def score(candidate):
        nonlocal target_norm
        result = peps_infidelity(candidate.copy(), target.copy(), norm_target=target_norm,
                                 **metric_options)
        log_a = _norm_log(_as_scaled_scalar(result['norm']), roundoff)
        log_b = _norm_log(_as_scaled_scalar(result['norm_target']), roundoff)
        mantissa, exponent = _as_scaled_scalar(result['overlap'])
        overlap = complex(mantissa) * 10.**(exponent - log_b)
        cost = 10.**(log_a - log_b) - 2. * overlap.real + 1.
        infidelity = float(result['infidelity'])
        if (not math.isfinite(cost) or not math.isfinite(infidelity)
                or cost < -roundoff or not -roundoff <= infidelity <= 1. + roundoff):
            raise ValueError('invalid whole-layer objective')
        target_norm = result['norm_target']
        return max(0., cost), max(0., min(1., infidelity))

    try:
        best_score = score(state)
    except (ValueError, OverflowError, ZeroDivisionError):
        report['termination_reason'] = 'invalid_environment'
        return state, report, boundary, overlap_boundary
    # The pure target is unchanged through every site/cycle. Its fit-cap norm
    # may differ from the independent evaluation cap, so cache them separately.
    if chi_pair(check_chi)[0] == norm_chi:
        fit_target_norm = target_norm
    else:
        opts = {**metric_options, 'chi': norm_chi}
        fit_target_norm = peps_norm(target.copy(), **opts)
    best, candidate = state, state
    report.update(infidelity_before=best_score[1], infidelity_after=best_score[1],
                  cost_history=[best_score[0]], evaluation_chi=chi_pair(check_chi),
                  termination_reason='max_sweeps')
    for cycle in range(sweeps):
        keys = [('x', i) for i in range(state.Lx)] + [('y', j) for j in range(state.Ly)]
        if cycle % 2:
            keys.reverse()
        records = []
        for key in keys:
            candidate, item, boundary, overlap_boundary = refine_strip(
                candidate, target, key=key, chi=chi, contraction_opt=contraction_opt,
                boundary_kwargs=boundary_kwargs, sweeps=1, rtol=None, rcond=rcond,
                boundary=boundary, overlap_boundary=overlap_boundary, reuse=reuse,
                solver=solver, balance=balance, max_matrix_size=max_matrix_size,
                target_norm=fit_target_norm,
            )
            records.append(item)
        report['sweeps'] += 1
        try:
            current = score(candidate)
        except (ValueError, OverflowError, ZeroDivisionError):
            report['cycles'].append({'accepted': False, 'strips': records})
            report['termination_reason'] = 'invalid_candidate'
            break
        previous = best_score[0]
        accepted = current[0] < previous and current[1] <= best_score[1] + roundoff
        report['cycles'].append({'accepted': accepted, 'cost': current[0],
                                 'infidelity': current[1], 'strips': records})
        report['cost_history'].append(current[0])
        if accepted:
            best, best_score = candidate, current
            report.update(accepted=True, infidelity_after=current[1], best_cost=current[0])
        if tolerance > 0 and (current[0] <= tolerance or abs(current[0] - previous) <= tolerance):
            report.update(converged=True, termination_reason='cost_tolerance')
            break
        if not accepted:
            report['termination_reason'] = 'cycle_rejected'
            break
    return best, report, boundary, overlap_boundary
