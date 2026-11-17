"""Independent ALS starts sharing one frozen target and environment."""

from dataclasses import replace
import math

import autoray as ar

from .._internal.random import backend_random_array


def validate_restarts(random_restarts, restart_seed):
    for name, value in (('random_restarts', random_restarts), ('restart_seed', restart_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')


def random_factors(initial, seed):
    """Match factor norms without changing a global backend random stream."""
    # Distinct integer seeds keep both factors independent, including with
    # older Autoray's supported NumPy-staging random-array compatibility route.
    factors = []
    for offset, template in enumerate(initial):
        value = backend_random_array(template.shape, like=template, rng=2*seed+offset)
        scale = ar.do('linalg.norm', template.reshape(-1))
        scale = ar.do('where', scale > 0, scale, ar.do('ones_like', scale))
        value = value * (scale / ar.do('linalg.norm', value.reshape(-1)))
        factors.append(value)
    return tuple(factors)


def select_best(solutions, cost, *, restart_seed, initial_name, target_norm=None):
    """Compare actual residuals in the same metric, keeping the first tie."""
    scores = [float(cost(solution)) for solution in solutions]
    finite = [i for i, score in enumerate(scores) if math.isfinite(score)]
    if not finite:
        raise ValueError('ALS restart trials have no finite objective')
    selected = min(finite, key=scores.__getitem__)
    best = solutions[selected]
    trials = []
    for i, (solution, score) in enumerate(zip(solutions, scores)):
        norm = solution.target_norm if solution.target_norm is not None else target_norm
        trials.append(dict(
            initialization=initial_name if i == 0 else 'random',
            seed=None if i == 0 else restart_seed+i-1,
            cost=score if math.isfinite(score) else None,
            normalized_residual=(score/norm if math.isfinite(score) and norm and norm > 0 else None),
            solver=solution.solver, iterations=solution.iterations,
            converged=solution.converged, termination_reason=solution.termination_reason,
        ))
    # The winning solution retains its own iteration/convergence fields;
    # total_iterations separately accounts for the extra optimization work.
    return replace(best, solver_report={**best.solver_report, 'restarts': dict(
        selected_trial=selected, trials=trials,
        total_iterations=(sum(s.iterations for s in solutions)
                          if all(s.iterations is not None for s in solutions) else None),
    )})
