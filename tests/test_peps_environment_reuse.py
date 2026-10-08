"""Cached cuts must agree with fresh contraction after local state changes."""

import pytest
import quimb.tensor as qtn

from pepsy.boundary.metrics import build_bra_ket
from pepsy.boundary.states import BdyMPS
from pepsy.boundary.sweeps import CompBdy
from pepsy.boundary._reuse import EnvironmentCache
from pepsy.tensors.contractions import build_optimizer


def setup(dtype='complex128'):
    torch = pytest.importorskip('torch')
    state = qtn.PEPS.rand(3, 4, bond_dim=2, dtype=dtype, seed=94)
    state.apply_to_arrays(lambda x: torch.tensor(x))
    _, network = build_bra_ket(ket=state)
    boundary = BdyMPS(tn_double=network, chi=16, lazy=True)
    boundary.mps_b.environment_cache = EnvironmentCache()
    policy = build_optimizer(max_repeats=1, parallel=False)
    return state, boundary, policy


def compressor(state, boundary, policy, **kwargs):
    _, network = build_bra_ket(ket=state)
    return CompBdy(network, boundary.mps_b, contraction_opt=policy,
                   fit_mode='direct', fit_max_bond=16, fit_cutoff=0., **kwargs)


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_cached_left_and_right_cuts_only_rebuild_changed_dependencies(axis, dtype):
    state, boundary, policy = setup(dtype)
    comp = compressor(state, boundary, policy)
    for side in ('left', 'right'):
        comp.move_bdy(direction=f'{axis}_{side}', n_iter=2)
    before = dict(boundary.mps_b)
    cache = boundary.mps_b.environment_cache
    previous_rebuilds = cache.rebuilds
    comp = compressor(state, boundary, policy)
    for side in ('left', 'right'):
        comp.move_bdy(direction=f'{axis}_{side}', n_iter=2)
    assert cache.rebuilds == previous_rebuilds
    assert cache.hits > 0
    # An in-place write must be detected as well as a new data allocation.
    site = (1, 2)
    state[site].data.mul_(1.03 + .02j)
    comp = compressor(state, boundary, policy)
    for side in ('left', 'right'):
        comp.move_bdy(direction=f'{axis}_{side}', n_iter=2)
    n = state.Lx if axis == 'x' else state.Ly
    changed = site[0] if axis == 'x' else site[1]
    for key, old in before.items():
        if key[0] != axis.upper():
            continue
        step = int(key[1:-2])
        affected = changed <= step if key[-1] == 'l' else changed >= n - 1 - step
        assert (boundary.mps_b[key] is old) == (not affected)
    cached = comp.run(direction=axis, n_iter=2)
    _, network = build_bra_ket(ket=state)
    fresh = BdyMPS(tn_double=network, chi=16, lazy=True)
    expected = compressor(state, fresh, policy).run(direction=axis, n_iter=2)
    assert complex(cached) == pytest.approx(complex(expected), rel=3e-5 if dtype == 'complex64' else 1e-11)


def test_changed_compression_policy_and_modified_boundary_invalidate_cache():
    state, boundary, policy = setup()
    comp = compressor(state, boundary, policy)
    comp.move_bdy(direction='y_left', n_iter=2)
    old = boundary.mps_b['Y0_l']
    old[next(iter(old.site_tags))].data.mul_(1.2)
    comp = compressor(state, boundary, policy)
    comp.move_bdy(direction='y_left', n_iter=2)
    assert boundary.mps_b['Y0_l'] is not old
    cache = boundary.mps_b.environment_cache
    hits = cache.hits
    comp.move_bdy(direction='y_left', n_iter=3)
    assert cache.hits == hits


def test_scaled_network_exponent_is_applied_even_when_cuts_are_reused():
    state, boundary, policy = setup()
    a = compressor(state, boundary, policy).run(direction='y', n_iter=2, strip_exponent=True)
    state.exponent += 50
    b = compressor(state, boundary, policy).run(direction='y', n_iter=2, strip_exponent=True)
    assert complex(a[0]) == pytest.approx(complex(b[0]), rel=1e-12)
    assert float(b[1] - a[1]) == pytest.approx(100.)
    assert boundary.mps_b.environment_cache.hits > 0


def test_local_bond_growth_keeps_untouched_left_and_right_cuts():
    import torch
    state, boundary, policy = setup()
    comp = compressor(state, boundary, policy)
    for side in ('left', 'right'):
        comp.move_bdy(direction=f'y_{side}', n_iter=2)
    left, right = boundary.mps_b['Y0_l'], boundary.mps_b['Y0_r']
    gate = torch.diag(torch.exp(torch.tensor([-.3j, .3j, .3j, -.3j], dtype=torch.complex128)))
    state.gate_(gate, ((1, 1), (1, 2)), contract='split', cutoff=0.)
    _, network = build_bra_ket(ket=state)
    boundary._tn_norm = network.copy()
    comp = compressor(state, boundary, policy)
    for side in ('left', 'right'):
        comp.move_bdy(direction=f'y_{side}', n_iter=2)
    assert boundary.mps_b['Y0_l'] is left
    assert boundary.mps_b['Y0_r'] is right
    fresh = BdyMPS(tn_double=network, chi=16, lazy=True)
    actual = comp.run(direction='y', n_iter=2)
    expected = compressor(state, fresh, policy).run(direction='y', n_iter=2)
    assert complex(actual) == pytest.approx(complex(expected), rel=1e-11)


def test_calibration_hands_off_both_norm_and_overlap_cuts():
    import torch
    from pepsy.optimizers.peps import PepsOptimizer
    state, _, policy = setup()
    state /= state.norm()
    gate = torch.diag(torch.exp(torch.tensor([-.2j, .2j, .2j, -.2j], dtype=torch.complex128)))
    opt = PepsOptimizer(state, [(gate, ((1, 0), (1, 1)))], chi=2,
                        contraction_opt=policy, fit_mode='direct',
                        boundary_convergence={'max_chi': 32},
                        optimizer='scipy', optimizer_options={'n_steps': 1},
                        sweep_optimize_kwargs={'n_round_trips': 0, 'compute_initial_loss': False,
                                               'compute_final_loss': False})
    opt.run(k_2q_batch=1, measure_infidelity=False, measure_final_infidelity=False,
            accept_if_improved=False)
    record = opt.get_step_records()[0]
    assert record['boundary_convergence']['converged']
    cache = record['optimizer_result']['environment_reuse']
    assert cache['bdy']['hits'] > 0
    assert cache['bdy_overlap']['hits'] > 0


@pytest.mark.parametrize('earlier_valid', [False, True])
def test_zero_norm_probes_do_not_handoff_missing_or_stale_cuts(monkeypatch, earlier_valid):
    from pepsy.optimizers.peps import PepsOptimizer
    from pepsy.optimizers.peps import optimizer as module
    state, _, policy = setup()
    opt = PepsOptimizer(state, chi=2, contraction_opt=policy, fit_mode='direct',
                        boundary_convergence={'max_chi': 8})
    # Exercise the cache-capable Torch path: an unusable metric must remain an
    # explicit convergence failure, never a missing-handle KeyError.
    def zero_metric(*args, **kwargs):
        if earlier_valid and kwargs['chi'][0] < 8:
            return {'norm': (1., 0.), 'norm_target': (1., 0.), 'overlap': (.8, 0.),
                    'norm_result': None, 'norm_target_result': None,
                    **{name: kwargs[name] for name in ('bdy', 'bdy_target', 'bdy_overlap')}}
        raise ZeroDivisionError('zero approximate norm')
    monkeypatch.setattr(module, 'boundary_infidelity', zero_metric)
    with pytest.warns(RuntimeWarning, match='convergence was not established'):
        result = opt._calibrate_sweep_boundaries(state, state, normalize_chi=4,
                                                 evaluation_chi=(4, 4))
    assert not result['converged']
    assert not result.get('fit_boundaries')
    assert not opt._checked_boundary_environments
