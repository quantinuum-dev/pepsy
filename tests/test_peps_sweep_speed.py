"""Fixed-chi sweep performance controls preserve fresh values and locality."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.sweep import _als
from pepsy.optimizers.sweep import SweepOptimizer
from pepsy.optimizers.sweep._als_contract import ContractionCache, contract_scaled
from test_peps_sweep_als import fixture, loss


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_expression_cache_tracks_values_exponents_shapes_and_index_order(backend):
    cache = ContractionCache(maxsize=2)
    rng = np.random.default_rng(31)
    for dimension, output in [(3, ('a','c')), (3, ('a','c')), (4, ('a','c')),
                              (4, ('c','a')), (3, ('a','c'))]:
        arrays = [rng.normal(size=(2, dimension)), rng.normal(size=(dimension, 5))]
        if backend == 'torch':
            import torch
            arrays = [torch.tensor(a) for a in arrays]
        network = qtn.TensorNetwork([qtn.Tensor(arrays[0], inds=('a','b')),
                                     qtn.Tensor(arrays[1], inds=('b','c'))])
        network.exponent = 70.
        actual, exponent = cache(network, output_inds=output, optimize='greedy', preserve_tensor=True)
        expected, power = contract_scaled(network, output_inds=output,
                                          optimize='greedy', preserve_tensor=True)
        np.testing.assert_allclose(ar.to_numpy(actual.data) * 10.**float(exponent-power),
                                   ar.to_numpy(expected.data), atol=1e-12)
    assert cache.hits == 1 and cache.misses == 4 and len(cache.expressions) == 2


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('method', ['dense-cg', 'cg', 'lbfgs'])
def test_cached_sweep_matches_uncached_reference(axis, backend, method):
    results = []
    for enabled in (False, True):
        sweep = fixture(backend, shape=(3, 3))
        report = sweep.optimize_strip(1, axis=axis, solver_options={
            'linear_solver': method, 'cache_contractions': enabled, 'rtol': 0.})
        results.append((sweep.state.to_dense(), report))
        assert report['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)
    np.testing.assert_allclose(ar.to_numpy(results[0][0]), ar.to_numpy(results[1][0]),
                               atol=2e-8, rtol=2e-8)
    assert results[1][1]['contraction_cache']['hits'] > 0


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('engine', ['dmrg', 'quimb-mps'])
def test_single_strip_freezes_exterior_and_preserves_caps(axis, engine):
    sweep = fixture(shape=(3, 3), engine=engine)
    before = {tag: sweep.state[tag].data for tag in sweep.state.site_tags}
    target = [t.data for t in sweep.state_target]
    chi = sweep.chi
    report = sweep.optimize_strip(1, axis=axis)
    active = sweep._site_tensor_tags(axis, 1)
    assert report['update_applied'] and report['sweep'] == 'strip'
    assert all(s['site'] in active for s in report['local_solves'])
    assert all(sweep.state[tag].data is value for tag,value in before.items() if tag not in active)
    assert all(t.data is value for t,value in zip(sweep.state_target,target))
    assert sweep.chi == chi
    assert report['loss_final'] == pytest.approx(loss(sweep.state, sweep.state_target), abs=1e-9)


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_skip_only_immediately_repeated_converged_endpoints(axis):
    results=[]
    for skip in (False, True):
        sweep=fixture(shape=(3,3))
        report=sweep.optimize_strip(1,axis=axis,solver_options={
            'skip_repeated_endpoints':skip,'rtol':0.})
        results.append((loss(sweep.state,sweep.state_target),report))
    assert results[1][0] == pytest.approx(results[0][0],abs=1e-9)
    assert len(results[1][1]['local_solves']) < len(results[0][1]['local_solves'])
    assert results[1][1]['skipped_site_visits']


def test_unconverged_lbfgs_endpoint_is_not_skipped():
    sweep=fixture(shape=(3,3))
    report=sweep.optimize_strip(1,solver_options={
        'linear_solver':'lbfgs','lbfgs_maxiter':1,'lbfgs_rtol':1e-14,
        'skip_repeated_endpoints':True,'rtol':0.})
    assert not report['skipped_site_visits']
    assert len(report['local_solves']) == 12


def test_absolute_stopping_waits_for_minimum_trips_and_patience():
    sweep=fixture()
    report=sweep.optimize_strip(0,solver_options={
        'n_round_trips':5,'min_round_trips':2,'patience':3,'atol':1.,'rtol':0.})
    assert report['n_round_trips'] == 3
    assert report['termination_reason'] == 'improvement_tolerance'
    assert len(report['round_trip_gains']) == 3
    assert report['loss_final'] <= report['loss_initial']


@pytest.mark.parametrize('controls',[{'atol':-1},{'atol':None},{'patience':0},
    {'min_round_trips':3},{'skip_repeated_endpoints':1},{'cache_contractions':None}])
def test_speed_controls_are_validated(controls):
    with pytest.raises(ValueError):
        _als.options(controls)


@pytest.mark.parametrize('axis,index',[('z',0),('x',-1),('y',99),('x',True)])
def test_strip_bounds_fail_before_mutation(axis,index):
    sweep=fixture()
    before=[t.data for t in sweep.state]
    with pytest.raises(ValueError):
        sweep.optimize_strip(index,axis=axis)
    assert all(t.data is value for t,value in zip(sweep.state,before))


@pytest.mark.parametrize('backend',['numpy','torch'])
@pytest.mark.parametrize('scope',['strip','global'])
def test_lazy_boundary_guesses_match_eager_and_remain_usable(backend,scope):
    base=fixture(backend,shape=(3,3))
    results=[]
    for lazy in (False,True):
        sweep=SweepOptimizer(base.state.copy(),base.state_target.copy(),chi=8,
            fit_mode='dmrg',contraction_opt='greedy',lazy_boundaries=lazy)
        if lazy:
            assert len(sweep.bdy.mps_b) == 0
        if scope == 'strip':
            report=sweep.optimize_strip(1,axis='x',env_n_iter=2)
            assert report['update_applied']
        else:
            sweep.set_optimize_kwargs(optimizer='als',axes=('y','x'),n_round_trips=0,
                env_n_iter=2,compute_initial_loss=False,compute_final_loss=False)
            sweep.run(progress=False,renormalize=False)
        first=ar.to_numpy(sweep.state.to_dense())
        # A restored lazy store must also support a later solve.
        sweep.optimize_strip(1,axis='y',env_n_iter=2)
        results.append((first,ar.to_numpy(sweep.state.to_dense())))
        assert sweep.chi == 8
        sweep.set_state(sweep.state.copy(),normalize_state=False)
        sweep.set_target(sweep.state_target.copy())
        assert sweep.bdy.lazy == lazy and sweep.bdy_overlap.lazy == lazy
    for a,b in zip(*results):
        np.testing.assert_allclose(a,b,rtol=2e-9,atol=2e-9)


@pytest.mark.parametrize('axis',['x','y'])
def test_single_strip_joint_gradient_fit_keeps_exterior(axis):
    sweep=fixture('torch',shape=(3,3))
    before=loss(sweep.state,sweep.state_target)
    exterior={tag:sweep.state[tag].data for tag in sweep.state.site_tags
              if tag not in sweep._site_tensor_tags(axis,1)}
    report=sweep.optimize_strip(1,axis=axis,solver='scipy',
        solver_options={'n_steps':10,'gtol':1e-8})
    assert report['update_applied']
    assert loss(sweep.state,sweep.state_target) < before
    assert all(sweep.state[tag].data is value for tag,value in exterior.items())
