"""Strip targets, SU guesses, native ALS selection, and fitted boundary reuse."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer
from pepsy.optimizers.peps._strip_batch import collect_strip_batch, layered_target, simple_update_guess
from pepsy.optimizers.sweep import SweepOptimizer
from pepsy.boundary import BdyMPS, CompBdy, build_bra_ket


def inputs(backend='numpy', shape=(3, 3)):
    state = qtn.PEPS.rand(*shape, bond_dim=2, seed=82, dtype='complex128')
    state /= state.norm()
    gate = np.diag(np.exp(.2j * np.array([1., -1., -1., 1.])))
    convert = np.asarray
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        convert = torch.as_tensor
        state.apply_to_arrays(convert)
    return state, convert(gate)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_su_guess_preserves_untruncated_state_and_does_not_mutate_input(backend):
    state, gate = inputs(backend, (2, 3))
    before = ar.to_numpy(state.to_dense()).copy()
    entries = [(gate, ((0, 0), (0, 1)), None), (gate, ((0, 1), (0, 2)), None)]
    exact = state.copy()
    for g, sites, _ in entries:
        exact.gate_(g, sites, contract='split', cutoff=0.)
    guess = simple_update_guess(state, entries, chi=8, cutoff=0., cutoff_mode='rel')
    np.testing.assert_allclose(ar.to_numpy(guess.to_dense()), ar.to_numpy(exact.to_dense()), atol=1e-11)
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), before)
    assert ar.infer_backend(next(iter(guess)).data) == backend


def test_strip_batches_stop_at_axis_and_one_site_barriers():
    state, gate = inputs()
    entries = [(gate, ((0, 0), (1, 0))), (gate, ((1, 0), (2, 0))),
               (np.eye(2), (0, 1)), (gate, ((0, 0), (0, 1))),
               (gate, ((0, 1), (0, 2))), (gate, ((1, 0), (1, 1)))]
    opt = PepsOptimizer(state, entries, chi=2, contraction_opt='greedy')
    assert collect_strip_batch(opt, 0)[1:] == (2, 2)
    assert collect_strip_batch(opt, 3)[1:] == (2, 5)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('solver', ['auto', 'cg'])
def test_strip_run_fixed_unitary_target_caps_and_normalization(backend, axis, solver, monkeypatch):
    state, gate = inputs(backend)
    sites = ([(0, i) for i in range(3)] if axis == 'x' else [(i, 0) for i in range(3)])
    entries = [(gate, (sites[0], sites[1])), (gate, (sites[1], sites[2]))]
    opt = PepsOptimizer(state, entries, chi=2, optimizer='als',
                        optimizer_options={'linear_solver': solver, 'n_round_trips': 1,
                                           'cg_shift': 1e-10},
                        boundary_chi=(16, 32), normalize_chi=32, evaluation_chi=(32, 64),
                        boundary_convergence=False, fit_mode='direct',
                        contraction_opt='greedy', sweep_kwargs={'local_contraction_opt': 'greedy'})
    observed = []
    original = SweepOptimizer.optimize_axis
    def check(self, actual_axis, **kwargs):
        assert self.chi == (16, 32)
        assert self.target_norm == (1., 0.)
        assert actual_axis == axis and kwargs['start_index'] == 0
        assert self.state_target.num_tensors == state.num_tensors + 2 * len(entries)
        assert '__PEPS_GATE__' in self.state_target.tags
        target = ar.to_numpy(self.state_target.to_dense())
        assert np.vdot(target, target).real == pytest.approx(1., abs=1e-9)
        before = ar.to_numpy(self.state.to_dense())
        original_target = target.copy()
        report = original(self, actual_axis, **kwargs)
        after = ar.to_numpy(self.state.to_dense())
        def fid(x):
            return abs(np.vdot(target, x))**2 / np.vdot(x, x).real
        assert fid(after) >= fid(before) - 1e-9
        np.testing.assert_allclose(ar.to_numpy(self.state_target.to_dense()), original_target, atol=0.)
        observed.append(report)
        return report
    monkeypatch.setattr(SweepOptimizer, 'optimize_axis', check)
    def no_target_contraction(*args, **kwargs):
        pytest.fail('Unitary target norm was recomputed')
    monkeypatch.setattr(opt, '_measure_target_norm', no_target_contraction)
    opt.run(k_2q_batch='strip', warmstart_method='simple-update', infidelity_tol=0.,
            normalize_initial=False)
    assert len(observed) == 1
    assert all(report['n_round_trips'] == 1 for report in observed[0])
    v = ar.to_numpy(opt.state.to_dense())
    assert np.vdot(v, v).real == pytest.approx(1., abs=1e-9)


@pytest.mark.parametrize('limit,expected', [(1024, 'pinv'), (1, 'cg')])
def test_auto_als_uses_spectral_or_matrix_free_solves(limit, expected):
    from test_peps_sweep_als import fixture, loss
    sweep = fixture()
    before = loss(sweep.state, sweep.state_target)
    report = sweep.optimize_strip(0, solver_options={
        'linear_solver': 'auto', 'max_matrix_size': limit, 'n_round_trips': 1,
        'cg_shift': 1e-10,
    })
    assert loss(sweep.state, sweep.state_target) < before
    assert any(s.get('solver') == expected and s['accepted'] for s in report['local_solves'])


@pytest.mark.parametrize('method', ['direct', 'src', 'sdc'])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_boundary_seed_once_then_refit_changed_target(method, backend, monkeypatch):
    state, _ = inputs(backend)
    _, network = build_bra_ket(ket=state)
    bdy = BdyMPS(tn_double=network, chi=8, lazy=True)
    original = qtn.tensor_network_1d_compress
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs.get('method'))
        return original(*args, **kwargs)
    monkeypatch.setattr(qtn, 'tensor_network_1d_compress', counted)
    def fit():
        _, tn = build_bra_ket(ket=state)
        c = CompBdy(tn, bdy.mps_b, contraction_opt='greedy', fit_mode='dmrg',
                    fit_init_strategy=f'reuse-{method}', fit_max_bond=8, fit_cutoff=0.)
        return c.run(direction='y', n_iter=2)
    first = fit()
    seeds = len(calls)
    assert seeds > 0
    assert complex(fit()) == pytest.approx(complex(first), rel=1e-9)
    assert len(calls) == seeds
    # A fitted MPS is a starting guess, not a valid old contraction.
    state[0, 0].modify(data=state[0, 0].data * 1.1)
    assert complex(fit()) == pytest.approx(complex(first) * 1.21, rel=1e-8)
    assert len(calls) == seeds


def test_two_strip_grouping_preserves_axis_and_spatial_barriers():
    state, gate = inputs()
    entries = [(gate, ((0, j), (1, j))) for j in (0, 1, 2)]
    entries += [(gate, ((0, 0), (0, 1)))]
    opt = PepsOptimizer(state, entries, chi=2, contraction_opt='greedy')
    assert collect_strip_batch(opt, 0, width=2)[1:] == (2, 2)
    assert collect_strip_batch(opt, 1, width=2)[1:] == (2, 3)


def test_strip_driver_transfers_guesses_and_refits():
    state, gate = inputs()
    entries = [(gate, ((0, j), (1, j))) for j in (0, 1)]
    opt = PepsOptimizer(
        state, entries, chi=2, optimizer='als', contraction_opt='greedy',
        boundary_chi=(8, 12), normalize_chi=32, evaluation_chi=(32, 64),
        boundary_convergence=False, fit_mode='dmrg',
        boundary_kwargs={'fit_init_strategy': 'reuse-direct'},
        optimizer_options={'linear_solver': 'auto', 'n_round_trips': 1},
        sweep_kwargs={'local_contraction_opt': 'greedy'},
    )
    opt.run(k_2q_batch='strip', warmstart_method='simple-update', infidelity_tol=0.)
    records = opt.get_step_records()
    assert len(records) == 2
    reused = records[1]['optimizer_result']['boundary_guesses_reused']
    assert reused['norm'] > 0
    assert reused['overlap'] > 0
    vector = ar.to_numpy(opt.state.to_dense())
    assert np.vdot(vector, vector).real == pytest.approx(1., abs=1e-9)


def test_sweep_metrics_share_known_target_norm_and_allow_independent_measurement():
    state, _ = inputs()
    target = state * 2.
    target.mangle_inner_()
    sweep = SweepOptimizer(state, target, target_norm=4., chi=(16, 32),
                           contraction_opt='greedy', renormalize_state=False)
    assert sweep.metrics()[0] == pytest.approx(1., abs=1e-9)
    assert sweep.infidelity(fit_mode='direct') == pytest.approx(0., abs=1e-9)
    sweep.set_target_norm(1.)
    # A supplied norm is trusted, not silently replaced by a measurement.
    assert sweep.metrics()[0] == pytest.approx(4., abs=1e-9)
    assert sweep.metrics(norm_target=None)[0] == pytest.approx(1., abs=1e-9)
    assert sweep.infidelity(norm_target=None, fit_mode='direct') == pytest.approx(0., abs=1e-9)


@pytest.mark.parametrize('axis', ['x', 'y'])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('gate_kind', ['diagonal', 'full-rank'])
def test_layered_target_exactness_and_boundary_fit_receives_gate_factors(axis, backend, gate_kind, monkeypatch):
    from pepsy.fitting import FIT

    state, gate = inputs(backend)
    if gate_kind == 'full-rank':
        rng = np.random.default_rng(84)
        unitary = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        gate = ar.do('array', unitary, like=gate)
    entries = [(gate, ((0, 0), (1, 0)), None), (gate, ((0, 0), (0, 1)), None),
               (gate, ((1, 0), (2, 0)), None), (gate, ((1, 2), (2, 2)), None)]
    target = layered_target(state, entries)
    absorbed = state.copy()
    for g, where, _ in entries:
        absorbed.gate_(g, where, contract='split', cutoff=0.)
    reference = ar.to_numpy(absorbed.to_dense()).ravel()
    np.testing.assert_allclose(ar.to_numpy(target.to_dense()).ravel(), reference, atol=1e-11)
    for original in state:
        site, = [tag for tag in original.tags if tag.startswith('I')]
        base = target[site, '__PEPS_BASE__']
        assert base.shape == original.shape
        np.testing.assert_array_equal(ar.to_numpy(base.data), ar.to_numpy(original.data))
    for t in target:
        assert len([tag for tag in t.tags if tag.startswith('I')]) == 1
        assert len([tag for tag in t.tags if tag.startswith('X')]) == 1
        assert len([tag for tag in t.tags if tag.startswith('Y')]) == 1

    seen = []
    init = FIT.__init__
    def check(self, tn, *args, **kwargs):
        if '__PEPS_GATE__' in tn.tags:
            seen.append(True)
            assert any('__PEPS_BASE__' in t.tags for t in tn)
        init(self, tn, *args, **kwargs)
    monkeypatch.setattr(FIT, '__init__', check)
    _, network = build_bra_ket(ket=target.copy(), bra=state)
    boundaries = BdyMPS(tn_double=network, chi=64, lazy=True)
    comp = CompBdy(network, boundaries.mps_b, fit_mode='dmrg',
                  fit_max_bond=64, contraction_opt='greedy')
    overlap = comp.run(direction=axis, n_iter=3)
    expected = np.vdot(ar.to_numpy(state.to_dense()).ravel(), reference)
    assert complex(overlap) == pytest.approx(expected, abs=1e-8)
    assert seen
