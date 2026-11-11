"""Gate-dependent QR/LQ grouping against independent dense PEPO targets and costs."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    exact_reduced_update_problem, prepare_reduced_bond_pair, solve_reduced_als,
)
from pepsy.bp.reduced_update import _svd_initial_factors


def dense(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def apply_dense(state, matrix, gate, where, side):
    sites = list(state.gen_site_coos())
    dims = [state[site].ind_size(fn(*site))
            for fn in (state.upper_ind, state.lower_ind) for site in sites]
    axes = [sites.index(site)+(len(sites) if side == 'lower' else 0) for site in where]
    sizes = [dims[axis] for axis in axes]
    rest = [axis for axis in range(len(dims)) if axis not in axes]
    value = np.tensordot(gate.reshape(*sizes, *sizes), matrix.reshape(dims),
                         axes=((2, 3), tuple(axes)))
    return value.transpose(np.argsort(axes+rest)).reshape(matrix.shape)


@pytest.mark.parametrize('backend', ['numpy', 'cpu', 'cuda'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('which', ['upper', 'lower', 'both'])
def test_selected_legs_reconstruct_targets_and_dense_weighted_errors(backend, dtype, which):
    state = qtn.PEPO.rand(2, 2, bond_dim=2, phys_dim=2, dtype=dtype, seed=114)
    rng = np.random.default_rng(491)
    # Unequal operator dimensions expose accidental upper/lower fusion or swaps.
    for site in state.sites:
        tensor = state[site]
        shape = tuple(3 if ix == state.upper_ind(*site) else dim
                      for ix, dim in zip(tensor.inds, tensor.shape))
        tensor.modify(data=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(dtype))
    state /= np.linalg.norm(state.to_dense(optimize='greedy'))
    if backend == 'numpy':
        convert = lambda x: np.asarray(x, dtype=dtype)
        solver = 'quimb'
    else:
        torch = pytest.importorskip('torch')
        if backend == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        convert = lambda x: torch.as_tensor(x, dtype=getattr(torch, dtype), device=backend)
        state.apply_to_arrays(convert)
        solver = 'torch'
    gauges = {ix: convert([.7, 1.2]).real for ix in state.inner_inds()}
    pair = prepare_reduced_bond_pair(state, gauges, where=((0, 1), (0, 0)),
                                     which=which, input_mode='su_core', run_bp=False)
    expected_physical = 6 if which == 'both' else 3 if which == 'upper' else 2
    assert pair.theta_shape[1:3] == (expected_physical, expected_physical)
    for site, outer, active in zip(pair.where, (pair.q_left, pair.q_right),
                                   (pair.r_left, pair.l_right)):
        if which != 'both':
            spectator = getattr(state, ('lower' if which == 'upper' else 'upper')+'_ind')(*site)
            assert spectator in outer.inds and spectator not in active.inds
    original = state.copy()
    original.gauge_simple_insert(gauges)
    before = dense(original.to_dense(optimize='greedy'))
    tolerance = dict(rtol=3e-5, atol=3e-5) if dtype == 'complex64' else dict(rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(dense(pair.reconstruct_tn().to_dense(optimize='greedy')),
                               before, **tolerance)
    expected = before
    payloads = []
    for side in (('upper', 'lower') if which == 'both' else (which,)):
        size = 9 if side == 'upper' else 4
        gate = np.linalg.qr(rng.normal(size=(size, size))+1j*rng.normal(size=(size, size)))[0]
        expected = apply_dense(state, expected, gate, pair.where, side)
        payloads.append(convert(gate))
    gate = tuple(payloads) if which == 'both' else payloads[0]
    problem = exact_reduced_update_problem(pair, gate, psd_project=False, optimize='greedy')
    target = pair.reconstruct_tn(*_svd_initial_factors(problem.target, 64))
    np.testing.assert_allclose(dense(target.to_dense(optimize='greedy')), expected, **tolerance)
    solution = solve_reduced_als(problem, solver=solver, max_bond=1, max_iterations=3,
                                 tol=0., monitor_convergence=True)
    fitted = dense(pair.reconstruct_tn(solution.left, solution.right).to_dense(optimize='greedy'))
    error = fitted-expected
    assert problem.cost(solution.theta()) == pytest.approx(
        float(np.vdot(error, error).real), rel=tolerance['rtol'], abs=tolerance['atol'])
    assert problem.target_norm == pytest.approx(float(np.vdot(expected, expected).real),
                                                rel=tolerance['rtol'])
    if which != 'both':
        other = 'lower' if which == 'upper' else 'upper'
        from pepsy.bp import su_cluster_reduced_update_problem
        with pytest.raises(ValueError, match='does not match'):
            su_cluster_reduced_update_problem(pair, payloads[0], which=other, optimize='greedy')


def test_bulk_one_sided_reduction_halves_each_reduced_virtual_dimension():
    state = qtn.PEPO.rand(4, 4, bond_dim=6, phys_dim=2, dtype='complex128', seed=51)
    where = ((1, 1), (1, 2))
    full = prepare_reduced_bond_pair(state, where=where, which='both', run_bp=False)
    for side in ('upper', 'lower'):
        pair = prepare_reduced_bond_pair(state, where=where, which=side, run_bp=False)
        assert full.theta_shape == (24, 4, 4, 24)
        assert pair.theta_shape == (12, 2, 2, 12)
        assert np.prod(full.theta_shape) == 16*np.prod(pair.theta_shape)


def test_joint_gate_and_adjoint_keep_both_legs_in_one_cluster_fit():
    torch = pytest.importorskip('torch')
    from pepsy.bp import apply_reduced_loop_cluster_gate

    state = qtn.PEPO.rand(2, 2, bond_dim=2, phys_dim=2, dtype='complex128', seed=59)
    state.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    where = ((0, 0), (0, 1))
    rng = np.random.default_rng(74)
    gate = np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))[0]
    expected = apply_dense(state, dense(state.to_dense()), gate, where, 'upper')
    expected = apply_dense(state, expected, gate.conj(), where, 'lower')
    payload = (torch.tensor(gate), torch.tensor(gate.conj()))
    result = apply_reduced_loop_cluster_gate(
        state, {}, payload, where=where, which='both', run_bp=False, max_bond=16,
        base_radius=3, max_loop_size=0, psd_project=False, optimize='greedy',
        als_opts=dict(solver='torch', max_iterations=2, tol=1e-12))
    assert result.pair.theta_shape[1:3] == (4, 4)
    np.testing.assert_allclose(dense(result.physical_tn.to_dense()), expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(dense(result.pair.gate_target_tn(payload).to_dense()), expected,
                               rtol=2e-11, atol=2e-11)
