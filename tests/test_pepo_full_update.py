"""Native PEPO full updates against independent dense operator actions."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer
from pepsy.optimizers.peps._full_update import ReducedPair

pytest.importorskip('torch')
pytestmark = [pytest.mark.optional, pytest.mark.peps]


def fixture(shape=(2, 2), dtype='complex128', seed=71):
    torch = pytest.importorskip('torch')
    state = qtn.PEPO.rand(*shape, bond_dim=2, phys_dim=2, dtype=dtype, seed=seed)
    state.apply_to_arrays(torch.tensor)
    state /= float(torch.linalg.norm(state.to_dense()))
    return state


def unitary(dimension, seed=19):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(size=(dimension, dimension))
                        + 1j*rng.normal(size=(dimension, dimension)))[0]


def dense(state):
    data = state.to_dense(optimize='greedy')
    return data.detach().cpu().numpy() if hasattr(data, 'detach') else np.asarray(data)


def action(state, matrix, gate, where, which):
    """Act on explicit row/column tensor axes; lower applies O @ G.T."""
    sites = list(state.gen_site_coos())
    upper = [state[s].ind_size(state.upper_ind(*s)) for s in sites]
    lower = [state[s].ind_size(state.lower_ind(*s)) for s in sites]
    dimensions = upper + lower
    axes = [sites.index(s) + (len(sites) if which == 'lower' else 0) for s in where]
    sizes = [dimensions[i] for i in axes]
    untouched = [i for i in range(len(dimensions)) if i not in axes]
    out = np.tensordot(gate.reshape(*sizes, *sizes), matrix.reshape(dimensions),
                       axes=(tuple(range(len(axes), 2*len(axes))), tuple(axes)))
    order = axes + untouched
    return out.transpose(np.argsort(order)).reshape(matrix.shape)


def fidelity(a, b):
    return abs(np.vdot(a, b))**2 / (np.vdot(a, a).real*np.vdot(b, b).real)


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('which', ['upper', 'lower', 'both'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cached_svd_start_and_small_core_balance_preserve_operator(monkeypatch, dtype, which, device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = fixture(dtype=dtype)
    state.apply_to_arrays(lambda x: x.to(device))
    state.exponent = .31
    gate = torch.tensor(unitary(4), dtype=getattr(torch, dtype), device=device)*(1.2+.3j)
    payload = (gate, gate.conj()) if which == 'both' else gate
    pair = ReducedPair(state, payload, ((0, 0), (0, 1)), chi=2, which=which)
    assert pair.dims == ((4, 4) if which == 'both' else (2, 2))
    for site, outer in zip(pair.where, (pair.reduced.q_left, pair.reduced.q_right)):
        if which != 'both':
            spectator = getattr(state, ('lower' if which == 'upper' else 'upper')+'_ind')(*site)
            assert spectator in outer.inds
    guess, target = pair.initial_states()
    # A complex normalization phase must update the cached starting factors too.
    pair.normalize_target(2.5+.1j)
    expected = dense(target) / np.sqrt(2.5+.1j)
    norm, _ = pair.environment(guess, chi=64, contraction_opt='greedy',
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    split, balance, cached = pair.split, pair.balance, pair._svd_guess_factors
    pair._svd_guess_factors = None
    pair.balance = lambda a, b: split(torch.einsum('apk,kbq->apbq', a, b), pair.chi)
    reference, _ = pair.optimize(norm, {'solver': 'quimb', 'max_iterations': 3})
    pair.balance, pair._svd_guess_factors = balance, cached
    def forbid_split(*args, **kwargs):
        pytest.fail('Ungauged FU repeated a full-pair SVD after initialization')
    monkeypatch.setattr(pair, 'split', forbid_split)
    output, report = pair.optimize(norm, {'solver': 'quimb', 'max_iterations': 3})
    tol = 5e-5 if dtype == 'complex64' else 3e-10
    assert report['fidelity'] == pytest.approx(fidelity(dense(output), expected), abs=tol)
    np.testing.assert_allclose(dense(output), dense(reference), rtol=tol, atol=tol)
    assert output.max_bond() <= 2
    assert next(iter(output)).data.device.type == device


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_small_core_balance_preserves_product_and_symmetric_singular_values(dtype):
    import torch
    pair = ReducedPair(fixture(dtype=dtype), torch.tensor(unitary(4), dtype=getattr(torch, dtype)),
                       ((0, 0), (0, 1)), chi=2, which='lower')
    rng = np.random.default_rng(72)
    def array(shape):
        return torch.tensor(rng.normal(size=shape)+1j*rng.normal(size=shape), dtype=getattr(torch, dtype))
    left, right = array((pair.na, pair.dims[0], 2)), array((2, pair.nb, pair.dims[1]))
    left *= 1e3
    right *= 1e-3*(1+.3j)
    product = np.einsum('apk,kbq->apbq', left.numpy(), right.numpy())
    a, b = pair.balance(left, right)
    tol = 5e-5 if dtype == 'complex64' else 2e-12
    np.testing.assert_allclose(np.einsum('apk,kbq->apbq', a.numpy(), b.numpy()), product, rtol=tol, atol=tol)
    am, bm = a.reshape(-1, 2).numpy(), b.permute(0, 2, 1).reshape(2, -1).numpy()
    singular = np.linalg.svd(product.transpose(0, 1, 3, 2).reshape(am.shape[0], bm.shape[1]), compute_uv=False)[:2]
    np.testing.assert_allclose(am.conj().T@am, np.diag(singular), rtol=tol, atol=tol)
    np.testing.assert_allclose(bm@bm.conj().T, np.diag(singular), rtol=tol, atol=tol)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_chi6_native_als_uses_dense_local_solver(monkeypatch, device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = qtn.PEPO.rand(2, 2, bond_dim=6, phys_dim=2, dtype='complex128', seed=82)
    state.apply_to_arrays(lambda x: torch.tensor(x, device=device))
    pair = ReducedPair(state, torch.tensor(unitary(4), device=device),
                       ((0, 0), (0, 1)), chi=6, which='lower')
    pair.initial_states(build_target=False)
    fit = qtn.tensor_network_fit_als
    calls = []
    def checked(*args, **kwargs):
        assert kwargs['dense_solve'] is True
        calls.append(True)
        return fit(*args, **kwargs)
    monkeypatch.setattr(qtn, 'tensor_network_fit_als', checked)
    norm = torch.eye(pair.na*pair.nb, dtype=torch.complex128, device=device)
    output, report = pair.optimize(norm, {'solver': 'auto', 'max_iterations': 2})
    assert calls and report['solver'] == 'quimb'
    assert np.isfinite(dense(output)).all()
    assert next(iter(output)).data.device.type == device


def optimizer(state, gates, *, fit_mode='direct', **kwargs):
    return PepsOptimizer(state, gates, chi=2, mode='full-update',
        contraction_opt='greedy', normalize_initial=False, boundary_convergence=False,
        boundary_chi=64, fit_mode=fit_mode, boundary_kwargs={'cutoff': 0., 'n_iter': 2},
        **kwargs)


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('which', ['upper', 'lower'])
@pytest.mark.parametrize('solver', ['qr', 'quimb'])
def test_reduced_pepo_als_matches_dense_frobenius_metric(dtype, which, solver):
    import torch
    state = fixture(dtype=dtype)
    arrays = {s: state[s].data.clone() for s in state.gen_site_coos()}
    gate = unitary(4)
    where = ((0, 1), (0, 0))
    pair = ReducedPair(state, torch.tensor(gate, dtype=getattr(torch, dtype)),
                       where, chi=2, which=which)
    guess, target = pair.initial_states()
    expected = action(state, dense(state), gate, where, which)
    tol = 3e-5 if dtype == 'complex64' else 2e-10
    np.testing.assert_allclose(dense(target), expected, rtol=tol, atol=tol)
    norm, _ = pair.environment(guess, chi=64, contraction_opt='greedy',
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    final, report = pair.optimize(norm, {'solver': solver, 'max_iterations': 8})
    assert report['success'] and report['solver'] == solver
    assert report['norm_clipped_eigenvalues'] >= 0
    assert report['fidelity'] == pytest.approx(fidelity(dense(final), expected), abs=tol)
    assert fidelity(dense(final), expected) >= fidelity(dense(guess), expected)-tol
    assert final.max_bond() <= 2 and isinstance(final, qtn.PEPO)
    for site in state.gen_site_coos():
        np.testing.assert_array_equal(state[site].data.numpy(), arrays[site].numpy())
        assert final[site].inds == state[site].inds
        assert final[site].tags == state[site].tags
        assert final[site].data.dtype == state[site].data.dtype
        assert final[site].data.device == state[site].data.device
        if site not in where:
            assert final[site].data is state[site].data


@pytest.mark.parametrize('which', [None, 'upper', 'lower'])
def test_exact_stream_preserves_operator_scale_leg_names_and_entry_overrides(which):
    import torch
    state = fixture()
    # Both names still follow the boundary k/b convention, but selector
    # routing must use the actual metadata rather than hard-coded prefixes.
    mapping = {}
    for site in state.gen_site_coos():
        mapping[state.upper_ind(*site)] = f'b{site[0]},{site[1]}'
        mapping[state.lower_ind(*site)] = f'k{site[0]},{site[1]}'
    state.reindex_(mapping)
    state = state.view_as(qtn.PEPO, upper_ind_id='b{},{}', lower_ind_id='k{},{}')
    state.exponent = .7
    before = dense(state)
    entries = [(unitary(2), ((0, 0),), 'upper'),
               (unitary(4, 23), ((0, 1), (1, 1)), None),
               (unitary(4, 29), ((1, 0), (0, 0)), 'lower'),
               (unitary(2, 31), ((1, 1),), None)]
    gates = [(torch.tensor(g), w[0] if len(w) == 1 else w, side) for g, w, side in entries]
    opt = PepsOptimizer(state, gates, chi=32, mode='full-update', which=which,
                        contraction_opt='greedy', normalize_initial=False,
                        boundary_convergence=False)
    expected = before
    for _ in range(2):
        for gate, where, side in entries:
            expected = action(state, expected, gate, where, side or which or 'upper')
        output = opt.run(normalize_final=False)
        np.testing.assert_allclose(dense(output), expected, atol=2e-10, rtol=2e-10)
        assert isinstance(output, qtn.PEPO)
        assert output.upper_ind_id == state.upper_ind_id
        assert output.lower_ind_id == state.lower_ind_id
        assert all(r['reason'] == 'exact_svd' for r in opt.get_step_records())
        assert all('Frobenius' in r['local_fidelity_convention'] for r in opt.get_step_records())
    np.testing.assert_array_equal(dense(state), before)


@pytest.mark.parametrize('tensor_mode,solver', [
    ('reduced', 'qr'), ('reduced', 'quimb'), ('reduced', 'lbfgs'), ('full', 'lbfgs')])
def test_driver_truncated_pepo_uses_native_solvers_and_normalized_metrics(tensor_mode, solver):
    import torch
    state = fixture(shape=(2, 3))
    gate, where = unitary(4), ((0, 0), (0, 1))
    expected = action(state, dense(state), gate, where, 'lower')
    opt = optimizer(state, [(torch.tensor(gate), where, 'lower')],
        full_update_kwargs={'tensor_mode': tensor_mode, 'solver': solver,
                            'max_iterations': 8, 'skip_exact': False})
    output = opt.run(normalize_final=True, measure_infidelity=True, measure_final_infidelity=True,
                     accept_if_improved=True)
    record = opt.get_step_records()[0]
    assert record['optimizer_attempted'] and record['which'] == 'lower'
    assert record['optimizer_result']['solver'] == solver
    assert record['optimizer_result']['tensor_mode'] == tensor_mode
    assert record['local_fidelity'] == pytest.approx(fidelity(dense(output), expected), abs=2e-9)
    assert record['post_infidelity'] == pytest.approx(1-fidelity(dense(output), expected), abs=2e-9)
    assert np.linalg.norm(dense(output)) == pytest.approx(1., abs=2e-10)
    assert isinstance(output, qtn.PEPO) and output.max_bond() <= 2


@pytest.mark.parametrize('scope', ['strip', 'layer'])
@pytest.mark.parametrize('warmstart_method', ['compress', 'simple-update'])
def test_pepo_refinement_and_boundary_reuse(scope, warmstart_method):
    import torch
    state = fixture(shape=(2, 3))
    gates = [(torch.tensor(unitary(4, seed)), where, which) for seed, where, which in [
        (19, ((0, 0), (0, 1)), 'upper'), (23, ((0, 1), (0, 2)), 'lower')]]
    opt = optimizer(state, gates, full_update_kwargs={'solver': 'qr', 'max_iterations': 4,
        'skip_exact': False, 'refine_sweeps': 1, 'refine_scope': scope})
    output = opt.run(normalize_final=True, warmstart_method=warmstart_method)
    records = opt.get_step_records()
    assert isinstance(output, qtn.PEPO) and output.max_bond() <= 2
    assert np.isfinite(dense(output)).all()
    assert np.linalg.norm(dense(output)) == pytest.approx(1., abs=2e-9)
    assert any(r[f'{scope}_refinement'] is not None for r in records)
    assert all(r['optimizer_result']['environment_reuse'] is not None for r in records)


@pytest.mark.parametrize('which', ['upper', 'lower'])
def test_pepo_unequal_operator_dimensions_are_retained(which):
    import torch
    state = fixture()
    generator = np.random.default_rng(21)
    for site in state.gen_site_coos():
        t = state[site]
        shape = tuple(3 if i == state.upper_ind(*site) else size for i, size in zip(t.inds, t.shape))
        t.modify(data=torch.tensor(generator.normal(size=shape)+1j*generator.normal(size=shape)))
    gate = unitary(9 if which == 'upper' else 4)
    where = ((0, 0), (1, 0))
    pair = ReducedPair(state, torch.tensor(gate), where, chi=2, which=which)
    _, target = pair.initial_states()
    expected = action(state, dense(state), gate, where, which)
    np.testing.assert_allclose(dense(target), expected, rtol=1e-11, atol=1e-11)
    assert dense(target).shape == (81, 16)
    guess, _ = pair.initial_states()
    norm, _ = pair.environment(guess, chi=64, contraction_opt='greedy',
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    output, report = pair.optimize(norm, {'solver': 'qr', 'max_iterations': 4})
    assert report['fidelity'] == pytest.approx(fidelity(dense(output), expected), abs=2e-9)
    with pytest.raises(ValueError, match='matching size'):
        ReducedPair(state, torch.tensor(unitary(4 if which == 'upper' else 9)),
                    where, chi=2, which=which)


def test_pepo_selector_validation_precedes_normalization_and_peps_stays_strict():
    import torch
    state = fixture()
    before = dense(state)
    opt = optimizer(state, [(torch.eye(4, dtype=torch.complex128), ((0, 0), (0, 1)))],
                    which='both')
    with pytest.raises(ValueError, match='which must'):
        opt.run()
    np.testing.assert_array_equal(dense(opt.state), before)
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, dtype='complex128', seed=7)
    peps.apply_to_arrays(torch.tensor)
    with pytest.raises(ValueError, match='PEPO legs'):
        optimizer(peps, [(torch.eye(4), ((0, 0), (0, 1)), 'upper')]).run()


def test_explicit_one_site_pepo_index_retains_gate_helper_semantics():
    import torch
    state = fixture()
    gate = unitary(2)
    expected = action(state, dense(state), gate, ((0, 0),), 'upper')
    opt = optimizer(state, [(torch.tensor(gate), state.upper_ind(0, 0))], which='lower')
    output = opt.run(normalize_final=False)
    np.testing.assert_allclose(dense(output), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('tensor_mode', ['reduced', 'full'])
def test_pepo_nonunitary_normalization(tensor_mode):
    import torch
    state = fixture()
    opt = optimizer(state, [(2*torch.tensor(unitary(4)), ((0, 0), (0, 1)), 'upper')],
        full_update_kwargs={'tensor_mode': tensor_mode, 'solver': 'lbfgs',
                            'skip_exact': False, 'max_iterations': 4})
    output = opt.run(non_unitary=True, normalize_target=True, normalize_final=True)
    assert isinstance(output, qtn.PEPO)
    assert np.linalg.norm(dense(output)) == pytest.approx(1., abs=2e-10)


@pytest.mark.parametrize('warmstart_method', ['compress', 'simple-update'])
def test_pepo_strip_cache_reuse_matches_uncached_updates(monkeypatch, warmstart_method):
    import torch
    from pepsy.boundary._reuse import StripEnvironmentCache
    state = fixture(shape=(4, 2))
    gates = [(torch.tensor(unitary(4, 19+i)), ((i, 1), (i+1, 1)),
              'upper' if i % 2 else 'lower') for i in range(3)]
    outputs = []
    for reuse in (True, False):
        opt = optimizer(state, gates,
            full_update_kwargs={'solver': 'qr', 'max_iterations': 4, 'skip_exact': False})
        cache = opt._full_update_strip_cache = StripEnvironmentCache()
        if not reuse:
            monkeypatch.setattr(cache, 'reduce', lambda strip, **kw: strip)
        outputs.append(dense(opt.run(normalize_final=False, warmstart_method=warmstart_method)))
        if reuse and warmstart_method == 'compress':
            assert opt.get_step_records()[-1]['optimizer_result']['strip_environment_reuse']['hits'] > 0
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pepo_positive_norm_projection_and_environment_gauges(device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = fixture()
    state.apply_to_arrays(lambda x: x.to(device))
    pair = ReducedPair(state, torch.tensor(unitary(4), device=device), ((0, 0), (0, 1)), chi=2,
                       which='lower')
    norm = torch.eye(pair.na*pair.nb, dtype=torch.complex128, device=device)
    norm[0, 0], norm[-1, -1], norm[0, 1] = -.01, 0., .01j
    output, report = pair.optimize(norm, {'solver': 'qr', 'gauge': True, 'max_iterations': 4})
    assert report['norm_negative_weight'] > 0 and report['norm_clipped_eigenvalues'] > 0
    assert report['norm_antihermitian_relative'] > 0
    assert isinstance(output, qtn.PEPO) and np.isfinite(dense(output)).all()


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('rank_case', ['zero', 'exact', 'discarded'])
def test_cuda_svd_rank_and_discarded_weight_match_cpu(dtype, rank_case):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    state = fixture(dtype=dtype)
    results = []
    for device in ('cpu', 'cuda'):
        local = state.copy()
        local.apply_to_arrays(lambda x: x.to(device))
        pair = ReducedPair(local, torch.tensor(unitary(4), dtype=getattr(torch, dtype), device=device),
                           ((0, 0), (0, 1)), chi=2, which='lower')
        rows, cols = pair.na * pair.dims[0], pair.dims[1] * pair.nb
        matrix = torch.zeros((rows, cols), dtype=getattr(torch, dtype), device=device)
        if rank_case != 'zero':
            eps = 2.**-23 if dtype == 'complex64' else 2.**-52
            values = ((1., .4, 0., 0.) if rank_case == 'exact' else
                      (1., .4, 2*eps*max(rows, cols), .5*eps*max(rows, cols)))
            matrix[torch.arange(4, device=device), torch.arange(4, device=device)] = torch.tensor(
                values, dtype=getattr(torch, dtype), device=device)
        pair.theta = matrix.reshape(pair.na, pair.dims[0], pair.dims[1], pair.nb).permute(0, 1, 3, 2)
        pair.initial_states(build_target=False)
        results.append((pair.target_rank, pair.warmstart_exact, pair.discarded_weight))
    assert results[0][0] == {'zero': 0, 'exact': 2, 'discarded': 3}[rank_case]
    assert results[1][:2] == results[0][:2]
    assert results[1][2] == pytest.approx(results[0][2], rel=1e-6, abs=1e-35)


def test_cuda_deferred_psd_diagnostics_match_scalar_reference():
    import torch
    from pepsy.bp.reduced_update import _psd_project_metric

    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    metric = torch.tensor([[1., .1j, 0.], [.03, -.2, 0.], [0., 0., .4]],
                          dtype=torch.complex128, device='cuda')
    expected = _psd_project_metric(metric, 0., return_spectrum=True)
    actual = _psd_project_metric(metric, 0., return_spectrum=True, defer_diagnostics=True)
    torch.testing.assert_close(actual[0], expected[0])
    assert float(actual[1]) == pytest.approx(expected[1])
    assert int(actual[2]) == expected[2]
    for a, b in zip(actual[3:], expected[3:]):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match='zero floor'):
        _psd_project_metric(metric, .01, defer_diagnostics=True)


@pytest.mark.parametrize('tensor_mode', ['reduced', 'full'])
@pytest.mark.parametrize('normalize_target', [False, True])
def test_operator_scale_and_complex_gate_phase_survive_actual_pair_fit(tensor_mode, normalize_target):
    import torch
    state = fixture()
    state.exponent = .7
    gate, where = (2+.3j)*unitary(4), ((0, 0), (0, 1))
    expected = action(state, dense(state), gate, where, 'lower')
    if normalize_target:
        expected /= np.linalg.norm(expected)
    opt = PepsOptimizer(state, [(torch.tensor(gate), where, 'lower')], chi=16,
        mode='full-update', normalize_initial=False, boundary_convergence=False,
        boundary_chi=64, contraction_opt='greedy', fit_mode='direct',
        full_update_kwargs={'tensor_mode': tensor_mode, 'solver': 'lbfgs',
                            'skip_exact': False, 'max_iterations': 8})
    output = opt.run(non_unitary=True, normalize_target=normalize_target, normalize_final=False)
    assert opt.get_step_records()[0]['optimizer_attempted']
    np.testing.assert_allclose(dense(output), expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize('which', ['upper', 'lower'])
def test_identity_operator_growth_matches_optimal_operator_schmidt_truncation(which):
    import torch
    from pepsy.tensors import id_to_pepo
    state = id_to_pepo((2, 2), dtype='complex128')
    state.apply_to_arrays(torch.tensor)
    gate = unitary(4)
    local = gate if which == 'upper' else gate.T
    matrix = local.reshape(2, 2, 2, 2).transpose(2, 0, 3, 1).reshape(4, 4)
    u, s, vh = np.linalg.svd(matrix)
    best = (u[:, :2]*s[:2]) @ vh[:2]
    best = best.reshape(2, 2, 2, 2).transpose(1, 3, 0, 2).reshape(4, 4)
    expected = np.kron(best, np.eye(4))
    opt = optimizer(state, [(torch.tensor(gate), ((0, 0), (0, 1)), which)],
                    full_update_kwargs={'solver': 'qr', 'max_iterations': 4})
    output = opt.run(normalize_final=False)
    record = opt.get_step_records()[0]
    assert record['optimizer_attempted'] and output.max_bond() == 2
    np.testing.assert_allclose(dense(output), expected, rtol=2e-11, atol=2e-11)
    assert record['local_fidelity'] == pytest.approx(float(np.sum(s[:2]**2)/np.sum(s**2)), abs=1e-11)


@pytest.mark.parametrize('selection', ['constructor', 'constructor-style', 'run-mode', 'run-style'])
@pytest.mark.parametrize('adaptive', [False, True])
def test_pepo_full_update_aliases_and_adaptive_boundaries(selection, adaptive):
    import torch
    state = fixture(shape=(2, 3))
    gate, where = unitary(4), ((0, 0), (0, 1))
    expected = action(state, dense(state), gate, where, 'lower')
    options = {'mode': 'full-update'} if selection == 'constructor' else {'mode': 'sweep'}
    if selection == 'constructor-style':
        options['update_style'] = 'two-site'
    run_options = ({'mode': 'full-update'} if selection == 'run-mode' else
                   {'update_style': 'two-site'} if selection == 'run-style' else {})
    opt = PepsOptimizer(state, [(torch.tensor(gate), where, 'lower')], chi=2,
        normalize_initial=False, contraction_opt='greedy', fit_mode='direct',
        boundary_kwargs={'cutoff': 0., 'n_iter': 2}, boundary_chi=16,
        boundary_convergence={'max_chi': 64} if adaptive else False,
        full_update_kwargs={'solver': 'qr', 'max_iterations': 4, 'skip_exact': False}, **options)
    output = opt.run(normalize_final=False, **run_options)
    record = opt.get_step_records()[0]
    assert record['update_style'] == 'two-site'
    assert record['local_fidelity'] == pytest.approx(fidelity(dense(output), expected), abs=2e-9)
    assert record['boundary_convergence']['converged'] if adaptive else record['boundary_convergence'] is None


@pytest.mark.parametrize('which', [None, 'lower'])
def test_smart_pepo_order_and_single_site_fusion_match_dense_stream(which):
    import torch
    state = fixture()
    expected = dense(state)
    entries = [(unitary(2, 7), ((0, 0),), None),
               (unitary(2, 8), ((0, 0),), None),
               (unitary(4), ((0, 0), (0, 1)), None),
               (unitary(2, 9), ((1, 0),), 'lower'),
               (unitary(4, 22), ((1, 0), (1, 1)), 'lower')]
    gates = []
    for gate, where, side in entries:
        expected = action(state, expected, gate, where, side or which or 'upper')
        gates.append((torch.tensor(gate), where[0] if len(where) == 1 else where, side))
    opt = PepsOptimizer(state, gates, chi=32, mode='full-update', which=which,
        gate_order='smart', normalize_initial=False, boundary_convergence=False,
        contraction_opt='greedy')
    output = opt.run(normalize_final=False)
    np.testing.assert_allclose(dense(output), expected, rtol=2e-11, atol=2e-11)
    assert opt.last_gate_order['original_step_groups'] == [[1, 2], [3], [4], [5]]
    assert len(opt.gates) == len(gates)


def test_pepo_inplace_mutation_invalidates_cached_transverse_environment():
    import torch
    state = fixture(shape=(2, 3))
    gates = [(torch.tensor(unitary(4)), ((0, 0), (0, 1)), 'lower')]
    controls = {'solver': 'qr', 'max_iterations': 4, 'skip_exact': False}
    cached = optimizer(state, gates, full_update_kwargs=controls)
    cached.run(normalize_final=False)
    tensor = cached.state[1, 0]
    shape = [1]*tensor.ndim
    shape[tensor.inds.index(cached.state.upper_ind(1, 0))] = 2
    tensor.data.mul_(torch.tensor([.7, 1.3]).reshape(shape))
    fresh = optimizer(cached.state, gates, full_update_kwargs=controls)
    a = dense(cached.run(normalize_final=False))
    b = dense(fresh.run(normalize_final=False))
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-11)


def test_periodic_pepo_interior_pair_matches_dense_frobenius_metric():
    import torch
    state = qtn.PEPO.rand(3, 3, bond_dim=2, phys_dim=2, dtype='complex128', cyclic=True, seed=7)
    state.apply_to_arrays(torch.tensor)
    state /= float(torch.linalg.norm(state.to_dense(optimize='greedy')))
    where, gate = ((0, 0), (0, 1)), unitary(4)
    pair = ReducedPair(state, torch.tensor(gate), where, chi=2)
    guess, _ = pair.initial_states()
    target = action(state, dense(state), gate, where, 'upper')
    norm, _ = pair.environment(guess, chi=64, contraction_opt='greedy',
                              boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.})
    output, report = pair.optimize(norm, {'solver': 'qr', 'max_iterations': 4})
    assert report['fidelity'] == pytest.approx(fidelity(dense(output), target), abs=2e-9)
    # The existing coordinate-adjacency contract does not include wrap bonds.
    with pytest.raises(ValueError, match='nearest-neighbor'):
        ReducedPair(state, torch.tensor(gate), ((0, 0), (0, 2)), chi=2)


@pytest.mark.parametrize('backend', ['torch-cuda', 'cupy'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('which', ['upper', 'lower'])
@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
@pytest.mark.parametrize('warmstart_method', ['compress', 'simple-update'])
def test_accelerator_pepo_als_keeps_arrays_on_device(
    monkeypatch, backend, dtype, which, fit_mode, warmstart_method,
):
    import autoray as ar
    import torch
    state = fixture(dtype=dtype)
    before = dense(state)
    native = state.copy()
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip('CuPy needs a CUDA device')
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(str(exc))
        convert = lambda x: cp.asarray(x.detach().numpy() if hasattr(x, 'detach') else x)
    else:
        if not torch.cuda.is_available():
            pytest.skip('Torch CUDA unavailable')
        convert = lambda x: torch.as_tensor(x, dtype=getattr(torch, dtype), device='cuda')
    native.apply_to_arrays(convert)
    gate, where = unitary(4), ((0, 0), (0, 1))
    expected = action(state, before, gate, where, which)
    opt = optimizer(native, [(convert(gate.astype(dtype)), where, which)], fit_mode=fit_mode,
                    full_update_kwargs={'solver': 'qr', 'max_iterations': 4, 'skip_exact': False})
    original = ar.to_numpy
    def scalar_only(value, *args, **kwargs):
        if backend == 'cupy' and isinstance(value, cp.ndarray):
            assert value.size == 1, 'PEPO array transferred to host'
        if isinstance(value, torch.Tensor) and value.is_cuda:
            assert value.numel() == 1, 'PEPO tensor transferred to host'
        return original(value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(ar, 'to_numpy', scalar_only)
        output = opt.run(normalize_final=False, warmstart_method=warmstart_method)
    assert isinstance(output, qtn.PEPO)
    sample = next(iter(native)).data
    assert all(t.data.dtype == sample.dtype and t.data.device == sample.device for t in output)
    actual = (cp.asnumpy(output.to_dense()) if backend == 'cupy'
              else output.to_dense().detach().cpu().numpy())
    tol = 5e-5 if dtype == 'complex64' else 2e-9
    assert opt.get_step_records()[0]['local_fidelity'] == pytest.approx(
        fidelity(actual, expected), abs=tol)


@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
@pytest.mark.parametrize('kind', ['pauli', 'projector'])
@pytest.mark.parametrize('which', ['upper', 'lower'])
@pytest.mark.parametrize('normalize_output', [False, True])
@pytest.mark.parametrize('warmstart_method', ['compress', 'simple-update'])
def test_arbitrary_operator_norms_are_measured_and_scale_is_preserved(
    monkeypatch, fit_mode, kind, which, normalize_output, warmstart_method,
):
    import torch
    from pepsy.tensors import id_to_pepo
    from pepsy.boundary.metrics import _as_scaled_scalar
    from pepsy.optimizers.peps import optimizer as module
    state = id_to_pepo((2, 2), dtype='complex128')
    state.apply_to_arrays(torch.tensor)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    factors = [x if kind == 'pauli' else np.diag([1, 0]).astype(complex),
               y, z, np.eye(2, dtype=complex)]
    for site, factor in zip(state.gen_site_coos(), factors):
        state = qtn.tensor_network_gate_inds(state, torch.tensor(factor),
                    (state.upper_ind(*site),), contract=True)
    scale = 2+.3j
    state[0, 0].modify(data=scale*state[0, 0].data)
    source = scale*np.kron(np.kron(np.kron(*factors[:2]), factors[2]), factors[3])
    np.testing.assert_allclose(dense(state), source, rtol=1e-12, atol=1e-12)
    gate, where = unitary(4), ((0, 0), (0, 1))
    target = action(state, source, gate, where, which)
    expected_norm = float(np.vdot(target, target).real)
    assert expected_norm == pytest.approx((16 if kind == 'pauli' else 8)*abs(scale)**2)
    measured_norms = []
    original = module.boundary_infidelity
    def measured(*args, **kwargs):
        assert kwargs.get('norm_target') is None
        result = original(*args, **kwargs)
        mantissa, exponent = _as_scaled_scalar(result['norm_target'])
        measured_norms.append(complex(mantissa)*10**exponent)
        return result
    monkeypatch.setattr(module, 'boundary_infidelity', measured)
    local = gate@np.kron(*factors[:2]) if which == 'upper' else np.kron(*factors[:2])@gate.T
    matrix = local.reshape(2, 2, 2, 2).transpose(2, 0, 3, 1).reshape(4, 4)
    u, s, vh = np.linalg.svd(matrix)
    best = ((u[:, :2]*s[:2])@vh[:2]).reshape(2, 2, 2, 2).transpose(1, 3, 0, 2).reshape(4, 4)
    expected = scale*np.kron(best, np.kron(*factors[2:]))
    if normalize_output:
        expected /= np.linalg.norm(expected)
    # No initial/final normalization flag is supplied for the default case.
    opt = PepsOptimizer(state, [(torch.tensor(gate), where, which)], chi=2,
        mode='full-update', fit_mode=fit_mode, contraction_opt='greedy',
        boundary_convergence=False, boundary_chi=64, boundary_kwargs={'cutoff': 0., 'n_iter': 2},
        full_update_kwargs={'solver': 'qr', 'max_iterations': 4, 'skip_exact': False})
    assert opt.normalize_initial is False
    output = opt.run(measure_infidelity=True, measure_final_infidelity=True,
                     warmstart_method=warmstart_method,
                     **({'normalize_final': True} if normalize_output else {}))
    assert measured_norms
    np.testing.assert_allclose(measured_norms, expected_norm, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(dense(output), expected, rtol=2e-10, atol=2e-10)
    record = opt.get_step_records()[0]
    assert record['local_fidelity'] == pytest.approx(fidelity(dense(output), target), abs=2e-10)
    assert record['post_infidelity'] == pytest.approx(1-fidelity(dense(output), target), abs=2e-10)
    assert len(opt.get_normalizations()) == (1 if normalize_output else 0)


@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_scaled_entangled_pepo_uses_measured_frobenius_norm(monkeypatch, fit_mode):
    import torch
    from pepsy.boundary.sweeps import CompBdy
    calls = {'direct': 0, 'dmrg': 0}
    compress, solve = CompBdy._compress_boundary, CompBdy._run_fit_solver
    def track_direct(self, *args, **kwargs):
        calls['direct'] += 1
        return compress(self, *args, **kwargs)
    def track_dmrg(self, *args, **kwargs):
        calls['dmrg'] += 1
        return solve(self, *args, **kwargs)
    monkeypatch.setattr(CompBdy, '_compress_boundary', track_direct)
    monkeypatch.setattr(CompBdy, '_run_fit_solver', track_dmrg)
    state = fixture(shape=(2, 3))
    state[0, 0].modify(data=(3.7+1.2j)*state[0, 0].data)
    source = dense(state)
    gate, where = unitary(4), ((0, 0), (0, 1))
    target = action(state, source, gate, where, 'upper')
    opt = PepsOptimizer(state, [(torch.tensor(gate), where)], chi=2,
        mode='full-update', fit_mode=fit_mode, contraction_opt='greedy',
        boundary_convergence=False, boundary_chi=64, boundary_kwargs={'cutoff': 0., 'n_iter': 2},
        full_update_kwargs={'solver': 'qr', 'max_iterations': 8, 'skip_exact': False})
    output = opt.run(measure_infidelity=True, measure_final_infidelity=True)
    assert calls[fit_mode] > 0
    assert calls['dmrg' if fit_mode == 'direct' else 'direct'] == 0
    assert not opt.get_normalizations()
    assert np.linalg.norm(dense(output)) > 2.
    record = opt.get_step_records()[0]
    assert record['post_infidelity'] == pytest.approx(1-fidelity(dense(output), target), abs=2e-9)


@pytest.mark.parametrize('normalize_initial', [None, False, True])
def test_automatic_normalization_tracks_replacement_state_kind(normalize_initial):
    import torch
    peps = qtn.PEPS.rand(2, 2, bond_dim=1, dtype='complex128', seed=71)
    peps.apply_to_arrays(torch.tensor)
    op = fixture()
    opt = PepsOptimizer(peps, [], chi=2, normalize_initial=normalize_initial)
    assert opt.normalize_initial == (True if normalize_initial is None else normalize_initial)
    opt.set_state(op)
    assert opt.normalize_initial == (False if normalize_initial is None else normalize_initial)
    opt.set_state(peps)
    assert opt.normalize_initial == (True if normalize_initial is None else normalize_initial)


@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_truncated_boundary_pepo_update_is_homogeneous_in_operator_scale(fit_mode):
    import torch
    state = fixture(shape=(3, 3))
    gate, where = unitary(4), ((1, 0), (1, 1))
    scale = 2.7-.6j
    outputs, fidelities = [], []
    for factor in (1., scale):
        source = state.copy()
        source[0, 0].modify(data=factor*source[0, 0].data)
        opt = PepsOptimizer(source, [(torch.tensor(gate), where)], chi=2,
            mode='full-update', fit_mode=fit_mode, contraction_opt='greedy',
            boundary_convergence=False, boundary_chi=2,
            boundary_kwargs={'cutoff': 0., 'n_iter': 2},
            full_update_kwargs={'solver': 'qr', 'max_iterations': 4, 'skip_exact': False})
        outputs.append(dense(opt.run()))
        fidelities.append(opt.get_step_records()[0]['local_fidelity'])
        assert not opt.get_normalizations()
    # The scale is outside the active pair and must survive boundary compression.
    np.testing.assert_allclose(outputs[1], scale*outputs[0], rtol=2e-8, atol=2e-10)
    assert fidelities[1] == pytest.approx(fidelities[0], abs=2e-9)


@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
@pytest.mark.parametrize('which', ['upper', 'lower'])
@pytest.mark.parametrize('warmstart_method', [None, 'simple-update'])
def test_default_pepo_update_never_measures_standalone_norm_or_fidelity(
    monkeypatch, fit_mode, which, warmstart_method,
):
    import torch
    from pepsy.optimizers.peps import optimizer as module
    def unexpected(*args, **kwargs):
        pytest.fail('default PEPO full update requested an independent norm/fidelity contraction')
    for name in ('boundary_norm', 'boundary_normalize', 'boundary_infidelity'):
        monkeypatch.setattr(module, name, unexpected)
    state = fixture(shape=(2, 3))
    state[0, 0].modify(data=(2+.7j)*state[0, 0].data)
    opt = PepsOptimizer(state, [(torch.tensor(unitary(4)), ((0, 0), (0, 1)), which)],
        chi=2, mode='full-update', fit_mode=fit_mode, contraction_opt='greedy',
        boundary_kwargs={'n_iter': 2}, full_update_kwargs={'max_iterations': 3, 'skip_exact': False})
    output = opt.run(warmstart_method=warmstart_method)
    assert isinstance(output, qtn.PEPO)
    assert not opt.get_normalizations() and not opt.get_evaluation_records()
    record = opt.get_step_records()[0]
    assert record['optimizer_attempted']
    assert record['pre_infidelity'] is None and record['post_infidelity'] is None


@pytest.mark.parametrize('kind,which', [('peps', None), ('pepo', 'upper'), ('pepo', 'lower')])
@pytest.mark.parametrize('tensor_mode,solver,gauge', [
    ('reduced', 'qr', False), ('reduced', 'quimb', False), ('reduced', 'qr', True),
    ('reduced', 'lbfgs', False), ('full', 'lbfgs', False)])
@pytest.mark.parametrize('fit_mode', ['direct', 'dmrg'])
def test_simple_update_warm_start_reaches_solver_and_preserves_exact_target(
    kind, which, tensor_mode, solver, gauge, fit_mode,
):
    import torch
    from pepsy.operators import gate_simple
    state = fixture(shape=(2, 3)) if kind == 'pepo' else qtn.PEPS.rand(
        2, 3, bond_dim=2, dtype='complex128', seed=71)
    if kind == 'peps':
        state.apply_to_arrays(torch.tensor)
    state[0, 0].modify(data=(2+.7j)*state[0, 0].data)
    source = dense(state)
    arrays = {site: state[site].data.clone() for site in state.gen_site_coos()}
    gate, where = torch.tensor(unitary(4)), ((0, 0), (0, 1))
    expected = state.copy()
    gauges = {}
    expected.gauge_all_simple_(gauges=gauges)
    gate_simple(expected, gate, where, which=which, gauges=gauges,
                max_bond=2, cutoff=0., renorm=False)
    expected.gauge_simple_insert(gauges)
    opt = optimizer(state, [(gate, where, which)], fit_mode=fit_mode,
        full_update_kwargs={'tensor_mode': tensor_mode, 'solver': solver, 'gauge': gauge,
                            'max_iterations': 8, 'skip_exact': False})
    pair, guess, target = opt._full_update_pair(gate, where, which,
                                               warmstart_method='simple-update')
    target_dense = (action(state, source, gate.numpy(), where, which) if kind == 'pepo'
                    else state.gate(gate, where, contract='split', cutoff=0.).to_dense().numpy())
    np.testing.assert_allclose(dense(target), target_dense, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(dense(guess), dense(expected), rtol=2e-9, atol=2e-9)
    output = opt.run(warmstart_method='simple-update', cutoff=0., normalize_final=False)
    record = opt.get_step_records()[0]
    assert record['warmstart_method'] == 'simple-update'
    report = record['optimizer_result']
    assert report['warmstart_fidelity'] == pytest.approx(
        fidelity(dense(expected), target_dense), abs=2e-9)
    assert report['fidelity'] == pytest.approx(fidelity(dense(output), target_dense), abs=2e-9)
    assert fidelity(dense(output), target_dense) >= fidelity(dense(expected), target_dense)-2e-9
    if solver in ('qr', 'quimb'):
        assert report['solver_costs'][0] == pytest.approx(report['warmstart_loss'], abs=2e-9)
    assert type(output) is type(state)
    for site in state.gen_site_coos():
        np.testing.assert_array_equal(state[site].data.numpy(), arrays[site].numpy())
        assert output[site].inds == state[site].inds
        assert output[site].data.dtype == state[site].data.dtype
        assert output[site].data.device == state[site].data.device


@pytest.mark.parametrize('which', ['upper', 'lower'])
@pytest.mark.parametrize('normalize_target', [False, True])
def test_su_warm_start_preserves_custom_unequal_legs_and_normalization(which, normalize_target):
    import torch
    state = fixture()
    generator = np.random.default_rng(21)
    for site in state.gen_site_coos():
        t = state[site]
        shape = tuple(3 if i == state.upper_ind(*site) else size for i, size in zip(t.inds, t.shape))
        t.modify(data=torch.tensor(generator.normal(size=shape)+1j*generator.normal(size=shape)))
    mapping = {state.upper_ind(*s): f'out{s[0]}_{s[1]}' for s in state.gen_site_coos()}
    mapping.update({state.lower_ind(*s): f'in{s[0]}_{s[1]}' for s in state.gen_site_coos()})
    state.reindex_(mapping)
    state = state.view_as(qtn.PEPO, upper_ind_id='out{}_{}', lower_ind_id='in{}_{}')
    state.exponent = .7
    gate = (1.3+.2j)*unitary(9 if which == 'upper' else 4)
    where = ((1, 0), (0, 0))
    target = action(state, dense(state), gate, where, which)
    if normalize_target:
        target /= np.linalg.norm(target)
    opt = optimizer(state, [(torch.tensor(gate), where, which)],
                    which='lower' if which == 'upper' else 'upper',
                    full_update_kwargs={'solver': 'quimb', 'max_iterations': 6, 'skip_exact': False})
    output = opt.run(warmstart_method='simple-update', non_unitary=True,
                     normalize_target=normalize_target, measure_infidelity=True,
                     measure_final_infidelity=True)
    report = opt.get_step_records()[0]
    assert dense(output).shape == (81, 16)
    assert output.upper_ind_id == state.upper_ind_id and output.lower_ind_id == state.lower_ind_id
    assert report['local_fidelity'] == pytest.approx(fidelity(dense(output), target), abs=2e-9)
    assert report['post_infidelity'] == pytest.approx(1-fidelity(dense(output), target), abs=2e-9)
