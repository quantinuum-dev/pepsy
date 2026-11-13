"""Protected marginals, independent block references and local DMT evolution."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import PauliDMTReport, PauliMPO
from pepsy.optimizers import PauliDMT


def _density(n=6, seed=31):
    rng = np.random.default_rng(seed)
    # Triangle inequality bounds the perturbation norm below the identity.
    words = rng.choice(list('IXYZ'), size=(60, n))
    terms = [(1 / 2**n, 'I' * n), (.02 / 2**n, 'ZZ'), (.01 / 2**n, 'X')]
    terms += [(float(rng.uniform(-.005, .005)) / 2**n, ''.join(w)) for w in words]
    return PauliMPO.from_terms(n, terms).compress_pauli(cutoff=0.0)


def _marginals(op):
    size = min(3, op.nsites)
    return [op.partial_trace(range(j, j + size), keep=True).to_dense()
            for j in range(op.nsites - size + 1)]


def _assert_preserved(a, b, atol=3e-13):
    for x, y in zip(_marginals(a), _marginals(b)):
        np.testing.assert_allclose(x, y, rtol=atol, atol=atol)


@pytest.mark.parametrize('variant', ['density', 'operator'])
def test_actual_compression_preserves_all_local_marginals_and_energy(variant):
    rho = _density()
    before = rho.to_dense()
    out, report = rho.compress_dmt(max_bond=10, variant=variant, return_report=True)
    assert isinstance(report, PauliDMTReport)
    assert max(out.pauli_bond_dimensions) <= 10
    assert any(r.input_rank > r.final_bond for r in report.bond_reports)
    assert np.linalg.norm(out.to_dense() - before) > 1e-6
    _assert_preserved(rho, out)
    h = PauliMPO.from_terms(6, [(1., 'ZZ'), (.7, 'X'), (.2, 'XYZ')])
    np.testing.assert_allclose(h.inner(out), h.inner(rho), atol=1e-13)
    np.testing.assert_allclose(out.trace(), rho.trace(), atol=1e-13)
    np.testing.assert_allclose(rho.to_dense(), before, atol=0.)
    assert report.max_constraint_error < 1e-13
    assert all(r.support_discarded_weight < 1e-13 for r in report.bond_reports)


def test_residual_cutoff_reduces_rank_without_cutting_local_information():
    rho = _density()
    out, report = rho.compress_dmt(max_bond=64, cutoff=.8, return_report=True)
    _assert_preserved(rho, out)
    assert any(r.discarded_weight > 1e-6 for r in report.bond_reports)


def test_operator_variant_preserves_traceless_partial_operators():
    rho = _density()
    observable = rho - PauliMPO.identity(6, coefficient=rho.trace() / 64)
    out = observable.compress_dmt(max_bond=10, variant='operator')
    _assert_preserved(observable, out)
    assert abs(out.trace()) < 1e-13
    with pytest.raises(ValueError, match='trace'):
        observable.compress_dmt(max_bond=10, variant='density')


@pytest.mark.parametrize('word', ['IIIIII', 'ZIIIII', 'ZIIIIZ'])
def test_dependent_and_zero_identity_environments(word):
    op = PauliMPO.from_terms(6, [(1., word)])
    out = op.compress_dmt(max_bond=1, variant='operator')
    np.testing.assert_allclose(out.to_dense(), op.to_dense(), atol=1e-13)
    zero = PauliMPO.zero(6).compress_dmt(max_bond=1, variant='operator')
    assert np.linalg.norm(zero.to_dense()) == 0


def test_failure_is_explicit_and_inplace_is_transactional():
    rho = _density()
    before = rho.to_dense()
    with pytest.raises(ValueError, match='protected budget'):
        rho.compress_dmt(max_bond=2, inplace=True)
    np.testing.assert_array_equal(rho.to_dense(), before)
    out, report = rho.compress_dmt(max_bond=12, inplace=True, return_report=True)
    assert out is rho and rho.compression_report is report


@pytest.mark.parametrize('kwargs', [
    {'max_bond': True}, {'max_bond': 0}, {'max_bond': 2.5},
    {'max_bond': 10, 'cutoff': -1.}, {'max_bond': 10, 'cutoff': float('nan')},
    {'max_bond': 10, 'variant': 'automatic'},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        _density(4).compress_dmt(**kwargs)


def _explicit_block(m, lv, rv, cap, variant):
    """Independent dense complete-basis implementation of the paper's rule."""
    from pepsy.operators._pauli_dmt import _basis
    ql = np.asarray(_basis(lv, 1e-14))
    qr = np.asarray(_basis(rv, 1e-14))
    # Complete the bases with NumPy QR; only the unprotected block is truncated.
    ql = np.linalg.qr(ql, mode='complete')[0]
    qr = np.linalg.qr(qr, mode='complete')[0]
    ml = ql.conj().T @ m @ qr
    p, q = lv.shape[1], rv.shape[1]
    if variant == 'density':
        a, b = ql.conj().T @ lv[:, :1], qr.conj().T @ rv[:, :1]
        baseline = ((ml @ b) @ (a.conj().T @ ml)) / (a.conj().T @ ml @ b)
    else:
        baseline = np.zeros_like(ml)
    c = ml - baseline
    u, s, vh = np.linalg.svd(c[p:, q:], full_matrices=False)
    rank = cap - p - q
    c[p:, q:] = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return ql @ (c + baseline) @ qr.conj().T


@pytest.mark.parametrize('variant', ['density', 'operator'])
def test_compact_kernel_agrees_with_complete_basis_reference(variant):
    from pepsy.operators._pauli_dmt import _split
    rng = np.random.default_rng(9)
    # Already diagonal Schmidt matrix: left/right support bases are identity.
    m = np.diag(np.linspace(2., .1, 16)).astype(complex)
    lv = rng.normal(size=(16, 4)) + 1j * rng.normal(size=(16, 4))
    rv = rng.normal(size=(16, 4)) + 1j * rng.normal(size=(16, 4))
    expected = _explicit_block(m, lv, rv, 11, variant)
    left, right, report = _split(
        m, lv, rv, max_bond=11, cutoff=0., variant=variant, absorb='right',
        bond=0, diagnostics=True,
    )
    np.testing.assert_allclose(left @ right, expected, atol=3e-13)
    assert report.final_bond <= 11


def test_small_but_resolved_physical_constraint_is_not_discarded():
    from pepsy.operators._pauli_dmt import _split
    m = np.diag([.99, .9, .05, .8, .7, .6, .5, .4, .3, .2, .1, 1e-6])
    eye = np.eye(12)
    lv = np.column_stack((eye[:, 0], eye[:, 1],
                          eye[:, 1] + 1e-10 * eye[:, 2], eye[:, 3]))
    rv = eye[:, [0, 4, 5, 6]]
    left, right, report = _split(
        m, lv, rv, max_bond=9, cutoff=0., variant='operator', absorb='right',
        bond=0, diagnostics=True,
    )
    delta = left @ right - m
    assert report.protected_left_rank == 4
    np.testing.assert_allclose(lv.T @ delta, 0., atol=1e-14)
    np.testing.assert_allclose(delta @ rv, 0., atol=1e-14)


def _local_gate():
    local = PauliMPO(2, [(1., 'ZZ'), (.7, 'XI'), (.2, 'IY')]).to_dense()
    return expm(-.17j * local)


@pytest.mark.parametrize('mode', ['conjugate', 'heisenberg'])
def test_local_gate_and_dmt_match_dense_target_marginals(mode):
    rho = _density()
    gate = _local_gate()
    full = np.kron(np.kron(np.eye(4), gate), np.eye(4))
    matrix = rho.to_dense()
    expected = full @ matrix @ full.conj().T if mode == 'conjugate' else full.conj().T @ matrix @ full
    # Convert only the small test oracle, independently of production gates.
    target = PauliMPO.from_dense(expected, max_qubits=6, atol=0.)
    out = rho.apply_gate_dmt(gate, (2, 3), max_bond=10, mode=mode)
    _assert_preserved(target, out)
    exact = rho.apply_gate_dmt(gate, (2, 3), max_bond=64, mode=mode)
    np.testing.assert_allclose(exact.to_dense(), expected, atol=3e-13)


def test_no_word_expansion_and_no_embedded_global_gate(monkeypatch):
    rho = _density()
    def forbidden(*args, **kwargs):
        raise AssertionError('word/dense/global gate path reached')
    monkeypatch.setattr(PauliMPO, '_materialize_core_terms', forbidden)
    monkeypatch.setattr(PauliMPO, 'to_dense', forbidden)
    monkeypatch.setattr(PauliMPO, 'product', forbidden)
    out = rho.compress_dmt(max_bond=10).apply_gate_dmt(_local_gate_matrix(), (1, 2), max_bond=10)
    assert out.nsites == 6


def _local_gate_matrix():
    return np.diag([1., 1j, -1j, 1.])


def test_workspace_reuses_preparation_and_bounds_report_history():
    rho = _density()
    driver = PauliDMT(rho, max_bond=10, diagnostics=True)
    # Caches must be invalidated correctly for both sweep directions and jumps.
    expected = rho
    for bond in (0, 2, 4, 3, 1, 0, 4):
        expected = expected.apply_gate_dmt(_local_gate_matrix(), (bond, bond + 1), max_bond=10)
        driver.apply_gate(_local_gate_matrix(), (bond, bond + 1))
        np.testing.assert_allclose(driver.state.to_dense(), expected.to_dense(), atol=3e-12)
        assert len(driver.report.bond_reports) == 1
    state = driver.state
    state._cores[0][...] = 0.
    assert np.linalg.norm(driver.state.to_dense()) > 0


def _hamiltonians(n):
    x = np.array([[0, 1], [1, 0]], dtype=float)
    z = np.diag([1., -1.])
    eye = np.eye(2)
    return [np.kron(z, z) + (.4 if j == 0 else .2) * np.kron(x, eye)
            + (.4 if j == n - 2 else .2) * np.kron(eye, x) for j in range(n - 1)]


@pytest.mark.parametrize('mode', ['conjugate', 'heisenberg'])
def test_evolution_dense_reference_and_second_order_convergence(mode):
    n = 4
    rho = _density(n)
    hs = _hamiltonians(n)
    h = sum(np.kron(np.kron(np.eye(2**j), h), np.eye(2**(n-j-2))) for j, h in enumerate(hs))
    u = expm(-.2j * h)
    expected = u @ rho.to_dense() @ u.conj().T if mode == 'conjugate' else u.conj().T @ rho.to_dense() @ u
    errors = []
    for steps in (2, 4):
        driver = PauliDMT(rho, max_bond=16, mode=mode)
        out = driver.evolve(hs, dt=.2 / steps, steps=steps)
        errors.append(np.linalg.norm(out.to_dense() - expected))
        assert driver.time == pytest.approx(.2)
        assert len(driver.report.bond_reports) <= 2 * len(hs)
    assert errors[1] < errors[0] / 3.8


def test_truncated_evolution_energy_drift_is_timestep_controlled():
    rho = _density()
    h = PauliMPO.from_terms(6, [(1., 'ZZ'), (.4, 'X')])
    initial_energy = h.inner(rho).real
    errors = []
    for dt in (.05, .025):
        driver = PauliDMT(rho, max_bond=10, diagnostics=True)
        out = driver.evolve(_hamiltonians(6), dt=dt, steps=round(.2 / dt))
        errors.append(abs(h.inner(out).real - initial_energy))
        assert max(out.pauli_bond_dimensions) <= 10
        assert abs(out.trace() - rho.trace()) < 1e-12
        assert driver.report.max_constraint_error < 1e-12
    assert errors[1] < errors[0] / 3.5


@pytest.mark.parametrize('backend', ['numpy', 'torch', 'jax'])
@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
def test_dense_backends_preserve_dtype_and_local_data(backend, dtype, request):
    rho = _density(5)
    cores = [np.asarray(a.real if dtype == 'float32' else a, dtype=dtype)
             for a in rho.to_pauli_cores()]
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        cores = [torch.tensor(a) for a in cores]
        host = lambda a: a.detach().cpu().numpy()
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        previous_x64 = jax.config.x64_enabled
        request.addfinalizer(lambda: jax.config.update('jax_enable_x64', previous_x64))
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        cores = [jnp.asarray(a) for a in cores]
        host = np.asarray
    else:
        host = np.asarray
    op = PauliMPO.from_pauli_cores(cores)
    out = op.apply_gate_dmt(_local_gate_matrix(), (1, 2), max_bond=10)
    assert str(out.to_pauli_cores()[0].dtype).split('.')[-1] == dtype
    dense = host(out.to_dense())
    baseline = PauliMPO.from_pauli_cores([host(a) for a in cores])
    target = baseline.apply_gate_dmt(_local_gate_matrix(), (1, 2), max_bond=64)
    tolerance = 5e-6 if dtype == 'float32' else 5e-13
    for a, b in zip(_marginals(PauliMPO.from_dense(dense, max_qubits=5)), _marginals(target)):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)
    # Exercise native EIGH/exponential gate preparation as well as explicit gates.
    evolved = PauliDMT(op, max_bond=10).evolve(_hamiltonians(5), dt=.03, steps=2)
    reference = PauliDMT(baseline, max_bond=10).evolve(_hamiltonians(5), dt=.03, steps=2)
    np.testing.assert_allclose(host(evolved.to_dense()), reference.to_dense(),
                               atol=tolerance, rtol=tolerance)


def test_public_namespace_exports():
    from pepsy.operators import PauliDMTBondReport
    assert PauliDMTBondReport.__name__ == 'PauliDMTBondReport'


def test_identity_with_cutoff_and_single_site_density():
    for n in (1, 2, 6):
        op = PauliMPO.identity(n, coefficient=1 / 2**n)
        out = op.compress_dmt(max_bond=1, cutoff=1e-10)
        np.testing.assert_allclose(out.to_dense(), op.to_dense(), atol=1e-14)


@pytest.mark.parametrize('variant', ['density', 'operator'])
def test_complex_virtual_gauges_preserve_the_same_physical_result(variant):
    # Use resolved Schmidt/constraint spectra: an approximation through
    # numerically singular supports does not define a unique global output.
    rng = np.random.default_rng(271)
    x = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    dense = x @ x.conj().T
    dense /= np.trace(dense)
    rho = PauliMPO.from_dense(dense, max_qubits=5).compress_pauli(cutoff=0.)
    gauged = _complex_gauge(rho)
    reference = rho.compress_dmt(max_bond=10, variant=variant)
    out = gauged.compress_dmt(max_bond=10, variant=variant)
    np.testing.assert_allclose(gauged.to_dense(), rho.to_dense(), atol=1e-13)
    np.testing.assert_allclose(out.to_dense(), reference.to_dense(), atol=2e-12)


def _complex_gauge(rho):
    rng = np.random.default_rng(171)
    cores = list(rho.to_pauli_cores())
    for j in range(len(cores) - 1):
        rank = cores[j].shape[1]
        q = np.linalg.qr(rng.normal(size=(rank, rank))
                         + 1j * rng.normal(size=(rank, rank)))[0]
        cores[j] = np.einsum('abp,bc->acp', cores[j], q)
        cores[j+1] = np.einsum('ab,bcp->acp', q.conj().T, cores[j+1])
    return PauliMPO.from_pauli_cores(cores)


@pytest.mark.parametrize('variant', ['density', 'operator'])
def test_near_dependent_complex_gauge_still_preserves_local_data(variant):
    # This sparse source develops very small Schmidt values and dependent
    # constraints. Retain the local guarantee even when the selected global
    # approximation is sensitive to the rank decisions at floating precision.
    rho = _complex_gauge(_density())
    out, report = rho.compress_dmt(max_bond=10, variant=variant, return_report=True)
    _assert_preserved(rho, out)
    assert report.max_constraint_error < 1e-13


def test_existing_bond_compression_avoids_enlarged_pair_svds(monkeypatch):
    import pepsy.operators._pauli_dmt as dmt
    rng = np.random.default_rng(111)
    cores = [rng.normal(size=(1, 12, 4))]
    cores += [rng.normal(size=(12, 12, 4)) for _ in range(6)]
    cores += [rng.normal(size=(12, 1, 4))]
    calls = []
    original = dmt.array_split
    def observe(matrix, **kwargs):
        if kwargs.get('method') == 'svd':
            calls.append(matrix.shape)
        return original(matrix, **kwargs)
    monkeypatch.setattr(dmt, 'array_split', observe)
    out = PauliMPO.from_pauli_cores(cores).compress_dmt(max_bond=10, variant='operator')
    assert max(out.pauli_bond_dimensions) <= 10
    assert calls and max(max(shape) for shape in calls) <= 12


def test_cached_environments_agree_with_uncached_for_degenerate_spectra():
    from pepsy.operators._pauli_dmt import _TraceEnvironments
    # Equal Pauli coefficients intentionally create SVD ties. Compare with
    # the same gauges but fresh environments, since a tied optimal SVD need
    # not select the same unprotected subspace after unrelated gauge changes.
    terms = [(1/64, 'IIIIII'), (.001, 'ZZ'), (.001, 'XY')]
    op = PauliMPO.from_terms(6, terms).compress_pauli(cutoff=0.)
    cached = PauliDMT(op, max_bond=8)
    fresh = PauliDMT(op, max_bond=8)
    for bond in (0, 2, 4, 3, 1):
        fresh._workspace.environments = _TraceEnvironments(fresh._workspace.cores)
        cached.apply_gate(_local_gate_matrix(), (bond, bond+1))
        fresh.apply_gate(_local_gate_matrix(), (bond, bond+1))
        np.testing.assert_allclose(cached.state.to_dense(), fresh.state.to_dense(), atol=2e-13)


def test_torch_fixed_rank_gate_gradient_matches_finite_difference():
    torch = pytest.importorskip('torch')
    from pepsy.backends import TorchLinalgConfig
    rho = _density(5)
    cores = [np.asarray(a).real for a in rho.to_pauli_cores()]
    probe = PauliMPO.from_terms(5, [(1., 'XYZXI')])
    generator = PauliMPO(2, [(1., 'ZX'), (.3, 'XI')]).to_dense()
    def value(angle):
        gate = expm(-1j * angle * generator)
        out = PauliMPO.from_pauli_cores(cores).apply_gate_dmt(gate, (1, 2), max_bond=10)
        return float(np.real(probe.inner(out)))
    theta = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
    tcores = [torch.tensor(a, dtype=torch.float64) for a in cores]
    tgen = torch.tensor(generator, dtype=torch.complex128)
    with TorchLinalgConfig(stabilized=True, svd_gap_rtol=1e-9).activated():
        gate = torch.matrix_exp(-1j * theta * tgen)
        out = PauliMPO.from_pauli_cores(tcores).apply_gate_dmt(gate, (1, 2), max_bond=10)
        loss = probe.inner(out).real
        loss.backward()
    expected = (value(.130001) - value(.129999)) / .000002
    assert abs(expected) > 1e-8
    assert np.isfinite(theta.grad.item())
    assert theta.grad.item() == pytest.approx(expected, rel=2e-3, abs=2e-7)


def test_rejects_nonlocal_gates_and_periodic_boundary():
    with pytest.raises(ValueError, match='adjacent'):
        _density(4).apply_gate_dmt(_local_gate_matrix(), (0, 2), max_bond=16)
    with pytest.raises(NotImplementedError, match='open'):
        PauliMPO.identity(4, boundary='periodic').compress_dmt(max_bond=16)


@pytest.mark.parametrize('bad', ['steps', 'dt', 'shape', 'hermiticity'])
def test_invalid_evolution_input_does_not_apply_any_gate(bad):
    rho = _density(4)
    driver = PauliDMT(rho, max_bond=16)
    hs = _hamiltonians(4)
    kwargs = dict(dt=.1, steps=1)
    if bad == 'steps':
        kwargs['steps'] = True
    elif bad == 'dt':
        kwargs['dt'] = float('nan')
    elif bad == 'shape':
        hs[-1] = np.eye(2)
    else:
        hs[-1] = hs[-1].copy()
        hs[-1][0, 1] += 1.
    with pytest.raises(ValueError):
        driver.evolve(hs, **kwargs)
    np.testing.assert_allclose(driver.state.to_dense(), rho.to_dense(), atol=1e-14)
