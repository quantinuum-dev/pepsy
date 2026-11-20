"""Public interval updates against independent physical density targets."""

import numpy as np
import pytest
from scipy.linalg import expm

from examples import dmt_2d_dynamics as exact
from examples import dmt_2d_window as window
from pepsy.backends import TorchLinalgConfig
from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.optimizers import MpoOptimizer, PauliDMT
from test_pauli_dmt_large import product_mixture


def matrix_unit_transfers(operators):
    from examples.dmt_torch_lattice import matrix_unit_transfers as convert

    return convert(operators)


def prepare(representation, *, cap=32, regions=None, checked=True, preservation_check='raise'):
    torch = pytest.importorskip('torch')
    rho, _, _ = product_mixture(6, rank=48, strength=.5, dtype='complex128')
    rho = PauliMPO.from_pauli_cores([torch.tensor(a) for a in rho.to_pauli_cores()])
    options = dict(regions=regions or square_lattice_dmt_regions(2, 3),
                   cutoff=0., region_selection='min_cover', check_regions=checked,
                   preservation_check=preservation_check)
    if representation == 'pauli':
        return PauliDMT(rho, max_bond=cap, diagnostics=True, **options), options, rho
    return MpoOptimizer(rho.to_mpo(), [], chi=cap, mode='dmt', contraction_opt='greedy'), options, rho


def update(run, representation, start, operators, options, **kwargs):
    if representation == 'pauli':
        return run.apply_transfer_mpo(start, operators, **kwargs)
    _, report = run.apply_transfer_mpo(start, matrix_unit_transfers(operators), return_report=True,
                                      **options, **kwargs)
    return report


def dense(run, representation):
    value = run.state.to_dense() if representation == 'pauli' else run.p.to_dense()
    return value.detach().numpy()


@pytest.mark.parametrize('representation', ['pauli', 'mpo'])
@pytest.mark.parametrize('absorb', ['left', 'right'])
@pytest.mark.parametrize('route', ['interval', 'full'])
def test_actual_loss_keeps_every_post_gate_physical_region(representation, absorb, route):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    with TorchLinalgConfig(cpu_svd='scipy_gesvd', quimb_split_drivers=True,
                           qr_rank_policy='native').activated():
        run, options, initial = prepare(representation)
        before = initial.to_dense().detach().numpy()
        info = run.info_c
        ops = [torch.tensor(a, dtype=torch.complex128) for a in window.zz_transfer(4, .23)]
        report = update(run, representation, 1, ops, options, absorb=absorb, update=route)
        actual = dense(run, representation)
        u = expm(-.23j*np.kron(exact.ref.Z, exact.ref.Z))
        target = exact.apply_density(before, u, (1, 4), 6)
        assert max(report.final_bond_dimensions) <= 32
        assert max(b.discarded_weight for b in report.bond_reports) > 1e-9
        assert np.linalg.norm(actual-target) > 1e-6
        for region in options['regions']:
            np.testing.assert_allclose(exact.ref.partial_trace(actual, region, 6),
                                       exact.ref.partial_trace(target, region, 6), atol=2e-12)
        np.testing.assert_allclose(np.trace(actual), np.trace(target), atol=2e-12)
        np.testing.assert_array_equal(initial.to_dense().detach().numpy(), before)
        assert run.info_c is info
        assert not report.preservation_errors and report.check_regions


@pytest.mark.parametrize('representation', ['pauli', 'mpo'])
def test_one_site_nonunitary_transfer_preserves_scale_and_failure_is_transactional(representation):
    torch = pytest.importorskip('torch')
    with TorchLinalgConfig(quimb_split_drivers=True, qr_rank_policy='native').activated():
        run, options, _ = prepare(representation, cap=64)
        before = dense(run, representation)
        # A scaled identity transfer maps rho -> 1.3 rho; it must not normalize.
        op = (1.3*torch.eye(4, dtype=torch.complex128)).reshape(1, 1, 4, 4)
        report = update(run, representation, 2, [op], options)
        np.testing.assert_allclose(dense(run, representation), 1.3*before, atol=2e-13)
        assert not report.bond_reports
        saved, info = dense(run, representation), dict(run.info_c)
        saved_report = run.report if representation == 'pauli' else run.get_dmt_reports()
        for bad in [torch.zeros_like(op), op*float('nan')]:
            with pytest.raises((ValueError, FloatingPointError)):
                update(run, representation, 2, [bad], options)
            np.testing.assert_array_equal(dense(run, representation), saved)
            assert run.info_c == info
            assert (run.report if representation == 'pauli' else run.get_dmt_reports()) == saved_report


def test_ordinary_interval_cache_reuses_preparation_and_invalidates_on_new_settings(monkeypatch):
    torch = pytest.importorskip('torch')
    import pepsy.optimizers.mpo.dmt as driver

    run, options, _ = prepare('mpo', cap=64, checked=False)
    op = torch.eye(4, dtype=torch.complex128).reshape(1, 1, 4, 4)
    update(run, 'mpo', 1, [op], options)
    original = driver._prepare_mpo_dmt
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, '_prepare_mpo_dmt', record)
    update(run, 'mpo', 3, [op], options)
    assert not calls
    update(run, 'mpo', 3, [op], options | dict(check_regions=True))
    assert len(calls) == 1


@pytest.mark.parametrize('representation', ['pauli', 'mpo'])
@pytest.mark.parametrize('policy', ['warn', 'raise'])
def test_one_site_full_check_failures_are_recorded_or_rolled_back(monkeypatch, representation, policy):
    torch = pytest.importorskip('torch')
    from pepsy.operators._pauli_dmt import _PauliDMTWorkspace

    run, options, _ = prepare(representation, cap=64, preservation_check=policy)
    before = dense(run, representation)
    original = _PauliDMTWorkspace._factor_interval

    def corrupt(self, far, near):
        original(self, far, near)
        if far == near:
            self.cores[near] = self.cores[near] * (1 + 1e-4)

    monkeypatch.setattr(_PauliDMTWorkspace, '_factor_interval', corrupt)
    identity = torch.eye(4, dtype=torch.complex128).reshape(1, 1, 4, 4)
    if policy == 'raise':
        with pytest.raises(FloatingPointError, match='local marginals'):
            update(run, representation, 2, [identity], options)
        np.testing.assert_array_equal(dense(run, representation), before)
    else:
        with pytest.warns(RuntimeWarning, match='marginal preservation'):
            report = update(run, representation, 2, [identity], options)
        assert not report.bond_reports
        assert report.update_errors and report.preservation_errors == report.update_errors
        np.testing.assert_allclose(dense(run, representation), before * (1 + 1e-4), atol=2e-12)


@pytest.mark.parametrize('representation', ['pauli', 'mpo'])
def test_full_interval_agree_with_exact_circuit_when_no_truncation(representation, tmp_path):
    pytest.importorskip('torch')
    from examples import dmt_torch_lattice as bench
    output = tmp_path/'trajectory.json'
    result = bench.trajectory((2, 2), chi=16, steps=2, representation=representation,
                              max_seconds=60, output=output)
    # The interval benchmark uses field/ZZ/field Strang, so use its own exact
    # bit-state schedule rather than the distributed-field edge schedule.
    from examples import dmt_2d_scaling as reference

    source = reference.problem(2, 2, .1)
    psi = reference.vector_step(reference.vector_step(source['psi'], source), source)
    truth = reference.local_reference(psi, source)['plaquettes']
    actual = np.load(str(output)+'.npz')['2']
    np.testing.assert_allclose(actual, truth, atol=2e-12)
    assert result['status'] == 'completed'
    assert result['samples'][-1]['trace_error'] < 2e-12
    assert result['samples'][-1]['max_local_hermiticity'] < 2e-12


@pytest.mark.parametrize('representation', ['pauli', 'mpo'])
def test_single_transfer_keeps_torch_parameter_gradient(representation):
    torch = pytest.importorskip('torch')
    run, options, _ = prepare(representation, cap=64, checked=False)
    theta = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    op = (theta*torch.eye(4, dtype=torch.complex128)).reshape(1, 1, 4, 4)
    update(run, representation, 2, [op], options)
    trace = run.state.trace() if representation == 'pauli' else run.p.trace()
    trace.real.backward()
    assert theta.grad.item() == pytest.approx(1., abs=2e-12)


def test_uncached_oversized_ordinary_source_is_compressed_only_after_gate():
    torch = pytest.importorskip('torch')
    rho, _, _ = product_mixture(8, rank=48, strength=.5, dtype='complex128')
    rho = PauliMPO.from_pauli_cores([torch.tensor(a) for a in rho.to_pauli_cores()])
    with TorchLinalgConfig(cpu_svd='scipy_gesvd', quimb_split_drivers=True,
                           qr_rank_policy='native').activated():
        before = rho.to_dense().detach().numpy()
        run = MpoOptimizer(rho.to_mpo(), [], chi=12, mode='dmt', contraction_opt='greedy')
        ops = [torch.tensor(a, dtype=torch.complex128) for a in window.zz_transfer(2, .27)]
        _, report = run.apply_transfer_mpo(3, matrix_unit_transfers(ops), cutoff=0.,
                                           check_regions=True, return_report=True)
        target = exact.apply_density(before, expm(-.27j*np.kron(exact.ref.Z, exact.ref.Z)), (3, 4), 8)
        actual = dense(run, 'mpo')
        assert run.p.max_bond() <= 12 and not report.preservation_errors
        for j in range(6):
            region = tuple(range(j, j+3))
            np.testing.assert_allclose(exact.ref.partial_trace(actual, region, 8),
                                       exact.ref.partial_trace(target, region, 8), atol=2e-12)
