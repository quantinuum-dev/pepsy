"""Large-chain DMT references without constructing a global density matrix."""

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import PauliMPO
from pepsy.optimizers import PauliDMT


_P = np.stack((np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], np.diag([1, -1])))
_P3 = np.stack([np.kron(np.kron(a, b), c) for a, b, c in itertools.product(_P, repeat=3)])


def product_mixture(n, *, rank=24, dtype='float64', seed=321, strength=.2):
    """PSD, trace-one MPO with an independent exact three-site oracle."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(.2, 1., rank)
    weights /= weights.sum()
    bloch = rng.normal(size=(rank, n, 3))
    bloch *= strength / np.linalg.norm(bloch, axis=-1, keepdims=True)
    local = np.concatenate((np.ones((rank, n, 1)), bloch), axis=-1) / 2
    cores = [(weights[:, None] * local[:, 0])[None, :, :]]
    for j in range(1, n - 1):
        core = np.zeros((rank, rank, 4))
        core[np.arange(rank), np.arange(rank)] = local[:, j]
        cores.append(core)
    cores.append(local[:, -1][:, None, :])
    return PauliMPO.from_pauli_cores([a.astype(dtype) for a in cores]), weights, local


def exact_marginal(weights, local, j):
    coefficients = np.einsum('a,ap,aq,ar->pqr', weights, local[:, j],
                             local[:, j + 1], local[:, j + 2])
    return np.einsum('p,pij->ij', coefficients.reshape(-1), _P3)


@pytest.mark.parametrize('n', [16, 64, 128])
@pytest.mark.parametrize('variant', ['density', 'operator'])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_large_chain_preserves_every_three_site_density_matrix(n, variant, dtype):
    rho, weights, local = product_mixture(n, dtype=dtype)
    out, report = rho.compress_dmt(max_bond=12, variant=variant, return_report=True)
    tolerance = 1e-10 if dtype == 'float64' else 2e-4
    assert max(out.pauli_bond_dimensions) <= 12
    assert any(b.input_rank > b.final_bond for b in report.bond_reports)
    assert abs(out.trace() - 1.) < tolerance
    for j in range(n - 2):
        actual = out.partial_trace(range(j, j + 3), keep=True).to_dense()
        np.testing.assert_allclose(actual, exact_marginal(weights, local, j),
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype,scale', [
    ('float64', 1e-180), ('float32', 1e-25),
    ('float64', 1e180), ('float32', 1e25),
])
def test_compression_is_homogeneous_at_extreme_operator_scale(dtype, scale):
    rho, _, _ = product_mixture(12, rank=16, dtype=dtype)
    reference = rho.compress_dmt(max_bond=12)
    # Scale one core directly to retain the requested dtype.
    cores = list(rho.to_pauli_cores())
    cores[0] = cores[0] * np.asarray(scale, dtype=dtype)
    out = PauliMPO.from_pauli_cores(cores).compress_dmt(max_bond=12)
    restored = list(out.to_pauli_cores())
    restored[-1] = restored[-1] / np.asarray(scale, dtype=dtype)
    restored = PauliMPO.from_pauli_cores(restored)
    tolerance = 2e-11 if dtype == 'float64' else 3e-5
    for j in range(10):
        np.testing.assert_allclose(
            restored.partial_trace(range(j, j + 3), keep=True).to_dense(),
            reference.partial_trace(range(j, j + 3), keep=True).to_dense(),
            atol=tolerance, rtol=tolerance,
        )


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('variant', ['density', 'operator'])
def test_insufficient_precision_is_explicit_and_larger_bond_recovers(dtype, variant):
    rho, weights, local = product_mixture(128, dtype=dtype, strength=.65)
    with pytest.raises((FloatingPointError, ValueError), match='precision'):
        rho.compress_dmt(max_bond=12, variant=variant, inplace=True)
    out = rho.compress_dmt(max_bond=24, variant=variant)
    tolerance = 3e-12 if dtype == 'float64' else 3e-5
    assert abs(out.trace() - 1.) < tolerance
    for j in range(126):
        np.testing.assert_allclose(
            out.partial_trace(range(j, j + 3), keep=True).to_dense(),
            exact_marginal(weights, local, j), atol=tolerance, rtol=tolerance,
        )


def single_z_coefficients(operator):
    """Independent O(L) coefficient contraction for all infinite-T Z probes."""
    cores = operator.to_pauli_cores()
    right = [None] * (len(cores) + 1)
    right[-1] = np.ones(1)
    for j in reversed(range(len(cores))):
        right[j] = cores[j][:, :, 0] @ right[j + 1]
    left, values = np.ones(1), []
    for j, core in enumerate(cores):
        values.append((left @ core[:, :, 3] @ right[j + 1]).item())
        left = left @ core[:, :, 0]
    return np.real_if_close(values)


@pytest.mark.slow
@pytest.mark.parametrize('n', [32, 128])
def test_large_xx_chain_infinite_temperature_dynamics(n):
    # Jordan-Wigner gives <Z_j(t) Z_c>/2**L = |exp(-i h t)[j,c]|**2.
    # The oracle is a one-particle L-by-L matrix, independent of tensor code.
    center = n // 2
    initial = PauliMPO.from_terms(n, [(1., (center,), 'Z')])
    bond_h = .5 * (np.kron(_P[1], _P[1]) + np.kron(_P[2], _P[2]))
    hopping = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    exact = abs(expm(-.5j * hopping)[:, center])**2
    errors = []
    for dt, steps in ((.05, 10), (.025, 20)):
        run = PauliDMT(initial, max_bond=16, variant='operator',
                       mode='heisenberg', diagnostics=True)
        out = run.evolve([bond_h] * (n - 1), dt=dt, steps=steps)
        actual = single_z_coefficients(out)
        errors.append(np.linalg.norm(actual - exact))
        assert abs(np.sum(actual) - 1.) < 2e-10
        assert run.time == pytest.approx(.5)
        assert max(out.pauli_bond_dimensions) <= 16
        assert len(run.report.bond_reports) <= 2 * n
    assert errors[1] < errors[0] / 3.8
    assert errors[1] < 1e-4


@pytest.mark.slow
def test_256_site_commuting_evolution_matches_three_spin_dense_reference():
    n, center, t = 256, 128, .7
    initial = PauliMPO.from_terms(n, [(1., (center,), 'X')])
    bond_h = np.kron(_P[3], _P[3])
    run = PauliDMT(initial, max_bond=8, variant='operator', mode='heisenberg')
    actual = run.evolve([bond_h] * (n - 1), dt=t / 5, steps=5)
    h = np.kron(bond_h, _P[0]) + np.kron(_P[0], bond_h)
    u = expm(-1j * t * h)
    target = u.conj().T @ np.kron(np.kron(_P[0], _P[1]), _P[0]) @ u
    # Normalized trace over the complement, avoiding a 2**256 trace factor.
    local = actual.partial_trace((center - 1, center, center + 1),
                                 keep=True, normalized=True).to_dense()
    np.testing.assert_allclose(local, target, atol=2e-12)


def test_huge_nonhermitian_hamiltonian_is_rejected_before_evolution():
    run = PauliDMT(PauliMPO.identity(2), max_bond=8)
    h = np.eye(4) * 1e200
    h[0, 1] = 1e200
    with pytest.raises(ValueError, match='Hermitian'):
        run.evolve([h], dt=1e-200)
    assert run.time == 0.


@pytest.mark.slow
@pytest.mark.parametrize('n,rank,cap', [(64, 48, 24), (256, 24, 12), (512, 24, 12)])
def test_larger_double_precision_chains_and_bonds(n, rank, cap):
    rho, weights, local = product_mixture(n, rank=rank)
    out = rho.compress_dmt(max_bond=cap)
    assert max(out.pauli_bond_dimensions) <= cap
    for j in range(n - 2):
        np.testing.assert_allclose(
            out.partial_trace(range(j, j + 3), keep=True).to_dense(),
            exact_marginal(weights, local, j), atol=2e-11, rtol=2e-11,
        )


def test_mixed_core_dtypes_are_rejected_clearly():
    cores = list(PauliMPO.identity(3).to_pauli_cores())
    cores[0] = cores[0].astype('complex64')
    with pytest.raises(TypeError, match='homogeneous core dtypes'):
        PauliMPO.from_pauli_cores(cores).compress_dmt(max_bond=12)


@pytest.mark.slow
def test_32_site_truncated_density_evolution_preserves_trace_and_energy_converges():
    n = 32
    rho, _, _ = product_mixture(n)
    h = .7 * np.kron(_P[3], _P[3]) + .23 * np.kron(_P[1], _P[0]) + .31 * np.kron(_P[0], _P[1])

    def energy(state):
        return sum(np.trace(state.partial_trace((j, j + 1), keep=True).to_dense() @ h)
                   for j in range(n - 1)).real

    original_energy, errors = energy(rho), []
    for dt, steps in ((.05, 4), (.025, 8)):
        run = PauliDMT(rho, max_bond=12, diagnostics=True)
        out = run.evolve([h] * (n - 1), dt=dt, steps=steps)
        assert abs(out.trace() - 1.) < 2e-10
        assert any(b.discarded_weight > 0 for b in run.report.bond_reports)
        assert max(out.pauli_bond_dimensions) <= 12
        errors.append(abs(energy(out) - original_energy))
    assert errors[1] < errors[0] / 3.
    assert errors[1] < 1e-4


@pytest.mark.slow
def test_repeated_identity_gates_reject_excessive_float32_trace_drift():
    rho, _, _ = product_mixture(128, dtype='float32', strength=.65)
    run = PauliDMT(rho, max_bond=12, variant='operator', trace_guard=True)
    with pytest.raises(FloatingPointError, match='trace at this precision'):
        for bond in range(127):
            run.apply_gate(np.eye(4), (bond, bond + 1))
    assert abs(run.state.trace() - 1.) < .1


@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='An unresolved nonzero trace cannot be inferred safely in auto mode')
def test_auto_trace_guard_cannot_classify_unresolved_float32_trace():
    rho, _, _ = product_mixture(128, dtype='float32', strength=.65)
    run = PauliDMT(rho, max_bond=12, variant='operator')
    for bond in range(127):
        run.apply_gate(np.eye(4), (bond, bond + 1))
    assert abs(run.state.trace() - 1.) < 2e-4


@pytest.mark.slow
@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_double_precision_rejects_large_trace_drift_during_gate_sweep(dtype):
    rho, _, _ = product_mixture(192, dtype=dtype, strength=.65)
    run = PauliDMT(rho, max_bond=12, variant='operator', cutoff='auto')
    with pytest.raises(FloatingPointError, match='trace at this precision'):
        for bond in range(191):
            run.apply_gate(np.eye(4), (bond, bond + 1))
    assert abs(run.state.trace() - 1.) < 1e-3


@pytest.mark.slow
def test_explicit_trace_guard_handles_trace_below_auto_classification():
    rho, _, _ = product_mixture(256, dtype='complex128', strength=.65)
    automatic = PauliDMT(rho, max_bond=12, variant='operator', cutoff='auto')
    assert not automatic.trace_guard_active
    run = PauliDMT(rho, max_bond=12, variant='operator', cutoff='auto', trace_guard=True)
    assert run.trace_guard_active
    with pytest.raises(FloatingPointError, match='trace at this precision'):
        for bond in range(255):
            run.apply_gate(np.eye(4), (bond, bond + 1))
    assert abs(run.state.trace() - 1.) < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize('cap', [12, 24])
def test_double_working_state_limits_identity_gate_drift_for_float32_source(cap):
    rho, weights, local = product_mixture(128, dtype='float32', strength=.65)
    # Explicit promotion preserves the float32 source values while allowing
    # all subsequent canonicalization and gates to run in double precision.
    rho = PauliMPO.from_pauli_cores([a.astype('float64') for a in rho.to_pauli_cores()])
    run = PauliDMT(rho, max_bond=cap, variant='operator')
    for bond in range(127):
        run.apply_gate(np.eye(4), (bond, bond + 1))
    out = run.state
    assert abs(out.trace() - 1.) < 2e-6
    for j in range(126):
        np.testing.assert_allclose(
            out.partial_trace(range(j, j + 3), keep=True).to_dense(),
            exact_marginal(weights, local, j), atol=2e-6, rtol=2e-6,
        )


@pytest.mark.slow
@pytest.mark.parametrize('variant', ['density', 'operator'])
@pytest.mark.parametrize('cutoff', [0., 'auto'])
def test_complex128_gauged_identity_sweep_on_ill_conditioned_chain(variant, cutoff):
    rho, weights, local = product_mixture(128, dtype='complex128', strength=.65)
    cores = list(rho.to_pauli_cores())
    rng = np.random.default_rng(19)
    for j in range(127):
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, cores[j].shape[1]))
        cores[j] = cores[j] * phases[None, :, None]
        cores[j + 1] = cores[j + 1] * phases.conj()[:, None, None]
    run = PauliDMT(PauliMPO.from_pauli_cores(cores), max_bond=12,
                   variant=variant, cutoff=cutoff)
    for j in range(127):
        report = run.apply_gate(np.eye(4), (j, j + 1))
        if j == 0 and cutoff == 'auto':
            assert report.residual_rank == 0
            assert report.discarded_weight == 0.
    out = run.state
    assert abs(out.trace() - 1.) < 3e-6
    for j in range(126):
        np.testing.assert_allclose(
            out.partial_trace(range(j, j + 3), keep=True).to_dense(),
            exact_marginal(weights, local, j), atol=3e-6, rtol=3e-6,
        )


@pytest.mark.slow
def test_auto_cutoff_matches_zero_when_no_additional_directions_are_dropped():
    rho, _, _ = product_mixture(128, dtype='complex128', strength=.65)
    cores = list(rho.to_pauli_cores())
    rng = np.random.default_rng(19)
    for j in range(127):
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi, cores[j].shape[1]))
        cores[j] *= phase[None, :, None]
        cores[j + 1] *= phase.conj()[:, None, None]
    initial = PauliMPO.from_pauli_cores(cores)
    outputs, ranks = [], []
    for cutoff in (0., 'auto'):
        run = PauliDMT(initial, max_bond=12, variant='operator', cutoff=cutoff)
        reports = [run.apply_gate(np.eye(4), (j, j + 1)) for j in range(127)]
        outputs.append(run.state)
        ranks.append([r.residual_rank for r in reports])
    assert ranks[0] == ranks[1]
    for j in range(126):
        np.testing.assert_allclose(
            outputs[0].partial_trace(range(j, j + 3), keep=True).to_dense(),
            outputs[1].partial_trace(range(j, j + 3), keep=True).to_dense(),
            atol=3e-12, rtol=3e-12,
        )
