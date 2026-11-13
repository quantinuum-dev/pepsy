"""Canonical invariants, bounded gauge moves, and safe preparation caches."""

import numpy as np
import pytest

from pepsy.operators import PauliMPO
from pepsy.optimizers import PauliDMT


@pytest.mark.parametrize('absorb', ['left', 'right'])
def test_thin_refactor_needs_one_qr_and_preserves_the_product(monkeypatch, absorb):
    import pepsy.operators._pauli_dmt as dmt
    rng = np.random.default_rng(883)
    left = rng.normal(size=(27, 12)) + 1j * rng.normal(size=(27, 12))
    right = rng.normal(size=(12, 31)) + 1j * rng.normal(size=(12, 31))
    calls, original = [], dmt.array_split

    def observe(matrix, **kwargs):
        calls.append(kwargs.get('method'))
        return original(matrix, **kwargs)

    monkeypatch.setattr(dmt, 'array_split', observe)
    a, b, discarded = dmt._refactor(left, right, absorb)
    np.testing.assert_allclose(a @ b, left @ right, atol=5e-14)
    gram = a.conj().T @ a if absorb == 'right' else b @ b.conj().T
    np.testing.assert_allclose(gram, np.eye(12), atol=1e-14)
    assert discarded == 0.
    assert calls == ['qr']


def test_basis_retry_reuses_unchanged_schmidt_decomposition(monkeypatch):
    import pepsy.operators._pauli_dmt as dmt
    calls, original = [], dmt._svd

    def observe(matrix, **kwargs):
        if matrix.shape == (12, 12) and kwargs.get('max_bond') is None:
            calls.append(matrix.shape)
        return original(matrix, **kwargs)

    monkeypatch.setattr(dmt, '_svd', observe)
    m = np.diag([.99, .9, .05, .8, .7, .6, .5, .4, .3, .2, .1, 1e-6])
    eye = np.eye(12)
    lv = np.column_stack((eye[:, 0], eye[:, 1],
                          eye[:, 1] + 1e-10 * eye[:, 2], eye[:, 3]))
    rv = eye[:, [0, 4, 5, 6]]
    a, b, report = dmt._split(m, lv, rv, max_bond=9, cutoff=0.,
                             variant='operator', absorb='right', bond=0,
                             diagnostics=True)
    assert report.protected_left_rank == 4
    np.testing.assert_allclose(lv.T @ (a @ b - m), 0., atol=1e-14)
    np.testing.assert_allclose((a @ b - m) @ rv, 0., atol=1e-14)
    assert len(calls) == 1


def test_standalone_compression_does_not_build_unused_gate_trace_guard(monkeypatch):
    import pepsy.operators._pauli_dmt as dmt

    def forbidden(*args, **kwargs):
        raise AssertionError('standalone compression does not apply gates')

    monkeypatch.setattr(dmt, '_trace_fraction', forbidden)
    observable = PauliMPO.from_terms(6, [(1., 'XIIIII'), (.3, 'ZZIIII')])
    result = observable.compress_dmt(max_bond=8, variant='operator')
    np.testing.assert_allclose(result.to_dense(), observable.to_dense(), atol=1e-13)


def test_canonical_center_metadata_matches_isometries_after_jumps():
    rng = np.random.default_rng(39)
    cores = [rng.normal(size=(1, 12, 4))]
    cores += [rng.normal(size=(12, 12, 4)) for _ in range(4)]
    cores += [rng.normal(size=(12, 1, 4))]
    run = PauliDMT(PauliMPO.from_pauli_cores(cores), max_bond=10, variant='operator')
    workspace = run._workspace
    for bond, absorb in ((0, 'right'), (3, 'left'), (1, 'left'), (4, 'right')):
        workspace.split_pair(bond, absorb=absorb)
        for j, core in enumerate(workspace.cores):
            if j < workspace.center:
                matrix = core.transpose(0, 2, 1).reshape(-1, core.shape[1])
                gram = matrix.conj().T @ matrix
            elif j > workspace.center:
                matrix = core.reshape(core.shape[0], -1)
                gram = matrix @ matrix.conj().T
            else:
                continue
            np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=3e-14)


def test_shared_hamiltonian_preparation_is_cached_only_within_call(monkeypatch):
    import pepsy.optimizers.pauli_dmt as driver
    counts = dict(eigh=0, transfer=0)
    original_do, original_transfer = driver.ar.do, driver._gate_transfer

    def observe_do(fn, *args, **kwargs):
        if fn == 'linalg.eigh':
            counts['eigh'] += 1
        return original_do(fn, *args, **kwargs)

    def observe_transfer(*args, **kwargs):
        counts['transfer'] += 1
        return original_transfer(*args, **kwargs)

    monkeypatch.setattr(driver.ar, 'do', observe_do)
    monkeypatch.setattr(driver, '_gate_transfer', observe_transfer)
    h = np.diag([.7, -.2, .4, -.9])
    initial = PauliMPO.from_terms(6, [(1., (2,), 'X')])
    run = PauliDMT(initial, max_bond=16, variant='operator')
    run.evolve([h] * 5, dt=.03)
    assert counts == dict(eigh=1, transfer=2)
    previous = run.state
    # A numerically tiny trace from a traceless observable must not be
    # mistaken for a resolved trace after recreating the workspace.
    assert not PauliDMT(previous, max_bond=16, variant='operator').trace_guard_active
    h[0, 1] = h[1, 0] = .31
    actual = run.evolve([h] * 5, dt=.03)
    assert counts == dict(eigh=2, transfer=4)
    expected = PauliDMT(previous, max_bond=16, variant='operator').evolve(
        [h.copy() for _ in range(5)], dt=.03)
    np.testing.assert_allclose(actual.to_dense(), expected.to_dense(), atol=2e-13)


def test_serpentine_sweeps_avoid_full_chain_return_moves(monkeypatch):
    n, steps = 16, 4
    run = PauliDMT(PauliMPO.identity(n), max_bond=4)
    workspace = run._workspace
    original, moves = workspace.move_center, []

    def observe(target):
        moves.append(abs(target - workspace.center))
        return original(target)

    monkeypatch.setattr(workspace, 'move_center', observe)
    h = np.diag([.7, -.2, .4, -.9])
    run.evolve([h] * (n - 1), dt=.03, steps=steps)
    # Each of the three parity layers crosses only the gaps between its
    # disjoint bonds; no extra end-to-end move between timesteps is needed.
    assert sum(moves) <= 3 * (n // 2 - 1) * steps


def test_sweep_orientation_persists_across_evolution_calls():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(32, 32))
    dense = x @ x.T
    dense /= np.trace(dense)
    initial = PauliMPO.from_dense(dense, max_qubits=5)
    h = np.diag([.7, -.2, .4, -.9])
    h[0, 1] = h[1, 0] = .23
    together = PauliDMT(initial, max_bond=10)
    expected = together.evolve([h] * 4, dt=.03, steps=4)
    split = PauliDMT(initial, max_bond=10)
    split.evolve([h] * 4, dt=.03, steps=2)
    actual = split.evolve([h] * 4, dt=.03, steps=2)
    np.testing.assert_allclose(actual.to_dense(), expected.to_dense(), atol=2e-13)
    assert split._workspace.center == together._workspace.center


def test_shared_hamiltonian_cache_preserves_torch_parameter_gradient():
    torch = pytest.importorskip('torch')
    from pepsy.backends import TorchLinalgConfig
    rng = np.random.default_rng(63)
    x = rng.normal(size=(8, 8))
    dense = x @ x.T
    dense /= np.trace(dense)
    initial = PauliMPO.from_dense(dense, max_qubits=3)
    x, y = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    base, generator = x + x.T, y + y.T
    probe = PauliMPO.from_terms(3, [(1., 'XYZ')])

    def value(theta):
        h = base + theta * generator
        out = PauliDMT(initial, max_bond=16).evolve([h] * 2, dt=.04, steps=2)
        return float(np.real(probe.inner(out)))

    theta = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
    h = torch.tensor(base, dtype=torch.complex128) + theta * torch.tensor(generator, dtype=torch.complex128)
    tinitial = PauliMPO.from_pauli_cores([
        torch.tensor(a, dtype=torch.complex128) for a in initial.to_pauli_cores()
    ])
    with TorchLinalgConfig(stabilized=True).activated():
        out = PauliDMT(tinitial, max_bond=16).evolve([h] * 2, dt=.04, steps=2)
        probe.inner(out).real.backward()
    expected = (value(.130001) - value(.129999)) / .000002
    assert abs(expected) > 1e-6
    assert np.isfinite(theta.grad.item())
    assert theta.grad.item() == pytest.approx(expected, rel=2e-4, abs=2e-7)
