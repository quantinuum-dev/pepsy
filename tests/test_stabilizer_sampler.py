"""Tests for scalable product-Pauli sampling of ``C|nu>``."""

from itertools import product

import numpy as np
import pytest

stim = pytest.importorskip("stim")

import pepsy
import quimb.tensor as qtn


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


def _apply_local(psi, gate, where, n):
    where = (where,) if isinstance(where, int) else tuple(where)
    k = len(where)
    tensor = np.asarray(psi).reshape((2,) * n)
    operator = np.asarray(gate).reshape((2,) * (2 * k))
    out = np.tensordot(
        operator,
        tensor,
        axes=(tuple(range(k, 2 * k)), where),
    )
    remaining = [q for q in range(n) if q not in where]
    order = [
        where.index(q) if q in where else k + remaining.index(q)
        for q in range(n)
    ]
    return out.transpose(order).reshape(-1)


def _measurement_rotation(axis):
    if axis == "Z":
        return I
    if axis == "X":
        return H
    return H @ np.diag([1.0, -1.0j])


def _basis_probabilities(state, basis):
    n = len(basis)
    rotated = np.asarray(state)
    for q, axis in enumerate(basis):
        rotated = _apply_local(
            rotated,
            _measurement_rotation(axis),
            q,
            n,
        )
    return np.abs(rotated) ** 2


def test_mps_stab_sampler_matches_dense_probabilities_in_x_y_and_mixed_bases():
    optimizer = pepsy.MpsStabOptimizer(3, chi=None).apply(
        [(H, 0), (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex), (0, 1)), (pepsy.rz(0.37), 2)]
    )
    sampler = pepsy.MpsStabSampler(optimizer)
    configs = np.asarray(list(product((0, 1), repeat=3)), dtype=np.int8)
    physical = optimizer.to_statevector()

    for basis in ("Z", "X", "Y", "XYZ"):
        resolved = (basis,) * 3 if len(basis) == 1 else tuple(basis)
        expected = _basis_probabilities(physical, resolved)
        actual = sampler.probabilities(configs, basis=basis)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        assert np.sum(actual) == pytest.approx(1.0)


def test_mps_stab_sampler_returns_mps_style_batches_without_mutating_state():
    optimizer = pepsy.MpsStabOptimizer(2, chi=None).apply([("h", 0), ("cnot", 0, 1)])
    before = optimizer.to_statevector().copy()
    sampler = pepsy.MpsStabSampler(optimizer)

    batch = sampler.sample_batch(32, seed=4, basis="random")

    assert batch.configs.shape == (32, 2)
    assert batch.probs.shape == (32,)
    assert len(batch.basis) == 2
    assert set(batch.basis) <= {"X", "Y", "Z"}
    assert batch.basis_probability == pytest.approx(3.0 ** -2)
    np.testing.assert_allclose(optimizer.to_statevector(), before)

    chunks = list(sampler.iter_samples(9, seed=8, basis="XY", chunk_size=4))
    assert [len(chunk) for chunk in chunks] == [4, 4, 1]
    assert all(chunk.basis == ("X", "Y") for chunk in chunks)


def test_mps_stab_sampler_uses_frame_projectors_not_dense_readout(monkeypatch):
    optimizer = pepsy.MpsStabOptimizer(8, chi=None).apply(
        [("h", 0), ("cnot", 0, 7), ("t", 3)]
    )
    sampler = pepsy.MpsStabSampler(optimizer)

    def fail_dense_readout(*_args, **_kwargs):
        raise AssertionError("sampling should not reconstruct C|nu> densely")

    monkeypatch.setattr(optimizer, "to_statevector", fail_dense_readout)
    samples = sampler.sample_arrays(16, seed=12, basis="Y")
    assert samples[0].shape == (16, 8)
    assert samples[1].shape == (16,)


def test_mps_stab_sampler_accepts_tableau_and_coefficient_mps_directly():
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(2)
    tableau.h(0)
    nu = qtn.MPS_computational_state("00", dtype="complex128")

    sampler = pepsy.MpsStabSampler(tableau, nu)
    configs, probs = sampler.sample_arrays(11, seed=9, basis="X", chunk_size=3)

    assert sampler.resolved_strategy == "frame_pauli"
    assert sampler.native_backend == "numpy"
    assert configs.shape == (11, 2)
    np.testing.assert_allclose(probs, 0.5)


def test_mps_stab_sampler_native_torch_batch_is_chunked_and_convertible():
    torch = pytest.importorskip("torch")
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(2)
    tableau.h(0)
    nu = qtn.MPS_computational_state("00", dtype="complex128")
    to_torch = pepsy.backend_torch(dtype=torch.complex128, device="cpu")

    sampler = pepsy.MpsStabSampler(
        tableau,
        nu,
        to_backend=to_torch,
        backend="native",
    )
    batch = sampler.sample_batch(13, seed=5, basis="Z", chunk_size=4)

    assert batch.backend == "torch"
    assert isinstance(batch.configs, torch.Tensor)
    assert isinstance(batch.probs, torch.Tensor)
    assert batch.configs.shape == (13, 2)
    assert batch.probs.shape == (13,)
    assert batch.configs.device.type == "cpu"
    numpy_batch = batch.to_numpy()
    assert numpy_batch.backend == "numpy"
    np.testing.assert_allclose(numpy_batch.probs, 0.5)
