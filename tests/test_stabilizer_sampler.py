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


def test_mps_stab_sampler_owns_branch_engine(monkeypatch):
    optimizer = pepsy.MpsStabOptimizer(4, chi=None).apply(
        [("h", 0), ("cnot", 0, 3), ("t", 1)]
    )
    sampler = pepsy.MpsStabSampler(optimizer)

    def fail_optimizer_sampling(*_args, **_kwargs):
        raise AssertionError("sampling branches should be owned by MpsStabSampler")

    monkeypatch.setattr(optimizer, "sample_bits", fail_optimizer_sampling)
    configs, probs = sampler.sample_arrays(12, seed=12, basis="Y", chunk_size=4)
    assert configs.shape == (12, 4)
    assert probs.shape == (12,)


def test_mps_stab_sampler_records_projector_local_infidelity_per_branch():
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(3)
    tableau.cnot(0, 1)  # physical Z_1 becomes coefficient Z_0 Z_1
    nu = qtn.MPS_computational_state("000", dtype="complex128")
    for q in range(3):
        nu.gate_(H, q, contract=True)  # parity projection creates entanglement

    optimizer = pepsy.MpsStabOptimizer.from_tableau_and_state(
        tableau,
        nu,
        chi=1,
        exact_cooling=False,
    )
    sampler = pepsy.MpsStabSampler(optimizer)
    sampler.sample_batch(
        32,
        seed=4,
        basis="Z",
        order=(1, 0, 2),
        shuffle=False,
        chunk_size=7,
    )

    diagnostics = sampler.get_sampling_diagnostics()
    first_readout = [record for record in diagnostics if record["depth"] == 0]
    assert first_readout
    assert {record["qubit"] for record in first_readout} == {1}
    assert all(record["absorb_basis"] is False for record in first_readout)
    assert all(record["branch_probability"] == pytest.approx(0.5) for record in first_readout)
    assert all(
        record["joint_probability"]
        == pytest.approx(record["prefix_probability"] * record["branch_probability"])
        for record in first_readout
    )
    assert any(record["projector_infidelity"] > 0.1 for record in first_readout)
    assert all(
        record["local_infidelity"] == pytest.approx(record["projector_infidelity"])
        for record in first_readout
    )

    # The nested records are defensive copies, not live optimizer history.
    diagnostics[0]["projector_infidelity"] = 0.0
    assert sampler.get_sampling_diagnostics()[0]["projector_infidelity"] != 0.0


def test_mps_stab_sampler_records_absorbed_localizer_events():
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(3)
    tableau.cnot(0, 1)
    nu = qtn.MPS_computational_state("000", dtype="complex128")
    for q in range(3):
        nu.gate_(H, q, contract=True)

    sampler = pepsy.MpsStabSampler(
        tableau,
        nu,
        chi=1,
        exact_cooling=False,
        absorb_basis=True,
    )
    sampler.sample_arrays(
        32,
        seed=4,
        basis="Z",
        order=(1, 0, 2),
        shuffle=False,
    )

    diagnostics = sampler.get_sampling_diagnostics()
    first_readout = [record for record in diagnostics if record["depth"] == 0]
    assert first_readout
    assert all(record["absorb_basis"] is True for record in first_readout)
    assert all(record["compression_events"] for record in first_readout)
    assert all(
        event["kind"] == "measurement_localizer"
        and event["local_infidelity"] is not None
        for record in first_readout
        for event in record["compression_events"]
    )
    assert all(record["norm_events"] for record in first_readout)
    assert all(record["projector_infidelity"] is not None for record in first_readout)


def test_mps_stab_sampler_disentangle_alias():
    optimizer = pepsy.MpsStabOptimizer(2).apply([("h", 0), ("cnot", 0, 1)])
    sampler = pepsy.MpsStabSampler(optimizer, disentangle=True)

    assert sampler.disentangle is True
    assert sampler.absorb_basis is True
    assert sampler.resolved_strategy == "frame_pauli_absorb"


def test_mps_stab_sampler_absorption_falls_back_if_localizer_zeroes_a_branch():
    # With a deliberately tiny chi, the approximate localizing CNOT can
    # remove a branch that had nonzero pre-localizer Born probability.
    optimizer = pepsy.MpsStabOptimizer(3, chi=1, exact_cooling=False).apply(
        [("cnot", 0, 1), ("rxx", 0.73, 0, 1)]
    )
    sampler = pepsy.MpsStabSampler(optimizer, absorb_basis=True)

    configs, probs = sampler.sample_arrays(
        128,
        seed=4,
        basis="Z",
        order=(1, 0, 2),
        shuffle=False,
    )
    diagnostics = sampler.get_sampling_diagnostics()

    assert configs.shape == (128, 3)
    assert np.all(np.isfinite(probs))
    assert any(
        record["condition_strategy"] == "frame_projector_fallback"
        for record in diagnostics
    )

    queried = sampler.probability_bits_many(
        configs,
        seed=4,
        basis="Z",
        order=(1, 0, 2),
    )
    np.testing.assert_allclose(queried, probs, atol=1e-12)
    for bits, probability in zip(configs[::19], probs[::19]):
        assert sampler.probability_bits(
            bits,
            seed=4,
            basis="Z",
            order=(1, 0, 2),
        ) == pytest.approx(probability, abs=1e-12)


def test_mps_stab_sampler_absorb_basis_recomputes_branch_frames_and_matches_dense():
    optimizer = pepsy.MpsStabOptimizer(3, chi=None).apply(
        [
            (H, 0),
            (
                np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    complex,
                ),
                (0, 1),
            ),
            (pepsy.rz(0.37), 2),
        ]
    )
    before = optimizer.to_statevector().copy()
    sampler = pepsy.MpsStabSampler(optimizer, absorb_basis=True)
    configs = np.asarray(list(product((0, 1), repeat=3)), dtype=np.int8)

    for basis in ("Z", "X", "Y", "XYZ"):
        resolved = (basis,) * 3 if len(basis) == 1 else tuple(basis)
        expected = _basis_probabilities(before, resolved)
        actual = sampler.probabilities(configs, basis=basis)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        assert np.sum(actual) == pytest.approx(1.0)

    np.testing.assert_allclose(optimizer.to_statevector(), before)
    assert sampler.absorb_basis is True
    assert sampler.resolved_strategy == "frame_pauli_absorb"

    # The legacy optimizer entry points remain delegates, including the new
    # branch-local absorption option.
    np.testing.assert_allclose(
        optimizer.probability_bits_many(
            configs,
            basis="XYZ",
            absorb_basis=True,
        ),
        _basis_probabilities(before, ("X", "Y", "Z")),
        atol=1e-10,
    )


def test_mps_stab_sampler_direct_constructor_forwards_chi_and_mode():
    tableau = stim.TableauSimulator()
    tableau.set_num_qubits(3)
    tableau.h(0)
    tableau.cnot(0, 1)
    nu = qtn.MPS_computational_state("000", dtype="complex128")

    sampler = pepsy.MpsStabSampler(
        tableau,
        nu,
        chi=4,
        mode="dmrg2",
        absorb_basis=True,
    )

    assert sampler.chi == 4
    assert sampler.mode == "dmrg2"
    assert sampler.absorb_basis is True
    samples = sampler.sample_bits(16, seed=8, shuffle=False)
    assert samples.shape == (16, 3)


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


def test_mps_stab_sampler_validates_public_sampling_contract():
    optimizer = pepsy.MpsStabOptimizer(2, chi=None).apply([("h", 0)])
    sampler = pepsy.MpsStabSampler(optimizer)

    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample_arrays(0)
    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample_bits(0)
    with pytest.raises(TypeError, match="integer"):
        sampler.sample_batch(True)
    with pytest.raises(ValueError, match="basis must"):
        sampler.sample_bits(2, basis="A")
    with pytest.raises(ValueError, match="permutation"):
        sampler.sample_bits(2, order=(0, 0))
    with pytest.raises(NotImplementedError, match="does not retain gradients"):
        sampler.sample_batch(2, track_grad=True)
    with pytest.raises(ValueError, match="Expected 'auto' or 'frame'"):
        pepsy.MpsStabSampler(optimizer, strategy="dense")


def test_mps_stab_sampler_probabilities_follow_shuffled_batch_configs():
    optimizer = pepsy.MpsStabOptimizer(3, chi=None).apply(
        [("h", 0), ("cnot", 0, 2), ("t", 1)]
    )
    sampler = pepsy.MpsStabSampler(optimizer)
    batch = sampler.sample_batch(64, seed=13, basis="XYZ", shuffle=True)

    np.testing.assert_allclose(
        sampler.probabilities(batch.configs, basis=batch.basis),
        batch.probs,
        atol=1e-12,
    )
