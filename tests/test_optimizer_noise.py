"""Regression tests for sampled Pauli-noise optimizer trajectories."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy


def _fidelity(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_depolarizing_model_and_validation():
    model = pepsy.PauliErrorModel.depolarizing(0.3)
    assert model.probabilities == pytest.approx(
        {"I": 0.7, "X": 0.1, "Y": 0.1, "Z": 0.1}
    )
    assert pepsy.PauliErrorModel.bit_flip(1.0).probabilities == pytest.approx(
        {"I": 0.0, "X": 1.0, "Y": 0.0, "Z": 0.0}
    )
    with pytest.raises(ValueError, match="must not exceed"):
        pepsy.PauliErrorModel(p_x=0.6, p_z=0.6)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pepsy.PauliErrorModel.depolarizing(1.1)


def test_sampled_stream_is_reproducible_and_skips_control_events():
    gates = [(qu.hadamard(), 0), ("measure", "Z", 0)]
    model = pepsy.PauliErrorModel.depolarizing(0.8)
    first = pepsy.sample_noisy_gate_streams(gates, model, 5, seed=12)
    second = pepsy.sample_noisy_gate_streams(gates, model, 5, seed=12)
    assert len(first) == len(second) == 5
    for left, right in zip(first, second):
        assert len(left) == len(right)
        for entry_left, entry_right in zip(left, right):
            if isinstance(entry_left[0], str):
                assert entry_left == entry_right
            else:
                np.testing.assert_allclose(entry_left[0], entry_right[0])
                assert entry_left[1] == entry_right[1]
        assert sum(
            entry[0] == "measure" for entry in left if isinstance(entry[0], str)
        ) == 1


def test_sampled_pauli_stream_replays_with_stn_and_stays_clifford():
    gates = [(qu.hadamard(), 0)]
    result = pepsy.run_noisy_shots(
        lambda: pepsy.MpsStabOptimizer(1),
        gates,
        pepsy.PauliErrorModel.bit_flip(1.0),
        shots=3,
        seed=2,
    )
    expected = (
        np.array([[0.0, 1.0], [1.0, 0.0]])
        @ np.array([[1.0, 1.0], [1.0, -1.0]])
        @ np.array([1.0, 0.0])
        / np.sqrt(2.0)
    )
    assert result.shots == 3
    assert all(
        _fidelity(shot.p.to_dense(), np.array([1.0, 0.0])) == pytest.approx(1.0)
        for shot in result.optimizers
    )
    assert all(
        shot_faults == (pepsy.PauliFault(0, 0, "X"),)
        for shot_faults in result.faults
    )
    for simulator in result.optimizers:
        assert _fidelity(simulator.to_statevector(), expected) == pytest.approx(1.0)


def test_sampled_pauli_stream_replays_with_mps_optimizer():
    gates = [(qu.hadamard(), 0)]
    result = pepsy.run_noisy_shots(
        lambda: pepsy.MpsOptimizer(
            qtn.MPS_computational_state("0"), chi=2, mode="mpo"
        ),
        gates,
        pepsy.PauliErrorModel.phase_flip(1.0),
        shots=2,
        seed=3,
        run_kwargs={"progbar": False},
    )
    expected = qu.pauli("Z") @ qu.hadamard() @ np.array([1.0, 0.0])
    for optimizer in result.optimizers:
        assert _fidelity(optimizer.p.to_dense(), expected) == pytest.approx(1.0)


def test_sampled_pauli_stream_replays_with_backend_matched_torch_gates():
    """Independent Pauli sampling keeps generated faults on the gate backend."""
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("0", dtype="complex128")
    state.apply_to_arrays(
        lambda array: torch.as_tensor(array, dtype=torch.complex128)
    )
    gate = torch.tensor(
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0),
        dtype=torch.complex128,
    )

    result = pepsy.run_noisy_shots(
        lambda: pepsy.MpsOptimizer(state, chi=2, mode="mpo"),
        [(gate, 0)],
        pepsy.PauliErrorModel.bit_flip(1.0),
        shots=1,
        seed=3,
        run_kwargs={"progbar": False},
    )

    optimizer = result.optimizers[0]
    assert type(optimizer.p[0].data).__module__.split(".", 1)[0] == "torch"
    expected = qu.pauli("X") @ qu.hadamard() @ np.array([1.0, 0.0])
    assert _fidelity(optimizer.p.to_dense(), expected) == pytest.approx(1.0)


def test_noisy_shots_require_fresh_compatible_optimizers():
    with pytest.raises(TypeError, match="optimizer_factory"):
        pepsy.run_noisy_shots(None, [], pepsy.PauliErrorModel(), shots=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        pepsy.sample_noisy_gate_streams([], pepsy.PauliErrorModel(), -1)
