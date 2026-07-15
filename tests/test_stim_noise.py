"""Tests for compiling every native Stim Pauli-noise channel."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy

stim = pytest.importorskip("stim")


def _fault_terms(sample):
    return {(fault.gate_index, fault.site, fault.pauli) for fault in sample.faults}


def test_stim_native_noise_set_is_complete():
    native_channels = {
        name
        for name, data in stim.gate_data().items()
        if data.is_noisy_gate
        and name
        not in {
            "M", "MPP", "MR", "MRX", "MRY", "MX", "MXX", "MY", "MYY", "MZZ",
        }
    }
    from pepsy.optimizers import noise

    assert native_channels == noise._STIM_NOISE_NAMES


def test_stim_compiler_samples_all_native_error_channel_forms():
    circuit = stim.Circuit(
        """
        X_ERROR(1) 0
        Y_ERROR(1) 1
        Z_ERROR(1) 2
        DEPOLARIZE1(0) 0
        DEPOLARIZE2(0) 0 1
        PAULI_CHANNEL_1(0, 0, 1) 2
        PAULI_CHANNEL_2(0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1) 0 1
        I_ERROR(1) 0
        II_ERROR(1) 0 1
        E(0) X0
        ELSE_CORRELATED_ERROR(1) Y1
        E(1) Z2
        ELSE_CORRELATED_ERROR(1) X1
        HERALDED_ERASE(1) 0
        HERALDED_PAULI_CHANNEL_1(0, 0, 0, 1) 1
        """
    )
    sample = pepsy.sample_stim_circuit(circuit, seed=5)
    terms = _fault_terms(sample)
    assert (0, 0, "X") in terms
    assert (1, 1, "Y") in terms
    assert (2, 2, "Z") in terms
    assert (5, 2, "Z") in terms
    assert (6, 0, "Z") in terms and (6, 1, "Z") in terms
    assert (10, 1, "Y") in terms
    assert (11, 2, "Z") in terms
    assert (14, 1, "Z") in terms
    assert sample.heralds[-1] == pepsy.StimHerald(14, 1, True)
    assert sample.heralds[-2].instruction_index == 13
    assert sample.heralds[-2].site == 0
    assert sample.heralds[-2].value is True

    coalesced = pepsy.run_coalesced_stim_shots(
        lambda: pepsy.MpsStabOptimizer(3), circuit, shots=8, seed=5
    )
    assert coalesced.shots == 8
    assert all(len(leaf.heralds) == 2 for leaf in coalesced.leaves)
    assert all(
        (pepsy.PauliFault(0, 0, "X") in leaf.faults)
        and (pepsy.PauliFault(14, 1, "Z") in leaf.faults)
        for leaf in coalesced.leaves
    )


def test_stim_compiler_flattens_repeats_and_translates_standard_operations():
    circuit = stim.Circuit(
        """
        R 0 1
        H 0
        CX 0 1
        MPP X0*X1
        REPEAT 2 {
            X_ERROR(1) 0
        }
        """
    )
    plan = pepsy.compile_stim_circuit(circuit)
    assert plan.num_qubits == 2
    sample = pepsy.sample_stim_circuit(plan, seed=2)
    assert sum(fault.pauli == "X" and fault.site == 0 for fault in sample.faults) == 2
    assert any(
        entry[0] == "measure" and entry[1:] == ("XX", (0, 1))
        for entry in sample.gate_stream
        if isinstance(entry[0], str)
    )


def test_stim_sampled_trajectory_replays_with_stn_and_ordinary_mps():
    circuit = "H 0\nZ_ERROR(1) 0"
    stn = pepsy.run_stim_shots(
        lambda: pepsy.MpsStabOptimizer(1), circuit, shots=2, seed=7
    )
    mps = pepsy.run_stim_shots(
        lambda: pepsy.MpsOptimizer(
            qtn.MPS_computational_state("0"), chi=2, mode="mpo"
        ),
        circuit,
        shots=2,
        seed=7,
        run_kwargs={"progbar": False},
    )
    expected = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
    for shot in stn.optimizers:
        np.testing.assert_allclose(shot.to_statevector(), expected, atol=1e-6)
    for shot in mps.optimizers:
        np.testing.assert_allclose(shot.p.to_dense().reshape(-1), expected, atol=1e-10)
    assert all(sample.faults == (pepsy.PauliFault(1, 0, "Z"),) for sample in stn.samples)


def test_mps_stab_optimizer_from_stim_keeps_sample_and_transforms_stream():
    circuit = "H 0\nZ_ERROR(1) 0\nM 0"
    received = []

    def omit_terminal_readout(stream):
        received.append(stream)
        return stream[:-1]

    sim = pepsy.MpsStabOptimizer.from_stim(
        circuit,
        seed=7,
        stream_transform=omit_terminal_readout,
        chi=2,
    )

    assert sim.n == 1
    assert sim.stim_plan.num_qubits == 1
    expected_sample = pepsy.sample_stim_circuit(circuit, seed=7)
    assert sim.stim_sample.faults == expected_sample.faults
    assert sim.stim_sample.heralds == expected_sample.heralds
    assert received[0] is sim.stim_sample.gate_stream
    assert sim.stim_sample.faults == (pepsy.PauliFault(1, 0, "Z"),)

    sim.run()
    expected = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(sim.to_statevector(), expected, atol=1e-6)


def test_mps_stab_optimizer_from_stim_rejects_conflicting_inputs():
    with pytest.raises(TypeError, match="derives state and gates"):
        pepsy.MpsStabOptimizer.from_stim("H 0", gates=[])
    with pytest.raises(TypeError, match="stream_transform"):
        pepsy.MpsStabOptimizer.from_stim("H 0", stream_transform=False)


def test_stim_replay_matches_the_ordinary_mps_backend():
    torch = pytest.importorskip("torch")

    def make_optimizer():
        state = qtn.MPS_computational_state("0", dtype="complex128")
        state.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.complex128))
        return pepsy.MpsOptimizer(state, chi=2, mode="mpo")

    result = pepsy.run_stim_shots(
        make_optimizer,
        "H 0\nZ_ERROR(1) 0",
        shots=1,
        seed=1,
        run_kwargs={"progbar": False},
    )
    assert type(result.optimizers[0].p[0].data).__module__.split(".", 1)[0] == "torch"


def test_stim_else_correlated_error_requires_a_contiguous_chain():
    with pytest.raises(ValueError, match="must immediately follow"):
        pepsy.sample_stim_circuit("ELSE_CORRELATED_ERROR(1) X0", seed=1)


def test_coalesced_stim_shots_share_ideal_prefix_and_retain_measurement_counts():
    """Compiled Stim noise and ancilla-like measurements use the shared tree."""
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return pepsy.MpsStabOptimizer(1, chi=4)

    result = pepsy.run_coalesced_stim_shots(
        factory,
        "H 0\nX_ERROR(0.1) 0\nM 0\nMR 0",
        shots=128,
        seed=11,
    )

    assert calls == 1
    assert result.shots == 128
    assert result.branches == 4
    assert {bool(leaf.faults) for leaf in result.leaves} == {False, True}
    assert all(len(leaf.measurements) == 2 for leaf in result.leaves)
    assert all(leaf.measurements[0].probability == pytest.approx(0.5) for leaf in result.leaves)
    assert all(np.allclose(leaf.optimizer.to_statevector(), [1.0, 0.0]) for leaf in result.leaves)
