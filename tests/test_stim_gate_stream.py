"""Unsampled Stim translation, analysis and prepared engine shot replay."""

import pickle

import numpy as np
import pytest

from pepsy.optimizers import (
    StabilizerMpsSimulator,
    compile_stim_circuit,
    compile_trajectory_stream,
    stim_plan_to_gate_stream,
    stim_readout_parities,
)


@pytest.fixture
def stim():
    return pytest.importorskip("stim")


def _prepare(source):
    plan = compile_stim_circuit(source)
    engine = StabilizerMpsSimulator(plan.num_qubits, chi=16, mode="direct")
    engine.compile(stim_plan_to_gate_stream(plan))
    return plan, engine


def test_queued_analysis_reuses_prepared_channels(monkeypatch):
    from pepsy.optimizers import noise

    _plan, engine = _prepare("DEPOLARIZE1(0.1) 0\nM 0")
    original = noise.compile_trajectory_stream
    observed = []

    def checked(gates, *args, **kwargs):
        assert gates is engine.compiled_stream
        result = original(gates, *args, **kwargs)
        assert result is gates
        observed.append(result)
        return result

    monkeypatch.setattr(noise, "compile_trajectory_stream", checked)
    assert engine.queued_stream_analysis().trajectory_entries == 1
    engine.queued_recommend_settings()
    assert len(observed) >= 2


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("strategy", ["auto", "coalesced"])
def test_coalesced_prefix_measurements_survive_branch_and_feedback(stim, workers, strategy):
    plan, engine = _prepare(
        "R 0 1 2\nX 0\nMR 0\nH 1\nM 1\nCX rec[-2] 2\nM 2\n"
        "DETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    result = engine.run(
        shots=32, strategy=strategy, workers=workers, seed=42, progress=False,
    )
    assert result.coalesced
    assert len(result.optimizers) >= 2
    for sim in result.optimizers:
        assert len(sim.measurements) == 3
        assert sim.measurements[0].outcome == -1
        assert sim.measurements[-1].outcome == -1
    detectors, observables = stim_readout_parities(
        plan, [sim.measurements for sim in result.optimizers],
    )
    assert np.all(detectors == 1)
    assert np.all(observables == 1)
    assert sum(result.counts) == 32


@pytest.mark.parametrize("signature,expected", [
    (("numpy", "complex128", None), (4, "thread")),
    (("torch", "complex128", "cpu"), (4, "thread")),
    (("torch", "complex64", "cuda:0"), (1, "serial")),
    (("torch", "complex64", "mps:0"), (1, "serial")),
    (("jax", "complex64", "TFRT_CPU_0"), (4, "thread")),
    (("jax", "complex64", "TFRT_GPU_0"), (1, "serial")),
    (("jax", "complex64", "TPU_0"), (1, "serial")),
    (("cupy", "complex128", "<CUDA Device 0>"), (1, "serial")),
    (("symmray", "complex64", "cuda:0", "torch"), (1, "serial")),
])
def test_shot_parallelism_uses_live_backend_signature(monkeypatch, signature, expected):
    from pepsy.optimizers import mpi
    from pepsy.optimizers.stabilizer_tn._backend import shot_parallelism

    monkeypatch.setattr(mpi, "_available_cpu_count", lambda: 8)
    assert shot_parallelism(
        signature, workers="auto", shots=4, parallel_backend="auto",
    ) == expected
    assert shot_parallelism(
        signature, workers=2, shots=4, parallel_backend="serial",
    ) == (2, "serial")
    assert shot_parallelism(
        signature, workers=2, shots=4, parallel_backend="auto",
    ) == (2, "gpu" if expected[0] == 1 else "thread")


def test_threaded_shots_initialize_stim_before_dispatch(monkeypatch):
    import threading
    from pepsy.optimizers.stabilizer_tn import mps_stab_optimizer as implementation

    original = implementation._tableau_from_exact_unitary
    threads = []

    def observe(gate):
        threads.append(threading.current_thread().name)
        return original(gate)

    engine = StabilizerMpsSimulator(1, gates=[("x_error", 0.5, 0)])
    monkeypatch.setattr(implementation, "_tableau_from_exact_unitary", observe)
    result = engine.run(shots=2, workers=2, seed=3, progress=False)
    assert result.shots == 2
    assert threads[0] == threading.current_thread().name


@pytest.mark.parametrize("strategy,workers", [
    ("independent", 1), ("independent", 2), ("coalesced", 1), ("auto", 1),
])
def test_prepared_shots_sample_fresh_noise_and_reuse_plan(stim, strategy, workers):
    plan, engine = _prepare(
        "R 0 1\nH 0\nCX 0 1\nX_ERROR(0.3) 1\nM 0 1\n"
        "DETECTOR rec[-1] rec[-2]\n"
        "OBSERVABLE_INCLUDE(2) rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-2]"
    )
    compiled = engine.compiled_stream
    before = engine.to_statevector().copy()

    def run():
        result = engine.run(shots=64, strategy=strategy, workers=workers,
                            seed=42, retain="all", progress=False)
        det, obs = stim_readout_parities(plan, result.measurements)
        assert sum(result.counts) == result.shots == 64
        assert set(det[:, 0]) == {0, 1}
        np.testing.assert_array_equal(det[:, 0], obs[:, 2])
        assert not np.any(obs[:, :2])
        assert all(len(s.measurements) == 2 for s in result.optimizers)
        assert all(abs(float(s.norm()) - 1) < 1e-10 for s in result.optimizers)
        return result.counts, det

    first, second = run(), run()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(engine.to_statevector(), before)
    assert engine.compiled_stream is compiled
    assert not engine.measurements


@pytest.mark.parametrize("name", ["X_ERROR", "Y_ERROR", "Z_ERROR", "I_ERROR"])
def test_one_qubit_grouped_noise_matches_stim_state(stim, name):
    source = f"R 0 1\nH 0\nS 0\n{name}(1) 0 1"
    _, engine = _prepare(source)
    result = engine.run(shots=1, strategy="independent", workers=1, seed=3, progress=False)
    reference = stim.TableauSimulator()
    reference.do_circuit(stim.Circuit(source))
    expected = reference.state_vector(endian="big")
    actual = result.optimizers[0].to_statevector().reshape(-1)
    assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("channel_index", range(15))
def test_two_qubit_pauli_channel_order_matches_stim(stim, channel_index):
    probs = [0] * 15
    probs[channel_index] = 1
    source = "R 0 1\nH 0\nS 0\nPAULI_CHANNEL_2(" + ",".join(map(str, probs)) + ") 0 1"
    _, engine = _prepare(source)
    result = engine.run(shots=1, strategy="independent", workers=1, progress=False)
    reference = stim.TableauSimulator()
    reference.do_circuit(stim.Circuit(source))
    expected = reference.state_vector(endian="big")
    actual = result.optimizers[0].to_statevector().reshape(-1)
    assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("noise", [
    "DEPOLARIZE1(0.2) 0 1", "DEPOLARIZE2(0.2) 0 1",
    "PAULI_CHANNEL_1(0.2,0,0) 0 1",
])
def test_translated_channels_remain_stochastic(stim, noise):
    _, engine = _prepare("R 0 1\n" + noise + "\nM 0 1")
    result = engine.run(shots=64, strategy="coalesced", workers=1, seed=42, progress=False)
    assert sum(result.counts) == 64
    assert len({tuple(m.outcome for m in s.measurements) for s in result.optimizers}) > 1


def test_hidden_reset_records_preserve_stim_offsets(stim):
    plan, engine = _prepare(
        "R 0 1\nH 0\nCX 0 1\nR 0\nM 0 1\n"
        "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    result = engine.run(shots=16, strategy="coalesced", workers=1, seed=42, progress=False)
    actual = stim_readout_parities(plan, result.measurements)
    expected = stim_readout_parities(plan, [s.measurements for s in result.optimizers])
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.any(actual[0])


def test_grouped_feedforward_and_repeat_blocks(stim):
    source = "R 0 1 2\nX 0\nM 0\nREPEAT 2 {\nCX rec[-1] 1 rec[-1] 2\n}\nM 1 2"
    _, engine = _prepare(source)
    result = engine.run(shots=4, strategy="coalesced", workers=1, seed=3, progress=False)
    for sim in result.optimizers:
        assert tuple(m.outcome for m in sim.measurements) == (-1, 1, 1)


def test_analyze_and_compile_are_separate_and_reusable(stim, monkeypatch):
    from pepsy.optimizers import noise

    gate_stream = [("h", 0), ("depolarize2", 0.1, 0, 1), ("measure", "ZZ", (0, 1))]
    plan = compile_trajectory_stream(gate_stream)
    engine = StabilizerMpsSimulator(2, chi=4, mode="direct")
    analysis = engine.analyze_stream(gate_stream, n_qubits=2)
    assert analysis.trajectory_entries == analysis.clifford_trajectory_entries == 1
    assert analysis.opaque_entries == 0
    assert analysis.touched_qubits == (0, 1)
    assert analysis.is_clifford_only
    assert engine.compiled_stream is None
    with monkeypatch.context() as patch:
        patch.setattr(noise, "_trajectory_entries", lambda _: pytest.fail("Plan reparsed"))
        assert engine.compile(plan) is engine
        assert engine.compiled_stream is plan
        assert engine.analyze_stream(plan, n_qubits=2) == analysis
        assert engine.set_gates(plan).compiled_stream is plan
    assert engine.queued_stream_analysis() == analysis
    assert not engine.measurements
    # Prepared plans survive process-worker serialization.
    restored = pickle.loads(pickle.dumps(plan))
    assert restored.trajectory_indices == plan.trajectory_indices


def test_kraus_analysis_knows_support_but_is_not_clifford(stim):
    analysis = StabilizerMpsSimulator.analyze_stream([("amplitude_damping", 0.2, 1)], n_qubits=2)
    assert analysis.trajectory_entries == 1
    assert analysis.clifford_trajectory_entries == analysis.opaque_entries == 0
    assert analysis.touched_qubits == (1,)
    assert not analysis.is_clifford_only
    with pytest.raises(ValueError, match="outside"):
        StabilizerMpsSimulator.analyze_stream([("x_error", 0.2, 2)], n_qubits=2)


def test_thread_workers_receive_the_same_prepared_plan(stim, monkeypatch):
    from pepsy.optimizers import noise

    _, engine = _prepare("R 0\nX_ERROR(0.2) 0\nM 0")
    observed = []
    original = noise.run_trajectory_shots

    def record_plan(factory, gates, *args, **kwargs):
        observed.append(gates)
        return original(factory, gates, *args, **kwargs)

    monkeypatch.setattr(noise, "run_trajectory_shots", record_plan)
    result = engine.run(shots=4, strategy="independent", workers=2, seed=7, progress=False)
    assert result.shots == 4
    assert len(observed) == 5  # outer run and four worker calls
    assert all(plan is engine.compiled_stream for plan in observed)


def test_failed_preparation_preserves_the_previous_queue(stim):
    _, engine = _prepare("R 0\nH 0")
    previous = engine.compiled_stream
    before = engine.to_statevector().copy()
    rng_before = repr(engine._rng.bit_generator.state)
    with pytest.raises(ValueError, match="probability"):
        engine.compile([("x_error", -0.1, 0)])
    assert engine.compiled_stream is previous
    np.testing.assert_array_equal(engine.to_statevector(), before)
    assert repr(engine._rng.bit_generator.state) == rng_before


@pytest.mark.parametrize("source", ["M(0.1) 0", "M !0", "MPP !X0", "MPAD 0 1"])
def test_compiler_rejects_unrepresented_readout_semantics(stim, source):
    with pytest.raises(NotImplementedError):
        compile_stim_circuit(source)


@pytest.mark.parametrize("source", ["E(0.1) X0", "HERALDED_ERASE(0.1) 0"])
def test_native_only_noise_is_rejected_by_gate_stream_lowering(stim, source):
    plan = compile_stim_circuit(source)
    with pytest.raises(NotImplementedError, match="run_stim_shots"):
        stim_plan_to_gate_stream(plan)


def test_readout_shapes_and_empty_annotations(stim):
    plan = compile_stim_circuit("R 0\nDETECTOR\nOBSERVABLE_INCLUDE(2)")
    assert compile_stim_circuit(plan) is plan
    det, obs = stim_readout_parities(plan, [[], []])
    assert det.shape == (2, 1)
    assert obs.shape == (2, 3)
    assert not np.any(det) and not np.any(obs)
    det, obs = stim_readout_parities(plan, [])
    assert det.shape == (0, 1) and obs.shape == (0, 3)
    with pytest.raises(TypeError, match="StimCircuitPlan"):
        stim_plan_to_gate_stream("R 0")
    with pytest.raises(ValueError, match="unavailable measurement"):
        stim_readout_parities(compile_stim_circuit("M 0\nDETECTOR rec[-1]"), [[]])
