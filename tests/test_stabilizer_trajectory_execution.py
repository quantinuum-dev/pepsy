"""MPS/STN trajectory scheduling and stabilizer-frame probability contracts."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import build_backend
from pepsy.optimizers import MpsOptimizer, StabilizerMpsSimulator, noise
from pepsy.optimizers.mps import _trajectory_execution as execution

pytest.importorskip("stim")


@pytest.fixture(params=["mps", "stabilizer"])
def engine_kind(request):
    return request.param


def _engine(kind, stream, *, converter=None, dtype="complex128"):
    if kind == "stabilizer":
        return StabilizerMpsSimulator(
            4, chi=2, mode="direct", dtype=dtype, to_backend=converter,
        ).compile(stream)
    state = qtn.MPS_computational_state("0000", dtype=dtype)
    if converter:
        state.apply_to_arrays(converter)
    return MpsOptimizer(state, stream, chi=2, mode="direct")


def _budget(sim, capacity):
    p = sim.p
    state_bytes = execution._estimated_state_bytes(p, sim.chi, sim.mode)
    if isinstance(sim, StabilizerMpsSimulator):
        state_bytes += 2 * sim.n * (2 * sim.n + 1) + 64 * len(sim._gate_stream)
    return (32 << 20) + (8 + 2 * capacity) * state_bytes


def _dense(sim):
    if isinstance(sim, StabilizerMpsSimulator):
        return sim.to_statevector()
    return ar.to_numpy(sim.to_dense()).ravel()


@pytest.mark.parametrize("workers", [1, 2])
def test_both_engines_continue_prefix_once_and_preserve_states(engine_kind, workers, monkeypatch):
    stream = [("h", 2), ("cnot", 2, 3)] + [("x_error", .01, 0)] * 4
    sim = _engine(engine_kind, stream)
    original = noise._run_trajectory_entries
    prefix_calls = []

    def count(opt, entries, kwargs, **options):
        if len(entries) == 2:
            prefix_calls.append(opt)
        return original(opt, entries, kwargs, **options)

    monkeypatch.setattr(noise, "_run_trajectory_entries", count)
    result = sim.run(shots=16, seed=1134, max_branches=4,
                     workers=workers, progress=False)
    assert result.diagnostics.continued_from_cap
    assert result.diagnostics.planned_strategy == "coalesced"
    assert "shared prefixes" in result.diagnostics.fallback_reason
    assert len(prefix_calls) == 1
    assert sum(result.counts) == 16
    for leaf in result.raw.leaves:
        assert [record.event_index for record in leaf.records] == [2, 3, 4, 5]
        parity = sum(record.label == "X" for record in leaf.records) % 2
        expected = np.zeros(16)
        expected[8 * parity] = expected[8 * parity + 3] = 2**-.5
        np.testing.assert_allclose(abs(_dense(leaf.optimizer)),
                                   expected, atol=1e-12)


@pytest.mark.parametrize("retain", ["all", "final"])
def test_retention_budget_fails_before_allocating_shots(engine_kind, retain, monkeypatch):
    sim = _engine(engine_kind, [("x_error", .01, 0)] * 4)
    budget = _budget(sim, 4)
    monkeypatch.setattr(sim, "_shot_factory", lambda: pytest.fail("allocated a shot"))
    with pytest.raises(MemoryError, match="requires 16"):
        sim.run(shots=16, strategy="independent", retain=retain,
                memory_budget=budget, progress=False)


def test_continuation_respects_memory_and_streamed_retention(engine_kind):
    sim = _engine(engine_kind, [("x_error", .01, 0)] * 4)
    kwargs = dict(shots=16, seed=1134, memory_budget=_budget(sim, 4), progress=False)
    with pytest.raises(MemoryError, match="requires 16"):
        sim.run(**kwargs)
    result = sim.run(retain="none", **kwargs)
    assert result.shots == 16
    assert result.optimizers == ()
    assert result.diagnostics.continued_from_cap
    assert result.diagnostics.memory_max_branches == 4


def test_auto_cpu_workers_share_policy_and_explicit_overrides(engine_kind):
    sim = _engine(engine_kind, [("x_error", .01, 0)] * 4)
    result = sim.run(shots=8, progress=False)
    assert result.diagnostics.workers == 1
    assert "shared prefixes" in result.diagnostics.execution_reason
    explicit = sim.run(shots=8, workers=2, progress=False)
    assert explicit.diagnostics.workers == 2
    assert explicit.diagnostics.execution_reason == "explicit worker override"


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "bad"])
def test_invalid_memory_budget_is_rejected(engine_kind, budget):
    with pytest.raises(ValueError, match="memory_budget"):
        _engine(engine_kind, [("x_error", .1, 0)]).run(
            shots=4, memory_budget=budget, progress=False,
        )


def test_memory_query_is_shared_and_zero_shots_need_no_capacity(engine_kind, monkeypatch):
    sim = _engine(engine_kind, [("x_error", .1, 0)])
    calls = []

    def available(array):
        calls.append(array)
        return 2 * _budget(sim, 4)

    monkeypatch.setattr(execution, "available_device_memory", available)
    result = sim.run(shots=2, memory_budget="auto", progress=False)
    assert len(calls) == 1
    assert result.diagnostics.memory_max_branches == 4
    assert sim.run(shots=0, memory_budget=1, progress=False).shots == 0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("backend", ["numpy", "torch", "cuda"])
def test_stabilizer_kraus_matches_dense_with_shared_expectations(dtype, backend, monkeypatch):
    converter = None
    if backend != "numpy":
        torch = pytest.importorskip("torch")
        if backend == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        converter = build_backend(
            device="cuda" if backend == "cuda" else "cpu",
            dtype=getattr(torch, dtype), set_default=False,
        )
    sim = _engine("stabilizer", [], converter=converter, dtype=dtype)
    sim.compile([("h", 0), ("cnot", 0, 3), ("ry", .43, 0)]).run()
    channel = noise.TrajectoryChannel.amplitude_damping(.23)
    dense = _dense(sim).reshape(2, -1)
    expected = np.asarray([np.linalg.norm(outcome.gate @ dense)**2
                           for outcome in channel.outcomes])
    expected /= expected.sum()
    original = sim._pauli_expectation
    evaluations = []

    def counted(terms, sign):
        evaluations.append((terms, sign))
        return original(terms, sign)

    monkeypatch.setattr(sim, "_pauli_expectation", counted)
    actual = noise._kraus_probabilities(sim, channel, (0,))
    np.testing.assert_allclose(actual, expected, atol=3e-6 if dtype == "complex64" else 1e-12)
    assert len(evaluations) == 2  # I and Z shared by both Kraus Gram operators.
    assert sim.backend_info()["dtype"] == dtype


def test_stabilizer_rare_kraus_mass_is_not_pruned_or_scaled_to_zero():
    sim = StabilizerMpsSimulator(1, chi=2, mode="direct")
    sim.compile([("x", 0)]).run()
    # A tiny but valid parent norm must not erase an even rarer channel outcome.
    sim.p[0].modify(data=sim.p[0].data * 1e-145)
    channel = noise.TrajectoryChannel.amplitude_damping(1e-20)
    probabilities = noise._kraus_probabilities(sim, channel, (0,))
    assert probabilities[1] == pytest.approx(1e-20, rel=1e-12, abs=0.)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_stabilizer_reversed_two_site_kraus_probabilities_with_layout(backend):
    converter = None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        converter = build_backend(device="cpu", dtype=torch.complex128, set_default=False)
    sim = StabilizerMpsSimulator(4, chi=4, to_backend=converter, exact_cooling=False)
    sim.apply_layout((2, 0, 3, 1), layout_report=False)
    sim.compile([
        ("h", 0), ("cnot", 0, 3), ("ry", .43, 0),
        ("rxx", .31, 0, 2), ("rz", .28, 3),
    ]).run()
    rng = np.random.default_rng(451)
    basis, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    channel = noise.TrajectoryChannel.kraus([
        (str(i), np.outer(basis[:, i], basis[:, i].conj())) for i in range(4)
    ])
    block = sim.to_statevector().reshape((2,) * 4).transpose(3, 0, 1, 2).reshape(4, -1)
    expected = np.asarray([np.linalg.norm(outcome.gate @ block)**2
                           for outcome in channel.outcomes])
    expected /= expected.sum()
    np.testing.assert_allclose(noise._kraus_probabilities(sim, channel, (3, 0)),
                               expected, atol=1e-12)
    assert sim.logical_order == [2, 0, 3, 1]


@pytest.mark.parametrize("strategy", ["independent", "coalesced", "auto"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_importance_sampled_rare_stabilizer_kraus_branch_normalizes(strategy, backend):
    converter = None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        converter = build_backend(device="cpu", dtype=torch.complex128, set_default=False)
    sim = StabilizerMpsSimulator(1, chi=2, to_backend=converter).compile([
        ("x", 0), ("amplitude_damping", 1e-30, 0),
    ])
    result = sim.run(
        shots=16, strategy=strategy, seed=1, progress=False,
        importance_sampling={"no_jump": .5, "jump": .5},
    )
    jumps = 0
    for optimizer, records, weight in zip(result.optimizers, result.records, result.weights):
        assert optimizer.norm() == pytest.approx(1.)
        if records[0].label == "jump":
            jumps += 1
            assert records[0].probability == pytest.approx(1e-30, rel=1e-12, abs=0.)
            assert weight == pytest.approx(2e-30, rel=1e-12, abs=0.)
            np.testing.assert_allclose(abs(optimizer.to_statevector()), [1., 0.], atol=1e-12)
    assert jumps


def test_stabilizer_macro_and_measurement_continuation_keep_classical_prefix():
    sim = StabilizerMpsSimulator(2, chi=2).compile([
        ("x", 0), ("measure_reset", "Z", 0), ("h", 1), ("measure", "Z", 1),
        ("if", -2, 1, ("x", 0)), ("measure", "Z", 0),
    ])
    raw = noise.run_coalesced_trajectory_shots(
        sim._shot_factory(), sim.compiled_stream, 16, max_branches=1,
        _continue_on_cap=True, seed=52,
    )
    assert raw.diagnostics.continued_from_cap
    for leaf in raw.leaves:
        assert [record.outcome for record in leaf.optimizer.measurements][::2] == [-1, -1]
    macro = StabilizerMpsSimulator(1, chi=2).compile([("x", 0)] * 4).run(
        shots=16, seed=1134, max_branches=2, progress=False,
        error_model=noise.PauliErrorModel(p_x=.01),
    )
    assert macro.diagnostics.continued_from_cap
    for leaf in macro.raw.leaves:
        expected = [1., 0.] if len(leaf.faults) % 2 == 0 else [0., 1.]
        np.testing.assert_allclose(abs(_dense(leaf.optimizer)), expected, atol=1e-12)


def test_stabilizer_mpi_rejects_explicit_budget_and_caps_auto_accelerator_workers(monkeypatch):
    from pepsy.optimizers import mpi

    sim = StabilizerMpsSimulator(2, chi=2).compile([("x_error", .1, 0)])
    with pytest.raises(ValueError, match="only for local shots"):
        sim.run(shots=4, mpi=True, memory_budget=1000, progress=False)
    monkeypatch.setattr(sim, "backend_info", lambda: {
        "backend": "torch", "dtype": "complex128", "device": "cuda:0",
    })

    class Runner:
        def __init__(self, factory, gates, comm):
            pass

        def run(self, shots, **kwargs):
            return kwargs

    monkeypatch.setattr(mpi, "MPIShotRunner", Runner)
    assert sim.run(shots=4, mpi=True, progress=False)["local_workers"] == 1
    assert sim.run(shots=4, mpi=True, workers=2, progress=False)["local_workers"] == 2
