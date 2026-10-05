"""Automatic trajectory scheduling must preserve the requested simulation."""

from types import SimpleNamespace

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer, noise
from pepsy.optimizers.mps import _trajectory_execution as execution


def test_worker_policy_uses_backend_circuit_and_thread_budget(monkeypatch):
    from pepsy.optimizers import mpi

    monkeypatch.setattr(mpi, "_available_cpu_count", lambda: 16)
    monkeypatch.setattr(execution, "_cpu_inner_threads", lambda backend: 4)
    state = SimpleNamespace(max_bond=lambda: 64)
    entries = [("cnot", 0, 1)] * 8

    def workers(info, shots=32, strategy="independent", gates=entries):
        return execution._automatic_shot_workers(info, state, gates, shots, strategy)[0]

    assert workers({"backend": "numpy"}) == 4
    assert workers({"backend": "torch", "device": "cpu"}) == 4
    assert workers({"backend": "numpy"}, shots=2) == 1
    assert workers({"backend": "numpy"}, gates=entries[:2]) == 1
    assert workers({"backend": "numpy"}, strategy="coalesced") == 1
    for info in (
        {"backend": "torch", "device": "cuda:0"},
        {"backend": "torch", "device": "mps:0"},
        {"backend": "cupy", "device": "<CUDA Device 0>"},
        {"backend": "jax", "device": "cuda:0"},
        {"backend": "jax", "device": "TFRT_CPU_0"},
        {"backend": "symmray", "array_backend": "cupy"},
    ):
        assert workers(info) == 1
    monkeypatch.setattr(execution, "_cpu_inner_threads", lambda backend: 32)
    assert workers({"backend": "numpy"}) == 1
    monkeypatch.setattr(execution, "_cpu_inner_threads", lambda backend: None)
    assert workers({"backend": "numpy"}) == 1


def test_large_cpu_auto_parallel_replay_matches_serial(monkeypatch):
    from pepsy.optimizers import mpi

    threadpoolctl = pytest.importorskip("threadpoolctl")
    monkeypatch.setattr(mpi, "_available_cpu_count", lambda: 8)
    with threadpoolctl.threadpool_limits(limits=1):
        initial = qtn.MPS_rand_state(12, 64, dtype="complex128", seed=891)
        sim = MpsOptimizer(initial, [("x_error", .5, 0)] * 8, chi=64)
        kwargs = dict(shots=4, seed=19, max_branches=2, progbar=False, progress=False)
        reference = sim.run(strategy="independent", workers=1, **kwargs)
        result = sim.run(**kwargs)
        assert result.diagnostics.planned_strategy == "independent"
        assert result.diagnostics.workers == 4
        assert result.raw.records == reference.raw.records
        for actual, wanted in zip(result.optimizers, reference.optimizers):
            np.testing.assert_allclose(actual.to_dense(), wanted.to_dense(), atol=1e-12)


@pytest.mark.parametrize("workers,parallel_workers,parallel_backend,expected", [
    ("auto", 1, "thread", 1), (2, 1, "thread", 2),
    ("auto", 2, "thread", 2), (2, 1, "serial", 1),
])
def test_public_worker_overrides_preserve_seeded_replay(
    workers, parallel_workers, parallel_backend, expected,
):
    initial = qtn.MPS_computational_state("0", dtype="complex128")
    sim = MpsOptimizer(initial, [("x_error", .3, 0)], chi=2)
    kwargs = dict(shots=16, seed=713, strategy="independent", progbar=False, progress=False)
    reference = sim.run(workers=1, **kwargs)
    result = sim.run(workers=workers, parallel_workers=parallel_workers,
                     parallel_backend=parallel_backend, **kwargs)
    assert result.diagnostics.workers == expected
    assert result.diagnostics.planned_strategy == "independent"
    assert result.diagnostics.fallback_reason is None
    assert result.raw.records == reference.raw.records
    for actual, wanted in zip(result.optimizers, reference.optimizers):
        np.testing.assert_allclose(actual.to_dense(), wanted.to_dense(), atol=1e-12)
    np.testing.assert_allclose(sim.to_dense(), initial.to_dense())


def test_rare_mixture_planning_does_not_evaluate_proposals():
    stream = [("x_error", 1e-4, 0)] * 10
    options = dict(max_branches=8)
    assert noise._resolve_auto_parallel_strategy(stream, 64, **options) == "coalesced"
    assert noise._resolve_auto_parallel_strategy(
        [("x_error", .5, 0)] * 10, 64, **options,
    ) == "independent"

    def proposal(*args):
        raise AssertionError("Planning evaluated a state-dependent proposal")

    assert noise._resolve_auto_parallel_strategy(
        stream, 64, importance_sampling=proposal, **options,
    ) == "independent"
    assert noise._resolve_auto_parallel_strategy(
        stream, 64, importance_sampling={"I": .5, "X": .5}, **options,
    ) == "independent"


@pytest.mark.parametrize("workers", [1, 2])
def test_auto_importance_replay_matches_independent_weights(workers):
    initial = qtn.MPS_computational_state("0", dtype="complex128")
    sim = MpsOptimizer(initial, [("x_error", 1e-4, 0)] * 6, chi=2)
    kwargs = dict(shots=16, seed=713, workers=workers, max_branches=8,
                  importance_sampling={"I": .5, "X": .5}, progbar=False, progress=False)
    reference = sim.run(strategy="independent", **kwargs)
    result = sim.run(strategy="auto", **kwargs)
    assert result.diagnostics.planned_strategy == "independent"
    assert not result.coalesced
    assert result.raw.records == reference.raw.records
    np.testing.assert_array_equal(result.raw.weights, reference.raw.weights)


@pytest.mark.parametrize("workers,expected", [("auto", 1), (2, 2)])
def test_mpi_accelerator_auto_uses_one_local_worker(monkeypatch, workers, expected):
    from pepsy.optimizers import mpi

    initial = qtn.MPS_computational_state("0", dtype="complex128")
    sim = MpsOptimizer(initial, [("x_error", .1, 0)], chi=2)
    monkeypatch.setattr(sim, "backend_info", lambda: {"backend": "torch", "device": "cuda:0"})

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, shots, **kwargs):
            assert shots == 16
            assert kwargs["strategy"] == "independent"
            assert kwargs["local_workers"] == expected
            return "dispatched"

    monkeypatch.setattr(mpi, "MPIShotRunner", Runner)
    assert sim.run(shots=16, mpi=True, workers=workers, progress=False) == "dispatched"


@pytest.mark.parametrize("stream", [
    [("x_error", .5, 0)], [("reset", 0)], [("measure", "Z", 0)],
])
def test_event_cap_applies_even_without_total_cap(stream):
    assert noise._resolve_auto_parallel_strategy(
        stream, 16, max_branches=None, max_branch_factor=1,
    ) == "independent"


def test_conditional_and_kraus_streams_use_structural_bounds():
    rare = [("x_error", 1e-4, 0)] * 10
    for suffix in (
        [("amplitude_damping", .01, 0)],
        [("measure", "Z", 0), ("if", -1, 1, ("reset", 0))],
    ):
        assert noise._resolve_auto_parallel_strategy(
            rare + suffix, 64, max_branches=8,
        ) == "independent"


@pytest.mark.parametrize("strategy", ["auto", "independent", "coalesced"])
@pytest.mark.parametrize("action", [("reset", 0), ("reset_z", 0),
                                    ("measure_reset", "Z", 0), ("mrz", 0)])
def test_conditional_resets_match_deterministic_reference(strategy, action):
    initial = qtn.MPS_computational_state("1", dtype="complex128")
    sim = MpsOptimizer(initial, [("measure", "Z", 0, -1), ("if", -1, 1, action)], chi=2)
    result = sim.run(shots=4, strategy=strategy, seed=1, progbar=False, progress=False)
    assert result.shots == 4
    for opt in result.optimizers:
        np.testing.assert_allclose(abs(opt.to_dense().ravel()), [1., 0.], atol=1e-12)


@pytest.mark.parametrize("workers", [1, 2])
def test_rare_mixture_actual_cap_overflow_matches_independent(workers):
    initial = qtn.MPS_computational_state("0", dtype="complex128")
    sim = MpsOptimizer(initial, [("x_error", .01, 0)] * 4, chi=2)
    kwargs = dict(shots=16, seed=1134, workers=workers, max_branches=4,
                  progress=False, progbar=False)
    # The expected occupancy bound is 1.64, but this seed creates five leaves.
    with pytest.raises(noise._CoalescedBranchCapExceeded):
        sim.run(strategy="coalesced", **kwargs)
    result = sim.run(strategy="auto", **kwargs)
    reference = sim.run(strategy="independent", **kwargs)
    assert result.diagnostics.planned_strategy == "coalesced"
    assert result.diagnostics.fallback_reason is not None
    assert not result.coalesced
    assert result.raw.records == reference.raw.records
    assert len(result.optimizers) == 16
    for actual, wanted in zip(result.optimizers, reference.optimizers):
        np.testing.assert_allclose(actual.to_dense(), wanted.to_dense(), atol=1e-12)


@pytest.mark.parametrize("workers", [1, 2])
def test_auto_cap_fallback_restarts_with_original_seed(monkeypatch, workers):
    initial = qtn.MPS_computational_state("0", dtype="complex128")
    sim = MpsOptimizer(initial, [("x_error", 1e-4, 0)] * 6, chi=2)
    kwargs = dict(shots=16, seed=71, workers=workers, progress=False, progbar=False)
    reference = sim.run(strategy="independent", **kwargs)
    original = noise.run_parallel_trajectory_shots
    attempts = []

    def capped(*args, **kwargs):
        attempts.append(True)
        raise noise._CoalescedBranchCapExceeded("test runtime cap")

    def parallel(*args, **kwargs):
        if kwargs["strategy"] == "coalesced":
            return capped(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(noise, "run_coalesced_trajectory_shots", capped)
    monkeypatch.setattr(noise, "run_parallel_trajectory_shots", parallel)
    result = sim.run(strategy="auto", max_branches=8, **kwargs)
    assert len(attempts) == 1
    assert not result.coalesced
    assert result.diagnostics.planned_strategy == "coalesced"
    assert result.diagnostics.fallback_reason == "coalesced branch cap exceeded"
    assert result.raw.records == reference.raw.records
    for actual, wanted in zip(result.optimizers, reference.optimizers):
        np.testing.assert_allclose(actual.to_dense(), wanted.to_dense(), atol=1e-12)


@pytest.mark.parametrize("backend", [
    "numpy",
    pytest.param("cuda", marks=[pytest.mark.optional, pytest.mark.integration]),
    pytest.param("cupy", marks=[pytest.mark.optional, pytest.mark.integration]),
])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("expected", ["coalesced", "independent"])
def test_auto_backend_replay_matches_explicit_strategy(backend, dtype, expected):
    initial = qtn.MPS_computational_state("1", dtype=dtype)
    if backend == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        initial.apply_to_arrays(lambda x: torch.tensor(x, dtype=getattr(torch, dtype), device="cuda"))
    elif backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        initial.apply_to_arrays(lambda x: cp.asarray(x, dtype=dtype))
    stream = ([("x_error", 1e-4, 0)] * 6 if expected == "coalesced"
              else [("amplitude_damping", .2, 0)] * 6)
    sim = MpsOptimizer(initial, stream, chi=2)
    kwargs = dict(shots=16, seed=713, max_branches=8, progbar=False, progress=False)
    reference = sim.run(strategy=expected, workers=1, **kwargs)
    result = sim.run(**kwargs)
    assert result.diagnostics.planned_strategy == expected
    assert result.diagnostics.workers == 1
    assert result.coalesced == (expected == "coalesced")
    assert result.counts == reference.counts
    assert result.shots == 16
    for actual, wanted in zip(result.optimizers, reference.optimizers):
        np.testing.assert_allclose(ar.to_numpy(actual.to_dense()), ar.to_numpy(wanted.to_dense()),
                                   atol=1e-6 if dtype == "complex64" else 1e-12)
        assert actual.backend_info() == sim.backend_info()
        assert float(np.linalg.norm(ar.to_numpy(actual.to_dense()))) == pytest.approx(1., abs=1e-6)
