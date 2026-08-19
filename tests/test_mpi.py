"""Focused tests for MPI shot orchestration without requiring mpi4py."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy


class _FakeComm:
    def __init__(self, rank=0, size=1, shared=None):
        self.rank = rank
        self.size = size
        self.shared = {} if shared is None else shared

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def bcast(self, value, root=0):
        index = self.shared.setdefault("broadcast_index", {}).setdefault(
            self.rank, 0
        )
        self.shared.setdefault("broadcast_values", {})
        if self.rank == root:
            self.shared["broadcast_values"][index] = value
        self.shared["broadcast_index"][self.rank] = index + 1
        if self.rank == root:
            return value
        return self.shared["broadcast_values"][index]

    def allreduce(self, value, op=None):
        return value

    def gather(self, value, root=0):
        return [value] if self.rank == root else None


class _SeedProbeOptimizer:
    def __init__(self):
        self._rng = None
        self.measurements = ()
        self.value = None

    def set_gates(self, gates):
        self.gates = tuple(gates)

    def run(self, **kwargs):
        del kwargs
        self.value = float(self._rng.random())


def _probe_factory():
    return _SeedProbeOptimizer()


def _run_probe(comm, shots, *, seed=23, local_workers=1):
    return pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=comm,
    ).run(
        shots,
        seed=seed,
        retain="final",
        local_workers=local_workers,
        local_backend="thread" if local_workers > 1 else "serial",
    )


def test_mpi_runner_is_lazy_and_reduces_local_observables():
    result = _run_probe(_FakeComm(), 6)

    assert result.shots == 6
    assert result.retain == "final"
    assert result.local_shots == 6
    assert tuple(result.shot_range) == tuple(range(6))
    estimate = result.reduce_mean(lambda optimizer: optimizer.value)
    expected = np.mean([optimizer.value for optimizer in result.local_result.optimizers])
    assert estimate == pytest.approx(expected)
    assert result.reduce_sum(4.5) == pytest.approx(4.5)
    assert np.array_equal(result.reduce_sum([1, 2]), np.asarray([1, 2]))
    assert len(result.rank_diagnostics) == 1
    assert result.rank_diagnostics[0].rank == 0
    assert result.rank_diagnostics[0].local_shots == 6
    assert result.rank_diagnostics[0].elapsed_seconds >= 0.0


def test_run_mpi_shots_convenience_entry_point():
    result = pepsy.run_mpi_shots(
        _probe_factory,
        [(np.eye(2), 0)],
        shots=2,
        comm=_FakeComm(),
        seed=5,
        retain="final",
    )

    assert result.local_shots == 2
    assert result.local_result is not None


@pytest.mark.parametrize(
    "mode",
    ("dmrg", "dmrg1", "dmrg2", "dmrg3", "fit", "mix", "mpo", "svd", "swap", "perm", "exact", "su"),
)
def test_mps_optimizer_run_mpi_keyword_covers_all_modes(mode):
    initial = qtn.MPS_computational_state("0", dtype="complex128")
    optimizer = pepsy.MpsOptimizer(
        initial,
        [(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 0)],
        chi=4,
        mode=mode,
    )

    result = optimizer.run(
        shots=2,
        seed=41,
        mpi=_FakeComm(),
        workers=1,
        progress=False,
        retain="final",
        run_kwargs={"progbar": False},
    )

    assert isinstance(result, pepsy.MPIShotResult)
    assert len(result.local_result.optimizers) == 2
    np.testing.assert_allclose(optimizer.p.to_dense(), initial.to_dense())


def test_mps_stabilizer_run_mpi_keyword_is_fresh_and_seeded():
    optimizer = pepsy.MpsStabOptimizer(1, gates=[("x", 0)])
    result = optimizer.run(
        shots=3,
        seed=42,
        mpi=_FakeComm(),
        workers=1,
        progress=False,
        retain="final",
    )

    assert isinstance(result, pepsy.MPIShotResult)
    assert len(result.local_result.optimizers) == 3
    assert optimizer.measurements == []


def test_mps_run_auto_workers_is_available_without_mpi():
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0", dtype="complex128"),
        [(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 0)],
        chi=4,
        mode="mpo",
    )

    result = optimizer.run(
        shots=3,
        seed=43,
        strategy="independent",
        workers="auto",
        progress=False,
        retain="final",
        run_kwargs={"progbar": False},
    )

    assert len(result.optimizers) == 3


def test_mps_run_progress_is_one_outer_shot_bar(monkeypatch):
    import importlib

    mpi_module = importlib.import_module("pepsy.optimizers.mpi")
    events = []

    class _Progress:
        def update(self, amount):
            events.append(("update", amount))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(
        mpi_module,
        "_make_progress_bar",
        lambda mode, total, **kwargs: _Progress(),
    )
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0", dtype="complex128"),
        [(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 0)],
        chi=4,
        mode="mpo",
    )

    optimizer.run(
        shots=4,
        seed=44,
        strategy="independent",
        workers=2,
        progress=True,
        retain="none",
        run_kwargs={"progbar": True},
    )

    assert sum(event[1] for event in events if event[0] == "update") == 4
    assert events[-1] == ("close",)


def test_mps_run_mpi_keyword_preserves_checkpoint_options(tmp_path):
    checkpoint = tmp_path / "mps-api"
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0", dtype="complex128"),
        [(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 0)],
        chi=4,
        mode="mpo",
    )

    first = optimizer.run(
        shots=4,
        seed=45,
        mpi=_FakeComm(),
        workers=1,
        progress=False,
        retain="final",
        chunk_size=2,
        checkpoint_path=checkpoint,
    )
    resumed = optimizer.run(
        shots=4,
        seed=45,
        mpi=_FakeComm(),
        workers=1,
        progress=False,
        retain="final",
        chunk_size=2,
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert first.resumed is False
    assert resumed.resumed is True
    assert resumed.reduce_mean(lambda state: state.p.norm()) == pytest.approx(
        first.reduce_mean(lambda state: state.p.norm())
    )
    resumed.cleanup_checkpoints()


def test_mpi_reduces_vector_observables_across_retained_states():
    result = _run_probe(_FakeComm(), 3)
    estimate = result.reduce_mean(
        lambda optimizer: np.asarray([optimizer.value, 2.0 * optimizer.value])
    )
    expected = np.mean(
        [
            [optimizer.value, 2.0 * optimizer.value]
            for optimizer in result.local_result.optimizers
        ],
        axis=0,
    )
    np.testing.assert_allclose(estimate, expected)


def test_mpi_normalizes_the_thread_backend_alias():
    result = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    ).run(2, seed=5, retain="final", local_backend="threads")
    assert result.local_shots == 2


def test_mpi_explicit_communicators_can_work_without_mpi4py(monkeypatch):
    import importlib

    mpi_module = importlib.import_module("pepsy.optimizers.mpi")

    def missing_mpi():
        raise ImportError("mpi4py intentionally unavailable")

    monkeypatch.setattr(mpi_module, "_load_mpi", missing_mpi)
    result = mpi_module.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    ).run(1, seed=6, retain="final")
    assert result.local_shots == 1


def test_mpi_without_a_communicator_reports_the_optional_dependency(monkeypatch):
    import importlib

    mpi_module = importlib.import_module("pepsy.optimizers.mpi")

    def missing_mpi():
        raise ImportError("mpi4py intentionally unavailable")

    monkeypatch.setattr(mpi_module, "_load_mpi", missing_mpi)
    with pytest.raises(ImportError, match="mpi4py"):
        mpi_module.MPIShotRunner(_probe_factory, [(np.eye(2), 0)])


def test_mpi_preflight_synchronizes_validation_errors():
    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    )
    with pytest.raises(pepsy.MPIShotError, match="nonnegative integer"):
        runner.run(-1, seed=5)


def test_mpi_diagnostics_can_be_disabled():
    result = _run_probe(_FakeComm(), 2, seed=5)
    assert result.rank_diagnostics
    disabled = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    ).run(2, seed=5, collect_diagnostics=False)
    assert disabled.rank_diagnostics == ()


def test_mpi_streaming_checkpoint_resume(tmp_path):
    checkpoint = tmp_path / "mpi-shots"
    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    )
    calls = 0

    def fail_after_two_chunks(optimizer):
        nonlocal calls
        calls += 1
        if calls > 4:
            raise RuntimeError("intentional checkpoint interruption")
        return optimizer.value

    with pytest.raises(pepsy.MPIShotError, match="checkpoint interruption"):
        runner.run(
            8,
            seed=12,
            observable=fail_after_two_chunks,
            chunk_size=2,
            checkpoint_path=checkpoint,
        )

    assert (checkpoint.parent / f"{checkpoint.name}.rank0.pkl").exists()
    resumed = runner.run(
        8,
        seed=12,
        observable=lambda optimizer: optimizer.value,
        chunk_size=2,
        checkpoint_path=checkpoint,
        resume=True,
    )
    fresh = runner.run(
        8,
        seed=12,
        observable=lambda optimizer: optimizer.value,
        chunk_size=2,
    )
    assert resumed.resumed is True
    assert resumed.checkpoint_path == str(checkpoint)
    assert resumed.reduce_mean() == pytest.approx(fresh.reduce_mean())

    with pytest.raises(pepsy.MPIShotError, match="metadata mismatch"):
        runner.run(
            9,
            seed=12,
            observable=lambda optimizer: optimizer.value,
            chunk_size=2,
            checkpoint_path=checkpoint,
            resume=True,
        )

    corrupt = tmp_path / "corrupt"
    corrupt_file = corrupt.parent / f"{corrupt.name}.rank0.pkl"
    corrupt_file.write_bytes(b"not a pickle")
    with pytest.raises(pepsy.MPIShotError, match="checkpoint"):
        runner.run(
            8,
            seed=12,
            observable=lambda optimizer: optimizer.value,
            chunk_size=2,
            checkpoint_path=corrupt,
            resume=True,
        )


def test_mpi_retained_optimizer_checkpoint_resume_and_retention(tmp_path):
    checkpoint = tmp_path / "retained-shots"
    calls = 0

    def failing_factory():
        nonlocal calls
        calls += 1
        if calls > 4:
            raise RuntimeError("intentional retained checkpoint interruption")
        return _probe_factory()

    with pytest.raises(pepsy.MPIShotError, match="retained checkpoint interruption"):
        pepsy.MPIShotRunner(
            failing_factory,
            [(np.eye(2), 0)],
            comm=_FakeComm(),
        ).run(
            8,
            seed=18,
            retain="final",
            chunk_size=2,
            checkpoint_path=checkpoint,
            checkpoint_keep=3,
        )

    snapshots = sorted(checkpoint.parent.glob(f"{checkpoint.name}.rank0.step*.pkl"))
    assert len(snapshots) == 3
    (checkpoint.parent / f"{checkpoint.name}.rank0.pkl").write_bytes(
        b"corrupt latest checkpoint"
    )
    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    )
    resumed = runner.run(
        8,
        seed=18,
        retain="final",
        chunk_size=2,
        checkpoint_path=checkpoint,
        checkpoint_keep=3,
        resume=True,
    )
    fresh = runner.run(8, seed=18, retain="final")
    assert resumed.resumed is True
    assert len(resumed.local_result.optimizers) == 8
    assert resumed.reduce_mean(lambda optimizer: optimizer.value) == pytest.approx(
        fresh.reduce_mean(lambda optimizer: optimizer.value)
    )
    assert resumed.rank_diagnostics[0].resumed is True
    assert len(
        list(checkpoint.parent.glob(f"{checkpoint.name}.rank0.step*.pkl"))
    ) == 3
    resumed.cleanup_checkpoints()
    assert not list(checkpoint.parent.glob(f"{checkpoint.name}.rank0*.pkl"))


def test_mpi_checkpoint_api_rejects_unsupported_modes(tmp_path):
    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    )
    with pytest.raises(pepsy.MPIShotError, match="strategy='independent'"):
        runner.run(
            2,
            seed=5,
            strategy="coalesced",
            retain="final",
            checkpoint_path=tmp_path / "coalesced",
        )
    with pytest.raises(pepsy.MPIShotError, match="retain='final' or 'all'"):
        runner.run(
            2,
            seed=5,
            retain="none",
            checkpoint_path=tmp_path / "none",
        )


def test_mpi_record_gathering_requires_explicit_record_retention():
    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    )
    final = runner.run(2, seed=4, retain="final")
    with pytest.raises(ValueError, match="retain='all'"):
        final.gather_records()

    all_records = runner.run(2, seed=4, retain="all")
    assert all_records.gather_records() == ((), ())


def test_mpi_runner_materializes_a_gate_generator_for_reuse():
    gates = (entry for entry in [(np.eye(2), 0)])
    runner = pepsy.MPIShotRunner(_probe_factory, gates, comm=_FakeComm())

    first = runner.run(1, seed=8, retain="final")
    second = runner.run(1, seed=8, retain="final")

    assert first.local_result.optimizers[0].value == pytest.approx(
        second.local_result.optimizers[0].value
    )


def test_mpi_runner_accepts_a_compiled_plan_for_both_runner_paths():
    plan = pepsy.compile_trajectory_stream([(np.eye(2), 0)])
    runner = pepsy.MPIShotRunner(_probe_factory, plan, comm=_FakeComm())

    trajectory = runner.run(1, seed=3, retain="final")
    noisy = runner.run(
        1,
        seed=3,
        error_model=pepsy.PauliErrorModel.bit_flip(0.0),
        retain="final",
    )

    assert trajectory.local_shots == noisy.local_shots == 1


def test_global_shot_seeds_are_invariant_to_rank_partitioning():
    serial = _run_probe(_FakeComm(rank=0, size=1), 8)
    shared = {}
    rank_zero = _run_probe(_FakeComm(rank=0, size=2, shared=shared), 8)
    rank_one = _run_probe(_FakeComm(rank=1, size=2, shared=shared), 8)

    serial_values = [optimizer.value for optimizer in serial.local_result.optimizers]
    distributed_values = [
        optimizer.value for optimizer in rank_zero.local_result.optimizers
    ] + [optimizer.value for optimizer in rank_one.local_result.optimizers]
    assert distributed_values == pytest.approx(serial_values)


def test_mpi_runner_can_use_existing_local_thread_backend():
    result = _run_probe(_FakeComm(), 8, local_workers=2)
    assert result.local_shots == 8
    assert len(result.local_result.optimizers) == 8


def test_mpi_runner_supports_rank_local_coalesced_batches():
    result = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [(np.asarray([[0.0, 1.0], [1.0, 0.0]]), 0)],
        comm=_FakeComm(),
    ).run(
        8,
        seed=9,
        error_model=pepsy.PauliErrorModel.bit_flip(1.0),
        strategy="coalesced",
        retain="final",
    )

    assert result.strategy == "coalesced"
    assert result.local_result.coalesced is True
    assert sum(result.local_result.counts) == 8


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_mpi_importance_reduction_matches_unbiased_result_estimate(strategy):
    identity = np.eye(2)
    flip = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", 0.99, identity), ("X", 0.01, flip))
    )
    policy = pepsy.ImportanceSamplingPolicy({0: {"I": 0.5, "X": 0.5}})
    result = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(1, chi=4),
        [pepsy.TrajectoryEvent(channel, 0)],
        comm=_FakeComm(),
    ).run(
        256,
        seed=19,
        strategy=strategy,
        importance_sampling=policy,
        retain="all",
    )
    values = [int(records[0].label == "X") for records in result.local_result.records]
    expected = result.local_result.estimate(values)
    by_optimizer = {
        id(optimizer): value
        for optimizer, value in zip(result.local_result.optimizers, values)
    }
    actual = result.reduce_mean(lambda optimizer: by_optimizer[id(optimizer)])
    assert actual == pytest.approx(expected)


def test_mpi_runner_streams_observables_in_bounded_chunks():
    runner = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    )
    streamed = runner.run(
        8,
        seed=12,
        observable=lambda optimizer: optimizer.value,
        chunk_size=2,
    )

    assert streamed.local_result is None
    assert streamed.retain == "none"
    expected = np.mean(
        [
            optimizer.value
            for optimizer in _run_probe(_FakeComm(), 8, seed=12).local_result.optimizers
        ]
    )
    assert streamed.reduce_mean() == pytest.approx(expected)
    with pytest.raises(TypeError, match="already contains"):
        streamed.reduce_mean(lambda optimizer: optimizer.value)


def test_mpi_result_rejects_post_run_observables_without_retained_states():
    result = pepsy.MPIShotRunner(
        _probe_factory,
        [(np.eye(2), 0)],
        comm=_FakeComm(),
    ).run(2, seed=12)

    with pytest.raises(ValueError, match="streaming reduction"):
        result.reduce_mean(lambda optimizer: optimizer.value)


def test_mpi_failure_is_surfaced_as_one_error():
    def failing_factory():
        raise RuntimeError("factory failed")

    with pytest.raises(pepsy.MPIShotError, match="factory failed"):
        pepsy.MPIShotRunner(
            failing_factory,
            [(np.eye(2), 0)],
            comm=_FakeComm(),
        ).run(1, seed=5)


@pytest.mark.parametrize(
    ("factory", "run_kwargs", "expected_type"),
    [
        (
            lambda: pepsy.MpsOptimizer(
                qtn.MPS_computational_state("0"), chi=4, mode="mpo"
            ),
            {"progbar": False},
            pepsy.MpsOptimizer,
        ),
        (
            lambda: pepsy.MpsStabOptimizer(1, chi=4),
            {},
            pepsy.MpsStabOptimizer,
        ),
        (
            lambda: pepsy.TreeOptimizer(None, n=1, chi=4, run=False),
            {"progbar": False},
            pepsy.TreeOptimizer,
        ),
        (
            lambda: pepsy.TreeStabOptimizer(1),
            {},
            pepsy.TreeStabOptimizer,
        ),
    ],
)
def test_mpi_runner_uses_the_common_factory_contract(
    factory, run_kwargs, expected_type
):
    gate = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    result = pepsy.MPIShotRunner(
        factory,
        [(gate, 0)],
        comm=_FakeComm(),
    ).run(2, seed=17, run_kwargs=run_kwargs, retain="final")

    assert all(isinstance(optimizer, expected_type) for optimizer in result.local_result.optimizers)
