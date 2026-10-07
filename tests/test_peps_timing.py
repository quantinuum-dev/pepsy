"""Optional profiling must preserve PEPS evolution and failure semantics."""

import json

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.fitting.local import FIT
from pepsy.optimizers.peps import PepsOptimizer


def make_optimizer():
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex128", seed=61)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    return PepsOptimizer(
        state, [(gate, ((0, 0), (0, 1)))], chi=1,
        optimizer_options={"maxeval": 4},
        sweep_optimize_kwargs={"n_round_trips": 1},
    )


def test_timing_preserves_refinement_and_reports_phases():
    pytest.importorskip("nlopt")
    plain, timed = make_optimizer(), make_optimizer()
    plain.run(infidelity_tol=0)
    streamed = []
    timed.run(infidelity_tol=0, timing=True, step_callback=streamed.append)
    np.testing.assert_allclose(timed.state.to_dense(), plain.state.to_dense(), atol=1e-12)
    np.testing.assert_allclose(timed.get_local_infidelities(), plain.get_local_infidelities(), atol=1e-12)
    report = timed.get_timing()
    assert report["status"] == "complete"
    assert report["synchronized"] is False
    assert report["total_seconds"] > 0
    for name in ("target", "compression", "normalization", "fidelity", "sweep"):
        assert report["seconds"][name] > 0
        assert report["calls"][name] >= 1
    record = timed.get_step_records()[0]
    assert streamed == timed.get_step_records()
    streamed[0]["timing"]["seconds"]["target"] = -1
    assert record["timing"]["seconds"]["target"] > 0
    assert record["optimizer_attempted"]
    assert record["timing"]["calls"]["sweep"] == 1
    inner = record["optimizer_result"]["timing"]
    assert inner["slices"]
    assert inner["boundary_seconds"] >= 0
    assert inner["optimize_seconds"] > 0
    assert inner["synchronized"] is False
    json.dumps(report)
    json.dumps(record)
    report["seconds"]["target"] = -1
    assert timed.get_timing()["seconds"]["target"] > 0
    assert plain.get_timing() == {}
    assert "timing" not in plain.get_step_records()[0]


def test_disabled_timing_never_synchronizes_and_run_totals_reset(monkeypatch):
    syncs = []
    monkeypatch.setattr(FIT, "synchronize_backend", lambda state: syncs.append(state.max_bond()))
    optimizer = make_optimizer()
    optimizer.run(optimize=False, timing=False, timing_sync_device=True)
    assert not syncs
    optimizer.run(optimize=False, timing=True, timing_sync_device=True)
    assert syncs
    assert optimizer.get_timing()["synchronized"] is True
    assert optimizer.get_timing()["calls"]["target"] == 1
    optimizer.run(optimize=False, timing=True)
    assert optimizer.get_timing()["calls"]["target"] == 1
    optimizer.run(optimize=False)
    assert optimizer.get_timing() == {}


def test_failed_phase_is_retained_and_active_timer_is_cleared(monkeypatch):
    optimizer = make_optimizer()

    def fail(*args, **kwargs):
        raise RuntimeError("target construction failed")

    monkeypatch.setattr(optimizer, "_build_target", fail)
    with pytest.raises(RuntimeError, match="target construction failed"):
        optimizer.run(timing=True)
    timing = optimizer.get_timing()
    assert timing["status"] == "failed"
    assert timing["calls"]["target"] == 1
    assert timing["seconds"]["target"] >= 0
    assert optimizer._phase_timer is None


def test_callback_keeps_completed_batches_before_a_later_failure(monkeypatch):
    optimizer = make_optimizer()
    optimizer.set_gates(optimizer.gates * 2)
    original = optimizer._build_target
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second batch failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_build_target", fail_second)
    streamed = []
    with pytest.raises(RuntimeError, match="second batch failed"):
        optimizer.run(optimize=False, timing=True, step_callback=streamed.append)
    assert len(streamed) == len(optimizer.get_step_records()) == 1
    assert streamed[0]["timing"]["calls"]["target"] == 1
    assert optimizer.get_timing()["status"] == "failed"
