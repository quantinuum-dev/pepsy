"""Regression tests for the state-aware MPS layout and schedule helpers."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers.mps import MpsGateStreamSchedule

pytestmark = pytest.mark.smoke


def _stream():
    return [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 3)),
        (qu.hadamard(), (1,)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]


def test_mps_replay_layout_objective_requires_optimizer_state():
    """A static finder cannot claim a state-dependent replay measurement."""
    with pytest.raises(ValueError, match="optimizer-backed finder"):
        pepsy.MpsOptimizer.LayoutFinder(_stream(), L=4).run(
            objective="replay",
            replay_candidates=1,
        )


def test_mps_replay_layout_objective_records_transient_bond_profile():
    """Replay selection measures every requested event on private state."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = pepsy.MpsOptimizer(
        state,
        gates=_stream(),
        chi=4,
        mode="svd",
    )
    before = optimizer.to_dense()
    before_stream = tuple(optimizer._gate_stream)

    plan = optimizer.current_gate_stream_layout(
        objective="replay",
        replay_candidates=1,
        replay_steps=4,
        replay_kwargs={"cutoff": 0.0},
    )

    report = plan["stats"]["replay"]
    assert plan["objective"] == "replay"
    assert report["status"] == "ok"
    assert len(report["profile"]) == 4
    assert report["peak_bond"] == max(
        [report["initial_bond"]]
        + [record["max_bond"] for record in report["profile"]]
    )
    assert report["peak_log2_bond"] == pytest.approx(
        np.log2(report["peak_bond"])
    )
    assert optimizer._persistent_layout_plan is None
    assert tuple(optimizer._gate_stream) == before_stream
    assert np.allclose(optimizer.to_dense(), before)


def test_mps_mountain_schedule_preserves_shared_site_order_and_replays():
    """The scheduler may move disjoint gates but never shared dependencies."""
    stream = _stream()
    schedule = pepsy.MpsOptimizer.gate_stream_schedule(
        stream,
        L=4,
        layout_order="quality",
        schedule_order="mountain",
    )

    assert isinstance(schedule, MpsGateStreamSchedule)
    assert set(schedule.metadata["event_order"]) == set(range(len(stream)))
    for site in range(4):
        input_order = [
            index for index, (_gate, where) in enumerate(stream) if site in where
        ]
        scheduled_order = [
            index
            for index in schedule.metadata["event_order"]
            if site in stream[index][1]
        ]
        assert scheduled_order == input_order

    disjoint = [
        (qu.CNOT(), (0, 3)),
        (qu.hadamard(), (1,)),
        (qu.CNOT(), (3, 2)),
    ]
    disjoint_schedule = pepsy.MpsOptimizer.gate_stream_schedule(
        disjoint,
        L=4,
        layout_order="input",
        schedule_order="mountain",
    )
    assert disjoint_schedule.metadata["event_order"] == (1, 0, 2)

    scheduled = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        chi=16,
        mode="svd",
    )
    scheduled.set_gate_schedule(schedule)
    scheduled.run(progbar=False, cutoff=0.0)

    reference = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=stream,
        chi=16,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0)
    physical = np.asarray(scheduled.to_dense(logical_order=False)).reshape(
        (2,) * 4
    )
    physical_positions = [schedule.site_order.index(site) for site in range(4)]
    logical = np.transpose(physical, axes=physical_positions).reshape(-1, 1)
    assert np.allclose(logical, reference.to_dense())


def test_mps_current_gate_stream_schedule_rejects_control_events():
    """Control boundaries are not silently reordered by the gate scheduler."""
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), ("measure", "Z", 0)],
        chi=4,
        mode="svd",
    )
    with pytest.raises(ValueError, match="measurement/reset"):
        optimizer.current_gate_stream_schedule(
            layout_order="input",
            schedule_order="mountain",
        )


def test_mps_gate_stream_schedule_tracks_direct_cap_lifetime():
    """Compiled cap schedules shift later physical positions after removal."""
    stream = [
        (qu.hadamard(), (0,)),
        ("cap", 1, [1.0, 0.0], "left"),
        (qu.CNOT(), (0, 1)),
    ]
    schedule = pepsy.MpsOptimizer.gate_stream_schedule(
        stream,
        L=3,
        layout_order=(2, 0, 1),
        schedule_order="mountain",
    )

    assert schedule.metadata["event_order"] == (0, 1, 2)
    assert schedule.metadata["mapped_where"] == ((1,), (2,), (1, 0))
    assert schedule.metadata["final_site_order"] == (1, 0)
    assert schedule.stream[1][0:2] == ("cap", 2)
    assert schedule.stream[2][1] == (1, 0)

    scheduled = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        chi=16,
        mode="svd",
    )
    scheduled.set_gate_schedule(schedule)
    scheduled.run(progbar=False, cutoff=0.0)

    reference = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=stream,
        chi=16,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0)
    physical = np.asarray(scheduled.to_dense(logical_order=False)).reshape(
        (2,) * scheduled.p.L
    )
    logical = np.transpose(physical, axes=[1, 0]).reshape(-1, 1)
    assert np.allclose(logical, reference.to_dense())


def test_mps_lifetime_layout_uses_roles_and_control_boundaries():
    """The generic QEC seed exposes reusable site lifetimes without reordering controls."""
    stream = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 2)),
        ("measure", "Z", 2, +1),
        ("reset", 2),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (1, 3)),
    ]
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=stream,
        chi=8,
        mode="svd",
        qubit_roles={0: "data", 1: "data", 2: "ancilla", 3: "ancilla"},
    )
    before_stream = tuple(optimizer._gate_stream)
    plan = optimizer.current_gate_stream_layout(
        order="lifetime",
        role_order=("data", "ancilla"),
    )

    usage = plan["site_usage"]
    assert plan["selected_order"] == "lifetime"
    assert usage[2]["role"] == "ancilla"
    assert usage[2]["measurement_indices"] == (2,)
    assert usage[2]["reset_indices"] == (3,)
    assert usage[2]["lifetimes"] == ((1, 2), (3, 3), (4, 4))
    assert usage[2]["lifetime_count"] == 3
    assert tuple(optimizer._gate_stream) == before_stream


def test_mps_role_grouped_layout_can_be_selected_explicitly():
    """Role grouping is available as an explicit, reproducible candidate."""
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1)), (qu.CNOT(), (2, 3))],
        chi=8,
        mode="svd",
        qubit_roles={0: "data", 1: "ancilla", 2: "data", 3: "ancilla"},
    )
    plan = optimizer.current_gate_stream_layout(
        order="role-grouped",
        role_order=("data", "ancilla"),
    )
    assert plan["selected_order"] == "role_grouped"
    assert plan["site_order"] == (0, 2, 1, 3)


def test_mps_quality_layout_uses_roles_coordinates_and_lifetimes_automatically():
    """The default static search consumes generic data/ancilla stream hints."""
    stream = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 2)),
        ("measure", "Z", 2, +1),
        ("reset", 2),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (1, 3)),
    ]
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=stream,
        chi=8,
        mode="svd",
        qubit_roles={0: "data", 1: "data", 2: "ancilla", 3: "ancilla"},
    )
    plan = optimizer.current_gate_stream_layout(
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert plan["coordinate_source"] == "provided"
    assert plan["site_coords"][2] == (0.0, 1.0)
    assert plan["qubit_roles"][2] == "ancilla"
    assert "role_interleaved" in plan["candidate_plans"]
    assert "coordinate_snake" in plan["candidate_plans"]
    assert plan["site_usage"][2]["lifetimes"] == ((1, 2), (3, 3), (4, 4))


def test_mps_raw_layout_finder_infers_graph_coordinates_and_control_lifetimes():
    """Raw streams need no optimizer or CSS-code object for generic seeds."""
    stream = [
        (qu.CNOT(), (0, 2)),
        ("measure", "Z", 2, +1),
        ("reset", 2),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (1, 3)),
    ]
    finder = pepsy.MpsOptimizer.LayoutFinder(stream, L=4)
    plan = finder.run(order="graph_embedding")

    assert plan["coordinate_source"] == "interaction_graph"
    assert set(plan["inferred_site_coords"]) == set(range(4))
    assert plan["site_usage"][2]["measurement_indices"] == (1,)
    assert plan["site_usage"][2]["reset_indices"] == (2,)
    assert plan["site_usage"][2]["lifetimes"] == ((0, 1), (2, 2), (3, 3))


def test_mps_smart_layout_replays_candidates_from_the_initial_state():
    """The smart objective makes initial-state-aware replay opt-in and clear."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = pepsy.MpsOptimizer(
        state,
        gates=_stream(),
        chi=8,
        mode="svd",
        qubit_roles={0: "data", 1: "data", 2: "ancilla", 3: "ancilla"},
    )
    before = optimizer.to_dense()
    plan = optimizer.current_gate_stream_layout(
        objective="smart",
        replay_candidates=2,
        replay_steps=3,
        replay_kwargs={"cutoff": 0.0},
    )

    assert plan["objective"] == "replay"
    assert plan["stats"]["replay"]["status"] == "ok"
    assert np.allclose(optimizer.to_dense(), before)


def test_mps_smart_layout_can_reorder_an_entangled_private_initial_state():
    """Smart replay pays the one-time private reorder for an entangled MPS."""
    state = qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=13)
    optimizer = pepsy.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 3)), (qu.CNOT(), (1, 2))],
        chi=8,
        mode="svd",
    )
    plan = optimizer.current_gate_stream_layout(
        objective="smart",
        replay_candidates=1,
        replay_steps=1,
        replay_kwargs={"cutoff": 0.0},
    )

    assert plan["requested_objective"] == "smart"
    assert plan["stats"]["replay"]["status"] == "ok"


def test_mps_replay_layout_plan_executes_the_selected_gate_order():
    """The joint pilot must make its winning schedule executable."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = pepsy.MpsOptimizer(
        state.copy(),
        gates=_stream(),
        chi=16,
        mode="svd",
    )
    plan = optimizer.current_gate_stream_layout(
        objective="replay",
        replay_candidates=1,
        replay_kwargs={"cutoff": 0.0},
    )

    report = plan["stats"]["replay"]
    assert plan["replay_schedule"] == "mountain"
    assert tuple(plan["replay_event_order"]) == tuple(report["event_order"])
    assert tuple(
        entry[1] for entry in plan["scheduled_stream"]
    ) == tuple(_stream()[index][1] for index in report["event_order"])

    reference = pepsy.MpsOptimizer(
        state.copy(),
        gates=_stream(),
        chi=16,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0)
    optimizer.run(layout=plan, layout_report=False, progbar=False, cutoff=0.0)
    assert np.allclose(optimizer.to_dense(), reference.to_dense())


def test_mps_replay_layout_objective_accepts_direct_caps():
    """State-aware layout replay profiles and executes a direct cap."""
    stream = [
        (qu.hadamard(), (0,)),
        ("cap", 1, [1.0, 0.0], "left"),
        (qu.CNOT(), (0, 1)),
    ]
    state = qtn.MPS_computational_state("000", dtype="complex128")
    optimizer = pepsy.MpsOptimizer(
        state.copy(),
        gates=stream,
        chi=16,
        mode="svd",
    )
    plan = optimizer.current_gate_stream_layout(
        order="input",
        objective="replay",
        replay_candidates=1,
        replay_kwargs={"cutoff": 0.0},
    )

    report = plan["stats"]["replay"]
    assert report["status"] == "ok"
    assert [record["event_type"] for record in report["profile"]] == [
        "gate",
        "cap",
        "gate",
    ]
    assert report["profile"][1]["allocated_length"] == 2

    reference = pepsy.MpsOptimizer(
        state.copy(),
        gates=stream,
        chi=16,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0)
    optimizer.run(layout=plan, layout_report=False, progbar=False, cutoff=0.0)
    assert np.allclose(
        np.asarray(optimizer.to_dense()).reshape(-1),
        np.asarray(reference.to_dense()).reshape(-1),
    )


def test_mps_replay_layout_schedules_gate_segments_around_controls():
    """Controls stay fixed while each surrounding gate segment is optimized."""
    stream = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        ("measure", "Z", 2, +1),
        (qu.CNOT(), (0, 3)),
        ("reset", 1),
        (qu.CNOT(), (3, 2)),
    ]
    optimizer = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=stream,
        chi=16,
        mode="svd",
    )
    plan = optimizer.current_gate_stream_layout(
        objective="replay",
        replay_candidates=1,
        replay_kwargs={"cutoff": 0.0},
    )
    report = plan["stats"]["replay"]
    event_order = tuple(report["event_order"])

    assert report["status"] == "ok"
    assert report["schedule_strategy"] == "mountain"
    assert event_order[2] == 2
    assert event_order[4] == 4
    assert set(event_order[:2]) == {0, 1}
    assert set(event_order[3:4]) == {3}
    assert set(event_order[5:]) == {5}

    reference = pepsy.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=stream,
        chi=16,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0, seed=7)
    optimizer.run(
        layout=plan,
        layout_report=False,
        progbar=False,
        cutoff=0.0,
        seed=7,
    )
    assert [record[:2] for record in optimizer.measurements] == [
        ("Z", (2,)),
    ]
    assert np.allclose(optimizer.to_dense(), reference.to_dense())


def test_mps_replay_measure_early_moves_only_disjoint_prefix_suffix():
    """The opt-in early policy crosses disjoint gates, never shared gates."""
    stream = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        ("measure", "Z", 0, +1),
    ]
    state = qtn.MPS_computational_state("00", dtype="complex128")
    optimizer = pepsy.MpsOptimizer(
        state.copy(),
        gates=stream,
        chi=8,
        mode="svd",
    )
    plan = optimizer.current_gate_stream_layout(
        objective="replay",
        replay_candidates=1,
        replay_schedule="measure-early",
        replay_kwargs={"cutoff": 0.0},
    )
    event_order = tuple(plan["stats"]["replay"]["event_order"])
    assert event_order.index(0) < event_order.index(2)
    assert event_order.index(2) < event_order.index(1)

    reference = pepsy.MpsOptimizer(
        state.copy(),
        gates=stream,
        chi=8,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0, seed=7)
    optimizer.run(
        layout=plan,
        layout_report=False,
        progbar=False,
        cutoff=0.0,
        seed=7,
    )
    assert [record[:2] for record in optimizer.measurements] == [("Z", (0,))]
    assert np.allclose(optimizer.to_dense(), reference.to_dense())
