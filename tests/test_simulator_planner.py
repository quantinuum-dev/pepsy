"""Focused tests for cross-simulator static planning."""

import pytest

import pepsy
from pepsy.optimizers import (
    SimulatorCandidate,
    SimulatorPlan,
    SimulatorPlanner,
    recommend_simulator,
)


def test_planner_prices_actual_dressed_supports_and_ranks_four_candidates():
    """Clifford-heavy work should expose and benefit from its dressed support."""
    stream = [("h", 0)] * 300 + [
        ("cnot", 0, 1),
        ("cnot", 1, 2),
        ("rz", 0.3, 2),
    ]

    advice = recommend_simulator(stream, n_qubits=3, chi=4)

    assert isinstance(advice, SimulatorPlan)
    assert advice.recommended == "MpsStabOptimizer"
    assert advice.best is advice.candidates[0]
    assert {candidate.optimizer for candidate in advice.candidates} == {
        "MpsOptimizer",
        "TreeOptimizer",
        "MpsStabOptimizer",
        "TreeStabOptimizer",
    }
    assert all(
        isinstance(candidate, SimulatorCandidate)
        for candidate in advice.candidates
    )
    assert advice.frame_events[-1]["support"] == (0, 1, 2)
    assert advice.candidate("MpsStabOptimizer").max_geometry == 3
    assert advice.candidates[0].relative_score == pytest.approx(1.0)
    assert advice["analysis"].clifford_entries == 302


def test_planner_exposes_chain_windows_and_tree_steiner_sizes():
    """Geometry diagnostics should use each optimizer's routed structure."""
    stream = [
        ("cnot", 0, 4),
        ("cnot", 0, 1),
        ("cnot", 0, 2),
        ("cnot", 0, 3),
    ]

    advice = SimulatorPlanner(stream, n_qubits=5, chi=8).plan()
    mps = advice.candidate("MpsOptimizer")
    tree = advice.candidate("TreeOptimizer")

    assert mps.geometry == "mps"
    assert tree.geometry == "tree"
    assert mps.multi_site_events == tree.multi_site_events == 4
    assert mps.max_geometry >= 2
    assert tree.max_geometry >= 2
    assert mps.layout["kind"] == "mps_gate_stream_layout"
    assert tree.layout.n == 5
    assert tree.layout_report["n_qubits"] == 5


def test_planner_marks_unprepassable_stabilizer_streams_unavailable():
    """A dynamic-width stream should retain only safely priced candidates."""
    stream = [("h", 0), ("cap", (0,), "left")]

    advice = SimulatorPlanner(stream, n_qubits=2, chi=4).recommend()

    assert advice.candidate("MpsOptimizer").applicable
    assert advice.candidate("TreeOptimizer").applicable
    assert not advice.candidate("MpsStabOptimizer").applicable
    assert not advice.candidate("TreeStabOptimizer").applicable
    assert any("cap changes" in warning for warning in advice.warnings)


def test_planner_validates_pricing_inputs_and_candidate_lookup():
    """Planner-owned settings and typed candidate lookup should be explicit."""
    with pytest.raises(TypeError, match="positive integer"):
        SimulatorPlanner([("t", 0)], chi=None)
    with pytest.raises(ValueError, match="planner-owned"):
        SimulatorPlanner(
            [("t", 0)],
            chi=4,
            tree_layout_kwargs={"chi": 8},
        )
    with pytest.raises(ValueError, match="circuit width"):
        SimulatorPlanner([], chi=4)

    advice = pepsy.recommend_simulator([], n_qubits=2, chi=4)
    with pytest.raises(KeyError, match="Unknown simulator"):
        advice.candidate("DenseSimulator")
