"""Contracts for MPS report formatting and timing summaries."""

from copy import deepcopy

import pytest

from pepsy.optimizers import MpsOptimizer
from pepsy.optimizers.mps.diagnostics import _summarize_fit_timing


pytestmark = [pytest.mark.core, pytest.mark.mps]


def test_fit_summary_preserves_call_counts_and_overlapping_phase_totals():
    """A repeated FIT call and a nested phase must not inflate the totals."""
    records = [
        {
            "fit_index": 0, "site_count": 2, "elapsed_seconds": 4.0,
            "canonicalization_seconds": 2.0,
            "moving_canonicalization_seconds": 0.5,
        },
        {"fit_index": 0, "site_timings": [{}, {}], "elapsed_seconds": 3.0},
        {"fit_index": 1, "site_count": 3, "elapsed_seconds": 1.0},
    ]
    original = deepcopy(records)

    result = _summarize_fit_timing(iter(records))

    assert result["calls"] == 2
    assert result["sweeps"] == 3
    assert result["site_updates"] == 7
    assert result["elapsed_seconds"] == 8.0
    assert result["canonicalization_seconds"] == 2.0
    assert result["moving_canonicalization_seconds"] == 0.5
    assert result["svd_seconds"] == 0.0
    assert records == original
    assert all(value == 0 for value in _summarize_fit_timing([]).values())


@pytest.mark.parametrize("objective", ["compression", "replay"])
def test_layout_report_keeps_objective_specific_measurements(objective):
    """Replay shows the graph proxy, while compression shows operator loads."""
    plan = {
        "objective": objective,
        "input_stats": {"loss": 8.0},
        "stats": {
            "loss": 4.0, "static_loss": 2.0,
            "max_operator_cut_load": 3, "total_operator_cut_load": 5,
            "rank_bounded_cuts": 2,
            "replay": {"status": "ok", "peak_bond": 4,
                       "peak_log2_bond": 2, "profile": [{}, {}]},
        },
    }
    original = deepcopy(plan)

    report = MpsOptimizer._layout_report_text(plan)

    if objective == "compression":
        assert "score: 8 -> 4 (50.0% lower)" in report
        assert "operator cut load max/total: 3/5 | bounded cut probes: 2" in report
    else:
        assert "graph proxy score: 8 -> 2 (75.0% lower)" in report
        assert "replay peak bond/log2: 4/2 | profiled events: 2" in report
    assert plan == original
    assert MpsOptimizer._layout_report_text({}) is None


def test_layout_report_preserves_subclass_formatters():
    """Delegating a report must still use the subclass's formatting hooks."""
    class CustomReport(MpsOptimizer):
        @staticmethod
        def _format_layout_value(value):
            return f"[{value}]"

        @classmethod
        def _format_layout_reduction(cls, before, after):
            return "custom reduction"

    report = CustomReport._layout_report_text({"input_stats": {"loss": 1}})
    assert "score: custom reduction" in report
    assert "event span max/mean: [0]/[0.0] -> [0]/[0.0]" in report
