"""Contracts for MPS report formatting and timing summaries."""

from copy import deepcopy

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer
from pepsy.optimizers.mps.diagnostics import _summarize_fit_timing


pytestmark = [pytest.mark.core, pytest.mark.mps]


@pytest.mark.parametrize("mode", ["direct", "mpo", "quimb-direct", "fit", "dmrg2",
                                 "dmrg3", "mix", "svd", "swap", "perm", "exact", "batch-exact"])
def test_public_progress_style_matches_standalone_native_bar(mode, monkeypatch):
    created = []

    class Bar:
        def __init__(self, **options):
            created.append(options)

        def update(self, n):
            pass

        def set_postfix(self, values):
            pass

        def close(self):
            pass

    monkeypatch.setattr("tqdm.tqdm", Bar)
    state = qtn.MPS_computational_state("00", dtype="complex128")
    opt = MpsOptimizer(state, [(qu.pauli("X"), (0,))], chi=2, mode=mode)
    opt.run(progbar=True)
    assert len(created) == 1
    style = MpsOptimizer.progress_style(mode)
    assert style == {key: created[0][key] for key in ("ascii", "colour")}
    style["colour"] = "changed"
    assert MpsOptimizer.progress_style(mode)["colour"] == created[0]["colour"]


@pytest.mark.parametrize("mode", ["direct", "dm", "zipup", "svd", "swap", "perm",
                                 "dmrg", "dmrg2", "dmrg3", "mix", "exact", "exact-batch"])
def test_caller_owned_progress_receives_native_gate_updates_without_changing_replay(mode):
    class Display:
        def __init__(self):
            self.n = 0
            self.updates = []
            self.postfixes = []
            self.closes = 0

        def update(self, n):
            self.n += n
            self.updates.append(n)

        def set_postfix(self, values):
            self.postfixes.append(dict(values))

        def close(self):
            self.closes += 1

    state = qtn.MPS_computational_state("0000", dtype="complex128")
    stream = [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2)), (qu.pauli("Y"), (1,))]
    quiet = MpsOptimizer(state, stream, chi=2, mode=mode)
    visible = MpsOptimizer(state, stream, chi=2, mode=mode)
    display = Display()
    for _ in range(2):
        quiet.run(cutoff=0., n_iter=2)
        visible.run(progbar=display, cutoff=0., n_iter=2)
        np.testing.assert_allclose(visible.to_dense(), quiet.to_dense(), atol=1e-12)
        assert visible.logical_order == quiet.logical_order
        assert visible.info_c == quiet.info_c
        assert visible.p.exponent == quiet.p.exponent
    assert display.n == 2 * len(stream)
    assert display.closes == 0
    if mode != "exact-batch":
        assert {"~F", "bond" if mode == "mix" else "bnd"} <= display.postfixes[-1].keys()
    display.close()
    assert display.closes == 1


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
