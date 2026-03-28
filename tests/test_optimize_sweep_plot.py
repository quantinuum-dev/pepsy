"""Tests for PEPSSweepOptimizer plotting helpers."""

import pytest

from pepsy.optimize_sweep import PEPSSweepOptimizer, SweepResult


def _sample_runs():
    return [
        {
            "loss_final": 0.9,
            "global_loss_after": 0.8,
            "global_fidelity_after": 0.2,
            "state_norm": 1.2,
            "bdy_norm_norm": 1.1,
            "bdy_norm_overlap": 1.3,
            "time_boundary": 0.4,
            "time_optimize": 0.6,
        },
        {
            "loss_final": 0.7,
            "global_loss_after": 0.6,
            "global_fidelity_after": 0.4,
            "state_norm": 1.05,
            "bdy_norm_norm": 1.02,
            "bdy_norm_overlap": 1.01,
            "time_boundary": 0.5,
            "time_optimize": 0.4,
        },
        {
            "loss_final": 0.5,
            "global_loss_after": 0.3,
            "global_fidelity_after": 0.7,
            "state_norm": 1.0,
            "bdy_norm_norm": 1.0,
            "bdy_norm_overlap": 1.0,
            "time_boundary": 0.45,
            "time_optimize": 0.35,
        },
    ]


def test_plot_runs_supports_requested_metrics_and_aliases():
    """plot_runs should accept core metric names and common aliases."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    runs = _sample_runs()
    fig, axes = PEPSSweepOptimizer.plot_runs(
        runs,
        metrics=["loss", "norm_peps", "bdy_overlap", "timing"],
        cumulative=False,
        show=False,
    )
    assert len(axes) == 4
    assert fig is not None


def test_plot_runs_cumulative_timing_grows_monotonically():
    """cumulative=True should monotonically accumulate timing-style series."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    runs = _sample_runs()
    _fig, axes = PEPSSweepOptimizer.plot_runs(
        runs,
        metrics=["time_total"],
        cumulative=True,
        show=False,
    )
    y = axes[0].lines[0].get_ydata()
    assert y[0] > 0
    assert y[1] >= y[0]
    assert y[2] >= y[1]


def test_plot_runs_rejects_unknown_metric():
    """Unknown metrics should fail with a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        PEPSSweepOptimizer.plot_runs(_sample_runs(), metrics=["not-a-metric"], show=False)


def test_instance_plot_uses_last_sweep_result_runs():
    """plot() should reuse cached latest sweep result when runs are omitted."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    opt = object.__new__(PEPSSweepOptimizer)
    opt._last_axis_runs = None  # pylint: disable=protected-access
    opt._last_sweep_result = SweepResult(  # pylint: disable=protected-access
        runs=_sample_runs(),
        fidelity_before=None,
        fidelity_after=None,
        loss_before=None,
        loss_after=None,
    )

    fig, axes = opt.plot(metrics=["loss", "time_total"], show=False)
    assert fig is not None
    assert len(axes) == 2


def test_plot_runs_uses_auto_log_for_most_metrics():
    """Auto scaling should log-scale loss/timing and keep fidelity linear."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    runs = _sample_runs()
    _fig, axes = PEPSSweepOptimizer.plot_runs(
        runs,
        metrics=["loss", "fidelity", "time_total"],
        show=False,
    )
    assert axes[0].get_yscale() == "log"
    assert axes[1].get_yscale() == "linear"
    assert axes[2].get_yscale() == "log"


def test_plot_runs_can_disable_log_scale():
    """log_scale=False should force linear axes for all metrics."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    runs = _sample_runs()
    _fig, axes = PEPSSweepOptimizer.plot_runs(
        runs,
        metrics=["loss", "time_total"],
        log_scale=False,
        show=False,
    )
    assert axes[0].get_yscale() == "linear"
    assert axes[1].get_yscale() == "linear"
