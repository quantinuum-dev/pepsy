"""Tests for SweepOptimizer APIs."""

from types import SimpleNamespace

import quimb.tensor as qtn
import pytest

import pepsy.optimize_sweep as sweep_mod
from pepsy.optimize_sweep import SweepOptimizer


def test_run_wrapper_maps_global_style_arguments(monkeypatch):
    """run() should forward canonical args into the global sweep runner."""
    opt = object.__new__(SweepOptimizer)
    opt.optimize_kwargs = {
        "axes": ("y", "x"),
        "n_round_trips": 2,
        "optimizer": "scipy",
        "optimizer_options": {"algorithm": "LBFGS"},
        "env_n_iter": 10,
    }
    captured = {}

    def _fake_run_global(self, **kwargs):  # pylint: disable=unused-argument
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(SweepOptimizer, "_run_global_sweeps", _fake_run_global)

    out = SweepOptimizer.run(
        opt,
        n_cycles=5,
        chi=24,
        progress=False,
        renormalize=True,
    )

    assert out == "ok"
    assert captured["n_cycles"] == 5
    assert captured["chi"] == 24
    assert captured["solver"] == "scipy"
    assert captured["solver_options"] == {"algorithm": "LBFGS"}
    assert captured["progress"] is False
    assert captured["track_boundary_fidelity"] is False
    assert captured["renormalize"] is True


def test_run_wrapper_rejects_alias_arguments():
    """run() should only accept canonical argument names."""
    opt = object.__new__(SweepOptimizer)
    with pytest.raises(TypeError):
        SweepOptimizer.run(opt, n=3)
    with pytest.raises(TypeError):
        SweepOptimizer.run(opt, solver="nlopt")
    with pytest.raises(TypeError):
        SweepOptimizer.run(opt, solver_options={"algorithm": "LD_VAR2"})
    with pytest.raises(TypeError):
        SweepOptimizer.run(opt, pbar=False)


def test_run_progress_postfix_reports_compact_cost_and_timing(monkeypatch):
    """Global progress postfix should include compact timing and cost diagnostics."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2
    opt.optimize_kwargs = {}
    opt.bdy = SimpleNamespace(norm=1.0)
    opt.bdy_overlap = SimpleNamespace(norm=1.0)

    class _FakeTQDM:  # pylint: disable=too-few-public-methods
        instances = []

        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.postfix = {}
            self.postfix_str = ""
            self.description_str = ""
            self.updated = 0
            _FakeTQDM.instances.append(self)

        def update(self, n=1):
            self.updated += int(n)

        def set_postfix(self, data):
            self.postfix = dict(data)

        def set_postfix_str(self, text):
            self.postfix_str = str(text)

        def set_description_str(self, text):
            self.description_str = str(text)

        def close(self):
            return None

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        renormalize,
    ):  # pylint: disable=unused-argument
        run_info = {
            "axis": axis,
            "sweep": "forward",
            "index": 0,
            "loss_final": 1.23e-4,
            "time_boundary": 0.42,
            "time_optimize": 0.84,
            "flops": 7.0,
            "peak_norm": 12.0,
            "peak_overlap": 13.5,
        }
        run_callback(run_info)
        return [run_info]

    monkeypatch.setattr(sweep_mod, "tqdm", _FakeTQDM)
    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)
    monkeypatch.setattr(SweepOptimizer, "_ensure_boundary_chi", lambda self, chi: None)
    monkeypatch.setattr(SweepOptimizer, "_approx_infidelity_loss", lambda self, env_n_iter=4: 0.0)
    monkeypatch.setattr(SweepOptimizer, "_collect_axis_run_traces", lambda self, axis_runs, cycle, axis: None)

    _ = SweepOptimizer.run(
        opt,
        n_cycles=1,
        progress=True,
        renormalize=False,
        track_boundary_fidelity=False,
    )

    assert _FakeTQDM.instances
    bar = _FakeTQDM.instances[-1]
    desc = bar.description_str
    postfix = bar.postfix_str
    assert "loss=" in desc
    assert "[best:" in desc
    assert "e" in desc  # scientific notation
    assert "t=" in postfix
    assert "cost=(" in postfix


def test_set_optimize_kwargs_uses_clear_canonical_keys():
    """set_optimize_kwargs should store only the explicit optimize keys."""
    opt = object.__new__(SweepOptimizer)
    opt.optimize_kwargs = {}
    SweepOptimizer.set_optimize_kwargs(
        opt,
        n_cycles=4,
        chi=40,
        optimizer="scipy",
        optimizer_options={"algorithm": "LBFGS"},
        progress=False,
        track_boundary_fidelity=True,
    )
    assert opt.optimize_kwargs["n_cycles"] == 4
    assert opt.optimize_kwargs["chi"] == 40
    assert opt.optimize_kwargs["optimizer"] == "scipy"
    assert opt.optimize_kwargs["optimizer_options"] == {"algorithm": "LBFGS"}
    assert opt.optimize_kwargs["progress"] is False
    assert opt.optimize_kwargs["track_boundary_fidelity"] is True


@pytest.mark.parametrize(
    ("debug", "fidel_arg", "expected_fidel"),
    [
        (False, None, False),
        (True, None, True),
        (False, True, True),
        (True, False, False),
    ],
)
def test_optimize_global_boundary_fidel_flag(debug, fidel_arg, expected_fidel, monkeypatch):
    """Boundary track_boundary_fidelity should default to debug unless explicitly overridden."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    captured = {}

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        captured["track_boundary_fidelity"] = track_boundary_fidelity
        captured["debug"] = debug
        captured["debug_loss_mode"] = debug_loss_mode
        captured["debug_loss_kwargs"] = debug_loss_kwargs
        return []

    def _fake_metrics(self):  # pylint: disable=unused-argument
        return 0.9, 0.1

    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)
    monkeypatch.setattr(SweepOptimizer, "metrics", _fake_metrics)

    kwargs = dict(
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        solver_options=None,
        env_n_iter=1,
        progress=False,
        debug=debug,
        renormalize=False,
    )
    if fidel_arg is not None:
        kwargs["track_boundary_fidelity"] = fidel_arg

    _ = SweepOptimizer._run_global_sweeps(opt, **kwargs)

    assert captured["debug"] is debug
    assert captured["track_boundary_fidelity"] is expected_fidel
    assert captured["debug_loss_mode"] == "exact"
    assert captured["debug_loss_kwargs"] is None


def test_optimize_global_non_debug_skips_metrics(monkeypatch):
    """When debug=False, optimize_global should not call exact metrics()."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        return []

    def _boom_metrics(self):  # pylint: disable=unused-argument
        raise AssertionError("metrics() should not be called when debug=False")

    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)
    monkeypatch.setattr(SweepOptimizer, "metrics", _boom_metrics)

    _ = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=False,
        track_boundary_fidelity=True,
        renormalize=False,
    )


def test_optimize_global_non_debug_uses_infidelity_for_before_after(monkeypatch):
    """debug=False should populate before/after via boundary infidelity."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    calls = {"infidelity": []}

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        return []

    def _fake_infidelity(self, **kwargs):  # pylint: disable=unused-argument
        calls["infidelity"].append(dict(kwargs))
        values = [0.25, 0.10]
        return values[len(calls["infidelity"]) - 1]

    def _boom_metrics(self):  # pylint: disable=unused-argument
        raise AssertionError("metrics() should not be called when debug=False")

    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)
    monkeypatch.setattr(SweepOptimizer, "infidelity", _fake_infidelity)
    monkeypatch.setattr(SweepOptimizer, "metrics", _boom_metrics)

    out = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=7,
        progress=False,
        debug=False,
        debug_loss_kwargs={"chi": 32},
        track_boundary_fidelity=False,
        renormalize=False,
    )

    assert len(calls["infidelity"]) == 2
    assert calls["infidelity"][0]["chi"] == 32
    assert calls["infidelity"][0]["n_iter"] == 7
    assert calls["infidelity"][0]["progress"] is False
    assert calls["infidelity"][0]["track_boundary_fidelity"] is False
    assert out.loss_before == pytest.approx(0.25)
    assert out.loss_after == pytest.approx(0.10)


def test_optimize_global_applies_chi_before_sweeps(monkeypatch):
    """run(chi=...) should expand boundaries before running sweeps."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    calls = {}

    def _fake_ensure(self, chi):  # pylint: disable=unused-argument
        calls["chi"] = chi

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        return []

    monkeypatch.setattr(SweepOptimizer, "_ensure_boundary_chi", _fake_ensure)
    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)
    monkeypatch.setattr(SweepOptimizer, "_approx_infidelity_loss", lambda self, **kwargs: 0.5)

    _ = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        chi=64,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=False,
        track_boundary_fidelity=False,
        renormalize=False,
    )

    assert calls["chi"] == 64


def test_set_chi_expands_boundaries_and_optionally_normalizes(monkeypatch):
    """set_chi should call boundary expansion and optional normalize."""
    opt = object.__new__(SweepOptimizer)
    calls = {"ensure": None, "normalize": None}

    def _fake_ensure(self, chi):  # pylint: disable=unused-argument
        calls["ensure"] = chi

    def _fake_normalize(self, **kwargs):  # pylint: disable=unused-argument
        calls["normalize"] = dict(kwargs)
        return 1.0

    monkeypatch.setattr(SweepOptimizer, "_ensure_boundary_chi", _fake_ensure)
    monkeypatch.setattr(SweepOptimizer, "normalize", _fake_normalize)

    out = SweepOptimizer.set_chi(
        opt,
        96,
        normalize_state=True,
        n_iter=7,
        direction="x",
        max_separation=2,
        progress=True,
        track_boundary_fidelity=True,
    )

    assert out is opt
    assert calls["ensure"] == 96
    assert calls["normalize"] == {
        "n_iter": 7,
        "direction": "x",
        "max_separation": 2,
        "progress": True,
        "track_boundary_fidelity": True,
    }


def test_optimize_global_collects_step_loss_and_step_timing_when_not_debug(monkeypatch):
    """Trace collection should run per step when debug=False."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        runs = [
            {
                "axis": axis,
                "sweep": "forward",
                "index": 0,
                "loss_final": 0.4,
                "time_boundary": 0.1,
                "time_optimize": 0.2,
                "history": [0.9, 0.4],
            },
            {
                "axis": axis,
                "sweep": "forward",
                "index": 1,
                "loss_final": 0.2,
                "time_boundary": 0.2,
                "time_optimize": 0.3,
                "history": [0.6, 0.2],
            },
            {
                "axis": axis,
                "sweep": "backward",
                "index": 0,
                "loss_final": 0.1,
                "time_boundary": 0.05,
                "time_optimize": 0.15,
                "history": [0.2, 0.1],
            },
        ]
        for run in runs:
            if run_callback is not None:
                run_callback(run)
        return runs

    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)

    out = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=False,
        track_boundary_fidelity=False,
        renormalize=False,
    )

    assert out.step_loss_trace == pytest.approx([0.4, 0.2, 0.1])
    assert len(out.inner_loss_traces) == 3
    assert out.inner_loss_traces[0] == pytest.approx([0.9, 0.4])
    assert out.inner_loss_traces[2] == pytest.approx([0.2, 0.1])
    assert len(out.step_trace) == 3
    assert out.step_trace[0]["axis"] == "x"
    assert out.step_trace[0]["move"] == "forward"
    assert out.step_trace[0]["time_boundary"] == pytest.approx(0.1)
    assert out.step_trace[0]["time_optimize"] == pytest.approx(0.2)
    assert out.step_trace[0]["time_total"] == pytest.approx(0.3)
    assert out.step_trace[2]["move"] == "backward"
    assert out.step_trace[2]["time_boundary"] == pytest.approx(0.05)
    assert out.step_trace[2]["time_optimize"] == pytest.approx(0.15)
    assert opt.step_loss_trace == pytest.approx([0.4, 0.2, 0.1])
    assert opt.inner_loss_traces[1] == pytest.approx([0.6, 0.2])


def test_optimize_global_resets_traces_each_call(monkeypatch):
    """New optimize_global calls should replace, not append, old traces."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    calls = {"count": 0}

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        calls["count"] += 1
        if calls["count"] == 1:
            return [{
                "axis": axis,
                "sweep": "forward",
                "index": 0,
                "loss_final": 0.5,
                "time_boundary": 0.1,
                "time_optimize": 0.2,
                "history": [0.8, 0.5],
            }]
        return [{
            "axis": axis,
            "sweep": "forward",
            "index": 0,
            "loss_final": 0.2,
            "time_boundary": 0.05,
            "time_optimize": 0.1,
            "history": [0.4, 0.2],
        }]

    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)

    _ = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=False,
        track_boundary_fidelity=False,
        renormalize=False,
    )
    assert opt.step_loss_trace == pytest.approx([0.5])

    _ = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=False,
        track_boundary_fidelity=False,
        renormalize=False,
    )
    assert opt.step_loss_trace == pytest.approx([0.2])
    assert len(opt.inner_loss_traces) == 1
    assert opt.inner_loss_traces[0] == pytest.approx([0.4, 0.2])


def test_optimize_global_debug_uses_exact_loss_per_step_trace(monkeypatch):
    """debug=True should use exact loss in per-step traces."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    def _fake_debug_loss(self, *, mode="exact", kwargs=None):  # pylint: disable=unused-argument
        return 0.33

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        runs = [
            {
                "axis": axis,
                "sweep": "forward",
                "index": 0,
                "loss_final": 0.5,
                "exact_loss_after": 0.2,
                "boundary_fidelity_norm": 0.91,
                "boundary_fidelity_overlap": 0.87,
                "time_boundary": 0.1,
                "time_optimize": 0.2,
                "history": [0.7, 0.5],
            },
            {
                "axis": axis,
                "sweep": "backward",
                "index": 0,
                "loss_final": 0.4,
                "exact_loss_after": 0.1,
                "boundary_fidelity_norm": 0.95,
                "boundary_fidelity_overlap": 0.90,
                "time_boundary": 0.05,
                "time_optimize": 0.15,
                "history": [0.6, 0.4],
            },
        ]
        for run in runs:
            if run_callback is not None:
                run_callback(run)
        return runs

    monkeypatch.setattr(SweepOptimizer, "_debug_loss", _fake_debug_loss)
    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)

    out = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=True,
        track_boundary_fidelity=True,
        renormalize=False,
    )

    assert out.step_loss_trace == pytest.approx([0.2, 0.1])
    assert out.step_trace[0]["boundary_fidelity_norm"] == pytest.approx(0.91)
    assert out.step_trace[0]["boundary_fidelity_overlap"] == pytest.approx(0.87)
    assert out.step_trace[1]["boundary_fidelity_norm"] == pytest.approx(0.95)
    assert out.step_trace[1]["boundary_fidelity_overlap"] == pytest.approx(0.90)
    assert len(out.step_trace) == 2
    assert out.step_trace[0]["move"] == "forward"
    assert out.step_trace[1]["move"] == "backward"


def test_optimize_global_debug_infidelity_mode_skips_metrics(monkeypatch):
    """debug_loss_mode='infidelity' should bypass metrics() and use infidelity()."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.Lx = 2
    opt.Ly = 2

    calls = {}

    def _fake_optimize_axis(
        self,
        axis,
        *,
        n_round_trips,
        solver,
        solver_options,
        env_n_iter,
        run_callback,
        track_boundary_fidelity,
        debug,
        debug_loss_mode,
        debug_loss_kwargs,
        renormalize,
    ):  # pylint: disable=unused-argument
        calls["axis_fidel"] = track_boundary_fidelity
        calls["axis_debug"] = debug
        calls["axis_debug_loss_mode"] = debug_loss_mode
        calls["axis_debug_loss_kwargs"] = debug_loss_kwargs
        return []

    def _fake_infidelity(self, **kwargs):  # pylint: disable=unused-argument
        calls.setdefault("infidelity_calls", []).append(dict(kwargs))
        return 0.2

    def _boom_metrics(self):  # pylint: disable=unused-argument
        raise AssertionError("metrics() should not be called in infidelity mode")

    monkeypatch.setattr(SweepOptimizer, "optimize_axis", _fake_optimize_axis)
    monkeypatch.setattr(SweepOptimizer, "infidelity", _fake_infidelity)
    monkeypatch.setattr(SweepOptimizer, "metrics", _boom_metrics)

    out = SweepOptimizer._run_global_sweeps(
        opt,
        axes=("x",),
        n_cycles=1,
        n_round_trips=1,
        solver="scipy",
        env_n_iter=1,
        progress=False,
        debug=True,
        debug_loss_mode="infidelity",
        debug_loss_kwargs={"chi": 18, "norm_target": 1.0},
        track_boundary_fidelity=None,
        renormalize=False,
    )

    assert out.loss_before == pytest.approx(0.2)
    assert out.loss_after == pytest.approx(0.2)
    assert len(calls["infidelity_calls"]) == 2
    assert calls["infidelity_calls"][0]["chi"] == 18
    assert calls["axis_fidel"] is True
    assert calls["axis_debug"] is True
    assert calls["axis_debug_loss_mode"] == "infidelity"
    assert calls["axis_debug_loss_kwargs"] == {"chi": 18, "norm_target": 1.0}


def test_infidelity_expands_boundaries_for_requested_chi(monkeypatch):
    """infidelity(chi=...) should expand both boundaries before contraction."""
    opt = object.__new__(SweepOptimizer)
    opt.state = object()
    opt.state_target = object()
    opt.contraction_opt = "auto-hq"
    opt.fit_mode = "eff"

    class _DummyBdy:
        def __init__(self, chi):
            self.chi = chi
            self.expands = []
            self.mps_b = {"Y0_l": object()}

        def expand_bnd(self, chi, inplace=True):  # pylint: disable=unused-argument
            self.expands.append(int(chi))
            self.chi = int(chi)
            return self

    opt.bdy = _DummyBdy(8)
    opt.bdy_overlap = _DummyBdy(10)

    captured = {}

    def _fake_boundary_infidelity(
        state,
        state_target,
        *,
        chi,
        norm,
        norm_target,
        bdy,
        bdy_overlap,
        contraction_opt,
        n_iter,
        direction,
        max_separation,
        progress,
        track_boundary_fidelity,
        fit_mode,
        single_layer,
    ):
        captured["chi"] = chi
        captured["norm_target"] = norm_target
        captured["bdy"] = bdy
        captured["bdy_overlap"] = bdy_overlap
        return {
            "infidelity": 0.1,
            "norm": 1.0,
            "norm_target": 1.0,
            "overlap": 1.0,
            "bdy": bdy,
            "bdy_target": None,
            "bdy_overlap": bdy_overlap,
        }

    monkeypatch.setattr(sweep_mod, "boundary_infidelity", _fake_boundary_infidelity)

    out = SweepOptimizer.infidelity(opt, chi=16, norm_target=1.0)

    assert out == pytest.approx(0.1)
    assert opt.bdy.expands == [16]
    assert opt.bdy_overlap.expands == [16]
    assert captured["chi"] == 16
    assert captured["norm_target"] == 1.0
    assert captured["bdy"] is opt.bdy
    assert captured["bdy_overlap"] is opt.bdy_overlap


def test_default_solver_options_match_expected_baseline():
    """Default solver options should expose the package baseline settings."""
    opts = SweepOptimizer.default_solver_options()
    assert opts["algorithm"] == "LBFGS"
    assert opts["lr"] == pytest.approx(1e-2)
    assert opts["n_steps"] == 50
    assert opts["maxeval"] == 100
    assert opts["ftol_rel"] == pytest.approx(1e-9)
    assert opts["xtol_rel"] == pytest.approx(1e-9)
    assert opts["patience"] == 40
    assert opts["min_steps"] == 10
    assert opts["restore_best"] is True
    assert opts["bad_max"] == 20
    assert opts["penalty_on_bad"] == pytest.approx(1e20)


def test_public_kwarg_guides_expose_supported_keys():
    """Public kwarg guide helpers should list supported API kwargs."""
    normalize_keys = SweepOptimizer.normalize_kwarg_names()
    optimize_keys = SweepOptimizer.optimize_kwarg_names()
    infidelity_keys = SweepOptimizer.infidelity_kwarg_names()
    guide = SweepOptimizer.kwarg_guide()

    assert "n_iter" in normalize_keys
    assert "optimizer" in optimize_keys
    assert "chi" in infidelity_keys
    assert guide["normalize"] == normalize_keys
    assert guide["optimize"] == optimize_keys
    assert guide["infidelity"] == infidelity_keys
    assert guide["optimizer_defaults"]["algorithm"] == "LBFGS"


def test_optimize_packed_params_uses_default_solver_options(monkeypatch):
    """Missing solver_options should fall back to SweepOptimizer defaults."""
    opt = object.__new__(SweepOptimizer)
    captured = {}

    def _fake_run(self, *, params_init, loss_fn, **kwargs):  # pylint: disable=unused-argument
        captured["params_init"] = params_init
        captured["solver"] = self.solver
        captured["solver_options"] = dict(self.options)
        captured["n_steps"] = self.n_steps
        return SimpleNamespace(params=params_init, history=[0.0])

    monkeypatch.setattr(sweep_mod.GradientOptimizer, "run", _fake_run)

    params_init = {"x": 1.0}
    out_params, history = SweepOptimizer._optimize_packed_params(
        opt,
        params_init,
        lambda _params: 0.0,
        solver="scipy",
        solver_options=None,
    )
    assert out_params == params_init
    assert history == [0.0]
    assert captured["solver"] == "scipy"
    assert captured["n_steps"] == 50
    assert captured["solver_options"]["algorithm"] == "LBFGS"
    assert captured["solver_options"]["maxeval"] == 100


@pytest.mark.parametrize(
    "solver_in",
    [
        "scipy_lbfgs",
        "nlopt_lbfgs",
    ],
)
def test_resolve_user_solver_rejects_aliases(solver_in):
    """Short solver aliases should be rejected in favor of canonical names."""
    with pytest.raises(ValueError, match="Unsupported solver"):
        SweepOptimizer._resolve_user_solver(solver_in)


def test_resolve_user_solver_keeps_canonical_with_nlopt_warning():
    """Canonical names should pass through, with NLopt guidance warning."""
    assert SweepOptimizer._resolve_user_solver("scipy") == "scipy"
    with pytest.warns(UserWarning, match="nlopt"):
        assert SweepOptimizer._resolve_user_solver("nlopt") == "nlopt"


def test_set_state_rebuilds_boundaries_and_normalizes(monkeypatch):
    """set_state should rebuild boundaries and normalize by default."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.bdy = SimpleNamespace(chi=13, mps_b={})
    opt.bdy_overlap = SimpleNamespace(chi=13, mps_b={})
    opt.contraction_opt = "auto-hq"
    opt.fit_mode = "eff"
    opt.state = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=1, dtype="complex128")
    opt.Lx, opt.Ly = 2, 2

    rebuilt = {}

    def _fake_build(state, target, *, chi, single_layer=False):
        rebuilt["state"] = state
        rebuilt["target"] = target
        rebuilt["chi"] = chi
        rebuilt["single_layer"] = single_layer
        return SimpleNamespace(chi=chi, mps_b={"Y0_l": object()}), SimpleNamespace(
            chi=chi,
            mps_b={"Y0_l": object()},
        )

    called_norm = {}

    def _fake_normalize(self, **kwargs):  # pylint: disable=unused-argument
        called_norm.update(kwargs)
        return 7.5

    monkeypatch.setattr(SweepOptimizer, "_build_boundary_pair", staticmethod(_fake_build))
    monkeypatch.setattr(SweepOptimizer, "normalize", _fake_normalize)

    new_state = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=2, dtype="complex128")
    out = SweepOptimizer.set_state(
        opt,
        new_state,
        n_iter=3,
        progress=True,
        track_boundary_fidelity=True,
    )

    assert out == 7.5
    assert opt.state is new_state
    assert rebuilt["state"] is new_state
    assert rebuilt["target"] is opt.state_target
    assert rebuilt["chi"] == 13
    assert called_norm["n_iter"] == 3
    assert called_norm["progress"] is True
    assert called_norm["track_boundary_fidelity"] is True


def test_set_state_can_skip_normalization(monkeypatch):
    """set_state(normalize_state=False) should rebuild without normalize call."""
    opt = object.__new__(SweepOptimizer)
    opt.state_target = object()
    opt.bdy = SimpleNamespace(chi=9, mps_b={})
    opt.bdy_overlap = SimpleNamespace(chi=9, mps_b={})
    opt.contraction_opt = "auto-hq"
    opt.fit_mode = "eff"
    opt.state = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=3, dtype="complex128")
    opt.Lx, opt.Ly = 2, 2

    def _fake_build(state, target, *, chi, single_layer=False):  # pylint: disable=unused-argument
        return SimpleNamespace(chi=chi, mps_b={"Y0_l": object()}), SimpleNamespace(
            chi=chi,
            mps_b={"Y0_l": object()},
        )

    monkeypatch.setattr(SweepOptimizer, "_build_boundary_pair", staticmethod(_fake_build))

    def _boom(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("normalize should not be called")

    monkeypatch.setattr(SweepOptimizer, "normalize", _boom)

    new_state = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=4, dtype="complex128")
    out = SweepOptimizer.set_state(opt, new_state, normalize_state=False)

    assert out is None
    assert opt.state is new_state


def test_set_target_rebuilds_overlap_boundary(monkeypatch):
    """set_target should rebuild bdy_overlap using current state and target."""
    opt = object.__new__(SweepOptimizer)
    opt.state = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=5, dtype="complex128")
    opt.state_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=6, dtype="complex128")
    opt.bdy = SimpleNamespace(chi=11, mps_b={})
    old_overlap = SimpleNamespace(chi=11, mps_b={"old": object()})
    opt.bdy_overlap = old_overlap

    captured = {}

    def _fake_build_bra_ket(*, ket=None, bra=None):
        captured["ket"] = ket
        captured["bra"] = bra
        return object(), object()

    def _fake_bdymps(*, tn_double=None, chi=None, single_layer=False):
        captured["tn_double"] = tn_double
        captured["chi"] = chi
        captured["single_layer"] = single_layer
        return SimpleNamespace(chi=chi, mps_b={"new": object()})

    monkeypatch.setattr("pepsy.optimize_sweep.build_bra_ket", _fake_build_bra_ket)
    monkeypatch.setattr("pepsy.optimize_sweep.BdyMPS", _fake_bdymps)

    target_new = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=7, dtype="complex128")
    SweepOptimizer.set_target(opt, target_new)

    assert opt.state_target is target_new
    assert captured["ket"] is target_new
    assert captured["bra"] is opt.state
    assert captured["chi"] == 11
    assert captured["single_layer"] is False
    assert opt.bdy_overlap is not old_overlap
    assert "new" in opt.bdy_overlap.mps_b


def test_normalize_state_uses_stored_normalize_defaults(monkeypatch):
    """_normalize_state should start from normalize_kwargs then apply sweep-time overrides."""
    opt = object.__new__(SweepOptimizer)
    opt.normalize_kwargs = {
        "contraction_opt": "auto-dummy",
        "n_iter": 99,
        "direction": "x",
        "max_separation": 3,
        "progress": True,
        "track_boundary_fidelity": True,
    }
    captured = {}

    def _fake_normalize(self, **kwargs):  # pylint: disable=unused-argument
        captured.update(kwargs)
        return 1.25

    monkeypatch.setattr(SweepOptimizer, "normalize", _fake_normalize)

    out = SweepOptimizer._normalize_state(opt, env_n_iter=7)

    assert out == 1.25
    assert captured["contraction_opt"] == "auto-dummy"
    assert captured["direction"] == "x"
    assert captured["max_separation"] == 3
    assert captured["n_iter"] == 7
    assert captured["progress"] is False
    assert captured["track_boundary_fidelity"] is False


def test_constructor_accepts_state_target_names():
    """Constructor should accept state/target names."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=11, dtype="complex128")
    peps_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=12, dtype="complex128")
    opt = SweepOptimizer(
        peps,
        peps_target,
        chi=8,
        contraction_opt="auto-hq",
        renormalize_state=False,
    )
    assert opt.state is peps
    assert opt.state_target is peps_target


def test_constructor_rejects_common_internal_indices():
    """Constructor should warn+raise when state/target share inner bond labels."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=19, dtype="complex128")
    peps_target = peps.copy()

    with pytest.warns(UserWarning, match="common internal indices"):
        with pytest.raises(ValueError, match="common internal indices"):
            SweepOptimizer(
                peps,
                peps_target,
                chi=8,
                contraction_opt="auto-hq",
                renormalize_state=False,
            )


def test_metrics_api():
    """metrics() should expose global fidelity and loss."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=13, dtype="complex128")
    peps_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=14, dtype="complex128")
    opt = SweepOptimizer(peps, peps_target, chi=8, contraction_opt="auto-hq", renormalize_state=False)
    fidelity, loss = opt.metrics()
    assert fidelity >= 0.0
    assert loss >= 0.0
    assert (fidelity + loss) == pytest.approx(1.0)


def test_metrics_uses_tn_fidelity_with_optimizer(monkeypatch):
    """metrics() should delegate fidelity computation to core.tn_fidelity."""
    opt = object.__new__(SweepOptimizer)
    opt.state = object()
    opt.state_target = object()
    opt.contraction_opt = "auto-hq"

    captured = {}

    def _fake_tn_fidelity(state, state_target, *, opt=None):
        captured["state"] = state
        captured["state_target"] = state_target
        captured["opt"] = opt
        return 0.75

    monkeypatch.setattr(sweep_mod, "tn_fidelity", _fake_tn_fidelity)

    fidelity, loss = SweepOptimizer.metrics(opt)

    assert fidelity == pytest.approx(0.75)
    assert loss == pytest.approx(0.25)
    assert captured["state"] is opt.state
    assert captured["state_target"] is opt.state_target
    assert captured["opt"] == "auto-hq"


def test_constructor_renormalize_uses_requested_options(monkeypatch):
    """Constructor should renormalize with user-provided keyword options."""
    called = {}

    def _fake_normalize(self, **kwargs):  # pylint: disable=unused-argument
        called.update(kwargs)
        return 1.0

    monkeypatch.setattr(SweepOptimizer, "normalize", _fake_normalize)

    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=15, dtype="complex128")
    peps_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=16, dtype="complex128")
    _ = SweepOptimizer(
        peps,
        peps_target,
        chi=8,
        n_iter=7,
        direction="x",
        max_separation=2,
        progress=True,
        track_boundary_fidelity=False,
        renormalize_state=True,
    )
    assert called["n_iter"] == 7
    assert called["direction"] == "x"
    assert called["max_separation"] == 2
    assert called["progress"] is True
    assert called["track_boundary_fidelity"] is False


def test_constructor_renormalize_kwargs_mapping_style(monkeypatch):
    """Constructor should accept init normalize options via renormalize_kwargs."""
    called = {}

    def _fake_normalize(self, **kwargs):  # pylint: disable=unused-argument
        called.update(kwargs)
        return 1.0

    monkeypatch.setattr(SweepOptimizer, "normalize", _fake_normalize)

    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=17, dtype="complex128")
    peps_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=18, dtype="complex128")
    _ = SweepOptimizer(
        peps,
        peps_target,
        chi=8,
        renormalize_state=True,
        renormalize_kwargs={
            "n_iter": 9,
            "direction": "x",
            "max_separation": 1,
            "progress": True,
            "track_boundary_fidelity": False,
        },
    )

    assert called["n_iter"] == 9
    assert called["direction"] == "x"
    assert called["max_separation"] == 1
    assert called["progress"] is True
    assert called["track_boundary_fidelity"] is False
