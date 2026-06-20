"""Tests for :mod:`pepsy.optimizers.peps`."""

import pytest

import pepsy.optimizers.peps as peps_mod
from pepsy.optimizers.peps import PepsOptimizer


class DummyState:
    """Tiny PEPS-like object for orchestration tests."""

    Lx = 2
    Ly = 2

    def __init__(self, bond=1, name="state"):
        self.bond = int(bond)
        self.name = name
        self.applied = []
        self.normalized = 0
        self.normalize_kwargs = []
        self.mangled = 0

    def copy(self):
        other = type(self)(self.bond, f"{self.name}.copy")
        other.applied = list(self.applied)
        other.normalized = self.normalized
        other.normalize_kwargs = [dict(opts) for opts in self.normalize_kwargs]
        other.mangled = self.mangled
        return other

    def max_bond(self):
        return self.bond

    def compress_all(self, max_bond=None, cutoff=1e-12, inplace=False, **kwargs):
        _ = cutoff, kwargs
        out = self if inplace else self.copy()
        if max_bond is not None:
            out.bond = min(out.bond, int(max_bond))
        return out

    def compress_all_(self, max_bond=None, cutoff=1e-12, **kwargs):
        _ = cutoff, kwargs
        if max_bond is not None:
            self.bond = min(self.bond, int(max_bond))
        return self

    def mangle_inner_(self, append=None, which=None):
        _ = append, which
        self.mangled += 1
        return self


class _FakeSymmrayArray:
    shape = (2, 2)
    dtype = "complex128"


_FakeSymmrayArray.__module__ = "symmray.fake"


class _FakeTensor:
    def __init__(self):
        self.data = _FakeSymmrayArray()


class SymmrayDummyState(DummyState):
    """Dummy state carrying Symmray-looking tensor data."""

    def __init__(self, bond=1, name="state"):
        super().__init__(bond=bond, name=name)
        self.tensor_map = {"site": _FakeTensor()}


def _install_fake_gate(monkeypatch):
    calls = []

    def _fake_gate(state, gate_payload, where=None, which=None, inplace=True, **opts):
        calls.append(
            {
                "state": state,
                "gate": gate_payload,
                "where": where,
                "which": which,
                "inplace": inplace,
                "opts": dict(opts),
            }
        )
        out = state if inplace else state.copy()
        bond = int(gate_payload.get("bond", out.bond))
        if opts.get("max_bond") is not None:
            bond = min(bond, int(opts["max_bond"]))
        out.bond = bond
        out.applied.append((gate_payload, where, which, dict(opts)))
        return out

    monkeypatch.setattr(peps_mod, "apply_gate", _fake_gate)
    return calls


def _install_fake_normalize(monkeypatch):
    calls = []

    def _fake_normalize(state, **kwargs):
        state.normalized += 1
        state.normalize_kwargs.append(dict(kwargs))
        calls.append((state, dict(kwargs)))
        return 1.0

    monkeypatch.setattr(peps_mod, "boundary_normalize", _fake_normalize)
    return calls


def test_peps_optimizer_within_chi_skips_infidelity_and_optimizer(monkeypatch):
    """Two-site gates whose target fits in chi should advance exactly."""
    gate_calls = _install_fake_gate(monkeypatch)

    def _boom_infidelity(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("infidelity should not be estimated within chi")

    class _BoomSweep:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            raise AssertionError("optimizer should not run within chi")

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _boom_infidelity)
    monkeypatch.setattr(peps_mod, "SweepOptimizer", _BoomSweep)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 2}, ((0, 0), (0, 1)))],
        chi=4,
        normalize_initial=False,
    )

    out = opt.run(progress=False)

    assert out.bond == 2
    assert len(gate_calls) == 1
    assert gate_calls[0]["opts"].get("max_bond") is None
    assert opt.step_records[0]["reason"] == "within_chi"
    assert opt.local_infidelities == [0.0]
    assert opt.get_fidelities()[-1] == pytest.approx(1.0)


def test_peps_optimizer_truncated_warmstart_below_tol_skips_sweep(monkeypatch):
    """A good chi-truncated warm start should be accepted without sweeps."""
    gate_calls = _install_fake_gate(monkeypatch)
    norm_calls = _install_fake_normalize(monkeypatch)

    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": 1.0e-12},
    )

    class _BoomSweep:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            raise AssertionError("optimizer should not run below tolerance")

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _BoomSweep)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    out = opt.run(progress=False, infidelity_tol=1.0e-9)

    assert out.bond == 3
    assert out.normalized == 1
    assert len(norm_calls) == 1
    assert norm_calls[0][1]["chi"] == 6
    assert norm_calls[0][1]["n_iter"] == 10
    assert norm_calls[0][1]["direction"] == "y"
    assert norm_calls[0][1]["max_separation"] == 1
    assert norm_calls[0][1]["track_boundary_fidelity"] is False
    assert norm_calls[0][1]["strip_exponent"] is True
    assert [call["opts"].get("max_bond") for call in gate_calls] == [None]
    assert gate_calls[0]["opts"]["path_canonize"] is True
    assert opt.step_records[0]["reason"] == "below_tol"
    assert opt.step_records[0]["pre_infidelity"] == pytest.approx(1.0e-12)
    normalization_events = opt.get_normalizations()
    assert len(normalization_events) == 1
    assert "state" not in normalization_events[0]
    assert normalization_events[0]["old_norm"] == pytest.approx(1.0)
    assert normalization_events[0]["state_max_bond"] == 3


def test_peps_optimizer_symmray_defaults_to_quimb_mps_boundaries(monkeypatch):
    """Symmray PEPS runs should avoid the PEPSY DMRG/FIT boundary path."""
    _install_fake_gate(monkeypatch)
    norm_calls = _install_fake_normalize(monkeypatch)
    infidelity_calls = []

    def _fake_infidelity(*args, **kwargs):
        infidelity_calls.append(dict(kwargs))
        return {"infidelity": 1.0e-12}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)

    opt = PepsOptimizer(
        SymmrayDummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    out = opt.run(progress=False, infidelity_tol=1.0e-9)

    assert out.bond == 3
    assert norm_calls[0][1]["method"] == "mps"
    assert norm_calls[0][1]["mode_"] == "mps"
    assert norm_calls[0][1]["balance_bonds"] is False
    assert infidelity_calls[0]["method"] == "mps"
    assert infidelity_calls[0]["mode_"] == "mps"


def test_peps_optimizer_runs_sweep_and_records_geometric_fidelity(monkeypatch):
    """Large warm-start infidelity should hand off to SweepOptimizer."""
    gate_calls = _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([0.2, 0.04])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )
    captured = {}

    class _FakeSweep:
        def __init__(self, *, state, state_target, **kwargs):
            captured["init"] = {
                "state": state,
                "state_target": state_target,
                "kwargs": dict(kwargs),
            }
            self.state = DummyState(bond=state.bond, name="swept")

        def set_optimize_kwargs(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return self

        def run(self):
            return {
                "best_loss": 0.05,
                "best_state": DummyState(bond=3, name="best"),
                "loss_after": 0.08,
                "runs": [{"state": DummyState(bond=99, name="heavy-run")}],
                "loss": [0.2, 0.05],
                "step_trace": [{"state": DummyState(bond=99, name="heavy-step")}],
                "inner_loss_traces": [[0.2, 0.1]],
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        boundary_chi=(5, 7),
        normalize_initial=False,
        sweep_optimize_kwargs={"n_cycles": 2, "optimizer": "scipy"},
    )

    out = opt.run(progress=False)

    assert out.name == "best"
    assert out.normalized == 1
    assert len(gate_calls) == 1
    assert gate_calls[0]["opts"].get("max_bond") is None
    assert captured["init"]["state"].bond == 3
    assert captured["init"]["state_target"].mangled == 1
    assert captured["init"]["kwargs"]["chi"] == (5, 7)
    assert captured["init"]["kwargs"]["target_norm"] == 1.0
    assert captured["init"]["kwargs"]["n_iter"] == 10
    assert captured["init"]["kwargs"]["direction"] == "y"
    assert captured["init"]["kwargs"]["max_separation"] == 1
    assert captured["init"]["kwargs"]["track_boundary_fidelity"] is False
    assert captured["init"]["kwargs"]["normalize_kwargs"]["strip_exponent"] is True
    assert captured["optimize_kwargs"]["n_cycles"] == 2
    assert captured["optimize_kwargs"]["n_round_trips"] == 4
    assert captured["optimize_kwargs"]["renormalize"] is False
    assert captured["optimize_kwargs"]["env_n_iter"] == 10
    assert captured["optimize_kwargs"]["progress"] is False
    assert captured["optimize_kwargs"]["progress_position"] == 0
    assert opt.local_infidelities[0] == pytest.approx(0.04)
    assert opt.get_fidelities()[-1] == pytest.approx(0.96)
    assert opt.get_infidelities()[-1] == pytest.approx(0.04)
    assert opt.step_records[0]["optimized"] is True
    assert opt.step_records[0]["optimizer_attempted"] is True
    assert opt.step_records[0]["optimizer_infidelity"] == pytest.approx(0.05)
    assert opt.step_records[0]["post_infidelity"] == pytest.approx(0.04)
    assert opt.step_records[0]["reason"] == "optimized"
    result_summary = opt.step_records[0]["optimizer_result"]
    assert result_summary["backend"] == "sweep"
    assert result_summary["best_loss"] == pytest.approx(0.05)
    assert result_summary["loss_after"] == pytest.approx(0.08)
    assert result_summary["n_runs"] == 1
    assert result_summary["loss_count"] == 2
    assert result_summary["step_count"] == 1
    assert result_summary["inner_loss_trace_count"] == 1
    assert "best_state" not in result_summary
    assert "runs" not in result_summary
    assert "step_trace" not in result_summary
    assert "inner_loss_traces" not in result_summary


def test_peps_optimizer_separates_optimization_normalize_and_evaluation_chi(monkeypatch):
    """Normalization and acceptance diagnostics can use stricter boundary chis."""
    _install_fake_gate(monkeypatch)
    norm_calls = _install_fake_normalize(monkeypatch)
    infidelity_calls = []
    infidelities = iter([0.2, 0.03])

    def _fake_infidelity(state, target, **kwargs):
        _ = state, target
        infidelity_calls.append(dict(kwargs))
        return {"infidelity": next(infidelities)}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)
    captured = {}

    class _FakeSweep:
        def __init__(self, *, state, state_target, **kwargs):
            captured["init"] = {
                "state": state,
                "state_target": state_target,
                "kwargs": dict(kwargs),
            }
            self.state = DummyState(bond=state.bond, name="swept")

        def set_optimize_kwargs(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return self

        def run(self):
            return {
                "best_loss": 0.05,
                "best_state": DummyState(bond=3, name="best"),
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        boundary_chi=4,
        normalize_chi=6,
        evaluation_chi=9,
        normalize_initial=True,
    )

    out = opt.run(progress=False, infidelity_tol=1.0e-9)

    assert out.name == "best"
    assert [call[1]["chi"] for call in norm_calls] == [6, 6, 6]
    assert [call["chi"] for call in infidelity_calls] == [9, 9]
    assert captured["init"]["kwargs"]["chi"] == 4
    assert captured["init"]["kwargs"]["normalize_kwargs"]["chi"] == 6
    assert captured["init"]["kwargs"]["normalize_kwargs"]["strip_exponent"] is True
    assert captured["optimize_kwargs"]["env_n_iter"] == 10
    record = opt.step_records[0]
    assert record["normalize_chi"] == 6
    assert record["evaluation_chi"] == 9
    assert record["pre_infidelity"] == pytest.approx(0.2)
    assert record["post_infidelity"] == pytest.approx(0.03)
    assert record["reason"] == "optimized"


def test_peps_optimizer_run_overrides_normalize_and_evaluation_chi(monkeypatch):
    """Per-run chi overrides should win over constructor diagnostic chis."""
    _install_fake_gate(monkeypatch)
    norm_calls = _install_fake_normalize(monkeypatch)
    infidelity_calls = []

    def _fake_infidelity(state, target, **kwargs):
        _ = state, target
        infidelity_calls.append(dict(kwargs))
        return {"infidelity": 0.2}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        boundary_chi=4,
        normalize_chi=6,
        evaluation_chi=9,
        normalize_initial=False,
    )

    out = opt.run(
        progress=False,
        optimize=False,
        normalize_chi=10,
        evaluation_chi=11,
    )

    assert out.bond == 3
    assert [call[1]["chi"] for call in norm_calls] == [10]
    assert [call["chi"] for call in infidelity_calls] == [11]
    record = opt.step_records[0]
    assert record["normalize_chi"] == 10
    assert record["evaluation_chi"] == 11
    assert record["reason"] == "warmstart"


def test_peps_optimizer_rejects_optimizer_when_final_infidelity_is_worse(monkeypatch):
    """Keep the normalized warm start if variational cleanup does not help."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([0.1, 0.2])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )

    class _FakeSweep:
        def __init__(self, *, state, state_target, **kwargs):
            _ = state_target, kwargs
            state.bond = 99
            state.name = "mutated-warmstart"
            self.state = state

        def set_optimize_kwargs(self, **kwargs):
            _ = kwargs
            return self

        def run(self):
            return {
                "best_loss": 0.05,
                "best_state": DummyState(bond=9, name="bad-candidate"),
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    out = opt.run(progress=False)

    record = opt.step_records[0]
    assert out.bond == 3
    assert out.name != "mutated-warmstart"
    assert out.name != "bad-candidate"
    assert record["reason"] == "optimizer_rejected"
    assert record["optimized"] is False
    assert record["optimizer_attempted"] is True
    assert record["pre_infidelity"] == pytest.approx(0.1)
    assert record["optimizer_infidelity"] == pytest.approx(0.05)
    assert record["post_infidelity"] == pytest.approx(0.2)
    assert record["final_infidelity"] == pytest.approx(0.1)
    assert opt.local_infidelities == [pytest.approx(0.1)]


def test_peps_optimizer_global_defaults_options_and_torch_svd(monkeypatch):
    """Global mode should build compact MPS loss options and route NLopt controls."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([0.3, 0.05])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )
    registered = []
    monkeypatch.setattr(peps_mod, "_reg_complex_svd_torch", lambda: registered.append(True))
    captured = {}

    class _FakeGlobal:
        def __init__(self, *, state, state_target, **kwargs):
            captured["init"] = {
                "state": state,
                "state_target": state_target,
                "kwargs": dict(kwargs),
            }
            self.losses = []

        def optimize_nlopt(self, **kwargs):
            captured["nlopt_kwargs"] = dict(kwargs)
            self.losses = [0.2, 0.05]
            return DummyState(bond=3, name="global-best")

    monkeypatch.setattr(peps_mod, "GlobalOptimizer", _FakeGlobal)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        boundary_chi=(16, 16),
        mode="global",
        contraction_opt="auto-hq",
        optimizer="nlopt",
        optimizer_options={
            "algorithm": "LD_VAR2",
            "n_steps": 25,
            "maxeval": 120,
            "ftol_rel": 1.0e-9,
            "xtol_rel": 1.0e-9,
            "progress": False,
            "device": "cpu",
        },
        normalize_initial=False,
    )

    out = opt.run(progress=False)

    assert out.name == "global-best"
    assert registered == [True]
    init_kwargs = captured["init"]["kwargs"]
    assert init_kwargs["chi"] == (16, 16)
    assert init_kwargs["norm_kwargs"]["chi"] == (16, 16)
    assert init_kwargs["norm_kwargs"]["mode"] == "mps"
    assert init_kwargs["norm_kwargs"]["mode_"] == "mps"
    assert init_kwargs["norm_kwargs"]["max_separation"] == 1
    assert init_kwargs["norm_kwargs"]["cutoff"] == pytest.approx(1.0e-12)
    assert init_kwargs["norm_kwargs"]["strip_exponent"] is True
    assert init_kwargs["loss_kwargs"]["sequence"] == ["xmax", "xmin", "ymin", "ymax"]
    assert init_kwargs["loss_kwargs"]["target_norm"] == 1.0
    assert init_kwargs["loss_kwargs"]["strip_exponent"] is True
    assert captured["nlopt_kwargs"]["optimizer"] == "LD_VAR2"
    assert captured["nlopt_kwargs"]["n"] == 120
    assert captured["nlopt_kwargs"]["ftol_rel"] == pytest.approx(1.0e-9)
    assert captured["nlopt_kwargs"]["xtol_rel"] == pytest.approx(1.0e-9)
    assert captured["nlopt_kwargs"]["progbar"] is False
    assert captured["nlopt_kwargs"]["device"] == "cpu"
    assert opt.step_records[0]["post_infidelity"] == pytest.approx(0.05)
    assert opt.step_records[0]["reason"] == "optimized"
    result_summary = opt.step_records[0]["optimizer_result"]
    assert result_summary["backend"] == "global"
    assert result_summary["optimizer"] == "LD_VAR2"
    assert result_summary["n"] == 120
    assert result_summary["loss_count"] == 2
    assert result_summary["loss_final"] == pytest.approx(0.05)
    assert result_summary["fallback_used"] is False
    assert isinstance(result_summary["optimizer"], str)
    assert "losses" not in result_summary


def test_peps_optimizer_global_nlopt_runtime_error_falls_back(monkeypatch):
    """NLopt runtime errors should fall back to a short LBFGS cleanup."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([0.2, 0.09])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )
    monkeypatch.setattr(peps_mod, "_reg_complex_svd_torch", lambda: None)
    captured = {}

    class _FakeNloptRuntimeError(Exception):
        pass

    _FakeNloptRuntimeError.__module__ = "nlopt.nlopt"

    class _FakeGlobal:
        def __init__(self, *, state, state_target, **kwargs):
            _ = state, state_target, kwargs
            self.losses = []

        def optimize_nlopt(self, **kwargs):
            captured["nlopt_kwargs"] = dict(kwargs)
            raise _FakeNloptRuntimeError("roundoff limited")

        def optimize(self, **kwargs):
            captured["fallback_kwargs"] = dict(kwargs)
            self.losses = [0.09]
            return DummyState(bond=3, name="fallback-best")

    monkeypatch.setattr(peps_mod, "GlobalOptimizer", _FakeGlobal)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        mode="global",
        optimizer="nlopt",
        normalize_initial=False,
    )

    with pytest.warns(RuntimeWarning, match="falling back"):
        out = opt.run(progress=False)

    record = opt.step_records[0]
    assert out.name == "fallback-best"
    assert captured["nlopt_kwargs"]["optimizer"] == "nlopt"
    assert captured["nlopt_kwargs"]["n"] == 1200
    assert captured["fallback_kwargs"]["optimizer"] == "lbfgs"
    assert captured["fallback_kwargs"]["n"] == 1
    assert captured["fallback_kwargs"]["progbar"] is False
    assert record["optimizer_result"]["fallback_used"] is True
    assert record["optimizer_result"]["fallback_error"] == "roundoff limited"
    assert record["optimizer_result"]["fallback_error_type"] == "_FakeNloptRuntimeError"
    assert isinstance(record["optimizer_result"]["optimizer"], str)
    assert "losses" not in record["optimizer_result"]
    assert record["post_infidelity"] == pytest.approx(0.09)
    assert record["reason"] == "optimized"


def test_peps_optimizer_global_default_budget_is_user_overridable(monkeypatch):
    """Global mode should default to n=1200 but let explicit n win."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": 0.05},
    )
    monkeypatch.setattr(peps_mod, "_reg_complex_svd_torch", lambda: None)
    captured = {}

    class _FakeGlobal:
        def __init__(self, *, state, state_target, **kwargs):
            _ = state, state_target, kwargs
            self.losses = [0.04]

        def optimize(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return DummyState(bond=3, name="global-default")

    monkeypatch.setattr(peps_mod, "GlobalOptimizer", _FakeGlobal)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        mode="global",
        normalize_initial=False,
        global_optimize_kwargs={"n": 7, "optimizer": "lbfgs"},
    )

    _ = opt.run(progress=False, measure_final_infidelity=False)

    assert captured["optimize_kwargs"]["n"] == 7
    assert captured["optimize_kwargs"]["optimizer"] == "lbfgs"


def test_peps_optimizer_progress_reports_geometric_infidelity(monkeypatch):
    """Progress bar should expose a compact MpsOptimizer-style postfix."""
    import tqdm as tqdm_pkg

    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": 1.0e-6},
    )
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, *args, **kwargs):
            _ = args
            self.kwargs = dict(kwargs)
            self.postfix_calls = []
            self.n = 0
            self.closed = False
            progress_instances.append(self)

        def set_postfix(self, postfix):
            self.postfix_calls.append(dict(postfix))

        def update(self, value):
            self.n += int(value)

        def close(self):
            self.closed = True

    monkeypatch.setattr(tqdm_pkg, "tqdm", _FakeTqdm)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    _ = opt.run(progress=True, infidelity_tol=1.0e-3)

    progress = progress_instances[-1]
    last = progress.postfix_calls[-1]
    assert progress.n == 1
    assert progress.closed is True
    assert progress.kwargs["position"] == 0
    assert progress.kwargs["ascii"] is True
    # No widget-only knobs that diverge from MpsOptimizer's bar.
    assert "unit" not in progress.kwargs
    assert "dynamic_ncols" not in progress.kwargs
    # Compact postfix mirrors MpsOptimizer: 2q, bnd, Icum (no why/I/Igeo/tgt).
    assert set(last.keys()) == {"2q", "bnd", "Icum"}
    assert last["2q"] == 1
    assert last["bnd"] == 3
    assert last["Icum"] == PepsOptimizer._format_progress_infidelity(
        opt.get_infidelities()[-1]
    )
    # Only one postfix update per gate step, matching MpsOptimizer.
    assert len(progress.postfix_calls) == 1


def test_peps_optimizer_inner_sweep_progress_is_silenced():
    """Inner sweep bar should be off so only the outer PEPS bar shows."""
    opt = PepsOptimizer(DummyState(bond=1), [], chi=2, normalize_initial=False)

    _init_kwargs, opt_kwargs, strip_exponent = opt._sweep_boundary_kwargs(progress=True)

    assert opt_kwargs["progress"] is False
    assert opt_kwargs["progress_position"] == 0
    assert strip_exponent is True


def test_peps_optimizer_nonunitary_normalizes_target_before_infidelity(monkeypatch):
    """Non-unitary runs should normalize both target and warm start."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)

    def _fake_infidelity(state, target, **kwargs):
        assert state.normalized >= 1
        assert target.normalized == 1
        assert kwargs["norm"] == 1.0
        assert kwargs["norm_target"] == 1.0
        assert kwargs["strip_exponent"] is True
        return {"infidelity": 1.0e-12}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    out = opt.run(non_unitary=True, infidelity_tol=1.0e-9, progress=False)

    assert out.bond == 3
    assert opt.step_records[0]["reason"] == "below_tol"


def test_peps_optimizer_batches_two_site_targets_before_truncation(monkeypatch):
    """Batched PEPS updates should absorb gates before the chi warm start."""
    gate_calls = _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelity_calls = []

    def _fake_infidelity(state, target, **kwargs):
        infidelity_calls.append(
            {
                "state_applied": list(state.applied),
                "target_applied": list(target.applied),
                "kwargs": dict(kwargs),
            }
        )
        return {"infidelity": 1.0e-12}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)

    gates = [
        ({"bond": 8, "label": "a"}, ((0, 0), (0, 1))),
        ({"bond": 9, "label": "b"}, ((1, 1),)),
        ({"bond": 10, "label": "c"}, ((1, 0), (1, 1))),
    ]
    opt = PepsOptimizer(DummyState(bond=1), gates, chi=3, normalize_initial=False)

    out = opt.run(progress=False, k_2q_batch=2, infidelity_tol=1.0e-9)

    assert out.bond == 3
    assert [call["gate"]["label"] for call in gate_calls] == ["a", "b", "c"]
    assert len(infidelity_calls) == 1
    assert [
        gate_payload["label"]
        for gate_payload, _where, _which, _opts in infidelity_calls[0]["target_applied"]
    ] == ["a", "b", "c"]
    assert [
        gate_payload["label"]
        for gate_payload, _where, _which, _opts in infidelity_calls[0]["state_applied"]
    ] == ["a", "b", "c"]
    record = opt.step_records[0]
    assert record["step"] == 3
    assert record["start_step"] == 1
    assert record["where"] == tuple(where for _gate, where in gates)
    assert record["batch_size"] == 3
    assert record["two_site_batch"] == 2
    assert record["reason"] == "below_tol"
    assert opt.last_result["step"] == 3
    assert opt.get_local_infidelities() == [pytest.approx(1.0e-12)]


def test_peps_optimizer_batches_before_sweep_optimizer(monkeypatch):
    """A batch that needs cleanup should run one sweep against the batch target."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([0.2, 0.03])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )
    captured = {}

    class _FakeSweep:
        def __init__(self, *, state, state_target, **kwargs):
            captured["state"] = state
            captured["state_target"] = state_target
            captured["kwargs"] = dict(kwargs)
            self.state = DummyState(bond=state.bond, name="swept")

        def set_optimize_kwargs(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return self

        def run(self):
            return {
                "best_loss": 0.04,
                "best_state": DummyState(bond=3, name="best"),
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    gates = [
        ({"bond": 8, "label": "a"}, ((0, 0), (0, 1))),
        ({"bond": 10, "label": "b"}, ((1, 0), (1, 1))),
    ]
    opt = PepsOptimizer(DummyState(bond=1), gates, chi=3, normalize_initial=False)

    out = opt.run(progress=False, k_2q_batch=2, infidelity_tol=1.0e-9)

    assert out.name == "best"
    assert captured["state"].bond == 3
    assert [
        gate_payload["label"]
        for gate_payload, _where, _which, _opts in captured["state_target"].applied
    ] == ["a", "b"]
    record = opt.step_records[0]
    assert record["step"] == 2
    assert record["batch_size"] == 2
    assert record["two_site_batch"] == 2
    assert record["optimizer_attempted"] is True
    assert record["reason"] == "optimized"


def test_peps_optimizer_rejects_invalid_two_site_batch_size():
    """The PEPS batch size should be a positive integer."""
    opt = PepsOptimizer(DummyState(bond=1), [], chi=2, normalize_initial=False)

    with pytest.raises(ValueError, match="k_2q_batch"):
        opt.run(k_2q_batch=0)


def test_peps_optimizer_forwards_entry_which_over_default(monkeypatch):
    """Bundled entry ``which`` should target lower/upper k-b families."""
    gate_calls = _install_fake_gate(monkeypatch)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 1}, ((0, 0),), "lower")],
        chi=2,
        which="upper",
        normalize_initial=False,
    )

    _ = opt.run(progress=False)

    assert gate_calls[0]["which"] == "lower"
