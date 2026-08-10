"""Tests for :mod:`pepsy.optimizers.peps.optimizer`."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.backends import TorchLinalgConfig
import pepsy.optimizers.peps.optimizer as peps_mod
import pepsy.optimizers.sweep.optimizer as sweep_mod
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

    def __init__(self, backend="torch"):
        self.backend = backend


_FakeSymmrayArray.__module__ = "symmray.fake"


class _FakeTensor:
    def __init__(self, backend="torch"):
        self.data = _FakeSymmrayArray(backend=backend)


class SymmrayDummyState(DummyState):
    """Dummy state carrying Symmray-looking tensor data."""

    def __init__(self, bond=1, name="state", backend="torch"):
        super().__init__(bond=bond, name=name)
        self.backend = backend
        self.tensor_map = {"site": _FakeTensor(backend=backend)}

    def copy(self):
        other = type(self)(self.bond, f"{self.name}.copy", backend=self.backend)
        other.applied = list(self.applied)
        other.normalized = self.normalized
        other.normalize_kwargs = [dict(opts) for opts in self.normalize_kwargs]
        other.mangled = self.mangled
        return other


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


def test_peps_optimizer_routes_two_site_boundary_policy(monkeypatch):
    """Boundary kwargs should configure normalization and inner PEPS sweeps."""
    calls = _install_fake_normalize(monkeypatch)
    infidelity_calls = []

    def _fake_infidelity(_state, _target, **kwargs):
        infidelity_calls.append(dict(kwargs))
        return {"infidelity": 0.0}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)
    policy = {
        "fit_mode": "two-site",
        "fit_max_bond": 7,
        "fit_sweep_sequence": "RL",
        "fit_cutoff_mode": "rsum2",
        "fit_min_iter": 2,
        "fit_rtol": 1.0e-8,
        "fit_patience": 3,
        "cutoff": 2.0e-10,
    }
    opt = PepsOptimizer(
        DummyState(bond=1),
        [],
        chi=3,
        boundary_kwargs=policy,
        normalize_initial=False,
    )

    opt.normalize()
    opt.estimate_infidelity(DummyState(), DummyState())
    init_kwargs, _, _ = opt._sweep_boundary_kwargs(progress=False)

    for key, value in policy.items():
        assert calls[-1][1][key] == value
        assert infidelity_calls[-1][key] == value
        assert init_kwargs[key] == value


def test_sweep_optimizer_builds_two_site_cached_boundary_pair():
    """PEPS sweep cleanup should construct both CompBdy objects consistently."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=5)
    target = state.copy()
    target.mangle_inner_("_target")
    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        chi=(3, 4),
        fit_mode="two-site",
        fit_sweep_sequence="LR",
        fit_cutoff_mode="rel",
        cutoff=2.0e-9,
        fit_min_iter=2,
        fit_rtol=3.0e-8,
        fit_patience=4,
        simplify=False,
    )
    assert max(mps.max_bond() for mps in sweep.bdy.mps_b.values()) == 1
    assert max(mps.max_bond() for mps in sweep.bdy_overlap.mps_b.values()) == 1
    norm_tn, overlap_tn = sweep._prepare_current_double_layers()

    comp_norm, comp_overlap = sweep._make_comp_pair(norm_tn, overlap_tn)

    assert comp_norm.fit_mode == comp_overlap.fit_mode == "two-site"
    assert comp_norm.fit_max_bond == 3
    assert comp_overlap.fit_max_bond == 4
    for comp in (comp_norm, comp_overlap):
        assert comp.fit_sweep_sequence == "LR"
        assert comp.fit_cutoff == 2.0e-9
        assert comp.fit_cutoff_mode == "rel"
        assert comp.fit_min_iter == 2
        assert comp.fit_rtol == 3.0e-8
        assert comp.fit_patience == 4

    sweep.set_chi((5, 6))
    assert max(mps.max_bond() for mps in sweep.bdy.mps_b.values()) == 1
    assert max(mps.max_bond() for mps in sweep.bdy_overlap.mps_b.values()) == 1
    comp_norm, comp_overlap = sweep._make_comp_pair(norm_tn, overlap_tn)
    assert comp_norm.fit_max_bond == 5
    assert comp_overlap.fit_max_bond == 6


def test_sweep_optimizer_two_site_boundary_move_grows_rank_locally():
    """A real PEPS boundary move should grow rank from the product warm start."""
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=71)
    target = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=73)
    target.mangle_inner_("_target")
    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        chi=(4, 5),
        fit_mode="two-site",
        fit_sweep_sequence="RL",
        simplify=False,
    )

    sweep._refresh_right_boundaries_once("y", env_n_iter=2)

    assert 1 < sweep.bdy.chi <= 4
    assert 1 < sweep.bdy_overlap.chi <= 5


def test_sweep_optimizer_one_site_set_chi_can_lower_boundary_caps():
    """One-site set_chi should treat a lower value as a target, not a no-op."""
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=75)
    target = state.copy()
    target.mangle_inner_("_target")
    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        chi=5,
        fit_mode="eff",
        simplify=False,
    )

    sweep.set_chi(2)

    assert sweep.chi == 2
    assert sweep.bdy.chi <= 2
    assert sweep.bdy_overlap.chi <= 2


def test_sweep_optimizer_two_site_policy_survives_target_replacement():
    """A new target chi should update only the stored overlap FIT cap."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=79)
    target = state.copy()
    target.mangle_inner_("_target")
    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        chi=(3, 4),
        fit_mode="two_site",
        simplify=False,
    )
    replacement = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        dtype="complex128",
        seed=83,
    )
    replacement.mangle_inner_("_replacement")

    returned = sweep.set_target(replacement, chi=7)
    norm_tn, overlap_tn = sweep._prepare_current_double_layers()
    comp_norm, comp_overlap = sweep._make_comp_pair(norm_tn, overlap_tn)

    assert returned is sweep
    assert sweep.fit_mode == "two-site"
    assert sweep.chi == (3, 7)
    assert sweep.bdy_overlap.chi == 1
    assert comp_norm.fit_max_bond == 3
    assert comp_overlap.fit_max_bond == 7


def test_sweep_optimizer_infidelity_uses_requested_chi_and_none_override(
    monkeypatch,
):
    """Warm rank must not become the cap, and None must disable inherited rtol."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=89)
    target = state.copy()
    target.mangle_inner_("_target")
    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        chi=(4, 6),
        fit_mode="two-site",
        cutoff=2.0e-10,
        fit_min_iter=2,
        fit_rtol=3.0e-8,
        fit_patience=4,
        simplify=False,
    )
    calls = []

    def fake_infidelity(_state, _target, **kwargs):
        calls.append(dict(kwargs))
        return {"infidelity": 0.0}

    monkeypatch.setattr(sweep_mod, "boundary_infidelity", fake_infidelity)

    sweep.infidelity()
    sweep.infidelity(fit_min_iter=None, fit_rtol=None)

    inherited, fixed_sweeps = calls
    assert inherited["chi"] == fixed_sweeps["chi"] == 6
    assert inherited["cutoff"] == fixed_sweeps["cutoff"] == 2.0e-10
    assert inherited["fit_min_iter"] == 2
    assert inherited["fit_rtol"] == 3.0e-8
    assert inherited["fit_patience"] == 4
    assert fixed_sweeps["fit_min_iter"] is None
    assert fixed_sweeps["fit_rtol"] is None


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

    out = opt.run(progress=False, normalize_target=False)

    assert out.bond == 2
    assert len(gate_calls) == 1
    assert gate_calls[0]["opts"].get("max_bond") is None
    assert opt.step_records[0]["reason"] == "within_chi"
    assert opt.local_infidelities == [0.0]
    assert opt.get_fidelities()[-1] == pytest.approx(1.0)


def test_peps_optimizer_truncated_warmstart_below_tol_skips_sweep(monkeypatch):
    """Default target normalization precedes warm-start acceptance."""
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
    assert out.normalized == 2
    assert len(norm_calls) == 2
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
    assert len(normalization_events) == 2
    assert "state" not in normalization_events[0]
    assert normalization_events[0]["old_norm"] == pytest.approx(1.0)
    assert normalization_events[0]["state_max_bond"] == 8
    assert normalization_events[1]["state_max_bond"] == 3


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


def test_peps_optimizer_rejects_non_torch_symmray_input():
    """Symmray PEPS cleanup is deliberately Torch/autograd-only."""
    with pytest.raises(TypeError, match="Torch-backed Symmray"):
        PepsOptimizer(
            SymmrayDummyState(bond=1, backend="numpy"),
            [],
            chi=3,
            normalize_initial=False,
        )


def test_peps_optimizer_rejects_mixed_symmray_and_dense_inputs():
    """A malformed gate target must not silently drop Symmray structure."""
    with pytest.raises(TypeError, match="do not mix Symmray and dense"):
        PepsOptimizer._require_torch_symmray_backend(
            SymmrayDummyState(bond=1),
            DummyState(bond=1),
            role="state and target",
        )


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
                "inner_best_loss_traces": [[0.2, 0.1]],
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

    out = opt.run(progress=False, sweep_progress=True)

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
    assert captured["optimize_kwargs"]["progress"] is True
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
    assert result_summary["inner_best_loss_trace_count"] == 1
    assert result_summary["min_inner_infidelity"] == pytest.approx(0.1)
    assert result_summary["inner_best_monotonic"] is True
    assert result_summary["best_after_abs_error"] == pytest.approx(0.03)
    assert result_summary["invalid_inner_loss_count"] == 0
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
    assert [call[1]["chi"] for call in norm_calls] == [6, 6, 6, 6]
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
    assert [call[1]["chi"] for call in norm_calls] == [10, 10]
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
    class _RecordingConfig:
        def __init__(self, **kwargs):
            registered.append(dict(kwargs))

        def register(self):
            return self

    monkeypatch.setattr(peps_mod, "TorchLinalgConfig", _RecordingConfig)
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
    assert registered == [{
        "mode": "complex",
        "stabilized": True,
        "quimb_split_drivers": False,
    }]
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


def test_peps_optimizer_accepts_one_torch_linalg_policy(monkeypatch):
    """PepsOptimizer uses the supplied class for SVD and QR registration."""
    registered = []

    def _register(policy):
        registered.append(policy)
        return policy

    monkeypatch.setattr(TorchLinalgConfig, "register", _register)
    policy = TorchLinalgConfig(
        mode="complex",
        stabilized=False,
        svd_driver="gesvdj",
        cpu_svd="scipy_gesdd",
    )

    opt = PepsOptimizer(
        DummyState(bond=1),
        [],
        chi=3,
        mode="global",
        torch_linalg_config=policy,
        normalize_initial=False,
    )
    opt._maybe_configure_torch_linalg(  # pylint: disable=protected-access
        opt.state,
        opt.state,
        {"autodiff_backend": "torch"},
    )

    assert registered == [policy]
    assert registered[0].exact is True
    assert registered[0].stabilized is False


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
    monkeypatch.setattr(peps_mod.TorchLinalgConfig, "register", lambda self: self)
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
    monkeypatch.setattr(peps_mod.TorchLinalgConfig, "register", lambda self: self)
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


def test_peps_optimizer_fidelity_trace_stays_finite_after_zero(monkeypatch):
    """A zero local-fidelity estimate should not poison later trace diagnostics."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([1.0, 0.01])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )

    gates = [
        ({"bond": 8, "label": "zero"}, ((0, 0), (0, 1))),
        ({"bond": 8, "label": "good"}, ((1, 0), (1, 1))),
    ]
    opt = PepsOptimizer(DummyState(bond=1), gates, chi=3, normalize_initial=False)

    opt.run(progress=False, optimize=False)

    assert opt.local_infidelities == [pytest.approx(1.0), pytest.approx(0.01)]
    assert opt.step_records[0]["fidelity"] == pytest.approx(0.0)
    assert opt.step_records[0]["geometric_fidelity"] == pytest.approx(
        peps_mod._TRACE_FIDELITY_FLOOR
    )
    assert opt.get_fidelities()[-1] == pytest.approx(
        (peps_mod._TRACE_FIDELITY_FLOOR * 0.99) ** 0.5
    )
    assert opt.get_infidelities()[-1] < 1.0


def test_peps_optimizer_run_resets_traces_by_default(monkeypatch):
    """Repeated run calls should start fresh diagnostics unless opted out."""
    _install_fake_gate(monkeypatch)
    _install_fake_normalize(monkeypatch)
    infidelities = iter([0.1, 0.2, 0.3])
    monkeypatch.setattr(
        peps_mod,
        "boundary_infidelity",
        lambda *args, **kwargs: {"infidelity": next(infidelities)},
    )

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    opt.run(progress=False, optimize=False)
    assert opt.local_infidelities == [pytest.approx(0.1)]

    opt.run(progress=False, optimize=False, reset_traces=False)
    assert opt.local_infidelities == [pytest.approx(0.1), pytest.approx(0.2)]
    assert len(opt.step_records) == 2

    opt.run(progress=False, optimize=False)
    assert opt.local_infidelities == [pytest.approx(0.3)]
    assert len(opt.step_records) == 1


def test_peps_optimizer_progress_reports_geometric_infidelity(monkeypatch):
    """Progress bar should expose a compact MpsOptimizer-style postfix."""
    import tqdm.auto as tqdm_auto

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

    monkeypatch.setattr(tqdm_auto, "tqdm", _FakeTqdm)

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
    assert progress.kwargs["unit"] == "gate"
    assert progress.kwargs["dynamic_ncols"] is True
    assert progress.kwargs["mininterval"] == pytest.approx(0.2)
    # The bar reports gate position/status, cumulative infidelity, the current
    # input/local values, and optimizer gain (pre-optimization minus final).
    assert set(last.keys()) == {
        "2q", "bnd", "step", "status", "infidelity", "input",
        "local_infidelity", "opt_gain",
    }
    assert last["2q"] == 1
    assert last["bnd"] == 3
    assert last["step"] == "1/1"
    assert last["status"] == "below_tol"
    assert last["input"] == PepsOptimizer._format_progress_infidelity(1.0e-6)
    assert last["infidelity"] == PepsOptimizer._format_progress_infidelity(
        opt.get_infidelities()[-1]
    )
    assert last["local_infidelity"] == PepsOptimizer._format_progress_infidelity(1.0e-6)
    assert last["opt_gain"] == PepsOptimizer._format_progress_infidelity(0.0)
    # Only one postfix update per gate step, matching MpsOptimizer.
    assert len(progress.postfix_calls) == 1


def test_peps_optimizer_progress_reports_optimizer_gain():
    """The progress postfix should expose positive local cleanup improvement."""
    opt = PepsOptimizer(DummyState(bond=1), [], chi=2, normalize_initial=False)
    opt._fidelity_count = 1
    opt.infidelities = [0.0, 0.03]
    opt.step_records = [{
        "pre_infidelity": 0.2,
        "final_infidelity": 0.04,
    }]

    postfix = opt._progress_postfix(two_site_count=1)

    assert postfix["infidelity"] == PepsOptimizer._format_progress_infidelity(0.03)
    assert postfix["local_infidelity"] == PepsOptimizer._format_progress_infidelity(0.04)
    assert postfix["opt_gain"] == PepsOptimizer._format_progress_infidelity(0.16)


def test_peps_optimizer_progress_reports_sweep_input_output():
    """Sweep progress should distinguish boundary input/output from final acceptance."""
    opt = PepsOptimizer(DummyState(bond=1), [], chi=2, normalize_initial=False)
    opt._fidelity_count = 1
    opt.infidelities = [0.0, 0.03]
    opt.step_records = [{
        "pre_infidelity": 0.2,
        "final_infidelity": 0.04,
        "optimizer_result": {
            "loss_before": 0.2,
            "loss_after": 0.05,
            "best_loss": 0.03,
            "n_runs": 4,
        },
    }]

    postfix = opt._progress_postfix(two_site_count=1)

    assert postfix["input"] == PepsOptimizer._format_progress_infidelity(0.2)
    assert postfix["sweep"] == (
        f"{PepsOptimizer._format_progress_infidelity(0.2)}"
        f"->{PepsOptimizer._format_progress_infidelity(0.05)}"
    )
    assert postfix["slice_best"] == PepsOptimizer._format_progress_infidelity(0.03)
    assert postfix["slices"] == 4
    assert postfix["local_infidelity"] == PepsOptimizer._format_progress_infidelity(0.04)
    assert postfix["opt_gain"] == PepsOptimizer._format_progress_infidelity(0.16)


def test_peps_optimizer_inner_sweep_progress_shows_directional_moves():
    """Outer progress enables a separate slice-level directional sweep bar."""
    opt = PepsOptimizer(DummyState(bond=1), [], chi=2, normalize_initial=False)

    _init_kwargs, opt_kwargs, strip_exponent = opt._sweep_boundary_kwargs(progress=True)

    assert opt_kwargs["progress"] is True
    assert opt_kwargs["progress_position"] == 1
    assert opt_kwargs["progress_leave"] is False
    assert strip_exponent is True

    _init_kwargs, opt_kwargs, _strip_exponent = opt._sweep_boundary_kwargs(
        progress=True,
        sweep_progress=False,
    )
    assert opt_kwargs["progress"] is False
    assert opt_kwargs["progress_position"] == 0

    configured = PepsOptimizer(
        DummyState(bond=1), [], chi=2, normalize_initial=False, sweep_progress=False
    )
    assert configured.sweep_progress is False

    direction_label = peps_mod.SweepOptimizer._sweep_direction_label
    assert direction_label("x", "forward") == "right"
    assert direction_label("x", "backward") == "left"
    assert direction_label("y", "forward") == "up"
    assert direction_label("y", "backward") == "down"


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


def test_peps_optimizer_symmray_sweep_uses_quimb_mps_boundaries(monkeypatch):
    """Symmray sweep cleanup should request Quimb MPS sweep environments."""
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
            self.state = state

        def set_optimize_kwargs(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return self

        def run(self):
            return {
                "best_loss": 0.04,
                "best_state": SymmrayDummyState(bond=3, name="best"),
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    opt = PepsOptimizer(
        SymmrayDummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        normalize_initial=False,
    )

    out = opt.run(progress=False, infidelity_tol=1.0e-9)

    assert out.name == "best"
    assert captured["kwargs"]["boundary_engine"] == "quimb-mps"
    assert captured["kwargs"]["normalize_kwargs"]["method"] == "mps"
    assert captured["kwargs"]["normalize_kwargs"]["mode_"] == "mps"
    assert captured["kwargs"]["normalize_kwargs"]["balance_bonds"] is False
    assert captured["kwargs"]["simplify"] is False
    assert captured["optimize_kwargs"]["env_n_iter"] == 10
    assert captured["optimize_kwargs"]["optimizer"] == "nlopt"
    assert captured["optimize_kwargs"]["optimizer_options"]["algorithm"] == "LD_LBFGS"


def test_sweep_scaled_scalar_preserves_torch_exponent_gradient():
    """Scaled Torch exponents must remain in the differentiable loss path."""
    torch = pytest.importorskip("torch")
    exponent = torch.tensor(3.0, requires_grad=True)
    mantissa, exponent_out = peps_mod.SweepOptimizer._as_scaled_scalar(
        (1.0, exponent),
        name="target_norm",
    )
    assert mantissa == 1.0
    assert exponent_out is exponent

    fidelity = peps_mod.SweepOptimizer._scaled_overlap_fidelity(
        (torch.tensor(1.0), exponent),
        (torch.tensor(1.0), 0.0),
        (1.0, 0.0),
    )
    (1.0 - fidelity).backward()
    assert exponent.grad is not None
    assert float(exponent.grad.abs()) > 0.0


def test_symmray_fermionic_local_bra_preserves_full_norm():
    """Fermionic local bras must use the full-network conjugation context."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")

    import pepsy as py
    from pepsy.boundary.metrics import build_bra_ket
    from pepsy.tensors import SymPEPS

    py.reg_rel_svd_torch()
    to_backend = py.build_backend(
        device="cpu",
        dtype=torch.complex128,
        requires_grad=False,
        set_default=False,
    )
    contraction = py.build_contraction(parallel=False, progbar=False)
    site_charge = py.site_charge_alternating((1, 0), (0, 1))
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        bond_dim=2,
        phys_dim=4,
        fermionic=True,
        site_charge=site_charge,
        seed=202,
        dtype="complex128",
        to_backend=to_backend,
        contraction_opt=contraction,
    ).peps
    target = state.copy()
    target.mangle_inner_("_target")

    _, norm = build_bra_ket(ket=state, bra=None)
    _, overlap = build_bra_ket(ket=target, bra=state)
    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        target_norm=1.0,
        chi=64,
        contraction_opt=contraction,
        boundary_engine="quimb-mps",
        simplify=False,
        renormalize_state=False,
    )
    local = state.select(["Y1"], "any")
    local_target = target.select(["Y1"], "any")
    bra_norm, bra_overlap = sweep._symmray_local_bras(local)

    def scalar(tn):
        value = tn.contract(all, optimize=contraction, strip_exponent=False)
        value = sweep._unwrap_contracted_scalar(value, name="norm")
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return complex(value)

    full_norm = scalar(norm)
    split_norm = scalar(norm.select(["Y0"], "any") | norm.select(["Y1"], "any"))
    local_norm = scalar(norm.select(["Y0"], "any") | local | bra_norm)
    full_overlap = scalar(overlap)
    local_overlap = scalar(
        overlap.select(["Y0"], "any") | local_target | bra_overlap
    )

    assert split_norm == pytest.approx(full_norm, rel=1.0e-12, abs=1.0e-12)
    assert local_norm == pytest.approx(full_norm, rel=1.0e-12, abs=1.0e-12)
    assert local_overlap == pytest.approx(full_overlap, rel=1.0e-12, abs=1.0e-12)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_symmray_fermionic_sweep_improves_global_infidelity(symmetry, axis):
    """A real fermionic sweep must improve both lattice directions."""
    pytest.importorskip("torch")
    pytest.importorskip("symmray")

    import pepsy as py
    from pepsy.tensors import SymPEPS

    py.reg_rel_svd_torch()
    contraction = py.build_contraction(parallel=False, progbar=False)
    kwargs = {
        "symmetry": symmetry,
        "bond_dim": 2,
        "phys_dim": 4 if symmetry == "U1U1" else 2,
        "fermionic": True,
        "dtype": "complex128",
        "contraction_opt": contraction,
    }
    if symmetry == "U1U1":
        kwargs["site_charge"] = py.site_charge_alternating((1, 0), (0, 1))

    state = SymPEPS.random(2, 2, seed=7, **kwargs).peps
    target = SymPEPS.random(2, 2, seed=107, **kwargs).peps
    target.mangle_inner_("_target")
    target_norm = complex(
        np.asarray((target.H & target).contract(all, optimize=contraction)).item()
    )
    before = float(1.0 - py.tn_fidelity(state, target, contraction_opt=contraction).real)

    sweep = peps_mod.SweepOptimizer(
        state,
        target,
        target_norm=target_norm,
        chi=64,
        contraction_opt=contraction,
        boundary_engine="quimb-mps",
        simplify=False,
        renormalize_state=False,
    )
    sweep.set_optimize_kwargs(
        axes=(axis,),
        n_cycles=1,
        n_round_trips=1,
        optimizer="scipy",
        optimizer_options={"n_steps": 3, "maxiter": 25},
        env_n_iter=4,
        progress=False,
        renormalize=False,
    )
    result = sweep.run()
    after = float(1.0 - py.tn_fidelity(state, target, contraction_opt=contraction).real)

    assert result["runs"]
    assert after < before


def test_peps_optimizer_explicit_quimb_boundary_engine_forwards_options(monkeypatch):
    """Dense PEPS cleanup can opt into Quimb MPS sweep boundaries explicitly."""
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
            self.state = state

        def set_optimize_kwargs(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return self

        def run(self):
            return {
                "best_loss": 0.04,
                "best_state": DummyState(bond=3, name="best"),
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    opt = PepsOptimizer(
        DummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        boundary_engine="quimb-mps",
        boundary_options={"cutoff": 1.0e-10, "canonize": False},
        normalize_initial=False,
    )

    out = opt.run(progress=False, infidelity_tol=1.0e-9)

    assert out.name == "best"
    assert captured["kwargs"]["boundary_engine"] == "quimb-mps"
    assert captured["kwargs"]["boundary_options"] == {
        "cutoff": 1.0e-10,
        "canonize": False,
    }
    assert captured["kwargs"]["normalize_kwargs"]["method"] == "mps"
    assert captured["kwargs"]["normalize_kwargs"]["mode_"] == "mps"
    assert captured["kwargs"]["normalize_kwargs"]["balance_bonds"] is False


def test_peps_optimizer_explicit_dmrg_boundary_engine_overrides_symmray_auto(monkeypatch):
    """An explicit DMRG selector should not be rewritten to Quimb MPS."""
    _install_fake_gate(monkeypatch)
    norm_calls = _install_fake_normalize(monkeypatch)
    infidelity_calls = []
    infidelities = iter([0.2, 0.03])

    def _fake_infidelity(*args, **kwargs):
        infidelity_calls.append(dict(kwargs))
        return {"infidelity": next(infidelities)}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", _fake_infidelity)
    captured = {}

    class _FakeSweep:
        def __init__(self, *, state, state_target, **kwargs):
            captured["state"] = state
            captured["state_target"] = state_target
            captured["kwargs"] = dict(kwargs)
            self.state = state

        def set_optimize_kwargs(self, **kwargs):
            captured["optimize_kwargs"] = dict(kwargs)
            return self

        def run(self):
            return {
                "best_loss": 0.04,
                "best_state": SymmrayDummyState(bond=3, name="best"),
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", _FakeSweep)

    opt = PepsOptimizer(
        SymmrayDummyState(bond=1),
        [({"bond": 8}, ((0, 0), (0, 1)))],
        chi=3,
        boundary_engine="dmrg",
        normalize_initial=False,
    )

    out = opt.run(progress=False, infidelity_tol=1.0e-9)

    assert out.name == "best"
    assert captured["kwargs"]["boundary_engine"] == "dmrg"
    assert "method" not in captured["kwargs"]["normalize_kwargs"]
    assert "method" not in infidelity_calls[0]
    assert "method" not in norm_calls[0][1]


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

    _ = opt.run(progress=False, normalize_target=False)

    assert gate_calls[0]["which"] == "lower"
