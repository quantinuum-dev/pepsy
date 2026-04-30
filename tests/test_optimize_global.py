"""Smoke/behavior tests for :mod:`pepsy.optimize_global`."""

import pytest
import quimb.tensor as qtn

from pepsy.optimize_global import GlobalOptimizer


def _rand_peps(seed: int):
    return qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=seed, dtype="complex128")


def test_norm_state_exact_returns_positive_value():
    """Exact-mode norm should return a finite positive scalar-like value."""
    peps = _rand_peps(seed=11)
    value = complex(GlobalOptimizer._norm_state(peps, mode="exact", opt="auto-hq"))
    assert abs(value) > 0.0


def test_normalize_state_is_inplace_and_unit_norm():
    """_normalize_state should mutate and return the same state object."""
    peps = _rand_peps(seed=12)
    out = GlobalOptimizer._normalize_state(peps, mode="exact", opt="auto-hq")
    assert out is peps
    value = complex(GlobalOptimizer._norm_state(peps, mode="exact", opt="auto-hq"))
    assert abs(value - 1.0) < 1e-6


def test_loss_state_identical_is_near_zero():
    """Fidelity-style loss should be near zero for identical states."""
    peps = _rand_peps(seed=13)
    peps_fix = peps.copy()
    val_ref = abs(complex(GlobalOptimizer._norm_state(peps_fix.copy(), mode="exact", opt="auto-hq")))
    loss = float(
        GlobalOptimizer._loss_state(
            peps,
            peps_fix,
            mode="exact",
            opt="auto-hq",
            cost_f="fid",
            val_=val_ref,
        )
    )
    assert loss < 1e-6


def test_norm_state_unknown_mode_raises():
    """Unknown norm mode should raise ValueError."""
    peps = _rand_peps(seed=14)
    with pytest.raises(ValueError, match="Unknown mode"):
        GlobalOptimizer._norm_state(peps, mode="unknown")


def test_loss_state_unknown_cost_raises():
    """Unknown loss function selector should raise ValueError."""
    peps = _rand_peps(seed=15)
    peps_fix = _rand_peps(seed=16)
    with pytest.raises(ValueError, match="Unknown cost function"):
        GlobalOptimizer._loss_state(
            peps,
            peps_fix,
            mode="exact",
            opt="auto-hq",
            cost_f="unknown",
            val_=1.0,
        )


def test_global_optimizer_builds_tnoptimizer():
    """GlobalOptimizer should build a TNOptimizer with merged loss kwargs."""
    peps = _rand_peps(seed=17)
    peps_target = _rand_peps(seed=18)
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    tnopt = opt.make_tn_optimizer(optimizer="adam", progbar=False, loss_kwargs={"cost_f": "fid", "val_": 1.0})
    assert isinstance(tnopt, qtn.TNOptimizer)


def test_global_optimizer_optimize_smoke():
    """Optimize should run a short TNOptimizer step and return a PEPS-like TN."""
    pytest.importorskip("torch")
    peps = _rand_peps(seed=19)
    peps_target = peps.copy()
    ref = abs(complex(GlobalOptimizer._norm_state(peps_target.copy(), mode="exact", opt="auto-hq")))
    opt = GlobalOptimizer(
        peps,
        peps_target,
        loss_kwargs={"mode": "exact", "opt": "auto-hq", "cost_f": "fid", "val_": ref},
    )
    out = opt.optimize(n=1, optimizer="adam", progbar=False, autodiff_backend="torch")
    assert out is not None


def test_global_optimizer_alias_methods_match_module_helpers():
    """Public class methods should match internal objective kernels."""
    peps = _rand_peps(seed=20)
    peps_target = peps.copy()
    norm_ref = abs(complex(GlobalOptimizer._norm_state(peps_target.copy(), mode="exact", opt="auto-hq")))
    opt = GlobalOptimizer(
        peps,
        peps_target,
        norm_kwargs={"mode": "exact", "opt": "auto-hq"},
        loss_kwargs={"mode": "exact", "opt": "auto-hq", "cost_f": "fid", "val_": norm_ref},
    )

    class_norm = complex(opt.norm())
    ref_norm = complex(GlobalOptimizer._norm_state(peps, mode="exact", opt="auto-hq"))
    assert abs(class_norm - ref_norm) < 1e-10

    class_loss = float(opt.loss())
    ref_loss = float(
        GlobalOptimizer._loss_state(
            peps,
            peps_target,
            mode="exact",
            opt="auto-hq",
            cost_f="fid",
            val_=norm_ref,
        )
    )
    assert abs(class_loss - ref_loss) < 1e-10

    out = opt.normalize()
    assert out is peps
    assert abs(complex(opt.norm()) - 1.0) < 1e-6


def test_global_optimizer_norm_filters_loss_only_kwargs():
    """norm() should not forward loss-only keys like cost_f/val_."""
    peps = _rand_peps(seed=21)
    peps_target = _rand_peps(seed=22)
    opt = GlobalOptimizer(
        peps,
        peps_target,
        loss_kwargs={
            "mode": "exact",
            "opt": "auto-hq",
            "cost_f": "fid",
            "val_": 1.0,
        },
    )
    # Should not raise TypeError from unexpected cost_f/val_ in _norm_state.
    value = complex(opt.norm())
    assert abs(value) > 0.0


def test_global_optimizer_warns_on_unknown_kwargs():
    """Unknown option keys should emit a warning and be ignored."""
    peps = _rand_peps(seed=23)
    peps_target = _rand_peps(seed=24)
    with pytest.warns(UserWarning, match="Ignoring unknown options"):
        _ = GlobalOptimizer(
            peps,
            peps_target,
            loss_kwargs={"mode": "exact", "opt": "auto-hq", "not_a_key": 123},
        )


def test_global_optimizer_kwarg_metadata_helpers():
    """GlobalOptimizer should expose kwarg-name helpers mirroring Sweep style."""
    norm_keys = GlobalOptimizer.norm_kwarg_names()
    normalize_keys = GlobalOptimizer.normalize_kwarg_names()
    loss_keys = GlobalOptimizer.loss_kwarg_names()
    guide = GlobalOptimizer.kwarg_guide()

    assert "mode" in norm_keys
    assert "opt" in norm_keys
    assert "contraction_opt" in norm_keys
    assert "contraction_opt_hyper" in norm_keys
    assert norm_keys == normalize_keys
    assert "cost_f" in loss_keys
    assert "val_" in loss_keys
    assert tuple(guide["norm"]) == norm_keys
    assert tuple(guide["normalize"]) == normalize_keys
    assert tuple(guide["loss"]) == loss_keys


def test_global_optimizer_accepts_canonical_contraction_kwargs():
    """Canonical contraction keyword names should work for norm/loss."""
    peps = _rand_peps(seed=39)
    peps_target = _rand_peps(seed=40)
    val_ref = abs(complex(GlobalOptimizer._norm_state(peps_target.copy(), mode="exact", opt="auto-hq")))
    opt = GlobalOptimizer(
        peps,
        peps_target,
        norm_kwargs={"mode": "exact", "contraction_opt": "auto-hq"},
        loss_kwargs={
            "mode": "exact",
            "contraction_opt": "auto-hq",
            "cost_f": "fid",
            "val_": val_ref,
        },
    )
    class_norm = complex(opt.norm())
    ref_norm = complex(GlobalOptimizer._norm_state(peps, mode="exact", opt="auto-hq"))
    assert abs(class_norm - ref_norm) < 1e-10

    class_loss = float(opt.loss())
    assert class_loss >= 0.0


def test_global_optimizer_optional_target_allows_norm_and_normalize():
    """state_target should be optional for norm/normalize workflows."""
    peps = _rand_peps(seed=25)
    opt = GlobalOptimizer(peps)
    val = complex(opt.norm(mode="exact", opt="auto-hq"))
    assert abs(val) > 0.0
    out = opt.normalize(mode="exact", opt="auto-hq")
    assert out is peps


def test_global_optimizer_optional_target_blocks_loss_methods():
    """loss and TNOptimizer construction should require a target state."""
    peps = _rand_peps(seed=26)
    opt = GlobalOptimizer(peps)
    with pytest.raises(ValueError, match="state_target is required for loss"):
        _ = opt.loss()
    with pytest.raises(ValueError, match="state_target is required for make_tn_optimizer"):
        _ = opt.make_tn_optimizer()


def test_global_optimizer_set_target_replaces_and_clears_target():
    """set_target should replace, clear, and support fluent chaining."""
    peps = _rand_peps(seed=41)
    target_a = _rand_peps(seed=42)
    target_b = _rand_peps(seed=43)
    opt = GlobalOptimizer(peps, target_a)

    out = opt.set_target(target_b)
    assert out is opt
    assert opt.state_target is target_b

    out = opt.set_target(None)
    assert out is opt
    assert opt.state_target is None
    with pytest.raises(ValueError, match="state_target is required for loss"):
        _ = opt.loss()


def test_global_optimizer_set_target_inplace_false_copies():
    """set_target(inplace=False) should copy when requested."""
    peps = _rand_peps(seed=44)
    target = _rand_peps(seed=45)
    opt = GlobalOptimizer(peps)

    _ = opt.set_target(target, inplace=False)
    assert opt.state_target is not target


@pytest.mark.parametrize("name", ["LBFGS", "lbfgs"])
def test_make_tn_optimizer_normalizes_lbfgs_aliases(monkeypatch, name):
    """LBFGS aliases should be normalized to ``L-BFGS-B`` for TNOptimizer."""
    peps = _rand_peps(seed=27)
    peps_target = _rand_peps(seed=28)
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    called = {}

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            _ = args
            called["optimizer"] = kwargs.get("optimizer")

    monkeypatch.setattr("pepsy.optimize_global.qtn.TNOptimizer", _FakeTNOptimizer)
    _ = opt.make_tn_optimizer(optimizer=name, progbar=False, loss_kwargs={"cost_f": "fid", "val_": 1.0})
    assert called["optimizer"] == "L-BFGS-B"


def test_global_optimizer_optimize_nlopt_forwards_args_and_updates_state(monkeypatch):
    """optimize_nlopt should forward controls and update ``state``."""
    peps = _rand_peps(seed=29)
    peps_target = _rand_peps(seed=30)
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    called = {}
    sentinel_out = object()

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            _ = args
            called["optimizer"] = kwargs.get("optimizer")

        def optimize_nlopt(self, **kwargs):
            called["nlopt_kwargs"] = dict(kwargs)
            return sentinel_out

    monkeypatch.setattr("pepsy.optimize_global.qtn.TNOptimizer", _FakeTNOptimizer)
    out = opt.optimize_nlopt(
        n=7,
        tol=1e-4,
        jac=True,
        hessp=False,
        ftol_rel=1e-5,
        ftol_abs=1e-6,
        xtol_rel=1e-7,
        xtol_abs=1e-8,
        optimizer="nlopt",
        progbar=False,
        loss_kwargs={"cost_f": "fid", "val_": 1.0},
    )

    assert called["optimizer"] == "L-BFGS-B"
    assert called["nlopt_kwargs"] == {
        "n": 7,
        "tol": 1e-4,
        "jac": True,
        "hessp": False,
        "ftol_rel": 1e-5,
        "ftol_abs": 1e-6,
        "xtol_rel": 1e-7,
        "xtol_abs": 1e-8,
    }
    assert out is sentinel_out
    assert opt.state is sentinel_out


def test_global_optimizer_optimize_nlopt_has_sensible_default_tolerances(monkeypatch):
    """optimize_nlopt should use stable default relative tolerances."""
    peps = _rand_peps(seed=31)
    peps_target = _rand_peps(seed=32)
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    called = {}
    sentinel_out = object()

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def optimize_nlopt(self, **kwargs):
            called["nlopt_kwargs"] = dict(kwargs)
            return sentinel_out

    monkeypatch.setattr("pepsy.optimize_global.qtn.TNOptimizer", _FakeTNOptimizer)
    out = opt.optimize_nlopt(
        n=3,
        optimizer="nlopt",
        progbar=False,
        loss_kwargs={"cost_f": "fid", "val_": 1.0},
    )
    assert called["nlopt_kwargs"]["ftol_rel"] == pytest.approx(1e-9)
    assert called["nlopt_kwargs"]["xtol_rel"] == pytest.approx(1e-9)
    assert called["nlopt_kwargs"]["ftol_abs"] == pytest.approx(0.0)
    assert called["nlopt_kwargs"]["xtol_abs"] == pytest.approx(0.0)
    assert out is sentinel_out
    assert opt.state is sentinel_out


def test_global_optimizer_optimize_rejects_nlopt_optimizer_name():
    """optimize should direct users to optimize_nlopt for NLopt routes."""
    peps = _rand_peps(seed=33)
    peps_target = _rand_peps(seed=34)
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    with pytest.raises(ValueError, match="optimize_nlopt"):
        _ = opt.optimize(n=1, optimizer="nlopt", progbar=False)


def test_global_optimizer_optimize_can_return_losses(monkeypatch):
    """optimize(..., return_losses=True) should return TNOptimizer losses."""
    peps = _rand_peps(seed=35)
    peps_target = peps.copy()
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    sentinel_out = object()

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.losses = [1.0, 0.5, 0.1]

        def optimize(self, **kwargs):
            _ = kwargs
            return sentinel_out

    monkeypatch.setattr("pepsy.optimize_global.qtn.TNOptimizer", _FakeTNOptimizer)
    out, losses = opt.optimize(
        n=3,
        optimizer="adam",
        progbar=False,
        loss_kwargs={"cost_f": "fid", "val_": 1.0},
        return_losses=True,
    )
    assert out is sentinel_out
    assert losses == (1.0, 0.5, 0.1)
    assert opt.state is sentinel_out


def test_global_optimizer_optimize_nlopt_can_return_losses(monkeypatch):
    """optimize_nlopt(..., return_losses=True) should return TNOptimizer losses."""
    peps = _rand_peps(seed=36)
    peps_target = peps.copy()
    opt = GlobalOptimizer(peps, peps_target, loss_kwargs={"mode": "exact", "opt": "auto-hq"})
    sentinel_out = object()

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.losses = [0.9, 0.4]

        def optimize_nlopt(self, **kwargs):
            _ = kwargs
            return sentinel_out

    monkeypatch.setattr("pepsy.optimize_global.qtn.TNOptimizer", _FakeTNOptimizer)
    out, losses = opt.optimize_nlopt(
        n=2,
        optimizer="nlopt",
        progbar=False,
        loss_kwargs={"cost_f": "fid", "val_": 1.0},
        return_losses=True,
    )
    assert out is sentinel_out
    assert losses == (0.9, 0.4)
    assert opt.state is sentinel_out
