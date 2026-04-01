"""Smoke/behavior tests for :mod:`pepsy.optimize_global`."""

import pytest
import quimb.tensor as qtn

from pepsy.optimize_global import GlobalOptimizer


def _rand_peps(seed: int):
    return qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=seed, dtype="complex128")


def test_norm_peps_exact_returns_positive_value():
    """Exact-mode norm should return a finite positive scalar-like value."""
    peps = _rand_peps(seed=11)
    value = complex(GlobalOptimizer._norm_peps(peps, mode="exact", opt="auto-hq"))
    assert abs(value) > 0.0


def test_normalize_peps_is_inplace_and_unit_norm():
    """_normalize_peps should mutate and return the same PEPS object."""
    peps = _rand_peps(seed=12)
    out = GlobalOptimizer._normalize_peps(peps, mode="exact", opt="auto-hq")
    assert out is peps
    value = complex(GlobalOptimizer._norm_peps(peps, mode="exact", opt="auto-hq"))
    assert abs(value - 1.0) < 1e-6


def test_loss_peps_identical_is_near_zero():
    """Fidelity-style loss should be near zero for identical states."""
    peps = _rand_peps(seed=13)
    peps_fix = peps.copy()
    val_ref = abs(complex(GlobalOptimizer._norm_peps(peps_fix.copy(), mode="exact", opt="auto-hq")))
    loss = float(
        GlobalOptimizer._loss_peps(
            peps,
            peps_fix,
            mode="exact",
            opt="auto-hq",
            cost_f="fid",
            val_=val_ref,
        )
    )
    assert loss < 1e-6


def test_norm_peps_unknown_mode_raises():
    """Unknown norm mode should raise ValueError."""
    peps = _rand_peps(seed=14)
    with pytest.raises(ValueError, match="Unknown mode"):
        GlobalOptimizer._norm_peps(peps, mode="unknown")


def test_loss_peps_unknown_cost_raises():
    """Unknown loss function selector should raise ValueError."""
    peps = _rand_peps(seed=15)
    peps_fix = _rand_peps(seed=16)
    with pytest.raises(ValueError, match="Unknown cost function"):
        GlobalOptimizer._loss_peps(
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
    peps = _rand_peps(seed=19)
    peps_target = peps.copy()
    ref = abs(complex(GlobalOptimizer._norm_peps(peps_target.copy(), mode="exact", opt="auto-hq")))
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
    norm_ref = abs(complex(GlobalOptimizer._norm_peps(peps_target.copy(), mode="exact", opt="auto-hq")))
    opt = GlobalOptimizer(
        peps,
        peps_target,
        norm_kwargs={"mode": "exact", "opt": "auto-hq"},
        loss_kwargs={"mode": "exact", "opt": "auto-hq", "cost_f": "fid", "val_": norm_ref},
    )

    class_norm = complex(opt.norm())
    ref_norm = complex(GlobalOptimizer._norm_peps(peps, mode="exact", opt="auto-hq"))
    assert abs(class_norm - ref_norm) < 1e-10

    class_loss = float(opt.loss())
    ref_loss = float(
        GlobalOptimizer._loss_peps(
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
    # Should not raise TypeError from unexpected cost_f/val_ in _norm_peps.
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
