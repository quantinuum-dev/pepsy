"""Tests for tree-network energy measurement and optimization."""

from types import SimpleNamespace

import numpy as np
import pytest

import pepsy


def _tree_state():
    plan = pepsy.TreePlan.from_order(range(2), structure="balanced")
    return pepsy.TreeTensorNetwork.from_plan(plan, dtype=complex)


def test_tree_energy_optimizer_reports_energy():
    state = _tree_state()
    z = np.diag([1.0, -1.0]).astype(complex)
    optimizer = pepsy.TreeEnergyOptimizer(
        state,
        terms={0: -z, 1: -z},
        energy_per_site=False,
        contraction_opt="auto-hq",
    )

    estimate = optimizer.energy()

    assert estimate.energy == pytest.approx(-2.0)
    assert estimate.energy_per_site == pytest.approx(-1.0)
    assert estimate.boundary_mode == "exact"


@pytest.mark.filterwarnings("ignore:The contraction tree is not a compressed one")
def test_tree_energy_optimizer_supports_tn_optimization():
    pytest.importorskip("torch")
    state = _tree_state()
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    optimizer = pepsy.TreeEnergyOptimizer(
        state,
        terms={0: -x, 1: -x},
        energy_per_site=False,
        contraction_opt="auto-hq",
    )
    before = float(optimizer.energy().energy)

    out, losses = optimizer.optimize(
        n=1,
        autodiff_backend="torch",
        progbar=False,
        return_losses=True,
    )

    assert isinstance(out, pepsy.TreeTensorNetwork)
    assert out.validate(check_canonical=True) is out
    assert losses
    assert np.isfinite(float(losses[-1]))
    after = float(optimizer.energy().energy)
    assert np.isfinite(after)
    assert after < before


def test_tree_energy_optimizer_fermion_tnopt_refreshes_norm():
    """Autodiff on native U1 trees uses a fresh Rayleigh-quotient norm."""
    pytest.importorskip("symmray")
    pytest.importorskip("torch")

    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1",
        dtype="complex128",
    )
    plan = pepsy.TreePlan.from_order(range(2), structure="balanced")
    state = pepsy.ps_to_ttn(
        2,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1)),
        dtype="complex128",
    )
    optimizer = pepsy.TreeEnergyOptimizer(
        state,
        terms={0: -fermion.observable("sx"), 1: -fermion.observable("sx")},
        normalized=True,
        energy_per_site=False,
        contraction_opt="auto-hq",
    )
    before = float(optimizer.energy().energy)

    out, losses = optimizer.optimize(
        n=8,
        optimizer="l-bfgs-b",
        autodiff_backend="torch",
        progbar=False,
        return_losses=True,
    )

    values = np.asarray(losses, dtype=float)
    assert isinstance(out, pepsy.TreeTensorNetwork)
    assert values.size
    assert np.all(np.isfinite(values))
    assert np.min(values) >= -2.0
    assert values[-1] <= before + 1.0e-10
    assert out.orthogonality_center is None
    assert np.isfinite(float(optimizer.energy().energy))


@pytest.mark.smoke
def test_peps_energy_optimizer_returns_best_state_on_nlopt_stop(monkeypatch):
    """NLopt exceptions return the best finite checkpoint, not the last trial."""
    nlopt = pytest.importorskip("nlopt")

    class FakeHandler:
        def __init__(self):
            self.calls = 0

        def value_and_grad(self, arrays):
            loss = (3.0, 1.0, 2.0)[self.calls]
            self.calls += 1
            return loss, np.zeros_like(arrays)

    class FakeTNOptimizer:
        def __init__(self):
            self.vectorizer = SimpleNamespace(vector=np.array([0.0]))
            self.handler = FakeHandler()
            self.losses = []

        def optimize(self, n, **kwargs):
            _ = n, kwargs
            for value in (0.0, 1.0, 2.0):
                self.vectorizer.vector[:] = value
                loss, _ = self.handler.value_and_grad(self.vectorizer.vector)
                self.losses.append(loss)
            raise nlopt.runtime_error("roundoff limited")

        def get_tn_opt(self):
            return self.vectorizer.vector.copy()

    optimizer = object.__new__(pepsy.PepsEnergyOptimizer)
    optimizer.loss_kwargs = {}
    optimizer.state = object()
    fake = FakeTNOptimizer()
    monkeypatch.setattr(optimizer, "make_tn_optimizer", lambda **kwargs: fake)
    monkeypatch.setattr(
        optimizer,
        "_state_for_autodiff_backend",
        lambda out, *args, **kwargs: out,
    )

    with pytest.warns(RuntimeWarning, match="best finite checkpoint"):
        out, losses = optimizer.optimize(
            n=3,
            optimizer="L-BFGS-B",
            optlib="nlopt",
            check_finite_gradient=False,
            progbar=False,
            return_losses=True,
        )

    assert np.array_equal(out, np.array([1.0]))
    assert losses == (3.0, 1.0, 2.0)
    assert np.array_equal(optimizer.state, np.array([1.0]))
