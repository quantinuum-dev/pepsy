"""Tests for tree-network energy measurement and optimization."""

from types import SimpleNamespace

import numpy as np
import pytest

import pepsy


def _tree_state():
    plan = pepsy.TreePlan.from_order(range(2), structure="balanced")
    return pepsy.TreeTensorNetwork.from_plan(plan, dtype=complex)


def test_energy_optimizers_preserve_supplied_torch_linalg_policy(monkeypatch):
    """Energy optimizers pass one user policy through their Torch setup."""
    policy = pepsy.TorchLinalgConfig(
        mode="real",
        stabilized=False,
        svd_driver="gesvdj",
    )
    registered = []

    def record_register(config):
        registered.append(config)
        return config

    monkeypatch.setattr(pepsy.TorchLinalgConfig, "register", record_register)
    pepsy.PepsEnergyOptimizer._configure_torch_linalg(
        None,
        {},
        quimb_split_drivers=True,
        torch_linalg_config=policy,
    )
    pepsy.MpsEnergyOptimizer._configure_torch_linalg(
        None,
        {},
        quimb_split_drivers=False,
        torch_linalg_config=policy,
    )

    assert registered[0] is not policy
    assert registered[0].quimb_split_drivers is True
    assert registered[0].svd_driver == "gesvdj"
    assert registered[1] is policy


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


@pytest.mark.smoke
def test_peps_energy_optimizer_applies_max_parameter_step(monkeypatch):
    """The optional trust box is applied without an extra loss evaluation."""

    class FakeTNOptimizer:
        def __init__(self):
            self.vectorizer = SimpleNamespace(vector=np.array([0.5, -1.0]))
            self.handler = SimpleNamespace()
            self.losses = []
            self._bounds = None

        @property
        def bounds(self):
            return self._bounds

        @bounds.setter
        def bounds(self, value):
            self._bounds = np.array((value,) * self.vectorizer.vector.size)

        def optimize(self, n, **kwargs):
            _ = n, kwargs
            return self.get_tn_opt()

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

    out, losses = optimizer.optimize(
        n=1,
        optimizer="L-BFGS-B",
        optlib="nlopt",
        check_finite_gradient=False,
        max_parameter_step=0.25,
        progbar=False,
        return_losses=True,
    )

    assert np.array_equal(
        fake.bounds,
        np.array([[0.25, 0.75], [-1.25, -0.75]]),
    )
    assert np.array_equal(out, np.array([0.5, -1.0]))
    assert losses == ()


@pytest.mark.smoke
def test_peps_energy_optimizer_validation_rolls_back_false_descent(monkeypatch):
    """Validation rejects a lower local loss with a worse checked energy."""

    class FakeHandler:
        def __init__(self):
            self.calls = 0

        def value_and_grad(self, arrays):
            loss = (0.5, 0.4)[self.calls]
            self.calls += 1
            return loss, np.zeros_like(arrays)

    class FakeTNOptimizer:
        def __init__(self):
            self.vectorizer = SimpleNamespace(vector=np.array([0.0]))
            self.handler = FakeHandler()
            self.losses = []
            self.next_value = 1.0

        def optimize(self, n, **kwargs):
            _ = n, kwargs
            self.vectorizer.vector[:] = self.next_value
            loss, _ = self.handler.value_and_grad(self.vectorizer.vector)
            self.losses.append(loss)
            self.next_value += 1.0
            return self.get_tn_opt()

        def get_tn_opt(self):
            return self.vectorizer.vector.copy()

    optimizer = object.__new__(pepsy.PepsEnergyOptimizer)
    optimizer.loss_kwargs = {
        "chi": 2,
        "boundary_mode": "mps",
        "cutoff": 0.0,
        "normalized": True,
        "energy_per_site": True,
        "real": True,
        "stabilize_state": False,
        "contraction_opt": "auto",
        "compute_kwargs": {},
    }
    optimizer.terms = {}
    optimizer.state = object()
    fake = FakeTNOptimizer()
    converted_states = []
    optimizer_kwargs = []
    progress_bars = []

    class FakeProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n = 0
            self.postfixes = []
            self.closed = False
            progress_bars.append(self)

        def update(self, amount):
            self.n += amount

        def set_postfix(self, **values):
            self.postfixes.append(dict(values))

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        optimizer,
        "make_tn_optimizer",
        lambda **kwargs: optimizer_kwargs.append(kwargs) or fake,
    )
    monkeypatch.setattr("tqdm.auto.tqdm", FakeProgress)
    monkeypatch.setattr(
        optimizer,
        "_validation_loss",
        lambda state, *, terms, loss_kwargs: {
            0.0: 0.0,
            1.0: -1.0,
            2.0: 0.0,
        }[float(np.asarray(state)[0])],
    )
    monkeypatch.setattr(
        optimizer,
        "_state_for_autodiff_backend",
        lambda out, *args, **kwargs: converted_states.append(out) or out,
    )

    with pytest.warns(RuntimeWarning, match="validation energy worsened"):
        out, losses = optimizer.optimize(
            n=20,
            optimizer="LD_VAR2",
            optlib="nlopt",
            check_finite_gradient=False,
            validate=True,
            validation_interval=5,
            progbar=True,
            return_losses=True,
        )

    assert np.array_equal(out, np.array([1.0]))
    assert losses == (0.5, 0.4)
    assert optimizer.validation_history == [(0, 0.0), (5, -1.0), (10, 0.0)]
    # Validation must use the same backend-conversion hook as the final state,
    # including the initial check and every candidate check.
    assert len(converted_states) >= 4
    assert optimizer_kwargs[0]["progbar"] is False
    assert len(progress_bars) == 1
    assert progress_bars[0].n == 10
    assert progress_bars[0].closed
    assert progress_bars[0].postfixes[-1] == {
        "train_chi": 2,
        "check_chi": 4,
        "step": 5,
        "local_E": "+4.000e-01",
        "check_E": "+0.000e+00",
        "status": "rollback",
    }


@pytest.mark.smoke
def test_peps_energy_optimizer_validation_interval_auto_and_explicit():
    """Validation cadence follows ``n`` unless the user overrides it."""
    optimizer_cls = pepsy.PepsEnergyOptimizer

    assert optimizer_cls._validation_chunk_size(100, None) == 20
    assert optimizer_cls._validation_chunk_size(20, None) == 10
    assert optimizer_cls._validation_chunk_size(100, 7) == 7

    with pytest.raises(ValueError, match="validation_interval"):
        optimizer_cls._validation_chunk_size(100, 0)
    with pytest.raises(TypeError, match="validation_interval"):
        optimizer_cls._validation_chunk_size(100, 2.5)


@pytest.mark.smoke
def test_peps_energy_optimizer_validation_keeps_symmray_backend():
    """Higher-chi validation converts Quimb's NumPy candidate for Symmray."""
    pytest.importorskip("symmray")
    pytest.importorskip("torch")
    pytest.importorskip("nlopt")
    import torch

    to_backend = pepsy.backend_torch(dtype=torch.float64, device="cpu")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        to_backend=to_backend,
    )
    setup = fermion.lattice_half_filling(
        2,
        2,
        pattern="checkerboard",
        cyclic=True,
    )
    state = pepsy.ps_to_peps(
        (2, 2),
        fermion=fermion,
        occupations=setup.occupations,
        seed=7,
        dtype="float64",
        cyclic=False,
    )
    state.apply_to_arrays(to_backend)
    terms = {
        edge: -fermion.hopping_operator()
        for edge in setup.edges
    }
    terms.update({
        site: fermion.onsite_term(site, U=2.0, mu=0.0)
        for site in setup.sites
    })

    optimizer = pepsy.PepsEnergyOptimizer(
        state,
        terms,
        chi=2,
        boundary_mode="mps",
        contraction_opt="auto-hq",
    )
    out, _ = optimizer.optimize(
        n=1,
        optimizer="LD_VAR2",
        optlib="nlopt",
        progbar=False,
        check_finite_gradient=False,
        validate=True,
        return_losses=True,
    )

    assert isinstance(out, type(state))
    assert optimizer.validation_history
    assert np.isfinite(float(optimizer.validation_history[0][1]))
    assert all(
        str(block.dtype) in {"torch.float64", "float64"}
        for tensor in out.tensor_map.values()
        for block in tensor.data.blocks.values()
    )
