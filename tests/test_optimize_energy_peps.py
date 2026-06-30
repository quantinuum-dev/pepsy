"""Tests for PEPS energy-objective optimization."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers import EnergyEstimate, PepsEnergyOptimizer
from pepsy.optimizers.energy.peps import PepsEnergyOptimizer as ModulePepsEnergyOptimizer


class _FakePeps:
    Lx = 2
    Ly = 3

    def __init__(self, value=12.0):
        self.value = value
        self.calls = []
        self.exact_calls = []
        self.balanced = False
        self.equalized_to = None

    def compute_local_expectation(self, terms, **kwargs):
        self.calls.append((terms, kwargs))
        return self.value

    def compute_local_expectation_exact(self, terms, **kwargs):
        self.exact_calls.append((terms, kwargs))
        return self.value

    def balance_bonds_(self):
        self.balanced = True
        return self

    def equalize_norms_(self, value):
        self.equalized_to = value
        return self


def _zz_terms():
    z_op = np.diag([1.0, -1.0])
    zz_op = np.kron(z_op, z_op).reshape(2, 2, 2, 2)
    return {((0, 0), (0, 1)): zz_op}


def test_peps_energy_loss_calls_local_expectation_with_expected_options():
    """loss() should route Pepsy options to quimb local expectation kwargs."""
    state = _FakePeps()
    terms = {"edge": object()}
    opt = PepsEnergyOptimizer(
        state,
        terms,
        chi=7,
        boundary_mode="ctmrg",
        cutoff=1.0e-8,
        normalized=True,
        contraction_opt="auto-hq",
        stabilize_state=True,
    )

    loss = opt.loss()

    assert loss == pytest.approx(2.0)
    assert state.balanced is True
    assert state.equalized_to == pytest.approx(1.0)
    called_terms, kwargs = state.calls[-1]
    assert called_terms is terms
    assert kwargs["max_bond"] == 7
    assert kwargs["cutoff"] == pytest.approx(1.0e-8)
    assert kwargs["normalized"] is True
    assert kwargs["mode"] == "projector"
    assert kwargs["contract_optimize"] == "auto-hq"


def test_peps_energy_stabilize_skips_balance_for_symmray_blocks():
    """Symmray PEPS stabilization should avoid fragile bond balancing."""

    class _SymmrayLikeData:
        blocks = {"q0": np.ones((1,))}

        def apply_to_arrays(self, fn):
            self.blocks = {
                sector: fn(block)
                for sector, block in self.blocks.items()
            }

    class _FakeTensor:
        data = _SymmrayLikeData()

    class _SymmrayPeps(_FakePeps):
        tensor_map = {"I0,0": _FakeTensor()}

        def balance_bonds_(self):
            raise AssertionError("Symmray-backed stabilization should not balance bonds.")

    state = _SymmrayPeps()
    opt = PepsEnergyOptimizer(
        state,
        {"edge": object()},
        stabilize_state=True,
    )

    assert opt.loss() == pytest.approx(2.0)
    assert state.equalized_to == pytest.approx(1.0)


def test_peps_energy_returns_full_and_per_site_estimate():
    """energy() should report both total energy and energy per site."""
    state = _FakePeps(value=9.0)
    opt = PepsEnergyOptimizer(state, {"edge": object()}, energy_per_site=True)

    estimate = opt.energy()

    assert isinstance(estimate, EnergyEstimate)
    assert estimate.energy == pytest.approx(9.0)
    assert estimate.energy_per_site == pytest.approx(1.5)
    assert estimate.num_sites == 6
    assert estimate.boundary_mode == "mps"
    assert estimate.as_dict()["energy"] == pytest.approx(9.0)


def test_peps_energy_exact_boundary_uses_exact_local_expectation():
    """Exact PEPS energy mode should avoid approximate boundary contraction."""
    state = _FakePeps(value=6.0)
    terms = {"edge": object()}
    opt = PepsEnergyOptimizer(
        state,
        terms,
        boundary_mode="exact",
        energy_per_site=False,
        contraction_opt="auto-hq",
    )

    assert opt.loss() == pytest.approx(6.0)
    assert state.calls == []
    called_terms, kwargs = state.exact_calls[-1]
    assert called_terms is terms
    assert kwargs["optimize"] == "auto-hq"
    assert kwargs["normalized"] is True


def test_peps_energy_accepts_local_ham_payload_mapping():
    """Hamiltonian payloads should resolve local_terms before measurement."""
    state = _FakePeps(value=4.0)
    terms = {"edge": object()}
    opt = PepsEnergyOptimizer(state, {"local_terms": terms}, energy_per_site=False)

    assert opt.loss() == pytest.approx(4.0)
    assert state.calls[-1][0] is terms


def test_peps_energy_loss_matches_direct_quimb_local_expectation():
    """The static energy loss should be a thin wrapper around quimb."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=23, dtype="complex128")
    terms = _zz_terms()
    direct = peps.compute_local_expectation(
        terms,
        max_bond=8,
        cutoff=0.0,
        normalized=True,
        mode="mps",
        contract_optimize="auto-hq",
    )
    opt = PepsEnergyOptimizer(
        peps,
        terms,
        chi=8,
        cutoff=0.0,
        normalized=True,
        energy_per_site=False,
        contraction_opt="auto-hq",
    )

    assert complex(opt.loss()) == pytest.approx(complex(direct))


def test_peps_energy_exact_loss_matches_direct_quimb_exact_expectation():
    """Exact mode should route to quimb's exact local expectation contraction."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=24, dtype="complex128")
    terms = _zz_terms()
    direct = peps.compute_local_expectation_exact(
        terms,
        optimize="auto-hq",
        normalized=True,
    )
    opt = PepsEnergyOptimizer(
        peps,
        terms,
        boundary_mode="exact",
        normalized=True,
        energy_per_site=False,
        contraction_opt="auto-hq",
    )

    assert complex(opt.loss()) == pytest.approx(complex(direct))


def test_peps_energy_make_tn_optimizer_and_optimize(monkeypatch):
    """TNOptimizer construction should receive terms as constants and update state."""
    calls = []
    out = _FakePeps(value=3.0)

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            calls.append((state, loss_fn, kwargs))
            self.losses = [2.0, 1.0]

        def optimize(self, n=220, **kwargs):
            calls.append(("optimize", n, kwargs))
            return out

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)
    state = _FakePeps(value=5.0)
    terms = {"edge": object()}
    opt = PepsEnergyOptimizer(state, terms)

    tnopt = opt.make_tn_optimizer(
        optimizer="lbfgs",
        autodiff_backend="jax",
        progbar=False,
        loss_kwargs={"chi": 5},
    )
    assert isinstance(tnopt, _FakeTNOptimizer)
    _, loss_fn, kwargs = calls[0]
    assert loss_fn is ModulePepsEnergyOptimizer._tnopt_loss
    assert kwargs["loss_constants"]["terms"] is terms
    assert kwargs["loss_kwargs"]["chi"] == 5
    assert kwargs["optimizer"] == "L-BFGS-B"
    assert kwargs["autodiff_backend"] == "jax"
    assert kwargs["progbar"] is False

    optimized, losses = opt.optimize(n=3, progbar=False, return_losses=True)
    assert optimized is out
    assert opt.state is out
    assert losses == (2.0, 1.0)
    assert calls[-1] == ("optimize", 3, {})


def test_peps_energy_make_tn_optimizer_prepares_autodiff_backend(monkeypatch):
    """Autodiff construction should register Pepsy's stable SVD hooks."""
    calls = []

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            _ = (state, loss_fn)
            calls.append(("tnopt", kwargs["autodiff_backend"]))

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)
    monkeypatch.setattr(
        "pepsy.optimizers.energy.peps.reg_rel_svd_torch",
        lambda: calls.append("torch-svd"),
    )
    monkeypatch.setattr(
        "pepsy.optimizers.energy.peps.reg_rel_svd_jax",
        lambda: calls.append("jax-svd"),
    )
    opt = PepsEnergyOptimizer(_FakePeps(value=5.0), {"edge": object()})

    opt.make_tn_optimizer(autodiff_backend="torch", progbar=False)
    opt.make_tn_optimizer(autodiff_backend="jax", progbar=False)

    assert calls == [
        "torch-svd",
        ("tnopt", "torch"),
        "jax-svd",
        ("tnopt", "jax"),
    ]


def test_peps_energy_optimize_falls_back_to_exact_gradient(monkeypatch):
    """A NaN MPS gradient should rebuild the objective with exact contraction."""
    out = _FakePeps(value=3.0)
    modes = []

    class _FakeVectorizer:
        def __init__(self):
            self.vector = np.array([1.0])

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            _ = (state, loss_fn, kwargs)
            self.mode = kwargs["loss_kwargs"]["boundary_mode"]
            modes.append(self.mode)
            self.vectorizer = _FakeVectorizer()
            self.losses = []
            self.loss = 123.0
            self.loss_best = 123.0
            self._n = 7

        def vectorized_value_and_grad(self, vector):
            assert np.array_equal(vector, np.array([1.0]))
            if self.mode == "mps":
                self.losses.append(float("nan"))
                return 1.0, np.array([np.nan])
            return 0.5, np.array([0.0])

        def optimize(self, n=220, **kwargs):
            assert self.mode == "exact"
            assert n == 3
            assert kwargs == {}
            self.losses = [0.5, 0.25]
            return out

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)
    state = _FakePeps(value=5.0)
    opt = PepsEnergyOptimizer(state, {"edge": object()})

    optimized, losses = opt.optimize(n=3, progbar=False, return_losses=True)

    assert optimized is out
    assert opt.state is out
    assert losses == (0.5, 0.25)
    assert modes == ["mps", "exact"]


def test_peps_energy_optimize_returns_state_if_fallback_gradient_nonfinite(monkeypatch):
    """If all autodiff gradients are bad, avoid raising or poisoning state."""
    modes = []

    class _FakeVectorizer:
        def __init__(self):
            self.vector = np.array([1.0])

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            _ = (state, loss_fn)
            self.mode = kwargs["loss_kwargs"]["boundary_mode"]
            modes.append(self.mode)
            self.vectorizer = _FakeVectorizer()
            self.losses = []

        def vectorized_value_and_grad(self, vector):
            _ = vector
            return 1.0, np.array([np.nan])

        def optimize(self, n=220, **kwargs):
            _ = (n, kwargs)
            raise AssertionError("optimize should not run with a NaN gradient.")

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)
    state = _FakePeps(value=5.0)
    opt = PepsEnergyOptimizer(state, {"edge": object()})

    with pytest.warns(RuntimeWarning, match="non-finite initial gradient"):
        optimized, losses = opt.optimize(n=3, progbar=False, return_losses=True)

    assert optimized is state
    assert opt.state is state
    assert losses == (1.0,)
    assert modes == ["mps", "exact"]


def test_peps_energy_make_tn_optimizer_converts_terms_to_autodiff_backend(monkeypatch):
    """Autodiff loss constants should use the same backend dtype as the PEPS."""
    torch = pytest.importorskip("torch")
    calls = []

    class _FakeTensor:
        def __init__(self):
            self.data = np.ones((1,), dtype=np.complex128)

    class _FakeBlockTerm:
        def __init__(self, blocks):
            self.blocks = blocks

        def copy(self):
            return type(self)({
                sector: block.copy()
                for sector, block in self.blocks.items()
            })

        def apply_to_arrays(self, fn):
            self.blocks = {
                sector: fn(block)
                for sector, block in self.blocks.items()
            }
            return self

    class _FakePepsWithBlocks(_FakePeps):
        tensor_map = {"I0,0": _FakeTensor()}

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            _ = (state, loss_fn)
            calls.append(kwargs)

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)

    term = _FakeBlockTerm({"q0": np.ones((1,), dtype=np.float64)})
    opt = PepsEnergyOptimizer(
        _FakePepsWithBlocks(value=5.0),
        {"edge": term},
    )

    opt.make_tn_optimizer(autodiff_backend="torch", progbar=False)

    converted = calls[-1]["loss_constants"]["terms"]["edge"].blocks["q0"]
    assert isinstance(converted, torch.Tensor)
    assert converted.dtype == torch.complex128
    assert isinstance(term.blocks["q0"], np.ndarray)
    assert term.blocks["q0"].dtype == np.float64


def test_peps_energy_normalize_translates_projector_mode(monkeypatch):
    """normalize() should translate local projector mode to global ctmrg mode."""
    calls = []
    state = _FakePeps(value=5.0)

    def _fake_normalize(state_arg, **kwargs):
        calls.append((state_arg, kwargs))
        return state_arg

    monkeypatch.setattr(
        "pepsy.optimizers.energy.peps.GlobalOptimizer._normalize_state",
        _fake_normalize,
    )
    opt = PepsEnergyOptimizer(state, {"edge": object()}, boundary_mode="projector")

    assert opt.normalize() is state
    assert calls[-1][0] is state
    assert calls[-1][1]["mode"] == "ctmrg"


def test_peps_energy_optimizer_public_exports():
    """New optimizer should resolve from package public namespaces."""
    assert pepsy.PepsEnergyOptimizer is PepsEnergyOptimizer
    assert pepsy.optimizers.PepsEnergyOptimizer is PepsEnergyOptimizer
