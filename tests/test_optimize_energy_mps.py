"""Tests for MPS energy-objective optimization."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers import EnergyEstimate, MpsEnergyOptimizer
from pepsy.optimizers.energy.peps import MpsEnergyOptimizer as ModuleMpsEnergyOptimizer


class _FakeTensor:
    def __init__(self, data):
        self.data = data


class _FakeMps:
    L = 4

    def __init__(self, value=8.0, *, data=None):
        self.value = value
        self.calls = []
        self.normalized_kwargs = None
        if data is not None:
            self.tensor_map = {"I0": _FakeTensor(data)}

    def compute_local_expectation_exact(self, terms, **kwargs):
        self.calls.append((terms, kwargs))
        return self.value

    def max_bond(self):
        return 3

    def normalize(self, **kwargs):
        self.normalized_kwargs = kwargs
        return 1.0


class _MpsWrapper:
    def __init__(self, mps):
        self.mps = mps


def test_mps_energy_loss_calls_exact_local_expectation_with_expected_options():
    """loss() should route MPS options to quimb exact local expectation kwargs."""
    state = _FakeMps()
    terms = {"edge": object()}
    opt = MpsEnergyOptimizer(
        state,
        terms,
        normalized=True,
        contraction_opt="auto-hq",
        progbar=True,
    )

    loss = opt.loss()

    assert loss == pytest.approx(2.0)
    called_terms, kwargs = state.calls[-1]
    assert called_terms is terms
    assert kwargs["optimize"] == "auto-hq"
    assert kwargs["normalized"] is True
    assert kwargs["progbar"] is True


def test_mps_energy_compute_kwargs_override_direct_progbar():
    """compute_kwargs should still be able to override the convenience flag."""
    state = _FakeMps()
    opt = MpsEnergyOptimizer(
        state,
        {"edge": object()},
        progbar=True,
        compute_kwargs={"progbar": False, "foo": "bar"},
    )

    opt.loss()

    _, kwargs = state.calls[-1]
    assert kwargs["progbar"] is False
    assert kwargs["foo"] == "bar"


def test_mps_energy_returns_full_and_per_site_estimate():
    """energy() should report both total energy and energy per site."""
    state = _FakeMps(value=8.0)
    opt = MpsEnergyOptimizer(state, {"edge": object()}, energy_per_site=True)

    estimate = opt.energy()

    assert isinstance(estimate, EnergyEstimate)
    assert estimate.energy == pytest.approx(8.0)
    assert estimate.energy_per_site == pytest.approx(2.0)
    assert estimate.num_sites == 4
    assert estimate.chi == 3
    assert estimate.boundary_mode == "exact"
    assert estimate.as_dict()["energy"] == pytest.approx(8.0)


def test_mps_energy_accepts_mpo_hamiltonian():
    """MPO Hamiltonians should contract as <psi|H|psi>."""
    mps = qtn.MPS_computational_state("01")
    z_op = np.diag([1.0, -1.0])
    mpo = qtn.MPO_product_operator(
        [z_op, z_op],
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    opt = MpsEnergyOptimizer(
        mps,
        mpo,
        energy_per_site=False,
        real=True,
    )

    estimate = opt.energy()

    assert estimate.energy == pytest.approx(-1.0)
    assert estimate.energy_per_site == pytest.approx(-0.5)
    assert estimate.boundary_mode == "exact"


def test_mps_energy_mpo_hamiltonian_uses_direct_contract(monkeypatch):
    """MPO Hamiltonians should bypass term-wise local expectations."""
    mps = qtn.MPS_computational_state("01")

    def fail_local_terms(*_args, **_kwargs):
        raise AssertionError("MPO energy must not use local-term expectation.")

    monkeypatch.setattr(mps, "compute_local_expectation_exact", fail_local_terms)
    z_op = np.diag([1.0, -1.0])
    mpo = qtn.MPO_product_operator(
        [z_op, z_op],
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    optimize_seen = []
    original_contract = qtn.TensorNetwork.contract

    def contract_spy(self, *args, **kwargs):
        optimize_seen.append(kwargs.get("optimize"))
        return original_contract(self, *args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, "contract", contract_spy)
    opt = MpsEnergyOptimizer(
        mps,
        mpo,
        normalized=False,
        energy_per_site=False,
        real=True,
        contraction_opt="greedy",
    )

    estimate = opt.energy()

    assert estimate.energy == pytest.approx(-1.0)
    assert optimize_seen[-1] == "greedy"


def test_mps_energy_accepts_wrapper_and_local_ham_payload_mapping():
    """SymMPS-like wrappers and local_terms payloads should resolve."""
    state = _FakeMps(value=4.0)
    terms = {"edge": object()}
    opt = MpsEnergyOptimizer(_MpsWrapper(state), {"local_terms": terms})

    assert opt.state is state
    assert opt.loss() == pytest.approx(1.0)
    assert state.calls[-1][0] is terms


def test_mps_energy_normalize_uses_mps_native_normalization():
    """MPS normalization should not route through PEPS boundary normalization."""
    state = _FakeMps()
    opt = MpsEnergyOptimizer(state, {"edge": object()})

    assert opt.normalize(eps=1.0e-12) is state
    assert state.normalized_kwargs == {"eps": 1.0e-12}


def test_mps_energy_loss_converts_terms_to_state_backend():
    """Exact MPS contraction terms should match the state backend and dtype."""
    torch = pytest.importorskip("torch")
    sample = torch.ones((1,), dtype=torch.complex128)
    state = _FakeMps(value=torch.tensor(6.0, dtype=torch.float64), data=sample)
    term = np.eye(4, dtype=np.float64).reshape(2, 2, 2, 2)
    opt = MpsEnergyOptimizer(
        state,
        {"edge": term},
        energy_per_site=False,
    )

    loss = opt.loss()

    called_terms, _ = state.calls[-1]
    converted = called_terms["edge"]
    assert float(loss) == pytest.approx(6.0)
    assert isinstance(converted, torch.Tensor)
    assert converted.dtype == torch.complex128
    assert isinstance(term, np.ndarray)
    assert term.dtype == np.float64


def test_mps_energy_make_tn_optimizer_and_optimize(monkeypatch):
    """TNOptimizer construction should receive MPS terms as constants."""
    calls = []
    out = _FakeMps(value=3.0)

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            calls.append((state, loss_fn, kwargs))
            self.losses = [2.0, 1.0]

        def optimize(self, n=220, **kwargs):
            calls.append(("optimize", n, kwargs))
            return out

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)
    state = _FakeMps(value=5.0)
    terms = {"edge": object()}
    opt = MpsEnergyOptimizer(state, terms)

    tnopt = opt.make_tn_optimizer(
        optimizer="lbfgs",
        autodiff_backend="jax",
        progbar=False,
        loss_kwargs={"progbar": False},
    )
    assert isinstance(tnopt, _FakeTNOptimizer)
    _, loss_fn, kwargs = calls[0]
    assert loss_fn is ModuleMpsEnergyOptimizer._tnopt_loss
    assert kwargs["loss_constants"]["terms"] is terms
    assert kwargs["loss_kwargs"]["progbar"] is False
    assert kwargs["optimizer"] == "L-BFGS-B"
    assert kwargs["autodiff_backend"] == "jax"
    assert kwargs["progbar"] is False

    optimized, losses = opt.optimize(n=3, progbar=False, return_losses=True)
    assert optimized is out
    assert opt.state is out
    assert losses == (2.0, 1.0)
    assert calls[-1] == ("optimize", 3, {})


def test_mps_energy_optimize_recasts_tnoptimizer_output_to_autodiff_backend(monkeypatch):
    """quimb returns optimized variables as NumPy; Pepsy should restore backend."""
    torch = pytest.importorskip("torch")
    calls = []

    state = qtn.MPS_computational_state("00")
    state.apply_to_arrays(pepsy.backend_torch(dtype=torch.float64, device="cpu"))

    out = qtn.MPS_computational_state("11")
    out.apply_to_arrays(lambda x: np.asarray(x, dtype=np.float64))

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            _ = (state, loss_fn)
            calls.append(kwargs)
            self.losses = [1.0]

        def optimize(self, n=220, **kwargs):
            _ = (n, kwargs)
            return out

    monkeypatch.setattr("pepsy.optimizers.energy.peps.qtn.TNOptimizer", _FakeTNOptimizer)
    terms = {(0, 1): np.eye(4, dtype=np.float64).reshape(2, 2, 2, 2)}
    opt = MpsEnergyOptimizer(state, terms)

    optimized, losses = opt.optimize(
        n=1,
        autodiff_backend="torch",
        device="cpu",
        progbar=False,
        normalize=True,
        return_losses=True,
    )

    block = next(iter(optimized.tensor_map.values())).data
    assert isinstance(block, torch.Tensor)
    assert block.dtype == torch.float64
    assert block.device.type == "cpu"
    assert losses == (1.0,)
    assert calls[-1]["autodiff_backend"] == "torch"


def test_mps_energy_optimizer_public_exports():
    """MPS energy optimizer should resolve from package public namespaces."""
    assert pepsy.MpsEnergyOptimizer is MpsEnergyOptimizer
    assert pepsy.optimizers.MpsEnergyOptimizer is MpsEnergyOptimizer
