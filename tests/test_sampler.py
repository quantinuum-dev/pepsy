"""Tests for PEPS BP importance sampling helpers."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.sampling import samplers as sampler_mod


class DummyTN:
    Lx = 2
    Ly = 2


class DummyFlatTN:
    def __init__(self, label):
        self.label = label

    def contract(self, *args, **kwargs):
        assert args == (all,)
        assert kwargs["strip_exponent"] is True
        return 2.5 + self.label, -3

    def contract_boundary(self, **kwargs):
        assert kwargs["max_bond"] == 7
        assert kwargs["mode"] == "mps"
        assert kwargs["final_contract_opts"]["strip_exponent"] is True
        assert kwargs["progbar"] is False
        return 3.5 + self.label, -2

    def contract_ctmrg(self, **kwargs):
        assert kwargs["max_bond"] == 7
        assert kwargs["final_contract_opts"]["strip_exponent"] is True
        assert kwargs["inplace"] is False
        assert kwargs["progbar"] is False
        return 4.5 + self.label, -1


def test_sampler_exact_method_collects_configs_and_scalars(monkeypatch):
    """Sampler should collect row-major configs, proposal weights, and amplitudes."""
    calls = []

    def fake_sample_d2bp(tn, **kwargs):
        calls.append(kwargs)
        idx = len(calls) - 1
        config = {
            "k0,0": idx % 2,
            "k0,1": 1,
            "k1,0": 0,
            "k1,1": 1,
        }
        return config, DummyFlatTN(idx), 0.0123 * (idx + 1)

    monkeypatch.setattr(sampler_mod, "sample_d2bp", fake_sample_d2bp)
    monkeypatch.setattr(sampler_mod, "build_optimizer", lambda **kwargs: "OPT")

    sampler = sampler_mod.PepsBpSampler(DummyTN())
    result = sampler.sample(samples=2, method="exact", seed=10)

    assert result.configs == [[0, 1, 0, 1], [1, 1, 0, 1]]
    assert result.omegas == ([1.23, 2.46], [-2, -2])
    assert result.ps == ([2.5, 3.5], [-3, -3])
    assert len(result) == 2
    assert [call["seed"] for call in calls] == [10, 11]
    assert all(call["update"] == "parallel" for call in calls)


def test_sampler_accepts_optimizer_object():
    """Constructor should accept an explicit optimizer object."""
    sampler = sampler_mod.PepsBpSampler(DummyTN(), optimizer="OPT")
    assert sampler.optimizer == "OPT"


def test_sampler_contraction_methods(monkeypatch):
    """MPS and CTMRG paths should forward contraction options correctly."""

    def fake_sample_d2bp(tn, **kwargs):  # pylint: disable=unused-argument
        config = {"k0,0": 0, "k0,1": 0, "k1,0": 1, "k1,1": 1}
        return config, DummyFlatTN(0), 1.0

    monkeypatch.setattr(sampler_mod, "sample_d2bp", fake_sample_d2bp)
    monkeypatch.setattr(sampler_mod, "build_optimizer", lambda **kwargs: "OPT")

    sampler = sampler_mod.PepsBpSampler(DummyTN())
    mps_result = sampler.sample(samples=1, chi=7, method="mps")
    ctmrg_result = sampler.sample(samples=1, chi=7, method="ctmrg")

    assert mps_result.ps == ([3.5], [-2])
    assert ctmrg_result.ps == ([4.5], [-1])


def test_sampler_rejects_unknown_contraction_method(monkeypatch):
    """Unknown contraction methods should fail loudly."""

    def fake_sample_d2bp(tn, **kwargs):  # pylint: disable=unused-argument
        config = {"k0,0": 0, "k0,1": 0, "k1,0": 0, "k1,1": 0}
        return config, DummyFlatTN(0), 1.0

    monkeypatch.setattr(sampler_mod, "sample_d2bp", fake_sample_d2bp)
    monkeypatch.setattr(sampler_mod, "build_optimizer", lambda **kwargs: "OPT")

    with pytest.raises(ValueError, match="Unknown contraction method"):
        sampler_mod.PepsBpSampler(DummyTN()).sample(method="bad")


def test_sampler_public_exports_resolve():
    """Sampler helpers should be available from the package namespace."""
    assert pepsy.PepsBpSampler is sampler_mod.PepsBpSampler
    assert pepsy.PEPSSampleResult is sampler_mod.PEPSSampleResult


def test_mps_sampler_rejects_incomplete_site_map():
    """MPS sampler should fail at construction for incomplete site maps."""
    psi = qtn.MPS_computational_state("00")
    with pytest.raises(ValueError, match="consecutive site indices"):
        sampler_mod.MpsSampler(psi, {0: (0, 0)})


def test_mps_sampler_rejects_bad_coordinates():
    """MPS sampler should require 2D integer coordinates."""
    psi = qtn.MPS_computational_state("0")
    with pytest.raises(TypeError, match="integer tuples"):
        sampler_mod.MpsSampler(psi, {0: (0.0, 0)})


def test_vec_sampler_rejects_invalid_vector_size():
    """Dense vector length must match the site-map Hilbert-space dimension."""
    with pytest.raises(ValueError, match=r"2\*\*L=4"):
        sampler_mod.VecSampler([1.0, 0.0, 0.0], {0: (0, 0), 1: (1, 0)})


def test_vec_sampler_rejects_zero_norm_state():
    """Zero vectors are not valid probability distributions."""
    with pytest.raises(ValueError, match="non-zero norm"):
        sampler_mod.VecSampler(np.zeros(4), {0: (0, 0), 1: (1, 0)})


def test_vec_sampler_samples_valid_dense_state():
    """Dense sampler should still produce 1D configs and 2D grids."""
    result = sampler_mod.VecSampler(
        np.array([0.0, 0.0, 1.0, 0.0]),
        {0: (0, 0), 1: (1, 0)},
    ).sample(2, seed=1)

    assert result.configs_1d == [[1, 0], [1, 0]]
    assert all(np.array_equal(grid, np.array([[1, 0]])) for grid in result.configs_2d)
    assert result.probs == [1.0, 1.0]
