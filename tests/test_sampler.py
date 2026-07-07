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


def test_mps_sampler_native_numpy_samples_product_state():
    """Native sampler should batch samples without quimb's CPU sampler."""
    psi = qtn.MPS_computational_state("101")
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0), 2: (2, 0)},
        backend="native",
    )

    result = sampler.sample(4, seed=1)

    assert sampler.resolved_backend == "numpy"
    assert result.configs_1d == [[1, 0, 1]] * 4
    assert all(
        np.array_equal(grid, np.array([[1, 0, 1]]))
        for grid in result.configs_2d
    )
    assert result.probs == [1.0] * 4


def test_mps_sampler_native_numpy_reports_born_probabilities():
    """Native sampler should return the probability of each sampled config."""
    site_probs = [(0.8, 0.2), (0.3, 0.7)]
    psi = qtn.MPS_product_state([
        np.sqrt(np.array(probs))
        for probs in site_probs
    ])
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )

    result = sampler.sample(20, seed=4)

    for config, prob in zip(result.configs_1d, result.probs):
        expected = site_probs[0][config[0]] * site_probs[1][config[1]]
        assert prob == pytest.approx(expected)


def test_mps_sampler_native_numpy_evaluates_batched_configs():
    """Native sampler should evaluate many configs without a Python loop."""
    site_probs = [(0.8, 0.2), (0.3, 0.7)]
    psi = qtn.MPS_product_state([
        np.sqrt(np.array(probs))
        for probs in site_probs
    ])
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )
    configs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)

    probabilities = sampler.probabilities(configs)
    amplitudes = sampler.amplitudes(configs)

    expected_probs = np.array([0.24, 0.56, 0.06, 0.14])
    np.testing.assert_allclose(probabilities, expected_probs)
    np.testing.assert_allclose(amplitudes, np.sqrt(expected_probs))


def test_mps_sampler_native_torch_keeps_tensors_on_torch():
    """Torch MPS tensors should stay on Torch through native sampling."""
    torch = pytest.importorskip("torch")
    psi = qtn.MPS_computational_state("10")
    psi.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.float64))

    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )
    result = sampler.sample(3, seed=2)

    assert sampler.resolved_backend == "torch"
    assert all(isinstance(array, torch.Tensor) for array in sampler._native_arrays)
    assert result.configs_1d == [[1, 0]] * 3
    assert result.probs == [1.0] * 3


def test_mps_sampler_native_torch_evaluates_batched_configs_on_torch():
    """Torch probability/amplitude evaluation should stay on Torch."""
    torch = pytest.importorskip("torch")
    site_probs = [(0.8, 0.2), (0.3, 0.7)]
    psi = qtn.MPS_product_state([
        np.sqrt(np.array(probs))
        for probs in site_probs
    ])
    psi.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )
    configs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)

    probabilities = sampler.probabilities(configs, to_numpy=False)
    amplitudes = sampler.amplitudes(configs, to_numpy=False)

    expected_probs = torch.tensor([0.24, 0.56, 0.06, 0.14], dtype=torch.float64)
    assert isinstance(probabilities, torch.Tensor)
    assert isinstance(amplitudes, torch.Tensor)
    torch.testing.assert_close(probabilities, expected_probs)
    torch.testing.assert_close(amplitudes, torch.sqrt(expected_probs))


def test_mps_sampler_native_cupy_evaluates_batched_configs_on_cupy():
    """CuPy probability/amplitude evaluation should stay on CuPy."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    site_probs = [(0.8, 0.2), (0.3, 0.7)]
    psi = qtn.MPS_product_state([
        np.sqrt(np.array(probs))
        for probs in site_probs
    ])
    psi.apply_to_arrays(lambda array: cupy.asarray(array, dtype=cupy.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )
    configs = cupy.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=cupy.int64)

    probabilities = sampler.probabilities(configs, to_numpy=False)
    amplitudes = sampler.amplitudes(configs, to_numpy=False)

    expected_probs = cupy.asarray([0.24, 0.56, 0.06, 0.14], dtype=cupy.float64)
    assert isinstance(probabilities, cupy.ndarray)
    assert isinstance(amplitudes, cupy.ndarray)
    cupy.testing.assert_allclose(probabilities, expected_probs)
    cupy.testing.assert_allclose(amplitudes, cupy.sqrt(expected_probs))


def test_mps_sampler_rejects_explicit_native_backend_mismatch():
    """Explicit native backend requests should fail instead of copying devices."""
    psi = qtn.MPS_computational_state("0")

    with pytest.raises(ValueError, match="backend='torch' requested"):
        sampler_mod.MpsSampler(psi, {0: (0, 0)}, backend="torch")


def test_mps_sampler_batched_config_evaluation_validates_configs():
    """Batched probability/amplitude helpers should reject bad configs."""
    psi = qtn.MPS_computational_state("00")
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )

    with pytest.raises(ValueError, match=r"shape \(batch, L=2\)"):
        sampler.probabilities(np.array([[0, 0, 0]]))

    with pytest.raises(ValueError, match="invalid physical index"):
        sampler.amplitudes(np.array([[0, 2]]))


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
