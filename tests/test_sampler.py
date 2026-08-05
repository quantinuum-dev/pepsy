"""Tests for PEPS BP importance sampling helpers."""

from itertools import product

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


def _dense_mps_from_symmray(psi):
    """Convert an ungraded Symmray MPS to an ordinary dense quimb MPS."""
    arrays = [
        sampler_mod.MpsSampler._site_array_lr_phys_r(psi, site).to_dense()
        for site in range(psi.L)
    ]
    return qtn.MatrixProductState(
        [
            arrays[0][0].T,
            *(array.transpose(0, 2, 1) for array in arrays[1:-1]),
            arrays[-1][:, :, 0],
        ],
        shape="lrp",
    )


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
    assert pepsy.FermionConfigurationEncoding is sampler_mod.FermionConfigurationEncoding
    assert pepsy.MpsDiagonalEstimate is sampler_mod.MpsDiagonalEstimate
    assert pepsy.MpsBatchSampleResult is sampler_mod.MpsBatchSampleResult


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


def test_mps_sampler_defaults_to_trivial_1d_site_map():
    """Omitting one_d_to_two_d should infer a trivial single-row 1D layout."""
    psi = qtn.MPS_computational_state("101")
    sampler = sampler_mod.MpsSampler(psi, backend="native")

    assert sampler.one_d_to_two_d == {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    assert (sampler.Lx, sampler.Ly) == (3, 1)

    result = sampler.sample(4, seed=1)
    assert sampler.resolved_backend == "numpy"
    assert result.configs_1d == [[1, 0, 1]] * 4


def test_mps_sampler_native_numpy_sample_arrays_returns_arrays():
    """Native raw sampling should return batched arrays directly."""
    psi = qtn.MPS_computational_state("101")
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0), 2: (2, 0)},
        backend="native",
    )

    configs, probs = sampler.sample_arrays(4, seed=1)

    assert isinstance(configs, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert configs.shape == (4, 3)
    assert probs.shape == (4,)
    np.testing.assert_array_equal(configs, np.array([[1, 0, 1]] * 4))
    np.testing.assert_allclose(probs, np.ones(4))


def test_mps_sampler_native_numpy_sample_batch_result_helpers():
    """Named batch samples should convert cleanly to legacy forms."""
    psi = qtn.MPS_computational_state("101")
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0), 2: (2, 0)},
        backend="native",
    )

    batch = sampler.sample_batch(4, seed=1)
    numpy_batch = batch.to_numpy()
    legacy = batch.to_sample_result()

    assert isinstance(batch, sampler_mod.MpsBatchSampleResult)
    assert len(batch) == 4
    assert batch.n_samples == 4
    assert batch.L == 3
    assert batch.backend == "numpy"
    assert numpy_batch.backend == "numpy"
    np.testing.assert_array_equal(batch.configs, np.array([[1, 0, 1]] * 4))
    np.testing.assert_allclose(batch.probs, np.ones(4))
    assert batch.configs_1d() == [[1, 0, 1]] * 4
    assert all(
        np.array_equal(grid, np.array([[1, 0, 1]]))
        for grid in batch.configs_2d()
    )
    np.testing.assert_allclose(batch.magnetizations(), [-1 / 3] * 4)
    assert legacy.configs_1d == [[1, 0, 1]] * 4
    assert legacy.probs == [1.0] * 4


def test_mps_sampler_native_site_ops_are_reused(monkeypatch):
    """Native branch-matrix setup should be cached across calls."""
    psi = qtn.MPS_computational_state("101")
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0), 2: (2, 0)},
        backend="native",
    )
    real_prepare = sampler_mod.MpsSampler._prepare_site_ops
    calls = []

    def spy_prepare(backend, arrays):
        calls.append(backend)
        return real_prepare(backend, arrays)

    monkeypatch.setattr(
        sampler_mod.MpsSampler,
        "_prepare_site_ops",
        staticmethod(spy_prepare),
    )

    sampler.sample_arrays(2, seed=1)
    sampler.probabilities(np.array([[1, 0, 1], [1, 0, 1]]))
    sampler.amplitudes(np.array([[1, 0, 1], [1, 0, 1]]))

    assert calls == ["numpy"]


def test_mps_sampler_refresh_rebuilds_cached_native_state():
    """Refreshing should pick up replacement MPS tensors after caching."""
    psi = qtn.MPS_computational_state("00")
    sampler = sampler_mod.MpsSampler(psi, backend="native")

    configs_before, _ = sampler.sample_arrays(3, seed=1)
    np.testing.assert_array_equal(configs_before, np.zeros((3, 2), dtype=int))
    assert sampler._native_site_ops is not None

    for site in range(psi.L):
        tensor = psi[site]
        physical_axis = tensor.inds.index(psi.site_ind(site))
        tensor.modify(data=np.flip(tensor.data, axis=physical_axis).copy())

    assert sampler.refresh() is sampler
    assert sampler._native_site_ops is None
    configs_after, _ = sampler.sample_arrays(3, seed=1)
    np.testing.assert_array_equal(configs_after, np.ones((3, 2), dtype=int))


def test_mps_sampler_refresh_rejects_a_different_mps_length():
    """Refresh keeps the fixed site-map length explicit."""
    psi = qtn.MPS_computational_state("00")
    sampler = sampler_mod.MpsSampler(psi, backend="native")

    with pytest.raises(ValueError, match="site map has length 2"):
        sampler.refresh(qtn.MPS_computational_state("000"))

    assert sampler._source_psi is psi


def test_mps_sampler_refresh_rebuilds_the_quimb_snapshot():
    """Refresh should also replace the copied historical Quimb state."""
    psi = qtn.MPS_computational_state("00")
    sampler = sampler_mod.MpsSampler(psi)
    configs_before, _ = sampler.sample_arrays(2, seed=1)
    np.testing.assert_array_equal(configs_before, np.zeros((2, 2), dtype=int))

    for site in range(psi.L):
        tensor = psi[site]
        physical_axis = tensor.inds.index(psi.site_ind(site))
        tensor.modify(data=np.flip(tensor.data, axis=physical_axis).copy())

    sampler.refresh()
    configs_after, _ = sampler.sample_arrays(2, seed=1)
    np.testing.assert_array_equal(configs_after, np.ones((2, 2), dtype=int))


def test_mps_sampler_numpy_qubit_path_avoids_generic_cdf(monkeypatch):
    """Binary NumPy sampling uses the direct Bernoulli draw path."""
    psi = qtn.MPS_product_state([
        np.sqrt(np.array([0.8, 0.2])),
        np.sqrt(np.array([0.3, 0.7])),
    ])
    sampler = sampler_mod.MpsSampler(psi, backend="native")

    def fail_cumsum(*args, **kwargs):
        raise AssertionError("binary NumPy sampling should not build a CDF")

    monkeypatch.setattr(sampler_mod.np, "cumsum", fail_cumsum)
    configs, probs = sampler.sample_arrays(4, seed=3)

    assert configs.shape == (4, 2)
    assert probs.shape == (4,)


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


def test_mps_sampler_native_numpy_matches_dense_random_mps():
    """Native NumPy evaluation should match exact dense Born probabilities."""
    L = 6
    psi = qtn.MPS_rand_state(L, bond_dim=4, phys_dim=2, dtype=float, seed=123)
    sampler = sampler_mod.MpsSampler(
        psi,
        {site: (site, 0) for site in range(L)},
        backend="native",
    )
    configs = np.array([
        [int(bit) for bit in format(index, f"0{L}b")]
        for index in range(2**L)
    ], dtype=np.int64)

    probabilities = sampler.probabilities(configs)
    amplitudes = sampler.amplitudes(configs)
    dense = psi.to_dense().reshape(-1)
    expected = np.abs(dense) ** 2
    expected = expected / expected.sum()

    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    np.testing.assert_allclose(np.abs(amplitudes) ** 2, expected, atol=1e-12)
    assert probabilities.sum() == pytest.approx(1.0)


def test_mps_sampler_native_numpy_handles_noncanonical_gauge():
    """Native conditionals should not assume right-canonical MPS tensors."""
    L = 5
    psi = qtn.MPS_rand_state(L, bond_dim=4, phys_dim=2, dtype=float, seed=123)
    bond = psi.bond(1, 2)
    gauge = np.linspace(0.5, 1.7, psi.ind_size(bond))

    for site, weights in ((1, gauge), (2, 1.0 / gauge)):
        tensor = psi[site]
        axis = tensor.inds.index(bond)
        shape = [1] * tensor.data.ndim
        shape[axis] = len(weights)
        tensor.modify(data=tensor.data * weights.reshape(shape))

    sampler = sampler_mod.MpsSampler(
        psi,
        {site: (site, 0) for site in range(L)},
        backend="native",
    )
    configs = np.array([
        [int(bit) for bit in format(index, f"0{L}b")]
        for index in range(2**L)
    ], dtype=np.int64)

    probabilities = sampler.probabilities(configs)
    amplitudes = sampler.amplitudes(configs)
    dense = psi.to_dense().reshape(-1)
    expected = np.abs(dense) ** 2
    expected = expected / expected.sum()

    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    np.testing.assert_allclose(np.abs(amplitudes) ** 2, expected, atol=1e-12)


def test_mps_sampler_native_numpy_agrees_with_quimb_sample_probabilities():
    """Native probabilities should agree with quimb for quimb-sampled configs."""
    L = 6
    psi = qtn.MPS_rand_state(L, bond_dim=4, phys_dim=2, dtype=float, seed=123)
    mapping = {site: (site, 0) for site in range(L)}
    quimb_sampler = sampler_mod.MpsSampler(psi, mapping, backend="quimb")
    native_sampler = sampler_mod.MpsSampler(psi, mapping, backend="native")

    configs, quimb_probs = quimb_sampler.sample_arrays(16, seed=7)
    native_probs = native_sampler.probabilities(configs)

    np.testing.assert_allclose(native_probs, quimb_probs, atol=1e-12)


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


def test_mps_sampler_native_torch_sample_arrays_stays_on_torch():
    """Torch raw sampling should return a batched tensor result."""
    torch = pytest.importorskip("torch")
    psi = qtn.MPS_computational_state("10")
    psi.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )

    configs, probs = sampler.sample_arrays(3, seed=2)

    assert isinstance(configs, torch.Tensor)
    assert isinstance(probs, torch.Tensor)
    assert configs.device == probs.device
    assert configs.shape == (3, 2)
    assert probs.shape == (3,)
    torch.testing.assert_close(
        configs,
        torch.tensor([[1, 0]] * 3, dtype=torch.long, device=configs.device),
    )
    torch.testing.assert_close(probs, torch.ones(3, dtype=probs.dtype))


def test_mps_sampler_torch_sampling_defaults_to_inference_mode():
    """Sampling should not retain a graph unless the caller opts in."""
    torch = pytest.importorskip("torch")
    psi = qtn.MPS_product_state([
        np.sqrt(np.array([0.8, 0.2])),
        np.sqrt(np.array([0.3, 0.7])),
    ])
    psi.apply_to_arrays(
        lambda array: torch.tensor(
            array,
            dtype=torch.float64,
            requires_grad=True,
        )
    )
    sampler = sampler_mod.MpsSampler(psi, backend="native")

    _, inference_probs = sampler.sample_arrays(4, seed=3)
    _, tracked_probs = sampler.sample_arrays(4, seed=3, track_grad=True)

    assert not inference_probs.requires_grad
    assert tracked_probs.requires_grad
    tracked_probs.sum().backward()
    assert any(psi[site].data.grad is not None for site in range(psi.L))


def test_mps_sampler_torch_compile_is_cached_when_available(monkeypatch):
    """The optional compiled inference path should reuse its compiled runner."""
    torch = pytest.importorskip("torch")
    psi = qtn.MPS_computational_state("10")
    psi.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        backend="native",
        torch_compile=True,
    )
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(
        sampler_mod.MpsSampler,
        "_torch_compile_supported",
        staticmethod(lambda _torch: True),
    )
    monkeypatch.setattr(torch, "compile", fake_compile)
    configs_1, probs_1 = sampler.sample_arrays(3)
    configs_2, probs_2 = sampler.sample_arrays(3)

    assert len(calls) == 1
    torch.testing.assert_close(configs_1, configs_2)
    torch.testing.assert_close(probs_1, probs_2)


def test_mps_sampler_torch_compile_falls_back_to_eager(monkeypatch):
    """Compiler failures must not prevent native Torch sampling."""
    torch = pytest.importorskip("torch")
    psi = qtn.MPS_computational_state("10")
    psi.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        backend="native",
        torch_compile=True,
    )

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(
        sampler_mod.MpsSampler,
        "_torch_compile_supported",
        staticmethod(lambda _torch: True),
    )
    monkeypatch.setattr(torch, "compile", fail_compile)
    configs, probs = sampler.sample_arrays(3)

    assert sampler._torch_compile_disabled
    torch.testing.assert_close(configs, torch.tensor([[1, 0]] * 3))
    torch.testing.assert_close(probs, torch.ones(3, dtype=probs.dtype))


def test_mps_sampler_native_torch_sample_batch_stays_on_torch():
    """Named Torch batch samples should keep tensors and convert on demand."""
    torch = pytest.importorskip("torch")
    psi = qtn.MPS_computational_state("10")
    psi.apply_to_arrays(lambda array: torch.as_tensor(array, dtype=torch.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )

    batch = sampler.sample_batch(3, seed=2)
    numpy_batch = batch.to_numpy()
    legacy = batch.to_sample_result()

    assert batch.backend == "torch"
    assert isinstance(batch.configs, torch.Tensor)
    assert isinstance(batch.probs, torch.Tensor)
    torch.testing.assert_close(
        batch.configs,
        torch.tensor([[1, 0]] * 3, dtype=torch.long, device=batch.configs.device),
    )
    torch.testing.assert_close(batch.probs, torch.ones(3, dtype=batch.probs.dtype))
    torch.testing.assert_close(
        batch.magnetizations(),
        torch.zeros(3, dtype=batch.probs.dtype, device=batch.probs.device),
    )
    assert isinstance(numpy_batch.configs, np.ndarray)
    assert numpy_batch.backend == "numpy"
    assert legacy.configs_1d == [[1, 0]] * 3


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


def test_mps_sampler_native_torch_matches_numpy_random_mps():
    """Torch native evaluation should agree with the NumPy dense reference."""
    torch = pytest.importorskip("torch")
    L = 6
    psi_np = qtn.MPS_rand_state(L, bond_dim=4, phys_dim=2, dtype=float, seed=123)
    psi_torch = psi_np.copy()
    psi_torch.apply_to_arrays(
        lambda array: torch.as_tensor(array, dtype=torch.float64)
    )
    sampler = sampler_mod.MpsSampler(
        psi_torch,
        {site: (site, 0) for site in range(L)},
        backend="native",
    )
    configs_np = np.array([
        [int(bit) for bit in format(index, f"0{L}b")]
        for index in range(2**L)
    ], dtype=np.int64)
    configs = torch.as_tensor(configs_np, dtype=torch.long)

    probabilities = sampler.probabilities(configs, to_numpy=False)
    amplitudes = sampler.amplitudes(configs, to_numpy=False)
    dense = psi_np.to_dense().reshape(-1)
    expected = torch.as_tensor(
        np.abs(dense) ** 2 / np.sum(np.abs(dense) ** 2),
        dtype=torch.float64,
    )
    batch = sampler.sample_batch(8, seed=5)
    batch_probabilities = sampler.probabilities(batch.configs, to_numpy=False)

    torch.testing.assert_close(probabilities, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        amplitudes.abs().square(),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        batch.probs,
        batch_probabilities,
        atol=1e-12,
        rtol=1e-12,
    )


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


def test_mps_sampler_native_cupy_sample_arrays_stays_on_cupy():
    """CuPy raw sampling should return batched arrays on CuPy."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    psi = qtn.MPS_computational_state("10")
    psi.apply_to_arrays(lambda array: cupy.asarray(array, dtype=cupy.float64))
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
        backend="native",
    )

    configs, probs = sampler.sample_arrays(3, seed=2)

    assert isinstance(configs, cupy.ndarray)
    assert isinstance(probs, cupy.ndarray)
    assert configs.shape == (3, 2)
    assert probs.shape == (3,)
    cupy.testing.assert_array_equal(
        configs,
        cupy.asarray([[1, 0]] * 3, dtype=cupy.int64),
    )
    cupy.testing.assert_allclose(probs, cupy.ones(3, dtype=probs.dtype))

    batch = sampler.sample_batch(3, seed=2)
    assert batch.backend == "cupy"
    assert isinstance(batch.configs, cupy.ndarray)
    assert isinstance(batch.probs, cupy.ndarray)
    cupy.testing.assert_array_equal(
        batch.configs,
        cupy.asarray([[1, 0]] * 3, dtype=cupy.int64),
    )
    assert isinstance(batch.to_numpy().configs, np.ndarray)


def test_mps_sampler_native_cupy_matches_numpy_random_mps():
    """CuPy native evaluation should agree with the NumPy dense reference."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    L = 6
    psi_np = qtn.MPS_rand_state(L, bond_dim=4, phys_dim=2, dtype=float, seed=123)
    psi_cupy = psi_np.copy()
    psi_cupy.apply_to_arrays(lambda array: cupy.asarray(array, dtype=cupy.float64))
    sampler = sampler_mod.MpsSampler(
        psi_cupy,
        {site: (site, 0) for site in range(L)},
        backend="native",
    )
    configs_np = np.array([
        [int(bit) for bit in format(index, f"0{L}b")]
        for index in range(2**L)
    ], dtype=np.int64)
    configs = cupy.asarray(configs_np, dtype=cupy.int64)

    probabilities = sampler.probabilities(configs, to_numpy=False)
    amplitudes = sampler.amplitudes(configs, to_numpy=False)
    dense = psi_np.to_dense().reshape(-1)
    expected = cupy.asarray(
        np.abs(dense) ** 2 / np.sum(np.abs(dense) ** 2),
        dtype=cupy.float64,
    )
    batch = sampler.sample_batch(8, seed=5)
    batch_probabilities = sampler.probabilities(batch.configs, to_numpy=False)

    cupy.testing.assert_allclose(probabilities, expected, atol=1e-12)
    cupy.testing.assert_allclose(cupy.abs(amplitudes) ** 2, expected, atol=1e-12)
    cupy.testing.assert_allclose(batch.probs, batch_probabilities, atol=1e-12)


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


def test_mps_sampler_sample_arrays_validates_sample_count():
    """Raw batched sampling should reject non-positive sample counts."""
    psi = qtn.MPS_computational_state("0")
    sampler = sampler_mod.MpsSampler(psi, {0: (0, 0)}, backend="native")

    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample_arrays(0)


def test_mps_sampler_sample_batch_validates_sample_count():
    """Named batched sampling should reject non-positive sample counts."""
    psi = qtn.MPS_computational_state("0")
    sampler = sampler_mod.MpsSampler(psi, {0: (0, 0)}, backend="native")

    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample_batch(0)


def test_mps_sampler_quimb_sample_batch_returns_numpy_result():
    """Default quimb backend should still support the named batch API."""
    psi = qtn.MPS_computational_state("10")
    sampler = sampler_mod.MpsSampler(
        psi,
        {0: (0, 0), 1: (1, 0)},
    )

    batch = sampler.sample_batch(3, seed=2)

    assert batch.backend == "numpy"
    assert isinstance(batch.configs, np.ndarray)
    assert isinstance(batch.probs, np.ndarray)
    assert batch.configs_1d() == [[1, 0]] * 3
    assert batch.to_sample_result().configs_1d == [[1, 0]] * 3


def test_mps_sampler_detects_symmray_u1u1_product_without_densifying(monkeypatch):
    """A U1U1 product MPS should retain its Symmray representation to sample."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    fermion = Fermion(spinful=True, symmetry="U1U1")
    psi = ps_to_mps(
        4,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
    )
    source_maps = [
        dict(psi[site].data.indices[psi[site].inds.index(psi.site_ind(site))].chargemap)
        for site in range(psi.L)
    ]

    def fail_dense(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("Symmray MPS sampling must not densify tensor data")

    monkeypatch.setattr(type(psi[0].data), "to_dense", fail_dense)
    sampler = sampler_mod.MpsSampler(psi)
    configs, probs = sampler.sample_arrays(4, seed=7)

    assert sampler.resolved_backend == "symmray"
    np.testing.assert_array_equal(configs, np.array([[2, 1, 2, 1]] * 4))
    np.testing.assert_allclose(probs, np.ones(4))
    assert [
        dict(psi[site].data.indices[psi[site].inds.index(psi.site_ind(site))].chargemap)
        for site in range(psi.L)
    ] == source_maps


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
def test_mps_sampler_symmray_fermionic_matches_direct_amplitudes(symmetry):
    """Entangled fermionic branches agree with direct sparse contraction."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        4,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(4),
        bond_dim=8,
        seed=3,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs, sampled_probs = sampler.sample_arrays(6, seed=9)
    probs = sampler.probabilities(configs)
    amplitudes = sampler.amplitudes(configs)
    norm = (psi.H @ psi).real

    assert sampler.resolved_backend == "symmray"
    np.testing.assert_allclose(probs, sampled_probs, atol=1e-12)
    np.testing.assert_allclose(np.abs(amplitudes) ** 2, sampled_probs, atol=1e-12)
    for config, probability in zip(configs, sampled_probs):
        branch = psi.copy()
        branch.isel_({
            branch.site_ind(site): int(code)
            for site, code in enumerate(config)
        })
        amplitude = np.asarray(branch.contract(all).data).item()
        assert probability == pytest.approx(abs(amplitude) ** 2 / norm)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
def test_mps_sampler_symmray_fermionic_sector_probabilities_are_exhaustive(
    symmetry,
):
    """All allowed physical codes normalize and forbidden sectors vanish."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    L = 3
    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        L,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(L),
        bond_dim=4,
        seed=11,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs = np.asarray(list(product(range(4), repeat=L)), dtype=np.int64)
    probabilities = sampler.probabilities(configs)
    amplitudes = sampler.amplitudes(configs)
    charge_maps = sampler.physical_code_maps
    target_charge = fermion.total_charge(fermion.half_filled_occupations(L))
    allowed = np.asarray(
        [
            fermion.total_charge(
                charge_maps[site][int(code)][0]
                for site, code in enumerate(config)
            )
            == target_charge
            for config in configs
        ]
    )

    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        probabilities,
        np.abs(amplitudes) ** 2,
        atol=1e-12,
    )
    np.testing.assert_allclose(probabilities[~allowed], 0.0, atol=1e-12)
    assert np.any(probabilities[allowed] > 1e-12)


@pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
def test_mps_sampler_symmray_dense_and_quimb_baselines_match(symmetry):
    """The benchmark's dense baselines preserve U1 fermionic Born weights."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        4,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(4),
        bond_dim=4,
        seed=13,
        dtype="complex128",
    ).mps
    sparse_sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    dense_psi = _dense_mps_from_symmray(psi)
    native_sampler = sampler_mod.MpsSampler(dense_psi, backend="native")
    quimb_sampler = sampler_mod.MpsSampler(dense_psi, backend="quimb")

    configs, sparse_probabilities = sparse_sampler.sample_arrays(16, seed=5)
    np.testing.assert_allclose(
        native_sampler.probabilities(configs),
        sparse_probabilities,
        atol=1e-12,
    )
    quimb_configs, quimb_probabilities = quimb_sampler.sample_arrays(16, seed=5)
    np.testing.assert_allclose(
        sparse_sampler.probabilities(quimb_configs),
        quimb_probabilities,
        atol=1e-12,
    )


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
def test_mps_sampler_fermion_diagonal_values_follow_symmray_physical_codes(
    symmetry,
):
    """Diagonal values use the physical code order of each fermionic symmetry."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        3,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(3),
        bond_dim=4,
        seed=15,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs = np.repeat(np.arange(4, dtype=np.int64)[:, None], psi.L, axis=1)
    expected_occupation = np.diag(fermion.observable("number").to_dense()).real
    expected_doublon = np.diag(fermion.observable("double").to_dense()).real

    np.testing.assert_allclose(
        sampler.fermion_diagonal_values(
            configs,
            fermion,
            "occupation",
            sites=0,
        ),
        expected_occupation,
    )
    np.testing.assert_allclose(
        sampler.fermion_diagonal_values(
            configs,
            fermion,
            "doublon",
            sites=0,
        ),
        expected_doublon,
    )
    np.testing.assert_allclose(
        sampler.fermion_diagonal_values(
            configs,
            fermion,
            "density_correlation",
            pairs=(0, 1),
        ),
        expected_occupation**2,
    )


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
def test_mps_sampler_fermion_diagonal_estimates_match_exact_born_sums(symmetry):
    """Sampled fermion observables agree with exact MPS Born sums."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    L = 3
    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        L,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(L),
        bond_dim=4,
        seed=17,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs = np.asarray(list(product(range(4), repeat=L)), dtype=np.int64)
    probabilities = sampler.probabilities(configs)
    observables = (
        ("occupation", {"sites": (0, 2)}),
        ("total_charge", {}),
        ("doublon", {}),
        ("density_correlation", {"pairs": ((0, 1), (1, 2))}),
    )

    for observable, kwargs in observables:
        exact = float(np.dot(
            probabilities,
            sampler.fermion_diagonal_values(configs, fermion, observable, **kwargs),
        ))
        estimate = sampler.estimate_fermion_diagonal(
            fermion,
            observable,
            n_samples=4096,
            seed=19,
            **kwargs,
        )

        assert isinstance(estimate, sampler_mod.MpsDiagonalEstimate)
        assert estimate.n_samples == 4096
        assert estimate.observable == observable
        assert abs(estimate.mean - exact) <= 7 * estimate.standard_error + 0.01


@pytest.mark.parametrize(
    ("symmetry", "occupations"),
    (
        ("Z2", (1, 1, 0, 2)),
        ("U1", (1, 1, 0, 2)),
        ("U1U1", ((1, 0), (0, 1), (0, 0), (1, 1))),
        ("Z2Z2", ((1, 0), (0, 1), (0, 0), (1, 1))),
    ),
)
def test_mps_sampler_symmray_supports_spinful_fermion_symmetries(
    monkeypatch,
    symmetry,
    occupations,
):
    """Fermionic charge maps, including degenerate sectors, stay sparse."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = ps_to_mps(4, fermion=fermion, occupations=occupations)
    source_maps = [
        dict(psi[site].data.indices[psi[site].inds.index(psi.site_ind(site))].chargemap)
        for site in range(psi.L)
    ]

    def fail_dense(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("Symmray MPS sampling must not densify tensor data")

    monkeypatch.setattr(type(psi[0].data), "to_dense", fail_dense)
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs, sampled_probs = sampler.sample_arrays(16, seed=4)

    assert sampler.resolved_backend == "symmray"
    assert configs.shape == (16, 4)
    assert np.all((0 <= configs) & (configs < 4))
    np.testing.assert_allclose(
        sampler.probabilities(configs),
        sampled_probs,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.abs(sampler.amplitudes(configs)) ** 2,
        sampled_probs,
        atol=1e-12,
    )
    assert [
        dict(psi[site].data.indices[psi[site].inds.index(psi.site_ind(site))].chargemap)
        for site in range(psi.L)
    ] == source_maps


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
def test_mps_sampler_symmray_supports_spinless_fermion_symmetries(symmetry):
    """The same physical-code reconstruction handles spinless fermions."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    psi = ps_to_mps(
        4,
        fermion=Fermion(spinful=False, symmetry=symmetry),
        occupations=(0, 1, 0, 1),
    )
    sampler = sampler_mod.MpsSampler(psi)
    configs, sampled_probs = sampler.sample_arrays(16, seed=4)

    assert sampler.resolved_backend == "symmray"
    np.testing.assert_array_equal(configs, np.array([[0, 1, 0, 1]] * 16))
    np.testing.assert_allclose(sampled_probs, np.ones(16))
    np.testing.assert_allclose(sampler.probabilities(configs), sampled_probs)


def test_mps_sampler_symmray_reuses_prefix_boundaries(monkeypatch):
    """A batch builds one conditional per distinct sampled prefix."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    psi = ps_to_mps(
        4,
        fermion=Fermion(spinful=True, symmetry="U1"),
        occupations=(1, 1, 0, 2),
    )
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    real_candidates = sampler_mod.MpsSampler._symmray_candidates.__func__
    calls = []

    def spy_candidates(cls, state, site, boundary):
        calls.append(site)
        return real_candidates(cls, state, site, boundary)

    monkeypatch.setattr(
        sampler_mod.MpsSampler,
        "_symmray_candidates",
        classmethod(spy_candidates),
    )
    configs, probs = sampler.sample_arrays(32, seed=3)

    assert calls[0] == 0
    assert len(calls) < 32 * psi.L
    np.testing.assert_allclose(sampler.probabilities(configs), probs, atol=1e-12)


@pytest.mark.parametrize(
    ("symmetry", "physical_sectors", "site_charges"),
    (
        ("Z2", {0: 1, 1: 1}, (0, 1, 0)),
        ("U1", {0: 1, 1: 1, 2: 1}, (1, 1, 1)),
        (
            "U1U1",
            {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            ((1, 0), (0, 1), (1, 0)),
        ),
        (
            "Z2Z2",
            {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            ((1, 0), (0, 1), (1, 0)),
        ),
    ),
)
def test_mps_sampler_symmray_supports_generic_abelian_mps(
    monkeypatch,
    symmetry,
    physical_sectors,
    site_charges,
):
    """Non-fermionic Symmray MPSs use the same no-densification sampler."""
    pytest.importorskip("symmray")
    from pepsy.tensors import SymMPS, site_charge_from_occupations

    psi = SymMPS.random(
        3,
        symmetry=symmetry,
        fermionic=False,
        bond_dim=4,
        phys_dim=physical_sectors,
        site_charge=site_charge_from_occupations(site_charges),
        seed=7,
        dtype="complex128",
    ).mps

    def fail_dense(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("Generic Symmray MPS sampling must not densify data")

    monkeypatch.setattr(type(psi[0].data), "to_dense", fail_dense)
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    local_dim = sum(physical_sectors.values())
    configs = np.asarray(list(product(range(local_dim), repeat=psi.L)), dtype=int)
    probabilities = sampler.probabilities(configs)
    amplitudes = sampler.amplitudes(configs)
    sampled_configs, sampled_probs = sampler.sample_arrays(32, seed=3)

    assert sampler.resolved_backend == "symmray"
    assert sampler.physical_code_maps is not None
    assert all(
        set(code_map) == set(range(local_dim))
        for code_map in sampler.physical_code_maps
    )
    np.testing.assert_allclose(np.abs(amplitudes) ** 2, probabilities, atol=1e-12)
    np.testing.assert_allclose(
        sampler.probabilities(sampled_configs),
        sampled_probs,
        atol=1e-12,
    )

    for config in sampled_configs[:4]:
        branch = psi.copy()
        branch.isel_({
            branch.site_ind(site): int(code)
            for site, code in enumerate(config)
        })
        value = branch.contract(all).data
        if hasattr(value, "get_scalar_element"):
            value = value.phase_sync().get_scalar_element()
        else:
            value = np.asarray(value).item()
        expected = abs(value) ** 2 / (psi.H @ psi).real
        assert sampler.probabilities(config[None, :])[0] == pytest.approx(expected)


def test_mps_sampler_symmray_physical_code_maps_retain_degenerate_sectors():
    """Source physical sectors remain interpretable after canonical pruning."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    psi = ps_to_mps(
        3,
        fermion=Fermion(spinful=True, symmetry="U1"),
        occupations=(1, 1, 0),
    )
    sampler = sampler_mod.MpsSampler(psi)
    maps = sampler.physical_code_maps

    assert maps == (
        {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 0)},
    ) * 3
    maps[0].clear()
    assert sampler.physical_code_maps[0][2] == (1, 1)


@pytest.mark.parametrize(
    ("symmetry", "expected"),
    (
        ("Z2", ((0, 0), (1, 1), (1, 0), (0, 1))),
        ("U1", ((0, 0), (0, 1), (1, 0), (1, 1))),
        ("U1U1", ((0, 0), (0, 1), (1, 0), (1, 1))),
        ("Z2Z2", ((0, 0), (0, 1), (1, 0), (1, 1))),
    ),
)
def test_mps_sampler_fermion_configuration_encoding_is_symmetry_aware(
    symmetry,
    expected,
):
    """Raw MPS codes decode consistently before entering a VMC workflow."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        3,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(3),
        bond_dim=4,
        seed=37,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    encoding = sampler.fermion_configuration_encoding(fermion)
    codes = np.repeat(np.arange(4, dtype=np.int64)[:, None], psi.L, axis=1)

    assert encoding.symmetry == symmetry
    assert encoding.spinful
    assert encoding.code_to_occupations == (expected,) * psi.L
    occupations = encoding.decode(codes)
    np.testing.assert_array_equal(occupations[:, 0], np.asarray(expected))
    np.testing.assert_array_equal(encoding.encode(occupations), codes)

    batch = sampler.sample_batch(8, seed=7, to_numpy=True, fermion=fermion)
    assert batch.configuration_encoding == encoding
    np.testing.assert_array_equal(batch.occupations(), encoding.decode(batch.configs))


def test_mps_sampler_fermion_configuration_encoding_rejects_wrong_symmetry():
    """MPS samples cannot silently reuse a codec from another symmetry."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    psi = ps_to_mps(
        2,
        fermion=Fermion(spinful=True, symmetry="U1"),
        occupations=(1, 1),
    )
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")

    with pytest.raises(ValueError, match="must match"):
        sampler.fermion_configuration_encoding(
            Fermion(spinful=True, symmetry="Z2")
        )


def test_mps_sampler_bound_fermion_sets_the_batch_configuration_contract():
    """A bound Fermion removes VMC code-convention boilerplate per batch."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry="Z2")
    psi = SymMPS.random(
        3,
        symmetry="Z2",
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(3),
        bond_dim=4,
        seed=43,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray", fermion=fermion)

    encoding = sampler.fermion_configuration_encoding()
    batch = sampler.sample_batch(8, seed=7, to_numpy=True)

    assert sampler.fermion is fermion
    assert batch.configuration_encoding == encoding
    np.testing.assert_array_equal(batch.occupations(), encoding.decode(batch.configs))

    codes = np.repeat(np.arange(4, dtype=np.int64)[:, None], psi.L, axis=1)
    np.testing.assert_allclose(
        sampler.fermion_diagonal_values(codes, "total_charge"),
        (0.0, 6.0, 3.0, 3.0),
    )


def test_mps_sampler_symmray_cached_branches_prune_charge_forbidden_work():
    """The sparse path skips impossible charge branches without densifying."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry="U1")
    psi = SymMPS.random(
        4,
        symmetry="U1",
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(4),
        bond_dim=4,
        seed=41,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs, probs = sampler.sample_arrays(32, seed=5)
    stats = sampler.symmray_sampling_stats

    assert stats["cached_local_slices"]
    assert stats["charge_pruned_branches"] > 0
    assert stats["candidate_contractions"] < 4 * stats["conditional_evaluations"]
    np.testing.assert_allclose(sampler.probabilities(configs), probs, atol=1e-12)


def test_mps_sampler_symmray_prefix_controls_bound_high_entropy_batches():
    """Prefix sharing reduces work and the auto cap falls back safely."""
    pytest.importorskip("symmray")
    from pepsy.tensors import SymMPS, site_charge_from_occupations

    psi = SymMPS.random(
        4,
        symmetry="U1",
        fermionic=False,
        bond_dim=4,
        phys_dim={0: 1, 1: 1, 2: 1},
        site_charge=site_charge_from_occupations((1, 1, 1, 1)),
        seed=7,
        dtype="complex128",
    ).mps
    shared = sampler_mod.MpsSampler(
        psi,
        backend="symmray",
        prefix_strategy="prefix",
        max_prefix_groups=None,
    )
    shared_configs, shared_probs = shared.sample_arrays(64, seed=3)
    bounded = sampler_mod.MpsSampler(
        psi,
        backend="symmray",
        prefix_strategy="auto",
        max_prefix_groups=1,
    )
    bounded_configs, bounded_probs = bounded.sample_arrays(64, seed=3)
    serial = sampler_mod.MpsSampler(
        psi,
        backend="symmray",
        prefix_strategy="serial",
    )
    serial_configs, serial_probs = serial.sample_arrays(64, seed=3)
    adaptive = sampler_mod.MpsSampler(
        psi,
        backend="symmray",
        prefix_strategy="auto",
        max_prefix_groups=None,
    )
    adaptive_configs, adaptive_probs = adaptive.sample_arrays(64, seed=3)

    assert shared.symmray_sampling_stats["conditional_evaluations"] < 64 * psi.L
    assert not shared.symmray_sampling_stats["serial_fallback"]
    assert bounded.symmray_sampling_stats["serial_fallback"]
    assert bounded.symmray_sampling_stats["max_active_prefix_groups"] <= 1
    assert serial.symmray_sampling_stats["conditional_evaluations"] == 64 * psi.L
    assert adaptive.symmray_sampling_stats["adaptive_serial_fallback"]
    np.testing.assert_allclose(
        shared.probabilities(shared_configs),
        shared_probs,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bounded.probabilities(bounded_configs),
        bounded_probs,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        serial.probabilities(serial_configs),
        serial_probs,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adaptive.probabilities(adaptive_configs),
        adaptive_probs,
        atol=1e-12,
    )


@pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
def test_mps_sampler_symmray_dense_strategy_uses_batched_native_kernel(symmetry):
    """The explicit dense strategy batches all Symmray shots safely."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry=symmetry)
    psi = SymMPS.random(
        4,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(4),
        bond_dim=4,
        seed=29,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(
        psi,
        backend="symmray",
        prefix_strategy="dense",
    )

    configs, sampled_probs = sampler.sample_arrays(64, seed=7)
    stats = sampler.symmray_sampling_stats

    assert stats["strategy"] == "dense"
    assert stats["conditional_evaluations"] == psi.L
    assert stats["dense_site_bytes"] > 0
    np.testing.assert_allclose(
        sampler.probabilities(configs),
        sampled_probs,
        atol=1e-12,
    )


def test_mps_sampler_rejects_invalid_symmray_prefix_controls():
    """Prefix controls should be explicit before any MPS preprocessing."""
    psi = qtn.MPS_computational_state("0")

    with pytest.raises(ValueError, match="Unknown Symmray prefix strategy"):
        sampler_mod.MpsSampler(psi, prefix_strategy="branchy")
    with pytest.raises(ValueError, match="max_prefix_groups"):
        sampler_mod.MpsSampler(psi, max_prefix_groups=0)


def test_mps_sampler_auto_selects_dense_within_memory_budget():
    """Auto strategy should use dense batching for a sufficiently large batch."""
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry="U1")
    psi = SymMPS.random(
        4,
        symmetry="U1",
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(4),
        bond_dim=4,
        seed=43,
        dtype="complex128",
    ).mps
    sampler = sampler_mod.MpsSampler(
        psi,
        strategy="auto",
        dense_min_samples=32,
        dense_memory_limit="1GiB",
    )

    configs, probs = sampler.sample_arrays(64, seed=5)
    stats = sampler.symmray_sampling_stats

    assert stats["requested_strategy"] == "auto"
    assert stats["strategy_selection"] == "auto_dense_within_budget"
    assert stats["strategy"] == "dense"
    np.testing.assert_allclose(sampler.probabilities(configs), probs, atol=1e-12)


def test_mps_sampler_rejects_conflicting_strategy_aliases_and_dense_budget():
    """The new strategy alias and dense guard should fail clearly."""
    psi = qtn.MPS_computational_state("0")
    with pytest.raises(ValueError, match="either strategy"):
        sampler_mod.MpsSampler(
            psi,
            strategy="dense",
            prefix_strategy="serial",
        )

    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymMPS

    fermion = Fermion(spinful=True, symmetry="U1")
    symm_psi = SymMPS.random(
        3,
        symmetry="U1",
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=fermion.half_filled_site_charge(3),
        bond_dim=3,
        seed=47,
        dtype="complex128",
    ).mps
    guarded = sampler_mod.MpsSampler(
        symm_psi,
        strategy="dense",
        dense_memory_limit="1B",
    )
    with pytest.raises(ValueError, match="above the configured limit"):
        guarded.sample_arrays(8, seed=1)


def test_mps_sampler_symmray_torch_nonfermionic_blocks_stay_on_torch():
    """Generic Symmray Torch blocks preserve device-resident outputs."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")
    from pepsy.tensors import SymMPS, site_charge_from_occupations

    psi = SymMPS.random(
        3,
        symmetry="U1",
        fermionic=False,
        bond_dim=3,
        phys_dim={0: 1, 1: 1, 2: 1},
        site_charge=site_charge_from_occupations((1, 1, 1)),
        seed=7,
        dtype="complex128",
        to_backend=torch.as_tensor,
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs, probs = sampler.sample_arrays(8, seed=3)

    assert isinstance(configs, torch.Tensor)
    assert isinstance(probs, torch.Tensor)
    torch.testing.assert_close(
        sampler.probabilities(configs, to_numpy=False),
        probs,
    )


def test_mps_sampler_symmray_cupy_nonfermionic_blocks_stay_on_cupy():
    """Generic Symmray CuPy blocks preserve device-resident outputs."""
    pytest.importorskip("symmray")
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")
    from pepsy.tensors import SymMPS, site_charge_from_occupations

    psi = SymMPS.random(
        3,
        symmetry="U1",
        fermionic=False,
        bond_dim=3,
        phys_dim={0: 1, 1: 1, 2: 1},
        site_charge=site_charge_from_occupations((1, 1, 1)),
        seed=7,
        dtype="complex128",
        to_backend=cupy.asarray,
    ).mps
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs, probs = sampler.sample_arrays(8, seed=3)

    assert isinstance(configs, cupy.ndarray)
    assert isinstance(probs, cupy.ndarray)
    cupy.testing.assert_allclose(
        sampler.probabilities(configs, to_numpy=False),
        probs,
    )


def test_mps_sampler_symmray_torch_blocks_stay_on_torch():
    """The Symmray path should keep Torch-backed blocks and samples on Torch."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")
    from pepsy.tensors import Fermion, ps_to_mps

    fermion = Fermion(spinful=True, symmetry="U1U1")
    psi = ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        to_backend=lambda array: torch.as_tensor(array),
    )
    sampler = sampler_mod.MpsSampler(psi, backend="symmray")
    configs, probs = sampler.sample_arrays(3, seed=7)
    batch = sampler.sample_batch(3, seed=7, fermion=fermion)

    assert sampler.resolved_backend == "symmray"
    assert isinstance(configs, torch.Tensor)
    assert isinstance(probs, torch.Tensor)
    assert batch.backend == "torch"
    assert isinstance(batch.occupations(), torch.Tensor)
    assert tuple(batch.occupations().shape) == (3, 3, 2)
    torch.testing.assert_close(
        configs,
        torch.tensor([[2, 1, 2]] * 3, dtype=torch.long, device=configs.device),
    )
    torch.testing.assert_close(probs, torch.ones(3, dtype=probs.dtype))


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
