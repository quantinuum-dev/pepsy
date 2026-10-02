"""Independent references for sampler input, scale and gradient contracts."""

import numpy as np
import pytest

from pepsy.optimizers import TreePlan, TreeTensorNetwork
from pepsy.sampling import TreeSampler

pytestmark = [pytest.mark.tree, pytest.mark.integration, pytest.mark.optional]


@pytest.fixture(params=["numpy", "torch", "torch_cuda", "cupy"])
def convert(request):
    if request.param.startswith("torch"):
        torch = pytest.importorskip("torch")
        if request.param == "torch_cuda" and not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        device = "cuda:0" if request.param == "torch_cuda" else "cpu"
        return lambda a: torch.as_tensor(a, device=device)
    if request.param == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        return cp.asarray
    return np.asarray


def _single_site(convert, values):
    state = TreeTensorNetwork.from_order([0], dtype="float64")
    state.node_tensor(state.node_of_qubit(0)).modify(data=convert(np.asarray(values)))
    return state


@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200])
@pytest.mark.parametrize("strategy", ["standard", "factor"])
def test_finite_root_scale_preserves_born_weights(convert, scale, strategy):
    state = _single_site(convert, np.asarray([1.0, 2.0]) * scale)
    sampler = TreeSampler(state, strategy=strategy)
    np.testing.assert_allclose(sampler.probabilities([[0], [1]]), [0.2, 0.8], rtol=1e-12)
    batch = sampler.sample_batch(13, seed=17).to_numpy()
    expected = (np.random.default_rng(17).random(13) > 0.2).astype(int)
    np.testing.assert_array_equal(batch.configs[:, 0], expected)
    np.testing.assert_allclose(batch.probs, np.asarray([0.2, 0.8])[expected], rtol=1e-12)


@pytest.mark.parametrize("values", [[0.0, 0.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_invalid_dense_norm_rejected(convert, values):
    with pytest.raises(ValueError, match="zero or non-finite"):
        TreeSampler(_single_site(convert, values))


@pytest.mark.parametrize("scale", [1e-320, 8e307])
def test_extreme_complex_root_scale_stays_finite(convert, scale):
    values = np.asarray([0.5 + 0.5j, 1.0 + 1.0j]) * scale
    sampler = TreeSampler(_single_site(convert, values))
    np.testing.assert_allclose(sampler.probabilities([[0], [1]]), [0.2, 0.8], rtol=1e-12)


def test_subnormal_complex64_score_stays_finite(convert):
    values = np.asarray([1e-40, 1.0], dtype=np.complex64)
    native = convert(values)
    if type(native).__module__.startswith("cupy"):
        # CuPy compiles arithmetic with -ftz=true. Record the absent upstream
        # capability explicitly; a passing normal-range test cannot cover it.
        if float((native.real * 1)[0].get()) == 0:
            pytest.skip("CuPy float32 arithmetic flushes subnormal source components to zero")
    sampler = TreeSampler(_single_site(convert, values))
    expected = float(values[0].real) ** 2
    np.testing.assert_allclose(sampler.probabilities([[0]]), [expected], rtol=3e-7, atol=0)


def test_scoring_normalization_keeps_torch_gradient():
    torch = pytest.importorskip("torch")
    values = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    state = _single_site(lambda _: values, [1.0, 2.0])
    sampler = TreeSampler(state)
    scores = sampler.probabilities([[0], [1]], to_numpy=False)
    gradient = torch.autograd.grad(scores[0], values, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.tensor([0.32, -0.16], dtype=torch.float64))
    total_gradient = torch.autograd.grad(scores.sum(), values)[0]
    torch.testing.assert_close(total_gradient, torch.zeros_like(values), atol=1e-14, rtol=0)


def test_invalid_configs_rejected_before_indexing(convert):
    sampler = TreeSampler(_single_site(convert, [1.0, 2.0]))
    for value in (-1, 2, 0.8, np.nan, np.inf, 2**64 - 1, 1j, "1"):
        for method in (sampler.amplitudes, sampler.probabilities,
                       sampler.single_site_flip_amplitude_ratios):
            with pytest.raises(ValueError, match="physical codes"):
                method([[value]])
    np.testing.assert_allclose(sampler.probabilities([[0.0], [1.0]]), [0.2, 0.8])
    assert sampler.probabilities(np.empty((0, 1), dtype=int)).shape == (0,)


def test_native_single_site_flip_ratios_preserve_charge():
    pytest.importorskip("symmray")
    state = TreeTensorNetwork.from_symmray_plan(
        TreePlan.from_order(range(4), max_arity=2, top_arity=2),
        symmetry="U1", physical_sectors={0: 1, 1: 1},
        leaf_charges={0: 0, 1: 1, 2: 0, 3: 1}, bond_dim=4,
        fermionic=False, seed=3,
    )
    sampler = TreeSampler(state, backend="symmray")
    configs = sampler.sample_batch(3, seed=11).configs
    actual = sampler.single_site_flip_amplitude_ratios(configs, to_numpy=False)
    assert actual.shape == (3, 4)
    np.testing.assert_array_equal(actual, np.zeros((3, 4)))
    for q in range(4):
        flipped = configs.copy()
        flipped[:, q] = 1 - flipped[:, q]
        expected = sampler.amplitudes(flipped) / sampler.amplitudes(configs)
        np.testing.assert_allclose(actual[:, q], expected)
    with pytest.raises(ValueError, match="physical codes"):
        sampler.probabilities([[-1, 0, 0, 1]])


def test_native_absent_charge_sector_remains_a_valid_code():
    pytest.importorskip("symmray")
    state = TreeTensorNetwork.from_symmray_plan(
        TreePlan.from_order(range(3)), symmetry="U1",
        physical_sectors={0: 1, 1: 1}, leaf_charges={0: 0, 1: 0, 2: 0},
        bond_dim=2, fermionic=False, seed=3,
    )
    sampler = TreeSampler(state, backend="symmray")
    assert all(len(codes) == 2 for codes in sampler.physical_code_maps)
    np.testing.assert_array_equal(sampler.probabilities([[1, 0, 0]]), [0.0])
    np.testing.assert_array_equal(sampler.single_site_flip_amplitude_ratios([[0, 0, 0]]),
                                  np.zeros((1, 3)))


@pytest.mark.parametrize("strategy", ["standard", "factor"])
def test_zero_conditional_rejected_instead_of_returning_samples(convert, strategy):
    sampler = TreeSampler(_single_site(convert, [1.0, 2.0]), strategy=strategy)
    sampler._arrays[sampler._root] = sampler._arrays[sampler._root] * 0
    with pytest.raises(ValueError, match="conditional norm"):
        sampler.sample_arrays(3, seed=0)


@pytest.mark.parametrize("n", [160, 320])
@pytest.mark.parametrize("strategy,chunk", [("standard", None), ("standard", 3),
                                           ("factor", None), ("factor", 3)])
def test_long_complex64_product_matches_analytic_draws(convert, strategy, chunk, n):
    state = TreeTensorNetwork.from_order(range(n), dtype="complex64")
    for q in range(n):
        tensor = state.node_tensor(state.node_of_qubit(q))
        tensor.modify(data=np.full(tensor.shape, 1 / np.sqrt(2), dtype=np.complex64))
    state.apply_to_arrays(convert)
    sampler = TreeSampler(state, strategy=strategy, chunk_size=chunk)
    batch = sampler.sample_batch(8, seed=11).to_numpy()
    draws = np.random.default_rng(11).random((n, 8))
    expected = np.empty((8, n), dtype=int)
    expected[:, sampler._sampling_qubits] = (draws.T > 0.5).astype(int)
    np.testing.assert_array_equal(batch.configs, expected)
    np.testing.assert_allclose(batch.probs, np.full(8, 2.0**-n), rtol=2e-5, atol=0)
    np.testing.assert_allclose(sampler.probabilities(batch.configs), np.full(8, 2.0**-n),
                               rtol=2e-5, atol=0)


@pytest.mark.parametrize("strategy", ["standard", "factor"])
def test_zero_draw_skips_zero_weight_branch(convert, strategy):
    sampler = TreeSampler(_single_site(convert, [0.0, 1.0]), strategy=strategy)
    with sampler._array_device_context():
        configs, probs = sampler._sample_arrays(
            3, np.random.default_rng(0), physical_draws=sampler._as_backend(np.zeros((1, 3))),
        )
    np.testing.assert_array_equal(sampler.probabilities(configs), np.ones(3))
    np.testing.assert_array_equal(sampler.probabilities([[1]]), [1.0])
    # The public score confirms the sampled code; the returned native weight
    # must agree as well without a host-only fallback.
    np.testing.assert_array_equal(sampler.sample_batch(3, seed=0).to_numpy().probs, [1.0] * 3)
    assert bool((probs == 1).all())


@pytest.mark.parametrize("strategy", ["standard", "factor"])
@pytest.mark.parametrize("chunk", [None, 2])
@pytest.mark.parametrize("dtype,components", [
    ("float64", [0.5085427383566555, 0.8648174831367542, 0.8668144811764711, 0.0]),
    ("float32", [0.1564759761095047, 0.8447070121765137, 0.22817593812942505, 0.0]),
])
def test_largest_draw_skips_trailing_zero_weight(convert, strategy, chunk, dtype, components):
    values = np.asarray(components, dtype=dtype)
    sampler = TreeSampler(_single_site(convert, values), strategy=strategy, chunk_size=chunk)

    class LargestDraws:
        def random(self, shape):
            return np.full(shape, np.nextafter(1.0, 0.0))

    sampler._rng = LargestDraws()
    batch = sampler.sample_batch(3).to_numpy()
    np.testing.assert_array_equal(batch.configs, np.full((3, 1), 2))
    weights = values.astype(np.float64) ** 2
    expected = weights[2] / weights.sum()
    tolerance = 2e-6 if dtype == "float32" else 1e-13
    np.testing.assert_allclose(batch.probs, expected, rtol=tolerance)
    np.testing.assert_allclose(sampler.probabilities(batch.configs), expected, rtol=tolerance)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_general_torch_scoring_gradients_match_dense_reference(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")
    plan = TreePlan.from_order([0, 1, 3], root_qubit=2)
    state = TreeTensorNetwork.rand(plan, D=3, seed=19, dtype="complex128")
    state.apply_to_arrays(lambda a: torch.as_tensor(a, device=device).requires_grad_())
    state.invalidate_canonical_form()
    inputs = tuple(t.data for t in state.tensors)
    dense = state.contract(all, output_inds=[state.site_ind(q) for q in range(4)]).data.reshape(-1)
    expected = dense.abs() ** 2
    expected = expected / expected.sum()
    configs = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1)
    actual = TreeSampler(state).probabilities(configs, to_numpy=False)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=1e-14)
    weights = torch.arange(16, dtype=torch.float64, device=device)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), inputs, retain_graph=True)
    expected_gradient = torch.autograd.grad((expected * weights).sum(), inputs)
    for a, e in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-10)
