"""Backend-resident dense Kraus batching and rare sampling probabilities."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer, noise


@pytest.fixture(params=[
    "numpy",
    pytest.param("torch", marks=pytest.mark.optional),
    pytest.param("cuda", marks=[pytest.mark.optional, pytest.mark.integration]),
    pytest.param("cupy", marks=[pytest.mark.optional, pytest.mark.integration]),
    pytest.param("jax", marks=[pytest.mark.optional, pytest.mark.integration]),
])
def convert(request):
    if request.param == "numpy":
        return lambda x: np.asarray(x, dtype="complex64")
    if request.param in {"torch", "cuda"}:
        torch = pytest.importorskip("torch")
        device = "cuda" if request.param == "cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        return lambda x: torch.tensor(np.array(x), device=device, dtype=torch.complex64)
    if request.param == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        return lambda x: cp.asarray(x, dtype="complex64")
    jnp = pytest.importorskip("jax.numpy")
    return lambda x: jnp.asarray(x, dtype="complex64")


@pytest.mark.parametrize("mode,where", [
    ("direct", (1, 2)), ("direct", (3, 0)), ("exact", (3, 0)),
])
def test_batched_probabilities_match_dense_and_only_download_norms(convert, mode, where, monkeypatch):
    rng = np.random.default_rng(712)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    p = qtn.MatrixProductState.from_dense(vector, [2]*4, cutoff=0.)
    p.apply_to_arrays(convert)
    opt = MpsOptimizer(p, chi=4, mode=mode)
    before = ar.to_numpy(opt.to_dense()).ravel()
    basis, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))
    channel = noise.TrajectoryChannel.kraus([
        (str(i), np.outer(basis[:, i], basis[:, i].conj())) for i in range(4)
    ])
    order = where + tuple(i for i in range(4) if i not in where)
    block = before.reshape((2,)*4).transpose(order).reshape(4, -1)
    expected = [np.linalg.norm(outcome.gate @ block)**2 for outcome in channel.outcomes]
    expected /= np.sum(expected)
    # Warm immutable gate conversion before measuring the readback boundary.
    noise._kraus_probabilities(opt, channel, where)
    original = ar.to_numpy
    downloads = []

    def only_norms(value):
        assert tuple(value.shape) == (4,), "Downloaded state or operator data"
        downloads.append(value)
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", only_norms)
        result = noise._kraus_probabilities(opt, channel, where)
    assert len(downloads) == 1
    assert ar.infer_backend(downloads[0]) == ar.infer_backend(p[0].data)
    np.testing.assert_allclose(result, expected, atol=5e-7)
    np.testing.assert_allclose(ar.to_numpy(opt.to_dense()).ravel(), before, atol=5e-7)
    assert all(t.data.dtype == p[0].data.dtype for t in opt.p.tensors)
    assert all(getattr(t.data, "device", None) == getattr(p[0].data, "device", None)
               for t in opt.p.tensors)


def test_batched_rare_probability_survives_float32_square_underflow(convert):
    p = qtn.MPS_computational_state("0", dtype="complex64")
    p.apply_to_arrays(convert)
    opt = MpsOptimizer(p, chi=2)
    amplitude = 1e-30
    channel = noise.TrajectoryChannel.kraus([
        ("common", np.eye(2)), ("rare", np.diag([amplitude, 0.])),
        ("impossible", np.zeros((2, 2))),
    ])
    actual = noise._kraus_probabilities(opt, channel, (0,))
    assert actual[1] == pytest.approx(amplitude**2, rel=1e-5, abs=0.)
    assert actual[2] == 0.


def test_bounded_batches_preserve_outcome_order_and_reuse_routing(convert, monkeypatch):
    p = qtn.MPS_rand_state(4, 3, dtype="complex64", seed=781)
    p.apply_to_arrays(convert)
    opt = MpsOptimizer(p, chi=4)
    # Nine outcomes force several batches, including an incomplete final batch.
    channel = noise.TrajectoryChannel.kraus([
        (str(i), np.sqrt((i+1)/45)*np.eye(4)) for i in range(9)
    ])
    monkeypatch.setattr(noise, "_KRAUS_BATCH_MAX_ELEMENTS", 64)
    original_swap = qtn.MatrixProductState.swap_site_to_
    original_namespace = noise._array_namespace
    swaps, batch_sizes = [], []

    def swap(state, *args, **kwargs):
        swaps.append(1)
        return original_swap(state, *args, **kwargs)

    # Observe the actual projected batches while leaving dispatch untouched.
    class Namespace:
        def __init__(self, xp):
            self.xp = xp

        def __getattr__(self, name):
            return getattr(self.xp, name)

        def matmul(self, matrices, block):
            batch_sizes.append(int(matrices.shape[0]))
            return self.xp.matmul(matrices, block)

    monkeypatch.setattr(noise, "_array_namespace", lambda like: Namespace(original_namespace(like)))
    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", swap)
    result = noise._kraus_probabilities(opt, channel, (3, 0))
    np.testing.assert_allclose(result, np.arange(1, 10)/45, atol=2e-7)
    assert len(swaps) == 2  # One support preparation, independent of outcomes.
    assert sum(batch_sizes) == 9 and max(batch_sizes) <= 2


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_batched_kraus_replay_preserves_backend_and_importance_weights(convert, strategy):
    p = qtn.MPS_computational_state("10", dtype="complex64")
    p.apply_to_arrays(convert)
    result = MpsOptimizer(p, [("amplitude_damping", .2, 0)], chi=2).run(
        shots=16, strategy=strategy, seed=9, workers=1,
        importance_sampling={"jump": .5, "no_jump": .5},
        progress=False, progbar=False,
    )
    seen = set()
    for optimizer, records, weight in zip(result.optimizers, result.records, result.weights):
        record = records[0]
        seen.add(record.label)
        probability = .2 if record.label == "jump" else .8
        assert record.probability == pytest.approx(probability, abs=2e-7)
        assert weight == pytest.approx(probability/.5, abs=4e-7)
        expected = [1, 0, 0, 0] if record.label == "jump" else [0, 0, 1, 0]
        np.testing.assert_allclose(ar.to_numpy(optimizer.to_dense()).ravel(), expected, atol=3e-7)
        assert optimizer.backend == ar.infer_backend(p[0].data)
    assert seen == {"jump", "no_jump"}


@pytest.mark.optional
@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cuda", "cupy"])
def test_real_gpu_batch_cache_follows_state_dtype(backend):
    if backend == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        convert = lambda x, dtype: torch.tensor(np.array(x), device="cuda", dtype=getattr(torch, dtype))
    else:
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        convert = lambda x, dtype: cp.asarray(x, dtype=dtype)
    channel = noise.TrajectoryChannel.amplitude_damping(.3)
    opt = None
    for dtype in ("float32", "float64"):
        state = qtn.MPS_computational_state("1", dtype=dtype)
        state.apply_to_arrays(lambda x: convert(x, dtype))
        if opt is None:
            opt = MpsOptimizer(state, chi=2)
        else:
            opt.set_p(state)
        np.testing.assert_allclose(noise._kraus_probabilities(opt, channel, (0,)), [.7, .3], atol=2e-7)
        assert ar.get_dtype_name(opt.p[0].data) == dtype


@pytest.mark.optional
def test_probability_query_preserves_live_torch_gradient():
    torch = pytest.importorskip("torch")
    theta = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    vector = torch.stack((theta.cos(), theta.sin())).to(torch.complex128)
    p = qtn.MPS_product_state([vector])
    opt = MpsOptimizer(p, chi=2)
    noise._kraus_probabilities(opt, noise.TrajectoryChannel.amplitude_damping(.3), (0,))
    loss = opt.to_dense().abs().square().reshape(-1)[1]
    loss.backward()
    assert theta.grad.item() == pytest.approx(np.sin(.74), abs=1e-12)
