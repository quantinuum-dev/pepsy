"""Native tensor/rho/cache placement for boundary and proposal-only sampling."""

from contextlib import nullcontext

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.backends import infer_backend_signature
from pepsy.sampling import PepsSampler


@pytest.mark.parametrize("backend", ["numpy", "torch_cpu", "torch_cuda", "jax_cpu", "cupy"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("mode", ["boundary", "none"])
def test_backend_native_sampling_caches_and_rhos(backend, dtype, mode, monkeypatch):
    context = nullcontext()
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        if backend == "torch_cuda" and not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        device = "cuda:0" if backend == "torch_cuda" else "cpu"
        convert = lambda a: torch.as_tensor(a.copy(), device=device)  # noqa: E731
    elif backend == "jax_cpu":
        jax = pytest.importorskip("jax")
        enable_x64 = getattr(jax, "enable_x64", None)
        if enable_x64 is None:
            from jax.experimental import enable_x64

        context = enable_x64()
        convert = lambda a: jax.device_put(a, jax.devices("cpu")[0])  # noqa: E731
    elif backend == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            available = cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError:
            available = 0
        if not available:
            pytest.skip("CuPy CUDA unavailable")
        convert = cupy.asarray
    else:
        convert = np.asarray

    with context:
        source = qtn.PEPS.rand(3, 2, 2, seed=71, dtype=dtype)
        source.apply_to_arrays(convert)
        signature = infer_backend_signature(source.tensors[0].data)
        to_numpy = ar.to_numpy

        def checked_transfer(array):
            if ar.infer_backend(array) != "numpy" and ar.ndim(array) > 0:
                assert np.dtype(ar.get_dtype_name(array)).kind not in "fc", "floating tensor copied to host"
            return to_numpy(array)

        with monkeypatch.context() as patch:
            patch.setattr(ar, "to_numpy", checked_transfer)
            sampler = PepsSampler(source, chi=8, chi_prime=4, amplitude_mode=mode,
                                  boundary_engine="quimb-mps", contraction_opt="greedy",
                                  row_contraction_opt="greedy")
            original_rho = sampler._row_local_rho
            seen = []

            def rho(*args):
                result = original_rho(*args)
                assert infer_backend_signature(result[0]) == signature
                seen.append(True)
                return result

            patch.setattr(sampler, "_row_local_rho", rho)
            batch = sampler.sample_batch(6, seed=9, chunk_size=3)
            assert seen
            networks = [sampler._ket, sampler._norm, *sampler._future_environments.values(),
                        sampler._last_boundary_mps]
            for network in networks:
                if network is not None:
                    assert all(infer_backend_signature(t.data) == signature for t in network)
            cache = sampler._initial_row_cache
            assert cache["factored"]
            for tensor in [t for factors in cache["local"] for t in factors] + list(cache["right"]):
                if tensor is not None:
                    assert infer_backend_signature(tensor.data) == signature
            if mode == "none":
                assert sampler._amplitude_leaves is None
                assert batch.ps is None
            else:
                assert all(infer_backend_signature(data) == signature
                           for data, _ in sampler._amplitude_leaves)
        # Result postprocessing is an explicitly documented host operation.
        assert np.isfinite(batch.log_probabilities).all()
        np.testing.assert_allclose(batch.normalized_weights.sum(), 1)
