"""Native MPS entropy diagnostics."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.tensors import mps_entanglement_entropy


def _available_backends():
    backends = ["numpy", "torch", "cupy"]
    try:
        import torch
    except ImportError:
        backends.remove("torch")
    else:
        if not torch.cuda.is_available():
            # CPU Torch still exercises the native backend path.
            pass
    try:
        import cupy
    except ImportError:
        backends.remove("cupy")
    else:
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                backends.remove("cupy")
        except cupy.cuda.runtime.CUDARuntimeError:
            backends.remove("cupy")
    return backends


@pytest.mark.parametrize("backend", _available_backends())
def test_mps_entropy_is_native_and_preserves_source(monkeypatch, backend):
    vector = np.zeros(16, dtype=np.complex128)
    vector[0], vector[-1] = np.sqrt(0.8), 1j * np.sqrt(0.2)
    state = qtn.MatrixProductState.from_dense(3.0 * vector, dims=[2] * 4)
    if backend == "torch":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pepsy.TorchLinalgConfig(
            mode="complex",
            stabilized=False,
            svd_driver="gesvd",
            cpu_svd="torch",
        ).register()
        state.apply_to_arrays(
            pepsy.backend_torch(device=device, dtype=torch.complex128),
        )
    elif backend == "cupy":
        state.apply_to_arrays(pepsy.backend_cupy(dtype="complex128"))

    before = [array.clone() if backend == "torch" else array.copy()
              for array in state.arrays]
    exponent = state.exponent = -7.0

    def forbidden(_value):
        raise AssertionError("MPS entropy must not convert tensor arrays to NumPy.")

    monkeypatch.setattr(pepsy.tensors.observables.ar, "to_numpy", forbidden)
    if backend == "torch":
        import torch

        monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)

    expected = -0.8 * np.log2(0.8) - 0.2 * np.log2(0.2)
    assert mps_entanglement_entropy(state) == pytest.approx(expected, abs=1e-12)
    assert mps_entanglement_entropy(state, method="eig") == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert state.exponent == exponent
    for actual, original in zip(state.arrays, before):
        if backend == "torch":
            import torch

            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        else:
            assert np.array_equal(actual, original)


def test_mps_entropy_rejects_cyclic_and_invalid_cuts():
    cyclic = qtn.MatrixProductState(
        [np.ones((2, 2, 2), dtype=complex)] * 3,
        shape="lrp",
    )
    with pytest.raises(ValueError, match="open MPS"):
        mps_entanglement_entropy(cyclic)

    state = qtn.MPS_computational_state("0000")
    with pytest.raises(ValueError, match="0 < cut < 4"):
        mps_entanglement_entropy(state, cut=0)


def test_mps_optimizer_and_sampler_delegate_to_package_entropy():
    state = qtn.MPS_ghz_state(4)
    optimizer = pepsy.MpsOptimizer(state, gates=[], chi=4, mode="direct")
    sampler = pepsy.MpsSampler(state, backend="native")

    assert optimizer.entropy() == pytest.approx(1.0, abs=1e-12)
    assert optimizer.entanglement_entropy(cut=2) == pytest.approx(1.0, abs=1e-12)
    assert sampler.entanglement_entropy() == pytest.approx(1.0, abs=1e-12)
