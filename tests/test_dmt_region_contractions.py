"""Independent partial traces and extreme-scale DMT region contractions."""

import math

import numpy as np
import pytest

from pepsy.operators._pauli_dmt import _region_trace_data, _traced
from pepsy.operators.pauli_mpo import _host_array


def _reference(cores, region, trace):
    """One-sided NumPy contraction with explicit physical-index bookkeeping."""
    value = np.ones(1)
    for site, core in enumerate(cores):
        if site in region:
            value = np.einsum("...a,abp->...pb", value, core)
        else:
            value = value @ np.einsum("abp,p->ab", core, trace)
    return value[..., 0].reshape(-1)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "complex128"])
def test_balanced_region_panels_match_independent_contraction(backend, ordinary, dtype, request):
    module = pytest.importorskip(backend)
    convert = np.asarray
    if backend == "torch":
        convert = module.as_tensor
    elif backend == "jax":
        previous = module.config.x64_enabled
        request.addfinalizer(lambda: module.config.update("jax_enable_x64", previous))
        module.config.update("jax_enable_x64", True)
        convert = pytest.importorskip("jax.numpy").asarray
    rng = np.random.default_rng(603)
    bonds = [1, 3, 3, 3, 3, 3, 1]
    cores = []
    for dl, dr in zip(bonds, bonds[1:]):
        # Preserve noncontiguous physical-before-bond strides, as after QR.
        a = rng.normal(size=(dl, 4, dr))
        if dtype.startswith("complex"):
            a = a + 1j * rng.normal(size=a.shape)
        cores.append(a.astype(dtype).transpose(0, 2, 1))
    trace = np.array([1., 0., 0., 1.] if ordinary else [1., 0., 0., 0.], dtype=dtype)
    regions = [(0,), (1, 5), (0, 2, 5), (0, 1, 4, 5), (0, 1, 2, 4, 5)]
    native = [convert(a) for a in cores]
    panels = _region_trace_data(native, regions, convert(trace) if ordinary else None)
    for region, (direction, log_scale) in zip(regions, panels):
        expected = _reference(cores, region, trace)
        actual = _host_array(direction) * math.exp(log_scale)
        np.testing.assert_allclose(actual, expected, rtol=3e-5 if dtype == "float32" else 3e-13,
                                   atol=3e-4 if dtype == "float32" else 3e-12)
    np.testing.assert_allclose(_host_array(_traced(native[2], convert(trace))),
                               np.einsum("abp,p->ab", cores[2], trace),
                               rtol=2e-6 if dtype == "float32" else 1e-14)


@pytest.mark.parametrize("scale", [1e-10, 1e10])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_region_panels_retain_extreme_global_scales_in_log_form(dtype, scale):
    n = 64
    local = np.array([.6, 0., 0., .4], dtype=dtype)
    core = (local * np.asarray(scale, dtype=dtype)).reshape(1, 1, 4)
    trace = np.array([1., 0., 0., 1.], dtype=dtype)
    regions = [(0,), (7, 56), (0, 1, 62, 63)]
    panels = _region_trace_data([core] * n, regions, trace)
    # Compute the oracle from the represented dtype, including input rounding.
    represented = core.reshape(-1).astype("float64")
    scalar = represented.sum()
    normalized = represented / scalar
    for region, (direction, log_scale) in zip(regions, panels):
        expected = np.ones(1)
        for _ in region:
            expected = np.kron(expected, normalized)
        norm = np.linalg.norm(expected)
        np.testing.assert_allclose(direction, expected / norm, atol=3e-7)
        assert log_scale == pytest.approx(n * math.log(scalar) + math.log(norm), abs=2e-5)


def test_zero_environment_and_qutrit_matrix_unit_regions():
    trace = np.eye(3).reshape(-1)
    local = np.arange(9, dtype=float) / 9
    cores = [local.reshape(1, 1, 9)] * 5
    cores[2] = np.zeros((1, 1, 9))
    for value, log_scale in _region_trace_data(cores, [(0, 4), (1, 2, 4)], trace):
        np.testing.assert_array_equal(value, 0.)
        assert log_scale == -math.inf
    cores[2] = np.diag([1., -1., 0.]).reshape(1, 1, 9)
    regions = [(0, 4), (1, 2, 4)]
    for region, (value, log_scale) in zip(regions, _region_trace_data(cores, regions, trace)):
        expected = _reference(cores, region, trace)
        actual = np.zeros_like(value) if log_scale == -math.inf else value * math.exp(log_scale)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
