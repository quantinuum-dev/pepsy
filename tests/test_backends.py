"""Tests for public backend conversion helpers."""

import numpy as np
import pytest

import pepsy


def test_to_float_is_public_backend_helper():
    assert pepsy.to_float is pepsy.backends.to_float


def test_to_float_handles_backend_scalar_without_numpy_coercion():
    class BackendScalar:
        shape = ()

        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 1.25

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("NumPy conversion should not be used.")

    assert pepsy.to_float(BackendScalar()) == pytest.approx(1.25)


def test_to_float_rejects_non_scalar_backend_array_before_numpy_coercion():
    class BackendVector:
        shape = (2,)

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("Non-scalar backend arrays should not coerce.")

    with pytest.raises(TypeError, match="shape"):
        pepsy.to_float(BackendVector())


def test_to_float_uses_real_component_by_default():
    value = np.asarray(2.5 - 1.0j)

    assert pepsy.to_float(value) == pytest.approx(2.5)
    with pytest.raises(TypeError):
        pepsy.to_float(value, real=False)


def test_to_float_handles_torch_scalar_if_available():
    torch = pytest.importorskip("torch")

    value = torch.tensor(3.5 + 1.25j, dtype=torch.complex128, requires_grad=True)

    assert pepsy.to_float(value) == pytest.approx(3.5)
