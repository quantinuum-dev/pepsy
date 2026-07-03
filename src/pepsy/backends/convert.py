"""Shared backend inference and array-conversion helpers."""

from __future__ import annotations

import numpy as np
import autoray as ar

_SCALAR_TYPES = (int, float, complex, bool, np.number)

_NUMPY_DTYPE_MAP = {
    "complex128": np.complex128,
    "complex64": np.complex64,
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
}


def _backend_scalar(value):
    """Return a Python scalar from scalar-like backend values."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        shape = tuple(shape)
        if shape != ():
            raise TypeError(f"Expected a scalar-like value, got shape {shape}.")

    obj = value
    for method_name in ("detach", "cpu"):
        method = getattr(obj, method_name, None)
        if callable(method):
            obj = method()

    get = getattr(obj, "get", None)
    if callable(get) and shape is not None:
        obj = get()

    item = getattr(obj, "item", None)
    if callable(item) and not isinstance(obj, _SCALAR_TYPES):
        try:
            obj = item()
        except (TypeError, ValueError, RuntimeError):
            pass

    if isinstance(obj, _SCALAR_TYPES):
        return obj

    arr = np.asarray(obj)
    if arr.shape != ():
        raise TypeError(f"Expected a scalar-like value, got shape {arr.shape}.")
    return arr.item()


def to_float(value, *, real=True):
    """Convert a scalar-like backend value to a Python ``float``.

    The input can be a Python scalar, NumPy scalar or scalar array, or a
    scalar-like backend tensor. Torch-style values are detached and moved to
    CPU before extracting ``.item()``; CuPy-style values are host-transferred
    with ``.get()`` when available. Non-scalar arrays raise ``TypeError``.

    Parameters
    ----------
    value
        Scalar-like object to convert.
    real
        If ``True`` (default), return the real component. This is convenient
        for expectation values whose imaginary part should be numerical noise.
        If ``False``, complex values follow Python's normal ``float(...)``
        rules and therefore raise when they are not real.
    """
    scalar = _backend_scalar(value)
    if real:
        scalar = np.real(scalar)
    return float(scalar)


def resolve_backend_sample_data(obj):
    """Return representative array data from an array- or gate-like object."""
    if obj is None:
        return None

    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return obj

    data = getattr(obj, "data", None)
    if hasattr(data, "shape") and hasattr(data, "dtype"):
        return data

    return None


def resolve_backend_sample_data_from_tn(tn):
    """Return representative array data from a tensor network-like object."""
    if tn is None:
        return None

    tensor_map = getattr(tn, "tensor_map", None)
    if tensor_map:
        first_tensor = next(iter(tensor_map.values()))
        data = getattr(first_tensor, "data", None)
        if hasattr(data, "shape") and hasattr(data, "dtype"):
            return data

    try:
        first_tensor = next(iter(tn))
    except (TypeError, StopIteration):
        return None

    data = getattr(first_tensor, "data", None)
    if hasattr(data, "shape") and hasattr(data, "dtype"):
        return data

    return None


def infer_backend_and_dtype(sample_data):
    """Infer backend name and dtype name from sample tensor data."""
    if sample_data is None:
        raise ValueError("Cannot infer backend: sample_data is None.")

    dtype_name = ar.get_dtype_name(sample_data)
    backend = ar.infer_backend(sample_data)
    return backend, dtype_name


def _build_to_numpy(sample_data, dtype_name, *, cast_complex_to_real=False):
    del sample_data
    if dtype_name not in _NUMPY_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype_name}' for numpy backend.")
    dtype = _NUMPY_DTYPE_MAP[dtype_name]

    def _to_numpy(x, dtype=dtype, cast_complex_to_real=cast_complex_to_real):
        arr = np.asarray(x)
        if cast_complex_to_real and np.issubdtype(dtype, np.floating) and np.iscomplexobj(arr):
            arr = arr.real
        target_dtype = dtype
        if (not cast_complex_to_real) and np.issubdtype(target_dtype, np.floating) and np.iscomplexobj(arr):
            # Preserve complex gate content when TN sample dtype is real.
            target_dtype = np.result_type(target_dtype, np.complex64)
        return np.asarray(arr, dtype=target_dtype)

    return _to_numpy


def _build_to_torch(sample_data, dtype_name, *, cast_complex_to_real=False):
    import torch  # pylint: disable=import-outside-toplevel

    dtype_map = {
        "complex128": torch.complex128,
        "complex64": torch.complex64,
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}' for torch backend.")

    dtype = getattr(sample_data, "dtype", None) or dtype_map[dtype_name]
    device = getattr(sample_data, "device", None)

    def _to_torch(
        x,
        dtype=dtype,
        device=device,
        cast_complex_to_real=cast_complex_to_real,
    ):
        if isinstance(x, torch.Tensor):
            arr = x
            if device is not None:
                arr = arr.to(device=device)
        else:
            if isinstance(x, np.ndarray) and (not x.flags.writeable):
                x = np.array(x, copy=True)
            kwargs = {"device": device} if device is not None else {}
            arr = torch.as_tensor(x, **kwargs)

        if cast_complex_to_real and dtype.is_floating_point and torch.is_complex(arr):
            arr = arr.real

        target_dtype = dtype
        if (not cast_complex_to_real) and target_dtype.is_floating_point and torch.is_complex(arr):
            target_dtype = torch.complex128 if target_dtype == torch.float64 else torch.complex64

        kwargs = {"dtype": target_dtype}
        if device is not None:
            kwargs["device"] = device
        return torch.as_tensor(arr, **kwargs)

    return _to_torch


def _build_to_cupy(sample_data, dtype_name, *, cast_complex_to_real=False):
    import cupy as cp  # pylint: disable=import-outside-toplevel

    dtype_map = {
        "complex128": cp.complex128,
        "complex64": cp.complex64,
        "float64": cp.float64,
        "float32": cp.float32,
        "float16": cp.float16,
        "int64": cp.int64,
        "int32": cp.int32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}' for cupy backend.")

    dtype = getattr(sample_data, "dtype", None) or dtype_map[dtype_name]
    device = getattr(sample_data, "device", None)

    def _to_cupy(
        x,
        dtype=dtype,
        device=device,
        cast_complex_to_real=cast_complex_to_real,
    ):
        if device is None:
            arr = cp.asarray(x)
        else:
            with device:
                arr = cp.asarray(x)

        if cast_complex_to_real and cp.issubdtype(dtype, cp.floating) and cp.iscomplexobj(arr):
            arr = arr.real

        target_dtype = dtype
        if (not cast_complex_to_real) and cp.issubdtype(target_dtype, cp.floating) and cp.iscomplexobj(arr):
            target_dtype = cp.result_type(target_dtype, cp.complex64)

        return arr.astype(target_dtype, copy=False)

    return _to_cupy


def _build_to_jax(sample_data, dtype_name, *, cast_complex_to_real=False):
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    dtype_map = {
        "complex128": jnp.complex128,
        "complex64": jnp.complex64,
        "float64": jnp.float64,
        "float32": jnp.float32,
        "float16": jnp.float16,
        "int64": jnp.int64,
        "int32": jnp.int32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}' for jax backend.")

    dtype = getattr(sample_data, "dtype", None) or dtype_map[dtype_name]
    device = getattr(sample_data, "device", None)

    def _to_jax(
        x,
        dtype=dtype,
        device=device,
        cast_complex_to_real=cast_complex_to_real,
    ):
        # Torch tensors need explicit host conversion before jnp.asarray.
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError:  # pragma: no cover - optional dependency
            torch = None
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        arr = jnp.asarray(x)

        if cast_complex_to_real and jnp.issubdtype(dtype, jnp.floating) and jnp.iscomplexobj(arr):
            arr = arr.real

        target_dtype = dtype
        if (not cast_complex_to_real) and jnp.issubdtype(target_dtype, jnp.floating) and jnp.iscomplexobj(arr):
            target_dtype = jnp.result_type(target_dtype, jnp.complex64)

        out = jnp.asarray(arr, dtype=target_dtype)
        if device is not None:
            out = jax.device_put(out, device)
        return out

    return _to_jax


def dispatch_backend_converter(
    *,
    backend,
    dtype_name,
    sample_data,
    cast_complex_to_real=False,
):
    """Return backend converter callable for the specified backend and dtype."""
    if sample_data is None:
        raise ValueError("Cannot infer backend: tensor network has no tensors.")

    if backend == "numpy":
        return _build_to_numpy(
            sample_data,
            dtype_name,
            cast_complex_to_real=cast_complex_to_real,
        )
    if backend == "torch":
        return _build_to_torch(
            sample_data,
            dtype_name,
            cast_complex_to_real=cast_complex_to_real,
        )
    if backend == "cupy":
        return _build_to_cupy(
            sample_data,
            dtype_name,
            cast_complex_to_real=cast_complex_to_real,
        )
    if backend == "jax":
        return _build_to_jax(
            sample_data,
            dtype_name,
            cast_complex_to_real=cast_complex_to_real,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def infer_backend_converter_from_sample(
    sample_data,
    *,
    cast_complex_to_real=False,
):
    """Infer and return converter callable from representative sample data."""
    if sample_data is None:
        return None

    backend, dtype_name = infer_backend_and_dtype(sample_data)
    try:
        return dispatch_backend_converter(
            backend=backend,
            dtype_name=dtype_name,
            sample_data=sample_data,
            cast_complex_to_real=cast_complex_to_real,
        )
    except ValueError:
        return None
