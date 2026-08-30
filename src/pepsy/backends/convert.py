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

    if shape is not None:
        try:
            obj = ar.to_numpy(value)
        except Exception:
            # Keep supporting duck-typed scalar wrappers with ``item`` but no
            # registered Autoray backend.
            obj = value
    else:
        obj = value

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
    scalar-like backend tensor. Autoray converts backend scalar arrays to host
    NumPy before extracting ``.item()``. Non-scalar arrays raise ``TypeError``.

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


def _is_symmray_array(value):
    """Return whether ``value`` is a Symmray block-sparse array."""
    return (
        type(value).__module__.split(".", 1)[0] == "symmray"
        or hasattr(value, "blocks") and hasattr(value, "indices")
    )


def _symmray_block_signatures(value):
    """Return backend signatures for the raw arrays held by a Symmray value."""
    blocks = getattr(value, "blocks", None)
    if not blocks:
        return ()
    signatures = []
    for block in blocks.values():
        backend, dtype = infer_backend_and_dtype(block)
        device = getattr(block, "device", None)
        signatures.append(
            (backend, str(dtype), None if device is None else str(device))
        )
    return tuple(signatures)


def infer_backend_signature(sample_data):
    """Infer comparable backend, dtype, device, and Symmray-block metadata.

    Dense arrays return the traditional ``(backend, dtype, device)`` tuple.
    Symmray arrays return ``(symmray, dtype, device, block_backend)`` where
    ``block_backend`` is the backend of their raw charge-sector blocks.  The
    extra field is essential: ``ar.infer_backend`` intentionally reports the
    structured Symmray container rather than the Torch/CuPy backend used by
    its blocks.
    """
    if sample_data is None:
        raise ValueError("Cannot infer backend: sample_data is None.")

    try:
        backend, dtype = infer_backend_and_dtype(sample_data)
    except (AttributeError, KeyError, TypeError, ValueError):
        # Untyped Python sequences are convenience inputs. Treat them as
        # non-backend data so callers can materialize them on the state
        # backend without emitting a transfer warning.
        try:
            array = np.asarray(sample_data)
            dtype = ar.get_dtype_name(array)
        except (TypeError, ValueError, KeyError) as exc:
            raise TypeError(
                "Could not infer a backend or dtype from the supplied array."
            ) from exc
        return "builtins", str(dtype), None
    device = getattr(sample_data, "device", None)
    device = None if device is None else str(device)
    if backend != "symmray" and not _is_symmray_array(sample_data):
        return backend, str(dtype), device

    block_signatures = _symmray_block_signatures(sample_data)
    if block_signatures:
        unique = set(block_signatures)
        if len(unique) != 1:
            raise TypeError(
                "Symmray arrays must use one underlying backend, dtype, and "
                f"device; found {sorted(unique)!r}."
            )
        block_backend, block_dtype, block_device = block_signatures[0]
        # The raw block dtype/device is authoritative for a structured array.
        dtype = block_dtype
        device = block_device
    else:
        block_backend = str(getattr(sample_data, "backend", "symmray"))
    return "symmray", str(dtype), device, str(block_backend)


def backend_signatures_compatible(source_signature, target_signature):
    """Return whether payloads can replay without implicit conversion.

    NumPy can safely promote mixed dtypes during its contractions, so dtype is
    ignored only for dense NumPy-to-NumPy payloads. Other array backends, and
    Symmray charge blocks, require matching dtypes for direct replay.
    """
    return (
        source_signature[0] == target_signature[0]
        and (
            source_signature[0] == "numpy"
            or source_signature[1] == target_signature[1]
        )
        and source_signature[2] == target_signature[2]
        and source_signature[3:] == target_signature[3:]
    )


def _backend_data_values(value):
    """Return array payloads from an array, tensor, or tensor network."""
    tensor_map = getattr(value, "tensor_map", None)
    if tensor_map is not None:
        values = tuple(
            getattr(tensor, "data", None) for tensor in tensor_map.values()
        )
        return tuple(data for data in values if data is not None)

    tensors = getattr(value, "tensors", None)
    if tensors is not None and not hasattr(value, "shape"):
        values = tuple(getattr(tensor, "data", None) for tensor in tensors)
        return tuple(data for data in values if data is not None)

    data = getattr(value, "data", None)
    if data is not None and hasattr(data, "shape") and hasattr(data, "dtype"):
        return (data,)
    return (value,)


def backend_infer(value):
    """Infer and validate backend metadata from an array or tensor network.

    Parameters
    ----------
    value
        An array-like payload, Quimb tensor, or tensor network such as an MPS
        or :class:`TreeTensorNetwork`. For a tensor network, every tensor is
        checked for one common backend, dtype, and device.

    Returns
    -------
    dict
        The normalized metadata mapping ``backend``, ``dtype``, and
        ``device``. Native Symmray arrays additionally include
        ``array_backend`` for their underlying NumPy, Torch, or CuPy blocks.
    """
    values = _backend_data_values(value)
    if not values:
        raise ValueError("Cannot infer backend: value contains no tensors.")

    signatures = tuple(infer_backend_signature(data) for data in values)
    signature = signatures[0]
    mismatched = tuple(candidate for candidate in signatures[1:] if candidate != signature)
    if mismatched:
        raise TypeError(
            "Backend arrays must use one compatible backend, dtype, and "
            f"device; found {signature!r} and {mismatched[0]!r}."
        )

    backend, dtype, device = signature[:3]
    info = {"backend": backend, "dtype": dtype, "device": device}
    if len(signature) > 3:
        info["array_backend"] = signature[3]
    return info


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
        arr = np.asarray(ar.to_numpy(x))
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
        try:
            # JAX accepts NumPy inputs directly, while Autoray provides the
            # explicit host boundary for Torch/CuPy and other array backends.
            if ar.infer_backend(x) != "jax":
                x = ar.to_numpy(x)
        except Exception:
            # Preserve support for custom array-likes which JAX can consume
            # even though Autoray cannot infer their namespace.
            pass

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
    if backend == "symmray" or _is_symmray_array(sample_data):
        blocks = getattr(sample_data, "blocks", None) or {}
        block_sample = next(iter(blocks.values()), None)
        if block_sample is None:
            return None
        block_converter = infer_backend_converter_from_sample(
            block_sample,
            cast_complex_to_real=cast_complex_to_real,
        )
        if block_converter is None:
            return None

        def _to_symmray_or_block(value):
            # Network-level ``apply_to_arrays`` callbacks may receive raw
            # sector blocks, while public payload conversion receives the
            # structured Symmray object. Support both call sites.
            if _is_symmray_array(value):
                converted = value.copy()
                converted.apply_to_arrays(block_converter)
                return converted
            return block_converter(value)

        return _to_symmray_or_block
    try:
        return dispatch_backend_converter(
            backend=backend,
            dtype_name=dtype_name,
            sample_data=sample_data,
            cast_complex_to_real=cast_complex_to_real,
        )
    except ValueError:
        return None
