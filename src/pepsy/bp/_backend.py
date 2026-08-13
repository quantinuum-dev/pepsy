"""Small Autoray helpers for backend-native BP compression.

Tensor-shaped values stay in their originating backend, including native
Symmray block-sparse arrays. Scalar diagnostics alone are converted to Python
values. Dense materialization remains an explicit caller choice.
"""

from __future__ import annotations

import numpy as np

import autoray as ar

from ._symmray import is_symmray_array


def native(value):
    """Return an array-like value without converting its backend."""
    if hasattr(value, "data") and not hasattr(value, "shape"):
        value = value.data
    return value


def copy(value):
    """Copy an array through its native Autoray backend."""
    value = native(value)
    if is_symmray_array(value):
        return value.copy()
    return ar.do("copy", value)


def backend(value) -> str:
    """Return the native Autoray backend name."""
    return ar.infer_backend(native(value))


def dtype_name(value) -> str:
    """Return the backend-independent dtype name."""
    return ar.get_dtype_name(native(value))


def eye(size: int, *, like):
    """Construct a native identity matching ``like``."""
    return ar.do("eye", size, like=native(like))


def _shape_tuple(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


def zeros(shape, *, like):
    """Construct native zeros matching ``like``."""
    return ar.do("zeros", _shape_tuple(shape), like=native(like))


def ones(shape, *, like):
    """Construct native ones matching ``like``."""
    return ar.do("ones", _shape_tuple(shape), like=native(like))


def array(value, *, like):
    """Convert a small initializer into the backend of ``like``."""
    result = ar.do("array", value, like=native(like))
    return ar.astype(result, dtype_name(like))


def cast_like(value, like):
    """Cast an existing value to the backend and dtype of ``like``."""
    value = native(value)
    like = native(like)
    if is_symmray_array(like):
        if not is_symmray_array(value):
            raise TypeError(
                "native Symmray compression requires native boundary data; "
                "dense closures cannot be implicitly charge-lifted"
            )
        result = value.copy()
        if dtype_name(value) != dtype_name(like) and hasattr(
            result, "apply_to_arrays"
        ):
            result.apply_to_arrays(
                lambda block: ar.astype(block, dtype_name(like))
            )
        return result
    if ar.infer_backend(value) == ar.infer_backend(like):
        return ar.astype(value, dtype_name(like))
    result = ar.do("array", value, like=like)
    return ar.astype(result, dtype_name(like))


def conj(value):
    return ar.do("conj", native(value))


def dag(value):
    """Return the conjugate transpose of a rank-two native array."""
    value = conj(value)
    return ar.do("transpose", value, axes=(1, 0))


def transpose(value, axes):
    return ar.do("transpose", native(value), axes=tuple(axes))


def reshape(value, shape):
    return ar.do("reshape", native(value), tuple(shape))


def einsum(subscripts: str, *operands):
    """Backend-native einsum without NumPy-only ``optimize=`` kwargs."""
    return ar.do(
        "einsum",
        subscripts,
        *(native(operand) for operand in operands),
    )


def scalar_float(value) -> float:
    """Convert a scalar diagnostic to Python ``float``."""
    return float(np.asarray(ar.to_numpy(value)))


def scalar_int(value) -> int:
    """Convert a scalar diagnostic to Python ``int``."""
    return int(np.asarray(ar.to_numpy(value)))


def scalar_bool(value) -> bool:
    """Convert a scalar diagnostic to Python ``bool``."""
    return bool(np.asarray(ar.to_numpy(value)))


def real(value):
    return ar.do("real", native(value))


def abs(value):
    return ar.do("abs", native(value))


def all_finite(value) -> bool:
    return scalar_bool(ar.do("all", ar.do("isfinite", native(value))))


def normalize_message_pairs(messages, ind_map):
    """Normalize copied directed BP message pairs with Quimb's convention."""
    from quimb.tensor.belief_propagation.bp_common import (
        normalize_message_pair,
    )

    normalized = dict(messages)
    for index, endpoints in ind_map.items():
        endpoints = tuple(endpoints)
        if len(endpoints) != 2:
            continue
        left_key = (index, endpoints[0])
        right_key = (index, endpoints[1])
        if left_key not in normalized or right_key not in normalized:
            continue
        left = normalized[left_key]
        right = normalized[right_key]
        left_shape = left.shape
        right_shape = right.shape
        left, right = normalize_message_pair(
            ar.do("reshape", left, (-1,)),
            ar.do("reshape", right, (-1,)),
        )
        normalized[left_key] = ar.do("reshape", left, left_shape)
        normalized[right_key] = ar.do("reshape", right, right_shape)
    return normalized


def is_complex(value) -> bool:
    return dtype_name(value).startswith("complex")


def lstsq_solution(result):
    """Extract the solution from NumPy/Torch/JAX ``lstsq`` results."""
    return result.solution if hasattr(result, "solution") else result[0]
