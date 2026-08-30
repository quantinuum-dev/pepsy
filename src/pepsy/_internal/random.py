"""Backend-native random-array helpers with an old-Autoray fallback."""

from __future__ import annotations

import autoray as ar
import numpy as np

__all__ = ["backend_random_array"]


def _fallback_dtype(dtype):
    """Choose a NumPy staging dtype only for old Autoray fallbacks."""
    name = str(dtype).lower()
    if "complex64" in name:
        return np.complex64
    if "complex" in name:
        return np.complex128
    if "float32" in name:
        return np.float32
    return np.float64


def _fallback_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if hasattr(rng, "normal"):
        return rng
    return np.random.default_rng(rng)


def backend_random_array(shape, *, like, dtype=None, scale=1.0, rng=None):
    """Draw a normal array on the same backend, device, and dtype as ``like``.

    Autoray 0.10's ``random.array`` is used directly, including its backend
    generator handling. Older Autoray releases do not expose that operation,
    so the deterministic NumPy fallback is converted through ``like``.
    """
    shape = tuple(int(size) for size in shape)
    try:
        ar.get_lib_fn(ar.infer_backend(like), "random.array")
    except (AttributeError, ImportError, KeyError, LookupError):
        random_source = _fallback_rng(rng)
        random_data = random_source.normal(size=shape)
        fallback_dtype = _fallback_dtype(dtype or getattr(like, "dtype", None))
        if np.issubdtype(fallback_dtype, np.complexfloating):
            random_data = random_data + 1j * random_source.normal(size=shape)
        random_data = (float(scale) * random_data).astype(fallback_dtype)
        return ar.do("array", random_data, like=like)

    return ar.do(
        "random.array",
        shape,
        dist="normal",
        scale=scale,
        dtype=dtype,
        like=like,
        rng=rng,
    )
