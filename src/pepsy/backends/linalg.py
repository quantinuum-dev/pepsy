"""Lazy compatibility facade for the legacy linear-algebra module.

The implementation now lives in backend-specific modules.  Keeping this
facade avoids importing JAX, SciPy, and Torch merely because an older caller
still imports ``pepsy.backends.linalg``.
"""

from importlib import import_module


_TORCH_NAMES = {
    "safe_inverse",
    "safe_inverse_2",
    "SVD",
    "SVD_real",
    "QR_real",
    "QR_complex",
    "reg_native_svd_torch",
    "reg_rel_svd_torch",
    "reg_complex_svd_torch",
    "reg_real_svd_torch",
    "reg_real_qr_torch",
    "reg_complex_qr_torch",
    "reg_quimb_torch_split_drivers",
    "reset_quimb_torch_split_drivers",
    "reset_torch_linalg_registrations",
    "reg_stop_gradient_torch",
    "stop_grad",
}
_JAX_NAMES = {
    "svd_jax",
    "h",
    "jaxsvd_fwd",
    "jaxsvd_bwd",
    "reg_native_svd_jax",
    "reg_complex_svd_jax",
    "reg_rel_svd_jax",
    "reg_real_svd_jax",
}

__all__ = sorted(_TORCH_NAMES | _JAX_NAMES)


def __getattr__(name):
    if name in _TORCH_NAMES:
        value = getattr(import_module(".linalg_torch", __package__), name)
    elif name in _JAX_NAMES:
        value = getattr(import_module(".linalg_jax", __package__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
