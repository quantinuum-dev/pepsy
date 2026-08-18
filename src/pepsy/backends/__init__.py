"""Lazy backend selection, conversion, and linear algebra helpers.

Importing the namespace itself stays NumPy/Autoray-only. Torch, JAX, CuPy,
and SciPy are loaded when a backend-specific symbol is actually requested.
"""

from importlib import import_module


_SYMBOL_MODULES = {
    "backend_infer": ".convert",
    "dispatch_backend_converter": ".convert",
    "infer_backend_and_dtype": ".convert",
    "infer_backend_signature": ".convert",
    "infer_backend_converter_from_sample": ".convert",
    "resolve_backend_sample_data": ".convert",
    "resolve_backend_sample_data_from_tn": ".convert",
    "to_float": ".convert",
    "backend_cupy": ".config",
    "backend_jax": ".config",
    "backend_numpy": ".config",
    "backend_torch": ".config",
    "build_backend": ".config",
    "get_default_array_backend": ".config",
    "get_default_grad_backend": ".config",
    "get_torch_linalg_config": ".config",
    "register_jax_linalg": ".config",
    "register_torch_linalg": ".config",
    "reset_linalg_registrations": ".config",
    "reset_default_backends": ".config",
    "set_default_array_backend": ".config",
    "set_default_grad_backend": ".config",
    "TorchLinalgConfig": ".config",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in {"config", "convert", "linalg", "linalg_jax", "linalg_torch"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
