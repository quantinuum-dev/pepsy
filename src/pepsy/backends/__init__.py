"""Backend selection, conversion, and linear algebra helpers."""

from importlib import import_module

from .convert import (
    dispatch_backend_converter,
    infer_backend_and_dtype,
    infer_backend_converter_from_sample,
    resolve_backend_sample_data,
    resolve_backend_sample_data_from_tn,
)
from ..tensors.core import (
    backend_cupy,
    backend_jax,
    backend_numpy,
    backend_torch,
    get_default_array_backend,
    get_default_grad_backend,
    register_torch_linalg,
    reset_default_backends,
    set_default_array_backend,
    set_default_grad_backend,
)

__all__ = [
    "backend_cupy",
    "backend_jax",
    "backend_numpy",
    "backend_torch",
    "dispatch_backend_converter",
    "infer_backend_and_dtype",
    "infer_backend_converter_from_sample",
    "resolve_backend_sample_data",
    "resolve_backend_sample_data_from_tn",
    "get_default_array_backend",
    "get_default_grad_backend",
    "register_torch_linalg",
    "reset_default_backends",
    "set_default_array_backend",
    "set_default_grad_backend",
]


def __getattr__(name):
    if name in {"config", "convert", "linalg", "linalg_jax", "linalg_torch"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
