"""Backend configuration facade."""

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
    "get_default_array_backend",
    "get_default_grad_backend",
    "register_torch_linalg",
    "reset_default_backends",
    "set_default_array_backend",
    "set_default_grad_backend",
]
