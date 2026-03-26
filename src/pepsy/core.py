"""Shared DMRG backend, optimizer, and fidelity helpers."""

import importlib.util
import warnings
from typing import Any

import numpy as np
import cotengra as ctg
import torch
import quimb.tensor as qtn

__all__ = [
    "backend_torch",
    "backend_numpy",
    "backend_jax",
    "register_torch_linalg",
    "register_jax_linalg",
    "set_default_array_backend",
    "get_default_array_backend",
    "set_default_grad_backend",
    "get_default_grad_backend",
    "reset_default_backends",
    "build_optimizer",
    "build_compressed_optimizer",
    "tn_fidelity",
]

_DEFAULT_ARRAY_BACKEND = None
_DEFAULT_GRAD_BACKEND = None


def _validate_backend_callable(name, fn):
    if fn is not None and not callable(fn):
        raise TypeError(f"{name} must be callable or None")


def set_default_array_backend(array_backend):
    """Set package-wide default array backend caster.

    Parameters
    ----------
    array_backend : callable | None
        Function mapping arrays to a target backend. ``None`` clears default.
    """
    _validate_backend_callable("array_backend", array_backend)
    global _DEFAULT_ARRAY_BACKEND  # pylint: disable=global-statement
    _DEFAULT_ARRAY_BACKEND = array_backend


def get_default_array_backend():
    """Return package-wide default array backend caster, or ``None``."""
    return _DEFAULT_ARRAY_BACKEND


def set_default_grad_backend(grad_backend):
    """Set package-wide default gradient backend caster.

    Parameters
    ----------
    grad_backend : callable | None
        Function mapping arrays to trainable backend tensors.
    """
    _validate_backend_callable("grad_backend", grad_backend)
    global _DEFAULT_GRAD_BACKEND  # pylint: disable=global-statement
    _DEFAULT_GRAD_BACKEND = grad_backend


def get_default_grad_backend():
    """Return package-wide default gradient backend caster, or ``None``."""
    return _DEFAULT_GRAD_BACKEND


def reset_default_backends():
    """Clear package-wide backend defaults."""
    global _DEFAULT_ARRAY_BACKEND  # pylint: disable=global-statement
    global _DEFAULT_GRAD_BACKEND  # pylint: disable=global-statement
    _DEFAULT_ARRAY_BACKEND = None
    _DEFAULT_GRAD_BACKEND = None

def backend_torch(device="cpu", dtype=None, requires_grad=False):

    def cast_array(x, device=device, dtype=dtype, requires_grad=requires_grad):

        if isinstance(x, torch.Tensor):
            out = x.detach() if requires_grad else x

            if dtype is None:
                out = out.to(device=device)
            else:
                out = out.to(device=device, dtype=dtype)

        else:
            if dtype is None:
                out = torch.as_tensor(x, device=device)
            else:
                out = torch.as_tensor(x, dtype=dtype, device=device)

        # Trainable tensors must be floating or complex
        if requires_grad and not (out.is_floating_point() or out.is_complex()):
            out = out.to(dtype=torch.float64)

        if requires_grad:
            out.requires_grad_(True)
        else:
            out.requires_grad_(False)

        return out

    return cast_array

def backend_numpy(dtype=np.float64):
    """Return a converter that materializes arrays as NumPy arrays."""

    def cast_array(x, dtype=dtype):
        return np.array(x, dtype=dtype)

    return cast_array


def backend_jax(dtype=None, device=None):
    """Return a converter that places arrays onto a specific JAX device."""
    try:
        import jax  # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "backend_jax requires optional dependencies 'jax' and 'jaxlib'. "
            "Install them with: pip install pepsy[jax]"
        ) from exc

    if dtype is None:
        dtype = jnp.float64
    if device is None:
        device = jax.devices("cpu")[0]

    def cast_array(x, dtype=dtype, device=device):
        return jax.device_put(jnp.array(x, dtype=dtype), device)

    return cast_array


def register_torch_linalg(mode="complex"):
    """Register custom torch linalg gradients in autoray.

    Parameters
    ----------
    mode : {"complex", "real"}, default="complex"
        Which SVD/QR registrations to install.
    """
    from . import linalg_registrations as lr  # pylint: disable=import-outside-toplevel

    if mode == "complex":
        lr.reg_complex_svd()
        lr.reg_complex_qr()
        return
    if mode == "real":
        lr.reg_real_svd()
        lr.reg_real_qr()
        return
    raise ValueError("mode must be 'complex' or 'real'")


def register_jax_linalg():
    """Register custom jax SVD gradient in autoray."""
    from . import linalg_registrations as lr  # pylint: disable=import-outside-toplevel

    lr.reg_complex_svd_jax()


def build_optimizer(
    progbar=True,
    alpha=32,
    target_size=2**34,
    subtree_size=12,
    max_time="rate:1e8",
    max_repeats=2**6,
    parallel=True,
    optlib="cmaes",
    directory="cash/",
    hash_method="b",
    seed=None,
):
    """Build and return a reusable cotengra contraction optimizer."""
    selected_optlib = optlib
    if selected_optlib == "cmaes" and importlib.util.find_spec("cmaes") is None:
        warnings.warn(
            "Package 'cmaes' not found. Falling back to optlib='random'.",
            RuntimeWarning,
        )
        selected_optlib = "random"
    opt = ctg.ReusableHyperOptimizer(
        minimize=f"combo-{alpha}",
        slicing_opts={"target_size": 2**40},
        slicing_reconf_opts={"target_size": target_size},
        reconf_opts={"subtree_size": subtree_size},
        max_repeats=max_repeats,
        parallel=parallel,
        optlib=selected_optlib,
        max_time=max_time,
        hash_method=hash_method,
        directory=directory,
        progbar=progbar,
        seed=seed,
    )
    return opt


def build_compressed_optimizer(
    progbar=True,
    chi=4,
    directory=None,
    max_repeats=2**8,
    max_time="rate:1e8",
    seed=None,
):
    """Build and return a reusable cotengra compressed optimizer."""
    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time=max_time,
        directory=directory,
        seed=seed,
    )
    return copt

def tn_fidelity(psi, psi_fix, seed=None):
    """Compute normalized MPS overlap fidelity."""
    opt: Any = build_optimizer(progbar=False, seed=seed)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt))
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ref = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1**2
    fidelity = complex(val_1 / (val_0 * val_ref)).real
    return fidelity

def add_cycle(peps, bond_dim, cylinder=False):
    Ly = peps.Ly
    Lx = peps.Lx
    for j in range(Ly):
        T1 = peps[f"I{Lx-1},{j}"]
        T2 = peps[f"I{0},{j}"]
        qtn.new_bond(T1, T2, size=bond_dim, name=None, axis1=0, axis2=0)

    if not cylinder:
        for i in range(Lx):
            T1 = peps[f"I{i},{Ly-1}"]
            T2 = peps[f"I{i},{0}"]
            qtn.new_bond(T1, T2, size=bond_dim, name=None, axis1=0, axis2=0)
    return peps
