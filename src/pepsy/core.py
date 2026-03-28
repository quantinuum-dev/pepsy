"""Shared DMRG backend, optimizer, and fidelity helpers."""

import importlib.util
import math
import warnings
from typing import Any

import numpy as np
import cotengra as ctg
import torch
import quimb.tensor as qtn

from ._tn_validation import _PHYS_OUTER, validate_tensor_network_tags

__all__ = [
    "backend_torch",
    "backend_numpy",
    "backend_cupy",
    "register_torch_linalg",
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
    """Return a converter that materializes arrays as torch tensors."""

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


def backend_cupy(device=None, dtype=None):
    """Return a converter that materializes arrays as CuPy arrays.

    Parameters
    ----------
    device : int | cupy.cuda.Device | None, optional
        Target CUDA device. If ``None``, use CuPy's current device.
    dtype : dtype-like | None, optional
        Target CuPy dtype. If ``None``, infer from input.
    """
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend_cupy requires optional dependency 'cupy'. "
            "Install it with: pip install cupy-cuda12x (or your CUDA variant)."
        ) from exc

    target_device = device
    if isinstance(target_device, int):
        target_device = cp.cuda.Device(target_device)

    def cast_array(x, device=target_device, dtype=dtype):
        if device is None:
            return cp.asarray(x, dtype=dtype)
        with device:
            return cp.asarray(x, dtype=dtype)

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
        lr.reg_complex_svd_torch()
        lr.reg_complex_qr_torch()
        return
    if mode == "real":
        lr.reg_real_svd_torch()
        lr.reg_real_qr_torch()
        return
    raise ValueError("mode must be 'complex' or 'real'")


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
        minimize=f"combo-{int(alpha)}",
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
        on_trial_error="ignore",
    )
    return opt


def build_compressed_optimizer(
    progbar=True,
    chi=4,
    directory=None,
    max_repeats=2**8,
    max_time="rate:1e8",
):
    """Build and return a reusable cotengra compressed optimizer."""
    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time=max_time,
        directory=directory,
    )
    return copt


def tn_fidelity(psi, psi_fix):
    """Compute normalized MPS overlap fidelity."""
    opt: Any = build_optimizer(progbar=False)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt))
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ref = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1**2
    fidelity = complex(val_1 / (val_0 * val_ref)).real
    return fidelity


def add_cycle(peps, bond_dim, cylinder=False):
    """Add periodic bonds to a PEPS network in x (and optional y) directions."""
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


def tn_applied(p, pepo):
    r"""Apply a PEPO operator to a PEPS ket: :math:`\hat{O}|\psi\rangle`.

    The PEPO ``k``-indices contract with the PEPS ``k``-indices on join.
    The PEPO ``b``-indices (output legs) are renamed to ``k``-indices so
    the result has the same physical index convention as a standard PEPS.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state :math:`|\psi\rangle`.  Outer indices must follow
        the ``k<int>[,<int>...]`` convention.
    pepo : qtn.TensorNetwork
        PEPO operator :math:`\hat{O}`.  Outer indices must follow the
        ``k<int>[,<int>...]`` and ``b<int>[,<int>...]`` convention.

    Returns
    -------
    qtn.TensorNetwork
        The resulting network :math:`\hat{O}|\psi\rangle` with ``k``-type
        physical indices.
    """
    # Validate lattice tags
    validate_tensor_network_tags(p)
    validate_tensor_network_tags(pepo)

    # Validate PEPS physical indices
    bad_p = [i for i in p.outer_inds() if not _PHYS_OUTER.fullmatch(i)]
    if bad_p:
        sample = ", ".join(sorted(bad_p)[:8])
        warnings.warn(
            f"PEPS outer indices expected format k/b<int>[,<int>...]. "
            f"Found non-matching: {sample}",
            stacklevel=2,
        )

    # Validate PEPO physical indices
    bad_pepo = [i for i in pepo.outer_inds() if not _PHYS_OUTER.fullmatch(i)]
    if bad_pepo:
        sample = ", ".join(sorted(bad_pepo)[:8])
        warnings.warn(
            f"PEPO outer indices expected format k/b<int>[,<int>...]. "
            f"Found non-matching: {sample}",
            stacklevel=2,
        )

    tn = p & pepo
    # The PEPS k-indices are now inner (contracted with PEPO k-indices).
    # Randomize them so they won't collide when we rename b -> k.
    contracted_k = {
        idx: qtn.rand_uuid()
        for idx in tn.inner_inds()
    }
    if contracted_k:
        tn.reindex_(contracted_k)
    # Rename PEPO output b-indices -> k-indices (physical convention)
    b_to_k = {
        idx: f"k{idx[1:]}"
        for idx in tn.outer_inds()
        if idx.startswith("b")
    }
    if b_to_k:
        tn.reindex_(b_to_k)
    return tn



def product_state_peps(lx: int, ly: int, dtype: str = "complex128", theta: float = 0.0):
    """Create a bond-dimension-1 product-state PEPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    lx : int
        Lattice size in x direction.
    ly : int
        Lattice size in y direction.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.

    Returns
    -------
    quimb.tensor.PEPS
        Initialized PEPS with bond dimension 1.
    """
    peps = qtn.PEPS.rand(Lx=lx, Ly=ly, bond_dim=1, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for tensor in peps:
        if len(tensor.data.shape) == 3:
            data = np.zeros([1, 1, 2], dtype=dtype)
            data[0, 0, :] = local_vec
            tensor.modify(data=data)
        if len(tensor.data.shape) == 4:
            data = np.zeros([1, 1, 1, 2], dtype=dtype)
            data[0, 0, 0, :] = local_vec
            tensor.modify(data=data)
        if len(tensor.data.shape) == 5:
            data = np.zeros([1, 1, 1, 1, 2], dtype=dtype)
            data[0, 0, 0, 0, :] = local_vec
            tensor.modify(data=data)
    peps.astype_(dtype)
    return peps
