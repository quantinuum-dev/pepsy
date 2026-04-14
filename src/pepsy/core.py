"""Shared DMRG backend, optimizer, and fidelity helpers."""

import math
import os
import warnings
from typing import Any

import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

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
    "contract_hypercompressed_tn",
    "tn_fidelity",
    "tn_norm",
    "tns_align",
    "expec_tn_1d",
    "ps_to_peps",
    "ps_to_mps",
    "random_haar_qubit",
    "hrps_to_peps",
    "hrps_to_mps",
    "pepo_identity",
    "add_cycle",
]

_DEFAULT_ARRAY_BACKEND = None
_DEFAULT_GRAD_BACKEND = None



def _patch_unhashable_device_namespace_key():
    """Patch autoray namespace cache keys for unhashable backend device objects."""
    try:
        import autoray  # pylint: disable=import-outside-toplevel
        import autoray.autoray as ar_core  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return

    if getattr(ar_core, "_pepsy_unhashable_device_patch", False):
        return

    original_get_namespace = ar_core.get_namespace

    def _safe_get_namespace(like=None, device=None, dtype=None, submodule=None):
        if (device is None) and (like is not None) and (not isinstance(like, str)):
            try:
                device = like.device
            except AttributeError:
                device = None

        if device is not None:
            try:
                hash(device)
            except TypeError:
                dev_id = getattr(device, "id", None)
                device = f"device:{dev_id}" if dev_id is not None else str(device)

        return original_get_namespace(
            like=like,
            device=device,
            dtype=dtype,
            submodule=submodule,
        )

    ar_core.get_namespace = _safe_get_namespace
    autoray.get_namespace = _safe_get_namespace

    try:
        import quimb.tensor.decomp as qtn_decomp  # pylint: disable=import-outside-toplevel

        qtn_decomp.get_namespace = _safe_get_namespace
    except Exception:  # pragma: no cover
        pass

    ar_core._pepsy_unhashable_device_patch = True


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
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "backend_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )

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

    _patch_unhashable_device_namespace_key()

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
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "register_torch_linalg requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
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
    progbar: bool = False,
    alpha: int = 64,
    max_time="rate:7e8", 
    max_repeats: int = 128,
    parallel="auto",
    optlib: str = "cmaes",
    directory=False,
    hash_method: str = "b",
    overwrite=False,
    on_trial_error: str = "warn",
    reconf_opts: dict | None = None,
    slicing_reconf_opts: dict | None = None,
):
    """Build a reusable cotengra contraction optimizer.

    Parameters
    ----------
    progbar : bool, optional
        Whether to show optimizer progress.
    alpha : int, optional
        Weight for the combo objective.
    max_time : str | float | None, optional
        Search budget for the hyper-optimizer.
    max_repeats : int, optional
        Maximum number of optimization trials.
    parallel : bool | str, optional
        Parallel search setting passed to cotengra.
    optlib : str, optional
        Backend optimizer library.
    directory : None | bool | str, optional
        Cache directory for reusable contraction trees.
    hash_method : str, optional
        Hashing method for reusable contraction lookup.
    overwrite : bool | str, optional
        Cache overwrite behavior.
    on_trial_error : str, optional
        How to handle individual trial failures.
    reconf_opts : dict | None, optional
        Options for subtree reconfiguration.
    slicing_reconf_opts : dict | None, optional
        Options for interleaved slicing and reconfiguration.
    """
    # cotengra expects directory to be str, True, or None — not False.
    if directory is False:
        directory = None

    kwargs = dict(
        minimize=f"combo-{int(alpha)}",
        max_time=max_time,
        max_repeats=max_repeats,
        parallel=parallel,
        optlib=optlib,
        directory=directory,
        hash_method=hash_method,
        overwrite=overwrite,
        progbar=progbar,
        on_trial_error=on_trial_error,
    )

    if reconf_opts is not None:
        kwargs["reconf_opts"] = reconf_opts

    if slicing_reconf_opts is not None:
        kwargs["slicing_reconf_opts"] = slicing_reconf_opts

    return ctg.ReusableHyperOptimizer(**kwargs)

def build_compressed_optimizer(
    progbar=True,
    chi=4,
    directory=None,
    max_repeats=2**8,
    max_time="rate:1e7",
):
    """Build and return a reusable cotengra compressed optimizer.

    Parameters
    ----------
    directory : None, True, or str, optional
        Passed directly to cotengra. ``None`` disables caching; ``True``
        auto-generates a directory in the current working directory.
    """
    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time=max_time,
        directory=directory,
    )
    return copt


def contract_hypercompressed_tn(
    tn,
    copt=None,
    max_bond=None,
    *,
    chi=None,
    output_inds=None,
    tree_gauge_distance=4,
    progbar=False,
    cutoff=1.0e-12,
    equalize_norms=False,
    inplace=False,
):
    """Contract a generic tensor network with compressed hyper-optimization.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Tensor network to compress-contract.
    copt : object, optional
        Reusable compressed cotengra optimizer. If ``None``, one is built
        with :func:`build_compressed_optimizer` using ``chi``.
    max_bond : int | None, optional
        Maximum retained bond dimension during compressed contraction.
        If ``None``, defaults to ``chi``.
    chi : int | None, optional
        Bond dimension used to build ``copt`` when ``copt`` is ``None``.
        Required if both ``copt`` and ``max_bond`` are missing.
    output_inds : sequence[str] | None, optional
        Output indices to preserve during contraction.
    tree_gauge_distance : int, optional
        Gauge distance passed to ``contract_compressed_``.
    progbar : bool, optional
        Whether to show progress during compressed contraction.
    cutoff : float, optional
        Truncation cutoff passed to ``contract_compressed_``.
    equalize_norms : bool | float, optional
        Norm equalization option passed to ``contract_compressed_``.
    inplace : bool, optional
        If ``True``, mutate ``tn`` directly. Otherwise, contract a copy.

    Returns
    -------
    qtn.TensorNetwork
        The compressed-contracted tensor network.
    """
    if max_bond is None:
        max_bond = chi

    if copt is None:
        if chi is None:
            raise ValueError(
                "When `copt` is not provided, please provide `chi` "
                "to build a compressed optimizer."
            )
        copt = build_compressed_optimizer(progbar=progbar, chi=chi)

    if max_bond is None:
        raise ValueError("Please provide `max_bond` (or `chi`) for compressed contraction.")

    tn_out = tn if inplace else tn.copy()
    tn_out.full_simplify_(seq="R", split_method="svd", inplace=True)
    tree = tn_out.contraction_tree(copt)
    tn_out.contract_compressed_(
        optimize=tree,
        output_inds=output_inds,
        max_bond=max_bond,
        tree_gauge_distance=tree_gauge_distance,
        equalize_norms=equalize_norms,
        cutoff=cutoff,
        progbar=progbar,
    )
    return tn_out


def tn_norm(
    psi,
    *,
    contraction_opt: Any | None = None,
):
    """Compute the norm of a tensor network state.

    Parameters
    ----------
    psi : qtn.TensorNetwork
        State whose norm is computed.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.

    Returns
    -------
    float
        ``|<psi|psi>|``.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    return abs((psi.H & psi).contract(all, optimize=contraction_opt))


def tn_fidelity(
    psi,
    psi_fix,
    *,
    opt: Any | None = None,
    contraction_opt: Any | None = None,
):
    """Compute normalized overlap fidelity.

    Parameters
    ----------
    psi : qtn.TensorNetwork
        Trial state.
    psi_fix : qtn.TensorNetwork
        Reference state.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.
    """
    if contraction_opt is None:
        contraction_opt = opt

    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)

    val_0 = abs((psi.H & psi).contract(all, optimize=contraction_opt))
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=contraction_opt))
    val_ref = abs((psi_fix.H & psi_fix).contract(all, optimize=contraction_opt))

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


def pepo_identity(lx, ly, dtype="complex128"):
    """Create bond-dimension-1 PEPO identity on an ``lx x ly`` lattice."""
    pepo = qtn.PEPO.rand(Lx=lx, Ly=ly, bond_dim=1, seed=666, dtype=dtype)
    eye = np.eye(2, dtype=dtype)

    for tensor in pepo:
        ndim = len(tensor.data.shape)
        if ndim == 4:
            data = np.zeros([1, 1, 2, 2], dtype=dtype)
            data[0, 0, :, :] = eye
            tensor.modify(data=data)
        elif ndim == 5:
            data = np.zeros([1, 1, 1, 2, 2], dtype=dtype)
            data[0, 0, 0, :, :] = eye
            tensor.modify(data=data)
        elif ndim == 6:
            data = np.zeros([1, 1, 1, 1, 2, 2], dtype=dtype)
            data[0, 0, 0, 0, :, :] = eye
            tensor.modify(data=data)

    return pepo


def tns_align(p, pepo):
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

    tn = p & pepo
    # Only randomize the physical k-indices (shared between p and pepo).
    # Virtual bond indices must NOT be renamed — they must stay stable so
    # the Y-cut outer indices of the double-layer TN match the stored
    # boundary MPS across repeated calls to _prepare_current_double_layers.
    # Use non-mutating reindex to avoid modifying the original p/pepo tensors
    # (quimb's & shares tensor objects, so reindex_ would mutate the originals).
    contracted_k = {
        idx: qtn.rand_uuid()
        for idx in tn.inner_inds()
        if isinstance(idx, str) and idx.startswith("k")
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



def expec_tn_1d(mpo, mps, *, contraction_opt=None):
    """Compute normalized 1D expectation value ``<mps|mpo|mps> / <mps|mps>``."""
    if contraction_opt is None:
        contraction_opt = "auto-hq"

    mps_n = mps.copy()
    mps_n.normalize()
    mps_h = mps_n.H
    mps_h.reindex_({f"k{i}": f"b{i}" for i in range(mps_n.L)})
    mpo_t = mpo * 1.0
    return (mps_h | mpo_t | mps_n).contract(all, optimize=contraction_opt)


def ps_to_peps(Lx: int, Ly: int, dtype: str = "complex128", theta: float = 0.0, cyclic: bool = False):
    """Create a bond-dimension-1 product-state PEPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool, optional
        If True, add periodic bonds (bond dimension 1) via :func:`add_cycle`.

    Returns
    -------
    quimb.tensor.PEPS
        Initialized PEPS with bond dimension 1.
    """
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for x in range(Lx):
        for y in range(Ly):
            tensor = peps[x, y]
            phys_ind = peps.site_ind(x, y)
            phys_axis = tensor.inds.index(phys_ind)
            data = np.zeros_like(tensor.data, dtype=dtype)

            slicer = [0] * data.ndim
            slicer[phys_axis] = slice(None)
            data[tuple(slicer)] = local_vec
            tensor.modify(data=data)
    peps.astype_(dtype)
    if cyclic:
        peps = add_cycle(peps, bond_dim=1)
    return peps


def ps_to_mps(L: int, dtype: str = "complex128", theta: float = 0.0, cyclic: bool = False):
    """Create a bond-dimension-1 product-state MPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    L : int
        Number of sites.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool, optional
        If True, create a periodic MPS with bond dimension 1.

    Returns
    -------
    quimb.tensor.MatrixProductState
        Initialized MPS with bond dimension 1.
    """
    mps = qtn.MPS_rand_state(L=L, bond_dim=1, phys_dim=2, cyclic=cyclic, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)

    for i in range(L):
        tensor = mps[i]
        phys_ind = mps.site_ind(i)
        phys_axis = tensor.inds.index(phys_ind)
        data = np.zeros_like(tensor.data, dtype=dtype)

        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    mps.astype_(dtype)
    return mps


def random_haar_qubit(seed=None, perturb=0.0):
    """Generate one random single-qubit Haar sample as ``(theta, phi)``.

    Parameters
    ----------
    seed : int | None, optional
        If set, produce a deterministic sample.
    perturb : float, optional
        Additive offset applied to both sampled parameters.

    Returns
    -------
    tuple[float, float]
        ``(theta, phi)`` Bloch angles.
    """
    rng = np.random.default_rng(seed)
    phi = 2 * np.pi * rng.random() + perturb
    z = 2 * rng.random() - 1 + perturb
    z = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z)
    return float(theta), float(phi)


def hrps_to_peps(
    Lx: int,
    Ly: int,
    dtype: str = "complex128",
    cyclic: bool = False,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
):
    """Create a bond-dimension-1 PEPS with per-site single-qubit Haar states.

    If ``haar_params`` is omitted, each site uses :func:`random_haar_qubit`.
    With ``seed`` set, site ``k`` uses ``seed + k`` for reproducible but
    distinct samples.
    """
    peps = ps_to_peps(Lx=Lx, Ly=Ly, dtype=dtype, theta=0.0, cyclic=cyclic)

    n_sites = Lx * Ly
    if haar_params is not None:
        if len(haar_params) != n_sites:
            raise ValueError(f"haar_params must have length {n_sites}.")
        params = list(haar_params)
    else:
        params = [
            random_haar_qubit(None if seed is None else int(seed) + k, perturb=perturb)
            for k in range(n_sites)
        ]

    for x in range(Lx):
        for y in range(Ly):
            idx = x * Ly + y
            theta, phi = params[idx]
            local_vec = np.array(
                [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
                dtype=dtype,
            )
            tensor = peps[x, y]
            phys_ind = peps.site_ind(x, y)
            phys_axis = tensor.inds.index(phys_ind)
            data = np.zeros_like(tensor.data, dtype=dtype)

            slicer = [0] * data.ndim
            slicer[phys_axis] = slice(None)
            data[tuple(slicer)] = local_vec
            tensor.modify(data=data)

    peps.astype_(dtype)
    return peps


def hrps_to_mps(
    L: int,
    dtype: str = "complex128",
    cyclic: bool = False,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
):
    """Create a bond-dimension-1 MPS with per-site single-qubit Haar states.

    If ``haar_params`` is omitted, each site uses :func:`random_haar_qubit`.
    With ``seed`` set, site ``k`` uses ``seed + k`` for reproducible but
    distinct samples.
    """
    mps = ps_to_mps(L=L, dtype=dtype, theta=0.0, cyclic=cyclic)

    if haar_params is not None:
        if len(haar_params) != L:
            raise ValueError(f"haar_params must have length {L}.")
        params = list(haar_params)
    else:
        params = [
            random_haar_qubit(None if seed is None else int(seed) + k, perturb=perturb)
            for k in range(L)
        ]

    for i in range(L):
        theta, phi = params[i]
        local_vec = np.array(
            [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
            dtype=dtype,
        )
        tensor = mps[i]
        phys_ind = mps.site_ind(i)
        phys_axis = tensor.inds.index(phys_ind)
        data = np.zeros_like(tensor.data, dtype=dtype)

        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    mps.astype_(dtype)
    return mps
