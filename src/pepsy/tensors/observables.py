"""Tensor-network observables and fidelity implementation."""

from __future__ import annotations

from numbers import Integral
from string import Formatter
from typing import Any

import autoray as ar
import numpy as np

from ..backends import get_torch_linalg_config, to_float
from .contractions import build_optimizer, tn_norm

__all__ = ["measure_obs", "mps_entanglement_entropy", "tn_fidelity"]


def _is_symmray_array(value):
    """Return whether ``value`` is a native Symmray block-sparse array."""
    try:
        return ar.infer_backend(value) == "symmray"
    except (AttributeError, TypeError):
        return hasattr(value, "blocks") and hasattr(value, "indices")


def _normalize_mps_entropy_method(method):
    """Normalize the supported MPS Schmidt-spectrum decomposition method."""
    method = str(method).strip().lower().replace("_", ":")
    if method == "eig":
        method = "svd:eig"
    if method not in {"svd", "svd:eig"}:
        raise ValueError("MPS entropy method must be 'svd', 'eig', or 'svd:eig'.")
    return method


def _copy_mps_for_entropy(mps):
    """Copy MPS data and metadata before a diagnostic canonicalization sweep."""
    work = mps.copy()

    def copy_block(block):
        if ar.infer_backend(block) == "torch":
            block = block.detach()
        return ar.do("copy", block)

    for tensor in work.tensors:
        data = tensor.data
        if _is_symmray_array(data):
            data = data.copy_with(
                blocks={key: copy_block(block) for key, block in data.blocks.items()}
            )
        else:
            data = copy_block(data)
        # Ignore possibly stale source isometry metadata during the diagnostic
        # sweep. The source MPS and its canonical metadata remain untouched.
        tensor.modify(data=data, left_inds=None)
    if hasattr(work, "exponent"):
        work.exponent = 0.0
    return work


def _mps_bond_singular_values(mps, cut, *, method):
    """Extract the Schmidt values at ``cut`` from a private canonical MPS."""
    info = {"cur_orthog": "calc"}
    mps.canonicalize_(cut, info=info)
    center = mps[cut]
    previous = mps[cut - 1]
    left_inds = tuple(ind for ind in center.inds if ind in previous.inds)
    if len(left_inds) != 1:
        raise ValueError("MPS entropy requires one incoming bond at the selected cut.")
    right_inds = tuple(ind for ind in center.inds if ind not in left_inds)
    transposed = center.transpose(*left_inds, *right_inds)
    matrix = ar.do(
        "fuse",
        transposed.data,
        range(len(left_inds)),
        range(len(left_inds), len(transposed.inds)),
    )

    if _is_symmray_array(matrix):
        if method == "svd:eig":
            return matrix.svd_via_eig(
                absorb=None,
                max_bond=-1,
                cutoff=-1.0,
            )[1]
        return matrix.svd(
            absorb=None,
            max_bond=-1,
            cutoff=-1.0,
        )[1]

    if method == "svd:eig":
        rows, cols = tuple(matrix.shape)
        if rows <= cols:
            gram = ar.do(
                "matmul",
                matrix,
                ar.do("transpose", ar.do("conj", matrix)),
            )
        else:
            gram = ar.do(
                "matmul",
                ar.do("transpose", ar.do("conj", matrix)),
                matrix,
            )
        eig_result = ar.do("linalg.eigh", gram)
        eigenvalues = (
            eig_result.eigenvalues
            if hasattr(eig_result, "eigenvalues")
            else eig_result[0]
        )
        eigenvalues = ar.do("where", eigenvalues > 0.0, eigenvalues, 0.0)
        return ar.do("flip", ar.do("sqrt", eigenvalues), axis=0)

    backend = ar.infer_backend(matrix)
    if backend == "torch":
        # ``svdvals`` keeps the decomposition native and avoids allocating U
        # and Vh. Apply the active Pepsy CUDA driver when configured; Torch
        # rejects ``driver=`` for CPU tensors. Deliberately customized CPU or
        # stabilized policies use the registered full-SVD wrapper instead.
        config = get_torch_linalg_config()
        use_native_values = config is None or (
            not config.stabilized and config.cpu_svd == "torch"
        )
        if use_native_values:
            svd_kwargs = {}
            driver = None if config is None else config.svd_driver
            if getattr(matrix, "is_cuda", False) and driver not in {None, "auto"}:
                svd_kwargs["driver"] = driver
            return ar.do("linalg.svdvals", matrix, **svd_kwargs)

    try:
        return ar.do(
            "linalg.svd",
            matrix,
            full_matrices=False,
            compute_uv=False,
        )
    except TypeError:
        return ar.do("linalg.svd", matrix, full_matrices=False)[1]


def _entropy_from_singular_values(singular_values):
    """Reduce a backend-native Schmidt spectrum to normalized base-2 entropy."""
    if hasattr(singular_values, "to_dense"):
        singular_values = singular_values.to_dense()
    weights = ar.do("abs", singular_values) ** 2
    total = ar.do("sum", weights)
    # Keep normalization and the zero-norm branch on the active backend. This
    # avoids an intermediate scalar readback (and GPU synchronization) before
    # the final Python result is requested.
    safe_total = ar.do("where", total > 0.0, total, 1.0)
    probabilities = weights / safe_total
    # Keep zero entries backend-native while avoiding 0 * log2(0).
    safe_probabilities = ar.do(
        "where", probabilities > 0.0, probabilities, 1.0,
    )
    value = -ar.do(
        "sum",
        probabilities * ar.do("log2", safe_probabilities),
    )
    value = to_float(value, real=True)
    if not np.isfinite(value):
        raise ValueError("MPS entropy requires a finite state norm and result.")
    return max(0.0, value)


def mps_entanglement_entropy(mps, cut=None, *, method="svd"):
    """Measure normalized base-2 entropy across one open-MPS bond.

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        Open-boundary MPS to measure. The input tensors, exponent, and
        canonical metadata are preserved.
    cut : int | None, optional
        Bipartition index, with sites ``0 .. cut - 1`` on the left. ``None``
        selects the middle cut ``mps.L // 2``.
    method : {"svd", "eig", "svd:eig"}, default="svd"
        Backend-native decomposition method. ``"eig"`` is an alias for
        ``"svd:eig"`` and uses the smaller Gram matrix.

    Returns
    -------
    float
        Von Neumann entropy in bits.

    Notes
    -----
    Canonicalization and the local SVD run on a private copy. Dense Torch and
    CuPy tensors stay on their original device through Autoray/native linalg;
    native Symmray spectra use their sector-aware SVD. Only scalar reductions
    are read back for this Python ``float`` result. Cyclic MPSs are rejected
    because their loop environment has no single open-chain Schmidt cut.
    """
    if not hasattr(mps, "L") or not hasattr(mps, "canonicalize_"):
        raise TypeError("mps must be an open Quimb MatrixProductState.")
    if getattr(mps, "cyclic", False):
        raise ValueError("mps_entanglement_entropy requires an open MPS.")
    n_sites = int(mps.L)
    if n_sites < 2:
        raise ValueError("mps_entanglement_entropy requires at least two sites.")
    if cut is None:
        cut = n_sites // 2
    elif isinstance(cut, bool) or not isinstance(cut, Integral):
        raise TypeError("cut must be an integer bipartition index.")
    cut = int(cut)
    if not 0 < cut < n_sites:
        raise ValueError(f"cut must satisfy 0 < cut < {n_sites}, got {cut}.")
    method = _normalize_mps_entropy_method(method)
    work = _copy_mps_for_entropy(mps)
    singular_values = _mps_bond_singular_values(work, cut, method=method)
    return _entropy_from_singular_values(singular_values)

def _count_format_fields(fmt):
    return sum(field is not None for _, field, _, _ in Formatter().parse(fmt))


def _build_ind_id(prefix, arity):
    return prefix + ",".join("{}" for _ in range(int(arity)))


def _infer_where_coord_arity(where):
    """Infer coordinate arity from ``where`` when unambiguous."""
    if isinstance(where, str):
        return None

    if isinstance(where, Integral):
        return 1

    if not isinstance(where, (list, tuple)):
        return None
    if not where:
        return None

    if all(isinstance(v, Integral) for v in where):
        if len(where) == 1:
            return 1
        return None

    for site in where:
        if isinstance(site, (list, tuple)) and site and all(
            isinstance(v, Integral) for v in site
        ):
            return len(site)

    return None


def _infer_phys_ind_id(tn, where):
    """Infer default ``k``-prefixed physical index format for ``where``."""
    arity_hint = _infer_where_coord_arity(where)
    if arity_hint is None:
        if hasattr(tn, "Lz"):
            arity_hint = 3
        elif hasattr(tn, "Lx") and hasattr(tn, "Ly"):
            arity_hint = 2
        else:
            arity_hint = 1

    return _build_ind_id("k", arity_hint)


def _where_to_phys_inds(where, *, ind_id="k{}"):
    """Convert user-provided site selector(s) to physical index names."""
    n_fields = _count_format_fields(ind_id)
    if n_fields < 1:
        raise ValueError("ind_id must include at least one format field, e.g. 'k{}'.")

    if isinstance(where, str):
        return [where]

    if isinstance(where, Integral):
        if n_fields != 1:
            raise TypeError(
                "Scalar integer sites require a 1-field ind_id like 'k{}'."
            )
        return [ind_id.format(int(where))]

    if not isinstance(where, (list, tuple)):
        raise TypeError("where must be a site or a sequence of sites.")
    if not where:
        raise ValueError("where must not be empty.")

    if n_fields > 1 and len(where) == n_fields and all(
        isinstance(v, Integral) for v in where
    ):
        return [ind_id.format(*[int(v) for v in where])]

    inds = []
    for site in where:
        if isinstance(site, str):
            inds.append(site)
            continue

        if isinstance(site, Integral):
            if n_fields != 1:
                raise TypeError(
                    "Integer site entries require a 1-field ind_id like 'k{}'."
                )
            inds.append(ind_id.format(int(site)))
            continue

        if isinstance(site, (list, tuple)):
            if n_fields == 1:
                if not site or not all(isinstance(v, Integral) for v in site):
                    raise TypeError(
                        "For ind_id='k{}', nested where entries must contain integer sites."
                    )
                inds.extend(ind_id.format(int(v)) for v in site)
                continue

            if len(site) != n_fields or not all(isinstance(v, Integral) for v in site):
                raise TypeError(
                    "Each site tuple/list must match the number of ind_id fields."
                )
            inds.append(ind_id.format(*[int(v) for v in site]))
            continue

        raise TypeError(
            "where entries must be index strings, integers, or tuples/lists "
            "matching ind_id."
        )

    return inds


def measure_obs(
    tn,
    obs,
    where,
    *,
    ind_id=None,
    bra=None,
    normalize=True,
    contraction_opt: Any | None = None,
):
    """Measure local observable(s) on a tensor network ket.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Ket tensor network.
    obs : array_like | sequence[array_like]
        Observable tensor(s) to apply. This can be a single observable or a
        sequence matched with ``where``.
    where : site selector | sequence[site selector]
        Site selector(s) matching ``obs``. For batched use, provide one entry
        per observable. Site formatting follows ``ind_id`` and mirrors
        :func:`pepsy.operators.gates.gate` single-gate ``where`` usage.
    ind_id : str | None, optional
        Site-index format. If ``None`` (default), assume ``k``-prefixed
        indices based on ``where`` and TN dimensionality
        (``"k{}"``, ``"k{},{}"``, or ``"k{},{},{}"``).
        Networks using other prefixes (for example ``"b{}"``) must set
        ``ind_id`` explicitly.
    bra : qtn.TensorNetwork | None, optional
        If provided, compute ``<bra|obs|tn>`` directly without normalization.
        If ``None``, normalization is controlled by ``normalize``.
    normalize : bool, default=True
        If ``True`` and ``bra`` is ``None``, compute
        ``<tn|obs|tn> / <tn|tn>``. If ``False``, return raw ``<tn|obs|tn>``
        without computing ``tn_norm``.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.

    Returns
    -------
    scalar
        Measured observable value.

    Notes
    -----
    This function applies observables using :func:`pepsy.operators.gates.gate` with
    ``contract=False`` on a copy of ``tn`` before contraction.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    # Local import avoids circular import at module load time.
    from ..operators.gates import gate  # pylint: disable=import-outside-toplevel

    if isinstance(obs, (list, tuple)):
        if not isinstance(where, (list, tuple)):
            raise ValueError(
                "When obs is a sequence, where must be a matching sequence with "
                "the same length."
            )
        if len(obs) != len(where):
            raise ValueError(
                "When obs is a sequence, where must be a matching sequence with "
                "the same length."
            )
        obs_where_pairs = zip(obs, where)
    else:
        obs_where_pairs = ((obs, where),)

    tn_obs = tn.copy()
    infer_ind_id = ind_id is None
    for obs_i, where_i in obs_where_pairs:
        ind_id_i = _infer_phys_ind_id(tn, where_i) if infer_ind_id else ind_id
        target_inds = _where_to_phys_inds(where_i, ind_id=ind_id_i)
        outer_inds = set(tn_obs.outer_inds())
        missing = [ind for ind in target_inds if ind not in outer_inds]
        if missing:
            missing_str = ", ".join(sorted(set(missing)))
            raise ValueError(
                "Could not find target physical indices in tn.outer_inds(): "
                f"{missing_str}. If your TN uses non-'k' physical index names, "
                "pass ind_id explicitly (for example ind_id='b{}')."
            )
        tn_obs = gate(
            tn_obs,
            obs_i,
            where=where_i,
            ind_id=ind_id_i,
            contract=False,
            inplace=False,
        )

    if bra is not None:
        return (bra & tn_obs).contract(all, optimize=contraction_opt)

    numer = (tn.H & tn_obs).contract(all, optimize=contraction_opt)
    if not normalize:
        return numer

    norm_ = tn_norm(tn, contraction_opt=contraction_opt)
    if norm_ == 0.0:
        raise ValueError("Cannot compute normalized observable for a zero-norm state.")
    return numer / norm_


def tn_fidelity(
    psi,
    psi_fix,
    *,
    contraction_opt: Any | None = None,
    simplify: bool = False,
    simplify_seq: str = "R",
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
    simplify : bool, default=False
        Whether to simplify each closed norm/overlap network before
        contraction.
    simplify_seq : str, optional
        Simplification sequence passed to ``full_simplify_`` when
        ``simplify=True``.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)

    def closed_overlap(left, right):
        tn = left.H & right
        if simplify:
            tn.full_simplify_(seq=simplify_seq, output_inds=())
        return abs(tn.contract(all, optimize=contraction_opt, output_inds=()))

    val_0 = closed_overlap(psi, psi)
    val_1 = closed_overlap(psi, psi_fix)
    val_ref = closed_overlap(psi_fix, psi_fix)

    val_1 = val_1**2
    fidelity = ar.do("abs", val_1) / (val_0 * val_ref)
    return ar.do("abs", fidelity)
