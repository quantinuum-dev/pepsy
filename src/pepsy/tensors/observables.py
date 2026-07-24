"""Tensor-network observables and fidelity implementation."""

from __future__ import annotations

from numbers import Integral
from string import Formatter
from typing import Any

import autoray as ar

from .contractions import build_optimizer, tn_norm

__all__ = ["measure_obs", "tn_fidelity"]

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
