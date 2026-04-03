"""Boundary-contraction helpers for norms, overlaps, and infidelity."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ._tn_validation import _PHYS_OUTER, validate_tensor_network_tags
from .boundary_states import BdyMPS
from .boundary_sweeps import CompBdy

__all__ = [
    "build_bra_ket",
    "BoundaryContractResult",
    "contract_boundary",
    "normalize",
    "infidelity",
]


@dataclass(frozen=True)
class BoundaryContractResult:
    """Structured result from :func:`contract_boundary`.

    Attributes
    ----------
    cost : complex | float
        Final scalar returned by the boundary contraction.
    fidel : list[float]
        Optional per-step fidelity values collected during the sweep.
    direction : str
        Direction selector used for the contraction call.
    n_iter : int
        Local fit iterations used per boundary update.
    max_separation : int
        Separation mode used during the sweep.
    """

    cost: complex | float
    fidel: list[float]
    direction: str
    n_iter: int
    max_separation: int


def _warn_nonstandard_physical_outer_inds(tn, role):
    """Warn when outer physical indices don't match ``k<int>[,<int>...]`` or ``b<int>[,<int>...]``."""
    bad = [
        idx
        for idx in tn.outer_inds()
        if not (isinstance(idx, str) and _PHYS_OUTER.fullmatch(idx))
    ]
    if bad:
        sample = ", ".join(sorted(bad)[:8])
        warnings.warn(
            f"{role} outer indices expected format k/b<int>[,<int>...]. "
            f"Found non-matching indices: {sample}",
            stacklevel=3,
        )


def _to_python_scalar(value):
    """Convert backend scalar-like objects (torch/numpy) to python scalar."""
    obj = value
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "item") and not isinstance(obj, (int, float, complex, bool)):
        try:
            obj = obj.item()
        except (ValueError, RuntimeError):  # backend-specific .item() failures
            pass
    return obj



def build_bra_ket(
    ket=None,
    *,
    bra=None,
):
    """Prepare tagged ``ket``/``bra`` networks and build a double-layer TN.

    Parameters
    ----------
    ket : qtn.TensorNetwork | PEPS
        Input ket network.
    bra : qtn.TensorNetwork | PEPS
        Optional bra network. If ``None``, ``ket.copy().conj()`` is used. Note: always conjugated.

    Returns
    -------
    tuple[qtn.TensorNetwork, qtn.TensorNetwork]
        ``(ket_tagged, norm_tagged)`` where:

        - ``ket_tagged`` is ``ket`` with tag ``"KET"``
        - ``norm_tagged`` is ``bra_tagged | ket_tagged``

    Notes
    -----
    ``ket`` is tagged in-place (no copy). The returned ``ket_tagged`` is
    the same object as ``ket``.

    In both cases (auto-generated or provided ``bra``), any shared internal
    indices between ket and bra are automatically renamed on the bra side as
    ``<original>_*`` to ensure disjointness.
    """
    if ket is None:
        raise ValueError("Provide ket.")

    validate_tensor_network_tags(ket)

    ket_tagged = ket
    auto_bra = bra is None
    bra_tagged = ket.conj() if auto_bra else bra.conj()

    # Ensure bra internal indices are disjoint from ket's.
    shared_inner = set(ket_tagged.inner_inds()) & set(bra_tagged.inner_inds())
    if shared_inner:
        reindex_map = {idx: f"{idx}_*" for idx in shared_inner}
        final_collisions = set(reindex_map.values()) & (
            set(ket_tagged.ind_map) | (set(bra_tagged.ind_map) - shared_inner)
        )
        if final_collisions:
            sample = ", ".join(sorted(final_collisions)[:8])
            raise ValueError(
                "Bra reindex idx -> idx_* collides with existing indices. "
                f"Collisions found: {sample}"
            )
        bra_tagged.reindex_(reindex_map)

    _warn_nonstandard_physical_outer_inds(ket_tagged, "ket")
    if not auto_bra:
        _warn_nonstandard_physical_outer_inds(bra_tagged, "bra")

    ket_tagged.add_tag("KET")
    bra_tagged.add_tag("BRA")
    norm_tagged = bra_tagged | ket_tagged
    return ket_tagged, norm_tagged


def contract_boundary(
    *,
    norm,
    mps_boundaries,
    contraction_opt="auto-hq",
    flat=False,
    fit_mode="eff",
    n_iter=10,
    retag=True,
    progress=True,
    track_boundary_fidelity=False,
    visualize=False,
    write_back=True,
    max_separation=1,
    direction="y",
    equalize_norms=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Approximate a scalar contraction using boundary-MPS sweeps.

    Parameters
    ----------
    norm : qtn.TensorNetwork
        Prebuilt double-layer network, usually from
        :func:`build_bra_ket`.
    mps_boundaries : dict[str, qtn.MatrixProductState]
        Boundary dictionary, usually from :class:`pepsy.boundary_states.BdyMPS`.
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer passed through to :class:`pepsy.boundary_sweeps.CompBdy`.
    flat : bool, default=False
        Forwarded to sweep backend.
    fit_mode : {"eff", "global"}, default="eff"
        Fit backend mode.
    n_iter : int, default=10
        Number of local fit iterations per step.
    retag : bool, default=True
        Forwarded to fitting backend.
    progress : bool, default=True
        Show progress bars.
    track_boundary_fidelity : bool, default=False
        If ``True``, collect per-step fidelity values in ``result.fidel``.
    visualize : bool, default=False
        Enable intermediate visualization in fitting backend.
    write_back : bool, default=True
        Whether to write fitted boundaries back into ``mps_boundaries``.
    max_separation : int, default=1
        Sweep separation mode.
    direction : str, default="y"
        Sweep selector.
    equalize_norms : bool, default=False
        Forwarded normalization option for local fit outputs.

    Returns
    -------
    BoundaryContractResult
        Structured contraction result with scalar ``cost`` and optional
        fidelity history ``fidel``.
    """
    if norm is None:
        raise ValueError("norm must not be None.")
    if not isinstance(mps_boundaries, dict):
        raise TypeError("mps_boundaries must be a dictionary of boundary states.")

    retag = bool(retag)
    norm_tagged = norm.copy()

    comp_bdy = CompBdy(
        norm_tagged,
        mps_boundaries,
        contraction_opt=contraction_opt,
        fit_mode=fit_mode,
    )

    cost = comp_bdy.run(
        n_iter=n_iter,
        retag=retag,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        visualize=visualize,
        flat=flat,
        write_back=write_back,
        max_separation=max_separation,
        direction=direction,
        equalize_norms=equalize_norms,
    )

    return BoundaryContractResult(
        cost=cost,
        fidel=list(comp_bdy.fidel),
        direction=direction,
        n_iter=n_iter,
        max_separation=max_separation,
    )


def normalize(
    p,
    *,
    chi=None,
    bdy=None,
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
):
    """Normalize a PEPS state in place using boundary contraction.

    This performs:
    ``build_bra_ket -> BdyMPS -> contract_boundary`` and normalizes
    ``p`` in place.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state.
    chi : int | None, default=None
        Boundary MPS bond dimension used when ``bdy`` is not provided.
    bdy : pepsy.boundary_states.BdyMPS | None, default=None
        Pre-built boundary object. If provided, ``normalize`` reuses it and
        skips creating a new :class:`pepsy.boundary_states.BdyMPS`.
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer.
    n_iter : int, default=10
        Number of local fit iterations per boundary step.
    direction : str, default="y"
        Sweep direction passed to :func:`contract_boundary`.
    max_separation : int, default=1
        Sweep separation mode.
    progress : bool, default=False
        Show progress bar.
    track_boundary_fidelity : bool, default=False
        Track fidelity history during boundary contraction.
    fit_mode : {"eff", "global"}, default="eff"
        Boundary fitting backend mode.
    single_layer : bool, default=False
        Boundary initializer mode for :class:`pepsy.boundary_states.BdyMPS`.

    Returns
    -------
    complex | float
        Old norm estimate returned by the boundary contraction
        before rescaling.
    """
    if p is None:
        raise ValueError("p must not be None.")

    ket_tagged, norm_tagged = build_bra_ket(ket=p, bra=None)

    if bdy is None:
        if chi is None:
            raise ValueError("Provide chi when bdy is not supplied.")

        bdy = BdyMPS(
            tn_double=norm_tagged,
            chi=chi,
            single_layer=single_layer,
        )
    elif not hasattr(bdy, "mps_b"):
        raise TypeError("bdy must expose attribute 'mps_b'.")

    result = contract_boundary(
        norm=norm_tagged,
        mps_boundaries=bdy.mps_b,
        contraction_opt=contraction_opt,
        fit_mode=fit_mode,
        n_iter=n_iter,
        progress=progress,
        direction=direction,
        max_separation=max_separation,
        track_boundary_fidelity=track_boundary_fidelity,
    )
    cost = result.cost
    old_norm = _to_python_scalar(cost)
    if abs(complex(old_norm)) == 0:
        raise ZeroDivisionError("Boundary norm cost is zero; cannot normalize state.")

    ket_tagged /= cost**0.5
    ket_tagged.balance_bonds_()
    return old_norm


def infidelity(
    p,
    p_target,
    *,
    chi=None,
    norm=None,
    norm_target=None,
    bdy=None,
    bdy_target=None,
    bdy_overlap=None,
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    r"""Compute the infidelity between two PEPS states via boundary contraction.

    The infidelity is defined as

    .. math::

        \mathcal{I} = 1
          - \frac{|\langle p_{\mathrm{target}} | p \rangle|^{2}}
                 {\langle p | p \rangle \;
                  \langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle}

    Up to three boundary contractions are performed to evaluate
    :math:`\langle p | p \rangle`,
    :math:`\langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle`, and
    :math:`\langle p_{\mathrm{target}} | p \rangle`.
    If ``norm`` or ``norm_target`` is supplied, the corresponding
    contraction is skipped.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Trial PEPS state.
    p_target : qtn.TensorNetwork
        Target PEPS state.
    chi : int | None, default=None
        Boundary MPS bond dimension. Required when the corresponding
        ``bdy*`` argument is not supplied.
    norm : complex | float | None, default=None
        Known value of :math:`\langle p | p \rangle`.  When provided the
        :math:`\langle p | p \rangle` boundary contraction is skipped
        (e.g. pass ``1`` for an already-normalized state).
    norm_target : complex | float | None, default=None
        Known value of
        :math:`\langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle`.
        When provided the corresponding contraction is skipped.
    bdy : pepsy.boundary_states.BdyMPS | None, default=None
        Pre-built boundary for the :math:`\langle p | p \rangle` network.
        Ignored when ``norm`` is given.
    bdy_target : pepsy.boundary_states.BdyMPS | None, default=None
        Pre-built boundary for the
        :math:`\langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle`
        network.  Ignored when ``norm_target`` is given.
    bdy_overlap : pepsy.boundary_states.BdyMPS | None, default=None
        Pre-built boundary for the
        :math:`\langle p_{\mathrm{target}} | p \rangle` overlap network.
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer passed to :func:`contract_boundary`.
    n_iter : int, default=10
        Number of local fit iterations per boundary step.
    direction : str, default="y"
        Sweep direction passed to :func:`contract_boundary`.
    max_separation : int, default=1
        Sweep separation mode.
    progress : bool, default=False
        Show progress bar.
    track_boundary_fidelity : bool, default=False
        Track per-step fidelity during boundary contraction.
    fit_mode : {"eff", "global"}, default="eff"
        Boundary fitting backend mode.
    single_layer : bool, default=False
        Boundary initializer mode for :class:`pepsy.boundary_states.BdyMPS`.

    Returns
    -------
    dict[str, object]
        Dictionary with:

        - ``infidelity``: :math:`1 - F` where :math:`F` is the fidelity
        - ``norm``: :math:`\langle p | p \rangle` (complex scalar)
        - ``norm_target``: :math:`\langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle`
        - ``overlap``: :math:`\langle p_{\mathrm{target}} | p \rangle` (complex scalar)
        - ``bdy``: boundary MPS for *p* (``None`` when ``norm`` was given)
        - ``bdy_target``: boundary MPS for *p_target* (``None`` when
          ``norm_target`` was given)
        - ``bdy_overlap``: boundary MPS for the overlap network
    """
    if p is None:
        raise ValueError("p must not be None.")
    if p_target is None:
        raise ValueError("p_target must not be None.")

    # Shared contraction kwargs
    _kw = dict(
        contraction_opt=contraction_opt,
        fit_mode=fit_mode,
        n_iter=n_iter,
        progress=progress,
        direction=direction,
        max_separation=max_separation,
        track_boundary_fidelity=track_boundary_fidelity,
    )

    # -- <p|p> --
    if norm is None:
        _, norm_tn = build_bra_ket(ket=p, bra=None)

        if bdy is None:
            if chi is None:
                raise ValueError("Provide chi when bdy is not supplied.")
            bdy = BdyMPS(tn_double=norm_tn, chi=chi, single_layer=single_layer)
        elif not hasattr(bdy, "mps_b"):
            raise TypeError("bdy must expose attribute 'mps_b'.")

        norm = complex(_to_python_scalar(
            contract_boundary(norm=norm_tn, mps_boundaries=bdy.mps_b, **_kw).cost
        ))
    else:
        norm = complex(norm)

    # -- <p_target|p_target> --
    if norm_target is None:
        _, norm_target_tn = build_bra_ket(ket=p_target, bra=None)

        if bdy_target is None:
            if chi is None:
                raise ValueError("Provide chi when bdy_target is not supplied.")
            bdy_target = BdyMPS(
                tn_double=norm_target_tn, chi=chi, single_layer=single_layer,
            )
        elif not hasattr(bdy_target, "mps_b"):
            raise TypeError("bdy_target must expose attribute 'mps_b'.")

        norm_target = complex(_to_python_scalar(
            contract_boundary(
                norm=norm_target_tn, mps_boundaries=bdy_target.mps_b, **_kw,
            ).cost
        ))
    else:
        norm_target = complex(norm_target)

    # -- <p_target|p> (overlap) --
    _, overlap_tn = build_bra_ket(ket=p, bra=p_target)

    if bdy_overlap is None:
        if chi is None:
            raise ValueError("Provide chi when bdy_overlap is not supplied.")
        bdy_overlap = BdyMPS(
            tn_double=overlap_tn, chi=chi, single_layer=single_layer,
        )
    elif not hasattr(bdy_overlap, "mps_b"):
        raise TypeError("bdy_overlap must expose attribute 'mps_b'.")

    overlap = complex(_to_python_scalar(
        contract_boundary(
            norm=overlap_tn, mps_boundaries=bdy_overlap.mps_b, **_kw,
        ).cost
    ))

    denom = abs(norm) * abs(norm_target)
    if denom == 0:
        raise ZeroDivisionError(
            "Norm product is zero; cannot compute infidelity."
        )

    fidelity = abs(overlap) ** 2 / denom

    return {
        "infidelity": 1 - fidelity,
        "norm": norm,
        "norm_target": norm_target,
        "overlap": overlap,
        "bdy": bdy,
        "bdy_target": bdy_target,
        "bdy_overlap": bdy_overlap,
    }
