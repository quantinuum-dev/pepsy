"""Boundary-contraction helpers for norms, overlaps, and infidelity."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass

from ..tensors.validation import _PHYS_OUTER, validate_tensor_network_tags
from .states import BdyMPS
from .sweeps import CompBdy

__all__ = [
    "build_bra_ket",
    "BoundaryContractResult",
    "contract_boundary",
    "contract_flat",
    "peps_normalize",
    "normalize",
    "boundary_norm",
    "peps_norm",
    "peps_infidelity",
    "peps_fidelity",
    "infidelity",
]

_DEFAULT_LAYER_TAGS = ("KET", "BRA")
_DEFAULT_BOUNDARY_SEQUENCE = ("xmax", "xmin", "ymin", "ymax")
_DEFAULT_BOUNDARY_SEQUENCE_3D = (
    "xmax",
    "xmin",
    "ymin",
    "ymax",
    "zmin",
    "zmax",
)


@dataclass(frozen=True)
class BoundaryContractResult:
    """Structured result from :func:`contract_boundary`.

    Fields store the contraction scalar plus sweep metadata.
    """

    cost: complex | float | tuple[complex | float, float]
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


def _is_scaled_scalar(value):
    return isinstance(value, (tuple, list)) and len(value) == 2


def _as_scaled_scalar(value, *, name="value"):
    """Return ``(mantissa, exponent)`` for scalar or stripped scalar input."""
    if _is_scaled_scalar(value):
        mantissa, exponent = value
        return _to_python_scalar(mantissa), float(_to_python_scalar(exponent))
    return _to_python_scalar(value), 0.0


def _safe_pow10(exponent):
    """Return ``10**exponent`` with a finite floating range."""
    exponent = float(exponent)
    if exponent <= -300.0:
        return 0.0
    if exponent >= 300.0:
        return 1.0e300
    return 10.0**exponent


def _scaled_to_complex(value):
    mantissa, exponent = _as_scaled_scalar(value)
    return complex(mantissa) * _safe_pow10(exponent)


def _format_scaled_output(value, *, strip_exponent):
    mantissa, exponent = _as_scaled_scalar(value)
    if strip_exponent:
        return mantissa, exponent
    return _scaled_to_complex((mantissa, exponent))


def _accumulate_tn_exponent(tn, exponent_delta):
    """Apply a base-10 exponent shift to a tensor network when available."""
    if exponent_delta == 0.0:
        return
    try:
        tn.exponent = float(getattr(tn, "exponent", 0.0)) + float(exponent_delta)
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - fallback
        # TensorNetwork normally exposes ``exponent``. If a compatible test
        # double does not, leave data-only scaling in place.
        return


def _normalize_by_scaled_norm(tn, norm_value):
    """Normalize ``tn`` using a scalar or ``(mantissa, exponent)`` norm."""
    mantissa, exponent = _as_scaled_scalar(norm_value)
    if abs(complex(mantissa)) == 0:
        raise ZeroDivisionError("Boundary norm cost is zero; cannot normalize state.")
    tn /= mantissa**0.5
    _accumulate_tn_exponent(tn, -0.5 * exponent)


def _scaled_overlap_fidelity(overlap, norm, norm_target):
    """Compute ``|overlap|**2 / (|norm| * |norm_target|)`` stably."""
    overlap_m, overlap_e = _as_scaled_scalar(overlap, name="overlap")
    norm_m, norm_e = _as_scaled_scalar(norm, name="norm")
    target_m, target_e = _as_scaled_scalar(norm_target, name="norm_target")

    denom_m = abs(complex(norm_m)) * abs(complex(target_m))
    if denom_m == 0:
        raise ZeroDivisionError(
            "Norm product is zero; cannot compute infidelity."
        )
    fidelity_m = (abs(complex(overlap_m)) ** 2) / denom_m
    fidelity_e = 2.0 * overlap_e - norm_e - target_e
    return fidelity_m * _safe_pow10(fidelity_e)


def _drop_existing_layer_tags(tn):
    """Remove internal KET/BRA layer tags if present."""
    stale = [tag for tag in ("KET", "BRA") if tag in getattr(tn, "tags", ())]
    if stale:
        tn.drop_tags(stale)


def _validate_chi(chi):
    """Validate and normalize an optional PEPS boundary bond dimension."""
    if chi is None:
        return None
    if not isinstance(chi, int):
        raise TypeError("chi must be an integer when provided.")
    if chi < 1:
        raise ValueError("chi must be >= 1 when provided.")
    return int(chi)


def _normalize_contraction_method(method):
    """Normalize PEPS metric contraction method names."""
    key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "fit": "dmrg",
        "boundary_fit": "dmrg",
        "boundary_dmrg": "dmrg",
        "dmrg": "dmrg",
        "boundary": "mps",
        "boundary_mps": "mps",
        "mps": "mps",
        "ctm": "ctmrg",
        "ctmrg": "ctmrg",
        "hotrg": "hotrg",
        "exact": "exact",
        "full": "exact",
    }
    if key == "rg":
        warnings.warn(
            "method='rg' has been renamed to method='ctmrg'.",
            UserWarning,
            stacklevel=3,
        )
        return "ctmrg"
    if key not in aliases:
        raise ValueError(
            "Unknown PEPS contraction method: "
            f"{method!r}. Expected 'dmrg', 'mps', 'ctmrg', 'hotrg', or 'exact'."
        )
    return aliases[key]


def _has_numbered_axis_tag(tn, axis):
    prefix = str(axis)
    for tag in getattr(tn, "tags", ()):
        if (
            isinstance(tag, str)
            and tag.startswith(prefix)
            and tag[len(prefix) :].isdigit()
        ):
            return True
    return False


def _infer_lattice_ndim(tn):
    """Infer whether ``tn`` is a 2D or 3D lattice TN when possible."""
    if getattr(tn, "Lz", None) is not None or _has_numbered_axis_tag(tn, "Z"):
        return 3
    if (
        getattr(tn, "Lx", None) is not None
        and getattr(tn, "Ly", None) is not None
    ):
        return 2
    if _has_numbered_axis_tag(tn, "X") and _has_numbered_axis_tag(tn, "Y"):
        return 2
    return None


def _default_quimb_sequence(tn, method):
    """Choose a quimb boundary sequence that matches the lattice dimension."""
    ndim = _infer_lattice_ndim(tn)
    if method == "hotrg":
        return ("x", "y", "z") if ndim == 3 else ("x", "y")
    if method in {"mps", "ctmrg"}:
        return _DEFAULT_BOUNDARY_SEQUENCE_3D if ndim == 3 else _DEFAULT_BOUNDARY_SEQUENCE
    return None


def _normalize_flat_contraction_method(method, tn):
    """Normalize method names for direct contraction of flat TNs."""
    key = str(method).strip().lower().replace("-", "_")
    if key == "auto":
        return "dmrg" if _infer_lattice_ndim(tn) == 2 else "mps"
    return _normalize_contraction_method(method)


def _call_with_accepted_kwargs(fn, **kwargs):
    """Call ``fn`` with only the keyword arguments it accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**kwargs)

    accepted = {key: val for key, val in kwargs.items() if key in sig.parameters}
    return fn(**accepted)


def _unpack_bdy_handle(handle, name):
    """Unpack a BdyMPS or ``{"bdy": BdyMPS}`` boundary handle."""
    holder = handle if isinstance(handle, dict) else None
    obj = None
    if holder is not None:
        obj = holder.get("bdy", None)
        if obj is not None and not hasattr(obj, "mps_b"):
            raise TypeError(f"{name}['bdy'] must expose attribute 'mps_b'.")
    elif handle is not None:
        obj = handle
        if not hasattr(obj, "mps_b"):
            raise TypeError(f"{name} must expose attribute 'mps_b'.")
    return obj, holder


def _retune_bdy_to_chi(obj, chi, name):
    """Retune an existing boundary object to the requested chi."""
    if obj is None or chi is None:
        return
    cur = getattr(obj, "chi", None)
    if cur is None or int(cur) == chi:
        return
    if not hasattr(obj, "expand_bnd"):
        raise TypeError(
            f"{name} has chi={cur} but cannot be retuned to chi={chi}; "
            "object must expose method 'expand_bnd'."
        )
    obj.expand_bnd(chi, inplace=True)


def _contract_quimb_double_layer(  # pylint: disable=too-many-arguments
    norm,
    *,
    method,
    chi,
    contraction_opt,
    max_separation,
    progress,
    strip_exponent,
    mode_,
    sequence,
    cutoff,
    equalize_norms,
    layer_tags,
):
    """Contract an already-built double-layer TN with a quimb-style method."""
    if method == "exact":
        return norm.contract(
            all,
            optimize=contraction_opt,
            strip_exponent=strip_exponent,
        )

    if chi is None:
        raise ValueError(f"Provide chi when method={method!r}.")

    final_contract_opts = {
        "optimize": contraction_opt,
        "strip_exponent": strip_exponent,
    }

    if method == "mps":
        contract_fn = getattr(norm, "contract_boundary", None)
        if not callable(contract_fn):
            raise TypeError("method='mps' requires a network with contract_boundary().")
        sequence = _default_quimb_sequence(norm, method) if sequence is None else sequence
        kwargs = dict(
            max_bond=chi,
            sequence=sequence,
            final_contract_opts=final_contract_opts,
            cutoff=cutoff,
            progbar=progress,
            max_separation=max_separation,
            equalize_norms=equalize_norms,
            inplace=False,
        )
        if mode_ is not None:
            kwargs["mode"] = mode_
        if layer_tags is not None:
            kwargs["layer_tags"] = list(layer_tags)
        return _call_with_accepted_kwargs(contract_fn, **kwargs)

    if method == "ctmrg":
        contract_fn = getattr(norm, "contract_ctmrg", None)
        if not callable(contract_fn):
            raise TypeError("method='ctmrg' requires a network with contract_ctmrg().")
        sequence = _default_quimb_sequence(norm, method) if sequence is None else sequence
        kwargs = dict(
            max_bond=chi,
            cutoff=cutoff,
            canonize=True,
            mode="projector",
            sequence=sequence,
            max_separation=max_separation,
            equalize_norms=equalize_norms,
            optimize=contraction_opt,
            final_contract=True,
            final_contract_opts=final_contract_opts,
            progbar=progress,
            inplace=False,
        )
        if layer_tags is not None:
            kwargs["layer_tags"] = list(layer_tags)
        return _call_with_accepted_kwargs(contract_fn, **kwargs)

    if method == "hotrg":
        contract_fn = getattr(norm, "contract_hotrg", None)
        if not callable(contract_fn):
            raise TypeError("method='hotrg' requires a network with contract_hotrg().")
        sequence = _default_quimb_sequence(norm, method) if sequence is None else sequence
        return _call_with_accepted_kwargs(
            contract_fn,
            max_bond=chi,
            cutoff=cutoff,
            sequence=sequence,
            max_separation=max_separation,
            equalize_norms=equalize_norms,
            optimize=contraction_opt,
            final_contract=True,
            final_contract_opts=final_contract_opts,
            progbar=progress,
            inplace=False,
        )

    raise ValueError(f"Unknown PEPS contraction method: {method!r}")


def _contract_peps_double_layer(  # pylint: disable=too-many-arguments
    norm,
    *,
    method,
    chi,
    bdy=None,
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    bdy_name="bdy",
    flat=False,
):
    """Contract a double-layer PEPS norm/overlap network by the selected method."""
    method = _normalize_contraction_method(method)
    chi = _validate_chi(chi)
    if layer_tags is None and not flat:
        layer_tags = _DEFAULT_LAYER_TAGS
    elif layer_tags is not None:
        layer_tags = tuple(layer_tags)

    if method == "dmrg":
        bdy_obj, bdy_holder = _unpack_bdy_handle(bdy, bdy_name)
        _retune_bdy_to_chi(bdy_obj, chi, bdy_name)
        if bdy_obj is None:
            if chi is None:
                raise ValueError(f"Provide chi when {bdy_name} is not supplied.")
            if flat:
                bdy_obj = BdyMPS(
                    tn_flat=norm,
                    chi=chi,
                    flat=True,
                    single_layer=single_layer,
                )
            else:
                bdy_obj = BdyMPS(
                    tn_double=norm,
                    chi=chi,
                    single_layer=single_layer,
                )
            if bdy_holder is not None:
                bdy_holder["bdy"] = bdy_obj

        result = contract_boundary(
            norm=norm,
            bdy=bdy_obj,
            contraction_opt=contraction_opt,
            fit_mode=fit_mode,
            n_iter=n_iter,
            progress=progress,
            direction=direction,
            max_separation=max_separation,
            track_boundary_fidelity=track_boundary_fidelity,
            visualize=visualize,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
            flat=flat,
        )
        return result.cost, bdy_obj

    cost = _contract_quimb_double_layer(
        norm,
        method=method,
        chi=chi,
        contraction_opt=contraction_opt,
        max_separation=max_separation,
        progress=progress,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
    )
    return cost, None


def contract_flat(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    tn,
    *,
    chi=None,
    bdy=None,
    method="auto",
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    visualize=False,
    strip_exponent=False,
    mode_=None,
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
):
    """Contract an already-flat PEPS-like tensor network.

    This helper is for networks that have already been flattened into a scalar
    tensor network, so it does not call :func:`build_bra_ket`. With
    ``method="auto"`` it uses the PEPSY DMRG/FIT boundary path for 2D networks
    and quimb's ``contract_boundary`` path for 3D networks. Explicit methods
    are ``"dmrg"``, ``"mps"``, ``"ctmrg"``, ``"hotrg"``, and ``"exact"``.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Already-flat tensor network to contract.
    chi : int | None, default=None
        Boundary bond dimension required by all approximate methods unless an
        existing ``bdy`` is supplied for ``method="dmrg"``.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Optional reusable boundary handle for ``method="dmrg"``. Dict holders
        are filled with ``dict["bdy"]`` when a new boundary is created.
    method : {"auto", "dmrg", "mps", "ctmrg", "hotrg", "exact"}, default="auto"
        Contraction backend. ``"dmrg"`` is only supported for 2D flat
        PEPS-like networks; quimb methods are used for 2D and 3D when the
        input network exposes the corresponding method.
    strip_exponent : bool, default=False
        If ``True``, return ``(mantissa, exponent)``.

    Returns
    -------
    complex | float | tuple[complex | float, float]
        Contraction scalar, optionally with stripped exponent.
    """
    if tn is None:
        raise ValueError("tn must not be None.")

    method = _normalize_flat_contraction_method(method, tn)
    if method == "dmrg" and _infer_lattice_ndim(tn) == 3:
        raise ValueError(
            "method='dmrg' is only supported for 2D flat tensor networks; "
            "use method='mps', 'ctmrg', 'hotrg', or 'exact' for 3D networks."
        )

    cost, _ = _contract_peps_double_layer(
        tn,
        method=method,
        chi=chi,
        bdy=bdy,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        single_layer=False,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        bdy_name="bdy",
        flat=True,
    )
    return _format_scaled_output(cost, strip_exponent=strip_exponent)


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

    # Layer tags are internal bookkeeping. Drop stale copies first so repeated
    # norm/overlap builds don't make bra tensors also selectable as ket tensors.
    _drop_existing_layer_tags(ket_tagged)
    _drop_existing_layer_tags(bra_tagged)
    ket_tagged.add_tag("KET")
    bra_tagged.add_tag("BRA")
    norm_tagged = bra_tagged | ket_tagged
    return ket_tagged, norm_tagged


def contract_boundary(
    *,
    norm,
    bdy=None,
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
    strip_exponent=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Approximate a scalar contraction using boundary-MPS sweeps.

    Parameters
    ----------
    norm : qtn.TensorNetwork
        Prebuilt double-layer network, usually from
        :func:`build_bra_ket`.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Boundary handle:

        - ``BdyMPS`` object: uses ``bdy.mps_b``
        - dict holder with ``{"bdy": <BdyMPS>}``: uses ``bdy["bdy"].mps_b``
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer passed through to :class:`pepsy.boundary.sweeps.CompBdy`.
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
        Whether to write fitted boundaries back into the boundary map.
    max_separation : int, default=1
        Sweep separation mode.
    direction : str, default="y"
        Sweep selector.
    equalize_norms : bool, default=False
        Forwarded normalization option for local fit outputs.
    strip_exponent : bool, default=False
        If ``True``, keep the final contraction as ``(mantissa, exponent)``.

    Returns
    -------
    BoundaryContractResult
        Structured contraction result with scalar ``cost`` and optional
        fidelity history ``fidel``.
    """
    if norm is None:
        raise ValueError("norm must not be None.")
    if bdy is None:
        raise ValueError("Provide bdy.")
    if hasattr(bdy, "mps_b"):
        mps_boundaries = bdy.mps_b
    elif isinstance(bdy, dict):
        bdy_obj = bdy.get("bdy", None)
        if not hasattr(bdy_obj, "mps_b"):
            raise TypeError("bdy dict must contain key 'bdy' with an object exposing attribute 'mps_b'.")
        mps_boundaries = bdy_obj.mps_b
    else:
        raise TypeError("bdy must be a BdyMPS-like object or a dict containing key 'bdy'.")

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
        strip_exponent=strip_exponent,
    )

    return BoundaryContractResult(
        cost=cost,
        fidel=list(comp_bdy.fidel),
        direction=direction,
        n_iter=n_iter,
        max_separation=max_separation,
    )


def _contract_state_norm(
    p,
    *,
    chi,
    bdy,
    method,
    contraction_opt,
    n_iter,
    direction,
    max_separation,
    progress,
    track_boundary_fidelity,
    fit_mode,
    single_layer,
    visualize,
    strip_exponent,
    mode_,
    sequence,
    cutoff,
    equalize_norms,
    layer_tags,
):
    """Build ``<p|p>``, set up the boundary, and contract it.

    Shared backend for :func:`peps_normalize` and :func:`boundary_norm`. Tags
    ``p`` in place via :func:`build_bra_ket` but does **not** rescale it.

    Returns
    -------
    tuple[BoundaryContractResult, qtn.TensorNetwork, object]
        ``(result, ket_tagged, bdy_obj)`` where ``result.cost`` is the
        ``<p|p>`` contraction scalar.
    """
    ket_tagged, norm_tagged = build_bra_ket(ket=p, bra=None)

    cost, bdy_obj = _contract_peps_double_layer(
        norm_tagged,
        method=method,
        chi=chi,
        bdy=bdy,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        bdy_name="bdy",
    )
    result = BoundaryContractResult(
        cost=cost,
        fidel=[],
        direction=direction,
        n_iter=n_iter,
        max_separation=max_separation,
    )
    return result, ket_tagged, bdy_obj


def peps_normalize(
    p,
    *,
    chi=None,
    bdy=None,
    method="dmrg",
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    balance_bonds=True,
):
    """Normalize a PEPS state in place using boundary contraction.

    With ``method="dmrg"`` this performs the existing
    ``build_bra_ket -> BdyMPS -> contract_boundary`` flow. Other methods use
    quimb-style double-layer contractions when available. The state is rescaled
    in place by ``1 / sqrt(<p|p>)``. To compute ``<p|p>`` without modifying
    ``p``, use :func:`boundary_norm`.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state.
    chi : int | None, default=None
        Boundary MPS bond dimension used when ``bdy`` is not provided.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Boundary handle:

        - ``BdyMPS``: reused and updated in place.
        - ``dict``: if ``dict["bdy"]`` exists it is reused; otherwise a new
          boundary is created (requires ``chi``) and written to ``dict["bdy"]``.
        - ``None``: a new boundary is created internally (requires ``chi``).

        Only used with ``method="dmrg"``.
    method : {"dmrg", "mps", "ctmrg", "hotrg", "exact"}, default="dmrg"
        Contraction backend. ``"dmrg"`` is the package BdyMPS/FIT path.
        ``"mps"``, ``"ctmrg"``, and ``"hotrg"`` call matching quimb methods
        on the double-layer network. ``"rg"`` is accepted as a deprecated alias
        for ``"ctmrg"``.
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
        Boundary initializer mode for :class:`pepsy.boundary.states.BdyMPS`.
    strip_exponent : bool, default=False
        If ``True``, use stripped boundary contractions and return
        ``(mantissa, exponent)`` for the old norm estimate.
    balance_bonds : bool, default=True
        If ``True``, call ``balance_bonds_()`` after rescaling the state.

    Returns
    -------
    complex | float
        The old norm estimate returned by the boundary contraction before
        rescaling.
    """
    if p is None:
        raise ValueError("p must not be None.")
    chi = _validate_chi(chi)

    result, ket_tagged, _ = _contract_state_norm(
        p,
        chi=chi,
        bdy=bdy,
        method=method,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
    )
    cost = result.cost
    old_norm = _format_scaled_output(cost, strip_exponent=strip_exponent)
    _normalize_by_scaled_norm(ket_tagged, cost)
    if balance_bonds:
        ket_tagged.balance_bonds_()
    return old_norm


def boundary_norm(
    p,
    *,
    chi=None,
    bdy=None,
    method="dmrg",
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
):
    """Compute ``<p|p>`` via boundary contraction without rescaling ``p``.

    This is the read-only counterpart of :func:`peps_normalize`: it performs the
    same ``build_bra_ket -> BdyMPS -> contract_boundary`` pipeline and returns
    the contraction scalar, but never divides the state by its norm. ``p`` is
    not modified (a copy is contracted internally).

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state.
    chi : int | None, default=None
        Boundary MPS bond dimension used when ``bdy`` is not provided.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Boundary handle, identical in meaning to :func:`peps_normalize`.
    method : {"dmrg", "mps", "ctmrg", "hotrg", "exact"}, default="dmrg"
        Contraction backend.
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
        Boundary initializer mode for :class:`pepsy.boundary.states.BdyMPS`.
    strip_exponent : bool, default=False
        If ``True``, return ``(mantissa, exponent)`` for the norm estimate.

    Returns
    -------
    complex | float
        The ``<p|p>`` norm estimate from the boundary contraction.
    """
    if p is None:
        raise ValueError("p must not be None.")
    chi = _validate_chi(chi)

    result, _, _ = _contract_state_norm(
        p.copy(),
        chi=chi,
        bdy=bdy,
        method=method,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
    )
    return _format_scaled_output(result.cost, strip_exponent=strip_exponent)


def peps_norm(
    p,
    *,
    chi=None,
    bdy=None,
    method="dmrg",
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
):
    """Compute the PEPS norm ``<p|p>`` without modifying ``p``.

    This is the PEPS-named alias for :func:`boundary_norm`. It accepts the same
    contraction methods and boundary controls, including ``method="dmrg"``,
    ``"mps"``, ``"ctmrg"``, ``"hotrg"``, and ``"exact"``.

    Returns
    -------
    complex | float | tuple[complex | float, float]
        Norm estimate. With ``strip_exponent=True`` this is
        ``(mantissa, exponent)``.
    """
    return boundary_norm(
        p,
        chi=chi,
        bdy=bdy,
        method=method,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
    )


def peps_infidelity(
    p,
    p_target,
    *,
    chi=None,
    norm=None,
    norm_target=None,
    bdy=None,
    bdy_target=None,
    bdy_overlap=None,
    method="dmrg",
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    r"""Compute the infidelity between two PEPS states via boundary contraction.

    The infidelity is defined as

    .. math::

        \mathcal{I} = 1
          - \frac{|\langle p_{\mathrm{target}} | p \rangle|^{2}}
                 {\langle p | p \rangle \;
                  \langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle}

    Up to three contractions are performed to evaluate
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
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Pre-built boundary for the :math:`\langle p | p \rangle` network.
        Also supports dict holder style ``{"bdy": <BdyMPS>}``.
        Ignored when ``norm`` is given.
    bdy_target : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Pre-built boundary for the
        :math:`\langle p_{\mathrm{target}} | p_{\mathrm{target}} \rangle`
        network. Also supports dict holder style ``{"bdy": <BdyMPS>}``.
        Ignored when ``norm_target`` is given.
    bdy_overlap : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Pre-built boundary for the
        :math:`\langle p_{\mathrm{target}} | p \rangle` overlap network.
        Also supports dict holder style ``{"bdy": <BdyMPS>}``.
    method : {"dmrg", "mps", "ctmrg", "hotrg", "exact"}, default="dmrg"
        Contraction backend. ``"dmrg"`` is the package BdyMPS/FIT path.
        ``"mps"``, ``"ctmrg"``, and ``"hotrg"`` call matching quimb methods
        on each double-layer network. ``"rg"`` is accepted as a deprecated
        alias for ``"ctmrg"``.
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
        Boundary initializer mode for :class:`pepsy.boundary.states.BdyMPS`.
    visualize : bool, default=False
    strip_exponent : bool, default=False
        If ``True``, keep norm and overlap contractions as
        ``(mantissa, exponent)`` pairs and compute the fidelity ratio without
        reconstructing large or tiny scalars.
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
    chi = _validate_chi(chi)
    method = _normalize_contraction_method(method)

    # Shared contraction kwargs
    _kw = dict(
        method=method,
        chi=chi,
        contraction_opt=contraction_opt,
        fit_mode=fit_mode,
        n_iter=n_iter,
        progress=progress,
        direction=direction,
        max_separation=max_separation,
        track_boundary_fidelity=track_boundary_fidelity,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
    )

    # -- <p|p> --
    bdy_obj = None
    if norm is None:
        _, norm_tn = build_bra_ket(ket=p, bra=None)
        cost, bdy_obj = _contract_peps_double_layer(
            norm_tn,
            bdy=bdy,
            single_layer=single_layer,
            bdy_name="bdy",
            **_kw,
        )
        norm = _format_scaled_output(
            cost,
            strip_exponent=strip_exponent,
        )
    else:
        norm = _format_scaled_output(norm, strip_exponent=strip_exponent)

    # -- <p_target|p_target> --
    bdy_target_obj = None
    if norm_target is None:
        _, norm_target_tn = build_bra_ket(ket=p_target, bra=None)
        cost, bdy_target_obj = _contract_peps_double_layer(
            norm_target_tn,
            bdy=bdy_target,
            single_layer=single_layer,
            bdy_name="bdy_target",
            **_kw,
        )
        norm_target = _format_scaled_output(
            cost,
            strip_exponent=strip_exponent,
        )
    else:
        norm_target = _format_scaled_output(norm_target, strip_exponent=strip_exponent)

    # -- <p_target|p> (overlap) --
    _, overlap_tn = build_bra_ket(ket=p, bra=p_target)
    cost, bdy_overlap_obj = _contract_peps_double_layer(
        overlap_tn,
        bdy=bdy_overlap,
        single_layer=single_layer,
        bdy_name="bdy_overlap",
        **_kw,
    )
    overlap = _format_scaled_output(
        cost,
        strip_exponent=strip_exponent,
    )

    fidelity = _scaled_overlap_fidelity(overlap, norm, norm_target)

    return {
        "infidelity": 1 - fidelity,
        "norm": norm,
        "norm_target": norm_target,
        "overlap": overlap,
        "bdy": bdy_obj,
        "bdy_target": bdy_target_obj,
        "bdy_overlap": bdy_overlap_obj,
    }


def peps_fidelity(
    p,
    p_target,
    *,
    chi=None,
    norm=None,
    norm_target=None,
    bdy=None,
    bdy_target=None,
    bdy_overlap=None,
    method="dmrg",
    contraction_opt="auto-hq",
    n_iter=10,
    direction="y",
    max_separation=1,
    progress=False,
    track_boundary_fidelity=False,
    fit_mode="eff",
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
):
    """Compute boundary-estimated PEPS fidelity.

    This is a convenience wrapper around :func:`peps_infidelity` that returns
    ``1 - infidelity``. Pass ``norm`` and/or ``norm_target`` when one of the
    states is already known to be normalized; the corresponding self-overlap
    contraction is then skipped.

    Returns
    -------
    float
        Boundary-estimated fidelity.
    """
    result = peps_infidelity(
        p,
        p_target,
        chi=chi,
        norm=norm,
        norm_target=norm_target,
        bdy=bdy,
        bdy_target=bdy_target,
        bdy_overlap=bdy_overlap,
        method=method,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
    )
    return 1 - result["infidelity"]


def normalize(*args, **kwargs):
    """Compatibility alias for :func:`peps_normalize`."""
    return peps_normalize(*args, **kwargs)


def infidelity(*args, **kwargs):
    """Compatibility alias for :func:`peps_infidelity`."""
    return peps_infidelity(*args, **kwargs)
