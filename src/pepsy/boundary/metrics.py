"""Boundary-contraction helpers for norms, overlaps, and infidelity."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
import math
import warnings
from dataclasses import dataclass, replace

import autoray as ar

from .._internal.quimb import quimb_2d_options, quimb_compression_options
from .._internal.quimb import (
    require_quimb_1d_callable_compression,
    require_quimb_ctmrg_mode,
    require_quimb_ctmrg_projector_canonize,
)
from ..tensors.validation import _PHYS_OUTER, validate_tensor_network_tags
from .states import BdyMPS
from ._fit_policy import (
    _FIT_QUIMB_MODES,
    _canonical_fit_layer_mode,
    _canonical_fit_layer_order,
    _canonical_fit_mode_selector,
)
from .sweeps import BoundaryFitDiagnostic, CompBdy

__all__ = [
    "build_bra_ket",
    "BoundaryContractResult",
    "BoundaryFitDiagnostic",
    "contract_boundary",
    "contract_flat",
    "contract_layered",
    "quimb_ctmrg_projector_compat",
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
_FLAT_BOUNDARY_DIRECTION_PRESETS = {
    "bottom-up": ("xmin",),
    "top-down": ("xmax",),
    "top-bottom": ("xmax", "xmin"),
    "bottom-top": ("xmin", "xmax"),
    "left-to-right": ("ymin",),
    "right-to-left": ("ymax",),
    "left-right": ("ymin", "ymax"),
    "right-left": ("ymax", "ymin"),
    "four-sided": _DEFAULT_BOUNDARY_SEQUENCE,
}
_CTMRG_MODES = frozenset({"projector", "projector2d", "l2bp"})


def _canonical_flat_boundary_direction(value, *, direction):
    """Resolve a readable flat-contraction direction preset.

    The returned ``middle_axis`` is non-``None`` only when opposing outer
    boundaries should be absorbed towards a selected central slab. All other
    presets map directly to an unconstrained Quimb boundary sequence.
    """
    if value is None:
        return None, None

    key = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "auto": "auto",
        "bottom": "bottom-up",
        "top": "top-down",
        "left": "left-to-right",
        "right": "right-to-left",
        "all": "four-sided",
    }
    key = aliases.get(key, key)
    if key == "auto":
        return None, None
    if key == "middle-out":
        axis = str(direction).strip().lower()[:1]
        if axis not in {"x", "y"}:
            raise ValueError(
                "direction must begin with 'x' or 'y' when "
                "boundary_direction='middle-out'."
            )
        return None, axis
    if key in {"middle-out-x", "middle-out-y"}:
        return None, key[-1]
    try:
        return _FLAT_BOUNDARY_DIRECTION_PRESETS[key], None
    except KeyError as exc:
        choices = ", ".join(
            repr(name)
            for name in (
                *_FLAT_BOUNDARY_DIRECTION_PRESETS,
                "middle-out-x",
                "middle-out-y",
            )
        )
        raise ValueError(
            f"Unknown flat boundary_direction {value!r}. "
            f"Expected one of {choices}."
        ) from exc


def _canonical_middle_slices(middle_slices, *, length):
    """Resolve and validate the contiguous target slab for middle-out."""
    if middle_slices is None:
        midpoint = length // 2
        return (midpoint,) if length % 2 else (midpoint - 1, midpoint)
    if isinstance(middle_slices, bool):
        raise TypeError("middle_slices must contain integer slice indices.")
    if isinstance(middle_slices, int):
        slices = (middle_slices,)
    else:
        try:
            slices = tuple(middle_slices)
        except TypeError as exc:
            raise TypeError(
                "middle_slices must be an integer or a sequence of integers."
            ) from exc
    if not slices or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in slices
    ):
        raise TypeError("middle_slices must contain integer slice indices.")
    slices = tuple(sorted(set(slices)))
    if slices[0] < 0 or slices[-1] >= length:
        raise ValueError(
            f"middle_slices must lie in [0, {length - 1}]; got {slices!r}."
        )
    if slices != tuple(range(slices[0], slices[-1] + 1)):
        raise ValueError(
            "middle_slices must select one contiguous central slab; "
            f"got {slices!r}."
        )
    return slices


def _flat_middle_around(tn, *, axis, middle_slices):
    """Build Quimb ``around`` coordinates for an inward target slab."""
    axis = str(axis).lower()
    length = getattr(tn, f"L{axis}", None)
    other_axis = "y" if axis == "x" else "x"
    other_length = getattr(tn, f"L{other_axis}", None)
    if not (
        isinstance(length, int)
        and length >= 1
        and isinstance(other_length, int)
        and other_length >= 1
    ):
        raise TypeError(
            "middle-out contraction requires a 2D Quimb lattice network "
            "with integer Lx and Ly."
        )
    is_cyclic = getattr(tn, f"is_cyclic_{axis}", None)
    if callable(is_cyclic) and is_cyclic():
        raise ValueError(
            f"middle-out-{axis} requires two open {axis} boundaries; "
            f"choose the other axis or explicitly cut the cyclic {axis} bond."
        )

    middle_slices = _canonical_middle_slices(middle_slices, length=length)
    if axis == "x":
        around = tuple(
            (middle, other)
            for middle in middle_slices
            for other in range(other_length)
        )
    else:
        around = tuple(
            (other, middle)
            for middle in middle_slices
            for other in range(other_length)
        )
    return around


def _canonical_ctmrg_mode(mode):
    """Normalize the finite-CTMRG boundary compression selector."""
    key = str(mode).strip().lower().replace("_", "-")
    if key == "projector-2d":
        key = "projector2d"
    if key not in _CTMRG_MODES:
        choices = ", ".join(repr(name) for name in sorted(_CTMRG_MODES))
        raise ValueError(f"ctmrg_mode must be one of {choices}; got {mode!r}.")
    return key


def _canonical_ctmrg_canonize(canonize, *, mode):
    """Resolve mode-aware CTMRG canonicalization without ignored choices."""
    if canonize is None:
        return mode != "projector2d"
    if isinstance(canonize, bool):
        if mode == "projector2d" and canonize:
            raise ValueError(
                "ctmrg_canonize is not used by ctmrg_mode='projector2d'; "
                "leave it as None or set it to False."
            )
        return canonize
    if isinstance(canonize, str):
        key = canonize.strip().lower().replace("_", "-")
        if key in {"layered", "bp"} and mode == "projector":
            return key
    raise ValueError(
        "ctmrg_canonize must be None or a bool; ctmrg_mode='projector' "
        "also accepts 'layered' and 'bp'."
    )


def _canonical_ctmrg_projector_region(region, *, mode):
    """Normalize the local CTMRG projector environment shape."""
    if region is None:
        return None
    if isinstance(region, str):
        key = region.strip().lower().replace(" ", "").replace("×", "x")
        if key in {"2x2", "2*2"}:
            region = (2, 2)
        elif key in {"2x3", "2*3"}:
            region = (2, 3)
    try:
        region = tuple(int(size) for size in region)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ctmrg_projector_region must be None, (2, 2), or (2, 3)."
        ) from exc
    if region not in {(2, 2), (2, 3)}:
        raise ValueError(
            "ctmrg_projector_region must be None, (2, 2), or (2, 3); "
            f"got {region!r}."
        )
    if mode == "l2bp":
        raise ValueError(
            "ctmrg_projector_region applies only to projector CTMRG modes, "
            "not ctmrg_mode='l2bp'."
        )
    if mode == "projector2d" and region != (2, 2):
        raise ValueError(
            "ctmrg_mode='projector2d' supports only its native (2, 2) "
            "projector region; use ctmrg_mode='projector' for (2, 3)."
        )
    return region


def _copy_ctmrg_mapping(value, *, name):
    """Copy an optional CTMRG option mapping with a precise API error."""
    if value is None:
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a mapping or None.") from exc


def _parsed_quimb_site_tag(site, index):
    """Predict Quimb's tag for one parsed 1D site or grouped site."""
    if isinstance(site, str):
        return site
    site = tuple(site)
    if len(site) == 1 and isinstance(site[0], str):
        return site[0]
    return f"__GST{index}__"


@contextmanager
def _quimb_projector_region_2x3(site_tags):
    """Expand each local 1D projector environment to three sites.

    Quimb's generic projector compressor inserts projectors across pairs of
    neighboring effective sites. This scoped adapter adds one adjacent site
    to alternating sides of those environments, preserving the central cut
    and all tags attached to the inserted projectors.
    """
    try:
        from quimb.tensor import tensor_core as qtc
    except ImportError:  # pragma: no cover - Quimb is an optional dependency
        yield
        return

    tensor_network = getattr(qtc, "TensorNetwork", None)
    original = getattr(tensor_network, "insert_compressor_between_regions", None)
    original_inplace = getattr(
        tensor_network,
        "insert_compressor_between_regions_",
        None,
    )
    if tensor_network is None or original is None:
        yield
        return

    ordered_tags = tuple(
        _parsed_quimb_site_tag(site, index)
        for index, site in enumerate(site_tags)
    )
    positions = {tag: index for index, tag in enumerate(ordered_tags)}
    stats = {"seen": 0, "expanded": 0}

    @wraps(original)
    def insert_regional_projector(self, ltags, rtags, *args, **kwargs):
        stats["seen"] += 1
        ltags = [ltags] if isinstance(ltags, str) else list(ltags)
        rtags = [rtags] if isinstance(rtags, str) else list(rtags)

        if len(ltags) == len(rtags) == 1:
            lpos = positions.get(ltags[0])
            rpos = positions.get(rtags[0])
            if lpos is not None and rpos is not None and abs(lpos - rpos) == 1:
                lower = min(lpos, rpos)
                upper = max(lpos, rpos)
                # Alternate which side supplies the third site, avoiding a
                # systematic left/right bias and falling back at boundaries.
                candidates = (upper + 1, lower - 1)
                if lower % 2:
                    candidates = tuple(reversed(candidates))
                extra = next(
                    (pos for pos in candidates if 0 <= pos < len(ordered_tags)),
                    None,
                )
                if extra is not None:
                    if extra < lower:
                        target = ltags if lpos == lower else rtags
                    else:
                        target = ltags if lpos == upper else rtags
                    target.append(ordered_tags[extra])
                    stats["expanded"] += 1

        return original(self, ltags, rtags, *args, **kwargs)

    @wraps(original_inplace or original)
    def insert_regional_projector_inplace(self, ltags, rtags, *args, **kwargs):
        kwargs.setdefault("inplace", True)
        return insert_regional_projector(self, ltags, rtags, *args, **kwargs)

    tensor_network.insert_compressor_between_regions = insert_regional_projector
    if original_inplace is not None:
        tensor_network.insert_compressor_between_regions_ = (
            insert_regional_projector_inplace
        )
    try:
        yield
        if stats["seen"] and not stats["expanded"] and len(ordered_tags) >= 3:
            raise RuntimeError(
                "Quimb's projector topology did not expose a neighboring "
                "three-site region for ctmrg_projector_region=(2, 3)."
            )
    finally:
        tensor_network.insert_compressor_between_regions = original
        if original_inplace is not None:
            tensor_network.insert_compressor_between_regions_ = original_inplace


def _ctmrg_regional_projector_compressor(
    tn,
    *,
    max_bond=None,
    cutoff=1.0e-10,
    site_tags=None,
    canonize=True,
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    inplace=False,
    **kwargs,
):
    """Run Quimb's projector compressor with a local 2x3 environment."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    if site_tags is None:
        site_tags = tn.site_tags
    with _quimb_projector_region_2x3(site_tags):
        return qtn.tensor_network_1d_compress(
            tn,
            max_bond=max_bond,
            cutoff=cutoff,
            method="projector",
            site_tags=site_tags,
            canonize=canonize,
            permute_arrays=permute_arrays,
            optimize=optimize,
            sweep_reverse=sweep_reverse,
            equalize_norms=equalize_norms,
            inplace=inplace,
            **kwargs,
        )


@contextmanager
def quimb_ctmrg_projector_compat():
    """Use current-network projectors for Quimb's cyclic CTMRG path.

    Some Quimb versions compute a row of CTMRG projectors from one snapshot
    while inserting those projectors into a network that is modified after
    each insertion. On cyclic networks with non-uniform effective bond
    dimensions this can leave the projector shapes out of sync with the
    current tensor network. Projectors are therefore computed from a fresh
    copy of the live insertion network, with a fresh fermionic dummy-mode
    namespace, and inserted back into that live network. Native zero-charge
    projector sectors also receive finite identity blocks.

    The patch is scoped to this context and does not modify the installed
    Quimb package or affect boundary-MPS contractions.
    """
    try:
        from quimb.tensor import tensor_core as qtc
    except ImportError:  # pragma: no cover - Quimb is an optional dependency
        yield
        return

    tensor_network = getattr(qtc, "TensorNetwork", None)
    original = getattr(
        tensor_network,
        "insert_compressor_between_regions",
        None,
    )
    original_inplace = getattr(
        tensor_network,
        "insert_compressor_between_regions_",
        None,
    )
    original_oblique = getattr(qtc, "compute_oblique_projectors", None)
    original_reduced_factor = getattr(
        qtc,
        "squared_op_to_reduced_factor",
        None,
    )
    if tensor_network is None or original is None:
        yield
        return

    is_fermionic = getattr(
        qtc,
        "isfermionic",
        lambda value: bool(getattr(value, "fermionic", False)),
    )

    @wraps(original)
    def insert_current_projector(self, ltags, rtags, *args, insert_into=None, **kwargs):
        if insert_into is None:
            return original(self, ltags, rtags, *args, **kwargs)

        # Quimb's CTMRG caller passes a calculation copy as ``self`` and the
        # live, progressively modified network as ``insert_into``. Recompute
        # from a fresh copy of the live network so bond ids and sizes match,
        # while separating the squared-environment dummy-mode namespace.
        current_kwargs = dict(kwargs)
        current_kwargs.pop("inplace", None)
        projector_source = _freshen_fermionic_dummy_modes(insert_into)
        return original(
            projector_source,
            ltags,
            rtags,
            *args,
            insert_into=insert_into,
            inplace=True,
            **current_kwargs,
        )

    @wraps(original_inplace or original)
    def insert_current_projector_inplace(
        self,
        ltags,
        rtags,
        *args,
        insert_into=None,
        **kwargs,
    ):
        # Quimb's arbitrary-geometry compression path calls the underscore
        # alias, whose partialmethod otherwise retains the unpatched original
        # implementation and recomputes projectors on a stale snapshot.
        kwargs.setdefault("inplace", True)
        return insert_current_projector(
            self,
            ltags,
            rtags,
            *args,
            insert_into=insert_into,
            **kwargs,
        )

    tensor_network.insert_compressor_between_regions = insert_current_projector
    if original_inplace is not None:
        tensor_network.insert_compressor_between_regions_ = (
            insert_current_projector_inplace
        )

    if callable(original_oblique):
        oblique_globals = original_oblique.__globals__

        @wraps(original_oblique)
        def compute_fermionic_oblique_projectors(
            Rl,
            Rr,
            max_bond=None,
            cutoff=0.0,
            absorb="both",
            cutoff_mode="rsum2",
            method="svd",
            **compress_opts,
        ):
            """Avoid zero-sector inverses in native fermionic projectors."""
            if not (is_fermionic(Rl) or is_fermionic(Rr)) or absorb != "both":
                return original_oblique(
                    Rl,
                    Rr,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    absorb=absorb,
                    cutoff_mode=cutoff_mode,
                    method=method,
                    **compress_opts,
                )

            Ut, st, VHt = oblique_globals["array_split"](
                Rl @ Rr,
                max_bond=-1 if max_bond is None else max_bond,
                cutoff=cutoff,
                absorb=None,
                cutoff_mode=cutoff_mode,
                method=method,
                **compress_opts,
            )

            # Quimb's default projector uses 1 / sqrt(s). For a sparse
            # symmetry sector with s == 0, the mathematically appropriate
            # pseudoinverse is zero rather than inf or NaN.
            inverse_sqrt = st.copy()
            zero_tol = 1.0e-12

            def _inverse_sqrt(block):
                out = ar.do("zeros_like", block)
                mask = ar.do("abs", block) > zero_tol
                out[mask] = 1.0 / ar.do("sqrt", block[mask])
                return out

            inverse_sqrt.apply_to_arrays(_inverse_sqrt)
            Pl = Rr @ oblique_globals["rdmul"](
                oblique_globals["dag"](VHt), inverse_sqrt
            )
            Pr = oblique_globals["ldmul"](
                inverse_sqrt,
                oblique_globals["dag"](Ut),
            ) @ Rl

            # If an entire charge block is empty, both projectors above are
            # zero in that block. Keep a finite identity block so subsequent
            # simple-gauge/canonicalization steps do not divide by zero.
            for sector, block in st.get_sector_block_pairs():
                if not bool(
                    ar.do(
                        "all",
                        ar.do("abs", block) <= zero_tol,
                    )
                ):
                    continue
                for projector in (Pl, Pr):
                    for key, projector_block in projector.get_sector_block_pairs():
                        if sector not in key:
                            continue
                        identity = ar.do("zeros_like", projector_block)
                        for i in range(min(identity.shape)):
                            identity[i, i] = 1.0
                        projector.set_block(key, identity)

            return Pl, Pr

        qtc.compute_oblique_projectors = compute_fermionic_oblique_projectors

    if callable(original_reduced_factor):
        reduced_impl = getattr(original_reduced_factor, "__wrapped__", None)
        reduced_globals = getattr(
            original_reduced_factor,
            "__globals__",
            getattr(reduced_impl, "__globals__", {}),
        )
        reduced_dag = reduced_globals.get("dag")

        @wraps(original_reduced_factor)
        def compute_fermionic_reduced_factor(
            x2,
            dl,
            dr,
            right=True,
            method="eigh",
            **reduce_opts,
        ):
            if not is_fermionic(x2):
                return original_reduced_factor(
                    x2,
                    dl,
                    dr,
                    right=right,
                    method=method,
                    **reduce_opts,
                )

            _assert_finite_symmray_blocks(x2, name="squared environment")
            if callable(reduced_dag):
                # Remove tiny anti-Hermitian roundoff before eigh. This is
                # deliberately after the finite check: symmetrization must
                # never hide a NaN or Inf produced upstream.
                x2 = 0.5 * (x2 + reduced_dag(x2))
                _assert_finite_symmray_blocks(
                    x2,
                    name="symmetrized squared environment",
                )

            return original_reduced_factor(
                x2,
                dl,
                dr,
                right=right,
                method=method,
                **reduce_opts,
            )

        qtc.squared_op_to_reduced_factor = compute_fermionic_reduced_factor
    try:
        yield
    finally:
        tensor_network.insert_compressor_between_regions = original
        if original_inplace is not None:
            tensor_network.insert_compressor_between_regions_ = original_inplace
        if callable(original_oblique):
            qtc.compute_oblique_projectors = original_oblique
        if callable(original_reduced_factor):
            qtc.squared_op_to_reduced_factor = original_reduced_factor


@contextmanager
def _quimb_ctmrg_mode_forwarding_compat(mode):
    """Filter Quimb CTMRG keywords that do not belong to a submode.

    Some Quimb releases route every top-level ``contract_ctmrg`` option into
    every boundary compressor. ``projector2d`` and ``l2bp`` then receive
    projector-only keywords and fail before doing any work. Keep the adapter
    scoped to the call and preserve all options each concrete mode supports.
    """
    try:
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - Quimb is an optional dependency
        yield
        return

    owner = getattr(qtn, "TensorNetwork2D", None)
    if owner is None:
        yield
        return

    if mode == "projector2d":
        name = "_contract_boundary_projector"
        original = getattr(owner, name, None)
        if not callable(original):
            yield
            return

        @wraps(original)
        def call_projector2d(self, *args, **kwargs):
            kwargs.pop("canonize_opts", None)
            return original(self, *args, **kwargs)

        replacement = call_projector2d
    elif mode == "l2bp":
        name = "_contract_boundary_core_via_1d"
        original = getattr(owner, name, None)
        if not callable(original):
            yield
            return

        @wraps(original)
        def call_l2bp(self, *args, **kwargs):
            selected = kwargs.get("method")
            if selected is None and len(args) > 5:
                selected = args[5]
            if selected == "l2bp":
                kwargs = dict(kwargs)
                for key in (
                    "canonize_opts",
                    "contract_opts",
                    "lazy",
                    "reduce_opts",
                ):
                    kwargs.pop(key, None)
            return original(self, *args, **kwargs)

        replacement = call_l2bp
    else:
        yield
        return

    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


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
    fit_diagnostics: tuple[BoundaryFitDiagnostic, ...] = ()


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


def _uses_symmray_arrays(tn):
    """Return whether a tensor network stores Symmray block-sparse arrays."""
    tensor_map = getattr(tn, "tensor_map", None)
    tensors = tensor_map.values() if tensor_map else tn
    try:
        iterator = iter(tensors)
    except TypeError:
        iterator = ()
    for tensor in iterator:
        data = getattr(tensor, "data", None)
        if data is None:
            continue
        if type(data).__module__.split(".", maxsplit=1)[0] == "symmray":
            return True
        if hasattr(data, "blocks") and hasattr(data, "apply_to_arrays"):
            return True
    return False


def _to_python_scalar(value):
    """Convert backend scalar-like objects (torch/numpy) to python scalar."""
    if isinstance(value, (int, float, complex, bool)):
        return value
    try:
        obj = ar.to_numpy(value)
    except Exception:
        obj = value
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
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


def _is_finite_scaled_scalar(value):
    mantissa, exponent = _as_scaled_scalar(value)
    try:
        mantissa = complex(mantissa)
        exponent = float(exponent)
    except (OverflowError, TypeError, ValueError):
        return False
    return (
        math.isfinite(mantissa.real)
        and math.isfinite(mantissa.imag)
        and math.isfinite(exponent)
    )


def _ensure_finite_norm_value(norm_value):
    if not _is_finite_scaled_scalar(norm_value):
        raise ValueError(
            "Boundary norm cost is not finite; cannot normalize state. "
            "Use strip_exponent=True for very large or very small norms, "
            "or inspect the state for NaN/Inf tensor entries."
        )


def _warn_retry_stripped_norm(reason):
    warnings.warn(
        "Boundary norm cost was "
        f"{reason}; retrying with strip_exponent=True. Pass "
        "strip_exponent=True to receive the stable (mantissa, exponent) "
        "old norm directly.",
        RuntimeWarning,
        stacklevel=3,
    )


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
    _ensure_finite_norm_value(norm_value)
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
    if getattr(fn, "__name__", "") in {"contract_boundary", "contract_ctmrg"}:
        kwargs = quimb_2d_options(fn, kwargs)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**kwargs)

    accepted = {key: val for key, val in kwargs.items() if key in sig.parameters}
    return fn(**accepted)


def _finish_quimb_around_contraction(reduced, *, final_contract_opts, method):
    """Exactly contract a network reduced around a protected middle slab."""
    contract_fn = getattr(reduced, "contract", None)
    if not callable(contract_fn):
        raise TypeError(
            f"method={method!r} did not return a tensor network when "
            "contracting around the middle slab."
        )
    return contract_fn(all, **final_contract_opts)


def _require_quimb_around_support(contract_fn, *, method):
    """Require the opt-in Quimb protected-region contraction API."""
    try:
        parameters = inspect.signature(contract_fn).parameters.values()
    except (TypeError, ValueError):
        return
    if any(parameter.name == "around" for parameter in parameters):
        return
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return
    raise NotImplementedError(
        f"The installed Quimb build does not support around=... for "
        f"method={method!r}; upgrade Quimb to use middle-out contraction."
    )


def _ctmrg_stabilization_kwargs(
    norm,
    *,
    reduce_opts=None,
    gauge_smudge=None,
    canonize_opts=None,
    projector_gauges=True,
):
    """Prepare numerically safer CTMRG projector options.

    Symmray environments can be very ill-conditioned, especially after the
    squared environment used by Quimb's oblique projector construction.  For
    those networks, add a small positive shift before the Hermitian
    factorization and a small gauge smudge by default.  Dense networks retain
    Quimb's existing defaults unless the caller explicitly supplies options.
    """
    reduce_opts = _copy_ctmrg_mapping(
        reduce_opts,
        name="ctmrg_reduce_opts",
    )
    canonize_opts = _copy_ctmrg_mapping(
        canonize_opts,
        name="ctmrg_canonize_opts",
    )

    symmray = _uses_symmray_arrays(norm)
    if symmray:
        reduce_opts.setdefault("method", "eigh")
        reduce_opts.setdefault("shift", 1.0e-12)
        if projector_gauges and gauge_smudge is None:
            gauge_smudge = 1.0e-10

    kwargs = {}
    if reduce_opts:
        kwargs["reduce_opts"] = reduce_opts
    if projector_gauges and gauge_smudge is not None:
        kwargs["gauge_smudge"] = gauge_smudge
    if symmray and projector_gauges and gauge_smudge is not None:
        # ``gauge_smudge`` only reaches projector construction in Quimb. The
        # native fermionic path also needs the same floor during the preceding
        # simple-gauge normalization, before any reduced-factor decomposition.
        canonize_opts.setdefault("smudge", gauge_smudge)
    if canonize_opts:
        kwargs["canonize_opts"] = canonize_opts
    return kwargs


def _freshen_fermionic_dummy_modes(tn):
    """Copy ``tn`` with a fresh namespace for Symmray dummy modes.

    Quimb forms a squared environment by conjugating a calculation copy and
    contracting it with the original. A double-layer fermionic network can
    already contain matching odd dummy modes, so reusing their labels in the
    calculation copy creates duplicate same-dual modes before Symmray can
    cancel conjugate pairs. Renaming each tensor's dummy modes gives every
    copied mode a unique label, while its conjugate receives the matching
    dual label during the squared-environment construction.
    """
    if not hasattr(tn, "copy"):
        return tn

    source = tn.copy()
    namespace = id(source)
    for tensor in source:
        data = getattr(tensor, "data", None)
        dummy_modes = getattr(data, "dummy_modes", ())
        if not dummy_modes or not hasattr(data, "modify"):
            continue

        fresh_modes = tuple(
            type(mode)(
                ("__pepsy_ctmrg_dummy__", namespace, id(tensor), mode_i),
                dual=mode.dual,
                parity=mode.parity,
            )
            for mode_i, mode in enumerate(dummy_modes)
        )
        data = data.copy()
        data.modify(dummy_modes=fresh_modes)
        tensor.modify(data=data)
    return source


def _assert_finite_symmray_blocks(value, *, name):
    """Raise before decomposition if a native block already is non-finite."""
    blocks = getattr(value, "blocks", None)
    if blocks is None:
        try:
            finite = bool(ar.do("all", ar.do("isfinite", value)))
        except (TypeError, ValueError):
            return
        if not finite:
            raise FloatingPointError(
                f"Non-finite values entered the native CTMRG {name}."
            )
        return

    for sector, block in blocks.items():
        try:
            finite = bool(ar.do("all", ar.do("isfinite", block)))
        except (TypeError, ValueError):
            continue
        if not finite:
            raise FloatingPointError(
                "Non-finite values entered the native CTMRG "
                f"{name} block for charge sector {sector!r}."
            )


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


def _retune_bdy_to_chi(obj, chi, name, *, expand_growth=True):
    """Retune an existing boundary object to the requested chi.

    Two-site FIT discovers rank through its local SVDs, so ``expand_growth``
    can be disabled to avoid globally padding a warm boundary before every
    contraction. A lower requested cap is always applied immediately.
    """
    if obj is None or chi is None:
        return
    cur = getattr(obj, "chi", None)
    if cur is None or int(cur) == chi:
        return
    if int(cur) < chi and not expand_growth:
        # Preserve a low-rank warm start. The caller passes ``chi`` directly
        # to the two-site split, which can populate useful sectors naturally.
        return
    if not hasattr(obj, "expand_bnd"):
        raise TypeError(
            f"{name} has chi={cur} but cannot be retuned to chi={chi}; "
            "object must expose method 'expand_bnd'."
        )
    obj.expand_bnd(chi, inplace=True)


def _contract_quimb_double_layer(  # pylint: disable=R0912,R0913,R0914,R0915
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
    around=None,
    cutoff,
    equalize_norms,
    layer_tags,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
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
        if around is not None:
            _require_quimb_around_support(contract_fn, method=method)
        sequence = _default_quimb_sequence(norm, method) if sequence is None else sequence
        kwargs = dict(
            max_bond=chi,
            sequence=sequence,
            final_contract_opts=final_contract_opts,
            cutoff=cutoff,
            canonize=True,
            progbar=progress,
            max_separation=max_separation,
            equalize_norms=equalize_norms,
            inplace=False,
        )
        if around is not None:
            kwargs["around"] = around
            kwargs["final_contract"] = False
        if mode_ is not None:
            kwargs["mode"] = mode_
        if layer_tags is not None:
            kwargs["layer_tags"] = list(layer_tags)
        contracted = _call_with_accepted_kwargs(contract_fn, **kwargs)
        if around is not None:
            return _finish_quimb_around_contraction(
                contracted,
                final_contract_opts=final_contract_opts,
                method=method,
            )
        return contracted

    if method == "ctmrg":
        contract_fn = getattr(norm, "contract_ctmrg", None)
        if not callable(contract_fn):
            raise TypeError("method='ctmrg' requires a network with contract_ctmrg().")
        if around is not None:
            _require_quimb_around_support(contract_fn, method=method)
        ctmrg_mode = _canonical_ctmrg_mode(ctmrg_mode)
        if ctmrg_mode != "projector":
            require_quimb_ctmrg_mode(ctmrg_mode)
        ctmrg_canonize = _canonical_ctmrg_canonize(
            ctmrg_canonize,
            mode=ctmrg_mode,
        )
        ctmrg_projector_region = _canonical_ctmrg_projector_region(
            ctmrg_projector_region,
            mode=ctmrg_mode,
        )
        if isinstance(ctmrg_canonize, str):
            require_quimb_ctmrg_projector_canonize(ctmrg_canonize)
        ctmrg_canonize_opts = _copy_ctmrg_mapping(
            ctmrg_canonize_opts,
            name="ctmrg_canonize_opts",
        )
        ctmrg_compress_opts = _copy_ctmrg_mapping(
            ctmrg_compress_opts,
            name="ctmrg_compress_opts",
        )

        if ctmrg_projector_region == (2, 3):
            require_quimb_ctmrg_mode("projector")
            require_quimb_1d_callable_compression()
            if _infer_lattice_ndim(norm) == 3:
                raise NotImplementedError(
                    "ctmrg_projector_region=(2, 3) currently supports only "
                    "finite 2D tensor networks."
                )
            for axis in ("x", "y"):
                is_cyclic = getattr(norm, f"is_cyclic_{axis}", None)
                if callable(is_cyclic) and is_cyclic():
                    raise NotImplementedError(
                        "ctmrg_projector_region=(2, 3) currently supports "
                        "only open boundary conditions."
                    )

        if _uses_symmray_arrays(norm) and not (
            ctmrg_mode == "projector"
            and ctmrg_canonize in {True, "layered"}
            and ctmrg_projector_region != (2, 3)
        ):
            raise NotImplementedError(
                "Native Symmray CTMRG currently supports only "
                "ctmrg_mode='projector' with ctmrg_canonize=True or "
                "'layered' and the native projector region; projector2d, "
                "l2bp, BP gauging, the 2x3 region, and disabled gauging are "
                "dense-only."
            )

        if ctmrg_mode == "l2bp":
            reduce_opts = _copy_ctmrg_mapping(
                ctmrg_reduce_opts,
                name="ctmrg_reduce_opts",
            )
            if reduce_opts or ctmrg_gauge_smudge is not None:
                raise ValueError(
                    "ctmrg_reduce_opts and ctmrg_gauge_smudge apply only to "
                    "projector CTMRG modes, not ctmrg_mode='l2bp'."
                )
            if ctmrg_canonize_opts:
                raise ValueError(
                    "ctmrg_canonize_opts applies only to "
                    "ctmrg_mode='projector'; configure l2bp through "
                    "ctmrg_compress_opts."
                )
            stabilization = {}
        elif ctmrg_mode == "projector2d":
            if ctmrg_canonize_opts:
                raise ValueError(
                    "ctmrg_canonize_opts is not used by "
                    "ctmrg_mode='projector2d'."
                )
            if ctmrg_gauge_smudge is not None:
                raise ValueError(
                    "ctmrg_gauge_smudge is not used by "
                    "ctmrg_mode='projector2d'."
                )
            stabilization = _ctmrg_stabilization_kwargs(
                norm,
                reduce_opts=ctmrg_reduce_opts,
                projector_gauges=False,
            )
        else:
            if ctmrg_canonize is False and ctmrg_canonize_opts:
                raise ValueError(
                    "ctmrg_canonize_opts is not used when "
                    "ctmrg_canonize=False."
                )
            stabilization = _ctmrg_stabilization_kwargs(
                norm,
                reduce_opts=ctmrg_reduce_opts,
                gauge_smudge=ctmrg_gauge_smudge,
                canonize_opts=ctmrg_canonize_opts,
            )

        sequence = _default_quimb_sequence(norm, method) if sequence is None else sequence
        quimb_ctmrg_mode = ctmrg_mode
        if ctmrg_projector_region == (2, 3):
            quimb_ctmrg_mode = _ctmrg_regional_projector_compressor
        kwargs = dict(
            max_bond=chi,
            cutoff=cutoff,
            canonize=ctmrg_canonize,
            mode=quimb_ctmrg_mode,
            sequence=sequence,
            max_separation=max_separation,
            equalize_norms=equalize_norms,
            optimize=contraction_opt,
            final_contract=around is None,
            final_contract_opts=final_contract_opts,
            progbar=progress,
            inplace=False,
        )
        if around is not None:
            kwargs["around"] = around
        kwargs.update(stabilization)
        if ctmrg_compress_opts:
            kwargs["compress_opts"] = ctmrg_compress_opts
        if layer_tags is not None:
            kwargs["layer_tags"] = list(layer_tags)

        def contract_ctmrg():
            with _quimb_ctmrg_mode_forwarding_compat(ctmrg_mode):
                return _call_with_accepted_kwargs(contract_fn, **kwargs)

        if ctmrg_mode in {"projector", "projector2d"}:
            with quimb_ctmrg_projector_compat():
                contracted = contract_ctmrg()
        else:
            contracted = contract_ctmrg()
        if around is not None:
            return _finish_quimb_around_contraction(
                contracted,
                final_contract_opts=final_contract_opts,
                method=method,
            )
        return contracted

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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    around=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    bdy_name="bdy",
    flat=False,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
):
    """Contract a double-layer PEPS norm/overlap network by the selected method."""
    method = _normalize_contraction_method(method)
    if fit_compression_opts and method != "dmrg":
        raise ValueError("fit_compression_opts requires method='dmrg' with a Quimb fit_mode.")
    fit_compression_opts = quimb_compression_options(
        _canonical_fit_mode_selector(fit_mode), fit_compression_opts
    )
    chi = _validate_chi(chi)
    fit_layer_mode = _canonical_fit_layer_mode(fit_layer_mode)
    fit_layer_order = _canonical_fit_layer_order(fit_layer_order)
    if (
        fit_layer_mode == "sequential"
        and fit_layer_order == "auto"
        and layer_tags is None
    ):
        raise ValueError(
            "fit_layer_order='auto' requires explicit layer_tags; the "
            "default BRA/KET order is semantically significant."
        )
    if layer_tags is None and not flat:
        layer_tags = _DEFAULT_LAYER_TAGS
    elif isinstance(layer_tags, str):
        layer_tags = (layer_tags,)
    elif layer_tags is not None:
        layer_tags = tuple(layer_tags)
    if fit_layer_mode == "sequential" and method != "dmrg":
        raise ValueError(
            "fit_layer_mode='sequential' requires method='dmrg'; "
            "method='mps' uses Quimb's native layer_tags handling."
        )

    if method == "dmrg":
        fit_mode = _canonical_fit_mode_selector(fit_mode)
        bdy_obj, bdy_holder = _unpack_bdy_handle(bdy, bdy_name)
        direct_compression = fit_mode in _FIT_QUIMB_MODES
        block_fit = fit_mode in {"two-site", "dmrg2"} or (
            fit_mode == "eff" and fit_block_size in {2, 3}
        )
        low_rank_start = direct_compression or block_fit
        _retune_bdy_to_chi(
            bdy_obj,
            chi,
            bdy_name,
            expand_growth=not low_rank_start,
        )
        if bdy_obj is None:
            if chi is None:
                raise ValueError(f"Provide chi when {bdy_name} is not supplied.")
            # Block-SVD fits discover rank as they sweep, while direct Quimb
            # modes replace each boundary without consuming an initial guess.
            # A product boundary avoids allocating and canonicalizing random
            # chi-wide bonds in either case.
            initial_chi = 1 if low_rank_start else chi
            if flat:
                bdy_obj = BdyMPS(
                    tn_flat=norm,
                    chi=initial_chi,
                    flat=True,
                    single_layer=single_layer,
                    lazy=True,
                )
            else:
                bdy_obj = BdyMPS(
                    tn_double=norm,
                    chi=initial_chi,
                    single_layer=single_layer,
                    lazy=True,
                )
            if bdy_holder is not None:
                bdy_holder["bdy"] = bdy_obj

        result = contract_boundary(
            norm=norm,
            bdy=bdy_obj,
            contraction_opt=contraction_opt,
            fit_mode=fit_mode,
            fit_layer_mode=fit_layer_mode,
            fit_layer_order=fit_layer_order,
            fit_init_strategy=fit_init_strategy,
            fit_init_seed=fit_init_seed,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_max_bond=chi if fit_max_bond is None else fit_max_bond,
            fit_sweep_sequence=fit_sweep_sequence,
            fit_cutoff=cutoff,
            fit_cutoff_mode=fit_cutoff_mode,
            fit_compression_opts=fit_compression_opts,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_timing=fit_timing,
            fit_timing_sync_device=fit_timing_sync_device,
            n_iter=n_iter,
            progress=progress,
            direction=direction,
            max_separation=max_separation,
            track_boundary_fidelity=track_boundary_fidelity,
            layer_tags=layer_tags,
            visualize=visualize,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
            flat=flat,
        )
        return result, bdy_obj

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
        around=around,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
    )
    return BoundaryContractResult(
        cost=cost,
        fidel=[],
        direction=direction,
        n_iter=n_iter,
        max_separation=max_separation,
    ), None


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
    fit_layer_mode="joint",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    visualize=False,
    strip_exponent=False,
    return_info=False,
    preserve_backend=False,
    mode_=None,
    compression_mode=None,
    sequence=None,
    boundary_direction=None,
    middle_slices=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
):
    """Contract one already-flattened effective PEPS-like layer.

    This helper is for a single lattice layer whose local bra/ket or operator
    contractions have already been performed. It does not call
    :func:`build_bra_ket`, and ``flat=True`` selects the direct first-slice
    boundary initialization in :class:`~pepsy.boundary.states.BdyMPS`.
    ``tn`` may still have ordinary lattice boundary indices, but it must not be
    interpreted as a stack of separately tagged ``BRA``/``PEPO``/``KET``
    layers. Use :func:`contract_boundary` with ``flat=False`` and a
    ``BdyMPS(tn_double=...)`` for an explicitly layered target.

    With ``method="auto"`` this uses the PEPSY DMRG/FIT boundary path for 2D
    networks and Quimb's ``contract_boundary`` path for 3D networks. Explicit
    methods are ``"dmrg"``, ``"mps"``, ``"ctmrg"``, ``"hotrg"``, and
    ``"exact"``.

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
    fit_mode : {"direct", "src", "src-mps", "zipup", "sdc", "sdcr", "dm", "eff", "two-site", "dmrg", "dmrg1", "dmrg2", "global"}, default="eff"
        Boundary compression mode. Quimb also accepts the supported
        ``*-first``/``*-oversample`` variants and ``src-mps`` aliases
        ``srcmps``. The direct modes compress each
        boundary target; ``"dmrg"`` and ``"dmrg1"`` alias ``"eff"``, while
        ``"dmrg2"`` uses two-site warm-up followed by one-site refinement.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        Must remain ``"joint"`` for this single-effective-layer API.
        Sequential layer absorption belongs to the multilayer
        :func:`contract_boundary` path.
    layer_tags : sequence[str] | None, default=None
        Optional layer tags forwarded to Quimb's native contraction methods.
        They do not turn this single-layer API into a multilayer contraction.
    ctmrg_mode : {"projector", "projector2d", "l2bp"}, default="projector"
        Quimb boundary compressor used by ``method="ctmrg"``. ``"projector"``
        preserves Pepsy's existing locally computed projector route;
        ``"projector2d"`` inserts explicit plaquette projectors, and
        ``"l2bp"`` uses lazy 2-norm belief-propagation compression.
    ctmrg_canonize : bool | {"layered", "bp"} | None, default=None
        Projector preconditioning. ``None`` preserves the existing
        ``True`` default for ``"projector"`` and enables local gauging for
        ``"l2bp"``. ``"layered"`` and ``"bp"`` apply only to
        ``ctmrg_mode="projector"``. ``"projector2d"`` does not use this
        option.
    ctmrg_projector_region : {(2, 2), (2, 3)} | str | None, default=None
        Local projector environment. ``None`` or ``(2, 2)`` preserves the
        native two-site route. ``(2, 3)`` expands each projector calculation
        to three neighboring boundary sites and is available for dense,
        open-boundary 2D networks with ``ctmrg_mode="projector"``.
    ctmrg_canonize_opts : mapping | None, default=None
        Options for ``ctmrg_mode="projector"`` canonicalization. With
        ``ctmrg_canonize="bp"`` these configure Quimb's dense D2BP solve.
    ctmrg_compress_opts : mapping | None, default=None
        Options for the selected Quimb CTMRG boundary compressor.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial boundary guess. ``"guess-src"`` applies Quimb
        SRC to a copy of each exact boundary target before FIT.
    fit_init_seed : int | None, default=0
        Seed forwarded to the disposable SRC guess.
    strip_exponent : bool, default=False
        If ``True``, return ``(mantissa, exponent)``.
    return_info : bool, default=False
        Return :class:`BoundaryContractResult` instead of only its ``cost``.
        The result includes per-boundary convergence diagnostics for DMRG.
    preserve_backend : bool, default=False
        Return the raw backend scalar (or raw ``(mantissa, exponent)`` pair)
        without converting it to Python numbers. Enable this for Torch/JAX
        autodiff through a flat contraction. The default preserves the
        reporting-oriented scalar API.
    compression_mode : str | None, default=None
        Compression kernel for ``method="mps"``, such as ``"direct"``,
        ``"dm"``, or ``"sdc"``. This is the readable spelling of the
        compatibility argument ``mode_``; do not supply both.
    boundary_direction : str | None, default=None
        Readable contraction schedule. One-sided choices are ``"bottom-up"``,
        ``"top-down"``, ``"left-to-right"``, and ``"right-to-left"``;
        two-/four-sided choices are ``"top-bottom"``, ``"bottom-top"``,
        ``"left-right"``, ``"right-left"``, and ``"four-sided"``. These
        map to Quimb boundary sequences for ``method="mps"`` or ``"ctmrg"``.
        ``"middle-out-x"`` and ``"middle-out-y"`` absorb the two opposing
        outer boundaries inward towards a protected central row or column.
        They support ``method="mps"`` (with direct compression by default) and
        ``method="ctmrg"``. ``"middle-out"`` takes its axis from ``direction``.
    middle_slices : int | sequence[int] | None, default=None
        Contiguous target row(s) or column(s) protected while the two outer
        boundaries are absorbed inward. The default is the central row for odd
        lengths and the central pair for even lengths. An interface can be
        targeted explicitly with ``middle_slices=(u_last, v_first)``.
    ctmrg_reduce_opts : mapping | None, default=None
        Optional options forwarded to Quimb's squared-environment
        factorization for ``method="ctmrg"``. Symmray networks receive
        ``method="eigh", shift=1e-12`` by default when this is omitted.
    ctmrg_gauge_smudge : float | None, default=None
        Relative regularization for CTMRG projector environments. Symmray
        networks default to ``1e-10`` when this is omitted.

    Returns
    -------
    complex | float | tuple[complex | float, float] | BoundaryContractResult
        Contraction scalar, optionally with stripped exponent, or the complete
        structured result when ``return_info=True``.
    """
    if tn is None:
        raise ValueError("tn must not be None.")

    fit_layer_mode = _canonical_fit_layer_mode(fit_layer_mode)
    if fit_layer_mode != "joint":
        raise ValueError(
            "contract_flat handles one already-flattened effective layer and "
            "requires fit_layer_mode='joint'; use contract_boundary with "
            "flat=False for multilayer sequential compression."
        )

    method = _normalize_flat_contraction_method(method, tn)
    if method == "dmrg" and _infer_lattice_ndim(tn) == 3:
        raise ValueError(
            "method='dmrg' is only supported for 2D flat tensor networks; "
            "use method='mps', 'ctmrg', 'hotrg', or 'exact' for 3D networks."
        )

    if compression_mode is not None:
        if mode_ is not None:
            raise ValueError("Supply only one of compression_mode and mode_.")
        if method != "mps":
            raise ValueError("compression_mode applies only to method='mps'.")
        mode_ = compression_mode

    if boundary_direction is not None and sequence is not None:
        raise ValueError("Supply only one of boundary_direction and sequence.")
    resolved_sequence, middle_axis = _canonical_flat_boundary_direction(
        boundary_direction,
        direction=direction,
    )
    if resolved_sequence is not None:
        if method not in {"mps", "ctmrg"}:
            raise ValueError(
                "boundary_direction presets require method='mps' or "
                "method='ctmrg'."
            )
        sequence = resolved_sequence
    if middle_slices is not None and middle_axis is None:
        raise ValueError(
            "middle_slices is used only with boundary_direction='middle-out', "
            "'middle-out-x', or 'middle-out-y'."
        )

    around = None
    if middle_axis is not None:
        if method not in {"mps", "ctmrg"}:
            raise ValueError(
                "middle-out contraction requires method='mps' or "
                "method='ctmrg'."
            )
        if _infer_lattice_ndim(tn) != 2:
            raise ValueError(
                "middle-out contraction currently supports only 2D networks."
            )
        if method == "mps":
            middle_mode = (
                "direct" if mode_ is None else str(mode_).strip().lower()
            )
            if middle_mode != "direct":
                raise ValueError(
                    "middle-out MPS contraction currently requires "
                    "compression_mode='direct'."
                )
            mode_ = "direct"
        sequence = (f"{middle_axis}min", f"{middle_axis}max")
        around = _flat_middle_around(
            tn,
            axis=middle_axis,
            middle_slices=middle_slices,
        )

    result, _ = _contract_peps_double_layer(
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
        fit_layer_mode=fit_layer_mode,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=False,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        around=around,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        bdy_name="bdy",
        flat=True,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
    )
    if middle_axis is not None:
        result = replace(result, direction=f"middle-out-{middle_axis}")
    cost = (
        result.cost
        if preserve_backend
        else _format_scaled_output(
            result.cost,
            strip_exponent=strip_exponent,
        )
    )
    if return_info:
        return replace(result, cost=cost)
    return cost


def _validate_layer_tags(layer_tags, tn):
    """Validate and canonicalize the explicit tags for ``contract_layered``."""
    if isinstance(layer_tags, str):
        tags = (layer_tags,)
    else:
        try:
            tags = tuple(str(tag) for tag in layer_tags)
        except TypeError as exc:
            raise TypeError(
                "layer_tags must be a string or sequence of strings."
            ) from exc

    if len(tags) < 2:
        raise ValueError(
            "contract_layered requires at least two distinct layer_tags."
        )
    if any(not tag for tag in tags):
        raise ValueError("layer_tags must contain only non-empty tags.")
    if len(set(tags)) != len(tags):
        raise ValueError("layer_tags must not contain duplicates.")

    network_tags = set(getattr(tn, "tags", ()))
    missing = tuple(tag for tag in tags if tag not in network_tags)
    if missing:
        missing_text = ", ".join(repr(tag) for tag in missing)
        raise ValueError(
            "contract_layered could not find layer tag(s) "
            f"{missing_text} in the supplied tensor network."
        )
    return tags


def contract_layered(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    tn,
    *,
    layer_tags,
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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    visualize=False,
    strip_exponent=False,
    return_info=False,
    cutoff=1.0e-12,
    equalize_norms=False,
):
    """Contract an explicitly tagged multilayer boundary network.

    This is the multilayer counterpart to :func:`contract_flat`. ``tn`` is
    expected to contain two or more distinct layer tags, supplied in the
    desired absorption order. The network is passed directly to the shared
    :class:`~pepsy.boundary.sweeps.CompBdy` engine with ``flat=False``; no
    giant flattened BRA--PEPO--KET tensor network is formed.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Preassembled, tagged multilayer tensor network.
    layer_tags : sequence[str]
        Distinct layer tags in absorption order, for example
        ``("BRA", "PEPO", "KET")``.
    chi : int | None, default=None
        Boundary bond dimension. Required when ``bdy`` is not supplied.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Optional reusable boundary handle.
    method : {"dmrg"}, default="dmrg"
        Layered networks use the Pepsy ``CompBdy`` engine. Native Quimb
        ``method="mps"`` remains available through :func:`contract_boundary`
        or the scalar metric helpers.
    fit_mode : {"direct", "src", "src-mps", "zipup", "sdc", "sdcr", "dm", "eff", "two-site", "dmrg", "dmrg1", "dmrg2", "global"}, default="eff"
        Boundary compression mode. Quimb also accepts the supported
        ``*-first``/``*-oversample`` variants and ``src-mps`` aliases
        ``srcmps``. Direct Quimb modes can be combined with
        ``fit_layer_mode="sequential"``; FIT modes use a joint target.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        Whether to compress all tagged layers jointly or absorb them one at a
        time in ``layer_tags`` order. Sequential mode is intended for the
        direct Quimb fit modes.
    fit_layer_order : {"input", "auto"}, default="input"
        Sequential absorption order. ``"input"`` preserves ``layer_tags``;
        ``"auto"`` estimates dense intermediate sizes and is only appropriate
        when the explicitly tagged layers are mathematically interchangeable.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial boundary guess used by FIT modes.
    strip_exponent : bool, default=False
        If ``True``, return the scalar as ``(mantissa, exponent)``.
    return_info : bool, default=False
        If ``True``, return :class:`BoundaryContractResult` with diagnostics.

    Notes
    -----
    Other parameters match :func:`contract_flat` and
    :func:`contract_boundary` where applicable.
    """
    if tn is None:
        raise ValueError("tn must not be None.")

    method = _normalize_contraction_method(method)
    if method != "dmrg":
        raise ValueError(
            "contract_layered uses method='dmrg' and the CompBdy engine; "
            "use contract_boundary for native Quimb method='mps'."
        )
    layer_tags = _validate_layer_tags(layer_tags, tn)
    fit_layer_mode = _canonical_fit_layer_mode(fit_layer_mode)
    fit_layer_order = _canonical_fit_layer_order(fit_layer_order)

    result, _ = _contract_peps_double_layer(
        tn,
        method="dmrg",
        chi=chi,
        bdy=bdy,
        contraction_opt=contraction_opt,
        n_iter=n_iter,
        direction=direction,
        max_separation=max_separation,
        progress=progress,
        track_boundary_fidelity=track_boundary_fidelity,
        fit_mode=fit_mode,
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=False,
        visualize=visualize,
        strip_exponent=strip_exponent,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        bdy_name="bdy",
        flat=False,
    )
    cost = _format_scaled_output(result.cost, strip_exponent=strip_exponent)
    if return_info:
        return replace(result, cost=cost)
    return cost


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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff=1.0e-12,
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
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
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    layer_tags=None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Approximate a scalar contraction using boundary-MPS sweeps.

    Parameters
    ----------
    norm : qtn.TensorNetwork
        Prebuilt 2D contraction network, usually the BRA--KET double layer
        returned by :func:`build_bra_ket`. Explicitly layered targets such as
        BRA--PEPO--KET may also be supplied here with ``flat=False``.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Boundary handle:

        - ``BdyMPS`` object: uses ``bdy.mps_b``
        - dict holder with ``{"bdy": <BdyMPS>}``: uses ``bdy["bdy"].mps_b``
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer passed through to :class:`pepsy.boundary.sweeps.CompBdy`.
    flat : bool, default=False
        Use the single-effective-layer initialization shortcut. Keep this
        ``False`` for a multi-layer target; use ``BdyMPS(tn_double=...)`` for
        the corresponding boundary object.
    fit_mode : {"direct", "src", "src-mps", "zipup", "sdc", "sdcr", "dm", "eff", "two-site", "dmrg", "dmrg1", "dmrg2", "global"}, default="eff"
        Boundary compression mode. Quimb also accepts the supported
        ``*-first``/``*-oversample`` variants and ``src-mps`` aliases
        ``srcmps``. The direct modes compress each
        boundary target; ``"dmrg"`` and ``"dmrg1"`` alias ``"eff"``, while
        ``"dmrg2"`` uses two-site warm-up followed by one-site refinement.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        Direct-mode layer policy. ``"joint"`` compresses all tagged layers
        together; ``"sequential"`` compresses them one at a time in the
        order given by ``layer_tags``. Only direct Quimb fit modes support the
        sequential policy.
    fit_layer_order : {"input", "auto"}, default="input"
        Sequential absorption order. ``"input"`` preserves ``layer_tags``;
        ``"auto"`` estimates dense intermediate sizes and requires explicit,
        mathematically interchangeable layer tags.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial boundary guess. ``"guess-src"`` applies Quimb
        SRC to a copy of each exact boundary target before FIT.
    fit_init_seed : int | None, default=0
        Seed forwarded to the disposable SRC guess.
    fit_block_size : {1, 2, 3}, default=1
        Block size used by ``FIT.run_eff`` when ``fit_mode="eff"``.
    fit_adaptive_sweeps : int | None, default=None
        For ``fit_mode="eff"`` and block size 2 or 3, number of initial
        block-SVD sweeps before one-site refinement. ``None`` keeps fixed
        block-size sweeps.
    layer_tags : sequence[str] | None, default=None
        Layer tags used for sequential direct compression, in absorption
        order. The standard two-layer order defaults to ``("KET", "BRA")``;
        supply tags such as ``("BRA", "PEPO", "KET")`` for three layers.
    fit_max_bond : int | None, default=None
        Two-site SVD cap. If omitted, use the boundary object's current bond.
        Higher-level PEPS helpers pass their requested ``chi`` explicitly.
    fit_sweep_sequence : str, default="RL"
        Repeating local-fit sweep directions. ``"RL"`` runs left-to-right
        and then right-to-left.
    fit_cutoff : float | {"auto"}, default=1e-12
        Boundary compression cutoff. ``"auto"`` selects the shared
        dtype-aware cutoff policy.
    fit_cutoff_mode : str | None | {"auto"}, default="auto"
        Quimb cutoff convention. ``"auto"`` and ``None`` resolve to
        ``"rsum2"``.
    fit_min_iter : int | None, default=None
        Minimum completed sweeps before adaptive stopping. ``FIT.run_eff``
        requires at least two when ``fit_rtol`` is enabled.
    fit_rtol : float | None, default=None
        Relative convergence tolerance. ``None`` runs exactly ``n_iter``
        sweeps, preserving legacy fixed-iteration behavior.
    fit_patience : int, default=1
        Consecutive converged sweeps required for adaptive stopping.
    fit_timing : bool, default=False
        Collect elapsed time and detailed two-site sweep timings in
        ``result.fit_diagnostics``.
    fit_timing_sync_device : bool, default=False
        Synchronize supported accelerators at FIT timing boundaries. Enable
        only for kernel-complete profiling because barriers add overhead.
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
        Structured contraction result with scalar ``cost``, optional fidelity
        history ``fidel``, and one typed convergence record per boundary fit.
    """
    if norm is None:
        raise ValueError("norm must not be None.")
    if bdy is None:
        raise ValueError("Provide bdy.")
    fit_layer_mode = _canonical_fit_layer_mode(fit_layer_mode)
    if flat and fit_layer_mode != "joint":
        raise ValueError(
            "flat=True handles one already-flattened effective layer and "
            "requires fit_layer_mode='joint'; use contract_layered or "
            "contract_boundary(..., flat=False) for multilayer compression."
        )
    if hasattr(bdy, "mps_b"):
        bdy_obj = bdy
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

    if fit_max_bond is None:
        fit_max_bond = getattr(bdy_obj, "chi", None)

    retag = bool(retag)
    norm_tagged = norm.copy()

    comp_bdy = CompBdy(
        norm_tagged,
        mps_boundaries,
        contraction_opt=contraction_opt,
        fit_mode=fit_mode,
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff=fit_cutoff,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        layer_tags=layer_tags,
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
        fit_diagnostics=tuple(getattr(comp_bdy, "fit_diagnostics", ())),
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
    fit_layer_mode,
    fit_layer_order,
    fit_init_strategy,
    fit_init_seed,
    fit_block_size,
    fit_adaptive_sweeps,
    fit_max_bond,
    fit_sweep_sequence,
    fit_cutoff_mode,
    fit_compression_opts,
    fit_min_iter,
    fit_rtol,
    fit_patience,
    fit_timing,
    fit_timing_sync_device,
    single_layer,
    visualize,
    strip_exponent,
    mode_,
    sequence,
    cutoff,
    equalize_norms,
    layer_tags,
    ctmrg_mode,
    ctmrg_canonize,
    ctmrg_projector_region,
    ctmrg_canonize_opts,
    ctmrg_compress_opts,
    ctmrg_reduce_opts,
    ctmrg_gauge_smudge,
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

    result, bdy_obj = _contract_peps_double_layer(
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
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
        bdy_name="bdy",
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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
    balance_bonds=True,
    return_info=False,
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
    fit_mode : {"direct", "src", "src-mps", "zipup", "sdc", "sdcr", "dm", "eff", "two-site", "dmrg", "dmrg1", "dmrg2", "global"}, default="eff"
        Boundary compression mode. Quimb also accepts the supported
        ``*-first``/``*-oversample`` variants and ``src-mps`` aliases
        ``srcmps``. The direct modes compress each
        boundary target; ``"dmrg"`` and ``"dmrg1"`` alias ``"eff"``, while
        ``"dmrg2"`` uses two-site warm-up followed by one-site refinement.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        Direct-mode layer policy. ``"joint"`` compresses all tagged layers
        together; ``"sequential"`` compresses them one at a time in the
        order given by ``layer_tags``.
    fit_layer_order : {"input", "auto"}, default="input"
        Sequential absorption order. ``"input"`` preserves ``layer_tags``;
        ``"auto"`` estimates dense intermediate sizes and requires explicit,
        mathematically interchangeable layer tags.
    layer_tags : sequence[str] | None, default=None
        Layer tags used for sequential direct compression, in absorption
        order. Use e.g. ``("BRA", "PEPO", "KET")`` for three layers.
    ctmrg_mode : {"projector", "projector2d", "l2bp"}, default="projector"
        Quimb boundary compressor used by ``method="ctmrg"``. The default
        preserves Pepsy's existing projector contraction.
    ctmrg_canonize : bool | {"layered", "bp"} | None, default=None
        CTMRG projector preconditioning. ``None`` preserves the current
        projector default of ``True``. The string choices apply only to
        ``ctmrg_mode="projector"``.
    ctmrg_projector_region : {(2, 2), (2, 3)} | str | None, default=None
        Optional local projector environment. The opt-in ``(2, 3)`` route is
        dense, 2D, and open-boundary only.
    ctmrg_canonize_opts, ctmrg_compress_opts : mapping | None, default=None
        Options for projector canonicalization and the selected CTMRG
        compressor, respectively.
    ctmrg_reduce_opts : mapping | None, default=None
        Squared-environment factorization options for projector modes.
    ctmrg_gauge_smudge : float | None, default=None
        Gauge regularization for ``ctmrg_mode="projector"``.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial boundary guess. ``"guess-src"`` applies Quimb
        SRC to a copy of each exact boundary target before FIT.
    fit_init_seed : int | None, default=0
        Seed forwarded to the disposable SRC guess.
    fit_block_size : {1, 2, 3}, default=1
        Block size used by ``FIT.run_eff`` when ``fit_mode="eff"``.
        Sizes 2 and 3 enable native block-SVD growth.
    fit_adaptive_sweeps : int | None, default=None
        For block size 2 or 3, use block updates for this many initial
        sweeps, then refine with one-site updates. ``None`` keeps fixed
        block-size sweeps.
    fit_max_bond : int | None, default=None
        Two-site SVD cap. Defaults to the requested boundary ``chi``.
    fit_sweep_sequence : str, default="RL"
        Repeating two-site sweep directions.
    fit_cutoff_mode : str | None | {"auto"}, default="auto"
        Quimb cutoff convention. ``"auto"`` and ``None`` resolve to
        ``"rsum2"``.
    fit_min_iter : int | None, default=None
        Minimum two-site sweeps before adaptive stopping.
    fit_rtol : float | None, default=None
        Relative convergence tolerance. ``None`` performs exactly ``n_iter``
        sweeps.
    fit_patience : int, default=1
        Consecutive converged sweeps required when ``fit_rtol`` is set.
    fit_timing : bool, default=False
        Include elapsed and detailed two-site sweep timings in boundary FIT
        diagnostics.
    fit_timing_sync_device : bool, default=False
        Synchronize supported accelerator kernels for timing accuracy.
    single_layer : bool, default=False
        Boundary initializer mode for :class:`pepsy.boundary.states.BdyMPS`.
    strip_exponent : bool, default=False
        If ``True``, use stripped boundary contractions and return
        ``(mantissa, exponent)`` for the old norm estimate. If ``False`` and
        the full norm estimate is non-finite, normalization retries with the
        stripped representation before mutating ``p`` and emits a
        ``RuntimeWarning``.
    balance_bonds : bool, default=True
        If ``True``, call ``balance_bonds_()`` after rescaling a dense state.
        Symmray block-sparse states always skip this step because Quimb's bond
        balancer currently needs a contraction unsupported by Symmray.
    return_info : bool, default=False
        Return :class:`BoundaryContractResult` with the old norm in ``cost``
        instead of returning only the old norm scalar.

    Returns
    -------
    complex | float | BoundaryContractResult
        The old norm estimate returned before rescaling, or the complete
        contraction result when ``return_info=True``.
    """
    if p is None:
        raise ValueError("p must not be None.")
    chi = _validate_chi(chi)

    contract_kwargs = dict(
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
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=single_layer,
        visualize=visualize,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
    )
    try:
        result, ket_tagged, _ = _contract_state_norm(
            p,
            strip_exponent=strip_exponent,
            **contract_kwargs,
        )
    except OverflowError:
        if strip_exponent:
            raise
        _warn_retry_stripped_norm("overflowing")
        result, ket_tagged, _ = _contract_state_norm(
            p,
            strip_exponent=True,
            **contract_kwargs,
        )
    cost = result.cost
    if not strip_exponent and not _is_finite_scaled_scalar(cost):
        _warn_retry_stripped_norm("not finite")
        result, ket_tagged, _ = _contract_state_norm(
            p,
            strip_exponent=True,
            **contract_kwargs,
        )
        cost = result.cost
    _ensure_finite_norm_value(cost)
    old_norm = _format_scaled_output(cost, strip_exponent=strip_exponent)
    _normalize_by_scaled_norm(ket_tagged, cost)
    if balance_bonds and not _uses_symmray_arrays(ket_tagged):
        ket_tagged.balance_bonds_()
    if return_info:
        return replace(result, cost=old_norm)
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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
    return_info=False,
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
    fit_mode : {"direct", "src", "src-mps", "zipup", "sdc", "sdcr", "dm", "eff", "two-site", "dmrg", "dmrg1", "dmrg2", "global"}, default="eff"
        Boundary compression mode. Quimb also accepts the supported
        ``*-first``/``*-oversample`` variants and ``src-mps`` aliases
        ``srcmps``. The direct modes compress each
        boundary target; ``"dmrg"`` and ``"dmrg1"`` alias ``"eff"``, while
        ``"dmrg2"`` uses two-site warm-up followed by one-site refinement.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        Direct-mode layer policy. ``"joint"`` compresses all tagged layers
        together; ``"sequential"`` compresses them one at a time in the
        order given by ``layer_tags``.
    fit_layer_order : {"input", "auto"}, default="input"
        Sequential absorption order. ``"input"`` preserves ``layer_tags``;
        ``"auto"`` estimates dense intermediate sizes and requires explicit,
        mathematically interchangeable layer tags.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial boundary guess. ``"guess-src"`` applies Quimb
        SRC to a copy of each exact boundary target before FIT.
    fit_init_seed : int | None, default=0
        Seed forwarded to the disposable SRC guess.
    fit_block_size : {1, 2, 3}, default=1
        Block size used by ``FIT.run_eff`` when ``fit_mode="eff"``.
        Sizes 2 and 3 enable native block-SVD growth.
    fit_adaptive_sweeps : int | None, default=None
        For block size 2 or 3, use block updates for this many initial
        sweeps, then refine with one-site updates. ``None`` keeps fixed
        block-size sweeps.
    fit_max_bond : int | None, default=None
        Two-site SVD cap. Defaults to the requested boundary ``chi``.
    fit_sweep_sequence : str, default="RL"
        Repeating two-site sweep directions.
    fit_cutoff_mode : str | None | {"auto"}, default="auto"
        Quimb cutoff convention. ``"auto"`` and ``None`` resolve to
        ``"rsum2"``.
    fit_min_iter : int | None, default=None
        Minimum two-site sweeps before adaptive stopping.
    fit_rtol : float | None, default=None
        Relative convergence tolerance. ``None`` performs exactly ``n_iter``
        sweeps.
    fit_patience : int, default=1
        Consecutive converged sweeps required when ``fit_rtol`` is set.
    fit_timing : bool, default=False
        Include elapsed and detailed two-site sweep timings in boundary FIT
        diagnostics.
    fit_timing_sync_device : bool, default=False
        Synchronize supported accelerator kernels for timing accuracy.
    single_layer : bool, default=False
        Boundary initializer mode for :class:`pepsy.boundary.states.BdyMPS`.
    strip_exponent : bool, default=False
        If ``True``, return ``(mantissa, exponent)`` for the norm estimate.
    ctmrg_mode : {"projector", "projector2d", "l2bp"}, default="projector"
        Quimb boundary compressor used by ``method="ctmrg"``.
    ctmrg_canonize : bool | {"layered", "bp"} | None, default=None
        Mode-aware CTMRG projector preconditioning.
    ctmrg_projector_region : {(2, 2), (2, 3)} | str | None, default=None
        Optional local projector environment. The opt-in ``(2, 3)`` route is
        dense, 2D, and open-boundary only.
    ctmrg_canonize_opts, ctmrg_compress_opts : mapping | None, default=None
        Canonicalization and selected-compressor options, respectively.
    ctmrg_reduce_opts : mapping | None, default=None
        Squared-environment factorization options for projector modes.
    ctmrg_gauge_smudge : float | None, default=None
        Gauge regularization for ``ctmrg_mode="projector"``.
    return_info : bool, default=False
        Return :class:`BoundaryContractResult` instead of only its ``cost``.

    Returns
    -------
    complex | float | BoundaryContractResult
        The ``<p|p>`` norm estimate, or the complete contraction result when
        ``return_info=True``.
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
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
    )
    cost = _format_scaled_output(result.cost, strip_exponent=strip_exponent)
    if return_info:
        return replace(result, cost=cost)
    return cost


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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
    return_info=False,
):
    """Compute the PEPS norm ``<p|p>`` without modifying ``p``.

    This is the PEPS-named alias for :func:`boundary_norm`. It accepts the same
    contraction methods and boundary controls, including ``method="dmrg"``,
    ``"mps"``, ``"ctmrg"``, ``"hotrg"``, and ``"exact"``.

    ``fit_block_size`` and ``fit_adaptive_sweeps`` are forwarded to the
    ``fit_mode="eff"`` boundary solver.

    For direct ``fit_mode`` values, ``fit_layer_mode="sequential"`` absorbs
    layers one at a time in the order given by ``layer_tags``. The default is
    ``fit_layer_mode="joint"``; use e.g. ``("BRA", "PEPO", "KET")`` for a
    tagged three-layer target.

    For ``method="ctmrg"``, ``ctmrg_mode`` selects ``"projector"`` (the
    compatibility default), ``"projector2d"``, or ``"l2bp"``. The remaining
    ``ctmrg_*`` arguments match :func:`contract_flat`.

    Returns
    -------
    complex | float | tuple[complex | float, float] | BoundaryContractResult
        Norm estimate. With ``strip_exponent=True`` this is
        ``(mantissa, exponent)``. With ``return_info=True``, return the
        structured contraction result and store that value in ``result.cost``.
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
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
        return_info=return_info,
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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
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
    fit_mode : {"direct", "src", "src-mps", "zipup", "sdc", "sdcr", "dm", "eff", "two-site", "dmrg", "dmrg1", "dmrg2", "global"}, default="eff"
        Boundary compression mode. Quimb also accepts the supported
        ``*-first``/``*-oversample`` variants and ``src-mps`` aliases
        ``srcmps``. The direct modes compress each
        boundary target; ``"dmrg"`` and ``"dmrg1"`` alias ``"eff"``, while
        ``"dmrg2"`` uses two-site warm-up followed by one-site refinement.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        Direct-mode layer policy. ``"joint"`` compresses all tagged layers
        together; ``"sequential"`` compresses them one at a time in the
        order given by ``layer_tags``.
    fit_layer_order : {"input", "auto"}, default="input"
        Sequential absorption order. ``"input"`` preserves ``layer_tags``;
        ``"auto"`` estimates dense intermediate sizes and requires explicit,
        mathematically interchangeable layer tags.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial boundary guess. ``"guess-src"`` applies Quimb
        SRC to a copy of each exact boundary target before FIT.
    fit_init_seed : int | None, default=0
        Seed forwarded to the disposable SRC guess.
    fit_block_size : {1, 2, 3}, default=1
        Block size used by ``FIT.run_eff`` when ``fit_mode="eff"``.
    fit_adaptive_sweeps : int | None, default=None
        For block size 2 or 3, number of initial block-SVD sweeps before
        one-site refinement. ``None`` keeps fixed block-size sweeps.
    fit_max_bond : int | None, default=None
        Two-site SVD cap. Defaults to the requested boundary ``chi``.
    fit_sweep_sequence : str, default="RL"
        Repeating two-site sweep directions.
    fit_cutoff_mode : str | None | {"auto"}, default="auto"
        Quimb cutoff convention. ``"auto"`` and ``None`` resolve to
        ``"rsum2"``.
    fit_min_iter : int | None, default=None
        Minimum two-site sweeps before adaptive stopping.
    fit_rtol : float | None, default=None
        Relative two-site convergence tolerance.
    fit_patience : int, default=1
        Consecutive converged sweeps required when ``fit_rtol`` is set.
    fit_timing : bool, default=False
        Include elapsed and detailed two-site sweep timings in each returned
        contraction result's FIT diagnostics.
    fit_timing_sync_device : bool, default=False
        Synchronize supported accelerator kernels for timing accuracy.
    single_layer : bool, default=False
        Boundary initializer mode for :class:`pepsy.boundary.states.BdyMPS`.
    visualize : bool, default=False
    strip_exponent : bool, default=False
        If ``True``, keep norm and overlap contractions as
        ``(mantissa, exponent)`` pairs and compute the fidelity ratio without
        reconstructing large or tiny scalars.
    ctmrg_mode : {"projector", "projector2d", "l2bp"}, default="projector"
        Quimb boundary compressor used by ``method="ctmrg"``.
    ctmrg_canonize : bool | {"layered", "bp"} | None, default=None
        Mode-aware CTMRG projector preconditioning.
    ctmrg_projector_region : {(2, 2), (2, 3)} | str | None, default=None
        Optional local projector environment. The opt-in ``(2, 3)`` route is
        dense, 2D, and open-boundary only.
    ctmrg_canonize_opts, ctmrg_compress_opts : mapping | None, default=None
        Canonicalization and selected-compressor options, respectively.
    ctmrg_reduce_opts : mapping | None, default=None
        Squared-environment factorization options for projector modes.
    ctmrg_gauge_smudge : float | None, default=None
        Gauge regularization for ``ctmrg_mode="projector"``.
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
        - ``norm_result`` / ``norm_target_result`` / ``overlap_result``:
          structured contraction results with boundary FIT diagnostics;
          known supplied norms have a corresponding result of ``None``
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
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
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
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
    )

    # -- <p|p> --
    bdy_obj = None
    norm_result = None
    if norm is None:
        _, norm_tn = build_bra_ket(ket=p, bra=None)
        norm_result, bdy_obj = _contract_peps_double_layer(
            norm_tn,
            bdy=bdy,
            single_layer=single_layer,
            bdy_name="bdy",
            **_kw,
        )
        norm = _format_scaled_output(
            norm_result.cost,
            strip_exponent=strip_exponent,
        )
    else:
        norm = _format_scaled_output(norm, strip_exponent=strip_exponent)

    # -- <p_target|p_target> --
    bdy_target_obj = None
    norm_target_result = None
    if norm_target is None:
        _, norm_target_tn = build_bra_ket(ket=p_target, bra=None)
        norm_target_result, bdy_target_obj = _contract_peps_double_layer(
            norm_target_tn,
            bdy=bdy_target,
            single_layer=single_layer,
            bdy_name="bdy_target",
            **_kw,
        )
        norm_target = _format_scaled_output(
            norm_target_result.cost,
            strip_exponent=strip_exponent,
        )
    else:
        norm_target = _format_scaled_output(norm_target, strip_exponent=strip_exponent)

    # -- <p_target|p> (overlap) --
    _, overlap_tn = build_bra_ket(ket=p, bra=p_target)
    overlap_result, bdy_overlap_obj = _contract_peps_double_layer(
        overlap_tn,
        bdy=bdy_overlap,
        single_layer=single_layer,
        bdy_name="bdy_overlap",
        **_kw,
    )
    overlap = _format_scaled_output(
        overlap_result.cost,
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
        "norm_result": norm_result,
        "norm_target_result": norm_target_result,
        "overlap_result": overlap_result,
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
    fit_layer_mode="joint",
    fit_layer_order="input",
    fit_init_strategy="direct",
    fit_init_seed=0,
    fit_block_size=1,
    fit_adaptive_sweeps=None,
    fit_max_bond=None,
    fit_sweep_sequence="RL",
    fit_cutoff_mode="auto",
    fit_compression_opts=None,
    fit_min_iter=None,
    fit_rtol=None,
    fit_patience=1,
    fit_timing=False,
    fit_timing_sync_device=False,
    single_layer=False,
    visualize=False,
    strip_exponent=False,
    mode_="mps",
    sequence=None,
    cutoff=1.0e-12,
    equalize_norms=False,
    layer_tags=None,
    ctmrg_mode="projector",
    ctmrg_canonize=None,
    ctmrg_projector_region=None,
    ctmrg_canonize_opts=None,
    ctmrg_compress_opts=None,
    ctmrg_reduce_opts=None,
    ctmrg_gauge_smudge=None,
    return_info=False,
):
    """Compute boundary-estimated PEPS fidelity.

    This is a convenience wrapper around :func:`peps_infidelity` that returns
    ``1 - infidelity``. Pass ``norm`` and/or ``norm_target`` when one of the
    states is already known to be normalized; the corresponding self-overlap
    contraction is then skipped. With ``return_info=True``, return the complete
    infidelity result dictionary plus its computed ``fidelity`` so opt-in FIT
    timing and convergence diagnostics remain observable.

    Returns
    -------
    float | dict[str, object]
        Boundary-estimated fidelity, or complete norm/overlap contraction
        information when ``return_info=True``.
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
        fit_layer_mode=fit_layer_mode,
        fit_layer_order=fit_layer_order,
        fit_init_strategy=fit_init_strategy,
        fit_init_seed=fit_init_seed,
        fit_block_size=fit_block_size,
        fit_adaptive_sweeps=fit_adaptive_sweeps,
        fit_max_bond=fit_max_bond,
        fit_sweep_sequence=fit_sweep_sequence,
        fit_cutoff_mode=fit_cutoff_mode,
        fit_compression_opts=fit_compression_opts,
        fit_min_iter=fit_min_iter,
        fit_rtol=fit_rtol,
        fit_patience=fit_patience,
        fit_timing=fit_timing,
        fit_timing_sync_device=fit_timing_sync_device,
        single_layer=single_layer,
        visualize=visualize,
        strip_exponent=strip_exponent,
        mode_=mode_,
        sequence=sequence,
        cutoff=cutoff,
        equalize_norms=equalize_norms,
        layer_tags=layer_tags,
        ctmrg_mode=ctmrg_mode,
        ctmrg_canonize=ctmrg_canonize,
        ctmrg_projector_region=ctmrg_projector_region,
        ctmrg_canonize_opts=ctmrg_canonize_opts,
        ctmrg_compress_opts=ctmrg_compress_opts,
        ctmrg_reduce_opts=ctmrg_reduce_opts,
        ctmrg_gauge_smudge=ctmrg_gauge_smudge,
    )
    fidelity = 1 - result["infidelity"]
    if return_info:
        result["fidelity"] = fidelity
        return result
    return fidelity


# Compatibility aliases for the former generic names. The boundary package
# facade emits the deprecation warning; direct aliases keep one implementation
# and preserve object identity for callers that introspect the leaf module.
normalize = peps_normalize
infidelity = peps_infidelity
