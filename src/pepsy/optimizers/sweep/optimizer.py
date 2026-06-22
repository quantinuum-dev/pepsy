"""High-level PEPS sweep optimizer with axis-alternating boundary updates."""

from __future__ import annotations

import time
import warnings
import math
import inspect
from collections.abc import Mapping
from typing import Any

import autoray as ar
import quimb.tensor as qtn
from tqdm.auto import tqdm

from ...boundary.metrics import peps_infidelity as boundary_infidelity
from ...boundary.metrics import build_bra_ket, peps_normalize
from ...boundary.states import BdyMPS
from ...boundary.sweeps import CompBdy
from ...tensors.core import tn_fidelity
from ...solvers.gradient import GradientOptimizer, SUPPORTED_SOLVERS
from ...tensors.validation import _PHYS_IND_PATTERN, _TAG_X, _TAG_Y
from .environments import (
    QuimbMpsBoundaryStore,
    normalize_boundary_engine,
    uses_symmray_arrays,
)

__all__ = ["SweepOptimizer"]


class _AttrDict(dict):
    """dict with attribute-style access for compatibility."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class SweepOptimizer:  # pylint: disable=too-many-instance-attributes
    """Optimize PEPS slices with alternating x/y boundary sweeps.

    Parameters
    ----------
    state : qtn.TensorNetwork
        Trainable PEPS-like tensor network.
    state_target : qtn.TensorNetwork
        Reference network for overlap objective.
    target_norm : complex | float | tuple[complex | float, float], default=1.0
        Known value of ``<state_target|state_target>`` for local sweep
        fidelity objectives. Pass either a scalar or a ``(mantissa, exponent)``
        pair such that ``norm = mantissa * 10**exponent``. The default keeps
        the historical normalized-target assumption.
    chi : int | tuple[int, int] | None, default=None
        Boundary bond dimension used when ``bdy``/``bdy_overlap`` are not
        supplied.  Pass a single ``int`` to use the same dimension for both
        boundaries, or ``(chi_bdy, chi_overlap)`` to set them independently.
    bdy : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Optional boundary container for norm contractions. A dict holder style
        ``{"bdy": <BdyMPS>}`` is also accepted and will be updated in place.
    bdy_overlap : pepsy.boundary.states.BdyMPS | dict | None, default=None
        Optional boundary container for overlap contractions. A dict holder
        style ``{"bdy": <BdyMPS>}`` is also accepted and will be updated in
        place.
    contraction_opt : object | str, default="auto-hq"
        Contraction optimizer.
    fit_mode : {"eff", "global"}, default="eff"
        Backend mode passed to :class:`pepsy.boundary.sweeps.CompBdy`.
    boundary_engine : {"auto", "dmrg", "quimb-mps"}, default="auto"
        Environment engine used during local sweeps. ``"dmrg"`` preserves the
        Pepsy ``BdyMPS``/``CompBdy`` path. ``"quimb-mps"`` builds reusable
        Quimb MPS environments with ``compute_x/y_environments(...)``.
        ``"auto"`` keeps dense backends on ``"dmrg"`` and routes Symmray
        networks to ``"quimb-mps"``.
    boundary_options : mapping | None, optional
        Extra options for the Quimb MPS environment engine, such as
        ``cutoff``, ``canonize``, ``mode``, ``layer_tags``,
        ``compress_opts``, and ``equalize_norms``.
    """

    _NORMALIZE_KEYS = frozenset({
        "chi",
        "contraction_opt",
        "n_iter",
        "direction",
        "max_separation",
        "progress",
        "track_boundary_fidelity",
        "strip_exponent",
        "method",
        "mode_",
        "sequence",
        "cutoff",
        "equalize_norms",
        "layer_tags",
        "balance_bonds",
    })
    _INFIDELITY_KEYS = frozenset({
        "chi",
        "norm",
        "norm_target",
        "contraction_opt",
        "n_iter",
        "direction",
        "max_separation",
        "progress",
        "track_boundary_fidelity",
        "single_layer",
        "strip_exponent",
        "method",
        "mode_",
        "sequence",
        "cutoff",
        "equalize_norms",
        "layer_tags",
    })
    _OPTIMIZE_KEYS = frozenset({
        "axes",
        "n_cycles",
        "n_round_trips",
        "chi",
        "optimizer",
        "optimizer_options",
        "env_n_iter",
        "progress",
        "progress_position",
        "progress_leave",
        "track_boundary_fidelity",
        "debug",
        "debug_loss_mode",
        "debug_loss_kwargs",
        "renormalize",
        "boundary_engine",
        "boundary_options",
    })
    _DEFAULT_SOLVER_OPTIONS = {
        "algorithm": "LBFGS",
        "lr": 1e-2,
        "n_steps": 30,
        "maxeval": 100,
        "ftol_rel": 1e-9,
        "xtol_rel": 1e-9,
        "patience": 40,
        "min_steps": 10,
        "restore_best": True,
        "bad_max": 20,
        "penalty_on_bad": 1e20,
    }


    @staticmethod
    def _unpack_chi(chi):
        """Return ``(chi_bdy, chi_overlap)`` from *chi*.

        Accepts ``int``, ``(int, int)`` tuple/list, or ``None``.
        """
        if chi is None:
            return None, None
        if isinstance(chi, (tuple, list)):
            if len(chi) != 2:
                raise ValueError(
                    "chi tuple must have exactly 2 elements: (chi_bdy, chi_overlap)"
                )
            return int(chi[0]), int(chi[1])
        return int(chi), int(chi)

    @classmethod
    def _normalize_chi(cls, chi):
        """Return the scalar chi used for state-norm contractions."""
        chi_bdy, _ = cls._unpack_chi(chi)
        return chi_bdy

    @staticmethod
    def _merge_opts(base, extra):
        merged = dict(base or {})
        if extra:
            merged.update(dict(extra))
        return merged

    @staticmethod
    def _as_scaled_scalar(value, *, name="value"):
        """Return ``(mantissa, exponent)`` for scalar or stripped scalar input."""
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"{name} must be a scalar or (mantissa, exponent).")
            return value[0], float(value[1])
        return value, 0.0

    @staticmethod
    def _network_exponent(tn):
        """Return the real base-10 exponent carried by a tensor network."""
        return complex(getattr(tn, "exponent", 0.0)).real

    @classmethod
    def _shift_scaled_exponent(cls, value, exponent_delta):
        """Add ``exponent_delta`` to a scalar or stripped scalar result."""
        mantissa, exponent = cls._as_scaled_scalar(value)
        return mantissa, exponent + float(exponent_delta)

    @staticmethod
    def _safe_pow10(exponent):
        """Return ``10**exponent`` without over/under-flowing Python floats."""
        # strip_exponent exponents are ordinary numbers in current quimb, but
        # keep backend-scalar support for compatibility with differentiable runs.
        if hasattr(exponent, "shape") or hasattr(exponent, "detach"):
            return 10.0 ** ar.do("clip", exponent, -300.0, 300.0)
        exponent = float(exponent)
        if exponent <= -300.0:
            return 0.0
        if exponent >= 300.0:
            return 1.0e300
        return 10.0**exponent

    @classmethod
    def _scaled_overlap_fidelity(cls, overlap, norm, target_norm):
        """Return ``|overlap|**2 / (|norm| * |target_norm|)`` stably.

        Each input may be either a scalar or a ``(mantissa, exponent)`` pair
        as returned by ``TensorNetwork.contract(..., strip_exponent=True)``.
        """
        overlap_m, overlap_e = cls._as_scaled_scalar(overlap, name="overlap")
        norm_m, norm_e = cls._as_scaled_scalar(norm, name="norm")
        target_m, target_e = cls._as_scaled_scalar(target_norm, name="target_norm")

        fid_m = (ar.do("abs", overlap_m) ** 2) / (
            ar.do("abs", norm_m) * ar.do("abs", target_m)
        )
        fid_e = 2.0 * overlap_e - norm_e - target_e
        return ar.do("abs", fid_m) * cls._safe_pow10(fid_e)

    def _local_norm_exponent(self):
        """Exponent for the represented local norm environment."""
        return 2.0 * self._network_exponent(self.state)

    def _local_overlap_exponent(self):
        """Exponent for the represented local target/state overlap environment."""
        return self._network_exponent(self.state) + self._network_exponent(self.state_target)

    def set_target_norm(self, target_norm=1.0):
        """Set the stored target norm used by local sweep objectives."""
        self.target_norm = self._as_scaled_scalar(
            1.0 if target_norm is None else target_norm,
            name="target_norm",
        )
        return self

    @classmethod
    def _pick_known_keys(cls, options, allowed_keys, *, warn_unknown=True):
        incoming = dict(options or {})
        filtered = {key: value for key, value in incoming.items() if key in allowed_keys}
        unknown = sorted(set(incoming) - set(allowed_keys))
        if warn_unknown and unknown:
            warnings.warn(
                f"Ignoring unknown options: {', '.join(unknown)}",
                UserWarning,
                stacklevel=3,
            )
        return filtered

    @classmethod
    def _merge_solver_options(cls, options):
        merged = cls.default_solver_options()
        if options:
            merged.update(dict(options))
        return merged

    @classmethod
    def default_solver_options(cls):
        """Return copy of package default local-solver options."""
        return dict(cls._DEFAULT_SOLVER_OPTIONS)

    @classmethod
    def normalize_kwarg_names(cls):
        """Return supported keyword names for :meth:`normalize` defaults."""
        return tuple(sorted(cls._NORMALIZE_KEYS))

    @classmethod
    def optimize_kwarg_names(cls):
        """Return supported keyword names for :meth:`set_optimize_kwargs`."""
        return tuple(sorted(cls._OPTIMIZE_KEYS))

    @classmethod
    def infidelity_kwarg_names(cls):
        """Return supported keyword names for :meth:`infidelity`."""
        return tuple(sorted(cls._INFIDELITY_KEYS))

    @classmethod
    def kwarg_guide(cls):
        """Return a compact guide of public kwargs and default solver options."""
        return {
            "normalize": cls.normalize_kwarg_names(),
            "optimize": cls.optimize_kwarg_names(),
            "infidelity": cls.infidelity_kwarg_names(),
            "optimizer_defaults": cls.default_solver_options(),
        }

    @classmethod
    def _collect_init_renormalize_kwargs(
        cls,
        *,
        chi=None,
        renormalize_kwargs=None,
        n_iter=None,
        direction=None,
        max_separation=None,
        progress=None,
        track_boundary_fidelity=None,
    ):
        """Collect init-time normalize kwargs from explicit and mapping styles."""
        legacy = {
            "chi": chi,
            "n_iter": n_iter,
            "direction": direction,
            "max_separation": max_separation,
            "progress": progress,
            "track_boundary_fidelity": track_boundary_fidelity,
        }
        # Only forward values the caller explicitly set.
        legacy = {k: v for k, v in legacy.items() if v is not None}
        out = cls._pick_known_keys(legacy, cls._NORMALIZE_KEYS, warn_unknown=False)
        if renormalize_kwargs:
            out.update(cls._pick_known_keys(renormalize_kwargs, cls._NORMALIZE_KEYS))
        return out

    def __init__(
        self,
        state,
        state_target,
        *,
        target_norm=1.0,
        chi=None,
        bdy=None,
        bdy_overlap=None,
        contraction_opt="auto-hq",
        fit_mode="eff",
        single_layer=False,
        simplify=True,
        normalize_kwargs: Mapping[str, Any] | None = None,
        optimize_kwargs: Mapping[str, Any] | None = None,
        renormalize_state=False,
        renormalize_kwargs: Mapping[str, Any] | None = None,
        boundary_engine: str | None = "auto",
        boundary_options: Mapping[str, Any] | None = None,
        n_iter: int | None = None,
        direction: str | None = None,
        max_separation: int | None = None,
        progress: bool | None = None,
        track_boundary_fidelity: bool | None = None,
    ):
        self._ensure_no_common_internal_indices(state, state_target)

        bdy_obj, bdy_holder = self._resolve_boundary_arg(bdy, "bdy")
        bdy_overlap_obj, bdy_overlap_holder = self._resolve_boundary_arg(
            bdy_overlap,
            "bdy_overlap",
        )
        self._bdy_holder = bdy_holder
        self._bdy_overlap_holder = bdy_overlap_holder
        self.boundary_options = dict(boundary_options or {})
        self.boundary_engine = normalize_boundary_engine(
            boundary_engine,
            state,
            state_target,
            boundaries_supplied=bdy_obj is not None,
        )

        if (bdy_obj is None) ^ (bdy_overlap_obj is None):
            raise ValueError(
                "Provide both boundaries (or holder dicts with key 'bdy') together, or neither."
            )

        if bdy_obj is None and bdy_overlap_obj is None:
            if chi is None:
                raise ValueError(
                    "Provide chi when bdy and bdy_overlap are not supplied."
                )
            if state_target is None:
                raise ValueError("state_target is required when boundaries are not supplied.")
            bdy_obj, bdy_overlap_obj = self._call_with_accepted_kwargs(
                self._build_boundary_pair,
                state,
                state_target,
                chi=chi,
                single_layer=single_layer,
                boundary_engine=self.boundary_engine,
                boundary_options=self.boundary_options,
            )

        self.state = state
        self.state_target = state_target
        self.set_target_norm(target_norm)
        self._set_boundary_pair(bdy_obj, bdy_overlap_obj)
        self.contraction_opt = contraction_opt
        self.fit_mode = fit_mode
        # Store chi as-is (may be int or (int, int) tuple).
        self.chi = chi if chi is not None else getattr(bdy_obj, "chi", None)
        self.single_layer = single_layer
        self.simplify = simplify
        self.direction = direction if direction is not None else "y"
        self.max_separation = max_separation if max_separation is not None else 1

        if normalize_kwargs is None:
            self.normalize_kwargs = {}
        else:
            self.normalize_kwargs = self._pick_known_keys(normalize_kwargs, self._NORMALIZE_KEYS)
        self.optimize_kwargs = self._pick_known_keys(optimize_kwargs, self._OPTIMIZE_KEYS)

        self.Lx, self.Ly = self._infer_shape(self.state)
        self._reset_run_traces()

        init_renormalize_kwargs = self._collect_init_renormalize_kwargs(
            chi=self.chi,
            renormalize_kwargs=renormalize_kwargs,
            n_iter=n_iter,
            direction=direction,
            max_separation=max_separation,
            progress=progress,
            track_boundary_fidelity=track_boundary_fidelity,
        )

        if renormalize_state:
            self.normalize(**init_renormalize_kwargs)

    @staticmethod
    def _ensure_no_common_internal_indices(state, state_target):
        """Require disjoint internal indices between ``state`` and ``state_target``."""
        if state is None or state_target is None:
            return
        if not hasattr(state, "inner_inds") or not hasattr(state_target, "inner_inds"):
            return
        state_inner = set(state.inner_inds())
        target_inner = set(state_target.inner_inds())
        shared = state_inner & target_inner
        if not shared:
            return

        sample = ", ".join(sorted(map(str, shared))[:8])
        msg = (
            "state and state_target share common internal indices. "
            f"Shared sample: {sample}"
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        raise ValueError(msg)

    @staticmethod
    def _call_with_accepted_kwargs(fn, *args, **kwargs):
        """Call ``fn`` while dropping unsupported keyword arguments."""
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(*args, **kwargs)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **accepted)

    @staticmethod
    def _resolve_boundary_arg(boundary, name):
        """Unpack boundary argument as ``(boundary_obj, holder_dict_or_none)``."""
        if boundary is None:
            return None, None
        if isinstance(boundary, dict):
            obj = boundary.get("bdy", None)
            if obj is not None and not hasattr(obj, "mps_b"):
                raise TypeError(f"{name}['bdy'] must expose attribute 'mps_b'.")
            return obj, boundary
        if not hasattr(boundary, "mps_b"):
            raise TypeError(f"{name} must expose attribute 'mps_b'.")
        return boundary, None

    @staticmethod
    def _build_boundary_pair(
        state,
        target,
        *,
        chi,
        single_layer=False,
        boundary_engine="dmrg",
        boundary_options=None,
    ):
        """Construct norm and overlap boundary MPS containers."""
        chi_bdy, chi_overlap = SweepOptimizer._unpack_chi(chi)
        boundary_engine = normalize_boundary_engine(boundary_engine, state, target)
        if boundary_engine == "quimb-mps":
            opts = dict(boundary_options or {})
            return (
                QuimbMpsBoundaryStore(chi=chi_bdy, **opts),
                QuimbMpsBoundaryStore(chi=chi_overlap, **opts),
            )

        _, state_norm = build_bra_ket(ket=state, bra=None)
        _, overlap_norm = build_bra_ket(ket=target, bra=state)
        bdy = BdyMPS(
            tn_double=state_norm,
            chi=chi_bdy,
            single_layer=single_layer,
        )
        bdy_overlap = BdyMPS(
            tn_double=overlap_norm,
            chi=chi_overlap,
            single_layer=single_layer,
        )
        return bdy, bdy_overlap

    def _set_boundary_pair(self, bdy, bdy_overlap):
        """Assign boundary objects and keep optional holder dicts in sync."""
        for name, obj in (("bdy", bdy), ("bdy_overlap", bdy_overlap)):
            if not hasattr(obj, "mps_b"):
                raise TypeError(f"{name} must expose attribute 'mps_b'.")
        self.bdy = bdy
        self.bdy_overlap = bdy_overlap
        if getattr(self, "_bdy_holder", None) is not None:
            self._bdy_holder["bdy"] = self.bdy
        if getattr(self, "_bdy_overlap_holder", None) is not None:
            self._bdy_overlap_holder["bdy"] = self.bdy_overlap

    def _update_boundaries_from_result(self, result):
        """Apply optional ``bdy``/``bdy_overlap`` entries from a result dict."""
        bdy_new = result.get("bdy") or self.bdy
        bdy_overlap_new = result.get("bdy_overlap") or self.bdy_overlap
        self._set_boundary_pair(bdy_new, bdy_overlap_new)

    @staticmethod
    def _infer_shape(state):
        """Infer ``(Lx, Ly)`` from ``X*`` and ``Y*`` tags."""
        max_x = None
        max_y = None
        for tag in getattr(state, "tags", ()):
            mx = _TAG_X.match(tag)
            my = _TAG_Y.match(tag)
            if mx:
                max_x = max(int(mx.group(1)), -1 if max_x is None else max_x)
            if my:
                max_y = max(int(my.group(1)), -1 if max_y is None else max_y)
        if max_x is None or max_y is None:
            raise ValueError("state must include X*/Y* tags to infer lattice shape.")
        return max_x + 1, max_y + 1

    def _reset_run_traces(self):
        """Reset lightweight traces collected during global sweep runs."""
        self.step_loss_trace = []
        # Keep legacy and new names synchronized.
        self.loss = self.step_loss_trace
        self.step_trace = []
        self.inner_loss_traces = []
        self.norm_trace = []
        self.fidels = []
        self.best_state = None
        self.best_loss = float("inf")

    def _best_nonnegative_from_history(self, history):
        """Return minimum finite non-negative loss from history or ``None``."""
        values = [
            float(v)
            for v in (history or ())
            if (v is not None) and math.isfinite(float(v)) and (float(v) >= 0.0)
        ]
        if not values:
            return None
        return min(values)

    def _maybe_store_best_state(self, loss_value):
        """Store current state snapshot when ``loss_value`` is a new valid best."""
        if loss_value is None:
            return
        loss_value = float(loss_value)
        if not math.isfinite(loss_value):
            return
        # Infidelity/loss should be non-negative; ignore negative artifacts.
        if loss_value < 0.0:
            return
        if loss_value < float(self.best_loss):
            self.best_loss = loss_value
            self.best_state = self.state.copy()

    def _ensure_boundary_chi(self, chi):
        """Retune both stored boundary objects to at least ``chi``.

        *chi* may be ``int`` (same for both) or ``(int, int)`` for
        ``(bdy, bdy_overlap)``.
        """
        if chi is None:
            return
        chi_bdy, chi_overlap = self._unpack_chi(chi)
        if chi_bdy < 1 or chi_overlap < 1:
            raise ValueError("chi must be >= 1")

        bdy = getattr(self, "bdy", None)
        if bdy is not None and getattr(bdy, "chi", 0) < chi_bdy:
            bdy.expand_bnd(chi_bdy, inplace=True)

        bdy_overlap = getattr(self, "bdy_overlap", None)
        if bdy_overlap is not None and getattr(bdy_overlap, "chi", 0) < chi_overlap:
            bdy_overlap.expand_bnd(chi_overlap, inplace=True)

    def set_chi(
        self,
        chi,
        *,
        normalize_state=False,
        n_iter=5,
        direction="y",
        max_separation=0,
        progress=False,
        track_boundary_fidelity=False,
    ):
        """Expand stored boundaries to ``chi`` and optionally renormalize state.

        Parameters
        ----------
        chi : int
            Target boundary bond dimension.
        normalize_state : bool, default=False
            If True, run :meth:`normalize` immediately after expanding
            boundaries.

        Returns
        -------
        SweepOptimizer
            ``self`` for chaining.
        """
        self._ensure_boundary_chi(chi)
        if normalize_state:
            self.normalize(
                n_iter=n_iter,
                direction=direction,
                max_separation=max_separation,
                progress=progress,
                track_boundary_fidelity=track_boundary_fidelity,
            )
        return self

    def set_state(
        self,
        state,
        *,
        chi=None,
        single_layer=False,
        normalize_state=True,
        n_iter=10,
        direction="y",
        max_separation=1,
        progress=False,
        track_boundary_fidelity=False,
    ):
        """Replace current state, rebuild boundaries, and optionally normalize.

        Parameters
        ----------
        state : qtn.TensorNetwork
            New trainable PEPS-like state.
        chi : int | None, default=None
            Boundary bond dimension for rebuilt boundaries. If omitted, uses
            ``self.bdy.chi`` when available.
        single_layer : bool, default=False
            Forwarded to :class:`pepsy.boundary.states.BdyMPS`.
        normalize_state : bool, default=True
            If True, normalize new state immediately using rebuilt boundaries.

        Returns
        -------
        complex | float | None
            Old norm returned by :meth:`normalize` when ``normalize_state`` is
            True, else None.
        """
        if self.state_target is None:
            raise ValueError("state_target must be set before calling set_state().")

        self._ensure_no_common_internal_indices(state, self.state_target)

        self.state = state
        self.Lx, self.Ly = self._infer_shape(self.state)

        if chi is None:
            chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")

        # chi may be int or (int, int) for separate bdy/bdy_overlap dims.
        bdy_new, bdy_overlap_new = self._call_with_accepted_kwargs(
            self._build_boundary_pair,
            self.state,
            self.state_target,
            chi=chi,
            single_layer=single_layer,
            boundary_engine=getattr(self, "boundary_engine", "dmrg"),
            boundary_options=getattr(self, "boundary_options", None),
        )
        self._set_boundary_pair(bdy_new, bdy_overlap_new)

        if normalize_state:
            return self.normalize(
                n_iter=n_iter,
                direction=direction,
                max_separation=max_separation,
                progress=progress,
                track_boundary_fidelity=track_boundary_fidelity,
            )
        return None

    def set_target(
        self,
        target,
        *,
        target_norm=1.0,
        chi=None,
        single_layer=False,
    ):
        """Replace target state and rebuild overlap boundary immediately.

        Parameters
        ----------
        target : qtn.TensorNetwork
            New target PEPS-like state.
        target_norm : complex | float | tuple[complex | float, float], default=1.0
            Known ``<target|target>`` value for local sweep objectives. Pass a
            ``(mantissa, exponent)`` pair to avoid reconstructing very large or
            very small scalar norms.
        chi : int | None, default=None
            Bond dimension for rebuilt overlap boundary. If omitted, uses
            ``self.bdy_overlap.chi`` when available.
        single_layer : bool, default=False
            Forwarded to :class:`pepsy.boundary.states.BdyMPS`.
        """
        self._ensure_no_common_internal_indices(self.state, target)
        self.state_target = target
        self.set_target_norm(target_norm)

        # Extract overlap chi from tuple or scalar.
        _, chi_overlap = self._unpack_chi(chi)
        if chi_overlap is None:
            chi_overlap = getattr(self.bdy_overlap, "chi", None)
        if chi_overlap is None:
            _, chi_overlap = self._unpack_chi(getattr(self, "chi", None))
        if chi_overlap is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")

        if getattr(self, "boundary_engine", "dmrg") == "quimb-mps":
            bdy_overlap_new = QuimbMpsBoundaryStore(
                chi=chi_overlap,
                **getattr(self, "boundary_options", {}),
            )
        else:
            _, overlap_norm = build_bra_ket(ket=self.state_target, bra=self.state)
            bdy_overlap_new = BdyMPS(
                tn_double=overlap_norm,
                chi=chi_overlap,
                single_layer=single_layer,
            )
        self._set_boundary_pair(self.bdy, bdy_overlap_new)

    def set_normalize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`normalize`."""
        if not hasattr(self, "normalize_kwargs"):
            self.normalize_kwargs = {}
        self.normalize_kwargs.update(self._pick_known_keys(kwargs, self._NORMALIZE_KEYS))
        return self

    def set_optimize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`run`."""
        if not hasattr(self, "optimize_kwargs"):
            self.optimize_kwargs = {}
        self.optimize_kwargs.update(self._pick_known_keys(kwargs, self._OPTIMIZE_KEYS))
        return self

    def normalize(self, state=None, **kwargs):
        """Normalize PEPS in place and return the old norm estimate."""
        state = self.state if state is None else state
        opts = self._merge_opts(getattr(self, "normalize_kwargs", {}), kwargs)
        uses_quimb = getattr(self, "boundary_engine", "dmrg") == "quimb-mps"
        contraction_opt = opts.get("contraction_opt", self.contraction_opt)
        n_iter = opts.get("n_iter", 10)
        direction = opts.get("direction", "y")
        max_separation = opts.get("max_separation", 1)
        progress = opts.get("progress", False)
        track_boundary_fidelity = opts.get("track_boundary_fidelity", False)
        strip_exponent = opts.get("strip_exponent", False)
        method = opts.get("method", "mps" if uses_quimb else "dmrg")
        chi = self._normalize_chi(opts.get("chi", getattr(self.bdy, "chi", None)))
        bdy = None if uses_quimb else self.bdy
        balance_bonds = opts.get("balance_bonds", not uses_quimb)

        if state is self.state:
            return peps_normalize(
                self.state,
                chi=chi,
                bdy=bdy,
                contraction_opt=contraction_opt,
                max_separation=max_separation,
                n_iter=n_iter,
                direction=direction,
                progress=progress,
                track_boundary_fidelity=track_boundary_fidelity,
                fit_mode=self.fit_mode,
                strip_exponent=strip_exponent,
                method=method,
                mode_=opts.get("mode_", "mps"),
                sequence=opts.get("sequence", None),
                cutoff=opts.get("cutoff", 1.0e-12),
                equalize_norms=opts.get("equalize_norms", False),
                layer_tags=opts.get("layer_tags", None),
                balance_bonds=balance_bonds,
            )

        if chi is None:
            raise ValueError("Provide chi via optimizer boundaries before normalizing external state.")
        return peps_normalize(
            state,
            chi=chi,
            contraction_opt=contraction_opt,
            max_separation=max_separation,
            n_iter=n_iter,
            direction=direction,
            progress=progress,
            track_boundary_fidelity=track_boundary_fidelity,
            fit_mode=self.fit_mode,
            strip_exponent=strip_exponent,
            method=method,
            mode_=opts.get("mode_", "mps"),
            sequence=opts.get("sequence", None),
            cutoff=opts.get("cutoff", 1.0e-12),
            equalize_norms=opts.get("equalize_norms", False),
            layer_tags=opts.get("layer_tags", None),
            balance_bonds=balance_bonds,
        )

    def _normalize_state(self, env_n_iter=4):
        # Start from user/default normalize kwargs, but keep sweep-time
        # normalization deterministic for stability.
        opts = self._pick_known_keys(
            getattr(self, "normalize_kwargs", {}),
            self._NORMALIZE_KEYS,
            warn_unknown=False,
        )
        opts.setdefault("direction", "y")
        opts.setdefault("max_separation", 1)
        opts["n_iter"] = env_n_iter
        opts["progress"] = False
        opts["track_boundary_fidelity"] = False
        return self.normalize(**opts)

    def infidelity(
        self,
        *,
        chi=None,
        norm=None,
        norm_target=None,
        contraction_opt=None,
        n_iter=5,
        direction="y",
        max_separation=1,
        progress=False,
        track_boundary_fidelity=False,
        single_layer=False,
        strip_exponent=False,
        method="dmrg",
        mode_="mps",
        sequence=None,
        cutoff=1.0e-12,
        equalize_norms=False,
        layer_tags=None,
    ):
        """Compute boundary-based infidelity for current ``(state, state_target)``.

        This reuses ``self.bdy`` and ``self.bdy_overlap``. If ``chi`` is
        provided and larger than current boundary bond dimension, both boundary
        objects are expanded before evaluation.

        Returns
        -------
        float
            Boundary infidelity value.
        """
        if self.state_target is None:
            raise ValueError(
                "state_target is required for infidelity(). "
                "Set it in constructor or via set_target()."
            )

        uses_quimb = getattr(self, "boundary_engine", "dmrg") == "quimb-mps"
        if uses_quimb and method == "dmrg":
            method = "mps"
        self._ensure_boundary_chi(chi)

        # boundary_infidelity() expects a scalar int chi; derive one from
        # the (possibly tuple) value after boundaries are already expanded.
        chi_bdy, chi_overlap = self._unpack_chi(chi)
        if chi_bdy is not None:
            chi_for_call = max(chi_bdy, chi_overlap)
        elif uses_quimb:
            chi_for_call = max(
                int(getattr(self.bdy, "chi", 1)),
                int(getattr(self.bdy_overlap, "chi", 1)),
            )
        elif norm_target is None:
            chi_for_call = max(int(getattr(self.bdy, "chi", 1)), int(getattr(self.bdy_overlap, "chi", 1)))
        else:
            chi_for_call = None

        result = self._call_with_accepted_kwargs(
            boundary_infidelity,
            self.state,
            self.state_target,
            chi=chi_for_call,
            norm=norm,
            norm_target=norm_target,
            bdy=None if uses_quimb else self.bdy,
            bdy_overlap=None if uses_quimb else self.bdy_overlap,
            contraction_opt=self.contraction_opt if contraction_opt is None else contraction_opt,
            n_iter=n_iter,
            direction=direction,
            max_separation=max_separation,
            progress=progress,
            track_boundary_fidelity=track_boundary_fidelity,
            fit_mode=self.fit_mode,
            single_layer=single_layer,
            strip_exponent=strip_exponent,
            method=method,
            mode_=mode_,
            sequence=sequence,
            cutoff=cutoff,
            equalize_norms=equalize_norms,
            layer_tags=layer_tags,
        )

        self._update_boundaries_from_result(result)

        return float(result["infidelity"])

    def metrics(self):
        """Return exact ``(fidelity, infidelity)`` for current state pair."""
        if self.state_target is None:
            raise ValueError(
                "state_target is required for metrics(). "
                "Set it in constructor or via set_target()."
            )
        fidelity = float(
            complex(
                tn_fidelity(
                    self.state,
                    self.state_target,
                    contraction_opt=self.contraction_opt,
                )
            ).real
        )
        if fidelity < 0.0 and abs(fidelity) < 1e-12:
            fidelity = 0.0
        infid = 1.0 - fidelity
        if infid < 0.0 and abs(infid) < 1e-12:
            infid = 0.0
        return fidelity, infid

    def _debug_loss(self, *, mode="exact", kwargs=None, env_n_iter=4):
        """Compute global diagnostic loss for debug/progress summaries."""
        mode = str(mode).strip().lower()
        kwargs = {} if kwargs is None else dict(kwargs)

        if mode == "exact":
            _fidel, loss = self.metrics()
            return float(loss)

        if mode == "infidelity":
            defaults = {
                "chi": getattr(self, "chi", None),
                "n_iter": int(env_n_iter),
                "direction": getattr(self, "direction", "y"),
                "max_separation": getattr(self, "max_separation", 1),
                "single_layer": bool(getattr(self, "single_layer", False)),
                "progress": False,
                "track_boundary_fidelity": False,
            }
            defaults.update(kwargs)
            return float(self.infidelity(**defaults))

        raise ValueError("debug_loss_mode must be 'exact' or 'infidelity'")

    def _approx_infidelity_loss(self, *, env_n_iter=4):
        """Return boundary-infidelity loss for lightweight global diagnostics."""
        try:
            result = self.infidelity(
                chi=self.chi,
                n_iter=env_n_iter,
                direction=self.direction,
                max_separation=self.max_separation,
                single_layer=self.single_layer,
                progress=False,
                track_boundary_fidelity=False,
            )
        except (AttributeError, ValueError, TypeError):
            return None
        return float(result)

    @staticmethod
    def _to_float_history(history):
        """Convert solver history entries into plain Python floats."""
        values = []
        for entry in history or ():
            value = entry
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            values.append(float(value))
        return values

    @staticmethod
    def _fidelity_to_infidelity(value):
        """Convert fidelity scalar to non-negative infidelity."""
        if value is None:
            return None
        infid = 1.0 - float(complex(value).real)
        if infid < 0.0 and abs(infid) < 1e-12:
            infid = 0.0
        return infid

    def _collect_axis_run_traces(self, axis_runs, *, cycle, axis):
        """Collect per-step traces from axis run records."""
        if not axis_runs:
            return

        for run_info in axis_runs:
            local_loss = run_info.get("exact_loss_after", run_info.get("loss_final"))
            if local_loss is None:
                step_loss = float("nan")
            else:
                step_loss = float(local_loss)
            self.step_loss_trace.append(step_loss)

            history = self._to_float_history(run_info.get("history"))
            self.inner_loss_traces.append(history)

            time_boundary = run_info.get("time_boundary")
            time_optimize = run_info.get("time_optimize")
            tb = None if time_boundary is None else float(time_boundary)
            to = None if time_optimize is None else float(time_optimize)
            if tb is None or to is None:
                ttot = None
            else:
                ttot = tb + to

            # Accept either fidelity or infidelity keys from run records and
            # populate both in step_trace for compatibility.
            boundary_fidelity_norm = run_info.get("boundary_fidelity_norm")
            boundary_fidelity_overlap = run_info.get("boundary_fidelity_overlap")
            boundary_infidelity_norm = run_info.get("boundary_infidelity_norm")
            boundary_infidelity_overlap = run_info.get("boundary_infidelity_overlap")

            if boundary_infidelity_norm is None and boundary_fidelity_norm is not None:
                boundary_infidelity_norm = self._fidelity_to_infidelity(boundary_fidelity_norm)
            if boundary_infidelity_overlap is None and boundary_fidelity_overlap is not None:
                boundary_infidelity_overlap = self._fidelity_to_infidelity(boundary_fidelity_overlap)

            if boundary_fidelity_norm is None and boundary_infidelity_norm is not None:
                boundary_fidelity_norm = 1.0 - float(boundary_infidelity_norm)
            if boundary_fidelity_overlap is None and boundary_infidelity_overlap is not None:
                boundary_fidelity_overlap = 1.0 - float(boundary_infidelity_overlap)

            self.step_trace.append(
                {
                    "cycle": int(cycle),
                    "axis": run_info.get("axis", axis),
                    "index": run_info.get("index"),
                    "move": run_info.get("sweep"),
                    "loss": step_loss,
                    "time_boundary": tb,
                    "time_optimize": to,
                    "time_total": ttot,
                    "boundary_fidelity_norm": boundary_fidelity_norm,
                    "boundary_fidelity_overlap": boundary_fidelity_overlap,
                    "boundary_infidelity_norm": boundary_infidelity_norm,
                    "boundary_infidelity_overlap": boundary_infidelity_overlap,
                }
            )
            self.fidels.append(
                {
                    "norm": boundary_infidelity_norm,
                    "overlap": boundary_infidelity_overlap,
                }
            )
            self.norm_trace.append(
                {
                    "bdy_norm": run_info.get("bdy_norm"),
                    "bdy_overlap_norm": run_info.get("bdy_overlap_norm"),
                }
            )

    def _axis_n(self, axis):
        if axis == "y":
            return self.Ly
        if axis == "x":
            return self.Lx
        raise ValueError("axis must be 'x' or 'y'")

    @staticmethod
    def _axis_tag(axis):
        if axis == "y":
            return "Y"
        if axis == "x":
            return "X"
        raise ValueError("axis must be 'x' or 'y'")

    def _site_tensor_tags(self, axis, index):
        if axis == "y":
            return [f"I{x},{index}" for x in range(self.Lx)]
        if axis == "x":
            return [f"I{index},{y}" for y in range(self.Ly)]
        raise ValueError("axis must be 'x' or 'y'")

    @staticmethod
    def _boundary_direction(axis, side):
        return f"{axis}_{side}"

    def _boundary_keys_for_index(self, index, axis):
        n = self._axis_n(axis)
        axis_tag = self._axis_tag(axis)
        if index < 0 or index > n - 1:
            raise ValueError(f"index must be in [0, {n - 1}] for axis={axis}")
        right_key = None if index == n - 1 else f"{axis_tag}{n - 2 - index}_r"
        left_key = f"{axis_tag}{index - 1}_l" if index > 0 else None
        return right_key, left_key

    def _prepare_current_double_layers(self):
        self.state, norm_tn = build_bra_ket(ket=self.state, bra=None)
        _, overlap_tn = build_bra_ket(ket=self.state_target, bra=self.state)

        return norm_tn, overlap_tn

    @staticmethod
    def _attach_boundaries(tn, boundaries, *, right_key=None, left_key=None):
        out = tn
        if right_key is not None:
            out = out | boundaries[right_key]
        if left_key is not None:
            out = out | boundaries[left_key]
        return out

    @staticmethod
    def _bra_with_reindexed_inner(local_tn):
        """Return ``local_tn.conj()`` with non-physical inner indices renamed."""
        bra = local_tn.conj()
        bra.reindex_(
            {
                idx: f"{idx}_*"
                for idx in bra.ind_map
                if not (isinstance(idx, str) and _PHYS_IND_PATTERN.fullmatch(idx))
            }
        )
        return bra

    @staticmethod
    def _resolve_user_solver(solver):
        """Validate solver names and emit practical warnings."""
        if not isinstance(solver, str):
            raise TypeError("solver must be a string")
        solver = solver.strip().lower()

        if solver not in SUPPORTED_SOLVERS:
            supported = ", ".join(SUPPORTED_SOLVERS)
            raise ValueError(f"Unsupported solver={solver!r}. Supported solvers: {supported}")

        if solver in {"nlopt", "fd-nlopt"}:
            warnings.warn(
                f"solver={solver!r} uses NLopt and can be sensitive to tolerances; "
                "consider tuning algorithm/maxeval/ftol_rel/xtol_rel.",
                UserWarning,
                stacklevel=2,
            )

        return solver

    def _estimate_slice_contraction_metrics(
        self,
        *,
        params_init,
        skeleton,
        slice_target,
        right_key,
        left_key,
    ):
        """Estimate FLOP/peak complexity for local norm and overlap networks."""
        local0 = qtn.unpack(params_init, skeleton)
        bra_norm0 = self._bra_with_reindexed_inner(local0)
        bra_overlap0 = local0.conj()
        norm_net0 = self._attach_boundaries(
            local0 | bra_norm0,
            self.bdy.mps_b,
            right_key=right_key,
            left_key=left_key,
        )
        overlap_net0 = self._attach_boundaries(
            slice_target | bra_overlap0,
            self.bdy_overlap.mps_b,
            right_key=right_key,
            left_key=left_key,
        )
        
        if self.simplify:
            norm_net0 = norm_net0.full_simplify(seq="R", split_method="svd", inplace=False)
            overlap_net0 = overlap_net0.full_simplify(seq="R", split_method="svd", inplace=False)

        tree_norm = norm_net0.contraction_tree(self.contraction_opt)
        tree_overlap = overlap_net0.contraction_tree(self.contraction_opt)
        flops_norm = float(tree_norm.contraction_cost(log=10))
        peak_norm = float(tree_norm.peak_size(log=2))
        flops_overlap = float(tree_overlap.contraction_cost(log=10))
        peak_overlap = float(tree_overlap.peak_size(log=2))
        return {
            "flops": max(flops_norm, flops_overlap),
            "peak_norm": peak_norm,
            "peak_overlap": peak_overlap,
        }

    def _optimize_packed_params(
        self,
        params_init,
        loss_fn,
        *,
        solver="scipy",
        solver_options=None,
    ):
        opts = self._merge_solver_options(solver_options)
        n_steps = int(opts.pop("n_steps", 30))
        # Allow callers to request a per-step gradient progress bar by setting
        # progress=True inside optimizer_options.  We pop it here so it never
        # reaches the backend solver (which doesn't know about it).
        grad_progress = bool(opts.pop("progress", False))
        runner = GradientOptimizer(
            solver=solver,
            n_steps=n_steps,
            options=opts,
            progress=grad_progress,
            verbose=False,
        )
        result = runner.run(params_init=params_init, loss_fn=loss_fn)
        return result.params, result.history

    def _apply_slice_update(self, index, params_opt, skeleton, axis):
        tn_opt = qtn.unpack(params_opt, skeleton)
        if not uses_symmray_arrays(tn_opt):
            tn_opt.balance_bonds_()

        for tag in self._site_tensor_tags(axis, index):
            self.state[tag].modify(data=tn_opt[tag].data)
        return tn_opt

    def _optimize_axis_slice_with_current_env(
        self,
        index,
        *,
        axis,
        solver="torch-adam",
        solver_options=None,
    ):
        axis_tag = self._axis_tag(axis)
        right_key, left_key = self._boundary_keys_for_index(index, axis)

        slice_state = self.state.select([f"{axis_tag}{index}"], "any")
        slice_target = self.state_target.select([f"{axis_tag}{index}"], "any")
        params_init, skeleton = qtn.pack(slice_state)
        metrics = self._estimate_slice_contraction_metrics(
            params_init=params_init,
            skeleton=skeleton,
            slice_target=slice_target,
            right_key=right_key,
            left_key=left_key,
        )

        def loss_fn(params_in):
            local = qtn.unpack(params_in, skeleton)
            bra_norm = self._bra_with_reindexed_inner(local)
            bra_overlap = local.conj()

            norm_net = self._attach_boundaries(
                local | bra_norm,
                self.bdy.mps_b,
                right_key=right_key,
                left_key=left_key,
            )
            overlap_net = self._attach_boundaries(
                slice_target | bra_overlap,
                self.bdy_overlap.mps_b,
                right_key=right_key,
                left_key=left_key,
            )

            if self.simplify:
                norm_net = norm_net.full_simplify(seq="R", split_method="svd", inplace=False)
                overlap_net = overlap_net.full_simplify(seq="R", split_method="svd", inplace=False)


            overlap_val = overlap_net.contract(
                all,
                optimize=self.contraction_opt,
                strip_exponent=True,
            )
            norm_val = norm_net.contract(
                all,
                optimize=self.contraction_opt,
                strip_exponent=True,
            )
            overlap_val = self._shift_scaled_exponent(
                overlap_val,
                self._local_overlap_exponent(),
            )
            norm_val = self._shift_scaled_exponent(
                norm_val,
                self._local_norm_exponent(),
            )
            fid = self._scaled_overlap_fidelity(
                overlap_val,
                norm_val,
                self.target_norm,
            )
            #infid = ar.do("clip", 1.0 - fid, 0.0, None)
            infid = 1. - fid
            return infid

        initial_loss = float(loss_fn(params_init))

        params_opt, history = self._optimize_packed_params(
            params_init,
            loss_fn,
            solver=solver,
            solver_options=solver_options,
        )
        history_values = self._to_float_history(history)
        final_loss = history_values[-1] if history_values else initial_loss
        params_opt = {
            k: v.detach().clone() if hasattr(v, "detach") else v
            for k, v in params_opt.items()
        }
        self._apply_slice_update(index, params_opt, skeleton, axis)
        # Track the best state using the minimum non-negative loss observed
        # during this local gradient optimization.
        self._maybe_store_best_state(self._best_nonnegative_from_history(history_values))
        return {
            "axis": axis,
            "index": index,
            "loss_initial": initial_loss,
            "loss_final": final_loss,
            "history": history_values,
            **metrics,
        }

    def _make_comp_pair(self, norm_tn, overlap_tn):
        comp_norm = CompBdy(norm_tn, self.bdy.mps_b, contraction_opt=self.contraction_opt, fit_mode=self.fit_mode)
        comp_overlap = CompBdy(overlap_tn, self.bdy_overlap.mps_b, contraction_opt=self.contraction_opt, fit_mode=self.fit_mode)
        return comp_norm, comp_overlap

    @staticmethod
    def _set_comp_norms(comp_norm, comp_overlap, *, norm_tn, overlap_tn):
        comp_norm.norm = norm_tn
        comp_overlap.norm = overlap_tn

    def _refresh_right_boundaries_once(self, axis, *, env_n_iter=4):
        norm_tn, overlap_tn = self._prepare_current_double_layers()
        if getattr(self, "boundary_engine", "dmrg") == "quimb-mps":
            del env_n_iter
            self._refresh_quimb_axis_boundaries(norm_tn, overlap_tn, axis)
            return
        comp_norm, comp_overlap = self._make_comp_pair(norm_tn, overlap_tn)
        comp_overlap.move_bdy(
            n_iter=env_n_iter,
            progress=False,
            direction=self._boundary_direction(axis, "right"),
            track_boundary_fidelity=False,
        )
        comp_norm.move_bdy(
            n_iter=env_n_iter,
            progress=False,
            direction=self._boundary_direction(axis, "right"),
            track_boundary_fidelity=False,
        )

    def _refresh_quimb_axis_boundaries(self, norm_tn, overlap_tn, axis, *, progress=False):
        """Rebuild Quimb MPS environments for the current double layers."""
        self.bdy.update_axis(norm_tn, axis, progress=progress)
        self.bdy_overlap.update_axis(overlap_tn, axis, progress=progress)
        return {"norm": None, "overlap": None}

    def _advance_boundary_one_step(
        self,
        index,
        *,
        side,
        axis,
        comp_norm,
        comp_overlap,
        env_n_iter=4,
        track_boundary_fidelity=False,
    ):
        n = self._axis_n(axis)
        if side == "left" and index <= 0:
            return {"norm": None, "overlap": None}
        if side == "right" and index >= n - 1:
            return {"norm": None, "overlap": None}
        pos = (index - 1) if side == "left" else (n - 2 - index)
        direction = self._boundary_direction(axis, side)
        for comp in (comp_overlap, comp_norm):
            comp.move_step_bdy(
                pos=pos,
                n_iter=env_n_iter,
                progress=False,
                direction=direction,
                track_boundary_fidelity=track_boundary_fidelity,
            )
        overlap_fidelity = None
        norm_fidelity = None
        if track_boundary_fidelity:
            if comp_overlap.fidel:
                overlap_fidelity = float(complex(comp_overlap.fidel[-1]).real)
            if comp_norm.fidel:
                norm_fidelity = float(complex(comp_norm.fidel[-1]).real)
        return {"norm": norm_fidelity, "overlap": overlap_fidelity}

    def _run_axis_half_sweep(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        indices,
        *,
        axis,
        update_side,
        sweep_name,
        solver="scipy",
        solver_options=None,
        env_n_iter=4,
        run_callback=None,
        track_boundary_fidelity=False,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
    ):
        """Run a single forward or backward half-sweep over *indices*."""
        runs = []
        comp_norm = None
        comp_overlap = None
        uses_quimb = getattr(self, "boundary_engine", "dmrg") == "quimb-mps"

        for index in indices:
            norm_tn, overlap_tn = self._prepare_current_double_layers()
            if uses_quimb:
                comp_norm = None
                comp_overlap = None
            elif comp_norm is None:
                comp_norm, comp_overlap = self._make_comp_pair(norm_tn, overlap_tn)
            else:
                self._set_comp_norms(comp_norm, comp_overlap, norm_tn=norm_tn, overlap_tn=overlap_tn)

            t0 = time.perf_counter()
            if uses_quimb:
                boundary_fidelity = self._refresh_quimb_axis_boundaries(
                    norm_tn,
                    overlap_tn,
                    axis,
                    progress=False,
                )
            else:
                boundary_fidelity = self._advance_boundary_one_step(
                    index,
                    side=update_side,
                    axis=axis,
                    comp_norm=comp_norm,
                    comp_overlap=comp_overlap,
                    env_n_iter=env_n_iter,
                    track_boundary_fidelity=track_boundary_fidelity,
                )
            boundary_infidelity_norm = self._fidelity_to_infidelity(
                boundary_fidelity.get("norm")
            )
            boundary_infidelity_overlap = self._fidelity_to_infidelity(
                boundary_fidelity.get("overlap")
            )
            t_bdy = time.perf_counter() - t0

            t0 = time.perf_counter()
            run_info = self._optimize_axis_slice_with_current_env(
                index,
                axis=axis,
                solver=solver,
                solver_options=solver_options,
            )
            t_opt = time.perf_counter() - t0

            run_info["sweep"] = sweep_name
            run_info["time_boundary"] = t_bdy
            run_info["time_optimize"] = t_opt
            run_info["boundary_infidelity_norm"] = boundary_infidelity_norm
            run_info["boundary_infidelity_overlap"] = boundary_infidelity_overlap
            if debug:
                try:
                    run_info["exact_loss_after"] = self._debug_loss(
                        mode=debug_loss_mode,
                        kwargs=debug_loss_kwargs,
                        env_n_iter=env_n_iter,
                    )
                except (AttributeError, TypeError, ValueError):
                    run_info["exact_loss_after"] = None
            try:
                run_info["bdy_norm"] = float(abs(self.bdy.norm))
            except Exception:
                run_info["bdy_norm"] = None
            try:
                run_info["bdy_overlap_norm"] = float(abs(self.bdy_overlap.norm))
            except Exception:
                run_info["bdy_overlap_norm"] = None
            runs.append(run_info)
            if run_callback is not None:
                run_callback(run_info)

        return runs

    def optimize_axis(
        self,
        axis,
        *,
        n_round_trips=1,
        solver="scipy",
        solver_options=None,
        env_n_iter=5,
        run_callback=None,
        track_boundary_fidelity=False,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
        renormalize=True,
    ):
        """Run one axis with forward + round-trip sweeps.

        Parameters
        ----------
        axis : {"x", "y"}
            Axis to sweep.
        n_round_trips : int, default=1
            Number of backward+forward round-trips after the initial forward pass.
        solver : str, default="scipy"
            Gradient solver name. Supported values: ``torch-adam``, ``scipy``,
            and ``nlopt``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        env_n_iter : int, default=4
            Local boundary-fit iterations per boundary move.
        run_callback : callable | None, default=None
            Optional callback called once per local slice update.
        track_boundary_fidelity : bool, default=False
            If ``True``, request boundary fidelity sampling during boundary moves.
        renormalize : bool, default=True
            If ``True``, normalize state at axis start and append to ``norm_trace``.
        """
        if self.state_target is None:
            raise ValueError(
                "state_target is required for optimize_axis(). "
                "Set it in constructor or via set_target()."
            )
        n = self._axis_n(axis)
        resolved_solver = self._resolve_user_solver(solver)
        all_runs = []

        self.bdy.normalize()
        self.bdy_overlap.normalize()

        if renormalize:
            old_norm = self._normalize_state(env_n_iter)
            self.norm_trace.append({"state_norm": float(abs(complex(old_norm)))})

        self._refresh_right_boundaries_once(axis, env_n_iter=env_n_iter)

        sweep_kwargs = dict(
            axis=axis,
            solver=resolved_solver,
            solver_options=solver_options,
            env_n_iter=env_n_iter,
            run_callback=run_callback,
            track_boundary_fidelity=track_boundary_fidelity,
            debug=debug,
            debug_loss_mode=debug_loss_mode,
            debug_loss_kwargs=debug_loss_kwargs,
        )

        all_runs.extend(
            self._run_axis_half_sweep(
                range(0, n),
                update_side="left",
                sweep_name="forward",
                **sweep_kwargs,
            )
        )

        for _trip in range(n_round_trips):
            all_runs.extend(
                self._run_axis_half_sweep(
                    range(n - 2, -1, -1),
                    update_side="right",
                    sweep_name="backward",
                    **sweep_kwargs,
                )
            )
            forward_start = 1 if n > 1 else n
            all_runs.extend(
                self._run_axis_half_sweep(
                    range(forward_start, n),
                    update_side="left",
                    sweep_name="forward",
                    **sweep_kwargs,
                )
            )

        return all_runs

    def _run_global_sweeps(
        self,
        *,
        axes=("y", "x"),
        n_cycles=1,
        n_round_trips=1,
        chi=None,
        solver="scipy",
        solver_options=None,
        env_n_iter=4,
        progress=True,
        progress_position=0,
        progress_leave=True,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
        track_boundary_fidelity=None,
        renormalize=True,
    ):
        """Run alternating axis sweeps and return a result dict.

        Parameters
        ----------
        chi : int | None, default=None
            If provided, expand stored boundaries to this bond dimension before
            running sweeps.
        solver : str, default="scipy"
            Gradient solver name. Supported values: ``torch-adam``, ``scipy``,
            and ``nlopt``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        env_n_iter : int, default=4
            Local boundary-fit iterations per boundary move.
        progress : bool, default=True
            Show global progress bar over all local updates.
        progress_position : int, default=0
            TQDM display row for nested progress bars.
        progress_leave : bool, default=True
            Whether the progress bar persists after completion. Set ``False``
            for nested sub-bars that should disappear when the run finishes.
        track_boundary_fidelity : bool | None, default=None
            Boundary-fidelity tracing flag for CompBdy updates. Defaults to
            ``False`` when not set explicitly.
        renormalize : bool, default=True
            If ``True``, normalize state at axis start and once again at run end.

        """
        if self.state_target is None:
            raise ValueError(
                "state_target is required for run(). "
                "Set it in constructor or via set_target()."
            )
        self._ensure_boundary_chi(chi)
        self._reset_run_traces()
        if track_boundary_fidelity is None:
            track_boundary_fidelity = bool(debug)
        else:
            track_boundary_fidelity = bool(track_boundary_fidelity)
        loss_mode = debug_loss_mode if debug else "infidelity"
        if (not debug) and (debug_loss_kwargs is None):
            loss_before = self._approx_infidelity_loss(env_n_iter=env_n_iter)
        else:
            try:
                loss_before = self._debug_loss(
                    mode=loss_mode,
                    kwargs=debug_loss_kwargs,
                    env_n_iter=env_n_iter,
                )
            except (AttributeError, TypeError, ValueError):
                loss_before = None
        all_runs = []
        axis_seq = list(axes)
        # Early-exit threshold: do not waste sweeps refining infidelity below
        # this floor (boundary contractions don't resolve smaller losses
        # reliably anyway).
        early_exit_tol = 1e-10
        early_exit = False

        # Skip the entire optimization if the initial loss is already at /
        # below the resolution floor (or a small negative artifact of finite
        # boundary chi). The current state is already good enough.
        if loss_before is not None and float(loss_before) < early_exit_tol:
            self._maybe_store_best_state(float(loss_before))
            return _AttrDict({
                "runs": [],
                "loss_before": loss_before,
                "loss_after": loss_before,
                "best_loss": (
                    None if not math.isfinite(float(self.best_loss))
                    else float(self.best_loss)
                ),
                "best_state": None if self.best_state is None else self.best_state.copy(),
                "loss": list(self.loss),
                "step_loss_trace": list(self.step_loss_trace),
                "step_trace": list(self.step_trace),
                "inner_loss_traces": [list(v) for v in self.inner_loss_traces],
                "norm_trace": list(self.norm_trace),
                "fidels": list(self.fidels),
                "bdy_norm": None,
                "bdy_overlap_norm": None,
                "converged": True,
                "early_exit": True,
            })

        def _steps_for_axis(axis_name):
            n = self._axis_n(axis_name)
            return n + (2 * n_round_trips * max(n - 1, 0))

        total_steps = n_cycles * sum(_steps_for_axis(axis_name) for axis_name in axis_seq)
        global_progress = None
        if progress:
            global_progress = tqdm(
                total=total_steps,
                desc="optimize",
                leave=progress_leave,
                position=progress_position,
                colour="gray",
                dynamic_ncols=True,
            )

        for cyc in range(n_cycles):
            for axis in axis_seq:
                def _on_run(run_info):
                    if global_progress is None:
                        return
                    global_progress.update(1)
                    local_loss = run_info.get("exact_loss_after", run_info.get("loss_final"))
                    best_now = getattr(self, "best_loss", float("inf"))
                    best_str = "na" if not math.isfinite(float(best_now)) else f"{float(best_now):.6e}"
                    head = f"loss=nan [best:{best_str}]"
                    if local_loss is not None:
                        cur = float(local_loss)
                        # Scientific notation keeps very small losses readable.
                        head = f"loss={cur:.6e} [best:{best_str}]"
                    t_opt = run_info.get("time_optimize")
                    t_bdy = run_info.get("time_boundary")
                    parts = []
                    if t_opt is not None or t_bdy is not None:
                        t_opt_s = "na" if t_opt is None else f"{float(t_opt):.2f}"
                        t_bdy_s = "na" if t_bdy is None else f"{float(t_bdy):.2f}"
                        parts.append(f"t={t_opt_s}/{t_bdy_s}s")
                    flops = run_info.get("flops")
                    peak_norm = run_info.get("peak_norm")
                    peak_overlap = run_info.get("peak_overlap")
                    peak_vals = [v for v in (peak_norm, peak_overlap) if v is not None]
                    peak2 = None if not peak_vals else float(max(peak_vals))
                    if flops is not None or peak2 is not None:
                        flops_s = "na" if flops is None else f"{float(flops):.2f}"
                        peak2_s = "na" if peak2 is None else f"{peak2:.2f}"
                        # cost=(log10 flops, log2 peak)
                        parts.append(f"cost=({flops_s},{peak2_s})")
                    axis_name = run_info.get("axis")
                    sweep_name = run_info.get("sweep")
                    index = run_info.get("index")
                    if axis_name is not None and sweep_name is not None and index is not None:
                        short = "fwd" if sweep_name == "forward" else "bwd"
                        parts.append(f"slice={axis_name}_{short}_{index}")
                    global_progress.set_description_str(head)
                    global_progress.set_postfix_str(" | ".join(parts))

                axis_runs = self._call_with_accepted_kwargs(
                    self.optimize_axis,
                    axis,
                    n_round_trips=n_round_trips,
                    solver=solver,
                    solver_options=solver_options,
                    env_n_iter=env_n_iter,
                    run_callback=_on_run,
                    track_boundary_fidelity=track_boundary_fidelity,
                    debug=debug,
                    debug_loss_mode=debug_loss_mode,
                    debug_loss_kwargs=debug_loss_kwargs,
                    renormalize=renormalize,
                )
                all_runs.extend(axis_runs)
                self._collect_axis_run_traces(
                    axis_runs,
                    cycle=cyc + 1,
                    axis=axis,
                )
                # Stop sweeping once the best observed loss is at the
                # boundary-MPS resolution floor.
                if math.isfinite(float(self.best_loss)) and float(self.best_loss) < early_exit_tol:
                    early_exit = True
                    break
            if early_exit:
                break

        # Restore the best snapshot so the returned state matches best_loss.
        if early_exit and self.best_state is not None:
            self.state = self.best_state.copy()
            self._set_boundary_pair(
                *self._call_with_accepted_kwargs(
                    self._build_boundary_pair,
                    self.state,
                    self.state_target,
                    chi=getattr(self, "chi", None),
                    single_layer=getattr(self, "single_layer", False),
                    boundary_engine=getattr(self, "boundary_engine", "dmrg"),
                    boundary_options=getattr(self, "boundary_options", None),
                )
            )

        if global_progress is not None:
            global_progress.close()

        if renormalize:
            old_norm = self._normalize_state(env_n_iter=env_n_iter)
            self.norm_trace.append({"state_norm": float(abs(complex(old_norm)))})

        if (not debug) and (debug_loss_kwargs is None):
            loss_after = self._approx_infidelity_loss(env_n_iter=env_n_iter)
        else:
            try:
                loss_after = self._debug_loss(
                    mode=loss_mode,
                    kwargs=debug_loss_kwargs,
                    env_n_iter=env_n_iter,
                )
            except (AttributeError, TypeError, ValueError):
                loss_after = None

        bdy_norm = None
        bdy_overlap_norm = None
        try:
            bdy_norm = float(abs(self.bdy.norm))
        except (AttributeError, TypeError, ValueError):
            bdy_norm = None
        try:
            bdy_overlap_norm = float(abs(self.bdy_overlap.norm))
        except (AttributeError, TypeError, ValueError):
            bdy_overlap_norm = None

        return _AttrDict({
            "runs": all_runs,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "best_loss": None if not math.isfinite(float(self.best_loss)) else float(self.best_loss),
            "best_state": None if self.best_state is None else self.best_state.copy(),
            "loss": list(self.loss),
            "step_loss_trace": list(self.step_loss_trace),
            "step_trace": list(self.step_trace),
            "inner_loss_traces": [list(v) for v in self.inner_loss_traces],
            "norm_trace": list(self.norm_trace),
            "fidels": list(self.fidels),
            "bdy_norm": bdy_norm,
            "bdy_overlap_norm": bdy_overlap_norm,
        })

    def run(
        self,
        *,
        n_cycles=None,
        chi=None,
        progress=None,
        debug=None,
        debug_loss_mode=None,
        debug_loss_kwargs=None,
        renormalize=None,
        track_boundary_fidelity=None,
    ):
        """High-level global sweep entrypoint.

        Parameters
        ----------
        n_cycles : int, default=1
            Number of global y/x cycles.
        chi : int | None, default=None
            If provided, expand boundary bond dimension before sweeps.
        progress : bool, default=True
            Show global sweep progress bar.
        renormalize : bool | None, default=None
            Override stored ``renormalize`` option for this call.
        """
        opts = dict(getattr(self, "optimize_kwargs", {}))
        if n_cycles is not None:
            opts["n_cycles"] = n_cycles
        if chi is not None:
            opts["chi"] = chi
        if progress is not None:
            opts["progress"] = progress
        if debug is not None:
            opts["debug"] = debug
        if debug_loss_mode is not None:
            opts["debug_loss_mode"] = debug_loss_mode
        if debug_loss_kwargs is not None:
            opts["debug_loss_kwargs"] = debug_loss_kwargs
        if renormalize is not None:
            opts["renormalize"] = renormalize
        if track_boundary_fidelity is not None:
            opts["track_boundary_fidelity"] = track_boundary_fidelity

        return self._run_global_sweeps(
            axes=opts.get("axes", ("y", "x")),
            n_cycles=opts.get("n_cycles", 1),
            n_round_trips=opts.get("n_round_trips", 1),
            chi=opts.get("chi"),
            solver=opts.get("optimizer", "scipy"),
            solver_options=opts.get("optimizer_options"),
            env_n_iter=opts.get("env_n_iter", 4),
            progress=opts.get("progress", True),
            progress_position=opts.get("progress_position", 0),
            progress_leave=opts.get("progress_leave", True),
            debug=opts.get("debug", False),
            debug_loss_mode=opts.get("debug_loss_mode", "exact"),
            debug_loss_kwargs=opts.get("debug_loss_kwargs"),
            renormalize=opts.get("renormalize", True),
            track_boundary_fidelity=opts.get("track_boundary_fidelity", False),
        )
