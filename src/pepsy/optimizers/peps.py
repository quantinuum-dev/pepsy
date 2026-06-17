"""PEPS/PEPO gate-stream optimization helpers."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from numbers import Integral
from typing import Any

from tqdm.auto import tqdm

from ..boundary.metrics import peps_infidelity as boundary_infidelity
from ..boundary.metrics import peps_normalize as boundary_normalize
from ..operators.gates import _normalize_gate_entries, gate as apply_gate
from ..tensors.core import reg_complex_svd_torch as _reg_complex_svd_torch
from .global_opt import GlobalOptimizer
from .sweep import SweepOptimizer

__all__ = ["PepsOptimizer"]

_DEFAULT_BOUNDARY_KWARGS = {
    "n_iter": 10,
    "direction": "y",
    "max_separation": 1,
    "track_boundary_fidelity": False,
    "strip_exponent": True,
}
_DEFAULT_GLOBAL_SEQUENCE = ("xmax", "xmin", "ymin", "ymax")
_DEFAULT_SWEEP_OPTIMIZE_KWARGS = {
    "n_round_trips": 4,
    "renormalize": False,
}
_DEFAULT_GLOBAL_OPTIMIZE_KWARGS = {
    "n": 100,
    "optimizer": "lbfgs",
}
_DEFAULT_GLOBAL_FALLBACK_KWARGS = {
    "n": 1,
    "optimizer": "lbfgs",
}


def _normalize_gate_queue(gates):
    """Return canonical ``[(gate, where, which), ...]`` gate entries."""
    entries = _normalize_gate_entries(
        gates,
        where=None,
        allow_empty=True,
        allow_which=True,
    )
    return [
        (gate_i, _freeze_where(where_i), which_i)
        for gate_i, where_i, which_i in entries
    ]


def _freeze_where(where):
    """Make location payloads stable for records without changing semantics."""
    if isinstance(where, list):
        return tuple(_freeze_where(item) for item in where)
    if isinstance(where, tuple):
        return tuple(_freeze_where(item) for item in where)
    return where


def _merge_opts(*options):
    """Merge optional mappings left-to-right, skipping ``None``."""
    merged = {}
    for option in options:
        if option:
            merged.update(dict(option))
    return merged


def _optimizer_key(optimizer):
    if not isinstance(optimizer, str):
        return ""
    return optimizer.strip().lower().replace("_", "-")


def _is_nlopt_optimizer(optimizer):
    key = _optimizer_key(optimizer)
    if key == "nlopt" or key.startswith("nlopt-"):
        return True
    upper = key.upper().replace("-", "_")
    return upper.startswith(("LD_", "LN_", "GD_", "GN_"))


class PepsOptimizer:  # pylint: disable=too-many-instance-attributes
    """Apply a PEPS/PEPO gate stream while keeping bonds capped at ``chi``.

    ``PepsOptimizer`` is a gate-by-gate compression driver. One-site gates are
    applied directly. For each two-site gate it first builds the exact
    post-gate target. If that target already fits inside ``chi`` it is accepted
    as-is. Otherwise a warm start is formed by compressing ``target.copy()`` to
    ``chi``; this warm start is either accepted by a boundary-fidelity check or
    refined against the exact target with ``mode="sweep"`` or ``mode="global"``.

    Boundary contractions use ``strip_exponent=True`` by default, so norm and
    overlap estimates are handled as ``(mantissa, exponent)`` pairs where the
    lower-level boundary helpers support it. The fidelity traces returned by
    :meth:`get_fidelities` and :meth:`get_infidelities` are local two-site gate
    monitors rather than exact global circuit fidelities; the cumulative proxy
    can be much looser than a direct overlap with an exact reference.

    Chi controls are intentionally split by role:

    - ``chi`` caps the PEPS/PEPO virtual bonds stored in the optimized state.
    - ``boundary_chi`` controls the boundary environments used inside the
      sweep/global optimizer backends.
    - ``normalize_chi`` controls calls to :func:`pepsy.peps_normalize`.
    - ``evaluation_chi`` controls calls to :func:`pepsy.peps_infidelity` used
      to accept/reject warm starts and optimized candidates.

    Parameters
    ----------
    state : qtn.TensorNetwork
        Initial PEPS/PEPO-like tensor network. It is copied unless
        ``inplace=True``.
    gates : sequence | None
        Canonical bundled gate stream ``[(gate, where), ...]`` or
        ``[(gate, where, which), ...]``. The latter can target ``"upper"``
        (``k...``) or ``"lower"`` (``b...``) physical-index families.
    chi : int
        Maximum trainable PEPS/PEPO virtual bond dimension.
    boundary_chi : int | tuple[int, int] | None, optional
        Boundary contraction bond dimension used by the sweep/global
        optimization backends. Defaults to ``chi``. Tuple values are forwarded
        to :class:`SweepOptimizer`; when no explicit ``normalize_chi`` or
        ``evaluation_chi`` is supplied, normalization uses the first entry and
        standalone infidelity estimates use the larger entry.
    normalize_chi : int | None, optional
        Boundary bond dimension used for PEPS normalization calls. Use this to
        normalize with a larger boundary than the trainable PEPS bond ``chi``
        or the optimizer environment ``boundary_chi``. If ``None``, the
        normalization chi follows ``boundary_chi``.
    evaluation_chi : int | None, optional
        Boundary bond dimension used for pre/post local infidelity estimates
        that decide whether the warm start or optimized candidate is accepted.
        This is the knob for stricter initial/final diagnostics. If ``None``,
        the evaluation chi follows ``boundary_chi``.
    mode : {"sweep", "global"}, default="sweep"
        Variational optimizer backend used when the warm start is not good
        enough.
    contraction_opt : str | object, optional
        Contraction path optimizer forwarded to boundary contractions and the
        variational backend. Defaults to ``"auto-hq"``.
    which : {"upper", "lower"} | None, optional
        Default physical-index family passed to :func:`pepsy.operators.gate`.
        Per-entry ``which`` values override this.
    inplace : bool, default=False
        If ``False``, copy ``state`` before applying gates.
    normalize_initial : bool, default=True
        Normalize the initial state once, on the first :meth:`run` call.
    boundary_kwargs : mapping, optional
        Shared PEPS boundary controls used for normalization, infidelity
        estimates, and sweep environment updates. Defaults are
        ``n_iter=10``, ``direction="y"``, ``max_separation=1``,
        ``track_boundary_fidelity=False``, and ``strip_exponent=True``.
    normalize_kwargs, infidelity_kwargs : mapping, optional
        Extra keyword arguments for boundary normalization and local infidelity
        estimates. These are merged after ``boundary_kwargs``.
    gate_kwargs, target_gate_kwargs, warmstart_gate_kwargs : mapping, optional
        Gate-application controls. ``target_gate_kwargs`` affect the exact
        target build. ``warmstart_gate_kwargs`` are used only by the routed
        warm-start fallback; the normal warm start is compressed from the
        already-built target.
    optimizer : str | None, optional
        Compact optimizer selection for the chosen variational backend. In
        sweep mode this maps to :class:`SweepOptimizer`'s local solver name.
        In global mode, ``"nlopt"`` together with
        ``optimizer_options={"algorithm": "LD_VAR2"}`` routes to
        :meth:`GlobalOptimizer.optimize_nlopt`.
    optimizer_options : mapping, optional
        Backend-specific optimizer controls. Sweep mode forwards this mapping
        as ``optimizer_options``. Global mode accepts practical aliases such as
        ``algorithm``, ``maxeval``/``n_steps``, tolerance keys, ``device``, and
        ``progress``.
    sweep_kwargs : mapping, optional
        Constructor options forwarded to :class:`SweepOptimizer`.
    sweep_optimize_kwargs : mapping, optional
        Per-run sweep controls. PEPS defaults use ``n_round_trips=4`` and
        ``renormalize=False``; pass explicit values here to override them.
    global_kwargs : mapping, optional
        Constructor options forwarded to :class:`GlobalOptimizer`.
    global_optimize_kwargs : mapping, optional
        Per-run global optimizer controls. The default global cleanup budget is
        ``n=100``; pass ``{"n": ...}`` to tune this sensitive value.
    global_fallback_kwargs : mapping, optional
        Global fallback controls used if an optional NLopt run raises an NLopt
        runtime error. Defaults to one LBFGS step.
    register_torch_svd : bool, default=True
        Register PEPSY's torch complex-SVD rule before torch global
        optimization.
    accept_if_improved : bool, default=True
        Keep the pre-optimization warm start when measured cleanup does not
        improve the local infidelity.
    """

    _ALLOWED_MODES = frozenset({"sweep", "global"})
    _PROGBAR_COLORS = {
        "sweep": "#1f77b4",
        "global": "#9467bd",
    }

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        state,
        gates=None,
        chi=None,
        *,
        boundary_chi=None,
        normalize_chi=None,
        evaluation_chi=None,
        mode="sweep",
        contraction_opt="auto-hq",
        which=None,
        inplace=False,
        normalize_initial=True,
        boundary_kwargs: Mapping[str, Any] | None = None,
        normalize_kwargs: Mapping[str, Any] | None = None,
        infidelity_kwargs: Mapping[str, Any] | None = None,
        gate_kwargs: Mapping[str, Any] | None = None,
        target_gate_kwargs: Mapping[str, Any] | None = None,
        warmstart_gate_kwargs: Mapping[str, Any] | None = None,
        optimizer=None,
        optimizer_options: Mapping[str, Any] | None = None,
        sweep_kwargs: Mapping[str, Any] | None = None,
        sweep_optimize_kwargs: Mapping[str, Any] | None = None,
        global_kwargs: Mapping[str, Any] | None = None,
        global_optimize_kwargs: Mapping[str, Any] | None = None,
        global_fallback_kwargs: Mapping[str, Any] | None = None,
        register_torch_svd=True,
        accept_if_improved=True,
    ):
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use PepsOptimizer(state, gates, chi) "
                    "or PepsOptimizer(state, chi) for an empty gate queue."
                )

        self.chi = self._validate_scalar_chi(chi, name="chi")
        self.boundary_chi = self._validate_boundary_chi(
            self.chi if boundary_chi is None else boundary_chi
        )
        self.normalize_chi = (
            None
            if normalize_chi is None
            else self._validate_scalar_chi(normalize_chi, name="normalize_chi")
        )
        self.evaluation_chi = (
            None
            if evaluation_chi is None
            else self._validate_scalar_chi(evaluation_chi, name="evaluation_chi")
        )
        self.mode = self._normalize_mode(mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.which = which
        self.inplace = bool(inplace)
        self.state = state if self.inplace else (state.copy() if hasattr(state, "copy") else state)
        self.gates = _normalize_gate_queue(gates)

        self.normalize_initial = bool(normalize_initial)
        self.boundary_kwargs = _merge_opts(
            _DEFAULT_BOUNDARY_KWARGS,
            boundary_kwargs,
        )
        self.normalize_kwargs = dict(normalize_kwargs or {})
        self.infidelity_kwargs = dict(infidelity_kwargs or {})
        self.gate_kwargs = dict(gate_kwargs or {})
        self.target_gate_kwargs = dict(target_gate_kwargs or {})
        self.warmstart_gate_kwargs = dict(warmstart_gate_kwargs or {})
        self.optimizer = optimizer
        self.optimizer_options = dict(optimizer_options or {})
        self.sweep_kwargs = dict(sweep_kwargs or {})
        self.sweep_optimize_kwargs = dict(sweep_optimize_kwargs or {})
        self.global_kwargs = dict(global_kwargs or {})
        self.global_optimize_kwargs = dict(global_optimize_kwargs or {})
        self.global_fallback_kwargs = _merge_opts(
            _DEFAULT_GLOBAL_FALLBACK_KWARGS,
            global_fallback_kwargs,
        )
        self.register_torch_svd = bool(register_torch_svd)
        self.accept_if_improved = bool(accept_if_improved)

        self._initial_normalized = False
        self._reset_traces()

    @classmethod
    def _normalize_mode(cls, mode):
        mode_norm = str(mode).strip().lower()
        if mode_norm not in cls._ALLOWED_MODES:
            allowed = ", ".join(sorted(cls._ALLOWED_MODES))
            raise ValueError(f"Unknown mode: {mode!r}. Expected one of: {allowed}.")
        return mode_norm

    @staticmethod
    def _validate_scalar_chi(chi, *, name):
        if not isinstance(chi, Integral):
            raise TypeError(f"{name} must be an integer.")
        chi = int(chi)
        if chi < 1:
            raise ValueError(f"{name} must be >= 1.")
        return chi

    @classmethod
    def _validate_boundary_chi(cls, chi):
        if isinstance(chi, (tuple, list)):
            if len(chi) != 2:
                raise ValueError("boundary_chi tuple must be length 2.")
            return (
                cls._validate_scalar_chi(chi[0], name="boundary_chi[0]"),
                cls._validate_scalar_chi(chi[1], name="boundary_chi[1]"),
            )
        return cls._validate_scalar_chi(chi, name="boundary_chi")

    def _boundary_chi_for_norm(self, override=None):
        if override is not None:
            return self._validate_scalar_chi(override, name="normalize_chi")
        if self.normalize_chi is not None:
            return int(self.normalize_chi)
        if isinstance(self.boundary_chi, tuple):
            return int(self.boundary_chi[0])
        return int(self.boundary_chi)

    def _boundary_chi_for_infidelity(self, override=None):
        if override is not None:
            return self._validate_scalar_chi(override, name="evaluation_chi")
        if self.evaluation_chi is not None:
            return int(self.evaluation_chi)
        if isinstance(self.boundary_chi, tuple):
            return max(int(self.boundary_chi[0]), int(self.boundary_chi[1]))
        return int(self.boundary_chi)

    def set_boundary_chi(
        self,
        boundary_chi=None,
        *,
        normalize_chi=None,
        evaluation_chi=None,
    ):
        """Update optimization, normalization, and diagnostic boundary chis.

        Parameters
        ----------
        boundary_chi : int | tuple[int, int] | None, optional
            Boundary chi used by the optimizer backends. ``None`` preserves the
            current value.
        normalize_chi : int | None, optional
            Boundary chi used by :meth:`normalize` and run-time normalization
            calls. ``None`` preserves the current value.
        evaluation_chi : int | None, optional
            Boundary chi used by :meth:`estimate_infidelity` and run-time
            accept/reject diagnostics. ``None`` preserves the current value.

        Returns
        -------
        PepsOptimizer
            ``self`` for chaining.
        """
        if boundary_chi is not None:
            self.boundary_chi = self._validate_boundary_chi(boundary_chi)
        if normalize_chi is not None:
            self.normalize_chi = self._validate_scalar_chi(
                normalize_chi,
                name="normalize_chi",
            )
        if evaluation_chi is not None:
            self.evaluation_chi = self._validate_scalar_chi(
                evaluation_chi,
                name="evaluation_chi",
            )
        return self

    def _reset_traces(self):
        self.losses = [1.0]
        self.infidelities = [0.0]
        self.local_infidelities = []
        self.step_records = []
        self.normalizations = []
        self._fidelity_log_sum = 0.0
        self._fidelity_count = 0
        self.last_result = None

    def set_state(self, state, *, normalize_initial=None):
        """Replace the current state and reset initial-normalization status."""
        self.state = state if self.inplace else (state.copy() if hasattr(state, "copy") else state)
        self._initial_normalized = False
        if normalize_initial is not None:
            self.normalize_initial = bool(normalize_initial)
        return self

    def set_mode(self, mode):
        """Switch variational backend between ``"sweep"`` and ``"global"``."""
        self.mode = self._normalize_mode(mode)
        return self

    def set_gates(self, gates):
        """Replace the queued gate stream."""
        self.gates = _normalize_gate_queue(gates)
        return self

    def add_gates(self, gates):
        """Append gates to the queued gate stream."""
        self.gates.extend(_normalize_gate_queue(gates))
        return self

    def normalize(self, state=None, *, normalize_chi=None, **kwargs):
        """Normalize ``state`` in place via PEPS boundary contraction.

        The optimizer's boundary defaults are used, including
        ``strip_exponent=True`` unless explicitly overridden. ``normalize_chi``
        temporarily overrides the constructor-level normalization chi for this
        call only.
        """
        state = self.state if state is None else state
        return self._normalize_state(
            state,
            normalize_kwargs=kwargs,
            normalize_chi=normalize_chi,
        )

    def _normalize_state(self, state, *, normalize_kwargs=None, normalize_chi=None):
        opts = _merge_opts(
            self.boundary_kwargs,
            self.normalize_kwargs,
            normalize_kwargs,
        )
        opts.setdefault("chi", self._boundary_chi_for_norm(normalize_chi))
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        old_norm = boundary_normalize(state, **opts)
        self.normalizations.append(self._normalization_record(state, old_norm))
        return old_norm

    def _ensure_initial_normalized(
        self,
        *,
        normalize_initial=None,
        normalize_kwargs=None,
        normalize_chi=None,
    ):
        do_normalize = self.normalize_initial if normalize_initial is None else bool(normalize_initial)
        if do_normalize and not self._initial_normalized:
            self._normalize_state(
                self.state,
                normalize_kwargs=normalize_kwargs,
                normalize_chi=normalize_chi,
            )
            self._initial_normalized = True

    @staticmethod
    def _max_bond(state):
        max_bond = getattr(state, "max_bond", None)
        if not callable(max_bond):
            return None
        return int(max_bond())

    def _site_count(self, where, state=None):
        """Infer how many physical sites a gate location targets."""
        state = self.state if state is None else state
        if isinstance(where, (str, Integral)):
            return 1
        if not isinstance(where, (tuple, list)) or len(where) == 0:
            raise ValueError("Invalid gate location.")
        if all(isinstance(item, str) for item in where):
            return len(where)
        if all(isinstance(item, Integral) for item in where):
            if hasattr(state, "Lz") or (hasattr(state, "Lx") and hasattr(state, "Ly")):
                return 1
            return len(where)
        return len(where)

    def _base_gate_options(self, *, cutoff, cutoff_mode, gate_kwargs=None):
        opts = {
            "contract": "reduce-split",
            "sequence": "auto",
            "path_canonize": True,
            "path_compress": False,
        }
        opts.update(self.gate_kwargs)
        opts.update(dict(gate_kwargs or {}))
        opts.setdefault("cutoff", cutoff)
        opts.setdefault("cutoff_mode", cutoff_mode)
        return opts

    def _target_gate_options(self, *, cutoff, cutoff_mode, gate_kwargs=None):
        opts = self._base_gate_options(
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )
        opts.pop("max_bond", None)
        opts.update(self.target_gate_kwargs)
        return opts

    def _warmstart_gate_options(self, *, cutoff, cutoff_mode, gate_kwargs=None):
        opts = self._base_gate_options(
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )
        opts.update(self.warmstart_gate_kwargs)
        opts["max_bond"] = self.chi
        return opts

    def _apply_gate_entry(self, state, gate_payload, where, which, *, opts, inplace=False):
        which_local = self.which if which is None else which
        return apply_gate(
            state,
            gate_payload,
            where=where,
            which=which_local,
            inplace=inplace,
            **opts,
        )

    def _build_target(self, state, gate_payload, where, which, *, cutoff, cutoff_mode, gate_kwargs):
        opts = self._target_gate_options(
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )
        state_work = state.copy() if hasattr(state, "copy") else state
        return self._apply_gate_entry(
            state_work,
            gate_payload,
            where,
            which,
            opts=opts,
            inplace=True,
        )

    def _build_routed_warmstart(
        self,
        state,
        gate_payload,
        where,
        which,
        *,
        cutoff,
        cutoff_mode,
        gate_kwargs,
    ):
        opts = self._warmstart_gate_options(
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )
        state_work = state.copy() if hasattr(state, "copy") else state
        out = self._apply_gate_entry(
            state_work,
            gate_payload,
            where,
            which,
            opts=opts,
            inplace=True,
        )
        return self._compress_to_chi(out, cutoff=cutoff)

    def _build_warmstart(
        self,
        target,
        state,
        gate_payload,
        where,
        which,
        *,
        cutoff,
        cutoff_mode,
        gate_kwargs,
    ):
        if hasattr(target, "copy"):
            warmstart = self._compress_to_chi(target.copy(), cutoff=cutoff)
            max_bond = self._max_bond(warmstart)
            if max_bond is None or max_bond <= self.chi:
                return warmstart

        return self._build_routed_warmstart(
            state,
            gate_payload,
            where,
            which,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )

    def _compress_to_chi(self, state, *, cutoff):
        max_bond = self._max_bond(state)
        if max_bond is None or max_bond <= self.chi:
            return state

        compress_all = getattr(state, "compress_all", None)
        if callable(compress_all):
            return compress_all(max_bond=self.chi, cutoff=cutoff, inplace=False)

        compress_all_ = getattr(state, "compress_all_", None)
        if callable(compress_all_):
            compress_all_(max_bond=self.chi, cutoff=cutoff)
        return state

    @staticmethod
    def _copy_and_mangle_target(target):
        target_opt = target.copy() if hasattr(target, "copy") else target
        mangle_inner = getattr(target_opt, "mangle_inner_", None)
        if callable(mangle_inner):
            mangle_inner()
        return target_opt

    @staticmethod
    def _clean_infidelity(value):
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get("infidelity")
        if value is None:
            return None
        value = float(complex(value).real)
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        return max(0.0, value)

    @staticmethod
    def _clip_fidelity(value):
        value = float(complex(value).real)
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        if value > 1.0 and abs(value - 1.0) < 1.0e-12:
            value = 1.0
        return min(1.0, max(0.0, value))

    @staticmethod
    def _trace_scalar(value):
        """Return a small Python scalar suitable for stored diagnostics."""
        if value is None:
            return None
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mantissa, exponent = value
            return (
                PepsOptimizer._trace_scalar(mantissa),
                PepsOptimizer._trace_scalar(exponent),
            )

        work = value
        for method_name in ("detach", "cpu"):
            method = getattr(work, method_name, None)
            if callable(method):
                try:
                    work = method()
                except Exception:  # pragma: no cover - best-effort tracing
                    break

        item = getattr(work, "item", None)
        if callable(item):
            try:
                work = item()
            except Exception:  # pragma: no cover - best-effort tracing
                pass

        try:
            scalar = complex(work)
        except (TypeError, ValueError):
            return repr(value)

        if abs(scalar.imag) <= 1.0e-15:
            return float(scalar.real)
        return scalar

    def _normalization_record(self, state, old_norm):
        """Build a lightweight normalization event without retaining ``state``."""
        return {
            "old_norm": self._trace_scalar(old_norm),
            "state_max_bond": self._max_bond(state),
        }

    def estimate_infidelity(self, state, target, *, evaluation_chi=None, **kwargs):
        """Estimate local boundary infidelity between normalized states.

        By default this assumes both inputs have norm one. Pass ``norm`` and
        ``norm_target`` when measuring unnormalized states. ``evaluation_chi``
        temporarily overrides the constructor-level diagnostic chi for this
        call only.
        """
        opts = _merge_opts(
            self.boundary_kwargs,
            self.infidelity_kwargs,
            kwargs,
        )
        opts.setdefault("chi", self._boundary_chi_for_infidelity(evaluation_chi))
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        opts.setdefault("norm", 1.0)
        opts.setdefault("norm_target", 1.0)
        return self._clean_infidelity(boundary_infidelity(state, target, **opts))

    def _sweep_boundary_kwargs(self, *, progress):
        opts = _merge_opts(self.boundary_kwargs)
        strip_exponent = opts.pop("strip_exponent", None)
        opts.setdefault("chi", self.boundary_chi)
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        # SweepOptimizer.run uses env_n_iter for boundary moves during sweeps.
        opt_kwargs = {
            "env_n_iter": opts.get("n_iter", _DEFAULT_BOUNDARY_KWARGS["n_iter"]),
            "track_boundary_fidelity": opts.get("track_boundary_fidelity", False),
            "progress": bool(progress),
            "progress_position": 1 if progress else 0,
        }
        return opts, opt_kwargs, strip_exponent

    def _apply_sweep_optimizer_options(self, opt_kwargs):
        opt_kwargs = dict(opt_kwargs or {})
        if self.optimizer is not None:
            opt_kwargs.setdefault("optimizer", self.optimizer)
        if self.optimizer_options:
            opt_kwargs.setdefault("optimizer_options", dict(self.optimizer_options))
        return opt_kwargs

    def _global_contraction_defaults(self, *, cutoff, progress):
        bopts = _merge_opts(self.boundary_kwargs)
        return {
            "contraction_opt": self.contraction_opt,
            "chi": self.boundary_chi,
            "mode": "mps",
            "mode_": "mps",
            "max_separation": bopts.get("max_separation", 1),
            "cutoff": cutoff,
            "progbar": bool(progress),
            "strip_exponent": bool(bopts.get("strip_exponent", True)),
        }

    def _global_loss_defaults(self, *, cutoff, progress):
        opts = self._global_contraction_defaults(cutoff=cutoff, progress=progress)
        opts["sequence"] = list(_DEFAULT_GLOBAL_SEQUENCE)
        opts["target_norm"] = 1.0
        return opts

    def _apply_global_optimizer_options(self, opt_kwargs):
        opt_kwargs = dict(opt_kwargs or {})
        option_payload = _merge_opts(
            self.optimizer_options,
            opt_kwargs.pop("optimizer_options", None),
        )
        if self.optimizer is not None:
            opt_kwargs.setdefault("optimizer", self.optimizer)

        if option_payload:
            options = dict(option_payload)
            algorithm = options.pop("algorithm", None)
            if algorithm is not None:
                if _optimizer_key(opt_kwargs.get("optimizer")) in {"", "nlopt"}:
                    opt_kwargs["optimizer"] = algorithm
                else:
                    opt_kwargs.setdefault("optimizer", algorithm)

            if "n" not in opt_kwargs:
                for n_key in ("maxeval", "n_steps"):
                    if n_key in options:
                        opt_kwargs["n"] = int(options.pop(n_key))
                        break
            else:
                options.pop("maxeval", None)
                options.pop("n_steps", None)

            if "progress" in options:
                opt_kwargs.setdefault("progbar", bool(options.pop("progress")))

            for key in (
                "tol",
                "jac",
                "hessp",
                "ftol_rel",
                "ftol_abs",
                "xtol_rel",
                "xtol_abs",
                "autodiff_backend",
                "device",
                "jit_fn",
            ):
                if key in options:
                    opt_kwargs.setdefault(key, options.pop(key))

            if not _is_nlopt_optimizer(opt_kwargs.get("optimizer")):
                opt_kwargs.update(options)

        opt_kwargs.setdefault("n", _DEFAULT_GLOBAL_OPTIMIZE_KWARGS["n"])
        opt_kwargs.setdefault("optimizer", _DEFAULT_GLOBAL_OPTIMIZE_KWARGS["optimizer"])
        return opt_kwargs

    def _maybe_register_torch_svd(self, opt_kwargs):
        if not self.register_torch_svd:
            return
        backend = opt_kwargs.get("autodiff_backend", "torch")
        if not isinstance(backend, str) or backend.strip().lower() != "torch":
            return
        try:
            _reg_complex_svd_torch()
        except ImportError as exc:
            warnings.warn(
                f"Could not register torch complex SVD rule: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _record_fidelity_progress(self, infidelity):
        infidelity = self._clean_infidelity(infidelity)
        if infidelity is None:
            return None, None

        fidelity = self._clip_fidelity(1.0 - infidelity)
        self.local_infidelities.append(infidelity)
        self._fidelity_count += 1

        if fidelity <= 0.0 or self._fidelity_log_sum == -math.inf:
            self._fidelity_log_sum = -math.inf
        else:
            self._fidelity_log_sum += math.log(fidelity)

        if self._fidelity_log_sum == -math.inf:
            cumulative_fidelity = 0.0
            geometric_fidelity = 0.0
        else:
            cumulative_fidelity = math.exp(self._fidelity_log_sum)
            geometric_fidelity = math.exp(self._fidelity_log_sum / self._fidelity_count)

        self.losses.append(float(geometric_fidelity))
        self.infidelities.append(float(1.0 - cumulative_fidelity))
        return float(fidelity), float(geometric_fidelity)

    def _optimize_with_sweep(
        self,
        state,
        target,
        *,
        progress,
        normalize_chi=None,
        sweep_kwargs=None,
        sweep_optimize_kwargs=None,
    ):
        target_opt = self._copy_and_mangle_target(target)
        boundary_init_kwargs, boundary_opt_kwargs, strip_exponent = self._sweep_boundary_kwargs(
            progress=progress,
        )
        init_kwargs = {
            "chi": self.boundary_chi,
            "target_norm": 1.0,
            "contraction_opt": self.contraction_opt,
            "renormalize_state": True,
            **boundary_init_kwargs,
        }
        init_kwargs.update(self.sweep_kwargs)
        init_kwargs.update(dict(sweep_kwargs or {}))
        normalize_payload = init_kwargs.get("normalize_kwargs")
        if normalize_chi is not None or self.normalize_chi is not None:
            normalize_payload = _merge_opts(
                normalize_payload,
                {"chi": self._boundary_chi_for_norm(normalize_chi)},
            )
        if strip_exponent is not None:
            normalize_payload = _merge_opts(
                normalize_payload,
                {"strip_exponent": bool(strip_exponent)},
            )
        if normalize_payload is not None:
            init_kwargs["normalize_kwargs"] = normalize_payload

        sweeper = SweepOptimizer(
            state=state,
            state_target=target_opt,
            **init_kwargs,
        )

        opt_kwargs = _merge_opts(
            _DEFAULT_SWEEP_OPTIMIZE_KWARGS,
            boundary_opt_kwargs,
            self.sweep_optimize_kwargs,
            sweep_optimize_kwargs,
        )
        opt_kwargs = self._apply_sweep_optimizer_options(opt_kwargs)
        opt_kwargs.setdefault("progress", bool(progress))
        if opt_kwargs:
            sweeper.set_optimize_kwargs(**opt_kwargs)

        result = sweeper.run()
        best_state = result.get("best_state") if isinstance(result, Mapping) else None
        state_out = best_state if best_state is not None else sweeper.state
        final_infidelity = None
        if isinstance(result, Mapping):
            final_infidelity = result.get("best_loss")
            if final_infidelity is None:
                final_infidelity = result.get("loss_after")
        return (
            state_out,
            self._clean_infidelity(final_infidelity),
            self._summarize_sweep_result(result),
        )

    def _optimize_with_global(
        self,
        state,
        target,
        *,
        progress,
        cutoff,
        normalize_chi=None,
        global_kwargs=None,
        global_optimize_kwargs=None,
    ):
        target_opt = self._copy_and_mangle_target(target)
        init_kwargs = {"chi": self.boundary_chi}
        init_kwargs.update(self.global_kwargs)
        init_kwargs.update(dict(global_kwargs or {}))

        norm_defaults = self._global_contraction_defaults(
            cutoff=cutoff,
            progress=False,
        )
        normalize_defaults = dict(norm_defaults)
        if normalize_chi is not None or self.normalize_chi is not None:
            normalize_defaults["chi"] = self._boundary_chi_for_norm(normalize_chi)
        loss_defaults = self._global_loss_defaults(
            cutoff=cutoff,
            progress=False,
        )
        init_kwargs["norm_kwargs"] = _merge_opts(
            norm_defaults,
            init_kwargs.get("norm_kwargs"),
        )
        if init_kwargs.get("normalize_kwargs") is None:
            init_kwargs["normalize_kwargs"] = dict(normalize_defaults)
        else:
            init_kwargs["normalize_kwargs"] = _merge_opts(
                normalize_defaults,
                init_kwargs.get("normalize_kwargs"),
            )
        loss_kwargs = dict(init_kwargs.get("loss_kwargs") or {})
        loss_kwargs = _merge_opts(loss_defaults, loss_kwargs)
        init_kwargs["loss_kwargs"] = loss_kwargs
        if init_kwargs.get("loss_opt") is not None:
            init_kwargs["loss_opt"] = _merge_opts(
                loss_defaults,
                init_kwargs.get("loss_opt"),
            )

        optimizer = GlobalOptimizer(
            state=state,
            state_target=target_opt,
            **init_kwargs,
        )

        opt_kwargs = _merge_opts(self.global_optimize_kwargs, global_optimize_kwargs)
        opt_kwargs = self._apply_global_optimizer_options(opt_kwargs)
        opt_kwargs.setdefault("progbar", bool(progress))
        optimizer_name = opt_kwargs.get("optimizer", "adam")
        use_nlopt = _is_nlopt_optimizer(optimizer_name)
        self._maybe_register_torch_svd(opt_kwargs)
        fallback_used = False
        fallback_error = None
        if use_nlopt:
            try:
                out = optimizer.optimize_nlopt(**opt_kwargs)
            except Exception as exc:  # pragma: no cover - optional nlopt path
                if "nlopt" not in type(exc).__module__:
                    raise
                fallback_error = exc
                warnings.warn(
                    f"NLopt optimization failed; falling back to lbfgs: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fallback_kwargs = _merge_opts(self.global_fallback_kwargs)
                fallback_kwargs.setdefault("progbar", bool(progress))
                self._maybe_register_torch_svd(fallback_kwargs)
                out = optimizer.optimize(**fallback_kwargs)
                fallback_used = True
        else:
            out = optimizer.optimize(**opt_kwargs)

        losses = None
        if isinstance(out, tuple) and len(out) == 2:
            state_out, losses = out
        else:
            state_out = out
        if losses is None:
            losses = tuple(getattr(optimizer, "losses", ()))
        final_infidelity = losses[-1] if losses else None
        return (
            state_out,
            self._clean_infidelity(final_infidelity),
            self._summarize_global_result(
                opt_kwargs=opt_kwargs,
                losses=losses,
                fallback_used=fallback_used,
                fallback_error=fallback_error,
            ),
        )

    def _summarize_sweep_result(self, result):
        """Return scalar sweep diagnostics without retaining TN objects."""
        summary = {"backend": "sweep"}
        if not isinstance(result, Mapping):
            return summary

        for key in (
            "loss_before",
            "loss_after",
            "best_loss",
            "bdy_norm",
            "bdy_overlap_norm",
        ):
            if key in result and result[key] is not None:
                summary[key] = self._trace_scalar(result[key])

        runs = result.get("runs")
        if runs is not None:
            summary["n_runs"] = len(runs)
        for source_key, target_key in (
            ("loss", "loss_count"),
            ("step_loss_trace", "step_loss_count"),
            ("step_trace", "step_count"),
            ("inner_loss_traces", "inner_loss_trace_count"),
            ("norm_trace", "norm_count"),
            ("fidels", "fidelity_count"),
        ):
            values = result.get(source_key)
            if values is not None:
                summary[target_key] = len(values)

        for key in ("converged", "early_exit"):
            if key in result and result[key] is not None:
                summary[key] = bool(result[key])
        return summary

    def _summarize_global_result(
        self,
        *,
        opt_kwargs,
        losses,
        fallback_used,
        fallback_error,
    ):
        """Return scalar global-optimizer diagnostics without retaining objects."""
        losses = tuple(losses or ())
        summary = {
            "backend": "global",
            "optimizer": str(opt_kwargs.get("optimizer", "")),
            "n": int(opt_kwargs.get("n", 0)),
            "loss_count": len(losses),
            "fallback_used": bool(fallback_used),
        }
        if losses:
            summary["loss_initial"] = self._trace_scalar(losses[0])
            summary["loss_final"] = self._trace_scalar(losses[-1])
        if fallback_error is not None:
            summary["fallback_error"] = str(fallback_error)
            summary["fallback_error_type"] = type(fallback_error).__name__
        return summary

    def _optimize_state(
        self,
        state,
        target,
        *,
        mode,
        progress,
        cutoff,
        normalize_chi,
        sweep_kwargs,
        sweep_optimize_kwargs,
        global_kwargs,
        global_optimize_kwargs,
    ):
        if mode == "sweep":
            return self._optimize_with_sweep(
                state,
                target,
                progress=progress,
                normalize_chi=normalize_chi,
                sweep_kwargs=sweep_kwargs,
                sweep_optimize_kwargs=sweep_optimize_kwargs,
            )
        if mode == "global":
            return self._optimize_with_global(
                state,
                target,
                progress=progress,
                normalize_chi=normalize_chi,
                global_kwargs=global_kwargs,
                global_optimize_kwargs=global_optimize_kwargs,
                cutoff=cutoff,
            )
        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _real_float(value):
        return float(complex(value).real)

    @staticmethod
    def _format_progress_infidelity(value):
        """Format progress infidelity in compact scientific notation."""
        text = f"{PepsOptimizer._real_float(value):#.0e}"
        if "e" not in text:
            return text
        mantissa, exponent = text.split("e", 1)
        sign = exponent[0] if exponent[:1] in "+-" else ""
        digits = exponent[1:] if sign else exponent
        digits = digits.lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"

    def _progress_postfix(
        self,
        *,
        reason,
        final_infidelity,
        two_site_count=None,
        target_max_bond=None,
    ):
        postfix = {
            "2q": self._fidelity_count if two_site_count is None else int(two_site_count),
            "bnd": self._max_bond(self.state),
            "why": reason,
        }
        if target_max_bond is not None:
            postfix["tgt"] = int(target_max_bond)
        if self._fidelity_count:
            postfix["Igeo"] = self._format_progress_infidelity(1.0 - self.losses[-1])
            postfix["Icum"] = self._format_progress_infidelity(self.infidelities[-1])
        if final_infidelity is not None:
            postfix["I"] = self._format_progress_infidelity(final_infidelity)
        return postfix

    def run(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        self,
        *,
        mode=None,
        progbar=False,
        progress=None,
        cutoff=1.0e-12,
        cutoff_mode="rel",
        non_unitary=False,
        normalize_target=None,
        normalize_initial=None,
        normalize_chi=None,
        evaluation_chi=None,
        normalize_final=True,
        infidelity_tol=1.0e-10,
        measure_infidelity=True,
        optimize=True,
        measure_final_infidelity=True,
        accept_if_improved=None,
        improvement_tol=0.0,
        gate_kwargs: Mapping[str, Any] | None = None,
        normalize_kwargs: Mapping[str, Any] | None = None,
        infidelity_kwargs: Mapping[str, Any] | None = None,
        sweep_kwargs: Mapping[str, Any] | None = None,
        sweep_optimize_kwargs: Mapping[str, Any] | None = None,
        global_kwargs: Mapping[str, Any] | None = None,
        global_optimize_kwargs: Mapping[str, Any] | None = None,
    ):
        """Run the queued gate stream and return the compressed state.

        One-site gates are applied directly. Two-site gates are applied exactly
        to form a target, then compressed to ``chi`` to form a warm start. The
        warm start is accepted immediately when it is already inside
        ``infidelity_tol``; otherwise it can be refined by the selected sweep or
        global optimizer.

        Parameters
        ----------
        mode : {"sweep", "global"} | None, optional
            Temporary backend override for this run.
        progress, progbar : bool, optional
            Show the outer PEPS progress bar. ``progress`` is preferred;
            ``progbar`` is kept as a short alias.
        cutoff, cutoff_mode
            Truncation settings passed to gate application and target-derived
            warm-start compression.
        non_unitary : bool, default=False
            If ``True``, target states produced by gates are explicitly
            normalized before fidelity estimates and optimization.
        normalize_target : bool | None, default=None
            Override whether target states are normalized. The default follows
            ``non_unitary``.
        normalize_initial : bool | None, optional
            Override the constructor's one-time initial normalization setting.
        normalize_chi : int | None, optional
            Per-run PEPS normalization boundary chi override. This affects
            initial, target, warm-start, and final-candidate normalization
            calls made during this run.
        evaluation_chi : int | None, optional
            Per-run boundary chi override for pre/post infidelity estimates.
            This is the recommended way to judge acceptance with a larger chi
            than the optimizer environment uses.
        normalize_final : bool, default=True
            Normalize an optimized candidate before measuring and accepting it.
        infidelity_tol : float, default=1e-10
            Accept the chi-truncated warm start without optimization when its
            estimated infidelity is at or below this threshold.
        measure_infidelity : bool, default=True
            Measure the warm-start local infidelity before deciding whether to
            optimize.
        optimize : bool, default=True
            If ``False``, accept the warm start after the optional infidelity
            estimate.
        measure_final_infidelity : bool, default=True
            Re-estimate infidelity after variational cleanup before deciding
            whether to accept the optimized state.
        accept_if_improved : bool | None, default=None
            If true, keep the chi-truncated warm start whenever the measured
            optimized state is not better than the warm start. The default uses
            the constructor setting.
        improvement_tol : float, default=0.0
            Required local-infidelity improvement when ``accept_if_improved`` is
            enabled.
        gate_kwargs, normalize_kwargs, infidelity_kwargs : mapping, optional
            Per-run overrides merged after the constructor-level settings.
        sweep_kwargs, sweep_optimize_kwargs : mapping, optional
            Per-run sweep backend and optimizer overrides.
        global_kwargs, global_optimize_kwargs : mapping, optional
            Per-run global backend and optimizer overrides.
        """
        if mode is not None:
            self.set_mode(mode)
        run_mode = self.mode
        show_progress = bool(progbar if progress is None else progress)
        normalize_target = bool(non_unitary) if normalize_target is None else bool(normalize_target)
        accept_if_improved = (
            self.accept_if_improved
            if accept_if_improved is None
            else bool(accept_if_improved)
        )

        self._ensure_initial_normalized(
            normalize_initial=normalize_initial,
            normalize_kwargs=normalize_kwargs,
            normalize_chi=normalize_chi,
        )

        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(self.gates),
                desc=f"peps-{run_mode}",
                unit="gate",
                leave=True,
                position=0,
                ascii=True,
                dynamic_ncols=True,
                colour=self._PROGBAR_COLORS[run_mode],
            )

        two_site_count = 0
        for step, (gate_payload, where, which) in enumerate(self.gates, start=1):
            site_count = self._site_count(where, self.state)
            reason = "1q"
            pre_infidelity = None
            final_infidelity = None
            target_max_bond = None
            optimizer_result = None
            optimized = False
            optimizer_attempted = False
            post_infidelity = None
            opt_infidelity = None

            if site_count == 1:
                opts = self._base_gate_options(
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    gate_kwargs=gate_kwargs,
                )
                self.state = self._apply_gate_entry(
                    self.state,
                    gate_payload,
                    where,
                    which,
                    opts=opts,
                    inplace=True,
                )
                if normalize_target:
                    self._normalize_state(
                        self.state,
                        normalize_kwargs=normalize_kwargs,
                        normalize_chi=normalize_chi,
                    )
                final_state = self.state
            elif site_count == 2:
                two_site_count += 1
                state_before = self.state
                target = self._build_target(
                    state_before,
                    gate_payload,
                    where,
                    which,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    gate_kwargs=gate_kwargs,
                )
                if normalize_target:
                    self._normalize_state(
                        target,
                        normalize_kwargs=normalize_kwargs,
                        normalize_chi=normalize_chi,
                    )
                target_max_bond = self._max_bond(target)

                if target_max_bond is not None and target_max_bond <= self.chi:
                    self.state = target
                    final_state = self.state
                    final_infidelity = 0.0
                    reason = "within_chi"
                else:
                    warmstart = self._build_warmstart(
                        target,
                        state_before,
                        gate_payload,
                        where,
                        which,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        gate_kwargs=gate_kwargs,
                    )
                    self._normalize_state(
                        warmstart,
                        normalize_kwargs=normalize_kwargs,
                        normalize_chi=normalize_chi,
                    )

                    if measure_infidelity:
                        pre_infidelity = self.estimate_infidelity(
                            warmstart,
                            target,
                            evaluation_chi=evaluation_chi,
                            **dict(infidelity_kwargs or {}),
                        )

                    if pre_infidelity is not None and pre_infidelity <= infidelity_tol:
                        self.state = warmstart
                        final_state = self.state
                        final_infidelity = pre_infidelity
                        reason = "below_tol"
                    elif not optimize:
                        self.state = warmstart
                        final_state = self.state
                        final_infidelity = pre_infidelity
                        reason = "warmstart"
                    else:
                        optimizer_attempted = True
                        warmstart_snapshot = None
                        if accept_if_improved and pre_infidelity is not None:
                            warmstart_snapshot = (
                                warmstart.copy()
                                if hasattr(warmstart, "copy")
                                else warmstart
                            )
                        if pbar is not None:
                            pbar.set_postfix(self._progress_postfix(
                                reason=f"opt:{run_mode}",
                                final_infidelity=pre_infidelity,
                                two_site_count=two_site_count,
                                target_max_bond=target_max_bond,
                            ))
                        final_state, opt_infidelity, optimizer_result = self._optimize_state(
                            warmstart,
                            target,
                            mode=run_mode,
                            progress=show_progress,
                            cutoff=cutoff,
                            normalize_chi=normalize_chi,
                            sweep_kwargs=sweep_kwargs,
                            sweep_optimize_kwargs=sweep_optimize_kwargs,
                            global_kwargs=global_kwargs,
                            global_optimize_kwargs=global_optimize_kwargs,
                        )
                        candidate_state = final_state
                        if normalize_final:
                            self._normalize_state(
                                candidate_state,
                                normalize_kwargs=normalize_kwargs,
                                normalize_chi=normalize_chi,
                            )
                        if measure_infidelity and measure_final_infidelity:
                            post_infidelity = self.estimate_infidelity(
                                candidate_state,
                                target,
                                evaluation_chi=evaluation_chi,
                                **dict(infidelity_kwargs or {}),
                            )
                        final_infidelity = (
                            post_infidelity
                            if post_infidelity is not None
                            else opt_infidelity
                        )
                        if final_infidelity is None:
                            final_infidelity = pre_infidelity

                        should_accept = True
                        if (
                            accept_if_improved
                            and pre_infidelity is not None
                            and final_infidelity is not None
                        ):
                            should_accept = (
                                float(final_infidelity)
                                < float(pre_infidelity) - float(improvement_tol)
                            )

                        if should_accept:
                            self.state = candidate_state
                            final_state = self.state
                            reason = "optimized"
                            optimized = True
                        else:
                            self.state = (
                                warmstart_snapshot
                                if warmstart_snapshot is not None
                                else warmstart
                            )
                            final_state = self.state
                            # This was measured before optimization on the
                            # warm start snapshot that we just restored.
                            final_infidelity = pre_infidelity
                            reason = "optimizer_rejected"

                fidelity, geometric_fidelity = self._record_fidelity_progress(final_infidelity)
                record = {
                    "step": int(step),
                    "where": where,
                    "which": self.which if which is None else which,
                    "target_max_bond": target_max_bond,
                    "state_max_bond": self._max_bond(self.state),
                    "normalize_chi": self._boundary_chi_for_norm(normalize_chi),
                    "evaluation_chi": self._boundary_chi_for_infidelity(evaluation_chi),
                    "pre_infidelity": pre_infidelity,
                    "optimizer_infidelity": opt_infidelity,
                    "post_infidelity": post_infidelity,
                    "final_infidelity": final_infidelity,
                    "fidelity": fidelity,
                    "geometric_fidelity": geometric_fidelity,
                    "optimized": optimized,
                    "optimizer_attempted": optimizer_attempted,
                    "reason": reason,
                    "optimizer_result": optimizer_result,
                }
                self.step_records.append(record)
            else:
                raise ValueError("PepsOptimizer supports one- and two-site gates only.")

            self.last_result = {
                "step": int(step),
                "where": where,
                "state": final_state,
                "reason": reason,
                "infidelity": final_infidelity,
            }
            if pbar is not None:
                pbar.set_postfix(self._progress_postfix(
                    reason=reason,
                    final_infidelity=final_infidelity,
                    two_site_count=two_site_count,
                    target_max_bond=target_max_bond,
                ))
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return self.state

    def get_fidelities(self):
        """Return the running geometric mean of measured local fidelities.

        This is a per-two-site-gate monitor, not an exact global fidelity. For
        small systems, use a direct overlap with a high-chi or dense reference
        when you need the true state-vs-ideal fidelity.
        """
        return list(self.losses)

    def get_infidelities(self):
        """Return ``1 - prod(F_local)`` for measured two-site gate fidelities.

        This cumulative trace is a local-fidelity proxy, not an exact global
        state-vs-ideal infidelity. It can overestimate the true global
        infidelity by orders of magnitude because later gates can coherently
        rotate or partially recover earlier local truncation errors.
        """
        return list(self.infidelities)

    def get_local_infidelities(self):
        """Return measured per-two-site-gate infidelities."""
        return list(self.local_infidelities)

    def get_step_records(self):
        """Return lightweight per-two-site-gate bookkeeping records."""
        return list(self.step_records)

    def get_normalizations(self):
        """Return lightweight normalization events recorded by this optimizer."""
        return list(self.normalizations)
