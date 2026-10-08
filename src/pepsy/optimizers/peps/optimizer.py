"""PEPS/PEPO gate-stream optimization helpers."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import Any
from copy import deepcopy

import autoray as ar

from ..._internal.cutoff import resolve_fit_rtol
from ...boundary._fit_policy import (
    _FIT_QUIMB_MODES,
    _SWEEP_BOUNDARY_INIT_KEYS,
    _canonical_fit_layer_mode,
    _canonical_fit_layer_order,
    _canonical_fit_mode_selector,
)
from ...boundary.metrics import peps_infidelity as boundary_infidelity
from ...boundary.metrics import peps_norm as boundary_norm
from ...boundary.metrics import peps_normalize as boundary_normalize
from ...boundary.metrics import _as_scaled_scalar, _normalize_by_scaled_norm, _scaled_overlap_fidelity
from ...boundary.metrics import build_bra_ket
from ...boundary.states import BdyMPS
from ...boundary._reuse import EnvironmentCache, StripEnvironmentCache, _array_stamp
from ...backends import (
    TorchLinalgConfig,
    register_jax_linalg,
    resolve_backend_sample_data_from_tn,
    to_float as _backend_to_float,
)
from ...operators.gates import (
    _normalize_gate_entries,
    _resolve_gate_cutoff,
    _resolve_gate_cutoff_mode,
    gate as apply_gate,
)
from ..global_opt import GlobalOptimizer
from ...tensors.contractions import build_optimizer
from ..sweep import SweepOptimizer
from ..sweep.environments import (
    canonical_boundary_engine_selector,
    normalize_boundary_engine,
    symmray_array_backends,
    uses_symmray_arrays,
)
from ._timing import profile_run, timed_phase
from ._boundary_convergence import chi_pair, convergence_options, select_boundary_chi
from ._gate_order import gate_order_option, ordered_gate_run

__all__ = ["PepsOptimizer"]

_DEFAULT_BOUNDARY_KWARGS = {
    "n_iter": 10,
    "fit_init_strategy": "guess-src",
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
_DEFAULT_SWEEP_SOLVER = "nlopt"
_DEFAULT_SWEEP_NLOPT_ALGORITHM = "LD_LBFGS"
_DEFAULT_SWEEP_NLOPT_OPTIONS = {
    "algorithm": _DEFAULT_SWEEP_NLOPT_ALGORITHM,
    "maxeval": 50,
    "ftol_rel": 1e-9,
    "ftol_abs": 1e-9,
    "xtol_rel": 1e-9,
    "restore_best": True,
}
_DEFAULT_GLOBAL_OPTIMIZE_KWARGS = {
    "n": 1200,
    "optimizer": "LD_VAR2",
}
_DEFAULT_GLOBAL_FALLBACK_KWARGS = {
    "n": 1,
    "optimizer": "lbfgs",
}
_TRACE_FIDELITY_FLOOR = 1.0e-15
_UNSET_METRIC_CHI = object()


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


def _prefer_boundary_engine_mps(opts, boundary_engine, *states, normalize=False):
    """Route metric contractions to Quimb MPS when the PEPS engine chooses it."""
    if normalize_boundary_engine(boundary_engine, *states) != "quimb-mps":
        return opts

    method = str(opts.get("method", "dmrg")).strip().lower().replace("-", "_")
    if method in {"", "auto", "dmrg", "fit"}:
        opts["method"] = "mps"
    opts.setdefault("mode_", "mps")
    if normalize:
        opts.setdefault("balance_bonds", False)
    return opts


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

    ``PepsOptimizer`` is a gate-stream compression driver. One-site gates are
    applied directly or folded into an ordered batch. For each batch it builds the exact
    post-gate target. If that target already fits inside ``chi`` it is accepted
    after output normalization. Otherwise a warm start is formed by compressing ``target.copy()`` to
    ``chi``; this warm start is either accepted by a boundary-fidelity check or
    refined against the exact target with ``mode="sweep"`` or ``mode="global"``.
    By default ``run(k_2q_batch="auto")`` batches gates in circuit order while
    the target bonds stay within ``2 * chi`` for diagonal qubit gates or
    ``4 * chi`` for other gates. A larger exact single-gate target is processed
    alone. Positive integers select a fixed two-site gate count.

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

    Important cautions
    ------------------
    - A second :meth:`run` call applies the queued gates again to the current
      state. The default ``reset_traces=True`` resets diagnostics only; use
      :meth:`set_state` when you want to replay from a fresh input state.
    - If ``normalize_chi`` or ``evaluation_chi`` is left as ``None``, standalone
      normalization and infidelity diagnostics use ``(4 * chi, 5 * chi)``
      as norm/overlap caps. Normalization uses only the first entry.
    - ``accept_if_improved=True`` is most meaningful with
      ``measure_final_infidelity=True``. If final measurement is disabled, the
      optimizer loss used as a fallback can come from the coarser
      ``boundary_chi`` environment while the pre-check used ``evaluation_chi``.
    - Symmray PEPS inputs must be Torch-backed. Their sweep cleanup defaults
      to NLopt ``LD_LBFGS`` while obtaining gradients from Torch autograd;
      provide both the PEPS and gates with the same Torch Symmray backend.
    - Unitary evolution is assumed by default: generated targets are not
      rescaled. ``non_unitary=True`` enables target normalization, and
      ``normalize_target`` explicitly overrides that policy. Retained outputs
      are normalized using ``normalize_chi`` by default.
      This does not implement the interval scheduling or norm-proxy machinery
      available in :class:`pepsy.optimizers.mps.MpsOptimizer`.
    - Step records and fidelity traces are per measured two-site gate batch.
      One-site gates applied outside a two-site batch are not recorded as
      separate steps. For PEPS lattice gates, prefer coordinate-tuple sites
      like ``((x0, y0), (x1, y1))`` over flat integer pairs.

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
        optimization backends. Defaults to ``(4 * chi, 5 * chi)`` for norm and
        overlap environments, respectively. A scalar sets both caps equally.
        Normalization and evaluation defaults are independent of this override.
    normalize_chi : int | tuple[int, int] | None, optional
        Boundary cap or norm/overlap pair used for PEPS normalization calls.
        Defaults to ``(4 * chi, 5 * chi)``; only the norm (first) entry is used.
    evaluation_chi : int | tuple[int, int] | None, optional
        Boundary bond dimension used for pre/post local infidelity estimates
        that decide whether the warm start or optimized candidate is accepted.
        This is the knob for stricter initial/final diagnostics. If ``None``,
        the evaluation chi defaults to ``(4 * chi, 5 * chi)``. Measured state
        norms use the first entry; the overlap uses the second. A scalar sets
        both. Unitary runs assume target norm one unless explicitly overridden.
    evaluation_max_retries : int, default=2
        Retry a substantially negative finite-cap infidelity with equal norm
        and overlap caps twice the previous maximum, up to this many times.
        Each retry warns and is recorded. Zero keeps the requested caps strict.
        Supplied norms (including the unitary run's target norm one) and exact
        contractions are never retried.
    evaluation_negative_tol : float, default=1e-3
        Approximate infidelities outside [0, 1] by at most this absolute tolerance
        warn and are clipped for decisions and fidelity bookkeeping. Raw values
        remain in evaluation and batch records. The dtype roundoff allowance is
        a lower bound; zero restores roundoff-only handling. Exact contractions
        retain roundoff-only handling regardless of this setting.
    mode : {"sweep", "global", "full-update"}, default="sweep"
        Variational optimizer backend used when the warm start is not good
        enough. ``full-update`` is an alias for sweep with two-site updates.
    update_style : {"row-column", "row", "column", "two-site"}, default="row-column"
        Sweep local update scope. Row/column fits use the existing variational
        solver; two-site uses reduced ALS and processes each two-site gate.
    gate_order : {"input", "column", "row"}, default="input"
        Reorder commuting two-qubit blocks into strip traversal. Disjoint gates
        commute; overlapping gates must both be diagonal. ``last_gate_order`` maps the executed
        traversal to one-based original queue positions; the queue is preserved.
    contraction_opt : str | object, optional
        Contraction path optimizer forwarded to boundary contractions and the
        variational backend. None builds Pepsy's reusable Cotengra optimizer.
    which : {"upper", "lower"} | None, optional
        Default physical-index family passed to :func:`pepsy.operators.gate`.
        Per-entry ``which`` values override this.
    inplace : bool, default=False
        If ``False``, copy ``state`` before applying gates.
    normalize_initial : bool, default=True
        Normalize the initial state once, on the first :meth:`run` call.
    fit_* : optional
        The boundary FIT controls accepted by :class:`SweepOptimizer` may be
        supplied directly here (for example ``fit_mode="dmrg2"`` or
        ``fit_layer_mode="sequential"``). Direct values override matching
        entries in ``boundary_kwargs``. Leaving them as ``None`` preserves the
        mapping-based compatibility API and the lower-level defaults.
    boundary_kwargs : mapping, optional
        Shared PEPS boundary controls used for normalization, infidelity
        estimates, and sweep environment updates. Defaults are
        ``n_iter=10``, ``direction="y"``, ``max_separation=1``,
        ``track_boundary_fidelity=False``, and ``strip_exponent=True``.
        Boundary FIT uses ``fit_init_strategy="guess-src"`` by default;
        explicit direct arguments or mapping entries override this policy.
        FIT controls such as ``fit_mode``, ``fit_layer_mode``, ``layer_tags``,
        the ``fit_*`` initialization/convergence options, and ``cutoff`` are
        shared across all three paths. Metric-only controls such as
        ``method``, ``mode_``, ``sequence``, and ``equalize_norms`` remain
        valid for standalone metric calls but are not passed to the delegated
        ``SweepOptimizer`` constructor. ``balance_bonds`` is a normalization-
        only option and belongs in ``normalize_kwargs``.
    boundary_engine : {"auto", "dmrg", "quimb-mps"}, default="auto"
        Boundary engine used by sweep cleanup. ``"auto"`` keeps dense inputs
        on Pepsy ``BdyMPS``/``CompBdy`` boundaries and routes Symmray-looking
        inputs to Quimb MPS boundaries. The same selector supplies default
        ``method="mps"`` metric contractions when Quimb MPS is selected.
    boundary_convergence : bool | mapping, default=True
        Before sweep refinement, recompute both state norms and their complex
        overlap in x and y at increasing boundary caps. Mapping controls are
        ``rtol=1e-5``, ``atol=1e-8`` (normalized overlap), ``schedule='d2'``,
        ``start_chi='auto'`` (D squared), ``max_chi='auto'`` (8 D squared),
        ``patience=2``, and ``warm_start=True``. Explicit start/max values
        accept scalar or norm/overlap pairs. D-squared probes add D squared
        and replace fixed norm/overlap caps for adaptive sweeps. The alternative
        ``schedule='geometric'`` uses fixed caps as minima and ``growth=2``.
        Reused DMRG boundaries are refitted and confirmed with fresh guesses.
        Selected caps stay fixed during fitting and final normalization.
        Norms must be positive and real to ``max(rtol, dtype_roundoff)``
        relative accuracy, fidelity physical within dtype roundoff, and estimates
        stable across caps and directions. Accepted caps are retained as a
        floor across runs; contraction values are always recomputed. Warn and
        continue at the limit if convergence fails. This is an empirical
        pre-fit check, not a guarantee throughout optimization. False retains
        fixed-cap behavior; global mode and optimize=False do not probe.
    boundary_options : mapping | None, optional
        Extra options forwarded to :class:`SweepOptimizer`'s Quimb MPS
        boundary store, such as ``cutoff``, ``canonize``, ``compress_opts``,
        ``equalize_norms``, ``layer_tags``, and ``mode``.
        These configure the reusable Quimb sweep environment; standalone
        :meth:`normalize` and :meth:`estimate_infidelity` use
        ``normalize_kwargs``/``infidelity_kwargs`` for metric-only options.
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
        sweep mode this maps to :class:`SweepOptimizer`'s local solver name;
        when left as ``None`` the sweep local solver defaults to NLopt
        ``LD_LBFGS``. Torch-backed Symmray states still supply gradients via
        Torch autograd. In global mode, ``"nlopt"`` together with
        ``optimizer_options={"algorithm": "LD_VAR2"}`` routes to
        :meth:`GlobalOptimizer.optimize_nlopt`.
    optimizer_options : mapping, optional
        Backend-specific optimizer controls. Sweep NLopt defaults are
        ``algorithm="LD_LBFGS"``, ``maxeval=50``, ``ftol_rel=ftol_abs=1e-9``,
        ``xtol_rel=1e-9``, and ``restore_best=True``. These limits apply to
        each local slice solve, not the complete circuit. Global mode accepts aliases such as
        ``algorithm``, ``maxeval``/``n_steps``, tolerance keys, ``device``, and
        ``progress``.
    sweep_kwargs : mapping, optional
        Constructor options forwarded to :class:`SweepOptimizer`.
    sweep_optimize_kwargs : mapping, optional
        Per-run sweep controls. PEPS defaults use ``n_round_trips=4`` and
        ``renormalize=False``; pass explicit values here to override them.
    sweep_progress : bool | None, default=None
        Show the internal directional sweep progress bar independently of
        the outer PEPS bar. ``None`` follows the outer ``progress`` setting;
        ``False`` hides it and ``True`` enables it.
    full_update_kwargs : mapping, optional
        Dense Torch/CuPy nearest-neighbor two-site ALS controls for
        ``mode="full-update"``: ``max_iterations=50``, ``rtol="auto"``,
        dtype-dependent ``rcond=None``, ``gauge=True``, and ``solver="auto"``.
        Auto uses existing Quimb ALS with a weighted-QR fallback. This mode keeps
        the outside tensors fixed and always processes one two-site gate.
        Positive ``refine_sweeps`` enables additional fixed-rank strip ALS
        (default zero) with ``refine_rtol="auto"``. ``accumulate_local_infidelity=True`` retains
        a running product of local pair fidelities alongside per-gate records.
    global_kwargs : mapping, optional
        Constructor options forwarded to :class:`GlobalOptimizer`.
        Torch global norm/overlap contractions default to ``cutoff=1e-10``;
        explicit ``norm_kwargs``, ``normalize_kwargs``, and ``loss_kwargs``
        or ``loss_opt`` values take precedence.
    global_optimize_kwargs : mapping, optional
        Per-run global optimizer controls. The default global cleanup budget is
        ``n=1200``; pass ``{"n": ...}`` to tune this sensitive value.
        Selecting ``autodiff_backend="jax"`` defaults to ``jit_fn=True`` and
        global loss options ``cutoff=0``, ``strip_exponent=False``. Explicit
        global optimizer and loss options take precedence. JAX global cleanup
        registers Pepsy's truncation-safe SVD derivative before tracing;
        QR continues to use native JAX autodiff.
    global_fallback_kwargs : mapping, optional
        Global fallback controls used if an optional NLopt run raises an NLopt
        runtime error. Defaults to one LBFGS step.
    torch_linalg_config : TorchLinalgConfig | None, optional
        One policy for Torch SVD autodiff, QR autodiff, CUDA/CPU SVD drivers,
        and Quimb's raw Symmray split paths. When omitted, global Torch
        cleanup creates a stabilized policy and infers ``mode`` from the
        state. Pass a policy explicitly when choosing native versus
        regularized SVD or a specific exact backend.
    register_torch_svd : bool, default=True
        Automatically register the Torch linalg policy before Torch global
        optimization. The legacy name is retained for compatibility; prefer
        ``torch_linalg_config=TorchLinalgConfig(...)`` for new code.
    accept_if_improved : bool, default=True
        Keep the pre-optimization warm start when measured cleanup does not
        improve the local infidelity.
    """

    _ALLOWED_MODES = frozenset({"sweep", "global", "full-update"})
    _PROGBAR_COLORS = {
        "sweep": "#1f77b4",
        "global": "#9467bd",
        "full-update": "#2ca02c",
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
        evaluation_max_retries=2,
        evaluation_negative_tol=1.0e-3,
        boundary_convergence=True,
        mode="sweep",
        update_style="row-column",
        gate_order="input",
        contraction_opt=None,
        which=None,
        inplace=False,
        normalize_initial=True,
        fit_mode=None,
        fit_layer_mode=None,
        fit_layer_order=None,
        layer_tags=None,
        fit_init_strategy=None,
        fit_init_seed=None,
        fit_block_size=None,
        fit_adaptive_sweeps=None,
        fit_max_bond=None,
        fit_sweep_sequence=None,
        fit_cutoff_mode=None,
        fit_compression_opts=None,
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=None,
        fit_timing=None,
        fit_timing_sync_device=None,
        boundary_kwargs: Mapping[str, Any] | None = None,
        boundary_engine="auto",
        boundary_options: Mapping[str, Any] | None = None,
        normalize_kwargs: Mapping[str, Any] | None = None,
        infidelity_kwargs: Mapping[str, Any] | None = None,
        gate_kwargs: Mapping[str, Any] | None = None,
        target_gate_kwargs: Mapping[str, Any] | None = None,
        warmstart_gate_kwargs: Mapping[str, Any] | None = None,
        optimizer=None,
        optimizer_options: Mapping[str, Any] | None = None,
        sweep_kwargs: Mapping[str, Any] | None = None,
        sweep_optimize_kwargs: Mapping[str, Any] | None = None,
        sweep_progress: bool | None = None,
        full_update_kwargs: Mapping[str, Any] | None = None,
        global_kwargs: Mapping[str, Any] | None = None,
        global_optimize_kwargs: Mapping[str, Any] | None = None,
        global_fallback_kwargs: Mapping[str, Any] | None = None,
        torch_linalg_config: TorchLinalgConfig | None = None,
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
            (4 * self.chi, 5 * self.chi) if boundary_chi is None else boundary_chi
        )
        self.normalize_chi = (
            None
            if normalize_chi is None
            else self._validate_boundary_chi(normalize_chi, name="normalize_chi")
        )
        self.evaluation_chi = (
            None
            if evaluation_chi is None
            else self._validate_boundary_chi(evaluation_chi, name="evaluation_chi")
        )
        self.evaluation_max_retries = self._validate_evaluation_retries(evaluation_max_retries)
        self.boundary_convergence = convergence_options(boundary_convergence, bond_dim=self.chi)
        self._boundary_chi_floor = None
        self.evaluation_negative_tol = float(evaluation_negative_tol)
        if not math.isfinite(self.evaluation_negative_tol) or self.evaluation_negative_tol < 0:
            raise ValueError("evaluation_negative_tol must be finite and non-negative.")
        self.mode = self._normalize_mode(mode)
        self.update_style = self._normalize_update_style(update_style)
        self.gate_order = gate_order_option(gate_order)
        self.last_gate_order = None
        from ._full_update import options as full_update_options
        self.full_update_kwargs = full_update_options(full_update_kwargs)
        self.contraction_opt = (
            build_optimizer(progbar=False) if contraction_opt is None else contraction_opt
        )
        self.which = which
        self.inplace = bool(inplace)
        self.state = state if self.inplace else (state.copy() if hasattr(state, "copy") else state)
        self._require_torch_symmray_backend(self.state, role="state")
        self.gates = _normalize_gate_queue(gates)

        self.normalize_initial = bool(normalize_initial)
        direct_fit_kwargs = {
            key: value
            for key, value in {
                "fit_mode": fit_mode,
                "fit_layer_mode": fit_layer_mode,
                "fit_layer_order": fit_layer_order,
                "layer_tags": layer_tags,
                "fit_init_strategy": fit_init_strategy,
                "fit_init_seed": fit_init_seed,
                "fit_block_size": fit_block_size,
                "fit_adaptive_sweeps": fit_adaptive_sweeps,
                "fit_max_bond": fit_max_bond,
                "fit_sweep_sequence": fit_sweep_sequence,
                "fit_cutoff_mode": fit_cutoff_mode,
                "fit_compression_opts": fit_compression_opts,
                "fit_min_iter": fit_min_iter,
                "fit_rtol": fit_rtol,
                "fit_patience": fit_patience,
                "fit_timing": fit_timing,
                "fit_timing_sync_device": fit_timing_sync_device,
            }.items()
            if value is not None
        }
        self.boundary_kwargs = _merge_opts(
            _DEFAULT_BOUNDARY_KWARGS,
            boundary_kwargs,
            direct_fit_kwargs,
        )
        self.boundary_engine = canonical_boundary_engine_selector(boundary_engine)
        self._validate_boundary_fit_policy()
        self.boundary_options = dict(boundary_options or {})
        self.normalize_kwargs = dict(normalize_kwargs or {})
        self.infidelity_kwargs = dict(infidelity_kwargs or {})
        self.gate_kwargs = dict(gate_kwargs or {})
        self.target_gate_kwargs = dict(target_gate_kwargs or {})
        self.warmstart_gate_kwargs = dict(warmstart_gate_kwargs or {})
        self.optimizer = optimizer
        self.optimizer_options = dict(optimizer_options or {})
        self.sweep_kwargs = dict(sweep_kwargs or {})
        self.sweep_optimize_kwargs = dict(sweep_optimize_kwargs or {})
        self._sweep_local_contraction_opt = self.contraction_opt if contraction_opt is None else None
        self.sweep_progress = None if sweep_progress is None else bool(sweep_progress)
        self.global_kwargs = dict(global_kwargs or {})
        self.global_optimize_kwargs = dict(global_optimize_kwargs or {})
        self.global_fallback_kwargs = _merge_opts(
            _DEFAULT_GLOBAL_FALLBACK_KWARGS,
            global_fallback_kwargs,
        )
        if torch_linalg_config is not None and not isinstance(
            torch_linalg_config,
            TorchLinalgConfig,
        ):
            raise TypeError(
                "torch_linalg_config must be a TorchLinalgConfig instance or None."
            )
        self.torch_linalg_config = torch_linalg_config
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
    def _validate_boundary_chi(cls, chi, *, name="boundary_chi"):
        if isinstance(chi, (tuple, list)):
            if len(chi) != 2:
                raise ValueError(f"{name} tuple must be length 2.")
            return (
                cls._validate_scalar_chi(chi[0], name=f"{name}[0]"),
                cls._validate_scalar_chi(chi[1], name=f"{name}[1]"),
            )
        return cls._validate_scalar_chi(chi, name=name)

    @staticmethod
    def _validate_evaluation_retries(value):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("evaluation_max_retries must be a non-negative integer.")
        return int(value)

    @staticmethod
    def _require_torch_symmray_backend(*states, role="inputs"):
        """Require a single Torch backend whenever Symmray data is present."""
        states = tuple(state for state in states if state is not None)
        symmray_flags = tuple(uses_symmray_arrays(state) for state in states)
        if not any(symmray_flags):
            return
        if not all(symmray_flags):
            raise TypeError(
                "PepsOptimizer requires matching Torch-backed Symmray arrays "
                f"for {role}; do not mix Symmray and dense tensor networks."
            )
        backends = symmray_array_backends(*states)
        if backends == {"torch"}:
            return
        found = ", ".join(sorted(backends)) or "unknown"
        raise TypeError(
            "PepsOptimizer requires Torch-backed Symmray arrays for autograd; "
            f"{role} use backend(s): {found}. Convert the PEPS and every gate "
            "with pepsy.backend_torch(...) before optimization."
        )

    def _resolve_metric_chi(self, override, configured, stored, options, *, name):
        """Resolve metric caps consistently for contractions and run records."""
        if options is not None and "chi" in options:
            value = options["chi"]
        elif override is not None:
            value = override
        elif configured is not None:
            value = configured
        elif "chi" in stored:
            value = stored["chi"]
        else:
            value = self.boundary_kwargs.get("chi", (4 * self.chi, 5 * self.chi))
        if value is None:
            return None
        return self._validate_boundary_chi(value, name=name)

    def _boundary_chi_for_norm(self, override=None, *, options=None):
        return self._resolve_metric_chi(
            override, self.normalize_chi, self.normalize_kwargs, options,
            name="normalize_chi",
        )

    def _boundary_chi_for_infidelity(self, override=None, *, options=None):
        return self._resolve_metric_chi(
            override, self.evaluation_chi, self.infidelity_kwargs, options,
            name="evaluation_chi",
        )

    def _resolve_infidelity_tol(self, value):
        """Use the current PEPS precision without transferring tensor data."""
        if isinstance(value, str) and value.strip().lower() == "auto":
            sample = resolve_backend_sample_data_from_tn(self.state)
            return resolve_fit_rtol("auto", dtype=getattr(sample, "dtype", "complex128"))
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("infidelity_tol must be 'auto' or a non-negative number.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("infidelity_tol must be 'auto' or a non-negative number.")
        return value

    def _validate_boundary_fit_policy(self):
        """Validate constructor-level FIT/layer policy before any contraction."""
        fit_mode = _canonical_fit_mode_selector(
            self.boundary_kwargs.get("fit_mode", "eff")
        )
        fit_layer_mode = _canonical_fit_layer_mode(
            self.boundary_kwargs.get("fit_layer_mode", "joint")
        )
        _canonical_fit_layer_order(
            self.boundary_kwargs.get("fit_layer_order", "input")
        )
        if fit_layer_mode == "sequential" and fit_mode not in _FIT_QUIMB_MODES:
            raise ValueError(
                "fit_layer_mode='sequential' is only supported with direct "
                f"Quimb fit modes; got fit_mode={fit_mode!r}."
            )

        metric_method = self.boundary_kwargs.get("method")
        if (
            fit_layer_mode == "sequential"
            and metric_method is not None
            and str(metric_method).strip().lower().replace("-", "_") != "dmrg"
        ):
            raise ValueError(
                "fit_layer_mode='sequential' requires method='dmrg' for "
                "Pepsy layered compression; native Quimb MPS metrics handle "
                "layer tags jointly."
            )

        if (
            fit_layer_mode == "sequential"
            and normalize_boundary_engine(self.boundary_engine, self.state)
            == "quimb-mps"
        ):
            raise ValueError(
                "fit_layer_mode='sequential' is not supported by the native "
                "Quimb MPS boundary engine; use boundary_engine='dmrg' for "
                "direct layered compression."
            )

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
        normalize_chi : int | tuple[int, int] | None, optional
            Boundary chi used by :meth:`normalize` and run-time normalization
            calls. ``None`` preserves the current value.
        evaluation_chi : int | tuple[int, int] | None, optional
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
            self.normalize_chi = self._validate_boundary_chi(
                normalize_chi,
                name="normalize_chi",
            )
        if evaluation_chi is not None:
            self.evaluation_chi = self._validate_boundary_chi(
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
        self.fit_diagnostics = []
        self.evaluation_records = []
        self.boundary_convergence_records = []
        self._last_evaluation_chi = None
        self._last_evaluation_raw_infidelity = None
        self._last_target_norm = 1.0
        self._fidelity_log_sum = 0.0
        self._fidelity_count = 0
        self._local_gate_log_fidelity = 0.0
        self._local_gate_fidelity_count = 0
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

    @staticmethod
    def _normalize_update_style(value):
        value = str(value).lower().strip()
        if value not in {'row-column', 'row', 'column', 'two-site'}:
            raise ValueError("update_style must be 'row-column', 'row', 'column', or 'two-site'")
        return value

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

    @timed_phase("normalization")
    def _normalize_state(self, state, *, normalize_kwargs=None, normalize_chi=None):
        opts = _merge_opts(
            self.boundary_kwargs,
            self.normalize_kwargs,
            normalize_kwargs,
        )
        opts["chi"] = self._boundary_chi_for_norm(normalize_chi, options=normalize_kwargs)
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        _prefer_boundary_engine_mps(opts, self.boundary_engine, state, normalize=True)
        requested_info = bool(opts.get("return_info", False))
        if opts.get("fit_timing", False):
            # Timing records live on BoundaryContractResult. Preserve the
            # historical scalar return from PepsOptimizer.normalize unless
            # the caller explicitly requested structured information.
            opts["return_info"] = True
        result = boundary_normalize(state, **opts)
        self._append_fit_diagnostics(result)
        old_norm = getattr(result, "cost", result)
        self.normalizations.append(self._normalization_record(state, old_norm))
        return result if requested_info else old_norm

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
    def _normalize_without_rescaling_sites(state, checked_norm, phase_site=None):
        """Store global magnitude in the exponent, changing at most one tensor."""
        mantissa, exponent = _as_scaled_scalar(checked_norm)
        magnitude = abs(complex(mantissa))
        if not math.isfinite(magnitude) or magnitude == 0 or not math.isfinite(exponent):
            raise ValueError('Cannot normalize with a zero or nonfinite checked norm')
        state.exponent -= .5 * (math.log10(magnitude) + exponent)
        phase = complex(mantissa) / magnitude
        if phase != 1.:
            tensor = next(iter(state)) if phase_site is None else state[phase_site]
            tensor.modify(data=tensor.data * phase**-.5)

    @timed_phase('normalization')
    def _normalize_cached_pair(self, state, boundary, *, chi, normalize_kwargs, phase_site):
        # The public metric setting accepts (norm, overlap), but boundary_norm
        # contracts only a norm. Keep None for explicit exact/reused metrics.
        chi = None if chi is None else chi_pair(chi)[0]
        custom = _merge_opts(self.normalize_kwargs, normalize_kwargs)
        if any(key != 'chi' for key in custom) or self.boundary_kwargs.get('balance_bonds', False):
            # User-specified balancing/methods keep their existing semantics.
            return self._normalize_state(state, normalize_chi=chi,
                                         normalize_kwargs={**custom, 'chi': chi})
        opts = {**self.boundary_kwargs, 'chi': chi, 'bdy': boundary,
                'contraction_opt': self.contraction_opt, 'strip_exponent': True,
                'progress': False}
        for key in ('norm', 'norm_target', 'bdy_target', 'bdy_overlap', 'balance_bonds'):
            opts.pop(key, None)
        value = boundary_norm(state, **opts)
        self._append_fit_diagnostics(value)
        value = getattr(value, 'cost', value)
        self._normalize_without_rescaling_sites(state, value, phase_site=phase_site)
        self.normalizations.append(self._normalization_record(state, value))
        return value

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
            cutoff=0.0,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )
        opts.update(self.target_gate_kwargs)
        for key, exact_value in (
            ("cutoff", 0.0), ("max_bond", None), ("bond_dim", None),
            ("path_compress", False), ("chi", None),
        ):
            if key in self.target_gate_kwargs and self.target_gate_kwargs[key] != exact_value:
                raise ValueError(
                    f"target_gate_kwargs[{key!r}] must be {exact_value!r} "
                    "to preserve the exact post-gate target."
                )
            opts[key] = exact_value
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
        if isinstance(where, str) or (
            isinstance(where, (tuple, list)) and all(isinstance(site, str) for site in where)
        ):
            # Physical-index gates go directly to Quimb's split API. Coordinate
            # routing controls are not decomposition arguments.
            opts = {
                key: value for key, value in opts.items()
                if key != "sequence" and not key.startswith("path_")
            }
            bond_dim = opts.pop("bond_dim", None)
            if opts.get("max_bond") is None and bond_dim is not None:
                opts["max_bond"] = bond_dim
        return apply_gate(
            state,
            gate_payload,
            where=where,
            which=which_local,
            inplace=inplace,
            **opts,
        )

    @staticmethod
    def _gate_bond_factor(gate_payload):
        """Use the exact diagonal-qubit rank bound; otherwise allow 4D.

        Do not inspect native block arrays or traced/trainable gate values.
        Zero entries of a trainable gate can have nonzero tangents.
        """
        if ar.infer_backend(gate_payload) not in {"numpy", "torch", "cupy"}:
            return 4
        if getattr(gate_payload, "requires_grad", False):
            return 4
        if tuple(getattr(gate_payload, "shape", ())) not in {(4, 4), (2, 2, 2, 2)}:
            return 4
        matrix = ar.do("reshape", gate_payload, (4, 4))
        off_diagonal = matrix - ar.do("diag", ar.do("diagonal", matrix))
        return 2 if int(ar.do("count_nonzero", off_diagonal)) == 0 else 4

    def _exact_rank_gate_options(self, state, gate_payload, where, opts):
        """Remove redundant split dimensions using an algebraic rank ceiling.

        A diagonal two-qubit operator is a sum of at most two product
        operators, hence rank <= 2D on a nearest-neighbor bond. This is not
        an approximation cutoff. Routed SWAPs and native symmetry arrays
        deliberately keep the unrestricted exact path.
        """
        if self._gate_bond_factor(gate_payload) != 2 or uses_symmray_arrays(state):
            return opts
        if opts.get("contract", "reduce-split") not in {"split", "reduce-split"}:
            return opts
        if not isinstance(where, (tuple, list)) or len(where) != 2:
            return opts
        if not all(isinstance(site, (tuple, list)) and len(site) == 2 for site in where):
            return opts
        a, b = where
        if sum(abs(x - y) for x, y in zip(a, b)) != 1:
            return opts
        bond_size = getattr(state, "bond_size", None)
        if not callable(bond_size):
            return opts
        return {**opts, "max_bond": 2 * bond_size(a, b)}

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
            opts=self._exact_rank_gate_options(state_work, gate_payload, where, opts),
            inplace=True,
        )

    @timed_phase("target")
    def _build_batch_target(self, state, batch_entries, *, cutoff, cutoff_mode, gate_kwargs):
        """Apply a collected gate batch onto a copy of ``state``."""
        opts = self._target_gate_options(
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )
        state_work = state.copy() if hasattr(state, "copy") else state
        for gate_payload, where, which in batch_entries:
            state_work = self._apply_gate_entry(
                state_work,
                gate_payload,
                where,
                which,
                opts=self._exact_rank_gate_options(state_work, gate_payload, where, opts),
                inplace=True,
            )
        return state_work

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
        return self._compress_to_chi(out, cutoff=cutoff, cutoff_mode=cutoff_mode)

    def _build_routed_batch_warmstart(
        self,
        state,
        batch_entries,
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
        for gate_payload, where, which in batch_entries:
            state_work = self._apply_gate_entry(
                state_work,
                gate_payload,
                where,
                which,
                opts=opts,
                inplace=True,
            )
        return self._compress_to_chi(state_work, cutoff=cutoff, cutoff_mode=cutoff_mode)

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
            warmstart = self._compress_to_chi(
                target.copy(), cutoff=cutoff, cutoff_mode=cutoff_mode
            )
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

    @timed_phase("compression")
    def _build_batch_warmstart(
        self,
        target,
        state,
        batch_entries,
        *,
        cutoff,
        cutoff_mode,
        gate_kwargs,
    ):
        if hasattr(target, "copy"):
            warmstart = self._compress_to_chi(
                target.copy(), cutoff=cutoff, cutoff_mode=cutoff_mode
            )
            max_bond = self._max_bond(warmstart)
            if max_bond is None or max_bond <= self.chi:
                return warmstart

        return self._build_routed_batch_warmstart(
            state,
            batch_entries,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            gate_kwargs=gate_kwargs,
        )

    def _collect_gate_batch(self, start_idx, k_2q_batch):
        """Collect a run of gates with up to ``k_2q_batch`` two-site entries."""
        batch_entries = []
        two_site_in_batch = 0
        idx = int(start_idx)

        while idx < len(self.gates) and two_site_in_batch < k_2q_batch:
            gate_payload, where, which = self.gates[idx]
            site_count = self._site_count(where, self.state)
            if site_count == 1:
                batch_entries.append((gate_payload, where, which))
            elif site_count == 2:
                batch_entries.append((gate_payload, where, which))
                two_site_in_batch += 1
            else:
                raise ValueError("PepsOptimizer supports one- and two-site gates only.")
            idx += 1

        return batch_entries, two_site_in_batch, idx

    @timed_phase("target")
    def _collect_auto_batch_target(self, start_idx, *, cutoff, cutoff_mode, gate_kwargs):
        """Grow an ordered exact target until bonds exceed its budget.

        Diagonal qubit batches use 2D; batches with other gates use 4D.
        Shared sites do not imply repeated bond growth: a nearest-neighbor
        ZZ layer can double every bond once while staying inside 2D.
        A single exact routed gate may exceed this budget and is processed
        alone, never truncated to meet a batching
        budget. Candidate copies keep a rejected look-ahead gate out of the
        accepted target and leave its queue position unchanged.
        """
        target = self.state
        entries = []
        n_two_site = 0
        idx = start_idx
        stop_reason = "end_of_queue"
        bond_factor = 2
        while idx < len(self.gates):
            entry = self.gates[idx]
            gate_payload, where, which = entry
            site_count = self._site_count(where, target)
            if site_count not in (1, 2):
                raise ValueError("PepsOptimizer supports one- and two-site gates only.")
            candidate_factor = (
                max(bond_factor, self._gate_bond_factor(gate_payload))
                if site_count == 2 else bond_factor
            )
            candidate = self._build_target(
                target, gate_payload, where, which,
                cutoff=cutoff, cutoff_mode=cutoff_mode, gate_kwargs=gate_kwargs,
            )
            candidate_bond = self._max_bond(candidate)
            exceeds_limit = candidate_bond is None or candidate_bond > candidate_factor * self.chi
            if exceeds_limit and n_two_site:
                stop_reason = "bond_limit"
                break
            target = candidate
            bond_factor = candidate_factor
            entries.append(entry)
            idx += 1
            if site_count == 2:
                n_two_site += 1
            if exceeds_limit:
                stop_reason = "single_gate_exceeds_limit"
                break
        return entries, n_two_site, idx, target, stop_reason, bond_factor * self.chi

    @staticmethod
    def _batch_record_payload(batch_entries):
        if len(batch_entries) == 1:
            _, where, which = batch_entries[0]
            return where, which
        return (
            tuple(where for _, where, _ in batch_entries),
            tuple(which for _, _, which in batch_entries),
        )

    def _compress_to_chi(self, state, *, cutoff, cutoff_mode):
        max_bond = self._max_bond(state)
        if max_bond is None or max_bond <= self.chi:
            return state

        compress_all = getattr(state, "compress_all", None)
        if callable(compress_all):
            return compress_all(
                max_bond=self.chi, cutoff=cutoff,
                cutoff_mode=cutoff_mode, inplace=False,
            )

        compress_all_ = getattr(state, "compress_all_", None)
        if callable(compress_all_):
            compress_all_(max_bond=self.chi, cutoff=cutoff, cutoff_mode=cutoff_mode)
        return state

    @staticmethod
    def _copy_and_mangle_target(target):
        target_opt = target.copy() if hasattr(target, "copy") else target
        mangle_inner = getattr(target_opt, "mangle_inner_", None)
        if callable(mangle_inner):
            mangle_inner()
        return target_opt

    @staticmethod
    def _clean_infidelity(value, *, roundoff_tol=1.0e-12):
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get("infidelity")
        if value is None:
            return None
        value = _backend_to_float(value)
        if not math.isfinite(value):
            raise ValueError("PEPS infidelity must be finite.")
        if value < -roundoff_tol:
            raise ValueError("PEPS infidelity is substantially negative.")
        if value > 1.0 + roundoff_tol:
            raise ValueError("PEPS infidelity is substantially above one.")
        return PepsOptimizer._clip_fidelity(value)

    def _clean_optimizer_infidelity(self, value, summary):
        """Apply the approximate diagnostic policy at either optimizer exit."""
        if value is None:
            return None
        raw = _backend_to_float(value)
        clean = self._clean_infidelity(
            raw, roundoff_tol=max(1.0e-12, self.evaluation_negative_tol),
        )
        summary.update(raw_infidelity=raw, infidelity=clean)
        if raw != clean:
            summary.update(clipped_infidelity=True,
                           negative_tolerance=self.evaluation_negative_tol)
            warnings.warn(
                f"Approximate {summary['backend']} infidelity ({raw:.3e}) outside "
                f"[0, 1]; continuing with {clean:g}. Raw estimate retained in diagnostics.",
                RuntimeWarning, stacklevel=2,
            )
        return clean

    @staticmethod
    def _clip_fidelity(value):
        return _backend_to_float(ar.do("clip", ar.do("real", value), 0.0, 1.0))

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

        try:
            real = _backend_to_float(ar.do("real", value), real=False)
            imag = _backend_to_float(ar.do("imag", value), real=False)
        except (TypeError, ValueError, RuntimeError):
            return repr(value)

        if abs(imag) <= 1.0e-15:
            return real
        return complex(real, imag)

    def _normalization_record(self, state, old_norm):
        """Build a lightweight normalization event without retaining ``state``."""
        return {
            "old_norm": self._trace_scalar(old_norm),
            "state_max_bond": self._max_bond(state),
        }

    def _append_fit_diagnostics(self, result):
        """Retain boundary FIT diagnostics without retaining contraction TNs."""
        if result is None:
            return
        if isinstance(result, (tuple, list)):
            self.fit_diagnostics.extend(result)
            return
        records = getattr(result, "fit_diagnostics", None)
        if records is not None:
            self.fit_diagnostics.extend(tuple(records))
            return
        if not isinstance(result, Mapping):
            return
        for key in ("norm_result", "norm_target_result", "overlap_result"):
            metric = result.get(key)
            if metric is not None:
                self.fit_diagnostics.extend(
                    tuple(getattr(metric, "fit_diagnostics", ()))
                )

    def _validate_adaptive_fit_caps(self, sweep_kwargs=None):
        policies = (self.boundary_kwargs, _merge_opts(self.sweep_kwargs, sweep_kwargs))
        if any(policy.get('fit_max_bond') is not None for policy in policies):
            raise ValueError(
                'Adaptive boundary convergence requires fit_max_bond=None so '
                'the effective FIT cap follows each probe chi. Use '
                'boundary_convergence start_chi/max_chi, or set '
                'boundary_convergence=False for a fixed fit_max_bond.'
            )

    @timed_phase("boundary_convergence")
    def _calibrate_sweep_boundaries(self, state, target, *, normalize_chi,
                                   evaluation_chi, sweep_kwargs=None, sweep_optimize_kwargs=None):
        """Recompute both norms and overlap on unchanged states at growing caps."""
        self._validate_adaptive_fit_caps(sweep_kwargs)
        # Use one stable, disjoint target namespace for calibration and fit.
        # This changes labels only, and permits same-topology cache reuse.
        target = target.copy()
        shared = set(state.inner_inds()) & set(target.inner_inds())
        occupied = set(state.ind_map) | set(target.ind_map)
        mapping = {}
        for index in shared:
            renamed = f'{index}__pepsy_target'
            while renamed in occupied:
                renamed += '_'
            mapping[index] = renamed
            occupied.add(renamed)
        target.reindex_(mapping)
        opts = {k: v for k, v in self.boundary_kwargs.items()
                if k not in {"balance_bonds", "chi", "norm", "norm_target"}}
        # Explicit sweep FIT controls take precedence over the shared defaults.
        overrides = _merge_opts(self.sweep_kwargs, sweep_kwargs)
        for key in tuple(opts):
            if key in overrides:
                opts[key] = overrides[key]
        for key in ("fit_mode", "cutoff", "fit_rtol", "fit_min_iter", "fit_patience"):
            if key in overrides:
                opts[key] = overrides[key]
        run_options = _merge_opts(self.sweep_optimize_kwargs, sweep_optimize_kwargs)
        if "env_n_iter" in run_options:
            opts["n_iter"] = run_options["env_n_iter"]
        if self.boundary_options or overrides.get("boundary_options"):
            raise ValueError("Adaptive boundary checks do not yet support boundary_options; "
                             "use shared boundary_kwargs or boundary_convergence=False.")
        if any(k in overrides for k in ("bdy", "bdy_overlap")):
            raise ValueError("Adaptive boundary checks require fresh boundary environments.")
        opts["contraction_opt"] = self.contraction_opt
        opts.setdefault('cutoff', 'auto')
        opts["progress"] = False
        opts["strip_exponent"] = True
        engine = overrides.get("boundary_engine", self.boundary_engine)
        engine = normalize_boundary_engine(engine, state, target)
        # Calibrate the actual fitting environments, not a separately requested
        # exact/readout metric which would be independent of boundary chi.
        opts["method"] = "mps" if engine == "quimb-mps" else "dmrg"
        caps = [chi_pair(overrides.get("chi", self.boundary_chi)), chi_pair(evaluation_chi)]
        norm_cap = chi_pair(normalize_chi)[0]
        minimum = (max(norm_cap, *(c[0] for c in caps)), max(c[1] for c in caps))
        if self.boundary_convergence["schedule"] == "d2":
            minimum = self.boundary_convergence["start_chi"]
        # Separate x/y guesses retain independent approximations. Optional
        # validated cuts may survive updates; fresh confirmation never does.
        warm_boundaries = {axis: {} for axis in ("x", "y")}
        can_reuse = (self.boundary_convergence["warm_start"] and engine == "dmrg"
                     and opts.get("fit_mode") in {"eff", "dmrg", "dmrg2", "two-site"})
        reuse_environments = (
            self.boundary_convergence["reuse_environments"] and engine == "dmrg"
            and all(_array_stamp(t.data) is not None for p in (state, target) for t in p)
        )
        retained_environments = getattr(self, '_checked_boundary_environments', {})
        measured_handles = {}

        def prepare_handles(handles, cap):
            layers = {
                'bdy': build_bra_ket(ket=state)[1],
                'bdy_target': build_bra_ket(ket=target)[1],
                'bdy_overlap': build_bra_ket(ket=target, bra=state)[1],
            }
            prepared = {}
            for name, network in layers.items():
                topology = (network.Lx, network.Ly)
                boundary = handles.get(name)
                if (boundary is None or getattr(boundary, '_reuse_topology', None) != topology):
                    # Individual changed cuts are invalidated by their source
                    # signatures, including index names and dimensions.
                    block = opts.get('fit_mode') in {'dmrg2', 'two-site'} or opts.get('fit_block_size', 1) > 1
                    rank = 1 if block or opts.get('fit_mode') in _FIT_QUIMB_MODES else cap[name == 'bdy_overlap']
                    boundary = BdyMPS(tn_double=network, chi=rank, lazy=True,
                                      single_layer=opts.get('single_layer', False))
                    boundary._reuse_topology = topology
                    boundary.mps_b.environment_cache = EnvironmentCache()
                else:
                    # Lazy guesses must refer to the current, owned network.
                    boundary._tn_norm = network.copy()
                prepared[name] = boundary
            return prepared

        def measure(cap, axis, *, fresh=False):
            axis_options = {"sequence": (f"{axis}min", f"{axis}max")} if engine == "quimb-mps" else {}
            handles = warm_boundaries[axis] if can_reuse and not fresh else {}
            if reuse_environments:
                retained = retained_environments.get(axis)
                if not fresh and retained is not None and retained[0] == cap:
                    handles = retained[1]
                reused = bool(handles)
                handles = prepare_handles(handles, cap)
            else:
                reused = bool(handles)
            # SRC builds a new guess and would discard the reused boundary.
            # Retuning/padding and all FIT updates remain in the boundary API.
            if reused:
                axis_options["fit_init_strategy"] = "direct"
            try:
                # Sweep objectives use <state|target>. Match that orientation
                # so the checked overlap boundaries transfer without rebuilding.
                first, second = (target, state) if reuse_environments else (state, target)
                norm_name, target_name = (('bdy_target', 'bdy') if reuse_environments
                                          else ('bdy', 'bdy_target'))
                result = boundary_infidelity(first, second, **{
                    **opts, **axis_options, "chi": cap, "direction": axis, "norm": None,
                    "norm_target": None, "bdy": handles.get(norm_name),
                    "bdy_target": handles.get(target_name), "bdy_overlap": handles.get("bdy_overlap"),
                })
                if reuse_environments:
                    for left, right in (("norm", "norm_target"), ("bdy", "bdy_target"),
                                        ("norm_result", "norm_target_result")):
                        result[left], result[right] = result[right], result[left]
            except ZeroDivisionError:
                # A zero approximate norm at a coarse cap is not convergence.
                warm_boundaries[axis].clear()
                measured_handles.pop(axis, None)
                return {k: (0.0, 0.0) for k in ("norm", "norm_target", "overlap")}
            self._append_fit_diagnostics(result)
            if can_reuse:
                warm_boundaries[axis] = {k: result[k] for k in ("bdy", "bdy_target", "bdy_overlap")
                                         if result.get(k) is not None}
            if reuse_environments:
                measured_handles[axis] = (cap, handles)
            sample = {k: _as_scaled_scalar(result[k]) for k in ("norm", "norm_target", "overlap")}
            if reuse_environments:
                # Preserve the public diagnostic convention <target|state>.
                value, exponent = sample['overlap']
                sample['overlap'] = (complex(value).conjugate(), exponent)
            sample["warm_started"] = reused
            sample["actual_boundary_bonds"] = {
                k: max((int(m.max_bond()) for m in result[k].mps_b.values()), default=1)
                for k in ("bdy", "bdy_target", "bdy_overlap") if result.get(k) is not None
            }
            return sample

        result = select_boundary_chi(
            measure, minimum=minimum, retained=self._boundary_chi_floor,
            options=self.boundary_convergence, roundoff=_resolve_gate_cutoff(state, "auto"),
            confirm=(lambda cap, axis: measure(cap, axis, fresh=True)) if can_reuse else None,
        )
        if reuse_environments:
            self._checked_boundary_environments = {
                axis: entry for axis, entry in measured_handles.items()
                if entry[0] == result['chi']
            }
        if (reuse_environments
                and all(axis in self._checked_boundary_environments for axis in ('x', 'y'))):
            # Each axis is checked independently. Merge only its own cuts for
            # fitting; no compressed values are borrowed across directions.
            fit_handles = {}
            for name in ('bdy', 'bdy_overlap'):
                first = measured_handles['y'][1][name]
                other = measured_handles['x'][1][name]
                for key, boundary in other.mps_b.items():
                    if key.startswith('X'):
                        first.mps_b[key] = boundary
                        entry = other.mps_b.environment_cache.entries.get(key)
                        if entry is not None:
                            first.mps_b.environment_cache.entries[key] = entry
                fit_handles[name] = first
            result['fit_boundaries'] = fit_handles
            result['fit_target'] = target
        # Keep cap history outside reset_traces: each new state must be tested,
        # while its fit never silently returns to a previously inadequate cap.
        self._boundary_chi_floor = result["chi"]
        record = {"converged": result["converged"], "chi": result["chi"],
                  "selection_status": "converged" if result["converged"] else "limit_unconverged",
                  "at_ceiling": result["chi"] == self.boundary_convergence["max_chi"],
                  "stop_reason": result["stop_reason"], "bond_dim": self.chi,
                  "warm_start_enabled": can_reuse,
                  "environment_reuse_enabled": reuse_environments,
                  "minimum_chi": minimum, "options": dict(self.boundary_convergence),
                  "attempts": []}
        def diagnostic_number(value):
            value = float(value)
            return value if math.isfinite(value) else repr(value)

        for attempt in result["attempts"]:
            saved = {k: v for k, v in attempt.items() if k != "samples"}
            saved["samples"] = {
                axis: {k: {"real": diagnostic_number(complex(v[0]).real),
                            "imag": diagnostic_number(complex(v[0]).imag),
                            "exponent": diagnostic_number(v[1])} for k, v in sample.items()
                       if k in {"norm", "norm_target", "overlap"}}
                for axis, sample in attempt["samples"].items()
            }
            saved["warm_started"] = {axis: s.get("warm_started", False)
                                     for axis, s in attempt["samples"].items()}
            saved["actual_boundary_bonds"] = {axis: s.get("actual_boundary_bonds", {})
                                              for axis, s in attempt["samples"].items()}
            record["attempts"].append(saved)
        self.boundary_convergence_records.append(record)
        result["record"] = record
        if not result["converged"]:
            warnings.warn(
                f"PEPS norm/overlap chi convergence was not established at chi={result['chi']}; "
                "continuing at the configured limit with boundary_converged=False. "
                "Increase boundary_convergence max_chi or inspect the raw contractions.",
                RuntimeWarning, stacklevel=2,
            )
        return result

    @timed_phase("fidelity")
    def estimate_infidelity(
        self, state, target, *, evaluation_chi=None, evaluation_max_retries=None, **kwargs,
    ):
        """Estimate normalized boundary infidelity, recomputing both norms.

        Explicit ``norm`` and ``norm_target`` values remain supported. Invalid
        finite-cap estimates retry at equal, doubled norm/overlap caps, at most
        ``evaluation_max_retries`` times (default two). Each retry warns and
        is recorded by :meth:`get_evaluation_records`; zero disables retries.
        Approximate values outside [0, 1] within ``evaluation_negative_tol``
        warn and are clipped without retries; raw estimates remain recorded.
        Larger persistent invalid estimates raise.
        """
        retries = self.evaluation_max_retries if evaluation_max_retries is None else (
            self._validate_evaluation_retries(evaluation_max_retries)
        )
        opts = _merge_opts(
            {
                key: value
                for key, value in self.boundary_kwargs.items()
                if key != "balance_bonds"
            },
            self.infidelity_kwargs,
            kwargs,
        )
        opts["chi"] = self._boundary_chi_for_infidelity(evaluation_chi, options=kwargs)
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        opts.setdefault("norm", None)
        opts.setdefault("norm_target", None)
        _prefer_boundary_engine_mps(opts, self.boundary_engine, state, target)
        record = {"requested_chi": opts["chi"], "attempts": []}
        self.evaluation_records.append(record)
        roundoff_tol = _resolve_gate_cutoff(state, "auto")
        negative_tol = roundoff_tol if str(opts.get("method", "dmrg")).lower() == "exact" else max(
            roundoff_tol, self.evaluation_negative_tol,
        )
        for attempt in range(retries + 1):
            self._last_evaluation_chi = opts["chi"]
            result = boundary_infidelity(state, target, **opts)
            self._last_target_norm = (
                result.get("norm_target", 1.0) if isinstance(result, Mapping) else 1.0
            )
            self._append_fit_diagnostics(result)
            raw = result.get("infidelity") if isinstance(result, Mapping) else result
            value = None if raw is None else _backend_to_float(raw)
            self._last_evaluation_raw_infidelity = value
            record["attempts"].append({"chi": opts["chi"], "infidelity": value})
            if (value is not None and math.isfinite(value)
                    and -negative_tol <= value <= 1.0 + negative_tol
                    and (value < -roundoff_tol or value > 1.0 + roundoff_tol)):
                bounded = self._clip_fidelity(value)
                record.update(
                    effective_chi=opts["chi"], raw_infidelity=value,
                    infidelity=bounded, clipped_negative=value < 0,
                    clipped_infidelity=True, negative_tolerance=negative_tol,
                )
                description = "Small negative" if value < 0 else "Above-one"
                bound_label = "zero" if bounded == 0 else "one"
                warnings.warn(
                    f"{description} approximate PEPS infidelity ({value:.3e}) "
                    f"at evaluation chi={opts['chi']!r}; continuing with {bound_label} "
                    f"within tolerance {negative_tol:.3e}. Raw estimate retained in diagnostics.",
                    RuntimeWarning, stacklevel=2,
                )
                return bounded
            if (value is None or not math.isfinite(value)
                    or -roundoff_tol <= value <= 1.0 + roundoff_tol):
                record["effective_chi"] = opts["chi"]
                return self._clean_infidelity(value, roundoff_tol=roundoff_tol)
            if (
                attempt == retries or opts["chi"] is None
                or str(opts.get("method", "dmrg")).lower() == "exact"
                or opts["norm"] is not None or opts["norm_target"] is not None
            ):
                violation = "negative" if value < 0 else "above one"
                raise ValueError(
                    f"PEPS infidelity is substantially {violation} ({value:.3e}) "
                    f"at evaluation chi={opts['chi']!r}; increase metric accuracy. "
                    "When assuming unit target norm, increase normalize_chi "
                    "as well, or set infidelity_kwargs={'norm_target': None} "
                    "to measure the target norm explicitly."
                )
            cap = opts["chi"]
            cap = 2 * (max(cap) if isinstance(cap, (tuple, list)) else cap)
            warnings.warn(
                f"Invalid PEPS boundary infidelity ({value:.3e}); retrying "
                f"both norms and overlap with chi={cap}.",
                RuntimeWarning, stacklevel=2,
            )
            opts["chi"] = cap

    def _measure_target_norm(self, target, *, evaluation_chi, metric_kwargs):
        """Resolve an unknown objective norm without an overlap contraction."""
        opts = _merge_opts(self.boundary_kwargs, self.infidelity_kwargs, metric_kwargs)
        target_boundary = opts.pop("bdy_target", None)
        for key in ("balance_bonds", "norm", "norm_target", "bdy", "bdy_overlap",
                    "evaluation_max_retries"):
            opts.pop(key, None)
        opts["bdy"] = target_boundary
        # Norm-only contraction takes a scalar, unlike fidelity's cap pair.
        opts["chi"] = evaluation_chi[0] if isinstance(evaluation_chi, (tuple, list)) else evaluation_chi
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        if opts.get("fit_timing", False):
            opts["return_info"] = True
        _prefer_boundary_engine_mps(opts, self.boundary_engine, target)
        result = boundary_norm(target, **opts)
        self._append_fit_diagnostics(result)
        return getattr(result, "cost", result)

    def _sweep_boundary_kwargs(self, *, progress, sweep_progress=None):
        opts = _merge_opts(self.boundary_kwargs)
        strip_exponent = opts.pop("strip_exponent", None)
        opts.setdefault("chi", self.boundary_chi)
        opts.setdefault("contraction_opt", self.contraction_opt)
        opts.setdefault("progress", False)
        # ``boundary_kwargs`` also configures the standalone metric helpers.
        # Keep metric-only names such as ``method`` and ``mode_`` in those
        # calls, but do not pass them as unsupported SweepOptimizer init args.
        boundary_init_kwargs = {
            key: value
            for key, value in opts.items()
            if key in _SWEEP_BOUNDARY_INIT_KEYS
        }
        inner_progress = bool(progress) if sweep_progress is None else bool(sweep_progress)
        # SweepOptimizer.run uses env_n_iter for boundary moves during sweeps.
        # Keep boundary-contraction progress silent, but let its one
        # slice-level bar show the directional sweep when the outer PEPS
        # progress bar is enabled. ``sweep_progress`` can explicitly override
        # that coupling. The gradient solver remains silent unless the caller
        # explicitly puts ``progress=True`` in optimizer_options.
        opt_kwargs = {
            "env_n_iter": opts.get("n_iter", _DEFAULT_BOUNDARY_KWARGS["n_iter"]),
            "track_boundary_fidelity": opts.get("track_boundary_fidelity", False),
            "progress": inner_progress,
            "progress_position": 1 if progress and inner_progress else 0,
            "progress_leave": False,
        }
        return boundary_init_kwargs, opt_kwargs, strip_exponent

    def _apply_sweep_optimizer_options(self, opt_kwargs):
        opt_kwargs = dict(opt_kwargs or {})
        if self.optimizer is not None:
            opt_kwargs.setdefault("optimizer", self.optimizer)
        if self.optimizer_options:
            opt_kwargs["optimizer_options"] = _merge_opts(
                self.optimizer_options, opt_kwargs.get("optimizer_options"),
            )
        opt_kwargs.setdefault("optimizer", _DEFAULT_SWEEP_SOLVER)
        if _optimizer_key(opt_kwargs.get("optimizer")) == "nlopt":
            opt_kwargs["optimizer_options"] = _merge_opts(
                _DEFAULT_SWEEP_NLOPT_OPTIONS,
                opt_kwargs.get("optimizer_options"),
            )
        return opt_kwargs

    def _global_contraction_defaults(self, *, cutoff, progress):
        # ``progress`` is accepted for signature symmetry but intentionally not
        # propagated to inner boundary contractions so only the outer PEPS bar
        # is shown (matching the MpsOptimizer visualization).
        del progress
        bopts = _merge_opts(self.boundary_kwargs)
        return {
            "contraction_opt": self.contraction_opt,
            "chi": self.boundary_chi,
            "mode": "mps",
            "mode_": "mps",
            "max_separation": bopts.get("max_separation", 1),
            "cutoff": cutoff,
            "progbar": False,
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

    @staticmethod
    def _torch_linalg_mode(*states):
        """Infer the Torch linalg mode from the first available tensor block."""
        found_dtype = False
        for state in states:
            if state is None:
                continue
            tensors = getattr(state, "tensor_map", None)
            if isinstance(tensors, Mapping):
                tensors = tensors.values()
            elif tensors is None:
                try:
                    tensors = iter(state)
                except TypeError:
                    tensors = ()
            for tensor in tensors:
                data = getattr(tensor, "data", None)
                blocks = getattr(data, "blocks", None)
                samples = blocks.values() if blocks is not None else (data,)
                for sample in samples:
                    dtype = getattr(sample, "dtype", None)
                    if dtype is None:
                        continue
                    found_dtype = True
                    if "complex" in str(dtype).lower():
                        return "complex"
        return "real" if found_dtype else "complex"

    def _maybe_configure_torch_linalg(self, state, target, opt_kwargs):
        """Configure the canonical Torch linalg stack for global cleanup."""
        if not self.register_torch_svd:
            return
        backend = opt_kwargs.get("autodiff_backend", "torch")
        if not isinstance(backend, str) or backend.strip().lower() != "torch":
            return

        symmray_blocks = uses_symmray_arrays(state, target)
        config = self.torch_linalg_config
        if config is None:
            # Global PEPS cleanup differentiates through SVD/QR. The default
            # therefore favors finite autodiff over the native-only policy,
            # while keeping the dtype-dependent real/complex choice automatic.
            config = TorchLinalgConfig(
                mode=self._torch_linalg_mode(state, target),
                stabilized=True,
                quimb_split_drivers=symmray_blocks,
            )
        elif symmray_blocks and not config.quimb_split_drivers:
            # Raw Symmray blocks bypass Autoray. Preserve the user's SVD/QR
            # choices but enable the matching Quimb registrations so the
            # configured policy actually covers the PEPS split path.
            config = replace(config, quimb_split_drivers=True)
        try:
            config.register()
        except ImportError as exc:
            warnings.warn(
                f"Could not configure Torch linalg: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _configure_global_linalg(self, state, target, opt_kwargs):
        """Install backend derivative rules before constructing/tracing the loss."""
        backend = opt_kwargs.get("autodiff_backend", "torch")
        if isinstance(backend, str) and backend.strip().lower() == "jax":
            # The existing JAX rule restores truncated SVD cotangents, then
            # delegates to native JAX differentiation. QR remains native.
            register_jax_linalg(stabilized=True)
        else:
            self._maybe_configure_torch_linalg(state, target, opt_kwargs)

    def _record_fidelity_progress(self, infidelity):
        infidelity = self._clean_infidelity(infidelity)
        if infidelity is None:
            return None, None

        fidelity = self._clip_fidelity(1.0 - infidelity)
        self.local_infidelities.append(infidelity)
        self._fidelity_count += 1

        trace_fidelity = max(float(fidelity), _TRACE_FIDELITY_FLOOR)
        self._fidelity_log_sum += math.log(trace_fidelity)

        cumulative_fidelity = math.exp(self._fidelity_log_sum)
        geometric_fidelity = math.exp(self._fidelity_log_sum / self._fidelity_count)

        self.losses.append(float(geometric_fidelity))
        self.infidelities.append(float(1.0 - cumulative_fidelity))
        return float(fidelity), float(geometric_fidelity)

    @timed_phase("sweep")
    def _optimize_with_sweep(
        self,
        state,
        target,
        *,
        progress,
        target_norm=1.0,
        sweep_progress=None,
        normalize_chi=_UNSET_METRIC_CHI,
        sweep_kwargs=None,
        sweep_optimize_kwargs=None,
    ):
        sweep_kwargs = dict(sweep_kwargs or {})
        checked_boundaries = sweep_kwargs.pop('_checked_boundaries', None)
        checked_target = sweep_kwargs.pop('_checked_target', None)
        # build_bra_ket already separates colliding ket/bra bonds. Keep the
        # calibrated labels when transferring environments into this fitter.
        target_opt = (checked_target if checked_boundaries is not None
                      else self._copy_and_mangle_target(target))
        boundary_init_kwargs, boundary_opt_kwargs, strip_exponent = self._sweep_boundary_kwargs(
            progress=progress,
            sweep_progress=sweep_progress,
        )
        init_kwargs = {
            "chi": self.boundary_chi,
            "target_norm": target_norm,
            "contraction_opt": self.contraction_opt,
            # run() already normalized the compressed warm start.
            "renormalize_state": False,
            "boundary_engine": self.boundary_engine,
            **boundary_init_kwargs,
        }
        if self.boundary_options:
            init_kwargs["boundary_options"] = dict(self.boundary_options)
        init_kwargs.update(self.sweep_kwargs)
        init_kwargs.update(dict(sweep_kwargs or {}))
        if checked_boundaries is not None:
            init_kwargs.update(checked_boundaries)
        if init_kwargs.get("local_contraction_opt") is None:
            # One reusable search cache across gate batches and run() calls.
            # This plans local objectives only, never caches contraction values.
            if self._sweep_local_contraction_opt is None:
                self._sweep_local_contraction_opt = build_optimizer(progbar=False)
            init_kwargs["local_contraction_opt"] = self._sweep_local_contraction_opt
        init_kwargs.setdefault("evaluation_negative_tol", self.evaluation_negative_tol)
        # ``full_simplify`` is not backend-safe for Symmray block trees. Make
        # the supported default explicit here so SweepOptimizer does not need
        # to correct it (and warn) during construction. An explicit
        # ``simplify=True`` in sweep_kwargs remains visible and is corrected by
        # SweepOptimizer with its actionable compatibility warning.
        if uses_symmray_arrays(state, target_opt) and "simplify" not in init_kwargs:
            init_kwargs["simplify"] = False
        boundary_engine = normalize_boundary_engine(
            init_kwargs.get("boundary_engine", "auto"),
            state,
            target_opt,
        )
        init_kwargs["boundary_engine"] = boundary_engine
        normalize_payload = _merge_opts(
            {"chi": self._boundary_chi_for_norm()
             if normalize_chi is _UNSET_METRIC_CHI else normalize_chi},
            init_kwargs.get("normalize_kwargs"),
        )
        if boundary_engine == "quimb-mps":
            normalize_payload = _merge_opts(
                {"method": "mps", "mode_": "mps", "balance_bonds": False},
                normalize_payload,
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
        # Internal start/end diagnostics must use the same target norm as
        # the local objective. Sweep's standalone diagnostic default measures
        # it anew; its existing diagnostic options let this driver reuse it.
        opt_kwargs["debug_loss_kwargs"] = _merge_opts(
            {"norm_target": init_kwargs["target_norm"]},
            opt_kwargs.get("debug_loss_kwargs"),
        )
        if sweep_progress is not None:
            # A top-level explicit setting wins over legacy nested
            # ``sweep_optimize_kwargs={"progress": ...}`` values.
            opt_kwargs["progress"] = bool(sweep_progress)
            opt_kwargs["progress_position"] = 1 if progress and sweep_progress else 0
            opt_kwargs["progress_leave"] = False
        opt_kwargs.setdefault("progress", bool(progress))
        if opt_kwargs:
            sweeper.set_optimize_kwargs(**opt_kwargs)

        result = sweeper.run()
        self._append_fit_diagnostics(getattr(sweeper, "fit_diagnostics", None))
        best_state = result.get("best_state") if isinstance(result, Mapping) else None
        state_out = best_state if best_state is not None else sweeper.state
        final_infidelity = None
        if isinstance(result, Mapping):
            final_infidelity = result.get("best_loss")
            if final_infidelity is None:
                final_infidelity = result.get("loss_after")
        summary = self._summarize_sweep_result(result)
        if checked_boundaries is not None:
            summary['environment_reuse'] = {
                name: boundary.mps_b.environment_cache.report()
                for name, boundary in checked_boundaries.items()
            }
        return state_out, self._clean_optimizer_infidelity(final_infidelity, summary), summary

    @timed_phase("global")
    def _optimize_with_global(
        self,
        state,
        target,
        *,
        progress,
        cutoff,
        target_norm=1.0,
        normalize_chi=_UNSET_METRIC_CHI,
        global_kwargs=None,
        global_optimize_kwargs=None,
    ):
        target_opt = self._copy_and_mangle_target(target)
        init_kwargs = {"chi": self.boundary_chi}
        init_kwargs.update(self.global_kwargs)
        init_kwargs.update(dict(global_kwargs or {}))
        opt_kwargs = _merge_opts(self.global_optimize_kwargs, global_optimize_kwargs)
        opt_kwargs = self._apply_global_optimizer_options(opt_kwargs)

        autodiff_backend = str(opt_kwargs.get("autodiff_backend", "torch")).lower()
        # Keep global boundary truncation separate from gate/warm-start cutoff.
        # Explicit per-metric constructor options below retain precedence.
        global_cutoff = 1.0e-10 if autodiff_backend == "torch" else cutoff
        norm_defaults = self._global_contraction_defaults(
            cutoff=global_cutoff,
            progress=False,
        )
        normalize_defaults = dict(norm_defaults)
        normalize_defaults["chi"] = (
            self._boundary_chi_for_norm()
            if normalize_chi is _UNSET_METRIC_CHI else normalize_chi
        )
        loss_defaults = self._global_loss_defaults(
            cutoff=global_cutoff,
            progress=False,
        )
        if autodiff_backend == "jax":
            # JIT needs fixed SVD ranks. Exponent stripping currently converts
            # JAX scalars to Python, losing gradients even without JIT.
            loss_defaults.update(cutoff=0.0, strip_exponent=False)
            opt_kwargs.setdefault("jit_fn", True)
        loss_defaults["target_norm"] = target_norm
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

        # The driver owns final normalization, including its opt-out.
        opt_kwargs["normalize"] = False
        opt_kwargs.setdefault("progbar", False)
        optimizer_name = opt_kwargs.get("optimizer", "adam")
        use_nlopt = _is_nlopt_optimizer(optimizer_name)
        self._configure_global_linalg(state, target_opt, opt_kwargs)
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
                fallback_kwargs["normalize"] = False
                fallback_kwargs.setdefault("progbar", False)
                self._configure_global_linalg(
                    state,
                    target_opt,
                    fallback_kwargs,
                )
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
        final_infidelity = getattr(optimizer, "final_loss", losses[-1] if losses else None)
        summary = self._summarize_global_result(
            opt_kwargs=opt_kwargs,
            losses=losses,
            fallback_used=fallback_used,
            fallback_error=fallback_error,
        )
        summary.update(getattr(optimizer, "optimization_info", {}))
        return state_out, self._clean_optimizer_infidelity(final_infidelity, summary), summary

    def _summarize_sweep_result(self, result):
        """Return scalar sweep diagnostics without retaining TN objects."""
        summary = {"backend": "sweep"}
        if not isinstance(result, Mapping):
            return summary

        for key in ("success", "termination_reason", "converged", "early_exit",
                    "initial_loss_reused", "final_loss_measured",
                    "applied_local_updates", "invalid_local_updates"):
            if key in result:
                summary[key] = result[key]

        for key in (
            "loss_before",
            "loss_after",
            "raw_loss_after",
            "best_loss",
            "bdy_norm",
            "bdy_overlap_norm",
        ):
            if key in result and result[key] is not None:
                summary[key] = self._trace_scalar(result[key])

        runs = result.get("runs")
        if runs is not None:
            summary["n_runs"] = len(runs)
            summary["invalid_loss_records"] = [
                {key: run[key] for key in (
                    "axis", "index", "sweep", "raw_loss_initial", "raw_loss_final",
                    "candidate_loss", "rejection_reason", "update_applied",
                ) if key in run}
                for run in runs if run.get("invalid_loss")
            ]
            summary["clipped_loss_records"] = [
                {key: run[key] for key in (
                    "axis", "index", "raw_loss_initial", "raw_loss_final",
                    "loss_initial", "loss_final",
                ) if key in run}
                for run in runs
                if any(
                    value is not None and (value < 0.0 or value > 1.0)
                    for value in (run.get("raw_loss_initial"), run.get("raw_loss_final"))
                )
            ]
            if getattr(self, "_phase_timer", None) is not None:
                # Existing slice timers are host wall times, not CUDA events.
                # Their sum excludes sweep setup and whole-state diagnostics.
                summary["timing"] = {
                    "synchronized": False,
                    "boundary_seconds": sum(float(r.get("time_boundary", 0.0)) for r in runs),
                    "optimize_seconds": sum(float(r.get("time_optimize", 0.0)) for r in runs),
                    "slices": [
                        {k: r[k] for k in ("sweep", "axis", "index", "time_boundary", "time_optimize")
                         if k in r}
                        for r in runs
                    ],
                }
        for source_key, target_key in (
            ("loss", "loss_count"),
            ("step_loss_trace", "step_loss_count"),
            ("step_trace", "step_count"),
            ("inner_loss_traces", "inner_loss_trace_count"),
            ("inner_best_loss_traces", "inner_best_loss_trace_count"),
            ("norm_trace", "norm_count"),
            ("fidels", "fidelity_count"),
        ):
            values = result.get(source_key)
            if values is not None:
                summary[target_key] = len(values)

        fit_diagnostics = result.get("fit_diagnostics")
        if fit_diagnostics is not None:
            summary["fit_diagnostic_count"] = len(fit_diagnostics)
            summary["fit_timing_count"] = sum(
                bool(getattr(diagnostic, "sweep_timings", ()))
                for diagnostic in fit_diagnostics
            )

        best_traces = result.get("inner_best_loss_traces")
        if best_traces:
            best_values = [
                float(value)
                for trace in best_traces
                for value in (trace or ())
                if value is not None and math.isfinite(float(value)) and float(value) >= 0.0
            ]
            if best_values:
                summary["min_inner_infidelity"] = min(best_values)
            finite_traces = [
                [float(value) for value in trace]
                for trace in best_traces
                if trace and all(
                    value is not None and math.isfinite(float(value))
                    for value in trace
                )
            ]
            summary["inner_best_monotonic"] = all(
                all(right <= left + 1.0e-14 for left, right in zip(trace, trace[1:]))
                for trace in finite_traces
            )

        runs = result.get("runs") or ()
        summary["invalid_inner_loss_count"] = sum(
            bool(run.get("invalid_loss")) for run in runs
        )
        if result.get("best_loss") is not None and result.get("loss_after") is not None:
            summary["best_after_abs_error"] = abs(
                float(result["best_loss"]) - float(result["loss_after"])
            )

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
        target_norm,
        progress,
        sweep_progress,
        cutoff,
        normalize_chi,
        sweep_kwargs,
        sweep_optimize_kwargs,
        global_kwargs,
        global_optimize_kwargs,
    ):
        self._require_torch_symmray_backend(
            state,
            target,
            role="state and target",
        )
        if mode == "sweep":
            return self._optimize_with_sweep(
                state,
                target,
                progress=progress,
                target_norm=target_norm,
                sweep_progress=sweep_progress,
                normalize_chi=normalize_chi,
                sweep_kwargs=sweep_kwargs,
                sweep_optimize_kwargs=sweep_optimize_kwargs,
            )
        if mode == "global":
            return self._optimize_with_global(
                state,
                target,
                progress=progress,
                target_norm=target_norm,
                normalize_chi=normalize_chi,
                global_kwargs=global_kwargs,
                global_optimize_kwargs=global_optimize_kwargs,
                cutoff=cutoff,
            )
        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _real_float(value):
        """Convert a backend scalar/tensor-like value to Python float."""
        return _backend_to_float(value)

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
        two_site_count=None,
        step=None,
        total_steps=None,
        status=None,
        input_infidelity=None,
    ):
        """Return compact gate and local-optimization progress diagnostics."""
        postfix = {
            "2q": self._fidelity_count if two_site_count is None else int(two_site_count),
            "bnd": self._max_bond(self.state),
        }
        if step is not None:
            total = "?" if total_steps is None else int(total_steps)
            postfix["step"] = f"{int(step)}/{total}"
        if status is not None:
            postfix["status"] = str(status)
        if self._fidelity_count:
            postfix["infidelity"] = self._format_progress_infidelity(
                self.infidelities[-1]
            )

        if self.step_records:
            record = self.step_records[-1]
            if input_infidelity is None:
                input_infidelity = record.get("pre_infidelity")
            optimizer_result = record.get("optimizer_result")
        else:
            optimizer_result = None
        if input_infidelity is not None:
            postfix["input"] = self._format_progress_infidelity(input_infidelity)

        if isinstance(optimizer_result, Mapping):
            sweep_in = optimizer_result.get("loss_before")
            sweep_out = optimizer_result.get("loss_after")
            if sweep_in is not None and sweep_out is not None:
                postfix["sweep"] = (
                    f"{self._format_progress_infidelity(sweep_in)}"
                    f"->{self._format_progress_infidelity(sweep_out)}"
                )
            sweep_best = optimizer_result.get("best_loss")
            if sweep_best is not None:
                postfix["slice_best"] = self._format_progress_infidelity(sweep_best)
            n_runs = optimizer_result.get("n_runs")
            if n_runs is not None:
                postfix["slices"] = int(n_runs)

        if self.step_records:
            record = self.step_records[-1]
            local_infidelity = record.get("final_infidelity")
            if local_infidelity is not None:
                postfix["local_infidelity"] = self._format_progress_infidelity(
                    local_infidelity
                )
            pre_infidelity = record.get("pre_infidelity")
            if pre_infidelity is not None and local_infidelity is not None:
                # Positive gain means the variational cleanup reduced the
                # local infidelity. A rejected or unchanged candidate shows 0.
                postfix["opt_gain"] = self._format_progress_infidelity(
                    float(pre_infidelity) - float(local_infidelity)
                )
        return postfix

    @timed_phase('target')
    def _full_update_pair(self, gate_payload, where):
        from ._full_update import ReducedPair
        pair = ReducedPair(self.state, gate_payload, where, self.chi)
        return pair, *pair.initial_states()

    @timed_phase('full_update_environment')
    def _full_update_environment(self, pair, guess, boundary, chi):
        reuse = self.boundary_convergence is None or self.boundary_convergence['reuse_environments']
        cache = getattr(self, '_full_update_strip_cache', None)
        if reuse and cache is None:
            cache = self._full_update_strip_cache = StripEnvironmentCache()
        return pair.environment(guess, boundary=boundary, chi=chi,
                                contraction_opt=self.contraction_opt,
                                boundary_kwargs=self.boundary_kwargs,
                                strip_cache=cache if reuse else None)

    @timed_phase('full_update_als')
    def _full_update_optimize(self, pair, norm):
        return pair.optimize(norm, self.full_update_kwargs)

    @timed_phase('full_update_refinement')
    def _full_update_refine(self, target, key, *, chi, boundary):
        from ._strip_update import refine_strip
        result, report, boundary, overlap = refine_strip(
            self.state, target, key=key, chi=chi, contraction_opt=self.contraction_opt,
            boundary_kwargs=self.boundary_kwargs, sweeps=self.full_update_kwargs['refine_sweeps'],
            rtol=self.full_update_kwargs['refine_rtol'], rcond=self.full_update_kwargs['rcond'],
            boundary=boundary, overlap_boundary=getattr(self, '_refinement_overlap_boundary', None),
            reuse=self.boundary_convergence is None or self.boundary_convergence['reuse_environments'],
        )
        self._full_update_boundary = boundary
        self._refinement_overlap_boundary = overlap
        return result, report, boundary

    def _record_local_gate_fidelity(self, fidelity):
        """Retain the pair estimate and optionally accumulate a stable product."""
        fidelity = (min(1., max(0., float(fidelity)))
                    if fidelity is not None and math.isfinite(fidelity) else None)
        accumulated = None
        if self.full_update_kwargs['accumulate_local_infidelity']:
            self._local_gate_fidelity_count += 1
            if fidelity is None:
                # A missing estimate must not silently produce a partial product.
                self._local_gate_log_fidelity = None
            elif self._local_gate_log_fidelity is not None:
                self._local_gate_log_fidelity += math.log(fidelity) if fidelity > 0. else -math.inf
            if self._local_gate_log_fidelity is not None:
                accumulated = -math.expm1(self._local_gate_log_fidelity)
        return {
            'local_fidelity': fidelity,
            'local_infidelity': 1. - fidelity if fidelity is not None else None,
            'accumulated_local_infidelity': accumulated,
            'accumulated_local_gate_count': self._local_gate_fidelity_count,
            'local_fidelity_convention': 'two-site positive-environment estimate before strip refinement',
        }

    def _run_full_update(self, *, normalize_target, normalize_final, normalize_chi, evaluation_chi,
                         normalize_kwargs, measure_infidelity, measure_final_infidelity,
                         accept_if_improved, improvement_tol, metric_kwargs,
                         cutoff, cutoff_mode, gate_kwargs, step_callback, progress):
        """Sequential nearest-neighbor updates sharing the checked boundary cache."""
        from ._strip_update import strip_key
        refine_target, refine_bonds, refine_start = None, set(), None
        refine_enabled = self.full_update_kwargs['refine_sweeps'] > 0
        entries = enumerate(self.gates, 1)
        if progress:
            from tqdm.auto import tqdm
            entries = tqdm(entries, total=len(self.gates), desc='PEPS full-update')
        for step, (gate_payload, where, which) in entries:
            if self._site_count(where, self.state) == 1:
                opts = self._target_gate_options(cutoff=cutoff, cutoff_mode=cutoff_mode,
                                                 gate_kwargs=gate_kwargs)
                self.state = self._apply_gate_entry(self.state, gate_payload, where, which,
                                                    opts=opts, inplace=False)
                continue
            before = self._phase_timer.snapshot() if self._phase_timer else None
            evaluation_start = len(self.evaluation_records)
            pair, guess, target = self._full_update_pair(gate_payload, where)
            if refine_enabled:
                if refine_target is None:
                    refine_target, refine_start = self.state.copy(), step
                refine_target = self._build_target(
                    refine_target, gate_payload, where, which, cutoff=0.,
                    cutoff_mode=cutoff_mode, gate_kwargs=gate_kwargs,
                )
                refine_bonds.add(tuple(sorted(tuple(site) for site in where)))
            if normalize_target:
                old_norm = self._normalize_state(
                    target, normalize_chi=normalize_chi, normalize_kwargs=normalize_kwargs,
                )
                old_norm = getattr(old_norm, 'cost', old_norm)
                pair.normalize_target(old_norm)
                self._normalize_without_rescaling_sites(guess, old_norm, phase_site=pair.where[0])
            calibration = None
            boundary = getattr(self, '_full_update_boundary', None)
            norm_environment_chi = chi_pair(self.boundary_chi)[0]
            if self.boundary_convergence is not None:
                calibration = self._calibrate_sweep_boundaries(
                    guess, target, normalize_chi=normalize_chi, evaluation_chi=evaluation_chi,
                )
                evaluation_chi = calibration['chi']
                normalize_chi = evaluation_chi[0]
                norm_environment_chi = normalize_chi
                boundary = calibration.get('fit_boundaries', {}).get('bdy')
            norm, boundary = self._full_update_environment(pair, guess, boundary, norm_environment_chi)
            self._full_update_boundary = boundary
            candidate, summary = self._full_update_optimize(pair, norm)
            cache = getattr(boundary.mps_b, 'environment_cache', None)
            summary['environment_reuse'] = cache.report() if cache is not None else None
            strip_cache = getattr(self, '_full_update_strip_cache', None)
            summary['strip_environment_reuse'] = strip_cache.report() if strip_cache is not None else None
            pre = post = None
            raw_pre = raw_post = None
            step_evaluation_chi = evaluation_chi
            if measure_infidelity:
                if calibration is not None:
                    sample = calibration['sample']
                    raw_pre = 1. - _scaled_overlap_fidelity(sample['overlap'], sample['norm'], sample['norm_target'])
                    pre = self._clip_fidelity(raw_pre) if math.isfinite(raw_pre) else None
                else:
                    pre = self.estimate_infidelity(guess, target, evaluation_chi=evaluation_chi, **metric_kwargs)
                    raw_pre = self._last_evaluation_raw_infidelity
                    step_evaluation_chi = self._last_evaluation_chi
            if normalize_final:
                self._normalize_cached_pair(candidate, boundary, chi=normalize_chi,
                                            normalize_kwargs=normalize_kwargs, phase_site=pair.where[0])
            if measure_infidelity and measure_final_infidelity:
                opts = {**metric_kwargs, 'chi': step_evaluation_chi, 'norm_target': None,
                        'evaluation_max_retries': 0}
                post = self.estimate_infidelity(candidate, target, **opts)
                raw_post = self._last_evaluation_raw_infidelity
            # An unconverged/unphysical estimate cannot reject a variational
            # result. ALS itself retains its best positive-environment cost.
            roundoff = _resolve_gate_cutoff(guess, 'auto')
            reliable = (raw_pre is not None and -roundoff <= raw_pre <= 1. + roundoff
                        and (calibration is None or calibration['converged']))
            post_reliable = (raw_post is not None and -roundoff <= raw_post <= 1. + roundoff)
            accepted = not (accept_if_improved and reliable and post_reliable
                            and post >= pre - improvement_tol)
            self.state = candidate if accepted else guess
            if not accepted and normalize_final:
                self._normalize_cached_pair(self.state, boundary, chi=normalize_chi,
                                            normalize_kwargs=normalize_kwargs, phase_site=pair.where[0])
            loss = (post if post is not None else summary['infidelity']) if accepted else pre
            local_metrics = self._record_local_gate_fidelity(
                summary['fidelity'] if accepted else summary['warmstart_fidelity'],
            )
            refinement = None
            if refine_enabled:
                key = strip_key(where)
                next_where = self.gates[step][1] if step < len(self.gates) else None
                next_key = strip_key(next_where)
                finished = (key != next_key or next_key is None or
                            tuple(sorted(tuple(site) for site in next_where)) in refine_bonds)
                if finished:
                    if normalize_target:
                        self._normalize_state(refine_target, normalize_chi=normalize_chi,
                                              normalize_kwargs=normalize_kwargs)
                    refine_calibration = None
                    if self.boundary_convergence is not None:
                        refine_calibration = self._calibrate_sweep_boundaries(
                            self.state, refine_target, normalize_chi=normalize_chi,
                            evaluation_chi=evaluation_chi,
                        )
                        evaluation_chi = refine_calibration['chi']
                        normalize_chi = evaluation_chi[0]
                        boundary = refine_calibration.get('fit_boundaries', {}).get('bdy')
                    refine_chi = normalize_chi if refine_calibration else norm_environment_chi
                    self.state, refinement, boundary = self._full_update_refine(
                        refine_target, key, chi=refine_chi, boundary=boundary,
                    )
                    refinement['start_step'] = refine_start
                    refinement['end_step'] = step
                    refinement['boundary_convergence'] = (refine_calibration['record']
                                                           if refine_calibration else None)
                    if refinement['accepted']:
                        if normalize_final:
                            self._normalize_cached_pair(self.state, boundary, chi=normalize_chi,
                                                        normalize_kwargs=normalize_kwargs, phase_site=pair.where[0])
                        # Pair diagnostics preceded refinement against a different
                        # (block) target. Never label them as the final state score.
                        if measure_infidelity and measure_final_infidelity:
                            post = self.estimate_infidelity(
                                self.state, target, **{**metric_kwargs, 'chi': evaluation_chi,
                                                      'norm_target': None, 'evaluation_max_retries': 0},
                            )
                            raw_post = self._last_evaluation_raw_infidelity
                            step_evaluation_chi = self._last_evaluation_chi
                            loss = post
                        else:
                            loss = None
                    refine_target, refine_bonds = None, set()
            fidelity, geometric = self._record_fidelity_progress(loss)
            record = {
                'step': step, 'start_step': step, 'where': where, 'which': which,
                'original_steps': [self.last_gate_order['original_steps'][step - 1]],
                'gate_order': self.last_gate_order['policy'], 'update_style': 'two-site',
                'batch_size': 1, 'two_site_batch': 1, 'k_2q_batch': 1,
                'batch_stop_reason': 'gate_count', 'batch_target_bond_limit': None,
                'target_max_bond': self._max_bond(target), 'state_max_bond': self._max_bond(self.state),
                'normalize_chi': normalize_chi, 'evaluation_chi': evaluation_chi,
                'norm_environment_chi': norm_environment_chi,
                'effective_evaluation_chi': step_evaluation_chi,
                'evaluation_records': deepcopy(self.evaluation_records[evaluation_start:]),
                'boundary_convergence': calibration['record'] if calibration is not None else None,
                'cutoff': cutoff, 'cutoff_mode': cutoff_mode,
                'pre_infidelity': pre, 'post_infidelity': post,
                'raw_pre_infidelity': raw_pre, 'raw_post_infidelity': raw_post,
                'optimizer_infidelity': summary['infidelity'], 'final_infidelity': loss,
                'fidelity': fidelity, 'geometric_fidelity': geometric,
                'optimized': accepted, 'optimizer_attempted': True,
                'reason': 'optimized' if accepted else 'optimizer_rejected', 'optimizer_result': summary,
                'strip_refinement': refinement,
                **local_metrics,
            }
            if before is not None:
                record['timing'] = self._phase_timer.since(before)
            self.step_records.append(record)
            self.last_result = {'step': step, 'where': where, 'state': self.state,
                                'reason': record['reason'], 'infidelity': loss}
            if step_callback is not None:
                step_callback(deepcopy(record))
        if normalize_final and self.gates and self._site_count(self.gates[-1][1], self.state) == 1:
            self._normalize_state(self.state, normalize_chi=normalize_chi,
                                  normalize_kwargs={**(normalize_kwargs or {}), 'chi': normalize_chi})
        return self.state

    @profile_run
    @ordered_gate_run
    def run(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        self,
        *,
        mode=None,
        update_style=None,
        gate_order=None,
        progbar=False,
        progress=None,
        cutoff="auto",
        cutoff_mode="auto",
        k_2q_batch="auto",
        non_unitary=False,
        normalize_target=None,
        normalize_initial=None,
        normalize_chi=None,
        evaluation_chi=None,
        normalize_final=True,
        infidelity_tol="auto",
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
        sweep_progress: bool | None = None,
        global_kwargs: Mapping[str, Any] | None = None,
        global_optimize_kwargs: Mapping[str, Any] | None = None,
        reset_traces=True,
        timing=False,
        timing_sync_device=False,
        step_callback=None,
    ):
        """Run the queued gate stream and return the compressed state.

        One-site gates are applied directly unless they are folded into a
        two-site batch. Two-site gates are applied exactly to form a target,
        then compressed to ``chi`` to form a warm start. The warm start is
        accepted immediately when it is already inside ``infidelity_tol``;
        otherwise it can be refined by the selected sweep or global optimizer.

        Parameters
        ----------
        mode : {"sweep", "global", "full-update"} | None, optional
            Temporary backend override for this run.
        update_style, gate_order : str | None, optional
            Temporary overrides of the constructor's local update scope and
            commuting-gate traversal. The original gate queue is preserved.
        progress, progbar : bool, optional
            Show the outer PEPS progress bar. ``progress`` is preferred;
            ``progbar`` is kept as a short alias.
        cutoff, cutoff_mode : float | str, default="auto"
            Truncation settings passed to gate application and target-derived
            warm-start compression. Automatic cutoff follows the current PEPS
            dtype: 1e-12 for float64/complex128, 1e-6 for float32/complex64,
            and 1e-3 for 16-bit data. Automatic cutoff mode is "rsum2".
        k_2q_batch : {"auto"} | int, default="auto"
            Automatic batches stop before a gate makes any target bond exceed
            ``2 * chi`` for diagonal qubit gates (including RZZ), or
            ``4 * chi`` when other two-site gates enter the batch.
            Gates may share sites; actual bond growth controls batching.
            A nearest-neighbor RZZ layer acting once on each bond therefore
            fits in one target within ``2 * chi``.
            The first two-site gate is always included, even if
            its exact target alone exceeds this budget. No target truncation
            is used to satisfy the budget. Positive integers absorb up to that
            many two-site gates, including overlapping ones, without a target
            bond budget. One-site gates encountered inside a batch are applied
            in circuit order and do not count toward the two-site limit.
            Leading one-site gates are applied directly; trailing one-site
            gates join an automatic batch until its next stopping condition.
        non_unitary : bool, default=False
            If ``True``, target states produced by gates are explicitly
            normalized before fidelity estimates and optimization.
        normalize_target : bool | None, default=None
            Follow ``non_unitary`` by default: unitary gate targets are not
            rescaled. True/False explicitly enables/disables target
            normalization. Unitary runs assume target norm one by default;
            use infidelity_kwargs={"norm_target": None} to measure it.
        normalize_initial : bool | None, optional
            Override the constructor's one-time initial normalization setting.
        normalize_chi : int | tuple[int, int] | None, optional
            Per-run PEPS normalization boundary chi override. This affects
            initial, target, warm-start, and final-candidate normalization
            calls made during this run.
        evaluation_chi : int | tuple[int, int] | None, optional
            Per-run boundary chi override for pre/post infidelity estimates.
            This is the recommended way to judge acceptance with a larger chi
            than the optimizer environment uses.
        normalize_final : bool, default=True
            Normalize optimized candidates before acceptance, retained targets
            that already fit chi, and output after standalone one-site gates.
            Truncated warm starts are always normalized before optimization.
            When disabled, target norms are measured unless explicitly supplied.
        infidelity_tol : float | {"auto"}, default="auto"
            Accept the chi-truncated warm start without optimization when its
            estimated infidelity is at or below this threshold. Automatic
            values use the MPS FIT tolerance scale: 1e-9 for float64/complex128,
            1e-5 for float32/complex64, and 1e-3 for 16-bit data. This is a
            refinement-entry threshold, not a guarantee on the final error.
        measure_infidelity : bool, default=True
            Measure the warm-start local infidelity before deciding whether to
            optimize. If disabled, optimization still resolves an unknown
            target norm once for its objective; adaptive boundary checking
            measures and reuses the target norm even for unitary targets.
        optimize : bool, default=True
            If ``False``, accept the warm start after the optional infidelity
            estimate.
        measure_final_infidelity : bool, default=True
            Re-estimate infidelity after variational cleanup before deciding
            whether to accept the optimized state. Keep this enabled when
            ``accept_if_improved`` should compare pre/post candidates at the
            same ``evaluation_chi``.
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
        sweep_progress : bool | None, optional
            Override the constructor-level internal directional sweep-bar
            setting for this run. ``None`` follows the constructor setting,
            then the outer ``progress`` value.
        global_kwargs, global_optimize_kwargs : mapping, optional
            Per-run global backend and optimizer overrides.
        reset_traces : bool, default=True
            Reset diagnostic traces at the start of this call. Calling
            ``run()`` again still applies the queued gates to the current
            state; this flag controls only the recorded losses, infidelities,
            step records, and normalization events.
        timing : bool, default=False
            Record per-batch and per-run target, compression, normalization,
            fidelity, and optimizer phase wall times. Sweep summaries include
            the existing boundary-update and local-solve slice timings.
        timing_sync_device : bool, default=False
            Synchronize at outer phase boundaries when timing is enabled.
            Inner sweep slice times remain unsynchronized host wall times.
        step_callback : callable | None, optional
            Called after each completed two-site batch with a detached scalar
            record. Allows incremental diagnostic output during long runs.
        """
        # Resolve on every run, before normalization or gate application. A
        # replacement state can have a different dtype. Quimb compress_all
        # and global cleanup must receive concrete numerical policies.
        if step_callback is not None and not callable(step_callback):
            raise TypeError("step_callback must be callable or None")
        cutoff = _resolve_gate_cutoff(self.state, cutoff)
        cutoff_mode = _resolve_gate_cutoff_mode(cutoff_mode)
        infidelity_tol = self._resolve_infidelity_tol(infidelity_tol)
        normalize_chi = self._boundary_chi_for_norm(normalize_chi, options=normalize_kwargs)
        evaluation_chi = self._boundary_chi_for_infidelity(evaluation_chi, options=infidelity_kwargs)
        run_mode = self.mode if mode is None else self._normalize_mode(mode)
        style = self.update_style if update_style is None else self._normalize_update_style(update_style)
        if run_mode == 'global' and style != 'row-column':
            raise ValueError('update_style applies to sweep mode, not global mode')
        if run_mode == 'sweep' and style == 'two-site':
            run_mode = 'full-update'
        if run_mode == 'full-update' and style in {'row', 'column'}:
            raise ValueError('full-update uses two-site updates, not row/column updates')
        if run_mode == 'sweep' and style in {'row', 'column'}:
            sweep_optimize_kwargs = {**(sweep_optimize_kwargs or {}),
                                     'axes': ('x',) if style == 'row' else ('y',)}
        adaptive_boundary = run_mode in {"sweep", "full-update"} and optimize and self.boundary_convergence is not None
        if adaptive_boundary:
            self._validate_adaptive_fit_caps(sweep_kwargs)
        if adaptive_boundary and self.boundary_convergence["schedule"] == "d2":
            evaluation_chi = self.boundary_convergence["start_chi"]
            normalize_chi = evaluation_chi[0]
            normalize_kwargs = {**(normalize_kwargs or {}), "chi": normalize_chi}
        if adaptive_boundary and self._boundary_chi_floor is not None:
            normalize_chi = max(chi_pair(normalize_chi)[0], self._boundary_chi_floor[0])
            evaluation_chi = tuple(max(a, b) for a, b in
                                   zip(chi_pair(evaluation_chi), self._boundary_chi_floor))
            normalize_kwargs = {**(normalize_kwargs or {}), "chi": normalize_chi}
        show_progress = bool(progbar if progress is None else progress)
        effective_sweep_progress = (
            self.sweep_progress
            if sweep_progress is None
            else bool(sweep_progress)
        )
        normalize_target = bool(non_unitary) if normalize_target is None else bool(normalize_target)
        metric_kwargs = dict(infidelity_kwargs or {})
        default_target_norm = 1.0 if not non_unitary and normalize_final else None
        metric_kwargs.setdefault(
            "norm_target", self.infidelity_kwargs.get(
                "norm_target", self.boundary_kwargs.get("norm_target", default_target_norm),
            ),
        )
        accept_if_improved = (
            self.accept_if_improved
            if accept_if_improved is None
            else bool(accept_if_improved)
        )
        if isinstance(k_2q_batch, str) and k_2q_batch.strip().lower() == "auto":
            k_2q_batch = "auto"
        else:
            k_2q_batch = self._validate_scalar_chi(k_2q_batch, name="k_2q_batch")
        if run_mode == 'full-update':
            if k_2q_batch not in ('auto', 1):
                raise ValueError('full-update is gate-by-gate; use k_2q_batch=1 or auto')
            if (self.which is not None or any(entry[2] is not None for entry in self.gates)
                    or self.gate_kwargs or self.target_gate_kwargs or self.warmstart_gate_kwargs
                    or gate_kwargs):
                raise ValueError('full-update accepts physical PEPS gate tensors directly; '
                                 'which and custom gate application options are unsupported')
            if not optimize or self.boundary_engine == 'quimb-mps' or self.boundary_options:
                raise ValueError('full-update requires optimize=True and Pepsy DMRG boundary environments')
            if (sweep_kwargs or sweep_optimize_kwargs or self.sweep_kwargs or self.sweep_optimize_kwargs
                    or global_kwargs or global_optimize_kwargs):
                raise ValueError('full-update uses full_update_kwargs and boundary_kwargs, not sweep/global run overrides')
        if reset_traces:
            self._reset_traces()

        self._ensure_initial_normalized(
            normalize_initial=normalize_initial,
            normalize_kwargs=normalize_kwargs,
            normalize_chi=normalize_chi,
        )
        self._require_torch_symmray_backend(self.state, role="state")
        if run_mode == 'full-update':
            return self._run_full_update(
                normalize_target=normalize_target,
                normalize_final=normalize_final, normalize_chi=normalize_chi,
                evaluation_chi=evaluation_chi, normalize_kwargs=normalize_kwargs,
                measure_infidelity=measure_infidelity, measure_final_infidelity=measure_final_infidelity,
                accept_if_improved=accept_if_improved, improvement_tol=improvement_tol,
                metric_kwargs=metric_kwargs, cutoff=cutoff, cutoff_mode=cutoff_mode,
                gate_kwargs=gate_kwargs, step_callback=step_callback, progress=show_progress,
            )

        pbar = None
        if show_progress:
            from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(self.gates),
                desc=f"PEPS {run_mode}",
                unit="gate",
                leave=True,
                position=0,
                ascii=True,
                dynamic_ncols=True,
                mininterval=0.2,
                colour=self._PROGBAR_COLORS[run_mode],
            )

        two_site_count = 0
        needs_final_normalization = False
        idx = 0
        while idx < len(self.gates):
            step = idx + 1
            gate_payload, where, which = self.gates[idx]
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
            final_state = self.state
            record_step = step
            record_where = where
            advanced = 1
            step_evaluation_chi = evaluation_chi
            target_norm = metric_kwargs["norm_target"]
            calibration = None
            effective_sweep_kwargs = sweep_kwargs

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
                self._require_torch_symmray_backend(
                    self.state,
                    role="state after gate application",
                )
                if normalize_target:
                    self._normalize_state(
                        self.state,
                        normalize_kwargs=normalize_kwargs,
                        normalize_chi=normalize_chi,
                    )
                needs_final_normalization = not normalize_target
                final_state = self.state
                idx += 1
            elif site_count == 2:
                timing_before = self._phase_timer.snapshot() if self._phase_timer else None
                evaluation_start = len(self.evaluation_records)
                state_before = self.state
                if k_2q_batch == "auto":
                    batch_entries, two_site_in_batch, next_idx, target, batch_stop, batch_bond_limit = (
                        self._collect_auto_batch_target(
                            idx, cutoff=cutoff, cutoff_mode=cutoff_mode,
                            gate_kwargs=gate_kwargs,
                        )
                    )
                else:
                    batch_entries, two_site_in_batch, next_idx = self._collect_gate_batch(
                        idx, k_2q_batch,
                    )
                    target = self._build_batch_target(
                        state_before, batch_entries, cutoff=cutoff,
                        cutoff_mode=cutoff_mode, gate_kwargs=gate_kwargs,
                    )
                    batch_stop = "gate_count" if next_idx < len(self.gates) else "end_of_queue"
                    batch_bond_limit = None
                if two_site_in_batch < 1:
                    raise RuntimeError("Gate batch unexpectedly contains no two-site gates.")

                two_site_count += two_site_in_batch
                record_where, record_which = self._batch_record_payload(batch_entries)
                if isinstance(record_which, tuple):
                    record_which = tuple(
                        self.which if which_i is None else which_i
                        for which_i in record_which
                    )
                else:
                    record_which = self.which if record_which is None else record_which
                record_step = next_idx
                advanced = next_idx - idx
                self._require_torch_symmray_backend(
                    state_before,
                    target,
                    role="state and gate-generated target",
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
                    if normalize_final and not normalize_target:
                        self._normalize_state(
                            self.state, normalize_kwargs=normalize_kwargs,
                            normalize_chi=normalize_chi,
                        )
                    final_state = self.state
                    final_infidelity = 0.0
                    reason = "within_chi"
                else:
                    warmstart = self._build_batch_warmstart(
                        target,
                        state_before,
                        batch_entries,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        gate_kwargs=gate_kwargs,
                    )
                    if adaptive_boundary:
                        calibration = self._calibrate_sweep_boundaries(
                            warmstart, target, normalize_chi=normalize_chi,
                            evaluation_chi=evaluation_chi, sweep_kwargs=sweep_kwargs,
                            sweep_optimize_kwargs=sweep_optimize_kwargs,
                        )
                        normalize_chi = calibration["chi"][0]
                        evaluation_chi = calibration["chi"]
                        step_evaluation_chi = evaluation_chi
                        normalize_kwargs = {**(normalize_kwargs or {}), "chi": normalize_chi}
                        metric_kwargs = {**metric_kwargs, "chi": evaluation_chi,
                                         "norm_target": calibration["sample"]["norm_target"],
                                         "evaluation_max_retries": 0}
                        target_norm = metric_kwargs["norm_target"]
                        effective_sweep_kwargs = {**(sweep_kwargs or {}), "chi": evaluation_chi,
                                                  "target_norm": target_norm}
                        if calibration.get('fit_boundaries') is not None:
                            effective_sweep_kwargs['_checked_boundaries'] = calibration['fit_boundaries']
                            effective_sweep_kwargs['_checked_target'] = calibration['fit_target']
                        # Reuse the checked norm rather than contracting it again.
                        checked_norm = calibration["sample"]["norm"]
                        for checked in (checked_norm, target_norm):
                            if (not all(math.isfinite(float(v)) for v in
                                        (complex(checked[0]).real, complex(checked[0]).imag, checked[1]))
                                    or complex(checked[0]).real <= 0):
                                raise ValueError("Adaptive boundary check found an unusable norm.")
                        custom_normalization = _merge_opts(self.normalize_kwargs, normalize_kwargs)
                        if any(key != "chi" for key in custom_normalization):
                            # Explicit normalization methods/options remain authoritative.
                            self._normalize_state(warmstart, normalize_kwargs=normalize_kwargs,
                                                  normalize_chi=normalize_chi)
                        else:
                            if calibration.get('fit_boundaries') is not None:
                                # Keep normalization's global magnitude in the
                                # TN exponent so all unchanged cuts stay valid.
                                # Preserve the old complex phase convention on
                                # one owned tensor, rather than scaling every site.
                                self._normalize_without_rescaling_sites(warmstart, checked_norm)
                            else:
                                _normalize_by_scaled_norm(warmstart, checked_norm)
                            self.normalizations.append(self._normalization_record(warmstart, checked_norm))
                    else:
                        self._normalize_state(
                            warmstart,
                            normalize_kwargs=normalize_kwargs,
                            normalize_chi=normalize_chi,
                        )

                    precheck_reliable = True
                    raw_pre_infidelity = None
                    if measure_infidelity:
                        self._last_evaluation_raw_infidelity = None
                        if calibration is not None:
                            sample = calibration["sample"]
                            raw = 1.0 - _scaled_overlap_fidelity(
                                sample["overlap"], sample["norm"], sample["norm_target"],
                            )
                            roundoff = _resolve_gate_cutoff(warmstart, "auto")
                            pre_infidelity = (self._clip_fidelity(raw)
                                              if math.isfinite(raw) and -roundoff <= raw <= 1 + roundoff
                                              else None)
                            self._last_evaluation_chi = evaluation_chi
                            self._last_evaluation_raw_infidelity = raw
                            self._last_target_norm = sample["norm_target"]
                            self.evaluation_records.append({
                                "requested_chi": evaluation_chi, "effective_chi": evaluation_chi,
                                "raw_infidelity": raw, "infidelity": pre_infidelity,
                                "source": "boundary_convergence", "converged": calibration["converged"],
                                "attempts": [{"chi": evaluation_chi, "infidelity": raw}],
                            })
                        else:
                            pre_infidelity = self.estimate_infidelity(
                                warmstart,
                                target,
                                evaluation_chi=evaluation_chi,
                                **metric_kwargs,
                            )
                        step_evaluation_chi = self._last_evaluation_chi
                        target_norm = self._last_target_norm
                        raw_pre_infidelity = self._last_evaluation_raw_infidelity
                        if run_mode == "sweep" and raw_pre_infidelity is not None:
                            roundoff = _resolve_gate_cutoff(warmstart, "auto")
                            precheck_reliable = (
                                -roundoff <= raw_pre_infidelity <= 1.0 + roundoff
                            )
                            if calibration is not None:
                                precheck_reliable = precheck_reliable and calibration["converged"]

                    if (precheck_reliable and pre_infidelity is not None
                            and pre_infidelity <= infidelity_tol):
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
                        if target_norm is None:
                            target_norm = self._measure_target_norm(
                                target, evaluation_chi=evaluation_chi,
                                metric_kwargs=metric_kwargs,
                            )
                        if pbar is not None:
                            pbar.set_postfix(self._progress_postfix(
                                two_site_count=two_site_count,
                                step=record_step,
                                total_steps=len(self.gates),
                                status="optimizing",
                                input_infidelity=pre_infidelity,
                            ))
                        warmstart_snapshot = None
                        if accept_if_improved and pre_infidelity is not None:
                            warmstart_snapshot = (
                                warmstart.copy()
                                if hasattr(warmstart, "copy")
                                else warmstart
                            )
                        effective_sweep_options = _merge_opts(
                            self.sweep_optimize_kwargs, sweep_optimize_kwargs,
                        )
                        if calibration is not None:
                            # The precheck selects the fit caps once. A stale
                            # optimize-time chi must not reset them mid-sweep.
                            # Constructor already received the selected pair;
                            # None avoids a second eager boundary expansion.
                            effective_sweep_options["chi"] = None
                        if run_mode == "sweep" and not effective_sweep_options.get("debug", False):
                            # The outer postcheck evaluates the retained candidate
                            # after normalization. Keep that authoritative check.
                            if (measure_infidelity and measure_final_infidelity
                                    and not effective_sweep_options.get("debug_loss_kwargs")):
                                effective_sweep_options.setdefault("compute_final_loss", False)
                            # Direct compression is independent of the boundary
                            # warm start and iteration budget. Reuse a precheck
                            # only for the identical default metric policy/caps;
                            # customized metrics keep a separate sweep check.
                            if (
                                pre_infidelity is not None
                                and self.boundary_kwargs == {
                                    **_DEFAULT_BOUNDARY_KWARGS, "fit_mode": "direct",
                                }
                                and not self.infidelity_kwargs and not infidelity_kwargs
                                and not self.sweep_kwargs and not sweep_kwargs
                                and not self.boundary_options
                                and normalize_boundary_engine(
                                    self.boundary_engine, warmstart, target,
                                ) == "dmrg"
                                and step_evaluation_chi == self.boundary_chi
                                and not effective_sweep_options.get("debug_loss_kwargs")
                                and not effective_sweep_options.get("renormalize", False)
                                and not effective_sweep_options.get("normalize_boundaries", False)
                                and "chi" not in effective_sweep_options
                            ):
                                effective_sweep_options.setdefault(
                                    "initial_loss", pre_infidelity if precheck_reliable
                                    else raw_pre_infidelity,
                                )
                        final_state, opt_infidelity, optimizer_result = self._optimize_state(
                            warmstart,
                            target,
                            mode=run_mode,
                            target_norm=target_norm,
                            progress=show_progress,
                            sweep_progress=effective_sweep_progress,
                            cutoff=cutoff,
                            normalize_chi=normalize_chi,
                            sweep_kwargs=effective_sweep_kwargs,
                            sweep_optimize_kwargs=effective_sweep_options,
                            global_kwargs=global_kwargs,
                            global_optimize_kwargs=global_optimize_kwargs,
                        )
                        candidate_state = final_state
                        cleanup_failed = (
                            isinstance(optimizer_result, Mapping)
                            and optimizer_result.get("success") is False
                        )
                        if normalize_final and not cleanup_failed:
                            self._normalize_state(
                                candidate_state,
                                normalize_kwargs=normalize_kwargs,
                                normalize_chi=normalize_chi,
                            )
                        if measure_infidelity and measure_final_infidelity and not cleanup_failed:
                            # A retry must not compare candidates measured at
                            # different caps. Use the precheck's effective cap,
                            # and remeasure the snapshot if the postcheck grows it.
                            comparison_opts = dict(metric_kwargs)
                            comparison_opts["chi"] = step_evaluation_chi
                            post_infidelity = self.estimate_infidelity(
                                candidate_state,
                                target,
                                **comparison_opts,
                            )
                            if self._last_evaluation_chi != step_evaluation_chi:
                                step_evaluation_chi = self._last_evaluation_chi
                                if warmstart_snapshot is not None:
                                    comparison_opts["chi"] = step_evaluation_chi
                                    comparison_opts.pop("evaluation_max_retries", None)
                                    self._last_evaluation_raw_infidelity = None
                                    pre_infidelity = self.estimate_infidelity(
                                        warmstart_snapshot, target,
                                        evaluation_max_retries=0, **comparison_opts,
                                    )
                                    if run_mode == "sweep":
                                        raw_pre = self._last_evaluation_raw_infidelity
                                        roundoff = _resolve_gate_cutoff(warmstart, "auto")
                                        precheck_reliable = (
                                            raw_pre is None
                                            or -roundoff <= raw_pre <= 1.0 + roundoff
                                        )
                        final_infidelity = (
                            post_infidelity
                            if post_infidelity is not None
                            else opt_infidelity
                        )
                        if final_infidelity is None:
                            final_infidelity = pre_infidelity

                        should_accept = not cleanup_failed
                        if (
                            not cleanup_failed
                            and accept_if_improved
                            and precheck_reliable
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
                            reason = "optimizer_failed" if cleanup_failed else "optimizer_rejected"

                needs_final_normalization = False
                fidelity, geometric_fidelity = self._record_fidelity_progress(final_infidelity)
                record = {
                    "step": int(record_step),
                    "start_step": int(step),
                    "original_steps": self.last_gate_order['original_steps'][idx:next_idx],
                    "gate_order": self.last_gate_order['policy'], "update_style": style,
                    "where": record_where,
                    "which": record_which,
                    "batch_size": len(batch_entries),
                    "two_site_batch": int(two_site_in_batch),
                    "k_2q_batch": k_2q_batch,
                    "batch_stop_reason": batch_stop,
                    "batch_target_bond_limit": batch_bond_limit,
                    "target_max_bond": target_max_bond,
                    "state_max_bond": self._max_bond(self.state),
                    "normalize_chi": normalize_chi,
                    "evaluation_chi": evaluation_chi,
                    "effective_evaluation_chi": step_evaluation_chi,
                    "evaluation_records": deepcopy(self.evaluation_records[evaluation_start:]),
                    "boundary_convergence": None if calibration is None else calibration["record"],
                    "cutoff": cutoff,
                    "cutoff_mode": cutoff_mode,
                    "infidelity_tol": infidelity_tol,
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
                if timing_before is not None:
                    record["timing"] = self._phase_timer.since(timing_before)
                if step_callback is not None:
                    step_callback(deepcopy(record))
                idx = next_idx
            else:
                raise ValueError("PepsOptimizer supports one- and two-site gates only.")

            self.last_result = {
                "step": int(record_step),
                "where": record_where,
                "state": final_state,
                "reason": reason,
                "infidelity": final_infidelity,
            }
            if pbar is not None:
                pbar.set_postfix(self._progress_postfix(
                    two_site_count=two_site_count,
                    step=record_step,
                    total_steps=len(self.gates),
                    status=reason,
                ))
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        if normalize_final and needs_final_normalization:
            self._normalize_state(
                self.state, normalize_kwargs=normalize_kwargs,
                normalize_chi=normalize_chi,
            )
        return self.state

    def get_fidelities(self):
        """Return the running geometric mean of measured local fidelities.

        This is a per-gate-batch monitor, not an exact global fidelity. For
        small systems, use a direct overlap with a high-chi or dense reference
        when you need the true state-vs-ideal fidelity.

        Zero local-fidelity estimates are floored only in this diagnostic trace
        so one saturated boundary estimate does not pin later progress to
        exactly zero.
        """
        return list(self.losses)

    def get_infidelities(self):
        """Return ``1 - prod(F_local)`` for measured gate-batch fidelities.

        This cumulative trace is a local-fidelity proxy, not an exact global
        state-vs-ideal infidelity. It can overestimate the true global
        infidelity by orders of magnitude because later gates can coherently
        rotate or partially recover earlier local truncation errors.

        The product uses the same tiny local-fidelity floor as
        :meth:`get_fidelities` to keep the trace finite.
        """
        return list(self.infidelities)

    def get_local_infidelities(self):
        """Return measured local infidelities, one per gate batch."""
        return list(self.local_infidelities)

    def get_step_records(self):
        """Return gate-batch records, stopping reasons and effective metric caps.

        Full updates always retain local pair fidelities before optional strip
        refinement. Their accumulated product is a diagnostic, not global fidelity.
        """
        return list(self.step_records)

    def get_boundary_convergence(self):
        """Return scalar probe records from this run and the retained cap floor."""
        return {"chi_floor": self._boundary_chi_floor,
                "records": deepcopy(self.boundary_convergence_records)}

    def get_timing(self):
        """Return detached phase totals for the last run, including failures."""
        return deepcopy(getattr(self, "_last_timing", {}))

    def get_normalizations(self):
        """Return lightweight normalization events recorded by this optimizer."""
        return list(self.normalizations)

    def get_fit_diagnostics(self):
        """Return FIT diagnostics collected from metric and sweep boundaries.

        Records are populated when ``fit_timing=True`` is enabled through
        ``boundary_kwargs`` or a per-call normalization/infidelity mapping.
        """
        return list(self.fit_diagnostics)

    def get_evaluation_records(self):
        """Return metric attempts, including requested and retry boundary caps."""
        return [
            {**record, "attempts": [dict(attempt) for attempt in record["attempts"]]}
            for record in self.evaluation_records
        ]
