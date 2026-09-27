"""MPS optimization helpers centered on :class:`MpsOptimizer`.

:class:`MpsOptimizer` replays a canonical bundled gate stream
``[(gate, where), ...]`` against an MPS, using one of several compression
backends. The default ``mode="direct"`` selects Quimb's direct compressor;
``mode="mpo"`` is only a compatibility alias for that algorithm.
``mode="perm"`` uses a lazy permutation swap network: non-local
two-site gates swap the right endpoint next to the left endpoint, apply the
gate with a local SVD split, and leave the resulting physical ordering in
place. The current physical-site-to-logical-site ordering is available as the
synchronized ``optimizer.qubits`` and ``optimizer.logical_order`` views.
For repeated layout-aware evolution, :meth:`MpsOptimizer.apply_layout`
installs a persistent position-to-logical mapping and never performs a
swap-back; logical readout is available through ``logical_order``,
``remap_sample``, and ``to_dense``.
Bare Quimb method names such as ``mode="src"`` and ``mode="zipup"`` are
accepted and normalized internally to ``"quimb-<method>"``. The explicit
``mode="direct"`` path (internally ``"quimb-direct"``) also accepts
explicit sub-MPO events of the form
``("submpo", mpo, where)`` or
``{"kind": "submpo", "mpo": mpo, "where": where}``.  In every mode the stream
may also carry *control events* that are state operations rather than gates.
Quimb compression modes apply sub-MPO events with ``gate_with_submpo_``;
DMRG keeps multi-site sub-MPOs as tagged lazy FIT target layers and uses the
same SRC warm-up policy as ordinary DMRG targets.  The stream may also carry:

* ``("measure", pauli, where[, outcome])`` — projectively measure a Pauli
  observable, collapse the MPS onto a sampled (or forced ``outcome``)
  eigenvalue, and append ``(pauli, where, outcome, prob)`` to
  :attr:`MpsOptimizer.measurements`.
* ``("cap", where, vec[, absorb])`` — contract site ``where``'s physical index
  with ``vec`` (e.g. ``[1, 1]``) and absorb the result into the ``absorb``
  (``"left"``/``"right"``) neighbour, shortening the MPS by one site.
* ``("reset", where[, basis])`` — mid-circuit reset of qubit(s) to the ``+1``
  eigenstate of ``basis`` (default ``"Z"``); the MPS length is unchanged.
* ``("measure_reset", basis, where[, outcome])`` — measure each target in
  ``basis``, record the outcome(s), then reset to the ``+1`` eigenstate.

Control events split the stream into gate/subMPO segments run through the
active mode and are applied directly to the state between segments, so the same
stream works in every mode. The default gate path assumes a norm-preserving
stream. Compressed DMRG/FIT, mixed, direct, swap/permutation, and SVD modes can
restore the raw unitary working norm, preventing deep low-precision
underflow. Non-unitary streams should use ``non_unitary=True``; when
``normalize_every`` is enabled this moves the orthogonality center to one site
after every replay step, normalizes that center tensor, and accumulates the
removed scale into ``p.exponent``. Quimb includes that exponent in ``p.norm()``,
so ``p.norm()`` still reports the represented state norm; inspect a copy with
``exponent=0`` to see the rescaled data norm.

On dense MPS states, a multi-site Pauli measurement is represented as a
bond-two windowed sub-MPO for ``(I + m P) / 2``. DMRG modes attach that operator
to an exact lazy target and use the regular FIT schedule and SRC warm-start to
compress the post-measurement state. Native Symmray and fermionic states keep
their dense projector fallback so charge and dummy-mode metadata are preserved.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from numbers import Integral
import math
import time
import types
import warnings
import weakref
import autoray as ar
import numpy as np
import quimb.tensor as qtn

# Retain legacy parser imports; new consumers use the shared owner directly.
from .._stream_events import (  # noqa: F401
    _CONDITIONAL_EVENT_ALIASES,
    _CONTROL_EVENT_NAMES,
    _MEASURE_RESET_ALIASES,
    _MEASURE_RESET_AXIS_ALIASES,
    _MISSING,
    _RESET_AXIS_ALIASES,
    _RESET_FLIP_AXES,
    _SUBMPO_EVENT_NAMES,
    _canonical_control_name,
    _conditional_event_parts,
    _conditional_support,
    _control_event_contains_cap,
    _control_event_parts,
    _is_axis_string,
    _is_control_event,
    _is_submpo_event,
    _normalize_absorb,
    _normalize_condition_bit,
    _normalize_control_axes,
    _normalize_control_outcomes,
    _normalize_control_where,
    _normalize_event_name,
    _normalize_submpo_where,
    _parse_control_mapping,
    _parse_control_tuple,
    _parse_measure_reset_tuple,
    _parse_reset_tuple,
    _resolve_conditional,
    _submpo_event_parts,
    conditional_event_parts,
    is_submpo_event,
    normalize_submpo_where,
    submpo_event_parts,
)
from ..._internal.quimb import quimb_compression_options
from ...backends import (
    backend_infer,
    backend_signatures_compatible,
    infer_backend_converter_from_sample,
    infer_backend_signature,
)
from ...fitting.local import FIT
from ..._internal.cutoff import dtype_auto_cutoff
from ..._internal.random import backend_random_array
from ..._internal.quimb import (
    quimb_1d_compression_method_available as _quimb_compression_method_available,  # noqa: F401
    quimb_1d_compression_cutoff_mode as _quimb_compression_cutoff_mode,
    run_seeded_quimb as _run_seeded_quimb,
    quimb_fit_guess_method,
    require_quimb_1d_compression_method as _require_quimb_compression_method,
)
from ...tensors.observables import mps_entanglement_entropy as _mps_entanglement_entropy
from ...tensors.observables import tn_fidelity
from ...operators.gates import gate as apply_gate
from . import _layout_execution
from ._streams import (  # noqa: F401 -- historical import aliases
    _MpsStreamPlan,
    _PAULI_1Q,
    _SYMBOLIC_GATE_NAMES,
    _SYMBOLIC_ONE_QUBIT_GATES,
    _SYMBOLIC_ONE_QUBIT_ROTATIONS,
    _SYMBOLIC_TWO_QUBIT_GATES,
    _SYMBOLIC_TWO_QUBIT_ROTATIONS,
    _contains_symbolic_gate,
    _normalize_gate_queue,
    _normalize_gate_where,
    _prepare_gate_stream,
    _resolve_symbolic_gate_entry,
    _resolve_symbolic_gate_stream,
    _symbolic_gate_entry,
    _symbolic_rotation_gate,
    _symbolic_rotation_name,
    _symbolic_targets,
)
from .compression import (  # noqa: F401 -- historical import aliases
    _MPO_COMPRESSION_METHODS,
    _MPO_METHODS_IGNORE_CUTOFF,
    _MPO_METHODS_IGNORE_CUTOFF_MODE,
    _MPO_METHODS_NEED_INTERIOR_WORKAROUND,
    _MPO_METHODS_USE_SEED,
    _apply_dense_gate_with_method,
    _apply_submpo_with_interior_workaround,
    _apply_submpo_with_interior_workaround_impl,
    _is_interior_submpo_span,
    guess,
    svd_guess,
)
from .diagnostics import (
    _FIT_TIMING_PHASES,  # noqa: F401 -- retain the existing private import path
    _format_layout_reduction,
    _format_layout_value,
    _layout_report_text,
    _summarize_fit_timing,
)
from ._exact_batch import iter_exact_batches, supports_batch
from .layout import (
    MpsGateStreamLayoutFinder,
    _normalize_layout_support,
    _normalize_site_roles,
    _gate_stream_site_usage,
    _unique_ordered,  # noqa: F401 -- retain the historical private import
)

from . import _controls, _norm

from ._controls import _CONTROL_CLIFFORDS  # noqa: F401 -- historical alias

__all__ = [
    "MpsOptimizer",
    "guess",
    "is_submpo_event",
    "normalize_submpo_where",
    "submpo_event_parts",
    "svd_guess",
]


_EXACT_MODES = frozenset({"exact", "exact-batch"})
_NORM_INCLUDES_EXPONENT_CACHE = {}
_SHOT_DEFAULT_MAX_BRANCHES = 128
_SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS = 0.1
_DEFAULT_FIT_INIT_STRATEGY = "guess_src"
_DEFAULT_CUTOFF_MODE = "rsum2"
_FIT_INIT_STRATEGIES = frozenset(
    {"auto", "direct", "random", "random_expand", "svd_guess"}
    | {f"guess_{method}" for method in _MPO_COMPRESSION_METHODS}
)


class _DeprecatedOptionDefault:
    """Readable signature sentinel for compatibility-only keyword values."""

    def __repr__(self):
        return "<deprecated>"


_DEPRECATED_OPTION = _DeprecatedOptionDefault()


def _array_backend_signature(array):
    """Return comparable backend / dtype / device metadata for an array."""
    return infer_backend_signature(array)


class MpsOptimizer:  # pylint: disable=too-many-instance-attributes
    """High-level wrapper for MPS gate-sweep objectives.

    Parameters
    ----------
    p : qtn.MatrixProductState
        Initial MPS state.
    gates : sequence[object] | None, optional
        Canonical bundled gate stream ``((gate, where), ...)`` (outer list/tuple
        accepted). If omitted, start with an empty queue and use
        :meth:`set_gates` or :meth:`add_gates` before ``run``. Each ``gate`` is
        applied on the ket family only (state evolution), using :func:`pepsy.operators.gates.gate`.
        Named entries matching the stabilizer stream grammar are also accepted,
        for example ``("H", 0)``, ``("rzz", theta, 0, 1)``, and the compact
        ``("rzz-0.23", 0, 1)`` form. They are
        materialized as ordinary gate matrices before stream validation.
        ``where`` supports one- or two-site locations in 1D/2D/3D forms.
        For a bare Quimb method such as ``mode="src"`` or the qualified
        ``mode="direct"`` or another ``mode="quimb-<method>"``, entries may
        also have the explicit sub-MPO form
        ``("submpo", mpo, where)`` or mapping form
        ``{"kind": "submpo", "mpo": mpo, "where": where}``, with a 1D
        support ``where``. :meth:`submpo_event` builds the tuple form.
        In any mode the stream may also carry control events
        ``("measure", pauli, where[, outcome])``,
        ``("cap", where, vec[, absorb])``, ``("reset", where[, basis])``, and
        ``("measure_reset", basis, where[, outcome])`` (built by
        :meth:`measure_event`, :meth:`cap_event`, :meth:`reset_event`, and
        :meth:`measure_reset_event`). Classical feed-forward events use
        ``("if", record, bit, action)`` or the equivalent mapping form and
        apply ``action`` only when the referenced measurement has the
        requested computational bit (``+1 -> 0``, ``-1 -> 1``); negative
        records are offsets from the latest measurement. A
        ``cap`` event shortens the MPS, so later event site labels refer to the
        shortened chain. Stream-local trajectory events and stochastic entries
        are also accepted. They are replayed through the shot runner when
        :meth:`run` is called, so state-dependent channels, measurements, and
        feed-forward actions are sampled independently per trajectory.
        Numeric gate and sub-MPO payloads must already match the MPS backend
        and device. Named gate entries are generated internally and use
        ``to_backend`` (or the initial-state backend when it is omitted).
        Mismatches in user-supplied numeric payloads are rejected with
        preparation guidance; use an explicit converter before constructing
        the optimizer or calling :meth:`set_gates`.
    chi : int
        Positive target/max bond dimension used by compressed modes. Mixed mode
        requires the initial MPS to have ``max_bond() <= chi`` and keeps its
        committed DMRG/MPO results at or below this limit.
    mode : str, default="direct"
        Gate-replay compression algorithm. ``"direct"`` uses Quimb's direct
        compression to the requested bond dimension; it is the default and
        preferred public name. ``"mpo"`` and ``"quimb"`` remain compatibility
        aliases for ``"direct"``. MPO describes an operator representation,
        not a separate algorithm; explicit sub-MPO inputs work in direct mode.
        Other Quimb methods accept their bare or ``"quimb-<method>"`` names;
        legacy ``"mpo-<method>"`` spellings also remain accepted.
        ``"fit"`` is the clear alias of the historical
        ``"dmrg"`` spelling. ``"dmrg1"`` uses at most two two-site growth
        sweeps, then one-site refinement; once every bond reaches its
        attainable physical/``chi`` ceiling, it latches one-site updates
        for the rest of the replay. An already-capped window starts
        directly with one-site sweeps. ``"dmrg2"`` uses two-site updates
        for the required warm-up (two sweeps by default), then one-site
        refinement. ``"dmrg3"`` follows the same fixed warm-up policy with
        three-site updates before one-site refinement. ``"mix"`` uses a
        disposable, chi-capped ``guess-direct`` state followed by
        transactional one-site DMRG/FIT for every eligible multi-site gate,
        both while bonds are growing and after they reach ``chi``.
        ``mode="perm"`` routes non-local two-site gates with Quimb's
        swap-and-split SVD path and keeps the resulting physical ordering.
        ``"exact"`` contracts the full state without truncation.
        ``"exact-batch"`` is an opt-in dense replay path that automatically
        fuses single-/two-qubit gates and broadcasts compact diagonal blocks.
        ``"batch-exact"`` is an alias for the same mode. Neither exact mode
        tracks MPS canonical metadata.
    contraction_opt : object | None, default="auto-hq"
        Canonical contraction path optimizer keyword.
    ind_id : str, default="k{}"
        Format string for site index labels used by exact gate application.
        Use "k{},{}" when gate sites are 2D coordinates like ``(i, j)``.
    inplace : bool, default=False
        Whether to optimize the provided input state object directly. If
        ``False``, a copy is made and the original input remains unchanged.
    The input state and gate stream are snapshotted at construction. Repeated
    ``run(shots=...)`` calls therefore restart every trajectory from the same
    initial state rather than continuing from an earlier ensemble.
    to_backend : callable | None, default=None
        Optional converter for named gate entries. For example,
        ``to_backend=pepsy.backend_torch(dtype=torch.complex64)`` converts
        the matrices generated for ``("H", 0)`` and ``("rzz", theta, 0, 1)``
        before they cross the stream boundary. When omitted, the converter is
        inferred from the initial MPS for named gates. Existing numeric gate
        and sub-MPO payloads retain the strict explicit-preparation contract.
    qubit_roles : mapping or sequence, optional
        Optional logical-site metadata such as ``{0: "data", 1: "ancilla"}``
        or one role label per initial MPS site. The metadata is preserved in
        layout plans and can be used with ``role_order`` during layout search;
        it does not alter execution semantics.
    Attributes
    ----------
    measurements : list[tuple]
        Results of ``("measure", ...)`` control events, appended in order as
        ``(pauli, where, outcome, prob)`` where ``outcome`` is ``+1``/``-1`` and
        ``prob`` is the Born probability of that outcome before collapse.
        Mid-circuit ``reset`` measurements are not recorded here.
    normalizations : list[dict]
        Automatic normalization events recorded during :meth:`run`. Each entry
        stores the 1-based gate step, removed local squared scale,
        orthogonality span, tensor sites that were rescaled, and resulting
        base-10 ``p.exponent``. The raw tensor data are rescaled; the
        represented norm remains available through ``p.norm()`` because quimb
        applies ``p.exponent``.
    norm_events : list[dict]
        Automatic norm-survival records for compressed gates and physical
        projective/Kraus boundaries. Physical branch probabilities are stored
        on their event but are not multiplied into compression infidelity.
        Accelerator compression records retain detached backend scalars;
        ``get_norm_events()`` returns independent Python-valued records.
    quality_checks : list[dict]
        Optional finite-data and canonical-gauge health records from
        ``run(quality_check_every=...)``.
    last_run_timing : dict | None
        Most recent opt-in replay timing record from ``run(timing=True)``.
        The record contains total replay time, inclusive stage totals, and,
        for mixed mode, the final ``last_mix_summary``. Use
        :meth:`get_run_timing` for a copy.
    qubits : list[int]
        Physical-position to logical-site mapping. In ``mode="perm"`` this is
        updated after each successful no-swap-back gate, matching Quimb's
        ``CircuitPermMPS`` convention.
    logical_order : list[int]
        The shared readout/layout view of the physical-position mapping. It
        mirrors :attr:`qubits` during ``mode="perm"`` and stores the fixed
        mapping installed by :meth:`apply_layout` otherwise.
    """

    _DMRG_MODE_ALIASES = {"dmrg1": 1, "dmrg2": 2, "dmrg3": 3}
    _ALLOWED_MODES = frozenset(
        {
            "dmrg",
            "dmrg1",
            "dmrg2",
            "dmrg3",
            "quimb",
            "mpo",
            "mix",
            "swap",
            "perm",
            "svd",
            "exact",
            "exact-batch",
            "batch-exact",
        }
        | _MPO_COMPRESSION_METHODS
        | {f"mpo-{method}" for method in _MPO_COMPRESSION_METHODS}
        | {f"quimb-{method}" for method in _MPO_COMPRESSION_METHODS}
    )
    LayoutFinder = MpsGateStreamLayoutFinder
    _ALLOWED_SUBMPO_METHODS = _MPO_COMPRESSION_METHODS
    _PROGBAR_COLORS = {
        "dmrg": "#1f77b4",
        "mpo": "#2ca02c",
        "mix": "#17becf",
        "swap": "#ff7f0e",
        "perm": "#8c564b",
        "svd": "#d62728",
        "exact": "#9467bd",
        "exact-batch": "#9467bd",
    }

    @classmethod
    def _normalize_mode(cls, mode):
        """Validate and normalize execution mode."""
        mode_norm = str(mode).strip().lower()
        if mode_norm == "batch-exact":
            mode_norm = "exact-batch"
        # Public ``direct`` names the compression algorithm. Historical
        # ``mpo``/``quimb`` spellings select exactly that same path; retain
        # them as silent aliases, not distinct replay modes.
        if mode_norm in {"mpo", "quimb"}:
            mode_norm = "direct"
        # ``fit`` names the algorithm while ``dmrg`` preserves the historical
        # mode spelling. DMRG1/2/3 are readable block-size aliases that share
        # the same implementation and are normalized to ``dmrg``. The alias
        # is recorded by the constructor before this function runs, so the
        # shared implementation can still select the requested schedule.
        if mode_norm == "fit" or mode_norm in cls._DMRG_MODE_ALIASES:
            mode_norm = "dmrg"
        elif mode_norm in _MPO_COMPRESSION_METHODS:
            # Bare Quimb method names are the user-facing spelling. Keep the
            # qualified form as the canonical internal mode so old
            # ``quimb-*`` and ``mpo-*`` aliases continue to behave identically.
            mode_norm = f"quimb-{mode_norm}"
        if mode_norm not in cls._ALLOWED_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return mode_norm

    @classmethod
    def _is_mpo_mode(cls, mode):
        """Recognize Quimb compression, including the default direct mode.

        The private helper's historical name refers to the shared operator
        application machinery, not a public algorithm named ``mpo``.
        """
        mode_norm = str(mode).strip().lower()
        return (
            mode_norm in {"mpo", "quimb"}
            or mode_norm in _MPO_COMPRESSION_METHODS - {"fit"}
            or mode_norm.startswith(("mpo-", "quimb-"))
        )

    @classmethod
    def _progress_mode_name(cls, mode):
        """Return the short active mode name shown by a replay progress bar."""
        mode_norm = str(mode).strip().lower()
        if cls._is_mpo_mode(mode_norm):
            return cls._mode_mpo_method(mode_norm)
        return mode_norm

    @classmethod
    def _mode_mpo_method(cls, mode):
        """Return the compression method encoded by a Quimb mode name."""
        mode_norm = str(mode).strip().lower()
        if mode_norm in {"mpo", "quimb"}:
            return "direct"
        if mode_norm in _MPO_COMPRESSION_METHODS - {"fit"}:
            return cls._normalize_submpo_method(mode_norm)
        for prefix in ("quimb-", "mpo-"):
            if mode_norm.startswith(prefix):
                return cls._normalize_submpo_method(mode_norm[len(prefix) :])
        return "direct"

    def _resolve_mpo_method(self, method):
        """Resolve an explicit method or the method encoded by ``self.mode``."""
        if method is None:
            return self._mode_mpo_method(self.mode)
        return self._normalize_submpo_method(method)

    @classmethod
    def _dmrg_alias_block_size(cls, mode):
        """Return the fixed block size requested by a DMRG mode alias."""
        return cls._DMRG_MODE_ALIASES.get(str(mode).strip().lower())

    @staticmethod
    def _effective_max_bond(p=None):
        """Return a numeric maximum bond, treating product-state ``None`` as 1."""
        value = p.max_bond() if p is not None else None
        return 1 if value is None else int(value)

    @classmethod
    def _normalize_submpo_method(cls, method):
        """Validate and normalize the sub-MPO compression method."""
        method_norm = str(method).strip().lower()
        if method_norm not in cls._ALLOWED_SUBMPO_METHODS:
            raise ValueError(f"Unknown subMPO method: {method}")
        _require_quimb_compression_method(method_norm)
        return method_norm

    def _submpo_compress_opts(
        self,
        method,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Return compression options for a sub-MPO method."""
        opts = {}
        # ``cutoff`` controls discarded singular weight for ordinary methods.
        # SRC/SRCMPS/SDCR are rank-controlled randomized projections, so Quimb
        # intentionally ignores a singular-value cutoff for those methods.
        opts["cutoff"] = (
            0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff
        )
        cutoff_mode = _quimb_compression_cutoff_mode(method, cutoff_mode)
        if (
            cutoff_mode is not None
            and method not in _MPO_METHODS_IGNORE_CUTOFF_MODE
        ):
            opts["cutoff_mode"] = cutoff_mode
        if method == "fit-projector":
            # The optional simple-update pre-gauge is singular on exact
            # product-state bonds. Match the dense-gate path and let the
            # projector fit run without that gauge.
            opts["canonize"] = False
        if method == "direct":
            return opts
        # Quimb's ``auto`` paths already choose their own contraction tree.
        # Only forward an explicit optimizer so this wrapper does not turn a
        # harmless default into an unsupported nested contraction option.
        optimize = self.contraction_opt
        if optimize is None:
            return opts
        if isinstance(optimize, str) and optimize.strip().lower() in {
            "auto",
            "auto-hq",
        }:
            return opts
        opts["optimize"] = optimize
        return opts

    @staticmethod
    def submpo_event(mpo, where):
        """Return a canonical explicit sub-MPO stream event.

        The returned entry can be placed directly inside the ``gates`` stream
        for the Quimb compression mode family. ``where`` is restricted to 1D integer MPS
        sites.
        """

        return ("submpo", mpo, _normalize_submpo_where(where))

    @staticmethod
    def submpo_event_parts(entry, *, normalize_where=False):
        """Return ``(mpo, where)`` when ``entry`` is a sub-MPO event."""

        return submpo_event_parts(entry, normalize_where=normalize_where)

    @staticmethod
    def is_submpo_event(entry):
        """Return whether ``entry`` is an explicit sub-MPO stream event."""

        return is_submpo_event(entry)

    @staticmethod
    def measure_event(pauli, where, outcome=None):
        """Return a canonical Pauli-measurement stream event.

        Collapses the MPS onto a sampled (or forced ``outcome``) eigenvalue of
        the Pauli observable ``pauli`` on ``where`` and appends the result to
        :attr:`measurements`. ``pauli`` is a string such as ``"Z"`` or ``"ZZ"``
        with one axis per site in ``where``.
        """
        where = _normalize_control_where(where)
        if outcome is None:
            return ("measure", str(pauli), where)
        return ("measure", str(pauli), where, int(outcome))

    @staticmethod
    def cap_event(where, vec, absorb="left"):
        """Return a canonical cap stream event.

        Contracts the physical index of site ``where`` with ``vec`` (e.g.
        ``[1, 1]``) and absorbs the resulting matrix into the ``absorb``
        neighbour (``"left"`` or ``"right"``), shortening the MPS by one site.
        """
        (site,) = _normalize_control_where(where, single=True)
        return ("cap", site, np.asarray(vec, dtype=complex).ravel(), _normalize_absorb(absorb))

    @staticmethod
    def reset_event(where, basis="Z"):
        """Return a canonical mid-circuit reset stream event.

        Resets qubit(s) ``where`` to the ``+1`` eigenstate of ``basis`` by a
        measurement collapse followed by a conditional anticommuting Pauli flip.
        The MPS length is unchanged and the internal measurements are not
        recorded. The legacy ``basis="Z"`` form returns ``("reset", where)``.
        """
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(basis, where, event="reset")
        if all(axis == "Z" for axis in axes):
            return ("reset", where)
        return ("reset", where, "".join(axes))

    @staticmethod
    def measure_reset_event(pauli, where, outcome=None):
        """Return a canonical measure-then-reset stream event.

        Each target is measured in the corresponding single-site Pauli basis,
        the outcome is appended to :attr:`measurements`, and the target is then
        reset to the ``+1`` eigenstate of that basis. A one-character ``pauli``
        is broadcast across multiple sites.
        """
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(pauli, where, event="measure_reset")
        if outcome is None:
            return ("measure_reset", "".join(axes), where)
        outcomes = _normalize_control_outcomes(
            outcome, where, event="measure_reset"
        )
        if len(outcomes) == 1:
            return ("measure_reset", "".join(axes), where, outcomes[0])
        return ("measure_reset", "".join(axes), where, outcomes)

    @staticmethod
    def control_event_parts(entry):
        """Return ``(name, payload, where)`` when ``entry`` is a control event."""

        return _control_event_parts(entry)

    @staticmethod
    def is_control_event(entry):
        """Return whether ``entry`` is a measure/cap/reset/MR control event."""

        return _is_control_event(entry)

    @classmethod
    def gate_stream_layout(  # pylint: disable=too-many-locals
        cls,
        gate_stream,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        site_coords=None,
        order="quality",
        objective="locality",
        refine_passes=8,
        refine_numba=True,
        spectral_dense_max=512,
        recursive_dense_max=1024,
        nevergrad_budget=64,
        nevergrad_seed=0,
        nevergrad_optimizer="OnePlusOne",
        kahypar_config_path=None,
        kahypar_seed=0,
        from_scratch=False,
        weight_fn=None,
        weight_mode="auto",
        schmidt_max_dim=4,
        max_operator_qubits=8,
        qubit_roles=None,
        role_order=None,
    ):
        """Find a good 1D MPS layout for a bundled gate stream.

        The layout depends only on the stream supports, not on MPS tensor
        values.  The returned plan includes the optimized site order,
        old-site to new-position map, original stream locations, and internal
        mapped locations. It does not mutate or return a replacement gate
        stream.

        Parameters
        ----------
        gate_stream
            Canonical bundled stream accepted by :class:`MpsOptimizer`,
            including explicit sub-MPO and direct cap events. Direct caps are
            treated as fixed lifetime boundaries by compiled replay.
        sites : sequence[hashable] | None
            Complete logical site labels to arrange. If omitted, sites are
            inferred from first use in ``gate_stream`` unless ``L`` is given.
        L : int | None
            Convenience for ``sites=range(L)``.
        lattice_shape : pair of int, optional
            The ``(Lx, Ly)`` shape used by named geometric orders such as
            ``"snake"``, ``"folded-snake"``, and ``"hilbert"``. The product
            must equal the number of MPS sites.
        lattice_site : callable, optional
            Optional ``(x, y) -> logical_site`` mapper for named geometric
            orders. The default is ``x * Ly + y``.
        site_coords : mapping or sequence, optional
            Optional arbitrary logical-site coordinates. Unlike
            ``lattice_shape``, this accepts irregular data/ancilla layouts and
            contributes coordinate and snake candidates to the search.
        objective : {"locality", "compression"}
            ``"locality"`` minimizes support span and cut congestion using
            event weights. ``"compression"`` ranks layouts by operator-
            Schmidt load over the MPS cuts, with path span as a tie-breaker.
        order : str
            One of ``"quality"``/``"auto"``/``"best"``, ``"recursive"``,
            ``"lifetime"``/``"role-grouped"``, ``"input"``, ``"degree"``,
            ``"bfs"``, ``"spectral"``,
            ``"nevergrad"``, ``"kahypar"``, the geometric lattice presets
            ``"row-major"``, ``"col-major"``, ``"snake"``,
            ``"folded-snake"``, and ``"hilbert"``, or the ``"*_refined"``
            variants. Geometric presets require ``lattice_shape``.
        refine_passes : int
            Number of greedy adjacent-swap improvement passes.
        refine_numba : bool
            Use the optional numba polish kernel when numba is installed.
        spectral_dense_max : int
            Maximum site count for dense spectral ordering. ``"auto"`` falls
            back to non-spectral candidates above this size.
        recursive_dense_max : int
            Maximum site count for dense recursive spectral bisection.
        nevergrad_budget : int
            Black-box optimization budget for optional nevergrad candidates.
        nevergrad_seed : int | None
            NumPy seed used while constructing the optional nevergrad candidate.
        nevergrad_optimizer : str
            Name of the nevergrad optimizer class to use.
        from_scratch : bool
            Omit the original site order from the searched candidates and from
            Nevergrad inoculation. The original order remains available in
            ``input_stats`` as a comparison baseline.
        kahypar_config_path : path-like | None
            KaHyPar ``.ini`` config path. If omitted, ``PEPSY_KAHYPAR_CONFIG``
            is used. KaHyPar is skipped unless a config is supplied.
        kahypar_seed : int
            Seed forwarded to KaHyPar recursive bisection.
        weight_fn : callable | None
            Optional ``weight_fn(payload, support, event_type)`` override for
            per-event layout weights.
        weight_mode : {"auto", "count", "angle", "operator_schmidt"}
            Built-in event weighting heuristic. ``"auto"`` uses angle metadata
            when present, otherwise a cheap two-site operator-Schmidt proxy for
            small dense gates, falling back to count weights.
        schmidt_max_dim : int
            Maximum local dimension for the optional operator-Schmidt proxy.
        max_operator_qubits : int | None
            Maximum support size for exact dense rank probes in the
            compression objective. Larger or opaque operators use a
            conservative operator-space rank bound and are marked as bounded
            in the returned diagnostics.
        qubit_roles : mapping or sequence, optional
            Optional logical-site role metadata, for example
            ``{0: "data", 1: "ancilla"}``, or one role per site. It is
            carried into the plan and diagnostics. When both ``data`` and
            ``ancilla`` are present, the quality search automatically adds
            role-grouped, role-interleaved, and lifetime candidates.
        role_order : sequence[str] or {"data_first", "ancilla_first"}, optional
            If ``qubit_roles`` is supplied, add role-grouped and lifetime-aware
            candidates to the scored search. The connectivity objective still
            chooses the winner; roles never force one contiguous block.
        site_usage : plan entry
            The returned plan includes per-site first/last use, interaction
            counts, measure/reset boundaries, and reusable lifetime intervals.

        Returns
        -------
        dict
            Layout plan with ``qubit_inds``/``site_order``, ``layout``/
            ``site_map``, original ``where``, internal ``mapped_where``,
            ``stats``, and ``candidate_scores``.
        """

        finder = cls.LayoutFinder(
            gate_stream,
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
            qubit_roles=qubit_roles,
            site_coords=site_coords,
        )
        return finder.run(
            order=order,
            objective=objective,
            refine_passes=refine_passes,
            refine_numba=refine_numba,
            spectral_dense_max=spectral_dense_max,
            recursive_dense_max=recursive_dense_max,
            nevergrad_budget=nevergrad_budget,
            nevergrad_seed=nevergrad_seed,
            nevergrad_optimizer=nevergrad_optimizer,
            kahypar_config_path=kahypar_config_path,
            kahypar_seed=kahypar_seed,
            from_scratch=from_scratch,
            weight_fn=weight_fn,
            weight_mode=weight_mode,
            schmidt_max_dim=schmidt_max_dim,
            max_operator_qubits=max_operator_qubits,
            role_order=role_order,
        )

    @classmethod
    def find_gate_stream_layout(cls, gate_stream, **kwargs):
        """Alias for :meth:`gate_stream_layout`."""

        return cls.gate_stream_layout(gate_stream, **kwargs)

    @classmethod
    def gate_stream_schedule(
        cls,
        gate_stream,
        *,
        sites=None,
        L=None,
        qubit_roles=None,
        site_coords=None,
        layout_order="quality",
        schedule_order="mountain",
        layout_kwargs=None,
    ):
        """Compile a layout plus dependency-safe gate ordering.

        The returned schedule is directly consumable by
        :meth:`set_gate_schedule`. It contains physical locations and a
        position-to-logical-site ``site_order``. Ordinary gate entries and
        direct cap events are accepted; sub-MPO entries require an identity
        layout because their site tags are not relabeled here. Measurement,
        reset, and feed-forward events need a stateful control path and must
        remain outside this compiled schedule.
        """
        finder_kwargs, run_kwargs = cls._split_layout_finder_kwargs(
            layout_kwargs
        )
        if qubit_roles is not None:
            finder_kwargs.setdefault("qubit_roles", qubit_roles)
        if site_coords is not None:
            finder_kwargs.setdefault("site_coords", site_coords)
        finder = cls.LayoutFinder(
            gate_stream,
            sites=sites,
            L=L,
            **finder_kwargs,
        )
        plan = finder.run(order=layout_order, **run_kwargs)
        return finder.compile_schedule(plan, strategy=schedule_order)

    def layout_finder(
        self,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        qubit_roles=None,
        site_coords=None,
    ):
        """Return a layout finder for the currently queued gate stream."""

        return type(self).LayoutFinder.from_optimizer(
            self,
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
            qubit_roles=qubit_roles,
            site_coords=site_coords,
        )

    def current_gate_stream_layout(
        self,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        qubit_roles=None,
        site_coords=None,
        **kwargs,
    ):
        """Find a layout for the queued stream, optionally overriding roles."""

        return self.layout_finder(
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
            qubit_roles=qubit_roles,
            site_coords=site_coords,
        ).run(**kwargs)

    def current_gate_stream_schedule(
        self,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        qubit_roles=None,
        site_coords=None,
        layout_order="quality",
        schedule_order="mountain",
        layout_kwargs=None,
    ):
        """Compile the queued stream for :meth:`set_gate_schedule`.

        Direct cap events are fixed lifetime barriers: their physical
        positions are removed before later locations are emitted. Measurement,
        reset, and feed-forward events remain in the stateful :meth:`run` path.
        """
        return _layout_execution.current_gate_stream_schedule(
            self,
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
            qubit_roles=qubit_roles,
            site_coords=site_coords,
            layout_order=layout_order,
            schedule_order=schedule_order,
            layout_kwargs=layout_kwargs,
        )

    @staticmethod
    def _split_layout_finder_kwargs(layout_kwargs):
        """Split finder-construction options from per-run layout options."""
        kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
        finder_kwargs = {}
        for name in (
            "lattice_shape",
            "lattice_site",
            "qubit_roles",
            "site_coords",
        ):
            if name in kwargs:
                finder_kwargs[name] = kwargs.pop(name)
        return finder_kwargs, kwargs

    def select_layout_for_compression(
        self,
        *,
        sites=None,
        L=None,
        layout_kwargs=None,
        pilot_candidates=4,
        pilot_steps=None,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        run_kwargs=None,
    ):
        """Select an MPS layout using a bounded, state-aware pilot replay.

        The finder first produces cheap static candidates with
        ``objective="compression"``. The best ``pilot_candidates`` are then
        replayed on independent copies of the current MPS using the real
        execution mode, ``chi``, cutoff, and backend. The returned plan is
        non-mutating and contains ``pilot`` diagnostics for every candidate.

        This method is intentionally separate from :meth:`run`: layout
        selection can be expensive and should be explicit in production
        workflows. ``pilot_steps`` limits the replay prefix while preserving
        the original optimizer and gate queue.
        """
        return _layout_execution.select_layout_for_compression(
            self,
            sites=sites,
            L=L,
            layout_kwargs=layout_kwargs,
            pilot_candidates=pilot_candidates,
            pilot_steps=pilot_steps,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            run_kwargs=run_kwargs,
        )

    def plot_layout(
        self,
        plan=None,
        *,
        sites=None,
        L=None,
        layout_kwargs=None,
        **plot_kwargs,
    ):
        """Plot the current gate-stream layout and selected MPS order.

        This is a convenience wrapper around
        :meth:`MpsGateStreamLayoutFinder.plot`. It returns ``(fig, ax)`` and
        does not mutate the optimizer or install the plotted layout. When
        ``plan`` is omitted, the finder computes its default quality plan;
        pass ``layout_kwargs`` to customize that search.
        """
        finder_kwargs, run_kwargs = self._split_layout_finder_kwargs(layout_kwargs)
        finder = self.layout_finder(
            sites=sites,
            L=L,
            **finder_kwargs,
        )
        if plan is None:
            plan = finder.run(**run_kwargs)
        return finder.plot(plan, **plot_kwargs)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        p,
        gates=None,
        chi=None,
        mode="direct",
        contraction_opt="auto-hq",
        ind_id="k{}",
        inplace=False,
        _capture_initial=True,
        to_backend=None,
        qubit_roles=None,
    ):
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpsOptimizer(p, gates, chi) "
                    "or MpsOptimizer(p, chi) for an empty gate queue."
                )
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer.")

        self.inplace = bool(inplace)
        self.p = self._install_represented_norm(p if self.inplace else p.copy())
        self._qubit_roles = _normalize_site_roles(
            qubit_roles, range(int(getattr(self.p, "L", 0)))
        )
        # Dynamic cap streams shorten the live MPS during replay. Keep a
        # small structural ledger separate from norm/compression diagnostics so
        # callers can inspect the effective register length without inferring
        # it from tensor tags or a private event queue.
        self._initial_mps_length = int(getattr(self.p, "L", 0))
        self._mps_length_history = [self._initial_mps_length]
        self.cap_history = []
        # ``L_eff`` is an operation-active support ledger, not a dense-state
        # or Schmidt-rank measurement. A product MPS starts at zero; replay
        # events activate their current site interval and caps remap/remove
        # that support as the register shrinks.
        self._effective_active_positions = set()
        self._effective_length_history = [0]
        self._effective_site_history = [()]
        self._effective_event_history = []
        self._initial_p = self.p.copy() if _capture_initial else None
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        self._symbolic_gate_to_backend = to_backend
        # A normal optimizer owns its stream cache. Shot-created optimizers
        # explicitly opt into sharing the immutable plan cache after
        # construction; initialize both fields before installing the plan so
        # the constructor follows the same path as set_gates/add_gates.
        self._shared_backend_cache = False
        self._backend_cache_plan = None
        plan = _prepare_gate_stream(
            gates,
            to_backend=to_backend,
            backend_sample=self._state_backend_like_for(self.p),
        )
        normalized_queue = self._normalize_stream_plan_queue(plan)
        self._validate_normalized_gate_queue(normalized_queue)
        self._install_stream_plan(plan, normalized_queue=normalized_queue)
        self.chi = int(chi)
        mode_name = str(mode).strip().lower()
        self._dmrg_mode_alias = (
            mode_name if mode_name in self._DMRG_MODE_ALIASES else None
        )
        self._dmrg_mode_block_size = self._dmrg_alias_block_size(mode_name)
        self.mode = self._normalize_mode(mode_name)
        self._validate_canonical_boundary(self.p, self.mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.ind_id = str(ind_id)

        self.info_c = {}
        # Keep the Quimb-compatible dynamic view and the shared readout/layout
        # view synchronized. ``perm`` changes both after each gate; a
        # persistent layout freezes the same position -> logical-site map.
        self._set_site_order(range(int(getattr(self.p, "L", 0))))
        self._persistent_layout_plan = None
        self.layout_plan = None
        self.normalizations = []
        self.norm_events = []
        self._norm_summary_cache = None
        self._norm_log_survival = 0.0
        self._pending_zero_norm = None
        self.quality_checks = []
        self.last_layout_plan = self._persistent_layout_plan
        self.scheduled_layout_plan = None
        self.scheduled_site_order = None
        self.mix_history = []
        self.last_mix_summary = None
        self.last_run_timing = None
        self._timing_state = None
        self._fit_copy_policy_cache = None
        self._replay_rank_cache = None
        self._replay_array_cache = None
        # Optional diagnostics, disabled for normal replay in every mode.
        # run(finite_check=True) enables them only for that replay and warns.
        self._finite_check_enabled = False
        self._mix_dmrg_disabled_reason = None
        self._mix_dmrg_failed_sweep = None
        self._last_dmrg_fit_diagnostics = None
        self._dmrg1_one_site_locked = False
        # Native Symmray one-site writeback can retain duplicate fermionic
        # dummy modes when a product-state DMRG1 warm-up has just saturated
        # its bonds. Keep that narrow initialization case on two-site FIT;
        # non-product native states retain the documented DMRG1 schedule.
        self._dmrg1_native_product_two_site = (
            self._dmrg_mode_alias == "dmrg1"
            and self._is_native_fermionic_product_state(self.p)
        )
        self.measurements = []
        self._control_operator_cache = None
        self._rng = np.random.default_rng()
        self._unitary_previous_norm = None
        self.backend = None
        self.backend_dtype = None
        self.backend_device = None
        self.array_backend = None
        self.backend_info()
        self._init_canonicalization()

    def _info_for_state(self, p, info=None):
        """Return canonical metadata owned by ``p``.

        ``info_c`` describes the live optimizer state only. Diagnostic and
        target-building paths frequently work on MPS copies, for which using
        that dictionary would make a temporary state's center look like the
        live state's center. Such copies get an isolated metadata dictionary.
        """
        if info is not None:
            return info
        return self.info_c if p is self.p else {}

    def _current_orthog(self, p=None, *, info=None):
        """Return cached ``(min_site, max_site)`` orthogonality span.

        Cached entries may be ``"calc"`` / ``None`` (recompute), an ``int``,
        or a 1- or 2-tuple. The stored form is always a 2-tuple with
        ``min <= max``.
        """
        state = self.p if p is None else p
        state_info = self._info_for_state(state, info)
        cur = state_info.get("cur_orthog", "calc")
        if cur == "calc" or cur is None:
            lo, hi = state.calc_current_orthog_center()
            cur = (int(lo), int(hi))
        elif isinstance(cur, Integral):
            cur = (int(cur), int(cur))
        elif len(cur) == 1:
            cur = (int(cur[0]), int(cur[0]))
        elif len(cur) == 2:
            cur = (int(min(cur)), int(max(cur)))
        else:
            raise ValueError("cur_orthog must be an int, (int,), or (int, int).")

        state_info["cur_orthog"] = cur
        return cur

    def _record_orthog_span(self, p, where, *, info=None):
        """Record a span known to remain canonical after a state update."""
        state_info = self._info_for_state(p, info)
        state_info["cur_orthog"] = self._normalize_span(where)
        return state_info["cur_orthog"]

    def _format_ind(self, site):
        """Format a site id using ``self.ind_id``."""
        if isinstance(site, (tuple, list)):
            return self.ind_id.format(*site)
        return self.ind_id.format(site)

    @staticmethod
    def _infer_gate_dims(gate, where):
        """Infer physical dimensions from an explicit rank-2n gate tensor."""
        shape = getattr(gate, "shape", None)
        if shape is None:
            return None
        try:
            shape = tuple(int(d) for d in shape)
        except (TypeError, ValueError):
            return None
        nsites = len(where)
        if len(shape) != 2 * nsites:
            return None
        dims_in = shape[:nsites]
        dims_out = shape[nsites:]
        if dims_in != dims_out:
            return None
        return dims_in

    @staticmethod
    def _is_symmray_array(value):
        """Return whether ``value`` looks like a Symmray block-sparse array."""
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @classmethod
    def _has_symmray_data(cls, tn):
        """Return whether any tensor data in ``tn`` is Symmray-backed."""
        return any(
            cls._is_symmray_array(tensor.data)
            for tensor in getattr(tn, "tensors", ())
        )

    def _replay_has_symmray_data(self, tn):
        """Classify an owned network once during backend-preserving replay.

        Use the actual network as a weak key, not its first tensor's type:
        another network may contain mixed arrays. Outside replay, inspect
        the supplied object every time. No tensor values are checked here.
        """
        cache = self._replay_array_cache
        if cache is None:
            return self._has_symmray_data(tn)
        try:
            return cache[tn]
        except KeyError:
            result = self._has_symmray_data(tn)
            cache[tn] = result
            return result

    def _inherit_replay_array_kind(self, result, source):
        """Carry classification through an owned, backend-preserving copy."""
        if self._replay_array_cache is not None:
            self._replay_array_cache[result] = self._replay_has_symmray_data(source)
        return result

    def _invalidate_replay_metadata(self):
        """Forget cached facts after state replacement or structural changes."""
        for cache in (
            self._fit_copy_policy_cache,
            self._replay_rank_cache,
            self._replay_array_cache,
        ):
            if cache is not None:
                cache.clear()

    @classmethod
    def _is_native_fermionic_product_state(cls, p):
        """Return whether ``p`` is a native Symmray fermionic product MPS."""
        if not cls._has_symmray_data(p):
            return False
        is_fermionic = getattr(p, "isfermionic", None)
        if not callable(is_fermionic) or not is_fermionic():
            return False
        try:
            return cls._effective_max_bond(p) <= 1
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _mps_data_is_finite(p):
        """Return whether tensor data contains only finite values.

        All dense tensors or symmetry blocks are reduced to scalar booleans on
        their live backend, combined there, and copied to the host once. In
        particular, this neither materializes CuPy/Torch tensors on the host
        nor synchronizes once per MPS site.
        """

        def iter_arrays(data):
            """Yield dense leaves from one dense or block-sparse array."""
            blocks = getattr(data, "blocks", None)
            if blocks is not None:
                if isinstance(blocks, Mapping):
                    blocks = blocks.values()
                try:
                    for block in blocks:
                        yield from iter_arrays(block)
                    return
                except TypeError:
                    pass
            yield data

        checks = []
        for tensor in getattr(p, "tensors", ()):
            for data in iter_arrays(tensor.data):
                try:
                    checks.append(ar.do("all", ar.do("isfinite", data)))
                    continue
                except Exception:
                    pass

                try:
                    if not bool(np.all(np.isfinite(np.asarray(data)))):
                        return False
                except Exception:
                    return False

        if checks:
            try:
                combined = checks[0]
                for check in checks[1:]:
                    combined = ar.do("logical_and", combined, check)
                if not bool(ar.to_numpy(combined)):
                    return False
            except Exception:
                # Unknown backends may not implement scalar logical-and. The
                # supported NumPy/Torch/CuPy path above always has one host
                # conversion; retain a conservative compatibility fallback.
                if not all(bool(ar.to_numpy(check)) for check in checks):
                    return False
        exponent = getattr(p, "exponent", 0.0)
        try:
            return bool(np.isfinite(float(exponent)))
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _is_nearest_neighbor_1d(where):
        """Return whether an integer two-site location is adjacent in MPS order."""
        if len(where) != 2:
            return True
        site0, site1 = where
        if not isinstance(site0, Integral) or not isinstance(site1, Integral):
            return True
        return abs(int(site0) - int(site1)) == 1

    def _validate_symmray_mode_support(self):
        """Fail early for Symmray/MPS combinations with known bad paths."""
        # Block FIT grows only charge sectors generated by the effective target
        # and uses Symmray's native block SVD, so DMRG no longer needs Quimb's
        # dense-style global padding. Other supported modes already dispatch
        # through block-aware gate and split implementations.
        return

    @staticmethod
    def _symmray_structural_zero_cutoff(p, cutoff, cutoff_mode):
        """Turn an exact native split into exact-zero pruning.

        Symmray deliberately keeps every singular direction when
        ``cutoff == 0``. Routed fermionic swaps can consequently retain
        structural zero sectors whose duplicate like-dual dummy modes are not
        valid inputs to a later partial environment contraction. The smallest
        positive value representable by the block's real dtype, interpreted
        as an absolute cutoff, removes only exact zeros: Symmray keeps values
        greater than or equal to the cutoff, so every representable nonzero
        singular value is retained. No tensor is flattened or converted.
        """
        cutoff = float(cutoff)
        if cutoff != 0.0 or not p.isfermionic():
            return cutoff, cutoff_mode

        dtype_name = str(getattr(p, "dtype", "float64")).lower()
        if "bfloat16" in dtype_name:
            # NumPy has no portable bfloat16 scalar. bfloat16 has no
            # subnormal range, so its smallest positive normal is exact here.
            structural_cutoff = 1.1754943508222875e-38
        elif "16" in dtype_name:
            structural_cutoff = np.nextafter(
                np.float16(0.0), np.float16(1.0)
            ).item()
        elif "32" in dtype_name or "complex64" in dtype_name:
            structural_cutoff = np.nextafter(
                np.float32(0.0), np.float32(1.0)
            ).item()
        elif "longdouble" in dtype_name or "float128" in dtype_name:
            structural_cutoff = np.nextafter(
                np.longdouble(0.0), np.longdouble(1.0)
            ).item()
        else:
            structural_cutoff = np.nextafter(
                np.float64(0.0), np.float64(1.0)
            ).item()
        # Preserve extended-precision scalars: coercing the smallest positive
        # ``longdouble`` to Python's binary64 ``float`` can turn it back into
        # zero and silently disable structural-zero pruning.
        return structural_cutoff, "abs"

    @staticmethod
    def _native_needs_safe_qr(p):
        """Return whether native QR needs the low-precision phase guard."""
        dtype_name = str(getattr(p, "dtype", "")).lower()
        return "complex64" in dtype_name or "float32" in dtype_name

    @staticmethod
    def _native_canonize_bond(p, left, right):
        """Canonize one native bond without phase-normalizing QR diagonals."""
        qtn.tensor_canonize_bond(
            p[left],
            p[right],
            stabilized=False,
        )

    def _native_canonicalize_pair(self, p, where, *, info=None):
        """Safely move a native MPS center around a target pair."""
        if info is None:
            info = {}
        i, j = min(where), max(where)
        current = info.get("cur_orthog")
        if current == "calc":
            current = None

        if current is None:
            current = p.calc_current_orthog_center()

        if current is None:
            for site in range(0, i):
                self._native_canonize_bond(p, site, site + 1)
            for site in range(p.L - 1, j, -1):
                self._native_canonize_bond(p, site, site - 1)
            info["cur_orthog"] = (i, j)
            return

        if isinstance(current, Integral):
            cmin = cmax = int(current)
        else:
            cmin, cmax = min(current), max(current)

        if i > cmin:
            for site in range(cmin, i):
                self._native_canonize_bond(p, site, site + 1)
        else:
            i = min(j, cmin)

        if j < cmax:
            for site in range(cmax, j, -1):
                self._native_canonize_bond(p, site, site - 1)
        else:
            j = max(i, cmax)

        info["cur_orthog"] = (i, j)

    def _native_swap_site_to(self, p, site, target, *, info, compress_opts):
        """Move one native site with safe per-bond canonicalization."""
        if site == target:
            return
        if site < target:
            sites = range(site, target)
            absorb = "right"
        else:
            sites = range(site - 1, target - 1, -1)
            absorb = "left"
        swap_opts = dict(compress_opts)
        swap_opts.setdefault("absorb", absorb)
        for left in sites:
            right = left + 1
            self._native_canonicalize_pair(
                p,
                (left, right),
                info=info,
            )
            p.swap_sites_with_compress_(
                left,
                right,
                info=info,
                **swap_opts,
            )

    def _native_gate_with_auto_swap(
        self,
        p,
        gate,
        where,
        *,
        info,
        swap_back,
        **compress_opts,
    ):
        """Apply a native gate while avoiding unsafe complex64 QR phases."""
        i, j = where
        if i > j:
            i, j = j, i
            final_gate_where = (i + 1, i)
            absorb = "left"
        else:
            final_gate_where = (i, i + 1)
            absorb = "right"

        gate_opts = dict(compress_opts)
        gate_opts.setdefault("absorb", absorb)
        need_to_swap = i + 1 != j
        if need_to_swap:
            self._native_swap_site_to(
                p,
                j,
                i + 1,
                info=info,
                compress_opts=compress_opts,
            )

        self._native_canonicalize_pair(p, (i, i + 1), info=info)
        p.gate_split_(gate, where=final_gate_where, **gate_opts)
        info["cur_orthog"] = (i + 1, i + 1)

        if need_to_swap and swap_back:
            self._native_swap_site_to(
                p,
                i + 1,
                j,
                info=info,
                compress_opts=compress_opts,
            )

        return p

    def _apply_symmray_auto_swap_gate(
        self,
        p,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
        max_bond=None,
        info=None,
        swap_back=True,
        method=None,
        seed=None,
    ):
        """Apply a Symmray two-site gate through quimb's block-aware swaps."""
        cutoff, cutoff_mode = self._symmray_structural_zero_cutoff(
            p,
            cutoff,
            cutoff_mode,
        )
        compress_opts = {
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
        }
        if max_bond is not None:
            compress_opts["max_bond"] = max_bond
        if method is not None:
            compress_opts["method"] = method
        if seed is not None:
            compress_opts["seed"] = seed
        if info is None:
            info = self.info_c
        if not self._native_needs_safe_qr(p):
            p.gate_with_auto_swap_(
                gate,
                where,
                info=info,
                swap_back=swap_back,
                **compress_opts,
            )
            return p
        return self._native_gate_with_auto_swap(
            p,
            gate,
            where,
            info=info,
            swap_back=swap_back,
            **compress_opts,
        )

    def _warm_start_native_fermionic_fit(
        self,
        p,
        gates,
        wheres,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Open missing native charge sectors without copying ``p_target``.

        Dense FIT uses a disposable randomized guess when dense rank growth is
        needed. Symmray's graded tensors need Quimb's native auto-swap/SVD
        route to create compatible charge blocks, so retain this native-only
        preparation. It mutates
        the current working MPS through the gate algebra; it never transfers
        tensors from the exact target network into ``fit.p``.
        """
        if not (self._replay_has_symmray_data(p) and p.isfermionic()):
            return False
        sites = tuple(site for where in wheres for site in where)
        if max(sites) - min(sites) <= 1:
            return False
        for gate, where in zip(gates, wheres):
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
            else:
                self._apply_symmray_auto_swap_gate(
                    p,
                    gate,
                    where,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=self.chi,
                    info=self.info_c,
                )
        return True

    def _build_symmray_auto_swap_target(
        self,
        p,
        gate,
        where,
        cutoff,
        cutoff_mode,
        *,
        copy=True,
        info=None,
    ):
        """Build an un-chi-capped target using Symmray-aware swap routing."""
        p_target = p.copy() if copy else p
        self._apply_symmray_auto_swap_gate(
            p_target,
            gate,
            where,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info={} if info is None else info,
        )
        return p_target

    @staticmethod
    def _validate_fit_target_strategy(strategy):
        """Normalize the exact FIT target representation policy."""
        strategy = str(strategy).strip().lower()
        if strategy not in {"auto", "layered", "mps"}:
            raise ValueError(
                "fit_target_strategy must be 'auto', 'layered', or 'mps'."
            )
        return strategy

    @staticmethod
    def _validate_fit_init_strategy(strategy):
        """Normalize the FIT initial-guess construction policy."""
        strategy = str(strategy).strip().lower()
        if strategy.startswith("guess-"):
            # ``quimb-<method>`` is the mode spelling; accept the matching
            # hyphenated form for the FIT policy as a readable alias while
            # retaining the historical ``guess_<method>`` name.
            strategy = "guess_" + strategy[len("guess-") :]
        strategy = {"mpo": "svd_guess"}.get(strategy, strategy)
        if strategy not in _FIT_INIT_STRATEGIES:
            raise ValueError(
                "fit_init_strategy must be one of 'auto', 'direct', "
                "'random', 'random_expand', or 'guess-<method>'."
            )
        return strategy

    def _apply_layered_target_gate(
        self,
        target,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Append an exact spatially split gate to a disposable FIT target.

        The gate itself is SVD-factorized across its two sites, but it is not
        contracted into the MPS and no state bond is truncated. FIT can then
        contract this paper-style layered target lazily, avoiding the rapidly
        growing intermediate MPS ranks produced by repeated direct gates.
        """
        if len(where) == 1:
            return self._apply_gate(
                target,
                gate,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )

        if len(where) != 2:
            raise ValueError("A layered FIT target gate must act on one or two sites.")
        if self._replay_has_symmray_data(target) or target.isfermionic():
            raise ValueError(
                "fit_target_strategy='layered' is not available for Symmray/"
                "fermionic data; use 'auto' or 'mps' for native graded routing."
            )

        sites = tuple(int(site) for site in where)
        inds = tuple(self._format_ind(site) for site in sites)
        self._timed_call(
            "gate.apply",
            qtn.tensor_network_gate_inds,
            target,
            gate,
            inds,
            contract="split-gate",
            inplace=True,
            method="svd",
            absorb="both",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        # Quimb intentionally leaves lazy gate tensors untagged. Distinct
        # endpoint tags let FIT select each half exactly once, including when
        # several sequential gates share a physical index.
        for site, index in zip(sites, inds):
            tids = tuple(target.ind_map[index])
            if len(tids) != 1:
                raise ValueError(
                    f"Layered FIT target index {index!r} is not uniquely owned."
                )
            target.tensor_map[tids[0]].add_tag(target.site_tag_id.format(site))
        return target

    @staticmethod
    def _align_layered_submpo_tags(submpo, target, where):
        """Copy and align an explicit sub-MPO with the target MPS site tags.

        FIT permits multiple target tensors per site, but every target tensor
        must carry exactly one site tag. Explicit sub-MPO events can arrive
        with a different site-tag formatter (or after a persistent layout
        remap), so align the operator copy before attaching it lazily.
        """
        submpo = submpo.copy()
        site_tag = getattr(submpo, "site_tag", None)
        if not callable(site_tag):
            raise TypeError(
                "Layered DMRG sub-MPO targets require a site-tagged operator."
            )

        active_sites = tuple(sorted({int(site) for site in where}))
        target_tags = set()
        for site in active_sites:
            old_tag = site_tag(site)
            new_tag = target.site_tag(site)
            target_tags.add(new_tag)
            if old_tag != new_tag:
                submpo.retag_({old_tag: new_tag})

        for tensor in submpo.tensors:
            tensor_site_tags = tuple(tag for tag in tensor.tags if tag in target_tags)
            if len(tensor_site_tags) != 1:
                raise ValueError(
                    "Each layered DMRG sub-MPO tensor must carry exactly one "
                    f"target site tag, got {tuple(tensor.tags)!r}."
                )
        return submpo

    def _build_submpo_fit_target(
        self,
        p,
        submpo,
        where,
        target_cutoff,
        cutoff_mode,
        *,
        target_strategy,
    ):
        """Build an exact layered or materialized target for a sub-MPO event."""
        start, stop = min(where), max(where)
        target = self._inherit_replay_array_kind(p.copy(), p)
        target.canonicalize_((start, stop), info={})

        # Target representation and FIT initial guess are independent knobs:
        # the target must preserve the exact operator action, while the guess
        # only affects how quickly the variational solve finds that target.
        # Layering is safe for dense data because each operator tensor keeps
        # its site tag; native graded data must stay on the materialized route
        # so charge sectors and dummy-mode metadata are not discarded.
        layered_supported = not (
            self._replay_has_symmray_data(target)
            or target.isfermionic()
            or submpo.isfermionic()
        )
        if target_strategy == "layered" and layered_supported:
            aligned_submpo = self._align_layered_submpo_tags(
                submpo,
                target,
                where,
            )
            target.gate_with_op_lazy_(
                aligned_submpo,
                inplace=True,
                inplace_op=False,
            )
            return target, "layered"

        if target_strategy == "layered" and not layered_supported:
            raise ValueError(
                "Layered DMRG sub-MPO targets are not available for "
                "Symmray/fermionic data; use fit_target_strategy='auto' or 'mps'."
            )

        target.gate_with_submpo_(
            submpo,
            where=where,
            method="direct",
            max_bond=None,
            cutoff=target_cutoff,
            cutoff_mode=cutoff_mode,
            info={},
            inplace_mpo=False,
        )
        return target, "mps"

    def _prepare_submpo_fit_initial_guess(
        self,
        p,
        submpo,
        where,
        *,
        block_size,
        strategy,
        fit_mpo_guess,
        rand_strength,
        seed,
        cutoff,
        cutoff_mode,
    ):
        """Build a disposable FIT guess from an explicit sub-MPO event."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        info = {
            "enabled": False,
            "rand_strength": float(rand_strength),
            "bonds": [],
            "sites": [],
            "expanded": False,
            "reason": "direct",
        }
        result = {
            "fit_guess": p,
            "strategy": "direct",
            "requested_strategy": requested_strategy,
            "guess_method": None,
            "guess_used": False,
            "svd_guess_used": False,
            "random_initialization": info,
        }
        if self._replay_has_symmray_data(p) or p.isfermionic():
            info["reason"] = (
                "native_sector_growth"
                if int(block_size) in {2, 3}
                else "native_one_site_fit"
            )
            return result

        start, stop = self._normalize_span(where)
        needs_growth = requested_strategy in {"random", "random_expand"} and not (
            FIT._active_bonds_at_rank_targets(p, start, stop, self.chi)  # pylint: disable=protected-access
        )
        if requested_strategy == "auto":
            selected_strategy = _DEFAULT_FIT_INIT_STRATEGY
        else:
            selected_strategy = (
                requested_strategy
                if requested_strategy.startswith("guess_")
                or requested_strategy == "svd_guess"
                else requested_strategy if needs_growth else "direct"
            )
        if (
            not fit_mpo_guess
            and requested_strategy in {"auto", _DEFAULT_FIT_INIT_STRATEGY}
        ):
            selected_strategy = "direct"

        if selected_strategy == "svd_guess":
            guess_method = "direct"
        elif selected_strategy.startswith("guess_"):
            guess_method = selected_strategy[len("guess_") :]
        else:
            guess_method = None

        if guess_method is not None:
            # An explicit ``guess-*`` request is a warm-start policy, not a
            # request to grow rank. Apply it even after the active bonds reach
            # their attainable size; otherwise the one-site phase would use a
            # different initial state from the growth phase.
            fit_guess = self._copy_fit_window_state(p, where)
            opts = self._submpo_compress_opts(
                guess_method,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            if (
                guess_method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
                and _is_interior_submpo_span(fit_guess, where)
            ):
                _apply_submpo_with_interior_workaround(
                    fit_guess,
                    submpo,
                    where,
                    chi=self.chi,
                    method=guess_method,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    info={},
                    inplace_mpo=False,
                    optimize=opts.get("optimize"),
                    seed=seed,
                )
            else:
                quimb_seed = seed if guess_method in _MPO_METHODS_USE_SEED else None
                _run_seeded_quimb(
                    quimb_seed,
                    fit_guess.gate_with_submpo_,
                    submpo,
                    where=where,
                    method=guess_method,
                    max_bond=self.chi,
                    info={},
                    inplace_mpo=False,
                    **opts,
                )
            result["fit_guess"] = fit_guess
            result["strategy"] = (
                selected_strategy
                if selected_strategy != "svd_guess"
                else "svd_guess"
            )
            result["guess_method"] = guess_method
            result["guess_used"] = True
            result["svd_guess_used"] = True
            info["reason"] = selected_strategy
            return result

        fit_guess, random_info = (
            self._build_randomized_fit_guess(
                p,
                (start, stop),
                block_size=block_size,
                rand_strength=rand_strength,
                expand=selected_strategy == "random_expand",
                seed=int(seed),
            )
            if selected_strategy in {"random", "random_expand"}
            else (p, info)
        )
        if selected_strategy == "direct":
            info["reason"] = "already_at_target"
        result["fit_guess"] = fit_guess
        result["strategy"] = selected_strategy
        result["random_initialization"] = random_info
        return result

    def _apply_gate(self, p, gate, where, **kwargs):
        """Apply a gate using this optimizer's physical-index convention."""
        kwargs.setdefault("ind_id", self.ind_id)
        if self._timing_state is None:
            return apply_gate(p, gate, where, **kwargs)
        return self._timed_call("gate.apply", apply_gate, p, gate, where, **kwargs)

    def _init_canonicalization(self):
        """Initialize canonical form and orthogonality center."""
        if self.mode in _EXACT_MODES:
            # Exact evolution does not use canonical metadata.
            self.info_c = {}
            return
        self._validate_canonical_boundary(self.p, self.mode)
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    @staticmethod
    def _validate_canonical_boundary(p, mode):
        """Reject periodic states from open-chain canonical MPS modes.

        A periodic MPS cannot be reduced to an exact single-site mixed-
        canonical center: the omitted loop environment is not an identity.
        FIT already requires an open guess, and every optimizer mode that uses
        ``info_c`` and one-center norms must enforce the same boundary contract.
        Exact mode does not consume canonical metadata.
        """
        if mode not in _EXACT_MODES and bool(getattr(p, "cyclic", False)):
            raise ValueError(
                "MpsOptimizer canonical modes require an open-boundary MPS; "
                "cyclic MPS data do not have an exact one-tensor canonical norm."
            )

    def _prepare_dmrg_state(self):
        """Prepare DMRG without globally padding every MPS bond.

        Two- and three-site FIT discover rank on visited bonds through native
        SVD splits. One-site compatibility runs expand only their active gate
        range immediately before fitting. Avoiding eager global padding removes
        an ``O(L * chi**2)`` memory cost on long, initially low-rank states.
        """
        self._ensure_tracked_center()

    def _prepare_one_site_dmrg_state(self, where):
        """Prepare active bond support for ordinary one-site DMRG.

        FIT only optimizes the interval spanned by the gate. Expanding every
        bond in a long MPS would waste ``O(L * chi**2)`` memory, so only the
        active internal indices are padded. Mixed mode deliberately bypasses
        this helper because its disposable direct-compressed guess owns rank
        growth. Native Symmray callers use two- or three-site FIT instead of
        this dense-style expansion path.
        """
        if self.chi <= 1 or getattr(self.p, "L", 0) <= 1:
            return

        xmin, xmax = min(where), max(where)
        if xmin == xmax:
            return
        target_sizes = self._mix_target_bond_dimensions()
        bonds_to_expand = [
            site
            for site in range(xmin, xmax)
            if int(self.p.bond_size(site, site + 1)) < target_sizes[site]
        ]
        if bonds_to_expand:
            if self._replay_has_symmray_data(self.p):
                raise ValueError(
                    "One-site FIT cannot pad native Symmray bonds safely; use "
                    "fit_block_size=2 or 3 so the native block SVD grows only "
                    "charge sectors present in the effective target."
                )
            by_target = {}
            for site in bonds_to_expand:
                by_target.setdefault(target_sizes[site], []).append(site)
            for target, sites in by_target.items():
                bond_inds = [self.p.bond(site, site + 1) for site in sites]
                # MatrixProductState overrides this method without exposing
                # ``inds_to_expand``. Calling the public TensorNetwork method
                # retains the MPS object while selecting only these bonds.
                qtn.TensorNetwork.expand_bond_dimension(
                    self.p,
                    int(target),
                    inds_to_expand=bond_inds,
                    inplace=True,
                )
            self._init_canonicalization()

    def _prepare_fit_window(self, where, *, block_size):
        """Prepare a FIT window without pre-expanding adaptive updates.

        Native two- and three-site FIT updates receive the current MPS bond
        dimensions unchanged. Their direction-aware SVD splits grow only the
        visited bonds, up to ``chi``. Ordinary one-site compatibility FIT may
        pre-size active bonds. Mixed one-site FIT instead receives its bond
        support from the disposable ``guess-direct`` state.
        """
        if int(block_size) == 1 and self.mode != "mix":
            self._prepare_one_site_dmrg_state(where)

    def _dmrg_fit_block_size(self, p, where, requested_block_size):
        """Resolve the live DMRG block size for an active window.

        DMRG1 uses two-site updates only during its bounded warm-up. Once the
        optimizer has latched the full-chain one-site phase, or the active
        bonds already have no attainable rank growth left, fixed-rank one-site
        sweeps are the correct and cheaper update.
        """
        xmin, xmax = self._normalize_span(where)
        active_block_size = min(
            int(requested_block_size),
            xmax - xmin + 1,
        )
        if self._dmrg_mode_alias == "dmrg1" and active_block_size == 2:
            if self._dmrg1_native_product_two_site:
                return active_block_size
            if self._dmrg1_one_site_locked:
                return 1
            if (
                xmax - xmin >= 2
                and FIT._active_bonds_at_rank_targets(  # pylint: disable=protected-access
                    p,
                    xmin,
                    xmax,
                    self.chi,
                )
            ):
                return 1
        return active_block_size

    def _dmrg1_all_bonds_at_rank_targets(self):
        """Return whether every MPS bond has reached its physical ceiling."""
        if self._dmrg_mode_alias != "dmrg1":
            return False
        target_sizes = self._mix_target_bond_dimensions()
        if not target_sizes:
            return True
        return all(
            int(self.p.bond_size(site, site + 1)) >= int(target)
            for site, target in enumerate(target_sizes)
        )

    def _maybe_lock_dmrg1_one_site_phase(self):
        """Latch DMRG1 into one-site updates after full-chain saturation."""
        if self._dmrg_mode_alias != "dmrg1":
            return False
        if not self._dmrg1_one_site_locked and self._dmrg1_all_bonds_at_rank_targets():
            self._dmrg1_one_site_locked = True
        return self._dmrg1_one_site_locked

    def _validate_dmrg1_iteration_budget(self, p, where, *, n_iter, block_size):
        """Require two growth sweeps plus refinement for uncapped DMRG1."""
        if self._dmrg_mode_alias != "dmrg1" or int(block_size) != 2:
            return
        # Three sweeps suffice at every rank. The default budget of eight
        # needs no tensor metadata inspection just to validate this minimum.
        if int(n_iter) >= 3:
            return
        xmin, xmax = self._normalize_span(where)
        if xmax - xmin < 2:
            return
        if FIT._active_bonds_at_rank_targets(  # pylint: disable=protected-access
            p,
            xmin,
            xmax,
            self.chi,
        ):
            return
        if int(n_iter) < 3:
            raise ValueError(
                "mode='dmrg1' requires n_iter >= 3 for an under-capacity "
                "window: two two-site growth sweeps and at least one "
                "one-site refinement sweep."
            )

    def set_p(self, p):
        """Assign a new state and reset state-dependent optimizer metadata.

        Replacing the represented state starts a new working-norm interval. In
        particular, a retained norm from the previous state must never become
        the unitary compression target for the replacement.
        """
        # Reject an incompatible caller-owned object before ``inplace=True``
        # can install any optimizer-local methods or metadata on it.
        self._validate_canonical_boundary(p, self.mode)
        new_p = self._install_represented_norm(p if self.inplace else p.copy())
        # Validate before replacing the live state so a mixed-backend input
        # cannot leave this optimizer half-updated after a failed assignment.
        self._state_backend_info_for(new_p)
        self._validate_normalized_gate_queue(
            (self.G, self.where, self.event_types),
            state=new_p,
        )
        self.p = new_p
        self._invalidate_replay_metadata()
        self._initial_p = self.p.copy()
        self._initial_mps_length = int(getattr(self.p, "L", 0))
        self._mps_length_history = [self._initial_mps_length]
        self.cap_history = []
        self._effective_active_positions = set()
        self._effective_length_history = [0]
        self._effective_site_history = [()]
        self._effective_event_history = []
        self._unitary_previous_norm = None
        self.norm_events = []
        self._norm_summary_cache = None
        self._norm_log_survival = 0.0
        self._pending_zero_norm = None
        self._dmrg1_one_site_locked = False
        self._dmrg1_native_product_two_site = (
            self._dmrg_mode_alias == "dmrg1"
            and self._is_native_fermionic_product_state(self.p)
        )
        self._set_site_order(range(int(getattr(self.p, "L", 0))))
        self._persistent_layout_plan = None
        self.layout_plan = None
        self.last_layout_plan = None
        self.backend_info()
        self._init_canonicalization()

    def sync_canonicalization(self, site=None):
        """Re-establish tracked canonical metadata after external MPS access.

        Quimb's canonical readout helpers, such as
        ``local_expectation_canonical``, move the live MPS orthogonality centre
        in place.  Internal Pepsy paths pass ``info_c`` and stay synchronized,
        but direct calls through :attr:`p` cannot update this optimizer's
        metadata.  Call this method before resuming a canonical-mode replay
        after such an external mutation.  It performs an explicit centre
        discovery, canonicalizes to a single site, and records the resulting
        ``info_c['cur_orthog']``.

        Post-run diagnostic readout should normally use ``p.copy()`` instead;
        this method is the recovery path when the live state was intentionally
        inspected or modified.

        Parameters
        ----------
        site : int, optional
            Site at which to leave the one-site canonical centre.  If omitted,
            the upper endpoint of Quimb's discovered canonical span is used.

        Returns
        -------
        tuple[int, int]
            The synchronized one-site canonical span.
        """
        if self.mode in _EXACT_MODES:
            raise ValueError(
                "sync_canonicalization requires a canonical MPS mode; "
                f"mode={self.mode!r} does not track info_c."
            )
        if not hasattr(self.p, "calc_current_orthog_center"):
            raise TypeError("the live state does not expose MPS canonical metadata.")

        self._invalidate_replay_metadata()
        current = self._normalize_span(self.p.calc_current_orthog_center())
        if site is None:
            site = current[1]
        site = int(site)
        if not 0 <= site < int(self.p.L):
            raise ValueError(
                f"site must lie in [0, {int(self.p.L)}), got {site}."
            )

        self.p.canonize(
            [site],
            cur_orthog=current,
            info=self.info_c,
        )
        self.info_c["cur_orthog"] = (site, site)
        return self.info_c["cur_orthog"]

    def normalize(self, eps=1e-15, insert=None):
        """Normalize current ``self.p`` in-place.

        Parameters
        ----------
        eps : float, default=1e-15
            Precision used by Quimb's general normalization path in exact
            mode. Canonical open-MPS modes use their tracked
            one-site center directly.
        insert : int | None, default=None
            Optional site where the normalization factor is inserted.

        Returns
        -------
        float | complex
            Previous raw ``self.p.H @ self.p`` value. Canonical open-MPS modes
            derive it from the tracked center; exact mode
            use Quimb's general normalization implementation. The removed norm
            factor is accumulated into ``self.p.exponent`` when present, so
            ``self.p.norm()`` continues to report the represented norm while
            the raw data norm becomes one.
        """
        return _norm.normalize(self, eps, insert)

    def entropy(self, cut=None, *, method="svd"):
        """Return normalized base-2 entropy across one MPS bond.

        The measurement is performed by
        :func:`pepsy.tensors.mps_entanglement_entropy` on a private copy, so
        the optimizer's live tensors, exponent, and canonical metadata are
        preserved.
        """
        return _mps_entanglement_entropy(self.p, cut=cut, method=method)

    def entanglement_entropy(self, cut=None, *, method="svd"):
        """Alias for :meth:`entropy` with an explicit diagnostic name."""
        return self.entropy(cut=cut, method=method)

    def _copy_impl(self, *, capture_initial):
        """Copy optimizer state, optionally retaining a shot-replay template."""
        history_copy = deepcopy if capture_initial else list
        trusted = not capture_initial and self.mode not in _EXACT_MODES and self._fit_window_copy_supported(self.p)
        if trusted:
            # Owned arrays preserve the existing isometries; no constructor,
            # recanonicalization, or discovery scan is required for this clone.
            copied = object.__new__(type(self))
            copied.__dict__ = self.__dict__.copy()
            copied.p = self._copy_fit_window_state(self.p, (0, self.p.L - 1))
            copied.info_c = deepcopy(self.info_c)
            copied._rng = np.random.default_rng()
            copied._timing_state = None
        else:
            copied = type(self)(
                self.p.copy(), gates=[], chi=self.chi, mode=self.mode,
                contraction_opt=self.contraction_opt, ind_id=self.ind_id,
                inplace=True,
                _capture_initial=False, to_backend=self._symbolic_gate_to_backend,
                qubit_roles=self._qubit_roles,
            )
        copied._dmrg_mode_block_size = self._dmrg_mode_block_size
        copied._dmrg_mode_alias = self._dmrg_mode_alias
        copied._dmrg1_native_product_two_site = (
            self._dmrg1_native_product_two_site
        )
        # ``MatrixProductState.copy()`` does not promise to preserve the
        # physical orthogonality centre. The constructor canonicalizes the
        # copied state, so its freshly initialized ``info_c`` is authoritative
        # here. Overwriting it with the source cache can claim that site 0 is
        # canonical while the copied tensors are centered at site ``L // 2``;
        # a subsequent projective replay can then lose the branch norm.
        if not trusted and copied.mode not in _EXACT_MODES:
            copied.info_c["cur_orthog"] = tuple(
                int(site) for site in copied.p.calc_current_orthog_center()
            )
        else:
            copied.info_c = deepcopy(self.info_c)
        copied.inplace = self.inplace
        # Coalesced trajectory branches are copied immediately before an
        # ordinary replay and never become nested shot runners. Avoid a
        # second MPS copy for that internal path; public ``copy()`` retains
        # the constructor-state template needed by ``run(shots=...)``.
        copied._initial_p = self.p.copy() if capture_initial else None
        copied._initial_mps_length = self._initial_mps_length
        copied._mps_length_history = list(self._mps_length_history)
        copied.cap_history = history_copy(self.cap_history)
        copied._effective_active_positions = set(self._effective_active_positions)
        copied._effective_length_history = list(self._effective_length_history)
        copied._effective_site_history = list(self._effective_site_history)
        copied._effective_event_history = history_copy(self._effective_event_history)
        copied._stream_plan = self._stream_plan
        copied._gate_stream = tuple(self._gate_stream)
        copied._has_trajectory_events = self._has_trajectory_events
        copied._shared_backend_cache = self._shared_backend_cache
        copied._backend_cache_plan = self._backend_cache_plan
        copied.G = list(self.G)
        copied.where = list(self.where)
        copied.event_types = list(self.event_types)
        copied._set_site_order(self.qubits)
        copied._persistent_layout_plan = deepcopy(self._persistent_layout_plan)
        copied.layout_plan = deepcopy(self.layout_plan)
        copied.last_layout_plan = deepcopy(self.last_layout_plan)
        copied.scheduled_layout_plan = deepcopy(self.scheduled_layout_plan)
        copied.scheduled_site_order = deepcopy(self.scheduled_site_order)
        copied.normalizations = history_copy(self.normalizations)
        copied.norm_events = history_copy(self.norm_events)
        copied._norm_log_survival = self._norm_log_survival
        copied._pending_zero_norm = self._pending_zero_norm
        copied.quality_checks = deepcopy(self.quality_checks)
        copied.mix_history = deepcopy(self.mix_history)
        copied.last_mix_summary = deepcopy(self.last_mix_summary)
        copied.last_run_timing = deepcopy(self.last_run_timing)
        copied._fit_copy_policy_cache = None
        copied._replay_rank_cache = None
        copied._replay_array_cache = None
        copied._finite_check_enabled = False
        copied._norm_summary_cache = None
        copied._mix_dmrg_disabled_reason = self._mix_dmrg_disabled_reason
        copied._mix_dmrg_failed_sweep = self._mix_dmrg_failed_sweep
        copied._dmrg1_one_site_locked = self._dmrg1_one_site_locked
        copied._last_dmrg_fit_diagnostics = deepcopy(
            self._last_dmrg_fit_diagnostics
        )
        copied.measurements = deepcopy(self.measurements)
        copied._unitary_previous_norm = self._unitary_previous_norm
        copied._trajectory_diagnostics = deepcopy(
            getattr(self, "_trajectory_diagnostics", None)
        )
        copied._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        if not capture_initial:
            # Internal replay only appends committed records. Isolate their
            # mutable public representation once, when publishing final leaves.
            copied._branch_shared_histories = True
        return copied

    _BRANCH_HISTORY_NAMES = (
        "cap_history", "_effective_event_history", "normalizations", "norm_events",
    )

    def _detach_branch_histories(self):
        """Give a published trajectory leaf independent mutable history records."""
        if getattr(self, "_branch_shared_histories", False):
            for name in self._BRANCH_HISTORY_NAMES:
                setattr(self, name, deepcopy(getattr(self, name)))
            self._branch_shared_histories = False

    def _copy_for_trajectory_branch(self):
        """Return a branch copy without an unused nested-shot snapshot."""
        return self._copy_impl(capture_initial=False)

    def copy(self) -> "MpsOptimizer":
        """Return an independent optimizer copy at its current MPS state.

        The copied optimizer owns a deep copy of the represented MPS and an
        independent canonical-centre cache. Queue entries are intentionally
        retained (without copying immutable gate payloads), so callers can
        continue a partially prepared replay independently. The copy also
        snapshots its current state for a later ``run(shots=...)`` call.
        """
        return self._copy_impl(capture_initial=True)

    def set_mode(self, mode):
        """Switch optimization mode while preserving the represented state."""
        old_mode = self.mode
        old_dmrg_alias = self._dmrg_mode_alias
        mode_name = str(mode).strip().lower()
        new_dmrg_alias = (
            mode_name if mode_name in self._DMRG_MODE_ALIASES else None
        )
        new_dmrg_block_size = self._dmrg_alias_block_size(mode_name)
        new_mode = self._normalize_mode(mode_name)
        if new_mode in _EXACT_MODES and self._persistent_layout_plan is not None:
            raise ValueError(
                f"cannot switch a persistent-layout optimizer to mode={new_mode!r}; "
                "read out the logical state or create a new optimizer."
            )
        self._validate_canonical_boundary(self.p, new_mode)
        self._invalidate_replay_metadata()
        old_perm_mode = old_mode == "perm"
        new_perm_mode = new_mode == "perm"
        if old_perm_mode and not new_perm_mode:
            # Other modes interpret integer ``where`` values as physical MPS
            # positions, so restore the logical ordering before switching.
            self._restore_permutation()
        elif not old_perm_mode and new_perm_mode:
            if self._persistent_layout_plan is not None:
                raise ValueError(
                    "cannot switch a persistent layout into mode='perm'; "
                    "use the persistent layout mapping for replay instead."
                )
            if old_mode in _EXACT_MODES:
                # Exact replay stores a contracted TensorNetwork rather than
                # an MPS, so it has no physical length from which to seed the
                # logical-to-physical permutation. Rebuild it before creating
                # the permutation bookkeeping below.
                self._ensure_mps_state()
            self._set_site_order(range(int(getattr(self.p, "L", 0))))
        if new_mode in _EXACT_MODES:
            # Exact contractions do not consume canonical metadata. Discard
            # the MPS-only cache so it cannot be mistaken for the contracted
            # TensorNetwork's state.
            self.info_c = {}
        self.mode = new_mode
        if old_mode != new_mode or old_dmrg_alias != new_dmrg_alias:
            self._last_dmrg_fit_diagnostics = None
        self._dmrg_mode_alias = new_dmrg_alias
        self._dmrg_mode_block_size = new_dmrg_block_size
        self._dmrg1_native_product_two_site = (
            self._dmrg_mode_alias == "dmrg1"
            and self._is_native_fermionic_product_state(self.p)
        )
        if old_mode != new_mode or old_dmrg_alias != new_dmrg_alias:
            self._dmrg1_one_site_locked = False
        if old_mode in _EXACT_MODES and self.mode not in _EXACT_MODES:
            # Exact mode stores a fully contracted TensorNetwork, so rebuild an
            # MPS before recreating canonical metadata for an MPS mode.
            self._ensure_mps_state()
            self._init_canonicalization()
        return self

    _restore_permutation = _layout_execution._restore_permutation

    def restore_qubit_order(self):
        """Restore ``p`` to logical site order and return the managed state."""
        self._restore_permutation()
        return self.p

    _set_site_order = _layout_execution._set_site_order

    _logical_to_physical_where = _layout_execution._logical_to_physical_where

    _record_permutation_move = _layout_execution._record_permutation_move

    _update_permutation_after_cap = _layout_execution._update_permutation_after_cap

    def logical_site(self, position):
        """Return the logical site currently stored at physical ``position``."""
        position = int(position)
        if not 0 <= position < len(self.logical_order):
            raise IndexError(
                f"physical position {position} is outside the MPS range "
                f"[0, {len(self.logical_order)})."
            )
        return int(self.logical_order[position])

    def position(self, site):
        """Return the physical position currently holding logical ``site``."""
        site = int(site)
        try:
            return int(self.logical_order.index(site))
        except ValueError as exc:
            raise ValueError(
                f"logical site {site} is not present in the current order "
                f"{self.logical_order!r}."
            ) from exc

    def remap_sample(self, config):
        """Remap a physical-order sample/configuration into logical order.

        ``config`` can be a length-``L`` vector or a batch with ``L`` as its
        final dimension. The returned NumPy array has logical site ``i`` at
        index ``i``.
        """
        return _layout_execution.remap_sample(self, config)

    def to_dense(self, logical_order=True, **kwargs):
        """Return the statevector with optional logical-site axis ordering.

        With ``logical_order=True`` (the default), axes are ordered by logical
        site labels even when the managed MPS is stored in a persistent layout.
        ``logical_order=False`` returns the underlying physical MPS ordering.
        """
        return _layout_execution.to_dense(self, logical_order, **kwargs)

    @property
    def gate_stream(self):
        """Return the snapshotted raw stream, including trajectory events."""
        return self._gate_stream

    @property
    def qubit_roles(self):
        """Return a copy of optional logical-site role metadata."""
        return dict(self._qubit_roles)

    @property
    def site_roles(self):
        """Alias for :attr:`qubit_roles` used by layout-oriented callers."""
        return self.qubit_roles

    def gate_stream_info(self):
        """Return stable stream and role metadata for layout tooling."""
        supports = tuple(
            _normalize_layout_support(where) for where in self.where
        )
        return {
            "sites": tuple(range(int(self._initial_mps_length))),
            "event_count": len(self._gate_stream),
            "event_types": tuple(self.event_types),
            "qubit_roles": self.qubit_roles,
            "site_usage": _gate_stream_site_usage(
                range(int(self._initial_mps_length)),
                supports,
                self.event_types,
                site_roles=self._qubit_roles,
            ),
            "has_trajectory_events": self.has_trajectory_events,
        }

    @property
    def allocated_length(self) -> int:
        """Return the current allocated MPS register length."""
        return int(getattr(self.p, "L", self._mps_length_history[-1]))

    @property
    def effective_active_sites(self) -> tuple[int, ...]:
        """Return current operation-active MPS positions."""
        L = self.allocated_length
        return tuple(
            int(position)
            for position in sorted(self._effective_active_positions)
            if 0 <= position < L
        )

    def _record_effective_event(self, where, *, event_type="gate"):
        """Record active support without inspecting tensor ranks or Schmidt values."""
        if event_type == "cap":
            raise ValueError("cap support must be remapped with _apply_effective_cap")
        positions = tuple(int(site) for site in where)
        previous_size = len(self._effective_active_positions)
        if positions:
            left, right = min(positions), max(positions)
            self._effective_active_positions.update(range(left, right + 1))
        active = (
            self._effective_site_history[-1]
            if len(self._effective_active_positions) == previous_size
            else self.effective_active_sites
        )
        self._effective_length_history.append(len(active))
        self._effective_site_history.append(active)
        self._effective_event_history.append(
            {
                "event_type": str(event_type),
                "where": tuple(positions),
                "L_eff": int(len(active)),
                "active_sites": active,
            }
        )

    def _apply_effective_cap(self, position):
        """Remap operation-active positions after a structural cap."""
        position = int(position)
        self._effective_active_positions = {
            active_position - (active_position > position)
            for active_position in self._effective_active_positions
            if active_position != position
        }
        active = self.effective_active_sites
        self._effective_length_history.append(len(active))
        self._effective_site_history.append(active)
        self._effective_event_history.append(
            {
                "event_type": "cap",
                "where": (position,),
                "L_eff": int(len(active)),
                "active_sites": active,
            }
        )

    @property
    def L_eff(self) -> int:
        """Return operation-active support length, with product start equal to zero.

        This is intentionally a lightweight replay ledger. It estimates the
        MPS support made active by the gate stream and reduced by caps; it does
        not compute Schmidt values, bond entropies, or dense-state ranks.
        """
        return len(self.effective_active_sites)

    @property
    def effective_mps_length(self) -> int:
        """Descriptive alias for :attr:`L_eff`."""
        return self.L_eff

    def mps_length_diagnostics(self):
        """Return allocated-register and operation-active length diagnostics.

        ``length_history`` is the allocated MPS register after construction
        and successful caps. ``L_eff`` and ``effective_length_history`` are a
        separate operation-active support ledger: they start at zero, grow
        when gate sites/intervals are replayed, and shrink when caps remove
        those positions. No Schmidt-rank or SVD inspection is performed.
        """
        history = tuple(int(length) for length in self._mps_length_history)
        effective_history = tuple(
            int(length) for length in self._effective_length_history
        )
        return {
            "initial_length": int(self._initial_mps_length),
            "peak_length": int(max(history, default=0)),
            "minimum_length": int(min(history, default=0)),
            "L_eff": int(self.L_eff),
            "effective_length": int(self.L_eff),
            "removed_sites": int(self._initial_mps_length - self.allocated_length),
            "caps": len(self.cap_history),
            "length_history": history,
            "cap_events": deepcopy(self.cap_history),
            "allocated_length": int(self.allocated_length),
            "allocated_length_history": history,
            "initial_effective_length": 0,
            "peak_effective_length": int(max(effective_history, default=0)),
            "minimum_effective_length": int(min(effective_history, default=0)),
            "effective_length_history": effective_history,
            "L_eff_history": effective_history,
            "effective_active_sites": self.effective_active_sites,
            "effective_site_history": deepcopy(self._effective_site_history),
            "effective_event_history": deepcopy(self._effective_event_history),
            "effective_length_model": "active-operation-envelope",
        }

    @property
    def has_trajectory_events(self):
        """Whether this optimizer owns a stream requiring shot replay."""
        return bool(self._has_trajectory_events)

    @staticmethod
    def _normalize_stream_plan_queue(plan):
        """Normalize a plan once for validation and single-state replay."""
        if plan.has_trajectory_events:
            return [], [], []
        return _normalize_gate_queue(plan.entries)

    def _validate_normalized_gate_queue(self, normalized_queue, *, state=None):
        """Validate an already-normalized queue without normalizing again."""
        gates, _wheres, event_types = normalized_queue
        self._validate_gate_stream_backend(gates, event_types, state=state)

    def _install_stream_plan(self, plan, *, normalized_queue=None):
        """Install a compiled plan and its already-normalized replay queue."""
        if not isinstance(plan, _MpsStreamPlan):
            raise TypeError("plan must be an internal MPS stream plan.")
        if normalized_queue is None:
            normalized_queue = self._normalize_stream_plan_queue(plan)
        self._stream_plan = plan
        self._gate_stream = plan.entries
        self._has_trajectory_events = plan.has_trajectory_events
        if not self._shared_backend_cache:
            self._backend_cache_plan = plan
        if self._has_trajectory_events:
            # Stochastic and stateful leakage entries are consumed by the shot
            # runner, which lowers each sampled branch into an ordinary stream.
            # They cannot be normalized into the single-state queue here.
            self.G, self.where, self.event_types = [], [], []
        else:
            self.G, self.where, self.event_types = normalized_queue

    def _shot_factory(self):
        """Build fresh optimizers from this instance's initial state."""
        template = self._initial_p
        mode = self._dmrg_mode_alias or self.mode
        stream = self._gate_stream
        constructor = {
            "chi": self.chi,
            "mode": mode,
            "contraction_opt": self.contraction_opt,
            "ind_id": self.ind_id,
            "to_backend": self._symbolic_gate_to_backend,
            "qubit_roles": self._qubit_roles,
        }

        def make_optimizer():
            options = dict(constructor)
            options["inplace"] = True
            options["_capture_initial"] = False
            optimizer = type(self)(template.copy(), [], **options)
            optimizer._stream_plan = self._stream_plan
            optimizer._gate_stream = stream
            optimizer._has_trajectory_events = self._has_trajectory_events
            optimizer._shared_backend_cache = True
            optimizer._backend_cache_plan = self._backend_cache_plan
            if self._has_trajectory_events:
                optimizer.G, optimizer.where, optimizer.event_types = [], [], []
            else:
                optimizer.G = list(self.G)
                optimizer.where = list(self.where)
                optimizer.event_types = list(self.event_types)
            if self._persistent_layout_plan is not None:
                # The shot template is already in the frozen physical order.
                # Install only the logical mapping on the child; calling
                # apply_layout again would reorder the state a second time.
                optimizer._persistent_layout_plan = deepcopy(
                    self._persistent_layout_plan
                )
                optimizer.layout_plan = deepcopy(self.layout_plan)
                optimizer.last_layout_plan = deepcopy(self.last_layout_plan)
                # The child starts from the same frozen physical order; keep
                # both public mapping views consistent without reordering it.
                optimizer._set_site_order(self.qubits)
            optimizer.scheduled_layout_plan = deepcopy(self.scheduled_layout_plan)
            optimizer.scheduled_site_order = deepcopy(self.scheduled_site_order)
            return optimizer

        return make_optimizer

    @staticmethod
    def _shot_runner_requested(
        shots,
        *,
        has_trajectory_events,
        error_model,
        strategy,
        run_kwargs,
        max_branches,
        importance_sampling,
        max_branch_factor,
        parallel_workers,
        parallel_backend,
        auto_max_expected_faults,
        retain,
        mpi=None,
        workers="auto",
        checkpoint_path=None,
        observable=None,
    ):
        """Return whether ``run`` needs the multi-shot trajectory machinery."""
        if mpi is not None and mpi is not False:
            return True
        if checkpoint_path is not None or observable is not None:
            return True
        if has_trajectory_events or error_model is not None:
            return True
        if isinstance(shots, bool) or not isinstance(shots, Integral):
            return True
        if int(shots) != 1:
            return True
        if workers not in {None, "auto"}:
            return True
        return any(
            (
                strategy != "auto",
                run_kwargs is not None,
                max_branches != _SHOT_DEFAULT_MAX_BRANCHES,
                importance_sampling is not None,
                max_branch_factor is not None,
                parallel_workers != 1,
                parallel_backend != "thread",
                auto_max_expected_faults
                != _SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS,
                retain != "all",
            )
        )

    def _validate_shot_compatibility(self, error_model=None):
        """Reject known-invalid mode and trajectory combinations early."""
        from ..noise import (  # pylint: disable=import-outside-toplevel
            _has_unforced_branching_control,
            _leakage_event_parts,
        )

        entries = self._gate_stream
        controls = tuple(
            entry for entry in entries if self.control_event_parts(entry) is not None
        )
        has_leakage = any(_leakage_event_parts(entry) is not None for entry in entries)
        has_submpo = any(_is_submpo_event(entry) for entry in entries)
        if has_submpo and not self._is_mpo_mode(self.mode):
            raise ValueError(
                "shot replay of sub-MPO events requires mode='direct' "
                "or another Quimb compression mode; "
                f"mode={self.mode!r} cannot consume sub-MPO payloads."
            )
        if self.mode == "mix" and (controls or has_leakage):
            raise ValueError(
                "mode='mix' is unitary-only and cannot replay controls or leakage."
            )
        if error_model is not None and _has_unforced_branching_control(entries):
            raise ValueError(
                "error_model shot replay cannot combine unforced controls; "
                "use stream-local trajectory events instead."
            )

    def _run_shots(
        self,
        shots,
        *,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="auto",
        max_branches=_SHOT_DEFAULT_MAX_BRANCHES,
        auto_max_expected_faults=_SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=False,
        checkpoint_id=None,
    ):
        """Replay this stream as an independent or coalesced shot ensemble."""
        self._validate_shot_compatibility(error_model=error_model)
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("shots must be a nonnegative integer.")
        if self.mode == "perm" and self.logical_order != list(
            range(int(getattr(self.p, "L", 0)))
        ):
            raise ValueError(
                "shot replay does not support a permuted live MPS; "
                "create a fresh optimizer before running shots."
            )
        if error_model is not None and self._has_trajectory_events:
            raise ValueError(
                "do not combine stream-local trajectory events with error_model; "
                "use one noise representation per gate stream."
            )

        from ..noise import (  # pylint: disable=import-outside-toplevel
            NoisyResult,
            run_noisy_shots,
            run_trajectory_shots,
        )

        mpi_enabled = mpi is not None and mpi is not False
        if not mpi_enabled and any(
            value is not None
            for value in (observable, checkpoint_path)
        ):
            raise ValueError(
                "observable and checkpoint options require mpi=True or an MPI communicator."
            )
        if not mpi_enabled and (
            resume
            or checkpoint_keep != 2
            or checkpoint_sync is not True
            or collect_diagnostics is not False
            or checkpoint_id is not None
        ):
            raise ValueError(
                "MPI checkpoint options require mpi=True or an MPI communicator."
            )

        if workers not in {None, "auto"}:
            if (
                isinstance(workers, bool)
                or not isinstance(workers, Integral)
                or workers < 1
            ):
                raise ValueError("workers must be a positive integer or 'auto'.")
        if workers in {None, "auto"} and parallel_workers != 1:
            workers = parallel_workers
        if mpi_enabled:
            from ..mpi import MPIShotRunner  # pylint: disable=import-outside-toplevel

            if strategy == "auto":
                strategy = "independent"
            child_kwargs = dict(run_kwargs or {})
            if progress not in {False, "never"}:
                child_kwargs["progbar"] = False
            communicator = None if mpi is True else mpi
            mpi_gates = (
                self._stream_plan.trajectory_plan
                if self._has_trajectory_events
                else self._gate_stream
            )
            runner = MPIShotRunner(
                self._shot_factory(),
                mpi_gates,
                comm=communicator,
            )
            return runner.run(
                shots,
                seed=seed,
                error_model=error_model,
                run_kwargs=child_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                retain=retain,
                local_workers=workers,
                local_backend="auto",
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
                progress=progress,
            )

        if workers in {None, "auto"}:
            from ..mpi import _resolve_local_workers  # pylint: disable=import-outside-toplevel

            workers = _resolve_local_workers(workers, shots=shots)
        from ..mpi import (  # pylint: disable=import-outside-toplevel
            _make_progress_bar,
            _validate_progress,
        )
        progress_strategy = strategy
        if workers > 1 and strategy == "auto":
            from ..noise import _resolve_auto_parallel_strategy

            progress_strategy = _resolve_auto_parallel_strategy(
                self._stream_plan.entries,
                shots,
                error_model=error_model,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                auto_max_expected_faults=auto_max_expected_faults,
            )

        progress_mode = _validate_progress(progress)
        progress_bar = _make_progress_bar(
            progress_mode,
            shots,
            desc="shots",
        ) if workers > 1 and progress_strategy == "independent" else None
        child_kwargs = dict(run_kwargs or {})
        if workers > 1 and progress_mode != "never":
            child_kwargs["progbar"] = False

        def update_progress(delta):
            if progress_bar is not None:
                progress_bar.update(int(delta))

        common = {
            "seed": seed,
            "run_kwargs": child_kwargs,
            "strategy": strategy,
            "max_branches": max_branches,
            "importance_sampling": importance_sampling,
            "max_branch_factor": max_branch_factor,
            "parallel_workers": workers,
            "parallel_backend": parallel_backend,
            "retain": retain,
        }
        try:
            if error_model is None:
                shot_gates = (
                    self._stream_plan.entries
                    if workers > 1
                    else self._stream_plan.trajectory_plan
                )
                raw = run_trajectory_shots(
                    self._shot_factory(),
                    shot_gates,
                    shots,
                    _progress=update_progress if progress_bar is not None else None,
                    **common,
                )
            else:
                raw = run_noisy_shots(
                    self._shot_factory(),
                    self._gate_stream,
                    error_model,
                    shots,
                    auto_max_expected_faults=auto_max_expected_faults,
                    _progress=update_progress if progress_bar is not None else None,
                    **common,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return NoisyResult(raw)

    def set_gates(self, gates):
        """Replace the current gate list.

        After calling this, ``run(...)`` applies only this new list
        (unless you call :meth:`add_gates` before running).
        """
        plan = _prepare_gate_stream(
            gates,
            to_backend=self._symbolic_gate_to_backend,
            backend_sample=self._state_backend_like(),
        )
        normalized_queue = self._normalize_stream_plan_queue(plan)
        self._validate_normalized_gate_queue(normalized_queue)
        self._shared_backend_cache = False
        self._install_stream_plan(plan, normalized_queue=normalized_queue)
        self.scheduled_layout_plan = None
        self.scheduled_site_order = None
        return self

    def set_gate_schedule(self, schedule, *, reorder_product_state=True):
        """Install a precompiled lifetime-aware gate-stream schedule.

        ``schedule`` is intentionally duck-typed so Pepsy does not depend on
        Tensy's scheduler package.  It must provide ``stream`` and may provide
        ``site_order`` / ``layout_plan`` as returned by Tensy's
        ``schedule_gate_stream``.  The stream already contains physical MPS
        positions and cap events shifted after each removal, so no layout
        replay is requested here (layout replay and cap replay are separate
        operations).

        A non-identity initial layout can be installed exactly only for a
        product input MPS.  For an entangled input, the caller must first
        supply the state in ``schedule.site_order`` or use an explicitly
        controlled lossy reorder.
        """
        return _layout_execution.set_gate_schedule(
            self,
            schedule,
            reorder_product_state=reorder_product_state,
        )

    def add_gates(self, gates):
        """Append gates to the existing gate list.

        This preserves previously queued gates and extends them with
        new ones.
        """
        new_plan = _prepare_gate_stream(
            gates,
            to_backend=self._symbolic_gate_to_backend,
            backend_sample=self._state_backend_like(),
        )
        plan = _prepare_gate_stream(
            self._gate_stream + new_plan.entries,
            to_backend=self._symbolic_gate_to_backend,
            backend_sample=self._state_backend_like(),
        )
        normalized_queue = self._normalize_stream_plan_queue(plan)
        self._validate_normalized_gate_queue(normalized_queue)
        self._install_stream_plan(plan, normalized_queue=normalized_queue)
        self.scheduled_layout_plan = None
        self.scheduled_site_order = None
        return self

    @staticmethod
    def _layout_request_enabled(layout):
        return layout is not None and layout is not False

    @staticmethod
    def _coalesce_layout_request(use_layout_finder, layout):
        """Resolve the explicit layout-finder keyword and compatibility alias."""
        primary = use_layout_finder
        alias = layout
        if (
            primary is not None
            and primary is not False
            and alias is not None
            and alias is not False
        ):
            raise ValueError(
                "Specify only one of use_layout_finder=... or layout=...."
            )
        if primary is not None and primary is not False:
            return primary
        return alias

    _resolve_run_layout = _layout_execution._resolve_run_layout

    _validate_layout_plan_for_mps = _layout_execution._validate_layout_plan_for_mps

    _explicit_layout_plan = _layout_execution._explicit_layout_plan

    _resolve_layout_plan_argument = _layout_execution._resolve_layout_plan_argument

    _product_site_vector = staticmethod(_layout_execution._product_site_vector)

    _relabel_product_mps = _layout_execution._relabel_product_mps

    def apply_layout(
        self,
        plan_or_order="quality",
        *,
        cutoff=None,
        cutoff_mode="rsum2",
        allow_lossy_reorder=False,
        layout_kwargs=None,
        layout_report=True,
    ):
        """Install a layout permanently and return this optimizer.

        Parameters
        ----------
        plan_or_order : mapping | str | sequence, default="quality"
            A plan returned by :meth:`gate_stream_layout`, a finder order name,
            or an explicit position-to-logical-site permutation.
        cutoff : float | None, default=None
            Cutoff for the one-time reorder of an initially entangled MPS.
            ``None`` uses ``1e-12``. This value is never used for product-state
            relabeling and is never used to restore the original order.
        cutoff_mode : str, default="rsum2"
            Cutoff mode for the optional one-time entangled-state reorder.
        allow_lossy_reorder : bool, default=False
            Allow the one-time reorder when ``p.max_bond() > 1``. If false,
            entangled initial states raise before mutation.
        layout_kwargs : mapping | None, default=None
            Extra keyword arguments passed to the layout finder for string
            ``plan_or_order`` values.
        layout_report : bool, default=True
            Print the usual layout summary when a finder plan is selected.

        Notes
        -----
        The installed ``logical_order`` maps physical MPS positions to logical
        site labels. Subsequent :meth:`run` calls reuse this map and do not
        reorder the MPS back to logical order. Use :meth:`to_dense` or
        :meth:`remap_sample` for logical-order readout.
        """
        return _layout_execution.apply_layout(
            self,
            plan_or_order,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            allow_lossy_reorder=allow_lossy_reorder,
            layout_kwargs=layout_kwargs,
            layout_report=layout_report,
        )

    _reorder_mps_to_logical_order = _layout_execution._reorder_mps_to_logical_order

    _normalize_visible_mps_order = _layout_execution._normalize_visible_mps_order

    _copy_submpo_for_layout = staticmethod(_layout_execution._copy_submpo_for_layout)

    _layout_run_sequences = _layout_execution._layout_run_sequences

    _format_layout_value = staticmethod(_format_layout_value)

    @classmethod
    def _format_layout_reduction(cls, before, after):
        """Format a reduction using this class's value formatter."""
        return _format_layout_reduction(
            before, after, format_value=cls._format_layout_value
        )

    @classmethod
    def _layout_report_text(cls, plan):
        """Format a layout report while preserving subclass formatters."""
        return _layout_report_text(
            plan,
            format_value=cls._format_layout_value,
            format_reduction=cls._format_layout_reduction,
        )

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=8,
        progbar=False,
        cutoff="auto",
        cutoff_mode="auto",
        mode=None,
        k_2q_batch=1,
        non_unitary=False,
        _trajectory_non_unitary=False,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        submpo_method=None,
        compression_seed=None,
        use_layout_finder=False,
        layout_order="quality",
        layout_kwargs=None,
        layout=None,
        layout_report=True,
        measure_renormalize=True,
        seed=None,
        mix_strict=False,
        mix_fit_min_iter=_DEPRECATED_OPTION,
        mix_fit_rtol=_DEPRECATED_OPTION,
        mix_fit_patience=_DEPRECATED_OPTION,
        mix_sticky_nonfinite=False,
        *,
        compression_opts=None,
        fit_min_iter=2,
        fit_rtol="auto",
        fit_patience=2,
        fit_block_size=None,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        fit_layer_size=None,
        fit_max_span="auto",
        fit_three_site_sweeps=_DEPRECATED_OPTION,
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_init_strategy=None,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        fit_stabilize_unitary=_DEPRECATED_OPTION,
        timing=False,
        timing_sync_device=False,
        quality_check_every=False,
        quality_check_repair=True,
        shots=1,
        error_model=None,
        strategy="auto",
        run_kwargs=None,
        max_branches=_SHOT_DEFAULT_MAX_BRANCHES,
        auto_max_expected_faults=_SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=False,
        checkpoint_id=None,
        retain="all",
    ):
        """Run the currently queued gates.

        Parameters
        ----------
        n_iter : int, default=8
            Inner iterations for DMRG local fits. In ``dmrg`` and ``mix``
            modes this is the maximum number of sweeps when adaptive FIT
            stopping is enabled; pass ``fit_rtol=None`` for fixed
            iterations. Adaptive rank-growing windows require at least two
            sweeps. An under-capacity non-adjacent ``dmrg1`` window requires
            ``n_iter >= 3`` so its two fixed growth sweeps leave room for
            one-site refinement. Once all attainable full-chain bond ceilings
            are reached, ``dmrg1`` stays in the one-site phase. The adjacent
            two-site exact fast path is exempt.
            Ignored by ``mpo``/``swap``/``svd``/``exact``.
        progbar : bool, default=False
            Show per-mode progress bars.
        cutoff : float | {"auto"}, default="auto"
            Truncation cutoff used in gate application and local fitting.
            The default ``"auto"`` selects a conservative dtype-aware value;
            pass an explicit number to preserve a fixed cutoff.
        cutoff_mode : str | None | {"auto"}, default="auto"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``. ``"auto"`` (and compatibility
            value ``None``) uses ``"rsum2"`` for Pepsy's ordinary compression
            paths while preserving Quimb's method-specific native default for
            the MPO path, notably ``"rsum1"`` for ``method="dm"``. Pass a
            string to override it.
        mode : str | None, default=None
            Optional compression-algorithm override for this run. If omitted,
            use the constructor's mode (``"direct"`` by default). If supplied,
            update ``self.mode`` before execution. ``"mpo"`` remains a
            compatibility alias for ``"direct"``. ``mode="perm"`` is the
            standalone lazy permutation swap-and-split path.
        k_2q_batch : int, default=1
            DMRG and mixed modes: number of contiguous two-qubit gates to batch
            into one local FIT update. In mixed mode, a failed batch is replayed
            through direct compression as one transaction. Standalone one-site gates use the
            exact direct path; an ordinary DMRG target block can also absorb
            intervening one-site gates before its shared FIT compression.
        non_unitary : bool, default=False
            Convenience flag for non-unitary gate streams. Normalization is
            only available when this is ``True``. This physical scale control
            is separate from unitary FIT working-norm stabilization. If
            enabled, local tensor scale control moves the
            orthogonality center to one site and normalizes it after every
            replay step. In DMRG, a step containing a multi-gate batch is
            normalized once after that batch. The removed scale is accumulated
            in ``p.exponent``. Exact replay preserves the physical operator
            scale directly and does not require this flag; it accepts the flag
            for trajectory compatibility but does not apply MPS scale control.
        normalize_every : int | bool | None, default=False
            Enable one-site normalization after every replay step for a
            non-unitary stream. Use ``True`` (or any positive integer); use
            ``False`` or ``None`` to leave tensor scales untouched. Integer
            values are accepted as a boolean-style convenience and do not
            select an interval.
        normalize_final : bool, default=False
            Normalize a trailing state if the final replay step was not
            already normalized. Requires ``non_unitary=True``.
        normalize_eps : float, default=1e-15
            Numerical threshold used by the final normalization path.
        submpo_method : str | None, default=None
            Optional compression-method override for the Quimb family. If
            omitted, a bare Quimb method such as ``mode="src"`` or the
            qualified ``mode="quimb-<method>"`` selects the method;
            the default ``mode="direct"`` selects ``"direct"``. The opt-in
            ``"sdc"`` and ``"sdc-oversample"`` methods require a Quimb build
            that provides those compressors; ``"sdcr"`` and
            ``"sdcr-oversample"`` are the randomized-environment variants.
            These methods never replace an existing default. The legacy
            ``mode="mpo-<method>"`` / ``mode="mpo"`` spellings remain valid.
            The method
            is forwarded to Quimb for both dense gates and explicit sub-MPO
            stream events.
        compression_seed : int | None, default=None
            Explicit seed forwarded to randomized Quimb compression methods
            such as ``src``, ``srcmps``, and randomized FIT variants. ``None``
            preserves Quimb's backend-global random state.
        compression_opts : mapping | None, default=None
            Independent intermediate/final Quimb compression controls:
            max_bond_oversample, cutoff_oversample, cutoff_mode_oversample,
            and compress_opts_final. The latter accepts method, cutoff, and
            cutoff_mode; chi remains the final bond cap. Unsupported options
            and native/non-Quimb replay paths reject explicit settings.
        use_layout_finder : bool | str | Mapping, default=False
            Deprecated compatibility path. If enabled, call
            :meth:`layout_finder`, temporarily replay the stream in the
            selected 1D site order, then restore the MPS to original site
            order. Use :meth:`apply_layout` for repeated evolution. ``True``
            uses ``layout_order``; a string is used as the order name; a
            mapping is treated as a precomputed layout plan.
        layout_order : str, default="quality"
            Order passed to :meth:`layout_finder().run` when
            ``use_layout_finder=True``.
        layout_kwargs : Mapping | None, default=None
            Extra keyword arguments forwarded to ``layout_finder().run``.
        layout : bool | str | Mapping | None, default=None
            Compatibility alias for ``use_layout_finder``.
        layout_report : bool, default=True
            Print a concise before/after layout summary when layout-aware
            replay is used.
        measure_renormalize : bool, default=True
            Whether ``("measure", ...)`` and ``("reset", ...)`` control events
            renormalize the MPS to unit norm after the projective collapse. The
            outcome's Born probability is still recorded in
            :attr:`measurements`. The layout finder works with measure/reset
            control events (recorded sites always use the logical labels) but
            not with ``cap`` events, which change the MPS length.
        seed : int | None, default=None
            If given, reseed the internal RNG used to sample ``measure``/
            ``reset`` outcomes before running, for reproducible collapses.
        mix_strict : bool, default=False
            In ``mode="mix"``, restore the committed state and re-raise an
            ordinary DMRG trial exception instead of falling back to MPO.
        mix_fit_min_iter, mix_fit_rtol, mix_fit_patience : optional
            Deprecated compatibility aliases for ``fit_min_iter``,
            ``fit_rtol``, and ``fit_patience``. New code should use the
            mode-neutral keyword-only names below.
        mix_sticky_nonfinite : bool, default=False
            After a mixed DMRG trial produces NaN or Inf, use MPO for the
            remainder of this :meth:`run` call instead of retrying DMRG on
            every subsequent gate.
        fit_min_iter : int, default=2
            Minimum FIT sweeps before adaptive convergence can stop in
            ``dmrg`` or ``mix`` mode. Values above ``n_iter`` are clamped to
            ``n_iter``.
        fit_rtol : {"auto"} | float | None, default="auto"
            Relative tolerance for DMRG FIT early stopping. ``"auto"``
            selects a dtype-aware tolerance: ``1e-3`` for 16-bit data,
            ``1e-5`` for 32-bit/``complex64`` data, and ``1e-9`` for higher
            precision data. Early stopping compares changes in the retained
            canonical-center norm ``A``. ``None`` disables early stopping and
            restores fixed ``n_iter`` behavior.
        fit_patience : int, default=2
            Number of same-phase sweep-norm samples in the convergence window.
            The default of two stops after one stable comparison between two
            one-site sweeps. Rank-adaptive DMRG still performs its minimum
            adaptive warm-up before this criterion can stop a run.
        fit_block_size : {1, 2, 3} | None, default=None
            Number of neighboring MPS tensors optimized by each FIT update.
            ``None`` selects two-site FIT for ordinary DMRG and one-site FIT
            for ``mode="mix"``. Mixed mode fixes this value at one: a
            chi-capped direct-compressed guess opens the active bond support
            before every eligible multi-site gate is refined with one-site
            FIT. Use ordinary ``mode="dmrg"`` to select block sizes two or
            three.
            Two-site FIT is recommended: it forms both physical legs and the
            two outer virtual legs, then uses a native SVD on the middle bond,
            allowing active bonds to grow up to ``chi``. One-site FIT is kept
            for compatibility with the original fixed-rank update. Three-site
            FIT forms a three-site wavefunction and performs two native,
            direction-aware SVD splits. If the active gate span contains only
            two sites, it automatically falls back to the two-site update.
            Two- and three-site FIT never pre-expand the MPS; only bonds
            visited by their native splits can grow.
        fit_adaptive_sweeps : int, default=2
            Generic ``mode="dmrg"``: minimum number of initial two- or
            three-site sweeps used to adapt the active bond spaces. Generic
            rank-adaptive DMRG continues block sweeps until every active bond
            reaches its physical ceiling; rank stagnation never triggers the
            transition. Long-range windows use the corresponding fixed block
            handoff so their terminal canonical center remains authoritative
            for unitary norm tracking. Otherwise, if a ceiling is not reached,
            the block phase uses all requested sweeps, and remaining sweeps use
            fixed-rank one-site FIT. For named ``mode="dmrg1"``, the two-site phase is fixed at
            two sweeps and this value does not extend it; after that, remaining
            sweeps use one-site FIT and the phase latches once all full-chain
            attainable ceilings are reached. For ``mode="dmrg2"`` and
            ``mode="dmrg3"``, this sets the required two- or three-site
            warm-up length. The value is clipped to ``n_iter`` and ignored
            for ``fit_block_size=1``; the default is two sweeps. Mixed mode's
            one-site path does not use this value as a two-site warm-up.
        fit_sweep_sequence : str, default="RL"
            Cyclic FIT sweep directions. ``"R"`` is left-to-right, ``"L"``
            is right-to-left, and ``"RL"`` alternates. Alternating sweeps avoid
            favoring one canonical direction.
        fit_layer_size : int | None, default=None
            Clear alias for ``k_2q_batch``: the number of sequential two-site
            circuit gates absorbed into one paper-style target block. This is
            independent of ``fit_block_size``, which controls the local
            variational wavefunction tensor.
        fit_max_span : int | {"auto"} | None, default="auto"
            Maximum inclusive spatial span of a batched FIT target. ``"auto"``
            keeps ordinary local layers together while splitting disjoint
            gates before they create an unnecessarily wide active window.
            ``None`` restores unrestricted gate-count batching.
        fit_three_site_sweeps : int, deprecated
            Compatibility alias for ``fit_adaptive_sweeps``. New code should
            use the common adaptive-sweep control.
        target_cutoff : float, default=0.0
            Cutoff used only while constructing the pre-FIT gate target.
            Keeping this at zero separates exact target construction from the
            output truncation controlled by ``cutoff``.
        fit_target_strategy : {"auto", "layered", "mps"}, default="auto"
            Exact target representation. ``"layered"`` keeps ordinary dense
            gates as lazily contracted operator-Schmidt tensors, avoiding
            intermediate target-MPS rank growth. ``"mps"`` materializes the
            traditional routed target. ``"auto"`` selects layered targets
            for NumPy/Torch/CuPy and the native MPS route for Symmray.
            Explicit multi-site sub-MPO events in DMRG use the same layered
            representation when the backend supports lazy FIT targets.
        fit_mpo_guess : bool, default=True
            Legacy compatibility switch for the named DMRG1/DMRG3 default
            ``"guess-src"`` policy. New code should use
            ``fit_init_strategy`` explicitly. Dense DMRG uses the disposable
            SRC guess in both the expansion and one-site/reached-chi phases.
            Native Symmray and fermionic routes use a disposable
            sector-preserving randomized guess for the ``guess-src`` policy;
            this does not replace the exact FIT target or live MPS.
        fit_init_strategy : {"auto", "direct", "random", "random_expand", "guess-<method>"} | None, default=None
            Select the disposable FIT initial guess. ``"direct"`` uses the
            current MPS, ``"random"`` perturbs existing tensors without
            changing bond dimensions, ``"random_expand"`` adds seeded
            directions on under-capacity active bonds, and
            ``"guess-<method>"`` uses the corresponding Quimb compression
            method on an isolated copy. For native Symmray/fermionic states,
            ``"guess-src"`` instead uses Symmray's sector-preserving
            randomized SVD on an isolated copy; other Quimb guess methods
            retain the native direct fallback. ``None`` selects
            ``"guess-src"`` for DMRG and the fixed ``"guess-direct"`` policy
            for mixed mode. ``"auto"`` selects ``"guess-src"`` in both
            expansion and reached-chi phases for ordinary DMRG.
            On native Symmray/fermionic states this is the sector-preserving
            randomized guess. Mixed mode requires ``"guess-direct"`` and
            uses the native auto-swap/SVD equivalent for Symmray arrays. The
            underscore spelling ``"guess_<method>"`` remains accepted as a
            compatibility alias.
        fit_init_rand_strength : float, default=0.0
            For dense two- and three-site FIT growth windows that are below
            their attainable physical/``chi`` bond ceilings, seed a
            disposable copy of the current MPS with random entries on only
            those active bonds. The exact FIT target is still built from the
            unmodified current MPS. Ordinary DMRG defaults to ``"guess-src"``
            and mixed mode fixes ``"guess-direct"``, so this strength is
            unused unless a random strategy is selected explicitly in DMRG.
            Set it to a positive value to enable random initialization. Native
            Symmray and fermionic routes ignore it.
        fit_init_seed : int, default=0
            Deterministic seed for ``"random"`` and ``"random_expand"`` FIT
            guesses and randomized Quimb methods selected through
            ``"guess-<method>"``. Native ``guess-src`` uses the same seed for
            Symmray randomized SVD. The underscore spelling is also accepted.
            The gate position is mixed into the
            per-window stream so repeated runs are reproducible without
            sharing a global RNG.
        fit_single_pair_fast_path : bool, default=False
            Stop an adjacent two-site FIT after its single exact variational
            update. This structural convergence is independent of ``rtol``;
            enable it when deliberately choosing the one-update fast path.
        fit_overlap_diagnostics : bool, default=False
            Contract the final fitted MPS against the disposable exact FIT
            target and report target-overlap fidelity. This adds an extra
            tensor-network contraction after each successful DMRG FIT update;
            when disabled, the overlap fields remain ``None`` while the
            ordinary FIT convergence metadata is still collected.
        stabilize_unitary : bool, default=False
            By default, retain the raw norm change after each unitary FIT or
            mixed/MPO/swap/permutation/SVD compression so norm loss remains
            observable. Set this to ``True`` to restore the working norm for
            numerical scale control; the discarded scale is not stored in
            ``p.exponent``. This option cannot be combined with
            ``non_unitary=True``.
        fit_stabilize_unitary : optional
            Deprecated compatibility alias for ``stabilize_unitary``.
        finite_check : bool, default=False
            Optional diagnostic tensor and scalar-norm validation, not
            required for normal optimization. Leave False to avoid the extra
            checks and possible accelerator synchronization. False disables
            FIT finite scans, non-finite convergence-norm checks, mixed-mode
            commit validation, and unitary norm-consistency validation.
            True also checks the final tensor data in every replay mode and
            emits one performance warning per replay, shared by all nested
            FIT calls. Shot workers inherit this flag
            unless ``run_kwargs`` overrides it. Norm calculations needed for
            convergence/accounting and explicit quality/overlap diagnostics
            remain independent, as do input validation and zero-divisor guards.
        timing : bool, default=False
            Record wall-clock replay timing in :attr:`last_run_timing` without
            printing. Timing is fully opt-in: disabled runs retain the normal
            path without profiling clocks or records. Enabled records include
            inclusive stage totals
            for gate preparation, canonicalization, gate application, FIT,
            normalization, control-event measurement, and the active mode
            replay. Mixed-mode records also include the
            final :attr:`last_mix_summary`. Profiling does not enable FIT SVD
            split diagnostics.
        timing_sync_device : bool, default=False
            When timing is enabled, synchronize supported CUDA/CuPy/JAX work
            at timing boundaries so reported values include device kernels.
            The accelerator route is resolved once; CPU data needs no barrier.
            Leave disabled for the lowest-overhead timing run.
        quality_check_every : int | bool | None, default=False
            If set, periodically check finite tensor data and canonical-gauge
            coverage after replay steps. ``True`` checks every step.
        quality_check_repair : bool, default=True
            Re-canonicalize the live MPS when a periodic gauge check detects
            missing canonical coverage.
        shots : int, default=1
            Number of trajectories to replay. The default preserves the
            single-state return value for ordinary streams. A stream-local
            noisy stream, an explicit ``shots != 1``, or shot-runner options
            dispatches to the trajectory result facade.
        error_model : PauliErrorModel | None, default=None
            Optional legacy Pauli error model for a clean gate stream. This
            selects the Pauli shot runner and cannot be combined with
            stream-local trajectory or leakage entries.
        strategy : {"auto", "independent", "coalesced"}, default="auto"
            Shot representation strategy. ``"auto"`` shares deterministic
            prefixes when the branch count remains bounded and otherwise
            restarts with independent trajectories.
        run_kwargs : mapping | None, default=None
            Explicit overrides for each fresh optimizer's ordinary replay.
            Top-level numerical options are inherited; values in this mapping
            take precedence. Shot scheduling and RNG remain parent controls.
        max_branches : int | None, default=128
            Safety cap for coalesced trajectory replay.
        auto_max_expected_faults : float, default=0.1
            Expected-fault threshold used by automatic legacy Pauli replay.
        importance_sampling : ImportanceSamplingPolicy | None, default=None
            Optional proposal policy for trajectory events.
        max_branch_factor : int | None, default=None
            Optional per-event branch-growth cap for coalesced replay.
        parallel_workers : int, default=1
            Number of workers for explicit parallel shot execution.
        parallel_backend : {"thread", "gpu", "serial"}, default="thread"
            Backend used for explicit parallel shot execution.
        mpi : bool | MPI communicator | None, default=None
            Run the shot ensemble collectively over MPI. ``True`` uses
            ``MPI.COMM_WORLD``; an explicit communicator can be supplied.
        workers : int | "auto" | None, default="auto"
            Local shot workers. ``"auto"`` uses the process CPU allowance and
            divides it across MPI ranks sharing a host. Use ``1`` to force
            serial local execution.
        progress : {"auto", True, False}, default="auto"
            Show one aggregate rank-zero shot progress bar for MPI runs.
        observable : callable, optional
            Observable evaluated during checkpointed MPI shot reduction.
        chunk_size : int, optional
            Number of shots processed in one checkpoint/reduction chunk.
        checkpoint_path : path-like, optional
            Destination for resumable MPI checkpoints.
        resume : bool, default=False
            Resume from ``checkpoint_path`` when a compatible checkpoint exists.
        checkpoint_keep : int, default=2
            Number of completed checkpoints retained on disk.
        checkpoint_sync : bool, default=True
            Synchronize checkpoint writes across MPI ranks.
        collect_diagnostics : bool, default=False
            Opt in to bounded-memory MPI diagnostic summaries and rank timing
            during reduction. Disabled runs skip those profiling clock reads.
        checkpoint_id : str, optional
            Stable identifier used to distinguish checkpoint streams.
            These options require ``mpi=True`` or an explicit MPI communicator.
        retain : {"all", "final", "none"}, default="all"
            Result retention policy for shot replay. ``"all"`` retains final
            states and replay metadata, ``"final"`` retains final states only,
            and ``"none"`` retains no optimizer states.

        Returns
        -------
        qtn.TensorNetwork | NoisyResult | MPIShotResult
            The updated ``self.p`` state for a single ordinary replay, or a
            stable noisy/MPI result facade when shot replay is selected.
        """
        if not isinstance(finite_check, (bool, np.bool_)):
            raise TypeError("finite_check must be a boolean.")
        finite_check = bool(finite_check)
        replay_options = locals()
        if self._shot_runner_requested(
            shots,
            has_trajectory_events=self._has_trajectory_events,
            error_model=error_model,
            strategy=strategy,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
            importance_sampling=importance_sampling,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
            auto_max_expected_faults=auto_max_expected_faults,
            retain=retain,
            mpi=mpi,
            workers=workers,
            checkpoint_path=checkpoint_path,
            observable=observable,
        ):
            if mode is not None:
                self.set_mode(mode)
            run_kwargs = self._resolve_shot_replay_options(replay_options, run_kwargs)
            return self._run_shots(
                shots,
                error_model=error_model,
                seed=seed,
                run_kwargs=run_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                auto_max_expected_faults=auto_max_expected_faults,
                importance_sampling=importance_sampling,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
                retain=retain,
                mpi=mpi,
                workers=workers,
                progress=progress,
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
            )

        timing = bool(timing)
        timing_sync_device = bool(timing_sync_device)
        if mode is not None:
            self.set_mode(mode)
        cutoff = self._resolve_cutoff(cutoff)
        quality_check_every = self._resolve_quality_check_every(
            quality_check_every
        )
        quality_check_repair = bool(quality_check_repair)
        self.quality_checks = []
        # Mixed mode is a fixed one-site FIT algorithm initialized from a
        # disposable direct-compressed guess. Ordinary DMRG retains its
        # two-site and ``guess-src`` defaults.
        if fit_block_size is None:
            fit_block_size = 1 if self.mode == "mix" else 2
        if fit_init_strategy is None:
            fit_init_strategy = (
                "guess_direct"
                if self.mode == "mix"
                else _DEFAULT_FIT_INIT_STRATEGY
            )

        # ``auto`` (and the legacy ``None``) uses Pepsy's relative squared
        # weight policy for ordinary paths. MPO paths retain Quimb's native
        # method default when auto is selected, notably rsum1 for ``dm``.
        mpo_cutoff_mode = self._resolve_cutoff_mode(
            cutoff_mode,
            preserve_mpo_default=self._is_mpo_mode(self.mode),
        )
        cutoff_mode = self._resolve_cutoff_mode(cutoff_mode)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.last_layout_plan = self._persistent_layout_plan
        G_seq = list(self.G)
        where_seq = list(self.where)
        event_seq = list(self.event_types)
        if self.mode == "mix":
            self.mix_history = []
            self.last_mix_summary = None
            self._mix_dmrg_disabled_reason = None
            self._mix_dmrg_failed_sweep = None
        if not G_seq:
            def run_empty():
                return self.p

            return self._run_with_fit_copy_policy(
                run_empty,
                enabled=timing,
                finite_check=finite_check,
                event_count=0,
                sync_device=timing_sync_device,
            )
        self._validate_symmray_mode_support()
        self._validate_event_stream_for_run(G_seq, where_seq, event_seq)
        has_control = any(
            event_type in _CONTROL_EVENT_NAMES for event_type in event_seq
        )
        has_cap = any(
            _control_event_contains_cap(event_type, payload)
            for payload, event_type in zip(G_seq, event_seq)
        )
        layout_request = self._coalesce_layout_request(use_layout_finder, layout)
        persistent_layout_active = self._persistent_layout_plan is not None
        if self.mode == "perm" and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "mode='perm' keeps a lazy logical-to-physical permutation; "
                "use either the perm mode or a persistent/transient layout, "
                "not both."
            )
        if has_cap and any(
            event_type == "conditional"
            and _control_event_contains_cap(event_type, payload)
            for payload, event_type in zip(G_seq, event_seq)
        ) and (persistent_layout_active or self._layout_request_enabled(layout_request)):
            raise ValueError(
                "layout replay does not support conditional cap events; the "
                "active branch is needed to update the shrinking layout."
            )
        # Preserve the logical (pre-layout) event locations so control-event
        # bookkeeping (e.g. recorded measurement sites) always refers to the
        # user's site labels even when the run replays in a layout order.
        logical_where_seq = list(where_seq)
        if persistent_layout_active:
            if self._layout_request_enabled(layout_request):
                raise ValueError(
                    "a persistent layout is already installed; call run() without "
                    "use_layout_finder/layout arguments."
                )
            layout_plan = self._persistent_layout_plan
            self.last_layout_plan = layout_plan
        else:
            if self._layout_request_enabled(layout_request):
                warnings.warn(
                    "use_layout_finder/layout performs a temporary reorder and "
                    "swap-back; call apply_layout(...) for a persistent layout.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _, layout_plan = self._resolve_run_layout(
                layout_request,
                layout_order,
                layout_kwargs,
            )
        layout_current_order = None
        if layout_plan is not None:
            if layout_report and not persistent_layout_active:
                report = self._layout_report_text(layout_plan)
                if report:
                    print(report)
            replay_event_order = layout_plan.get("replay_event_order")
            if replay_event_order is not None:
                try:
                    replay_event_order = tuple(
                        int(index) for index in replay_event_order
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "layout replay event order must contain integer indices."
                    ) from exc
                expected_event_order = tuple(range(len(G_seq)))
                if (
                    len(replay_event_order) != len(G_seq)
                    or set(replay_event_order) != set(expected_event_order)
                ):
                    raise ValueError(
                        "layout replay event order must be a permutation of the "
                        "queued stream events."
                    )
                G_seq = [G_seq[index] for index in replay_event_order]
                where_seq = [where_seq[index] for index in replay_event_order]
                event_seq = [event_seq[index] for index in replay_event_order]
                logical_where_seq = [
                    logical_where_seq[index] for index in replay_event_order
                ]
            layout_order_tuple = tuple(layout_plan["site_order"])
            G_seq, where_seq = self._layout_run_sequences(
                G_seq,
                where_seq,
                event_seq,
                layout_plan,
            )

        non_unitary = bool(non_unitary)
        if non_unitary or has_control:
            # Non-unitary and control-event runs can change the represented
            # norm without a unitary compression, so the next unitary stream
            # must establish a fresh local norm reference.
            self._unitary_previous_norm = None
        if not non_unitary:
            if normalize_every is not None and normalize_every is not False:
                raise ValueError("normalize_every requires non_unitary=True.")
            if normalize_final:
                raise ValueError("normalize_final requires non_unitary=True.")
        # This is the public-to-backend policy boundary. Validate the stream
        # and layout first, then resolve sentinels once so every mode receives
        # the same numeric cutoff and normalization contract. In particular,
        # ``None`` means Pepsy's default for ordinary paths but remains
        # observable as an omission for Quimb's method-specific MPO defaults.
        normalize_every = self._normalize_every_interval(
            normalize_every,
            non_unitary=non_unitary,
        )
        fit_min_iter = self._resolve_legacy_fit_option(
            canonical_name="fit_min_iter",
            canonical_value=fit_min_iter,
            canonical_default=2,
            legacy_name="mix_fit_min_iter",
            legacy_value=mix_fit_min_iter,
        )
        fit_rtol = self._resolve_legacy_fit_option(
            canonical_name="fit_rtol",
            canonical_value=fit_rtol,
            canonical_default="auto",
            legacy_name="mix_fit_rtol",
            legacy_value=mix_fit_rtol,
        )
        fit_patience = self._resolve_legacy_fit_option(
            canonical_name="fit_patience",
            canonical_value=fit_patience,
            canonical_default=2,
            legacy_name="mix_fit_patience",
            legacy_value=mix_fit_patience,
        )
        stabilize_unitary = self._resolve_legacy_fit_option(
            canonical_name="stabilize_unitary",
            canonical_value=stabilize_unitary,
            canonical_default=False,
            legacy_name="fit_stabilize_unitary",
            legacy_value=fit_stabilize_unitary,
        )
        fit_overlap_diagnostics = bool(fit_overlap_diagnostics)
        if stabilize_unitary and non_unitary:
            raise ValueError(
                "stabilize_unitary=True cannot be combined with "
                "non_unitary=True; unitary norm restoration is not valid for "
                "a norm-changing stream."
            )
        if stabilize_unitary and not non_unitary:
            # Each stabilized unitary stream needs a fresh raw-norm reference.
            # Mixed mode then carries this working value across its trials.
            self._unitary_previous_norm = None
        if compression_seed is not None:
            if not isinstance(compression_seed, Integral) or isinstance(
                compression_seed, bool
            ):
                raise ValueError("compression_seed must be an integer or None.")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative.")
        if self.mode in {"dmrg", "mix"}:
            if self._dmrg_mode_block_size is not None:
                # A named DMRG mode is an explicit block-size choice. Use the
                # generic ``mode='dmrg'`` spelling when custom per-run block
                # sizes are needed.
                alias_block_size = (
                    2 if self._dmrg_mode_block_size == 1
                    else self._dmrg_mode_block_size
                )
                if (
                    isinstance(fit_block_size, Integral)
                    and int(fit_block_size)
                    not in {1, 2, alias_block_size}
                ):
                    raise ValueError(
                        f"mode='dmrg{self._dmrg_mode_block_size}' fixes "
                        "fit_block_size; use mode='dmrg' for a custom value."
                    )
                fit_block_size = alias_block_size
            if self.mode == "mix" and non_unitary and not _trajectory_non_unitary:
                raise ValueError("mode='mix' is only for unitary gate streams.")
            if not isinstance(n_iter, Integral) or int(n_iter) < 1:
                raise ValueError("n_iter must be a positive integer.")
            if not isinstance(k_2q_batch, Integral) or k_2q_batch < 1:
                raise ValueError("k_2q_batch must be a positive integer.")
            if fit_layer_size is not None:
                if (
                    not isinstance(fit_layer_size, Integral)
                    or int(fit_layer_size) < 1
                ):
                    raise ValueError("fit_layer_size must be a positive integer or None.")
                if int(k_2q_batch) != 1 and int(k_2q_batch) != int(fit_layer_size):
                    raise ValueError(
                        "fit_layer_size and k_2q_batch specify different target "
                        "layer sizes; pass only one or make them equal."
                    )
                k_2q_batch = int(fit_layer_size)
            fit_max_span = self._resolve_fit_max_span(
                fit_max_span,
                k_2q_batch,
            )
            if (
                not isinstance(fit_block_size, Integral)
                or int(fit_block_size) not in {1, 2, 3}
            ):
                raise ValueError("fit_block_size must be 1, 2, or 3.")
            fit_block_size = int(fit_block_size)
            if self.mode == "mix" and fit_block_size != 1:
                raise ValueError(
                    "mode='mix' fixes fit_block_size=1; use mode='dmrg' "
                    "for two- or three-site FIT."
                )
            fit_adaptive_sweeps = self._resolve_legacy_fit_option(
                canonical_name="fit_adaptive_sweeps",
                canonical_value=fit_adaptive_sweeps,
                canonical_default=2,
                legacy_name="fit_three_site_sweeps",
                legacy_value=fit_three_site_sweeps,
            )
            if (
                not isinstance(fit_adaptive_sweeps, Integral)
                or int(fit_adaptive_sweeps) < 1
            ):
                raise ValueError("fit_adaptive_sweeps must be a positive integer.")
            fit_adaptive_sweeps = min(int(fit_adaptive_sweeps), int(n_iter))
            fit_sweep_sequence = FIT._validate_sweep_sequence(
                fit_sweep_sequence
            )
            target_cutoff = float(target_cutoff)
            if not np.isfinite(target_cutoff) or target_cutoff < 0.0:
                raise ValueError(
                    "target_cutoff must be a finite non-negative number."
                )
            fit_target_strategy = self._validate_fit_target_strategy(
                fit_target_strategy
            )
            fit_init_strategy = self._validate_fit_init_strategy(
                fit_init_strategy
            )
            if self.mode == "mix" and fit_init_strategy != "guess_direct":
                raise ValueError(
                    "mode='mix' fixes fit_init_strategy='guess-direct'; "
                    "use mode='dmrg' for another FIT initialization."
                )
            try:
                fit_init_rand_strength = float(fit_init_rand_strength)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "fit_init_rand_strength must be finite and non-negative."
                ) from exc
            if not np.isfinite(fit_init_rand_strength) or fit_init_rand_strength < 0.0:
                raise ValueError(
                    "fit_init_rand_strength must be finite and non-negative."
                )
            if not isinstance(fit_init_seed, Integral) or isinstance(
                fit_init_seed, bool
            ):
                raise ValueError("fit_init_seed must be an integer.")
            fit_init_seed = int(fit_init_seed)
            if fit_init_seed < 0:
                raise ValueError("fit_init_seed must be non-negative.")
            if fit_target_strategy == "layered" and (
                self._replay_has_symmray_data(self.p) or self.p.isfermionic()
            ):
                raise ValueError(
                    "fit_target_strategy='layered' is not available for "
                    "Symmray/fermionic MPS data; use 'auto' or 'mps'."
                )
            if (
                not isinstance(fit_min_iter, Integral)
                or int(fit_min_iter) < 1
            ):
                raise ValueError("fit_min_iter must be a positive integer.")
            if (
                not isinstance(fit_patience, Integral)
                or int(fit_patience) < 1
            ):
                raise ValueError("fit_patience must be a positive integer.")
            # The FIT convergence test is relative to the current target and
            # remains meaningful when that target has a non-unit norm. Do not
            # disable adaptive stopping merely because the physical stream
            # changes norm.
            fit_rtol = self._resolve_fit_rtol(fit_rtol)
            current_max_bond = self.p.max_bond() if self.mode == "mix" else None
            if (
                self.mode == "mix"
                and current_max_bond is not None
                and current_max_bond > self.chi
            ):
                raise ValueError(
                    "mode='mix' requires the initial MPS max bond to be <= chi; "
                    "compress the state first or increase chi."
                )
        if self.mode in _EXACT_MODES and (
            normalize_every is not None or normalize_final
        ):
            raise ValueError(
                "automatic normalization uses MPS canonicalization and is not "
                "available in exact mode."
            )

        submpo_method = self._resolve_mpo_method(submpo_method)
        compression_opts = quimb_compression_options(submpo_method, compression_opts)
        if compression_opts and (not self._is_mpo_mode(self.mode) or self.backend == "symmray"):
            raise NotImplementedError("compression_opts requires dense Quimb MPS compression replay.")

        mode_kwargs = dict(
            n_iter=n_iter,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            mpo_cutoff_mode=mpo_cutoff_mode,
            k_2q_batch=k_2q_batch,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            compression_opts=compression_opts,
            mix_strict=bool(mix_strict),
            fit_min_iter=int(fit_min_iter),
            fit_rtol=fit_rtol,
            fit_patience=int(fit_patience),
            mix_sticky_nonfinite=bool(mix_sticky_nonfinite),
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            fit_max_span=fit_max_span,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_mpo_guess=bool(fit_mpo_guess),
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=bool(fit_single_pair_fast_path),
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=bool(stabilize_unitary),
            quality_check_every=quality_check_every,
            quality_check_repair=quality_check_repair,
        )

        # Validate options before mutating a transient layout; both execution
        # paths below restore the order in their finally blocks.
        if layout_plan is not None and not persistent_layout_active:
            layout_current_order = self._reorder_mps_to_logical_order(layout_order_tuple)
            if has_cap:
                # Control dispatch receives already-mapped physical positions,
                # while cap bookkeeping still needs the logical label at each
                # live position in order to compact the shrinking register.
                self._set_site_order(layout_current_order)

        if has_control:
            try:
                return self._run_with_fit_copy_policy(
                    lambda: self._run_segmented(
                        G_seq,
                        where_seq,
                        event_seq,
                        logical_where_seq=logical_where_seq,
                        progbar=progbar,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        measure_renormalize=measure_renormalize,
                        where_is_physical=persistent_layout_active,
                        mode_kwargs=mode_kwargs,
                    ),
                    enabled=timing,
                    finite_check=finite_check,
                    event_count=len(G_seq),
                    sync_device=timing_sync_device,
                )
            finally:
                if layout_current_order is not None:
                    current_order = (
                        tuple(self.logical_order)
                        if has_cap
                        else tuple(layout_current_order)
                    )
                    self._reorder_mps_to_logical_order(
                        tuple(range(int(getattr(self.p, "L", 0)))),
                        current_order=current_order,
                    )
                    self._set_site_order(
                        tuple(range(int(getattr(self.p, "L", 0))))
                    )
                    self._normalize_visible_mps_order()

        try:
            return self._run_with_fit_copy_policy(
                lambda: self._execute_mode(
                    G_seq,
                    where_seq,
                    event_seq,
                    logical_where_seq=logical_where_seq,
                    progbar=progbar,
                    **mode_kwargs,
                ),
                enabled=timing,
                finite_check=finite_check,
                event_count=len(G_seq),
                sync_device=timing_sync_device,
            )
        finally:
            if layout_current_order is not None:
                current_order = (
                    tuple(self.logical_order)
                    if has_cap
                    else tuple(layout_current_order)
                )
                self._reorder_mps_to_logical_order(
                    tuple(range(int(getattr(self.p, "L", 0)))),
                    current_order=current_order,
                )
                self._set_site_order(
                    tuple(range(int(getattr(self.p, "L", 0))))
                )
                self._normalize_visible_mps_order()

    @staticmethod
    def _resolve_shot_replay_options(values, overrides):
        """Share ordinary replay settings with shots; explicit child settings win.

        Shot scheduling, shot RNG, and mode selection belong to the parent.
        This is the single boundary for per-trajectory numerical options.
        """
        names = """n_iter progbar cutoff cutoff_mode k_2q_batch non_unitary
            normalize_every normalize_final normalize_eps submpo_method compression_seed compression_opts
            use_layout_finder layout_order layout_kwargs layout layout_report measure_renormalize
            mix_strict mix_fit_min_iter mix_fit_rtol mix_fit_patience mix_sticky_nonfinite
            fit_min_iter fit_rtol fit_patience fit_block_size fit_adaptive_sweeps
            fit_sweep_sequence fit_layer_size fit_max_span fit_three_site_sweeps
            target_cutoff fit_target_strategy fit_mpo_guess fit_init_strategy
            fit_init_rand_strength fit_init_seed fit_single_pair_fast_path finite_check
            fit_overlap_diagnostics stabilize_unitary fit_stabilize_unitary timing
            timing_sync_device quality_check_every quality_check_repair""".split()
        # Identity sentinels must not cross serialization/deep-copy boundaries.
        resolved = {name: values[name] for name in names if values[name] is not _DEPRECATED_OPTION}
        resolved.update(overrides or {})
        return resolved

    def _run_with_fit_copy_policy(self, executor, *, finite_check=False, **timing_options):
        """Scope copy capabilities and runtime validation to one replay.

        State replacement and control boundaries may change the MPS object,
        but validated replay preserves its array backend. Classify each
        network/array type once, lazily, and release the cache on all exits.
        Calls outside replay always inspect their actual input state for
        copy capabilities. Runtime non-finite validation is opt-in in all modes.
        """
        previous_cache = self._fit_copy_policy_cache
        previous_ranks = self._replay_rank_cache
        previous_arrays = self._replay_array_cache
        previous_finite_check = self._finite_check_enabled
        self._fit_copy_policy_cache = {}
        self._replay_rank_cache = {}
        self._replay_array_cache = weakref.WeakKeyDictionary()
        self._finite_check_enabled = bool(finite_check)
        try:
            if finite_check:
                # Diagnostic validation is off by default. Warn once for this
                # replay; owned FIT instances suppress only their duplicate.
                warnings.warn(
                    "MpsOptimizer finite_check is enabled: this optional "
                    "diagnostic is off by default and is not required for "
                    "normal optimization. It adds tensor/norm checks and can "
                    "synchronize devices; use finite_check=False to avoid "
                    "this overhead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                def checked_executor():
                    result = executor()
                    if not self._mps_data_is_finite(self.p):
                        raise FloatingPointError("Replay produced non-finite MPS data.")
                    return result

                return self._run_with_timing(checked_executor, **timing_options)
            def deferred_checked_executor():
                result = executor()
                self._check_deferred_norm_errors()
                return result

            return self._run_with_timing(deferred_checked_executor, **timing_options)
        finally:
            self._fit_copy_policy_cache = previous_cache
            self._replay_rank_cache = previous_ranks
            self._replay_array_cache = previous_arrays
            self._finite_check_enabled = previous_finite_check

    def _run_with_timing(
        self,
        executor,
        *,
        enabled,
        event_count,
        sync_device=False,
    ):
        """Execute one replay segment and optionally retain wall-clock timing."""
        if not enabled:
            return executor()

        previous_timing_state = self._timing_state
        self._timing_state = {
            "stages": {},
            "fit_steps": [],
            "fit_call_count": 0,
            "sync_device": bool(sync_device),
            "synchronizer": (
                FIT._make_backend_synchronizer(self.p)
                if sync_device
                else None
            ),
        }
        self._sync_timing_device()
        started = time.perf_counter()
        status = "complete"
        try:
            return executor()
        except BaseException:
            status = "failed"
            raise
        finally:
            self._sync_timing_device()
            try:
                final_bond = int(self.p.max_bond())
            except (AttributeError, TypeError, ValueError):
                final_bond = None
            timing_state = self._timing_state
            self.last_run_timing = {
                "status": status,
                "mode": self.mode,
                "mode_alias": self._dmrg_mode_alias,
                "event_count": int(event_count),
                "elapsed_seconds": float(time.perf_counter() - started),
                "final_bond": final_bond,
                "chi": int(self.chi),
                "backend": self.backend,
                "backend_dtype": self.backend_dtype,
                "backend_device": self.backend_device,
                "timing_sync_device": bool(sync_device),
                # The completed timing state has no remaining writer. Transfer
                # its containers directly and keep the defensive copy at the
                # public ``get_run_timing`` boundary.
                "stages": timing_state["stages"],
                "fit_steps": timing_state["fit_steps"],
                "fit_totals": _summarize_fit_timing(
                    timing_state["fit_steps"]
                ),
                # FIT diagnostics are a flat scalar record. Copy the mapping
                # without adding an internal deep-copy cost; the public
                # ``get_run_timing`` boundary owns the defensive deep copy.
                "fit_diagnostics": (
                    None
                    if self._last_dmrg_fit_diagnostics is None
                    else dict(self._last_dmrg_fit_diagnostics)
                ),
                "mix_summary": (
                    deepcopy(self.last_mix_summary)
                    if self.mode == "mix"
                    else None
                ),
            }
            self._timing_state = previous_timing_state

    def _record_timing_stage(self, name, elapsed):
        """Accumulate one completed inclusive stage measurement."""
        stage = self._timing_state["stages"].setdefault(
            str(name),
            {"calls": 0, "elapsed_seconds": 0.0},
        )
        stage["calls"] += 1
        stage["elapsed_seconds"] += elapsed

    def _sync_timing_device(self, value=None):
        """Apply an accelerator barrier only for synchronized profiling."""
        if self._timing_state is None:
            return
        synchronizer = self._timing_state.get("synchronizer")
        if synchronizer is not None:
            target = self.p if value is None else value
            synchronizer.synchronize(target, fallback=self.p)

    def _timed_call(self, name, function, *args, **kwargs):
        """Call ``function`` and time it only during an opt-in run."""
        if self._timing_state is None:
            return function(*args, **kwargs)

        self._sync_timing_device()
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        except BaseException:
            self._sync_timing_device()
            self._record_timing_stage(name, time.perf_counter() - started)
            raise

        # Torch and CuPy synchronize their device/stream globally. JAX work is
        # tied to returned arrays, so wait on the actual stage result rather
        # than an unrelated already-ready tensor from the live MPS.
        self._sync_timing_device(result)
        self._record_timing_stage(name, time.perf_counter() - started)
        return result

    def get_run_timing(self):
        """Return the most recent opt-in replay and stage timing record."""
        return deepcopy(self.last_run_timing)

    def get_fit_diagnostics(self):
        """Return a copy of the latest DMRG/FIT convergence diagnostics.

        The result is ``None`` before a DMRG/FIT update has completed and for
        replay modes that do not use FIT. The returned dictionary is
        independent of the optimizer's internal diagnostic state. Successful
        FIT updates include the ordinary convergence metadata. The optional
        ``fit_overlap_fidelity`` and ``fit_overlap_infidelity`` fields are
        populated only when ``run(fit_overlap_diagnostics=True)`` requests a
        direct contraction of the fitted MPS with the disposable exact FIT
        target. Those fields are target-overlap diagnostics and are
        intentionally separate from norm-survival fields such as
        ``cumulative_fidelity``. If the optional contraction is unavailable
        for a backend, both values are ``None`` and ``fit_overlap_error``
        records the diagnostic failure without rejecting the successful FIT
        update.
        """
        return deepcopy(self._last_dmrg_fit_diagnostics) if self.mode in {"dmrg", "mix"} else None

    def _fit_overlap_diagnostics(self, target, fitted):
        """Return the optional direct FIT-target overlap readout.

        This is deliberately separate from the automatic norm ledger.  The
        norm ledger is available for every compression backend and only reads
        the retained canonical centre.  This contraction compares the final
        FIT MPS with the disposable exact DMRG target, so it is a genuine
        target-state overlap but is specific to DMRG and costs an additional
        contraction.
        """
        # DMRG already paid for the target construction. Keep this diagnostic
        # contraction deterministic and local: the high-level ``auto-hq``
        # optimizer can create a multiprocessing pool, which is unnecessary
        # for a one-window overlap and unavailable in restricted runtimes.
        contraction_opt = self.contraction_opt
        if contraction_opt is None or (
            isinstance(contraction_opt, str)
            and contraction_opt.strip().lower() in {"auto", "auto-hq"}
        ):
            contraction_opt = "greedy"
        try:
            overlap = tn_fidelity(
                target.copy(),
                fitted.copy(),
                contraction_opt=contraction_opt,
            )
            overlap = float(ar.do("real", overlap))
            if not math.isfinite(overlap):
                raise ValueError("FIT target overlap is non-finite")
        except Exception as exc:  # diagnostic only; FIT result remains valid
            return {
                "fit_overlap_fidelity": None,
                "fit_overlap_infidelity": None,
                "fit_overlap_error": f"{type(exc).__name__}: {exc}",
            }
        overlap = min(1.0, max(0.0, overlap))
        return {
            "fit_overlap_fidelity": overlap,
            "fit_overlap_infidelity": float(1.0 - overlap),
            "fit_overlap_error": None,
        }

    def _execute_mode(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        n_iter,
        progbar,
        cutoff,
        cutoff_mode,
        mpo_cutoff_mode=None,
        k_2q_batch,
        normalize_every,
        normalize_final,
        normalize_eps,
        non_unitary,
        submpo_method,
        compression_seed=None,
        compression_opts=None,
        mix_strict=False,
        fit_min_iter=2,
        fit_rtol=None,
        fit_patience=2,
        mix_sticky_nonfinite=False,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        fit_max_span="auto",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        quality_check_every=None,
        quality_check_repair=True,
    ):
        """Dispatch a gate/subMPO segment to the active mode backend.

        This is the mode-specific core of :meth:`run`; ``G_seq``/``where_seq``/
        ``event_seq`` must contain only ``"gate"``/``"submpo"`` events. Control
        events (measure/cap/reset) are handled by :meth:`_run_segmented`.
        """
        # The stream-install boundary already normalized and validated these
        # payloads. Do not repeat that work for each control-delimited segment.
        # Dispatch directly to the mode implementations, which share the same
        # validated stream but have different state contracts:
        # DMRG owns local variational targets, MPO owns Quimb compression,
        # swap/perm own endpoint movement, and exact deliberately bypasses
        # canonical-center bookkeeping. Keeping the branches here prevents a
        # control-event caller from accidentally selecting a gate-only kernel.
        if self.mode == "dmrg":
            self._timed_call("dmrg.prepare", self._prepare_dmrg_state)
            self._timed_call(
                "dmrg.replay",
                self._run_dmrg,
                G_seq,
                where_seq,
                event_seq=event_seq,
                n_iter=n_iter,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                k_2q_batch=k_2q_batch,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                fit_min_iter=fit_min_iter,
                fit_rtol=fit_rtol,
                fit_patience=fit_patience,
                fit_finite_check=finite_check,
                fit_block_size=fit_block_size,
                fit_adaptive_sweeps=fit_adaptive_sweeps,
                fit_sweep_sequence=fit_sweep_sequence,
                fit_max_span=fit_max_span,
                target_cutoff=target_cutoff,
                fit_target_strategy=fit_target_strategy,
                fit_mpo_guess=fit_mpo_guess,
                fit_init_strategy=fit_init_strategy,
                fit_init_rand_strength=fit_init_rand_strength,
                fit_init_seed=fit_init_seed,
                fit_single_pair_fast_path=fit_single_pair_fast_path,
                finite_check=finite_check,
                fit_overlap_diagnostics=fit_overlap_diagnostics,
                stabilize_unitary=stabilize_unitary,
                quality_check_every=quality_check_every,
                quality_check_repair=quality_check_repair,
            )
            return self.p

        if self.mode == "mix":
            self._timed_call(
                "mix.replay",
                self._run_mix,
                G_seq,
                where_seq,
                event_seq,
                logical_where_seq=logical_where_seq,
                n_iter=n_iter,
                k_2q_batch=k_2q_batch,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                submpo_method=submpo_method,
                compression_seed=compression_seed,
                compression_opts=compression_opts,
                mix_strict=mix_strict,
                fit_min_iter=fit_min_iter,
                fit_rtol=fit_rtol,
                fit_patience=fit_patience,
                sticky_nonfinite=mix_sticky_nonfinite,
                fit_block_size=fit_block_size,
                fit_adaptive_sweeps=fit_adaptive_sweeps,
                fit_sweep_sequence=fit_sweep_sequence,
                fit_max_span=fit_max_span,
                target_cutoff=target_cutoff,
                fit_target_strategy=fit_target_strategy,
                fit_init_strategy=fit_init_strategy,
                fit_init_rand_strength=fit_init_rand_strength,
                fit_init_seed=fit_init_seed,
                fit_single_pair_fast_path=fit_single_pair_fast_path,
                finite_check=finite_check,
                fit_overlap_diagnostics=fit_overlap_diagnostics,
                stabilize_unitary=stabilize_unitary,
                non_unitary=non_unitary,
                quality_check_every=quality_check_every,
                quality_check_repair=quality_check_repair,
            )
            return self.p

        if self._is_mpo_mode(self.mode):
            # Report the selected compression method rather than the internal
            # implementation family. The default direct compressor reports
            # ``direct.replay``, including when selected via a legacy alias.
            replay_name = self._mode_mpo_method(self.mode)
            self._timed_call(
                f"{replay_name}.replay",
                self._run_mpo,
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=mpo_cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                submpo_method=submpo_method,
                compression_seed=compression_seed,
                compression_opts=compression_opts,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "swap":
            self._timed_call(
                "swap.replay",
                self._run_swap,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "perm":
            self._timed_call(
                "perm.replay",
                self._run_perm,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "svd":
            self._timed_call(
                "svd.replay",
                self._run_svd,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "exact-batch":
            self._timed_call(
                "exact-batch.replay",
                self._run_exact_batch,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return self.p

        if self.mode == "exact":
            self._timed_call(
                "exact.replay",
                self._run_exact,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return self.p

        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_segmented(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        progbar,
        cutoff,
        cutoff_mode,
        measure_renormalize,
        where_is_physical=False,
        mode_kwargs,
    ):
        """Replay a stream containing measure/cap/reset control events.

        Consecutive ``"gate"``/``"submpo"`` events are grouped into segments run
        through :meth:`_execute_mode` (using the active mode), while control
        events are applied directly to ``self.p`` between segments so the same
        stream works in every mode. ``cap`` events change the MPS length, so
        later event site labels refer to the shortened chain.

        ``where_seq`` holds the execution locations (already mapped into the
        active layout order when a layout is used); ``logical_where_seq`` holds
        the matching user-facing locations for bookkeeping such as recorded
        measurement sites. When no layout is active the two are identical.
        ``where_is_physical`` prevents persistent-layout locations from being
        mapped a second time by the control-event dispatcher.
        """
        if logical_where_seq is None:
            logical_where_seq = where_seq
        seg_G = []
        seg_where = []
        seg_logical_where = []
        seg_event = []

        def flush():
            # A control event is a state boundary: all preceding gates must be
            # committed before its expectation/probability is evaluated, and
            # all following gates must see the collapsed/reset/capped state.
            # Therefore segments are intentionally never allowed to cross a
            # control event, even when the active mode could batch the gates.
            if seg_G:
                self._execute_mode(
                    list(seg_G),
                    list(seg_where),
                    list(seg_event),
                    logical_where_seq=list(seg_logical_where),
                    progbar=progbar,
                    **mode_kwargs,
                )
                seg_G.clear()
                seg_where.clear()
                seg_logical_where.clear()
                seg_event.clear()

        for payload, where, logical_where, event_type in zip(
            G_seq, where_seq, logical_where_seq, event_seq
        ):
            if event_type in _CONTROL_EVENT_NAMES:
                flush()
                # ``where`` may already be physical when a persistent layout
                # is active. ``record_where`` remains logical so measurements
                # and feed-forward records use the user's labels.
                self._apply_control_event(
                    event_type,
                    payload,
                    where,
                    record_where=logical_where,
                    where_is_physical=where_is_physical,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    measure_renormalize=measure_renormalize,
                    mode_kwargs=mode_kwargs,
                )
            else:
                seg_G.append(payload)
                seg_where.append(where)
                seg_logical_where.append(logical_where)
                seg_event.append(event_type)

        flush()
        return self.p

    # ------------------------------------------------------------------ #
    # Control events (measure / cap / reset)
    # ------------------------------------------------------------------ #
    _apply_control_event = _controls._apply_control_event

    _apply_control_event_impl = _controls._apply_control_event_impl

    def _ensure_mps_state(self):
        """Ensure ``self.p`` is a :class:`qtn.MatrixProductState`.

        The exact modes fully contract the state into a single dense tensor;
        control events operate on MPS structure, so rebuild an MPS from the
        physical indices (in ``self.ind_id`` order) when needed.
        """
        p = self.p
        if isinstance(p, qtn.MatrixProductState):
            return p
        outer = set(p.outer_inds())
        ordered = []
        site = 0
        while True:
            ind = self._format_ind(site)
            if ind not in outer:
                break
            ordered.append(ind)
            site += 1
        if len(ordered) != len(outer):
            raise ValueError(
                "cannot rebuild an MPS for a control event: physical indices "
                "are not the standard 1D site-index family."
            )
        dense = p.contract(all, output_inds=ordered, optimize=self.contraction_opt)
        # Quimb splits backend arrays directly. A NumPy bridge here downloads
        # the full state and invalidates the already-validated gate backend.
        arr = dense.data if isinstance(dense, qtn.Tensor) else dense
        mps = qtn.MatrixProductState.from_dense(
            arr,
            ar.shape(arr),
            site_ind_id=self.ind_id,
            site_tag_id=getattr(p, "site_tag_id", "I{}"),
        )
        self.p = self._install_represented_norm(mps)
        self.backend_info()
        # Freshly rebuilt: mark the centre as unknown so the next control event
        # establishes a tracked orthogonality centre (never via a blind scan).
        self.info_c["cur_orthog"] = None
        return self.p

    def _ensure_tracked_center(self):
        """Guarantee ``info_c['cur_orthog']`` is a concrete tracked centre.

        Control events always move the orthogonality centre explicitly rather
        than rescanning with ``calc_current_orthog_center``. When the centre is
        unknown (e.g. a freshly rebuilt exact-mode state, or an ``exact``-mode
        run that never canonicalized), establish one by canonicalizing to site
        ``0`` with a full-span ``cur_orthog`` and record it.
        """
        cur = self.info_c.get("cur_orthog")
        if cur not in (None, "calc"):
            return
        L = int(getattr(self.p, "L", 0))
        if L <= 0:
            return
        self.p.canonize(
            [0],
            cur_orthog=(0, max(0, L - 1)),
            info=self.info_c,
        )
        self.info_c["cur_orthog"] = (0, 0)

    def _state_backend_like(self):
        """Return a representative backend array from ``self.p`` tensor data."""
        return self._state_backend_like_for(self.p)

    @staticmethod
    def _state_backend_info_for(state):
        """Validate and describe the common backend of an MPS-like state."""
        return backend_infer(state)

    def backend_info(self):
        """Return the state-derived backend, dtype, and device diagnostics."""
        info = self._state_backend_info_for(self.p)
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        return info

    @staticmethod
    def _state_backend_like_for(state):
        """Return a representative raw array from an MPS-like state."""
        for tensor in getattr(state, "tensors", ()):
            return tensor.data
        return None

    @staticmethod
    def _backend_mismatch_hint(target_signature):
        """Return concise guidance for preparing a stream payload."""
        if target_signature[0] == "symmray":
            return (
                "Native Symmray states require native Symmray gates with the "
                "matching charge and fermionic metadata; a dense gate cannot "
                "be made safe by a generic cast."
            )
        return (
            "Convert the payload yourself with the same converter used for the "
            "MPS, for example `gate = to_backend(gate)`, before passing it to "
            "MpsOptimizer or set_gates."
        )

    @staticmethod
    def _backend_signatures_compatible(source_signature, target_signature):
        """Return whether two payloads can be used without backend transfer."""
        if source_signature[:1] == target_signature[:1] == ("symmray",):
            # A real native gate is safe to apply to a complex native state:
            # the contraction promotes the result while preserving charge and
            # fermionic metadata. Requiring an exact dtype here rejected
            # ordinary imaginary-time streams whose gates are real-valued.
            return (
                source_signature[0] == target_signature[0]
                and source_signature[2:] == target_signature[2:]
                and np.can_cast(
                    np.dtype(source_signature[1]),
                    np.dtype(target_signature[1]),
                    casting="safe",
                )
            )
        return backend_signatures_compatible(source_signature, target_signature)

    def _validate_gate_stream_backend(
        self,
        gates,
        event_types,
        *,
        state=None,
        path_prefix="stream",
    ):
        """Require user-supplied gate payloads to match the live MPS backend.

        Backend conversion is intentionally an explicit caller operation. This
        validation runs at every public stream/state boundary, while the
        execution path only receives payloads that are already compatible.
        Control events are excluded because their internal dense operators are
        deliberately created and mapped by the optimizer itself. Conditional
        gate actions are checked recursively.
        """
        if not gates:
            return
        if len(gates) != len(event_types):
            raise ValueError(
                "MpsOptimizer backend validation requires payloads and event "
                "types to have the same length."
            )
        state = self.p if state is None else state
        like = self._state_backend_like_for(state)
        if like is None:
            return
        target_signature = _array_backend_signature(like)
        mismatches = []
        for index, (payload, event_type) in enumerate(zip(gates, event_types)):
            path = f"{path_prefix}[{index}]"
            if event_type == "gate":
                source_signature = _array_backend_signature(payload)
                if not self._backend_signatures_compatible(
                    source_signature, target_signature
                ):
                    mismatches.append((path, "gate", source_signature))
            elif event_type == "submpo":
                tensors = tuple(getattr(payload, "tensors", ()))
                source_signatures = {
                    _array_backend_signature(tensor.data) for tensor in tensors
                }
                for source_signature in sorted(
                    (
                        source_signature
                        for source_signature in source_signatures
                        if not self._backend_signatures_compatible(
                            source_signature, target_signature
                        )
                    ),
                    key=repr,
                ):
                    mismatches.append((path, "sub-MPO", source_signature))
            elif event_type == "conditional":
                action = payload.get("action")
                action_gates, _, action_types = _normalize_gate_queue((action,))
                self._validate_gate_stream_backend(
                    action_gates,
                    action_types,
                    state=state,
                    path_prefix=f"{path}.action",
                )

        if not mismatches:
            return
        details = "; ".join(
            f"{path} ({kind}) has {source!r}"
            for path, kind, source in mismatches[:8]
        )
        if len(mismatches) > 8:
            details += f"; ... and {len(mismatches) - 8} more"
        raise TypeError(
            "MpsOptimizer requires every gate and sub-MPO payload to match the "
            f"MPS backend/device and required dtype {target_signature!r} "
            "before use; "
            f"{details}. {self._backend_mismatch_hint(target_signature)}"
        )

    def _to_state_backend(self, array):
        """Return ``array`` cast to the backend and dtype owned by ``self.p``."""
        like = self._state_backend_like()
        if like is None:
            return np.asarray(ar.to_numpy(array), dtype=complex)
        target_signature = _array_backend_signature(like)
        source_signature = _array_backend_signature(array)
        if source_signature == target_signature:
            return array
        if self._is_symmray_array(array) and self._is_symmray_array(like):
            # Symmray arrays deliberately do not implement Autoray's generic
            # ``array(..., like=symmray_array)`` constructor. Their outer
            # object has no scalar dtype either, so the generic dtype fast
            # path below cannot establish compatibility. Native Symmray gates
            # already carry their own block backend and must pass through as
            # graded arrays rather than being rebuilt as dense payloads.
            return array
        if target_signature[0] == "symmray" and source_signature[0] != "symmray":
            raise TypeError(
                "Cannot convert a dense gate/operator payload into a native "
                "Symmray MPS without charge and fermionic metadata. Build the "
                "payload as a Symmray array on the target U1/U1U1 backend."
            )
        converter = infer_backend_converter_from_sample(like)
        if converter is not None:
            return converter(array)
        if target_signature[0] == "numpy":
            return ar.to_numpy(array)
        # Keep the old Autoray fallback for optional/custom dense backends.
        return ar.do("array", array, like=like)

    def to_backend(self, array):
        """Return ``array`` on the backend currently owned by ``self.p``.

        Already-compatible arrays are returned by identity. This public helper
        is intentionally state-derived so replacing the MPS with :meth:`set_p`
        automatically changes the target backend without stale converter state.
        """
        return self._to_state_backend(array)

    _control_operator = _controls._control_operator

    _one_site_projector = _controls._one_site_projector

    _pauli_operator = _controls._pauli_operator

    _build_pauli_projector_submpo = _controls._build_pauli_projector_submpo

    _apply_submpo_with_method = _controls._apply_submpo_with_method

    _apply_dense_operator = _controls._apply_dense_operator

    _state_expectation = _controls._state_expectation

    _state_operator_expectation = _controls._state_operator_expectation

    _measurement_probabilities = _controls._measurement_probabilities

    _pauli_amplitude_probabilities = _controls._pauli_amplitude_probabilities

    _scaled_norm_value = staticmethod(_controls._scaled_norm_value)

    _control_state_norm = _controls._control_state_norm

    _recanonize_center = _controls._recanonize_center

    _finish_measurement_center = _controls._finish_measurement_center

    _apply_measure_event = _controls._apply_measure_event

    _apply_basis_flip = _controls._apply_basis_flip

    _apply_reset_event = _controls._apply_reset_event

    _apply_measure_reset_event = _controls._apply_measure_reset_event

    _apply_cap_event = _controls._apply_cap_event

    _validate_event_stream_for_run = _controls._validate_event_stream_for_run

    _real_float = staticmethod(_norm._real_float)

    _start_unitary_norm_tracking = _norm._start_unitary_norm_tracking

    _check_deferred_norm_errors = _norm._check_deferred_norm_errors

    _accumulate_norm_survival = _norm._accumulate_norm_survival

    _norm_event_to_host = _norm._norm_event_to_host

    _invalidate_unitary_norm_baseline = _norm._invalidate_unitary_norm_baseline

    _fidelity_ratio_from_norms = staticmethod(_norm._fidelity_ratio_from_norms)

    _unitary_norm_overshoot_tolerance = _norm._unitary_norm_overshoot_tolerance

    _record_norm_event = _norm._record_norm_event

    _compact_norm_summary = _norm._compact_norm_summary

    def norm_diagnostics(self, *, include_history=True):
        """Return automatic norm-based compression diagnostics.

        ``local_fidelity`` and ``cumulative_fidelity`` are fidelities measured
        from retained canonical-centre norms. They are compression-survival
        proxies, not directional overlaps with an independently supplied
        target state. DMRG target overlap, when available, is reported
        separately by :meth:`get_fit_diagnostics`.
        Born probabilities for stochastic branches remain in ``norm_events``
        and do not reduce cumulative compression fidelity.

        ``state_norm`` and ``norm`` are the live represented MPS norm.
        ``cumulative_norm`` is instead the square root of
        ``cumulative_fidelity``. The latter is a retained-compression proxy,
        not a second reading of the live state norm.

        ``include_history=False`` omits historical arrays and incrementally
        summarizes append-only events. Do not edit committed event dictionaries
        when using this polling path. Full historical output remains the default.
        """
        return _norm.norm_diagnostics(self, include_history=include_history)

    _accumulate_exponent = staticmethod(_norm._accumulate_exponent)

    @staticmethod
    def _class_norm_includes_exponent(p):
        """Return whether the installed quimb ``norm`` already uses exponent."""
        if not hasattr(p, "exponent"):
            return False

        exponent_orig = p.exponent
        try:
            p.exponent = 0.0
            norm0 = type(p).norm(p)
            p.exponent = 1.0
            norm1 = type(p).norm(p)
        except Exception:
            return False
        finally:
            p.exponent = exponent_orig

        denom = ar.do("abs", norm0)
        try:
            if MpsOptimizer._real_float(denom) == 0.0:
                return False
            ratio = MpsOptimizer._real_float(ar.do("abs", norm1) / denom)
        except Exception:
            return False
        return abs(ratio - 10.0) < 1.0e-8

    @staticmethod
    def _install_represented_norm(p):
        """Make ``p.norm()`` include PEPSY's accumulated base-10 exponent.

        Some quimb versions apply ``TensorNetwork.exponent`` in MPS ``norm``
        already, while others ignore it. PEPSY uses exponent to keep
        non-unitary working data normalized while preserving the represented
        state scale, so optimizer-managed states get a small instance-local
        wrapper only when the installed quimb needs one.
        """
        if (
            (not hasattr(p, "norm"))
            or (not hasattr(p, "exponent"))
            or getattr(p, "_pepsy_norm_includes_exponent", False)
        ):
            return p

        norm_cache_key = type(p)
        norm_includes_exponent = _NORM_INCLUDES_EXPONENT_CACHE.get(
            norm_cache_key,
            _MISSING,
        )
        if norm_includes_exponent is _MISSING:
            norm_includes_exponent = MpsOptimizer._class_norm_includes_exponent(p)
            _NORM_INCLUDES_EXPONENT_CACHE[norm_cache_key] = norm_includes_exponent

        if norm_includes_exponent:
            p._pepsy_norm_includes_exponent = True
            return p

        def _norm_with_exponent(self, output_inds=None, squared=False, **contract_opts):
            raw_norm = type(self).norm(
                self,
                output_inds=output_inds,
                squared=squared,
                **contract_opts,
            )
            exponent = getattr(self, "exponent", 0.0)
            if exponent == 0:
                return raw_norm
            scale_power = 2 * exponent if squared else exponent
            factor = MpsOptimizer._scaled_norm_value(1., scale_power)
            return raw_norm * factor

        p.norm = types.MethodType(_norm_with_exponent, p)
        p._pepsy_norm_includes_exponent = True
        return p

    _normalize_span = staticmethod(_norm._normalize_span)

    _canonical_span_norm = _norm._canonical_span_norm

    _retained_center_norm_impl = _norm._retained_center_norm_impl

    _retained_center_norm = _norm._retained_center_norm

    def _build_norm_target(
        self,
        p,
        gate,
        where,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="mps",
        copy=True,
        info=None,
    ):
        """Build an exact pre-output-compression target.

        ``layered`` stores dense gates as small lazy tensors rather than
        repeatedly SVD-compressing an ever-growing target MPS. ``mps`` keeps
        the legacy routed target and remains the native-safe Symmray route.
        """
        target_strategy = self._validate_fit_target_strategy(target_strategy)
        if target_strategy == "auto":
            target_strategy = (
                "mps"
                if self._replay_has_symmray_data(p) or p.isfermionic()
                else "layered"
            )

        p_target = self._inherit_replay_array_kind(p.copy(), p) if copy else p
        target_info = {} if info is None else info
        if len(where) == 1:
            self._apply_layered_target_gate(
                p_target,
                gate,
                where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return p_target

        if target_strategy == "layered":
            return self._apply_layered_target_gate(
                p_target,
                gate,
                where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )

        if self._replay_has_symmray_data(p_target):
            # Keep one native tensor per MPS site. A lazy ``split-gate`` target
            # is useful for one-site contractions but leaves extra gate tensors
            # carrying overlapping site tags, which does not define a unique
            # two-site middle bond. Auto-swap uses Symmray's graded split path
            # and remains uncapped because no ``max_bond`` is supplied.
            return self._build_symmray_auto_swap_target(
                p_target,
                gate,
                where,
                cutoff,
                cutoff_mode,
                copy=False,
                info=target_info,
            )

        p_target.gate_nonlocal_(
            gate,
            where,
            dims=self._infer_gate_dims(gate, where),
            max_bond=None,
            info=target_info,
            method="direct",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return p_target

    @staticmethod
    def _build_lazy_submpo_target(p, submpo, where, *, copy=True):
        """Build an exact lazy target by attaching a sub-MPO to ``p``."""
        p_target = p.copy() if copy else p
        p_target.gate_with_submpo_(
            submpo,
            where=where,
            method="lazy",
            inplace_mpo=False,
        )
        return p_target

    def _fit_window_copy_supported(self, p):
        """Inspect array capabilities once before sharing exterior data."""
        if self._replay_has_symmray_data(p) or p.isfermionic():
            return False
        return all(
            ar.infer_backend(t.data) in {"numpy", "torch", "jax"}
            for t in p.tensors
        )

    def _fit_window_copy_policy(self, p):
        """Resolve the replay's existing active-window ownership policy."""
        cache = self._fit_copy_policy_cache
        if cache is None:
            supported = self._fit_window_copy_supported(p)
        else:
            key = (type(p), type(p[0].data))
            if key not in cache:
                cache[key] = self._fit_window_copy_supported(p)
            supported = cache[key]
        return supported

    def _fit_rollback_snapshot(self, p, where):
        """Retain an untouched dense state until the actual guess is known.

        Dense target/guess preparation does not mutate p. If the selected
        guess aliases p, the caller must copy this snapshot before FIT runs.
        Native warm-starts may mutate p earlier and still require a full copy.
        """
        if self._fit_window_copy_policy(p):
            return p
        return self._copy_fit_window_state(p, where)

    def _copy_fit_window_state(self, p, where):
        """Copy active FIT data, retaining read-only exterior arrays.

        Tensor metadata is independent, and every active array is owned.
        Quimb canonicalization replaces exterior arrays in the private copy.
        Native/unknown array types retain the conservative full deep copy.
        """
        supported = self._fit_window_copy_policy(p)
        if not supported:
            return self._inherit_replay_array_kind(p.copy(deep=True), p)
        start, stop = min(where), max(where)
        copied = p.copy()
        for site in range(start, stop + 1):
            tensor = copied[site]
            # Autoray's Torch ``copy`` detaches the autograd graph. Clone
            # directly so a disposable FIT state retains parameter gradients.
            data = tensor.data
            data = data.clone() if ar.infer_backend(data) == "torch" else ar.do("copy", data)
            tensor.modify(
                data=data, left_inds=tensor.left_inds
            )
        return self._inherit_replay_array_kind(copied, p)

    def _build_compression_fit_guess(
        self,
        p,
        gate,
        where,
        *,
        method,
        cutoff,
        cutoff_mode,
        seed=None,
    ):
        """Build a disposable Quimb-compressed guess from one gate."""
        method = quimb_fit_guess_method(method, p)
        result = guess(
            self._copy_fit_window_state(p, where),
            gate,
            where,
            inplace=True,
            method=method,
            dims=self._infer_gate_dims(gate, where),
            chi=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            seed=seed,
        )
        return self._inherit_replay_array_kind(result, p)

    def _build_compression_submpo_fit_guess(
        self,
        p,
        submpo,
        where,
        *,
        method,
        cutoff,
        cutoff_mode,
        seed=None,
    ):
        """Build a disposable compressed FIT guess from a sub-MPO."""
        guess_mps = self._copy_fit_window_state(p, where)
        self._apply_submpo_with_method(
            guess_mps,
            submpo,
            where,
            method=method,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info={},
            seed=seed,
        )
        return guess_mps

    def _build_compression_batch_fit_guess(
        self,
        p,
        gates,
        wheres,
        *,
        method,
        cutoff,
        cutoff_mode,
        seed=None,
    ):
        """Build a disposable Quimb-compressed guess for a gate batch."""
        guess_mps = self._copy_fit_window_state(
            p, tuple(site for where in wheres for site in where)
        )
        for i, (gate, where) in enumerate(zip(gates, wheres)):
            guess(
                guess_mps,
                gate,
                where,
                method=method,
                dims=self._infer_gate_dims(gate, where),
                chi=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                seed=None if seed is None else int(seed) + i,
                inplace=True,
            )
        return guess_mps

    def _native_src_fit_guess_enabled(
        self,
        strategy,
        fit_mpo_guess,
    ):
        """Return whether the native Symmray SRC-style guess is requested."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        if requested_strategy == "auto":
            requested_strategy = _DEFAULT_FIT_INIT_STRATEGY
        if requested_strategy != "guess_src":
            return False
        if (
            not fit_mpo_guess
            and self._dmrg_mode_alias in {"dmrg1", "dmrg3"}
            and str(strategy).strip().lower() in {"auto", _DEFAULT_FIT_INIT_STRATEGY}
        ):
            return False
        return True

    def _build_native_randomized_fit_guess(
        self,
        p,
        gates,
        wheres,
        *,
        cutoff,
        cutoff_mode,
        seed,
    ):
        """Build a native Symmray SRC-style FIT guess.

        Quimb's dense SRC compressor cannot preserve Symmray charge sectors or
        fermionic dummy-mode metadata. Symmray does expose a native randomized
        truncated SVD, however, so apply the gate sequence on a disposable
        native MPS using ``svd:rand`` at every two-site split. This provides the
        same randomized compressed warm-start role without constructing a dense
        gate, MPO, or random dense tensor.
        """
        if not self._replay_has_symmray_data(p):
            raise TypeError(
                "native randomized FIT guesses require native Symmray data."
            )

        guess_mps = p.copy(deep=True)
        guess_info = {}
        # Cumulative discarded-weight bounds need the full sector spectra.
        # Symmray's eager randomized driver cannot honor them. Preserve the
        # requested accuracy policy using a deterministic native split for
        # this disposable guess; compatible policies still use randomized SVD.
        cumulative_cutoff = cutoff_mode in {
            3, 4, 5, 6, "sum2", "rsum2", "sum1", "rsum1",
        } and cutoff is not None and cutoff > 0.0
        split_method = "svd" if cumulative_cutoff else "svd:rand"
        for index, (gate, where) in enumerate(zip(gates, wheres)):
            where = tuple(int(site) for site in where)
            if len(where) == 1:
                self._apply_gate(
                    guess_mps,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
            elif len(where) == 2:
                self._apply_symmray_auto_swap_gate(
                    guess_mps,
                    gate,
                    where,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=self.chi,
                    info=guess_info,
                    method=split_method,
                    **({"seed": int(seed) + index} if not cumulative_cutoff else {}),
                )
            else:
                raise ValueError(
                    "Native randomized FIT guesses support one- or two-site "
                    "gates only."
                )

        return guess_mps, {
            "backend": "symmray",
            "method": split_method,
            "fallback_reason": "cumulative_cutoff" if cumulative_cutoff else None,
            "seed": int(seed),
            "gate_count": len(gates),
        }

    def _build_native_direct_fit_guess(
        self,
        p,
        gates,
        wheres,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Build a chi-capped native direct guess without densification."""
        guess_mps = p.copy(deep=True)
        guess_info = {}
        for gate, where in zip(gates, wheres):
            where = tuple(int(site) for site in where)
            if len(where) == 1:
                self._apply_gate(
                    guess_mps,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
            elif len(where) == 2:
                self._apply_symmray_auto_swap_gate(
                    guess_mps,
                    gate,
                    where,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=self.chi,
                    info=guess_info,
                )
            else:
                raise ValueError(
                    "Native direct FIT guesses support one- or two-site "
                    "gates only."
                )

        return guess_mps, {
            "backend": "symmray-auto-swap",
            "method": "direct",
            "gate_count": len(gates),
        }

    @staticmethod
    def _fit_random_data(data, shape, *, strength, rng):
        """Generate deterministic random data on ``data``'s backend."""
        dtype_name = str(getattr(data, "dtype", "float64"))
        if "complex64" in dtype_name:
            random_dtype = np.complex64
        elif "complex" in dtype_name:
            random_dtype = np.complex128
        elif "float32" in dtype_name:
            random_dtype = np.float32
        else:
            random_dtype = np.float64
        return backend_random_array(
            shape,
            like=data,
            dtype=random_dtype,
            scale=float(strength),
            rng=rng,
        )

    def _build_randomized_fit_guess(
        self,
        p,
        where,
        *,
        block_size,
        rand_strength,
        expand=True,
        seed=0,
    ):
        """Prepare a dense FIT guess with deterministic random initialization.

        The exact gate target remains built from the unmodified current MPS.
        ``expand=False`` perturbs only existing tensors, while ``expand=True``
        also adds random directions on active bonds below their physical/``chi``
        ceiling. Native Symmray and fermionic states retain their graded
        sector-growth path and are never padded with dense random data.
        """
        info = {
            "enabled": False,
            "rand_strength": float(rand_strength),
            "bonds": [],
            "sites": [],
            "expanded": bool(expand),
            "reason": None,
        }
        if int(block_size) not in {2, 3}:
            info["reason"] = "one_site_fit"
            return p, info
        if self._replay_has_symmray_data(p) or p.isfermionic():
            info["reason"] = "native_sector_growth"
            return p, info
        if float(rand_strength) == 0.0:
            info["reason"] = "disabled"
            return p, info

        xmin, xmax = self._normalize_span(where)
        guess = self._inherit_replay_array_kind(p.copy(deep=True), p)
        like = p[xmin].data
        try:
            # Keep the old-Autoray fallback paired with backend_random_array.
            ar.get_lib_fn(ar.infer_backend(like), "random.array")
            ar.get_lib_fn(ar.infer_backend(like), "random.default_rng")
        except (AttributeError, ImportError, KeyError, LookupError):
            rng = np.random.default_rng(int(seed))
        else:
            # Autoray also infers the generator device for Torch from like.
            # A NumPy Generator cannot be passed to Torch's manual_seed.
            rng = ar.do("random.default_rng", int(seed), like=like)
        bonds = []
        if expand:
            target_sizes = FIT._active_bond_rank_targets(  # pylint: disable=protected-access
                p,
                xmin,
                xmax,
                self.chi,
            )
            if target_sizes is None:
                info["reason"] = "no_active_rank_targets"
                return p, info
            for site, target_size in zip(range(xmin, xmax), target_sizes):
                current_size = int(p.bond_size(site, site + 1))
                target_size = int(target_size)
                if current_size < target_size:
                    bonds.append((site, current_size, target_size))
            if not bonds:
                info["reason"] = "already_at_target"
                return p, info

            by_target = {}
            for site, current_size, target_size in bonds:
                by_target.setdefault(target_size, []).append(
                    (site, current_size, target_size)
                )
            for target_size, target_bonds in by_target.items():
                bond_inds = [
                    guess.bond(site, site + 1)
                    for site, _, _ in target_bonds
                ]
                qtn.TensorNetwork.expand_bond_dimension(
                    guess,
                    target_size,
                    mode="zeros",
                    inds_to_expand=bond_inds,
                    inplace=True,
                )
                for site, current_size, _ in target_bonds:
                    bond = guess.bond(site, site + 1)
                    for tensor in guess.tensors:
                        if bond not in tensor.inds:
                            continue
                        axis = tensor.inds.index(bond)
                        old_slices = [slice(None)] * tensor.ndim
                        old_slices[axis] = slice(0, current_size)
                        old_data = tensor.data[tuple(old_slices)]
                        random_shape = list(tensor.shape)
                        random_shape[axis] = target_size - current_size
                        random_data = self._fit_random_data(
                            tensor.data,
                            random_shape,
                            strength=rand_strength,
                            rng=rng,
                        )
                        tensor.modify(
                            data=ar.do(
                                "concatenate",
                                (old_data, random_data),
                                axis=axis,
                            )
                        )
        else:
            for site in range(xmin, xmax + 1):
                tensor = guess[site]
                random_data = self._fit_random_data(
                    tensor.data,
                    tensor.shape,
                    strength=rand_strength,
                    rng=rng,
                )
                tensor.modify(data=ar.do("add", tensor.data, random_data))
                info["sites"].append(int(site))

        guess_info = {}
        self.canonize_mps(guess, (xmin, xmax), info=guess_info)
        info["enabled"] = True
        info["bonds"] = [
            {
                "bond": int(site),
                "current_rank": int(current_size),
                "target_rank": int(target_size),
                "new_rank": int(guess.bond_size(site, site + 1)),
            }
            for site, current_size, target_size in bonds
        ]
        return guess, info

    def _prepare_fit_initial_guess(
        self,
        p,
        gates,
        wheres,
        *,
        block_size,
        strategy,
        fit_mpo_guess,
        rand_strength,
        seed,
        cutoff,
        cutoff_mode,
        submpo=False,
        native_source=None,
    ):
        """Select the disposable FIT guess without changing the live MPS."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        info = {
            "enabled": False,
            "rand_strength": float(rand_strength),
            "bonds": [],
            "sites": [],
            "expanded": False,
            "reason": "direct",
        }
        result = {
            "fit_guess": p,
            "strategy": "direct",
            "requested_strategy": requested_strategy,
            "guess_method": None,
            "guess_used": False,
            "svd_guess_used": False,
            "guess_backend": None,
            "native_randomized_guess_used": False,
            "random_initialization": info,
        }
        if self._replay_has_symmray_data(p) or p.isfermionic():
            if (
                not submpo
                and native_source is not None
                and self._native_src_fit_guess_enabled(
                    requested_strategy,
                    fit_mpo_guess,
                )
            ):
                fit_guess, native_info = self._build_native_randomized_fit_guess(
                    native_source,
                    gates,
                    wheres,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    seed=seed,
                )
                info.update(native_info)
                info["reason"] = "native_src"
                result["fit_guess"] = fit_guess
                result["strategy"] = "guess_src"
                result["guess_method"] = "src"
                result["guess_used"] = True
                result["svd_guess_used"] = True
                result["guess_backend"] = f"symmray-{native_info['method']}"
                result["native_randomized_guess_used"] = native_info["method"] == "svd:rand"
                return result
            if not submpo and requested_strategy in {
                "guess_direct",
                "svd_guess",
            }:
                fit_guess, native_info = self._build_native_direct_fit_guess(
                    p if native_source is None else native_source,
                    gates,
                    wheres,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                )
                info.update(native_info)
                info["reason"] = "native_direct"
                result["fit_guess"] = fit_guess
                result["strategy"] = requested_strategy
                result["guess_method"] = "direct"
                result["guess_used"] = True
                result["svd_guess_used"] = True
                result["guess_backend"] = "symmray-auto-swap"
                return result
            info["reason"] = (
                "native_sector_growth"
                if int(block_size) in {2, 3}
                else "native_one_site_fit"
            )
            result["strategy"] = "direct"
            return result

        start, stop = self._normalize_span(wheres[0] if len(wheres) == 1 else (
            min(site for where in wheres for site in where),
            max(site for where in wheres for site in where),
        ))
        needs_growth = requested_strategy in {"random", "random_expand"} and not (
            FIT._active_bonds_at_rank_targets(p, start, stop, self.chi)  # pylint: disable=protected-access
        )
        is_named_svd_window = len(gates) == 1 and self._dmrg_mode_alias in {
            "dmrg1",
            "dmrg3",
        }
        if requested_strategy == "auto":
            selected_strategy = _DEFAULT_FIT_INIT_STRATEGY
        else:
            # An explicit Quimb guess is a warm-start policy, not only a rank
            # enrichment policy. Keep applying it after the active bonds have
            # reached their attainable rank so the one-site FIT phase receives
            # the same SRC-prepared state as the expansion phase.
            selected_strategy = (
                requested_strategy
                if requested_strategy.startswith("guess_")
                or requested_strategy == "svd_guess"
                else requested_strategy if needs_growth else "direct"
            )
        if (
            not fit_mpo_guess
            and is_named_svd_window
            and requested_strategy in {"auto", _DEFAULT_FIT_INIT_STRATEGY}
        ):
            selected_strategy = "direct"

        if selected_strategy == "svd_guess":
            guess_method = "direct"
        elif selected_strategy.startswith("guess_"):
            guess_method = selected_strategy[len("guess_") :]
        else:
            guess_method = None

        if guess_method is not None:
            if len(gates) == 1:
                if submpo:
                    fit_guess = self._build_compression_submpo_fit_guess(
                        p,
                        gates[0],
                        wheres[0],
                        method=guess_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        seed=seed,
                    )
                else:
                    fit_guess = self._build_compression_fit_guess(
                        p,
                        gates[0],
                        wheres[0],
                        method=guess_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        seed=seed,
                    )
            else:
                fit_guess = self._build_compression_batch_fit_guess(
                    p,
                    gates,
                    wheres,
                    method=guess_method,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    seed=seed,
                )
            result["fit_guess"] = fit_guess
            result["strategy"] = "svd_guess"
            if selected_strategy != "svd_guess":
                result["strategy"] = selected_strategy
            result["guess_method"] = guess_method
            result["guess_used"] = True
            result["svd_guess_used"] = True
            info["reason"] = selected_strategy
            return result

        fit_guess, random_info = self._build_randomized_fit_guess(
            p,
            (start, stop),
            block_size=block_size,
            rand_strength=rand_strength,
            expand=selected_strategy == "random_expand",
            seed=int(seed),
        ) if selected_strategy in {"random", "random_expand"} else (p, info)
        if selected_strategy == "direct":
            info["reason"] = "already_at_target"
        result["fit_guess"] = fit_guess
        result["strategy"] = selected_strategy
        result["random_initialization"] = random_info
        return result

    _normalize_every_interval = staticmethod(_norm._normalize_every_interval)

    _accumulate_exponent_log10 = staticmethod(_norm._accumulate_exponent_log10)

    _event_old_norm_from_log10 = staticmethod(_norm._event_old_norm_from_log10)

    _normalize_orthog_tensors = _norm._normalize_orthog_tensors

    _normalize_in_canonical_range = _norm._normalize_in_canonical_range

    _normalize_canonical_center = _norm._normalize_canonical_center

    _normalize_canonical_center_impl = _norm._normalize_canonical_center_impl

    _maybe_normalize_after_step = _norm._maybe_normalize_after_step

    _maybe_normalize_final = _norm._maybe_normalize_final

    @staticmethod
    def _format_progress_scalar(value):
        """Format displayed progress scalar with stable precision."""
        return f"{MpsOptimizer._real_float(value):.6f}"

    def _cumulative_fidelity(self):
        """Return displayed cumulative fidelity measured from norms.

        This is not the live MPS norm and is not a target-state overlap. It is
        the product of the local squared canonical-centre norm-survival ratios
        accumulated in ``_norm_log_survival``.
        """
        return self._real_float(ar.do("exp", self._norm_log_survival))

    @staticmethod
    def _collect_dmrg_batch(
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        max_span=None,
    ):
        """Collect a DMRG batch with an optional spatial-span cap."""
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = where_seq[idx]
            gate = G_seq[idx]
            if max_span is not None and batch_where:
                sites = [site for previous in batch_where for site in previous]
                sites.extend(where)
                proposed_span = max(sites) - min(sites) + 1
                if proposed_span > int(max_span):
                    break
            if len(where) == 1:
                batch_where.append(where)
                batch_G.append(gate)
            elif len(where) == 2:
                batch_where.append(where)
                batch_G.append(gate)
                two_qubit_in_batch += 1
            else:
                raise ValueError("Each gate location must have one or two sites.")
            idx += 1

        return batch_G, batch_where, two_qubit_in_batch, idx

    def _build_dmrg_batch_target(
        self,
        p,
        batch_G,
        batch_where,
        target_cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="mps",
    ):
        """Apply a DMRG target block without output-compression truncation."""
        p_g = self._inherit_replay_array_kind(p.copy(), p)
        for gate, where in zip(batch_G, batch_where):
            if len(where) == 1:
                self._apply_layered_target_gate(
                    p_g,
                    gate,
                    where,
                    cutoff=target_cutoff,
                    cutoff_mode=cutoff_mode,
                )
            else:
                p_g = self._build_norm_target(
                    p_g,
                    gate,
                    where,
                    target_cutoff,
                    cutoff_mode,
                    target_strategy=target_strategy,
                    copy=False,
                )
        return p_g

    def _stabilize_unitary_compression_state(
        self,
        p,
        where,
        target_norm,
        *,
        current_norm=None,
        center_site=None,
        restore=True,
    ):
        """Record compression norm survival and optionally restore its scale.

        The removed scale is deliberately *not* accumulated into ``exponent``
        for unitary evolution: it is approximation loss, not physical
        non-unitary evolution. ``restore=False`` preserves the historical
        un-stabilized output while still recording the norm change.
        """
        span = self._normalize_span(where)
        if current_norm is None or center_site is None:
            current_norm = self._canonical_span_norm(p, span)
            center = int(span[1])
        else:
            center = int(center_site)
            if not span[0] <= center <= span[1]:
                raise ValueError(
                    f"FIT center {center} is outside active span {span}."
                )
        backend_norms = (
            not self._finite_check_enabled
            and ar.infer_backend(current_norm) in {"torch", "jax", "cupy"}
        )
        if backend_norms:
            current_value = ar.do("stop_gradient", ar.do("abs", current_norm))
            if ar.infer_backend(target_norm) != ar.infer_backend(current_value):
                target_norm = ar.do("full_like", current_value, target_norm)
            target_value = ar.do("stop_gradient", ar.do("abs", target_norm))
            zero = ar.do("logical_or", current_value == 0.0, target_value == 0.0)
            self._pending_zero_norm = (
                zero if self._pending_zero_norm is None
                else ar.do("logical_or", self._pending_zero_norm, zero)
            )
            self._record_norm_event(
                "unitary_compression", expected_norm=target_value,
                observed_norm=current_value, where=span,
            )
            if restore:
                # Keep normalization differentiable; only the diagnostic copy
                # is detached. Zero states are rejected at the replay boundary.
                denominator = ar.do("where", current_value == 0.0, 1.0, current_norm)
                p[center].modify(data=p[center].data * (target_norm / denominator))
            self._unitary_previous_norm = target_value if restore else current_value
            self._record_orthog_span(p, (center, center))
            return

        current_float = self._real_float(ar.do("abs", current_norm))
        target_float = self._real_float(ar.do("abs", target_norm))
        if (
            current_float == 0.0
            or target_float == 0.0
            or (self._finite_check_enabled and (
                not np.isfinite(current_float) or not np.isfinite(target_float)
            ))
        ):
            raise FloatingPointError(
                "Cannot stabilize a unitary FIT state with a zero or non-finite norm."
            )
        self._record_norm_event(
            "unitary_compression",
            expected_norm=target_float,
            observed_norm=current_float,
            where=span,
        )
        if restore:
            p[center].modify(data=p[center].data * (target_norm / current_norm))
        else:
            self._unitary_previous_norm = current_float
        self._record_orthog_span(p, (center, center))
        if restore:
            self._unitary_previous_norm = target_float

    def _stabilize_unitary_fit_state(self, *args, **kwargs):
        """Compatibility wrapper for the generalized compression stabilizer."""
        return self._stabilize_unitary_compression_state(*args, **kwargs)

    def _run_mix_mpo_step(
        self,
        gate,
        where,
        event_type,
        *,
        step,
        cutoff,
        cutoff_mode,
        submpo_method,
        compression_seed=None,
        compression_opts=None,
        stabilize_unitary,
    ):
        """Apply one mixed-mode step with the selected Quimb compressor."""
        self._run_mpo(
            [gate],
            [where],
            [event_type],
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=None,
            normalize_final=False,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            compression_opts=compression_opts,
            stabilize_unitary=stabilize_unitary,
        )

    def _run_mix_mpo_batch(
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        steps,
        cutoff,
        cutoff_mode,
        submpo_method,
        compression_seed=None,
        compression_opts=None,
        stabilize_unitary,
    ):
        """Apply a mixed-mode fallback batch with Quimb compression."""
        self._run_mpo(
            G_seq,
            where_seq,
            event_seq,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=None,
            normalize_final=False,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            compression_opts=compression_opts,
            stabilize_unitary=stabilize_unitary,
        )
        active_where = (
            min(site for where in where_seq for site in where),
            max(site for where in where_seq for site in where),
        )
        self._validate_mix_norm(active_where, operation="MPO batch")

    def _run_mix_dmrg(self, *args, fit_block_size, **kwargs):
        """Run the fixed mixed one-site DMRG schedule."""
        if int(fit_block_size) != 1:
            raise ValueError("mixed DMRG requires fit_block_size=1.")
        old_dmrg_alias = self._dmrg_mode_alias
        old_dmrg_block_size = self._dmrg_mode_block_size
        self._dmrg_mode_alias = "dmrg1"
        self._dmrg_mode_block_size = 1
        kwargs["fit_block_size"] = 1
        try:
            return self._run_dmrg(*args, **kwargs)
        finally:
            self._dmrg_mode_alias = old_dmrg_alias
            self._dmrg_mode_block_size = old_dmrg_block_size

    def _run_mix_dmrg_step(
        self,
        gate,
        where,
        *,
        step,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        cutoff,
        cutoff_mode,
        fit_block_size=1,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_init_strategy="guess_direct",
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
    ):
        """Apply one mixed-mode step through the DMRG backend."""
        self._run_mix_dmrg(
            [gate],
            [where],
            n_iter=n_iter,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=1,
            normalize_every=None,
            normalize_final=False,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=finite_check,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=stabilize_unitary,
        )
        self._validate_mix_norm(where, operation="DMRG step")

    def _run_mix_dmrg_batch(
        self,
        G_seq,
        where_seq,
        *,
        steps,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        cutoff,
        cutoff_mode,
        fit_block_size=1,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_init_strategy="guess_direct",
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
    ):
        """Apply a contiguous two-site batch through the DMRG backend."""
        self._run_mix_dmrg(
            G_seq,
            where_seq,
            n_iter=n_iter,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=len(G_seq),
            normalize_every=None,
            normalize_final=False,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=finite_check,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=stabilize_unitary,
        )
        active_where = (
            min(site for where in where_seq for site in where),
            max(site for where in where_seq for site in where),
        )
        self._validate_mix_norm(active_where, operation="DMRG batch")
        final_bond = self._effective_max_bond(self.p)
        if final_bond > int(self.chi):
            raise RuntimeError(
                "DMRG batch exceeded the mixed-mode chi bond limit."
            )
        return final_bond

    def _collect_mix_dmrg_batch(
        self,
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        target_sizes=None,
        allow_short=False,
        max_span=None,
    ):
        """Collect contiguous DMRG-ready gates for one mixed transaction."""
        batch_G = []
        batch_where = []
        idx = start_idx
        while (
            idx < len(G_seq)
            and len(batch_G) < int(k_2q_batch)
            and len(where_seq[idx]) == 2
            and (
                allow_short
                or not self._mix_active_bond_is_short(
                    where_seq[idx], target_sizes=target_sizes
                )
            )
        ):
            if max_span is not None and batch_where:
                sites = [site for previous in batch_where for site in previous]
                sites.extend(where_seq[idx])
                proposed_span = max(sites) - min(sites) + 1
                if proposed_span > int(max_span):
                    break
            batch_G.append(G_seq[idx])
            batch_where.append(where_seq[idx])
            idx += 1
        return batch_G, batch_where, idx

    def _mix_state_snapshot(self):
        """Capture mutable optimizer state before a trial mixed update."""
        return {
            "p": self.p,
            "p_exponent": getattr(self.p, "exponent", None),
            "info_c": deepcopy(self.info_c),
            "unitary_previous_norm": self._unitary_previous_norm,
            "norm_log_survival": self._norm_log_survival,
            "pending_zero_norm": self._pending_zero_norm,
            "lengths": {
                "normalizations": len(self.normalizations),
                "norm_events": len(self.norm_events),
            },
        }

    @staticmethod
    def _copy_mix_tensor_data(data):
        """Copy one tensor payload without changing its array backend."""
        clone = getattr(data, "clone", None)
        if callable(clone):
            return clone()
        copy_data = getattr(data, "copy", None)
        if callable(copy_data):
            return copy_data()
        return deepcopy(data)

    def _mix_transaction_sites(self, where, info):
        """Return the tensors that a transactional trial can modify.

        FIT canonicalizes from the tracked center to the active window before
        optimizing it.  Copy that connecting interval, rather than the whole
        MPS.  If the center is unknown, retain the old full-copy safety rule.
        """
        L = int(getattr(self.p, "L", 0))
        if L <= 0:
            return ()
        active = (min(where), max(where))
        current = info.get("cur_orthog") if isinstance(info, dict) else None
        if (
            not isinstance(current, (tuple, list))
            or len(current) != 2
            or not all(isinstance(site, Integral) for site in current)
        ):
            return tuple(range(L))
        start = min(active[0], int(current[0]))
        stop = max(active[1], int(current[1]))
        return tuple(range(max(0, start), min(L - 1, stop) + 1))

    def _copy_mix_trial(self, committed_p, where, info):
        """Make a shallow MPS copy with an isolated mutable trial window."""
        sites = self._mix_transaction_sites(where, info)
        try:
            trial_p = committed_p.copy(deep=False)
        except (AttributeError, TypeError, ValueError):
            trial_p = self._inherit_replay_array_kind(
                committed_p.copy(deep=True), committed_p
            )
            return trial_p, tuple(range(int(committed_p.L)))
        for site in sites:
            tensor = trial_p[site]
            tensor.modify(
                data=self._copy_mix_tensor_data(tensor.data),
                left_inds=tensor.left_inds,
            )
        return self._inherit_replay_array_kind(trial_p, committed_p), sites

    def _validate_mix_norm(self, where, *, operation):
        """Validate mixed-mode commit norms only when finite_check is enabled.

        Mixed replay already leaves a tracked canonical center after each
        compression. Reading that center's Frobenius norm is sufficient for
        the normal health check and avoids scanning every tensor payload on
        every transaction. Full tensor-data checks remain available through
        ``quality_check_every``.
        """
        if not self._finite_check_enabled:
            return None
        try:
            retained_norm, _ = self._retained_center_norm(self.p, where)
            norm_value = self._real_float(ar.do("abs", retained_norm))
            exponent = float(getattr(self.p, "exponent", 0.0))
        except Exception as exc:
            raise FloatingPointError(
                f"{operation} retained norm could not be validated: {exc}"
            ) from exc
        if (
            norm_value <= 0.0
            or not np.isfinite(norm_value)
            or not np.isfinite(exponent)
        ):
            raise FloatingPointError(
                f"{operation} produced a zero or non-finite retained norm."
            )
        return norm_value

    def _restore_mix_state(self, snapshot):
        """Restore a mixed-mode transaction without changing caller identity."""
        self.p = snapshot["p"]
        if snapshot["p_exponent"] is not None:
            self.p.exponent = snapshot["p_exponent"]
        self.info_c = snapshot["info_c"]
        self._unitary_previous_norm = snapshot["unitary_previous_norm"]
        self._norm_log_survival = snapshot["norm_log_survival"]
        self._pending_zero_norm = snapshot["pending_zero_norm"]
        for attr, length in snapshot["lengths"].items():
            del getattr(self, attr)[length:]
        self._norm_summary_cache = None

    def _commit_mix_trial(self, committed_p, trial_p, *, sites=None):
        """Commit a successful trial while honoring ``inplace=True``."""
        if not self.inplace:
            self.p = trial_p
            return
        if trial_p is not committed_p:
            if len(committed_p.tensors) != len(trial_p.tensors):
                raise RuntimeError(
                    "mixed-mode trial changed the number of MPS tensors; "
                    "cannot preserve inplace object identity."
                )
            # DMRG can legitimately replace virtual bonds, especially when a
            # non-nearest gate is routed through a trial MPS. Preserve the
            # caller-owned network object, but adopt the valid trial graph
            # rather than rejecting its fresh bond labels. Take detached
            # snapshots first because the shallow trial copy can still share
            # tensors outside its transaction window.
            requested_sites = (
                tuple(range(len(committed_p.tensors)))
                if sites is None
                else tuple(sites)
            )
            changed_sites = set()
            for site in range(len(committed_p.tensors)):
                committed_tensor = committed_p[site]
                trial_tensor = trial_p[site]
                if committed_p.site_ind(site) != trial_p.site_ind(site):
                    raise RuntimeError(
                        "mixed-mode trial changed a physical MPS index; "
                        "cannot preserve inplace object identity."
                    )
                if committed_tensor.tags != trial_tensor.tags:
                    raise RuntimeError(
                        "mixed-mode trial changed MPS tensor tags; "
                        "cannot preserve inplace object identity."
                    )
                if committed_tensor.inds != trial_tensor.inds:
                    changed_sites.add(site)
            commit_sites = tuple(sorted(set(requested_sites) | changed_sites))
            trial_records = []
            for site in commit_sites:
                trial_tensor = trial_p[site]
                trial_records.append(
                    (
                        self._copy_mix_tensor_data(trial_tensor.data),
                        tuple(trial_tensor.inds),
                        trial_tensor.tags,
                        trial_tensor.left_inds,
                    )
                )
            # Include every changed endpoint in addition to the transaction
            # window so both sides of a newly labelled virtual bond are
            # updated. Unchanged tensors outside that window are not copied.
            for site, (data, inds, tags, left_inds) in zip(
                commit_sites,
                trial_records,
            ):
                committed_p[site].modify(
                    data=data,
                    inds=inds,
                    tags=tags,
                    left_inds=left_inds,
                )
            reset_cached = getattr(committed_p, "reset_cached_properties", None)
            if callable(reset_cached):
                reset_cached()
            committed_p.exponent = trial_p.exponent
        self.p = committed_p

    @staticmethod
    def _resolve_legacy_fit_option(
        *,
        canonical_name,
        canonical_value,
        canonical_default,
        legacy_name,
        legacy_value,
    ):
        """Resolve one deprecated FIT option without silently mixing policies.

        Canonical controls have readable defaults in the public signature.
        A supplied legacy value can replace that default, preserving old call
        sites. If the caller also selects a different non-default canonical
        value, fail early instead of guessing which convergence or
        stabilization policy they intended.
        """
        if legacy_value is _DEPRECATED_OPTION:
            return canonical_value
        warnings.warn(
            f"{legacy_name} is deprecated; use {canonical_name} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if (
            canonical_value != canonical_default
            and canonical_value != legacy_value
        ):
            raise ValueError(
                f"{canonical_name} and deprecated {legacy_name} specify "
                "different values; pass only the canonical option."
            )
        return legacy_value

    def _resolve_fit_rtol(self, value):
        """Return a validated dtype-aware FIT stopping tolerance."""
        if value == "auto":
            dtype = str(self.backend_dtype).lower()
            if "16" in dtype:
                return 1e-3
            if "32" in dtype or "complex64" in dtype:
                return 1e-5
            return 1e-9
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fit_rtol must be 'auto', a non-negative number, or None."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "fit_rtol must be 'auto', a non-negative number, or None."
            )
        return value

    def _resolve_cutoff(self, value):
        """Return a validated truncation cutoff, including ``"auto"``."""
        if value == "auto":
            return dtype_auto_cutoff(self.backend_dtype)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cutoff must be 'auto' or a non-negative number."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("cutoff must be 'auto' or a non-negative number.")
        return value

    @staticmethod
    def _resolve_cutoff_mode(value, *, preserve_mpo_default=False):
        """Resolve ``cutoff_mode='auto'`` without leaking it to Quimb.

        Ordinary Pepsy compression uses relative discarded squared weight,
        which is the natural state-fidelity metric. Quimb MPO methods retain
        their own defaults when requested through ``auto``; this matters for
        the density-matrix compressor, whose native default is ``rsum1``.
        ``None`` remains a compatibility spelling for the same policy.
        """
        if value is None or (
            isinstance(value, str) and value.strip().lower() == "auto"
        ):
            if preserve_mpo_default:
                return None
            return _DEFAULT_CUTOFF_MODE
        return value

    @staticmethod
    def _resolve_fit_max_span(value, k_2q_batch):
        """Resolve the maximum inclusive spatial span of a FIT batch."""
        if value is None:
            return None
        if value == "auto":
            # A local layer of ``k`` neighboring two-site gates spans at most
            # roughly ``2 * k`` sites. Long-range first gates are always kept,
            # even when their individual span exceeds this soft cap.
            return max(3, 2 * int(k_2q_batch) + 1)
        if not isinstance(value, Integral) or int(value) < 2:
            raise ValueError(
                "fit_max_span must be 'auto', None, or an integer >= 2."
            )
        return int(value)

    @staticmethod
    def _resolve_quality_check_every(value):
        """Resolve the optional replay quality-check interval."""
        if value is None or value is False or value == 0:
            return None
        if value is True:
            return 1
        if not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(
                "quality_check_every must be None, bool, or a positive integer."
            )
        return int(value)

    def _run_quality_check(self, step, where, *, repair):
        """Record finite-data and canonical-gauge health at a replay step."""
        p = self.p
        finite = bool(self._mps_data_is_finite(p))
        if not finite:
            record = {
                "step": int(step),
                "where": tuple(where),
                "finite": False,
                "canonical_ok": False,
                "repaired": False,
            }
            self.quality_checks.append(record)
            raise FloatingPointError(
                f"quality check at step {step} found non-finite MPS data."
            )

        span = self._current_orthog(p)
        record = {
            "step": int(step),
            "where": tuple(where),
            "finite": True,
            "orthog_span": tuple(span),
            "max_bond": int(p.max_bond()),
            "repaired": False,
        }
        try:
            left_count, right_count = p.count_canonized()
            expected_canonical_sites = max(int(p.L) - 1, 0)
            canonical_ok = (
                int(left_count) + int(right_count)
                >= expected_canonical_sites
            )
            record.update(
                {
                    "canonical_left": int(left_count),
                    "canonical_right": int(right_count),
                    "expected_canonical_sites": expected_canonical_sites,
                    "canonical_ok": bool(canonical_ok),
                }
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            record.update(
                {
                    "canonical_ok": None,
                    "canonical_error": f"{type(exc).__name__}: {exc}",
                }
            )
            canonical_ok = True

        if not canonical_ok and repair:
            self.canonize_mps(p, span[1])
            self._record_orthog_span(p, (span[1], span[1]))
            record["repaired"] = True
            left_count, right_count = p.count_canonized()
            record["canonical_left_after"] = int(left_count)
            record["canonical_right_after"] = int(right_count)
            record["canonical_ok"] = (
                int(left_count) + int(right_count)
                >= max(int(p.L) - 1, 0)
            )
        self.quality_checks.append(record)
        return record

    def _maybe_run_quality_check(self, step, where, every, *, repair):
        """Run a periodic quality check when the requested interval is due."""
        if every is None or int(step) % int(every):
            return None
        return self._timed_call(
            "quality.check",
            self._run_quality_check,
            step,
            where,
            repair=repair,
        )

    @staticmethod
    def _mix_error_is_nonfinite(exc):
        """Return whether an exception reports NaN or infinite numerics."""
        if isinstance(exc, FloatingPointError):
            return True
        if "linalg" in type(exc).__name__.casefold():
            return True
        message = str(exc).casefold()
        return any(
            marker in message
            for marker in ("nan", "infs", "non-finite", "nonfinite", "infinite")
        )

    def _mix_target_bond_dimensions(self):
        """Reuse physical rank ceilings while register geometry is unchanged.

        The returned private list is read-only to callers. Actual bond sizes
        are deliberately not cached: compression can change them each gate.
        State, cap, and layout changes invalidate the replay cache; chi and
        length are also part of the key. Standalone calls always recompute.
        """
        cache = self._replay_rank_cache
        if cache is None:
            return self._compute_mix_target_bond_dimensions()
        key = (int(getattr(self.p, "L", 0)), int(self.chi))
        if key not in cache:
            cache.clear()
            cache[key] = self._compute_mix_target_bond_dimensions()
        return cache[key]

    def _compute_mix_target_bond_dimensions(self):
        """Return each bond's ``chi``-capped physical rank ceiling."""
        L = int(getattr(self.p, "L", 0))
        if L <= 1:
            return []
        dims = []
        for site in range(L):
            try:
                dim = int(self.p.phys_dim(site))
            except (AttributeError, TypeError, ValueError):
                dim = int(self.p.ind_size(self._format_ind(site)))
            dims.append(dim)

        left_caps = []
        rank = 1
        for site in range(L - 1):
            rank = min(int(self.chi), rank * dims[site])
            left_caps.append(rank)

        right_caps = [1] * (L - 1)
        rank = 1
        for site in range(L - 1, 0, -1):
            rank = min(int(self.chi), rank * dims[site])
            right_caps[site - 1] = rank
        return [
            min(int(self.chi), left, right)
            for left, right in zip(left_caps, right_caps)
        ]

    def _mix_active_bond_is_short(self, where, *, target_sizes=None):
        """Return whether an active bond is below its attainable target."""
        if self.chi <= 1 or len(where) < 2:
            return False
        xmin, xmax = min(where), max(where)
        if target_sizes is None:
            target_sizes = self._mix_target_bond_dimensions()
        return any(
            int(self.p.bond_size(site, site + 1)) < target_sizes[site]
            for site in range(xmin, xmax)
        )

    def _run_mix(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        sticky_nonfinite,
        k_2q_batch=1,
        fit_max_span=None,
        mix_strict=False,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        submpo_method="direct",
        compression_seed=None,
        compression_opts=None,
        fit_block_size=1,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_init_strategy="guess_direct",
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        non_unitary=False,
        quality_check_every=None,
        quality_check_repair=True,
    ):
        """Apply phase-independent guess-direct/DMRG1 with an MPO fallback.

        Every eligible multi-site gate builds a disposable chi-capped direct
        guess, then runs one-site FIT against a separately constructed exact
        target. This is the same while bonds are growing and after they reach
        ``chi``. Non-unitary trajectory branches use the explicit MPO fallback
        because mixed FIT is defined only for unitary working-norm updates.
        """
        mix_started = (
            time.perf_counter() if self._timing_state is not None else None
        )
        if non_unitary:
            # Mixed FIT's transactional contract is unitary. A selected Kraus
            # branch is still a valid noisy gate, so keep the requested mode
            # but use its physical MPO compression backend for this step.
            self._run_mpo(
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=None,
                normalize_final=False,
                non_unitary=True,
                submpo_method=submpo_method,
                compression_seed=compression_seed,
                compression_opts=compression_opts,
                stabilize_unitary=False,
            )
            return self.p
        if any(event_type == "submpo" for event_type in event_seq):
            raise ValueError("mode='mix' currently supports gate streams only.")

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="mix",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["mix"],
            )

        if logical_where_seq is None:
            logical_where_seq = where_seq
        if len(logical_where_seq) != len(where_seq):
            raise ValueError("logical and execution gate streams must have equal length.")

        target_sizes = self._mix_target_bond_dimensions()
        target_bond = max(target_sizes, default=1)
        mix_step_offset = len(self.mix_history)
        mpo_steps = sum(event["backend"] == "mpo" for event in self.mix_history)
        dmrg_steps = sum(event["backend"] == "dmrg" for event in self.mix_history)
        fallback_steps = sum(
            event.get("reason", "").startswith("dmrg_fallback")
            for event in self.mix_history
        )

        def append_entries(entries):
            self.mix_history.extend(entries)
            if pbar is not None:
                final = entries[-1]
                postfix = {
                    "backend": final["backend"],
                    "mpo": mpo_steps,
                    "dmrg": dmrg_steps,
                    "fallback": fallback_steps,
                    "bond": f"{final['end_bond']}/{self.chi}",
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                }
                pbar.set_postfix(postfix)
                pbar.update(len(entries))

        mpo_state_needs_check = False
        mpo_state_check_where = None

        def check_pending_mpo_state():
            """Validate one completed contiguous direct/MPO block."""
            nonlocal mpo_state_needs_check, mpo_state_check_where
            if not mpo_state_needs_check:
                return
            self._validate_mix_norm(
                mpo_state_check_where,
                operation="MPO step",
            )
            mpo_state_needs_check = False
            mpo_state_check_where = None

        # A completed transaction's maximum is also the next one's initial
        # maximum. Keep this local to the segment, and discard it after any
        # quality check that could repair/change tensor shapes.
        current_bond = None
        idx = 0
        try:
            while idx < len(G_seq):
                gate = G_seq[idx]
                where = where_seq[idx]
                event_type = event_seq[idx]
                logical_where = logical_where_seq[idx]
                if len(where) not in {1, 2}:
                    raise ValueError("Each gate location must have one or two sites.")

                step = mix_step_offset + idx + 1
                start_bond = (
                    self._effective_max_bond(self.p)
                    if current_bond is None else current_bond
                )
                use_mpo = (
                    len(where) == 1
                    or self._mix_dmrg_disabled_reason is not None
                )
                if use_mpo:
                    self._run_mix_mpo_step(
                        gate,
                        where,
                        event_type,
                        step=step,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        submpo_method=submpo_method,
                        compression_seed=compression_seed,
                        compression_opts=compression_opts,
                        stabilize_unitary=stabilize_unitary,
                    )
                    mpo_steps += 1
                    mpo_state_needs_check = True
                    mpo_state_check_where = where
                    if len(where) == 1:
                        reason = "one_site_exact"
                    else:
                        reason = "dmrg_disabled_nonfinite"
                    current_bond = self._effective_max_bond(self.p)
                    entry = {
                        "step": int(step),
                        "where": tuple(logical_where),
                        "execution_where": tuple(where),
                        "start_bond": start_bond,
                        "target_bond": int(target_bond),
                        "backend": "mpo",
                        "reason": reason,
                        "end_bond": current_bond,
                    }
                    if self._mix_dmrg_disabled_reason is not None:
                        entry["dmrg_disabled_reason"] = (
                            self._mix_dmrg_disabled_reason
                        )
                        entry["failed_sweep"] = self._mix_dmrg_failed_sweep
                    append_entries([entry])
                    idx += 1
                    if self._maybe_run_quality_check(
                        step,
                        where,
                        quality_check_every,
                        repair=quality_check_repair,
                    ) is not None:
                        current_bond = None
                    continue

                check_pending_mpo_state()
                batch_G, batch_where, next_idx = self._collect_mix_dmrg_batch(
                    G_seq,
                    where_seq,
                    idx,
                    k_2q_batch,
                    target_sizes=target_sizes,
                    allow_short=True,
                    max_span=fit_max_span,
                )
                batch_steps = [
                    mix_step_offset + position + 1
                    for position in range(idx, next_idx)
                ]
                batch_logical_where = logical_where_seq[idx:next_idx]
                if len(batch_where) == 1:
                    active_where = batch_where[0]
                else:
                    xmin = min(min(where_i) for where_i in batch_where)
                    xmax = max(max(where_i) for where_i in batch_where)
                    active_where = (xmin, xmax)
                snapshot = self._mix_state_snapshot()
                committed_p = snapshot["p"]
                self._last_dmrg_fit_diagnostics = None
                try:
                    # DMRG/FIT can mutate its input before it raises or
                    # produces invalid data. Isolate the canonicalization
                    # path and active window, while sharing untouched MPS
                    # tensors with the committed state. Keep the committed
                    # state as the MPO fallback target.
                    trial_p, transaction_sites = self._copy_mix_trial(
                        committed_p,
                        active_where,
                        snapshot["info_c"],
                    )
                    trial_p = self._install_represented_norm(trial_p)
                    self.p = trial_p
                    self.info_c = deepcopy(snapshot["info_c"])
                    # The DMRG executor prepares the window before fitting.
                    # Repeating that preparation here rescans identical ranks.
                    trial_final_bond = self._run_mix_dmrg_batch(
                        batch_G,
                        batch_where,
                        steps=batch_steps,
                        n_iter=n_iter,
                        fit_min_iter=fit_min_iter,
                        fit_rtol=fit_rtol,
                        fit_patience=fit_patience,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        fit_block_size=fit_block_size,
                        fit_adaptive_sweeps=fit_adaptive_sweeps,
                        fit_sweep_sequence=fit_sweep_sequence,
                        target_cutoff=target_cutoff,
                        fit_target_strategy=fit_target_strategy,
                        fit_init_strategy=fit_init_strategy,
                        fit_init_rand_strength=fit_init_rand_strength,
                        fit_init_seed=fit_init_seed,
                        fit_single_pair_fast_path=fit_single_pair_fast_path,
                        finite_check=finite_check,
                        fit_overlap_diagnostics=fit_overlap_diagnostics,
                        stabilize_unitary=stabilize_unitary,
                    )
                    fit_diagnostics = deepcopy(
                        self._last_dmrg_fit_diagnostics or {}
                    )
                    self._commit_mix_trial(
                        committed_p,
                        self.p,
                        sites=transaction_sites,
                    )
                except Exception as exc:  # fallback is the point of mix mode
                    self._restore_mix_state(snapshot)
                    if mix_strict:
                        raise
                    fit_diagnostics = deepcopy(
                        self._last_dmrg_fit_diagnostics or {}
                    )
                    if sticky_nonfinite and self._mix_error_is_nonfinite(exc):
                        self._mix_dmrg_disabled_reason = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._mix_dmrg_failed_sweep = getattr(
                            exc,
                            "fit_iteration",
                            fit_diagnostics.get("iterations") or None,
                        )
                    fallback_trial, fallback_sites = self._copy_mix_trial(
                        committed_p,
                        active_where,
                        snapshot["info_c"],
                    )
                    fallback_trial = self._install_represented_norm(fallback_trial)
                    try:
                        self.p = fallback_trial
                        self.info_c = deepcopy(snapshot["info_c"])
                        self._run_mix_mpo_batch(
                            batch_G,
                            batch_where,
                            event_seq[idx:next_idx],
                            steps=batch_steps,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            submpo_method=submpo_method,
                            compression_seed=compression_seed,
                            compression_opts=compression_opts,
                            stabilize_unitary=stabilize_unitary,
                        )
                        self._commit_mix_trial(
                            committed_p,
                            self.p,
                            sites=fallback_sites,
                        )
                        mpo_state_needs_check = False
                    except BaseException:
                        self._restore_mix_state(snapshot)
                        raise
                    mpo_steps += len(batch_G)
                    fallback_steps += len(batch_G)
                    final_bond = self._effective_max_bond(self.p)
                    current_bond = final_bond
                    entries = []
                    for offset, (step_i, where_i, logical_i) in enumerate(
                        zip(batch_steps, batch_where, batch_logical_where)
                    ):
                        entries.append(
                            {
                                "step": int(step_i),
                                "where": tuple(logical_i),
                                "execution_where": tuple(where_i),
                                "start_bond": start_bond,
                                "target_bond": int(target_bond),
                                "backend": "mpo",
                                "reason": (
                                    "dmrg_fallback"
                                    if offset == 0
                                    else "dmrg_fallback_batch"
                                ),
                                "fallback_error": f"{type(exc).__name__}: {exc}",
                                "fit_iterations": fit_diagnostics.get(
                                    "iterations", 0
                                ),
                                "fit_converged": fit_diagnostics.get(
                                    "converged", False
                                ),
                                "fit_relative_change": fit_diagnostics.get(
                                    "relative_change"
                                ),
                                "dmrg_disabled": (
                                    self._mix_dmrg_disabled_reason is not None
                                ),
                                "failed_sweep": self._mix_dmrg_failed_sweep,
                                "end_bond": final_bond,
                            }
                        )
                    append_entries(entries)
                    idx = next_idx
                    if self._maybe_run_quality_check(
                        batch_steps[-1],
                        active_where,
                        quality_check_every,
                        repair=quality_check_repair,
                    ) is not None:
                        current_bond = None
                    continue
                except BaseException:
                    self._restore_mix_state(snapshot)
                    raise

                dmrg_steps += len(batch_G)
                # Reuse the maximum already measured to enforce chi before
                # commit; commit preserves the validated trial's dimensions.
                final_bond = (
                    self._effective_max_bond(self.p)
                    if trial_final_bond is None else trial_final_bond
                )
                current_bond = final_bond
                entries = []
                for offset, (step_i, where_i, logical_i) in enumerate(
                    zip(batch_steps, batch_where, batch_logical_where)
                ):
                    entries.append(
                        {
                            "step": int(step_i),
                            "where": tuple(logical_i),
                            "execution_where": tuple(where_i),
                            "start_bond": start_bond,
                            "target_bond": int(target_bond),
                            "backend": "dmrg",
                            "reason": (
                                "guess_direct_dmrg1"
                                if offset == 0
                                else "dmrg_batch"
                            ),
                            "fit_iterations": fit_diagnostics.get("iterations"),
                            "fit_converged": fit_diagnostics.get("converged"),
                            "fit_relative_change": fit_diagnostics.get(
                                "relative_change"
                            ),
                            "end_bond": final_bond,
                        }
                    )
                append_entries(entries)
                idx = next_idx
                if self._maybe_run_quality_check(
                    batch_steps[-1],
                    active_where,
                    quality_check_every,
                    repair=quality_check_repair,
                ) is not None:
                    current_bond = None
            check_pending_mpo_state()
        finally:
            if pbar is not None:
                pbar.close()

        self.last_mix_summary = {
            "elapsed_seconds": (
                None
                if mix_started is None
                else float(time.perf_counter() - mix_started)
            ),
            "mpo_steps": int(mpo_steps),
            "dmrg_steps": int(dmrg_steps),
            "fallback_steps": int(fallback_steps),
            "final_bond": (
                self._effective_max_bond(self.p)
                if current_bond is None else current_bond
            ),
            "chi": int(self.chi),
            "target_bond": int(target_bond),
            "dmrg_disabled": self._mix_dmrg_disabled_reason is not None,
            "dmrg_disabled_reason": self._mix_dmrg_disabled_reason,
            "failed_sweep": self._mix_dmrg_failed_sweep,
        }

    def _run_fit_gate(self, fit, **kwargs):
        """Run the gate-restricted FIT solver.

        ``FIT.run_gate`` is the MpsOptimizer DMRG kernel. It is the
        gate-window specialization of ``FIT.run_eff``: both reuse cached
        environments, but ``run_gate`` keeps the variational update inside
        the interval touched by the current gate or batch. Calling
        ``run_eff`` here would refit the complete MPS after every gate and
        would no longer implement local DMRG-style compression.
        """
        kwargs.setdefault(
            "two_site_transition_sweeps", 1 if self._dmrg_mode_alias == "dmrg3" else 0
        )
        # FIT is owned by this optimizer and discarded after the call. The
        # active replay already warned about diagnostics; keep every check
        # enabled without emitting another warning for each gate or segment.
        fit._finite_check_warning_handled = self._finite_check_enabled
        if self._timing_state is None:
            return fit.run_gate(**kwargs)
        kwargs.setdefault("timing", True)
        kwargs.setdefault(
            "timing_sync_device",
            bool(self._timing_state.get("sync_device", False)),
        )
        fit_index = int(self._timing_state["fit_call_count"])
        self._timing_state["fit_call_count"] += 1
        try:
            return self._timed_call("dmrg.fit", fit.run_gate, **kwargs)
        finally:
            # FIT is optimizer-owned and discarded after this call. Move its
            # records into the run-level collector instead of copying every
            # nested per-site dictionary multiple times.
            for record in fit._take_timing_records():
                record["fit_index"] = fit_index
                record["record_index"] = len(self._timing_state["fit_steps"])
                self._timing_state["fit_steps"].append(record)

    def _run_dmrg_measurement(
        self,
        submpo,
        where,
        *,
        n_iter,
        cutoff,
        cutoff_mode,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        fit_block_size,
        fit_adaptive_sweeps,
        fit_sweep_sequence,
        target_cutoff,
        fit_target_strategy,
        fit_mpo_guess,
        fit_init_strategy,
        fit_init_rand_strength,
        fit_init_seed,
        fit_single_pair_fast_path,
        measurement_index,
        finite_check=False,
        fit_overlap_diagnostics=False,
    ):
        """Apply a multi-site projective measurement through DMRG FIT.

        The unnormalized post-measurement state is first attached as a lazy
        sub-MPO target. FIT then compresses that target on the measurement
        span, using the same block schedule and SRC warm-start policy as an
        ordinary DMRG gate. The target remains unnormalized so the caller can
        record the Born-branch norm before renormalizing the live state.
        """
        p = self.p
        # ``where`` is the full support of the Pauli string for the sub-MPO,
        # while the DMRG/FIT window is represented by its endpoint span.
        where_sites = tuple(int(site) for site in where)
        span = (min(where_sites), max(where_sites))
        requested_block_size = min(
            int(fit_block_size),
            span[1] - span[0] + 1,
        )
        self._validate_dmrg1_iteration_budget(
            p,
            span,
            n_iter=n_iter,
            block_size=requested_block_size,
        )
        self._prepare_fit_window(span, block_size=fit_block_size)
        self.canonize_mps(p, span)
        state_snapshot = self._fit_rollback_snapshot(p, span)
        info_snapshot = dict(self.info_c)

        fit = None
        fit_error = None
        fit_initialization = {
            "strategy": "direct",
            "requested_strategy": fit_init_strategy,
            "guess_method": None,
            "guess_used": False,
            "svd_guess_used": False,
            "random_initialization": {
                "enabled": False,
                "reason": "direct",
            },
        }
        active_fit_block_size = self._dmrg_fit_block_size(
            p,
            span,
            fit_block_size,
        )
        adaptive_sweeps = (
            2 if self._dmrg_mode_alias == "dmrg1" else int(fit_adaptive_sweeps)
        )
        adaptive_rank_schedule = self._dmrg_mode_alias not in {
            "dmrg1",
            "dmrg2",
            "dmrg3",
        } and not (
            self._dmrg_mode_alias is None
            and active_fit_block_size in {2, 3}
            and span[1] - span[0] + 1 > active_fit_block_size
        )
        target_strategy = self._validate_fit_target_strategy(fit_target_strategy)
        _ = target_cutoff  # lazy targets are kept exact until FIT compression

        try:
            # Keep the projected state unnormalized throughout FIT. The final
            # center norm is the branch-amplitude measurement used by the
            # caller; renormalization belongs only to the post-collapse finish
            # so a DMRG approximation cannot replace the physical branch
            # probability with a post-localizer value.
            p_target = self._timed_call(
                "dmrg.target",
                self._build_lazy_submpo_target,
                p,
                submpo,
                span,
            )
            self._inherit_replay_array_kind(p_target, p)
            fit_initialization = self._timed_call(
                "dmrg.fit_guess",
                self._prepare_fit_initial_guess,
                p,
                (submpo,),
                (span,),
                block_size=active_fit_block_size,
                strategy=fit_init_strategy,
                fit_mpo_guess=fit_mpo_guess,
                rand_strength=fit_init_rand_strength,
                seed=(
                    int(fit_init_seed)
                    + 1000003 * int(measurement_index)
                    + 1009 * int(span[0])
                    + int(span[1])
                ),
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                submpo=True,
            )
            if state_snapshot is p and fit_initialization["fit_guess"] is p:
                state_snapshot = self._copy_fit_window_state(p, span)
            fit = FIT(
                p_target,
                p=fit_initialization["fit_guess"],
                cutoffs=cutoff,
                contraction_opt=self.contraction_opt,
                retag=False,
                range_int=[span[0], span[1]],
                inplace=True,
                copy_target=False,
            )
            self._run_fit_gate(
                fit,
                n_iter=n_iter,
                verbose=False,
                min_iter=fit_min_iter,
                rtol=fit_rtol,
                patience=fit_patience,
                finite_check=finite_check,
                block_size=active_fit_block_size,
                sweep_sequence=fit_sweep_sequence,
                max_bond=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                single_pair_fast_path=fit_single_pair_fast_path,
                adaptive_block_sweeps=adaptive_sweeps,
                adaptive_until_rank=adaptive_rank_schedule,
                final_one_site_sweeps=0,
                collect_split_diagnostics=False,
            )
        except Exception as exc:  # retain the direct compressed fallback
            # FIT failures are transactional for this multi-site window.
            # Restore both tensor data and canonical metadata before using the
            # direct MPO fallback; otherwise partial variational writeback
            # could be silently combined with the fallback target.
            fit_error = exc

        fit_norm = None if fit is None else fit.final_norm
        fit_center = None if fit is None else fit.final_center_site
        if fit_error is not None:
            self.p = self._install_represented_norm(state_snapshot)
            self.info_c = info_snapshot
            try:
                self._apply_submpo_with_method(
                    self.p,
                    submpo,
                    span,
                    method="direct",
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    info=self.info_c,
                )
            except Exception:
                raise fit_error.with_traceback(fit_error.__traceback__)
            current_span = self._current_orthog(self.p)
            center = (
                int(current_span[0])
                if current_span[0] == current_span[1]
                else int(span[1])
            )
            if current_span[0] != current_span[1]:
                self.canonize_mps(self.p, center)
            projected_norm = self._real_float(
                ar.do("abs", self.p[center].norm())
            )
            self._last_dmrg_fit_diagnostics = {
                "iterations": 0 if fit is None else int(fit.iterations_run),
                "converged": False if fit is None else bool(fit.converged),
                "convergence_reason": (
                    None if fit is None else fit.convergence_reason
                ),
                "relative_change": (
                    None if fit is None else fit.last_relative_change
                ),
                "center_site": center,
                "block_size": int(active_fit_block_size),
                "adaptive_sweeps": 0 if fit is None else int(fit.adaptive_sweeps_run),
                "one_site_refinement_sweeps": (
                    0 if fit is None else int(fit.one_site_sweeps_run)
                ),
                "native_fermionic_warm_start": False,
                "mpo_fit_guess_used": False,
                "svd_guess_used": False,
                "guess_used": False,
                "guess_method": None,
                "fit_init_strategy": "direct",
                "fit_init_strategy_requested": fit_init_strategy,
                "random_initialization": fit_initialization.get(
                    "random_initialization"
                ),
                "target_strategy": target_strategy,
                "fit_overlap_diagnostics": bool(fit_overlap_diagnostics),
                "target_representation": "lazy_submpo",
                "backend": "mpo",
                "fallback": True,
                "fallback_reason": "fit_exception",
                "fallback_error": f"{type(fit_error).__name__}: {fit_error}",
                "fit_overlap_fidelity": None,
                "fit_overlap_infidelity": None,
                "fit_overlap_error": None,
            }
            return projected_norm, center

        self.p = self._install_represented_norm(fit.p)
        center = int(fit_center) if fit_center is not None else int(span[1])
        if fit_center is None:
            self.canonize_mps(self.p, center)
        self._record_orthog_span(self.p, (center, center))
        projected_norm = (
            self._real_float(ar.do("abs", fit_norm))
            if fit_norm is not None
            else self._real_float(ar.do("abs", self.p[center].norm()))
        )
        fit_overlap = (
            self._fit_overlap_diagnostics(p_target, fit.p)
            if fit_overlap_diagnostics
            else {}
        )
        self._last_dmrg_fit_diagnostics = {
            "iterations": int(fit.iterations_run),
            "converged": bool(fit.converged),
            "convergence_reason": fit.convergence_reason,
            "relative_change": fit.last_relative_change,
            "center_site": center,
            "block_size": int(active_fit_block_size),
            "adaptive_sweeps": int(fit.adaptive_sweeps_run),
            "one_site_refinement_sweeps": int(fit.one_site_sweeps_run),
            "native_fermionic_warm_start": False,
            "mpo_fit_guess_used": bool(fit_initialization["svd_guess_used"]),
            "svd_guess_used": bool(fit_initialization["svd_guess_used"]),
            "guess_used": bool(fit_initialization["guess_used"]),
            "guess_method": fit_initialization["guess_method"],
            "fit_init_strategy": fit_initialization["strategy"],
            "fit_init_strategy_requested": fit_initialization[
                "requested_strategy"
            ],
            "random_initialization": fit_initialization[
                "random_initialization"
            ],
            "target_strategy": target_strategy,
            "fit_overlap_diagnostics": bool(fit_overlap_diagnostics),
            "target_representation": "lazy_submpo",
            "fit_overlap_fidelity": fit_overlap.get("fit_overlap_fidelity"),
            "fit_overlap_infidelity": fit_overlap.get("fit_overlap_infidelity"),
            "fit_overlap_error": fit_overlap.get("fit_overlap_error"),
            "backend": "fit",
            "fallback": False,
        }
        self._maybe_lock_dmrg1_one_site_phase()
        return projected_norm, center

    def _run_dmrg(
        self,
        G_seq,
        where_seq,
        n_iter,
        event_seq=None,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        k_2q_batch=1,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=1,
        fit_finite_check=None,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        fit_max_span=None,
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        quality_check_every=None,
        quality_check_repair=True,
    ):
        """Apply gates with local DMRG-style fitting."""
        if event_seq is None:
            event_seq = ("gate",) * len(G_seq)
        if len(event_seq) != len(G_seq):
            raise ValueError("DMRG event metadata must match the gate stream length.")
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")
        fit_target_strategy = self._validate_fit_target_strategy(
            fit_target_strategy
        )
        if fit_target_strategy == "auto":
            # Layered targets retain exact operator factors without building a
            # growing target MPS, but they rely on ordinary dense site tags.
            # Native Symmray/fermionic states therefore stay on their graded
            # materialized route, where charge and dummy-mode metadata survive.
            fit_target_strategy = (
                "mps"
                if self._replay_has_symmray_data(self.p) or self.p.isfermionic()
                else "layered"
            )

        # ``dmrg1`` has a bounded two-site warm-up: exactly two sweeps for an
        # under-capacity window, then one-site refinement. It latches into the
        # one-site phase once every full-chain bond reaches its attainable
        # physical/chi ceiling. ``dmrg2`` and ``dmrg3`` retain their configured
        # fixed warm-ups. Generic ``dmrg`` remains rank-adaptive for local
        # windows and uses the fixed canonical handoff for long-range windows.
        adaptive_rank_schedule = self._dmrg_mode_alias not in {
            "dmrg1",
            "dmrg2",
            "dmrg3",
        }
        adaptive_sweeps = (
            2 if self._dmrg_mode_alias == "dmrg1" else int(fit_adaptive_sweeps)
        )

        self._last_dmrg_fit_diagnostics = None
        p = self.p
        self._maybe_lock_dmrg1_one_site_phase()
        two_qubit_count = 0
        submpo_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)

        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress_mode = self._dmrg_mode_alias or "dmrg"
            pbar = tqdm(
                total=len(G_seq),
                desc=progress_mode,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["dmrg"],
            )
        else:
            pbar = None

        def use_single_pair_fast_path(xmin, xmax, active_block_size):
            """Apply the named DMRG2 adjacent-pair schedule exception."""
            return bool(fit_single_pair_fast_path) or (
                self._dmrg_mode_alias == "dmrg2"
                and int(xmax) == int(xmin) + 1
                and int(active_block_size) == 2
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            where = where_seq[idx]
            gate = G_seq[idx]
            event_type = event_seq[idx]
            if len(where) == 1:
                if event_type == "submpo":
                    # A one-site sub-MPO does not need a variational window;
                    # keep this small compatibility path on the direct MPO
                    # application route.
                    self._run_mpo(
                        [gate],
                        [where],
                        [event_type],
                        progbar=False,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        normalize_every=None,
                        normalize_final=False,
                        non_unitary=non_unitary,
                        stabilize_unitary=stabilize_unitary,
                    )
                    p = self.p
                    compressed = True
                else:
                    self._apply_gate(
                        p,
                        gate,
                        where,
                        contract=True,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        inplace=True,
                    )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                is_submpo = event_type == "submpo"
                if not is_submpo and len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                if is_submpo and len(where) < 2:
                    raise ValueError(
                        "DMRG sub-MPO events require at least two support sites."
                    )

                if is_submpo or k_2q_batch == 1:
                    if is_submpo:
                        submpo_count += 1
                    else:
                        two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    active_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    active_single_pair_fast_path = use_single_pair_fast_path(
                        xmin,
                        xmax,
                        active_fit_block_size,
                    )
                    self._validate_dmrg1_iteration_budget(
                        p,
                        (xmin, xmax),
                        n_iter=n_iter,
                        block_size=active_fit_block_size,
                    )
                    self._prepare_fit_window(
                        (xmin, xmax),
                        block_size=fit_block_size,
                    )
                    self.canonize_mps(p, (xmin, xmax))
                    unitary_target_norm = self._unitary_previous_norm

                    # Keep a transaction for an unexpected FIT exception.
                    # Normal low-rank long-range starts are repaired directly
                    # below by randomized initialization of the disposable FIT
                    # guess; norm loss is never silently converted into an MPO
                    # result.
                    fit_state_snapshot = (
                        self._fit_rollback_snapshot(p, (xmin, xmax))
                        if xmax - xmin > 1 and self.mode != "mix"
                        else None
                    )
                    fit_info_snapshot = (
                        dict(self.info_c)
                        if fit_state_snapshot is not None
                        else None
                    )
                    native_fit_guess_source = None
                    if (
                        not is_submpo
                        and self._replay_has_symmray_data(p)
                        and (
                            self._native_src_fit_guess_enabled(
                                fit_init_strategy,
                                fit_mpo_guess,
                            )
                            or fit_init_strategy in {
                                "guess_direct",
                                "svd_guess",
                            }
                        )
                    ):
                        native_fit_guess_source = (
                            fit_state_snapshot
                            if fit_state_snapshot is not None
                            else p.copy(deep=True)
                        )

                    if is_submpo:
                        # An explicit sub-MPO is already the operator target;
                        # keep it as a lazy layer rather than densifying it or
                        # applying it to the live MPS before FIT.
                        p_g, active_target_strategy = self._timed_call(
                            "dmrg.target",
                            self._build_submpo_fit_target,
                            p,
                            gate,
                            where,
                            target_cutoff,
                            cutoff_mode,
                            target_strategy=fit_target_strategy,
                        )
                        native_fermionic_warm_start = False
                    else:
                        # Ordinary gates use the same target policy, but the
                        # native fermionic warm start is allowed to open charge
                        # sectors before FIT. That warm start is a disposable
                        # preparation step and never substitutes for ``p_g``.
                        active_target_strategy = fit_target_strategy
                        p_g = self._timed_call(
                            "dmrg.target",
                            self._build_norm_target,
                            p,
                            gate,
                            where,
                            target_cutoff,
                            cutoff_mode,
                            target_strategy=fit_target_strategy,
                        )
                        if fit_init_strategy in {
                            "guess_direct",
                            "svd_guess",
                        }:
                            # The native direct guess already applies the gate
                            # and opens compatible sectors on its private copy.
                            # Replaying a second warm start on ``p`` would add
                            # work without changing the FIT initialization.
                            native_fermionic_warm_start = False
                        else:
                            native_fermionic_warm_start = self._timed_call(
                                "dmrg.native_warm_start",
                                self._warm_start_native_fermionic_fit,
                                p,
                                (gate,),
                                (where,),
                                cutoff=cutoff,
                                cutoff_mode=cutoff_mode,
                            )
                    active_fit_block_size = self._dmrg_fit_block_size(
                        p,
                        (xmin, xmax),
                        fit_block_size,
                    )
                    fit_guess_seed = (
                        int(fit_init_seed)
                        + 1000003 * int(idx)
                        + 1009 * int(xmin)
                        + int(xmax)
                    )
                    if is_submpo:
                        fit_initialization = self._timed_call(
                            "dmrg.fit_guess",
                            self._prepare_submpo_fit_initial_guess,
                            p,
                            gate,
                            where,
                            block_size=active_fit_block_size,
                            strategy=fit_init_strategy,
                            fit_mpo_guess=fit_mpo_guess,
                            rand_strength=fit_init_rand_strength,
                            seed=fit_guess_seed,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                        )
                    else:
                        fit_initialization = self._timed_call(
                            "dmrg.fit_guess",
                            self._prepare_fit_initial_guess,
                            p,
                            (gate,),
                            (where,),
                            block_size=active_fit_block_size,
                            strategy=fit_init_strategy,
                            fit_mpo_guess=fit_mpo_guess,
                            rand_strength=fit_init_rand_strength,
                            seed=fit_guess_seed,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            native_source=native_fit_guess_source,
                        )
                    fit_guess = fit_initialization["fit_guess"]
                    if fit_state_snapshot is p and fit_guess is p:
                        # Direct FIT mutates the live state; isolate rollback
                        # before entering the solver. An owned SRC/random
                        # guess leaves the retained original state untouched.
                        fit_state_snapshot = self._copy_fit_window_state(p, (xmin, xmax))
                    svd_guess_used = fit_initialization["svd_guess_used"]
                    mpo_fit_guess_used = svd_guess_used
                    random_initialization = fit_initialization[
                        "random_initialization"
                    ]
                    active_adaptive_sweeps = adaptive_sweeps
                    # A rank-adaptive sweep can finish a long-range window
                    # before its terminal one-site canonicalization has
                    # completed.  The local FIT norm then no longer describes
                    # the whole represented state, and the unitary norm guard
                    # correctly rejects the next gate.  Use the fixed
                    # block schedule for that window; it retains the same
                    # randomized FIT initialization while guaranteeing the
                    # canonical handoff. Named modes already use this path.
                    active_adaptive_rank_schedule = (
                        adaptive_rank_schedule
                        and not (
                            self._dmrg_mode_alias is None
                            and active_fit_block_size in {2, 3}
                            and xmax - xmin + 1 > active_fit_block_size
                        )
                    )
                    fit = FIT(
                        p_g,
                        p=fit_guess,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=True,
                        copy_target=False,
                    )
                    # Apply the selected one-, two-, or three-site FIT update to
                    # this gate window. ``run_gate`` reuses environments on
                    # both sides while leaving the rest of the MPS fixed.
                    fit_error = None
                    try:
                        self._run_fit_gate(
                            fit,
                            n_iter=n_iter,
                            verbose=False,
                            min_iter=fit_min_iter,
                            rtol=fit_rtol,
                            patience=fit_patience,
                            finite_check=fit_finite_check,
                            block_size=active_fit_block_size,
                            sweep_sequence=fit_sweep_sequence,
                            max_bond=self.chi,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            single_pair_fast_path=active_single_pair_fast_path,
                            adaptive_block_sweeps=active_adaptive_sweeps,
                            adaptive_until_rank=active_adaptive_rank_schedule,
                            final_one_site_sweeps=0,
                            collect_split_diagnostics=False,
                        )
                    except Exception as exc:
                        fit_error = exc
                    finally:
                        self._last_dmrg_fit_diagnostics = {
                            "iterations": int(fit.iterations_run),
                            "converged": bool(fit.converged),
                            "convergence_reason": fit.convergence_reason,
                            "relative_change": fit.last_relative_change,
                            "center_site": fit.final_center_site,
                            "block_size": int(active_fit_block_size),
                            "adaptive_sweeps": int(fit.adaptive_sweeps_run),
                            "one_site_refinement_sweeps": int(
                                fit.one_site_sweeps_run
                            ),
                            "native_fermionic_warm_start": bool(
                                native_fermionic_warm_start
                            ),
                            "mpo_fit_guess_used": bool(mpo_fit_guess_used),
                            "svd_guess_used": bool(svd_guess_used),
                            "guess_used": bool(fit_initialization["guess_used"]),
                            "guess_method": fit_initialization["guess_method"],
                            "guess_backend": fit_initialization.get(
                                "guess_backend"
                            ),
                            "native_randomized_guess_used": bool(
                                fit_initialization.get(
                                    "native_randomized_guess_used", False
                                )
                            ),
                            "fit_init_strategy": fit_initialization["strategy"],
                            "fit_init_strategy_requested": fit_initialization[
                                "requested_strategy"
                            ],
                            "random_initialization": random_initialization,
                            "target_strategy": active_target_strategy,
                            "fit_overlap_diagnostics": bool(
                                fit_overlap_diagnostics
                            ),
                            # Filled only after FIT succeeds.  This is a
                            # target-overlap diagnostic, not norm survival.
                            "fit_overlap_fidelity": None,
                            "fit_overlap_infidelity": None,
                            "fit_overlap_error": None,
                        }

                    fit_center = fit.final_center_site
                    fit_norm = fit.final_norm
                    fit_fallback_reason = None
                    if fit_error is not None and self.mode != "mix":
                        fit_fallback_reason = "fit_exception"

                    if fit_fallback_reason is not None:
                        if fit_state_snapshot is None:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise RuntimeError(
                                "DMRG FIT requested an MPO fallback without "
                                "a transactional state snapshot."
                            )
                        self.p = self._install_represented_norm(
                            fit_state_snapshot
                        )
                        self.info_c = fit_info_snapshot
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "mpo",
                                "fallback": True,
                                "fallback_reason": fit_fallback_reason,
                                "fit_norm": (
                                    None
                                    if fit_norm is None
                                    else self._real_float(
                                        ar.do("abs", fit_norm)
                                    )
                                ),
                            }
                        )
                        try:
                            self._run_mpo(
                                [gate],
                                [where],
                                [event_type],
                                progbar=False,
                                cutoff=cutoff,
                                cutoff_mode=cutoff_mode,
                                normalize_every=None,
                                normalize_final=False,
                                non_unitary=non_unitary,
                                stabilize_unitary=stabilize_unitary,
                            )
                        except Exception:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise
                        p = self.p
                    else:
                        if fit_error is not None:
                            raise fit_error.with_traceback(
                                fit_error.__traceback__
                            )
                        p = self._install_represented_norm(fit.p)
                        self.p = p
                        self._record_orthog_span(
                            p,
                            (fit_center, fit_center)
                            if fit_center is not None
                            else (xmin, xmax),
                        )
                        if not non_unitary:
                            self._timed_call(
                                "dmrg.stabilize",
                                self._stabilize_unitary_fit_state,
                                p,
                                (xmin, xmax),
                                unitary_target_norm,
                                current_norm=fit_norm,
                                center_site=fit_center,
                                restore=stabilize_unitary,
                            )
                        fit_overlap = (
                            {}
                            if self.mode == "mix" or not fit_overlap_diagnostics
                            else self._fit_overlap_diagnostics(p_g, fit.p)
                        )
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "fit",
                                "fallback": False,
                                "fit_overlap_diagnostics": bool(
                                    fit_overlap_diagnostics
                                ),
                                **fit_overlap,
                            }
                        )
                    self._maybe_lock_dmrg1_one_site_phase()
                    self._last_dmrg_fit_diagnostics[
                        "dmrg1_one_site_locked"
                    ] = bool(self._dmrg1_one_site_locked)
                    idx += 1
                    advanced = 1
                    last_where = (xmin, xmax)
                    compressed = True
                else:
                    batch_G, batch_where, two_qubit_in_batch, next_idx = (
                        self._collect_dmrg_batch(
                            G_seq,
                            where_seq,
                            idx,
                            k_2q_batch,
                            max_span=fit_max_span,
                        )
                    )
                    if two_qubit_in_batch < 1:
                        raise RuntimeError("DMRG batch unexpectedly contains no two-qubit gates.")

                    two_qubit_count += two_qubit_in_batch
                    batch_span_sites = [site for where_i in batch_where for site in where_i]
                    xmin, xmax = min(batch_span_sites), max(batch_span_sites)
                    active_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    self._validate_dmrg1_iteration_budget(
                        p,
                        (xmin, xmax),
                        n_iter=n_iter,
                        block_size=active_fit_block_size,
                    )
                    self._prepare_fit_window(
                        (xmin, xmax),
                        block_size=fit_block_size,
                    )
                    self.canonize_mps(p, (xmin, xmax))
                    unitary_target_norm = self._unitary_previous_norm
                    fit_state_snapshot = (
                        self._fit_rollback_snapshot(p, (xmin, xmax))
                        if xmax - xmin > 1 and self.mode != "mix"
                        else None
                    )
                    fit_info_snapshot = (
                        dict(self.info_c)
                        if fit_state_snapshot is not None
                        else None
                    )
                    native_fit_guess_source = None
                    if (
                        self._replay_has_symmray_data(p)
                        and (
                            self._native_src_fit_guess_enabled(
                                fit_init_strategy,
                                fit_mpo_guess,
                            )
                            or fit_init_strategy in {
                                "guess_direct",
                                "svd_guess",
                            }
                        )
                    ):
                        native_fit_guess_source = (
                            fit_state_snapshot
                            if fit_state_snapshot is not None
                            else p.copy(deep=True)
                        )
                    p_g = self._timed_call(
                        "dmrg.target",
                        self._build_dmrg_batch_target,
                        p,
                        batch_G,
                        batch_where,
                        target_cutoff,
                        cutoff_mode,
                        target_strategy=fit_target_strategy,
                    )
                    if fit_init_strategy in {
                        "guess_direct",
                        "svd_guess",
                    }:
                        native_fermionic_warm_start = False
                    else:
                        native_fermionic_warm_start = self._timed_call(
                            "dmrg.native_warm_start",
                            self._warm_start_native_fermionic_fit,
                            p,
                            batch_G,
                            batch_where,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                        )
                    active_fit_block_size = self._dmrg_fit_block_size(
                        p,
                        (xmin, xmax),
                        fit_block_size,
                    )
                    active_single_pair_fast_path = use_single_pair_fast_path(
                        xmin,
                        xmax,
                        active_fit_block_size,
                    )
                    fit_initialization = self._timed_call(
                        "dmrg.fit_guess",
                        self._prepare_fit_initial_guess,
                        p,
                        batch_G,
                        batch_where,
                        block_size=active_fit_block_size,
                        strategy=fit_init_strategy,
                        fit_mpo_guess=fit_mpo_guess,
                        rand_strength=fit_init_rand_strength,
                        seed=(
                            int(fit_init_seed)
                            + 1000003 * int(idx)
                            + 1009 * int(xmin)
                            + int(xmax)
                        ),
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        native_source=native_fit_guess_source,
                    )
                    fit_guess = fit_initialization["fit_guess"]
                    if fit_state_snapshot is p and fit_guess is p:
                        fit_state_snapshot = self._copy_fit_window_state(p, (xmin, xmax))
                    random_initialization = fit_initialization[
                        "random_initialization"
                    ]
                    active_adaptive_sweeps = adaptive_sweeps
                    active_adaptive_rank_schedule = (
                        adaptive_rank_schedule
                        and not (
                            self._dmrg_mode_alias is None
                            and active_fit_block_size in {2, 3}
                            and xmax - xmin + 1 > active_fit_block_size
                        )
                    )
                    fit = FIT(
                        p_g,
                        p=fit_guess,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=True,
                        copy_target=False,
                    )
                    fit_error = None
                    try:
                        self._run_fit_gate(
                            fit,
                            n_iter=n_iter,
                            verbose=False,
                            min_iter=fit_min_iter,
                            rtol=fit_rtol,
                            patience=fit_patience,
                            finite_check=fit_finite_check,
                            block_size=active_fit_block_size,
                            sweep_sequence=fit_sweep_sequence,
                            max_bond=self.chi,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            single_pair_fast_path=active_single_pair_fast_path,
                            adaptive_block_sweeps=active_adaptive_sweeps,
                            adaptive_until_rank=active_adaptive_rank_schedule,
                            final_one_site_sweeps=0,
                            collect_split_diagnostics=False,
                        )
                    except Exception as exc:
                        fit_error = exc
                    finally:
                        self._last_dmrg_fit_diagnostics = {
                            "iterations": int(fit.iterations_run),
                            "converged": bool(fit.converged),
                            "convergence_reason": fit.convergence_reason,
                            "relative_change": fit.last_relative_change,
                            "center_site": fit.final_center_site,
                            "block_size": int(active_fit_block_size),
                            "adaptive_sweeps": int(fit.adaptive_sweeps_run),
                            "one_site_refinement_sweeps": int(
                                fit.one_site_sweeps_run
                            ),
                            "native_fermionic_warm_start": bool(
                                native_fermionic_warm_start
                            ),
                            "mpo_fit_guess_used": bool(
                                fit_initialization["svd_guess_used"]
                            ),
                            "svd_guess_used": bool(
                                fit_initialization["svd_guess_used"]
                            ),
                            "guess_used": bool(fit_initialization["guess_used"]),
                            "guess_method": fit_initialization["guess_method"],
                            "guess_backend": fit_initialization.get(
                                "guess_backend"
                            ),
                            "native_randomized_guess_used": bool(
                                fit_initialization.get(
                                    "native_randomized_guess_used", False
                                )
                            ),
                            "fit_init_strategy": fit_initialization["strategy"],
                            "fit_init_strategy_requested": fit_initialization[
                                "requested_strategy"
                            ],
                            "random_initialization": random_initialization,
                            "target_strategy": fit_target_strategy,
                            "fit_overlap_diagnostics": bool(
                                fit_overlap_diagnostics
                            ),
                            # Filled only after FIT succeeds.  This is a
                            # target-overlap diagnostic, not norm survival.
                            "fit_overlap_fidelity": None,
                            "fit_overlap_infidelity": None,
                            "fit_overlap_error": None,
                        }

                    fit_center = fit.final_center_site
                    fit_norm = fit.final_norm
                    fit_fallback_reason = None
                    if fit_error is not None and self.mode != "mix":
                        fit_fallback_reason = "fit_exception"

                    if fit_fallback_reason is not None:
                        if fit_state_snapshot is None:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise RuntimeError(
                                "DMRG FIT requested an MPO fallback without "
                                "a transactional state snapshot."
                            )
                        self.p = self._install_represented_norm(
                            fit_state_snapshot
                        )
                        self.info_c = fit_info_snapshot
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "mpo",
                                "fallback": True,
                                "fallback_reason": fit_fallback_reason,
                                "fit_norm": (
                                    None
                                    if fit_norm is None
                                    else self._real_float(
                                        ar.do("abs", fit_norm)
                                    )
                                ),
                            }
                        )
                        try:
                            self._run_mpo(
                                batch_G,
                                batch_where,
                                ["gate"] * len(batch_G),
                                progbar=False,
                                cutoff=cutoff,
                                cutoff_mode=cutoff_mode,
                                normalize_every=None,
                                normalize_final=False,
                                non_unitary=non_unitary,
                                stabilize_unitary=stabilize_unitary,
                            )
                        except Exception:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise
                        p = self.p
                    else:
                        if fit_error is not None:
                            raise fit_error.with_traceback(
                                fit_error.__traceback__
                            )
                        p = self._install_represented_norm(fit.p)
                        self.p = p
                        self._record_orthog_span(
                            p,
                            (fit_center, fit_center)
                            if fit_center is not None
                            else (xmin, xmax),
                        )
                        if not non_unitary:
                            self._timed_call(
                                "dmrg.stabilize",
                                self._stabilize_unitary_fit_state,
                                p,
                                (xmin, xmax),
                                unitary_target_norm,
                                current_norm=fit_norm,
                                center_site=fit_center,
                                restore=stabilize_unitary,
                            )
                        fit_overlap = (
                            {}
                            if self.mode == "mix" or not fit_overlap_diagnostics
                            else self._fit_overlap_diagnostics(p_g, fit.p)
                        )
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "fit",
                                "fallback": False,
                                "fit_overlap_diagnostics": bool(
                                    fit_overlap_diagnostics
                                ),
                                **fit_overlap,
                            }
                        )
                    self._maybe_lock_dmrg1_one_site_phase()
                    self._last_dmrg_fit_diagnostics[
                        "dmrg1_one_site_locked"
                    ] = bool(self._dmrg1_one_site_locked)
                    advanced = next_idx - idx
                    idx = next_idx
                    last_where = (xmin, xmax)
                    compressed = True

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._maybe_run_quality_check(
                idx,
                last_where,
                quality_check_every,
                repair=quality_check_repair,
            )

            self._record_effective_event(last_where, event_type="gate")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                if submpo_count:
                    postfix["mpo"] = submpo_count
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_mpo(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        event_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        submpo_method="direct",
        compression_seed=None,
        compression_opts=None,
        stabilize_unitary=False,
    ):
        """Replay gates/sub-MPOs with Quimb compression (direct by default).

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        The historical private name describes operator application machinery;
        the public mode names the algorithm, such as ``direct`` or ``src``.
        """
        p = self.p
        mpo_method = self._normalize_submpo_method(submpo_method)
        # Report the selected algorithm (``direct``, ``src``, etc.), not
        # the historical private helper name.
        timing_name = mpo_method
        gate_cutoff_mode = (
            _DEFAULT_CUTOFF_MODE
            if cutoff_mode is None
            else cutoff_mode
        )
        # ``mpo_cutoff_mode`` is intentionally resolved by ``run`` before
        # entering this backend: ``None`` means keep Quimb's native default
        # for methods such as ``dm``. One-site gates do not invoke that MPO
        # compressor, so they use the ordinary Pepsy cutoff policy here.
        mpo_compress_opts = self._submpo_compress_opts(
            mpo_method,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        mpo_compress_opts.update(compression_opts or {})
        mpo_optimize = mpo_compress_opts.get("optimize")
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        submpo_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress_mode = self._progress_mode_name(self.mode)
            pbar = tqdm(
                total=len(G_seq),
                desc=progress_mode,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS.get(
                    progress_mode,
                    self._PROGBAR_COLORS["mpo"],
                ),
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            unitary_target_norm = (
                self._unitary_previous_norm if not non_unitary else None
            )
            where = where_seq[idx]
            gate = G_seq[idx]
            event_type = event_seq[idx]
            if event_type == "submpo":
                # The payload is already an exact operator representation.
                # Apply it through ``gate_with_submpo_`` so the selected
                # compressor sees the original MPO; do not densify it merely
                # to reuse the ordinary gate branch.
                submpo_count += 1
                xmin, xmax = min(where), max(where)
                if (
                    mpo_method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
                    and _is_interior_submpo_span(p, where)
                ):
                    _apply_submpo_with_interior_workaround(
                        p,
                        gate,
                        where,
                        chi=self.chi,
                        method=mpo_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        info=self.info_c,
                        inplace_mpo=False,
                        optimize=mpo_optimize,
                        seed=compression_seed,
                        compression_opts=compression_opts,
                    )
                else:
                    self.canonize_mps(p, (xmin, xmax))
                    submpo_opts = dict(mpo_compress_opts)
                    if mpo_method in _MPO_METHODS_USE_SEED:
                        submpo_opts["seed"] = compression_seed
                    _run_seeded_quimb(
                        None,
                        p.gate_with_submpo_,
                        gate,
                        where=where,
                        method=mpo_method,
                        max_bond=self.chi,
                        info=self.info_c,
                        inplace_mpo=False,
                        **submpo_opts,
                    )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        f"{timing_name}.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )
            elif len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=gate_cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = self.backend == "symmray"
                self.canonize_mps(p, (xmin, xmax))
                if use_symmray_auto_swap:
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=gate_cutoff_mode,
                        max_bond=self.chi,
                    )
                else:
                    _apply_dense_gate_with_method(
                        p,
                        gate,
                        where,
                        dims=self._infer_gate_dims(gate, where),
                        chi=self.chi,
                        method=mpo_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        info=self.info_c,
                        optimize=mpo_optimize,
                        seed=compression_seed,
                        compression_opts=compression_opts,
                    )
                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        f"{timing_name}.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._record_effective_event(where, event_type=event_type)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                if submpo_count:
                    postfix["mpo"] = submpo_count
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_swap(self, *args, **kwargs):
        """Apply gates with swap-network compression, swapping back."""
        return self._run_swap_network(*args, swap_back=True, mode_name="swap", **kwargs)

    def _run_perm(self, *args, **kwargs):
        """Apply gates with lazy swap-network compression."""
        return self._run_swap_network(*args, swap_back=False, mode_name="perm", **kwargs)

    def _run_swap_network(  # pylint: disable=too-many-locals,too-many-arguments
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        stabilize_unitary=False,
        *,
        swap_back,
        mode_name,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates. When
        ``swap_back`` is false, ``where_seq`` is interpreted as logical sites,
        the current ``self.qubits`` mapping translates them to physical sites,
        and the right endpoint remains at the left endpoint's neighbour.
        """
        # ``swap_back`` is the semantic switch: ``swap`` restores the input
        # logical order after each nonlocal gate, while ``perm`` leaves the
        # physical order changed and updates both public mapping views for
        # later gates and logical readout.
        p = self.p
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc=mode_name,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS[mode_name],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            unitary_target_norm = (
                self._unitary_previous_norm if not non_unitary else None
            )
            logical_where = where_seq[idx]
            where = (
                tuple(int(site) for site in logical_where)
                if swap_back
                else self._logical_to_physical_where(logical_where)
            )
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                self.canonize_mps(p, (xmin, xmax))

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                if self._replay_has_symmray_data(p) and self._native_needs_safe_qr(p):
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        max_bond=self.chi,
                        info=self.info_c,
                        swap_back=swap_back,
                    )
                else:
                    p.gate_with_auto_swap_(
                        gate,
                        where,
                        info=self.info_c,
                        max_bond=self.chi,
                        swap_back=swap_back,
                        **compress_opts,
                    )
                if not swap_back:
                    self._record_permutation_move(where)

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        f"{mode_name}.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._record_effective_event(last_where, event_type="gate")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_svd(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        stabilize_unitary=False,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``. Symmray-backed
        MPS data use quimb's block-aware auto-swap split path by default as a
        conservative choice for block-sparse edge cases.
        """
        # SVD mode deliberately exposes the local gate-split algorithm. It is
        # useful as a transparent reference, whereas MPO/DMRG modes retain a
        # full operator target and can choose more global compression schemes.
        p = self.p
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="svd",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["svd"],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            unitary_target_norm = (
                self._unitary_previous_norm if not non_unitary else None
            )
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = self.backend == "symmray"
                self.canonize_mps(p, (xmin, xmax))
                if use_symmray_auto_swap:
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        max_bond=self.chi,
                    )
                else:
                    self._apply_gate(
                        p,
                        gate,
                        where,
                        contract="reduce-split",
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        inplace=True,
                    )
                    self.canonize_mps(p, (xmin, xmax))

                    for i in range(xmax, xmin, -1):
                        p.right_canonize_site(i, bra=None)
                    p.left_compress(
                        start=xmin,
                        stop=xmax,
                        max_bond=self.chi,
                        **compress_opts,
                    )
                    # ``right_canonize_site`` plus left-to-right compression
                    # leaves the complete raw norm at the right endpoint.
                    self._record_orthog_span(p, (xmax, xmax))

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        "svd.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._record_effective_event(last_where, event_type="gate")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_exact_batch(
        self, G_seq, where_seq, progbar=False, cutoff=1e-12, cutoff_mode="rsum2"
    ):
        """Replay bounded fused gates while retaining dense state semantics."""
        if not supports_batch(self.p):
            return self._run_exact(
                G_seq, where_seq, progbar=progbar,
                cutoff=cutoff, cutoff_mode=cutoff_mode,
            )
        # A single-tensor network is already contracted. Copy its metadata,
        # not its exponentially large array; each batch produces a new array.
        if self.p.num_tensors == 1:
            self.p = self._install_represented_norm(self.p.copy())
        else:
            tensor = self.p.contract(all, optimize=self.contraction_opt)
            self.p = self._install_represented_norm(qtn.TensorNetwork([tensor]))
        self.info_c = {}
        tensor = self.p.tensors[0]
        pbar = None
        if progbar:
            from tqdm import tqdm

            pbar = tqdm(
                total=len(G_seq), desc="exact-batch", ascii=True,
                colour=self._PROGBAR_COLORS["exact-batch"],
            )
        try:
            for batch in iter_exact_batches(
                G_seq, where_seq, self._format_ind,
                backend=ar.infer_backend(tensor.data),
                state_size=int(np.prod(tensor.data.shape)),
            ):
                batch.apply(tensor)
                for where in batch.locations:
                    self._record_effective_event(where, event_type="gate")
                if pbar is not None:
                    pbar.update(len(batch.locations))
        finally:
            if pbar is not None:
                pbar.close()

    def _run_exact(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
    ):
        """Apply gates exactly using in-place ``contract=True`` application.

        Progress bar counts all gates for consistency with other modes.
        """
        self.p = self.p.contract(all, optimize="auto-hq")
        self.p = self._install_represented_norm(qtn.TensorNetwork([self.p]))
        self.info_c = {}
        p = self.p
        two_qubit_count = 0
        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="exact",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["exact"],
            )

        for gate, where in zip(G_seq, where_seq):
            if len(where) not in (1, 2):
                raise ValueError("Each gate location must have one or two sites.")

            inds = [self._format_ind(site) for site in where]
            qtn.tensor_network_gate_inds(
                p,
                gate,
                inds,
                contract=True,
                info=None,
                inplace=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            self._record_effective_event(where, event_type="gate")

            if len(where) == 1:
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "2q": two_qubit_count,
                            "~F": self._format_progress_scalar(
                                self._cumulative_fidelity()
                            ),
                            "bnd": "inf",
                        }
                    )
                    pbar.update(1)
                continue

            two_qubit_count += 1
            if pbar is not None:
                pbar.set_postfix(
                    {
                        "2q": two_qubit_count,
                        "~F": self._format_progress_scalar(
                            self._cumulative_fidelity()
                        ),
                        "bnd": "inf",
                    }
                )
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = self._install_represented_norm(p)

    def canonize_mps(self, p, where, *, info=None):
        """Update canonical form and optionally accumulate its wall time."""
        if self._timing_state is None:
            return self._canonize_mps_impl(p, where, info=info)
        return self._timed_call(
            "canonicalize",
            self._canonize_mps_impl,
            p,
            where,
            info=info,
        )

    def _canonize_mps_impl(self, p, where, *, info=None):
        """Update canonical form around a one- or two-site gate span.

        ``where`` may be an int, a 1-tuple ``(site,)``, or a 2-tuple
        ``(xmin, xmax)``. Integers and singletons collapse to a single-site
        orthogonality center.
        """
        if isinstance(where, Integral):
            site = int(where)
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 1:
            site = int(where[0])
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 2:
            site0, site1 = int(where[0]), int(where[1])
            xmin, xmax = min(site0, site1), max(site0, site1)
            where_canon = [xmin, xmax]
            target_orthog = (xmin, xmax)
        else:
            raise ValueError("where must be an int, (int,), or (int, int).")

        state_info = self._info_for_state(p, info)
        p.canonize(
            where_canon,
            cur_orthog=self._current_orthog(p, info=state_info),
            info=state_info,
        )
        # Preserve the fitting-window semantics expected by gate updates.
        state_info["cur_orthog"] = target_orthog
        return target_orthog

    def get_quality_checks(self):
        """Return periodic finite-data and canonical-gauge check records."""
        return deepcopy(self.quality_checks)

    def get_normalizations(self):
        """Return automatic normalization events recorded during ``run``.

        Each event contains the 1-based ``step``, removed local ``old_norm``,
        active ``span``, rescaled ``sites``, per-tensor ``scales``, total
        ``log10_scale``, event ``reason``, and resulting base-10 ``exponent``.
        """
        return deepcopy(self.normalizations)

    def get_norm_events(self):
        """Return independent norm-survival records with Python scalar values."""
        self._check_deferred_norm_errors()
        return deepcopy([self._norm_event_to_host(event) for event in self.norm_events])
