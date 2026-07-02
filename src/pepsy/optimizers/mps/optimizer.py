"""MPS optimization helpers centered on :class:`MpsOptimizer`.

:class:`MpsOptimizer` replays a canonical bundled gate stream
``[(gate, where), ...]`` against an MPS, using one of several compression
backends.  ``mode="mpo"`` also accepts explicit sub-MPO events of the form
``("submpo", mpo, where)`` or
``{"kind": "submpo", "mpo": mpo, "where": where}``.  The default path assumes
a norm-preserving stream and does not renormalize.  Non-unitary streams should use ``non_unitary=True``; when
``normalize_every`` is enabled this normalizes the active MPS tensor data after
compressed updates while accumulating the removed scale into ``p.exponent``.
Quimb includes that exponent in ``p.norm()``, so ``p.norm()`` still reports the
represented state norm; inspect a copy with ``exponent=0`` to see the rescaled
data norm. Diagnostics are separate opt-ins:
``track_norm_infidelity=True`` records a cheap norm-ratio proxy, while
``track_infidelity=True`` records a true normalized-overlap metric, reports
cumulative infidelity in progress bars, and keeps the running geometric mean
of measured local true fidelities in ``losses`` for compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
import types
import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...fitting.local import FIT
from ...operators.gates import _normalize_gate_entries, gate as apply_gate
from ...tensors.core import tn_fidelity, tn_norm

__all__ = [
    "MpsOptimizer",
    "is_submpo_event",
    "normalize_submpo_where",
    "submpo_event_parts",
]


_SUBMPO_EVENT_NAMES = frozenset({"submpo", "mpo"})
_MISSING = object()


def _normalize_event_name(name):
    """Normalize a stream event name for matching."""
    return str(name).replace("-", "_").strip().lower()


def _normalize_submpo_where(where):
    """Normalize sub-MPO support sites to a non-empty tuple of 1D ints."""
    if isinstance(where, Integral):
        return (int(where),)
    if (
        isinstance(where, (tuple, list))
        and len(where) > 0
        and all(isinstance(site, Integral) for site in where)
    ):
        return tuple(int(site) for site in where)
    raise ValueError(
        "subMPO event where must be a non-empty sequence of 1D sites."
    )


def normalize_submpo_where(where):
    """Return canonical 1D support sites for a sub-MPO stream event."""

    return _normalize_submpo_where(where)


def _submpo_event_parts(entry):
    """Return ``(mpo, where)`` if ``entry`` is a sub-MPO event, else ``None``."""
    if (
        isinstance(entry, tuple)
        and len(entry) == 3
        and isinstance(entry[0], str)
        and _normalize_event_name(entry[0]) in _SUBMPO_EVENT_NAMES
    ):
        return entry[1], entry[2]

    if not isinstance(entry, Mapping):
        return None

    kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
    if kind is _MISSING or _normalize_event_name(kind) not in _SUBMPO_EVENT_NAMES:
        return None

    mpo = entry.get(
        "mpo",
        entry.get("submpo", entry.get("operator", entry.get("payload", _MISSING))),
    )
    where = entry.get("where", entry.get("sites", _MISSING))
    if mpo is _MISSING or where is _MISSING:
        raise ValueError(
            "subMPO stream event mappings must contain 'mpo' and 'where'."
        )
    return mpo, where


def submpo_event_parts(entry, *, normalize_where=False):
    """Return ``(mpo, where)`` for a public sub-MPO stream event.

    Returns ``None`` when ``entry`` is not an explicit sub-MPO event. Mapping
    events must contain both an MPO payload and support sites, matching the
    accepted :class:`MpsOptimizer` stream contract.
    """

    parts = _submpo_event_parts(entry)
    if parts is None:
        return None
    mpo, where = parts
    if normalize_where:
        where = _normalize_submpo_where(where)
    return mpo, where


def _is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""
    return submpo_event_parts(entry) is not None


def is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""

    return submpo_event_parts(entry) is not None


def _normalize_gate_queue(gates):
    """Return ``(payloads, wheres, event_types)`` from bundled stream input."""
    submpo_parts = _submpo_event_parts(gates)
    if submpo_parts is not None:
        mpo, where = submpo_parts
        return [mpo], [_normalize_submpo_where(where)], ["submpo"]

    if isinstance(gates, (tuple, list)) and any(
        _is_submpo_event(entry) for entry in gates
    ):
        payloads = []
        wheres = []
        event_types = []
        for entry in gates:
            submpo_parts = _submpo_event_parts(entry)
            if submpo_parts is not None:
                mpo, where = submpo_parts
                payloads.append(mpo)
                wheres.append(_normalize_submpo_where(where))
                event_types.append("submpo")
                continue
            gate_entries = _normalize_gate_entries(
                (entry,),
                where=None,
                allow_empty=False,
            )
            gate, where = gate_entries[0]
            payloads.append(gate)
            wheres.append(tuple(where) if isinstance(where, list) else where)
            event_types.append("gate")
        return payloads, wheres, event_types

    entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], [], []
    gate_list, where_list = zip(*entries)
    return (
        list(gate_list),
        [tuple(w) if isinstance(w, list) else w for w in where_list],
        ["gate"] * len(gate_list),
    )


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
        ``where`` supports one- or two-site locations in 1D/2D/3D forms.
        For ``mode="mpo"``, entries may also have the explicit sub-MPO form
        ``("submpo", mpo, where)`` or mapping form
        ``{"kind": "submpo", "mpo": mpo, "where": where}``, with a 1D
        support ``where``. :meth:`submpo_event` builds the tuple form.
    chi : int
        Maximum bond dimension used by MPO/swap/SVD compression modes.
    mode : {"dmrg", "mpo", "swap", "svd", "exact"}, default="dmrg"
        Optimization backend.
    contraction_opt : object | None, default="auto-hq"
        Canonical contraction path optimizer keyword.
    ind_id : str, default="k{}"
        Format string for site index labels used by exact gate application.
        Use "k{},{}" when gate sites are 2D coordinates like ``(i, j)``.
    inplace : bool, default=False
        Whether to optimize the provided input state object directly. If
        ``False``, a copy is made and the original input remains unchanged.

    Attributes
    ----------
    normalizations : list[dict]
        Automatic normalization events recorded during :meth:`run`. Each entry
        stores the 1-based gate step, removed local squared scale,
        orthogonality span, tensor sites that were rescaled, and resulting
        base-10 ``p.exponent``. The raw tensor data are rescaled; the
        represented norm remains available through ``p.norm()`` because quimb
        applies ``p.exponent``.
    infidelities : list[float]
        Cumulative infidelity trace. When ``track_infidelity=True``, compressed
        two-site updates append the true normalized-overlap infidelity from
        :func:`pepsy.tensors.core.tn_fidelity`. Otherwise, when
        ``track_norm_infidelity=True``, they append the cheaper norm-ratio
        proxy ``1 - product((||approx|| / ||target||)**2)``.
    losses : list[float]
        Fidelity-like progress trace. By default this stores legacy
        norm-preservation proxy samples. With ``track_infidelity=True`` it
        instead stores the running geometric mean of the measured local true
        fidelities, computed from accumulated log fidelities for stability.
    """

    _ALLOWED_MODES = frozenset({"dmrg", "mpo", "swap", "svd", "exact"})
    _ALLOWED_SUBMPO_METHODS = frozenset(
        {
            "direct",
            "dm",
            "zipup",
            "zipup-first",
            "zipup-oversample",
            "src",
            "src-first",
            "src-oversample",
            "srcmps",
            "srcmps-first",
            "srcmps-oversample",
            "fit",
            "fit-direct",
            "fit-dm",
            "fit-zipup",
            "fit-zipup-first",
            "fit-zipup-oversample",
            "fit-src",
            "fit-src-first",
            "fit-src-oversample",
            "fit-oversample",
        }
    )
    _PROGBAR_COLORS = {
        "dmrg": "#1f77b4",
        "mpo": "#2ca02c",
        "swap": "#ff7f0e",
        "svd": "#d62728",
        "exact": "#9467bd",
    }

    @classmethod
    def _normalize_mode(cls, mode):
        """Validate and normalize execution mode."""
        mode_norm = str(mode).strip().lower()
        if mode_norm not in cls._ALLOWED_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return mode_norm

    @classmethod
    def _normalize_submpo_method(cls, method):
        """Validate and normalize the sub-MPO compression method."""
        method_norm = str(method).strip().lower()
        if method_norm not in cls._ALLOWED_SUBMPO_METHODS:
            raise ValueError(f"Unknown subMPO method: {method}")
        return method_norm

    def _submpo_compress_opts(self, method):
        """Return compression options for a sub-MPO method."""
        if method == "direct":
            return {}
        optimize = self.contraction_opt
        if optimize is None:
            return {}
        if isinstance(optimize, str) and optimize.strip().lower() in {
            "auto",
            "auto-hq",
        }:
            return {}
        return {"optimize": optimize}

    @staticmethod
    def submpo_event(mpo, where):
        """Return a canonical explicit sub-MPO stream event.

        The returned entry can be placed directly inside the ``gates`` stream
        for ``mode="mpo"``. ``where`` is restricted to 1D integer MPS sites.
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

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        p,
        gates=None,
        chi=None,
        mode="dmrg",
        contraction_opt="auto-hq",
        ind_id="k{}",
        inplace=False,
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

        self.inplace = bool(inplace)
        self.p = self._install_represented_norm(p if self.inplace else p.copy())
        self.G, self.where, self.event_types = _normalize_gate_queue(gates)
        self.chi = chi
        self.mode = self._normalize_mode(mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.ind_id = str(ind_id)

        self.info_c = {}
        self.losses = [1.0]
        self.normalizations = []
        self.infidelities = [0.0]
        self.true_infidelities = [0.0]
        self.infidelity_samples = []
        self.norm_infidelity_samples = []
        self._true_fidelity_log_sum = 0.0
        self._true_fidelity_count = 0
        self._norm_fidelity_proxy = 1.0
        self._init_canonicalization()

    def _current_orthog(self, p=None):
        """Return cached ``(min_site, max_site)`` orthogonality span.

        Cached entries may be ``"calc"`` / ``None`` (recompute), an ``int``,
        or a 1- or 2-tuple. The stored form is always a 2-tuple with
        ``min <= max``.
        """
        cur = self.info_c.get("cur_orthog", "calc")
        state = self.p if p is None else p
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

        self.info_c["cur_orthog"] = cur
        return cur

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
        """Fail early for Symmray/MPS mode combinations with known bad paths."""
        if not self._has_symmray_data(self.p):
            return

        if self.mode == "dmrg" and self.p.max_bond() < self.chi:
            raise ValueError(
                "Symmray MPS data cannot be automatically expanded for "
                "mode='dmrg' because quimb's bond-dimension expansion calls "
                "dense-style pad on the Symmray backend. Construct the SymMPS "
                "with bond_dim >= chi, or run with chi <= p.max_bond()."
            )

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
    ):
        """Apply a Symmray two-site gate through quimb's block-aware swaps."""
        compress_opts = {
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
        }
        if max_bond is not None:
            compress_opts["max_bond"] = max_bond
        p.gate_with_auto_swap_(
            gate,
            where,
            info=self.info_c if info is None else info,
            swap_back=True,
            **compress_opts,
        )
        return p

    def _build_symmray_auto_swap_target(self, p, gate, where, cutoff, cutoff_mode):
        """Build an un-chi-capped target using Symmray-aware swap routing."""
        p_target = p.copy()
        self._apply_symmray_auto_swap_gate(
            p_target,
            gate,
            where,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info={},
        )
        return p_target

    def _apply_gate(self, p, gate, where, **kwargs):
        """Apply a gate using this optimizer's physical-index convention."""
        kwargs.setdefault("ind_id", self.ind_id)
        return apply_gate(p, gate, where, **kwargs)

    def _init_canonicalization(self):
        """Initialize canonical form and orthogonality center."""
        if self.mode == "exact":
            # Exact evolution does not require a canonicalized input state.
            self.info_c = {}
            return
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    def _prepare_dmrg_state(self):
        """Ensure DMRG starts from at least ``chi`` bond dimension."""
        if self.p.max_bond() < self.chi:
            self.p.expand_bond_dimension(self.chi, inplace=True)
            self._init_canonicalization()

    def set_p(self, p):
        """Assign a new state and reset canonicalization metadata."""
        self.p = self._install_represented_norm(p if self.inplace else p.copy())
        self._init_canonicalization()

    def normalize(self, eps=1e-15, insert=None):
        """Normalize current ``self.p`` in-place.

        Parameters
        ----------
        eps : float, default=1e-15
            Precision used by cyclic MPS normalization.
        insert : int | None, default=None
            Optional site where the normalization factor is inserted.

        Returns
        -------
        float | complex
            Previous ``self.p.H @ self.p`` value returned by quimb.
            The corresponding removed norm factor is accumulated into
            ``self.p.exponent`` when present, so ``self.p.norm()`` continues to
            report the represented norm while the raw data norm becomes one.
        """
        old_norm = self.p.normalize(eps=eps, insert=insert)
        self._accumulate_exponent(self.p, old_norm**0.5)
        self._current_orthog(self.p)
        return old_norm

    def set_mode(self, mode):
        """Switch optimization mode while keeping ``p`` and ``info_c``."""
        old_mode = self.mode
        self.mode = self._normalize_mode(mode)
        if old_mode == "exact" and self.mode != "exact":
            # Recreate canonical metadata when leaving exact mode.
            self._init_canonicalization()
        return self

    def set_gates(self, gates):
        """Replace the current gate list.

        After calling this, ``run(...)`` applies only this new list
        (unless you call :meth:`add_gates` before running).
        """
        self.G, self.where, self.event_types = _normalize_gate_queue(gates)
        return self

    def add_gates(self, gates):
        """Append gates to the existing gate list.

        This preserves previously queued gates and extends them with
        new ones.
        """
        G_new, where_new, event_types_new = _normalize_gate_queue(gates)
        self.G.extend(G_new)
        self.where.extend(where_new)
        self.event_types.extend(event_types_new)
        return self

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=5,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        mode=None,
        fidelity_samples=None,
        k_2q_batch=1,
        non_unitary=False,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
        track_infidelity=False,
        submpo_method="direct",
    ):
        """Run the currently queued gates.

        Parameters
        ----------
        n_iter : int, default=5
            Inner iterations for DMRG local fits. Ignored by
            ``mpo``/``swap``/``svd``/``exact``.
        progbar : bool, default=False
            Show per-mode progress bars.
        cutoff : float, default=1e-12
            Truncation cutoff used in gate application and local fitting.
        cutoff_mode : str, default="rsum2"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``.
        mode : {"dmrg", "mpo", "swap", "svd", "exact"} | None, default=None
            Optional mode override for this run. If supplied, updates
            ``self.mode`` before execution.
        fidelity_samples : int | None, default=None
            Compression modes (``mpo``/``swap``/``svd``): number of
            intermediate norm-preservation proxy samples taken during the run.
            ``None`` keeps the historical default of ``10`` for
            unitary/default runs and disables norm-proxy sampling for
            ``non_unitary=True``. When an integer is supplied, a final sample
            is always recorded at the end; use ``0`` to record only the final
            sample. This legacy trace is returned by :meth:`get_fidelities`;
            it is not the true overlap diagnostic controlled by
            ``track_infidelity``.
        k_2q_batch : int, default=1
            DMRG mode only: number of sequential two-qubit gates to batch
            into one local FIT update. The FIT window uses the batch-wide
            ``[xmin, xmax]`` from all two-qubit gate locations in the batch.
        non_unitary : bool, default=False
            Convenience flag for non-unitary gate streams. Normalization is
            only available when this is ``True``; default/unitary runs never
            normalize. If enabled, local tensor scale control runs after every
            compressed two-site update or DMRG batch. No trailing
            normalization is performed unless ``normalize_final=True``.
        normalize_every : int | bool | None, default=False
            Compatibility switch for non-unitary local scale control. Any
            positive integer or ``True`` enables normalization after compressed
            updates; ``False`` or ``None`` disables it.
        normalize_final : bool, default=False
            If local scale control is enabled, also normalize once at the end
            of the run when the final gate was not already normalized. This
            also requires ``non_unitary=True``.
        normalize_eps : float, default=1e-15
            Retained for compatibility with older full-normalization paths.
            Local tensor scale control does not use this value.
        track_norm_infidelity : bool, default=False
            If ``True``, build pre-compression norm targets and append the
            cumulative norm-ratio infidelity proxy for compressed two-site
            updates. This diagnostic is intentionally off by default for the
            fast non-unitary path.
        track_infidelity : bool, default=False
            If ``True``, build the pre-compression target and measure the true
            normalized-overlap fidelity with :func:`pepsy.tensors.core.tn_fidelity`
            after each compressed two-site update or DMRG batch. The progress
            bar then reports cumulative infidelity. For compatibility,
            :meth:`get_fidelities` still reports the running geometric mean of
            these local true fidelities. This is expensive and disabled by
            default.
        submpo_method : str, default="direct"
            MPO mode only: compression method used for explicit sub-MPO stream
            events. This is forwarded to quimb's
            ``MatrixProductState.gate_with_submpo_``.

        Returns
        -------
        qtn.TensorNetwork
            The updated ``self.p`` state after replaying the queued gate stream.
        """
        if mode is not None:
            self.set_mode(mode)

        G_seq = list(self.G)
        where_seq = list(self.where)
        event_seq = list(self.event_types)
        if not G_seq:
            return self.p
        self._validate_symmray_mode_support()
        self._validate_event_stream_for_run(G_seq, where_seq, event_seq)

        non_unitary = bool(non_unitary)
        if not non_unitary:
            if normalize_every is not None and normalize_every is not False:
                raise ValueError("normalize_every requires non_unitary=True.")
            if normalize_final:
                raise ValueError("normalize_final requires non_unitary=True.")
        fidelity_samples = self._resolve_fidelity_samples(
            fidelity_samples,
            non_unitary=non_unitary,
        )
        normalize_every = self._normalize_every_interval(
            normalize_every,
            non_unitary=non_unitary,
        )
        track_norm_infidelity = bool(track_norm_infidelity)
        track_infidelity = bool(track_infidelity)
        if (
            normalize_every is not None
            or track_norm_infidelity
            or track_infidelity
        ) and self.mode == "exact":
            raise ValueError(
                "automatic normalization and infidelity diagnostics use "
                "MPS canonicalization and are not available in exact mode."
            )
        record_fit_losses = (
            ((not non_unitary) or (fidelity_samples is not None))
            and not track_infidelity
        )

        if self.mode == "dmrg":
            self._prepare_dmrg_state()
            self._run_dmrg(
                G_seq,
                where_seq,
                n_iter=n_iter,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                k_2q_batch=k_2q_batch,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                track_norm_infidelity=track_norm_infidelity,
                track_infidelity=track_infidelity,
                record_fit_losses=record_fit_losses,
            )
            return self.p

        if self.mode == "mpo":
            self._run_mpo(
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fidelity_samples=fidelity_samples,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                track_norm_infidelity=track_norm_infidelity,
                track_infidelity=track_infidelity,
                submpo_method=submpo_method,
            )
            return self.p

        if self.mode == "swap":
            self._run_swap(
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fidelity_samples=fidelity_samples,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                track_norm_infidelity=track_norm_infidelity,
                track_infidelity=track_infidelity,
            )
            return self.p

        if self.mode == "svd":
            self._run_svd(
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fidelity_samples=fidelity_samples,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                track_norm_infidelity=track_norm_infidelity,
                track_infidelity=track_infidelity,
            )
            return self.p

        if self.mode == "exact":
            self._run_exact(
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fidelity_samples=fidelity_samples,
            )
            return self.p

        raise ValueError(f"Unknown mode: {self.mode}")

    def _validate_event_stream_for_run(self, G_seq, where_seq, event_seq):
        """Validate queued event metadata before replay."""
        if not (len(G_seq) == len(where_seq) == len(event_seq)):
            raise ValueError(
                "MpsOptimizer event stream metadata is inconsistent: "
                "payloads, wheres, and event types must have the same length."
            )

        unknown = sorted(set(event_seq) - {"gate", "submpo"})
        if unknown:
            raise ValueError(f"Unknown MPS stream event type(s): {unknown!r}.")

        has_submpo = any(event_type == "submpo" for event_type in event_seq)
        if has_submpo and self.mode != "mpo":
            raise ValueError("subMPO stream events currently require mode='mpo'.")

        if not has_submpo:
            return

        L = int(getattr(self.p, "L", 0))
        for step, (where, event_type) in enumerate(
            zip(where_seq, event_seq),
            start=1,
        ):
            if event_type != "submpo":
                continue
            if len(set(where)) != len(where):
                raise ValueError(
                    f"subMPO event at step {step} has repeated site(s): {where!r}."
                )
            out_of_range = [site for site in where if site < 0 or site >= L]
            if out_of_range:
                raise ValueError(
                    f"subMPO event at step {step} references site(s) outside "
                    f"the MPS range [0, {L}): {out_of_range!r}."
                )

    @staticmethod
    def _normalize_fidelity_samples(fidelity_samples):
        """Validate and normalize fidelity-sample count."""
        if fidelity_samples is None:
            return None
        samples = int(fidelity_samples)
        if samples < 0:
            raise ValueError("fidelity_samples must be >= 0.")
        return samples

    @classmethod
    def _resolve_fidelity_samples(cls, fidelity_samples, *, non_unitary):
        """Resolve run-time norm-proxy sampling defaults."""
        if fidelity_samples is None:
            return None if non_unitary else 10
        return cls._normalize_fidelity_samples(fidelity_samples)

    @staticmethod
    def _sampling_steps(total_steps, fidelity_samples):
        """Return gate-step indices at which to sample norm proxy."""
        if total_steps <= 0 or fidelity_samples is None:
            return set()

        samples = MpsOptimizer._normalize_fidelity_samples(fidelity_samples)
        sample_steps = set()

        if total_steps > 1 and samples > 0:
            interior_count = min(samples, total_steps - 1)
            for step in np.linspace(1, total_steps - 1, num=interior_count, dtype=int):
                sample_steps.add(int(step))

        # Always include final progress step.
        sample_steps.add(total_steps)
        return sample_steps

    @staticmethod
    def _real_float(value):
        """Convert backend scalar/tensor-like values to Python float (real part)."""
        real_value = ar.do("real", value)
        item = getattr(real_value, "item", None)
        if callable(item):
            try:
                real_value = item()
            except TypeError:
                pass
        return float(real_value)

    def _append_norm_proxy_sample(self, p):
        """Append current state norm proxy as a real float and return it."""
        norm_val = self._real_float(p.norm())
        self.losses.append(norm_val)
        return norm_val

    def _record_true_fidelity_progress(self, local_fidelity):
        """Append stable geometric-mean true fidelity progress."""
        fidelity = self._clip_fidelity(local_fidelity)
        self._true_fidelity_count += 1

        if fidelity <= 0.0 or np.isneginf(self._true_fidelity_log_sum):
            self._true_fidelity_log_sum = -np.inf
        else:
            self._true_fidelity_log_sum += float(np.log(fidelity))

        if np.isneginf(self._true_fidelity_log_sum):
            cumulative_fidelity = 0.0
            geometric_fidelity = 0.0
        else:
            cumulative_fidelity = float(np.exp(self._true_fidelity_log_sum))
            geometric_fidelity = float(
                np.exp(self._true_fidelity_log_sum / self._true_fidelity_count)
            )

        self.losses.append(geometric_fidelity)
        return cumulative_fidelity, geometric_fidelity

    @staticmethod
    def _accumulate_exponent(p, scale):
        """Accumulate an extracted multiplicative ``scale`` into ``p.exponent``."""
        if hasattr(p, "exponent"):
            p.exponent = p.exponent + ar.do("log10", ar.do("abs", scale))

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

        if MpsOptimizer._class_norm_includes_exponent(p):
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
            return raw_norm * (10**scale_power)

        p.norm = types.MethodType(_norm_with_exponent, p)
        p._pepsy_norm_includes_exponent = True
        return p

    @staticmethod
    def _normalize_span(where):
        """Return ``(xmin, xmax)`` for an int, singleton, or two-site span."""
        if isinstance(where, Integral):
            site = int(where)
            return site, site
        if len(where) == 1:
            site = int(where[0])
            return site, site
        if len(where) == 2:
            site0, site1 = int(where[0]), int(where[1])
            return min(site0, site1), max(site0, site1)
        raise ValueError("where must be an int, (int,), or (int, int).")

    def _canonical_span_norm(self, p, where, *, fallback=True):
        """Return the raw working-data norm without densifying wide spans.

        Normalization needs the norm of the current tensor data, excluding the
        accumulated ``p.exponent`` scale. Contracting a canonical span into a
        dense block can explode for long-range gates because the span retains
        all physical legs, so use ``tn_norm``'s double-layer contraction instead.
        """
        _ = where, fallback
        exponent = getattr(p, "exponent", None)
        try:
            if exponent is not None:
                p.exponent = 0.0
            mantissa, exponent_sq = tn_norm(
                p,
                contraction_opt=self.contraction_opt,
                strip_exponent=True,
            )
            return ar.do("sqrt", ar.do("abs", mantissa)) * 10 ** (
                float(exponent_sq) / 2.0
            )
        finally:
            if exponent is not None:
                p.exponent = exponent

    @staticmethod
    def _norm_ratio_fidelity(approx_norm, target_norm):
        """Return clipped ``(||approx|| / ||target||)**2``."""
        approx = MpsOptimizer._real_float(approx_norm)
        target = MpsOptimizer._real_float(target_norm)

        if target <= 0.0:
            return 1.0 if approx <= 0.0 else 0.0

        fidelity = (approx / target) ** 2
        return min(1.0, max(0.0, float(fidelity)))

    @staticmethod
    def _clip_fidelity(value):
        """Convert a fidelity scalar to ``[0, 1]`` as a Python float."""
        fidelity = MpsOptimizer._real_float(value)
        return min(1.0, max(0.0, fidelity))

    def _append_true_infidelity_sample(self, target, approx, *, step, where):
        """Append cumulative true infidelity for a compressed update."""
        local_fidelity = self._clip_fidelity(
            tn_fidelity(
                approx,
                target,
                contraction_opt=self.contraction_opt,
            )
        )
        cumulative_fidelity, geometric_fidelity = self._record_true_fidelity_progress(
            local_fidelity
        )
        cumulative_infidelity = 1.0 - cumulative_fidelity

        sample = {
            "step": int(step),
            "where": tuple(where),
            "fidelity": local_fidelity,
            "geometric_fidelity": geometric_fidelity,
            "local_infidelity": 1.0 - local_fidelity,
            "infidelity": cumulative_infidelity,
        }
        self.infidelity_samples.append(sample)
        self.true_infidelities.append(cumulative_infidelity)
        self.infidelities.append(cumulative_infidelity)
        return sample

    def _append_norm_infidelity_sample(
        self,
        approx_norm,
        target_norm,
        *,
        step,
        where,
        record_trace=True,
    ):
        """Append cumulative norm-ratio infidelity for a compressed update."""
        local_fidelity = self._norm_ratio_fidelity(approx_norm, target_norm)
        self._norm_fidelity_proxy *= local_fidelity
        cumulative_infidelity = 1.0 - self._norm_fidelity_proxy

        sample = {
            "step": int(step),
            "where": tuple(where),
            "target_norm": self._real_float(target_norm),
            "approx_norm": self._real_float(approx_norm),
            "local_infidelity": 1.0 - local_fidelity,
            "infidelity": cumulative_infidelity,
        }
        self.norm_infidelity_samples.append(sample)
        if record_trace:
            self.infidelities.append(cumulative_infidelity)
        return cumulative_infidelity

    def _build_norm_target(self, p, gate, where, cutoff, cutoff_mode="rsum2"):
        """Build the pre-chi-compression target used for norm diagnostics."""
        p_target = p.copy()
        if len(where) == 1:
            self._apply_gate(
                p_target,
                gate,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )
            return p_target

        return self._apply_gate(
            p_target,
            gate,
            where,
            contract="split-gate",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            inplace=False,
        )

    @staticmethod
    def _normalize_every_interval(normalize_every, non_unitary=False):
        """Return whether non-unitary local scale control is enabled.

        Normalization is only meaningful for non-unitary streams. Callers
        validate explicit normalization requests before this helper is reached.
        """
        if not non_unitary:
            return None
        if normalize_every is None or normalize_every is False:
            return None
        if normalize_every is True:
            return True
        if not isinstance(normalize_every, Integral):
            raise TypeError("normalize_every must be a positive integer, bool, or None.")

        interval = int(normalize_every)
        if interval < 1:
            raise ValueError("normalize_every must be >= 1 when enabled.")
        return True

    @staticmethod
    def _normalization_insert_site(p, fallback_span):
        """Choose an insertion site inside ``fallback_span``."""
        span = MpsOptimizer._normalize_span(fallback_span)
        try:
            lo, hi = p.calc_current_orthog_center()
            site = int(hi if hi is not None else lo)
            if span[0] <= site <= span[1]:
                return site
        except Exception:  # pragma: no cover - defensive for quimb variants
            pass
        return int(span[-1])

    @staticmethod
    def _tensor_local_scale(data):
        """Return a cheap local Frobenius scale for one tensor's data."""
        return ar.do("linalg.norm", data)

    @staticmethod
    def _accumulate_exponent_log10(p, log10_scale):
        """Accumulate an extracted base-10 log scale into ``p.exponent``."""
        if hasattr(p, "exponent"):
            p.exponent = p.exponent + log10_scale

    @staticmethod
    def _event_old_norm_from_log10(log10_old_norm):
        """Return a float old-norm value from its base-10 log when possible."""
        if not np.isfinite(log10_old_norm):
            return np.inf if log10_old_norm > 0.0 else 0.0
        max_log10 = np.log10(np.finfo(float).max)
        if log10_old_norm > max_log10:
            return np.inf
        if log10_old_norm < -max_log10:
            return 0.0
        return float(10.0**log10_old_norm)

    def _normalize_orthog_tensors(
        self,
        p,
        where,
        *,
        step,
        reason,
        canonicalize=False,
    ):
        """Rescale tensors in the active orthogonality span and track exponent."""
        fallback_span = self._normalize_span(where)
        if canonicalize:
            span = self.canonize_mps(p, fallback_span)
        else:
            try:
                span = self._current_orthog(p)
            except Exception:  # pragma: no cover - defensive for quimb variants
                span = fallback_span
            if not (span[0] <= fallback_span[0] and fallback_span[1] <= span[1]):
                span = fallback_span
                self.info_c["cur_orthog"] = span

        sites = tuple(range(int(span[0]), int(span[1]) + 1))
        scaled_sites = []
        scales = []
        log10_scales = []
        log10_total_scale = 0.0

        for site in sites:
            data = p[site].data
            scale = self._tensor_local_scale(data)
            scale_abs = ar.do("abs", scale)
            scale_float = self._real_float(scale_abs)
            if scale_float == 0.0 or not np.isfinite(scale_float):
                continue

            p[site].modify(data=data / scale)
            log10_scale = ar.do("log10", scale_abs)
            self._accumulate_exponent_log10(p, log10_scale)

            log10_scale_float = self._real_float(log10_scale)
            scaled_sites.append(int(site))
            scales.append(scale_float)
            log10_scales.append(log10_scale_float)
            log10_total_scale += log10_scale_float

        if not scaled_sites:
            return None

        self.info_c["cur_orthog"] = span
        insert = self._normalization_insert_site(p, span)
        event = {
            "step": int(step),
            "old_norm": self._event_old_norm_from_log10(2.0 * log10_total_scale),
            "span": tuple(span),
            "insert": int(insert),
            "sites": tuple(scaled_sites),
            "scales": tuple(scales),
            "log10_scale": float(log10_total_scale),
            "log10_scales": tuple(log10_scales),
            "reason": str(reason),
            "method": "local_tensors",
            "exponent": self._real_float(getattr(p, "exponent", 0.0)),
        }
        self.normalizations.append(event)
        return event

    def _normalize_in_canonical_range(self, p, where, *, step, eps=1e-15):
        """Canonicalize ``where`` and apply local tensor scale control."""
        _ = eps
        return self._normalize_orthog_tensors(
            p,
            where,
            step=step,
            reason="final",
            canonicalize=True,
        )

    def _maybe_normalize_after_compression(
        self,
        p,
        *,
        step,
        where,
        normalize_every,
    ):
        """Apply local scale control after a compressed update when enabled."""
        if normalize_every is not None:
            return self._normalize_orthog_tensors(
                p,
                where,
                step=step,
                reason="compression",
                canonicalize=False,
            )
        return None

    def _maybe_normalize_final(
        self,
        p,
        *,
        step,
        last_normalized_step,
        where,
        normalize_every,
        normalize_final,
        normalize_eps,
    ):
        """Optionally normalize at run end if local scale control was active."""
        if (
            normalize_every is not None
            and normalize_final
            and step > 0
            and last_normalized_step != step
        ):
            return self._normalize_in_canonical_range(
                p,
                where,
                step=step,
                eps=normalize_eps,
            )
        return None

    @staticmethod
    def _format_progress_scalar(value):
        """Format displayed progress scalar with stable precision."""
        return f"{MpsOptimizer._real_float(value):.6f}"

    @staticmethod
    def _format_progress_infidelity(value):
        """Format progress infidelity in compact scientific notation."""
        text = f"{MpsOptimizer._real_float(value):#.0e}"
        if "e" not in text:
            return text
        mantissa, exponent = text.split("e", 1)
        sign = exponent[0] if exponent[:1] in "+-" else ""
        digits = exponent[1:] if sign else exponent
        digits = digits.lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"

    @staticmethod
    def _collect_dmrg_batch(G_seq, where_seq, start_idx, k_2q_batch):
        """Collect a DMRG batch starting at a two-qubit gate index."""
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = where_seq[idx]
            gate = G_seq[idx]
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

    def _build_dmrg_batch_target(self, p, batch_G, batch_where, cutoff, cutoff_mode="rsum2"):
        """Apply a collected DMRG batch onto a copy of ``p``."""
        p_g = p.copy()
        for gate, where in zip(batch_G, batch_where):
            if len(where) == 1:
                self._apply_gate(
                    p_g,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
            else:
                p_g = self._apply_gate(
                    p_g,
                    gate,
                    where,
                    contract="split-gate",
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=False,
                )
        return p_g

    def _run_dmrg(
        self,
        G_seq,
        where_seq,
        n_iter,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        k_2q_batch=1,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
        track_infidelity=False,
        record_fit_losses=True,
    ):
        """Apply gates with local DMRG-style fitting."""
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")

        p = self.p
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None
        true_cumulative_infidelity = None

        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="dmrg",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["dmrg"],
            )
        else:
            pbar = None

        idx = 0
        while idx < len(G_seq):
            compressed = False
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
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")

                if k_2q_batch == 1:
                    two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    self.canonize_mps(p, (xmin, xmax))

                    p_g = self._apply_gate(
                        p,
                        gate,
                        where,
                        contract="split-gate",
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        inplace=False,
                    )
                    target_norm = (
                        self._canonical_span_norm(p_g, (xmin, xmax))
                        if track_norm_infidelity
                        else None
                    )
                    fit = FIT(
                        p_g,
                        p=p,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=False,
                    )
                    fit.run_gate(n_iter=n_iter, verbose=False)

                    p = self._install_represented_norm(fit.p)
                    if record_fit_losses:
                        self.losses.append(self._real_float(fit.local_norm_trace[-1]))
                    if track_norm_infidelity:
                        self._append_norm_infidelity_sample(
                            self._canonical_span_norm(p, (xmin, xmax)),
                            target_norm,
                            step=idx + 1,
                            where=(xmin, xmax),
                            record_trace=not track_infidelity,
                        )
                    if track_infidelity:
                        sample = self._append_true_infidelity_sample(
                            p_g,
                            p,
                            step=idx + 1,
                            where=(xmin, xmax),
                        )
                        true_cumulative_infidelity = sample["infidelity"]
                    idx += 1
                    advanced = 1
                    last_where = (xmin, xmax)
                    compressed = True
                else:
                    batch_G, batch_where, two_qubit_in_batch, next_idx = self._collect_dmrg_batch(
                        G_seq, where_seq, idx, k_2q_batch
                    )
                    if two_qubit_in_batch < 1:
                        raise RuntimeError("DMRG batch unexpectedly contains no two-qubit gates.")

                    two_qubit_count += two_qubit_in_batch
                    batch_span_sites = [site for where_i in batch_where for site in where_i]
                    xmin, xmax = min(batch_span_sites), max(batch_span_sites)
                    self.canonize_mps(p, (xmin, xmax))
                    p_g = self._build_dmrg_batch_target(p, batch_G, batch_where, cutoff, cutoff_mode)
                    target_norm = (
                        self._canonical_span_norm(p_g, (xmin, xmax))
                        if track_norm_infidelity
                        else None
                    )
                    fit = FIT(
                        p_g,
                        p=p,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                    )
                    fit.run_gate(n_iter=n_iter, verbose=False)

                    p = self._install_represented_norm(fit.p)
                    if record_fit_losses:
                        self.losses.append(self._real_float(fit.local_norm_trace[-1]))
                    if track_norm_infidelity:
                        self._append_norm_infidelity_sample(
                            self._canonical_span_norm(p, (xmin, xmax)),
                            target_norm,
                            step=next_idx,
                            where=(xmin, xmax),
                            record_trace=not track_infidelity,
                        )
                    if track_infidelity:
                        sample = self._append_true_infidelity_sample(
                            p_g,
                            p,
                            step=next_idx,
                            where=(xmin, xmax),
                        )
                        true_cumulative_infidelity = sample["infidelity"]
                    advanced = next_idx - idx
                    idx = next_idx
                    last_where = (xmin, xmax)
                    compressed = True

            event = (
                self._maybe_normalize_after_compression(
                    p,
                    step=idx,
                    where=last_where,
                    normalize_every=normalize_every,
                )
                if compressed
                else None
            )
            if event is not None:
                last_normalized_step = idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if track_infidelity and true_cumulative_infidelity is not None:
                    postfix["Icum"] = self._format_progress_infidelity(true_cumulative_infidelity)
                elif record_fit_losses:
                    postfix["~F"] = self._format_progress_scalar(self.losses[-1])
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
        fidelity_samples=10,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
        track_infidelity=False,
        submpo_method="direct",
    ):
        """Apply gates with MPO-style nonlocal compression.

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        """
        p = self.p
        two_qubit_count = 0
        submpo_count = 0
        sample_steps = (
            set()
            if track_infidelity
            else self._sampling_steps(len(G_seq), fidelity_samples)
        )
        norm_proxy = (
            None
            if track_infidelity or fidelity_samples is None
            else self.losses[-1]
        )
        true_cumulative_infidelity = None
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="mpo",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["mpo"],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            where = where_seq[idx]
            gate = G_seq[idx]
            event_type = event_seq[idx]
            if event_type == "submpo":
                submpo_count += 1
                xmin, xmax = min(where), max(where)
                method = self._normalize_submpo_method(submpo_method)
                submpo_compress_opts = self._submpo_compress_opts(method)
                if track_norm_infidelity:
                    self.canonize_mps(p, (xmin, xmax))
                if track_norm_infidelity or track_infidelity:
                    p_target = p.copy()
                    p_target.gate_with_submpo_(
                        gate,
                        where=where,
                        method=method,
                        max_bond=None,
                        info={},
                        inplace_mpo=False,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        **submpo_compress_opts,
                    )
                else:
                    p_target = None
                if track_norm_infidelity:
                    target_norm = self._canonical_span_norm(p_target, (xmin, xmax))
                else:
                    target_norm = None

                p.gate_with_submpo_(
                    gate,
                    where=where,
                    method=method,
                    max_bond=self.chi,
                    info=self.info_c,
                    inplace_mpo=False,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    **submpo_compress_opts,
                )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                        record_trace=not track_infidelity,
                    )
                if track_infidelity:
                    sample = self._append_true_infidelity_sample(
                        p_target,
                        p,
                        step=idx,
                        where=(xmin, xmax),
                    )
                    true_cumulative_infidelity = sample["infidelity"]
            elif len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = (
                    self._has_symmray_data(p)
                    and not self._is_nearest_neighbor_1d(where)
                )
                if track_norm_infidelity:
                    self.canonize_mps(p, (xmin, xmax))
                if track_norm_infidelity or track_infidelity:
                    if use_symmray_auto_swap:
                        p_target = self._build_symmray_auto_swap_target(
                            p,
                            gate,
                            where,
                            cutoff,
                            cutoff_mode,
                        )
                    else:
                        p_target = self._build_norm_target(
                            p,
                            gate,
                            where,
                            cutoff,
                            cutoff_mode,
                        )
                else:
                    p_target = None
                if track_norm_infidelity:
                    target_norm = self._canonical_span_norm(p_target, (xmin, xmax))
                else:
                    target_norm = None
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
                    p.gate_nonlocal_(
                        gate,
                        where,
                        dims=self._infer_gate_dims(gate, where),
                        max_bond=self.chi,
                        info=self.info_c,
                        method="direct",
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                        record_trace=not track_infidelity,
                    )
                if track_infidelity:
                    sample = self._append_true_infidelity_sample(
                        p_target,
                        p,
                        step=idx,
                        where=(xmin, xmax),
                    )
                    true_cumulative_infidelity = sample["infidelity"]

            event = (
                self._maybe_normalize_after_compression(
                    p,
                    step=idx,
                    where=last_where,
                    normalize_every=normalize_every,
                )
                if compressed
                else None
            )
            if event is not None:
                last_normalized_step = idx

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if submpo_count:
                    postfix["mpo"] = submpo_count
                if track_infidelity and true_cumulative_infidelity is not None:
                    postfix["Icum"] = self._format_progress_infidelity(true_cumulative_infidelity)
                elif norm_proxy is not None:
                    postfix["~F"] = self._format_progress_scalar(norm_proxy)
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

    def _run_swap(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        fidelity_samples=10,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
        track_infidelity=False,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = (
            set()
            if track_infidelity
            else self._sampling_steps(len(G_seq), fidelity_samples)
        )
        norm_proxy = (
            None
            if track_infidelity or fidelity_samples is None
            else self.losses[-1]
        )
        true_cumulative_infidelity = None
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="swap",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["swap"],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
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
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                if track_norm_infidelity:
                    self.canonize_mps(p, (xmin, xmax))
                if track_norm_infidelity or track_infidelity:
                    p_target = self._build_norm_target(
                        p,
                        gate,
                        where,
                        cutoff,
                        cutoff_mode,
                    )
                else:
                    p_target = None
                if track_norm_infidelity:
                    target_norm = self._canonical_span_norm(p_target, (xmin, xmax))
                else:
                    target_norm = None

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                p.gate_with_auto_swap_(
                    gate,
                    where,
                    info=self.info_c,
                    max_bond=self.chi,
                    swap_back=True,
                    **compress_opts,
                )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                        record_trace=not track_infidelity,
                    )
                if track_infidelity:
                    sample = self._append_true_infidelity_sample(
                        p_target,
                        p,
                        step=idx,
                        where=(xmin, xmax),
                    )
                    true_cumulative_infidelity = sample["infidelity"]

            event = (
                self._maybe_normalize_after_compression(
                    p,
                    step=idx,
                    where=last_where,
                    normalize_every=normalize_every,
                )
                if compressed
                else None
            )
            if event is not None:
                last_normalized_step = idx

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if track_infidelity and true_cumulative_infidelity is not None:
                    postfix["Icum"] = self._format_progress_infidelity(true_cumulative_infidelity)
                elif norm_proxy is not None:
                    postfix["~F"] = self._format_progress_scalar(norm_proxy)
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
        fidelity_samples=10,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
        track_infidelity=False,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``. Symmray-backed
        MPS data use quimb's block-aware auto-swap split path instead, since
        ``reduce-split`` loses Symmray fusion metadata in current quimb/Symmray.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = (
            set()
            if track_infidelity
            else self._sampling_steps(len(G_seq), fidelity_samples)
        )
        norm_proxy = (
            None
            if track_infidelity or fidelity_samples is None
            else self.losses[-1]
        )
        true_cumulative_infidelity = None
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
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = self._has_symmray_data(p)
                if track_norm_infidelity:
                    self.canonize_mps(p, (xmin, xmax))
                if use_symmray_auto_swap:
                    if track_norm_infidelity or track_infidelity:
                        p_target = self._build_symmray_auto_swap_target(
                            p,
                            gate,
                            where,
                            cutoff,
                            cutoff_mode,
                        )
                    else:
                        p_target = None
                    target_norm = (
                        self._canonical_span_norm(p_target, (xmin, xmax))
                        if track_norm_infidelity
                        else None
                    )
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
                    target_norm = (
                        self._canonical_span_norm(p, (xmin, xmax))
                        if track_norm_infidelity
                        else None
                    )
                    p_target = p.copy() if track_infidelity else None
                    self.canonize_mps(p, (xmin, xmax))

                    for i in range(xmax, xmin, -1):
                        p.right_canonize_site(i, bra=None)
                    p.left_compress(
                        start=xmin,
                        stop=xmax,
                        max_bond=self.chi,
                        **compress_opts,
                    )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                        record_trace=not track_infidelity,
                    )
                if track_infidelity:
                    sample = self._append_true_infidelity_sample(
                        p_target,
                        p,
                        step=idx,
                        where=(xmin, xmax),
                    )
                    true_cumulative_infidelity = sample["infidelity"]

            event = (
                self._maybe_normalize_after_compression(
                    p,
                    step=idx,
                    where=last_where,
                    normalize_every=normalize_every,
                )
                if compressed
                else None
            )
            if event is not None:
                last_normalized_step = idx

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "bnd": p.max_bond(),
                }
                if track_infidelity and true_cumulative_infidelity is not None:
                    postfix["Icum"] = self._format_progress_infidelity(true_cumulative_infidelity)
                elif norm_proxy is not None:
                    postfix["~F"] = self._format_progress_scalar(norm_proxy)
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

    def _run_exact(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        fidelity_samples=10,
    ):
        """Apply gates exactly using in-place ``contract=True`` application.

        Progress bar counts all gates for consistency with other modes.
        """
        self.p = self.p.contract(all, optimize="auto-hq")
        self.p = self._install_represented_norm(qtn.TensorNetwork([self.p]))
        p = self.p
        two_qubit_count = 0
        # Keep parameter for API compatibility; exact mode does not sample fidelity.
        _ = fidelity_samples

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

            if len(where) == 1:
                if pbar is not None:
                    pbar.set_postfix(
                        {"2q": two_qubit_count, "~F": self._format_progress_scalar(1.0), "bnd": "inf"}
                    )
                    pbar.update(1)
                continue

            two_qubit_count += 1
            if pbar is not None:
                pbar.set_postfix({"2q": two_qubit_count, "~F": self._format_progress_scalar(1.0), "bnd": "inf"})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = self._install_represented_norm(p)

    def canonize_mps(self, p, where):
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

        p.canonize(where_canon, cur_orthog=self._current_orthog(p))
        # Preserve the fitting-window semantics expected by gate updates.
        self.info_c["cur_orthog"] = target_orthog
        return target_orthog

    def get_fidelities(self):
        """Return the active fidelity-like progress trace.

        This returns ``self.losses`` for compatibility. With
        ``track_infidelity=True``, entries after the initial ``1.0`` are the
        running geometric mean of measured local true fidelities. Otherwise,
        default/unitary compressed runs use this as the legacy represented-norm
        proxy trace sampled by ``fidelity_samples``. For ``non_unitary=True``
        without diagnostics this trace stays at ``[1.0]`` unless
        ``fidelity_samples`` is supplied explicitly.
        """
        return self.losses

    def get_infidelities(self):
        """Return the cumulative infidelity trace.

        The initial value is ``0.0``. A new value is appended for each
        compressed two-site update sampled while ``track_infidelity=True`` or,
        when that is disabled, ``track_norm_infidelity=True``.
        """
        return self.infidelities

    def get_true_infidelities(self):
        """Return the cumulative true normalized-overlap infidelity trace."""
        return self.true_infidelities

    def get_infidelity_samples(self):
        """Return detailed true normalized-overlap infidelity sample records.

        Each record contains ``step``, ``where``, local ``fidelity``, running
        ``geometric_fidelity``, ``local_infidelity``, and cumulative
        ``infidelity``.
        """
        return self.infidelity_samples

    def get_norm_infidelity_samples(self):
        """Return detailed norm-ratio infidelity sample records.

        Each record contains ``step``, ``where``, ``target_norm``,
        ``approx_norm``, ``local_infidelity``, and cumulative ``infidelity``.
        """
        return self.norm_infidelity_samples

    def get_normalizations(self):
        """Return automatic normalization events recorded during ``run``.

        Each event contains the 1-based ``step``, removed local ``old_norm``,
        active ``span``, rescaled ``sites``, per-tensor ``scales``, total
        ``log10_scale``, event ``reason``, and resulting base-10 ``exponent``.
        """
        return self.normalizations
