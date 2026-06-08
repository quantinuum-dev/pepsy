"""MPS optimization helpers centered on :class:`MpsOptimizer`.

:class:`MpsOptimizer` replays a canonical bundled gate stream
``[(gate, where), ...]`` against an MPS, using one of several compression
backends.  Non-unitary streams can opt into norm-aware canonicalization with
``non_unitary=True`` or ``normalize_every=...``; this keeps the working MPS
normalized while recording the removed norm factors and, for compressed
two-site updates, a norm-ratio infidelity proxy.
"""

from __future__ import annotations

from numbers import Integral
import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..fitting.local import FIT
from ..operators.gates import _normalize_gate_entries, gate as apply_gate

__all__ = ["MpsOptimizer"]


def _normalize_gate_queue(gates):
    """Return ``(gate_list, where_list)`` from canonical bundled stream input."""
    entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], []
    gate_list, where_list = zip(*entries)
    return list(gate_list), [tuple(w) if isinstance(w, list) else w for w in where_list]


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
        stores the 1-based gate step, previous norm, canonicalization span, and
        tensor site where the normalization factor was inserted.
    infidelities : list[float]
        Cumulative norm-ratio infidelity proxy. For compressed non-unitary
        updates this uses ``1 - product((||approx|| / ||target||)**2)`` so
        physical norm changes from the gate are divided out.
    """

    _ALLOWED_MODES = frozenset({"dmrg", "mpo", "swap", "svd", "exact"})
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
        self.p = p if self.inplace else p.copy()
        self.G, self.where = _normalize_gate_queue(gates)
        self.chi = chi
        self.mode = self._normalize_mode(mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.ind_id = str(ind_id)

        self.info_c = {}
        self.losses = [1.0]
        self.normalizations = []
        self.infidelities = [0.0]
        self.norm_infidelity_samples = []
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
        self.p = p if self.inplace else p.copy()
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
        """
        old_norm = self.p.normalize(eps=eps, insert=insert)
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
        self.G, self.where = _normalize_gate_queue(gates)
        return self

    def add_gates(self, gates):
        """Append gates to the existing gate list.

        This preserves previously queued gates and extends them with
        new ones.
        """
        G_new, where_new = _normalize_gate_queue(gates)
        self.G.extend(G_new)
        self.where.extend(where_new)
        return self

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=5,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        mode=None,
        fidelity_samples=10,
        k_2q_batch=1,
        non_unitary=False,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
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
        cutoff_mode : str, default="rel"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``.
        mode : {"dmrg", "mpo", "swap", "svd", "exact"} | None, default=None
            Optional mode override for this run. If supplied, updates
            ``self.mode`` before execution.
        fidelity_samples : int, default=10
            Compression modes (``mpo``/``swap``/``svd``): number of
            intermediate norm-proxy samples taken during the run.
            A final sample is always recorded at the end.
        k_2q_batch : int, default=1
            DMRG mode only: number of sequential two-qubit gates to batch
            into one local FIT update. The FIT window uses the batch-wide
            ``[xmin, xmax]`` from all two-qubit gate locations in the batch.
        non_unitary : bool, default=False
            Convenience flag for non-unitary gate streams. If ``True`` and
            ``normalize_every`` is omitted, normalize at every gate-count
            boundary. Batched DMRG updates normalize once at the batch end.
        normalize_every : int | bool | None, default=None
            Periodically normalize the MPS after this many queued gate steps.
            Batches that cross an interval normalize once at the batch end.
            The normalization factor is inserted inside the latest
            canonicalization range. Disabled by default; ``True`` means every
            step and ``False`` disables normalization.
        normalize_final : bool, default=True
            If periodic normalization is enabled, also normalize once at the
            end of the run when the final gate did not land on an interval.
        normalize_eps : float, default=1e-15
            Precision passed to :meth:`qtn.MatrixProductState.normalize`.

        Returns
        -------
        qtn.TensorNetwork
            The updated ``self.p`` state after replaying the queued gate stream.
        """
        if mode is not None:
            self.set_mode(mode)

        G_seq = list(self.G)
        where_seq = list(self.where)
        if not G_seq:
            return self.p

        normalize_every = self._normalize_every_interval(
            normalize_every,
            non_unitary=non_unitary,
        )
        if normalize_every is not None and self.mode == "exact":
            raise ValueError(
                "automatic normalization uses MPS canonicalization and is not "
                "available in exact mode."
            )
        track_norm_infidelity = normalize_every is not None

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
            )
            return self.p

        if self.mode == "mpo":
            self._run_mpo(
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

    @staticmethod
    def _normalize_fidelity_samples(fidelity_samples):
        """Validate and normalize fidelity-sample count."""
        samples = int(fidelity_samples)
        if samples < 0:
            raise ValueError("fidelity_samples must be >= 0.")
        return samples

    @staticmethod
    def _sampling_steps(total_steps, fidelity_samples):
        """Return gate-step indices at which to sample norm proxy."""
        if total_steps <= 0:
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
        """Return ``||p||`` from a canonical center/span block.

        This assumes tensors outside ``where`` are already isometric. The span
        itself can contain multiple tensors per site, e.g. after ``split-gate``.
        """
        xmin, xmax = self._normalize_span(where)
        try:
            tags = [p.site_tag(i) for i in range(xmin, xmax + 1)]
            block = p.select(tags, which="any")
            if isinstance(block, qtn.TensorNetwork):
                if block.num_tensors == 0:
                    raise ValueError("canonical span selected no tensors.")
                block = block.contract(
                    all,
                    output_inds=block.outer_inds(),
                    optimize=self.contraction_opt,
                )
            return ar.do("linalg.norm", block.data)
        except Exception:
            if not fallback:
                raise
            return p.norm(optimize=self.contraction_opt)

    @staticmethod
    def _norm_ratio_fidelity(approx_norm, target_norm):
        """Return clipped ``(||approx|| / ||target||)**2``."""
        approx = MpsOptimizer._real_float(approx_norm)
        target = MpsOptimizer._real_float(target_norm)

        if target <= 0.0:
            return 1.0 if approx <= 0.0 else 0.0

        fidelity = (approx / target) ** 2
        return min(1.0, max(0.0, float(fidelity)))

    def _append_norm_infidelity_sample(self, approx_norm, target_norm, *, step, where):
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
        self.infidelities.append(cumulative_infidelity)
        return cumulative_infidelity

    @staticmethod
    def _build_norm_target(p, gate, where, cutoff, cutoff_mode="rel"):
        """Build the pre-chi-compression target used for norm diagnostics."""
        p_target = p.copy()
        if len(where) == 1:
            apply_gate(
                p_target,
                gate,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )
            return p_target

        return apply_gate(
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
        """Return periodic normalization interval, or ``None`` if disabled."""
        if normalize_every is None:
            return 1 if non_unitary else None
        if normalize_every is False:
            return None
        if normalize_every is True:
            return 1
        if not isinstance(normalize_every, Integral):
            raise TypeError("normalize_every must be a positive integer, bool, or None.")

        interval = int(normalize_every)
        if interval < 1:
            raise ValueError("normalize_every must be >= 1 when enabled.")
        return interval

    @staticmethod
    def _normalization_due(prev_step, step, normalize_every):
        """Return whether a periodic normalization boundary was crossed."""
        if normalize_every is None:
            return False
        return step // normalize_every > prev_step // normalize_every

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

    def _normalize_in_canonical_range(self, p, where, *, step, eps=1e-15):
        """Canonicalize ``where``, normalize within that range, and record it."""
        span = self.canonize_mps(p, where)
        insert = self._normalization_insert_site(p, span)
        norm = self._canonical_span_norm(p, span)
        old_norm = norm**2
        try:
            p[insert].modify(data=p[insert].data / norm)
        except Exception:
            old_norm = p.normalize(eps=eps, insert=insert)
        self.info_c["cur_orthog"] = span

        event = {
            "step": int(step),
            "old_norm": self._real_float(old_norm),
            "span": tuple(span),
            "insert": int(insert),
        }
        self.normalizations.append(event)
        return event

    def _maybe_normalize_after_step(
        self,
        p,
        *,
        prev_step,
        step,
        where,
        normalize_every,
        normalize_eps,
    ):
        """Apply automatic normalization if the configured interval is due."""
        if self._normalization_due(prev_step, step, normalize_every):
            return self._normalize_in_canonical_range(
                p,
                where,
                step=step,
                eps=normalize_eps,
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
        """Optionally normalize at run end if periodic normalization was active."""
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
    def _format_progress_fidelity(value):
        """Format displayed progress fidelity proxy with stable precision."""
        return f"{MpsOptimizer._real_float(value):.6f}"

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

    @staticmethod
    def _build_dmrg_batch_target(p, batch_G, batch_where, cutoff, cutoff_mode="rel"):
        """Apply a collected DMRG batch onto a copy of ``p``."""
        p_g = p.copy()
        for gate, where in zip(batch_G, batch_where):
            if len(where) == 1:
                apply_gate(p_g, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
            else:
                p_g = apply_gate(
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
        cutoff_mode="rel",
        k_2q_batch=1,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
    ):
        """Apply gates with local DMRG-style fitting."""
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")

        p = self.p
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

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
            prev_idx = idx
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
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

                    p_g = apply_gate(
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

                    p = fit.p
                    self.losses.append(self._real_float(fit.local_norm_trace[-1]))
                    if track_norm_infidelity:
                        self._append_norm_infidelity_sample(
                            self._canonical_span_norm(p, (xmin, xmax)),
                            target_norm,
                            step=idx + 1,
                            where=(xmin, xmax),
                        )
                    idx += 1
                    advanced = 1
                    last_where = (xmin, xmax)
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

                    p = fit.p
                    self.losses.append(self._real_float(fit.local_norm_trace[-1]))
                    if track_norm_infidelity:
                        self._append_norm_infidelity_sample(
                            self._canonical_span_norm(p, (xmin, xmax)),
                            target_norm,
                            step=next_idx,
                            where=(xmin, xmax),
                        )
                    advanced = next_idx - idx
                    idx = next_idx
                    last_where = (xmin, xmax)

            event = self._maybe_normalize_after_step(
                p,
                prev_step=prev_idx,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                normalize_eps=normalize_eps,
            )
            if event is not None:
                last_normalized_step = idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_fidelity(self.losses[-1]),
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

        self.p = p

    def _run_mpo(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
    ):
        """Apply gates with MPO-style nonlocal compression.

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]
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
            prev_idx = idx
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
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
                    p_target = self._build_norm_target(
                        p,
                        gate,
                        where,
                        cutoff,
                        cutoff_mode,
                    )
                    target_norm = self._canonical_span_norm(p_target, (xmin, xmax))
                else:
                    target_norm = None
                p.gate_nonlocal_(
                    gate,
                    where,
                    max_bond=self.chi,
                    info=self.info_c,
                    method="direct",
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                )
                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                    )

            event = self._maybe_normalize_after_step(
                p,
                prev_step=prev_idx,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                normalize_eps=normalize_eps,
            )
            if event is not None:
                last_normalized_step = idx

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_fidelity(norm_proxy),
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

        self.p = p

    def _run_swap(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]
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
            prev_idx = idx
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
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
                    p_target = self._build_norm_target(
                        p,
                        gate,
                        where,
                        cutoff,
                        cutoff_mode,
                    )
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
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                    )

            event = self._maybe_normalize_after_step(
                p,
                prev_step=prev_idx,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                normalize_eps=normalize_eps,
            )
            if event is not None:
                last_normalized_step = idx

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_fidelity(norm_proxy),
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

        self.p = p

    def _run_svd(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        track_norm_infidelity=False,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]
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
            prev_idx = idx
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff}
                xmin, xmax = sorted(where)
                if track_norm_infidelity:
                    self.canonize_mps(p, (xmin, xmax))
                apply_gate(
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
                if track_norm_infidelity:
                    self._append_norm_infidelity_sample(
                        self._canonical_span_norm(p, (xmin, xmax)),
                        target_norm,
                        step=idx,
                        where=(xmin, xmax),
                    )

            event = self._maybe_normalize_after_step(
                p,
                prev_step=prev_idx,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                normalize_eps=normalize_eps,
            )
            if event is not None:
                last_normalized_step = idx

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_fidelity(norm_proxy),
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

        self.p = p

    def _run_exact(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
    ):
        """Apply gates exactly using in-place ``contract=True`` application.

        Progress bar counts all gates for consistency with other modes.
        """
        self.p = self.p.contract(all, optimize="auto-hq")
        self.p = qtn.TensorNetwork([self.p])
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
                        {"2q": two_qubit_count, "~F": self._format_progress_fidelity(1.0), "bnd": "inf"}
                    )
                    pbar.update(1)
                continue

            two_qubit_count += 1
            if pbar is not None:
                pbar.set_postfix({"2q": two_qubit_count, "~F": self._format_progress_fidelity(1.0), "bnd": "inf"})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p

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
        """Return the running loss history."""
        return self.losses

    def get_infidelities(self):
        """Return the cumulative norm-ratio infidelity proxy trace.

        The initial value is ``0.0``. A new value is appended for each
        compressed two-site update sampled while norm-aware normalization is
        enabled.
        """
        return self.infidelities

    def get_norm_infidelity_samples(self):
        """Return detailed norm-ratio infidelity sample records.

        Each record contains ``step``, ``where``, ``target_norm``,
        ``approx_norm``, ``local_infidelity``, and cumulative ``infidelity``.
        """
        return self.norm_infidelity_samples

    def get_normalizations(self):
        """Return automatic normalization events recorded during ``run``.

        Each event contains the 1-based ``step``, removed ``old_norm``,
        canonical ``span``, and tensor ``insert`` site that received the
        normalization factor.
        """
        return self.normalizations
