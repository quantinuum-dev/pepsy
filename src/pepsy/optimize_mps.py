"""MPS optimization helpers centered on :class:`MpsOptimizer`."""

from __future__ import annotations

from numbers import Integral
import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .fit import FIT
from .gate import _normalize_gate_entries, gate as apply_gate

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
        applied on the ket family only (state evolution), using :func:`pepsy.gate.gate`.
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
        self._init_canonicalization()

    def _current_orthog(self, p=None):
        """Return cached ``(min_site, max_site)`` orthogonality span."""
        cur = self.info_c.get("cur_orthog", "calc")
        state = self.p if p is None else p
        if cur == "calc" or cur is None:
            lo, hi = state.calc_current_orthog_center()
            cur = (int(lo), int(hi))
        elif isinstance(cur, int):
            cur = (int(cur), int(cur))
        else:
            cur = (int(min(cur)), int(max(cur)))

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
    ):
        """Apply gates with local DMRG-style fitting."""
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")

        p = self.p
        two_qubit_count = 0

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
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
                idx += 1
                advanced = 1
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
                    idx += 1
                    advanced = 1
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
                    advanced = next_idx - idx
                    idx = next_idx

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

        self.p = p

    def _run_mpo(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
    ):
        """Apply gates with MPO-style nonlocal compression.

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

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
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
                idx += 1
                advanced = 1
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
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

        self.p = p

    def _run_swap(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

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
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
                idx += 1
                advanced = 1
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

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

        self.p = p

    def _run_svd(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

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
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                apply_gate(p, gate, where, contract=True, cutoff=cutoff, cutoff_mode=cutoff_mode, inplace=True)
                idx += 1
                advanced = 1
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff}
                apply_gate(
                    p,
                    gate,
                    where,
                    contract="reduce-split",
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                xmin, xmax = sorted(where)
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
        """Update canonical form around a two-site gate span."""
        xmin, xmax = sorted(where)
        p.canonize([xmin, xmax], cur_orthog=self._current_orthog(p))
        # Preserve the original 2-site fitting window semantics.
        self.info_c["cur_orthog"] = (xmin, xmax)

    def get_fidelities(self):
        """Return the running loss history."""
        return self.losses
