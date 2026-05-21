"""MPO optimization helpers centered on :class:`MpoOptimizer`."""

from __future__ import annotations

from numbers import Integral

import autoray as ar
import numpy as np

from .core import tn_norm
from .fit import FIT
from .gates import _normalize_gate_entries, gate as apply_gate, gate_nonlocal_opt

__all__ = ["MpoOptimizer"]


def _normalize_gate_queue(gates):
    """Return ``(gate_list, where_list)`` from canonical bundled stream input."""
    entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], []
    gate_list, where_list = zip(*entries)
    return list(gate_list), [tuple(w) if isinstance(w, list) else w for w in where_list]


class MpoOptimizer:
    """High-level wrapper for MPO gate sweeps.

    Parameters
    ----------
    mpo : qtn.MatrixProductOperator
        Initial MPO.
    gates : sequence[object] | None, optional
        Canonical bundled gate stream ``((gate, where), ...)``. For each entry:
        - ``gate`` applies ``G O G†`` (same gate on ket and bra families),
        - ``(G,)`` applies ket-only (shorthand for ``(G, None)``),
        - ``(G, B)`` applies ``G O B†``,
        - ``(G, None)`` applies ket-only,
        - ``(None, B)`` applies bra-only.
        Either side may be ``None`` to skip it.
    chi : int
        Working bond dimension.
    mode : {"dmrg", "svd", "mpo"}, default="dmrg"
        Execution backend.
    ind_id_k : str, default="k{}"
        Site-index format string for ket-family physical legs.
    ind_id_b : str, default="b{}"
        Site-index format string for bra-family physical legs.
    contraction_opt : object | None, optional
        Contraction path optimizer keyword used inside :class:`FIT`.
    inplace : bool, default=False
        Whether to optimize the provided input MPO object directly. If
        ``False``, a copy is made and the original input remains unchanged.
    """

    _ALLOWED_MODES = frozenset({"dmrg", "svd", "mpo"})

    @classmethod
    def _normalize_mode(cls, mode):
        """Validate and normalize execution mode."""
        mode_norm = str(mode).strip().lower()
        if mode_norm not in cls._ALLOWED_MODES:
            supported = ", ".join(sorted(cls._ALLOWED_MODES))
            raise ValueError(f"Unknown mode: {mode}. Supported modes: {supported}")
        return mode_norm

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mpo,
        gates=None,
        chi=None,
        mode="dmrg",
        ind_id_k="k{}",
        ind_id_b="b{}",
        contraction_opt=None,
        inplace=False,
    ):
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpoOptimizer(mpo, gates, chi) "
                    "or MpoOptimizer(mpo, chi) for an empty gate queue."
                )

        self.inplace = bool(inplace)
        self.p = mpo if self.inplace else mpo.copy()
        self.G, self.where = _normalize_gate_queue(gates)
        self.chi = int(chi)
        self.mode = self._normalize_mode(mode)
        self.ind_id_k = str(ind_id_k)
        self.ind_id_b = str(ind_id_b)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt

        self.norm_mpo = self._measure_norm(self.p)
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

    def _init_canonicalization(self):
        """Initialize canonical form and orthogonality center."""
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    def _prepare_dmrg_state(self):
        """Ensure DMRG starts from at least ``chi`` bond dimension."""
        if self.p.max_bond() < self.chi:
            self.p.expand_bond_dimension(self.chi, inplace=True)
            self._init_canonicalization()

    def set_mpo(self, mpo):
        """Assign a new MPO and reset canonicalization metadata."""
        self.p = mpo if self.inplace else mpo.copy()
        self.norm_mpo = self._measure_norm(self.p)
        self._init_canonicalization()
        return self

    def set_mode(self, mode):
        """Set execution mode."""
        self.mode = self._normalize_mode(mode)
        return self

    def set_gates(self, gates):
        """Replace the current gate queue with canonical bundled entries."""
        self.G, self.where = _normalize_gate_queue(gates)
        return self

    def add_gates(self, gates):
        """Append canonical bundled entries to the existing gate queue."""
        G_new, where_new = _normalize_gate_queue(gates)
        self.G.extend(G_new)
        self.where.extend(where_new)
        return self

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

        samples = MpoOptimizer._normalize_fidelity_samples(fidelity_samples)
        sample_steps = set()

        if total_steps > 1 and samples > 0:
            interior_count = min(samples, total_steps - 1)
            for step in np.linspace(1, total_steps - 1, num=interior_count, dtype=int):
                sample_steps.add(int(step))

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
        """Append current normalized MPO norm and return it."""
        norm_val = self._normalize_norm(self._measure_norm(p))
        self.losses.append(norm_val)
        return norm_val

    def _measure_norm(self, p):
        """Measure the MPO squared norm as (mantissa, exponent) for stability.

        Returns ``(m, e)`` where ``<O|O> = m * 10^e``.
        """
        mantissa, exponent = tn_norm(p, contraction_opt=self.contraction_opt, strip_exponent=True)
        return self._real_float(mantissa), float(exponent)

    def _normalize_norm(self, norm_val):
        """Normalize a (mantissa, exponent) norm against the initial MPO norm.

        Returns sqrt(<O|O>) / sqrt(<O0|O0>), i.e. the ratio of actual norms,
        consistent with p.norm() / initial.norm().
        """
        m, e = norm_val
        m0, e0 = self.norm_mpo
        if m0 == 0.0:
            return 0.0 if m == 0.0 else float("inf")
        return float(np.sqrt(abs(m / m0))) * 10 ** ((e - e0) / 2)

    @staticmethod
    def _prepare_gate_tensor(gate, n_sites):
        """Return a gate tensor in ket-index ordering."""
        if n_sites == 1:
            return ar.do("transpose", gate, (1, 0))
        elif n_sites == 2:
            shape = getattr(gate, "shape", ())
            if len(shape) == 2:
                din, dout = shape
                if int(din) != int(dout):
                    raise ValueError(
                        "Two-site gate matrix must be square with shape (d**2, d**2)."
                    )
                return ar.do("transpose", gate, (1, 0))
            elif len(shape) == 4:
                return ar.do("transpose", gate, (2, 3, 0, 1))
            else:
                raise ValueError(
                    "Two-site gate must have shape (d**2, d**2) or (d, d, d, d)."
                )
        else:
            raise ValueError("Each gate location must have one or two sites.")

    @staticmethod
    def _prepare_gate_pair(gate, n_sites, bra_gate=None):
        """Return gate tensors for ket/bra index families.

        ``gate`` is applied on ket indices (transpose/index-order normalization).
        ``bra_gate`` is applied on bra indices as ``bra_gate†`` after the same
        normalization. Either side may be omitted with ``None``.
        """
        if gate is None and bra_gate is None:
            raise ValueError("At least one of ket gate or bra gate must be provided.")

        g_k = None if gate is None else MpoOptimizer._prepare_gate_tensor(gate, n_sites)
        if bra_gate is None:
            g_b = None
        else:
            g_b = ar.do("conj", MpoOptimizer._prepare_gate_tensor(bra_gate, n_sites))
        return g_k, g_b

    @staticmethod
    def _parse_gate_entry(G_i, where_i):
        """Normalize one gate-stream entry to ``(ket_gate, bra_gate, where)``.

        Bare gates default to two-sided MPO evolution by mapping ``G`` to
        ``(G, G)``, which is interpreted as ``G`` on ket and ``G†`` on bra.
        """
        where_norm = tuple(where_i)
        if len(where_norm) not in (1, 2):
            raise ValueError("Each gate location must have one or two sites.")

        if isinstance(G_i, (tuple, list)):
            if len(G_i) == 1:
                # Explicit ket-only shorthand: (G,) -> (G, None)
                gate, bra_gate = G_i[0], None
            elif len(G_i) == 2:
                gate, bra_gate = G_i
            else:
                raise ValueError("Each MPO gate entry must be G, (G,), or (G, B).")
        else:
            # Default MPO evolution applies U on ket and U† on bra.
            gate, bra_gate = G_i, G_i

        if gate is None and bra_gate is None:
            raise ValueError("Each gate entry must provide at least one of G or B.")
        return gate, bra_gate, where_norm

    def _apply_gate_pair(
        self,
        p,
        gate,
        where,
        bra_gate=None,
        *,
        cutoff,
        cutoff_mode="rel",
        contract,
        inplace=True,
    ):
        """Apply one gate to both ket and bra legs of ``p``."""
        n_sites = len(where)
        g_k, g_b = self._prepare_gate_pair(gate, n_sites, bra_gate=bra_gate)

        if g_k is not None:
            apply_gate(
                p,
                g_k,
                where,
                ind_id=self.ind_id_k,
                contract=contract,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=inplace,
            )
        if g_b is not None:
            apply_gate(
                p,
                g_b,
                where,
                ind_id=self.ind_id_b,
                contract=contract,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=inplace,
            )

    def _build_dmrg_target(self, p, gate, where, bra_gate, cutoff, cutoff_mode="rel"):
        """Build target MPO after applying a two-site gate pair."""
        p_g = p.copy()
        self._apply_gate_pair(
            p_g,
            gate,
            where,
            bra_gate=bra_gate,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            contract="split-gate",
            inplace=True,
        )
        return p_g

    @staticmethod
    def _collect_dmrg_batch(G_seq, where_seq, start_idx, k_2q_batch):
        """Collect a DMRG batch starting at a two-site gate index."""
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = tuple(where_seq[idx])
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

    def _build_dmrg_batch_target(self, p, batch_G, batch_where, cutoff, cutoff_mode="rel"):
        """Apply a collected DMRG batch onto a copy of ``p``."""
        p_g = p.copy()
        for G_i, where_i in zip(batch_G, batch_where):
            gate, bra_gate, where = self._parse_gate_entry(G_i, where_i)
            contract = True if len(where) == 1 else "split-gate"
            self._apply_gate_pair(
                p_g,
                gate,
                where,
                bra_gate=bra_gate,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                contract=contract,
                inplace=True,
            )
        return p_g

    def _run_dmrg(self, G_seq, where_seq, n_iter, progbar=False, cutoff=1e-12, cutoff_mode="rel", k_2q_batch=1, fidelity_samples=10):
        """Apply gates with local DMRG-style fitting updates."""
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")

        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="dmrg_mpo",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(G_seq):
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
                idx += 1
                advanced = 1
            elif n_sites == 2:
                if k_2q_batch == 1:
                    two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    self.canonize_mpo(p, (xmin, xmax))
                    p_g = self._build_dmrg_target(p, gate, where, bra_gate, cutoff, cutoff_mode)

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

                    self.info_c["cur_orthog"] = (xmin, xmax)
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
                    self.canonize_mpo(p, (xmin, xmax))
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

                    self.info_c["cur_orthog"] = (xmin, xmax)
                    advanced = next_idx - idx
                    idx = next_idx
            else:
                raise ValueError("Each gate location must have one or two sites.")

            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        self.p = p

    def _run_svd(self, G_seq, where_seq, progbar=False, cutoff=1e-12, cutoff_mode="rel", fidelity_samples=10):
        """Apply gates with local SVD compression for two-site updates."""
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
                colour="CYAN",
            )

        idx = 0
        while idx < len(G_seq):
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            elif n_sites == 2:
                two_qubit_count += 1
                xmin, xmax = sorted(where)

                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract="reduce-split",
                    inplace=True,
                )

                self.canonize_mpo(p, (xmin, xmax))
                for i in range(xmax, xmin, -1):
                    p.right_canonize_site(i, bra=None)
                p.left_compress(
                    start=xmin,
                    stop=xmax,
                    max_bond=self.chi,
                    cutoff=cutoff,
                )
            else:
                raise ValueError("Each gate location must have one or two sites.")

            idx += 1
            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p


    def _run_mpo(self, G_seq, where_seq, progbar=False, cutoff=1e-12, cutoff_mode="rel", fidelity_samples=10):
        """Apply gates using MPO-style nonlocal compression via gate_nonlocal_opt.

        For two-site gates, the raw gate/bra_gate tensors are applied independently
        to the upper (ket) and lower (bra) layers of the MPO using gate_nonlocal_opt.
        One-site gates are applied directly via _apply_gate_pair.
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
                colour="GREEN",
            )

        idx = 0
        while idx < len(G_seq):
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            elif n_sites == 2:
                two_qubit_count += 1
                g_k, g_b = self._prepare_gate_pair(gate, n_sites, bra_gate=bra_gate)
                if g_k is not None:
                    p = gate_nonlocal_opt(
                        p, g_k, where,
                        which="upper", method="direct",
                        info=self.info_c, inplace=True,
                        ind_id_k=self.ind_id_k, ind_id_b=self.ind_id_b,
                        max_bond=self.chi, cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                if g_b is not None:
                    p = gate_nonlocal_opt(
                        p, g_b, where,
                        which="lower", method="direct",
                        info=self.info_c, inplace=True,
                        ind_id_k=self.ind_id_k, ind_id_b=self.ind_id_b,
                        max_bond=self.chi, cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                self.p = p
            else:
                raise ValueError("Each gate location must have one or two sites.")

            idx += 1
            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(self.p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": norm_proxy,
                    "bnd": self.p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p


    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=6,
        *,
        mode=None,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rel",
        fidelity_samples=10,
        k_2q_batch=1,
    ):
        """Run queued gates on both MPO index families.

        Parameters
        ----------
        n_iter : int, default=6
            Inner iterations for DMRG ``FIT`` updates on two-site gates.
            Ignored by ``svd`` mode.
        mode : {"dmrg", "svd", "mpo"} | None, default=None
            Optional mode override for this run.
        progbar : bool, default=False
            Show tqdm progress bar.
        cutoff : float, default=1e-12
            Truncation cutoff used in gate application and compression.
        cutoff_mode : str, default="rel"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``.
        fidelity_samples : int, default=10
            ``svd`` mode only: number of intermediate norm-proxy samples.
            A final sample is always recorded at the end of the run.
        k_2q_batch : int, default=1
            ``dmrg`` mode only: number of sequential two-qubit gates to batch
            into one local FIT update. The FIT window uses the batch-wide
            ``[xmin, xmax]`` from all gate locations in the batch.

        Returns
        -------
        qtn.MatrixProductOperator
            Updated MPO after replaying the queued gate stream.
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

        supported = ", ".join(sorted(self._ALLOWED_MODES))
        raise ValueError(f"Unknown mode: {self.mode}. Supported modes: {supported}")

    def canonize_mpo(self, p, where):
        """Update canonical form around a one- or two-site gate span.

        ``where`` may be an int, a 1-tuple ``(site,)``, or a 2-tuple
        ``(xmin, xmax)``.  Integers and singletons collapse to a single-site
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
            xmin, xmax = min(int(where[0]), int(where[1])), max(int(where[0]), int(where[1]))
            where_canon = [xmin, xmax]
            target_orthog = (xmin, xmax)
        else:
            raise ValueError("where must be an int, (int,), or (int, int).")

        p.canonize(where_canon, cur_orthog=self._current_orthog(p))
        self.info_c["cur_orthog"] = target_orthog

    def get_fidelities(self):
        """Return the running loss history."""
        return self.losses