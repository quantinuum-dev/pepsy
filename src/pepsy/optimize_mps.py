"""MPS optimization helpers centered on :class:`MpsOptimizer`."""

from __future__ import annotations

from numbers import Integral
import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .fit import FIT
from .gate import apply_gate_1d

__all__ = ["MpsOptimizer"]


class MpsOptimizer:  # pylint: disable=too-many-instance-attributes
    """High-level wrapper for MPS gate-sweep objectives.

    Parameters
    ----------
    p : qtn.MatrixProductState
        Initial MPS state.
    gates : sequence[tuple[sequence[int], object]] | None, optional
        Gate stream as ``(where, G)`` tuples. If omitted, start with an empty
        queue and use :meth:`set_gates` or :meth:`add_gates` before ``run``.
    chi : int
        Maximum bond dimension used by MPO/swap/SVD compression modes.
    mode : {"dmrg", "mpo", "swap", "svd", "exact"}, default="dmrg"
        Optimization backend.
    contraction_opt : object | None, optional
        Canonical contraction path optimizer keyword.
    ind_id : str, default="k{}"
        Format string for site index labels used by exact gate application.
        Use "k{},{}" when gate sites are 2D coordinates like ``(i, j)``.
    """

    _ALLOWED_MODES = frozenset({"dmrg", "mpo", "swap", "svd", "exact"})

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
        contraction_opt=None,
        ind_id="k{}",
    ):
        if chi is None:
            # Allow shorthand: MpsOptimizer(p, chi, mode=...)
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpsOptimizer(p, gates, chi) "
                    "or MpsOptimizer(p, chi) for an empty gate queue."
                )

        if gates is None:
            gates = []

        self.p = p
        self.gates = list(gates)
        self.chi = chi
        self.mode = self._normalize_mode(mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.ind_id = str(ind_id)

        self.info_c = {}
        self.fidelity_trace = [1.0]
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
        self.p = p
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
        """Replace the current gate list with ``gates``.

        After calling this, ``run(...)`` applies only this new list
        (unless you call :meth:`add_gates` before running).
        """
        self.gates = list(gates)
        return self

    def add_gates(self, gates):
        """Append ``gates`` to the existing gate list.

        This preserves previously queued gates and extends them with
        new ones.
        """
        self.gates.extend(list(gates))
        return self

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=6,
        progbar=False,
        cutoff=1e-12,
        mode=None,
        fidelity_samples=10,
        k_2q_batch=1,
    ):
        """Run the currently queued gates.

        Parameters
        ----------
        n_iter : int, default=6
            Inner iterations for DMRG local fits. Ignored by
            ``mpo``/``swap``/``svd``/``exact``.
        progbar : bool, default=False
            Show per-mode progress bars.
        cutoff : float, default=1e-12
            Truncation cutoff used in gate application and local fitting.
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
        """
        if mode is not None:
            self.set_mode(mode)

        gates = list(self.gates)
        if not gates:
            return

        if self.mode == "dmrg":
            self._prepare_dmrg_state()
            self._run_dmrg(
                gates,
                n_iter=n_iter,
                progbar=progbar,
                cutoff=cutoff,
                k_2q_batch=k_2q_batch,
            )
            return

        if self.mode == "mpo":
            self._run_mpo(
                gates,
                progbar=progbar,
                cutoff=cutoff,
                fidelity_samples=fidelity_samples,
            )
            return

        if self.mode == "swap":
            self._run_swap(
                gates,
                progbar=progbar,
                cutoff=cutoff,
                fidelity_samples=fidelity_samples,
            )
            return

        if self.mode == "svd":
            self._run_svd(
                gates,
                progbar=progbar,
                cutoff=cutoff,
                fidelity_samples=fidelity_samples,
            )
            return

        if self.mode == "exact":
            self._run_exact(
                gates,
                progbar=progbar,
                cutoff=cutoff,
                fidelity_samples=fidelity_samples,
            )
            return

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
        self.fidelity_trace.append(norm_val)
        return norm_val

    @staticmethod
    def _collect_dmrg_batch(gates, start_idx, k_2q_batch):
        """Collect a DMRG batch starting at a two-qubit gate index."""
        batch_ops = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(gates) and two_qubit_in_batch < k_2q_batch:
            where, gate = gates[idx]
            if len(where) == 1:
                batch_ops.append((where, gate))
            elif len(where) == 2:
                batch_ops.append((where, gate))
                two_qubit_in_batch += 1
            else:
                raise ValueError("Each gate location must have one or two sites.")
            idx += 1

        return batch_ops, two_qubit_in_batch, idx

    @staticmethod
    def _build_dmrg_batch_target(p, batch_ops, cutoff):
        """Apply a collected DMRG batch onto a copy of ``p``."""
        p_g = p.copy()
        for where, gate in batch_ops:
            if len(where) == 1:
                apply_gate_1d(p_g, where, gate, contract=True, cutoff=cutoff, inplace=True)
            else:
                p_g = apply_gate_1d(
                    p_g,
                    where,
                    gate,
                    contract="split-gate",
                    cutoff=cutoff,
                    inplace=False,
                )
        return p_g

    def _run_dmrg(
        self,
        gates,
        n_iter,
        progbar=False,
        cutoff=1e-12,
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
                total=len(gates),
                desc="dmrg",
                leave=True,
                position=0,
                colour="CYAN",
            )
        else:
            pbar = None

        idx = 0
        while idx < len(gates):
            where, gate = gates[idx]
            if len(where) == 1:
                apply_gate_1d(p, where, gate, contract=True, cutoff=cutoff, inplace=True)
                idx += 1
                advanced = 1
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")

                if k_2q_batch == 1:
                    two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    self.canonize_mps(p, (xmin, xmax))
                    p_g = apply_gate_1d(
                        p,
                        where,
                        gate,
                        contract="split-gate",
                        cutoff=cutoff,
                        inplace=False,
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
                    self.fidelity_trace.append(complex(fit.local_norm_trace[-1]).real)
                    idx += 1
                    advanced = 1
                else:
                    batch_ops, two_qubit_in_batch, next_idx = self._collect_dmrg_batch(
                        gates, idx, k_2q_batch
                    )
                    if two_qubit_in_batch < 1:
                        raise RuntimeError("DMRG batch unexpectedly contains no two-qubit gates.")

                    two_qubit_count += two_qubit_in_batch
                    batch_span_sites = [site for batch_where, _ in batch_ops for site in batch_where]
                    xmin, xmax = min(batch_span_sites), max(batch_span_sites)
                    self.canonize_mps(p, (xmin, xmax))
                    p_g = self._build_dmrg_batch_target(p, batch_ops, cutoff)
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
                    self.fidelity_trace.append(complex(fit.local_norm_trace[-1]).real)
                    advanced = next_idx - idx
                    idx = next_idx

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self.fidelity_trace[-1],
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        self.p = p

    def _run_mpo(  # pylint: disable=too-many-locals
        self,
        gates,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=10,
    ):
        """Apply gates with MPO-style nonlocal compression.

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(gates), fidelity_samples)
        norm_proxy = self.fidelity_trace[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(gates),
                desc="mpo",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(gates):
            where, gate = gates[idx]
            if len(where) == 1:
                apply_gate_1d(p, where, gate, contract=True, cutoff=cutoff, inplace=True)
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
                )
                idx += 1
                advanced = 1

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

    def _run_swap(  # pylint: disable=too-many-locals
        self,
        gates,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=10,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(gates), fidelity_samples)
        norm_proxy = self.fidelity_trace[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(gates),
                desc="swap",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(gates):
            where, gate = gates[idx]
            if len(where) == 1:
                apply_gate_1d(p, where, gate, contract=True, cutoff=cutoff, inplace=True)
                idx += 1
                advanced = 1
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff}
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
                    "~F": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        self.p = p

    def _run_svd(  # pylint: disable=too-many-locals
        self,
        gates,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=10,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(gates), fidelity_samples)
        norm_proxy = self.fidelity_trace[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(gates),
                desc="svd",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(gates):
            where, gate = gates[idx]
            if len(where) == 1:
                apply_gate_1d(p, where, gate, contract=True, cutoff=cutoff, inplace=True)
                idx += 1
                advanced = 1
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff}
                apply_gate_1d(
                    p,
                    where,
                    gate,
                    contract="reduce-split",
                    cutoff=cutoff,
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
                    "~F": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        self.p = p

    def _run_exact(  # pylint: disable=too-many-locals
        self,
        gates,
        progbar=False,
        cutoff=1e-12,
        fidelity_samples=10,
    ):
        """Apply gates exactly using in-place ``contract=True`` application.

        Progress bar counts only 2-qubit gates.
        """
        self.p = self.p.contract(all, optimize="auto-hq")
        self.p = qtn.TensorNetwork([self.p])
        p = self.p
        two_qubit_count = 0
        # Keep parameter for API compatibility; exact mode does not sample fidelity.
        _ = fidelity_samples
        total_two_qubit = sum(1 for where, _ in gates if len(where) == 2)

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=total_two_qubit,
                desc="exact",
                leave=True,
                position=0,
                colour="CYAN",
            )

        for where, gate in gates:
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
            )

            if len(where) == 1:
                continue

            two_qubit_count += 1
            if pbar is not None:
                pbar.set_postfix({"2q": two_qubit_count})
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
        """Return the running fidelity history."""
        return self.fidelity_trace
