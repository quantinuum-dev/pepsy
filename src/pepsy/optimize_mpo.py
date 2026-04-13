"""MPO optimization helpers centered on :class:`MpoOptimizer`."""

from __future__ import annotations

from numbers import Integral

import autoray as ar

from .fit import FIT
from .gate import apply_gate_1d

__all__ = ["MpoOptimizer"]


class MpoOptimizer:
    """High-level wrapper for MPO gate sweeps.

    Parameters
    ----------
    mpo : qtn.MatrixProductOperator
        Initial MPO.
    gates : sequence[tuple[sequence[int], object]] | None, optional
        Gate stream as ``(where, G)`` tuples.
    chi : int
        Working bond dimension for DMRG-style local fitting.
    mode : {"dmrg"}, default="dmrg"
        Execution mode. Only ``dmrg`` is currently supported.
    contraction_opt : object | None, optional
        Contraction path optimizer keyword used inside :class:`FIT`.
    """

    _ALLOWED_MODES = frozenset({"dmrg"})

    @classmethod
    def _normalize_mode(cls, mode):
        mode_norm = str(mode).strip().lower()
        if mode_norm not in cls._ALLOWED_MODES:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: dmrg")
        return mode_norm

    def __init__(
        self,
        mpo,
        gates=None,
        chi=None,
        mode="dmrg",
        ind_id_k="k{}",
        ind_id_b="b{}",
        contraction_opt=None,
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

        if gates is None:
            gates = []

        self.p = mpo
        self.mpo = self.p
        self.gates = list(gates)
        self.chi = int(chi)
        self.mode = self._normalize_mode(mode)
        self.ind_id_k = str(ind_id_k)
        self.ind_id_b = str(ind_id_b)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt

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
        self.p = mpo
        self.mpo = self.p
        self._init_canonicalization()
        return self

    def set_mode(self, mode):
        """Set execution mode (currently only ``dmrg``)."""
        self.mode = self._normalize_mode(mode)
        return self

    def set_gates(self, gates):
        """Replace the current gate list with ``gates``."""
        self.gates = list(gates)
        return self

    def add_gates(self, gates):
        """Append ``gates`` to the current gate list."""
        self.gates.extend(list(gates))
        return self

    @staticmethod
    def _prepare_gate_pair(gate, n_sites):
        """Return gate tensors for ket/bra index families.

        Quimb two-site gates are commonly supplied as rank-2 matrices
        ``(d**2, d**2)``. In that case, a plain matrix transpose is
        equivalent to the old reshape+rank-4 transpose path and avoids
        extra reshaping overhead.
        """
        if n_sites == 1:
            g_k = ar.do("transpose", gate, (1, 0))
        elif n_sites == 2:
            shape = getattr(gate, "shape", ())
            if len(shape) == 2:
                din, dout = shape
                if int(din) != int(dout):
                    raise ValueError(
                        "Two-site gate matrix must be square with shape (d**2, d**2)."
                    )
                g_k = ar.do("transpose", gate, (1, 0))
            elif len(shape) == 4:
                g_k = ar.do("transpose", gate, (2, 3, 0, 1))
            else:
                raise ValueError(
                    "Two-site gate must have shape (d**2, d**2) or (d, d, d, d)."
                )
        else:
            raise ValueError("Each gate location must have one or two sites.")

        g_b = ar.do("conj", g_k)
        return g_k, g_b

    def _apply_gate_pair(
        self,
        p,
        where,
        gate,
        *,
        cutoff,
        contract,
    ):
        """Apply one gate to both ket and bra legs of ``p``."""
        n_sites = len(where)
        g_k, g_b = self._prepare_gate_pair(gate, n_sites)

        apply_gate_1d(
            p,
            where,
            g_k,
            ind_id=self.ind_id_k,
            contract=contract,
            cutoff=cutoff,
            inplace=True,
        )
        apply_gate_1d(
            p,
            where,
            g_b,
            ind_id=self.ind_id_b,
            contract=contract,
            cutoff=cutoff,
            inplace=True,
        )

    def _build_dmrg_target(self, p, where, gate, cutoff):
        """Build target MPO after applying a two-site gate pair."""
        p_g = p.copy()
        self._apply_gate_pair(
            p_g,
            where,
            gate,
            cutoff=cutoff,
            contract="split-gate",
        )
        return p_g

    def _run_dmrg(self, gates, n_iter, progbar=False, cutoff=1e-12):
        """Apply gates with local DMRG-style fitting updates."""
        p = self.p
        two_qubit_count = 0

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(gates),
                desc="dmrg_mpo",
                leave=True,
                position=0,
                colour="CYAN",
            )

        for where, gate in gates:
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    where,
                    gate,
                    cutoff=cutoff,
                    contract=True,
                )
            elif n_sites == 2:
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                self.canonize_mpo(p, (xmin, xmax))
                p_g = self._build_dmrg_target(p, where, gate, cutoff)

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

                if fit.local_norm_trace:
                    self.fidelity_trace.append(complex(fit.local_norm_trace[-1]).real)
                else:
                    self.fidelity_trace.append(self.fidelity_trace[-1])

                # Preserve 2-site fitting window metadata semantics.
                self.info_c["cur_orthog"] = (xmin, xmax)
            else:
                raise ValueError("Each gate location must have one or two sites.")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self.fidelity_trace[-1],
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p
        self.mpo = self.p

    def run(
        self,
        n_iter=6,
        *,
        mode=None,
        progbar=False,
        cutoff=1e-12,
    ):
        """Run queued gates on both MPO index families.

        Parameters
        ----------
        n_iter : int, default=6
            Inner iterations for local DMRG-style FIT updates on two-site gates.
        mode : {"dmrg"} | None, default=None
            Optional mode override for this run.
        progbar : bool, default=False
            Show tqdm progress bar.
        cutoff : float, default=1e-12
            Truncation cutoff used in gate application and local fitting.
        """
        if mode is not None:
            self.set_mode(mode)
        if self.mode != "dmrg":
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes: dmrg")

        gates = list(self.gates)
        if not gates:
            return self.mpo

        self._prepare_dmrg_state()
        self._run_dmrg(gates, n_iter=n_iter, progbar=progbar, cutoff=cutoff)
        return self.mpo

    def canonize_mpo(self, p, where):
        """Update canonical form around a two-site gate span."""
        xmin, xmax = sorted(where)
        p.canonize([xmin, xmax], cur_orthog=self._current_orthog(p))
        self.info_c["cur_orthog"] = (xmin, xmax)

    def get_fidelities(self):
        """Return the running local-fidelity proxy history."""
        return self.fidelity_trace
