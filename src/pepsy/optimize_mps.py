"""MPS optimization helpers centered on :class:`MpsOptimizer`."""

from __future__ import annotations


from .dmrg_fit import FIT
from .gate import gate_1d

__all__ = ["MpsOptimizer"]


class MpsOptimizer:
    """High-level wrapper for MPS gate-sweep objectives.

    Parameters
    ----------
    p : qtn.MatrixProductState
        Initial MPS state.
    gates : sequence[tuple[sequence[int], object]]
        Gate stream as ``(where, G)`` tuples.
    chi : int
        Maximum bond dimension used by SVD mode.
    mode : {"dmrg", "svd"}, default="dmrg"
        Optimization backend.
    engine : str | None, default=None
        Backward-compatible alias for ``mode``.
    """

    _ALLOWED_MODES = frozenset({"dmrg", "svd"})

    @classmethod
    def _normalize_mode(cls, mode):
        mode_norm = str(mode).strip().lower()
        if mode_norm not in cls._ALLOWED_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return mode_norm

    def __init__(self, p, gates, chi, mode="dmrg", engine=None):
        self.p = p
        self.gates = gates
        self.chi = chi
        self.mode = self._normalize_mode(engine if engine is not None else mode)

        self.info_c = {}
        self.Fidel_l = [1.0]
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

    @staticmethod
    def _safe_max_bond(p):
        """Return ``p.max_bond()`` when available, else ``None``."""
        try:
            return p.max_bond()
        except Exception:  # pragma: no cover - defensive fallback
            return None

    def _init_canonicalization(self):
        """Initialize canonical form and orthogonality center."""
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    def set_p(self, p):
        """Assign a new state and reset canonicalization metadata."""
        self.p = p
        self._init_canonicalization()

    def set_mode(self, mode):
        """Switch optimization mode while keeping ``p`` and ``info_c``."""
        self.mode = self._normalize_mode(mode)
        return self

    def run(self, n_iter=6, progbar=False, cutoff=1e-12, mode=None):
        """Run a full pass over all gates.

        Parameters
        ----------
        n_iter : int, default=6
            Inner iterations for DMRG local fits.
        progbar : bool, default=False
            Accepted for API compatibility.
        cutoff : float, default=1e-12
            Truncation cutoff used in gate application and local fitting.
        mode : {"dmrg", "svd"} | None, default=None
            Optional mode override for this run. If supplied, updates
            ``self.mode`` before execution.
        """
        if mode is not None:
            self.set_mode(mode)

        if self.mode == "dmrg":
            self._run_dmrg(n_iter=n_iter, progbar=progbar, cutoff=cutoff)
            return

        if self.mode == "svd":
            self._run_svd(progbar=progbar, cutoff=cutoff)
            return

        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_dmrg(self, n_iter, progbar=False, cutoff=1e-12):
        """Apply gates with local DMRG-style fitting."""
        p = self.p
        two_qubit_count = 0

        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            iterator = tqdm(
                self.gates,
                total=len(self.gates),
                desc="dmrg",
                leave=True,
                position=0,
                colour="CYAN",
            )
        else:
            iterator = self.gates

        for where, gate in iterator:
            if len(where) == 1:
                gate_1d(p, where, gate, contract=True, cutoff=cutoff, inplace=True)
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")

                two_qubit_count += 1
                self.canonize_mps(p, where)

                p_g = gate_1d(
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
                    opt=None,
                    re_tag=False,
                    range_int=self._current_orthog(p),
                )
                fit.run_gate(n_iter=n_iter, verbose=False)

                p = fit.p
                self.Fidel_l.append(complex(fit.loss_[-1] ** 2).real)

            if progbar:
                postfix = {
                    "two_q": two_qubit_count,
                    "F": self.Fidel_l[-1],
                    "bnd": self._safe_max_bond(p),
                }
                iterator.set_postfix(postfix)

        self.p = p

    def _run_svd(self, progbar=False, cutoff=1e-12):
        """Apply gates with direct ``gate_nonlocal_`` compression."""
        p = self.p
        two_qubit_count = 0

        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            iterator = tqdm(
                self.gates,
                total=len(self.gates),
                desc="svd",
                leave=True,
                position=0,
                colour="CYAN",
            )
        else:
            iterator = self.gates

        for where, gate in iterator:
            if len(where) == 1:
                gate_1d(p, where, gate, contract=True, cutoff=cutoff, inplace=True)
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")

                two_qubit_count += 1
                p_g = gate_1d(
                    p,
                    where,
                    gate,
                    contract="split-gate",
                    cutoff=cutoff,
                    inplace=False,
                )
                p.gate_nonlocal_(
                    gate,
                    where,
                    max_bond=self.chi,
                    info=self.info_c,
                    method="direct",
                    cutoff=cutoff,
                )
                fidelity = p.norm()
                self.Fidel_l.append(complex(fidelity).real)

            if progbar:
                postfix = {
                    "two_q": two_qubit_count,
                    "F": self.Fidel_l[-1],
                    "bnd": self._safe_max_bond(p),
                }
                iterator.set_postfix(postfix)

        self.p = p

    def canonize_mps(self, p, where):
        """Update canonical form around a two-site gate span."""
        xmin, xmax = sorted(where)
        p.canonize([xmin, xmax], cur_orthog=self._current_orthog(p))
        # Preserve the original 2-site fitting window semantics.
        self.info_c["cur_orthog"] = (xmin, xmax)

    def get_fidelities(self):
        """Return the running fidelity history."""
        return self.Fidel_l

    def get_losses(self):
        """Compatibility shim retained for old callers."""
        return []
