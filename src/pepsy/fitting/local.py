"""DMRG local fitting utilities for MPS/MPO tensor networks.

This module provides local and environment-based sweep routines used by
boundary contraction code. The focus is to keep tensor-index handling explicit
and fail early when input structure is inconsistent.
"""

import logging
import math
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..tensors.core import tn_fidelity

logger = logging.getLogger(__name__)

__all__ = [
    "FIT",
    "internal_inds",
]


def internal_inds(psi):
    """Return all internal (non-open) indices of ``psi``."""
    open_inds = psi.outer_inds()
    inner = []
    for tensor in psi:
        for ind in tensor.inds:
            if ind not in open_inds:
                inner.append(ind)
    return inner


class FIT:  # pylint: disable=too-many-instance-attributes
    """Local tensor fitting of an MPS/MPO against a target tensor network.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Target tensor network to fit.
    p : qtn.MatrixProductState | qtn.MatrixProductOperator
        Initial state to optimize.
    cutoffs : float, default=1e-12
        Numerical cutoff used by local decompositions/truncations.
    backend : str | None, default=None
        Backend specification for tensor operations.
    site_tag_id : str, default="I{}"
        Site-tag format used by ``p`` and local environment builders.
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer used for effective-environment contractions.
    range_int : sequence[int] | None, default=None
        Optional active interval ``(start, stop)`` used by :meth:`run_gate`.
    retag : bool, default=False
        If ``True``, regenerate tags on ``tn`` from ``p`` site connectivity.
    info : dict[str, Any] | None, default=None
        Optional scratch dictionary used by callers to store metadata.
    warning : bool, default=False
        Enable warning logs for fallback and retagging edge-cases.
    inplace : bool, default=False
        If ``True``, optimize ``p`` in place; otherwise operate on ``p.copy()``.
    """

    def __init__(
        self,
        tn: qtn.TensorNetwork,
        p: Optional[qtn.TensorNetwork] = None,
        cutoffs: float = 1e-12,
        backend: Optional[str] = None,
        site_tag_id: str = "I{}",
        contraction_opt: str = "auto-hq",
        range_int: Optional[Sequence[int]] = None,
        retag: bool = False,
        info: Optional[Dict[str, Any]] = None,
        warning: bool = False,
        inplace: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        if p is None:
            raise ValueError("Initial MPS `p` must be provided for FIT.")
        if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
            raise TypeError("Initial MPS `p` must be MatrixProductState or MatrixProductOperator.")
        if not isinstance(site_tag_id, str) or "{}" not in site_tag_id:
            raise ValueError("site_tag_id must be a format string containing '{}'.")

        self.L = int(p.L)

        self.p = p if inplace else p.copy()

        self.tn = tn.copy()

        if site_tag_id:
            site_ind_id = getattr(self.p, "site_ind_id", None)
            self.p.view_as_(
                qtn.MatrixProductState,
                L=self.L,
                site_tag_id=site_tag_id,
                site_ind_id=site_ind_id,
                cyclic=False,
            )

        self.site_tag_id = site_tag_id

        # Contraction path optimizer spec.
        self.contraction_opt = contraction_opt

        # cutoffs and underlying backend
        self.cutoffs = cutoffs
        self.backend = backend

        # warnings being printed or not
        self.warning = warning

        # Diagnostics collected during sweeps.
        self.fidelity_trace: List[float] = []
        self.local_norm_trace: List[float] = []
        self.sweep_norm_trace: List[float] = []
        self.iterations_run = 0
        self.converged = False
        self.last_relative_change: Optional[float] = None
        self.info: Dict[str, Any] = info or {}
        self.range_int: List[int] = list(range_int) if range_int is not None else []
        if self.range_int:
            if len(self.range_int) != 2:
                raise ValueError("range_int must be a sequence of two integers: (start, stop).")
            start, stop = self.range_int
            if start >= stop:
                raise ValueError("range_int must satisfy start < stop.")


        # Reindex tensor network with random UUIDs for internal indices
        self.tn.reindex_({idx: qtn.rand_uuid() for idx in self.tn.inner_inds()})

        if set(self.tn.outer_inds()) != set(self.p.outer_inds()):
            raise ValueError("tn and p have different outer indices.")

        # Re-tag TN for effective environments when requested.
        if retag:
            self._re_tag()

    def visual(
        self,
        figsize=(14, 14),
        layout="neato",
        show_tags=False,
        tags_: Optional[Sequence[str]] = None,
        show_inds=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Visualize the combined target network and current fitted state."""
        tag_list = tags_ if tags_ is not None else []
        tags = [self.site_tag_id.format(i) for i in range(self.L)] + tag_list
        return (self.tn & self.p).draw(
            tags,
            legend=False,
            show_inds=show_inds,
            show_tags=show_tags,
            figsize=figsize,
            node_outline_darkness=0.1,
            node_outline_size=None,
            highlight_inds_color="darkred",
            edge_scale=2.0,
            layout=layout,
            refine_layout="auto",
            highlight_inds=self.p.outer_inds(),
        )

    # -------------------------
    # Tagging methods
    # -------------------------
    def _deep_tag(self):
        """
        Propagates tags through the tensor network to ensure every tensor
        receives at least one site tag. Useful for layered TNs.
        """
        tn = self.tn
        count = 1

        while count >= 1:
            tags = tn.tags
            count = 0
            for tag in tags:
                tids = tn.tag_map[tag]
                neighbors = qtn.oset()
                for tid in tids:
                    t = tn.tensor_map[tid]
                    for ix in t.inds:
                        neighbors |= tn.ind_map[ix]
                for tid in neighbors:
                    t = tn.tensor_map[tid]
                    if not t.tags:
                        t.add_tag(tag)
                        count += 1

    def _re_tag(self):
        """Assign site tags on target TN tensors based on current boundary state."""
        # Drop all existing tags first.
        tn = self.tn
        tn.drop_tags()

        # Get outer indices and all site tags from current state.
        p = self.p
        site_tags = [self.site_tag_id.format(i) for i in range(p.L)]
        inds = list(p.outer_inds())

        # First-layer tagging: pick tensor directly connected to each boundary index.
        for site_tag in site_tags:
            site_outer = [idx for idx in p[site_tag].inds if idx in inds]
            if not site_outer:
                continue
            idx = site_outer[0]

            tids = list(tn.ind_map.get(idx, ()))
            if not tids:
                continue
            t = tn.tensor_map[tids[0]]

            if not t.tags:
                t.add_tag(site_tag)

        untagged_tensors = [tensor for tensor in tn if not tensor.tags]
        if untagged_tensors:
            if self.warning:
                logger.warning(
                    "%d tensors are still untagged after initial retagging; "
                    "propagating tags through neighbors.",
                    len(untagged_tensors),
                )
            self._deep_tag()

    def run(self, n_iter=6, verbose=False):
        """Run basic left-to-right local fitting sweeps.

        Parameters
        ----------
        n_iter : int
            Number of complete sweeps.
        verbose : bool
            If ``True``, append per-sweep fidelity values to ``self.fidelity_trace``.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        psi = self.p
        L = self.L
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        for _ in range(n_iter):
            for site in range(L):
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1

                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                psi_h = psi.H.select([site_tag_id.format(site)], "!any")
                tn_ = psi_h | self.tn

                # Contract and normalize
                f = tn_.contract(all, optimize=contraction_opt)
                f = f.transpose(*psi[site].inds)

                # norm_f is never applied (f.data used as-is); keep only for diagnostics if needed
                # norm_f = (f.H & f).contract(all) ** 0.5
                # self.local_norm_trace.append(complex(norm_f).real)

                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = tn_fidelity(self.tn, psi)
                self.fidelity_trace.append(ar.do("real", fidelity))

    def _build_env_right(self, psi, env_right):
        """Build inclusive right environments for all sites.

        Populates ``env_right[site_tag]`` for each site, where each entry is
        the contraction of the current site block and everything to its right.
        """
        L = self.L
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        # iterate from rightmost to leftmost
        for i in reversed(range(L)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block

            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)
            else:
                # tie to previously computed right environment
                t |= env_right[site_tag_id.format(i + 1)]
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)

    def _right_range(self, psi, env_right, start, stop):
        """Build right environments over a restricted ``[start, stop]`` window.

        This variant is used by :meth:`run_gate` and supports partially
        available right boundaries at interval edges.
        """
        L = self.L
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        indx = None
        indx_ = None
        # iterate from rightmost to leftmost
        for count, i in enumerate(range(stop, start, -1)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            # Is there any tensor in tn to be included in env
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block

            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)
            else:
                if count == 0:
                    indx = psi.bond(stop + 1, stop)
                    indx_ = self.tn.bond(stop + 1, stop)

                # tie to previously computed right environment
                if env_right[site_tag_id.format(i + 1)] is not None:
                    t |= env_right[site_tag_id.format(i + 1)]
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)
                else:
                    if indx is None or indx_ is None:
                        raise ValueError("Right-range boundary indices are not initialized.")
                    t = t.reindex({indx: indx_})
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)

    def _left_range(self, psi, site, count, env_left):
        """Update left environment incrementally for current site."""

        # get tensor at site from p
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block

        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)
        else:
            if count == 1:
                indx = psi.bond(site - 1, site)
                indx_ = self.tn.bond(site - 1, site)
                t = t.copy()
                t = t.reindex({indx: indx_})
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)
            else:
                t |= env_left[site_tag_id.format(site - 1)]
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)

    def _update_env_left(self, psi, site: int, env_left):
        """Update left environment incrementally for current site."""

        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block

        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)
        else:
            t |= env_left[site_tag_id.format(site - 1)]
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)

    def run_eff(
        self,
        n_iter=6,
        verbose=False,
    ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Run environment-based fitting sweeps with cached left/right blocks.

        This method avoids rebuilding full contractions at each site by
        incrementally reusing left and right environments.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        contraction_opt = self.contraction_opt

        if L == 1:
            if self.warning:
                logger.warning("run_eff called for L=1; falling back to run().")
            self.run(n_iter=n_iter, verbose=verbose)
            return

        env_left = {site_tag_id.format(i): None for i in range(psi.L)}
        env_right = {site_tag_id.format(i): None for i in range(psi.L)}

        for _ in range(n_iter):
            for site in range(L):
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1
                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                if site == 0:
                    self._build_env_right(psi, env_right)
                else:
                    self._update_env_left(psi, site - 1, env_left)

                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                tn = None
                if site == 0:
                    if tn_site is not None:
                        tn = tn_site | env_right[site_tag_id.format(site + 1)]
                    else:
                        tn = env_right[site_tag_id.format(site + 1)]

                if 0 < site < L - 1:
                    if tn_site is not None:
                        tn = (
                            tn_site
                            | env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )
                    else:
                        tn = (
                            env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )

                if site == L - 1:
                    if tn_site is not None:
                        tn = tn_site | env_left[site_tag_id.format(site - 1)]
                    else:
                        tn = env_left[site_tag_id.format(site - 1)]

                if tn is None:
                    raise ValueError("Failed to build effective tensor for current site.")

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=contraction_opt).transpose(
                        *psi[site_tag_id.format(site)].inds
                    )
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                else:
                    raise TypeError("Unexpected effective tensor type during run_eff.")

                # norm_f is never applied (f.data used as-is); keep only for diagnostics if needed
                # norm_f = (f.H & f).contract(all) ** 0.5
                # self.local_norm_trace.append(complex(norm_f).real)

                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = tn_fidelity(self.tn, psi)
                self.fidelity_trace.append(ar.do("real", fidelity))

    def run_gate(
        self,
        n_iter=6,
        verbose=False,
        *,
        min_iter=None,
        rtol=None,
        patience=1,
        finite_check=None,
    ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Run fitting restricted to ``range_int`` with gate-style sweeps.

        The algorithm canonicalizes the active interval and updates tensors
        using effective environments built from neighboring boundaries. By
        default, exactly ``n_iter`` sweeps are performed. Supplying ``rtol``
        enables early stopping after ``min_iter`` sweeps once the final local
        norm changes by at most ``rtol`` for ``patience`` consecutive sweeps.
        ``finite_check=True`` checks only the already-computed per-site norm
        scalars, transferring one tiny vector per sweep. A callable retains
        the general state-check callback behavior.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be a positive integer.")
        n_iter = int(n_iter)
        if min_iter is None:
            min_iter = n_iter if rtol is None else 1
        if not isinstance(min_iter, Integral) or int(min_iter) < 1:
            raise ValueError("min_iter must be a positive integer or None.")
        min_iter = min(int(min_iter), n_iter)
        if rtol is not None:
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError("rtol must be a finite non-negative number or None.")
        if not isinstance(patience, Integral) or int(patience) < 1:
            raise ValueError("patience must be a positive integer.")
        patience = int(patience)
        if finite_check not in (None, False, True) and not callable(finite_check):
            raise TypeError("finite_check must be bool, callable, or None.")

        self.sweep_norm_trace = []
        self.iterations_run = 0
        self.converged = False
        self.last_relative_change = None

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        contraction_opt = self.contraction_opt

        if L == 1:
            if self.warning:
                logger.warning("run_gate called for L=1; falling back to run().")
            self.run(n_iter=n_iter, verbose=verbose)
            return

        if len(self.range_int) != 2:
            raise ValueError("range_int must be set to (start, stop) before calling run_gate.")
        start, stop = self.range_int
        if start < 0 or stop >= L or start > stop:
            raise ValueError(f"range_int={self.range_int} is out of bounds for L={L}.")
        if stop == start:
            raise ValueError("run_gate requires range_int spanning at least two sites.")

        env_left = {site_tag_id.format(i): None for i in range(psi.L)}
        env_right = {site_tag_id.format(i): None for i in range(psi.L)}

        previous_sweep_norm = None
        stable_sweeps = 0
        for sweep in range(1, n_iter + 1):
            self.iterations_run = sweep
            sweep_norm_start = len(self.local_norm_trace)
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)

            for count_, site in enumerate(range(start, stop + 1)):
                if count_ == 0:
                    self._right_range(psi, env_right, start, stop)
                else:
                    self._left_range(psi, site - 1, count_, env_left)

                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                tn = None
                if site == 0:
                    if tn_site is not None:
                        tn = tn_site | env_right[site_tag_id.format(site + 1)]
                    else:
                        tn = env_right[site_tag_id.format(site + 1)]

                if 0 < site < L - 1:
                    # Boundary consistency: the left and right indices must match between tn and p
                    if count_ == 0:
                        indx = psi.bond(start - 1, start)
                        indx_ = self.tn.bond(start - 1, start)
                        if tn_site is not None:
                            tn_site = tn_site.reindex({indx_: indx})
                    if count_ == stop - start:
                        indx = psi.bond(stop + 1, stop)
                        indx_ = self.tn.bond(stop + 1, stop)
                        if tn_site is not None:
                            tn_site = tn_site.reindex({indx_: indx})

                    if tn_site is not None:
                        if (
                            env_right[site_tag_id.format(site + 1)] is not None
                            and env_left[site_tag_id.format(site - 1)] is not None
                        ):
                            tn = (
                                tn_site
                                | env_right[site_tag_id.format(site + 1)]
                                | env_left[site_tag_id.format(site - 1)]
                            )
                        elif env_left[site_tag_id.format(site - 1)] is not None:
                            tn = tn_site | env_left[site_tag_id.format(site - 1)]
                        elif env_right[site_tag_id.format(site + 1)] is not None:
                            tn = tn_site | env_right[site_tag_id.format(site + 1)]
                        else:
                            tn = tn_site
                    else:
                        tn = (
                            env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )

                if site == L - 1:
                    if tn_site is not None:
                        tn = tn_site | env_left[site_tag_id.format(site - 1)]
                    else:
                        tn = env_left[site_tag_id.format(site - 1)]

                if tn is None:
                    raise ValueError("Failed to build effective tensor for gate sweep.")

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=contraction_opt).transpose(
                        *psi[site_tag_id.format(site)].inds
                    )
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                else:
                    raise TypeError("Unexpected effective tensor type during run_gate.")

                norm_f = (f.H & f).contract(all) ** 0.5
                self.local_norm_trace.append(ar.do("real", norm_f))

                # Update tensor data
                psi[site].modify(data=f.data)

                if site < stop:
                    psi.left_canonize_site(site, bra=None)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = tn_fidelity(self.tn, psi)
                self.fidelity_trace.append(ar.do("real", fidelity))

            if callable(finite_check) and not bool(finite_check(psi)):
                error = FloatingPointError(
                    f"FIT gate sweep {sweep} produced non-finite tensor data."
                )
                error.fit_iteration = sweep
                raise error

            if finite_check is True or rtol is not None:
                sweep_scalars = self.local_norm_trace[sweep_norm_start:]
                try:
                    sweep_norms = np.asarray(
                        ar.to_numpy(ar.do("stack", sweep_scalars))
                    ).reshape(-1)
                except Exception:
                    # Compatibility fallback for a backend without scalar
                    # stack support. NumPy, Torch, and CuPy use the fast path.
                    sweep_norms = np.asarray(
                        [float(ar.to_numpy(value)) for value in sweep_scalars]
                    )
                if finite_check is True and not bool(np.all(np.isfinite(sweep_norms))):
                    error = FloatingPointError(
                        f"FIT gate sweep {sweep} produced a non-finite local norm."
                    )
                    error.fit_iteration = sweep
                    raise error

            if rtol is not None:
                sweep_norm = float(sweep_norms[-1])
                self.sweep_norm_trace.append(sweep_norm)
                if not math.isfinite(sweep_norm):
                    error = FloatingPointError(
                        f"FIT gate sweep {sweep} produced a non-finite local norm."
                    )
                    error.fit_iteration = sweep
                    raise error
                if previous_sweep_norm is not None:
                    scale = max(
                        abs(sweep_norm),
                        abs(previous_sweep_norm),
                        float.fromhex("0x1.0p-1022"),
                    )
                    relative_change = abs(
                        sweep_norm - previous_sweep_norm
                    ) / scale
                    self.last_relative_change = relative_change
                    if relative_change <= rtol:
                        stable_sweeps += 1
                    else:
                        stable_sweeps = 0
                    if sweep >= min_iter and stable_sweeps >= patience:
                        self.converged = True
                        break
                previous_sweep_norm = sweep_norm
