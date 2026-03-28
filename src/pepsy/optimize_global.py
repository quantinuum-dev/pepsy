"""Global PEPS objective helpers centered on :class:`GlobalOptimizer`."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import autoray as ar
import quimb.tensor as qtn

__all__ = ["GlobalOptimizer"]


class GlobalOptimizer:
    """High-level wrapper for global PEPS optimization objectives.

    This class keeps a trainable PEPS and target PEPS together with reusable
    defaults. Public entrypoints are :meth:`norm`, :meth:`normalize`,
    :meth:`loss`, :meth:`make_tn_optimizer`, and :meth:`optimize`.

    Internal objective kernels are implemented as:

    - :meth:`_norm_peps`
    - :meth:`_normalize_peps`
    - :meth:`_loss_peps`
    """

    _NORM_KEYS = frozenset({
        "opt",
        "copt",
        "chi",
        "layer_tags",
        "equalize_norms",
        "mode",
        "mode_",
        "max_separation",
        "sequence",
        "progbar",
        "cutoff",
    })
    _LOSS_KEYS = frozenset({
        "opt",
        "copt",
        "cost_f",
        "val_",
        "progbar",
        "chi",
        "mode",
        "cutoff",
        "mode_",
        "max_separation",
        "equalize_norms",
        "sequence",
    })

    @staticmethod
    def _merge_opts(base, extra):
        merged = dict(base or {})
        if extra:
            merged.update(dict(extra))
        return merged

    @classmethod
    def _pick_known_keys(cls, options, allowed_keys, *, warn_unknown=True):
        incoming = dict(options or {})
        filtered = {key: value for key, value in incoming.items() if key in allowed_keys}
        unknown = sorted(set(incoming) - set(allowed_keys))
        if warn_unknown and unknown:
            warnings.warn(
                f"Ignoring unknown options: {', '.join(unknown)}",
                UserWarning,
                stacklevel=3,
            )
        return filtered

    @staticmethod
    def _apply_hyperoptimized_compressed(
        tn,
        copt,
        max_bond,
        output_inds=None,
        tree_gauge_distance=4,
        progbar=False,
        cutoff=1.0e-12,
        equalize_norms=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        tn.full_simplify_(seq="R", split_method="svd", inplace=True)

        tree = tn.contraction_tree(copt)
        tn_ = tn.copy()

        tn_.contract_compressed_(
            optimize=tree,
            output_inds=output_inds,
            max_bond=max_bond,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
        )
        return tn_

    @staticmethod
    def _norm_peps(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        peps,
        opt=None,
        copt=None,
        chi=20,
        layer_tags=None,
        equalize_norms=False,
        mode="exact",
        mode_="mps",
        max_separation=1,
        sequence=None,
        progbar=False,
        cutoff=1e-12,
    ):
        """Compute ``<peps|peps>`` using exact, boundary, rg, or hyper modes."""
        if layer_tags is None:
            layer_tags = ["KET", "BRA"]
        if sequence is None:
            sequence = ["xmin", "xmax", "ymin", "ymax"]
        if (mode == "hyper") and (copt is None):
            warnings.warn(
                "mode='hyper' requested but copt is None; provide copt_() for stable behavior.",
                RuntimeWarning,
                stacklevel=2,
            )

        peps.add_tag("KET")
        peps_h = peps.conj().retag({"KET": "BRA"})
        norm = peps_h | peps

        if mode == "mps":
            return norm.contract_boundary(
                max_bond=chi,
                mode=mode_,
                sequence=sequence,
                final_contract_opts={"optimize": opt},
                cutoff=cutoff,
                progbar=progbar,
                layer_tags=layer_tags,
                max_separation=max_separation,
                equalize_norms=equalize_norms,
            )

        if mode == "rg":
            return norm.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                canonize=True,
                mode="projector",
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                layer_tags=["KET", "BRA"],
                final_contract=True,
                final_contract_opts={"optimize": opt},
                progbar=progbar,
                inplace=False,
            )

        if mode == "exact":
            norm = norm.full_simplify(seq="R", output_inds={}, split_method="svd", inplace=False)
            return norm.contract(all, optimize=opt)

        if mode == "hyper":
            overlap = GlobalOptimizer._apply_hyperoptimized_compressed(
                norm,
                copt,
                chi,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                progbar=progbar,
            )
            return overlap ^ all

        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _normalize_peps(peps, **peps_norm_opts):
        """Normalize a PEPS in place via :meth:`_norm_peps`."""
        peps /= GlobalOptimizer._norm_peps(peps, **peps_norm_opts) ** 0.5
        return peps

    @staticmethod
    def _loss_peps(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        peps,
        peps_fix,
        opt=None,
        copt=None,
        cost_f="fid",
        val_=1.0,
        progbar=False,
        chi=60,
        mode="mps",
        cutoff=0.0,
        mode_="mps",
        max_separation=1,
        equalize_norms=False,
        sequence=("xmin", "xmax", "ymin", "ymax"),
    ):
        """Compute overlap-based loss between trainable and target PEPS."""
        if (mode == "hyper") and (copt is None):
            warnings.warn(
                "mode='hyper' requested but copt is None; provide copt_() for stable behavior.",
                RuntimeWarning,
                stacklevel=2,
            )

        peps.add_tag("KET")
        peps_fix.add_tag("KET")

        peps_h = peps.conj().retag({"KET": "BRA"})
        norm = peps_h | peps
        norm_ = peps_h | peps_fix

        if mode == "mps":
            val_0 = norm.contract_boundary(
                max_bond=chi,
                mode=mode_,
                final_contract_opts={"optimize": opt},
                progbar=progbar,
                layer_tags=["KET", "BRA"],
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=sequence,
                equalize_norms=equalize_norms,
            )
            val_1 = norm_.contract_boundary(
                max_bond=chi,
                mode=mode_,
                final_contract_opts={"optimize": opt},
                progbar=progbar,
                layer_tags=["KET", "BRA"],
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=sequence,
                equalize_norms=equalize_norms,
            )
        elif mode == "rg":
            val_0 = norm.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                canonize=True,
                mode="projector",
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                layer_tags=["KET", "BRA"],
                final_contract=True,
                final_contract_opts={"optimize": opt},
                progbar=progbar,
                inplace=False,
            )
            val_1 = norm_.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                canonize=True,
                mode="projector",
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                layer_tags=["KET", "BRA"],
                final_contract=True,
                final_contract_opts={"optimize": opt},
                progbar=progbar,
                inplace=False,
            )
        elif mode == "hyper":
            overlap = GlobalOptimizer._apply_hyperoptimized_compressed(
                norm,
                copt,
                chi,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                progbar=progbar,
            )
            val_0 = overlap ^ all
            overlap = GlobalOptimizer._apply_hyperoptimized_compressed(
                norm_,
                copt,
                chi,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                progbar=progbar,
            )
            val_1 = overlap ^ all
        elif mode == "exact":
            val_0 = norm.contract(all, optimize=opt)
            val_1 = norm_.contract(all, optimize=opt)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if cost_f == "fid":
            pr_norm = ar.do("abs", val_ * val_0)
            val_1 = ar.do("abs", val_1) ** 2
            in_f = 1 - val_1 / pr_norm
            return ar.do("abs", in_f)

        if cost_f == "logfidelity":
            val_0 = ar.do("abs", val_0)
            val_0 = ar.do("log", val_0)
            val_4 = ar.do("log", val_)
            val_1 = ar.do("abs", val_1)
            val_1 = ar.do("log", val_1)
            return -val_1 + (val_0 + val_4) * 0.5

        if cost_f == "dis":
            val_0 = ar.do("abs", val_0)
            val_2 = ar.do("conj", val_1)
            return abs(val_ + val_0 - val_1 - val_2)

        raise ValueError(f"Unknown cost function: {cost_f}")

    @staticmethod
    def _tnopt_loss(peps, *, peps_target, **loss_kwargs):
        """Adapter for ``qtn.TNOptimizer`` using :meth:`_loss_peps`."""
        return GlobalOptimizer._loss_peps(peps, peps_target, **loss_kwargs)

    def __init__(
        self,
        peps,
        peps_target=None,
        *,
        norm_kwargs: Mapping[str, Any] | None = None,
        normalize_kwargs: Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
    ):
        self.peps = peps
        self.peps_target = peps_target
        self.loss_kwargs = self._pick_known_keys(loss_kwargs, self._LOSS_KEYS)
        self.norm_kwargs = self._pick_known_keys(norm_kwargs, self._NORM_KEYS)

        if not self.norm_kwargs and loss_kwargs:
            self.norm_kwargs = self._pick_known_keys(
                loss_kwargs,
                self._NORM_KEYS,
                warn_unknown=False,
            )

        if normalize_kwargs is None:
            self.normalize_kwargs = dict(self.norm_kwargs)
        else:
            self.normalize_kwargs = self._pick_known_keys(normalize_kwargs, self._NORM_KEYS)

    def set_norm_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`norm`."""
        self.norm_kwargs.update(self._pick_known_keys(kwargs, self._NORM_KEYS))
        return self

    def set_normalize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`normalize`."""
        self.normalize_kwargs.update(self._pick_known_keys(kwargs, self._NORM_KEYS))
        return self

    def set_loss_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`loss` and TNOptimizer loss."""
        self.loss_kwargs.update(self._pick_known_keys(kwargs, self._LOSS_KEYS))
        return self

    def norm(self, peps=None, **kwargs):
        """Evaluate ``<peps|peps>`` with configured contraction options."""
        state = self.peps if peps is None else peps
        opts = self._merge_opts(self.norm_kwargs, kwargs)
        return self._norm_peps(state, **opts)

    def normalize(self, peps=None, **kwargs):
        """Normalize PEPS in place with configured contraction options."""
        state = self.peps if peps is None else peps
        opts = self._merge_opts(self.normalize_kwargs, kwargs)
        return self._normalize_peps(state, **opts)

    def loss(self, peps=None, *, peps_target=None, **kwargs):
        """Evaluate configured global loss against target PEPS."""
        state = self.peps if peps is None else peps
        target = self.peps_target if peps_target is None else peps_target
        if target is None:
            raise ValueError(
                "peps_target is required for loss(). "
                "Provide peps_target in constructor or call loss(peps_target=...)."
            )
        opts = self._merge_opts(self.loss_kwargs, kwargs)
        return self._loss_peps(state, target, **opts)

    def make_tn_optimizer(
        self,
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        **tnopt_kwargs,
    ):
        """Construct a configured :class:`quimb.tensor.TNOptimizer`."""
        merged_loss_kwargs = self._merge_opts(self.loss_kwargs, loss_kwargs)

        constants = {}
        if self.peps_target is not None:
            constants["peps_target"] = self.peps_target
        if loss_constants:
            constants.update(dict(loss_constants))
        if constants.get("peps_target") is None:
            raise ValueError(
                "peps_target is required for make_tn_optimizer(). "
                "Provide it in constructor or via loss_constants={'peps_target': ...}."
            )

        return qtn.TNOptimizer(
            self.peps,
            self._tnopt_loss,
            loss_constants=constants,
            loss_kwargs=merged_loss_kwargs,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
            **tnopt_kwargs,
        )

    def optimize(
        self,
        *,
        n=220,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        **optimize_kwargs,
    ):
        """Run TNOptimizer and return optimized PEPS."""
        tnopt = self.make_tn_optimizer(
            loss_kwargs=loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
        )
        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.peps = out
        return out
