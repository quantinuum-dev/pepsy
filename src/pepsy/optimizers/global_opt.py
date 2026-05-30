"""Global state objective helpers centered on :class:`GlobalOptimizer`."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import autoray as ar
import quimb.tensor as qtn

from ..tensors.core import build_optimizer, contract_hypercompressed_tn

__all__ = ["GlobalOptimizer"]

_DEFAULT_SEQUENCE = ("xmax", "xmin", "ymin", "ymax")
_DEFAULT_LAYER_TAGS = ("KET", "BRA")
_DEFAULT_NLOPT_FTOL_REL = 1e-9
_DEFAULT_NLOPT_FTOL_ABS = 0.0
_DEFAULT_NLOPT_XTOL_REL = 1e-9
_DEFAULT_NLOPT_XTOL_ABS = 0.0


class GlobalOptimizer:
    """High-level wrapper for global state optimization objectives.

    This class keeps a trainable state and target state together with reusable
    defaults. Public entrypoints are :meth:`norm`, :meth:`normalize`,
    :meth:`loss`, :meth:`make_tn_optimizer`, and :meth:`optimize`.

    Internal objective kernels are implemented as:

    - :meth:`_norm_state`
    - :meth:`_normalize_state`
    - :meth:`_loss_state`
    """

    _NORM_KEYS = frozenset({
        "opt",
        "copt",
        "contraction_opt",
        "contraction_opt_hyper",
        "hyper_contraction_opt",
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
        "contraction_opt",
        "contraction_opt_hyper",
        "hyper_contraction_opt",
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

    def __init__(
        self,
        state,
        state_target=None,
        *,
        chi: int | tuple[int, int] = 64,
        norm_kwargs: Mapping[str, Any] | None = None,
        normalize_kwargs: Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_opt: Mapping[str, Any] | None = None,
    ):
        self.state = state
        self.state_target = state_target
        self.losses: list[float] = []
        merged_loss_options = self._merge_opts(loss_kwargs, loss_opt)
        self.loss_opt = self._pick_known_keys(merged_loss_options, self._LOSS_KEYS)
        self.norm_kwargs = self._pick_known_keys(norm_kwargs, self._NORM_KEYS)

        if not self.norm_kwargs and merged_loss_options:
            self.norm_kwargs = self._pick_known_keys(
                merged_loss_options,
                self._NORM_KEYS,
                warn_unknown=False,
            )

        self.norm_kwargs.setdefault("chi", chi)
        if normalize_kwargs is None:
            # norm_kwargs and normalize_kwargs are the same dict by default.
            self.normalize_kwargs = self.norm_kwargs
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

    def set_loss_opt(self, **kwargs):
        """Update stored defaults for :meth:`loss` and TNOptimizer loss."""
        self.loss_opt.update(self._pick_known_keys(kwargs, self._LOSS_KEYS))
        return self

    def set_loss_kwargs(self, **kwargs):
        """Update stored loss defaults via :meth:`set_loss_opt`."""
        return self.set_loss_opt(**kwargs)

    def set_target(self, state_target, *, inplace=True):
        """Replace stored target state used by :meth:`loss` and optimizers.

        Parameters
        ----------
        state_target : qtn.TensorNetwork | None
            New target state. Pass ``None`` to clear the stored target.
        inplace : bool, default=True
            If ``True``, store ``state_target`` directly. If ``False`` and
            ``state_target`` has ``copy()``, store a shallow copy instead.

        Returns
        -------
        GlobalOptimizer
            ``self`` for fluent chaining.
        """
        if (not inplace) and state_target is not None and hasattr(state_target, "copy"):
            state_target = state_target.copy()
        self.state_target = state_target
        return self

    @classmethod
    def norm_kwarg_names(cls):
        """Return supported keyword names for :meth:`norm` defaults."""
        return tuple(sorted(cls._NORM_KEYS))

    @classmethod
    def normalize_kwarg_names(cls):
        """Return supported keyword names for :meth:`normalize` defaults."""
        return tuple(sorted(cls._NORM_KEYS))

    @classmethod
    def loss_kwarg_names(cls):
        """Return supported keyword names for :meth:`loss` defaults."""
        return tuple(sorted(cls._LOSS_KEYS))

    @classmethod
    def kwarg_guide(cls):
        """Return compact guide of public kwargs for this optimizer."""
        return {
            "norm": cls.norm_kwarg_names(),
            "normalize": cls.normalize_kwarg_names(),
            "loss": cls.loss_kwarg_names(),
        }

    @staticmethod
    def _merge_opts(base, extra):
        merged = dict(base or {})
        if extra:
            merged.update(dict(extra))
        return merged

    @staticmethod
    def _normalize_optimizer_name(optimizer):
        """Normalize common optimizer aliases for ``qtn.TNOptimizer``."""
        if not isinstance(optimizer, str):
            return optimizer

        key = optimizer.strip().lower().replace("_", "-")
        if key in {"lbfgs", "l-bfgs", "lbfgsb", "l-bfgs-b"}:
            return "L-BFGS-B"
        return optimizer

    @staticmethod
    def _normalize_nlopt_optimizer_name(optimizer):
        """Normalize NLOPT-prefixed aliases to TNOptimizer optimizer names."""
        if not isinstance(optimizer, str):
            return optimizer
        key = optimizer.strip().lower().replace("_", "-")
        if key == "nlopt":
            return "l-bfgs-b"
        if key.startswith("nlopt-"):
            return key.removeprefix("nlopt-")
        return optimizer

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
        filtered = cls._normalize_contraction_option_keys(filtered, warn_conflict=warn_unknown)
        return filtered

    @staticmethod
    def _normalize_contraction_option_keys(options, *, warn_conflict=True):
        """Normalize canonical contraction keyword aliases."""
        opts = dict(options or {})
        if ("contraction_opt" not in opts) and ("opt" in opts):
            opts["contraction_opt"] = opts.pop("opt")

        hyper_key = None
        if "contraction_opt_hyper" in opts:
            hyper_key = "contraction_opt_hyper"
        elif "hyper_contraction_opt" in opts:
            hyper_key = "hyper_contraction_opt"
        if hyper_key is not None:
            hv = opts.pop(hyper_key)
            if hv is not None:
                if ("copt" in opts) and (opts["copt"] != hv) and warn_conflict:
                    warnings.warn(
                        "Both 'copt' and a hyper contraction option were provided; "
                        "using the canonical hyper contraction option.",
                        UserWarning,
                        stacklevel=3,
                    )
                opts["copt"] = hv

        return opts

    @staticmethod
    def _split_chi(chi):
        """Parse ``chi`` for norm/overlap contractions.

        ``chi`` may be either:
        - ``int``: same value is used for norm and overlap contractions.
        - ``(int, int)``: ``(chi_norm, chi_overlap)``.
        """
        if isinstance(chi, tuple):
            if len(chi) != 2:
                raise ValueError("chi tuple must be length 2: (chi_norm, chi_overlap).")
            chi_norm, chi_overlap = chi
        else:
            chi_norm = chi
            chi_overlap = chi

        if not isinstance(chi_norm, Integral) or not isinstance(chi_overlap, Integral):
            raise TypeError("chi must be an int or a tuple of two ints.")
        if chi_norm <= 0 or chi_overlap <= 0:
            raise ValueError("chi values must be positive.")

        return int(chi_norm), int(chi_overlap)

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
        return contract_hypercompressed_tn(
            tn=tn,
            copt=copt,
            max_bond=max_bond,
            chi=max_bond,
            output_inds=output_inds,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            cutoff=cutoff,
            progbar=progbar,
            inplace=False,
        )

    @staticmethod
    def _norm_state(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        state,
        opt=None,
        contraction_opt=None,
        copt=None,
        contraction_opt_hyper=None,
        hyper_contraction_opt=None,
        chi=64,
        layer_tags=None,
        equalize_norms=False,
        mode="mps",
        mode_="mps",
        max_separation=1,
        sequence=None,
        progbar=False,
        cutoff=1e-12,
    ):
        """Compute ``<state|state>`` using exact, boundary, rg, or hyper modes."""
        chi_norm, _ = GlobalOptimizer._split_chi(chi)
        kw = GlobalOptimizer._normalize_contraction_option_keys(
            {
                "opt": opt,
                "contraction_opt": contraction_opt,
                "copt": copt,
                "contraction_opt_hyper": contraction_opt_hyper,
                "hyper_contraction_opt": hyper_contraction_opt,
            },
            warn_conflict=False,
        )
        contraction_opt = kw.get("contraction_opt")
        copt = kw.get("copt")

        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)

        if layer_tags is None:
            layer_tags = list(_DEFAULT_LAYER_TAGS)
        if sequence is None:
            sequence = list(_DEFAULT_SEQUENCE)
        if (mode == "hyper") and (copt is None):
            warnings.warn(
                "mode='hyper' requested but no hyper contraction optimizer was supplied. "
                "Provide `contraction_opt_hyper`.",
                RuntimeWarning,
                stacklevel=2,
            )

        state.add_tag("KET")
        state_h = state.conj().retag({"KET": "BRA"})
        norm = state_h | state

        if mode == "mps":
            return norm.contract_boundary(
                max_bond=chi_norm,
                mode=mode_,
                sequence=sequence,
                final_contract_opts={"optimize": contraction_opt},
                cutoff=cutoff,
                progbar=progbar,
                layer_tags=layer_tags,
                max_separation=max_separation,
                equalize_norms=equalize_norms,
            )

        if mode == "rg":
            return norm.contract_ctmrg(
                max_bond=chi_norm,
                cutoff=cutoff,
                canonize=True,
                mode="projector",
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                layer_tags=list(_DEFAULT_LAYER_TAGS),
                final_contract=True,
                final_contract_opts={"optimize": contraction_opt},
                progbar=progbar,
                inplace=False,
            )

        if mode == "exact":
            norm = norm.full_simplify(seq="R", output_inds={}, split_method="svd", inplace=False)
            return norm.contract(all, optimize=contraction_opt)

        if mode == "hyper":
            overlap = GlobalOptimizer._apply_hyperoptimized_compressed(
                norm,
                copt,
                chi_norm,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                progbar=progbar,
            )
            return overlap ^ all

        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _normalize_state(state, **norm_opts):
        """Normalize a state in place via :meth:`_norm_state`."""
        state /= GlobalOptimizer._norm_state(state, **norm_opts) ** 0.5
        return state

    @staticmethod
    def _loss_state(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        state,
        state_target,
        opt=None,
        contraction_opt=None,
        copt=None,
        contraction_opt_hyper=None,
        hyper_contraction_opt=None,
        cost_f="fid",
        val_=1.0,
        progbar=False,
        chi=64,
        mode="mps",
        cutoff=1e-12,
        mode_="mps",
        max_separation=1,
        equalize_norms=False,
        sequence=_DEFAULT_SEQUENCE,
    ):
        """Compute overlap-based loss between trainable and target states."""
        chi_norm, chi_overlap = GlobalOptimizer._split_chi(chi)
        kw = GlobalOptimizer._normalize_contraction_option_keys(
            {
                "opt": opt,
                "contraction_opt": contraction_opt,
                "copt": copt,
                "contraction_opt_hyper": contraction_opt_hyper,
                "hyper_contraction_opt": hyper_contraction_opt,
            },
            warn_conflict=False,
        )
        contraction_opt = kw.get("contraction_opt")
        copt = kw.get("copt")

        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)

        if (mode == "hyper") and (copt is None):
            warnings.warn(
                "mode='hyper' requested but no hyper contraction optimizer was supplied. "
                "Provide `contraction_opt_hyper`.",
                RuntimeWarning,
                stacklevel=2,
            )

        state.add_tag("KET")
        state_target.add_tag("KET")

        state_h = state.conj().retag({"KET": "BRA"})
        norm = state_h | state
        norm_ = state_h | state_target

        if mode == "mps":
            val_0 = norm.contract_boundary(
                max_bond=chi_norm,
                mode=mode_,
                final_contract_opts={"optimize": contraction_opt},
                progbar=progbar,
                layer_tags=list(_DEFAULT_LAYER_TAGS),
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=sequence,
                equalize_norms=equalize_norms,
            )
            val_1 = norm_.contract_boundary(
                max_bond=chi_overlap,
                mode=mode_,
                final_contract_opts={"optimize": contraction_opt},
                progbar=progbar,
                layer_tags=list(_DEFAULT_LAYER_TAGS),
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=sequence,
                equalize_norms=equalize_norms,
            )
        elif mode == "rg":
            val_0 = norm.contract_ctmrg(
                max_bond=chi_norm,
                cutoff=cutoff,
                canonize=True,
                mode="projector",
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                layer_tags=list(_DEFAULT_LAYER_TAGS),
                final_contract=True,
                final_contract_opts={"optimize": contraction_opt},
                progbar=progbar,
                inplace=False,
            )
            val_1 = norm_.contract_ctmrg(
                max_bond=chi_overlap,
                cutoff=cutoff,
                canonize=True,
                mode="projector",
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                layer_tags=list(_DEFAULT_LAYER_TAGS),
                final_contract=True,
                final_contract_opts={"optimize": contraction_opt},
                progbar=progbar,
                inplace=False,
            )
        elif mode == "hyper":
            overlap = GlobalOptimizer._apply_hyperoptimized_compressed(
                norm,
                copt,
                chi_norm,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                progbar=progbar,
            )
            val_0 = overlap ^ all
            overlap = GlobalOptimizer._apply_hyperoptimized_compressed(
                norm_,
                copt,
                chi_overlap,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                progbar=progbar,
            )
            val_1 = overlap ^ all
        elif mode == "exact":
            val_0 = norm.contract(all, optimize=contraction_opt)
            val_1 = norm_.contract(all, optimize=contraction_opt)
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
    def _tnopt_loss(state, *, state_target, **loss_kwargs):
        """Adapter for ``qtn.TNOptimizer`` using :meth:`_loss_state`."""
        return GlobalOptimizer._loss_state(state, state_target, **loss_kwargs)

    def norm(self, state=None, **kwargs):
        """Evaluate ``<state|state>`` and return a plain Python scalar."""
        state = self.state if state is None else state
        if kwargs:
            self.norm_kwargs.update(self._pick_known_keys(kwargs, self._NORM_KEYS, warn_unknown=False))
        val = self._norm_state(state, **self.norm_kwargs)
        try:
            return complex(val)
        except Exception:
            return val

    def normalize(self, state=None, **kwargs):
        """Normalize state in place with configured contraction options."""
        state = self.state if state is None else state
        if kwargs:
            self.norm_kwargs.update(self._pick_known_keys(kwargs, self._NORM_KEYS, warn_unknown=False))
        return self._normalize_state(state, **self.norm_kwargs)

    def loss(self, state=None, *, state_target=None, **kwargs):
        """Evaluate configured global loss against target PEPS."""
        state = self.state if state is None else state
        target = self.state_target if state_target is None else state_target
        if target is None:
            raise ValueError(
                "state_target is required for loss(). "
                "Provide state_target in constructor or call loss(state_target=...)."
            )
        opts = self._merge_opts(self.loss_opt, kwargs)
        return self._loss_state(state, target, **opts)

    def make_tn_optimizer(
        self,
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_opt: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        **tnopt_kwargs,
    ):
        """Construct a configured :class:`quimb.tensor.TNOptimizer` for this state."""
        merged_loss_opt = self._merge_opts(self.loss_opt, self._merge_opts(loss_kwargs, loss_opt))
        optimizer = self._normalize_optimizer_name(optimizer)

        constants = {}
        if self.state_target is not None:
            constants["state_target"] = self.state_target
        if loss_constants:
            constants.update(dict(loss_constants))
        if constants.get("state_target") is None:
            raise ValueError(
                "state_target is required for make_tn_optimizer(). "
                "Provide it in constructor or via loss_constants={'state_target': ...}."
            )

        return qtn.TNOptimizer(
            self.state,
            self._tnopt_loss,
            loss_constants=constants,
            loss_kwargs=merged_loss_opt,
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
        chi: int | tuple[int, int] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_opt: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        return_losses: bool = False,
        **optimize_kwargs,
    ):
        """Run ``TNOptimizer.optimize``. Losses stored in ``self.losses``."""
        if isinstance(optimizer, str):
            key = optimizer.strip().lower().replace("_", "-")
            if key == "nlopt" or key.startswith("nlopt-"):
                raise ValueError(
                    "NLopt optimizers are handled by optimize_nlopt(). "
                    "Use optimize_nlopt(..., optimizer='nlopt')."
                )

        merged_loss_opt = self._merge_opts(self._merge_opts(loss_kwargs, loss_opt), {"chi": chi} if chi is not None else None)
        tnopt = self.make_tn_optimizer(
            loss_kwargs=loss_kwargs,
            loss_opt=merged_loss_opt,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
        )
        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.losses = list(getattr(tnopt, "losses", ()))
        if self.normalize_kwargs and hasattr(out, "add_tag"):
            out = self._normalize_state(out, **self.normalize_kwargs)
        self.state = out
        if return_losses:
            return out, tuple(self.losses)
        return out

    def optimize_nlopt(
        self,
        *,
        n=220,
        chi: int | tuple[int, int] | None = None,
        tol=None,
        jac=True,
        hessp=False,
        ftol_rel=_DEFAULT_NLOPT_FTOL_REL,
        ftol_abs=_DEFAULT_NLOPT_FTOL_ABS,
        xtol_rel=_DEFAULT_NLOPT_XTOL_REL,
        xtol_abs=_DEFAULT_NLOPT_XTOL_ABS,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_opt: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "nlopt",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        return_losses: bool = False,
        **tnopt_kwargs,
    ):
        """Run ``TNOptimizer.optimize_nlopt``. Losses stored in ``self.losses``."""
        optimizer_for_tn = self._normalize_nlopt_optimizer_name(optimizer)
        merged_loss_opt = self._merge_opts(self._merge_opts(loss_kwargs, loss_opt), {"chi": chi} if chi is not None else None)
        tnopt = self.make_tn_optimizer(
            loss_kwargs=loss_kwargs,
            loss_opt=merged_loss_opt,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer_for_tn,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
            **tnopt_kwargs,
        )
        try:
            out = tnopt.optimize_nlopt(
                n=n,
                tol=tol,
                jac=jac,
                hessp=hessp,
                ftol_rel=ftol_rel,
                ftol_abs=ftol_abs,
                xtol_rel=xtol_rel,
                xtol_abs=xtol_abs,
            )
        except Exception as exc:
            # nlopt raises nlopt.nlopt.runtime_error which does NOT inherit
            # from Python's RuntimeError, so quimb's internal handler misses
            # it.  Fall back to the best state found so far.
            if "nlopt" in type(exc).__module__:
                warnings.warn(
                    f"NLopt optimization stopped early: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                out = tnopt.get_tn_opt()
            else:
                raise
        self.losses = list(getattr(tnopt, "losses", ()))
        if self.normalize_kwargs and hasattr(out, "add_tag"):
            out = self._normalize_state(out, **self.normalize_kwargs)
        self.state = out
        if return_losses:
            return out, tuple(self.losses)
        return out
