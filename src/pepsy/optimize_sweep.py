"""High-level PEPS sweep optimizer with axis-alternating boundary updates."""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import quimb.tensor as qtn
from tqdm import tqdm

from .boundary_metrics import infidelity as boundary_infidelity
from .boundary_metrics import prepare_boundary_inputs, normalize
from .boundary_states import BdyMPS
from .boundary_sweeps import CompBdy
from .core import tn_fidelity
from .gradient_solver import optimize_packed_params as run_gradient_solver

_PHYS_IND_PATTERN = re.compile(r"^k\d+(?:,\d+)*$")
_TAG_X = re.compile(r"^X(\d+)$")
_TAG_Y = re.compile(r"^Y(\d+)$")

__all__ = ["SweepOptimizer", "SweepResult"]


@dataclass(frozen=True)
class SweepResult:
    """Return object for global sweep runs."""

    runs: list[dict[str, Any]]
    loss_before: float | None
    loss_after: float | None
    loss: list[float] | None = None
    step_loss_trace: list[float] | None = None
    step_trace: list[dict[str, Any]] | None = None
    inner_loss_traces: list[list[float]] | None = None
    norm_trace: list[float] | None = None


class SweepOptimizer:  # pylint: disable=too-many-instance-attributes
    """Optimize PEPS slices with alternating x/y boundary sweeps.

    Parameters
    ----------
    state : qtn.TensorNetwork
        Trainable PEPS-like tensor network.
    target : qtn.TensorNetwork
        Reference network for overlap objective.
    chi : int | None, default=None
        Boundary bond dimension used when ``bdy``/``bdy_overlap`` are not
        supplied.
    bdy : pepsy.boundary_states.BdyMPS | None, default=None
        Optional pre-built boundary container for norm contractions.
    bdy_overlap : pepsy.boundary_states.BdyMPS | None, default=None
        Optional pre-built boundary container for overlap contractions.
    opt : object | str, default="auto-hq"
        Contraction optimizer.
    dmrg_run : {"eff", "global"}, default="eff"
        Backend mode passed to :class:`pepsy.boundary_sweeps.CompBdy`.
    """

    _NORMALIZE_KEYS = frozenset({
        "opt",
        "n_iter",
        "direction",
        "max_separation",
        "pbar",
        "boundary_fidel",
    })
    _INFIDELITY_KEYS = frozenset({
        "chi",
        "norm",
        "norm_target",
        "opt",
        "n_iter",
        "direction",
        "max_separation",
        "pbar",
        "boundary_fidel",
        "single_layer",
    })
    _OPTIMIZE_KEYS = frozenset({
        "axes",
        "n",
        "n_round_trips",
        "chi",
        "optimizer",
        "optimizer_options",
        "env_n_iter",
        "progbar",
        "debug",
        "debug_loss_mode",
        "debug_loss_kwargs",
        "boundary_fidel",
        "renormalize",
    })
    _DEFAULT_SOLVER_OPTIONS = {
        "algorithm": "LBFGS",
        "lr": 1e-2,
        "n_steps": 50,
        "maxeval": 100,
        "ftol_rel": 1e-9,
        "xtol_rel": 1e-9,
        "patience": 40,
        "min_steps": 10,
        "restore_best": True,
        "bad_max": 20,
        "penalty_on_bad": 1e20,
    }

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

    @classmethod
    def default_solver_options(cls):
        """Return copy of package default local-solver options."""
        return dict(cls._DEFAULT_SOLVER_OPTIONS)

    @classmethod
    def normalize_kwarg_names(cls):
        """Return supported keyword names for :meth:`normalize` defaults."""
        return tuple(sorted(cls._NORMALIZE_KEYS))

    @classmethod
    def optimize_kwarg_names(cls):
        """Return supported keyword names for :meth:`set_optimize_kwargs`."""
        return tuple(sorted(cls._OPTIMIZE_KEYS))

    @classmethod
    def infidelity_kwarg_names(cls):
        """Return supported keyword names for :meth:`infidelity`."""
        return tuple(sorted(cls._INFIDELITY_KEYS))

    @classmethod
    def kwarg_guide(cls):
        """Return a compact guide of public kwargs and default solver options."""
        return {
            "normalize": cls.normalize_kwarg_names(),
            "optimize": cls.optimize_kwarg_names(),
            "infidelity": cls.infidelity_kwarg_names(),
            "optimizer_defaults": cls.default_solver_options(),
        }

    @classmethod
    def _merge_solver_options(cls, options):
        merged = cls.default_solver_options()
        if options:
            merged.update(dict(options))
        return merged

    @classmethod
    def _collect_init_renormalize_kwargs(
        cls,
        *,
        renormalize_kwargs=None,
        n_iter=None,
        direction=None,
        max_separation=None,
        pbar=None,
        boundary_fidel=None,
    ):
        """Collect init-time normalize kwargs from explicit and mapping styles."""
        legacy = {
            "n_iter": n_iter,
            "direction": direction,
            "max_separation": max_separation,
            "pbar": pbar,
            "boundary_fidel": boundary_fidel,
        }
        # Only forward values the caller explicitly set.
        legacy = {k: v for k, v in legacy.items() if v is not None}
        out = cls._pick_known_keys(legacy, cls._NORMALIZE_KEYS, warn_unknown=False)
        if renormalize_kwargs:
            out.update(cls._pick_known_keys(renormalize_kwargs, cls._NORMALIZE_KEYS))
        return out

    def __init__(
        self,
        state,
        state_target,
        *,
        chi=None,
        bdy=None,
        bdy_overlap=None,
        opt="auto-hq",
        dmrg_run="eff",
        single_layer=False,
        normalize_kwargs: Mapping[str, Any] | None = None,
        optimize_kwargs: Mapping[str, Any] | None = None,
        renormalize_state=False,
        renormalize_kwargs: Mapping[str, Any] | None = None,
        n_iter: int | None = None,
        direction: str | None = None,
        max_separation: int | None = None,
        pbar: bool | None = None,
        boundary_fidel: bool | None = None,
    ):
        if (bdy is None) ^ (bdy_overlap is None):
            raise ValueError("Provide both bdy and bdy_overlap together, or neither.")

        if bdy is None and bdy_overlap is None:
            if chi is None:
                raise ValueError(
                    "Provide chi when bdy and bdy_overlap are not supplied."
                )
            if state_target is None:
                raise ValueError("state_target is required when boundaries are not supplied.")
            bdy, bdy_overlap = self._build_boundary_pair(
                state,
                state_target,
                chi=chi,
                single_layer=single_layer,
            )
        for name, obj in (("bdy", bdy), ("bdy_overlap", bdy_overlap)):
            if not hasattr(obj, "mps_b"):
                raise TypeError(f"{name} must expose attribute 'mps_b'.")

        self.state = state
        self.state_target = state_target
        self.bdy = bdy
        self.bdy_overlap = bdy_overlap
        self.opt = opt
        self.dmrg_run = dmrg_run

        if normalize_kwargs is None:
            self.normalize_kwargs = {}
        else:
            self.normalize_kwargs = self._pick_known_keys(normalize_kwargs, self._NORMALIZE_KEYS)
        self.optimize_kwargs = self._pick_known_keys(optimize_kwargs, self._OPTIMIZE_KEYS)

        self.Lx, self.Ly = self._infer_shape(self.state)
        self._reset_run_traces()

        init_renormalize_kwargs = self._collect_init_renormalize_kwargs(
            renormalize_kwargs=renormalize_kwargs,
            n_iter=n_iter,
            direction=direction,
            max_separation=max_separation,
            pbar=pbar,
            boundary_fidel=boundary_fidel,
        )

        if renormalize_state:
            self.normalize(**init_renormalize_kwargs)

    @staticmethod
    def _build_boundary_pair(state, target, *, chi, single_layer=False):
        """Construct norm and overlap boundary MPS containers."""
        _, state_norm = prepare_boundary_inputs(ket=state, bra=None)
        _, overlap_norm = prepare_boundary_inputs(ket=target, bra=state)
        bdy = BdyMPS(
            tn_double=state_norm,
            chi=chi,
            single_layer=single_layer,
        )
        bdy_overlap = BdyMPS(
            tn_double=overlap_norm,
            chi=chi,
            single_layer=single_layer,
        )
        return bdy, bdy_overlap

    def _reset_run_traces(self):
        """Reset lightweight traces collected during :meth:`optimize_global`."""
        self.step_loss_trace = []
        # Keep legacy and new names synchronized.
        self.loss = self.step_loss_trace
        self.step_trace = []
        self.inner_loss_traces = []
        self.norm_trace = []

    @staticmethod
    def _to_float_history(history):
        """Convert solver history entries into plain Python floats."""
        values = []
        for entry in history or ():
            value = entry
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            values.append(float(value))
        return values

    @staticmethod
    def _fidelity_to_infidelity(value):
        """Convert fidelity scalar to non-negative infidelity."""
        if value is None:
            return None
        infid = 1.0 - float(complex(value).real)
        if infid < 0.0 and abs(infid) < 1e-12:
            infid = 0.0
        return infid

    def _collect_axis_run_traces(self, axis_runs, *, cycle, axis):
        """Collect per-step traces from axis run records."""
        if not axis_runs:
            return

        _ = axis
        for run_info in axis_runs:
            # In debug runs track exact per-step loss when available.
            local_loss = run_info.get("exact_loss_after")
            if local_loss is None:
                local_loss = run_info.get("loss_final")
            if local_loss is None:
                step_loss = float("nan")
            else:
                step_loss = float(local_loss)
            self.step_loss_trace.append(step_loss)

            history = self._to_float_history(run_info.get("history"))
            self.inner_loss_traces.append(history)

            time_boundary = run_info.get("time_boundary")
            time_optimize = run_info.get("time_optimize")
            tb = None if time_boundary is None else float(time_boundary)
            to = None if time_optimize is None else float(time_optimize)
            if tb is None or to is None:
                ttot = None
            else:
                ttot = tb + to

            self.step_trace.append(
                {
                    "cycle": int(cycle),
                    "axis": run_info.get("axis", axis),
                    "index": run_info.get("index"),
                    "move": run_info.get("sweep"),
                    "loss": step_loss,
                    "time_boundary": tb,
                    "time_optimize": to,
                    "time_total": ttot,
                    "boundary_fidelity_norm": run_info.get("boundary_fidelity_norm"),
                    "boundary_fidelity_overlap": run_info.get("boundary_fidelity_overlap"),
                }
            )

    _SOLVER_ALIASES = {
        "scipy": "scipy-lbfgs",
        "scipy_lbfgs": "scipy-lbfgs",
        "nlopt": "nlopt-lbfgs",
        "nlopt_lbfgs": "nlopt-lbfgs",
    }

    @staticmethod
    def _infer_shape(state):
        """Infer ``(Lx, Ly)`` from ``X*`` and ``Y*`` tags."""
        max_x = None
        max_y = None
        for tag in getattr(state, "tags", ()):
            mx = _TAG_X.match(tag)
            my = _TAG_Y.match(tag)
            if mx:
                max_x = max(int(mx.group(1)), -1 if max_x is None else max_x)
            if my:
                max_y = max(int(my.group(1)), -1 if max_y is None else max_y)
        if max_x is None or max_y is None:
            raise ValueError("state must include X*/Y* tags to infer lattice shape.")
        return max_x + 1, max_y + 1

    def metrics(self):
        """Return global normalized ``(fidelity, loss)``."""
        if self.state_target is None:
            raise ValueError(
                "state_target is required for metrics(). "
                "Set it in constructor or via set_target()."
            )
        fidelity = tn_fidelity(self.state, self.state_target, opt=self.opt)
        loss = 1.0 - fidelity
        return float(fidelity), float(loss)

    def _approx_infidelity_loss(self, *, env_n_iter=4, kwargs=None):
        """Return boundary-infidelity loss for lightweight global diagnostics."""
        infid_kwargs = self._pick_known_keys(
            kwargs,
            self._INFIDELITY_KEYS,
            warn_unknown=False,
        )
        infid_kwargs.setdefault("n_iter", env_n_iter)
        infid_kwargs.setdefault("pbar", False)
        infid_kwargs.setdefault("boundary_fidel", False)
        try:
            result = self.infidelity(**infid_kwargs)
        except (AttributeError, ValueError, TypeError):
            return None
        return float(result)

    def _ensure_boundary_chi(self, chi):
        """Retune both stored boundary objects to at least ``chi``."""
        if chi is None:
            return
        if not isinstance(chi, int):
            raise TypeError("chi must be an integer")
        if chi < 1:
            raise ValueError("chi must be >= 1")

        bdy = getattr(self, "bdy", None)
        if bdy is not None and getattr(bdy, "chi", 0) < chi:
            bdy.expand_bnd(chi, inplace=True)

        bdy_overlap = getattr(self, "bdy_overlap", None)
        if bdy_overlap is not None and getattr(bdy_overlap, "chi", 0) < chi:
            bdy_overlap.expand_bnd(chi, inplace=True)

    def set_chi(
        self,
        chi,
        *,
        normalize_state=False,
        n_iter=5,
        direction="y",
        max_separation=0,
        pbar=False,
        boundary_fidel=False,
    ):
        """Expand stored boundaries to ``chi`` and optionally renormalize state.

        Parameters
        ----------
        chi : int
            Target boundary bond dimension.
        normalize_state : bool, default=False
            If True, run :meth:`normalize` immediately after expanding
            boundaries.

        Returns
        -------
        SweepOptimizer
            ``self`` for chaining.
        """
        self._ensure_boundary_chi(chi)
        if normalize_state:
            self.normalize(
                n_iter=n_iter,
                direction=direction,
                max_separation=max_separation,
                pbar=pbar,
                boundary_fidel=boundary_fidel,
            )
        return self

    def infidelity(
        self,
        *,
        chi=None,
        norm=None,
        norm_target=1.0,
        opt=None,
        n_iter=5,
        direction="y",
        max_separation=0,
        pbar=False,
        boundary_fidel=False,
        single_layer=False,
    ):
        """Compute boundary-based infidelity for current ``(state, state_target)``.

        This reuses ``self.bdy`` and ``self.bdy_overlap``. If ``chi`` is
        provided and larger than current boundary bond dimension, both boundary
        objects are expanded before evaluation.

        Returns
        -------
        float
            Boundary infidelity value.
        """
        if self.state_target is None:
            raise ValueError(
                "state_target is required for infidelity(). "
                "Set it in constructor or via set_target()."
            )

        self._ensure_boundary_chi(chi)

        chi_for_call = chi
        if chi_for_call is None and norm_target is None:
            # infidelity() needs chi when it has to build bdy_target internally.
            chi_for_call = max(int(getattr(self.bdy, "chi", 1)), int(getattr(self.bdy_overlap, "chi", 1)))

        result = boundary_infidelity(
            self.state,
            self.state_target,
            chi=chi_for_call,
            norm=norm,
            norm_target=norm_target,
            bdy=self.bdy,
            bdy_overlap=self.bdy_overlap,
            opt=self.opt if opt is None else opt,
            n_iter=n_iter,
            direction=direction,
            max_separation=max_separation,
            pbar=pbar,
            boundary_fidel=boundary_fidel,
            dmrg_run=self.dmrg_run,
            single_layer=single_layer,
        )

        if result.get("bdy") is not None:
            self.bdy = result["bdy"]
        if result.get("bdy_overlap") is not None:
            self.bdy_overlap = result["bdy_overlap"]

        return float(result["infidelity"])

    def _debug_loss(self, *, mode="exact", kwargs=None):
        """Compute debug-only loss with selected evaluation mode."""
        mode = "exact" if mode is None else str(mode).strip().lower()
        if mode == "exact":
            _, loss = self.metrics()
            return float(loss)
        if mode in {"infidelity", "boundary", "bdy"}:
            return float(self.infidelity(**dict(kwargs or {})))
        raise ValueError("debug_loss_mode must be 'exact' or 'infidelity'.")

    def set_normalize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`normalize`."""
        if not hasattr(self, "normalize_kwargs"):
            self.normalize_kwargs = {}
        self.normalize_kwargs.update(self._pick_known_keys(kwargs, self._NORMALIZE_KEYS))
        return self

    def set_optimize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`optimize`."""
        if not hasattr(self, "optimize_kwargs"):
            self.optimize_kwargs = {}
        self.optimize_kwargs.update(self._pick_known_keys(kwargs, self._OPTIMIZE_KEYS))
        return self

    def _axis_n(self, axis):
        if axis == "y":
            return self.Ly
        if axis == "x":
            return self.Lx
        raise ValueError("axis must be 'x' or 'y'")

    @staticmethod
    def _axis_tag(axis):
        if axis == "y":
            return "Y"
        if axis == "x":
            return "X"
        raise ValueError("axis must be 'x' or 'y'")

    def _site_tensor_tags(self, axis, index):
        if axis == "y":
            return [f"I{x},{index}" for x in range(self.Lx)]
        if axis == "x":
            return [f"I{index},{y}" for y in range(self.Ly)]
        raise ValueError("axis must be 'x' or 'y'")

    @staticmethod
    def _boundary_direction(axis, side):
        return f"{axis}_{side}"

    def _prepare_current_double_layers(self):
        self.state, norm_tn = prepare_boundary_inputs(ket=self.state, bra=None)
        _, overlap_tn = prepare_boundary_inputs(ket=self.state_target, bra=self.state)

        return norm_tn, overlap_tn

    def _boundary_keys_for_index(self, index, axis):
        n = self._axis_n(axis)
        axis_tag = self._axis_tag(axis)
        if index < 0 or index > n - 1:
            raise ValueError(f"index must be in [0, {n - 1}] for axis={axis}")
        right_key = None if index == n - 1 else f"{axis_tag}{n - 2 - index}_r"
        left_key = f"{axis_tag}{index - 1}_l" if index > 0 else None
        return right_key, left_key

    @staticmethod
    def _attach_boundaries(tn, boundaries, *, right_key=None, left_key=None):
        out = tn
        if right_key is not None:
            out = out | boundaries[right_key]
        if left_key is not None:
            out = out | boundaries[left_key]
        return out

    @staticmethod
    def _resolve_user_solver(solver):
        """Normalize user-facing solver shortcuts and emit practical warnings."""
        if not isinstance(solver, str):
            raise TypeError("solver must be a string")
        solver = solver.strip().lower()
        solver = SweepOptimizer._SOLVER_ALIASES.get(solver, solver)
        if solver == "lbfgs":
            warnings.warn(
                "solver='lbfgs' defaults to SciPy L-BFGS-B in sweep optimization. "
                "Use solver='torch-lbfgs' to force torch.optim.LBFGS.",
                UserWarning,
                stacklevel=3,
            )
            return "scipy-lbfgs"
        if solver == "nlopt-lbfgs":
            warnings.warn(
                "solver='nlopt-lbfgs' uses NLopt on CPU float64 parameter vectors. "
                "Tune NLopt controls (algorithm/maxeval/ftol_rel/xtol_rel) for your problem.",
                UserWarning,
                stacklevel=3,
            )
        return solver

    def _optimize_packed_params(
        self,
        params_init,
        loss_fn,
        *,
        solver="scipy",
        solver_options=None,
    ):
        opts = self._merge_solver_options(solver_options)
        n_steps = int(opts.pop("n_steps", 100))
        return run_gradient_solver(
            params_init,
            loss_fn,
            solver=solver,
            solver_options=opts,
            n_steps=n_steps,
            pbar=False,
        )

    def _apply_slice_update(self, index, params_opt, skeleton, axis):
        tn_opt = qtn.unpack(params_opt, skeleton)
        tn_opt.balance_bonds_()

        for tag in self._site_tensor_tags(axis, index):
            self.state[tag].modify(data=tn_opt[tag].data)
        return tn_opt

    def _normalize_state(self, env_n_iter=4):
        return self.normalize(
            n_iter=env_n_iter,
            direction="y",
            max_separation=1,
            pbar=False,
            boundary_fidel=False,
        )

    def normalize(self, state=None, **kwargs):
        """Normalize PEPS in place and return the old norm estimate."""
        state = self.state if state is None else state
        opts = self._merge_opts(getattr(self, "normalize_kwargs", {}), kwargs)
        opt = opts.get("opt", self.opt)
        n_iter = opts.get("n_iter", 5)
        direction = opts.get("direction", "y")
        max_separation = opts.get("max_separation", 0)
        pbar = opts.get("pbar", False)
        boundary_fidel = opts.get("boundary_fidel", False)

        if state is self.state:
            return normalize(
                self.state,
                bdy=self.bdy,
                opt=opt,
                max_separation=max_separation,
                n_iter=n_iter,
                direction=direction,
                pbar=pbar,
                boundary_fidel=boundary_fidel,
                dmrg_run=self.dmrg_run,
            )

        chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi via optimizer boundaries before normalizing external state.")
        return normalize(
            state,
            chi=chi,
            opt=opt,
            max_separation=max_separation,
            n_iter=n_iter,
            direction=direction,
            pbar=pbar,
            boundary_fidel=boundary_fidel,
            dmrg_run=self.dmrg_run,
        )

    def set_state(
        self,
        state,
        *,
        chi=None,
        single_layer=False,
        normalize_state=True,
        n_iter=5,
        direction="y",
        max_separation=0,
        pbar=False,
        boundary_fidel=False,
    ):
        """Replace current state, rebuild boundaries, and optionally normalize.

        Parameters
        ----------
        state : qtn.TensorNetwork
            New trainable PEPS-like state.
        chi : int | None, default=None
            Boundary bond dimension for rebuilt boundaries. If omitted, uses
            ``self.bdy.chi`` when available.
        single_layer : bool, default=False
            Forwarded to :class:`pepsy.boundary_states.BdyMPS`.
        normalize_state : bool, default=True
            If True, normalize new state immediately using rebuilt boundaries.

        Returns
        -------
        complex | float | None
            Old norm returned by :meth:`normalize` when ``normalize_state`` is
            True, else None.
        """
        if self.state_target is None:
            raise ValueError("state_target must be set before calling set_state().")

        self.state = state
        self.Lx, self.Ly = self._infer_shape(self.state)

        if chi is None:
            chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")

        self.bdy, self.bdy_overlap = self._build_boundary_pair(
            self.state,
            self.state_target,
            chi=chi,
            single_layer=single_layer,
        )

        if normalize_state:
            return self.normalize(
                n_iter=n_iter,
                direction=direction,
                max_separation=max_separation,
                pbar=pbar,
                boundary_fidel=boundary_fidel,
            )
        return None

    def set_target(
        self,
        target,
        *,
        chi=None,
        single_layer=False,
    ):
        """Replace target state and rebuild overlap boundary immediately.

        Parameters
        ----------
        target : qtn.TensorNetwork
            New target PEPS-like state.
        chi : int | None, default=None
            Bond dimension for rebuilt overlap boundary. If omitted, uses
            ``self.bdy_overlap.chi`` when available.
        single_layer : bool, default=False
            Forwarded to :class:`pepsy.boundary_states.BdyMPS`.
        """
        self.state_target = target

        if chi is None:
            chi = getattr(self.bdy_overlap, "chi", None)
        if chi is None:
            chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")

        _, overlap_norm = prepare_boundary_inputs(ket=self.state_target, bra=self.state)
        self.bdy_overlap = BdyMPS(
            tn_double=overlap_norm,
            chi=chi,
            single_layer=single_layer,
        )

    def _make_comp_pair(self, norm_tn, overlap_tn):
        comp_norm = CompBdy(norm_tn, self.bdy.mps_b, opt=self.opt, dmrg_run=self.dmrg_run)
        comp_overlap = CompBdy(overlap_tn, self.bdy_overlap.mps_b, opt=self.opt, dmrg_run=self.dmrg_run)
        return comp_norm, comp_overlap

    @staticmethod
    def _set_comp_norms(comp_norm, comp_overlap, *, norm_tn, overlap_tn):
        comp_norm.norm = norm_tn
        comp_overlap.norm = overlap_tn

    def _refresh_right_boundaries_once(self, axis, *, env_n_iter=4):
        norm_tn, overlap_tn = self._prepare_current_double_layers()
        comp_norm, comp_overlap = self._make_comp_pair(norm_tn, overlap_tn)
        comp_overlap.move_bdy(
            n_iter=env_n_iter,
            pbar=False,
            direction=self._boundary_direction(axis, "right"),
            boundary_fidel=False,
        )
        comp_norm.move_bdy(
            n_iter=env_n_iter,
            pbar=False,
            direction=self._boundary_direction(axis, "right"),
            boundary_fidel=False,
        )

    def _advance_boundary_one_step(
        self,
        index,
        *,
        side,
        axis,
        comp_norm,
        comp_overlap,
        env_n_iter=4,
        boundary_fidel=False,
    ):
        n = self._axis_n(axis)
        if side == "left" and index <= 0:
            return {"norm": None, "overlap": None}
        if side == "right" and index >= n - 1:
            return {"norm": None, "overlap": None}
        pos = (index - 1) if side == "left" else (n - 2 - index)
        direction = self._boundary_direction(axis, side)
        for comp in (comp_overlap, comp_norm):
            comp.move_step_bdy(
                pos=pos,
                n_iter=env_n_iter,
                pbar=False,
                direction=direction,
                boundary_fidel=boundary_fidel,
            )
        overlap_fidelity = None
        norm_fidelity = None
        if boundary_fidel:
            if comp_overlap.fidel:
                overlap_fidelity = float(complex(comp_overlap.fidel[-1]).real)
            if comp_norm.fidel:
                norm_fidelity = float(complex(comp_norm.fidel[-1]).real)
        return {"norm": norm_fidelity, "overlap": overlap_fidelity}

    def _optimize_axis_slice_with_current_env(
        self,
        index,
        *,
        axis,
        solver="adam",
        solver_options=None,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
    ):
        _ = debug_loss_mode, debug_loss_kwargs
        axis_tag = self._axis_tag(axis)
        right_key, left_key = self._boundary_keys_for_index(index, axis)

        slice_state = self.state.select([f"{axis_tag}{index}"], "any")
        slice_target = self.state_target.select([f"{axis_tag}{index}"], "any")
        params_init, skeleton = qtn.pack(slice_state)

        def loss_fn(params_in):
            local = qtn.unpack(params_in, skeleton)
            bra_norm = local.conj()
            bra_norm.reindex_(
                {
                    idx: f"{idx}_*"
                    for idx in bra_norm.ind_map
                    if not (isinstance(idx, str) and _PHYS_IND_PATTERN.fullmatch(idx))
                }
            )
            bra_overlap = local.conj()

            norm_net = self._attach_boundaries(
                local | bra_norm,
                self.bdy.mps_b,
                right_key=right_key,
                left_key=left_key,
            )
            overlap_net = self._attach_boundaries(
                slice_target | bra_overlap,
                self.bdy_overlap.mps_b,
                right_key=right_key,
                left_key=left_key,
            )
            overlap_val = abs(overlap_net.contract(all, optimize=self.opt)) ** 2
            norm_val = abs(norm_net.contract(all, optimize=self.opt))
            return 1 - (overlap_val / (norm_val + 1e-12)).clamp(max=1.0)

        initial_loss = float(loss_fn(params_init))

        params_opt, history = self._optimize_packed_params(
            params_init,
            loss_fn,
            solver=solver,
            solver_options=solver_options,
        )
        history_values = self._to_float_history(history)
        final_loss_raw = loss_fn(params_opt)

        if hasattr(final_loss_raw, "detach"):
            final_loss_raw = final_loss_raw.detach()
        if hasattr(final_loss_raw, "cpu"):
            final_loss_raw = final_loss_raw.cpu()
        final_loss = float(final_loss_raw)
        params_opt = {k: v.detach().clone() for k, v in params_opt.items()}
        self._apply_slice_update(index, params_opt, skeleton, axis)
        run_info = {
            "axis": axis,
            "index": index,
            "right_key": right_key,
            "left_key": left_key,
            "loss_initial": initial_loss,
            "loss_final": final_loss,
        }
        run_info["history"] = history_values
        if not debug:
            # Strict non-debug path: do not compute any exact global diagnostics.
            return run_info

        exact_loss = 1.0 - tn_fidelity(self.state, self.state_target, opt=self.opt)
        run_info["exact_loss_after"] = float(exact_loss)
        state_norm = abs(complex((self.state.H & self.state).contract(all, optimize=self.opt)))
        run_info["state_norm_after"] = float(state_norm)

        return run_info

    def _run_axis_half_sweep(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        indices,
        *,
        axis,
        update_side,
        sweep_name,
        solver="scipy",
        solver_options=None,
        env_n_iter=4,
        run_callback=None,
        boundary_fidel=False,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
    ):
        """Run a single forward or backward half-sweep over *indices*."""
        runs = []
        comp_norm = None
        comp_overlap = None

        for index in indices:
            norm_tn, overlap_tn = self._prepare_current_double_layers()
            if comp_norm is None:
                comp_norm, comp_overlap = self._make_comp_pair(norm_tn, overlap_tn)
            else:
                self._set_comp_norms(comp_norm, comp_overlap, norm_tn=norm_tn, overlap_tn=overlap_tn)

            t0 = time.perf_counter()
            boundary_fidelity = self._advance_boundary_one_step(
                index,
                side=update_side,
                axis=axis,
                comp_norm=comp_norm,
                comp_overlap=comp_overlap,
                env_n_iter=env_n_iter,
                boundary_fidel=boundary_fidel,
            )
            boundary_infidelity_norm = self._fidelity_to_infidelity(
                boundary_fidelity.get("norm")
            )
            boundary_infidelity_overlap = self._fidelity_to_infidelity(
                boundary_fidelity.get("overlap")
            )
            t_bdy = time.perf_counter() - t0

            t0 = time.perf_counter()
            run_info = self._optimize_axis_slice_with_current_env(
                index,
                axis=axis,
                solver=solver,
                solver_options=solver_options,
                debug=debug,
                debug_loss_mode=debug_loss_mode,
                debug_loss_kwargs=debug_loss_kwargs,
            )
            t_opt = time.perf_counter() - t0

            run_info["sweep"] = sweep_name
            run_info["time_boundary"] = t_bdy
            run_info["time_optimize"] = t_opt
            run_info["boundary_infidelity_norm"] = boundary_infidelity_norm
            run_info["boundary_infidelity_overlap"] = boundary_infidelity_overlap
            runs.append(run_info)
            if run_callback is not None:
                run_callback(run_info)

        return runs

    def optimize_axis(
        self,
        axis,
        *,
        n_round_trips=1,
        solver="scipy",
        solver_options=None,
        env_n_iter=4,
        run_callback=None,
        boundary_fidel=False,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
        renormalize=False,
    ):
        """Run one axis with forward + round-trip sweeps.

        Parameters
        ----------
        solver : str, default="scipy"
            Gradient solver name. Supported values include ``adam``, ``lbfgs``,
            ``scipy-lbfgs``, and ``nlopt-lbfgs``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        """
        if self.state_target is None:
            raise ValueError(
                "state_target is required for optimize_axis(). "
                "Set it in constructor or via set_target()."
            )
        n = self._axis_n(axis)
        resolved_solver = self._resolve_user_solver(solver)
        all_runs = []

        self.bdy.normalize()
        self.bdy_overlap.normalize()

        if renormalize:
            old_norm = self._normalize_state(env_n_iter)
            self.norm_trace.append(float(abs(complex(old_norm))))

        self._refresh_right_boundaries_once(axis, env_n_iter=env_n_iter)

        sweep_kwargs = dict(
            axis=axis,
            solver=resolved_solver,
            solver_options=solver_options,
            env_n_iter=env_n_iter,
            run_callback=run_callback,
            boundary_fidel=boundary_fidel,
            debug=debug,
            debug_loss_mode=debug_loss_mode,
            debug_loss_kwargs=debug_loss_kwargs,
        )

        all_runs.extend(
            self._run_axis_half_sweep(
                range(0, n),
                update_side="left",
                sweep_name="forward",
                **sweep_kwargs,
            )
        )

        for _trip in range(n_round_trips):
            all_runs.extend(
                self._run_axis_half_sweep(
                    range(n - 2, -1, -1),
                    update_side="right",
                    sweep_name="backward",
                    **sweep_kwargs,
                )
            )
            forward_start = 1 if n > 1 else n
            all_runs.extend(
                self._run_axis_half_sweep(
                    range(forward_start, n),
                    update_side="left",
                    sweep_name="forward",
                    **sweep_kwargs,
                )
            )

        return all_runs

    def optimize_global(
        self,
        *,
        axes=("y", "x"),
        n_cycles=1,
        n_round_trips=1,
        chi=None,
        solver="scipy",
        solver_options=None,
        env_n_iter=4,
        pbar=True,
        debug=False,
        debug_loss_mode="exact",
        debug_loss_kwargs=None,
        boundary_fidel=None,
        renormalize=False,
    ):
        """Run alternating axis sweeps and return a :class:`SweepResult`.

        Parameters
        ----------
        chi : int | None, default=None
            If provided, expand stored boundaries to this bond dimension before
            running sweeps.
        solver : str, default="scipy"
            Gradient solver name. Supported values include ``adam``, ``lbfgs``,
            ``scipy-lbfgs``, and ``nlopt-lbfgs``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        debug : bool, default=False
            When True, compute exact global-loss diagnostics. When False, exact
            contractions are skipped and only boundary/local quantities are
            tracked.
        debug_loss_mode : {"exact", "infidelity"}, default="exact"
            Diagnostic loss backend used only when ``debug=True``.
        debug_loss_kwargs : Mapping[str, Any] | None, default=None
            Extra kwargs passed to debug loss backend. For
            ``debug_loss_mode="infidelity"`` these are forwarded to
            :meth:`infidelity`.
        boundary_fidel : bool | None, default=None
            Boundary-fidelity tracing flag for CompBdy updates. If ``None``,
            follows ``debug``. Set explicitly to force behavior.

        """
        if self.state_target is None:
            raise ValueError(
                "state_target is required for optimize_global(). "
                "Set it in constructor or via set_target()."
            )
        debug = bool(debug)
        self._ensure_boundary_chi(chi)
        self._reset_run_traces()
        if debug:
            loss_before = self._debug_loss(
                mode=debug_loss_mode,
                kwargs=debug_loss_kwargs,
            )
        else:
            loss_before = self._approx_infidelity_loss(
                env_n_iter=env_n_iter,
                kwargs=debug_loss_kwargs,
            )
        boundary_fidel = bool(debug) if boundary_fidel is None else bool(boundary_fidel)
        all_runs = []
        axis_seq = list(axes)

        def _steps_for_axis(axis_name):
            n = self._axis_n(axis_name)
            return n + (2 * n_round_trips * max(n - 1, 0))

        total_steps = n_cycles * sum(_steps_for_axis(axis_name) for axis_name in axis_seq)
        global_progress = None
        if pbar:
            global_progress = tqdm(
                total=total_steps,
                desc="bdy_dmrg:",
                leave=True,
                position=0,
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="magenta",
                disable=not pbar,
            )

        for cyc in range(n_cycles):
            for axis in axis_seq:
                def _on_run(run_info, *, cyc_num=cyc + 1, axis_name=axis):
                    _ = cyc_num, axis_name
                    if global_progress is None:
                        return
                    global_progress.update(1)
                    postfix = {}
                    local_loss = run_info.get("loss_final")
                    if local_loss is not None:
                        postfix["loss"] = f"{float(local_loss):.6f}"
                    if debug:
                        exact_loss = run_info.get("exact_loss_after")
                        if exact_loss is not None:
                            postfix["loss_ex"] = f"{float(exact_loss):.6f}"
                    axis_name = run_info.get("axis")
                    sweep_name = run_info.get("sweep")
                    index = run_info.get("index")
                    if axis_name is not None and sweep_name is not None and index is not None:
                        short = "fwd" if sweep_name == "forward" else "bwd"
                        postfix["slice"] = f"{axis_name}_{short}_{index}"
                    global_progress.set_postfix(postfix)

                axis_runs = self.optimize_axis(
                    axis,
                    n_round_trips=n_round_trips,
                    solver=solver,
                    solver_options=solver_options,
                    env_n_iter=env_n_iter,
                    run_callback=_on_run,
                    boundary_fidel=boundary_fidel,
                    debug=debug,
                    debug_loss_mode=debug_loss_mode,
                    debug_loss_kwargs=debug_loss_kwargs,
                    renormalize=renormalize,
                )
                all_runs.extend(axis_runs)
                self._collect_axis_run_traces(
                    axis_runs,
                    cycle=cyc + 1,
                    axis=axis,
                )

        if global_progress is not None:
            global_progress.close()

        if renormalize:
            old_norm = self._normalize_state(env_n_iter=env_n_iter)
            self.norm_trace.append(float(abs(complex(old_norm))))

        if debug:
            loss_after = self._debug_loss(
                mode=debug_loss_mode,
                kwargs=debug_loss_kwargs,
            )
        else:
            loss_after = self._approx_infidelity_loss(
                env_n_iter=env_n_iter,
                kwargs=debug_loss_kwargs,
            )
        result = SweepResult(
            runs=all_runs,
            loss_before=loss_before,
            loss_after=loss_after,
            loss=list(self.loss),
            step_loss_trace=list(self.step_loss_trace),
            step_trace=list(self.step_trace),
            inner_loss_traces=[list(v) for v in self.inner_loss_traces],
            norm_trace=list(self.norm_trace),
        )
        return result

    def run(
        self,
        *,
        n=None,
        chi=None,
        progbar=None,
        debug=None,
        renormalize=None,
    ):
        """High-level wrapper around :meth:`optimize_global`.

        Parameters
        ----------
        n : int, default=1
            Number of global y/x cycles.
        chi : int | None, default=None
            If provided, expand boundary bond dimension before sweeps.
        progbar : bool, default=True
            Show global sweep progress bar.
        debug : bool | None, default=None
            Enable exact global metrics collection when True.
        """
        opts = dict(getattr(self, "optimize_kwargs", {}))
        if n is not None:
            opts["n"] = n
        if chi is not None:
            opts["chi"] = chi
        if progbar is not None:
            opts["progbar"] = progbar
        if debug is not None:
            opts["debug"] = debug
        if renormalize is not None:
            opts["renormalize"] = renormalize

        return self.optimize_global(
            axes=opts.get("axes", ("y", "x")),
            n_cycles=opts.get("n", 1),
            n_round_trips=opts.get("n_round_trips", 1),
            chi=opts.get("chi"),
            solver=opts.get("optimizer", "scipy"),
            solver_options=opts.get("optimizer_options"),
            env_n_iter=opts.get("env_n_iter", 4),
            pbar=opts.get("progbar", True),
            debug=opts.get("debug", False),
            debug_loss_mode=opts.get("debug_loss_mode", "exact"),
            debug_loss_kwargs=opts.get("debug_loss_kwargs"),
            renormalize=opts.get("renormalize", False),
        )
