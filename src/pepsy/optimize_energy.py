"""Energy-based PEPS sweep optimizer with axis-alternating boundary updates."""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import quimb.tensor as qtn
from tqdm import tqdm

from .boundary_metrics import ContractBoundary, normalize, prepare_boundary_inputs
from .boundary_states import BdyMPS
from .boundary_sweeps import CompBdy
from .gradient_solver import optimize_packed_params as run_gradient_solver

_PHYS_IND_PATTERN = re.compile(r"^k\d+(?:,\d+)*$")
_TAG_X = re.compile(r"^X(\d+)$")
_TAG_Y = re.compile(r"^Y(\d+)$")

__all__ = ["EnergyOptimizer", "EnergyResult"]


@dataclass(frozen=True)
class EnergyResult:
    """Return object for global energy-sweep runs."""

    runs: list[dict[str, Any]]
    energy_before: float | None
    energy_after: float | None
    energy: list[float] | None = None
    norm_trace: list[float] | None = None


class EnergyOptimizer:  # pylint: disable=too-many-instance-attributes
    """Optimize PEPS slices to minimize energy ``<psi|H|psi>/<psi|psi>``.

    Parameters
    ----------
    state : qtn.TensorNetwork
        Trainable PEPS-like state.
    pepo : qtn.TensorNetwork
        Hamiltonian PEPO with physical indices ``k...`` (upper) and ``b...``
        (lower) matching the lattice shape of ``state``.
    chi : int | None, default=None
        Boundary bond dimension used when ``bdy``/``bdy_energy`` are not
        supplied.
    bdy : pepsy.boundary_states.BdyMPS | None, default=None
        Optional pre-built boundary container for norm contractions.
    bdy_energy : pepsy.boundary_states.BdyMPS | None, default=None
        Optional pre-built boundary container for energy numerator
        contractions.
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
    _OPTIMIZE_KEYS = frozenset({
        "axes",
        "n",
        "n_round_trips",
        "chi",
        "optimizer",
        "optimizer_options",
        "env_n_iter",
        "progbar",
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
    _SOLVER_ALIASES = {
        "scipy": "scipy-lbfgs",
        "scipy_lbfgs": "scipy-lbfgs",
        "nlopt": "nlopt-lbfgs",
        "nlopt_lbfgs": "nlopt-lbfgs",
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
    def optimize_kwarg_names(cls):
        """Return supported keyword names for :meth:`set_optimize_kwargs`."""
        return tuple(sorted(cls._OPTIMIZE_KEYS))

    @classmethod
    def normalize_kwarg_names(cls):
        """Return supported keyword names for :meth:`normalize`."""
        return tuple(sorted(cls._NORMALIZE_KEYS))

    @classmethod
    def kwarg_guide(cls):
        """Return a compact guide of public kwargs and default solver options."""
        return {
            "normalize": cls.normalize_kwarg_names(),
            "optimize": cls.optimize_kwarg_names(),
            "optimizer_defaults": cls.default_solver_options(),
        }

    @classmethod
    def _merge_solver_options(cls, options):
        merged = cls.default_solver_options()
        if options:
            merged.update(dict(options))
        return merged

    @staticmethod
    def _to_python_scalar(value):
        """Convert backend scalar-like objects (torch/numpy) to python scalar."""
        obj = value
        if hasattr(obj, "detach"):
            obj = obj.detach()
        if hasattr(obj, "cpu"):
            obj = obj.cpu()
        if hasattr(obj, "item") and not isinstance(obj, (int, float, complex, bool)):
            try:
                obj = obj.item()
            except (ValueError, RuntimeError):
                pass
        return obj

    @staticmethod
    def _to_float_history(history):
        """Convert solver history entries into plain Python floats."""
        values = []
        for entry in history or ():
            values.append(float(EnergyOptimizer._to_python_scalar(entry)))
        return values

    @staticmethod
    def _detach_solver_params(params):
        """Drop autograd history without cloning parameter arrays."""
        out = {}
        for key, value in dict(params).items():
            if hasattr(value, "detach"):
                out[key] = value.detach()
            else:
                out[key] = value
        return out

    @staticmethod
    def _real_value(value):
        """Return real part while preserving backend tensor type when possible."""
        if hasattr(value, "real"):
            return value.real
        return complex(value).real

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

    @staticmethod
    def _ensure_pepo_shape(pepo, lx, ly):
        """Validate PEPO lattice shape matches state shape."""
        if getattr(pepo, "Lx", None) != lx or getattr(pepo, "Ly", None) != ly:
            raise ValueError(
                "pepo shape mismatch: "
                f"expected ({lx}, {ly}), got ({getattr(pepo, 'Lx', None)}, {getattr(pepo, 'Ly', None)})."
            )

    @classmethod
    def _collect_init_renormalize_kwargs(
        cls,
        *,
        renormalize_kwargs=None,
        n_iter=None,
        direction=None,
        max_separation=None,
        pbar=None,
        boundary_fidel=False,
    ):
        """Collect init-time normalize kwargs from explicit and mapping styles."""
        legacy = {
            "n_iter": n_iter,
            "direction": direction,
            "max_separation": max_separation,
            "pbar": pbar,
            "boundary_fidel": boundary_fidel,
        }
        legacy = {k: v for k, v in legacy.items() if v is not None}
        out = cls._pick_known_keys(legacy, cls._NORMALIZE_KEYS, warn_unknown=False)
        if renormalize_kwargs:
            out.update(cls._pick_known_keys(renormalize_kwargs, cls._NORMALIZE_KEYS))
        return out

    @staticmethod
    def _build_hamiltonian_ket(state, pepo):
        """Build ``H|psi>`` network with physical output indices named ``k...``."""
        ket_state = state.copy()
        pepo_op = pepo.copy()

        # Route PEPO input legs through fresh physical IDs to avoid
        # collisions, then map PEPO output b... -> k... on the result.
        reindex_k_rand = {
            idx: qtn.rand_uuid()
            for idx in ket_state.outer_inds()
            if isinstance(idx, str) and idx.startswith("k")
        }
        if reindex_k_rand:
            ket_state.reindex_(reindex_k_rand)
            pepo_map = {
                idx: rand_idx
                for idx, rand_idx in reindex_k_rand.items()
                if idx in pepo_op.ind_map
            }
            if pepo_map:
                pepo_op.reindex_(pepo_map)

        ket_h = pepo_op | ket_state
        reindex_bk = {
            idx: f"k{idx[1:]}"
            for idx in ket_h.outer_inds()
            if isinstance(idx, str) and idx.startswith("b")
        }
        if reindex_bk:
            ket_h.reindex_(reindex_bk)
        ket_h.add_tag("PEPO_PEPS")
        return ket_h

    @staticmethod
    def _build_boundary_pair(state, pepo, *, chi, single_layer=False):
        """Construct norm and energy boundary MPS containers."""
        _, norm_tn = prepare_boundary_inputs(ket=state, bra=None)
        ket_h = EnergyOptimizer._build_hamiltonian_ket(state, pepo)
        _, energy_tn = prepare_boundary_inputs(ket=ket_h, bra=state)

        bdy = BdyMPS(
            tn_double=norm_tn,
            chi=chi,
            single_layer=single_layer,
        )
        bdy_energy = BdyMPS(
            tn_double=energy_tn,
            chi=chi,
            single_layer=single_layer,
        )
        return bdy, bdy_energy

    def __init__(
        self,
        state,
        pepo,
        *,
        chi=None,
        bdy=None,
        bdy_energy=None,
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
        boundary_fidel: bool = False,
    ):
        if (bdy is None) ^ (bdy_energy is None):
            raise ValueError("Provide both bdy and bdy_energy together, or neither.")

        self.state = state
        self.Lx, self.Ly = self._infer_shape(self.state)

        pepo_use = pepo.copy()
        pepo_use.add_tag("pepo")
        self.pepo = pepo_use
        self._ensure_pepo_shape(self.pepo, self.Lx, self.Ly)

        if bdy is None and bdy_energy is None:
            if chi is None:
                raise ValueError(
                    "Provide chi when bdy and bdy_energy are not supplied."
                )
            bdy, bdy_energy = self._build_boundary_pair(
                self.state,
                self.pepo,
                chi=chi,
                single_layer=single_layer,
            )

        for name, obj in (("bdy", bdy), ("bdy_energy", bdy_energy)):
            if not hasattr(obj, "mps_b"):
                raise TypeError(f"{name} must expose attribute 'mps_b'.")

        self.bdy = bdy
        self.bdy_energy = bdy_energy
        self.opt = opt
        self.dmrg_run = dmrg_run

        if normalize_kwargs is None:
            self.normalize_kwargs = {}
        else:
            self.normalize_kwargs = self._pick_known_keys(normalize_kwargs, self._NORMALIZE_KEYS)
        self.optimize_kwargs = self._pick_known_keys(optimize_kwargs, self._OPTIMIZE_KEYS)

        self._reset_run_traces()
        self.pepo_peps = self._build_hamiltonian_ket(self.state, self.pepo)

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

    def _reset_run_traces(self):
        """Reset lightweight traces collected during :meth:`optimize_global`."""
        self.energy_trace = []
        self.norm_trace = []

    def _ensure_boundary_chi(self, chi):
        """Retune both stored boundary objects to at least ``chi``."""
        if chi is None:
            return
        if not isinstance(chi, int):
            raise TypeError("chi must be an integer")
        if chi < 1:
            raise ValueError("chi must be >= 1")

        for obj in (self.bdy, self.bdy_energy):
            if getattr(obj, "chi", 0) < chi:
                obj.expand_bnd(chi, inplace=True)

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
        """Expand stored boundaries to ``chi`` and optionally renormalize state."""
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

    @staticmethod
    def _axis_n_for_shape(axis, *, lx, ly):
        if axis == "y":
            return ly
        if axis == "x":
            return lx
        raise ValueError("axis must be 'x' or 'y'")

    def _axis_n(self, axis):
        return self._axis_n_for_shape(axis, lx=self.Lx, ly=self.Ly)

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
        _, norm_tn = prepare_boundary_inputs(ket=self.state, bra=None)
        self.pepo_peps = self._build_hamiltonian_ket(self.state, self.pepo)
        _, energy_tn = prepare_boundary_inputs(ket=self.pepo_peps, bra=self.state)
        return norm_tn, energy_tn

    def _refresh_energy_boundary(self, *, chi=None, single_layer=False):
        """Rebuild energy boundary container to match current normalized state."""
        self.pepo_peps = self._build_hamiltonian_ket(self.state, self.pepo)
        _, energy_tn = prepare_boundary_inputs(ket=self.pepo_peps, bra=self.state)
        if chi is None:
            chi = getattr(self.bdy_energy, "chi", None)
        if chi is None:
            chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")
        self.bdy_energy = BdyMPS(
            tn_double=energy_tn,
            chi=chi,
            single_layer=single_layer,
        )

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
        solver = EnergyOptimizer._SOLVER_ALIASES.get(solver, solver)
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

    def _make_comp_pair(self, norm_tn, energy_tn):
        comp_norm = CompBdy(norm_tn, self.bdy.mps_b, opt=self.opt, dmrg_run=self.dmrg_run)
        comp_energy = CompBdy(energy_tn, self.bdy_energy.mps_b, opt=self.opt, dmrg_run=self.dmrg_run)
        return comp_norm, comp_energy

    @staticmethod
    def _set_comp_norms(comp_norm, comp_energy, *, norm_tn, energy_tn):
        comp_norm.norm = norm_tn
        comp_energy.norm = energy_tn

    def _refresh_right_boundaries_once(self, axis, *, env_n_iter=10):
        norm_tn, energy_tn = self._prepare_current_double_layers()
        comp_norm, comp_energy = self._make_comp_pair(norm_tn, energy_tn)

        comp_energy.move_bdy(
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
        comp_energy,
        env_n_iter=10,
        boundary_fidel=False,
    ):
        n = self._axis_n(axis)
        if side == "left" and index <= 0:
            return {"norm": None, "energy": None}
        if side == "right" and index >= n - 1:
            return {"norm": None, "energy": None}

        pos = (index - 1) if side == "left" else (n - 2 - index)
        direction = self._boundary_direction(axis, side)
        for comp in (comp_energy, comp_norm):
            comp.move_step_bdy(
                pos=pos,
                n_iter=env_n_iter,
                pbar=False,
                direction=direction,
                boundary_fidel=boundary_fidel,
            )

        energy_fidelity = None
        norm_fidelity = None
        if boundary_fidel:
            if comp_energy.fidel:
                energy_fidelity = float(complex(comp_energy.fidel[-1]).real)
            if comp_norm.fidel:
                norm_fidelity = float(complex(comp_norm.fidel[-1]).real)

        return {"norm": norm_fidelity, "energy": energy_fidelity}

    def _optimize_axis_slice_with_current_env(
        self,
        index,
        *,
        axis,
        solver="adam",
        solver_options=None,
    ):
        axis_tag = self._axis_tag(axis)
        right_key, left_key = self._boundary_keys_for_index(index, axis)

        slice_state = self.state.select([f"{axis_tag}{index}"], "any")
        pepo_slice = self.pepo.select([f"{axis_tag}{index}"], "any")
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
            norm_net = self._attach_boundaries(
                local | bra_norm,
                self.bdy.mps_b,
                right_key=right_key,
                left_key=left_key,
            )

            local_h = self._build_hamiltonian_ket(local, pepo_slice)

            bra_energy = local.conj()
            bra_energy.reindex_(
                {
                    idx: f"{idx}_*"
                    for idx in bra_energy.ind_map
                    if not (isinstance(idx, str) and _PHYS_IND_PATTERN.fullmatch(idx))
                }
            )
            energy_net = self._attach_boundaries(
                local_h | bra_energy,
                self.bdy_energy.mps_b,
                right_key=right_key,
                left_key=left_key,
            )

            energy_val = self._real_value(energy_net.contract(all, optimize=self.opt))
            # Keep local denominator strictly non-negative for numerical stability.
            norm_val = abs(norm_net.contract(all, optimize=self.opt))
            # Keep ratio effectively scale-invariant under global state rescaling.
            return energy_val / (norm_val + 1e-30)

        initial_loss = float(self._to_python_scalar(loss_fn(params_init)))

        params_opt, history = self._optimize_packed_params(
            params_init,
            loss_fn,
            solver=solver,
            solver_options=solver_options,
        )
        history_values = self._to_float_history(history)
        final_loss = float(self._to_python_scalar(loss_fn(params_opt)))

        params_opt = self._detach_solver_params(params_opt)
        self._apply_slice_update(index, params_opt, skeleton, axis)

        run_info = {
            "axis": axis,
            "index": index,
            "right_key": right_key,
            "left_key": left_key,
            "energy_initial": initial_loss,
            "energy_final": final_loss,
            "history": history_values,
        }
        # Keep key for notebook compatibility without extra exact contractions.
        run_info["state_norm_after"] = None
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
        env_n_iter=10,
        run_callback=None,
        boundary_fidel=False,
    ):
        """Run a single forward or backward half-sweep over *indices*."""
        runs = []
        comp_norm = None
        comp_energy = None

        for index in indices:
            norm_tn, energy_tn = self._prepare_current_double_layers()
            if comp_norm is None:
                comp_norm, comp_energy = self._make_comp_pair(norm_tn, energy_tn)
            else:
                self._set_comp_norms(
                    comp_norm,
                    comp_energy,
                    norm_tn=norm_tn,
                    energy_tn=energy_tn,
                )

            t0 = time.perf_counter()
            boundary_fidelity = self._advance_boundary_one_step(
                index,
                side=update_side,
                axis=axis,
                comp_norm=comp_norm,
                comp_energy=comp_energy,
                env_n_iter=env_n_iter,
                boundary_fidel=boundary_fidel,
            )
            t_bdy = time.perf_counter() - t0

            t0 = time.perf_counter()
            run_info = self._optimize_axis_slice_with_current_env(
                index,
                axis=axis,
                solver=solver,
                solver_options=solver_options,
            )
            t_opt = time.perf_counter() - t0

            run_info["sweep"] = sweep_name
            run_info["time_boundary"] = t_bdy
            run_info["time_optimize"] = t_opt
            run_info["boundary_fidelity_norm"] = boundary_fidelity.get("norm")
            run_info["boundary_fidelity_energy"] = boundary_fidelity.get("energy")
            runs.append(run_info)
            if run_callback is not None:
                run_callback(run_info)

        return runs

    def _normalize_state(self, env_n_iter=10):
        # Start from user/default normalize kwargs, but keep sweep-time
        # normalization deterministic for stability.
        opts = self._pick_known_keys(
            getattr(self, "normalize_kwargs", {}),
            self._NORMALIZE_KEYS,
            warn_unknown=False,
        )
        opts.setdefault("direction", "y")
        opts.setdefault("max_separation", 1)
        opts["n_iter"] = env_n_iter
        opts["pbar"] = False
        opts["boundary_fidel"] = False
        return self.normalize(**opts)

    def normalize(self, state=None, **kwargs):
        """Normalize PEPS in place and refresh energy boundaries."""
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
        """Replace current state, rebuild boundaries, and optionally normalize."""
        self.state = state
        self.Lx, self.Ly = self._infer_shape(self.state)

        if chi is None:
            chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")

        self.bdy, self.bdy_energy = self._build_boundary_pair(
            self.state,
            self.pepo,
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

    def set_pepo(
        self,
        pepo,
        *,
        chi=None,
        single_layer=False,
    ):
        """Replace Hamiltonian PEPO and rebuild energy boundary immediately."""
        pepo_use = pepo.copy()
        pepo_use.add_tag("pepo")
        self.pepo = pepo_use
        self._ensure_pepo_shape(self.pepo, self.Lx, self.Ly)

        if chi is None:
            chi = getattr(self.bdy_energy, "chi", None)
        if chi is None:
            chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi when current boundary chi is unavailable.")

        self._refresh_energy_boundary(chi=chi, single_layer=single_layer)

    def _collect_axis_run_traces(self, axis_runs, *, cycle, axis):
        """Collect per-step traces from axis run records."""
        if not axis_runs:
            return
        _ = cycle, axis
        for run_info in axis_runs:
            val = run_info.get("energy_final")
            if val is None:
                self.energy_trace.append(float("nan"))
            else:
                self.energy_trace.append(float(val))

    def energy(
        self,
        *,
        opt=None,
        n_iter=5,
        direction="y",
        max_separation=0,
        pbar=False,
        boundary_fidel=False,
    ):
        """Compute boundary-estimated global energy ``<psi|H|psi>/<psi|psi>``."""
        norm_tn, energy_tn = self._prepare_current_double_layers()
        opt_use = self.opt if opt is None else opt

        norm_cost = ContractBoundary(
            norm=norm_tn,
            mps_boundaries=self.bdy.mps_b,
            opt=opt_use,
            dmrg_run=self.dmrg_run,
            n_iter=n_iter,
            pbar=pbar,
            direction=direction,
            max_separation=max_separation,
            boundary_fidel=boundary_fidel,
        ).cost
        energy_cost = ContractBoundary(
            norm=energy_tn,
            mps_boundaries=self.bdy_energy.mps_b,
            opt=opt_use,
            dmrg_run=self.dmrg_run,
            n_iter=n_iter,
            pbar=pbar,
            direction=direction,
            max_separation=max_separation,
            boundary_fidel=boundary_fidel,
        ).cost

        denom = complex(self._to_python_scalar(norm_cost))
        numer = complex(self._to_python_scalar(energy_cost))
        if abs(denom) == 0:
            raise ZeroDivisionError("Norm is zero; cannot compute energy ratio.")
        return float((numer / denom).real)

    def optimize_axis(
        self,
        axis,
        *,
        n_round_trips=1,
        solver="scipy",
        solver_options=None,
        env_n_iter=10,
        run_callback=None,
        boundary_fidel=False,
        renormalize=True,
    ):
        """Run one axis with forward + round-trip sweeps."""
        n = self._axis_n(axis)
        resolved_solver = self._resolve_user_solver(solver)
        all_runs = []

        self.bdy.normalize()
        self.bdy_energy.normalize()

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
        env_n_iter=10,
        pbar=True,
        boundary_fidel=False,
        renormalize=True,
    ):
        """Run alternating axis sweeps and return :class:`EnergyResult`."""
        self._ensure_boundary_chi(chi)
        self._reset_run_traces()

        energy_before = self.energy(
            n_iter=env_n_iter,
            pbar=False,
            boundary_fidel=False,
        )

        boundary_fidel = bool(boundary_fidel)
        all_runs = []
        axis_seq = list(axes)

        def _steps_for_axis(axis_name):
            n_axis = self._axis_n(axis_name)
            return n_axis + (2 * n_round_trips * max(n_axis - 1, 0))

        total_steps = n_cycles * sum(_steps_for_axis(axis_name) for axis_name in axis_seq)
        global_progress = None
        if pbar:
            global_progress = tqdm(
                total=total_steps,
                desc="energy_dmrg:",
                leave=True,
                position=0,
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="cyan",
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
                    local_energy = run_info.get("energy_final")
                    if local_energy is not None:
                        postfix["E_loc"] = f"{float(local_energy):.6f}"
                    axis_name_ = run_info.get("axis")
                    sweep_name = run_info.get("sweep")
                    index = run_info.get("index")
                    if axis_name_ is not None and sweep_name is not None and index is not None:
                        short = "fwd" if sweep_name == "forward" else "bwd"
                        postfix["slice"] = f"{axis_name_}_{short}_{index}"
                    global_progress.set_postfix(postfix)

                axis_runs = self.optimize_axis(
                    axis,
                    n_round_trips=n_round_trips,
                    solver=solver,
                    solver_options=solver_options,
                    env_n_iter=env_n_iter,
                    run_callback=_on_run,
                    boundary_fidel=boundary_fidel,
                    # Apply state renormalization once at global-run end to
                    # avoid compounding boundary-estimation noise.
                    renormalize=False,
                )
                all_runs.extend(axis_runs)
                self._collect_axis_run_traces(axis_runs, cycle=cyc + 1, axis=axis)

        if global_progress is not None:
            global_progress.close()

        energy_after = self.energy(
            n_iter=env_n_iter,
            pbar=False,
            boundary_fidel=False,
        )

        if renormalize:
            old_norm = self._normalize_state(env_n_iter=env_n_iter)
            self.norm_trace.append(float(abs(complex(old_norm))))

        return EnergyResult(
            runs=all_runs,
            energy_before=energy_before,
            energy_after=energy_after,
            energy=list(self.energy_trace),
            norm_trace=list(self.norm_trace),
        )

    def set_normalize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`normalize`."""
        if not hasattr(self, "normalize_kwargs"):
            self.normalize_kwargs = {}
        self.normalize_kwargs.update(self._pick_known_keys(kwargs, self._NORMALIZE_KEYS))
        return self

    def set_optimize_kwargs(self, **kwargs):
        """Update stored defaults for :meth:`optimize_global`."""
        if not hasattr(self, "optimize_kwargs"):
            self.optimize_kwargs = {}
        self.optimize_kwargs.update(self._pick_known_keys(kwargs, self._OPTIMIZE_KEYS))
        return self

    def run(
        self,
        *,
        n=None,
        chi=None,
        progbar=None,
        renormalize=None,
    ):
        """High-level wrapper around :meth:`optimize_global`."""
        opts = dict(getattr(self, "optimize_kwargs", {}))
        if n is not None:
            opts["n"] = n
        if chi is not None:
            opts["chi"] = chi
        if progbar is not None:
            opts["progbar"] = progbar
        if renormalize is not None:
            opts["renormalize"] = renormalize

        return self.optimize_global(
            axes=opts.get("axes", ("y", "x")),
            n_cycles=opts.get("n", 1),
            n_round_trips=opts.get("n_round_trips", 1),
            chi=opts.get("chi"),
            solver=opts.get("optimizer", "scipy"),
            solver_options=opts.get("optimizer_options"),
            env_n_iter=opts.get("env_n_iter", 10),
            pbar=opts.get("progbar", True),
            boundary_fidel=opts.get("boundary_fidel", False),
            renormalize=opts.get("renormalize", True),
        )
