"""Energy-based PEPS sweep optimizer with axis-alternating boundary updates."""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import quimb.tensor as qtn
from tqdm.auto import tqdm

from ..boundary.metrics import contract_boundary, build_bra_ket, normalize
from ..boundary.states import BdyMPS
from ..boundary.sweeps import CompBdy
from ..tensors.core import tns_align
from ..solvers.gradient import GradientOptimizer, SUPPORTED_SOLVERS
from ..tensors.validation import _PHYS_IND_PATTERN, _TAG_X, _TAG_Y

__all__ = ["EnergyOptimizer"]


class EnergyOptimizer:  # pylint: disable=too-many-instance-attributes
    """Optimize PEPS slices to minimize energy ``<psi|H|psi>/<psi|psi>``.

    Parameters
    ----------
    state : qtn.TensorNetwork
        Trainable PEPS-like state.
    pepo : qtn.TensorNetwork
        Hamiltonian PEPO with physical indices ``k...`` (upper) and ``b...``
        (lower) matching the lattice shape of ``state``. A copy is stored and
        tagged with ``"pepo"``.
    chi : int | None, default=None
        Boundary bond dimension used when ``bdy``/``bdy_energy`` are not
        supplied.
    bdy : pepsy.boundary.states.BdyMPS | None, default=None
        Optional pre-built boundary container for norm contractions.
    bdy_energy : pepsy.boundary.states.BdyMPS | None, default=None
        Optional pre-built boundary container for energy numerator
        contractions.
    contraction_opt : object | str, default="auto-hq"
        Contraction optimizer.
    fit_mode : {"eff", "global"}, default="eff"
        Backend mode passed to :class:`pepsy.boundary.sweeps.CompBdy`.
    """

    _NORMALIZE_KEYS = frozenset({
        "contraction_opt",
        "n_iter",
        "direction",
        "max_separation",
        "progress",
        "track_boundary_fidelity",
    })
    _OPTIMIZE_KEYS = frozenset({
        "axes",
        "n_cycles",
        "n_round_trips",
        "chi",
        "optimizer",
        "optimizer_options",
        "env_n_iter",
        "progress",
        "track_boundary_fidelity",
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
        """Drop autograd history and clone parameter arrays."""
        out = {}
        for key, value in dict(params).items():
            if hasattr(value, "detach"):
                out[key] = value.detach().clone()
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
        progress=None,
        track_boundary_fidelity=None,
    ):
        """Collect init-time normalize kwargs from explicit and mapping styles."""
        legacy = {
            "n_iter": n_iter,
            "direction": direction,
            "max_separation": max_separation,
            "progress": progress,
            "track_boundary_fidelity": track_boundary_fidelity,
        }
        legacy = {k: v for k, v in legacy.items() if v is not None}
        out = cls._pick_known_keys(legacy, cls._NORMALIZE_KEYS, warn_unknown=False)
        if renormalize_kwargs:
            out.update(cls._pick_known_keys(renormalize_kwargs, cls._NORMALIZE_KEYS))
        return out

    @staticmethod
    def _build_hamiltonian_ket(state, pepo):
        """Build ``H|psi>`` network with physical output indices named ``k...``."""
        ket_h = tns_align(state, pepo)
        ket_h.add_tag("PEPO_PEPS")
        return ket_h

    @staticmethod
    def _build_boundary_pair(state, pepo, *, chi, single_layer=False):
        """Construct norm and energy boundary MPS containers."""
        _, norm_tn = build_bra_ket(ket=state, bra=None)
        ket_h = EnergyOptimizer._build_hamiltonian_ket(state, pepo)
        _, energy_tn = build_bra_ket(ket=ket_h, bra=state)

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

    @staticmethod
    def _resolve_boundaries(
        state,
        pepo,
        *,
        chi,
        bdy,
        bdy_energy,
        single_layer=False,
    ):
        """Resolve boundary containers, building them when not supplied."""
        if bdy is not None or bdy_energy is not None:
            return bdy, bdy_energy
        if chi is None:
            raise ValueError(
                "Provide chi when bdy and bdy_energy are not supplied."
            )
        return EnergyOptimizer._build_boundary_pair(
            state,
            pepo,
            chi=chi,
            single_layer=single_layer,
        )

    @staticmethod
    def _validate_boundaries(bdy, bdy_energy):
        """Validate both boundary containers expose boundary MPS dict."""
        for name, obj in (("bdy", bdy), ("bdy_energy", bdy_energy)):
            if not hasattr(obj, "mps_b"):
                raise TypeError(f"{name} must expose attribute 'mps_b'.")

    def __init__(
        self,
        state,
        pepo,
        *,
        chi=None,
        bdy=None,
        bdy_energy=None,
        contraction_opt="auto-hq",
        fit_mode="eff",
        single_layer=False,
        normalize_kwargs: Mapping[str, Any] | None = None,
        optimize_kwargs: Mapping[str, Any] | None = None,
        renormalize_state=False,
        renormalize_kwargs: Mapping[str, Any] | None = None,
        n_iter: int | None = None,
        direction: str | None = None,
        max_separation: int | None = None,
        progress: bool | None = None,
        track_boundary_fidelity: bool | None = None,
    ):
        if (bdy is None) ^ (bdy_energy is None):
            raise ValueError("Provide both bdy and bdy_energy together, or neither.")

        self.state = state
        self.Lx, self.Ly = self._infer_shape(self.state)

        pepo_use = pepo.copy()
        pepo_use.add_tag("pepo")
        self.pepo = pepo_use
        self._ensure_pepo_shape(self.pepo, self.Lx, self.Ly)

        bdy, bdy_energy = self._resolve_boundaries(
            self.state,
            self.pepo,
            chi=chi,
            bdy=bdy,
            bdy_energy=bdy_energy,
            single_layer=single_layer,
        )
        self._validate_boundaries(bdy, bdy_energy)

        self.bdy = bdy
        self.bdy_energy = bdy_energy
        self.contraction_opt = contraction_opt
        self.fit_mode = fit_mode

        if normalize_kwargs is None:
            self.normalize_kwargs = {}
        else:
            self.normalize_kwargs = self._pick_known_keys(normalize_kwargs, self._NORMALIZE_KEYS)
        self.optimize_kwargs = self._pick_known_keys(optimize_kwargs, self._OPTIMIZE_KEYS)

        self._reset_run_traces()

        init_renormalize_kwargs = self._collect_init_renormalize_kwargs(
            renormalize_kwargs=renormalize_kwargs,
            n_iter=n_iter,
            direction=direction,
            max_separation=max_separation,
            progress=progress,
            track_boundary_fidelity=track_boundary_fidelity,
        )
        if renormalize_state:
            self.normalize(**init_renormalize_kwargs)

    def _reset_run_traces(self):
        """Reset lightweight traces collected during :meth:`optimize_global`."""
        self.energy_trace = []
        # Expose per-step loss (energy) via .loss for a consistent interface
        # shared with SweepOptimizer.
        self.loss = self.energy_trace
        self.norm_trace = []
        self.fidels = []

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
        progress=False,
        track_boundary_fidelity=False,
    ):
        """Expand stored boundaries to ``chi`` and optionally renormalize state."""
        self._ensure_boundary_chi(chi)
        if normalize_state:
            self.normalize(
                n_iter=n_iter,
                direction=direction,
                max_separation=max_separation,
                progress=progress,
                track_boundary_fidelity=track_boundary_fidelity,
            )
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
        _, norm_tn = build_bra_ket(ket=self.state, bra=None)
        self.pepo_peps = self._build_hamiltonian_ket(self.state, self.pepo)
        _, energy_tn = build_bra_ket(ket=self.pepo_peps, bra=self.state)
        return norm_tn, energy_tn

    def _refresh_energy_boundary(self, *, chi=None, single_layer=False):
        """Rebuild energy boundary container to match current normalized state."""
        self.pepo_peps = self._build_hamiltonian_ket(self.state, self.pepo)
        _, energy_tn = build_bra_ket(ket=self.pepo_peps, bra=self.state)
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
    def _bra_with_reindexed_inner(local_tn):
        """Return ``local_tn.conj()`` with non-physical inner indices renamed."""
        bra = local_tn.conj()
        bra.reindex_(
            {
                idx: f"{idx}_*"
                for idx in bra.ind_map
                if not (isinstance(idx, str) and _PHYS_IND_PATTERN.fullmatch(idx))
            }
        )
        return bra

    def _estimate_slice_contraction_metrics(
        self,
        *,
        params_init,
        skeleton,
        pepo_slice,
        right_key,
        left_key,
    ):
        """Estimate FLOP/peak complexity for local norm and energy networks."""
        local0 = qtn.unpack(params_init, skeleton)
        bra_norm0 = self._bra_with_reindexed_inner(local0)
        norm_net0 = self._attach_boundaries(
            local0 | bra_norm0,
            self.bdy.mps_b,
            right_key=right_key,
            left_key=left_key,
        )

        local_h0 = self._build_hamiltonian_ket(local0, pepo_slice)
        bra_energy0 = self._bra_with_reindexed_inner(local0)
        energy_net0 = self._attach_boundaries(
            local_h0 | bra_energy0,
            self.bdy_energy.mps_b,
            right_key=right_key,
            left_key=left_key,
        )

        tree_norm = norm_net0.contraction_tree(self.contraction_opt)
        tree_energy = energy_net0.contraction_tree(self.contraction_opt)
        flops_norm = float(tree_norm.contraction_cost(log=10))
        peak_norm = float(tree_norm.peak_size(log=2))
        flops_energy = float(tree_energy.contraction_cost(log=10))
        peak_energy = float(tree_energy.peak_size(log=2))

        return {
            "flops_norm": flops_norm,
            "peak_norm": peak_norm,
            "flops_energy": flops_energy,
            "peak_energy": peak_energy,
            # Keep overlap-style key for downstream UI parity.
            "flops_overlap": flops_energy,
            "peak_overlap": peak_energy,
        }

    @staticmethod
    def _resolve_user_solver(solver):
        """Validate solver names and emit practical warnings."""
        if not isinstance(solver, str):
            raise TypeError("solver must be a string")
        solver = solver.strip().lower()

        if solver not in SUPPORTED_SOLVERS:
            supported = ", ".join(SUPPORTED_SOLVERS)
            raise ValueError(f"Unsupported solver={solver!r}. Supported solvers: {supported}")

        if solver in {"nlopt", "fd-nlopt"}:
            warnings.warn(
                f"solver={solver!r} uses NLopt on CPU float64 parameter vectors. "
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
        runner = GradientOptimizer(
            solver=solver,
            n_steps=n_steps,
            options=opts,
            progress=False,
            verbose=False,
        )
        result = runner.run(params_init=params_init, loss_fn=loss_fn)
        return result.params, result.history

    def _apply_slice_update(self, index, params_opt, skeleton, axis):
        tn_opt = qtn.unpack(params_opt, skeleton)
        tn_opt.balance_bonds_()

        for tag in self._site_tensor_tags(axis, index):
            self.state[tag].modify(data=tn_opt[tag].data)
        return tn_opt

    def _make_comp_pair(self, norm_tn, energy_tn):
        comp_norm = CompBdy(norm_tn, self.bdy.mps_b, contraction_opt=self.contraction_opt, fit_mode=self.fit_mode)
        comp_energy = CompBdy(energy_tn, self.bdy_energy.mps_b, contraction_opt=self.contraction_opt, fit_mode=self.fit_mode)
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
            progress=False,
            direction=self._boundary_direction(axis, "right"),
            track_boundary_fidelity=False,
        )
        comp_norm.move_bdy(
            n_iter=env_n_iter,
            progress=False,
            direction=self._boundary_direction(axis, "right"),
            track_boundary_fidelity=False,
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
        track_boundary_fidelity=False,
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
                progress=False,
                direction=direction,
                track_boundary_fidelity=track_boundary_fidelity,
            )

        energy_fidelity = None
        norm_fidelity = None
        if track_boundary_fidelity:
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
        solver="torch-adam",
        solver_options=None,
    ):
        axis_tag = self._axis_tag(axis)
        right_key, left_key = self._boundary_keys_for_index(index, axis)
        slice_state = self.state.select([f"{axis_tag}{index}"], "any")
        pepo_slice = self.pepo.select([f"{axis_tag}{index}"], "any")
        params_init, skeleton = qtn.pack(slice_state)
        metrics = self._estimate_slice_contraction_metrics(
            params_init=params_init,
            skeleton=skeleton,
            pepo_slice=pepo_slice,
            right_key=right_key,
            left_key=left_key,
        )

        def loss_fn(params_in):
            local = qtn.unpack(params_in, skeleton)

            bra_norm = self._bra_with_reindexed_inner(local)
            norm_net = self._attach_boundaries(
                local | bra_norm,
                self.bdy.mps_b,
                right_key=right_key,
                left_key=left_key,
            )
            local_h = self._build_hamiltonian_ket(local, pepo_slice)
            bra_energy = self._bra_with_reindexed_inner(local)
            energy_net = self._attach_boundaries(
                local_h | bra_energy,
                self.bdy_energy.mps_b,
                right_key=right_key,
                left_key=left_key,
            )

            energy_val = self._real_value(energy_net.contract(all, optimize=self.contraction_opt))
            # Keep local denominator strictly non-negative for numerical stability.
            norm_val = abs(norm_net.contract(all, optimize=self.contraction_opt))
            # Keep ratio effectively scale-invariant under global state rescaling.
            return energy_val / (norm_val + 1e-12)

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

        return {
            "axis": axis,
            "index": index,
            "energy_initial": initial_loss,
            "energy_final": final_loss,
            "history": history_values,
            **metrics,
        }

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
        track_boundary_fidelity=False,
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
                track_boundary_fidelity=track_boundary_fidelity,
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
        opts["progress"] = False
        opts["track_boundary_fidelity"] = False
        return self.normalize(**opts)

    def normalize(self, state=None, **kwargs):
        """Normalize a PEPS state with boundary contraction.

        When ``state is self.state``, the stored norm boundary ``self.bdy`` is
        reused. External states are normalized using ``chi=self.bdy.chi``.
        """
        state = self.state if state is None else state
        opts = self._merge_opts(getattr(self, "normalize_kwargs", {}), kwargs)
        contraction_opt = opts.get("contraction_opt", self.contraction_opt)
        n_iter = opts.get("n_iter", 10)
        direction = opts.get("direction", "y")
        max_separation = opts.get("max_separation", 1)
        progress = opts.get("progress", False)
        track_boundary_fidelity = opts.get("track_boundary_fidelity", False)

        if state is self.state:
            return normalize(
                self.state,
                bdy=self.bdy,
                contraction_opt=contraction_opt,
                max_separation=max_separation,
                n_iter=n_iter,
                direction=direction,
                progress=progress,
                track_boundary_fidelity=track_boundary_fidelity,
                fit_mode=self.fit_mode,
            )

        chi = getattr(self.bdy, "chi", None)
        if chi is None:
            raise ValueError("Provide chi via optimizer boundaries before normalizing external state.")
        return normalize(
            state,
            chi=chi,
            contraction_opt=contraction_opt,
            max_separation=max_separation,
            n_iter=n_iter,
            direction=direction,
            progress=progress,
            track_boundary_fidelity=track_boundary_fidelity,
            fit_mode=self.fit_mode,
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
        progress=False,
        track_boundary_fidelity=False,
    ):
        """Replace current state, rebuild boundaries, and optionally normalize.

        Returns
        -------
        complex | float | None
            Old norm returned by :meth:`normalize` when ``normalize_state`` is
            True, else ``None``.
        """
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
                progress=progress,
                track_boundary_fidelity=track_boundary_fidelity,
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

    def _collect_axis_run_traces(self, axis_runs):
        """Collect per-step traces from axis run records."""
        if not axis_runs:
            return
        for run_info in axis_runs:
            val = run_info.get("energy_final")
            if val is None:
                self.energy_trace.append(float("nan"))
            else:
                self.energy_trace.append(float(val))
            self.fidels.append(
                {
                    "norm": run_info.get("boundary_fidelity_norm"),
                    "energy": run_info.get("boundary_fidelity_energy"),
                }
            )

    def energy(
        self,
        *,
        contraction_opt=None,
        n_iter=5,
        direction="y",
        max_separation=0,
        progress=False,
        track_boundary_fidelity=False,
        return_info=False,
    ):
        """Compute boundary-estimated global energy ``<psi|H|psi>/<psi|psi>``.

        Parameters
        ----------
        contraction_opt : str | object | None, default=None
            Contraction optimizer override. Uses ``self.contraction_opt`` when
            ``None``.
        n_iter : int, default=5
            Local boundary-fit iterations.
        direction : {"x", "y"}, default="y"
            Boundary sweep direction.
        max_separation : int, default=0
            Boundary sweep separation mode.
        progress : bool, default=False
            Show boundary sweep progress bars.
        track_boundary_fidelity : bool, default=False
            Track per-step boundary fidelities for norm and energy networks.
        return_info : bool, default=False
            If ``True``, return a dictionary with energy and fidelity traces.

        Returns
        -------
        float | dict
            Energy scalar by default. If ``return_info=True``, returns a dict
            with keys:
            ``energy``, ``norm_cost``, ``energy_cost``,
            ``boundary_fidelity_norm``, ``boundary_fidelity_energy``,
            ``boundary_fidelity_norm_last``, ``boundary_fidelity_energy_last``.
        """
        norm_tn, energy_tn = self._prepare_current_double_layers()
        opt_use = self.contraction_opt if contraction_opt is None else contraction_opt

        norm_result = contract_boundary(
            norm=norm_tn,
            bdy=self.bdy,
            contraction_opt=opt_use,
            fit_mode=self.fit_mode,
            n_iter=n_iter,
            progress=progress,
            direction=direction,
            max_separation=max_separation,
            track_boundary_fidelity=track_boundary_fidelity,
        )
        energy_result = contract_boundary(
            norm=energy_tn,
            bdy=self.bdy_energy,
            contraction_opt=opt_use,
            fit_mode=self.fit_mode,
            n_iter=n_iter,
            progress=progress,
            direction=direction,
            max_separation=max_separation,
            track_boundary_fidelity=track_boundary_fidelity,
        )

        denom = complex(self._to_python_scalar(norm_result.cost))
        numer = complex(self._to_python_scalar(energy_result.cost))
        if abs(denom) == 0:
            raise ZeroDivisionError("Norm is zero; cannot compute energy ratio.")
        energy_value = float((numer / denom).real)

        if not return_info:
            return energy_value

        norm_fidel = [float(complex(v).real) for v in norm_result.fidel]
        energy_fidel = [float(complex(v).real) for v in energy_result.fidel]
        return {
            "energy": energy_value,
            "norm_cost": denom,
            "energy_cost": numer,
            "boundary_fidelity_norm": norm_fidel,
            "boundary_fidelity_energy": energy_fidel,
            "boundary_fidelity_norm_last": norm_fidel[-1] if norm_fidel else None,
            "boundary_fidelity_energy_last": energy_fidel[-1] if energy_fidel else None,
        }

    def optimize_axis(
        self,
        axis,
        *,
        n_round_trips=1,
        solver="scipy",
        solver_options=None,
        env_n_iter=10,
        run_callback=None,
        track_boundary_fidelity=False,
        renormalize=True,
    ):
        """Run one axis with forward + round-trip sweeps.

        Parameters
        ----------
        axis : {"x", "y"}
            Axis to sweep.
        n_round_trips : int, default=1
            Number of backward+forward round-trips after the initial forward pass.
        solver : str, default="scipy"
            Local solver name passed to gradient backend.
        solver_options : dict | None, default=None
            Extra solver controls.
        env_n_iter : int, default=10
            Local boundary-fit iterations per boundary move.
        track_boundary_fidelity : bool, default=False
            Enable per-step boundary fidelity sampling.
        renormalize : bool, default=True
            Normalize state at axis start and append to ``norm_trace``.
        """
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
            track_boundary_fidelity=track_boundary_fidelity,
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
        progress=True,
        track_boundary_fidelity=False,
        renormalize=True,
    ):
        """Run alternating axis sweeps and return :class:`EnergyResult`.

        Parameters
        ----------
        axes : sequence[{"x", "y"}], default=("y", "x")
            Axis order used for each cycle.
        n_cycles : int, default=1
            Number of global cycles.
        n_round_trips : int, default=1
            Number of backward+forward round-trips per axis.
        chi : int | None, default=None
            Optional boundary bond dimension expansion before running.
        solver : str, default="scipy"
            Local solver name passed to gradient backend.
        solver_options : dict | None, default=None
            Extra solver controls.
        env_n_iter : int, default=10
            Local boundary-fit iterations per boundary move.
        progress : bool, default=True
            Show global progress bar over local updates.
        track_boundary_fidelity : bool, default=False
            Enable per-step boundary fidelity sampling.
        renormalize : bool, default=True
            Normalize state once after sweeps and append to ``norm_trace``.
        """
        self._ensure_boundary_chi(chi)
        self._reset_run_traces()

        energy_before = self.energy(
            n_iter=env_n_iter,
            progress=False,
            track_boundary_fidelity=False,
        )

        track_boundary_fidelity = bool(track_boundary_fidelity)
        all_runs = []
        axis_seq = list(axes)

        def _steps_for_axis(axis_name):
            n_axis = self._axis_n(axis_name)
            return n_axis + (2 * n_round_trips * max(n_axis - 1, 0))

        total_steps = n_cycles * sum(_steps_for_axis(axis_name) for axis_name in axis_seq)
        global_progress = None
        if progress:
            global_progress = tqdm(
                total=total_steps,
                desc="energy_dmrg:",
                leave=True,
                position=0,
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="#009fd4",
                disable=not progress,
            )

        for cyc in range(n_cycles):
            for axis in axis_seq:

                def _on_run(run_info):
                    if global_progress is None:
                        return
                    global_progress.update(1)
                    postfix = {}
                    local_energy = run_info.get("energy_final")
                    if local_energy is not None:
                        postfix["E_loc"] = f"{float(local_energy):.6f}"
                    t_bdy = run_info.get("time_boundary")
                    t_opt = run_info.get("time_optimize")
                    if t_bdy is not None:
                        postfix["t_bdy"] = f"{float(t_bdy):.2f}s"
                    if t_opt is not None:
                        postfix["t_opt"] = f"{float(t_opt):.2f}s"
                    flops_norm = run_info.get("flops_norm")
                    flops_overlap = run_info.get("flops_overlap")
                    if flops_norm is not None:
                        postfix["flops_norm"] = f"{float(flops_norm):.2f}"
                    if flops_overlap is not None:
                        postfix["flops_overlap"] = f"{float(flops_overlap):.2f}"
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
                    track_boundary_fidelity=track_boundary_fidelity,
                    # Apply state renormalization once at global-run end to
                    # avoid compounding boundary-estimation noise.
                    renormalize=False,
                )
                all_runs.extend(axis_runs)
                self._collect_axis_run_traces(axis_runs)

        if global_progress is not None:
            global_progress.close()

        energy_after = self.energy(
            n_iter=env_n_iter,
            progress=False,
            track_boundary_fidelity=False,
        )

        if renormalize:
            old_norm = self._normalize_state(env_n_iter=env_n_iter)
            self.norm_trace.append(float(abs(complex(old_norm))))

        return {
            "runs": all_runs,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "energy": list(self.energy_trace),
            "loss": list(self.loss),
            "norm_trace": list(self.norm_trace),
            "fidels": list(self.fidels),
        }

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
        n_cycles=None,
        chi=None,
        progress=None,
        renormalize=None,
        track_boundary_fidelity=None,
    ):
        """High-level wrapper around :meth:`optimize_global`.

        Any explicit argument here overrides values stored in
        ``self.optimize_kwargs`` for the current call.
        """
        opts = dict(getattr(self, "optimize_kwargs", {}))
        if n_cycles is not None:
            opts["n_cycles"] = n_cycles
        if chi is not None:
            opts["chi"] = chi
        if progress is not None:
            opts["progress"] = progress
        if renormalize is not None:
            opts["renormalize"] = renormalize
        if track_boundary_fidelity is not None:
            opts["track_boundary_fidelity"] = track_boundary_fidelity

        return self.optimize_global(
            axes=opts.get("axes", ("y", "x")),
            n_cycles=opts.get("n_cycles", 1),
            n_round_trips=opts.get("n_round_trips", 1),
            chi=opts.get("chi"),
            solver=opts.get("optimizer", "scipy"),
            solver_options=opts.get("optimizer_options"),
            env_n_iter=opts.get("env_n_iter", 10),
            progress=opts.get("progress", True),
            track_boundary_fidelity=opts.get("track_boundary_fidelity", False),
            renormalize=opts.get("renormalize", True),
        )
