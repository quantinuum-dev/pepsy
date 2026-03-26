"""High-level PEPS sweep optimizer with axis-alternating boundary updates."""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass
from typing import Any

import quimb.tensor as qtn
from tqdm import tqdm

from .boundary_norm import prepare_boundary_inputs, normalize
from .boundary_sweeps import CompBdy
from .gradient_solver import optimize_packed_params as run_gradient_solver

_PHYS_IND_PATTERN = re.compile(r"^k\d+(?:,\d+)*$")
_TAG_X = re.compile(r"^X(\d+)$")
_TAG_Y = re.compile(r"^Y(\d+)$")

__all__ = ["PEPSSweepOptimizer", "SweepResult"]


@dataclass(frozen=True)
class SweepResult:
    """Return object for global sweep runs."""

    runs: list[dict[str, Any]]
    fidelity_before: float | None
    fidelity_after: float | None
    loss_before: float | None
    loss_after: float | None


class PEPSSweepOptimizer:  # pylint: disable=too-many-instance-attributes
    """Optimize PEPS slices with alternating x/y boundary sweeps.

    Parameters
    ----------
    state : qtn.TensorNetwork
        Trainable PEPS-like tensor network.
    target : qtn.TensorNetwork
        Reference network for overlap objective.
    bdy : pepsy.boundary_states.BdyMPS
        Boundary container used for norm contractions.
    bdy_overlap : pepsy.boundary_states.BdyMPS
        Boundary container used for overlap contractions.
    opt : object
        Contraction optimizer.
    dmrg_run : {"eff", "global"}, default="eff"
        Backend mode passed to :class:`pepsy.boundary_sweeps.CompBdy`.
    """

    def __init__(
        self,
        state,
        target,
        *,
        bdy,
        bdy_overlap,
        opt,
        dmrg_run="eff",
        equalize_norms=1,
    ):
        self.state = state
        self.target = target
        self.bdy = bdy
        self.bdy_overlap = bdy_overlap
        self.opt = opt
        self.dmrg_run = dmrg_run
        self.equalize_norms = equalize_norms

        self.Lx, self.Ly = self._infer_shape(self.state)

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
        norm_state = abs(complex((self.state.H & self.state).contract(all, optimize=self.opt)))
        norm_target = abs(complex((self.target.H & self.target).contract(all, optimize=self.opt)))
        overlap = complex((self.target.H & self.state).contract(all, optimize=self.opt))
        fidelity = (abs(overlap) ** 2) / (norm_state * norm_target)
        loss = 1.0 - fidelity
        return float(fidelity.real), float(loss.real)

    @staticmethod
    def format_runs_table(runs):
        """Format run records into a compact plain-text table."""
        headers = [
            "axis",
            "sweep",
            "index",
            "loss_before",
            "loss_after",
            "delta",
            "global_loss_after",
        ]
        rows = []
        for r in runs:
            l0 = float(r.get("loss_initial", r.get("history", [float("nan")])[0]))
            l1 = float(r.get("loss_final", r.get("history", [float("nan"), float("nan")])[-1]))
            d = float(r.get("loss_delta", l1 - l0))
            g = float(r.get("global_loss_after", float("nan")))
            rows.append(
                [
                    str(r.get("axis", "")),
                    str(r.get("sweep", "")),
                    str(r.get("index", "")),
                    f"{l0:.8f}",
                    f"{l1:.8f}",
                    f"{d:+.8f}",
                    f"{g:.8f}",
                ]
            )

        if not rows:
            return "(no runs)"

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def _fmt(row):
            return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

        sep = "-+-".join("-" * w for w in widths)
        lines = [_fmt(headers), sep]
        lines.extend(_fmt(row) for row in rows)
        return "\n".join(lines)

    @staticmethod
    def print_runs_table(runs):
        """Print compact run summary."""
        print(PEPSSweepOptimizer.format_runs_table(runs))

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
        _, overlap_tn = prepare_boundary_inputs(ket=self.target, bra=self.state)

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
        solver="adam",
        solver_options=None,
    ):
        opts = dict(solver_options or {})
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
        tn_opt.equalize_norms_(self.equalize_norms)
        for tag in self._site_tensor_tags(axis, index):
            self.state[tag].modify(data=tn_opt[tag].data)
        return tn_opt

    def _normalize_state(self, env_n_iter=4):
        normalize_result = normalize(
            self.state,
            chi=self.bdy.chi,
            bdy=self.bdy,
            opt=self.opt,
            max_separation=1,
            n_iter=env_n_iter,
        )
        self.state = normalize_result["state"]

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
            fidel_=False,
        )
        comp_norm.move_bdy(
            n_iter=env_n_iter,
            pbar=False,
            direction=self._boundary_direction(axis, "right"),
            fidel_=False,
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
        fidel_=False,
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
                fidel_=fidel_,
            )
        overlap_fidelity = None
        norm_fidelity = None
        if fidel_:
            if comp_overlap.fidel:
                overlap_fidelity = float(complex(comp_overlap.fidel[-1]).real)
            if comp_norm.fidel:
                norm_fidelity = float(complex(comp_norm.fidel[-1]).real)
        return {"norm": norm_fidelity, "overlap": overlap_fidelity}

    def _optimize_axis_slice_with_current_env(
        self,
        index,
        norm_tn,
        overlap_tn,
        *,
        axis,
        solver="adam",
        solver_options=None,
        store_history=False,
        debug=False,
    ):
        axis_tag = self._axis_tag(axis)
        right_key, left_key = self._boundary_keys_for_index(index, axis)

        slice_state = self.state.select([f"{axis_tag}{index}"], "any")
        slice_target = self.target.select([f"{axis_tag}{index}"], "any")
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
            return 1 - (overlap_val / norm_val)

        initial_loss = float(loss_fn(params_init))
        params_opt, history = self._optimize_packed_params(
            params_init,
            loss_fn,
            solver=solver,
            solver_options=solver_options,
        )
        final_loss = float(loss_fn(params_opt).detach().cpu())

        params_opt = {k: v.detach().clone() for k, v in params_opt.items()}
        self._apply_slice_update(index, params_opt, skeleton, axis)

        run_info = {
            "axis": axis,
            "index": index,
            "right_key": right_key,
            "left_key": left_key,
            "loss_initial": initial_loss,
            "loss_final": final_loss,
            "loss_delta": final_loss - initial_loss,
        }
        if debug:
            _, global_loss = self.metrics()
            global_loss = float(global_loss)
            run_info["global_loss_after"] = global_loss
            run_info["global_fidelity_after"] = 1.0 - global_loss
            norm_state = abs(complex(
                (self.state.H & self.state).contract(all, optimize=self.opt)
            ))
            run_info["state_norm"] = float(norm_state)
            try:
                run_info["bdy_norm_norm"] = float(complex(self.bdy.norm).real)
                run_info["bdy_norm_overlap"] = float(complex(self.bdy_overlap.norm).real)
            except (ValueError, AttributeError):
                pass
        run_info["history"] = history if store_history else [initial_loss, final_loss]

        return run_info

    def _run_axis_half_sweep(
        self,
        indices,
        *,
        axis,
        update_side,
        sweep_name,
        solver="adam",
        solver_options=None,
        env_n_iter=4,
        store_history=False,
        run_callback=None,
        fidel_=False,
        debug=False,
    ):
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
                fidel_=fidel_,
            )
            t_bdy = time.perf_counter() - t0

            t0 = time.perf_counter()
            run_info = self._optimize_axis_slice_with_current_env(
                index,
                norm_tn,
                overlap_tn,
                axis=axis,
                solver=solver,
                solver_options=solver_options,
                store_history=store_history,
                debug=debug,
            )
            t_opt = time.perf_counter() - t0

            run_info["sweep"] = sweep_name
            run_info["time_boundary"] = t_bdy
            run_info["time_optimize"] = t_opt
            run_info["boundary_fidelity_norm"] = boundary_fidelity.get("norm")
            run_info["boundary_fidelity_overlap"] = boundary_fidelity.get("overlap")
            runs.append(run_info)
            if run_callback is not None:
                run_callback(run_info)

        return runs

    def optimize_axis(
        self,
        axis,
        *,
        n_round_trips=1,
        solver="adam",
        solver_options=None,
        env_n_iter=4,
        store_history=False,
        run_callback=None,
        fidel_=False,
        debug=False,
        renormalize=False,
    ):
        """Run one axis with forward + round-trip sweeps.

        Parameters
        ----------
        solver : str, default="adam"
            Gradient solver name. Supported values include ``adam``, ``lbfgs``,
            ``scipy-lbfgs``, and ``nlopt-lbfgs``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        """
        n = self._axis_n(axis)
        resolved_solver = self._resolve_user_solver(solver)
        all_runs = []

        if renormalize:
            self._normalize_state(env_n_iter)

        self._refresh_right_boundaries_once(axis, env_n_iter=env_n_iter)

        sweep_kwargs = dict(
            axis=axis,
            solver=resolved_solver,
            solver_options=solver_options,
            env_n_iter=env_n_iter,
            store_history=store_history,
            run_callback=run_callback,
            fidel_=fidel_,
            debug=debug,
        )

        all_runs.extend(
            self._run_axis_half_sweep(
                range(0, n),
                update_side="left",
                sweep_name="forward",
                **sweep_kwargs,
            )
        )

        for trip in range(n_round_trips):
            if renormalize:
                self._normalize_state(env_n_iter)

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
        solver="adam",
        solver_options=None,
        env_n_iter=4,
        pbar=True,
        store_history=False,
        debug=False,
        renormalize=False,
    ):
        """Run alternating axis sweeps and return a :class:`SweepResult`.

        Parameters
        ----------
        solver : str, default="adam"
            Gradient solver name. Supported values include ``adam``, ``lbfgs``,
            ``scipy-lbfgs``, and ``nlopt-lbfgs``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        debug : bool, default=False
            When True, compute exact metrics and boundary fidelities per step.
        """
        if debug:
            fid_before, loss_before = self.metrics()
        else:
            fid_before, loss_before = None, None
        all_runs = []
        axis_seq = list(axes)

        def _current_chi():
            values = []
            for obj in (self.bdy, self.bdy_overlap):
                chi_val = getattr(obj, "chi", None)
                if chi_val is not None:
                    values.append(int(chi_val))
            return max(values) if values else None

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
                colour="CYAN",
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
                    loss_final = run_info.get("loss_final")
                    if loss_final is not None:
                        postfix["loss"] = f"{float(loss_final):.6f}"
                    chi_now = _current_chi()
                    if chi_now is not None:
                        postfix["chi"] = chi_now
                    sweep_name = run_info.get("sweep")
                    axis_name = run_info.get("axis")
                    if sweep_name is not None:
                        postfix["dir"] = (
                            f"{axis_name}_{sweep_name}" if axis_name is not None else sweep_name
                        )
                    global_progress.set_postfix(postfix)

                all_runs.extend(
                    self.optimize_axis(
                        axis,
                        n_round_trips=n_round_trips,
                        solver=solver,
                        solver_options=solver_options,
                        env_n_iter=env_n_iter,
                        store_history=store_history,
                        run_callback=_on_run,
                        fidel_=debug,
                        debug=debug,
                        renormalize=renormalize,
                    )
                )

        if global_progress is not None:
            global_progress.close()

        if debug:
            fid_after, loss_after = self.metrics()
        else:
            fid_after, loss_after = None, None
        return SweepResult(
            runs=all_runs,
            fidelity_before=fid_before,
            fidelity_after=fid_after,
            loss_before=loss_before,
            loss_after=loss_after,
        )
