"""High-level PEPS sweep optimizer with axis-alternating boundary updates."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import quimb.tensor as qtn
from tqdm import tqdm

from .boundary_norm import prepare_boundary_inputs
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
    fidelity_before: float
    fidelity_after: float
    loss_before: float
    loss_after: float


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
    ):
        self.state = state
        self.target = target
        self.bdy = bdy
        self.bdy_overlap = bdy_overlap
        self.opt = opt
        self.dmrg_run = dmrg_run

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
    def _merge_solver_options(*, solver, solver_options=None, adam_options=None):
        """Merge new generic solver options with legacy Adam-only options."""
        merged = {}
        if solver_options is not None:
            if not isinstance(solver_options, dict):
                raise TypeError("solver_options must be a dict or None")
            merged.update({k: v for k, v in solver_options.items() if v is not None})

        if adam_options is None:
            return merged
        if solver != "adam":
            raise ValueError("adam_options is only valid when solver='adam'")
        if not isinstance(adam_options, dict):
            raise TypeError("adam_options must be a dict or None")

        overlap = set(merged) & set(adam_options)
        if overlap:
            keys = ", ".join(sorted(overlap))
            raise ValueError(
                f"Duplicate option keys in solver_options and adam_options: {keys}"
            )
        merged.update({k: v for k, v in adam_options.items() if v is not None})
        return merged

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
        lr=1e-2,
        adam_options=None,
        n_steps=100,
        log_every=20,
        show_opt_progress=False,
        opt_desc=None,
        progress_callback=None,
    ):
        effective_options = self._merge_solver_options(
            solver=solver,
            solver_options=solver_options,
            adam_options=adam_options,
        )
        return run_gradient_solver(
            params_init,
            loss_fn,
            solver=solver,
            solver_options=effective_options,
            lr=lr,
            n_steps=n_steps,
            log_every=log_every,
            show_opt_progress=show_opt_progress,
            opt_desc=opt_desc,
            progress_callback=progress_callback,
        )

    def _apply_slice_update(self, index, params_opt, skeleton, axis):
        tn_opt = qtn.unpack(params_opt, skeleton)
        for tag in self._site_tensor_tags(axis, index):
            self.state[tag].modify(data=tn_opt[tag].data)
        return tn_opt

    def _make_comp_pair(self, norm_tn, overlap_tn):
        comp_norm = CompBdy(norm_tn, self.bdy.mps_b, opt=self.opt, dmrg_run=self.dmrg_run)
        comp_overlap = CompBdy(overlap_tn, self.bdy_overlap.mps_b, opt=self.opt, dmrg_run=self.dmrg_run)
        return comp_norm, comp_overlap

    @staticmethod
    def _set_comp_norms(comp_norm, comp_overlap, *, norm_tn, overlap_tn):
        comp_norm.norm = norm_tn
        comp_overlap.norm = overlap_tn

    def _refresh_right_boundaries_once(self, axis, *, env_n_iter=4, show_env_progress=True):
        norm_tn, overlap_tn = self._prepare_current_double_layers()
        comp_norm, comp_overlap = self._make_comp_pair(norm_tn, overlap_tn)
        comp_overlap.move_bdy(
            n_iter=env_n_iter,
            pbar=show_env_progress,
            direction=self._boundary_direction(axis, "right"),
            fidel_=False,
        )
        comp_norm.move_bdy(
            n_iter=env_n_iter,
            pbar=show_env_progress,
            direction=self._boundary_direction(axis, "right"),
            fidel_=False,
        )

    def _advance_left_boundary_one_step(
        self,
        index,
        *,
        axis,
        comp_norm,
        comp_overlap,
        env_n_iter=4,
        show_env_progress=True,
        fidel_=False,
    ):
        if index <= 0:
            return {"norm": None, "overlap": None}
        pos = index - 1
        comp_overlap.move_step_bdy(
            pos=pos,
            n_iter=env_n_iter,
            pbar=show_env_progress,
            direction=self._boundary_direction(axis, "left"),
            fidel_=fidel_,
        )
        overlap_fidelity = None
        if fidel_ and comp_overlap.fidel:
            overlap_fidelity = float(complex(comp_overlap.fidel[-1]).real)
        comp_norm.move_step_bdy(
            pos=pos,
            n_iter=env_n_iter,
            pbar=show_env_progress,
            direction=self._boundary_direction(axis, "left"),
            fidel_=fidel_,
        )
        norm_fidelity = None
        if fidel_ and comp_norm.fidel:
            norm_fidelity = float(complex(comp_norm.fidel[-1]).real)
        return {"norm": norm_fidelity, "overlap": overlap_fidelity}

    def _advance_right_boundary_one_step(
        self,
        index,
        *,
        axis,
        comp_norm,
        comp_overlap,
        env_n_iter=4,
        show_env_progress=True,
        fidel_=False,
    ):
        n = self._axis_n(axis)
        if index >= n - 1:
            return {"norm": None, "overlap": None}
        pos = (n - 2) - index
        comp_overlap.move_step_bdy(
            pos=pos,
            n_iter=env_n_iter,
            pbar=show_env_progress,
            direction=self._boundary_direction(axis, "right"),
            fidel_=fidel_,
        )
        overlap_fidelity = None
        if fidel_ and comp_overlap.fidel:
            overlap_fidelity = float(complex(comp_overlap.fidel[-1]).real)
        comp_norm.move_step_bdy(
            pos=pos,
            n_iter=env_n_iter,
            pbar=show_env_progress,
            direction=self._boundary_direction(axis, "right"),
            fidel_=fidel_,
        )
        norm_fidelity = None
        if fidel_ and comp_norm.fidel:
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
        lr=1e-2,
        adam_options=None,
        n_steps=100,
        log_every=20,
        store_history=False,
        show_opt_progress=False,
        progress_callback=None,
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
            lr=lr,
            adam_options=adam_options,
            n_steps=n_steps,
            log_every=log_every,
            show_opt_progress=show_opt_progress,
            opt_desc=f"{axis_tag}{index}",
            progress_callback=progress_callback,
        )
        final_loss = float(loss_fn(params_opt).detach().cpu())


        params_opt = {k: v.detach().clone() for k, v in params_opt.items()}
        self._apply_slice_update(index, params_opt, skeleton, axis)

        _, global_loss = self.metrics()
        global_loss = float(global_loss)
        global_fidelity = 1.0 - global_loss

        run_info = {
            "axis": axis,
            "index": index,
            "right_key": right_key,
            "left_key": left_key,
            "loss_initial": initial_loss,
            "loss_final": final_loss,
            "loss_delta": final_loss - initial_loss,
            "global_loss_after": global_loss,
            "global_fidelity_after": global_fidelity,
        }
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
        lr=1e-2,
        adam_options=None,
        n_steps=100,
        log_every=20,
        env_n_iter=4,
        show_env_progress=True,
        store_history=False,
        show_sweep_progress=False,
        show_opt_progress=False,
        run_callback=None,
        fidel_=False,
    ):
        runs = []
        comp_norm = None
        comp_overlap = None

        index_iter = indices
        if show_sweep_progress:
            index_seq = list(indices)
            index_iter = tqdm(
                index_seq,
                total=len(index_seq),
                desc=f"sweep:{axis}:{sweep_name}",
                leave=False,
                colour="GREEN",
            )

        for index in index_iter:
            if show_sweep_progress:
                index_iter.set_postfix({"dir": sweep_name, "idx": index, "stage": "env"})
            norm_tn, overlap_tn = self._prepare_current_double_layers()
            if comp_norm is None:
                comp_norm, comp_overlap = self._make_comp_pair(norm_tn, overlap_tn)
            else:
                self._set_comp_norms(comp_norm, comp_overlap, norm_tn=norm_tn, overlap_tn=overlap_tn)

            if update_side == "left":
                boundary_fidelity = self._advance_left_boundary_one_step(
                    index,
                    axis=axis,
                    comp_norm=comp_norm,
                    comp_overlap=comp_overlap,
                    env_n_iter=env_n_iter,
                    show_env_progress=show_env_progress,
                    fidel_=fidel_,
                )
            elif update_side == "right":
                boundary_fidelity = self._advance_right_boundary_one_step(
                    index,
                    axis=axis,
                    comp_norm=comp_norm,
                    comp_overlap=comp_overlap,
                    env_n_iter=env_n_iter,
                    show_env_progress=show_env_progress,
                    fidel_=fidel_,
                )
            else:
                raise ValueError("update_side must be 'left' or 'right'")

            def _on_step(step_num, step_loss):
                if show_sweep_progress:
                    index_iter.set_postfix(
                        {
                            "dir": sweep_name,
                            "idx": index,
                            "step": step_num,
                            "loss": f"{step_loss:.6f}",
                        }
                    )

            run_info = self._optimize_axis_slice_with_current_env(
                index,
                norm_tn,
                overlap_tn,
                axis=axis,
                solver=solver,
                solver_options=solver_options,
                lr=lr,
                adam_options=adam_options,
                n_steps=n_steps,
                log_every=log_every,
                store_history=store_history,
                show_opt_progress=show_opt_progress,
                progress_callback=_on_step,
            )
            run_info["sweep"] = sweep_name
            run_info["boundary_fidelity_norm"] = boundary_fidelity.get("norm")
            run_info["boundary_fidelity_overlap"] = boundary_fidelity.get("overlap")
            runs.append(run_info)
            if run_callback is not None:
                run_callback(run_info)
            if show_sweep_progress:
                index_iter.set_postfix(
                    {
                        "dir": sweep_name,
                        "idx": index,
                        "loss": f"{run_info['loss_final']:.6f}",
                        "G": f"{run_info['global_loss_after']:.6f}",
                    }
                )

        return runs

    def optimize_axis(
        self,
        axis,
        *,
        n_round_trips=1,
        solver="adam",
        solver_options=None,
        lr=1e-2,
        adam_options=None,
        n_steps=100,
        log_every=20,
        env_n_iter=4,
        show_env_progress=True,
        store_history=False,
        show_sweep_progress=False,
        show_opt_progress=False,
        run_callback=None,
        fidel_=False,
    ):
        """Run one axis with forward + round-trip sweeps.

        Parameters
        ----------
        solver : str, default="adam"
            Gradient solver name. Supported values include ``adam``, ``lbfgs``,
            ``scipy-lbfgs``, and ``nlopt-lbfgs``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        adam_options : dict | None, default=None
            Legacy alias for Adam-specific options. Only valid with
            ``solver='adam'``.
        """
        n = self._axis_n(axis)
        resolved_solver = self._resolve_user_solver(solver)
        all_runs = []

        self._refresh_right_boundaries_once(axis, env_n_iter=env_n_iter, show_env_progress=show_env_progress)
        all_runs.extend(
            self._run_axis_half_sweep(
                range(0, n),
                axis=axis,
                update_side="left",
                sweep_name="forward",
                solver=resolved_solver,
                solver_options=solver_options,
                lr=lr,
                adam_options=adam_options,
                n_steps=n_steps,
                log_every=log_every,
                env_n_iter=env_n_iter,
                show_env_progress=show_env_progress,
                store_history=store_history,
                show_sweep_progress=show_sweep_progress,
                show_opt_progress=show_opt_progress,
                run_callback=run_callback,
                fidel_=fidel_,
            )
        )

        for trip in range(n_round_trips):
            all_runs.extend(
                self._run_axis_half_sweep(
                    range(n - 2, -1, -1),
                    axis=axis,
                    update_side="right",
                    sweep_name="backward",
                    solver=resolved_solver,
                    solver_options=solver_options,
                    lr=lr,
                    adam_options=adam_options,
                    n_steps=n_steps,
                    log_every=log_every,
                    env_n_iter=env_n_iter,
                    show_env_progress=show_env_progress,
                    store_history=store_history,
                    show_sweep_progress=show_sweep_progress,
                    show_opt_progress=show_opt_progress,
                    run_callback=run_callback,
                    fidel_=fidel_,
                )
            )

            forward_start = 1 if n > 1 else n
            all_runs.extend(
                self._run_axis_half_sweep(
                    range(forward_start, n),
                    axis=axis,
                    update_side="left",
                    sweep_name="forward",
                    solver=resolved_solver,
                    solver_options=solver_options,
                    lr=lr,
                    adam_options=adam_options,
                    n_steps=n_steps,
                    log_every=log_every,
                    env_n_iter=env_n_iter,
                    show_env_progress=show_env_progress,
                    store_history=store_history,
                    show_sweep_progress=show_sweep_progress,
                    show_opt_progress=show_opt_progress,
                    run_callback=run_callback,
                    fidel_=fidel_,
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
        lr=1e-2,
        adam_options=None,
        n_steps=100,
        log_every=20,
        env_n_iter=4,
        pbar=True,
        fidel_=False,
        store_history=False,
    ):
        """Run alternating axis sweeps and return a :class:`SweepResult`.

        Parameters
        ----------
        solver : str, default="adam"
            Gradient solver name. Supported values include ``adam``, ``lbfgs``,
            ``scipy-lbfgs``, and ``nlopt-lbfgs``.
        solver_options : dict | None, default=None
            Extra backend-specific options for the selected ``solver``.
        adam_options : dict | None, default=None
            Legacy alias for Adam-specific options. Only valid with
            ``solver='adam'``.
        """
        fid_before, loss_before = self.metrics()
        resolved_solver = self._resolve_user_solver(solver)
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
                    fidelity_now = run_info.get("global_fidelity_after")
                    if fidelity_now is None:
                        fidelity_now, _loss_now = self.metrics()
                    postfix = {"F": float(fidelity_now)}
                    chi_now = _current_chi()
                    if chi_now is not None:
                        postfix["chi"] = chi_now
                    sweep_name = run_info.get("sweep")
                    axis_name = run_info.get("axis")
                    if sweep_name is not None:
                        postfix["dir"] = (
                            f"{axis_name}_{sweep_name}" if axis_name is not None else sweep_name
                        )
                    if fidel_:
                        f_norm = run_info.get("boundary_fidelity_norm")
                        f_overlap = run_info.get("boundary_fidelity_overlap")
                        if f_norm is not None:
                            postfix["FbN"] = float(f_norm)
                        if f_overlap is not None:
                            postfix["FbO"] = float(f_overlap)
                    global_progress.set_postfix(
                        postfix
                    )

                all_runs.extend(
                    self.optimize_axis(
                        axis,
                        n_round_trips=n_round_trips,
                        solver=resolved_solver,
                        solver_options=solver_options,
                        lr=lr,
                        adam_options=adam_options,
                        n_steps=n_steps,
                        log_every=log_every,
                        env_n_iter=env_n_iter,
                        show_env_progress=False,
                        store_history=store_history,
                        show_sweep_progress=False,
                        show_opt_progress=False,
                        run_callback=_on_run,
                        fidel_=fidel_,
                    )
                )

        if global_progress is not None:
            global_progress.close()

        fid_after, loss_after = self.metrics()
        return SweepResult(
            runs=all_runs,
            fidelity_before=fid_before,
            fidelity_after=fid_after,
            loss_before=loss_before,
            loss_after=loss_after,
        )
