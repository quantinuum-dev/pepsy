"""Diagnostic plotting utilities for PEPS sweep results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimize_sweep import SweepResult

__all__ = ["plot_sweep_diagnostics", "plot_inner_loss", "plot_global_loss_trajectory"]

# Panel names the user can request.
PANELS = (
    "fidelity",
    "loss",
    "boundary_fidelity",
    "boundary_norms",
    "state_norm",
    "loss_delta",
    "timing",
    "cumulative_time",
)

# Panels that require debug=True data.
_DEBUG_PANELS = {"boundary_fidelity", "boundary_norms", "state_norm"}


def plot_sweep_diagnostics(
    sweep_result: SweepResult,
    *,
    panels=None,
    show_inner_loss: bool = False,
    show_global_loss: bool = False,
) -> None:
    """Plot a diagnostic grid from a :class:`SweepResult`.

    Parameters
    ----------
    panels : sequence[str] | None, default=None
        Which panels to show.  ``None`` shows all available panels.
        Choose from: ``"fidelity"``, ``"loss"``, ``"boundary_fidelity"``,
        ``"boundary_norms"``, ``"state_norm"``, ``"loss_delta"``,
        ``"timing"``, ``"cumulative_time"``.
        Debug-only panels are silently skipped when debug data is absent.
    show_inner_loss : bool, default=False
        Show per-step inner optimizer loss trajectory subplots.
    show_global_loss : bool, default=False
        Show concatenated global loss trajectory.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    runs = sweep_result.runs
    if not runs:
        print("No sweep runs to plot.")
        return

    steps = np.arange(len(runs))

    # --- Extract per-step data ---
    approx_fidelity = np.array([1.0 - r["loss_final"] for r in runs])
    exact_fidelity = [r.get("global_fidelity_after") for r in runs]
    approx_loss = np.array([r["loss_final"] for r in runs])
    exact_loss = [r.get("global_loss_after") for r in runs]
    bdy_fid_norm = [r.get("boundary_fidelity_norm") for r in runs]
    bdy_fid_overlap = [r.get("boundary_fidelity_overlap") for r in runs]
    bdy_norm_n = [r.get("bdy_norm_norm") for r in runs]
    bdy_norm_o = [r.get("bdy_norm_overlap") for r in runs]
    state_norms = [r.get("state_norm") for r in runs]
    time_bdy = np.array([r.get("time_boundary", 0.0) for r in runs])
    time_opt = np.array([r.get("time_optimize", 0.0) for r in runs])
    deltas = np.array([r["loss_delta"] for r in runs])

    has_debug = all(v is not None for v in exact_fidelity)

    # --- Resolve requested panels ---
    if panels is None:
        active = [p for p in PANELS if has_debug or p not in _DEBUG_PANELS]
    else:
        active = [
            p for p in panels
            if p in PANELS and (has_debug or p not in _DEBUG_PANELS)
        ]
    if not active:
        print("No panels to plot (check panel names or debug data).")
        return

    # --- Style ---
    _v = plt.cm.viridis(np.linspace(0.10, 0.90, 10))
    _line_kw = dict(linewidth=1.5, alpha=0.9)

    C = {
        "approx": _v[1],
        "exact": _v[7],
        "fbn": _v[3],
        "fbo": _v[8],
        "norm_bdy": _v[2],
        "ovlp_bdy": _v[6],
        "state": _v[4],
        "bdy_time": _v[3],
        "opt_time": _v[7],
        "good": "#22c55e",
        "bad": "#ef4444",
        "total": _v[5],
    }

    n_panels = len(active)
    n_cols = min(2, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig, axes_flat = plt.subplots(
            n_rows, n_cols, figsize=(7.5 * n_cols, 4.25 * n_rows), dpi=120,
            squeeze=False,
        )
        fig.patch.set_facecolor("white")

        for idx, name in enumerate(active):
            row, col = divmod(idx, n_cols)
            ax = axes_flat[row, col]

            if name == "fidelity":
                ax.plot(steps, approx_fidelity, color=C["approx"],
                        label="approx (boundary)", zorder=3, **_line_kw)
                if has_debug:
                    ax.plot(steps, exact_fidelity, color=C["exact"],
                            label="exact (full)", zorder=3, **_line_kw)
                ax.set_ylabel("Fidelity")
                ax.set_title("Fidelity")
                ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
                ax.grid(alpha=0.15, lw=0.6)

            elif name == "loss":
                ax.semilogy(steps, approx_loss, color=C["approx"],
                            label="approx loss", **_line_kw)
                if has_debug:
                    ax.semilogy(steps, exact_loss, color=C["exact"],
                                label="exact loss", **_line_kw)
                ax.set_ylabel("Loss")
                ax.set_title("Loss (log scale)")
                ax.legend(fontsize=9, framealpha=0.9)
                ax.grid(alpha=0.15, lw=0.6, which="both")
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            elif name == "boundary_fidelity":
                fbn_idx = [i for i, v in enumerate(bdy_fid_norm) if v is not None]
                fbn_val = [bdy_fid_norm[i] for i in fbn_idx]
                fbo_idx = [i for i, v in enumerate(bdy_fid_overlap) if v is not None]
                fbo_val = [bdy_fid_overlap[i] for i in fbo_idx]
                if fbn_val:
                    ax.plot(fbn_idx, fbn_val, color=C["fbn"],
                            label="norm", zorder=3, **_line_kw)
                if fbo_val:
                    ax.plot(fbo_idx, fbo_val, color=C["fbo"],
                            label="overlap", zorder=3, **_line_kw)
                ax.set_ylabel("Boundary Fidelity")
                ax.set_xlabel("Step")
                ax.set_title("Boundary MPS Fitting Fidelity")
                ax.legend(fontsize=9, framealpha=0.9)
                ax.grid(alpha=0.15, lw=0.6)

            elif name == "boundary_norms":
                nn_idx = [i for i, v in enumerate(bdy_norm_n) if v is not None]
                nn_val = [bdy_norm_n[i] for i in nn_idx]
                no_idx = [i for i, v in enumerate(bdy_norm_o) if v is not None]
                no_val = [bdy_norm_o[i] for i in no_idx]
                if nn_val:
                    ax.semilogy(nn_idx, nn_val, color=C["norm_bdy"],
                                label="norm bdy MPS", **_line_kw)
                if no_val:
                    ax.semilogy(no_idx, no_val, color=C["ovlp_bdy"],
                                label="overlap bdy MPS", **_line_kw)
                ax.set_ylabel("Mean Norm")
                ax.set_xlabel("Step")
                ax.set_title("Boundary MPS Norms (log)")
                ax.legend(fontsize=9, framealpha=0.9)
                ax.grid(alpha=0.15, lw=0.6, which="both")
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            elif name == "state_norm":
                sn_idx = [i for i, v in enumerate(state_norms) if v is not None]
                sn_val = [state_norms[i] for i in sn_idx]
                if sn_val:
                    ax.semilogy(sn_idx, sn_val, color=C["state"],
                                label="||ψ||² (exact)", zorder=3, **_line_kw)
                    ax.axhline(1.0, ls="--", lw=1.0, color="#9ca3af",
                               label="ideal = 1", zorder=1)
                ax.set_ylabel("State Norm")
                ax.set_xlabel("Step")
                ax.set_title("Exact State Norm (log)")
                ax.legend(fontsize=9, framealpha=0.9)
                ax.grid(alpha=0.15, lw=0.6, which="both")
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            elif name == "loss_delta":
                colors = [C["good"] if d <= 0 else C["bad"] for d in deltas]
                ax.bar(steps, deltas, color=colors, alpha=0.75, width=0.7,
                       edgecolor="white", linewidth=0.4)
                ax.axhline(0, ls="-", lw=0.8, color="black", zorder=1)
                ax.set_ylabel("Δ Loss")
                ax.set_xlabel("Step")
                ax.set_title("Loss Change per Step")
                ax.grid(alpha=0.15, lw=0.6, axis="y")

            elif name == "timing":
                ax.bar(steps, time_bdy, width=0.7, color=C["bdy_time"], alpha=0.8,
                       label="boundary move", edgecolor="white", linewidth=0.4)
                ax.bar(steps, time_opt, width=0.7, bottom=time_bdy,
                       color=C["opt_time"], alpha=0.8,
                       label="optimization", edgecolor="white", linewidth=0.4)
                ax.set_ylabel("Time (s)")
                ax.set_xlabel("Step")
                ax.set_title("Time per Step")
                ax.legend(fontsize=9, framealpha=0.9)
                ax.grid(alpha=0.15, lw=0.6, axis="y")

            elif name == "cumulative_time":
                cum_bdy = np.cumsum(time_bdy)
                cum_opt = np.cumsum(time_opt)
                ax.fill_between(steps, 0, cum_bdy, alpha=0.25, color=C["bdy_time"])
                ax.fill_between(steps, cum_bdy, cum_bdy + cum_opt,
                                alpha=0.25, color=C["opt_time"])
                ax.plot(steps, cum_bdy, color=C["bdy_time"],
                        label="boundary", **_line_kw)
                ax.plot(steps, cum_opt, color=C["opt_time"],
                        label="optimize", **_line_kw)
                ax.plot(steps, cum_bdy + cum_opt, color=C["total"],
                        label="total", **_line_kw)
                ax.set_ylabel("Cumulative Time (s)")
                ax.set_xlabel("Step")
                ax.set_title("Cumulative Time")
                ax.legend(fontsize=9, framealpha=0.9)
                ax.grid(alpha=0.15, lw=0.6)

        # Hide unused axes.
        for idx in range(n_panels, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes_flat[row, col].set_visible(False)

        fig.suptitle(
            "PEPS Boundary-DMRG Sweep Diagnostics",
            fontsize=16, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    total_bdy = time_bdy.sum()
    total_opt = time_opt.sum()
    print(
        f"boundary: {total_bdy:.1f}s  |  "
        f"optimize: {total_opt:.1f}s  |  "
        f"total: {total_bdy + total_opt:.1f}s"
    )
    if has_debug:
        print(
            f"fidelity: {sweep_result.fidelity_before:.8f} "
            f"→ {sweep_result.fidelity_after:.8f}"
        )

    if show_inner_loss:
        plot_inner_loss(sweep_result)
    if show_global_loss:
        plot_global_loss_trajectory(sweep_result)


def plot_inner_loss(sweep_result: SweepResult) -> None:
    """Plot inner optimizer loss trajectories for each sweep step.

    Only steps with more than 2 history entries are shown.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    runs = sweep_result.runs
    histories = [(i, r.get("history", [])) for i, r in enumerate(runs)]
    histories = [(i, h) for i, h in histories if len(h) > 2]

    if not histories:
        print("No inner loss history available (run with debug=True).")
        return

    n = len(histories)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes_ = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=120, squeeze=False,
    )
    fig.patch.set_facecolor("white")

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    for plot_idx, (step_idx, hist) in enumerate(histories):
        row, col = divmod(plot_idx, n_cols)
        ax = axes_[row, col]
        r = runs[step_idx]
        label = f"{r.get('axis', '?')}_{r.get('sweep', '?')[:3]}_{r.get('index', '?')}"
        ax.semilogy(hist, color=cmap[plot_idx], linewidth=1.5, alpha=0.9)
        ax.set_title(f"step {step_idx}: {label}", fontsize=10, fontweight="bold")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.15, lw=0.6, which="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for plot_idx in range(n, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes_[row, col].set_visible(False)

    fig.suptitle("Inner Optimizer Loss Trajectories", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    plt.show()


def plot_global_loss_trajectory(sweep_result: SweepResult) -> None:
    """Plot all inner optimizer losses as one continuous trajectory.

    Concatenates the ``history`` vectors from every sweep step into a single
    semilogy curve, with vertical lines marking step boundaries and colored
    background bands for each step.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    runs = sweep_result.runs
    histories = [(i, r.get("history", [])) for i, r in enumerate(runs)]
    histories = [(i, h) for i, h in histories if len(h) > 0]

    if not histories:
        print("No loss history available (run with debug=True).")
        return

    # Build concatenated vector and step boundaries.
    all_loss: list[float] = []
    boundaries: list[int] = [0]
    labels: list[str] = []
    for step_idx, hist in histories:
        all_loss.extend(hist)
        boundaries.append(len(all_loss))
        r = runs[step_idx]
        labels.append(
            f"{r.get('axis', '?')}_{r.get('sweep', '?')[:3]}_{r.get('index', '?')}"
        )

    iters = np.arange(len(all_loss))
    loss_arr = np.array(all_loss, dtype=float)
    n_steps = len(labels)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_steps, 1)))

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=120)
    fig.patch.set_facecolor("white")

    # Colored bands per step.
    for k in range(n_steps):
        lo, hi = boundaries[k], boundaries[k + 1]
        ax.axvspan(lo, hi, alpha=0.08, color=cmap[k])

    # Main curve.
    ax.semilogy(iters, loss_arr, color="#3b82f6", linewidth=1.2, alpha=0.9)

    # Step boundary markers.
    for k in range(1, len(boundaries) - 1):
        ax.axvline(boundaries[k], ls="--", lw=0.6, color="#9ca3af", alpha=0.6)

    # Tick labels at step midpoints (show only a subset to avoid clutter).
    mids = [(boundaries[k] + boundaries[k + 1]) / 2 for k in range(n_steps)]
    max_labels = 30
    if n_steps > max_labels:
        step_every = max(1, n_steps // max_labels)
        tick_pos = [mids[k] for k in range(0, n_steps, step_every)]
        tick_lbl = [labels[k] for k in range(0, n_steps, step_every)]
    else:
        tick_pos, tick_lbl = mids, labels
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=60, ha="right", fontsize=7)

    ax.set_xlabel("global iteration")
    ax.set_ylabel("loss")
    ax.set_title("Global Loss Trajectory", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.15, lw=0.6, which="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.show()
