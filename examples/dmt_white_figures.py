"""Compare Pepsy trajectories with digitized White et al. Figs. 7 and 8.

Fig. 7 input: dmt_ising_paper_energy.py --backend torch --sample-every 16.
Fig. 8 input: dmt_ising_dynamics.py --backend torch --models tilted
  --initial near-y-figure --dt .5 --time 50 --sample-every 1
  --unmerged-turnaround --chis 16 32 64
  --methods mps-svd mps-direct mpo-dmt pauli-dmt.

The independent Fig. 8 gate reference must match the published reference
before a comparison is accepted. Failed Fig. 7 runs remain explicit failures.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def figure_mode(data):
    """PDF convention: zero-based Fourier phase, energy H_Pauli/4."""
    return np.real(data["energy_fourier"] * np.exp(-1j * np.pi / 4)) / 4


def comparison_times(series, endpoint):
    """The PDF's last x=endpoint-1 sample contains the endpoint state.

    Retain the raw digitized x coordinates in the fixture. This alignment
    is independently supported by the Fig. 8 untruncated reference and
    Fig. 7's MPS endpoint; all preceding samples keep their printed time.
    """
    times = np.array(series["times"], dtype=float)
    if len(times) and times[-1] == endpoint - 1:
        times[-1] = endpoint
    return times


def plot_continuations(folders, baseline, paper, output, report):
    """Keep completed warning-mode curves separate from earlier strict stops."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for method, label, style in (("mps", "Published MPS", ":"),
                                  ("dmt", "Published DMT", "--")):
        curve = paper["fig7"][method]
        times = comparison_times(curve, 40)
        values = np.asarray(curve["values"])
        axes[0, 0].plot(times, values, ls=style, color="black", label=label)
        axes[0, 1].semilogy(times[1:], np.maximum(abs(values[1:] - values[0]), 1e-16),
                             ls=style, color="black")
    trajectories = [("Pepsy MPS complex128", np.load(baseline / "mps-direct.npz"))]
    report["fig7"]["continuations"] = []
    for folder in folders:
        config = json.loads((folder / "config.json").read_text())
        rows = json.loads((folder / "summary.json").read_text())
        for row in rows:
            data = np.load(folder / (row["method"] + ".npz"))
            label = ("Pauli float64" if row["method"].endswith("real") else
                     "Ordinary MPO complex128" if row["method"] == "mpo-dmt" else
                     "Pauli complex128")
            if config.get("cpu_svd") == "scipy_gesvd":
                label += " / GESVD"
            if config.get("skip_region_checks"):
                label += " / no full-region checks"
            trajectories.append((label, data))
            published = paper["fig7"]["dmt"]
            indices = np.searchsorted(data["paper_times"], comparison_times(published, 40))
            energy_difference = data["energy"].real[indices] / 4 - np.asarray(published["values"])
            report["fig7"]["continuations"].append({
                "config": config, **row,
                "figure_energy_real_rmse_vs_published_dmt": float(np.sqrt(np.mean(energy_difference**2))),
                "figure_energy_real_max_error_vs_published_dmt": float(abs(energy_difference).max()),
                "max_absolute_imaginary_energy_paper": float(abs(data["energy"].imag / 4).max()),
                "max_absolute_trace_drift": float(abs(data["trace"] - 1).max()),
                "final_trace_real": float(data["trace"][-1].real),
                "final_trace_imag": float(data["trace"][-1].imag)})
    records = []
    for label, data in trajectories:
        t = data["paper_times"]
        energy = data["energy"].real / 4
        line, = axes[0, 0].plot(t, energy, label=label)
        style = {"color": line.get_color()}
        axes[0, 1].semilogy(t[1:], np.maximum(abs(energy[1:] - energy[0]), 1e-16), **style)
        axes[1, 0].semilogy(t, np.maximum(abs(data["trace"] - 1), 1e-16), **style)
        axes[1, 1].plot(t, data["min_local_eigenvalue"], **style)
        for i, time in enumerate(t):
            records.append({"method": label, "time": float(time),
                            "energy_real": float(data["energy"][i].real / 4),
                            "energy_imag": float(data["energy"][i].imag / 4),
                            "trace_real": float(data["trace"][i].real),
                            "trace_imag": float(data["trace"][i].imag),
                            "min_local_eigenvalue": float(data["min_local_eigenvalue"][i]),
                            "local_hermiticity_defect": float(data["max_local_hermiticity_defect"][i])})
    with (output / "fig7-continuations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    axes[1, 1].set_yscale("symlog", linthresh=1e-6)
    for ax, label in zip(axes.flat, ("Total energy (paper units)", "Absolute energy drift",
                                   "Absolute trace drift", "Minimum three-site eigenvalue")):
        ax.set_xlabel("Time (published-figure convention)")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle("L=64, chi=16: continuing after precision warnings\n"
                 "Energy conservation does not certify physical or accurate dynamics", fontsize=17)
    for extension in ("png", "svg"):
        fig.savefig(output / f"fig7-continuations.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig7-input", type=Path, default=Path("/tmp/pepsy-white-fig7-torch"))
    parser.add_argument("--fig8-input", type=Path, default=Path("/tmp/pepsy-white-fig8-l16-torch"))
    parser.add_argument("--fig7-continuations", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("/tmp/pepsy-white-comparison"))
    args = parser.parse_args()
    paper = json.loads((Path(__file__).parent / "data/white2018_figures.json").read_text())
    reference = np.load(args.fig8_input / "tilted-trotter.npz")
    times = reference["times"] * 2
    eps = figure_mode(reference)
    author_ref = paper["fig8"]["16"]["mps-chi256"]
    indices = np.searchsorted(times, comparison_times(author_ref, 100))
    difference = float(np.max(abs(eps[indices] - author_ref["values"])))
    if difference > 1e-8:
        raise AssertionError(f"Published semiexact reference mismatch: {difference}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.fig8_input / "summary.json").read_text())
    failures_file = args.fig7_input / "failures.json"
    failures = json.loads(failures_file.read_text()) if failures_file.exists() else []
    report = {"paper": paper["source"], "reference_max_difference": difference,
              "conventions": {"state": "[-i(1+g), 1-g] normalized; period-eight g=+-0.1",
                              "gate_time_per_figure_time": .5, "energy_scale": .25,
                              "fourier_phase_shift": "exp(-i*pi/4)",
                              "final_exported_sample": "PDF x=99 -> t=100 in Fig.8; x=39 -> t=40 in Fig.7"},
              "configs": {"fig7": json.loads((args.fig7_input / "config.json").read_text()),
                          "fig8": json.loads((args.fig8_input / "config.json").read_text())},
              "fig7": {"completed": json.loads((args.fig7_input / "summary.json").read_text()),
                       "failed": failures}, "fig8": []}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
                         "legend.fontsize": 11, "lines.linewidth": 1.8})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, layout="constrained")
    styles = {"mps-svd": ("MPS TEBD", "#d55e00", "-"),
              "mpo-dmt": ("Ordinary MPO DMT (non-Hermitian)", "#009e73", "--"),
              "pauli-dmt": ("Native Pauli DMT", "#0072b2", "-")}
    for col, cap in enumerate((16, 32, 64)):
        axes[0, col].plot(times, eps, color="black", lw=2.1, label="Independent untruncated gates")
        published = paper["fig8"]["16"][f"dmt-chi{cap}"]
        pt = comparison_times(published, 100)
        pv = np.array(published["values"])
        pe = eps[np.searchsorted(times, pt)]
        axes[0, col].plot(pt, pv, color="#9467bd", ls=":", label="Published DMT (digitized)")
        axes[1, col].semilogy(pt[1:], np.maximum(abs(pv[1:] - pe[1:]), 1e-16), color="#9467bd", ls=":")
        for row in rows:
            if row["chi"] != cap:
                continue
            method = row["method"]
            data = np.load(args.fig8_input / f"tilted-{method}-chi{cap}.npz")
            values = figure_mode(data)
            error = abs(values - eps)
            record = dict(row, figure_mode_rmse=float(np.sqrt(np.mean(error[indices]**2))),
                          figure_mode_max_error=float(error.max()),
                          figure_mode_rmse_all101=float(np.sqrt(np.mean(error**2))))
            record["published_dmt_mode_rmse"] = float(np.sqrt(np.mean((pv - pe)**2)))
            report["fig8"].append(record)
            if method in styles:
                label, color, style = styles[method]
                axes[0, col].plot(times, values, color=color, ls=style, label=label)
                axes[1, col].semilogy(times[1:], np.maximum(error[1:], 1e-16), color=color, ls=style)
        axes[0, col].set_title(rf"$L=16,\ \chi={cap}$")
        axes[0, col].set_ylim(-.001, .034)
        axes[1, col].set_ylim(1e-7, .12)
        axes[1, col].set_xlabel("Time (published-figure convention)")
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
    axes[0, 0].set_ylabel(r"Re $\langle\epsilon_{\pi/4}\rangle$")
    axes[1, 0].set_ylabel("Absolute mode error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Fig. 8 test with Pepsy: Torch complex128\nUpper panels use the paper's vertical range; lower panels show all errors", fontsize=17)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"fig8.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained",
                              gridspec_kw={"width_ratios": [1.4, 1]})
    for name, label, color, style in (("mps", "Published MPS", "#d55e00", ":"),
                                      ("dmt", "Published DMT", "black", "-")):
        data = paper["fig7"][name]
        axes[0].plot(comparison_times(data, 40), data["values"], color=color, ls=style, label=label)
    data = np.load(args.fig7_input / "mps-direct.npz")
    energy = data["energy"].real / 4
    t7 = data["paper_times"]
    axes[0].plot(t7, energy, color="#d55e00", label="Pepsy MPS complex128")
    authors = paper["fig7"]["mps"]
    i7 = np.searchsorted(t7, comparison_times(authors, 40))
    report["fig7"]["mps_max_difference"] = float(abs(energy[i7] - authors["values"]).max())
    first30 = np.asarray(authors["times"]) <= 30
    report["fig7"]["mps_max_difference_through30"] = float(abs(energy[i7][first30] - np.asarray(authors["values"])[first30]).max())
    axes[0].set_xlabel("Time (published-figure convention)")
    axes[0].set_ylabel("Total energy (published-figure units)")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].axis("off")
    status = ["Pepsy L=64 DMT status", ""]
    for failure in failures:
        reason = ("Local-marginal preservation guard" if "marginals" in failure["error"]
                  else "Trace preservation guard" if "trace" in failure["error"]
                  else "Numerical precision guard")
        status.extend([failure["method"] + ": STOPPED", reason, ""])
    status.extend(["No completed Pepsy DMT curve is drawn.",
                   "Trace and local-density guards stayed enabled."])
    import textwrap
    axes[1].text(0, .96, "\n".join(textwrap.fill(line, 52) if line else "" for line in status),
                 va="top", fontsize=12, linespacing=1.4)
    fig.suptitle("Fig. 7 test: L=64, chi=16, dt=0.0625", fontsize=17)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"fig7.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.fig7_continuations:
        plot_continuations(args.fig7_continuations, args.fig7_input, paper, args.output, report)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["method", "chi", "backend", "dtype", "figure_mode_rmse", "figure_mode_max_error",
               "published_dmt_mode_rmse", "max_local_hermiticity_defect", "min_local_eigenvalue"]
    with (args.output / "fig8-results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["fig8"])
    print(json.dumps({"reference_max_difference": difference,
                      "fig7_failed_methods": [f["method"] for f in failures],
                      "fig8_completed_rows": len(report["fig8"])}))


if __name__ == "__main__":
    main()
