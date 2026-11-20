"""White et al. Fig. 7 energy test: L=64, chi=16, near-Y, dt=0.0625.

The PDF curves select gates exp(-i H_Pauli dt/2), with plotted energy
<H_Pauli>/4. This empirical time mapping is checked against Fig. 8's
untruncated reference; it differs from a literal reading of the equations.
The balanced near-Y amplitudes match Figs. 4 and 7's initial data, rather
than the literal printed state equation. Keeps both turnaround half gates.
This compares conservation, not accuracy:
there is no exact 64-site reference or digitized published curve here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

try:
    from examples import dmt_ising_dynamics as bench
except ModuleNotFoundError:
    import dmt_ising_dynamics as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--time", type=float, default=40.)
    parser.add_argument("--dt", type=float, default=.0625)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--preservation-check", choices=("raise", "warn", "off"), default="warn")
    parser.add_argument("--skip-region-checks", action="store_true",
                        help="skip full marginal verification, keeping bond checks unless policy is off")
    parser.add_argument("--cpu-svd", choices=("torch", "scipy_gesdd", "scipy_gesvd"), default="torch")
    parser.add_argument("--initial", choices=("near-y", "near-y-figure"), default="near-y-figure")
    parser.add_argument("--methods", nargs="+", default=["mps-direct", "pauli-dmt-real"],
                        choices=("mps-direct", "mpo-dmt", "pauli-dmt", "pauli-dmt-real"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-paper-energy"))
    args = parser.parse_args()
    steps = round(args.time / args.dt) if args.dt > 0 else 0
    if (args.length < 3 or steps < 1 or args.chi < 1 or args.sample_every < 1
            or steps % args.sample_every or not np.isclose(steps * args.dt, args.time)):
        parser.error("require valid positive dimensions and equally spaced samples including the endpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    error_path = args.output / "preservation-errors.jsonl"
    error_path.write_text("")

    def log_preservation_error(record):
        with error_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    (args.output / "config.json").write_text(json.dumps(vars(args) | {
        "output": str(args.output), "energy_scale": .25, "gate_time_scale": .5, "J": 1.,
        "hx": .9045, "hz": .8090, "turnaround": "two separately compressed half gates",
        "cutoff": 0., "truncation": "paper", "reference": None}, indent=2) + "\n")
    hx, hz = bench.MODELS["tilted"]
    vectors = bench.initial_vectors(args.length, args.initial)
    gates = bench.sweep_gates(bench.bond_hamiltonians(args.length, hx, hz), args.dt / 2,
                              merge_turnaround=False)
    samples = set(range(0, steps + 1, args.sample_every))
    times = np.arange(0, steps + 1, args.sample_every) * args.dt
    curves, results = {}, []
    failures = []
    with threadpool_limits(limits=1):
        if args.backend == "torch":
            import torch
            torch.set_num_threads(1)
            from pepsy.backends import TorchLinalgConfig
            TorchLinalgConfig(cpu_svd=args.cpu_svd).register()
        for method in args.methods:
            try:
                rows, info = bench.evolve(method, vectors, gates, args.chi, hx, hz, steps,
                                          samples, {1, steps // 2, steps}, backend=args.backend,
                                          preservation_check=args.preservation_check,
                                          preservation_log=log_preservation_error,
                                          check_regions=not args.skip_region_checks)
            except FloatingPointError as exc:
                failures.append({"method": method, "error": str(exc),
                                 "context": getattr(exc, "__notes__", [])})
                (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                continue
            data = bench.pack(rows)
            energy = data["energy"] / 4
            curves[method] = energy
            result = info | {"energy_initial_paper": float(energy[0].real),
                             "energy_final_paper": float(energy[-1].real),
                             "max_total_energy_drift_paper": float(abs(energy - energy[0]).max()),
                             "max_local_hermiticity_defect": float(data["max_local_hermiticity_defect"].max()),
                             "min_triple_eigenvalue": float(data["min_local_eigenvalue"].min())}
            results.append(result)
            np.savez_compressed(args.output / f"{method}.npz", paper_times=times, **data)
            (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    if not curves:
        raise RuntimeError(f"No completed trajectories: {failures}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "legend.fontsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for name, energy in curves.items():
        axes[0].plot(times, energy.real, label=name)
        axes[1].semilogy(times[1:], np.maximum(abs(energy[1:] - energy[0]), 1e-16), label=name)
    axes[0].axhline(next(iter(curves.values()))[0].real, color="black", ls=":", label="Initial energy")
    for ax, ylabel in zip(axes, ("Total energy (paper spin units)", "Absolute total energy drift")):
        ax.set_xlabel("Time (paper spin units)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f"White et al. Fig. 7 energy test, L={args.length}, chi={args.chi}")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"energy.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if failures:
        raise RuntimeError(f"Incomplete comparison; completed trajectories saved. Failures: {failures}")


if __name__ == "__main__":
    main()
