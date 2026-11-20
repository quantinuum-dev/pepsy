"""Research prototype: real-Pauli interval DMT without global 2**N arrays.

The physical model and product-mixture input match dmt_2d_scaling. Full and
window sweeps are alternative compression schedules, not identical algorithms.
At 6x6 only commuting-ZZ local oracles and chi convergence are exact checks.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import warnings

import autoray as ar
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from pepsy.operators import square_lattice_dmt_regions
from pepsy.operators._pauli_dmt import _PauliDMTWorkspace
from pepsy.operators.pauli_mpo import _array_like, _host_array
from pepsy.tensors import OneDMap

try:
    from examples import dmt_ising_dynamics as ref
except ModuleNotFoundError:
    import dmt_ising_dynamics as ref


def problem(lx, ly, dt, *, hx=1.05, hz=.5, weight=.6):
    """Geometry and rank-two mixture only: storage grows linearly in N."""
    if min(lx, ly) < 2 or not np.isfinite(dt) or dt <= 0 or not 0 <= weight <= 1:
        raise ValueError("Require sides >=2, positive finite dt, and weight in [0, 1].")
    mapper = OneDMap(lx, ly, mode="snake-row-major")
    coords, mapping = mapper.build()
    edges = ([tuple(sorted((mapping[x, y], mapping[x + 1, y])))
              for x in range(lx - 1) for y in range(ly)]
             + [tuple(sorted((mapping[x, y], mapping[x, y + 1])))
                for x in range(lx) for y in range(ly - 1)])
    vectors, factors = [], []
    for j in range(lx * ly):
        x, y = coords[j]
        theta = np.pi / 2 + .35 * (-1)**(x + y)
        phi = .7 + .3 * np.cos(np.pi * x / (lx - 1))
        vectors.append(np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)]))
        factors.append(.5 * np.array([1., np.sin(theta) * np.cos(phi),
                                      np.sin(theta) * np.sin(phi), np.cos(theta)]))
    cores = []
    for j, factor in enumerate(factors):
        a = np.zeros((2, 2, 4))
        a[0, 0] = factor
        a[1, 1, 0] = .5
        if j == 0:
            a = np.einsum("a,abp->bp", [weight, 1 - weight], a)[None]
        elif j == lx * ly - 1:
            a = a.sum(axis=1)[:, None]
        cores.append(a)
    return dict(n=lx * ly, shape=[lx, ly], edges=edges, vectors=vectors, cores=cores,
                regions=square_lattice_dmt_regions(lx, ly, mapper=mapper, kind="plaquettes"),
                dt=dt, hx=hx, hz=hz, weight=weight)


def field_transfer(dt, hx, hz):
    """Half-step unitary's real Pauli transfer matrix, with exact I block."""
    u = expm(-.5j * dt * (hx * ref.X + hz * ref.Z))
    t = np.zeros((4, 4))
    t[0, 0] = 1.
    for i in range(1, 4):
        for j in range(1, 4):
            t[i, j] = (.5 * np.trace(ref.PAULIS[i] @ u @ ref.PAULIS[j] @ u.conj().T)).real
    return t


def zz_transfer(span, angle, *, dtype=np.float64):
    """Exact rank-four conjugation channel for exp(-i angle Z Z)."""
    if span < 2:
        raise ValueError("ZZ endpoints must be distinct.")
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    even, odd = np.diag([1., 0, 0, 1]), np.diag([0., 1, 1, 0])
    d, f = np.zeros((4, 4)), np.zeros((4, 4))
    d[3, 0] = d[0, 3] = 1.
    f[2, 1], f[1, 2] = 1., -1.
    left = np.stack([even, odd, s * d, s * f])[None]
    right = np.stack([even + c * odd, odd + c * even, f, d])[:, None]
    middle = np.einsum("ab,pq->abpq", np.eye(4), np.eye(4))
    return [a.astype(dtype) for a in [left] + [middle] * (span - 2) + [right]]


def workspace(cores, model, chi, *, truncation="paper"):
    return _PauliDMTWorkspace(_cores=cores, max_bond=chi, regions=model["regions"],
                             region_selection=model.get("region_selection", "min_cover"), check_regions=False,
                             preservation_check="warn", truncation=truncation,
                             max_refinement_sweeps=0)


def field_layer(ws, transfer):
    ws.cores[:] = [ar.do("einsum", "qp,abp->abq", transfer, a) for a in ws.cores]
    ws.environments.invalidate(0, len(ws.cores))


def exact_target(cores, start, ops):
    result = list(cores)
    for i, op in enumerate(ops, start):
        v = result[i]
        result[i] = ar.do("einsum", "abp,xyqp->axbyq", v, op).reshape(
            v.shape[0] * op.shape[0], v.shape[1] * op.shape[1], 4)
    return result


def gate(ws, edge, angle, *, update="window", truncation="paper"):
    a, b = sorted(edge)
    dtype = np.complex128 if ar.get_dtype_name(ws.cores[0]) == "complex128" else np.float64
    ops = [_array_like(op, ws.cores[0]) for op in zz_transfer(b - a + 1, angle, dtype=dtype)]
    if update == "window":
        out = ws.copy()
        direction = "right" if abs(ws.center - a) <= abs(ws.center - b) else "left"
        out.apply_window(a, ops, absorb=direction)
    elif update == "full":
        cores = exact_target(ws.cores, a, ops)
        model = dict(regions=ws.regions, region_selection=ws.region_selection)
        out = workspace(cores, model, ws.max_bond, truncation=truncation)
        out.compress_sweep()
    else:
        raise ValueError("update must be window or full")
    return out


def readout(ws, model):
    """Independent physical-index RDM contractions, outside update timing."""
    physical = [np.einsum("abp,pij->abij", _host_array(a), ref.PAULIS) for a in ws.cores]
    panels = ref.Marginals(physical)
    regions = [panels.region(tuple(sorted(r))) for r in model["regions"]]
    one = [panels.region((j,)) for j in range(model["n"])]
    e = sum(np.trace(panels.region(tuple(sorted(edge))) @ np.kron(ref.Z, ref.Z))
            for edge in model["edges"])
    e += sum(np.trace(p @ (model["hx"] * ref.X + model["hz"] * ref.Z)) for p in one)
    return dict(trace_error=float(abs(panels.trace - 1)), energy=float(e.real / model["n"]),
                local_hermiticity=float(max(np.linalg.norm(p - p.conj().T) for p in regions)),
                min_local_eigenvalue=float(min(np.linalg.eigvalsh((p + p.conj().T) / 2)[0] for p in regions)),
                max_bond=max(a.shape[1] for a in ws.cores),
                storage_mib=sum(a.nbytes for a in ws.cores) / 2**20), regions


def audit_gate(ws, edge, angle, model, *, update="window", truncation="paper"):
    """Untimed all-plaquette check against this gate's exact current target."""
    from types import SimpleNamespace

    a, b = sorted(edge)
    dtype = np.complex128 if ar.get_dtype_name(ws.cores[0]) == "complex128" else np.float64
    ops = [_array_like(op, ws.cores[0]) for op in zz_transfer(b - a + 1, angle, dtype=dtype)]
    cores = exact_target(ws.cores, a, ops)
    _, expected = readout(SimpleNamespace(cores=cores), model)
    out = gate(ws, edge, angle, update=update, truncation=truncation)
    _, actual = readout(out, model)
    return float(max(np.linalg.norm(x - y) for x, y in zip(actual, expected)))


def commuting_oracle(model, region, time):
    """Exact ZZ-only RDM from its finite neighborhood, never a global vector."""
    region = tuple(sorted(region))
    edges = [edge for edge in model["edges"] if set(edge) & set(region)]
    sites = sorted(set(region).union(*map(set, edges)))
    psi = ref.product_vector([model["vectors"][j] for j in sites])
    gate_zz = expm(-1j * time * np.kron(ref.Z, ref.Z))
    for edge in edges:
        psi = ref.apply_vector(psi, gate_zz, tuple(sites.index(j) for j in edge), len(sites))
    rho = ref.vector_marginal(psi, tuple(sites.index(j) for j in region), len(sites))
    return model["weight"] * rho + (1 - model["weight"]) * np.eye(len(rho)) / len(rho)


def static_mixture_audit(lx=6, ly=6, *, chi=96, components=128, seed=716):
    """Lossy compression against independently summed product-state RDMs."""
    model = problem(lx, ly, .1)
    rng = np.random.default_rng(seed)
    bloch = rng.normal(size=(model["n"], components, 3))
    bloch *= .9 / np.linalg.norm(bloch, axis=2, keepdims=True)
    factors = .5 * np.concatenate([np.ones((*bloch.shape[:2], 1)), bloch], axis=2)
    weights = rng.uniform(.5, 1.5, components)
    weights /= weights.sum()
    cores = []
    for j, factor in enumerate(factors):
        a = np.zeros((components, components, 4))
        a[np.arange(components), np.arange(components)] = factor
        if j == 0:
            a = np.einsum("a,abp->bp", weights, a)[None]
        elif j == model["n"] - 1:
            a = a.sum(axis=1)[:, None]
        cores.append(a)
    started = perf_counter()
    ws = workspace(cores, model, chi)
    ws.compress_sweep()
    seconds = perf_counter() - started
    row, panels = readout(ws, model)
    errors = []
    for region, actual in zip(model["regions"], panels):
        expected = np.zeros_like(actual)
        for k, weight in enumerate(weights):
            product = np.ones((1, 1), dtype=complex)
            for site in sorted(region):
                product = np.kron(product, np.einsum("p,pij->ij", factors[site, k], ref.PAULIS))
            expected += weight * product
        errors.append(np.linalg.norm(actual - expected))
    row.update(components=components, chi=chi, compression_seconds=seconds,
               max_preservation_error=float(max(errors)),
               actual_truncation=any(r.input_rank > r.final_bond for r in ws.reports),
               preservation_errors=list(ws.report().preservation_errors))
    return row


def run(model, chi, steps, *, update="window", order="fixed", truncation="paper",
        sample_every=5, max_seconds=120., checkpoint=None, dtype=np.float64, backend="numpy"):
    cores = [a.astype(dtype) for a in model["cores"]]
    if backend == "torch":
        import torch
        torch.set_num_threads(1)
        cores = [torch.from_numpy(a) for a in cores]
    elif backend != "numpy":
        raise ValueError("backend must be numpy or torch")
    ws = workspace(cores, model, chi, truncation=truncation)
    transfer = _array_like(field_transfer(model["dt"], model["hx"], model["hz"]), cores[0])
    result = dict(shape=model["shape"], chi=chi, update=update, order=order, truncation=truncation,
                  region_selection=ws.region_selection,
                  dtype=np.dtype(dtype).name, backend=backend, steps=0, status="running", step_seconds=[],
                  samples=[], warnings=[], preservation_errors=[], update_seconds=0.,
                  readout_seconds=0., max_constraint_error=None,
                  versions={k: version(k) for k in ("numpy", "scipy", "quimb", "autoray")})
    arrays = {}
    def sample(step):
        started = perf_counter()
        row, panels = readout(ws, model)
        row.update(step=step, time=step * model["dt"])
        if model["hx"] == model["hz"] == 0:
            row["commuting_oracle_error"] = float(max(np.linalg.norm(p - commuting_oracle(model, r, row["time"]))
                                                     for p, r in zip(panels, model["regions"])))
        result["samples"].append(row)
        if row["trace_error"] > 1e-8:
            message = f"Global trace error at step {step}: {row['trace_error']:.6g}; continuing."
            result["preservation_errors"].append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        arrays[str(step)] = np.stack(panels)
        result["readout_seconds"] += perf_counter() - started
    sample(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            for step in range(1, steps + 1):
                started = perf_counter()
                field_layer(ws, transfer)
                remaining = list(model["edges"])
                while remaining:
                    index = (min(range(len(remaining)), key=lambda i: min(abs(ws.center - s) for s in remaining[i]))
                             if order == "nearest" else 0)
                    edge = remaining.pop(index)
                    ws = gate(ws, edge, model["dt"], update=update, truncation=truncation)
                    for report in ws.reports:
                        result["preservation_errors"].extend(report.preservation_errors)
                        if report.constraint_error is not None:
                            result["max_constraint_error"] = max(result["max_constraint_error"] or 0., report.constraint_error)
                field_layer(ws, transfer)
                result["step_seconds"].append(perf_counter() - started)
                result["update_seconds"] = sum(result["step_seconds"])
                result["steps"] = step
                if step % sample_every == 0 or step == steps or result["update_seconds"] >= max_seconds:
                    sample(step)
                if checkpoint:
                    Path(checkpoint).write_text(json.dumps(result, indent=2) + "\n")
                    np.savez_compressed(str(checkpoint) + ".npz", **arrays)
                print(f"{model['shape']} chi={chi} {update}/{order}/{truncation} step={step} seconds={result['step_seconds'][-1]:.3f}", flush=True)
                if result["update_seconds"] >= max_seconds:
                    break
            result["status"] = "complete" if result["steps"] == steps else "time_budget"
        except (ValueError, FloatingPointError) as exc:
            result["status"], result["error"] = "failed", str(exc)
        result["warnings"] = [str(w.message) for w in caught]
    if checkpoint:
        Path(checkpoint).write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(str(checkpoint) + ".npz", **arrays,
                            **{f"core_{i}": _host_array(a) for i, a in enumerate(ws.cores)})
    return result, ws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=[6, 6])
    parser.add_argument("--chi", type=int, default=96)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--update", choices=["full", "window"], default="window")
    parser.add_argument("--order", choices=["fixed", "nearest"], default="fixed")
    parser.add_argument("--truncation", choices=["paper", "optimized"], default="paper")
    parser.add_argument("--region-selection", choices=["smallest", "min_cover", "rank_aware"], default="min_cover")
    parser.add_argument("--complex", action="store_true")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--commuting", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-window.json"))
    parser.add_argument("--plot-summary", type=Path, help="Plot a saved aggregate; skip evolution.")
    args = parser.parse_args()
    if args.plot_summary:
        plot_summary(json.loads(args.plot_summary.read_text()), args.output.with_suffix(""))
        return
    model = problem(*args.shape, args.dt, hx=0. if args.commuting else 1.05,
                    hz=0. if args.commuting else .5)
    model["region_selection"] = args.region_selection
    with threadpool_limits(limits=1):
        run(model, args.chi, args.steps, update=args.update, order=args.order,
            truncation=args.truncation, sample_every=args.sample_every,
            max_seconds=args.max_seconds, checkpoint=args.output,
            dtype=np.complex128 if args.complex else np.float64, backend=args.backend)


def plot_summary(summary, prefix):
    """Render saved measurements; no trajectories or references are rerun."""
    import matplotlib.pyplot as plt

    runs = {r["tag"]: r for r in summary["runs"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    tags = ["full96", "window96", "nearest96", "seed96", "complex96", "window128", "window192"]
    labels = ["full 96", "window 96", "nearest 96", "seeds 96", "complex 96", "window 128", "192 invalid"]
    values = [float(np.median(runs[t]["step_seconds"][1:])) for t in tags]
    bars = ax.bar(labels, values, color=[".5", "C0", "C2", "C4", "C1", "C0", "C3"])
    bars[-1].set_hatch("//")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.tick_params(axis="x", labelrotation=28)
    ax.set(ylabel="Median update seconds (steps 2+)", title="6×6 pilot cost; same thread setting")
    ax = axes[0, 1]
    for row in summary["small_exact"]["runs"]:
        if row["label"] == "nearest":
            continue
        ax.plot([s["time"] for s in row["samples"]],
                [s["exact_local_error"] for s in row["samples"]], "o-", label=row["label"])
    ax.set(xlabel="Time", ylabel="Max plaquette Frobenius error", title="3×3, χ=64: continuous-time exact reference")
    ax.legend()
    ax = axes[1, 0]
    gap = summary["convergence"]["per_plaquette_norms"]
    ax.plot(range(1, len(gap) + 1), gap, "o-", color="C4")
    ax.set(xlabel="Plaquette inventory index", ylabel="RDM Frobenius difference",
           title="6×6 at t=0.5: χ=96 versus 128 (no exact reference)")
    ax = axes[1, 1]
    samples = runs["window96-long"]["samples"]
    times = [s["time"] for s in samples]
    ax.semilogy(times, [max(s["trace_error"], 1e-16) for s in samples], "o-", label="Trace error")
    ax.axhline(1e-8, color=".6", ls=":", label="Sample warning threshold")
    ax.set(xlabel="Time", ylabel="Absolute trace error", title="6×6, χ=96: bounded continuation")
    other = ax.twinx()
    local_line, = other.plot(times, [s["min_local_eigenvalue"] for s in samples], "s--", color="C1",
                             label="Min plaquette eigenvalue")
    other.set(ylabel="Minimum plaquette eigenvalue", ylim=(0, .03))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [local_line], labels + [local_line.get_label()], loc="upper right", fontsize=8)
    fig.suptitle("Mapped-MPO DMT: speed gains, finite-size accuracy, and unresolved χ convergence")
    for suffix in ("png", "svg"):
        fig.savefig(f"{prefix}.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
