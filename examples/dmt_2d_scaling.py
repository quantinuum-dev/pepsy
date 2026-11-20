"""Mapped 2D MPO dynamics through 4x4 with sparse, local exact references.

H = sum_<ij> ZZ + 1.05 sum_i X + .5 sum_i Z, with every lattice edge.
The second-order field/commuting-ZZ/field schedule applies onsite unitaries
exactly and compresses after each two-site gate. No global density is built.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import warnings

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from threadpoolctl import threadpool_limits

from pepsy.operators import compress_mpo_dmt

try:
    from examples import dmt_2d_dynamics as small
except ModuleNotFoundError:
    import dmt_2d_dynamics as small

ref = small.ref


def problem(lx, ly, dt, *, map_mode="snake-row-major"):
    model = small.problem(lx, ly, dt, dense_reference=False, map_mode=map_mode)
    model["gates"] = [(expm(-1j * dt * np.kron(ref.Z, ref.Z)), edge) for edge in model["edges"]]
    model["field_gate"] = expm(-.5j * dt * (model["hx"] * ref.X + model["hz"] * ref.Z))
    model["zz_phase"] = np.exp(-1j * dt * model["zz_diagonal"])
    model["zz_operators"] = [zz_operator(model["n"], edge, dt) for edge in model["edges"]]
    return model


def zz_operator(n, where, angle):
    """Exact rank-two exp(-i angle Z_i Z_j), without a gate-factor SVD."""
    a, b = sorted(where)
    cores = []
    for site in range(n):
        if site == a:
            core = np.stack([np.cos(angle) * ref.I, -1j * np.sin(angle) * ref.Z])[None]
        elif site == b:
            core = np.stack([ref.I, ref.Z])[:, None]
        elif a < site < b:
            core = np.einsum("ab,ij->abij", np.eye(2), ref.I)
        else:
            core = ref.I[None, None].copy()
        if site == 0:
            core = core[0]
        elif site == n - 1:
            core = core[:, 0]
        cores.append(core)
    return qtn.MatrixProductOperator(cores)


def field_layer(mpo, unitary):
    """Exact physical-index rotations; no bonds or singular values change."""
    for site in range(mpo.L):
        mpo.gate_upper_(unitary, site, contract=True)
        mpo.gate_lower_(unitary.conj(), site, contract=True)


def vector_step(psi, model):
    n, gate = model["n"], model["field_gate"]
    for site in range(n):
        psi = ref.apply_vector(psi, gate, (site,), n)
    # All ZZ terms commute: this diagonal multiplication is exactly their
    # uncompressed product, independent of the MPO's edge-by-edge updates.
    psi = model["zz_phase"] * psi
    for site in range(n):
        psi = ref.apply_vector(psi, gate, (site,), n)
    return psi


def local_reference(psi, model):
    weight, n = model["pure_weight"], model["n"]
    groups = dict(plaquettes=model["regions"], one=[(j,) for j in range(n)], edges=model["edges"])
    return {name: np.array([weight * ref.vector_marginal(psi, region, n)
                            + (1 - weight) * np.eye(2**len(region)) / 2**len(region)
                            for region in regions]) for name, regions in groups.items()}


def references(model, dt, steps, sample_every, *, cache=None, config=None):
    fingerprint = dict(config=config, schema=1, schedule="field-zz-field", dt=dt, steps=steps,
                       sample_every=sample_every, edges=model["edges"], hx=model["hx"], hz=model["hz"],
                       weight=model["pure_weight"], initial_sha256=hashlib.sha256(model["psi"].tobytes()).hexdigest())
    key = json.dumps(fingerprint, sort_keys=True)
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["config"]) != key:
                raise ValueError("Reference cache configuration differs from this run")
            samples = {int(step): {kind: {group: data[f"{step}_{kind}_{group}"]
                                         for group in ("plaquettes", "one", "edges")}
                                  for kind in ("exact", "circuit")} for step in data["steps"]}
            initial_energy = energy(samples[0]["exact"], model)
            return samples, dict(cache_hit=True, preparation_seconds=0.,
                                 max_energy_drift=float(max(abs(energy(s["exact"], model) - initial_energy)
                                                            for s in samples.values())),
                                 max_vector_norm_error=float(data["norm_error"]))
    started = perf_counter()
    sample_steps = sorted(set(range(0, steps + 1, sample_every)) | {steps})
    exact, circuit = model["psi"].copy(), model["psi"].copy()
    last_exact, samples, norm_error = 0, {}, 0.
    for step in range(steps + 1):
        if step:
            circuit = vector_step(circuit, model)
        if step not in sample_steps:
            continue
        if step:
            exact = expm_multiply((-1j * dt * (step - last_exact)) * model["hamiltonian"],
                                  exact, traceA=0.)
        last_exact = step
        norm_error = max(norm_error, abs(np.vdot(exact, exact) - 1), abs(np.vdot(circuit, circuit) - 1))
        samples[step] = dict(exact=local_reference(exact, model), circuit=local_reference(circuit, model))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{step}_{kind}_{group}": value for step, sample in samples.items()
                  for kind, groups in sample.items() for group, value in groups.items()}
        np.savez_compressed(cache, config=key, steps=sample_steps, norm_error=float(norm_error), **arrays)
    initial_energy = energy(samples[0]["exact"], model)
    return samples, dict(cache_hit=False, preparation_seconds=perf_counter() - started,
                         max_energy_drift=float(max(abs(energy(s["exact"], model) - initial_energy)
                                                    for s in samples.values())),
                         max_vector_norm_error=float(norm_error))


def mpo_panels(mpo, model):
    panels = ref.Marginals(ref.mpo_cores(mpo))
    groups = dict(plaquettes=model["regions"], one=[(j,) for j in range(model["n"])], edges=model["edges"])
    def region_in_order(region):
        ordered = tuple(sorted(region))
        matrix = panels.region(ordered)
        if tuple(region) != ordered:
            k = len(region)
            order = [ordered.index(site) for site in region]
            matrix = matrix.reshape((2,) * (2 * k)).transpose(order + [j + k for j in order]).reshape(2**k, 2**k)
        return matrix
    return {name: np.array([region_in_order(region) for region in regions])
            for name, regions in groups.items()}, panels


def energy(groups, model):
    xyz = np.einsum("aij,pji->ap", groups["one"], ref.PAULIS[1:])
    zz = np.einsum("aij,ji->a", groups["edges"], np.kron(ref.Z, ref.Z))
    return (zz.sum() + model["hx"] * xyz[:, 0].sum() + model["hz"] * xyz[:, 2].sum()) / model["n"]


def reference_diagnostics(samples, model, dt):
    return [dict(time=step * dt, step=step,
                 trotter_plaquette_error=float(max(np.linalg.norm(a - b) for a, b in zip(
                     sample["exact"]["plaquettes"], sample["circuit"]["plaquettes"]))),
                 trotter_energy_error_per_site=float(abs(energy(sample["exact"], model)
                                                         - energy(sample["circuit"], model))),
                 exact_energy_per_site=float(energy(sample["exact"], model).real))
            for step, sample in samples.items()]


def global_operator_norms(cores):
    """Double-chain contractions, with a cancellation floor for Hermiticity.

    This measures HS norms, not global eigenvalues or positive semidefiniteness.
    A near-zero difference of overlaps cannot resolve tiny Hermiticity defects.
    """
    norm, square = np.ones((1, 1), complex), np.ones((1, 1), complex)
    for a in cores:
        norm = np.einsum("ac,abij,cdij->bd", norm, a.conj(), a, optimize=True)
        square = np.einsum("ac,abij,cdji->bd", square, a, a, optimize=True)
    norm2, trace_square = float(norm[0, 0].real), square[0, 0]
    floor = 32 * len(cores) * max(max(a.shape[:2]) for a in cores) * np.finfo(float).eps
    if norm2 <= 0 or not np.isfinite(norm2):
        raise FloatingPointError("Global MPO HS norm is nonpositive or nonfinite")
    defect2 = 2 * (norm2 - trace_square.real) / norm2
    if defect2 < -floor:
        raise FloatingPointError("Hermiticity overlap has an inconsistent negative squared norm")
    return dict(hs_norm=float(np.sqrt(norm2)),
                global_relative_hermiticity=float(np.sqrt(max(0., defect2))),
                hermiticity_overlap_resolution=float(np.sqrt(floor)),
                global_min_eigenvalue=None)


def diagnostics(mpo, truth, model):
    actual, panels = mpo_panels(mpo, model)
    exact, circuit = truth["exact"], truth["circuit"]
    trace = panels.trace
    value, target, circuit_energy = energy(actual, model), energy(exact, model), energy(circuit, model)
    xyz = np.einsum("aij,pji->ap", actual["one"] - exact["one"], ref.PAULIS[1:])
    local = actual["plaquettes"]
    normalized_error = (max(np.linalg.norm(a / trace - b) for a, b in zip(local, exact["plaquettes"]))
                        if abs(trace) > 1e-14 else None)
    return dict(trace_real=float(trace.real), trace_imag=float(trace.imag), trace_error=float(abs(trace - 1)),
                energy_per_site_real=float(value.real), energy_per_site_imag=float(value.imag),
                energy_error_per_site=float(abs(value - target)),
                energy_circuit_error_per_site=float(abs(value - circuit_energy)),
                trotter_energy_error_per_site=float(abs(circuit_energy - target)),
                energy_error_trace_normalized=float(abs(value / trace - target)) if abs(trace) > 1e-14 else None,
                max_plaquette_error=float(max(np.linalg.norm(a - b) for a, b in zip(local, exact["plaquettes"]))),
                max_plaquette_error_trace_normalized=float(normalized_error) if normalized_error is not None else None,
                max_plaquette_circuit_error=float(max(np.linalg.norm(a - b) for a, b in zip(local, circuit["plaquettes"]))),
                trotter_plaquette_error=float(max(np.linalg.norm(a - b) for a, b in zip(exact["plaquettes"], circuit["plaquettes"]))),
                max_xyz_error=float(abs(xyz).max()),
                min_plaquette_eigenvalue=float(min(np.linalg.eigvalsh((a + a.conj().T) / 2)[0] for a in local)),
                max_plaquette_hermiticity=float(max(np.linalg.norm(a - a.conj().T) for a in local)),
                **global_operator_norms(panels.cores))


def trajectory(model, truth, *, chi, steps, dt, method="dmt", selection="smallest", fit_sweeps=4,
               max_seconds=None, checkpoint=None, progress=False, update="routed", profile_step=None):
    opt = small.AuditedOptimizer(model["initial"], [(u.T, where) for u, where in model["gates"]],
                                 chi=chi, mode="dmt" if method == "dmt" else "dmrg", model=model,
                                 contraction_opt="greedy")
    row = dict(method=method, chi=chi, region_selection=selection,
               update=update if method == "dmt" else "fit", status="running", completed_steps=0,
               update_seconds=0., readout_seconds=0., samples=[], step_seconds=[], stages={},
               warning_events=[], preservation_errors=[], field_seconds=0., max_bond_observed=0,
               profiled_steps=[], profile=[])
    audit_steps = {1, max(1, steps // 2), steps}
    for step in range(steps + 1):
        if step:
            started = perf_counter()
            audit_before, caught = opt.audit_seconds, []
            backup = opt.p.copy(deep=True)
            stage_samples = []
            profiler = None
            if step == profile_step:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    field_start = perf_counter()
                    field_layer(opt.p, model["field_gate"])
                    row["field_seconds"] += perf_counter() - field_start
                    opt.audit, opt.gate_number = step in audit_steps, 0
                    if opt.audit:
                        audit_start = perf_counter()
                        opt.audit_source = ref.Marginals([a.copy() for a in ref.mpo_cores(opt.p)])
                        opt.audit_seconds += perf_counter() - audit_start
                    options = dict(cutoff=0., target_cutoff=0., fidelity_samples=0, timing=True)
                    if method == "dmt":
                        options.update(dmt_regions=model["regions"], dmt_preservation_check="warn",
                                       dmt_region_selection=selection)
                    else:
                        options.update(n_iter=fit_sweeps, fit_init_strategy="direct", fit_rtol=None,
                                       fit_block_size=2, fit_target_strategy="layered")
                    if update == "analytic" and method == "dmt":
                        for operator in model["zz_operators"]:
                            build_start = perf_counter()
                            # ZZ unitaries are diagonal, hence conjugation is
                            # their operator adjoint. Each exact multiplication
                            # grows only by the known rank-two gate bond.
                            target = operator.apply(opt.p, compress=False).apply(operator.conj(), compress=False)
                            build_seconds = perf_counter() - build_start
                            compress_start = perf_counter()
                            candidate = compress_mpo_dmt(target, max_bond=chi, cutoff=0., regions=model["regions"],
                                                         region_selection=selection, check_regions=False,
                                                         preservation_check="warn")
                            compress_seconds = perf_counter() - compress_start
                            report = getattr(candidate, "compression_report", None)
                            if report is not None:
                                row["preservation_errors"].extend(dict(step=step, message=message, count=count)
                                    for message, count in Counter(report.preservation_errors).items())
                            opt._check_finite(False, candidate)
                            opt.set_mpo(candidate)
                            stage_samples.append({"stages": {
                                "analytic.target": dict(calls=1, elapsed_seconds=build_seconds),
                                "analytic.compress": dict(calls=1, elapsed_seconds=compress_seconds)}})
                    elif update == "full" and method == "dmt":
                        # A separate public replay call has no retained routed
                        # workspace: each exact gate target gets a full DMT
                        # sweep. This is a different, explicitly named update
                        # order, not an optimization-equivalent SWAP shortcut.
                        for unitary, where in model["gates"]:
                            opt.set_gates([(unitary.T, where)])
                            opt.run(**options)
                            stage_samples.append(opt.get_run_timing())
                    else:
                        opt.run(**options)
                        stage_samples.append(opt.get_run_timing())
                    field_start = perf_counter()
                    field_layer(opt.p, model["field_gate"])
                    row["field_seconds"] += perf_counter() - field_start
                row["completed_steps"] = step
            except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                opt.p = backup
                row.update(status="failed", failure_step=step, error=f"{type(exc).__name__}: {exc}")
            if profiler is not None:
                profiler.disable()
            elapsed = perf_counter() - started - (opt.audit_seconds - audit_before)
            if profiler is not None:
                import pstats
                stats = pstats.Stats(profiler).stats
                row["profiled_steps"].append(step)
                row["profile"] = [dict(function=f"{Path(file).name}:{line}:{name}", calls=calls,
                                        self_seconds=own, cumulative_seconds=total)
                                  for (file, line, name), (_, calls, own, total, _) in
                                  sorted(stats.items(), key=lambda item: -item[1][3])[:30]]
            row["update_seconds"] += elapsed
            row["step_seconds"].append(elapsed)
            for message, count in Counter(str(w.message) for w in caught).items():
                row["warning_events"].append(dict(step=step, message=message, count=count))
            if method == "dmt":
                failures = Counter(message for report in opt.get_dmt_reports()
                                   for message in report.preservation_errors)
                row["preservation_errors"].extend(dict(step=step, message=message, count=count)
                                                   for message, count in failures.items())
            for timing in stage_samples:
                for name, measurement in timing.get("stages", {}).items():
                    aggregate = row["stages"].setdefault(name, dict(seconds=0., calls=0))
                    aggregate["seconds"] += measurement["elapsed_seconds"]
                    aggregate["calls"] += measurement["calls"]
            if row["status"] == "failed":
                break
        row["max_bond_observed"] = max(row["max_bond_observed"], opt.p.max_bond())
        if opt.p.max_bond() > chi:
            raise AssertionError("Completed output exceeds the requested bond cap")
        if step in truth:
            readout_start = perf_counter()
            try:
                row["samples"].append(dict(time=step * dt, step=step, bond=opt.p.max_bond(),
                                           **diagnostics(opt.p, truth[step], model)))
            except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                row.update(status="failed", failure_stage="readout", error=f"{type(exc).__name__}: {exc}")
                break
            row["readout_seconds"] += perf_counter() - readout_start
        if progress and step:
            print(json.dumps(dict(method=method, chi=chi, step=step, time=step * dt,
                                  seconds=row["step_seconds"][-1], status=row["status"])), flush=True)
        if checkpoint is not None:
            checkpoint(row)
        if max_seconds is not None and row["update_seconds"] >= max_seconds and step < steps:
            row.update(status="time_budget", error="Requested update-time budget reached between timesteps")
            break
    if row["status"] == "running":
        row["status"] = "completed"
    row.update(audit_seconds=opt.audit_seconds, audited_two_site_gates=opt.audited_gates,
               max_per_gate_preservation_error=opt.max_preservation_error,
               storage_dtypes=sorted({str(t.data.dtype) for t in opt.p}),
               storage_mib=sum(t.data.nbytes for t in opt.p) / 2**20,
               process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024))
    return row


def plot_results(records, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    timestep_shapes, curves = set(), 0
    for record in records:
        shape, dt = tuple(record["config"]["shape"]), record["config"]["dt"]
        for row in record["rows"]:
            if not row["samples"] or row["completed_steps"] == 0:
                continue
            color = plt.get_cmap("tab10")(curves % 10)
            curves += 1
            style = "--" if row["method"] == "fit" else ":" if row["update"] == "routed" else "-"
            method_label = "FIT" if row["method"] == "fit" else f"DMT {row['update']}"
            label = f"{shape[0]}×{shape[1]} χ={row['chi']} {method_label}"
            samples = row["samples"]
            times = [s["time"] for s in samples]
            for axis, key in zip(axes.flat[:3], ["max_plaquette_error", "energy_error_per_site", "trace_error"]):
                axis.plot(times, np.maximum([s[key] for s in samples], 1e-16), style, color=color,
                          marker=".", markersize=4, label=label)
            axes[1, 0].plot(times, [s["min_plaquette_eigenvalue"] for s in samples], style, color=color,
                            marker=".", markersize=4)
            axes[1, 1].plot(times, [max(s["global_relative_hermiticity"], s["hermiticity_overlap_resolution"])
                                    for s in samples], style, color=color, marker=".", markersize=4)
            profiled = row.get("profiled_steps", [])
            elapsed = [(i + 1, value) for i, value in enumerate(row["step_seconds"]) if i + 1 not in profiled]
            chunks = [elapsed[i:i + 10] for i in range(0, len(elapsed), 10)]
            axes[1, 2].plot([dt * np.mean([i for i, _ in chunk]) for chunk in chunks],
                            [np.median([v for _, v in chunk]) for chunk in chunks], style, color=color)
            if (shape, dt) not in timestep_shapes:
                axes[0, 0].plot(times, np.maximum([s["trotter_plaquette_error"] for s in samples], 1e-16),
                                "-.", color=color, alpha=.5, label=f"{shape[0]}×{shape[1]} timestep error")
                timestep_shapes.add((shape, dt))
    titles = ["Raw plaquette error vs continuous exact", "Raw energy error per site", "Raw trace error",
              "Minimum local Hermitian-part eigenvalue", "Global HS Hermiticity (censored at resolution)",
              "Seconds per step (10-step medians)"]
    for i, (axis, title) in enumerate(zip(axes.flat, titles)):
        axis.set(title=title, xlabel="t (J=1)")
        axis.grid(alpha=.2)
        if i != 3:
            axis.set_yscale("log")
        else:
            axis.axhline(0, color="black", linewidth=.7)
    axes[0, 0].set_ylim(bottom=1e-4)
    axes[0, 1].set_ylim(bottom=1e-5)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("2D Ising MPO dynamics • complex128 • local exact references • no global PSD certificate")
    destination = Path(destination)
    figure.savefig(destination, dpi=180)
    svg = destination.with_suffix(".svg")
    figure.savefig(svg)
    # Matplotlib's path exporter emits trailing spaces on coordinate lines.
    svg.write_text("\n".join(line.rstrip(" \t") for line in svg.read_text().split("\n")))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--chis", type=int, nargs="+", default=[64])
    parser.add_argument("--methods", nargs="+", choices=["dmt", "fit"], default=["dmt"])
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--region-selection", choices=["smallest", "min_cover"], default="smallest")
    parser.add_argument("--map-mode", default="snake-row-major")
    parser.add_argument("--update", choices=["routed", "full", "analytic"], default="routed")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--profile-step", type=int)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-2d-scaling.json"))
    parser.add_argument("--plot-only", type=Path, nargs="+", help="Plot saved result JSONs instead of evolving")
    args = parser.parse_args()
    if args.plot_only:
        plot_results([json.loads(path.read_text()) for path in args.plot_only], args.output.with_suffix(".png"))
        return
    if min(args.steps, args.sample_every, *args.chis) < 1:
        parser.error("steps, sample-every and chis must be positive")
    if args.profile_step is not None and not 1 <= args.profile_step <= args.steps:
        parser.error("profile-step must be between 1 and steps")
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("max-seconds must be positive and finite")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    reference_config = {k: config[k] for k in ("shape", "dt", "steps", "sample_every", "map_mode")}
    result = dict(config=config, reference_config=reference_config, rows=[], dtype="complex128",
                  schedule="exact field half / all commuting ZZ edges / exact field half",
                  versions={p: version(p) for p in ["numpy", "scipy", "quimb", "autoray", "cotengra"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        model = problem(*args.shape, args.dt, map_mode=args.map_mode)
        truth, preparation = references(model, args.dt, args.steps, args.sample_every,
                                        cache=args.reference_cache, config=reference_config)
        result.update(reference=preparation, edges=model["edges"], regions=model["regions"])
        result["reference_samples"] = reference_diagnostics(truth, model, args.dt)
        if args.references_only:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return
        for method in args.methods:
            for chi in args.chis:
                def checkpoint(row):
                    partial = {**result, "rows": result["rows"] + [row]}
                    args.output.write_text(json.dumps(partial, indent=2, allow_nan=False) + "\n")
                row = trajectory(model, truth, chi=chi, steps=args.steps, dt=args.dt, method=method,
                                 selection=args.region_selection, max_seconds=args.max_seconds,
                                 checkpoint=checkpoint, progress=True, update=args.update,
                                 profile_step=args.profile_step)
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
