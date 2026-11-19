"""Benchmark 3/4/5-site DMT protection and larger chains/bonds in complex128.

Compression uses positive product mixtures and an independent oracle for
every protected window. Dynamics uses an exact statevector under the same
Ising gates; every protection width is assessed with the same five-site RDMs.
Timing excludes independent readout. Full-region verification can be enabled
explicitly; warning mode and bond checks are retained by default.
"""

import argparse
import cProfile
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
import warnings

import autoray as ar
import numpy as np
from threadpoolctl import threadpool_limits

from pepsy.operators import PauliMPO, compress_mpo_dmt
from pepsy.optimizers import MpsOptimizer, PauliDMT

try:
    from examples import dmt_ising_dynamics as bench
    from examples.pauli_dmt_benchmark import _mixture
except ModuleNotFoundError:
    import dmt_ising_dynamics as bench
    from pauli_dmt_benchmark import _mixture


def windows(length, width):
    return tuple(tuple(range(j, j + width)) for j in range(length - width + 1))


def native(a, backend):
    if backend == "numpy":
        return np.asarray(a, dtype=np.complex128)
    import torch
    return torch.tensor(np.array(a, copy=True), dtype=torch.complex128)


def check_storage(arrays, backend):
    assert {ar.infer_backend(a) for a in arrays} == {backend}
    assert {ar.get_dtype_name(a) for a in arrays} == {"complex128"}


def mixture_window(weights, local, region):
    """Physical product-mixture oracle, independent of MPO trace maps."""
    value = weights[:, None, None].astype(complex)
    for site in region:
        density = np.einsum("rp,pij->rij", local[:, site], bench.PAULIS)
        size = value.shape[1] * 2
        value = np.einsum("rab,rcd->racbd", value, density).reshape(len(weights), size, size)
    return value.sum(axis=0)


def gated_window(before, gate, where, region):
    """Physical local oracle; at most width+1 sites for overlapping gates."""
    if not set(region).intersection(where):
        return before.region(region)
    union = tuple(sorted(set(region) | set(where)))
    eye = np.eye(2**len(union))
    positions = tuple(union.index(s) for s in where)
    u = np.column_stack([bench.apply_vector(v, gate, positions, len(union)) for v in eye.T])
    target = u @ before.region(union) @ u.conj().T
    return bench.partial_trace(target, tuple(union.index(s) for s in region), len(union))


def profile_rows(profiler):
    return [{"function": f"{Path(file).name}:{line}:{name}", "calls": calls,
             "self_seconds": own, "cumulative_seconds": cumulative}
            for (file, line, name), (_, calls, own, cumulative, _) in
            sorted(_profile_stats(profiler),
                   key=lambda item: -item[1][3])[:20]]


def _profile_stats(profiler):
    import pstats
    return pstats.Stats(profiler).stats.items()


def compression_case(length, chi, width, *, backend="torch", method="pauli", repeats=3,
                     check_regions=False, profile=False):
    # Input rank exceeds the requested cap: timing must include actual loss.
    rank = chi + max(8, chi // 4)
    initial, weights, local = _mixture(length, rank, "complex128", strength=.2)
    if backend != "numpy":
        initial = PauliMPO.from_pauli_cores([native(a, backend)
                                           for a in initial.to_pauli_cores(copy=False)])
    regions = windows(length, width)
    options = {"max_bond": chi, "cutoff": 0., "return_report": True,
               "regions": None if width == 3 else regions,
               "check_regions": check_regions, "preservation_check": "warn"}
    if method == "pauli":
        def run():
            return initial.compress_dmt(**options)
    else:
        initial = initial.to_mpo()

        def run():
            return compress_mpo_dmt(initial, **options)
    run()  # dispatch/BLAS warmup, outside timing
    seconds = []
    for _ in range(repeats):
        start = perf_counter()
        result, report = run()
        seconds.append(perf_counter() - start)
    stats = []
    if profile:
        profiler = cProfile.Profile()
        profiler.runcall(run)
        stats = profile_rows(profiler)
    check_storage(result.to_pauli_cores(copy=False) if method == "pauli"
                  else [t.data for t in result], backend)
    cores = bench.pauli_cores(result) if method == "pauli" else bench.mpo_cores(result)
    actual = bench.Marginals(cores)
    error, minimum, hermiticity = 0., 0., 0.
    for region in regions:
        density = actual.region(region)
        target = mixture_window(weights, local, region)
        error = max(error, float(np.linalg.norm(density - target)))
        hermiticity = max(hermiticity, float(np.linalg.norm(density - density.conj().T)))
        minimum = min(minimum, float(np.linalg.eigvalsh((density + density.conj().T) / 2).min()))
    bonds = result.pauli_bond_dimensions if method == "pauli" else result.bond_sizes()
    assert max(bonds) <= chi
    return {"case": "compression", "method": method, "backend": backend, "dtype": "complex128",
            "length": length, "chi": chi, "width": width, "input_rank": rank,
            "check_regions": check_regions, "diagnostics": True, "preservation_check": "warn",
            "median_seconds": float(np.median(seconds)), "sample_seconds": seconds,
            "final_max_bond": max(bonds), "checked_windows": len(regions),
            "rank_reduced_bonds": sum(b.input_rank > b.final_bond for b in report.bond_reports),
            "max_protected_rank_sum": max(b.protected_left_rank + b.protected_right_rank
                                          for b in report.bond_reports),
            "max_window_frobenius_error": error, "trace_defect": float(abs(actual.trace - 1)),
            "minimum_window_eigenvalue": minimum, "max_window_hermiticity_defect": hermiticity,
            "preservation_errors": report.preservation_errors, "profile": stats}


def dynamics_case(length, chi, width, *, backend="torch", steps=20, dt=.15, method="pauli",
                  truncation="paper"):
    hx, hz = bench.MODELS["tilted"]
    vectors = bench.initial_vectors(length, "near-y-figure")
    gates = bench.sweep_gates(bench.bond_hamiltonians(length, hx, hz), dt)
    native_gates = [(native(g, backend), w) for g, w in gates]
    psi = bench.product_vector(vectors)
    regions = windows(length, width) if method != "mps" else ()
    if method == "mps":
        import quimb.tensor as qtn
        initial = qtn.MPS_product_state(vectors)
        initial.apply_to_arrays(lambda a: native(a, backend))
        opt = MpsOptimizer(initial, native_gates, chi=chi, mode="direct", contraction_opt="greedy")
    else:
        initial = PauliMPO.from_pauli_cores([
            native(np.einsum("pij,ji->p", bench.PAULIS, np.outer(v, v.conj())) / 2, backend)
            .reshape(1, 1, 4) for v in vectors])
        opt = PauliDMT(initial, max_bond=chi, cutoff=0.,
                       regions=None if width == 3 else regions,
                       check_regions=False, preservation_check="warn",
                       truncation=truncation, max_refinement_sweeps=0)
    samples, elapsed, reduced, errors, audit_error = [], 0., 0, [], 0.
    sample_steps = {0, steps // 4, steps // 2, 3 * steps // 4, steps}

    def read(step):
        if method == "mps":
            approximate = np.asarray(opt.p.to_dense()).reshape(-1)
            trace = np.vdot(approximate, approximate)

            def density(region):
                return bench.vector_marginal(approximate, region, length) / trace
        else:
            panels = bench.Marginals(bench.pauli_cores(opt.state))
            trace = panels.trace

            def density(region):
                return panels.region(region) / trace
        row = {"time": step * dt, "trace_defect": float(abs(trace - 1))}
        for size in (3, 5):
            differences, minimum = [], 0.
            for region in windows(length, size):
                value = density(region)
                target = bench.vector_marginal(psi, region, length) / np.vdot(psi, psi)
                differences.append(np.linalg.norm(value - target)**2)
                minimum = min(minimum, float(np.linalg.eigvalsh((value + value.conj().T) / 2).min()))
            row[f"rdm{size}_rmse"] = float(np.sqrt(np.mean(differences)))
            row[f"rdm{size}_min_eigenvalue"] = minimum
        samples.append(row)

    read(0)
    for step in range(1, steps + 1):
        if method == "mps":
            start = perf_counter()
            opt.run(cutoff=0., stabilize_unitary=True)
            elapsed += perf_counter() - start
        for i, (gate, where) in enumerate(gates):
            if method != "mps":
                # Audit every gate in both first and final steps, untimed.
                before = bench.Marginals(bench.pauli_cores(opt.state)) if step in (1, steps) else None
                start = perf_counter()
                bond = opt.apply_gate(native_gates[i][0], where)
                elapsed += perf_counter() - start
                reduced += int(bond.discarded_weight * 2**(length / 2) > 1e-12)
                errors.extend(bond.preservation_errors)
                if before is not None:
                    after = bench.Marginals(bench.pauli_cores(opt.state))
                    for region in regions:
                        target = gated_window(before, gate, where, region)
                        audit_error = max(audit_error, float(np.linalg.norm(after.region(region) - target)))
            psi = bench.apply_vector(psi, gate, where, length)
        if step in sample_steps:
            read(step)
    bonds = opt.p.bond_sizes() if method == "mps" else opt.state.pauli_bond_dimensions
    check_storage([t.data for t in opt.p] if method == "mps"
                  else opt.state.to_pauli_cores(copy=False), backend)
    assert max(bonds) <= chi
    return {"case": "dynamics", "method": method, "backend": backend, "dtype": "complex128",
            "length": length, "chi": chi, "width": width if method != "mps" else None,
            "steps": steps, "dt": dt, "time_units": "Pauli ZZ coupling = 1",
            "truncation": truncation if method != "mps" else None,
            "max_refinement_sweeps": 0 if method != "mps" else None,
            "evolution_seconds": elapsed, "final_max_bond": max(bonds),
            "lossy_splits": reduced if method != "mps" else None,
            "max_audited_protected_window_error": audit_error if method != "mps" else None,
            "preservation_errors": errors, "samples": samples}


def large_dynamics_case(length, chi, width, *, backend="torch", steps=2, dt=.1,
                        truncation="paper"):
    """Time fully occupied MPO bonds and audit sampled gates on long chains."""
    rho, _, _ = _mixture(length, chi + 8, "complex128", strength=.2)
    if backend != "numpy":
        rho = PauliMPO.from_pauli_cores([native(a, backend) for a in rho.to_pauli_cores(copy=False)])
    regions = windows(length, width)
    protection = None if width == 3 else regions
    rho = rho.compress_dmt(max_bond=chi, cutoff=0., regions=protection,
                           check_regions=False, preservation_check="warn",
                           truncation=truncation, max_refinement_sweeps=0)
    opt = PauliDMT(rho, max_bond=chi, cutoff=0., regions=protection,
                   check_regions=False, preservation_check="warn",
                   truncation=truncation, max_refinement_sweeps=0)
    hx, hz = bench.MODELS["tilted"]
    gates = bench.sweep_gates(bench.bond_hamiltonians(length, hx, hz), dt)
    native_gates = [(native(g, backend), w) for g, w in gates]
    seconds, errors, reduced, audit_error, checked = [], [], 0, 0., 0
    selected = {0, (length - 1) // 2, length - 2, len(gates) - 1}
    for step_index in range(steps):
        elapsed = 0.
        for i, (g, w) in enumerate(gates):
            before = bench.Marginals(bench.pauli_cores(opt.state)) if i in selected else None
            start = perf_counter()
            report = opt.apply_gate(native_gates[i][0], w)
            elapsed += perf_counter() - start
            reduced += int(report.input_rank > report.final_bond)
            errors.extend(report.preservation_errors)
            if before is not None:
                after = bench.Marginals(bench.pauli_cores(opt.state))
                for region in regions:
                    error = np.linalg.norm(after.region(region) - gated_window(before, g, w, region))
                    audit_error = max(audit_error, float(error))
                checked += 1
        seconds.append(elapsed)
        print(json.dumps({"case": "large-dynamics-progress", "length": length, "chi": chi,
                          "width": width, "completed_steps": step_index + 1,
                          "steps": steps, "evolution_seconds": sum(seconds)}), flush=True)
    state = opt.state
    check_storage(state.to_pauli_cores(copy=False), backend)
    measured = bench.measure_mpo(bench.pauli_cores(state), hx, hz)
    return {"case": "large-dynamics", "method": "pauli", "backend": backend, "dtype": "complex128",
            "length": length, "chi": chi, "width": width, "steps": steps, "dt": dt,
            "truncation": truncation, "max_refinement_sweeps": 0,
            "median_step_seconds": float(np.median(seconds)), "step_seconds": seconds,
            "gates_per_step": len(gates), "final_max_bond": max(state.pauli_bond_dimensions),
            "rank_reduced_gates": reduced, "audited_gates": checked,
            "max_audited_protected_window_error": audit_error,
            "accuracy_reference": "each sampled gate's exact local target, not a globally exact trajectory",
            "trace_defect": float(abs(measured["trace"] - 1)),
            "preservation_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("compression", "dynamics", "large-dynamics"), default="compression")
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--chis", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--widths", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--cpu-svd", choices=("torch", "scipy_gesdd", "scipy_gesvd"), default="torch")
    parser.add_argument("--stabilized-linalg", action="store_true",
                        help="use supported regularized linalg with GESVD fallback on SVD failure")
    parser.add_argument("--methods", nargs="+", choices=("pauli", "mpo", "mps"), default=["pauli"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=.15)
    parser.add_argument("--truncation", choices=("paper", "optimized"), default="paper")
    parser.add_argument("--check-regions", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-window-scaling.json"))
    args = parser.parse_args()
    if (min(args.lengths) < 5 or max(args.widths) > min(args.lengths) or min(args.widths) < 3
            or min(args.chis) < 1 or args.repeats < 1 or args.steps < 1
            or not np.isfinite(args.dt) or args.dt <= 0):
        parser.error("require L>=5, 3<=width<=L and positive chi/repeats/steps/dt")
    if args.case == "dynamics" and (max(args.lengths) > 16 or "mpo" in args.methods):
        parser.error("exact dynamics benchmark requires L<=16 and methods pauli/mps")
    if args.case == "compression" and "mps" in args.methods:
        parser.error("compression benchmark requires methods pauli/mpo")
    if args.case == "large-dynamics" and args.methods != ["pauli"]:
        parser.error("large dynamics benchmark currently uses method pauli")
    computational_errors = (ValueError, FloatingPointError, np.linalg.LinAlgError)
    if args.backend == "torch":
        import torch
        from pepsy.backends import TorchLinalgConfig
        torch.set_num_threads(1)
        TorchLinalgConfig(cpu_svd=args.cpu_svd, stabilized=args.stabilized_linalg,
                          svd_fallback="scipy_gesvd", qr_rank_policy="native").register()
        computational_errors += (torch.linalg.LinAlgError,)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    packages = ("numpy", "scipy", "quimb", "autoray")
    if args.backend == "torch":
        packages += ("torch",)
    report = {"config": vars(args) | {"output": str(args.output)}, "threads": 1,
              "versions": {p: version(p) for p in packages},
              "eigenvalue_convention": "minimum clipped at zero; negative-part diagnostic",
              "rows": []}
    with threadpool_limits(limits=1):
        for length in args.lengths:
            for chi in args.chis:
                for method in args.methods:
                    for width in args.widths if method != "mps" else [None]:
                        try:
                            if args.case == "compression":
                                row = compression_case(length, chi, width, backend=args.backend,
                                                       method=method, repeats=args.repeats,
                                                       check_regions=args.check_regions, profile=args.profile)
                            elif args.case == "dynamics":
                                row = dynamics_case(length, chi, width, backend=args.backend,
                                                    method=method, steps=args.steps, dt=args.dt,
                                                    truncation=args.truncation)
                            else:
                                row = large_dynamics_case(length, chi, width, backend=args.backend,
                                                          steps=args.steps, dt=args.dt,
                                                          truncation=args.truncation)
                        except computational_errors as exc:
                            row = {"length": length, "chi": chi, "width": width, "method": method,
                                   "status": "failed", "error": str(exc)}
                            warnings.warn(str(row), RuntimeWarning, stacklevel=1)
                        report["rows"].append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps({k: v for k, v in row.items()
                                          if k not in ("profile", "samples", "preservation_errors")}), flush=True)


if __name__ == "__main__":
    main()
