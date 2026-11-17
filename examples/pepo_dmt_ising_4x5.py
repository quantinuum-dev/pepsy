"""Compare one 4x5 TFIM Trotter step with a state-vector gate reference.

Run with OPENBLAS_NUM_THREADS=1; see the accompanying development note for
scope, timing and the all-L-region resource limit.
"""

import argparse
import json
import time

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm

from pepsy.operators.gates import gate_simple
from pepsy.optimizers import MpoOptimizer, PepoOptimizer
from pepsy.optimizers._pepo_dmt import l_regions
from pepsy.tensors import OneDMap


def vector_gate(psi, gate, sites):
    """Apply a one- or two-site gate in row-major coordinate order."""
    n = 20
    axes = tuple(x * 5 + y for x, y in sites)
    rest = tuple(i for i in range(n) if i not in axes)
    order = axes + rest
    value = psi.reshape((2,) * n).transpose(order).reshape(2 ** len(axes), -1)
    value = (gate @ value).reshape((2,) * n).transpose(np.argsort(order))
    return value.reshape(-1)


def vector_density(psi, region):
    axes = tuple(x * 5 + y for x, y in region)
    order = axes + tuple(i for i in range(20) if i not in axes)
    value = psi.reshape((2,) * 20).transpose(order).reshape(8, -1)
    return value @ value.conj().T


def pepo_density(pepo, region):
    tensors, outputs = [], []
    for site in pepo.gen_site_coos():
        tensor = pepo[site]
        up, lo = pepo.upper_ind(*site), pepo.lower_ind(*site)
        if site in region:
            tensors.append(tensor)
            outputs.extend((up, lo))
        else:
            tensors.append(tensor.trace(up, lo, preserve_tensor=True))
    panel = qtn.tensor_contract(*tensors, output_inds=outputs, optimize="greedy").data
    k = len(region)
    axes = tuple(range(0, 2 * k, 2)) + tuple(range(1, 2 * k, 2))
    return panel.transpose(axes).reshape(2 ** k, 2 ** k) * 10 ** pepo.exponent


def mpo_density(mpo, region, index):
    positions = tuple(sorted(index[site] for site in region))
    value = np.ones(1, dtype=complex)
    for i in range(mpo.L):
        bonds = (mpo.bond(i - 1, i),) if i else ()
        bonds += (mpo.bond(i, i + 1),) if i + 1 < mpo.L else ()
        tensor = mpo[i].transpose(*(bonds + (mpo.upper_ind(i), mpo.lower_ind(i))))
        core = np.asarray(tensor.data).reshape(value.shape[-1], -1, 2, 2)
        if i in positions:
            value = np.moveaxis(np.tensordot(value, core, axes=(-1, 0)), -3, -1)
        else:
            value = value @ np.trace(core, axis1=2, axis2=3)
    value = value[..., 0]
    # The chain's site order differs from the requested row-major coordinates.
    perm = [positions.index(index[site]) for site in region]
    axes = tuple(2 * i for i in perm) + tuple(2 * i + 1 for i in perm)
    return value.transpose(axes).reshape(8, 8) * 10 ** mpo.exponent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpo-chi", type=int, default=8)
    parser.add_argument("--pepo-chi", type=int, default=4)
    parser.add_argument("--boundary-chi", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--refine-method", choices=("slsqp", "als"), default="slsqp")
    parser.add_argument("--als-sweeps", type=int, default=4)
    parser.add_argument("--weighted-regions", action="store_true")
    parser.add_argument("--probe-lossy", action="store_true")
    parser.add_argument("--compare-refinements", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.no_refine and args.weighted_regions:
        parser.error("--weighted-regions requires refinement")
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1., -1.]).astype(complex)
    local = (np.eye(2) + x) / 2
    dt, field = .05, .7
    zz, xgate = expm(1j * dt * np.kron(z, z)), expm(1j * dt * field * x)
    edges = [((i, j), (i, j + 1)) for i in range(4) for j in range(4)]
    edges += [((i, j), (i + 1, j)) for i in range(3) for j in range(5)]
    gates = [(zz, edge) for edge in edges]
    gates += [(xgate, ((i, j),)) for i in range(4) for j in range(5)]
    gates *= args.steps
    protected = ((1, 1), (1, 2), (2, 1))
    weighted = [(((0, 0), (0, 1)), 1.), (((2, 2), (2, 3)), 2.)] if args.weighted_regions else None
    regions = l_regions((4, 5))
    _, index = OneDMap(4, 5, mode="snake-row-major").build()
    mpo_regions = [tuple(sorted(index[s] for s in protected))]
    mpo_gates = [(g.T, tuple(index[s] for s in sites)) for g, sites in gates]

    start = time.perf_counter()
    vector = np.full(2 ** 20, 2 ** -10, dtype=complex)
    for gate, sites in gates:
        vector = vector_gate(vector, gate, sites)
    vector_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mpo_opt = MpoOptimizer(qtn.MPO_product_operator([local] * 20), mpo_gates,
                           chi=args.mpo_chi, mode="dmt", contraction_opt="greedy")
    mpo = mpo_opt.run(cutoff=0., dmt_regions=mpo_regions, dmt_diagnostics=True,
                      fidelity_samples=0)
    mpo_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pepo_opt = PepoOptimizer(qtn.PEPO_product_operator([[local] * 5 for _ in range(4)]),
                             gates, chi=args.pepo_chi, mode="dmt", regions=[protected],
                             boundary_chi=args.boundary_chi, dmt_max_map_elements=8000000,
                             dmt_refine=not args.no_refine, dmt_refine_method=args.refine_method,
                             dmt_refine_sweeps=args.als_sweeps, dmt_objective_regions=weighted)
    pepo = pepo_opt.run()
    pepo_seconds = time.perf_counter() - start

    exact = {region: vector_density(vector, region) for region in regions}
    start = time.perf_counter()
    mpo_panels = {r: mpo_density(mpo, r, index) for r in regions}
    pepo_panels = {r: pepo_density(pepo, r) for r in regions}
    mpo_error = {r: float(np.linalg.norm(mpo_panels[r] - exact[r])) for r in regions}
    pepo_error = {r: float(np.linalg.norm(pepo_panels[r] - exact[r])) for r in regions}
    readout_seconds = time.perf_counter() - start
    def panel_checks(panels):
        return {
            "max_trace_error": max(float(abs(np.trace(p) - 1)) for p in panels.values()),
            "max_hermiticity_error": max(float(np.linalg.norm(p - p.conj().T)) for p in panels.values()),
            "min_hermitian_eigenvalue": min(float(np.linalg.eigvalsh((p + p.conj().T) / 2)[0])
                                            for p in panels.values()),
        }
    pepo_records = pepo_opt.get_step_records()
    mpo_records = mpo_opt.get_dmt_reports()
    bonds = [b for report in mpo_records for b in report.bond_reports]
    report = {
        "model": "4x5 OBC TFIM: H=-sum(ZZ)-0.7 sum(X)",
        "dt": dt, "trotter_steps": args.steps, "gates": len(gates), "protected": protected,
        "regions_checked": len(regions),
        "seconds": {"reference": vector_seconds, "mpo": mpo_seconds,
                    "pepo": pepo_seconds, "readout": readout_seconds},
        "mpo": {"chi": args.mpo_chi, "actual_max_bond": mpo.max_bond(),
                "selected_frobenius_error": mpo_error[protected],
                "max_l_frobenius_error": max(mpo_error.values()),
                "local_physicality": panel_checks(mpo_panels),
                "max_reported_constraint_error": max((b.constraint_error or 0. for b in bonds), default=0.),
                "max_reported_discarded_weight": max((b.discarded_weight for b in bonds), default=0.)},
        "pepo": {"chi": args.pepo_chi, "boundary_chi": args.boundary_chi,
                 "actual_max_bond": pepo.max_bond(),
                 "refinement_enabled": not args.no_refine,
                 "refinement_method": args.refine_method if not args.no_refine else None,
                 "weighted_regions": weighted,
                 "als_updates": sum(r["als_updates"] for r in pepo_records),
                 "selected_frobenius_error": pepo_error[protected],
                 "max_l_frobenius_error": max(pepo_error.values()),
                 "local_physicality": panel_checks(pepo_panels),
                 "max_reported_map_residual": max(r["max_density_residual"] for r in pepo_records),
                 "refined_gates": sum(r["objective_refined"] for r in pepo_records),
                 "max_pair_error_improvement": max(r["seed_pair_error"] - r["pair_error"]
                                                   for r in pepo_records),
                 "max_pair_error": max(r["pair_error"] for r in pepo_records)},
    }
    if args.probe_lossy or args.compare_refinements:
        # Compare the compressed state with an independently formed exact
        # two-site gate target from the already-compressed PEPO.
        target = pepo.copy()
        gauges = {}
        gate_simple(target, zz, where=edges[0], gauges=gauges, max_bond=64,
                    cutoff=0., renorm=False)
        target.gauge_simple_insert(gauges)
        choices = ("none", "als", "slsqp") if args.compare_refinements else (
            "none" if args.no_refine else args.refine_method,
        )
        target_panels = {region: pepo_density(target, region) for region in regions}
        comparisons = {}
        for choice in choices:
            next_opt = PepoOptimizer(pepo, [(zz, edges[0])], chi=args.pepo_chi,
                                     mode="dmt", regions=[protected],
                                     boundary_chi=args.boundary_chi,
                                     dmt_max_map_elements=8000000, dmt_refine=choice != "none",
                                     dmt_refine_method="slsqp" if choice == "none" else choice,
                                     dmt_refine_sweeps=args.als_sweeps,
                                     dmt_objective_regions=weighted if choice != "none" else None)
            start = time.perf_counter()
            compressed = next_opt.run()
            seconds = time.perf_counter() - start
            errors = {region: float(np.linalg.norm(pepo_density(compressed, region)
                                                   - target_panels[region])) for region in regions}
            record = next_opt.get_step_records()[0]
            comparisons[choice] = {
                "seconds": seconds,
                "selected_frobenius_error": errors[protected],
                "max_l_frobenius_error": max(errors.values()),
                "max_error_region": max(errors, key=errors.get),
                "map_residual": record["max_density_residual"],
                "seed_pair_error": record["seed_pair_error"],
                "pair_error": record["pair_error"],
                "als_updates": record["als_updates"],
            }
            if weighted:
                comparisons[choice]["weighted_cost_independent"] = sum(
                    weight * float(np.linalg.norm(pepo_density(compressed, region)
                                                  - pepo_density(target, region)) ** 2)
                    for region, weight in weighted
                )
        report["lossy_refinement_comparison"] = comparisons
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
