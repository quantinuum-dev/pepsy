"""Benchmark SymDMRG2 on small open-boundary FH U1U1 chains.

This script is intentionally lightweight: it constructs a deterministic
Symmray-backed Fermi-Hubbard MPS/MPO, runs ``pepsy.SymDMRG2`` with profiling
enabled, and emits JSON suitable for local comparison across implementation
changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def alternating_occupations(length):
    """Return a deterministic half-filled-ish spin pattern."""
    pattern = ((1, 0), (0, 1))
    return [pattern[site % 2] for site in range(int(length))]


def build_fh_u1u1_case(
    *,
    length,
    lattice_shape=None,
    mapper_mode="snake",
    bond_dim,
    seed,
    hopping,
    interaction,
    chemical_potential,
    mpo_cutoff=1e-12,
):
    """Build a Symmray FH U1U1 state and OBC or mapped-2D MPO."""
    import quimb.tensor as qtn

    from pepsy.tensors import (
        OneDMap,
        SymHamiltonian,
        SymMPS,
        site_charge_from_occupations,
    )

    mapper = None
    if lattice_shape is None:
        length = int(length)
        edges = [(site, site + 1) for site in range(length - 1)]
        compress_mpo = False
        lattice_shape_out = None
    else:
        lx, ly = (int(dim) for dim in lattice_shape)
        if lx < 1 or ly < 1:
            raise ValueError("lattice_shape dimensions must be positive.")
        length = lx * ly
        edges = tuple(qtn.edges_2d_square(lx, ly))
        mapper = OneDMap(lx, ly, mode=mapper_mode)
        compress_mpo = True
        lattice_shape_out = (lx, ly)

    occupations = alternating_occupations(length)
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        length,
        bond_dim=int(bond_dim),
        site_charge=site_charge_from_occupations(occupations),
        seed=int(seed),
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        edges,
        t=float(hopping),
        U=float(interaction),
        mu=float(chemical_potential),
    )
    mpo = ham.to_mpo(
        L=length,
        mapper=mapper,
        compress=compress_mpo,
        cutoff=float(mpo_cutoff),
    )
    return {
        "occupations": occupations,
        "state": state,
        "mpo": mpo,
        "length": length,
        "edges": tuple(edges),
        "lattice_shape": lattice_shape_out,
        "mapper_mode": None if mapper is None else str(mapper_mode),
        "mpo_compress": compress_mpo,
        "mpo_cutoff": float(mpo_cutoff),
    }


def run_benchmark(
    *,
    length=4,
    lattice_shape=None,
    mapper_mode="snake",
    chi=8,
    initial_bond_dim=4,
    sweeps=2,
    sweep_sequence="RL",
    seed=13,
    hopping=1.0,
    interaction=1.0,
    chemical_potential=0.1,
    local_solver="lanczos",
    dense_threshold=0,
    local_eig_tol=1e-10,
    local_eig_ncv=8,
    sector_enrichment="none",
    sector_enrichment_bond_dim=None,
    sector_noise=0.0,
    matvec_backend="auto",
    matvec_layout="unfused",
    norm_check="strict",
    norm_check_interval=1,
    residual_check="sampled",
    residual_check_interval=1,
    residual_check_tol=None,
    matvec_diagnostics="sampled",
    matvec_diagnostics_interval=1,
    compute_initial_energy=False,
    mpo_cutoff=1e-12,
    include_events=False,
):
    """Run one benchmark and return a JSON-serializable result dict."""
    import pepsy

    case = build_fh_u1u1_case(
        length=length,
        lattice_shape=lattice_shape,
        mapper_mode=mapper_mode,
        bond_dim=initial_bond_dim,
        seed=seed,
        hopping=hopping,
        interaction=interaction,
        chemical_potential=chemical_potential,
        mpo_cutoff=mpo_cutoff,
    )
    state = case["state"]
    mpo = case["mpo"]
    opt = pepsy.SymDMRG2(
        mpo,
        state,
        chi=int(chi),
        cutoff=1e-10,
        local_solver=local_solver,
        dense_threshold=int(dense_threshold),
        local_eig_tol=float(local_eig_tol),
        local_eig_ncv=int(local_eig_ncv),
        sector_enrichment=sector_enrichment,
        sector_enrichment_bond_dim=sector_enrichment_bond_dim,
        sector_noise=float(sector_noise),
        matvec_backend=matvec_backend,
        matvec_layout=matvec_layout,
        norm_check=norm_check,
        norm_check_interval=int(norm_check_interval),
        residual_check=residual_check,
        residual_check_interval=int(residual_check_interval),
        residual_check_tol=residual_check_tol,
        matvec_diagnostics=matvec_diagnostics,
        matvec_diagnostics_interval=int(matvec_diagnostics_interval),
        compute_initial_energy=compute_initial_energy,
        profile=True,
    )

    start = time.perf_counter()
    opt.solve(max_sweeps=int(sweeps), sweep_sequence=sweep_sequence, tol=0.0)
    elapsed = time.perf_counter() - start

    result = {
        "case": {
            "length": int(case["length"]),
            "lattice_shape": (
                None
                if case["lattice_shape"] is None
                else [int(dim) for dim in case["lattice_shape"]]
            ),
            "edges": [list(edge) for edge in case["edges"]],
            "num_edges": len(case["edges"]),
            "mapper_mode": case["mapper_mode"],
            "mpo_compress": bool(case["mpo_compress"]),
            "mpo_cutoff": float(case["mpo_cutoff"]),
            "chi": int(chi),
            "initial_bond_dim": int(initial_bond_dim),
            "sweeps": int(sweeps),
            "sweep_sequence": str(sweep_sequence),
            "seed": int(seed),
            "occupations": [list(item) for item in case["occupations"]],
            "hopping": float(hopping),
            "interaction": float(interaction),
            "chemical_potential": float(chemical_potential),
            "local_solver": str(local_solver),
            "dense_threshold": int(dense_threshold),
            "matvec_backend": str(matvec_backend),
            "matvec_layout": str(matvec_layout),
            "norm_check": str(norm_check),
            "norm_check_interval": int(norm_check_interval),
            "residual_check": str(residual_check),
            "residual_check_interval": int(residual_check_interval),
            "residual_check_tol": residual_check_tol,
            "matvec_diagnostics": str(matvec_diagnostics),
            "matvec_diagnostics_interval": int(matvec_diagnostics_interval),
            "compute_initial_energy": compute_initial_energy,
            "sector_enrichment": str(sector_enrichment),
            "sector_enrichment_bond_dim": sector_enrichment_bond_dim,
            "sector_noise": float(sector_noise),
        },
        "result": {
            "energy": None if opt.energy is None else float(opt.energy),
            "converged": bool(opt.converged),
            "elapsed": float(elapsed),
            "max_bond": int(opt.state.max_bond()),
            "num_sweeps": len(opt.energies),
            "num_svd_diagnostics": len(opt.svd_diagnostics),
            "num_norm_identity_diagnostics": len(opt.norm_identity_diagnostics),
            "num_residual_diagnostics": len(opt.residual_diagnostics),
            "num_matvec_diagnostics": len(opt.matvec_diagnostic_records),
            "num_local_solve_diagnostics": len(opt.local_solve_diagnostics),
        },
        "profile": opt.profile_summary(),
        "compression": opt.compression_summary(),
    }
    if include_events:
        result["profile_events"] = list(opt.profile_diagnostics)
        result["matvec_diagnostics"] = list(opt.matvec_diagnostic_records)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument(
        "--lattice-shape",
        type=int,
        nargs=2,
        metavar=("LX", "LY"),
        default=None,
        help="Build a mapped LX-by-LY square-lattice MPO instead of a 1D chain.",
    )
    parser.add_argument("--mapper-mode", default="snake")
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--initial-bond-dim", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--sweep-sequence", default="RL")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--interaction", type=float, default=1.0)
    parser.add_argument("--chemical-potential", type=float, default=0.1)
    parser.add_argument("--local-solver", default="lanczos")
    parser.add_argument("--dense-threshold", type=int, default=0)
    parser.add_argument("--local-eig-tol", type=float, default=1e-10)
    parser.add_argument("--local-eig-ncv", type=int, default=8)
    parser.add_argument("--sector-enrichment", default="none")
    parser.add_argument("--sector-enrichment-bond-dim", type=int, default=None)
    parser.add_argument("--sector-noise", type=float, default=0.0)
    parser.add_argument("--matvec-backend", default="auto")
    parser.add_argument("--matvec-layout", default="unfused")
    parser.add_argument("--norm-check", default="strict")
    parser.add_argument("--norm-check-interval", type=int, default=1)
    parser.add_argument("--residual-check", default="sampled")
    parser.add_argument("--residual-check-interval", type=int, default=1)
    parser.add_argument("--residual-check-tol", type=float, default=None)
    parser.add_argument("--matvec-diagnostics", default="sampled")
    parser.add_argument("--matvec-diagnostics-interval", type=int, default=1)
    parser.add_argument("--compute-initial-energy", action="store_true")
    parser.add_argument("--mpo-cutoff", type=float, default=1e-12)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = run_benchmark(
        length=args.length,
        lattice_shape=args.lattice_shape,
        mapper_mode=args.mapper_mode,
        chi=args.chi,
        initial_bond_dim=args.initial_bond_dim,
        sweeps=args.sweeps,
        sweep_sequence=args.sweep_sequence,
        seed=args.seed,
        hopping=args.hopping,
        interaction=args.interaction,
        chemical_potential=args.chemical_potential,
        local_solver=args.local_solver,
        dense_threshold=args.dense_threshold,
        local_eig_tol=args.local_eig_tol,
        local_eig_ncv=args.local_eig_ncv,
        sector_enrichment=args.sector_enrichment,
        sector_enrichment_bond_dim=args.sector_enrichment_bond_dim,
        sector_noise=args.sector_noise,
        matvec_backend=args.matvec_backend,
        matvec_layout=args.matvec_layout,
        norm_check=args.norm_check,
        norm_check_interval=args.norm_check_interval,
        residual_check=args.residual_check,
        residual_check_interval=args.residual_check_interval,
        residual_check_tol=args.residual_check_tol,
        matvec_diagnostics=args.matvec_diagnostics,
        matvec_diagnostics_interval=args.matvec_diagnostics_interval,
        compute_initial_energy=args.compute_initial_energy,
        mpo_cutoff=args.mpo_cutoff,
        include_events=args.include_events,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
