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
    bond_dim,
    seed,
    hopping,
    interaction,
    chemical_potential,
):
    """Build a Symmray FH U1U1 state and OBC MPO."""
    from pepsy.tensors import (
        SymHamiltonian,
        SymMPS,
        site_charge_from_occupations,
    )

    occupations = alternating_occupations(length)
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        int(length),
        bond_dim=int(bond_dim),
        site_charge=site_charge_from_occupations(occupations),
        seed=int(seed),
        dtype="complex128",
    )
    edges = [(site, site + 1) for site in range(int(length) - 1)]
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        edges,
        t=float(hopping),
        U=float(interaction),
        mu=float(chemical_potential),
    )
    return occupations, state, ham.to_mpo(L=int(length), compress=False)


def run_benchmark(
    *,
    length=4,
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
    include_events=False,
):
    """Run one benchmark and return a JSON-serializable result dict."""
    import pepsy

    occupations, state, mpo = build_fh_u1u1_case(
        length=length,
        bond_dim=initial_bond_dim,
        seed=seed,
        hopping=hopping,
        interaction=interaction,
        chemical_potential=chemical_potential,
    )
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
        profile=True,
    )

    start = time.perf_counter()
    opt.solve(max_sweeps=int(sweeps), sweep_sequence=sweep_sequence, tol=0.0)
    elapsed = time.perf_counter() - start

    result = {
        "case": {
            "length": int(length),
            "chi": int(chi),
            "initial_bond_dim": int(initial_bond_dim),
            "sweeps": int(sweeps),
            "sweep_sequence": str(sweep_sequence),
            "seed": int(seed),
            "occupations": [list(item) for item in occupations],
            "hopping": float(hopping),
            "interaction": float(interaction),
            "chemical_potential": float(chemical_potential),
            "local_solver": str(local_solver),
            "dense_threshold": int(dense_threshold),
            "matvec_backend": str(matvec_backend),
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
            "num_local_solve_diagnostics": len(opt.local_solve_diagnostics),
        },
        "profile": opt.profile_summary(),
    }
    if include_events:
        result["profile_events"] = list(opt.profile_diagnostics)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4)
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
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = run_benchmark(
        length=args.length,
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
