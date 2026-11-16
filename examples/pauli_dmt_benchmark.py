"""Profile reproducible 1D/lattice DMT workloads with checks outside timing.

Run from an installed Pepsy checkout, for example::

    python examples/pauli_dmt_benchmark.py --case all --repeats 5 --profile

The optional ``threadpoolctl`` package controls CPU threads. JSON rows report
steady, warmed timings; evolution includes workspace creation and two steps
on adjacent MPO bonds, including for the lattice-protected case.
"""

from __future__ import annotations

import argparse
import cProfile
from importlib.metadata import version
from itertools import product
import json
import platform
import pstats
from time import perf_counter

import numpy as np

from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.optimizers import PauliDMT


# L, mixture rank, cap, physical lattice shape (or chain), evolution steps.
CASES = {
    "1d-compress": (64, 48, 16, None, 0),
    "2d-compress": (25, 96, 72, (5, 5), 0),
    "1d-evolve": (16, 16, 12, None, 2),
    "2d-evolve": (9, 32, 40, (3, 3), 2),
}


def _mixture(nsites, rank, dtype):
    """Trace-one positive product mixture and independent local factors."""
    rng = np.random.default_rng(321)
    weights = rng.uniform(.2, 1., rank)
    weights /= weights.sum()
    bloch = rng.normal(size=(rank, nsites, 3))
    bloch *= .5 / np.linalg.norm(bloch, axis=-1, keepdims=True)
    local = np.concatenate((np.ones((rank, nsites, 1)), bloch), axis=-1) / 2
    cores = [(weights[:, None] * local[:, 0])[None, :, :]]
    for site in range(1, nsites - 1):
        core = np.zeros((rank, rank, 4))
        core[np.arange(rank), np.arange(rank)] = local[:, site]
        cores.append(core)
    cores.append(local[:, -1][:, None, :])
    return PauliMPO.from_pauli_cores([core.astype(dtype) for core in cores]), weights, local


def _marginal_error(operator, weights, local, regions):
    paulis = np.stack((np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], np.diag([1, -1])))
    bases = {}
    errors = []
    for region in regions:
        size = len(region)
        if size not in bases:
            basis = []
            for factors in product(paulis, repeat=size):
                matrix = np.ones((1, 1))
                for factor in factors:
                    matrix = np.kron(matrix, factor)
                basis.append(matrix)
            bases[size] = np.stack(basis)
        panel = weights
        for site in region:
            factors = local[:, site].reshape((len(weights),) + (1,) * (panel.ndim - 1) + (4,))
            panel = panel[..., None] * factors
        expected = np.einsum('p,pij->ij', panel.sum(axis=0).reshape(-1), bases[size])
        actual = operator.partial_trace(region, keep=True).to_dense()
        errors.append(float(np.linalg.norm(actual - expected)))
    return max(errors, default=0.)


def benchmark(case, *, dtype, repeats, profile=False):
    nsites, rank, cap, shape, steps = CASES[case]
    initial, weights, local = _mixture(nsites, rank, dtype)
    regions = (square_lattice_dmt_regions(*shape, map_mode="snake-row-major")
               if shape is not None else None)
    h = np.diag([.7, -.2, .4, -.9])
    h[0, 1] = h[1, 0] = .2

    def run():
        if steps:
            driver = PauliDMT(initial, max_bond=cap, cutoff="auto", regions=regions)
            return driver.evolve([h] * (nsites - 1), dt=.03, steps=steps)
        return initial.compress_dmt(max_bond=cap, cutoff="auto", regions=regions)

    run()  # Exclude first-call compilation and dispatch setup.
    seconds = []
    for _ in range(repeats):
        start = perf_counter()
        result = run()
        seconds.append(perf_counter() - start)

    # Validation/readout can itself be expensive and is intentionally untimed.
    # Only compression has an independent initial-region oracle: subsequent
    # evolution can propagate discarded correlations into protected regions.
    trace_error = float(abs(result.trace() - 1.))
    checked_regions = regions or tuple(tuple(range(j, j + 3)) for j in range(nsites - 2))
    marginal_error = None if steps else _marginal_error(result, weights, local, checked_regions)
    if max(result.pauli_bond_dimensions) > cap or trace_error > 2e-9:
        raise RuntimeError(f"{case}: bond or trace validation failed.")
    if marginal_error is not None and marginal_error > 2e-9:
        raise RuntimeError(f"{case}: independent marginal check failed ({marginal_error:.3g}).")
    report = result.compression_report
    print(json.dumps({
        "case": case, "dtype": dtype, "nsites": nsites, "input_rank": rank,
        "max_bond": cap, "protected_regions": len(checked_regions), "steps": steps,
        "median_seconds": float(np.median(seconds)), "sample_seconds": seconds,
        "final_max_bond": max(result.pauli_bond_dimensions), "trace_error": trace_error,
        "max_marginal_frobenius_error": marginal_error,
        "max_discarded_coefficient_norm": max(b.discarded_weight for b in report.bond_reports),
    }), flush=True)
    if profile:
        profiler = cProfile.Profile()
        profiler.runcall(run)
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--dtype", choices=("float64", "complex128"), default="float64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("--repeats and --threads must be positive.")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        parser.error("this benchmark needs the optional threadpoolctl package.")
    print(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in ("numpy", "quimb", "autoray")},
        "threads": args.threads, "repeats": args.repeats,
    }), flush=True)
    with threadpool_limits(limits=args.threads):
        for case in CASES if args.case == "all" else (args.case,):
            benchmark(case, dtype=args.dtype, repeats=args.repeats, profile=args.profile)


if __name__ == "__main__":
    main()
