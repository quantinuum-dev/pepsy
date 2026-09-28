"""Benchmark explicit compression of an exact fixed-channel cluster MPO.

Run from the Pepsy repository root with its development environment active::

    python examples/cluster_mpo_bond_compression.py

The reference is the uncompressed operator at the chosen cluster order p,
not the global exponential. No dense operator is formed. Timed compression
excludes the separately reported Frobenius error contraction.
"""

import argparse
import json
import statistics
import time

from pepsy.operators import MPOBasis


def benchmark(*, repeats=3, bonds=(2, 4, 8, 16, 32)):
    """Return time, bond, and relative-error data for a 2x3 square lattice."""
    if repeats < 1 or not bonds or any(chi < 1 for chi in bonds):
        raise ValueError("repeats and all requested bond caps must be positive.")

    shape = (2, 3)
    sites = tuple((i, j) for i in range(shape[0]) for j in range(shape[1]))
    edges = tuple(
        (site, (site[0] + di, site[1] + dj))
        for site in sites
        for di, dj in ((1, 0), (0, 1))
        if site[0] + di < shape[0] and site[1] + dj < shape[1]
    )
    terms = [(('X', 0.2), site) for site in sites]
    terms += [(('ZZ', 0.4), edge) for edge in edges]
    basis = MPOBasis.from_terms(terms, shape=shape)
    compiled = basis.compile_graph_cluster_expansion(
        graph="square", cluster_size=2, assembly="recursive",
        factorization="fixed", cutoff=0.0,
    )
    step = -0.03j
    compiled(step)  # Warm backend dispatch; no numerical result is cached.
    construction_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        exact_cluster = compiled(step)
        construction_times.append(time.perf_counter() - start)

    results = []
    for chi in bonds:
        compression_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            exact_cluster.compress_numerical(
                max_bond=chi, cutoff=0.0,
            )
            compression_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        _, report = exact_cluster.compress_numerical(
            max_bond=chi, cutoff=0.0, estimate_error=True,
            return_report=True,
        )
        results.append({
            "chi": chi,
            "final_bond_dimensions": report.final_bond_dimensions,
            "truncated": report.truncated,
            "median_compression_seconds": statistics.median(compression_times),
            "compression_and_error_seconds": time.perf_counter() - start,
            "relative_error_vs_uncompressed_cluster": float(
                report.operator_frobenius_relative_error
            ),
        })
    return {
        "shape": shape,
        "cluster_size": 2,
        "step": str(step),
        "repeats": repeats,
        "initial_bond_dimensions": exact_cluster.bond_dimensions,
        "median_construction_seconds": statistics.median(construction_times),
        "compression": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bonds", type=int, nargs="+", default=(2, 4, 8, 16, 32))
    args = parser.parse_args()
    print(json.dumps(benchmark(repeats=args.repeats, bonds=args.bonds), indent=2))


if __name__ == "__main__":
    main()
