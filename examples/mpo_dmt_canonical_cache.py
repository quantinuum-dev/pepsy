"""Compare persistent canonical DMT replay with full sweeps, in complex128.

All gates are onsite, so a positive product-mixture oracle stays inexpensive
at large L. Preparation is untimed; the occupied MPO bonds reach chi.
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from pepsy.optimizers import MpoOptimizer

try:
    from examples.dmt_ising_dynamics import X, Z, Marginals, mpo_cores
except ModuleNotFoundError:
    from dmt_ising_dynamics import X, Z, Marginals, mpo_cores


def benchmark(n=16, chi=32, repeats=5, seed=217):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(chi, n, 2)) + 1j * rng.normal(size=(chi, n, 2))
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    factors = vectors[..., :, None] * vectors[..., None, :].conj()
    weights = rng.uniform(.5, 1.5, chi)
    weights /= weights.sum()
    terms = [w * qtn.MPO_product_operator(list(f)) for w, f in zip(weights, factors)]
    initial = terms[0]
    for term in terms[1:]:
        initial = initial + term
    gates = [expm(-.07j * (1 + j / n) * (X + .4 * Z)) for j in range(n)]
    evolved = np.stack([np.linalg.matrix_power(g, repeats) for g in gates])
    local = np.einsum("aij,rajk,alk->rail", evolved, factors, evolved.conj())
    regions = [tuple(range(j, j + 3)) for j in range(n - 2)]
    expected = []
    for region in regions:
        rho = np.zeros((8, 8), dtype=complex)
        for weight, component in zip(weights, local):
            block = np.ones((1, 1))
            for j in region:
                block = np.kron(block, component[j])
            rho += weight * block
        expected.append(rho)
    runs = []
    for update in ("full", "canonical"):
        opt = MpoOptimizer(initial, [(g.T, (j,)) for j, g in enumerate(gates)],
                           chi=chi, mode="dmt", contraction_opt="greedy")
        opt.compress_dmt(cutoff=0., check_regions=False)
        info = opt.info_c
        seconds, stages = [], Counter()
        for _ in range(repeats):
            started = perf_counter()
            opt.run(cutoff=0., dmt_update=update, fidelity_samples=0, timing=True)
            seconds.append(perf_counter() - started)
            for name, row in opt.get_run_timing()["stages"].items():
                stages[name] += row["calls"]
            assert opt.info_c is info
        panels = Marginals(mpo_cores(opt.p))
        error = max(np.linalg.norm(panels.region(r) - truth) for r, truth in zip(regions, expected))
        runs.append(dict(update=update, seconds=seconds, median_seconds=float(np.median(seconds)),
                         stage_calls=dict(stages), max_triple_oracle_error=float(error),
                         trace_error=float(abs(panels.trace - 1)), max_bond=opt.p.max_bond(),
                         dtype=str(opt.p[0].data.dtype), info_c_stable=True))
    return dict(n=n, chi=chi, repeats=repeats, seed=seed, runs=runs,
                speedup=runs[0]["median_seconds"] / runs[1]["median_seconds"],
                versions={k: version(k) for k in ("quimb", "autoray", "numpy", "scipy")},
                accelerate_threads=os.environ.get("VECLIB_MAXIMUM_THREADS"),
                scope="Onsite complex128 replay after untimed canonical preparation; not entangling dynamics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mpo-dmt-canonical-cache.json"))
    args = parser.parse_args()
    if args.length < 3 or args.chi < 1 or args.repeats < 1:
        parser.error("Require length >= 3, chi >= 1, repeats >= 1.")
    with threadpool_limits(limits=1):
        result = benchmark(args.length, args.chi, args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
