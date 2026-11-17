"""Compare paper DMT, shared protection and constrained bond refinement.

Run from the repository root with single-thread BLAS, e.g.:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/dmt_refinement_benchmark.py
Validation and global-error contractions are outside the compression timing.
"""

import argparse
import json
from time import perf_counter

import numpy as np

from mpo_dmt_benchmark import compression_relative_error, marginal, mixture, product_panel
from pepsy.operators import compress_mpo_dmt, square_lattice_dmt_regions


def benchmark(square, cap, sweeps):
    initial, weights, local = mixture(square*square, 136, 'complex128')
    regions = square_lattice_dmt_regions(square, square)
    reference = [product_panel(weights, local, r) for r in regions]
    for method, selection in [('paper', 'smallest'), ('paper', 'min_cover'),
                              ('optimized', 'min_cover')]:
        start = perf_counter()
        result, report = compress_mpo_dmt(
            initial, max_bond=cap, regions=regions, truncation=method,
            region_selection=selection, max_refinement_sweeps=sweeps, return_report=True,
        )
        seconds = perf_counter()-start
        defect = max(float(np.linalg.norm(marginal(result,r)-p)) for r,p in zip(regions,reference))
        error = compression_relative_error(result, weights, local)
        counts = {}
        for bond in report.bond_reports:
            counts[bond.truncation_method] = counts.get(bond.truncation_method, 0) + 1
        row = dict(square=square, chi=cap, initial_rank=136, truncation=method,
                   region_selection=selection, max_refinement_sweeps=sweeps,
                   seconds=seconds, max_density_defect=defect,
                   relative_hs_error=error, final_max_bond=result.max_bond(), methods=counts)
        print(json.dumps(row), flush=True)
        assert result.max_bond() <= cap and defect < 1e-10, row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--square', type=int, choices=[6, 8], nargs='+', default=[6, 8])
    parser.add_argument('--refinement-sweeps', type=int, default=2)
    args = parser.parse_args()
    if args.refinement_sweeps < 0:
        parser.error('--refinement-sweeps must be nonnegative')
    for square in args.square:
        benchmark(square, {6:96, 8:128}[square], args.refinement_sweeps)
