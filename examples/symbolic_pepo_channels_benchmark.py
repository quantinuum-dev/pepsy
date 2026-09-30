"""Compare PEPO channel preparations on small exact controls.

Run with single-threaded BLAS. Timings are medians after warming residual
geometry; preparation memory is Python/NumPy tracemalloc, not process RSS or
CUDA/backward memory. Capped gradients belong to the approximated PEPO.
"""

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np
import torch

from pepsy.operators import (
    ClusterPlan, MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    prepare_cluster_channels,
)


def builder(case):
    supports, words, order = {
        'branch': (((0, 1), (0, 2), (0, 3)), ('XZ',)*3, 4),
        'loop': (((0, 1), (1, 3), (2, 3), (0, 2)), ('XZ',)*4, 3),
        'higher_body': (((0, 2, 3),), ('XYZ',), 3),
        'crossing': (((0, 3), (1, 2)), ('XZ',)*2, 2),
    }[case]
    factors = [[MPOProductTerm.from_pauli(s, word, coefficient=MPOParameter('a'))
                for s, word in zip(supports, words)],
               [MPOProductTerm.from_pauli((i,), 'X', coefficient=MPOParameter('b')) for i in range(4)]]
    geometry = ClusterPlan.from_terms([t for f in factors for t in f], sites=4, shape=(2, 2),
                                      cyclic=case == 'crossing', cluster_size=order)
    return PEPOClusterProductExpansion.from_plan(geometry, factors, layout='square', factorization='fixed')


def evaluate(plan):
    theta = torch.tensor([.31, -.17], dtype=torch.float64, requires_grad=True)
    matrix = plan.exp(-.1j, dict(zip('ab', theta))).to_dense(optimize='greedy')
    weight = torch.arange(256, dtype=torch.float64).reshape(16, 16)/256
    gradient, = torch.autograd.grad(((matrix.real+.23*matrix.imag)*weight).sum(), theta)
    return matrix.detach().numpy(), gradient.detach().numpy()


def main(repeats, preparations):
    for case in ('branch', 'loop', 'higher_body', 'crossing'):
        source = builder(case)
        reference = {'a': .2, 'b': .3}
        source.residuals(-.1j, reference)
        exact = prepare_cluster_channels(source, -.1j, reference)
        target, target_gradient = evaluate(exact)
        for count in (1, 6):
            samples = [dict(parameters={'a': i*.1, 'b': .1-i*.03}) for i in range(count-1)]
            for cap in (1, 4, None):
                for mode in preparations:
                    def prepare():
                        return prepare_cluster_channels(source, -.1j, reference,
                            preparation=mode, max_bond=cap, samples=samples)
                    times = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        plan = prepare()
                        times.append(time.perf_counter()-start)
                    tracemalloc.start()
                    measured = prepare()
                    peak = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    del measured
                    actual, gradient = evaluate(plan)
                    replay = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        evaluate(plan)
                        replay.append(time.perf_counter()-start)
                    report = plan.report
                    print(json.dumps(dict(case=case, references=count, cap=cap, preparation=mode,
                        preparation_ms=1000*statistics.median(times), preparation_tracemalloc_bytes=peak,
                        forward_backward_ms=1000*statistics.median(replay),
                        operator_relative_error=float(np.linalg.norm(actual-target)/np.linalg.norm(target)),
                        gradient_relative_error=float(np.linalg.norm(gradient-target_gradient)/np.linalg.norm(target_gradient)),
                        input_bonds=report.get('unreduced_bond_dimensions', report['input_bond_dimensions']),
                        output_bonds=report['bond_dimensions'], sparse_blocks=report['sparse_blocks'],
                        input_sparse_blocks=report.get('unreduced_sparse_blocks', report['sparse_blocks']),
                        output_dense_entries=report['output_dense_entries'],
                        structural_reuse=report['structural_reuse'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--preparations', nargs='+', choices=('reference', 'symbolic', 'algebraic'),
                        default=('reference', 'symbolic'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    main(args.repeats, args.preparations)
