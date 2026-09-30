"""Measure exact MPO preparation without complete collection enumeration.

Run with single-thread BLAS/OpenMP settings. Forty sites contain 1,048,575
nonempty disjoint collections but only four active frontier states per cut.
Only constructed-MPO traces are used, avoiding an exponential global dense
reference. The two-site cluster expansion is exact for this disjoint model.
"""

import gc
import json
import statistics
import time
import tracemalloc

import torch

from pepsy.operators import MPOClusterProductExpansion, MPOParameter, MPOProductTerm, prepare_cluster_channels


def builder(blocks):
    size = 4*blocks
    edges = [(4*i+j, 4*i+j+2) for i in range(blocks) for j in (0, 1)]
    factors = [[MPOProductTerm.from_pauli(edge, 'ZZ', coefficient=MPOParameter('a')) for edge in edges],
               [MPOProductTerm.from_pauli((i,), 'X', coefficient=MPOParameter('b')) for i in range(size)]]
    return MPOClusterProductExpansion(size, factors, graph=(range(size), edges), cluster_size=2,
        factorization='fixed', cutoff=0., graph_assembly='exact', collection_budget=128)


def main():
    for blocks in (1, 2, 3, 10):
        for method in ('reference', 'frontier'):
            count = 2**(2*blocks)-1
            if method == 'reference' and count > 128:
                print(json.dumps(dict(sites=4*blocks, method=method, collections=count,
                                      skipped='exceeds source collection_budget=128')), flush=True)
                continue
            source = builder(blocks)
            def prepare():
                return prepare_cluster_channels(source, -.1j, {'a': .3, 'b': .2}, preparation=method)
            start = time.perf_counter()
            plan = prepare()
            preparation = time.perf_counter()-start
            gc.collect()
            tracemalloc.start()
            memory_plan = prepare()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del memory_plan
            durations = []
            for _ in range(7):
                theta = torch.tensor([.31, -.19], dtype=torch.float64, requires_grad=True)
                start = time.perf_counter()
                value = plan.trace_exp(-.1j, dict(zip('ab', theta)), normalized=True)
                gradient, = torch.autograd.grad(value.real, theta)
                durations.append(time.perf_counter()-start)
            expected = (torch.cos(.1*theta[0])*torch.cos(.1*theta[1])**2)**(2*blocks)
            expected_gradient, = torch.autograd.grad(expected, theta)
            print(json.dumps(dict(sites=4*blocks, method=method, collections=count,
                preparation_ms=1000*preparation, preparation_tracemalloc_peak_bytes=peak,
                forward_backward_ms=1000*statistics.median(durations[2:]),
                trace_absolute_error=float(abs(value.detach().real-expected.detach())),
                gradient_absolute_error=float(torch.linalg.norm(gradient-expected_gradient)),
                input_max_bond=max(plan.report['input_bond_dimensions']),
                output_max_bond=max(plan.report['bond_dimensions']),
                max_frontier_states=max(plan.report.get('frontier_state_counts', (0,))),
                projection_contractions=plan.report['projection_contractions'])), flush=True)
            del plan


if __name__ == '__main__':
    main()
