"""Measure fixed-order MPO accuracy/cost as chi increases (optional Torch).

Run with single-thread BLAS settings for comparable CPU measurements, e.g.::

    python examples/compact_cluster_channels_benchmark.py --profile-memory

The exact reference is the uncompressed cluster expansion at the same order,
not the exact global exponential. Timings exclude preparation/compilation.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from pepsy.operators import (
    MPOClusterProductExpansion, MPOParameter, MPOProductTerm,
    prepare_cluster_channels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', type=int, default=6)
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--caps', type=int, nargs='+', default=[1, 3, 6, 12, 20, 25])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--profile-memory', action='store_true')
    args = parser.parse_args()
    factors = [[MPOProductTerm.from_pauli((i, i+1), 'ZZ', coefficient=MPOParameter('a'))
                for i in range(args.sites-1)],
               [MPOProductTerm.from_pauli((i,), 'X', coefficient=MPOParameter('b'))
                for i in range(args.sites)]]
    builder = MPOClusterProductExpansion(args.sites, factors, cluster_size=args.order,
                                         factorization='fixed', cutoff=0.)
    reference = {'a': .3, 'b': .2}
    heldout = [.31, .19]
    step = -.1j
    target = builder.exp(step, dict(zip('ab', heldout))).to_mpo().to_dense()
    theta = torch.tensor(heldout, dtype=torch.float64, requires_grad=True)
    matrix = builder.exp(step, dict(zip('ab', theta))).to_mpo().to_dense()
    value = torch.trace(matrix)/2**args.sites
    target_gradient, = torch.autograd.grad(value.real+.23*value.imag, theta)
    del matrix, value, theta

    for reuse, cap in [(False, None), *((True, c) for c in args.caps), (True, None)]:
        start = time.perf_counter()
        plan = prepare_cluster_channels(builder, step, reference, max_bond=cap, structural_reuse=reuse)
        preparation = time.perf_counter()-start
        actual = plan.exp(step, dict(zip('ab', heldout))).to_dense()
        error = np.linalg.norm(actual-target)/np.linalg.norm(target)

        def forward():
            parameters = torch.tensor(heldout, dtype=torch.float64, requires_grad=True)
            trace = plan.trace_exp(step, dict(zip('ab', parameters)), normalized=True)
            return trace.real+.23*trace.imag, parameters

        durations = []
        for _ in range(args.repeats+2):
            start = time.perf_counter()
            loss, parameters = forward()
            middle = time.perf_counter()
            gradient, = torch.autograd.grad(loss, parameters)
            end = time.perf_counter()
            durations.append((1000*(middle-start), 1000*(end-middle)))
        gradient_error = float(torch.linalg.norm(gradient-target_gradient))
        del loss, parameters, gradient
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x.numel()) or x), lambda x: x):
            loss, parameters = forward()
        torch.autograd.grad(loss, parameters)
        del loss, parameters
        memory = {}
        if args.profile_memory:
            with tempfile.TemporaryDirectory() as directory:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            profile_memory=True, record_shapes=True, with_stack=True) as profile:
                    loss, parameters = forward()
                    torch.autograd.grad(loss, parameters)
                path = Path(directory)/'memory.json'
                profile.export_memory_timeline(str(path), device='cpu')
                _, sizes = json.loads(path.read_text())
                totals = [sum(row) for row in sizes]
                memory = {'profiled_peak_tensor_bytes': max(totals),
                          'profiled_initial_tensor_bytes': totals[0]}
                del loss, parameters
        print(json.dumps(dict(
            sites=args.sites, cluster_order=args.order, max_bond=cap, structural_reuse=reuse,
            preparation_ms=1000*preparation, bond_dimensions=plan.report['bond_dimensions'],
            symbolic_report=plan.report['structural_reuse'],
            output_entries=plan.report['output_dense_entries'],
            peak_projection_entries=plan.report['peak_projection_entries'],
            saved_entries_with_repeats=sum(saved),
            forward_ms=statistics.median(d[0] for d in durations[2:]),
            backward_ms=statistics.median(d[1] for d in durations[2:]),
            relative_operator_error=float(error), normalized_trace_gradient_absolute_error=gradient_error,
            **memory)), flush=True)


if __name__ == '__main__':
    main()
