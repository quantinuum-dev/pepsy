"""Small CPU measurement-routing benchmark; requires Pepsy stabilizer/Torch extras.

Run from the checkout: python examples/stabilizer_measurement_benchmark.py
Reports three warm-run medians after one warm-up. Resets retain normal routing.
The fixed comparison explicitly disables basis absorption for visible measurements.
"""

import json
import statistics
import time

import numpy as np
import pepsy
import torch


def stream_for(case):
    if case == 'clifford':
        return ([('h', 0)] + [('cnot', q, q + 1) for q in range(5)]
                + [('measure', 'Z', q) for q in range(6)] + [('reset', q) for q in range(6)]) * 3
    return ([('h', 0), ('t', 0), ('rxx', .31, 0, 1), ('h', 4), ('cnot', 4, 5),
             ('measure', 'X', 5), ('measure', 'Z', 0), ('reset', 0), ('reset', 5)]) * 3


reports = []
for backend in ('numpy', 'torch'):
    convert = (None if backend == 'numpy' else
               pepsy.build_backend(device='cpu', dtype=torch.complex128, set_default=False))
    for case in ('clifford', 'mixed'):
        reference = None
        for routing in ('auto', 'fixed'):
            stream = stream_for(case)
            if routing == 'fixed':
                stream = [(*entry, None, False) if entry[0] == 'measure' else entry for entry in stream]
            times = []
            for repeat in range(4):
                engine = pepsy.StabilizerMpsSimulator(6, chi=64, mode='direct', cutoff=0.,
                                                     exact_cooling=False, to_backend=convert)
                started = time.perf_counter()
                engine.compile(stream)
                compile_seconds = time.perf_counter() - started
                started = time.perf_counter()
                batch = engine.run(shots=16, strategy='independent', workers=1,
                                   seed=37, retain='all', progress=False)
                seconds = time.perf_counter() - started
                if repeat:
                    times.append(seconds)
            vectors = [sim.to_statevector() for sim in batch.optimizers]
            for vector in vectors:
                np.testing.assert_allclose(np.vdot(vector, vector).real, 1., rtol=0., atol=1e-12)
            outcomes = [[record.outcome for record in sim.measurements] for sim in batch.optimizers]
            if reference is None:
                reference = (vectors, outcomes)
            else:
                assert outcomes == reference[1]
                for actual, expected in zip(vectors, reference[0]):
                    assert abs(np.vdot(actual, expected))**2 > 1 - 1e-11
            report = dict(backend=backend, case=case, routing=routing, qubits=6, shots=16,
                          chi=64, mode='direct', workers=1, seed=37,
                          compile_seconds=compile_seconds, median_run_seconds=statistics.median(times),
                          repetitions=3, warmups=1, run_seconds=times,
                          routing_counts=batch.measurement_routing_diagnostics(),
                          max_coefficient_bond=max(sim.p.max_bond() for sim in batch.optimizers))
            reports.append(report)
            print(json.dumps(report), flush=True)
