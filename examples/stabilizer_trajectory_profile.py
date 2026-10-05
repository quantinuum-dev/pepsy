"""Profile mixed magic/noise CPU replay without changing simulator settings.

Requires Pepsy stabilizer and Torch extras. Example:
python examples/stabilizer_trajectory_profile.py --output /tmp/stn-profile.json
"""

import argparse
import cProfile
from contextlib import ExitStack
from functools import wraps
import importlib.metadata
import json
from pathlib import Path
import pstats
import statistics
import time

import numpy as np
import pepsy
import torch
import quimb.tensor as qtn
from unittest.mock import patch

from pepsy.optimizers.stabilizer_tn import _tableau_measurement


def make_stream(workload):
    if workload == "measurement":
        n = 6
        stream = [
            ("h", 0), ("t", 0), ("rxx", .31, 0, 1),
            ("h", 4), ("cnot", 4, 5), ("amplitude_damping", .17, 0),
            ("x_error", .07, 5), ("measure", "X", 5),
            ("measure", "Z", 0), ("if", -1, 1, ("z", 1)),
            ("reset", 0), ("reset", 5),
        ] * 4
    else:
        n = 12
        stream = [("h", q) for q in range(n)]
        for layer in range(4):
            stream += [("t", q) for q in range(n)]
            stream += [("cnot", q, q + 1) for q in range(n - 1)]
            stream += [("rxx", .31, q, q + 1) for q in range(layer % 2, n - 1, 2)]
            stream += [("amplitude_damping", .17, layer), ("x_error", .07, n - 1)]
    return n, stream + [("measure", "Z", q) for q in range(n)]


def profile_rows(stats):
    root = Path(__file__).resolve().parents[1]
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        try:
            source = str(Path(filename).relative_to(root))
        except ValueError:
            source = filename.split("/site-packages/")[-1] if "/site-packages/" in filename else Path(filename).name
        rows.append(dict(source=source, line=line, function=name, primitive_calls=primitive,
                         calls=calls, self_seconds=own, cumulative_seconds=cumulative))
    return rows


def snapshot(batch):
    return [(tuple(record.outcome for record in sim.measurements),
             tuple(record.label for record in records), weight)
            for sim, records, weight in zip(batch.optimizers, batch.records, batch.weights)]


def timed_stages(engine, options):
    """Independent nested wall timers; restored immediately after this CPU run."""
    frames, totals = [], {}

    def wrap(original, label):
        @wraps(original)
        def timed(*args, **kwargs):
            frame = [0.]
            frames.append(frame)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                frames.pop()
                if frames:
                    frames[-1][0] += elapsed
                record = totals.setdefault(label, dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
                record["calls"] += 1
                record["inclusive_seconds"] += elapsed
                record["exclusive_seconds"] += max(0., elapsed - frame[0])
        return timed

    methods = ("_evolve_p", "_apply_dense_gate", "_apply_operator_sum", "_apply_projector",
               "_pauli_expectation", "_trajectory_kraus_probabilities", "_measurement_probabilities",
               "_make_norm_event", "_commit_norm_event", "_canonize_p", "measure")
    targets = [(pepsy.StabilizerMpsSimulator, name) for name in methods]
    targets += [(_tableau_measurement, "_product_states"), (_tableau_measurement, "_simulator"),
                (qtn.MatrixProductState, "gate_with_submpo_")]
    with ExitStack() as scope:
        for owner, name in targets:
            scope.enter_context(patch.object(owner, name, wrap(getattr(owner, name), name)))
        started = time.perf_counter()
        batch = engine.run(**options)
        seconds = time.perf_counter() - started
    assert not frames
    assert sum(row["exclusive_seconds"] for row in totals.values()) <= seconds
    return batch, seconds, totals


def run_case(backend, workload, strategy, shots):
    n, stream = make_stream(workload)
    converter = (None if backend == "numpy" else
                 pepsy.build_backend(device="cpu", dtype=torch.complex128, set_default=False))
    engine = pepsy.StabilizerMpsSimulator(
        n, chi=32, cutoff=1e-12, mode="direct", exact_cooling=False,
        to_backend=converter,
    )
    start = time.perf_counter()
    engine.compile(stream)
    compile_seconds = time.perf_counter() - start
    options = dict(shots=shots, strategy=strategy, workers=1, seed=37,
                   retain="all", progress=False)
    times = []
    for repeat in range(4):
        start = time.perf_counter()
        baseline = engine.run(**options)
        elapsed = time.perf_counter() - start
        if repeat:
            times.append(elapsed)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    profiled = engine.run(**options)
    profiler.disable()
    profile_seconds = time.perf_counter() - start
    timed, timed_seconds, stage_times = timed_stages(engine, options)
    assert snapshot(timed) == snapshot(baseline)
    assert timed.counts == baseline.counts
    assert snapshot(profiled) == snapshot(baseline)
    assert profiled.counts == baseline.counts
    assert sum(profiled.counts) == shots
    assert len(profiled.optimizers) == len(profiled.records) == len(profiled.weights)
    for sim in profiled.optimizers:
        np.testing.assert_allclose(sim.norm(), 1., rtol=0., atol=1e-11)
        # Terminal physical Z readout certifies the conditional final state.
        for q, record in enumerate(sim.measurements[-n:]):
            np.testing.assert_allclose(sim.expectation("Z", q), record.outcome, rtol=0., atol=1e-10)
    stats = pstats.Stats(profiler)
    rows = profile_rows(stats)
    incomplete = [row for row in rows if row["primitive_calls"] == 0 and row["calls"] > 0]
    stages = {
        "_trajectory_kraus_probabilities", "_dense_gate_target_norm", "_pauli_expectation",
        "_apply_dense_gate", "_apply_projector", "_apply_localizer_to_p", "_apply_submpo",
        "_canonize_p_single", "_make_norm_event", "_commit_norm_event", "_product_states",
        "certificate", "pauli_decomposition", "frame_pauli", "tensor_contract", "tensor_split",
        "_run_trajectory_entries", "_run_segmented", "_measurement_probabilities",
        "_kraus_probabilities_impl", "_normalize_trajectory_branch", "gate_with_submpo",
    }
    report = dict(
        backend=backend, workload=workload, strategy=strategy, qubits=n, entries=len(stream),
        shots=shots, seed=37, workers=1, chi=32, cutoff=1e-12, mode="direct",
        exact_cooling=False, compile_seconds=compile_seconds, run_seconds=times,
        median_run_seconds=statistics.median(times), warmups=1, repetitions=3,
        profile_seconds=profile_seconds, profiler_total_seconds=stats.total_tt,
        cprofile_incomplete_cumulative=bool(incomplete),
        cprofile_zero_primitive_functions=[row["function"] for row in incomplete],
        retained_leaves=len(profiled.optimizers), routing=profiled.measurement_routing_diagnostics(),
        max_final_coefficient_bond=max(sim.p.max_bond() for sim in profiled.optimizers),
        max_recorded_coefficient_bond=max(max(sim.bond_history) for sim in profiled.optimizers),
        timed_run_seconds=timed_seconds, wall_stages=stage_times,
        top_self=sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[:25],
        top_cumulative=sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:25],
        stages=sorted((row for row in rows if row["function"] in stages),
                      key=lambda row: row["cumulative_seconds"], reverse=True),
    )
    print(json.dumps({key: report[key] for key in ("backend", "workload", "strategy", "median_run_seconds", "profile_seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=16)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    versions = {name: importlib.metadata.version(name) for name in
                ("pepsy", "stim", "quimb", "autoray", "cotengra", "symmray", "torch")}
    reports = [run_case(backend, workload, strategy, args.shots)
               for backend in ("numpy", "torch") for workload in ("measurement", "entangling")
               for strategy in ("independent", "coalesced")]
    args.output.write_text(json.dumps(dict(versions=versions, cases=reports), indent=2) + "\n")


if __name__ == "__main__":
    main()
