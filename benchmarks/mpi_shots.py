"""Measure MPI shot throughput for the stabilizer-MPS backend.

Examples::

    mpiexec --oversubscribe -n 4 python benchmarks/mpi_shots.py \
        --shots 10000 --qubits 16 --depth 8
    mpiexec --oversubscribe -n 4 python benchmarks/mpi_shots.py \
        --strategy coalesced --error-rate 1e-3

The benchmark reports the slowest-rank wall time and the global shot rate.
It is intentionally a script rather than a pytest benchmark: MPI process
counts and CPU oversubscription are machine-specific.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

import pepsy


def _circuit(qubits: int, depth: int):
    hadamard = np.asarray([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    gates = []
    for layer in range(depth):
        for site in range(qubits):
            gates.append((hadamard if layer % 2 == 0 else np.eye(2), site))
        for site in range(qubits - 1):
            gates.append((cnot, (site, site + 1)))
    return gates


def _parse_workers(value):
    value = str(value).strip().lower()
    if value == "auto":
        return "auto"
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "workers must be a positive integer or 'auto'"
        ) from exc
    if workers < 1:
        raise argparse.ArgumentTypeError(
            "workers must be a positive integer or 'auto'"
        )
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument(
        "--strategy", choices=("independent", "coalesced"), default="independent"
    )
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default="auto",
        help="local workers per MPI rank (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.shots < 0 or args.qubits < 1 or args.depth < 1 or args.chi < 1:
        parser.error("shots must be nonnegative and qubits/depth/chi must be positive")
    if not 0.0 <= args.error_rate <= 1.0:
        parser.error("error-rate must be between zero and one")

    comm = MPI.COMM_WORLD
    runner = pepsy.MPIShotRunner(
        lambda: pepsy.MpsStabOptimizer(args.qubits, chi=args.chi),
        _circuit(args.qubits, args.depth),
        comm=comm,
    )
    error_model = (
        None
        if args.error_rate == 0.0
        else pepsy.PauliErrorModel.bit_flip(args.error_rate)
    )
    comm.Barrier()
    started = MPI.Wtime()
    result = runner.run(
        args.shots,
        seed=args.seed,
        error_model=error_model,
        strategy=args.strategy,
        retain="none",
        local_workers=args.workers,
        local_backend="auto",
        progress=False,
    )
    elapsed = MPI.Wtime() - started
    slowest = comm.allreduce(elapsed, op=MPI.MAX)
    completed = comm.allreduce(result.local_shots, op=MPI.SUM)
    if comm.Get_rank() == 0:
        print(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "workers": args.workers,
                    "ranks": comm.Get_size(),
                    "shots": completed,
                    "qubits": args.qubits,
                    "depth": args.depth,
                    "chi": args.chi,
                    "error_rate": args.error_rate,
                    "slowest_rank_seconds": slowest,
                    "shots_per_second": completed / slowest if slowest else None,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
