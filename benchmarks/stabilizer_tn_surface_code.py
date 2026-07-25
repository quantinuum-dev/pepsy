"""Scalable end-to-end Stim/QEC benchmark for the two STN frontends.

The circuit is a checkerboard plaquette patch with configurable distance. It
measures commuting Z/X checks for several rounds, with detector records
comparing repeated checks and a final logical Z observable. ``distance=2``
retains the original 8-qubit four-check smoke patch; ``distance=3`` uses a
9-data-qubit grid and ``distance=5`` uses a 25-data-qubit grid. The benchmark
exercises resets, measurements, native one- and two-qubit Stim noise, detector
coordinates, coherent ZZ crosstalk, and finite-``chi`` truncation.

Run from the repository root::

    NUMBA_DISABLE_CACHING=1 python benchmarks/stabilizer_tn_surface_code.py \
        --distance 3 --shots 32 --rounds 3 --chi 1 2 4 none

The output is JSON so it can be captured by a benchmark harness.  The reported
``storage_elements`` is a tensor-storage proxy, not a process RSS measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

import pepsy


def surface_code_style_circuit(
    distance: int = 3,
    rounds: int = 2,
    depolarize1: float = 0.0001,
    depolarize2: float = 0.001,
) -> str:
    """Build a repeated-check checkerboard surface-code-style circuit.

    For ``distance >= 3``, data qubits are a ``distance x distance`` grid and
    every 2x2 plaquette gets one alternating X/Z check ancilla. The ``distance
    == 2`` branch preserves the original four-data/four-check smoke patch used
    by the fast regression test. The final data readout is included to provide
    a stable logical-observable record; this remains a benchmark patch rather
    than a full code-capacity decoder experiment.
    """
    distance = int(distance)
    if distance < 2:
        raise ValueError("distance must be at least 2.")
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("rounds must be positive.")
    depolarize1 = float(depolarize1)
    if not 0.0 <= depolarize1 <= 1.0:
        raise ValueError("depolarize1 must lie in [0, 1].")
    depolarize2 = float(depolarize2)
    if not 0.0 <= depolarize2 <= 1.0:
        raise ValueError("depolarize2 must lie in [0, 1].")

    data_count = distance * distance

    def data_site(row, column):
        return row * distance + column

    if distance == 2:
        checks = (
            (4, "Z", (0, 1), 0, 0),
            (5, "Z", (2, 3), 0, 1),
            (6, "X", (0, 2), 1, 0),
            (7, "X", (1, 3), 1, 1),
        )
    else:
        checks = []
        for row in range(distance - 1):
            for column in range(distance - 1):
                checks.append(
                    (
                        data_count + len(checks),
                        "Z" if (row + column) % 2 == 0 else "X",
                        (
                            data_site(row, column),
                            data_site(row + 1, column),
                            data_site(row, column + 1),
                            data_site(row + 1, column + 1),
                        ),
                        row,
                        column,
                    )
                )
        checks = tuple(checks)

    ancillas = tuple(check[0] for check in checks)
    lines = [
        "R " + " ".join(str(site) for site in range(data_count)),
        "R " + " ".join(str(site) for site in ancillas),
    ]
    check_count = len(checks)
    for round_index in range(rounds):
        lines.append("R " + " ".join(str(site) for site in ancillas))
        for ancilla, basis, data, row, column in checks:
            if basis == "Z":
                for data_site in data:
                    lines.append(f"CX {data_site} {ancilla}")
                    lines.append(
                        f"DEPOLARIZE2({depolarize2}) {data_site} {ancilla}"
                    )
            else:
                lines.append(f"H {ancilla}")
                lines.append(f"DEPOLARIZE1({depolarize1}) {ancilla}")
                for data_site in data:
                    lines.append(f"CX {ancilla} {data_site}")
                    lines.append(
                        f"DEPOLARIZE2({depolarize2}) {ancilla} {data_site}"
                    )
                lines.append(f"H {ancilla}")
                lines.append(f"DEPOLARIZE1({depolarize1}) {ancilla}")
            lines.append(f"M {ancilla}")

        for check_index, (ancilla, _basis, _data, row, column) in enumerate(checks):
            offset = -(check_count - check_index)
            previous = f" rec[{offset - check_count}]" if round_index else ""
            lines.append(f"DETECTOR({row}, {column}, {round_index}) rec[{offset}]{previous}")

    lines.append("M " + " ".join(str(site) for site in range(data_count)))
    logical_targets = " ".join(
        f"rec[-{data_count - column}]" for column in range(distance)
    )
    lines.append(f"OBSERVABLE_INCLUDE(0) {logical_targets}")
    return "\n".join(lines) + "\n"


def _max_bond(optimizer) -> int:
    for candidate in (
        getattr(getattr(optimizer, "state", None), "p", None),
        getattr(optimizer, "p", None),
    ):
        method = getattr(candidate, "max_bond", None)
        if callable(method):
            value = method()
            return 1 if value is None else int(value)
    raise TypeError("optimizer does not expose a coefficient-state max_bond().")


def _storage_elements(optimizer) -> int:
    """Return a backend-independent tensor storage proxy."""
    network = getattr(getattr(optimizer, "state", None), "p", None)
    if network is None:
        network = getattr(optimizer, "p", None)
    tensors = getattr(network, "tensors", ())
    return int(sum(np.prod(tuple(tensor.shape), dtype=int) for tensor in tensors))


def _state_norm(optimizer) -> float:
    method = getattr(optimizer, "norm", None)
    if not callable(method):
        raise TypeError("optimizer does not expose norm().")
    return float(method())


def _fidelity(left, right) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0.0:
        return 0.0
    return float(abs(np.vdot(left, right)) / denominator)


def _outcome_histogram(result) -> dict[str, int]:
    values = []
    for shot in result.measurements:
        values.append("".join("1" if record.outcome < 0 else "0" for record in shot))
    return dict(sorted(Counter(values).items()))


def _record_histogram(records) -> dict[str, int]:
    values = ["".join("1" if record.value else "0" for record in shot) for shot in records]
    return dict(sorted(Counter(values).items()))


def _comparison_metrics(result, reference, *, compare_dense: bool) -> dict[str, float | None]:
    """Compare replay records and optionally dense states to a same-seed reference."""
    shots = max(1, len(result.optimizers))
    syndrome_mismatches = sum(
        left != right for left, right in zip(result.syndromes, reference.syndromes)
    )
    observable_mismatches = sum(
        left != right for left, right in zip(result.observables, reference.observables)
    )
    fidelities = []
    if compare_dense:
        fidelities = [
            _fidelity(left.to_statevector(), right.to_statevector())
            for left, right in zip(result.optimizers, reference.optimizers)
        ]
    return {
        "reference_syndrome_mismatch_rate": syndrome_mismatches / shots,
        "reference_observable_mismatch_rate": observable_mismatches / shots,
        "minimum_state_fidelity": min(fidelities, default=None),
    }


def _benchmark_stim_replay(optimizer_cls, plan, shots, chi, seed):
    start = time.perf_counter()
    result = pepsy.run_stim_shots(
        lambda: optimizer_cls(plan.num_qubits, chi=chi),
        plan,
        shots=shots,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    return {
        "optimizer": optimizer_cls.__name__,
        "chi": chi,
        "shots": shots,
        "wall_time_s": elapsed,
        "shots_per_second": 0.0 if elapsed == 0.0 else shots / elapsed,
        "max_bond": max((_max_bond(opt) for opt in result.optimizers), default=1),
        "storage_elements": max(
            (_storage_elements(opt) for opt in result.optimizers), default=0
        ),
        "minimum_norm": min((_state_norm(opt) for opt in result.optimizers), default=1.0),
        "measurement_histogram": _outcome_histogram(result),
        "syndrome_histogram": _record_histogram(result.syndromes),
        "observable_histogram": _record_histogram(result.observables),
        "result": result,
    }


def _benchmark_crosstalk(optimizer_cls, plan, shots, chi, seed, theta):
    """Replay the compiled samples after inserting coherent ZZ rotations."""
    model = pepsy.CoherentCrosstalkModel(theta)
    child_seeds = np.random.SeedSequence(seed).spawn(int(shots))
    optimizers = []
    samples = []
    start = time.perf_counter()
    for child_seed in child_seeds:
        noise_seed, optimizer_seed, crosstalk_seed = child_seed.spawn(3)
        sample = pepsy.sample_stim_circuit(plan, seed=noise_seed)
        stream = model.transform(sample.gate_stream, seed=crosstalk_seed)
        optimizer = optimizer_cls(plan.num_qubits, chi=chi, seed=optimizer_seed)
        optimizer.set_gates(stream)
        optimizer.run()
        optimizers.append(optimizer)
        samples.append(sample)
    result = pepsy.StimShotResult(tuple(optimizers), tuple(samples), plan)
    elapsed = time.perf_counter() - start
    return {
        "optimizer": optimizer_cls.__name__,
        "chi": chi,
        "theta": theta,
        "shots": shots,
        "wall_time_s": elapsed,
        "shots_per_second": 0.0 if elapsed == 0.0 else shots / elapsed,
        "max_bond": max((_max_bond(opt) for opt in optimizers), default=1),
        "storage_elements": max(
            (_storage_elements(opt) for opt in optimizers), default=0
        ),
        "minimum_norm": min((_state_norm(opt) for opt in optimizers), default=1.0),
        "measurement_histogram": _outcome_histogram(result),
        "syndrome_histogram": _record_histogram(result.syndromes),
        "observable_histogram": _record_histogram(result.observables),
        "result": result,
    }


def run_benchmark(
    *,
    distance=3,
    rounds=2,
    depolarize1=0.0001,
    depolarize2=0.001,
    shots=32,
    chi_values=(1, 2, 4, None),
    seed=7,
    crosstalk_theta=0.01,
    dense_state_limit=8,
):
    """Run exact equivalence, chi convergence, and coherent-crosstalk cases.

    Dense statevector fidelity is reported only when the compiled patch has at
    most ``dense_state_limit`` qubits. Larger cases remain tensor-network-only
    and compare structured records, norms, bonds, and storage proxies.
    """
    circuit = surface_code_style_circuit(
        distance=distance,
        rounds=rounds,
        depolarize1=depolarize1,
        depolarize2=depolarize2,
    )
    plan = pepsy.compile_stim_circuit(circuit)
    exact = {
        optimizer_cls.__name__: _benchmark_stim_replay(
            optimizer_cls, plan, shots, None, seed
        )
        for optimizer_cls in (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer)
    }
    mps_result = exact["MpsStabOptimizer"]["result"]
    tree_result = exact["TreeStabOptimizer"]["result"]
    compare_dense = plan.num_qubits <= int(dense_state_limit)
    exact_comparison = {
        "measurements_equal": mps_result.measurements == tree_result.measurements,
        "syndromes_equal": mps_result.syndromes == tree_result.syndromes,
        "observables_equal": mps_result.observables == tree_result.observables,
        "minimum_state_fidelity": min(
            _fidelity(left.to_statevector(), right.to_statevector())
            for left, right in zip(mps_result.optimizers, tree_result.optimizers)
        ) if compare_dense else None,
        "dense_statevector_comparison": compare_dense,
    }

    convergence = []
    for chi in chi_values:
        if isinstance(chi, str) and chi.lower() == "none":
            chi = None
        for optimizer_cls in (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer):
            row = _benchmark_stim_replay(optimizer_cls, plan, shots, chi, seed)
            row.update(
                _comparison_metrics(
                    row["result"],
                    exact[optimizer_cls.__name__]["result"],
                    compare_dense=compare_dense,
                )
            )
            row.pop("result")
            convergence.append(row)

    crosstalk_runs = [
        _benchmark_crosstalk(
            optimizer_cls, plan, shots, None, seed, crosstalk_theta
        )
        for optimizer_cls in (pepsy.MpsStabOptimizer, pepsy.TreeStabOptimizer)
    ]
    crosstalk_mps = crosstalk_runs[0]["result"]
    crosstalk_tree = crosstalk_runs[1]["result"]
    crosstalk_equivalence = {
        "measurements_equal": crosstalk_mps.measurements == crosstalk_tree.measurements,
        "syndromes_equal": crosstalk_mps.syndromes == crosstalk_tree.syndromes,
        "observables_equal": crosstalk_mps.observables == crosstalk_tree.observables,
        "minimum_state_fidelity": min(
            _fidelity(left.to_statevector(), right.to_statevector())
            for left, right in zip(crosstalk_mps.optimizers, crosstalk_tree.optimizers)
        ) if compare_dense else None,
        "dense_statevector_comparison": compare_dense,
    }
    crosstalk = [
        {key: value for key, value in row.items() if key != "result"}
        for row in crosstalk_runs
    ]
    return {
        "circuit": {
            "num_qubits": plan.num_qubits,
            "distance": int(distance),
            "operations": len(plan.operations),
            "detectors": len(plan.detectors),
            "observables": len(plan.observables),
            "rounds": int(rounds),
            "depolarize1": float(depolarize1),
            "depolarize2": float(depolarize2),
        },
        "seed": seed,
        "exact_equivalence": exact_comparison,
        "exact_performance": [
            {key: value for key, value in row.items() if key != "result"}
            for row in exact.values()
        ],
        "chi_convergence": convergence,
        "coherent_crosstalk": crosstalk,
        "coherent_crosstalk_equivalence": crosstalk_equivalence,
    }


def _parse_chi(value):
    value = str(value).strip().lower()
    if value == "none":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("chi must be positive or 'none'.")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--depolarize1", type=float, default=0.0001)
    parser.add_argument("--depolarize2", type=float, default=0.001)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--chi", nargs="+", type=_parse_chi, default=(1, 2, 4, None))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crosstalk-theta", type=float, default=0.01)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive.")
    print(
        json.dumps(
            run_benchmark(
                distance=args.distance,
                rounds=args.rounds,
                depolarize1=args.depolarize1,
                depolarize2=args.depolarize2,
                shots=args.shots,
                chi_values=tuple(args.chi),
                seed=args.seed,
                crosstalk_theta=args.crosstalk_theta,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
