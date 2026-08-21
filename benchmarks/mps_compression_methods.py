"""Compare the Quimb gate-compression methods used by :class:`MpsOptimizer`.

The benchmark applies the same long-range unitary gates to a random, already
entangled MPS with every ``quimb-*`` method.  Accuracy is measured against a
dense reference produced by the same gate stream with Quimb's lazy method.
The default cases cover one end-to-end gate, one interior gate, and a short
long-range circuit.

Examples
--------

Run a quick comparison::

    source /Users/rezah/envs/genpy/bin/activate
    python benchmarks/mps_compression_methods.py --length 10 --chi 4

The reported time is the median of the measured runs after one warm-up run.
Randomized methods are run with a different reproducibility seed per repeat,
although their compression noise is intentionally part of the method.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import quimb as qu
import quimb.tensor as qtn

import pepsy


METHODS = (
    "direct",
    "dm",
    "zipup",
    "zipup-first",
    "zipup-oversample",
    "sdc",
    "sdc-oversample",
    "src",
    "src-first",
    "src-oversample",
    "srcmps",
    "srcmps-first",
    "srcmps-oversample",
    "fit",
    "fit-zipup",
    "fit-projector",
    "fit-oversample",
)


def _random_gate(seed: int):
    """Return a deterministic random two-site unitary."""

    return qu.rand_uni(4, seed=seed)


def _make_case(case: str, length: int, seed: int):
    """Construct a realistic entangled MPS and a long-range gate stream."""

    if length < 6:
        raise ValueError("length must be at least 6 for the interior case")

    state = qtn.MPS_rand_state(
        length,
        bond_dim=min(4, 2 ** (length // 2)),
        phys_dim=2,
        dtype="complex128",
        seed=seed,
    )
    if case == "endpoint":
        gates = [(_random_gate(seed + 10), (0, length - 1))]
    elif case == "interior":
        gates = [(_random_gate(seed + 10), (1, length - 2))]
    elif case == "circuit":
        gates = [
            (_random_gate(seed + 10), (0, length - 1)),
            (_random_gate(seed + 11), (1, length - 2)),
            (_random_gate(seed + 12), (2, length - 3)),
            (_random_gate(seed + 13), (0, length // 2)),
        ]
    else:
        raise ValueError(f"unknown case: {case}")
    return state, gates


def _dense_reference(state, gates):
    """Apply the gate stream without truncation and return its dense vector."""

    reference = state.copy(deep=True)
    for gate, where in gates:
        reference.gate_nonlocal_(gate, where, method="lazy")
    return np.asarray(reference.to_dense()).reshape(-1)


def _accuracy(vector, reference):
    """Return phase-aligned relative L2 error and normalized fidelity."""

    vector_norm = np.linalg.norm(vector)
    reference_norm = np.linalg.norm(reference)
    overlap = np.vdot(vector, reference)
    phase = overlap.conjugate() / abs(overlap) if abs(overlap) else 1.0
    error = np.linalg.norm(vector - phase * reference) / reference_norm
    fidelity = abs(overlap) ** 2 / (vector_norm * reference_norm) ** 2
    return float(error), float(fidelity)


def _run_once(state, gates, method, chi, cutoff, seed):
    """Apply one method through the public Pepsy optimizer API."""

    # Quimb's randomized MPO methods receive their reproducibility seed through
    # the optimizer's compression API. FIT initialization has a separate seed.
    optimizer = pepsy.MpsOptimizer(
        state.copy(deep=True),
        gates=gates,
        chi=chi,
        mode=f"quimb-{method}",
    )
    return optimizer.run(
        cutoff=cutoff,
        progbar=False,
        compression_seed=seed,
        stabilize_unitary=False,
        fit_init_seed=seed,
        collect_diagnostics=False,
    )


def benchmark_case(case, length, chi, cutoff, repeats, seed):
    """Benchmark all methods for one gate stream."""

    state, gates = _make_case(case, length, seed)
    reference = _dense_reference(state, gates)
    results = []
    for method in METHODS:
        # Avoid charging the first method-specific contraction-plan setup to
        # the reported time.
        _run_once(state, gates, method, chi, cutoff, seed)
        samples = []
        for repeat in range(repeats):
            started = perf_counter()
            try:
                result = _run_once(
                    state,
                    gates,
                    method,
                    chi,
                    cutoff,
                    seed + repeat + 1,
                )
            except Exception as exc:  # keep one method from hiding others
                results.append(
                    {
                        "case": case,
                        "method": method,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            elapsed = perf_counter() - started
            vector = np.asarray(result.to_dense()).reshape(-1)
            error, fidelity = _accuracy(vector, reference)
            samples.append(
                {
                    "seconds": elapsed,
                    "error": error,
                    "fidelity": fidelity,
                    "max_bond": int(result.max_bond()),
                }
            )
        if len(samples) != repeats:
            continue
        results.append(
            {
                "case": case,
                "method": method,
                "status": "ok",
                "seconds_median": float(np.median([x["seconds"] for x in samples])),
                "seconds_min": float(min(x["seconds"] for x in samples)),
                "seconds_max": float(max(x["seconds"] for x in samples)),
                "error_median": float(np.median([x["error"] for x in samples])),
                "fidelity_median": float(
                    np.median([x["fidelity"] for x in samples])
                ),
                "max_bond": max(x["max_bond"] for x in samples),
            }
        )
    return results


def _format_table(results):
    """Format benchmark results as a compact Markdown table."""

    lines = [
        "| method | median ms | relative L2 error | fidelity | max bond |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        if result["status"] != "ok":
            lines.append(f"| {result['method']} | ERROR | {result['error']} | | |")
            continue
        lines.append(
            "| {method} | {ms:.3f} | {error:.3e} | {fidelity:.8f} | {bond} |".format(
                method=result["method"],
                ms=1000.0 * result["seconds_median"],
                error=result["error_median"],
                fidelity=result["fidelity_median"],
                bond=result["max_bond"],
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--case",
        choices=("endpoint", "interior", "circuit", "all"),
        default="all",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the raw result records to PATH",
    )
    args = parser.parse_args()
    if args.length < 6 or args.chi < 1 or args.cutoff < 0 or args.repeats < 1:
        parser.error("length >= 6, chi >= 1, cutoff >= 0, and repeats >= 1 are required")

    cases = ("endpoint", "interior", "circuit") if args.case == "all" else (args.case,)
    all_results = []
    for case in cases:
        results = benchmark_case(
            case,
            args.length,
            args.chi,
            args.cutoff,
            args.repeats,
            args.seed,
        )
        all_results.extend(results)
        print(f"\n### {case} (L={args.length}, chi={args.chi})\n")
        print(_format_table(results))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(all_results, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
