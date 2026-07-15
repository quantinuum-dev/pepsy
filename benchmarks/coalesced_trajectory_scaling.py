"""Benchmark independent versus count-coalesced MPS Pauli trajectories.

The benchmark sweeps fault probability, circuit depth, and shot count. It
compares :func:`pepsy.run_noisy_shots`, which retains one optimizer per shot,
against :func:`pepsy.run_coalesced_noisy_shots`, which shares equal no-error
prefixes and keeps a count per final branch.

Examples
--------
    source ~/envs/py312/bin/activate
    python benchmarks/coalesced_trajectory_scaling.py
    python benchmarks/coalesced_trajectory_scaling.py \
        --p-list 0,1e-4,1e-3,1e-2 --depth-list 2,4,8 --shots-list 32,128,512
    python benchmarks/coalesced_trajectory_scaling.py --backend torch --device cuda

The GPU option accelerates the ordinary MPS gate and terminal tensor work. It
does not make divergent noise branches a uniform ``vmap`` batch; the reported
``coalesced_states`` makes the structural sharing visible directly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np


def _parse_csv(value, cast):
    """Parse a nonempty comma-separated command-line list."""
    values = [cast(item) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value.")
    return values


def brickwall_gate_stream(
    n: int, depth: int, *, backend: str = "numpy", device: str | None = None
):
    """Return a small entangling MPS gate stream with deterministic structure."""
    import quimb as qu

    backend = str(backend).lower()

    def on_backend(matrix):
        if backend == "numpy":
            return matrix
        if backend in {"torch", "cuda"}:
            import torch

            target = "cuda" if backend == "cuda" and device is None else (device or "cpu")
            return torch.as_tensor(
                np.array(matrix, copy=True), dtype=torch.complex128, device=target
            )
        raise ValueError("backend must be 'numpy', 'torch', or 'cuda'.")

    stream = []
    for layer in range(int(depth)):
        for site in range(int(n)):
            if (layer + site) % 2 == 0:
                stream.append((on_backend(qu.hadamard()), site))
        for left in range(layer % 2, int(n) - 1, 2):
            stream.append((on_backend(qu.CNOT()), (left, left + 1)))
    return stream


def _state_factory(n: int, chi: int, backend: str, device: str | None):
    """Build a repeatable ordinary-MPS factory, optionally with Torch tensors."""
    import quimb.tensor as qtn
    from pepsy import MpsOptimizer

    state = qtn.MPS_computational_state("0" * int(n), dtype="complex128")
    backend = str(backend).lower()
    if backend == "numpy":
        pass
    elif backend in {"torch", "cuda"}:
        import torch

        device = "cuda" if backend == "cuda" and device is None else (device or "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA was requested but no CUDA device is available.")
        state.apply_to_arrays(
            lambda array: torch.as_tensor(
                array, dtype=torch.complex128, device=device
            )
        )
    else:
        raise ValueError("backend must be 'numpy', 'torch', or 'cuda'.")

    return lambda: MpsOptimizer(state, chi=int(chi), mode="mpo")


def noisy_target_count(n: int, depth: int) -> int:
    """Count post-gate Pauli-channel targets in :func:`brickwall_gate_stream`."""
    targets = 0
    for layer in range(int(depth)):
        targets += sum((layer + site) % 2 == 0 for site in range(int(n)))
        targets += 2 * len(range(layer % 2, int(n) - 1, 2))
    return targets


def _median_elapsed(fn, repeats: int, *, synchronize=None):
    """Return the median wall time and final value of a repeated callable."""
    elapsed = []
    value = None
    for _ in range(int(repeats)):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        value = fn()
        if synchronize is not None:
            synchronize()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed)), value


def run_case(
    *,
    n: int,
    depth: int,
    shots: int,
    probability: float,
    chi: int,
    seed: int,
    repeats: int = 1,
    backend: str = "numpy",
    device: str | None = None,
):
    """Time one independent/coalesced pair and return JSON-ready metrics."""
    from pepsy import (
        PauliErrorModel,
        run_coalesced_noisy_shots,
        run_noisy_shots,
    )

    n = int(n)
    depth = int(depth)
    shots = int(shots)
    if n < 2 or depth < 1 or shots < 1:
        raise ValueError("n >= 2, depth >= 1, and shots >= 1 are required.")
    stream = brickwall_gate_stream(n, depth, backend=backend, device=device)
    model = PauliErrorModel.depolarizing(float(probability))
    noisy_targets = noisy_target_count(n, depth)
    factory = _state_factory(n, chi, backend, device)
    run_kwargs = {"progbar": False, "fidelity_samples": 0}
    synchronize = None
    torch_device = "cuda" if backend == "cuda" and device is None else device
    if backend in {"torch", "cuda"} and str(torch_device or "").startswith("cuda"):
        import torch

        synchronize = lambda: torch.cuda.synchronize(torch_device)

    baseline_s, baseline = _median_elapsed(
        lambda: run_noisy_shots(
            factory, stream, model, shots, seed=seed, run_kwargs=run_kwargs
        ),
        repeats,
        synchronize=synchronize,
    )
    coalesced_s, coalesced = _median_elapsed(
        lambda: run_coalesced_noisy_shots(
            factory, stream, model, shots, seed=seed, run_kwargs=run_kwargs
        ),
        repeats,
        synchronize=synchronize,
    )
    branches = coalesced.branches
    return {
        "n": n,
        "depth": depth,
        "gates": len(stream),
        "noisy_targets": noisy_targets,
        "expected_faults": noisy_targets * float(probability),
        "shots": shots,
        "probability": float(probability),
        "chi": int(chi),
        "backend": str(backend),
        "device": device,
        "independent_s": baseline_s,
        "coalesced_s": coalesced_s,
        "speedup": baseline_s / coalesced_s if coalesced_s > 0.0 else None,
        "independent_states": len(baseline.optimizers),
        "coalesced_states": branches,
        "state_reduction": shots / branches,
        "represented_shots": coalesced.shots,
    }


def _case_key(row):
    """Return the resume key for one benchmark case."""
    return (
        row["n"],
        row["depth"],
        row["shots"],
        row["probability"],
        row["chi"],
        row["backend"],
        row["device"],
    )


def _with_derived_metrics(row):
    """Fill report-only fields omitted by checkpoints from older scripts."""
    row = dict(row)
    noisy_targets = noisy_target_count(row["n"], row["depth"])
    row.setdefault("noisy_targets", noisy_targets)
    row.setdefault("expected_faults", noisy_targets * row["probability"])
    return row


def _report_config(args, probabilities, depths, shots_list):
    """Build the JSON-stable benchmark configuration section."""
    return {
        "n": int(args.n),
        "p_list": probabilities,
        "depth_list": depths,
        "shots_list": shots_list,
        "chi": int(args.chi),
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "backend": args.backend,
        "device": args.device,
    }


def run(args, *, existing_results=(), checkpoint=None):
    """Run the requested probability/depth/shot sweep."""
    probabilities = _parse_csv(args.p_list, float)
    depths = _parse_csv(args.depth_list, int)
    shots_list = _parse_csv(args.shots_list, int)
    report = {
        "config": _report_config(args, probabilities, depths, shots_list),
        "results": [],
    }
    existing = {_case_key(row): row for row in existing_results}
    for probability in probabilities:
        for depth in depths:
            for shots in shots_list:
                key = (
                    int(args.n),
                    int(depth),
                    int(shots),
                    float(probability),
                    int(args.chi),
                    str(args.backend),
                    args.device,
                )
                if key in existing:
                    report["results"].append(_with_derived_metrics(existing[key]))
                    continue
                row = run_case(
                    n=args.n,
                    depth=depth,
                    shots=shots,
                    probability=probability,
                    chi=args.chi,
                    seed=args.seed,
                    repeats=args.repeats,
                    backend=args.backend,
                    device=args.device,
                )
                report["results"].append(row)
                if getattr(args, "progress", False):
                    print(
                        "completed "
                        f"p={probability:.2e}, depth={depth}, shots={shots}: "
                        f"speedup={row['speedup']:.2f}, "
                        f"leaves={row['coalesced_states']}",
                        flush=True,
                    )
                if checkpoint is not None:
                    checkpoint(report)
    return report


def _print_table(report):
    """Print a compact comparison table alongside optional JSON output."""
    header = (
        f"{'p':>9} {'lambda':>8} {'depth':>5} {'shots':>7} {'ind[s]':>10} {'coal[s]':>10} "
        f"{'speedup':>8} {'states':>12} {'reduction':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        print(
            f"{row['probability']:>9.2e} {row['expected_faults']:>8.2e} "
            f"{row['depth']:>5} {row['shots']:>7} "
            f"{row['independent_s']:>10.4f} {row['coalesced_s']:>10.4f} "
            f"{row['speedup']:>8.2f} "
            f"{row['coalesced_states']:>5}/{row['independent_states']:<6} "
            f"{row['state_reduction']:>10.2f}"
        )


def build_arg_parser():
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="number of qubits")
    parser.add_argument("--depth-list", default="2,4,8")
    parser.add_argument("--shots-list", default="32,128,512")
    parser.add_argument("--p-list", default="0,1e-4,1e-3,1e-2")
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--backend", choices=("numpy", "torch", "cuda"), default="numpy")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--json", default=None, help="optional path for JSON report")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print each completed probability/depth/shot case immediately",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed matching cases from --json and checkpoint every case",
    )
    return parser


def _write_json(path, report):
    """Atomically checkpoint a JSON-ready benchmark report."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    temporary.replace(path)


def main():
    """Run the command-line benchmark."""
    args = build_arg_parser().parse_args()
    if args.resume and not args.json:
        raise SystemExit("--resume requires --json PATH.")
    existing_results = ()
    if args.resume and Path(args.json).is_file():
        with open(args.json, encoding="utf-8") as handle:
            existing_results = json.load(handle).get("results", ())
    checkpoint = (
        (lambda report: _write_json(args.json, report)) if args.json else None
    )
    report = run(args, existing_results=existing_results, checkpoint=checkpoint)
    _print_table(report)
    if args.json:
        _write_json(args.json, report)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
