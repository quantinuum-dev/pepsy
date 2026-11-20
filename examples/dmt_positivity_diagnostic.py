"""Locate global and local positivity loss separately from DMT preservation.

Uses independent physical-index partial traces. Full-operator eigenvalues
are computed only for L<=8; larger runs inspect three/four-site densities.
Run from the checkout root: python examples/dmt_positivity_diagnostic.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pepsy.operators import PauliMPO
from pepsy.optimizers import PauliDMT

try:
    from examples import dmt_ising_dynamics as bench
except ModuleNotFoundError:
    import dmt_ising_dynamics as bench


def minimum(matrix):
    return float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())


def diagnose(n=16, chi=16, dt=.05, steps=100, *, backend="numpy", real=True,
             threshold=1e-9, output=None):
    hx, hz = bench.MODELS["tilted"]
    vectors = bench.initial_vectors(n)
    cores = [np.einsum("pij,ji->p", bench.PAULIS, np.outer(v, v.conj()))[None, None] / 2
             for v in vectors]
    if real:
        cores = [core.real for core in cores]
    if backend == "torch":
        import torch
        torch.set_num_threads(1)
        dtype = torch.float64 if real else torch.complex128
        cores = [torch.tensor(core, dtype=dtype) for core in cores]
    driver = PauliDMT(PauliMPO.from_pauli_cores(cores), max_bond=chi, cutoff=0., diagnostics=True)
    gates = bench.sweep_gates(bench.bond_hamiltonians(n, hx, hz), dt)
    result = {"length": n, "chi": chi, "dt": dt, "backend": backend,
              "dtype": "float64" if real else "complex128", "threshold": threshold,
              "first_global_loss": None, "first_local_loss": None, "checked_gates": 0,
              "max_triple_defect": 0.}
    for step in range(1, steps + 1):
        for index, (gate, where) in enumerate(gates):
            old = bench.Marginals(bench.pauli_cores(driver.state))
            old_triples = np.array([old.region(range(j, j + 3)) for j in range(n - 2)]) / old.trace
            target = bench.gated_triples(old, gate, where) / old.trace
            payload = gate
            if backend == "torch":
                payload = torch.tensor(gate, dtype=torch.complex128)
            driver.apply_gate(payload, where)
            new = bench.Marginals(bench.pauli_cores(driver.state))
            actual = np.array([new.region(range(j, j + 3)) for j in range(n - 2)]) / new.trace
            defect = float(np.linalg.norm(actual - target, axis=(1, 2)).max())
            result["max_triple_defect"] = max(result["max_triple_defect"], defect)
            result["checked_gates"] += 1
            if defect > 1e-9:
                raise AssertionError("protected triples differ from the gated target")
            event = {"step": step, "gate_index": index, "bond": list(where),
                     "step_time_interval": [(step - 1) * dt, step * dt],
                     "pre_gate_min_triple_eigenvalue": min(map(minimum, old_triples)),
                     "gated_target_min_triple_eigenvalue": min(map(minimum, target)),
                     "post_compression_min_triple_eigenvalue": min(map(minimum, actual)),
                     "max_triple_preservation_defect": defect,
                     "max_local_hermiticity_defect": float(np.linalg.norm(
                         actual - actual.conj().transpose(0, 2, 1), axis=(1, 2)).max())}
            if n <= 8 and result["first_global_loss"] is None:
                before = old.region(range(n)) / old.trace
                after = new.region(range(n)) / new.trace
                if minimum(after) < -threshold:
                    result["first_global_loss"] = event | {
                        "pre_gate_global_min_eigenvalue": minimum(before),
                        "post_compression_global_min_eigenvalue": minimum(after)}
            if event["post_compression_min_triple_eigenvalue"] < -threshold:
                window = int(np.argmin(list(map(minimum, actual))))
                union = tuple(sorted(set(range(window, window + 3)).union(where)))
                result["first_local_loss"] = event | {
                    "triple_sites": list(range(window, window + 3)),
                    "pre_gate_union_sites": list(union),
                    "pre_gate_union_min_eigenvalue": minimum(old.region(union) / old.trace)}
                if output is not None:
                    np.savez_compressed(output.with_suffix(".npz"), pre_gate_triples=old_triples,
                                        gated_target_triples=target, post_compression_triples=actual,
                                        pre_gate_union=old.region(union) / old.trace)
                if result["first_local_loss"]["gated_target_min_triple_eigenvalue"] >= -threshold:
                    raise AssertionError("local negativity appeared during compression")
                if output is not None:
                    output.write_text(json.dumps(result, indent=2) + "\n")
                return result
    raise RuntimeError("no local positivity loss found in the requested interval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--dt", type=float, default=.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--complex", action="store_true", dest="complex_storage")
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-positivity.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        result = diagnose(args.length, args.chi, args.dt, args.steps, backend=args.backend,
                          real=not args.complex_storage, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
