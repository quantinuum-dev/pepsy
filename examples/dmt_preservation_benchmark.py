"""Time DMT verification separately from readout on a short Ising trajectory.

Uses complex128, one CPU thread, two untimed warmup steps and repeated
timesteps at L=64 by default. Each configuration starts from the same state
and schedule. Diagnostics and state readout are outside the timed work.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quimb.tensor as qtn
from threadpoolctl import threadpool_limits

from pepsy.operators import PauliMPO
from pepsy.optimizers import MpoOptimizer, PauliDMT

try:
    from examples import dmt_ising_dynamics as bench
except ModuleNotFoundError:
    import dmt_ising_dynamics as bench


def measure(*, length, repeats, backend):
    hx, hz = bench.MODELS["tilted"]
    vectors = bench.initial_vectors(length, "near-y-figure")
    gates = bench.sweep_gates(bench.bond_hamiltonians(length, hx, hz), .0625 / 2,
                             merge_turnaround=False)

    def convert(a):
        if backend == "numpy":
            return np.asarray(a, dtype=np.complex128)
        import torch
        return torch.tensor(np.array(a, copy=True), dtype=torch.complex128)

    native_gates = [(convert(u), w) for u, w in gates]
    configurations = [("pauli-dmt", "warn", True), ("pauli-dmt", "warn", False),
                      ("pauli-dmt", "off", False),
                      ("mpo-dmt", "warn", True), ("mpo-dmt", "warn", False),
                      ("mpo-dmt", "off", False)]
    outputs, rows = {}, []
    for method, policy, regions in configurations:
        if method == "pauli-dmt":
            cores = [convert(np.einsum("pij,ji->p", bench.PAULIS, np.outer(v, v.conj())) / 2)
                     .reshape(1, 1, 4) for v in vectors]
            opt = PauliDMT(PauliMPO.from_pauli_cores(cores), max_bond=16, cutoff=0.,
                           preservation_check=policy, check_regions=regions)

            def step():
                for u, w in native_gates:
                    opt.apply_gate(u, w)

            def read():
                return bench.measure_mpo(bench.pauli_cores(opt.state), hx, hz)
        else:
            initial = qtn.MPO_product_operator([np.outer(v, v.conj()) for v in vectors])
            initial.apply_to_arrays(convert)
            opt = MpoOptimizer(initial, [(u.T, w) for u, w in native_gates], chi=16,
                               mode="dmt", contraction_opt="greedy")

            def step():
                opt.run(cutoff=0., fidelity_samples=0, dmt_diagnostics=False,
                        dmt_preservation_check=policy, dmt_check_regions=regions)

            def read():
                return bench.measure_mpo(bench.mpo_cores(opt.p), hx, hz)

        for _ in range(2):
            step()
        seconds = []
        for _ in range(repeats):
            start = perf_counter()
            step()
            seconds.append(perf_counter() - start)
        result = read()
        key = f"{method}:{policy}:regions={regions}"
        outputs[key] = result
        reference = outputs[f"{method}:warn:regions=True"]
        rows.append({"method": method, "preservation_check": policy, "check_regions": regions,
                     "median_seconds": float(np.median(seconds)), "sample_seconds": seconds,
                     "trace_defect": float(abs(result["trace"] - 1)),
                     "max_triple_difference_vs_checked": float(np.max(np.linalg.norm(
                         result["triple_rdm"] - reference["triple_rdm"], axis=(-1, -2))))})
        print(json.dumps(rows[-1]), flush=True)
    return {"length": length, "chi": 16, "backend": backend, "dtype": "complex128",
            "gate_dtype": "complex128", "threads": 1, "warmup_steps": 2,
            "timed_steps": repeats, "gates_per_step": len(gates), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-preservation-timing.json"))
    args = parser.parse_args()
    if args.length < 3 or args.repeats < 1:
        parser.error("require length >= 3 and repeats >= 1")
    if args.backend == "torch":
        import torch
        from pepsy.backends import TorchLinalgConfig
        torch.set_num_threads(1)
        TorchLinalgConfig(cpu_svd="torch").register()
    with threadpool_limits(limits=1):
        result = measure(length=args.length, repeats=args.repeats, backend=args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
