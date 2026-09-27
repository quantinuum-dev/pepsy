"""Bounded PEPS sampling benchmark; write JSON outside the repository.

Example: python benchmarks/peps_sampling.py --backend torch --device cuda
Stage measurements explicitly synchronize CUDA and are a separate instrumented
run. Main repeat timings synchronize only at batch boundaries. No GPU workload
is launched automatically by this module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import sys
from time import perf_counter

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from pepsy.operators import gate_simple
from pepsy.sampling import PepsSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--shape", default="4x4")
    parser.add_argument("--bond", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=.2)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--chi-prime", type=int, default=8)
    parser.add_argument("--engine", choices=("quimb-mps", "dmrg"), default="quimb-mps")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--row-mode", choices=("reference", "dense", "factored"), default="factored")
    parser.add_argument("--cache-mib", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline", type=Path, help="optional saved sampler module for comparison")
    parser.add_argument("--contraction-opt", default="pepsy",
                        help="pepsy reusable optimizer, or a Cotengra preset")
    parser.add_argument("--row-contraction-opt", default="auto-hq")
    parser.add_argument("--amplitude-max-intermediate-mib", type=int)
    parser.add_argument("--amplitude-max-cost", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lx, ly = map(int, args.shape.split("x"))
    if min(lx, ly, args.bond, args.samples, args.repeats) < 1:
        parser.error("sizes and counts must be positive")
    torch = None
    if args.backend == "torch":
        import torch
        from pepsy.backends import TorchLinalgConfig
        torch.set_num_threads(1)
        TorchLinalgConfig(mode="complex", stabilized=False, cpu_svd="torch").register()
        dtype = getattr(torch, args.dtype)
        convert = lambda a: torch.as_tensor(a, dtype=dtype, device=args.device)
    else:
        if args.device != "cpu":
            parser.error("numpy requires --device cpu")
        convert = lambda a: np.asarray(a, dtype=args.dtype)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    # A diagonal product wall followed by second-order nearest-neighbor Ising
    # gates. The measured target is this evolved, truncated PEPS, not an exact
    # simulation of the physical Hamiltonian.
    vectors = []
    for x in range(lx):
        row = []
        for y in range(ly):
            v = np.array([np.cos(.3), np.sin(.3)], dtype=args.dtype)
            row.append(convert(v if x + y < (lx + ly - 2) / 2 else v[::-1].copy()))
        vectors.append(row)
    state = qtn.PEPS.product_state(vectors)
    rx = convert(np.cos(args.dt / 2) * np.eye(2)
                 - 1j * np.sin(args.dt / 2) * np.array([[0, 1], [1, 0]]))
    zz = convert(np.diag(np.exp(1j * args.dt * np.array([1, -1, -1, 1]))))
    single = [(rx, ((x, y),)) for x in range(lx) for y in range(ly)]
    edges = [((x, y), (x + 1, y)) for x in range(lx - 1) for y in range(ly)]
    edges += [((x, y), (x, y + 1)) for x in range(lx) for y in range(ly - 1)]
    gauges = {}
    with torch.inference_mode() if torch is not None else nullcontext():
        for _ in range(args.steps):
            gate_simple(state, single + [(zz, e) for e in edges] + single, gauges=gauges,
                        max_bond=args.bond, cutoff=0, renorm=False, inplace=True)
        state.gauge_simple_insert(gauges)
        cls = PepsSampler
        if args.baseline:
            spec = importlib.util.spec_from_file_location("pepsy.sampling._benchmark_baseline", args.baseline)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cls = module.PepsSampler
        from pepsy.tensors import build_optimizer
        optimizer = (build_optimizer(parallel=False) if args.contraction_opt == "pepsy"
                     else args.contraction_opt)
        options = dict(chi=args.chi, chi_prime=args.chi_prime, boundary_engine=args.engine,
                       contraction_opt=optimizer, rho_positivity="absolute",
                       row_cache_max_bytes=0 if args.row_mode == "reference" else args.cache_mib * 2**20)
        if not args.baseline:
            options.update(
                row_contraction_opt=args.row_contraction_opt,
                amplitude_max_intermediate_bytes=(None if args.amplitude_max_intermediate_mib is None
                                                  else args.amplitude_max_intermediate_mib * 2**20),
                amplitude_max_cost=args.amplitude_max_cost,
            )
            options["row_cache_mode"] = "factored" if args.row_mode == "factored" else "dense"
        sync()
        start = perf_counter()
        sampler = cls(state, **options)
        sync()
        setup = perf_counter() - start
        call_options = {} if args.chunk_size is None else dict(chunk_size=args.chunk_size)
        sampler.sample_batch(min(4, args.samples), seed=10, **call_options)
        sync()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for repeat in range(args.repeats):
            sync()
            start = perf_counter()
            batch = sampler.sample_batch(args.samples, seed=70 + repeat, **call_options)
            sync()
            times.append(perf_counter() - start)
        peak_cuda = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        counts, seconds = defaultdict(int), defaultdict(float)
        methods = {
            "conditionals": ("_local_rho", "_row_local_rho"),
            "row_prepare": ("_build_row_transfer_cache",),
            "row_prefix": ("_advance_row_prefix",),
            "boundary_update": ("_update_conditioned_boundary",),
            "validation_and_draw": ("_draw_grouped_choices",),
            "amplitude": ("_projected_amplitude_scaled",) if hasattr(sampler, "_projected_amplitude_scaled")
                         else ("_projected_amplitude",),
        }
        for stage, names in methods.items():
            for name in names:
                original = getattr(sampler, name)

                def timed(*a, _fn=original, _stage=stage, **kw):
                    sync()
                    t = perf_counter()
                    try:
                        return _fn(*a, **kw)
                    finally:
                        sync()
                        seconds[_stage] += perf_counter() - t
                        counts[_stage] += 1
                setattr(sampler, name, timed)
        sync()
        start = perf_counter()
        batch = sampler.sample_batch(args.samples, seed=70, **call_options)
        sync()
        profiled = perf_counter() - start
        accuracy = None
        if lx * ly <= 16:
            reference = state.copy()
            reference.apply_to_arrays(lambda a: np.asarray(ar.to_numpy(a), dtype="complex128"))
            order = [(x, y) for y in range(ly) for x in range(lx)]
            vector = reference.to_dense([state.site_ind(*s) for s in order]).ravel()
            indices = np.asarray(batch.configs) @ (2 ** np.arange(lx * ly - 1, -1, -1))
            expected = vector[indices]
            phases = np.asarray(batch.ps[0])
            amplitude = phases * 10. ** np.asarray(batch.ps[1])
            accuracy = {"max_absolute_amplitude_error": float(np.max(abs(amplitude - expected))),
                        "max_log_q_vs_born": float(np.max(abs(batch.log_probabilities -
                                                   np.log(abs(expected)**2 / np.vdot(vector, vector).real))))}
        report = dict(parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      setup_seconds=setup, repeat_seconds=times,
                      median_seconds=statistics.median(times),
                      shots_per_second=args.samples / statistics.median(times),
                      profiled_seconds=profiled, synchronized_stage_seconds=dict(seconds),
                      stage_calls=dict(counts), peak_cuda_allocated_bytes=peak_cuda,
                      process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      batch_stats=sampler.batch_stats, row_cache_stats=sampler.row_cache_stats,
                      amplitude_plan=getattr(sampler, "amplitude_plan_info", None),
                      weights=batch.weight_diagnostics, accuracy=accuracy)
        output = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(output + "\n")
        print(output)


if __name__ == "__main__":
    main()
