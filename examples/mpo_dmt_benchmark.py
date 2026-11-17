"""CPU benchmarks with independent local-density validation outside timing.

Run ``python examples/mpo_dmt_benchmark.py --case all --repeats 3`` or use
``--square 6 8 --chis 64 96 128 --operation compress --methods dmt``.
Custom chi sweeps keep the input rank fixed; protected-budget rejections are
reported explicitly. ``--schedule center`` includes long interior MPO spans.
This compares algorithms with different preservation contracts, not equal-error
compressors. Replay timings include optimizer construction and exact targets.
"""

import argparse
import cProfile
from importlib.metadata import version
import json
import pstats
import sys
from time import perf_counter

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from scipy.linalg import expm

from pepsy.operators import compress_mpo_dmt, square_lattice_dmt_regions
from pepsy.optimizers import MpoOptimizer
from pepsy.tensors import OneDMap


CASES = {
    "4x4-compress": (4, 4, 72, 64, False),
    "5x5-compress": (5, 5, 96, 72, False),
    "3x3-replay": (3, 3, 64, 40, True),
    "4x4-replay": (4, 4, 72, 64, True),
    "3x3-layer": (3, 3, 64, 40, True),
}


def _peak_rss_mib():
    try:
        import resource
    except ImportError:  # unavailable on Windows
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)


def mixture(n, rank, dtype):
    rng = np.random.default_rng(321)
    weights = rng.uniform(.2, 1., rank)
    weights /= weights.sum()
    v = rng.normal(size=(rank, n, 3))
    v *= .5 / np.linalg.norm(v, axis=-1, keepdims=True)
    local = np.zeros((rank, n, 2, 2), dtype=complex)
    local[..., 0, 0], local[..., 1, 1] = (1 + v[..., 2]) / 2, (1 - v[..., 2]) / 2
    local[..., 0, 1] = (v[..., 0] - 1j * v[..., 1]) / 2
    local[..., 1, 0] = local[..., 0, 1].conj()
    arrays = [weights[:, None, None] * local[:, 0]]
    for j in range(1, n - 1):
        a = np.zeros((rank, rank, 2, 2), dtype=complex)
        a[np.arange(rank), np.arange(rank)] = local[:, j]
        arrays.append(a)
    arrays.append(local[:, -1])
    return qtn.MatrixProductOperator([a.astype(dtype) for a in arrays]), weights, local


def marginal(mpo, region):
    """Direct matrix-unit trace contraction, independent of DMT's panels."""
    value = np.ones(1, dtype=complex)
    for i in range(mpo.L):
        bonds = (mpo.bond(i - 1, i),) if i else ()
        bonds += (mpo.bond(i, i + 1),) if i + 1 < mpo.L else ()
        a = mpo[i].transpose(*(bonds + (mpo.upper_ind(i), mpo.lower_ind(i)))).data
        a = a.reshape(value.shape[-1], -1, 2, 2)
        if i in region:
            value = np.moveaxis(np.tensordot(value, a, axes=(-1, 0)), -3, -1)
        else:
            value = value @ np.trace(a, axis1=2, axis2=3)
    k = len(region)
    order = tuple(range(0, 2 * k, 2)) + tuple(range(1, 2 * k, 2))
    return value[..., 0].transpose(order).reshape(2**k, 2**k) * 10. ** mpo.exponent


def product_panel(weights, local, region):
    panel = np.zeros((2**len(region),) * 2, dtype=complex)
    for weight, factors in zip(weights, local):
        term = np.ones((1, 1))
        for i in region:
            term = np.kron(term, factors[i])
        panel += weight * term
    return panel


def compression_relative_error(result, weights, local):
    """Global Hilbert-Schmidt error against the input mixture, without densifying.

    Full compression returns a left-isometric chain with its norm at the last
    core. The other two overlaps use independent product-mixture contractions.
    """
    gram = np.ones((len(weights), len(weights)))
    for site in range(result.L):
        factors = local[:, site]
        gram *= np.einsum("ist,jst->ij", factors.conj(), factors).real
    reference_norm2 = weights @ gram @ weights
    environment = weights[:, None].astype(complex)
    for site in range(result.L):
        bonds = (result.bond(site - 1, site),) if site else ()
        bonds += (result.bond(site, site + 1),) if site + 1 < result.L else ()
        core = result[site].transpose(*(bonds + (result.upper_ind(site), result.lower_ind(site)))).data
        core = core.reshape(environment.shape[1], -1, 2, 2)
        environment = np.einsum("ja,abst,jst->jb", environment, core, local[:, site].conj(), optimize=True)
    cross = environment.sum() * 10. ** result.exponent
    center = result[result.L - 1].data
    result_norm2 = np.vdot(center, center).real * 10. ** (2 * result.exponent)
    error2 = (reference_norm2 + result_norm2 - 2 * cross.real) / reference_norm2
    # Cancellation limits an overlap-based diagnostic near an exact result.
    # Permit only a roundoff-sized negative value, never a large inconsistency.
    if error2 < -2e-5:
        raise AssertionError(f"Inconsistent global compression norm diagnostic: {error2}")
    return float(np.sqrt(max(0., error2)))


def embed_gate(operator, where, n):
    k = len(where)
    identity = np.eye(2**n, dtype=complex).reshape((2,) * n + (2**n,))
    applied = np.tensordot(operator.reshape((2,) * (2 * k)), identity,
                           axes=(tuple(range(k, 2 * k)), where))
    labels = list(where) + [i for i in range(n) if i not in where]
    return applied.transpose(tuple(np.argsort(labels)) + (n,)).reshape(2**n, 2**n)


def unitary_target_panels(mpo, payload, where, regions):
    """Independent gate oracle on at most |R union support| physical sites.

    Disjoint marginals and the trace are invariant under unitary conjugation
    to the gate's working precision. Intersecting regions use their enlarged
    input density, apply the small dense gate, then trace the extra sites.
    No DMT basis, canonical workspace or full MPO target is used.
    """
    unitary = payload.reshape(4, 4).T
    tolerance = 2e-6 if payload.dtype == np.complex64 else 1e-13
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), atol=tolerance, rtol=0.)
    panels = {}
    for region in regions:
        if not set(region).intersection(where):
            panels[region] = marginal(mpo, region)
            continue
        union = tuple(sorted(set(region).union(where)))
        operator = embed_gate(unitary, tuple(union.index(s) for s in where), len(union))
        reduced = marginal(mpo, union)
        target = operator @ reduced @ operator.conj().T
        panels[region] = qu.partial_trace(target, [2] * len(union), tuple(union.index(s) for s in region))
    return panels


class CheckedReplay(MpoOptimizer):
    """Untimed validation of every gate target, not only the final state."""

    def __init__(self, *args, regions, **kwargs):
        self.regions = regions
        self.expected_panels = None
        self.max_local_error = 0.
        self.max_trace_error = 0.
        self.validation_seconds = 0.
        super().__init__(*args, **kwargs)
        self.reference_step = 0

    def _check_finite(self, finite_check, p=None):
        started = perf_counter()
        super()._check_finite(finite_check, p)
        if self.mode == "dmt":
            gates, wheres = self._execution_stream()
            gate, bra, where = self._parse_gate_entry(gates[self.reference_step], wheres[self.reference_step])
            if bra is not gate:
                raise ValueError("This benchmark oracle requires unitary conjugation payloads")
            self.expected_panels = unitary_target_panels(self.p, gate, where, (*self.regions, ()))
            self.reference_step += 1
        if self.expected_panels is not None:
            for region, expected in self.expected_panels.items():
                error = float(np.linalg.norm(marginal(p, region) - expected))
                if region:
                    self.max_local_error = max(self.max_local_error, error)
                else:
                    self.max_trace_error = max(self.max_trace_error, error)
            self.expected_panels = None
        self.validation_seconds += perf_counter() - started


def benchmark(case, dtype, repeats, profile=False, cutoff="auto", cutoff_mode="auto", *,
              config=None, methods=("dmt", "svd"), schedule="sample", region_kind="plaquettes",
              single_pass=False):
    lx, ly, rank, cap, replay = CASES[case] if config is None else config
    initial, weights, local = mixture(lx * ly, rank, dtype)
    mapper = OneDMap(lx, ly, mode="snake-row-major")
    _, mapping = mapper.build()
    regions = square_lattice_dmt_regions(lx, ly, mapper=mapper, kind=region_kind)
    rng = np.random.default_rng(17)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    u = expm(-.08j * (h + h.conj().T)).astype(dtype)
    # Four physical lattice edges, including distant MPO endpoints. Payload
    # transpose gives the physical action U rho U-dagger in this class API.
    edges = [((0, 0), (1, 0)), ((0, 1), (1, 1)),
             ((1, 0), (2, 0)), ((1, 1), (1, 2))]
    if schedule == "center":
        x, y = (lx - 2) // 2, (ly - 2) // 2
        outer = 0 if y % 2 == 0 else lx - 1
        edges = [((x, y), (x + 1, y)), ((x + 1, y), (x + 2, y)),
                 ((x, y), (x, y + 1)), ((outer, y), (outer, y + 1))]
    if case.endswith("-layer") or schedule == "layer":
        edges = ([((x, y), (x + 1, y)) for x in range(lx - 1) for y in range(ly)]
                 + [((x, y), (x, y + 1)) for x in range(lx) for y in range(ly - 1)])
    queue = [(u.T, (mapping[a], mapping[b])) for a, b in edges] if replay else []

    def run(method):
        if replay:
            opt = MpoOptimizer(initial, queue, chi=cap, mode=method, contraction_opt="greedy")
            if method == "svd":
                # Local SVD replay leaves unvisited initial bonds untouched.
                # Add a global sweep per step so both methods obey the cap.
                for entry in queue:
                    opt.set_gates([entry])
                    opt.run(cutoff=cutoff, cutoff_mode=cutoff_mode, fidelity_samples=0)
                    opt.compress(cutoff=cutoff, cutoff_mode="rsum2" if cutoff_mode == "auto" else cutoff_mode)
                return opt.p
            return opt.run(cutoff=cutoff, cutoff_mode=cutoff_mode, fidelity_samples=0,
                           **({"dmt_regions": regions} if method == "dmt" else {}))
        if method == "dmt":
            return compress_mpo_dmt(initial, max_bond=cap, regions=regions,
                                    cutoff=cutoff, cutoff_mode=cutoff_mode)
        result = initial.copy()
        result.compress(max_bond=cap, cutoff=(1e-6 if dtype == "complex64" else 1e-12)
                        if cutoff == "auto" else cutoff,
                        cutoff_mode="rsum2" if cutoff_mode == "auto" else cutoff_mode, form="left")
        return result

    for method in methods:
        try:
            if single_pass:
                started = perf_counter()
                checked = CheckedReplay(initial, queue, chi=cap, mode=method,
                                        contraction_opt="greedy", regions=regions)
                result = checked.run(cutoff=cutoff, cutoff_mode=cutoff_mode, dmt_regions=regions,
                                     fidelity_samples=0, timing=True)
                seconds = [perf_counter() - started - checked.validation_seconds]
            else:
                run(method)
        except (ValueError, FloatingPointError) as exc:
            # Tight protected budgets are an expected outcome of a chi sweep.
            # Record the explicit rejection and continue to the next case.
            if "DMT" not in str(exc):
                raise
            print(json.dumps({"case": case, "method": method, "dtype": dtype,
                              "rank": rank, "cap": cap, "region_kind": region_kind,
                              "status": "rejected", "error_type": type(exc).__name__,
                              "error": str(exc)}), flush=True)
            continue
        if not single_pass:
            seconds = []
            for _ in range(repeats):
                started = perf_counter()
                result = run(method)
                seconds.append(perf_counter() - started)
        if replay:
            if not single_pass:
                checked = CheckedReplay(initial, queue, chi=cap, mode=method,
                                        contraction_opt="greedy", regions=regions)
            if method == "dmt":
                if not single_pass:
                    checked.run(cutoff=cutoff, cutoff_mode=cutoff_mode, dmt_regions=regions, fidelity_samples=0)
                error, trace_error = checked.max_local_error, checked.max_trace_error
            else:
                # The SVD path does not build every unitary target. Explicit
                # step-wise source/target references make its error comparable.
                checked.max_local_error = checked.max_trace_error = 0.
                for payload, where in queue:
                    panels = unitary_target_panels(checked.p, payload, where, (*regions, ()))
                    checked.set_gates([(payload, where)])
                    checked.expected_panels = None
                    checked.run(cutoff=cutoff, cutoff_mode=cutoff_mode, fidelity_samples=0)
                    checked.compress(cutoff=cutoff, cutoff_mode="rsum2" if cutoff_mode == "auto" else cutoff_mode)
                    for r, panel in panels.items():
                        error_here = float(np.linalg.norm(marginal(checked.p, r) - panel))
                        if r:
                            checked.max_local_error = max(checked.max_local_error, error_here)
                        else:
                            checked.max_trace_error = max(checked.max_trace_error, error_here)
                error, trace_error = checked.max_local_error, checked.max_trace_error
        else:
            error = max(float(np.linalg.norm(marginal(result, r) - product_panel(weights, local, r)))
                        for r in regions)
            trace_error = float(np.linalg.norm(marginal(result, ()) - 1.))
        row = {"case": case, "method": method, "dtype": dtype, "cutoff": cutoff, "cutoff_mode": cutoff_mode,
               "status": "ok", "region_kind": region_kind, "gates": len(queue),
               "gate_spans": [abs(w[0] - w[1]) for _, w in queue],
               "sites": lx * ly, "rank": rank, "cap": cap,
               "max_bond": result.max_bond(), "median_seconds": float(np.median(seconds)),
               "state_storage_mib": sum(a.nbytes for a in result.arrays) / 2**20,
               "process_peak_rss_mib": _peak_rss_mib(),
               "max_local_error": error, "max_trace_error": trace_error}
        if not replay:
            row["relative_hilbert_schmidt_error"] = compression_relative_error(result, weights, local)
        if replay and method == "dmt":
            if single_pass:
                row["timing_scope"] = "one_run_wall_minus_oracle"
                row["validation_seconds"] = checked.validation_seconds
                row["timing_stages"] = {key: value for key, value in checked.get_run_timing()["stages"].items()
                                        if key != "dmt.replay"}
            else:
                timed = MpoOptimizer(initial, queue, chi=cap, mode="dmt", contraction_opt="greedy")
                timed.run(cutoff=cutoff, cutoff_mode=cutoff_mode, dmt_regions=regions, fidelity_samples=0, timing=True)
                row["timing_stages"] = timed.get_run_timing()["stages"]
        if replay and initial.L <= 9:
            # Distinguish exact per-step protection from accumulated error
            # relative to the completely untruncated circuit trajectory.
            exact = marginal(initial, tuple(range(initial.L)))
            for payload, where in queue:
                full = embed_gate(payload.T, where, initial.L)
                exact = full @ exact @ full.conj().T
            row["exact_trajectory_local_error"] = max(
                float(np.linalg.norm(marginal(result, r) - qu.partial_trace(exact, [2] * initial.L, r)))
                for r in regions
            )
        print(json.dumps(row), flush=True)
        assert result.max_bond() <= cap
        tolerance = 1e-5 if dtype == "complex64" else 1e-10
        if method == "dmt" and (error > tolerance or trace_error > tolerance):
            raise AssertionError(f"DMT validation failed: {row}")
        if profile and method == "dmt":
            profiler = cProfile.Profile()
            profiler.runcall(run, method)
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--single-pass", action="store_true",
                        help="one validated custom DMT replay, subtracting oracle time; no warmup or extra timing runs")
    parser.add_argument("--square", type=int, nargs="+", help="custom square sizes, e.g. 6 8")
    parser.add_argument("--chis", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--initial-rank", type=int, help="fixed input rank across a custom chi sweep")
    parser.add_argument("--operation", choices=["compress", "replay", "both"], default="both")
    parser.add_argument("--methods", nargs="+", choices=["dmt", "svd"], default=["dmt", "svd"])
    parser.add_argument("--schedule", choices=["sample", "center", "layer"], default="sample")
    parser.add_argument("--region-kind", choices=["plaquettes", "triples", "l_shapes"], default="plaquettes")
    parser.add_argument("--cutoff", default="auto", help="auto (default) or a nonnegative number")
    parser.add_argument("--cutoff-mode", default="auto", choices=["auto", "rel", "abs", "sum1", "rsum1", "sum2", "rsum2"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.single_pass and (not args.square or args.operation != "replay" or args.methods != ["dmt"]
                             or args.repeats != 1 or args.profile):
        parser.error("--single-pass requires --square, --operation replay, --methods dmt, --repeats 1, without --profile")
    if (any(c < 1 for c in args.chis) or (args.square and any(n < 3 for n in args.square))
            or (args.initial_rank is not None and args.initial_rank < 1)):
        parser.error("square sizes must be at least 3; chi and initial rank must be positive")
    if args.cutoff != "auto":
        try:
            args.cutoff = float(args.cutoff)
        except ValueError:
            parser.error("--cutoff must be auto or a nonnegative finite number")
        if not np.isfinite(args.cutoff) or args.cutoff < 0:
            parser.error("--cutoff must be auto or a nonnegative finite number")
    print(json.dumps({"versions": {name: version(name) for name in ("quimb", "autoray", "numpy")}}))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        parser.error("this CPU benchmark requires the optional threadpoolctl package")

    with threadpool_limits(limits=1):
        common = dict(methods=args.methods, schedule=args.schedule, region_kind=args.region_kind,
                      single_pass=args.single_pass)
        if args.square:
            rank = args.initial_rank or max(args.chis) + 8
            operations = ["compress", "replay"] if args.operation == "both" else [args.operation]
            for size in args.square:
                for chi in args.chis:
                    for operation in operations:
                        name = f"{size}x{size}-{operation}-chi{chi}"
                        benchmark(name, args.dtype, args.repeats, args.profile, args.cutoff, args.cutoff_mode,
                                  config=(size, size, rank, chi, operation == "replay"), **common)
        else:
            for name in CASES if args.case == "all" else [args.case]:
                benchmark(name, args.dtype, args.repeats, args.profile, args.cutoff, args.cutoff_mode, **common)
