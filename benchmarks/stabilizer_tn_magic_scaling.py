"""Benchmark the Stabilizer Tensor Network (STN) magic-vs-chi scaling.

Compares three STN treatments of T-doped Clifford circuits:

* ``direct`` applies every T rotation to ``|nu>``;
* ``immediate`` teleports every T through one recycled ancilla and immediately
  projects it;
* ``deferred`` implements the MAST protocol: one fresh magic ancilla per T,
  then end-of-circuit basis-updating projections in a chosen order.

The report separates circuit replay and final-projection time for deferred
MAST, and records the peak coefficient-MPS bond rather than only its final
value. Clifford gates are free tableau updates; non-Clifford resource handling
is the quantity being compared.

For each ``N`` the script uses the same deterministic random circuit in every
mode, records peak/final ``|nu>`` bond, pseudo-stabilizer rank (for small
systems), total wall time, and deferred replay/projection times, then emits JSON
and a human-readable table. Use ``--no-exact-cooling`` to isolate injection from
the constructive exact-cooling pre-check.

Examples
--------
    source ~/envs/py312/bin/activate
    python benchmarks/stabilizer_tn_magic_scaling.py
    python benchmarks/stabilizer_tn_magic_scaling.py --n-list 8,16,32,64 --t 4
    python benchmarks/stabilizer_tn_magic_scaling.py --t 3 --backend torch
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np


_ONE_Q_CLIFFORD = ("h", "s", "sdg", "x", "y", "z")


def random_clifford_t_circuit(n, t, depth, seed):
    """Return a deterministic Clifford gate stream with ``t`` interspersed T gates.

    Each of ``depth`` layers applies a random single-qubit Clifford per qubit and
    a brick-wall of CNOTs; ``t`` ``("t", q)`` gates are placed at distinct
    (layer, qubit) slots.  The T-count is exactly ``t`` regardless of ``n``.
    """
    rng = np.random.default_rng(seed)
    t = min(int(t), depth * n)
    slots = set()
    while len(slots) < t:
        slots.add((int(rng.integers(depth)), int(rng.integers(n))))
    stream = []
    for layer in range(depth):
        for q in range(n):
            stream.append((_ONE_Q_CLIFFORD[int(rng.integers(len(_ONE_Q_CLIFFORD)))], q))
            if (layer, q) in slots:
                stream.append(("t", q))
        for a in range(layer % 2, n - 1, 2):  # brick-wall CNOTs
            stream.append(("cnot", a, a + 1))
    return stream


def _resolve_backend(name):
    if name in (None, "", "numpy"):
        return None
    import pepsy as py

    if name == "torch":
        import torch

        return py.backend_torch(dtype=torch.complex128, device="cpu")
    if name == "cupy":
        return py.backend_cupy()
    if name == "jax":
        return py.backend_jax(dtype="complex128")
    raise ValueError(f"unknown backend {name!r} (use numpy/torch/cupy/jax).")


def run_case(
    n,
    t,
    depth,
    seed,
    chi,
    mode,
    to_backend,
    rank_max_n,
    exact_cooling=True,
    deferred_projection_order="middle_out",
):
    """Run one (n, mode) case and return a metrics dict."""
    from pepsy.optimizers import MpsStabOptimizer

    stream = random_clifford_t_circuit(n, t, depth, seed)
    n_two_qubit = sum(1 for e in stream if e[0] == "cnot")

    mode = str(mode).strip().lower()
    common = {
        "chi": chi,
        "to_backend": to_backend,
        "exact_cooling": exact_cooling,
    }
    start = time.perf_counter()
    projection_report = None
    if mode in ("immediate", "injection"):
        sim = MpsStabOptimizer.with_injection(
            n, stream, n_ancilla=1, **common
        )
        projection_report = sim.last_immediate_injection_report
        mode = "immediate"
    elif mode == "deferred":
        sim = MpsStabOptimizer.with_deferred_injection(
            n,
            stream,
            projection_order=deferred_projection_order,
            **common,
        )
        projection_report = sim.last_deferred_injection_report
    elif mode == "direct":
        sim = MpsStabOptimizer(n, **common).apply(stream)
    else:
        raise ValueError(
            f"unknown mode {mode!r}; use direct, immediate, deferred, or injection."
        )
    elapsed = time.perf_counter() - start

    rank = sim.pseudo_stabilizer_rank() if sim.n <= rank_max_n else None
    return {
        "n": int(n),
        "t": int(t),
        "mode": mode,
        "gates": len(stream),
        "two_qubit_gates": int(n_two_qubit),
        "peak_nu_bond": int(max(sim.bond_history)),
        # Backward-compatible name retained for callers of the original benchmark.
        "max_nu_bond": int(max(sim.bond_history)),
        "final_nu_bond": int(sim.state.max_bond()),
        "bond_bound_2_to_t": int(2 ** t),
        "pseudo_stabilizer_rank": None if rank is None else int(rank),
        "replay_elapsed_s": float(
            elapsed
            if mode == "direct"
            else (
                projection_report["replay_elapsed_s"]
                if mode == "deferred"
                else elapsed - projection_report["projection_elapsed_s"]
            )
        ),
        "projection_elapsed_s": float(
            0.0 if mode == "direct" else projection_report["projection_elapsed_s"]
        ),
        "pre_projection_peak_bond": (
            None if mode != "deferred" else projection_report["pre_projection_peak_bond"]
        ),
        "projection_peak_bond": (
            None if mode == "direct" else projection_report["projection_peak_bond"]
        ),
        "elapsed_s": float(elapsed),
    }


def run(args):
    n_list = [int(x) for x in str(args.n_list).split(",") if x.strip()]
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    to_backend = _resolve_backend(args.backend)
    results = []
    for n in n_list:
        for mode in modes:
            res = run_case(
                n=n,
                t=args.t,
                depth=args.depth,
                seed=args.seed,
                chi=args.chi,
                mode=mode,
                to_backend=to_backend,
                rank_max_n=args.rank_max_n,
                exact_cooling=args.exact_cooling,
                deferred_projection_order=args.deferred_projection_order,
            )
            results.append(res)
    return {
        "config": {
            "n_list": n_list,
            "t": int(args.t),
            "depth": int(args.depth),
            "seed": int(args.seed),
            "chi": args.chi,
            "modes": modes,
            "backend": args.backend or "numpy",
            "exact_cooling": bool(args.exact_cooling),
            "deferred_projection_order": args.deferred_projection_order,
        },
        "results": results,
    }


def _print_table(report):
    cfg = report["config"]
    print(
        f"# STN magic-vs-chi scaling  (t={cfg['t']} T-gates, depth={cfg['depth']}, "
        f"chi={cfg['chi']}, backend={cfg['backend']})"
    )
    print(
        f"# exact_cooling={cfg['exact_cooling']}; deferred order="
        f"{cfg['deferred_projection_order']}\n"
    )
    header = (
        f"{'mode':>10} {'N':>5} {'gates':>7} {'2q':>5} {'peak':>6} "
        f"{'final':>6} {'rank':>6} {'replay[s]':>10} {'proj-bond':>10} {'proj[s]':>9} {'total[s]':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in report["results"]:
        rank = "-" if r["pseudo_stabilizer_rank"] is None else r["pseudo_stabilizer_rank"]
        replay = "-" if r["replay_elapsed_s"] is None else f"{r['replay_elapsed_s']:.3f}"
        projection = (
            f"{r['projection_elapsed_s']:.3f}"
        )
        projection_bond = (
            "-" if r["projection_peak_bond"] is None else r["projection_peak_bond"]
        )
        print(
            f"{r['mode']:>10} {r['n']:>5} {r['gates']:>7} {r['two_qubit_gates']:>5} "
            f"{r['peak_nu_bond']:>6} {r['final_nu_bond']:>6} {str(rank):>6} "
            f"{replay:>10} {str(projection_bond):>10} {projection:>9} {r['elapsed_s']:>9.3f}"
        )
    print()
    for mode in cfg["modes"]:
        label = "immediate" if mode == "injection" else mode
        bonds = [r["peak_nu_bond"] for r in report["results"] if r["mode"] == label]
        print(
            f"{label}: peak |nu> bond over N = {bonds}"
        )
    print(
        "\nnote: immediate injection recycles one ancilla, whereas deferred MAST "
        "uses one fresh ancilla per injected rotation and reports its final "
        "projection cost separately."
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-list", default="6,12,24,48",
                        help="comma-separated qubit counts to sweep")
    parser.add_argument("--t", type=int, default=3, help="number of T gates (fixed)")
    parser.add_argument("--depth", type=int, default=6, help="Clifford circuit depth")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--chi", type=int, default=None,
                        help="max |nu> bond (None = exact)")
    parser.add_argument("--modes", default="direct,immediate,deferred",
                        help="comma-separated: direct, immediate, deferred (or injection alias)")
    parser.add_argument(
        "--exact-cooling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable constructive exact cooling (use --no-exact-cooling to isolate MAST)",
    )
    parser.add_argument(
        "--deferred-projection-order",
        default="middle_out",
        choices=("middle_out", "input", "min_span"),
        help="end-of-circuit magic-register projection order for deferred mode",
    )
    parser.add_argument("--backend", default=None,
                        help="numpy (default), torch, cupy, or jax")
    parser.add_argument("--rank-max-n", type=int, default=12,
                        help="report pseudo-stabilizer rank only for N <= this")
    parser.add_argument("--json", default=None, help="optional path to write JSON")
    return parser


def main():
    args = build_arg_parser().parse_args()
    report = run(args)
    _print_table(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
