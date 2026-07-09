"""Benchmark the Stabilizer Tensor Network (STN) magic-vs-chi scaling.

Demonstrates the central STN claim (Masot-Llima & Garcia-Saez, PRL 133, 230601,
Fig. 2; magic-state injection follow-up arXiv:2411.12482): the coefficient MPS
``|nu>`` bond of ``pepsy.MpsStabOptimizer`` is bounded by the *number of
non-Clifford (T) gates* ``t`` — at most ``2**t`` — and stays flat as the qubit
count ``N`` grows.  Clifford gates are free (they only update the stim tableau),
so a T-doped Clifford circuit costs ``O(poly N)`` at fixed ``t``.

For each ``N`` the script builds a deterministic random Clifford circuit with
``t`` interspersed ``T`` gates and runs it two ways:

* direct: every ``T`` acts on ``|nu>`` via the exact rotation path;
* injection: every ``T`` is teleported through a single recycled magic ancilla
  (``with_injection``), keeping the non-Clifford cost off ``|nu>``.

It records the max ``|nu>`` bond, the pseudo-stabilizer rank (for small ``N``),
and wall time, and emits JSON plus a human-readable table.

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


def run_case(n, t, depth, seed, chi, mode, to_backend, rank_max_n):
    """Run one (n, mode) case and return a metrics dict."""
    from pepsy.optimizers import MpsStabOptimizer

    stream = random_clifford_t_circuit(n, t, depth, seed)
    n_two_qubit = sum(1 for e in stream if e[0] == "cnot")

    start = time.perf_counter()
    if mode == "injection":
        sim = MpsStabOptimizer.with_injection(
            n, stream, n_ancilla=1, chi=chi, to_backend=to_backend
        )
    else:
        sim = MpsStabOptimizer(n, chi=chi, to_backend=to_backend).apply(stream)
    elapsed = time.perf_counter() - start

    rank = sim.pseudo_stabilizer_rank() if n <= rank_max_n else None
    return {
        "n": int(n),
        "t": int(t),
        "mode": mode,
        "gates": len(stream),
        "two_qubit_gates": int(n_two_qubit),
        "max_nu_bond": int(sim.state.max_bond()),
        "bond_bound_2_to_t": int(2 ** t),
        "pseudo_stabilizer_rank": None if rank is None else int(rank),
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
        },
        "results": results,
    }


def _print_table(report):
    cfg = report["config"]
    print(
        f"# STN magic-vs-chi scaling  (t={cfg['t']} T-gates, depth={cfg['depth']}, "
        f"chi={cfg['chi']}, backend={cfg['backend']})"
    )
    print(f"# bond bound 2^t = {2 ** cfg['t']}; expect max |nu> bond flat in N\n")
    header = f"{'mode':>10} {'N':>5} {'gates':>7} {'2q':>5} {'maxBond':>8} {'rank':>6} {'time[s]':>9}"
    print(header)
    print("-" * len(header))
    for r in report["results"]:
        rank = "-" if r["pseudo_stabilizer_rank"] is None else r["pseudo_stabilizer_rank"]
        print(
            f"{r['mode']:>10} {r['n']:>5} {r['gates']:>7} {r['two_qubit_gates']:>5} "
            f"{r['max_nu_bond']:>8} {str(rank):>6} {r['elapsed_s']:>9.3f}"
        )
    # headline: is the bond flat in N (per mode)?
    print()
    for mode in cfg["modes"]:
        bonds = [r["max_nu_bond"] for r in report["results"] if r["mode"] == mode]
        print(
            f"{mode}: max |nu> bond over N = {bonds}  "
            f"(<= 2^t = {2 ** cfg['t']}: {all(b <= 2 ** cfg['t'] for b in bonds)})"
        )
    print(
        "\nnote: both modes keep the |nu> bond bounded by 2^t and flat in N (the STN "
        "property).\n      'injection' additionally confines magic to a recyclable "
        "ancilla; its cost\n      depends on the ancilla-to-data distance (localizer "
        "swaps), so it can be\n      slower than 'direct' for small t on a 1D layout."
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
    parser.add_argument("--modes", default="direct,injection",
                        help="comma-separated: direct, injection")
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
