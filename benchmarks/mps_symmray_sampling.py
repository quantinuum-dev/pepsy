"""Benchmark sparse Symmray MPS sampling against dense alternatives.

Examples
--------
Run all sampler variants for a charge-restricted bosonic U1 batch:

``python benchmarks/mps_symmray_sampling.py --symmetry U1 --samples 4096``

Run only the sparse path while comparing its prefix policies:

``python benchmarks/mps_symmray_sampling.py --fermionic --variants symmray --strategy prefix``
``python benchmarks/mps_symmray_sampling.py --fermionic --variants symmray --strategy serial``

Each variant runs in a fresh process. ``peak_rss_mib`` captures its complete
setup high-water mark, while ``resident_rss_mib`` is measured after temporary
conversion data is released. This avoids inheriting allocations from another
route while keeping conversion cost visible.
Fermionic Z2/Z2Z2 inputs are reported as sparse-only: directly expanding their
graded virtual legs into ordinary dense MPS tensors does not preserve the state.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import traceback
from time import perf_counter

import numpy as np
import quimb.tensor as qtn

from pepsy.sampling import MpsSampler
from pepsy.tensors import Fermion, SymMPS, site_charge_from_occupations


def _nonfermionic_setup(symmetry, length):
    if symmetry == "U1":
        return {0: 1, 1: 1, 2: 1}, (1,) * length
    if symmetry == "Z2":
        return {0: 1, 1: 1}, tuple(site % 2 for site in range(length))
    sectors = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    charges = tuple(
        (1, 0) if site % 2 == 0 else (0, 1)
        for site in range(length)
    )
    return sectors, charges


def _build_state(args):
    if args.fermionic:
        fermion = Fermion(spinful=True, symmetry=args.symmetry)
        sectors = fermion.physical_sectors
        occupations = fermion.half_filled_occupations(args.length)
    else:
        sectors, occupations = _nonfermionic_setup(args.symmetry, args.length)
    return SymMPS.random(
        args.length,
        symmetry=args.symmetry,
        fermionic=args.fermionic,
        bond_dim=args.bond_dim,
        phys_dim=sectors,
        site_charge=site_charge_from_occupations(occupations),
        seed=args.seed,
        dtype="complex128",
    ).mps


def _dense_mps_from_symmray(psi):
    """Expand local sparse blocks into a dense quimb MPS baseline."""
    arrays = []
    for site in range(psi.L):
        tensor = psi[site]
        site_ind = psi.site_ind(site)
        left_ind = psi.bond(site - 1, site) if site else None
        right_ind = psi.bond(site, site + 1) if site < psi.L - 1 else None
        if left_ind is None:
            data = tensor.transpose(site_ind, right_ind).data
            data = data.to_dense().reshape((1, data.shape[0], data.shape[1]))
        elif right_ind is None:
            data = tensor.transpose(left_ind, site_ind).data
            data = data.to_dense().reshape((data.shape[0], data.shape[1], 1))
        else:
            data = tensor.transpose(left_ind, site_ind, right_ind).data.to_dense()
        arrays.append(np.asarray(data))

    # quimb's ``lrp`` convention stores the physical leg last, whereas the
    # sampler's dense internal convention is ``lpr``.
    return qtn.MatrixProductState(
        [
            arrays[0][0].T,
            *(array.transpose(0, 2, 1) for array in arrays[1:-1]),
            arrays[-1][:, :, 0],
        ],
        shape="lrp",
    )


def _peak_rss_mib():
    """Return the process high-water RSS in MiB on Unix-like systems."""
    try:
        import resource  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - Windows fallback
        return None

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":  # macOS reports bytes rather than KiB.
        return rss / (1024.0**2)
    return rss / 1024.0


def _resident_rss_mib():
    """Return current RSS in MiB when the platform exposes it."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0**2)
    except (IndexError, OSError, ValueError):  # pragma: no cover - procfs edge case
        return None


def _dense_baseline_unavailable_reason(args):
    """Return why an ordinary dense MPS would be an invalid comparator."""
    if args.fermionic and args.symmetry in {"Z2", "Z2Z2"}:
        return (
            "Fermionic Z2 graded virtual-leg phases cannot be represented by "
            "locally calling to_dense() and handing the result to an ordinary "
            "quimb MPS. Only MpsSampler's Symmray route is valid."
        )
    return None


def _benchmark_variant(config, variant):
    """Build, warm, and time one sampler variant in its own process."""
    args = argparse.Namespace(**config)
    unavailable_reason = _dense_baseline_unavailable_reason(args)
    if variant != "symmray" and unavailable_reason is not None:
        return {
            "status": "unsupported",
            "reason": unavailable_reason,
        }

    started = perf_counter()
    sparse_psi = _build_state(args)
    if variant == "symmray":
        psi = sparse_psi
        sampler = MpsSampler(
            psi,
            backend="symmray",
            prefix_strategy=args.strategy,
            max_prefix_groups=args.max_prefix_groups,
        )
    else:
        psi = _dense_mps_from_symmray(sparse_psi)
        del sparse_psi
        sampler = MpsSampler(psi, backend=variant)
    setup_seconds = perf_counter() - started

    warmup_samples = min(args.samples, 32)
    configs, probabilities = sampler.sample_arrays(warmup_samples, seed=args.seed)
    if not np.allclose(
        sampler.probabilities(configs),
        probabilities,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(f"{variant} sampler returned inconsistent probabilities.")

    elapsed = []
    stats = []
    for repeat in range(args.repeats):
        start = perf_counter()
        sampler.sample_arrays(args.samples, seed=args.seed + repeat)
        elapsed.append(perf_counter() - start)
        if variant == "symmray":
            stats.append(sampler.symmray_sampling_stats)

    median_seconds = float(np.median(elapsed))
    gc.collect()
    resident_rss_mib = _resident_rss_mib()
    peak_rss_mib = _peak_rss_mib()
    if resident_rss_mib is not None and peak_rss_mib is not None:
        # Some container runtimes report ``ru_maxrss`` at a coarser cadence
        # than procfs; a peak cannot be lower than the current residency.
        peak_rss_mib = max(peak_rss_mib, resident_rss_mib)
    result = {
        "status": "ok",
        "backend": sampler.resolved_backend,
        "setup_seconds": float(setup_seconds),
        "seconds_median": median_seconds,
        "samples_per_second": float(args.samples / median_seconds),
        "peak_rss_mib": peak_rss_mib,
        "resident_rss_mib": resident_rss_mib,
    }
    if stats:
        result.update(
            {
                "conditional_evaluations_median": float(
                    np.median([item["conditional_evaluations"] for item in stats])
                ),
                "candidate_contractions_median": float(
                    np.median([item["candidate_contractions"] for item in stats])
                ),
                "charge_pruned_branches_median": float(
                    np.median([item["charge_pruned_branches"] for item in stats])
                ),
                "max_active_prefix_groups": max(
                    item["max_active_prefix_groups"] for item in stats
                ),
                "serial_fallback": any(item["serial_fallback"] for item in stats),
                "adaptive_serial_fallback": any(
                    item["adaptive_serial_fallback"] for item in stats
                ),
            }
        )
    return result


def _benchmark_worker(config, variant, result_queue):
    """Send a worker result or traceback back to the benchmark parent."""
    try:
        result_queue.put(("ok", _benchmark_variant(config, variant)))
    except BaseException:  # pragma: no cover - propagated to the CLI parent
        result_queue.put(("error", traceback.format_exc()))


def _run_isolated_variant(config, variant):
    """Run one variant separately so high-water RSS remains comparable."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_benchmark_worker,
        args=(config, variant, result_queue),
    )
    process.start()
    process.join()
    if process.exitcode:
        raise RuntimeError(
            f"{variant} benchmark worker exited with status {process.exitcode}."
        )
    status, result = result_queue.get()
    if status != "ok":
        raise RuntimeError(f"{variant} benchmark worker failed:\n{result}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symmetry",
        choices=("Z2", "U1", "U1U1", "Z2Z2"),
        default="U1",
    )
    parser.add_argument("--fermionic", action="store_true")
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strategy",
        choices=("auto", "prefix", "serial"),
        default="auto",
    )
    parser.add_argument("--max-prefix-groups", type=int, default=256)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("symmray", "native", "quimb"),
        default=("symmray", "native", "quimb"),
        help=(
            "Sampler implementations to measure. Each runs in an isolated "
            "process for comparable peak RSS."
        ),
    )
    args = parser.parse_args(argv)
    if args.length < 2 or args.bond_dim < 1 or args.samples < 1 or args.repeats < 1:
        parser.error(
            "length >= 2, bond-dim >= 1, samples >= 1, and repeats >= 1 "
            "are required"
        )

    payload = {
        "symmetry": args.symmetry,
        "fermionic": args.fermionic,
        "length": args.length,
        "bond_dim": args.bond_dim,
        "samples": args.samples,
        "strategy": args.strategy,
        "max_prefix_groups": args.max_prefix_groups,
        "variants": {
            variant: _run_isolated_variant(vars(args), variant)
            for variant in args.variants
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
