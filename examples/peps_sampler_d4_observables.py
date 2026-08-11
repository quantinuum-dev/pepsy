"""Compare 4x4 D=4 PEPS boundary sampling with exact diagonal observables.

Run, for example::

    python examples/peps_sampler_d4_observables.py --samples 512

The exact observables are computed from one- and two-site norm density
matrices. Boundary samples use the returned PEPS amplitudes and proposal
probabilities to form importance-weighted estimates.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import quimb.tensor as qtn

import pepsy


L = 4
SITES = tuple((x, y) for y in range(L) for x in range(L))


def _scalar(value):
    """Extract a scalar from a Quimb tensor or an array scalar."""
    return np.asarray(getattr(value, "data", value)).reshape(()).item()


def _one_site_z(sampler, site, optimizer):
    """Compute an exact one-site Z expectation from the full norm network."""
    rho, _ = sampler._local_rho(sampler._norm.copy(), site)
    diagonal = np.array(np.real(np.diag(rho)), copy=True)
    diagonal /= diagonal.sum()
    return float(diagonal[0] - diagonal[1])


def _two_site_zz(sampler, site_a, site_b, optimizer):
    """Compute an exact diagonal two-site ZZ expectation."""
    network = sampler._norm.copy()
    output_inds = []
    for i, site in enumerate((site_a, site_b)):
        tag = sampler._site_tags[site]
        ket_ind = sampler._site_inds[site]
        bra_ind = f"{ket_ind}__benchmark_bra_{i}"
        network.select([tag, "BRA"], which="all").reindex_({ket_ind: bra_ind})
        output_inds.extend((ket_ind, bra_ind))

    rho = network.contract(
        all,
        output_inds=output_inds,
        optimize=optimizer,
    )
    diagonal = np.real(np.einsum("iijj->ij", rho.data))
    diagonal /= diagonal.sum()
    z = np.array([1.0, -1.0])
    return float(z @ diagonal @ z)


def _exact_observables(sampler, optimizer):
    """Return exact diagonal observables for the benchmark state."""
    observables = {
        "Z(0,0)": _one_site_z(sampler, (0, 0), optimizer),
        "Z(3,3)": _one_site_z(sampler, (3, 3), optimizer),
        "ZZ((0,0),(1,0))": _two_site_zz(
            sampler, (0, 0), (1, 0), optimizer
        ),
        "ZZ((0,0),(0,1))": _two_site_zz(
            sampler, (0, 0), (0, 1), optimizer
        ),
    }
    observables["Z_mean"] = float(
        np.mean([_one_site_z(sampler, site, optimizer) for site in SITES])
    )
    return observables


def _sample_observables(result, norm):
    """Return raw and importance-weighted estimates from a sample result."""
    configs = np.asarray(result.configs, dtype=int)
    omega_m, omega_e = result.omegas
    amplitude_m, amplitude_e = result.ps
    omega_m = np.asarray(omega_m)
    omega_e = np.asarray(omega_e, dtype=float)
    amplitude_m = np.asarray(amplitude_m)
    amplitude_e = np.asarray(amplitude_e, dtype=float)

    # The scaled result fields avoid underflow; reconstruct weights in log10
    # space before exponentiating the modest benchmark-sized values.
    log_weights = (
        2.0 * (np.log(np.abs(amplitude_m)) + amplitude_e * math.log(10.0))
        - (np.log(omega_m) + omega_e * math.log(10.0))
        - math.log(norm)
    )
    weights = np.exp(log_weights)

    z00 = 1.0 - 2.0 * configs[:, 0]
    z33 = 1.0 - 2.0 * configs[:, 15]
    z10 = 1.0 - 2.0 * configs[:, 1]
    z01 = 1.0 - 2.0 * configs[:, 4]
    values = {
        "Z(0,0)": z00,
        "Z(3,3)": z33,
        "ZZ((0,0),(1,0))": z00 * z10,
        "ZZ((0,0),(0,1))": z00 * z01,
        "Z_mean": 1.0 - 2.0 * configs.mean(axis=1),
    }
    return {
        name: (float(value.mean()), float(np.mean(weights * value)))
        for name, value in values.items()
    }, float(weights.mean()), float(
        weights.sum() ** 2 / (len(weights) * np.sum(weights**2))
    )


def main():
    """Build the state, compute exact values, and compare both boundaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--sample-chi", type=int, default=8)
    parser.add_argument("--marginal-chi", type=int, default=8)
    parser.add_argument("--engine", choices=("quimb-mps", "dmrg", "both"), default="both")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--sample-seed", type=int, default=20260807)
    parser.add_argument("--max-repeats", type=int, default=8)
    parser.add_argument("--max-time", type=float, default=0.75)
    args = parser.parse_args()

    peps = qtn.PEPS.rand(
        Lx=L,
        Ly=L,
        bond_dim=4,
        phys_dim=2,
        seed=args.seed,
        dtype="complex128",
    )
    optimizer = pepsy.build_contraction(
        parallel=False,
        progbar=False,
        max_repeats=args.max_repeats,
        max_time=args.max_time,
    )
    exact = pepsy.PepsSampler(peps, contraction_opt=optimizer)
    norm = float(np.real(_scalar(exact._norm.contract(all, optimize=optimizer))))
    exact_values = _exact_observables(exact, optimizer)

    print(f"4x4 PEPS, D=4, samples={args.samples}")
    print(f"sample_chi={args.sample_chi}, marginal_chi={args.marginal_chi}")
    print("exact:")
    for name, value in exact_values.items():
        print(f"  {name:24s} {value:+.8f}")

    engines = ("quimb-mps", "dmrg") if args.engine == "both" else (args.engine,)
    for engine in engines:
        start = time.perf_counter()
        sampler = pepsy.PepsSampler(
            peps,
            sample_chi=args.sample_chi,
            marginal_chi=args.marginal_chi,
            boundary_engine=engine,
            ket_compression="quimb",
            contraction_opt=optimizer,
        )
        result = sampler.sample_batch(args.samples, seed=args.sample_seed)
        estimates, weight_mean, ess_fraction = _sample_observables(result, norm)
        print(f"\n{engine}: setup+sampling={time.perf_counter() - start:.2f}s")
        print(f"  weight_mean={weight_mean:.6f}, ESS_fraction={ess_fraction:.3f}")
        print(f"  prefix_groups={sampler.batch_stats}")
        for name, exact_value in exact_values.items():
            raw, importance = estimates[name]
            print(
                f"  {name:24s} exact={exact_value:+.8f} "
                f"raw={raw:+.8f} importance={importance:+.8f}"
            )


if __name__ == "__main__":
    main()
