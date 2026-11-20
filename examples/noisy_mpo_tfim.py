"""Compare noisy 4×4 TFIM MPO replay with exact Pauli-basis observables."""

import argparse
from importlib.metadata import version
import json
import sys
from time import perf_counter

import numpy as np
import quimb.tensor as qtn

from pepsy import MpoOptimizer


def square_bonds(size):
    def site(row, col):
        return row * size + (col if row % 2 == 0 else size - 1 - col)

    return [
        (site(row, col), site(other_row, other_col))
        for row in range(size)
        for col in range(size)
        for other_row, other_col in ((row, col + 1), (row + 1, col))
        if other_row < size and other_col < size
    ]


def strang_step(nsites, bonds, dt, field, p1, p2, noisy_pairs):
    """Use H = -sum(ZZ) - field * sum(X) and a terminal noise layer."""
    half_field = [("rx", -field * dt, i) for i in range(nsites)]
    interaction = [("rzz", -2 * dt, i, j) for i, j in bonds]
    noise = [("depolarize1", p1, i) for i in range(nsites)]
    noise.extend(("depolarize2", p2, i, j) for i, j in noisy_pairs)
    return half_field + interaction + half_field + noise


def adjoint_pauli_terms(observable, stream):
    """Propagate a Pauli observable backwards without dropping coefficients."""
    coefficients = {observable: 1.0}
    peak_words = 1
    for name, argument, *sites in reversed(stream):
        support = sum(1 << site for site in sites)
        if name.startswith("depolarize"):
            decay = 1 - argument * (4 / 3 if len(sites) == 1 else 16 / 15)
            coefficients = {
                (x, z): decay * value if (x | z) & support else value
                for (x, z), value in coefficients.items()
            }
            continue

        axis_x, axis_z = (support, 0) if name == "rx" else (0, support)
        cosine, sine = np.cos(argument), np.sin(argument)
        updated = {}
        for (x, z), value in coefficients.items():
            if ((axis_x & z).bit_count() + (axis_z & x).bit_count()) % 2 == 0:
                updated[x, z] = updated.get((x, z), 0.0) + value
                continue
            updated[x, z] = updated.get((x, z), 0.0) + cosine * value
            new_x, new_z = x ^ axis_x, z ^ axis_z
            # For P(x,z)=i^popcount(x&z) X^x Z^z, the anticommuting
            # contribution to U† P U is i sin(theta) times the product A P.
            phase = (
                1 + (x & z).bit_count() - (new_x & new_z).bit_count()
                + 2 * (axis_z & x).bit_count()
            ) % 4
            sign = 1 if phase == 0 else -1
            key = new_x, new_z
            updated[key] = updated.get(key, 0.0) + sign * sine * value
        coefficients = updated
        peak_words = max(peak_words, len(coefficients))
    return coefficients, peak_words


def first_unitary_state(nsites, step):
    """Evolve all |0> through the first noiseless Strang step as a statevector."""
    basis = np.arange(1 << nsites)
    state = np.zeros(len(basis), dtype=complex)
    state[0] = 1
    for name, angle, *sites in step:
        if name.startswith("depolarize"):
            break
        if name == "rx":
            flipped = basis ^ (1 << sites[0])
            state = np.cos(angle / 2) * state - 1j * np.sin(angle / 2) * state[flipped]
        else:
            i, j = sites
            eigenvalue = 1 - 2 * (((basis >> i) ^ (basis >> j)) & 1)
            state *= np.exp(-0.5j * angle * eigenvalue)
    return state


def exact_expectations(operators, state, remaining_stream):
    """Pair exact adjoint observables with the pure first-step state."""
    by_flip = {}
    peak_words = 0
    for index, operator in enumerate(operators):
        coefficients, words = adjoint_pauli_terms(operator, remaining_stream)
        for (x, z), value in coefficients.items():
            by_flip.setdefault(x, []).append((index, z, value))
        peak_words = max(peak_words, words)
    basis = np.arange(len(state))
    output = np.zeros(len(operators), dtype=complex)
    for x, queries in by_flip.items():
        # All Z masks for a given X mask are Walsh coefficients of the same
        # shifted statevector overlap. Reuse the transform for those queries.
        overlap = state * state[basis ^ x].conj()
        span = 1
        while span < len(state):
            halves = overlap.reshape(-1, 2, span)
            left = halves[:, 0, :].copy()
            right = halves[:, 1, :]
            halves[:, 0, :] = left + right
            halves[:, 1, :] = left - right
            span *= 2
        for index, z, value in queries:
            output[index] += value * 1j**((x & z).bit_count() % 4) * overlap[z]
    return output, peak_words


def readout_cores(mpo):
    for i in range(mpo.L):
        left = (mpo.bond(i - 1, i),) if i else ()
        right = (mpo.bond(i, i + 1),) if i + 1 < mpo.L else ()
        tensor = mpo[i].transpose(*left, *right, mpo.upper_ind(i), mpo.lower_ind(i))
        shape = (
            tensor.ind_size(left[0]) if left else 1,
            tensor.ind_size(right[0]) if right else 1,
            2,
            2,
        )
        yield tensor.data.reshape(shape)


def mpo_expectation(cores, observable):
    x, z = observable
    paulis = (
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.diag([1, -1]),
        np.array([[0, -1j], [1j, 0]]),
    )
    row = np.ones(1, dtype=complex)
    for i, core in enumerate(cores):
        pauli = paulis[((x >> i) & 1) + 2 * ((z >> i) & 1)]
        row = row @ np.einsum("abuv,vu->ab", core, pauli)
    return row[0]


def observables(nsites, bonds, field, values):
    return {
        "mean_z": float(np.mean(values[:nsites])),
        "mean_x": float(np.mean(values[nsites:2 * nsites])),
        "energy_per_site": float(
            -(sum(values[2 * nsites:]) + field * sum(values[nsites:2 * nsites])) / nsites
        ),
        "mean_zz": float(np.mean(values[2 * nsites:2 * nsites + len(bonds)])),
    }


def benchmark(chis, steps, mode="dmrg2", *, p1=0.01, p2=0.01,
              all_noise_bonds=False, exact_steps=2, fit_init_strategy="guess-src",
              fit_sweeps=2):
    size, dt, field = 4, 0.10, 1.05
    nsites = size**2
    bonds = square_bonds(size)
    noisy_pairs = bonds if all_noise_bonds else [(5, 6)]
    step = strang_step(nsites, bonds, dt, field, p1, p2, noisy_pairs)
    n_unitary_gates = 2 * nsites + len(bonds)
    z_sites = [(0, 1 << i) for i in range(nsites)]
    x_sites = [(1 << i, 0) for i in range(nsites)]
    zz_bonds = [(0, (1 << i) | (1 << j)) for i, j in bonds]
    operators = z_sites + x_sites + zz_bonds
    fit_options = (
        {"n_iter": fit_sweeps, "fit_init_strategy": fit_init_strategy}
        if mode.startswith("dmrg") else {}
    )
    result = {
        "size": [size, size],
        "mode": mode,
        "site_order": "row-wise snake",
        "initial_state": "all |0>",
        "dt": dt,
        "field": field,
        "depolarizing_probability": p1 if p1 == p2 else None,
        "depolarizing1_probability": p1,
        "depolarizing2_probability": p2,
        "additional_two_site_noise": noisy_pairs[0] if len(noisy_pairs) == 1 else None,
        "two_site_noise_bonds": noisy_pairs,
        "noise_events_per_step": len(step) - n_unitary_gates,
        "unitary_gates_per_step": n_unitary_gates,
        "fit_options": fit_options,
        "bonds": bonds,
        "versions": {name: version(name) for name in ("quimb", "autoray", "cotengra", "numpy")},
        "exact_reference": "First unitary step by statevector, remaining circuit by Pauli adjoint",
        "times": [],
        "cases": {},
    }
    exact_values = []
    if exact_steps:
        first_state = first_unitary_state(nsites, step)
        first_noise = step[n_unitary_gates:]
        for n in range(1, min(steps, exact_steps) + 1):
            started = perf_counter()
            values, peak_words = exact_expectations(
                operators, first_state, first_noise + step * (n - 1)
            )
            exact_values.append(values)
            result["times"].append({
                "t": n * dt,
                "exact": observables(nsites, bonds, field, values.real.tolist()),
                "exact_seconds": perf_counter() - started,
                "peak_pauli_words": peak_words,
            })
            print(f"Exact observables at t={n * dt:g}", file=sys.stderr, flush=True)

    for chi in chis:
        zero = np.diag([1.0, 0.0]).astype(complex)
        rho = qtn.MPO_product_operator([zero] * nsites)
        opt = MpoOptimizer(rho, step, chi=chi, mode=mode, contraction_opt="greedy")
        case = []
        for n in range(1, steps + 1):
            started = perf_counter()
            rho = opt.run(cutoff=1e-12, progbar=False, fidelity_samples=0, **fit_options)
            elapsed = perf_counter() - started
            cores = list(readout_cores(rho))
            unscaled_trace = mpo_expectation(cores, (0, 0))
            normalized = np.array(
                [mpo_expectation(cores, operator) / unscaled_trace for operator in operators]
            )
            observation = {
                "t": n * dt,
                "mpo": observables(nsites, bonds, field, normalized.real.tolist()),
                "local_observables": {
                    "z": normalized[:nsites].real.tolist(),
                    "x": normalized[nsites:2 * nsites].real.tolist(),
                    "zz": normalized[2 * nsites:].real.tolist(),
                },
                "max_observable_error": None,
                "max_z_error": None,
                "trace_real": float(unscaled_trace.real * 10**rho.exponent),
                "trace_imag": float(unscaled_trace.imag * 10**rho.exponent),
                "max_bond": rho.max_bond(),
                "seconds": elapsed,
            }
            if n <= len(exact_values):
                error = np.abs(normalized - exact_values[n - 1])
                observation["max_observable_error"] = float(np.max(error))
                observation["max_z_error"] = float(np.max(error[:nsites]))
            case.append(observation)
            print(f"{mode} chi={chi}, t={n * dt:g}, {elapsed:.2f} s", file=sys.stderr, flush=True)
        result["cases"][str(chi)] = case
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chi", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--mode", choices=("dmrg", "dmrg2", "direct"), default="dmrg2")
    parser.add_argument("--p1", type=float, default=0.01, help="single-site depolarizing probability")
    parser.add_argument("--p2", type=float, default=0.01, help="two-site depolarizing probability")
    parser.add_argument("--all-noise-bonds", action="store_true", help="apply p2 on all 24 bonds")
    parser.add_argument("--exact-steps", type=int, default=2, help="number of early exact comparisons")
    parser.add_argument("--fit-init-strategy", default="guess-src", help="initial MPO for DMRG FIT")
    parser.add_argument("--fit-sweeps", type=int, default=2, help="sweep budget per FIT")
    args = parser.parse_args()
    if args.steps < 1 or min(args.chi) < 1 or args.fit_sweeps < 1:
        parser.error("--steps, --chi, and --fit-sweeps must be positive")
    if args.exact_steps < 0:
        parser.error("--exact-steps must be nonnegative")
    if not all(0.0 <= probability <= 1.0 for probability in (args.p1, args.p2)):
        parser.error("--p1 and --p2 must be probabilities in [0, 1]")
    print(json.dumps(benchmark(
        args.chi, args.steps, args.mode, p1=args.p1, p2=args.p2,
        all_noise_bonds=args.all_noise_bonds, exact_steps=args.exact_steps,
        fit_init_strategy=args.fit_init_strategy, fit_sweeps=args.fit_sweeps,
    ), indent=2))


if __name__ == "__main__":
    main()
