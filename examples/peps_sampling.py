"""Small direct PEPS sampling example with χ/χ′ and logarithmic weights.

Run from a Pepsy development environment: python examples/peps_sampling.py
"""

import numpy as np
import quimb.tensor as qtn

from pepsy.sampling import PepsSampler


def main():
    peps = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=7)
    sampler = PepsSampler(
        peps,
        chi=8,                 # Cached future double-layer environments.
        chi_prime=4,           # Conditioned single-layer ket boundary.
        boundary_engine="dmrg",
        cutoff="auto",
        cutoff_mode="auto",
        contraction_opt="greedy",
    )
    batch = sampler.sample_batch(samples=16, seed=11)
    print("site order:", sampler.site_order)
    print("first configuration:", batch.configs[0])
    print("first log proposal probability:", batch.log_probabilities[0])
    print("first log importance weight:", batch.log_weights[0])

    # Exact contraction is practical for this small demonstration only.
    exact = PepsSampler(peps, contraction_opt="greedy")
    log_born = np.array([exact.log_probability(c) for c in batch.configs])
    error = np.max(np.abs(batch.log_probabilities - log_born))
    print("largest sampled log-probability error versus exact:", error)
    # Truncated boundaries can change q(S); log_weights uses the original Psi(S).


if __name__ == "__main__":
    main()
