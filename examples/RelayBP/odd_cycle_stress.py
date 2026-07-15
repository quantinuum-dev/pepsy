"""Stress Relay D1BP on polarized, near-deterministic odd cycles.

This is a convergence stress test, not an accuracy claim.  The networks are
positive but strongly frustrated, so their exact scalar contractions remain
available while ordinary loopy BP has a poor uncontrolled approximation.
"""

from __future__ import annotations

import warnings

import numpy as np
import quimb.tensor as qtn
from quimb.tensor.belief_propagation import D1BP

from pepsy.bp import one_norm_bp, relay_bp


def _odd_antiferromagnetic_cycle(epsilon):
    """Return a positive triangle whose edge factors nearly flip a bit."""
    factor = np.array([[epsilon, 1.0], [1.0, epsilon]])
    return qtn.TensorNetwork(
        [
            qtn.Tensor(factor, inds=("ab", "ca")),
            qtn.Tensor(factor, inds=("ab", "bc")),
            qtn.Tensor(factor, inds=("bc", "ca")),
        ]
    )


def _polarized_messages(tn):
    """Choose a deterministic non-fixed initial message for every edge end."""
    return {
        key: np.array([1.0, 0.0])
        for key in D1BP(tn, update="parallel").messages
    }


def run_stress_cases():
    """Run two exact-reference cases where parallel D1BP stalls."""
    records = []
    for epsilon in (1e-3, 1e-2):
        tn = _odd_antiferromagnetic_cycle(epsilon)
        exact = float(tn.contract())
        initial = _polarized_messages(tn)
        common = {
            "method": "d1bp",
            "init_messages": initial,
            "update": "parallel",
            "diis": False,
            "max_iterations": 100,
            "tol": 1e-10,
        }
        # The stall is expected and recorded below, so avoid duplicating it as
        # a Quimb warning in this runnable comparison.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Belief propagation did not converge.*",
                category=UserWarning,
            )
            plain = one_norm_bp(tn, **common)
        relay = relay_bp(
            tn,
            **common,
            num_relays=5,
            memory_first_leg=True,
            gamma_range=(0.2, 0.8),
            seed=0,
        )
        relay_estimate = float(relay.contract())
        records.append(
            {
                "epsilon": epsilon,
                "exact": exact,
                "plain_converged": plain.converged,
                "plain_max_mdiff": plain.max_mdiff,
                "relay_converged": relay.converged,
                "relay_iterations": relay.iterations,
                "relay_num_legs": relay.num_legs_run,
                "relay_max_mdiff": relay.max_mdiff,
                "relay_relative_error": abs(relay_estimate - exact) / abs(exact),
            }
        )
    return records


def main():
    """Print strict convergence and exact-reference accuracy for each case."""
    for record in run_stress_cases():
        print(
            f"epsilon={record['epsilon']:.0e} "
            f"plain_converged={record['plain_converged']} "
            f"plain_max_mdiff={record['plain_max_mdiff']:.3e} "
            f"relay_converged={record['relay_converged']} "
            f"relay_iterations={record['relay_iterations']} "
            f"relay_relative_error={record['relay_relative_error']:.3e}"
        )


if __name__ == "__main__":
    main()
