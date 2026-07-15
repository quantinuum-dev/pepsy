"""Compare raw, simple-update-initialized, and Relay D1BP to exact contraction.

Run from the repository root after installing pepsy and quimb::

    python examples/RelayBP/simple_update_relay_comparison.py

The network is deliberately tiny so that its exact contraction is an
independent reference.  It is still loopy, so D1BP is an approximation: the
example records accuracy rather than treating agreement with exact contraction
as a convergence criterion.
"""

from __future__ import annotations

import quimb.tensor as qtn

from pepsy.bp import one_norm_bp, run_d1bp_from_simple_update_gauges


def run_comparison():
    """Run the three D1BP initializations and return reproducible metrics."""
    tn = qtn.TN2D_classical_ising_partition_function(3, 3, beta=0.2)
    exact = float(tn.contract(optimize="auto-hq"))

    # This is Quimb's actual simple-update routine. It conditions the tensors
    # while moving the diagonal bond environments into ``gauges``.
    core = tn.copy()
    gauges = {}
    core.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-12)
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)
    su_representation_error = abs(float(rebuilt.contract()) - exact) / abs(exact)

    runs = {
        "plain_d1bp": one_norm_bp(
            tn,
            method="d1bp",
            max_iterations=1000,
            tol=1e-10,
        ),
        "su_initialized_d1bp": run_d1bp_from_simple_update_gauges(
            core,
            gauges,
            run_opts={"max_iterations": 1000, "tol": 1e-10},
        ),
        "su_initialized_relay_d1bp": run_d1bp_from_simple_update_gauges(
            core,
            gauges,
            use_relay=True,
            run_opts={"max_iterations": 1000, "tol": 1e-10},
            relay_opts={
                "num_relays": 3,
                # A plain first leg preserves the ordinary fixed point on an
                # easy instance; the other legs exercise per-node memory.
                "memory_first_leg": False,
                "gamma_range": (0.1, 0.2),
                "seed": 0,
            },
        ),
    }

    records = {}
    for name, result in runs.items():
        estimate = float(result.contract())
        records[name] = {
            "converged": result.converged,
            "quimb_converged": result.quimb_converged,
            "iterations": result.iterations,
            "num_legs": result.num_legs_run,
            "max_mdiff": result.max_mdiff,
            "estimate": estimate,
            "relative_error": abs(estimate - exact) / abs(exact),
        }

    return {
        "exact": exact,
        "simple_update_representation_error": su_representation_error,
        "runs": records,
    }


def main():
    """Print the convergence and exact-reference accuracy comparison."""
    comparison = run_comparison()
    print(f"exact contraction: {comparison['exact']:.12g}")
    print(
        "simple-update representation relative error: "
        f"{comparison['simple_update_representation_error']:.3e}"
    )
    print(
        "name                         converged  legs  iterations  "
        "relative error"
    )
    for name, record in comparison["runs"].items():
        print(
            f"{name:28} {str(record['converged']):9} "
            f"{record['num_legs']:5d} {record['iterations']:11d} "
            f"{record['relative_error']:.3e}"
        )


if __name__ == "__main__":
    main()
