"""Belief-propagation contraction tools for pepsy (pepsy PLAN.md section 1).

This subpackage wraps quimb's belief propagation and adds two
convergence-robust improvements:

* :func:`relay_bp` -- the disordered-memory / relay-BP extension that hardens
  the 1-norm or dense 2-norm BP *fixed point* (arXiv:2506.01779).
  :func:`one_norm_bp` and :func:`two_norm_bp` provide the corresponding plain
  runners.
* :func:`loop_cluster_expand` -- the loop cluster expansion (arXiv:2510.05647),
  a systematic loop correction that becomes exact when the cluster set covers
  the system. Fixed-point BP messages are still what justify the formal loop
  cancellations and fastest finite-cluster convergence.

Most BP tools are imported explicitly as ``pepsy.bp``. The supported top-level
workflow entry points are :func:`pepsy.one_norm_bp`, :func:`pepsy.gauge_all`,
and :func:`pepsy.gauge_all_simple`.
"""

from .cluster import (
    BPCandidateScore,
    BPCandidateSelection,
    ConnectedLoop,
    LinkedClusterCache,
    LinkedClusterResult,
    LinkedClusterTerm,
    ScalarClusterCache,
    LoopClusterResult,
    linked_cluster_expand,
    loop_cluster_expand,
    norm1_gloop_expand,
    select_bp_candidate,
)
from .gauges import (
    GaugeResult,
    RelayGaugeOptions,
    compare_simple_update_gauges,
    compare_simple_update_to_bp,
    copy_gauges,
    d1bp_from_simple_update_gauges,
    d2bp_from_simple_update_gauges,
    gauge_all,
    gauge_all_simple,
    gauge_all_simple_with_bp_check,
    relay_gauge_all_simple,
    run_d1bp_from_simple_update_gauges,
    run_d2bp_from_simple_update_gauges,
    simple_update_bp_residual,
    simple_update_core_and_gauges_from_d2bp,
    simple_update_core_and_gauges_from_messages,
    simple_update_gauges_from_messages,
    simple_update_messages_from_gauges,
)
from .relay import (
    BPState,
    BPUpdateResult,
    RelayBPResult,
    one_norm_bp,
    relay_bp,
    two_norm_bp,
)
from .reduced_update import (
    ExactReducedUpdateProblem,
    LoopClusterReducedUpdateProblem,
    LoopClusterTerm,
    ReducedALSSolution,
    ReducedBondPair,
    SUClusterReducedUpdateProblem,
    exact_reduced_update_problem,
    loop_cluster_reduced_update_problem,
    prepare_reduced_bond_pair,
    solve_reduced_als,
    su_cluster_reduced_update_problem,
)

__all__ = [
    "GaugeResult",
    "ExactReducedUpdateProblem",
    "BPCandidateScore",
    "BPCandidateSelection",
    "BPState",
    "BPUpdateResult",
    "ConnectedLoop",
    "LinkedClusterCache",
    "LinkedClusterResult",
    "LinkedClusterTerm",
    "LoopClusterReducedUpdateProblem",
    "LoopClusterResult",
    "LoopClusterTerm",
    "RelayGaugeOptions",
    "ReducedALSSolution",
    "ReducedBondPair",
    "SUClusterReducedUpdateProblem",
    "ScalarClusterCache",
    "compare_simple_update_gauges",
    "compare_simple_update_to_bp",
    "copy_gauges",
    "d1bp_from_simple_update_gauges",
    "d2bp_from_simple_update_gauges",
    "exact_reduced_update_problem",
    "gauge_all",
    "gauge_all_simple",
    "gauge_all_simple_with_bp_check",
    "relay_gauge_all_simple",
    "RelayBPResult",
    "loop_cluster_reduced_update_problem",
    "loop_cluster_expand",
    "linked_cluster_expand",
    "norm1_gloop_expand",
    "one_norm_bp",
    "prepare_reduced_bond_pair",
    "relay_bp",
    "run_d1bp_from_simple_update_gauges",
    "run_d2bp_from_simple_update_gauges",
    "simple_update_bp_residual",
    "simple_update_core_and_gauges_from_d2bp",
    "simple_update_core_and_gauges_from_messages",
    "simple_update_gauges_from_messages",
    "simple_update_messages_from_gauges",
    "select_bp_candidate",
    "solve_reduced_als",
    "su_cluster_reduced_update_problem",
    "two_norm_bp",
]
