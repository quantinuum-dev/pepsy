"""Belief-propagation contraction tools for pepsy (pepsy PLAN.md section 1).

This subpackage wraps quimb's belief propagation and adds two
convergence-robust improvements:

* :func:`relay_bp` / :func:`one_norm_bp` -- the disordered-memory / relay-BP
  extension that hardens the BP *fixed point* (arXiv:2506.01779).
* :func:`loop_cluster_expand` -- the loop cluster expansion (arXiv:2510.05647),
  a systematic loop correction that becomes exact when the cluster set covers
  the system. Fixed-point BP messages are still what justify the formal loop
  cancellations and fastest finite-cluster convergence.

Most BP tools are imported explicitly as ``pepsy.bp``. The supported top-level
workflow entry points are :func:`pepsy.gauge_all` and
:func:`pepsy.gauge_all_simple`.
"""

from .cluster import (
    ScalarClusterCache,
    LoopClusterResult,
    loop_cluster_expand,
    norm1_gloop_expand,
)
from .gauges import (
    GaugeResult,
    RelayGaugeOptions,
    compare_simple_update_gauges,
    compare_simple_update_to_bp,
    copy_gauges,
    d1bp_from_simple_update_gauges,
    gauge_all,
    gauge_all_simple,
    gauge_all_simple_with_bp_check,
    relay_gauge_all_simple,
    run_d1bp_from_simple_update_gauges,
    simple_update_bp_residual,
    simple_update_core_and_gauges_from_messages,
    simple_update_gauges_from_messages,
    simple_update_messages_from_gauges,
)
from .relay import RelayBPResult, one_norm_bp, relay_bp

__all__ = [
    "GaugeResult",
    "LoopClusterResult",
    "RelayGaugeOptions",
    "ScalarClusterCache",
    "compare_simple_update_gauges",
    "compare_simple_update_to_bp",
    "copy_gauges",
    "d1bp_from_simple_update_gauges",
    "gauge_all",
    "gauge_all_simple",
    "gauge_all_simple_with_bp_check",
    "relay_gauge_all_simple",
    "RelayBPResult",
    "loop_cluster_expand",
    "norm1_gloop_expand",
    "one_norm_bp",
    "relay_bp",
    "run_d1bp_from_simple_update_gauges",
    "simple_update_bp_residual",
    "simple_update_core_and_gauges_from_messages",
    "simple_update_gauges_from_messages",
    "simple_update_messages_from_gauges",
]
