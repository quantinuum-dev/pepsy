"""Belief-propagation contraction tools for pepsy (pepsy PLAN.md section 1).

This subpackage wraps quimb's belief propagation and adds two
convergence-robust improvements:

* :func:`relay_bp` / :func:`one_norm_bp` -- the disordered-memory / relay-BP
  extension that hardens the BP *fixed point* (arXiv:2506.01779).
* :func:`loop_cluster_expand` -- the loop cluster expansion (arXiv:2510.05647),
  a systematic loop correction that converges to the exact contraction with
  cluster size and does not require BP to reach a fixed point for correctness.

It is imported explicitly as ``pepsy.bp`` (kept out of the lazy top-level
namespace while it is a prototype).
"""

from .cluster import LoopClusterResult, loop_cluster_expand
from .relay import RelayBPResult, one_norm_bp, relay_bp

__all__ = [
    "LoopClusterResult",
    "RelayBPResult",
    "loop_cluster_expand",
    "one_norm_bp",
    "relay_bp",
]
