"""Tests for the loop cluster expansion wrapper (pepsy.bp.cluster).

Covers the thin quimb wrapper (A) and an empirical validation (D) of the
central claim of arXiv:2510.05647: the loop cluster expansion converges to the
exact contraction with cluster size, and at a system-covering cluster the
estimate is exact *independently of whether the BP messages converged*.
"""

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")
pytest.importorskip("quimb.tensor.belief_propagation")

from pepsy.bp import LoopClusterResult, loop_cluster_expand


def _peps_and_exact():
    """A small deterministic 3x3, D=2 PEPS and its exact 2-norm."""
    peps = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1)
    exact = float((peps.H & peps).contract(optimize="auto-hq"))
    return peps, exact


def test_exports():
    from pepsy import bp

    assert {"LoopClusterResult", "loop_cluster_expand"} <= set(bp.__all__)


def test_cluster_expansion_converges_to_exact():
    peps, exact = _peps_and_exact()

    # small cluster ~ the (poor, for a random PEPS) plain-BP estimate
    res_small = loop_cluster_expand(peps, gloops=3)
    err_small = abs(res_small.estimate - exact) / abs(exact)

    # system-covering cluster -> exact
    res_big = loop_cluster_expand(peps, gloops=12)
    err_big = abs(res_big.estimate - exact) / abs(exact)

    assert isinstance(res_big, LoopClusterResult)
    assert res_big.norm == "2norm" and res_big.combine == "prod"
    assert res_big.bp_converged is True
    assert err_big < 1e-8  # converged to exact
    assert err_big < err_small  # loop clusters systematically improve BP


def test_error_decreases_with_cluster_size():
    peps, exact = _peps_and_exact()
    # reuse one converged BP fixed point and sweep the cluster size
    res = loop_cluster_expand(peps, gloops=3)
    errs = [
        abs(res.expand(c) - exact) / abs(exact) for c in (3, 6, 8, 12)
    ]
    # first vs last: a large, systematic reduction toward exact
    assert errs[-1] < 1e-8
    assert errs[-1] < errs[0] / 10


def test_bp_convergence_not_required_for_correctness():
    # THE claim (arXiv:2510.05647): at a system-covering cluster the estimate is
    # exact and independent of whether the BP messages converged.
    peps, exact = _peps_and_exact()

    res_conv = loop_cluster_expand(peps, gloops=12, max_iterations=500, tol=1e-12)
    res_unconv = loop_cluster_expand(peps, gloops=12, max_iterations=1, tol=0.0)

    # the two runs really are in different message states ...
    assert res_conv.bp_converged is True
    assert res_unconv.bp_converged is False
    # ... yet both hit the exact value, and agree with each other
    assert abs(res_conv.estimate - exact) / abs(exact) < 1e-8
    assert abs(res_unconv.estimate - exact) / abs(exact) < 1e-8
    assert abs(res_conv.estimate - res_unconv.estimate) / abs(exact) < 1e-8


def test_message_reuse_expand_matches_fresh_run():
    peps, exact = _peps_and_exact()
    # BP is deterministic, so reusing a converged fixed point at a new cluster
    # size must match a fresh converged run at that size.
    res6 = loop_cluster_expand(peps, gloops=6)
    res8 = loop_cluster_expand(peps, gloops=8)
    assert abs(res6.expand(8) - res8.estimate) < 1e-8 * abs(exact)


def test_sum_formula_needs_1norm():
    peps, _ = _peps_and_exact()
    with pytest.raises(ValueError):
        loop_cluster_expand(peps, gloops=6, combine="sum")  # norm='2norm' default
    # and via the result helper
    res = loop_cluster_expand(peps, gloops=6)
    with pytest.raises(ValueError):
        res.expand(6, combine="sum")


def test_1norm_sum_and_prod_run():
    peps, _ = _peps_and_exact()
    tnorm = peps.H & peps  # closed scalar tensor network
    for combine in ("prod", "sum"):
        res = loop_cluster_expand(tnorm, gloops=6, norm="1norm", combine=combine)
        assert np.isfinite(res.estimate)
        assert res.norm == "1norm" and res.combine == combine


def test_bad_norm_raises():
    peps, _ = _peps_and_exact()
    with pytest.raises(ValueError):
        loop_cluster_expand(peps, gloops=3, norm="3norm")
