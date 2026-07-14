"""Tests for the loop cluster expansion wrapper (pepsy.bp.cluster).

Covers the thin quimb wrapper (A) and an empirical validation (D) of a key
finite-system property of the loop cluster expansion: at a system-covering
cluster the estimate is exact *independently of whether the BP messages
converged*.  Away from that limit, fixed-point BP messages are what justify the
formal loop cancellations and the cleanest finite-cluster behavior.
"""

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")
pytest.importorskip("quimb.tensor.belief_propagation")

from pepsy.bp import (  # noqa: E402
    LoopClusterResult,
    ScalarClusterCache,
    compare_simple_update_gauges,
    compare_simple_update_to_bp,
    d1bp_from_simple_update_gauges,
    gauge_all_simple_with_bp_check,
    loop_cluster_expand,
    norm1_gloop_expand,
    run_d1bp_from_simple_update_gauges,
    simple_update_bp_residual,
    simple_update_gauges_from_messages,
)


def _peps_and_exact():
    """A small deterministic 3x3, D=2 PEPS and its exact 2-norm."""
    peps = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1)
    exact = float((peps.H & peps).contract(optimize="auto-hq"))
    return peps, exact


def test_exports():
    from pepsy import bp

    assert {
        "LoopClusterResult",
        "ScalarClusterCache",
        "compare_simple_update_gauges",
        "compare_simple_update_to_bp",
        "d1bp_from_simple_update_gauges",
        "gauge_all_simple_with_bp_check",
        "loop_cluster_expand",
        "norm1_gloop_expand",
        "run_d1bp_from_simple_update_gauges",
        "simple_update_bp_residual",
        "simple_update_gauges_from_messages",
    } <= set(bp.__all__)


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
    assert err_big < err_small  # full cluster removes boundary-message error


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


def test_system_covering_cluster_is_message_independent():
    # At a system-covering cluster the estimate is exact and independent of
    # whether the BP messages converged. This is weaker than saying arbitrary
    # non-fixed messages give the formal finite-cluster BP loop expansion.
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


def _scalar_two_site_tree():
    """Closed scalar tree whose exact contraction is 13."""
    return qtn.TensorNetwork(
        [
            qtn.Tensor(np.array([1.0, 2.0]), inds=("x",)),
            qtn.Tensor(np.array([3.0, 5.0]), inds=("x",)),
        ]
    )


def _projected_scalar_ladder():
    """A six-site 3x2 scalar TN with two elementary plaquette loops."""
    psi = qtn.PEPS.rand(3, 2, bond_dim=2, phys_dim=2, seed=4)
    tn = psi.copy()
    vector = np.array([0.8, 0.6])
    for ix in list(tn.outer_inds()):
        (tid,) = tn._get_tids_from_inds(ix)
        tn.tensor_map[tid].vector_reduce_(ix, vector)
    return tn


def _projected_scalar_ladder_su_core():
    """A projected scalar ladder with simple-update gauges removed to a store."""
    tn = _projected_scalar_ladder()
    gauges = {}
    core = tn.copy()
    core.gauge_all_simple_(gauges=gauges, max_iterations=20, tol=1e-12)
    return core, gauges


def test_1norm_bp_baseline_is_exact_on_tree():
    tn = _scalar_two_site_tree()
    exact = tn.contract()

    res = loop_cluster_expand(tn, gloops=0, norm="1norm", tol=1e-12)

    assert res.bp_converged is True
    assert np.isclose(res.estimate, exact)
    assert sorted((len(region), count) for region, count in res.region_counts) == [
        (1, 1),
        (1, 1),
    ]


def test_1norm_ladder_overlap_counts_and_cached_expansion():
    tn = _projected_scalar_ladder()
    cache = ScalarClusterCache()
    res = loop_cluster_expand(tn, gloops=4, norm="1norm", cache=cache, tol=1e-12)

    assert sorted((len(region), count) for region, count in res.region_counts) == [
        (2, -1),
        (4, 1),
        (4, 1),
    ]
    assert len(cache.loops_by_max_size) == 1
    assert np.isclose(res.expand(4), res.estimate)


def test_1norm_system_covering_cluster_is_exact():
    tn = _projected_scalar_ladder()
    exact = tn.contract(optimize="auto-hq")
    res = loop_cluster_expand(tn, gloops=6, norm="1norm", tol=1e-12)

    assert np.isclose(res.estimate, exact)
    assert [(len(region), count) for region, count in res.region_counts] == [(6, 1)]


def test_simple_update_gauge_messages_round_trip_to_su_shape():
    core, gauges = _projected_scalar_ladder_su_core()

    bp = d1bp_from_simple_update_gauges(core, gauges)
    bp_gauges = simple_update_gauges_from_messages(bp)
    comparison = compare_simple_update_gauges(gauges, bp_gauges)

    assert comparison["num_bonds"] == len(gauges)
    assert comparison["max_rel_l2"] < 1e-12


def test_simple_update_bp_residual_is_finite():
    core, gauges = _projected_scalar_ladder_su_core()

    residual = simple_update_bp_residual(core, gauges)

    assert np.isfinite(residual)
    assert residual >= 0.0


def test_gauge_all_simple_with_bp_check_records_su_and_bp_traces():
    tn = _projected_scalar_ladder()

    core, gauges, info = gauge_all_simple_with_bp_check(
        tn,
        max_iterations=3,
        su_tol=0.0,
        bp_check_every=1,
    )

    assert core is not tn
    assert len(gauges) == len(tn.inner_inds())
    assert info["iterations"] == 3
    assert len(info["su_max_sdiffs"]) == 3
    assert len(info["bp_max_mdiffs"]) == 3
    assert all(np.isfinite(x) for x in info["su_max_sdiffs"])
    assert all(np.isfinite(x) for x in info["bp_max_mdiffs"])


def test_norm1_gloop_expand_with_su_gauges_full_region_is_exact():
    core, gauges = _projected_scalar_ladder_su_core()
    full = core.copy()
    full.gauge_simple_insert(gauges)
    exact = full.contract(optimize="auto-hq")

    estimate = norm1_gloop_expand(
        core,
        gloops=(tuple(core.tensor_map),),
        gauges=gauges,
        autoreduce=False,
        optimize="auto-hq",
    )

    assert np.isclose(estimate, exact)


def test_converged_su_gauges_make_tree_autoreduce_trivial():
    tn = _scalar_two_site_tree()
    gauges = {}
    core = tn.copy()
    core.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-12)
    tree_region = (tuple(core.tensor_map),)

    z_default = norm1_gloop_expand(
        core,
        gloops=tree_region,
        gauges=gauges,
        autoreduce=True,
    )
    z_keep_trees = norm1_gloop_expand(
        core,
        gloops=tree_region,
        gauges=gauges,
        autoreduce=False,
    )

    assert np.isclose(z_default, z_keep_trees)


def test_norm1_gloop_expand_can_use_relay_bp_from_su_initialization():
    core, gauges = _projected_scalar_ladder_su_core()
    full = core.copy()
    full.gauge_simple_insert(gauges)
    exact = full.contract(optimize="auto-hq")

    res = norm1_gloop_expand(
        core,
        gloops=6,
        gauges=gauges,
        run_bp=True,
        bp_runner="relay",
        relay_opts={"num_relays": 2, "seed": 0},
        tol=1e-10,
        optimize="auto-hq",
        return_result=True,
    )
    comparison = compare_simple_update_to_bp(gauges, res.bp)

    assert res.bp_converged is True
    assert np.isclose(res.estimate, exact)
    assert comparison["num_bonds"] == len(gauges)
    assert np.isfinite(comparison["max_rel_l2"])


def test_loop_cluster_expand_relay_from_su_drops_init_only_options():
    core, gauges = _projected_scalar_ladder_su_core()

    res = loop_cluster_expand(
        core,
        gloops=4,
        norm="1norm",
        gauges=gauges,
        run_bp=True,
        bp_runner="relay",
        message_power=1.0,
        relay_opts={"num_relays": 2, "seed": 0},
        max_iterations=100,
        tol=1e-10,
        require_fixed_point=False,
    )

    assert np.isfinite(res.estimate)


def test_run_d1bp_from_simple_update_gauges_uses_relay_api():
    core, gauges = _projected_scalar_ladder_su_core()

    res = run_d1bp_from_simple_update_gauges(
        core,
        gauges,
        use_relay=True,
        run_opts={"max_iterations": 100, "tol": 1e-10},
        relay_opts={"num_relays": 2, "seed": 0},
    )

    assert res.converged
    assert len(res.snapshot()) == len(res.messages)


def test_run_d1bp_from_simple_update_gauges_forwards_bp_options():
    core, gauges = _projected_scalar_ladder_su_core()

    res = run_d1bp_from_simple_update_gauges(
        core,
        gauges,
        bp_opts={"normalize": "L1", "distance": "cosine"},
        run_opts={"max_iterations": 1, "tol": 0.0},
    )

    assert res.bp._normalize == "L1"
    assert res.bp._distance == "cosine"


def test_bad_norm_raises():
    peps, _ = _peps_and_exact()
    with pytest.raises(ValueError):
        loop_cluster_expand(peps, gloops=3, norm="3norm")
