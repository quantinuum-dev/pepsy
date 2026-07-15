"""Tests for the loop cluster expansion wrapper (pepsy.bp.cluster).

Covers the thin quimb wrapper (A) and an empirical validation (D) of a key
finite-system property of the loop cluster expansion: at a system-covering
cluster the estimate is exact *independently of whether the BP messages
converged*.  Away from that limit, fixed-point BP messages are what justify the
formal loop cancellations and the cleanest finite-cluster behavior.
"""

import numpy as np
import pytest
from pathlib import Path
import runpy

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
    relay_gauge_all_simple,
    run_d1bp_from_simple_update_gauges,
    simple_update_bp_residual,
    simple_update_core_and_gauges_from_messages,
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
        "relay_gauge_all_simple",
        "run_d1bp_from_simple_update_gauges",
        "simple_update_bp_residual",
        "simple_update_core_and_gauges_from_messages",
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


def _positive_triangle():
    """A small positive loopy scalar TN suitable for D1BP and real SU."""
    return qtn.TensorNetwork(
        [
            qtn.Tensor(
                np.array([[1.1, 0.3], [0.2, 0.9]]), inds=("ab", "ca")
            ),
            qtn.Tensor(
                np.array([[0.8, 0.4], [0.5, 1.2]]), inds=("ab", "bc")
            ),
            qtn.Tensor(
                np.array([[0.7, 0.6], [0.9, 1.0]]), inds=("bc", "ca")
            ),
        ]
    )


def _real_su_triangle():
    """Return a Quimb simple-update core and its externally stored gauges."""
    tn = _positive_triangle()
    gauges = {}
    core = tn.copy()
    core.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-12)
    return tn, core, gauges


def _assert_same_tensor_data(left, right):
    assert set(left.tensor_map) == set(right.tensor_map)
    for tid in left.tensor_map:
        lt = left.tensor_map[tid]
        rt = right.tensor_map[tid]
        assert lt.inds == rt.inds
        np.testing.assert_allclose(lt.data, rt.data, rtol=1e-10, atol=1e-12)


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
    res = loop_cluster_expand(
        tn,
        gloops=4,
        norm="1norm",
        cache=cache,
        tol=1e-12,
        require_fixed_point=False,
    )

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
    res = loop_cluster_expand(
        tn,
        gloops=6,
        norm="1norm",
        tol=1e-12,
        require_fixed_point=False,
    )

    assert np.isclose(res.estimate, exact)
    assert [(len(region), count) for region, count in res.region_counts] == [(6, 1)]


def test_1norm_strict_mode_rejects_a_rolling_bp_plateau():
    with pytest.raises(RuntimeError, match="requires converged BP messages"):
        loop_cluster_expand(
            _projected_scalar_ladder(),
            gloops=4,
            norm="1norm",
            tol=1e-12,
        )


def test_1norm_rejects_open_tensor_networks_cleanly():
    open_tn = qtn.TensorNetwork([qtn.Tensor(np.array([1.0, 2.0]), inds=("x",))])
    with pytest.raises(ValueError, match="closed graph"):
        loop_cluster_expand(open_tn, gloops=0, norm="1norm")


def test_scalar_cluster_cache_rejects_a_different_topology():
    cache = ScalarClusterCache()
    loop_cluster_expand(
        _scalar_two_site_tree(),
        gloops=0,
        norm="1norm",
        cache=cache,
        run_bp=False,
    )
    with pytest.raises(ValueError, match="different tensor-network topology"):
        loop_cluster_expand(
            _projected_scalar_ladder(),
            gloops=4,
            norm="1norm",
            cache=cache,
            run_bp=False,
            require_fixed_point=False,
        )


def test_simple_update_gauge_messages_round_trip_to_su_shape():
    core, gauges = _projected_scalar_ladder_su_core()

    bp = d1bp_from_simple_update_gauges(core, gauges)
    bp_gauges = simple_update_gauges_from_messages(bp)
    comparison = compare_simple_update_gauges(gauges, bp_gauges)

    assert comparison["num_bonds"] == len(gauges)
    assert comparison["max_rel_l2"] < 1e-12


def test_d1bp_messages_round_trip_losslessly_through_su_core_and_gauges():
    from pepsy.bp import one_norm_bp

    result = one_norm_bp(_positive_triangle(), method="d1bp", tol=1e-10)
    assert result.converged

    core, gauges = simple_update_core_and_gauges_from_messages(result.bp)
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)
    _assert_same_tensor_data(rebuilt, result.bp.tn)

    # SU-style symmetric initialization recovers the same D1BP gauge and
    # Bethe contraction, even though the directed message gauge is discarded.
    round_trip = run_d1bp_from_simple_update_gauges(
        core,
        gauges,
        run_opts={"tol": 1e-10},
    )
    assert round_trip.converged
    assert np.isclose(round_trip.contract(), result.contract())
    comparison = compare_simple_update_gauges(
        gauges,
        simple_update_gauges_from_messages(round_trip.bp),
    )
    assert comparison["max_rel_l2"] < 1e-7


def test_relay_bp_runs_from_real_quimb_su_and_can_return_to_su():
    original, core, gauges = _real_su_triangle()
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)
    # Quimb's simple update produces a conditioned SU representation rather
    # than merely moving diagonal factors tensor-by-tensor, but it preserves
    # the scalar network it represents.
    assert np.isclose(rebuilt.contract(), original.contract())

    relay = run_d1bp_from_simple_update_gauges(
        core,
        gauges,
        use_relay=True,
        run_opts={"tol": 1e-10},
        relay_opts={
            "num_relays": 2,
            "memory_first_leg": True,
            "gamma_range": (0.1, 0.2),
            "seed": 0,
        },
    )
    assert relay.converged
    assert relay.num_legs_run == 2

    # Real SU can yield nearly singular BP products. Smudging preserves the
    # represented TN exactly while producing a well-defined SU initializer.
    relay_core, relay_gauges = simple_update_core_and_gauges_from_messages(
        relay.bp,
        smudge=1e-10,
    )
    relay_rebuilt = relay_core.copy()
    relay_rebuilt.gauge_simple_insert(relay_gauges)
    _assert_same_tensor_data(relay_rebuilt, relay.bp.tn)

    from_su = run_d1bp_from_simple_update_gauges(
        relay_core,
        relay_gauges,
        run_opts={"tol": 1e-10},
    )
    assert from_su.converged
    assert np.isclose(from_su.contract(), relay.contract())


def test_exact_su_and_relay_d1bp_comparison_example():
    """The documented comparison tracks convergence and loopy-BP error."""
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "RelayBP"
        / "simple_update_relay_comparison.py"
    )
    comparison = runpy.run_path(str(example_path))["run_comparison"]()

    assert comparison["simple_update_representation_error"] < 1e-12
    runs = comparison["runs"]
    assert set(runs) == {
        "plain_d1bp",
        "su_initialized_d1bp",
        "su_initialized_relay_d1bp",
    }
    assert all(record["converged"] for record in runs.values())
    assert runs["su_initialized_relay_d1bp"]["num_legs"] == 3
    assert all(record["relative_error"] < 0.01 for record in runs.values())

    estimates = [record["estimate"] for record in runs.values()]
    np.testing.assert_allclose(estimates, estimates[0], rtol=1e-10, atol=0.0)


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


def test_gauge_bp_check_always_refreshes_the_returned_state():
    _, _, info = gauge_all_simple_with_bp_check(
        _projected_scalar_ladder(),
        max_iterations=3,
        su_tol=0.0,
        bp_tol=0.0,
        bp_check_every=2,
    )

    # Sweep 2 is the regular cadence; sweep 3 is the mandatory final refresh.
    assert [check["iteration"] for check in info["bp_checks"]] == [2, 3]
    assert info["bp_checks"][-1]["max_mdiff"] == info["bp_max_mdiff"]


def test_relay_simple_update_breaks_a_real_su_gauge_stall():
    tn = _projected_scalar_ladder()
    exact = tn.contract()

    plain = tn.copy()
    plain_gauges = {}
    plain_info = {}
    plain.gauge_all_simple_(
        gauges=plain_gauges,
        max_iterations=200,
        tol=1e-2,
        fuse_multibonds=False,
        info=plain_info,
    )
    assert plain_info["max_sdiff"] > 1e-2

    core, gauges, info = relay_gauge_all_simple(
        tn,
        max_iterations=200,
        tol=1e-2,
        num_relays=3,
        memory_first_leg=True,
        gamma_range=(0.8, 0.95),
        seed=0,
    )
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)

    assert info["converged"]
    assert info["max_sdiff"] < 1e-2
    assert info["num_legs_run"] == 3
    assert np.isclose(rebuilt.contract(), exact)


def test_parallel_relay_simple_update_preserves_the_tensor_network():
    tn = _projected_scalar_ladder()
    exact = tn.contract()

    core, gauges, info = relay_gauge_all_simple(
        tn,
        max_iterations=3,
        num_relays=1,
        damping=0.3,
        diis={"max_history": 3, "beta": 0.5},
        parallel=True,
        max_workers=2,
    )
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)

    assert info["parallel"] is True
    assert info["legs"][0]["diis_steps"] >= 2
    assert len(gauges) == len(tn.inner_inds())
    assert np.isclose(rebuilt.contract(), exact)


def test_relay_simple_update_validates_nonnegative_memory_controls():
    with pytest.raises(ValueError, match="0 <= min"):
        relay_gauge_all_simple(_projected_scalar_ladder(), gamma_range=(-0.1, 0.2))
    with pytest.raises(ValueError, match="fuse_multibonds=False"):
        relay_gauge_all_simple(
            _projected_scalar_ladder(),
            fuse_multibonds=True,
        )
    with pytest.raises(ValueError, match="damping"):
        relay_gauge_all_simple(_projected_scalar_ladder(), damping=1.0)
    with pytest.raises(TypeError, match="diis"):
        relay_gauge_all_simple(_projected_scalar_ladder(), diis="yes")


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


def test_run_d1bp_from_simple_update_gauges_accepts_a_warm_snapshot():
    core, gauges = _projected_scalar_ladder_su_core()
    first = run_d1bp_from_simple_update_gauges(
        core,
        gauges,
        run_opts={"max_iterations": 100, "tol": 1e-10},
    )
    second = run_d1bp_from_simple_update_gauges(
        core,
        gauges,
        run_opts={
            "max_iterations": 100,
            "tol": 1e-10,
            "init_messages": first.snapshot(),
        },
    )

    assert first.converged and second.converged
    assert np.isclose(first.contract(), second.contract())


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
