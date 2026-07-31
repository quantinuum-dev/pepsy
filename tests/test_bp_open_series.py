"""Tests for the explicit open-edge BP rho loop series."""

from collections import Counter
from itertools import combinations

import numpy as np
import quimb.tensor as qtn

from pepsy.bp import (
    OpenLoopSeriesCache,
    OpenLoopSeriesSweepResult,
    compute_local_expectation_open_loop_series,
    partial_trace_open_loop_series_expand,
    partial_trace_open_loop_series_sweep,
    two_norm_bp,
)
from pepsy.bp.series import _open_term_family


def _edge_degrees(tn, edges):
    degrees = {}
    for index in edges:
        left, right = tn.ind_map[index]
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1
    return degrees


def _notebook_terms(tn, max_degree, allowed_tids, excluded_edges):
    """Mirror ``quf.combine_elements`` without importing the notebook code."""
    edges = tuple(
        index
        for index, tids in tn.ind_map.items()
        if len(tids) == 2 and index not in excluded_edges
    )
    expected = set()
    for degree in range(1, max_degree + 1):
        for selected in combinations(edges, degree):
            degrees = _edge_degrees(tn, selected)
            dangling = {
                tid for tid, value in degrees.items() if value == 1
            }
            if not dangling or dangling <= allowed_tids:
                expected.add(frozenset(selected))
    return expected


def test_open_rho_series_keeps_the_long_range_path_and_is_exact_on_a_tree():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1908,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    exact = state.partial_trace(
        where,
        max_bond=64,
        optimize="auto-hq",
        flatten=True,
        normalized=True,
    )
    info = {}
    rho = partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    terms = info["open_rho_terms_list"]
    assert len(terms) == 1
    assert terms[0].degree == 3
    assert np.max(np.abs(rho - exact)) < 1e-10
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)


def test_open_scalar_series_inserts_a_two_site_gate_and_normalizes_after_sum():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1930,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    gate = np.diag([1.0, 2.0, 3.0, 4.0])
    exact = state.compute_local_expectation_exact(
        {where: gate}, normalized=True, optimize="auto-hq"
    )
    info = {}
    value = compute_local_expectation_open_loop_series(
        state,
        {where: gate},
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(info["open_scalar_denominator"], 1.0, atol=1e-12)
    np.testing.assert_allclose(info["open_scalar_numerator"], value, atol=1e-12)


def test_open_scalar_series_reports_and_applies_contraction_cost_limits():
    """Cost limits screen terms using Cotengra tree diagnostics."""
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1932,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    gate = np.diag([1.0, 2.0, 3.0, 4.0])
    info = {}
    value = compute_local_expectation_open_loop_series(
        state,
        {where: gate},
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
        max_flops_log10=2.0,
        max_peak_memory_log2=30.0,
    )

    assert info["open_scalar_cost_limits"] == {
        "max_flops_log10": 2.0,
        "max_peak_memory_log2": 30.0,
    }
    assert info["open_scalar_edge_term_costs"] == info[
        "open_scalar_term_costs"
    ]
    assert info["open_scalar_edge_skipped_terms"] == info[
        "open_scalar_skipped_terms"
    ]
    assert not info["open_scalar_cluster_region_costs"]
    assert not info["open_scalar_terms"]
    assert info["open_scalar_skipped_terms"]
    cost = next(iter(info["open_scalar_skipped_terms"].values()))["norm"]
    assert set(cost) == {"flops_log10", "peak_memory_log2"}
    assert cost["flops_log10"] > 2.0
    assert np.isfinite(value)


def test_open_rho_series_allows_only_rho_site_dangling_vertices():
    state = qtn.PEPS.rand(
        2,
        3,
        bond_dim=2,
        phys_dim=2,
        seed=1909,
        dtype="complex128",
    )
    where = ((0, 0), (0, 2))
    tags = [state.site_tag(coo) for coo in where]
    allowed_tids = frozenset(state._get_tids_from_tags(tags, "any"))
    excluded_edges = frozenset(state._select_tids(allowed_tids).inner_inds())

    terms = OpenLoopSeriesCache().terms_for(
        state,
        4,
        allowed_tids,
        excluded_edges=excluded_edges,
    )
    assert any(
        {tid for tid, degree in _edge_degrees(state, term.edges).items() if degree == 1}
        == allowed_tids.intersection(_edge_degrees(state, term.edges))
        and len(term.edges) == 2
        for term in terms
    )
    assert any(
        not any(degree == 1 for degree in _edge_degrees(state, term.edges).values())
        and len(term.edges) == 4
        for term in terms
    )
    for term in terms:
        dangling = {
            tid for tid, degree in _edge_degrees(state, term.edges).items() if degree == 1
        }
        assert dangling <= allowed_tids


def test_open_rho_terms_match_notebook_filter_and_classify_disconnected_loops():
    state = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        seed=1913,
        dtype="complex128",
    )
    where = ((0, 0), (0, 2))
    tags = [state.site_tag(coo) for coo in where]
    allowed_tids = frozenset(state._get_tids_from_tags(tags, "any"))
    excluded_edges = frozenset(state._select_tids(allowed_tids).inner_inds())
    terms = OpenLoopSeriesCache().terms_for(
        state,
        6,
        allowed_tids,
        excluded_edges=excluded_edges,
    )

    assert {frozenset(term.edges) for term in terms} == _notebook_terms(
        state,
        6,
        allowed_tids,
        excluded_edges,
    )
    families = Counter(_open_term_family(state, term) for term in terms)
    assert families["open_path"]
    assert families["closed_loop"]
    assert families["path_plus_loop"]


def test_four_by_four_long_range_cutoffs_have_expected_geometry():
    """The first 4x4 terms are paths, plaquettes, and attached loops."""
    state = qtn.PEPS.rand(
        4,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1915,
        dtype="complex128",
    )
    where = ((1, 1), (3, 3))
    allowed_tids = frozenset(
        state._get_tids_from_tags([state.site_tag(coo) for coo in where], "any")
    )
    excluded_edges = frozenset(state._select_tids(allowed_tids).inner_inds())
    cache = OpenLoopSeriesCache()

    geometries = {}
    for cutoff in (4, 5, 6):
        terms = cache.terms_for(
            state,
            cutoff,
            allowed_tids,
            excluded_edges=excluded_edges,
        )
        geometries[cutoff] = Counter(
            (len(term.edges), _open_term_family(state, term))
            for term in terms
        )

    assert geometries[4] == Counter(
        {(4, "open_path"): 6, (4, "closed_loop"): 9}
    )
    assert geometries[5] == geometries[4] + Counter(
        {(5, "path_plus_loop"): 6}
    )
    assert geometries[6] == geometries[5] + Counter(
        {
            (6, "open_path"): 14,
            (6, "closed_loop"): 12,
            (6, "path_plus_loop"): 14,
        }
    )


def test_open_rho_series_reports_term_families_and_reuses_one_bp_run():
    state = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        seed=1914,
        dtype="complex128",
    )
    where = ((0, 0), (0, 2))
    bp = two_norm_bp(
        state,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    info = {}
    cache = OpenLoopSeriesCache()
    rho = partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=4,
        messages=bp.messages,
        run_bp=False,
        cache=cache,
        info=info,
    )

    assert info["open_rho_family_counts"]["open_path"]
    assert info["open_rho_family_counts"]["closed_loop"]
    assert set(info["open_rho_term_families"].values()) == {
        "open_path",
        "closed_loop",
    }
    assert info["open_rho_edge_term_costs"]
    assert not info["open_rho_cluster_region_costs"]
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)

    other_info = {}
    other_rho = partial_trace_open_loop_series_expand(
        state,
        ((0, 0), (2, 2)),
        gloops=4,
        messages=bp.messages,
        run_bp=False,
        cache=cache,
        info=other_info,
    )
    assert other_info["open_rho_terms_list"]
    np.testing.assert_allclose(np.trace(other_rho), 1.0, atol=1e-12)
    assert len(cache.terms_by_key) == 2


def test_open_rho_series_reuses_one_d2bp_message_set():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1910,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    bp = two_norm_bp(
        state,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    fresh = partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    reused = partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=3,
        messages=bp.messages,
        run_bp=False,
    )
    np.testing.assert_allclose(reused, fresh, rtol=1e-10, atol=1e-12)


def test_open_rho_series_incrementally_reuses_contracted_terms():
    state = qtn.PEPS.rand(
        2,
        3,
        bond_dim=2,
        phys_dim=2,
        seed=1911,
        dtype="complex128",
    )
    where = ((0, 0), (0, 2))
    bp = two_norm_bp(
        state,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    cache = OpenLoopSeriesCache()
    info = {}
    partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=2,
        messages=bp.messages,
        run_bp=False,
        cache=cache,
        info=info,
    )
    first_terms = dict(info["open_rho_terms"])
    first_base = info["open_rho_base_terms"]

    partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=4,
        messages=bp.messages,
        run_bp=False,
        cache=cache,
        info=info,
    )

    assert len(info["open_rho_terms"]) > len(first_terms)
    assert info["open_rho_base_terms"] is first_base
    for key, rho_term in first_terms.items():
        assert info["open_rho_terms"][key] is rho_term


def test_open_rho_series_sweep_reuses_bp_across_supports_and_cutoffs():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1916,
        dtype="complex128",
    )
    supports = (((0, 0), (0, 3)), ((0, 0), (0, 1), (0, 3)))
    result = partial_trace_open_loop_series_sweep(
        state,
        supports,
        (0, 2, 3),
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    assert isinstance(result, OpenLoopSeriesSweepResult)
    assert result.bp_converged
    assert result.bp_iterations is not None
    for support in supports:
        assert tuple(support) in result.rhos
        for cutoff in (0, 2, 3):
            rho = result.get_rho(support, cutoff)
            np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
            assert result.diagnostics[tuple(support)][cutoff]["term_count"] >= 0


def test_open_rho_series_is_exact_for_a_tree_with_a_multi_site_support():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1917,
        dtype="complex128",
    )
    where = ((0, 0), (0, 1), (0, 3))
    exact = state.partial_trace(
        where,
        max_bond=64,
        optimize="auto-hq",
        flatten=True,
        normalized=True,
    )
    info = {}
    rho = partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=2,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    np.testing.assert_allclose(rho, exact, rtol=1e-10, atol=1e-12)
    assert info["open_rho_excluded_edges"]
