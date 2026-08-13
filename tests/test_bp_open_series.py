"""Tests for the explicit open-edge BP rho loop series."""

from collections import Counter
from itertools import combinations

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    OpenLoopEnumerationLimitError,
    OpenLoopBudgetError,
    OpenLoopMeasurementResult,
    OpenLoopObservableTerm,
    OpenLoopSeriesCache,
    OpenLoopSeriesDiagnosticCache,
    OpenLoopSeriesSweepResult,
    compute_local_expectation_open_loop_series,
    adaptive_open_loop_series,
    diagnose_open_rho_series,
    diagnose_open_loop_series,
    partial_trace_open_loop_series_expand,
    partial_trace_open_loop_series_sweep,
    rho_expand,
    two_norm_bp,
)
from pepsy.bp.series import (
    _discover_grid_corridor_paths,
    _grid_corridor_context,
    _open_term_family,
)


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


def test_rho_expand_dispatches_route_specific_cutoffs():
    state = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        seed=1909,
        dtype="float64",
    )
    where = ((0, 0), (1, 1))
    bp = two_norm_bp(state, max_iterations=100, tol=1e-10, diis=False)
    expected = partial_trace_open_loop_series_expand(
        state,
        where,
        edge_cutoff=2,
        messages=bp.messages,
        run_bp=False,
    )
    actual = rho_expand(
        state,
        where,
        cutoff=2,
        expansion="open",
        messages=bp.messages,
        run_bp=False,
    )
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(TypeError, match="edge_cutoff"):
        rho_expand(
            state,
            where,
            cutoff=2,
            expansion="open",
            edge_cutoff=2,
        )


def test_open_terms_stream_shortest_support_path_first():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1908,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    tags = [state.site_tag(coo) for coo in where]
    tids = frozenset(state._get_tids_from_tags(tags, "any"))
    excluded_edges = frozenset(state._select_tids(tids).inner_inds())
    iterator = OpenLoopSeriesCache().iter_terms_for(
        state,
        3,
        tids,
        excluded_edges=excluded_edges,
    )

    first = next(iterator)
    assert first.degree == 3
    assert _open_term_family(state, first) == "open_path"


def test_open_series_enumeration_limits_raise_before_partial_contraction():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1918,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    with pytest.raises(OpenLoopEnumerationLimitError, match="max_terms"):
        partial_trace_open_loop_series_expand(
            state,
            where,
            edge_cutoff=3,
            max_terms=0,
            max_iterations=200,
            tol=1e-10,
            diis=False,
        )
    with pytest.raises(
        OpenLoopEnumerationLimitError,
        match="max_enumeration_memory",
    ):
        partial_trace_open_loop_series_expand(
            state,
            where,
            edge_cutoff=3,
            max_enumeration_memory=1,
            max_iterations=200,
            tol=1e-10,
            diis=False,
        )


def test_edge_cutoff_is_the_explicit_name_for_dense_open_series():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1919,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    named = partial_trace_open_loop_series_expand(
        state,
        where,
        edge_cutoff=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    legacy = partial_trace_open_loop_series_expand(
        state,
        where,
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    np.testing.assert_allclose(named, legacy, rtol=1e-10, atol=1e-12)


def test_open_series_rejects_mixed_edge_and_cluster_cutoffs():
    state = qtn.PEPS.rand(1, 4, bond_dim=2, phys_dim=2, seed=1921)
    with pytest.raises(TypeError, match="edge_cutoff and cluster_size"):
        partial_trace_open_loop_series_expand(
            state,
            ((0, 0), (0, 3)),
            edge_cutoff=3,
            cluster_size=4,
            max_iterations=100,
        )


def test_corridor_mode_keeps_shortest_paths_and_local_loop_decorations():
    state = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        seed=1936,
        dtype="complex128",
    )
    info = {}
    partial_trace_open_loop_series_expand(
        state,
        ((0, 0), (2, 2)),
        corridor_width=1,
        max_path_candidates=4,
        loop_decoration_size=4,
        loop_radius=2,
        corridor_segment_length=2,
        max_loop_clusters_per_segment=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    corridor = info["open_rho_corridor"]
    assert corridor["path_count"] == 4
    assert corridor["shortest_path_length"] == 4
    assert corridor["search_backend"] == "rectangular_grid"
    assert corridor["corridor_edges"] < len(state.ind_map)
    assert info["open_rho_family_counts"]["open_path"] == 4
    assert info["open_rho_family_counts"]["closed_loop"]
    assert info["open_rho_family_counts"]["path_plus_loop"]


def test_corridor_mode_limits_geometry_before_contraction():
    state = qtn.PEPS.rand(3, 3, bond_dim=2, phys_dim=2, seed=1937)
    with pytest.raises(OpenLoopEnumerationLimitError, match="max_corridor_edges"):
        partial_trace_open_loop_series_expand(
            state,
            ((0, 0), (2, 2)),
            corridor_width=1,
            max_corridor_edges=1,
            max_iterations=200,
            tol=1e-10,
            diis=False,
        )


def test_loop_term_budget_is_separate_from_total_term_budget():
    state = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        seed=1938,
        dtype="complex128",
    )
    with pytest.raises(OpenLoopEnumerationLimitError, match="max_loop_terms"):
        partial_trace_open_loop_series_expand(
            state,
            ((0, 0), (0, 2)),
            edge_cutoff=6,
            max_terms=100,
            max_loop_terms=0,
            max_iterations=200,
            tol=1e-10,
            diis=False,
        )


def test_corridor_mode_can_use_compressed_boundary_contraction():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1938,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    gate = np.diag([1.0, 2.0, 3.0, 4.0])
    info = {}
    value = compute_local_expectation_open_loop_series(
        state,
        {where: gate},
        corridor_width=0,
        max_path_candidates=1,
        loop_decoration_size=4,
        corridor_max_bond=4,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    assert np.isfinite(value)
    assert info["open_scalar_corridor"]["path_count"] == 1


def test_open_measurement_diagnostic_selects_auto_route_and_reuses_terms():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1944,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    gate = np.diag([1.0, 2.0, 3.0, 4.0])
    diagnostic_cache = OpenLoopSeriesDiagnosticCache()
    diagnostic = diagnose_open_loop_series(
        state,
        OpenLoopObservableTerm(where, gate),
        edge_cutoff=3,
        mode="auto",
        auto_corridor_distance=1,
        max_path_candidates=1,
        loop_decoration_size=2,
        max_loop_clusters_per_segment=1,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        diagnostic_cache=diagnostic_cache,
    )

    support = diagnostic.for_support(where)
    assert diagnostic.routes == {where: "corridor"}
    assert support["terms"]
    assert support["term_costs"]
    assert diagnostic.total_flops_log10 is not None
    assert diagnostic.peak_memory_log2 is not None

    info = {}
    value = compute_local_expectation_open_loop_series(
        state,
        {where: gate},
        edge_cutoff=3,
        mode="auto",
        auto_corridor_distance=1,
        max_path_candidates=1,
        loop_decoration_size=2,
        max_loop_clusters_per_segment=1,
        diagnostic=diagnostic,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )
    assert np.isfinite(value)
    assert info["open_scalar_mode"] == "auto"
    assert tuple(info["open_scalar_requested_terms"]) == support["terms"]
    assert info["open_scalar_region_path_cache"]

    cached = diagnose_open_loop_series(
        state,
        {where: gate},
        edge_cutoff=3,
        mode="auto",
        auto_corridor_distance=1,
        max_path_candidates=1,
        loop_decoration_size=2,
        max_loop_clusters_per_segment=1,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        diagnostic_cache=diagnostic_cache,
    )
    assert cached.cache_hits == 1


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
    assert len(info["open_rho_region_path_cache"]) <= len(
        info["open_rho_terms_list"]
    )
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


def test_open_series_reuses_contraction_paths_for_shared_regions():
    state = qtn.PEPS.rand(
        3,
        2,
        bond_dim=2,
        phys_dim=2,
        cyclic=(True, True),
        seed=1922,
        dtype="complex128",
    )
    info = {}
    partial_trace_open_loop_series_expand(
        state,
        ((0, 0), (2, 1)),
        edge_cutoff=4,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )
    assert len(info["open_rho_region_path_cache"]) < len(
        info["open_rho_terms_list"]
    )


def test_pbc_corridor_keeps_parallel_period_two_bonds_as_distinct_paths():
    """A 3x2 torus is a multigraph, not a simple coordinate graph."""
    state = qtn.PEPS.rand(
        3,
        2,
        bond_dim=2,
        phys_dim=2,
        cyclic=(True, True),
        seed=1948,
    )
    context = _grid_corridor_context(state)
    neighbors = list(context["neighbors"]((0, 0)))
    seam_edges = {
        edge for neighbor, _, edge in neighbors if neighbor == (0, 1)
    }
    assert len(seam_edges) == 2

    paths, corridor_edges, diagnostics = _discover_grid_corridor_paths(
        state,
        ((0, 0), (2, 1)),
        corridor_width=0,
        max_path_candidates=20,
    )
    assert diagnostics["path_count"] == 4
    assert len({path.edges for path in paths}) == 4
    assert len(corridor_edges) == len(set(corridor_edges))
    assert all(len(path.edges) == 2 for path in paths)


def test_open_series_production_result_reports_budget_and_resources():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1949,
        dtype="complex128",
    )
    result = compute_local_expectation_open_loop_series(
        state,
        {((0, 0), (0, 3)): np.eye(4)},
        edge_cutoff=3,
        max_flops_log10=2.0,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        return_result=True,
        measure_resources=True,
    )
    assert isinstance(result, OpenLoopMeasurementResult)
    assert not result.complete
    assert result.omitted_terms
    assert result.resources["enabled"]
    assert result.info["open_scalar_complete"] is False
    with pytest.raises(OpenLoopBudgetError):
        compute_local_expectation_open_loop_series(
            state,
            {((0, 0), (0, 3)): np.eye(4)},
            edge_cutoff=3,
            max_flops_log10=2.0,
            max_iterations=200,
            tol=1e-10,
            diis=False,
            on_budget="raise",
        )


def test_rho_diagnostic_and_adaptive_corridor_ladder():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1950,
        dtype="complex128",
    )
    support = ((0, 0), (0, 3))
    diagnostic = diagnose_open_rho_series(
        state,
        (support,),
        edge_cutoff=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    record = diagnostic.for_support(support)
    assert record["observable_kind"] == "rho"
    assert record["output_shape"] == (4, 4)
    assert record["logical_output_shape"] == (2, 2, 2, 2)
    assert record["output_memory_bytes"] > 0

    rho_result = partial_trace_open_loop_series_expand(
        state,
        support,
        edge_cutoff=1,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        measure_resources=True,
        return_result=True,
    )
    assert isinstance(rho_result, OpenLoopMeasurementResult)
    assert rho_result.value.shape == (4, 4)
    assert rho_result.normalization is not None
    assert rho_result.resources["enabled"]

    adaptive = adaptive_open_loop_series(
        state,
        {support: np.eye(4)},
        corridor_widths=(0, 1),
        edge_cutoff=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        min_stable=1,
    )
    assert adaptive.values
    assert adaptive.settings[0]["corridor_width"] == 0
    assert adaptive.diagnostics[0] is not None
    assert adaptive.bp is not None


def test_open_series_public_controls_are_validated_and_cache_is_positional_safe():
    cache = OpenLoopSeriesDiagnosticCache({})
    assert cache.diagnostics_by_key == {}

    with pytest.raises(ValueError, match="positive integer"):
        diagnose_open_rho_series(
            None,
            (((0, 0),),),
            max_rho_identity_dimension=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        adaptive_open_loop_series(
            None,
            {},
            corridor_widths=(-1,),
        )
    with pytest.raises(ValueError, match="positive integer"):
        adaptive_open_loop_series(
            None,
            {},
            corridor_widths=(0,),
            min_stable=0,
        )


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


def test_open_rho_series_sweep_accepts_route_specific_edge_cutoffs():
    state = qtn.PEPS.rand(
        1,
        4,
        bond_dim=2,
        phys_dim=2,
        seed=1923,
        dtype="complex128",
    )
    result = partial_trace_open_loop_series_sweep(
        state,
        (((0, 0), (0, 3)),),
        edge_cutoffs=(0, 3),
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    assert result.get_rho(((0, 0), (0, 3)), 0) is not None
    assert result.get_rho(((0, 0), (0, 3)), 3) is not None


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
