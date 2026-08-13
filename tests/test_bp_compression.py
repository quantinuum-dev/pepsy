"""Tests for selected-bond BP/SU cluster compression."""

from itertools import combinations

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    BondClusterCompressionResult,
    BondLoopSeriesCompressionResult,
    CompressionBudgetError,
    OpenLoopSeriesCache,
    compress_bond_loop_series,
    compress_bond_cluster,
    cut_edge_loop_series_expand,
    gauge_all_simple,
    two_norm_bp,
)
from pepsy.bp.series import _iter_open_edge_loops


def _selected_bond(tn, where):
    left = next(iter(tn._get_tids_from_tags((tn.site_tag(where[0]),), "any")))
    right = next(iter(tn._get_tids_from_tags((tn.site_tag(where[1]),), "any")))
    return next(iter(qtn.bonds(tn.tensor_map[left], tn.tensor_map[right])))


def _brute_force_admissible_cut_terms(tn, *, cut_bond, endpoints):
    """Return every non-vacuum Q-edge subset allowed by a cut A--B bond."""
    candidate_edges = tuple(
        index
        for index, tids in tn.ind_map.items()
        if len(tids) == 2 and index != cut_bond
    )
    terms = set()
    for degree in range(1, len(candidate_edges) + 1):
        for selected in combinations(candidate_edges, degree):
            q_degrees = {}
            for index in selected:
                left, right = tn.ind_map[index]
                q_degrees[left] = q_degrees.get(left, 0) + 1
                q_degrees[right] = q_degrees.get(right, 0) + 1
            dangling = {
                tid for tid, q_degree in q_degrees.items() if q_degree == 1
            }
            if dangling <= set(endpoints):
                terms.add(frozenset(selected))
    return candidate_edges, terms


def test_selected_peps_bond_uses_bp_cluster_and_reduces_only_that_bond():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=41,
    )
    where = ((0, 0), (1, 0))
    selected = _selected_bond(peps, where)
    original_dims = {index: peps.ind_size(index) for index in peps.inner_inds()}
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = compress_bond_cluster(
        peps,
        where=where,
        boundary_messages=bp.messages,
        max_distance=0,
        max_bond=1,
        steps=4,
        tol=0.0,
        als_opts={"solver_maxiter": 8},
        seed=17,
    )

    assert isinstance(result, BondClusterCompressionResult)
    assert isinstance(result.compressed, qtn.PEPS)
    assert result.environment_projection == {
        "hermitian_project": True,
        "psd_project": False,
        "psd_floor": 0.0,
    }
    assert result.normalization["method"] == "quimb.decomp.array_split"
    assert result.normalization["absorb"] == "both"
    assert result.normalization["preserve_norm"] is True
    assert result.normalization["scalar_factor"] > 0.0
    np.testing.assert_allclose(
        result.normalization["norm_after_maps"],
        result.normalization["norm_before"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(result.compressed.norm(), peps.norm(), rtol=1e-12)
    assert result.normalization["product_relative_error"] < 1e-10
    assert result.errors[1] <= result.errors[0] + 1e-10
    assert result.B_reduce.shape == (2, 2, 2, 2)
    matrix = result.B_reduce.transpose(2, 3, 0, 1).reshape(4, 4)
    np.testing.assert_allclose(matrix, matrix.conj().T)
    assert set(result.compressed.outer_inds()) == set(peps.outer_inds())
    assert result.compressed.ind_size(next(
        index
        for index in result.compressed.inner_inds()
        if result.compressed.ind_size(index) == 1
        and index not in original_dims
    )) == 1
    assert all(
        result.compressed.ind_size(index) == dimension
        for index, dimension in original_dims.items()
        if index != selected and index in result.compressed.ind_map
    )
    left, right = result.bond_maps[selected]
    assert left.shape == (2, 1)
    assert right.shape == (1, 2)
    np.testing.assert_allclose(left.conj().T @ left, right @ right.conj().T)
    assert result.boundary_inds


def test_cut_edge_loop_series_reaches_full_finite_environment():
    """Adding all admissible Q terms reproduces the finite cut environment."""
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=42,
    )
    where = ((0, 0), (1, 0))
    bp = two_norm_bp(peps, max_iterations=100, tol=1e-12, diis=False)

    vacuum = cut_edge_loop_series_expand(
        peps,
        where=where,
        edge_cutoff=0,
        boundary_messages=bp.messages,
        run_bp=False,
        optimize="greedy",
    )
    complete = cut_edge_loop_series_expand(
        peps,
        where=where,
        edge_cutoff=None,
        boundary_messages=bp.messages,
        run_bp=False,
        optimize="greedy",
    )
    reference = compress_bond_cluster(
        peps,
        where=where,
        boundary_messages=bp.messages,
        max_distance=2,
        max_bond=1,
        b_reduce=False,
        steps=1,
        contract_optimize="greedy",
    )

    assert vacuum.term_count_by_degree == {0: 1}
    assert complete.complete
    assert len(complete.terms) > len(vacuum.terms)
    np.testing.assert_allclose(complete.environment, reference.B_reduce, atol=1e-10)


def test_cut_edge_loop_series_reuses_open_loop_geometry_cache(monkeypatch):
    """A populated topology cache avoids rediscovering cut Q configurations."""
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=420,
    )
    where = ((0, 0), (1, 0))
    bp = two_norm_bp(peps, max_iterations=100, tol=1e-12, diis=False)
    cache = OpenLoopSeriesCache()
    first = cut_edge_loop_series_expand(
        peps,
        where=where,
        edge_cutoff=None,
        boundary_messages=bp.messages,
        run_bp=False,
        optimize="greedy",
        cache=cache,
    )

    def fail_if_enumerated(*args, **kwargs):
        raise AssertionError("cached cut-edge geometry was enumerated again")

    monkeypatch.setattr("pepsy.bp.series._iter_open_edge_loops", fail_if_enumerated)
    second = cut_edge_loop_series_expand(
        peps,
        where=where,
        edge_cutoff=None,
        boundary_messages=bp.messages,
        run_bp=False,
        optimize="greedy",
        cache=cache,
    )

    assert first.terms == second.terms
    np.testing.assert_allclose(first.environment, second.environment)


def test_cut_edge_loop_series_enumerates_all_and_only_admissible_q_subsets():
    """The cut series retains every Q subset whose only dangling ends are A/B."""
    peps = qtn.PEPS.rand(
        3,
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=43,
    )
    where = ((0, 0), (1, 0))
    left = next(iter(peps._get_tids_from_tags((peps.site_tag(where[0]),), "any")))
    right = next(iter(peps._get_tids_from_tags((peps.site_tag(where[1]),), "any")))
    cut_bond = _selected_bond(peps, where)
    candidate_edges, expected = _brute_force_admissible_cut_terms(
        peps,
        cut_bond=cut_bond,
        endpoints=(left, right),
    )

    actual_terms = tuple(
        _iter_open_edge_loops(
            peps,
            len(candidate_edges),
            allowed_tids=(left, right),
            excluded_edges=(cut_bond,),
        )
    )
    actual = {frozenset(term.edges) for term in actual_terms}

    assert len(actual_terms) == len(actual)
    assert actual == expected
    for term in actual_terms:
        assert term.tids == frozenset(
            tid for index in term.edges for tid in peps.ind_map[index]
        )
    assert any(len(term) == 3 for term in actual)  # open A--B path
    assert any(
        all(
            sum(tid in peps.ind_map[index] for index in term) != 1
            for tid in set().union(*(peps.ind_map[index] for index in term))
        )
        for term in actual
    )  # closed/generalized loop

    for cutoff in (0, 3, 6, len(candidate_edges)):
        at_cutoff = {
            frozenset(term.edges)
            for term in _iter_open_edge_loops(
                peps,
                cutoff,
                allowed_tids=(left, right),
                excluded_edges=(cut_bond,),
            )
        }
        assert at_cutoff == {term for term in expected if len(term) <= cutoff}


def test_cut_edge_loop_series_compression_uses_explicit_degree_cutoff():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=44,
    )
    result = compress_bond_loop_series(
        peps,
        where=((0, 0), (1, 0)),
        max_bond=1,
        edge_cutoff=None,
        bp_opts={"max_iterations": 100, "tol": 1e-12, "diis": False},
        steps=2,
        tol=0.0,
        als_opts={"solver_maxiter": 8},
    )

    assert isinstance(result, BondLoopSeriesCompressionResult)
    assert result.environment_projection == {
        "hermitian_project": True,
        "psd_project": False,
        "psd_floor": 0.0,
    }
    assert result.normalization["method"] == "quimb.decomp.array_split"
    assert result.normalization["absorb"] == "both"
    assert result.normalization["preserve_norm"] is True
    assert result.normalization["scalar_factor"] > 0.0
    np.testing.assert_allclose(
        result.normalization["norm_after_maps"],
        result.normalization["norm_before"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(result.compressed.norm(), peps.norm(), rtol=1e-12)
    assert result.normalization["product_relative_error"] < 1e-10
    assert result.complete
    assert result.term_count >= 2
    assert result.B_reduce.shape == (2, 2, 2, 2)
    assert result.errors[1] <= result.errors[0] + 1e-10
    assert any(
        result.compressed.ind_size(index) == 1
        for index in result.compressed.inner_inds()
    )


def test_compression_environment_projection_controls_and_legacy_alias():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=47,
    )
    where = ((0, 0), (1, 0))
    bp = two_norm_bp(peps, max_iterations=20, tol=0.0, diis=False)

    result = compress_bond_cluster(
        peps,
        where=where,
        boundary_messages=bp.messages,
        max_distance=0,
        max_bond=1,
        psd_project=True,
        steps=1,
    )
    assert result.environment_projection == {
        "hermitian_project": True,
        "psd_project": True,
        "psd_floor": 0.0,
    }
    matrix = result.B_reduce.transpose(2, 3, 0, 1).reshape(4, 4)
    np.testing.assert_allclose(matrix, matrix.conj().T)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12

    legacy = compress_bond_cluster(
        peps,
        where=where,
        boundary_messages=bp.messages,
        max_distance=0,
        max_bond=1,
        b_reduce=False,
        steps=1,
    )
    assert legacy.environment_projection["psd_project"] is False

    with pytest.raises(ValueError, match="requires hermitian_project=True"):
        compress_bond_cluster(
            peps,
            where=where,
            max_bond=1,
            hermitian_project=False,
            psd_project=True,
        )


def test_cut_edge_loop_series_exposes_costs_and_enforces_budget():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=45,
    )
    where = ((0, 0), (1, 0))
    bp = two_norm_bp(peps, max_iterations=20, tol=0.0, diis=False)

    result = cut_edge_loop_series_expand(
        peps,
        where=where,
        edge_cutoff=0,
        boundary_messages=bp.messages,
        run_bp=False,
        optimize="greedy",
        cost_check=True,
        max_flops_log10=100.0,
        max_peak_memory_log2=100.0,
    )

    assert result.contraction_costs
    assert result.contraction_cost is not None
    assert result.bp_info["cost_check"] is True
    assert result.cost_limits == {
        "max_flops_log10": 100.0,
        "max_peak_memory_log2": 100.0,
    }

    with pytest.raises(CompressionBudgetError, match="cut-edge loop-series"):
        cut_edge_loop_series_expand(
            peps,
            where=where,
            edge_cutoff=0,
            boundary_messages=bp.messages,
            run_bp=False,
            optimize="greedy",
            max_flops_log10=0.0,
        )


def test_cut_edge_loop_series_compression_forwards_cost_policy():
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=46,
    )
    result = compress_bond_loop_series(
        peps,
        where=((0, 0), (1, 0)),
        max_bond=1,
        edge_cutoff=0,
        bp_opts={"max_iterations": 100, "tol": 1e-12, "diis": False},
        cost_check=True,
        max_flops_log10=100.0,
        max_peak_memory_log2=100.0,
        steps=1,
        als_opts={"solver_maxiter": 4},
    )

    assert result.contraction_cost is not None
    assert result.cost_limits == {
        "max_flops_log10": 100.0,
        "max_peak_memory_log2": 100.0,
    }


def test_selected_pepo_bond_preserves_separate_operator_legs():
    pepo = qtn.PEPO.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=43,
    )
    where = ((0, 0), (1, 0))
    selected = _selected_bond(pepo, where)
    bp = two_norm_bp(pepo, max_iterations=2, tol=0.0, diis=False)

    result = compress_bond_cluster(
        pepo,
        where=where,
        boundary_messages=bp.messages,
        max_distance=0,
        max_bond=1,
        steps=3,
        tol=0.0,
        als_opts={"solver_maxiter": 8},
        seed=19,
    )

    out = result.compressed
    assert isinstance(out, qtn.PEPO)
    assert selected not in out.ind_map
    assert set(out.outer_inds()) == set(pepo.outer_inds())
    assert any(out.ind_size(index) == 1 for index in out.inner_inds())
    for site in ((0, 0), (1, 0)):
        tid = next(iter(out._get_tids_from_tags((out.site_tag(site),), "any")))
        tensor = out.tensor_map[tid]
        assert out.lower_ind(site) in tensor.inds
        assert out.upper_ind(site) in tensor.inds


@pytest.mark.parametrize("chi", [2, 3])
def test_d4_peps_selected_bond_compression_for_chi_two_and_three(chi):
    """Check the requested D=4 local path for both retained dimensions."""
    peps = qtn.PEPS.rand(
        4,
        4,
        bond_dim=4,
        phys_dim=2,
        dtype="float64",
        seed=20260811,
    )
    where = ((1, 1), (1, 2))
    selected = _selected_bond(peps, where)
    original_dims = {index: peps.ind_size(index) for index in peps.inner_inds()}
    bp = two_norm_bp(peps, max_iterations=2, tol=0.0, diis=False)

    result = compress_bond_cluster(
        peps,
        where=where,
        boundary_messages=bp.messages,
        max_distance=1,
        max_bond=chi,
        steps=2,
        tol=0.0,
        als_opts={"solver_maxiter": 8},
        seed=chi,
    )

    new_bonds = tuple(index for index in result.compressed.inner_inds() if index not in original_dims)
    assert len(new_bonds) == 1
    assert result.compressed.ind_size(new_bonds[0]) == chi
    assert result.B_reduce.shape == (4, 4, 4, 4)
    assert result.errors[1] <= result.errors[0] + 1e-9
    assert all(
        result.compressed.ind_size(index) == dimension
        for index, dimension in original_dims.items()
        if index != selected
    )
    left, right = result.bond_maps[selected]
    assert left.shape == (4, chi)
    assert right.shape == (chi, 4)


def test_selected_bond_accepts_su_vectors_as_boundary_closures():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, dtype="float64", seed=47)
    core, gauges, _ = gauge_all_simple(
        peps,
        progbar=False,
    )
    where = ((0, 0), (1, 0))
    result = compress_bond_cluster(
        core,
        where=where,
        gauges=gauges,
        max_distance=0,
        max_bond=1,
        steps=2,
        tol=0.0,
        seed=23,
    )
    assert isinstance(result.compressed, qtn.PEPS)
    assert result.boundary_inds
    assert result.clipped_eigenvalues >= 0


def test_selected_bond_can_run_fresh_bp_for_an_open_cluster():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=53)
    result = compress_bond_cluster(
        peps,
        where=((0, 0), (1, 0)),
        max_distance=0,
        max_bond=1,
        steps=1,
        bp_opts={"max_iterations": 1, "tol": 0.0, "diis": False},
    )
    assert result.bp_info["source"] == "fresh_bp"
    assert result.boundary_inds


def test_selected_bond_can_require_explicit_closures_and_preflight_cost():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=54)
    with pytest.raises(ValueError, match="closure"):
        compress_bond_cluster(
            peps,
            where=((0, 0), (1, 0)),
            max_distance=0,
            max_bond=1,
            steps=1,
            run_bp=False,
        )

    bp = two_norm_bp(peps, max_iterations=1, tol=0.0, diis=False)
    result = compress_bond_cluster(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_distance=0,
        max_bond=1,
        steps=1,
        cost_check=True,
    )
    assert result.contraction_cost["flops_log10"] >= 0.0
    assert result.contraction_cost["peak_memory_log2"] >= 0.0
    with pytest.raises(CompressionBudgetError):
        compress_bond_cluster(
            peps,
            where=((0, 0), (1, 0)),
            boundary_messages=bp.messages,
            max_distance=0,
            max_bond=1,
            steps=1,
            max_flops_log10=-1.0,
        )


def test_selected_su_core_mode_returns_the_physical_network():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, dtype="float64", seed=55)
    core, gauges, _ = gauge_all_simple(peps, max_iterations=2, tol=0.0)
    result = compress_bond_cluster(
        core,
        gauges=gauges,
        where=((0, 0), (1, 0)),
        max_bond=2,
        steps=1,
    )
    expected = core.copy()
    expected.gauge_simple_insert(gauges)
    assert np.allclose(result.compressed.to_dense(), expected.to_dense())
    assert result.bp_info is None


def test_selected_bond_validates_als_protected_options():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=59)
    bp = two_norm_bp(peps, max_iterations=1, tol=0.0, diis=False)
    with pytest.raises(TypeError, match="cannot override"):
        compress_bond_cluster(
            peps,
            where=((0, 0), (1, 0)),
            boundary_messages=bp.messages,
            max_bond=1,
            als_opts={"tags": "other"},
        )


def test_selected_bond_preserves_torch_backend_arrays():
    torch = pytest.importorskip("torch")
    peps = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        phys_dim=2,
        dtype="float64",
        seed=61,
    )
    for tensor in peps.tensor_map.values():
        tensor.modify(data=torch.as_tensor(tensor.data))

    bp = two_norm_bp(peps, max_iterations=1, tol=0.0, diis=False)
    result = compress_bond_cluster(
        peps,
        where=((0, 0), (1, 0)),
        boundary_messages=bp.messages,
        max_bond=1,
        steps=2,
        tol=0.0,
    )

    assert isinstance(result.B_reduce, torch.Tensor)
    left, right = result.bond_maps[result.bond_ind]
    assert isinstance(left, torch.Tensor)
    assert isinstance(right, torch.Tensor)
    assert all(isinstance(tensor.data, torch.Tensor)
               for tensor in result.compressed.tensor_map.values())
