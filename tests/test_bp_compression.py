"""Tests for selected-bond BP/SU cluster compression."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    BondClusterCompressionResult,
    BondLoopSeriesCompressionResult,
    CompressionBudgetError,
    compress_bond_loop_series,
    compress_bond_cluster,
    cut_edge_loop_series_expand,
    gauge_all_simple,
    two_norm_bp,
)


def _selected_bond(tn, where):
    left = next(iter(tn._get_tids_from_tags((tn.site_tag(where[0]),), "any")))
    right = next(iter(tn._get_tids_from_tags((tn.site_tag(where[1]),), "any")))
    return next(iter(qtn.bonds(tn.tensor_map[left], tn.tensor_map[right])))


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
    assert result.errors[1] <= result.errors[0] + 1e-10
    assert result.B_reduce.shape == (2, 2, 2, 2)
    matrix = result.B_reduce.transpose(2, 3, 0, 1).reshape(4, 4)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12
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
    assert result.complete
    assert result.term_count >= 2
    assert result.B_reduce.shape == (2, 2, 2, 2)
    assert result.errors[1] <= result.errors[0] + 1e-10
    assert any(
        result.compressed.ind_size(index) == 1
        for index in result.compressed.inner_inds()
    )


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
