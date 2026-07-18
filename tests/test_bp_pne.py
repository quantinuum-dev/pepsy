"""Tests for partitioned network expansions (PNE)."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    PNEExpansionResult,
    PNEExpansionTerm,
    RecursivePNEExpansionResult,
    WeightPassingResult,
    loop_expand,
    pne_projector_diagnostics,
    pne_projectors,
    partitioned_expand,
    recursive_partitioned_expand,
    select_pne_partitions,
    weight_pass,
)


def _ring4(seed=11):
    rng = np.random.default_rng(seed)
    return qtn.TensorNetwork(
        [
            qtn.Tensor(
                rng.random((2, 2)) + 0.2,
                inds=(f"e{i}", f"e{(i - 1) % 4}"),
            )
            for i in range(4)
        ]
    )


def test_pne_exports_and_selector():
    from pepsy import bp

    assert {
        "PNEExpansionResult",
        "PNEExpansionTerm",
        "PNEPartitionScore",
        "PNEPartitionSelection",
        "RecursivePNEExpansionResult",
        "pne_projector_diagnostics",
        "pne_projectors",
        "partitioned_expand",
        "recursive_partitioned_expand",
        "select_pne_partitions",
    } <= set(bp.__all__)

    tn = _ring4()
    result = loop_expand(
        tn,
        expansion="pne",
        norm="1norm",
        partition_inds=("e0",),
        include_residue=True,
        max_iterations=500,
        tol=1e-10,
    )
    assert isinstance(result, PNEExpansionResult)
    assert result.expansion == "pne"
    assert result.cutoff_kind == "selected-partition-indices"


@pytest.mark.parametrize("form", ["linear", "combinatorial"])
def test_d1_pne_with_residue_is_exact(form):
    tn = _ring4()
    exact = tn.contract(optimize="auto-hq")
    result = partitioned_expand(
        tn,
        partition_inds=("e0", "e2"),
        norm="1norm",
        form=form,
        include_residue=True,
        max_iterations=500,
        tol=1e-10,
    )

    assert np.allclose(result.estimate, exact, rtol=1e-10, atol=1e-10)
    assert all(isinstance(term, PNEExpansionTerm) for term in result.terms)


def test_d2_pne_with_residue_is_exact():
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, phys_dim=2, seed=12)
    exact = (peps.H & peps).contract(optimize="auto-hq")
    partition_ind = next(iter(peps.inner_inds()))
    result = partitioned_expand(
        peps,
        partition_inds=(partition_ind,),
        norm="2norm",
        form="linear",
        include_residue=True,
        max_iterations=500,
        tol=1e-10,
    )

    assert np.allclose(result.estimate, exact, rtol=1e-10, atol=1e-10)


def test_pne_can_use_explicit_projector_without_fixed_point():
    tn = _ring4(seed=13)
    # P = I makes the one-term linear approximation equal to the exact
    # contraction, while run_bp=False demonstrates that convergence is not a
    # prerequisite when the dominant projector is supplied directly.
    result = partitioned_expand(
        tn,
        partition_inds=("e0",),
        norm="1norm",
        projectors={"e0": np.eye(2)},
        run_bp=False,
        include_residue=False,
    )

    assert result.bp_converged is None
    assert np.allclose(result.estimate, tn.contract(), rtol=1e-10, atol=1e-10)


def test_projector_diagnostics_and_partition_selection():
    tn = _ring4(seed=15)
    selection = select_pne_partitions(
        tn,
        norm="1norm",
        max_partitions=2,
        partition_opts={"max_iterations": 500, "tol": 1e-10},
    )
    assert len(selection.indices) == 2
    assert len(selection.scores) == 4
    projectors = pne_projectors(selection.scores[0].result)
    diagnostics = pne_projector_diagnostics(projectors)
    assert set(diagnostics) == set(tn.inner_inds())
    assert diagnostics[selection.scores[0].index]["rank"] == 1


def test_open_d1_pne_returns_an_open_tensor():
    rng = np.random.default_rng(16)
    tn = qtn.TensorNetwork(
        [
            qtn.Tensor(rng.random((2, 2)) + 0.2, inds=("left", "x")),
            qtn.Tensor(rng.random((2, 2)) + 0.2, inds=("x", "right")),
        ]
    )
    result = partitioned_expand(
        tn,
        partition_inds=("x",),
        norm="1norm",
        allow_open=True,
        open_inds=("left", "right"),
        projectors={"x": np.eye(2)},
        run_bp=False,
        include_residue=True,
    )
    exact = tn.contract(output_inds=("left", "right"))
    assert result.estimate.inds == exact.inds
    assert np.allclose(result.estimate.data, exact.data)


def test_open_d2_pne_returns_a_density_tensor():
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, phys_dim=2, seed=17)
    partition_ind = next(iter(peps.inner_inds()))
    result = partitioned_expand(
        peps,
        partition_inds=(partition_ind,),
        norm="2norm",
        open_inds=("k0,0",),
        include_residue=True,
        max_iterations=500,
        tol=1e-10,
    )
    assert result.estimate.inds == ("k0,0", "__pne_bra_'k0,0'")
    assert result.estimate.shape == (2, 2)


def test_factorized_multi_index_partition_residue_is_exact():
    tn = _ring4(seed=14)
    exact = tn.contract(optimize="auto-hq")
    result = partitioned_expand(
        tn,
        partitions=(("e0", "e1"), ("e2",)),
        norm="1norm",
        form="combinatorial",
        include_residue=True,
        max_iterations=500,
        tol=1e-10,
    )

    assert np.allclose(result.estimate, exact, rtol=1e-10, atol=1e-10)


def test_weight_passing_returns_higher_rank_projectors():
    tn = _ring4(seed=20)
    exact = tn.contract(optimize="auto-hq")
    result = weight_pass(tn, max_iterations=50, tol=1e-9)

    assert isinstance(result, WeightPassingResult)
    projectors = result.projectors(rank=2, indices=("e0", "e2"))
    assert all(
        np.linalg.matrix_rank(projector) == 2
        for projector in projectors.values()
    )
    expanded = partitioned_expand(
        result.network,
        norm="1norm",
        partition_inds=("e0", "e2"),
        projectors=result.projectors(rank=2),
        run_bp=False,
        include_residue=True,
    )
    assert np.allclose(result.network.contract(), exact, rtol=1e-10, atol=1e-10)
    assert np.allclose(expanded.estimate, exact, rtol=1e-10, atol=1e-10)


def test_multi_index_linear_form_is_rejected_explicitly():
    with pytest.raises(ValueError, match="multi-index.*combinatorial"):
        partitioned_expand(
            _ring4(),
            partitions=(("e0", "e1"),),
            norm="1norm",
            form="linear",
        )


def test_recursive_partition_schedule_is_exact_when_residues_are_retained():
    tn = _ring4(seed=18)
    exact = tn.contract(optimize="auto-hq")
    result = recursive_partitioned_expand(
        tn,
        partition_levels=(("e0", "e1"), ("e2",)),
        norm="1norm",
        form="linear",
        include_residue=True,
        max_iterations=500,
        tol=1e-10,
    )

    assert isinstance(result, RecursivePNEExpansionResult)
    assert len(result.terms) == 6
    assert np.allclose(result.estimate, exact, rtol=1e-10, atol=1e-10)


def test_expansion_benchmark_is_json_ready():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bp_expansions.py"
    spec = importlib.util.spec_from_file_location("bp_expansions", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    records = module.benchmark_bp_expansions(
        _ring4(seed=19),
        norm="1norm",
        partition_inds=("e0",),
        loop_cutoff=3,
        cluster_cutoff=3,
        max_iterations=500,
        tol=1e-10,
    )
    assert {record.method for record in records} == {
        "bp",
        "loop_series",
        "loop_cluster",
        "pne",
    }
    assert all(isinstance(record.as_dict(), dict) for record in records)

    higher_rank = module.benchmark_bp_expansions(
        _ring4(seed=21),
        norm="1norm",
        auto_select=True,
        max_partitions=2,
        weight_passing_rank=2,
        loop_cutoff=2,
        cluster_cutoff=2,
        max_iterations=100,
        tol=1e-8,
    )
    assert {record.method for record in higher_rank} == {
        "bp",
        "loop_series",
        "loop_cluster",
        "pne",
        "pne_weight_pass",
    }
