"""Tests for the edge-resolved P+Q BP loop-series expansion."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    LoopSeriesCache,
    LoopSeriesResult,
    LoopSeriesTerm,
    loop_expand,
    loop_series_expand,
)


def _ring4(seed=1):
    rng = np.random.default_rng(seed)
    tensors = []
    for tid in range(4):
        tensors.append(
            qtn.Tensor(
                rng.random((2, 2)) + 0.2,
                inds=(f"e{tid}", f"e{(tid - 1) % 4}"),
                tags={f"T{tid}"},
            )
        )
    return qtn.TensorNetwork(tensors)


def _square_with_chord(seed=2):
    rng = np.random.default_rng(seed)
    edge_names = {
        0: ("e01", "e30", "e02"),
        1: ("e01", "e12"),
        2: ("e12", "e23", "e02"),
        3: ("e23", "e30"),
    }
    return qtn.TensorNetwork(
        [
            qtn.Tensor(
                rng.random(tuple(2 for _ in inds)) + 0.2,
                inds=inds,
                tags={tid},
            )
            for tid, inds in edge_names.items()
        ]
    )


def test_public_loop_series_exports():
    from pepsy import bp

    assert {
        "LoopSeriesCache",
        "LoopSeriesResult",
        "LoopSeriesTerm",
        "loop_expand",
        "loop_series_expand",
    } <= set(bp.__all__)


def test_common_selector_keeps_the_two_expansion_structures_explicit():
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, phys_dim=2, seed=6)

    series = loop_expand(
        peps,
        gloops=4,
        expansion="series",
        max_iterations=500,
        tol=1e-10,
        multi_excitation_correct=False,
    )
    cluster = loop_expand(
        peps,
        gloops=4,
        expansion="loop_cluster",
        max_iterations=500,
        tol=1e-10,
    )

    assert isinstance(series, LoopSeriesResult)
    assert series.expansion == "series"
    assert series.cutoff_kind == "excited-bond-degree"
    assert cluster.expansion == "cluster"
    assert cluster.cutoff_kind == "tensor-region-size"
    assert series.terms[0].degree == 4

    with pytest.raises(ValueError, match="edge-resolved P/Q"):
        loop_expand(peps, expansion="unknown")
    with pytest.raises(TypeError, match="only a loop-cluster option"):
        loop_expand(peps, expansion="series", combine="sum")
    with pytest.raises(TypeError, match="only a loop-series option"):
        loop_expand(peps, expansion="cluster", multi_excitation_correct=False)


def test_edge_enumerator_keeps_embeddings_and_chord_subsets_distinct():
    tn = _square_with_chord()
    terms = LoopSeriesCache().terms_for(tn, 5)

    assert {term.degree for term in terms} == {3, 4, 5}
    support_four = [term for term in terms if term.tids == frozenset(range(4))]
    assert len(support_four) == 2
    assert {term.degree for term in support_four} == {4, 5}
    assert all(term.edges != support_four[0].edges for term in support_four[1:])


def test_integer_cutoff_counts_separate_square_embeddings():
    peps = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, phys_dim=2, seed=3)
    result = loop_series_expand(
        peps,
        gloops=4,
        max_iterations=500,
        tol=1e-10,
        multi_excitation_correct=False,
    )

    assert isinstance(result, LoopSeriesResult)
    assert len(result.terms) == 4
    assert {term.degree for term in result.terms} == {4}
    assert len({term.edges for term in result.terms}) == 4


def test_d2_loop_series_matches_quimb_for_a_region_without_chords():
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, phys_dim=2, seed=4)
    regions = tuple(peps.gen_gloops(max_size=4))

    from quimb.tensor.belief_propagation import D2BP

    bp = D2BP(peps)
    bp.run(max_iterations=500, tol=1e-10, tol_rolling_diff=0.0)
    expected = bp.contract_loop_series_expansion(
        regions,
        multi_excitation_correct=False,
    )
    actual = loop_series_expand(
        peps,
        gloops=regions,
        max_iterations=500,
        tol=1e-10,
        multi_excitation_correct=False,
    )

    assert np.allclose(actual.estimate, expected, rtol=1e-10, atol=1e-10)


def test_d1_loop_series_matches_quimb_on_a_ring():
    tn = _ring4()
    regions = tuple(tn.gen_gloops(max_size=4))

    from quimb.tensor.belief_propagation import D1BP

    bp = D1BP(tn)
    bp.run(max_iterations=500, tol=1e-10, tol_rolling_diff=0.0)
    expected = bp.contract_loop_series_expansion(
        regions,
        multi_excitation_correct=False,
    )
    actual = loop_series_expand(
        tn,
        gloops=regions,
        norm="1norm",
        max_iterations=500,
        tol=1e-10,
        multi_excitation_correct=False,
    )

    assert np.allclose(actual.estimate, expected, rtol=1e-10, atol=1e-10)


def test_explicit_edge_term_and_result_expansion_reuse_bp():
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, phys_dim=2, seed=5)
    edge_terms = tuple(
        LoopSeriesTerm(term.edges)
        for term in LoopSeriesCache().terms_for(peps, 4)
    )
    first = loop_series_expand(
        peps,
        gloops=edge_terms,
        max_iterations=500,
        tol=1e-10,
        multi_excitation_correct=False,
    )
    second = first.expand(
        edge_terms,
        multi_excitation_correct=False,
    )

    assert np.allclose(first.estimate, second, rtol=1e-10, atol=1e-10)
    assert first.messages
