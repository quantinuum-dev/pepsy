"""Explicit post-compression error measured against the fixed p-cluster MPO."""

import numpy as np
import pytest

from pepsy.operators import MPOBasis

pytestmark = [pytest.mark.integration, pytest.mark.operators, pytest.mark.mpo]


def test_fixed_mpo_numerical_compression_error_matches_dense_reference():
    sites = tuple((i, j) for i in range(2) for j in range(3))
    edges = tuple(
        (site, (site[0] + di, site[1] + dj))
        for site in sites
        for di, dj in ((1, 0), (0, 1))
        if site[0] + di < 2 and site[1] + dj < 3
    )
    terms = [(('X', 0.2), site) for site in sites]
    terms += [(('ZZ', 0.4), edge) for edge in edges]
    basis = MPOBasis.from_terms(terms, shape=(2, 3))
    compiled = basis.compile_graph_cluster_expansion(
        graph="square", cluster_size=2, assembly="recursive",
        factorization="fixed", cutoff=0.0,
    )
    exact_cluster = compiled(-0.03j)
    original = exact_cluster.to_mpo().to_dense()
    initial_bonds = exact_cluster.bond_dimensions
    for chi in (4, 16, 32):
        compressed, report = exact_cluster.compress_numerical(
            max_bond=chi, cutoff=0.0, estimate_error=True,
            return_report=True,
        )
        actual_error = np.linalg.norm(compressed.to_dense() - original) / np.linalg.norm(original)
        assert report.method == "quimb"
        assert report.error_estimator == "tensor-network-frobenius"
        assert max(report.final_bond_dimensions) <= chi
        assert report.operator_frobenius_relative_error == pytest.approx(
            actual_error, rel=2e-3, abs=1e-15,
        )
        assert exact_cluster.bond_dimensions == initial_bonds
        np.testing.assert_allclose(exact_cluster.to_mpo().to_dense(), original, atol=1e-13)


def test_torch_postcompression_error_matches_dense_reference():
    torch = pytest.importorskip("torch")
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    basis = MPOBasis.from_local_terms(
        3,
        [((0,), (x,), 0.2), ((0, 1), (z, z), 0.4), ((1, 2), (z, z), 0.4)],
    )
    semantic = basis.compile_graph_cluster_expansion(
        graph="chain", cluster_size=3, assembly="recursive",
        factorization="fixed", cutoff=0.0,
    )(torch.tensor(0.03, dtype=torch.float64))
    compressed, report = semantic.compress_numerical(
        max_bond=1, cutoff=0.0, estimate_error=True, return_report=True,
    )
    original = semantic.to_mpo().to_dense()
    expected = torch.linalg.vector_norm(original - compressed.to_dense()) / torch.linalg.vector_norm(original)
    torch.testing.assert_close(report.operator_frobenius_relative_error, expected, atol=1e-12, rtol=1e-10)
    assert report.truncated
