"""Tests for the semantic higher-order MPO foundation."""

import numpy as np

import pepsy
from pepsy.operators import (
    FirstDegreeMPO,
    MPOCompressionReport,
    MPOLevel,
    MPOLevelToken,
    MPOProductTerm,
)


def _paulis():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.diag([1.0, -1.0])
    return x, y, z


def _two_term_mpo():
    x, _, z = _paulis()
    return FirstDegreeMPO.from_local_terms(
        3,
        [
            MPOProductTerm((0, 1), (x, x)),
            MPOProductTerm((1, 2), (z, z)),
        ],
    )


def test_first_degree_mpo_public_exports_resolve():
    """The new semantic MPO layer belongs to ``pepsy.operators``."""
    assert FirstDegreeMPO is pepsy.operators.FirstDegreeMPO
    assert MPOLevel is pepsy.operators.MPOLevel
    assert MPOLevelToken is pepsy.operators.MPOLevelToken
    assert MPOProductTerm is pepsy.operators.MPOProductTerm
    assert MPOCompressionReport is pepsy.operators.MPOCompressionReport
    assert "FirstDegreeMPO" in pepsy.operators.__all__


def test_first_degree_mpo_builds_exact_local_term_sum():
    """Factorized local terms compile to the expected ordinary MPO."""
    x, _, z = _paulis()
    H = _two_term_mpo()
    expected = np.kron(np.kron(x, x), np.eye(2))
    expected += np.kron(np.kron(np.eye(2), z), z)

    np.testing.assert_allclose(H.to_mpo().to_dense(), expected)
    assert H.degree == 1
    assert H.is_first_degree
    assert H.bond_dimensions == (3, 3)
    assert H.levels[1][0].history == (MPOLevelToken(1),)
    assert H.levels[1][2].history[0].level == 2


def test_first_degree_mpo_add_scale_and_product_are_exact():
    """The foundational algebra agrees with dense operator algebra."""
    H = _two_term_mpo()
    dense = H.to_mpo().to_dense()

    np.testing.assert_allclose(H.add(H).to_mpo().to_dense(), 2.0 * dense)
    np.testing.assert_allclose(H.scale(-0.25).to_mpo().to_dense(), -0.25 * dense)
    np.testing.assert_allclose(
        H.non_disjoint_product(H).to_mpo().to_dense(), dense @ dense
    )
    np.testing.assert_allclose(
        H.power(3).to_mpo().to_dense(), dense @ dense @ dense
    )
    np.testing.assert_allclose(
        H.commutator(H).to_mpo().to_dense(), np.zeros_like(dense)
    )


def test_first_degree_mpo_exact_history_compression_preserves_operator():
    """Paper-style history merges are exact and reduce redundant channels."""
    H2 = _two_term_mpo().power(2)
    compressed = H2.compress_exact()

    assert isinstance(compressed.compression_report, MPOCompressionReport)
    assert compressed.compression_report.exact is True
    assert compressed.compression_report.merged_channels > 0
    assert compressed.bond_dimensions == (6, 6)
    np.testing.assert_allclose(
        compressed.to_mpo().to_dense(),
        H2.to_mpo().to_dense(),
        atol=0.0,
    )


def test_first_degree_mpo_compression_can_update_in_place():
    """The explicit in-place option keeps the object identity."""
    H2 = _two_term_mpo().power(2)
    result = H2.compress_exact(inplace=True)

    assert result is H2
    assert H2.compression_report.final_bond_dimensions == (6, 6)


def test_first_degree_mpo_identity_is_degree_zero():
    """The identity is available as the neutral algebra element."""
    identity = FirstDegreeMPO.identity(3, 2)
    np.testing.assert_allclose(identity.to_mpo().to_dense(), np.eye(8))
    assert identity.degree == 0
