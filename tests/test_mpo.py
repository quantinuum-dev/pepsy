"""Tests for the semantic higher-order MPO foundation."""

import numpy as np
import pytest

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


def test_extensive_exponential_builds_local_order_one_mpo():
    """Order one uses local MPO blocks and folds the done rail into one."""
    H = _two_term_mpo()
    U = H.extensive_exponential(0.01, order=1)

    assert U.metadata["operation"] == "extensive_exponential"
    assert U.metadata["order"] == 1
    assert U.bond_dimensions == (2, 2)
    assert all(array.ndim == 4 for array in U.arrays)
    assert all(level.history[0].level == 1 for level in U.levels[0])
    assert all(level.history[0].level == 1 for level in U.levels[-1])


def test_extensive_exponential_matches_dense_taylor_orders():
    """The tensor-network construction has the paper's expected order."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    H = _two_term_mpo()
    dense_h = H.to_mpo().to_dense()
    identity = np.eye(dense_h.shape[0])

    for order in (1, 2):
        errors = []
        for dt in (1.0e-2, 5.0e-3):
            dense_u = H.extensive_exponential(dt, order=order).to_mpo().to_dense()
            errors.append(np.linalg.norm(dense_u - scipy_linalg.expm(dt * dense_h)))

        assert errors[1] / errors[0] == pytest.approx(
            0.25 if order == 1 else 0.125,
            rel=3.0e-3,
        )
        zero_step = H.extensive_exponential(0.0, order=order).to_mpo().to_dense()
        np.testing.assert_allclose(zero_step, identity)


def test_extensive_exponential_handles_one_site_terms():
    """The finite-chain boundary construction also covers L=1."""
    _, _, z = _paulis()
    H = FirstDegreeMPO.from_local_terms(
        1,
        [MPOProductTerm((0,), (z,))],
    )

    for order in (1, 2):
        U = H.extensive_exponential(0.2, order=order)
        expected = np.eye(2) + 0.2 * z
        if order == 2:
            expected = expected + 0.2**2 * (z @ z) / 2
        np.testing.assert_allclose(U.to_mpo().to_dense(), expected)


def test_extensive_exponential_supports_generic_order_three_histories():
    """Generic histories reproduce the expected third-order scaling."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    H = _two_term_mpo()
    dense_h = H.to_mpo().to_dense()
    errors = []
    for dt in (1.0e-2, 5.0e-3):
        U = H.extensive_exponential(dt, order=3)
        errors.append(np.linalg.norm(U.to_mpo().to_dense() - scipy_linalg.expm(dt * dense_h)))
        assert U.metadata["algorithms"] == (1, 2)
        assert all(len(level.history) == 3 for level in U.levels[1])

    assert errors[1] / errors[0] == pytest.approx(1.0 / 16.0, rel=3.0e-3)


def test_extensive_exponential_algorithm_three_keeps_bond_dimension():
    """The extension adds selected next-order terms without new channels."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    H = _two_term_mpo()
    dense_h = H.to_mpo().to_dense()
    plain = H.extensive_exponential(0.01, order=2)
    extended = H.extensive_exponential(0.01, order=2, extend=True)

    assert extended.bond_dimensions == plain.bond_dimensions
    assert extended.metadata["algorithms"] == (1, 2, 3)
    assert extended.metadata["extension_terms"] > 0
    plain_error = np.linalg.norm(plain.to_mpo().to_dense() - scipy_linalg.expm(0.01 * dense_h))
    extended_error = np.linalg.norm(
        extended.to_mpo().to_dense() - scipy_linalg.expm(0.01 * dense_h),
    )
    assert extended_error < plain_error


def test_extensive_exponential_algorithm_four_is_explicit_and_order_controlled():
    """Approximate compression is opt-in and lowers the analytical rank."""
    H = _two_term_mpo()
    exact = H.extensive_exponential(0.01, order=2)
    approximate = H.extensive_exponential(0.01, order=2, approximate=True)

    assert approximate.metadata["algorithms"] == (1, 2, 4)
    assert approximate.metadata["approximate"] is True
    assert approximate.metadata["approximate_history_merges"] > 0
    assert all(
        approximate_dim <= exact_dim
        for approximate_dim, exact_dim in zip(
            approximate.bond_dimensions,
            exact.bond_dimensions,
        )
    )


def test_extensive_exponential_mps_expectation_and_application_are_tensor_network_paths(
    monkeypatch,
):
    """The public MPS helpers contract and apply without MPO densification."""
    qtn = pytest.importorskip("quimb.tensor")
    scipy_linalg = pytest.importorskip("scipy.linalg")
    H = _two_term_mpo()
    state = qtn.MPS_computational_state("000")
    dense_h = H.to_mpo().to_dense()
    state_vector = np.asarray(state.to_dense()).reshape(-1)

    errors = []
    for dt in (1.0e-2, 5.0e-3):
        U = H.extensive_exponential(dt, order=3)
        exact = scipy_linalg.expm(dt * dense_h)
        expected = np.vdot(state_vector, exact @ state_vector)
        errors.append(abs(U.expectation(state) - expected))
    assert errors[1] / errors[0] == pytest.approx(1.0 / 16.0, rel=3.0e-3)

    U = H.extensive_exponential(0.01, order=2)
    expected_state = U.to_mpo().to_dense() @ state_vector

    def forbid_mpo_dense(*args, **kwargs):
        del args, kwargs
        raise AssertionError("MPS application must not densify the MPO")

    monkeypatch.setattr(qtn.MatrixProductOperator, "to_dense", forbid_mpo_dense)
    applied = U.apply_to_mps(state, method="direct", cutoff=0.0)
    np.testing.assert_allclose(
        np.asarray(applied.to_dense()).reshape(-1),
        expected_state,
        atol=1.0e-10,
    )
