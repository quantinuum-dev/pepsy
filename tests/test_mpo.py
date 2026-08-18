"""Tests for the semantic higher-order MPO foundation."""

from math import factorial

import numpy as np
import pytest

import pepsy
from pepsy.operators import (
    FirstDegreeMPO,
    MPOCompressionReport,
    MPOBasis,
    MPOLevel,
    MPOLevelToken,
    MPOParameter,
    MPODifferentiableCompressionReport,
    MPONumericalCompressionReport,
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
    assert MPOBasis is pepsy.operators.MPOBasis
    assert MPOParameter is pepsy.operators.MPOParameter
    assert MPOLevel is pepsy.operators.MPOLevel
    assert MPOLevelToken is pepsy.operators.MPOLevelToken
    assert MPOProductTerm is pepsy.operators.MPOProductTerm
    assert MPOCompressionReport is pepsy.operators.MPOCompressionReport
    assert (
        MPODifferentiableCompressionReport
        is pepsy.operators.MPODifferentiableCompressionReport
    )
    assert (
        MPONumericalCompressionReport
        is pepsy.operators.MPONumericalCompressionReport
    )
    assert "FirstDegreeMPO" in pepsy.operators.__all__


def test_mpo_basis_reuses_compiled_automaton_for_rebinding():
    """Parameter rebinding changes weights without rebuilding topology."""
    x, _, z = _paulis()
    basis = MPOBasis.from_local_terms(
        3,
        [
            MPOProductTerm((0, 1), (x, x), coefficient=MPOParameter("J")),
            {"sites": (1, 2), "operators": (z, z), "parameter": "K"},
        ],
    )

    first = basis.build({"J": 2.0, "K": -0.5})
    second = basis.build({"J": -1.0, "K": 0.25})
    identity = np.eye(2)
    expected_first = (
        2.0 * np.kron(np.kron(x, x), identity)
        - 0.5 * np.kron(np.kron(identity, z), z)
    )
    expected_second = (
        -np.kron(np.kron(x, x), identity)
        + 0.25 * np.kron(np.kron(identity, z), z)
    )

    np.testing.assert_allclose(first.to_mpo().to_dense(), expected_first)
    np.testing.assert_allclose(second.to_mpo().to_dense(), expected_second)
    assert basis.cache_info["compiled"] is True
    assert basis.cache_info["compiled_terms"] == 2
    assert basis.cache_info["builds"] == 2
    assert first.bond_dimensions == basis.bond_dimensions
    assert [level.history[0].level for level in first.levels[1]].count(1) == 1
    assert [level.history[0].level for level in first.levels[1]].count(3) == 1


def test_mpo_basis_evolution_keeps_parameterized_coefficients_differentiable():
    """The cached basis feeds the paper-style evolution MPO unchanged."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 2), "ZX", MPOParameter("theta"))],
    )

    U = basis.evolution_mpo(
        {"theta": theta},
        dt=time,
        order=2,
        mode="optimal",
    )
    loss = sum(array.real.sum() for array in U.arrays)
    theta_grad, time_grad = torch.autograd.grad(loss, (theta, time))

    assert any(array.requires_grad for array in U.arrays)
    assert torch.isfinite(theta_grad)
    assert torch.isfinite(time_grad)
    assert basis.cache_info["builds"] == 1


def test_mpo_basis_shares_suffixes_and_assembles_terminal_coefficient_groups():
    """Shared suffix channels remain exact when several terms end together."""
    x, y, z = _paulis()
    basis = MPOBasis.from_local_terms(
        5,
        [
            MPOProductTerm((0, 4), (z, x), coefficient=MPOParameter("a")),
            MPOProductTerm((2, 4), (y, x), coefficient=MPOParameter("b")),
        ],
    )
    expected = (
        1.25 * np.kron(np.kron(np.kron(np.kron(z, np.eye(2)), np.eye(2)), np.eye(2)), x)
        - 0.5 * np.kron(np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), y), np.eye(2)), x)
    )
    bound = basis.build({"a": 1.25, "b": -0.5})

    np.testing.assert_allclose(bound.to_mpo().to_dense(), expected)
    assert bound.bond_dimensions == basis.bond_dimensions
    assert basis.bond_dimensions[-1] < 4


def test_mpo_basis_batches_coefficients_and_reuses_history_topology():
    """Coefficient batches and repeated evolution share structural history."""
    x, _, z = _paulis()
    basis = MPOBasis.from_local_terms(
        3,
        [
            MPOProductTerm((0, 1), (x, x), coefficient=MPOParameter("a")),
            MPOProductTerm((1, 2), (z, z), coefficient=MPOParameter("b")),
        ],
    )

    first = basis.extensive_exponential(
        0.01,
        {"a": 0.7, "b": -0.2},
        order=2,
    )
    second = basis.extensive_exponential(
        0.01,
        coefficients=np.array([0.7, -0.2]),
        order=2,
    )

    assert basis.coefficients({"a": 0.7, "b": -0.2}).shape == (2,)
    assert first.metadata["history_cache_hit"] is False
    assert second.metadata["history_cache_hit"] is True
    assert basis.template.history_cache_info["orders"] == (2,)
    np.testing.assert_allclose(first.to_mpo().to_dense(), second.to_mpo().to_dense())


def test_history_algorithms_reuse_symbolic_execution_plans():
    """Algorithms 1--3 reuse topology plans without retaining tensor values."""
    H = _two_term_mpo()
    first = H.extensive_exponential(0.01, order=2, mode="optimal")
    second = H.extensive_exponential(0.02, order=2, mode="optimal")

    assert first.metadata["compression_plan_cache_hit"] is False
    assert first.metadata["extension_plan_cache_hit"] is False
    assert second.metadata["compression_plan_cache_hit"] is True
    assert second.metadata["extension_plan_cache_hit"] is True
    assert H.history_cache_info["compression_plan_orders"] == (2,)
    assert H.history_cache_info["extension_plan_orders"] == (2,)
    assert H.history_cache_info["extension_plan_batches"][2] > 0
    assert all(
        not hasattr(value, "requires_grad")
        for plan in H._history_extension_plan_cache.values()
        for batch in plan["batches"]
        for value in batch.values()
    )


def test_fixed_rank_compression_has_fixed_bonds_and_report():
    """Fixed-rank compression is separate from semantic history compression."""
    H = _two_term_mpo()
    compressed, report = H.compress_fixed_rank(2, return_report=True)
    exact = H.compress_fixed_rank(3)

    assert isinstance(report, MPODifferentiableCompressionReport)
    assert report.method == "fixed-rank-tt-svd"
    assert report.max_bond == 2
    assert report.truncated is True
    assert compressed.bond_dimensions == (2, 2)
    assert compressed.metadata["history_valid"] is False
    np.testing.assert_allclose(
        exact.to_mpo().to_dense(),
        H.to_mpo().to_dense(),
    )
    with pytest.raises(ValueError, match="fixed-rank compression"):
        compressed.extensive_exponential(0.01, order=2)


def test_fixed_rank_compression_preserves_autodiff():
    """Torch gradients pass through the fixed-rank SVD sweep."""
    torch = pytest.importorskip("torch")
    x, _, z = _paulis()
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    basis = MPOBasis.from_local_terms(
        3,
        [
            MPOProductTerm((0, 1), (x, x)),
            MPOProductTerm((1, 2), (z, z)),
        ],
    )
    H = basis.build(coefficients=torch.stack((theta, 0.3 * theta)))
    compressed = H.compress_fixed_rank(2)
    loss = sum(array.real.sum() for array in compressed.arrays)
    gradient, = torch.autograd.grad(loss, theta)

    assert torch.isfinite(gradient)


def test_first_degree_mpo_exposes_optional_compression_report_slot():
    """The report attribute is stable before and after compression."""
    H = _two_term_mpo()

    assert H.compression_report is None
    assert H.compress_exact().compression_report is not None


def test_first_degree_mpo_builds_exact_local_term_sum():
    """Factorized local terms compile to the expected ordinary MPO."""
    x, _, z = _paulis()
    H = _two_term_mpo()
    expected = np.kron(np.kron(x, x), np.eye(2))
    expected += np.kron(np.kron(np.eye(2), z), z)

    np.testing.assert_allclose(H.to_mpo().to_dense(), expected)
    assert H.to_mpo().cyclic is False
    assert H.degree == 1
    assert H.is_first_degree
    assert H.bond_dimensions == (3, 3)
    assert H.levels[1][0].history == (MPOLevelToken(1),)
    assert H.levels[1][2].history[0].level == 2


def test_first_degree_mpo_parses_compact_pauli_terms():
    """Pauli labels compile to an exact long-range product operator."""
    identity = np.eye(2)
    x, y, z = _paulis()
    H = FirstDegreeMPO.from_pauli_terms(
        5,
        [((0, 2, 4), "ZXY", 0.7)],
    )
    expected = 0.7 * np.kron(
        np.kron(np.kron(np.kron(z, identity), x), identity), y,
    )

    np.testing.assert_allclose(H.to_mpo().to_dense(), expected)
    np.testing.assert_allclose(
        MPOProductTerm.from_pauli((0, 1), "ZX").operators[0],
        z,
    )


def test_first_degree_mpo_shares_pauli_prefixes_exactly():
    """Repeated Pauli paths share channels without changing the operator."""
    shared = FirstDegreeMPO.from_pauli_terms(
        5,
        [((0, 4), "ZX", 1.0), ((0, 4), "ZY", 2.0)],
    )
    unshared = FirstDegreeMPO.from_pauli_terms(
        5,
        [((0, 4), "ZX", 1.0), ((0, 4), "ZY", 2.0)],
        share_channels=False,
    )

    assert shared.bond_dimensions == (3, 3, 3, 3)
    assert unshared.bond_dimensions == (4, 4, 4, 4)
    np.testing.assert_allclose(
        shared.to_mpo().to_dense(),
        unshared.to_mpo().to_dense(),
    )

    suffix_shared = FirstDegreeMPO.from_pauli_terms(
        6,
        [((0, 4), "ZX"), ((2, 4), "YX")],
    )
    suffix_unshared = FirstDegreeMPO.from_pauli_terms(
        6,
        [((0, 4), "ZX"), ((2, 4), "YX")],
        share_channels=False,
    )
    assert suffix_shared.bond_dimensions == (3, 3, 3, 3, 2)
    assert suffix_unshared.bond_dimensions == (3, 3, 4, 4, 2)
    np.testing.assert_allclose(
        suffix_shared.to_mpo().to_dense(),
        suffix_unshared.to_mpo().to_dense(),
    )


def test_first_degree_mpo_rejects_unknown_pauli_labels():
    """Compact labels fail early with a useful error."""
    with pytest.raises(ValueError, match="Pauli labels"):
        FirstDegreeMPO.from_pauli_terms(3, [((0, 2), "ZA")])


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
        assert U.metadata["algorithms"] == ("one-site-taylor",)


def test_extensive_exponential_one_site_supports_arbitrary_order_and_extension():
    """One-site Taylor evaluation supports high orders and Algorithm 3 mode."""
    _, _, z = _paulis()
    H = FirstDegreeMPO.from_local_terms(
        1,
        [MPOProductTerm((0,), (z,))],
    )

    for order in (3, 5):
        U = H.extensive_exponential(0.2, order=order)
        expected = sum(
            0.2**power * np.linalg.matrix_power(z, power) / factorial(power)
            for power in range(order + 1)
        )
        np.testing.assert_allclose(U.to_mpo().to_dense(), expected)
        assert U.metadata["order"] == order

        extended = H.extensive_exponential(0.2, order=order, mode="optimal")
        expected_extended = sum(
            0.2**power
            * np.linalg.matrix_power(z, power)
            / factorial(power)
            for power in range(order + 2)
        )
        np.testing.assert_allclose(
            extended.to_mpo().to_dense(),
            expected_extended,
        )
        assert extended.metadata["extension_requested"] is True


def test_extensive_exponential_one_site_arbitrary_order_supports_torch_autograd():
    """The direct local Taylor loop preserves Torch parameter gradients."""
    torch = pytest.importorskip("torch")
    _, _, z = _paulis()
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    H = FirstDegreeMPO.from_local_terms(
        1,
        [MPOProductTerm((0,), (z,), coefficient)],
    )
    U = H.extensive_exponential(time, order=5, mode="optimal")
    loss = U.arrays[0].real.sum()
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficient, time),
    )
    assert torch.isfinite(coefficient_gradient)
    assert torch.isfinite(time_gradient)


def test_extensive_exponential_one_site_arbitrary_order_supports_jax_jit():
    """The direct local Taylor loop remains functional under JAX JIT."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    _, _, z = _paulis()

    def objective(coefficient, time):
        H = FirstDegreeMPO.from_local_terms(
            1,
            [MPOProductTerm((0,), (z,), coefficient)],
        )
        U = H.extensive_exponential(time, order=5, mode="optimal")
        return jnp.real(U.arrays[0]).sum()

    value, gradients = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1)),
    )(0.7, 0.2)
    assert jnp.isfinite(value)
    assert all(jnp.isfinite(gradient) for gradient in gradients)


def test_extensive_exponential_streaming_and_sparse_storage_match_dense():
    """Ephemeral storage modes avoid dead local products without changing U."""
    H = _two_term_mpo()
    dense = H.extensive_exponential(
        0.01,
        order=3,
        mode="optimal",
        cache_history=True,
        history_storage="dense",
    )
    streaming = H.extensive_exponential(
        0.01,
        order=3,
        mode="optimal",
        cache_history=False,
    )
    sparse = H.extensive_exponential(
        0.01,
        order=3,
        mode="optimal",
        cache_history=False,
        history_storage="sparse",
    )

    expected = dense.to_mpo().to_dense()
    np.testing.assert_allclose(streaming.to_mpo().to_dense(), expected)
    np.testing.assert_allclose(sparse.to_mpo().to_dense(), expected)
    assert streaming.metadata["history_storage"] == "streaming"
    assert sparse.metadata["history_storage"] == "sparse"
    assert sparse.metadata["history_storage_blocks"]["stored_blocks"] < (
        sparse.metadata["history_storage_blocks"]["total_blocks"]
    )
    assert streaming.history_cache_info["orders"] == ()
    assert sparse.history_cache_info["orders"] == ()


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


def test_extensive_exponential_uses_reachable_history_channels():
    """Raw histories omit channels unreachable from the finite left boundary."""
    H = _two_term_mpo()
    U = H.extensive_exponential(0.01, order=3)

    assert U.metadata["history_generation"] == "reachable"
    assert U.metadata["initial_bond_dimensions"][0] < 3**3


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
    assert H.history_cache_info["orders"] == (2,)
    plain_error = np.linalg.norm(plain.to_mpo().to_dense() - scipy_linalg.expm(0.01 * dense_h))
    extended_error = np.linalg.norm(
        extended.to_mpo().to_dense() - scipy_linalg.expm(0.01 * dense_h),
    )
    assert extended_error < plain_error


def test_extensive_exponential_can_release_history_topology_after_build():
    """One-off large-order builds can avoid retaining the topology cache."""
    H = _two_term_mpo()
    result = H.extensive_exponential(
        0.01,
        order=3,
        cache_history=False,
    )

    assert result.metadata["cache_history"] is False
    assert result.metadata["history_cache_hit"] is False
    assert H.history_cache_info["orders"] == ()


def test_extensive_exponential_optimal_mode_selects_paper_extension():
    """The named optimal mode is the exact Algorithms 1--3 policy."""
    H = _two_term_mpo()
    explicit = H.extensive_exponential(0.01, order=2, extend=True)
    named = H.extensive_exponential(0.01, order=2, mode="optimal")

    assert named.metadata["mode"] == "optimal"
    assert named.metadata["algorithms"] == (1, 2, 3)
    assert named.bond_dimensions == explicit.bond_dimensions
    np.testing.assert_allclose(
        named.to_mpo().to_dense(),
        explicit.to_mpo().to_dense(),
    )


def test_extensive_exponential_bond_guard_can_raise_or_warn():
    """Temporary history growth is bounded before later compression."""
    H = _two_term_mpo()
    with pytest.raises(MemoryError, match="max_bond"):
        H.extensive_exponential(0.01, order=2, max_bond=1)

    with pytest.warns(RuntimeWarning, match="max_bond"):
        warned = H.extensive_exponential(
            0.01,
            order=2,
            max_bond=1,
            on_exceed="warn",
        )
    assert warned.metadata["max_bond"] == 1
    assert warned.metadata["on_exceed"] == "warn"


def test_extensive_exponential_rejects_conflicting_mode_flags():
    """Named policies cannot silently disagree with legacy flags."""
    with pytest.raises(ValueError, match="cannot be combined"):
        _two_term_mpo().extensive_exponential(
            0.01,
            order=2,
            mode="optimal",
            approximate=True,
        )


def test_parameterized_pauli_hamiltonian_preserves_torch_autograd():
    """Backend scalar coefficients survive evolution MPO construction."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    H = FirstDegreeMPO.from_pauli_terms(
        3,
        [((0, 2), "ZX", theta)],
    )

    U = H.extensive_exponential(
        -1j * time,
        order=2,
        mode="optimal",
    )
    dense = U.to_mpo().to_dense()
    assert isinstance(dense, torch.Tensor)
    assert dense.requires_grad
    loss = dense.real.sum()
    theta_grad, time_grad = torch.autograd.grad(loss, (theta, time))
    assert torch.isfinite(theta_grad)
    assert torch.isfinite(time_grad)


def test_parameterized_observable_expectation_preserves_torch_autograd():
    """Parameterized Pauli terms can also be used as observables."""
    torch = pytest.importorskip("torch")
    qtn = pytest.importorskip("quimb.tensor")
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    observable = FirstDegreeMPO.from_pauli_terms(
        3,
        [((0, 2), "ZZ", theta)],
    )
    state = qtn.MPS_computational_state("000")

    value = observable.expectation(state)
    assert isinstance(value, torch.Tensor)
    assert value.requires_grad
    torch.testing.assert_close(value.real, theta)
    (gradient,) = torch.autograd.grad(value.real, (theta,))
    torch.testing.assert_close(gradient, torch.ones_like(theta))


def test_parameterized_pauli_hamiltonian_supports_jax_autodiff():
    """Functional history updates keep the JAX autodiff path available."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def objective(theta, time):
        H = FirstDegreeMPO.from_pauli_terms(
            3,
            [((0, 2), "ZX", theta)],
        )
        U = H.extensive_exponential(
            -1j * time,
            order=2,
            mode="optimal",
            cache_history=False,
        )
        return sum(jnp.real(array).sum() for array in U.arrays)

    value, gradients = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1)),
    )(0.7, 0.01)
    assert jnp.isfinite(value)
    assert all(jnp.isfinite(gradient) for gradient in gradients)


def test_parameterized_mpo_basis_supports_jax_batched_coefficients():
    """The finite optimal path supports JAX coefficient batches under jit."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )

    def objective(coefficients, time):
        U = basis.evolution_mpo(
            coefficients=coefficients,
            dt=time,
            order=2,
            mode="optimal",
            cache_history=False,
        )
        return sum(jnp.real(array).sum() for array in U.arrays)

    value, gradients = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1)),
    )(jnp.array([0.7, -0.2]), 0.01)

    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)


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


def test_numerical_compression_delegates_to_quimb_with_report():
    """Numerical truncation is explicit and drops stale semantic histories."""
    U = _two_term_mpo().extensive_exponential(0.01, order=2)
    compressed, report = U.compress_numerical(
        form="flat",
        max_bond=1,
        cutoff=0.0,
        return_report=True,
    )

    assert isinstance(report, MPONumericalCompressionReport)
    assert report.method == "quimb"
    assert report.max_bond == 1
    assert report.cutoff == 0.0
    assert report.truncated is True
    assert report.truncation_error is None
    assert compressed.cyclic is False
    assert compressed.bond_sizes() == [1, 1]
    assert compressed.pepsy_first_degree is None
    assert compressed.pepsy_numerical_compression_report is report


def test_numerical_compression_can_estimate_operator_frobenius_error():
    """Compression diagnostics can contract an MPO-level error estimate."""
    U = _two_term_mpo().extensive_exponential(0.01, order=2)
    original_dense = U.to_mpo().to_dense()
    compressed, report = U.compress_numerical(
        form="flat",
        max_bond=1,
        cutoff=0.0,
        estimate_error=True,
        return_report=True,
    )

    expected = np.linalg.norm(original_dense - compressed.to_dense())
    assert report.error_estimator == "tensor-network-frobenius"
    assert report.operator_frobenius_error == pytest.approx(expected)
    assert report.truncation_error == pytest.approx(expected)
    assert report.operator_frobenius_relative_error == pytest.approx(
        expected / np.linalg.norm(original_dense),
    )


def test_numerical_compression_validates_max_bond():
    """The Pepsy wrapper rejects invalid numerical compression policies."""
    U = _two_term_mpo().extensive_exponential(0.01, order=1)
    with pytest.raises(ValueError, match="max_bond"):
        U.compress_numerical(max_bond=0)


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
