"""Tests for the semantic higher-order MPO foundation."""

from math import factorial

import numpy as np
import pytest

import pepsy
from pepsy.operators import (
    CompiledMPOEvolution,
    CompiledMPOExp,
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


def _symmray_mpo_to_dense(mpo):
    """Unfuse Symmray sectors before comparing in physical basis order."""
    value = mpo.to_dense()
    if hasattr(value, "unfuse_all"):
        value = value.unfuse_all()
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    value = np.asarray(value)
    dimension = int(round(np.sqrt(value.size)))
    return value.reshape(dimension, dimension)


def test_first_degree_mpo_public_exports_resolve():
    """The new semantic MPO layer belongs to ``pepsy.operators``."""
    assert FirstDegreeMPO is pepsy.operators.FirstDegreeMPO
    assert CompiledMPOEvolution is pepsy.operators.CompiledMPOEvolution
    assert CompiledMPOExp is pepsy.operators.CompiledMPOExp
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
    assert "CompiledMPOEvolution" in pepsy.operators.__all__
    assert "CompiledMPOExp" in pepsy.operators.__all__


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


def test_mpo_basis_exp_is_explicit_and_time_evolution_is_its_real_time_alias():
    """The generic exponential API makes the sign convention visible."""
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )
    coefficients = np.array([0.7, -0.2])

    explicit = basis.exp(
        -1j * 0.01,
        coefficients,
        order=2,
        mode="optimal",
    )
    real_time = basis.time_evolution(
        0.01,
        coefficients,
        order=2,
        mode="optimal",
    )

    assert explicit.metadata["operation"] == "exp"
    assert real_time.metadata["operation"] == "time_evolution"
    for explicit_array, real_time_array in zip(explicit.arrays, real_time.arrays):
        np.testing.assert_allclose(explicit_array, real_time_array)

    keyword = basis.exp(
        step=-1j * 0.01,
        coefficients=coefficients,
        order=2,
        mode="optimal",
    )
    legacy_keyword = basis.exp(
        dt=-1j * 0.01,
        coefficients=coefficients,
        order=2,
        mode="optimal",
    )
    for keyword_array, legacy_array in zip(
        keyword.arrays,
        legacy_keyword.arrays,
    ):
        np.testing.assert_allclose(keyword_array, legacy_array)
    with pytest.raises(TypeError, match="either step or dt"):
        basis.exp(
            step=0.01,
            dt=0.01,
            coefficients=coefficients,
        )


def test_mpo_basis_approximate_evolution_keeps_parameters_differentiable():
    """The cached Algorithm-4 plan does not capture backend values."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )

    U = basis.evolution_mpo(
        coefficients=torch.stack((theta, -0.3 * theta)),
        dt=time,
        order=2,
        mode="approximate",
    )
    loss = sum(array.real.sum() for array in U.arrays)
    theta_grad, time_grad = torch.autograd.grad(loss, (theta, time))

    assert U.metadata["algorithms"] == (1, 2, 3, 4)
    assert any(array.requires_grad for array in U.arrays)
    assert torch.isfinite(theta_grad)
    assert torch.isfinite(time_grad)


def test_mpo_basis_evolution_can_apply_a_final_chi_compression():
    """Final MPO chi is distinct from the temporary history-bond guard."""
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )

    mpo, report = basis.evolution_mpo(
        coefficients=np.array([0.7, -0.2]),
        dt=0.01,
        order=2,
        mode="optimal",
        chi=1,
        cutoff=0.0,
        return_report=True,
    )

    assert report.max_bond == 1
    assert report.final_bond_dimensions == (1, 1)
    assert mpo.bond_sizes() == [1, 1]
    assert mpo.pepsy_evolution_metadata["chi"] == 1
    second = basis.evolution_mpo(
        coefficients=np.array([0.6, -0.1]),
        dt=0.02,
        order=2,
        mode="optimal",
        chi=1,
    )
    assert mpo.pepsy_evolution_metadata["history_cache_hit"] is False
    assert second.pepsy_evolution_metadata["history_cache_hit"] is True
    assert basis.template.history_cache_info["orders"] == (2,)


def test_mpo_basis_evolution_chi_fixed_rank_preserves_autodiff():
    """The parameter-to-MPO path can use fixed-rank differentiable chi."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )

    compressed, report = basis.evolution_mpo(
        coefficients=torch.stack((theta, -0.3 * theta)),
        dt=time,
        order=2,
        mode="optimal",
        chi=2,
        differentiable=True,
        return_report=True,
    )
    loss = sum(array.real.sum() for array in compressed.arrays)
    theta_grad, time_grad = torch.autograd.grad(loss, (theta, time))

    assert isinstance(report, MPODifferentiableCompressionReport)
    assert compressed.bond_dimensions == (2, 2)
    assert compressed.metadata["chi"] == 2
    assert torch.isfinite(theta_grad)
    assert torch.isfinite(time_grad)


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


def test_mpo_basis_exponential_cache_can_be_cleared_explicitly():
    """The basis keeps topology reusable without forcing indefinite retention."""
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )

    first = basis.exp(
        -1j * 0.01,
        coefficients=np.array([0.7, -0.2]),
        order=2,
        mode="algorithm4",
    )
    assert first.metadata["history_cache_hit"] is False
    assert basis.cache_info["history"]["orders"] == (2,)

    basis.clear_history_cache()
    assert basis.cache_info["history"]["orders"] == ()
    assert basis.cache_info["history"]["approximation_plan_orders"] == ()

    second = basis.time_evolution(
        0.01,
        coefficients=np.array([0.7, -0.2]),
        order=2,
        mode="algorithm4",
    )
    assert second.metadata["history_cache_hit"] is False


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
        site_plan is None or site_plan["left_targets"].ndim == 1
        for site_plan in H._history_extension_plan_cache[2]["site_plans"]
    )
    assert all(
        not hasattr(value, "requires_grad")
        for plan in H._history_extension_plan_cache.values()
        for batch in plan["batches"]
        for value in batch.values()
    )
    assert all(
        not hasattr(value, "requires_grad")
        for plan in H._history_extension_plan_cache.values()
        for site_plan in plan["site_plans"]
        if site_plan is not None
        for value in site_plan.values()
    )


def test_history_tensor_execution_plan_is_cached_separately_from_values():
    """Repeated numerical builds reuse gather metadata without values."""
    H = _two_term_mpo()
    first = H.extensive_exponential(0.01, order=2, mode="base")
    second = H.extensive_exponential(0.02, order=2, mode="base")

    assert first.metadata["tensor_plan_cache_hit"] is False
    assert second.metadata["tensor_plan_cache_hit"] is True
    assert H.history_cache_info["tensor_plan_orders"] == (2,)
    assert all(
        not hasattr(value, "requires_grad")
        for plan in H._history_tensor_plan_cache.values()
        for site_plan in plan
        for value in site_plan.values()
        if value is not None
    )


def test_basis_raw_array_and_batch_apis_preserve_numpy_values():
    """Raw tensor APIs match semantic outputs and share the history cache."""
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )
    coefficients = np.array([0.7, -0.2])
    semantic = basis.exp(
        0.01,
        coefficients=coefficients,
        order=2,
        mode="base",
    )
    raw = basis.exp_arrays(
        0.01,
        coefficients=coefficients,
        order=2,
        mode="base",
    )
    for raw_array, semantic_array in zip(raw, semantic.arrays):
        np.testing.assert_allclose(raw_array, semantic_array)

    batch = basis.exp_batch(
        step=0.01,
        coefficients=np.stack((coefficients, [0.2, 0.3])),
        order=2,
        mode="base",
    )
    assert batch[0].shape[0] == 2
    np.testing.assert_allclose(batch[0][0], raw[0])


def test_compiled_exp_reuses_slot_bank_without_building_mpo():
    """The value-only evaluator matches the semantic path without rebinding."""
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )
    compiled = basis.compile_exp(order=2, mode="optimal")
    assert isinstance(compiled, CompiledMPOExp)
    assert isinstance(compiled, CompiledMPOEvolution)
    assert compiled is basis.compile_exp(order=2, mode="optimal")
    assert compiled is basis.compile_evolution(order=2, mode="optimal")
    assert basis.cache_info["builds"] == 0
    assert compiled.cache_info["fused_slot_sites"] > 0

    coefficients = np.array([0.7, -0.2])
    raw = compiled.exp_arrays(step=0.01, coefficients=coefficients)
    assert basis.cache_info["builds"] == 0

    semantic = compiled.exp(step=0.01, coefficients=coefficients).arrays
    for compiled_array, semantic_array in zip(raw, semantic):
        np.testing.assert_allclose(compiled_array, semantic_array)


def test_compiled_evolution_keeps_torch_graph_for_coefficients_and_time():
    """Static slot banks do not capture a stale Torch autodiff graph."""
    torch = pytest.importorskip("torch")
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )
    compiled = basis.compile_exp(order=2, mode="optimal")
    coefficients = torch.tensor(
        [0.7, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    arrays = compiled.exp_arrays(-1j * time, coefficients=coefficients)
    loss = sum(array.real.sum() for array in arrays)
    coefficient_grad, time_grad = torch.autograd.grad(loss, (coefficients, time))

    assert any(array.requires_grad for array in arrays)
    assert torch.isfinite(coefficient_grad).all()
    assert torch.isfinite(time_grad)
    assert basis.cache_info["builds"] == 0


def test_basis_batch_api_preserves_torch_autodiff():
    """Native Torch batching keeps coefficient and time gradients connected."""
    torch = pytest.importorskip("torch")
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )
    coefficients = torch.tensor(
        [[0.7, -0.2], [0.2, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

    arrays = basis.exp_batch(
        time,
        coefficients,
        order=2,
        mode="base",
    )
    loss = sum(array.real.sum() for array in arrays)
    coefficient_grad, time_grad = torch.autograd.grad(loss, (coefficients, time))

    assert arrays[0].shape[0] == 2
    assert torch.isfinite(coefficient_grad).all()
    assert torch.isfinite(time_grad)


def test_basis_raw_and_batch_apis_support_jax_jit_and_grad():
    """JAX compilation sees only backend arrays and structural constants."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )
    coefficients = jnp.array([0.7, -0.2])
    arrays = jax.jit(
        lambda values: basis.exp_arrays(
            jnp.array(0.01),
            coefficients=values,
            order=2,
            mode="base",
        )
    )(coefficients)
    gradient = jax.grad(
        lambda values: sum(
            jnp.real(array).sum()
            for array in basis.exp_arrays(
                jnp.array(0.01),
                coefficients=values,
                order=2,
                mode="base",
            )
        )
    )(coefficients)
    batch = basis.exp_batch(
        jnp.array(0.01),
        jnp.stack((coefficients, jnp.array([0.2, 0.3]))),
        order=2,
        mode="base",
    )

    assert arrays[0].shape[0] == 1
    assert batch[0].shape[0] == 2
    assert jnp.isfinite(gradient).all()


def test_history_algorithm_four_reuses_structural_index_plan():
    """Algorithm 4 compiles its ordered virtual-index merges once."""
    H = _two_term_mpo()
    first = H.extensive_exponential(0.01, order=2, mode="approximate")
    second = H.extensive_exponential(0.02, order=2, mode="approximate")

    assert first.metadata["approximation_plan_cache_hit"] is False
    assert second.metadata["approximation_plan_cache_hit"] is True
    assert H.history_cache_info["approximation_plan_orders"] == (2,)
    actions = H._history_approximation_plan_cache[2]["actions"]
    assert actions
    assert all(
        all(isinstance(value, int) for value in action)
        for action in actions
    )
    assert all(
        not hasattr(value, "requires_grad")
        for plan in H._history_approximation_plan_cache.values()
        for action in plan["actions"]
        for value in action
    )
    uncached = H.extensive_exponential(
        0.02,
        order=2,
        mode="approximate",
        cache_history=False,
        history_storage="streaming",
    )
    for cached_array, uncached_array in zip(second.arrays, uncached.arrays):
        np.testing.assert_allclose(cached_array, uncached_array)


def test_algorithm_four_mode_skips_expensive_next_order_extension():
    """The fast named policy selects Algorithm 4 without Algorithm 3."""
    H = _two_term_mpo()
    output = H.extensive_exponential(0.01, order=2, mode="algorithm4")

    assert output.metadata["mode"] == "algorithm4"
    assert output.metadata["algorithms"] == (1, 2, 4)
    assert output.metadata["extension_terms"] == 0
    assert output.metadata["approximate_history_merges"] > 0


def test_algorithm_four_fused_replay_matches_sequential_reference():
    """Fusing one bond's merges preserves the ordered Algorithm-4 update."""
    H = FirstDegreeMPO.from_pauli_terms(
        7,
        [((site, site + 1), "XX") for site in range(6)]
        + [((site,), "Z") for site in range(7)],
    )
    order = 2
    dt = -1j * 0.013
    arrays, levels, _, _ = H._history_power_data(
        order,
        cache_history=False,
        history_storage="auto",
    )
    compression_plan, _ = H._history_compression_plan(
        levels,
        order,
        cache_history=False,
    )
    H._algorithm_one(arrays, levels, order, dt, plan=compression_plan)
    H._algorithm_two(arrays, levels, plan=compression_plan)

    reference_arrays = [np.array(array, copy=True) for array in arrays]
    reference_levels = [list(bond_levels) for bond_levels in levels]
    fused_arrays = [np.array(array, copy=True) for array in arrays]
    fused_levels = [list(bond_levels) for bond_levels in levels]
    approximation_plan, _ = H._history_approximation_plan(
        reference_levels,
        order,
        cache_history=False,
    )
    for bond, source, target, number_of_threes in approximation_plan["actions"]:
        if (
            source >= len(reference_levels[bond])
            or target >= len(reference_levels[bond])
        ):
            continue
        coefficient = (
            dt ** number_of_threes
            * factorial(order - number_of_threes)
            / factorial(order)
            if number_of_threes <= order
            else 0.0
        )
        H._remove_history_column(
            reference_arrays,
            reference_levels,
            bond,
            source,
            target,
            coefficient,
        )

    H._algorithm_four(
        fused_arrays,
        fused_levels,
        order,
        dt,
        cache_history=False,
    )
    assert fused_levels == reference_levels
    for fused_array, reference_array in zip(fused_arrays, reference_arrays):
        np.testing.assert_allclose(fused_array, reference_array)


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
    block_sparse = H.extensive_exponential(
        0.01,
        order=3,
        mode="optimal",
        cache_history=False,
        history_storage="block_sparse",
    )

    expected = dense.to_mpo().to_dense()
    np.testing.assert_allclose(streaming.to_mpo().to_dense(), expected)
    np.testing.assert_allclose(sparse.to_mpo().to_dense(), expected)
    np.testing.assert_allclose(block_sparse.to_mpo().to_dense(), expected)
    assert streaming.metadata["history_storage"] == "streaming"
    assert sparse.metadata["history_storage"] == "sparse"
    assert block_sparse.metadata["history_storage"] == "block_sparse"
    assert block_sparse.is_block_sparse
    assert all(count > 0 for count in block_sparse.sparse_block_counts)
    assert sparse.metadata["history_storage_blocks"]["stored_blocks"] < (
        sparse.metadata["history_storage_blocks"]["total_blocks"]
    )
    block_info = block_sparse.metadata["history_storage_blocks"]
    assert block_info["stored_blocks"] < block_info["total_blocks"]
    assert block_info["materialized_dense_virtual_tensors"] is False
    assert streaming.history_cache_info["orders"] == ()
    assert sparse.history_cache_info["orders"] == ()
    assert block_sparse.history_cache_info["orders"] == ()


def test_block_sparse_history_preserves_torch_autograd():
    """Sparse virtual transforms keep backend blocks in the autodiff graph."""
    torch = pytest.importorskip("torch")
    x, _, z = _paulis()
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    step = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    hamiltonian = FirstDegreeMPO.from_local_terms(
        3,
        [
            MPOProductTerm((0, 1), (x, x), coefficient=coefficient),
            MPOProductTerm((1, 2), (z, z)),
        ],
    )

    exponential = hamiltonian.extensive_exponential(
        step,
        order=2,
        mode="approximate",
        history_storage="block_sparse",
    )
    loss = sum(array.real.sum() for array in exponential.arrays)
    coefficient_gradient, step_gradient = torch.autograd.grad(
        loss,
        (coefficient, step),
    )

    assert exponential.is_block_sparse
    assert torch.isfinite(coefficient_gradient)
    assert torch.isfinite(step_gradient)


@pytest.mark.parametrize(
    ("symmetry", "physical_charges", "forward_charge", "backward_charge"),
    [
        ("U1", (0, 1), -1, 1),
        ("Z2", (0, 1), 1, 1),
        ("U1U1", ((0, 0), (1, 0)), (-1, 0), (1, 0)),
        ("Z2Z2", ((0, 0), (1, 0)), (1, 0), (1, 0)),
    ],
)
def test_higher_order_block_sparse_symmetry_matches_dense_mpo(
    symmetry,
    physical_charges,
    forward_charge,
    backward_charge,
):
    """Algorithms 1--4 compile directly to native Abelian charge blocks."""
    pytest.importorskip("symmray")
    raising = np.array([[0.0, 0.0], [1.0, 0.0]])
    lowering = raising.T
    diagonal = np.diag([-0.2, 0.2])
    terms = []
    for left_site in (0, 1):
        terms.extend([
            MPOProductTerm(
                (left_site, left_site + 1),
                (raising, lowering),
                charge=forward_charge,
            ),
            MPOProductTerm(
                (left_site, left_site + 1),
                (lowering, raising),
                charge=backward_charge,
            ),
        ])
    terms.extend(
        MPOProductTerm((site,), (diagonal,))
        for site in range(3)
    )
    symmetric_h = FirstDegreeMPO.from_local_terms(
        3,
        terms,
        symmetry=symmetry,
        physical_charges=physical_charges,
    )
    dense_h = FirstDegreeMPO.from_local_terms(3, terms)

    symmetric_u = symmetric_h.extensive_exponential(
        -0.01j,
        order=2,
        mode="approximate",
    )
    dense_u = dense_h.extensive_exponential(
        -0.01j,
        order=2,
        mode="approximate",
        history_storage="dense",
    )
    compiled = symmetric_u.to_mpo()

    assert symmetric_u.is_block_sparse
    assert symmetric_u.metadata["history_storage"] == "block_sparse"
    assert all(
        type(tensor.data).__name__ == f"{symmetry}Array"
        for tensor in compiled.tensors
    )
    storage = symmetric_u.metadata["history_storage_blocks"]
    assert storage["stored_blocks"] < storage["total_blocks"]
    np.testing.assert_allclose(
        _symmray_mpo_to_dense(compiled),
        dense_u.to_mpo().to_dense(),
    )
    np.testing.assert_allclose(
        _symmray_mpo_to_dense(symmetric_h.to_mpo()),
        dense_h.to_mpo().to_dense(),
    )


def test_higher_order_symmetry_rejects_incorrect_virtual_charge():
    """A nonzero block cannot be silently placed in the wrong charge sector."""
    pytest.importorskip("symmray")
    raising = np.array([[0.0, 0.0], [1.0, 0.0]])
    lowering = raising.T
    hamiltonian = FirstDegreeMPO.from_local_terms(
        2,
        [MPOProductTerm((0, 1), (raising, lowering), charge=1)],
        symmetry="U1",
        physical_charges=(0, 1),
    )

    with pytest.raises(ValueError, match="violates the configured U1 charge flow"):
        hamiltonian.to_mpo()


def test_u1_symmetry_supports_degenerate_physical_sectors():
    """Sector blocks retain every local basis state inside a charge degeneracy."""
    pytest.importorskip("symmray")
    raising = np.zeros((4, 4))
    raising[1, 0] = 1.0
    raising[2, 0] = -0.5
    raising[3, 1] = 0.75
    raising[3, 2] = 0.25
    lowering = raising.T
    terms = [
        MPOProductTerm((0, 1), (raising, lowering), charge=-1),
        MPOProductTerm((0, 1), (lowering, raising), charge=1),
    ]
    symmetric_h = FirstDegreeMPO.from_local_terms(
        2,
        terms,
        symmetry="U1",
        physical_charges=(0, 1, 1, 2),
    )
    dense_h = FirstDegreeMPO.from_local_terms(2, terms)
    symmetric_u = symmetric_h.exp(-0.01j, order=2, mode="base")
    dense_u = dense_h.exp(
        -0.01j,
        order=2,
        mode="base",
        history_storage="dense",
    )

    np.testing.assert_allclose(
        _symmray_mpo_to_dense(symmetric_u.to_mpo()),
        dense_u.to_mpo().to_dense(),
    )


def test_symmetry_requires_contiguous_physical_charge_sectors():
    """Reject a dense basis order that Symmray cannot represent unchanged."""
    with pytest.raises(ValueError, match="group equal charge sectors contiguously"):
        FirstDegreeMPO.from_local_terms(
            1,
            [MPOProductTerm((0,), (np.eye(3),))],
            symmetry="U1",
            physical_charges=(0, 1, 0),
        )


def test_one_site_u1_exponential_compiles_native_rank_two_array():
    """The local Taylor path uses the same symmetric compilation boundary."""
    pytest.importorskip("symmray")
    diagonal = np.diag([-0.3, 0.7])
    hamiltonian = FirstDegreeMPO.from_local_terms(
        1,
        [MPOProductTerm((0,), (diagonal,))],
        symmetry="U1",
        physical_charges=(0, 1),
    )
    exponential = hamiltonian.exp(-0.02j, order=4, mode="optimal")
    compiled = exponential.to_mpo()

    assert type(compiled.tensors[0].data).__name__ == "U1Array"
    np.testing.assert_allclose(
        _symmray_mpo_to_dense(compiled),
        exponential.arrays[0][0, 0],
    )


def test_one_site_u1_zero_operator_retains_native_complex_dtype():
    """Symmray receives a valid zero sector even when every entry vanishes."""
    pytest.importorskip("symmray")
    hamiltonian = FirstDegreeMPO.from_local_terms(
        1,
        [MPOProductTerm((0,), (np.zeros((2, 2), dtype=complex),))],
        symmetry="U1",
        physical_charges=(0, 1),
    )
    compiled = hamiltonian.to_mpo()

    assert type(compiled.tensors[0].data).__name__ == "U1Array"
    assert np.issubdtype(compiled.tensors[0].data.dtype, np.complexfloating)
    np.testing.assert_array_equal(
        _symmray_mpo_to_dense(compiled),
        np.zeros((2, 2)),
    )


def test_mpo_basis_preserves_symmetry_for_parameterized_exponential():
    """Reusable coefficient binding keeps charge metadata and sparse history."""
    pytest.importorskip("symmray")
    raising = np.array([[0.0, 0.0], [1.0, 0.0]])
    lowering = raising.T
    basis = MPOBasis.from_local_terms(
        2,
        [
            MPOProductTerm(
                (0, 1),
                (raising, lowering),
                coefficient=MPOParameter("J"),
                charge=-1,
            ),
            MPOProductTerm(
                (0, 1),
                (lowering, raising),
                coefficient=MPOParameter("J"),
                charge=1,
            ),
        ],
        symmetry="U1",
        physical_charges=(0, 1),
    )

    exponential = basis.exp(
        -0.01j,
        {"J": 0.7},
        order=2,
        mode="base",
    )

    assert exponential.symmetry == "U1"
    assert exponential.physical_charges == (0, 1)
    assert exponential.is_block_sparse
    assert all(
        type(tensor.data).__name__ == "U1Array"
        for tensor in exponential.to_mpo().tensors
    )

    compiled = basis.compile_exp(order=2, mode="base")
    compiled_exponential = compiled.exp(-0.01j, {"J": 0.7})
    assert compiled_exponential.symmetry == "U1"
    assert compiled_exponential.is_block_sparse
    np.testing.assert_allclose(
        _symmray_mpo_to_dense(compiled_exponential.to_mpo()),
        _symmray_mpo_to_dense(exponential.to_mpo()),
    )


def test_higher_order_symmetry_rejects_graded_fermionic_compilation():
    """Bosonic history tensors must not claim sign-correct fermionic output."""
    pytest.importorskip("symmray")
    diagonal = np.diag([0.0, 1.0])
    hamiltonian = FirstDegreeMPO.from_local_terms(
        1,
        [MPOProductTerm((0,), (diagonal,))],
        symmetry="U1",
        physical_charges=(0, 1),
        fermionic=True,
    )

    with pytest.raises(NotImplementedError, match="sign-preserving"):
        hamiltonian.to_mpo()


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

    assert explicit.metadata["mode"] == "optimal"
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
    assert H.history_cache_info["orders"] == ()

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


def test_parameterized_mpo_basis_algorithm_four_supports_jax_autodiff():
    """The fast Algorithm-4 policy keeps JAX values and time differentiable."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    basis = MPOBasis.from_pauli_terms(
        3,
        [((0, 1), "XX"), ((1, 2), "ZZ")],
    )

    def objective(coefficients, time):
        U = basis.exp(
            -1j * time,
            coefficients=coefficients,
            order=2,
            mode="algorithm4",
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
