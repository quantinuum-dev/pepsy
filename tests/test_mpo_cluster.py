"""Tests for the finite MPO cluster-basis expansion."""

import warnings

import numpy as np
import pytest

from pepsy.operators import (
    ClusterLattice,
    ClusterBasisExpansion,
    CompiledMPOClusterExp,
    MPOBasis,
    MPOClusterBasisExpansion,
    MPOClusterFactor,
    MPOGraphClusterBasisExpansion,
    MPOParameter,
    MPOProductTerm,
    exp_mpo_cluster,
)


scipy_linalg = pytest.importorskip("scipy.linalg")


def _paulis():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    return x, z


def _kron_all(values):
    result = values[0]
    for value in values[1:]:
        result = np.kron(result, value)
    return result


def test_cluster_basis_is_exact_when_cluster_covers_the_chain():
    """The connected residual hierarchy reproduces a complete local product."""
    x, z = _paulis()
    terms = [((0, 1), (x, x)), ((1, 2), (z, z))]
    basis = MPOClusterBasisExpansion.from_local_terms(3, terms, cluster_size=3)
    result = basis.exp(0.03).to_mpo().to_dense()
    hamiltonian = _kron_all((x, x, np.eye(2))) + _kron_all((np.eye(2), z, z))
    np.testing.assert_allclose(result, scipy_linalg.expm(0.03 * hamiltonian), atol=1e-12)


def test_cluster_cutoff_is_size_extensive_and_improves_with_cluster_size():
    """Adding the connected three-site residual improves a four-site chain."""
    x, z = _paulis()
    terms = [
        ((0, 1), (x, x)),
        ((1, 2), (z, z)),
        ((2, 3), (x, x)),
    ]
    # Keep the reference construction explicit so the test does not depend on
    # the cluster implementation's residual bookkeeping.
    hamiltonian = np.zeros((16, 16))
    for support, operators in terms:
        factors = [np.eye(2)] * 4
        for site, operator in zip(support, operators):
            factors[site] = operator
        hamiltonian += _kron_all(factors)
    reference = scipy_linalg.expm(0.08 * hamiltonian)
    order_two = ClusterBasisExpansion.from_local_terms(4, terms, cluster_size=2).exp(0.08)
    order_three = ClusterBasisExpansion.from_local_terms(4, terms, cluster_size=3).exp(0.08)
    error_two = np.linalg.norm(order_two.to_mpo().to_dense() - reference)
    error_three = np.linalg.norm(order_three.to_mpo().to_dense() - reference)
    assert error_three < error_two
    assert order_two.metadata["cluster_report"].initial_bond_dimensions


def test_cluster_basis_supports_ordered_exponential_factors():
    """The factor list represents ``exp(A) @ exp(B)`` in list order."""
    x, z = _paulis()
    factor_a = MPOClusterFactor([((0, 1), (x, x))], coefficient=0.2)
    factor_b = MPOClusterFactor([((1, 2), (z, z))], coefficient=-0.3)
    basis = MPOClusterBasisExpansion.from_factors(
        3,
        [factor_a, factor_b],
        cluster_size=3,
    )
    result = basis.exp(0.05).to_mpo().to_dense()
    a = 0.01 * _kron_all((x, x, np.eye(2)))
    b = -0.015 * _kron_all((np.eye(2), z, z))
    np.testing.assert_allclose(result, scipy_linalg.expm(a) @ scipy_linalg.expm(b), atol=1e-12)


def test_three_ordered_factors_have_a_reusable_compiled_surface():
    """A compiled topology evaluates ``exp(A) exp(B) exp(C)`` exactly."""
    x, z = _paulis()
    factors = [
        MPOClusterFactor([((0, 1), (x, x))], coefficient=0.2),
        MPOClusterFactor([((1, 2), (z, z))], coefficient=-0.3),
        MPOClusterFactor([((0, 1), (z, x))], coefficient=0.4),
    ]
    expansion = MPOClusterBasisExpansion.from_factors(
        3,
        factors,
        cluster_size=3,
        cutoff=0.0,
    )
    compiled = expansion.compile_exp()
    assert isinstance(compiled, CompiledMPOClusterExp)

    result = compiled(0.05).to_mpo().to_dense()
    a = 0.01 * _kron_all((x, x, np.eye(2)))
    b = -0.015 * _kron_all((np.eye(2), z, z))
    c = 0.02 * _kron_all((z, x, np.eye(2)))
    expected = scipy_linalg.expm(a) @ scipy_linalg.expm(b) @ scipy_linalg.expm(c)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert compiled.cache_info["builds"] == 1
    assert compiled.cache_info["static_matrix_count"] > 0

    compiled.exp(0.02)
    assert compiled.cache_info["builds"] == 2


def test_ordered_factors_can_be_built_from_mpo_bases_with_bond_control():
    """Reusable MPO bases preserve factor order and expose rank diagnostics."""
    x, z = _paulis()
    bases = (
        MPOBasis.from_local_terms(3, [((0, 1), (x, x))]),
        MPOBasis.from_local_terms(3, [((1, 2), (z, z))]),
        MPOBasis.from_local_terms(3, [((0, 1), (z, x))]),
    )
    expansion = MPOClusterBasisExpansion.from_mpo_bases(
        bases,
        coefficients=(0.2, -0.3, 0.4),
        cluster_size=3,
        cutoff=0.0,
        max_bond=1,
    )
    result = expansion.exp(0.05)
    report = expansion.last_report
    assert report.max_bond == 1
    assert report.local_svd_truncated
    assert all(
        all(rank <= 1 for rank in ranks)
        for _interval, ranks in report.residual_ranks
    )
    assert result.metadata["history_valid"] is False


def test_mpo_basis_reuses_cluster_topology_without_caching_values():
    """The optimization-facing MPO basis caches plans, not evaluated MPOs."""
    x, z = _paulis()
    basis = MPOBasis.from_local_terms(
        3,
        [((0, 1), (x, x), MPOParameter("J")), ((1, 2), (z, z), MPOParameter("K"))],
    )
    compiled = basis.compile_cluster_expansion(cluster_size=3, cutoff=0.0)
    assert basis.cache_info["compiled_cluster_variants"] == 1
    first = compiled.exp(0.01, parameters={"J": 0.7, "K": -0.2})
    second = compiled.exp(0.02, parameters={"J": -0.4, "K": 0.3})
    assert first is not second
    assert compiled.cache_info["builds"] == 2
    basis.clear_cluster_expansion_cache()
    assert basis.cache_info["compiled_cluster_variants"] == 0


def test_ordered_mpo_bases_require_matching_geometry():
    """The ordered-basis constructor rejects silently misaligned factors."""
    x, _z = _paulis()
    first = MPOBasis.from_local_terms(3, [((0, 1), (x, x))])
    second = MPOBasis.from_local_terms(2, [((0, 1), (x, x))])
    with pytest.raises(ValueError, match="chain lengths"):
        MPOClusterBasisExpansion.from_mpo_bases((first, second))


def test_mpo_basis_cluster_convenience_resolves_parameters():
    """The existing parameter basis can use the local cluster engine."""
    x, z = _paulis()
    basis = MPOBasis.from_local_terms(
        3,
        [
            ((0, 1), (x, x), MPOParameter("J")),
            ((1, 2), (z, z), MPOParameter("K")),
        ],
    )
    result = basis.cluster_expansion(
        0.02,
        {"J": 0.7, "K": -0.2},
        cluster_size=3,
    )
    hamiltonian = 0.7 * _kron_all((x, x, np.eye(2))) - 0.2 * _kron_all((np.eye(2), z, z))
    expected = scipy_linalg.expm(0.02 * hamiltonian)
    np.testing.assert_allclose(result.to_mpo().to_dense(), expected, atol=1e-12)

def test_graph_cluster_expansion_keeps_a_long_range_two_site_cluster():
    """A distant edge carries the skipped sites' singleton background."""
    x, z = _paulis()
    basis = MPOBasis.from_local_terms(
        4,
        [((0, 3), (x, x), 0.7), ((1,), (z,), 0.4)],
    )
    graph = ClusterLattice.from_edges(range(4), [(0, 3)], name="long-range")
    compiled = basis.compile_graph_cluster_expansion(
        graph=graph,
        cluster_size=2,
        cutoff=0.0,
    )
    assert isinstance(compiled.basis, MPOGraphClusterBasisExpansion)
    result = compiled.exp(0.07).to_mpo().to_dense()
    generator = (
        0.7 * _kron_all((x, np.eye(2), np.eye(2), x))
        + 0.4 * _kron_all((np.eye(2), z, np.eye(2), np.eye(2)))
    )
    expected = scipy_linalg.expm(0.07 * generator)
    np.testing.assert_allclose(result, expected, atol=1e-11)
    assert compiled.basis.last_report.cluster_mode == "graph"
    assert compiled.basis.last_report.graph_cluster_count == 5


def test_graph_cluster_expansion_maps_square_coordinates_and_preserves_factor_order():
    """Coordinate graphs support long-range edges and ordered local factors."""
    x, z = _paulis()
    basis_a = MPOBasis.from_square_lattice(
        2,
        2,
        [{"locations": ((0, 0), (1, 1)), "paulis": "XX"}],
    )
    basis_b = MPOBasis.from_square_lattice(
        2,
        2,
        [{"locations": ((0, 0), (1, 1)), "paulis": "ZZ"}],
    )
    inferred = basis_a.compile_graph_cluster_expansion(
        cluster_size=2,
        cutoff=0.0,
    )
    inferred.exp(0.01)
    assert inferred.cache_info["graph_edge_count"] == 5
    graph = ClusterLattice.from_edges(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        [
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((1, 0), (1, 1)),
            ((0, 0), (1, 1)),
        ],
    )
    expansion = MPOClusterBasisExpansion.from_mpo_bases(
        (basis_a, basis_b),
        coefficients=(0.2, -0.3),
        graph=graph,
        cluster_size=2,
        cutoff=0.0,
    )
    result = expansion.exp(0.05).to_mpo().to_dense()
    mapping = basis_a.lattice_to_chain
    xx = _kron_all((x, np.eye(2), x, np.eye(2)))
    zz = _kron_all((z, np.eye(2), z, np.eye(2)))
    expected = scipy_linalg.expm(0.01 * xx) @ scipy_linalg.expm(-0.015 * zz)
    np.testing.assert_allclose(result, expected, atol=1e-11)
    assert mapping[(1, 1)] == 2
    assert expansion.last_report.cluster_mode == "graph"


def test_graph_cluster_expansion_keeps_products_of_crossing_long_range_clusters():
    """Disjoint long-range clusters may nest in the MPO chain ordering."""
    x, _z = _paulis()
    lattice = ClusterLattice.from_edges(range(4), [(0, 3), (1, 2)])
    basis = MPOClusterBasisExpansion(
        4,
        [
            MPOClusterFactor(
                [
                    ((0, 3), (x, x), 0.7),
                    ((1, 2), (x, x), -0.4),
                ]
            )
        ],
        graph=lattice,
        cluster_size=2,
        cutoff=0.0,
    )
    result = basis.exp(0.05).to_mpo().to_dense()
    first = 0.035 * _kron_all((x, np.eye(2), np.eye(2), x))
    second = -0.02 * _kron_all((np.eye(2), x, x, np.eye(2)))
    expected = scipy_linalg.expm(first) @ scipy_linalg.expm(second)
    np.testing.assert_allclose(result, expected, atol=1e-11)


def test_cluster_schmidt_path_keeps_torch_gradients_finite():
    """Exact zero residual channels do not send SVD gradients through a zero."""
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    expansion = MPOClusterBasisExpansion.from_local_terms(
        3,
        [MPOProductTerm((0, 1), (x, x), coefficient)],
        cluster_size=2,
    )
    value = sum(array.real.sum() for array in expansion.exp(time).arrays)
    coefficient_gradient, time_gradient = torch.autograd.grad(
        value,
        (coefficient, time),
    )
    assert torch.isfinite(coefficient_gradient)
    assert torch.isfinite(time_gradient)


def test_three_ordered_factors_keep_all_torch_autodiff_graphs_finite():
    """Factor coefficients and the common step remain differentiable."""
    torch = pytest.importorskip("torch")
    x, z = _paulis()
    coefficients = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (0.2, -0.3, 0.4)
    )
    time = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    expansion = MPOClusterBasisExpansion.from_mpo_bases(
        (
            MPOBasis.from_local_terms(3, [((0, 1), (x, x))]),
            MPOBasis.from_local_terms(3, [((1, 2), (z, z))]),
            MPOBasis.from_local_terms(3, [((0, 1), (z, x))]),
        ),
        coefficients=coefficients,
        cluster_size=3,
    )
    arrays = expansion.compile_exp().exp(time).arrays
    loss = sum(array.square().sum() for array in arrays)
    gradients = torch.autograd.grad(loss, (*coefficients, time))
    assert all(torch.isfinite(gradient) for gradient in gradients)


def test_three_ordered_factors_keep_jax_autodiff_graph_finite():
    """The JAX path uses a static, trace-safe local factorization."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    z = jnp.diag(jnp.array([1.0, -1.0]))

    def loss(values):
        expansion = MPOClusterBasisExpansion.from_mpo_bases(
            (
                MPOBasis.from_local_terms(3, [((0, 1), (x, x))]),
                MPOBasis.from_local_terms(3, [((1, 2), (z, z))]),
                MPOBasis.from_local_terms(3, [((0, 1), (z, x))]),
            ),
            coefficients=values[:3],
            cluster_size=3,
        )
        return sum(jnp.real(array).sum() for array in expansion.exp(values[3]).arrays)

    gradients = jax.grad(loss)(jnp.array([0.2, -0.3, 0.4, 0.01]))
    assert np.all(np.isfinite(np.asarray(gradients)))


def test_exp_mpo_cluster_matches_term_centric_exp_mpo_surface():
    """The cluster facade parses compact terms and returns a Quimb MPO."""
    x, z = _paulis()
    result, report = exp_mpo_cluster(
        [((0, 1), "XX", 1.0), ((1, 2), "ZZ", -0.2)],
        0.03,
        shape=3,
        cluster_size=3,
        cutoff=0.0,
        return_report=True,
    )
    hamiltonian = _kron_all((x, x, np.eye(2))) - 0.2 * _kron_all(
        (np.eye(2), z, z)
    )
    np.testing.assert_allclose(
        result.to_dense(),
        scipy_linalg.expm(0.03 * hamiltonian),
        atol=1.0e-12,
    )
    assert report.cluster_size == 3
    assert result.pepsy_cluster_report is report
    assert result.pepsy_cluster_metadata["cluster_mode"] == "interval"


def test_exp_mpo_cluster_preserves_requested_backend_and_coefficients():
    """The term facade applies ``to_backend`` before local cluster work."""
    torch = pytest.importorskip("torch")
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    step = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    converted = []

    def to_backend(value):
        converted.append(value)
        return torch.as_tensor(value, dtype=torch.float64)

    semantic = exp_mpo_cluster(
        [((0, 1), (x, x), coefficient)],
        step,
        shape=3,
        cluster_size=2,
        cutoff=0.0,
        to_backend=to_backend,
        return_semantic=True,
    )
    assert all(isinstance(array, torch.Tensor) for array in semantic.arrays)
    assert all(
        isinstance(tensor.data, torch.Tensor)
        for tensor in semantic.to_mpo().tensors
    )
    loss = sum(array.real.sum() for array in semantic.arrays)
    gradients = torch.autograd.grad(loss, (coefficient, step))
    assert all(torch.isfinite(gradient) for gradient in gradients)

    exp_mpo_cluster(
        [((0, 1), (x, x), 0.7)],
        0.01,
        shape=3,
        cluster_size=2,
        cutoff=0.0,
        to_backend=to_backend,
        return_semantic=True,
    )
    assert any(np.ndim(value) == 0 for value in converted)


def test_exp_mpo_cluster_maps_coordinate_graphs_and_supports_ordered_factors():
    """The high-level facade maps graph coordinates before MPO assembly."""
    x, z = _paulis()
    graph = ClusterLattice.from_edges(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        [
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((1, 0), (1, 1)),
            ((0, 0), (1, 1)),
        ],
    )
    result = exp_mpo_cluster(
        step=0.05,
        shape=(2, 2),
        graph=graph,
        cluster_size=2,
        cutoff=0.0,
        factors=[
            [
                {
                    "operator": "XX",
                    "location": ((0, 0), (1, 1)),
                    "coefficient": 0.2,
                }
            ],
            [
                {
                    "operator": "ZZ",
                    "location": ((0, 0), (1, 1)),
                    "coefficient": -0.3,
                }
            ],
        ],
    )
    mapping = MPOBasis.from_square_lattice(
        2,
        2,
        [{"locations": ((0, 0), (1, 1)), "paulis": "XX"}],
    ).lattice_to_chain
    xx = _kron_all((x, np.eye(2), x, np.eye(2)))
    zz = _kron_all((z, np.eye(2), z, np.eye(2)))
    expected = scipy_linalg.expm(0.01 * xx) @ scipy_linalg.expm(-0.015 * zz)
    np.testing.assert_allclose(result.to_dense(), expected, atol=1.0e-11)
    assert result.pepsy_cluster_metadata["cluster_mode"] == "graph"
    assert result.pepsy_cluster_metadata["factor_count"] == 2
    assert mapping[(1, 1)] == 2


def test_exp_mpo_cluster_accepts_square_graph_shorthand_and_cyclic_edges():
    """Shape plus a compact graph name is enough for periodic square graphs."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    term = {
        "operator": "XX",
        "location": ((0, 1), (2, 1)),
        "coefficient": 0.3,
    }
    result = exp_mpo_cluster(
        [term],
        0.05,
        shape=(3, 2),
        graph="square",
        cyclic=True,
        cluster_size=2,
        cutoff=0.0,
    )
    mapping = MPOBasis.from_terms([term], shape=(3, 2)).lattice_to_chain
    sites = sorted((mapping[(0, 1)], mapping[(2, 1)]))
    factors = [np.eye(2) for _ in range(6)]
    factors[sites[0]] = x
    factors[sites[1]] = x
    hamiltonian = factors[0]
    for factor in factors[1:]:
        hamiltonian = np.kron(hamiltonian, factor)
    expected = scipy_linalg.expm(0.015 * hamiltonian)
    np.testing.assert_allclose(result.to_dense(), expected, atol=1.0e-12)


def test_interval_cluster_expansion_keeps_explicit_string_operators():
    """Term-centric cluster construction retains fermionic gap operators."""
    x, z = _paulis()
    term = MPOProductTerm((0, 2), (x, x), string_operators=(z,))
    result = exp_mpo_cluster(
        [term],
        0.1,
        shape=3,
        cluster_size=3,
        cutoff=0.0,
    )
    expected = scipy_linalg.expm(0.1 * _kron_all((x, z, x)))
    np.testing.assert_allclose(result.to_dense(), expected, atol=1.0e-12)


def test_auto_graph_assembly_bounds_wide_mpo_collection_plans():
    """Wide graph orderings fall back before materializing all collections."""
    lattice = ClusterLattice.square(3, 3)
    terms = [
        {
            "operator": "ZZ",
            "location": (source, target),
            "coefficient": 0.7,
        }
        for source, target in lattice.edges
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, report = exp_mpo_cluster(
            terms,
            0.01,
            shape=(3, 3),
            graph=lattice,
            cluster_size=2,
            cutoff=0.0,
            return_semantic=True,
            return_report=True,
        )

    assert any(
        "bounded one-cluster approximation" in str(item.message)
        for item in caught
    )
    assert report.graph_assembly == "bounded"
    assert report.graph_collection_order == 1
    assert report.graph_collection_count == 0
    assert report.graph_collection_truncated
    assert report.graph_frontier_width > 0
    assert result.metadata["cluster_report"] is report


def test_exact_graph_assembly_rejects_a_collection_budget_overflow():
    """Exact graph assembly fails before entering an unsafe large plan."""
    z = np.diag([1.0, -1.0])
    lattice = ClusterLattice.square(3, 3)
    terms = [
        ((source, target), (z, z), 0.7)
        for source, target in lattice.edges
    ]

    with pytest.raises(ValueError, match="exceeds collection_budget"):
        exp_mpo_cluster(
            terms,
            0.01,
            shape=(3, 3),
            graph=lattice,
            cluster_size=2,
            cutoff=0.0,
            graph_assembly="exact",
            collection_budget=10,
            return_semantic=True,
        )


def test_bounded_graph_assembly_reports_requested_collection_order():
    """The bounded policy exposes its approximation axis in diagnostics."""
    x, _z = _paulis()
    lattice = ClusterLattice.square(2, 2)
    terms = [
        ((source, target), (x, x), 0.7)
        for source, target in lattice.edges
    ]
    result, report = exp_mpo_cluster(
        terms,
        0.01,
        shape=(2, 2),
        graph=lattice,
        cluster_size=2,
        cutoff=0.0,
        graph_assembly="bounded",
        max_collection_order=1,
        return_semantic=True,
        return_report=True,
    )

    assert report.graph_assembly == "bounded"
    assert report.graph_collection_order == 1
    assert report.graph_collection_truncated
    assert result.pepsy_cluster_metadata["selected_graph_assembly"] == "bounded"
