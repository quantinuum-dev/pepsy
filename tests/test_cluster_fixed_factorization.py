"""Exact fixed-index cluster construction and verified spatial declarations."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterLattice,
    MPOBasis,
    MPOParameter,
    PauliPEPOBasis,
    PauliPEPOTerm,
    PEPOClusterProductExpansion,
    exp_mpo_cluster,
    exp_mpo_cluster_product,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])


def forbid_svd(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("fixed construction must not invoke SVD")

    import pepsy.operators.mpo_product as mpo
    import pepsy.operators.mpo_semantic as semantic
    import pepsy.operators.pepo_dense as pepo

    monkeypatch.setattr(mpo, "_fixed_rank_svd", fail)
    monkeypatch.setattr(semantic, "_fixed_rank_svd", fail)
    monkeypatch.setattr(pepo, "_fixed_rank_svd", fail)
    monkeypatch.setattr(np.linalg, "svd", fail)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.linalg, "svd", fail)


@pytest.mark.parametrize("assembly", ["direct", "recursive"])
def test_fixed_mpo_zero_crossing_values_gradients_and_shapes(monkeypatch, assembly):
    torch = pytest.importorskip("torch")
    basis = MPOBasis.from_local_terms(
        3, [((0, 1), (Z, Z), MPOParameter("j")), ((1, 2), (X, X), 0.3)]
    )
    options = dict(
        graph="chain", cluster_size=3, cutoff=0.0, factorization="fixed", assembly=assembly
    )
    compiled = basis.compile_graph_cluster_expansion(**options)
    assert compiled is basis.compile_graph_cluster_expansion(**options)
    assert compiled is not basis.compile_graph_cluster_expansion(
        **(options | {"factorization": "auto"})
    )
    forbid_svd(monkeypatch)
    h0 = torch.as_tensor(np.kron(np.kron(Z, Z), np.eye(2)))
    h1 = torch.as_tensor(np.kron(np.eye(2), np.kron(X, X)))
    weights = torch.arange(64, dtype=torch.float64).reshape(8, 8) / 64
    shapes = []
    for coupling, time in ((0.0, 0.1), (0.4, 0.0), (0.4, 0.1)):
        j = torch.tensor(coupling, dtype=torch.float64, requires_grad=True)
        t = torch.tensor(time, dtype=torch.float64, requires_grad=True)
        semantic, report = compiled(t, parameters={"j": j}, return_report=True)
        actual = semantic.to_mpo().to_dense()
        expected = torch.matrix_exp(t * (j * h0 + 0.3 * h1))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        a = torch.autograd.grad((actual * weights).sum(), (j, t), retain_graph=True)
        b = torch.autograd.grad((expected * weights).sum(), (j, t))
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)
        shapes.append(tuple(tuple(core.shape) for core in semantic.arrays))
        assert report.factorization == "fixed"
        assert report.assembly_compression_count == 0
    assert shapes[0] == shapes[1] == shapes[2]


def test_fixed_mpo_preserves_disjoint_collections_and_factor_order(monkeypatch):
    graph = ClusterLattice.from_edges(range(4), ((0, 2), (1, 3)))
    a = [((0, 2), (Z, X), 0.2), ((1, 3), (X, Z), -0.3)]
    b = [((i,), (X,), 0.1) for i in range(4)]
    options = dict(shape=4, graph=graph, cluster_size=2, cutoff=0.0, graph_assembly="exact")
    reference = exp_mpo_cluster_product([a, b], 0.12, **options).to_dense()
    forbid_svd(monkeypatch)
    result = exp_mpo_cluster_product(
        [a, b], 0.12, factorization="fixed", assembly="recursive", **options
    )
    np.testing.assert_allclose(result.to_dense(), reference, atol=1e-12)
    assert result.pepsy_cluster_report.factorization == "fixed"


@pytest.mark.parametrize(
    "shape,order,located", [((2, 2), 4, False), ((1, 3), 3, True), ((1, 5), 5, False)]
)
def test_fixed_pepo_no_svd_dense_values_and_zero_gradients(monkeypatch, shape, order, located):
    torch = pytest.importorskip("torch")
    if located:
        terms = [PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=(0, i)) for i in range(3)]
        terms += [PauliPEPOTerm("edge", "ZZ", 0.3, where=((0, i), (0, i + 1))) for i in range(2)]
    else:
        terms = [PauliPEPOTerm("onsite", "X", MPOParameter("h")), PauliPEPOTerm("edge", "ZZ", 0.3)]
    basis = PauliPEPOBasis(*shape, terms, order=order, factorization="fixed")
    compiled = basis.compile_exp()
    assert compiled is basis.compile_exp()
    assert compiled.cache_info["factorization"] == "fixed"
    forbid_svd(monkeypatch)
    sizes = []
    for value in (0.0, 0.2):
        h = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        t = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
        active = compiled(t, parameters={"h": h})
        actual = active.to_pepo().to_dense()
        # Independent dense Hamiltonian in the PEPO's lexicographic site order.
        sites = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        def embed(operators):
            result = np.ones((1, 1))
            for site in sites:
                result = np.kron(result, operators.get(site, np.eye(2)))
            return torch.as_tensor(result, dtype=torch.complex128)

        onsite = sum(embed({site: X}) for site in sites)
        edges = [
            (site, (site[0] + di, site[1] + dj))
            for site in sites
            for di, dj in ((1, 0), (0, 1))
            if site[0] + di < shape[0] and site[1] + dj < shape[1]
        ]
        generator = h * onsite + 0.3 * sum(embed({a: Z, b: Z}) for a, b in edges)
        expected = torch.matrix_exp(t * generator)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        weights = torch.linspace(0.0, 1.0, actual.numel(), dtype=torch.float64).reshape(
            actual.shape
        )
        a = torch.autograd.grad((actual.real * weights).sum(), (h, t), retain_graph=True)
        b = torch.autograd.grad((expected.real * weights).sum(), (h, t))
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-10)
        sizes.append(
            {
                site: tuple((key, tuple(block.shape)) for key, block in blocks.items())
                for site, blocks in active.blocks.items()
            }
        )
    assert sizes[0] == sizes[1]


def test_fixed_pepo_ordered_product_and_compression_guard(monkeypatch):
    a = PauliPEPOBasis(1, 2, [("edge", "ZZ")], order=2, factorization="fixed")
    b = PauliPEPOBasis(1, 2, [("onsite", "X")], order=2, factorization="fixed")
    product = PEPOClusterProductExpansion.from_bases([a, b], coefficients=[0.2, -0.3])
    compiled = product.compile_exp()
    forbid_svd(monkeypatch)
    actual = compiled(0.1).to_dense()
    expected = expm(0.02 * np.kron(Z, Z)) @ expm(
        -0.03 * (np.kron(X, np.eye(2)) + np.kron(np.eye(2), X))
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    with pytest.raises(ValueError, match="compress=False"):
        compiled(0.1, compress=True)


def test_fixed_policy_rejects_hidden_decompositions():
    terms = [((0, 1), (Z, Z))]
    for options, message in [
        ({"max_bond": 2}, "max_bond"),
        ({"cutoff": 1e-8}, "cutoff"),
        ({"assembly_chi": 4, "assembly": "recursive", "graph": "chain"}, "assembly_chi"),
        ({"chi": 4}, "chi=None"),
    ]:
        with pytest.raises(ValueError, match=message):
            exp_mpo_cluster(
                terms, 0.1, **(dict(shape=2, factorization="fixed", cutoff=0.0) | options)
            )
    # Fixed native sectors are now supported; they must keep the exact target.
    native = exp_mpo_cluster(terms, 0.1, shape=2, factorization='fixed', cutoff=0.,
                            symmetry='U1', physical_charges=(0, 1))
    np.testing.assert_allclose(native.to_dense(), expm(.1*np.kron(Z, Z)), atol=1e-13)
    with pytest.raises(ValueError, match="max_tree_rank"):
        PauliPEPOBasis(1, 2, [("edge", "ZZ")], factorization="fixed", max_tree_rank=2)
    with pytest.raises(ValueError, match="factorization"):
        PauliPEPOBasis(1, 2, [("edge", "ZZ")], factorization="unknown")


def test_declared_mpo_reflection_preserves_parameter_bindings_and_cache():
    terms = [((i,), (X,), MPOParameter("h")) for i in range(3)]
    basis = MPOBasis.from_local_terms(3, terms)
    options = dict(factorization="fixed", cutoff=0.0, spatial_symmetries=[(2, 1, 0)])
    compiled = basis.compile_cluster_expansion(**options)
    assert compiled is basis.compile_cluster_expansion(**options)
    assert compiled.cache_info["declared_spatial_symmetry_count"] == 1
    assert compiled.cache_info["spatial_plan"]["reused"] > 0
    independent = MPOBasis.from_local_terms(
        3, [((i,), (X,), MPOParameter(f"h{i}")) for i in range(3)]
    )
    with pytest.raises(ValueError, match="coefficient bindings"):
        independent.compile_cluster_expansion(**options)
    with pytest.raises(ValueError, match="permutation"):
        basis.compile_cluster_expansion(spatial_symmetries=[(0, 0, 1)])


def test_declared_pepo_rotations_reflections_and_override_safety():
    sites = [(i, j) for i in range(2) for j in range(2)]
    rotation = {site: (site[1], 1 - site[0]) for site in sites}
    reflection = {site: (site[0], 1 - site[1]) for site in sites}
    terms = [PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site) for site in sites]
    options = dict(order=2, factorization="fixed")
    basis = PauliPEPOBasis(2, 2, terms, spatial_symmetries=[rotation, reflection], **options)
    plain = PauliPEPOBasis(2, 2, terms, spatial_reuse=False, **options)
    assert basis.compile_exp().cache_info["declared_spatial_symmetry_count"] == 2
    values = [0.1, 0.2, 0.3, 0.4]
    # Explicit coefficient vectors remain independent, despite default bindings.
    np.testing.assert_allclose(
        basis.exp(0.1, coefficients=values, materialize=True).to_dense(),
        plain.exp(0.1, coefficients=values, materialize=True).to_dense(),
        atol=1e-12,
    )
    asymmetric = terms + [PauliPEPOTerm("edge", "XY", 0.2, where=((0, 0), (0, 1)))]
    with pytest.raises(ValueError, match="Hamiltonian terms"):
        PauliPEPOBasis(2, 2, asymmetric, spatial_symmetries=[rotation], **options)


def test_declared_periodic_translation_and_open_boundary_rejection():
    sites = [(0, j) for j in range(3)]
    shift = {site: (0, (site[1] + 1) % 3) for site in sites}
    terms = [("edge", "ZZ")]
    basis = PauliPEPOBasis(1, 3, terms, order=2, cyclic=(False, True), spatial_symmetries=[shift])
    assert basis.compile_exp().cache_info["declared_spatial_symmetry_count"] == 1
    with pytest.raises(ValueError, match="lattice edges"):
        PauliPEPOBasis(1, 3, terms, order=2, spatial_symmetries=[shift])


def test_fixed_mode_rejects_unverified_c4_transport():
    with pytest.raises(ValueError, match="endpoint reversal"):
        PauliPEPOBasis(2, 2, [("edge", "XY")], symmetry="C4", factorization="fixed")


def test_fixed_split_preserves_device_and_dtype():
    torch = pytest.importorskip("torch")
    from pepsy.operators._cluster_factorization import fixed_split

    for dtype in (torch.float32, torch.complex128):
        for shape in ((4, 16), (16, 4)):
            matrix = torch.empty(shape, dtype=dtype, device="meta")
            a, b = fixed_split(matrix)
            assert a.dtype == b.dtype == dtype
            assert a.device == b.device == matrix.device


def test_fixed_generic_pepo_reuses_prepared_tree_structure(monkeypatch):
    from pepsy.operators import pepo_dense

    pepo_dense._backend_tree_topology.cache_clear()
    basis = PauliPEPOBasis(1, 5, [("onsite", "X"), ("edge", "ZZ")], order=5, factorization="fixed")
    compiled = basis.compile_exp()
    before = pepo_dense._backend_tree_topology.cache_info()
    assert before.currsize > 0
    forbid_svd(monkeypatch)
    compiled(0.01, coefficients=(0.2, 0.3))
    after = pepo_dense._backend_tree_topology.cache_info()
    assert after.misses == before.misses
    assert after.hits > before.hits


def test_fixed_local_jax_jit_keeps_zero_derivative(monkeypatch):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from pepsy.operators.mpo_product import _operator_schmidt

    forbid_svd(monkeypatch)
    matrix = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16

    def evaluate(coefficient):
        cores = _operator_schmidt(coefficient * matrix, 2, 2, 0.0, factorization="fixed")
        return jnp.einsum("aij,akl->ikjl", cores[0][0], cores[1][:, 0]).reshape(4, 4)

    compiled = jax.jit(evaluate)
    np.testing.assert_allclose(compiled(0.2), 0.2 * matrix, atol=1e-7)
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(evaluate(value))))(0.0)
    np.testing.assert_allclose(gradient, matrix.sum(), atol=1e-6)


@pytest.mark.parametrize("ordered", [False, True])
def test_fixed_one_shot_dense_local_operator_has_no_hidden_svd(monkeypatch, ordered):
    from pepsy.operators import MPOLocalOperatorTerm

    torch = pytest.importorskip("torch")
    forbid_svd(monkeypatch)
    operator = torch.tensor(
        np.kron(Z, Z) + 0.2 * np.kron(X, np.eye(2)),
        dtype=torch.float64, requires_grad=True,
    )
    term = MPOLocalOperatorTerm((0, 1), operator)
    options = dict(shape=2, cluster_size=2, cutoff=0.0, factorization="fixed")
    if ordered:
        result = exp_mpo_cluster_product([[term], [((0,), (X,), 0.3)]], 0.1, **options)
        expected = torch.matrix_exp(0.1 * operator) @ torch.matrix_exp(
            torch.as_tensor(0.03 * np.kron(X, np.eye(2)))
        )
    else:
        result = exp_mpo_cluster([term], 0.1, **options)
        expected = torch.matrix_exp(0.1 * operator)
    actual = result.to_dense()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    (a,) = torch.autograd.grad(actual.sum(), (operator,), retain_graph=True)
    (b,) = torch.autograd.grad(expected.sum(), (operator,))
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
