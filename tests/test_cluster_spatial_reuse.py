"""Observable regressions for conservative MPO/PEPO spatial target reuse."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterLattice,
    MPOBasis,
    MPOClusterFactor,
    MPOClusterProductExpansion,
    MPOParameter,
    MPOProductTerm,
    PauliPEPOBasis,
    PauliPEPOTerm,
    PEPOClusterProductExpansion,
    exp_mpo_cluster_product,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.diag([1.0, -1.0])
SQUARE_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))


def _mpo_factors(*, anisotropic=False):
    terms = [MPOProductTerm((i,), (X,), MPOParameter("h")) for i in range(4)]
    terms += [MPOProductTerm(edge, (Z, Z), MPOParameter("j")) for edge in SQUARE_EDGES]
    if anisotropic:
        terms += [MPOProductTerm((0, 1), (X, Y), 0.23)]
    return (
        MPOClusterFactor(terms),
        MPOClusterFactor([MPOProductTerm((i,), (Z,), -0.4) for i in range(4)]),
    )


@pytest.mark.parametrize("anisotropic", [False, True])
def test_mpo_square_reuse_preserves_ordered_targets_and_residuals(anisotropic):
    graph = ClusterLattice(tuple(range(4)), SQUARE_EDGES)
    options = dict(graph=graph, cluster_size=4, graph_assembly="exact", cutoff=0.0)
    reduced = MPOClusterProductExpansion(4, _mpo_factors(anisotropic=anisotropic), **options)
    direct = MPOClusterProductExpansion(
        4, _mpo_factors(anisotropic=anisotropic), spatial_reuse=False, **options
    )
    assert reduced.cache_info["spatial_plan"]["reused"] > 0
    if not anisotropic:
        # All four edges and all four bent triples share local Hamiltonians.
        assert reduced.cache_info["spatial_plan"]["representatives"] == 4
    for step, params in ((-0.03j, {"h": 0.2, "j": 0.7}), (0.02, {"h": -0.4, "j": 0.1})):
        actual = reduced._residuals(step, params)
        expected = direct._residuals(step, params)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-14)
        dense = reduced.exp(step, parameters=params).to_mpo().to_dense()
        np.testing.assert_allclose(
            dense, direct.exp(step, parameters=params).to_mpo().to_dense(), atol=2e-12
        )
        local = direct._graph_local_exponential((0, 1, 2, 3), step, params)
        np.testing.assert_allclose(dense, local, atol=2e-12)


def test_mpo_independent_equal_parameters_are_not_symmetries():
    terms = [MPOProductTerm((i,), (X,), MPOParameter(f"h{i}")) for i in range(3)]
    basis = MPOClusterProductExpansion.from_local_terms(3, terms, cluster_size=1)
    assert basis.cache_info["spatial_plan"]["reused"] == 0
    # Distinct defaults must also prevent equivalence when parameters=None.
    defaults = [MPOProductTerm((i,), (X,), MPOParameter("h", default=float(i))) for i in range(3)]
    basis = MPOClusterProductExpansion.from_local_terms(3, defaults, cluster_size=1)
    assert basis.cache_info["spatial_plan"]["reused"] == 0


def test_mpo_reuse_torch_gradient_matches_direct_on_repeated_calls():
    torch = pytest.importorskip("torch")
    terms = [MPOProductTerm((i,), (X,), MPOParameter("h")) for i in range(3)]
    terms += [MPOProductTerm((i, i + 1), (Z, Z), 0.7) for i in range(2)]
    reduced = MPOClusterProductExpansion.from_local_terms(3, terms, cluster_size=3)
    direct = MPOClusterProductExpansion.from_local_terms(
        3, terms, cluster_size=3, spatial_reuse=False
    )
    assert reduced.cache_info["spatial_plan"]["reused"] == 3
    for h, t in ((0.2, 0.04), (-0.3, 0.07)):
        parameter = torch.tensor(h, dtype=torch.float64, requires_grad=True)
        time = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        a = reduced._residuals(-1j * time, {"h": parameter})
        b = direct._residuals(-1j * time, {"h": parameter})
        weight = torch.linspace(0.0, 1.0, 64, dtype=torch.float64).reshape(8, 8)
        ga = torch.autograd.grad((a[(0, 2)].real * weight).sum(), (parameter, time))
        gb = torch.autograd.grad((b[(0, 2)].real * weight).sum(), (parameter, time))
        assert torch.allclose(a[(0, 2)], b[(0, 2)], atol=1e-13, rtol=1e-13)
        for actual, expected in zip(ga, gb):
            assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_mpo_spatial_policy_is_forwarded_and_separates_compilation_cache():
    basis = MPOBasis(2, [MPOProductTerm((0,), (X,)), MPOProductTerm((1,), (X,))])
    assert basis.compile_cluster_expansion(
        spatial_reuse=True
    ) is not basis.compile_cluster_expansion(spatial_reuse=False)
    assert basis.compile_graph_cluster_expansion(
        graph="chain", spatial_reuse=True
    ) is not basis.compile_graph_cluster_expansion(graph="chain", spatial_reuse=False)
    with pytest.raises(TypeError, match="spatial_reuse"):
        basis.compile_cluster_expansion(spatial_reuse=1)
    result = exp_mpo_cluster_product([basis], 0.03, spatial_reuse=False)
    np.testing.assert_allclose(
        result.to_dense(), expm(0.03 * (np.kron(X, np.eye(2)) + np.kron(np.eye(2), X))), atol=1e-12
    )


def _pepo_terms(*, located, asymmetric=False):
    if not located:
        return [
            PauliPEPOTerm("onsite", "X", 0.2),
            PauliPEPOTerm("edge", "XY" if asymmetric else "ZZ", 0.7),
        ]
    terms = [PauliPEPOTerm("onsite", "X", 0.2, where=(i, j)) for i in range(2) for j in range(2)]
    sites = ((0, 0), (0, 1), (1, 0), (1, 1))
    terms += [
        PauliPEPOTerm("edge", "XY" if asymmetric else "ZZ", 0.7, where=(sites[a], sites[b]))
        for a, b in SQUARE_EDGES
    ]
    return terms


@pytest.mark.parametrize("located", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_pepo_square_reuse_preserves_full_ordered_operator(located, asymmetric):
    outputs = []
    for reuse in (True, False):
        a = PauliPEPOBasis.compile(
            2, 2, _pepo_terms(located=located, asymmetric=asymmetric), order=4, spatial_reuse=reuse
        )
        b = PauliPEPOBasis.compile(2, 2, [("onsite", "Z", -0.3)], order=4, spatial_reuse=reuse)
        compiled = PEPOClusterProductExpansion.from_bases([a, b]).compile_exp()
        outputs.append(compiled.exp(-0.023j).to_dense())
        if reuse:
            assert any(plan["reused"] for plan in a.cache_info["spatial_plans"])
    np.testing.assert_allclose(*outputs, atol=3e-12)


def test_pepo_localized_binding_modes_preserve_independent_gradients():
    torch = pytest.importorskip("torch")
    terms = [PauliPEPOTerm("onsite", "X", 0.2, where=(0, i)) for i in range(3)]
    terms += [PauliPEPOTerm("edge", "ZZ", 0.7, where=((0, i), (0, i + 1))) for i in range(2)]
    basis = PauliPEPOBasis.compile(1, 3, terms, order=3).compile_exp()
    # Numeric defaults may share targets; vector overrides are independent slots.
    assert any(plan["reused"] for plan in basis.cache_info["spatial_plans"])
    for values in ([0.2, 0.2, 0.2, 0.7, 0.7], [0.1, -0.3, 0.4, 0.2, -0.6]):
        coeff = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        time = torch.tensor(0.021, dtype=torch.float64, requires_grad=True)
        actual = basis.exp(-1j * time, coefficients=coeff, materialize=True).to_dense()
        matrices = []
        for i in range(3):
            local = [np.eye(2)] * 3
            local[i] = X
            matrices.append(np.kron(np.kron(local[0], local[1]), local[2]))
        matrices += [np.kron(np.kron(Z, Z), np.eye(2)), np.kron(np.kron(np.eye(2), Z), Z)]
        h = sum(c * torch.tensor(m, dtype=torch.complex128) for c, m in zip(coeff, matrices))
        expected = torch.matrix_exp(-1j * time * h)
        weight = torch.linspace(0.0, 1.0, 64, dtype=torch.float64).reshape(8, 8)
        ga = torch.autograd.grad((actual.real * weight).sum(), (coeff, time))
        gb = torch.autograd.grad((expected.real * weight).sum(), (coeff, time))
        assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
        for a, b in zip(ga, gb):
            assert torch.allclose(a, b, atol=1e-10, rtol=1e-10)


def test_pepo_periodic_target_reuse_preserves_bond_multiplicity():
    # Use actual PBC occurrences, including both length-two parallel bonds.
    terms = [PauliPEPOTerm("onsite", "X", 0.2, where=(i, j)) for i in range(2) for j in range(2)]
    for i in range(2):
        for j in range(2):
            terms.append(
                PauliPEPOTerm("edge", "XY", 0.7, where=((i, j), ((i + 1) % 2, j)), direction="u")
            )
            terms.append(
                PauliPEPOTerm("edge", "ZZ", -0.3, where=((i, j), (i, (j + 1) % 2)), direction="r")
            )
    reduced = PauliPEPOBasis.compile(2, 2, terms, cyclic=True, order=4)
    direct = PauliPEPOBasis.compile(2, 2, terms, cyclic=True, order=4, spatial_reuse=False)
    a = reduced.exp(-0.013j, materialize=True).to_dense()
    b = direct.exp(-0.013j, materialize=True).to_dense()
    np.testing.assert_allclose(a, b, atol=3e-12)
    assert reduced.cache_info["last_local_targets_evaluated"] < 13


def test_mpo_native_charge_assembly_preserves_spatial_reuse():
    pytest.importorskip("symmray")
    terms = [MPOProductTerm((i,), (Z,), 0.2) for i in range(3)]
    terms += [MPOProductTerm((i, i + 1), (Z, Z), 0.7) for i in range(2)]
    kwargs = dict(cluster_size=3, symmetry="U1", physical_charges=(0, 1))
    reduced = MPOClusterProductExpansion.from_local_terms(3, terms, **kwargs)
    direct = MPOClusterProductExpansion.from_local_terms(3, terms, spatial_reuse=False, **kwargs)
    a = reduced.exp(0.03).to_mpo()
    b = direct.exp(0.03).to_mpo()
    assert all(hasattr(t.data, "blocks") for t in a.tensors)
    np.testing.assert_allclose(a.to_dense(), b.to_dense(), atol=1e-12)


def test_square_snake_reuse_reduces_targets_without_changing_geometry():
    sites = [(i, j) for i in range(5) for j in range(6)]
    edges = [(site, (site[0] + 1, site[1])) for site in sites if site[0] < 4]
    edges += [(site, (site[0], site[1] + 1)) for site in sites if site[1] < 5]
    terms = [(("X", MPOParameter("h")), site) for site in sites]
    terms += [(("ZZ", MPOParameter("j")), edge) for edge in edges]
    basis = MPOBasis.from_terms(terms, shape=(5, 6), map_mode="snake")
    compiled = basis.compile_graph_cluster_expansion(graph="square", cluster_size=4)
    info = compiled.cache_info
    assert info["graph_cluster_count"] == 492
    assert info["spatial_plan"] == {
        "targets": 492,
        "representatives": 6,
        "reused": 486,
        "search_fallbacks": 0,
    }


def test_square_reuse_permutation_preserves_torch_gradients():
    torch = pytest.importorskip("torch")
    graph = ClusterLattice(tuple(range(4)), SQUARE_EDGES)
    bases = [
        MPOClusterProductExpansion(
            4, _mpo_factors(), graph=graph, cluster_size=3, spatial_reuse=reuse
        )
        for reuse in (True, False)
    ]
    assert any(
        axes != tuple(range(len(axes))) for _, axes in bases[0]._spatial_plan.entries.values()
    )
    for value in (0.2, -0.1):
        h = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        time = torch.tensor(0.07, dtype=torch.float64, requires_grad=True)
        residuals = [basis._residuals(-1j * time, {"h": h, "j": 0.7}) for basis in bases]
        gradients = []
        for result in residuals:
            # Include every reused occurrence so shared gradient contributions
            # must accumulate, with a nonuniform probe sensitive to axis order.
            loss = sum(
                (
                    matrix.real
                    * torch.linspace(0.0, 1.0, matrix.numel(), dtype=torch.float64).reshape(
                        matrix.shape
                    )
                ).sum()
                for matrix in result.values()
            )
            gradients.append(torch.autograd.grad(loss, (h, time)))
        for actual, expected in zip(*gradients):
            assert torch.allclose(actual, expected, atol=2e-11, rtol=2e-11)


def test_uniform_pepo_reuse_preserves_torch_coefficient_and_time_gradients():
    torch = pytest.importorskip("torch")
    bases = [
        PauliPEPOBasis.compile(
            2, 2, [("onsite", "X"), ("edge", "XY")], order=4, spatial_reuse=reuse
        ).compile_exp()
        for reuse in (True, False)
    ]
    for values in ([0.2, 0.7], [-0.4, 0.3]):
        coefficients = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        time = torch.tensor(0.031, dtype=torch.float64, requires_grad=True)
        outputs = [
            basis.exp(-1j * time, coefficients=coefficients, materialize=True).to_dense()
            for basis in bases
        ]
        assert torch.allclose(*outputs, atol=2e-12, rtol=2e-12)
        gradients = [
            torch.autograd.grad(matrix.real.sum(), (coefficients, time)) for matrix in outputs
        ]
        for actual, expected in zip(*gradients):
            assert torch.allclose(actual, expected, atol=2e-11, rtol=2e-11)


def test_localized_pepo_reuses_lower_support_contractions(monkeypatch):
    torch = pytest.importorskip("torch")
    from pepsy.operators import pepo_basis

    sites = tuple((i, j) for i in range(2) for j in range(2))
    terms = [
        PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site)
        for site in sites
    ]
    terms += [
        PauliPEPOTerm("edge", "ZZ", 0.7, where=(sites[a], sites[b]))
        for a, b in SQUARE_EDGES
    ]
    original = pepo_basis._contract_active_support_backend
    calls = []

    def count_contract(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pepo_basis, "_contract_active_support_backend", count_contract)
    h = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    outputs = []
    gradients = []
    counts = []
    for reuse in (False, True):
        compiled = PauliPEPOBasis.compile(
            2, 2, terms, order=4, factorization="fixed", spatial_reuse=reuse
        ).compile_exp()
        calls.clear()
        output = compiled(-1j * time, parameters={"h": h}).to_pepo().to_dense()
        outputs.append(output)
        counts.append(len(calls))
        weight = torch.linspace(0.0, 1.0, output.numel(), dtype=torch.float64)
        gradients.append(torch.autograd.grad(
            (output.real.reshape(-1) * weight).sum(), (h, time)
        ))
    assert counts == [9, 3]
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-11, rtol=2e-11)
    for direct, reused in zip(*gradients):
        torch.testing.assert_close(direct, reused, atol=2e-10, rtol=2e-10)
