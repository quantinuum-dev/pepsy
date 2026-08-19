"""Focused tests for the dense PEPO cluster-expansion vertical slice."""

import numpy as np
import pytest
from scipy.linalg import expm

import pepsy
from pepsy.operators import (
    ActivePEPOBlocks,
    ClusterExpansionReport,
    ClusterExpansionPlan,
    CompiledPEPOExp,
    MPOParameter,
    PauliPEPOBasis,
    PauliPEPOTerm,
    build_cluster_expansion_pepo,
    build_itf_cluster_expansion_pepo,
)


def _itf_terms():
    z = np.diag([1.0, -1.0])
    onsite = np.array([[0.0, 0.2], [0.2, 0.0]])
    return np.kron(z, z), onsite


def _three_site_itf_hamiltonian(twosite, onesite):
    identity = np.eye(2)
    return (
        np.kron(np.kron(onesite, identity), identity)
        + np.kron(np.kron(identity, onesite), identity)
        + np.kron(np.kron(identity, identity), onesite)
        + np.kron(twosite, identity)
        + np.kron(identity, twosite)
    )


def _four_site_chain_hamiltonian(twosite, onesite):
    identity = np.eye(2)
    result = np.zeros((16, 16))
    for position in range(4):
        factors = [identity] * 4
        factors[position] = onesite
        result += np.kron(np.kron(np.kron(factors[0], factors[1]), factors[2]), factors[3])
    result += np.kron(twosite, np.eye(4))
    result += np.kron(np.kron(identity, twosite), identity)
    result += np.kron(np.eye(4), twosite)
    return result


def test_cluster_expansion_builder_is_public():
    """The PEPO builder is exposed through both operator namespaces."""
    assert build_cluster_expansion_pepo is pepsy.build_cluster_expansion_pepo
    assert build_cluster_expansion_pepo is pepsy.operators.build_cluster_expansion_pepo
    assert build_itf_cluster_expansion_pepo is pepsy.build_itf_cluster_expansion_pepo
    assert ClusterExpansionPlan is pepsy.ClusterExpansionPlan
    assert ClusterExpansionReport is pepsy.ClusterExpansionReport
    assert ActivePEPOBlocks is pepsy.ActivePEPOBlocks
    assert PauliPEPOBasis is pepsy.PauliPEPOBasis
    assert PauliPEPOTerm is pepsy.PauliPEPOTerm
    assert CompiledPEPOExp is pepsy.CompiledPEPOExp
    assert "build_cluster_expansion_pepo" in pepsy.operators.__all__


def test_order_three_is_exact_on_a_three_site_chain():
    """All connected clusters through size three reproduce the local exponential."""
    twosite, onesite = _itf_terms()
    beta = 0.1
    exact = expm(-beta * _three_site_itf_hamiltonian(twosite, onesite))

    order_two = build_cluster_expansion_pepo(
        1,
        3,
        beta,
        twosite,
        onesite,
        order=2,
    )
    order_three = build_cluster_expansion_pepo(
        1,
        3,
        beta,
        twosite,
        onesite,
        order=3,
    )

    error_two = np.linalg.norm(np.asarray(order_two.to_dense()) - exact)
    error_three = np.linalg.norm(np.asarray(order_three.to_dense()) - exact)
    assert error_two > 1e-8
    assert error_three < 1e-12


def test_cluster_expansion_builds_periodic_pepo_without_snake_embedding():
    """The local PEPO construction supports a genuine square-lattice torus."""
    twosite, onesite = _itf_terms()
    pepo = build_cluster_expansion_pepo(
        3,
        3,
        0.03,
        twosite,
        onesite,
        order=3,
        cyclic=True,
        symmetry="C4",
    )

    assert pepo.Lx == 3
    assert pepo.Ly == 3
    assert all(tensor.shape[:4] == (5, 5, 5, 5) for tensor in pepo)


def test_itf_cluster_expansion_convenience_matches_local_builder():
    """The ITF helper uses the same ZZ-plus-X convention as ``ham_tn``."""
    twosite, onesite = _itf_terms()
    direct = build_cluster_expansion_pepo(2, 3, 0.03, twosite, onesite, order=2)
    helper = build_itf_cluster_expansion_pepo(
        2,
        3,
        0.03,
        J=1.0,
        field=0.2,
        order=2,
    )
    np.testing.assert_allclose(direct.to_dense(), helper.to_dense())


def test_plan_reuses_topology_and_keeps_active_blocks_sparse():
    """Plans cache geometry while materialization remains an explicit step."""
    twosite, onesite = _itf_terms()
    plan = ClusterExpansionPlan(2, 3, twosite, onesite, order=3)
    directions = plan.site_directions
    sparse = plan.build(0.03, materialize=False)

    assert isinstance(sparse, ActivePEPOBlocks)
    assert plan.site_directions is directions
    assert sparse.active_nbytes < sparse.dense_nbytes
    np.testing.assert_allclose(sparse.to_pepo().to_dense(), plan.build(0.03).to_dense())


def test_c4_reduction_solves_two_orbits_and_preserves_dense_result():
    """ITF C4 reduction handles one corner and one straight representative."""
    twosite, onesite = _itf_terms()
    plan = ClusterExpansionPlan(
        2,
        3,
        twosite,
        onesite,
        order=3,
        symmetry="C4",
    )
    assert plan.pair_representatives == (("u", "r"), ("u", "d"))

    reduced = plan.build(0.1)
    unreduced = build_cluster_expansion_pepo(
        2,
        3,
        0.1,
        twosite,
        onesite,
        order=3,
    )
    np.testing.assert_allclose(reduced.to_dense(), unreduced.to_dense(), atol=1e-11)


def test_cluster_expansion_rejects_unimplemented_higher_order():
    """Orders beyond the tree implementation fail explicitly."""
    twosite, onesite = _itf_terms()
    with pytest.raises(NotImplementedError, match="orders 1 through 4"):
        build_cluster_expansion_pepo(2, 2, 0.1, twosite, onesite, order=5)


def test_order_four_path_is_exact_and_reports_local_residuals():
    """The four-site path tree closes the finite-chain residual."""
    twosite, onesite = _itf_terms()
    beta = 0.03
    active, report = build_cluster_expansion_pepo(
        1,
        4,
        beta,
        twosite,
        onesite,
        order=4,
        materialize=False,
        return_report=True,
    )

    exact = expm(-beta * _four_site_chain_hamiltonian(twosite, onesite))
    order_three = build_cluster_expansion_pepo(
        1, 4, beta, twosite, onesite, order=3
    ).to_dense()
    order_four = active.to_pepo().to_dense()

    assert isinstance(report, ClusterExpansionReport)
    assert report.order == 4
    assert report.cluster_counts["four_site_path"] == 1
    assert report.residual_norms["four_site_path"] < 1e-12
    assert np.linalg.norm(order_four - exact) < 1e-11
    assert np.linalg.norm(order_four - exact) < np.linalg.norm(order_three - exact)
    assert active.active_nbytes < active.dense_nbytes


def test_order_four_includes_embedded_star_trees():
    """A rectangular lattice exposes the degree-three tree sector."""
    twosite, onesite = _itf_terms()
    active, report = ClusterExpansionPlan(
        2,
        3,
        twosite,
        onesite,
        order=4,
    ).build(0.03, materialize=False, return_report=True)

    assert report.cluster_counts["four_site_star"] > 0
    assert report.residual_norms["four_site_star"] < 1e-10
    assert active.active_block_count > 0


def test_order_four_includes_and_closes_a_four_site_plaquette_loop():
    """The explicit loop sector closes the smallest square cluster."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    def kron_all(factors):
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    hamiltonian = 0.2 * sum(
        (
            kron_all([x if site == position else identity for site in range(4)])
            for position in range(4)
        ),
        start=np.zeros((16, 16)),
    )
    for first, second in ((0, 1), (0, 2), (2, 3), (1, 3)):
        factors = [identity] * 4
        factors[first] = z
        factors[second] = z
        hamiltonian += kron_all(factors)

    plan = ClusterExpansionPlan(
        2,
        2,
        np.kron(z, z),
        0.2 * x,
        order=4,
    )
    active, report = plan.build(0.01, materialize=False, return_report=True)
    exact = expm(-0.01 * hamiltonian)

    assert report.cluster_counts["four_site_loop"] == 1
    assert report.loop_rank == 16
    assert report.residual_norms["four_site_loop"] > 0.0
    np.testing.assert_allclose(active.to_pepo().to_dense(), exact, atol=1e-11)


def test_order_four_c4_periodic_plan_handles_rotated_path_orbits():
    """Periodic C4 paths build without conflating reversal with rotation."""
    twosite, onesite = _itf_terms()
    active, report = ClusterExpansionPlan(
        3,
        3,
        twosite,
        onesite,
        order=4,
        cyclic=True,
        symmetry="C4",
    ).build(0.03, materialize=False, return_report=True)

    assert report.cluster_counts["four_site_path"] > 0
    assert active.active_block_count > 0
    assert active.active_nbytes < active.dense_nbytes


def _pauli_chain_hamiltonian(nsites, onsite_coefficient, edge_coefficient):
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)

    def kron_all(factors):
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    result = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for site in range(nsites):
        factors = [identity] * nsites
        factors[site] = x
        result += onsite_coefficient * kron_all(factors)
    for site in range(nsites - 1):
        factors = [identity] * nsites
        factors[site] = z
        factors[site + 1] = z
        result += edge_coefficient * kron_all(factors)
    return result


@pytest.mark.parametrize("order, nsites", ((2, 2), (3, 3), (4, 4)))
def test_pauli_pepo_basis_matches_finite_chain_through_tree_order(order, nsites):
    """Fixed Pauli channels close every finite chain tree through order four."""
    basis = PauliPEPOBasis.compile(
        1,
        nsites,
        [
            PauliPEPOTerm("onsite", "X", coefficient=0.2),
            PauliPEPOTerm("edge", "ZZ", coefficient=1.0),
        ],
        order=order,
    )
    pepo = basis.exp(step=-1j * 0.01, materialize=True)
    exact = expm(-1j * 0.01 * _pauli_chain_hamiltonian(nsites, 0.2, 1.0))
    np.testing.assert_allclose(pepo.to_dense(), exact, atol=1e-11)


def test_pauli_pepo_basis_keeps_torch_coefficient_and_time_graph():
    """Coefficient slots and real-time steps remain differentiable."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=2,
    )
    coefficients = torch.tensor(
        [0.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.evaluate(
        tau=tau,
        coefficients=coefficients,
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficients, tau),
    )
    assert any(
        getattr(block, "requires_grad", False)
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.isfinite(time_gradient)


def test_pauli_pepo_basis_resolves_mpo_parameter_references_with_autodiff():
    """PEPO coefficient references follow the MPOBasis parameter contract."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [
            PauliPEPOTerm("onsite", "X", coefficient=MPOParameter("h")),
            PauliPEPOTerm("edge", "ZZ", coefficient=MPOParameter("J")),
        ],
        order=2,
    )
    h = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    j = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.compile_exp().exp(
        -1j * tau,
        parameters={"h": h, "J": j},
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    gradients = torch.autograd.grad(loss, (h, j, tau))
    assert all(torch.isfinite(gradient) for gradient in gradients)


def test_pauli_pepo_c4_rotation_preserves_torch_autodiff():
    """C4 active-axis rotations stay inside the backend autodiff graph."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=3,
        cyclic=True,
        symmetry="C4",
    )
    coefficients = torch.tensor(
        [0.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.compile_exp().exp(
        -1j * tau,
        coefficients=coefficients,
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficients, tau),
    )
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.isfinite(time_gradient)


def test_pauli_pepo_plaquette_loop_preserves_torch_autodiff():
    """The fixed-rank loop channel keeps coefficient graphs intact."""
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=4,
    )
    coefficients = torch.tensor(
        [0.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    active = basis.compile_exp().exp(
        -1j * tau,
        coefficients=coefficients,
        materialize=False,
    )
    loss = sum(
        block.real.sum()
        for site_blocks in active.blocks.values()
        for block in site_blocks.values()
    )
    coefficient_gradient, time_gradient = torch.autograd.grad(
        loss,
        (coefficients, tau),
    )
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.isfinite(time_gradient)


def test_pauli_pepo_basis_reuses_topology_not_backend_values():
    """Repeated evaluations update slots while retaining compiled geometry."""
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [("onsite", "X"), {"support": "edge", "paulis": "ZZ"}],
        order=2,
    )
    first = basis.evaluate(tau=0.01, coefficients=np.array([0.2, 1.0]))
    second = basis.evaluate(tau=0.02, coefficients=np.array([0.3, 0.7]))
    assert basis.cache_info["builds"] == 2
    assert first.bond_dim == second.bond_dim
    assert first.blocks is not second.blocks


def test_active_pepo_blocks_compact_orphans_and_materialize_native_symmray():
    """Diagonal PEPOs compact inactive channels into native charge blocks."""
    pytest.importorskip("symmray")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "Z", 0.2), ("edge", "ZZ")],
        order=4,
    )
    active = basis.exp(0.01, materialize=False)
    compact = active.compact()
    assert compact.bond_dim < active.bond_dim
    native = compact.to_symmray_pepo(symmetry="U1")
    assert all(
        hasattr(native[i, j].data, "blocks")
        for i in range(2)
        for j in range(2)
    )

    dense = compact.to_pepo()
    output_inds = dense.outer_inds()
    native_tensor = native.contract(output_inds=output_inds)
    dense_tensor = dense.contract(output_inds=output_inds)
    np.testing.assert_allclose(
        native_tensor.data.to_dense(),
        dense_tensor.data,
        atol=1e-12,
    )


def test_native_symmray_conversion_rejects_mixed_operator_charge():
    """A homogeneous native charge cannot hide a symmetry-breaking slot."""
    pytest.importorskip("symmray")
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X", 0.2), ("edge", "ZZ")],
        order=2,
    )
    with pytest.raises(ValueError, match="not compatible with Z2 charge"):
        basis.exp(0.01, materialize=False).to_symmray_pepo(symmetry="Z2")


def test_pauli_pepo_compile_exp_is_the_preferred_cached_interface():
    """The PEPO cache uses ``exp`` without an evolution-specific name."""
    basis = PauliPEPOBasis.compile(
        1,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=2,
    )
    compiled = basis.compile_exp()
    assert isinstance(compiled, CompiledPEPOExp)
    assert compiled is basis.compile_exp()
    first = compiled.exp(-1j * 0.01, coefficients=np.array([0.2, 1.0]))
    second = compiled(-1j * 0.02, coefficients=np.array([0.3, 0.7]))
    assert first.bond_dim == second.bond_dim
    assert basis.cache_info["compiled_exp"]


def test_pauli_pepo_compile_exp_prepares_value_independent_cluster_plans():
    """Compiled PEPO exponentials cache geometry but not backend values."""
    basis = PauliPEPOBasis.compile(
        4,
        4,
        [("onsite", "X"), ("edge", "ZZ")],
        order=4,
        symmetry="C4",
    )
    assert basis.cache_info["cluster_embedding_plans"] == 0
    compiled = basis.compile_exp()
    first_plan_count = basis.cache_info["cluster_embedding_plans"]
    assert first_plan_count > 0

    first = compiled.exp(-1j * 0.01, coefficients=np.array([0.2, 1.0]))
    second = compiled.exp(-1j * 0.02, coefficients=np.array([0.3, 0.7]))

    assert basis.cache_info["cluster_embedding_plans"] == first_plan_count
    assert first.blocks is not second.blocks
