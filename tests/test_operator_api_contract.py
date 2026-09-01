"""Executable contract for the organized operator API tiers."""

import importlib

import pytest

import pepsy.operators as operators


pytestmark = pytest.mark.smoke


_CANONICAL_OWNERS = {
    "MPOBasis": ".mpo_higher_order",
    "MPOParameter": ".mpo_higher_order",
    "MPOProductTerm": ".mpo_higher_order",
    "MPOLocalOperatorTerm": ".mpo_higher_order",
    "FirstDegreeMPO": ".mpo_higher_order",
    "CompiledMPOExp": ".mpo_higher_order",
    "exp_mpo": ".mpo_higher_order",
    "MPOClusterProductExpansion": ".mpo_product",
    "MPOGraphClusterProductExpansion": ".mpo_product",
    "CompiledMPOClusterProduct": ".mpo_product",
    "exp_mpo_cluster": ".mpo_product",
    "ActivePEPOBlocks": ".pepo_cluster",
    "GraphActivePEPOBlocks": ".pepo_cluster",
    "PauliPEPOBasis": ".pepo_cluster",
    "CompiledPEPOExp": ".pepo_cluster",
    "ClusterExpansionPlan": ".pepo_cluster",
    "GraphClusterExpansionPlan": ".pepo_cluster",
    "PEPOClusterProductExpansion": ".pepo_cluster",
    "CompiledPEPOClusterProduct": ".pepo_cluster",
    "PauliMPO": ".pauli_mpo",
    "MPOAutomaton": ".mpo_automaton",
    "OperatorReportInfo": ".diagnostics",
}

_COMPATIBILITY_ALIASES = {
    "CompiledMPOEvolution": "CompiledMPOExp",
    "MPOClusterBasisExpansion": "MPOClusterProductExpansion",
    "MPOGraphClusterBasisExpansion": "MPOGraphClusterProductExpansion",
    "CompiledMPOClusterExp": "CompiledMPOClusterProduct",
    "ClusterBasisExpansion": "MPOClusterProductExpansion",
    "ClusterExpansionBasis": "MPOClusterProductExpansion",
    "ClusterExpBasis": "MPOClusterProductExpansion",
    "MPOClusterExpansion": "MPOClusterProductExpansion",
}


def test_canonical_operator_symbols_have_declared_owners():
    """Canonical names remain available from the public facade and owner."""
    for name, relative_module in _CANONICAL_OWNERS.items():
        assert name in operators.__all__
        owner = importlib.import_module(relative_module, operators.__name__)
        assert getattr(operators, name) is getattr(owner, name)


def test_compatibility_operator_symbols_are_explicit_aliases():
    """Compatibility names point at canonical objects rather than duplicates."""
    for compatibility_name, canonical_name in _COMPATIBILITY_ALIASES.items():
        assert compatibility_name in operators.__all__
        assert getattr(operators, compatibility_name) is getattr(
            operators,
            canonical_name,
        )


def test_canonical_lifecycle_boundaries_are_present():
    """Plans, compiled evaluators, and result materialization stay discoverable."""
    assert callable(operators.MPOBasis.compile_exp)
    assert callable(operators.PauliPEPOBasis.compile_exp)
    assert callable(operators.ClusterExpansionPlan.build)
    assert callable(operators.PEPOClusterProductExpansion.compile_exp)
    assert callable(operators.FirstDegreeMPO.to_mpo)
    assert callable(operators.ActivePEPOBlocks.to_pepo)
    assert callable(operators.GraphActivePEPOBlocks.to_tensor_network)


def test_operator_family_facades_keep_construction_domains_separate():
    """The named modules expose one construction family each."""
    higher_order = importlib.import_module(
        ".mpo_higher_order", operators.__name__
    )
    mpo_basis = importlib.import_module(".mpo_basis", operators.__name__)
    mpo_semantic = importlib.import_module(".mpo_semantic", operators.__name__)
    mpo_cluster = importlib.import_module(".mpo_cluster", operators.__name__)
    mpo_product = importlib.import_module(".mpo_product", operators.__name__)
    pepo_cluster = importlib.import_module(".pepo_cluster", operators.__name__)
    pepo_dense = importlib.import_module(".pepo_dense", operators.__name__)
    pepo_geometry = importlib.import_module(".pepo_geometry", operators.__name__)
    pepo_active = importlib.import_module(".pepo_active", operators.__name__)
    pepo_basis = importlib.import_module(".pepo_basis", operators.__name__)
    pepo_product = importlib.import_module(".pepo_product", operators.__name__)

    assert "MPOClusterProductExpansion" not in higher_order.__all__
    assert "PEPOClusterProductExpansion" not in higher_order.__all__
    assert "MPOBasis" not in mpo_cluster.__all__
    assert "PEPOClusterProductExpansion" not in mpo_cluster.__all__
    assert "MPOBasis" not in mpo_semantic.__all__
    assert "MPOBasis" not in pepo_cluster.__all__
    assert "MPOClusterBasisExpansion" not in pepo_cluster.__all__
    assert "PauliPEPOBasis" not in pepo_dense.__all__

    assert higher_order.MPOBasis is mpo_basis.MPOBasis
    assert higher_order.CompiledMPOExp is mpo_basis.CompiledMPOExp
    assert higher_order.exp_mpo is mpo_basis.exp_mpo
    assert higher_order.FirstDegreeMPO is mpo_semantic.FirstDegreeMPO
    assert operators.FirstDegreeMPO is mpo_semantic.FirstDegreeMPO
    assert operators.MPOClusterProductExpansion is mpo_product.MPOClusterProductExpansion
    assert operators.CompiledMPOClusterProduct is mpo_product.CompiledMPOClusterProduct
    assert operators.MPOClusterBasisExpansion is mpo_product.MPOClusterProductExpansion
    assert operators.CompiledMPOClusterExp is mpo_product.CompiledMPOClusterProduct
    assert operators.MPOGraphClusterBasisExpansion is mpo_product.MPOGraphClusterProductExpansion
    assert mpo_cluster.MPOClusterBasisExpansion is mpo_product.MPOClusterProductExpansion
    assert mpo_cluster.CompiledMPOClusterExp is mpo_product.CompiledMPOClusterProduct
    assert pepo_cluster.ClusterExpansionPlan is pepo_dense.ClusterExpansionPlan
    assert pepo_cluster.ClusterLattice is pepo_dense.ClusterLattice
    assert pepo_geometry.ClusterLattice is pepo_dense.ClusterLattice
    assert pepo_cluster.PEPOClusterProductExpansion is operators.PEPOClusterProductExpansion
    assert pepo_cluster.PEPOClusterProductExpansion is pepo_product.PEPOClusterProductExpansion
    assert pepo_cluster.CompiledPEPOClusterProduct is pepo_product.CompiledPEPOClusterProduct
    assert pepo_cluster.ActivePEPOBlocks is pepo_active.ActivePEPOBlocks
    assert pepo_cluster.GraphActivePEPOBlocks is pepo_active.GraphActivePEPOBlocks
    assert pepo_cluster.PauliPEPOBasis is pepo_basis.PauliPEPOBasis
    assert pepo_cluster.CompiledPEPOExp is pepo_basis.CompiledPEPOExp


def test_report_summary_uses_one_cross_family_vocabulary():
    """Concrete reports expose a stable summary without losing detail."""
    mpo_info = operators.MPONumericalCompressionReport(
        method="quimb",
        form="right",
        max_bond=4,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        initial_bond_dimensions=(2,),
        final_bond_dimensions=(1,),
        truncated=True,
    ).api_info
    pepo_info = operators.ClusterExpansionReport(
        beta=0.01,
        order=3,
        local_dim=2,
        edge_rank=1,
        tree_rank=2,
        loop_rank=0,
        cluster_counts={},
        residual_norms={},
        relative_residual_norms={},
        active_block_count=1,
        active_nbytes=16,
        dense_nbytes=32,
    ).api_info

    assert mpo_info.as_dict() == {
        "family": "mpo",
        "algorithm": "numerical_compression",
        "representation": "materialized_mpo",
        "order": None,
        "factor_count": None,
        "truncated": True,
        "differentiable": None,
    }
    assert pepo_info.family == "pepo"
    assert pepo_info.algorithm == "cluster_expansion"
    assert pepo_info.representation == "active_pepo"
    assert pepo_info.order == 3
