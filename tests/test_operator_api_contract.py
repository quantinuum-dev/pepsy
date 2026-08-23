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
    "MPOClusterBasisExpansion": ".mpo_cluster",
    "MPOGraphClusterBasisExpansion": ".mpo_cluster",
    "CompiledMPOClusterExp": ".mpo_cluster",
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
    "ClusterBasisExpansion": "MPOClusterBasisExpansion",
    "ClusterExpansionBasis": "MPOClusterBasisExpansion",
    "ClusterExpBasis": "MPOClusterBasisExpansion",
    "MPOClusterExpansion": "MPOClusterBasisExpansion",
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
    mpo_cluster = importlib.import_module(".mpo_cluster", operators.__name__)
    pepo_cluster = importlib.import_module(".pepo_cluster", operators.__name__)
    pepo_product = importlib.import_module(".pepo_product", operators.__name__)

    assert "MPOClusterBasisExpansion" not in higher_order.__all__
    assert "PEPOClusterProductExpansion" not in higher_order.__all__
    assert "MPOBasis" not in mpo_cluster.__all__
    assert "PEPOClusterProductExpansion" not in mpo_cluster.__all__
    assert "MPOBasis" not in pepo_cluster.__all__
    assert "MPOClusterBasisExpansion" not in pepo_cluster.__all__

    assert pepo_cluster.PEPOClusterProductExpansion is operators.PEPOClusterProductExpansion
    assert pepo_cluster.PEPOClusterProductExpansion is pepo_product.PEPOClusterProductExpansion
    assert pepo_cluster.CompiledPEPOClusterProduct is pepo_product.CompiledPEPOClusterProduct


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
