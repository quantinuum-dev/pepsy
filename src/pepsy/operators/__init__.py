"""Gate and Hamiltonian operators.

Gate constructors and tensor-network builders are loaded only when one of
their public names is requested.  This keeps the namespace cheap for callers
that only need boundary contraction or sampling utilities.
"""

from importlib import import_module


_GATE_EXPORTS = [
    "gate",
    "gate_mpo_auto_swap",
    "gate_loop_cluster",
    "gate_simple",
    "renorm_gauge",
    "build_pepo_from_gates",
    "build_mpo_from_gates",
    "pauli",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "h",
    "hadamard",
    "cnot",
    "cx",
    "cy",
    "cz",
    "swap",
    "iswap",
    "phase",
    "u1",
    "u2",
    "cphase",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cu2",
    "cu3",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "u3",
    "su4",
    "fsim",
    "fsimg",
]
_SYMBOL_MODULES = {name: ".gates" for name in _GATE_EXPORTS}
_SYMBOL_MODULES["ham_tn"] = ".hamiltonians"
_SYMBOL_MODULES["build_cluster_expansion_pepo"] = ".pepo_cluster"
_SYMBOL_MODULES["build_model_cluster_expansion_pepo"] = ".pepo_cluster"
_SYMBOL_MODULES["build_itf_cluster_expansion_pepo"] = ".pepo_cluster"
_SYMBOL_MODULES["build_real_time_cluster_expansion_pepo"] = ".pepo_cluster"
_SYMBOL_MODULES["compose_pepo_layers"] = ".pepo_cluster"
_SYMBOL_MODULES["compose_cluster_expansion_pepo"] = ".pepo_cluster"
_SYMBOL_MODULES["generate_connected_cluster_shapes"] = ".pepo_cluster"
_SYMBOL_MODULES["build_graph_cluster_expansion_pepo"] = ".pepo_cluster"
_CLUSTER_EXPORTS = [
    "ActivePEPOBlocks",
    "GraphActivePEPOBlocks",
    "GraphClusterExpansionPlan",
    "ClusterInternalSymmetry",
    "ClusterLattice",
    "ConnectedClusterShape",
    "GraphConnectedClusterShape",
    "ClusterExpansionReport",
    "ClusterExpansionPlan",
    "ClusterModelAdapter",
    "adapt_cluster_model",
    "PauliPEPOTerm",
    "PauliPEPOBasis",
    "CompiledPEPOExp",
    "PEPOClusterFactor",
    "PEPOClusterProductExpansion",
    "CompiledPEPOClusterProduct",
]
_SYMBOL_MODULES.update({name: ".pepo_cluster" for name in _CLUSTER_EXPORTS})
_AUTOMATON_EXPORTS = ["MPOChannel", "MPOTransition", "MPOAutomaton"]
_SYMBOL_MODULES.update({name: ".mpo_automaton" for name in _AUTOMATON_EXPORTS})
# Canonical higher-order MPO construction API.  Keep compatibility spellings
# in a separate group below so the public facade makes the migration direction
# visible without changing which names resolve.
_MPO_EXPORTS = [
    "MPOParameter",
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOLocalOperatorTerm",
    "MPOBraiding",
    "MPOPhysicalSpace",
    "MPOCompressionReport",
    "MPONumericalCompressionReport",
    "MPODifferentiableCompressionReport",
    "MPOAdaptiveCompressionReport",
    "MPOChargeValidationReport",
    "MPOBlock",
    "MPOBlockPlan",
    "FirstDegreeMPO",
    "CompiledMPOExp",
    "MPOBasis",
    "exp_mpo",
]
_SYMBOL_MODULES.update({name: ".mpo_higher_order" for name in _MPO_EXPORTS})
_MPO_COMPATIBILITY_EXPORTS = ["CompiledMPOEvolution"]
_SYMBOL_MODULES.update({name: ".mpo" for name in _MPO_COMPATIBILITY_EXPORTS})

# Connected MPO product names are canonical. The old ``BasisExpansion`` and
# ``CompiledMPOClusterExp`` spellings remain explicit compatibility aliases.
_MPO_CLUSTER_EXPORTS = [
    "MPOClusterFactor",
    "MPOClusterExpansionReport",
    "MPOClusterProductExpansion",
    "MPOGraphClusterProductExpansion",
    "CompiledMPOClusterProduct",
    "compress_mpo_product",
    "exp_mpo_cluster",
    "exp_mpo_cluster_product",
]
_SYMBOL_MODULES.update({name: ".mpo_product" for name in _MPO_CLUSTER_EXPORTS})
_MPO_CLUSTER_COMPATIBILITY_EXPORTS = [
    "MPOClusterBasisExpansion",
    "MPOGraphClusterBasisExpansion",
    "CompiledMPOClusterExp",
    "ClusterBasisExpansion",
    "ClusterExpansionBasis",
    "ClusterExpBasis",
    "MPOClusterExpansion",
]
_SYMBOL_MODULES.update(
    {name: ".mpo_cluster" for name in _MPO_CLUSTER_COMPATIBILITY_EXPORTS}
)
_DIAGNOSTIC_EXPORTS = ["OperatorReportInfo"]
_SYMBOL_MODULES.update({name: ".diagnostics" for name in _DIAGNOSTIC_EXPORTS})
_SYMBOL_MODULES["MPOBraiding"] = ".mpo_space"
_SYMBOL_MODULES["MPOPhysicalSpace"] = ".mpo_space"
_SYMBOL_MODULES["PauliMPO"] = ".pauli_mpo"
_SYMBOL_MODULES["decompose_pauli"] = ".pauli_mpo"
_SYMBOL_MODULES["PauliCompressionReport"] = ".pauli_mpo"
_SYMBOL_MODULES["PauliBondCompressionReport"] = ".pauli_mpo"
_SUBMODULES = (
    "cluster",
    "gates",
    "hamiltonians",
    "mpo",
    "mpo_semantic",
    "mpo_block_plan",
    "mpo_higher_order",
    "mpo_automaton",
    "mpo_cluster",
    "mpo_product",
    "pepo_dense",
    "pepo_geometry",
    "pepo_cluster",
    "diagnostics",
    "pauli_mpo",
)

__all__ = [
    *_GATE_EXPORTS,
    "build_cluster_expansion_pepo",
    "build_model_cluster_expansion_pepo",
    "build_itf_cluster_expansion_pepo",
    "build_real_time_cluster_expansion_pepo",
    "compose_pepo_layers",
    "compose_cluster_expansion_pepo",
    "generate_connected_cluster_shapes",
    "build_graph_cluster_expansion_pepo",
    *_CLUSTER_EXPORTS,
    "ham_tn",
    *_AUTOMATON_EXPORTS,
    *_MPO_EXPORTS,
    *_MPO_COMPATIBILITY_EXPORTS,
    *_MPO_CLUSTER_EXPORTS,
    *_MPO_CLUSTER_COMPATIBILITY_EXPORTS,
    *_DIAGNOSTIC_EXPORTS,
    "PauliMPO",
    "decompose_pauli",
    "PauliCompressionReport",
    "PauliBondCompressionReport",
    *_SUBMODULES,
]


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
