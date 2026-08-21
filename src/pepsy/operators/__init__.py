"""Gate and Hamiltonian operators.

Gate constructors and tensor-network builders are loaded only when one of
their public names is requested.  This keeps the namespace cheap for callers
that only need boundary contraction or sampling utilities.
"""

from importlib import import_module


_GATE_EXPORTS = [
    "gate",
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
_SYMBOL_MODULES["build_cluster_expansion_pepo"] = ".cluster"
_SYMBOL_MODULES["build_model_cluster_expansion_pepo"] = ".cluster"
_SYMBOL_MODULES["build_itf_cluster_expansion_pepo"] = ".cluster"
_SYMBOL_MODULES["build_real_time_cluster_expansion_pepo"] = ".cluster"
_SYMBOL_MODULES["compose_pepo_layers"] = ".cluster"
_SYMBOL_MODULES["compose_cluster_expansion_pepo"] = ".cluster"
_SYMBOL_MODULES["generate_connected_cluster_shapes"] = ".cluster"
_SYMBOL_MODULES["build_graph_cluster_expansion_pepo"] = ".cluster"
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
]
_SYMBOL_MODULES.update({name: ".cluster" for name in _CLUSTER_EXPORTS})
_AUTOMATON_EXPORTS = ["MPOChannel", "MPOTransition", "MPOAutomaton"]
_SYMBOL_MODULES.update({name: ".mpo_automaton" for name in _AUTOMATON_EXPORTS})
_MPO_EXPORTS = [
    "MPOParameter",
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOCompressionReport",
    "MPONumericalCompressionReport",
    "MPODifferentiableCompressionReport",
    "FirstDegreeMPO",
    "CompiledMPOExp",
    "CompiledMPOEvolution",
    "MPOBasis",
]
_SYMBOL_MODULES.update({name: ".mpo" for name in _MPO_EXPORTS})
_SYMBOL_MODULES["PauliMPO"] = ".pauli_mpo"
_SYMBOL_MODULES["decompose_pauli"] = ".pauli_mpo"
_SYMBOL_MODULES["PauliCompressionReport"] = ".pauli_mpo"
_SYMBOL_MODULES["PauliBondCompressionReport"] = ".pauli_mpo"
_SUBMODULES = (
    "cluster",
    "gates",
    "hamiltonians",
    "mpo",
    "mpo_automaton",
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
