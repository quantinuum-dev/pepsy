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
_AUTOMATON_EXPORTS = ["MPOChannel", "MPOTransition", "MPOAutomaton"]
_SYMBOL_MODULES.update({name: ".mpo_automaton" for name in _AUTOMATON_EXPORTS})
_MPO_EXPORTS = [
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOCompressionReport",
    "FirstDegreeMPO",
]
_SYMBOL_MODULES.update({name: ".mpo" for name in _MPO_EXPORTS})
_SUBMODULES = ("gates", "hamiltonians", "mpo", "mpo_automaton")

__all__ = [
    *_GATE_EXPORTS,
    "ham_tn",
    *_AUTOMATON_EXPORTS,
    *_MPO_EXPORTS,
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
