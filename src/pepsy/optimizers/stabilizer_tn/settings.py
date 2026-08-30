"""Shared configuration defaults for stabilizer tensor-network frontends."""

# Explicit dense Pauli decomposition enumerates ``4**k`` terms. Keep the
# shared TreeStab default stable, while the coefficient-MPS STN frontend uses
# the more permissive three-qubit default below.
DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS = 2
DEFAULT_MPS_STAB_MAX_PAULI_DECOMPOSITION_QUBITS = 3

__all__ = [
    "DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS",
    "DEFAULT_MPS_STAB_MAX_PAULI_DECOMPOSITION_QUBITS",
]
