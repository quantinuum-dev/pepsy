"""Shared configuration defaults for stabilizer tensor-network frontends."""

# Explicit dense Pauli decomposition enumerates ``4**k`` terms.  Keep the
# stabilizer frontends conservative by default; callers can opt into a larger
# support explicitly when the operator is known to be sparse or cheap.
DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS = 2

__all__ = ["DEFAULT_MAX_PAULI_DECOMPOSITION_QUBITS"]
