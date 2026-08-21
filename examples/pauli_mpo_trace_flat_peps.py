"""Contract ``trace(A @ B @ C @ D)`` as a flat 2D tensor network.

Each row is one native ``PauliMPO`` coefficient tensor train.  The local map
``P[p, upper, lower]`` converts only the four-valued Pauli label into the
two-dimensional qubit legs needed to compose adjacent operator rows.  No
global dense operator or ordinary Quimb MPO is built.

The four rows are periodic vertically, so the bottom output leg connects back
to the top input leg and closes the trace.  Horizontally, each row retains its
ordinary MPO bonds.  The result is a scalar 2D flat PEPS-like network with
``Lx=4`` and ``Ly=nsites``.
"""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn

import pepsy
from pepsy.operators import PauliMPO


def _pauli_basis_tensor():
    """Return local matrices in native ``I, X, Y, Z`` order."""

    return np.stack(
        (
            np.eye(2, dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.diag([1.0, -1.0]).astype(complex),
        )
    )


def pauli_product_flat_network(operators):
    """Build a 2D flat network for a vertically traced Pauli-MPO product.

    Parameters
    ----------
    operators : sequence of PauliMPO
        Operators to multiply from top to bottom.  The vertical direction is
        closed, implementing the trace.

    Returns
    -------
    quimb.tensor.TensorNetwork2D
        A closed, flat 2D tensor network with one tensor per row and site.
    """

    operators = tuple(operators)
    if len(operators) < 2:
        raise ValueError("at least two PauliMPO rows are required.")
    nsites = operators[0].nsites
    if any(operator.nsites != nsites for operator in operators[1:]):
        raise ValueError("all PauliMPO rows must have the same nsites.")

    basis = _pauli_basis_tensor()
    nrows = len(operators)
    tensors = []
    for row, operator in enumerate(operators):
        for col, core in enumerate(operator.to_pauli_cores(copy=False)):
            # C[left, right, p] P[p, upper, lower]
            local = np.einsum("abp,pud->abud", np.asarray(core), basis)
            upper = f"v{(row - 1) % nrows},{col}"
            lower = f"v{row},{col}"

            if nsites == 1:
                data = local[0, 0]
                inds = (upper, lower)
            elif col == 0:
                data = local[0]
                inds = (f"h{row},{col}", upper, lower)
            elif col == nsites - 1:
                data = local[:, 0]
                inds = (f"h{row},{col - 1}", upper, lower)
            else:
                data = local
                inds = (f"h{row},{col - 1}", f"h{row},{col}", upper, lower)

            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=inds,
                    tags=(f"I{row},{col}", f"X{row}", f"Y{col}"),
                )
            )

    # TensorNetwork2D infers the periodic vertical bond from the shared
    # ``v{last},{col}`` / ``v0,{col}`` indices.
    return qtn.TensorNetwork(tensors).view_as(
        qtn.TensorNetwork2D,
        Lx=nrows,
        Ly=nsites,
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
    )


def main():
    operators = (
        PauliMPO.from_terms(4, [(0.7, "XX"), (0.2, "Z")]).compress_pauli(max_bond=4),
        PauliMPO.from_terms(4, [(0.5, "YY"), (-0.3j, "X")]).compress_pauli(max_bond=4),
        PauliMPO.from_terms(4, [(1.1, "ZZ"), (0.15, "YXY")]).compress_pauli(max_bond=4),
        PauliMPO.from_terms(4, [(0.9, "X"), (0.25, "III")]).compress_pauli(max_bond=4),
    )

    native_value = (operators[0] @ operators[1] @ operators[2] @ operators[3]).trace()
    flat = pauli_product_flat_network(operators)
    flat_value = pepsy.contract_flat(
        flat,
        method="exact",
        contraction_opt="greedy",
    )

    print(f"flat shape: {flat.Lx} x {flat.Ly}")
    print(f"native trace: {native_value}")
    print(f"flat trace:   {flat_value}")
    np.testing.assert_allclose(flat_value, native_value, atol=1.0e-12)


if __name__ == "__main__":
    main()
