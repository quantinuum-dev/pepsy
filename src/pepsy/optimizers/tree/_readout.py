"""Projected-amplitude Born probabilities for tree controls.

These kernels receive a prepared TTN and a backend conversion callback. They
do not access optimizer settings, histories, RNGs, or canonical-center routing.
"""

import numpy as np
import quimb.tensor as qtn

from ...backends import to_float


def single_pauli_probabilities(tensor, physical, pauli, *, to_backend):
    """Project a normalized canonical tensor before squaring amplitudes."""
    weights = []
    for sign in (1, -1):
        projector = to_backend(
            0.5 * (np.eye(2) + sign * pauli)
        )
        amplitude = tensor.gate(projector, physical).norm()
        weights.append(float(abs(to_float(amplitude, real=True))) ** 2)
    total = sum(weights)
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Measurement requires a finite nonzero working norm.")
    return tuple(weight / total for weight in weights)


def product_pauli_probabilities(state, axes, where, snodes, order, hub, scale,
                                *, to_backend):
    """Carry parity through lossless QR messages on a canonical active tree.

    The caller validates dense qubits and prepares the hub gauge. Numerical
    messages are local to this call; physical site order is independent of
    the tree peel order. The represented exponent cancels from both weights.
    """
    hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
    copy_data = np.zeros((2, 2, 2), dtype=complex)
    copy_data[0, 0, 0] = copy_data[1, 1, 1] = 1.
    xor_data = np.zeros((2, 2, 2), dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            xor_data[a, b, a ^ b] = 1.
    copy_data = to_backend(copy_data)
    xor_data = to_backend(xor_data)
    target_axes = dict(zip(where, axes))
    local, parity = {}, {}
    for nid in snodes:
        tensor = state.node_tensor(nid).copy()
        if nid == hub:
            tensor = tensor / scale
        q = state.plan.qubit_of_node.get(nid)
        if q in target_axes:
            physical = state.site_ind(q)
            axis = target_axes[q]
            if axis != "Z":
                rotation = hadamard if axis == "X" else hadamard @ np.diag([1., -1j])
                tensor = tensor.gate(to_backend(rotation), physical)
            bit, out = qtn.rand_uuid(), qtn.rand_uuid()
            tensor = qtn.tensor_contract(
                tensor, qtn.Tensor(copy_data, inds=(out, physical, bit)),
            ).reindex_({out: physical})
            parity[nid] = bit
        local[nid] = tensor
    for u, v in order:
        bond, bit = state.bond(u, v), parity[u]
        tensor = local.pop(u)
        _, message = tensor.split(
            left_inds=[ix for ix in tensor.inds if ix not in (bond, bit)],
            # Only R contributes to the two norms; the implicit Q
            # basis is orthonormal and need not be materialized.
            method="qr", absorb="rfactor", get="tensors",
            bond_ind=qtn.rand_uuid(),
        )
        if v in parity:
            out = qtn.rand_uuid()
            local[v] = qtn.tensor_contract(
                local[v], message,
                qtn.Tensor(xor_data, inds=(parity[v], bit, out)),
            )
            parity[v] = out
        else:
            local[v] = qtn.tensor_contract(local[v], message)
            parity[v] = bit
    weights = [
        float(abs(to_float(local[hub].isel({parity[hub]: bit}).norm(), real=True))) ** 2
        for bit in (0, 1)
    ]
    total = sum(weights)
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Measurement requires a finite nonzero working norm.")
    return tuple(weight / total for weight in weights)
