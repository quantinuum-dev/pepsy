"""Dense Torch qMERA energy with a static Cotengra path and Torch-only trace."""

from __future__ import annotations

import numpy as np
import cotengra as ctg

from .compiled import _BRA_TAG, _KET_TAG, _compiled_tns, _gate_id_for_tensor


def _contraction_plan(tn, *, optimize, torch):
    """Freeze a local cone into constants, gate slots, and binary einsums."""
    tensors = tuple(tn.tensor_map.values())
    names = {}
    inputs = []
    operands = []
    for tensor in tensors:
        labels = tuple(names.setdefault(ind, len(names)) for ind in tensor.inds)
        inputs.append(labels)
        gate_id = _gate_id_for_tensor(tensor)
        if gate_id is None:
            operands.append(torch.as_tensor(tensor.data))
        else:
            tags = tensor.tags
            if _BRA_TAG not in tags and _KET_TAG not in tags:
                raise ValueError(f"Could not identify bra/ket copy for {gate_id!r}.")
            operands.append((gate_id, _BRA_TAG in tags))
    tree = ctg.array_contract_tree(
        tuple(tuple(t.inds) for t in tensors),
        output=(),
        shapes=tuple(t.shape for t in tensors),
        optimize=optimize,
    )
    # Assign each intermediate a permanent slot so graph capture never needs
    # dynamic Python list deletion or sorting.
    active = [(inds, pos) for pos, inds in enumerate(inputs)]
    steps = []
    for i, j in tree.get_path():
        (left, left_id), (right, right_id) = active[i], active[j]
        other = set().union(*(set(inds) for k, (inds, _) in enumerate(active) if k not in (i, j)))
        output = tuple(label for label in dict.fromkeys(left + right) if label in other)
        steps.append((left_id, right_id, left, right, output))
        for k in sorted((i, j), reverse=True):
            active.pop(k)
        active.append((output, len(operands) + len(steps) - 1))
    if len(active) != 1:
        raise ValueError("Cotengra did not return a complete qMERA contraction path.")
    return tuple(operands), tuple(steps), active[0]


def build_torch_fullgraph_loss(
    schedule,
    compiled_chunks,
    *,
    gate_registry,
    array_backend,
    physical_dim,
    product_state_factory,
    normalized,
    energy_per_site,
    real,
):
    """Return a Torch-only loss using frozen local-cone contraction paths.

    Cotengra and Quimb run only while this callable is constructed. One gate
    tensor is built per placement for the entire Hamiltonian evaluation.
    """
    import torch

    if physical_dim != 2:
        raise ValueError("torch_fullgraph requires two-level spin sites.")
    compiled_chunks = tuple(compiled_chunks)
    if not compiled_chunks:
        raise ValueError("hamiltonian contains no local terms.")
    if any(item.fermionic for item in compiled_chunks):
        raise ValueError("torch_fullgraph supports dense spin gates, not graded Symmray arrays.")
    placements = schedule.placements_by_id()
    gate_ids = tuple(
        dict.fromkeys(
            gate_id for item in compiled_chunks for gate_id in item.schedule_placement_ids
        )
    )
    cones = []
    for item in compiled_chunks:
        numerator, denominator = _compiled_tns(
            schedule,
            item.chunk,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            product_state_factory=product_state_factory,
        )
        cones.append(
            (
                _contraction_plan(numerator, optimize=item.optimize, torch=torch),
                _contraction_plan(denominator, optimize=item.optimize, torch=torch)
                if normalized
                else None,
                item.chunk.term.weight,
            )
        )
    constants = [
        operand
        for numerator, denominator, _ in cones
        for plan in (numerator, denominator)
        if plan is not None
        for operand in plan[0]
        if isinstance(operand, torch.Tensor)
    ]
    if not constants:
        raise ValueError("qMERA Torch contraction has no static tensors.")
    reference = constants[0]
    if not reference.is_complex():
        reference = reference.to(torch.complex128)
    eye = torch.eye(4, dtype=reference.dtype, device=reference.device)
    paulis = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.diag([1, -1]).astype(np.complex128),
    }
    gate_data = []
    for gate_id in gate_ids:
        placement = placements[gate_id]
        spec = gate_registry.get(placement.gate_family)
        words = spec.torch_pauli_words
        if words is None or spec.contextual_generator is not None:
            raise ValueError(
                f"torch_fullgraph requires Pauli-rotation metadata for gate family {spec.name!r}."
            )
        matrices = tuple(
            torch.as_tensor(
                np.kron(paulis[word[0]], paulis[word[1]]), dtype=eye.dtype, device=eye.device
            )
            for word in words
        )
        gate_data.append((gate_id, placement.param_key, matrices))
    gate_data = tuple(gate_data)
    cones = tuple(cones)

    def contract(plan, gates):
        operands, steps, (last, last_id) = plan
        arrays = [
            operand
            if isinstance(operand, torch.Tensor)
            else torch.conj(gates[operand[0]])
            if operand[1]
            else gates[operand[0]]
            for operand in operands
        ]
        for left_id, right_id, left, right, output in steps:
            arrays.append(torch.einsum(arrays[left_id], left, arrays[right_id], right, output))
        if last:
            return torch.einsum(arrays[last_id], last, ())
        return arrays[last_id]

    def loss(parameters):
        gates = {}
        for gate_id, param_key, matrices in gate_data:
            angles = parameters[param_key]
            result = eye
            for index, pauli in enumerate(matrices):
                angle = angles[index] / 2
                result = (torch.cos(angle) * eye - 1j * torch.sin(angle) * pauli) @ result
            gates[gate_id] = result.reshape(2, 2, 2, 2)
        total = None
        for numerator, denominator, weight in cones:
            value = contract(numerator, gates)
            if denominator is not None:
                value = value / contract(denominator, gates)
            if weight != 1.0:
                value = value * weight
            total = value if total is None else total + value
        if energy_per_site:
            total = total / schedule.geometry.num_sites
        return torch.real(total) if real else total

    return loss
