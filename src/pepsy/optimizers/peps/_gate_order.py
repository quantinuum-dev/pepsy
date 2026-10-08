"""Conservative traversal of commuting gate layers, preserving all barriers."""

from functools import wraps
from numbers import Integral
from contextlib import nullcontext

import autoray as ar


def gate_order_option(value):
    value = str(value).lower().strip()
    if value not in {'input', 'column', 'row'}:
        raise ValueError("gate_order must be 'input', 'column', or 'row'")
    return value


def order_gates(optimizer, policy):
    """Traverse contiguous commuting two-qubit blocks one strip at a time.

    Disjoint supports commute; overlapping gates must both be exactly
    diagonal. Single-site, trainable, or unsupported gates remain barriers.
    """
    entries = list(enumerate(optimizer.gates))
    if policy == 'input':
        return entries
    major = 1 if policy == 'column' else 0

    def sortable(entry):
        gate, where, which = entry[1]
        return (which is None and optimizer.which is None
                and isinstance(where, (tuple, list)) and len(where) == 2
                and all(isinstance(site, (tuple, list)) and len(site) == 2
                        and all(isinstance(i, Integral) for i in site) for site in where)
                and ar.infer_backend(gate) in {'numpy', 'torch', 'cupy'}
                and not getattr(gate, 'requires_grad', False)
                and tuple(getattr(gate, 'shape', ())) in {(4, 4), (2, 2, 2, 2)})

    def key(entry):
        a, b = sorted(entry[1][1])
        # Visit one strip at a time, reversing the inner traversal on alternate
        # strips. Preserve input order for repeated gates on the same bond.
        orientation = a[major] != b[major]
        # Finish the preferred orientation before turning to transverse
        # strips; interleaving the orientations discards partial strip caches.
        axis = 1 - major if orientation else major
        strip = min(a[axis], b[axis])
        inner = min(a[1 - axis], b[1 - axis])
        return orientation, strip, inner if strip % 2 == 0 else -inner

    ordered, block = [], []
    supports = set()
    nondiagonal_supports = set()
    for entry in entries:
        if sortable(entry):
            support = {tuple(site) for site in entry[1][1]}
            diagonal = optimizer._gate_bond_factor(entry[1][0]) == 2
            conflicts = nondiagonal_supports if diagonal else supports
            if support & conflicts:
                ordered.extend(sorted(block, key=key))
                block.clear()
                supports.clear()
                nondiagonal_supports.clear()
            block.append(entry)
            supports.update(support)
            if not diagonal:
                nondiagonal_supports.update(support)
        else:
            ordered.extend(sorted(block, key=key))
            block.clear()
            supports.clear()
            nondiagonal_supports.clear()
            ordered.append(entry)
    ordered.extend(sorted(block, key=key))
    return ordered


def ordered_gate_run(method):
    """Use a temporary traversal; retain the user's queue even after failure."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        sample = next(iter(getattr(self.state, 'tensors', ())), None)
        data = getattr(sample, 'data', None)
        # CuPy allocations use the current device, including Quimb's native
        # local solves. Bind the complete run to the state's device.
        context = data.device if ar.infer_backend(data) == 'cupy' else nullcontext()
        with context:
            requested = kwargs.get('gate_order')
            policy = gate_order_option(self.gate_order if requested is None else requested)
            ordered = order_gates(self, policy)
            original = self.gates
            self.last_gate_order = {'policy': policy,
                                    'original_steps': [i + 1 for i, _ in ordered]}
            self.gates = [entry for _, entry in ordered]
            try:
                return method(self, *args, **kwargs)
            finally:
                self.gates = original
    return wrapped
