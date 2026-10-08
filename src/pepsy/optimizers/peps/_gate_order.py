"""Conservative traversal of commuting gate layers, preserving all barriers."""

from functools import wraps
from numbers import Integral


def gate_order_option(value):
    value = str(value).lower().strip()
    if value not in {'input', 'column', 'row'}:
        raise ValueError("gate_order must be 'input', 'column', or 'row'")
    return value


def order_gates(optimizer, policy):
    """Sort only contiguous diagonal two-qubit gates with coordinate sites.

    Exact diagonality is required, not an approximate commutator test. Every
    other gate is a barrier, including single-site gates and trainable gates.
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
                and optimizer._gate_bond_factor(gate) == 2)

    def key(entry):
        a, b = sorted(entry[1][1])
        # Visit one strip at a time, reversing the inner traversal on alternate
        # strips. Preserve input order for repeated gates on the same bond.
        strip = min(a[major], b[major])
        inner = min(a[1 - major], b[1 - major])
        orientation = a[major] != b[major]
        return strip, inner if strip % 2 == 0 else -inner, orientation

    ordered, block = [], []
    for entry in entries:
        if sortable(entry):
            block.append(entry)
        else:
            ordered.extend(sorted(block, key=key))
            block.clear()
            ordered.append(entry)
    ordered.extend(sorted(block, key=key))
    return ordered


def ordered_gate_run(method):
    """Use a temporary traversal; retain the user's queue even after failure."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
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
