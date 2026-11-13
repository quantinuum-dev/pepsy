"""Conservative traversal of commuting gate layers, preserving all barriers."""

from functools import wraps
from numbers import Integral
from contextlib import nullcontext
from itertools import product

import autoray as ar

from ...operators.gates import _is_operator_tensor_network, _normalize_gate_which


def _gate_leg(optimizer, entry):
    if _is_operator_tensor_network(getattr(optimizer, 'state', None)):
        return _normalize_gate_which(entry[2] if entry[2] is not None else optimizer.which) or 'upper'
    return None


def gate_order_option(value):
    value = str(value).lower().strip()
    if value not in {'input', 'column', 'row', 'smart'}:
        raise ValueError("gate_order must be 'input', 'column', 'row', or 'smart'")
    return value


def fixed_gate_sites(optimizer, entry):
    """Identify fixed dense qubit gates; all other entries are barriers."""
    gate, where, which = entry
    operator = _is_operator_tensor_network(getattr(optimizer, 'state', None))
    if ((not operator and (which is not None or optimizer.which is not None))
            or ar.infer_backend(gate) not in {'numpy', 'torch', 'cupy'}
            or getattr(gate, 'requires_grad', False)):
        return None
    def site(value):
        return (isinstance(value, (tuple, list)) and len(value) == 2
                and all(isinstance(i, Integral) for i in value))
    shape = tuple(getattr(gate, 'shape', ()))
    if operator:
        sites = (where,) if site(where) else where
        if (not isinstance(sites, (tuple, list)) or not all(map(site, sites))
                or any(optimizer.state.ind_size(getattr(optimizer.state, f'{_gate_leg(optimizer, entry)}_ind')(*s)) != 2
                       for s in sites)):
            return None
    if site(where) and shape == (2, 2):
        return (tuple(where),)
    if (isinstance(where, (tuple, list)) and len(where) == 2 and all(map(site, where))
            and shape in {(4, 4), (2, 2, 2, 2)}
            and sum(abs(a - b) for a, b in zip(*where)) == 1):
        return tuple(map(tuple, where))
    return None


def _pauli_patterns(n):
    """Exact entry patterns for span(I, P), with no fitted coefficients."""
    paulis = {'I': ((1, 0), (0, 1)), 'X': ((0, 1), (1, 0)),
              'Y': ((0, -1j), (1j, 0)), 'Z': ((1, 0), (0, -1))}
    words, first, second, rows, cols = [], [], [], [], []
    size = 2**n
    eye = [[int(i == j) for j in range(size)] for i in range(size)]
    for word in product('IXYZ', repeat=n):
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                value = 1
                for k, letter in enumerate(word):
                    value *= paulis[letter][(i >> (n-1-k)) & 1][(j >> (n-1-k)) & 1]
                row.append(value)
            matrix.append(row)
        if all(letter in 'IZ' for letter in word):
            pivot = next((i for i in range(size) if matrix[i][i] == -1), 0)
            a = [[int(i == j and matrix[i][i] == 1) for j in range(size)] for i in range(size)]
            b = [[int(i == j and matrix[i][i] == -1) for j in range(size)] for i in range(size)]
            rows.append(pivot)
            cols.append(pivot)
        else:
            pivot = next(j for j in range(size) if matrix[0][j] != 0)
            a, b = eye, [[v / matrix[0][pivot] for v in row] for row in matrix]
            rows.append(0)
            cols.append(pivot)
        words.append(word)
        first.append(a)
        second.append(b)
    return words, first, second, rows, cols


_PATTERNS = {n: _pauli_patterns(n) for n in (1, 2)}


def _pauli_support(gate, n, constants):
    """Recognize Pauli rotations and diagonal gates using exact equalities.

    Native arrays hold all comparisons. One bitmask and a diagonal predicate
    reach the host; no near-commuting tolerance can erase a small gate term.
    """
    from ...backends import infer_backend_converter_from_sample, infer_backend_signature
    matrix = ar.do('reshape', gate, (2**n, 2**n)) + 0j
    words, first, second, rows, cols = _PATTERNS[n]
    key = (n, infer_backend_signature(matrix))
    if key not in constants:
        convert = infer_backend_converter_from_sample(matrix)
        constants[key] = convert(first), convert(second), convert([2**i for i in range(len(words))]).real
    first, second, weights = constants[key]
    expected = matrix[0, 0] * first + matrix[rows, cols][:, None, None] * second
    matches = ar.do('all', matrix[None, :, :] == expected, axis=(1, 2))
    mask = int(ar.do('sum', matches * weights))
    if mask:
        index = (mask & -mask).bit_length() - 1
        return (words[index],)
    if bool(ar.do('all', matrix == ar.do('diag', ar.do('diagonal', matrix)))):
        return tuple(product('IZ', repeat=n))
    return None


def _cuda_pauli_supports(gates, n, constants):
    """Classify fixed CUDA gates in bounded groups with one host read each."""
    import torch
    from ...backends import infer_backend_converter_from_sample, infer_backend_signature

    words, first, second, rows, cols = _PATTERNS[n]
    matrix = torch.stack([gate.reshape(2**n, 2**n) for gate in gates]) + 0j
    key = (n, infer_backend_signature(matrix[0]))
    if key not in constants:
        convert = infer_backend_converter_from_sample(matrix[0])
        constants[key] = convert(first), convert(second), convert([2**i for i in range(len(words))]).real
    first, second, weights = constants[key]
    expected = (matrix[:, 0, 0, None, None, None] * first[None]
                + matrix[:, rows, cols, None, None] * second[None])
    matches = torch.all(matrix[:, None] == expected, dim=(-2, -1))
    masks = torch.sum(matches * weights, dim=-1).to(torch.int64)
    diagonal = torch.all(matrix == torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1)),
                         dim=(-2, -1))
    stats = torch.stack((masks, diagonal.to(torch.int64)), dim=-1).cpu().tolist()
    result = []
    for mask, is_diagonal in stats:
        if mask:
            result.append((words[(mask & -mask).bit_length() - 1],))
        elif is_diagonal:
            result.append(tuple(product('IZ', repeat=n)))
        else:
            result.append(None)
    return result


def _cuda_diagonal_gates(gates):
    """Apply the existing exact diagonal test to a bounded CUDA batch."""
    import torch

    matrices = torch.stack([gate.reshape(4, 4) for gate in gates])
    diagonal = torch.diag_embed(torch.diagonal(matrices, dim1=-2, dim2=-1))
    return (torch.count_nonzero(matrices - diagonal, dim=(-2, -1)) == 0).cpu().tolist()


def _commute(a, b):
    sites_a, words_a = a[:2]
    sites_b, words_b = b[:2]
    if len(a) > 2 and len(b) > 2 and a[2] != b[2]:
        return True
    shared = set(sites_a) & set(sites_b)
    if not shared:
        return True
    if words_a is None or words_b is None:
        return False
    pairs = [(sites_a.index(site), sites_b.index(site)) for site in shared]
    return all(sum(x[i] != 'I' and y[j] != 'I' and x[i] != y[j] for i, j in pairs) % 2 == 0
               for x in words_a for y in words_b)


def _strip(sites):
    a, b = sites
    return ('column', a[1]) if a[1] == b[1] else ('row', a[0])


def _schedule(entries, descriptors, preferred, *, pair_opposite_legs=False):
    """Greedy topological traversal, pulling in required single-site gates."""
    predecessors = [set() for _ in entries]
    followers = [set() for _ in entries]
    seen = {}
    for j, descriptor in enumerate(descriptors):
        previous = set().union(*(seen.get(site, set()) for site in descriptor[0]))
        for i in previous:
            if not _commute(descriptors[i], descriptor):
                predecessors[j].add(i)
                followers[i].add(j)
        for site in descriptor[0]:
            seen.setdefault(site, set()).add(j)
    remaining = set(range(len(entries)))
    pairs = {i for i in remaining if len(descriptors[i][0]) == 2}
    ordered, active, used = [], None, set()
    pending_counts = [len(pred) for pred in predecessors]
    ready = set()

    def unlock(nodes):
        # Single-site ancestors do not block pair selection once their own
        # pair dependencies are satisfied. Release them logically here; emit
        # them physically only when the chosen pair actually needs them.
        while nodes:
            i = nodes.pop()
            if i in pairs:
                ready.add(i)
                continue
            for j in followers[i]:
                pending_counts[j] -= 1
                if pending_counts[j] == 0:
                    nodes.append(j)

    unlock([i for i in remaining if pending_counts[i] == 0])

    def emit(i):
        ordered.append(entries[i])
        remaining.remove(i)
        for j in followers[i]:
            predecessors[j].discard(i)
    def single_order(i):
        descriptor = descriptors[i]
        return descriptor[0], (descriptor[2] or '') if len(descriptor) > 2 else '', i
    while pairs:
        def priority(i):
            sites = descriptors[i][0]
            strip, bond = _strip(sites), frozenset(sites)
            inner = min(site[0 if strip[0] == 'column' else 1] for site in sites)
            return (strip != active or bond in used, strip != active,
                    strip[0] != preferred, strip[1], inner if strip[1] % 2 == 0 else -inner, i)

        chosen = min(ready, key=priority)
        selected = [chosen]
        if pair_opposite_legs and len(descriptors[chosen]) > 2:
            support, _, leg = descriptors[chosen]
            partner = next((i for i in sorted(ready) if i != chosen
                            and set(descriptors[i][0]) == set(support)
                            and descriptors[i][2] != leg), None)
            if partner is not None:
                selected.append(partner)
        pending = [i for j in selected for i in predecessors[j]]
        required = set()
        while pending:
            ancestor = pending.pop()
            if ancestor not in required:
                required.add(ancestor)
                pending.extend(predecessors[ancestor])
        # All required ancestors are single-site operations on this bond.
        # Different sites commute; preserve original order within each site.
        for i in sorted(required, key=single_order):
            emit(i)
        for item in selected:
            emit(item)
            pairs.remove(item)
            ready.remove(item)
            # Release each emitted pair's dependencies exactly once.
            unlock([item])
        sites = descriptors[chosen][0]
        strip = _strip(sites)
        if strip != active:
            active, used = strip, set()
        bond = frozenset(sites)
        if bond in used:
            used.clear()
        used.add(bond)
    for i in sorted(remaining, key=single_order):
        emit(i)
    return ordered


def _smart_order(optimizer, entries, *, pair_opposite_legs=False):
    ordered, block, descriptors = [], [], []
    signatures, constants = {}, {}
    # Classifications depend only on gate values, so they can be computed
    # before scheduling while unsupported gates remain ordering barriers.
    # Group only nondifferentiable Torch CUDA gates; other backends retain
    # the ordinary scalar path. Bound memory independently of stream length.
    from ...backends import infer_backend_signature
    cuda_groups = {}
    seen = set()
    for _, entry in entries:
        sites = fixed_gate_sites(optimizer, entry)
        gate = entry[0]
        key = (id(gate), len(sites)) if sites is not None else None
        if (key is not None and key not in seen and ar.infer_backend(gate) == 'torch'
                and gate.is_cuda):
            seen.add(key)
            cuda_groups.setdefault((len(sites), infer_backend_signature(gate)), []).append((key, gate))
    for (arity, _), group in cuda_groups.items():
        for start in range(0, len(group), 128):
            batch = group[start:start+128]
            values = _cuda_pauli_supports([gate for _, gate in batch], arity, constants)
            signatures.update((item[0], value) for item, value in zip(batch, values))
    def flush():
        if block:
            # Compare refinement-block counts, then strip changes. This is a
            # bounded heuristic, not a search over all circuit permutations.
            candidates = [_schedule(block, descriptors, preferred,
                                    pair_opposite_legs=pair_opposite_legs)
                          for preferred in ('column', 'row')]
            def cost(candidate):
                blocks, changes, active, used = 0, 0, None, set()
                pending = None
                for _, entry in candidate:
                    sites = fixed_gate_sites(optimizer, entry)
                    if len(sites) == 1:
                        pending = None
                        continue
                    strip, bond = _strip(sites), frozenset(sites)
                    leg = _gate_leg(optimizer, entry)
                    if (pair_opposite_legs and pending is not None
                            and pending[0] == bond and pending[1] != leg):
                        pending = None
                        continue
                    if strip != active or bond in used:
                        blocks += 1
                        used.clear()
                    changes += active is not None and strip != active
                    active = strip
                    used.add(bond)
                    pending = bond, leg
                return blocks, changes
            ordered.extend(min(candidates, key=cost))
            block.clear()
            descriptors.clear()
    for index, entry in entries:
        sites = fixed_gate_sites(optimizer, entry)
        if sites is None:
            flush()
            ordered.append((index, entry))
            continue
        gate = entry[0]
        key = (id(gate), len(sites))
        if key not in signatures:
            signatures[key] = _pauli_support(gate, len(sites), constants)
        block.append((index, entry))
        descriptors.append((sites, signatures[key], _gate_leg(optimizer, entry)))
    flush()
    return ordered


def _fuse_singles(optimizer, ordered, gate_kwargs=None):
    """Fuse adjacent same-site fixed rotations; retain every original index."""
    from ...backends import infer_backend_signature
    # Applying transpose/dagger after B @ A reverses the intended product.
    # Preserve separate operations when any gate path requests a transform.
    policies = [getattr(optimizer, name, {}) for name in
                ('gate_kwargs', 'target_gate_kwargs', 'warmstart_gate_kwargs')]
    policies.append(gate_kwargs or {})
    if any(opts.get('dagger') or opts.get('transpose') for opts in policies):
        return ordered, [[i + 1] for i, _ in ordered]
    result, groups = [], []
    for index, entry in ordered:
        sites = fixed_gate_sites(optimizer, entry)
        if (result and sites is not None and len(sites) == 1
                and fixed_gate_sites(optimizer, result[-1][1]) == sites
                and _gate_leg(optimizer, result[-1][1]) == _gate_leg(optimizer, entry)
                and infer_backend_signature(entry[0]) == infer_backend_signature(result[-1][1][0])):
            old_index, (gate, where, which) = result[-1]
            result[-1] = old_index, (entry[0] @ gate, where, which)
            groups[-1].append(index + 1)
        else:
            result.append((index, entry))
            groups.append([index + 1])
    return result, groups


def order_gates(optimizer, policy, *, pair_opposite_legs=False):
    """Traverse legal gate orders with the selected strip scheduling policy.

    Row/column policies reorder contiguous commuting two-qubit blocks.
    Smart scheduling additionally recognizes Pauli dependencies and moves
    fixed single-site gates. Unsupported/trainable gates remain barriers.
    """
    entries = list(enumerate(optimizer.gates))
    if policy == 'input':
        return entries
    if policy == 'smart':
        return _smart_order(optimizer, entries, pair_opposite_legs=pair_opposite_legs)
    major = 1 if policy == 'column' else 0

    def sortable(entry):
        gate, where, which = entry[1]
        operator = _is_operator_tensor_network(getattr(optimizer, 'state', None))
        return ((operator or (which is None and optimizer.which is None))
                and isinstance(where, (tuple, list)) and len(where) == 2
                and all(isinstance(site, (tuple, list)) and len(site) == 2
                        and all(isinstance(i, Integral) for i in site) for site in where)
                and ar.infer_backend(gate) in {'numpy', 'torch', 'cupy'}
                and not getattr(gate, 'requires_grad', False)
                and tuple(getattr(gate, 'shape', ())) in {(4, 4), (2, 2, 2, 2)}
                and (not operator or fixed_gate_sites(optimizer, entry[1]) is not None))

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

    from ...backends import infer_backend_signature
    cuda_diagonal, cuda_groups = {}, {}
    seen = set()
    for entry in entries:
        gate = entry[1][0]
        if (sortable(entry) and id(gate) not in seen and ar.infer_backend(gate) == 'torch'
                and gate.is_cuda):
            seen.add(id(gate))
            cuda_groups.setdefault(infer_backend_signature(gate), []).append(gate)
    for group in cuda_groups.values():
        for start in range(0, len(group), 128):
            batch = group[start:start+128]
            cuda_diagonal.update((id(gate), diagonal)
                                 for gate, diagonal in zip(batch, _cuda_diagonal_gates(batch)))

    ordered, block = [], []
    supports = set()
    nondiagonal_supports = set()
    for entry in entries:
        if sortable(entry):
            support = {(tuple(site), _gate_leg(optimizer, entry[1])) for site in entry[1][1]}
            gate = entry[1][0]
            diagonal = (cuda_diagonal[id(gate)] if id(gate) in cuda_diagonal
                        else optimizer._gate_bond_factor(gate) == 2)
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
            if requested is None and self._gate_order_auto:
                mode = kwargs.get('mode') or self.mode
                style = kwargs.get('update_style') or self.update_style
                requested = self._default_gate_order(
                    self._normalize_mode(mode), self._normalize_update_style(style),
                )
            policy = gate_order_option(self.gate_order if requested is None else requested)
            if policy == 'smart' and any(ar.infer_backend(t.data) not in {'numpy', 'torch', 'cupy'}
                                         for t in getattr(self.state, 'tensors', ())):
                raise TypeError("gate_order='smart' requires dense NumPy, Torch, or CuPy state arrays")
            ordered = order_gates(self, policy,
                                  pair_opposite_legs=bool(kwargs.get('fuse_pepo_legs', False)))
            if policy == 'smart':
                ordered, groups = _fuse_singles(self, ordered, gate_kwargs=kwargs.get('gate_kwargs'))
            else:
                groups = [[i + 1] for i, _ in ordered]
            original = self.gates
            self.last_gate_order = {'policy': policy,
                                    'original_steps': [i + 1 for i, _ in ordered],
                                    'original_step_groups': groups,
                                    'input_gate_count': len(original),
                                    'compiled_gate_count': len(ordered),
                                    'single_qubit_gates_fused': len(original) - len(ordered)}
            self.gates = [entry for _, entry in ordered]
            try:
                return method(self, *args, **kwargs)
            finally:
                self.gates = original
    return wrapped
