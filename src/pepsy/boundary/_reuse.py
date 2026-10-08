"""Validated, local boundary reuse for dense Torch and CuPy PEPS workflows.

No scalar contractions are cached. Unsupported arrays use the ordinary path.
Torch version counters detect in-place writes without transferring GPU data.
CuPy uses exact device-side snapshot comparisons; only a boolean reaches the
host. Weak references release snapshots when the corresponding arrays die.
"""

import weakref


_cupy_versions = {}


def _array_stamp(data):
    if type(data).__module__.split('.')[0] == 'cupy':
        import cupy as cp

        key = id(data)
        metadata = (tuple(data.shape), str(data.dtype), str(data.device))
        old = _cupy_versions.get(key)
        if old is not None and old[0]() is data:
            ref, snapshot, version, previous_metadata = old
            if metadata == previous_metadata and bool(cp.array_equal(data, snapshot)):
                return (key, version, *metadata)
            version += 1
        else:
            ref = weakref.ref(data, lambda _: _cupy_versions.pop(key, None))
            version = 0
        _cupy_versions[key] = (ref, data.copy(), version, metadata)
        return (key, version, *metadata)
    if type(data).__module__.split('.')[0] != 'torch' or data.requires_grad:
        return None
    try:
        version = data._version
    except RuntimeError:  # Inference tensors do not track in-place changes.
        return None
    return (id(data), version, tuple(data.shape), str(data.dtype), str(data.device))


def layer_sources(ket, bra, norm):
    """Attach source arrays, not freshly allocated conjugates, to a double layer."""
    entries = []
    for source, layer in ((ket, 'KET'), (bra, 'BRA')):
        for tensor in source:
            if _array_stamp(tensor.data) is None:
                return
            sites = [t for t in tensor.tags if t.startswith('I') and ',' in t]
            if len(sites) != 1:
                return
            actual = norm[sites[0], layer]
            if not hasattr(actual, 'inds'):
                return
            entries.append((tensor.data, tuple(actual.inds), tuple(sorted(actual.tags))))
    norm._pepsy_boundary_sources = tuple(entries)


def _network_stamp(network):
    stamps, refs = [], []
    for tensor in network:
        stamp = _array_stamp(tensor.data)
        if stamp is None:
            return None
        stamps.append((stamp, tuple(tensor.inds), tuple(sorted(tensor.tags))))
        refs.append(tensor.data)
    return (tuple(stamps), getattr(network, 'exponent', 0.)), refs


class EnvironmentCache:
    """One entry per directional cut, validated against its slice and predecessor."""

    def __init__(self):
        self.entries = {}
        self.hits = 0
        self.rebuilds = 0

    def signature(self, comp, key, previous):
        sources = getattr(comp.norm, '_pepsy_boundary_sources', None)
        if sources is None or comp.flat or not comp.write_back or comp.track_boundary_fidelity:
            return None
        axis, step, side = key[0], int(key[1:-2]), key[-1]
        size = comp.Lx if axis == 'X' else comp.Ly
        index = step if side == 'l' else size - 1 - step
        tag = f'{axis}{index}'
        stamps, refs = [], []
        for data, inds, tags in sources:
            if tag not in tags:
                continue
            stamp = _array_stamp(data)
            if stamp is None:
                return None
            stamps.append((stamp, inds, tags))
            refs.append(data)
        if not stamps:
            return None
        predecessor = None
        if previous is not None:
            value = _network_stamp(previous)
            if value is None:
                return None
            predecessor, previous_refs = value
            refs.extend(previous_refs)
        # Include all numerical boundary-fit controls. Timing/progress do not
        # affect the mathematical environment and need not invalidate it.
        fields = ('fit_mode', 'fit_layer_mode', 'fit_layer_order', 'layer_tags',
                  'fit_init_strategy', 'fit_init_seed', 'fit_block_size',
                  'fit_adaptive_sweeps', 'fit_max_bond', 'fit_sweep_sequence',
                  'fit_cutoff', 'fit_cutoff_mode', 'fit_compression_opts',
                  'fit_min_iter', 'fit_rtol', 'fit_patience', 'n_iter',
                  'equalize_norms', 'retag')
        def policy_value(name):
            if name == 'fit_cutoff':
                return comp._resolve_fit_cutoff(comp.norm)
            if name == 'layer_tags' and comp.layer_tags is None:
                return ('KET', 'BRA')
            return getattr(comp, name)

        policy = tuple((name, repr(policy_value(name))) for name in fields)
        policy += (('contraction_opt', id(comp.contraction_opt)),
                   ('fit_contraction_opt', id(comp.fit_contraction_opt)))
        return (tuple(stamps), predecessor, policy), refs

    def lookup(self, comp, key, previous):
        signature = self.signature(comp, key, previous)
        entry = self.entries.get(key)
        current = dict.get(comp.mps_boundaries, key)
        if signature is not None and entry is not None and current is not None:
            output = _network_stamp(current)
            if (signature[0] == entry[0] and output is not None
                    and output[0] == entry[1]):
                self.hits += 1
                return current, signature
        self.rebuilds += 1
        return None, signature

    def store(self, key, signature, result):
        output = _network_stamp(result)
        if signature is None or output is None:
            self.entries.pop(key, None)
        else:
            # Retain array references to prevent Python id reuse, not copies.
            self.entries[key] = (signature[0], output[0], signature[1] + output[1])

    def report(self):
        return {'hits': self.hits, 'rebuilds': self.rebuilds}


class StripEnvironmentCache:
    """Exact prefix/suffix contractions within one active PEPS strip.

    The outer boundary MPS have already been compressed. These partial
    contractions introduce no further approximation. Source/version checks
    include boundary tensors, so a changed transverse cut invalidates its
    dependent prefixes/suffixes as well as a changed physical PEPS tensor.
    """

    def __init__(self):
        self.scope = None
        self.entries = {}
        self.hits = 0
        self.rebuilds = 0

    def reduce(self, strip, *, sources, axis, coordinate, length, active, optimize):
        import quimb.tensor as qtn

        scope = (axis, coordinate, length, id(optimize))
        if scope != self.scope:
            self.entries.clear()
            self.scope = scope
        source_arrays = {(inds, tags): data for data, inds, tags in sources}
        first, last = sorted(active)
        result = strip.select((f'{axis}{first}', f'{axis}{last}'), which='any')
        # The reduced norm is needed only up to a global positive scale.
        # Do not replicate the full strip's exponent in each selected slice.
        result.exponent = 0.
        for side, positions in (('left', range(first)),
                                ('right', range(length - 1, last, -1))):
            previous = None
            for position in positions:
                layer = strip.select(f'{axis}{position}')
                layer.exponent = 0.
                stamps, refs = [], []
                for tensor in layer:
                    labels = (tuple(tensor.inds), tuple(sorted(tensor.tags)))
                    data = source_arrays.get(labels, tensor.data)
                    stamp = _array_stamp(data)
                    stamps.append((stamp, labels))
                    refs.append(data)
                prior = None if previous is None else _network_stamp(previous)
                if prior is not None:
                    refs.extend(prior[1])
                valid = all(stamp is not None for stamp, _ in stamps)
                valid = valid and (previous is None or prior is not None)
                signature = (tuple(stamps), None if prior is None else prior[0])
                key = (side, position)
                entry = self.entries.get(key)
                if valid and entry is not None and signature == entry[0]:
                    output = _network_stamp(entry[1])
                    if output is not None and output[0] == entry[2]:
                        self.hits += 1
                        previous = entry[1]
                        continue
                self.rebuilds += 1
                network = layer if previous is None else layer | previous
                tensor, exponent = network.contract(
                    all, output_inds=network.outer_inds(), optimize=optimize,
                    strip_exponent=True, preserve_tensor=True,
                )
                previous = qtn.TensorNetwork([tensor])
                previous.exponent = exponent
                output = _network_stamp(previous)
                if valid and output is not None:
                    # Keep input arrays alive to prevent Python id recycling.
                    self.entries[key] = (signature, previous, output[0], refs)
                else:
                    self.entries.pop(key, None)
            if previous is not None:
                result = result | previous
        return result

    def report(self):
        return {'hits': self.hits, 'rebuilds': self.rebuilds}
