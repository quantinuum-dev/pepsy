"""Validated, local boundary reuse for owned dense Torch PEPS workflows.

No scalar contractions are cached. Unsupported arrays use the ordinary path.
Torch version counters detect in-place writes without transferring GPU data.
"""


def _array_stamp(data):
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
