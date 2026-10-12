"""Linear-work directional environments for an owned one-site ALS sweep."""

import quimb.tensor as qtn


class StripSweepCursor:
    """Cache only the fixed future and updated past of a synchronous sweep.

    The caller owns the network and changes only the active site's tensors
    between consecutive ``local`` calls. Rebuild at every direction change;
    no external mutation or cross-run cache is trusted by this private cursor.
    """

    def __init__(self, network, *, axis, length, optimize):
        self.network = network
        self.layers = [network.select(f'{axis}{i}') for i in range(length)]
        for layer in self.layers:
            layer.exponent = 0.
        self.optimize = optimize
        self.hits = self.rebuilds = 0

    def _extend(self, layer, previous):
        network = layer if previous is None else layer | previous
        tensor, exponent = network.contract(
            all, output_inds=network.outer_inds(), optimize=self.optimize,
            strip_exponent=True, preserve_tensor=True,
        )
        result = qtn.TensorNetwork([tensor])
        result.exponent = exponent
        self.rebuilds += 1
        return result

    def start(self, reverse=False):
        self.order = list(range(len(self.layers)))
        if reverse:
            self.order.reverse()
        self.future = {}
        previous = None
        for position in reversed(self.order[1:]):
            previous = self._extend(self.layers[position], previous)
            self.future[position] = previous
        self.past = None
        self.offset = 0

    def local(self, position):
        if self.offset >= len(self.order) or position != self.order[self.offset]:
            raise ValueError('strip cursor requires consecutive active-site updates')
        if self.offset:
            previous = self.order[self.offset - 1]
            self.past = self._extend(self.layers[previous], self.past)
        result = self.layers[position].copy()
        if self.past is not None:
            result = result | self.past
            self.hits += 1
        if self.offset + 1 < len(self.order):
            result = result | self.future[self.order[self.offset + 1]]
            self.hits += 1
        result.exponent += self.network.exponent
        self.offset += 1
        return result

    def report(self):
        return {'hits': self.hits, 'rebuilds': self.rebuilds, 'strategy': 'directional-cursor'}
