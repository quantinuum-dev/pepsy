"""Persistent SU weights with compensated, pair-local physical writeback."""

import autoray as ar
import quimb.tensor as qtn

from ...boundary._reuse import _network_stamp
from ...operators import gate_simple


class SimpleUpdateWarmStart:
    """Keep diagonal SU closures as warm starts between accepted FU fits.

    The physical network is authoritative. Remove its half-weights before a
    local SU operation, then restore them without touching exterior arrays.
    After ALS, refresh the active bond exactly in its current tensor basis.
    """

    def __init__(self):
        self.gauges = {}
        self.signature = None
        self.references = None
        self.initializations = self.reuses = self.refreshes = 0

    def commit(self, state):
        stamp = _network_stamp(state)
        self.signature, self.references = (stamp if stamp is not None else (None, None))

    def matches(self, state):
        stamp = _network_stamp(state)
        return stamp is not None and self.signature is not None and stamp[0] == self.signature

    def _weights(self, state, where):
        sample = next(iter(state)).data
        inner = set(state.inner_inds())
        indices = {ix for site in where for ix in state[site].inds if ix in inner}
        for index in indices:
            value = self.gauges.get(index)
            if value is None or value.shape != (state.ind_size(index),):
                value = ar.do('ones', (state.ind_size(index),), like=sample).real
            # Null sectors require a positive conditioner to compensate newly
            # fitted physical tensors. This changes no physical scale.
            floor = 1e-6 if ar.get_dtype_name(sample) == 'complex64' else 1e-12
            peak = ar.do('maximum', ar.do('max', value), ar.do('ones_like', value[0]))
            self.gauges[index] = ar.do('maximum', value, peak * floor)

    def _half_weights(self, network, where, *, inverse=False):
        for site in where:
            tensor = network[site]
            indices = tuple(ix for ix in tensor.inds if ix in self.gauges)
            values = [ar.do('sqrt', self.gauges[ix]) for ix in indices]
            if inverse:
                values = [1 / value for value in values]
            tensor.multiply_index_diagonals_(indices, values)

    def prepare(self, state, gate, where, *, which, chi, cutoff, cutoff_mode):
        if not self.matches(state):
            core = state.copy()
            self.gauges = {}
            core.gauge_all_simple_(gauges=self.gauges)
            source = core.copy()
            source.gauge_simple_insert(self.gauges)
            self.initializations += 1
        else:
            source = state
            self._weights(state, where)
            core = state.copy()
            self._half_weights(core, where, inverse=True)
            self.reuses += 1
        actions = list(zip(('upper', 'lower'), gate)) if which == 'both' else [(which, gate)]
        for i, (leg, payload) in enumerate(actions):
            final = i+1 == len(actions)
            gate_simple(core, payload, where, which=leg, gauges=self.gauges,
                        max_bond=chi if final else None, cutoff=cutoff if final else 0.,
                        cutoff_mode=cutoff_mode, renorm=False)
        self._half_weights(core, where)
        guess = source.copy()
        for site in where:
            guess[site].modify(data=core[site].transpose(*source[site].inds).data)
        self.commit(source)
        return source, guess

    def finish(self, state, where):
        self._weights(state, where)
        core = state.copy()
        self._half_weights(core, where, inverse=True)
        qtn.tensor_gauge_simple_bond(core[where[0]], core[where[1]], self.gauges,
                                    renorm=False, smudge=0., fuse_multibonds=False)
        self._half_weights(core, where)
        for site in where:
            core[site].transpose_(*state[site].inds)
        self.refreshes += 1
        self.commit(core)
        return core

    def report(self):
        return dict(initializations=self.initializations, reuses=self.reuses,
                    local_refreshes=self.refreshes)
