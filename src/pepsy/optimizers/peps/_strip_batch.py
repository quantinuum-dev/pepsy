"""Fixed row/column gate targets and disposable simple-update guesses."""

from numbers import Integral

import quimb.tensor as qtn

from ...operators import gate_simple
from ...backends import infer_backend_signature


def layered_target(state, entries, *, gate_bond_factor=None):
    """Attach exact split gates, assigning each factor one lattice endpoint.

    Site, row and column tags partition every factor unambiguously for
    directional boundary FIT and local ALS. No base PEPS tensor is absorbed
    into a gate or given a larger virtual bond.
    """
    target = state.copy()
    target.add_tag('__PEPS_BASE__')
    for number, (gate, where, which) in enumerate(entries):
        if which is not None:
            raise ValueError('layered targets require physical PEPS gates')
        single = len(where) == 2 and all(isinstance(v, Integral) for v in where)
        if not single:
            strip_key(where)
        sites = (where,) if single else where
        inds = tuple(target.site_ind(*site) for site in sites)
        # Reuse the driver's proven diagonal-qubit Schmidt-rank bound. For
        # general, traced or trainable gates retain every split dimension.
        max_bond = (2 if not single and gate_bond_factor is not None
                    and gate_bond_factor(gate) == 2 else None)
        qtn.tensor_network_gate_inds(
            target, gate, inds, contract=False if single else 'split-gate', inplace=True,
            method='svd', cutoff=0., absorb='both', max_bond=max_bond,
            tags=('__PEPS_GATE__', f'__PEPS_GATE_{number}__'),
        )
        for site, ind in zip(sites, inds):
            tids = tuple(target.ind_map[ind])
            if len(tids) != 1:
                raise ValueError('A split gate must uniquely own each output index')
            tensor = target.tensor_map[tids[0]]
            x, y = site
            for tag in (target.site_tag(x, y), target.x_tag(x), target.y_tag(y)):
                tensor.add_tag(tag)
    return target


def strip_key(where):
    """Return the containing strip of an adjacent physical PEPS gate."""
    try:
        (x0, y0), (x1, y1) = where
        if abs(x0 - x1) + abs(y0 - y1) == 1:
            return ('y', y0) if y0 == y1 else ('x', x0)
    except (TypeError, ValueError):
        pass
    raise ValueError('strip batches require adjacent two-site PEPS gates')


def collect_strip_batch(owner, start, width=1):
    """Group consecutive parallel strips, preserving every ordering barrier."""
    key = strip_key(owner.gates[start][1])
    indices = {key[1]}
    end = start
    while end < len(owner.gates):
        _, where, which = owner.gates[end]
        if owner._site_count(where, owner.state) != 2:
            break
        if which is not None or owner.which is not None:
            raise ValueError('strip batches require physical PEPS gates')
        axis, index = strip_key(where)
        proposed = indices | {index}
        if axis != key[0] or max(proposed) - min(proposed) >= width:
            break
        indices = proposed
        end += 1
    return owner.gates[start:end], end - start, end


def simple_update_guess(state, entries, *, chi, cutoff, cutoff_mode):
    """Evolve a private gauge representation, absorbing its gauges exactly once."""
    guess = state.copy()
    gauges = {}
    guess.gauge_all_simple_(gauges=gauges)
    for gate, where, which in entries:
        if which is not None:
            raise ValueError('simple-update guesses require physical PEPS gates')
        gate_simple(guess, gate, where, gauges=gauges, max_bond=chi,
                    cutoff=cutoff, cutoff_mode=cutoff_mode, renorm=False)
    guess.gauge_simple_insert(gauges)
    return guess


def reuse_boundary_guesses(sweeper, previous):
    """Copy compatible fitted MPS guesses, never old numerical cut caches."""
    reused = {'norm': 0, 'overlap': 0}
    if previous is None or sweeper.boundary_engine != 'dmrg':
        return reused
    for name, new, old in zip(reused, (sweeper.bdy, sweeper.bdy_overlap), previous):
        for key, guess in old.mps_b.items():
            if not getattr(guess, '_pepsy_boundary_fitted', False):
                continue
            if key not in new.mps_b:
                continue
            template = new.mps_b[key]
            if (guess.L != template.L or guess.max_bond() > new.chi
                    or {i: guess.ind_size(i) for i in guess.outer_inds()}
                    != {i: template.ind_size(i) for i in template.outer_inds()}
                    or infer_backend_signature(guess.arrays[0])
                    != infer_backend_signature(template.arrays[0])):
                continue
            owned = guess.copy()
            owned.expand_bond_dimension(template.max_bond(), rand_strength=0.)
            owned._pepsy_boundary_fitted = True
            new.mps_b[key] = owned
            reused[name] += 1
    return reused
