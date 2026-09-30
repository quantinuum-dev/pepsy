"""Exact MPO preparation from unfinished clusters crossing each chain cut.

A state contains only active clusters. Completed disjoint choices are summed
by MPO paths, never enumerated as full collections. This is a structural
finite-state construction, not a globally minimal weighted automaton.
"""

from itertools import product
from math import prod

import numpy as np

from ._mpo_sparse import SparseVirtualTensor


class FrontierMPO:
    """Reachable active-cluster states and their exact local transitions."""

    def __init__(self, source, memory_budget, state_budget=None):
        from .cluster_channels import _budget
        from .mpo_product import _operator_schmidt

        if source.cutoff not in (None, 0.) or source.max_bond is not None or source.assembly_chi is not None:
            raise ValueError('frontier preparation requires uncapped fixed source factors and assembly.')
        if source.graph is not None and source.assembly != 'recursive' and (
                source.graph_assembly != 'exact' or source.max_collection_order is not None):
            raise ValueError("frontier preparation requires graph_assembly='exact' or uncapped recursive assembly; "
                             'bounded/auto collection targets must use reference preparation.')
        self.source = source
        self.keys = tuple(source._intervals if source.graph is None else source._graph_clusters)
        self.clusters = {key: tuple(range(key[0], key[1]+1)) if source.graph is None else tuple(key)
                         for key in self.keys}
        self.singles = {sites[0]: key for key, sites in self.clusters.items() if len(sites) == 1}
        self.core_shapes, core_charges = {}, {}
        native = source.physical_space.symmetry is not None
        if native:
            from ._cluster_native import _symmetry

            symmetry = _symmetry(source.physical_space)
            zero = symmetry.combine()
        else:
            zero = 0
        for key, sites in self.clusters.items():
            if len(sites) == 1:
                continue
            shape = (source.phys_dim**len(sites),)*2
            _budget(shape, memory_budget)
            cores = _operator_schmidt(np.zeros(shape), len(sites), source.phys_dim, 0.,
                                      factorization='fixed', physical_space=source.physical_space)
            self.core_shapes[key] = tuple(c.shape for c in cores)
            if native:
                core_charges[key] = cores.charges
        starts = [[] for _ in range(source.L)]
        masks = {}
        for key in self.core_shapes:
            starts[min(self.clusters[key])].append(key)
            masks[key] = sum(1 << site for site in self.clusters[key])

        # All states are reachable. Every partial choice has a completion by
        # using only singleton branches at the remaining unoccupied sites.
        states, transitions = [((),)], []
        state_budget = source.assembly_state_budget if state_budget is None else state_budget
        for site in range(source.L):
            next_states, site_transitions = {(): None}, []
            for active in states[-1]:
                occupied = [key for key in active if site in self.clusters[key]]
                choices = [None]
                if not occupied:
                    used = 0
                    for key in active:
                        used |= masks[key]
                    choices += [key for key in starts[site] if not used & masks[key]]
                for chosen in choices:
                    collection = tuple(sorted((*active, chosen) if chosen is not None else active))
                    outgoing = tuple(key for key in collection if max(self.clusters[key]) > site)
                    next_states[outgoing] = None
                    if len(next_states) > state_budget:
                        raise ValueError(f'frontier preparation exceeds frontier_state_budget={state_budget} '
                                         f'at site {site}; no cluster contributions were dropped.')
                    site_transitions.append((active, outgoing, collection))
            states.append(tuple(sorted(next_states)))
            transitions.append(tuple(site_transitions))
        self.states, self.transitions = tuple(states), tuple(transitions)
        self.offsets, dimensions, charges = [], [], []
        for cut, cut_states in enumerate(states):
            offsets, offset, local_charges = {}, 0, []
            for active in cut_states:
                offsets[active] = offset
                rank = prod(self.rank(key, cut) for key in active)
                offset += rank
                # The endpoint maps scale quadratically in the unprojected
                # frontier dimension; refuse oversized maps before allocating.
                _budget((offset, offset), memory_budget)
                if native:
                    indices = (core_charges[key][sum(site < cut for site in self.clusters[key])]
                               for key in active)
                    local_charges.extend(symmetry.combine(*q) for q in product(*indices))
                else:
                    local_charges.extend([zero]*rank)
            self.offsets.append(offsets)
            dimensions.append(offset)
            charges.append(tuple(local_charges))
        self.dimensions, self.charges = tuple(dimensions), tuple(charges)
        self.report = dict(preparation='frontier', full_collections_enumerated=0,
                           frontier_state_budget=state_budget,
                           frontier_state_counts=tuple(map(len, states)),
                           frontier_transition_count=sum(map(len, transitions)),
                           frontier_bond_dimensions=self.dimensions[1:-1])

    def rank(self, key, cut):
        sites = self.clusters[key]
        count = sum(site < cut for site in sites)
        return 1 if count in (0, len(sites)) else self.core_shapes[key][count-1][1]

    def rows(self, active, cut):
        offset = self.offsets[cut][active]
        return range(offset, offset+prod(self.rank(key, cut) for key in active))

    def arrays(self, residuals):
        """Sparse reference transitions; no complete collection or product core."""
        from .mpo_product import _operator_schmidt

        source = self.source
        cores = {key: _operator_schmidt(residuals[key], len(self.clusters[key]), source.phys_dim, 0.,
                                        factorization='fixed', physical_space=source.physical_space)
                 for key in self.core_shapes}
        arrays = []
        for site, transitions in enumerate(self.transitions):
            array = SparseVirtualTensor((*self.dimensions[site:site+2], source.phys_dim, source.phys_dim))
            for incoming, outgoing, collection in transitions:
                physical = next((key for key in collection if site in self.clusters[key]), None)
                local = None if physical is None else cores[physical][self.clusters[physical].index(site)]
                wires = tuple(key for key in collection if key != physical)
                wire_ranges = tuple(range(self.rank(key, site)) for key in wires)
                for positions in product(*wire_ranges):
                    fixed = dict(zip(wires, positions))
                    for left, right in product(range(local.shape[0]) if local is not None else (0,),
                                               range(local.shape[1]) if local is not None else (0,)):
                        pair = []
                        for cut, active, value in ((site, incoming, left), (site+1, outgoing, right)):
                            index = 0
                            for key in active:
                                index = index*self.rank(key, cut)+(value if key == physical else fixed[key])
                            pair.append(self.offsets[cut][active]+index)
                        block = residuals[self.singles[site]] if local is None else local[left, right]
                        array._add_block(tuple(pair), block)
            arrays.append(array)
        return tuple(arrays)

    def evaluate(self, residuals):
        from ._cluster_native import charge_levels
        from .mpo_semantic import FirstDegreeMPO

        source = self.source
        native = source.physical_space.symmetry is not None
        return FirstDegreeMPO(self.arrays(residuals), degree=source.cluster_size,
                             levels=charge_levels(self.charges) if native else None,
                             physical_space=source.physical_space,
                             metadata={'history_valid': False, '_native_charge_validated': native})
