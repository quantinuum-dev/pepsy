"""Call-local exact grouping and factor transfers for tree sampling."""

from __future__ import annotations

import math

import numpy as np


class _FactorWorkspaceExceeded(Exception):
    """Request a smaller chunk before materializing a large remainder."""


class _FactorSamplingContext:
    """Scratch state owned by one sampling call, never by the sampler."""

    min_parent = 64
    min_collapse = 65536

    def __init__(self, sampler, cache_bytes, workspace_bytes):
        self.sampler = sampler
        self.cache_bytes = cache_bytes
        self.workspace_bytes = workspace_bytes
        self.cache = {}
        self.cache_used = 0
        self._root_density = None
        self.sites = {}
        self.radix = None
        dims = [int(sampler._arrays[sampler._node_of_qubit[q]].shape[1])
                for q in range(sampler.nqubits)]
        product = 1
        radix = []
        for dim in reversed(dims):
            radix.append(product if dim > 1 else 0)
            product *= dim
        # Codes range from zero through product - 1. Overflow falls back to
        # exact row grouping, without numeric keys or cross-chunk caching.
        if product <= 2**63:
            self.radix = sampler._as_backend(np.asarray(radix[::-1], dtype=np.int64))
        largest = 1
        for node, arr in sampler._arrays.items():
            size = math.prod(arr.shape)
            children = sampler._children[node]
            remainder = size // arr.shape[1] if len(arr.shape) > 1 else size
            if remainder < self.min_collapse:
                largest = max(largest, remainder)
            offset = 2 if node in sampler._qubit_of_node else 1
            for i, dim in enumerate(arr.shape[offset:]):
                tail = math.prod(arr.shape[offset+i+1:])
                if node != sampler._root:
                    grouped = (
                        arr.shape[0] >= self.min_parent or remainder >= self.min_collapse
                        or arr.shape[0] * dim * tail >= self.min_collapse
                    )
                    if (i == 0 and node not in sampler._qubit_of_node) or not grouped:
                        largest = max(largest, dim**2)
                elif (i > 0 or node in sampler._qubit_of_node) and tail > dim:
                    # Root input is pure. The virtual root's first density
                    # stays singleton, and compact factors handle short tails.
                    largest = max(largest, dim**2)
            if not children:
                largest = max(largest, size)
        template = sampler._arrays[sampler._root]
        self.itemsize = (template.element_size() if sampler.resolved_backend == "torch"
                         else template.dtype.itemsize)
        # Target the largest per-shot remainder/density, not total memory.
        self.batch_limit = max(1, workspace_bytes // (largest * self.itemsize))

    def check_remainder(self, rows, elements):
        limit = max(1, self.workspace_bytes // (elements * self.itemsize))
        if rows > limit:
            raise _FactorWorkspaceExceeded

    def clear(self):
        self.cache.clear()
        self._root_density = None
        self.cache_used = 0

    def root_density(self, vector, ur):
        """Reuse the unconditioned virtual root's first marginal within a call."""
        if self._root_density is None:
            value = self.factor_density(vector[:, None, :], ur)
            size = math.prod(value.shape) * self.itemsize
            if self.cache_used + size <= self.cache_bytes:
                self._root_density = value
                self.cache_used += size
            return value
        return self._root_density

    def _subtree_sites(self, node):
        if node not in self.sites:
            sites = []
            stack = [node]
            while stack:
                current = stack.pop()
                q = self.sampler._qubit_of_node.get(current)
                if q is not None:
                    sites.append(q)
                stack.extend(self.sampler._children[current])
            self.sites[node] = sites
        return self.sites[node]

    def groups(self, configs, node=None):
        """Group whole prefixes for densities, subtree codes for remainders."""
        sampler = self.sampler
        sites = None if node is None else self._subtree_sites(node)
        codes = configs if sites is None else configs[:, sites]
        if self.radix is not None:
            radix = self.radix if sites is None else self.radix[sites]
            keys = (sampler._sum(codes * radix, axis=1)
                    if sampler.resolved_backend == "torch" else codes @ radix)
            axis = None
        else:
            keys = codes
            axis = 0
        if sampler.resolved_backend == "torch":
            xp = sampler._xp()
            unique, inverse, counts = xp.unique(
                keys, dim=axis, sorted=True, return_inverse=True, return_counts=True,
            )
            starts = counts.cumsum(0) - counts
            indices = xp.argsort(inverse, stable=True)[starts]
        else:
            unique, indices, inverse = sampler._xp().unique(
                keys, axis=axis, return_index=True, return_inverse=True,
            )
        return unique, indices, inverse.reshape(-1)

    def transfer(self, node, incoming, ur, configs, *, factor=False, incoming_inverse=None):
        """Reuse child densities and compute every absent prefix only once."""
        sampler = self.sampler
        compute = self.factor_density if factor else self.density
        def select(indices):
            if incoming_inverse is not None and len(indices) == len(incoming):
                # Compact factors arrive in sorted whole-prefix order, and
                # no physical draw precedes this node's first-child transfer.
                return incoming
            return incoming[indices if incoming_inverse is None else incoming_inverse[indices]]

        if incoming.shape[0] == 1 or ur.shape[0] < self.min_parent:
            values = compute(incoming, ur)
            return values if incoming_inverse is None or incoming.shape[0] == 1 else values[incoming_inverse]
        keys, indices, inverse = self.groups(configs)
        previous = self.cache.get(node) if self.radix is not None else None
        if previous is None:
            values = compute(select(indices), ur)
            if self.radix is not None:
                size = len(keys) * (8 + math.prod(values.shape[1:]) * self.itemsize)
                if self.cache_used + size <= self.cache_bytes:
                    self.cache[node] = (keys, values)
                    self.cache_used += size
            return values[inverse]

        xp = sampler._xp()
        oldkeys, oldvalues = previous
        positions = xp.searchsorted(oldkeys, keys)
        clipped = sampler._clip_max(positions, len(oldkeys) - 1)
        missing = (positions == len(oldkeys)) | (oldkeys[clipped] != keys)
        newkeys = keys[missing]
        if len(newkeys) == 0:
            return oldvalues[positions][inverse]
        newvalues = compute(select(indices[missing]), ur)
        # Assemble results even when admission is refused. Never recompute
        # hits or the just-computed misses on cache exhaustion.
        values = sampler._zeros((len(keys),) + tuple(newvalues.shape[1:]), dtype=newvalues.dtype)
        values[~missing] = oldvalues[positions[~missing]]
        values[missing] = newvalues
        size = len(newkeys) * (8 + math.prod(newvalues.shape[1:]) * self.itemsize)
        if self.cache_used + size <= self.cache_bytes:
            # Both sets are sorted and disjoint. Scatter directly into their
            # merged positions instead of concatenating and sorting a second
            # full copy of the (much larger) density values.
            old_positions = sampler._arange(len(oldkeys)) + xp.searchsorted(newkeys, oldkeys)
            new_positions = sampler._arange(len(newkeys)) + xp.searchsorted(oldkeys, newkeys)
            length = len(oldkeys) + len(newkeys)
            merged_keys = sampler._zeros((length,), dtype=oldkeys.dtype)
            merged_values = sampler._zeros((length,) + tuple(newvalues.shape[1:]), dtype=newvalues.dtype)
            merged_keys[old_positions] = oldkeys
            merged_keys[new_positions] = newkeys
            merged_values[old_positions] = oldvalues
            merged_values[new_positions] = newvalues
            self.cache[node] = (merged_keys, merged_values)
            self.cache_used += size
        return values[inverse]

    def density(self, incoming, ur):
        return self.sampler._sample_first_child_density(incoming, ur, workspace_bytes=self.workspace_bytes)

    def collapse(self, remainder, message, representatives, inverse):
        """Gather a large compact remainder in tiles, keeping its output small."""
        sampler = self.sampler
        _, parent, child, future = remainder.shape
        width = max(1, self.workspace_bytes // ((parent * child * future + parent * future) * self.itemsize))
        result = sampler._zeros((len(representatives), parent, future), dtype=remainder.dtype)
        for start in range(0, len(representatives), width):
            stop = min(start + width, len(representatives))
            selected = representatives[start:stop]
            rows = selected if inverse is None else inverse[selected]
            gathered = remainder[rows]
            result[start:stop] = sampler._einsum("BpcF,Bc->BpF", gathered, message[selected])
            del gathered
        return result

    def project_vector(self, vector, remainder, representatives, inverse):
        """Contract shot-dependent vectors without expanding a compact tensor."""
        sampler = self.sampler
        rows = remainder.shape[0] if representatives is None else len(representatives)
        _, parent, child, future = remainder.shape
        self.check_remainder(rows, child * future)
        width = max(1, self.workspace_bytes // ((parent * child * future + child * future) * self.itemsize))
        result = sampler._zeros((rows, child, future), dtype=remainder.dtype)
        for start in range(0, rows, width):
            stop = min(start + width, rows)
            if representatives is None:
                gathered = remainder[start:stop]
            else:
                selected = representatives[start:stop]
                gathered = remainder[selected if inverse is None else inverse[selected]]
            incoming = vector if vector.shape[0] == 1 else vector[start:stop]
            result[start:stop] = sampler._einsum("Ba,BacF->BcF", incoming, gathered)
            del gathered
        return result

    def factor_density(self, factor, ur):
        """Transfer a known factor, tiling projected and rearranged buffers."""
        sampler = self.sampler
        samples, rank, parent = factor.shape
        _, child, future = ur.shape
        per_shot = (3 * rank * child * future + rank * parent + child * child) * self.itemsize
        width = max(1, self.workspace_bytes // per_shot)
        result = sampler._zeros((samples, child, child), dtype=ur.dtype)
        for start in range(0, samples, width):
            stop = min(start + width, samples)
            projected = (factor[start:stop].reshape(-1, parent) @ ur.reshape(parent, -1))
            projected = projected.reshape(stop - start, rank, child, future)
            bra = sampler._moveaxis(projected, 2, 1).reshape(stop - start, child, -1)
            result[start:stop] = bra @ sampler._moveaxis(bra.conj(), 1, 2)
            del bra, projected
        return result

    def project_factor_density(self, factor, remainder, representatives, inverse, factor_inverse):
        """Tile conditional factors as well as compact remainder gathers."""
        sampler = self.sampler
        rows = remainder.shape[0] if representatives is None else len(representatives)
        _, parent, child, future = remainder.shape
        rank = factor.shape[1]
        self.check_remainder(rows, child * child)
        per_shot = (rank * parent + parent * child * future + 2 * rank * child * future + child * child)
        width = max(1, self.workspace_bytes // (per_shot * self.itemsize))
        result = sampler._zeros((rows, child, child), dtype=remainder.dtype)
        for start in range(0, rows, width):
            stop = min(start + width, rows)
            selected = sampler._arange(stop - start) + start if representatives is None else representatives[start:stop]
            incoming = factor if factor.shape[0] == 1 else factor[selected if factor_inverse is None else factor_inverse[selected]]
            gathered = remainder[selected if inverse is None else inverse[selected]]
            projected = sampler._einsum("Bra,BacF->BrcF", incoming, gathered)
            result[start:stop] = sampler._einsum("BrcF,BrdF->Bcd", projected, projected.conj())
            del incoming, gathered, projected
        return result
