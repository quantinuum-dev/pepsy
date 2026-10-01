"""Exact native tree factors and call-local measurement-prefix sharing."""

from __future__ import annotations

import autoray as ar
import numpy as np
import quimb.tensor as qtn


def _block_bytes(block):
    if hasattr(block, "element_size"):
        return block.numel() * block.element_size()
    return block.nbytes


class _SymmrayFactorContext:
    """Own conditional native factors for one sampling call only.

    A prefix is represented by a canonical TTN with its selected physical
    legs removed. Lossless, sector-preserving QR moves its centre to the next
    physical site. The canonical exterior then supplies a graded identity,
    letting one-tensor network norms give exact conditional weights. Prefix
    groups share these factors without expanding a dense bond space.
    """

    def __init__(self, sampler, count):
        self.sampler = sampler
        self.cache = {}
        self.cache_used = 0
        # Each prefix is visited once within a chunk. Retaining it only helps
        # later chunks, so a one-chunk request needs no cross-chunk payload.
        self.cache_bytes = (
            sampler.cache_bytes
            if sampler.chunk_size is not None and sampler.chunk_size < count
            else 0
        )
        self.source_blocks = {
            id(block)
            for tensor in sampler._symmray_state["tn"].tensors
            for block in tensor.data.blocks.values()
        } if self.cache_bytes else set()

    def clear(self):
        self.cache.clear()
        self.cache_used = 0

    def conditional(self, branch, depth, prefix):
        """Return a private gauged factor, native norms and host draw weights."""
        if prefix in self.cache:
            return self.cache[prefix]
        sampler = self.sampler
        qubit = sampler._sampling_qubits[depth]
        node = sampler._node_of_qubit[qubit]
        gauged = branch.copy()
        gauged.shift_orthogonality_center(node)
        center = gauged.node_tensor(node)
        physical = gauged.site_ind(qubit)
        index = center.data.indices[center.inds.index(physical)]
        current_codes = {
            metadata: code
            for code, metadata in enumerate(sampler._symmray_code_metadata(index))
        }
        norms = []
        local_codes = []
        for metadata in sampler._physical_code_maps[qubit].values():
            code = current_codes.get(metadata)
            local_codes.append(code)
            if code is None:
                norms.append(0.0)
            else:
                selected = center.isel({physical: code})
                # Network conjugation supplies the outer-leg parity phases;
                # Tensor.H alone is not a fermionic identity environment.
                singleton = qtn.TensorNetwork([selected])
                norms.append(sampler._symmray_norm_value(singleton))
        weights = sampler._symmray_output(norms, dtype="float64")
        weights = ar.do("clip", weights, 0.0, None)
        host = np.asarray(ar.to_numpy(weights), dtype=np.float64)
        total = float(host.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Symmray tree sampling reached a zero or non-finite conditional norm.")
        probabilities = weights / ar.do("sum", weights)
        entry = gauged, tuple(local_codes), tuple(norms), probabilities, host / total
        # Charge each retained payload conservatively, excluding the captured
        # source blocks. Shared non-source blocks may be charged repeatedly;
        # metadata and autograd allocations are not a total-memory guarantee.
        size = len(prefix) * 8 + len(norms) * 24
        if self.cache_used + size > self.cache_bytes:
            return entry
        seen = set(self.source_blocks)
        for tensor in gauged.tensors:
            for block in tensor.data.blocks.values():
                if id(block) not in seen:
                    size += _block_bytes(block)
                    if self.cache_used + size > self.cache_bytes:
                        return entry
                    seen.add(id(block))
        if self.cache_used + size <= self.cache_bytes:
            self.cache[prefix] = entry
            self.cache_used += size
        return entry

    def sample(self, count, rng):
        """Group shot-major draws by prefix while visiting factors depth-first."""
        sampler = self.sampler
        configs = np.empty((count, sampler.nqubits), dtype=np.int64)
        values = [None] * count
        # The standard native sampler consumes one rng.choice uniform per
        # physical site, shot-major. Grouping must retain that exact ordering.
        draws = rng.random((count, sampler.nqubits))
        chunk = min(sampler.chunk_size or count, count)
        for start in range(0, count, chunk):
            rows = np.arange(start, min(start + chunk, count))
            stack = [(0, rows, sampler._symmray_state["tn"], (), 1.0)]
            while stack:
                depth, rows, branch, prefix, probability = stack.pop()
                gauged, local_codes, norms, probabilities, host = self.conditional(branch, depth, prefix)
                qubit = sampler._sampling_qubits[depth]
                node = sampler._node_of_qubit[qubit]
                center = gauged.node_tensor(node)
                cumulative = np.cumsum(host)
                # Match Generator.choice's final CDF normalization. Rounding
                # below one must not expose a trailing zero-weight sector.
                cumulative /= cumulative[-1]
                choices = np.searchsorted(cumulative, draws[rows, depth], side="right")
                choices = np.minimum(choices, len(host) - 1)
                configs[rows, qubit] = choices
                for code in np.unique(choices):
                    selected_rows = rows[choices == code]
                    if host[code] <= 0.0 or local_codes[code] is None:
                        raise ValueError("Symmray tree sampling selected a zero-weight branch.")
                    selected_probability = probability * probabilities[code]
                    if depth + 1 == sampler.nqubits:
                        for row in selected_rows:
                            values[row] = selected_probability
                        continue
                    selected = center.isel({gauged.site_ind(qubit): local_codes[code]})
                    child = gauged.copy()
                    child.node_tensor(node).modify(
                        data=selected.data / (norms[code] ** 0.5),
                        inds=selected.inds, left_inds=None,
                    )
                    # A projection at the centre leaves all exterior factors
                    # isometric. Normalize each prefix to avoid long-shot norm
                    # underflow; its probability remains in float64 separately.
                    child.orthogonality_center = node
                    child._invalidate_norm_cache()
                    stack.append((depth + 1, selected_rows, child, prefix + (int(code),), selected_probability))
        return configs, sampler._symmray_output(values, dtype="float64")
