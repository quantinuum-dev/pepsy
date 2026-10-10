"""Bounded expression reuse for dense ALS; never retain input tensor values."""

from collections import OrderedDict

import autoray as ar
import cotengra as ctg
import quimb.tensor as qtn


class ContractionCache:
    """Reuse topology/shape programs while supplying all current arrays."""

    def __init__(self, maxsize=128):
        self.expressions = OrderedDict()
        self.maxsize = maxsize
        self.hits = self.misses = 0

    def __call__(self, network, *, output_inds=(), optimize, preserve_tensor=False):
        tensors = network.tensors
        if not tensors:
            return network.contract(all, output_inds=output_inds, optimize=optimize,
                                    strip_exponent=True, preserve_tensor=preserve_tensor)
        labels = {}

        def label(index):
            return labels.setdefault(index, len(labels))

        inputs = tuple(tuple(label(i) for i in t.inds) for t in tensors)
        output = tuple(label(i) for i in output_inds)
        shapes = tuple(t.shape for t in tensors)
        backend = ar.infer_backend(tensors[0].data)
        # This cache is local to a strip, so optimizer identity remains live.
        policy = optimize if isinstance(optimize, (str, tuple)) else id(optimize)
        key = (inputs, output, shapes, backend, policy)
        expression = self.expressions.get(key)
        if expression is None:
            expression = ctg.array_contract_expression(
                inputs, output=output, shapes=shapes, optimize=optimize,
                strip_exponent=True)
            self.expressions[key] = expression
            self.misses += 1
            if len(self.expressions) > self.maxsize:
                self.expressions.popitem(last=False)
        else:
            self.expressions.move_to_end(key)
            self.hits += 1
        value, exponent = expression(*(t.data for t in tensors), backend=backend)
        if output or preserve_tensor:
            tags = tuple(dict.fromkeys(tag for t in tensors for tag in t.tags))
            value = qtn.Tensor(value, inds=output_inds, tags=tags)
        return value, exponent + network.exponent

    def report(self):
        return dict(hits=self.hits, misses=self.misses, size=len(self.expressions))


def contract_scaled(network, *, output_inds=(), optimize, preserve_tensor=False):
    """Uncached reference using the same exponent-preserving public API."""
    return network.contract(all, output_inds=output_inds, optimize=optimize,
                            strip_exponent=True, preserve_tensor=preserve_tensor)
