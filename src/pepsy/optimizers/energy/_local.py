"""Shared local-observable readout for energy optimizers.

Only contraction paths outlive a call. Density matrices and environments are
state-dependent and must be rebuilt after evolution or autodiff parameter updates.
"""

from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral
import math
import warnings

import autoray as ar


def normalize_groups(groups, sites):
    """Validate named sums, preserving operator order on each support."""
    if not isinstance(groups, Mapping):
        raise TypeError("observables must map names to {support: operator} mappings.")
    sites = set(sites)
    result = {}
    for name, terms in groups.items():
        if not isinstance(terms, Mapping):
            raise TypeError("each observable must be a {support: operator} mapping.")
        normalized = {}
        unchanged = True
        for where, operator in terms.items():
            support = (where,) if isinstance(where, Integral) else tuple(where)
            if (not support or any(isinstance(s, bool) or not isinstance(s, Integral)
                                   or s not in sites for s in support)
                    or len(set(support)) != len(support)):
                raise ValueError(f"invalid observable support: {where!r}")
            unchanged &= isinstance(where, tuple)
            if support in normalized:
                normalized[support] = normalized[support] + operator
                unchanged = False
            else:
                normalized[support] = operator
        result[name] = terms if unchanged else normalized
    return result


def validate_batch(torch_batch, chunk_size):
    if torch_batch is True:
        method = "einsum"
    elif torch_batch is False:
        method = "loop"
    elif isinstance(torch_batch, str) and torch_batch in {"einsum", "vmap"}:
        method = torch_batch
    else:
        raise ValueError("torch_batch must be False, True, 'einsum', or 'vmap'.")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer or None.")
    return method


def trace_observables(rhos, groups, *, torch_batch=False, chunk_size=32):
    """Contract operators with shared RDMs, optionally batching pure Torch work."""
    method = validate_batch(torch_batch, chunk_size)
    values = {name: 0.0 for name in groups}
    buckets = defaultdict(list)
    used = False
    for name, terms in groups.items():
        for where, operator in terms.items():
            rho = rhos[where]
            dim = math.isqrt(ar.size(rho))
            rho = ar.do("reshape", rho, (dim, dim))
            backend = ar.infer_backend(rho)
            op_backend = ar.infer_backend(operator)
            if op_backend not in {backend, "numpy", "builtins"}:
                raise TypeError("observable arrays must match the state backend or be NumPy arrays.")
            if backend == "torch":
                import torch

                operator = torch.as_tensor(operator, device=rho.device)
                dtype = torch.promote_types(rho.dtype, operator.dtype)
                rho, operator = rho.to(dtype=dtype), operator.to(dtype=dtype)
            elif op_backend != backend:
                operator = ar.do("array", operator, like=rho)
            operator = ar.do("reshape", operator, (dim, dim))
            if method != "loop" and backend == "torch" and rho.layout == torch.strided:
                buckets[(dim, rho.dtype, rho.device)].append((name, rho, operator))
            else:
                values[name] = values[name] + ar.do("sum", rho * ar.do("transpose", operator))
    for bucket in buckets.values():
        size = len(bucket) if chunk_size is None else chunk_size
        for start in range(0, len(bucket), size):
            chunk = bucket[start:start + size]
            rs = torch.stack([entry[1] for entry in chunk])
            ops = torch.stack([entry[2] for entry in chunk])
            # Vectorize only a pure, fixed-shape contraction. Planning,
            # canonicalization, and mutable TN wrappers remain outside vmap.
            if method == "vmap":
                batch_values = torch.vmap(lambda r, op: (r * op.T).sum())(rs, ops)
            else:
                batch_values = torch.einsum("bij,bji->b", rs, ops)
            for entry, value in zip(chunk, batch_values.unbind()):
                values[entry[0]] = values[entry[0]] + value
            used = True
    return values, method if used else "loop"


def measure_mps_groups(state, groups, *, route, optimize, normalized=True,
                       info=None, torch_batch=False, chunk_size=32):
    """Measure named local sums with one RDM per unique ordered support."""
    method = validate_batch(torch_batch, chunk_size)
    if route not in {"canonical", "envs", "exact"}:
        raise ValueError("route must be 'canonical', 'envs', or 'exact'.")
    groups = normalize_groups(groups, range(state.L))
    exponent = getattr(state, "exponent", 0.0)
    if exponent != 0.0:
        # Cancel common global scale before any environment contraction, not
        # after potentially overflowing its numerator and denominator.
        state = state.copy()
        state.exponent = 0.0
    supports = tuple(dict.fromkeys(where for terms in groups.values() for where in terms))
    native = state.isfermionic() or any(ar.infer_backend(t.data) == "symmray" for t in state.tensors)
    effective_route = "exact" if native else route
    if not native and route != "exact" and not callable(getattr(state, "compute_partial_traces", None)):
        warnings.warn(
            "This Quimb build lacks batched compute_partial_traces; using exact local readout.",
            RuntimeWarning, stacklevel=2,
        )
        effective_route = "exact"
    metadata = dict(route=effective_route, requested_route=route, torch_batch="loop",
                    num_supports=len(supports))
    route = effective_route

    def finish(values):
        # Quimb's canonical and environment routes differ in whether an
        # unnormalized RDM carries exponent. Handle it once at this boundary.
        if not normalized and exponent != 0.0:
            factor = ar.do("power", 10.0, 2 * exponent)
            values = {name: value * factor if groups[name] else value for name, value in values.items()}
        return values, metadata

    if not supports:
        return {name: 0.0 for name in groups}, metadata
    if native:
        # Preserve the established graded operator insertion and array types.
        # Dense RDM batching cannot substitute for Symmray's phase conventions.
        cache = {}
        result = {}
        for name, terms in groups.items():
            total = 0.0
            for where, operator in terms.items():
                key = (where, id(operator))
                if key not in cache:
                    cache[key] = state.local_expectation_exact(
                        operator, where, normalized=normalized, optimize=optimize,
                    )
                total = total + cache[key]
            result[name] = total
        return finish(result)
    sample = next(iter(state.tensors)).data
    sample_backend = ar.infer_backend(sample)
    prepared = all(ar.infer_backend(op) == sample_backend and (
        sample_backend == "numpy" or ar.get_dtype_name(op) == ar.get_dtype_name(sample)
    ) for terms in groups.values() for op in terms.values())
    if len(groups) == 1 and method == "loop" and route != "exact" and prepared:
        name, terms = next(iter(groups.items()))
        # Preserve the inexpensive energy-only public Quimb path.
        value = state.compute_local_expectation(
            terms, normalized=normalized, route=route,
            info=None if info is None else dict(info), inplace=False, optimize=optimize,
        )
        return finish({name: value})
    if route == "exact":
        rhos = {where: state.partial_trace_exact(
            where, normalized=normalized, get="matrix", optimize=optimize,
        ) for where in supports}
    else:
        # Quimb's envs route shares one sweep across all intervals; canonical
        # uses one copy and moves its tracked center across the support list.
        rhos = state.compute_partial_traces(
            supports, normalized=normalized, get="matrix", route=route,
            info=None if info is None else dict(info), inplace=False, optimize=optimize,
        )
    values, used = trace_observables(rhos, groups, torch_batch=torch_batch, chunk_size=chunk_size)
    metadata["torch_batch"] = used
    return finish(values)
