"""Physical trace and virtual-bond contraction of constructed PEPOs."""

from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from math import prod

import autoray as ar
import numpy as np

from .mpo_automaton import _as_backend, _backend_reference


_DEFAULT_STATE_BUDGET = 100_000


def _trace_options(state_budget, contract_opts, *, materialized):
    """Validate mutually exclusive sparse and materialized trace controls."""
    if not materialized:
        return {"state_budget": state_budget}
    if state_budget not in (None, _DEFAULT_STATE_BUDGET):
        raise ValueError(
            "state_budget controls sparse active-PEPO traces and cannot be "
            "combined with materialized contraction or compression."
        )
    if contract_opts is None:
        return {}
    if not isinstance(contract_opts, Mapping):
        raise TypeError("contract_opts must be a mapping or None.")
    return dict(contract_opts)


def _join_groups(legs, keys, shared):
    axes = tuple(legs.index(leg) for leg in shared)
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[tuple(key[axis] for axis in axes)].append(index)
    return groups


def _join_size(left, right):
    a, ak = left
    b, bk = right
    shared = tuple(leg for leg in a if leg in b)
    ag, bg = _join_groups(a, ak, shared), _join_groups(b, bk, shared)
    return sum(len(indices) * len(bg.get(key, ())) for key, indices in ag.items())


@lru_cache(maxsize=16)
def _join_plan(a, ak, b, bk, state_budget):
    shared = tuple(leg for leg in a if leg in b)
    left_axes = tuple(i for i, leg in enumerate(a) if leg not in shared)
    right_axes = tuple(i for i, leg in enumerate(b) if leg not in shared)
    legs = tuple(a[i] for i in left_axes) + tuple(b[i] for i in right_axes)
    ag, bg = _join_groups(a, ak, shared), _join_groups(b, bk, shared)
    size = sum(len(indices) * len(bg.get(key, ())) for key, indices in ag.items())
    if state_budget is not None and size > state_budget:
        raise ValueError("state_budget exceeded during sparse PEPO trace contraction.")
    keys, rows, columns, destinations = {}, [], [], []
    for key, left in ag.items():
        for i in left:
            prefix = tuple(ak[i][axis] for axis in left_axes)
            for j in bg.get(key, ()):
                output = prefix + tuple(bk[j][axis] for axis in right_axes)
                destinations.append(keys.setdefault(output, len(keys)))
                rows.append(i)
                columns.append(j)
    return (legs, tuple(keys), np.asarray(rows, dtype=int), np.asarray(columns, dtype=int),
            np.asarray(destinations, dtype=int))


def _trace_active(active, *, normalized, state_budget):
    """Contract sparse site factors; use only the PEPO's actual block values."""
    from .mpo_semantic import _scatter_add_2d
    from .pepo_active import ActivePEPOBlocks, _OPPOSITE_DIRECTION, _site_after

    if state_budget is not None and (
        isinstance(state_budget, bool) or not isinstance(state_budget, (int, np.integer))
        or state_budget < 1
    ):
        raise ValueError("state_budget must be a positive integer or None.")
    if isinstance(active, ActivePEPOBlocks):
        blocks = active._trace_blocks()
        leg_ids = {}
        for site, directions in active.site_directions.items():
            for direction in directions:
                if (site, direction) not in leg_ids:
                    neighbor = _site_after(site, direction, active.lx, active.ly, active.cyclic)
                    edge = len(leg_ids) // 2
                    leg_ids[site, direction] = edge
                    leg_ids[neighbor, _OPPOSITE_DIRECTION[direction]] = edge
        site_legs = {site: tuple(leg_ids[site, d] for d in directions)
                     for site, directions in active.site_directions.items()}
    else:
        blocks, site_legs = active.blocks, active.site_directions
    reference = _backend_reference(block for entries in blocks.values() for block in entries.values())
    factors = []
    for site, entries in blocks.items():
        values = ar.do("stack", tuple(_as_backend(ar.do("trace", block), like=reference)
                                      for block in entries.values()))
        keys = tuple(entries)
        # Only ordinary NumPy values can be pruned numerically. Live backend
        # zeros must retain all derivatives and a static contraction schedule.
        if ar.infer_backend(values) == "numpy":
            keep = np.flatnonzero(values)
            keys, values = tuple(keys[i] for i in keep), values[keep]
        if state_budget is not None and len(keys) > state_budget:
            raise ValueError("state_budget exceeded by sparse PEPO trace site blocks.")
        factors.append((tuple(site_legs[site]), keys, values))
    while len(factors) > 1:
        pairs = [(i, j) for i in range(len(factors)) for j in range(i + 1, len(factors))]
        connected = [(i, j) for i, j in pairs if set(factors[i][0]) & set(factors[j][0])]
        i, j = min(connected or pairs,
                   key=lambda pair: _join_size(factors[pair[0]][:2], factors[pair[1]][:2]))
        a, ak, av = factors[i]
        b, bk, bv = factors[j]
        legs, keys, ai, bi, dest = _join_plan(a, ak, b, bk, state_budget)
        products = av[_as_backend(ai, like=av)] * bv[_as_backend(bi, like=bv)]
        accumulated = ar.do("zeros", (len(keys), 1), like=products)
        values = _scatter_add_2d(accumulated, dest, np.zeros(len(dest), dtype=int), products)[:, 0]
        factors[i] = (legs, keys, values)
        factors.pop(j)
    value = ar.do("sum", factors[0][2])
    return value / active.physical_dim**len(blocks) if normalized else value


def trace_pepo(pepo, *, normalized=False, state_budget=None, **contract_opts):
    """Trace an existing square/graph PEPO, including its actual truncations.

    Active blocks use exact sparse virtual-bond contraction without dense
    site allocation or a separate cluster scalar expansion. ``state_budget``
    optionally caps site entries and pairwise sparse contraction work.
    Materialized Quimb PEPOs/graph networks use their public trace operation;
    ``contract_opts`` applies to that path, not sparse active blocks.
    """
    from .pepo_active import ActivePEPOBlocks, GraphActivePEPOBlocks

    if isinstance(pepo, (ActivePEPOBlocks, GraphActivePEPOBlocks)):
        if contract_opts:
            raise ValueError("contract_opts requires a materialized PEPO; sparse active traces use state_budget.")
        return _trace_active(pepo, normalized=normalized, state_budget=state_budget)
    if state_budget is not None:
        raise ValueError("state_budget applies only to active PEPO blocks.")
    if hasattr(pepo, "upper_inds") and hasattr(pepo, "lower_inds"):
        left, right = tuple(pepo.upper_inds), tuple(pepo.lower_inds)
    else:
        outer = tuple(pepo.outer_inds())
        left = tuple(ind for ind in outer if isinstance(ind, tuple)
                     and len(ind) == 2 and ind[0] == "graph-bra")
        right = tuple(("graph-ket", ind[1]) for ind in left)
        if not left or set(outer) != set(left + right):
            raise TypeError("expected a square PEPO or a materialized graph PEPO.")
    dimension = prod(pepo.ind_size(ind) for ind in left)
    value = pepo.trace(left, right, **contract_opts)
    return value / dimension if normalized else value
