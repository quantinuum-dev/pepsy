"""Scalar trace of a complete connected-cluster expansion.

The normalized trace is multiplicative on disjoint supports. Consequently
operator residuals can be replaced by their scalar traces before summing
compatible cluster collections. No operator factorization is needed.
"""

from numbers import Integral

from .cluster_plan import _collections


def compile_trace_plan(length, clusters, state_budget):
    """Cache only the value-independent full-lattice subset recursion."""
    if (state_budget is not None and
            (not isinstance(state_budget, Integral) or
             isinstance(state_budget, bool) or state_budget < 1)):
        raise ValueError("state_budget must be a positive integer or None.")
    ordered = tuple(sorted(clusters, key=lambda sites: (len(sites), sites)))
    if len(set(ordered)) != len(ordered):
        raise ValueError("connected trace clusters must be unique.")
    if any(not sites or tuple(sorted(sites)) != sites for sites in ordered):
        raise ValueError("connected trace clusters must be sorted, nonempty site tuples.")
    try:
        plan = _collections(length, ordered, state_budget)
    except ValueError as exc:
        if "exceeds assembly_state_budget" not in str(exc):
            raise
        raise ValueError(
            f"complete trace expansion exceeds state_budget={state_budget}; "
            "increase it explicitly or reduce cluster order / graph width. "
            "No collections were omitted."
        ) from exc
    return ordered, plan


def evaluate_trace_plan(ordered, plan, local_traces, *, phys_dim, normalized):
    """Subtract proper connected partitions, then sum all disjoint placements.

    ``local_traces[C]`` is Tr(W_C) / d**|C| for the *ordered* local product.
    The subtraction is a scalar form of the connected operator residual
    recurrence. Every backend scalar is fresh for this call and remains in its
    autodiff graph.
    """
    reference = local_traces[ordered[0]]
    zero = reference * 0
    one = zero + 1
    anchored = [[] for _ in range(max(site for cluster in ordered for site in cluster) + 1)]
    residuals = {}

    for cluster in ordered:
        full = sum(1 << site for site in cluster)
        lower_cache = {0: one}

        def lower(mask):
            if mask not in lower_cache:
                site = (mask & -mask).bit_length() - 1
                total = zero
                for block_mask, block in anchored[site]:
                    if mask & block_mask == block_mask:
                        total = total + residuals[block] * lower(mask ^ block_mask)
                lower_cache[mask] = total
            return lower_cache[mask]

        residuals[cluster] = local_traces[cluster] - lower(full)
        anchored[cluster[0]].append((full, cluster))

    values = {0: one}
    for mask in plan["order"][1:]:
        total = zero
        for child, cluster in plan["branches"][mask]:
            total = total + residuals[cluster] * values[child]
        values[mask] = total
    result = values[plan["root"]]
    return result if normalized else result * phys_dim ** plan["root"].bit_count()
