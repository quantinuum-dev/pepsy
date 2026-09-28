"""Value-independent subset recurrence for complete cluster collections."""

from numbers import Integral


def validate_assembly_state_budget(value):
    """Validate the hard number of memoized subproblems, including the base."""
    if value is None:
        return None
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError("assembly_state_budget must be a positive integer or None.")
    return int(value)


def compile_collection_recursion(length, clusters, state_budget):
    """Compile all disjoint collections without enumerating the collections.

    For the first available site i, either take its singleton background or
    select a cluster containing i. The remaining available-site mask is the
    shared child subproblem. Every compatible collection has exactly one
    branch sequence. Removing occupied sites always decreases the mask, so
    integer order is a topological evaluation order with no Python recursion.
    """
    anchored = [[] for _ in range(length)]
    for cluster in clusters:
        if len(cluster) > 1:
            mask = sum(1 << site for site in cluster)
            anchored[min(cluster)].append((mask, cluster))
    root = (1 << length) - 1
    pending = [root]
    discovered = {root}
    branches = {}
    while pending:
        mask = pending.pop()
        if mask == 0:
            branches[mask] = ()
            continue
        first_bit = mask & -mask
        site = first_bit.bit_length() - 1
        choices = [(mask ^ first_bit, (site,))]
        choices.extend(
            (mask ^ cluster_mask, cluster)
            for cluster_mask, cluster in anchored[site]
            if mask & cluster_mask == cluster_mask
        )
        branches[mask] = tuple(choices)
        for child, _cluster in choices:
            if child not in discovered:
                if state_budget is not None and len(discovered) >= state_budget:
                    raise ValueError(
                        "complete graph assembly exceeds assembly_state_budget="
                        f"{state_budget}; increase it explicitly or reduce the "
                        "cluster size / graph width. No collections were omitted."
                    )
                discovered.add(child)
                pending.append(child)
    counts = {0: 1}
    uses = dict.fromkeys(branches, 0)
    order = tuple(sorted(branches))
    for mask in order[1:]:
        counts[mask] = sum(counts[child] for child, _ in branches[mask])
        for child, _ in branches[mask]:
            uses[child] += 1
    return {
        "strategy": "exact",
        "collections": (),
        "collection_order": None,
        "collection_count": counts[root] - 1,
        "collection_truncated": False,
        "planner": "subset_dp",
        "planner_state_count": len(branches),
        "planner_state_budget": state_budget,
        "root": root,
        "branches": branches,
        "order": order,
        "uses": uses,
    }
