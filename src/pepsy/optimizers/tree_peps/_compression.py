"""Geometry-aware compression schedules shared by TreePeps and TreePEPO."""

from __future__ import annotations


def normalize_tree_compression_order(order):
    """Normalize the native tree compression scheduling policy."""

    if order is None:
        return "rank"
    order = str(order).strip().lower().replace("-", "_")
    aliases = {
        "auto": "rank",
        "rank_aware": "rank",
        "tree": "depth",
        "deterministic": "depth",
    }
    order = aliases.get(order, order)
    if order not in {"rank", "depth"}:
        raise ValueError(
            "tree compression order must be 'rank' or 'depth', "
            f"got {order!r}."
        )
    return order


def tree_compression_order(
    plan,
    *,
    center,
    nodes,
    order="rank",
    tensor_getter,
    bond_getter,
):
    """Return a safe, geometry-aware leaf-to-center edge schedule.

    ``rank`` is a cheap live-rank heuristic.  It only considers leaves of the
    still-active connected tree, then scores the next edge using the current
    tensor dimensions and bond dimension.  Thus a reduced child branch is
    visible to the next decision, while every edge is still eliminated toward
    the selected center.  ``depth`` retains the simple farthest-first order.

    The function deliberately receives tensor and bond accessors rather than a
    concrete network class: TreePeps and TreePepo have different physical-leg
    conventions but share this tree scheduling rule.
    """

    order = normalize_tree_compression_order(order)
    center = plan.resolve_site(center)
    nodes = frozenset(plan.resolve_site(node) for node in nodes)
    if not nodes or center not in nodes or not plan.is_connected(nodes):
        raise ValueError(
            "tree compression requires a connected node set containing center"
        )

    if order == "depth":
        return tuple(
            (
                node,
                plan.path(node, center)[1],
            )
            for node in sorted(
                (node for node in nodes if node != center),
                key=lambda node: (-len(plan.path(node, center)), int(node)),
            )
        )

    remaining = set(nodes)
    schedule = []
    while len(remaining) > 1:
        leaves = [
            node
            for node in remaining
            if node != center
            and sum(neighbor in remaining for neighbor in plan.neighbors(node)) == 1
        ]
        if not leaves:
            raise ValueError("tree compression requires a connected node set")

        scored = []
        for node in leaves:
            neighbor = next(
                neighbor
                for neighbor in plan.neighbors(node)
                if neighbor in remaining
            )
            tensor = tensor_getter(node)
            target = tensor_getter(neighbor)
            bond = bond_getter(node, neighbor)
            scored.append(
                (*tree_edge_rank_key(tensor, target, bond), int(node), node, neighbor)
            )

        _, _, _, _, node, neighbor = min(scored)
        schedule.append((node, neighbor))
        remaining.remove(node)

    return tuple(schedule)


def tree_edge_rank_key(tensor, target, bond):
    """Return the live local rank score for one candidate tree edge."""

    left_dim = _external_dim(tensor, bond)
    right_dim = _external_dim(target, bond)
    rank_bound = min(left_dim, right_dim)
    bond_dim = int(tensor.ind_size(bond))
    return rank_bound, bond_dim, left_dim * right_dim


def _external_dim(tensor, bond):
    """Return the product of all live tensor dimensions except ``bond``."""

    dimension = 1
    for index in tensor.inds:
        if index != bond:
            dimension *= int(tensor.ind_size(index))
    return max(1, dimension)
