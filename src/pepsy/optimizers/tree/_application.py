"""Validated operator geometry and local target preparation for tree replay.

These helpers receive TTNs, operators and explicit options. Canonical movement,
algorithm dispatch and successful state installation remain at the application
boundary, with no retained numerical environment or optimizer reference here.
"""

from dataclasses import dataclass
import heapq

import numpy as np
import quimb.tensor as qtn

from ...fitting.tree import _region_path
from ._policy import _normalize_where
from .layout import TreePlan
from .ttn import _contract_two_tensors


@dataclass(frozen=True, slots=True)
class OperatorApplication:
    """One validated support/region; no tensor IDs or cacheable array values."""

    declared: tuple
    support: tuple
    region: frozenset
    exponent: float


def _same_tree_plan(left, right):
    """Return whether two plans describe the same rooted tree geometry."""
    return (
        isinstance(left, TreePlan)
        and isinstance(right, TreePlan)
        and left.root == right.root
        and left.children == right.children
        and left.qubit_of_leaf == right.qubit_of_leaf
        and left.root_qubit == right.root_qubit
    )


def plan_operator_application(state, tree_mpo, where, *, full_tree):
    """Validate representation and freeze the active geometry before mutation.

    Only unchanged builder-certified exterior identities can shorten a route.
    FIT retains its Steiner-region convention; full operator routing includes
    every structural node when the complete physical support is active.
    """
    plan = getattr(tree_mpo, "plan", None)
    networks = getattr(tree_mpo, "tree_networks", None)
    if plan is None or networks is None:
        raise TypeError(
            "apply_sub_mpotree requires a TreeMPO/TTNO payload with a TreePlan."
        )
    if not _same_tree_plan(state.plan, plan):
        raise ValueError("TreeMPO and state use different TreePlans.")
    networks = tuple(networks)
    if len(networks) != 1:
        raise NotImplementedError(
            "apply_sub_mpotree currently requires one TreeMPO network; "
            "multi-sector TreeMPO expectation remains supported separately."
        )
    sites = tuple(sorted(state.plan.node_of_qubit))
    declared = sites if where is None else _normalize_where(where)
    if len(set(declared)) != len(declared):
        raise ValueError(
            f"TreeMPO application support repeats a site: {declared!r}."
        )
    operator_support = getattr(tree_mpo, "operator_support", None)
    if operator_support is not None:
        operator_support = tuple(sorted(_normalize_where(operator_support)))
        if any(site not in sites for site in operator_support):
            raise ValueError(
                "TreeMPO operator_support contains sites outside its "
                f"TreePlan: {operator_support!r}."
            )
    if tuple(sorted(declared)) == sites:
        active_support = sites if operator_support is None else operator_support
    elif operator_support is not None and set(declared) == set(operator_support):
        # A complete TreeMPO may be declared by its non-identity support.
        # The identity proof below decides whether its exterior can be
        # stripped or the complete operator must use the full tree.
        active_support = operator_support
    else:
        raise ValueError(
            "a TreeMPO application must declare every physical site of its "
            "TreePlan, or exactly its known operator_support; got "
            f"{declared!r} for sites {sites!r}."
        )
    if bool(getattr(tree_mpo, "fermionic", False)) != bool(
        getattr(state, "fermionic", False)
    ):
        raise TypeError(
            "TreeMPO and TreeTensorNetwork must agree on the fermionic "
            "backend when applying a TreeMPO."
        )
    if bool(getattr(state, "fermionic", False)) and (
        getattr(tree_mpo, "symmetry", None)
        != getattr(state, "symmetry", None)
    ):
        raise TypeError(
            "native TreeMPO and TreeTensorNetwork must use the same "
            f"symmetry, got operator={getattr(tree_mpo, 'symmetry', None)!r} "
            f"and state={getattr(state, 'symmetry', None)!r}."
        )
    if hasattr(tree_mpo, "validate"):
        tree_mpo.validate()

    # Physical support alone cannot prove that exterior tensors are unit
    # identities after operator canonicalization, scaling, or tensor edits.
    # Builder-generated gates retain their minimal route while unchanged;
    # arbitrary/re-gauged operators conservatively use every tree tensor.
    identity_exterior = getattr(tree_mpo, "_identity_exterior_unchanged", None)
    if tuple(active_support) != sites and (
        identity_exterior is None or not identity_exterior()
    ):
        active_support = sites
    operator_exponent = float(getattr(tree_mpo, "exponent", 0.0))
    if not np.isfinite(operator_exponent):
        raise ValueError("TreeMPO exponent must be finite.")

    active_nodes = tuple(state.plan.node_of_qubit[q] for q in active_support)
    region = (
        frozenset(state.plan.nodes()) if full_tree and active_support == sites
        else frozenset(state.steiner_nodes(active_nodes))
    )
    return OperatorApplication(declared, active_support, region, operator_exponent)


def operator_local_tensors(state, tree_mpo, snodes, *, layered,
                           phase_start, phase_event):
    """Make private local layers, or the existing exact local contractions.

    The caller prepares the exterior gauge first. Profiling callbacks carry
    timing only; neither numerical settings nor histories enter this helper.
    Live state indices are read after canonical preparation.
    """
    local = {}
    state_inds = {}
    operator_inds = {}
    for nid in snodes:
        state_t = state.node_tensor(nid).copy()
        state_inds[nid] = set(state_t.inds)
        op_t = tree_mpo.node_tensor(nid).copy()
        # The state plan is the validated routing authority. Use
        # its adjacency here so compatible TreeMPO views need only
        # expose node tensors, not duplicate the neighbor API.
        for neighbor in state.neighbors(nid):
            if neighbor in snodes:
                continue
            shared = qtn.bonds(
                op_t, tree_mpo.node_tensor(neighbor),
            )
            if len(shared) != 1:
                raise ValueError(
                    "TreeMPO boundary must have one virtual bond "
                    f"on edge {(nid, neighbor)!r}."
                )
            edge = next(iter(shared))
            if int(op_t.ind_size(edge)) != 1:
                raise ValueError(
                    "TreeMPO operator_support omits a nontrivial "
                    f"boundary bond on edge {(nid, neighbor)!r}."
                )
            op_t = op_t.isel({edge: 0})
        qubit = state.plan.qubit_of_node.get(nid)
        if qubit is not None:
            upper = tree_mpo.upper_ind(qubit)
            lower = tree_mpo.lower_ind(qubit)
            physical = state.site_ind(qubit)
            if upper not in op_t.inds or lower not in op_t.inds:
                raise ValueError(
                    f"TreeMPO is missing physical site {qubit!r}."
                )
            op_t.reindex_({
                lower: physical,
                upper: physical + "*",
            })
        if layered:
            local[nid] = [state_t, op_t]
            continue
        absorb_started = phase_start()
        try:
            if qubit is not None:
                local[nid] = _contract_two_tensors(
                    state_t, op_t, shared_ind=physical,
                ).reindex_({physical + "*": physical})
            else:
                local[nid] = qtn.tensor_contract(state_t, op_t)
        finally:
            phase_event(
                "tensor_absorption", absorb_started,
                support=() if qubit is None else (qubit,),
                route="subtreempo",
            )
        operator_inds[nid] = (
            set(local[nid].inds) - state_inds[nid]
        )

    return local, state_inds, operator_inds


def peel_order(state, snodes, *, path_order=None, path_routing=True):
    """Return ``(peels, hub)`` for recursive leaf-to-hub application.

    ``peels`` is a list of ``(u, v)`` edges: repeatedly a current
    subtree-leaf ``u`` (with a single remaining subtree neighbour ``v``) is
    peeled off toward ``v`` until a single ``hub`` node remains -- the node
    the orthogonality centre ends on. A path is peeled from one endpoint
    to the other when ``path_routing`` is requested. Exact TreeMPO preparation
    and branched regions use smallest-id leaf peeling before compression.
    """
    path = None
    if path_routing:
        path = _region_path(state, snodes) if path_order is None else path_order
    if path is not None:
        return list(zip(path, path[1:])), path[-1]
    remaining = set(snodes)
    adj = {
        u: tuple(w for w in state.neighbors(u) if w in remaining)
        for u in remaining
    }
    degree = {
        u: sum(w in remaining for w in neighbours)
        for u, neighbours in adj.items()
    }
    leaves = [u for u, degree_u in degree.items() if degree_u == 1]
    heapq.heapify(leaves)
    peels = []
    while len(remaining) > 1:
        while leaves and leaves[0] not in remaining:
            heapq.heappop(leaves)
        if not leaves:
            raise ValueError("subtree peel order requires a connected tree")
        leaf = heapq.heappop(leaves)
        v = next(w for w in adj[leaf] if w in remaining)
        peels.append((leaf, v))
        remaining.discard(leaf)
        degree[leaf] = 0
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, v)
    hub = min(remaining)
    return peels, hub
