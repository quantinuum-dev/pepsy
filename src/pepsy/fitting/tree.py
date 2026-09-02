"""Tree-native local variational fitting.

This module contains the tree counterpart of :class:`pepsy.fitting.FIT`.
Unlike the chain FIT implementation, a tree fit has no distinguished left and
right boundary.  It therefore caches one overlap environment for every
directed tree edge and moves an explicit orthogonality centre along the unique
tree geodesic between local update blocks.

The fitted network is expected to expose the small geometry interface supplied
by ``TreeTensorNetwork`` and ``TreePeps``: ``plan``, ``node_tensor``, ``bond``,
``neighbors``, ``site_ind``, canonicalization methods, and ``copy``. The target
can be one of those tree objects or a plain Quimb tensor network whose layer
tensors are tagged by the fitted structural node tags. Keeping this interface
duck-typed lets the same kernel serve both tree optimizers without importing
either optimizer here.
"""

from __future__ import annotations

from itertools import combinations
import math
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .._internal.random import backend_random_array

__all__ = ["TreeFIT"]


def _randomize_tree_guess(
    state,
    region,
    *,
    target=None,
    max_bond=None,
    strength=0.0,
    expand=False,
    seed=0,
):
    """Build a deterministic dense randomized warm-start for ``TreeFIT``.

    ``random`` perturbs the existing active tensors. ``random_expand`` also
    grows active tree bonds towards the exact target rank (capped by
    ``max_bond``), filling only the new directions with seeded noise. Native
    Symmray/fermionic data is left untouched because dense noise cannot
    preserve its charge sectors and graded index metadata.
    """

    region = frozenset(region)
    info = {
        "enabled": False,
        "rand_strength": float(strength),
        "expanded": bool(expand),
        "bonds": [],
        "sites": [],
        "reason": None,
    }
    guess = state.copy()
    if float(strength) == 0.0:
        info["reason"] = "disabled"
        return guess, info

    tensors = [_tensor_of(guess, node) for node in _nodes_of(guess)]
    if any(
        ar.infer_backend(tensor.data) == "symmray"
        or bool(getattr(tensor.data, "fermionic", False))
        for tensor in tensors
    ):
        info["reason"] = "native_sector_growth"
        return guess, info

    rng = np.random.default_rng(int(seed))
    if expand and target is not None:
        planned = []
        for node0, node1 in getattr(guess.plan, "tree_edges", ()):
            if node0 not in region or node1 not in region:
                continue
            fitted_bond = guess.bond(node0, node1)
            target_bond = target.bond(node0, node1)
            current = int(guess.ind_size(fitted_bond))
            target_rank = int(target.ind_size(target_bond))
            if max_bond is not None:
                target_rank = min(target_rank, int(max_bond))
            if target_rank > current:
                planned.append((node0, node1, current, target_rank, fitted_bond))

        # Quimb expands all requested indices to the same minimum size, so
        # process bonds in rank groups and then add noise to only their new
        # slices. This is valid for arbitrary tree degree, not just paths.
        for target_rank in sorted({item[3] for item in planned}):
            group = [item for item in planned if item[3] == target_rank]
            guess.expand_bond_dimension(
                target_rank,
                mode="zeros",
                inds_to_expand=[item[4] for item in group],
                inplace=True,
            )
            for node0, node1, current, _, fitted_bond in group:
                for node in (node0, node1):
                    tensor = _tensor_of(guess, node)
                    axis = tensor.inds.index(fitted_bond)
                    old_slices = [slice(None)] * tensor.ndim
                    old_slices[axis] = slice(0, current)
                    new_shape = list(tensor.shape)
                    new_shape[axis] = target_rank - current
                    random_data = backend_random_array(
                        new_shape,
                        like=tensor.data,
                        dtype=getattr(tensor.data, "dtype", None),
                        scale=float(strength),
                        rng=rng,
                    )
                    old_data = tensor.data[tuple(old_slices)]
                    tensor.modify(data=ar.do(
                        "concatenate", (old_data, random_data), axis=axis
                    ))
                info["bonds"].append({
                    "bond": tuple(sorted((node0, node1))),
                    "current_rank": current,
                    "target_rank": target_rank,
                    "new_rank": int(guess.ind_size(fitted_bond)),
                })

    for node in sorted(region):
        tensor = _tensor_of(guess, node)
        random_data = backend_random_array(
            tensor.shape,
            like=tensor.data,
            dtype=getattr(tensor.data, "dtype", None),
            scale=float(strength),
            rng=rng,
        )
        tensor.modify(data=ar.do("add", tensor.data, random_data))
        info["sites"].append(node)

    invalidate = getattr(guess, "invalidate_canonical_form", None)
    if callable(invalidate):
        invalidate()
    guess.canonize_subtree_(region)
    info["enabled"] = True
    return guess, info


def _nodes_of(state):
    """Return all structural nodes in deterministic order."""

    plan = state.plan
    nodes = getattr(plan, "nodes", None)
    if callable(nodes):
        return tuple(sorted(nodes()))
    return tuple(sorted(state.sites))


def _neighbors_of(state, node):
    """Return structural neighbours for either tree state class."""

    neighbors = getattr(state, "neighbors", None)
    if callable(neighbors):
        return tuple(neighbors(node))
    return tuple(state.plan.neighbors(node))


def _path_of(state, node0, node1):
    """Return the unique structural path between two nodes."""

    path = getattr(state, "node_path", None)
    if callable(path):
        return tuple(path(node0, node1))
    return tuple(state.plan.path(node0, node1))


def _is_connected(state, nodes):
    """Check connectivity for both tree-plan implementations."""

    nodes = frozenset(nodes)
    if len(nodes) <= 1:
        return True
    checker = getattr(state.plan, "is_connected", None)
    if callable(checker):
        return bool(checker(nodes))
    start = next(iter(nodes))
    reached = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in _neighbors_of(state, node):
            if neighbor in nodes and neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return reached == nodes


def _component_of(state, start, blocked):
    """Return the component containing ``start`` after cutting an edge."""

    component = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in _neighbors_of(state, node):
            if node == start and neighbor == blocked:
                continue
            if neighbor not in component:
                component.add(neighbor)
                stack.append(neighbor)
    return frozenset(component)


def _physical_ind(state, node):
    """Return the physical index on ``node``, or ``None`` for virtual nodes."""

    plan = state.plan
    qubit_of_node = getattr(plan, "qubit_of_node", None)
    if qubit_of_node is not None:
        qubit = qubit_of_node.get(node)
        if qubit is None:
            return None
        return state.site_ind(qubit)

    # TreePepsPlan has one physical site per structural node.
    try:
        return state.site_ind(node)
    except (KeyError, ValueError, IndexError):
        return None


def _tensor_of(state, node):
    """Get one structural tensor from either supported tree state."""

    return state.node_tensor(node)


def _exponent_value(network):
    """Return Quimb's represented base-ten exponent as a float."""

    value = getattr(network, "exponent", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scalar_value(value):
    """Convert a scalar backend value into a Python complex number."""

    if hasattr(value, "data"):
        value = value.data
    try:
        value = ar.to_numpy(value)
    except (AttributeError, ImportError, TypeError, ValueError):
        value = np.asarray(value)
    return np.asarray(value).reshape(()).item()


def _scale_stripped(mantissa, exponent):
    """Reconstruct a scalar from Quimb's mantissa/base-ten exponent pair."""

    try:
        return mantissa * (10.0 ** float(exponent))
    except (OverflowError, FloatingPointError):
        return np.inf if float(exponent) >= 0.0 else 0.0


class TreeFIT:
    """Locally fit a bounded-bond tree tensor network to a target tree.

    The implementation mirrors the responsibilities of the chain ``FIT``
    class while replacing chain environments with cached directed tree
    messages.  For a directed edge ``u -> v``, the cached message is the
    contraction of the target and conjugated fitted-state branches on the
    component containing ``u``.  It has the target bond legs crossing that
    tree edge and one fitted bond leg.  A local block contracts only its target
    tensors with the messages on its boundary, so repeated one-, two-, and
    three-node updates do not recontract the untouched branches. A target node
    group can contain several layer tensors and several target bonds can cross
    one fitted tree edge.

    Parameters
    ----------
    tn : tree-compatible tensor network
        Fused or correctly tagged layered target network. Every target tensor
        must carry exactly one structural node tag, and target bonds between
        different node groups must follow the fitted tree topology. Local
        layer bonds are allowed inside a node group. The target is copied by
        default and its virtual indices are privately reindexed, while
        physical indices remain shared with ``p``.
    p : tree tensor network
        Initial bounded-bond tree to optimize.
    max_bond : int or None, default=None
        Maximum dimension of internal update bonds. ``None`` keeps the current
        dimensions unless a cutoff removes directions.
    cutoffs : float, default=1e-12
        Singular-value cutoff used when splitting two- or three-node blocks.
    cutoff_mode : str, default="rsum2"
        Quimb singular-value cutoff convention.
    contraction_opt : object, default="auto-hq"
        Contraction optimizer forwarded to local environment contractions.
    split_method : {"direct", "dm", "src"}, default="direct"
        Local decomposition used to split fitted blocks. ``sdc`` is accepted
        as an alias for the deterministic direct local split.
    retag : bool, default=False
        Align the copied target's structural node tags with ``p``. This is
        useful when two tree objects use different ``node_tag_id`` formats;
        physical/site tags and tensor order are preserved.
    info : dict, optional
        Caller-owned diagnostics mapping. FIT-style live metadata is written
        here without replacing the supplied object.
    warning : bool, default=False
        Reserved for compatibility with FIT diagnostics and fallback warnings.
    inplace : bool, default=False
        Whether to optimize the supplied ``p`` object directly.
    copy_target : bool, default=True
        Copy ``tn`` before private virtual-index reindexing. Set to ``False``
        only when the target is disposable and ownership is transferred.
    """

    def __init__(
        self,
        tn,
        p,
        *,
        max_bond=None,
        cutoffs=1e-12,
        cutoff_mode="rsum2",
        contraction_opt="auto-hq",
        split_method="direct",
        split_seed=0,
        inplace=False,
        retag=False,
        info=None,
        warning=False,
        copy_target=True,
    ):
        self._validate_geometry(tn, p)
        if max_bond is not None:
            if isinstance(max_bond, bool) or not isinstance(max_bond, Integral):
                raise TypeError("max_bond must be an integer or None")
            max_bond = int(max_bond)
            if max_bond < 1:
                raise ValueError("max_bond must be positive")
        cutoffs = float(cutoffs)
        if cutoffs < 0.0:
            raise ValueError("cutoffs must be non-negative")
        split_method = str(split_method).strip().lower().replace("-", "_")
        split_method = {"svd": "direct", "eigh": "dm", "sdc": "direct"}.get(
            split_method, split_method
        )
        if split_method not in {"direct", "dm", "src"}:
            raise ValueError("split_method must be 'direct', 'dm', or 'src'")
        if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
            raise TypeError("split_seed must be an integer")
        if int(split_seed) < 0:
            raise ValueError("split_seed must be non-negative")

        self.p = p if inplace else p.copy()
        self.tn = tn.copy() if copy_target else tn
        self.max_bond = max_bond
        self.cutoffs = cutoffs
        self.cutoff_mode = cutoff_mode
        self.contraction_opt = contraction_opt
        self.split_method = split_method
        self.split_seed = int(split_seed)
        self.nodes = _nodes_of(self.p)
        self._node_set = frozenset(self.nodes)
        self.retag = bool(retag)
        self.warning = bool(warning)
        self.info = info if info is not None else {}
        self._target_tensors = self._collect_target_groups()
        if retag:
            self._retag_target()
        self._components = {
            (u, v): self._component(u, v) for u in self.nodes for v in _neighbors_of(self.p, u)
        }
        self._target_physical = {
            node: _physical_ind(self.p, node) for node in self.nodes
        }
        self._validate_target_groups()
        self.target_layout = (
            "fused"
            if len(self.tn.tensors) == len(self.nodes)
            and all(len(tensors) == 1 for tensors in self._target_tensors.values())
            else "layered"
        )
        self._target_bonds = {}
        self._private_target_bonds = {}
        self._prepare_private_target_indices()

        self._messages = {}
        self._effective_cache = {}
        self.environment_cache_hits = 0
        self.environment_cache_misses = 0
        self.iterations_run = 0
        self.converged = False
        self.convergence_reason = None
        self.last_relative_change = None
        self.last_norm = None
        self.last_overlap = None
        self.adaptive_sweeps_run = 0
        self.one_site_sweeps_run = 0
        self.block_size_trace = []
        self.timing_records = []
        self._split_counter = 0

    @staticmethod
    def _validate_geometry(target, state):
        """Validate the common tree-state interface and geometry."""

        state_required = ("plan", "node_tensor", "bond", "copy")
        if not all(hasattr(state, name) for name in state_required):
            raise TypeError("p must be a TreeTensorNetwork or TreePeps state")
        if not all(hasattr(target, name) for name in ("copy", "tensors")):
            raise TypeError(
                "tn must be a tree-compatible tensor network target"
            )
        state_nodes = set(_nodes_of(state))
        has_target_geometry = all(
            hasattr(target, name) for name in ("plan", "node_tensor", "bond")
        )
        target_nodes = set(_nodes_of(target)) if has_target_geometry else state_nodes
        if target_nodes != state_nodes:
            raise ValueError("target and fitted tree must contain the same nodes")
        structural = tuple(_tensor_of(state, node) for node in state_nodes)
        if len({id(tensor) for tensor in structural}) != len(state_nodes):
            raise ValueError("fitted tree must expose one tensor per structural node")
        tensors = getattr(state, "tensors", None)
        if tensors is not None and len(tensors) != len(state_nodes):
            raise ValueError("fitted tree must contain one tensor per structural node")
        state_edges = {
            frozenset((node, neighbor))
            for node in state_nodes
            for neighbor in _neighbors_of(state, node)
            if node != neighbor
        }
        if has_target_geometry:
            target_structural = tuple(
                _tensor_of(target, node) for node in target_nodes
            )
            if len({id(tensor) for tensor in target_structural}) != len(target_nodes):
                raise ValueError(
                    "target structural node tags must identify one backbone "
                    "tensor per tree node; additional layer tensors are allowed"
                )
            target_edges = {
                frozenset((node, neighbor))
                for node in target_nodes
                for neighbor in _neighbors_of(target, node)
                if node != neighbor
            }
            if target_edges != state_edges:
                raise ValueError(
                    "target and fitted tree must use the same tree topology"
                )
            for node in state_nodes:
                if _physical_ind(target, node) != _physical_ind(state, node):
                    raise ValueError(
                        "target and fitted tree must use matching physical indices"
                    )
                for neighbor in _neighbors_of(state, node):
                    if neighbor not in target_nodes:
                        raise ValueError(
                            "target and fitted tree use different tree plans"
                        )
                    if len(qtn.bonds(
                        _tensor_of(target, node), _tensor_of(target, neighbor)
                    )) != 1:
                        raise ValueError(
                            "target structural backbone edges must have exactly one bond"
                        )
        for node in state_nodes:
            for neighbor in _neighbors_of(state, node):
                if len(qtn.bonds(_tensor_of(state, node), _tensor_of(state, neighbor))) != 1:
                    raise ValueError("fitted tree edges must have exactly one bond")

    def _target_node_tags(self, node):
        """Return candidate structural tags for a target node."""

        tags = []
        target_node_tag = getattr(self.tn, "node_tag", None)
        if callable(target_node_tag):
            tags.append(target_node_tag(node))
        state_node_tag = getattr(self.p, "node_tag", None)
        if callable(state_node_tag):
            tags.append(state_node_tag(node))
        target_tag_id = getattr(self.tn, "_node_tag_id", None)
        if target_tag_id is not None:
            tags.append(str(target_tag_id).format(node))
        state_tag_id = getattr(self.p, "_node_tag_id", None)
        if state_tag_id is not None:
            tags.append(str(state_tag_id).format(node))
        return tuple(dict.fromkeys(tags))

    def _collect_target_groups(self):
        """Group every target tensor by one and only one node tag."""

        tensor_map = getattr(self.tn, "tensor_map", None)
        tag_map = getattr(self.tn, "tag_map", None)
        if tensor_map is None or tag_map is None:
            raise TypeError(
                "tn must expose tensor_map and tag_map so layered target "
                "tensors can be assigned to structural nodes"
            )
        tensor_order = {tid: i for i, tid in enumerate(tensor_map)}
        groups = {}
        owners = {}
        for node in self.nodes:
            tids = []
            for tag in self._target_node_tags(node):
                tids.extend(tag_map.get(tag, ()))
            tids = tuple(dict.fromkeys(tids))
            if not tids:
                raise ValueError(
                    f"target tensors for structural node {node!r} are not "
                    "tagged with its node tag"
                )
            tensors = tuple(
                tensor_map[tid]
                for tid in sorted(tids, key=tensor_order.__getitem__)
            )
            groups[node] = tensors
            for tensor in tensors:
                tensor_id = id(tensor)
                previous = owners.setdefault(tensor_id, node)
                if previous != node:
                    raise ValueError(
                        "each target tensor must carry exactly one structural "
                        "node tag; a tensor is tagged for multiple tree nodes"
                    )

        if len(owners) != len(self.tn.tensors):
            raise ValueError(
                "every target tensor must carry exactly one structural node "
                "tag; untagged or ambiguously tagged layer tensors cannot be "
                "assigned to TreeFIT"
            )
        return groups

    def _retag_target(self):
        """Align every target layer tensor with its fitted node tag."""

        target_node_tag_id = getattr(self.tn, "_node_tag_id", None)
        target_node_tag = getattr(self.tn, "node_tag", None)
        state_node_tag_id = getattr(self.p, "_node_tag_id", None)
        state_node_tag = getattr(self.p, "node_tag", None)
        if state_node_tag_id is None or not callable(state_node_tag):
            raise TypeError(
                "retag=True requires tree objects with structural node tags"
            )

        for node, tensors in self._target_tensors.items():
            target_tags = set(self._target_node_tags(node))
            target_tags.discard(state_node_tag(node))
            backbone = (
                _tensor_of(self.tn, node)
                if callable(target_node_tag)
                else None
            )
            for tensor in tensors:
                tags = set(tensor.tags)
                if tensor is not backbone:
                    tags.difference_update(target_tags)
                tags.add(state_node_tag(node))
                tensor.modify(tags=tags)

        # Keep the target's own node lookup API coherent after changing the
        # tags. A tree object reserves its native node tag for the unique
        # structural backbone tensor. Layer tensors therefore retain the
        # fitted tag while the backbone keeps the target's native tag, so the
        # tree object's node lookup still selects the backbone. Fused targets
        # can safely switch the native format.
        layered = any(len(tensors) > 1 for tensors in self._target_tensors.values())
        if hasattr(self.tn, "_node_tag_id") and not layered:
            self.tn._node_tag_id = state_node_tag_id
        if layered and target_node_tag_id is not None:
            self.info["target_node_tag_id"] = target_node_tag_id
        for cache_name in ("_node_tid_cache", "_tree_peps_tid_cache"):
            self.tn.__dict__.pop(cache_name, None)
        self.info["retagged"] = True

    def _validate_target_groups(self):
        """Check target outputs and inter-node bonds after layer grouping."""

        target_outer = set(self.tn.outer_inds())
        state_outer = set(self.p.outer_inds())
        if target_outer != state_outer:
            raise ValueError(
                "target and fitted tree must use matching physical outer indices"
            )
        for node, physical in self._target_physical.items():
            if physical is None:
                continue
            matches = [
                tensor for tensor in self._target_tensors[node]
                if physical in tensor.inds
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"target physical index {physical!r} must occur exactly "
                    f"once in structural node group {node!r}"
                )

        node_of_tensor = {
            id(tensor): node
            for node, tensors in self._target_tensors.items()
            for tensor in tensors
        }
        state_edges = {
            frozenset((node, neighbor))
            for node in self.nodes
            for neighbor in _neighbors_of(self.p, node)
            if node != neighbor
        }
        for index in self.tn.inner_inds():
            owners = set()
            for tensor_id in self.tn.ind_map.get(index, ()):
                tensor = self.tn.tensor_map[tensor_id]
                node = node_of_tensor.get(id(tensor))
                if node is not None:
                    owners.add(node)
            if len(owners) > 1:
                if len(owners) != 2 or frozenset(owners) not in state_edges:
                    raise ValueError(
                        "target inter-node virtual bonds must follow the fitted "
                        "tree topology"
                    )

    def _component(self, start, blocked):
        """Return the component containing ``start`` after cutting an edge."""

        return _component_of(self.p, start, blocked)

    def _prepare_private_target_indices(self):
        """Reindex target virtual bonds so target and state never cross-connect."""

        physical = {index for index in self._target_physical.values() if index is not None}
        mapping = {}
        all_target_tensors = [
            tensor
            for node in self.nodes
            for tensor in self._target_tensors[node]
        ]
        for tensor in all_target_tensors:
            for index in tensor.inds:
                if index not in physical and index not in mapping:
                    mapping[index] = qtn.rand_uuid()
        for tensor in all_target_tensors:
            tensor.reindex_(mapping)

        def target_bonds(node0, node1):
            bonds = []
            seen = set()
            for tensor0 in self._target_tensors[node0]:
                for tensor1 in self._target_tensors[node1]:
                    for index in qtn.bonds(tensor0, tensor1):
                        if index not in seen:
                            seen.add(index)
                            bonds.append(index)
            return tuple(bonds)

        for node in self.nodes:
            for neighbor in _neighbors_of(self.p, node):
                if (node, neighbor) in self._target_bonds:
                    continue
                target_bond = target_bonds(node, neighbor)
                if not target_bond:
                    raise ValueError(
                        f"target node groups {node!r} and {neighbor!r} must "
                        "share at least one virtual bond"
                    )
                fitted_bond = self.p.bond(node, neighbor)
                self._target_bonds[(node, neighbor)] = target_bond
                self._target_bonds[(neighbor, node)] = target_bond
                self._private_target_bonds[(node, neighbor)] = (
                    target_bond,
                    fitted_bond,
                )
                self._private_target_bonds[(neighbor, node)] = (
                    target_bond,
                    fitted_bond,
                )

    def clear_environment_cache(self):
        """Discard cached branch entanglement environments."""

        self._messages.clear()
        self._effective_cache.clear()
        return self

    def environment_cache_info(self):
        """Return cache size and hit/miss counters."""

        return {
            "messages": len(self._messages),
            "effective_blocks": len(self._effective_cache),
            "hits": int(self.environment_cache_hits),
            "misses": int(self.environment_cache_misses),
        }

    def _message(self, outside, inside):
        """Construct or retrieve one directed target/state branch message."""

        key = (outside, inside)
        cached = self._messages.get(key)
        if cached is not None:
            self.environment_cache_hits += 1
            return cached
        self.environment_cache_misses += 1
        component = self._components[key]
        target_tensors = [
            tensor.copy()
            for node in component
            for tensor in self._target_tensors[node]
        ]
        state_tensors = [_tensor_of(self.p, node).H for node in component]
        target_bonds, fitted_bond = self._private_target_bonds[key]
        message = qtn.TensorNetwork([*target_tensors, *state_tensors]).contract(
            output_inds=(*target_bonds, fitted_bond),
            optimize=self.contraction_opt,
        )
        self._messages[key] = message
        return message

    def _boundary_edges(self, block):
        """Return sorted edges crossing from ``block`` to its exterior."""

        block = frozenset(block)
        return tuple(sorted(
            (node, neighbor)
            for node in block
            for neighbor in _neighbors_of(self.p, node)
            if neighbor not in block
        ))

    def _effective_block(self, block):
        """Build the projected target tensor for a connected local block."""

        block = frozenset(block)
        key = tuple(sorted(block))
        cached = self._effective_cache.get(key)
        if cached is not None:
            self.environment_cache_hits += 1
            return cached.copy()
        self.environment_cache_misses += 1
        tensors = [
            tensor.copy()
            for node in block
            for tensor in self._target_tensors[node]
        ]
        boundary = self._boundary_edges(block)
        for inside, outside in boundary:
            tensors.append(self._message(outside, inside).copy())
        output_inds = tuple(
            self._target_physical[node]
            for node in sorted(block)
            if self._target_physical[node] is not None
        ) + tuple(
            self.p.bond(inside, outside) for inside, outside in boundary
        )
        effective = qtn.TensorNetwork(tensors).contract(
            output_inds=output_inds,
            optimize=self.contraction_opt,
        )
        self._effective_cache[key] = effective.copy()
        return effective

    def _invalidate_for_block(self, block):
        """Invalidate only messages whose component contains an updated node."""

        block = frozenset(block)
        for key, component in tuple(self._components.items()):
            if component.intersection(block):
                self._messages.pop(key, None)
        for key in tuple(self._effective_cache):
            effective_block = frozenset(key)
            depends_on_changed_message = any(
                self._components[(outside, inside)].intersection(block)
                for inside, outside in self._boundary_edges(effective_block)
            )
            if effective_block.intersection(block) or depends_on_changed_message:
                self._effective_cache.pop(key, None)

    def _canonicalize_for_block(self, block, center):
        """Prepare isometric exterior branches and move the centre."""

        region = frozenset(block)
        current_region = getattr(self.p, "canonical_region", None)
        current_center = getattr(self.p, "orthogonality_center", None)
        if current_center is not None:
            # A single centre already makes every branch outside any
            # connected block isometric towards that block. Moving it only
            # changes tensors on the unique centre path, so invalidate those
            # directed messages and retain all untouched branch environments.
            if current_center != center:
                path = _path_of(self.p, current_center, center)
                self._invalidate_for_block(path)
            self.p.shift_orthogonality_center(center, _skip_validate=True)
            return

        is_canonical = getattr(self.p, "is_subtree_canonical_form", None)
        if current_region != region or not (
            callable(is_canonical) and is_canonical(region)
        ):
            # With no single tracked centre there is no safe incremental path
            # to identify. Establish the block gauge once and discard the
            # basis-dependent messages conservatively.
            self.clear_environment_cache()
            self.p.canonize_subtree_(region)
        self.p.shift_orthogonality_center(center)

    def _split_method(self):
        return {
            "direct": "svd",
            "dm": "svd:eig",
            "src": "svd:rand",
        }[self.split_method]

    def _block_center(self, block, preferred=None):
        """Choose a tree-medial centre for a connected local block."""

        block = tuple(block)
        if preferred in block:
            return preferred
        return min(
            block,
            key=lambda node: (
                max(len(_path_of(self.p, node, other)) for other in block),
                sum(len(_path_of(self.p, node, other)) for other in block),
                node,
            ),
        )

    def _local_legs(self, node, block):
        """Return physical and exterior fitted-bond legs owned by ``node``."""

        inds = []
        physical = _physical_ind(self.p, node)
        if physical is not None:
            inds.append(physical)
        for neighbor in sorted(_neighbors_of(self.p, node)):
            if neighbor not in block:
                inds.append(self.p.bond(node, neighbor))
        return tuple(inds)

    def _factor_block(self, effective, block, center):
        """Factor a projected block back onto its original tree edges."""

        block = frozenset(block)
        if len(block) == 1:
            node = next(iter(block))
            return {node: effective}

        parent = {center: None}
        queue = [center]
        while queue:
            node = queue.pop(0)
            for neighbor in sorted(_neighbors_of(self.p, node)):
                if neighbor in block and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        leaves = sorted(
            node for node in block if node != center and not any(
                child in block and parent.get(child) == node for child in block
            )
        )
        remaining = effective
        factors = {}
        for leaf in leaves:
            local_inds = self._local_legs(leaf, block)
            right_inds = tuple(ind for ind in remaining.inds if ind not in local_inds)
            if not local_inds or not right_inds:
                raise ValueError(
                    "TreeFIT could not identify a non-empty local block split"
                )
            edge = (leaf, parent[leaf])
            bond_ind = self.p.bond(*edge)
            split_kwargs = {
                "method": self._split_method(),
                "absorb": "right",
                "max_bond": self.max_bond,
                "cutoff": self.cutoffs,
                "cutoff_mode": self.cutoff_mode,
                "get": "tensors",
                "bond_ind": bond_ind,
            }
            if self.split_method == "src":
                split_kwargs["seed"] = self.split_seed + self._split_counter
                self._split_counter += 1
            left, remaining = qtn.tensor_split(
                remaining,
                local_inds,
                right_inds=right_inds,
                **split_kwargs,
            )
            left.modify(tags=_tensor_of(self.p, leaf).tags)
            factors[leaf] = left
        remaining.modify(tags=_tensor_of(self.p, center).tags)
        factors[center] = remaining
        return factors

    def _install_block(self, factors, block, center, *, validate=True):
        """Install fitted block tensors and restore canonical metadata."""

        # Center movement has already prepared every untouched branch in the
        # required gauge. Clearing/rebuilding ``left_inds`` for the complete
        # tree here would turn each local update into an O(N) operation and
        # would also discard the very proofs used to skip later QR moves.
        invalidate_norm = getattr(self.p, "_invalidate_norm_cache", None)
        if callable(invalidate_norm):
            invalidate_norm()
        for node in block:
            fitted = factors[node]
            live = _tensor_of(self.p, node)
            left_inds = None if node == center else fitted.left_inds
            live.modify(
                data=fitted.data,
                inds=fitted.inds,
                tags=live.tags,
                left_inds=left_inds,
            )
        self.p._canonical_region = frozenset({center})
        if validate:
            self.p.validate(check_canonical=True)

    def _global_overlap(self):
        """Contract the current fitted state against the target."""

        mantissa, exponent = self._global_overlap_stripped()
        return _scale_stripped(mantissa, exponent)

    def _global_overlap_stripped(self):
        """Return the target overlap as a mantissa and base-ten exponent."""

        network = qtn.TensorNetwork([
            *[
                tensor.copy()
                for node in self.nodes
                for tensor in self._target_tensors[node]
            ],
            *[_tensor_of(self.p, node).H for node in self.nodes],
        ])
        value, exponent = network.contract(
            all,
            optimize=self.contraction_opt,
            strip_exponent=True,
        )
        return (
            _scalar_value(value),
            float(exponent)
            + _exponent_value(self.tn)
            + _exponent_value(self.p),
        )

    @staticmethod
    def _log_stripped_norm(network):
        """Return ``log(norm)`` without materializing a large scale."""

        mantissa, exponent = network.norm(strip_exponent=True)
        mantissa = abs(_scalar_value(mantissa))
        if mantissa == 0.0:
            return -np.inf
        return float(np.log(mantissa) + float(exponent) * np.log(10.0))

    def _network_norm(self, network):
        """Read a represented tree norm without changing its gauge."""

        mantissa, exponent = network.norm(strip_exponent=True)
        return float(abs(_scale_stripped(
            _scalar_value(mantissa), float(exponent)
        )))

    def _normalized_overlap_fidelity(self):
        """Return normalized target overlap using stripped exponents."""

        overlap, overlap_exponent = self._global_overlap_stripped()
        log_overlap = -np.inf if abs(overlap) == 0.0 else float(
            np.log(abs(overlap)) + overlap_exponent * np.log(10.0)
        )
        log_target = self._log_stripped_norm(self.tn)
        log_fitted = self._log_stripped_norm(self.p)
        log_fidelity = 2.0 * (log_overlap - log_target - log_fitted)
        if not np.isfinite(log_fidelity):
            return 0.0 if log_fidelity < 0.0 else 1.0
        return float(min(1.0, max(0.0, np.exp(log_fidelity))))

    def fit_block(self, block, *, center=None, validate=True):
        """Perform one cached local variational update on a connected block."""

        block = frozenset(block)
        if not block:
            raise ValueError("fit block must be non-empty")
        if not block.issubset(self._node_set):
            raise ValueError("fit block contains an unknown tree node")
        if len(block) > 3:
            raise ValueError("TreeFIT supports one-, two-, and three-node blocks")
        if not _is_connected(self.p, block):
            raise ValueError("fit block must be a connected subtree")
        center = self._block_center(block, preferred=center)
        self._canonicalize_for_block(block, center)
        effective = self._effective_block(block)
        factors = self._factor_block(effective, block, center)
        self._install_block(factors, block, center, validate=validate)
        self._invalidate_for_block(block)
        return {
            "block": tuple(sorted(block)),
            "center": center,
            "block_size": len(block),
            "cache": self.environment_cache_info(),
        }

    def _connected_edges(self, region):
        """Return deterministic local two-node blocks in a region."""

        region = frozenset(region)
        center = self._block_center(region)
        return sorted(
            ((node, neighbor) for node in region for neighbor in _neighbors_of(self.p, node)
             if neighbor in region and node < neighbor),
            key=lambda edge: (
                min(len(_path_of(self.p, center, edge[0])), len(_path_of(self.p, center, edge[1]))),
                edge,
            ),
        )

    def _connected_triples(self, region):
        """Return every connected three-node block in a region once."""

        region = frozenset(region)
        triples = set()
        for center in sorted(region):
            neighbors = sorted(
                neighbor for neighbor in _neighbors_of(self.p, center)
                if neighbor in region
            )
            for left, right in combinations(neighbors, 2):
                triples.add(tuple(sorted((left, center, right))))
        return sorted(triples)

    def _sweep_blocks(self, region, block_size, direction):
        """Return one inward or outward sequence of local update blocks."""

        region = frozenset(region)
        if block_size == 1:
            center = self._block_center(region)
            order = sorted(
                region,
                key=lambda node: (
                    len(_path_of(self.p, node, center)),
                    node,
                ),
                reverse=direction == "in",
            )
            return [(node,) for node in order]
        if block_size == 2:
            blocks = self._connected_edges(region)
        else:
            blocks = self._connected_triples(region)
            if not blocks:
                blocks = self._connected_edges(region)
        if not blocks:
            # A one-site gate still has a valid one-site FIT update when the
            # requested DMRG block size is two or three.
            blocks = [(node,) for node in sorted(region)]
        center = self._block_center(region)
        return sorted(
            blocks,
            key=lambda block: (
                min(len(_path_of(self.p, node, center)) for node in block),
                block,
            ),
            reverse=direction == "in",
        )

    def _active_edge_rank_targets(self, region, *, state=None):
        """Return physical rank ceilings for tree edges inside ``region``.

        The adaptive DMRG phase follows FIT's chain rule: it is governed by
        the physical Hilbert-space capacity available on either side of an
        edge, including the live state bonds where the active region meets an
        untouched exterior. It must not use the raw bond dimension of a
        factorized operator-state target, which can be larger than the actual
        state rank (for example, a CNOT acting on ``|00>``).
        """

        if self.max_bond is None:
            return None
        targets = []
        region = frozenset(region)
        state = self.p if state is None else state
        for node in sorted(region):
            for neighbor in _neighbors_of(state, node):
                if neighbor not in region or node >= neighbor:
                    continue
                sides = (
                    region.intersection(_component_of(state, node, neighbor)),
                    region.intersection(_component_of(state, neighbor, node)),
                )
                side_caps = []
                for side in sides:
                    capacity = 1
                    for member in side:
                        physical = _physical_ind(state, member)
                        if physical is not None:
                            capacity *= int(state.ind_size(physical))
                        for outside in _neighbors_of(state, member):
                            if outside not in region:
                                capacity *= int(
                                    state.ind_size(state.bond(member, outside))
                                )
                    side_caps.append(capacity)
                targets.append(((node, neighbor), min(
                    int(self.max_bond), *side_caps
                )))
        return tuple(targets)

    def _active_bonds_at_rank_targets(self, region, *, state=None):
        """Return whether every active tree edge is at its target ceiling."""

        state = self.p if state is None else state
        targets = self._active_edge_rank_targets(region, state=state)
        if targets is None:
            return False
        return all(
            int(state.ind_size(state.bond(*edge))) >= int(target)
            for edge, target in targets
        )

    def run_gate(
        self,
        region,
        n_iter=6,
        verbose=False,
        *,
        block_size=2,
        sweep_sequence="RL",
        min_iter=None,
        rtol=None,
        patience=1,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        final_one_site_sweeps=0,
    ):
        """Run cached tree FIT sweeps over a connected active region.

        ``block_size`` is the number of connected structural tree nodes in a
        local update. ``sweep_sequence`` accepts ``"RL"``/``"LR"`` (the
        MPS-compatible spellings) and maps them to inward/outward tree sweeps.
        The target remains fixed and the fitted state is updated in place.
        ``adaptive_block_sweeps`` enables the MPS-compatible larger-block
        warm-up followed by one-site refinement. ``adaptive_until_rank``
        extends that warm-up until the active physical rank ceilings are
        reached, and ``final_one_site_sweeps`` adds optional one-site polish.
        ``verbose=True`` records one normalized target fidelity per completed
        sweep, matching the chain FIT diagnostic behavior. When
        ``adaptive_block_sweeps`` is supplied, the first requested number of
        sweeps use ``block_size`` and the remaining sweeps use one-site
        refinement. ``adaptive_until_rank=True`` keeps the larger block until
        the active tree edges reach their target/max-bond ceilings, subject to
        the minimum warm-up. ``final_one_site_sweeps`` adds fixed-rank polish
        sweeps after the requested iterations.
        """

        if isinstance(region, Integral):
            region = (region,)
        region = frozenset(region)
        if not region or not region.issubset(self._node_set):
            raise ValueError("region must contain known tree nodes")
        if not _is_connected(self.p, region):
            raise ValueError("region must be a connected subtree")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be a positive integer")
        if int(block_size) not in {1, 2, 3}:
            raise ValueError("block_size must be 1, 2, or 3")
        # A short tree window follows FIT's active-span behavior: a requested
        # three-node update on a two-node region is an ordinary two-node
        # update, and a one-node region is necessarily one-site.
        block_size = min(int(block_size), len(region))
        sequence = str(sweep_sequence).strip().upper().replace("-", "")
        if sequence not in {"RL", "LR", "INOUT", "OUTIN"}:
            raise ValueError("sweep_sequence must be 'RL', 'LR', 'INOUT', or 'OUTIN'")
        directions = {
            "RL": ("in", "out"),
            "LR": ("out", "in"),
            "INOUT": ("in", "out"),
            "OUTIN": ("out", "in"),
        }[sequence]
        if min_iter is None:
            min_iter = 1
        if int(min_iter) < 1:
            raise ValueError("min_iter must be positive")
        if not isinstance(patience, Integral) or int(patience) < 1:
            raise ValueError("patience must be positive")
        if rtol is not None:
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError("rtol must be a finite non-negative number")

        adaptive_schedule = adaptive_block_sweeps is not None
        if adaptive_schedule:
            if (
                not isinstance(adaptive_block_sweeps, Integral)
                or int(adaptive_block_sweeps) < 1
            ):
                raise ValueError(
                    "adaptive_block_sweeps must be a positive integer or None"
                )
            adaptive_block_sweeps = min(int(adaptive_block_sweeps), int(n_iter))
        else:
            adaptive_block_sweeps = int(n_iter)
        if not isinstance(final_one_site_sweeps, Integral) or int(
            final_one_site_sweeps
        ) < 0:
            raise ValueError("final_one_site_sweeps must be a non-negative integer")
        final_one_site_sweeps = int(final_one_site_sweeps)
        adaptive_until_rank = bool(adaptive_until_rank)

        previous = None
        stable = 0
        self.iterations_run = 0
        self.fidelity_trace = []
        self.last_relative_change = None
        self.last_overlap = None
        self.converged = False
        self.convergence_reason = None
        self.adaptive_sweeps_run = 0
        self.one_site_sweeps_run = 0
        self.block_size_trace = []
        active_edges = tuple(
            (node, neighbor)
            for node in sorted(region)
            for neighbor in _neighbors_of(self.p, node)
            if neighbor in region and node < neighbor
        )
        adaptive_phase_done = not (
            adaptive_until_rank
            and block_size in {2, 3}
            and bool(active_edges)
        )
        if adaptive_until_rank and not adaptive_phase_done:
            adaptive_phase_done = self._active_bonds_at_rank_targets(region)
        rank_targets = (
            self._active_edge_rank_targets(region)
            if adaptive_until_rank and not adaptive_phase_done
            else None
        )

        def block_size_for_sweep(sweep_number):
            if block_size not in {2, 3}:
                return 1
            if adaptive_until_rank:
                use_block = not adaptive_phase_done
            else:
                use_block = sweep_number <= adaptive_block_sweeps
            return block_size if use_block else 1

        for iteration in range(1, int(n_iter) + 1):
            active_block_size = block_size_for_sweep(iteration)
            previous_block_size = (
                None if not self.block_size_trace else self.block_size_trace[-1]
            )
            if previous_block_size is not None and active_block_size != previous_block_size:
                # A block-to-one-site transition starts a new convergence
                # phase, just as in FIT.run_gate.
                previous = None
                stable = 0
                self.last_relative_change = None
            self.block_size_trace.append(active_block_size)
            if active_block_size == 1:
                self.one_site_sweeps_run += 1
            else:
                self.adaptive_sweeps_run += 1
            for direction in directions:
                for block in self._sweep_blocks(region, active_block_size, direction):
                    self.fit_block(block, validate=False)
            self.iterations_run = iteration
            self.final_direction = directions[-1]
            self.final_center_site = getattr(self.p, "orthogonality_center", None)
            self.p.validate(check_canonical=True)
            norm = self._network_norm(self.p)
            self.last_norm = norm
            if verbose:
                self.fidelity_trace.append(self._normalized_overlap_fidelity())
            if rtol is not None:
                try:
                    self.last_overlap = self._global_overlap()
                    objective = max(
                        0.0, 1.0 - self._normalized_overlap_fidelity()
                    )
                except Exception:  # pragma: no cover - backend diagnostic fallback
                    objective = norm
                if previous is not None:
                    relative_change = abs(objective - previous) / max(
                        abs(previous), 1e-300
                    )
                    self.last_relative_change = float(relative_change)
                    warmup_incomplete = (
                        adaptive_schedule
                        and active_block_size > 1
                        and iteration < adaptive_block_sweeps
                    )
                    warmup_finished_with_refinement = (
                        adaptive_schedule
                        and active_block_size > 1
                        and iteration == adaptive_block_sweeps
                        and iteration < int(n_iter)
                    )
                    adaptive_rank_incomplete = (
                        adaptive_until_rank
                        and not adaptive_phase_done
                        and active_block_size > 1
                    )
                    if (
                        iteration >= int(min_iter)
                        and relative_change <= float(rtol)
                        and not (
                            warmup_incomplete
                            or warmup_finished_with_refinement
                            or adaptive_rank_incomplete
                        )
                    ):
                        stable += 1
                    else:
                        stable = 0
                    if stable >= int(patience):
                        self.converged = True
                        self.convergence_reason = "rtol"
                        break
                if warmup_finished_with_refinement or adaptive_rank_incomplete:
                    previous = None
                    stable = 0
                    self.last_relative_change = None
                else:
                    previous = objective
            else:
                previous = norm

            if (
                adaptive_until_rank
                and not adaptive_phase_done
                and active_block_size > 1
                and iteration >= adaptive_block_sweeps
                and rank_targets is not None
                and all(
                    int(self.p.ind_size(self.p.bond(*edge))) >= int(target)
                    for edge, target in rank_targets
                )
            ):
                adaptive_phase_done = True
                previous = None
                stable = 0
                self.last_relative_change = None
        else:
            self.convergence_reason = "max_iter"

        if (
            not self.converged
            and final_one_site_sweeps > 0
            and block_size in {2, 3}
            and len(region) >= 3
        ):
            for _ in range(final_one_site_sweeps):
                self.block_size_trace.append(1)
                self.one_site_sweeps_run += 1
                for direction in directions:
                    for block in self._sweep_blocks(region, 1, direction):
                        self.fit_block(block, validate=False)
                self.iterations_run += 1
                self.final_direction = directions[-1]
                self.final_center_site = getattr(
                    self.p, "orthogonality_center", None
                )
                self.p.validate(check_canonical=True)
                self.last_norm = self._network_norm(self.p)
                if verbose:
                    self.fidelity_trace.append(self._normalized_overlap_fidelity())
        if self.converged is False and self.convergence_reason is None:
            self.convergence_reason = "max_iter"
        return self

    def run_eff(
        self,
        n_iter=6,
        verbose=False,
        *,
        block_size=1,
        sweep_sequence="RL",
        min_iter=None,
        rtol=None,
        patience=1,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        final_one_site_sweeps=0,
    ):
        """Fit the complete tree using the cached local environment engine."""

        return self.run_gate(
            self.nodes,
            n_iter=n_iter,
            block_size=block_size,
            sweep_sequence=sweep_sequence,
            min_iter=min_iter,
            rtol=rtol,
            patience=patience,
            verbose=verbose,
            adaptive_block_sweeps=adaptive_block_sweeps,
            adaptive_until_rank=adaptive_until_rank,
            final_one_site_sweeps=final_one_site_sweeps,
        )

    def run(
        self,
        n_iter=6,
        verbose=False,
        *,
        block_size=1,
        sweep_sequence="RL",
        min_iter=None,
        rtol=None,
        patience=1,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        final_one_site_sweeps=0,
    ):
        """Run a complete-tree FIT sweep with the chain-compatible API.

        Unlike the chain implementation there is no useful tree analogue of
        a left-to-right full-contraction reference path: the directed-message
        engine is the natural full-tree update. ``run`` therefore intentionally
        delegates to :meth:`run_eff`, while keeping FIT's positional
        ``n_iter``/``verbose`` call shape.
        """

        return self.run_eff(
            n_iter=n_iter,
            block_size=block_size,
            sweep_sequence=sweep_sequence,
            min_iter=min_iter,
            rtol=rtol,
            patience=patience,
            verbose=verbose,
            adaptive_block_sweeps=adaptive_block_sweeps,
            adaptive_until_rank=adaptive_until_rank,
            final_one_site_sweeps=final_one_site_sweeps,
        )

    def fit_diagnostics(self, *, overlap=False):
        """Return a copy-safe summary of the latest tree FIT run."""

        result = {
            "iterations": int(self.iterations_run),
            "converged": bool(self.converged),
            "convergence_reason": self.convergence_reason,
            "relative_change": self.last_relative_change,
            "final_norm": self.last_norm,
            "adaptive_sweeps": int(self.adaptive_sweeps_run),
            "one_site_refinement_sweeps": int(self.one_site_sweeps_run),
            "block_size_trace": tuple(self.block_size_trace),
            "target_layout": self.target_layout,
            "cache": self.environment_cache_info(),
        }
        if overlap:
            try:
                overlap_mantissa, overlap_exponent = (
                    self._global_overlap_stripped()
                )
                overlap_value = _scale_stripped(
                    overlap_mantissa, overlap_exponent
                )
                fidelity = self._normalized_overlap_fidelity()
                result.update({
                    "overlap": overlap_value,
                    "local_fidelity": float(fidelity),
                    "local_infidelity": float(max(0.0, 1.0 - fidelity)),
                    "overlap_mantissa": overlap_mantissa,
                    "overlap_exponent": overlap_exponent,
                })
            except Exception as exc:  # diagnostics must not invalidate a fit
                result.update({
                    "overlap": None,
                    "local_fidelity": None,
                    "local_infidelity": None,
                    "overlap_error": str(exc),
                })
        return result
