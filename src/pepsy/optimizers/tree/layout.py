"""Tree-structure search for :class:`TreeOptimizer` gate-stream replay.

The paper *Simulating quantum circuits using tree tensor networks*
(Seitz, Medina, Cruz, Huang, Mendl; Quantum 7, 964, 2023; arXiv:2206.01000)
first fixes a rooted tree adapted to the entanglement the circuit is expected
to generate, then applies gates to it.

:class:`TreeLayoutFinder` builds that structure from the two-qubit connectivity
of a bundled gate stream.  It reuses the interaction-graph and recursive
spectral-bisection machinery written for the MPS layout finder
(:mod:`pepsy.optimizers.mps.layout`); where the MPS finder *flattens* the
bisection recursion into a 1D order, the tree finder *keeps* the recursion as
the rooted tree structure.  Strongly coupled qubits end up as nearby leaves,
minimising the tree-path length that two-qubit gates must thread across.

The structure is not restricted to strictly-binary trees.  Internal nodes may
have any arity: ``max_arity`` gives flatter ``k``-ary trees (shallower
geodesics), while ``structure="adaptive"`` reads the gate-stream interaction
graph and lets each level branch into as many children as it has strongly
coupled communities.  By default the finder *searches* a small set of
candidate arities (``max_arity=(2, 3, 4)``) and keeps the objective-best plan;
pass a scalar ``max_arity=2`` to opt back into a single fixed binary tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import autoray as ar
import numpy as np

from ..mps.layout import (
    _gate_stream_adjacency,
    _gate_stream_event_weights,
    _gate_stream_pair_weights,
    _gate_stream_spectral_order,
    _normalize_layout_gate_queue,
    _normalize_layout_support,
    _operator_schmidt_rank_bound,
    _operator_schmidt_rank_info as _mps_operator_schmidt_rank_info,
    _normalize_weight_mode,
)
from ..mps.optimizer import _control_event_parts as _mps_control_event_parts
from .._layout_visualization import (
    add_order_colorbar,
    coordinate_lattice_edge_keys,
    coordinate_lattice_edges,
    event_color,
    finish_schematic_axes,
    matplotlib_modules,
    resolve_site_coords,
    scale_color,
)

__all__ = ["TreePlan", "TreeLayoutFinder"]

_DEFAULT_MAX_ARITY = object()
_DEFAULT_CHI = object()
_DEFAULT_ORDER = object()
_DEFAULT_SEARCH_OPTION = object()
_DEFAULT_SCALE_MARKERS = ("o",)


def _looks_like_tree_tensor_network(value):
    """Identify a TTN input without importing ``ttn`` (avoids a cycle)."""
    return (
        getattr(value, "plan", None) is not None
        and getattr(value, "tensor_map", None) is not None
    )


def _normalize_hybrid_weights(weights):
    """Validate path / peak-load / total-load hybrid objective weights."""
    if weights is None:
        values = (1.0, 1.0, 0.25)
    elif isinstance(weights, Mapping):
        aliases = {
            "path": "path",
            "max_edge_load": "max_edge_load",
            "peak_load": "max_edge_load",
            "total_edge_load": "total_edge_load",
            "total_load": "total_edge_load",
        }
        normalized = {}
        for key, value in weights.items():
            name = aliases.get(str(key).replace("-", "_").strip().lower())
            if name is None:
                raise ValueError(
                    "hybrid_weights keys must be 'path', 'max_edge_load', "
                    "or 'total_edge_load'."
                )
            if name in normalized:
                raise ValueError(f"duplicate hybrid weight {name!r}.")
            normalized[name] = value
        values = tuple(
            normalized.get(name, 0.0)
            for name in ("path", "max_edge_load", "total_edge_load")
        )
    else:
        try:
            values = tuple(weights)
        except TypeError as exc:
            raise ValueError(
                "hybrid_weights must be a three-item sequence, mapping, or None."
            ) from exc
        if len(values) != 3:
            raise ValueError(
                "hybrid_weights must contain path, max-edge-load, and "
                "total-edge-load weights."
            )
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("hybrid_weights must be finite non-negative numbers.") from exc
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("hybrid_weights must be finite non-negative numbers.")
    if not any(values):
        raise ValueError("at least one hybrid weight must be positive.")
    return values


def _normalize_layout_refinement(refine):
    """Normalize an optional deterministic fixed-plan refinement mode."""
    if refine is None or refine is False:
        return None
    name = str(refine).replace("-", "_").strip().lower()
    aliases = {
        "adjacent": "greedy",
        "adjacent_swaps": "greedy",
        "local": "greedy",
    }
    name = aliases.get(name, name)
    if name != "greedy":
        raise ValueError("refine must be None or 'greedy'.")
    return name


def _normalize_layout_search(search):
    """Normalize an optional offline fixed-plan search mode."""
    if search is None or search is False:
        return None
    name = str(search).replace("-", "_").strip().lower()
    aliases = {"ng": "nevergrad", "never_grad": "nevergrad"}
    name = aliases.get(name, name)
    if name != "nevergrad":
        raise ValueError("search must be None or 'nevergrad'.")
    return name


def _normalize_layout_order(order):
    """Normalize the optional high-quality layout mode."""
    if order is None:
        return None
    name = str(order).replace("-", "_").strip().lower()
    aliases = {
        "auto": "quality",
        "best": "quality",
        "best_quality": "quality",
    }
    name = aliases.get(name, name)
    if name != "quality":
        raise ValueError("order must be None or 'quality'.")
    return name


def _nevergrad_available():
    """Return whether the optional Nevergrad dependency can be imported."""
    try:
        import nevergrad  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return True


def _validate_search_budget(value, name):
    """Validate a positive bounded layout-search evaluation budget."""
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _safe_exp2(value):
    """Return ``2**value`` without emitting overflow warnings."""
    if value > np.log2(np.finfo(float).max):
        return float("inf")
    return float(np.exp2(value))


def _normalize_layout_objective(objective):
    """Normalize the tree-layout objective name."""
    name = str(objective).replace("-", "_").strip().lower()
    aliases = {
        "distance": "path",
        "path_length": "path",
        "edge": "congestion",
        "edge_load": "congestion",
        "bond": "congestion",
        "bond_load": "congestion",
        "combined": "hybrid",
        "compress": "compression",
        "accuracy": "compression",
        "bond_growth": "compression",
    }
    name = aliases.get(name, name)
    if name not in {"path", "congestion", "hybrid", "compression"}:
        raise ValueError(
            f"Unknown tree layout objective {objective!r}. "
            "Expected 'path', 'congestion', 'compression', or 'hybrid'."
        )
    return name


def _operator_schmidt_rank(payload, support, left_support):
    """Return an operator-Schmidt rank across a support bipartition."""
    support = tuple(support)
    left_support = tuple(left_support)
    left_set = set(left_support)
    if not left_set or left_set == set(support):
        return 1
    default_bound = _operator_schmidt_rank_bound(support, left_support)
    try:
        array = ar.to_numpy(payload)
    except Exception:
        return default_bound
    if array.size != 4 ** len(support):
        return default_bound
    try:
        array = array.reshape((2,) * (2 * len(support)))
        positions = {site: pos for pos, site in enumerate(support)}
        left_positions = [positions[site] for site in left_support]
        right_positions = [
            positions[site] for site in support if site not in left_set
        ]
        axes = (
            left_positions
            + [len(support) + pos for pos in left_positions]
            + right_positions
            + [len(support) + pos for pos in right_positions]
        )
        matrix = array.transpose(axes).reshape(
            4 ** len(left_positions),
            4 ** len(right_positions),
        )
        return max(1, int(np.linalg.matrix_rank(matrix)))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return default_bound


def _submpo_schmidt_rank_bound(payload, support, left_support):
    """Return an MPO-bond upper bound without calling ``to_dense``.

    An MPO's operator Schmidt rank across a site bipartition is bounded by the
    product of the virtual MPO bonds crossing that bipartition.  This is a
    conservative diagnostic, but unlike lowering an MPO to a dense matrix it
    remains cheap for wide supports.  ``None`` means that ``payload`` does not
    expose the Quimb MPO site interface.
    """
    gen_sites = getattr(payload, "gen_sites_present", None)
    site_tag = getattr(payload, "site_tag", None)
    tag_map = getattr(payload, "tag_map", None)
    tensor_map = getattr(payload, "tensor_map", None)
    if not all((callable(gen_sites), callable(site_tag),
                tag_map is not None, tensor_map is not None)):
        return None
    try:
        present = tuple(gen_sites())
        support = tuple(support)
        if set(present) != set(support):
            return None
        tensors = []
        for site in present:
            tids = tuple(tag_map[site_tag(site)])
            if len(tids) != 1:
                return None
            tensors.append(tensor_map[tids[0]])
        left = set(left_support)
        if not left or left == set(support):
            return 1
        rank = 1
        for left_site, right_site, left_tensor, right_tensor in zip(
            present, present[1:], tensors, tensors[1:]
        ):
            if (left_site in left) == (right_site in left):
                continue
            shared = set(left_tensor.inds).intersection(right_tensor.inds)
            if len(shared) != 1:
                return None
            rank *= int(payload.ind_size(next(iter(shared))))
        return max(1, int(rank))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _validate_chi(chi):
    """Coerce and validate an optional ``chi`` selection budget."""
    if chi is None:
        return None
    chi = int(chi)
    if chi < 1:
        raise ValueError("chi must be a positive integer.")
    return chi


def _normalize_arity_candidates(max_arity):
    """Return ``(representative_arity, candidates)`` from a ``max_arity`` arg.

    ``max_arity`` may be a single int (a fixed arity), ``None`` (unbounded), or
    an iterable of candidate arities to *search* (the finder default
    ``(2, 3, 4)``).  ``candidates`` is ``None`` unless a search set was given;
    the representative single arity is what the legacy single-plan builders use
    and is the first concrete candidate.
    """
    if max_arity is None:
        return None, None
    if isinstance(max_arity, Integral):
        return int(max_arity), None
    if isinstance(max_arity, (str, bytes)):
        return int(max_arity), None
    if hasattr(max_arity, "__iter__"):
        cand = []
        for a in max_arity:
            key = None if a is None else int(a)
            if key is not None and key < 2:
                raise ValueError("arity candidates must be >= 2 or None.")
            if key not in cand:
                cand.append(key)
        if not cand:
            raise ValueError("max_arity iterable must be non-empty.")
        representative = next((a for a in cand if a is not None), None)
        return representative, tuple(cand)
    return int(max_arity), None



def _chi_cut_fields(plan, chi):
    """Return ``{max_bond_cut, chi_overflow, exact_at_chi}`` for ``plan``.

    ``max_bond_cut`` is the widest qubit bipartition any bond induces.  With a
    finite ``chi`` the structure can hold an arbitrary state exactly only when
    ``2 ** max_bond_cut <= chi``; ``chi_overflow`` is how many qubits the widest
    bond exceeds ``log2(chi)`` (0 when the structure is exact at ``chi``).
    """
    mbc = plan.max_bond_cut()
    fields = {"max_bond_cut": mbc}
    if chi is not None:
        log_chi = float(np.log2(chi))
        fields["chi_overflow"] = max(0.0, mbc - log_chi)
        fields["exact_at_chi"] = mbc <= log_chi
    return fields


def _tree_node_scales(plan):
    """Return hierarchical scales, with leaves at scale zero."""
    scales = {}

    def visit(node):
        if node in scales:
            return scales[node]
        children = tuple(plan.children.get(node, ()))
        if not children:
            scale = 0
        else:
            scale = 1 + max(visit(child) for child in children)
        scales[node] = scale
        return scale

    visit(plan.root)
    return scales


class TreePlan:
    """A rooted tree over ``n`` qubits (any internal-node arity).

    Nodes are integer ids. Leaves map one-to-one to qubits. Optionally, one
    additional qubit can be carried by the structural root via ``root_qubit``;
    this gives a binary top tensor two child bonds plus one open physical leg.
    Other internal nodes carry no physical qubit. A strictly-binary tree (every
    internal node with two children) is the common default, but the structure
    supports arbitrary arity so a level can branch into as many subtrees as the
    gate stream suggests. The plan is a pure structure description: it carries
    no tensor data and is consumed by
    :class:`~pepsy.optimizers.tree.TreeOptimizer` to build the tree tensor
    network.
    """

    def __init__(
        self, root, children, parent, qubit_of_leaf, *, root_qubit=None
    ):
        self.root = root
        self.children = dict(children)
        self.parent = dict(parent)
        self.qubit_of_leaf = dict(qubit_of_leaf)
        self.leaf_of_qubit = {q: nid for nid, q in self.qubit_of_leaf.items()}
        self.root_qubit = (
            None if root_qubit is None else int(root_qubit)
        )
        if self.root_qubit is not None and self.root in self.qubit_of_leaf:
            raise ValueError(
                "the root cannot carry both a leaf qubit and root_qubit; "
                "insert a unary structural root above the leaf."
            )
        self.qubit_of_node = dict(self.qubit_of_leaf)
        if self.root_qubit is not None:
            self.qubit_of_node[self.root] = self.root_qubit
        self.node_of_qubit = {
            q: nid for nid, q in self.qubit_of_node.items()
        }
        self.n = len(self.node_of_qubit)
        self._path_cache = {}

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_order(cls, order, *, weights=None, structure="quality",
                   max_arity=2, community_frac=0.35, star_frac=0.75,
                   dense_max=512, root_qubit=None):
        """Build a rooted tree by recursive partition of ``order``.

        Parameters
        ----------
        order : sequence of int
            The qubit labels to place as leaves. When ``root_qubit`` is given,
            ``order`` contains every other qubit and the combined labels must
            still be ``0..n-1``.
        weights : mapping, optional
            Unordered ``(qi, qj) -> weight`` interaction weights.  Used to
            spectrally reorder each recursion level (``structure="quality"``)
            and to detect communities (``structure="adaptive"``).
        structure : {"quality", "balanced", "adaptive"}
            ``"quality"`` spectrally (Fiedler) reorders each level before
            splitting; ``"balanced"`` splits the given ``order`` directly;
            ``"adaptive"`` partitions each level into strongly coupled
            communities of the induced interaction graph so the arity of a node
            follows the gate connectivity, and collapses a densely coupled
            block (a near-clique) into a single flat *star* node.  All three
            respect ``max_arity``.
        max_arity : int or None
            Maximum number of children per internal node.  ``2`` (default)
            reproduces the strictly-binary tree; larger values give flatter
            ``k``-ary trees with shorter geodesics; ``None`` leaves the arity
            unbounded (``"adaptive"`` may then emit wide star nodes).
        community_frac : float
            For ``structure="adaptive"``: an induced edge is treated as a strong
            intra-community link when its weight is at least ``community_frac``
            times the largest induced edge weight at that level.
        star_frac : float
            For ``structure="adaptive"``: when a block is a single strong
            community whose fraction of present strong edges is at least
            ``star_frac`` (a near-clique), it becomes a flat star of leaves
            (all pairwise geodesics length two) instead of being bisected.
        dense_max : int
            Maximum subsystem size for dense spectral reordering.
        root_qubit : int, optional
            Qubit label carried by the top tensor rather than a leaf.
        """
        order = list(order)
        if not order and root_qubit is None:
            raise ValueError("order must contain at least one qubit.")
        try:
            order = [int(q) for q in order]
        except (TypeError, ValueError) as exc:
            raise ValueError("order must contain integer qubit labels.") from exc
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError("root_qubit must be an integer or None.") from exc
        all_qubits = order + ([] if root_qubit is None else [root_qubit])
        if sorted(all_qubits) != list(range(len(all_qubits))):
            raise ValueError(
                "leaf order plus root_qubit must be a permutation of "
                "qubit labels 0..n-1."
            )
        if structure not in {"quality", "balanced", "adaptive"}:
            raise ValueError(
                "structure must be 'quality', 'balanced', or 'adaptive'."
            )
        if max_arity is not None:
            max_arity = int(max_arity)
            if max_arity < 2:
                raise ValueError("max_arity must be >= 2 (or None).")
        counter = [0]
        children = {}
        parent = {}
        qubit_of_leaf = {}

        def new_node():
            nid = counter[0]
            counter[0] += 1
            return nid

        def induced(nodes):
            if not weights:
                return {}
            node_set = set(nodes)
            return {
                edge: w
                for edge, w in weights.items()
                if edge[0] in node_set and edge[1] in node_set
            }

        def make_leaf(q):
            nid = new_node()
            children[nid] = ()
            qubit_of_leaf[nid] = q
            return nid

        def make_internal(child_ids):
            nid = new_node()
            children[nid] = tuple(child_ids)
            for c in child_ids:
                parent[c] = nid
            return nid

        def kary_split(qs):
            """Split ``qs`` into up to ``max_arity`` contiguous balanced parts.

            Cut points use ``floor(i * L / k)`` so the two-way case reproduces
            the previous ``mid = len(qs) // 2`` bisection exactly.
            """
            length = len(qs)
            k = length if max_arity is None else max_arity
            k = min(k, length)
            if k <= 1:
                return [qs]
            cuts = [length * i // k for i in range(k + 1)]
            groups = [qs[cuts[i]:cuts[i + 1]] for i in range(k)]
            return [g for g in groups if g]

        def strong_adjacency(qs):
            """Return ``(adjacency, threshold)`` for the induced graph or None."""
            sub = induced(qs)
            if not sub:
                return None
            max_w = max(sub.values())
            if max_w <= 0.0:
                return None
            adj = _gate_stream_adjacency(qs, sub)
            return adj, float(community_frac) * max_w

        def communities(qs):
            """Return strongly coupled communities of ``qs`` or ``None``."""
            info = strong_adjacency(qs)
            if info is None:
                return None
            adj, thresh = info
            rank = {q: i for i, q in enumerate(qs)}
            seen = set()
            comps = []
            for start in qs:
                if start in seen:
                    continue
                stack = [start]
                seen.add(start)
                comp = []
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nb, w in adj[cur].items():
                        if nb not in seen and w >= thresh:
                            seen.add(nb)
                            stack.append(nb)
                comps.append(sorted(comp, key=lambda x: rank[x]))
            comps.sort(key=lambda c: rank[c[0]])
            return comps

        def is_near_clique(qs):
            """Return ``True`` when strong edges nearly fully connect ``qs``."""
            info = strong_adjacency(qs)
            if info is None:
                return False
            adj, thresh = info
            m = len(qs)
            if m < 3:
                return False
            strong = 0
            for i, a in enumerate(qs):
                for b in qs[i + 1:]:
                    if adj[a].get(b, 0.0) >= thresh:
                        strong += 1
            total = m * (m - 1) // 2
            return total > 0 and strong / total >= float(star_frac)

        def split(qs):
            """Return the child qubit-groups for the internal node over ``qs``."""
            groups = None
            if structure == "adaptive":
                comps = communities(qs)
                if comps is not None and len(comps) >= 2:
                    if max_arity is None or len(comps) <= max_arity:
                        groups = comps
                    # else: too many communities for the arity cap; fall back to
                    # a spectral k-ary split (deeper recursion still resolves
                    # communities inside each part).
                elif (max_arity is None or len(qs) <= max_arity) \
                        and is_near_clique(qs):
                    # A densely coupled block is flattest as a star of leaves.
                    groups = [[q] for q in qs]
            if groups is None:
                qs2 = qs
                if structure in ("quality", "adaptive"):
                    spectral = _gate_stream_spectral_order(
                        qs, induced(qs), dense_max=dense_max
                    )
                    if spectral:
                        qs2 = spectral
                groups = kary_split(qs2)
            return groups

        def build(qs):
            qs = list(qs)
            if len(qs) == 1:
                return make_leaf(qs[0])
            groups = split(qs)
            if len(groups) < 2:
                # Degenerate split (e.g. all mass in one part): force a split so
                # recursion always makes progress.
                mid = max(1, len(qs) // 2)
                groups = [qs[:mid], qs[mid:]]
            child_ids = [build(g) for g in groups]
            return make_internal(child_ids)

        if order:
            root = build(order)
            if root_qubit is not None and root in qubit_of_leaf:
                # With one non-root qubit, ``build`` returns that physical
                # leaf itself. The top qubit needs its own tensor, so insert a
                # unary structural root rather than putting two physical legs
                # on the same node.
                root = make_internal((root,))
        else:
            root = new_node()
            children[root] = ()
        return cls(
            root,
            children,
            parent,
            qubit_of_leaf,
            root_qubit=root_qubit,
        )

    @classmethod
    def from_children(
        cls, children, qubit_of_leaf, *, root=None, root_qubit=None
    ):
        """Build and validate a :class:`TreePlan` from an explicit tree.

        This is the general entry point for arbitrary (non-binary) trees: a
        caller or a custom layout strategy supplies the ``children`` map and the
        leaf-to-qubit assignment, and this validates that they describe a single
        rooted tree covering qubits ``0..n-1`` exactly once.

        Parameters
        ----------
        children : mapping
            ``node_id -> tuple(child_ids)``.  Leaves map to an empty tuple.
        qubit_of_leaf : mapping
            ``leaf_id -> qubit`` for every leaf node.
        root : int, optional
            The root node id.  Inferred as the unique parent-less node when
            omitted.
        root_qubit : int, optional
            Qubit label carried by ``root`` rather than by a leaf.
        """
        children = {int(k): tuple(int(c) for c in v)
                    for k, v in children.items()}
        qubit_of_leaf = {int(k): int(q) for k, q in qubit_of_leaf.items()}

        parent = {}
        for nid, ch in children.items():
            for c in ch:
                if c in parent:
                    raise ValueError(f"node {c} has more than one parent")
                if c not in children:
                    raise ValueError(
                        f"child {c} of node {nid} is not a declared node"
                    )
                parent[c] = nid

        roots = [nid for nid in children if nid not in parent]
        if root is None:
            if len(roots) != 1:
                raise ValueError(
                    f"expected exactly one root, found {sorted(roots)}"
                )
            root = roots[0]
        else:
            root = int(root)
            if root not in children or root in parent:
                raise ValueError(f"invalid root {root}")
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError("root_qubit must be an integer or None.") from exc

        leaves = set()
        for nid, ch in children.items():
            if ch:
                if nid in qubit_of_leaf:
                    raise ValueError(
                        f"internal node {nid} must not have a qubit"
                    )
            else:
                leaves.add(nid)
                if (
                    nid not in qubit_of_leaf
                    and not (nid == root and root_qubit is not None)
                ):
                    raise ValueError(f"leaf node {nid} is missing a qubit")
        expected_leaf_nodes = (
            leaves - {root}
            if root_qubit is not None and not children[root]
            else leaves
        )
        if set(qubit_of_leaf) != expected_leaf_nodes:
            raise ValueError(
                "qubit_of_leaf must map exactly the leaf nodes"
            )
        if root_qubit is not None and root in qubit_of_leaf:
            raise ValueError("the root cannot carry both a leaf and root qubit")
        qs = sorted(
            [
                *qubit_of_leaf.values(),
                *([] if root_qubit is None else [root_qubit]),
            ]
        )
        if qs != list(range(len(qs))):
            raise ValueError(
                "leaf qubits plus root_qubit must be 0..n-1 without repeats"
            )

        seen = set()
        stack = [root]
        while stack:
            x = stack.pop()
            if x in seen:
                raise ValueError("cycle detected in tree")
            seen.add(x)
            stack.extend(children[x])
        if seen != set(children):
            unreached = set(children) - seen
            raise ValueError(
                f"nodes not reachable from root {root}: {sorted(unreached)}"
            )
        return cls(
            root,
            children,
            parent,
            qubit_of_leaf,
            root_qubit=root_qubit,
        )

    #: Fixed number of legs on the top tensor of a :meth:`build_layered` tree.
    LAYERED_ROOT_ARITY = 3

    @classmethod
    def build_layered(cls, order, *, block_size=4, root_qubit=None):
        """Build a fixed-structure layered tree with a ternary top tensor.

        The structure is fixed; only ``block_size`` is tunable:

        * **First layer** (leaf-parent "blocking" nodes): each node groups
          ``block_size`` consecutive qubits from ``order`` into one virtual
          bond.  This is the only choosable layer.
        * **Middle layers**: strictly binary (two bonds in, one out).
        * **Top tensor (root)**: always :attr:`LAYERED_ROOT_ARITY` (three)
          children when there are at least three blocks; fewer only in the
          degenerate small-``n`` case where three blocks do not exist.

        Parameters
        ----------
        order : sequence of int
            Leaf-qubit labels in the desired spatial order. Strongly coupled
            qubits should be consecutive so they land in the same block; use
            :meth:`TreeLayoutFinder.qubit_order` to obtain an
            entanglement-adapted ordering, or
            :meth:`TreeLayoutFinder.recommend_layered` to also search
            ``block_size``. Together with an optional ``root_qubit``, the
            labels must cover ``0..n-1``.
        block_size : int
            Number of physical qubits per leaf-parent node. Default 4.
        root_qubit : int, optional
            Qubit label carried by the top tensor rather than a leaf.
        """
        order = list(order)
        if not order and root_qubit is None:
            raise ValueError("order must be non-empty.")
        order = [int(q) for q in order]
        if root_qubit is not None:
            root_qubit = int(root_qubit)
        all_qubits = order + ([] if root_qubit is None else [root_qubit])
        n = len(order)
        if sorted(all_qubits) != list(range(len(all_qubits))):
            raise ValueError(
                "leaf order plus root_qubit must be a permutation of 0..n-1."
            )
        if not isinstance(block_size, Integral):
            raise ValueError("block_size must be an integer >= 1.")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        counter = [0]
        children_map = {}
        qubit_of_leaf = {}

        def new_node():
            nid = counter[0]
            counter[0] += 1
            return nid

        # Leaves: one node per qubit in the given order.
        leaf_ids = []
        for q in order:
            nid = new_node()
            children_map[nid] = ()
            qubit_of_leaf[nid] = q
            leaf_ids.append(nid)

        if not leaf_ids:
            root_nid = new_node()
            children_map[root_nid] = ()
            return cls.from_children(
                children_map,
                qubit_of_leaf,
                root=root_nid,
                root_qubit=root_qubit,
            )

        # First layer: group block_size leaves into one blocking node.
        # A single-leaf chunk skips the parent and uses the leaf directly.
        block_nodes = []
        for start in range(0, n, block_size):
            chunk = leaf_ids[start: start + block_size]
            if len(chunk) == 1:
                block_nodes.append(chunk[0])
            else:
                nid = new_node()
                children_map[nid] = tuple(chunk)
                block_nodes.append(nid)

        # Middle layers: binary tree over block_nodes.
        def binary_subtree(nodes):
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            left = binary_subtree(nodes[:mid])
            right = binary_subtree(nodes[mid:])
            nid = new_node()
            children_map[nid] = (left, right)
            return nid

        # Top tensor: fixed ternary root (or fewer only when < 3 blocks exist).
        num_blocks = len(block_nodes)
        if num_blocks == 1:
            # The blocking node is already a valid root.  In particular, do
            # not add a unary wrapper for n=1 or n <= block_size: it adds a
            # useless bond and makes the fixed layered family less efficient.
            # The exception is a physical root over one physical leaf: those
            # two qubits require distinct tensors joined by one bond.
            root_nid = block_nodes[0]
            if root_qubit is not None and not children_map[root_nid]:
                child = root_nid
                root_nid = new_node()
                children_map[root_nid] = (child,)
            return cls.from_children(
                children_map,
                qubit_of_leaf,
                root=root_nid,
                root_qubit=root_qubit,
            )
        root_arity = min(cls.LAYERED_ROOT_ARITY, num_blocks)
        if num_blocks <= root_arity:
            # Fewer blocks than the target arity: root takes them all directly.
            root_nid = new_node()
            children_map[root_nid] = tuple(block_nodes)
        else:
            # Split blocks into root_arity contiguous groups; each group is a
            # binary middle subtree whose root becomes a direct child of the
            # top tensor.
            root_children = []
            for i in range(root_arity):
                start = num_blocks * i // root_arity
                end = num_blocks * (i + 1) // root_arity
                root_children.append(binary_subtree(block_nodes[start:end]))
            root_nid = new_node()
            children_map[root_nid] = tuple(root_children)

        return cls.from_children(
            children_map,
            qubit_of_leaf,
            root=root_nid,
            root_qubit=root_qubit,
        )

    # -- queries --------------------------------------------------------------

    def nodes(self):
        """Return all node ids (leaves and internal nodes)."""
        return list(self.children.keys())

    def leaves(self):
        """Return the leaf node ids."""
        return list(self.qubit_of_leaf.keys())

    def is_leaf(self, nid):
        return len(self.children.get(nid, ())) == 0

    def max_arity(self):
        """Return the largest number of children over all internal nodes."""
        return max((len(ch) for ch in self.children.values()), default=0)

    def is_binary(self):
        """Return ``True`` when every internal node has exactly two children."""
        return all(len(ch) in (0, 2) for ch in self.children.values())

    def max_bond_cut(self):
        """Return the largest qubit bipartition induced by any tree bond.

        Every parent-child bond splits the qubits into the child's subtree
        (``k`` qubits) and the rest (``n - k``).  The Schmidt rank that bond can
        carry is bounded by ``2 ** min(k, n - k)``, so this maximum
        ``min(k, n - k)`` over all bonds is a purely structural, ``chi``-free
        accuracy ceiling: the tree can represent an *arbitrary* state exactly
        only when ``chi >= 2 ** max_bond_cut``.  A structure whose
        ``max_bond_cut`` exceeds ``log2(chi)`` must truncate at its widest bond
        regardless of the gate stream.
        """
        # One post-order pass to size every subtree, then reduce over bonds.
        visit = []
        stack = [self.root]
        while stack:
            x = stack.pop()
            visit.append(x)
            stack.extend(self.children[x])
        size = {}
        for x in reversed(visit):
            ch = self.children[x]
            local = 1 if x in self.qubit_of_node else 0
            size[x] = local + sum(size[c] for c in ch)
        best = 0
        for x, s in size.items():
            if x == self.root:
                continue
            best = max(best, min(s, self.n - s))
        return best

    def node_path(self, a, b):
        """Return the node id path from node ``a`` to node ``b`` (inclusive)."""
        if a not in self.children or b not in self.children:
            raise ValueError(f"nodes {a!r} and {b!r} must belong to the tree")
        cached = self._path_cache.get((a, b))
        if cached is not None:
            return list(cached)
        if a == b:
            result = [a]
            self._path_cache[(a, b)] = tuple(result)
            return result
        ancestors = []
        x = a
        while x is not None:
            ancestors.append(x)
            x = self.parent.get(x)
        depth = {v: i for i, v in enumerate(ancestors)}
        tail = []
        x = b
        while x not in depth:
            tail.append(x)
            x = self.parent.get(x)
            if x is None:
                raise ValueError("nodes are not in the same tree")
        lca = x
        result = ancestors[: depth[lca] + 1] + list(reversed(tail))
        self._path_cache[(a, b)] = tuple(result)
        return result

    def subtree_qubit_masks(self):
        """Return an integer bit mask of qubits below every node.

        Integer masks make repeated layout and preflight cut tests much cheaper
        than rebuilding Python ``set`` objects for every edge. Python integers
        remain exact for arbitrary qubit counts.
        """
        visit = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            visit.append(node)
            stack.extend(self.children[node])
        masks = {}
        for node in reversed(visit):
            mask = 0
            q = self.qubit_of_node.get(node)
            if q is not None:
                mask |= 1 << q
            for child in self.children[node]:
                mask |= masks[child]
            masks[node] = mask
        return masks

    def tree_distance(self, qa, qb):
        """Return the node-path length between physical qubits ``qa`` and ``qb``."""
        na = self.node_of_qubit[qa]
        nb = self.node_of_qubit[qb]
        return len(self.node_path(na, nb)) - 1

    def remove_qubit(self, q):
        """Return a plan with physical qubit ``q`` removed and labels compacted."""
        q = int(q)
        if q == self.root_qubit:
            if self.n <= 1:
                raise ValueError("cannot remove the only qubit from a tree.")
            qubit_of_leaf = {
                node: old_q - 1 if old_q > q else old_q
                for node, old_q in self.qubit_of_leaf.items()
            }
            return type(self).from_children(
                self.children,
                qubit_of_leaf,
                root=self.root,
                root_qubit=None,
            )
        return self.remove_leaf(q)

    def remove_leaf(self, q):
        """Return a plan with qubit ``q`` capped and its unary parent removed.

        The remaining logical labels are compacted in the same way as a
        one-dimensional MPS cap: labels above ``q`` shift down by one. The
        surviving parent node is retained when possible, which keeps tensor
        identities stable for callers holding live node references.
        """
        if q not in self.leaf_of_qubit:
            raise ValueError(f"qubit {q!r} is not present in the tree.")
        if self.n <= 1:
            raise ValueError("cannot remove the only qubit from a tree.")
        leaf = self.leaf_of_qubit[q]
        parent = self.parent.get(leaf)
        if parent is None:
            raise ValueError("cannot remove the root leaf from a multi-qubit tree.")

        children = {node: tuple(ch) for node, ch in self.children.items()}
        qubit_of_leaf = dict(self.qubit_of_leaf)
        children[parent] = tuple(c for c in children[parent] if c != leaf)
        del children[leaf]
        del qubit_of_leaf[leaf]

        # A virtual-only tree node may not become unary. A physical root is
        # different: its one child plus root physical leg is still a meaningful
        # rank-two top tensor, so retain that unary structural root.
        physical_root = (
            parent == self.root and self.root_qubit is not None
        )
        if len(children[parent]) == 1 and not physical_root:
            child = children[parent][0]
            children[parent] = children[child]
            del children[child]
            if child in qubit_of_leaf:
                qubit_of_leaf[parent] = qubit_of_leaf.pop(child)

        for node, old_q in tuple(qubit_of_leaf.items()):
            if old_q > q:
                qubit_of_leaf[node] = old_q - 1
        root_qubit = self.root_qubit
        if root_qubit is not None and root_qubit > q:
            root_qubit -= 1
        return type(self).from_children(
            children,
            qubit_of_leaf,
            root=self.root,
            root_qubit=root_qubit,
        )

    def __repr__(self):
        n_internal = sum(1 for nid in self.nodes() if not self.is_leaf(nid))
        root_site = (
            ""
            if self.root_qubit is None
            else f", root_qubit={self.root_qubit}"
        )
        return (
            f"TreePlan(n={self.n}, root={self.root}, "
            f"internal_nodes={n_internal}, "
            f"max_arity={self.max_arity()}{root_site})"
        )


class TreeLayoutFinder:
    """Find a rooted tree structure adapted to a gate stream.

    Parameters
    ----------
    gates : bundled gate stream, optional
        ``[(gate, where), ...]`` entries.  Two-qubit ``where`` supports define
        the weighted interaction graph. This finder does not accept a tensor
        network state: pass that separately as ``state=`` to
        :class:`TreeOptimizer`. Ignored when ``supports`` is given.
    n : int, optional
        Number of qubits.  Inferred from the stream when omitted.
    root_qubit : int, optional
        Designated qubit carried by the top tensor instead of a leaf. It remains
        part of every path, Steiner-subtree, and congestion calculation.
    supports : sequence of sequences, optional
        Explicit interaction supports, used instead of extracting them from
        ``gates``.
    structure : {"quality", "balanced", "adaptive"}
        Partition strategy passed to :meth:`TreePlan.from_order`.  ``"quality"``
        and ``"balanced"`` build strictly-binary trees when ``max_arity=2``;
        ``"adaptive"`` lets each level branch into its strongly coupled
        communities so the arity follows the gate connectivity.
    max_arity : int, None, or iterable of ints
        Maximum children per internal node.  A scalar builds one fixed tree
        (``2`` gives the binary tree; larger values or ``None`` give flatter /
        wider trees).  An iterable of candidate arities makes :meth:`run` *search*
        them and keep the objective-best plan; this is the default
        ``(2, 3, 4)``.  Pass a scalar to opt back into a single fixed tree.
    chi : int, optional
        Bond-dimension budget used to bias the default arity search toward plans
        that stay exact at ``chi`` (see :meth:`recommend_arities`).  ``None``
        keeps the search purely objective-driven.  :class:`TreeOptimizer`
        forwards its own ``chi`` here automatically.
    community_frac : float
        Strong-edge fraction for ``structure="adaptive"`` (see
        :meth:`TreePlan.from_order`).
    star_frac : float
        Near-clique density threshold for ``structure="adaptive"`` star nodes
        (see :meth:`TreePlan.from_order`).
    dense_max : int
        Maximum subsystem size for dense spectral reordering.
    objective : {"path", "congestion", "compression", "hybrid"}
        Layout objective. `"path"` preserves the co-occurrence/path-length
        heuristic; `"congestion"` selects among layout candidates using the
        predicted operator-Schmidt load on tree edges. `"hybrid"` combines
        normalized path, peak-edge-load, and total-edge-load costs using
        ``hybrid_weights``.
    order : {None, "quality"}, optional
        Optional high-quality offline mode. `"quality"` enables bounded
        greedy refinement and opportunistic Nevergrad refinement; omitted
        keeps the fast deterministic candidate selection.
    hybrid_weights : mapping or sequence of three floats, optional
        Weights for the hybrid path, maximum edge load, and total edge load.
        The default is ``(1.0, 1.0, 0.25)``.
    refine : {None, "greedy"}
        Optional fixed-plan local search used by :meth:`run` and recommendation
        methods. `"greedy"` tries adjacent leaf-label swaps before simulation;
        it never changes a live :class:`TreeOptimizer` tree.
    refine_budget : int, optional
        Maximum greedy swap proposals per candidate plan. Defaults to at most
        64 proposals when refinement is enabled.
    search : {None, "nevergrad"}
        Optional offline derivative-free refinement. It is never run unless
        requested and requires the optional ``nevergrad`` package.
    search_budget : int
        Number of Nevergrad objective evaluations per candidate plan.
    seed : int
        Reproducible seed used by the optional Nevergrad stage.
    nevergrad_optimizer : str
        Nevergrad registry optimizer name, default ``"OnePlusOne"``.
    weight_mode : {"count", "auto", "angle", "operator_schmidt"}
        Event weighting used for the interaction graph. `"count"` is the
        backward-compatible default.
    """

    def __init__(self, gates=None, n=None, *, supports=None, structure="quality",
                 max_arity=(2, 3, 4), community_frac=0.35, star_frac=0.75,
                 dense_max=512, objective="path", weight_mode="count", chi=None,
                 max_operator_qubits=8, hybrid_weights=None, refine=None,
                 refine_budget=None, search=None, search_budget=128, seed=0,
                 nevergrad_optimizer="OnePlusOne", order=None, root_qubit=None):
        if (
            _looks_like_tree_tensor_network(gates)
            or _looks_like_tree_tensor_network(supports)
        ):
            raise TypeError(
                "TreeLayoutFinder accepts a circuit gate stream or supports, "
                "not a TreeTensorNetwork. Build the layout from the circuit "
                "and pass the TTN separately as TreeOptimizer(state=...)."
            )
        if supports is None:
            payloads, wheres, event_types = self._events_from_gates(gates)
            supports = wheres
        else:
            supports = list(supports)
            payloads = [None] * len(supports)
            event_types = ["support"] * len(supports)
        supports = [tuple(_normalize_layout_support(s)) for s in supports]
        self.payloads = tuple(payloads)
        self.event_types = tuple(event_types)
        inferred = -1
        for support in supports:
            for site in support:
                if isinstance(site, Integral):
                    inferred = max(inferred, site)
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "root_qubit must be an integer or None."
                ) from exc
            inferred = max(inferred, root_qubit)
        if n is None:
            n = inferred + 1
        try:
            n = int(n)
        except (TypeError, ValueError) as exc:
            raise ValueError("n must be a positive integer.") from exc
        if n <= 0:
            raise ValueError(
                "Could not infer qubit count; pass n explicitly."
            )
        if root_qubit is not None:
            if not 0 <= root_qubit < n:
                raise ValueError(
                    f"root_qubit {root_qubit!r} is outside 0..{n - 1}."
                )
        normalized_supports = []
        for support in supports:
            if len(set(support)) != len(support):
                raise ValueError(
                    f"layout support contains duplicate qubits: {support!r}."
                )
            for site in support:
                if not isinstance(site, Integral):
                    raise ValueError(
                        "tree layout supports must contain integer qubits; "
                        f"got {site!r}."
                    )
                if not 0 <= int(site) < n:
                    raise ValueError(
                        f"layout support qubit {site!r} is outside 0..{n - 1}."
                    )
            normalized_supports.append(tuple(int(site) for site in support))
        self.n = n
        self.root_qubit = root_qubit
        self.leaf_qubits = tuple(
            q for q in range(self.n) if q != self.root_qubit
        )
        self.supports = tuple(normalized_supports)
        self.structure = structure
        self.max_arity, self.arity_candidates = _normalize_arity_candidates(
            max_arity
        )
        self.chi = _validate_chi(chi)
        self.community_frac = float(community_frac)
        self.star_frac = float(star_frac)
        self.dense_max = int(dense_max)
        self.objective = _normalize_layout_objective(objective)
        self.hybrid_weights = _normalize_hybrid_weights(hybrid_weights)
        self.weight_mode = _normalize_weight_mode(weight_mode)
        self.order = _normalize_layout_order(order)
        self.refine = _normalize_layout_refinement(refine)
        if refine_budget is not None:
            refine_budget = _validate_search_budget(refine_budget, "refine_budget")
        self.refine_budget = refine_budget
        self.search = _normalize_layout_search(search)
        self.search_budget = _validate_search_budget(search_budget, "search_budget")
        try:
            self.seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer.") from exc
        self.nevergrad_optimizer = str(nevergrad_optimizer)
        if not self.nevergrad_optimizer:
            raise ValueError("nevergrad_optimizer must be a non-empty string.")
        if max_operator_qubits is not None:
            try:
                max_operator_qubits = int(max_operator_qubits)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_operator_qubits must be a positive integer or None."
                ) from exc
            if max_operator_qubits < 1:
                raise ValueError(
                    "max_operator_qubits must be a positive integer or None."
                )
        self.max_operator_qubits = max_operator_qubits

        # Layout search asks for the same structural quantities several times
        # (once per candidate arity and once per diagnostic).  Keep these
        # caches local to this immutable stream description; callers still get
        # fresh dictionaries from the public diagnostic methods below.
        self._plan_cache = {}
        self._edge_load_cache = {}
        self._rank_diagnostics_cache = {}
        self._schmidt_rank_cache = {}
        self._similarity_cache = {}
        self._congestion_weights_cache = None
        self._balanced_plan_cache = None

        sites = list(range(self.n))
        self.event_weights = tuple(
            _gate_stream_event_weights(
                self.payloads,
                self.supports,
                self.event_types,
                weight_mode=self.weight_mode,
            )
        )
        self.event_weights = tuple(
            0.0 if str(event_type).lower() in {
                "measure", "reset", "measure_reset", "cap"
            } else weight
            for weight, event_type in zip(self.event_weights, self.event_types)
        )
        self.pair_weights = _gate_stream_pair_weights(
            supports, sites, self.event_weights
        )

    @staticmethod
    def _events_from_gates(gates):
        """Return payloads, supports, and event types for layout analysis."""
        if gates is None:
            return [], [], []

        # ``_normalize_gate_entries`` intentionally accepts concrete bundled
        # sequences, not one-shot iterators.  A gate stream is commonly a
        # generator, and the optimizer may pass the same stream to both its
        # queue normalizer and this finder, so materialize it exactly once at
        # the boundary.
        if hasattr(gates, "__next__"):
            gates = list(gates)

        # Control events carry support but are not gate entries.  Strip them
        # before delegating ordinary entries to the MPS layout normalizer, so
        # a mixed stream can still build an interaction-aware tree.
        control = _mps_control_event_parts(gates)
        if control is not None:
            return [None], [control[2]], [control[0]]
        if isinstance(gates, (tuple, list)):
            if not any(
                _mps_control_event_parts(entry) is not None
                for entry in gates
            ):
                return _normalize_layout_gate_queue(gates)
            payloads = []
            supports = []
            event_types = []
            for entry in gates:
                control = _mps_control_event_parts(entry)
                if control is not None:
                    payloads.append(None)
                    supports.append(control[2])
                    event_types.append(control[0])
                    continue
                one_payload, one_where, one_type = _normalize_layout_gate_queue(
                    (entry,)
                )
                payloads.extend(one_payload)
                supports.extend(one_where)
                event_types.extend(one_type)
            return payloads, supports, event_types

        payloads, supports, event_types = _normalize_layout_gate_queue(gates)
        return payloads, supports, event_types

    @staticmethod
    def _supports_from_gates(gates):
        """Return only multi-site supports for compatibility with old callers."""
        _payloads, supports, _event_types = TreeLayoutFinder._events_from_gates(gates)
        return [
            tuple(_normalize_layout_support(support))
            for support in supports
            if len(_normalize_layout_support(support)) >= 2
        ]

    def _resolve_search_settings(
        self,
        *,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
    ):
        """Resolve method overrides against finder-owned search defaults."""
        if refine is _DEFAULT_SEARCH_OPTION:
            refine = self.refine
        else:
            refine = _normalize_layout_refinement(refine)
        if refine_budget is _DEFAULT_SEARCH_OPTION:
            refine_budget = self.refine_budget
        elif refine_budget is not None:
            refine_budget = _validate_search_budget(
                refine_budget, "refine_budget"
            )
        if refine is not None and refine_budget is None:
            refine_budget = max(1, min(len(self.leaf_qubits) - 1, 64))

        if search is _DEFAULT_SEARCH_OPTION:
            search = self.search
        else:
            search = _normalize_layout_search(search)
        if search_budget is _DEFAULT_SEARCH_OPTION:
            search_budget = self.search_budget
        else:
            search_budget = _validate_search_budget(search_budget, "search_budget")
        if seed is _DEFAULT_SEARCH_OPTION:
            seed = self.seed
        else:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("seed must be an integer.") from exc
        if nevergrad_optimizer is _DEFAULT_SEARCH_OPTION:
            nevergrad_optimizer = self.nevergrad_optimizer
        else:
            nevergrad_optimizer = str(nevergrad_optimizer)
            if not nevergrad_optimizer:
                raise ValueError("nevergrad_optimizer must be a non-empty string.")
        return {
            "refine": refine,
            "refine_budget": refine_budget,
            "search": search,
            "search_budget": search_budget,
            "seed": seed,
            "nevergrad_optimizer": nevergrad_optimizer,
        }

    @staticmethod
    def _leaf_nodes(plan):
        """Return the deterministic leaf-position order of a plan."""
        return tuple(sorted(plan.qubit_of_leaf))

    def _leaf_order(self, plan):
        """Return the qubit label assigned to each deterministic leaf position."""
        return tuple(plan.qubit_of_leaf[leaf] for leaf in self._leaf_nodes(plan))

    def _plan_with_leaf_order(self, plan, order):
        """Return ``plan``'s immutable topology with a new leaf assignment."""
        order = tuple(int(q) for q in order)
        if set(order) != set(self.leaf_qubits) or len(order) != len(
            self.leaf_qubits
        ):
            raise ValueError(
                "leaf order must contain every non-root qubit exactly once."
            )
        qubit_of_leaf = dict(plan.qubit_of_leaf)
        for leaf, qubit in zip(self._leaf_nodes(plan), order):
            qubit_of_leaf[leaf] = qubit
        return TreePlan.from_children(
            plan.children,
            qubit_of_leaf,
            root=plan.root,
            root_qubit=plan.root_qubit,
        )

    def _plan_with_leaf_swap(self, plan, left_leaf, right_leaf):
        """Swap two labels while retaining the tree topology exactly."""
        qubit_of_leaf = dict(plan.qubit_of_leaf)
        qubit_of_leaf[left_leaf], qubit_of_leaf[right_leaf] = (
            qubit_of_leaf[right_leaf],
            qubit_of_leaf[left_leaf],
        )
        return TreePlan.from_children(
            plan.children,
            qubit_of_leaf,
            root=plan.root,
            root_qubit=plan.root_qubit,
        )

    def _path_score_and_max(self, plan):
        """Return the weighted interaction path sum and longest active path."""
        score = 0.0
        max_path = 0
        for (qa, qb), weight in self.pair_weights.items():
            distance = plan.tree_distance(qa, qb)
            score += float(weight) * distance
            max_path = max(max_path, distance)
        return float(score), int(max_path)

    def _path_score_after_leaf_swap(self, plan, left_leaf, right_leaf, score):
        """Return the exact path-score update for a two-label leaf swap."""
        qa = plan.qubit_of_leaf[left_leaf]
        qb = plan.qubit_of_leaf[right_leaf]
        change = 0.0
        for q in range(self.n):
            if q == qa or q == qb:
                continue
            weight_a = self.pair_weights.get(tuple(sorted((qa, q))), 0.0)
            weight_b = self.pair_weights.get(tuple(sorted((qb, q))), 0.0)
            if weight_a:
                change += float(weight_a) * (
                    plan.tree_distance(qb, q) - plan.tree_distance(qa, q)
                )
            if weight_b:
                change += float(weight_b) * (
                    plan.tree_distance(qa, q) - plan.tree_distance(qb, q)
                )
        return float(score + change)

    @staticmethod
    def _normalized_cost(value, reference):
        """Normalize a non-negative layout metric against a fixed baseline."""
        if reference > 0.0:
            return float(value / reference)
        return 0.0 if value == 0.0 else float(value)

    def _hybrid_key(self, plan):
        """Return normalized distance and rank-load cost for hybrid selection."""
        score, max_path = self._path_score_and_max(plan)
        loads = self.edge_loads(plan)
        max_load = max(loads.values(), default=0.0)
        total_load = sum(loads.values())

        balanced = self._balanced_plan()
        balanced_score, _ = self._path_score_and_max(balanced)
        balanced_loads = self.edge_loads(balanced)
        balanced_max_load = max(balanced_loads.values(), default=0.0)
        balanced_total_load = sum(balanced_loads.values())
        path_weight, max_load_weight, total_load_weight = self.hybrid_weights
        hybrid = (
            path_weight * self._normalized_cost(score, balanced_score)
            + max_load_weight * self._normalized_cost(max_load, balanced_max_load)
            + total_load_weight * self._normalized_cost(
                total_load, balanced_total_load
            )
        )
        return (
            float(hybrid),
            float(max_load),
            float(total_load),
            float(score),
            int(max_path),
        )

    def _tensor_cost_key(self, plan):
        """Return a chi-scaled proxy for local TTN tensor cost.

        A wider node reduces geodesic distance but increases the number of
        virtual legs on one tensor.  The exact contraction cost depends on
        the realized bond dimensions, so this uses the configured ``chi`` (or
        a conservative qubit bond of two) to rank structures without ever
        allocating tensors.
        """
        chi = max(2, int(self.chi or 2))
        log_chi = float(np.log2(chi))
        degrees = []
        log_sizes = []
        for node, children in plan.children.items():
            if not children:
                continue
            virtual_degree = len(children) + (1 if node in plan.parent else 0)
            physical_legs = 1 if node in plan.qubit_of_node else 0
            degrees.append(virtual_degree)
            log_sizes.append(virtual_degree * log_chi + physical_legs)
        if not degrees:
            return (0.0, 0.0, 0, 0)
        max_log_size = max(log_sizes)
        # log2(sum(2**log_size)) without overflowing for large chi/arity.
        shifted = np.asarray(log_sizes, dtype=float) - max_log_size
        total_log_size = max_log_size + float(np.log2(np.exp2(shifted).sum()))
        return (
            float(max_log_size),
            float(total_log_size),
            int(max(degrees)),
            int(sum(degrees)),
        )

    def _objective_key(self, plan):
        """Return the selected objective's deterministic comparison key."""
        if self.objective == "path":
            return self._path_score_and_max(plan)
        if self.objective == "congestion":
            return self._congestion_key(plan)
        if self.objective == "compression":
            loads = self.edge_loads(plan)
            values = tuple(loads.values())
            tensor_cost = self._tensor_cost_key(plan)
            return (
                max(values, default=0.0),
                sum(values),
                tensor_cost[0],
                tensor_cost[1],
                self.score(plan),
                max(
                    (plan.tree_distance(a, b) for a in range(self.n)
                     for b in range(a + 1, self.n)),
                    default=0,
                ),
            )
        return self._hybrid_key(plan)

    def _selection_key(self, plan, chi):
        """Return the objective key with the optional chi feasibility prefix."""
        key = self._objective_key(plan)
        if chi is not None:
            return (_chi_cut_fields(plan, chi)["chi_overflow"],) + key
        return key

    def _selection_loss(self, plan, chi):
        """Return a scalar surrogate for derivative-free layout search."""
        key = self._objective_key(plan)
        if self.objective == "path":
            value = key[0]
        elif self.objective in {"congestion", "compression"}:
            value = key[0] + 1.0e-6 * key[1] + 1.0e-12 * key[2]
        else:
            value = key[0]
        if chi is not None:
            value += 1.0e6 * _chi_cut_fields(plan, chi)["chi_overflow"]
        return float(value)

    def _discard_plan_cache(self, plan):
        """Release diagnostics retained only for a rejected temporary plan."""
        cached = self._edge_load_cache.get(id(plan))
        if cached is not None and cached[0] is plan:
            del self._edge_load_cache[id(plan)]
        cached = self._rank_diagnostics_cache.get(id(plan))
        if cached is not None and cached[0] is plan:
            del self._rank_diagnostics_cache[id(plan)]

    def _refine_plan_greedy(self, plan, *, chi, budget, progbar=False):
        """Greedily improve a fixed topology through adjacent leaf swaps."""
        initial_key = self._selection_key(plan, chi)
        leaf_nodes = self._leaf_nodes(plan)
        if len(leaf_nodes) < 2 or budget < 1:
            return plan, {
                "method": "greedy",
                "evaluations": 0,
                "accepted_moves": 0,
                "initial_key": initial_key,
                "final_key": initial_key,
            }

        current = plan
        current_key = initial_key
        current_path_score = self.score(current)
        evaluations = 0
        accepted_moves = 0
        position = 0
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout greedy",
                leave=False,
            )
        while position < len(leaf_nodes) - 1 and evaluations < budget:
            left_leaf = leaf_nodes[position]
            right_leaf = leaf_nodes[position + 1]
            evaluations += 1
            if progress is not None:
                progress.update()
            if self.objective == "path":
                candidate_path_score = self._path_score_after_leaf_swap(
                    current, left_leaf, right_leaf, current_path_score
                )
                if candidate_path_score >= current_path_score - 1.0e-12:
                    position += 1
                    continue
                candidate = self._plan_with_leaf_swap(
                    current, left_leaf, right_leaf
                )
                candidate_key = self._selection_key(candidate, chi)
            else:
                candidate = self._plan_with_leaf_swap(
                    current, left_leaf, right_leaf
                )
                candidate_key = self._selection_key(candidate, chi)
                candidate_path_score = None

            if candidate_key < current_key:
                self._discard_plan_cache(current)
                current = candidate
                current_key = candidate_key
                if candidate_path_score is None:
                    current_path_score = self.score(current)
                else:
                    current_path_score = candidate_path_score
                accepted_moves += 1
                position = max(0, position - 1)
            else:
                self._discard_plan_cache(candidate)
                position += 1
        if progress is not None:
            progress.close()
        return current, {
            "method": "greedy",
            "evaluations": evaluations,
            "accepted_moves": accepted_moves,
            "initial_key": initial_key,
            "final_key": current_key,
        }

    def _refine_plan_nevergrad(
        self, plan, *, chi, budget, seed, optimizer_name, progbar=False
    ):
        """Use Nevergrad to refine a leaf assignment before simulation starts."""
        try:
            import nevergrad as ng
        except ImportError as exc:
            raise ImportError(
                "Nevergrad tree-layout search requires the optional dependency. "
                "Install it with `pip install pepsy[layout]`."
            ) from exc

        try:
            optimizer_class = ng.optimizers.registry[optimizer_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Nevergrad optimizer {optimizer_name!r}."
            ) from exc

        initial_plan = plan
        initial_key = self._selection_key(initial_plan, chi)
        initial_order = self._leaf_order(initial_plan)
        if len(initial_order) < 2 or budget < 1:
            return initial_plan, {
                "method": "nevergrad",
                "optimizer": optimizer_name,
                "budget": budget,
                "evaluations": 0,
                "seed": seed,
                "initial_key": initial_key,
                "final_key": initial_key,
                "improved": False,
            }
        leaf_qubits = tuple(initial_order)
        priorities = np.arange(len(leaf_qubits), dtype=float)
        parametrization = ng.p.Array(init=priorities)
        if hasattr(parametrization, "set_bounds"):
            parametrization.set_bounds(
                -float(len(leaf_qubits)),
                float(2 * len(leaf_qubits)),
            )
        optimizer = optimizer_class(parametrization=parametrization, budget=budget)
        random_state = getattr(optimizer.parametrization, "random_state", None)
        if random_state is not None:
            random_state.seed(seed)

        losses = {}
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout nevergrad",
                leave=False,
            )

        def loss(values):
            if progress is not None:
                progress.update()
            values = np.asarray(values)
            order = tuple(
                leaf_qubits[int(position)]
                for position in np.argsort(values, kind="stable")
            )
            cached = losses.get(order)
            if cached is not None:
                return cached
            candidate = self._plan_with_leaf_order(initial_plan, order)
            value = self._selection_loss(candidate, chi)
            losses[order] = value
            self._discard_plan_cache(candidate)
            return value

        try:
            recommendation = optimizer.minimize(loss)
        finally:
            if progress is not None:
                progress.close()
        final_order = tuple(
            leaf_qubits[int(position)]
            for position in np.argsort(
                np.asarray(recommendation.value), kind="stable"
            )
        )
        candidate = self._plan_with_leaf_order(initial_plan, final_order)
        candidate_key = self._selection_key(candidate, chi)
        if candidate_key < initial_key:
            final_plan = candidate
            improved = True
        else:
            self._discard_plan_cache(candidate)
            final_plan = initial_plan
            improved = False
        return final_plan, {
            "method": "nevergrad",
            "optimizer": optimizer_name,
            "budget": budget,
            "evaluations": len(losses),
            "seed": seed,
            "initial_key": initial_key,
            "final_key": self._selection_key(final_plan, chi),
            "improved": improved,
        }

    def _improve_plan(self, plan, *, chi, settings, progbar=False):
        """Run the requested pre-simulation plan refinements in sequence."""
        initial_order = self._leaf_order(plan)
        initial_key = self._selection_key(plan, chi)
        info = {
            "initial_order": initial_order,
            "initial_key": initial_key,
            "refinement": None,
            "search": None,
        }
        if settings["refine"] == "greedy":
            plan, info["refinement"] = self._refine_plan_greedy(
                plan,
                chi=chi,
                budget=settings["refine_budget"],
                progbar=progbar,
            )
        if settings["search"] == "nevergrad":
            plan, info["search"] = self._refine_plan_nevergrad(
                plan,
                chi=chi,
                budget=settings["search_budget"],
                seed=settings["seed"],
                optimizer_name=settings["nevergrad_optimizer"],
                progbar=progbar,
            )
        info["final_order"] = self._leaf_order(plan)
        info["final_key"] = self._selection_key(plan, chi)
        return plan, info

    def _build_plan(self, weights, *, structure=None,
                    max_arity=_DEFAULT_MAX_ARITY):
        """Build one deterministic candidate tree from pair weights."""
        if max_arity is _DEFAULT_MAX_ARITY:
            max_arity = self.max_arity
        structure = self.structure if structure is None else structure
        key = (structure, max_arity, id(weights))
        # ``weights`` is normally one of the finder-owned cached mappings.  The
        # identity component avoids returning a plan built from a different
        # caller-supplied weighting mapping with the same arity.
        cached = self._plan_cache.get(key)
        if cached is not None and cached[0] is weights:
            return cached[1]
        plan = TreePlan.from_order(
            self.leaf_qubits,
            weights=weights,
            structure=structure,
            max_arity=max_arity,
            community_frac=self.community_frac,
            star_frac=self.star_frac,
            dense_max=self.dense_max,
            root_qubit=self.root_qubit,
        )
        self._plan_cache[key] = (weights, plan)
        return plan

    def _schmidt_rank(self, payload, support, left_support):
        """Return a cached numeric operator-Schmidt rank or bound."""
        return self._schmidt_rank_info(payload, support, left_support)["rank"]

    def _schmidt_rank_info(self, payload, support, left_support):
        """Return rank metadata used by compression diagnostics.

        ``exact=False`` is deliberate for opaque native arrays, MPO bond
        bounds, and supports larger than ``max_operator_qubits``.  The numeric
        rank is then a conservative operator-space bound, never an optimistic
        hard-coded rank-two fallback.
        """
        if (
            self.max_operator_qubits is not None
            and len(support) > self.max_operator_qubits
        ):
            return {
                "rank": _operator_schmidt_rank_bound(support, left_support),
                "exact": False,
                "reason": "max_operator_qubits",
            }
        # For an ordinary dense gate, its Schmidt rank depends on the operator
        # data and *wire positions* in ``support``, not on the global qubit
        # labels. Reusing the same CNOT/CZ/parameterized matrix across many
        # pairs should therefore reuse the small SVD. A structured MPO carries
        # explicit site labels, so retain its label-sensitive cache key.
        support = tuple(support)
        left_set = frozenset(left_support)
        is_structured_mpo = callable(getattr(payload, "gen_sites_present", None))
        if is_structured_mpo:
            partition_key = left_set
            support_key = support
        else:
            positions = {site: pos for pos, site in enumerate(support)}
            partition_key = tuple(
                positions[site] for site in support if site in left_set
            )
            support_key = len(support)
        key = (id(payload), support_key, partition_key)
        cached = self._schmidt_rank_cache.get(key)
        if cached is not None and cached[0] is payload:
            return cached[1]
        rank = _submpo_schmidt_rank_bound(payload, support, left_support)
        if rank is not None:
            info = {
                "rank": int(rank),
                "exact": False,
                "reason": "mpo_bond_bound",
            }
        else:
            info = _mps_operator_schmidt_rank_info(
                payload,
                support,
                left_support,
                max_operator_qubits=self.max_operator_qubits,
            )
        self._schmidt_rank_cache[key] = (payload, info)
        return info

    def _candidate_plans(self, max_arity):
        """Build the candidate plans considered by the selected objective."""
        interaction_plan = self._build_plan(self._similarity_weights())
        if max_arity != self.max_arity:
            interaction_plan = self._build_plan(
                self._similarity_weights(), max_arity=max_arity
            )
        if self.objective == "path":
            return {"interaction": interaction_plan}

        congestion_plan = self._build_plan(
            self._similarity_weights(self._congestion_pair_weights()),
            max_arity=max_arity,
        )
        balanced_plan = self._build_plan(
            self._similarity_weights(), structure="balanced",
            max_arity=max_arity,
        )
        return {
            "interaction": interaction_plan,
            "congestion": congestion_plan,
            "balanced": balanced_plan,
        }

    def _select_plan(self, max_arity):
        """Select one plan without changing the finder's stored diagnostics."""
        candidates = self._candidate_plans(max_arity)
        selected = min(
            candidates,
            key=lambda name: self._objective_key(candidates[name]),
        )
        return candidates[selected]

    def qubit_order(self):
        """Return a spectral qubit ordering adapted to the gate-stream interactions.

        The order is the global Fiedler spectral reordering of the leaf qubits
        under the similarity weights used internally by the layout finder.
        A configured ``root_qubit`` is fixed at the root and omitted from this
        returned leaf order. Strongly coupled leaf qubits end up consecutive,
        which is the ideal input for :meth:`TreePlan.build_layered` so that
        blocks group entangled qubits together.

        Returns
        -------
        list of int
            Every non-root qubit exactly once (all ``0..n-1`` qubits when
            ``root_qubit`` is ``None``).
        """
        weights = self._similarity_weights()
        order = _gate_stream_spectral_order(
            list(self.leaf_qubits), weights, dense_max=self.dense_max
        )
        return order if order else list(self.leaf_qubits)

    def layered(self, block_size=4, *, order=None):
        """Build a fixed layered tree for a chosen ``block_size`` (no search).

        This is the direct, single-``block_size`` counterpart to
        :meth:`recommend_layered`.  It orders the qubits with the spectral
        :meth:`qubit_order` (so strongly coupled qubits share a block) and
        returns the :class:`TreePlan` from :meth:`TreePlan.build_layered`
        straight away -- no candidate sweep, no wrapper dict.

        ``block_size`` is a cost/accuracy knob, not something to maximize
        blindly.  A blocking node fuses ``block_size`` physical qubits into one
        tensor, so every intra-block correlation is represented *exactly*, but
        that tensor has ``2 ** block_size`` physical dimension.  Larger blocks
        are therefore more accurate at fixed ``chi`` yet exponentially more
        expensive: pick the largest block that fits your memory budget rather
        than searching, since a congestion search cannot see the block tensor's
        exponential cost and simply trends toward the widest block.

        Parameters
        ----------
        block_size : int
            Number of physical qubits per leaf-parent (blocking) node.
        order : sequence of int, optional
            Qubit order fed to :meth:`TreePlan.build_layered`.  Defaults to the
            spectral :meth:`qubit_order`.

        Returns
        -------
        TreePlan
            The layered plan (blocking layer, binary middle, ternary top).
        """
        if order is None:
            order = self.qubit_order()
        else:
            order = [int(q) for q in order]
        return TreePlan.build_layered(
            order,
            block_size=block_size,
            root_qubit=self.root_qubit,
        )

    def recommend_layered(
        self,
        block_sizes=(2, 3, 4),
        *,
        order=None,
        chi=_DEFAULT_CHI,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
        progbar=False,
    ):
        """Optimize the fixed layered structure over ``block_size``.

        The structure family is fixed by :meth:`TreePlan.build_layered`
        (a ``block_size`` blocking layer, binary middle layers, and a ternary
        top tensor); only the blocking width is free.  This builds one layered
        plan per candidate ``block_size`` on the entanglement-adapted qubit
        order and returns the plan that minimizes the selected layout
        objective, mirroring :meth:`recommend_arities`.

        Parameters
        ----------
        block_sizes : iterable of int
            Candidate blocking widths (physical qubits per leaf-parent node).
        order : sequence of int, optional
            Qubit order fed to :meth:`TreePlan.build_layered`.  Defaults to the
            spectral :meth:`qubit_order` so strongly coupled qubits share a
            block.
        chi : int, optional
            Bond-dimension budget.  The path/congestion objectives are
            ``chi``-blind cost proxies that can favour a wider block whose
            widest bond overflows ``chi`` (see :meth:`TreePlan.max_bond_cut`).
            When ``chi`` is given the recommendation is made ``chi``-aware:
            candidates are ranked first by ``chi_overflow`` (how far the widest
            bond exceeds ``log2(chi)``), so a structure that is *exact* at
            ``chi`` is preferred, and the layout objective only breaks ties
            among equally-overflowing candidates.  Each candidate additionally
            reports ``max_bond_cut``, ``chi_overflow``, and ``exact_at_chi``.
            When omitted, uses the ``chi`` supplied to the finder; pass
            ``chi=None`` explicitly for a chi-blind comparison.
        refine : {None, "greedy"}, optional
            Override the finder refinement setting. `"greedy"` performs a
            bounded adjacent leaf-swap search on each candidate tree.
        refine_budget : int, optional
            Maximum greedy proposals per candidate. When omitted, an enabled
            greedy search uses at most ``min(n - 1, 64)`` proposals.
        search : {None, "nevergrad"}, optional
            Override the finder offline search setting. Nevergrad optimizes
            only the returned fixed plan; it never mutates a live TTN.
        search_budget, seed, nevergrad_optimizer
            Optional Nevergrad configuration for each candidate plan.
        progbar : bool, optional
            Display greedy and Nevergrad search progress for each candidate.

        Returns
        -------
        dict
            ``{"objective", "recommended_block_size", "order", "chi", "plan",
            "candidates"}``.  Each candidate carries its ``block_size``, the
            :class:`TreePlan`, ``max_bond_cut``, and the same structural/cost
            summary fields as :meth:`recommend_arities`.
        """
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        settings = self._resolve_search_settings(
            refine=refine,
            refine_budget=refine_budget,
            search=search,
            search_budget=search_budget,
            seed=seed,
            nevergrad_optimizer=nevergrad_optimizer,
        )
        options = []
        for bs in block_sizes:
            key = int(bs)
            if key < 1:
                raise ValueError("block_sizes must be >= 1.")
            if key not in options:
                options.append(key)
        if not options:
            raise ValueError("block_sizes must contain at least one option.")

        if order is None:
            order = self.qubit_order()
        else:
            order = [int(q) for q in order]

        candidates = []
        for bs in options:
            plan = TreePlan.build_layered(
                order,
                block_size=bs,
                root_qubit=self.root_qubit,
            )
            plan, planning = self._improve_plan(
                plan,
                chi=chi,
                settings=settings,
                progbar=progbar,
            )
            report = self.report(
                plan, include_edge_loads=self.objective != "path"
            )
            arity_histogram = {}
            for node, children in plan.children.items():
                if not children:
                    continue
                arity_histogram[len(children)] = (
                    arity_histogram.get(len(children), 0) + 1
                )
            candidates.append({
                "block_size": bs,
                "actual_max_arity": plan.max_arity(),
                "root_arity": len(plan.children[plan.root]),
                "arity_histogram": arity_histogram,
                "score": report["score"],
                "max_path": report["max_path"],
                "max_edge_load": report["max_edge_load"],
                "peak_bond_growth": report["peak_bond_growth"],
                "max_virtual_degree": report["max_virtual_degree"],
                "total_virtual_degree": report["total_virtual_degree"],
                "estimated_max_tensor_log2": report[
                    "estimated_max_tensor_log2"
                ],
                "estimated_total_tensor_log2": report[
                    "estimated_total_tensor_log2"
                ],
                **_chi_cut_fields(plan, chi),
                "order": self._leaf_order(plan),
                "planning": planning,
                "plan": plan,
            })

        def candidate_key(candidate):
            return self._selection_key(candidate["plan"], chi) + (
                candidate["block_size"],
            )

        recommended = min(candidates, key=candidate_key)
        return {
            "objective": self.objective,
            "recommended_block_size": recommended["block_size"],
            "initial_order": tuple(order),
            "order": recommended["order"],
            "chi": chi,
            "refine": settings["refine"],
            "search": settings["search"],
            "plan": recommended["plan"],
            "candidates": candidates,
        }

    def run(
        self,
        *,
        order=_DEFAULT_ORDER,
        chi=_DEFAULT_CHI,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
        progbar=False,
    ):
        """Return a TreePlan for the selected layout objective.

        When the finder was built with a set of candidate arities (the default
        ``max_arity=(2, 3, 4)``), this searches them with
        :meth:`recommend_arities` -- ``chi``-aware when the finder carries a
        ``chi`` -- and returns the objective-best plan.  A scalar ``max_arity``
        builds one fixed plan.

        ``chi`` and the fixed-plan ``refine`` / ``search`` controls can be
        overridden for this call. Pass ``progbar=True`` to display greedy and
        Nevergrad search progress. Omitted values inherit the corresponding
        finder settings, so the original zero-argument behavior is unchanged.

        ``order="quality"`` is a convenience mode matching the MPS layout
        API: it enables bounded greedy refinement and opportunistic Nevergrad
        refinement when the optional dependency is installed. If Nevergrad is
        unavailable, quality mode falls back to greedy refinement. Pass
        ``search=None`` or ``refine=None`` explicitly to disable either stage.
        """
        if order is _DEFAULT_ORDER:
            order = self.order
        else:
            order = _normalize_layout_order(order)
        if order == "quality":
            if refine is _DEFAULT_SEARCH_OPTION:
                refine = "greedy"
            if search is _DEFAULT_SEARCH_OPTION:
                search = "nevergrad" if _nevergrad_available() else None
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        settings = self._resolve_search_settings(
            refine=refine,
            refine_budget=refine_budget,
            search=search,
            search_budget=search_budget,
            seed=seed,
            nevergrad_optimizer=nevergrad_optimizer,
        )
        if self.arity_candidates is not None:
            rec = self.recommend_arities(
                self.arity_candidates,
                chi=chi,
                progbar=progbar,
                **settings,
            )
            self._last_arity_recommendation = rec
            self._selected_candidate = f"arity={rec['recommended_max_arity']}"
            self._last_candidate_scores = {
                f"arity={cand['max_arity']}": self._selection_key(
                    cand["plan"], rec["chi"]
                )
                for cand in rec["candidates"]
            }
            return rec["plan"]
        candidates = self._candidate_plans(self.max_arity)
        if settings["refine"] is not None or settings["search"] is not None:
            candidates = {
                name: self._improve_plan(
                    plan,
                    chi=chi,
                    settings=settings,
                    progbar=progbar,
                )[0]
                for name, plan in candidates.items()
            }
        selected = min(
            candidates,
            key=lambda name: self._selection_key(candidates[name], chi),
        )
        self._last_candidates = candidates
        self._last_candidate_scores = {
            name: self._selection_key(plan, chi)
            for name, plan in candidates.items()
        }
        self._selected_candidate = selected
        return candidates[selected]

    def candidate_plans(self, *, chi=_DEFAULT_CHI):
        """Return immutable candidate plans for optional pilot replay.

        The normal :meth:`run` path remains static and cheap. This method
        exposes the interaction, congestion, balanced, and arity candidates
        that a state-aware pilot can compare without rebuilding the finder.
        Candidate names are stable strings such as
        ``"congestion:arity=2"``.
        """
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        arities = (
            tuple(self.arity_candidates)
            if self.arity_candidates is not None
            else (self.max_arity,)
        )
        result = {}
        for arity in arities:
            plans = self._candidate_plans(arity)
            for name, plan in plans.items():
                key = f"{name}:arity={arity}"
                result[key] = {
                    "plan": plan,
                    "objective_key": self._selection_key(plan, chi),
                    "path_score": self.score(plan),
                    "tensor_cost": self._tensor_cost_key(plan),
                    "edge_loads": self.edge_loads(plan),
                }
        return result

    def recommend_arities(
        self,
        max_arities=(2, 3, 4),
        *,
        chi=_DEFAULT_CHI,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
        progbar=False,
    ):
        """Compare binary and wider trees and return the best candidate.

        The returned mapping contains the recommended :class:`TreePlan` under
        ``"plan"`` and candidate plans alongside structural/cost summaries
        under ``"candidates"``.  Wider arities shorten paths but increase local
        tensor degree, so the recommendation uses the selected layout
        objective and reports both effects.

        Parameters
        ----------
        max_arities : iterable of int or None
            Candidate maximum arities (``2`` = binary; ``None`` = unbounded).
        chi : int, optional
            Bond-dimension budget.  When given, the recommendation is made
            ``chi``-aware exactly as in :meth:`recommend_layered`: candidates
            are ranked first by ``chi_overflow`` so a structure that stays
            exact at ``chi`` (widest bond ``<= log2(chi)``) is preferred, and
            the layout objective only breaks ties.  Each candidate reports
            ``max_bond_cut``, ``chi_overflow``, and ``exact_at_chi``.
            When omitted, uses the ``chi`` supplied to the finder; pass
            ``chi=None`` explicitly for a chi-blind comparison.
        refine, refine_budget, search, search_budget, seed, nevergrad_optimizer
            Optional fixed-plan search controls with the same meaning as in
            :meth:`recommend_layered`. They are applied to each arity candidate
            before selecting one final immutable plan.
        progbar : bool, optional
            Display greedy and Nevergrad search progress for each candidate.
        """
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        settings = self._resolve_search_settings(
            refine=refine,
            refine_budget=refine_budget,
            search=search,
            search_budget=search_budget,
            seed=seed,
            nevergrad_optimizer=nevergrad_optimizer,
        )
        options = []
        for arity in max_arities:
            if arity is None:
                key = None
            else:
                key = int(arity)
                if key < 2:
                    raise ValueError("max_arities must be >= 2 or None.")
            if key not in options:
                options.append(key)
        if not options:
            raise ValueError("max_arities must contain at least one option.")

        candidates = []
        for arity in options:
            plan = self._select_plan(arity)
            plan, planning = self._improve_plan(
                plan,
                chi=chi,
                settings=settings,
                progbar=progbar,
            )
            report = self.report(
                plan, include_edge_loads=self.objective != "path"
            )
            arity_histogram = {}
            for node, children in plan.children.items():
                if not children:
                    continue
                arity_histogram[len(children)] = (
                    arity_histogram.get(len(children), 0) + 1
                )
            candidates.append({
                "max_arity": arity,
                "actual_max_arity": plan.max_arity(),
                "is_binary": plan.is_binary(),
                "arity_histogram": arity_histogram,
                "max_virtual_degree": max(
                    (
                        len(children) + (1 if node in plan.parent else 0)
                        for node, children in plan.children.items()
                        if children
                    ),
                    default=0,
                ),
                "total_virtual_degree": sum(
                    len(children) + (1 if node in plan.parent else 0)
                    for node, children in plan.children.items()
                    if children
                ),
                "score": report["score"],
                "max_path": report["max_path"],
                "max_edge_load": report["max_edge_load"],
                "peak_bond_growth": report["peak_bond_growth"],
                "estimated_max_tensor_log2": report[
                    "estimated_max_tensor_log2"
                ],
                "estimated_total_tensor_log2": report[
                    "estimated_total_tensor_log2"
                ],
                **_chi_cut_fields(plan, chi),
                "order": self._leaf_order(plan),
                "planning": planning,
                "plan": plan,
            })

        def candidate_key(candidate):
            return self._selection_key(candidate["plan"], chi) + (
                candidate["actual_max_arity"],
            )

        recommended = min(candidates, key=candidate_key)
        return {
            "objective": self.objective,
            "recommended_max_arity": recommended["max_arity"],
            "chi": chi,
            "refine": settings["refine"],
            "search": settings["search"],
            "plan": recommended["plan"],
            "candidates": candidates,
        }

    def recommend_layout(self, max_arities=(2, 3, 4), **kwargs):
        """Alias for :meth:`recommend_arities` with a layout-oriented name."""
        return self.recommend_arities(max_arities=max_arities, **kwargs)

    def _congestion_pair_weights(self):
        """Return pair weights proportional to gate Schmidt load."""
        cached = getattr(self, "_congestion_weights_cache", None)
        if cached is not None:
            return cached
        event_weights = []
        for payload, support, event_type in zip(
            self.payloads, self.supports, self.event_types
        ):
            if len(support) < 2 or str(event_type).lower() in {
                "measure", "reset", "measure_reset", "cap"
            }:
                event_weights.append(0.0)
                continue
            if payload is None:
                event_weights.append(1.0)
                continue
            support = tuple(dict.fromkeys(support))
            logs = []
            if len(support) <= 8:
                for mask in range(1, (1 << len(support)) - 1):
                    left = tuple(
                        site for i, site in enumerate(support)
                        if mask & (1 << i)
                    )
                    rank = self._schmidt_rank(payload, support, left)
                    logs.append(float(np.log2(rank)))
            else:
                for site in support:
                    rank = self._schmidt_rank(payload, support, (site,))
                    logs.append(float(np.log2(rank)))
            event_weights.append(max(logs, default=1.0))
        self._congestion_weights_cache = _gate_stream_pair_weights(
            self.supports,
            range(self.n),
            event_weights,
        )
        return self._congestion_weights_cache

    def _subtree_qubits(self, plan):
        """Return the qubits below each node of plan."""
        return {
            node: frozenset(
                q for q in range(self.n) if mask & (1 << q)
            )
            for node, mask in plan.subtree_qubit_masks().items()
        }

    def edge_loads(self, plan=None):
        """Return predicted log-bond growth for every tree edge."""
        if plan is None:
            plan = self.run()
        cache_key = id(plan)
        cached = self._edge_load_cache.get(cache_key)
        if cached is not None and cached[0] is plan:
            return dict(cached[1])
        below = plan.subtree_qubit_masks()
        loads = {
            (parent, child): 0.0
            for parent, children in plan.children.items()
            for child in children
        }
        rank_diagnostics = {
            "exact_events": 0,
            "bounded_events": 0,
            "reasons": {},
        }
        for payload, support, event_type in zip(
            self.payloads, self.supports, self.event_types
        ):
            support = tuple(dict.fromkeys(support))
            if len(support) < 2 or str(event_type).lower() in {
                "measure", "reset", "measure_reset", "cap"
            }:
                continue
            support_mask = 0
            for site in support:
                support_mask |= 1 << site

            # An edge crosses the support iff it belongs to the minimal
            # subtree spanning the support nodes. Scanning every tree edge
            # is needlessly O(n) for each event; for the dominant two-qubit
            # case this reduces the work to the site-to-site geodesic.
            site_nodes = [plan.node_of_qubit[site] for site in support]
            if len(site_nodes) == 2:
                # This branch dominates ordinary circuit layout. Avoid sets
                # and an all-node parent scan: every path hop is one crossed
                # rooted tree edge.
                path = plan.node_path(site_nodes[0], site_nodes[1])
                crossed_edges = [
                    (u, v) if plan.parent.get(v) == u else (v, u)
                    for u, v in zip(path, path[1:])
                ]
            else:
                span_nodes = set()
                anchor = site_nodes[0]
                for site_node in site_nodes:
                    span_nodes.update(plan.node_path(anchor, site_node))
                crossed_edges = [
                    (parent, node)
                    for node in span_nodes
                    if (parent := plan.parent.get(node)) in span_nodes
                ]

            for edge in crossed_edges:
                _parent, child = edge
                left_mask = support_mask & below[child]
                if not left_mask or left_mask == support_mask:
                    continue
                left = tuple(
                    site for site in support if left_mask & (1 << site)
                )
                info = self._schmidt_rank_info(payload, support, left)
                rank = int(info["rank"])
                loads[edge] += float(np.log2(max(1, rank)))
                if info["exact"]:
                    rank_diagnostics["exact_events"] += 1
                else:
                    rank_diagnostics["bounded_events"] += 1
                    reason = info["reason"]
                    rank_diagnostics["reasons"][reason] = (
                        rank_diagnostics["reasons"].get(reason, 0) + 1
                    )
        # Retain the plan alongside its id so a future id reuse cannot return
        # diagnostics for an unrelated short-lived plan.
        self._edge_load_cache[cache_key] = (plan, dict(loads))
        self._rank_diagnostics_cache[cache_key] = (
            plan,
            rank_diagnostics,
        )
        return dict(loads)

    def _congestion_key(self, plan):
        """Return the lexicographic key used by the load-aware objective."""
        loads = self.edge_loads(plan)
        values = tuple(loads.values())
        return (
            max(values, default=0.0),
            sum(values),
            self.score(plan),
            max(
                (plan.tree_distance(a, b) for a in range(self.n)
                 for b in range(a + 1, self.n)),
                default=0,
            ),
        )

    def _similarity_weights(self, pair_weights=None):
        """Return the qubit-pair similarity of Seitz et al. (Eq. 1).

        ``s(qi, qj) = |G(qi) & G(qj)| + 1 / (|G(qi)| + |G(qj)|)`` where ``G(q)``
        is the set of multi-qubit events acting on qubit ``q``.  The integer
        co-occurrence term ``|G(qi) & G(qj)|`` is exactly the interaction weight
        already accumulated in :attr:`pair_weights`; the ``1/(deg_i + deg_j)``
        term is a tie-breaker that gently favours grouping qubits participating
        in fewer gates, biasing the recursive bisection towards balanced
        subtrees when co-occurrence counts tie.  Only the bisection uses this
        augmented similarity; :meth:`score` keeps the pure interaction weight.
        """
        if pair_weights is None:
            pair_weights = self.pair_weights
        cache_key = id(pair_weights)
        cached = self._similarity_cache.get(cache_key)
        if cached is not None and cached[0] is pair_weights:
            return cached[1]

        degree = {q: 0.0 for q in range(self.n)}
        for support in self.supports:
            for site in set(support):
                if isinstance(site, int) and 0 <= site < self.n:
                    degree[site] += 1.0
        sim = dict(pair_weights)
        for qi in range(self.n):
            for qj in range(qi + 1, self.n):
                deg = degree[qi] + degree[qj]
                if deg > 0.0:
                    sim[(qi, qj)] = sim.get((qi, qj), 0.0) + 1.0 / deg
        self._similarity_cache[cache_key] = (pair_weights, sim)
        return sim

    def score(self, plan):
        """Return the total interaction-weighted tree-path length of ``plan``.

        Lower is better: this is the quantity the tree structure minimises
        (short physical-node paths for strongly coupled qubits).
        """
        return self._path_score_and_max(plan)[0]

    def _balanced_plan(self):
        """Return the cached index-order balanced comparison plan."""
        if self._balanced_plan_cache is None:
            self._balanced_plan_cache = TreePlan.from_order(
                self.leaf_qubits,
                structure="balanced",
                root_qubit=self.root_qubit,
            )
        return self._balanced_plan_cache

    def report(self, plan=None, *, include_edge_loads=True):
        """Return layout-quality diagnostics for ``plan`` (or a fresh run).

        The dominant lever for tree-tensor-network accuracy at fixed ``chi`` is
        how well the tree keeps strongly coupled qubits as nearby nodes: a
        two-qubit gate threads its virtual bond along the whole site-to-site
        geodesic, and every crossed bond can grow.  This report summarises those
        geodesic lengths over the interaction graph and compares the chosen
        structure against a naive balanced index-order tree (lower ``score`` is
        better).
        """
        if plan is None:
            plan = self.run()
        dists = []
        total_weight = 0.0
        weighted_sum = 0.0
        for (qa, qb), weight in self.pair_weights.items():
            d = plan.tree_distance(qa, qb)
            dists.append(d)
            weighted_sum += float(weight) * d
            total_weight += float(weight)
        n_pairs = len(dists)
        balanced = self._balanced_plan()
        balanced_score = self.score(balanced)
        if include_edge_loads:
            loads = self.edge_loads(plan)
            balanced_loads = self.edge_loads(balanced)
            rank_info = self._rank_diagnostics_cache.get(id(plan), (plan, {}))[1]
        else:
            loads = None
            balanced_loads = None
            rank_info = {}
        max_load = max(loads.values(), default=0.0) if loads is not None else None
        total_load = sum(loads.values()) if loads is not None else None
        balanced_max_load = (
            max(balanced_loads.values(), default=0.0)
            if balanced_loads is not None else None
        )
        balanced_total_load = (
            sum(balanced_loads.values())
            if balanced_loads is not None else None
        )
        hybrid_cost = None
        if self.objective == "hybrid" and loads is not None:
            hybrid_cost = self._hybrid_key(plan)[0]
        arity_histogram = {}
        for node, children in plan.children.items():
            if children:
                arity_histogram[len(children)] = (
                    arity_histogram.get(len(children), 0) + 1
                )
        tensor_cost = self._tensor_cost_key(plan)
        objective_key = self._objective_key(plan)
        return {
            "n_qubits": self.n,
            "n_interacting_pairs": n_pairs,
            "objective": self.objective,
            "weight_mode": self.weight_mode,
            "hybrid_weights": (
                self.hybrid_weights if self.objective == "hybrid" else None
            ),
            "hybrid_cost": hybrid_cost,
            "objective_key": objective_key,
            "path_score": float(weighted_sum),
            "compression_score": (
                float(objective_key[0] + objective_key[1])
                if self.objective == "compression" else None
            ),
            "root": plan.root,
            "root_qubit": plan.root_qubit,
            "is_binary": plan.is_binary(),
            "max_arity": plan.max_arity(),
            "arity_histogram": arity_histogram,
            "score": float(weighted_sum),
            "max_path": int(max(dists)) if dists else 0,
            "mean_path": float(sum(dists) / n_pairs) if n_pairs else 0.0,
            "weighted_mean_path": (
                float(weighted_sum / total_weight) if total_weight else 0.0
            ),
            "balanced_score": float(balanced_score),
            "score_ratio_vs_balanced": (
                float(weighted_sum / balanced_score) if balanced_score else 0.0
            ),
            "edge_loads": loads,
            "total_edge_load": float(total_load) if total_load is not None else None,
            "max_edge_load": float(max_load) if max_load is not None else None,
            "peak_bond_growth": (
                _safe_exp2(max_load) if max_load is not None else None
            ),
            "balanced_max_edge_load": (
                float(balanced_max_load)
                if balanced_max_load is not None else None
            ),
            "balanced_total_edge_load": (
                float(balanced_total_load)
                if balanced_total_load is not None else None
            ),
            "balanced_peak_bond_growth": (
                _safe_exp2(balanced_max_load)
                if balanced_max_load is not None else None
            ),
            "peak_bond_growth_log2": (
                float(max_load) if max_load is not None else None
            ),
            "balanced_peak_bond_growth_log2": (
                float(balanced_max_load)
                if balanced_max_load is not None else None
            ),
            "rank_exact_events": int(rank_info.get("exact_events", 0)),
            "rank_bounded_events": int(rank_info.get("bounded_events", 0)),
            "rank_bound_reasons": dict(rank_info.get("reasons", {})),
            "max_virtual_degree": tensor_cost[2],
            "total_virtual_degree": tensor_cost[3],
            "estimated_max_tensor_log2": tensor_cost[0],
            "estimated_total_tensor_log2": tensor_cost[1],
            "selected_candidate": getattr(self, "_selected_candidate", "interaction"),
            "candidate_scores": getattr(self, "_last_candidate_scores", {}),
        }

    def _plot_gate_routes(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(10, 8),
        cmap="turbo",
        color_by="gate",
        scale_cmap="viridis",
        scale_markers=_DEFAULT_SCALE_MARKERS,
        lattice=True,
        show_gate_connectivity=True,
        show_gate_paths=False,
        show_node_ids=False,
        show_site_labels=False,
        show_event_labels=False,
        colorbar=False,
        show_axes=False,
        show_title=False,
        rubberband=False,
        node_size=58,
        event_linewidth=2.0,
        event_alpha=0.5,
        tree_edge_alpha=0.38,
        gate_path_curvature=0.08,
    ):
        """Plot a tree plan over the physical lattice and gate connectivity.

        By default this draws only the explicit TTN geometry over the optional
        physical background. Pass ``show_gate_paths=True`` to add gate-stream
        route overlays as a separate diagnostic layer; those routes are not
        tensor legs. Pass ``rubberband=True`` for the
        physical-lattice rubberband view, where each non-root tree cluster is
        wrapped by a rounded translucent band. In either view,
        ``color_by="scale"`` uses colors independent of gate-stream length;
        the explicit tree view also uses circle markers by default (custom
        marker cycles can be supplied with ``scale_markers``). When enabled,
        gate-path edges are kept visually distinct: structural edges are
        straight grey segments, while colored gate routes are offset by
        small deterministic arcs (controlled by ``gate_path_curvature``).
        No stream-order colorbar or title is shown by default. Pass
        ``site_coords={qubit: (x, y)}`` to place the physical leaves on an
        existing lattice; internal tree nodes are then placed above the
        supplied leaves. Without coordinates, leaves use their deterministic
        tree order and the plot becomes a clean rooted-tree view. The default
        presentation is axis-free, following quimb's schematic drawing style;
        set ``show_axes=True`` to retain Matplotlib axes.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes, ready for further customization or saving.
        """
        plt, colormaps, ScalarMappable, Normalize, FancyArrowPatch = (
            matplotlib_modules()
        )
        if plan is None:
            plan = self.run()
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan returned by run().")
        color_by = str(color_by).replace("-", "_").strip().lower()
        color_by = {"stream": "gate", "event": "gate", "level": "scale"}.get(
            color_by, color_by
        )
        if color_by not in {"gate", "scale"}:
            raise ValueError("color_by must be 'gate' or 'scale'.")
        try:
            scale_markers = tuple(scale_markers)
        except TypeError as exc:
            raise TypeError("scale_markers must be a non-empty sequence.") from exc
        if not scale_markers:
            raise ValueError("scale_markers must be a non-empty sequence.")
        if rubberband:
            return self.plot_rubberband(
                plan,
                site_coords=site_coords,
                ax=ax,
                figsize=figsize,
                cmap=cmap,
                color_by=color_by,
                scale_cmap=scale_cmap,
                lattice=lattice,
                show_gate_connectivity=show_gate_connectivity,
                show_site_nodes=True,
                colorbar=colorbar,
                show_axes=show_axes,
                show_title=show_title,
                band_alpha=event_alpha,
                band_linewidth=event_linewidth,
                node_size=node_size,
            )
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure

        qubits = tuple(range(plan.n))
        supplied_coords = site_coords is not None
        logical_coords = resolve_site_coords(qubits, site_coords)
        leaf_order = tuple(sorted(plan.qubit_of_leaf))
        leaf_position = {node: index for index, node in enumerate(leaf_order)}
        positions = {}

        if supplied_coords:
            for node, qubit in plan.qubit_of_leaf.items():
                positions[node] = logical_coords[qubit]
            if plan.root_qubit is not None:
                # The root physical site shares the root node, so it is shown
                # at the root's eventual position below.
                positions[plan.root] = logical_coords[plan.root_qubit]
        else:
            for node, index in leaf_position.items():
                positions[node] = (float(index), 0.0)

        def place_internal(node):
            if node in positions:
                return positions[node]
            child_points = [place_internal(child) for child in plan.children[node]]
            x = sum(point[0] for point in child_points) / len(child_points)
            y = max(point[1] for point in child_points) + 1.0
            positions[node] = (x, y)
            return positions[node]

        place_internal(plan.root)
        node_scales = _tree_node_scales(plan)
        n_scales = max(node_scales.values(), default=0) + 1
        # If a root qubit was supplied, its physical coordinate should not
        # flatten the structural root into the lattice. Keep the tree center
        # while still recording the logical root-site label at that node.
        if plan.root_qubit is not None and not supplied_coords:
            positions[plan.root] = (
                positions[plan.root][0],
                positions[plan.root][1],
            )

        if lattice:
            for left, right in coordinate_lattice_edges(logical_coords):
                x0, y0 = logical_coords[left]
                x1, y1 = logical_coords[right]
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color="#d5d9de",
                    linewidth=1.0,
                    alpha=0.78,
                    zorder=1,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(logical_coords)
                if lattice
                else set()
            )
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    if frozenset((left, right)) in lattice_pairs:
                        continue
                    x0, y0 = logical_coords[left]
                    x1, y1 = logical_coords[right]
                    ax.plot(
                        (x0, x1),
                        (y0, y1),
                        color="#7e8995",
                        linewidth=0.72,
                        linestyle="-",
                        alpha=0.62,
                        zorder=1,
                    )

        # Draw the rooted tree underneath the gate ribbons.
        for parent, children in plan.children.items():
            for child in children:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color="#aeb6bf",
                    linewidth=1.05,
                    alpha=tree_edge_alpha,
                    zorder=2,
                )

        internal = [node for node in plan.nodes() if not plan.is_leaf(node)]
        leaves = list(plan.leaves())
        if color_by == "scale":
            def draw_scale_nodes(nodes, size):
                for scale in sorted({node_scales[node] for node in nodes}):
                    scale_nodes = [
                        node for node in nodes if node_scales[node] == scale
                    ]
                    marker = scale_markers[scale % len(scale_markers)]
                    ax.scatter(
                        [positions[node][0] for node in scale_nodes],
                        [positions[node][1] for node in scale_nodes],
                        s=size,
                        marker=marker,
                        color=scale_color(
                            colormaps, scale_cmap, scale, n_scales
                        ),
                        edgecolors="#41464c",
                        linewidths=0.7,
                        zorder=5,
                    )

            draw_scale_nodes(internal, node_size * 0.82)
            draw_scale_nodes(leaves, node_size)
        else:
            if internal:
                ax.scatter(
                    [positions[node][0] for node in internal],
                    [positions[node][1] for node in internal],
                    s=node_size * 0.82,
                    color="#7b8188",
                    edgecolors="#41464c",
                    linewidths=0.7,
                    zorder=5,
                )
            if leaves:
                ax.scatter(
                    [positions[node][0] for node in leaves],
                    [positions[node][1] for node in leaves],
                    s=node_size,
                    c=[plan.qubit_of_leaf[node] for node in leaves],
                    cmap=colormaps.get_cmap(cmap),
                    vmin=0,
                    vmax=max(1, plan.n - 1),
                    edgecolors="#41464c",
                    linewidths=0.7,
                    zorder=5,
                )

        n_events = len(self.supports)
        if show_gate_paths:
            event_weights = tuple(self.event_weights)
            max_weight = max(event_weights, default=1.0)
            for event_index, (support, weight) in enumerate(
                zip(self.supports, event_weights)
            ):
                support = tuple(dict.fromkeys(support))
                event_color_value = event_color(
                    colormaps, cmap, event_index, n_events
                )
                width = event_linewidth * (
                    0.75
                    + 0.75 * (float(weight) / max(max_weight, 1.0)) ** 0.5
                )
                if gate_path_curvature:
                    side = 1.0 if event_index % 2 == 0 else -1.0
                    magnitude = 1.0 + float((event_index // 2) % 3)
                    route_curvature = (
                        side * float(gate_path_curvature) * magnitude
                    )
                else:
                    route_curvature = 0.0
                paths = []
                for left, right in zip(support, support[1:]):
                    path = plan.node_path(
                        plan.node_of_qubit[left], plan.node_of_qubit[right]
                    )
                    paths.append(path)
                segments = set()
                for path in paths:
                    for left, right in zip(path, path[1:]):
                        edge = (left, right) if left < right else (right, left)
                        if edge in segments:
                            continue
                        segments.add(edge)
                        x0, y0 = positions[left]
                        x1, y1 = positions[right]
                        if color_by == "scale":
                            segment_color = scale_color(
                                colormaps,
                                scale_cmap,
                                max(node_scales[left], node_scales[right]),
                                n_scales,
                            )
                        else:
                            segment_color = event_color_value
                        ax.add_patch(
                            FancyArrowPatch(
                                (x0, y0),
                                (x1, y1),
                                arrowstyle="-",
                                connectionstyle=(
                                    f"arc3,rad={route_curvature:.4g}"
                                ),
                                linewidth=width,
                                color=segment_color,
                                alpha=event_alpha,
                                zorder=3,
                            )
                        )
                if color_by == "gate":
                    for qubit in support:
                        x, y = positions[plan.node_of_qubit[qubit]]
                        ax.scatter(
                            [x], [y], s=node_size * 1.25,
                            color=[event_color_value], alpha=event_alpha,
                            edgecolors="white", linewidths=0.5, zorder=7,
                        )
                if show_event_labels and support:
                    node = plan.node_of_qubit[support[0]]
                    x, y = positions[node]
                    ax.text(
                        x,
                        y,
                        str(event_index),
                        color=(
                            event_color_value
                            if color_by == "gate"
                            else "#59636e"
                        ),
                        fontsize=8,
                        ha="center",
                        va="center",
                        zorder=8,
                    )

        if show_site_labels:
            for qubit in qubits:
                node = plan.node_of_qubit[qubit]
                x, y = positions[node]
                ax.annotate(
                    f"q{qubit}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#374151",
                    zorder=9,
                )
        if show_node_ids:
            for node in plan.nodes():
                x, y = positions[node]
                ax.annotate(
                    f"n{node}",
                    (x, y),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#5b6168",
                    zorder=9,
                )

        colorbar_count = (
            n_events if color_by == "gate" and show_gate_paths else n_scales
        )
        if colorbar and colorbar_count:
            add_order_colorbar(
                fig,
                ax,
                colormaps,
                ScalarMappable,
                Normalize,
                cmap if color_by == "gate" else scale_cmap,
                colorbar_count,
                label=(
                    "gate stream order"
                    if color_by == "gate"
                    else "tree scale (leaf = 0)"
                ),
            )
        title = (
            "Tree layout finder — "
            + ("colored gate paths" if color_by == "gate" else "scale-colored tree")
        )
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("layout x")
            ax.set_ylabel("layout y")
            ax.margins(0.14)
            ax.set_aspect("equal", adjustable="datalim")
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
                margins=0.14,
            )
        return fig, ax

    def plot_tent(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(8, 7),
        cmap="turbo",
        edge_cmap="GnBu",
        node_cmap="YlOrRd",
        color_by="scale",
        edge_color="#2f80a0",
        show_edge_arrows=False,
        arrow_size=8.0,
        order=True,
        lattice=True,
        show_gate_connectivity=True,
        show_node_ids=False,
        show_site_labels=False,
        colorbar=False,
        show_axes=False,
        show_title=False,
        node_size=38,
        edge_linewidth=1.35,
        edge_alpha=1.0,
        vertical_spacing=None,
    ):
        """Plot the hierarchy as a Cotengra-style tent over the raw graph.

        Physical sites and gate connectivity stay in the lower, grey raw
        graph. Internal TTN nodes are lifted above the mean position of their
        descendant sites, and each parent-child hierarchy edge uses one
        uniform solid color by default. Pass ``edge_color=None`` to match
        each incoming edge to the node it terminates at (so ``node_cmap``
        controls both). The default has no arrows, matching Cotengra's
        structural tent view; pass
        ``show_edge_arrows=True`` only when parent-to-child direction is
        needed. This is deliberately a structural
        visualization:
        gate-by-gate route overlays are not drawn. Set ``order=True`` to place
        hierarchy nodes by a deterministic post-order traversal, matching the
        ordering option in Cotengra's tent plots. Use ``color_by="order"`` if
        the same traversal should also control the colors.
        """
        plt, colormaps, ScalarMappable, Normalize, _FancyArrowPatch = (
            matplotlib_modules()
        )
        if plan is None:
            plan = self.run()
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan returned by run().")
        color_by = str(color_by).replace("-", "_").strip().lower()
        color_by = {"level": "scale", "size": "scale"}.get(
            color_by, color_by
        )
        if color_by not in {"scale", "order"}:
            raise ValueError("color_by must be 'scale' or 'order'.")
        try:
            arrow_size = float(arrow_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "arrow_size must be a positive real number."
            ) from exc
        if not np.isfinite(arrow_size) or arrow_size <= 0.0:
            raise ValueError("arrow_size must be a positive real number.")
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure

        qubits = tuple(range(plan.n))
        coords = resolve_site_coords(qubits, site_coords)
        node_scales = _tree_node_scales(plan)
        n_scales = max(node_scales.values(), default=0) + 1

        if lattice:
            for left, right in coordinate_lattice_edges(coords):
                ax.plot(
                    (coords[left][0], coords[right][0]),
                    (coords[left][1], coords[right][1]),
                    color="#d5d9de",
                    linewidth=1.0,
                    alpha=0.78,
                    zorder=1,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(coords)
                if lattice
                else set()
            )
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    if frozenset((left, right)) in lattice_pairs:
                        continue
                    ax.plot(
                        (coords[left][0], coords[right][0]),
                        (coords[left][1], coords[right][1]),
                        color="#7e8995",
                        linewidth=0.72,
                        linestyle="-",
                        alpha=0.62,
                        zorder=1,
                    )

        subtree_qubits = {}

        def gather_qubits(node):
            if node in subtree_qubits:
                return subtree_qubits[node]
            result = []
            if node in plan.qubit_of_leaf:
                result.append(plan.qubit_of_leaf[node])
            if node == plan.root and plan.root_qubit is not None:
                result.append(plan.root_qubit)
            for child in plan.children.get(node, ()):
                result.extend(gather_qubits(child))
            subtree_qubits[node] = tuple(result)
            return subtree_qubits[node]

        for node in plan.nodes():
            gather_qubits(node)

        x_span = max(
            max(point[0] for point in coords.values())
            - min(point[0] for point in coords.values()),
            1.0,
        )
        y_max = max(point[1] for point in coords.values())
        if vertical_spacing is None:
            # Keep the tent compact for square 2-D lattices. The previous
            # spacing made a 6x6 lattice grow into a very tall strip even
            # though ``figsize`` only changes the canvas, not the geometry.
            vertical_spacing = max(0.55, 0.16 * x_span)
        vertical_spacing = float(vertical_spacing)
        if vertical_spacing <= 0.0:
            raise ValueError("vertical_spacing must be positive.")

        positions = {
            node: coords[qubit]
            for node, qubit in plan.qubit_of_leaf.items()
        }
        if plan.root_qubit is not None:
            positions[plan.root] = coords[plan.root_qubit]

        internal = [node for node in plan.nodes() if not plan.is_leaf(node)]
        if order or color_by == "order":
            postorder = []

            def visit(node):
                for child in plan.children.get(node, ()):
                    visit(child)
                postorder.append(node)

            visit(plan.root)
            order_values = {node: i for i, node in enumerate(postorder)}
            order_count = max(1, len(postorder))
            internal_order = {
                node: index
                for index, node in enumerate(
                    node for node in postorder if node in internal
                )
            }
            order_span = max(1, n_scales - 1)
            order_denominator = max(1, len(internal) - 1)
            for node in internal:
                sites = gather_qubits(node)
                x = sum(coords[qubit][0] for qubit in sites) / len(sites)
                if order:
                    # Preserve post-order relationships without giving every
                    # internal node a separate vertical layer. A separate
                    # layer for all nodes makes larger 2D circuits needlessly
                    # tall and narrow without conveying extra geometry.
                    height = 1.0 + order_span * (
                        internal_order[node] / order_denominator
                    )
                else:
                    height = 1.0 + order_values[node] / order_count
                y = y_max + vertical_spacing * height
                positions[node] = (x, y)
            n_colors = order_count if color_by == "order" else n_scales
        else:
            order_values = None
            for node in internal:
                sites = gather_qubits(node)
                x = sum(coords[qubit][0] for qubit in sites) / len(sites)
                y = y_max + vertical_spacing * (node_scales[node] + 1.0)
                positions[node] = (x, y)
            n_colors = n_scales

        def node_color(node):
            if color_by == "order":
                return event_color(
                    colormaps, cmap, order_values[node], n_colors
                )
            return scale_color(
                colormaps, node_cmap, node_scales[node], n_colors
            )

        def hierarchy_edge_color(parent, child):
            if edge_color is not None:
                return edge_color
            # ``None`` means "follow the node palette": this is intentionally
            # the node color itself rather than a separate edge colormap, so
            # an incoming edge and its child are visually identical.
            return node_color(child)

        for parent, children in plan.children.items():
            for child in children:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                # The incoming edge is colored like the node it terminates at,
                # making each scale/order layer visually self-consistent.
                edge_color_value = hierarchy_edge_color(parent, child)
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color=edge_color_value,
                    linewidth=edge_linewidth,
                    alpha=edge_alpha,
                    zorder=2,
                )
                if show_edge_arrows:
                    dx = x1 - x0
                    dy = y1 - y0
                    ax.add_patch(
                        _FancyArrowPatch(
                            (x0 + 0.42 * dx, y0 + 0.42 * dy),
                            (x0 + 0.62 * dx, y0 + 0.62 * dy),
                            arrowstyle="-|>",
                            mutation_scale=arrow_size,
                            linewidth=max(0.6, 0.75 * edge_linewidth),
                            color=edge_color_value,
                            shrinkA=0.0,
                            shrinkB=0.0,
                            zorder=3,
                        )
                    )

        for node in plan.nodes():
            x, y = positions[node]
            ax.scatter(
                [x],
                [y],
                s=node_size,
                marker="o",
                color=[node_color(node)],
                edgecolors="#41464c",
                linewidths=0.65,
                zorder=4,
            )

        if show_site_labels:
            for qubit in qubits:
                node = plan.node_of_qubit[qubit]
                x, y = positions[node]
                ax.annotate(
                    f"q{qubit}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#374151",
                    zorder=5,
                )
        if show_node_ids:
            for node in plan.nodes():
                x, y = positions[node]
                ax.annotate(
                    f"n{node}",
                    (x, y),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#5b6168",
                    zorder=5,
                )

        if colorbar and n_colors:
            add_order_colorbar(
                fig,
                ax,
                colormaps,
                ScalarMappable,
                Normalize,
                cmap if color_by == "order" else node_cmap,
                n_colors,
                label=(
                    "tree order" if color_by == "order" else "tree scale"
                ),
            )

        title = "Tree tent"
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("layout x")
            ax.set_ylabel("hierarchy height")
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.14)
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
                margins=0.14,
            )
        return fig, ax

    # The public default is the structural tent view. Keep the older direct
    # route renderer private so the hierarchy cannot be mistaken for a set of
    # gate-stream legs.
    plot = plot_tent

    def plot_rubberband(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(10, 8),
        cmap="Spectral",
        color_by="gate",
        scale_cmap="viridis",
        lattice=True,
        show_gate_connectivity=True,
        show_site_nodes=True,
        colorbar=False,
        show_axes=False,
        show_title=False,
        band_alpha=0.68,
        band_linewidth=1.35,
        band_padding=0.12,
        node_size=58,
    ):
        """Plot hierarchical tree clusters as smooth rubberband regions.

        This is the physical-lattice counterpart to Quimb's contraction-tree
        ``plot_rubberband`` view: the lattice and gate connectivity remain
        grey, while each non-root tree cluster is wrapped by a rounded,
        translucent colored band. The default ``color_by="gate"`` uses a
        ``Spectral`` post-order progression, matching Cotengra's many-color
        rubberband view. ``color_by="scale"`` is available when one stable
        color is wanted for each tree scale measured from the leaves.

        The default presentation has no axes, site labels, or title. It
        returns a normal Matplotlib ``(fig, ax)`` pair for further styling.
        """
        plt, colormaps, ScalarMappable, Normalize, _FancyArrowPatch = (
            matplotlib_modules()
        )
        from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415

        if plan is None:
            plan = self.run()
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan returned by run().")
        color_by = str(color_by).replace("-", "_").strip().lower()
        color_by = {"stream": "gate", "event": "gate", "level": "scale"}.get(
            color_by, color_by
        )
        if color_by not in {"gate", "scale"}:
            raise ValueError("color_by must be 'gate' or 'scale'.")
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure

        qubits = tuple(range(plan.n))
        coords = resolve_site_coords(qubits, site_coords)
        node_scales = _tree_node_scales(plan)
        n_scales = max(node_scales.values(), default=0) + 1

        if lattice:
            for left, right in coordinate_lattice_edges(coords):
                ax.plot(
                    (coords[left][0], coords[right][0]),
                    (coords[left][1], coords[right][1]),
                    color="#c7cdd3",
                    linewidth=1.0,
                    alpha=0.62,
                    zorder=1,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(coords)
                if lattice
                else set()
            )
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    if frozenset((left, right)) in lattice_pairs:
                        continue
                    ax.plot(
                        (coords[left][0], coords[right][0]),
                        (coords[left][1], coords[right][1]),
                        color="#87919b",
                        linewidth=0.7,
                        linestyle="-",
                        alpha=0.42,
                        zorder=1,
                    )

        subtree_qubits = {}

        def gather_qubits(node):
            if node in subtree_qubits:
                return subtree_qubits[node]
            children = tuple(plan.children.get(node, ()))
            if not children:
                result = (plan.qubit_of_leaf[node],)
            else:
                result = tuple(
                    qubit
                    for child in children
                    for qubit in gather_qubits(child)
                )
                if node == plan.root and plan.root_qubit is not None:
                    result += (plan.root_qubit,)
            subtree_qubits[node] = result
            return result

        band_nodes = []

        def visit(node):
            for child in plan.children.get(node, ()):
                visit(child)
            if plan.children.get(node):
                band_nodes.append(node)

        visit(plan.root)
        n_bands = max(1, len(band_nodes))
        for band_index, node in enumerate(band_nodes):
            sites = tuple(dict.fromkeys(gather_qubits(node)))
            if len(sites) < 2:
                continue
            points = [coords[qubit] for qubit in sites]
            xmin = min(point[0] for point in points)
            xmax = max(point[0] for point in points)
            ymin = min(point[1] for point in points)
            ymax = max(point[1] for point in points)
            padding = band_padding + 0.012 * band_index
            width = max(xmax - xmin, 0.16) + 2.0 * padding
            height = max(ymax - ymin, 0.16) + 2.0 * padding
            rounding = min(0.28, 0.45 * min(width, height))
            if color_by == "scale":
                color = scale_color(
                    colormaps,
                    scale_cmap,
                    node_scales[node],
                    n_scales,
                )
            else:
                color = event_color(colormaps, cmap, band_index, n_bands)
            ax.add_patch(
                FancyBboxPatch(
                    (xmin - padding, ymin - padding),
                    width,
                    height,
                    boxstyle=f"round,pad=0,rounding_size={rounding}",
                    fill=False,
                    edgecolor=color,
                    linewidth=band_linewidth,
                    alpha=band_alpha,
                    # Draw inner/earlier contractions above outer/later
                    # bands, as in Cotengra, so overlapping bands remain
                    # individually legible.
                    zorder=3.0 + (n_bands - band_index) / n_bands,
                )
            )

        if show_site_nodes:
            ax.scatter(
                [coords[qubit][0] for qubit in qubits],
                [coords[qubit][1] for qubit in qubits],
                s=node_size,
                marker="o",
                color="#858b91",
                edgecolors="#3f454b",
                linewidths=0.75,
                zorder=5,
            )

        if colorbar and (n_bands if color_by == "gate" else n_scales):
            add_order_colorbar(
                fig,
                ax,
                colormaps,
                ScalarMappable,
                Normalize,
                cmap if color_by == "gate" else scale_cmap,
                n_bands if color_by == "gate" else n_scales,
                label=(
                    "rubberband order"
                    if color_by == "gate"
                    else "tree scale (leaf = 0)"
                ),
            )

        title = "Tree rubberband"
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("logical site x")
            ax.set_ylabel("logical site y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.14)
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
                margins=0.14,
            )
        return fig, ax

    plot_layout = plot
