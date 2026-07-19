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
coupled communities.  Binary trees remain the default and a valid special case
(``max_arity=2``).
"""

from __future__ import annotations

from ..mps.layout import (
    _gate_stream_adjacency,
    _gate_stream_pair_weights,
    _gate_stream_spectral_order,
    _normalize_layout_gate_queue,
    _normalize_layout_support,
)

__all__ = ["TreePlan", "TreeLayoutFinder"]


class TreePlan:
    """A rooted tree over ``n`` qubit leaves (any internal-node arity).

    Nodes are integer ids.  Leaves map one-to-one to qubits; internal nodes have
    one or more children.  A strictly-binary tree (every internal node with two
    children) is the common default, but the structure supports arbitrary arity
    so a level can branch into as many subtrees as the gate stream suggests.
    The plan is a pure structure description: it carries no tensor data and is
    consumed by :class:`~pepsy.optimizers.tree.TreeOptimizer` to build the tree
    tensor network.
    """

    def __init__(self, root, children, parent, qubit_of_leaf):
        self.root = root
        self.children = dict(children)
        self.parent = dict(parent)
        self.qubit_of_leaf = dict(qubit_of_leaf)
        self.leaf_of_qubit = {q: nid for nid, q in self.qubit_of_leaf.items()}
        self.n = len(self.qubit_of_leaf)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_order(cls, order, *, weights=None, structure="quality",
                   max_arity=2, community_frac=0.35, star_frac=0.75,
                   dense_max=512):
        """Build a rooted tree by recursive partition of ``order``.

        Parameters
        ----------
        order : sequence of int
            The qubit labels to place as leaves.
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
        """
        order = list(order)
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
            k = 2 if max_arity is None else max_arity
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

        root = build(order)
        return cls(root, children, parent, qubit_of_leaf)

    @classmethod
    def from_children(cls, children, qubit_of_leaf, *, root=None):
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

        leaves = set()
        for nid, ch in children.items():
            if ch:
                if nid in qubit_of_leaf:
                    raise ValueError(
                        f"internal node {nid} must not have a qubit"
                    )
            else:
                leaves.add(nid)
                if nid not in qubit_of_leaf:
                    raise ValueError(f"leaf node {nid} is missing a qubit")
        if set(qubit_of_leaf) != leaves:
            raise ValueError(
                "qubit_of_leaf must map exactly the leaf nodes"
            )
        qs = sorted(qubit_of_leaf.values())
        if qs != list(range(len(qs))):
            raise ValueError("leaf qubits must be 0..n-1 without repeats")

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
        return cls(root, children, parent, qubit_of_leaf)

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

    def node_path(self, a, b):
        """Return the node id path from node ``a`` to node ``b`` (inclusive)."""
        if a == b:
            return [a]
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
        return ancestors[: depth[lca] + 1] + list(reversed(tail))

    def tree_distance(self, qa, qb):
        """Return the leaf-to-leaf path length between qubits ``qa`` and ``qb``."""
        la = self.leaf_of_qubit[qa]
        lb = self.leaf_of_qubit[qb]
        return len(self.node_path(la, lb)) - 1

    def __repr__(self):
        n_internal = sum(1 for nid in self.nodes() if not self.is_leaf(nid))
        return (
            f"TreePlan(n={self.n}, root={self.root}, "
            f"internal_nodes={n_internal}, max_arity={self.max_arity()})"
        )


class TreeLayoutFinder:
    """Find a rooted tree structure adapted to a gate stream.

    Parameters
    ----------
    gates : bundled gate stream, optional
        ``[(gate, where), ...]`` entries.  Two-qubit ``where`` supports define
        the weighted interaction graph.  Ignored when ``supports`` is given.
    n : int, optional
        Number of qubits.  Inferred from the stream when omitted.
    supports : sequence of sequences, optional
        Explicit interaction supports, used instead of extracting them from
        ``gates``.
    structure : {"quality", "balanced", "adaptive"}
        Partition strategy passed to :meth:`TreePlan.from_order`.  ``"quality"``
        and ``"balanced"`` build strictly-binary trees when ``max_arity=2``;
        ``"adaptive"`` lets each level branch into its strongly coupled
        communities so the arity follows the gate connectivity.
    max_arity : int or None
        Maximum children per internal node (``2`` gives the binary tree; larger
        values or ``None`` give flatter / wider trees).
    community_frac : float
        Strong-edge fraction for ``structure="adaptive"`` (see
        :meth:`TreePlan.from_order`).
    star_frac : float
        Near-clique density threshold for ``structure="adaptive"`` star nodes
        (see :meth:`TreePlan.from_order`).
    dense_max : int
        Maximum subsystem size for dense spectral reordering.
    """

    def __init__(self, gates=None, n=None, *, supports=None, structure="quality",
                 max_arity=2, community_frac=0.35, star_frac=0.75,
                 dense_max=512):
        if supports is None:
            supports = self._supports_from_gates(gates)
        supports = [tuple(_normalize_layout_support(s)) for s in supports]
        self.supports = supports

        inferred = -1
        for support in supports:
            for site in support:
                if isinstance(site, int):
                    inferred = max(inferred, site)
        if n is None:
            n = inferred + 1
        if n <= 0:
            raise ValueError(
                "Could not infer qubit count; pass n explicitly."
            )
        self.n = int(n)
        self.structure = structure
        self.max_arity = None if max_arity is None else int(max_arity)
        self.community_frac = float(community_frac)
        self.star_frac = float(star_frac)
        self.dense_max = int(dense_max)

        sites = list(range(self.n))
        self.pair_weights = _gate_stream_pair_weights(supports, sites)

    @staticmethod
    def _supports_from_gates(gates):
        if gates is None:
            return []
        _, wheres, event_types = _normalize_layout_gate_queue(gates)
        supports = []
        for where, etype in zip(wheres, event_types):
            support = _normalize_layout_support(where)
            if len(support) >= 2:
                supports.append(support)
        return supports

    def run(self):
        """Return a :class:`TreePlan` for the interaction graph."""
        order = list(range(self.n))
        return TreePlan.from_order(
            order,
            weights=self._similarity_weights(),
            structure=self.structure,
            max_arity=self.max_arity,
            community_frac=self.community_frac,
            star_frac=self.star_frac,
            dense_max=self.dense_max,
        )

    def _similarity_weights(self):
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
        degree = {q: 0.0 for q in range(self.n)}
        for support in self.supports:
            for site in set(support):
                if isinstance(site, int) and 0 <= site < self.n:
                    degree[site] += 1.0
        sim = dict(self.pair_weights)
        for qi in range(self.n):
            for qj in range(qi + 1, self.n):
                deg = degree[qi] + degree[qj]
                if deg > 0.0:
                    sim[(qi, qj)] = sim.get((qi, qj), 0.0) + 1.0 / deg
        return sim

    def score(self, plan):
        """Return the total interaction-weighted tree-path length of ``plan``.

        Lower is better: this is the quantity the tree structure minimises
        (short leaf-to-leaf paths for strongly coupled qubits).
        """
        total = 0.0
        for (qa, qb), weight in self.pair_weights.items():
            total += float(weight) * plan.tree_distance(qa, qb)
        return total

    def report(self, plan=None):
        """Return layout-quality diagnostics for ``plan`` (or a fresh run).

        The dominant lever for tree-tensor-network accuracy at fixed ``chi`` is
        how well the tree keeps strongly coupled qubits as nearby leaves: a
        two-qubit gate threads its virtual bond along the whole leaf-to-leaf
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
        balanced = TreePlan.from_order(range(self.n), structure="balanced")
        balanced_score = self.score(balanced)
        return {
            "n_qubits": self.n,
            "n_interacting_pairs": n_pairs,
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
        }
