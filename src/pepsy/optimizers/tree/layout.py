"""Tree-structure search for :class:`TreeOptimizer` gate-stream replay.

The paper *Simulating quantum circuits using tree tensor networks*
(Seitz, Medina, Cruz, Huang, Mendl; Quantum 7, 964, 2023; arXiv:2206.01000)
first fixes a rooted tree adapted to the entanglement the circuit is expected
to generate, then applies gates to it.

:class:`TreeLayoutFinder` builds that structure from the two-qubit connectivity
of a bundled gate stream.  It reuses the interaction-graph and recursive
spectral-bisection machinery written for the MPS layout finder
(:mod:`pepsy.optimizers.mps.layout`); where the MPS finder *flattens* the
bisection recursion into a 1D order, the tree finder *keeps* the bisection
dendrogram as the rooted binary tree structure.  Strongly coupled qubits end
up as nearby leaves, minimising the tree-path length that two-qubit gates must
thread across.
"""

from __future__ import annotations

from ..mps.layout import (
    _gate_stream_pair_weights,
    _gate_stream_spectral_order,
    _normalize_layout_gate_queue,
    _normalize_layout_support,
)

__all__ = ["TreePlan", "TreeLayoutFinder"]


class TreePlan:
    """A rooted binary tree over ``n`` qubit leaves.

    Nodes are integer ids.  Leaves map one-to-one to qubits; internal nodes have
    two children.  The plan is a pure structure description: it carries no
    tensor data and is consumed by :class:`~pepsy.optimizers.tree.TreeOptimizer`
    to build the tree tensor network.
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
    def from_order(cls, order, *, weights=None, structure="quality", dense_max=512):
        """Build a balanced binary tree by recursive bisection of ``order``.

        Parameters
        ----------
        order : sequence of int
            The qubit labels to place as leaves.
        weights : mapping, optional
            Unordered ``(qi, qj) -> weight`` interaction weights.  When given
            with ``structure="quality"`` each recursion level is split by a
            spectral (Fiedler) order of the induced subgraph.
        structure : {"quality", "balanced"}
            ``"quality"`` uses spectral bisection (entanglement adapted);
            ``"balanced"`` splits the given ``order`` in half at each level.
        """
        order = list(order)
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

        def build(qs):
            qs = list(qs)
            if len(qs) == 1:
                nid = new_node()
                children[nid] = ()
                qubit_of_leaf[nid] = qs[0]
                return nid
            if structure == "quality" and len(qs) > 2:
                sub = induced(qs)
                spectral = _gate_stream_spectral_order(qs, sub, dense_max=dense_max)
                if spectral:
                    qs = spectral
            mid = len(qs) // 2
            left = build(qs[:mid])
            right = build(qs[mid:])
            nid = new_node()
            children[nid] = (left, right)
            parent[left] = nid
            parent[right] = nid
            return nid

        root = build(order)
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
            f"internal_nodes={n_internal})"
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
    structure : {"quality", "balanced"}
        Bisection strategy passed to :meth:`TreePlan.from_order`.
    dense_max : int
        Maximum subsystem size for dense spectral bisection.
    """

    def __init__(self, gates=None, n=None, *, supports=None, structure="quality",
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
