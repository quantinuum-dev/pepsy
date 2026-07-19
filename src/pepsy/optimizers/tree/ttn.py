"""A first-class tree-tensor-network state class :class:`TreeTensorNetwork`.

This is the tree analogue of ``quimb``'s :class:`quimb.tensor.MatrixProductState`:
a thin, geometry-owning subclass of ``quimb``'s arbitrary-geometry vector class
:class:`quimb.tensor.TensorNetworkGenVector`.  It carries a rooted
:class:`~pepsy.optimizers.tree.TreePlan` (internal nodes of any arity, not just
binary), a configurable physical-index /
site-tag / node-tag naming scheme, and deterministic tree-edge bond names, so
that higher-level code (notably :class:`~pepsy.optimizers.tree.TreeOptimizer`)
can talk in *node ids* and *qubit labels* while all the heavy lifting --
canonicalisation, bond compression, gate application, dense read-out, copying --
is inherited unchanged from ``quimb``.

Layout of a tree state
----------------------
* every node of the plan (leaf **and** internal) is one tensor, tagged with the
  structural node tag ``node_tag_id.format(nid)`` (default ``"N{}"``);
* leaf tensors additionally carry the ``quimb`` site tag
  ``site_tag_id.format(q)`` (default ``"I{}"``) and the physical index
  ``site_ind_id.format(q)`` (default ``"k{}"``) for qubit ``q`` -- so the
  inherited ``quimb`` site/ ``local_expectation`` machinery treats the leaves as
  the sites and the internal nodes as ancillary bond carriers;
* adjacent nodes ``a`` and ``b`` share the deterministic virtual bond index
  ``_tb{lo}_{hi}`` with ``lo, hi = sorted((a, b))``.

Because the geometry (``_plan``) and naming (``_node_tag_id``) are declared in
:attr:`TreeTensorNetwork._EXTRA_PROPS` they survive ``.copy()`` and every
``quimb`` view/selection operation, exactly like ``site_ind_id`` does for an
MPS.
"""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn
from quimb.tensor import TensorNetworkGenVector
from quimb.tensor.tensor_core import TensorNetwork

from .layout import TreePlan

__all__ = ["TreeTensorNetwork"]


def _bond_index(a, b):
    """Return the deterministic virtual-bond index name for edge ``(a, b)``."""
    lo, hi = (a, b) if a < b else (b, a)
    return f"_tb{lo}_{hi}"


def _ascii_place(s, width, col):
    """Return ``s`` padded to ``width`` with its centre aligned at column ``col``."""
    start = col - (len(s) - 1) // 2
    start = max(0, min(start, width - len(s)))
    return " " * start + s + " " * (width - start - len(s))



class TreeTensorNetwork(TensorNetworkGenVector):
    """A rooted tree-tensor-network state over qubit leaves.

    Subclasses :class:`quimb.tensor.TensorNetworkGenVector`, so it *is* a
    ``quimb`` tensor network: all of ``quimb``'s arbitrary-geometry methods
    (``canonize_around``, ``canonize_between``, ``compress_between``,
    ``gate_inds``, ``local_expectation``, ``to_dense``, ``copy`` ...) work
    directly.  The tree geometry is owned by a :class:`TreePlan`; this class adds
    the node/site/index naming glue on top.

    Prefer the builders :meth:`from_plan`, :meth:`from_order`, and :meth:`rand`
    over calling the constructor with raw tensors.

    Parameters
    ----------
    ts : sequence of quimb.tensor.Tensor or quimb.tensor.TensorNetwork
        The tensors of the network, or an existing network to cast/copy.
    plan : TreePlan
        The rooted tree structure (internal nodes may have any arity).  Required
        unless ``ts`` is an existing tensor network being copied/cast (then it
        is taken from ``ts``).
    sites : sequence, optional
        The site (qubit) labels.  Defaults to ``range(plan.n)``.
    site_tag_id, site_ind_id, node_tag_id : str
        Format strings for the ``quimb`` site tag, the physical index, and the
        structural node tag.  Defaults ``"I{}"``, ``"k{}"``, ``"N{}"``.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_site_ind_id",
        "_plan",
        "_node_tag_id",
        "_orthog_center",
    )

    def __init__(self, ts=(), *, plan=None, sites=None, site_tag_id="I{}",
                 site_ind_id="k{}", node_tag_id="N{}", **tn_opts):
        # Copy / cast path: quimb's base ``__init__`` copies ``_EXTRA_PROPS``
        # (``_plan``, ``_node_tag_id``, ...) straight off ``ts``; returning here
        # avoids clobbering them with the fresh-construction defaults below.
        if isinstance(ts, TensorNetwork):
            super().__init__(ts, **tn_opts)
            return
        super().__init__(ts, **tn_opts)
        if plan is None:
            raise ValueError(
                "TreeTensorNetwork requires a TreePlan (pass plan=...)."
            )
        self._plan = plan
        self._sites = tuple(range(plan.n)) if sites is None else tuple(sites)
        self._site_tag_id = site_tag_id
        self._site_ind_id = site_ind_id
        self._node_tag_id = node_tag_id
        # Node id of the current orthogonality centre (``None`` if unknown).
        # Tracked here -- surviving ``.copy()`` via ``_EXTRA_PROPS`` -- so the
        # canonical form is a property of the *state*, not of any one driver.
        self._orthog_center = None

    # -- geometry / naming ----------------------------------------------------

    @property
    def plan(self):
        """The :class:`TreePlan` describing the tree structure."""
        return self._plan

    @property
    def node_tag_id(self):
        """Format string for the structural node tag (e.g. ``"N{}"``)."""
        return self._node_tag_id

    @property
    def root(self):
        """The root node id of the tree."""
        return self._plan.root

    @property
    def orthogonality_center(self):
        """Node id of the tracked orthogonality centre (``None`` if unknown).

        This is the tree analogue of an MPS canonical centre: when it is a node
        ``c`` every *other* tensor is an isometry whose legs point toward ``c``
        (``absorb="right"`` convention), so the whole state norm collapses onto
        the single centre tensor.  It is updated in place by
        :meth:`shift_orthogonality_center` and :meth:`canonize_around_node_`,
        and -- being declared in :attr:`_EXTRA_PROPS` -- it survives ``.copy()``
        and every ``quimb`` view/selection, so any holder of the state (the
        :class:`~pepsy.optimizers.tree.TreeOptimizer`, a sampler, a direct user)
        reads a single consistent centre rather than tracking its own.
        """
        return getattr(self, "_orthog_center", None)

    @orthogonality_center.setter
    def orthogonality_center(self, value):
        if value is not None and value not in self._plan.children:
            raise ValueError(f"{value!r} is not a node of the tree.")
        self._orthog_center = value

    def _with_center(self, nid):
        """Set the tracked centre and return ``self`` (builder convenience)."""
        self._orthog_center = nid
        return self

    @property
    def nqubits(self):
        """Number of qubit leaves (an alias of :attr:`nsites`)."""
        return self._plan.n

    def node_tag(self, nid):
        """Return the structural tag of node ``nid``."""
        return self._node_tag_id.format(nid)

    def node_tid(self, nid):
        """Return the tensor id of node ``nid`` via a self-healing cache.

        ``quimb`` mints a fresh tensor identity whenever a tensor is rebuilt
        (e.g. ``gate_inds_`` on a leaf), so a stale cache entry simply misses the
        ``tensor_map`` membership check and is recomputed from the tag map.
        The cache lives in ``__dict__`` (not ``_EXTRA_PROPS``) so a freshly
        copied network starts with an empty, independent cache.
        """
        cache = self.__dict__.get("_node_tid_cache")
        if cache is None:
            cache = self.__dict__["_node_tid_cache"] = {}
        tid = cache.get(nid)
        if tid is not None and tid in self.tensor_map:
            return tid
        tid = next(iter(self.tag_map[self.node_tag(nid)]))
        cache[nid] = tid
        return tid

    def node_tensor(self, nid):
        """Return the live :class:`quimb.tensor.Tensor` for node ``nid``."""
        return self.tensor_map[self.node_tid(nid)]

    def bond(self, a, b):
        """Return the shared virtual-bond index name for adjacent nodes ``a``/``b``."""
        if b not in self.neighbors(a):
            raise ValueError(f"nodes {a} and {b} are not adjacent in the tree.")
        return _bond_index(a, b)

    # -- plan delegators ------------------------------------------------------

    def is_leaf(self, nid):
        """Whether node ``nid`` is a leaf (carries a physical qubit)."""
        return self._plan.is_leaf(nid)

    def parent(self, nid):
        """Return the parent node id of ``nid`` (``None`` at the root)."""
        return self._plan.parent.get(nid)

    def children(self, nid):
        """Return the child node ids of ``nid`` (empty for a leaf)."""
        return self._plan.children[nid]

    def neighbors(self, nid):
        """Return the adjacent node ids of ``nid`` (children plus parent)."""
        nbrs = list(self._plan.children[nid])
        up = self._plan.parent.get(nid)
        if up is not None:
            nbrs.append(up)
        return nbrs

    def node_path(self, a, b):
        """Return the inclusive node-id path from node ``a`` to node ``b``."""
        return self._plan.node_path(a, b)

    def leaf_of_qubit(self, q):
        """Return the leaf node id carrying qubit ``q``."""
        return self._plan.leaf_of_qubit[q]

    def qubit_of_leaf(self, nid):
        """Return the qubit label carried by leaf node ``nid``."""
        return self._plan.qubit_of_leaf[nid]

    def tree_distance(self, qa, qb):
        """Return the leaf-to-leaf path length between qubits ``qa`` and ``qb``."""
        return self._plan.tree_distance(qa, qb)

    def steiner_nodes(self, leaves):
        """Return the node set of the minimal subtree spanning ``leaves``.

        The tree has a unique path between any two nodes, so the union of the
        paths from ``leaves[0]`` to every other leaf is exactly the minimal
        connected subtree (Steiner tree) that contains all of them.
        """
        leaves = list(leaves)
        root_leaf = leaves[0]
        nodes = set()
        for lf in leaves:
            nodes.update(self._plan.node_path(root_leaf, lf))
        return nodes

    # -- edge-level canonical / compression helpers ---------------------------

    def _track_edge_center(self, a, b, absorb):
        """Update the tracked centre after a gauge move across edge ``a -> b``.

        A single ``absorb="right"`` move makes ``a`` isometric and pushes the
        centre onto ``b``; it therefore advances a centre sitting on ``a`` to
        ``b`` (and symmetrically for ``"left"``).  Any other prior centre is no
        longer the global centre after a lone edge move, so it is set to
        ``None`` (unknown) rather than left lying about the canonical form.
        """
        cur = self.orthogonality_center
        if absorb == "right" and cur == a:
            self._orthog_center = b
        elif absorb == "left" and cur == b:
            self._orthog_center = a
        else:
            self._orthog_center = None

    def canonize_edge_(self, a, b, absorb="right"):
        """Canonicalise across the tree edge ``a -> b`` in place.

        Thin wrapper over :meth:`quimb.tensor.TensorNetwork.canonize_between`
        that resolves node ids to node tags; ``absorb="right"`` leaves node ``a``
        isometric and pushes the orthogonality centre onto node ``b``.  The
        tracked :attr:`orthogonality_center` is advanced accordingly.
        """
        self.canonize_between(self.node_tag(a), self.node_tag(b), absorb=absorb)
        self._track_edge_center(a, b, absorb)
        return self

    def compress_edge_(self, a, b, *, max_bond=None, cutoff=1e-12,
                       absorb="right"):
        """Compress the tree edge ``a -> b`` in place.

        Thin wrapper over :meth:`quimb.tensor.TensorNetwork.compress_between`
        (local reduced compression, ``canonize_distance=0``) resolving node ids
        to node tags.  The tracked :attr:`orthogonality_center` is advanced as
        for :meth:`canonize_edge_` (compression moves the gauge the same way).
        """
        self.compress_between(
            self.node_tag(a),
            self.node_tag(b),
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb,
        )
        self._track_edge_center(a, b, absorb)
        return self

    def canonize_around_node_(self, nid):
        """Canonicalise the whole tree around node ``nid`` and track it as centre.

        Every non-``nid`` tensor becomes an isometry pointing toward ``nid``, so
        the state norm collapses onto the ``nid`` tensor; :attr:`orthogonality_center`
        is set to ``nid``.  This is the O(N) "establish a centre from scratch"
        path -- prefer :meth:`shift_orthogonality_center` for an incremental move
        from a *known* centre.
        """
        if nid not in self._plan.children:
            raise ValueError(f"{nid!r} is not a node of the tree.")
        self.canonize_around_(self.node_tag(nid))
        self._orthog_center = nid
        return self

    def shift_orthogonality_center(self, new, *, absorb="right"):
        """Move the tracked orthogonality centre to node ``new`` in place.

        The tree analogue of :meth:`quimb.tensor.MatrixProductState.shift_orthogonality_center`:
        the centre is walked to ``new`` along the *unique tree geodesic* from the
        current centre, canonicalising one edge at a time with a lossless QR
        (:meth:`quimb.tensor.TensorNetwork.canonize_between`).  Only the tensors
        on that path are touched, so a nearby move is O(path length), not O(N).

        * If the centre is already ``new`` this is a no-op (idempotent).
        * If the centre is currently unknown (``None``) it falls back once to the
          O(N) :meth:`canonize_around_node_`, then subsequent moves are
          incremental.

        Returns ``self`` so moves can be chained.
        """
        if new not in self._plan.children:
            raise ValueError(f"{new!r} is not a node of the tree.")
        cur = self.orthogonality_center
        if cur == new:
            return self
        if cur is None:
            return self.canonize_around_node_(new)
        path = self._plan.node_path(cur, new)
        for u, v in zip(path, path[1:]):
            self.canonize_between(self.node_tag(u), self.node_tag(v),
                                  absorb=absorb)
        self._orthog_center = new
        return self

    def is_canonical_form(self, center=None, *, tol=1e-9):
        """Return whether the tree is in canonical form about ``center``.

        Checks the defining property directly: every node other than ``center``
        must be an isometry when all its legs *except* the one pointing toward
        ``center`` are treated as inputs (i.e. ``T @ T^dagger`` over those legs is
        the identity on the toward-centre bond).  ``center`` defaults to the
        tracked :attr:`orthogonality_center`; an unknown centre returns ``False``.
        Primarily a diagnostic / test aid.
        """
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return False
        for nid in self._plan.nodes():
            if nid == center:
                continue
            toward = self._plan.node_path(nid, center)[1]
            t = self.node_tensor(nid)
            bond = next(iter(qtn.bonds(t, self.node_tensor(toward))))
            tc = t.H.reindex({bond: bond + "*"})
            prod = qtn.tensor_contract(t, tc, output_inds=[bond, bond + "*"])
            d = int(prod.shape[0])
            if not np.allclose(np.asarray(prod.data), np.eye(d), atol=tol):
                return False
        return True

    # -- dense read-out -------------------------------------------------------

    def to_statevector(self, order=None):
        """Return the dense statevector in qubit order.

        Parameters
        ----------
        order : sequence of int, optional
            Qubit order of the flattened output (default ``range(nqubits)``).
        """
        if order is None:
            order = range(self._plan.n)
        out_inds = [self.site_ind(q) for q in order]
        return np.asarray(self.to_dense(out_inds)).reshape(-1)

    # -- ascii drawing --------------------------------------------------------

    def _bond_dim(self, a, b):
        """Return the virtual-bond dimension of the tree edge ``(a, b)`` (1 if absent)."""
        ix = _bond_index(a, b)
        return int(self.ind_size(ix)) if ix in self.ind_map else 1

    def ascii_tree(self, *, bond_dims=True, node_ids=False):
        """Return a top-down ASCII drawing of the tree, drawn root-first.

        The tree analogue of a ``quimb`` MPS ``show``: the root sits at the top
        and the qubit leaves at the bottom, each internal node marked ``●`` and
        each leaf ``◆`` with its qubit label ``q{q}`` beneath it.  When
        ``bond_dims`` is true every edge is annotated with the dimension of the
        virtual bond joining a node to its parent (so growing entanglement shows
        up as growing numbers on the branches)::

                   ●
              ┌────┴────┐
              1         1
              ●         ●
            ┌─┴──┐    ┌─┴──┐
            1    1    1    1
            ◆    ◆    ◆    ◆
            q0   q1   q2   q3

        Parameters
        ----------
        bond_dims : bool
            Annotate each branch with its virtual-bond dimension (default True).
        node_ids : bool
            Also print the structural node id next to each ``●`` (default False).
        """
        plan = self._plan
        gap = 3  # blank columns between sibling subtrees

        def render(nid):
            """Return ``(lines, root_col, width)`` for the subtree rooted at ``nid``."""
            if plan.is_leaf(nid):
                label = f"q{plan.qubit_of_leaf[nid]}"
                w = max(1, len(label))
                col = (w - 1) // 2
                return [_ascii_place("◆", w, col), label.center(w)], col, w

            marker = f"●{nid}" if node_ids else "●"
            blocks = []
            for child in plan.children[nid]:
                lines, col, w = render(child)
                if bond_dims:
                    d = str(self._bond_dim(child, nid))
                    if len(d) > w:  # widen so a fat bond number still fits
                        lp = (len(d) - w) // 2
                        lines = [
                            " " * lp + ln + " " * (len(d) - w - lp)
                            for ln in lines
                        ]
                        col, w = col + lp, len(d)
                    lines = [_ascii_place(d, w, col)] + lines
                blocks.append((lines, col, w))

            # lay the child subtrees side by side, recording each root column
            offsets, cur = [], 0
            for _, _, w in blocks:
                offsets.append(cur)
                cur += w + gap
            total_w = cur - gap
            child_cols = [off + col for off, (_, col, _) in zip(offsets, blocks)]
            pcol = (child_cols[0] + child_cols[-1]) // 2

            # connector row joining the parent tick to each child stem
            conn = [" "] * total_w
            for i in range(child_cols[0], child_cols[-1] + 1):
                conn[i] = "─"
            for j, cc in enumerate(child_cols):
                conn[cc] = (
                    "┌" if j == 0
                    else "┐" if j == len(child_cols) - 1
                    else "┬"
                )
            conn[pcol] = {
                " ": "┴", "─": "┴", "┌": "├", "┐": "┤", "┬": "┼",
            }[conn[pcol]]

            # stack the child blocks under parent marker + connector rows
            height = max(len(lines) for lines, _, _ in blocks)
            body = []
            for r in range(height):
                row = []
                for k, (lines, _, w) in enumerate(blocks):
                    row.append(lines[r] if r < len(lines) else " " * w)
                    if k != len(blocks) - 1:
                        row.append(" " * gap)
                body.append("".join(row))

            lines = [
                _ascii_place(marker, total_w, pcol),
                "".join(conn),
            ] + body
            return lines, pcol, total_w

        lines, _, _ = render(plan.root)
        return "\n".join(line.rstrip() for line in lines)

    def show(self, *, bond_dims=True, node_ids=False):
        """Print the top-down ASCII drawing of the tree (see :meth:`ascii_tree`).

        The tree analogue of a ``quimb`` MPS ``show``: the root is at the top,
        the qubit leaves at the bottom, internal nodes are ``●`` and leaves
        ``◆`` labelled with their qubit, and every edge is annotated with its
        current virtual-bond dimension.
        """
        print(self.ascii_tree(bond_dims=bond_dims, node_ids=node_ids))


    # -- builders -------------------------------------------------------------

    @classmethod
    def from_plan(cls, plan, *, dtype=complex, site_tag_id="I{}",
                  site_ind_id="k{}", node_tag_id="N{}"):
        """Build the product state ``|0...0>`` on the geometry ``plan``.

        Every virtual bond starts at dimension 1, so the state is trivially in
        canonical form (each tensor is an isometry) with the root as the
        orthogonality centre.
        """
        tensors = []
        for nid in plan.nodes():
            inds = []
            shape = []
            tags = [node_tag_id.format(nid)]
            leaf = plan.is_leaf(nid)
            if leaf:
                q = plan.qubit_of_leaf[nid]
                inds.append(site_ind_id.format(q))
                shape.append(2)
                tags.append(site_tag_id.format(q))
            for child in plan.children[nid]:
                inds.append(_bond_index(nid, child))
                shape.append(1)
            up = plan.parent.get(nid)
            if up is not None:
                inds.append(_bond_index(nid, up))
                shape.append(1)
            if leaf:
                data = np.zeros(shape, dtype=dtype)
                data[tuple([0] * len(shape))] = 1.0  # |0>
            else:
                data = np.ones(shape, dtype=dtype)
            tensors.append(qtn.Tensor(data, inds=inds, tags=tags))
        return cls(
            tensors,
            plan=plan,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )._with_center(plan.root)

    @classmethod
    def from_order(cls, order, *, weights=None, structure="quality",
                   max_arity=2, community_frac=0.35, star_frac=0.75,
                   dtype=complex, site_tag_id="I{}", site_ind_id="k{}",
                   node_tag_id="N{}"):
        """Build a product state on a tree partitioned from ``order``.

        Convenience wrapper that first builds a :class:`TreePlan` with
        :meth:`TreePlan.from_order` and then :meth:`from_plan`.  ``max_arity``
        and ``structure`` control the tree shape (see
        :meth:`TreePlan.from_order`); the defaults reproduce the binary tree.
        """
        plan = TreePlan.from_order(
            order, weights=weights, structure=structure,
            max_arity=max_arity, community_frac=community_frac,
            star_frac=star_frac,
        )
        return cls.from_plan(
            plan,
            dtype=dtype,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )

    @classmethod
    def rand(cls, plan, D=4, *, phys_dim=2, dtype=complex, seed=None,
             canonicalize=True, site_tag_id="I{}", site_ind_id="k{}",
             node_tag_id="N{}"):
        """Build a random tree state with virtual bond dimension ``D``.

        Every virtual (tree-edge) bond is given dimension ``D`` and every leaf a
        physical dimension ``phys_dim``.  Useful for tests and benchmarks.  When
        ``canonicalize`` is true the state is canonicalised around the root.
        """
        rng = np.random.default_rng(seed)
        is_complex = np.iscomplexobj(np.zeros(1, dtype=dtype))

        def _rand(shape):
            arr = rng.standard_normal(shape)
            if is_complex:
                arr = arr + 1j * rng.standard_normal(shape)
            return arr.astype(dtype)

        tensors = []
        for nid in plan.nodes():
            inds = []
            shape = []
            tags = [node_tag_id.format(nid)]
            leaf = plan.is_leaf(nid)
            if leaf:
                q = plan.qubit_of_leaf[nid]
                inds.append(site_ind_id.format(q))
                shape.append(phys_dim)
                tags.append(site_tag_id.format(q))
            for child in plan.children[nid]:
                inds.append(_bond_index(nid, child))
                shape.append(D)
            up = plan.parent.get(nid)
            if up is not None:
                inds.append(_bond_index(nid, up))
                shape.append(D)
            tensors.append(qtn.Tensor(_rand(shape), inds=inds, tags=tags))
        ttn = cls(
            tensors,
            plan=plan,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )
        if canonicalize:
            ttn.canonize_around_node_(plan.root)
        return ttn

    # -- repr -----------------------------------------------------------------

    def __repr__(self):
        return (
            f"{type(self).__name__}(nqubits={self.nqubits}, "
            f"ntensors={self.num_tensors}, max_bond={self.max_bond()})"
        )
