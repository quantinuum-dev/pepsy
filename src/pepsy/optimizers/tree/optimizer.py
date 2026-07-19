"""Tree-tensor-network gate-stream simulator centered on :class:`TreeOptimizer`.

:class:`TreeOptimizer` implements the rooted tree-tensor-network circuit
simulator of *Simulating quantum circuits using tree tensor networks*
(Seitz, Medina, Cruz, Huang, Mendl; Quantum 7, 964, 2023; arXiv:2206.01000).

A quantum state is stored as a rooted tree tensor network whose leaves
carry the physical qubit indices.  Internal nodes may have any arity: the
default structure is a strictly-binary tree, but flatter ``k``-ary trees
(``max_arity``) or gate-connectivity-driven communities
(``structure="adaptive"``) are supported unchanged.  A bundled gate stream
``[(gate, where), ...]`` is replayed:

* single-qubit gates are absorbed into the leaf tensor (no bond growth); a
  unitary one-qubit gate preserves the tree canonical form regardless of where
  the orthogonality centre sits;
* two-qubit gates on leaves ``a`` and ``b`` are SVD-split into two factors
  joined by a virtual bond; the factors are absorbed into the two leaves and
  the virtual bond is *threaded exactly* (no truncation) along the tree path
  from ``a`` to ``b``.  Only once both factors are in place is a single
  canonical compression sweep run back along the path, truncating every
  touched bond to ``chi`` -- so each truncation sees the complete gate, which
  is markedly more accurate under a finite ``chi`` than truncating each hop as
  the bond is threaded (Seitz et al., Figs. 3-6).

The orthogonality centre is tracked as a node id and moved *smartly* along the
tree geodesic with per-edge canonicalisation, mirroring the
``info_c["cur_orthog"]`` centre tracking of :class:`pepsy.MpsOptimizer`.  The
tree structure is chosen by :class:`TreeLayoutFinder` (entanglement-adapted
recursive partition) unless an explicit :class:`TreePlan` is supplied.

This is the tensor-network glue only: the heavy lifting (arbitrary-geometry
canonicalisation, bond compression, tensor splitting, tree path finding) uses
``quimb`` primitives.
"""

from __future__ import annotations

import contextlib
from numbers import Integral

import numpy as np
import quimb.tensor as qtn

from ...operators.gates import _normalize_gate_entries
from .layout import TreeLayoutFinder, TreePlan
from .ttn import TreeTensorNetwork

__all__ = ["TreeOptimizer"]

try:  # threadpoolctl is a NumPy/SciPy transitive dependency; treat as optional.
    from threadpoolctl import ThreadpoolController as _ThreadpoolController

    # Built once (it scans the loaded BLAS/OpenMP libraries): reused per gate so
    # ``.limit(...)`` is a cheap set/restore rather than a fresh library scan.
    _THREAD_CONTROLLER = _ThreadpoolController()
except Exception:  # pragma: no cover - threadpoolctl missing
    _THREAD_CONTROLLER = None


def _normalize_where(where):
    """Return a tuple of int qubit labels for a gate support."""
    if isinstance(where, Integral):
        return (int(where),)
    return tuple(int(site) for site in where)


class TreeOptimizer:
    """Replay a bundled gate stream on a rooted tree tensor network.

    Parameters
    ----------
    gates : bundled gate stream, optional
        ``[(gate, where), ...]`` entries with ``where`` an int (one-qubit) or a
        pair of ints (two-qubit).  Replayed eagerly on construction when given.
    n : int, optional
        Number of qubits.  Inferred from ``gates`` / ``tree`` when omitted.
    chi : int
        Maximum virtual bond dimension enforced during two-qubit threading.
    cutoff : float
        Relative singular-value cutoff for truncations.
    structure : {"quality", "balanced", "adaptive"}
        Tree-structure strategy used when ``tree`` is not supplied.
    max_arity : int or None
        Maximum children per internal node for the auto-built structure.  ``2``
        (default) gives a binary tree; larger values or ``None`` give flatter /
        wider trees.  Ignored when an explicit ``tree`` is supplied.
    tree : TreePlan, optional
        Explicit tree structure (any arity).  When omitted a
        :class:`TreeLayoutFinder` builds one from the gate stream.
    dtype : numpy dtype
        Data type of the initial product state (default ``complex128``).
    threads : int or None
        BLAS/OpenMP thread cap applied around gate application and the heavy
        contraction read-outs.  Tree tensors are small (rank ``<= 3``, bounded
        by ``chi``), so multi-threaded linear algebra is dominated by thread
        launch/synchronisation overhead: capping to ``1`` (the default) makes
        replay both markedly faster and stable in wall-clock time.  Pass
        ``None`` to leave the ambient thread count untouched (worthwhile only in
        a large-``chi`` regime where a single contraction is itself large).
    seed : int or None
        Seed for the internal random generator used by :meth:`measure` and
        :meth:`reset`.
    run : bool
        Whether to replay ``gates`` immediately (default ``True``).

    Attributes
    ----------
    tn : TreeTensorNetwork
        The live tree tensor network (a geometry-owning ``quimb`` subclass).
    plan : TreePlan
        The tree structure.
    center : int or None
        Node id of the current orthogonality centre (``None`` if unknown).
    """

    def __init__(self, gates=None, n=None, *, chi=64, cutoff=1e-12,
                 structure="quality", max_arity=2, community_frac=0.35,
                 star_frac=0.75, tree=None, dtype=complex, threads=1,
                 seed=None, run=True):
        self.G, self.where = self._normalize_gate_queue(gates)

        if n is None:
            if tree is not None:
                n = tree.n
            else:
                n = 1 + max(
                    (max(w) for w in self.where if len(w) > 0),
                    default=-1,
                )
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("Could not infer qubit count; pass n explicitly.")

        self.chi = int(chi)
        self.cutoff = float(cutoff)
        self.structure = structure
        self.max_arity = None if max_arity is None else int(max_arity)
        self.community_frac = float(community_frac)
        self.star_frac = float(star_frac)
        self.dtype = dtype
        self.threads = None if threads is None else int(threads)
        self.rng = np.random.default_rng(seed)

        if tree is None:
            tree = TreeLayoutFinder(
                gates=gates, n=self.n, structure=structure,
                max_arity=self.max_arity, community_frac=self.community_frac,
                star_frac=self.star_frac,
            ).run()
        if not isinstance(tree, TreePlan):
            raise TypeError("tree must be a TreePlan or None.")
        self.plan = tree

        self.tn = self._build_product_state()
        # A freshly built product state has every virtual bond at dimension 1,
        # so each tensor is trivially isometric and the state is already in
        # canonical form with the root as orthogonality centre (norm 1).
        # ``from_plan`` records that centre on the network; assert it here so the
        # first gate skips an O(N) canonicalisation.
        self.center = self.plan.root
        self._thread_ind = None

        if run and self.G:
            self.run()

    # -- stream normalization -------------------------------------------------

    @staticmethod
    def _normalize_gate_queue(gates):
        if gates is None:
            return [], []
        entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
        payloads = [g for g, _ in entries]
        wheres = [_normalize_where(w) for _, w in entries]
        return payloads, wheres

    # -- construction ---------------------------------------------------------

    def _phys(self, q):
        return self.tn.site_ind(q)

    def _tag(self, nid):
        return self.tn.node_tag(nid)

    def _tid(self, nid):
        """Return the tensor id of node ``nid`` (self-healing cache on the TTN)."""
        return self.tn.node_tid(nid)

    def _thread_ctx(self):
        """Context manager capping BLAS/OpenMP threads for the small tree ops."""
        if _THREAD_CONTROLLER is not None and self.threads is not None:
            return _THREAD_CONTROLLER.limit(limits=self.threads)
        return contextlib.nullcontext()

    def _bond_name(self, u, v):
        lo, hi = (u, v) if u < v else (v, u)
        return f"_tb{lo}_{hi}"

    def _neighbors(self, nid):
        """Return the adjacent node ids of ``nid`` (children plus parent)."""
        return self.tn.neighbors(nid)

    def _steiner_nodes(self, leaves):
        """Return the node set of the minimal subtree spanning ``leaves``."""
        return self.tn.steiner_nodes(leaves)

    def _build_product_state(self):
        return TreeTensorNetwork.from_plan(self.plan, dtype=self.dtype)

    # -- canonical centre tracking -------------------------------------------

    @property
    def center(self):
        """Node id of the current orthogonality centre (``None`` if unknown).

        A thin view onto the *single* centre tracked by the underlying
        :class:`TreeTensorNetwork` (:attr:`TreeTensorNetwork.orthogonality_center`),
        so the optimizer and the state can never disagree about the canonical
        form; it is carried across :meth:`copy` with the network.
        :attr:`orthogonality_center` is a name-parity alias.
        """
        return self.tn.orthogonality_center

    @center.setter
    def center(self, value):
        self.tn.orthogonality_center = value

    @property
    def orthogonality_center(self):
        """Alias of :attr:`center` matching :attr:`TreeTensorNetwork.orthogonality_center`."""
        return self.tn.orthogonality_center

    @orthogonality_center.setter
    def orthogonality_center(self, value):
        self.tn.orthogonality_center = value

    def _move_center(self, target):
        """Move the orthogonality centre to node ``target`` along the geodesic.

        Delegates to :meth:`TreeTensorNetwork.shift_orthogonality_center`: a
        no-op when already centred, an incremental per-edge QR walk along the
        tree geodesic from a known centre, or a single O(N) canonicalisation
        when the centre is unknown.
        """
        self.tn.shift_orthogonality_center(target)

    def shift_orthogonality_center(self, node):
        """Move the orthogonality centre to ``node`` along the tree geodesic.

        Public entry point to the same incremental per-edge canonicalisation the
        optimizer runs internally before gates and read-outs: the centre is
        walked to ``node`` with a lossless per-edge QR (a no-op when already
        centred, or a single O(N) canonicalisation when the centre is unknown),
        mirroring :meth:`TreeTensorNetwork.shift_orthogonality_center` and the
        MPS ``shift_orthogonality_center``.  Returns ``self`` so moves chain.
        """
        self._move_center(node)
        return self

    def is_canonical_form(self, center=None, *, tol=1e-9):
        """Whether the state is in canonical form about ``center``.

        ``center`` defaults to the tracked :attr:`center`.  Delegates to
        :meth:`TreeTensorNetwork.is_canonical_form`: every non-centre tensor must
        be an isometry pointing toward the centre.  A diagnostic / test aid.
        """
        return self.tn.is_canonical_form(center, tol=tol)

    @property
    def canonical_region(self):
        """Frozenset of node ids forming the canonicalised subtree (``None`` if unknown).

        A thin view onto :attr:`TreeTensorNetwork.canonical_region`, the range /
        subtree generalisation of :attr:`center`: every tensor outside the region
        points inward toward it.  Assigning validates connectedness.
        """
        return self.tn.canonical_region

    @canonical_region.setter
    def canonical_region(self, value):
        self.tn.canonical_region = value

    def canonize_subtree(self, nodes, *, span=False):
        """Canonicalise the state around the connected subtree ``nodes``.

        The range / subtree generalisation of :meth:`shift_orthogonality_center`:
        every tensor outside the subtree is gauged to point inward, so the whole
        state norm is carried by the subtree tensors.  Delegates to
        :meth:`TreeTensorNetwork.canonize_subtree_`; pass ``span=True`` to
        auto-expand to the minimal connected subtree spanning ``nodes``.  Returns
        ``self`` so calls chain.
        """
        self.tn.canonize_subtree_(nodes, span=span)
        return self

    def canonize_around_qubits(self, qubits):
        """Canonicalise around the minimal subtree spanning ``qubits``.

        The qubit-level "range canonicalisation" entry point: gauge every tensor
        outside the minimal connected subtree spanning the given qubits' leaves
        to point inward, so the reduced state on those qubits is captured by that
        subtree.  Delegates to :meth:`TreeTensorNetwork.canonize_around_qubits_`.
        Returns ``self``.
        """
        self.tn.canonize_around_qubits_(qubits)
        return self

    def is_subtree_canonical_form(self, nodes=None, *, span=False, tol=1e-9):
        """Whether the state is canonical about the subtree ``nodes``.

        ``nodes`` defaults to the tracked :attr:`canonical_region`.  Delegates to
        :meth:`TreeTensorNetwork.is_subtree_canonical_form`: every tensor outside
        the subtree must be an isometry pointing inward.  A diagnostic / test aid.
        """
        return self.tn.is_subtree_canonical_form(nodes, span=span, tol=tol)

    # -- gate application -----------------------------------------------------

    def run(self, gates=None):
        """Replay ``gates`` (or the construction stream) on the tree. Returns self."""
        if gates is not None:
            self.G, self.where = self._normalize_gate_queue(gates)
        for gate, where in zip(self.G, self.where):
            self.apply_gate(gate, where)
        return self

    def apply_gate(self, gate, where):
        """Apply a one- or two-qubit gate at qubit support ``where``."""
        where = _normalize_where(where)
        with self._thread_ctx():
            if len(where) == 1:
                self.apply_1q(gate, where[0])
            elif len(where) == 2:
                if where[0] == where[1]:
                    raise ValueError(
                        "A two-qubit gate needs two distinct qubits; "
                        f"got where={where}."
                    )
                self.apply_2q(gate, where[0], where[1])
            else:
                raise NotImplementedError(
                    "TreeOptimizer supports one- and two-qubit gates only."
                )

    def apply_1q(self, gate, q):
        """Absorb a one-qubit gate into the leaf tensor of qubit ``q``."""
        gate = np.asarray(gate).reshape(2, 2)
        self.tn.gate_inds_(gate, [self._phys(q)], contract=True)

    def apply_2q(self, gate, qa, qb):
        """Apply a two-qubit gate to leaves ``qa`` and ``qb``.

        Following Seitz et al. (Figs. 3-6): SVD-split the gate into two factors
        joined by a virtual bond, absorb the left factor into leaf ``a`` and the
        right into leaf ``b``, threading the virtual bond *exactly* through the
        intermediate nodes along the tree geodesic.  Only once **both** factors
        are present is a single canonical compression sweep run back along the
        path, so every bond truncation sees the complete gate.
        """
        plan = self.plan
        la = plan.leaf_of_qubit[qa]
        lb = plan.leaf_of_qubit[qb]
        pa, pb = self._phys(qa), self._phys(qb)

        # Fast path: sibling leaves share a parent, so the gate correlation is
        # carried entirely by the parent blob -- a single two-site update
        # (contract the three tensors, apply the gate, re-split by SVD) avoids
        # the QR threading and double-bond fusion of the general geodesic route.
        parent = plan.parent.get(la)
        if parent is not None and plan.parent.get(lb) == parent:
            self._apply_2q_siblings(gate, qa, qb, la, lb, parent)
            return

        # Centre on the source leaf so the exact threading pushes an isometric
        # front toward leaf b (the orthogonality centre rides the virtual bond).
        self._move_center(la)

        gate = np.asarray(gate).reshape(2, 2, 2, 2)  # (out_a, out_b, in_a, in_b)
        self._thread_ind = qtn.rand_uuid()
        gt = qtn.Tensor(gate, inds=("_na", "_nb", pa, pb))
        left, right = gt.split(
            left_inds=("_na", pa),
            method="svd",
            cutoff=0.0,
            absorb="both",
            get="tensors",
            bond_ind=self._thread_ind,
        )

        # Absorb the left factor into leaf a; its new physical index keeps name.
        ta = self.tn.tensor_map[self._tid(la)]
        merged_a = qtn.tensor_contract(ta, left).reindex_({"_na": pa})
        ta.modify(data=merged_a.data, inds=merged_a.inds)

        # Thread the virtual bond exactly along the tree geodesic to leaf b.
        path = plan.node_path(la, lb)
        for u, v in zip(path, path[1:]):
            self._thread_hop(u, v)

        # Absorb the right factor into leaf b, consuming the virtual bond.
        tb = self.tn.tensor_map[self._tid(lb)]
        merged_b = qtn.tensor_contract(right, tb).reindex_({"_nb": pb})
        tb.modify(data=merged_b.data, inds=merged_b.inds)
        self.center = lb
        self._thread_ind = None

        # Canonical compression sweep back along the path, now with both gate
        # factors present, truncating every touched bond to ``chi``.
        self._compress_path(path)

    def _apply_2q_siblings(self, gate, qa, qb, la, lb, parent):
        """Apply a two-qubit gate to sibling leaves sharing ``parent``.

        The two leaves and their shared parent are contracted into one blob, the
        gate is applied, and the blob is re-split by two truncating SVDs -- a
        direct two-site update (cf. the MPS two-site gate) that needs no bond
        threading because both leaves already meet at ``parent``.  With the
        centre first moved onto ``parent`` the surrounding tree is isometric, so
        each SVD truncates against an isometric environment and both gate
        factors are present before any truncation.
        """
        pa, pb = self._phys(qa), self._phys(qb)
        self._move_center(parent)
        tla = self.tn.tensor_map[self._tid(la)]
        tp = self.tn.tensor_map[self._tid(parent)]
        tlb = self.tn.tensor_map[self._tid(lb)]
        e_la = self._bond_name(la, parent)
        e_lb = self._bond_name(lb, parent)

        blob = qtn.tensor_contract(tla, tp, tlb)
        g = qtn.Tensor(
            np.asarray(gate).reshape(2, 2, 2, 2),
            inds=(pa + "*", pb + "*", pa, pb),
        )
        blob = qtn.tensor_contract(blob, g).reindex_(
            {pa + "*": pa, pb + "*": pb}
        )

        # Split off leaf a (isometric), then leaf b, leaving the centre at the
        # parent; both new bonds keep their canonical tree-edge names.
        la_t, rem = blob.split(
            left_inds=[pa], method="svd", max_bond=self.chi,
            cutoff=self.cutoff, absorb="right", get="tensors", bond_ind=e_la,
        )
        lb_t, p_t = rem.split(
            left_inds=[pb], method="svd", max_bond=self.chi,
            cutoff=self.cutoff, absorb="right", get="tensors", bond_ind=e_lb,
        )
        tla.modify(data=la_t.data, inds=la_t.inds)
        tlb.modify(data=lb_t.data, inds=lb_t.inds)
        tp.modify(data=p_t.data, inds=p_t.inds)
        self.center = parent

    def _thread_hop(self, u, v):
        """Thread the virtual bond exactly from node ``u`` to adjacent ``v``.

        The bond is moved by an *economical QR* factorisation (Seitz et al.,
        Fig. 6): the intermediate node keeps its isometric ``Q`` factor and the
        upper-triangular ``R`` carries the virtual bond forward to ``v``, moving
        the orthogonality centre with it.  QR is lossless and cheaper than the
        SVD used for the final truncating sweep, so a single gate grows a
        crossed bond by at most the gate rank ``k <= 4`` above its pre-gate
        size, and that growth is undone by :meth:`_compress_path`.
        """
        tu = self.tn.tensor_map[self._tid(u)]
        tv = self.tn.tensor_map[self._tid(v)]
        edge = next(iter(qtn.bonds(tu, tv)))
        left_inds = [ix for ix in tu.inds if ix not in (edge, self._thread_ind)]
        keep, carry = tu.split(
            left_inds=left_inds,
            method="qr",
            absorb="right",
            get="tensors",
        )
        merged_v = qtn.tensor_contract(carry, tv)
        tu.modify(data=keep.data, inds=keep.inds)
        tv.modify(data=merged_v.data, inds=merged_v.inds)

    def _compress_path(self, path):
        """Canonically compress every bond along ``path`` down to ``chi``.

        The orthogonality centre sits at ``path[-1]`` on entry; sweeping back to
        ``path[0]`` with ``absorb="right"`` truncates each bond with an isometric
        environment (``canonize_distance=0`` local reduced compression) and
        leaves the centre at ``path[0]``.  This is the re-orthonormalisation
        sweep of Seitz et al. (Fig. 6) applied along the gate geodesic.
        """
        for v, u in zip(path[::-1], path[-2::-1]):
            self.tn.compress_between(
                self._tag(v),
                self._tag(u),
                max_bond=self.chi,
                cutoff=self.cutoff,
                absorb="right",
            )
        self.center = path[0]

    # -- readout --------------------------------------------------------------

    def max_bond(self):
        """Return the largest virtual bond dimension in the tree."""
        return self.tn.max_bond()

    def show(self, *, bond_dims=True, node_ids=False):
        """Print a top-down ASCII drawing of the tree with current bond dims.

        Delegates to :meth:`TreeTensorNetwork.show`: the root sits at the top,
        the qubit leaves at the bottom, internal nodes are ``●``, leaves ``◆``
        labelled with their qubit, and each edge carries its virtual-bond
        dimension -- the tree analogue of a ``quimb`` MPS ``show``.
        """
        self.tn.show(bond_dims=bond_dims, node_ids=node_ids)

    def bond_report(self):
        """Return a summary of the current virtual bond dimensions.

        A quick health check over the tree edges (inner indices): the maximum
        and mean bond dimension, the number of bonds and tensors, and the
        requested ``chi``.  Bonds pinned at ``chi`` mean the truncation is
        active (raise ``chi`` for more accuracy); bonds well below ``chi`` mean
        ``chi`` is not the accuracy bottleneck.
        """
        bonds = self.tn.inner_inds()
        bond_dims = [int(self.tn.ind_size(ix)) for ix in bonds]
        return {
            "max_bond": max(bond_dims) if bond_dims else 1,
            "mean_bond": (
                float(sum(bond_dims) / len(bond_dims)) if bond_dims else 1.0
            ),
            "n_bonds": len(bond_dims),
            "n_tensors": self.tn.num_tensors,
            "chi": self.chi,
        }

    def to_dense(self):
        """Return the dense statevector with index order ``k0, k1, ..., k(n-1)``."""
        with self._thread_ctx():
            return self.tn.to_statevector(range(self.n))

    def norm(self):
        """Return the state norm ``<psi|psi>**0.5``.

        When the orthogonality centre is known the norm is the norm of that
        single centre tensor: every other tensor is isometric and telescopes to
        the identity between bra and ket.  This is the tree analogue of the
        one-site canonical norm tracked by :class:`pepsy.MpsOptimizer`.  With an
        unknown centre it falls back to the full doubled-tree contraction.
        """
        if self.center is not None:
            t = self.tn.tensor_map[self._tid(self.center)]
            val = qtn.tensor_contract(t.H, t, output_inds=[])
            return float(np.sqrt(np.abs(val)))
        with self._thread_ctx():
            val = (self.tn.H & self.tn).contract(output_inds=[])
        return float(np.sqrt(np.abs(val)))

    def normalize(self):
        """Normalise the state in place. Returns the previous norm."""
        nrm = self.norm()
        if nrm > 0:
            tid = self._tid(self.center) if self.center is not None else self._tid(
                self.plan.root
            )
            t = self.tn.tensor_map[tid]
            t.modify(data=t.data / nrm)
        return nrm

    def local_expectation(self, op, where):
        """Return ``<psi|op|psi> / <psi|psi>`` for an operator on ``where``.

        For a single-site operator this moves the tracked orthogonality centre
        onto the target leaf and contracts only that centre tensor with ``op``:
        every other tensor is isometric and cancels between bra and ket, the
        tree analogue of ``MpsOptimizer.local_expectation_canonical``.  A
        multi-site operator is contracted over only the **minimal subtree
        spanning the target leaves**: with the centre inside that subtree every
        tensor outside it is isometric and its bra/ket copies cancel to the
        identity on the shared boundary bond, so the cost scales with the
        operator's spread rather than the whole tree.
        """
        where = _normalize_where(where)
        if len(where) == 1:
            q = where[0]
            leaf = self.plan.leaf_of_qubit[q]
            self._move_center(leaf)
            t = self.tn.tensor_map[self._tid(leaf)]
            p = self._phys(q)
            mat = np.asarray(op).reshape(2, 2)
            gt = qtn.Tensor(mat, inds=(p + "*", p))
            bra = t.H.reindex_({p: p + "*"})
            num = qtn.tensor_contract(bra, gt, t, output_inds=[])
            den = qtn.tensor_contract(t.H, t, output_inds=[])
            return num / den

        phys = [self._phys(q) for q in where]
        leaves = [self.plan.leaf_of_qubit[q] for q in where]
        snodes = self._steiner_nodes(leaves)
        # The reduced contraction is exact only when the centre lies inside the
        # spanning subtree (then the isometric exterior cancels); ensure it.
        if self.center not in snodes:
            self._move_center(leaves[0])
        # Bonds internal to the subtree must be renamed in the bra so they do not
        # collide with the ket; boundary bonds stay shared so the isometric
        # exterior contributes an identity between bra and ket.
        internal = set()
        for nid in snodes:
            for nb in self._neighbors(nid):
                if nb in snodes:
                    internal.add(self._bond_name(nid, nb))
        ket_ts = [self.tn.tensor_map[self._tid(nid)].copy() for nid in snodes]
        ket = qtn.TensorNetwork(ket_ts)
        mat = np.asarray(op).reshape([2] * (2 * len(where)))
        gt = qtn.Tensor(mat, inds=[p + "*" for p in phys] + phys)
        with self._thread_ctx():
            internal_map = {ix: qtn.rand_uuid() for ix in internal}
            bra_num = ket.H.reindex(
                {**internal_map, **{p: p + "*" for p in phys}}
            )
            num = (bra_num & gt & ket).contract(output_inds=[])
            bra_den = ket.H.reindex(internal_map)
            den = (bra_den & ket).contract(output_inds=[])
        return num / den

    def measure(self, q, outcome=None):
        """Projectively measure qubit ``q`` in the computational basis.

        Moves the orthogonality centre onto the leaf, reads the single-site Born
        probabilities from that one canonical tensor, samples (or forces via
        ``outcome``) a result, projects the leaf onto it, and renormalises.
        Returns the outcome bit.  Because the centre sits on the leaf the
        probabilities are exact regardless of the global state norm.
        """
        with self._thread_ctx():
            leaf = self.plan.leaf_of_qubit[q]
            self._move_center(leaf)
            t = self.tn.tensor_map[self._tid(leaf)]
            p = self._phys(q)
            ax = t.inds.index(p)
            arr = np.moveaxis(t.data, ax, 0).reshape(2, -1)
            w = np.einsum("ij,ij->i", arr, arr.conj()).real
            total = float(w.sum())
            if total <= 0:
                raise ValueError("Cannot measure a zero-norm state.")
            probs = np.clip(w, 0.0, None)
            probs = probs / probs.sum()
            if outcome is None:
                outcome = int(self.rng.choice(2, p=probs))
            else:
                outcome = int(outcome)
            proj = np.zeros((2, 2), dtype=self.dtype)
            proj[outcome, outcome] = 1.0
            self.apply_1q(proj, q)
            self.normalize()
        return outcome

    def reset(self, q):
        """Reset qubit ``q`` to ``|0>`` (measure, then flip if it was ``|1>``)."""
        if self.measure(q) == 1:
            x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=self.dtype)
            self.apply_1q(x, q)
        return 0

    def copy(self):
        """Return an independent optimizer at the current tree state.

        The returned optimizer owns its own copy of the live tensor network --
        which carries the tracked orthogonality centre with it -- so it can be
        evolved separately, useful for branching experiments or trial gate
        sequences.  The immutable :class:`TreePlan` is shared; the gate queue is
        retained (gate payloads are not copied) but not replayed.
        """
        other = type(self)(
            None,
            n=self.n,
            chi=self.chi,
            cutoff=self.cutoff,
            structure=self.structure,
            tree=self.plan,
            dtype=self.dtype,
            threads=self.threads,
            run=False,
        )
        # ``TreeTensorNetwork.copy`` carries ``_canonical_region`` (an
        # _EXTRA_PROPS field), so the copied network already reports the right
        # centre / canonical region.
        other.tn = self.tn.copy()
        other.G = list(self.G)
        other.where = list(self.where)
        return other

    @classmethod
    def find_tree_layout(cls, gates, n=None, *, structure="quality",
                         max_arity=2, community_frac=0.35, star_frac=0.75):
        """Return the :class:`TreePlan` a :class:`TreeLayoutFinder` would use."""
        return TreeLayoutFinder(
            gates=gates, n=n, structure=structure,
            max_arity=max_arity, community_frac=community_frac,
            star_frac=star_frac,
        ).run()

    @classmethod
    def convergence_sweep(cls, gates, n=None, chi_values=(2, 4, 8, 16, 32), *,
                          ops=None, structure="quality", max_arity=2,
                          community_frac=0.35, star_frac=0.75, tree=None,
                          dense_cap=1 << 14):
        """Replay ``gates`` at several ``chi`` and report convergence.

        The tree structure is built once and reused for every ``chi`` so the
        comparison isolates the truncation effect from the layout choice.  For
        each ``chi`` the achieved ``max_bond``, the state ``norm``, and the
        expectation of every ``(operator, where)`` in ``ops`` are recorded.
        When the Hilbert space is small (``2**n <= dense_cap``) the fidelity
        against the untruncated statevector is also reported; the observable
        drift between consecutive ``chi`` values is always reported as a
        reference-free convergence signal.

        Parameters
        ----------
        ops : sequence of ``(operator, where)``, optional
            Observables tracked across ``chi``, labelled ``op{i}`` in the output.
        chi_values : sequence of int
            Bond-dimension caps to sweep (sorted ascending internally).
        tree : TreePlan, optional
            Fixed structure to reuse; inferred once from ``gates`` when omitted.
        dense_cap : int
            Skip the exact-fidelity reference when ``2**n`` exceeds this.

        Returns
        -------
        list of dict
            One record per ``chi`` (ascending) with ``chi``, ``max_bond``,
            ``norm``, ``expectations``, ``fidelity`` (or ``None``), and
            ``max_drift`` (max ``|Delta<op>|`` from the previous ``chi``, or
            ``None`` for the first).
        """
        chi_values = sorted(int(c) for c in chi_values)
        if tree is None:
            probe = cls(gates, n=n, structure=structure, max_arity=max_arity,
                        community_frac=community_frac, star_frac=star_frac,
                        run=False)
            tree = probe.plan
            n = probe.n
        elif n is None:
            n = tree.n

        exact_state = None
        if (1 << n) <= dense_cap:
            ref = cls(gates, n=n, tree=tree, chi=(1 << n), run=True)
            exact_state = ref.to_dense()
            nrm = np.linalg.norm(exact_state)
            if nrm > 0:
                exact_state = exact_state / nrm

        ops = list(ops) if ops else []
        records = []
        prev_vals = None
        for chi in chi_values:
            opt = cls(gates, n=n, tree=tree, chi=chi, run=True)
            expectations = {}
            vals = []
            for i, (op, where) in enumerate(ops):
                val = complex(opt.local_expectation(op, where))
                expectations[f"op{i}"] = val
                vals.append(val)
            fidelity = None
            if exact_state is not None:
                psi = opt.to_dense()
                nrm = np.linalg.norm(psi)
                if nrm > 0:
                    psi = psi / nrm
                    fidelity = float(abs(np.vdot(exact_state, psi)) ** 2)
            max_drift = None
            if prev_vals is not None and vals:
                max_drift = float(
                    max(abs(a - b) for a, b in zip(vals, prev_vals))
                )
            if vals:
                prev_vals = vals
            records.append({
                "chi": chi,
                "max_bond": opt.max_bond(),
                "norm": opt.norm(),
                "expectations": expectations,
                "fidelity": fidelity,
                "max_drift": max_drift,
            })
        return records

    def __repr__(self):
        return (
            f"TreeOptimizer(n={self.n}, chi={self.chi}, "
            f"max_bond={self.max_bond()}, gates={len(self.G)})"
        )
