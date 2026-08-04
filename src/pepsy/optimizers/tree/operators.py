"""Tree-plan-aware native fermionic operator construction.

``SymHamiltonian.to_mpo`` is still used to build the ordinary, low-bond chain
MPO.  A chain's Jordan--Wigner wire, however, is not an embedding of a
branched fermionic tree.  This module therefore also constructs a native tree
operator, without applying it to the state.  Exact tree readout contracts
that operator between a private bra and ket copy.

Two routes are provided:

* general neutral native term sums are decomposed term-by-term on the
  TreePlan Steiner subtrees and amalgamated into one direct-sum TTNO;
* a rank-one pair correlator with separable coefficients is compiled into a
  four-state endpoint automaton.  This is the compact route for the full
  staggered eta-pair observable and keeps its tree bond independent of the
  lattice size.

``TreeMPO`` is a Quimb ``TensorNetworkGenOperator`` over the TreePlan geometry,
analogous to ``TreeTensorNetwork`` being a ``TensorNetworkGenVector``.  The
``tree_mpo`` compatibility builder additionally returns a regular Quimb MPO
for MPS/MPO APIs and attaches the ``TreeMPO`` to it.  The two representations
remain separate throughout the tree contraction.
"""

from __future__ import annotations

import heapq
from numbers import Integral
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .layout import TreePlan

__all__ = ["TreeMPO", "build_tree_operator", "tree_mpo"]


def _as_numpy(data, *, dtype=None):
    """Convert a dense backend array to host NumPy construction data."""
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "get"):
        data = data.get()
    return np.asarray(data, dtype=dtype)


def _tree_plan_signature(plan):
    """Return a stable structural signature for a tree-MPO annotation."""
    return (
        int(plan.root),
        tuple(
            (int(node), tuple(int(child) for child in children))
            for node, children in sorted(plan.children.items())
        ),
        tuple(
            (int(node), int(qubit))
            for node, qubit in sorted(plan.qubit_of_leaf.items())
        ),
        None if plan.root_qubit is None else int(plan.root_qubit),
    )


def _tree_node_selector(plan, selector):
    """Resolve one public tree-node selector to a structural node id."""
    if isinstance(selector, str):
        if selector.startswith("N"):
            selector = selector[1:]
        else:
            raise ValueError(
                "TreeMPO geometry selectors must be node ids or N<node> tags."
            )
    try:
        node = int(selector)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"invalid TreeMPO node selector {selector!r}.") from exc
    if node not in plan.children:
        raise ValueError(f"{node!r} is not a TreePlan node.")
    return node


def _tree_region_selector(plan, selector):
    """Resolve one or more node selectors to a connected TreePlan region."""
    if isinstance(selector, (tuple, list, set, frozenset)):
        nodes = tuple(_tree_node_selector(plan, node) for node in selector)
    else:
        nodes = (_tree_node_selector(plan, selector),)
    if not nodes:
        raise ValueError("TreeMPO canonical regions cannot be empty.")
    region = set(nodes)
    for node in nodes[1:]:
        region.update(plan.node_path(nodes[0], node))
    return frozenset(region)


def _tree_subtree_span(plan, nodes):
    """Return the minimal connected node set spanning ``nodes``."""
    nodes = tuple(nodes)
    if not nodes:
        raise ValueError("need at least one tree node to span a subtree.")
    region = {nodes[0]}
    for node in nodes[1:]:
        region.update(plan.node_path(nodes[0], node))
    return frozenset(region)


class TreeMPO(qtn.TensorNetworkGenOperator):
    """TreePlan-aware operator with dense and native Symmray backends.

    ``TreeMPO`` is the operator-level API for measurements on a
    :class:`TreeTensorNetwork`. It subclasses Quimb's generalized operator
    network, so common methods such as ``sites``, ``site_tag``, ``upper_ind``,
    ``lower_ind``, ``to_dense``, ``H``, and ``copy`` operate on its primary
    TreePlan network. It deliberately keeps the optional linear chain MPO
    separate from the tree representation:

    ``chain_mpo``
        The ordinary Quimb ``MatrixProductOperator`` produced by
        ``SymHamiltonian.to_mpo``.  This is useful for MPS workflows.

    ``tree_networks``
        One or more operator tensor networks whose physical indices are
        labelled by the logical qubits in ``plan``. General native terms are
        combined into one Symmray TTNO whose graded source/target channels
        preserve the fermionic contraction rules.

    General neutral native sums use one direct-sum TTNO. Each term is first
    factorized on its native graded TreePlan subtree, then all term channels
    are amalgamated on common charge-aware virtual bonds. The resulting
    operator can be canonicalized and compressed with native graded QR/SVD.
    Structured observables such as the eta-pair table may use a smaller
    compact network instead.
    """

    # Match Quimb's generalized operator API while retaining the additional
    # TreePlan/chain representation metadata owned by this class.
    _EXTRA_PROPS = qtn.TensorNetworkGenOperator._EXTRA_PROPS + (
        "_plan",
        "_node_tag_id",
        "_pepsy_backend",
        "tree_networks",
        "_canonical_region",
        "chain_mpo",
        "terms",
        "fermionic",
        "symmetry",
        "cutoff",
        "compressed",
    )

    def __init__(
        self,
        plan=None,
        tree_networks=None,
        *,
        chain_mpo=None,
        terms=None,
        backend="dense",
        fermionic=False,
        symmetry=None,
        cutoff=1e-12,
        compressed=False,
        sites=None,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        virtual=True,
        deep=False,
    ):
        if isinstance(plan, TreeMPO) and tree_networks is None:
            source = plan
            plan = source.plan
            networks = tuple(
                network.copy(virtual=virtual, deep=deep)
                for network in source.tree_networks
            )
            chain_mpo = (
                None
                if source.chain_mpo is None
                else source.chain_mpo.copy(virtual=virtual, deep=deep)
            )
            terms = source.terms
            backend = source.backend
            fermionic = source.fermionic
            symmetry = source.symmetry
            cutoff = source.cutoff
            compressed = source.compressed
            sites = source.sites
            site_tag_id = source.site_tag_id
            upper_ind_id = source.upper_ind_id
            lower_ind_id = source.lower_ind_id
            node_tag_id = source.node_tag_id
        elif tree_networks is None:
            raise TypeError("TreeMPO requires a tree operator network.")
        else:
            networks = (
                tuple(tree_networks)
                if isinstance(tree_networks, (tuple, list))
                else (tree_networks,)
            )

        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan.")
        if not networks or any(network is None for network in networks):
            raise ValueError("TreeMPO requires at least one tree operator network.")
        # The first network is the primary generalized tree operator. Use a
        # virtual Quimb view so inherited operator methods such as
        # ``sites``, ``upper_ind``, ``lower_ind``, ``to_dense``, ``bond``, and
        # ``H`` operate on the same tensors as ``tree_networks[0]``.
        super().__init__(networks[0], virtual=True)
        self._plan = plan
        self.tree_networks = networks
        self.chain_mpo = chain_mpo
        self.terms = None if terms is None else dict(terms)
        self._pepsy_backend = str(backend)
        self.fermionic = bool(fermionic)
        self.symmetry = symmetry
        self.cutoff = float(cutoff)
        self.compressed = bool(compressed)
        self._sites = (
            tuple(sorted(plan.node_of_qubit)) if sites is None else tuple(sites)
        )
        self._site_tag_id = site_tag_id
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._node_tag_id = node_tag_id
        self._canonical_region = None
        self.pepsy_tree_plan_signature = _tree_plan_signature(plan)
        if chain_mpo is not None:
            chain_mpo.pepsy_tree_plan_signature = self.pepsy_tree_plan_signature
            chain_mpo.pepsy_tree_terms = (
                None if self.terms is None else dict(self.terms)
            )
            chain_mpo.pepsy_tree_operator = self
            chain_mpo.pepsy_tree_operator_networks = self.tree_networks

    @property
    def backend(self):
        """Return the logical Pepsy backend label for this operator."""
        return self._pepsy_backend

    @property
    def pepsy_backend(self):
        """Compatibility view used by Quimb's structured-network copier."""
        return self._pepsy_backend

    @property
    def plan(self):
        """The :class:`TreePlan` describing the operator geometry."""
        return self._plan

    @property
    def node_tag_id(self):
        """Format string for structural tree-node tags."""
        return self._node_tag_id

    @property
    def site_ind_id(self):
        """Alias for the operator's upper physical-index format."""
        return self.upper_ind_id

    @site_ind_id.setter
    def site_ind_id(self, value):
        self.upper_ind_id = value

    def site_ind(self, site):
        """Return the ket-like physical index for ``site``."""
        return self.upper_ind(site)

    @property
    def root(self):
        """The structural root node id."""
        return self.plan.root

    @property
    def canonical_region(self):
        """The currently canonicalized connected operator region."""
        return self._canonical_region

    @property
    def orthogonality_center(self):
        """The single canonical node, or ``None`` for a larger region."""
        region = self.canonical_region
        return next(iter(region)) if region is not None and len(region) == 1 else None

    @property
    def fermionic(self):
        """Whether the operator stores native fermionic arrays."""
        return self._fermionic

    @fermionic.setter
    def fermionic(self, value):
        self._fermionic = bool(value)

    @property
    def symmetry(self):
        """Native Symmray symmetry label, if present."""
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value):
        self._symmetry = value

    @classmethod
    def from_hamiltonian(
        cls,
        plan,
        hamiltonian,
        *,
        chain_mpo=None,
        cutoff=1e-12,
        max_bond=None,
        compress=True,
        dtype=None,
        fermionic=True,
    ):
        """Construct a ``TreeMPO`` from a ``SymHamiltonian``."""
        from ...tensors.symmetric import SymHamiltonian

        if not isinstance(hamiltonian, SymHamiltonian):
            raise TypeError("hamiltonian must be a SymHamiltonian instance.")
        networks = _build_tree_operator(
            plan,
            hamiltonian,
            cutoff=cutoff,
            max_bond=max_bond,
            compress=compress,
            dtype=dtype,
            fermionic=fermionic,
        )
        if isinstance(networks, (tuple, list)):
            native_networks = tuple(networks)
        else:
            native_networks = (networks,)
        backend = "symmray" if fermionic else "dense"
        operator = cls(
            plan,
            native_networks,
            chain_mpo=chain_mpo,
            terms=hamiltonian.terms,
            backend=backend,
            fermionic=fermionic,
            symmetry=hamiltonian.symmetry,
            cutoff=cutoff,
            compressed=compress,
        )
        if compress:
            operator.compress(max_bond=max_bond, cutoff=cutoff)
        return operator

    @classmethod
    def from_terms(
        cls,
        plan,
        terms,
        *,
        chain_mpo=None,
        cutoff=1e-12,
        dtype=None,
        max_bond=None,
        compress=True,
    ):
        """Construct one ordinary dense TTNO from a term mapping.

        ``terms`` maps an integer site or support tuple to a dense operator
        array. The dense route is useful for non-fermionic trees and for
        callers that already have Jordan--Wigner-compatible local matrices.
        """
        if not hasattr(terms, "items"):
            raise TypeError("terms must be a mapping of supports to operators.")
        network = _combined_tree_operator(
            plan,
            terms,
            symmetry=None,
            cutoff=cutoff,
            dtype=dtype,
            fermionic=False,
        )
        operator = cls(
            plan,
            network,
            chain_mpo=chain_mpo,
            terms=terms,
            backend="dense",
            fermionic=False,
            cutoff=cutoff,
            compressed=compress,
        )
        if compress:
            operator.compress(max_bond=max_bond, cutoff=cutoff)
        return operator

    @classmethod
    def from_dense(
        cls,
        plan,
        array=None,
        dims=2,
        *,
        tree=None,
        sites=None,
        tags=None,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        **split_opts,
    ):
        """Build an exact tree operator from a dense matrix.

        This is the tree analogue of ``MatrixProductOperator.from_dense``.
        The matrix is decomposed over the supplied ``TreePlan`` with lossless
        leaf-to-root SVDs. Only the physical site ordering differs from the
        chain constructor: ``sites`` labels the plan's logical qubits.
        """
        if not isinstance(plan, TreePlan):
            if tree is None:
                raise TypeError("pass a TreePlan with `tree=` or as the first argument.")
            if array is not None:
                raise TypeError("dense array was supplied more than once.")
            array = plan
            plan = tree
        elif tree is not None and tree is not plan:
            raise ValueError("plan and tree specify different TreePlans.")
        if array is None:
            raise TypeError("TreeMPO.from_dense requires a dense matrix.")
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan.")
        if sites is None:
            sites = tuple(sorted(plan.node_of_qubit))
        else:
            sites = tuple(int(site) for site in sites)
        if sites != tuple(sorted(sites)):
            raise ValueError("TreeMPO.from_dense requires sorted site labels.")
        if set(sites) != set(plan.node_of_qubit):
            raise ValueError(
                "TreeMPO.from_dense currently requires one matrix site per tree site."
            )
        if isinstance(dims, Integral):
            dims = (int(dims),) * len(sites)
        else:
            dims = tuple(int(dim) for dim in dims)
        if len(dims) != len(sites):
            raise ValueError("dims must have one entry per TreePlan site.")
        if np.prod(dims, dtype=int) ** 2 != np.size(array):
            raise ValueError("array size does not match the supplied physical dims.")
        network = _tree_operator_from_dense(
            plan,
            array,
            sites=sites,
            dims=dims,
            split_opts=split_opts,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )
        if tags is not None:
            network.add_tag(tags)
        return cls(
            plan,
            network,
            backend="dense",
            fermionic=False,
            sites=sites,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        plan,
        bond_dim,
        *,
        phys_dim=2,
        dtype=float,
        sites=None,
        tags=None,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
    ):
        """Build a tree operator from a tensor filling function.

        ``fill_fn`` is called as ``fill_fn(shape)`` for each plan node, where
        ``shape`` is ordered as physical upper/lower legs followed by the
        node's tree bonds. A scalar ``bond_dim`` or one value per edge is
        accepted through the same uniform tree convention.
        """
        if sites is None:
            sites = tuple(sorted(plan.node_of_qubit))
        else:
            sites = tuple(sites)
        network = _tree_operator_from_fill_fn(
            plan,
            fill_fn,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            dtype=dtype,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )
        if tags is not None:
            network.add_tag(tags)
        return cls(
            plan,
            network,
            backend="dense",
            fermionic=False,
            sites=sites,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )

    @classmethod
    def rand(
        cls,
        plan,
        bond_dim,
        *,
        phys_dim=2,
        dtype=complex,
        seed=None,
        **operator_opts,
    ):
        """Build a random dense TreeMPO with uniform virtual bond size."""
        rng = np.random.default_rng(seed)

        def fill(shape):
            if np.issubdtype(np.dtype(dtype), np.complexfloating):
                return (
                    rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)
                ).astype(dtype)
            return rng.standard_normal(shape).astype(dtype)

        return cls.from_fill_fn(
            fill,
            plan,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            dtype=dtype,
            **operator_opts,
        )

    @property
    def tree_network(self):
        """Return the sole tree network, or raise for a term sum."""
        if len(self.tree_networks) != 1:
            raise AttributeError(
                "this TreeMPO contains multiple internal networks; use "
                "tree_networks or expectation()"
            )
        return self.tree_networks[0]

    @property
    def nqubits(self):
        """Number of logical physical sites in the TreePlan."""
        return self.plan.n

    @property
    def top_arity(self):
        """Number of virtual child bonds at the structural root."""
        return self.plan.top_arity

    @property
    def max_virtual_degree(self):
        """Largest number of virtual tree bonds on one operator tensor."""
        return self.plan.max_virtual_degree()

    @property
    def max_tensor_rank(self):
        """Largest virtual/physical leg count on one operator tensor."""
        return self.plan.max_tensor_rank()

    def node_tag(self, node):
        """Return the structural tag for a TreePlan node."""
        return self._node_tag_id.format(int(node))

    def node_tensor(self, node):
        """Return a primary TTNO tensor by TreePlan node id."""
        return self.tree_networks[0][self.node_tag(node)]

    def _select_tids(self, tids, virtual=True, with_exponent=False):
        """Select a structured view while keeping its primary network live."""
        selected = super()._select_tids(
            tids,
            virtual=virtual,
            with_exponent=with_exponent,
        )
        # Quimb's generic ``new(like=...)`` copies the extra properties from
        # the source, including ``tree_networks``. Replace that source tuple
        # with the selected view so inherited selection methods never mutate
        # or inspect the original operator by accident.
        selected.tree_networks = (qtn.TensorNetwork(selected, virtual=True),)
        selected.chain_mpo = None
        return selected

    def neighbors(self, node):
        """Return the TreePlan neighbors of a structural node."""
        node = int(node)
        if node not in self.plan.children:
            raise ValueError(f"{node!r} is not a TreePlan node.")
        return tuple(self.plan.children[node]) + (
            (self.plan.parent[node],)
            if self.plan.parent.get(node) is not None
            else ()
        )

    def is_leaf(self, node):
        """Whether ``node`` is a structural leaf."""
        return self.plan.is_leaf(int(node))

    def parent(self, node):
        """Return the parent structural node, or ``None`` at the root."""
        node = int(node)
        if node not in self.plan.children:
            raise ValueError(f"{node!r} is not a TreePlan node.")
        return self.plan.parent.get(node)

    def children(self, node):
        """Return the structural children of ``node``."""
        node = int(node)
        if node not in self.plan.children:
            raise ValueError(f"{node!r} is not a TreePlan node.")
        return self.plan.children[node]

    def node_path(self, node1, node2):
        """Return the inclusive structural path between two nodes."""
        return self.plan.node_path(int(node1), int(node2))

    def leaf_of_qubit(self, qubit):
        """Return the structural leaf carrying ``qubit``."""
        return self.plan.leaf_of_qubit[int(qubit)]

    def qubit_of_leaf(self, node):
        """Return the qubit carried by a structural leaf."""
        return self.plan.qubit_of_leaf[int(node)]

    def qubit_of_node(self, node):
        """Return the qubit carried by a node, or ``None`` if virtual."""
        return self.plan.qubit_of_node.get(int(node))

    def node_of_qubit(self, qubit):
        """Return the structural node carrying ``qubit``."""
        return self.plan.node_of_qubit[int(qubit)]

    def tree_distance(self, qubit1, qubit2):
        """Return the structural distance between two physical sites."""
        return self.plan.tree_distance(int(qubit1), int(qubit2))

    def steiner_nodes(self, nodes):
        """Return the minimal connected subtree spanning ``nodes``."""
        return self.plan.steiner_nodes(tuple(int(node) for node in nodes))

    def subtree_span(self, nodes):
        """Return the minimal connected subtree spanning arbitrary nodes."""
        return _tree_subtree_span(
            self.plan, tuple(int(node) for node in nodes),
        )

    def is_binary(self, *, allow_ternary_root=True):
        """Whether this operator's tree is binary below its root."""
        return self.plan.is_binary(allow_ternary_root=allow_ternary_root)

    def bond(self, node, neighbor):
        """Return the live operator bond between adjacent TreePlan nodes."""
        node = int(node)
        neighbor = int(neighbor)
        if neighbor not in self.neighbors(node):
            raise ValueError(
                f"nodes {node} and {neighbor} are not adjacent in the tree."
            )
        shared = qtn.bonds(
            self.node_tensor(node), self.node_tensor(neighbor),
        )
        if len(shared) != 1:
            raise ValueError(
                f"nodes {node} and {neighbor} must share exactly one bond; "
                f"found {sorted(shared)}."
            )
        return next(iter(shared))

    def validate(self):
        """Validate the primary TTNO against its TreePlan geometry."""
        network = self.tree_networks[0]
        for node in self.plan.nodes():
            tensor = self.node_tensor(node)
            expected = set(self.neighbors(node))
            physical = self.plan.qubit_of_node.get(node)
            expected_inds = {
                f"_pepsy_tnno_{min(node, other)}_{max(node, other)}"
                for other in expected
            }
            if physical is not None:
                expected_inds.update((
                    self.upper_ind(physical),
                    self.lower_ind(physical),
                ))
            if set(tensor.inds) != expected_inds:
                raise ValueError(
                    f"TreeMPO node {node} has unexpected indices: "
                    f"{tensor.inds!r}."
                )
        for node in self.plan.nodes():
            for neighbor in self.plan.children[node]:
                self.bond(node, neighbor)
        if set(network.outer_inds()) != {
            self.upper_ind(site) for site in self.sites
        } | {
            self.lower_ind(site) for site in self.sites
        }:
            raise ValueError("TreeMPO has unexpected outer physical indices.")
        return self

    def max_bond(self):
        """Return the largest virtual bond among the tree networks."""
        bonds = []
        for network in self.tree_networks:
            for index in network.inner_inds():
                bonds.append(network.ind_size(index))
        return max(bonds, default=1)

    def bond_size(self, node, neighbor):
        """Return the dimension of one live operator tree bond."""
        return self.node_tensor(node).ind_size(self.bond(node, neighbor))

    def bond_sizes(self):
        """Return operator bond dimensions in deterministic tree-edge order."""
        return tuple(
            self.bond_size(node, child)
            for node in self.plan.nodes()
            for child in self.plan.children[node]
        )

    def edge_nodes(self):
        """Return all directed parent-child tree edges."""
        return tuple(
            (node, child)
            for node in self.plan.nodes()
            for child in self.plan.children[node]
        )

    @property
    def L(self):
        """Number of logical physical sites, as in a chain MPO."""
        return self.nsites

    @property
    def cyclic(self):
        """TreeMPOs are open tree networks, never cyclic chains."""
        return False

    def to_dense(self, *inds_seq, to_qarray=False, **contract_opts):
        """Contract the complete operator, summing internal term networks."""
        if len(self.tree_networks) == 1 and not self.fermionic:
            return qtn.TensorNetworkGenOperator.to_dense(
                self,
                *inds_seq,
                to_qarray=to_qarray,
                **contract_opts,
            )
        if not inds_seq:
            inds_seq = (self.upper_inds_present, self.lower_inds_present)
        values = []
        for network in self.tree_networks:
            if self.fermionic:
                # Symmray's block-sparse contraction assumes a neutral scalar
                # when it closes all internal legs. A charged operator has
                # nonzero open physical charge, so densify each local block
                # first and contract the ordinary tree network. This is only
                # the explicit ``to_dense`` escape hatch; native expectation
                # and compression remain graded and factorized.
                dense_tensors = []
                for tensor in network:
                    data = tensor.data
                    if hasattr(data, "to_dense"):
                        data = data.to_dense()
                    dense_tensors.append(qtn.Tensor(
                        data,
                        inds=tensor.inds,
                        tags=tensor.tags,
                    ))
                network = qtn.TensorNetwork(dense_tensors)
            view = qtn.TensorNetworkGenOperator(
                network,
                virtual=True,
            )
            view._sites = self.sites
            view._site_tag_id = self.site_tag_id
            view._upper_ind_id = self.upper_ind_id
            view._lower_ind_id = self.lower_ind_id
            values.append(view.to_dense(*inds_seq, **contract_opts))
        result = values[0]
        for value in values[1:]:
            result = result + value
        if to_qarray:
            import quimb as qu

            return qu.qarray(result)
        return result

    def identity(self, *, phys_dim=None, dtype=None):
        """Return the exact bond-one identity TreeMPO on this plan."""
        if phys_dim is None:
            phys_dim = tuple(self.phys_dim(site) for site in self.sites)
        if dtype is None:
            dtype = self.dtype
        network = _identity_tree_operator(
            self.plan,
            phys_dim=phys_dim,
            dtype=dtype,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
        )
        return type(self)(
            self.plan,
            network,
            backend="dense",
            fermionic=False,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
        )

    def add_MPO(
        self,
        other,
        inplace=False,
        negate=False,
        compress=False,
        **compress_opts,
    ):
        """Add another matching tree operator by arbitrary-geometry direct sum."""
        if not isinstance(other, TreeMPO):
            other = getattr(other, "pepsy_tree_operator", None)
        if not isinstance(other, TreeMPO):
            raise TypeError("other must be a TreeMPO or an annotated chain MPO.")
        if self.pepsy_tree_plan_signature != other.pepsy_tree_plan_signature:
            raise ValueError("TreeMPOs must use the same TreePlan.")
        if self.fermionic != other.fermionic:
            raise TypeError("cannot add dense and native TreeMPOs.")
        if len(self.tree_networks) != len(other.tree_networks):
            raise ValueError("TreeMPO term-network counts must match.")

        networks = []
        for left, right in zip(self.tree_networks, other.tree_networks):
            networks.append(qtn.tensor_network_ag_sum(
                left,
                right,
                site_tags=tuple(self.node_tag(node) for node in self.plan.nodes()),
                negate=negate,
                compress=compress,
                **compress_opts,
            ))
        chain = None
        if self.chain_mpo is not None and other.chain_mpo is not None:
            chain = self.chain_mpo.add_MPO(
                other.chain_mpo,
                inplace=False,
                negate=negate,
                compress=compress,
                **compress_opts,
            )
        terms = None
        if self.terms is not None and other.terms is not None:
            terms = dict(self.terms)
            for support, value in other.terms.items():
                if support in terms:
                    terms[support] = terms[support] + ((-1) if negate else 1) * value
                else:
                    terms[support] = ((-1) if negate else 1) * value
        result = type(self)(
            self.plan,
            tuple(networks),
            chain_mpo=chain,
            terms=terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=compress,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
        )
        if inplace:
            self.__dict__.clear()
            self.__dict__.update(result.__dict__)
            return self
        return result

    add_MPO_ = lambda self, other, **kwargs: self.add_MPO(  # noqa: E731
        other, inplace=True, **kwargs,
    )

    def matrix_element(self, bra, ket=None):
        """Return ``<bra|TreeMPO|ket>`` for computational-basis strings."""
        if ket is None:
            ket = bra
        bra = tuple(int(value) for value in bra)
        ket = tuple(int(value) for value in ket)
        if len(bra) != self.nsites or len(ket) != self.nsites:
            raise ValueError("basis configurations must match TreeMPO.nsites.")
        selector = {}
        for site, bra_value, ket_value in zip(self.sites, bra, ket):
            selector[self.upper_ind(site)] = bra_value
            selector[self.lower_ind(site)] = ket_value
        value = 0.0
        for network in self.tree_networks:
            value = value + network.isel(selector).contract(all)
        return value

    def amplitude(self, configuration):
        """Return the diagonal computational-basis matrix element."""
        return self.matrix_element(configuration)

    def singular_values(self, node, neighbor=None, *, method="svd"):
        """Return singular values across one tree operator edge."""
        if neighbor is None:
            try:
                node, neighbor = node
            except (TypeError, ValueError) as exc:
                raise TypeError("singular_values needs an operator edge.") from exc
        node = _tree_node_selector(self.plan, node)
        neighbor = _tree_node_selector(self.plan, neighbor)
        if neighbor not in self.neighbors(node):
            raise ValueError("singular_values requires adjacent tree nodes.")
        work = self.copy()
        work.canonicalize(center=neighbor)
        tensor = work.node_tensor(node)
        bond = work.bond(node, neighbor)
        return tensor.singular_values(
            tuple(ind for ind in tensor.inds if ind != bond),
            method=method,
        )

    def rand_state(self, bond_dim, **state_opts):
        """Return a random :class:`TreeTensorNetwork` on the same plan."""
        from .ttn import TreeTensorNetwork

        return TreeTensorNetwork.rand(self.plan, D=bond_dim, **state_opts)

    def show(self, *, bond_dims=True, node_ids=False, color=False):
        """Print a compact top-down tree drawing for this operator."""
        del color

        def render(node, prefix="", is_last=True):
            qubit = self.plan.qubit_of_node.get(node)
            label = f"N{node}" if node_ids else "●"
            if qubit is not None:
                label += f" q{qubit}"
            lines = [prefix + ("└─ " if is_last else "├─ ") + label]
            children = tuple(self.plan.children[node])
            for index, child in enumerate(children):
                edge = self.bond_size(node, child) if bond_dims else None
                edge_label = f" [{edge}]" if edge is not None else ""
                child_lines = render(
                    child,
                    prefix + ("   " if is_last else "│  "),
                    index == len(children) - 1,
                )
                child_lines[0] = child_lines[0] + edge_label
                lines.extend(child_lines)
            return lines

        lines = render(self.plan.root, "", True)
        print("\n".join(lines))

    def canonicalize(self, center=None, *, inplace=True):
        """Canonicalize every stored TTNO around one TreePlan node.

        This is the tree equivalent of an MPO mixed-canonical gauge. The
        default is inplace, matching Quimb's MPO canonicalization methods;
        pass ``inplace=False`` to obtain an independent operator.
        """
        if center is None:
            center = self.plan.root
        center = _tree_node_selector(self.plan, center)
        target = self if inplace else self.copy()
        for network in target.tree_networks:
            _canonicalize_tree_operator(network, target.plan, center)
        target._canonical_region = frozenset({center})
        return target

    def canonicalize_(self, center=None):
        """Inplace alias for :meth:`canonicalize`."""
        return self.canonicalize(center=center, inplace=True)

    canonize = canonicalize_

    def invalidate_canonical_form(self):
        """Forget operator gauge metadata after an unmanaged tensor edit."""
        self._canonical_region = None
        return self

    def isometry_direction(self, node):
        """Return the neighbour receiving a node's canonical QR factor."""
        node = _tree_node_selector(self.plan, node)
        tensor = self.node_tensor(node)
        if tensor.left_inds is None:
            return None
        right_inds = [ind for ind in tensor.inds if ind not in tensor.left_inds]
        if len(right_inds) != 1:
            return None
        for neighbor in self.neighbors(node):
            if right_inds[0] == self.bond(node, neighbor):
                return neighbor
        return None

    def isometry_map(self):
        """Return the live QR orientation map for all TreePlan nodes."""
        return {
            node: self.isometry_direction(node)
            for node in self.plan.nodes()
        }

    def is_subtree_canonical_form(self, nodes=None, *, span=False):
        """Check the lossless QR metadata around a connected operator region."""
        if nodes is None:
            region = self.canonical_region
            if region is None:
                return False
        else:
            region = _tree_region_selector(self.plan, nodes) if span else frozenset(
                _tree_node_selector(self.plan, node) for node in nodes
            )
            if _tree_subtree_span(self.plan, region) != region:
                return False
        for node in self.plan.nodes():
            if node in region:
                continue
            path = min(
                (
                    self.plan.node_path(node, target)
                    for target in region
                ),
                key=len,
            )
            if self.isometry_direction(node) != path[1]:
                return False
        return True

    def is_canonical_form(self, center=None):
        """Check whether the operator has a one-node canonical region."""
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return False
        return self.is_subtree_canonical_form((center,))

    def shift_orthogonality_center(self, current, new):
        """Move the operator QR centre to another TreePlan node."""
        del current
        return self.canonicalize(center=new, inplace=True)

    def calc_current_orthog_center(self):
        """Return the current operator canonical region bounds."""
        region = self.canonical_region
        if not region:
            return None
        ordered = sorted(region)
        return ordered[0], ordered[-1]

    def left_canonicalize(self, *, center=None, inplace=False, **kwargs):
        """MPO-compatible alias for a root-oriented tree QR sweep."""
        del kwargs
        return self.canonicalize(
            center=self.plan.root if center is None else center,
            inplace=inplace,
        )

    left_canonicalize_ = lambda self, **kwargs: self.left_canonicalize(  # noqa: E731
        inplace=True, **kwargs,
    )
    left_canonize = left_canonicalize_

    def right_canonicalize(self, *, center=None, inplace=False, **kwargs):
        """MPO-compatible alias for a root-oriented tree QR sweep."""
        del kwargs
        return self.canonicalize(
            center=self.plan.root if center is None else center,
            inplace=inplace,
        )

    right_canonicalize_ = lambda self, **kwargs: self.right_canonicalize(  # noqa: E731
        inplace=True, **kwargs,
    )
    right_canonize = right_canonicalize_

    def compress_site(self, node, *, max_bond=None, cutoff=None, **kwargs):
        """Compress all tree bonds consistently around ``node``."""
        del kwargs
        self.canonicalize(center=node)
        return self.compress(max_bond=max_bond, cutoff=cutoff)

    def left_compress(self, *, max_bond=None, cutoff=None, **kwargs):
        """Tree analogue of a left-to-right MPO compression sweep."""
        del kwargs
        return self.compress(max_bond=max_bond, cutoff=cutoff)

    def right_compress(self, *, max_bond=None, cutoff=None, **kwargs):
        """Tree analogue of a right-to-left MPO compression sweep."""
        del kwargs
        return self.compress(max_bond=max_bond, cutoff=cutoff)

    def canonize_around(
        self, tags, which="all", *, inplace=False, **canonize_opts,
    ):
        """Quimb-style alias for TreePlan-centered TTNO canonicalization.

        Tree operators have a rooted geometry rather than a one-dimensional
        tag interval, so the supported target is one TreePlan node. The
        additional Quimb options are accepted for API familiarity and are
        intentionally ignored after validating that the target is a node.
        """
        del which, canonize_opts
        if isinstance(tags, (tuple, list, set, frozenset)):
            if len(tags) == 0:
                raise ValueError("TreeMPO.canonize_around needs one node.")
            if len(tags) != 1:
                target = self if inplace else self.copy()
                region = _tree_region_selector(target.plan, tags)
                for network in target.tree_networks:
                    _canonicalize_tree_operator_region(
                        network, target.plan, region,
                    )
                target._canonical_region = region
                return target
        target = self if inplace else self.copy()
        return target.canonicalize(center=tags, inplace=True)

    def canonize_around_(self, tags, **kwargs):
        """In-place Quimb-style alias for :meth:`canonize_around`."""
        kwargs["inplace"] = True
        return self.canonize_around(tags, **kwargs)

    def canonize_between(
        self, tags1, tags2, *, inplace=False, absorb="right", **canonize_opts,
    ):
        """Canonicalize the operator exterior to a TreePlan path.

        A path is the tree analogue of the mixed-canonical interval used by
        an MPS. ``absorb`` and other Quimb gauge options are accepted for API
        compatibility; the lossless native QR policy controls the operation.
        """
        del absorb, canonize_opts
        node1 = _tree_node_selector(self.plan, tags1)
        node2 = _tree_node_selector(self.plan, tags2)
        region = frozenset(self.plan.node_path(node1, node2))
        target = self if inplace else self.copy()
        for network in target.tree_networks:
            _canonicalize_tree_operator_region(network, target.plan, region)
        target._canonical_region = region
        return target

    def canonize_between_(self, tags1, tags2, **kwargs):
        """In-place alias for :meth:`canonize_between`."""
        kwargs["inplace"] = True
        return self.canonize_between(tags1, tags2, **kwargs)

    def compress_between(
        self, tags1, tags2, max_bond=None, cutoff=1e-10, **compress_opts,
    ):
        """Quimb-style compression entry point for a tree operator.

        A TreeMPO compression is a global leaf-to-root sweep so every edge
        sees the complete operator sum. ``tags1`` and ``tags2`` identify an
        adjacent TreePlan edge for validation; the configured sweep then
        compresses all TreePlan bonds consistently.
        """
        inplace = compress_opts.pop("inplace", True)
        del compress_opts
        node1 = _tree_node_selector(self.plan, tags1)
        node2 = _tree_node_selector(self.plan, tags2)
        if node2 not in self.neighbors(node1):
            raise ValueError(
                "TreeMPO.compress_between requires adjacent TreePlan nodes."
            )
        target = self if inplace else self.copy()
        return target.compress(max_bond=max_bond, cutoff=cutoff)

    def compress_between_(self, tags1, tags2, **kwargs):
        """In-place alias for :meth:`compress_between`."""
        kwargs["inplace"] = True
        return self.compress_between(tags1, tags2, **kwargs)

    def compress(self, *, max_bond=None, cutoff=None):
        """Compress the TTNO on every TreePlan edge with native SVD."""
        if cutoff is None:
            cutoff = self.cutoff
        cutoff = float(cutoff)
        reports = []
        for network in self.tree_networks:
            reports.append(_compress_tree_operator(
                network,
                self.plan,
                max_bond=max_bond,
                cutoff=cutoff,
            ))
        self.cutoff = cutoff
        self.compressed = True
        self.pepsy_compression_report = reports[0] if len(reports) == 1 else reports
        self._canonical_region = None
        return self

    def copy(self, virtual=False, deep=False, *, conj=False, transpose=False):
        """Copy the operator and both of its optional representations.

        The signature follows Quimb's tensor-network ``copy`` API. ``virtual``
        keeps tensor data shared while copying the network structure; ``deep``
        requests independent numeric data as in the underlying Quimb views.
        ``conj`` and ``transpose`` are accepted as convenient MPO-compatible
        view operations.
        """
        chain_mpo = (
            None
            if self.chain_mpo is None
            else self.chain_mpo.copy(virtual=virtual, deep=deep)
        )
        copied = type(self)(
            self.plan,
            tuple(
                network.copy(virtual=virtual, deep=deep)
                for network in self.tree_networks
            ),
            chain_mpo=chain_mpo,
            terms=self.terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=self.compressed,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
        )
        if chain_mpo is not None:
            chain_mpo.pepsy_tree_plan_signature = copied.pepsy_tree_plan_signature
            chain_mpo.pepsy_tree_terms = (
                None if copied.terms is None else dict(copied.terms)
            )
            chain_mpo.pepsy_tree_operator = copied
            chain_mpo.pepsy_tree_operator_networks = copied.tree_networks
        if hasattr(self, "pepsy_compression_report"):
            copied.pepsy_compression_report = self.pepsy_compression_report
        copied._canonical_region = self.canonical_region
        if transpose:
            copied._transpose_operator_inplace()
        if conj:
            copied.conj(inplace=True)
        return copied

    def _transpose_operator_inplace(self):
        """Transpose every local upper/lower physical pair in place."""
        for network in self.tree_networks:
            for tensor in network:
                physical_axes = []
                for site in self.sites:
                    upper = self.upper_ind(site)
                    lower = self.lower_ind(site)
                    if upper in tensor.inds and lower in tensor.inds:
                        physical_axes.append((
                            tensor.inds.index(upper),
                            tensor.inds.index(lower),
                        ))
                if not physical_axes:
                    continue
                permutation = list(range(tensor.ndim))
                for upper_axis, lower_axis in physical_axes:
                    permutation[upper_axis], permutation[lower_axis] = (
                        permutation[lower_axis], permutation[upper_axis]
                    )
                tensor.modify(data=ar.do("transpose", tensor.data, permutation))
        if self.chain_mpo is not None:
            for tensor in self.chain_mpo:
                axes = []
                for site in self.sites:
                    upper = self.upper_ind(site)
                    lower = self.lower_ind(site)
                    if upper in tensor.inds and lower in tensor.inds:
                        axes.append((
                            tensor.inds.index(upper),
                            tensor.inds.index(lower),
                        ))
                permutation = list(range(tensor.ndim))
                for upper_axis, lower_axis in axes:
                    permutation[upper_axis], permutation[lower_axis] = (
                        permutation[lower_axis], permutation[upper_axis]
                    )
                if axes:
                    tensor.modify(data=ar.do("transpose", tensor.data, permutation))
        return self

    def conj(
        self,
        mangle_inner=False,
        output_inds=None,
        phase_dual=True,
        inplace=False,
    ):
        """Conjugate every stored tree operator like a Quimb operator view."""
        if inplace:
            for network in self.tree_networks:
                network.conj(
                    mangle_inner=mangle_inner,
                    output_inds=output_inds,
                    phase_dual=phase_dual,
                    inplace=True,
                )
            if self.chain_mpo is not None:
                self.chain_mpo.conj(
                    mangle_inner=mangle_inner,
                    output_inds=output_inds,
                    phase_dual=phase_dual,
                    inplace=True,
                )
            return self

        networks = tuple(
            network.conj(
                mangle_inner=mangle_inner,
                output_inds=output_inds,
                phase_dual=phase_dual,
            )
            for network in self.tree_networks
        )
        chain_mpo = (
            None
            if self.chain_mpo is None
            else self.chain_mpo.conj(
                mangle_inner=mangle_inner,
                output_inds=output_inds,
                phase_dual=phase_dual,
            )
        )
        return type(self)(
            self.plan,
            networks,
            chain_mpo=chain_mpo,
            terms=self.terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=self.compressed,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
        )

    def expectation(self, state, *, normalized=True, optimize="auto"):
        """Evaluate ``<state|TreeMPO|state>`` in one public operation."""
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        tree = getattr(state, "tn", state)
        if getattr(tree, "plan", None) is None:
            raise TypeError("state must be a TreeTensorNetwork or TreeOptimizer.")
        if _tree_plan_signature(tree.plan) != self.pepsy_tree_plan_signature:
            raise ValueError("TreeMPO and state use different TreePlans.")
        if self.fermionic and not getattr(tree, "fermionic", False):
            raise TypeError("native TreeMPO requires a native fermionic tree state.")
        if self.fermionic is False and getattr(tree, "fermionic", False):
            raise TypeError("dense TreeMPO cannot be contracted with a native fermionic tree.")

        sites = tuple(sorted(tree.plan.node_of_qubit))
        numerator = 0.0
        for operator in self.tree_networks:
            ket = tree.copy()
            operator_work = operator.copy()
            ket_reindex = {}
            operator_reindex = {}
            for site in sites:
                physical = tree.site_ind(site)
                upper = f"k{site}"
                lower = f"b{site}"
                if upper not in operator_work.ind_map or lower not in operator_work.ind_map:
                    raise ValueError(f"TreeMPO is missing physical site {site!r}.")
                fresh = qtn.rand_uuid()
                ket_reindex[physical] = fresh
                operator_reindex[lower] = fresh
            ket.reindex_(ket_reindex)
            operator_work.reindex_(operator_reindex)
            numerator = numerator + (tree.H | operator_work | ket).contract(
                all,
                optimize=optimize,
            )
        if not normalized:
            return numerator
        denominator = (tree.H | tree).contract(all, optimize=optimize)
        return numerator / denominator

    def __repr__(self):
        return (
            f"TreeMPO(nsite={self.plan.n}, backend={self.backend!r}, "
            f"networks={len(self.tree_networks)}, max_bond={self.max_bond()})"
        )

    def __getattr__(self, name):
        """Preserve the old metadata attributes on compact tree operators."""
        if name.startswith("pepsy_tree_operator_"):
            networks = self.__dict__.get("tree_networks", ())
            if len(networks) == 1:
                return getattr(networks[0], name)
        raise AttributeError(name)


def _term_support(where):
    """Normalize one Hamiltonian key to an integer support tuple."""
    if isinstance(where, Integral):
        support = (int(where),)
    else:
        try:
            support = tuple(int(site) for site in where)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "tree MPO Hamiltonian term locations must be integer sites "
                "or tuples of integer sites."
            ) from exc
    if not support:
        raise ValueError("tree MPO Hamiltonian term supports cannot be empty.")
    if len(set(support)) != len(support):
        raise ValueError(
            f"tree MPO Hamiltonian term support {support!r} repeats a site."
        )
    return support


def _expanded_index_charges(index):
    """Expand a native block index into its dense charge ordering."""
    chargemap = getattr(index, "chargemap", None)
    if chargemap is None:
        raise TypeError("native operator factors must expose block charges.")
    return [charge for charge, size in chargemap.items() for _ in range(size)]


def _operator_charge_neg(charge):
    """Negate one Abelian charge used by a native operator channel."""
    if isinstance(charge, tuple):
        return tuple(-value for value in charge)
    return -charge


def _operator_charge_sub(left, right):
    """Subtract two expanded physical charges componentwise."""
    if isinstance(left, tuple):
        return tuple(a - b for a, b in zip(left, right))
    return left - right


def _operator_charge_from_matrix(data, physical_map, *, tol=1e-10):
    """Infer the homogeneous local operator charge from a dense matrix."""
    values = {
        _operator_charge_sub(physical_map[int(out)], physical_map[int(inp)])
        for out, inp in np.argwhere(np.abs(data) > tol)
    }
    if len(values) != 1:
        raise ValueError(
            "operator-Schmidt factors must have one homogeneous physical "
            f"charge, got {sorted(values, key=repr)!r}."
        )
    return values.pop()


def _operator_native_channels(
    term, support, *, symmetry, dtype=None, cutoff=1e-12,
):
    """Split one native two-site term into charged local operator channels."""
    original_support = tuple(int(site) for site in support)
    support = tuple(sorted(original_support))
    if len(support) != 2:
        raise ValueError("native operator channels require two sites.")
    if tuple(_term_support(support)) != support:
        raise ValueError("operator support must contain distinct sites.")
    ordered_term = term
    if original_support != support:
        ordered_term = term.transpose((1, 0, 3, 2))

    # ``cutoff=0`` retains structural zero sectors as separate channels. The
    # small fixed threshold removes only those exact numerical zeros; the
    # user-facing TreeMPO cutoff is applied later to the combined TTNO.
    structural_cutoff = 64.0 * np.finfo(float).eps
    fused = ordered_term.fuse((0, 2), (1, 3))
    left, _, right = fused.svd(
        absorb="right", cutoff=structural_cutoff,
    )
    if left is None or right is None:
        raise ValueError("could not split a native two-site operator.")
    left = left.unfuse(0).transpose((2, 0, 1))
    right = right.unfuse(1)
    left_data = _as_numpy(left.to_dense(), dtype=dtype)
    right_data = _as_numpy(right.to_dense(), dtype=dtype)
    physical_map = _expanded_index_charges(left.indices[1])
    if _expanded_index_charges(left.indices[2]) != physical_map:
        raise ValueError("native operator factors have mismatched physical maps.")
    channels = []
    for channel in range(left_data.shape[0]):
        source = left_data[channel]
        target = right_data[channel]
        if not np.any(np.abs(source) > 1e-10):
            continue
        if not np.any(np.abs(target) > 1e-10):
            raise ValueError("native operator SVD produced an empty channel.")
        source_charge = _operator_charge_from_matrix(source, physical_map)
        target_charge = _operator_charge_from_matrix(target, physical_map)
        if target_charge != _operator_charge_neg(source_charge):
            raise ValueError(
                "native operator channel charges do not cancel: "
                f"{source_charge!r} and {target_charge!r}."
            )
        channels.append((source, target, source_charge))
    if not channels:
        raise ValueError("native two-site operator has no nonzero channels.")
    return channels, physical_map


def _operator_dense_channels(operator, support, *, dtype=None, cutoff=1e-12):
    """Split one ordinary dense two-site term into local channels."""
    support = tuple(sorted(int(site) for site in support))
    data = _dense_operator_array(operator, dtype=dtype)
    if data.ndim != 4 or data.shape[0] != data.shape[1] or data.shape[0] != data.shape[2]:
        raise ValueError("dense two-site operators must have shape (d, d, d, d).")
    if data.shape[2] != data.shape[3]:
        raise ValueError("dense two-site operators must have matching input legs.")
    dim = data.shape[0]
    matrix = data.transpose(0, 2, 1, 3).reshape(dim * dim, dim * dim)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    structural_cutoff = max(64.0 * np.finfo(float).eps, float(cutoff))
    channels = []
    for channel, value in enumerate(singular):
        if float(value) <= structural_cutoff:
            continue
        scale = np.sqrt(value)
        channels.append((
            (left[:, channel] * scale).reshape(dim, dim),
            (scale * right[channel, :]).reshape(dim, dim),
            0,
        ))
    if not channels:
        raise ValueError("dense two-site operator has no nonzero channels.")
    return channels, [0] * dim


def _operator_valid_child_states(nchildren, nstate, nchannel, done):
    """Generate only the valid sparse automaton child configurations."""
    if not nchildren:
        return [()]
    states = {(0,) * nchildren}
    active_states = tuple(range(1, done))
    for child in range(nchildren):
        for state in active_states:
            values = [0] * nchildren
            values[child] = state
            states.add(tuple(values))
        values = [0] * nchildren
        values[child] = done
        states.add(tuple(values))
    nchannel = int(nchannel)
    source = lambda channel: 1 + channel
    target = lambda channel: 1 + nchannel + channel
    for left in range(nchildren):
        for right in range(left + 1, nchildren):
            for channel in range(nchannel):
                for first, second in (
                    (source(channel), target(channel)),
                    (target(channel), source(channel)),
                ):
                    values = [0] * nchildren
                    values[left] = first
                    values[right] = second
                    states.add(tuple(values))
    return tuple(sorted(states))


def _combined_tree_operator(
    plan, terms, *, symmetry=None, cutoff=1e-12, dtype=None, fermionic=True,
):
    """Build one TreePlan TTNO for a neutral one-/two-site term mapping.

    Each two-site operator-Schmidt channel becomes a source/target charge
    channel. At a branching node, the channel automaton can collect one source
    and one target from different child subtrees before closing into the
    neutral ``done`` sector. This is the tree analogue of the start/channel/
    done construction used by a native chain MPO, but the channels follow the
    selected TreePlan rather than a Jordan--Wigner wire.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    if not hasattr(terms, "items") or not terms:
        raise ValueError("At least one operator term is required.")

    channels = []
    one_site = {}
    physical_map = None
    for where, term in terms.items():
        support = _term_support(where)
        if any(site not in plan.node_of_qubit for site in support):
            raise ValueError(f"operator support {support!r} is outside the TreePlan.")
        if len(support) == 1:
            data = (
                _dense_operator_array(term, dtype=dtype)
                if not fermionic else _as_numpy(term.to_dense(), dtype=dtype)
            )
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError("one-site operators must be square matrices.")
            if physical_map is None:
                physical_map = (
                    _expanded_index_charges(term.indices[0])
                    if fermionic else [0] * data.shape[0]
                )
            if fermionic:
                charge = _operator_charge_from_matrix(data, physical_map)
                zero = (
                    tuple(0 for _ in charge) if isinstance(charge, tuple) else 0
                )
                if charge != zero:
                    raise ValueError(
                        "combined native TreeMPO currently requires neutral "
                        "one-site terms."
                    )
            site = support[0]
            one_site[site] = one_site.get(site, 0) + data
            continue
        if len(support) != 2:
            raise NotImplementedError(
                "combined TreeMPO currently supports one- and two-site terms; "
                "use a precompiled structured TTNO for higher-rank terms."
            )
        if fermionic:
            term_channels, term_map = _operator_native_channels(
                term, support, symmetry=symmetry, dtype=dtype, cutoff=cutoff,
            )
        else:
            term_channels, term_map = _operator_dense_channels(
                term, support, dtype=dtype, cutoff=cutoff,
            )
        if physical_map is None:
            physical_map = list(term_map)
        elif list(term_map) != list(physical_map):
            raise ValueError("all TreeMPO terms must share one physical map.")
        for source, target, charge in term_channels:
            channels.append({
                "source": source,
                "target": target,
                "charge": charge if fermionic else 0,
                "sites": tuple(sorted(support)),
            })

    if physical_map is None:
        raise ValueError("At least one operator term is required.")
    if fermionic:
        first_charge = physical_map[0]
        zero = tuple(0 for _ in first_charge) if isinstance(first_charge, tuple) else 0
    else:
        zero = 0
    nchannel = len(channels)
    if not nchannel and not one_site:
        raise ValueError("operator terms produced no nonzero channels.")

    source_id = lambda channel: 1 + channel
    target_id = lambda channel: 1 + nchannel + channel
    done = 1 + 2 * nchannel
    state_map = [zero]
    state_map.extend(
        _operator_charge_neg(channel["charge"]) for channel in channels
    )
    state_map.extend(channel["charge"] for channel in channels)
    state_map.append(zero)
    physical_dim = len(physical_map)
    tensors = []

    for node in plan.nodes():
        children = tuple(plan.children[node])
        parent = plan.parent.get(node)
        has_parent = parent is not None
        neighbors = list(children) + ([parent] if has_parent else [])
        maps = [state_map] * len(neighbors)
        duals = [True] * len(children) + ([False] if has_parent else [])
        inds = [
            f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
            for neighbor in neighbors
        ]
        qubit = plan.qubit_of_node.get(node)
        if qubit is not None:
            # Native tree leaves conventionally expose physical legs before
            # their single virtual parent.  This ordering is not cosmetic:
            # Symmray's graded contraction phase depends on the ordered leg
            # exterior.  Keep the TTNO leaf in the same convention as the
            # native state and direct local-observable route.
            maps = [physical_map, physical_map] + maps
            duals = [False, True] + duals
            inds = [f"k{qubit}", f"b{qubit}"] + inds
        shape = [len(index_map) for index_map in maps]
        data = np.zeros(shape, dtype=dtype or complex)
        identity = (
            np.eye(physical_dim, dtype=data.dtype)
            if qubit is not None else 1.0
        )
        endpoint = {}
        for channel, info in enumerate(channels):
            if qubit == info["sites"][0]:
                endpoint.setdefault("source", []).append(channel)
            if qubit == info["sites"][1]:
                endpoint.setdefault("target", []).append(channel)

        valid_children = _operator_valid_child_states(
            len(children), len(state_map), nchannel, done,
        )
        for child_states in valid_children:
            active = []
            completed = False
            invalid = False
            for state in child_states:
                if state == 0:
                    continue
                if state == done:
                    if completed or active:
                        invalid = True
                    completed = True
                    continue
                if state < source_id(nchannel):
                    channel = state - 1
                    flag = "source"
                else:
                    channel = state - target_id(0)
                    flag = "target"
                if completed or any(
                    old_channel == channel and old_flag == flag
                    for old_channel, old_flag in active
                ):
                    invalid = True
                active.append((channel, flag))
            if invalid or len({channel for channel, _ in active}) > 1:
                continue
            if completed:
                base = done
            elif active:
                flags = {flag for _, flag in active}
                channel = active[0][0]
                base = (
                    done
                    if flags == {"source", "target"}
                    else source_id(channel)
                    if "source" in flags
                    else target_id(channel)
                )
            else:
                base = 0

            options = [(base, identity)]
            if base == 0 and qubit in one_site:
                options.append((done, one_site[qubit]))
            if base != done:
                for channel in endpoint.get("source", ()):
                    local = channels[channel]["source"]
                    if base == 0:
                        options.append((source_id(channel), local))
                    elif base == target_id(channel):
                        options.append((done, local))
                for channel in endpoint.get("target", ()):
                    local = channels[channel]["target"]
                    if base == 0:
                        options.append((target_id(channel), local))
                    elif base == source_id(channel):
                        options.append((done, local))

            for output, local in options:
                if not has_parent and output != done:
                    continue
                index = child_states + (output,) if has_parent else child_states
                if qubit is not None:
                    data[(slice(None), slice(None)) + index] += local
                else:
                    data[index] += local

        if fermionic:
            native = _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=zero,
                label=None,
            )
        else:
            native = data
        tags = [f"N{node}"]
        if qubit is not None:
            tags.append(f"I{qubit}")
        tensors.append(qtn.Tensor(native, inds=inds, tags=tags))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = (
        "native_tree_tnno" if fermionic else "dense_tree_tnno"
    )
    network.pepsy_tree_operator_bond = len(state_map)
    network.pepsy_tree_operator_raw_bond = len(state_map)
    network.pepsy_tree_operator_is_ttno = True
    return network


def _tree_plan_neighbors(plan, node):
    """Return a plan node's children followed by its optional parent."""
    return tuple(plan.children[node]) + (
        (plan.parent[node],) if plan.parent.get(node) is not None else ()
    )


def _tree_operator_peel_order(plan, nodes):
    """Return a deterministic leaf-to-hub order for a connected node set."""
    remaining = set(nodes)
    adjacency = {
        node: tuple(
            neighbor for neighbor in _tree_plan_neighbors(plan, node)
            if neighbor in remaining
        )
        for node in remaining
    }
    degree = {
        node: sum(neighbor in remaining for neighbor in neighbors)
        for node, neighbors in adjacency.items()
    }
    leaves = [node for node, value in degree.items() if value == 1]
    heapq.heapify(leaves)
    order = []
    while len(remaining) > 1:
        while leaves and leaves[0] not in remaining:
            heapq.heappop(leaves)
        if not leaves:
            raise ValueError("operator decomposition requires a connected tree")
        leaf = heapq.heappop(leaves)
        neighbor = next(
            node for node in adjacency[leaf] if node in remaining
        )
        order.append((leaf, neighbor))
        remaining.remove(leaf)
        degree[leaf] = 0
        degree[neighbor] -= 1
        if degree[neighbor] == 1:
            heapq.heappush(leaves, neighbor)
    return tuple(order), next(iter(remaining))


def _tree_operator_from_dense(
    plan,
    array,
    *,
    sites,
    dims,
    split_opts,
    site_tag_id,
    upper_ind_id,
    lower_ind_id,
    node_tag_id,
):
    """Decompose a dense matrix exactly across a TreePlan."""
    data = ar.do("reshape", array, tuple(dims) + tuple(dims))
    upper = tuple(upper_ind_id.format(site) for site in sites)
    lower = tuple(lower_ind_id.format(site) for site in sites)
    blob = qtn.Tensor(data, inds=upper + lower)

    owned = {node: [] for node in plan.nodes()}
    for site in sites:
        owned[plan.node_of_qubit[site]].extend((
            upper_ind_id.format(site),
            lower_ind_id.format(site),
        ))
    factors = {}
    peel_order, hub = _tree_operator_peel_order(plan, set(plan.nodes()))
    opts = dict(split_opts)
    opts.setdefault("method", "svd")
    opts.setdefault("absorb", "right")
    opts.setdefault("cutoff", 0.0)
    opts.setdefault("get", "tensors")

    for node, neighbor in peel_order:
        bond_ind = f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
        left, blob = blob.split(
            left_inds=tuple(owned[node]),
            right_inds=tuple(ind for ind in blob.inds if ind not in owned[node]),
            bond_ind=bond_ind,
            **opts,
        )
        factors[node] = left
        owned[neighbor].append(bond_ind)
    factors[hub] = blob

    tensors = []
    for node in plan.nodes():
        tensor = factors[node]
        qubit = plan.qubit_of_node.get(node)
        neighbors = tuple(plan.children[node]) + (
            (plan.parent[node],) if plan.parent.get(node) is not None else ()
        )
        desired = [
            *( (
                upper_ind_id.format(qubit),
                lower_ind_id.format(qubit),
            ) if qubit is not None else () ),
            *(
                f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
                for neighbor in neighbors
            ),
        ]
        tensor = tensor.transpose(*desired)
        tensor.add_tag(node_tag_id.format(node))
        if qubit is not None:
            tensor.add_tag(site_tag_id.format(qubit))
        tensors.append(tensor)

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_tnno"
    network.pepsy_tree_operator_is_ttno = True
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    return network


def _tree_operator_from_fill_fn(
    plan,
    fill_fn,
    *,
    bond_dim,
    phys_dim,
    dtype,
    site_tag_id,
    upper_ind_id,
    lower_ind_id,
    node_tag_id,
):
    """Build a regular dense TTNO from local filled tensors."""
    if isinstance(bond_dim, Integral):
        edge_dims = {
            (min(node, child), max(node, child)): int(bond_dim)
            for node in plan.nodes()
            for child in plan.children[node]
        }
    else:
        edge_values = tuple(int(value) for value in bond_dim)
        edges = tuple(
            (node, child)
            for node in plan.nodes()
            for child in plan.children[node]
        )
        if len(edge_values) != len(edges):
            raise ValueError("bond_dim must be one value per TreePlan edge.")
        edge_dims = {
            (min(node, child), max(node, child)): value
            for (node, child), value in zip(edges, edge_values)
        }

    if isinstance(phys_dim, Integral):
        physical_dims = {site: int(phys_dim) for site in plan.node_of_qubit}
    else:
        values = tuple(int(value) for value in phys_dim)
        sites = tuple(sorted(plan.node_of_qubit))
        if len(values) != len(sites):
            raise ValueError("phys_dim must have one value per tree site.")
        physical_dims = dict(zip(sites, values))

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = tuple(plan.children[node]) + (
            (plan.parent[node],) if plan.parent.get(node) is not None else ()
        )
        inds = [
            *((
                upper_ind_id.format(qubit),
                lower_ind_id.format(qubit),
            ) if qubit is not None else ()),
            *(
                f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
                for neighbor in neighbors
            ),
        ]
        shape = [
            *( (physical_dims[qubit], physical_dims[qubit])
               if qubit is not None else () ),
            *(
                edge_dims[(min(node, neighbor), max(node, neighbor))]
                for neighbor in neighbors
            ),
        ]
        try:
            data = fill_fn(tuple(shape))
        except TypeError:
            data = fill_fn(node, tuple(shape))
        tensor = qtn.Tensor(np.asarray(data, dtype=dtype), inds=inds)
        tensor.add_tag(node_tag_id.format(node))
        if qubit is not None:
            tensor.add_tag(site_tag_id.format(qubit))
        tensors.append(tensor)

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_tnno"
    network.pepsy_tree_operator_is_ttno = True
    return network


def _identity_tree_operator(
    plan,
    *,
    phys_dim=2,
    dtype=complex,
    site_tag_id="I{}",
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    node_tag_id="N{}",
):
    """Build an exact bond-one identity TTNO."""
    if isinstance(phys_dim, Integral):
        physical_dims = {site: int(phys_dim) for site in plan.node_of_qubit}
    else:
        sites = tuple(sorted(plan.node_of_qubit))
        values = tuple(int(value) for value in phys_dim)
        if len(values) != len(sites):
            raise ValueError("phys_dim must have one value per tree site.")
        physical_dims = dict(zip(sites, values))

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = tuple(plan.children[node]) + (
            (plan.parent[node],) if plan.parent.get(node) is not None else ()
        )
        inds = [
            *((
                upper_ind_id.format(qubit),
                lower_ind_id.format(qubit),
            ) if qubit is not None else ()),
            *(
                f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
                for neighbor in neighbors
            ),
        ]
        shape = [
            *((physical_dims[qubit], physical_dims[qubit])
              if qubit is not None else ()),
            *(1 for _ in neighbors),
        ]
        data = np.zeros(shape, dtype=dtype)
        if qubit is None:
            data[...] = 1
        else:
            data[(slice(None), slice(None)) + (0,) * len(neighbors)] = np.eye(
                physical_dims[qubit], dtype=dtype,
            )
        tensor = qtn.Tensor(data, inds=inds, tags=[node_tag_id.format(node)])
        if qubit is not None:
            tensor.add_tag(site_tag_id.format(qubit))
        tensors.append(tensor)
    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_identity"
    network.pepsy_tree_operator_is_ttno = True
    return network


def _native_tree_term_network(
    plan, term, support, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Decompose one native term into a graded TTNO on the selected tree.

    The decomposition is performed on the native operator tensor itself, not
    on a dense Jordan--Wigner matrix.  The physical upper/lower pair at every
    supported site is fused into one packed leg and the resulting tensor is
    peeled across the TreePlan Steiner subtree with native Symmray SVDs. This
    is the important fermionic distinction from factorizing ordinary dense
    local matrices: the native fuse/SVD retains the graded phases at every
    branch of the tree.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    support = _term_support(support)
    if any(site not in plan.node_of_qubit for site in support):
        raise ValueError(f"term support {support!r} is outside the TreePlan.")

    physical_map = _expanded_index_charges(term.indices[0])
    zero = (
        tuple(0 for _ in physical_map[0])
        if physical_map and isinstance(physical_map[0], tuple)
        else 0
    )

    endpoint_nodes = tuple(plan.node_of_qubit[site] for site in support)
    if len(support) == 1:
        factors = {
            endpoint_nodes[0]: qtn.Tensor(
                term,
                inds=(f"k{support[0]}", f"b{support[0]}"),
            )
        }
        active_nodes = {endpoint_nodes[0]}
    else:
        ordered_support = tuple(sorted(support))
        if support == ordered_support:
            ordered_term = term
        else:
            rank = len(support)
            order = tuple(sorted(range(rank), key=support.__getitem__))
            ordered_term = term.transpose(
                (*order, *(axis + rank for axis in order))
            )
        rank = len(ordered_support)
        fused = ordered_term.fuse(*(
            (axis, axis + rank) for axis in range(rank)
        ))
        packed_inds = tuple(f"_pepsy_op_packed_{site}" for site in ordered_support)
        blob = qtn.Tensor(fused, inds=packed_inds)
        ordered_nodes = tuple(plan.node_of_qubit[site] for site in ordered_support)
        active_nodes = {ordered_nodes[0]}
        for target in ordered_nodes[1:]:
            anchor = min(
                active_nodes,
                key=lambda node: len(plan.node_path(node, target)),
            )
            active_nodes.update(plan.node_path(anchor, target))
        peel_order, hub = _tree_operator_peel_order(plan, active_nodes)
        owned = {node: set() for node in active_nodes}
        for node, site in zip(ordered_nodes, ordered_support):
            owned[node].add(f"_pepsy_op_packed_{site}")
        factors = {}
        for node, neighbor in peel_order:
            left_inds = tuple(
                index for index in blob.inds if index in owned[node]
            )
            if not left_inds:
                raise RuntimeError(
                    f"operator decomposition lost subtree payload at {node}."
                )
            left, right = blob.split(
                left_inds=left_inds,
                method="svd",
                absorb="right",
                cutoff=max(64.0 * np.finfo(float).eps, float(cutoff)),
                get="tensors",
                bond_ind=f"_pepsy_op_bond_{node}_{neighbor}",
            )
            factors[node] = left
            blob = right
            owned[neighbor].add(f"_pepsy_op_bond_{node}_{neighbor}")
        factors[hub] = blob

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    def rebuild_with_axis(data, maps, duals, dense):
        return _native_from_dense(
            dense,
            symmetry=symmetry,
            index_maps=maps,
            duals=duals,
            charge=getattr(data, "charge", zero),
        )

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = _tree_plan_neighbors(plan, node)
        if node in factors:
            factor = factors[node]
            data = factor.data
            inds = list(factor.inds)
            if qubit in support:
                packed = next(
                    (
                        index for index in inds
                        if index.startswith("_pepsy_op_packed_")
                    ),
                    None,
                )
                if packed is not None:
                    axis = inds.index(packed)
                    data = data.unfuse(axis)
                    inds[axis:axis + 1] = [f"k{qubit}", f"b{qubit}"]
            elif qubit is not None:
                # A physical TreePlan root can lie on the active Steiner
                # subtree without being an endpoint. Its operator action is
                # the identity, so add that even physical pair explicitly.
                dense = _as_numpy(data.to_dense(), dtype=dtype or complex)
                dense = np.einsum(
                    "ab,...->ab...",
                    np.eye(len(physical_map), dtype=dense.dtype),
                    dense,
                )
                maps = [physical_map, physical_map] + [
                    _expanded_index_charges(index) for index in data.indices
                ]
                duals = [False, True] + [
                    index.dual for index in data.indices
                ]
                data = rebuild_with_axis(data, maps, duals, dense)
                inds = [f"k{qubit}", f"b{qubit}"] + inds

            # Rename the native decomposition's temporary bond labels before
            # adding the trivial exterior bonds.
            inds = [
                edge_name(*map(int, index.removeprefix("_pepsy_op_bond_").split("_")))
                if index.startswith("_pepsy_op_bond_") else index
                for index in inds
            ]
            existing = set(inds)
            for neighbor in neighbors:
                index = edge_name(node, neighbor)
                if index in existing:
                    continue
                dense = np.expand_dims(
                    _as_numpy(data.to_dense(), dtype=dtype or complex),
                    axis=-1,
                )
                maps = [
                    _expanded_index_charges(axis)
                    for axis in data.indices
                ] + [[zero]]
                duals = [axis.dual for axis in data.indices] + [
                    neighbor in plan.children[node]
                ]
                data = rebuild_with_axis(data, maps, duals, dense)
                inds.append(index)
                existing.add(index)

            desired = [
                *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
                *(edge_name(node, neighbor) for neighbor in neighbors),
            ]
            tensor = qtn.Tensor(data, inds=inds).transpose(*desired)
            tensor.add_tag(f"N{node}")
            if qubit is not None:
                tensor.add_tag(f"I{qubit}")
            tensors.append(tensor)
            continue

        # Nodes outside the term's Steiner subtree carry a neutral identity.
        maps = []
        duals = []
        inds = []
        if qubit is not None:
            maps.extend((physical_map, physical_map))
            duals.extend((False, True))
            inds.extend((f"k{qubit}", f"b{qubit}"))
        for neighbor in neighbors:
            maps.append([zero])
            duals.append(neighbor in plan.children[node])
            inds.append(edge_name(node, neighbor))
        shape = tuple(len(index_map) for index_map in maps)
        data = np.zeros(shape, dtype=dtype or complex)
        if qubit is None:
            data[(0,) * len(neighbors)] = 1.0
        else:
            data[(slice(None), slice(None)) + (0,) * len(neighbors)] = np.eye(
                len(physical_map), dtype=data.dtype
            )
        tensors.append(qtn.Tensor(
            _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=zero,
            ),
            inds=inds,
            tags=[f"N{node}"] + ([f"I{qubit}"] if qubit is not None else []),
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_tree_term_tnno"
    network.pepsy_tree_operator_is_ttno = True
    return network


def _normalize_native_term_edge_orientation(network, plan, *, symmetry, dtype=None):
    """Normalize native term-network virtual duals to the TreePlan orientation."""
    if not hasattr(symmetry, "parity"):
        from symmray import get_symmetry  # pylint: disable=import-outside-toplevel

        symmetry = get_symmetry(symmetry)
    for node in plan.nodes():
        tensor = network[f"N{node}"]
        for neighbor in _tree_plan_neighbors(plan, node):
            edge = f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
            axis = tensor.inds.index(edge)
            index = tensor.data.indices[axis]
            desired_dual = neighbor in plan.children[node]
            if index.dual == desired_dual:
                continue
            old_charges = _expanded_index_charges(index)
            relabelled = [_operator_charge_neg(charge) for charge in old_charges]
            zero = (
                tuple(0 for _ in relabelled[0])
                if relabelled and isinstance(relabelled[0], tuple)
                else 0
            )
            probe = _native_from_dense(
                np.zeros((len(relabelled),), dtype=dtype or complex),
                symmetry=symmetry,
                index_maps=[relabelled],
                duals=[desired_dual],
                charge=zero,
            )
            new_charges = _expanded_index_charges(probe.indices[0])
            old_positions = {}
            for position, charge in enumerate(old_charges):
                old_positions.setdefault(charge, []).append(position)
            used = {charge: 0 for charge in old_positions}
            permutation = []
            for charge in new_charges:
                old_charge = _operator_charge_neg(charge)
                position = used[old_charge]
                permutation.append(old_positions[old_charge][position])
                used[old_charge] = position + 1
            dense = np.take(
                _as_numpy(tensor.data.to_dense(), dtype=dtype or complex),
                permutation,
                axis=axis,
            )
            # Reversing a fermionic virtual edge is a graded dualization, not
            # just a charge-label permutation.  The plan-parent endpoint
            # carries the parity gauge associated with that reversal.  Apply
            # it once per edge (the child endpoint gets the dual charge map,
            # but not a second parity phase).
            if desired_dual:
                parity = np.asarray(
                    [
                        -1 if symmetry.parity(charge) else 1
                        for charge in new_charges
                    ],
                    dtype=dense.dtype,
                )
                phase_shape = [1] * dense.ndim
                phase_shape[axis] = len(parity)
                dense = dense * parity.reshape(phase_shape)
            maps = [
                new_charges if current_axis == axis else
                _expanded_index_charges(current)
                for current_axis, current in enumerate(tensor.data.indices)
            ]
            duals = [
                desired_dual if current_axis == axis else current.dual
                for current_axis, current in enumerate(tensor.data.indices)
            ]
            rebuilt = _native_from_dense(
                dense,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=getattr(tensor.data, "charge", 0),
            )
            tensor.modify(data=rebuilt)
    return network


def _native_term_sum_tree_operator(
    plan, terms, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Direct-sum exact native term TTNOs into one operator network."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    term_networks = []
    for where, term in terms.items():
        network = _native_tree_term_network(
            plan,
            term,
            _term_support(where),
            symmetry=symmetry,
            cutoff=cutoff,
            dtype=dtype,
        )
        term_networks.append(_normalize_native_term_edge_orientation(
            network, plan, symmetry=symmetry, dtype=dtype,
        ))
    term_networks = tuple(term_networks)
    if not term_networks:
        raise ValueError("At least one operator term is required.")

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    physical_map = None
    for term in terms.values():
        physical_map = _expanded_index_charges(term.indices[0])
        break
    zero = (
        tuple(0 for _ in physical_map[0])
        if physical_map and isinstance(physical_map[0], tuple)
        else 0
    )
    edge_maps = {}
    edge_positions = {index: {} for index in range(len(term_networks))}
    for node in plan.nodes():
        for neighbor in plan.children[node]:
            edge = (node, neighbor)
            edge_index = edge_name(*edge)
            charges = []
            for index, network in enumerate(term_networks):
                tensor = network[f"N{neighbor}"]
                local = tensor.data.indices[tensor.inds.index(edge_index)]
                local_charges = _expanded_index_charges(local)
                charges.extend(local_charges)
            # Symmray groups an index's sectors by charge, so a raw
            # concatenation of per-term charge lists is not a set of
            # contiguous direct-sum slices when two terms share a charge.
            # Build the actual expanded order through the same native index
            # constructor and allocate duplicate charge sectors term by term.
            probe = _native_from_dense(
                np.zeros((len(charges),), dtype=dtype or complex),
                symmetry=symmetry,
                index_maps=[charges],
                duals=[False],
                charge=zero,
            )
            global_charges = _expanded_index_charges(probe.indices[0])
            edge_maps[edge_index] = global_charges
            positions_by_charge = {}
            for position, charge in enumerate(global_charges):
                positions_by_charge.setdefault(charge, []).append(position)
            used_by_charge = {charge: 0 for charge in positions_by_charge}
            for index, network in enumerate(term_networks):
                tensor = network[f"N{neighbor}"]
                local = tensor.data.indices[tensor.inds.index(edge_index)]
                local_charges = _expanded_index_charges(local)
                positions = []
                for charge in local_charges:
                    offset = used_by_charge[charge]
                    positions.append(positions_by_charge[charge][offset])
                    used_by_charge[charge] = offset + 1
                edge_positions[index][edge_index] = tuple(positions)

    tensors = []
    for node in plan.nodes():
        neighbors = _tree_plan_neighbors(plan, node)
        qubit = plan.qubit_of_node.get(node)
        desired = [
            *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
            *(edge_name(node, neighbor) for neighbor in neighbors),
        ]
        global_maps = []
        global_duals = []
        if qubit is not None:
            global_maps.extend((physical_map, physical_map))
            global_duals.extend((False, True))
        for neighbor in neighbors:
            index = edge_name(node, neighbor)
            global_maps.append(edge_maps[index])
            # Native tree decomposition can orient an odd operator bond
            # differently at a hub than the ordinary state-tree convention.
            # The two endpoint dual flags are part of the graded operator
            # data; recomputing them from the plan would change valid local
            # sectors and drop them during ``from_dense``.
            reference = term_networks[0][f"N{node}"]
            reference_axis = reference.data.indices[
                reference.inds.index(index)
            ]
            global_duals.append(reference_axis.dual)
        shape = tuple(len(index_map) for index_map in global_maps)
        data = np.zeros(shape, dtype=dtype or complex)
        for term_index, network in enumerate(term_networks):
            tensor = network[f"N{node}"].transpose(*desired)
            local = _as_numpy(tensor.data.to_dense(), dtype=data.dtype)
            slices = []
            if qubit is not None:
                slices.extend((slice(None), slice(None)))
            for neighbor in neighbors:
                slices.append(edge_positions[term_index][edge_name(node, neighbor)])
            # ``np.ix_`` is needed here because charge-grouped duplicate
            # sectors are generally interleaved across the direct-sum axis.
            data[np.ix_(*[
                np.arange(local.shape[axis]) if isinstance(selection, slice)
                else np.asarray(selection)
                for axis, selection in enumerate(slices)
            ])] += local
        tensors.append(qtn.Tensor(
            _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=global_maps,
                duals=global_duals,
                charge=zero,
            ),
            inds=desired,
            tags=[f"N{node}"] + ([f"I{qubit}"] if qubit is not None else []),
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_tree_tnno"
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    network.pepsy_tree_operator_is_ttno = True
    return network


def _tree_operator_tensor(network, node):
    """Fetch one operator tensor by its stable TreePlan node tag."""
    return network[f"N{node}"]


def _tree_operator_bond(network, plan, node, neighbor):
    """Find the unique live operator bond for one TreePlan edge."""
    left = _tree_operator_tensor(network, node)
    right = _tree_operator_tensor(network, neighbor)
    shared = tuple(set(left.inds).intersection(right.inds))
    if len(shared) != 1:
        raise ValueError(
            f"operator TTNO edge {(node, neighbor)!r} has {len(shared)} bonds."
        )
    return shared[0]


def _tree_operator_qr(tensor, *, left_inds, bond_ind):
    """Run the lossless dense/native QR policy for one operator tensor."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel
    from .ttn import _native_qr_split_tensor  # pylint: disable=import-outside-toplevel

    options = {
        "left_inds": tuple(left_inds),
        "right_inds": (bond_ind,),
        "method": "qr",
        "absorb": "right",
        "cutoff": 0.0,
        "get": "tensors",
    }
    options["bond_ind"] = qtn.rand_uuid()
    return _native_qr_split_tensor(tensor, **options)


def _canonicalize_tree_operator(network, plan, center):
    """Canonicalize a tree operator by lossless QR from leaves to center."""
    return _canonicalize_tree_operator_region(network, plan, {center})


def _canonicalize_tree_operator_region(network, plan, region):
    """Canonicalize the complement of a connected tree region inwards."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    region = frozenset(region)
    if not region or not region.issubset(plan.children):
        raise ValueError("operator canonicalization region is invalid.")
    if _tree_subtree_span(plan, region) != region:
        raise ValueError("operator canonicalization region must be connected.")

    def distance_to_region(node):
        return min(len(plan.node_path(node, target)) for target in region)

    order = sorted(
        (node for node in plan.nodes() if node not in region),
        key=distance_to_region,
        reverse=True,
    )
    for node in order:
        path = min(
            (
                plan.node_path(node, target)
                for target in region
                if target != node
            ),
            key=len,
        )
        neighbor = path[1]
        tensor = _tree_operator_tensor(network, node)
        target = _tree_operator_tensor(network, neighbor)
        bond = _tree_operator_bond(network, plan, node, neighbor)
        kept, carry = _tree_operator_qr(
            tensor,
            left_inds=tuple(index for index in tensor.inds if index != bond),
            bond_ind=bond,
        )
        merged = qtn.tensor_contract(carry, target)
        tensor.modify(
            data=kept.data,
            inds=kept.inds,
            left_inds=kept.left_inds,
        )
        target.modify(
            data=merged.data,
            inds=merged.inds,
            left_inds=None,
        )
    network.pepsy_tree_operator_center = (
        next(iter(region)) if len(region) == 1 else None
    )
    network.pepsy_tree_operator_canonical = True
    return network


def _compress_tree_operator(network, plan, *, max_bond, cutoff):
    """Compress one combined TTNO edge-by-edge with native graded SVD."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    raw_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    if cutoff == 0.0 and max_bond is None:
        _canonicalize_tree_operator(network, plan, plan.root)
        final_bond = raw_bond
        return {
            "compressed": False,
            "cutoff": cutoff,
            "requested_max_bond": None,
            "raw_max_bond": raw_bond,
            "final_max_bond": final_bond,
            "rank_reduced": False,
        }

    order = sorted(
        (node for node in plan.nodes() if node != plan.root),
        key=lambda node: len(plan.node_path(node, plan.root)),
        reverse=True,
    )
    for node in order:
        neighbor = plan.node_path(node, plan.root)[1]
        tensor = _tree_operator_tensor(network, node)
        target = _tree_operator_tensor(network, neighbor)
        bond = _tree_operator_bond(network, plan, node, neighbor)
        combined = qtn.tensor_contract(tensor, target)
        left_inds = tuple(index for index in tensor.inds if index != bond)
        right_inds = tuple(index for index in target.inds if index != bond)
        options = {
            "left_inds": left_inds,
            "right_inds": right_inds,
            "method": "svd",
            "absorb": "right",
            "cutoff": cutoff,
            "cutoff_mode": "rsum2",
            "get": "tensors",
            "bond_ind": bond,
        }
        if max_bond is not None:
            options["max_bond"] = int(max_bond)
        left, right = combined.split(**options)
        tensor.modify(
            data=left.data,
            inds=left.inds,
            left_inds=left.left_inds,
        )
        target.modify(
            data=right.data,
            inds=right.inds,
            left_inds=right.left_inds,
        )
    final_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_bond = final_bond
    network.pepsy_tree_operator_canonical = False
    return {
        "compressed": True,
        "cutoff": cutoff,
        "requested_max_bond": None if max_bond is None else int(max_bond),
        "raw_max_bond": raw_bond,
        "final_max_bond": final_bond,
        "rank_reduced": final_bond < raw_bond,
        "max_bond_exceeded": (
            max_bond is not None and final_bond > int(max_bond)
        ),
    }


def _relocate_mpo(mpo, order, *, upper_ind_id, lower_ind_id):
    """Relabel chain positions on ``mpo`` back to logical qubit labels."""
    tag_map = {
        mpo.site_tag(position): mpo.site_tag(qubit)
        for position, qubit in enumerate(order)
    }
    index_map = {
        upper_ind_id.format(position): upper_ind_id.format(qubit)
        for position, qubit in enumerate(order)
    }
    index_map.update({
        lower_ind_id.format(position): lower_ind_id.format(qubit)
        for position, qubit in enumerate(order)
    })
    mpo.retag_(tag_map)
    mpo.reindex_(index_map)
    mpo.pepsy_tree_order = tuple(order)
    mpo.pepsy_tree_native = any(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in mpo
    )
    return mpo


def _native_from_dense(
    data, *, symmetry, index_maps, duals, charge, label=None,
):
    """Create one native Symmray tensor lazily."""
    from symmray import utils as sr_utils  # pylint: disable=import-outside-toplevel

    return sr_utils.from_dense(
        data,
        symmetry=symmetry,
        index_maps=index_maps,
        duals=duals,
        fermionic=True,
        charge=charge,
        label=label,
    )


def _pair_coefficient_factors(terms, nsite):
    """Factor an off-diagonal symmetric coefficient table, if possible."""
    first_support, first_term = next(iter(terms.items()))
    first_matrix = _as_numpy(first_term.to_dense()).reshape(
        (first_term.shape[0] * first_term.shape[2],) * 2
    )
    table = np.zeros((nsite, nsite), dtype=complex)
    for where, term in terms.items():
        support = _term_support(where)
        if len(support) != 2 or support[0] >= support[1]:
            return None
        matrix = _as_numpy(term.to_dense()).reshape(
            (term.shape[0] * term.shape[2],) * 2
        )
        denominator = np.vdot(first_matrix, first_matrix)
        ratio = np.vdot(first_matrix, matrix) / denominator
        if not np.allclose(matrix, ratio * first_matrix, rtol=1e-10, atol=1e-12):
            return None
        table[support] = ratio / 2.0
        table[support[::-1]] = ratio / 2.0

    nonzero = np.argwhere(np.abs(table) > 1e-14)
    if len(nonzero) < 3:
        return None
    i0, j0 = map(int, nonzero[0])
    candidates = [
        index for index in range(nsite)
        if index not in {i0, j0} and abs(table[i0, index]) > 1e-14
    ]
    if not candidates or abs(table[i0, j0]) <= 1e-14:
        return None
    k = candidates[0]
    a = np.zeros(nsite, dtype=complex)
    b = np.zeros(nsite, dtype=complex)
    a[i0] = 1.0
    b[j0] = table[i0, j0]
    b[k] = table[i0, k]
    a[k] = table[k, j0] / b[j0]
    if abs(a[k]) <= 1e-14 or abs(b[k]) <= 1e-14:
        return None
    a[j0] = table[j0, k] / b[k]
    b[i0] = table[k, i0] / a[k]
    for index in range(nsite):
        if index != j0 and abs(b[j0]) > 1e-14:
            a[index] = table[index, j0] / b[j0]
        if index != i0:
            b[index] = table[i0, index] / a[i0]

    for i in range(nsite):
        for j in range(nsite):
            if i == j:
                continue
            if not np.allclose(
                a[i] * b[j], table[i, j], rtol=1e-9, atol=1e-11,
            ):
                return None
    return first_term, first_support, a, b


def _pair_endpoint_automaton(
    plan, terms, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Compile a separable pair correlator into a four-state tree operator."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    factored = _pair_coefficient_factors(terms, plan.n)
    if factored is None:
        return None
    first_term, first_support, source_weights, target_weights = factored
    fused = first_term.fuse((0, 2), (1, 3))
    left, _, right = fused.svd(absorb="right", cutoff=cutoff)
    if left is None or right is None:
        return None
    left = left.unfuse(0).transpose((2, 0, 1))
    right = right.unfuse(1)
    left_data = _as_numpy(left.to_dense(), dtype=dtype or complex)[0]
    right_data = _as_numpy(right.to_dense(), dtype=dtype or complex)[0]
    physical_map = _expanded_index_charges(left.indices[1])
    physical_dim = len(physical_map)
    zero = 0 if symmetry in {"U1", "Z2"} else (0, 0)
    pair_charge = getattr(first_term, "pair_charge", None)
    if pair_charge is None:
        # The local factor's first physical charge is sufficient to infer the
        # endpoint channel charge for the standard pair observable.
        pair_charge = physical_map[-1]
    opposite_pair = (
        tuple(-value for value in pair_charge)
        if isinstance(pair_charge, tuple) else -pair_charge
    )
    state_map = [zero, opposite_pair, pair_charge, zero]
    tensors = []

    for node in plan.nodes():
        children = tuple(plan.children[node])
        parent = plan.parent.get(node)
        qubit = plan.qubit_of_node.get(node)
        has_parent = parent is not None
        edges = list(children) + ([parent] if has_parent else [])
        shape = [4] * len(edges)
        maps = [state_map] * len(edges)
        duals = [True] * len(children) + ([False] if has_parent else [])
        inds = [
            f"_to{min(node, neighbor)}_{max(node, neighbor)}"
            for neighbor in edges
        ]
        if qubit is not None:
            shape.extend((physical_dim, physical_dim))
            maps.extend((physical_map, physical_map))
            duals.extend((False, True))
            inds.extend((f"k{qubit}", f"b{qubit}"))
        data = np.zeros(shape, dtype=dtype or complex)

        if qubit is not None:
            source = source_weights[qubit] * left_data
            target = target_weights[qubit] * right_data
            identity = np.eye(physical_dim, dtype=data.dtype)

        for child_states in (
            np.ndindex(*(4 for _ in children)) if children else [()]
        ):
            source_count = sum(state & 1 for state in child_states)
            target_count = sum((state >> 1) & 1 for state in child_states)
            if source_count > 1 or target_count > 1:
                continue
            base = source_count | (target_count << 1)
            options = [(base, identity if qubit is not None else 1.0)]
            if qubit is not None:
                if not source_count:
                    options.append((1 | (target_count << 1), source))
                if not target_count:
                    options.append((source_count | 2, target))
            for output, local_operator in options:
                if has_parent:
                    index = child_states + (output,)
                else:
                    if output != 3:
                        continue
                    index = child_states
                data[index] += local_operator

        array = _native_from_dense(
            data,
            symmetry=symmetry,
            index_maps=maps,
            duals=duals,
            charge=zero,
        )
        tags = [f"N{node}"]
        if qubit is not None:
            tags.append(f"I{qubit}")
        tensors.append(qtn.Tensor(array, inds=inds, tags=tags))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "pair_endpoint_automaton"
    network.pepsy_tree_operator_bond = 4
    return network


def _pair_chain_mpo(
    terms, *, symmetry, nsite, cutoff=1e-12, dtype=None,
    upper_ind_id="k{}", lower_ind_id="b{}", site_tag_id="I{}",
):
    """Build the compact native chain MPO for a symmetric pair table.

    The two active bond sectors represent ``pair_create`` before
    ``pair_annihilate`` and the reversed ordering.  The two neutral sectors
    are the open and closed boundaries, so the dense bond dimension is four
    (and the native charge blocks remain explicit).  This is the chain
    counterpart of :func:`_pair_endpoint_automaton`.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    factored = _pair_coefficient_factors(terms, nsite)
    if factored is None:
        return None
    first_term, _, source_weights, target_weights = factored
    fused = first_term.fuse((0, 2), (1, 3))
    left, _, right = fused.svd(absorb="right", cutoff=cutoff)
    if left is None or right is None or left.shape[1] != 1:
        return None
    left = left.unfuse(0).transpose((2, 0, 1))
    right = right.unfuse(1)
    left_data = _as_numpy(left.to_dense(), dtype=dtype or complex)[0]
    right_data = _as_numpy(right.to_dense(), dtype=dtype or complex)[0]
    physical_map = _expanded_index_charges(left.indices[1])
    physical_dim = len(physical_map)
    zero = 0 if symmetry in {"U1", "Z2"} else (0, 0)
    pair_charge = physical_map[-1]
    opposite_pair = (
        tuple(-value for value in pair_charge)
        if isinstance(pair_charge, tuple) else -pair_charge
    )
    state_map = [zero, opposite_pair, pair_charge, zero]
    identity = np.eye(physical_dim, dtype=dtype or complex)

    def make_array(data, maps, duals):
        return _native_from_dense(
            data,
            symmetry=symmetry,
            index_maps=maps,
            duals=duals,
            charge=zero,
        )

    arrays = []
    for position in range(nsite):
        source = source_weights[position] * left_data
        target = target_weights[position] * right_data
        if position == 0:
            data = np.zeros((4, physical_dim, physical_dim), dtype=identity.dtype)
            data[0] = identity
            data[1] = source
            data[2] = target
            arrays.append(make_array(
                data,
                [state_map, physical_map, physical_map],
                [False, False, True],
            ))
            continue
        if position == nsite - 1:
            data = np.zeros((4, physical_dim, physical_dim), dtype=identity.dtype)
            data[3] = identity
            # Active sectors close into the neutral done boundary.
            data[1] = target
            data[2] = source
            arrays.append(make_array(
                data,
                [state_map, physical_map, physical_map],
                [True, False, True],
            ))
            continue

        data = np.zeros(
            (4, 4, physical_dim, physical_dim), dtype=identity.dtype,
        )
        for state in range(4):
            data[state, state] = identity
        data[0, 1] = source
        data[0, 2] = target
        data[1, 3] = target
        data[2, 3] = source
        arrays.append(make_array(
            data,
            [state_map, state_map, physical_map, physical_map],
            [True, False, False, True],
        ))

    return qtn.MatrixProductOperator(
        arrays,
        sites=range(nsite),
        L=nsite,
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
    )


def _tree_tensor_network_for_term(
    plan, term, support, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Build an exact native fallback network for one higher-rank term.

    Higher-rank terms can still be kept as complete graded operator tensors
    for callers that explicitly need this compatibility fallback. The normal
    Hamiltonian path uses :func:`_native_tree_term_network` and amalgamates
    the resulting factors into one canonicalizable TTNO rather than a list of
    hyperedges.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    support = tuple(int(site) for site in support)
    if not support:
        raise ValueError("native tree term support cannot be empty.")
    if any(site not in plan.node_of_qubit for site in support):
        raise ValueError(f"term support {support!r} is outside the TreePlan.")
    expected_rank = 2 * len(support)
    if len(term.indices) != expected_rank:
        raise TypeError(
            f"a {len(support)}-site native term must have rank "
            f"{expected_rank}, got {len(term.indices)}."
        )

    physical_map = _expanded_index_charges(term.indices[0])
    physical_dim = len(physical_map)
    zero = getattr(term, "zero_charge", None)
    if zero is None:
        zero = 0 if symmetry in {"U1", "Z2"} else (0, 0)
    operator_dtype = np.dtype(dtype or _as_numpy(term.to_dense()).dtype)

    # The term's native indices are ordered as all upper physical legs,
    # followed by all lower physical legs. Keep that ordering intact while
    # assigning the logical tree-site labels.
    term_inds = [f"k{site}" for site in support]
    term_inds.extend(f"b{site}" for site in support)
    term_tags = [f"N{plan.node_of_qubit[site]}" for site in support]
    term_tags.extend(f"I{site}" for site in support)
    tensors = [qtn.Tensor(term, inds=term_inds, tags=term_tags)]

    support_set = set(support)
    for site in sorted(plan.node_of_qubit):
        if site in support_set:
            continue
        identity = _native_from_dense(
            np.eye(physical_dim, dtype=operator_dtype),
            symmetry=symmetry,
            index_maps=[physical_map, physical_map],
            duals=[False, True],
            charge=zero,
        )
        node = plan.node_of_qubit[site]
        tensors.append(qtn.Tensor(
            identity,
            inds=[f"k{site}", f"b{site}"],
            tags=[f"N{node}", f"I{site}"],
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_term_hyperedge"
    span = {plan.node_of_qubit[support[0]]}
    for site in support[1:]:
        anchor = next(iter(span))
        span.update(plan.node_path(anchor, plan.node_of_qubit[site]))
    network.pepsy_tree_operator_path = tuple(sorted(span))
    return network


def _dense_operator_array(operator, *, dtype=None):
    """Extract one ordinary dense operator array."""
    if hasattr(operator, "to_dense"):
        operator = operator.to_dense()
    elif hasattr(operator, "data"):
        operator = operator.data
    return _as_numpy(operator, dtype=dtype)


def _dense_tree_tensor_network_for_term(plan, operator, support, *, dtype=None):
    """Build one exact ordinary dense tree operator hyperedge."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    support = tuple(int(site) for site in support)
    data = _dense_operator_array(operator, dtype=dtype)
    expected_rank = 2 * len(support)
    if data.ndim != expected_rank:
        raise ValueError(
            f"a {len(support)}-site dense term must have rank {expected_rank}, "
            f"got {data.ndim}."
        )
    if any(site not in plan.node_of_qubit for site in support):
        raise ValueError(f"term support {support!r} is outside the TreePlan.")
    physical_dim = data.shape[0]
    if any(size != physical_dim for size in data.shape):
        raise ValueError("dense tree terms must have one physical dimension.")
    tensors = [qtn.Tensor(
        data,
        inds=[f"k{site}" for site in support]
        + [f"b{site}" for site in support],
        tags=[f"N{plan.node_of_qubit[site]}" for site in support]
        + [f"I{site}" for site in support],
    )]
    support_set = set(support)
    identity = np.eye(physical_dim, dtype=data.dtype)
    for site in sorted(plan.node_of_qubit):
        if site in support_set:
            continue
        node = plan.node_of_qubit[site]
        tensors.append(qtn.Tensor(
            identity,
            inds=[f"k{site}", f"b{site}"],
            tags=[f"N{node}", f"I{site}"],
        ))
    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_term_hyperedge"
    span = {plan.node_of_qubit[support[0]]}
    for site in support[1:]:
        anchor = next(iter(span))
        span.update(plan.node_path(anchor, plan.node_of_qubit[site]))
    network.pepsy_tree_operator_path = tuple(sorted(span))
    return network


def _terms_by_operator_charge(terms):
    """Group native terms without mixing their Symmray operator charges."""
    grouped = {}
    for where, term in terms.items():
        grouped.setdefault(getattr(term, "charge", 0), {})[where] = term
    return grouped


def _charge_is_zero(charge):
    """Return whether an Abelian scalar or tuple charge is neutral."""
    if isinstance(charge, tuple):
        return all(value == 0 for value in charge)
    return charge == 0


def _build_mixed_charge_tree_operator(
    plan,
    hamiltonian,
    *,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    dtype=None,
    fermionic=True,
    to_backend=None,
):
    """Build one public ``TreeMPO`` from separate native charge networks."""
    from ...tensors.symmetric import (
        SymHamiltonian,
        _apply_to_tensor_network_arrays,
    )

    networks = []
    for charge, sector_terms in _terms_by_operator_charge(
        hamiltonian.terms
    ).items():
        if not _charge_is_zero(charge):
            # A nonzero-charge TTNO cannot be amalgamated into a neutral
            # direct-sum tensor: the charge belongs to one open operator
            # boundary tensor. Keep each charged term as its own homogeneous
            # network and let the public TreeMPO sum them.
            for where, term in sector_terms.items():
                network = _native_tree_term_network(
                    plan,
                    term,
                    _term_support(where),
                    symmetry=hamiltonian.symmetry,
                    cutoff=cutoff,
                    dtype=dtype,
                )
                networks.append(_normalize_native_term_edge_orientation(
                    network,
                    plan,
                    symmetry=hamiltonian.symmetry,
                    dtype=dtype,
                ))
            continue
        sector_hamiltonian = SymHamiltonian.from_terms(
            hamiltonian.model,
            hamiltonian.symmetry,
            sector_terms,
            parameters=hamiltonian.parameters,
        )
        sector_operator = _build_tree_operator(
            plan,
            sector_hamiltonian,
            cutoff=cutoff,
            max_bond=max_bond,
            compress=False,
            dtype=dtype,
            fermionic=fermionic,
        )
        if isinstance(sector_operator, TreeMPO):
            networks.extend(sector_operator.tree_networks)
        else:
            networks.append(sector_operator)

    operator = TreeMPO(
        plan,
        tuple(networks),
        terms=hamiltonian.terms,
        backend="symmray" if fermionic else "dense",
        fermionic=fermionic,
        symmetry=hamiltonian.symmetry,
        compressed=False,
    )
    if compress:
        operator.compress(max_bond=max_bond, cutoff=cutoff)
    if to_backend is not None:
        for network in operator.tree_networks:
            _apply_to_tensor_network_arrays(network, to_backend)
    return operator


def _build_tree_operator(
    plan,
    hamiltonian,
    *,
    cutoff=1e-12,
    max_bond=None,
    compress=True,
    dtype=None,
    fermionic=True,
):
    """Build the backend-specific tree representation used by ``TreeMPO``."""
    symmetry = hamiltonian.symmetry
    terms = hamiltonian.terms

    if any(
        not _charge_is_zero(charge)
        for charge in _terms_by_operator_charge(terms)
    ):
        # A charged TTNO carries its operator charge on an open boundary
        # tensor. Keep charged terms as separate homogeneous networks rather
        # than forcing them into the neutral direct-sum construction below.
        return tuple(
            _normalize_native_term_edge_orientation(
                _native_tree_term_network(
                    plan,
                    term,
                    _term_support(where),
                    symmetry=symmetry,
                    cutoff=cutoff,
                    dtype=dtype,
                ),
                plan,
                symmetry=symmetry,
                dtype=dtype,
            )
            for where, term in terms.items()
        )

    # The full staggered eta correlator is a symmetric rank-one pair table.
    # Compile it before falling back to one actual tree contraction per term;
    # this keeps p_eta_stag2 at a four-state tree bond for arbitrary N.
    if fermionic:
        # The factorization helper assumes every term is a two-site operator.
        # In particular, an onsite-only Hamiltonian is a valid generic TTNO
        # input but must go directly to the combined automaton.
        is_pair_table = (
            len(terms) >= 3
            and all(len(_term_support(where)) == 2 for where in terms)
        )
        pair_network = _pair_endpoint_automaton(
            plan, terms, symmetry=symmetry, cutoff=cutoff, dtype=dtype,
        ) if is_pair_table else None
        if pair_network is not None:
            return pair_network
        return _native_term_sum_tree_operator(
            plan,
            terms,
            symmetry=symmetry,
            cutoff=cutoff,
            dtype=dtype,
        )

    return _combined_tree_operator(
        plan,
        terms,
        symmetry=symmetry,
        cutoff=cutoff,
        dtype=dtype,
        fermionic=False,
    )


def _annotate_tree_mpo(
    mpo,
    plan,
    terms,
    tree_operator,
    *,
    symmetry=None,
    compressed=False,
    cutoff=1e-12,
    max_bond=None,
    to_backend=None,
):
    """Attach a public :class:`TreeMPO` to a compatibility chain MPO."""
    mpo.pepsy_tree_plan_signature = _tree_plan_signature(plan)
    mpo.pepsy_tree_terms = dict(terms)
    if isinstance(tree_operator, TreeMPO):
        operator = tree_operator
    else:
        networks = (
            tuple(tree_operator)
            if isinstance(tree_operator, (tuple, list))
            else (tree_operator,)
        )
        native = any(
            type(tensor.data).__name__.endswith("FermionicArray")
            for network in networks
            for tensor in network
        )
        operator = TreeMPO(
            plan,
            networks,
            chain_mpo=mpo,
            terms=terms,
            backend="symmray" if native else "dense",
            fermionic=native,
            symmetry=symmetry,
            compressed=compressed,
        )
    mpo.pepsy_tree_operator = operator
    mpo.pepsy_tree_operator_networks = operator.tree_networks
    if compressed:
        operator.compress(max_bond=max_bond, cutoff=cutoff)
    if to_backend is not None:
        # Move the primary TreeMPO representation onto the requested backend
        # so ``TreeMPO.expectation`` contracts against a matching tree state.
        # The compatibility chain MPO is converted separately by the caller;
        # here the structured operator networks are moved in place after any
        # numpy-side compression completes.
        from ...tensors.symmetric import _apply_to_tensor_network_arrays

        for network in operator.tree_networks:
            _apply_to_tensor_network_arrays(network, to_backend)
    return mpo


def tree_mpo(
    plan,
    hamiltonian,
    *,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    dtype=None,
    fermionic=True,
    charge_sectors=False,
    to_backend=None,
):
    """Build a low-bond native chain MPO plus its TreePlan embedding.

    The returned value is the ordinary chain MPO produced by
    :meth:`SymHamiltonian.to_mpo`, with logical site labels restored after the
    selected ``TreePlan.mpo_order`` construction.  On a native tree state,
    :meth:`TreeTensorNetwork.expectation_mpo_exact` uses the attached native
    tree embedding and contracts ``tree.H | tree_operator | tree``.  The chain
    MPO is never moved into the tree or compressed as part of that readout.
    """
    if not isinstance(plan, TreePlan):
        raise TypeError("plan must be a TreePlan.")

    from ...tensors.symmetric import SymHamiltonian

    if not isinstance(hamiltonian, SymHamiltonian):
        raise TypeError("hamiltonian must be a SymHamiltonian instance.")
    if not fermionic:
        warnings.warn(
            "tree_mpo(..., fermionic=False) selects the explicit "
            "Jordan--Wigner compatibility path. For a native fermionic "
            "TreeTensorNetwork, keep fermionic=True so graded Symmray "
            "blocks and signs are preserved.",
            UserWarning,
            stacklevel=2,
        )

    order = plan.mpo_order()
    if len(order) != plan.n or set(order) != set(range(plan.n)):
        raise ValueError(
            "TreePlan.mpo_order() must contain every logical qubit exactly once."
        )
    position = {qubit: index for index, qubit in enumerate(order)}
    mapped_terms = {}
    for where, term in hamiltonian.terms.items():
        support = _term_support(where)
        try:
            mapped_support = tuple(position[qubit] for qubit in support)
        except KeyError as exc:
            raise ValueError(
                f"Hamiltonian term support {support!r} is outside the tree "
                f"qubits 0 .. {plan.n - 1}."
            ) from exc
        mapped_terms[mapped_support] = term

    if not mapped_terms:
        raise ValueError("At least one Hamiltonian term is required.")
    mapped_hamiltonian = SymHamiltonian.from_terms(
        hamiltonian.model,
        hamiltonian.symmetry,
        mapped_terms,
        parameters=hamiltonian.parameters,
    )
    compact = None
    if fermionic and not charge_sectors:
        compact = _pair_chain_mpo(
            mapped_terms,
            symmetry=hamiltonian.symmetry,
            nsite=plan.n,
            cutoff=cutoff,
            dtype=dtype,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
        )
    if compact is not None:
        built = compact
    else:
        built = mapped_hamiltonian.to_mpo(
            L=plan.n,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
            dtype=dtype,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            to_backend=to_backend,
        )
    if charge_sectors:
        terms_by_charge = {}
        for where, term in hamiltonian.terms.items():
            terms_by_charge.setdefault(getattr(term, "charge", 0), {})[
                where
            ] = term
        result = {}
        for charge, mpo in built.items():
            sector_terms = terms_by_charge.get(charge, {})
            # A charge-sector MPO must carry the matching native tree
            # embedding. Reusing the full Hamiltonian embedding here would
            # silently add terms from the other returned sectors.
            sector_hamiltonian = SymHamiltonian.from_terms(
                hamiltonian.model,
                hamiltonian.symmetry,
                sector_terms,
                parameters=hamiltonian.parameters,
            ) if sector_terms else None
            tree_operator = (
                _build_tree_operator(
                    plan,
                    sector_hamiltonian,
                    cutoff=cutoff,
                    max_bond=max_bond,
                    compress=compress,
                    dtype=dtype,
                    fermionic=fermionic,
                )
                if sector_hamiltonian is not None else []
            )
            result[charge] = _annotate_tree_mpo(
                _relocate_mpo(
                    mpo,
                    order,
                    upper_ind_id=upper_ind_id,
                    lower_ind_id=lower_ind_id,
                ),
                plan,
                sector_terms,
                tree_operator,
                symmetry=hamiltonian.symmetry,
                compressed=compress,
                cutoff=cutoff,
                max_bond=max_bond,
                to_backend=to_backend,
            )
        return result
    tree_operator = _build_tree_operator(
        plan,
        hamiltonian,
        cutoff=cutoff,
        max_bond=max_bond,
        compress=compress,
        dtype=dtype,
        fermionic=fermionic,
    )
    return _annotate_tree_mpo(
        _relocate_mpo(
            built,
            order,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
        ),
        plan,
        hamiltonian.terms,
        tree_operator,
        symmetry=hamiltonian.symmetry,
        compressed=compress,
        cutoff=cutoff,
        max_bond=max_bond,
        to_backend=to_backend,
    )


def build_tree_operator(
    plan,
    hamiltonian,
    *,
    max_bond=None,
    cutoff=1e-12,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    compress=True,
    dtype=None,
    fermionic=True,
    charge_sectors=False,
    to_backend=None,
):
    """Build the canonical native :class:`TreeMPO` for a ``TreePlan``.

    The compatibility :func:`tree_mpo` builder returns the ordinary linear
    chain MPO and attaches this tree operator to it. This function returns
    only the tree-native operator, keeping the two tensor-network geometries
    explicit. Mixed native charges are combined into one ``TreeMPO`` with one
    homogeneous Symmray network per charge. With ``charge_sectors=True`` it
    instead returns ``{charge: TreeMPO}`` for callers that need separate
    sector objects.
    """
    if fermionic and not charge_sectors:
        from ...tensors.symmetric import SymHamiltonian

        if not isinstance(hamiltonian, SymHamiltonian):
            raise TypeError("hamiltonian must be a SymHamiltonian instance.")
        if any(
            not _charge_is_zero(charge)
            for charge in _terms_by_operator_charge(hamiltonian.terms)
        ):
            return _build_mixed_charge_tree_operator(
                plan,
                hamiltonian,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                dtype=dtype,
                fermionic=fermionic,
                to_backend=to_backend,
            )
    built = tree_mpo(
        plan,
        hamiltonian,
        max_bond=max_bond,
        cutoff=cutoff,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        compress=compress,
        dtype=dtype,
        fermionic=fermionic,
        charge_sectors=charge_sectors,
        to_backend=to_backend,
    )
    if isinstance(built, dict):
        return {
            charge: mpo.pepsy_tree_operator
            for charge, mpo in built.items()
        }
    return built.pepsy_tree_operator
