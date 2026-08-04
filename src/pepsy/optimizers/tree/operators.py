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

The returned public object remains a regular Quimb MPO for compatibility with
MPS/MPO APIs.  It carries the tree operator as private metadata consumed by
``TreeTensorNetwork.expectation_mpo_exact``.  The two networks remain
separate throughout the contraction.
"""

from __future__ import annotations

import heapq
from numbers import Integral
import warnings

import numpy as np

from .layout import TreePlan

__all__ = ["TreeMPO", "tree_mpo"]


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


class TreeMPO:
    """TreePlan-aware operator with dense and native Symmray backends.

    ``TreeMPO`` is the operator-level API for measurements on a
    :class:`TreeTensorNetwork`.  It deliberately keeps the optional linear
    chain MPO separate from the tree representation:

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

    def __init__(
        self,
        plan,
        tree_networks,
        *,
        chain_mpo=None,
        terms=None,
        backend="dense",
        fermionic=False,
        symmetry=None,
        cutoff=1e-12,
        compressed=False,
    ):
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan.")
        if isinstance(tree_networks, (tuple, list)):
            networks = tuple(tree_networks)
        else:
            networks = (tree_networks,)
        if not networks or any(network is None for network in networks):
            raise ValueError("TreeMPO requires at least one tree operator network.")
        self.plan = plan
        self.tree_networks = networks
        self.chain_mpo = chain_mpo
        self.terms = None if terms is None else dict(terms)
        self.backend = str(backend)
        self.fermionic = bool(fermionic)
        self.symmetry = symmetry
        self.cutoff = float(cutoff)
        self.compressed = bool(compressed)
        self.pepsy_tree_plan_signature = _tree_plan_signature(plan)
        if chain_mpo is not None:
            chain_mpo.pepsy_tree_plan_signature = self.pepsy_tree_plan_signature
            chain_mpo.pepsy_tree_terms = (
                None if self.terms is None else dict(self.terms)
            )
            chain_mpo.pepsy_tree_operator = self
            chain_mpo.pepsy_tree_operator_networks = self.tree_networks

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

    @property
    def tree_network(self):
        """Return the sole tree network, or raise for a term sum."""
        if len(self.tree_networks) != 1:
            raise AttributeError(
                "this TreeMPO contains multiple internal networks; use "
                "tree_networks or expectation()"
            )
        return self.tree_networks[0]

    def max_bond(self):
        """Return the largest virtual bond among the tree networks."""
        bonds = []
        for network in self.tree_networks:
            for index in network.inner_inds():
                bonds.append(network.ind_size(index))
        return max(bonds, default=1)

    def canonicalize(self, center=None):
        """Canonicalize every stored TTNO around one TreePlan node."""
        if center is None:
            center = self.plan.root
        for network in self.tree_networks:
            _canonicalize_tree_operator(network, self.plan, center)
        return self

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
        return self

    def copy(self):
        """Copy the operator and both of its optional representations."""
        chain_mpo = None if self.chain_mpo is None else self.chain_mpo.copy()
        copied = type(self)(
            self.plan,
            tuple(network.copy() for network in self.tree_networks),
            chain_mpo=chain_mpo,
            terms=self.terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=self.compressed,
        )
        if chain_mpo is not None:
            chain_mpo.pepsy_tree_plan_signature = copied.pepsy_tree_plan_signature
            chain_mpo.pepsy_tree_terms = (
                None if copied.terms is None else dict(copied.terms)
            )
            chain_mpo.pepsy_tree_operator = copied
            chain_mpo.pepsy_tree_operator_networks = copied.tree_networks
        return copied

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
    left_data = np.asarray(left.to_dense(), dtype=dtype)
    right_data = np.asarray(right.to_dense(), dtype=dtype)
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
                if not fermionic else np.asarray(term.to_dense(), dtype=dtype)
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
    term_charge = getattr(term, "charge", zero)
    if term_charge != zero:
        raise ValueError(
            "a single native TTNO must be neutral; use charge_sectors=True "
            "for a charged operator sum."
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
                dense = np.asarray(data.to_dense(), dtype=dtype or complex)
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
                    np.asarray(data.to_dense(), dtype=dtype or complex),
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
                np.asarray(tensor.data.to_dense(), dtype=dtype or complex),
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
            local = np.asarray(tensor.data.to_dense(), dtype=data.dtype)
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
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    if center not in plan.children:
        raise ValueError(f"operator canonicalization center {center!r} is invalid.")
    order = sorted(
        (node for node in plan.nodes() if node != center),
        key=lambda node: len(plan.node_path(node, center)),
        reverse=True,
    )
    for node in order:
        neighbor = plan.node_path(node, center)[1]
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
    network.pepsy_tree_operator_center = int(center)
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
    first_matrix = np.asarray(first_term.to_dense()).reshape(
        (first_term.shape[0] * first_term.shape[2],) * 2
    )
    table = np.zeros((nsite, nsite), dtype=complex)
    for where, term in terms.items():
        support = _term_support(where)
        if len(support) != 2 or support[0] >= support[1]:
            return None
        matrix = np.asarray(term.to_dense()).reshape(
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
    left_data = np.asarray(left.to_dense(), dtype=dtype or complex)[0]
    right_data = np.asarray(right.to_dense(), dtype=dtype or complex)[0]
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
    left_data = np.asarray(left.to_dense(), dtype=dtype or complex)[0]
    right_data = np.asarray(right.to_dense(), dtype=dtype or complex)[0]
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
    operator_dtype = np.dtype(dtype or np.asarray(term.to_dense()).dtype)

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
    return np.asarray(operator, dtype=dtype)


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
    )
