"""Tree state regression tests."""


import numpy as np
import pytest
import quimb.tensor as qtn
import pepsy
from pepsy.optimizers.tree import TreeLayoutFinder
from pepsy.optimizers.tree import TreeMPO
from pepsy.optimizers.tree import TreeOptimizer
from pepsy.optimizers.tree import TreePlan
from pepsy.optimizers.tree import TreeTensorNetwork


from _tree_test_helpers import (
    _exact_state,
    _fidelity,
    _rand_unitary,
    _random_stream,
    _sv_apply_1q,
    _two_branch_flip_submpo,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.tree]


def _nonbinary_plan():
    """Two arity-3 star nodes under a binary root over qubits 0..5."""
    children = {
        0: (), 1: (), 2: (), 3: (), 4: (), 5: (),
        6: (0, 1, 2), 7: (3, 4, 5), 8: (6, 7),
    }
    qubit_of_leaf = {i: i for i in range(6)}
    return TreePlan.from_children(children, qubit_of_leaf)


def _entangled_ttn(seed=0, n=6, D=3, structure="balanced"):
    """A canonical-at-root random tree state for centre-movement tests."""
    plan = TreePlan.from_order(range(n), structure=structure)
    return TreeTensorNetwork.rand(plan, D=D, seed=seed)


def test_ttn_from_plan_is_product_state():
    """from_plan builds |0...0> with the expected tags, indices, and sites."""
    plan = TreePlan.from_order(range(5), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    assert isinstance(ttn, TreeTensorNetwork)
    assert ttn.nqubits == 5 == ttn.nsites
    assert tuple(ttn.sites) == (0, 1, 2, 3, 4)
    # site index / tag / node tag conventions
    assert ttn.site_ind(2) == "k2"
    assert ttn.site_tag(2) == "I2"
    assert ttn.node_tag(plan.root) == f"N{plan.root}"
    # dense state is exactly |0...0>
    sv = ttn.to_statevector()
    assert sv.shape == (2**5,)
    assert abs(sv[0] - 1.0) < 1e-12
    assert np.linalg.norm(sv[1:]) < 1e-12


def test_binary_tree_supports_a_three_virtual_leg_top_tensor():
    """A ternary virtual root keeps every tensor in the binary rank class."""
    plan = TreePlan.from_order(
        range(9), structure="balanced", max_arity=2, top_arity=3,
    )
    assert plan.top_arity == 3
    assert plan.is_binary()
    assert not plan.is_strictly_binary()
    assert all(
        len(children) in (0, 2)
        for node, children in plan.children.items()
        if node != plan.root
    )

    ttn = TreeTensorNetwork.from_plan(plan)
    assert len(ttn.node_tensor(plan.root).inds) == 3
    assert ttn.max_virtual_degree == 3
    assert ttn.max_tensor_rank == 3
    assert ttn.validate(check_canonical=True) is ttn

    ordered = TreeTensorNetwork.from_order(
        range(9), max_arity=2, top_arity=3,
    )
    assert ordered.top_arity == 3
    assert len(ordered.node_tensor(ordered.plan.root).inds) == 3

    finder = TreeLayoutFinder([], n=9, max_arity=2, top_arity=3)
    found = finder.run()
    report = finder.report(found)
    assert found.top_arity == 3
    assert found.is_binary()
    assert report["top_arity"] == 3
    assert report["max_tensor_rank"] == 3

    automatic = TreeOptimizer(
        [], n=9, max_arity=2, top_arity=3, run=False,
    )
    assert automatic.plan.top_arity == 3
    assert automatic.tn.max_tensor_rank == 3

    layout = TreeLayoutFinder([], n=9, max_arity=2, top_arity=3)
    from_layout = TreeOptimizer([], layout=layout, run=False)
    assert from_layout.top_arity == 3
    assert from_layout.plan.top_arity == 3

    product = pepsy.ps_to_ttn(9, max_arity=2, top_arity=3)
    assert product.top_arity == 3
    assert product.max_tensor_rank == 3

    random = pepsy.hrs_to_ttn(9, max_arity=2, top_arity=3, seed=11)
    assert random.top_arity == 3
    assert random.max_tensor_rank == 3


def test_binary_tree_with_ternary_root_is_the_shared_default():
    """All high-level tree builders share the rank-three root convention."""
    plan = TreePlan.from_order(range(9), structure="balanced")
    assert plan.top_arity == 3
    assert plan.is_binary()
    assert not plan.is_strictly_binary()

    ordered = TreeTensorNetwork.from_order(range(9))
    assert ordered.top_arity == 3
    assert len(ordered.node_tensor(ordered.plan.root).inds) == 3

    finder = TreeLayoutFinder([], n=9)
    found = finder.run()
    assert found.top_arity == 3
    assert found.is_binary()

    optimizer = TreeOptimizer([], n=9, run=False)
    assert optimizer.plan.top_arity == 3
    assert optimizer.tn.max_tensor_rank == 3

    product = pepsy.ps_to_ttn(9)
    random = pepsy.hrs_to_ttn(9, seed=11)
    assert product.top_arity == random.top_arity == 3
    assert product.max_tensor_rank == random.max_tensor_rank == 3

    # A physical root cannot also use three incoming virtual bonds, and small
    # systems naturally fall back to the ordinary binary root.
    rooted = TreePlan.from_order(range(8), root_qubit=8)
    assert rooted.top_arity == 2
    assert TreePlan.from_order(range(2)).top_arity == 2


def test_ps_to_ttn_matches_product_state_constructor_api():
    """The high-level TTN constructor mirrors ``ps_to_mps`` amplitudes."""
    theta = 0.31
    expected = np.array([1.0], dtype="complex128")
    local = np.array([np.cos(theta), np.sin(theta)], dtype="complex128")
    for _ in range(4):
        expected = np.kron(expected, local)

    state = pepsy.ps_to_ttn(4, theta=theta)
    assert isinstance(state, TreeTensorNetwork)
    assert state.max_bond() == 1
    assert np.allclose(state.to_statevector(), expected)
    assert state.is_canonical_form(state.root)

    expanded = pepsy.ps_to_ttn(4, chi=2, rand_strength=0.0)
    assert expanded.max_bond() == 2
    assert np.allclose(expanded.to_statevector(), [1.0] + [0.0] * 15)

    plan = TreePlan.from_order(range(4), structure="balanced")
    explicit = pepsy.ps_to_ttn(4, tree=plan)
    assert explicit.plan is plan


def test_product_ttn_constructors_support_a_physical_root_site():
    """Product/random public constructors resolve every physical node."""
    theta = 0.23
    local = np.array([np.cos(theta), np.sin(theta)], dtype="complex128")
    expected = local
    for _ in range(4):
        expected = np.kron(expected, local)

    plan = TreePlan.from_order(
        [0, 1, 3, 4], structure="balanced", root_qubit=2,
    )
    explicit = pepsy.ps_to_ttn(5, tree=plan, theta=theta)
    automatic = pepsy.ps_to_ttn(5, root_qubit=2, theta=theta)
    smallest = pepsy.ps_to_ttn(2, root_qubit=1, theta=theta)
    random = pepsy.hrs_to_ttn(5, root_qubit=2, seed=11)

    assert explicit.plan is plan
    assert automatic.plan.root_qubit == 2
    assert smallest.plan.n == 2
    assert smallest.plan.root_qubit == 1
    assert random.plan.root_qubit == 2
    assert np.allclose(explicit.to_statevector(), expected)
    assert np.allclose(automatic.to_statevector(), expected)
    assert random.to_statevector().shape == (2**5,)
    assert explicit.validate(check_canonical=True) is explicit

    with pytest.raises(ValueError, match="root_qubit does not match"):
        pepsy.ps_to_ttn(5, tree=plan, root_qubit=3)


def test_ttn_copy_preserves_geometry_and_type():
    """copy() keeps the plan, ids, and class, with an independent tid cache."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    other = ttn.copy()
    assert type(other) is TreeTensorNetwork
    assert other.plan is ttn.plan
    assert other.site_ind_id == ttn.site_ind_id
    assert other.node_tag_id == ttn.node_tag_id
    # tid cache is rebuilt lazily on the copy (fresh tensor identities)
    assert other.node_tid(2) in other.tensor_map


def test_plain_tensor_network_cast_requires_explicit_plan():
    """A generic Quimb network cannot silently become geometry-owning."""
    plain = qtn.TensorNetwork([
        qtn.Tensor(np.ones(2), inds=("k0",)),
    ])
    with pytest.raises(TypeError, match="explicit TreePlan"):
        TreeTensorNetwork(plain)


def test_layout_rejects_out_of_range_supports_early():
    """Bad interaction supports fail during layout construction, not replay."""
    with pytest.raises(ValueError, match="outside"):
        TreeLayoutFinder(supports=[(0, 3)], n=2)


def test_three_qubit_torch_operator_is_backend_coerced():
    """Torch operators work with the default NumPy-backed TTN state."""
    torch = pytest.importorskip("torch")
    opt = TreeOptimizer(None, n=3, run=False)
    with pytest.warns(UserWarning, match="backend-compatible gate"):
        opt.apply_subtree_operator(
            torch.eye(8, dtype=torch.complex128), (0, 1, 2)
        )
    assert np.allclose(opt.to_dense(), [1.0] + [0.0] * 7)


def test_tree_torch_state_stays_native_across_public_operations():
    """Tree controls, Pauli helpers, and readout preserve a Torch TTN."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(3), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    assert state.validate_isometry_metadata() is state
    h = to_backend(
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    )
    cnot = to_backend(np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    ))

    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], state=state, chi=8)
    assert opt.backend_info() == {
        "backend": "torch", "dtype": "complex128", "device": "cpu",
    }
    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)
    opt.apply_subtree_operator(to_backend(np.eye(8, dtype=complex)), (0, 1, 2))
    assert opt.tn.validate(check_canonical=True) is opt.tn
    opt.project_pauli("ZZ", (0, 1), +1)
    assert opt.measure(2, outcome=0) == 0
    assert opt.reset(2) == 0
    opt.cap(2, to_backend(np.array([1.0, 0.0], dtype=complex)))
    opt.apply_pauli_rotation(0.2, "XZ", (0, 1))
    opt.apply_pauli_sum([(1.0, {0: "X", 1: "Z"})])

    assert all(torch.is_tensor(tensor.data) for tensor in opt.tn.tensor_map.values())


def test_tree_canonical_check_uses_backend_scalar_reduction(monkeypatch):
    """Canonical diagnostics must not convert device tensors with NumPy."""
    torch = pytest.importorskip("torch")
    import importlib

    ttn_module = importlib.import_module("pepsy.optimizers.tree.ttn")
    state = TreeTensorNetwork.from_plan(TreePlan.from_order(range(3)))
    state.apply_to_arrays(
        pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    )

    def fail_to_numpy(_value):
        raise AssertionError("canonical checks must stay on the live backend")

    monkeypatch.setattr(ttn_module.ar, "to_numpy", fail_to_numpy)
    assert state.is_canonical_form()


def test_tree_warns_once_when_a_gate_does_not_match_the_state_backend():
    """User payload mismatches are explicit while compatibility is preserved."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    opt = TreeOptimizer(None, state=state, run=False)

    with pytest.warns(UserWarning, match="backend-compatible gate"):
        opt.apply_1q(np.eye(2, dtype=complex), 0)
    opt.apply_1q(np.eye(2, dtype=complex), 1)
    assert opt.backend_info()["backend"] == "torch"


def test_tree_gate_stream_backend_requires_explicit_preparation():
    """Every stream gate must already match the live TTN backend."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    gates = [
        np.eye(2, dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    ]
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    assert all(isinstance(gate, np.ndarray) for gate in gates)

    matching = [to_backend(gate) for gate in gates]
    opt.set_gates([(matching[0], 0), (matching[1], 1)])
    with pytest.raises(TypeError, match=r"stream\[1\].*gate"):
        opt.set_gates([(matching[0], 0), (gates[1], 1)])


def test_tree_gate_stream_backend_checks_late_payloads():
    """A matching first gate cannot hide a later backend mismatch."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    matching = to_backend(np.eye(2, dtype=complex))
    foreign = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    with pytest.raises(TypeError, match=r"stream\[1\].*gate"):
        opt._validate_gate_stream_backend(
            [matching, foreign], ["gate", "gate"]
        )


def test_tree_optimizer_reports_symmray_block_backend():
    """Native fermionic TTNs report their underlying block backend."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")

    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(3), structure="balanced")
    state = pepsy.ps_to_ttn(
        3,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        dtype="complex128",
    )
    state.apply_to_arrays(
        pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    )
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    assert opt.backend_info() == {
        "backend": "symmray",
        "dtype": "complex128",
        "device": "cpu",
        "array_backend": "torch",
    }


def test_tree_submpo_stream_backend_requires_explicit_preparation():
    """Every stream sub-MPO tensor must match the live TTN backend."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    submpo = _two_branch_flip_submpo(L=2, sites=(0, 1), targets=(0, 1))
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    with pytest.raises(TypeError, match=r"stream\[0\].*sub-MPO"):
        opt.set_gates([opt.submpo_event(submpo, (0, 1))])
    assert all(isinstance(tensor.data, np.ndarray) for tensor in submpo.tensors)

    prepared = opt.to_backend(submpo)
    opt.set_gates([opt.submpo_event(prepared, (0, 1))])


def test_tree_treemppo_stream_backend_requires_explicit_preparation():
    """Every TreeMPO tensor must match the live TTN backend."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    identity = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
    tree_mpo = TreeMPO.from_terms(
        plan,
        {(0, 1): identity},
        compress=False,
    )
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    with pytest.raises(TypeError, match=r"stream\[0\].*TreeMPO"):
        opt.set_gates([opt.subtreempo_event(tree_mpo, (0, 1))])

    prepared = opt.to_backend(tree_mpo)
    assert all(
        torch.is_tensor(tensor.data)
        for network in prepared.tree_networks
        for tensor in network
    )
    assert all(
        isinstance(tensor.data, np.ndarray)
        for network in tree_mpo.tree_networks
        for tensor in network
    )
    opt.set_gates([opt.subtreempo_event(prepared, (0, 1))])
    opt.run()
    np.testing.assert_allclose(opt.to_dense(), [1.0, 0.0, 0.0, 0.0])
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_rejects_a_mixed_backend_initial_state():
    """A TTN must use one backend, dtype, and device across all tensors."""
    torch = pytest.importorskip("torch")
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    leaf = state.leaf_of_qubit(0)
    state.node_tensor(leaf).modify(
        data=torch.as_tensor(state.node_tensor(leaf).data, dtype=torch.complex128)
    )

    with pytest.raises(TypeError, match="one compatible backend"):
        TreeOptimizer(None, state=state, run=False)


def test_ttn_geometry_helpers_match_plan():
    """Geometry delegators agree with the underlying TreePlan."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    root = ttn.root
    for child in ttn.children(root):
        assert ttn.parent(child) == root
        assert root in ttn.neighbors(child)
        # deterministic, symmetric bond name
        assert ttn.bond(child, root) == ttn.bond(root, child)
    leaf = ttn.leaf_of_qubit(0)
    assert ttn.qubit_of_leaf(leaf) == 0
    assert ttn.is_leaf(leaf)
    # steiner subtree of two leaves == their node path
    la, lb = ttn.leaf_of_qubit(0), ttn.leaf_of_qubit(5)
    assert ttn.steiner_nodes([la, lb]) == set(ttn.node_path(la, lb))
    with pytest.raises(ValueError):
        ttn.bond(la, lb)  # non-adjacent


def test_tree_edge_entropy_is_zero_for_product_state_and_preserves_gauge():
    """The all-edge diagnostic is non-mutating and vanishes for |0...0>."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    before = state.to_statevector().copy()
    region = state.canonical_region

    entropies, edges = state.tree_edge_entropies(return_edges=True)

    assert edges == state.tree_edges()
    assert len(edges) == len(plan.parent)
    assert np.all(entropies >= 0.0)
    np.testing.assert_allclose(entropies, 0.0, atol=1e-13)
    np.testing.assert_allclose(state.to_statevector(), before, atol=1e-13)
    assert state.canonical_region == region
    np.testing.assert_allclose(
        state.entanglement_entropy(), entropies, atol=1e-13,
    )


def test_tree_edge_entropy_matches_bell_state_and_optimizer_delegate():
    """Every Bell-state leaf cut carries one bit of tree-edge entropy."""
    plan = TreePlan.from_order(range(2), structure="balanced", top_arity=2)
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    optimizer = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))],
        n=2,
        tree=plan,
        chi=None,
        cutoff=0.0,
    )

    entropies, edges = optimizer.tree_edge_entropies(return_edges=True)

    assert edges == ((plan.root, plan.leaf_of_qubit[0]),
                     (plan.root, plan.leaf_of_qubit[1]))
    np.testing.assert_allclose(entropies, 1.0, atol=1e-12)
    assert optimizer.entropy(edges[0]) == pytest.approx(1.0)
    np.testing.assert_allclose(
        optimizer.to_dense(),
        np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0),
        atol=1e-12,
    )


def test_tree_edge_entropy_supports_native_u1_and_torch_states():
    """Entropy uses compact native spectra on structured and Torch trees."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=False,
        symmetry="U1",
        dtype="complex128",
    )
    native = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=(0, 1, 0, 1),
    )
    native_entropies = native.tree_edge_entropies()
    assert native_entropies.shape == (len(plan.parent),)
    np.testing.assert_allclose(native_entropies, 0.0, atol=1e-13)

    torch = pytest.importorskip("torch")
    dense = TreeTensorNetwork.rand(plan, D=2, seed=17, dtype="complex128")
    dense.apply_to_arrays(
        lambda array: torch.as_tensor(array, dtype=torch.complex128),
    )
    torch_entropies = dense.tree_edge_entropies(method="eig")
    assert torch_entropies.shape == (len(plan.parent),)
    assert np.all(np.isfinite(torch_entropies))
    assert np.all(torch_entropies >= 0.0)


def test_ttn_validate_checks_structure_and_canonicality():
    """TreeTensorNetwork.validate catches malformed physical legs."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    assert ttn.validate(check_canonical=True) is ttn

    broken = ttn.copy()
    broken.reindex_({broken.site_ind(0): "broken-physical"})
    with pytest.raises(ValueError, match="missing physical index"):
        broken.validate()


def test_tree_canonize_mps_compatibility_entry_point():
    """Shared coefficient frontends can use the MPS canonicalization name."""
    opt = TreeOptimizer(None, n=4, run=False)
    info = {}
    assert opt.canonize_mps(opt.p, (0, 3), info=info) == (0, 3)
    assert info["cur_orthog"] == (0, 3)
    assert opt.is_subtree_canonical_form()
    assert opt.canonize_mps(opt.p, 2, info=info) == (2, 2)
    assert opt.is_canonical_form(opt.plan.leaf_of_qubit[2])


def test_ttn_rand_is_canonical_around_root():
    """rand(canonicalize=True) leaves the root tensor as the orthogonality centre."""
    import quimb.tensor as qtn

    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.rand(plan, D=4, seed=0)
    root_t = ttn.node_tensor(ttn.root)
    canon_norm = float(
        np.sqrt(np.abs(qtn.tensor_contract(root_t.H, root_t, output_inds=[])))
    )
    full_norm = float(np.sqrt(np.abs((ttn.H & ttn).contract(output_inds=[]))))
    assert np.isclose(canon_norm, full_norm)


@pytest.mark.filterwarnings(
    "ignore:The contraction tree is not a compressed one"
)
def test_ttn_gate_and_local_expectation():
    """Inherited gate/canonicalisation/expectation work on the tree."""
    plan = TreePlan.from_order(range(5), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    ttn.gate_inds_(x, [ttn.site_ind(2)], contract=True)
    ttn.canonize_around_node_(ttn.leaf_of_qubit(2))
    val = ttn.local_expectation(z, [2], max_bond=None, optimize="auto")
    assert abs(val + 1.0) < 1e-9  # <Z> = -1 after X


def test_ttn_multisite_local_expectation_contracts_only_canonical_subtree():
    """The custom Steiner-subtree readout matches a full dense-TTN overlap."""
    rng = np.random.default_rng(41)
    ttn = TreeTensorNetwork.rand(TreePlan.from_order(range(5)), D=3, seed=7)
    matrix = rng.standard_normal((4, 4)) + 1.0j * rng.standard_normal((4, 4))
    operator = matrix + matrix.conj().T
    where = (1, 4)
    operated = qtn.tensor_network_gate_inds(
        ttn,
        operator,
        [ttn.site_ind(site) for site in where],
        contract=False,
        inplace=False,
        tags=[],
    )
    expected = (ttn.H | operated).contract(all, optimize="auto")
    expected /= (ttn.H | ttn).contract(all, optimize="auto")

    actual = ttn.local_expectation(operator, where, max_bond=None, optimize="auto")
    assert actual == pytest.approx(expected)


def test_optimizer_state_is_a_tree_tensor_network():
    """TreeOptimizer builds its state on the TreeTensorNetwork class."""
    opt = TreeOptimizer(None, n=4)
    assert isinstance(opt.tn, TreeTensorNetwork)
    assert opt.tn.plan is opt.plan


def test_ttn_show_ascii_tree(capsys):
    """show() prints a top-down tree with leaf/qubit labels and bond dims."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    text = ttn.ascii_tree()
    # root marker on top, qubit leaves labelled at the bottom
    assert text.splitlines()[0].strip() == "\u25cf"
    for q in range(4):
        assert f"q{q}" in text
    assert "\u25c6" in text  # leaf markers drawn
    # box-drawing connectors are used
    assert "\u2534" in text and "\u250c" in text

    def dim_rows(drawing):
        # bond-dim annotation rows contain only digits and whitespace
        return [
            ln for ln in drawing.splitlines()
            if ln.strip() and all(c.isdigit() or c.isspace() for c in ln)
        ]

    # product state: every annotated bond dimension is 1
    rows = dim_rows(text)
    assert rows and all(set(ln.split()) <= {"1"} for ln in rows)
    # dropping bond dims removes the annotation rows but keeps the structure
    assert not dim_rows(ttn.ascii_tree(bond_dims=False))
    # the coloured drawing embeds ANSI escapes but strips back to the plain one
    colored = ttn.ascii_tree(color=True)
    assert "\x1b[" in colored
    import re as _re
    assert _re.sub(r"\x1b\[[0-9;]*m", "", colored) == text
    # show() prints the coloured drawing by default (+ trailing newline)
    ttn.show()
    assert capsys.readouterr().out.rstrip("\n") == colored
    # ...and the plain drawing when colour is disabled
    ttn.show(color=False)
    assert capsys.readouterr().out.rstrip("\n") == text
    # optimizer delegates to the state's drawing
    TreeOptimizer(None, n=4).show(color=False)
    assert capsys.readouterr().out.rstrip("\n") == text


def test_from_children_builds_and_validates():
    """from_children builds an arbitrary-arity tree and validates its shape."""
    plan = _nonbinary_plan()
    assert plan.n == 6
    assert plan.root == 8
    assert plan.max_arity() == 3
    assert not plan.is_binary()
    assert plan.parent[6] == 8 and plan.parent[0] == 6
    # star geodesics inside a clique are length two (vs up to three when split)
    assert plan.tree_distance(0, 1) == 2
    assert plan.tree_distance(0, 2) == 2


def test_from_children_rejects_invalid_trees():
    """from_children raises on malformed children / leaf maps."""
    # a node with two parents
    with pytest.raises(ValueError):
        TreePlan.from_children(
            {0: (), 1: (), 2: (0, 1), 3: (0,)}, {0: 0, 1: 1}
        )
    # a leaf missing its qubit assignment
    with pytest.raises(ValueError):
        TreePlan.from_children({0: (), 1: (), 2: (0, 1)}, {0: 0})
    # leaf qubits must be 0..n-1 without gaps
    with pytest.raises(ValueError):
        TreePlan.from_children(
            {0: (), 1: (), 2: (0, 1)}, {0: 0, 1: 2}
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nonbinary_tree_matches_statevector(seed):
    """Untruncated replay on a hand-built non-binary tree is exact."""
    rng = np.random.default_rng(seed)
    n = 6
    plan = _nonbinary_plan()
    stream = _random_stream(n, 8 * n, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=256)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


@pytest.mark.parametrize("max_arity", [3, 4])
def test_kary_layout_flatter_and_exact(max_arity):
    """k-ary layouts raise the arity and still replay exactly at large chi."""
    rng = np.random.default_rng(11)
    n = 8
    plan = TreePlan.from_order(range(n), structure="balanced",
                               max_arity=max_arity)
    assert plan.max_arity() <= max_arity
    assert plan.max_arity() > 2  # genuinely non-binary
    stream = _random_stream(n, 40, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=1 << n)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_binary_defaults_unchanged():
    """max_arity=2 keeps the strictly-binary tree for every structure."""
    for structure in ("quality", "balanced"):
        plan = TreePlan.from_order(range(9), structure=structure)
        assert plan.is_binary()
    # a scalar max_arity=2 opts back into a binary layout-finder tree
    rng = np.random.default_rng(2)
    stream = _random_stream(8, 60, rng)
    assert TreeLayoutFinder(stream, n=8, max_arity=2).run().is_binary()


def test_adaptive_layout_emits_star_for_cliques():
    """Adaptive layout collapses mutually coupled cliques into flat stars."""
    stream = []
    for _ in range(20):
        for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            stream.append((np.eye(4, dtype=complex), (a, b)))
    stream.append((np.eye(4, dtype=complex), (2, 3)))  # weak cross link
    finder = TreeLayoutFinder(stream, n=6, structure="adaptive",
                              max_arity=None)
    plan = finder.run()
    assert plan.max_arity() == 3  # each clique becomes an arity-3 star
    # every intra-clique geodesic is the star length two
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        assert plan.tree_distance(a, b) == 2
    # and it is a better structure than the binary layout for these weights
    binary = TreePlan.from_order(range(6), weights=finder._similarity_weights(),
                                 structure="quality", max_arity=2)
    assert finder.score(plan) < finder.score(binary)


def test_adaptive_layout_replays_exactly():
    """A star-containing adaptive tree replays a random circuit exactly."""
    rng = np.random.default_rng(7)
    n = 6
    # build an adaptive plan from a clustered stream, then replay a fresh one
    layout_stream = []
    for _ in range(15):
        for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            layout_stream.append((_rand_unitary(2, rng), (a, b)))
    plan = TreeLayoutFinder(layout_stream, n=n, structure="adaptive",
                            max_arity=None).run()
    assert not plan.is_binary()
    stream = _random_stream(n, 40, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=1 << n)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_nonbinary_ascii_tree_renders_arity():
    """ascii_tree draws an internal node with more than two children."""
    plan = _nonbinary_plan()
    ttn = TreeTensorNetwork.from_plan(plan)
    text = ttn.ascii_tree()
    for q in range(6):
        assert f"q{q}" in text
    # an arity-3 star centres the middle child under the parent stem ('┼')
    assert "\u253c" in text


def test_isometry_metadata_api_has_one_network_owned_orientation_map():
    """Product construction and optimizer delegates expose one live map."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    directions = ttn.isometry_map()

    assert directions[plan.root] is None
    for nid in plan.nodes():
        if nid == plan.root:
            continue
        assert directions[nid] == plan.parent[nid]
        assert ttn.can_skip_canonize(nid, plan.parent[nid])
        assert ttn.can_skip_canonize(
            plan.parent[nid], nid, absorb="left",
        )
    assert ttn.validate_isometry_metadata() is ttn
    assert ttn.validate(check_canonical=True) is ttn

    opt = TreeOptimizer(None, tree=plan, state=ttn, run=False)
    assert opt.isometry_map() == directions
    leaf = plan.leaf_of_qubit[0]
    assert opt.isometry_direction(leaf) == plan.parent[leaf]
    assert opt.can_skip_canonize(leaf, plan.parent[leaf])
    assert opt.validate_isometry_metadata() is opt


def test_isometry_metadata_validation_detects_cleared_local_proof():
    """A live canonical-region claim cannot outlast cleared ``left_inds``."""
    ttn = _entangled_ttn(seed=43)
    leaf = ttn.leaf_of_qubit(0)
    tensor = ttn.node_tensor(leaf)
    assert ttn.isometry_direction(leaf) == ttn.parent(leaf)

    # Quimb correctly clears ``left_inds`` whenever tensor data changes.
    tensor.modify(data=np.array(tensor.data))
    assert ttn.isometry_direction(leaf) is None
    with pytest.raises(ValueError, match="must be isometric"):
        ttn.validate_isometry_metadata()
    with pytest.raises(ValueError, match="must be isometric"):
        ttn.validate(check_canonical=True)

    ttn.invalidate_canonical_form()
    assert ttn.validate_isometry_metadata() is ttn


def test_shift_center_lossless_and_recanonical():
    """Shifting the centre preserves the state exactly and re-canonicalises."""
    ttn = _entangled_ttn(seed=1)
    assert ttn.orthogonality_center == ttn.root
    assert ttn.is_canonical_form()  # about the tracked centre
    sv0 = ttn.to_statevector()
    for target in (
        ttn.leaf_of_qubit(5), ttn.leaf_of_qubit(0), ttn.root,
        ttn.leaf_of_qubit(3),
    ):
        ttn.shift_orthogonality_center(target)
        assert ttn.orthogonality_center == target
        assert ttn.is_canonical_form(target)
        assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_shift_center_idempotent_touches_nothing():
    """Shifting to the current centre is a no-op that mutates no tensor."""
    ttn = _entangled_ttn(seed=2)
    snap = {nid: np.array(ttn.node_tensor(nid).data) for nid in ttn.plan.nodes()}
    ttn.shift_orthogonality_center(ttn.orthogonality_center)
    for nid in ttn.plan.nodes():
        assert np.array_equal(ttn.node_tensor(nid).data, snap[nid])


def test_dense_edge_canonization_skips_proven_isometry(monkeypatch):
    """A direct lossless edge move reuses a live dense ``left_inds`` proof."""
    ttn = _entangled_ttn(seed=40)
    leaf = ttn.leaf_of_qubit(0)
    parent = ttn.parent(leaf)
    ttn.shift_orthogonality_center(parent)
    assert ttn.orthogonality_center == parent
    calls = []
    canonize_between = ttn.canonize_between

    def traced_canonize_between(*args, **kwargs):
        calls.append((args, kwargs))
        return canonize_between(*args, **kwargs)

    monkeypatch.setattr(ttn, "canonize_between", traced_canonize_between)
    before = {
        nid: np.array(ttn.node_tensor(nid).data)
        for nid in ttn.plan.nodes()
    }

    assert ttn.can_skip_canonize(leaf, parent)
    ttn.canonize_edge_(leaf, parent)

    assert not calls
    assert all(
        np.array_equal(ttn.node_tensor(nid).data, data)
        for nid, data in before.items()
    )


def test_shift_center_from_unknown_canonicalises_once():
    """An unknown centre falls back to a full canonicalisation about the target."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=3, canonicalize=False)
    assert ttn.orthogonality_center is None
    assert not ttn.is_canonical_form()
    leaf = ttn.leaf_of_qubit(4)
    ttn.shift_orthogonality_center(leaf)
    assert ttn.orthogonality_center == leaf
    assert ttn.is_canonical_form(leaf)


def test_center_move_only_touches_geodesic():
    """A centre move is O(path length): off-geodesic tensors are untouched."""
    ttn = _entangled_ttn(seed=4)
    src = ttn.orthogonality_center
    dst = ttn.leaf_of_qubit(5)
    path = set(ttn.node_path(src, dst))
    off = [nid for nid in ttn.plan.nodes() if nid not in path]
    assert off  # the geodesic does not span the whole tree
    snap = {nid: np.array(ttn.node_tensor(nid).data) for nid in off}
    ttn.shift_orthogonality_center(dst)
    for nid in off:
        assert np.array_equal(ttn.node_tensor(nid).data, snap[nid])


def test_center_moves_use_lossless_qr_in_quimb(monkeypatch):
    """Known-centre moves explicitly select Quimb's non-truncating QR path."""
    ttn = _entangled_ttn(seed=41)
    calls = []
    canonize_between = ttn.canonize_between

    def traced_canonize_between(*args, **kwargs):
        calls.append(dict(kwargs))
        return canonize_between(*args, **kwargs)

    monkeypatch.setattr(ttn, "canonize_between", traced_canonize_between)
    ttn.shift_orthogonality_center(ttn.leaf_of_qubit(5))

    assert calls
    assert all(call["method"] == "qr" for call in calls)
    assert all(call["cutoff"] == 0.0 for call in calls)
    assert ttn.is_canonical_form()


def test_shift_center_recovers_from_multinode_region_locally():
    """A tracked canonical region is reduced by QR without touching its exterior."""
    ttn = _entangled_ttn(seed=42)
    region = {ttn.root, *ttn.children(ttn.root)}
    ttn.canonize_subtree_(region)
    assert ttn.orthogonality_center is None

    outside = [nid for nid in ttn.plan.nodes() if nid not in region]
    snapshot = {
        nid: np.array(ttn.node_tensor(nid).data)
        for nid in outside
    }
    target = sorted(region - {ttn.root})[0]

    ttn.shift_orthogonality_center(target)

    assert ttn.orthogonality_center == target
    assert ttn.is_canonical_form(target)
    for nid in outside:
        assert np.array_equal(ttn.node_tensor(nid).data, snapshot[nid])


@pytest.mark.parametrize("absorb", ["left", "right"])
def test_region_recovery_preserves_leaf_order_without_repeated_scans(monkeypatch, absorb):
    ttn = TreeTensorNetwork.from_order(range(64), structure="balanced")
    region = set(ttn.plan.nodes())
    target = ttn.plan.node_of_qubit[0]
    remaining = set(region)
    expected = []
    while len(remaining) > 1:
        node = min(n for n in remaining if n != target
                   and sum(v in remaining for v in ttn.neighbors(n)) == 1)
        neighbor = next(v for v in ttn.neighbors(node) if v in remaining)
        expected.append((node, neighbor) if absorb == "right" else (neighbor, node))
        remaining.remove(node)

    # Preserve builder-proven isometries while exercising regional recovery.
    ttn.canonical_region = region
    neighbors = ttn.neighbors
    canonize = ttn.canonize_edge_
    visits = 0
    actual = []

    def counted(node):
        nonlocal visits
        visits += 1
        return neighbors(node)

    def recorded(a, b, **kwargs):
        actual.append((a, b))
        return canonize(a, b, **kwargs)

    monkeypatch.setattr(ttn, "neighbors", counted)
    monkeypatch.setattr(ttn, "canonize_edge_", recorded)
    ttn.shift_orthogonality_center(target, absorb=absorb)
    assert actual == expected
    assert visits < 20 * len(region)
    assert ttn.orthogonality_center == target
    assert ttn.is_canonical_form(target)


def test_two_qubit_anchor_uses_nearest_endpoint(monkeypatch):
    """A non-sibling gate starts from the endpoint nearest the centre."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    opt = TreeOptimizer(None, n=8, tree=plan, run=False)
    near = plan.leaf_of_qubit[7]
    opt.shift_orthogonality_center(near)

    moves = []
    move_center = opt._move_center

    def traced_move_center(target):
        moves.append(target)
        return move_center(target)

    monkeypatch.setattr(opt, "_move_center", traced_move_center)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    opt.apply_2q(np.kron(x, x), 0, 7)

    assert moves == [near]
    assert opt.center == near


def test_shift_center_validates_node():
    """Shifting to a non-node raises loudly."""
    ttn = _entangled_ttn(seed=5)
    with pytest.raises(ValueError):
        ttn.shift_orthogonality_center(9999)


def test_canonize_edge_tracks_centre_honestly():
    """A lone edge move advances the centre by one hop or marks it unknown."""
    ttn = _entangled_ttn(seed=6)  # centre at root
    root = ttn.root
    c0, c1 = ttn.children(root)[:2]
    ttn.canonize_edge_(root, c0, absorb="right")  # centre root -> c0
    assert ttn.orthogonality_center == c0
    # an edge move not starting at the centre cannot leave a global centre
    ttn.canonize_edge_(root, c1, absorb="right")
    assert ttn.orthogonality_center is None


def test_center_survives_copy():
    """The tracked centre rides along with a network / optimizer copy."""
    opt = TreeOptimizer(None, n=6, chi=8)
    opt._move_center(opt.plan.leaf_of_qubit[4])
    ttn2 = opt.tn.copy()
    assert ttn2.orthogonality_center == opt.tn.orthogonality_center
    other = opt.copy()
    assert other.center == opt.center
    assert other.tn.orthogonality_center == opt.tn.orthogonality_center


def test_optimizer_center_is_network_view():
    """optimizer.center is a single value shared with the network; moves stay canonical."""
    rng = np.random.default_rng(11)
    n = 6
    opt = TreeOptimizer(_random_stream(n, 20, rng), n=n, chi=1 << n)  # exact
    assert opt.center == opt.tn.orthogonality_center
    for q in (0, 5, 2):
        leaf = opt.plan.leaf_of_qubit[q]
        opt._move_center(leaf)
        assert opt.center == leaf == opt.tn.orthogonality_center
        assert opt.tn.is_canonical_form(leaf)


def test_optimizer_public_canonicalisation_api():
    """TreeOptimizer exposes the same public canonicalisation surface as its state."""
    rng = np.random.default_rng(21)
    n = 6
    opt = TreeOptimizer(_random_stream(n, 18, rng), n=n, chi=1 << n)  # exact
    # name-parity alias reads the single shared centre
    assert opt.orthogonality_center == opt.center == opt.tn.orthogonality_center
    # public shift returns self and moves the shared centre, staying canonical
    leaf = opt.plan.leaf_of_qubit[4]
    assert opt.shift_orthogonality_center(leaf) is opt
    assert opt.center == leaf
    assert opt.is_canonical_form()  # about the tracked centre
    assert opt.is_canonical_form(leaf)
    # the alias setter writes straight through to the network
    opt.orthogonality_center = opt.plan.root
    assert opt.tn.orthogonality_center == opt.plan.root


def test_nonbinary_center_movement_is_canonical():
    """Centre movement is exact and canonical on a non-binary tree, incl. internal nodes."""
    plan = _nonbinary_plan()
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=7)
    sv0 = ttn.to_statevector()
    for target in (ttn.leaf_of_qubit(0), 6, 7, ttn.leaf_of_qubit(5), ttn.root):
        ttn.shift_orthogonality_center(target)
        assert ttn.orthogonality_center == target
        assert ttn.is_canonical_form(target)
        assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_subtree_canonicalisation_lossless_and_isometric():
    """Canonicalising around a connected subtree is lossless and gauges outside inward."""
    ttn = _entangled_ttn(seed=1)
    region = {ttn.root, *ttn.children(ttn.root)}
    assert len(region) > 1
    sv0 = ttn.to_statevector()
    ttn.canonize_subtree_(region)
    assert ttn.canonical_region == frozenset(region)
    # a multi-node region has no single orthogonality centre
    assert ttn.orthogonality_center is None
    assert ttn.is_subtree_canonical_form()          # tracked region
    assert ttn.is_subtree_canonical_form(region)    # explicit region
    assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_subtree_norm_concentrates_on_region():
    """After subtree canonicalisation the whole squared norm is carried by the region."""
    import quimb.tensor as qtn
    ttn = _entangled_ttn(seed=2)
    region = {ttn.root, *ttn.children(ttn.root)}
    ttn.canonize_subtree_(region)
    full = float(abs((ttn.H | ttn) ^ all))
    reg = qtn.TensorNetwork([ttn.node_tensor(n).copy() for n in region])
    assert np.isclose(float(abs((reg.H | reg) ^ all)), full)


def test_single_node_subtree_is_orthogonality_center():
    """A one-node subtree is exactly an orthogonality centre."""
    ttn = _entangled_ttn(seed=3)
    leaf = ttn.leaf_of_qubit(4)
    ttn.canonize_subtree_({leaf})
    assert ttn.canonical_region == frozenset({leaf})
    assert ttn.orthogonality_center == leaf
    assert ttn.is_canonical_form()
    assert ttn.is_subtree_canonical_form({leaf})


def test_subtree_span_and_connectivity_validation():
    """subtree_span links arbitrary nodes; a disconnected region needs span=True."""
    ttn = _entangled_ttn(seed=4)
    la, lb = ttn.leaf_of_qubit(0), ttn.leaf_of_qubit(5)
    span = ttn.subtree_span({la, lb})
    assert set(ttn.node_path(la, lb)) == span
    # a disconnected node set raises unless auto-spanned
    with pytest.raises(ValueError):
        ttn.canonize_subtree_({la, lb})
    ttn.canonize_subtree_({la, lb}, span=True)
    assert ttn.canonical_region == frozenset(span)
    assert ttn.is_subtree_canonical_form()


def test_canonize_around_qubits_range():
    """Qubit-level range canonicalisation spans the right subtree and stays canonical."""
    ttn = _entangled_ttn(seed=5)
    sv0 = ttn.to_statevector()
    ttn.canonize_around_qubits_([1, 2, 3])
    leaves = [ttn.leaf_of_qubit(q) for q in (1, 2, 3)]
    assert ttn.canonical_region == frozenset(ttn.subtree_span(leaves))
    assert ttn.is_subtree_canonical_form()
    assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10
    # a single qubit collapses to a one-leaf orthogonality centre
    ttn.canonize_around_qubits_([2])
    assert ttn.orthogonality_center == ttn.leaf_of_qubit(2)


def test_subtree_region_survives_copy():
    """A multi-node canonical region rides along with a copy."""
    ttn = _entangled_ttn(seed=6)
    region = {ttn.root, *ttn.children(ttn.root)}
    ttn.canonize_subtree_(region)
    clone = ttn.copy()
    assert clone.canonical_region == ttn.canonical_region
    assert clone.orthogonality_center is None
    assert clone.is_subtree_canonical_form()


def test_nonbinary_subtree_canonicalisation():
    """Subtree canonicalisation works around an internal star node on a non-binary tree."""
    plan = _nonbinary_plan()
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=7)
    sv0 = ttn.to_statevector()
    region = {8, 6, 7}  # root plus both arity-3 star nodes
    ttn.canonize_subtree_(region)
    assert ttn.canonical_region == frozenset(region)
    assert ttn.is_subtree_canonical_form()
    assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_optimizer_subtree_canonicalisation_api():
    """TreeOptimizer mirrors the state's public subtree-canonicalisation surface."""
    rng = np.random.default_rng(31)
    n = 6
    opt = TreeOptimizer(_random_stream(n, 16, rng), n=n, chi=1 << n)  # exact
    region = {opt.plan.root, *opt.plan.children[opt.plan.root]}
    # public canonize_subtree returns self and installs the shared region view
    assert opt.canonize_subtree(region) is opt
    assert opt.canonical_region == opt.tn.canonical_region == frozenset(region)
    assert opt.is_subtree_canonical_form()
    # qubit-level range entry point
    assert opt.canonize_around_qubits([0, 5]) is opt
    leaves = [opt.plan.leaf_of_qubit[q] for q in (0, 5)]
    assert opt.canonical_region == frozenset(opt.tn.subtree_span(leaves))
    assert opt.is_subtree_canonical_form()
    # the region setter writes straight through to the network
    opt.canonical_region = {opt.plan.root}
    assert opt.tn.canonical_region == frozenset({opt.plan.root})
    assert opt.orthogonality_center == opt.plan.root


def test_live_bond_dimensions_survive_general_threading():
    """Tree edge diagnostics use live bonds after a non-sibling gate."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(None, n=8, tree=plan, chi=8)
    opt.apply_gate(h, 0)
    opt.apply_gate(cnot, (0, 6))

    for node in plan.nodes():
        for child in plan.children[node]:
            ix = opt.tn.bond(node, child)
            assert ix in opt.tn.ind_map
            assert opt.tn._bond_dim(node, child) == opt.tn.ind_size(ix)


def test_nonunitary_one_qubit_gate_recenters_state():
    """A non-unitary one-qubit gate cannot leave a stale canonical centre."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    projector = np.diag([1.0, 0.0]).astype(complex)
    opt = TreeOptimizer([(h, 0), (cnot, (0, 7))], n=8, tree=plan)
    opt.apply_gate(projector, 7)

    assert opt.is_canonical_form()
    assert np.isclose(opt.norm(), np.linalg.norm(opt.to_dense()))


def test_forced_measurement_validates_outcome_probability():
    """Invalid or impossible forced outcomes raise before collapsing the state."""
    opt = TreeOptimizer(None, n=2)
    with pytest.raises(ValueError, match="outcome must be 0 or 1"):
        opt.measure(0, outcome=2)
    with pytest.raises(ValueError, match="zero probability"):
        opt.measure(0, outcome=1)
    assert np.isclose(opt.norm(), 1.0)


def test_multinode_region_normalization_preserves_canonicality():
    """Normalization scales inside a multi-node canonical region."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    opt = TreeOptimizer(None, n=8, tree=plan)
    opt.tn = TreeTensorNetwork.rand(plan, D=2, seed=31)
    leaf = plan.leaf_of_qubit[0]
    region = {leaf, plan.parent[leaf]}
    opt.tn.canonize_subtree_(region)
    opt.tn.node_tensor(leaf).modify(data=3.0 * opt.tn.node_tensor(leaf).data)

    opt.normalize()

    assert np.isclose(opt.norm(), 1.0)
    assert opt.is_subtree_canonical_form()


def test_shift_center_supports_left_absorption_orientation():
    """The optional left-absorption orientation still centres the target."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=32)
    leaf = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(leaf, absorb="left")
    assert ttn.orthogonality_center == leaf
    assert ttn.is_canonical_form()


def test_tree_plan_rejects_malformed_orders():
    """TreePlan.from_order enforces the same qubit-label contract as from_children."""
    with pytest.raises(ValueError, match="at least one"):
        TreePlan.from_order([])
    with pytest.raises(ValueError, match="permutation"):
        TreePlan.from_order([0, 0])
    with pytest.raises(ValueError, match="structure"):
        TreePlan.from_order(range(2), structure="unknown")


def test_tree_gate_queue_set_and_add():
    """TreeOptimizer exposes queue replacement and extension like MpsOptimizer."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(None, n=2)
    assert opt.set_gates([(x, 0)]) is opt
    assert opt.add_gates([(x, 1)]) is opt
    assert len(opt.G) == 2
    opt.run()
    assert _fidelity(opt.to_dense(), np.array([0.0, 0.0, 0.0, 1.0])) > 1 - 1e-12


def test_optimizer_accepts_and_copies_initial_ttn():
    """TreeOptimizer can evolve an arbitrary supplied tree state independently."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=33)
    before = state.to_statevector()
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    opt = TreeOptimizer([(x, 0)], n=6, tn=state, chi=8)

    expected = _sv_apply_1q(before, x, 0, 6)
    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10
    assert _fidelity(before, state.to_statevector()) > 1 - 1e-12
    assert opt.set_tn(state) is opt
