"""Tree layout regression tests."""


import inspect
import sys
import types
import numpy as np
import pytest
import pepsy
from pepsy.optimizers.tree import TreeLayoutFinder
from pepsy.optimizers.tree import TreeOptimizer
from pepsy.optimizers.tree import TreePlan
from pepsy.optimizers.tree import TreeTensorNetwork


from _tree_test_helpers import (
    _exact_state,
    _fidelity,
    _rand_unitary,
    _random_stream,
    _two_branch_flip_submpo,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.tree]


def test_user_supplied_plan_runs():
    """A caller-provided TreePlan is honoured."""
    rng = np.random.default_rng(7)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    assert isinstance(plan, TreePlan)
    stream = _random_stream(n, 20, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_root_physical_qubit_is_first_class_tree_site():
    """A binary top tensor can own one physical qubit alongside two bonds."""
    plan = TreePlan.from_order(
        range(4), structure="balanced", root_qubit=4,
    )
    state = TreeTensorNetwork.from_plan(plan)
    root = state.node_tensor(plan.root)

    assert plan.n == 5
    assert plan.root_qubit == 4
    assert plan.node_of_qubit[4] == plan.root
    assert 4 not in plan.leaf_of_qubit
    assert set(root.inds) == {
        state.site_ind(4),
        *(state.bond(plan.root, child) for child in plan.children[plan.root]),
    }
    assert root.ndim == 3
    assert set(state.outer_inds()) == {
        state.site_ind(q) for q in range(plan.n)
    }
    expected = np.zeros(2**plan.n)
    expected[0] = 1.0
    assert np.array_equal(state.to_statevector(), expected)
    assert state.validate(check_canonical=True) is state


@pytest.mark.parametrize("root_qubit", [0, 1])
def test_two_qubit_tree_uses_distinct_root_and_leaf_sites(root_qubit):
    """The smallest root-site tree uses a unary root over one physical leaf."""
    leaf_qubit = 1 - root_qubit
    plan = TreePlan.from_order(
        [leaf_qubit], structure="balanced", root_qubit=root_qubit,
    )
    layered = TreePlan.build_layered(
        [leaf_qubit], block_size=2, root_qubit=root_qubit,
    )
    found = TreeLayoutFinder(
        [], n=2, root_qubit=root_qubit, max_arity=2,
    ).run()
    automatic = TreeOptimizer(
        None, n=2, root_qubit=root_qubit, max_arity=2, run=False,
    )

    for candidate in (plan, layered, found):
        assert candidate.n == 2
        assert candidate.root_qubit == root_qubit
        assert candidate.node_of_qubit[root_qubit] == candidate.root
        assert candidate.node_of_qubit[leaf_qubit] != candidate.root
        assert len(candidate.children[candidate.root]) == 1
    assert automatic.plan.node_of_qubit[root_qubit] == automatic.plan.root
    assert automatic.tn.validate(check_canonical=True) is automatic.tn

    stream = [
        (pepsy.h(), root_qubit),
        (pepsy.cnot(), (root_qubit, leaf_qubit)),
    ]
    opt = TreeOptimizer(stream, tree=plan, chi=8)
    assert _fidelity(_exact_state(stream, 2), opt.to_dense()) > 1 - 1e-12
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_root_physical_qubit_gate_and_submpo_replay_are_exact():
    """Direct gates and a structured sub-MPO can target the top physical leg."""
    plan = TreePlan.from_order(
        range(4), structure="balanced", root_qubit=4,
    )
    direct_stream = [(pepsy.h(), 0), (pepsy.cnot(), (0, 4))]
    direct = TreeOptimizer(
        direct_stream, tree=plan, chi=32, cutoff=0.0,
    )
    assert _fidelity(
        _exact_state(direct_stream, plan.n), direct.to_dense()
    ) > 1 - 1e-10
    assert direct.tn.validate(check_canonical=True) is direct.tn

    where = (1, 3, 4)
    mpo = _two_branch_flip_submpo(
        L=plan.n,
        sites=where,
        targets=where,
        w0=0.0,
        w1=1.0,
    )
    submpo = TreeOptimizer(
        None, tree=plan, chi=32, cutoff=0.0, run=False,
    )
    submpo.apply_submpo(mpo, where)
    expected = np.zeros(2**plan.n, dtype=complex)
    expected[int("01011", 2)] = 1.0
    assert np.allclose(submpo.to_dense(), expected)
    assert submpo.tn.validate(check_canonical=True) is submpo.tn


def test_root_physical_qubit_layout_and_cap_are_root_aware():
    """Layout scoring reaches the root site and capping removes only its leg."""
    finder = TreeLayoutFinder(
        supports=[(0, 4), (0, 4), (1, 2)],
        n=5,
        root_qubit=4,
        structure="balanced",
        max_arity=2,
    )
    plan = finder.run(refine="greedy", refine_budget=16)
    root_path = plan.node_path(plan.node_of_qubit[0], plan.root)
    loads = finder.edge_loads(plan)
    path_edges = {
        (u, v) if plan.parent.get(v) == u else (v, u)
        for u, v in zip(root_path, root_path[1:])
    }

    assert plan.root_qubit == 4
    assert plan.node_of_qubit[4] == plan.root
    assert all(loads[edge] > 0.0 for edge in path_edges)
    assert finder.report(plan)["root_qubit"] == 4

    automatic = TreeOptimizer(
        None, root_qubit=4, max_arity=2, chi=16, run=False,
    )
    assert automatic.n == 5
    assert automatic.plan.root_qubit == 4

    opt = TreeOptimizer(None, tree=plan, chi=16, run=False)
    assert opt.layout_report()["root_qubit"] == 4
    opt.apply_1q(pepsy.h(), 4)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert opt.tn.local_expectation(x, 4) == pytest.approx(1.0)
    opt.cap(4, [1.0, 0.0])
    assert opt.plan.root_qubit is None
    assert opt.n == 4
    assert opt.to_dense().shape == (2**4,)
    assert opt.norm() == pytest.approx(1 / np.sqrt(2))
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_explicit_tree_rejects_mismatched_n():
    """Explicit plans enforce the same qubit-count invariant as finders."""
    plan = TreePlan.from_order(
        range(4), structure="balanced", root_qubit=4,
    )
    with pytest.raises(
        ValueError, match=r"tree contains 5 qubits, but n=4"
    ):
        TreeOptimizer(None, n=4, tree=plan, run=False)
    with pytest.raises(
        ValueError, match=r"tree contains 5 qubits, but n=4"
    ):
        TreeOptimizer(None, n=4, layout=plan, run=False)


def test_layout_finder_builds_strict_binary_tree_when_requested():
    """Explicit top_arity=2 opts out of the ternary virtual root."""
    rng = np.random.default_rng(8)
    n = 8
    stream = _random_stream(n, 60, rng)
    plan = TreeLayoutFinder(stream, n=n, max_arity=2, top_arity=2).run()
    assert plan.n == n
    assert set(plan.leaf_of_qubit) == set(range(n))
    # every internal node has exactly two children
    for nid in plan.nodes():
        assert len(plan.children[nid]) in (0, 2)
    # tree distances are well-defined for all pairs
    for a in range(n):
        for b in range(a + 1, n):
            assert plan.tree_distance(a, b) >= 1


def test_tree_layout_accepts_explicit_fixed_site_order():
    """An explicit order builds the requested binary/ternary-root tree."""
    order = pepsy.square_lattice_zigzag(2, 2)
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 1)), (pepsy.cnot(), (2, 3))],
        n=4,
        max_arity=2,
        top_arity=3,
    )
    plan = finder.run(order=order)

    assert tuple(plan.qubit_of_leaf.values()) == order
    assert plan.top_arity == 3
    assert plan.is_binary()


@pytest.mark.parametrize(
    "mode",
    [
        "row-major",
        "col-major",
        "snake",
        "snake-row-major",
        "folded-snake",
        "folded-snake-row-major",
        "hilbert",
        "hilbert-row-major",
    ],
)
def test_tree_layout_finder_supports_onedmap_lattice_presets(mode):
    """Named Tree presets preserve the corresponding OneDMap leaf traversal."""
    Lx = Ly = 4
    finder = TreeLayoutFinder(
        [],
        n=Lx * Ly,
        structure="quality",
        max_arity=2,
        top_arity=2,
        lattice_shape=(Lx, Ly),
    )
    one_d_to_lattice, _ = pepsy.OneDMap.build(Lx, Ly, mode=mode)
    expected = tuple(x * Ly + y for x, y in one_d_to_lattice.values())

    plan = finder.run(order=mode)

    assert plan.mpo_order() == expected


def test_tree_layout_aliases_match_onedmap_hilbert_orientation():
    """Tree aliases preserve both rectangular generalized-Hilbert orientations."""
    finder = TreeLayoutFinder(
        [],
        n=15,
        max_arity=2,
        top_arity=2,
        lattice_shape=(3, 5),
    )
    expected = {
        mode: tuple(x * 5 + y for x, y in pepsy.OneDMap.build(
            3, 5, mode="hilbert-row-major" if "row" in mode else "hilbert",
        )[0].values())
        for mode in ("hilbert-row", "hilbert-row-major", "hilbert")
    }

    assert finder.run(order="hilbert-row").mpo_order() == expected["hilbert-row"]
    assert finder.run(order="hilbert-row-major").mpo_order() == expected[
        "hilbert-row-major"
    ]
    assert finder.run(order="hilbert").mpo_order() == expected["hilbert"]
    assert expected["hilbert-row"] != expected["hilbert"]


@pytest.mark.parametrize("mode", [
    "row-major",
    "col-major",
    "snake",
    "snake-row-major",
    "folded-snake",
    "folded-snake-row-major",
    "hilbert",
    "hilbert-row-major",
])
@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (1, 7), (2, 5)])
def test_tree_geometric_order_is_preserved_at_every_hierarchy_layer(
    mode, shape,
):
    """Geometric trees coarsen adjacent OneDMap intervals at every level."""
    Lx, Ly = shape
    n = Lx * Ly
    finder = TreeLayoutFinder(
        [],
        n=n,
        max_arity=2,
        top_arity=3,
        lattice_shape=shape,
    )
    expected = tuple(
        x * Ly + y
        for x, y in pepsy.OneDMap.build(Lx, Ly, mode=mode)[0].values()
    )
    plan = finder.run(order=mode)

    assert plan.mpo_order() == expected
    masks = plan.subtree_qubit_masks()
    positions = {qubit: i for i, qubit in enumerate(expected)}

    # Every internal node must partition one contiguous path interval into
    # consecutive child intervals. This includes the top layer, rather than
    # checking only the leaf MPO order.
    for node, children in plan.children.items():
        if len(children) < 2:
            continue
        intervals = []
        for child in children:
            child_positions = sorted(
                positions[q]
                for q in range(n)
                if masks[child] & (1 << q)
            )
            intervals.append((child_positions[0], child_positions[-1]))
            assert child_positions == list(
                range(child_positions[0], child_positions[-1] + 1)
            )
        assert intervals[0][0] == min(interval[0] for interval in intervals)
        for previous, current in zip(intervals, intervals[1:]):
            assert previous[1] + 1 == current[0]

    # These cases are large enough to exercise the root plus multiple middle
    # layers; the top children must themselves be non-leaf subtrees.
    assert len(plan.children[plan.root]) == 3
    assert all(plan.children[child] for child in plan.children[plan.root])


def test_tree_lattice_order_helper_and_missing_shape_error():
    """Tree geometric orders are reusable and require an explicit shape."""
    expected = (0, 2, 4, 1, 3, 5)
    assert TreeLayoutFinder.lattice_order(
        2, 3, "row-major", site=lambda x, y: y * 2 + x,
    ) == expected

    finder = TreeLayoutFinder([], n=4)
    with pytest.raises(ValueError, match="lattice_shape"):
        finder.run(order="hilbert")


def test_tree_alternating_lattice_orders_and_coarse_blocks():
    """Coarse presets traverse complete blocks before moving to the next."""
    shape = (4, 3)
    row_alternate = TreeLayoutFinder.lattice_order(
        *shape, "alternate-x"
    )
    col_alternate = TreeLayoutFinder.lattice_order(
        *shape, "alternate-y"
    )
    assert row_alternate == (0, 3, 6, 9, 10, 7, 4, 1, 2, 5, 8, 11)
    assert col_alternate == (0, 1, 2, 5, 4, 3, 6, 7, 8, 11, 10, 9)

    coarse_row = TreeLayoutFinder.lattice_order(
        *shape, "coarse-row-major", grain=(2, 1)
    )
    assert coarse_row == (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11)

    coarse_y = TreeLayoutFinder.lattice_order(
        *shape, "coarse-alternate-y", grain=(1, 2)
    )
    assert coarse_y == col_alternate

    for mode in (
        "coarse-row-major",
        "coarse-col-major",
        "coarse-snake",
        "coarse-snake-row-major",
        "coarse-folded-snake",
        "coarse-folded-snake-row-major",
        "coarse-hilbert",
        "coarse-hilbert-row-major",
    ):
        order = TreeLayoutFinder.lattice_order(
            5, 4, mode, grain=(2, 2)
        )
        assert len(order) == 20
        assert set(order) == set(range(20))

        coords = [(q // 4, q % 4) for q in order]
        blocks = [(x // 2, y // 2) for x, y in coords]
        block_runs = []
        start = 0
        while start < len(blocks):
            block = blocks[start]
            end = start + 1
            while end < len(blocks) and blocks[end] == block:
                end += 1
            block_runs.append(block)
            assert all(current == block for current in blocks[start:end])
            start = end
        assert len(block_runs) == 6
        assert len(set(block_runs)) == 6


def test_tree_coarse_order_handoff_and_validation():
    """TreeOptimizer and finder diagnostics retain coarse-layout settings."""
    finder = TreeLayoutFinder(
        [],
        n=12,
        max_arity=2,
        top_arity=2,
        lattice_shape=(4, 3),
        order="coarse-alternate-x",
        coarse_grain=(2, 1),
    )
    plan = finder.run()
    assert plan.mpo_order() == (0, 3, 6, 9, 10, 7, 4, 1, 2, 5, 8, 11)
    assert finder.report(plan)["coarse_grain"] == (2, 1)

    forwarded = TreeOptimizer.find_tree_layout(
        [],
        n=12,
        max_arity=2,
        top_arity=2,
        lattice_shape=(4, 3),
        order="coarse-row-major",
        coarse_grain=2,
    )
    assert forwarded.mpo_order() == (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11)

    with pytest.raises(ValueError, match="coarse_grain"):
        TreeLayoutFinder.lattice_order(
            4, 3, "coarse-snake", grain=(0, 1)
        )


def test_tree_3d_lattice_orders_support_axis_alternation():
    """Tree presets preserve the shared 3D OneDMap traversal vocabulary."""
    shape = (3, 2, 2)
    size = np.prod(shape)
    coords = {
        q: (q // (shape[1] * shape[2]),
            (q // shape[2]) % shape[1],
            q % shape[2])
        for q in range(size)
    }
    for mode in (
        "row-major",
        "col-major",
        "snake",
        "snake-row-major",
        "alternate-x",
        "alternate-y",
        "alternate-z",
    ):
        expected = tuple(
            x * shape[1] * shape[2] + y * shape[2] + z
            for x, y, z in pepsy.OneDMap.build(
                *shape[:2], Lz=shape[2], mode=mode,
            )[0].values()
        )
        order = TreeLayoutFinder.lattice_order(*shape, mode=mode)
        assert order == expected
        assert set(order) == set(range(size))

    # Each non-coarse alternating path is a nearest-neighbor 3D traversal.
    for mode in ("alternate-x", "alternate-y", "alternate-z"):
        order = TreeLayoutFinder.lattice_order(*shape, mode=mode)
        assert all(
            sum(abs(coords[left][axis] - coords[right][axis]) for axis in range(3))
            == 1
            for left, right in zip(order[:-1], order[1:])
        )


@pytest.mark.parametrize(
    ("mode", "grain"),
    [
        ("coarse-alternate-x", (2, 1, 1)),
        ("coarse-alternate-y", (1, 2, 1)),
        ("coarse-alternate-z", (1, 1, 2)),
    ],
)
def test_tree_3d_coarse_alternating_axes_keep_blocks_and_paths(mode, grain):
    """3D coarse alternation keeps each block contiguous and path-connected."""
    shape = (4, 3, 3)
    size = int(np.prod(shape))
    order = TreeLayoutFinder.lattice_order(*shape, mode=mode, grain=grain)
    assert len(order) == size
    assert set(order) == set(range(size))

    coords = {
        q: (q // (shape[1] * shape[2]),
            (q // shape[2]) % shape[1],
            q % shape[2])
        for q in range(size)
    }
    block_ids = [
        tuple(coords[q][axis] // grain[axis] for axis in range(3))
        for q in order
    ]
    runs = []
    start = 0
    while start < len(block_ids):
        block = block_ids[start]
        end = start + 1
        while end < len(block_ids) and block_ids[end] == block:
            end += 1
        assert all(block_id == block for block_id in block_ids[start:end])
        runs.append(block)
        start = end
    expected_blocks = int(np.prod([
        (length + block - 1) // block
        for length, block in zip(shape, grain)
    ]))
    assert len(runs) == len(set(runs)) == expected_blocks
    assert all(
        sum(abs(coords[left][axis] - coords[right][axis]) for axis in range(3))
        == 1
        for left, right in zip(order[:-1], order[1:])
    )


def test_tree_3d_coarse_order_handoff_and_site_mapper():
    """Finder and TreeOptimizer retain 3D coarse-layout metadata."""
    shape = (3, 2, 2)
    finder = TreeLayoutFinder(
        [],
        n=int(np.prod(shape)),
        max_arity=2,
        top_arity=2,
        lattice_shape=shape,
        order="coarse-alternate-z",
        coarse_grain=(1, 1, 2),
    )
    plan = finder.run()
    expected = tuple(
        x * shape[1] * shape[2] + y * shape[2] + z
        for x, y, z in pepsy.OneDMap.build(
            *shape[:2], Lz=shape[2], mode="alternate-z",
        )[0].values()
    )
    assert plan.mpo_order() == expected
    report = finder.report(plan)
    assert report["lattice_shape"] == shape
    assert report["coarse_grain"] == (1, 1, 2)

    custom = TreeLayoutFinder.lattice_order(
        3, 2, 2, "alternate-z",
        site=lambda x, y, z: z * 6 + y * 3 + x,
    )
    assert set(custom) == set(range(12))

    forwarded = TreeOptimizer.find_tree_layout(
        [],
        n=12,
        max_arity=2,
        top_arity=2,
        lattice_shape=shape,
        order="coarse-alternate-x",
        coarse_grain=2,
    )
    assert set(forwarded.mpo_order()) == set(range(12))


def test_quality_layout_not_worse_than_balanced():
    """Entanglement-adapted structure scores no worse than balanced order."""
    rng = np.random.default_rng(9)
    n = 8
    # locally clustered interactions: quality bisection should exploit them
    stream = []
    for _ in range(80):
        a = int(rng.integers(n - 1))
        b = a + 1 if rng.random() < 0.85 else int(rng.integers(n))
        if a == b:
            b = (b + 1) % n
        stream.append((_rand_unitary(2, rng), (a, b)))
    finder = TreeLayoutFinder(stream, n=n, structure="quality")
    quality = finder.run()
    balanced = TreePlan.from_order(range(n), structure="balanced")
    assert finder.score(quality) <= finder.score(balanced)


def test_congestion_layout_uses_operator_schmidt_edge_load():
    """The load-aware diagnostic predicts the product of crossed ranks."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    rng = np.random.default_rng(101)
    generic = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    finder = TreeLayoutFinder(
        [(cnot, (0, 3)), (generic, (0, 3))],
        n=4,
        objective="congestion",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    loads = finder.edge_loads(plan)
    path = plan.node_path(plan.leaf_of_qubit[0], plan.leaf_of_qubit[3])
    path_edges = {
        (u, v) if plan.parent.get(v) == u else (v, u)
        for u, v in zip(path, path[1:])
    }

    assert path_edges
    assert all(loads[edge] == pytest.approx(3.0) for edge in path_edges)
    assert max(loads.values()) == pytest.approx(3.0)
    report = finder.report(plan)
    assert report["objective"] == "congestion"
    assert report["peak_bond_growth"] == pytest.approx(8.0)


def test_compression_layout_reports_rank_bounds_and_tensor_cost():
    """Compression selection is explicit and honest for wide operators."""
    gate = np.eye(8, dtype=complex)
    finder = TreeLayoutFinder(
        [(gate, (0, 1, 2))],
        n=3,
        objective="compression",
        max_arity=(2, 3),
        max_operator_qubits=2,
        chi=2,
    )
    plan = finder.run()
    report = finder.report(plan)

    assert report["objective"] == "compression"
    assert report["rank_bounded_events"] > 0
    assert report["rank_bound_reasons"]["max_operator_qubits"] > 0
    assert report["estimated_max_tensor_log2"] >= 0.0
    assert len(report["objective_key"]) >= 5


def test_hypergraph_layout_scores_full_multisite_supports():
    """Direct mode ranks original hyperedges on every crossed tree cut."""
    rng = np.random.default_rng(104)
    gate = _rand_unitary(3, rng)
    supports = ((0, 1, 2), (2, 3, 4))
    finder = TreeLayoutFinder(
        [(gate, supports[0]), (gate, supports[1])],
        n=5,
        max_arity=2,
        objective="hypergraph",
    )
    plan = TreePlan.from_order(range(5), structure="balanced", max_arity=2)

    loads = finder.edge_loads(plan)
    below = plan.subtree_qubit_masks()
    expected = {edge: 0.0 for edge in loads}
    for payload, support in zip(finder.payloads, finder.supports):
        support_mask = sum(1 << q for q in support)
        for edge in expected:
            _parent, child = edge
            left_mask = support_mask & below[child]
            if not left_mask or left_mask == support_mask:
                continue
            left = tuple(q for q in support if left_mask & (1 << q))
            expected[edge] += np.log2(finder._schmidt_rank(payload, support, left))

    assert loads == pytest.approx(expected)
    report = finder.report(plan)
    assert report["objective"] == "hypergraph"
    assert report["hypergraph_score"] == {
        "max_edge_load": max(loads.values()),
        "total_edge_load": sum(loads.values()),
    }

    recommendation = finder.recommend_arities((2,), chi=None)
    assert recommendation["refine"] == "greedy"
    assert recommendation["topology_refine"] == "nni"
    assert recommendation["candidates"][0]["planning"][
        "topology_refinement"
    ]["method"] == "nni"


def test_tree_compression_layout_pilot_is_non_mutating():
    """Tree pilot selection compares copied product states only."""
    gates = [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))]
    opt = TreeOptimizer(gates, n=4, chi=2, run=False)
    original_plan = opt.plan

    selected = opt.select_layout_for_compression(
        pilot_candidates=1,
        pilot_steps=1,
    )

    assert selected["selected_candidate"]
    assert selected["pilot"]["reports"]
    assert any(
        name.startswith("quality:")
        for name in selected["pilot"]["pilot_candidates"]
    )
    assert opt.plan is original_plan
    assert opt.max_bond() == 1


def test_tree_layout_pilot_feedback_is_iterative_and_edge_aware():
    """Full-tree pilots feed bounded hot-edge proposals into the next round."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    gates = [(h, 0), (pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))]
    opt = TreeOptimizer(gates, n=4, max_arity=2, chi=1, run=False)
    original_plan = opt.plan

    selected = opt.optimize_layout(
        objective="full_tree",
        pilot_candidates=1,
        pilot_steps=3,
        rounds=2,
        topology_budget=1,
        refine_budget=1,
        search_budget=2,
    )

    assert selected["pilot"]["objective"] == "full_tree"
    assert selected["pilot"]["n_rounds"] == 2
    assert len(selected["pilot"]["rounds"]) == 2
    assert opt.plan is original_plan
    report = selected["pilot"]["reports"][selected["selected_candidate"]]
    assert report["status"] == "ok"
    assert report["update_runtime_seconds"] >= 0.0
    assert isinstance(report["edge_diagnostics"], dict)
    assert opt.max_bond() == 1


def test_tree_layout_pilots_can_run_in_parallel():
    """Independent layout pilots preserve the normal report contract."""
    gates = [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))]
    opt = TreeOptimizer(gates, n=4, chi=1, run=False)

    selected = opt.optimize_layout(
        objective="full_tree",
        pilot_candidates=2,
        pilot_workers=2,
        pilot_steps=2,
        rounds=1,
        topology_budget=1,
        refine_budget=1,
        search_budget=2,
    )

    assert selected["pilot"]["selected_candidate"]
    assert len(selected["pilot"]["reports"]) == 2
    assert all(
        report["status"] == "ok"
        for report in selected["pilot"]["reports"].values()
    )


def test_tree_layout_targeted_candidates_are_static_and_bounded():
    """Hot-edge proposal generation never allocates or mutates a TTN."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))],
        n=4,
        max_arity=2,
        objective="full_tree",
        chi=None,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", max_arity=2)
    edge = next(
        (parent, child)
        for parent, children in plan.children.items()
        for child in children
    )
    proposals = finder.targeted_candidates(
        plan,
        {edge: {"truncated": 1, "discarded_fraction": 0.5}},
        budget=3,
        seed=3,
    )

    assert len(proposals) <= 3
    assert all(candidate.is_binary() for candidate in proposals)
    unchanged = TreePlan.from_order(range(4), structure="balanced", max_arity=2)
    assert plan.children == unchanged.children
    assert plan.qubit_of_leaf == unchanged.qubit_of_leaf


def test_tree_candidate_plans_include_quality_for_state_aware_pilots():
    """Quality refinement is exposed as an explicit pilot candidate."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))],
        n=4,
        max_arity=2,
        objective="compression",
    )

    candidates = finder.candidate_plans(
        chi=2,
        include_quality=True,
        quality_refine_budget=2,
        quality_topology_budget=2,
    )

    quality = candidates["quality:arity=2"]
    assert quality["planning"]["topology_refinement"]["method"] == "nni"
    assert quality["planning"]["refinement"]["method"] == "greedy"
    assert quality["plan"].is_binary()
    assert not any(
        name.startswith("quality:")
        for name in finder.candidate_plans(chi=2)
    )


def test_full_tree_profile_reports_dynamic_cost_at_all_scales():
    """Whole-tree mode exposes width, work, demand, and scale diagnostics."""
    gates = [
        (pepsy.cnot(), (0, 1)),
        (pepsy.cnot(), (1, 2)),
        (pepsy.cnot(), (3, 4)),
        (pepsy.cnot(), (2, 4)),
    ]
    finder = TreeLayoutFinder(
        gates, n=5, max_arity=2, objective="full_tree", chi=4,
    )
    plan = TreePlan.from_order(range(5), structure="balanced", max_arity=2)
    profile = finder.full_tree_profile(plan)

    assert profile["event_count"] == len(gates)
    assert profile["peak_tensor_log2"] >= 1.0
    assert profile["peak_work_log2"] >= profile["peak_tensor_log2"]
    assert profile["total_route_length"] > 0
    assert profile["scales"]
    assert all(
        {
            "node_count",
            "edge_count",
            "peak_tensor_log2",
            "peak_edge_demand_log2",
        } <= set(scale_info)
        for scale_info in profile["scales"].values()
    )

    report = finder.report(plan)
    assert report["objective"] == "full_tree"
    assert report["full_tree"] == profile
    assert len(report["objective_key"]) == 10
    assert report["objective_key"][:4] == (
        profile["peak_overflow_log2"],
        profile["total_overflow_log2"],
        profile["peak_edge_demand_log2"],
        profile["total_edge_demand_log2"],
    )


def test_full_tree_6x6_pbc_calibrates_against_actual_replay():
    """All-scale overflow ranking tracks real capped-tree replay pressure."""
    Lx = Ly = 6
    n = Lx * Ly

    def site(x, y):
        return x * Ly + y

    edges = []
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy in ((1, 0), (0, 1)):
                edge = tuple(sorted((
                    site(x, y),
                    site((x + dx) % Lx, (y + dy) % Ly),
                )))
                if edge[0] != edge[1] and edge not in edges:
                    edges.append(edge)

    gates = (
        [(pepsy.h(), q) for q in range(n)]
        + [(pepsy.cphase(np.pi / 4), edge) for edge in edges]
    )
    finder = TreeLayoutFinder(
        gates,
        n=n,
        max_arity=(2, 3, 4),
        top_arity=3,
        objective="full_tree",
        chi=4,
        seed=11,
    )
    recommendation = finder.recommend_arities(
        (2, 3, 4),
        refine=None,
        topology_refine=None,
        search=None,
    )

    calibrated = []
    for candidate in recommendation["candidates"]:
        optimizer = TreeOptimizer(
            gates,
            n=n,
            tree=candidate["plan"],
            chi=4,
            cutoff=0.0,
            track_truncation=False,
            run=False,
        )
        optimizer.run()
        history = optimizer.truncation_history
        calibrated.append({
            "arity": candidate["max_arity"],
            "predicted_total_overflow": candidate["full_tree_profile"][
                "total_overflow_log2"
            ],
            "actual_total_excess": sum(
                max(0, event["before_bond"] - 4) for event in history
            ),
            "actual_truncations": sum(
                bool(event["truncated"]) for event in history
            ),
        })

    by_arity = {item["arity"]: item for item in calibrated}
    assert recommendation["recommended_max_arity"] == 4
    assert (
        by_arity[4]["predicted_total_overflow"]
        < by_arity[3]["predicted_total_overflow"]
        < by_arity[2]["predicted_total_overflow"]
    )
    assert (
        by_arity[4]["actual_total_excess"]
        < by_arity[3]["actual_total_excess"]
        < by_arity[2]["actual_total_excess"]
    )
    assert (
        by_arity[4]["actual_truncations"]
        < by_arity[3]["actual_truncations"]
        < by_arity[2]["actual_truncations"]
    )


def test_full_tree_anneals_subtrees_without_changing_binary_contract():
    """All-scale subtree search preserves the requested binary tree shape."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1)),
         (pepsy.cnot(), (2, 5)), (pepsy.cnot(), (4, 5))],
        n=6,
        max_arity=2,
        top_arity=3,
        objective="full_tree",
        chi=4,
        seed=7,
    )
    recommendation = finder.recommend_arities(
        (2,),
        topology_budget=3,
        search_budget=4,
        refine=None,
    )
    candidate = recommendation["candidates"][0]
    planning = candidate["planning"]
    plan = recommendation["plan"]

    assert plan.top_arity == 3
    assert plan.is_binary()
    assert planning["topology_refinement"]["method"] == "subtree"
    assert planning["search"]["method"] == "subtree"
    assert planning["search"]["search"] == "anneal"
    assert candidate["full_tree_profile"]["scales"]


def test_full_tree_hybrid_quality_search_is_static_and_budgeted():
    """Full-tree quality combines topology and leaf search without a TTN."""
    pytest.importorskip("nevergrad")
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1)),
         (pepsy.cnot(), (2, 5)), (pepsy.cnot(), (4, 5))],
        n=6,
        max_arity=(2,),
        top_arity=3,
        objective="full_tree",
        seed=7,
    )
    plan = finder.run(
        order="quality",
        topology_budget=2,
        refine_budget=2,
        search_budget=6,
    )

    planning = finder._last_arity_recommendation["candidates"][0]["planning"]
    search = planning["search"]
    assert finder.chi is None
    assert plan.is_binary()
    assert search["method"] == "hybrid"
    assert search["anneal"]["search"] == "anneal"
    assert search["nevergrad"]["method"] == "nevergrad"
    assert search["evaluations"] <= 6


def test_tree_edge_loads_match_full_edge_reference():
    """Steiner-only edge scanning preserves the full congestion calculation."""
    rng = np.random.default_rng(109)
    n = 9
    stream = _random_stream(n, 25, rng, two_qubit_frac=0.8)
    finder = TreeLayoutFinder(stream, n=n, objective="congestion")
    plan = TreePlan.from_order(range(n), structure="balanced")

    got = finder.edge_loads(plan)
    below = plan.subtree_qubit_masks()
    expected = {edge: 0.0 for edge in got}
    for payload, support, event_type in zip(
        finder.payloads, finder.supports, finder.event_types
    ):
        support = tuple(dict.fromkeys(support))
        if len(support) < 2 or event_type in {
            "measure", "reset", "measure_reset", "cap",
        }:
            continue
        support_mask = sum(1 << q for q in support)
        for edge in expected:
            _parent, child = edge
            left_mask = support_mask & below[child]
            if not left_mask or left_mask == support_mask:
                continue
            left = tuple(q for q in support if left_mask & (1 << q))
            rank = finder._schmidt_rank(payload, support, left)
            expected[edge] += np.log2(rank)

    assert got == pytest.approx(expected)


def test_tree_layout_reuses_dense_gate_schmidt_rank_across_labels():
    """One gate matrix needs one rank calculation for each wire partition."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder(
        [(cnot, (0, 1)), (cnot, (2, 3))], n=4,
        objective="congestion",
    )

    assert finder._schmidt_rank(cnot, (0, 1), (0,)) == 2
    assert finder._schmidt_rank(cnot, (2, 3), (2,)) == 2
    assert len(finder._schmidt_rank_cache) == 1


def test_optimizer_exposes_congestion_layout_objective():
    """TreeOptimizer can request the rank-aware automatic layout."""
    rng = np.random.default_rng(102)
    stream = _random_stream(6, 20, rng, two_qubit_frac=0.8)
    opt = TreeOptimizer(
        stream,
        n=6,
        max_arity=2,
        layout_objective="congestion",
        layout_weight_mode="operator_schmidt",
        run=False,
    )

    assert opt.layout_objective == "congestion"
    assert opt.plan.n == 6
    assert opt.plan.is_binary()


def test_optimizer_defaults_to_congestion_layout_for_replay_performance():
    """Automatic optimizer layouts default to finite-chi edge pressure."""
    opt = TreeOptimizer(None, n=6, run=False)

    assert opt.layout_objective == "congestion"
    assert opt.layout_finder.objective == "congestion"
    assert opt.mode == "auto"
    assert opt.threads == 1
    assert opt.subtree_workers == 1
    assert opt.track_truncation is False
    assert opt.track_infidelity is True
    assert inspect.signature(TreeOptimizer).parameters["cutoff"].default == "auto"
    assert inspect.signature(TreeOptimizer).parameters["cutoff_mode"].default == "auto"
    assert opt.cutoff == pytest.approx(1e-12)
    assert opt.cutoff_mode == "rsum2"
    assert opt.profile is False


def test_layout_recommends_arity_and_reports_tree_shape():
    """The finder compares binary/wider candidates and exposes their costs."""
    rng = np.random.default_rng(103)
    stream = _random_stream(8, 30, rng, two_qubit_frac=0.8)
    finder = TreeLayoutFinder(stream, n=8, objective="congestion")

    recommendation = finder.recommend_arities((2, 3))
    assert recommendation["recommended_max_arity"] in (2, 3)
    assert len(recommendation["candidates"]) == 2
    assert all("max_virtual_degree" in item
               for item in recommendation["candidates"])
    report = finder.report(recommendation["plan"])
    assert report["arity_histogram"]
    assert report["max_arity"] in (2, 3)


def test_layout_finder_layered_direct_block_size():
    """`layered` builds a valid fixed layered tree for a chosen block_size."""
    rng = np.random.default_rng(107)
    stream = _random_stream(12, 40, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=12, weight_mode="operator_schmidt")

    plan = finder.layered(block_size=4)
    assert isinstance(plan, TreePlan)
    assert plan.n == 12
    # Top tensor is fixed ternary once there are at least three blocks.
    assert len(plan.children[plan.root]) == 3
    # The blocking layer groups block_size leaves per blocking node.
    assert 4 in {len(ch) for ch in plan.children.values() if ch}
    # Direct build matches recommend_layered's plan for the same block_size.
    recommended = finder.recommend_layered(block_sizes=(4,))
    assert plan.children == recommended["plan"].children


def test_layered_greedy_refinement_improves_without_changing_tree_shape():
    """Planning swaps leaf labels but preserves the immutable TTN topology."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder([(cnot, (0, 2))], n=4, max_arity=2)
    initial = TreePlan.build_layered(range(4), block_size=2)

    choice = finder.recommend_layered(
        block_sizes=(2,),
        order=range(4),
        refine="greedy",
        refine_budget=3,
    )
    candidate = choice["candidates"][0]
    refined = choice["plan"]

    assert choice["refine"] == "greedy"
    assert refined.children == initial.children
    assert finder.score(refined) < finder.score(initial)
    assert refined.tree_distance(0, 2) == 2
    assert candidate["planning"]["refinement"]["accepted_moves"] >= 1


def test_hybrid_layout_objective_reports_normalized_combined_cost():
    """Hybrid selection combines path and operator-Schmidt edge-load costs."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder(
        [(cnot, (0, 3)), (cnot, (1, 2))],
        n=4,
        objective="hybrid",
        hybrid_weights={"path": 1.0, "max_edge_load": 2.0},
    )
    plan = finder.recommend_arities((2, 3))["plan"]
    report = finder.report(plan)

    assert report["objective"] == "hybrid"
    assert report["hybrid_weights"] == (1.0, 2.0, 0.0)
    assert np.isfinite(report["hybrid_cost"])
    assert report["total_edge_load"] is not None


def test_layered_nevergrad_search_is_optional_and_pre_simulation(monkeypatch):
    """Nevergrad refines only a returned plan and is optional at import time."""
    class FakeArray:
        def __init__(self, *, init):
            self.value = np.asarray(init)

        def set_bounds(self, _lower, _upper):
            return self

    class FakeOptimizer:
        def __init__(self, *, parametrization, budget):
            self.parametrization = parametrization
            self.budget = budget

        def minimize(self, loss):
            loss(self.parametrization.value)
            proposal = np.array([0.0, 2.0, 1.0, 3.0])
            loss(proposal)
            return types.SimpleNamespace(value=proposal)

    fake_nevergrad = types.SimpleNamespace(
        p=types.SimpleNamespace(Array=FakeArray),
        optimizers=types.SimpleNamespace(registry={"OnePlusOne": FakeOptimizer}),
    )
    monkeypatch.setitem(sys.modules, "nevergrad", fake_nevergrad)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder([(cnot, (0, 2))], n=4, max_arity=2)
    initial = TreePlan.build_layered(range(4), block_size=2)

    choice = finder.recommend_layered(
        block_sizes=(2,),
        order=range(4),
        search="nevergrad",
        search_budget=4,
        seed=17,
    )
    search_info = choice["candidates"][0]["planning"]["search"]

    assert choice["search"] == "nevergrad"
    assert choice["plan"].children == initial.children
    assert finder.score(choice["plan"]) <= finder.score(initial)
    assert search_info["method"] == "nevergrad"
    assert search_info["evaluations"] == 2


def test_nevergrad_layout_search_explains_missing_optional_dependency(monkeypatch):
    """The optional Nevergrad dependency fails with an actionable message."""
    monkeypatch.setitem(sys.modules, "nevergrad", None)
    finder = TreeLayoutFinder([], n=4, max_arity=2)

    with pytest.raises(ImportError, match=r"pepsy\[layout\]"):
        finder.recommend_layered(
            block_sizes=(2,), order=range(4), search="nevergrad"
        )


def test_tree_qubit_order_honors_configured_dense_limit(monkeypatch):
    """The layered spectral order uses the finder's dense-size policy."""
    import pepsy.optimizers.tree.layout as tree_layout

    seen = {}

    def ordered(sites, _weights, *, dense_max):
        seen["dense_max"] = dense_max
        return list(sites)

    monkeypatch.setattr(tree_layout, "_gate_stream_spectral_order", ordered)
    finder = TreeLayoutFinder([], n=6, dense_max=17)

    assert finder.qubit_order() == list(range(6))
    assert seen["dense_max"] == 17


def test_treeplan_max_bond_cut_is_structural():
    """`max_bond_cut` is the widest qubit bipartition, set by shape alone."""
    order = list(range(16))
    # block_size=3 groups into an even 2+2+2 ternary top -> widest cut is 6.
    assert TreePlan.build_layered(order, block_size=3).max_bond_cut() == 6
    # block_size=4 forces a {1,1,2}-block top -> one 8-vs-8 cut.
    assert TreePlan.build_layered(order, block_size=4).max_bond_cut() == 8
    # A single leaf has no bonds.
    assert TreePlan.build_layered([0]).max_bond_cut() == 0


def test_recommend_layered_chi_aware_prefers_exact_structure():
    """With ``chi`` set, the layered recommendation avoids chi-overflow bonds."""
    rng = np.random.default_rng(211)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, weight_mode="operator_schmidt")

    # chi-blind candidates always carry ``max_bond_cut`` but no chi fields.
    blind = finder.recommend_layered(block_sizes=(3, 4))
    assert "chi" in blind and blind["chi"] is None
    for c in blind["candidates"]:
        assert "max_bond_cut" in c
        assert "chi_overflow" not in c and "exact_at_chi" not in c

    # chi=64 fits a 6-qubit cut exactly (2**6) but not an 8-qubit cut (2**8):
    # block_size=4 (cut 8) overflows, block_size=3 (cut 6) is exact.
    aware = finder.recommend_layered(block_sizes=(3, 4), chi=64)
    assert aware["chi"] == 64
    assert aware["recommended_block_size"] == 3
    assert aware["plan"].max_bond_cut() <= 6
    by_bs = {c["block_size"]: c for c in aware["candidates"]}
    assert by_bs[3]["exact_at_chi"] and by_bs[3]["chi_overflow"] == 0.0
    assert not by_bs[4]["exact_at_chi"] and by_bs[4]["chi_overflow"] == 2.0
    # The recommended candidate always has the minimum chi_overflow.
    overflows = [c["chi_overflow"] for c in aware["candidates"]]
    assert by_bs[aware["recommended_block_size"]]["chi_overflow"] == min(overflows)


def test_recommend_layered_inherits_finder_chi_unless_overridden():
    """The direct layered recommendation follows the finder's chi policy."""
    rng = np.random.default_rng(216)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(
        stream, n=16, chi=64, weight_mode="operator_schmidt"
    )

    inherited = finder.recommend_layered(block_sizes=(3, 4))
    assert inherited["chi"] == 64
    assert inherited["recommended_block_size"] == 3

    blind = finder.recommend_layered(block_sizes=(3, 4), chi=None)
    assert blind["chi"] is None


def test_recommend_arities_chi_aware_minimizes_overflow():
    """``chi``-aware arity search prefers a structure exact at ``chi``."""
    rng = np.random.default_rng(212)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, objective="congestion")

    rec = finder.recommend_arities((2, 3, 4), chi=64)
    assert rec["chi"] == 64
    for c in rec["candidates"]:
        assert {"max_bond_cut", "chi_overflow", "exact_at_chi"} <= set(c)
    recommended = next(
        c for c in rec["candidates"]
        if c["max_arity"] == rec["recommended_max_arity"]
    )
    overflows = [c["chi_overflow"] for c in rec["candidates"]]
    assert recommended["chi_overflow"] == min(overflows)
    # recommend_layout forwards chi unchanged.
    assert finder.recommend_layout((2, 3, 4), chi=64)["chi"] == 64


def test_recommend_layered_rejects_bad_chi():
    """A non-positive ``chi`` budget is rejected."""
    finder = TreeLayoutFinder([], n=4, structure="balanced")
    with pytest.raises(ValueError, match="chi must be a positive integer"):
        finder.recommend_layered(block_sizes=(2,), chi=0)


def test_layout_finder_uses_binary_ternary_root_by_default():
    """The finder default is fixed binary below a ternary virtual root."""
    rng = np.random.default_rng(213)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, weight_mode="operator_schmidt")

    assert finder.arity_candidates is None
    assert finder.chi is None
    searched = finder.run()
    assert searched.top_arity == 3
    assert searched.is_binary()

    # Candidate arity search remains available explicitly.
    blind = finder.recommend_arities((2, 3, 4))
    assert blind["chi"] is None

    # A scalar max_arity opts back into a single fixed binary tree.
    fixed = TreeLayoutFinder(stream, n=16, max_arity=2,
                             weight_mode="operator_schmidt")
    assert fixed.arity_candidates is None
    assert fixed.run().is_binary()


def test_layout_finder_run_accepts_search_overrides(monkeypatch):
    """Tree ``run`` mirrors MPS by accepting per-run quality-search controls."""
    finder = TreeLayoutFinder([], n=4, max_arity=(2, 3), chi=8)
    captured = {}
    recommend_arities = finder.recommend_arities

    def capture(max_arities, **kwargs):
        captured.update(kwargs)
        return recommend_arities(max_arities, **kwargs)

    monkeypatch.setattr(finder, "recommend_arities", capture)
    plan = finder.run(
        chi=None,
        refine="greedy",
        refine_budget=2,
        search=None,
        search_budget=7,
        seed=11,
        nevergrad_optimizer="OnePlusOne",
        progbar=True,
    )

    assert isinstance(plan, TreePlan)
    assert captured == {
        "chi": None,
        "refine": "greedy",
        "refine_budget": 2,
        "topology_refine": None,
        "topology_budget": None,
        "search": None,
        "search_budget": 7,
        "seed": 11,
        "nevergrad_optimizer": "OnePlusOne",
        "progbar": True,
    }
    assert finder._last_arity_recommendation["refine"] == "greedy"
    assert finder._last_arity_recommendation["chi"] is None


def test_layout_finder_explicit_arity_search_is_chi_aware():
    """An explicit candidate search remains ``chi``-aware."""
    rng = np.random.default_rng(214)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, chi=64,
                              weight_mode="operator_schmidt")

    assert finder.chi == 64
    searched = finder.run()
    aware = finder.recommend_arities((2, 3, 4), chi=64)
    assert searched.top_arity == 3
    assert searched.is_binary()
    # The chi-aware search never overflows chi by more than the binary tree.
    by_arity = {c["max_arity"]: c for c in aware["candidates"]}
    chosen = by_arity[aware["recommended_max_arity"]]
    assert chosen["chi_overflow"] <= by_arity[2]["chi_overflow"]


def test_layout_report_summarizes_quality():
    """TreeLayoutFinder.report exposes geodesic + score diagnostics."""
    rng = np.random.default_rng(11)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    finder = TreeLayoutFinder(stream, n=n)
    rep = finder.report()
    assert rep["n_qubits"] == n
    assert rep["n_interacting_pairs"] >= 1
    assert rep["max_path"] >= 1
    assert rep["weighted_mean_path"] > 0.0
    # the chosen quality structure is no worse than a balanced index tree
    assert rep["score"] <= rep["balanced_score"] + 1e-9


def test_tree_layout_finder_plot_defaults_to_tent():
    """The public plot shows the structural tent, not gate-route overlays."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2, top_arity=2)
    plan = finder.run()
    assert plan.is_binary()
    assert len(plan.children[plan.root]) == 2
    fig, ax = finder.plot(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert fig is ax.figure
    assert ax.get_title() == ""
    assert not ax.patches
    assert len(fig.axes) == 1
    assert not ax.axison  # schematic-style presentation by default
    assert not ax.texts
    assert len(ax.collections) == 1 + sum(
        not plan.is_leaf(node) for node in plan.nodes()
    )
    plt.close(fig)


def test_tree_layout_finder_can_hide_gate_paths_for_structural_view():
    """The structural view makes the binary TTN edges unambiguous."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot(
        plan,
        lattice=False,
        show_gate_connectivity=False,
        show_edge_arrows=False,
    )

    assert len(ax.lines) == len(plan.nodes()) - 1
    assert not ax.patches
    assert not ax.texts
    assert not ax.axison
    plt.close(fig)


def test_tree_layout_finder_plot_tent_draws_hierarchy_over_raw_graph():
    """Tent plotting separates the binary hierarchy from raw connectivity."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert plan.is_binary()
    assert fig is ax.figure
    assert not ax.patches
    assert not ax.texts
    assert not ax.axison
    assert len(ax.lines) >= len(plan.nodes()) - 1
    assert len(ax.collections) == 1 + sum(
        not plan.is_leaf(node) for node in plan.nodes()
    )
    plt.close(fig)


def test_tree_layout_tent_can_hide_physical_leaf_nodes():
    """Physical plus-mark backdrops can replace tree leaf circles."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))],
        n=4,
        max_arity=2,
    )
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        lattice=False,
        show_gate_connectivity=False,
        show_leaf_nodes=False,
    )

    assert len(ax.collections) == sum(
        not plan.is_leaf(node) for node in plan.nodes()
    )
    assert len(ax.lines) == len(plan.nodes()) - 1
    plt.close(fig)


def test_tree_layout_scale_colors_do_not_depend_on_gate_stream_length():
    """Scale coloring remains fixed when the gate stream changes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plan = TreePlan.from_order(range(8), structure="balanced", max_arity=2)
    streams = [
        [(pepsy.cnot(), (0, 7))],
        [
            (pepsy.cnot(), (0, 7)),
            (pepsy.cnot(), (1, 6)),
            (pepsy.cnot(), (2, 5)),
            (pepsy.cnot(), (3, 4)),
        ],
    ]
    structural_colors = []
    scale_node_colors = []
    for gates in streams:
        finder = TreeLayoutFinder(gates, n=8, max_arity=2)
        fig, ax = finder.plot_tent(
            plan,
            color_by="scale",
            edge_color=None,
            show_edge_arrows=False,
        )
        # Gate-connectivity overlays are disabled by default, so only the
        # one-dimensional physical lattice precedes the hierarchy edges.
        background_lines = len(plan.leaves()) - 1
        structural_colors.append(
            tuple(
                tuple(line.get_color())
                for line in ax.lines[background_lines:]
            )
        )
        scale_node_colors.append(
            tuple(
                tuple(collection.get_facecolors()[0])
                for collection in ax.collections
            )
        )
        assert len(fig.axes) == 1
        assert len(ax.collections) == 1 + sum(
            not plan.is_leaf(node) for node in plan.nodes()
        )
        plt.close(fig)

    assert structural_colors[0] == structural_colors[1]
    assert scale_node_colors[0] == scale_node_colors[1]
    assert len(set(structural_colors[0])) > 1
    assert len(set(scale_node_colors[0])) > 1


def test_tree_layout_tent_edges_match_order_colors_by_default():
    """Tent hierarchy edges follow the default order color palette."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot_tent(plan, color_by="scale")

    background_lines = len(plan.leaves()) - 1
    hierarchy_colors = {
        line.get_color() for line in ax.lines[background_lines:]
    }
    assert len(hierarchy_colors) > 1
    assert not ax.patches
    plt.close(fig)


def test_tree_layout_tent_can_highlight_leaf_edges():
    """The physical-to-first-parent layer can use a contrasting color."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
        top_arity=2,
    )
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        lattice=False,
        show_gate_connectivity=False,
        leaf_edge_color="#2563eb",
    )

    line_index = 0
    for parent, children in plan.children.items():
        for child in children:
            if plan.is_leaf(child):
                assert ax.lines[line_index].get_color() == "#2563eb"
            line_index += 1
    assert line_index == len(plan.nodes()) - 1
    plt.close(fig)


def test_tree_layout_tent_colored_edges_match_child_nodes():
    """Colored incoming edges use the same scale color as their child node."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
    )
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        color_by="scale",
        edge_color=None,
        show_edge_arrows=False,
    )

    background_lines = plan.n - 1
    internal_nodes = [node for node in plan.nodes() if not plan.is_leaf(node)]
    node_colors = {
        node: tuple(collection.get_facecolors()[0])
        for node, collection in zip(internal_nodes, ax.collections[1:])
    }
    hierarchy_lines = ax.lines[background_lines:]
    line_index = 0
    for parent, children in plan.children.items():
        for child in children:
            if not plan.is_leaf(child):
                assert tuple(
                    hierarchy_lines[line_index].get_color()
                ) == pytest.approx(node_colors[child])
            line_index += 1
    assert line_index == len(hierarchy_lines)
    plt.close(fig)


def test_tree_layout_tent_validates_arrow_size():
    """Arrow marker sizing rejects values Matplotlib cannot render usefully."""
    pytest.importorskip("matplotlib")
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 1))], n=2, max_arity=2
    )
    plan = finder.run()

    with pytest.raises(ValueError, match="arrow_size"):
        finder.plot_tent(plan, arrow_size=0.0)


def test_tree_layout_finder_plot_rubberband_is_axis_free_and_unlabeled():
    """Rubberband plots show clusters without plot text or axes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
        (pepsy.cnot(), (1, 2)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot_rubberband(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert fig is ax.figure
    assert ax.get_title() == ""
    assert not ax.axison
    assert not ax.texts
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_tree_layout_quality_order_enables_bounded_refinement(monkeypatch):
    """Tree order='quality' mirrors the MPS high-quality mode."""
    monkeypatch.setitem(sys.modules, "nevergrad", None)
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
        order="quality",
    )
    captured = {}

    def fake_improve(plan, *, chi, settings, progbar=False):
        captured.update(settings)
        return plan, {"method": "test"}

    monkeypatch.setattr(finder, "_improve_plan", fake_improve)
    plan = finder.run()

    assert plan.n == 4
    assert finder.objective == "full_tree"
    assert captured["refine"] == "greedy"
    assert captured["topology_refine"] == "subtree"
    assert captured["search"] == "anneal"
    assert captured["search_budget"] == finder.search_budget


def test_tree_layout_quality_run_upgrades_a_fast_finder(monkeypatch):
    """The explicit quality run is the full-tree mode even after construction."""
    monkeypatch.setitem(sys.modules, "nevergrad", None)
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
    )
    captured = {}

    def fake_improve(plan, *, chi, settings, progbar=False):
        captured.update(settings)
        return plan, {"method": "test"}

    monkeypatch.setattr(finder, "_improve_plan", fake_improve)
    plan = finder.run(order="quality")

    assert plan.n == 4
    assert finder.objective == "full_tree"
    assert captured["topology_refine"] == "subtree"
    assert captured["search"] == "anneal"


def test_tree_layout_nni_refinement_changes_binary_topology():
    """Quality refinement can move a correlated subtree, not only labels."""
    cnot = pepsy.cnot()
    finder = TreeLayoutFinder(
        [(cnot, (0, 2)), (cnot, (0, 3))],
        n=4,
        max_arity=2,
        top_arity=2,
        objective="path",
    )
    initial = TreePlan.from_order(
        range(4), structure="balanced", max_arity=2, top_arity=2,
    )

    refined, planning = finder._refine_plan_topology(
        initial,
        chi=None,
        budget=4,
    )

    assert refined.is_binary()
    assert refined.children != initial.children
    assert finder.score(refined) < finder.score(initial)
    assert planning["accepted_moves"] >= 1


def test_tree_layout_temporal_weights_apply_to_paths_and_edge_loads():
    """Recent-event weighting affects both locality and Schmidt-load scoring."""
    cnot = pepsy.cnot()
    gates = [(cnot, (0, 1)), (cnot, (2, 3))]
    full = TreeLayoutFinder(gates, n=4, max_arity=2, objective="congestion")
    recent = TreeLayoutFinder(
        gates,
        n=4,
        max_arity=2,
        objective="congestion",
        time_decay=0.5,
        time_window=1,
    )

    assert full.temporal_factors == (1.0, 1.0)
    assert recent.temporal_factors == (0.0, 1.0)
    assert sum(recent.event_weights) == pytest.approx(1.0)
    assert sum(recent.edge_loads(recent.run()).values()) < sum(
        full.edge_loads(full.run()).values()
    )
    report = recent.report()
    assert report["time_decay"] == pytest.approx(0.5)
    assert report["time_window"] == 1
    assert report["active_events"] == 1

    opt = TreeOptimizer(
        gates,
        n=4,
        max_arity=2,
        layout_time_decay=0.5,
        layout_time_window=1,
        run=False,
    )
    assert opt.layout_finder.time_window == 1
    assert opt.layout_finder.time_decay == pytest.approx(0.5)


def test_tree_layout_order_rejects_non_quality_modes():
    """Tree layouts expose quality mode rather than 1-D order names."""
    finder = TreeLayoutFinder([], n=4, max_arity=2)
    with pytest.raises(ValueError, match="order"):
        finder.run(order="input")


def test_tree_layout_rubberband_defaults_to_cotengra_ordered_colors():
    """Default rubberbands use distinct post-order Spectral colors."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
    )
    fig, ax = finder.plot_rubberband(
        finder.run(),
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    expected = matplotlib.colormaps["Spectral"]
    assert np.allclose(
        ax.patches[0].get_edgecolor()[:3], expected(0.0)[:3]
    )
    assert np.allclose(
        ax.patches[-1].get_edgecolor()[:3], expected(1.0)[:3]
    )
    assert ax.patches[0].get_zorder() > ax.patches[-1].get_zorder()
    plt.close(fig)


def test_tree_optimizer_plot_layout_with_explicit_plan_is_non_mutating():
    """The tree optimizer wrapper plots an explicit plan without replay."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [(pepsy.cnot(), (0, 3))]
    plan = TreeLayoutFinder(gates, n=4, max_arity=2).run()
    opt = TreeOptimizer(gates, tree=plan, run=False)
    before = opt.to_dense().copy()
    fig, _ = opt.plot_layout(site_coords={q: (q, 0) for q in range(4)})

    assert np.allclose(opt.to_dense(), before)
    plt.close(fig)
