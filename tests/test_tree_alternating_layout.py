"""Spatial topology, rather than just traversal, of x/y pairing trees."""

import numpy as np
import pytest

from pepsy.optimizers.tree import TreeLayoutFinder, TreeMPO, TreePlan, TreeTensorNetwork

pytestmark = [pytest.mark.tree, pytest.mark.integration]


@pytest.mark.parametrize("shape", [(9, 10), (4, 4), (3, 5), (1, 7), (7, 1), (1, 1)])
def test_alternating_hierarchy_contains_exact_dyadic_rectangles(shape):
    lx, ly = shape
    plan = TreePlan.from_alternating_lattice(shape)
    masks = plan.subtree_qubit_masks()
    expected = {1 << q for q in range(lx * ly)}
    widths = [1, 1]
    axis = 0
    level_shapes = [shape]
    while widths[0] < lx or widths[1] < ly:
        if widths[axis] < shape[axis]:
            widths[axis] *= 2
            wx, wy = widths
            for x in range(0, lx, wx):
                for y in range(0, ly, wy):
                    expected.add(sum(
                        1 << (i * ly + j)
                        for i in range(x, min(x + wx, lx))
                        for j in range(y, min(y + wy, ly))
                    ))
            level_shapes.append(((lx + wx - 1) // wx, (ly + wy - 1) // wy))
        axis = 1 - axis

    assert set(masks.values()) == expected
    assert len(plan.children) == 2 * lx * ly - 1
    assert plan.n == lx * ly
    assert plan.is_strictly_binary()
    assert plan.top_arity == (0 if shape == (1, 1) else 2)
    assert sorted(plan.qubit_of_leaf.values()) == list(range(lx * ly))
    assert plan.map_mode == "alternating-xy"
    positions = {q: i for i, q in enumerate(plan.mpo_order())}
    for mask in masks.values():
        occupied = sorted(positions[q] for q in positions if mask & (1 << q))
        assert occupied == list(range(occupied[0], occupied[-1] + 1))
    if shape == (9, 10):
        assert level_shapes == [
            (9, 10), (5, 10), (5, 5), (3, 5), (3, 3),
            (2, 3), (2, 2), (1, 2), (1, 1),
        ]


def test_alternating_finder_preserves_custom_site_labels_and_override():
    def site(x, y):
        return y * 3 + (x if y % 2 == 0 else 2 - x)

    expected = TreePlan.from_alternating_lattice((3, 4), site=site)
    finder = TreeLayoutFinder(
        [], n=12, lattice_shape=(3, 4), lattice_site=site,
        map_mode="alternating_xy", coarse_grain=(3, 2),
    )
    plan = finder.run()
    assert plan.children == expected.children
    assert plan.qubit_of_leaf == expected.qubit_of_leaf
    masks = plan.subtree_qubit_masks()
    assert all((1 << site(0, y)) | (1 << site(1, y)) in masks.values() for y in range(4))
    assert TreeLayoutFinder.lattice_order(
        3, 4, "alternating-xy", site=site,
    ) == plan.mpo_order()
    report = finder.report(plan)
    assert report["map_mode"] == "alternating-xy"
    assert report["coarse_grain"] is None
    assert report["top_arity"] == 2

    # The ordinary finder retains its historical ternary default; selecting
    # this topology at run time still uses the requested binary hierarchy.
    ordinary = TreeLayoutFinder([], n=12, lattice_shape=(3, 4), lattice_site=site)
    assert ordinary.run(order="row-major").top_arity == 3
    overridden = ordinary.run(order="alternating-xy")
    assert overridden.children == plan.children
    assert overridden.qubit_of_leaf == plan.qubit_of_leaf
    assert ordinary.report(overridden)["map_mode"] == "alternating-xy"


@pytest.mark.parametrize("kwargs, message", [
    ({"root_qubit": 0}, "root_qubit"),
    ({"max_arity": 3}, "max_arity"),
    ({"max_arity": (2, 3)}, "max_arity"),
    ({"top_arity": 3}, "top_arity"),
    ({"lattice_shape": None}, "lattice_shape"),
    ({"lattice_shape": (2, 2, 2)}, "lattice_shape"),
])
def test_alternating_finder_rejects_incompatible_topologies(kwargs, message):
    options = {"n": 8, "lattice_shape": (4, 2), "order": "alternating-xy"}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TreeLayoutFinder([], **options).run()


@pytest.mark.parametrize("top_arity", [None, 2])
def test_alternating_finder_accepts_explicit_binary_root(top_arity):
    plan = TreeLayoutFinder(
        [], n=6, lattice_shape=(3, 2), order="alternating-xy", top_arity=top_arity,
    ).run()
    assert plan.is_strictly_binary()


def test_alternating_plan_validates_shape_and_site_permutation():
    for shape in (None, (2, 2, 2), (0, 3)):
        with pytest.raises(ValueError, match="lattice_shape"):
            TreePlan.from_alternating_lattice(shape)
    with pytest.raises(ValueError):
        TreePlan.from_alternating_lattice((3, 2), site=lambda x, y: 0)


def test_alternating_map_mode_flows_to_state_and_native_tree_operator():
    plan = TreePlan.from_alternating_lattice((3, 2))
    state = TreeTensorNetwork.from_plan(plan)
    operator = TreeMPO.from_terms(plan, {(0,): np.diag([1., -1.])}, compress=False)
    assert state.map_mode == operator.map_mode == "alternating-xy"
    assert np.isfinite(operator.expectation(state))
