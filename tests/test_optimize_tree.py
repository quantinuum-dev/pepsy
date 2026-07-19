"""Tests for the tree-tensor-network gate simulator (:class:`TreeOptimizer`)."""

import numpy as np
import pytest

from pepsy.optimizers.tree import (
    TreeLayoutFinder,
    TreeOptimizer,
    TreePlan,
    TreeTensorNetwork,
)


# -- exact statevector reference ----------------------------------------------


def _sv_apply_1q(psi, g, q, n):
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q).reshape(-1)


def _sv_apply_2q(psi, g, a, b, n):
    g = g.reshape(2, 2, 2, 2)
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=([2, 3], [a, b]))
    return np.moveaxis(psi, [0, 1], [a, b]).reshape(-1)


def _rand_unitary(k, rng):
    m = rng.standard_normal((2**k, 2**k)) + 1j * rng.standard_normal((2**k, 2**k))
    q, _ = np.linalg.qr(m)
    return q


def _random_stream(n, ngates, rng, two_qubit_frac=0.5):
    stream = []
    for _ in range(ngates):
        if n >= 2 and rng.random() < two_qubit_frac:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((_rand_unitary(2, rng), (int(a), int(b))))
        else:
            stream.append((_rand_unitary(1, rng), int(rng.integers(n))))
    return stream


def _exact_state(stream, n):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    for g, where in stream:
        if isinstance(where, int):
            psi = _sv_apply_1q(psi, g, where, n)
        else:
            psi = _sv_apply_2q(psi, g, where[0], where[1], n)
    return psi


def _sv_expect(psi, op, where, n):
    """Exact ``<psi|op|psi>`` for a (multi-site) operator from the dense state."""
    psi = psi.reshape([2] * n)
    k = len(where)
    op = np.asarray(op).reshape([2] * (2 * k))
    o = np.tensordot(op, psi, axes=(list(range(k, 2 * k)), list(where)))
    o = np.moveaxis(o, range(k), where)
    return np.vdot(psi.reshape(-1), o.reshape(-1))


def _fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2 / (
        np.vdot(a, a).real * np.vdot(b, b).real
    )


# -- tests --------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_matches_statevector(n, seed):
    """Untruncated tree replay reproduces the exact statevector."""
    rng = np.random.default_rng(seed)
    stream = _random_stream(n, 8 * n, rng)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_single_qubit_stream():
    """A one-qubit tree replays single-qubit gates correctly."""
    rng = np.random.default_rng(3)
    stream = [(_rand_unitary(1, rng), 0) for _ in range(5)]
    opt = TreeOptimizer(stream, n=1)
    psi = _exact_state(stream, 1)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-10


def test_local_expectation_matches_exact():
    """Single-site expectations from the tree match the dense state."""
    rng = np.random.default_rng(4)
    n = 6
    stream = _random_stream(n, 40, rng)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)
    psi /= np.linalg.norm(psi)

    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    for q in range(n):
        for op in (z, x):
            exact = np.vdot(psi, _sv_apply_1q(psi, op, q, n)).real
            got = opt.local_expectation(op, q).real
            assert abs(got - exact) < 1e-8


def test_chi_truncation_caps_bond():
    """The maximum bond never exceeds the requested chi."""
    rng = np.random.default_rng(5)
    n = 8
    stream = _random_stream(n, 80, rng)
    chi = 4
    opt = TreeOptimizer(stream, n=n, chi=chi)
    assert opt.max_bond() <= chi


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_truncated_fidelity_improves_with_chi(seed):
    """Threading the whole gate before truncating yields high truncated fidelity.

    The gate is threaded *exactly* along the tree geodesic and only compressed
    once both factors are present (Seitz et al., Figs. 3-6).  Every bond
    truncation therefore sees the complete gate, so fidelity rises
    monotonically with ``chi`` and reaches good accuracy at moderate ``chi`` --
    unlike truncating each hop before the far gate factor has been absorbed.
    """
    rng = np.random.default_rng(seed)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    psi = _exact_state(stream, n)

    fids = [
        _fidelity(psi, TreeOptimizer(stream, n=n, chi=chi).to_dense())
        for chi in (2, 4, 8)
    ]
    # monotone non-decreasing in chi (allowing tiny numerical slack)
    assert fids[1] >= fids[0] - 1e-9
    assert fids[2] >= fids[1] - 1e-9
    # moderate chi already recovers most of the state
    assert fids[2] > 0.4


def test_normalize_sets_unit_norm():
    """normalize() rescales the represented state to unit norm."""
    rng = np.random.default_rng(6)
    n = 5
    stream = _random_stream(n, 30, rng)
    opt = TreeOptimizer(stream, n=n, chi=8)  # truncated -> norm < 1
    opt.normalize()
    assert abs(opt.norm() - 1.0) < 1e-9


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


def test_layout_finder_builds_valid_tree():
    """The layout finder returns a rooted binary tree over all qubits."""
    rng = np.random.default_rng(8)
    n = 8
    stream = _random_stream(n, 60, rng)
    plan = TreeLayoutFinder(stream, n=n).run()
    assert plan.n == n
    assert set(plan.leaf_of_qubit) == set(range(n))
    # every internal node has exactly two children
    for nid in plan.nodes():
        assert len(plan.children[nid]) in (0, 2)
    # tree distances are well-defined for all pairs
    for a in range(n):
        for b in range(a + 1, n):
            assert plan.tree_distance(a, b) >= 1


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


def test_public_api_exports_tree_optimizer():
    """TreeOptimizer is exposed through the public namespaces."""
    import pepsy

    assert pepsy.TreeOptimizer is TreeOptimizer
    from pepsy.optimizers import TreeOptimizer as FromOptimizers

    assert FromOptimizers is TreeOptimizer


# -- diagnostics --------------------------------------------------------------


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


def test_bond_report_reflects_chi():
    """bond_report caps at chi and counts the tree tensors."""
    rng = np.random.default_rng(12)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=4)
    rep = opt.bond_report()
    assert rep["chi"] == 4
    assert rep["max_bond"] <= 4
    assert rep["mean_bond"] <= rep["max_bond"]
    assert rep["n_tensors"] == len(opt.plan.nodes())


def test_convergence_sweep_reports_rising_fidelity():
    """convergence_sweep reuses one tree and reports monotone fidelity."""
    rng = np.random.default_rng(13)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    recs = TreeOptimizer.convergence_sweep(
        stream, n=n, chi_values=(8, 2, 4, 64), ops=[(z, 0), (z, n - 1)]
    )
    # sorted ascending internally
    assert [r["chi"] for r in recs] == [2, 4, 8, 64]
    fids = [r["fidelity"] for r in recs]
    assert all(f is not None for f in fids)
    for a, b in zip(fids, fids[1:]):
        assert b >= a - 1e-9
    assert fids[-1] > 1 - 1e-6
    assert recs[0]["max_drift"] is None
    assert all(r["max_drift"] is not None for r in recs[1:])
    assert all(len(r["expectations"]) == 2 for r in recs)
    assert all(r["max_bond"] <= r["chi"] for r in recs)


def test_convergence_sweep_skips_fidelity_when_large():
    """The dense fidelity reference is skipped past dense_cap."""
    rng = np.random.default_rng(14)
    n = 6
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.6)
    recs = TreeOptimizer.convergence_sweep(
        stream, n=n, chi_values=(2, 4), dense_cap=8
    )
    assert all(r["fidelity"] is None for r in recs)


# -- stability / speed hardening ----------------------------------------------


def test_fresh_state_is_canonical_at_root():
    """A newly built product state is canonical with the root as centre.

    Every virtual bond starts at dimension 1, so each tensor is trivially
    isometric: the tree is already normalised with the root as orthogonality
    centre, and no canonicalisation is needed before the first gate.
    """
    opt = TreeOptimizer(None, n=8, chi=16)
    assert opt.center == opt.plan.root
    # one-site canonical norm (uses the tracked centre) is exactly 1
    assert abs(opt.norm() - 1.0) < 1e-12
    # ...and it agrees with the full doubled-tree contraction
    opt.center = None
    assert abs(opt.norm() - 1.0) < 1e-12


def test_two_qubit_gate_rejects_repeated_qubit():
    """A two-qubit gate on a single qubit is rejected loudly."""
    rng = np.random.default_rng(15)
    opt = TreeOptimizer(None, n=4, chi=8)
    with pytest.raises(ValueError, match="two distinct qubits"):
        opt.apply_gate(_rand_unitary(2, rng), (2, 2))


def test_tid_cache_self_heals_after_leaf_replacement():
    """The node->tid cache stays valid after gates replace leaf tensors."""
    rng = np.random.default_rng(16)
    n = 6
    opt = TreeOptimizer(None, n=n, chi=16)
    # warm the cache
    for nid in opt.plan.nodes():
        assert opt._tid(nid) in opt.tn.tensor_map
    # single-qubit gates rebuild leaf tensors (new tids)
    for q in range(n):
        opt.apply_1q(_rand_unitary(1, rng), q)
    # cache still resolves every node to a live tensor id
    for nid in opt.plan.nodes():
        assert opt._tid(nid) in opt.tn.tensor_map


def test_copy_is_independent():
    """copy() yields an optimizer that evolves without touching the original."""
    rng = np.random.default_rng(17)
    n = 5
    stream = _random_stream(n, 20, rng)
    base = TreeOptimizer(stream, n=n, chi=16)
    before = base.to_dense()

    clone = base.copy()
    assert clone.plan is base.plan
    assert clone.chi == base.chi and clone.threads == base.threads
    assert _fidelity(before, clone.to_dense()) > 1 - 1e-12

    clone.apply_gate(_rand_unitary(2, rng), (0, 1))
    # the original is unchanged; the clone has diverged
    assert np.allclose(base.to_dense(), before)
    assert not np.allclose(base.to_dense(), clone.to_dense())


def test_threads_setting_preserves_result():
    """The thread cap is a performance knob only; results are identical."""
    rng = np.random.default_rng(18)
    n = 7
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    a = TreeOptimizer(stream, n=n, chi=8, threads=1).to_dense()
    b = TreeOptimizer(stream, n=n, chi=8, threads=None).to_dense()
    assert np.allclose(a, b)


# -- sibling fast path / measurement / multi-site expectation -----------------


def test_sibling_fast_path_matches_statevector():
    """Two-qubit gates on sibling leaves reproduce the exact statevector.

    A balanced plan over ``range(4)`` makes qubits ``(0, 1)`` and ``(2, 3)``
    siblings, so every two-qubit gate here takes the parent-blob fast path.
    """
    rng = np.random.default_rng(20)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    stream = [
        (_rand_unitary(2, rng), (0, 1) if rng.random() < 0.5 else (2, 3))
        for _ in range(30)
    ]
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_mixed_paths_match_statevector():
    """A stream mixing sibling and non-sibling two-qubit gates stays exact."""
    rng = np.random.default_rng(21)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


@pytest.mark.parametrize("seed", [22, 23])
def test_multisite_local_expectation_matches_exact(seed):
    """Canonical multi-site expectations match the dense state exactly."""
    rng = np.random.default_rng(seed)
    n = 7
    stream = _random_stream(n, 50, rng)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)
    psi /= np.linalg.norm(psi)

    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    zz = np.kron(z, z)
    zxz = np.kron(np.kron(z, x), z)
    for where in [(0, 3), (1, 5), (2, 6), (0, 6)]:
        got = opt.local_expectation(zz, where)
        exact = _sv_expect(psi, zz, where, n)
        assert abs(got - exact) < 1e-9
    for where in [(0, 3, 6), (1, 2, 5)]:
        got = opt.local_expectation(zxz, where)
        exact = _sv_expect(psi, zxz, where, n)
        assert abs(got - exact) < 1e-9


def test_measure_born_statistics_and_collapse():
    """Measurement samples the Born rule and collapses to a unit-norm state."""
    theta = 0.7
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    base = TreeOptimizer([(ry, 0)], n=3, chi=4, seed=0)

    n_shots = 3000
    ones = sum(base.copy().measure(0) for _ in range(n_shots))
    assert abs(ones / n_shots - s**2) < 0.03  # p(1) = sin^2(theta/2)

    forced = base.copy()
    assert forced.measure(0, outcome=0) == 0
    assert abs(forced.norm() - 1.0) < 1e-9


def test_reset_forces_ground_state():
    """reset() returns a qubit to |0> regardless of its prior value."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    opt = TreeOptimizer([(x, 1)], n=3, chi=4, seed=1)  # qubit 1 in |1>
    opt.reset(1)
    assert opt.local_expectation(z, 1).real > 1 - 1e-9  # <Z> = +1 -> |0>
    assert abs(opt.norm() - 1.0) < 1e-9


def test_measure_is_seed_reproducible():
    """Two optimizers with the same seed measure the same outcome."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    a = TreeOptimizer([(h, 0)], n=2, seed=42).measure(0)
    b = TreeOptimizer([(h, 0)], n=2, seed=42).measure(0)
    assert a == b


# -- TreeTensorNetwork class --------------------------------------------------


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
    # show() prints the same drawing (+ trailing newline)
    ttn.show()
    assert capsys.readouterr().out.rstrip("\n") == text
    # optimizer delegates to the state's drawing
    TreeOptimizer(None, n=4).show()
    assert capsys.readouterr().out.rstrip("\n") == text


# -- non-binary / arbitrary-arity trees ---------------------------------------


def _nonbinary_plan():
    """Two arity-3 star nodes under a binary root over qubits 0..5."""
    children = {
        0: (), 1: (), 2: (), 3: (), 4: (), 5: (),
        6: (0, 1, 2), 7: (3, 4, 5), 8: (6, 7),
    }
    qubit_of_leaf = {i: i for i in range(6)}
    return TreePlan.from_children(children, qubit_of_leaf)


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
    # the layout finder default is still a binary tree
    rng = np.random.default_rng(2)
    stream = _random_stream(8, 60, rng)
    assert TreeLayoutFinder(stream, n=8).run().is_binary()


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


# -- orthogonality-centre movement --------------------------------------------


def _entangled_ttn(seed=0, n=6, D=3, structure="balanced"):
    """A canonical-at-root random tree state for centre-movement tests."""
    plan = TreePlan.from_order(range(n), structure=structure)
    return TreeTensorNetwork.rand(plan, D=D, seed=seed)


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







