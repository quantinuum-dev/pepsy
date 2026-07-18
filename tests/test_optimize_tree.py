"""Tests for the tree-tensor-network gate simulator (:class:`TreeOptimizer`)."""

import numpy as np
import pytest

from pepsy.optimizers.tree import TreeLayoutFinder, TreeOptimizer, TreePlan


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


