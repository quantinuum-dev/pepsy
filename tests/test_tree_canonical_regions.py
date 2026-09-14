"""Local canonical-region preparation and proof-safe oversampled rounding."""

import numpy as np
import pytest

from pepsy import Fermion, TreeOptimizer, TreePlan, TreeTensorNetwork, ps_to_ttn


def _state(backend="numpy", n=8):
    plan = TreePlan.from_order(range(n), structure="balanced", top_arity=2)
    if backend == "native":
        pytest.importorskip("symmray")
        fermion = Fermion(spinful=False, symmetry="U1", dtype="complex128")
        state = ps_to_ttn(n, tree=plan, fermion=fermion,
                          occupations=tuple(i % 2 for i in range(n)))
        # Establish explicit graded QR proofs, not just product-state gauge.
        state.invalidate_canonical_form()
        state.canonize_around_node_(plan.root)
        return state
    state = TreeTensorNetwork.rand(plan, D=3, seed=41, dtype="complex128")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
    return state


@pytest.mark.parametrize("backend", ["numpy", "torch", "native"])
@pytest.mark.parametrize("recovery", ["invalidate", "sync"])
def test_raw_edit_recovery_clears_stale_isometry_proofs(backend, recovery):
    opt = TreeOptimizer(None, state=_state(backend), run=False)
    state = opt.tn
    tensor = state.node_tensor(state.plan.node_of_qubit[0])
    before = opt.to_dense().copy()
    opt.norm()  # prime any native norm cache before the unmanaged edit
    assert tensor.left_inds is not None
    if backend == "native":
        for block in tensor.data.blocks.values():
            block[...] *= 2
    else:
        tensor.data[...] *= 2
    if recovery == "invalidate":
        arrays = [t.data for t in state.tensors]
        assert state.invalidate_canonical_form() is state
        assert state.canonical_region is None
        assert all(t.left_inds is None for t in state.tensors)
        assert all(t.data is a for t, a in zip(state.tensors, arrays))
        state.shift_orthogonality_center(state.plan.root)
    else:
        opt.sync_canonicalization()
    assert state.is_canonical_form(tol=1e-10)
    state.validate_isometry_metadata()
    np.testing.assert_allclose(opt.to_dense(), 2 * before, atol=1e-10)
    assert opt.norm() == pytest.approx(np.linalg.norm(2 * before))


def test_public_handoff_recovers_numerically_invalid_region():
    state = _state()
    state.node_tensor(state.plan.node_of_qubit[0]).data[...] *= 2
    expected = state.to_statevector().copy()
    opt = TreeOptimizer(None, state=state, run=False)
    assert opt.tn.is_canonical_form()
    np.testing.assert_allclose(opt.to_dense(), expected, atol=1e-10)


@pytest.mark.parametrize("geometry", ["overlap", "contained", "disjoint"])
@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_known_region_preparation_only_peels_required_nodes(geometry, backend, monkeypatch):
    state = _state(backend, n=16)
    plan = state.plan
    old = state.steiner_nodes([plan.node_of_qubit[q] for q in (0, 1, 2)])
    targets = {"overlap": (0, 1), "contained": (0, 1, 2, 3), "disjoint": (14, 15)}
    new = state.steiner_nodes([plan.node_of_qubit[q] for q in targets[geometry]])
    state.canonize_subtree_(old)
    before = state.to_statevector().copy()
    arrays = {n: state.node_tensor(n).data for n in plan.nodes()}
    moves = []
    canonize = state.canonize_edge_

    def checked(a, b, **kwargs):
        assert a not in new  # never decompose a tensor being retained
        moves.append((a, b))
        return canonize(a, b, **kwargs)

    def no_global(*args, **kwargs):
        raise AssertionError("known canonical region entered full-tree preparation")

    monkeypatch.setattr(state, "canonize_edge_", checked)
    monkeypatch.setattr(state, "canonize_around_", no_global)
    state.canonize_subtree_(new)
    assert state.canonical_region == frozenset(new)
    assert state.is_subtree_canonical_form(tol=1e-10)
    state.validate_isometry_metadata()
    changed = {n for edge in moves for n in edge}
    if geometry == "contained":
        assert not moves
    elif geometry == "overlap":
        assert len(moves) == len(old - new)
        assert changed <= old
    else:
        connector = plan.node_path(min(old), min(new))
        entry = next(n for n in connector if n in new)
        work = old | set(connector[:connector.index(entry) + 1])
        assert len(moves) == len(work) - 1
        assert changed <= work
    for node in set(plan.nodes()) - changed:
        assert state.node_tensor(node).data is arrays[node]
    np.testing.assert_allclose(state.to_statevector(), before, atol=1e-10)


def test_unknown_region_still_uses_full_recovery(monkeypatch):
    state = _state()
    state.invalidate_canonical_form()
    calls = []
    original = state.canonize_around_

    def record(*args, **kwargs):
        calls.append(kwargs["which"])
        return original(*args, **kwargs)

    monkeypatch.setattr(state, "canonize_around_", record)
    state.canonize_around_qubits_((0, 1))
    assert calls == ["any"]
    assert state.is_subtree_canonical_form()


@pytest.mark.parametrize("n", [16, 64])
def test_overlapping_region_visits_do_not_grow_with_exterior(n, monkeypatch):
    plan = TreePlan.from_order(range(n), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    old = state.steiner_nodes([plan.node_of_qubit[q] for q in (0, 1, 2)])
    new = state.steiner_nodes([plan.node_of_qubit[q] for q in (0, 1)])
    state.canonize_subtree_(old)
    moves, neighbor_queries = [], []
    canonize, neighbors = state.canonize_edge_, state.neighbors

    def record(a, b, **kwargs):
        moves.append((a, b))
        return canonize(a, b, **kwargs)

    def query(node):
        neighbor_queries.append(node)
        return neighbors(node)

    monkeypatch.setattr(state, "canonize_edge_", record)
    monkeypatch.setattr(state, "neighbors", query)
    state.canonize_subtree_(new)
    assert len(moves) == len(old - new) == 3
    assert set(neighbor_queries) <= old
    assert len(neighbor_queries) < 20 * len(old)
    assert state.is_subtree_canonical_form()


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "sdcr"])
def test_replay_reuses_overlapping_canonical_region(mode, monkeypatch):
    opt = TreeOptimizer(None, state=_state(), mode=mode, chi=64, cutoff=0.,
                        compression_seed=11, track_infidelity=False, run=False)
    plan = opt.plan
    old = opt.tn.steiner_nodes([plan.node_of_qubit[q] for q in (0, 1, 2)])
    opt.tn.canonize_subtree_(old)
    before = opt.to_dense().copy()

    def no_global(*args, **kwargs):
        raise AssertionError("replay revisited the full exterior")

    monkeypatch.setattr(opt.tn, "canonize_around_", no_global)
    opt.apply_gate(np.eye(4, dtype=complex), (0, 1), track_norm=False)
    assert opt.tn.is_canonical_form()
    np.testing.assert_allclose(opt.to_dense(), before, atol=1e-10)


@pytest.mark.parametrize("caller", ["state", "optimizer"])
@pytest.mark.parametrize("shape", ["path", "branch"])
@pytest.mark.parametrize("backend", ["numpy", "torch", "native"])
def test_rounding_keeps_cut_order_and_only_required_return_moves(caller, shape, backend, monkeypatch):
    opt = TreeOptimizer(None, state=_state(backend), chi=2, run=False)
    state, plan = opt.tn, opt.plan
    if shape == "path":
        path = plan.node_path(plan.node_of_qubit[0], plan.node_of_qubit[7])
        nodes, hub = set(path), path[0]
    else:
        nodes, hub = set(plan.nodes()), plan.root
    state.shift_orthogonality_center(hub)
    reference = state.copy()
    expected_edges, returns = [], []

    def legacy_round(node, parent):
        for child in sorted(v for v in reference.neighbors(node) if v in nodes and v != parent):
            expected_edges.append((child, node))
            reference.compress_edge_(child, node, max_bond=2, cutoff=1e-12,
                                     cutoff_mode="rsum2", absorb="left", reduced="right")
            legacy_round(child, node)
            reference.canonize_edge_(child, node)

    legacy_round(hub, None)
    actual_edges = []
    compress, canonize = state.compress_edge_, state.canonize_edge_

    def checked_compress(a, b, **kwargs):
        assert state.is_subtree_canonical_form({a, b}, tol=1e-10)
        # Native charge alignment may conservatively reject a local proof;
        # its existing full graded-SVD fallback remains deliberately intact.
        if not state.fermionic:
            assert state.can_skip_canonize(a, b, absorb="right")
        actual_edges.append((a, b))
        result = compress(a, b, **kwargs)
        assert state.is_canonical_form(tol=1e-10)
        return result

    def record_return(a, b, **kwargs):
        returns.append((a, b))
        return canonize(a, b, **kwargs)

    monkeypatch.setattr(state, "compress_edge_", checked_compress)
    monkeypatch.setattr(state, "canonize_edge_", record_return)
    if caller == "optimizer":
        records = opt._round_successive_subtree(nodes, hub, max_bond=2, cutoff=1e-12)
        assert [(a, b) for a, b, *_ in records] == expected_edges
    else:
        state._round_successive_region(nodes, hub, max_bond=2, cutoff=1e-12,
                                       cutoff_mode="rsum2")
    assert actual_edges == expected_edges
    assert len(returns) == (0 if shape == "path" else len(expected_edges))
    assert state.orthogonality_center == (path[-1] if shape == "path" else hub)
    state.validate_isometry_metadata()
    np.testing.assert_allclose(state.to_statevector(), reference.to_statevector(), atol=1e-10)


@pytest.mark.parametrize("mode", ["src-oversample", "sdc-oversample", "sdcr-oversample", "zipup-oversample"])
def test_oversampled_replay_finishes_at_last_path_cut(mode):
    opt = TreeOptimizer(None, state=_state(), mode=mode, chi=2, cutoff=1e-12,
                        compression_seed=17, run=False)
    rng = np.random.default_rng(19)
    for support in [(0, 7), (1, 6), (0, 5)]:
        start = len(opt.truncation_history)
        gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        opt.apply_gate(gate, support)
        cuts = [e for e in opt.truncation_history[start:]
                if e["kind"] == mode.replace("-", "_")]
        path = opt.plan.node_path(*(opt.plan.node_of_qubit[q] for q in support))
        assert len(cuts) == len(path) - 1
        assert opt.center == cuts[-1]["edge"][0]
        assert opt.tn.is_canonical_form()
        opt.validate_isometry_metadata()
