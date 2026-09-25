"""Endpoint routing and FIT sweeps on a tree's induced active path."""

import numpy as np
import pytest

from pepsy import TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.fitting import TreeFIT
from pepsy.fitting.tree import _region_path


def _problem():
    # Permuted physical labels, an operated physical root, and exterior legs
    # on the path all matter: neither arity nor global degree identifies it.
    plan = TreePlan.from_order((7, 2, 5, 1, 6, 3, 4), root_qubit=0,
                               structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=81)
    path = tuple(plan.node_path(plan.node_of_qubit[7], plan.node_of_qubit[4]))
    return plan, state, path


def test_active_path_uses_induced_geometry_and_incoming_center():
    plan, state, path = _problem()
    assert plan.root in path
    assert any(len(state.neighbors(n)) > 2 for n in path)
    state.shift_orthogonality_center(path[-1])
    assert _region_path(state, path) == path[::-1]
    assert _region_path(state, (path[2],)) == (path[2],)
    assert _region_path(state, (path[0], path[-1])) is None
    branch = state.steiner_nodes([plan.node_of_qubit[q] for q in (7, 2, 4)])
    assert _region_path(state, branch) is None
    state.invalidate_canonical_form()
    expected = path if path[0] < path[-1] else path[::-1]
    assert _region_path(state, path) == expected


@pytest.mark.parametrize("size,sequence", [
    (1, "inward-outward"), (2, "inward-outward"), (3, "inward-outward"),
    (1, "LR"), (2, "LR"), (3, "LR"),
])
def test_auto_fit_windows_advance_centers_without_interior_qr(size, sequence, monkeypatch):
    _, state, path = _problem()
    state.shift_orthogonality_center(path[0])
    fit = TreeFIT(state, state, traversal="auto", max_bond=4, cutoffs=0.)
    update = fit.fit_block
    updates = []
    expected_paths = (path, path[::-1]) if sequence != "LR" else (path[::-1], path)

    def checked(block, *, center=None, **kwargs):
        if updates and size > 1:
            assert fit.p.orthogonality_center in block
        result = update(block, center=center, **kwargs)
        updates.append((tuple(block), fit.p.orthogonality_center))
        # Assert numerical isometries, not just trusted center metadata.
        assert fit.p.is_canonical_form(center)
        return result

    monkeypatch.setattr(fit, "fit_block", checked)
    fit.run_gate(path, n_iter=2, block_size=size, sweep_sequence=sequence)
    expected = [(p[i:i + size], p[i + size - 1])
                for p in expected_paths for i in range(len(path) - size + 1)] * 2
    assert updates == expected
    assert fit.fit_diagnostics()["path_endpoints"] == (path[0], path[-1])
    assert fit.final_center_site == expected_paths[-1][-1]
    np.testing.assert_allclose(fit.p.to_dense(), state.to_dense(), atol=1e-11)


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "zipup", "dmrg2", "dmrg3"])
def test_path_replay_preserves_logical_gate_order_and_scale(mode):
    plan, state, path = _problem()
    state.exponent = 2.
    # Nonunitary and nonsymmetric, with reversed arguments and a physical
    # root in the middle. Its three-site Steiner region is still a path.
    where = (4, 0, 7)
    gate = np.random.default_rng(33).normal(size=(8, 8)).astype(complex) / 8
    axes = where + tuple(q for q in range(8) if q not in where)
    dense = state.to_dense().reshape((2,) * 8).transpose(axes).reshape(8, -1)
    expected = (gate @ dense).reshape((2,) * 8).transpose(np.argsort(axes)).reshape(-1)
    opt = TreeOptimizer(None, state=state, mode=mode, chi=256, cutoff=0.,
                        fit_rtol=None, run=False)
    opt.apply_gate(gate, where, track_norm=False)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=2e-10)
    assert opt.tn.is_canonical_form(opt.center)
    assert opt.tn.exponent == 2.
    if mode.startswith("dmrg"):
        diag = opt.get_fit_diagnostics()
        assert diag["resolved_traversal"] == "path"
        assert set(diag["path_endpoints"]) == {path[0], path[-1]}
    assert opt.tn.plan.node_of_qubit == plan.node_of_qubit


@pytest.mark.parametrize("route", ["treempo", "operator", "submpo"])
def test_direct_routes_complete_operator_then_compresses_once(route, monkeypatch):
    import quimb.tensor as qtn

    plan, state, path = _problem()
    state.shift_orthogonality_center(path[0])
    gate = np.linalg.qr(np.random.default_rng(20).normal(size=(8, 8)))[0].astype(complex)
    where = (7, 0, 4)
    opt = TreeOptimizer(None, state=state, mode="direct", chi=2, cutoff=0., run=False)
    routed = []
    routing_done = False
    compressed = []
    canonized = []
    routing = opt._route_subtree_messages
    compress = opt._compress_edge_compat
    canonize = opt.tn.canonize_edge_

    def record_route(local, state_inds, op_inds, order, **kwargs):
        nonlocal routing_done
        if route == "treempo":
            assert all(isinstance(layers, list) and len(layers) == 2
                       for layers in local.values())
        routed.extend(order)
        result = routing(local, state_inds, op_inds, order, **kwargs)
        routing_done = True
        return result

    def record_compress(u, v, **kwargs):
        assert routing_done
        assert {frozenset(e) for e in routed} == {
            frozenset(e) for e in zip(path, path[1:])
        }
        assert len(routed) == len(path) - 1
        compressed.append((u, v))
        return compress(u, v, **kwargs)

    def record_canonize(u, v, **kwargs):
        canonized.append((u, v))
        return canonize(u, v, **kwargs)

    monkeypatch.setattr(opt, "_route_subtree_messages", record_route)
    monkeypatch.setattr(opt, "_compress_edge_compat", record_compress)
    monkeypatch.setattr(opt.tn, "canonize_edge_", record_canonize)
    if route == "treempo":
        opt.apply_subtreempo(TreeMPO.from_gate(plan, gate, where))
    elif route == "operator":
        opt.apply_subtree_operator(gate, where)
    else:
        mpo = qtn.MatrixProductOperator.from_dense(gate, sites=where, L=8, cutoff=0.)
        opt.apply_submpo(mpo, where)
    assert compressed == list(zip(path[::-1], path[-2::-1]))
    # Metadata-only hub recovery may call canonize_edge_ before compression;
    # a return walk would leave the center at the wrong end afterwards.
    assert opt.center == path[0]
    assert len(canonized) <= 2 * (len(path) - 1)
    assert opt.max_bond() <= 2
    assert opt.tn.is_canonical_form(opt.center)


@pytest.mark.parametrize("sequence", ["inward-outward", "outward-inward"])
@pytest.mark.parametrize("guess", ["guess-src", "guess-sdc", "guess-direct", "guess-zipup"])
def test_guess_finishes_at_frozen_first_fit_endpoint(sequence, guess, monkeypatch):
    _, state, path = _problem()
    state.shift_orthogonality_center(path[0])
    optimizer = TreeOptimizer(None, state=state, mode="dmrg3", chi=4, cutoff=0.,
                              fit_sweep_sequence=sequence, fit_init_strategy=guess,
                              fit_rtol=None, run=False)
    first = path[0] if sequence == "inward-outward" else path[-1]
    update = TreeFIT.fit_block
    entries = []

    def checked(fit, block, **kwargs):
        if not entries:
            assert fit.p.orthogonality_center == first
        entries.append(tuple(block))
        return update(fit, block, **kwargs)

    monkeypatch.setattr(TreeFIT, "fit_block", checked)
    gate = np.linalg.qr(np.random.default_rng(92).normal(size=(4, 4)))[0].astype(complex)
    optimizer.apply_gate(gate, (7, 4))
    assert entries[0][0] == first
    diag = optimizer.get_fit_diagnostics()
    assert diag["path_endpoints"] == (path[0], path[-1])
    assert diag["block_size_trace"] == (3, 3, 2, 1)


def test_frozen_invalid_path_rejected_before_mutating_state():
    _, state, path = _problem()
    fit = TreeFIT(state, state, traversal="auto", inplace=True)
    before = state.to_dense().copy()
    center = state.orthogonality_center
    with pytest.raises(ValueError, match="path order"):
        fit.run_gate(path, _path_order=path[::2] + path[1::2])
    assert state.orthogonality_center == center
    np.testing.assert_array_equal(state.to_dense(), before)


def test_failed_streamed_path_does_not_install_partial_operator(monkeypatch):
    _, state, _ = _problem()
    gate = np.linalg.qr(np.random.default_rng(93).normal(size=(4, 4)))[0].astype(complex)
    opt = TreeOptimizer(None, state=state, chi=2, cutoff=0., run=False)
    before = opt.to_dense().copy()
    route = opt._qr_route_message
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected path QR failure")
        return route(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(opt, "_qr_route_message", fail)
        with pytest.raises(RuntimeError, match="injected path"):
            opt.apply_gate(gate, (7, 4))
    np.testing.assert_allclose(opt.to_dense(), before, atol=1e-11)
    assert opt._active_update is None
    expected = TreeOptimizer([(gate, (7, 4))], state=state, chi=2, cutoff=0.)
    opt.apply_gate(gate, (7, 4))
    np.testing.assert_allclose(opt.to_dense(), expected.to_dense(), atol=1e-10)


@pytest.mark.parametrize("target", ["zero", "weak"])
def test_short_path_fallback_preserves_zero_and_weak_targets(target):
    plan = TreePlan.from_order((1,), root_qubit=0)
    state = TreeTensorNetwork.from_plan(plan)
    gate = np.zeros((4, 4), dtype=complex)
    if target == "weak":
        gate[:, 0] = (1., 0., 0., 1e-10)
    opt = TreeOptimizer(None, state=state, mode="dmrg3", chi=2, cutoff=0.,
                        fit_rtol=None, run=False)
    opt.apply_gate(gate, (0, 1), track_norm=False)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), gate[:, 0], atol=1e-13)
    assert opt.get_fit_diagnostics()["block_size_trace"] == (2, 2, 1, 1)
    assert opt.tn.is_canonical_form(opt.center)


def test_unknown_center_orientation_is_frozen_before_norm_preparation():
    # The smallest interior id is nearer endpoint 9, but an unknown center
    # must choose endpoint 1 deterministically before norm recovery moves it.
    plan = TreePlan.from_children(
        {0: (9, 6), 9: (), 6: (1, 8), 1: (), 8: ()},
        {9: 0, 1: 1, 8: 2}, root=0,
    )
    state = TreeTensorNetwork.rand(plan, D=2, seed=5)
    opt = TreeOptimizer(None, state=state, mode="dmrg2", chi=8, cutoff=0., run=False)
    opt.tn.invalidate_canonical_form()
    opt.apply_gate(np.eye(4, dtype=complex), (0, 1))
    assert opt.get_fit_diagnostics()["path_endpoints"] == (1, 9)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), state.to_dense().reshape(-1),
                               atol=1e-11)


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc"])
@pytest.mark.parametrize("gauge", ["inside", "outside", "region", "unknown"])
def test_operator_preparation_only_gauges_the_exterior(mode, gauge, monkeypatch):
    plan, state, path = _problem()
    if gauge == "outside":
        state.shift_orthogonality_center(plan.node_of_qubit[2])
    elif gauge == "region":
        state.canonize_subtree_(path[1:-1])
    else:
        state.shift_orthogonality_center(plan.root)
    opt = TreeOptimizer(None, state=state, mode=mode, chi=64, cutoff=0., run=False)
    if gauge == "unknown":
        opt.tn.invalidate_canonical_form()
    gate = np.linalg.qr(np.random.default_rng(102).normal(size=(4, 4)))[0].astype(complex)
    axes = (7, 4, 0, 1, 2, 3, 5, 6)
    dense = state.to_dense().reshape((2,) * 8).transpose(axes).reshape(4, -1)
    expected = (gate @ dense).reshape((2,) * 8).transpose(np.argsort(axes)).reshape(-1)
    canonize = opt.tn.canonize_edge_
    prepare = opt.tn.canonize_subtree_
    moves, full_preparations, entries = [], [], []

    def record_move(u, v, **kwargs):
        moves.append((u, v))
        return canonize(u, v, **kwargs)

    def record_prepare(nodes, **kwargs):
        full_preparations.append(frozenset(nodes))
        return prepare(nodes, **kwargs)

    method = "_route_subtree_messages" if mode in {"direct", "dm"} else "_successive_subtree_messages"
    route = getattr(opt, method)

    def check_preparation(*args, **kwargs):
        entries.append(opt.center)
        assert opt.tn.is_subtree_canonical_form(path)
        if gauge in {"inside", "region"}:
            assert not moves and not full_preparations
        elif gauge == "outside":
            assert moves and not full_preparations
            assert all(u not in path for u, _ in moves)
            assert moves[-1][1] in path
        else:
            assert full_preparations == [frozenset(path)]
        return route(*args, **kwargs)

    monkeypatch.setattr(opt.tn, "canonize_edge_", record_move)
    monkeypatch.setattr(opt.tn, "canonize_subtree_", record_prepare)
    monkeypatch.setattr(opt, method, check_preparation)
    opt.apply_gate(gate, (7, 4), track_norm=False)
    assert len(entries) == 1
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=2e-11)
    assert opt.tn.is_canonical_form(opt.center)


@pytest.mark.parametrize("where", [(7, 4), (7, 2, 4)])
def test_parallel_direct_waves_preserve_truncated_path_and_branch_states(where):
    _, state, _ = _problem()
    size = 2 ** len(where)
    rng = np.random.default_rng(103)
    gate = np.linalg.qr(rng.normal(size=(size, size))
                        + 1j * rng.normal(size=(size, size)))[0]
    serial = TreeOptimizer([(gate, where)], state=state, mode="direct", chi=2,
                           cutoff=0., subtree_workers=1)
    parallel = TreeOptimizer([(gate, where)], state=state, mode="direct", chi=2,
                             cutoff=0., subtree_workers=3)
    np.testing.assert_allclose(parallel.to_dense(), serial.to_dense(), atol=2e-11)
    assert serial.max_bond() <= 2 and parallel.max_bond() <= 2
    assert serial.center == parallel.center
    assert parallel.tn.is_canonical_form(parallel.center)


def test_path_profile_includes_planning_and_identifies_deferred_merges(monkeypatch):
    from pepsy.optimizers.tree import optimizer as module

    plan, state, path = _problem()
    opt = TreeOptimizer(None, state=state, chi=4, cutoff=0., profile=True, run=False)
    operator = TreeMPO.from_gate(plan, np.eye(4, dtype=complex), (7, 4))
    clock = [0.]
    peel = opt._peel_order

    def timed_peel(*args, **kwargs):
        clock[0] += 1.
        return peel(*args, **kwargs)

    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(opt, "_peel_order", timed_peel)
    opt.apply_subtreempo(operator)
    events = opt.profile_report()["events"]
    planning = [e for e in events if e["kind"] == "metadata_path"]
    assert len(planning) == 1 and planning[0]["seconds"] == 1.
    merges = [e for e in events if e["kind"] == "subtree_hub_merge"]
    assert merges and all(e["deferred"] for e in merges)
    absorptions = [e for e in events if e.get("route") == "subtreempo_layered"]
    assert len(absorptions) == len(path)
