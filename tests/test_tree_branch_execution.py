"""Branch compression retains its final center without a trailing QR walk."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import Fermion, TreeOptimizer, TreePlan, TreeTensorNetwork, ps_to_ttn
from pepsy.fitting.tree import _region_path


def _returning_branch(opt, nodes, hub, **kwargs):
    """Reference sweep: visit every edge, explicitly return after every child."""
    assert _region_path(opt.tn, nodes) is None
    opt._move_center(hub)

    def visit(node, parent):
        for child in sorted(set(opt.tn.neighbors(node)) & set(nodes) - {parent}):
            cutoff = kwargs.get("cutoff")
            if cutoff is None:
                cutoff = opt.cutoff
            if kwargs.get("preserve_subcap", True):
                cutoff = opt._subtree_cutoff_for_size(
                    opt.tn.ind_size(opt.tn.bond(node, child)),
                    max_bond=kwargs.get("max_bond"), cutoff=cutoff,
                )
            reduced, proven = opt._edge_reduction(
                node, child, max_bond=kwargs.get("max_bond"), cutoff=cutoff,
            )
            opt._compress_edge_compat(
                node, child, max_bond=kwargs.get("max_bond"), cutoff=cutoff,
                reduced=reduced, reduction_proven=proven,
            )
            visit(child, node)
            opt.tn.canonize_edge_(child, node, absorb="right")

    visit(hub, None)
    opt.center = hub


def _trace_branch(opt, monkeypatch, *, legacy=False):
    subtree = opt._compress_subtree
    edge = opt._compress_edge_compat
    canonize = opt.tn.canonize_edge_
    trace = {"active": False, "cutting": False, "events": []}

    def traced_subtree(nodes, hub, **kwargs):
        trace.update(active=True, hub=hub, nodes=frozenset(nodes))
        try:
            if legacy:
                return _returning_branch(opt, nodes, hub, **kwargs)
            return subtree(nodes, hub, **kwargs)
        finally:
            trace["active"] = False

    def traced_edge(u, v, **kwargs):
        assert opt.center == u
        trace["events"].append(("cut", u, v))
        trace["cutting"] = True
        try:
            result = edge(u, v, **kwargs)
        finally:
            trace["cutting"] = False
        assert opt.center == v
        assert opt.tn.is_canonical_form(v, tol=1e-10)
        opt.validate_isometry_metadata()
        return result

    def traced_canonize(u, v, **kwargs):
        if trace["active"] and not trace["cutting"]:
            trace["events"].append(("return", u, v))
        return canonize(u, v, **kwargs)

    monkeypatch.setattr(opt, "_compress_subtree", traced_subtree)
    monkeypatch.setattr(opt, "_compress_edge_compat", traced_edge)
    monkeypatch.setattr(opt.tn, "canonize_edge_", traced_canonize)
    return trace


def _check_final_descent(candidate, reference, actual, old):
    last_cut = max(i for i, event in enumerate(old["events"]) if event[0] == "cut")
    assert actual["events"] == old["events"][:last_cut + 1]
    assert len([e for e in actual["events"] if e[0] == "cut"]) == len(actual["nodes"]) - 1
    final = actual["events"][-1][2]
    assert candidate.center == candidate.tn.orthogonality_center == final
    assert candidate.canonical_region == frozenset({final})
    assert reference.center == old["hub"]
    path = candidate.plan.node_path(final, actual["hub"])
    assert old["events"][last_cut + 1:] == [
        ("return", u, v) for u, v in zip(path, path[1:])
    ]
    assert len(path) > 1
    candidate.tn.validate(check_canonical=True, tol=1e-10)
    candidate.validate_isometry_metadata()
    np.testing.assert_allclose(candidate.to_dense(), reference.to_dense(), atol=2e-11)
    assert candidate.norm() == pytest.approx(reference.norm(), abs=2e-12)
    assert candidate.get_norm_events()[-1]["local_fidelity"] == pytest.approx(
        reference.get_norm_events()[-1]["local_fidelity"], abs=2e-12,
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mode", ["direct", "dm"])
@pytest.mark.parametrize("arity", [2, 3])
@pytest.mark.parametrize("chi,cutoff", [(2, 0.), (2, 0.01), (64, 0.), (64, 0.1)])
def test_branch_skips_only_final_return(backend, mode, arity, chi, cutoff, monkeypatch):
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=arity)
    state = TreeTensorNetwork.rand(plan, D=3, seed=54, dtype="complex128")
    rng = np.random.default_rng(65)
    gate = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))[0]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
        gate = torch.as_tensor(gate)
    candidate, reference = [
        TreeOptimizer(None, state=state, mode=mode, chi=chi, cutoff=cutoff, run=False)
        for _ in range(2)
    ]
    candidate.normalize()
    reference.normalize()
    actual = _trace_branch(candidate, monkeypatch)
    old = _trace_branch(reference, monkeypatch, legacy=True)
    candidate.apply_gate(gate, (0, 2, 7))
    reference.apply_gate(gate, (0, 2, 7))
    _check_final_descent(candidate, reference, actual, old)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mode", ["direct", "dm"])
@pytest.mark.parametrize("hub", [10, 40, 7, 80])
@pytest.mark.parametrize("preserve_subcap", [False, True])
def test_uneven_branch_from_arbitrary_hub(
    backend, mode, hub, preserve_subcap, monkeypatch,
):
    # Irregular ids, unary nodes, a physical root, and untouched exterior legs.
    plan = TreePlan.from_children(
        {40: (10, 80), 10: (3, 52), 3: (7, 9), 52: (60,),
         80: (20, 90), 20: (21, 22), 7: (), 9: (), 60: (),
         21: (), 22: (), 90: ()},
        {7: 4, 9: 1, 60: 6, 21: 2, 22: 5, 90: 3}, root=40, root_qubit=0,
    )
    state = TreeTensorNetwork.rand(plan, D=3, seed=15, dtype="complex128")
    state.multiply_(1 / state.norm())
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
    candidate, reference = [
        TreeOptimizer(None, state=state, mode=mode, chi=1, cutoff=0., run=False)
        for _ in range(2)
    ]
    actual = _trace_branch(candidate, monkeypatch)
    old = _trace_branch(reference, monkeypatch, legacy=True)
    nodes = state.steiner_nodes((7, 60, 22))
    assert _region_path(state, nodes) is None
    for opt in (candidate, reference):
        with opt._update("subtree", (4, 6, 5)):
            opt._compress_subtree(
                nodes, hub, max_bond=3, cutoff=.1, preserve_subcap=preserve_subcap,
            )
    _check_final_descent(candidate, reference, actual, old)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mode", ["direct", "dm", "dmrg2", "dmrg3"])
@pytest.mark.parametrize("chi", [2, 64])
def test_branch_followed_by_other_supports_keeps_local_norm(backend, mode, chi):
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=81, dtype="complex128")
    state.multiply_(1 / state.norm())
    expected = state.to_dense().reshape(-1)
    convert = np.asarray
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda a: torch.as_tensor(a, dtype=torch.complex128)
        state.apply_to_arrays(convert)
    opt = TreeOptimizer(None, state=state, mode=mode, chi=chi, cutoff=0.,
                        fit_init_strategy="guess-direct", fit_rtol=None, run=False)
    rng = np.random.default_rng(92)
    # Branch, crossing path, single site, a different branch, then short path.
    for where in ((0, 2, 7), (1, 6), (3,), (1, 4, 6), (4, 5)):
        n = 2 ** len(where)
        gate = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
        axes = where + tuple(q for q in range(8) if q not in where)
        block = expected.reshape((2,) * 8).transpose(axes).reshape(n, -1)
        expected = (gate @ block).reshape((2,) * 8).transpose(np.argsort(axes)).reshape(-1)
        before = np.linalg.norm(opt.to_dense())
        opt.apply_gate(convert(gate), where)
        dense = opt.to_dense().reshape(-1)
        assert opt.norm() == pytest.approx(np.linalg.norm(dense), abs=2e-11)
        opt.tn.validate(check_canonical=True, tol=1e-10)
        opt.validate_isometry_metadata()
        if mode in {"direct", "dm"}:
            assert opt.get_norm_events()[-1]["local_fidelity"] == pytest.approx(
                (np.linalg.norm(dense) / before) ** 2, abs=2e-11,
            )
        if chi == 64:
            np.testing.assert_allclose(dense, expected, atol=2e-11)
    events = opt.get_norm_events()
    assert opt.norm_diagnostics()["cumulative_fidelity"] == pytest.approx(
        np.prod([event["local_fidelity"] for event in events]), abs=2e-12,
    )


@pytest.mark.parametrize("chi", [2, 32])
def test_native_branch_retains_graded_final_center(chi, monkeypatch):
    pytest.importorskip("symmray")
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    fermion = Fermion(spinful=False, symmetry="U1", dtype="complex128")
    state = ps_to_ttn(8, tree=plan, fermion=fermion, occupations=(0, 1) * 4)
    seed = TreeOptimizer(None, state=state, chi=32, cutoff=0., run=False)
    hopping = fermion.hopping_gate(.3, t=1., imaginary=False)
    for pair in ((0, 7), (1, 6), (2, 5), (3, 4)):
        seed.apply_2q(hopping, *pair)
    candidate, reference = [
        TreeOptimizer(None, state=seed.tn, chi=chi, cutoff=0., run=False)
        for _ in range(2)
    ]
    actual = _trace_branch(candidate, monkeypatch)
    old = _trace_branch(reference, monkeypatch, legacy=True)
    sites = (0, 2, 7)
    local_ops = [fermion.onsite_gate(.07, site=q, mu=.4, imaginary=False) for q in sites]
    operator = qtn.MPO_product_operator(local_ops, sites=sites, L=8)
    candidate.apply_submpo(operator, sites)
    reference.apply_submpo(operator, sites)
    _check_final_descent(candidate, reference, actual, old)
    # A following path crosses the old hub. Graded center moves must remain valid.
    monkeypatch.undo()
    reference._move_center(candidate.center)
    candidate.apply_2q(hopping, 1, 6)
    reference.apply_2q(hopping, 1, 6)
    candidate.tn.validate(check_canonical=True, tol=1e-10)
    candidate.validate_isometry_metadata()
    np.testing.assert_allclose(candidate.to_dense(), reference.to_dense(), atol=2e-11)


@pytest.mark.parametrize("chi", [2, 64])
def test_branch_final_center_preserves_torch_state_gradients(chi, monkeypatch):
    torch = pytest.importorskip("torch")
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=17, dtype="float64")
    state.multiply_(1 / state.norm())
    state.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True))
    parameters = tuple(t.data for t in state.tensors)
    candidate, reference = [
        TreeOptimizer(None, state=state, chi=chi, cutoff=0., run=False)
        for _ in range(2)
    ]
    monkeypatch.setattr(reference, "_compress_subtree", lambda nodes, hub, **kwargs:
                        _returning_branch(reference, nodes, hub, **kwargs))
    gate = torch.as_tensor(np.linalg.qr(np.random.default_rng(39).normal(size=(8, 8)))[0])
    gradients = []
    for opt in (candidate, reference):
        opt.apply_gate(gate, (0, 2, 7))
        vector = opt.tn.to_dense().reshape(-1)
        weights = torch.linspace(.5, 1.5, vector.numel(), dtype=vector.dtype)
        loss = (weights * vector.square()).sum()
        gradients.append(torch.autograd.grad(loss, parameters, retain_graph=True))
    torch.testing.assert_close(candidate.tn.to_dense(), reference.tn.to_dense(),
                               atol=2e-11, rtol=2e-11)
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mode", ["direct", "dm"])
@pytest.mark.parametrize("exponent", [-400., 400.])
@pytest.mark.parametrize("tracking", [False, True])
def test_branch_final_center_stabilizes_extreme_scale(backend, mode, exponent, tracking):
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=3, seed=54, dtype="complex128")
    state.multiply_(1 / state.norm())
    rng = np.random.default_rng(65)
    gate = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))[0]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
        gate = torch.as_tensor(gate)
    state.exponent = exponent
    raw = TreeOptimizer(None, state=state, mode=mode, chi=2, cutoff=0., run=False)
    stable = TreeOptimizer(None, state=state, mode=mode, chi=2, cutoff=0., run=False,
                           stabilize_unitary=True, track_infidelity=tracking)
    vectors = []
    for opt in (raw, stable):
        opt.apply_gate(gate, (0, 2, 7))
        assert opt.tn.exponent == exponent
        opt.tn.validate(check_canonical=True, tol=1e-10)
        opt.validate_isometry_metadata()
        stripped = opt.tn.copy()
        stripped.exponent = 0.
        vectors.append(stripped.to_statevector())
    retained_norm = np.linalg.norm(vectors[0])
    assert retained_norm < .999  # The update must actually lose norm.
    assert stable.center == raw.center
    np.testing.assert_allclose(vectors[1], vectors[0] / retained_norm, atol=2e-11)
    assert np.linalg.norm(vectors[1]) == pytest.approx(1., abs=2e-12)
    if tracking:
        assert stable.get_norm_events()[-1]["local_fidelity"] == pytest.approx(
            retained_norm ** 2, abs=2e-12,
        )
    else:
        assert not stable.get_norm_events()
