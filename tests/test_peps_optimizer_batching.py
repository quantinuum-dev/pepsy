"""Exact gate batching and metric regressions for the PEPS stream driver."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer
from pepsy.optimizers.peps import optimizer as peps_mod
from pepsy.optimizers.sweep import optimizer as sweep_mod
from pepsy.boundary import metrics as boundary_metrics

pytestmark = [pytest.mark.core, pytest.mark.peps, pytest.mark.integration]


def _product(lx=2, ly=3):
    return qtn.PEPS.product_state([[np.array([1., 0.], complex)] * ly] * lx)


def _unitary():
    rng = np.random.default_rng(31)
    return np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]


def _dense_apply(vector, gate, sites):
    n = int(np.log2(vector.size))
    order = list(sites) + [i for i in range(n) if i not in sites]
    data = vector.reshape((2,) * n).transpose(order).reshape(2 ** len(sites), -1)
    return (gate @ data).reshape((2,) * n).transpose(np.argsort(order)).reshape(-1)


@pytest.mark.parametrize("diagonal,factor", [(True, 2), (False, 4)])
def test_exact_target_rank_and_auto_budget_follow_gate_structure(diagonal, factor):
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=17)
    gate = np.diag(np.exp(-.3j * np.array([1, -1, -1, 1]))) if diagonal else _unitary()
    gates = [(gate, ((1, 1), (1, 2))), (gate, ((0, 0), (0, 1)))]
    opt = PepsOptimizer(state, gates, chi=2, contraction_opt="greedy", boundary_convergence=False)
    entries, count, next_idx, target, stop, limit = opt._collect_auto_batch_target(
        0, cutoff=1e-12, cutoff_mode="rsum2", gate_kwargs=None,
    )
    assert len(entries) == count == next_idx == 2
    assert stop == "end_of_queue"
    assert target.max_bond() == limit == factor * opt.chi
    expected = state.to_dense().reshape(-1)
    for sites in [(4, 5), (0, 1)]:
        expected = _dense_apply(expected, gate, sites)
    np.testing.assert_allclose(target.to_dense().reshape(-1), expected, atol=2e-12)


@pytest.mark.optional
def test_diagonal_rank_bound_on_torch_preserves_dense_state():
    torch = pytest.importorskip("torch")
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=17)
    state.apply_to_arrays(torch.as_tensor)
    gate = torch.diag(torch.exp(-.3j * torch.tensor([1., -1., -1., 1.], dtype=torch.float64)))
    opt = PepsOptimizer(state, chi=2, contraction_opt="greedy", boundary_convergence=False)
    target = opt._build_target(
        state, gate.reshape(2, 2, 2, 2), ((1, 1), (1, 2)), None,
        cutoff=1e-12, cutoff_mode="rsum2", gate_kwargs=None,
    )
    assert target.max_bond() == 4
    expected = _dense_apply(state.to_dense().numpy().reshape(-1), gate.numpy(), (4, 5))
    np.testing.assert_allclose(target.to_dense().numpy().reshape(-1), expected, atol=2e-12)
    # Trainable zeros need not have zero off-diagonal derivatives.
    assert opt._gate_bond_factor(gate.requires_grad_()) == 4


@pytest.mark.parametrize("path", ["within_chi", "warmstart", "optimized", "optimizer_rejected", "one_site"])
def test_retained_output_is_normalized_without_rescaling_unitary_target(path, monkeypatch):
    state = _product(2, 2)
    gates = [(_unitary(), ((0, 0), (0, 1)))]
    if path == "one_site":
        gates = [(np.array([[0, 1], [1, 0]], complex), (0, 0))]
    opt = PepsOptimizer(state, gates, chi=4 if path == "within_chi" else 1,
                        normalize_chi=16, normalize_initial=False, contraction_opt="greedy", boundary_convergence=False)
    normalized = []
    original = peps_mod.boundary_normalize
    def capture_norm(state, **kwargs):
        normalized.append((state.max_bond(), kwargs["chi"]))
        return original(state, **kwargs)
    monkeypatch.setattr(peps_mod, "boundary_normalize", capture_norm)
    def fake_optimize(warmstart, target, **kwargs):
        # The exact target has not been passed to the normalization routine.
        assert all(bond == 1 for bond, _ in normalized)
        return warmstart * 3, None, None
    monkeypatch.setattr(opt, "_optimize_state", fake_optimize)
    out = opt.run(optimize=path != "warmstart", accept_if_improved=path == "optimizer_rejected",
                  improvement_tol=1e-12)
    vector = out.to_dense().reshape(-1)
    assert np.vdot(vector, vector).real == pytest.approx(1., abs=2e-12)
    assert all(cap == 16 for _, cap in normalized)
    if path != "one_site":
        assert opt.step_records[-1]["reason"] == path
        if path != "within_chi":
            assert all(bond == 1 for bond, _ in normalized)


@pytest.mark.parametrize("batch,counts", [("auto", [4]), (1, [1, 1, 1, 1]),
                                         (2, [2, 2]), (3, [3, 1])])
def test_batches_preserve_dense_circuit_and_single_gate_order(batch, counts):
    """Shared sites do not split a batch or move intervening one-site gates."""
    state = _product()
    gate = _unitary()
    x = np.array([[0., 1.], [1., 0.]], complex)
    gates = [(gate, ((0, 0), (1, 0))), (x, (0, 0)),
             (gate, ((0, 1), (1, 1))), (gate, ((0, 2), (1, 2))),
             (gate, ((0, 0), (0, 1)))]
    vector = state.to_dense().reshape(-1)
    original = vector.copy()
    for payload, sites in zip([gate, x, gate, gate, gate], [(0, 3), (0,), (1, 4), (2, 5), (0, 1)]):
        vector = _dense_apply(vector, payload, sites)
    opt = PepsOptimizer(state, gates, chi=8, contraction_opt="greedy", boundary_convergence=False)
    # Omit the option in the auto case to exercise the actual public default.
    out = opt.run(**({} if batch == "auto" else {"k_2q_batch": batch}))
    np.testing.assert_allclose(out.to_dense().reshape(-1), vector, atol=2e-12)
    np.testing.assert_array_equal(state.to_dense().reshape(-1), original)
    assert [r["two_site_batch"] for r in opt.step_records] == counts
    if batch == "auto":
        assert opt.step_records[0]["batch_size"] == 5
        assert opt.step_records[0]["batch_stop_reason"] == "end_of_queue"
        assert all(r["target_max_bond"] <= 16 for r in opt.step_records)


@pytest.mark.optional
def test_auto_matches_index_and_coordinate_site_aliases_on_torch():
    """Shared sites specified by either alias preserve gate order in one target."""
    torch = pytest.importorskip("torch")
    state = _product(2, 2)
    state.apply_to_arrays(torch.as_tensor)
    gate = torch.as_tensor(_unitary())
    gates = [(gate, ((0, 0), (0, 1))), (gate, ("k0,0", "k1,0"))]
    opt = PepsOptimizer(state, gates, chi=8, contraction_opt="greedy", boundary_convergence=False)
    out = opt.run()
    vector = np.zeros(16, complex)
    vector[0] = 1
    vector = _dense_apply(_dense_apply(vector, gate.numpy(), (0, 1)), gate.numpy(), (0, 2))
    np.testing.assert_allclose(out.to_dense().numpy().reshape(-1), vector, atol=2e-12)
    assert [r["two_site_batch"] for r in opt.step_records] == [2]
    assert opt.step_records[0]["batch_stop_reason"] == "end_of_queue"
    assert next(iter(out)).data.dtype == torch.complex128


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_auto_second_order_layer_builds_one_exact_target_before_refinement(monkeypatch, backend):
    state = _product()
    xhalf = np.array([[np.cos(.07), -1j*np.sin(.07)],
                     [-1j*np.sin(.07), np.cos(.07)]])
    zz = np.diag(np.exp(-.13j * np.array([1., -1., -1., 1.])))
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.as_tensor)
        convert = torch.as_tensor
    else:
        convert = np.asarray
    sites = [(i, j) for i in range(2) for j in range(3)]
    edges = [(s, (s[0]+di, s[1]+dj)) for s in sites for di, dj in [(1, 0), (0, 1)]
             if s[0]+di < 2 and s[1]+dj < 3]
    gates = ([(convert(xhalf), s) for s in sites]
             + [(convert(zz), edge) for edge in edges]
             + [(convert(xhalf), s) for s in sites])
    expected = np.zeros(64, complex)
    expected[0] = 1
    for gate, where in ([(xhalf, (i,)) for i in range(6)]
                        + [(zz, tuple(sites.index(s) for s in e)) for e in edges]
                        + [(xhalf, (i,)) for i in range(6)]):
        expected = _dense_apply(expected, gate, where)
    opt = PepsOptimizer(state, gates, chi=1, contraction_opt="greedy", fit_mode="direct", boundary_convergence=False)
    targets = []

    def capture(warmstart, target, **kwargs):
        targets.append(target.copy())
        return warmstart, None, {}

    monkeypatch.setattr(opt, "_optimize_state", capture)
    output = opt.run(measure_infidelity=False, measure_final_infidelity=False)
    assert len(targets) == len(opt.step_records) == 1
    assert opt.step_records[0]["two_site_batch"] == len(edges) == 7
    assert opt.step_records[0]["step"] == len(gates)
    vector = targets[0].to_dense().reshape(-1)
    if backend == "torch":
        vector = vector.numpy()
    np.testing.assert_allclose(vector, expected, atol=2e-12)
    assert targets[0].max_bond() <= 2
    assert output.max_bond() == 1


def test_auto_5x6_d4_zz_layer_stays_within_d8():
    state = qtn.PEPS.rand(5, 6, bond_dim=4, dtype="complex128", seed=73)
    original = [tensor.data.copy() for tensor in state]
    gate = np.diag(np.exp(-.1j * np.array([1., -1., -1., 1.])))
    gates = [(gate, ((i, j), (i+di, j+dj))) for i in range(5) for j in range(6)
             for di, dj in [(1, 0), (0, 1)] if i+di < 5 and j+dj < 6]
    opt = PepsOptimizer(state, gates, chi=4, normalize_initial=False, contraction_opt="greedy", boundary_convergence=False)
    entries, count, next_idx, target, stop, limit = opt._collect_auto_batch_target(
        0, cutoff=1e-12, cutoff_mode="rsum2", gate_kwargs=None,
    )
    assert len(entries) == count == next_idx == len(gates) == 49
    assert stop == "end_of_queue"
    assert target.max_bond() == limit == 8
    for tensor, before in zip(state, original):
        np.testing.assert_array_equal(tensor.data, before)


def test_single_gate_can_exceed_auto_budget_without_truncating_target():
    """An oversized input still produces an exact target beyond the budget."""
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=17)
    gate = _unitary()
    opt = PepsOptimizer(state, [(gate, ((1, 1), (1, 2)))], chi=1,
                        contraction_opt="greedy", boundary_convergence=False)
    target = opt._collect_auto_batch_target(
        0, cutoff=1e-12, cutoff_mode="rsum2", gate_kwargs=None,
    )
    entries, n_two, next_idx, exact, stop, limit = target
    assert len(entries) == n_two == next_idx == 1
    assert stop == "single_gate_exceeds_limit"
    assert limit == 4 * opt.chi
    assert exact.max_bond() > limit
    expected = _dense_apply(state.to_dense().reshape(-1), gate, (4, 5))
    np.testing.assert_allclose(exact.to_dense().reshape(-1), expected, atol=2e-12)


def test_routed_batch_stops_at_bond_limit_without_consuming_next_gate():
    """Disjoint endpoints can share route bonds; the rejected gate stays queued."""
    state = _product()
    gate = _unitary()
    gates = [(gate, ((0, 0), (1, 2))), (gate, ((1, 0), (0, 2)))]
    auto = PepsOptimizer(state, gates, chi=1, contraction_opt="greedy", boundary_convergence=False)
    fixed = PepsOptimizer(state, gates, chi=1, contraction_opt="greedy", boundary_convergence=False)
    actual = auto.run(optimize=False)
    expected = fixed.run(k_2q_batch=1, optimize=False)
    np.testing.assert_allclose(actual.to_dense(), expected.to_dense(), atol=2e-12)
    assert auto.step_records[0]["batch_stop_reason"] == "bond_limit"
    assert auto.step_records[0]["target_max_bond"] <= 4
    assert [r["two_site_batch"] for r in auto.step_records] == [1, 1]
    assert [r["step"] for r in auto.step_records] == [1, 2]


@pytest.mark.parametrize("per_call", [False, True])
def test_bond_dim_alias_cannot_truncate_exact_target(per_call):
    state = _product(2, 2)
    gate = _unitary()
    opts = {"bond_dim": 1}
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=2,
                        gate_kwargs=None if per_call else opts, contraction_opt="greedy", boundary_convergence=False)
    out = opt.run(gate_kwargs=opts if per_call else None)
    expected = _dense_apply(state.to_dense().reshape(-1), gate, (0, 1))
    np.testing.assert_allclose(out.to_dense().reshape(-1), expected, atol=2e-12)
    assert opt.step_records[0]["target_max_bond"] == 2


def test_measured_target_norm_identity_update_retries_inconsistent_boundary_metrics():
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=17)
    before = state.to_dense().reshape(-1)
    opt = PepsOptimizer(state, [(np.eye(4), ((1, 1), (1, 2)))], chi=2,
                        contraction_opt="greedy", evaluation_negative_tol=0, boundary_convergence=False)
    with pytest.warns(RuntimeWarning, match="Invalid PEPS boundary infidelity"):
        out = opt.run(infidelity_kwargs={"norm_target": None})
    after = out.to_dense().reshape(-1)
    fidelity = abs(np.vdot(before, after)) ** 2 / (np.vdot(before, before).real * np.vdot(after, after).real)
    assert fidelity == pytest.approx(1., abs=1e-12)
    assert opt.step_records[0]["reason"] == "below_tol"
    assert opt.step_records[0]["effective_evaluation_chi"] > 10
    attempts = opt.get_evaluation_records()[0]["attempts"]
    assert attempts[0]["infidelity"] < -1e-5
    assert len(attempts) == 2


@pytest.mark.parametrize("accurate", [False, True])
def test_unit_target_norm_identity_requires_accurate_normalization(accurate):
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=17)
    opt = PepsOptimizer(state, [(np.eye(4), ((1, 1), (1, 2)))], chi=2,
                        contraction_opt="greedy", evaluation_negative_tol=0,
                        **({"normalize_chi": 32, "evaluation_chi": 32} if accurate else {}), boundary_convergence=False)
    if not accurate:
        with pytest.raises(ValueError, match="increase normalize_chi"):
            opt.run()
    else:
        out = opt.run()
        vector = out.to_dense().reshape(-1)
        expected = state.to_dense().reshape(-1)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vector, expected, atol=2e-12)
        assert opt.step_records[0]["final_infidelity"] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("mode", ["sweep", "global"])
def test_default_policy_continues_real_coarse_identity_contraction(mode, monkeypatch):
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=17)
    opt = PepsOptimizer(state, [(np.eye(4), ((1, 1), (1, 2)))], chi=2,
                        mode=mode, contraction_opt="greedy", boundary_convergence=False)
    attempted = []

    def reject_refinement(warmstart, target, **kwargs):
        attempted.append(True)
        return warmstart, None, {"success": False}

    # Exercise the real coarse precheck and rollback, without making this
    # policy regression an unbounded variational fit of an identity gate.
    monkeypatch.setattr(opt, "_optimize_state", reject_refinement)
    with pytest.warns(RuntimeWarning, match="continuing with zero"):
        out = opt.run()
    before, after = state.to_dense().reshape(-1), out.to_dense().reshape(-1)
    fidelity = abs(np.vdot(before, after)) ** 2 / (np.vdot(before, before).real * np.vdot(after, after).real)
    assert fidelity == pytest.approx(1., abs=1e-12)
    record = opt.step_records[0]
    if mode == "sweep":
        assert attempted == [True]
        assert record["optimizer_attempted"]
        assert record["reason"] == "optimizer_failed"
    else:
        assert attempted == []
        assert record["reason"] == "below_tol"
    metric = opt.get_evaluation_records()[0]
    assert -1e-3 < metric["raw_infidelity"] < -1e-5
    assert metric["infidelity"] == 0.
    assert len(metric["attempts"]) == 1


@pytest.mark.parametrize("constructor_options,run_options,known", [
    ({}, {}, 1.),
    ({}, {"non_unitary": True}, None),
    ({}, {"normalize_final": False}, None),
    ({"infidelity_kwargs": {"norm_target": None}}, {}, None),
    ({"infidelity_kwargs": {"norm_target": 2.}}, {"infidelity_kwargs": {"norm_target": None}}, None),
    ({}, {"infidelity_kwargs": {"norm_target": 2.}}, 2.),
])
def test_run_target_norm_policy_skips_actual_target_contraction(
    monkeypatch, constructor_options, run_options, known,
):
    original = peps_mod.boundary_infidelity
    results = []
    def capture(*args, **kwargs):
        assert kwargs["norm_target"] == known
        result = original(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(peps_mod, "boundary_infidelity", capture)
    opt = PepsOptimizer(_product(2, 2), [(_unitary(), ((0, 0), (0, 1)))],
                        chi=1, contraction_opt="greedy",
                        sweep_optimize_kwargs={"n_cycles": 0}, **constructor_options, boundary_convergence=False)
    opt.run(**run_options)
    assert len(results) == 2  # Both pre- and post-optimization checks.
    assert all((result["norm_target_result"] is None) == (known is not None)
               for result in results)
    assert all(result["norm_result"] is not None for result in results)


@pytest.mark.parametrize("fit_mode,overlap_checks", [("eff", 3), ("direct", 2)])
def test_default_sweep_skips_all_target_contractions_and_duplicate_normalization(
    monkeypatch, fit_mode, overlap_checks,
):
    contractions = []
    normalize_calls = []
    contract = boundary_metrics._contract_peps_double_layer
    parent_normalize = peps_mod.boundary_normalize
    sweep_normalize = sweep_mod.peps_normalize
    def capture_contract(*args, **kwargs):
        contractions.append(kwargs.get("bdy_name"))
        assert kwargs.get("bdy_name") != "bdy_target"
        return contract(*args, **kwargs)
    def parent_norm(state, **kwargs):
        normalize_calls.append(("driver", id(state)))
        return parent_normalize(state, **kwargs)
    def sweep_norm(state, **kwargs):
        normalize_calls.append(("sweep", id(state)))
        return sweep_normalize(state, **kwargs)
    monkeypatch.setattr(boundary_metrics, "_contract_peps_double_layer", capture_contract)
    monkeypatch.setattr(peps_mod, "boundary_normalize", parent_norm)
    monkeypatch.setattr(sweep_mod, "peps_normalize", sweep_norm)
    opt = PepsOptimizer(_product(2, 2), [(_unitary(), ((0, 0), (0, 1)))],
                        chi=1, contraction_opt="greedy", fit_mode=fit_mode, boundary_convergence=False)
    out = opt.run()
    # Always retain outer pre/post checks. Only iterative FIT needs its own
    # initial diagnostic; direct compression reuses the matching outer check.
    assert contractions.count("bdy_overlap") == overlap_checks
    assert [owner for owner, _ in normalize_calls] == ["driver"] * 3
    assert opt.step_records[0]["optimizer_result"]["invalid_inner_loss_count"] == 0
    assert np.linalg.norm(out.to_dense()) == pytest.approx(1., abs=1e-12)


def test_explicit_sweep_initial_normalization_override_is_preserved(monkeypatch):
    calls = []
    original = sweep_mod.peps_normalize
    def normalize(state, **kwargs):
        calls.append(kwargs["chi"])
        return original(state, **kwargs)
    monkeypatch.setattr(sweep_mod, "peps_normalize", normalize)
    opt = PepsOptimizer(_product(2, 2), [(_unitary(), ((0, 0), (0, 1)))],
                        chi=1, contraction_opt="greedy",
                        sweep_kwargs={"renormalize_state": True,
                                      "renormalize_kwargs": {"chi": 7}},
                        sweep_optimize_kwargs={"n_cycles": 0}, boundary_convergence=False)
    opt.run()
    assert calls == [7]


@pytest.mark.parametrize("mode", ["sweep", "global"])
@pytest.mark.parametrize("known", [None, (4., 0.)])
@pytest.mark.parametrize("evaluation_chi", [7, (7, 9)])
def test_nonunitary_objective_norm_is_resolved_without_fidelity_measurements(
    monkeypatch, mode, known, evaluation_chi,
):
    original_norm = peps_mod.boundary_norm
    norms = []
    def measure(target, **kwargs):
        norms.append(kwargs)
        return original_norm(target, **kwargs)
    monkeypatch.setattr(peps_mod, "boundary_norm", measure)
    opt = PepsOptimizer(_product(2, 2), [(2 * _unitary(), ((0, 0), (0, 1)))],
                        chi=1, contraction_opt="greedy", mode=mode, evaluation_chi=evaluation_chi,
                        global_optimize_kwargs={"n": 2}, boundary_convergence=False)
    original_optimize = opt._optimize_state
    def optimize(state, target, **kwargs):
        mantissa, exponent = kwargs["target_norm"]
        assert complex(mantissa).real * 10 ** exponent == pytest.approx(4., abs=1e-12)
        return original_optimize(state, target, **kwargs)
    monkeypatch.setattr(opt, "_optimize_state", optimize)
    out = opt.run(non_unitary=True, normalize_target=False, measure_infidelity=False,
                  infidelity_kwargs={"norm_target": known})
    assert len(norms) == (1 if known is None else 0)
    if norms:
        assert norms[0]["chi"] == 7
    if mode == "sweep":
        record = opt.step_records[0]
        assert record["optimizer_result"]["invalid_inner_loss_count"] == 0
        # The rank-one optimum is fixed by the two-qubit state's Schmidt values.
        target_vector = _dense_apply(_product(2, 2).to_dense().reshape(-1), 2 * _unitary(), (0, 1))
        singular_values = np.linalg.svd(target_vector.reshape(2, -1), compute_uv=False)
        expected_loss = 1 - singular_values[0] ** 2 / np.vdot(target_vector, target_vector).real
        assert record["final_infidelity"] == pytest.approx(expected_loss, abs=1e-12)
    assert np.linalg.norm(out.to_dense()) == pytest.approx(1., abs=1e-12)


def test_metric_recomputes_norms_at_requested_evaluation_accuracy():
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=17)
    opt = PepsOptimizer(state, chi=2, contraction_opt="greedy", boundary_convergence=False)
    opt.normalize()
    assert opt.estimate_infidelity(opt.state, opt.state, evaluation_chi=32) == pytest.approx(0, abs=1e-12)
    # Explicit unnormalized input also has scale-invariant fidelity.
    scaled = opt.state * 3
    assert opt.estimate_infidelity(opt.state, scaled, evaluation_chi=32) == pytest.approx(0, abs=1e-12)


@pytest.mark.optional
def test_unknown_native_target_norm_preserves_torch_symmetry_storage(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymPEPS, site_charge_alternating

    def convert(array):
        return torch.as_tensor(array, dtype=torch.complex128)
    state = SymPEPS.random(
        2, 2, symmetry="U1", bond_dim=2, phys_dim=2, fermionic=True,
        site_charge=site_charge_alternating(0, 1), seed=17,
        dtype="complex128", to_backend=convert,
    ).peps
    gate = 2 * Fermion(spinful=False, symmetry="U1", to_backend=convert).hopping_gate(.1, t=1.)
    norms = []
    original = peps_mod.boundary_norm
    def measure(target, **kwargs):
        assert all(type(tensor.data).__name__ == "U1FermionicArray" for tensor in target)
        value = original(target, **kwargs)
        norms.append(value)
        return value
    monkeypatch.setattr(peps_mod, "boundary_norm", measure)
    opt = PepsOptimizer(state, [(gate, ((0, 0), (0, 1)))], chi=1,
                        contraction_opt="greedy", sweep_optimize_kwargs={"n_cycles": 0}, boundary_convergence=False)
    out = opt.run(non_unitary=True, normalize_target=False, measure_infidelity=False)
    assert len(norms) == 1
    mantissa, exponent = norms[0]
    assert float(abs(mantissa) * 10 ** exponent) == pytest.approx(4., abs=1e-12)
    assert float(abs((out.H & out).contract(all, optimize="greedy"))) == pytest.approx(1., abs=1e-12)
    assert all(type(tensor.data).__name__ == "U1FermionicArray" for tensor in out)


def test_metric_retries_are_bounded_and_can_be_disabled(monkeypatch):
    calls = []
    def invalid(*args, **kwargs):
        calls.append(kwargs["chi"])
        return {"infidelity": -.1}
    monkeypatch.setattr(peps_mod, "boundary_infidelity", invalid)
    opt = PepsOptimizer(_product(2, 2), chi=1, boundary_convergence=False)
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="substantially negative"):
        opt.estimate_infidelity(opt.state, opt.state)
    assert calls == [(4, 5), 10, 20]
    calls.clear()
    with pytest.raises(ValueError, match="substantially negative"):
        opt.estimate_infidelity(opt.state, opt.state, evaluation_max_retries=0)
    assert calls == [(4, 5)]


@pytest.mark.parametrize("raw_error", [-6.86e-10, -3.267e-6, -5e-4])
def test_small_negative_metric_continues_gate_stream_and_preserves_raw_error(monkeypatch, raw_error):
    import json

    gates = [(_unitary(), ((0, 0), (0, 1)))] * 2
    reference = PepsOptimizer(_product(2, 2), gates, chi=1, contraction_opt="greedy", boundary_convergence=False)
    expected = reference.run(optimize=False, measure_infidelity=False, k_2q_batch=1)
    calls = []
    values = iter([raw_error, 2e-10])

    def metric(*args, **kwargs):
        calls.append(kwargs)
        return {"infidelity": next(values)}

    monkeypatch.setattr(peps_mod, "boundary_infidelity", metric)
    opt = PepsOptimizer(_product(2, 2), gates, chi=1, contraction_opt="greedy", boundary_convergence=False)
    # A clipped initial score requests refinement. If refinement fails, the
    # gate stream still continues from the warm start with raw diagnostics.
    monkeypatch.setattr(opt, "_optimize_state", lambda warmstart, target, **kw: (
        warmstart, None, {"success": False},
    ))
    streamed = []
    with pytest.warns(RuntimeWarning, match="Small negative approximate PEPS infidelity"):
        output = opt.run(k_2q_batch=1, step_callback=streamed.append)
    assert len(calls) == len(streamed) == 2
    assert all(c["norm_target"] == 1.0 for c in calls)
    assert streamed[0]["final_infidelity"] == 0.0
    assert streamed[1]["final_infidelity"] == 2e-10
    diagnostic = streamed[0]["evaluation_records"][0]
    assert diagnostic["clipped_negative"]
    assert diagnostic["raw_infidelity"] == raw_error
    assert diagnostic["attempts"] == [{"chi": (4, 5), "infidelity": raw_error}]
    assert [r["optimizer_attempted"] for r in streamed] == [True, False]
    assert streamed[0]["reason"] == "optimizer_failed"
    assert opt.get_evaluation_records()[0] == diagnostic
    json.dumps(streamed)
    np.testing.assert_allclose(output.to_dense(), expected.to_dense(), atol=1e-12)


@pytest.mark.parametrize("value,options,metric_options", [
    (-6.86e-10, {"evaluation_negative_tol": 0}, {}),
    (-6.86e-10, {}, {"method": "exact"}),
    (-1.01e-3, {}, {}),
    (1.00101, {}, {}),
    (1.0005, {}, {"method": "exact"}),
    (float("nan"), {}, {}),
    (float("inf"), {}, {}),
])
def test_negative_allowance_keeps_strict_and_nonfinite_guards(
    monkeypatch, value, options, metric_options,
):
    monkeypatch.setattr(peps_mod, "boundary_infidelity", lambda *a, **kw: {"infidelity": value})
    opt = PepsOptimizer(_product(2, 2), chi=1, **options, boundary_convergence=False)
    with pytest.raises(ValueError, match="negative|finite|above one"):
        opt.estimate_infidelity(opt.state, opt.state, norm_target=1., **metric_options)


@pytest.mark.parametrize("tolerance", [-1., float("nan"), float("inf")])
def test_negative_allowance_requires_finite_nonnegative_tolerance(tolerance):
    with pytest.raises(ValueError, match="evaluation_negative_tol"):
        PepsOptimizer(_product(2, 2), chi=1, evaluation_negative_tol=tolerance, boundary_convergence=False)


def test_negative_allowance_can_be_configured(monkeypatch):
    monkeypatch.setattr(peps_mod, "boundary_infidelity", lambda *a, **kw: {"infidelity": -2e-8})
    opt = PepsOptimizer(_product(2, 2), chi=1, evaluation_negative_tol=3e-8, boundary_convergence=False)
    with pytest.warns(RuntimeWarning, match="continuing with zero"):
        assert opt.estimate_infidelity(opt.state, opt.state, norm_target=1.) == 0.
    assert opt.get_evaluation_records()[0]["negative_tolerance"] == 3e-8


def test_above_one_metric_clips_without_retry_and_retains_raw_value(monkeypatch):
    monkeypatch.setattr(peps_mod, "boundary_infidelity", lambda *a, **kw: {"infidelity": 1.0005})
    opt = PepsOptimizer(_product(2, 2), chi=1, boundary_convergence=False)
    with pytest.warns(RuntimeWarning, match="continuing with one"):
        assert opt.estimate_infidelity(opt.state, opt.state, norm_target=1.) == 1.
    record = opt.get_evaluation_records()[0]
    assert record["raw_infidelity"] == 1.0005
    assert record["clipped_infidelity"] and not record["clipped_negative"]
    assert len(record["attempts"]) == 1


@pytest.mark.parametrize("raw_loss", [-5e-4, 1.0005])
@pytest.mark.parametrize("phase", ["initial", "final"])
def test_streamed_sweep_summary_preserves_both_clipping_bounds(monkeypatch, raw_loss, phase):
    import json

    clipped_run = {
        "axis": "y", "index": 1,
        f"raw_loss_{phase}": raw_loss,
        f"loss_{phase}": float(np.clip(raw_loss, 0., 1.)),
    }

    class FakeSweep:
        def __init__(self, state, **kwargs):
            self.state = state

        def set_optimize_kwargs(self, **kwargs):
            pass

        def run(self):
            return {
                "best_state": self.state, "best_loss": .05,
                "runs": [
                    {"axis": "x", "index": 0, "raw_loss_initial": .1, "raw_loss_final": .05},
                    clipped_run,
                    {"axis": "x", "index": 1},
                ],
            }

    monkeypatch.setattr(peps_mod, "SweepOptimizer", FakeSweep)
    estimates = iter([.1, .05])
    monkeypatch.setattr(peps_mod, "boundary_infidelity",
                        lambda *a, **kw: {"infidelity": next(estimates)})
    opt = PepsOptimizer(_product(2, 2), [(_unitary(), ((0, 0), (0, 1)))],
                        chi=1, contraction_opt="greedy", boundary_convergence=False)
    streamed = []
    opt.run(k_2q_batch=1, step_callback=streamed.append)
    saved = json.loads(json.dumps(streamed))
    assert saved[0]["optimizer_result"]["clipped_loss_records"] == [clipped_run]
    assert saved[0]["optimized"]
    assert saved[0]["final_infidelity"] == .05


@pytest.mark.parametrize("mode", ["sweep", "global"])
@pytest.mark.parametrize("raw_loss", [-3.267e-6, -5e-4, 1.0005])
def test_inner_loss_clipping_keeps_gate_stream_and_outer_acceptance(monkeypatch, mode, raw_loss):
    class FakeOptimizer:
        def __init__(self, state, **kwargs):
            self.state = state
            self.losses = [raw_loss]
            self.final_loss = raw_loss

        def set_optimize_kwargs(self, **kwargs):
            pass

        def run(self):
            return {"best_state": self.state, "best_loss": raw_loss}

        def optimize_nlopt(self, **kwargs):
            return self.state

    monkeypatch.setattr(peps_mod, "SweepOptimizer", FakeOptimizer)
    monkeypatch.setattr(peps_mod, "GlobalOptimizer", FakeOptimizer)
    estimates = iter([.1, .05, .1, .05])
    monkeypatch.setattr(peps_mod, "boundary_infidelity",
                        lambda *a, **kw: {"infidelity": next(estimates)})
    opt = PepsOptimizer(_product(2, 2), [(_unitary(), ((0, 0), (0, 1)))] * 2,
                        chi=1, mode=mode, contraction_opt="greedy", register_torch_svd=False, boundary_convergence=False)
    with pytest.warns(RuntimeWarning, match="continuing with"):
        opt.run(k_2q_batch=1)
    assert len(opt.step_records) == 2
    for record in opt.step_records:
        assert record["optimized"]
        assert record["post_infidelity"] == record["final_infidelity"] == .05
        assert record["optimizer_infidelity"] == np.clip(raw_loss, 0., 1.)
        assert record["optimizer_result"]["raw_infidelity"] == raw_loss
        assert record["optimizer_result"]["clipped_infidelity"]


def test_temporary_run_mode_does_not_change_configured_backend():
    opt = PepsOptimizer(_product(2, 2), chi=2, contraction_opt="greedy", boundary_convergence=False)
    opt.run(mode="global")
    assert opt.mode == "sweep"
    opt.set_mode("global")
    opt.run(mode="sweep")
    assert opt.mode == "global"


def test_sweep_fit_records_reach_public_getter(monkeypatch):
    original = peps_mod.SweepOptimizer
    observed = []
    class CapturedSweep(original):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            observed.extend(self.fit_diagnostics)
            return result
    monkeypatch.setattr(peps_mod, "SweepOptimizer", CapturedSweep)
    opt = PepsOptimizer(_product(2, 2), [(_unitary(), ((0, 0), (0, 1)))], chi=1,
                        contraction_opt="greedy", fit_timing=True,
                        sweep_optimize_kwargs={"n_cycles": 0}, boundary_convergence=False)
    opt.run()
    assert observed
    returned = opt.get_fit_diagnostics()
    assert all(any(record is item for item in returned) for record in observed)


def test_nlopt_defaults_allow_constructor_and_run_overrides():
    opt = PepsOptimizer(_product(2, 2), chi=2, optimizer_options={"maxeval": 25}, boundary_convergence=False)
    defaults = opt._apply_sweep_optimizer_options({})["optimizer_options"]
    assert defaults == {"algorithm": "LD_LBFGS", "maxeval": 25,
                        "ftol_rel": 1e-9, "ftol_abs": 1e-9,
                        "xtol_rel": 1e-9, "restore_best": True}
    overridden = opt._apply_sweep_optimizer_options(
        {"optimizer_options": {"maxeval": 15, "ftol_rel": 1e-7}},
    )["optimizer_options"]
    assert overridden["maxeval"] == 15
    assert overridden["ftol_rel"] == 1e-7
    partial = opt._apply_sweep_optimizer_options(
        {"optimizer_options": {"ftol_rel": 1e-7}},
    )["optimizer_options"]
    assert partial["maxeval"] == 25


@pytest.mark.optional
def test_native_fermionic_auto_batch_preserves_state_and_backend():
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, SymPEPS, site_charge_alternating

    def convert(array):
        return torch.as_tensor(array, dtype=torch.complex128)

    state = SymPEPS.random(
        2, 2, symmetry="U1", bond_dim=2, phys_dim=2, fermionic=True,
        site_charge=site_charge_alternating(0, 1), seed=17,
        dtype="complex128", to_backend=convert,
    ).peps
    fermion = Fermion(spinful=False, symmetry="U1", to_backend=convert)
    gate = fermion.hopping_gate(.1, t=1.)
    gates = [(gate, ((0, 0), (0, 1))), (gate, ((1, 0), (1, 1)))]
    auto = PepsOptimizer(state, gates, chi=8, contraction_opt="greedy", boundary_convergence=False)
    fixed = PepsOptimizer(state, gates, chi=8, contraction_opt="greedy", boundary_convergence=False)
    actual = auto.run()
    expected = fixed.run(k_2q_batch=1)
    overlap = (actual.H & expected).contract(all, optimize="greedy")
    actual_norm = (actual.H & actual).contract(all, optimize="greedy")
    expected_norm = (expected.H & expected).contract(all, optimize="greedy")
    assert float(abs(overlap) ** 2 / abs(actual_norm * expected_norm)) == pytest.approx(1., abs=1e-12)
    assert [record["two_site_batch"] for record in auto.step_records] == [2]
    for tensor in actual:
        assert type(tensor.data).__name__ == "U1FermionicArray"
        assert all(block.dtype == torch.complex128 and block.device.type == "cpu"
                   for block in tensor.data.blocks.values())
