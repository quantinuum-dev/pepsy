"""Device preservation and explicit readout boundaries for tree replay."""

from types import SimpleNamespace

import autoray as ar
import numpy as np
import pytest
import quimb as qu

from pepsy import SubTreeMPO, TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.optimizers.tree import optimizer as optimizer_module
from pepsy.optimizers.tree._diagnostics import diagnostic_to_host, norm_event

pytestmark = [pytest.mark.integration, pytest.mark.optional]


@pytest.fixture(params=["torch", "jax", "cuda", "cupy"])
def convert(request):
    if request.param in {"torch", "cuda"}:
        torch = pytest.importorskip("torch")
        device = "cuda" if request.param == "cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        return lambda x: torch.tensor(np.array(x), dtype=torch.complex64, device=device)
    if request.param == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CuPy GPU unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CuPy GPU unavailable")
        return lambda x: cp.asarray(x, dtype=cp.complex64)
    jnp = pytest.importorskip("jax.numpy")
    return lambda x: jnp.asarray(x, dtype=jnp.complex64)


def _state():
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=7, dtype="complex64")
    state.multiply_(1 / state.norm())
    return state


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "zipup"])
@pytest.mark.parametrize("exponent", [0., -400., 400.])
def test_compression_keeps_norm_ledger_on_backend_without_clocks(mode, exponent, monkeypatch):
    torch = pytest.importorskip("torch")
    state = _state()
    state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex64))
    gate = torch.tensor(np.array(qu.CNOT()), dtype=torch.complex64)
    opt = TreeOptimizer([(gate, (0, 3))] * 3, state=state, chi=2,
                        mode=mode, cutoff=0., run=False)
    # Uncached operator SVD rank selection is an upstream host decision;
    # separately guard that preparation against full-array transfers below.
    opt.run()
    opt.tn.exponent = exponent

    def forbidden(*args, **kwargs):
        raise AssertionError("default compression read a host scalar or profiling clock")

    with monkeypatch.context() as patch:
        for name in ("item", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(ar, "to_numpy", forbidden)
        patch.setattr(optimizer_module, "time", SimpleNamespace(perf_counter=forbidden))
        opt.run()
    assert isinstance(opt._norm_log_survival, torch.Tensor)
    assert not opt._norm_log_survival.requires_grad
    assert all(event["elapsed_seconds"] is None for event in opt.update_history)
    assert not opt.profile_events
    records = opt.get_norm_events()
    assert all(isinstance(e["local_fidelity"], float) for e in records)
    assert opt.norm_diagnostics()["cumulative_fidelity"] == pytest.approx(
        np.prod([e["local_fidelity"] for e in records]), abs=1e-12)


def test_backend_replay_matches_numpy_and_does_not_download_gate_matrices(convert, monkeypatch):
    initial = _state()
    gates = [(np.array(qu.hadamard(), dtype=np.complex64), 0),
             (np.array(qu.CNOT(), dtype=np.complex64), (0, 3))] * 2
    reference = TreeOptimizer(gates, state=initial, chi=2, cutoff=0.)
    state = initial.copy()
    state.apply_to_arrays(convert)
    opt = TreeOptimizer([(convert(g), w) for g, w in gates], state=state,
                        chi=2, cutoff=0., run=False)
    signature = opt.backend_info()
    original = ar.to_numpy

    def scalar_only(value):
        assert ar.size(value) == 1, "gate or state matrix downloaded during replay"
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", scalar_only)
        opt.run()
        opt.apply_1q(convert(qu.hadamard()), 1)
    reference.apply_1q(np.array(qu.hadamard(), dtype=np.complex64), 1)
    assert opt.backend_info() == signature
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-5)
    assert opt.norm_diagnostics()["cumulative_fidelity"] == pytest.approx(
        reference.norm_diagnostics()["cumulative_fidelity"], abs=1e-5)
    clone = opt.copy()
    assert clone.get_norm_events() == opt.get_norm_events()
    opt.get_norm_events()[0]["local_fidelity"] = -1.
    assert opt.get_norm_events()[0]["local_fidelity"] >= 0.


@pytest.mark.parametrize("mode", ["direct", "dm"])
def test_branch_final_center_and_following_path_preserve_backend(convert, mode):
    plan = TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    initial = TreeTensorNetwork.rand(plan, D=2, seed=81, dtype="complex64")
    initial.multiply_(1 / initial.norm())
    reference = TreeOptimizer(None, state=initial, mode=mode, chi=2, cutoff=0., run=False)
    initial.apply_to_arrays(convert)
    candidate = TreeOptimizer(None, state=initial, mode=mode, chi=2, cutoff=0., run=False)
    signature = candidate.backend_info()
    rng = np.random.default_rng(65)
    for where in ((0, 2, 7), (1, 6)):
        size = 2 ** len(where)
        gate = np.linalg.qr(
            rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        )[0].astype("complex64")
        reference.apply_gate(gate, where)
        candidate.apply_gate(convert(gate), where)
        assert candidate.backend_info() == signature
        assert candidate.center == reference.center
        assert candidate.tn.is_canonical_form(candidate.center, tol=2e-5)
        candidate.validate_isometry_metadata()
        np.testing.assert_allclose(candidate.to_dense(), reference.to_dense(), atol=2e-5)
        assert candidate.norm() == pytest.approx(reference.norm(), abs=2e-5)


@pytest.mark.parametrize("where", [(3,), (3, 0), (3, 0, 2)])
@pytest.mark.parametrize("exponent", [-400., 0., 400.])
def test_kraus_probabilities_use_exact_local_readout(convert, where, exponent, monkeypatch):
    from copy import deepcopy

    from pepsy import TrajectoryChannel
    from pepsy.optimizers.noise import _kraus_probabilities

    state = _state()
    dense = state.to_statevector()
    size = 2 ** len(where)
    weights = np.linspace(.1, .9, size)
    unitary = np.linalg.qr(np.random.default_rng(9).normal(size=(size, size)))[0]
    matrices = (np.diag(np.sqrt(weights)), unitary @ np.diag(np.sqrt(1 - weights)))
    channel = TrajectoryChannel.kraus(list(zip(("first", "second"), matrices)))
    axes = where + tuple(q for q in range(4) if q not in where)
    local = dense.reshape((2,) * 4).transpose(axes).reshape(size, -1)
    expected = [np.linalg.norm(matrix @ local) ** 2 / np.linalg.norm(local) ** 2
                for matrix in matrices]
    state.apply_to_arrays(convert)
    opt = TreeOptimizer(None, state=state, mode="dmrg2", chi=1,
                        stabilize_unitary=True, run=False)
    opt.tn.exponent = exponent
    tensors = [(tensor, tensor.data, tensor.left_inds) for tensor in opt.tn.tensors]
    center = opt.center
    rng = deepcopy(opt.rng.bit_generator.state)
    signature = opt.backend_info()

    def forbidden(*args, **kwargs):
        raise AssertionError("Kraus probability evaluation replayed or densified a trial state")

    monkeypatch.setattr(opt, "copy", forbidden)
    monkeypatch.setattr(TreeOptimizer, "apply_gate", forbidden)
    monkeypatch.setattr(TreeTensorNetwork, "to_dense", forbidden)
    actual = _kraus_probabilities(opt, channel, where)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert opt.center == center
    assert opt.tn.exponent == exponent
    assert opt.rng.bit_generator.state == rng
    assert opt.backend_info() == signature
    assert not opt.get_norm_events() and not opt.get_fit_diagnostics()
    for tensor, data, left_inds in tensors:
        assert tensor.data is data
        assert tensor.left_inds == left_inds


@pytest.mark.parametrize("strategy", ["random", "random_expand"])
def test_random_fit_uses_backend_generator_and_is_reproducible(convert, strategy):
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(3), structure="balanced"),
                                   D=2, seed=17, dtype="complex64")
    expected = TreeOptimizer([(np.array(qu.hadamard(), dtype=np.complex64), 0),
                             (np.array(qu.CNOT(), dtype=np.complex64), (0, 2))],
                             state=state, chi=4, cutoff=0.).to_dense()
    state.apply_to_arrays(convert)
    gates = [(convert(qu.hadamard()), 0), (convert(qu.CNOT()), (0, 2))]
    outputs = []
    for _ in range(2):
        opt = TreeOptimizer(gates, state=state, chi=4, mode="dmrg2", cutoff=0.,
                            fit_n_iter=3, fit_rtol=None, fit_init_strategy=strategy,
                            fit_init_rand_strength=1e-3, fit_init_seed=13)
        outputs.append(opt.to_dense())
        assert opt.backend_info()["backend"] == ar.infer_backend(state.node_tensor(state.plan.root).data)
    np.testing.assert_allclose(outputs[0], expected, atol=2e-5)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_device_norm_events_preserve_tree_zero_and_nonfinite_policy():
    torch = pytest.importorskip("torch")
    cpu_log, device_log = 0., 0.
    for expected, observed in [(1., .8), (0., 0.), (0., 1.), (1., 0.),
                               (1., np.nan), (np.inf, np.inf), (1., np.inf),
                               (np.inf, 1.), (1., 1.)]:
        active = dict(update=1, kind="gate", support=(0, 1), norm_before=expected)
        cpu_log, cpu_event = norm_event(active, observed, cpu_log)
        active["norm_before"] = torch.tensor(expected, dtype=torch.float64, requires_grad=True)
        device_log, device_event = norm_event(active, torch.tensor(observed, dtype=torch.float64), device_log)
        host_event = diagnostic_to_host(device_event)
        for key, value in cpu_event.items():
            if isinstance(value, float):
                np.testing.assert_allclose(host_event[key], value, atol=1e-14, equal_nan=True)
            else:
                assert host_event[key] == value
        assert not device_log.requires_grad
        assert all(not x.requires_grad for x in device_event.values() if isinstance(x, torch.Tensor))


def test_native_torch_tree_retains_device_norm_history():
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")
    from pepsy import Fermion, backend_torch, ps_to_ttn

    fermion = Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(3), structure="balanced")
    state = ps_to_ttn(3, tree=plan, fermion=fermion,
                      occupations=((1, 0), (0, 1), (1, 0)), dtype="complex128")
    backend = backend_torch(dtype=torch.complex128)
    state.apply_to_arrays(backend)
    gate = fermion.hopping_gate(.2, t=1.)
    gate.apply_to_arrays(backend)
    opt = TreeOptimizer([(gate, (0, 2))], state=state, chi=16, cutoff=0.)
    assert isinstance(opt._norm_log_survival, torch.Tensor)
    assert opt.norm_diagnostics()["cumulative_fidelity"] == pytest.approx(1., abs=1e-12)


def test_dense_operator_constructors_preserve_backend(convert, monkeypatch):
    state = _state()
    state.apply_to_arrays(convert)
    dense = convert(np.eye(16))
    original = ar.to_numpy

    def scalar_only(value):
        assert ar.size(value) == 1, "operator matrix downloaded during construction"
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", scalar_only)
        operators = [TreeMPO.from_dense(state.plan, dense),
                     TreeMPO.from_gate(state.plan, convert(qu.CNOT()), (0, 3)),
                     TreeMPO.from_pauli_sum(state.plan, [(1., {0: "X", 3: "Y"})], like=dense)]
    for operator in operators:
        assert all(ar.infer_backend(t.data) == ar.infer_backend(dense)
                   for t in operator.tensors)
        operator.validate()


def test_pauli_sum_numpy_like_selects_dtype():
    plan = TreePlan.from_order(range(3), structure="balanced")
    operator = TreeMPO.from_pauli_sum(plan, [(1., {0: "X", 2: "Y"})],
                                    like=np.ones((), dtype="complex64"))
    assert all(t.dtype == "complex64" for t in operator.tensors)


@pytest.mark.parametrize("compress", [False, True])
def test_higher_order_term_sum_stays_on_backend(convert, compress, monkeypatch):
    plan = TreePlan.from_order(range(4), structure="balanced")
    rng = np.random.default_rng(42)
    terms = {
        (0, 2, 3): (rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))).astype("complex64"),
        (1,): np.asarray(qu.pauli("Y"), dtype="complex64"),
        (0, 3): np.asarray(qu.CNOT(), dtype="complex64"),
    }
    expected = np.zeros((16, 16), dtype="complex64")
    for where, term in terms.items():
        order = where + tuple(site for site in range(4) if site not in where)
        axes = tuple(order.index(site) for site in range(4))
        block = np.kron(term, np.eye(2 ** (4 - len(where)))).reshape((2,) * 8)
        expected += block.transpose(axes + tuple(i + 4 for i in axes)).reshape(16, 16)
    backend_terms = {where: convert(term.reshape((2,) * (2 * len(where))))
                     for where, term in terms.items()}
    sample = next(iter(backend_terms.values()))
    original = ar.to_numpy

    def scalar_only(value):
        assert ar.size(value) == 1, "direct-sum construction downloaded a tensor"
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", scalar_only)
        operator = TreeMPO.from_terms(plan, backend_terms, compress=compress, cutoff=0.)
    operator.validate()
    for tensor in operator.tensors:
        assert ar.infer_backend(tensor.data) == ar.infer_backend(sample)
        assert tensor.data.dtype == sample.dtype
        if ar.infer_backend(sample) == "jax":
            assert tensor.data.devices() == sample.devices()
        else:
            assert getattr(tensor.data, "device", None) == getattr(sample, "device", None)
    np.testing.assert_allclose(ar.to_numpy(operator.to_dense()), expected, atol=2e-5)


def test_higher_order_term_sum_rejects_mixed_backends():
    torch = pytest.importorskip("torch")
    plan = TreePlan.from_order(range(3), structure="balanced")
    with pytest.raises(ValueError, match="share an array backend and device"):
        TreeMPO.from_terms(plan, {
            (0, 1, 2): torch.eye(8, dtype=torch.complex64).reshape((2,) * 6),
            (0,): np.eye(2, dtype="complex64"),
        }, compress=False)


def test_host_scale_event_does_not_mutate_previous_device_ledger():
    torch = pytest.importorskip("torch")
    previous = torch.tensor(-.1, dtype=torch.float64)
    active = dict(update=2, kind="gate", support=(0, 1), norm_before=1.)
    updated, _ = norm_event(active, .8, previous)
    assert float(previous) == -.1
    assert float(updated) < float(previous)


@pytest.mark.parametrize("host_norm", [1e-100, 1e100])
@pytest.mark.parametrize("host_before", [True, False])
def test_norm_ledger_handles_entering_and_leaving_host_scale(convert, host_norm, host_before):
    device_norm = ar.do("real", convert(1.))
    expected, observed = ((host_norm, device_norm) if host_before
                          else (device_norm, host_norm))
    active = dict(update=2, kind="gate", support=(0,), norm_before=expected)
    log, event = norm_event(active, observed, ar.do("real", convert(-.1)))
    active["norm_before"] = host_norm if host_before else 1.
    reference_log, reference = norm_event(active, 1. if host_before else host_norm, -.1)
    assert log == pytest.approx(reference_log, abs=1e-7)
    for key in ("expected_norm", "observed_norm", "local_fidelity", "local_infidelity"):
        assert event[key] == reference[key]


def test_warm_control_replay_builds_operators_on_backend(convert, monkeypatch):
    state = _state()
    stream = [("measure", "XY", (0, 3), 1), ("reset_y", 1), ("measure", "Z", 2, -1)]
    reference = TreeOptimizer(stream, state=state, chi=8, cutoff=0., seed=11)
    state.apply_to_arrays(convert)
    opt = TreeOptimizer(stream, state=state, chi=8, cutoff=0., seed=11, run=False)
    for name in ("I", "X", "Y", "Z", "H", "HY", "COPY", "XOR"):
        opt._control_tensor(name)
    original = opt._as_state_backend

    def native_only(value, **kwargs):
        assert ar.infer_backend(value) == opt.backend, "control tensor uploaded from host"
        return original(value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(opt, "_as_state_backend", native_only)
        opt.run()
    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=2e-5)
    np.testing.assert_allclose([m[3] for m in opt.measurements],
                               [m[3] for m in reference.measurements], atol=2e-5)


@pytest.mark.parametrize("exponent", [-100., 100.])
def test_represented_norm_preserves_exponent_range(convert, exponent):
    state = _state()
    state.apply_to_arrays(convert)
    gate = convert(qu.CNOT())
    opt = TreeOptimizer([(gate, (0, 3))], state=state, chi=8, cutoff=0., run=False)
    # Exercise a backend ledger followed by explicit extracted-scale tracking.
    opt.run()
    # Establish the unit-norm baseline after complex64 QR/SVD roundoff in the
    # warm-up, so this test isolates extracted-scale range on each device.
    opt.normalize()
    opt.tn.exponent = exponent
    before = opt.norm()
    opt.run()
    assert np.isfinite(before) and before > 0.
    assert opt.norm() / before == pytest.approx(1., abs=2e-6)
    assert opt.norm() / 10. ** exponent == pytest.approx(1., abs=2e-6)
    assert opt.get_norm_events()[-1]["local_fidelity"] == pytest.approx(1., abs=2e-6)

    # A scaled operator can cancel the exponent during a single update,
    # switching its final norm back to backend-only bookkeeping.
    before = opt.norm()
    identity = SubTreeMPO.from_gate(opt.plan, convert(np.eye(2)), 0)
    identity.exponent = -exponent
    opt.apply_sub_mpotree(identity)
    assert opt.tn.exponent == 0.
    assert opt.norm() == pytest.approx(1., abs=2e-6)
    _, reference = norm_event(dict(update=1, kind="gate", support=(0,),
                                   norm_before=before), opt.norm(), 0.)
    assert opt.get_norm_events()[-1]["local_fidelity"] == pytest.approx(
        reference["local_fidelity"], rel=2e-6, abs=0.)


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3", "mix"])
@pytest.mark.parametrize("tracking, exponent", [(False, 0.), (True, 1.), (False, 1.)])
def test_fit_target_norm_without_ledger_shortcut(convert, mode, tracking, exponent):
    state = TreeTensorNetwork.from_plan(TreePlan.from_order(range(4), structure="balanced"))
    state.apply_to_arrays(convert)
    state.exponent = exponent
    gates = [(convert(qu.hadamard()), 0), (convert(qu.CNOT()), (0, 3))]
    opt = TreeOptimizer(gates, state=state, mode=mode, chi=2, cutoff=0., run=False,
                        track_infidelity=tracking, stabilize_unitary=False,
                        fit_init_strategy="guess-direct")
    signature = opt.backend_info()
    opt.run()
    expected = np.zeros(16)
    expected[[0, 9]] = 10.**exponent / np.sqrt(2.)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=2e-5)
    assert opt.backend_info() == signature


@pytest.mark.parametrize("mode", ["direct", "dmrg2"])
@pytest.mark.parametrize("exponent", [-400., 400.])
def test_stabilization_with_extreme_scale_keeps_backend(convert, mode, exponent):
    state = _state()
    gate = np.array(qu.CNOT(), dtype=np.complex64)
    reference = TreeOptimizer([(gate, (0, 3))], state=state, mode=mode, chi=1,
                              stabilize_unitary=True, fit_init_strategy="guess-direct")
    state.apply_to_arrays(convert)
    state.exponent = exponent
    opt = TreeOptimizer([(convert(gate), (0, 3))], state=state, mode=mode, chi=1,
                        stabilize_unitary=True, fit_init_strategy="guess-direct", run=False)
    signature = opt.backend_info()
    opt.run()
    assert opt.tn.exponent == exponent
    working = opt.tn.copy()
    working.exponent = 0.
    np.testing.assert_allclose(ar.to_numpy(working.to_dense()).reshape(-1),
                               reference.to_dense(), atol=2e-5)
    assert opt.backend_info() == signature
    assert opt.norm_diagnostics()["local_fidelity"] == pytest.approx(
        reference.norm_diagnostics()["local_fidelity"], abs=2e-5)
