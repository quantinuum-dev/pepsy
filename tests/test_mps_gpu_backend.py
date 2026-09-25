"""Backend preservation and transfer boundaries in MPS replay."""

import autoray as ar
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy import MpsOptimizer

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


@pytest.mark.parametrize("transition", ["direct", "perm", "measure"])
def test_exact_reconstruction_preserves_backend_and_following_gate(convert, transition, monkeypatch):
    initial = qtn.MPS_computational_state("000", dtype="complex64", site_ind_id="q{}")
    initial.apply_to_arrays(convert)
    initial.exponent = 0.5
    gates = [(convert(qu.hadamard()), (0,)), (convert(qu.CNOT()), (0, 2))]
    opt = MpsOptimizer(initial, gates, chi=4, mode="exact", ind_id="q{}")
    expected_backend = opt.backend_info()
    to_numpy = ar.to_numpy

    def scalar_only(value):
        if ar.infer_backend(value) != "numpy" and ar.size(value) > 1:
            raise AssertionError("State reconstruction downloaded a tensor")
        return to_numpy(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", scalar_only)
        opt.run(progbar=False, cutoff=0.)
        if transition == "measure":
            opt.set_gates([("measure", "Z", 0, 1), (convert(qu.hadamard()), (1,))])
        else:
            opt.set_mode(transition)
            opt.set_gates([(convert(qu.hadamard()), (1,))])
        opt.run(progbar=False, cutoff=0.)
    assert opt.backend_info() == expected_backend
    reference = np.zeros(8, dtype=complex)
    if transition == "measure":
        reference[[0, 2]] = 1 / np.sqrt(2)
    else:
        reference[[0, 2, 5, 7]] = 0.5 * np.sqrt(10)
    np.testing.assert_allclose(to_numpy(opt.to_dense()).ravel(), reference, atol=2e-6)


@pytest.mark.parametrize("strategy", ["random", "random_expand"])
def test_random_fit_uses_reproducible_backend_generator(convert, strategy, monkeypatch):
    state = qtn.MPS_computational_state("0000", dtype="complex64")
    state.apply_to_arrays(convert)
    gates = [(convert(qu.hadamard()), (0,)), (convert(qu.CNOT()), (0, 3))]
    draws = []
    do = ar.do

    def record(fn, *args, **kwargs):
        result = do(fn, *args, **kwargs)
        if fn == "random.array":
            draws.append(result)
            like = kwargs["like"]
            assert ar.infer_backend(result) == ar.infer_backend(like)
            assert result.dtype == like.dtype
            assert getattr(result, "device", None) == getattr(like, "device", None)
        return result

    monkeypatch.setattr(ar, "do", record)
    for _ in range(2):
        opt = MpsOptimizer(state, gates, chi=4, mode="dmrg2")
        opt.run(progbar=False, cutoff=0., n_iter=3, fit_rtol=None,
                fit_init_strategy=strategy, fit_init_rand_strength=1e-3, fit_init_seed=13)
        assert opt.get_fit_diagnostics()["random_initialization"]["enabled"]
        expected = np.zeros(16)
        expected[[0, 9]] = 1 / np.sqrt(2)
        np.testing.assert_allclose(ar.to_numpy(opt.to_dense()).ravel(), expected, atol=2e-5)
    half = len(draws) // 2
    assert half > 1
    for first, second in zip(draws[:half], draws[half:]):
        np.testing.assert_array_equal(ar.to_numpy(first), ar.to_numpy(second))
    # Distinct same-shaped draws advance one generator rather than reseeding it.
    if draws[0].shape == draws[1].shape:
        assert not np.array_equal(ar.to_numpy(draws[0]), ar.to_numpy(draws[1]))


def test_warm_controls_do_not_upload_host_operators(convert, monkeypatch):
    initial = qtn.MPS_rand_state(4, 2, dtype="complex64", seed=7)
    stream = [("measure", "XY", (0, 3), 1), ("reset_y", 1), ("measure", "Z", 2, -1)]
    reference = MpsOptimizer(initial, stream, chi=8)
    reference.run(progbar=False, cutoff=0., seed=11)
    state = initial.copy()
    state.apply_to_arrays(convert)
    opt = MpsOptimizer(state, stream, chi=8)
    # Warm the seven fixed matrices once. Sub-MPOs/projectors must subsequently
    # be assembled on-device, including tensors on sites between the support.
    for name in ("I", "X", "Y", "Z", "H", "HY", "CX"):
        opt._control_operator(name)
    original = opt._to_state_backend

    def already_native(value):
        assert ar.infer_backend(value) == opt.backend
        return original(value)

    monkeypatch.setattr(opt, "_to_state_backend", already_native)
    opt.run(progbar=False, cutoff=0., seed=11)
    np.testing.assert_allclose(ar.to_numpy(opt.to_dense()), reference.to_dense(), atol=3e-6)
    np.testing.assert_allclose([m[3] for m in opt.measurements],
                               [m[3] for m in reference.measurements], atol=3e-6)


def test_control_constants_are_owned_and_follow_state_replacement():
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("00", dtype="complex128")
    opt = MpsOptimizer(state, [], chi=2)
    op = opt._control_operator("X")
    op[:] = 0
    np.testing.assert_array_equal(opt._control_operator("X"), qu.pauli("X"))
    for dtype in (torch.complex64, torch.complex128):
        replacement = state.copy()
        replacement.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))
        opt.set_p(replacement)
        projector, _ = opt._build_pauli_projector_submpo("XY", (0, 1), -1)
        assert all(t.data.dtype == dtype for t in projector.tensors)
        # Neither consumer writes nor branch writes can poison the cache.
        branch = opt._copy_for_trajectory_branch()
        branch._control_operator("Y").zero_()
        torch.testing.assert_close(opt._control_operator("Y"), torch.tensor(qu.pauli("Y"), dtype=dtype))


def test_native_projector_route_does_not_construct_dense_constants(monkeypatch):
    pytest.importorskip("symmray")
    from pepsy import Fermion, ps_to_mps

    fermion = Fermion(spinful=True, symmetry="U1", dtype="complex128")
    state = ps_to_mps(4, fermion=fermion,
                       occupations=fermion.half_filled_occupations(4), seed=5,
                       dtype="complex128")
    opt = MpsOptimizer(state, [], chi=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native projector was replaced with dense constants")

    monkeypatch.setattr(opt, "_control_operator", forbidden)
    assert opt._build_pauli_projector_submpo("ZZ", (0, 3), 1) is None


@pytest.mark.parametrize("mode", ["direct", "svd", "swap", "perm", "dmrg2", "mix"])
def test_unitary_ledger_reads_only_one_boolean_per_replay(mode, monkeypatch):
    """No norm/fidelity becomes a host scalar inside the unitary gate loop."""
    torch = pytest.importorskip("torch")
    state = qtn.MPS_rand_state(4, 2, seed=27, dtype="complex128")
    state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    gate = torch.tensor(np.array(qu.CNOT()), dtype=torch.complex128)
    opt = MpsOptimizer(state, [(gate, (0, 3))] * 3, chi=2, mode=mode)
    item = torch.Tensor.item
    tensor_bool = torch.Tensor.__bool__
    tensor_float = torch.Tensor.__float__
    import quimb.tensor.decomp as qd

    trim = qd._trim_and_renorm_svd_result

    def select_rank(*args, **kwargs):
        # Upstream adaptive rank allocation is a separate host boundary.
        # Only Pepsy's norm/fidelity ledger is constrained by this test.
        with monkeypatch.context() as rank_patch:
            rank_patch.setattr(torch.Tensor, "item", item)
            rank_patch.setattr(torch.Tensor, "__bool__", tensor_bool)
            rank_patch.setattr(torch.Tensor, "__float__", tensor_float)
            return trim(*args, **kwargs)

    reads = []

    def boolean_only(value, *args, **kwargs):
        assert value.dtype == torch.bool, "Norm/fidelity was read back during replay"
        reads.append(value)
        return item(value, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor converted to a Python float/bool during replay")

    to_numpy = ar.to_numpy

    def no_device_array_read(value):
        assert not isinstance(value, torch.Tensor), "Backend array transferred during replay"
        return to_numpy(value)

    with monkeypatch.context() as patch:
        patch.setattr(qd, "_trim_and_renorm_svd_result", select_rank)
        patch.setattr(torch.Tensor, "item", boolean_only)
        patch.setattr(torch.Tensor, "__float__", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        patch.setattr(ar, "to_numpy", no_device_array_read)
        # FIT convergence is disabled here; its stopping decisions are a
        # separate, intentional host boundary from the compression ledger.
        opt.run(progbar=False, cutoff=0., n_iter=3, fit_rtol=None)
    assert len(reads) == 1
    assert isinstance(opt._unitary_previous_norm, torch.Tensor)
    assert isinstance(opt._norm_log_survival, torch.Tensor)
    assert all(isinstance(event["local_infidelity"], torch.Tensor) for event in opt.norm_events)
    events = opt.get_norm_events()
    expected = np.prod([min((e["observed_norm"] / e["expected_norm"]) ** 2, 1.) for e in events])
    assert opt.norm_diagnostics()["fidelity"] == pytest.approx(expected, abs=1e-12)
    assert all(isinstance(e["local_infidelity"], float) for e in events)


@pytest.mark.parametrize("restore", [False, True])
def test_backend_ledger_matches_cpu_and_preserves_state_dtype(convert, restore):
    initial = qtn.MPS_rand_state(4, 3, seed=19, dtype="complex64")
    stream = [(np.array(qu.CNOT()), (0, 3))] * 3
    cpu = MpsOptimizer(initial, stream, chi=2)
    cpu.run(progbar=False, cutoff=0., stabilize_unitary=restore)
    state = initial.copy()
    state.apply_to_arrays(convert)
    opt = MpsOptimizer(state, [(convert(g), w) for g, w in stream], chi=2)
    info = opt.backend_info()
    opt.run(progbar=False, cutoff=0., stabilize_unitary=restore)
    assert opt.backend_info() == info
    assert ar.infer_backend(opt._norm_log_survival) == info["backend"]
    np.testing.assert_allclose(ar.to_numpy(opt.to_dense()), cpu.to_dense(), atol=1e-5)
    full, compact = opt.norm_diagnostics(), opt.norm_diagnostics(include_history=False)
    assert full["infidelity"] == pytest.approx(cpu.norm_diagnostics()["infidelity"], abs=3e-6)
    assert compact["infidelity"] == full["infidelity"]
    assert compact["local_infidelity"] == full["local_infidelity"]
    # A physical boundary after device-only compression must still compose
    # with the same ledger and retain its mantissa/exponent bookkeeping.
    opt.set_gates([("measure", "Z", 1, 1)])
    opt.run(progbar=False, cutoff=0.)
    event = opt.get_norm_events()[-1]
    assert event["physical_boundary"]
    assert event["local_infidelity"] == pytest.approx(0., abs=3e-6)


def test_device_diagnostic_history_is_detached_and_transactional():
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    opt = MpsOptimizer(state, [], chi=2)
    scale = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    expected = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    opt._stabilize_unitary_compression_state(
        opt.p, (0, 1), expected, current_norm=scale, center_site=1, restore=False)
    assert not opt._unitary_previous_norm.requires_grad
    assert not opt._norm_log_survival.requires_grad
    assert all(not v.requires_grad for v in opt.norm_events[0].values() if isinstance(v, torch.Tensor))
    clone = opt.copy()  # Torch deepcopy rejects tensors with retained grad_fn.
    snapshot = opt._mix_state_snapshot()
    opt._stabilize_unitary_compression_state(
        opt.p, (0, 1), expected, current_norm=scale * 0., center_site=1, restore=False)
    opt._restore_mix_state(snapshot)
    opt._check_deferred_norm_errors()
    assert opt.get_norm_events() == clone.get_norm_events()
    assert opt._cumulative_fidelity() == pytest.approx(0.64)


def test_norm_namespaces_support_older_autoray(monkeypatch):
    torch = pytest.importorskip("torch")
    from pepsy.optimizers.tree._diagnostics import diagnostic_to_host, norm_event

    opt = MpsOptimizer(qtn.MPS_computational_state("00"), [], chi=2)
    before, after = torch.tensor(1.), torch.tensor(.8)
    active = dict(update=1, kind="gate", support=(0, 1), norm_before=before)
    expected_log, expected_event = norm_event(active, after, 0.)
    monkeypatch.delattr(ar, "get_namespace")
    log, event = norm_event(active, after, 0.)
    torch.testing.assert_close(log, expected_log)
    assert diagnostic_to_host(event) == diagnostic_to_host(expected_event)
    opt._record_norm_event("unitary_compression", expected_norm=before, observed_norm=after)
    assert opt.get_norm_events()[0]["cumulative_fidelity"] == pytest.approx(.64)
    # A later host event must switch dispatch back to the device ledger.
    cumulative, _ = opt._accumulate_norm_survival(.5)
    assert isinstance(cumulative, torch.Tensor)
    assert float(cumulative) == pytest.approx(.32)


@pytest.mark.parametrize("strict", [False, True])
def test_zero_norm_still_raises_before_replay_returns(strict):
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    opt = MpsOptimizer(state, [(torch.zeros((4, 4), dtype=torch.complex128), (0, 1))], chi=2)
    with pytest.raises(FloatingPointError, match="zero or non-finite norm"):
        opt.run(progbar=False, cutoff=0., stabilize_unitary=True, finite_check=strict, timing=True)
    assert opt.get_run_timing()["status"] == "failed"


def test_native_torch_compression_retains_device_norm_ledger():
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")
    from pepsy import Fermion, backend_torch, ps_to_mps

    fermion = Fermion(spinful=True, symmetry="U1", dtype="complex128")
    state = ps_to_mps(3, fermion=fermion,
                       occupations=fermion.half_filled_occupations(3), seed=5,
                       dtype="complex128")
    backend = backend_torch(dtype=torch.complex128)
    state.apply_to_arrays(backend)
    gate = fermion.hopping_gate(0.2, t=1.)
    gate.apply_to_arrays(backend)
    opt = MpsOptimizer(state, [(gate, (0, 2))], chi=8, mode="swap")
    opt.run(progbar=False, cutoff=0.)
    assert opt.backend_info()["array_backend"] == "torch"
    assert isinstance(opt._norm_log_survival, torch.Tensor)
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=1e-12)


def test_backend_ledger_preserves_invalid_complete_loss_and_nan_semantics():
    torch = pytest.importorskip("torch")
    opt = MpsOptimizer(qtn.MPS_computational_state("00"), [], chi=2)

    def record(expected, observed):
        opt._record_norm_event(
            "unitary_compression",
            expected_norm=torch.tensor(expected, dtype=torch.float64),
            observed_norm=torch.tensor(observed, dtype=torch.float64),
        )

    record(0., 0.5)
    assert opt.get_norm_events()[-1]["local_fidelity"] is None
    assert opt._cumulative_fidelity() == 1.  # Invalid events do not contribute.
    record(1., 0.)
    assert opt.get_norm_events()[-1]["cumulative_infidelity"] == 1.
    record(1., float("nan"))
    assert np.isnan(opt.get_norm_events()[-1]["local_fidelity"])
    assert opt._cumulative_fidelity() == 0.  # Complete loss remains absorbing.


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_complete_loss_overrides_previous_nan_in_ledger(backend):
    opt = MpsOptimizer(qtn.MPS_computational_state("00"), [], chi=2)
    convert = np.asarray if backend == "numpy" else pytest.importorskip("torch").tensor
    for observed in (float("nan"), 0., float("nan")):
        opt._record_norm_event(
            "unitary_compression", expected_norm=convert(1.), observed_norm=convert(observed),
        )
    events = opt.get_norm_events()
    assert np.isnan(events[0]["cumulative_fidelity"])
    assert events[1]["cumulative_fidelity"] == 0.
    assert events[2]["cumulative_infidelity"] == 1.
    # A host-computed physical event must compose with the backend ledger too.
    opt = MpsOptimizer(qtn.MPS_computational_state("00"), [], chi=2)
    opt._record_norm_event(
        "unitary_compression", expected_norm=convert(1.), observed_norm=convert(float("nan")),
    )
    opt._record_norm_event("projective", expected_norm=1., observed_norm=0., physical_boundary=True)
    assert opt._cumulative_fidelity() == 0.


@pytest.mark.parametrize("device", ["cpu", "mps_metadata", "mps"])
def test_torch_ledger_precision_respects_metal_device(device, monkeypatch):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    opt = MpsOptimizer(qtn.MPS_computational_state("00"), [], chi=2)
    actual_device = "mps" if device == "mps" else "cpu"
    expected = torch.tensor(1., dtype=torch.float32, device=actual_device)
    observed = torch.tensor(.8, dtype=torch.float32, device=actual_device)
    with monkeypatch.context() as patch:
        if device == "mps_metadata":
            # Exercise precision selection on CPU CI; this does not validate
            # Metal kernels, which are covered only by the actual-device case.
            patch.setattr(torch.Tensor, "device", property(lambda self: torch.device("mps")))
        opt._record_norm_event("unitary_compression", expected_norm=expected, observed_norm=observed)
    assert opt._norm_log_survival.dtype == (torch.float64 if device == "cpu" else torch.float32)
    assert opt._norm_log_survival.device == observed.device
    assert opt._cumulative_fidelity() == pytest.approx(.64, abs=1e-6)
