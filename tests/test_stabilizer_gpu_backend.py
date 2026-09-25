"""Backend preservation for stabilizer coefficient evolution and bookkeeping."""

from types import SimpleNamespace

import autoray as ar
import numpy as np
import pytest

pytest.importorskip("stim")

from pepsy import StabilizerMpsSimulator, StabilizerTreeSimulator
from pepsy.optimizers.stabilizer_tn import mps_stab_optimizer as mps_module
from pepsy.optimizers.tree_stabilizer import optimizer as tree_module
from pepsy.optimizers.stabilizer_tn.operators import pauli_combo_submpo, pauli_sum_submpo
from pepsy.optimizers.stabilizer_tn._backend import compression_event, diagnostic_scalar, scalar_to_host

pytestmark = [pytest.mark.integration, pytest.mark.optional]


@pytest.fixture(params=["torch", "jax", "cuda", "cupy"])
def convert(request):
    if request.param in {"torch", "cuda"}:
        torch = pytest.importorskip("torch")
        device = "cuda" if request.param == "cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        return lambda x: torch.as_tensor(x, dtype=torch.complex64, device=device).clone()
    if request.param == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return lambda x: jnp.asarray(x, dtype=jnp.complex64)
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CuPy GPU unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy GPU unavailable")
    return lambda x: cp.asarray(x, dtype=cp.complex64)


STREAM = [("h", 1), ("cnot", 1, 2), ("rxx", .31, 0, 3),
          ("rzz", .27, 1, 3), ("rot", .43, "XYZ", (0, 1, 2))] * 2


@pytest.mark.parametrize("simulator", [StabilizerMpsSimulator, StabilizerTreeSimulator])
def test_named_replay_matches_numpy_without_host_tensor_construction(convert, simulator, monkeypatch):
    opts = dict(chi=2, cutoff=0., exact_cooling=False, dtype="complex64", seed=11)
    reference = simulator(4, **opts).apply(STREAM)
    sim = simulator(4, to_backend=convert, **opts)
    original = ar.to_numpy

    def scalar_only(array):
        assert ar.size(array) == 1, "coefficient array downloaded during replay"
        return original(array)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", scalar_only)
        sim.apply(STREAM)
    assert sim.backend_info()["backend"] == ar.infer_backend(convert(0.))
    actual, expected = sim.to_statevector(), reference.to_statevector()
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, expected * phase / abs(phase), atol=2e-5)
    assert sim.norm() == pytest.approx(reference.norm(), abs=2e-5)
    assert isinstance(sim.norm_diagnostics()["local_fidelity"], float)
    assert sim.norm_diagnostics()["cumulative_fidelity"] == pytest.approx(
        reference.norm_diagnostics()["cumulative_fidelity"], abs=2e-5)


@pytest.mark.parametrize("simulator", [StabilizerMpsSimulator, StabilizerTreeSimulator])
def test_torch_unitary_ledger_does_not_read_scalars_or_clocks(simulator, monkeypatch):
    torch = pytest.importorskip("torch")
    convert = lambda x: torch.tensor(np.array(x), dtype=torch.complex64, requires_grad=True)
    sim = simulator(4, chi=2, cutoff=0., exact_cooling=False, to_backend=convert)

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary coefficient replay read a host scalar or clock")

    with monkeypatch.context() as patch:
        for name in ("item", "__float__", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(ar, "to_numpy", forbidden)
        for module in (mps_module, tree_module):
            patch.setattr(module, "time", SimpleNamespace(perf_counter=forbidden))
        sim.apply(STREAM)
    if simulator is StabilizerMpsSimulator:
        assert isinstance(sim._current_infidelity, torch.Tensor)
        assert not sim._current_infidelity.requires_grad
        assert isinstance(sim.get_infidelities()[-1], float)
        assert sim.get_compression_norm_events()[-1]["valid"] is True
    else:
        assert isinstance(sim.tree_optimizer._norm_log_survival, torch.Tensor)
        assert not sim.tree_optimizer._norm_log_survival.requires_grad


def test_pauli_builders_construct_on_selected_backend(convert):
    like = convert(0.)
    terms = {0: "X", 3: "Y"}
    branches = [(.3, {}), (.2j, terms), (.1, {1: "Z"})]
    for builder, args in [(pauli_combo_submpo, (.3, .2j, terms, 4)),
                          (pauli_sum_submpo, (branches, 4))]:
        ref, sites = builder(*args, dtype="complex64")
        op, actual_sites = builder(*args, like=like)
        assert sites == actual_sites
        assert all(ar.infer_backend(t.data) == ar.infer_backend(like) for t in op.tensors)
        assert all(ar.get_dtype_name(t.data) == "complex64" for t in op.tensors)
        np.testing.assert_allclose(ar.to_numpy(op.to_dense()), ref.to_dense(), atol=1e-6)


def test_pauli_builder_uses_autoray_fallback_without_namespace(monkeypatch):
    torch = pytest.importorskip("torch")
    like = torch.ones((), dtype=torch.complex64)
    branches = [(1., {0: "X", 3: "Y"}), (.2j, {1: "Z"})]
    reference, _ = pauli_sum_submpo(branches, 4, like=like)
    monkeypatch.delattr(ar, "get_namespace")
    actual, _ = pauli_sum_submpo(branches, 4, like=like)
    assert all(isinstance(t.data, torch.Tensor) for t in actual.tensors)
    torch.testing.assert_close(actual.to_dense(), reference.to_dense())


def test_real_coefficient_state_keeps_y_rotation():
    torch = pytest.importorskip("torch")
    sim = StabilizerMpsSimulator(2, dtype="float64", exact_cooling=False,
                                 to_backend=lambda x: torch.as_tensor(x, dtype=torch.float64))
    sim.apply([("ry", .37, 0)])
    np.testing.assert_allclose(sim.to_statevector(),
                               [np.cos(.37 / 2), 0., np.sin(.37 / 2), 0.], atol=1e-12)
    assert all(t.data.dtype == torch.float64 for t in sim.p.tensors)


@pytest.mark.parametrize("simulator", [StabilizerMpsSimulator, StabilizerTreeSimulator])
def test_exact_cooling_keeps_coefficient_tensors_on_backend(convert, simulator, monkeypatch):
    opts = dict(chi=4, cutoff=0., dtype="complex64")
    stream = [("rxx", .31, 0, 3), ("ryy", .47, 1, 2)]
    reference = simulator(4, **opts).apply(stream)
    sim = simulator(4, to_backend=convert, **opts)
    original = ar.to_numpy

    def decision_only(array):
        assert ar.size(array) <= 3, "exact cooling downloaded a coefficient tensor"
        return original(array)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", decision_only)
        sim.apply(stream)
    assert sim.exact_cooling_events
    actual, expected = sim.to_statevector(), reference.to_statevector()
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, expected * phase / abs(phase), atol=2e-5)


@pytest.mark.parametrize("expand", [False, True])
def test_mps_random_fit_guess_uses_backend_rng(convert, expand):
    sim = StabilizerMpsSimulator(4, chi=4, to_backend=convert,
                                 fit_init_rand_strength=.03, fit_init_seed=7)
    outputs = [sim._fit_randomized_guess(sim.p, range(4), block_size=2, expand=expand)
               for _ in range(2)]
    assert all(ar.infer_backend(t.data) == sim.backend for t in outputs[0].tensors)
    assert outputs[0] is not sim.p
    np.testing.assert_array_equal(ar.to_numpy(outputs[0].to_dense()), ar.to_numpy(outputs[1].to_dense()))


def test_device_compression_ledger_zero_nonfinite_and_underflow_policy(convert, monkeypatch):
    cpu = StabilizerMpsSimulator(2)
    for before, after in [(0., -.1), (0., -30.), (-np.inf, 0.),
                          (0., -np.inf), (np.inf, np.inf), (np.nan, 0.)]:
        cpu._compression_segment_log_survival = 0.
        cpu._current_infidelity = 0.
        cpu._compression_norm_events.clear()
        monkeypatch.setattr(cpu, "_norm_log10", lambda: after)
        cpu._record_compression_norm_event(None, 0., before_log_norm=before)
        log, loss, record = compression_event(
            diagnostic_scalar(convert(before)), diagnostic_scalar(convert(after)),
            0., 0., 0., step=1, kind="unitary_compression")
        record = scalar_to_host(record)
        assert bool(record["valid"]) == bool(cpu._compression_norm_events)
        assert scalar_to_host(log) == pytest.approx(cpu._compression_segment_log_survival, abs=2e-5)
        assert scalar_to_host(loss) == pytest.approx(cpu._current_infidelity, abs=1e-6)


def test_device_ledger_copy_and_measurement_boundaries(convert):
    sim = StabilizerMpsSimulator(4, chi=2, cutoff=0., exact_cooling=False,
                                 dtype="complex64", to_backend=convert, seed=8).apply(STREAM)
    clone = sim.copy()
    before = sim.get_compression_norm_events()
    clone.apply([("rxx", .61, 0, 2)])
    assert sim.get_compression_norm_events() == before
    # Reading diagnostics produces independent dictionaries, with host floats.
    clone.get_compression_norm_events()[0]["local_fidelity"] = -1.
    assert clone.get_compression_norm_events()[0]["local_fidelity"] >= 0.
    sim.measure("Z", 0)
    assert sim.norm() == pytest.approx(1., abs=2e-6)
    assert sim.get_norm_events()[-1]["valid"] is True
    sim.apply([("rxx", .21, 1, 3)])
    assert np.isfinite(sim.norm_diagnostics()["cumulative_fidelity"])


@pytest.mark.parametrize("exponent", [-155., 155.])
def test_mps_device_ledger_preserves_exponent_transitions(convert, exponent):
    sim = StabilizerMpsSimulator(4, chi=2, cutoff=0., exact_cooling=False,
                                 dtype="complex64", to_backend=convert).apply(STREAM)
    sim.p.exponent = exponent
    sim.apply([("rxx", .23, 0, 3)])
    assert sim.norm() > 0. and np.isfinite(sim.norm())
    assert np.isfinite(sim.get_compression_norm_events()[-1]["local_infidelity"])
    sim.p.exponent = 0.
    sim.apply([("rxx", .17, 0, 3)])
    assert np.isfinite(sim.norm_diagnostics()["cumulative_fidelity"])
