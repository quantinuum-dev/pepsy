"""Owned canonical metadata and native gauge reuse on the Pauli path."""

import numpy as np
import pytest
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from pepsy.operators import PauliMPO
from pepsy.optimizers import MpoOptimizer, PauliDMT


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def operator(n=6):
    rng = np.random.default_rng(827)
    terms = [(1. / 2**n, "I" * n)]
    terms += [(.002 / 2**n, "".join(rng.choice(list("IXYZ"), n))) for _ in range(25)]
    return PauliMPO.from_terms(n, terms)


def assert_gauge(op, center):
    for j, core in enumerate(op.to_pauli_cores(copy=False)):
        a = np.asarray(core)
        if j < center:
            m = a.transpose(0, 2, 1).reshape(-1, a.shape[1])
            gram = m.conj().T @ m
        elif j > center:
            m = a.transpose(0, 2, 1).reshape(a.shape[0], -1)
            gram = m @ m.conj().T
        else:
            continue
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=3e-13)


def test_state_copy_and_drivers_have_independent_info_dictionaries():
    op = operator(4).canonicalize_native(center=2)
    copied = op.copy()
    driver = PauliDMT(op, max_bond=16)
    snapshot = driver.state
    ordinary = MpoOptimizer(op.to_mpo(), [], chi=16, mode="dmt", contraction_opt="greedy")
    objects = [op, copied, driver, snapshot, ordinary]
    assert len({id(x.info_c) for x in objects}) == len(objects)
    for obj in objects:
        info = obj.info_c
        before = (obj.p.to_dense() if isinstance(obj, MpoOptimizer)
                  else obj.state.to_dense() if isinstance(obj, PauliDMT) else obj.to_dense())
        assert obj.sync_canonicalization(1) == (1, 1)
        assert obj.info_c is info and info["cur_orthog"] == (1, 1)
        after = (obj.p.to_dense() if isinstance(obj, MpoOptimizer)
                 else obj.state.to_dense() if isinstance(obj, PauliDMT) else obj.to_dense())
        np.testing.assert_allclose(after, before, atol=2e-14)


def test_known_native_center_moves_only_the_required_distance(monkeypatch):
    import pepsy.operators.pauli_mpo as native

    op = operator(7).canonicalize_native(center=1)
    before, info = op.to_dense(), op.info_c
    calls, original = [], native.ar.do

    def observed(name, *args, **kwargs):
        if name == "linalg.qr":
            calls.append(1)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(native.ar, "do", observed)
    op.canonicalize_native(center=4, inplace=True)
    assert len(calls) == 3
    assert op.info_c is info and info["cur_orthog"] == (4, 4)
    calls.clear()
    op.canonicalize_native(center=4, inplace=True)
    assert not calls
    op.canonicalize_native(center=2, inplace=True)
    assert len(calls) == 2
    assert_gauge(op, 2)
    np.testing.assert_allclose(op.to_dense(), before, atol=2e-14)


def test_dmt_reuses_the_state_gauge_without_initial_recanonicalization(monkeypatch):
    from pepsy.operators._pauli_dmt import _PauliDMTWorkspace

    op = operator().canonicalize_native(center=3)
    before = op.to_dense()

    def forbidden(*args, **kwargs):
        raise AssertionError("known state gauge was discarded")

    monkeypatch.setattr(_PauliDMTWorkspace, "move_center", forbidden)
    driver = PauliDMT(op, max_bond=64)
    assert driver.info_c["cur_orthog"] == (3, 3)
    gate = expm(-.13j * np.diag([1., -1., -1., 1.]))
    result = op.apply_gate_dmt(gate, (2, 3), max_bond=64)
    assert result.info_c["cur_orthog"] == (3, 3)
    full = np.kron(np.eye(4), np.kron(gate, np.eye(4)))
    np.testing.assert_allclose(result.to_dense(), full @ before @ full.conj().T, atol=2e-14)
    assert_gauge(result, 3)


@pytest.mark.parametrize("form,center", [(None, 0), ("right", 0), ("left", 5), (0, 0), (2, 2), ("flat", None)])
@pytest.mark.parametrize("inplace", [False, True])
def test_native_compression_records_only_established_gauges(form, center, inplace):
    op = operator()
    info = op.info_c
    result = op.compress_pauli(max_bond=10, form=form, inplace=inplace)
    assert result.info_c["cur_orthog"] == (None if center is None else (center, center))
    if inplace:
        assert result.info_c is info
    else:
        assert result.info_c is not info and info["cur_orthog"] is None
    if center is not None:
        assert_gauge(result, center)


@pytest.mark.parametrize("inplace", [False, True])
def test_dmt_compression_records_its_final_center(inplace):
    op = operator().canonicalize_native(center=2)
    info = op.info_c
    result = op.compress_dmt(max_bond=10, inplace=inplace)
    assert result.info_c["cur_orthog"] == (5, 5)
    assert_gauge(result, 5)
    assert (result.info_c is info) == inplace


@pytest.mark.parametrize("action", ["gate", "channel", "damping"])
def test_managed_inplace_mutation_invalidates_native_metadata(action):
    op = operator(4).canonicalize_native(center=2)
    info = op.info_c
    z = np.diag([1., -1.])
    if action == "gate":
        op.apply_gate(expm(-.1j * z), (1,), inplace=True)
    elif action == "channel":
        op.apply_channel([np.sqrt(.8) * np.eye(2), np.sqrt(.2) * z], (1,), inplace=True)
    else:
        op.apply_daoe(max_weight=2, gamma=.2, inplace=True)
    assert op.info_c is info and info["cur_orthog"] is None


def test_raw_buffer_sync_preserves_operator_and_dictionary_identity():
    op = operator(4).canonicalize_native(center=2)
    info = op.info_c
    op.to_pauli_cores(copy=False)[0][...] *= 1.05
    before = op.to_dense()
    assert op.sync_canonicalization() == (2, 2)
    assert op.info_c is info
    assert_gauge(op, 2)
    np.testing.assert_allclose(op.to_dense(), before, atol=2e-14)


def test_torch_buffer_version_prevents_an_invalid_same_center_noop(monkeypatch):
    torch = pytest.importorskip("torch")
    import pepsy.operators.pauli_mpo as native

    torch.set_num_threads(1)
    op = PauliMPO.from_pauli_cores([torch.tensor(a, dtype=torch.complex128)
                                   for a in operator(4).to_pauli_cores()]).canonicalize_native(center=2)
    info = op.info_c
    op.to_pauli_cores(copy=False)[0].mul_(1.01)
    before = op.to_dense().numpy()
    calls, original = [], native.ar.do

    def observed(name, *args, **kwargs):
        if name == "linalg.qr":
            calls.append(1)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(native.ar, "do", observed)
    op.canonicalize_native(center=2, inplace=True)
    assert calls and op.info_c is info
    np.testing.assert_allclose(op.to_dense().numpy(), before, atol=2e-14)


def test_driver_metadata_tracks_partial_failure_and_independent_snapshots(monkeypatch):
    driver = PauliDMT(operator(5), max_bond=16)
    info, before = driver.info_c, driver.state.to_dense()
    snapshot = driver.state

    def fail(*args, **kwargs):
        driver._workspace.move_center(4)
        raise FloatingPointError("injected split failure")

    monkeypatch.setattr(driver._workspace, "split_pair", fail)
    with pytest.raises(FloatingPointError, match="injected"):
        driver.apply_gate(np.eye(4), (1, 2))
    assert driver.info_c is info and info["cur_orthog"] == (4, 4)
    assert snapshot.info_c["cur_orthog"] == (0, 0)
    np.testing.assert_allclose(driver.state.to_dense(), before, atol=2e-14)


def test_symbolic_metadata_does_not_trigger_native_materialization(monkeypatch):
    op = PauliMPO.from_terms(256, [(1., "X" * 256)])

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata access compiled the sparse operator")

    monkeypatch.setattr(op, "to_pauli_cores", forbidden)
    assert op.info_c == {"cur_orthog": None}


def test_driver_evolution_refreshes_info_without_sharing_snapshot_metadata():
    driver = PauliDMT(operator(4), max_bond=16)
    info = driver.info_c
    h = np.diag([.3, -.7, -.7, .3])
    for _ in range(2):
        result = driver.evolve([h] * 3, dt=.03, steps=2)
        assert driver.info_c is info
        assert result.info_c == info and result.info_c is not info
        assert_gauge(result, info["cur_orthog"][0])
    reports, time = driver.report.bond_reports, driver.time
    driver.sync_canonicalization(1)
    assert driver.report.bond_reports == reports and driver.time == time


def test_evolution_failure_updates_the_public_center(monkeypatch):
    driver = PauliDMT(operator(5), max_bond=16)
    info = driver.info_c
    split, calls = driver._workspace.split_pair, 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            driver._workspace.move_center(4)
            raise FloatingPointError("injected evolution failure")
        return split(*args, **kwargs)

    monkeypatch.setattr(driver._workspace, "split_pair", fail_second)
    with pytest.raises(FloatingPointError, match="injected"):
        driver.evolve([np.diag([.3, -.7, -.7, .3])] * 4, dt=.03)
    assert driver.info_c is info and info["cur_orthog"] == (4, 4)
    assert driver.time == 0.
    assert_gauge(driver.state, 4)


def test_editing_info_alone_cannot_assert_an_unprepared_gauge():
    op = operator(4).canonicalize_native(center=2)
    before = op.to_dense()
    op.info_c["cur_orthog"] = (1, 1)
    op.canonicalize_native(center=1, inplace=True)
    assert_gauge(op, 1)
    np.testing.assert_allclose(op.to_dense(), before, atol=2e-14)


def test_native_qr_failure_keeps_previous_state_and_info(monkeypatch):
    import pepsy.operators.pauli_mpo as native

    op = operator(5).canonicalize_native(center=1)
    before, info = op.to_dense(), op.info_c
    original, calls = native.ar.do, 0

    def fail_second(name, *args, **kwargs):
        nonlocal calls
        if name == "linalg.qr":
            calls += 1
            if calls == 2:
                raise RuntimeError("injected QR failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(native.ar, "do", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        op.canonicalize_native(center=4, inplace=True)
    assert op.info_c is info and info["cur_orthog"] == (1, 1)
    np.testing.assert_array_equal(op.to_dense(), before)


def test_jax_jit_and_gradient_through_cached_native_gauge_moves():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(391)
    cores = [rng.normal(size=shape) + 1j * rng.normal(size=shape)
             for shape in ((1, 2, 4), (2, 2, 4), (2, 1, 4))]
    expected = PauliMPO.from_pauli_cores(cores).trace().real
    with jax.enable_x64(True):
        def value(scale):
            arrays = [jnp.asarray(a) for a in cores]
            arrays[0] = scale * arrays[0]
            op = PauliMPO.from_pauli_cores(arrays, copy=False).canonicalize_native(center=0)
            op.canonicalize_native(center=2, inplace=True)
            return op.trace().real

        actual, gradient = jax.jit(jax.value_and_grad(value))(.7)
        np.testing.assert_allclose(actual, .7 * expected, atol=2e-12)
        np.testing.assert_allclose(gradient, expected, atol=2e-12)


@pytest.mark.parametrize("kind", ["pauli", "ordinary"])
def test_failed_raw_buffer_sync_leaves_an_unknown_gauge(monkeypatch, kind):
    if kind == "pauli":
        import pepsy.operators.pauli_mpo as native

        obj = operator(4).canonicalize_native(center=2)
        obj.to_pauli_cores(copy=False)[0][...] *= 1.03
        original = native.ar.do

        def fail(name, *args, **kwargs):
            if name == "linalg.qr":
                raise RuntimeError("injected sync failure")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(native.ar, "do", fail)
    else:
        obj = MpoOptimizer(operator(4).to_mpo(), [], chi=16, mode="dmt", contraction_opt="greedy")
        obj.p[0].data[...] *= 1.03

        def fail(*args, **kwargs):
            raise RuntimeError("injected sync failure")

        monkeypatch.setattr(obj.p, "canonize", fail)
    info = obj.info_c
    with pytest.raises(RuntimeError, match="injected"):
        obj.sync_canonicalization(1)
    assert obj.info_c is info and info["cur_orthog"] is None
