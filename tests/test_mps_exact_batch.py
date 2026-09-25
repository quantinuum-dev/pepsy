"""Exact fusion preserves state, backend, ownership, and stream boundaries."""

import autoray as ar
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy import MpsOptimizer, backend_torch, rx, rxx, ryy, rz, rzz
from pepsy.optimizers.mps import _exact_batch as batch_module


def _stream(length):
    return (
        [(rx(0.21), (i,)) for i in range(length)]
        + [(rzz(0.37), (i, (i + 2) % length)) for i in range(length)]
        + [(qu.CNOT(), (length - 1, 0)), (qu.hadamard(), (2,))]
        + [(rx(-0.13), (i,)) for i in reversed(range(length))]
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_exact_batch_repeated_replay_preserves_scale_backend_and_input(backend):
    state = qtn.MPS_rand_state(7, 3, dtype="complex128", seed=13)
    state.exponent = 0.3
    gates = _stream(7) + [(np.diag([1.2, 0.4]).astype(complex), (3,))]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = backend_torch(dtype=torch.complex128)
        state.apply_to_arrays(convert)
        gates = [(convert(np.array(g)), where) for g, where in gates]
    before = ar.to_numpy(state.to_dense()).copy()
    reference = MpsOptimizer(state, gates, chi=1, mode="exact")
    batched = MpsOptimizer(state, gates, chi=1, mode="exact-batch")
    for _ in range(2):
        reference.run()
        batched.run()
        np.testing.assert_allclose(
            ar.to_numpy(batched.to_dense()),
            ar.to_numpy(reference.to_dense()),
            atol=2e-13,
            rtol=2e-13,
        )
        assert batched.info_c == {}
        assert ar.infer_backend(batched.p.tensors[0].data) == backend
        assert batched.mps_length_diagnostics() == reference.mps_length_diagnostics()
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), before)
    assert set(batched.p.tags) == set(reference.p.tags)


def test_exact_batch_fuses_single_qubit_layers_and_compact_diagonals():
    gates = _stream(7)
    blocks = list(
        batch_module.iter_exact_batches(
            [g for g, _ in gates],
            [where for _, where in gates],
            "k{}".format,
        )
    )
    assert len(blocks) < len(gates)
    assert len(blocks[0].locations) == 4
    diagonal = next(block for block in blocks if block.diagonal)
    assert len(diagonal.locations) == 7
    assert np.prod(diagonal.operator.shape) == 2**7
    assert all(len(b.inds) <= (12 if b.diagonal else 4) for b in blocks)
    diagonal_layer = list(
        batch_module.iter_exact_batches(
            [np.diag([1, 1j])] * 14,
            [(i,) for i in range(14)],
            "k{}".format,
        )
    )
    assert [len(b.inds) for b in diagonal_layer] == [12, 2]
    assert max(np.prod(b.operator.shape) for b in diagonal_layer) == 4096


def test_exact_batch_keeps_tiny_off_diagonal_terms_and_mutable_gates():
    state = qtn.MPS_computational_state("00", dtype="complex128")
    gate = np.eye(2, dtype=complex)
    gate[1, 0] = 1e-14
    opt = MpsOptimizer(state, [(gate, (0,))], chi=1, mode="exact-batch")
    opt.run()
    assert opt.to_dense().ravel()[2] == pytest.approx(1e-14, rel=1e-12, abs=0)
    gate[1, 0] = 0.25
    opt.run()
    assert opt.to_dense().ravel()[2] == pytest.approx(0.25 + 1e-14, abs=1e-16)


def test_exact_batch_avoids_recontracting_dense_state_and_diagonal_permutation(monkeypatch):
    state = qtn.MPS_computational_state("00000", dtype="complex128")
    opt = MpsOptimizer(state, _stream(5), chi=2, mode="exact-batch")
    opt.run()
    opt.set_gates([(rzz(0.19), (4, 0)), (rzz(-0.23), (1, 3))])
    old = opt.p.tensors[0]
    data, inds = old.data.copy(), old.inds

    def forbidden(*args, **kwargs):
        raise AssertionError("already-dense diagonal replay must not contract")

    monkeypatch.setattr(opt.p, "contract", forbidden)
    opt.run()
    assert opt.p.tensors[0].inds == inds
    np.testing.assert_array_equal(old.data, data)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_exact_batch_flushes_at_measure_reset_condition_and_cap(backend):
    state = qtn.MPS_computational_state("000", dtype="complex128", site_ind_id="q{}")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(backend_torch(dtype=torch.complex128))
    gates = [
        ("h", 0),
        ("cnot", 0, 2),
        ("measure", "Z", 0, -1),
        ("if", -1, 1, ("x", 1)),
        ("reset", 2),
        ("cap", 2, [1, 0]),
        ("x", 0),
    ]
    opt = MpsOptimizer(state, gates, chi=8, mode="exact-batch", ind_id="q{}")
    opt.run(seed=15)
    np.testing.assert_allclose(ar.to_numpy(opt.to_dense()).ravel(), [0, 1, 0, 0], atol=1e-12)
    assert opt.measurements[0][3] == pytest.approx(0.5)
    assert opt.allocated_length == 2
    assert ar.infer_backend(opt.p.tensors[0].data) == backend


def test_exact_batch_torch_gradients_include_zero_off_diagonal_entries():
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("000", dtype="complex128")
    state.apply_to_arrays(backend_torch(dtype=torch.complex128))
    matrix = torch.eye(2, dtype=torch.complex128, requires_grad=True)
    gate = backend_torch(dtype=torch.complex128)(rx(0.2))
    gradients = []
    outputs = []
    for mode in ("exact", "exact-batch"):
        opt = MpsOptimizer(state, [(matrix, (2,)), (gate, (0,)), (matrix, (1,))], chi=1, mode=mode)
        opt.run()
        value = opt.to_dense().real.sum()
        gradients.append(torch.autograd.grad(value, matrix)[0])
        outputs.append(value.detach())
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert abs(gradients[1][1, 0]) > 0.1


def test_exact_batch_mode_transitions_keep_torch_and_physical_order():
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("0000", dtype="complex64", site_ind_id="q{}")
    convert = backend_torch(dtype=torch.complex64)
    state.apply_to_arrays(convert)
    gates = [(convert(np.array(qu.hadamard())), (3,)), (convert(np.array(qu.CNOT())), (3, 0))]
    opt = MpsOptimizer(state, gates, chi=8, mode="exact-batch", ind_id="q{}")
    opt.run()
    expected = opt.to_dense().clone()
    opt.set_mode("exact")
    opt.set_gates([])
    opt.run()
    opt.set_mode("exact-batch")
    opt.set_mode("svd")
    assert opt.info_c["cur_orthog"] is not None
    assert all(t.data.dtype == torch.complex64 for t in opt.p.tensors)
    assert opt.p.site_ind_id == "q{}"
    torch.testing.assert_close(opt.to_dense().ravel(), expected.ravel(), atol=2e-6, rtol=2e-6)


def test_exact_batch_coalesced_kraus_replay_matches_reference():
    state = qtn.MPS_computational_state("11", dtype="complex128")
    stream = [("amplitude_damping", 0.25, 0), ("h", 1), ("x", 0)]
    results = []
    for mode in ("exact", "exact-batch"):
        opt = MpsOptimizer(state, stream, chi=4, mode=mode)
        results.append(opt.run(shots=128, strategy="coalesced", seed=52))
    a, b = results
    assert a.branches == b.branches == 2
    np.testing.assert_array_equal(a.counts, b.counts)
    for left, right in zip(a.leaves, b.leaves):
        assert left.records[0].probability == pytest.approx(right.records[0].probability)
        np.testing.assert_allclose(
            left.optimizer.to_dense(), right.optimizer.to_dense(), atol=1e-13
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_batch_torch_lazy_conjugates_and_device(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    convert = backend_torch(dtype=torch.complex64, device=device)
    state = qtn.MPS_computational_state("00101", dtype="complex64")
    state.apply_to_arrays(convert)
    gates = [(convert(np.array(g)).conj(), where) for g, where in _stream(5)]
    outputs = []
    for mode in ("exact", "exact-batch"):
        opt = MpsOptimizer(state, gates, chi=2, mode=mode)
        opt.run()
        result = opt.to_dense()
        assert result.dtype == torch.complex64
        assert result.device.type == device
        outputs.append(result)
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-6)


def test_exact_batch_preserves_exact_mode_restrictions_and_cyclic_input():
    state = qtn.MPS_rand_state(4, 2, cyclic=True, dtype="complex128", seed=2)
    opt = MpsOptimizer(state, [("x", 1)], chi=4, mode="exact-batch")
    with pytest.raises(ValueError, match="automatic normalization"):
        opt.run(non_unitary=True, normalize_final=True)
    opt.run()
    assert opt.info_c == {}
    opt.set_mode("svd")
    assert not opt.p.cyclic


def test_exact_batch_qudit_falls_back_without_coercion():
    state = qtn.MPS_rand_state(3, 2, phys_dim=3, dtype="complex128", seed=3)
    gates = [(np.diag([1.0, 0.8, -0.2]).astype(complex), (1,))]
    a = MpsOptimizer(state, gates, chi=4, mode="exact")
    b = MpsOptimizer(state, gates, chi=4, mode="exact-batch")
    a.run()
    b.run()
    np.testing.assert_allclose(a.to_dense(), b.to_dense(), atol=1e-13)


def test_exact_batch_rejects_persistent_layout_and_gibbs_replay():
    from pepsy import GibbsMps

    state = qtn.MPS_computational_state("000")
    opt = MpsOptimizer(state, [], chi=4, mode="svd")
    opt.apply_layout((2, 0, 1), layout_report=False)
    with pytest.raises(ValueError, match="persistent-layout"):
        opt.set_mode("exact-batch")
    assert opt.mode == "svd"
    with pytest.raises(ValueError, match="ordinary open MPS"):
        GibbsMps([(("Z", 0.3), 0)], shape=2).prepare(0.1, mode="exact-batch")


def test_exact_batch_cupy_preserves_device_and_state():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA is unavailable")
    if not available:
        pytest.skip("CUDA is unavailable")
    state = qtn.MPS_computational_state("00101", dtype="complex64")
    state.apply_to_arrays(cp.asarray)
    gates = [(cp.asarray(g, dtype=cp.complex64), where) for g, where in _stream(5)]
    outputs = []
    for mode in ("exact", "exact-batch"):
        opt = MpsOptimizer(state, gates, chi=1, mode=mode)
        opt.run()
        result = opt.to_dense()
        assert result.dtype == cp.complex64
        assert result.device == state.tensors[0].data.device
        outputs.append(result)
    cp.testing.assert_allclose(outputs[0], outputs[1], atol=2e-6, rtol=2e-6)


def test_exact_batch_native_symmray_uses_reference_fallback():
    pytest.importorskip("symmray")
    from pepsy.tensors import SymMPS, site_charge_from_occupations
    from pepsy import rz

    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=1,
        dtype="complex128",
    )
    gate = state.operator_from_dense(rz(0.1), charge=0, sites=1)
    a = MpsOptimizer(state.tn, [(gate, (2,))], chi=4, mode="exact")
    b = MpsOptimizer(state.tn, [(gate, (2,))], chi=4, mode="exact-batch")
    a.run()
    b.run()
    assert ar.infer_backend(b.p.tensors[0].data) == "symmray"
    np.testing.assert_allclose(
        a.p.tensors[0].data.to_dense(),
        b.p.tensors[0].transpose(*a.p.tensors[0].inds).data.to_dense(),
        atol=1e-13,
    )
    before_transition = b.to_dense().to_dense().ravel()
    b.set_mode("svd")
    assert ar.infer_backend(b.p.tensors[0].data) == "symmray"
    assert b.info_c["cur_orthog"] is not None
    np.testing.assert_allclose(
        b.to_dense().to_dense().ravel(), before_transition, atol=1e-13
    )



@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_exact_batch_equal_zz_layer_uses_one_phase_pass(backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA is unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA is unavailable")
    else:
        pytest.importorskip("numba")
        cp = None

    n = 17
    state = qtn.MPS_product_state(
        [np.array([1.0, 0.15j * (i + 1)]) for i in range(n)]
    )
    gate = np.asarray(rzz(0.17), dtype=np.complex64)
    edges = [(i, i + 1) for i in range(n - 1)]
    edges[3] = tuple(reversed(edges[3]))
    gates = [(gate, edge) for edge in edges]
    if cp is not None:
        state.apply_to_arrays(lambda array: cp.asarray(array, dtype=cp.complex64))
        gate = cp.asarray(gate)
        gates = [(gate, edge) for edge in edges]
    else:
        state.apply_to_arrays(lambda array: np.asarray(array, dtype=np.complex64))

    blocks = list(batch_module.iter_exact_batches(
        [value for value, _ in gates],
        [where for _, where in gates],
        "k{}".format,
        backend=backend,
        state_size=2**n,
    ))
    assert len(blocks) == 1
    assert blocks[0].kind == "phase"
    assert len(blocks[0].locations) == n - 1

    reference = MpsOptimizer(state, gates, chi=1, mode="exact")
    optimized = MpsOptimizer(state, gates, chi=1, mode="exact-batch")
    reference.run()
    optimized.run()
    np.testing.assert_allclose(
        ar.to_numpy(optimized.to_dense()),
        ar.to_numpy(reference.to_dense()),
        atol=2e-5,
        rtol=2e-5,
    )

    varied = [(np.asarray(rzz(0.03 * (i + 1))), edge)
              for i, edge in enumerate(edges)]
    blocks = list(batch_module.iter_exact_batches(
        [value for value, _ in varied],
        [where for _, where in varied],
        "k{}".format,
        backend="numpy",
        state_size=2**n,
    ))
    assert all(not isinstance(block, batch_module.ExactStructuredBatch)
               for block in blocks)

    # A repeated edge can form a second value class or split the phase run.
    # Replay must still count every original gate.
    repeated = gates + gates[:1]
    blocks = list(batch_module.iter_exact_batches(
        [value for value, _ in repeated],
        [where for _, where in repeated],
        "k{}".format,
        backend=backend,
        state_size=2**n,
    ))
    assert sum(len(block.locations) for block in blocks) == len(repeated)
    reference = MpsOptimizer(state, repeated, chi=1, mode="exact")
    optimized = MpsOptimizer(state, repeated, chi=1, mode="exact-batch")
    reference.run()
    optimized.run()
    np.testing.assert_allclose(
        ar.to_numpy(optimized.to_dense()),
        ar.to_numpy(reference.to_dense()),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_exact_batch_two_value_zz_layer_uses_one_phase_pass(backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA is unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA is unavailable")
    else:
        pytest.importorskip("numba")
        cp = None

    # The extra two CuPy sites make the full-state traffic large enough for
    # one grouped pass to beat four ordinary diagonal blocks.
    nx, ny = 4, 5
    n = nx * ny + (2 if cp is not None else 0)
    state = qtn.MPS_product_state(
        [np.array([1.0, 0.08 + 0.01j * i], dtype=np.complex64)
         for i in range(n)]
    )
    gates_by_class = (
        np.asarray(rzz(0.17), dtype=np.complex64),
        np.diag(np.asarray(
            [1.01 + 0.02j, 0.97 - 0.03j,
             0.97 - 0.03j, 1.01 + 0.02j],
            dtype=np.complex64,
        )),
    )
    x_edges = [(x * ny + y, (x + 1) * ny + y)
               for x in range(nx - 1) for y in range(ny)]
    y_edges = [(x * ny + y, x * ny + y + 1)
               for x in range(nx) for y in range(ny - 1)]
    edges = x_edges + y_edges
    edges[3] = tuple(reversed(edges[3]))
    gates = [(gates_by_class[int(i >= len(x_edges))], edge)
             for i, edge in enumerate(edges)]
    if cp is not None:
        state.apply_to_arrays(cp.asarray)
        gates = [(cp.asarray(gate), edge) for gate, edge in gates]

    def planned(stream, size):
        return list(batch_module.iter_exact_batches(
            [gate for gate, _ in stream],
            [where for _, where in stream],
            "k{}".format,
            backend=backend,
            state_size=size,
        ))

    blocks = planned(gates, 2**n)
    assert len(blocks) == 1
    assert blocks[0].kind == "phase"
    assert len(blocks[0].locations) == len(edges)
    small_blocks = planned(gates, 2**17)
    assert not (len(small_blocks) == 1
                and getattr(small_blocks[0], "kind", None) == "phase")

    before = ar.to_numpy(state.to_dense()).copy()
    reference = MpsOptimizer(state, gates, chi=1, mode="exact")
    optimized = MpsOptimizer(state, gates, chi=1, mode="exact-batch")
    reference.run()
    optimized.run()
    np.testing.assert_allclose(
        ar.to_numpy(optimized.to_dense()),
        ar.to_numpy(reference.to_dense()),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), before)

    three_classes = gates + [
        (cp.asarray(rzz(0.31), dtype=cp.complex64)
         if cp is not None else np.asarray(rzz(0.31), dtype=np.complex64),
         (0, n - 1))
    ]
    assert all(getattr(block, "kind", None) != "phase"
               for block in planned(three_classes, 2**n))


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_exact_batch_compacts_interleaved_z_stream_and_replans_mutations(backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA is unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA is unavailable")
    else:
        pytest.importorskip("numba")
        cp = None

    nx, ny = 4, 5
    active = nx * ny
    n = active + (2 if cp is not None else 0)
    state = qtn.MPS_product_state([
        np.array([1.0, 0.06 + 0.005j * i], dtype=np.complex64)
        for i in range(n)
    ])
    edges = (
        [(x * ny + y, (x + 1) * ny + y)
         for x in range(nx - 1) for y in range(ny)]
        + [(x * ny + y, x * ny + y + 1)
           for x in range(nx) for y in range(ny - 1)]
    )
    two = np.asarray(rzz(0.17), dtype=np.complex64)
    one = np.asarray(rz(0.09), dtype=np.complex64)
    if cp is not None:
        state.apply_to_arrays(cp.asarray)
        two, one = cp.asarray(two), cp.asarray(one)
    gates = []
    for i in range(len(edges)):
        gates.append((two, edges[i]))
        if i < active:
            gates.append((one, (i,)))
        gates.append((two, edges[i][::-1]))
        if i < active:
            gates.append((one, (i,)))

    def planned(stream):
        return list(batch_module.iter_exact_batches(
            [gate for gate, _ in stream],
            [where for _, where in stream],
            "k{}".format,
            backend=backend,
            state_size=2**n,
        ))

    blocks = planned(gates)
    assert len(blocks) == 1 and blocks[0].kind == "phase"
    assert len(blocks[0].locations) == len(gates)
    assert len(blocks[0].operator[0]) == len(edges) + active
    assert len({frozenset(where) for where in blocks[0].operator[0]}) == (
        len(edges) + active
    )
    barrier = planned(gates + [(rx(0.11), (0,))] + gates)
    assert [block.kind for block in barrier
            if isinstance(block, batch_module.ExactStructuredBatch)] == [
                "phase", "phase",
            ]

    before = ar.to_numpy(state.to_dense()).copy()
    reference = MpsOptimizer(state, gates, chi=1, mode="exact")
    optimized = MpsOptimizer(state, gates, chi=1, mode="exact-batch")
    for replay in range(2):
        if replay:
            one[1, 1] *= 1.03
            assert not np.array_equal(
                blocks[0].operator[2], planned(gates)[0].operator[2]
            )
        reference.run()
        optimized.run()
        np.testing.assert_allclose(
            ar.to_numpy(optimized.to_dense()),
            ar.to_numpy(reference.to_dense()),
            atol=3e-5,
            rtol=3e-5,
        )
        assert ar.infer_backend(optimized.p.tensors[0].data) == backend
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), before)


def test_exact_batch_pure_z_gpu_plan_avoids_short_phase_pass():
    gate = np.asarray(rz(0.09), dtype=np.complex64)
    n = 22
    blocks = list(batch_module.iter_exact_batches(
        [gate] * n,
        [(i,) for i in range(n)],
        "k{}".format,
        backend="cupy",
        state_size=2**n,
    ))
    assert len(blocks) == 2
    assert all(getattr(block, "kind", None) != "phase" for block in blocks)


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_exact_batch_same_pair_parity_preserves_order_and_scale(backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA is unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA is unavailable")
    else:
        pytest.importorskip("numba")
        cp = None

    n = 17
    state = qtn.MPS_product_state(
        [np.array([1.0, 0.1 + 0.02j * (i + 1)]) for i in range(n)]
    )
    nonunitary = np.zeros((4, 4), dtype=np.complex64)
    nonunitary[0, 0] = 1.1
    nonunitary[0, 3] = 0.04j
    nonunitary[3, 0] = -0.07
    nonunitary[3, 3] = 0.9
    nonunitary[1, 1] = 0.8
    nonunitary[1, 2] = 0.03
    nonunitary[2, 1] = -0.05j
    nonunitary[2, 2] = 1.2
    gates = [
        (np.asarray(rxx(0.23), dtype=np.complex64), (0, n - 1)),
        (np.asarray(ryy(-0.19), dtype=np.complex64), (n - 1, 0)),
        (np.asarray(rzz(0.31), dtype=np.complex64), (0, n - 1)),
        (nonunitary, (n - 1, 0)),
    ]
    if cp is not None:
        state.apply_to_arrays(lambda array: cp.asarray(array, dtype=cp.complex64))
        gates = [(cp.asarray(gate), where) for gate, where in gates]
    else:
        state.apply_to_arrays(lambda array: np.asarray(array, dtype=np.complex64))

    blocks = list(batch_module.iter_exact_batches(
        [value for value, _ in gates],
        [where for _, where in gates],
        "k{}".format,
        backend=backend,
        state_size=2**n,
    ))
    assert len(blocks) == 1
    assert blocks[0].kind == "parity"
    reference = MpsOptimizer(state, gates, chi=1, mode="exact")
    optimized = MpsOptimizer(state, gates, chi=1, mode="exact-batch")
    reference.run()
    optimized.run()
    np.testing.assert_allclose(
        ar.to_numpy(optimized.to_dense()),
        ar.to_numpy(reference.to_dense()),
        atol=2e-5,
        rtol=2e-5,
    )

    perturbed = np.asarray(rxx(0.23)).reshape(4, 4).copy()
    perturbed[0, 1] = 1e-14
    unsafe = [(perturbed, (0, n - 1)), (np.asarray(ryy(0.2)), (0, n - 1))]
    blocks = list(batch_module.iter_exact_batches(
        [value for value, _ in unsafe],
        [where for _, where in unsafe],
        "k{}".format,
        backend="numpy",
        state_size=2**n,
    ))
    assert all(not isinstance(block, batch_module.ExactStructuredBatch)
               for block in blocks)



def test_exact_batch_grouped_zz_preserves_nonunitary_scale():
    pytest.importorskip("numba")
    n = 17
    state = qtn.MPS_product_state(
        [np.array([1.0, 0.1j * (i + 1)], dtype=np.complex64)
         for i in range(n)]
    )
    gate = np.diag(
        np.asarray([1.02 + 0.01j, 0.94 - 0.02j,
                    0.94 - 0.02j, 1.02 + 0.01j], dtype=np.complex64)
    )
    gates = [(gate, (i, i + 1)) for i in range(n - 1)]
    reference = MpsOptimizer(state, gates, chi=1, mode="exact")
    optimized = MpsOptimizer(state, gates, chi=1, mode="exact-batch")
    reference.run()
    optimized.run()
    np.testing.assert_allclose(
        optimized.to_dense(),
        reference.to_dense(),
        atol=2e-5,
        rtol=2e-5,
    )
