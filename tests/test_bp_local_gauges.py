"""Local SU gauge extraction preserves fitted states and exterior tensors."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    apply_reduced_loop_cluster_gate, compress_reduced_loop_cluster,
    prepare_reduced_bond_pair,
)
from pepsy.bp.reduced_update import ReducedALSSolution, _local_svd_reduced_state
from pepsy.operators import gate_loop_cluster


@pytest.mark.parametrize("kind", ["peps", "pepo"])
@pytest.mark.parametrize("backend", ["numpy", "torch", "cuda"])
def test_tree_gauge_matches_independent_schmidt_spectrum(kind, backend):
    """An untruncated chain must retain the physical Schmidt weights."""
    factory = qtn.PEPS if kind == "peps" else qtn.PEPO
    core = factory.rand(2, 3, bond_dim=1, dtype="complex128", seed=58)
    if backend != "numpy":
        torch = pytest.importorskip("torch")
        if backend == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        device = "cuda" if backend == "cuda" else "cpu"
        core.apply_to_arrays(lambda x: torch.as_tensor(x, device=device))
    gauges = {}
    rng = np.random.default_rng(998)
    for where in (((0, 0), (0, 1)), ((0, 1), (0, 2))):
        gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        if backend != "numpy":
            gate = torch.as_tensor(gate, device=device)
        core = gate_loop_cluster(
            core, gate, where=where, gauges=gauges,
            which="upper" if kind == "pepo" else None,
            max_bond=4, base_radius=0, psd_project=False, optimize="greedy",
            als_opts={"max_iterations": 2}, inplace=True,
        )
    physical = core.copy()
    physical.gauge_simple_insert(gauges)
    nlegs = 6 if kind == "peps" else 12
    data = _dense(physical.to_dense()).reshape((2,) * nlegs)
    axes = (2,) if kind == "peps" else (2, 8)
    matrix = np.moveaxis(data, axes, tuple(range(nlegs - len(axes), nlegs)))
    spectrum = np.linalg.svd(matrix.reshape(-1, 2 ** len(axes)), compute_uv=False)
    actual = _dense(gauges[core.bond((0, 1), (0, 2))])
    np.testing.assert_allclose(
        actual / np.linalg.norm(actual), spectrum / np.linalg.norm(spectrum),
        rtol=2e-11, atol=2e-11,
    )


def test_zero_cluster_gate_stream_recovers_native_simple_update():
    core = qtn.PEPS.rand(2, 3, bond_dim=1, dtype="complex128", seed=58)
    native = core.copy()
    gauges, native_gauges = {}, {}
    rng = np.random.default_rng(998)
    # Adjacent gates create nonuniform exterior gauges before truncation.
    bonds = (((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 0), (1, 0)),
             ((1, 0), (1, 1)), ((0, 1), (1, 1)))
    for where in bonds:
        gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        native.gate_simple_(gate, where=where, gauges=native_gauges,
                            max_bond=2, renorm=False, cutoff=0, smudge=0)
        core = gate_loop_cluster(core, gate, where=where, gauges=gauges,
            max_bond=2, base_radius=0, psd_project=False, optimize="greedy",
            als_opts=dict(max_iterations=40, tol=1e-13, monitor_convergence=True),
            inplace=True)
        expected, actual = native.copy(), core.copy()
        expected.gauge_simple_insert(native_gauges)
        actual.gauge_simple_insert(gauges)
        error = np.linalg.norm(actual.to_dense() - expected.to_dense())
        assert error / np.linalg.norm(expected.to_dense()) < 1e-6


def _dense(value):
    return np.asarray(ar.to_numpy(value))


@pytest.mark.parametrize("kind", ["peps", "pepo"])
@pytest.mark.parametrize("mode", ["physical", "su_core", "zero_gauge"])
@pytest.mark.parametrize("backend", ["numpy", "torch", "cuda"])
def test_local_svd_preserves_optimized_state_and_exterior(kind, mode, backend):
    factory = qtn.PEPS if kind == "peps" else qtn.PEPO
    tn = factory.rand(3, 2, bond_dim=2, phys_dim=2, dtype="complex128", seed=61)
    tn.exponent = 1.25
    if backend != "numpy":
        torch = pytest.importorskip("torch")
        if backend == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        for tensor in tn.tensors:
            tensor.modify(data=torch.as_tensor(
                tensor.data, device="cuda" if backend == "cuda" else "cpu",
            ))
    gauges = {
        index: ar.do("array", np.array([0.4, 1.3]), like=tn.tensors[0].data)
        for index in tn.inner_inds()
    } if mode != "physical" else {}
    where = ((1, 0), (1, 1))
    if mode == "zero_gauge":
        index = tn.bond((0, 0), (1, 0))
        gauges[index] = ar.do("array", np.array([0.0, 1.3]), like=tn.tensors[0].data)
    original = {tid: _dense(t.data).copy() for tid, t in tn.tensor_map.items()}
    old_gauges = {index: _dense(g).copy() for index, g in gauges.items()}
    pair = prepare_reduced_bond_pair(
        tn, gauges, where=where, run_bp=False,
        input_mode="physical" if mode == "physical" else "su_core",
    )
    solution = ReducedALSSolution(
        left=2.3 * pair.r_left.data, right=0.7 * pair.l_right.data, costs=(),
    )
    expected = pair.reconstruct_tn(solution.left, solution.right)
    core, updated, physical, info, _ = _local_svd_reduced_state(
        tn, pair, solution, inplace=False,
    )
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(updated)
    for actual in (rebuilt, physical):
        np.testing.assert_allclose(
            _dense(actual.to_dense()), _dense(expected.to_dense()),
            rtol=2e-11, atol=2e-11,
        )
    # Independent full-site SU-weighted spectrum, not the reduced-coordinate
    # spectrum: the latter omits half of every exterior gauge.
    weighted = [expected.tensor_map[tid].copy() for tid in (pair.left_tid, pair.right_tid)]
    for tensor in weighted:
        for index in tensor.inds:
            if index != pair.bond_ind and index in gauges:
                tensor.multiply_index_diagonal_(index, gauges[index] ** 0.5)
    left_inds, right_inds = [tuple(ix for ix in t.inds if ix != pair.bond_ind) for t in weighted]
    matrix = (weighted[0] @ weighted[1]).to_dense(left_inds, right_inds)
    spectrum = np.linalg.svd(_dense(matrix), compute_uv=False)
    np.testing.assert_allclose(_dense(updated[pair.bond_ind]), spectrum[:2], rtol=1e-12)
    assert info["method"] == "local_svd"
    for index, gauge in old_gauges.items():
        np.testing.assert_array_equal(_dense(gauges[index]), gauge)
        if index != pair.bond_ind:
            np.testing.assert_array_equal(_dense(updated[index]), gauge)
    for tid, data in original.items():
        np.testing.assert_array_equal(_dense(tn.tensor_map[tid].data), data)
        if tid not in (pair.left_tid, pair.right_tid):
            np.testing.assert_array_equal(_dense(core.tensor_map[tid].data), data)
    if backend != "numpy":
        assert all(isinstance(t.data, torch.Tensor) for t in core.tensors)
        assert updated[pair.bond_ind].device == tn.tensors[0].data.device


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("operation", ["gate", "compression"])
def test_gate_update_keeps_exterior_gauges_and_never_runs_global_su(monkeypatch, inplace, operation):
    import pepsy.bp as bp
    import pepsy.bp.gauges as gauge_module

    tn = qtn.PEPS.rand(3, 2, bond_dim=2, dtype="complex128", seed=71)
    gauges = {index: np.array([0.6, 1.2]) for index in tn.inner_inds()}
    originals = {index: g.copy() for index, g in gauges.items()}
    tensors = {tid: t.data.copy() for tid, t in tn.tensor_map.items()}
    spectator_arrays = {tid: t.data for tid, t in tn.tensor_map.items()}

    def forbidden(*args, **kwargs):
        pytest.fail("a local pair update ran gauge_all_simple")

    monkeypatch.setattr(bp, "gauge_all_simple", forbidden)
    monkeypatch.setattr(gauge_module, "gauge_all_simple", forbidden)
    options = dict(
        where=((1, 0), (1, 1)), max_bond=1,
        als_opts={"max_iterations": 2}, inplace=inplace,
    )
    if operation == "gate":
        result = apply_reduced_loop_cluster_gate(
            tn, gauges, np.diag([1.0, 1j, -1j, 1.0]), **options,
        )
    else:
        result = compress_reduced_loop_cluster(
            tn, gauges=gauges, regauge=True, **options,
        )
    assert (result.core is tn) == inplace
    for index, old in originals.items():
        if index != result.pair.bond_ind:
            np.testing.assert_array_equal(result.gauges[index], old)
            np.testing.assert_array_equal(gauges[index], old)
    for tid, data in tensors.items():
        if tid not in (result.pair.left_tid, result.pair.right_tid):
            np.testing.assert_array_equal(result.core.tensor_map[tid].data, data)
            assert result.core.tensor_map[tid].data is spectator_arrays[tid]
    if not inplace:
        for tid, data in tensors.items():
            np.testing.assert_array_equal(tn.tensor_map[tid].data, data)
        np.testing.assert_array_equal(gauges[result.pair.bond_ind], originals[result.pair.bond_ind])
    else:
        assert result.gauges is gauges
    rebuilt = result.core.copy()
    rebuilt.gauge_simple_insert(result.gauges)
    np.testing.assert_allclose(rebuilt.to_dense(), result.physical_tn.to_dense(), rtol=1e-11, atol=1e-11)


def test_gate_stream_initializes_only_identity_gauges(monkeypatch):
    import pepsy.bp as bp
    import pepsy.bp.gauges as gauge_module

    def forbidden(*args, **kwargs):
        pytest.fail("gate stream ran gauge_all_simple")

    monkeypatch.setattr(bp, "gauge_all_simple", forbidden)
    monkeypatch.setattr(gauge_module, "gauge_all_simple", forbidden)
    tn = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=73)
    gauges = {}
    out, results = gate_loop_cluster(
        tn, np.eye(4), where=((0, 0), (0, 1)), gauges=gauges,
        max_bond=2, als_opts={"max_iterations": 1},
        inplace=False, return_results=True,
    )
    assert results[0].su_info["method"] == "local_svd"
    for index, gauge in gauges.items():
        if index != results[0].pair.bond_ind:
            np.testing.assert_array_equal(gauge, np.ones(2))
    rebuilt = out.copy()
    rebuilt.gauge_simple_insert(gauges)
    np.testing.assert_allclose(rebuilt.to_dense(), results[0].physical_tn.to_dense(), rtol=1e-11, atol=1e-11)


def test_two_gate_stream_preserves_state_amplitude_without_truncation():
    tn = qtn.PEPS.rand(3, 2, bond_dim=2, dtype="complex128", seed=81)
    gauges = {index: np.array([0.5, 1.4]) for index in tn.inner_inds()}
    physical = tn.copy()
    physical.gauge_simple_insert(gauges)
    where = ((1, 0), (1, 1))
    first = np.diag(np.exp(0.2j * np.array([1, -1, -1, 1])))
    second = np.diag(np.exp(0.13j * np.array([1, 1, -1, -1])))
    expected = physical.gate(first, where=where, contract=False)
    expected = expected.gate(second, where=where, contract=False)
    core, results = gate_loop_cluster(
        tn, ((first, where), (second, where)), gauges=gauges,
        max_bond=4, als_opts={"max_iterations": 2},
        inplace=False, return_results=True,
    )
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)
    np.testing.assert_allclose(rebuilt.to_dense(), expected.to_dense(), rtol=1e-10, atol=1e-10)
    assert len(results) == 2
