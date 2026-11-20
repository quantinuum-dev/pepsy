"""Interval sweeps versus independent physical-index and bit-state oracles."""

import numpy as np
import pytest
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from examples import dmt_2d_scaling as old
from examples import dmt_2d_window as bench
from pepsy.operators import PauliMPO
from pepsy.operators._pauli_dmt import _PauliDMTWorkspace


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_window_actual_loss_preserves_post_gate_regions_and_input(direction, dtype):
    model = bench.problem(2, 3, .17)
    rng = np.random.default_rng(517)
    ranks = [1, 4, 16, 25, 16, 4, 1]
    cores = [rng.normal(size=(a, b, 4)).astype(dtype) for a, b in zip(ranks, ranks[1:])]
    if dtype == np.complex128:
        cores = [a + 1j * rng.normal(size=a.shape) for a in cores]
    ws = _PauliDMTWorkspace(_cores=cores, max_bond=32, regions=model["regions"],
                            check_regions=True, region_selection="min_cover")
    before = PauliMPO.from_pauli_cores(ws.cores).to_dense()
    env = (list(ws.environments.left), list(ws.environments.right), ws.center)
    out = ws.copy()
    out.apply_window(1, bench.zz_transfer(4, .17, dtype=dtype), absorb=direction)
    actual = PauliMPO.from_pauli_cores(out.cores).to_dense()
    target = old.small.apply_density(before, expm(-.17j * np.kron(bench.ref.Z, bench.ref.Z)), (1, 4), 6)
    assert np.linalg.norm(actual - target) > 1e-4
    assert max(a.shape[1] for a in out.cores) <= 32
    for region in model["regions"]:
        np.testing.assert_allclose(bench.ref.partial_trace(actual, sorted(region), 6),
                                   bench.ref.partial_trace(target, sorted(region), 6), atol=2e-9, rtol=2e-12)
    np.testing.assert_array_equal(PauliMPO.from_pauli_cores(ws.cores).to_dense(), before)
    assert ws.center == env[2] and ws.environments.left == env[0] and ws.environments.right == env[1]
    assert not out.report().preservation_errors
    for j, a in enumerate(out.cores):
        if j < out.center:
            matrix = a.transpose(0, 2, 1).reshape(-1, a.shape[1])
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(a.shape[1]), atol=2e-12)
        elif j > out.center:
            matrix = a.transpose(0, 2, 1).reshape(a.shape[0], -1)
            np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(a.shape[0]), atol=2e-12)


@pytest.mark.parametrize("update", ["full", "window"])
@pytest.mark.parametrize("order", ["fixed", "nearest"])
def test_real_evolution_matches_exact_untruncated_circuit(update, order):
    model = bench.problem(2, 3, .1)
    reference = old.problem(2, 3, .1)
    row, ws = bench.run(model, 64, 2, update=update, order=order, sample_every=1)
    psi = old.vector_step(old.vector_step(reference["psi"], reference), reference)
    expected = old.local_reference(psi, reference)["plaquettes"]
    _, panels = bench.readout(ws, model)
    np.testing.assert_allclose(panels, expected, atol=3e-14)
    assert row["status"] == "complete" and not row["preservation_errors"]
    assert all(a.dtype == np.float64 for a in ws.cores)
    assert row["samples"][-1]["local_hermiticity"] < 1e-14


@pytest.mark.parametrize("direction", ["left", "right"])
def test_window_torch_matches_numpy_and_preserves_complex128(direction):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = bench.problem(2, 3, .1)
    numpy_ws = bench.workspace([a.astype(complex) for a in model["cores"]], model, 64)
    torch_ws = bench.workspace([torch.tensor(a, dtype=torch.complex128) for a in model["cores"]], model, 64)
    ops = bench.zz_transfer(6, .1, dtype=complex)
    numpy_ws.apply_window(0, ops, absorb=direction)
    torch_ws.apply_window(0, [torch.tensor(a) for a in ops], absorb=direction)
    expected = PauliMPO.from_pauli_cores(numpy_ws.cores).to_dense()
    actual = PauliMPO.from_pauli_cores(torch_ws.cores).to_dense().numpy()
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    assert all(a.dtype == torch.complex128 for a in torch_ws.cores)


@pytest.mark.parametrize("kind", ["empty", "dtype", "shape", "nan", "boundary"])
def test_invalid_transfer_does_not_mutate_workspace(kind):
    model = bench.problem(2, 3, .1)
    ws = bench.workspace(model["cores"], model, 64)
    before, center = list(ws.cores), ws.center
    ops = bench.zz_transfer(3, .1)
    if kind == "empty":
        ops = []
    elif kind == "dtype":
        ops[0] = ops[0].astype(np.complex128)
    elif kind == "shape":
        ops[0] = ops[0][..., :3]
    elif kind == "nan":
        ops[0][0, 0, 0, 0] = np.nan
    else:
        ops[-1] = np.repeat(ops[-1], 2, axis=1)
    with pytest.raises(ValueError):
        ws.apply_window(1, ops)
    assert ws.center == center
    assert all(a is b for a, b in zip(before, ws.cores))


def test_6x6_builder_and_commuting_neighborhood_oracle():
    model = bench.problem(6, 6, .1, hx=0., hz=0.)
    assert model["n"] == 36 and len(model["edges"]) == 60 and len(model["regions"]) == 25
    assert sum(a.nbytes for a in model["cores"]) < 5000
    small = bench.problem(2, 3, .1, hx=0., hz=0.)
    psi = bench.ref.product_vector(small["vectors"])
    gate = expm(-.23j * np.kron(bench.ref.Z, bench.ref.Z))
    for edge in small["edges"]:
        psi = bench.ref.apply_vector(psi, gate, edge, 6)
    for region in small["regions"]:
        region = tuple(sorted(region))
        exact = .6 * bench.ref.vector_marginal(psi, region, 6) + .4 * np.eye(16) / 16
        np.testing.assert_allclose(bench.commuting_oracle(small, region, .23), exact, atol=1e-15)


def test_static_positive_mixture_oracle_checks_actual_truncation():
    result = bench.static_mixture_audit(2, 3, chi=32, components=64)
    assert result["actual_truncation"]
    assert result["max_preservation_error"] < 1e-12
    assert result["trace_error"] < 1e-12
    assert not result["preservation_errors"]


def test_gate_audit_uses_exact_post_gate_target():
    model = bench.problem(2, 3, .1)
    ws = bench.workspace(model["cores"], model, 64)
    assert bench.audit_gate(ws, (0, 5), .1, model) < 1e-13


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_window_fresh_trace_guard_detects_an_unreported_change(monkeypatch, policy):
    model = bench.problem(2, 3, .1)
    ws = bench.workspace(model["cores"], model, 64)
    ws.preservation_check = policy
    split = ws.split_pair

    def corrupt(bond, **kwargs):
        report = split(bond, **kwargs)
        ws.cores[ws.center] = 1.1 * ws.cores[ws.center]
        ws.environments.invalidate(ws.center, ws.center + 1)
        return report

    monkeypatch.setattr(ws, "split_pair", corrupt)
    if policy == "warn":
        with pytest.warns(RuntimeWarning, match="continuing"):
            ws.apply_window(0, bench.zz_transfer(6, .1))
        assert any("fresh relative defect" in e for e in ws.report().preservation_errors)
    else:
        with pytest.raises(FloatingPointError, match="fresh relative defect"):
            ws.apply_window(0, bench.zz_transfer(6, .1))


def test_torch_complex128_benchmark_matches_numpy():
    pytest.importorskip("torch")
    model = bench.problem(2, 3, .1)
    numpy_row, numpy_ws = bench.run(model, 64, 2, dtype=np.complex128)
    torch_row, torch_ws = bench.run(model, 64, 2, dtype=np.complex128, backend="torch")
    np.testing.assert_allclose(bench.readout(numpy_ws, model)[1], bench.readout(torch_ws, model)[1], atol=3e-14)
    assert not numpy_row["preservation_errors"] and not torch_row["preservation_errors"]


def test_center_metadata_survives_an_interrupted_gauge_move(monkeypatch):
    import pepsy.operators._pauli_dmt as kernel

    model = bench.problem(2, 3, .1)
    ws = bench.workspace(model["cores"], model, 64)
    before = PauliMPO.from_pauli_cores(ws.cores).to_dense()
    qr, calls = kernel._qr, 0

    def fail_second(matrix):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected factorization failure")
        return qr(matrix)

    monkeypatch.setattr(kernel, "_qr", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        ws.move_center(5)
    assert ws.center == 1
    np.testing.assert_allclose(PauliMPO.from_pauli_cores(ws.cores).to_dense(), before, atol=1e-15)
