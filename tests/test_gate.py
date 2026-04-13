"""Tests for gate-routing helpers in :mod:`pepsy.gate`."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import ps_to_peps
from pepsy.gate import (
    apply_gate_1d,
    apply_gate_2d,
    apply_gates_2d,
    gen_long_range_swap_path,
    pauli,
    x,
    y,
    z,
)


def test_gen_long_range_swap_path_adjacent():
    """Adjacent coordinates should return a single final pair."""
    path = list(gen_long_range_swap_path((0, 0), (0, 1)))
    assert path == [((0, 0), (0, 1))]


def test_gen_long_range_swap_path_xy_order():
    """x_then_y should move along x first and y second."""
    path = list(gen_long_range_swap_path((0, 0), (1, 1), sequence="x_then_y"))
    assert path == [((0, 0), (1, 0)), ((1, 0), (1, 1))]


def test_gen_long_range_swap_path_cyclic_adjacent_wrap():
    """Cyclic routing should treat edge-wrapped neighbors as adjacent."""
    path = list(gen_long_range_swap_path((0, 0), (3, 0), cyclic=True, Lx=4, Ly=4))
    assert path == [((0, 0), (3, 0))]


def test_gen_long_range_swap_path_cyclic_uses_shorter_wrap_route():
    """Cyclic routing should choose shortest wrapped displacement."""
    path = list(
        gen_long_range_swap_path(
            (0, 0),
            (4, 0),
            sequence="x_then_y",
            cyclic=True,
            Lx=6,
            Ly=6,
        )
    )
    assert path == [((0, 0), (5, 0)), ((5, 0), (4, 0))]


def test_ps_to_peps_builds_product_state():
    """ps_to_peps should build a valid bond-dimension-1 PEPS."""
    peps = ps_to_peps(2, 3, dtype="complex128", theta=0.123)
    assert peps.Lx == 2
    assert peps.Ly == 3
    assert int(peps.max_bond()) == 1


def test_apply_2dtn_one_site_inplace():
    """One-site gate application should modify in place and return same object."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    out = apply_gate_2d(peps, gate, ((0, 0),), cutoff=1e-12)
    assert out is peps


def test_apply_2dtn_two_site_nearest_inplace():
    """Two-site nearest-neighbor gate should run and return same object."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    out = apply_gate_2d(peps, gate, ((0, 0), (0, 1)), cutoff=1e-12)
    assert out is peps


def test_apply_2dtn_invalid_where_raises():
    """where must contain one or two coordinates."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="where must contain one or two"):
        apply_gate_2d(peps, gate, ((0, 0), (0, 1), (1, 1)))


def test_apply_2dtn_rejects_duplicate_two_site_coordinates():
    """Two-site gate call should require distinct coordinates."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    with pytest.raises(ValueError, match="distinct coordinates"):
        apply_gate_2d(peps, gate, ((0, 0), (0, 0)))


def test_apply_gates_accepts_mixed_site_and_edge_specs():
    """apply_gates_2d should accept one-site (i,j) and two-site edge tuples."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    rx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rzz = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    sites = [(0, 0), (1, 1)]
    edges = [((0, 0), (0, 1))]
    gates = []
    for edge in edges:
        gates.append((edge, rzz))
    for site in sites:
        gates.append((site, rx))

    out = apply_gates_2d(
        peps,
        gates,
        bra=False,
        contract="reduce-split",
        tags=[],
        cutoff=1.0e-12,
        sequence="x_then_y",
    )
    assert out is peps


def test_apply_gates_invalid_where_raises():
    """Invalid where entries should fail with clear ValueError."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="Invalid where specification"):
        apply_gates_2d(peps, [(((0, 0), (1, 1), (1, 0)), gate)])


def test_apply_gates_runs_final_compress_when_chi_given():
    """chi should trigger final compress_all_ after gate application."""
    class _DummyPeps:
        def __init__(self):
            self.called = None

        def compress_all_(self, *, max_bond, cutoff):
            self.called = (max_bond, cutoff)
            return self

    dummy = _DummyPeps()
    out = apply_gates_2d(dummy, [], chi=2, chi_cutoff=1.0e-12)
    assert out is dummy
    assert dummy.called == (2, 1.0e-12)


def test_apply_gates_invalid_chi_raises():
    """Non-positive chi should fail with ValueError."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    with pytest.raises(ValueError, match="positive integer"):
        apply_gates_2d(peps, [], chi=0)


def test_apply_2dtn_infers_swap_backend_from_gate(monkeypatch):
    """SWAP tensors should follow the backend inferred from the gate tensor."""
    torch = pytest.importorskip("torch")
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(gate)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = object()
    gate = torch.eye(4, dtype=torch.complex128).reshape(2, 2, 2, 2)
    apply_gate_2d(peps, gate, ((0, 0), (0, 2)), sequence="x_then_y")

    assert len(calls) == 3
    assert isinstance(calls[0], torch.Tensor)
    assert isinstance(calls[2], torch.Tensor)
    assert calls[0].dtype == gate.dtype
    assert calls[2].dtype == gate.dtype


def test_apply_2dtn_prefers_network_backend_over_gate_backend(monkeypatch):
    """SWAP tensors should match PEPS backend when gate backend differs."""
    torch = pytest.importorskip("torch")
    calls = []

    class _DummyTensor:
        def __init__(self, data):
            self.data = data

    class _DummyPeps:
        def __init__(self, data):
            self.tensor_map = {0: _DummyTensor(data)}

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(gate)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = _DummyPeps(torch.ones((2, 2), dtype=torch.complex128))
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    apply_gate_2d(peps, gate, ((0, 0), (0, 2)), sequence="x_then_y")

    assert len(calls) == 3
    assert isinstance(calls[0], torch.Tensor)
    assert isinstance(calls[2], torch.Tensor)


def test_apply_2d_gate_cyclic_wrap_avoids_swap_chain(monkeypatch):
    """Cyclic nearest-wrap neighbors should be applied directly with no SWAP chain."""
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(tuple(inds))

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = object()
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    apply_gate_2d(peps, gate, ((0, 0), (3, 0)), cyclic=True, Lx=4, Ly=4)

    assert len(calls) == 1
    assert calls[0] == ("k0,0", "k3,0")


@pytest.mark.parametrize("contract", ["split", "reduce-split"])
def test_apply_gate_1d_split_routes_long_range_via_swaps(monkeypatch, contract):
    """split/reduce-split should route long-range 1D gates with SWAP chains."""
    calls = []
    original_gate_inds = qtn.tensor_network_gate_inds

    def _recording_gate_inds(peps, gate, inds, **kwargs):
        calls.append(tuple(inds))
        return original_gate_inds(peps, gate, inds, **kwargs)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _recording_gate_inds)

    mps = qtn.MPS_computational_state("00000", dtype=np.complex128)
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate_1d(mps, (0, 4), gate, contract=contract, inplace=True)
    assert out is mps
    assert calls == [
        ("k0", "k1"),
        ("k1", "k2"),
        ("k2", "k3"),
        ("k3", "k4"),
        ("k2", "k3"),
        ("k1", "k2"),
        ("k0", "k1"),
    ]


def test_apply_gate_1d_split_gate_keeps_direct_two_site_application(monkeypatch):
    """split-gate should keep the direct two-site application path."""
    calls = []
    original_gate_inds = qtn.tensor_network_gate_inds

    def _recording_gate_inds(peps, gate, inds, **kwargs):
        calls.append(tuple(inds))
        return original_gate_inds(peps, gate, inds, **kwargs)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _recording_gate_inds)

    mps = qtn.MPS_computational_state("00000", dtype=np.complex128)
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate_1d(mps, (0, 4), gate, contract="split-gate", inplace=True)
    assert out is mps
    assert calls == [("k0", "k4")]


def test_apply_gate_1d_prefers_network_backend_over_gate_backend(monkeypatch):
    """1D SWAP tensors should match MPS backend when gate backend differs."""
    torch = pytest.importorskip("torch")
    calls = []
    original_gate_inds = qtn.tensor_network_gate_inds

    def _recording_gate_inds(peps, gate, inds, **kwargs):
        calls.append(gate)
        return original_gate_inds(peps, gate, inds, **kwargs)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _recording_gate_inds)

    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate_1d(mps, (0, 2), gate, contract="split", inplace=True)
    assert out is mps
    assert isinstance(calls[0], torch.Tensor)


def test_apply_gate_1d_infers_cupy_backend_from_network(monkeypatch):
    """1D SWAP tensors should become CuPy when the network tensors are CuPy."""
    cp = pytest.importorskip("cupy")
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(gate)
        return peps

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    try:
        mps.apply_to_arrays(cp.asarray)
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA-capable device available for CuPy backend test.")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate_1d(mps, (0, 2), gate, contract="split", inplace=True)
    assert out is mps
    assert isinstance(calls[0], cp.ndarray)


def test_pauli_matches_axis_helpers():
    """pauli('X'/'Y'/'Z') should match x()/y()/z() helpers."""
    assert np.allclose(pauli("X"), x())
    assert np.allclose(pauli("Y"), y())
    assert np.allclose(pauli("Z"), z())
    assert np.allclose(pauli("z"), z())
