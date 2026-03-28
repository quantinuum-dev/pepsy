"""Tests for gate-routing helpers in :mod:`pepsy.gate`."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import product_state_peps
from pepsy.gate import apply_2dtn_, apply_gates, gen_long_range_swap_path


def test_gen_long_range_swap_path_adjacent():
    """Adjacent coordinates should return a single final pair."""
    path = list(gen_long_range_swap_path((0, 0), (0, 1)))
    assert path == [((0, 0), (0, 1))]


def test_gen_long_range_swap_path_xy_order():
    """x_then_y should move along x first and y second."""
    path = list(gen_long_range_swap_path((0, 0), (1, 1), sequence="x_then_y"))
    assert path == [((0, 0), (1, 0)), ((1, 0), (1, 1))]


def test_product_state_peps_builds_product_state():
    """product_state_peps should build a valid bond-dimension-1 PEPS."""
    peps = product_state_peps(2, 3, dtype="complex128", theta=0.123)
    assert peps.Lx == 2
    assert peps.Ly == 3
    assert int(peps.max_bond()) == 1


def test_apply_2dtn_one_site_inplace():
    """One-site gate application should modify in place and return same object."""
    peps = product_state_peps(2, 2, dtype="complex128")
    gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    out = apply_2dtn_(peps, gate, ((0, 0),), cutoff=1e-12)
    assert out is peps


def test_apply_2dtn_two_site_nearest_inplace():
    """Two-site nearest-neighbor gate should run and return same object."""
    peps = product_state_peps(2, 2, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    out = apply_2dtn_(peps, gate, ((0, 0), (0, 1)), cutoff=1e-12)
    assert out is peps


def test_apply_2dtn_invalid_where_raises():
    """where must contain one or two coordinates."""
    peps = product_state_peps(2, 2, dtype="complex128")
    gate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="where must contain one or two"):
        apply_2dtn_(peps, gate, ((0, 0), (0, 1), (1, 1)))


def test_apply_2dtn_rejects_duplicate_two_site_coordinates():
    """Two-site gate call should require distinct coordinates."""
    peps = product_state_peps(2, 2, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    with pytest.raises(ValueError, match="distinct coordinates"):
        apply_2dtn_(peps, gate, ((0, 0), (0, 0)))


def test_apply_gates_accepts_mixed_site_and_edge_specs():
    """apply_gates should accept one-site (i,j) and two-site edge tuples."""
    peps = product_state_peps(2, 2, dtype="complex128")
    rx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rzz = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    sites = [(0, 0), (1, 1)]
    edges = [((0, 0), (0, 1))]
    gates = []
    for edge in edges:
        gates.append((edge, rzz))
    for site in sites:
        gates.append((site, rx))

    out = apply_gates(
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
    peps = product_state_peps(2, 2, dtype="complex128")
    gate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="Invalid where specification"):
        apply_gates(peps, [(((0, 0), (1, 1), (1, 0)), gate)])


def test_apply_gates_runs_final_compress_when_chi_given():
    """chi should trigger final compress_all_ after gate application."""
    class _DummyPeps:
        def __init__(self):
            self.called = None

        def compress_all_(self, *, max_bond, cutoff):
            self.called = (max_bond, cutoff)
            return self

    dummy = _DummyPeps()
    out = apply_gates(dummy, [], chi=2, chi_cutoff=1.0e-12)
    assert out is dummy
    assert dummy.called == (2, 1.0e-12)


def test_apply_gates_invalid_chi_raises():
    """Non-positive chi should fail with ValueError."""
    peps = product_state_peps(2, 2, dtype="complex128")
    with pytest.raises(ValueError, match="positive integer"):
        apply_gates(peps, [], chi=0)


def test_apply_2dtn_infers_swap_backend_from_gate(monkeypatch):
    """SWAP tensors should follow gate backend when to_backend is not provided."""
    torch = pytest.importorskip("torch")
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(gate)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = object()
    gate = torch.eye(4, dtype=torch.complex128).reshape(2, 2, 2, 2)
    apply_2dtn_(peps, gate, ((0, 0), (0, 2)), sequence="x_then_y")

    assert len(calls) == 3
    assert isinstance(calls[0], torch.Tensor)
    assert isinstance(calls[2], torch.Tensor)
    assert calls[0].dtype == gate.dtype
    assert calls[2].dtype == gate.dtype
