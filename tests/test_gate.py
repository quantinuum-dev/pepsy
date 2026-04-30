"""Tests for gate-routing helpers in :mod:`pepsy.gate`."""

import warnings

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import ps_to_peps
from pepsy.gate import (
    build_mpo_from_gates,
    build_pepo_from_gates,
    gate as apply_gate,
    gen_long_range_swap_path_2d,
    gen_long_range_swap_path_3d,
    pauli,
    x,
    y,
    z,
)


def test_gen_long_range_swap_path_2d_adjacent():
    """Adjacent coordinates should return a single final pair."""
    path = list(gen_long_range_swap_path_2d((0, 0), (0, 1)))
    assert path == [((0, 0), (0, 1))]


def test_gen_long_range_swap_path_2d_xy_order():
    """x_then_y should move along x first and y second."""
    path = list(gen_long_range_swap_path_2d((0, 0), (1, 1), sequence="x_then_y"))
    assert path == [((0, 0), (1, 0)), ((1, 0), (1, 1))]


def test_gen_long_range_swap_path_2d_cyclic_adjacent_wrap():
    """Cyclic routing should treat edge-wrapped neighbors as adjacent."""
    path = list(gen_long_range_swap_path_2d((0, 0), (3, 0), cyclic=True, Lx=4, Ly=4))
    assert path == [((0, 0), (3, 0))]


def test_gen_long_range_swap_path_2d_cyclic_uses_shorter_wrap_route():
    """Cyclic routing should choose shortest wrapped displacement."""
    path = list(
        gen_long_range_swap_path_2d(
            (0, 0),
            (4, 0),
            sequence="x_then_y",
            cyclic=True,
            Lx=6,
            Ly=6,
        )
    )
    assert path == [((0, 0), (5, 0)), ((5, 0), (4, 0))]


def test_gen_long_range_swap_path_3d_xyz_order():
    """3D routing should support deterministic axis-order sequences."""
    path = list(
        gen_long_range_swap_path_3d(
            (0, 0, 0),
            (1, 1, 1),
            sequence="xyz",
        )
    )
    assert path == [
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (1, 1, 1)),
    ]


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
    out = apply_gate(peps, gate, ((0, 0),), cutoff=1e-12)
    assert out is peps


@pytest.mark.parametrize("contract", [False, True])
def test_apply_2dtn_one_site_uses_requested_contract(monkeypatch, contract):
    """One-site 2D gates should honor the caller-provided contract mode."""
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(kwargs["contract"])
        return peps

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    out = apply_gate(peps, gate, ((0, 0),), contract=contract, cutoff=1e-12)

    assert out is peps
    assert calls == [contract]


@pytest.mark.parametrize("contract", ["split", "reduce-split", "split-gate"])
def test_apply_2dtn_one_site_coerces_non_bool_contract_to_true(monkeypatch, contract):
    """One-site 2D gates should coerce non-bool contract values to True."""
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(kwargs["contract"])
        return peps

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    out = apply_gate(peps, gate, ((0, 0),), contract=contract, cutoff=1e-12)

    assert out is peps
    assert calls == [True]


def test_apply_2dtn_two_site_nearest_inplace():
    """Two-site nearest-neighbor gate should run and return same object."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    out = apply_gate(peps, gate, ((0, 0), (0, 1)), cutoff=1e-12)
    assert out is peps


def test_apply_2dtn_invalid_where_raises():
    """where must contain one or two coordinates."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="Invalid where specification"):
        apply_gate(peps, gate, ((0, 0), (0, 1), (1, 1)))


def test_apply_2dtn_rejects_duplicate_two_site_coordinates():
    """Two-site gate call should require distinct coordinates."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    with pytest.raises(ValueError, match="distinct coordinates"):
        apply_gate(peps, gate, ((0, 0), (0, 0)))


def test_apply_3dtn_split_gate_keeps_direct_two_site_application(monkeypatch):
    """split-gate should keep the direct two-site application path in 3D."""
    calls = []

    def _fake_gate_inds(tn, gate, inds, **kwargs):
        calls.append(tuple(inds))
        return tn

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        Lx = 1
        Ly = 1
        Lz = 3

        def outer_inds(self):
            return ()

    tn = _Dummy3DTN()
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate(tn, gate, ((0, 0, 0), (0, 0, 2)), contract="split-gate")
    assert out is tn
    assert calls == [("k0,0,0", "k0,0,2")]


def test_gates_tn_3d_invalid_where_raises():
    """Invalid 3D where entries should fail with clear ValueError."""
    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        def compress_all_(self, *, max_bond, cutoff):
            _ = (max_bond, cutoff)
            return self

    tn = _Dummy3DTN()
    gate = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="Invalid where specification"):
        apply_gate(tn, [(gate, ((0, 0, 0), (0, 0)))])


def test_gates_tn_1d_applies_mixed_site_specs():
    """gate should accept canonical ``[(gate, where), ...]`` streams."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    zz_gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    gates = [(x_gate, (1,)), (zz_gate, (1, 2))]

    out = apply_gate(mps, gates, contract="split-gate", inplace=True)
    assert out is mps


def test_gate_dispatches_to_1d_for_mps_two_site_where(monkeypatch):
    """Dispatcher should route ``where=(i,j)`` on MPS to 1D helper."""
    calls = []

    def _fake_gate_tn_1d(tn, G, where, **kwargs):
        calls.append(("1d", where, kwargs.get("contract")))
        return tn

    def _fake_gate_tn_2d(tn, G, where, **kwargs):
        calls.append(("2d", where, kwargs.get("contract")))
        return tn

    monkeypatch.setattr("pepsy.gate._apply_gate_1d", _fake_gate_tn_1d)
    monkeypatch.setattr("pepsy.gate._apply_gate_2d", _fake_gate_tn_2d)

    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    G = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate(mps, G, (1, 2))
    assert out is mps
    assert calls == [("1d", (1, 2), "split-gate")]


def test_gate_1d_dispatch_mutates_mps_in_place():
    """1D dispatcher should preserve in-place semantics for MPS inputs."""
    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    out = apply_gate(mps, G, (1,))

    assert out is mps
    assert np.allclose(mps.to_dense().ravel(), np.array([0, 0, 1, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize("dim", ("1d", "2d", "3d"))
def test_gate_sequence_dispatches_to_bulk_helpers(monkeypatch, dim):
    """Bundled streams should route each entry through the matching helper."""
    calls = []

    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        Lx = 2
        Ly = 2
        Lz = 2

    if dim == "1d":
        tn = qtn.MPS_computational_state("0000", dtype=np.complex128)
        where = [(1,), (2,)]
        helper_name = "_apply_gate_1d"
        contract = "split-gate"
    elif dim == "2d":
        tn = ps_to_peps(2, 2, dtype="complex128")
        where = [((0, 0),), ((0, 1),)]
        helper_name = "_apply_gate_2d"
        contract = "reduce-split"
    else:
        tn = _Dummy3DTN()
        where = [((0, 0, 0),), ((0, 0, 1),)]
        helper_name = "_apply_gate_3d"
        contract = "reduce-split"

    gates = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]

    def _fake_bulk_helper(tn_i, G_arg, where=None, **kwargs):
        calls.append((tn_i, G_arg, where, kwargs.get("contract"), "inplace" in kwargs))
        return tn_i

    monkeypatch.setattr(f"pepsy.gate.{helper_name}", _fake_bulk_helper)

    gate_pairs = list(zip(gates, where))
    out = apply_gate(tn, gate_pairs, contract=contract)
    assert out is tn
    assert len(calls) == len(gates)
    for idx, (tn_i, G_arg, where_arg, contract_i, has_inplace_kw) in enumerate(calls):
        assert tn_i is tn
        assert G_arg is gates[idx]
        assert where_arg == where[idx]
        assert contract_i is True
        assert has_inplace_kw is (dim == "1d")


@pytest.mark.parametrize("dim", ("1d", "2d", "3d"))
def test_gate_sequence_dispatch_inplace_false_copies(monkeypatch, dim):
    """Bundled streams should copy TN first when ``inplace=False``."""
    calls = []

    class _Dummy2DTN:  # pylint: disable=too-few-public-methods
        Lx = 2
        Ly = 2

        def copy(self):
            return _Dummy2DTN()

    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        Lx = 2
        Ly = 2
        Lz = 2

        def copy(self):
            return _Dummy3DTN()

    if dim == "1d":
        tn = qtn.MPS_computational_state("0000", dtype=np.complex128)
        where = [(1,), (2,)]
        helper_name = "_apply_gate_1d"
    elif dim == "2d":
        tn = _Dummy2DTN()
        where = [((0, 0),), ((0, 1),)]
        helper_name = "_apply_gate_2d"
    else:
        tn = _Dummy3DTN()
        where = [((0, 0, 0),), ((0, 0, 1),)]
        helper_name = "_apply_gate_3d"

    gates = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]

    def _fake_bulk_helper(tn_i, G_arg, where=None, **kwargs):
        calls.append((tn_i, G_arg, where, "inplace" in kwargs))
        return tn_i

    monkeypatch.setattr(f"pepsy.gate.{helper_name}", _fake_bulk_helper)

    gate_pairs = list(zip(gates, where))
    out = apply_gate(tn, gate_pairs, inplace=False)
    assert out is not tn
    assert len(calls) == len(gates)
    for idx, (tn_i, G_arg, where_arg, has_inplace_kw) in enumerate(calls):
        assert tn_i is out
        assert G_arg is gates[idx]
        assert where_arg == where[idx]
        assert has_inplace_kw is (dim == "1d")


def test_gate_sequence_requires_matching_where():
    """Plain gate lists should require canonical bundled ``(gate, where)`` entries."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    gates = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]

    with pytest.raises(ValueError, match="bundled stream"):
        apply_gate(mps, gates, where=[(0,)])

    with pytest.raises(ValueError, match="bundled stream"):
        apply_gate(mps, gates, where=None)


def test_gate_rejects_legacy_gates_and_wheres_alias_pair():
    """``(gates, wheres)`` alias should raise instead of being auto-zipped."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_gate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    G = [x_gate, z_gate]
    where = [(1,), (2,)]

    with pytest.raises(ValueError, match="alias"):
        apply_gate(mps, (G, where))

def test_gate_accepts_sequence_of_bundled_pairs():
    """gate should accept a bundled gate stream ``[(gate, where), ...]``."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_gate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    out = apply_gate(mps, [(x_gate, (1,)), (z_gate, (2,))])
    assert out is mps


def test_gate_rejects_bundled_single_gate_and_where_pair_alias():
    """Single bundled pair alias should be rejected for clarity."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Single bundled pair alias"):
        apply_gate(mps, (x_gate, (1,)))

    with pytest.raises(ValueError, match="Single bundled pair alias"):
        apply_gate(mps, [x_gate, (2,)])


def test_gate_accepts_tuple_outer_container_for_bundled_stream():
    """Bundled streams can use list or tuple outer containers."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_gate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    out = apply_gate(mps, ((x_gate, (1,)), (z_gate, (2,))))
    assert out is mps


def test_gate_rejects_mixed_bundled_stream_shapes():
    """Bundled streams should reject mixed entry shapes instead of guessing."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_gate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="exact shape"):
        apply_gate(mps, [(x_gate, (1,)), z_gate])


def test_gates_tn_1d_requires_where_for_gate_sequences():
    """gate should require explicit ``where`` for gate sequences."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_gate = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="bundled stream"):
        apply_gate(mps, [x_gate, z_gate], contract="split-gate")


def test_gate_non_k_prefix_requires_explicit_ind_id_for_1d():
    """Numeric 1D where should require explicit ind_id for non-k prefixes."""
    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    mps.reindex_({f"k{i}": f"b{i}" for i in range(3)})
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="pass ind_id explicitly"):
        apply_gate(mps, G, 1)

    out = apply_gate(mps, G, 1, ind_id="b{}", inplace=False)
    assert out is not mps


def test_gate_sequence_non_k_prefix_uses_custom_ind_id_for_all_entries():
    """Bundled 1D streams should honor custom ``ind_id`` on every gate."""
    mps = qtn.MPS_computational_state("0000", dtype=np.complex128)
    mps.reindex_({f"k{i}": f"b{i}" for i in range(4)})
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    zz_gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    gates = [(x_gate, (1,)), (zz_gate, (1, 2))]

    with pytest.raises(ValueError, match="pass ind_id explicitly"):
        apply_gate(mps, gates, contract="split-gate")

    out = apply_gate(mps, gates, ind_id="b{}", contract="split-gate")
    assert out is mps


def test_gate_dispatches_to_2d_for_peps_ambiguous_two_int_where(monkeypatch):
    """Dispatcher should treat ``where=(i,j)`` as 2D coordinate on PEPS."""
    calls = []

    def _fake_gate_tn_1d(tn, G, where, **kwargs):
        calls.append(("1d", where, kwargs.get("contract")))
        return tn

    def _fake_gate_tn_2d(tn, G, where, **kwargs):
        calls.append(("2d", where, kwargs.get("contract")))
        return tn

    monkeypatch.setattr("pepsy.gate._apply_gate_1d", _fake_gate_tn_1d)
    monkeypatch.setattr("pepsy.gate._apply_gate_2d", _fake_gate_tn_2d)

    peps = ps_to_peps(2, 2, dtype="complex128")
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    out = apply_gate(peps, G, (1, 0))
    assert out is peps
    assert calls == [("2d", ((1, 0),), True)]


def test_gate_non_k_prefix_requires_explicit_ind_id_for_2d():
    """2D coordinate where should require explicit ind_id for non-k prefixes."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    peps.reindex_({f"k{i},{j}": f"b{i},{j}" for i in range(2) for j in range(2)})
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="pass ind_id explicitly"):
        apply_gate(peps, G, (1, 0))

    out = apply_gate(peps, G, (1, 0), ind_id="b{},{}", inplace=False)
    assert out is not peps


def test_gate_ind_id_placeholder_mismatch_raises_clear_error():
    """Mismatched ``ind_id`` placeholders should fail with a clear error."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="incompatible with site coordinate"):
        apply_gate(peps, G, (1, 0), ind_id="b{}")


def test_gate_rejects_single_integer_where_on_peps():
    """A PEPS coordinate must not be misinterpreted as a 1D site index."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Could not infer gate dimensionality"):
        apply_gate(peps, G, (1,))


def test_gate_3d_dispatch_supports_inplace_false(monkeypatch):
    """3D dispatch should support ``inplace=False`` and pass kwargs cleanly."""
    calls = []

    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        Lx = 2
        Ly = 2
        Lz = 2

        def copy(self):
            return _Dummy3DTN()

    def _fake_gate_tn_3d(tn, G, where, **kwargs):
        calls.append(("3d", where, kwargs.get("contract"), "inplace" in kwargs))
        return tn

    monkeypatch.setattr("pepsy.gate._apply_gate_3d", _fake_gate_tn_3d)

    tn = _Dummy3DTN()
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    out = apply_gate(tn, G, (1, 0, 1), inplace=False)

    assert out is not tn
    assert calls == [("3d", ((1, 0, 1),), True, False)]


def test_gate_requires_where_for_single_gate():
    """Single-gate dispatch should error when ``where`` is omitted."""
    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    G = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="where must be provided"):
        apply_gate(mps, G, where=None)


def test_gate_accepts_string_index_selectors(monkeypatch):
    """String where selectors should route through tensor_network_gate_inds."""
    calls = []

    def _fake_gate_inds(tn, gate, inds, **kwargs):
        calls.append((tuple(inds), kwargs.copy()))
        return tn

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    class _DummyTN:  # pylint: disable=too-few-public-methods
        pass

    tn = _DummyTN()
    gate_op = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    out = apply_gate(
        tn,
        gate_op,
        where=("k1", "k2"),
        contract=False,
        inplace=False,
        ind_id="k{}",
    )

    assert out is tn
    assert calls == [(("k1", "k2"), {"contract": False, "inplace": True, "cutoff_mode": "rel"})]


def test_gate_1d_general_tn_without_l_uses_gate_inds(monkeypatch):
    """1D numeric where on generic TNs should route through gate_inds directly."""
    calls = []

    def _fake_gate_inds(tn, gate, inds, **kwargs):
        calls.append((tuple(inds), kwargs.copy()))
        return tn

    def _fail_gate_tn_1d(*args, **kwargs):
        raise AssertionError("gate should not be used for TNs without L.")

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)
    monkeypatch.setattr("pepsy.gate._apply_gate_1d", _fail_gate_tn_1d)

    class _DummyTN:  # pylint: disable=too-few-public-methods
        def outer_inds(self):
            return ("b1",)

    tn = _DummyTN()
    gate_op = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    out = apply_gate(
        tn,
        gate_op,
        where=1,
        ind_id="b{}",
        contract=False,
        inplace=False,
        site_tags="I{}",
        dtype="complex128",
    )

    assert out is tn
    assert calls == [(("b1",), {"contract": False, "inplace": True, "cutoff_mode": "rel"})]


def test_apply_gates_accepts_mixed_site_and_edge_specs():
    """gate should accept matching ``G`` and ``where`` sequences."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    rx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rzz = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    sites = [(0, 0), (1, 1)]
    edges = [((0, 0), (0, 1))]
    G = []
    where = []
    for edge in edges:
        G.append(rzz)
        where.append(edge)
    for site in sites:
        G.append(rx)
        where.append(site)

    gates = list(zip(G, where))
    out = apply_gate(
        peps,
        gates,
        bra=False,
        contract="reduce-split",
        tags=[],
        cutoff=1.0e-12,
        sequence="x_then_y",
    )
    assert out is peps


def test_gates_tn_2d_accepts_bundled_pair_stream():
    """gate should also accept ``[(gate, where), ...]``."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    rx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rzz = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    gates = [(rzz, ((0, 0), (0, 1))), (rx, (1, 1))]

    out = apply_gate(
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
        apply_gate(peps, [(gate, ((0, 0), (1, 1), (1, 0)))])


def test_apply_gates_runs_final_compress_when_chi_given():
    """chi should trigger final compress_all_ after gate application."""
    class _DummyPeps:
        def __init__(self):
            self.called = None

        def compress_all_(self, *, max_bond, cutoff):
            self.called = (max_bond, cutoff)
            return self

    dummy = _DummyPeps()
    out = apply_gate(dummy, [], chi=2, chi_cutoff=1.0e-12)
    assert out is dummy
    assert dummy.called == (2, 1.0e-12)


def test_apply_gates_invalid_chi_raises():
    """Non-positive chi should fail with ValueError."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    with pytest.raises(ValueError, match="positive integer"):
        apply_gate(peps, [], chi=0)


def test_build_mpo_from_gates_accepts_bundled_stream():
    """MPO builder should accept canonical bundled gate streams."""
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    zz_gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    mpo = build_mpo_from_gates(
        ((x_gate, (0,)), (zz_gate, (0, 1))),
        max_bond=8,
        contract="split",
    )
    assert mpo.L == 2
    assert mpo.max_bond() >= 1


def test_build_mpo_from_gates_accepts_single_gate_where():
    """MPO builder should accept single-gate form with explicit where."""
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    mpo = build_mpo_from_gates(
        x_gate,
        where=(1,),
        max_bond=8,
        contract="split",
    )
    assert mpo.L == 2
    assert mpo.max_bond() >= 1


def test_build_pepo_from_gates_accepts_bundled_stream():
    """PEPO builder should accept canonical bundled gate streams."""
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    zz_gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)
    pepo = build_pepo_from_gates(
        ((x_gate, ((0, 0),)), (zz_gate, ((0, 0), (1, 1)))),
        max_bond=8,
        contract="split",
    )
    assert (pepo.Lx, pepo.Ly) == (2, 2)
    assert pepo.max_bond() >= 1


def test_build_pepo_from_gates_accepts_single_gate_where():
    """PEPO builder should accept single-gate form with explicit where."""
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pepo = build_pepo_from_gates(
        x_gate,
        where=((1, 1),),
        max_bond=8,
        contract="split",
    )
    assert (pepo.Lx, pepo.Ly) == (2, 2)
    assert pepo.max_bond() >= 1


def test_build_operator_from_gates_rejects_explicit_index_where():
    """Operator builders should reject explicit index-name selectors."""
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="Explicit index-name where selectors"):
        build_mpo_from_gates(x_gate, where="k0")


def test_apply_2dtn_without_tn_backend_keeps_swap_numpy(monkeypatch):
    """Without TN backend sample, SWAP tensors should remain numpy."""
    torch = pytest.importorskip("torch")
    calls = []

    def _fake_gate_inds(peps, gate, inds, **kwargs):
        calls.append(gate)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    peps = object()
    gate = torch.eye(4, dtype=torch.complex128).reshape(2, 2, 2, 2)
    apply_gate(peps, gate, ((0, 0), (0, 2)), sequence="x_then_y")

    assert len(calls) == 3
    assert isinstance(calls[0], np.ndarray)
    assert isinstance(calls[2], np.ndarray)
    assert isinstance(calls[1], torch.Tensor)


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
    apply_gate(peps, gate, ((0, 0), (0, 2)), sequence="x_then_y")

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
    apply_gate(peps, gate, ((0, 0), (3, 0)), cyclic=True, Lx=4, Ly=4)

    assert len(calls) == 1
    assert calls[0] == ("k0,0", "k3,0")


def test_gate_tn_2d_split_gate_keeps_direct_two_site_application(monkeypatch):
    """split-gate should keep the direct two-site application path in 2D."""
    calls = []
    original_gate_inds = qtn.tensor_network_gate_inds

    def _recording_gate_inds(peps, gate, inds, **kwargs):
        calls.append(tuple(inds))
        return original_gate_inds(peps, gate, inds, **kwargs)

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _recording_gate_inds)

    peps = ps_to_peps(1, 3, dtype="complex128")
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate(peps, gate, ((0, 0), (0, 2)), contract="split-gate")
    assert out is peps
    assert calls == [("k0,0", "k0,2")]


def test_gate_tn_2d_rejects_bundled_single_pair_alias():
    """gate should reject ``(gate, where)`` shorthand alias."""
    peps = ps_to_peps(2, 2, dtype="complex128")
    gate_op = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Single bundled pair alias"):
        apply_gate(peps, (gate_op, ((0, 0),)), cutoff=1e-12)


@pytest.mark.parametrize("contract", [False, True])
def test_apply_gate_1d_one_site_uses_requested_contract(monkeypatch, contract):
    """One-site 1D gates should honor the caller-provided contract mode."""
    calls = []

    def _fake_gate_inds(tn, gate, inds, **kwargs):
        calls.append(kwargs["contract"])
        return tn

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    out = apply_gate(mps, gate, (1,), contract=contract, inplace=True)

    assert out is mps
    assert calls == [contract]


@pytest.mark.parametrize("contract", ["split", "reduce-split", "split-gate"])
def test_apply_gate_1d_one_site_coerces_non_bool_contract_to_true(monkeypatch, contract):
    """One-site 1D gates should coerce non-bool contract values to True."""
    calls = []

    def _fake_gate_inds(tn, gate, inds, **kwargs):
        calls.append(kwargs["contract"])
        return tn

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    out = apply_gate(mps, gate, (1,), contract=contract, inplace=True)

    assert out is mps
    assert calls == [True]


def test_apply_gate_1d_complex_gate_does_not_warn_real_cast():
    """Complex one-site gates should not be cast-to-real when TN sample is real."""
    mps = qtn.MPS_computational_state("000", dtype=np.float64)
    gate = np.array([[1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = apply_gate(mps, gate, (1,), inplace=False)

    msgs = [str(w.message) for w in caught]
    assert not any("Casting complex values to real discards the imaginary part" in m for m in msgs)
    assert np.iscomplexobj(out.to_dense())


def test_gate_tn_1d_rejects_bundled_single_pair_alias():
    """gate should reject ``(gate, where)`` shorthand alias."""
    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    gate_op = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Single bundled pair alias"):
        apply_gate(mps, (gate_op, (1,)), contract=False, inplace=True)


def test_gate_tn_1d_rejects_list_alias_for_bundled_single_pair():
    """Single bundled pair alias should be rejected for list form too."""
    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    gate_op = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Single bundled pair alias"):
        apply_gate(mps, [gate_op, (1,)], contract=False, inplace=True)


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

    out = apply_gate(mps, gate, (0, 4), contract=contract, inplace=True)
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

    out = apply_gate(mps, gate, (0, 4), contract="split-gate", inplace=True)
    assert out is mps
    assert calls == [("k0", "k4")]


def test_apply_gate_1d_mixed_backend_user_gate_raises():
    """User-supplied gate backend should match TN backend for 1D application."""
    torch = pytest.importorskip("torch")

    mps = qtn.MPS_computational_state("000", dtype=np.complex128)
    mps.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    with pytest.raises(TypeError):
        apply_gate(mps, gate, (0, 2), contract="split", inplace=True)


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

    out = apply_gate(mps, gate, (0, 2), contract="split", inplace=True)
    assert out is mps
    assert isinstance(calls[0], cp.ndarray)


def test_gate_tn_3d_rejects_bundled_single_pair_alias(monkeypatch):
    """gate should reject ``(gate, where)`` shorthand alias."""
    calls = []

    def _fake_gate_inds(tn, gate, inds, **kwargs):
        calls.append(tuple(inds))
        return tn

    monkeypatch.setattr(qtn, "tensor_network_gate_inds", _fake_gate_inds)

    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        Lx = 1
        Ly = 1
        Lz = 3

        def outer_inds(self):
            return ()

    tn = _Dummy3DTN()
    gate_op = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    with pytest.raises(ValueError, match="Single bundled pair alias"):
        apply_gate(tn, (gate_op, ((0, 0, 0), (0, 0, 2))), contract="split-gate")
    assert calls == []


def test_gates_tn_3d_accepts_bundled_pair_stream(monkeypatch):
    """gate should also accept ``[(gate, where), ...]``."""
    calls = []

    def _fake_gate_tn_3d(tn, gate, where, **kwargs):
        calls.append((gate, where, kwargs.get("contract")))
        return tn

    monkeypatch.setattr("pepsy.gate._apply_gate_3d", _fake_gate_tn_3d)

    class _Dummy3DTN:  # pylint: disable=too-few-public-methods
        Lx = 1
        Ly = 1
        Lz = 2

        def compress_all_(self, *, max_bond, cutoff):
            _ = (max_bond, cutoff)
            return self

    tn = _Dummy3DTN()
    gate_a = np.eye(2, dtype=np.complex128)
    gate_b = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    out = apply_gate(
        tn,
        [(gate_a, (0, 0, 0)), (gate_b, ((0, 0, 0), (0, 0, 1)))],
        contract="split-gate",
    )

    assert out is tn
    assert calls == [
        (gate_a, ((0, 0, 0),), True),
        (gate_b, ((0, 0, 0), (0, 0, 1)), "split-gate"),
    ]


def test_pauli_matches_axis_helpers():
    """pauli('X'/'Y'/'Z') should match x()/y()/z() helpers."""
    assert np.allclose(pauli("X"), x())
    assert np.allclose(pauli("Y"), y())
    assert np.allclose(pauli("Z"), z())
    assert np.allclose(pauli("z"), z())
