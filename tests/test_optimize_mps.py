"""Tests for :mod:`pepsy.optimizers.mps.optimizer`."""

from numbers import Real
import sys
import types

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


def _non_unitary_entangling_gate():
    """Return a small two-site filter that creates entanglement from |++>."""
    return np.diag([1.0, 0.5, 0.5, 2.0]).astype(complex)


def _two_branch_flip_submpo(*, L, sites, targets, w0=0.7, w1=0.3):
    """Return ``w0 * I + w1 * prod(X_targets)`` as a sparse-site MPO."""
    eye = np.eye(2, dtype=np.complex128)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sites = tuple(sites)
    targets = set(targets)
    branch0 = [eye.copy() for _site in sites]
    branch1 = [flip.copy() if site in targets else eye.copy() for site in sites]
    branch0[0] *= w0
    branch1[0] *= w1
    mpo0 = qtn.MPO_product_operator(
        branch0,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    mpo1 = qtn.MPO_product_operator(
        branch1,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    return mpo0.add_MPO(mpo1)


def _mps_data_norm(mps):
    """Return the MPS norm without its stored global exponent."""
    mps_data = mps.copy()
    mps_data.exponent = 0.0
    return mps_data.norm()


def _tensor_data_norm(mps, site):
    """Return the Frobenius norm of one MPS tensor's data."""
    return float(np.linalg.norm(np.asarray(mps[site].data)))


def _assert_event_sites_locally_normalized(mps, event):
    """Check that every tensor rescaled by an event has local norm one."""
    for site in event["sites"]:
        assert _tensor_data_norm(mps, site) == pytest.approx(1.0)


def test_mps_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpsOptimizer`` mode validation."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


def test_mps_optimizer_svd_smoke():
    """SVD mode should apply mixed 1q/2q gates without errors."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.p.max_bond() <= 8


def test_mps_optimizer_accepts_bundled_gate_stream():
    """Construction should accept ``[(gate, where), ...]`` with ``where=None``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_forwards_custom_ind_id_to_gate_application():
    """Optimizer gate application should honor non-default physical indices."""
    p0 = qtn.MPS_computational_state("000", dtype=np.complex128)
    p0.reindex_({f"k{i}": f"b{i}" for i in range(3)})
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(x_gate, (1,))],
        chi=4,
        mode="svd",
        ind_id="b{}",
    )
    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p
    assert set(out.outer_inds()) == {"b0", "b1", "b2"}


def test_mps_optimizer_set_and_add_gates_accept_bundled_gate_stream():
    """set_gates/add_gates should accept bundled gate-stream entries."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), chi=8, mode="svd")

    opt.set_gates([(qu.hadamard(), (1,))])
    opt.add_gates([(qu.CNOT(), (0, 3))])

    assert len(opt.G) == 2
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_mpo_mode_applies_submpo_stream_event():
    """MPO mode should apply explicit sparse sub-MPO stream events."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    )

    out = opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(1, 3)]
    assert np.allclose(vec, expected)
    assert out.max_bond() <= 8


def test_mps_optimizer_mpo_mode_accepts_submpo_mapping_event():
    """Mapping events should provide a readable public sub-MPO stream API."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[{"kind": "submpo", "mpo": mpo, "where": [0, 2]}],
        chi=8,
        mode="mpo",
    )

    out = opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[10] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(0, 2)]
    assert np.allclose(vec, expected)


def test_mps_optimizer_submpo_diagnostics_do_not_consume_event_mpo():
    """Diagnostic target construction should not mutate reusable event MPOs."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    )
    out = opt.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        non_unitary=True,
        track_norm_infidelity=True,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert len(opt.get_norm_infidelity_samples()) == 1
    assert np.allclose(vec, expected)

    reuse = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0, fidelity_samples=0)
    reuse_vec = reuse.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    assert np.allclose(reuse_vec, expected)


def test_mps_optimizer_submpo_stream_events_require_mpo_mode():
    """Non-MPO modes should reject sub-MPO stream events clearly."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 2))],
        chi=8,
        mode="svd",
    )

    with pytest.raises(ValueError, match="subMPO stream events"):
        opt.run(progbar=False)


def test_mps_optimizer_submpo_stream_validates_support_sites():
    """Sub-MPO support should be a unique in-range set of 1D sites."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))

    repeated = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 0))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="repeated site"):
        repeated.run(progbar=False)

    out_of_range = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 3))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="outside the MPS range"):
        out_of_range.run(progbar=False)


def test_mps_optimizer_default_inplace_false_keeps_input_unchanged():
    """Default construction should work on a copy and keep input state intact."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    p0_ref = p0.copy()
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0, gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.p is not p0
    assert np.allclose(p0.to_dense(), p0_ref.to_dense())
    assert not np.allclose(out.to_dense(), p0_ref.to_dense())


def test_mps_optimizer_inplace_true_updates_input_state():
    """inplace=True should optimize the original input state object."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    p0_ref = p0.copy()
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0, gates=gates, chi=8, mode="svd", inplace=True)
    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.p is p0
    assert out is p0
    assert not np.allclose(p0.to_dense(), p0_ref.to_dense())


def test_mps_optimizer_rejects_noncanonical_bundled_gate_aliases():
    """Bundled gate input should require the canonical list/tuple shapes."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")

    opt = py.MpsOptimizer(p0.copy(), gates=((qu.hadamard(), (1,)),), chi=8, mode="svd")
    assert opt.where == [(1,)]

    with pytest.raises(ValueError, match="exact shape"):
        py.MpsOptimizer(p0.copy(), gates=[(qu.hadamard(), (1,)), qu.CNOT()], chi=8, mode="svd")


def test_mps_optimizer_run_returns_state_for_empty_queue():
    """run() should return the managed MPS even when there are no gates."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")

    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "svd", "exact"])
def test_mps_optimizer_run_returns_state_after_updates(mode):
    """run() should return the updated MPS for every execution mode."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=2, fidelity_samples=1)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["mpo", "svd", "swap"])
def test_mps_optimizer_compression_modes_sample_norm_proxy(mode):
    """Compression modes should append norm-proxy samples to fidelity trace."""
    p0 = qtn.MPS_computational_state("00000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT(), qu.phase_gate(0.2), qu.CNOT()]
    where = [(1,), (0, 4), (2,), (1, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    before = len(opt.get_fidelities())
    opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)
    trace = opt.get_fidelities()

    assert len(trace) >= before + 1
    assert isinstance(trace[-1], Real)


def test_mps_optimizer_unitary_default_still_samples_norm_proxy():
    """Default/unitary runs should keep the historical norm-proxy sampling."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert len(opt.get_fidelities()) > 1


def test_mps_optimizer_svd_forwards_cutoff_mode_to_final_compression(monkeypatch):
    """SVD mode should honor explicit cutoff_mode in its chi compression pass."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="svd")
    calls = []
    original_left_compress = opt.p.left_compress

    def _recording_left_compress(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_left_compress(*args, **kwargs)

    monkeypatch.setattr(opt.p, "left_compress", _recording_left_compress)

    opt.run(progbar=False, cutoff=1.0e-9, cutoff_mode="rsum2")

    assert calls
    assert calls[-1]["cutoff"] == pytest.approx(1.0e-9)
    assert calls[-1]["cutoff_mode"] == "rsum2"


def test_mps_optimizer_rejects_negative_fidelity_samples():
    """Negative fidelity_samples should fail clearly."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.CNOT()]
    where = [(0, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="mpo")
    with pytest.raises(ValueError, match="fidelity_samples must be >= 0"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)


def test_mps_optimizer_non_unitary_flag_normalizes_one_site_gate():
    """non_unitary=True should normalize at run end for short streams."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)

    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (1,))], chi=8, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
    )

    events = opt.get_normalizations()
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)
    assert opt.p.exponent == pytest.approx(np.log10(2.0))
    assert opt.get_infidelities() == [0.0]
    assert opt.get_norm_infidelity_samples() == []
    assert len(events) == 1
    assert events[0]["step"] == 1
    assert events[0]["old_norm"] == pytest.approx(4.0)
    assert events[0]["span"] == (1, 1)
    assert events[0]["insert"] == 1
    assert events[0]["exponent"] == pytest.approx(np.log10(2.0))
    assert opt.info_c["cur_orthog"] == (1, 1)


def test_mps_optimizer_manual_normalize_accumulates_exponent():
    """Manual normalization should preserve represented norm via ``p.exponent``."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    p0[0].modify(data=2.0 * p0[0].data)

    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    old_norm = opt.normalize(insert=0)

    assert old_norm == pytest.approx(4.0)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)
    assert opt.p.exponent == pytest.approx(np.log10(2.0))


def test_mps_optimizer_non_unitary_default_does_not_normalize():
    """The non-unitary flag should not enable scale control by default."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
        (qu.hadamard(), (2,)),
        (_non_unitary_entangling_gate(), (2, 3)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt_none = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_final=False)
    opt_none.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=None,
        normalize_final=False,
    )

    assert opt.get_normalizations() == []
    assert opt_none.get_normalizations() == []
    assert opt.p.exponent == pytest.approx(0.0)
    assert opt_none.p.exponent == pytest.approx(0.0)


def test_mps_optimizer_non_unitary_scale_control_skips_diagnostics():
    """Fast non-unitary scale control should not collect diagnostics by default."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
    )
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    assert opt.get_fidelities() == [1.0]
    assert opt.get_infidelities() == [0.0]
    assert opt.get_true_infidelities() == [0.0]
    assert opt.get_infidelity_samples() == []
    assert opt.get_norm_infidelity_samples() == []
    assert [event["step"] for event in events] == [3]
    assert events[0]["reason"] == "compression"
    _assert_event_sites_locally_normalized(opt.p, events[0])


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "svd"])
def test_mps_optimizer_normalization_insert_site_stays_inside_span(mode):
    """Normalization events should insert factors inside the canonical span."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 1)), (qu.CNOT(), (2, 3))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=1)

    assert opt.p.norm() == pytest.approx(1.0)
    assert [event["span"] for event in opt.get_normalizations()] == [(0, 1), (2, 3)]
    assert all(
        event["span"][0] <= event["insert"] <= event["span"][1]
        for event in opt.get_normalizations()
    )


def test_mps_optimizer_normalize_every_enables_compression_and_final():
    """Enabled normalize_every should scale compressed updates and final tails."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, non_unitary=True, normalize_final=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert [event["step"] for event in events] == [2, 3]
    assert [event["reason"] for event in events] == ["compression", "final"]
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    _assert_event_sites_locally_normalized(opt.p, events[-1])


def test_mps_optimizer_normalize_final_can_be_disabled():
    """normalize_final=False should skip the trailing normalization."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, normalize_final=False, non_unitary=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert [event["step"] for event in opt.get_normalizations()] == [2]
    assert opt.get_normalizations()[0]["reason"] == "compression"


def test_mps_optimizer_automatic_normalization_rejects_exact_mode():
    """Exact mode has no MPS canonicalization range for automatic normalization."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, non_unitary=True, normalize_every=True)


def test_mps_optimizer_track_infidelity_rejects_exact_mode():
    """Exact mode has no compressed target for infidelity diagnostics."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[(qu.CNOT(), (0, 1))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, track_infidelity=True)


def test_mps_optimizer_rejects_invalid_normalize_every():
    """normalize_every should fail clearly for non-positive intervals."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.eye(2, dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="svd")

    with pytest.raises(ValueError, match="normalize_every must be >= 1"):
        opt.run(progbar=False, normalize_every=0, non_unitary=True)


def test_mps_optimizer_normalization_options_require_non_unitary():
    """Normalization controls should not act as aliases for ``non_unitary=True``."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.eye(2, dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="svd")

    with pytest.raises(ValueError, match="normalize_every requires non_unitary=True"):
        opt.run(progbar=False, normalize_every=1)

    with pytest.raises(ValueError, match="normalize_final requires non_unitary=True"):
        opt.run(progbar=False, normalize_final=True)


@pytest.mark.parametrize("where", [(1, 2), (0, 3)])
def test_mps_optimizer_canonical_span_norm_matches_full_target_norm(where):
    """Canonical span norm should match full norm for split-gate targets."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128")
    gate = _non_unitary_entangling_gate()
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")

    xmin, xmax = sorted(where)
    opt.canonize_mps(opt.p, (xmin, xmax))
    target = py.gate(
        opt.p,
        gate,
        where,
        contract="split-gate",
        cutoff=1e-12,
        cutoff_mode="rel",
        inplace=False,
    )

    local_norm = opt._canonical_span_norm(target, (xmin, xmax))  # pylint: disable=protected-access
    assert local_norm == pytest.approx(target.norm())


def test_mps_optimizer_canonical_span_norm_ignores_stored_exponent():
    """Internal normalization should measure raw data, not represented scale."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    opt.p.exponent = 3.0

    raw = opt.p.copy()
    raw.exponent = 0.0
    measured = opt._canonical_span_norm(opt.p, (0, 3))  # pylint: disable=protected-access

    assert measured == pytest.approx(raw.norm())
    assert opt.p.exponent == pytest.approx(3.0)


def test_mps_optimizer_norm_infidelity_uses_tn_norm_strip_exponent(monkeypatch):
    """Norm-infidelity diagnostics should measure raw norms through ``tn_norm``."""
    calls = []
    original_tn_norm = mps_optimizer_module.tn_norm

    def _spy_tn_norm(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_tn_norm(*args, **kwargs)

    monkeypatch.setattr(mps_optimizer_module, "tn_norm", _spy_tn_norm)

    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="mpo")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    assert len(samples) == 1
    assert len(calls) >= 2
    assert all(call["strip_exponent"] is True for call in calls)
    assert all(call["contraction_opt"] == opt.contraction_opt for call in calls)


def test_mps_optimizer_non_unitary_norm_infidelity_matches_svd_target():
    """SVD non-unitary proxy should match quimb's target infidelity."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates[:2]:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)
    target = py.gate(
        target,
        gates[-1][0],
        gates[-1][1],
        contract="reduce-split",
        cutoff=1e-12,
        cutoff_mode="rel",
        inplace=False,
    )

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    proxy = opt.get_infidelities()[-1]
    actual = target.distance_normalized(opt.p, normalized="infidelity", optimize="auto-hq")

    assert len(samples) == 1
    assert samples[0]["step"] == 3
    assert samples[0]["where"] == (0, 1)
    assert samples[0]["local_infidelity"] == pytest.approx(proxy)
    assert proxy == pytest.approx(float(np.real(actual)))
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 0.0


def test_mps_optimizer_true_infidelity_matches_tn_fidelity():
    """track_infidelity=True should record the true normalized-overlap metric."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates[:2]:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)
    target = py.gate(
        target,
        gates[-1][0],
        gates[-1][1],
        contract="reduce-split",
        cutoff=1e-12,
        cutoff_mode="rel",
        inplace=False,
    )

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_infidelity=True,
    )

    samples = opt.get_infidelity_samples()
    actual_fidelity = float(np.real(py.tn_fidelity(opt.p, target, contraction_opt="auto-hq")))
    actual_fidelity = min(1.0, max(0.0, actual_fidelity))

    assert len(samples) == 1
    assert samples[0]["step"] == 3
    assert samples[0]["where"] == (0, 1)
    assert samples[0]["fidelity"] == pytest.approx(actual_fidelity)
    assert samples[0]["local_infidelity"] == pytest.approx(1.0 - actual_fidelity)
    assert opt.get_true_infidelities()[-1] == pytest.approx(1.0 - actual_fidelity)
    assert opt.get_infidelities()[-1] == pytest.approx(1.0 - actual_fidelity)
    assert opt.get_norm_infidelity_samples() == []
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 0.0


def test_mps_optimizer_true_infidelity_losses_are_geometric_mean():
    """True-fidelity tracking should store geometric-mean progress in losses."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    filter_gate = _non_unitary_entangling_gate()
    gates = [
        (h_gate, (0,)),
        (h_gate, (1,)),
        (h_gate, (2,)),
        (h_gate, (3,)),
        (filter_gate, (0, 3)),
        (filter_gate, (1, 2)),
        (filter_gate, (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=1,
        fidelity_samples=5,
        track_infidelity=True,
    )

    log_sum = 0.0
    expected = []
    for count, sample in enumerate(opt.get_infidelity_samples(), start=1):
        fidelity = sample["fidelity"]
        if fidelity <= 0.0 or np.isneginf(log_sum):
            log_sum = -np.inf
            geometric_fidelity = 0.0
        else:
            log_sum += np.log(fidelity)
            geometric_fidelity = float(np.exp(log_sum / count))
        expected.append(geometric_fidelity)
        assert sample["geometric_fidelity"] == pytest.approx(geometric_fidelity)

    assert len(opt.get_infidelity_samples()) == 3
    assert opt.get_fidelities()[0] == pytest.approx(1.0)
    assert opt.get_fidelities()[1:] == pytest.approx(expected)
    assert len(opt.get_fidelities()) == len(opt.get_infidelity_samples()) + 1


def test_mps_optimizer_true_infidelity_progress_reports_infidelity(monkeypatch):
    """With track_infidelity=True, tqdm postfix should show infidelity."""
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.n = 0
            self.postfix_calls = []
            progress_instances.append(self)

        def set_postfix(self, postfix):
            self.postfix_calls.append(dict(postfix))

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm))

    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (_non_unitary_entangling_gate(), (0, 3)),
    ]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")

    opt.run(
        progbar=True,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=1,
        track_infidelity=True,
    )

    progress = progress_instances[-1]
    assert progress.n == len(gates)
    last = progress.postfix_calls[-1]
    assert "Icum" in last
    assert "Fgeom" not in last
    assert last["Icum"] == opt._format_progress_infidelity(opt.get_infidelities()[-1])


def test_mps_optimizer_progress_infidelity_uses_compact_scientific_format():
    """Tiny displayed infidelities should not round to 0.000000."""
    assert py.MpsOptimizer._format_progress_infidelity(1e-9) == "1.e-9"
    assert py.MpsOptimizer._format_progress_infidelity(0.0) == "0.e+0"


@pytest.mark.parametrize("mode", ["dmrg", "mpo"])
def test_mps_optimizer_true_infidelity_smoke_other_modes(mode):
    """Other compressed modes should support the true infidelity diagnostic."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
    opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=2,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_infidelity=True,
    )

    samples = opt.get_infidelity_samples()
    assert len(samples) == 1
    assert 0.0 <= samples[0]["fidelity"] <= 1.0
    assert 0.0 <= samples[0]["local_infidelity"] <= 1.0
    assert 0.0 <= opt.get_true_infidelities()[-1] <= 1.0
    assert opt.get_norm_infidelity_samples() == []
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 1.0


def test_mps_optimizer_dmrg_non_unitary_matches_mpo_accuracy():
    """DMRG should match MPO accuracy for normalized non-unitary MPS updates."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    filter_gate = _non_unitary_entangling_gate()
    gates = [
        (h_gate, (0,)),
        (h_gate, (1,)),
        (h_gate, (2,)),
        (h_gate, (3,)),
        (filter_gate, (0, 3)),
        (filter_gate, (1, 2)),
        (filter_gate, (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)

    results = {}
    for mode in ("dmrg", "mpo"):
        opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
        opt.run(
            progbar=False,
            cutoff=1e-12,
            n_iter=20,
            non_unitary=True,
            normalize_every=1,
            track_infidelity=True,
        )

        fidelity = float(np.real(py.tn_fidelity(opt.p, target, contraction_opt="auto-hq")))
        results[mode] = {
            "fidelity": fidelity,
            "represented_norm": float(np.real(opt.p.norm())),
            "cumulative_infidelity": opt.get_infidelities()[-1],
        }

        events = opt.get_normalizations()
        assert len(events) == 3
        assert all(event["reason"] == "compression" for event in events)
        _assert_event_sites_locally_normalized(opt.p, events[-1])
        assert len(opt.get_infidelity_samples()) == 3
        assert fidelity > 0.92

    assert results["dmrg"]["fidelity"] == pytest.approx(
        results["mpo"]["fidelity"],
        abs=1e-12,
    )
    assert results["dmrg"]["represented_norm"] == pytest.approx(
        results["mpo"]["represented_norm"],
        abs=1e-12,
    )
    assert results["dmrg"]["cumulative_infidelity"] == pytest.approx(
        results["mpo"]["cumulative_infidelity"],
        abs=1e-12,
    )


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "svd"])
def test_mps_optimizer_non_unitary_norm_infidelity_smoke_other_modes(mode):
    """All compressed modes should expose a bounded non-unitary proxy."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=2,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    assert len(samples) == 1
    assert 0.0 <= samples[0]["local_infidelity"] <= 1.0
    assert 0.0 <= opt.get_infidelities()[-1] <= 1.0
    _assert_event_sites_locally_normalized(opt.p, opt.get_normalizations()[-1])
    assert opt.p.norm() > 1.0
