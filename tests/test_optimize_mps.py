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


def _mps_data_norm(mps):
    """Return the MPS norm without its stored global exponent."""
    mps_data = mps.copy()
    mps_data.exponent = 0.0
    return mps_data.norm()


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
    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_final=True)

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


def test_mps_optimizer_non_unitary_defaults_to_20_gate_normalization():
    """The non-unitary convenience flag should normalize every 20 gates by default."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)) for _ in range(21)]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_final=False)

    assert _mps_data_norm(opt.p) == pytest.approx(2.0)
    assert opt.p.norm() == pytest.approx(2.0**21)
    assert opt.p.exponent == pytest.approx(20 * np.log10(2.0))
    assert [event["step"] for event in opt.get_normalizations()] == [20]


def test_mps_optimizer_non_unitary_default_skips_diagnostics():
    """Fast non-unitary runs should only normalize unless diagnostics are requested."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_final=True)

    events = opt.get_normalizations()
    expected_true_norm = np.sqrt(events[-1]["old_norm"])
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(expected_true_norm)
    assert opt.p.exponent == pytest.approx(np.log10(expected_true_norm))
    assert opt.get_fidelities() == [1.0]
    assert opt.get_infidelities() == [0.0]
    assert opt.get_true_infidelities() == [0.0]
    assert opt.get_infidelity_samples() == []
    assert opt.get_norm_infidelity_samples() == []
    assert [event["step"] for event in events] == [3]


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "svd"])
def test_mps_optimizer_normalization_insert_site_stays_inside_span(mode):
    """Normalization events should insert factors inside the canonical span."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (0,)), (qu.hadamard(), (1,))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=1)

    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(1.0)
    assert [event["span"] for event in opt.get_normalizations()] == [(0, 0), (1, 1)]
    assert all(
        event["span"][0] <= event["insert"] <= event["span"][1]
        for event in opt.get_normalizations()
    )


def test_mps_optimizer_normalize_every_controls_frequency_and_final():
    """normalize_every should respect the requested interval and final pass."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (scale, (0,)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, non_unitary=True, normalize_final=True)

    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(8.0)
    assert [event["step"] for event in opt.get_normalizations()] == [2, 3]
    assert opt.p.exponent == pytest.approx(np.log10(8.0))


def test_mps_optimizer_normalize_final_can_be_disabled():
    """normalize_final=False should skip the trailing normalization."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (scale, (0,)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, normalize_final=False, non_unitary=True)

    assert _mps_data_norm(opt.p) == pytest.approx(2.0)
    assert opt.p.norm() == pytest.approx(8.0)
    assert [event["step"] for event in opt.get_normalizations()] == [2]
    assert opt.p.exponent == pytest.approx(np.log10(4.0))


def test_mps_optimizer_automatic_normalization_rejects_exact_mode():
    """Exact mode has no MPS canonicalization range for automatic normalization."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, non_unitary=True)


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
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
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
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
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
        normalize_final=True,
        track_infidelity=True,
    )

    samples = opt.get_infidelity_samples()
    assert len(samples) == 1
    assert 0.0 <= samples[0]["fidelity"] <= 1.0
    assert 0.0 <= samples[0]["local_infidelity"] <= 1.0
    assert 0.0 <= opt.get_true_infidelities()[-1] <= 1.0
    assert opt.get_norm_infidelity_samples() == []
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
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
            "exponent": opt.p.exponent,
            "cumulative_infidelity": opt.get_infidelities()[-1],
        }

        assert _mps_data_norm(opt.p) == pytest.approx(1.0)
        assert len(opt.get_normalizations()) == len(gates)
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
    assert results["dmrg"]["exponent"] == pytest.approx(
        results["mpo"]["exponent"],
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
        normalize_final=True,
        track_norm_infidelity=True,
    )

    samples = opt.get_norm_infidelity_samples()
    assert len(samples) == 1
    assert 0.0 <= samples[0]["local_infidelity"] <= 1.0
    assert 0.0 <= opt.get_infidelities()[-1] <= 1.0
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() > 1.0
