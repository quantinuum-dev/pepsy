"""Coefficient-frame compression parity with the ordinary MPS engine."""

import numpy as np
import pytest

pytest.importorskip("stim")
qtn = pytest.importorskip("quimb.tensor")

from pepsy.optimizers import MpsOptimizer, StabilizerMpsSimulator
from pepsy.optimizers.stabilizer_tn.operators import pauli_combo_submpo

pytestmark = [pytest.mark.core, pytest.mark.optional]


@pytest.mark.parametrize("mode", ["dmrg2", "dmrg3"])
@pytest.mark.parametrize("fast_path", [False, True])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_adjacent_coefficient_fit_matches_mps_budget_and_state(mode, fast_path, dtype):
    """The same exact target uses the same budget and yields the same state."""
    state = qtn.MPS_rand_state(4, 2, dtype=dtype, seed=91)
    mpo, where = pauli_combo_submpo(
        0.8, -0.6j, {1: "X", 2: "X"}, 4, dtype=dtype,
    )
    options = dict(
        n_iter=5, fit_rtol=None, fit_single_pair_fast_path=fast_path,
        cutoff=0.0,
    )
    ordinary = MpsOptimizer(
        state.copy(), [("submpo", mpo, where)], chi=2, mode=mode,
    )
    stabilizer = StabilizerMpsSimulator(
        state.copy(), [("submpo", mpo, where)], chi=2, mode=mode,
        exact_cooling=False, compression_seed=17,
    )
    ordinary.run(compression_seed=17, **options)
    stabilizer.run(**options)
    for optimizer in [ordinary, stabilizer]:
        diagnostic = optimizer.get_fit_diagnostics()
        assert diagnostic["iterations"] == (1 if fast_path else 5)
    assert stabilizer.get_fit_diagnostics()["fit_single_pair_fast_path"] is fast_path
    a = ordinary.to_dense().ravel()
    b = stabilizer.state.p.to_dense().ravel()
    # Normalize away engine-specific diagnostic scale, retaining direction.
    fidelity = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    assert fidelity == pytest.approx(1.0, abs=2e-6 if dtype == "complex64" else 1e-10)
    assert stabilizer.state.p.max_bond() <= 2


@pytest.mark.parametrize("method", ["direct", "zipup", "src"])
def test_coefficient_native_compression_matches_mps_state(method):
    """Both wrappers apply the same coefficient sub-MPO through Quimb."""
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=19)
    mpo, where = pauli_combo_submpo(
        0.8, -0.6j, {0: "X", 3: "Z"}, 4, dtype="complex128",
    )
    ordinary = MpsOptimizer(
        state.copy(), [("submpo", mpo, where)], chi=2, mode=method,
    )
    stabilizer = StabilizerMpsSimulator(
        state.copy(), [("submpo", mpo, where)], chi=2, mode=method,
        compression_seed=17,
    )
    ordinary.run(compression_seed=17, cutoff=0.0)
    stabilizer.run(cutoff=0.0)
    a, b = ordinary.to_dense().ravel(), stabilizer.state.p.to_dense().ravel()
    fidelity = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    assert fidelity == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("mode", ["dmrg", "fit", "dmrg2", "dmrg3"])
@pytest.mark.parametrize("n_iter,pair_cap,expected", [(5, None, 5), (5, 3, 3), (3, 5, 3)])
def test_adjacent_budget_cap_and_default_block_match_mps(mode, n_iter, pair_cap, expected):
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=91)
    mpo, where = pauli_combo_submpo(.8, -.6j, {1: "X", 2: "X"}, 4, dtype="complex128")
    # Use an identical deterministic guess to isolate schedule/target parity
    # from the engines' different randomized SRC seed streams.
    settings = dict(
        n_iter=n_iter, fit_single_pair_n_iter=pair_cap, fit_rtol=None,
        fit_init_strategy="guess-direct",
    )
    ordinary = MpsOptimizer(state.copy(), [("submpo", mpo, where)], chi=2, mode=mode)
    sim = StabilizerMpsSimulator(
        state.copy(), [("submpo", mpo, where)], chi=2, mode=mode,
    )
    ordinary.run(**settings)
    sim.run(**settings)
    assert ordinary.get_fit_diagnostics()["iterations"] == expected
    diag = sim.get_fit_diagnostics()
    assert diag["iterations"] == expected
    assert diag["block_size"] == (1 if mode in {"dmrg", "fit"} else 2)
    a, b = ordinary.to_dense().ravel(), sim.state.p.to_dense().ravel()
    fidelity = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    assert fidelity == pytest.approx(1., abs=1e-10)
    assert sim._fit_single_pair_n_iter is None


@pytest.mark.parametrize("mode", ["dmrg", "fit"])
@pytest.mark.parametrize("block", [2, 3, True])
def test_one_site_mode_rejects_block_override_before_mutation(mode, block):
    sim = StabilizerMpsSimulator(3, [("rxx", .3, 0, 2)], mode=mode, chi=2)
    before = sim.state.p.to_dense().copy()
    with pytest.raises(ValueError, match="fixes fit_block_size to 1"):
        sim.run(fit_block_size=block)
    np.testing.assert_array_equal(sim.state.p.to_dense(), before)
    assert len(sim._queue) == 1
    assert sim._fit_block_size is None


@pytest.mark.parametrize("cap", [0, -1, True, 2.5, "3"])
def test_invalid_pair_cap_rejected_before_mutation(cap):
    sim = StabilizerMpsSimulator(3, [("rxx", .3, 0, 2)], mode="dmrg2", chi=2)
    before = sim.state.p.to_dense().copy()
    with pytest.raises(ValueError, match="fit_single_pair_n_iter"):
        sim.run(fit_single_pair_n_iter=cap, n_iter=7)
    np.testing.assert_array_equal(sim.state.p.to_dense(), before)
    assert len(sim._queue) == 1
    assert sim._fit_n_iter == 8


@pytest.mark.parametrize("mode", ["dmrg", "dmrg2", "dmrg3"])
def test_pair_cap_does_not_leak_into_long_window_or_next_run(mode, monkeypatch):
    from pepsy.fitting.local import FIT

    calls = []
    original = FIT.run_gate

    def observe(fit, *args, **kwargs):
        calls.append(kwargs["n_iter"])
        return original(fit, *args, **kwargs)

    monkeypatch.setattr(FIT, "run_gate", observe)
    sim = StabilizerMpsSimulator(4, chi=2, mode=mode, exact_cooling=False)
    sim.set_gates([("rxx", .3, 0, 1), ("rxx", .4, 0, 3)]).run(
        n_iter=5, fit_single_pair_n_iter=3, fit_rtol=None,
    )
    assert calls == [3, 5]
    calls.clear()
    sim.set_gates([("rxx", .3, 0, 1)]).run(n_iter=4, fit_rtol=None)
    assert calls == [4]


def test_shot_replay_inherits_pair_cap_and_supports_override(monkeypatch):
    from pepsy.fitting.local import FIT

    calls = []
    original = FIT.run_gate

    def observe(fit, *args, **kwargs):
        calls.append(kwargs["n_iter"])
        return original(fit, *args, **kwargs)

    monkeypatch.setattr(FIT, "run_gate", observe)
    sim = StabilizerMpsSimulator(3, [("rxx", .3, 0, 1)], chi=2, mode="dmrg2", exact_cooling=False)
    sim.run(shots=2, strategy="independent", n_iter=5, fit_single_pair_n_iter=3, fit_rtol=None)
    assert calls and set(calls) == {3}
    calls.clear()
    sim.run(shots=2, strategy="independent", n_iter=5, fit_single_pair_n_iter=3,
            fit_rtol=None, run_kwargs={"fit_single_pair_n_iter": 4})
    assert calls and set(calls) == {4}


@pytest.mark.parametrize("where_terms", [{0: "X", 3: "Z"}, {1: "X", 2: "Z"}])
def test_oversample_controls_match_mps_and_preserve_caller_options(where_terms):
    from copy import deepcopy

    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=91)
    mpo, where = pauli_combo_submpo(.8, -.6j, where_terms, 4, dtype="complex128")
    settings = {
        "max_bond_oversample": 3,
        "cutoff_oversample": 0.,
        "cutoff_mode_oversample": "rel",
        "compress_opts_final": {"cutoff": 0., "cutoff_mode": "rsum2"},
    }
    saved = deepcopy(settings)
    ordinary = MpsOptimizer(state.copy(), [("submpo", mpo, where)], chi=2, mode="zipup-oversample")
    sim = StabilizerMpsSimulator(state.copy(), [("submpo", mpo, where)], chi=2, mode="zipup-oversample")
    ordinary.run(compression_opts=settings, cutoff=0.)
    sim.run(compression_opts=settings, cutoff=0.)
    a, b = ordinary.to_dense().ravel(), sim.state.p.to_dense().ravel()
    fidelity = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    assert fidelity == pytest.approx(1., abs=1e-10)
    assert sim.state.p.max_bond() <= 2
    assert settings == saved
    assert sim._compression_opts == {}


@pytest.mark.parametrize("mode,settings,error", [
    ("dmrg2", {"max_bond_oversample": 3}, NotImplementedError),
    ("zipup-oversample", {"max_bond_oversample": 0}, ValueError),
    ("direct", {"unknown": 1}, ValueError),
    ("direct", [], TypeError),
])
def test_invalid_compression_options_leave_queue_and_configuration_unchanged(mode, settings, error):
    sim = StabilizerMpsSimulator(3, [("rxx", .3, 0, 2)], chi=2, mode=mode)
    before = sim.state.p.to_dense().copy()
    with pytest.raises(error):
        sim.run(compression_opts=settings, n_iter=7)
    np.testing.assert_array_equal(sim.state.p.to_dense(), before)
    assert len(sim._queue) == 1
    assert sim._compression_opts == {}
    assert sim._fit_n_iter == 8


def test_shot_replay_forwards_compression_options(monkeypatch):
    seen = []
    original = StabilizerMpsSimulator._apply_quimb_submpo

    def observe(sim, *args, **kwargs):
        seen.append(dict(sim._compression_opts))
        return original(sim, *args, **kwargs)

    monkeypatch.setattr(StabilizerMpsSimulator, "_apply_quimb_submpo", observe)
    mpo, where = pauli_combo_submpo(.8, -.6j, {0: "X", 2: "Z"}, 3, dtype="complex128")
    sim = StabilizerMpsSimulator(
        3, [("submpo", mpo, where)], chi=2, mode="zipup-oversample",
    )
    sim.run(shots=2, strategy="independent",
            compression_opts={"max_bond_oversample": 3})
    assert seen == [{"max_bond_oversample": 3}] * 2
    assert sim._compression_opts == {}


def test_compression_options_restore_after_failed_operator_update(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected compressor failure")

    monkeypatch.setattr(StabilizerMpsSimulator, "_apply_quimb_submpo", fail)
    mpo, where = pauli_combo_submpo(.8, -.6j, {0: "X", 2: "Z"}, 3, dtype="complex128")
    sim = StabilizerMpsSimulator(
        3, [("submpo", mpo, where)], chi=2, mode="zipup-oversample",
    )
    before = sim.state.p.to_dense().copy()
    with pytest.raises(RuntimeError, match="injected compressor failure"):
        sim.run(compression_opts={"max_bond_oversample": 3}, fit_single_pair_n_iter=2)
    np.testing.assert_array_equal(sim.state.p.to_dense(), before)
    assert len(sim._queue) == 1
    assert sim._compression_opts == {}
    assert sim._fit_single_pair_n_iter is None


def test_branch_sum_fallback_rejects_explicit_compression_options():
    sim = StabilizerMpsSimulator(3, chi=2, mode="zipup-oversample")
    sim._compression_opts = {"max_bond_oversample": 3}
    before = sim.state.p.to_dense().copy()
    with pytest.raises(NotImplementedError, match="branch-sum"):
        sim._apply_operator_sum([(1., {0: "X"})], unitary=True)
    np.testing.assert_array_equal(sim.state.p.to_dense(), before)


def test_new_controls_are_not_installed_when_later_configuration_is_invalid():
    sim = StabilizerMpsSimulator(3, [("rxx", .3, 0, 2)], chi=2, mode="zipup-oversample")
    with pytest.raises(ValueError, match="fit_init_seed"):
        sim.run(compression_opts={"max_bond_oversample": 3},
                fit_single_pair_n_iter=2, fit_init_seed=-1)
    assert len(sim._queue) == 1
    assert sim._compression_opts == {}
    assert sim._fit_single_pair_n_iter is None
