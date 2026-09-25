"""MPS fit kernels regression tests."""


import inspect
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
import pepsy as py
import pepsy.fitting.local as fitting_local_module
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


pytestmark = [pytest.mark.core, pytest.mark.mps]


def _three_site_ghz_target():
    """Return a four-site GHZ-like target embedded in a five-site MPS."""
    state = qtn.MPS_computational_state("00000", dtype="complex128")
    target = state.copy()
    target.gate_(qu.hadamard(), 0, contract=True)
    for where in ((0, 1), (1, 2), (2, 3)):
        target.gate_(
            qu.CNOT(),
            where,
            contract="split",
            max_bond=2,
            cutoff=0.0,
        )
    return state, target


def test_mps_optimizer_finite_check_combines_backend_scalars_once(monkeypatch):
    """A whole-MPS health check should perform one backend-to-host conversion."""
    torch = pytest.importorskip("torch")
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=20
    )
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    original_to_numpy = mps_optimizer_module.ar.to_numpy
    conversions = []

    def counted_to_numpy(value):
        conversions.append(value)
        return original_to_numpy(value)

    monkeypatch.setattr(mps_optimizer_module.ar, "to_numpy", counted_to_numpy)

    assert py.MpsOptimizer._mps_data_is_finite(state)
    assert len(conversions) == 1


def test_fit_gate_cheap_finite_check_transfers_one_vector_per_sweep(monkeypatch):
    """Cheap FIT health checks should transfer one tiny vector per sweep."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=201
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])
    original_to_numpy = mps_optimizer_module.ar.to_numpy
    conversions = []

    def counted_to_numpy(value):
        conversions.append(value)
        return original_to_numpy(value)

    monkeypatch.setattr(mps_optimizer_module.ar, "to_numpy", counted_to_numpy)

    fit.run_gate(rtol=None, n_iter=3, finite_check=True)

    assert fit.iterations_run == 3
    assert len(conversions) == 3
    # One native finite flag per active tensor plus the terminal center norm's
    # finite flag are transferred together; local norms are reduced only once
    # per completed sweep.
    assert all(np.asarray(original_to_numpy(value)).size == 4 for value in conversions)
    assert len(fit.local_norm_trace) == 3


def test_fit_gate_timing_records_sweep_and_site_steps():
    """FIT timing is opt-in and reports the active interval."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=202
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(block_size=1, n_iter=2, timing=True)

    records = fit.get_timing()
    assert [record["sweep"] for record in records] == [1, 2]
    assert all(record["status"] == "complete" for record in records)
    assert all(record["range_int"] == (0, 2) for record in records)
    assert all(record["site_count"] == 3 for record in records)
    assert all(record["timing_schema"] == 3 for record in records)
    assert all(record["active_site_count"] == 3 for record in records)
    assert all(record["update_count"] == 3 for record in records)
    assert all(record["svd_seconds"] == 0.0 for record in records)
    assert all(
        {
            "effective_seconds",
            "svd_seconds",
            "writeback_seconds",
            "environment_seconds",
            "canonicalization_seconds",
            "moving_environment_seconds",
        }.issubset(site_timing)
        for record in records
        for site_timing in record["site_timings"]
    )
    assert all(
        {
            "canonicalization_seconds",
            "sweep_preparation_canonicalization_seconds",
            "fixed_environment_seconds",
            "moving_environment_seconds",
            "moving_canonicalization_seconds",
            "sweep_overhead_seconds",
        }.issubset(record)
        for record in records
    )


def test_fit_gate_disabled_timing_never_reads_a_clock():
    """The normal FIT path bypasses timing marks and record allocation."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=212,
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    def fail_timing_mark(*_values):
        raise AssertionError("timing=False must not read the profiling clock")

    fit._timing_mark = fail_timing_mark
    fit.run_gate(n_iter=2, timing=False)

    assert fit.get_timing() == []


def test_fit_gate_timing_separates_fixed_and_moving_environments():
    """Two-site FIT timing exposes environment and decomposition phases."""
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=203
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 3])

    fit.run_gate(n_iter=2, block_size=2, timing=True)

    records = fit.get_timing()
    assert all(record["block_size"] == 2 for record in records)
    assert all(record["fixed_environment_seconds"] >= 0.0 for record in records)
    assert all(record["canonicalization_seconds"] >= 0.0 for record in records)
    assert all(record["moving_environment_seconds"] >= 0.0 for record in records)
    assert all(record["sweep_overhead_seconds"] >= 0.0 for record in records)
    for record in records:
        assert record["moving_environment_seconds"] == pytest.approx(
            sum(
                site["moving_environment_seconds"]
                for site in record["site_timings"]
            )
        )
        assert record["canonicalization_seconds"] >= (
            record["sweep_preparation_canonicalization_seconds"]
        )


def test_fit_gate_two_site_grows_only_active_bonds():
    """Two-site FIT should discover rank without globally padding the MPS."""
    initial = qtn.MPS_computational_state("0000", dtype="complex128")
    initial.gate_(qu.hadamard(), 0, contract=True)
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 2),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    fit = py.FIT(target, p=initial, range_int=[0, 2], cutoffs=0.0)

    fit.run_gate(
        n_iter=4,
        block_size=2,
        sweep_sequence="RL",
        max_bond=4,
        cutoff=0.0,
    )

    assert fit.p.bond_size(0, 1) > 1
    assert fit.p.bond_size(1, 2) > 1
    assert fit.p.bond_size(2, 3) == 1
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(
        1.0,
        abs=1.0e-10,
    )
    assert [record["direction"] for record in fit.get_timing()] == []


def test_fit_gate_randomized_guess_handles_cutoff_from_product_state():
    """A seeded disposable guess opens remote-gate sectors before the cutoff."""
    initial = qtn.MPS_computational_state("0000", dtype="complex128")
    initial.gate_(qu.hadamard(), 0, contract=True)
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 3),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    optimizer = py.MpsOptimizer(initial, gates=[], chi=2, mode="dmrg2")
    guess, initialization = optimizer._build_randomized_fit_guess(
        initial,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-4,
    )
    fit = py.FIT(
        target,
        p=guess,
        range_int=[0, 3],
        cutoffs=1.0e-12,
        inplace=True,
    )

    fit.run_gate(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=1.0e-12,
    )

    assert fit.p is guess
    assert initialization["enabled"] is True
    assert [initial.bond_size(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert [fit.p.bond_size(i, i + 1) for i in range(3)] == [2, 2, 2]
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)


@pytest.mark.parametrize("block_size", (2, 3))
def test_fit_run_eff_native_blocks_grow_full_chain(block_size):
    """Full-chain block FIT should grow only bonds supported by the target."""
    initial = qtn.MPS_computational_state("00000", dtype="complex128")
    target = initial.copy()
    target.gate_(qu.hadamard(), 2, contract=True)
    target.gate_(
        qu.CNOT(),
        (2, 3),
        contract="split",
        max_bond=2,
        cutoff=0.0,
    )
    fit = py.FIT(
        target,
        p=initial,
        cutoffs=1.0e-12,
        contraction_opt="greedy",
    )

    fit.run_eff(
        n_iter=2,
        verbose=True,
        block_size=block_size,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=1.0e-12,
    )

    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)
    assert [fit.p.bond_size(site, site + 1) for site in range(4)] == [1, 1, 2, 1]
    assert len(fit.fidelity_trace) == 2
    split_key = "two_site_splits" if block_size == 2 else "three_site_splits"
    assert fit.info[split_key]


@pytest.mark.parametrize("block_size", (2, 3))
def test_fit_run_eff_adaptive_block_warmup_then_one_site_refinement(block_size):
    """Full-chain run_eff can switch from block growth to one-site updates."""
    initial = qtn.MPS_computational_state("00000", dtype="complex128")
    target = initial.copy()
    target.gate_(qu.hadamard(), 2, contract=True)
    target.gate_(
        qu.CNOT(),
        (2, 3),
        contract="split",
        max_bond=2,
        cutoff=0.0,
    )
    fit = py.FIT(target, p=initial, cutoffs=0.0)

    fit.run_eff(
        n_iter=4,
        block_size=block_size,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
    )

    assert fit.iterations_run == 4
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run == 2
    assert fit._sweep_environment_reuse_count == 3
    split_key = "two_site_splits" if block_size == 2 else "three_site_splits"
    assert fit.info[split_key]
    assert fit.p.bond_size(2, 3) == 2


@pytest.mark.parametrize("block_size", (2, 3))
@pytest.mark.parametrize("sweep_sequence", ("RL", "LR"))
def test_fit_run_eff_transition_cache_matches_rebuild(
    block_size,
    sweep_sequence,
):
    """Block-to-one-site cache extensions preserve the rebuilt result."""
    initial = qtn.MPS_rand_state(
        5,
        bond_dim=1,
        phys_dim=2,
        dtype="complex128",
        seed=610,
    )
    target = qtn.MPS_rand_state(
        5,
        bond_dim=3,
        phys_dim=2,
        dtype="complex128",
        seed=611,
    )
    options = {
        "n_iter": 3,
        "block_size": block_size,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 3,
        "cutoff": 1.0e-12,
    }
    cached = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt._allow_sweep_environment_reuse = False

    cached.run_eff(**options)
    rebuilt.run_eff(**options)

    assert cached._sweep_environment_reuse_count == 2
    assert rebuilt._sweep_environment_reuse_count == 0
    assert np.allclose(
        cached.p.to_dense(),
        rebuilt.p.to_dense(),
        atol=1.0e-12,
    )


def test_fit_run_eff_adaptive_rtol_waits_for_one_site_phase():
    """Adaptive run_eff resets tolerance at the block-to-one-site boundary."""
    initial = qtn.MPS_computational_state("00000", dtype="complex128")
    target = initial.copy()
    target.gate_(qu.hadamard(), 2, contract=True)
    target.gate_(
        qu.CNOT(),
        (2, 3),
        contract="split",
        max_bond=2,
        cutoff=0.0,
    )
    fit = py.FIT(target, p=initial, cutoffs=0.0)

    fit.run_eff(
        n_iter=5,
        block_size=2,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        rtol=1.0,
        patience=2,
    )

    assert fit.iterations_run >= 3
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run >= 1
    assert len(fit.sweep_norm_trace) == fit.iterations_run
    assert fit.convergence_reason in {"relative_tolerance", "max_sweeps"}


def test_fit_run_eff_default_keeps_fixed_rank_one_site_compatibility():
    """The default full-chain path remains a fixed-rank one-site solver."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, cutoffs=0.0)

    fit.run_eff(n_iter=2)

    assert fit.p.max_bond() == 1
    assert "two_site_splits" not in fit.info
    assert "three_site_splits" not in fit.info


@pytest.mark.parametrize("sweep_sequence", ("RL", "LR"))
def test_fit_run_eff_fixed_one_site_reuses_opposite_sweep_cache(
    sweep_sequence,
):
    """Fixed-sweep one-site run_eff reuses compatible dense environments."""
    initial = qtn.MPS_rand_state(
        5,
        bond_dim=1,
        phys_dim=2,
        dtype="complex128",
        seed=612,
    )
    target = qtn.MPS_rand_state(
        5,
        bond_dim=3,
        phys_dim=2,
        dtype="complex128",
        seed=613,
    )
    options = {
        "n_iter": 4,
        "sweep_sequence": sweep_sequence,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt._allow_sweep_environment_reuse = False

    cached.run_eff(**options)
    rebuilt.run_eff(**options)

    assert cached._sweep_environment_reuse_count == 3
    assert rebuilt._sweep_environment_reuse_count == 0
    assert len(cached.local_norm_trace) == 4
    assert np.allclose(
        cached.p.to_dense(),
        rebuilt.p.to_dense(),
        atol=1.0e-12,
    )


def test_fit_run_eff_one_site_default_alternates_directions():
    """Default run_eff sweeps left-to-right and then right-to-left."""
    initial = qtn.MPS_computational_state("000", dtype="complex128")
    fit = py.FIT(initial.copy(), p=initial, cutoffs=0.0)

    fit.run_eff(n_iter=2)

    assert fit.iterations_run == 2
    assert fit.final_direction == "L"
    assert fit.final_center_site == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_iter": 1, "rtol": 1.0e-8}, "n_iter >= 2"),
        ({"n_iter": 2, "rtol": 1.0e-8, "min_iter": 1}, "min_iter >= 2"),
    ],
)
def test_fit_run_eff_rtol_requires_two_sweeps(kwargs, match):
    """Adaptive run_eff must have two retained norms to compare."""
    initial = qtn.MPS_computational_state("000", dtype="complex128")
    fit = py.FIT(initial.copy(), p=initial, cutoffs=0.0)

    with pytest.raises(ValueError, match=match):
        fit.run_eff(**kwargs)


def test_fit_run_eff_three_site_requires_three_sites():
    """A three-site full-chain update needs a sufficiently long chain."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    fit = py.FIT(state.copy(), p=state)

    with pytest.raises(ValueError, match="at least three sites"):
        fit.run_eff(block_size=3)


def test_fit_gate_three_site_native_splits_and_keeps_outside_bonds():
    """Three-site FIT should use two native splits within the active window."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        collect_split_diagnostics=True,
        n_iter=2,
        block_size=3,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        three_site_sweeps=2,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)
    assert fit.p.bond_size(3, 4) == 1
    assert len(fit.info["three_site_splits"]) == 4
    assert all(
        len(record["truncation_errors"]) == 2
        for record in fit.info["three_site_splits"]
    )
    timing = fit.get_timing()
    assert [record["direction"] for record in timing] == ["R", "L"]
    assert all(record["block_size"] == 3 for record in timing)
    assert all(
        len(site_timing["sites"]) == 3
        for record in timing
        for site_timing in record["site_timings"]
    )


def test_fit_gate_three_site_warmup_then_one_site_refinement():
    """Three-site warm-up should switch to one-site polishing sweeps."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        adaptive_block_sweeps=None, two_site_transition_sweeps=0,
        collect_split_diagnostics=True,
        n_iter=3,
        block_size=3,
        three_site_sweeps=1,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        timing=True,
    )

    assert [record["block_size"] for record in fit.get_timing()] == [3, 1, 1]
    assert len(fit.info["three_site_splits"]) == 2
    # The 3->1 transition extends two terminal boundaries instead of
    # rebuilding the fixed side; the following 1->1 sweep reuses normally.
    assert fit._sweep_environment_reuse_count == 2


def test_fit_gate_polish_sweeps_update_iteration_diagnostics():
    """Explicit one-site polish sweeps count in FIT diagnostics."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        adaptive_block_sweeps=None, two_site_transition_sweeps=0, rtol=None,
        n_iter=1,
        block_size=3,
        three_site_sweeps=1,
        final_one_site_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        timing=True,
    )

    assert fit.iterations_run == 3
    assert fit.adaptive_sweeps_run == 1
    assert fit.one_site_sweeps_run == 2
    assert [record["sweep"] for record in fit.get_timing()] == [1, 2, 3]


def test_fit_gate_two_site_warmup_then_one_site_refinement():
    """Two-site warm-up should switch to fixed-rank one-site sweeps."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        collect_split_diagnostics=True,
        n_iter=4,
        block_size=2,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        timing=True,
    )

    timing = fit.get_timing()
    assert [record["block_size"] for record in timing] == [2, 2, 1, 1]
    assert len(fit.info["two_site_splits"]) == 6
    assert fit._sweep_environment_reuse_count == 3
    assert all(
        record["svd_seconds"] == 0.0
        for record in timing[2:]
    )


def test_fit_adaptive_rank_targets_follow_open_chain_capacity():
    """Adaptive FIT should use attainable 2, 4, 8, ... bond ceilings."""
    state = qtn.MPS_computational_state("00000000", dtype="complex128")

    assert py.FIT._active_bond_rank_targets(  # pylint: disable=protected-access
        state,
        0,
        7,
        16,
    ) == (2, 4, 8, 16, 8, 4, 2)

    optimizer = py.MpsOptimizer(state, gates=[], chi=16, mode="dmrg1")
    assert optimizer._mix_target_bond_dimensions() == [2, 4, 8, 16, 8, 4, 2]


def test_dmrg1_leaves_adaptive_phase_after_two_sweeps_on_rank_stagnation():
    """DMRG1 does not extend its two-site phase when rank growth stalls."""
    state = qtn.MPS_computational_state("000", dtype="complex128")
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=6,
        cutoff=1.0e-12,
        fit_adaptive_sweeps=6,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1, 1, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 4
    assert optimizer._last_dmrg_fit_diagnostics["dmrg1_one_site_locked"] is False


@pytest.mark.parametrize("n_iter", [1, 2])
def test_dmrg1_growth_requires_room_for_one_site_refinement(n_iter):
    """DMRG1 growth needs two block sweeps plus one refinement sweep."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    with pytest.raises(ValueError, match="n_iter >= 3"):
        optimizer.run(progbar=False, n_iter=n_iter, fit_rtol=None)


def test_dmrg1_under_capacity_grows_twice_then_refines():
    """DMRG1 grows an under-capacity window twice before refinement."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    bell_gate = cnot @ np.kron(hadamard, np.eye(2))
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(bell_gate, (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 1


@pytest.mark.parametrize("fit_mpo_guess", [True, False])
def test_dmrg1_optional_svd_guess(fit_mpo_guess):
    """DMRG1 can toggle the legacy switch for the direct-SVD guess."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    gate = cnot @ np.kron(hadamard, np.eye(2))
    state = qtn.MPS_computational_state("000", dtype="complex128")
    stream = [(gate, (0, 2))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0, stabilize_unitary=False)
    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="dmrg1",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        stabilize_unitary=False,
        fit_mpo_guess=fit_mpo_guess,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    assert (
        optimizer.get_fit_diagnostics()["mpo_fit_guess_used"]
        is fit_mpo_guess
    )
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1]


@pytest.mark.parametrize("fit_mpo_guess", [True, False])
def test_dmrg3_optional_svd_guess(fit_mpo_guess):
    """DMRG3 can toggle the legacy switch for the direct-SVD guess."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    gate = cnot @ np.kron(hadamard, np.eye(2))
    state = qtn.MPS_computational_state("000", dtype="complex128")
    stream = [(gate, (0, 2))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0, stabilize_unitary=False)
    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="dmrg3",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        stabilize_unitary=False,
        fit_mpo_guess=fit_mpo_guess,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    assert (
        optimizer.get_fit_diagnostics()["mpo_fit_guess_used"]
        is fit_mpo_guess
    )
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [3, 3, 2]


def test_dmrg1_latches_one_site_phase_after_full_chain_saturation():
    """After filling all bonds, later DMRG1 windows stay one-site."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    bell_gate = cnot @ np.kron(hadamard, np.eye(2))
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[
            (bell_gate, (0, 2)),
            (np.eye(4), (0, 2)),
        ],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        timing=True,
    )

    records = optimizer.get_run_timing()["fit_steps"]
    assert [record["block_size"] for record in records] == [2, 2, 1, 1, 1, 1]
    assert [record["fit_index"] for record in records] == [0, 0, 0, 1, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["dmrg1_one_site_locked"] is True


def test_dmrg1_already_at_ceiling_starts_with_one_site_sweeps():
    """A full-rank DMRG1 window should not repeat two-site warm-up."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=123,
    )
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4, dtype=np.complex128), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=1.0e-12,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [1, 1, 1]
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["adaptive_sweeps"] == 0
    assert diagnostics["one_site_refinement_sweeps"] == 3
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["guess_used"] is True
    assert optimizer._last_dmrg_fit_diagnostics["dmrg1_one_site_locked"] is True


def test_dmrg1_reopens_block_warmup_for_rank_preserving_nonlocal_target():
    """DMRG1 must rotate saturated subspaces for a nonlocal gate."""
    state = (
        qtn.MPS_computational_state("00000000", dtype="complex128")
        + qtn.MPS_computational_state("11111111", dtype="complex128")
    ) / np.sqrt(2.0)
    controlled_phase = np.diag([1.0, 1.0, 1.0, -1.0]).astype("complex128")
    stream = [(controlled_phase, (0, 7))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="mpo",
    ).run(progbar=False, cutoff=1.0e-12, stabilize_unitary=False)
    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="dmrg1",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=6,
        cutoff=1.0e-12,
        target_cutoff=1.0e-12,
        fit_rtol=None,
        stabilize_unitary=False,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["random_initialization"]["enabled"] is False
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [1, 1, 1, 1, 1, 1]


def test_dmrg1_default_ftol_window_uses_two_one_site_samples():
    """The default window of two stops after two stable one-site norms."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=124,
    )
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4, dtype=np.complex128), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=1.0e9,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["iterations"] == 2
    assert (
        optimizer._last_dmrg_fit_diagnostics["convergence_reason"]
        == "relative_tolerance"
    )


def test_dmrg2_switches_after_required_two_site_warmup():
    """DMRG2 uses two sites twice, then one-site refinement."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=123,
    )
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=1.0e-12,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 1


def test_dmrg2_rtol_can_stop_after_two_site_warmup():
    """DMRG2 tolerance stopping starts only after its two-site phase."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=1.0e9,
        fit_patience=1,
        cutoff=1.0e-12,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2


def test_dmrg3_rtol_can_stop_after_three_site_warmup():
    """DMRG3 tolerance stopping starts only after its three-site phase."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg3",
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=1.0e9,
        fit_patience=1,
        cutoff=1.0e-12,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [3, 3, 2, 1, 1]
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["adaptive_sweeps"] == 3
    assert diagnostics["one_site_refinement_sweeps"] == 2
    assert diagnostics["convergence_reason"] == "relative_tolerance"


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fit_gate_large_window_block_sizes_compare_with_timing(block_size):
    """Large active windows compare rank growth and expose benchmark stages."""
    length = 14
    max_bond = 8
    initial = qtn.MPS_rand_state(
        length,
        bond_dim=1,
        phys_dim=2,
        dtype="complex128",
        seed=921,
    )
    target = qtn.MPS_rand_state(
        length,
        bond_dim=max_bond,
        phys_dim=2,
        dtype="complex128",
        seed=922,
    )
    fit = py.FIT(target, p=initial, range_int=[0, length - 1], cutoffs=1.0e-10)

    fit.run_gate(
        adaptive_block_sweeps=None, two_site_transition_sweeps=0,
        n_iter=2,
        block_size=block_size,
        sweep_sequence="RL",
        max_bond=max_bond,
        cutoff=1.0e-10,
        timing=True,
    )

    fidelity = float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    )
    records = fit.get_timing()
    assert [record["direction"] for record in records] == ["R", "L"]
    assert all(record["timing_schema"] == 3 for record in records)
    assert all(record["active_site_count"] == length for record in records)
    expected_blocks = [block_size, block_size]
    if block_size == 3:
        expected_blocks = [3, 1]
    assert [record["block_size"] for record in records] == expected_blocks
    assert [record["update_count"] for record in records] == [
        length - active_block_size + 1
        for active_block_size in expected_blocks
    ]
    assert all(
        all(
            site_timing[stage] >= 0.0
            for site_timing in record["site_timings"]
            for stage in (
                "effective_seconds",
                "svd_seconds",
                "writeback_seconds",
                "environment_seconds",
            )
        )
        for record in records
    )
    assert all(
        record["svd_seconds"]
        == pytest.approx(
            sum(site_timing["svd_seconds"] for site_timing in record["site_timings"])
        )
        for record in records
    )

    if block_size == 1:
        assert fit.p.max_bond() == 1
        assert fidelity < 0.1
    elif block_size == 2:
        assert 1 < fit.p.max_bond() < max_bond
        assert 0.4 < fidelity < 0.9
    else:
        # The default three-site schedule uses one warm-up sweep followed by
        # one-site refinement, so the refinement sweep cannot open additional
        # bonds to reach the old two-three-site-sweep rank.
        assert 1 < fit.p.max_bond() < max_bond
        assert fidelity > 0.5


def test_fit_gate_three_site_direct_and_generic_routes_match():
    """Three-site dense direct environments must match the generic route."""
    initial, target = _three_site_ghz_target()
    options = {
        "n_iter": 2,
        "block_size": 3,
        "sweep_sequence": "RL",
        "max_bond": 2,
        "cutoff": 0.0,
    }
    direct = py.FIT(
        target,
        p=initial,
        range_int=[0, 3],
        environment_strategy="mps-direct",
    )
    generic = py.FIT(
        target,
        p=initial,
        range_int=[0, 3],
        environment_strategy="generic",
    )

    direct.run_gate(**options)
    generic.run_gate(**options)

    assert np.allclose(
        direct.p.to_dense(),
        generic.p.to_dense(),
        atol=1.0e-10,
    )


def test_fit_dense_direct_environment_matches_generic_route():
    """The dense MPS specialization must preserve the generic FIT result."""
    initial = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=208
    )
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 2),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    direct = py.FIT(
        target,
        p=initial,
        range_int=[0, 2],
        environment_strategy="mps-direct",
    )
    generic = py.FIT(
        target,
        p=initial,
        range_int=[0, 2],
        environment_strategy="generic",
    )

    options = {
        "n_iter": 2,
        "block_size": 2,
        "sweep_sequence": "RL",
        "max_bond": 3,
        "cutoff": 1.0e-12,
    }
    direct.run_gate(**options)
    generic.run_gate(**options)

    assert direct.environment_strategy == "mps-direct"
    assert generic.environment_strategy == "generic"
    assert np.allclose(direct.p.to_dense(), generic.p.to_dense(), atol=1.0e-11)


def test_fit_gate_reuses_dense_opposite_sweep_environments():
    """Dense R/L sweeps reuse only compatible cached boundary environments."""
    initial = qtn.MPS_rand_state(
        10, bond_dim=1, phys_dim=2, dtype="complex128", seed=601
    )
    target = qtn.MPS_rand_state(
        10, bond_dim=3, phys_dim=2, dtype="complex128", seed=602
    )
    options = {
        "n_iter": 4,
        "block_size": 2,
        "sweep_sequence": "RL",
        "max_bond": 3,
        "cutoff": 1.0e-12,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, range_int=[0, 9])
    uncached = py.FIT(target, p=initial, range_int=[0, 9])
    uncached._allow_sweep_environment_reuse = False

    cached.run_gate(**options)
    uncached.run_gate(**options)

    assert cached._sweep_environment_reuse_count == 3
    assert uncached._sweep_environment_reuse_count == 0
    cached_dense = np.asarray(cached.p.to_dense()).reshape(-1)
    uncached_dense = np.asarray(uncached.p.to_dense()).reshape(-1)
    overlap = np.vdot(cached_dense, uncached_dense)
    assert abs(overlap) ** 2 == pytest.approx(
        np.vdot(cached_dense, cached_dense).real
        * np.vdot(uncached_dense, uncached_dense).real,
        rel=1.0e-9,
        abs=1.0e-12,
    )


@pytest.mark.parametrize("direction", ["R", "L"])
@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fit_gate_builds_only_fixed_environments_reachable_by_block(
    monkeypatch,
    direction,
    block_size,
):
    """Fresh sweeps contract only boundaries reachable by their blocks."""
    initial = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=603
    )
    target = qtn.MPS_rand_state(
        4, bond_dim=3, phys_dim=2, dtype="complex128", seed=604
    )
    fit = py.FIT(target, p=initial, range_int=[0, 3])
    overlap_calls = 0
    original = fit._overlap_environment_site

    def count_overlap(*args, **kwargs):
        nonlocal overlap_calls
        overlap_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_overlap_environment_site", count_overlap)
    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        block_size=block_size,
        sweep_sequence=direction,
        max_bond=3,
        cutoff=1.0e-12,
    )

    window_size = 4
    fixed_count = window_size - block_size
    moving_count = (
        window_size - 1
        if block_size == 1
        else window_size - block_size
    )
    assert overlap_calls == fixed_count + moving_count


@pytest.mark.parametrize("direction", ["R", "L"])
def test_fit_single_pair_fast_path_builds_no_active_environments(
    monkeypatch,
    direction,
):
    """A terminal update covering the full window needs no active cache."""
    initial = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=605
    )
    fit = py.FIT(initial.copy(), p=initial, range_int=[1, 2])
    overlap_calls = 0
    original = fit._overlap_environment_site

    def count_overlap(*args, **kwargs):
        nonlocal overlap_calls
        overlap_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_overlap_environment_site", count_overlap)
    fit.run_gate(
        n_iter=7,
        block_size=2,
        sweep_sequence=direction,
        max_bond=2,
        rtol=None,
        single_pair_fast_path=True,
    )

    assert overlap_calls == 0


@pytest.mark.parametrize("sweep_sequence", ["RL", "LR"])
def test_fit_reuses_reversed_two_site_cache_for_one_site_refinement(
    sweep_sequence,
):
    """The final two-site boundaries exactly serve reversed one-site FIT."""
    initial = qtn.MPS_rand_state(
        5, bond_dim=1, phys_dim=2, dtype="complex128", seed=606
    )
    target = qtn.MPS_rand_state(
        5, bond_dim=3, phys_dim=2, dtype="complex128", seed=607
    )
    options = {
        "n_iter": 3,
        "block_size": 2,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 3,
        "cutoff": 1.0e-12,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, range_int=[0, 4])
    conservative = py.FIT(target, p=initial, range_int=[0, 4])
    overlap_calls = {"cached": 0, "conservative": 0}
    cached_overlap = cached._overlap_environment_site
    conservative_overlap = conservative._overlap_environment_site

    def count_cached(*args, **kwargs):
        overlap_calls["cached"] += 1
        return cached_overlap(*args, **kwargs)

    def count_conservative(*args, **kwargs):
        overlap_calls["conservative"] += 1
        return conservative_overlap(*args, **kwargs)

    cached._overlap_environment_site = count_cached
    conservative._overlap_environment_site = count_conservative
    conservative._allow_sweep_environment_reuse = False

    cached.run_gate(**options)
    conservative.run_gate(**options)

    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert overlap_calls == {"cached": 14, "conservative": 20}
    assert np.allclose(
        cached.p.to_dense(),
        conservative.p.to_dense(),
        atol=1.0e-12,
    )


@pytest.mark.parametrize("sweep_sequence", ["RL", "LR"])
def test_fit_reuses_reversed_three_site_cache_for_one_site_refinement(
    sweep_sequence,
):
    """Three-site FIT extends only two terminal boundaries before 1-site."""
    initial = qtn.MPS_rand_state(
        5, bond_dim=1, phys_dim=2, dtype="complex128", seed=608
    )
    target = qtn.MPS_rand_state(
        5, bond_dim=3, phys_dim=2, dtype="complex128", seed=609
    )
    options = {
        "two_site_transition_sweeps": 0,
        "n_iter": 3,
        "block_size": 3,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 3,
        "cutoff": 1.0e-12,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, range_int=[0, 4])
    conservative = py.FIT(target, p=initial, range_int=[0, 4])
    overlap_calls = {"cached": 0, "conservative": 0}
    cached_overlap = cached._overlap_environment_site
    conservative_overlap = conservative._overlap_environment_site

    def count_cached(*args, **kwargs):
        overlap_calls["cached"] += 1
        return cached_overlap(*args, **kwargs)

    def count_conservative(*args, **kwargs):
        overlap_calls["conservative"] += 1
        return conservative_overlap(*args, **kwargs)

    cached._overlap_environment_site = count_cached
    conservative._overlap_environment_site = count_conservative
    conservative._allow_sweep_environment_reuse = False

    cached.run_gate(**options)
    conservative.run_gate(**options)

    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert overlap_calls == {"cached": 12, "conservative": 16}
    assert np.allclose(
        cached.p.to_dense(),
        conservative.p.to_dense(),
        atol=1.0e-12,
    )


def test_fit_auto_cutoff_is_dtype_aware():
    """FIT's automatic cutoff follows the fitted tensor precision."""
    initial = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex64", seed=209
    )
    target = initial.copy(deep=True)
    target.gate_nonlocal_(qu.CNOT(), (0, 2), max_bond=None, cutoff=0.0)
    fit = py.FIT(target, p=initial, range_int=[0, 2])

    fit.run_gate(adaptive_block_sweeps=None, n_iter=1, block_size=2, cutoff="auto")

    assert fit.info["cutoff_requested"] == "auto"
    assert fit.info["cutoff_resolved"] == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    ("dtype", "expected_cutoff"),
    [("complex64", 1.0e-6), ("complex128", 1.0e-12)],
)
def test_mps_optimizer_default_cutoff_policy_is_dtype_aware(
    monkeypatch,
    dtype,
    expected_cutoff,
):
    """An omitted MPS run cutoff resolves from the live tensor dtype."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0", dtype=dtype),
        gates=[(qu.hadamard(), (0,))],
        chi=2,
        mode="svd",
    )
    calls = {}

    def record_execute(*args, **kwargs):
        calls.update(kwargs)
        return optimizer.p

    monkeypatch.setattr(optimizer, "_execute_mode", record_execute)
    optimizer.run(progbar=False)

    assert calls["cutoff"] == pytest.approx(expected_cutoff)
    assert calls["cutoff_mode"] == "rsum2"
    assert calls["mpo_cutoff_mode"] == "rsum2"


def test_fit_retag_resolves_environment_and_preserves_info_object():
    """Retagging precedes route selection and caller diagnostics stay live."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=210
    )
    target = state.copy()
    target.drop_tags()
    info = {}

    fit = py.FIT(
        target,
        p=state,
        range_int=[0, 2],
        retag=True,
        info=info,
        environment_strategy="mps-direct",
    )
    fit.run_gate(adaptive_block_sweeps=None, collect_split_diagnostics=True, n_iter=1, block_size=2, max_bond=2)

    assert fit.environment_strategy == "mps-direct"
    assert fit.info is info
    assert info["two_site_splits"]


def test_fit_retag_keeps_layered_tensor_ownership_near_mps_sites():
    """Retagging uses nearest graph sites without reordering the target."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    bonds = [f"bond{i}" for i in range(3)]
    tensors = []

    for site in range(4):
        inds = []
        shape = []
        if site:
            inds.append(bonds[site - 1])
            shape.append(1)
        if site < 3:
            inds.append(bonds[site])
            shape.append(1)
        inds.append(state.site_ind(site))
        shape.append(2)
        if site == 1:
            inds.append("path-left")
            shape.append(1)
        if site == 2:
            inds.append("path-right")
            shape.append(1)
        tensors.append(qtn.Tensor(np.ones(shape), inds=inds))

    # These tensors have no physical legs. Their graph distances to sites 1
    # and 2 are deliberately different, so propagation must not simply walk
    # in the first tag's direction.
    tensors.extend(
        [
            qtn.Tensor(
                np.ones((1, 1)),
                inds=("path-left", "path-middle"),
            ),
            qtn.Tensor(
                np.ones((1, 1)),
                inds=("path-middle", "path-right"),
            ),
        ]
    )
    target = qtn.TensorNetwork(tensors)
    original_order = tuple(target.tensor_map)

    fit = py.FIT(target, p=state, retag=True)

    assert tuple(fit.tn.tensor_map) == original_order

    left_path = fit.tn.tensor_map[original_order[4]]
    right_path = fit.tn.tensor_map[original_order[5]]
    assert set(left_path.tags) == {"I1", "I2"}
    assert set(right_path.tags) == {"I1", "I2"}


def test_fit_retag_preserves_layered_mps_backbone_regions():
    """Canonical layered tags keep base MPS tensors on their original sites."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    target = optimizer._build_norm_target(
        state,
        qu.CNOT(),
        (1, 2),
        0.0,
        target_strategy="layered",
    )

    fit = py.FIT(target, p=state, retag=True)

    assert [set(tensor.tags) for tensor in fit.tn] == [
        {"I0"},
        {"I1"},
        {"I2"},
        {"I3"},
        {"I1"},
        {"I2"},
    ]


def test_fit_direct_environment_requires_unique_tensor_per_site():
    """A tensor carrying every site tag must not be cached multiple times."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=211
    )
    collapsed_target = qtn.TensorNetwork([state.contract(all)])

    automatic = py.FIT(
        collapsed_target,
        p=state,
        range_int=[0, 2],
    )
    assert automatic.environment_strategy == "generic"

    with pytest.raises(ValueError, match="exactly one target tensor per site"):
        py.FIT(
            collapsed_target,
            p=state,
            range_int=[0, 2],
            environment_strategy="mps-direct",
        )


def test_new_fit_configuration_is_keyword_only():
    """New policy controls must not extend the legacy positional API."""
    fit_parameters = inspect.signature(py.FIT).parameters
    run_parameters = inspect.signature(py.MpsOptimizer.run).parameters

    assert fit_parameters["environment_strategy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["n_iter"].default == 8
    assert run_parameters["cutoff"].default == "auto"
    assert run_parameters["cutoff_mode"].default == "auto"
    assert run_parameters["fit_rtol"].default == "auto"
    assert run_parameters["quality_check_every"].default is False
    assert run_parameters["fit_overlap_diagnostics"].default is False
    assert run_parameters["fit_init_rand_strength"].default == 0.0
    for name in (
        "fit_min_iter",
        "fit_rtol",
        "fit_patience",
        "fit_block_size",
        "fit_adaptive_sweeps",
        "fit_sweep_sequence",
        "fit_layer_size",
        "fit_max_span",
        "fit_three_site_sweeps",
        "target_cutoff",
        "fit_target_strategy",
        "fit_single_pair_fast_path",
        "fit_overlap_diagnostics",
        "stabilize_unitary",
        "fit_stabilize_unitary",
        "timing",
        "timing_sync_device",
        "quality_check_every",
        "quality_check_repair",
    ):
        assert run_parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_mps_optimizer_fit_defaults_are_adaptive_and_fixed_pair_sweeps():
    """The public DMRG defaults retain requested adjacent-pair sweeps."""
    fit_rtol = inspect.signature(py.MpsOptimizer.run).parameters["fit_rtol"]
    fast_path = inspect.signature(py.MpsOptimizer.run).parameters[
        "fit_single_pair_fast_path"
    ]

    assert fit_rtol.default == "auto"
    assert fast_path.default is False


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [("complex64", 1.0e-5), ("complex128", 1.0e-9)],
)
def test_mps_optimizer_auto_fit_rtol_tracks_state_dtype(dtype, expected):
    """The automatic FIT tolerance follows the live MPS precision."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype=dtype),
        gates=[],
        chi=2,
        mode="dmrg",
    )

    assert optimizer._resolve_fit_rtol("auto") == pytest.approx(expected)


def test_mps_optimizer_default_runs_requested_adjacent_pair_sweeps():
    """The default MPS optimizer path does not stop after one pair update."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(progbar=False, n_iter=2, fit_rtol=None, timing=True)

    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["iterations"] == 2
    assert diagnostics["convergence_reason"] != "single_pair_exact"


def test_mps_optimizer_dmrg2_adjacent_pair_defaults_to_one_update():
    """Named DMRG2 keeps its one-update schedule for neighboring gates."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(progbar=False, n_iter=8, fit_rtol=None, timing=True)

    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["iterations"] == 1
    assert diagnostics["convergence_reason"] == "single_pair_exact"


def test_fit_two_site_single_pair_fast_path_is_structurally_converged():
    """One variational pair needs one effective tensor and one native SVD."""
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=215
    )
    fit = py.FIT(state.copy(), p=state, range_int=[1, 2])

    fit.run_gate(
        n_iter=7,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        rtol=None,
        single_pair_fast_path=True,
    )

    assert fit.iterations_run == 1
    assert fit.converged is True
    assert fit.convergence_reason == "single_pair_exact"
    assert fit.last_relative_change == 0.0
    assert fit.final_center_site == 2
    assert fit.final_direction == "R"


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
def test_dmrg_modes_advance_after_one_update_per_two_site_window(mode):
    """A two-site window gets one exact update, independent of n_iter."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[
            (qu.hadamard(), (0,)),
            (qu.CNOT(), (0, 1)),
            (qu.CNOT(), (1, 2)),
        ],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=None,
        fit_single_pair_fast_path=True,
        timing=True,
    )

    records = optimizer.get_run_timing()["fit_steps"]
    assert [record["fit_index"] for record in records] == [0, 1]
    assert [record["sweep"] for record in records] == [1, 1]
    assert [record["block_size"] for record in records] == [2, 2]
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["iterations"] == 1
    assert diagnostics["block_size"] == 2
    assert diagnostics["adaptive_sweeps"] == 1
    assert diagnostics["one_site_refinement_sweeps"] == 0
    assert diagnostics["convergence_reason"] == "single_pair_exact"
    assert diagnostics["center_site"] == 2


def test_dense_layered_fit_target_matches_materialized_mps_target():
    """Lazy gate layers must preserve the exact uncompressed target state."""
    state = qtn.MPS_rand_state(
        5, bond_dim=3, phys_dim=2, dtype="complex128", seed=216
    )
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    gate = qu.CNOT()

    materialized = optimizer._build_norm_target(
        state,
        gate,
        (0, 4),
        0.0,
        target_strategy="mps",
    )
    layered = optimizer._build_norm_target(
        state,
        gate,
        (0, 4),
        0.0,
        target_strategy="layered",
    )

    assert layered.num_tensors == state.L + 2
    assert np.allclose(layered.to_dense(), materialized.to_dense(), atol=1.0e-12)


def test_layered_fit_resolves_boundary_bonds_locally_and_caches_them():
    """Layered boundary discovery must not scan the global target index map."""
    state = qtn.MPS_rand_state(
        6, bond_dim=2, phys_dim=2, dtype="complex128", seed=608
    )
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    target = optimizer._build_norm_target(
        state,
        qu.CNOT(),
        (2, 3),
        0.0,
        target_strategy="layered",
    )
    fit = py.FIT(
        target,
        p=state,
        range_int=[2, 3],
        copy_target=False,
    )

    class LocalOnlyIndexMap(dict):
        def items(self):
            raise AssertionError("layered FIT scanned the global index map")

    fit.tn.ind_map = LocalOnlyIndexMap(fit.tn.ind_map)
    fit.run_gate(
        n_iter=1,
        block_size=2,
        max_bond=4,
        cutoff=0.0,
        single_pair_fast_path=True,
    )

    assert set(fit._target_bond_cache) == {(1, 2), (3, 4)}
    assert fit._target_bond(1, 2) == fit._target_bond_cache[(1, 2)]
    assert fit._target_bond(3, 4) == fit._target_bond_cache[(3, 4)]


def test_layered_dmrg_batch_target_avoids_intermediate_mps_rank_growth():
    """A target block should add two small tensors per gate, not copy/split MPSs."""
    state = qtn.MPS_rand_state(
        5, bond_dim=2, phys_dim=2, dtype="complex128", seed=217
    )
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    gates = [qu.CNOT(), qu.CNOT()]
    locations = [(0, 4), (1, 3)]

    layered = optimizer._build_dmrg_batch_target(
        state,
        gates,
        locations,
        0.0,
        target_strategy="layered",
    )
    materialized = optimizer._build_dmrg_batch_target(
        state,
        gates,
        locations,
        0.0,
        target_strategy="mps",
    )

    assert layered.num_tensors == state.L + 2 * len(gates)
    assert np.allclose(layered.to_dense(), materialized.to_dense(), atol=1.0e-12)


def test_unitary_fit_stabilization_reuses_known_center_norm(monkeypatch):
    """Stabilization should not canonicalize a FIT result a second time."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="dmrg")
    state[1].modify(data=state[1].data * 0.5)

    def unexpected_canonicalization(*_args, **_kwargs):
        raise AssertionError("known FIT center should bypass canonicalization")

    monkeypatch.setattr(
        optimizer,
        "_canonical_span_norm",
        unexpected_canonicalization,
    )
    optimizer._stabilize_unitary_fit_state(
        state,
        (0, 1),
        1.0,
        current_norm=0.5,
        center_site=1,
    )

    assert float(np.real(state.norm())) == pytest.approx(1.0, abs=1.0e-12)
    assert optimizer._current_orthog(state) == (1, 1)


def test_fit_gate_two_site_timing_reports_pairs_and_directions():
    """Alternating two-site timing should identify every optimized pair."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=204
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        timing=True,
    )

    records = fit.get_timing()
    assert [record["direction"] for record in records] == ["R", "L"]
    assert all(record["block_size"] == 2 for record in records)
    assert all(record["site_count"] == 2 for record in records)
    assert all(
        len(site_timing["sites"]) == 2
        for record in records
        for site_timing in record["site_timings"]
    )
    assert all(
        {
            "effective_seconds",
            "svd_seconds",
            "writeback_seconds",
            "environment_seconds",
        }.issubset(site_timing)
        for record in records
        for site_timing in record["site_timings"]
    )


def test_fit_gate_two_site_final_polish_only_spans_large_windows():
    """Two-site FIT can polish a large window without touching a pair window."""
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=205
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])
    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        final_one_site_sweeps=1,
        timing=True,
    )

    records = fit.get_timing()
    assert [record["block_size"] for record in records] == [2, 1]
    assert [record["direction"] for record in records] == ["R", "L"]
    assert [record["site_count"] for record in records] == [2, 3]
    assert fit._sweep_environment_reuse_count == 1

    pair = py.FIT(state.copy(), p=state, range_int=[1, 2])
    pair.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        block_size=2,
        final_one_site_sweeps=1,
        timing=True,
    )
    assert [record["block_size"] for record in pair.get_timing()] == [2]


def test_fit_alternating_sweeps_reuse_opposite_canonical_form(monkeypatch):
    """An R/L pair should not repeat the canonicalization boundary pass."""
    initial = qtn.MPS_rand_state(
        5, bond_dim=2, phys_dim=2, dtype="complex128", seed=220
    )
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 3),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    options = {
        "block_size": 2,
        "max_bond": 3,
        "cutoff": 1.0e-12,
    }

    # Two separate calls intentionally force the old, conservative
    # preparation pass before the L sweep and provide a numerical reference.
    reference = py.FIT(target, p=initial.copy(), range_int=[0, 3])
    reference.run_gate(adaptive_block_sweeps=None, n_iter=1, sweep_sequence="R", **options)
    reference.run_gate(adaptive_block_sweeps=None, n_iter=1, sweep_sequence="L", **options)

    counts = {"left": 0, "right": 0}
    original_left = qtn.MatrixProductState.left_canonize_site
    original_right = qtn.MatrixProductState.right_canonize_site

    def count_left(state, *args, **kwargs):
        counts["left"] += 1
        return original_left(state, *args, **kwargs)

    def count_right(state, *args, **kwargs):
        counts["right"] += 1
        return original_right(state, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "left_canonize_site", count_left)
    monkeypatch.setattr(qtn.MatrixProductState, "right_canonize_site", count_right)

    optimized = py.FIT(target, p=initial.copy(), range_int=[0, 3])
    optimized.run_gate(n_iter=2, sweep_sequence="RL", **options)

    # The first R sweep prepares sites 3, 2, and 1. The following L sweep
    # consumes the canonical form produced by the R sweep's SVDs directly.
    assert counts == {"left": 0, "right": 3}
    assert np.allclose(
        optimized.p.to_dense(),
        reference.p.to_dense(),
        atol=1.0e-10,
    )


def test_timing_synchronizer_waits_on_jax_stage_outputs():
    """JAX barriers follow new stage results rather than an old MPS leaf."""

    class FakeJaxArray:
        __module__ = "jaxlib._jax"

        def __init__(self):
            self.waits = 0

        def block_until_ready(self):
            self.waits += 1

    source = FakeJaxArray()
    result_a = FakeJaxArray()
    result_b = FakeJaxArray()
    synchronizer = fitting_local_module._BackendSynchronizer.from_value(source)

    assert synchronizer.backend == "jax"
    synchronizer.synchronize((result_a, result_b), fallback=source)
    assert result_a.waits == 1
    assert result_b.waits == 1
    assert source.waits == 0
    assert (
        fitting_local_module._BackendSynchronizer.from_value(np.ones(1))
        is None
    )


def test_dmrg_synchronized_timing_marks_device_complete_stages(monkeypatch):
    """Opt-in synchronized profiling should be visible in every timing layer."""
    synchronizations = []

    class RecordingSynchronizer:
        def synchronize(self, value, *, fallback=None):
            synchronizations.append((value, fallback))

    monkeypatch.setattr(
        py.FIT,
        "_make_backend_synchronizer",
        staticmethod(lambda _state: RecordingSynchronizer()),
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        timing=True,
        timing_sync_device=True,
    )

    timing = optimizer.get_run_timing()
    assert synchronizations
    assert any(value is not optimizer.p for value, _fallback in synchronizations)
    assert timing["timing_sync_device"] is True
    assert timing["fit_steps"][0]["timing_sync_device"] is True
    assert timing["stages"]["dmrg.stabilize"]["calls"] == 1


def test_fit_gate_timing_retains_failed_partial_sweep():
    """Profiling must keep work completed before a failed FIT validation."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=206
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    with pytest.raises(FloatingPointError, match="non-finite tensor data"):
        fit.run_gate(
            n_iter=2,
            block_size=2,
            max_bond=2,
            finite_check=lambda _state: False,
            timing=True,
        )

    records = fit.get_timing()
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["site_count"] == 2
    assert records[0]["error"].startswith("FloatingPointError:")


def test_dmrg_complex64_deep_unitary_stream_keeps_working_norm_stable():
    """Unitary FIT stabilization should prevent complex64 norm underflow."""
    state = qtn.MPS_computational_state("00", dtype="complex64")
    gates = [(qu.hadamard(dtype="complex64"), (0,)), (qu.CNOT(dtype="complex64"), (0, 1))] * 180
    optimizer = py.MpsOptimizer(
        state,
        gates=gates,
        chi=1,
        mode="dmrg",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=1,
        cutoff=0.0,
        target_cutoff=0.0,
        stabilize_unitary=True,
    )

    raw = out.copy()
    raw.exponent = 0.0
    assert py.MpsOptimizer._mps_data_is_finite(out)
    assert float(np.real(raw.norm())) == pytest.approx(1.0, abs=2.0e-5)


def test_unitary_norm_overshoot_tolerance_is_dtype_aware():
    """Float32 roundoff is tolerated without hiding larger overshoots."""
    complex64 = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex64"),
        gates=[],
        chi=2,
        mode="mpo",
    )
    complex128 = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="mpo",
    )

    assert complex64._unitary_norm_overshoot_tolerance() == pytest.approx(
        128.0 * np.finfo(np.float32).eps
    )
    assert complex128._unitary_norm_overshoot_tolerance() == pytest.approx(
        1.0e-6
    )
    complex64._finite_check_enabled = True
    complex128._finite_check_enabled = True

    event = complex64._record_norm_event(
        "unitary_compression",
        expected_norm=1.0,
        observed_norm=np.sqrt(1.0 + 1.0e-5),
        where=(0, 1),
    )
    assert event["fidelity_raw"] == pytest.approx(1.0 + 1.0e-5)
    assert event["local_fidelity"] == pytest.approx(1.0)

    with pytest.raises(FloatingPointError, match="squared ratio"):
        complex64._record_norm_event(
            "unitary_compression",
            expected_norm=1.0,
            observed_norm=np.sqrt(1.0 + 2.0e-5),
            where=(0, 1),
        )

    with pytest.raises(FloatingPointError, match="squared ratio"):
        complex128._record_norm_event(
            "unitary_compression",
            expected_norm=1.0,
            observed_norm=np.sqrt(1.0 + 2.0e-6),
            where=(0, 1),
        )


def test_dmrg_torch_complex64_two_site_fit_grows_native_dense_bond():
    """Torch complex64 FIT should retain dtype while using two-site SVD."""
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("00", dtype="complex64")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex64, device="cpu"))
    gates = [
        (
            torch.as_tensor(
                np.array(qu.hadamard(), copy=True), dtype=torch.complex64
            ),
            (0,),
        ),
        (
            torch.as_tensor(np.array(qu.CNOT(), copy=True), dtype=torch.complex64),
            (0, 1),
        ),
    ]
    optimizer = py.MpsOptimizer(
        state,
        gates=gates,
        chi=2,
        mode="fit",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=0.0,
        target_cutoff=0.0,
    )

    expected = np.zeros(4, dtype=np.complex64)
    expected[[0, 3]] = 1.0 / np.sqrt(2.0)
    assert out.max_bond() == 2
    assert all(tensor.data.dtype == torch.complex64 for tensor in out.tensors)
    assert np.allclose(out.to_dense().cpu().numpy().reshape(-1), expected, atol=2.0e-6)


def test_fit_gate_three_site_torch_complex64_uses_native_splits():
    """Torch complex64 three-site FIT should preserve the backend dtype."""
    torch = pytest.importorskip("torch")
    initial, target = _three_site_ghz_target()
    initial.apply_to_arrays(py.backend_torch(dtype=torch.complex64, device="cpu"))
    target.apply_to_arrays(py.backend_torch(dtype=torch.complex64, device="cpu"))
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        n_iter=2,
        block_size=3,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
    )

    assert all(tensor.data.dtype == torch.complex64 for tensor in fit.p.tensors)
    assert np.allclose(
        fit.p.to_dense().cpu().numpy(),
        target.to_dense().cpu().numpy(),
        atol=2.0e-5,
    )


def test_fit_run_gate_reuse_resets_per_run_traces_and_split_diagnostics():
    """Reusing FIT should report only the latest invocation's sweep work."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=250,
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=2,
        verbose=True,
        block_size=2,
        max_bond=2,
        collect_split_diagnostics=True,
    )
    assert len(fit.local_norm_trace) == 2
    assert len(fit.info["two_site_splits"]) == 4

    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        verbose=True,
        block_size=2,
        max_bond=2,
        collect_split_diagnostics=True,
    )

    assert len(fit.local_norm_trace) == 1
    assert len(fit.fidelity_trace) == 1
    assert len(fit.info["two_site_splits"]) == 2
    assert "three_site_splits" not in fit.info


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fit_reduces_exactly_one_tensor_norm_per_sweep(monkeypatch, block_size):
    """Intermediate local updates must not pay for unused norm reductions."""
    state = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=251,
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 3])
    original_norm = qtn.Tensor.norm
    norm_calls = []

    def count_norm(tensor, *args, **kwargs):
        norm_calls.append(tensor)
        return original_norm(tensor, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "norm", count_norm)
    fit.run_gate(
        n_iter=2,
        block_size=block_size,
        three_site_sweeps=2 if block_size == 3 else 1,
        max_bond=2,
        rtol=None,
    )

    assert len(norm_calls) == 2
    assert len(fit.local_norm_trace) == 2


@pytest.mark.parametrize("method_name", ["run", "run_eff"])
@pytest.mark.parametrize("n_iter", [0, -1, 1.5])
def test_fit_full_chain_runs_validate_iteration_count(method_name, n_iter):
    """All FIT entry points reject empty, negative, and fractional sweeps."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    fit = py.FIT(state.copy(), p=state)

    with pytest.raises(ValueError, match="n_iter must be a positive integer"):
        getattr(fit, method_name)(n_iter=n_iter)
