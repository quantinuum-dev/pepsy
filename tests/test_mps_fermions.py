"""MPS fermions regression tests."""


import numpy as np
import pytest
import quimb.tensor as qtn
import pepsy as py
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


pytestmark = [pytest.mark.core, pytest.mark.optional, pytest.mark.mps]


# The fit spelling aliases dmrg and has separate mapping/replay regressions.
# Named DMRG schedules are distinct algorithms and remain in every matrix.
_NATIVE_REPLAY_MODES = (
    "dmrg", "dmrg1", "dmrg2", "dmrg3", "mpo", "svd", "swap", "perm", "mix", "exact",
)


def test_fit_gate_three_site_symmray_uses_native_block_splits():
    """Symmray three-site FIT should preserve charge and fermionic blocks."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=1,
        dtype="complex128",
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(
        adaptive_block_sweeps=None, collect_split_diagnostics=True,
        n_iter=1,
        block_size=3,
        sweep_sequence="R",
        max_bond=4,
        cutoff=0.0,
    )

    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in fit.p.tensors
    )
    assert fit.info["three_site_splits"][0]["truncation_errors"] == (
        0.0,
        0.0,
    )


def test_fit_symmray_native_environment_matches_generic_route():
    """Native Symmray environments preserve the generic FIT result."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=2,
        dtype="complex128",
    )
    target = state.copy(deep=True)
    target[1].modify(data=target[1].data * 0.75)

    native = py.FIT(
        target,
        p=state.copy(deep=True),
        range_int=[0, 2],
        environment_strategy="symmray-native",
    )
    generic = py.FIT(
        target.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[0, 2],
        environment_strategy="generic",
    )
    native.run_gate(n_iter=2, block_size=2, sweep_sequence="RL", cutoff=0.0)
    generic.run_gate(n_iter=2, block_size=2, sweep_sequence="RL", cutoff=0.0)

    assert native.environment_strategy == "symmray-native"
    native_dense = np.asarray(native.p.to_dense().to_dense()).reshape(-1)
    generic_dense = np.asarray(generic.p.to_dense().to_dense()).reshape(-1)
    overlap = np.vdot(native_dense, generic_dense)
    assert abs(overlap) ** 2 == pytest.approx(
        np.vdot(native_dense, native_dense).real
        * np.vdot(generic_dense, generic_dense).real,
        rel=1.0e-9,
        abs=1.0e-14,
    )


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (0, 1, 0, 1, 0, 1, 0, 1)),
        (
            True,
            "U1U1",
            ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)),
        ),
    ],
)
def test_fit_symmray_native_environment_preserves_dummy_modes(
    block_size, spinful, symmetry, occupations
):
    """Long native sweeps preserve dummy modes without tensor densification."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        8,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=7,
        dtype="complex128",
    )

    native = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[0, 7],
        environment_strategy="symmray-native",
    )
    generic = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[0, 7],
        environment_strategy="generic",
    )

    native.run_gate(
        n_iter=4,
        min_iter=1,
        rtol=None,
        patience=99,
        block_size=block_size,
        max_bond=4,
        cutoff=1.0e-12,
        sweep_sequence="R",
    )
    generic.run_gate(
        n_iter=4,
        min_iter=1,
        rtol=None,
        patience=99,
        block_size=block_size,
        max_bond=4,
        cutoff=1.0e-12,
        sweep_sequence="R",
    )

    native_dense = np.asarray(native.p.to_dense().to_dense()).reshape(-1)
    generic_dense = np.asarray(generic.p.to_dense().to_dense()).reshape(-1)
    overlap = np.vdot(native_dense, generic_dense)
    assert abs(overlap) ** 2 == pytest.approx(
        np.vdot(native_dense, native_dense).real
        * np.vdot(generic_dense, generic_dense).real,
        rel=1.0e-9,
        abs=1.0e-14,
    )
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        for tensor in native.p.tensors
    )


def test_fit_auto_selects_native_symmray_environment():
    """Automatic FIT routing uses the native Symmray environment path."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=False, symmetry="U1", dtype="complex128")
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=(1, 0, 1),
        seed=3,
        dtype="complex128",
    )
    fit = py.FIT(state.copy(deep=True), p=state, range_int=[0, 2])

    assert fit.environment_strategy == "symmray-native"


@pytest.mark.parametrize("block_size", [1, 2, 3])
@pytest.mark.parametrize("sweep_sequence", ["R", "RL"])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (1, 0, 1, 0, 1, 0)),
        (
            True,
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        ),
        (False, "Z2", (1, 0, 1, 0, 1, 0)),
    ],
)
def test_fit_fermionic_native_writeback_gauge_is_phase_safe(
    block_size,
    sweep_sequence,
    spinful,
    symmetry,
    occupations,
):
    """Every native FIT block size preserves an exact graded MPS target."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=17,
        dtype="complex128",
    )
    fit = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )
    run_options = {
        "n_iter": 2,
        "min_iter": 1,
        "rtol": None,
        "block_size": block_size,
        "sweep_sequence": sweep_sequence,
        "max_bond": 8,
        "cutoff": 1.0e-12,
    }
    if block_size > 1:
        run_options["adaptive_block_sweeps"] = 2
    fit.run_gate(**run_options)

    assert fit.info["fermionic_sweep_sequence"] == {
        "requested": sweep_sequence,
        "used": sweep_sequence,
        "reason": "native_conjugated_fit_gauge",
    }
    assert fit._sweep_environment_reuse_count == (1 if sweep_sequence == "RL" else 0)
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, state, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("block_size", [2, 3])
def test_fit_run_eff_fermionic_blocks_alternate_natively(block_size):
    """Full-chain native blocks reuse the conjugated RL environment gauge."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        chi=2,
        random_rounds=2,
        seed=31,
        dtype="complex128",
    )
    fit = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        environment_strategy="symmray-native",
    )
    fit.run_eff(
        n_iter=2,
        block_size=block_size,
        sweep_sequence="RL",
        max_bond=8,
        cutoff=1.0e-12,
    )

    assert fit._sweep_environment_reuse_count == 1
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, state, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize("sweep_sequence", ["RL", "LR"])
def test_fit_fermionic_block_cache_feeds_one_site_refinement(
    monkeypatch,
    sweep_sequence,
    block_size,
):
    """Native U1U1 fermions reuse reversed block-to-1 caches natively."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        chi=2,
        random_rounds=2,
        seed=61,
        dtype="complex128",
    )
    cached = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )
    conservative = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )
    conservative._allow_sweep_environment_reuse = False
    options = {
        "n_iter": 3,
        "block_size": block_size,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 8,
        "cutoff": 1.0e-12,
        "rtol": None,
    }

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native cache reuse must not call to_dense")

    array_types = {type(tensor.data) for tensor in state}
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        cached.run_gate(two_site_transition_sweeps=0, **options)
        conservative.run_gate(two_site_transition_sweeps=0, **options)

    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in cached.p
    )
    assert float(
        np.real(
            py.tn_fidelity(
                cached.p,
                conservative.p,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize(
    ("symmetry", "phys_dim", "occupations", "block_size", "sweep_sequence"),
    [
        ("U1", {0: 1, 1: 1}, [1, 0, 1, 0, 1, 0], 2, "RL"),
        (
            "U1U1",
            {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1},
            [(1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)],
            3,
            "LR",
        ),
        ("Z2", {0: 1, 1: 1}, [0, 0, 0, 0, 0, 0], 1, "RL"),
    ],
)
def test_fit_bosonic_symmray_reuses_native_reversed_environments(
    monkeypatch,
    symmetry,
    phys_dim,
    occupations,
    block_size,
    sweep_sequence,
):
    """Native bosonic caches match conservative rebuilds without densifying."""
    pytest.importorskip("symmray")
    common = {
        "L": 6,
        "symmetry": symmetry,
        "phys_dim": phys_dim,
        "site_charge": py.site_charge_from_occupations(occupations),
        "bond_dim": 3,
        "dtype": "complex128",
    }
    guess = py.SymMPS.random(seed=71, **common).tn
    target = py.SymMPS.random(seed=72, **common).tn
    cached = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
    )
    conservative = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
        environment_strategy="symmray-native",
    )
    generic = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
        environment_strategy="generic",
    )
    conservative._allow_sweep_environment_reuse = False
    options = {
        "n_iter": 3,
        "min_iter": 1,
        "rtol": None,
        "block_size": block_size,
        "sweep_sequence": sweep_sequence,
        "max_bond": 4,
        "cutoff": 1.0e-12,
    }
    if block_size > 1:
        options["adaptive_block_sweeps"] = 2

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native bosonic environment reuse must stay sparse")

    array_types = {type(tensor.data) for tensor in (*guess.tensors, *target.tensors)}
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        cached.run_gate(two_site_transition_sweeps=0, **options)
        conservative.run_gate(two_site_transition_sweeps=0, **options)

    assert cached.environment_strategy == "symmray-native"
    assert cached._allow_sweep_environment_reuse is True
    assert generic._allow_sweep_environment_reuse is False
    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and not type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in cached.p
    )
    assert float(
        np.real(
            py.tn_fidelity(
                cached.p,
                conservative.p,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-10)


def test_fit_fermionic_failure_restores_physical_ket(monkeypatch):
    """A failed native sweep cannot leak FIT's conjugated working gauge."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=False,
        symmetry="U1",
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        5,
        fermion=fermion,
        occupations=(1, 0, 1, 0, 1),
        chi=2,
        random_rounds=2,
        seed=41,
        dtype="complex128",
    )
    fit = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 3],
        environment_strategy="symmray-native",
    )

    def fail_sweep(*_args, **_kwargs):
        raise RuntimeError("injected native sweep failure")

    monkeypatch.setattr(fit, "_run_gate_two_site_sweep", fail_sweep)
    with pytest.raises(RuntimeError, match="injected native sweep failure"):
        fit.run_gate(
            adaptive_block_sweeps=None,
            n_iter=1,
            block_size=2,
            sweep_sequence="R",
            max_bond=8,
            cutoff=1.0e-12,
        )

    assert fit._fermionic_bra_working is False
    assert fit._fermionic_left_exterior_environment is None
    assert fit._fermionic_right_exterior_environment is None
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, state, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("block_size", [1, 2, 3])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (1, 0, 1, 0, 1, 0)),
        (
            True,
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        ),
        (False, "Z2", (1, 0, 1, 0, 1, 0)),
    ],
)
def test_fit_fermionic_arbitrary_target_keeps_native_guess_separate(
    block_size,
    spinful,
    symmetry,
    occupations,
    monkeypatch,
):
    """Native FIT does not replace its current state with the target."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    target = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=37,
        dtype="complex128",
    )
    guess = target.copy(deep=True)
    fit = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
        environment_strategy="symmray-native",
    )

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native sector initialization must not call to_dense")

    def fail_network_contract(*_args, **_kwargs):
        raise AssertionError(
            "native sector initialization must not build a temporary "
            "TensorNetwork"
        )

    run_options = {
        "n_iter": 2,
        "min_iter": 1,
        "rtol": None,
        "block_size": block_size,
        "sweep_sequence": "RL",
        "max_bond": 2,
        "cutoff": 1.0e-12,
    }
    if block_size > 1:
        run_options["adaptive_block_sweeps"] = 2
    array_types = {type(tensor.data) for tensor in (*guess, *target)}
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        patcher.setattr(qtn.TensorNetwork, "contract", fail_network_contract)
        fit.run_gate(**run_options)

    assert "native_sector_initialization" not in fit.info
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
@pytest.mark.parametrize("fit_init_strategy", [None, "guess_src"])
def test_mps_optimizer_native_guess_src_uses_sector_preserving_randomized_guess(
    symmetry,
    fit_init_strategy,
    monkeypatch,
):
    """Native default and explicit ``guess_src`` supply a randomized guess."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.ps_to_mps(
        4,
        fermion=fermion,
        occupations=fermion.half_filled_occupations(4),
        seed=17,
        dtype="complex128",
    )
    gate_stream = [(fermion.hopping_gate(0.2, t=1.0), (0, 3))]

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native guess_src must not call dense compression")

    def fail_quimb_guess(*_args, **_kwargs):
        raise AssertionError("native guess_src must not call Quimb guess()")

    monkeypatch.setattr(
        mps_optimizer_module,
        "_apply_dense_gate_with_method",
        fail_dense,
    )
    monkeypatch.setattr(mps_optimizer_module, "guess", fail_quimb_guess)

    optimizer = py.MpsOptimizer(state, gate_stream, chi=8, mode="dmrg2")
    run_kwargs = {}
    if fit_init_strategy is not None:
        run_kwargs["fit_init_strategy"] = fit_init_strategy
    out = optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        cutoff=1.0e-12,
        fit_init_seed=23,
        cutoff_mode="rel",
        stabilize_unitary=False,
        **run_kwargs,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy_requested"] == "guess_src"
    assert diagnostics["fit_init_strategy"] == "guess_src"
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["guess_used"] is True
    assert diagnostics["svd_guess_used"] is True
    assert diagnostics["guess_backend"] == "symmray-svd:rand"
    assert diagnostics["native_randomized_guess_used"] is True
    assert diagnostics["random_initialization"]["reason"] == "native_src"
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in out.tensors
    )


def test_mps_optimizer_mix_native_guess_direct_uses_auto_swap(monkeypatch):
    """Mixed native direct guesses stay block-sparse and disposable."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        4,
        fermion=fermion,
        occupations=fermion.half_filled_occupations(4),
        seed=17,
        dtype="complex128",
    )

    def fail_dense_guess(*args, **kwargs):
        raise AssertionError("mixed native guess-direct must not use dense compression")

    monkeypatch.setattr(
        mps_optimizer_module,
        "_apply_dense_gate_with_method",
        fail_dense_guess,
    )
    optimizer = py.MpsOptimizer(
        state,
        [(fermion.hopping_gate(0.2, t=1.0), (0, 3))],
        chi=8,
        mode="mix",
    )
    out = optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert optimizer.mix_history[0]["reason"] == "guess_direct_dmrg1"
    assert diagnostics["block_size"] == 1
    assert diagnostics["fit_init_strategy"] == "guess_direct"
    assert diagnostics["guess_method"] == "direct"
    assert diagnostics["guess_backend"] == "symmray-auto-swap"
    assert diagnostics["guess_used"] is True
    assert diagnostics["native_fermionic_warm_start"] is False
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in out.tensors
    )


def test_fit_run_eff_fermionic_keeps_current_state_as_initial_guess():
    """Native block run_eff does not replace its current state."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    occupations = (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )
    target = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=37,
        dtype="complex128",
    )
    guess = target.copy(deep=True)
    fit = py.FIT(
        target,
        p=guess,
        environment_strategy="symmray-native",
    )
    fit.run_eff(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=1.0e-12,
    )

    assert "native_sector_initialization" not in fit.info
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


def test_fit_fermionic_partial_window_reports_disconnected_target_sectors():
    """A disconnected partial FIT reports its fixed-boundary sector issue."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    occupations = (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )
    guess = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=29,
        dtype="complex128",
    )
    target = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=37,
        dtype="complex128",
    )
    fit = py.FIT(
        target,
        p=guess,
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )

    with pytest.raises(
        ValueError,
        match="disconnected charge-sector support",
    ):
        fit.run_gate(
            n_iter=2,
            block_size=2,
            sweep_sequence="RL",
            max_bond=2,
            cutoff=1.0e-12,
        )


@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (1, 0, 1, 0, 1, 0)),
        (
            True,
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        ),
        (False, "Z2", (1, 0, 1, 0, 1, 0)),
    ],
)
@pytest.mark.parametrize("fit_sweep_sequence", ["R", "RL"])
@pytest.mark.parametrize("mode", ["dmrg1", "dmrg2", "dmrg3"])
def test_mps_optimizer_named_dmrg_long_range_fermions_stay_native_and_exact(
    spinful,
    symmetry,
    occupations,
    fit_sweep_sequence,
    mode,
    monkeypatch,
):
    """Every named DMRG mode keeps U1/U1U1/Z2 grading end to end."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=23,
        dtype="complex128",
    )
    gate = fermion.hopping_gate(0.02, t=1.0)
    stream = [(gate, (0, 3))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=16,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0)
    raw_zero_cutoff_target = state.copy(deep=True)
    raw_zero_cutoff_target.gate_with_auto_swap_(
        gate,
        (0, 3),
        info={},
        swap_back=True,
        cutoff=0.0,
        cutoff_mode="rsum2",
    )
    assert float(
        np.real(
            py.tn_fidelity(
                reference,
                raw_zero_cutoff_target,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-12)

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native fermionic DMRG must not call to_dense")

    def fail_network_contract(*_args, **_kwargs):
        raise AssertionError(
            "native fermionic FIT must not build a temporary TensorNetwork"
        )

    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=16,
        mode=mode,
    )
    array_types = {type(tensor.data) for tensor in (*state, *reference)}
    array_types.add(type(gate))
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        patcher.setattr(qtn.TensorNetwork, "contract", fail_network_contract)
        out = optimizer.run(
            progbar=False,
            n_iter=3,
            fit_rtol=None,
            cutoff=1.0e-12,
            target_cutoff=0.0,
            fit_sweep_sequence=fit_sweep_sequence,
            stabilize_unitary=False,
        )

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in out
    )
    assert optimizer._last_dmrg_fit_diagnostics[
        "native_fermionic_warm_start"
    ] is True
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    if mode == "dmrg1" and diagnostics["block_size"] == 1:
        # The native warm start can already fill every active rank ceiling.
        # DMRG1 then correctly spends the complete budget on one-site FIT.
        assert diagnostics["adaptive_sweeps"] == 0
        assert diagnostics["one_site_refinement_sweeps"] == 3
    else:
        assert diagnostics["block_size"] == (3 if mode == "dmrg3" else 2)
        assert diagnostics["adaptive_sweeps"] >= 2
    assert (
        diagnostics["adaptive_sweeps"]
        + diagnostics["one_site_refinement_sweeps"]
        == 3
    )
    if mode in {"dmrg2", "dmrg3"}:
        assert diagnostics["adaptive_sweeps"] == (3 if mode == "dmrg3" else 2)
        assert diagnostics["one_site_refinement_sweeps"] == (0 if mode == "dmrg3" else 1)
    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    "symmetry",
    ["U1", "U1U1"],
)
@pytest.mark.parametrize("mode", _NATIVE_REPLAY_MODES)
def test_mps_optimizer_spinful_fermion_symmetry_mode_matrix_matches_mpo(
    symmetry,
    mode,
):
    """All supported MPS modes preserve native spinful U1 charge sectors."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.ps_to_mps(
        4,
        fermion=fermion,
        occupations=fermion.half_filled_occupations(4),
        seed=41,
        dtype="complex128",
    )
    gate_stream = [(fermion.hopping_gate(0.02, t=1.0), (0, 3))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        gate_stream,
        chi=16,
        mode="mpo",
    ).run(
        progbar=False,
        cutoff=0.0,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        gate_stream,
        chi=16,
        mode=mode,
    )
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    if mode == "perm":
        optimizer.restore_qubit_order()
        compared = optimizer.p
    else:
        compared = out

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in compared.tensors
    )
    assert float(
        np.real(py.tn_fidelity(compared, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    ("spinful", "symmetry"),
    [
        (False, "U1"),
        (True, "U1"),
        (True, "U1U1"),
        (False, "Z2"),
        (True, "Z2"),
    ],
)
@pytest.mark.parametrize("mode", _NATIVE_REPLAY_MODES)
def test_mps_optimizer_fermion_gate_stream_ps_to_mps_all_modes_native(
    spinful,
    symmetry,
    mode,
):
    """Fermion gate streams remain native and exact across all MPS modes."""
    pytest.importorskip("symmray")

    def build_case():
        fermion = py.Fermion(
            spinful=spinful,
            symmetry=symmetry,
            dtype="complex128",
        )
        params = {"t": 0.7, "mu": 0.13}
        if spinful:
            params["U"] = 1.2
        else:
            params["V"] = 0.2
        stream = list(
            fermion.gate_stream(
                ((0, 1), (1, 2), (2, 3)),
                0.01,
                sites=range(4),
                order=1,
                **params,
            )
        )
        state = py.ps_to_mps(
            4,
            fermion=fermion,
            occupations=fermion.half_filled_occupations(4),
            seed=7,
            dtype="complex128",
        )
        return state, stream

    reference_state, reference_stream = build_case()
    reference = py.MpsOptimizer(
        reference_state,
        reference_stream,
        chi=16,
        mode="exact",
    ).run(
        progbar=False,
        cutoff=0.0,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )

    state, stream = build_case()
    assert all(
        type(gate).__module__.split(".", 1)[0] == "symmray"
        and type(gate).__name__.endswith("FermionicArray")
        for gate, _ in stream
    )
    optimizer = py.MpsOptimizer(
        state,
        stream,
        chi=16,
        mode=mode,
    )
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    if mode == "perm":
        optimizer.restore_qubit_order()
        compared = optimizer.p
    else:
        compared = out

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in compared.tensors
    )
    assert float(
        np.real(py.tn_fidelity(compared, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    ("spinful", "symmetry"),
    [
        (False, "U1"),
        (False, "Z2"),
        (True, "U1"),
        (True, "U1U1"),
        (True, "Z2"),
    ],
)
@pytest.mark.parametrize("occupation_kind", ["vacuum", "full"])
def test_mps_optimizer_complex64_native_fit_short_sector_edges(
    spinful,
    symmetry,
    occupation_kind,
):
    """FIT handles short native sectors at complex64 precision."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex64",
    )
    if spinful:
        occupation = (0, 0) if occupation_kind == "vacuum" else (1, 1)
    else:
        occupation = 0 if occupation_kind == "vacuum" else 1
    occupations = (occupation, occupation)
    params = {"t": 0.3, "mu": 0.1, "V": 0.2}
    if spinful:
        params["U"] = 0.8
    stream = list(
        fermion.gate_stream(
            ((0, 1),),
            0.01,
            sites=(0, 1),
            order=2,
            **params,
        )
    )
    state = py.ps_to_mps(
        2,
        fermion=fermion,
        occupations=occupations,
        seed=13,
        dtype="complex64",
    )
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=4,
        mode="exact",
    ).run(
        progbar=False,
        cutoff=0.0,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )
    optimizer = py.MpsOptimizer(state, stream, chi=4, mode="fit")
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-6,
        target_cutoff=0.0,
        n_iter=2,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    assert py.MpsOptimizer._mps_data_is_finite(out)
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        and all(
            np.dtype(block.dtype) == np.dtype("complex64")
            for block in tensor.data.blocks.values()
        )
        for tensor in out.tensors
    )
    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=2.0e-5)


@pytest.mark.slow
@pytest.mark.parametrize("symmetry", ["U1", "U1U1", "Z2"])
@pytest.mark.parametrize("mode", _NATIVE_REPLAY_MODES)
def test_mps_optimizer_3x4_pbc_hubbard_long_range_modes_native(
    symmetry,
    mode,
):
    """Stress native Hubbard modes on a periodic 3x4 lattice."""
    pytest.importorskip("symmray")
    Lx, Ly = 3, 4
    mapper = py.OneDMap(Lx, Ly, mode="snake")
    idx2coo, coo2idx = mapper.build()
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    setup = fermion.lattice_half_filling(Lx, Ly, cyclic=True)
    occupations = tuple(
        setup.occupations[idx2coo[index]] for index in range(Lx * Ly)
    )
    mapped_edges = tuple(
        tuple(coo2idx[site] for site in edge) for edge in setup.edges
    )
    stream = list(
        fermion.gate_stream(
            mapped_edges,
            0.002,
            sites=range(Lx * Ly),
            order=1,
            t=0.8,
            U=2.0,
            mu=0.1,
        )
    )
    long_range_gate = fermion.hopping_gate(0.003, t=0.4)
    stream.extend(
        [
            (
                long_range_gate,
                (coo2idx[(0, 0)], coo2idx[(2, 2)]),
            ),
            (
                long_range_gate,
                (coo2idx[(0, 3)], coo2idx[(2, 0)]),
            ),
        ]
    )
    assert len(set(setup.edges)) == 24
    assert len(stream) == 38
    assert all(
        type(gate).__module__.split(".", 1)[0] == "symmray"
        and type(gate).__name__.endswith("FermionicArray")
        for gate, _ in stream
    )

    state = py.ps_to_mps(
        Lx * Ly,
        fermion=fermion,
        occupations=occupations,
        seed=12,
        dtype="complex128",
    )
    optimizer = py.MpsOptimizer(
        state,
        stream,
        chi=8 if mode == "exact" else 16,
        mode=mode,
    )
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-8 if mode == "exact" else 1.0e-10,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    compared = out
    if mode == "perm":
        optimizer.restore_qubit_order()
        compared = optimizer.p

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in compared.tensors
    )
    if mode == "exact":
        assert len(compared.tensors) == 1
        assert float(np.real(py.to_float(compared.norm()))) == pytest.approx(
            1.0,
            abs=1.0e-8,
        )
    else:
        reference_state = py.ps_to_mps(
            Lx * Ly,
            fermion=py.Fermion(
                spinful=True,
                symmetry=symmetry,
                dtype="complex128",
            ),
            occupations=occupations,
            seed=12,
            dtype="complex128",
        )
        reference = py.MpsOptimizer(
            reference_state,
            stream,
            chi=16,
            mode="mpo",
        ).run(
            progbar=False,
            cutoff=0.0,
            target_cutoff=0.0,
            stabilize_unitary=False,
        )
        assert float(
            np.real(
                py.tn_fidelity(
                    compared,
                    reference,
                    contraction_opt="greedy",
                )
            )
        ) == pytest.approx(1.0, abs=5.0e-5)


@pytest.mark.slow
@pytest.mark.parametrize("symmetry", ["U1", "U1U1", "Z2"])
def test_mps_optimizer_complex64_3x4_pbc_perm_stays_finite_native(symmetry):
    """Native complex64 lazy swaps remain finite on the hard lattice case."""
    pytest.importorskip("symmray")
    Lx, Ly = 3, 4
    mapper = py.OneDMap(Lx, Ly, mode="snake")
    idx2coo, coo2idx = mapper.build()
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex64",
    )
    setup = fermion.lattice_half_filling(Lx, Ly, cyclic=True)
    occupations = tuple(
        setup.occupations[idx2coo[index]] for index in range(Lx * Ly)
    )
    mapped_edges = tuple(
        tuple(coo2idx[site] for site in edge) for edge in setup.edges
    )
    stream = list(
        fermion.gate_stream(
            mapped_edges,
            0.002,
            sites=range(Lx * Ly),
            order=1,
            t=0.8,
            U=2.0,
            mu=0.1,
        )
    )
    long_range_gate = fermion.hopping_gate(0.003, t=0.4)
    stream.extend(
        [
            (
                long_range_gate,
                (coo2idx[(0, 0)], coo2idx[(2, 2)]),
            ),
            (
                long_range_gate,
                (coo2idx[(0, 3)], coo2idx[(2, 0)]),
            ),
        ]
    )
    assert {
        np.dtype(block.dtype)
        for gate, _where in stream
        for block in gate.blocks.values()
    } == {np.dtype("complex64")}
    state = py.ps_to_mps(
        Lx * Ly,
        fermion=fermion,
        occupations=occupations,
        seed=12,
        dtype="complex64",
    )
    reference_state = py.ps_to_mps(
        Lx * Ly,
        fermion=py.Fermion(
            spinful=True,
            symmetry=symmetry,
            dtype="complex64",
        ),
        occupations=occupations,
        seed=12,
        dtype="complex64",
    )
    reference = py.MpsOptimizer(
        reference_state,
        stream,
        chi=16,
        mode="mpo",
    ).run(
        progbar=False,
        cutoff=1.0e-6,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )
    optimizer = py.MpsOptimizer(state, stream, chi=16, mode="perm")
    optimizer.run(
        progbar=False,
        cutoff=1.0e-6,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )
    optimizer.restore_qubit_order()

    compared = optimizer.p
    assert py.MpsOptimizer._mps_data_is_finite(compared)
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        and all(
            np.dtype(block.dtype) == np.dtype("complex64")
            for block in tensor.data.blocks.values()
        )
        for tensor in compared.tensors
    )
    assert float(
        np.real(
            py.tn_fidelity(
                compared,
                reference,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=3.0e-4)
