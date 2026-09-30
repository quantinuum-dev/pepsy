"""MPS compression modes regression tests."""


import inspect
import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
import pepsy as py
import pepsy.fitting.local as fitting_local_module
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


from _mps_test_helpers import (
    _non_unitary_entangling_gate,
    _two_branch_flip_submpo,
)


pytestmark = [pytest.mark.core, pytest.mark.mps]


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
def test_mps_optimizer_long_range_dmrg_seeds_disposable_fit_guess(mode):
    """DMRG keeps the target exact and seeds only the disposable FIT guess."""
    stream = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 7)),
    ]
    reference = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        stream,
        chi=4,
        mode="mpo",
    ).run(progbar=False)

    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        stream,
        chi=4,
        mode=mode,
    )
    out = optimizer.run(progbar=False, n_iter=4, fit_rtol=None)

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    assert out.max_bond() == 2
    assert optimizer.norm_diagnostics()["infidelity"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["backend"] == "fit"
    assert diagnostics["fallback"] is False
    initialization = diagnostics["random_initialization"]
    assert diagnostics["mpo_fit_guess_used"] is True
    assert diagnostics["guess_used"] is True
    assert diagnostics["guess_method"] == "src"
    assert initialization["enabled"] is False
    assert initialization["reason"] == "guess_src"


def test_randomized_fit_guess_is_disposable_and_active_only():
    """Random initialization expands a copy, never the live current MPS."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="dmrg2")
    guess, initialization = optimizer._build_randomized_fit_guess(
        state,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-4,
    )
    assert guess is not state
    assert [state.bond_size(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert [guess.bond_size(i, i + 1) for i in range(3)] == [2, 2, 2]
    assert initialization["enabled"] is True
    assert [record["new_rank"] for record in initialization["bonds"]] == [2, 2, 2]


def test_random_fit_guess_preserves_existing_bond_dimensions_and_is_deterministic():
    """The random-only guess perturbs p without changing its bond ranks."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="dmrg2")
    guess_a, info_a = optimizer._build_randomized_fit_guess(
        state,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-2,
        expand=False,
        seed=17,
    )
    guess_b, info_b = optimizer._build_randomized_fit_guess(
        state,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-2,
        expand=False,
        seed=17,
    )

    assert info_a["enabled"] is True
    assert info_a["expanded"] is False
    assert info_a["sites"] == [0, 1, 2, 3]
    assert info_b == info_a
    assert [guess_a.bond_size(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert all(
        np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(guess_a.tensors, guess_b.tensors)
    )
    assert any(
        not np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(state.tensors, guess_a.tensors)
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "random",
        "random_expand",
        "guess_direct",
        "guess_zipup",
        "svd_guess",
    ],
)
def test_mps_optimizer_fit_initial_guess_strategy_is_diagnostic(strategy):
    """Each initial-guess strategy remains separate from FIT's target."""
    stream = [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 3))]
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        stream,
        chi=2,
        mode="dmrg2",
    )
    optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=1.0e-12,
        fit_rtol=None,
        fit_init_strategy=strategy,
        fit_init_rand_strength=1.0e-2,
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy_requested"] == strategy
    assert diagnostics["fit_init_strategy"] == strategy
    assert diagnostics["backend"] == "fit"
    if strategy == "random":
        assert diagnostics["random_initialization"]["expanded"] is False
        assert diagnostics["random_initialization"]["enabled"] is True
    elif strategy == "random_expand":
        assert diagnostics["random_initialization"]["expanded"] is True
        assert diagnostics["random_initialization"]["enabled"] is True
    elif strategy.startswith("guess_") or strategy == "svd_guess":
        assert diagnostics["svd_guess_used"] is True
        assert diagnostics["guess_used"] is True
    else:
        assert diagnostics["mpo_fit_guess_used"] is False


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    ],
)
def test_mps_optimizer_mpo_method_modes(method):
    """Every Quimb method is selectable as ``quimb-<method>``."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=f"quimb-{method}",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert out.max_bond() <= 2
    assert optimizer.mode == f"quimb-{method}"


@pytest.mark.parametrize(
    "method", ["sdc", "sdc-oversample", "sdcr", "sdcr-oversample"]
)
def test_mps_optimizer_successive_compression_modes_are_version_gated(method):
    """New SDC/SDCR modes never silently fall back on older Quimb."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=f"quimb-{method}",
    )
    supported = mps_optimizer_module._quimb_compression_method_available(method)

    if not supported:
        with pytest.raises(
            NotImplementedError,
            match=f"{method.split('-', 1)[0]} compressor",
        ):
            optimizer.run(progbar=False, cutoff=1.0e-12)
        return

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )
    assert out.max_bond() <= 2
    assert optimizer.mode == f"quimb-{method}"


@pytest.mark.parametrize("method", ["sdc", "sdcr"])
def test_mps_optimizer_bare_successive_mode_normalizes_to_quimb(method):
    """Bare SDC and SDCR spellings are first-class MPS modes."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=method,
    )
    assert optimizer.mode == f"quimb-{method}"

    if not mps_optimizer_module._quimb_compression_method_available(method):
        with pytest.raises(
            NotImplementedError,
            match=f"{method} compressor",
        ):
            optimizer.run(progbar=False, cutoff=1.0e-12)
        return

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )
    assert out.max_bond() <= 2


def test_mps_optimizer_sdcr_uses_relative_environment_cutoff():
    """SDCR never forwards cumulative cutoffs to randomized environments."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=2,
        mode="quimb-sdcr",
    )

    options = optimizer._submpo_compress_opts(  # pylint: disable=protected-access
        "sdcr",
        cutoff=1.0e-12,
        cutoff_mode="rsum2",
    )

    assert options["cutoff"] == 0.0
    assert options["cutoff_mode"] == "rel"


@pytest.mark.parametrize(
    "method", ["sdc", "sdc-oversample", "sdcr", "sdcr-oversample"]
)
def test_mps_optimizer_successive_fit_init_strategies_are_version_gated(method):
    """SDC and SDCR are available as FIT warm-start methods when installed."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg2",
    )
    supported = mps_optimizer_module._quimb_compression_method_available(method)

    if not supported:
        with pytest.raises(
            NotImplementedError,
            match=f"{method.split('-', 1)[0]} compressor",
        ):
            optimizer.run(
                progbar=False,
                n_iter=2,
                fit_rtol=None,
                fit_init_strategy=f"guess-{method}",
                stabilize_unitary=False,
            )
        return

    out = optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        fit_init_strategy=f"guess-{method}",
        stabilize_unitary=False,
    )
    diagnostics = optimizer.get_fit_diagnostics()
    assert out.max_bond() <= 2
    assert diagnostics["fit_init_strategy"] == f"guess_{method}"
    assert diagnostics["guess_method"] == method
    assert diagnostics["guess_used"] is True


@pytest.mark.parametrize(
    "mode, method",
    [
        ("quimb", "direct"),
        ("quimb-direct", "direct"),
        ("quimb-src", "src"),
        ("mpo", "direct"),
        ("mpo-direct", "direct"),
        ("mpo-src", "src"),
    ],
)
def test_mps_optimizer_quimb_mode_aliases(mode, method):
    """Quimb names are canonical while the old MPO names remain valid."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert out.max_bond() <= 2
    assert optimizer._mode_mpo_method(optimizer.mode) == method


@pytest.mark.parametrize(
    ("mode", "canonical", "method"),
    [
        ("direct", "quimb-direct", "direct"),
        ("dm", "quimb-dm", "dm"),
        ("zipup", "quimb-zipup", "zipup"),
        ("src", "quimb-src", "src"),
        ("srcmps", "quimb-srcmps", "srcmps"),
        ("fit-projector", "quimb-fit-projector", "fit-projector"),
    ],
)
def test_mps_optimizer_accepts_bare_quimb_method_modes(mode, canonical, method):
    """Bare Quimb method names normalize to the qualified backend mode."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=2,
        mode=mode,
    )

    assert optimizer.mode == canonical
    assert optimizer._is_mpo_mode(mode)
    assert optimizer._mode_mpo_method(mode) == method


def test_mps_optimizer_defaults_to_direct_and_preserves_legacy_aliases():
    """Omitting mode selects direct compression, including sub-MPO replay."""
    assert inspect.signature(py.MpsOptimizer).parameters["mode"].default == "direct"
    state = qtn.MPS_rand_state(4, bond_dim=3, dtype="complex128", seed=917)
    stream = [(qu.rand_uni(4, seed=918), (0, 3))]
    stream.append(("submpo", qtn.MPO_identity(2, dtype="complex128"), (1, 2)))
    reference = None
    for kwargs in ({}, {"mode": "direct"}, {"mode": "mpo"}, {"mode": "quimb"}):
        optimizer = py.MpsOptimizer(state, stream, chi=2, **kwargs)
        optimizer.run(cutoff=0., timing=True)
        assert optimizer.mode == "quimb-direct"
        assert optimizer._progress_mode_name(optimizer.mode) == "direct"
        assert optimizer.get_run_timing()["mode"] == "quimb-direct"
        assert optimizer.get_fit_diagnostics() is None
        result = optimizer.to_dense()
        if reference is None:
            reference = result
        else:
            np.testing.assert_allclose(result, reference, atol=1e-12)


def test_mps_optimizer_fit_name_remains_dmrg_alias():
    """Bare ``fit`` remains DMRG while Quimb FIT stays qualified."""
    dmrg = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="fit",
    )
    quimb_fit = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="quimb-fit",
    )

    assert dmrg.mode == "dmrg"
    assert quimb_fit.mode == "quimb-fit"


@pytest.mark.parametrize("mode", ["src", "zipup"])
def test_mps_optimizer_runs_bare_quimb_method_mode(mode):
    """Bare SRC and zip-up names select the corresponding replay backend."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert optimizer.mode == f"quimb-{mode}"
    assert out.max_bond() <= 2


@pytest.mark.parametrize(
    "mode, timing_name",
    [("mpo", "direct"), ("quimb-src", "src")],
)
def test_mps_optimizer_timing_names_follow_quimb_method(mode, timing_name):
    """MPO-family timing stages expose the selected compressor name."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
        timing=True,
    )

    stages = optimizer.get_run_timing()["stages"]
    assert stages[f"{timing_name}.replay"]["calls"] == 1
    assert stages[f"{timing_name}.stabilize"]["calls"] == 1


@pytest.mark.parametrize(
    "mode, expected_desc",
    [
        ("quimb", "direct"),
        ("quimb-dm", "dm"),
        ("quimb-src", "src"),
        ("mpo", "direct"),
        ("mpo-zipup", "zipup"),
        ("mpo-src", "src"),
    ],
)
def test_mps_optimizer_progress_bar_uses_mode_name(
    monkeypatch,
    mode,
    expected_desc,
):
    """MPO-family progress bars show only the selected replay mode."""
    import tqdm as tqdm_module

    descriptors = []

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            del args
            descriptors.append(kwargs["desc"])

        def set_postfix(self, _postfix):
            pass

        def update(self, _count):
            pass

        def close(self):
            pass

    monkeypatch.setattr(tqdm_module, "tqdm", FakeProgress)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=True,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert descriptors == [expected_desc]


@pytest.mark.parametrize(
    "mode, expected_desc",
    [
        ("fit", "dmrg"),
        ("dmrg", "dmrg"),
        ("dmrg1", "dmrg1"),
        ("dmrg2", "dmrg2"),
        ("dmrg3", "dmrg3"),
    ],
)
def test_mps_optimizer_dmrg_progress_bar_uses_schedule_name(
    monkeypatch,
    mode,
    expected_desc,
):
    """Named DMRG schedules identify themselves in the progress bar."""
    import tqdm as tqdm_module

    descriptors = []

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            del args
            descriptors.append(kwargs["desc"])

        def set_postfix(self, _postfix):
            pass

        def update(self, _count):
            pass

        def close(self):
            pass

    monkeypatch.setattr(tqdm_module, "tqdm", FakeProgress)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        [(qu.CNOT(), (0, 1))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=True,
        n_iter=2,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    assert descriptors == [expected_desc]


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    ],
)
def test_mps_optimizer_guess_method_strategies(method):
    """Every Quimb method is available as a ``guess-<method>`` policy."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        fit_init_strategy=f"guess-{method}",
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy"] == f"guess_{method}"
    assert diagnostics["guess_method"] == method
    assert diagnostics["guess_used"] is True


def test_mps_optimizer_hyphenated_guess_strategy_alias():
    """The canonical ``guess-<method>`` spelling normalizes cleanly."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        fit_init_strategy="guess-src",
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy_requested"] == "guess_src"
    assert diagnostics["fit_init_strategy"] == "guess_src"
    assert diagnostics["guess_method"] == "src"


def test_mps_optimizer_default_src_guess_reaches_one_site_fit():
    """The default DMRG warm-start remains SRC after reaching ``chi``."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=20260823,
        ),
        [(np.eye(4, dtype=np.complex128), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=1,
        fit_rtol=None,
        fit_block_size=1,
        timing=True,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["block_size"] == 1
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["guess_used"] is True


def test_mps_optimizer_src_compression_seed_is_reproducible():
    """Explicit seeds control both MPO replay and disposable SRC guesses."""
    state = qtn.MPS_rand_state(
        6,
        bond_dim=2,
        phys_dim=2,
        seed=23,
        dtype="complex128",
    )
    gate = qu.CNOT()

    guess_optimizer = py.MpsOptimizer(state, gates=[], chi=3, mode="dmrg2")
    guess_a = guess_optimizer._build_compression_fit_guess(
        state,
        gate,
        (0, 5),
        method="src",
        cutoff=1.0e-12,
        cutoff_mode="rsum2",
        seed=17,
    )
    guess_b = guess_optimizer._build_compression_fit_guess(
        state,
        gate,
        (0, 5),
        method="src",
        cutoff=1.0e-12,
        cutoff_mode="rsum2",
        seed=17,
    )
    assert all(
        np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(guess_a.tensors, guess_b.tensors)
    )

    replay_a = py.MpsOptimizer(
        state.copy(), [(gate, (0, 5))], chi=3, mode="quimb-src"
    ).run(
        progbar=False,
        compression_seed=17,
        stabilize_unitary=False,
    )
    replay_b = py.MpsOptimizer(
        state.copy(), [(gate, (0, 5))], chi=3, mode="quimb-src"
    ).run(
        progbar=False,
        compression_seed=17,
        stabilize_unitary=False,
    )
    assert all(
        np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(replay_a.tensors, replay_b.tensors)
    )


@pytest.mark.parametrize("method", ["srcmps", "fit", "fit-oversample"])
def test_mps_optimizer_compression_seed_is_not_forwarded_as_quimb_option(method):
    """Seeded Quimb methods use Quimb's RNG without leaking ``seed`` kwargs."""
    state = qtn.MPS_rand_state(
        6,
        bond_dim=2,
        phys_dim=2,
        seed=23,
        dtype="complex128",
    )
    gate = qu.rand_uni(4, seed=11)
    outputs = []
    for _ in range(2):
        optimizer = py.MpsOptimizer(
            state.copy(deep=True),
            [(gate, (0, 5))],
            chi=3,
            mode=f"quimb-{method}",
        )
        out = optimizer.run(
            progbar=False,
            cutoff=1.0e-12,
            compression_seed=17,
            stabilize_unitary=False,
        )
        outputs.append([np.array(tensor.data, copy=True) for tensor in out.tensors])

    assert all(
        np.array_equal(tensor_a, tensor_b)
        for tensor_a, tensor_b in zip(outputs[0], outputs[1])
    )


def test_mps_optimizer_submpo_fit_projector_product_state_is_finite():
    """Explicit endpoint sub-MPOs disable the singular projector pre-gauge."""
    mpo = qtn.MatrixProductOperator.from_dense(
        qu.CNOT(),
        dims=(2, 2),
        sites=(0, 3),
        L=4,
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [py.MpsOptimizer.submpo_event(mpo, (0, 3))],
        chi=2,
        mode="quimb-fit-projector",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        non_unitary=True,
        stabilize_unitary=False,
    )

    assert all(np.isfinite(np.asarray(tensor.data)).all() for tensor in out.tensors)


@pytest.mark.parametrize("method", ["zipup-first", "fit-zipup", "fit-projector"])
def test_mps_optimizer_interior_oversampled_mpo_methods(method):
    """Nested Quimb methods also work on an interior sub-MPO span."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            8,
            bond_dim=2,
            phys_dim=2,
            seed=3,
            dtype="complex128",
        ),
        [(qu.CNOT(), (1, 6))],
        chi=3,
        mode=f"quimb-{method}",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert out.max_bond() <= 3
    assert all(np.isfinite(np.asarray(tensor.data)).all() for tensor in out.tensors)


@pytest.mark.parametrize("method", ["zipup-first", "fit-zipup"])
def test_mps_optimizer_interior_submpo_oversampled_methods(method):
    """Explicit interior sub-MPO events use the same safe local path."""
    mpo = _two_branch_flip_submpo(
        L=8,
        sites=(1, 6),
        targets=(1, 6),
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            8,
            bond_dim=2,
            phys_dim=2,
            seed=3,
            dtype="complex128",
        ),
        [py.MpsOptimizer.submpo_event(mpo, (1, 6))],
        chi=3,
        mode=f"quimb-{method}",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
        non_unitary=True,
    )

    assert out.max_bond() <= 3
    assert all(np.isfinite(np.asarray(tensor.data)).all() for tensor in out.tensors)


@pytest.mark.parametrize("cutoff_mode", [None, "auto"])
def test_mps_optimizer_dm_uses_native_cutoff_mode_by_default(
    monkeypatch,
    cutoff_mode,
):
    """The MPO density-matrix method keeps Quimb's native rsum1 default."""
    calls = []

    def fake_gate_nonlocal_(self, gate, where, **kwargs):
        calls.append(kwargs)
        kwargs["info"]["cur_orthog"] = (min(where), min(where))
        return self

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_nonlocal_",
        fake_gate_nonlocal_,
    )

    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="quimb-dm",
    )
    optimizer.run(
        progbar=False,
        stabilize_unitary=False,
        cutoff_mode=cutoff_mode,
    )
    assert "cutoff_mode" not in calls[-1]

    optimizer.run(
        progbar=False,
        cutoff_mode="rsum2",
        stabilize_unitary=False,
    )
    assert calls[-1]["cutoff_mode"] == "rsum2"


@pytest.mark.parametrize("mode", ["mpo", "swap", "svd"])
def test_mps_optimizer_compression_timing_separates_norm_stages(mode):
    """Enabled timing records stabilization work without diagnostic stages."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(_non_unitary_entangling_gate(), (0, 3))],
        chi=1,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        non_unitary=True,
        normalize_final=False,
        timing=True,
    )

    stages = optimizer.get_run_timing()["stages"]
    assert not any(name.startswith("infidelity.") for name in stages)
    assert not any(name.endswith(".stabilize") for name in stages)


def test_mps_optimizer_empty_run_retains_timing_sync_request():
    """An empty replay reports the same synchronization option it received."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(timing=True, timing_sync_device=True)

    timing = optimizer.get_run_timing()
    assert timing["event_count"] == 0
    assert timing["timing_sync_device"] is True


def test_mps_optimizer_mix_skips_elapsed_clock_when_timing_is_disabled():
    """Mixed-mode summaries avoid clock reads outside opt-in profiling."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="mix",
    )

    optimizer.run(progbar=False, timing=False)

    assert optimizer.last_mix_summary["elapsed_seconds"] is None


def test_mps_optimizer_timing_reports_fit_sweeps_and_sites():
    """Opt-in DMRG timing exposes FIT sweep and active-site records."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=31),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    opt.run(progbar=False, n_iter=3, timing=True)

    timing = opt.get_run_timing()
    fit_steps = timing["fit_steps"]
    assert timing["stages"]["dmrg.target"]["calls"] == 1
    assert len(fit_steps) == 3
    assert [record["sweep"] for record in fit_steps] == [1, 2, 3]
    assert all(record["status"] == "complete" for record in fit_steps)
    assert all(record["range_int"] == (0, 2) for record in fit_steps)
    assert [record["site_count"] for record in fit_steps] == [2, 2, 3]
    assert [record["direction"] for record in fit_steps] == ["R", "L", "R"]
    assert [record["block_size"] for record in fit_steps] == [2, 2, 1]
    assert [record["fit_index"] for record in fit_steps] == [0, 0, 0]
    assert [record["record_index"] for record in fit_steps] == [0, 1, 2]
    assert [len(record["site_timings"]) for record in fit_steps] == [2, 2, 3]
    assert all(
        record["elapsed_seconds"] >= 0.0
        and all(site["elapsed_seconds"] >= 0.0 for site in record["site_timings"])
        for record in fit_steps
    )


def test_mps_optimizer_timing_distinguishes_fit_calls_from_sweeps():
    """All sweeps from one target fit share a stable FIT call identifier."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3, bond_dim=2, phys_dim=2, dtype="complex128", seed=209
        ),
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (0, 2))],
        chi=2,
        mode="fit",
    )

    optimizer.run(progbar=False, n_iter=3, timing=True)

    records = optimizer.get_run_timing()["fit_steps"]
    assert [record["fit_index"] for record in records] == [0, 0, 0, 1, 1, 1]
    assert [record["record_index"] for record in records] == [0, 1, 2, 3, 4, 5]
    assert [record["sweep"] for record in records] == [1, 2, 3, 1, 2, 3]


def test_mps_optimizer_timing_does_not_enable_split_diagnostics(monkeypatch):
    """Profiling records clocks without changing FIT's SVD metadata path."""
    observed = []
    original_run_gate = mps_optimizer_module.FIT.run_gate

    def inspect_run_gate(self, *args, **kwargs):
        observed.append(kwargs.get("collect_split_diagnostics"))
        result = original_run_gate(self, *args, **kwargs)
        assert "two_site_splits" not in self.info
        return result

    monkeypatch.setattr(
        mps_optimizer_module.FIT,
        "run_gate",
        inspect_run_gate,
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=210,
        ),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(progbar=False, n_iter=3, fit_rtol=None, timing=True)

    assert observed == [False]
    assert optimizer.get_run_timing()["fit_steps"]


def test_mps_optimizer_timing_transfers_records_without_internal_copies(
    monkeypatch,
):
    """Detailed FIT records move into the run result and copy only on read."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=211,
        ),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    def fail_internal_copy(_value):
        raise AssertionError("timing collection must transfer owned records")

    with monkeypatch.context() as context:
        context.setattr(mps_optimizer_module, "deepcopy", fail_internal_copy)
        context.setattr(fitting_local_module, "deepcopy", fail_internal_copy)
        optimizer.run(progbar=False, n_iter=3, fit_rtol=None, timing=True)

    timing = optimizer.get_run_timing()
    assert timing["fit_steps"]
    timing["fit_steps"].clear()
    assert optimizer.last_run_timing["fit_steps"]
