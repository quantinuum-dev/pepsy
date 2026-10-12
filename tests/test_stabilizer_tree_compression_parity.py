"""Coefficient-state compression must use the ordinary TreeOptimizer engine."""

import numpy as np
import pytest
import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext

pytest.importorskip("stim")

from pepsy.optimizers import StabilizerTreeSimulator
from pepsy.optimizers.tree import TreeOptimizer, TreePlan, TreeTensorNetwork


ROUTING_MODES = (
    "tree_mpo_direct", "tree_mpo_dm", "src", "sdc", "sdcr", "zipup",
    "src-oversample", "sdc-oversample", "sdcr-oversample", "zipup-oversample",
    "dmrg", "dmrg2", "dmrg3", "mix", "mpo",
)


@pytest.mark.parametrize("mode", ROUTING_MODES)
@pytest.mark.parametrize("representation", ["gate", "compact", "full"])
def test_tree_stab_nonunitary_operator_matches_engine(mode, representation):
    from pepsy.optimizers.tree import SubTreeMPO, TreeMPO

    plan = TreePlan.from_order(range(5), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=71)
    options = dict(
        tree=plan, mode=mode, chi=2, cutoff=.001, cutoff_mode="rsum2",
        max_operator_qubits=3,
        compression_seed=51, fit_init_seed=52, fit_n_iter=3, fit_rtol=None,
        track_infidelity=True,
    )
    ordinary = TreeOptimizer(state=state, run=False, **options)
    with pytest.warns(DeprecationWarning) if mode == "mpo" else nullcontext():
        sim = StabilizerTreeSimulator(state, exact_cooling=False, **options)
    rng = np.random.default_rng(72)
    gate = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    gate /= np.linalg.norm(gate)
    where = (0, 2, 4)
    if representation == "gate":
        ordinary.apply_gate(gate, where)
        sim._apply_tree_gate(gate, where)
    else:
        builder = SubTreeMPO if representation == "compact" else TreeMPO
        operator = builder.from_gate(plan, gate, where)
        ordinary.apply_sub_mpotree(operator, where, track_norm=False)
        sim.apply([sim.subtreempo_event(operator, where)])
    np.testing.assert_allclose(
        sim.p.to_dense(), ordinary.tn.to_dense(), atol=2e-9, rtol=2e-9,
    )
    sim.p.validate(check_canonical=True)
    assert sim.p.max_bond() <= 2
    assert [entry["kind"] for entry in sim.tree_optimizer.update_history] == [
        entry["kind"] for entry in ordinary.update_history
    ]
    if ordinary.get_fit_diagnostics() is not None:
        assert sim.get_fit_diagnostics()["block_size_trace"] == (
            ordinary.get_fit_diagnostics()["block_size_trace"]
        )


@pytest.mark.parametrize("mode", ROUTING_MODES[:-1])
@pytest.mark.parametrize("cutoff_mode", ["rel", "rsum2"])
def test_tree_stab_cap_honors_relative_cutoff_and_selected_algorithm(mode, cutoff_mode):
    sim = StabilizerTreeSimulator(
        4, mode=mode, chi=2, cutoff=.01, cutoff_mode=cutoff_mode,
        compression_seed=51, fit_init_seed=52, fit_n_iter=3, fit_rtol=None,
    )
    # Scaling a nonzero physical cap must not become an absolute singular
    # value cutoff. The old independent dense builder discarded this state.
    sim.cap(1, [1e-4, 0])
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1e-4
    np.testing.assert_allclose(sim.to_statevector(), expected, atol=1e-12)
    sim.p.validate(check_canonical=True)
    if mode in {"dmrg", "dmrg2", "dmrg3", "mix"}:
        assert sim.get_fit_diagnostics() is not None
    assert sim.tree_optimizer.update_history[-1]["kind"] == "subtreempo"


@pytest.mark.parametrize("mode", ROUTING_MODES[:-1])
def test_tree_stab_zero_cap_is_exact_rank_one(mode):
    sim = StabilizerTreeSimulator(4, mode=mode, chi=2, cutoff=.01)
    sim.cap(1, [0, 1])
    np.testing.assert_array_equal(sim.to_statevector(), np.zeros(8))
    assert sim.p.max_bond() == 1
    sim.p.validate(check_canonical=True)


@pytest.mark.parametrize("mode", [
    "tree_mpo_direct", "tree_mpo_dm", "src", "sdc", "sdcr", "zipup",
    "src-oversample", "sdc-oversample", "sdcr-oversample", "zipup-oversample",
    "dmrg", "fit", "dmrg2", "dmrg3", "mix",
])
@pytest.mark.parametrize("where,axes", [((0, 5), "XY"), ((0, 2, 5), "XYZ")])
def test_tree_stab_compressed_rotation_matches_engine(mode, where, axes):
    plan = TreePlan.from_order(range(6), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=21)
    options = dict(
        tree=plan, mode=mode, chi=2, cutoff=1e-12, seed=32,
        compression_seed=42, fit_init_seed=43, fit_n_iter=4,
        fit_rtol=None, fit_min_iter=2,
    )
    ordinary = TreeOptimizer(state=state, run=False, **options)
    stabilizer = StabilizerTreeSimulator(state, exact_cooling=False, **options)

    ordinary.apply_pauli_rotation(.37, axes, where, _force_tree_mpo=True)
    stabilizer.apply([("rot", .37, axes, where)])

    np.testing.assert_allclose(
        stabilizer.p.to_dense(), ordinary.tn.to_dense(), atol=2e-9, rtol=2e-9,
    )
    assert stabilizer.p.max_bond() <= 2
    stabilizer.p.validate(check_canonical=True)
    diagnostic = stabilizer.get_fit_diagnostics()
    reference = ordinary.get_fit_diagnostics()
    if reference is None:
        assert diagnostic is None
    else:
        for field in ("iterations", "block_size_trace", "fit_init_strategy",
                      "split_method", "convergence_reason"):
            assert diagnostic[field] == reference[field]
        if mode in {"dmrg", "fit"}:
            assert diagnostic["block_size_trace"] == (1, 1, 1, 1)
        elif mode == "dmrg3":
            assert diagnostic["block_size_trace"] == (3, 3, 2, 1)


@pytest.mark.parametrize("mode", ["dmrg", "fit", "dmrg2", "dmrg3"])
def test_tree_stab_uncapped_rotation_matches_dense(mode):
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=25)
    before = state.to_dense().reshape(-1)
    sim = StabilizerTreeSimulator(
        state, tree=plan, mode=mode, chi=None, cutoff=1e-12,
        exact_cooling=False, fit_rtol=None, fit_n_iter=4,
    )
    theta = .43
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    pauli = np.kron(np.kron(np.kron(x, np.eye(2)), np.eye(2)), y)
    expected = np.cos(theta / 2) * before - 1j * np.sin(theta / 2) * pauli @ before
    sim.apply([("rot", theta, "XY", (0, 3))])
    np.testing.assert_allclose(sim.to_statevector(), expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("mode", ["dmrg", "fit"])
def test_tree_stab_generic_dmrg_rejects_larger_blocks(mode):
    with pytest.raises(ValueError, match="fixes fit_block_size=1"):
        StabilizerTreeSimulator(4, mode=mode, fit_block_size=2)


def test_tree_stab_dmrg_keeps_dm_split_separate_from_guess_policy():
    sim = StabilizerTreeSimulator(
        4, mode="dmrg", compression_mode="dm", fit_init_strategy="guess-src",
        fit_init_seed=17, chi=2, fit_n_iter=3, fit_rtol=None,
        exact_cooling=False,
    )
    sim.apply([("rot", .37, "XX", (0, 3))])
    diagnostic = sim.get_fit_diagnostics()
    assert diagnostic["split_method"] == "dm"
    assert diagnostic["fit_init_strategy"] == "guess_src"
    assert diagnostic["block_size_trace"] == (1, 1, 1)


def test_tree_stab_failed_fit_preserves_state_and_retry_queue(monkeypatch):
    from pepsy.fitting import TreeFIT

    sim = StabilizerTreeSimulator(4, mode="dmrg", chi=2, exact_cooling=False)
    before = sim.p.to_dense().copy()

    def fail(*args, **kwargs):
        raise RuntimeError("FIT failed")

    with monkeypatch.context() as patch:
        patch.setattr(TreeFIT, "run_gate", fail)
        with pytest.raises(RuntimeError, match="FIT failed"):
            sim.apply([("rot", .37, "XX", (0, 3))])
    np.testing.assert_allclose(sim.p.to_dense(), before, atol=1e-12)
    assert sim.gate_stream() == (("rot", .37, "XX", (0, 3)),)
    assert sim.get_fit_diagnostics() is None
    sim.run()
    assert sim.tree_optimizer.get_fit_diagnostics() is not None


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_tree_stab_shots_inherit_dmrg_controls_without_changing_parent(strategy):
    sim = StabilizerTreeSimulator(
        4, gates=[("rot", .37, "XX", (0, 3)), ("depolarize1", .2, 0)],
        mode="dmrg2", chi=2, fit_n_iter=3, fit_rtol=None, fit_init_seed=31,
        exact_cooling=False,
    )
    before = sim.p.to_dense().copy()
    queue = sim.gate_stream()
    result = sim.run(shots=8, seed=21, strategy=strategy, workers=1)
    assert result.optimizers
    for child in result.optimizers:
        assert child.mode == "dmrg2"
        assert child.tree_optimizer.fit_n_iter == 3
        assert child.tree_optimizer.fit_init_seed == 31
        assert child.get_fit_diagnostics()["block_size_trace"] == (2, 2, 1)
    np.testing.assert_array_equal(sim.p.to_dense(), before)
    assert sim.gate_stream() == queue
    assert sim.get_fit_diagnostics() is None


def test_tree_stab_cold_threaded_dmrg_shots_finish():
    """First-use Stim matrix conversion must precede concurrent replay."""
    code = textwrap.dedent("""
        from pepsy.optimizers import StabilizerTreeSimulator
        sim = StabilizerTreeSimulator(
            4, gates=[("rot", .37, "XX", (0, 3)), ("depolarize1", .2, 0)],
            mode="dmrg2", chi=2, fit_n_iter=3, fit_rtol=None,
            exact_cooling=False,
        )
        result = sim.run(shots=4, seed=21, strategy="independent", progress=False)
        assert len(result.optimizers) == 4
        for child in result.optimizers:
            assert child.get_fit_diagnostics()["block_size_trace"] == (2, 2, 1)
    """)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=30, check=False, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("rebuild", ["copy", "frame_layout", "cap"])
def test_tree_stab_preserves_live_compression_settings(rebuild):
    sim = StabilizerTreeSimulator(
        4, mode="dmrg3", chi=3, compression_seed=21, fit_n_iter=5,
        fit_rtol=None, fit_patience=2, fit_min_iter=3,
        fit_adaptive_sweeps=1, fit_two_site_transition_sweeps=2,
        fit_init_strategy="guess-direct", fit_init_seed=44,
        fit_sweep_sequence="outward-inward", fit_traversal="depth",
        fit_single_node_fast_path=False, track_infidelity=False,
        stabilize_unitary=True, max_bond_oversample=7,
        max_intermediate_bond=100, subtree_workers=1, record_history=False,
        profile=True, profile_sync=True, track_bond_diagnostics=True,
    )
    expected = sim._coefficient_tree_settings()
    if rebuild == "copy":
        sim = sim.copy()
    elif rebuild == "frame_layout":
        plan = TreePlan.from_order((3, 2, 1, 0), structure="balanced")
        with pytest.warns(UserWarning, match="product"):
            sim.apply_frame_layout(plan)
    else:
        sim.cap(3, [1, 0])
    actual = sim._coefficient_tree_settings()
    for key in expected:
        if key not in {"n", "top_arity"}:
            assert actual[key] == expected[key], key
    assert sim.mode == "dmrg3"
    assert sim.norm_events is sim.tree_optimizer.norm_events
    sim.p.validate(check_canonical=True)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_tree_stab_dmrg_preserves_backend(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        def convert(a):
            return torch.tensor(a, dtype=torch.complex128)
    else:
        jax = pytest.importorskip("jax")
        if not jax.config.x64_enabled:
            pytest.skip("JAX x64 is disabled")
        def convert(a):
            return jax.numpy.asarray(a, dtype=jax.numpy.complex128)
    sim = StabilizerTreeSimulator(
        4, mode="dmrg", chi=2, to_backend=convert, exact_cooling=False,
        fit_init_strategy="guess-direct", fit_rtol=None, fit_n_iter=2,
    )
    sim.apply([("rot", .37, "XX", (0, 3))])
    assert sim.backend_info()["backend"] == backend
    assert sim.get_fit_diagnostics()["block_size_trace"] == (1, 1)
    sim.p.validate(check_canonical=True)
    before_cap = sim.to_statevector().reshape((2,) * 4)
    expected_cap = before_cap[..., 0].reshape(-1)
    sim.cap(3, [1, 0])
    assert sim.backend_info()["backend"] == backend
    np.testing.assert_allclose(sim.to_statevector(), expected_cap, atol=2e-9)
    assert sim.get_fit_diagnostics()["block_size_trace"] == (1, 1)
    sim.p.validate(check_canonical=True)
