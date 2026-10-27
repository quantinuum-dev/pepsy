"""Tests for purified finite-temperature MPS preparation."""

import numpy as np
import pytest
import quimb.tensor as qtn
from scipy.linalg import expm

from pepsy import GibbsMps, bell_to_mps
from pepsy.operators import MPOBasis, MPOParameter, MPOProductTerm
from pepsy.optimizers.mps import gibbs as gibbs_module


def test_bell_to_mps_is_a_quimb_interleaved_identity_purification():
    """The reusable Bell constructor has the same layout as GibbsMps."""
    state = bell_to_mps(2)

    assert state.L == 4
    assert state.cyclic is False
    np.testing.assert_allclose(
        np.asarray(state.partial_trace_to_mpo(keep=(0, 2)).to_dense()),
        np.eye(4) / 4.0,
        atol=1.0e-12,
    )

    cyclic = bell_to_mps(2, cyclic=True)
    assert cyclic.cyclic is True


def test_gibbs_mps_uses_interleaved_bell_pairs_and_traces_ancillas():
    """The beta-zero purification reduces to the normalized identity."""
    state = GibbsMps([(("Z", 0.3), 0)], shape=2)

    assert state.mps.L == 4
    assert state.physical_sites == (0, 2)
    assert state.ancilla_sites == (1, 3)
    assert state.mps.cyclic is False
    assert state.mps.norm() == pytest.approx(1.0)

    dense = np.asarray(state.to_dense())
    np.testing.assert_allclose(dense, np.eye(4) / 4.0, atol=1.0e-12)
    assert state.trace() == pytest.approx(1.0)
    assert state.partition_function() == pytest.approx(4.0)


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_trace_options_keep_the_quimb_mpo_path():
    """Explicit Quimb contraction options do not densify the purification."""
    state = GibbsMps([(("ZZ", 0.3), (0, 1))], shape=2)
    state.prepare(0.2, n_steps=2, cutoff=0.0)

    native = state.raw_mpo
    configured = state.to_mpo(
        normalized=False,
        contract_opts={"optimize": "greedy"},
    )
    np.testing.assert_allclose(
        np.asarray(configured.to_dense()),
        np.asarray(native.to_dense()),
        atol=1.0e-12,
    )
    assert configured.L == state.length


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_uses_quimb_native_trace_with_stored_scale(monkeypatch):
    """Scaled readout stays on Quimb's native partial-trace path."""
    state = GibbsMps([(("ZZ", 0.3), (0, 1))], shape=2)
    state.prepare(0.2, n_steps=1, normalize_every=True, cutoff=0.0)
    calls = []
    native = state.mps.partial_trace_to_mpo

    def traced(*args, **kwargs):
        calls.append((args, kwargs))
        return native(*args, **kwargs)

    monkeypatch.setattr(state.mps, "partial_trace_to_mpo", traced)
    state.to_mpo(normalized=False)

    assert calls
    assert calls[0][1]["keep"] == state.physical_sites


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_reuses_one_resolved_quimb_ordering(monkeypatch):
    """Metadata and execution share the same graph-layer schedule."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("XX", 0.2), (1, 2)),
        ],
        shape=3,
    )
    native = qtn.LocalHamGen.get_trotter_gates
    native_ordering = qtn.LocalHamGen.get_auto_ordering
    seen = []
    ordering_calls = []

    def traced(self, x, **kwargs):
        seen.append(kwargs["ordering"])
        return native(self, x, **kwargs)

    def traced_ordering(self, *args, **kwargs):
        ordering_calls.append((args, kwargs))
        return native_ordering(self, *args, **kwargs)

    monkeypatch.setattr(qtn.LocalHamGen, "get_trotter_gates", traced)
    monkeypatch.setattr(qtn.LocalHamGen, "get_auto_ordering", traced_ordering)
    state.prepare(0.2, n_steps=1, cutoff=0.0)
    state.prepare(0.2, n_steps=1, cutoff=0.0)

    assert seen == [state.trotter_layers, state.trotter_layers]
    assert len(ordering_calls) == 1


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_random_metadata_matches_executable_schedule():
    """Randomized layer metadata and gate replay use one draw."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("XX", 0.2), (1, 2)),
            (("YY", -0.1), (2, 3)),
            (("ZZ", 0.3), (3, 4)),
        ],
        shape=5,
    )
    state.prepare(
        0.2,
        n_steps=1,
        trotter_order=1,
        trotter_ordering="random",
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    assert all(
        trotter_gate.where in state.trotter_layers[trotter_gate.layer]
        for trotter_gate in state.trotter_gates
    )


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_second_order_trotter_matches_small_exact_reference():
    """Second-order imaginary-time replay converges to the exact Gibbs state."""
    terms = [
        (("ZZ", 0.7), (0, 1)),
        (("X", -0.2), 0),
    ]
    state = GibbsMps(terms, shape=2)
    state.prepare(0.4, n_steps=40, chi=32, mode="mpo", cutoff=0.0)

    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = 0.7 * np.kron(z, z) - 0.2 * np.kron(x, np.eye(2))
    exact = expm(-0.4 * hamiltonian)
    exact /= np.trace(exact)

    np.testing.assert_allclose(
        np.asarray(state.to_dense()),
        exact,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert state.optimizer is not None
    assert state.optimizer.norm_diagnostics()["tracking"] is True
    assert state.optimizer._unitary_previous_norm is None


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_energy_matches_dense_reduction_with_auto_cutoff(monkeypatch):
    """Local energy avoids forming an MPO while retaining the prepared H."""
    built_optimizers = []
    native_build_optimizer = gibbs_module.build_optimizer

    def tracked_build_optimizer(**kwargs):
        optimizer = native_build_optimizer(**kwargs)
        built_optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(gibbs_module, "build_optimizer", tracked_build_optimizer)
    terms = [
        (("ZZ", -1.0), (0, 1)),
        (("X", 0.6), 0),
        (("X", 0.4), 0),
        (("X", 1.0), 1),
    ]
    state = GibbsMps(terms, shape=2)
    with pytest.raises(RuntimeError, match="call prepare"):
        state.energy()
    state.prepare(0.4, n_steps=8, chi=16, cutoff="auto", cutoff_mode="auto")

    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    identity = np.eye(2)
    hamiltonian = -np.kron(z, z) + np.kron(x, identity) + np.kron(identity, x)
    dense_energy = np.trace(np.asarray(state.to_dense()) @ hamiltonian)
    native_local_expectation = state.mps.compute_local_expectation
    received_optimizers = []

    def tracked_local_expectation(*args, **kwargs):
        received_optimizers.append(kwargs.get("optimize"))
        return native_local_expectation(*args, **kwargs)

    monkeypatch.setattr(
        state.mps, "compute_local_expectation", tracked_local_expectation,
    )
    assert state.energy() == pytest.approx(dense_energy, abs=1e-11)
    assert len(built_optimizers) == 1
    assert received_optimizers[0] is built_optimizers[0]

    state.prepare(0.0, n_steps=8, chi=16, cutoff="auto", cutoff_mode="auto")
    assert state.n_steps == 0
    assert state.energy() == pytest.approx(0.0, abs=1e-12)
    assert len(built_optimizers) == 1


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_energy_uses_the_latest_bound_coefficient():
    term = MPOProductTerm.from_pauli(
        (0,), "Z", coefficient=MPOParameter("field"),
    )
    state = GibbsMps([term], shape=1)
    for field in (0.3, 0.7):
        state.prepare(
            0.2, n_steps=2, parameters={"field": field},
            cutoff="auto", cutoff_mode="auto",
        )
        assert state.energy() == pytest.approx(
            -field * np.tanh(0.2 * field), abs=1e-12,
        )


@pytest.mark.parametrize("beta,dt,n_steps", [
    (float("nan"), None, None),
    (float("inf"), None, None),
    (0.2, float("nan"), None),
    (0.2, float("inf"), None),
    (0.2, float("nan"), 2),
])
def test_gibbs_mps_rejects_nonfinite_imaginary_time(beta, dt, n_steps):
    state = GibbsMps([(("Z", 0.3), 0)], shape=1)
    with pytest.raises(ValueError, match="finite"):
        state.prepare(beta, dt=dt, n_steps=n_steps)


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_uses_quimb_trotter_schedule_and_physical_mapping():
    """Quimb's schedule metadata survives the purification site mapping."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("XX", 0.2), (1, 2)),
        ],
        shape=3,
    )
    state.prepare(
        0.2,
        n_steps=2,
        trotter_order=2,
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    assert state.trotter_layers == (((0, 1),), ((1, 2),))
    assert len(state.trotter_gates) == 6
    assert all(
        hasattr(gate, attribute)
        for gate in state.trotter_gates
        for attribute in ("frac", "layer", "step")
    )
    assert [gate.step for gate in state.trotter_gates] == [0] * 3 + [1] * 3
    assert all(
        where == tuple(2 * site for site in logical_where)
        for (gate, where), logical_where in zip(
            state.gates,
            (trotter_gate.where for trotter_gate in state.trotter_gates),
        )
    )


@pytest.mark.parametrize("order", (1, 4))
@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_accepts_quimb_first_and_fourth_order_schedules(order):
    """The public order control follows Quimb's supported product formulas."""
    state = GibbsMps([(("ZZ", 0.2), (0, 1))], shape=2)
    state.prepare(0.1, n_steps=1, trotter_order=order, cutoff=0.0)

    assert state.trotter_order == order
    assert state.trotter_gates
    assert all(gate.where == (0, 1) for gate in state.trotter_gates)


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_applies_disconnected_onsite_terms_exactly():
    """One-site terms outside the interaction graph avoid fake edges."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("X", -0.2), 3),
        ],
        shape=4,
    )
    state.prepare(0.4, n_steps=2, cutoff=0.0)

    assert len(state.trotter_gates) == 1
    assert state.gates[-1][1] == (6,)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        np.asarray(state.gates[-1][0]),
        expm(0.2 * 0.2 * x),
        atol=1.0e-12,
    )


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_forwards_direct_mps_controls_and_normalization():
    """Common MpsOptimizer controls are available without nested kwargs."""
    state = GibbsMps([(("ZZ", 0.7), (0, 1))], shape=2)
    state.prepare(
        0.2,
        n_steps=2,
        chi=8,
        mode="direct",
        contraction_opt="auto",
        n_iter=1,
        normalize_every=True,
        normalize_final=True,
        cutoff=0.0,
    )

    assert state.optimizer.mode == "quimb-direct"
    assert state.optimizer.get_normalizations()
    assert np.isfinite(float(np.real(state.trace())))


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_tracks_log_partition_function_through_rescaling():
    """Log-Z and normalized readout retain scale-control bookkeeping."""
    terms = [
        (("ZZ", 0.7), (0, 1)),
        (("X", -0.2), 0),
    ]
    state = GibbsMps(terms, shape=2)
    state.prepare(
        0.4,
        n_steps=1,
        chi=32,
        cutoff=0.0,
        normalize_every=True,
    )

    z = 4.170801899580418
    assert state.trace() == pytest.approx(z / 4.0)
    assert state.partition_function() == pytest.approx(z)
    assert state.log_partition_function() == pytest.approx(np.log(z))
    np.testing.assert_allclose(
        np.asarray(state.to_mpo().to_dense()),
        np.asarray(state.raw_mpo.to_dense()) / (z / 4.0),
        atol=1.0e-12,
    )
    assert np.trace(np.asarray(state.to_mpo().to_dense())) == pytest.approx(1.0)


def test_gibbs_mps_accepts_lattice_terms_through_one_d_map():
    """Regular-lattice terms use the same OneDMap traversal as MPOBasis."""
    terms = [
        {"operator": "Z", "location": (0, 0), "coefficient": 0.2},
        {"operator": "ZZ", "location": ((0, 0), (1, 0)), "coefficient": 0.4},
    ]
    state = GibbsMps(terms, shape=(2, 2), map_mode="row-major")

    expected_basis = MPOBasis.from_terms(terms, shape=(2, 2), map_mode="row-major")
    assert state.length == 4
    assert state.mapper.mode == "row-major"
    assert state.basis.lattice_to_chain == expected_basis.lattice_to_chain
    assert state.basis.terms[1].sites == (
        expected_basis.lattice_to_chain[(0, 0)],
        expected_basis.lattice_to_chain[(1, 0)],
    )


def test_gibbs_mps_infers_coordinate_dimension_and_shape():
    """Natural 2D coordinates work without an explicit shape argument."""
    lx, ly = 3, 4
    edges = qtn.edges_2d_square(lx, ly, cyclic=True)
    sites = sorted({site for edge in edges for site in edge})
    terms = [(("zz", 0.3), (u, v)) for (u, v) in edges]
    terms += [(("x", -0.1), site) for site in sites]

    state = GibbsMps(terms, map_mode="row-major")

    assert state.shape == (lx, ly)
    assert state.length == lx * ly
    assert state.basis.lattice_shape == (lx, ly)
    assert state.mapper.mode == "row-major"
    assert state.basis.lattice_to_chain[(1, 0)] == 4


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_infers_triangular_coordinate_graph():
    """Graph geometry comes from the supplied triangular edge list."""
    lx, ly = 2, 3
    edges = qtn.edges_2d_triangular(lx, ly, cyclic=True)
    sites = sorted({site for edge in edges for site in edge})
    terms = [(("zz", 0.3), (u, v)) for (u, v) in edges]
    terms += [(("x", -0.1), site) for site in sites]

    state = GibbsMps(terms, map_mode="snake")
    state.prepare(
        beta=0.05,
        n_steps=1,
        trotter_order=1,
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    assert state.shape == (lx, ly)
    assert state.length == lx * ly
    assert len(state._trotter_ham.terms) == len(set(edges))
    assert state.trotter_gates


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_fuses_connected_onsite_terms_into_pair_terms():
    """Onsite terms are combined before the Quimb Trotter schedule."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("X", 0.2), 0),
            (("Z", -0.1), 0),
            (("X", 0.3), 1),
        ],
        shape=2,
    )
    state.prepare(
        beta=0.2,
        n_steps=1,
        trotter_order=1,
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    expected = (
        0.7 * np.kron(z, z)
        + 0.2 * np.kron(x, identity)
        - 0.1 * np.kron(z, identity)
        + 0.3 * np.kron(identity, x)
    )
    np.testing.assert_allclose(
        np.asarray(state._trotter_ham.get_gate((0, 1))),
        expected,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state.trotter_gates[0].U),
        expm(-0.1 * expected),
        atol=1.0e-12,
    )
    assert len(state.trotter_gates) == 1
    assert len(state.gates) == 1


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_defaults_to_direct_replay_mode():
    """The public default uses the explicit direct-mode spelling."""
    state = GibbsMps([(("ZZ", 0.2), (0, 1))], shape=2)
    state.prepare(beta=0.1, n_steps=1, cutoff=0.0)

    assert state.optimizer.mode == "quimb-direct"


@pytest.mark.parametrize("options,beta,expected_steps", [
    ({}, 0.4, 8),
    ({"dt": None}, 0.466, 10),
    ({"beta_step": 0.1}, 0.4, 4),
    ({"dt": 0.05}, 0.4, 4),
])
@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_default_second_order_step_bound(options, beta, expected_steps):
    """Thermal beta increments and legacy purification-time steps are explicit."""
    state = GibbsMps([(("ZZ", 0.2), (0, 1))], shape=2)
    state.prepare(beta, cutoff=0.0, **options)

    assert state.trotter_order == 2
    assert state.n_steps == expected_steps
    assert float(state.trotter_step) == pytest.approx(beta / (2 * expected_steps))
    assert float(state.beta_step) == pytest.approx(beta / expected_steps)

    state.prepare(beta, n_steps=2, cutoff=0.0, **options)
    assert state.n_steps == 2
    assert float(state.trotter_step) == pytest.approx(beta / 4)


@pytest.mark.parametrize("beta_step", [0.0, -0.05, float("nan"), float("inf")])
def test_gibbs_mps_rejects_invalid_beta_step(beta_step):
    state = GibbsMps([(("Z", 0.3), 0)], shape=1)
    with pytest.raises(ValueError, match="beta_step must be finite and positive"):
        state.prepare(0.2, beta_step=beta_step)


@pytest.mark.parametrize("mode", ["direct", "dmrg"])
@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_evolve_measures_intermediate_betas_without_restarting(mode):
    """Every live snapshot follows the same cumulative thermal evolution."""
    state = GibbsMps([(("ZZ", 0.7), (0, 1)), (("X", 0.3), 0)], shape=2)
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = 0.7 * np.kron(z, z) + 0.3 * np.kron(x, np.eye(2))
    betas, optimizers = [], []
    for snapshot in state.evolve(
        0.2, mode=mode, chi=8, cutoff=0.0, normalize_every=True, measure_every=1,
    ):
        beta = float(snapshot.beta)
        betas.append(beta)
        optimizers.append(snapshot.optimizer)
        rho = expm(-beta * hamiltonian)
        partition = np.trace(rho)
        expected_energy = np.trace(hamiltonian @ rho) / partition
        energy = snapshot.energy()
        assert np.imag(energy) == pytest.approx(0.0, abs=1e-12)
        assert float(np.real(energy)) == pytest.approx(expected_energy, abs=1e-11)
        assert float(snapshot.log_partition_function()) == pytest.approx(
            np.log(partition), abs=1e-11,
        )
        assert snapshot.n_steps == len(betas) - 1
        assert snapshot.optimizer.norm_diagnostics()["local_infidelity"] is None
        if snapshot.n_steps:
            assert snapshot.local_infidelity == pytest.approx(0.0, abs=1e-11)
        else:
            assert snapshot.local_infidelity is None

    assert betas == pytest.approx([0.0, 0.05, 0.1, 0.15, 0.2])
    assert all(optimizer is optimizers[0] for optimizer in optimizers)
    assert float(state.beta_step) == pytest.approx(0.05)
    assert float(state.trotter_step) == pytest.approx(0.025)


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_default_measurements_reuse_optimizer_and_known_center(monkeypatch):
    state = GibbsMps([(("ZZ", 0.7), (0, 1)), (("X", 0.3), 0)], shape=2)
    built = []
    original_build = gibbs_module.build_optimizer

    def tracked_build(**kwargs):
        assert kwargs == {"progbar": False, "parallel": False}
        result = original_build(**kwargs)
        built.append(result)
        return result

    monkeypatch.setattr(gibbs_module, "build_optimizer", tracked_build)
    original_expectation = qtn.MatrixProductState.compute_local_expectation
    received_terms = []

    def checked_expectation(mps, terms, **kwargs):
        assert kwargs["optimize"] is built[0]
        assert kwargs["info"] == state.optimizer.info_c
        assert kwargs["info"] is not state.optimizer.info_c
        assert kwargs["inplace"] is False
        received_terms.append(terms)
        return original_expectation(mps, terms, **kwargs)

    def fail_center_scan(*args, **kwargs):
        pytest.fail("energy should reuse the tracked canonical center")

    readings = []
    h = 0.7 * np.kron(np.diag([1., -1.]), np.diag([1., -1.]))
    h += 0.3 * np.kron(np.array([[0., 1.], [1., 0.]]), np.eye(2))
    for snapshot in state.evolve(0.6, chi=8, cutoff=0.0):
        before_info = dict(snapshot.optimizer.info_c)
        before_arrays = [t.data.copy() for t in snapshot.mps.tensors]
        with monkeypatch.context() as patch:
            patch.setattr(qtn.MatrixProductState, "compute_local_expectation", checked_expectation)
            patch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", fail_center_scan)
            energy = snapshot.energy()
        assert snapshot.optimizer.info_c == before_info
        for tensor, before in zip(snapshot.mps.tensors, before_arrays):
            np.testing.assert_array_equal(tensor.data, before)
        exact = expm(-float(snapshot.beta) * h)
        assert energy == pytest.approx(np.trace(h @ exact) / np.trace(exact), abs=2e-12)
        readings.append((float(snapshot.beta), snapshot.n_steps))
    assert np.asarray(readings) == pytest.approx(np.asarray([(0., 0), (.25, 5), (.5, 10), (.6, 12)]))
    assert len(built) == 1
    assert all(terms is received_terms[0] for terms in received_terms)


@pytest.mark.usefixtures("quimb_trotter")
@pytest.mark.parametrize("route", ["canonical", "envs", "exact"])
def test_gibbs_shared_measurement_uses_physical_sites_and_current_beta(route):
    z = np.diag([1., -1.])
    x = np.array([[0., 1.], [1., 0.]])
    h = 0.7 * np.kron(z, z) + 0.3 * np.kron(x, np.eye(2))
    gibbs = GibbsMps([(("ZZ", 0.7), (0, 1)), (("X", 0.3), 0)], shape=2)
    evaluators = []
    for state in gibbs.evolve(0.6, chi=8, cutoff=0.):
        result = state.measure({"x0": {(0,): x}, "zz": {(0, 1): np.kron(z, z)}}, route=route)
        evaluators.append(state._energy_evaluator)
        rho = expm(-float(state.beta) * h)
        rho /= np.trace(rho)
        assert result["energy"] == pytest.approx(np.trace(rho @ h), abs=2e-12)
        assert result["observables"]["x0"] == pytest.approx(np.trace(rho @ np.kron(x, np.eye(2))), abs=2e-12)
        assert result["observables"]["zz"] == pytest.approx(np.trace(rho @ np.kron(z, z)), abs=2e-12)
        assert result["metadata"]["num_supports"] == 2
    assert all(evaluator is evaluators[0] for evaluator in evaluators)


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_evolve_measurement_intervals_match_dense_and_fused_prepare(monkeypatch):
    terms = [(("ZZ", 0.7), (0, 1)), (("X", 0.3), 1)]
    state = GibbsMps(terms, shape=2)
    h = 0.7 * np.kron(np.diag([1., -1.]), np.diag([1., -1.]))
    h += 0.3 * np.kron(np.eye(2), np.array([[0., 1.], [1., 0.]]))
    # Measurement is explicit: iteration must never invoke energy itself.
    monkeypatch.setattr(state, "energy", lambda: pytest.fail("implicit energy measurement"))
    observations, identities = [], []
    for snapshot in state.evolve(0.23, measure_at=[0.07, 0.16], chi=8, cutoff=0.0):
        observations.append(float(snapshot.beta))
        identities.append(snapshot.optimizer)
        exact = expm(-float(snapshot.beta) * h)
        np.testing.assert_allclose(snapshot.to_dense(), exact / np.trace(exact), atol=2e-12)
        if snapshot.n_steps:
            assert float(snapshot.beta_step) <= 0.05
    assert observations == pytest.approx([0.0, 0.07, 0.16, 0.23])
    assert all(opt is identities[0] for opt in identities)

    sparse = GibbsMps(terms, shape=2)
    steps = [(float(s.beta), s.n_steps) for s in sparse.evolve(
        0.25, n_steps=5, measure_every=2, chi=8, cutoff=0.0,
    )]
    assert np.asarray(steps) == pytest.approx(np.asarray([(0., 0), (.1, 2), (.2, 4), (.25, 5)]))
    fused = GibbsMps(terms, shape=2).prepare(0.25, n_steps=5, chi=8, cutoff=0.0)
    np.testing.assert_allclose(sparse.to_dense(), fused.to_dense(), atol=2e-12)


@pytest.mark.parametrize("options", [
    {"measure_every": 0}, {"measure_every": True},
    {"measure_at": [0.2, 0.1]}, {"measure_at": [0.1, 0.1]},
    {"measure_at": [-0.1]}, {"measure_at": [float("nan")]},
    {"measure_at": [0.3]}, {"measure_at": [0.1], "n_steps": 4},
    {"measure_at": [0.1], "measure_every": 2},
])
def test_gibbs_evolve_rejects_invalid_measurement_schedule_before_preparing(options):
    state = GibbsMps([(("ZZ", 0.7), (0, 1))], shape=2)
    with pytest.raises((ValueError, TypeError)):
        list(state.evolve(0.2, **options))
    assert state.optimizer is None


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_evolve_final_only_preserves_fused_finite_chi_schedule():
    terms = [(("ZZ", 0.7), (0, 1)), (("XX", 0.4), (1, 2)), (("Z", 0.2), 3)]
    state = GibbsMps(terms, shape=4)
    outputs = [float(s.beta) for s in state.evolve(
        0.3, n_steps=3, measure_every=10, chi=2, cutoff=0.0,
    )]
    reference = GibbsMps(terms, shape=4).prepare(0.3, n_steps=3, chi=2, cutoff=0.0)
    assert outputs == pytest.approx([0., 0.3])
    np.testing.assert_allclose(state.to_dense(), reference.to_dense(), atol=2e-12)
    assert float(state.log_partition_function()) == pytest.approx(
        float(reference.log_partition_function()), abs=2e-12,
    )


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_evolve_preserves_complete_second_order_steps_and_isolated_terms():
    terms = [
        (("ZZ", 0.7), (0, 1)),
        (("XX", 0.2), (1, 2)),
        (("X", -0.3), 3),
    ]
    continuous = GibbsMps(terms, shape=4)
    for snapshot in continuous.evolve(0.17, n_steps=3, chi=32, cutoff=0.0, measure_every=1):
        if snapshot.n_steps:
            assert {gate.step for gate in snapshot.trotter_gates} == {
                snapshot.n_steps - 1,
            }
    independent = GibbsMps(terms, shape=4)
    independent.prepare(
        0.17, n_steps=3, chi=32, cutoff=0.0, trotter_fuse_adjacent=False,
    )
    np.testing.assert_allclose(
        np.asarray(continuous.to_dense()), np.asarray(independent.to_dense()),
        atol=1e-11,
    )
    assert continuous.log_partition_function() == pytest.approx(
        independent.log_partition_function(), abs=1e-11,
    )
    assert [float(state.beta) for state in continuous.evolve(0.0)] == [0.0]
    np.testing.assert_allclose(np.asarray(continuous.to_dense()), np.eye(16) / 16)


def test_gibbs_mps_infers_flat_integer_chain_shape():
    """Flat integer locations remain an inferred one-dimensional chain."""
    state = GibbsMps(
        [
            (("zz", 0.3), (0, 1)),
            (("x", -0.1), 0),
        ]
    )

    assert state.shape == (2,)
    assert state.length == 2
    assert state.basis.lattice_shape is None


def test_gibbs_mps_materializes_term_generators_once():
    """Generator-based term input remains available for backend inference."""
    terms = ((("Z", 0.2), site) for site in range(2))
    state = GibbsMps(terms, shape=2)

    assert state.basis.num_terms == 2
    assert state.mps.L == 4


def test_gibbs_mps_rejects_unsupported_multisite_and_string_gap_terms():
    """The first step reports unsupported gate arities explicitly."""
    with pytest.raises(NotImplementedError, match="only one- and two-site"):
        GibbsMps([(("XXX", 0.1), (0, 1, 2))], shape=3)

    with pytest.raises(NotImplementedError, match="string operators across"):
        GibbsMps(
            [
                {
                    "operator": "XX",
                    "location": (0, 2),
                    "coefficient": 0.1,
                    "string_operators": ("Z",),
                }
            ],
            shape=3,
        )


def test_gibbs_mps_rejects_layout_override_during_replay():
    """Ancilla positions remain meaningful only without MPS reordering."""
    state = GibbsMps([(("ZZ", 0.1), (0, 1))], shape=2)

    with pytest.raises(TypeError, match="layout"):
        state.prepare(0.1, n_steps=1, run_kwargs={"layout": "quality"})


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_keeps_identity_snapshot_with_inplace_optimizer():
    """The identity accessor remains un evolved even for in-place replay."""
    state = GibbsMps([(("Z", 0.1), 0)], shape=1)
    state.prepare(
        0.2,
        n_steps=1,
        optimizer_kwargs={"inplace": True},
        cutoff=0.0,
    )

    np.testing.assert_allclose(
        np.asarray(
            state.identity_mps.partial_trace_to_mpo(
                keep=state.physical_sites,
                upper_ind_id="b{}",
            ).to_dense()
        ),
        np.eye(2) / 2.0,
        atol=1.0e-12,
    )


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_keeps_explicit_backend_for_state_and_gates():
    """Explicit ``to_backend`` reaches Bell tensors, gates, and the MPO."""
    torch = pytest.importorskip("torch")
    import pepsy

    backend = pepsy.backend_torch(dtype=torch.complex128, device="cpu")
    state = GibbsMps(
        [(("ZZ", 0.4), (0, 1)), (("X", -0.1), 0)],
        shape=2,
        to_backend=backend,
    )
    state.prepare(0.2, n_steps=2, chi=16, cutoff=0.0)

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in state.mps)
    assert all(isinstance(gate, torch.Tensor) for gate, _where in state.gates)
    energy = state.energy()
    assert isinstance(energy, torch.Tensor)
    assert energy.device.type == "cpu"
    assert energy.dtype == torch.complex128
    rho = state.to_mpo(normalized=False)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in rho)


@pytest.mark.parametrize("trajectory", [False, True, "measure_at", "measure_at_endpoint", "sparse"])
@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_preserves_autograd_gate_payloads(trajectory):
    """Backend placement must not detach differentiable generated gates."""
    torch = pytest.importorskip("torch")
    import pepsy

    beta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    backend = pepsy.backend_torch(dtype=torch.complex128, device="cpu")
    state = GibbsMps(
        [(("Z", coefficient), 0)],
        shape=1,
        to_backend=backend,
    )
    if trajectory in {"measure_at", "measure_at_endpoint"}:
        targets = [0.07, 0.13, 0.2] if trajectory == "measure_at_endpoint" else [0.07, 0.13]
        for _snapshot in state.evolve(beta, measure_at=targets, chi=8, cutoff=0.0):
            pass
        assert state.beta is beta
    elif trajectory == "sparse":
        for _snapshot in state.evolve(beta, n_steps=5, measure_every=2, chi=8, cutoff=0.0):
            pass
    elif trajectory:
        for _snapshot in state.evolve(beta, n_steps=2, chi=8, cutoff=0.0):
            pass
    else:
        state.prepare(beta, n_steps=2, chi=8, cutoff=0.0)
    loss = torch.real(state.to_mpo(normalized=False).to_dense()[0, 0])
    grad_beta, grad_coefficient = torch.autograd.grad(
        loss,
        (beta, coefficient),
        retain_graph=True,
    )

    assert torch.isfinite(grad_beta)
    assert torch.isfinite(grad_coefficient)

    thermal_energy = torch.real(state.energy())
    grad_energy_beta, grad_energy_coefficient = torch.autograd.grad(
        thermal_energy,
        (beta, coefficient),
    )
    expected_energy = -coefficient.detach() * torch.tanh(
        beta.detach() * coefficient.detach()
    )
    assert thermal_energy.item() == pytest.approx(expected_energy.item(), abs=1e-10)
    u = beta.detach() * coefficient.detach()
    sech_squared = 1.0 / torch.cosh(u) ** 2
    assert grad_energy_beta.item() == pytest.approx(
        (-coefficient.detach() ** 2 * sech_squared).item(), abs=1e-10,
    )
    assert grad_energy_coefficient.item() == pytest.approx(
        (-torch.tanh(u) - u * sech_squared).item(), abs=1e-9,
    )


@pytest.mark.usefixtures("quimb_trotter")
def test_gibbs_mps_jax_gate_schedule_preserves_backend_arrays():
    """Unhashable JAX exponents use the native schedule plus Autoray gates."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import pepsy

    backend = pepsy.backend_jax(dtype=jnp.complex64)
    state = GibbsMps(
        [
            (("ZZ", jnp.asarray(0.4)), (0, 1)),
            (("X", jnp.asarray(-0.1)), 0),
        ],
        shape=2,
        to_backend=backend,
    )
    state.prepare(jnp.asarray(0.2), n_steps=2, cutoff=0.0)

    assert isinstance(state.trotter_gates[0].U, jax.Array)
    assert isinstance(state.gates[0][0], jax.Array)
