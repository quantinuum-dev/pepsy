"""Exact native measurement routing and frame-consistent collapse."""

from copy import deepcopy

import numpy as np
import pytest

from pepsy import build_backend
from pepsy.optimizers import StabilizerMpsSimulator, compile_stim_circuit, stim_plan_to_gate_stream
from pepsy.optimizers.stabilizer_tn import _tableau_measurement as native

stim = pytest.importorskip("stim")


@pytest.fixture(params=["numpy", "torch"])
def converter(request):
    if request.param == "numpy":
        return None
    torch = pytest.importorskip("torch")
    return build_backend(device="cpu", dtype=torch.complex128, set_default=False)


def _same_state(actual, expected):
    actual = np.asarray(actual).ravel()
    expected = np.asarray(expected).ravel()
    assert np.linalg.norm(actual) == pytest.approx(1., abs=1e-12)
    assert abs(np.vdot(actual, expected))**2 == pytest.approx(1., abs=1e-12)


def _project(vector, pauli, site, outcome, n):
    matrix = stim.PauliString(pauli).to_unitary_matrix(endian="big")
    block = vector.reshape((2,) * n)
    order = (site,) + tuple(i for i in range(n) if i != site)
    selected = block.transpose(order).reshape(2, -1)
    selected = (selected + outcome * matrix @ selected) / 2
    selected = selected.reshape((2,) * n).transpose(np.argsort(order)).ravel()
    return selected / np.linalg.norm(selected)


@pytest.mark.parametrize("strategy", ["independent", "coalesced", "auto"])
def test_clifford_feedback_reset_stream_never_uses_coefficient_projectors(converter, strategy, monkeypatch):
    plan = compile_stim_circuit("H 0\nCX 0 1\nMR 0\nCX rec[-1] 1\nRX 2\nM 1 2")
    stream = stim_plan_to_gate_stream(plan)
    if converter:
        def convert_entry(entry):
            if isinstance(entry[0], np.ndarray):
                return (converter(entry[0].copy()), *entry[1:])
            if entry[0] == "if":
                return (*entry[:3], convert_entry(entry[3]))
            return entry
        stream = tuple(map(convert_entry, stream))
    sim = StabilizerMpsSimulator(3, chi=1, to_backend=converter).compile(stream)

    def expensive(*args, **kwargs):
        pytest.fail("pure Clifford trajectory used a coefficient measurement contraction")

    monkeypatch.setattr(StabilizerMpsSimulator, "_pauli_expectation", expensive)
    monkeypatch.setattr(StabilizerMpsSimulator, "_apply_projector", expensive)
    monkeypatch.setattr(StabilizerMpsSimulator, "_apply_localizer_to_p", expensive)
    result = sim.run(shots=16, strategy=strategy, workers=2, seed=13, progress=False)
    assert sum(result.counts) == 16
    for optimizer in result.optimizers:
        outcomes = [record.outcome for record in optimizer.measurements]
        assert len(outcomes) == 3
        ref = stim.TableauSimulator()
        ref.set_num_qubits(3)
        ref.h(0)
        ref.cnot(0, 1)
        ref.postselect_observable(stim.PauliString("Z__"), desired_value=outcomes[0] < 0)
        ref.reset(0)
        if outcomes[0] < 0:
            ref.x(1)
        ref.reset_x(2)
        ref.postselect_observable(stim.PauliString("_Z_"), desired_value=outcomes[1] < 0)
        ref.postselect_observable(stim.PauliString("__Z"), desired_value=outcomes[2] < 0)
        _same_state(optimizer.to_statevector(), ref.state_vector(endian="big"))
        assert optimizer.p.max_bond() == 1
        assert all(event.measurement_backend == "stim" for event in optimizer.norm_events)


def test_compilation_does_not_measure_or_consume_rng():
    sim = StabilizerMpsSimulator(2, seed=9)
    before = deepcopy(sim._rng.bit_generator.state)
    basis = sim.state._sim.current_inverse_tableau()
    sim.compile([("h", 0), ("measure", "Z", 0)])
    assert sim._rng.bit_generator.state == before
    assert sim.state._sim.current_inverse_tableau() == basis
    assert sim.measurements == []


def test_native_measurement_avoids_finite_chi_projector_loss_and_explicit_false_preserves_basis():
    sim = StabilizerMpsSimulator(2, chi=1)
    sim.measure("XX", (0, 1), outcome=1)
    _same_state(sim.to_statevector(), np.array([1., 0., 0., 1.]) / 2**.5)
    assert sim.norm_events[-1].measurement_backend == "stim"
    assert sim.norm_events[-1].projector_infidelity == pytest.approx(0., abs=1e-12)
    fixed = StabilizerMpsSimulator(2, chi=1)
    before = fixed.state._sim.current_inverse_tableau()
    fixed.measure("XX", (0, 1), outcome=1, absorb_basis=False)
    assert fixed.state._sim.current_inverse_tableau() == before
    assert fixed.norm_events[-1].measurement_backend == "mps"
    assert fixed.norm_events[-1].projector_infidelity > 0


def test_native_forced_impossible_is_atomic_and_flags_are_validated():
    sim = StabilizerMpsSimulator(2).apply([("h", 0), ("cnot", 0, 1)])
    sim.measure("Z", 0, outcome=1)
    state = sim.to_statevector()
    basis = sim.state._sim.current_inverse_tableau()
    rng = deepcopy(sim._rng.bit_generator.state)
    records = deepcopy(sim.measurements)
    events = deepcopy(sim.norm_events)
    with pytest.raises(ValueError, match="0 probability"):
        sim.measure("Z", 1, outcome=-1)
    np.testing.assert_array_equal(sim.to_statevector(), state)
    assert sim.state._sim.current_inverse_tableau() == basis
    assert sim._rng.bit_generator.state == rng
    assert sim.measurements == records
    assert sim.norm_events == events
    with pytest.raises(ValueError):
        sim.measure("Z", 1, absorb_basis=True, disentangle=False)


def test_coefficient_bit_rebasing_preserves_layout_phase_and_norm(converter):
    sim = StabilizerMpsSimulator(3, to_backend=converter).apply_layout((2, 0, 1), layout_report=False)
    site = sim._mps_site(0)
    data = sim.p[site].data
    # Flip a coefficient bit externally, then let the runtime recertify it.
    import autoray as ar
    axis = sim.p[site].inds.index(sim.p.site_ind(site))
    sim.p[site].modify(data=2j * ar.do("flip", data, axis=axis))
    before = sim.to_statevector() / 2
    sim.measure("Z", 0, outcome=-1)
    _same_state(sim.to_statevector(), before)
    assert native.computational_coefficient_bits(sim) == (False, False, False)
    assert sim.norm_events[-1].pre_norm == pytest.approx(2.)
    assert sim.norm_events[-1].projector_survival == pytest.approx(1.)
    assert sim.logical_order == [2, 0, 1]


def test_t_gate_fallback_and_harmless_t_remains_native(converter):
    sim = StabilizerMpsSimulator(1, to_backend=converter).apply([("h", 0), ("t", 0)])
    expected = _project(sim.to_statevector(), "Z", 0, -1, 1)
    sim.measure("Z", 0, outcome=-1)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "mps"
    # A numerical projector need not leave an exactly computational frame.
    # Re-entry requires an exact certificate, never a tolerance-based guess.
    harmless = StabilizerMpsSimulator(1, to_backend=converter).apply([("t", 0)])
    harmless.measure("Z", 0)
    assert harmless.norm_events[-1].measurement_backend == "stim"


def test_identity_measurement_preserves_scale_and_norm_history():
    sim = StabilizerMpsSimulator(1)
    sim.p[0].modify(data=2 * sim.p[0].data)
    before = sim.to_statevector()
    assert sim.measure("I", 0, outcome=1) == 1
    np.testing.assert_array_equal(sim.to_statevector(), before)
    assert sim.norm_events == []


def test_basis_updating_magic_measurement_can_return_to_native(converter):
    sim = StabilizerMpsSimulator(1, to_backend=converter).apply([("h", 0), ("t", 0)])
    expected = _project(sim.to_statevector(), "Z", 0, -1, 1)
    sim.measure("Z", 0, outcome=-1, disentangle=True)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "mps"
    assert native.computational_coefficient_bits(sim) == (True,)
    sim.measure("Z", 0, outcome=-1)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "stim"


def test_partial_tableau_measurement_preserves_entangled_magic_complement(converter, monkeypatch):
    sim = StabilizerMpsSimulator(4, chi=4, exact_cooling=False, to_backend=converter)
    sim.apply([("h", 0), ("t", 0), ("rxx", .41, 0, 1)])
    assert native.certified_tableau(sim) is None
    before = sim.to_statevector()
    expected = _project(before, "X", 3, -1, 4)
    original = sim._pauli_expectation
    monkeypatch.setattr(sim, "_pauli_expectation", lambda *a: pytest.fail("contracted magic complement"))
    sim.measure("X", 3, outcome=-1)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "stim_region"
    monkeypatch.setattr(sim, "_pauli_expectation", original)
    before = sim.to_statevector()
    expected = _project(before, "X", 0, 1, 4)
    sim.measure("X", 0, outcome=1)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "mps"


def test_small_magic_amplitude_and_external_edits_are_never_certified():
    sim = StabilizerMpsSimulator(1)
    data = sim.p[0].data.copy()
    data.reshape(-1)[1] = 1e-20
    sim.p[0].modify(data=data)
    assert native.certified_tableau(sim) is None
    assert native.probabilities(sim, stim.PauliString("X")) is None


def test_regional_joint_collapse_with_nontrivial_basis_and_coefficient_bit(converter):
    sim = StabilizerMpsSimulator(5, chi=8, exact_cooling=False, to_backend=converter)
    sim.apply([("h", 0), ("t", 0), ("rxx", .31, 0, 1), ("h", 3), ("cnot", 3, 4)])
    import autoray as ar
    tensor = sim.p[sim._mps_site(2)]
    axis = tensor.inds.index(sim.p.site_ind(sim._mps_site(2)))
    tensor.modify(data=ar.do("flip", tensor.data, axis=axis))
    before = sim.to_statevector()
    operator = stim.PauliString("__X_X").to_unitary_matrix(endian="big")
    expected = (before - operator @ before) / 2
    expected /= np.linalg.norm(expected)
    sim.measure("XX", (2, 4), outcome=-1)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "stim_region"
    assert sim.norm_events[-1].projector_infidelity == pytest.approx(0., abs=1e-12)


def test_trainable_zero_amplitudes_take_general_path():
    torch = pytest.importorskip("torch")
    convert = build_backend(device="cpu", dtype=torch.complex128, set_default=False)
    sim = StabilizerMpsSimulator(1, to_backend=convert)
    sim.p[0].data.requires_grad_(True)
    assert native.certified_tableau(sim) is None
    assert native.probabilities(sim, stim.PauliString("Z")) is None


_EIGENVECTORS = {
    "Z+": (1., 0.), "Z-": (0., 1.), "X+": (1., 1.),
    "X-": (1., -1.), "Y+": (1., 1j), "Y-": (1., -1j),
}


def _set_coefficient_eigenstate(sim, logical, label):
    tensor = sim.p[sim._mps_site(logical)]
    values = np.array(_EIGENVECTORS[label], dtype=complex)
    values *= (1 + 2j) / np.linalg.norm(values)
    from pepsy.optimizers.stabilizer_tn._backend import array_namespace
    xp = array_namespace(tensor.data)
    # Use the live converter explicitly to preserve the engine's backend.
    if sim.to_backend is not None:
        values = sim.to_backend(values)
    tensor.modify(data=xp.reshape(values, tensor.shape))
    sim.state.info["cur_orthog"] = None


@pytest.mark.parametrize("label", tuple(_EIGENVECTORS))
@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_all_exact_pauli_eigenstates_collapse_without_mps_projectors(converter, label, axis, monkeypatch):
    sim = StabilizerMpsSimulator(3, to_backend=converter).apply_layout((2, 0, 1), layout_report=False)
    sim.apply([("h", 0), ("cnot", 0, 2)])
    _set_coefficient_eigenstate(sim, 1, label)
    before = sim.to_statevector()
    before /= np.linalg.norm(before)
    observable = stim.PauliString("_" + axis + "_")
    operator = observable.to_unitary_matrix(endian="big")
    plus_probability = np.linalg.norm((before + operator @ before) / 2)**2
    outcome = 1 if plus_probability > .1 else -1
    expected = _project(before, axis, 1, outcome, 3)
    monkeypatch.setattr(sim, "_apply_projector", lambda *a, **k: pytest.fail("used MPS projector"))
    weights = native.probabilities(sim, observable)
    assert weights[0] == pytest.approx(plus_probability, abs=1e-12)
    sim.measure(axis, 1, outcome=outcome)
    _same_state(sim.to_statevector(), expected)
    assert native.computational_coefficient_bits(sim) == (False,) * 3
    assert sim.norm_events[-1].measurement_backend == "stim"
    assert sim.logical_order == [2, 0, 1]


@pytest.mark.parametrize("labels", [("X+", "Y-"), ("X-", "Y+"), ("Y+", "X-"), ("Y-", "X+")])
def test_regional_xy_eigenstates_preserve_entangled_magic_state(converter, labels):
    sim = StabilizerMpsSimulator(5, chi=8, exact_cooling=False, to_backend=converter)
    sim.apply([("h", 0), ("t", 0), ("rxx", .31, 0, 1), ("h", 3), ("cnot", 3, 4)])
    _set_coefficient_eigenstate(sim, 3, labels[0])
    _set_coefficient_eigenstate(sim, 4, labels[1])
    frame = stim.PauliString("___ZX")
    physical = sim.state._sim.current_inverse_tableau().inverse()(frame)
    before = sim.to_statevector()
    before /= np.linalg.norm(before)
    operator = physical.to_unitary_matrix(endian="big")
    expected = (before - operator @ before) / 2
    expected /= np.linalg.norm(expected)
    sim.measure(str(physical)[1:].replace("_", "I"), tuple(range(5)), outcome=-1)
    _same_state(sim.to_statevector(), expected)
    assert sim.norm_events[-1].measurement_backend == "stim_region"


@pytest.mark.parametrize("label", ["X+", "X-", "Y+", "Y-"])
def test_nearly_pauli_eigenstates_are_not_certified(label):
    sim = StabilizerMpsSimulator(1)
    data = np.asarray(_EIGENVECTORS[label], dtype=complex)
    data[1] += 1e-12
    sim.p[0].modify(data=data)
    assert native.certified_tableau(sim) is None


def test_tiny_imaginary_perturbation_of_x_eigenstate_is_not_certified():
    sim = StabilizerMpsSimulator(1)
    sim.p[0].modify(data=np.array([1., 1. + 1e-20j]))
    assert native.certified_tableau(sim) is None


def test_routing_diagnostics_report_committed_events_and_survive_copy_rollback():
    sim = StabilizerMpsSimulator(4, chi=4, exact_cooling=False)
    sim.apply([("h", 0), ("t", 0), ("rxx", .31, 0, 1)])
    sim.measure("X", 3, outcome=1)
    sim.measure("X", 0, outcome=1)
    sim.measure("Z", 2, outcome=1, disentangle=False)
    expected = {
        "counts": {"stim": 0, "stim_region": 1, "mps": 2},
        "fallback_reasons": {"uncertified_entanglement": 1, "explicit_fixed_basis": 1},
        "total": 3,
    }
    assert sim.measurement_routing_diagnostics() == expected
    assert sim.norm_diagnostics()["measurement_routing"] == expected
    assert sim.copy().measurement_routing_diagnostics() == expected
    snapshot = sim._execution_snapshot()
    sim.expectation("Z", 2)
    sim.measure("I", 2, outcome=1)
    with pytest.raises(ValueError):
        sim.measure("Z", 2, outcome=-1)
    assert sim.measurement_routing_diagnostics() == expected
    sim.measure("Z", 2, outcome=1)
    sim._restore_execution_snapshot(snapshot)
    assert sim.measurement_routing_diagnostics() == expected


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("retain", ["all", "final", "none"])
def test_ensemble_routing_counts_use_multiplicity_and_report_missing_history(strategy, retain):
    engine = StabilizerMpsSimulator(1).compile([
        ("h", 0), ("measure", "Z", 0, None, False), ("measure", "Z", 0),
    ])
    result = engine.run(shots=16, seed=42, strategy=strategy, retain=retain, progress=False)
    report = result.measurement_routing_diagnostics()
    available = sum(result.counts)
    assert report["represented_shots"] == available
    assert report["unavailable_shots"] == 16 - available
    assert report["counts"] == {"stim": available, "stim_region": 0, "mps": available}
    assert report["total"] == 2 * available
    assert report["fallback_reasons"] == ({"explicit_fixed_basis": available} if available else {})
    if retain == "all":
        assert available == 16
    if retain == "none":
        assert available == 0
    assert engine.measurement_routing_diagnostics()["total"] == 0


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("entry", [
    ("measure", "Z", 0, None, False),
    ("measure_reset", "Z", 0, None, False),
    ("mrz", 0, None, False),
])
def test_replay_preserves_explicit_fixed_basis_controls(strategy, entry):
    engine = StabilizerMpsSimulator(1).compile([("h", 0), entry])
    result = engine.run(shots=8, seed=3, strategy=strategy, progress=False)
    for sim in result.optimizers:
        assert sim.norm_events[0].measurement_backend == "mps"
        assert sim.norm_events[0].measurement_fallback_reason == "explicit_fixed_basis"


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("entry", [
    ("measure", "Z", 0),
    ("measure", "Z", 0, None, None),
    ("measure_reset", "Z", 0),
    ("measure_reset", "Z", 0, None, None),
    ("mrz", 0),
    ("mrz", 0, None, None),
])
def test_replay_defaults_to_native_disentangling(strategy, entry):
    engine = StabilizerMpsSimulator(2).compile([("h", 0), entry])
    result = engine.run(shots=8, seed=3, strategy=strategy, progress=False)
    for sim in result.optimizers:
        assert sim.norm_events[0].measurement_backend == "stim"
        assert sim.state.max_bond() == 1


@pytest.mark.parametrize("api", ["measure", "measure_many", "measure_reset"])
@pytest.mark.parametrize("options", [{}, {"disentangle": None}, {"disentangle": True},
                                     {"disentangle": False}, {"absorb_basis": False}])
def test_measurement_default_updates_basis_after_native_fallback(converter, api, options):
    sim = StabilizerMpsSimulator(2, chi=8, exact_cooling=False, to_backend=converter)
    sim.apply([("h", 0), ("t", 0), ("cnot", 0, 1)])
    before = sim.to_statevector()
    tableau = sim.state._sim.current_inverse_tableau()
    if api == "measure_many":
        sim.measure_many([("Z", 1, +1)], **options)
    else:
        getattr(sim, api)("Z", 1, outcome=+1, **options)
    _same_state(sim.to_statevector(), _project(before, "Z", 1, +1, 2))
    enabled = options.get("disentangle", options.get("absorb_basis")) is not False
    assert sim.norm_events[0].measurement_backend == "mps"
    assert sim.norm_events[0].kind == ("measure_absorb" if enabled else "measure")
    assert (sim.state._sim.current_inverse_tableau() != tableau) == enabled


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("operation", ["measure", "measure_reset"])
def test_tree_replay_retains_explicit_none_fixed_basis_policy(strategy, operation):
    from pepsy import StabilizerTreeSimulator

    gates = [("h", 0), ("cnot", 0, 1)]
    reference = StabilizerTreeSimulator(2).apply(gates)
    tableau = reference.state._sim.current_inverse_tableau()
    engine = StabilizerTreeSimulator(2).set_gates([
        *gates, (operation, "Z", 0, +1, None),
    ])
    result = engine.run(shots=2, strategy=strategy, seed=3, progress=False)
    for sim in result.optimizers:
        _same_state(sim.to_statevector(), np.array([1., 0., 0., 0.]))
        assert sim.state._sim.current_inverse_tableau() == tableau
