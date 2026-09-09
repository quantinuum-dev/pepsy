"""Born probabilities agree across tree measurement APIs and shot replay."""

from copy import deepcopy

import numpy as np
import pytest

from pepsy import TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.optimizers.noise import _coalesced_measurement_probabilities


PAULIS = {
    "X": np.array([[0., 1.], [1., 0.]], dtype=complex),
    "Y": np.array([[0., -1j], [1j, 0.]], dtype=complex),
    "Z": np.diag([1., -1.]).astype(complex),
}


def _rare_state(axes, *, mode="direct", backend="numpy", amplitude=1e-9):
    n = 5
    plan = TreePlan.from_order((3, 1, 4, 0, 2), structure="balanced")
    sites = (4, 0, 2)[:len(axes)]
    rotation = np.array([[np.sqrt(1. - amplitude**2), -amplitude],
                         [amplitude, np.sqrt(1. - amplitude**2)]], dtype=complex)
    h = np.array([[1., 1.], [1., -1.]], dtype=complex) / np.sqrt(2.)
    gates = [(rotation, sites[0])]
    for site, axis in zip(sites, axes):
        if axis != "Z":
            gates.append((h if axis == "X" else np.diag([1., 1j]) @ h, site))
    state = TreeOptimizer(None, tree=plan, run=False).tn
    if backend == "torch":
        torch = pytest.importorskip("torch")
        gates = [(torch.as_tensor(gate), site) for gate, site in gates]
        state.apply_to_arrays(torch.as_tensor)
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        if not jax.config.x64_enabled:
            pytest.skip("rare complex128 branch check requires JAX x64")
        gates = [(jax.numpy.asarray(gate), site) for gate, site in gates]
        state.apply_to_arrays(jax.numpy.asarray)
    opt = TreeOptimizer(gates, n=n, tree=plan, tn=state, cutoff=0., mode=mode,
                        chi=8, fit_rtol=None, seed=51)
    return opt, sites


def _dense_projection(state, axes, sites, outcome):
    vector = state.reshape((2,) * int(np.log2(state.size)))
    transformed = vector
    for axis, site in zip(axes, sites):
        transformed = np.moveaxis(np.tensordot(PAULIS[axis], transformed,
                                              axes=((1,), (site,))), 0, site)
    projected = 0.5 * (vector + outcome * transformed)
    return projected.reshape(-1)


@pytest.mark.parametrize("axes", ["Z", "X", "Y", "ZZ", "XYZ"])
def test_rare_born_weights_and_collapse_match_dense_reference(axes):
    opt, sites = _rare_state(axes)
    vector = opt.to_dense().reshape(-1)
    expected = _dense_projection(vector, axes, sites, -1)
    probability = np.vdot(expected, expected).real / np.vdot(vector, vector).real
    assert 0.5e-18 < probability < 2e-18
    before_rng = deepcopy(opt.rng.bit_generator.state)
    before_history = deepcopy(opt.update_history)
    probabilities = opt._measurement_probabilities(axes, sites)
    assert probabilities[1] == pytest.approx(probability, rel=1e-6, abs=0.)
    assert sum(probabilities) == pytest.approx(1.)
    assert opt.rng.bit_generator.state == before_rng
    assert opt.update_history == before_history
    np.testing.assert_allclose(opt.to_dense().reshape(-1), vector, atol=1e-14)
    outcome, actual = opt.measure_pauli(axes, sites, -1)
    assert outcome == -1 and actual == pytest.approx(probability, rel=1e-6, abs=0.)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected / np.linalg.norm(expected),
                               atol=5e-7, rtol=5e-7)
    assert opt.norm() == pytest.approx(1.)
    assert opt.tn.is_canonical_form(opt.center)


@pytest.mark.parametrize("mode", ["direct", "dm", "src", "sdc", "zipup", "dmrg2"])
def test_rare_measurement_agrees_between_direct_queue_and_coalesced(mode):
    opt, sites = _rare_state("Z", mode=mode)
    parent = opt.to_dense().copy()
    event = opt.measure_event("Z", sites, -1)
    direct = opt.copy()
    _, p = direct.measure_pauli("Z", sites, -1)
    queued = opt.copy().run([event])
    assert queued.measurements[-1][3] == pytest.approx(p, rel=1e-12, abs=0.)
    for strategy in ("independent", "coalesced"):
        result = opt.run([event], shots=3, strategy=strategy, seed=7)
        for child in result.optimizers:
            assert child.measurements[-1][3] == pytest.approx(p, rel=1e-12, abs=0.)
            np.testing.assert_allclose(child.to_dense(), direct.to_dense(), atol=1e-12)
    np.testing.assert_allclose(queued.to_dense(), direct.to_dense(), atol=1e-12)
    np.testing.assert_array_equal(opt.to_dense(), parent)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_tree_parity_probabilities_preserve_backend(backend):
    opt, sites = _rare_state("XYZ", backend=backend)
    assert opt._measurement_probabilities("XYZ", sites)[1] == pytest.approx(1e-18, rel=2e-6, abs=0.)
    opt.measure_pauli("XYZ", sites, -1)
    assert opt.backend == backend
    assert opt.norm() == pytest.approx(1.)


def test_coalesced_probability_hook_preserves_rare_negative_weight():
    opt, sites = _rare_state("ZZ")
    p_plus, p_minus = _coalesced_measurement_probabilities(opt, "ZZ", sites)
    assert p_plus == 1. and p_minus == pytest.approx(1e-18, rel=1e-12, abs=0.)


def test_parity_coherence_and_extreme_represented_scales():
    opt, sites = _rare_state("ZZ", amplitude=0.3)
    opt.apply_1q(np.array([[1., 1.], [1., -1.]], dtype=complex) / np.sqrt(2.), sites[1])
    vector = opt.to_dense().reshape(-1)
    expected = _dense_projection(vector, "ZZ", sites, 1)
    probabilities = opt._measurement_probabilities("ZZ", sites)
    opt.tn.exponent = 400.
    assert opt._measurement_probabilities("ZZ", sites) == pytest.approx(probabilities)
    opt.measure_pauli("ZZ", sites, 1)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected / np.linalg.norm(expected), atol=1e-12)


def test_wide_parity_probability_uses_no_dense_state_operator_or_optimizer_copy(monkeypatch):
    opt = TreeOptimizer(None, n=24, max_operator_qubits=2, run=False)
    def forbidden(*args, **kwargs):
        pytest.fail("probability query materialized an unnecessary state or operator")
    monkeypatch.setattr(opt, "to_dense", forbidden)
    monkeypatch.setattr(opt, "copy", forbidden)
    monkeypatch.setattr(np, "kron", forbidden)
    assert opt._measurement_probabilities("Z" * 24, tuple(range(24))) == (1., 0.)


def test_probability_readout_rejects_duplicate_and_inactive_labels_before_gauge_changes():
    opt, _ = _rare_state("Z")
    center = opt.center
    for axes, where in (("ZZ", (0, 0)), ("Z", (9,))):
        with pytest.raises(ValueError):
            opt._measurement_probabilities(axes, where)
        assert opt.center == center


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_entangled_root_site_parity_matches_dense_in_complex64(backend):
    plan = TreePlan.from_order(range(1, 8), structure="balanced", root_qubit=0)
    state = TreeTensorNetwork.rand(plan, D=3, dtype="complex64", seed=71)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.as_tensor)
    opt = TreeOptimizer(None, tn=state, cutoff=0., run=False)
    axes, sites = "YZXZ", (7, 0, 3, 1)
    vector = opt.to_dense().reshape(-1)
    expected = [np.linalg.norm(_dense_projection(vector, axes, sites, sign))**2
                / np.linalg.norm(vector)**2 for sign in (1, -1)]
    probabilities = opt._measurement_probabilities(axes, sites)
    np.testing.assert_allclose(probabilities, expected, atol=5e-7)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), vector, atol=5e-7)
    assert opt.backend == backend and "complex64" in str(opt.backend_dtype)
