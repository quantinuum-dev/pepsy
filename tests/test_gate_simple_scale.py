"""Simple-update gauge scaling preserves amplitudes and logarithmic norms."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import loop_cluster_expand, two_norm_bp
from pepsy.operators import gate_simple
from pepsy.tensors import ps_to_peps


def physical(core, gauges):
    state = core.copy()
    state.gauge_simple_insert(gauges)
    return state


def vector(core, gauges):
    return physical(core, gauges).to_dense(optimize="auto-hq").reshape(-1)


def to_backend(tn, backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        tn.apply_to_arrays(lambda data: torch.as_tensor(data.copy(), device="cpu"))
        return lambda data: torch.as_tensor(data, device="cpu")
    return np.asarray


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("where", [(0, 1), (0, 2)])
def test_scaled_su_adjacent_and_routed_gates_match_dense(backend, where):
    state = qtn.MPS_rand_state(3, bond_dim=2, seed=8, dtype="complex128")
    converter = to_backend(state, backend)
    state.exponent = 0.37
    gauges = {}
    state.gauge_all_simple_(gauges=gauges)
    before = np.asarray(ar.to_numpy(vector(state, gauges))).reshape(2, 2, 2)
    rng = np.random.default_rng(2)
    gate = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / 3
    order = [*where, *(i for i in range(3) if i not in where)]
    expected = (gate @ before.transpose(order).reshape(4, -1)).reshape(2, 2, 2)
    expected = expected.transpose(np.argsort(order)).reshape(-1)
    source = [ar.to_numpy(t.data).copy() for t in state]
    exponent_before = state.exponent
    evolved = gate_simple(
        state, converter(gate), where=where, gauges=gauges,
        renorm=False, strip_exponent=True, cutoff=0., inplace=False,
    )
    np.testing.assert_allclose(ar.to_numpy(vector(evolved, gauges)), expected, atol=2e-11)
    assert state.exponent == exponent_before
    for t, original in zip(state, source):
        np.testing.assert_array_equal(ar.to_numpy(t.data), original)
    for gauge in gauges.values():
        assert np.mean(np.abs(ar.to_numpy(gauge)) ** 2) == pytest.approx(1.)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_scaled_truncated_su_matches_unscaled_physical_state(backend):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=9, dtype="complex128")
    converter = to_backend(state, backend)
    gauges = {}
    state.gauge_all_simple_(gauges=gauges)
    other = state.copy()
    other_gauges = {ix: ar.do("copy", s) for ix, s in gauges.items()}
    rng = np.random.default_rng(3)
    gates = tuple(
        (converter(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))), edge)
        for edge in state.gen_bond_coos()
    )
    for strip, core, weights in ((False, state, gauges), (True, other, other_gauges)):
        gate_simple(core, gates, gauges=weights, renorm=False, strip_exponent=strip,
                    max_bond=1, cutoff=1e-12, cutoff_mode="rsum2")
    assert state.max_bond() == other.max_bond() == 1
    np.testing.assert_allclose(ar.to_numpy(vector(other, other_gauges)),
                               ar.to_numpy(vector(state, gauges)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("factor", [1e-4, 1e4])
def test_scaled_su_retains_norms_beyond_float_range(factor):
    state = ps_to_peps(2, 2, dtype="complex128")
    state.exponent = 2.5
    gauges = {}
    gate = np.eye(4) * factor
    gate_simple(state, ((gate, ((0, 0), (0, 1))),) * 100, gauges=gauges,
                renorm=False, strip_exponent=True, cutoff=0.)
    expected_log_norm2 = 2 * (2.5 + 100 * np.log10(factor))
    snapshot = physical(state, gauges)
    bp = two_norm_bp(snapshot, tol=1e-12)
    assert bp.converged
    cluster = loop_cluster_expand(snapshot, gloops=[], messages=bp.snapshot(),
                                  run_bp=False, strip_exponent=True)
    for mantissa, exponent in (
        snapshot.make_norm().contract(all, strip_exponent=True),
        bp.contract(strip_exponent=True), cluster.estimate,
    ):
        log_norm2 = np.log10(abs(complex(mantissa))) + float(exponent)
        assert log_norm2 == pytest.approx(expected_log_norm2, abs=1e-10)
    assert all(np.all(np.isfinite(t.data)) for t in state)
    assert all(np.all(np.isfinite(s)) for s in gauges.values())


def test_scaled_su_preserves_torch_gate_gradient():
    torch = pytest.importorskip("torch")
    state = qtn.MPS_rand_state(2, bond_dim=2, seed=11, dtype="complex128")
    to_backend(state, "torch")
    gauges = {}
    state.gauge_all_simple_(gauges=gauges)
    before = vector(state, gauges)
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    gate = torch.diag(torch.stack((1+theta, 1-theta, 2+theta, 1.2-theta))).to(torch.complex128)
    expected = gate @ before
    gate_simple(state, gate, where=(0, 1), gauges=gauges, renorm=False,
                strip_exponent=True, cutoff=0.)
    actual = vector(state, gauges)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    actual_loss = torch.sum(abs(actual) ** 2)
    expected_loss = torch.sum(abs(expected) ** 2)
    actual_grad, = torch.autograd.grad(actual_loss, theta, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected_loss, theta)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("options,match", [
    ({"renorm": True}, "renorm=False"),
    ({"cutoff_mode": "abs", "cutoff": 1e-6}, "relative cutoff"),
])
def test_scaled_su_rejects_incompatible_policy_before_mutation(options, match):
    state = qtn.MPS_computational_state("00", dtype="complex128")
    original = state.to_dense().copy()
    gauges = {}
    opts = {"renorm": False, "strip_exponent": True, **options}
    with pytest.raises(ValueError, match=match):
        gate_simple(state, np.eye(4), where=(0, 1), gauges=gauges, **opts)
    np.testing.assert_array_equal(state.to_dense(), original)
    assert not gauges
    assert state.exponent == 0


@pytest.mark.parametrize("symmetry", ["U1", "U1U1", "Z2"])
def test_scaled_native_fermionic_su_matches_unscaled_norm(symmetry):
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, OneDMap

    fermion = Fermion(spinful=True, symmetry=symmetry)
    edge = ((0, 0), (0, 1))
    term = fermion.operator_term(
        [(1., ((edge[0], "create_up"), (edge[1], "annihilate_up")))],
        sites=edge, add_hc=True,
    )
    state = fermion.to_pepo({edge: term}, Lx=2, Ly=2,
                           mapper=OneDMap(2, 2, mode="snake-row-major"),
                           max_bond=16, compress=False)
    gauges = {}
    state.gauge_all_simple_(gauges=gauges)
    other = state.copy()
    other_gauges = {ix: s.copy() for ix, s in gauges.items()}
    gate = fermion.hopping_gate(0.01, t=1.).H
    for strip, core, weights in ((False, state, gauges), (True, other, other_gauges)):
        gate_simple(core, gate, where=edge, gauges=weights, max_bond=16,
                    cutoff=0., contract="split", renorm=False, strip_exponent=strip)
    expected = complex(physical(state, gauges).norm())
    actual = complex(physical(other, other_gauges).norm())
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert all(type(t.data).__name__.endswith("FermionicArray") for t in other)
    assert all(type(s).__name__ == "BlockVector" for s in other_gauges.values())
