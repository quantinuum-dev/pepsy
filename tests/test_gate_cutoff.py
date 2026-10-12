"""Public gate truncation follows network precision before upstream dispatch."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import gate, gate_simple


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("policy", ["default", "auto", 0.0, 1e-3])
def test_public_gate_cutoff_changes_retained_schmidt_rank(backend, dtype, simple, policy):
    # The small Schmidt weight (1e-8) lies between the two dtype thresholds.
    state = qtn.MPS_computational_state("00", dtype=dtype)
    angle = 1e-4
    payload = np.eye(4, dtype=dtype)
    payload[0, 0] = payload[3, 3] = np.cos(angle)
    payload[3, 0] = np.sin(angle)
    payload[0, 3] = -np.sin(angle)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda x: torch.as_tensor(x.copy()))
        payload = torch.as_tensor(payload)
    options = {"contract": "split", "inplace": False}
    if policy != "default":
        options.update(cutoff=policy, cutoff_mode="auto")
    original = ar.to_numpy(state.to_dense()).copy()
    if simple:
        gauges = {}
        result = gate_simple(state, payload, (0, 1), gauges=gauges,
                             renorm=False, strip_exponent=True, **options)
        physical = result.copy()
        physical.gauge_simple_insert(gauges)
    else:
        result = gate(state, payload, (0, 1), **options)
        physical = result
    retained = policy == 0.0 or (dtype == "complex128" and policy in ("default", "auto"))
    assert result.max_bond() == (2 if retained else 1)
    expected = [np.cos(angle), 0, 0, np.sin(angle) if retained else 0]
    np.testing.assert_allclose(ar.to_numpy(physical.to_dense()).ravel(), expected,
                               atol=3e-7 if dtype == "complex64" else 1e-12)
    np.testing.assert_array_equal(ar.to_numpy(state.to_dense()), original)
    assert ar.get_dtype_name(next(iter(result)).data) == dtype


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("cutoff", [-1, float("nan"), float("inf"), "bad"])
def test_invalid_cutoff_rejected_before_mutation(simple, cutoff):
    state = qtn.MPS_computational_state("00", dtype="complex128")
    before = state.to_dense().copy()
    gauges = {}
    with pytest.raises(ValueError, match="cutoff must be"):
        if simple:
            gate_simple(state, np.eye(4), (0, 1), gauges=gauges, cutoff=cutoff)
        else:
            gate(state, np.eye(4), (0, 1), cutoff=cutoff)
    np.testing.assert_array_equal(state.to_dense(), before)
    assert not gauges


@pytest.mark.parametrize("simple", [False, True])
def test_routed_peps_auto_cutoff_matches_explicit_policy(simple):
    state = qtn.PEPS.rand(2, 3, bond_dim=2, seed=18, dtype="complex64")
    payload = np.diag([1, 1j, -1j, -1]).astype("complex64")
    outputs = []
    for cutoff in ("auto", 1e-6):
        options = dict(cutoff=cutoff, cutoff_mode="auto", max_bond=2,
                       path_compress=True, path_compress_cutoff="auto", inplace=False)
        if simple:
            gauges = {}
            result = gate_simple(state, payload, ((0, 0), (0, 2)), gauges=gauges,
                                 renorm=False, **options)
            result.gauge_simple_insert(gauges)
        else:
            result = gate(state, payload, ((0, 0), (0, 2)), **options)
        outputs.append(result.to_dense(optimize="greedy"))
    np.testing.assert_allclose(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,cutoff", [("complex64", 1e-6), ("complex128", 1e-12)])
def test_native_simple_update_auto_matches_explicit_policy(dtype, cutoff):
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, OneDMap

    fermion = Fermion(spinful=True, symmetry="U1")
    edge = ((0, 0), (0, 1))
    term = fermion.operator_term(
        [(1., ((edge[0], "create_up"), (edge[1], "annihilate_up")))],
        sites=edge, add_hc=True,
    )
    state = fermion.to_pepo({edge: term}, Lx=2, Ly=2,
                           mapper=OneDMap(2, 2, mode="snake-row-major"),
                           max_bond=16, compress=False)
    state.apply_to_arrays(lambda data: data.astype(dtype))
    payload = fermion.hopping_gate(0.01, t=1.).copy()
    payload.apply_to_arrays(lambda data: data.astype(dtype))
    norms = []
    for policy in ("auto", cutoff):
        gauges = {}
        result = gate_simple(state, payload, edge, gauges=gauges, cutoff=policy,
                             max_bond=16, contract="split", renorm=False,
                             strip_exponent=True, inplace=False)
        result.gauge_simple_insert(gauges)
        assert all(type(t.data).__name__ == "U1FermionicArray" for t in result)
        assert all(ar.get_dtype_name(t.data) == dtype for t in result)
        norms.append(complex(result.norm()))
    np.testing.assert_allclose(*norms, rtol=0, atol=0)
