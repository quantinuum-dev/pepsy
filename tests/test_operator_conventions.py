"""Dense oracles for operator construction and lazy application conventions."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators.gates import (
    build_mpo_from_gates,
    build_pepo_from_gates,
    gate_with_submpo,
)
from pepsy.tensors import tns_align


def _matrices():
    rng = np.random.default_rng(928)
    return tuple(
        rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        for n in (2, 4, 2)
    )


@pytest.mark.parametrize("geometry", ["mpo", "pepo"])
@pytest.mark.parametrize("tensor_gate", [False, True])
def test_gate_builders_preserve_noncommuting_application_order(geometry, tensor_gate):
    first, middle, last = _matrices()
    gate = middle.reshape(2, 2, 2, 2) if tensor_gate else middle
    if geometry == "mpo":
        build = build_mpo_from_gates
        where = [(0,), (0, 1), (1,)]
    else:
        build = build_pepo_from_gates
        where = [((0, 0),), ((0, 0), (1, 0)), ((1, 0),)]
    stream = list(zip((first, gate, last), where))
    result = build(stream, max_bond=4, cutoff=0.0)
    expected = np.kron(np.eye(2), last) @ middle @ np.kron(first, np.eye(2))
    np.testing.assert_allclose(result.to_dense(), expected, atol=1e-12)


@pytest.mark.parametrize("transpose", [False, True])
def test_pepo_state_application_matches_dense_and_preserves_inputs(transpose):
    first, middle, last = _matrices()
    operator = build_pepo_from_gates(
        [(middle, ((0, 0), (1, 0)))], max_bond=4, cutoff=0.0
    )
    state = qtn.PEPS.product_state({(0, 0): first[:, 0], (1, 0): last[:, 0]})
    operator_before = operator.to_dense().copy()
    state_before = state.to_dense().reshape(-1).copy()
    original_inds = tuple(t.inds for t in state)
    output = tns_align(state, operator, transpose=transpose)
    got = output.to_dense(["k0,0", "k1,0"]).reshape(-1)
    expected = (middle.T if transpose else middle) @ state_before
    np.testing.assert_allclose(got, expected, atol=1e-12)
    np.testing.assert_array_equal(operator.to_dense(), operator_before)
    np.testing.assert_array_equal(state.to_dense().reshape(-1), state_before)
    assert tuple(t.inds for t in state) == original_inds
    assert output.num_tensors == state.num_tensors + operator.num_tensors


@pytest.mark.parametrize("which", ["upper", "lower"])
@pytest.mark.parametrize(
    "transpose,inplace_mpo,inplace",
    [(False, False, False), (True, False, True), (False, True, False)],
)
def test_lazy_submpo_applied_for_both_copy_options(which, transpose, inplace_mpo, inplace):
    rng = np.random.default_rng(93)
    applied = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    target = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    submpo = qtn.MatrixProductOperator.from_dense(applied, cutoff=0.0)
    mpo = qtn.MatrixProductOperator.from_dense(target, cutoff=0.0)
    submpo_before = submpo.to_dense().copy()
    result = gate_with_submpo(
        mpo, submpo, where=(0, 1), which=which, transpose=transpose,
        inplace=inplace, inplace_mpo=inplace_mpo, max_bond=4, cutoff=0.0,
    )
    transformed = applied.T if transpose else applied
    expected = transformed @ target if which == "upper" else target @ transformed
    np.testing.assert_allclose(result.to_dense(), expected, atol=1e-12)
    if not inplace:
        np.testing.assert_allclose(mpo.to_dense(), target, atol=1e-12)
    else:
        assert result is mpo
    if not inplace_mpo:
        np.testing.assert_array_equal(submpo.to_dense(), submpo_before)
