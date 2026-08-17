"""Tests for the explicit MPO channel/transition automaton."""

import numpy as np
import pytest

from pepsy.operators import MPOAutomaton, MPOChannel, MPOTransition


def _kron_site_operator(operators):
    """Build a site-major dense product operator for a short chain."""
    result = np.asarray(operators[0])
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def _arrays_to_dense(arrays):
    """Contract small dense MPO arrays without importing Quimb."""
    arrays = tuple(np.asarray(array) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    current = arrays[0]
    physical_dim = current.shape[-1]
    for array in arrays[1:-1]:
        current = np.einsum("aij,abkl->bikjl", current, array)
        current = current.reshape(
            current.shape[0],
            current.shape[1] * current.shape[2],
            current.shape[3] * current.shape[4],
        )
    current = np.einsum("aij,akl->ikjl", current, arrays[-1])
    return current.reshape(physical_dim ** len(arrays), -1)


def test_factorized_paths_materialize_exact_mpo_without_compression():
    """Explicit channels produce the sum of local product operators."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    automaton = MPOAutomaton(4)
    automaton.add_local_term(1, x, coefficient=0.25)
    automaton.add_factorized_term((0, 3), (z, z), coefficient=-1.5)

    mpo = automaton.to_mpo()
    expected = (
        0.25 * _kron_site_operator((identity, x, identity, identity))
        - 1.5 * _kron_site_operator((z, identity, identity, z))
    )

    assert automaton.bond_dimensions == (3, 3, 3)
    assert mpo.pepsy_automaton.bond_dimensions == automaton.bond_dimensions
    np.testing.assert_allclose(mpo.to_dense(), expected)


def test_legacy_channels_and_transitions_round_trip():
    """Existing tuple data can be wrapped and emitted unchanged."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    channels = [
        [(("start",), None), (("done",), None), ("path", None)],
        [(("start",), None), (("done",), None), ("path", None)],
    ]
    transitions = [
        [(("start",), "path", x)],
        [("path", "path", identity)],
        [("path", ("done",), x)],
    ]

    automaton = MPOAutomaton.from_legacy(
        channels,
        transitions,
        start_state=("start",),
        done_state=("done",),
    )
    round_trip_channels, round_trip_transitions = automaton.to_legacy()

    assert round_trip_channels == channels
    assert len(round_trip_transitions) == len(transitions)
    for got, expected in zip(round_trip_transitions, transitions):
        assert got[0][:2] == expected[0][:2]
        np.testing.assert_allclose(got[0][2], expected[0][2])
    np.testing.assert_allclose(
        automaton.to_mpo().to_dense(),
        _kron_site_operator((x, identity, x)),
    )


def test_string_operators_are_explicit_and_not_factorized():
    """A supplied intermediate string is placed directly on the path."""
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])

    automaton = MPOAutomaton(3)
    automaton.add_factorized_term(
        (0, 2),
        (x, x),
        string_operators=(z,),
    )

    np.testing.assert_allclose(
        automaton.to_mpo().to_dense(),
        _kron_site_operator((x, z, x)),
    )
    assert automaton.channels[0][2].state == ("term", 0)


def test_validation_catches_topology_and_shape_errors():
    """Malformed transitions fail before any tensor-network construction."""
    with pytest.raises(ValueError, match="invalid edge"):
        MPOAutomaton.from_legacy(
            [[("start",), ("done",)]],
            [[(("missing",), ("done",), np.eye(2))],
             [(("done",), ("done",), np.eye(2))]],
        ).validate()

    automaton = MPOAutomaton(
        2,
        channels=[[MPOChannel(("start",)), MPOChannel(("done",))]],
        transitions=[
            [MPOTransition(("start",), ("done",), np.eye(2))],
            [MPOTransition(("done",), ("done",), np.eye(3))],
        ],
    )
    with pytest.raises(ValueError, match="one shape"):
        automaton.to_mpo()


def test_compression_must_be_an_explicit_follow_up_operation():
    """The structural builder cannot silently invoke an SVD."""
    automaton = MPOAutomaton(1)
    automaton.add_local_term(0, np.eye(2))
    with pytest.raises(ValueError, match="never compresses"):
        automaton.to_mpo(compress=True)


def test_backend_parameters_are_kept_in_raw_mpo_tensors():
    """A Torch parameter remains connected to the exactly assembled MPO."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.4, requires_grad=True)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    automaton = MPOAutomaton(1)
    automaton.add_local_term(0, theta * x)
    mpo = automaton.to_mpo()

    mpo.to_dense()[0, 1].backward()
    assert theta.grad.item() == pytest.approx(1.0)


def test_product_paths_support_more_than_two_sites():
    """The general path primitive handles higher-order product terms."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    automaton = MPOAutomaton(5)
    automaton.add_product_term(
        (0, 2, 4),
        (x, z, x),
        coefficient=0.5,
        string_operators=(z, identity),
    )

    expected = 0.5 * _kron_site_operator((x, z, z, identity, x))
    np.testing.assert_allclose(_arrays_to_dense(automaton.to_arrays()), expected)
    assert automaton.bond_dimensions == (3, 3, 3, 3)


def test_composition_builds_exact_operator_products():
    """Channel pairing composes MPOs without forming a global input."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    automaton = MPOAutomaton(2)
    automaton.add_local_term(0, x)
    automaton.add_local_term(1, z)
    product = automaton.compose(automaton)

    expected = _kron_site_operator((x, identity)) + _kron_site_operator((identity, z))
    expected = expected @ expected
    np.testing.assert_allclose(_arrays_to_dense(product.to_arrays()), expected)


def test_identity_and_direct_sum_preserve_backend_free_exact_paths():
    """Identity plus a renamed automaton produces an exact direct sum."""
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])

    correction = MPOAutomaton(2)
    correction.add_local_term(0, x)
    correction.add_local_term(1, x)
    total = MPOAutomaton.identity(2, 2)
    total.add_automaton(correction, coefficient=0.25)

    expected = np.eye(4) + 0.25 * (
        _kron_site_operator((x, identity))
        + _kron_site_operator((identity, x))
    )
    np.testing.assert_allclose(_arrays_to_dense(total.to_arrays()), expected)
    assert total.power(0).to_arrays()[0].shape == (2, 2, 2)


def test_trim_removes_dead_channels_without_changing_the_operator():
    """Graph trimming is exact and keeps the structural compression explicit."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    automaton = MPOAutomaton(3)
    automaton.add_factorized_term((0, 2), (x, x))
    automaton.add_channel(0, ("dead",))
    automaton.add_channel(1, ("dead",))

    trimmed = automaton.trim()

    assert automaton.bond_dimensions == (4, 4)
    assert trimmed.bond_dimensions == (3, 3)
    np.testing.assert_allclose(
        _arrays_to_dense(trimmed.to_arrays()),
        _arrays_to_dense(automaton.to_arrays()),
    )
