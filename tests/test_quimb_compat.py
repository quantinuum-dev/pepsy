"""Focused compatibility tests for optional Quimb capabilities."""

import inspect

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.bp._symmray import (
    _quimb_safe_inverse_supports_symmray,
    install_quimb_symmray_compat,
)
from pepsy._internal.quimb import (
    quimb_bp_constructor_option_supported,
    quimb_bp_constructor_options,
    quimb_gloop_options,
    quimb_process_loop_series_expansion_weights,
    quimb_mpo_auto_swap_function,
)
from pepsy._internal.random import backend_random_array


def test_bp_constructor_capability_probe_keeps_run_options_out_of_ctor():
    """Constructor filtering follows the installed BP signature."""
    options = quimb_bp_constructor_options(
        "d1bp",
        {"damping": 0.2, "diis": True, "not_a_constructor_option": 1},
    )

    assert "damping" in options
    assert "not_a_constructor_option" not in options
    if quimb_bp_constructor_option_supported("d1bp", "diis"):
        assert options["diis"] is True
    else:
        assert "diis" not in options


def test_backend_random_array_is_seeded_and_native_for_numpy():
    """The randomized FIT helper can use Autoray's native random operation."""
    like = np.zeros((2, 3), dtype=np.complex64)
    first = backend_random_array(
        like.shape, like=like, dtype=np.complex64, scale=0.2, rng=17
    )
    second = backend_random_array(
        like.shape, like=like, dtype=np.complex64, scale=0.2, rng=17
    )

    assert first.dtype == np.complex64
    np.testing.assert_array_equal(first, second)


def test_gloop_options_are_capability_checked():
    """Explicit generalized-loop controls fail clearly when unavailable."""
    options = {"join_overlap": 2}
    if "join_overlap" in inspect.signature(
        qtn.TensorNetwork.gen_gloops
    ).parameters:
        assert quimb_gloop_options(options) == options
    else:
        with pytest.raises(NotImplementedError, match="generalized-loop"):
            quimb_gloop_options(options)


def test_loop_series_weight_adapter_handles_num_tensors_change():
    """Loop-series resummation adapts to Quimb's new tensor-count argument."""
    from quimb.tensor.belief_propagation.bp_common import (
        process_loop_series_expansion_weights,
    )

    weights = {frozenset((0, 1)): 0.1}
    suppression = quimb_process_loop_series_expansion_weights(
        weights,
        num_tensors=2,
        multi_excitation_correct=True,
        tol_correction=1e-14,
        maxiter_correction=100,
        return_all=True,
    )

    assert np.isfinite(suppression[frozenset((0, 1))])
    if "num_tensors" in inspect.signature(
        process_loop_series_expansion_weights
    ).parameters:
        assert suppression[frozenset((0, 1))] == pytest.approx(
            0.9127652716086229
        )


def test_safe_inverse_compat_defers_to_fixed_quimb():
    """Do not replace Quimb's implementation after its upstream fix."""
    qd = pytest.importorskip("quimb.tensor.decomp")
    pytest.importorskip("symmray")
    if getattr(qd, "_pepsy_symmray_safe_inverse", False):
        pytest.skip("the compatibility wrapper was already installed")
    if not _quimb_safe_inverse_supports_symmray(qd):
        pytest.skip("the installed Quimb build needs the compatibility wrapper")

    original = qd.safe_inverse
    install_quimb_symmray_compat()

    assert qd.safe_inverse is original


def test_mpo_auto_swap_is_explicit_and_preserves_long_range_identity():
    """The prototype delegates to Quimb without changing ordinary gate paths."""
    mpo = qtn.MPO_identity(4, phys_dim=2)
    gate = np.eye(4).reshape(2, 2, 2, 2)

    assert quimb_mpo_auto_swap_function(mpo) is not None
    result = pepsy.gate_mpo_auto_swap(
        mpo,
        gate,
        (0, 3),
        swap_back=True,
        cutoff=1e-12,
    )

    assert result.L == 4
    np.testing.assert_allclose(result.to_dense(), np.eye(16))


def test_periodic_bond_names_distinguish_length_two_open_and_wrapping_bonds():
    """Pepsy's cycle helper uses Quimb's lattice-aware bond identity."""
    mapper = qtn.LatticeBondMap(2, 2)
    assert mapper((0, 0), (1, 0)) != mapper((1, 0), (2, 0))

    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=1, seed=5)
    pepsy.add_cycle(peps, bond_dim=1)

    assert len(qtn.bonds(peps["I0,0"], peps["I1,0"])) == 2
    assert len(qtn.bonds(peps["I0,0"], peps["I0,1"])) == 2
