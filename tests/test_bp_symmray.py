"""Symmray fermionic coverage for Pepsy's 2-norm BP corrections."""

from __future__ import annotations

import numpy as np
import pytest

sr = pytest.importorskip("symmray")
qtn = pytest.importorskip("quimb.tensor")

from pepsy.bp import (  # noqa: E402
    gauge_all,
    loop_cluster_expand,
    loop_series_expand,
    one_norm_bp,
    partitioned_expand,
    relay_bp,
    two_norm_bp,
    weight_pass,
)
from pepsy.tensors import (  # noqa: E402
    SymPEPS,
    site_charge_alternating,
)


@pytest.mark.parametrize("bond_dim", (2, 3))
def test_fermionic_u1_two_norm_bp_is_exact_on_a_tree(bond_dim):
    """D2BP preserves the native graded contraction on a PEPS tree."""
    state = SymPEPS.random(
        1,
        4,
        symmetry="U1",
        bond_dim=bond_dim,
        phys_dim=2,
        fermionic=True,
        seed=700 + bond_dim,
        dtype="complex128",
    )

    exact = complex(state.norm())
    bp = two_norm_bp(
        state.tn,
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )

    assert bp.converged
    assert abs(complex(bp.contract()) - exact) <= 1e-10 * max(1.0, abs(exact))

    corrected = loop_cluster_expand(
        state.tn,
        gloops=0,
        norm="2norm",
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )
    assert corrected.bp_converged
    assert abs(complex(corrected.estimate) - exact) <= 1e-10 * max(
        1.0, abs(exact)
    )


def test_d2bp_repairs_labelled_odd_arrays_after_new_with():
    """D2BP repairs the phase metadata dropped by Symmray ``new_with``."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=2,
        phys_dim=2,
        fermionic=True,
        seed=750,
        dtype="complex128",
    )
    broken = state.tn.copy()
    for tensor in broken.tensors:
        data = tensor.data
        if not getattr(data, "parity", 0):
            continue
        tensor.modify(
            data=type(data).from_blocks(
                data.blocks,
                duals=data.indices,
                charge=data.charge,
                symmetry=data.symmetry,
                phases=data.phases,
                label=data.label,
                dummy_modes=(),
            )
        )

    assert all(
        not tensor.data.dummy_modes
        for tensor in broken.tensors
        if getattr(tensor.data, "parity", 0)
    )

    result = two_norm_bp(
        broken,
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )
    assert result.converged
    assert all(
        tensor.data.dummy_modes
        for tensor in result.bp.tn.tensors
        if getattr(tensor.data, "parity", 0)
    )

    cluster = loop_cluster_expand(
        broken,
        gloops=0,
        norm="2norm",
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )
    assert cluster.bp_converged
    assert all(
        tensor.data.dummy_modes
        for tensor in cluster.bp.tn.tensors
        if getattr(tensor.data, "parity", 0)
    )


@pytest.mark.parametrize("bond_dim", (2, 3))
def test_fermionic_u1_loop_cluster_runs_on_3x4_peps(bond_dim):
    """D2BP loop clusters pad sparse charge blocks without densifying."""
    state = SymPEPS.random(
        3,
        4,
        symmetry="U1",
        bond_dim=bond_dim,
        phys_dim=2,
        fermionic=True,
        seed=800 + bond_dim,
        dtype="complex128",
    )

    result = loop_cluster_expand(
        state.tn,
        gloops=4,
        norm="2norm",
        max_iterations=300,
        tol=1e-10,
        diis=False,
    )

    assert result.bp_converged
    assert result.bp.__class__.__name__ == "D2BP"
    assert np.isfinite(float(np.real(result.estimate)))
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in result.messages.values()
    )
    assert all(
        type(tensor.data).__name__ == "U1FermionicArray"
        for tensor in state.tn.tensors
    )


def _fermionic_symmetry_cases():
    return (
        (
            "U1",
            "fermi_hubbard",
            {"symmetry": "U1", "phys_dim": 4},
        ),
        (
            "U1U1",
            "fermi_hubbard_u1u1",
            {
                "symmetry": "U1U1",
                "phys_dim": 4,
                "site_charge": site_charge_alternating(
                    (1, 0), (0, 1)
                ),
            },
        ),
        (
            "Z2",
            "fermi_hubbard_spinless",
            {"symmetry": "Z2", "phys_dim": 2},
        ),
    )


@pytest.mark.parametrize(
    ("label", "model", "state_options"),
    _fermionic_symmetry_cases(),
)
def test_fermionic_sympeps_su_d2bp_bridge_preserves_symmetry_blocks(
    label, model, state_options
):
    """SU gauges and BP loop expansion preserve native fermionic blocks."""
    state = SymPEPS.for_model(
        model,
        3,
        4,
        bond_dim=2,
        seed=1200,
        dtype="complex128",
        **state_options,
    )
    exact_norm = state.tn.norm()

    forward = gauge_all(
        state.tn,
        start="su",
        target="bp",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
        bp_options={
            "run_opts": {
                "max_iterations": 300,
                "tol": 1e-10,
                "diis": False,
            }
        },
    )
    assert forward.bp.converged, label
    reconstructed = forward.core.copy()
    reconstructed.gauge_simple_insert(forward.gauges)
    np.testing.assert_allclose(reconstructed.norm(), exact_norm, rtol=1e-10)
    assert all(
        type(message).__name__ == type(state.tn.tensors[0].data).__name__
        for message in forward.messages.values()
    )

    cluster = loop_cluster_expand(
        forward.bp.tn,
        gloops=4,
        norm="2norm",
        messages=forward.messages,
        run_bp=False,
    )
    assert np.isfinite(float(np.real(cluster.estimate))), label

    reverse = gauge_all(
        state.tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options={
            "run_opts": {
                "max_iterations": 300,
                "tol": 1e-10,
                "diis": False,
            }
        },
        conversion_options={"smudge": 1e-12},
    )
    assert reverse.bp.converged, label
    reverse_reconstructed = reverse.core.copy()
    reverse_reconstructed.gauge_simple_insert(reverse.gauges)
    np.testing.assert_allclose(
        reverse_reconstructed.norm(), exact_norm, rtol=1e-10
    )


def test_fermionic_u1_d2bp_corrections_and_relay_use_native_messages():
    """The D2 relay, series, and PNE paths share native Symmray handling."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=2,
        phys_dim=2,
        fermionic=True,
        seed=1300,
        dtype="complex128",
    )
    bond = next(iter(state.tn.inner_inds()))

    relay = relay_bp(
        state.tn,
        method="d2bp",
        num_relays=2,
        max_iterations=200,
        tol=1e-10,
    )
    assert relay.converged
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in relay.messages.values()
    )

    series = loop_series_expand(
        state.tn,
        2,
        norm="2norm",
        max_iterations=200,
        tol=1e-10,
    )
    assert series.bp_converged
    assert np.isfinite(float(np.real(series.estimate)))

    pne = partitioned_expand(
        state.tn,
        partition_inds=(bond,),
        norm="2norm",
        max_iterations=200,
        tol=1e-10,
    )
    assert pne.bp_converged
    assert np.isfinite(float(np.real(pne.estimate)))
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in pne.messages.values()
    )


def _closed_u1_scalar_network():
    """Build a small native Symmray scalar network for the D1 compatibility path."""
    maps = ({0: 0, 1: 1},) * 2
    left = sr.U1Array.from_dense(
        np.diag([2.0, 3.0]),
        index_maps=maps,
        duals=(False, True),
        charge=0,
    )
    right = sr.U1Array.from_dense(
        np.diag([5.0, 7.0]),
        index_maps=maps,
        duals=(True, False),
        charge=0,
    )
    return qtn.TensorNetwork(
        [
            qtn.Tensor(left, inds=("x", "y")),
            qtn.Tensor(right, inds=("x", "y")),
        ]
    )


def test_native_symmray_closed_scalar_network_works_through_one_norm_apis():
    """Valid closed 1-norm inputs use the dense-compatible BP shadow."""
    tn = _closed_u1_scalar_network()

    result = one_norm_bp(
        tn,
        method="d1bp",
        max_iterations=100,
        tol=1e-10,
    )
    assert result.converged
    assert all(type(tensor.data).__name__ == "ndarray" for tensor in result.bp.tn)

    cluster = loop_cluster_expand(
        tn,
        0,
        norm="1norm",
        max_iterations=100,
        tol=1e-10,
    )
    assert cluster.bp_converged
    assert np.isfinite(float(np.real(cluster.estimate)))

    series = loop_series_expand(
        tn,
        0,
        norm="1norm",
        max_iterations=100,
        tol=1e-10,
    )
    assert series.bp_converged
    assert np.isfinite(float(np.real(series.estimate)))

    pne = partitioned_expand(
        tn,
        partition_inds=("x",),
        norm="1norm",
        max_iterations=100,
        tol=1e-10,
    )
    assert pne.bp_converged
    assert np.isfinite(float(np.real(pne.estimate)))

    weights = weight_pass(tn, max_iterations=2)
    assert all(
        type(tensor.data).__name__ == "ndarray" for tensor in weights.network
    )
