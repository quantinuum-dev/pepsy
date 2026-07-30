"""Symmray fermionic coverage for Pepsy's 2-norm BP corrections."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("symmray")

from pepsy.bp import gauge_all, loop_cluster_expand, two_norm_bp  # noqa: E402
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
