"""Regression coverage for native fermionic PEPS boundary contraction."""

import numpy as np
import pytest

import pepsy


torch = pytest.importorskip("torch")


def _torch_float64(array):
    return torch.as_tensor(array, dtype=torch.float64)


def _untruncated_boundary_norm(state, *, strip_exponent=False):
    return (state.H & state).contract_boundary(
        max_bond=None,
        cutoff=0.0,
        canonize=True,
        mode="mps",
        sequence=("xmin", "xmax", "ymin", "ymax"),
        equalize_norms=False,
        final_contract_opts={
            "optimize": "greedy",
            "strip_exponent": strip_exponent,
        },
        inplace=False,
    )


@pytest.mark.integration
def test_u1u1_fermionic_boundary_qr_preserves_untruncated_norm():
    """Graded QR/LQ remains a lossless gauge move for native PEPS.

    One autodiff step installs the same raw-Torch Quimb split drivers used by
    ``PepsEnergyOptimizer``.  The subsequent boundary contraction must still
    agree with the direct norm when neither cutoff nor boundary bond limit is
    requested.
    """
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        to_backend=_torch_float64,
    )
    setup = fermion.lattice_half_filling(2, 4, cyclic=False)
    state = pepsy.hrs_to_peps(
        2,
        4,
        fermion=fermion,
        occupations=setup.occupations,
        chi=4,
        seed=83,
        dtype="float64",
        cyclic=False,
        normalize=True,
        to_backend=_torch_float64,
    )
    hamiltonian = fermion.hamiltonian(setup.edges, t=1.0, U=8.0, mu=0.0)
    optimizer = pepsy.PepsEnergyOptimizer(
        state,
        hamiltonian.terms,
        chi=16,
        boundary_mode="mps",
        cutoff=0.0,
        contraction_opt="greedy",
        compute_kwargs={
            "first_contract": "x",
            "second_dense": True,
            "canonize": False,
        },
    )
    state, losses = optimizer.optimize(
        n=1,
        optimizer="adam",
        autodiff_backend="torch",
        fallback_boundary_mode=None,
        progbar=False,
        return_losses=True,
    )
    assert losses
    assert np.isfinite(np.asarray(losses, dtype=float)).all()

    reference = (state.H & state).contract(all, optimize="greedy")
    for strip_exponent in (False, True):
        estimate = _untruncated_boundary_norm(
            state,
            strip_exponent=strip_exponent,
        )
        if strip_exponent:
            mantissa, exponent = estimate
            estimate = mantissa * (10.0 ** exponent)
        relative_error = abs(complex(estimate / reference) - 1.0)

        assert relative_error < 1.0e-10
        assert np.isfinite(relative_error)
