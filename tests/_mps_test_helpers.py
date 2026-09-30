"""Shared small reference builders for MPS regression tests."""


import numpy as np
import quimb.tensor as qtn


def _non_unitary_entangling_gate():
    """Return a small two-site filter that creates entanglement from |++>."""
    return np.diag([1.0, 0.5, 0.5, 2.0]).astype(complex)


def _two_branch_flip_submpo(*, L, sites, targets, w0=0.7, w1=0.3):
    """Return ``w0 * I + w1 * prod(X_targets)`` as a sparse-site MPO."""
    eye = np.eye(2, dtype=np.complex128)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sites = tuple(sites)
    targets = set(targets)
    branch0 = [eye.copy() for _site in sites]
    branch1 = [flip.copy() if site in targets else eye.copy() for site in sites]
    branch0[0] *= w0
    branch1[0] *= w1
    mpo0 = qtn.MPO_product_operator(
        branch0,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    mpo1 = qtn.MPO_product_operator(
        branch1,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    return mpo0.add_MPO(mpo1)


def _mps_data_norm(mps):
    """Return the MPS norm without its stored global exponent."""
    mps_data = mps.copy()
    mps_data.exponent = 0.0
    return mps_data.norm()
