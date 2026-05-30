"""Helper functions for the MPS simulator notebook."""

import math
import numpy as np
import quimb
import quimb.tensor as qtn
import pepsy as py
from pepsy.tensors import OneDMap, expec_mpo


def build_lattice(Lx, Ly, coupling_j, field_h, cyclic=True, lattice="square", mode="hilbert"):
    """Build the 2D ITF lattice: mapper, Hamiltonian MPO, edges, sites.

    Returns
    -------
    dict with keys: mpo_H, edges_1d, sites, mapper, res (full build_itf_lattice output).
    """
    mapper = OneDMap(Lx, Ly, mode="hilbert")
    res = py.ham_tn.build_itf_lattice(
        L_x=Lx, L_y=Ly, lattice=lattice, cyclic=cyclic,
        J=coupling_j, field=field_h,
        mapper=mapper,
    )
    mpo_H = res["mpo"]
    edges_1d = res["edges_1d"]
    sites = sorted({(site,) for edge in edges_1d for site in edge})
    return {
        "mpo_H": mpo_H,
        "edges_1d": edges_1d,
        "sites": sites,
        "mapper": mapper,
        "res": res,
    }


def build_initial_state(L, coupling_j, field_h, z_coord=4, theta_offset=None):
    """Build the intermediate-temperature product state (arXiv:2503.20870).

    Parameters
    ----------
    theta_offset : float or None
        Offset added to arcsin(h/(J·z)). Default is 2π/9 (paper value).

    Returns
    -------
    psi0 : quimb.tensor.MatrixProductState
    theta_paper : float
    """
    if theta_offset is None:
        theta_offset = 2 * math.pi / 9
    ratio = field_h / (coupling_j * z_coord)
    sin2theta = np.clip(ratio, -1.0, 1.0)
    theta_min = np.arcsin(sin2theta)
    theta_paper = theta_min + theta_offset
    psi0 = py.ps_to_mps(L, theta=0.5 * theta_paper)
    return psi0, theta_paper


def build_trotter_gates(sites, edges_1d, field_h, coupling_j, dt, to_backend):
    """Build 2nd-order Strang-splitting Trotter gate stream.

    RX(h·dt) → RZZ(2J·dt) → RX(h·dt)

    Convention: py.rx(θ) = exp(-i θ X / 2), so the half-step
    exp(-i h·dt/2 · X) requires rx(h·dt).

    Returns
    -------
    list of (gate, where) tuples.
    """
    rx_half = to_backend(py.rx(field_h * dt))
    rzz_full = to_backend(py.rzz(2 * coupling_j * dt))

    gates = []
    for site in sites:
        gates.append((rx_half, site))
    for edge in edges_1d:
        gates.append((rzz_full, edge))
    for site in sites:
        gates.append((rx_half, site))
    return gates


def build_mpo_z_sq(Lx, Ly, mapper):
    """Build the (ΣZ_i/L)² MPO for measuring magnetization squared.

    Returns
    -------
    mpo_z_sq_offdiag : quimb.tensor.MatrixProductOperator
        Off-diagonal part: (1/L²) Σ_{i≠j} Z_i Z_j
    diagonal_shift : float
        Constant 1/L to add (from diagonal Z_i²=I terms).
    mpo_z : quimb.tensor.MatrixProductOperator
        The magnetization MPO M = (1/L) Σ Z_i (useful on its own).
    """
    L = Lx * Ly
    Z_op = quimb.pauli("Z", dtype="complex128")

    builder = py.ham_tn(Lx=Lx, Ly=Ly, mapper=mapper, data_type="complex128")

    # Sites in 2D coordinates (what build_mpo expects)
    res = py.ham_tn.build_itf_lattice(
        L_x=Lx, L_y=Ly, lattice="square", cyclic=True,
        J=-1, field=1, mapper=mapper,
    )
    all_sites_2d = list(res["one_d_to_two_d"].values())

    # M = (1/L) Σ_i Z_i
    z_terms = [((Z_op,), (site,), 1.0 / L) for site in all_sites_2d]
    mpo_z = builder.build_mpo(z_terms, compress_each=True)

    # M² off-diagonal = (1/L²) Σ_{i≠j} Z_i Z_j
    z_sq_terms = []
    for i, site_i in enumerate(all_sites_2d):
        for j, site_j in enumerate(all_sites_2d):
            if i != j:
                z_sq_terms.append(
                    ((Z_op, Z_op), (site_i, site_j), 1.0 / (L * L))
                )

    mpo_z_sq_offdiag = builder.build_mpo(z_sq_terms, compress_each=True)
    diagonal_shift = 1.0 / L  # from Z_i² = I

    return mpo_z_sq_offdiag, diagonal_shift, mpo_z


def measure_energy(psi, mpo_H, L, optimizer):
    """<H>/L via MPO."""
    e = expec_mpo(mpo_H, psi, contraction_opt=optimizer)
    return float(np.real(e)) / L


def measure_z(psi, mpo_z, optimizer):
    """<ΣZ/L> = <M> via MPO."""
    m = expec_mpo(mpo_z, psi, contraction_opt=optimizer)
    return float(np.real(m))


def measure_z_sq(psi, mpo_z_sq_offdiag, diagonal_shift, optimizer):
    """<(ΣZ/L)²> = <M²> via MPO + diagonal shift."""
    off_diag = expec_mpo(mpo_z_sq_offdiag, psi, contraction_opt=optimizer)
    return float(np.real(off_diag)) + diagonal_shift
