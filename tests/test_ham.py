"""Regression tests for :mod:`pepsy.ham` builders."""

import quimb

import pepsy as py


def test_build_mpo_single_site_term_works_for_ly1():
    """Single-term MPO build should work for 1D (Ly=1) layouts."""
    builder = py.ham_tn(L_x=2, L_y=1, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    mpo = builder.build_mpo(
        [
            (((0, 0),), (z_op,)),
        ],
        compress_each=False,
    )

    assert mpo.L == 2


def test_build_itf_lattice_ly1_has_chain_edges():
    """Ly=1 ITF lattice should reduce to a nearest-neighbor 1D chain."""
    out = py.ham_tn.build_itf_lattice(
        L_x=5,
        L_y=1,
        lattice="square",
        J=1.0,
        field=0.5,
        return_edges=True,
    )

    assert out["builder"].L_x == 5
    assert out["builder"].L_y == 1
    assert out["mpo"].L == 5
    assert out["pepo"].Lx == 5
    assert out["pepo"].Ly == 1
    assert all(y == 0 for _, y in out["one_d_to_two_d"].values())

    expected = {frozenset((i, i + 1)) for i in range(4)}
    got = {frozenset(edge) for edge in out["edges_1d"]}
    assert got == expected
