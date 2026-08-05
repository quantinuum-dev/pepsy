"""Tests for the compact native eta-pair structure-factor MPO."""

import numpy as np
import pytest

import pepsy
from pepsy.tensors import Fermion, SymMPS, site_charge_from_occupations


pytest.importorskip("symmray")


def _dense_mpo(mpo, length):
    tensor = mpo.copy().contract(all)
    dense = tensor.to_dense(
        [f"k{site}" for site in range(length)],
        [f"b{site}" for site in range(length)],
    )
    if hasattr(dense, "to_dense"):
        dense = dense.to_dense()
    return np.asarray(dense)


def _explicit_structure_factor_terms(fermion, signs, normalization):
    terms = {}
    length = len(signs)
    for left in range(length):
        for right in range(left + 1, length):
            terms[(left, right)] = (
                normalization
                * signs[left]
                * signs[right]
                * fermion.operator_term(
                    [
                        (
                            1.0,
                            ((left, "pair_create"), (right, "pair_annihilate")),
                        ),
                        (
                            1.0,
                            ((right, "pair_create"), (left, "pair_annihilate")),
                        ),
                    ],
                    sites=(left, right),
                    charge=fermion.zero_charge,
                )
            )
    return terms


def test_factorized_eta_pair_mpo_matches_explicit_native_mpo():
    length = 4
    signs = (1, -1, -1, 1)
    normalization = 1.0 / length
    fermion = Fermion(spinful=True, symmetry="U1", dtype="complex128")

    compact = fermion.eta_pair_structure_factor_mpo(
        length,
        signs=signs,
        normalization=normalization,
    )
    explicit = fermion.hamiltonian(
        _explicit_structure_factor_terms(fermion, signs, normalization)
    ).to_mpo(L=length, fermionic=True, compress=False)

    np.testing.assert_allclose(
        _dense_mpo(compact, length),
        _dense_mpo(explicit, length),
        atol=1e-12,
    )
    assert compact.max_bond() == 4
    assert compact.pepsy_compression_report["factorized_pair"] is True
    assert compact.pepsy_compression_report["pair_term_count"] == 6

    state = SymMPS.random(
        length,
        symmetry="U1",
        fermionic=True,
        phys_dim=fermion.physical_sectors,
        site_charge=site_charge_from_occupations([1] * length),
        bond_dim=3,
        seed=7,
        dtype="complex128",
    )
    compact_value = pepsy.MpsEnergyOptimizer(
        state,
        compact,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=False,
    ).energy().energy
    explicit_value = pepsy.MpsEnergyOptimizer(
        state,
        explicit,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=False,
    ).energy().energy
    assert complex(compact_value) == pytest.approx(complex(explicit_value))


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_factorized_eta_pair_mpo_preserves_charge_rank(symmetry):
    fermion = Fermion(spinful=True, symmetry=symmetry, dtype="complex64")
    mpo = fermion.eta_pair_structure_factor_mpo(
        6,
        signs=(1, -1, 1, -1, 1, -1),
        normalization=1.0 / 6.0,
    )

    report = mpo.pepsy_compression_report
    assert mpo.max_bond() == 4
    assert report["raw_max_bond"] == 4
    assert report["final_max_bond"] == 4
    assert report["pair_term_count"] == 15
