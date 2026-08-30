"""Small exact cross-family comparisons for the organized operator API."""

import numpy as np
import pytest

from pepsy.operators import (
    ClusterExpansionPlan,
    MPOBasis,
    MPOClusterProductExpansion,
    PauliPEPOBasis,
    PEPOClusterProductExpansion,
)


pytestmark = pytest.mark.smoke
scipy_linalg = pytest.importorskip("scipy.linalg")


def _paulis():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    return x, z


def _sum_edge_operator(length, operator):
    identity = np.eye(2)
    result = np.zeros((2**length, 2**length))
    for site in range(length - 1):
        factors = [identity] * length
        factors[site] = operator[0]
        factors[site + 1] = operator[1]
        embedded = factors[0]
        for factor in factors[1:]:
            embedded = np.kron(embedded, factor)
        result += embedded
    return result


@pytest.mark.parametrize("length", (2, 3))
def test_two_and_three_site_operator_families_match_dense_references(length):
    """MPO, PEPO, cluster, and ordered-product paths share exact small cases."""
    x, z = _paulis()
    edge_x = _sum_edge_operator(length, (x, x))
    edge_z = _sum_edge_operator(length, (z, z))
    hamiltonian = edge_x + 0.4 * edge_z
    step = 0.02

    exact = scipy_linalg.expm(step * hamiltonian)
    ordered_exact = scipy_linalg.expm(step * edge_x) @ scipy_linalg.expm(
        0.4 * step * edge_z
    )

    local_terms = [
        ((site, site + 1), (x, x), 1.0)
        for site in range(length - 1)
    ] + [
        ((site, site + 1), (z, z), 0.4)
        for site in range(length - 1)
    ]
    mpo_cluster = MPOClusterProductExpansion.from_local_terms(
        length,
        local_terms,
        cluster_size=length,
        cutoff=0.0,
    ).exp(step).to_mpo().to_dense()

    mpo_x = MPOBasis.from_local_terms(
        length,
        [((site, site + 1), (x, x)) for site in range(length - 1)],
    )
    mpo_z = MPOBasis.from_local_terms(
        length,
        [((site, site + 1), (z, z)) for site in range(length - 1)],
    )
    mpo_product = MPOClusterProductExpansion.from_mpo_bases(
        (mpo_x, mpo_z),
        coefficients=(1.0, 0.4),
        cluster_size=length,
        cutoff=0.0,
    ).compile_exp().exp(step).to_mpo().to_dense()

    fixed_basis = PauliPEPOBasis.compile(
        length,
        1,
        [("edge", "XX"), ("edge", "ZZ")],
        order=length,
    )
    fixed_pepo = fixed_basis.exp(
        step,
        coefficients=(1.0, 0.4),
        materialize=True,
    ).to_dense()

    pepo_x = PauliPEPOBasis.compile(length, 1, [("edge", "XX")], order=length)
    pepo_z = PauliPEPOBasis.compile(length, 1, [("edge", "ZZ")], order=length)
    pepo_product = PEPOClusterProductExpansion.from_bases(
        (pepo_x, pepo_z),
        coefficients=(1.0, 0.4),
    ).compile_exp().exp(step).to_dense()

    local_hamiltonian = np.kron(x, x) + 0.4 * np.kron(z, z)
    dense_pepo = ClusterExpansionPlan(
        length,
        1,
        local_hamiltonian,
        np.zeros((2, 2)),
        order=length,
    ).build(step, materialize=True).to_dense()

    results = {
        "mpo_cluster": (mpo_cluster, exact),
        "mpo_product": (mpo_product, ordered_exact),
        "fixed_pepo": (fixed_pepo, exact),
        "pepo_product": (pepo_product, ordered_exact),
        # Dense PEPO's public beta convention is exp(-beta * H).
        "dense_pepo": (dense_pepo, scipy_linalg.expm(-step * hamiltonian)),
    }
    for name, (actual, expected) in results.items():
        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(actual, expected, atol=1.0e-11, err_msg=name)
