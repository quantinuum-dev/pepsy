"""Compare public Pepsy BP/cluster norms with exact nine-spin amplitudes."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import loop_cluster_expand, two_norm_bp
from pepsy.tensors import ps_to_peps


def exact_norm_squared(psi):
    """Independent single-layer oracle, without BP or boundary truncation."""
    vector = np.asarray(ar.to_numpy(psi.to_dense(optimize="auto-hq"))).reshape(-1)
    return np.vdot(vector, vector).real


def cluster_norms(psi):
    bp = two_norm_bp(psi, tol=1e-10, max_iterations=1000)
    assert bp.converged
    assert bp.max_mdiff < 1e-10
    clusters = loop_cluster_expand(
        psi, gloops=[], norm="2norm", messages=bp.snapshot(), run_bp=False
    )
    values = {
        0: complex(clusters.estimate),
        4: complex(clusters.expand(4)),
        9: complex(clusters.expand(9)),
    }
    np.testing.assert_allclose(values[0], complex(bp.contract()), rtol=1e-11)
    for value in values.values():
        assert np.isfinite(value)
        assert value.real > 0
        assert abs(value.imag) < 1e-10 * value.real
    return values


def test_scaled_3x3_product_norm_is_exact_already_at_bp():
    psi = ps_to_peps(3, 3, dtype="complex128")
    psi.multiply_(3.0 * np.exp(0.41j))
    expected = exact_norm_squared(psi)
    assert expected == pytest.approx(9.0, rel=1e-12)
    for value in cluster_norms(psi).values():
        np.testing.assert_allclose(value, expected, rtol=1e-11)


def test_3x3_tree_peps_norm_is_exact_already_at_bp():
    psi = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=4)
    path = [(i, j) for i in range(3)
            for j in (range(3) if i % 2 == 0 else range(2, -1, -1))]
    tree_edges = {frozenset((a, b)) for a, b in zip(path, path[1:])}
    # Keep a nontrivially entangled snake tree, with the remaining lattice
    # bonds reduced to dimension one. Preserve the PEPS geometry and tags.
    for a, b in psi.gen_bond_coos():
        if frozenset((a, b)) in tree_edges:
            continue
        index = psi.bond(a, b)
        for site in (a, b):
            tensor = psi[site]
            tensor.modify(data=np.take(tensor.data, [0], axis=tensor.inds.index(index)))
    expected = exact_norm_squared(psi)
    for value in cluster_norms(psi).values():
        np.testing.assert_allclose(value, expected, rtol=1e-10)


@pytest.mark.parametrize("seed,backend", [(0, "numpy"), (1, "numpy"),
                                         (2, "numpy"), (0, "torch")])
def test_3x3_random_peps_full_cluster_exact_and_scale_preserved(seed, backend):
    psi = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=seed)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        psi.apply_to_arrays(lambda data: torch.as_tensor(data.copy(), device="cpu"))
    original = [ar.to_numpy(t.data).copy() for t in psi]
    expected = exact_norm_squared(psi)
    values = cluster_norms(psi)
    # C=0 and C=4 are approximations on this grid. C=9 covers the whole
    # system and must agree with all 512 independently contracted amplitudes.
    np.testing.assert_allclose(values[9], expected, rtol=1e-11)
    for tensor, before in zip(psi, original):
        np.testing.assert_array_equal(ar.to_numpy(tensor.data), before)

    scaled = psi.multiply(3.0 * np.exp(0.41j))
    scaled_values = cluster_norms(scaled)
    np.testing.assert_allclose(exact_norm_squared(scaled), 9 * expected, rtol=1e-11)
    for cutoff in values:
        np.testing.assert_allclose(scaled_values[cutoff], 9 * values[cutoff], rtol=1e-9)
