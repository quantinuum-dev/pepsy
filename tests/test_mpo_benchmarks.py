"""Small dense accuracy benchmarks for higher-order MPO baselines.

These are intentionally regression-sized rather than performance harnesses:
they compare the finite MPO construction with first-order Trotter and a
two-site cluster-expansion baseline on a four-site chain. Larger timing runs
belong outside the package repository.
"""

from itertools import product

import numpy as np
import pytest

from pepsy.operators import FirstDegreeMPO, MPOProductTerm


def _paulis():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    return x, z


def _embedded_product(L, site, left, right):
    identity = np.eye(left.shape[0])
    factors = [identity for _ in range(L)]
    factors[site] = left
    factors[site + 1] = right
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _baseline_operators(dt):
    scipy_linalg = pytest.importorskip("scipy.linalg")
    x, z = _paulis()
    L = 4
    local_terms = (
        (0, x, x),
        (1, z, z),
        (2, x, z),
    )
    local_operators = tuple(
        _embedded_product(L, site, left, right)
        for site, left, right in local_terms
    )
    hamiltonian = sum(local_operators)
    exact = scipy_linalg.expm(dt * hamiltonian)

    # First-order Lie-Trotter: product of exact exponentials of the local
    # two-site terms in a fixed left-to-right ordering.
    trotter = np.eye(2**L)
    for local_operator in local_operators:
        trotter = scipy_linalg.expm(dt * local_operator) @ trotter

    # The p=2 cluster baseline keeps each connected two-site exponential
    # correction and all products whose supports are disjoint. Overlapping
    # connected clusters are omitted, as in a finite truncation of the
    # cluster expansion.
    corrections = tuple(
        scipy_linalg.expm(dt * local_operator) - np.eye(2**L)
        for local_operator in local_operators
    )
    cluster = np.zeros_like(exact)
    for selected_mask in product((False, True), repeat=len(corrections)):
        selected = tuple(
            index for index, chosen in enumerate(selected_mask) if chosen
        )
        if any(right - left == 1 for left, right in zip(selected, selected[1:])):
            continue
        term = np.eye(2**L)
        for index in selected:
            term = corrections[index] @ term
        cluster += term

    return L, local_terms, exact, trotter, cluster


def test_higher_order_mpo_accuracy_benchmark_against_trotter_and_cluster():
    """Report deterministic errors against both finite-chain baselines."""
    dt = 0.08
    L, local_terms, exact, trotter, cluster = _baseline_operators(dt)
    x, z = _paulis()
    H = FirstDegreeMPO.from_local_terms(
        L,
        [
            MPOProductTerm((site, site + 1), (left, right))
            for site, left, right in local_terms
        ],
    )

    errors = {}
    for order in (1, 2, 3):
        U = H.extensive_exponential(
            dt,
            order=order,
            cache_history=False,
        )
        errors[f"mpo_order_{order}"] = np.linalg.norm(
            U.to_mpo().to_dense() - exact,
        )
    errors["trotter_first_order"] = np.linalg.norm(trotter - exact)
    errors["cluster_two_site"] = np.linalg.norm(cluster - exact)

    # The MPO's Taylor order controls the expected convergence, while the
    # baselines are retained as independent reference methods rather than
    # being asserted to have the same error ordering.
    assert errors["mpo_order_3"] < errors["mpo_order_2"] < errors["mpo_order_1"]
    assert all(np.isfinite(value) for value in errors.values())
    assert errors["trotter_first_order"] > 0.0
    assert errors["cluster_two_site"] > 0.0
    assert x.shape == z.shape == (2, 2)
