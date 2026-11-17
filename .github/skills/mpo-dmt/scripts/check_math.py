#!/usr/bin/env python3
"""Independent dense checks of the DMT skill's mathematics; no state writes."""

from __future__ import annotations

import argparse

import numpy as np


def coefficients(operator, n):
    """Interleave ket/bra indices into local matrix-unit coordinates."""
    order = tuple(j for i in range(n) for j in (i, n + i))
    return operator.reshape((2,) * (2 * n)).transpose(order).reshape((4,) * n)


def operator_from_coefficients(tensor, n):
    order = tuple(range(0, 2 * n, 2)) + tuple(range(1, 2 * n, 2))
    return tensor.reshape((2,) * (2 * n)).transpose(order).reshape(2**n, 2**n)


def marginal(operator, n, kept):
    """Physical partial trace, independent of virtual protection maps."""
    tensor = operator.reshape((2,) * (2 * n))
    remaining = n
    for site in reversed(range(n)):
        if site not in kept:
            tensor = np.trace(tensor, axis1=site, axis2=site + remaining)
            remaining -= 1
    return tensor.reshape(2**remaining, 2**remaining)


def check_physical_constraints():
    rng = np.random.default_rng(412)
    n, cap = 6, 16
    sample = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    rho = sample @ sample.conj().T
    rho /= np.trace(rho)
    left, spectrum, right = np.linalg.svd(coefficients(rho, n).reshape(64, 64))
    matrix = np.diag(spectrum).astype(complex)
    assert matrix.shape == (4 * cap, 4 * cap)
    tau = np.array([1., 0., 0., 1.])
    fl = left.reshape(4, 4, 4, 64)
    fr = right.reshape(64, 4, 4, 4)
    for _ in range(2):
        fl = np.tensordot(tau, fl, axes=(0, 0))
        fr = np.tensordot(fr, tau, axes=(-1, 0))
    fr = fr.T

    # Arbitrary changes test the substitution identities, not only their zeros.
    change = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    physical = operator_from_coefficients(left @ change @ right, n)
    left_change = operator_from_coefficients(fl @ change @ right, 4)
    right_change = operator_from_coefficients(left @ change @ fr.T, 4)
    np.testing.assert_allclose(marginal(physical, n, range(2, 6)), left_change, atol=1e-12)
    np.testing.assert_allclose(marginal(physical, n, range(4)), right_change, atol=1e-12)
    wrong = operator_from_coefficients(left @ change @ fr.conj().T, 4)
    assert np.linalg.norm(wrong - right_change) > 1e-3

    p, q = np.linalg.matrix_rank(fl), np.linalg.matrix_rank(fr)
    ql = np.linalg.svd(fl.conj().T, full_matrices=False)[0][:, :p]
    qr = np.linalg.svd(fr.T, full_matrices=False)[0][:, :q]
    pl = np.eye(64) - ql @ ql.conj().T
    pr = np.eye(64) - qr @ qr.conj().T
    tl, tr = tau @ fl, tau @ fr
    baseline = np.outer(matrix @ tr, tl @ matrix) / (tl @ matrix @ tr)
    residual = pl @ (matrix - baseline) @ pr
    u, s, vh = np.linalg.svd(residual)
    rank = cap - p - q
    embedded = matrix - residual + (u[:, :rank] * s[:rank]) @ vh[:rank]
    change = embedded - matrix
    np.testing.assert_allclose(fl @ change, 0., atol=1e-12)
    np.testing.assert_allclose(change @ fr.T, 0., atol=1e-12)
    assert np.linalg.matrix_rank(embedded, tol=1e-11) <= cap

    uc, s_small, vc_dag = np.linalg.svd(embedded)
    uc, small, vc_dag = uc[:, :cap], np.diag(s_small[:cap]), vc_dag[:cap]
    assert uc.shape == (4 * cap, cap) and small.shape == (cap, cap)
    np.testing.assert_allclose(uc @ small @ vc_dag, embedded, atol=1e-12)
    compressed_coefficients = (left @ uc) @ small @ (vc_dag @ right)
    compressed = operator_from_coefficients(compressed_coefficients, n)
    np.testing.assert_allclose(np.linalg.norm(compressed - rho), np.linalg.norm(change), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(change)**2, np.sum(s[rank:]**2), atol=1e-12)
    worst = 0.
    for first in range(4):
        region = range(first, first + 3)
        defect = np.linalg.norm(marginal(compressed, n, region) - marginal(rho, n, region))
        worst = max(worst, defect)
        assert defect < 1e-12

    pauli = np.array([np.eye(2), [[0, 1], [1, 0]],
                      [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex) / np.sqrt(2)
    transform = pauli.conj().reshape(4, 4)
    np.testing.assert_allclose(transform @ transform.conj().T, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(np.array([np.sqrt(2), 0, 0, 0]) @ transform, tau, atol=1e-14)
    transform3 = np.kron(np.kron(transform, transform), transform)
    pauli_coefficients = (transform3 @ compressed_coefficients @ transform3.T).reshape((4,) * n)
    for first in range(4):
        selection = tuple(slice(None) if first <= j < first + 3 else 0 for j in range(n))
        reduced = pauli_coefficients[selection].reshape(64) * np.sqrt(2)**3
        physical = operator_from_coefficients(transform3.conj().T @ reduced, 3)
        np.testing.assert_allclose(physical, marginal(compressed, n, range(first, first + 3)), atol=1e-12)
    print(f"Complex six-site trace identities, dimensions, tails and both bases: passed ({worst:.3g} marginal defect)")


def check_completion_and_counterexample():
    rng = np.random.default_rng(27)
    p, n, cap = 3, 14, 7
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    matrix[:p, :p] += 5 * np.eye(p)
    shift = matrix[p:, :p] @ np.linalg.solve(matrix[:p, :p], matrix[:p, p:])
    u, s, vh = np.linalg.svd(matrix[p:, p:] - shift)
    rank = cap - p
    optimum = matrix.copy()
    optimum[p:, p:] = shift + (u[:, :rank] * s[:rank]) @ vh[:rank]
    np.testing.assert_allclose(optimum[:p], matrix[:p])
    np.testing.assert_allclose(optimum[:, :p], matrix[:, :p])
    assert np.linalg.matrix_rank(optimum, tol=1e-10) == cap
    np.testing.assert_allclose(np.linalg.norm(optimum - matrix)**2, np.sum(s[rank:]**2), atol=1e-11)

    matrix = np.diag([1., .3, .1])
    dmt = np.diag([1., 0., 0.])
    better = np.diag([1., .3, 0.])
    for candidate in (dmt, better):
        np.testing.assert_allclose(candidate[0], matrix[0])
        np.testing.assert_allclose(candidate[:, 0], matrix[:, 0])
        assert np.linalg.matrix_rank(candidate) <= 2
    np.testing.assert_allclose(np.linalg.norm(dmt - matrix)**2, .10)
    np.testing.assert_allclose(np.linalg.norm(better - matrix)**2, .01)
    print("Invertible-corner completion and conservative-DMT counterexample: passed")


def check_pepsy_kernel():
    # Private maintenance probe: recheck this signature when the kernel changes.
    from pepsy.operators import _pauli_dmt

    matrix = np.diag([1., .3, .1])
    vector = np.array([[1.], [0.], [0.]])
    left, right, _ = _pauli_dmt._split_scaled(
        matrix, vector, vector, max_bond=2, cutoff=0., variant="density",
        absorb="right", bond=0, diagnostics=True,
    )
    np.testing.assert_allclose(left @ right, np.diag([1., 0., 0.]), atol=1e-13)
    print(f"Current Pepsy conservative-budget example: passed ({_pauli_dmt.__file__})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-kernel", action="store_true", help="also check the installed Pepsy splitting kernel")
    args = parser.parse_args()
    check_physical_constraints()
    check_completion_and_counterexample()
    if args.check_kernel:
        check_pepsy_kernel()
