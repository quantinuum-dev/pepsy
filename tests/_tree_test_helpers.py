"""Shared small reference builders for tree regression tests."""


import numpy as np
import quimb.tensor as qtn


def _sv_apply_1q(psi, g, q, n):
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q).reshape(-1)


def _sv_apply_2q(psi, g, a, b, n):
    g = g.reshape(2, 2, 2, 2)
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=([2, 3], [a, b]))
    return np.moveaxis(psi, [0, 1], [a, b]).reshape(-1)


def _sv_apply_kq(psi, g, where, n):
    """Exact statevector application of a ``k``-qubit operator on ``where``."""
    k = len(where)
    g = np.asarray(g).reshape([2] * (2 * k))
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(where)))
    return np.moveaxis(psi, range(k), where).reshape(-1)


def _rand_unitary(k, rng):
    m = rng.standard_normal((2**k, 2**k)) + 1j * rng.standard_normal((2**k, 2**k))
    q, _ = np.linalg.qr(m)
    return q


def _random_stream(n, ngates, rng, two_qubit_frac=0.5):
    stream = []
    for _ in range(ngates):
        if n >= 2 and rng.random() < two_qubit_frac:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((_rand_unitary(2, rng), (int(a), int(b))))
        else:
            stream.append((_rand_unitary(1, rng), int(rng.integers(n))))
    return stream


def _two_branch_flip_submpo(*, L, sites, targets, w0=0.7, w1=0.3):
    """Return ``w0 * I + w1 * prod(X_targets)`` as a sparse-site MPO."""
    eye = np.eye(2, dtype=complex)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
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


def _exact_state(stream, n):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    for g, where in stream:
        if isinstance(where, int):
            psi = _sv_apply_1q(psi, g, where, n)
        else:
            psi = _sv_apply_2q(psi, g, where[0], where[1], n)
    return psi


def _fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2 / (
        np.vdot(a, a).real * np.vdot(b, b).real
    )
