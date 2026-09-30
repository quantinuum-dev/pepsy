"""Independent operator checks for QR-only MPO delinearisation."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import (
    MPOClusterProductExpansion, MPOProductTerm, delinearize_mpo,
    prepare_cluster_channels,
)
from pepsy.operators.mpo_delinearize import _column_basis

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def without_svd():
    stack = ExitStack()
    for name in ('numpy.linalg.svd', 'scipy.linalg.svd', 'numpy.linalg.lstsq'):
        stack.enter_context(patch(name, side_effect=AssertionError('SVD path')))
    return stack


def dependent_operator(dtype):
    eye = np.eye(2, dtype=dtype)
    x = np.array([[0, 1], [1, 0]], dtype=dtype)
    z = np.diag([1, -1]).astype(dtype)
    a, b = (2+3j, -.4j) if np.issubdtype(dtype, np.complexfloating) else (2., -.4)
    left = np.stack([eye, x, a*eye+b*x])
    right = np.stack([z, eye, x])
    mpo = qtn.MatrixProductOperator([left, right], shape='lrud',
                                    upper_ind_id='out{}', lower_ind_id='in{}', site_tag_id='site{}')
    mpo[0].add_tag('retain-me')
    reference = np.kron(eye, z)+np.kron(x, eye)+np.kron(a*eye+b*x, x)
    return mpo, reference


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_nonparallel_linear_combination_is_removed_without_svd(dtype):
    mpo, reference = dependent_operator(dtype)
    original = [a.copy() for a in mpo.arrays]
    eps = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
    with without_svd():
        result, report = delinearize_mpo(mpo, rtol=100*eps, return_report=True)
    assert result.bond_sizes() == [2]
    assert report.initial_bond_dimensions == (3,)
    assert report.final_bond_dimensions == (2,)
    assert report.converged
    assert all(a.dtype == dtype for a in result.arrays)
    assert set(result.tags) == set(mpo.tags)
    assert set(result.outer_inds()) == set(mpo.outer_inds())
    for a, b in zip(mpo.arrays, original):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(result.to_dense(), reference, atol=300*eps, rtol=300*eps)
    assert result.pepsy_first_degree is None
    assert result.pepsy_block_plan is None


def test_zero_policy_is_explicit_and_weak_independent_columns_survive():
    matrix = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., -1.]])
    with without_svd():
        _, transfer, _ = _column_basis(matrix, 1e-13, True)
        assert transfer is None  # dependencies here require zero cancellation
        basis, transfer, _ = _column_basis(matrix, 1e-13, False)
        assert basis.shape[1] == 2
        np.testing.assert_allclose(basis @ transfer, matrix, atol=1e-14)
        weak = np.diag([1., 1e-25])
        _, transfer, _ = _column_basis(weak, 1e-10, False)
        assert transfer is None  # no absolute small-value pruning


def test_weak_channel_with_large_neighbor_is_not_lost():
    eye = np.eye(2)
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    mpo = qtn.MatrixProductOperator([np.stack([eye, 1e-20*x]),
                                    np.stack([eye, 1e20*z])], shape='lrud')
    with without_svd():
        result = delinearize_mpo(mpo)
    np.testing.assert_allclose(result.to_dense(), np.kron(eye, eye)+np.kron(x, z), atol=1e-14)
    assert result.bond_sizes() == [2]


@pytest.mark.parametrize('length', [1, 2, 4])
def test_identity_zero_and_single_site_boundaries(length):
    for scale in (0., 1., .3):
        mpo = (qtn.MatrixProductOperator([np.eye(2, dtype=complex)], shape='ud')
               if length == 1 else qtn.MPO_identity(length, dtype='complex128'))
        mpo[0].modify(data=mpo[0].data*scale)
        with without_svd():
            result = delinearize_mpo(mpo)
        np.testing.assert_allclose(result.to_dense(), scale*np.eye(2**length), atol=1e-14)
        assert result.bond_sizes() == [1]*(length-1)


@pytest.mark.parametrize('boundary', ['OBC', 'PBC'])
@pytest.mark.parametrize('joint', [False, True])
def test_frontier_delinearisation_preserves_complete_cluster_operator(boundary, joint):
    from scipy.linalg import expm

    n = 4
    edges = sorted({tuple(sorted((i, (i+r) % n))) for r in (1, 2) for i in range(n)
                    if boundary == 'PBC' or i+r < n})
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1., -1.])
    pauli = {'X': x, 'Y': y, 'Z': z}
    axes = [('X', 'Z'), ('Y', 'Z'), ('Z', 'X')] if joint else [('Z', 'X')]
    factors, matrices = [], []
    for bond, field in axes:
        terms = [MPOProductTerm.from_pauli((i,), field, coefficient=.4) for i in range(n)]
        terms += [MPOProductTerm.from_pauli(edge, bond*2, coefficient=.7) for edge in edges]
        factors.append(terms)
        h = np.zeros((2**n, 2**n), dtype=complex)
        for sites, axis, coefficient in [*((edge, bond, .7) for edge in edges),
                                         *(([i], field, .4) for i in range(n))]:
            product = np.array([[1.]])
            for i in range(n):
                product = np.kron(product, pauli[axis] if i in sites else np.eye(2))
            h += coefficient*product
        matrices.append(h)
    owner = MPOClusterProductExpansion(n, factors, graph=(range(n), edges), cluster_size=n,
        factorization='fixed', cutoff=0., graph_assembly='exact', assembly='recursive')
    target = np.eye(2**n, dtype=complex)
    for h in matrices:
        target = target @ expm(-.08j*h)
    with without_svd():
        channels = prepare_cluster_channels(owner, -.08j, preparation='frontier')
        frontier = channels.exp(-.08j)
        result, report = delinearize_mpo(frontier, preserve_zeros=False, return_report=True)
    np.testing.assert_allclose(result.to_dense(), target, atol=2e-11, rtol=2e-11)
    assert all(a <= b for a, b in zip(report.final_bond_dimensions, report.initial_bond_dimensions))
    assert max(report.final_bond_dimensions) < max(report.initial_bond_dimensions)


@pytest.mark.parametrize('time', [.03, .06, .12])
def test_periodic_nnn_frontier_does_not_amplify_transfer_roundoff(time):
    n = 6
    weights = {tuple(sorted((i, (i+r) % n))): value
               for r, value in ((1, 1.), (2, .5)) for i in range(n)}
    terms = [MPOProductTerm.from_pauli((i,), 'X') for i in range(n)]
    terms += [MPOProductTerm.from_pauli(edge, 'ZZ', coefficient=v) for edge, v in weights.items()]
    owner = MPOClusterProductExpansion(n, [terms], graph=(range(n), tuple(weights)), cluster_size=2,
        factorization='fixed', cutoff=0., assembly='recursive', graph_assembly='exact',
        spatial_symmetries=[tuple(reversed(range(n))), tuple((i+1) % n for i in range(n))])
    with without_svd():
        plan = prepare_cluster_channels(owner, -1j*time, preparation='frontier')
        frontier = plan.exp(-1j*time)
        result = delinearize_mpo(frontier)
    expected = frontier.to_dense()
    # Keeping small columns first previously produced huge transfer weights
    # and a 2.7e-9 relative matrix error at t=.06 despite local fits <1e-12.
    relative = np.linalg.norm(result.to_dense()-expected)/np.linalg.norm(expected)
    assert relative < 2e-12
    assert result.max_bond() < frontier.max_bond()


@pytest.mark.parametrize('kwargs, error', [
    ({'rtol': -1}, ValueError), ({'rtol': np.nan}, ValueError),
    ({'rtol': np.inf}, ValueError), ({'preserve_zeros': 1}, TypeError),
    ({'rtol': True}, ValueError), ({'rtol': 1j}, ValueError),
    ({'max_sweeps': 0}, ValueError), ({'max_sweeps': True}, ValueError),
])
def test_invalid_options(kwargs, error):
    with pytest.raises(error):
        delinearize_mpo(qtn.MPO_identity(2), **kwargs)


def test_backend_and_topology_guards_do_not_coerce():
    with pytest.raises(TypeError, match='Quimb'):
        delinearize_mpo(np.eye(4))
    with pytest.raises(NotImplementedError, match='open-chain'):
        delinearize_mpo(qtn.MPO_identity(3, cyclic=True))
    mpo = qtn.MPO_identity(2)
    mpo[0].modify(data=mpo[0].data.astype(int))
    with pytest.raises(TypeError, match='dense NumPy'):
        delinearize_mpo(mpo)
    mpo = qtn.MPO_identity(2)
    mpo[0].modify(data=np.full_like(mpo[0].data, np.nan))
    with pytest.raises(ValueError, match='finite'):
        delinearize_mpo(mpo)


def test_autodiff_arrays_are_rejected_without_detaching():
    torch = pytest.importorskip('torch')
    mpo = qtn.MPO_identity(2)
    mpo.apply_to_arrays(lambda a: torch.tensor(a, requires_grad=True))
    with pytest.raises(TypeError, match='autodiff'):
        delinearize_mpo(mpo)
    assert all(a.requires_grad for a in mpo.arrays)


def test_native_arrays_are_rejected_without_flattening():
    pytest.importorskip('symmray')
    owner = MPOClusterProductExpansion(2, [[MPOProductTerm.from_pauli((0, 1), 'ZZ')]],
        cluster_size=2, factorization='fixed', cutoff=0., symmetry='U1', physical_charges=(0, 1))
    native = owner.exp(-.1j).to_mpo()
    with pytest.raises(TypeError, match='native symmetry'):
        delinearize_mpo(native)
