"""Native Abelian cluster MPOs agree with independent dense targets."""

from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import MPOClusterProductExpansion

pytestmark = [pytest.mark.integration, pytest.mark.operators]
pytest.importorskip('symmray')


def embedded(sites, operators, length):
    local = dict(zip(sites, operators))
    identity = np.eye(operators[0].shape[0])
    return reduce(np.kron, (local.get(i, identity) for i in range(length)))


def reference(factors, length, step):
    d = factors[0][0][1][0].shape[0]
    result = np.eye(d**length, dtype=complex)
    for terms in factors:
        h = sum(c * embedded(sites, ops, length) for sites, ops, c in terms)
        result = result @ expm(step*h)
    return result


def hopping(a, b, up, strength):
    return [((a, b), (up, up.T), strength), ((a, b), (up.T, up), strength)]


@pytest.mark.parametrize('symmetry,charges', [
    ('U1', (0, 0, 1)), ('Z2', (0, 0, 1)),
    ('U1U1', ((0, 0), (0, 1), (1, 0), (1, 1))),
    ('Z2Z2', ((0, 0), (0, 1), (1, 0), (1, 1))),
])
@pytest.mark.parametrize('cutoff,factorization', [(0., 'auto'), (1e-12, 'auto'), (0., 'fixed')])
def test_three_site_joint_native_sectors_and_degenerate_physical_charges(symmetry, charges, cutoff, factorization):
    d = len(charges)
    diagonal = np.diag(np.linspace(-.4, .7, d))
    up = np.zeros((d, d))
    up[0, -1] = 1.
    other = np.zeros((d, d))
    other[0, 1] = 1.
    factors = [hopping(0, 1, up, .3) + [((0,), (diagonal,), .2)],
               hopping(1, 2, other, -.27) + [((1,), (diagonal,), -.4)],
               [((0, 1, 2), (up, diagonal, up.T), .13),
                ((0, 1, 2), (up.T, diagonal, up), .13)]]
    source = MPOClusterProductExpansion(3, factors, cluster_size=3, cutoff=cutoff,
                                       symmetry=symmetry, physical_charges=charges, factorization=factorization)
    semantic, report = source.compile_exp().exp(-.17j, return_report=True)
    native = semantic.to_mpo()
    np.testing.assert_allclose(native.to_dense(), reference(factors, 3, -.17j), atol=3e-12)
    if cutoff == 0.:
        expected = reference(factors, 3, -.17j).reshape((d,)*6)
        groups = ([native.upper_ind(i) for i in (2, 0, 1)],
                  [native.lower_ind(i) for i in (1, 2, 0)])
        reordered = expected.transpose(2, 0, 1, 4, 5, 3).reshape(d**3, d**3)
        np.testing.assert_allclose(native.to_dense(*groups), reordered, atol=3e-12)
    assert semantic.validate_charge_flow().native
    assert report.native_block_sparse
    assert all(hasattr(tensor.data, 'blocks') for tensor in native)


@pytest.mark.parametrize('edges', [((0, 2), (1, 3)), ((0, 3), (1, 2))])
@pytest.mark.parametrize('cutoff', [0., 1e-12])
def test_native_crossing_and_nested_paths_with_onsite_background(edges, cutoff):
    up = np.array([[0., 1.], [0., 0.]])
    z = np.diag([1., -1.])
    factors = [[((i,), (z,), .1*(i+1)) for i in range(4)],
               sum((hopping(*edge, up, .3) for edge in edges), []),
               [((i,), (z,), -.07*(4-i)) for i in range(4)]]
    # Disjoint graph components make p=2 exact even with overlapping spans.
    source = MPOClusterProductExpansion(4, factors, graph=(range(4), edges), graph_assembly='exact',
                                       cluster_size=2, cutoff=cutoff,
                                       symmetry='U1', physical_charges=(0, 1))
    semantic = source.exp(-.17j)
    np.testing.assert_allclose(semantic.to_mpo().to_dense(), reference(factors, 4, -.17j), atol=3e-12)
    assert semantic.validate_charge_flow().native


@pytest.mark.parametrize('step', [0., -.13j])
@pytest.mark.parametrize('zero_generator', [False, True])
def test_native_zero_channels_keep_identity(step, zero_generator):
    up = np.array([[0., 1.], [0., 0.]])
    factors = [hopping(0, 1, up, 0. if zero_generator else .3)]
    source = MPOClusterProductExpansion(3, factors, cluster_size=3, cutoff=0.,
                                       symmetry='U1', physical_charges=(0, 1))
    semantic = source.exp(step)
    np.testing.assert_allclose(semantic.to_mpo().to_dense(), reference(factors, 3, step), atol=2e-12)
    assert semantic.validate_charge_flow().native


def test_native_rank_cap_selects_global_sector_spectrum():
    up = np.array([[0., 1.], [0., 0.]])
    z = np.diag([1., -1.])
    factors = [hopping(0, 1, up, .3) + [((0, 1), (z, z), .47)]]
    source = MPOClusterProductExpansion(2, factors, cluster_size=2, cutoff=0., max_bond=1,
                                       symmetry='U1', physical_charges=(0, 1))
    semantic, report = source.compile_exp().exp(.23, return_report=True)
    actual = semantic.to_mpo().to_dense()
    target = reference(factors, 2, .23)
    residual = target - np.eye(4)
    matrix = residual.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = np.linalg.svd(matrix)
    best = (u[:, :1] * s[:1]) @ vh[:1]
    expected = np.eye(4) + best.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    assert report.local_svd_truncated
    assert all(max(ranks, default=0) <= 1 for _, ranks in report.residual_ranks)
    assert semantic.validate_charge_flow().native


@pytest.mark.parametrize('cluster_size', [1, 2])
def test_native_rejects_nonconserving_target(cluster_size):
    x = np.array([[0., 1.], [1., 0.]])
    source = MPOClusterProductExpansion(2, [[((0,), (x,), .2)]], cluster_size=cluster_size,
                                       symmetry='U1', physical_charges=(0, 1), cutoff=0.)
    with pytest.raises(ValueError, match='charge'):
        source.exp(.1).to_mpo()


@pytest.mark.parametrize('cluster_size', [1, 2])
def test_native_tensor_backend_preserves_gradient(cluster_size):
    torch = pytest.importorskip('torch')
    z = np.diag([1., -1.])
    source = MPOClusterProductExpansion(2, [[((0,), (z,), .2)]], cluster_size=cluster_size,
                                       symmetry='U1', physical_charges=(0, 1), cutoff=0.)
    step = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    dense = source.exp(step, materialize=True).to_dense()
    gradient, = torch.autograd.grad(dense.real.trace(), step)
    torch.testing.assert_close(gradient, .8*torch.sinh(.2*step))


@pytest.mark.parametrize('dtype', [np.float32, np.complex64, np.float64, np.complex128])
def test_native_four_site_local_factorization_preserves_dtype_and_reconstruction(dtype):
    from pepsy.operators import MPOPhysicalSpace
    from pepsy.operators._cluster_native import sector_operator_schmidt

    rng = np.random.default_rng(307)
    charges = np.array([i.bit_count() for i in range(16)])
    matrix = rng.normal(size=(16, 16))
    if np.issubdtype(dtype, np.complexfloating):
        matrix = matrix + 1j*rng.normal(size=(16, 16))
    matrix = np.where(charges[:, None] == charges[None, :], matrix, 0.).astype(dtype)
    cores = sector_operator_schmidt(matrix, 4, MPOPhysicalSpace(2, symmetry='U1',
                                   physical_charges=(0, 1)), 0., None)
    assert all(core.dtype == dtype for core in cores)
    result = cores[0][0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=(0, 0))
        result = np.moveaxis(result, -3, 0)
    result = result[0].transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(16, 16)
    np.testing.assert_allclose(result, matrix, atol=2e-6 if matrix.real.dtype == np.float32 else 3e-14)
