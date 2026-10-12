"""Single-factor limits and independent ordered-product maturity checks."""

from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterPlan, MPOBasis, MPOClusterFactor, MPOClusterProductExpansion,
    MPOParameter, MPOProductTerm, PEPOClusterProductExpansion, PauliPEPOBasis,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
PAULI = dict(I=np.eye(2), X=np.array([[0., 1.], [1., 0.]]),
             Y=np.array([[0., -1j], [1j, 0.]]), Z=np.diag([1., -1.]))
SPECS = ((('XX', (0, 1), 'a'), ('Z', (0,), .1)),
         (('YZ', (1, 2), 'b'), ('X', (1,), .2)),
         (('XZ', (0, 2), 'c'), ('Y', (2,), .3)))


def embedded(word, sites, cluster):
    letters = dict(zip(sites, word))
    value = np.ones((1, 1), dtype=complex)
    for site in cluster:
        value = np.kron(value, PAULI[letters.get(site, 'I')])
    return value


def local_reference(cluster, params):
    result = np.eye(2**len(cluster), dtype=complex)
    for terms, scale in zip(SPECS, (.7, params['s'], -.4)):
        h = np.zeros_like(result)
        for word, sites, coefficient in terms:
            if set(sites) <= set(cluster):
                h += (params[coefficient] if isinstance(coefficient, str) else coefficient) * embedded(word, sites, cluster)
        result = result @ expm(-.11j*params['t']*scale*h)
    return result


def reference(params, cutoff):
    if cutoff == 3:
        return local_reference((0, 1, 2), params)
    singles = [local_reference((i,), params) for i in range(3)]
    result = np.kron(np.kron(singles[0], singles[1]), singles[2])
    # On three sites no two edge residuals are disjoint. Enumerate all three
    # pair corrections explicitly, independently of the package planner.
    for pair in combinations(range(3), 2):
        other = next(i for i in range(3) if i not in pair)
        residual = local_reference(pair, params) - np.kron(*(singles[i] for i in pair))
        order = (*pair, other)
        axes = tuple(order.index(i) for i in range(3))
        correction = np.kron(residual, singles[other]).reshape((2,)*6)
        result += correction.transpose(axes + tuple(i+3 for i in axes)).reshape(8, 8)
    return result


def make_product(kind, cutoff):
    factors = [MPOClusterFactor([
        MPOProductTerm.from_pauli(sites, word, coefficient=(MPOParameter(c) if isinstance(c, str) else c))
        for word, sites, c in terms], scale)
        for terms, scale in zip(SPECS, (.7, MPOParameter('s'), -.4))]
    plan = ClusterPlan.from_terms((term for f in factors for term in f.terms),
                                 sites=3, shape=(1, 3), cluster_size=cutoff)
    if kind.startswith('mpo'):
        opts = {'assembly': 'recursive'} if kind == 'mpo-recursive' else {}
        result = MPOClusterProductExpansion.from_plan(
            plan, factors, cutoff=0., factorization='fixed', graph_assembly='exact', **opts)
    else:
        result = PEPOClusterProductExpansion.from_plan(
            plan, factors, layout=kind, factorization='fixed')
    assert set(map(frozenset, plan.index_edges)) == {frozenset(p) for p in combinations(range(3), 2)}
    return result.compile_exp()


@pytest.mark.parametrize('kind', ['mpo', 'mpo-recursive', 'graph', 'square'])
@pytest.mark.parametrize('cutoff', [2, 3])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_joint_union_geometry_targets_and_all_live_gradients(kind, cutoff, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    product = make_product(kind, cutoff)
    weights = np.arange(64).reshape(8, 8)/64
    for values in ((.2, -.3, .4, -.5, .8), (0., 0., 0., 0., .8)):
        theta = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
        params = dict(zip('abcst', theta))
        result = product.exp(-.11j*params['t'], params, materialize=False)
        dense = result.to_mpo().to_dense() if kind.startswith('mpo') else result.to_dense()
        expected = reference(dict(zip('abcst', values)), cutoff)
        assert dense.device == theta.device
        np.testing.assert_allclose(dense.detach().cpu().numpy(), expected, atol=3e-12)
        trace = product.trace_exp(-.11j*params['t'], params)
        torch.testing.assert_close(trace, torch.trace(dense), atol=3e-12, rtol=3e-12)
        loss = ((dense.real+.37*dense.imag)*torch.tensor(weights, device=device)).sum()
        gradient, = torch.autograd.grad(loss, theta)
        def objective(v):
            matrix = reference(dict(zip('abcst', v)), cutoff)
            return ((matrix.real+.37*matrix.imag)*weights).sum()
        delta = 1e-5
        fd = [(objective(np.array(values)+delta*np.eye(5)[i])
               - objective(np.array(values)-delta*np.eye(5)[i]))/(2*delta) for i in range(5)]
        np.testing.assert_allclose(gradient.detach().cpu().numpy(), fd, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_uniform_single_and_identity_extended_joint_match(cyclic, order):
    basis = PauliPEPOBasis(2, 2, [('onsite', 'X', .2), ('edge', 'ZZ', .3)],
                          order=order, cyclic=cyclic, factorization='fixed')
    zero = PauliPEPOBasis(2, 2, [('onsite', 'Y', 0.)],
                         order=order, cyclic=cyclic, factorization='fixed')
    single = basis.compile_exp().exp(-.11j)
    product = PEPOClusterProductExpansion.from_bases([zero, basis, zero]).compile_exp()
    joint = product.exp(-.11j, materialize=False)
    np.testing.assert_allclose(joint.trace(), single.trace(), atol=2e-12)
    # Full matrices remain modest for these finite located channels and for
    # order one/two; higher open uniform orders can have huge dense sites.
    if cyclic or order <= 2:
        np.testing.assert_allclose(joint.to_dense(), single.to_dense(), atol=2e-12)


def test_single_mpo_interaction_factory_matches_one_factor_joint():
    basis = MPOBasis.from_local_terms(3, [((0, 2), (PAULI['X'], PAULI['Z']), .2)])
    one = MPOClusterProductExpansion.from_mpo_basis(basis, graph='interactions', cluster_size=2)
    joint = MPOClusterProductExpansion.from_bases([basis], graph='interactions', cluster_size=2)
    expected = expm(.03*.2*embedded('XZ', (0, 2), (0, 1, 2)))
    for source in (one, joint):
        np.testing.assert_allclose(source.exp(.03, materialize=True).to_dense(), expected, atol=2e-12)


def test_joint_native_u1_diagonal_mpo_keeps_metadata_and_report():
    pytest.importorskip('symmray')
    factors = [[((0,), (PAULI['Z'],), .2)],
               [((0, 1), (PAULI['Z'], PAULI['Z']), .3)],
               [((1,), (PAULI['Z'],), -.4)]]
    source = MPOClusterProductExpansion(2, factors, cluster_size=2,
                                       symmetry='U1', physical_charges=(0, 1)).compile_exp()
    actual, report = source.exp(-.13j, materialize=True, return_report=True)
    h = (.2*np.kron(PAULI['Z'], PAULI['I']),
         .3*np.kron(PAULI['Z'], PAULI['Z']), -.4*np.kron(PAULI['I'], PAULI['Z']))
    expected = expm(-.13j*h[0]) @ expm(-.13j*h[1]) @ expm(-.13j*h[2])
    np.testing.assert_allclose(actual.to_dense(), expected, atol=2e-12)
    assert report.native_block_sparse and report.symmetry == 'U1'


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('hopping', [False, True])
def test_native_u1_hopping_and_zero_cutoff_materialize(joint, hopping):
    """Charge-carrying and null SVD channels preserve the represented operator."""
    pytest.importorskip('symmray')
    up = np.array([[0., 1.], [0., 0.]])
    factors = ([[((0, 1), (up, up.T), .3), ((0, 1), (up.T, up), .3)]] if hopping
               else [[((0, 1), (PAULI['Z'], PAULI['Z']), .3)]])
    if joint:
        factors = [[((0,), (PAULI['Z'],), .2)], *factors, [((1,), (PAULI['Z'],), -.4)]]
    source = MPOClusterProductExpansion(2, factors, cluster_size=2, cutoff=1e-12 if hopping else 0.,
                                       symmetry='U1', physical_charges=(0, 1))
    dense_source = MPOClusterProductExpansion(2, factors, cluster_size=2, cutoff=0.)
    semantic = source.exp(-.13j)
    actual = semantic.to_mpo()
    np.testing.assert_allclose(actual.to_dense(), dense_source.exp(-.13j, materialize=True).to_dense(),
                               atol=2e-12)
    assert semantic.validate_charge_flow().native


@pytest.mark.parametrize('options', [{'max_bond': 1}, {'compress': True, 'materialize': False}])
def test_joint_pepo_rejects_unapplied_compression_before_evaluation(options):
    calls = []
    def coefficient(_):
        calls.append(True)
        return .2
    bases = [PauliPEPOBasis(1, 2, [('edge', word, coefficient)], cluster_size=2)
             for word in ('XX', 'YZ', 'ZZ')]
    product = PEPOClusterProductExpansion.from_bases(bases).compile_exp()
    with pytest.raises(ValueError, match='compress=True'):
        product.exp(-.13j, {}, **options)
    assert not calls
