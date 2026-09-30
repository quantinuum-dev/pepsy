"""Native cluster channels retain zero/nonzero coefficient derivatives."""

from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import MPOClusterFactor, MPOClusterProductExpansion, MPOParameter

pytestmark = [pytest.mark.integration, pytest.mark.operators]
pytest.importorskip('symmray')
UP = np.array([[0., 1.], [0., 0.]])
Z = np.diag([1., -1.])


def source(mode='interval', symmetry='U1', assembly='direct', **options):
    edges = ((0, 2), (1, 3)) if mode == 'graph' else ((0, 1), (1, 2))
    length = 4 if mode == 'graph' else 3
    factors = [MPOClusterFactor([((*edge,), ops, MPOParameter(name))
               for ops in ((UP, UP.T), (UP.T, UP))], scale)
               for edge, name, scale in zip(edges, ('a', 'b'), (1., MPOParameter('s')))]
    factors.append(MPOClusterFactor([((i,), ((i+1)*Z,), MPOParameter('c')) for i in range(length)]))
    opts = dict(cluster_size=2 if mode == 'graph' else 3, cutoff=0., factorization='fixed',
                symmetry=symmetry, physical_charges=(0, 1), assembly=assembly)
    if mode != 'interval':
        opts.update(graph=(range(length), edges), graph_assembly='exact')
    opts.update(options)
    return MPOClusterProductExpansion(length, factors, **opts).compile_exp()


def reference(values, mode):
    a, b, c, scale, step = values
    length = 4 if mode == 'graph' else 3
    edges = ((0, 2), (1, 3)) if mode == 'graph' else ((0, 1), (1, 2))
    def embed(operators):
        return reduce(np.kron, (operators.get(i, np.eye(2)) for i in range(length)))
    h = [value*(embed({edge[0]: UP, edge[1]: UP.T})+embed({edge[0]: UP.T, edge[1]: UP}))
         for edge, value in zip(edges, (a, b*scale))]
    h.append(c*sum(embed({i: (i+1)*Z}) for i in range(length)))
    return reduce(np.matmul, (expm(-.2j*step*matrix) for matrix in h))


def finite_difference(values, mode):
    weights = np.arange(4**(3 if mode == 'interval' else 4)).reshape(reference(values, mode).shape)/23
    def objective(v):
        matrix = reference(v, mode)
        return ((matrix.real+.37*matrix.imag)*weights).sum()
    delta = 1e-5
    return [(objective(values+delta*np.eye(5)[i])-objective(values-delta*np.eye(5)[i]))/(2*delta)
            for i in range(5)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode,assembly', [('interval', 'direct'), ('graph', 'direct'), ('graph', 'recursive')])
@pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
def test_torch_native_joint_value_and_all_gradients(device, mode, assembly, symmetry):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    compiled = source(mode, symmetry, assembly)
    for values in (np.array([.3, -.4, .1, .7, .8]), np.array([0., 0., 0., .7, .8])):
        theta = torch.tensor(values, device=device, requires_grad=True)
        semantic, report = compiled.exp(-.2j*theta[-1], dict(zip('abcst', theta)), return_report=True)
        native = semantic.to_mpo()
        dense = native.to_dense()
        assert report.native_block_sparse and all(hasattr(t.data, 'blocks') for t in native)
        assert dense.device == theta.device
        np.testing.assert_allclose(dense.detach().cpu().numpy(), reference(values, mode), atol=3e-12)
        weights = torch.arange(dense.numel(), device=device, dtype=theta.dtype).reshape(dense.shape)/23
        gradient, = torch.autograd.grad(((dense.real+.37*dense.imag)*weights).sum(), theta)
        np.testing.assert_allclose(gradient.detach().cpu().numpy(), finite_difference(values, mode),
                                   atol=4e-8, rtol=2e-6)


@pytest.mark.parametrize('mode,assembly', [('interval', 'direct'), ('graph', 'direct'), ('graph', 'recursive')])
def test_jax_native_joint_jit_and_gradients(mode, assembly):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    compiled = source(mode, assembly=assembly)
    def objective(theta):
        dense = compiled.exp(-.2j*theta[-1], dict(zip('abcst', theta)), materialize=True).to_dense()
        weights = jnp.arange(dense.size).reshape(dense.shape)/23
        return ((dense.real+.37*dense.imag)*weights).sum(), dense
    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    for values in (np.array([.3, -.4, .1, .7, .8]), np.array([0., 0., 0., .7, .8])):
        (_, dense), gradient = evaluate(jnp.asarray(values))
        np.testing.assert_allclose(dense, reference(values, mode), atol=3e-12)
        np.testing.assert_allclose(gradient, finite_difference(values, mode), atol=4e-8, rtol=2e-6)


def test_native_static_proof_rejects_independent_nonconserving_zero_bindings():
    torch = pytest.importorskip('torch')
    x = UP+UP.T
    y = -1j*(UP-UP.T)
    factors = [[((0, 1), (x, x), MPOParameter('a')), ((0, 1), (y, y), MPOParameter('b'))]]
    compiled = MPOClusterProductExpansion(2, factors, symmetry='U1', physical_charges=(0, 1),
                                         factorization='fixed', cutoff=0.)
    with pytest.raises(ValueError, match='every independent coefficient'):
        compiled.exp(.1, {'a': torch.tensor(0., requires_grad=True), 'b': torch.tensor(0., requires_grad=True)})


@pytest.mark.parametrize('assembly', ['recursive', 'streaming'])
@pytest.mark.parametrize('form', ['left', 'right'])
@pytest.mark.parametrize('cutoff', [None, 1e-8])
def test_native_bounded_assembly_preserves_charge_and_total_cap(assembly, form, cutoff):
    values = np.array([.3, -.4, .1, .7, .8])
    for chi in (1, 16):
        compiled = source('graph', assembly=assembly, factorization='auto', assembly_chi=chi,
                          assembly_cutoff=cutoff, assembly_form=form)
        semantic, report = compiled.exp(-.2j*values[-1], dict(zip('abcst', values)), return_report=True)
        assert max(semantic.bond_dimensions) <= chi
        assert semantic.validate_charge_flow().native
        assert report.assembly_compression_count > 0 and report.native_block_sparse
        dense = semantic.to_mpo().to_dense()
        if chi == 16 and cutoff is None:
            np.testing.assert_allclose(dense, reference(values, 'graph'), atol=3e-12)
        elif chi == 16:
            # The requested relative discarded-weight cutoff is approximate
            # even when the bond cap could hold the exact operator.
            assert np.linalg.norm(dense-reference(values, 'graph')) < 1e-3
        else:
            assert np.linalg.norm(dense-reference(values, 'graph')) > 1e-3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('assembly', ['recursive', 'streaming'])
def test_native_bounded_torch_gradients_against_finite_differences(device, assembly):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from pepsy import TorchLinalgConfig
    compiled = source('graph', assembly=assembly, factorization='auto', assembly_chi=3)
    values = np.array([.3, -.4, .1, .7, .8])
    theta = torch.tensor(values, device=device, requires_grad=True)
    def objective(v):
        dense = compiled.exp(-.2j*v[-1], dict(zip('abcst', v)), materialize=True).to_dense()
        return (dense.real+.37*dense.imag).sum()
    # Paired-factor VJPs must work without the regularized QR/SVD registry.
    with TorchLinalgConfig(stabilized=False, quimb_split_drivers=False).activated():
        actual = objective(theta)
        gradient, = torch.autograd.grad(actual, theta)
        delta = 1e-5
        fd = [(objective(values+delta*np.eye(5)[i])-objective(values-delta*np.eye(5)[i]))/(2*delta)
              for i in range(5)]
    np.testing.assert_allclose(actual.detach().cpu().numpy(), objective(values), atol=3e-12)
    np.testing.assert_allclose(gradient.detach().cpu().numpy(), fd, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize('assembly', ['streaming', 'recursive'])
def test_native_connected_graph_full_cluster_with_intermediate_compression(assembly):
    values = np.array([.3, -.4, .1, .7, .8])
    compiled = source('chain', assembly=assembly, factorization='auto', assembly_chi=16)
    native = compiled.exp(-.2j*values[-1], dict(zip('abcst', values)), materialize=True)
    np.testing.assert_allclose(native.to_dense(), reference(values, 'chain'), atol=3e-12)


def test_native_compressed_jax_trace_has_explicit_rank_selection_error():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    compiled = source('graph', assembly='recursive', factorization='auto', assembly_chi=3)
    def objective(theta):
        return compiled.exp(-.2j*theta[-1], dict(zip('abcst', theta)), materialize=True).to_dense().real.sum()
    with pytest.raises(NotImplementedError, match='assembly_chi=None'):
        jax.jit(jax.grad(objective))(jnp.array([.3, -.4, .1, .7, .8]))


def test_native_public_semantic_compression_keeps_sectors_and_rejects_dense_fallback():
    from pepsy.operators import FirstDegreeMPO
    values = np.array([.3, -.4, .1, .7, .8])
    semantic = source('graph').exp(-.2j*values[-1], dict(zip('abcst', values)))
    semantic = FirstDegreeMPO(semantic.arrays, levels=semantic.levels,
                              physical_space=semantic.physical_space,
                              upper_ind_id='out{}', lower_ind_id='in{}', site_tag_id='S{}')
    compressed, report = semantic.compress_adaptive(3, cutoff=0., return_report=True)
    assert compressed.symmetry == 'U1' and compressed.validate_charge_flow().native
    assert (compressed.upper_ind_id, compressed.lower_ind_id, compressed.site_tag_id) == ('out{}', 'in{}', 'S{}')
    assert max(compressed.bond_dimensions) <= 3
    assert report.method == 'native-sector-projector'
    assert report.differentiable is False  # Discrete sector/rank selection is not differentiated.
    with pytest.raises(ValueError, match='fixed native sector allocation'):
        semantic.compress_fixed_rank(3)


def test_native_compression_rejects_charge_violations_before_projection():
    from pepsy.operators import FirstDegreeMPO
    invalid = FirstDegreeMPO.from_local_terms(2, [((0,), (UP+UP.T,))],
                                              symmetry='U1', physical_charges=(0, 1))
    with pytest.raises(ValueError, match='charge flow'):
        invalid.compress_adaptive(1, cutoff=0.)
