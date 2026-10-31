"""Prepared SU preserves ordered adaptive replay and its physical derivatives."""

import pytest


@pytest.mark.parametrize('cap,cutoff', [(2, 1e-12), (1, 1e-12), (2, .4)])
@pytest.mark.parametrize('smudge', [0., 1e-12])
@pytest.mark.parametrize('backend', [None, 'aot_eager'])
def test_prepared_matches_sequential_matrix_and_gradient(cap, cutoff, smudge, backend):
    _check_prepared(cap, cutoff, smudge, backend)


def _check_prepared(cap, cutoff, smudge, backend, *, renorm_bonds=True,
                    svd_driver=None, zero_external=False, shared=False):
    torch = pytest.importorskip('torch')
    import quimb.tensor as qtn
    from pepsy.backends import TorchLinalgConfig
    from pepsy.operators.simple_update import PreparedSimpleUpdate

    network = qtn.PEPO.rand(2, 2, bond_dim=2, dtype='complex128', seed=21)
    network.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    network.exponent = -.7
    gauges = {ix: torch.tensor([.7, 1.1], dtype=torch.float64) for ix in network.inner_inds()}
    if zero_external:
        shared = set(network[network.site_tag((0, 0))].inds) & set(network[network.site_tag((1, 0))].inds)
        gauges[next(iter(shared))][0] = 0.
    original_gauges = {k: v.clone() for k, v in gauges.items()}
    schedule = [((((0, 0), (0, 1))), 'upper'),
                ((((1, 0), (1, 1))), 'lower'),
                ((((0, 1),)), 'lower'),
                ((((0, 0), (1, 0))), 'lower'),
                ((((0, 1), (1, 1))), 'upper')]
    prepared = PreparedSimpleUpdate(network, gauges, schedule, max_bond=cap,
        cutoff=cutoff, smudge=smudge, torch_backend=backend, renorm_bonds=renorm_bonds)
    plans = [prepared]
    if shared:
        # Independent shorter schedules share compatible ready updates. The
        # final plan uses another rank cap and must retain its own fallback.
        for stop, rank in ((3, cap), (1, 1)):
            plans.append(PreparedSimpleUpdate(network, gauges, schedule[:stop],
                max_bond=rank, cutoff=cutoff, smudge=smudge,
                torch_backend=backend, renorm_bonds=renorm_bonds))
    angles = torch.tensor([[.17, -.12, .2, .08, -.11], [.22, .06, -.1, .12, .15]],
                          dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    gates = []
    for k, (sites, _) in enumerate(schedule):
        p = x if len(sites) == 1 else torch.kron(x, z)
        angle = angles[:, k, None, None]
        gates.append(angle.cos()*torch.eye(p.shape[0]) - 1j*angle.sin()*p)

    def dense(pair):
        tn, weights = pair
        tn = tn.copy()
        tn.gauge_simple_insert(weights)
        return tn.to_dense(optimize='greedy')

    with TorchLinalgConfig(mode='complex', stabilized=True, qr_rank_policy='adaptive',
                           svd_driver=svd_driver).activated():
        batches = [gates[:len(plan.schedule)] for plan in plans]
        results = (PreparedSimpleUpdate.apply_many(plans, batches) if shared else
                   [prepared.apply(gates)])
        actual = torch.stack([dense(p) for pairs in results for p in pairs])
        expected = torch.stack([dense(p) for plan, batch in zip(plans, batches)
                                for p in plan._adaptive(batch)])
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
        probe = torch.randn(actual.shape, dtype=actual.dtype, device='cpu',
                            generator=torch.Generator().manual_seed(29)).to(actual.device)
        ga = torch.autograd.grad((actual.conj()*probe).real.sum(), angles, retain_graph=True)[0]
        ge = torch.autograd.grad((expected.conj()*probe).real.sum(), angles)[0]
        torch.testing.assert_close(ga, ge, atol=2e-10, rtol=2e-9)
    if not zero_external:
        assert prepared.info['fallback'] == (cap == 1 or cutoff == .4)
    if shared:
        assert all(plan.info['shared_batches'] > 0 for plan in plans[:2])
        assert plans[-1].info['shared_batches'] == 0
        assert plans[-1].info['fallback']
    assert network.exponent == -.7
    assert all(torch.equal(w, original_gauges[k]) for k, w in gauges.items())


@pytest.mark.parametrize('backend', [None, 'aot_eager'])
def test_prepared_without_scale_extraction(backend):
    _check_prepared(2, 1e-12, 1e-12, backend, renorm_bonds=False)


def test_prepared_zero_external_support():
    _check_prepared(2, 1e-12, 0., None, zero_external=True)


@pytest.mark.parametrize('backend', [None, 'aot_eager'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('renorm_bonds', [False, True])
def test_shared_ready_updates_preserve_heterogeneous_schedules(backend, device, renorm_bonds):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with torch.device(device):
        _check_prepared(2, 1e-12, 1e-12, backend, shared=True,
            renorm_bonds=renorm_bonds, svd_driver='gesvdj' if device == 'cuda' else None)


def test_shared_updates_validate_batch_count():
    from pepsy.operators import PreparedSimpleUpdate
    assert PreparedSimpleUpdate.apply_many([], []) == []
    with pytest.raises(ValueError, match='one gate list'):
        PreparedSimpleUpdate.apply_many([], [[]])


@pytest.mark.parametrize('backend', [None, 'aot_eager'])
@pytest.mark.parametrize('driver', ['gesvd', 'gesvdj'])
def test_prepared_cuda_saved_tensors_and_derivatives(backend, driver):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with torch.device('cuda'):
        _check_prepared(2, 1e-12, 1e-12, backend, svd_driver=driver)


@pytest.mark.parametrize('mode', ['abs', 'rel', 'sum1', 'sum2', 'rsum1', 'rsum2'])
def test_member_ranks_match_individual_quimb_truncation(mode):
    torch = pytest.importorskip('torch')
    from quimb.tensor.decomp import svd_truncated
    from pepsy.operators._simple_update_torch import _rank
    spectrum = torch.tensor([[1., .002, .0001], [1., .9, .4], [0., 0., 0.]], dtype=torch.float64)
    actual = _rank(spectrum, .01, mode, 2)
    expected = [svd_truncated(torch.diag(s), cutoff=.01, cutoff_mode=mode,
                              max_bond=2, absorb=None)[1].shape[-1] for s in spectrum]
    assert actual.tolist() == expected


def test_overcomplete_bond_uses_adaptive_replay_and_auto_policy():
    torch = pytest.importorskip('torch')
    import quimb.tensor as qtn
    from pepsy.backends import TorchLinalgConfig
    from pepsy.operators import PreparedSimpleUpdate
    network = qtn.PEPO.rand(1, 2, bond_dim=5, dtype='complex128', seed=65)
    network.apply_to_arrays(lambda x: torch.tensor(x))
    schedule = [(((0, 0), (0, 1)), 'upper')]
    plan = PreparedSimpleUpdate(network, {}, schedule, max_bond=5)
    gate = torch.eye(4, dtype=torch.complex128).unsqueeze(0)
    with TorchLinalgConfig(mode='complex', stabilized=True, qr_rank_policy='adaptive').activated():
        actual, weights = plan.apply([gate])[0]
    actual.gauge_simple_insert(weights)
    torch.testing.assert_close(actual.to_dense(), network.to_dense(), atol=1e-12, rtol=1e-12)
    assert plan.options['cutoff'] == 1e-12
    assert plan.options['cutoff_mode'] == 'rsum2'
    assert plan.info['fallback']
