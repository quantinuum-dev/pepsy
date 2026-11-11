"""Explicit and adjacent joint cluster updates retain chronology, Cotengra and clipped patches."""

import numpy as np
import pytest

from pepsy.bp import PreparedLoopClusterUpdate
from pepsy.operators import gate_loop_cluster
from pepsy.tensors import id_to_pepo
from test_pepo_reduced_legs import apply_dense, dense

torch = pytest.importorskip('torch')


def state(device='cpu'):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    value = id_to_pepo(2, 3)
    value.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128, device=device))
    return value


def gates(device='cpu'):
    rng = np.random.default_rng(19)
    return [torch.tensor(np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))[0],
                         device=device) for _ in range(2)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('first', ['upper', 'lower'])
def test_explicit_and_automatic_joint_updates_match_dense_and_each_other(device, reverse, first):
    core, payloads = state(device), gates(device)
    where = ((0, 1), (1, 1))
    second_where = tuple(reversed(where)) if reverse else where
    other_side = 'lower' if first == 'upper' else 'upper'
    schedule = [(where, first), (second_where, other_side)]
    expected = dense(core.to_dense())
    for payload, (sites, side) in zip(payloads, schedule):
        expected = apply_dense(core, expected, dense(payload), sites, side)
    aligned_other = payloads[1].reshape(2, 2, 2, 2).permute(1, 0, 3, 2) if reverse else payloads[1]
    joint = tuple((payloads[0], aligned_other) if first == 'upper' else (aligned_other, payloads[0]))
    settings = dict(cluster_mode='graphdistance', base_radius=1, fillin=True)
    outputs = []
    for execution in ('prepared', 'reference', 'explicit-prepared', 'explicit-reference'):
        gauges, seen = {}, []
        if execution in {'prepared', 'explicit-prepared'}:
            actual_schedule = [(where, 'both')] if execution == 'explicit-prepared' else schedule
            bank = [joint] if execution == 'explicit-prepared' else payloads
            plan = PreparedLoopClusterUpdate(core, actual_schedule, max_bond=16, max_iterations=2,
                tol=1e-12, fuse_pepo_legs=True, optimize='greedy', **settings)
            out, rows = plan.apply(core.copy(), bank, gauges=gauges, return_results=True,
                materialize_physical=False, step_callback=lambda i, r: seen.append(i))
            assert plan.info['joint_pair_updates'] == 1 and plan.info['physical_gate_actions'] == 2
        else:
            common = dict(gauges=gauges, max_bond=16, max_loop_size=0, psd_project=False,
                optimize='greedy', return_results=True, strip_exponent=True,
                materialize_physical=False, step_callback=lambda i, r: seen.append(i),
                als_opts=dict(solver='torch', max_iterations=2, tol=1e-12), **settings)
            if execution == 'explicit-reference':
                out, rows = gate_loop_cluster(core.copy(), joint, where=where, which='both', **common)
            else:
                entries = [(payload, sites, side) for payload, (sites, side) in zip(payloads, schedule)]
                out, rows = gate_loop_cluster(core.copy(), entries, fuse_pepo_legs=True, **common)
        assert len(rows) == 1 and rows[0].pair.which == 'both'
        assert rows[0].pair.theta_shape[1:3] == (4, 4)
        assert seen == ([0] if execution.startswith('explicit') else [1])
        physical = out.copy()
        physical.gauge_simple_insert(gauges)
        np.testing.assert_allclose(dense(physical.to_dense()), expected, atol=3e-10, rtol=3e-10)
        outputs.append(dense(rows[0].problem.environment.data))
    for environment in outputs[1:]:
        np.testing.assert_allclose(environment, outputs[0], atol=2e-12, rtol=2e-12)


def test_apply_boundaries_and_intervening_actions_prevent_joint_fits():
    core, payloads = state(), gates()
    where = ((0, 1), (1, 1))
    schedule = [(where, 'upper'), (where, 'lower')]
    plan = PreparedLoopClusterUpdate(core, schedule, max_bond=4, fuse_pepo_legs=True, optimize='greedy')
    work, weights = core.copy(), {}
    for i in range(2):
        plan.apply(work, payloads, gauges=weights, start=i, stop=i+1)
    assert plan.info['joint_pair_updates'] == 0 and plan.info['pair_updates'] == 2
    schedule = [(where, 'upper'), (((0, 0),), 'upper'), (where, 'lower'), (where, 'lower')]
    plan = PreparedLoopClusterUpdate(core, schedule, max_bond=4, fuse_pepo_legs=True, optimize='greedy')
    plan.apply(core.copy(), [payloads[0], torch.eye(2, dtype=torch.complex128), *payloads], gauges={})
    assert plan.info['joint_pair_updates'] == 0 and plan.info['pair_updates'] == 3


@pytest.mark.parametrize('joint', [False, True])
def test_environment_uses_one_cotengra_contraction_per_fit(monkeypatch, joint):
    import cotengra as ct

    build = ct.array_contract_expression
    programs, calls = [], []
    def observed(inputs, output, **kwargs):
        function = build(inputs, output, **kwargs)
        if len(output) == 4 and all('__pepsy_reduced' in ix for ix in output):
            assert isinstance(kwargs['optimize'], ct.ContractionTree)
            programs.append(output)
            def run(*args, **kw):
                calls.append(output)
                return function(*args, **kw)
            return run
        return function
    monkeypatch.setattr(ct, 'array_contract_expression', observed)
    core, payloads = state(), gates()
    where = ((0, 1), (1, 1))
    plan = PreparedLoopClusterUpdate(core, [(where, 'upper'), (where, 'lower')],
        max_bond=4, max_iterations=3, fuse_pepo_legs=joint, optimize='greedy')
    plan.apply(core, payloads, gauges={}, materialize_physical=False)
    assert programs and len(calls) == (1 if joint else 2)


@pytest.mark.parametrize('shape', [(2, 2), (2, 5), (5, 2), (4, 5), (5, 4)])
def test_every_bond_patch_matches_independent_clipped_rectangle(shape):
    lx, ly = shape
    core = id_to_pepo(lx, ly)
    core.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    bonds = [((i, j), (i+di, j+dj)) for i in range(lx) for j in range(ly)
             for di, dj in ((1, 0), (0, 1)) if i+di < lx and j+dj < ly]
    plan = PreparedLoopClusterUpdate(core, [(sites, 'upper') for sites in bonds],
        cluster_mode='graphdistance', base_radius=1, fillin=True, optimize='greedy')
    for slot in plan.slots:
        (a, b), (c, d) = slot.sites
        expected = {core.site_tag(i, j) for i in range(max(0, min(a, c)-1), min(lx, max(a, c)+2))
                    for j in range(max(0, min(b, d)-1), min(ly, max(b, d)+2))}
        region = plan._region_for(core, slot)
        actual = {core.site_tag(i, j) for i, j in core.sites
                  if core.site_tag(i, j) in {tag for tid in region for tag in core.tensor_map[tid].tags}}
        assert actual == expected


def test_inactive_corner_storage_stays_decoupled_in_both_orientations():
    core = id_to_pepo(4, 4)
    core.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    corners = ((0, 0), (3, 3))
    corner_arrays = [core[site].data.clone() for site in corners]
    a, b = ((0, 1), (0, 2)), ((1, 0), (2, 0))
    payloads = gates()*2
    schedule = [(a, 'upper'), (a, 'lower'), (b, 'lower'), (b, 'upper')]
    exact = core.copy()
    for payload, (sites, side) in zip(payloads, schedule):
        exact.gate_inds_(payload, tuple(getattr(exact, side+'_ind')(*site) for site in sites),
                         contract='split', cutoff=0.)
    plan = PreparedLoopClusterUpdate(core, schedule, max_bond=16, max_iterations=2,
        tol=1e-12, fuse_pepo_legs=True, cluster_mode='graphdistance', base_radius=1,
        fillin=True, optimize='greedy')
    weights = {}
    out = plan.apply(core.copy(), payloads, gauges=weights)
    for site, before in zip(corners, corner_arrays):
        torch.testing.assert_close(out[site].data, before)
        assert all(out.ind_size(ix) == 1 for ix in out[site].inds if ix in set(out.inner_inds()))
    physical = out.copy()
    physical.gauge_simple_insert(weights)
    active = set(a+b)
    for probe in range(3):
        selected = {fn(*site): ((sum(site)+k+probe) % 2 if site in active else 0)
                    for site in core.sites for k, fn in enumerate((core.upper_ind, core.lower_ind))}
        actual = physical.isel(selected).contract(all, optimize='greedy')
        expected = exact.isel(selected).contract(all, optimize='greedy')
        torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_joint_update_preserves_gate_derivatives(device):
    import quimb.tensor as qtn
    from pepsy import TorchLinalgConfig

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    core = qtn.PEPO.rand(2, 2, bond_dim=2, dtype='complex128', seed=227)
    core.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128, device=device))
    sites = ((0, 0), (0, 1))
    plan = PreparedLoopClusterUpdate(core, [(sites, 'upper'), (sites, 'lower')],
        max_bond=2, max_iterations=3, tol=0., fuse_pepo_legs=True, optimize='greedy')
    def loss(value):
        z = torch.tensor([1., -1., -1., 1.], dtype=torch.float64, device=device)
        payloads = [torch.diag(torch.exp(-1j*value*z)), gates(device)[1]]
        weights = {}
        out = plan.apply(core.copy(), payloads, gauges=weights, materialize_physical=False)
        physical = out.copy()
        physical.gauge_simple_insert(weights)
        return physical.to_dense().real.sum()
    value = torch.tensor(.17, dtype=torch.float64, device=device, requires_grad=True)
    with TorchLinalgConfig(stabilized=True).activated():
        derivative = torch.autograd.grad(loss(value), value)[0]
        delta = 1e-5
        finite = (loss(value.detach()+delta)-loss(value.detach()-delta))/(2*delta)
    assert abs(float(derivative)) > 1e-5
    torch.testing.assert_close(derivative, finite, rtol=5e-4, atol=5e-5)


def test_joint_payload_validation_precedes_mutation():
    core = state()
    sites = ((0, 1), (1, 1))
    plan = PreparedLoopClusterUpdate(core, [(sites, 'both')], optimize='greedy')
    before, weights = [tensor.data for tensor in core], {}
    with pytest.raises(ValueError, match='gate arrays'):
        plan.apply(core, [(gates()[0], torch.eye(2, dtype=torch.complex128))], gauges=weights)
    assert not weights
    assert all(tensor.data is old for tensor, old in zip(core, before))
    with pytest.raises(ValueError, match='batch_size=1'):
        PreparedLoopClusterUpdate(core, [(sites, 'upper')], batch_size='auto', fuse_pepo_legs=True)
