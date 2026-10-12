"""Exact batched targets and differentiated layered boundary compression."""

import autoray as ar
import numpy as np
import pytest

from pepsy.optimizers import GlobalOptimizer, PepsOptimizer
from pepsy.optimizers.peps._strip_batch import layered_target
from test_peps_strip_batches import inputs
from test_peps_global_jax import jax_x64  # noqa: F401

pytestmark = [pytest.mark.optional, pytest.mark.integration]


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_default_global_mps_layered_target_value_and_gradient(backend, jax_x64):  # noqa: F811
    pytest.importorskip(backend)
    state, _ = inputs()
    rng = np.random.default_rng(819)
    gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    target = layered_target(state, [(gate, ((0, 1), (1, 1)), None),
                                    (gate, ((1, 1), (2, 1)), None)])
    original = tuple((t.data, tuple(t.tags), t.inds) for t in target)
    original_state = tuple((t.data, tuple(t.tags), t.inds) for t in state)
    x, y = (tn.to_dense().ravel() for tn in (state, target))
    expected = 1 - abs(np.vdot(x, y))**2 / (np.vdot(x, x).real * np.vdot(y, y).real)
    optimizer = GlobalOptimizer(
        state, target, chi=(32, 64),
        loss_kwargs={'contraction_opt': 'greedy'},
        norm_kwargs={'contraction_opt': 'greedy'},
    )
    assert optimizer.loss() == pytest.approx(expected, abs=2e-9)
    assert GlobalOptimizer._norm_state(target, chi=64, contraction_opt='greedy') == pytest.approx(1., abs=2e-9)
    tnopt = optimizer.make_tn_optimizer(autodiff_backend=backend, progbar=False)
    vector = tnopt.vectorizer.vector.copy()
    direction = rng.normal(size=vector.size)
    direction /= np.linalg.norm(direction)
    value, gradient = tnopt.vectorized_value_and_grad(vector)
    assert value == pytest.approx(expected, abs=2e-9)
    eps = 1e-5
    finite_difference = (tnopt.vectorized_value(vector + eps * direction)
                         - tnopt.vectorized_value(vector - eps * direction)) / (2 * eps)
    assert np.isfinite(gradient).all()
    assert gradient @ direction == pytest.approx(finite_difference, abs=2e-7, rel=1e-4)
    assert all(t.data is a and tuple(t.tags) == tags and t.inds == inds
               for t, (a, tags, inds) in zip(target, original))
    assert all(t.data is a and tuple(t.tags) == tags and t.inds == inds
               for t, (a, tags, inds) in zip(state, original_state))


def test_jax_diagonal_rank_bound_preserves_target_and_tracing(jax_x64):  # noqa: F811
    state, gate = inputs()
    reference = state.copy().gate_(gate, ((0, 0), (1, 0)), contract='split', cutoff=0.)
    state.apply_to_arrays(jax_x64.numpy.asarray)
    gate = jax_x64.numpy.asarray(gate)
    driver = PepsOptimizer(state, chi=2, contraction_opt='greedy')
    target = driver._build_target(
        state, gate, ((0, 0), (1, 0)), None,
        cutoff=0., cutoff_mode='rel', gate_kwargs=None,
    )
    assert target.max_bond() == 4
    np.testing.assert_allclose(ar.to_numpy(target.to_dense()), reference.to_dense(), atol=1e-11)
    # Traced zeros can have nonzero tangents; their full rank must be retained.
    assert int(jax_x64.jit(driver._gate_bond_factor)(gate)) == 4


@pytest.mark.optional
@pytest.mark.integration
@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('batch,width', [('strip', 1), ('strip', 2), ('all', 1)])
@pytest.mark.parametrize('representation,gate_kind', [
    ('peps', 'diagonal'), ('layered', 'diagonal'), ('layered', 'full-rank'),
])
def test_global_batch_exact_target_gradient_and_compression(backend, batch, width, representation, gate_kind, monkeypatch, jax_x64):  # noqa: F811
    pytest.importorskip(backend)
    state, gate = inputs()
    if gate_kind == 'full-rank':
        rng = np.random.default_rng(84)
        gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    entries = [(gate, ((0, 0), (1, 0))), (gate, ((1, 0), (2, 0)))]
    if width == 2 or batch == 'all':
        entries += [(gate, ((0, 1), (1, 1)))]
    if batch == 'all':
        entries += [(np.diag(np.exp(.1j * np.array([1., -1.]))), (2, 2)),
                    (gate, ((2, 1), (2, 2)))]
    exact = state.copy()
    for g, sites in entries:
        exact.gate_(g, sites, contract='split', cutoff=0.)
    reference = exact.to_dense().ravel()
    original = state.to_dense().copy()
    convert = (jax_x64.numpy.asarray if backend == 'jax'
               else pytest.importorskip('torch').as_tensor)
    state.apply_to_arrays(convert)
    entries = [(convert(g), sites) for g, sites in entries]
    make_optimizer = GlobalOptimizer.make_tn_optimizer
    checked = []

    def check(self, **kwargs):
        assert self.target_norm == (1., 0.) or self.target_norm == 1.
        np.testing.assert_allclose(ar.to_numpy(self.state_target.to_dense()).ravel(), reference, atol=1e-10)
        assert self.loss_opt['chi'] == (8, 12)
        if representation == 'layered':
            assert '__PEPS_GATE__' in self.state_target.tags
            assert max(t.size for t in self.state_target.select('__PEPS_BASE__')) <= max(t.size for t in state)
        tnopt = make_optimizer(self, **kwargs)
        x = tnopt.vectorizer.vector.copy()
        direction = np.random.default_rng(73).normal(size=x.size)
        direction /= np.linalg.norm(direction)
        _, gradient = tnopt.vectorized_value_and_grad(x)
        eps = 1e-5
        fd = (tnopt.vectorized_value(x + eps * direction)
              - tnopt.vectorized_value(x - eps * direction)) / (2 * eps)
        assert np.isfinite(gradient).all()
        assert gradient @ direction == pytest.approx(fd, abs=2e-7, rel=1e-4)
        tnopt.vectorizer.vector[:] = x
        tnopt.losses.clear()
        checked.append(True)
        return tnopt

    monkeypatch.setattr(GlobalOptimizer, 'make_tn_optimizer', check)
    opt = PepsOptimizer(
        state, entries, chi=2, mode='global', contraction_opt='greedy',
        boundary_chi=(8, 12), normalize_chi=32, evaluation_chi=(32, 64),
        boundary_convergence=False, fit_mode='direct',
        global_kwargs={'loss_kwargs': {'mode': 'mps', 'mode_': 'direct',
                                      'layer_tags': ('BRA', 'KET'), 'cutoff': 0.}},
        global_optimize_kwargs={'autodiff_backend': backend, 'optimizer': 'L-BFGS-B', 'n': 3},
    )
    def no_target_norm(*args, **kwargs):
        pytest.fail('The unitary target norm must not be recomputed')
    monkeypatch.setattr(opt, '_measure_target_norm', no_target_norm)
    out = opt.run(k_2q_batch=batch, strip_batch_size=width,
                  warmstart_method='simple-update', infidelity_tol=0.,
                  target_representation=representation)
    assert checked == [True]
    record, = opt.get_step_records()
    assert record['post_infidelity'] < record['pre_infidelity']
    assert out.max_bond() <= 2
    vector = ar.to_numpy(out.to_dense()).ravel()
    assert np.vdot(vector, vector).real == pytest.approx(1., abs=1e-9)
    np.testing.assert_allclose(ar.to_numpy(state.to_dense()), original, atol=1e-15)
    assert all(ar.infer_backend(t.data) == backend for t in out)


def test_global_known_target_norm_and_shared_caps(monkeypatch):
    state, _ = inputs()
    optimizer = GlobalOptimizer(state, state.copy(), chi=(8, 12), target_norm=1.,
                                loss_kwargs={'contraction_opt': 'greedy'})
    import quimb.tensor as qtn
    contract = qtn.TensorNetwork2D.contract_boundary
    calls = []
    def count(self, **kwargs):
        calls.append((kwargs['max_bond'], tuple(kwargs['layer_tags'])))
        return contract(self, **kwargs)
    monkeypatch.setattr(qtn.TensorNetwork2D, 'contract_boundary', count)
    assert abs(optimizer.loss(layer_tags=('BRA', 'KET'), mode_='direct')) < 1e-8
    assert calls == [(8, ('BRA', 'KET')), (12, ('BRA', 'KET'))]
    optimizer.set_target_norm(4.)
    optimizer.set_target(state.copy())
    assert optimizer.target_norm == 1.
    assert optimizer.normalize_kwargs['chi'] == (8, 12)


@pytest.mark.parametrize('backend,jit_fn', [('torch', None), ('torch', True), ('jax', None)])
def test_standalone_global_registers_before_differentiated_boundary(backend, jit_fn, jax_x64):  # noqa: F811
    state, gate = inputs()
    target = state.copy()
    target.gate_(gate, ((0, 0), (1, 0)), contract='split', cutoff=0.)
    optimizer = GlobalOptimizer(
        state, target, chi=(8, 12), target_norm=1.,
        loss_kwargs={'contraction_opt': 'greedy', 'mode_': 'direct',
                     'layer_tags': ('BRA', 'KET')},
    )
    tnopt = optimizer.make_tn_optimizer(autodiff_backend=backend, jit_fn=jit_fn, progbar=False)
    x = tnopt.vectorizer.vector.copy()
    direction = np.random.default_rng(71).normal(size=x.size)
    direction /= np.linalg.norm(direction)
    value, grad = tnopt.vectorized_value_and_grad(x)
    assert value > 0.
    eps = 1e-5
    fd = (tnopt.vectorized_value(x + eps * direction)
          - tnopt.vectorized_value(x - eps * direction)) / (2 * eps)
    assert np.isfinite(grad).all()
    assert grad @ direction == pytest.approx(fd, abs=2e-7, rel=1e-4)
