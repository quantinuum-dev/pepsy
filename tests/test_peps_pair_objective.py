"""Cached full-tensor objectives must never materialize the pair norm matrix."""

import numpy as np
import pytest

from pepsy.boundary._reuse import StripEnvironmentCache
from pepsy.optimizers.peps._full_update import FullPair
from test_peps_full_update import fixture


def prepare(state, gate, where=((1, 1), (1, 2)), *, boundary=None, cache=None, chi=32):
    pair = FullPair(state, gate, where, chi=state.max_bond())
    guess, target = pair.initial_states()
    objective, boundary = pair.environment(
        guess, boundary=boundary, chi=chi, contraction_opt='greedy',
        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.}, strip_cache=cache,
    )
    return pair, guess, target, objective, boundary


def test_cached_scalar_loss_and_complex_gradients_match_exact_state(monkeypatch):
    torch = pytest.importorskip('torch')
    state, gate, _ = fixture(shape=(3, 3))
    pair, guess, target, objective, _ = prepare(state, gate)
    assert objective.target_tensor_count == 3  # two original sites and a gate
    params = {k: v.clone().requires_grad_() for k, v in objective.params.items()}

    def no_rebuild(*args, **kwargs):
        pytest.fail('a local objective evaluation rebuilt its contraction expression')

    monkeypatch.setattr('cotengra.array_contract_expression', no_rebuild)
    monkeypatch.setattr('cotengra.array_contract_tree', no_rebuild)
    # Gradients must be fresh each time; constants cannot retain a spent graph.
    for scale in (1., .97):
        trial = {k: scale * v for k, v in params.items()}
        loss = objective.loss(trial)
        actual = torch.autograd.grad(loss, tuple(params.values()), retain_graph=True)
        dense = guess.copy()
        for site, array in zip(pair.where, trial.values()):
            dense[site].modify(data=array)
        reference = target.to_dense()
        error = dense.to_dense() - reference
        expected_loss = (error.abs()**2).sum() / (reference.abs()**2).sum()
        expected = torch.autograd.grad(expected_loss, tuple(params.values()))
        torch.testing.assert_close(loss, expected_loss, rtol=1e-9, atol=1e-12)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-11)
    assert objective.expression_builds == 2


def test_bulk_d4_full_update_never_uses_a_dense_norm_or_psd_projection(monkeypatch):
    state, gate, _ = fixture(shape=(3, 4), bond=4)

    def forbidden(*args, **kwargs):
        pytest.fail('full tensor update attempted a dense norm/PSD operation')

    monkeypatch.setattr('pepsy.optimizers.peps._full_update._psd_project_metric', forbidden)
    monkeypatch.setattr('pepsy.optimizers.peps._full_update.ReducedPair.environment', forbidden)
    pair, _, _, objective, _ = prepare(state, gate, chi=8)
    out, report = pair.optimize(objective, {'max_iterations': 2})
    assert report['norm_matrix_formed'] is False
    assert report['autodiff_backend'] == 'torch'
    assert report['contraction_cache']['expression_builds'] == 2
    assert max(objective.largest_intermediates) < 4**12
    assert out.max_bond() == 4
    assert all(np.prod(out[site].shape) == 2 * 4**4 for site in pair.where)


def test_cached_pair_constants_are_snapshots_and_new_gate_rebuilds_target():
    torch = pytest.importorskip('torch')
    state, gate, _ = fixture(shape=(3, 3))
    cache = StripEnvironmentCache()
    _, _, _, objective, boundary = prepare(state, gate, cache=cache)
    original = float(objective.loss(objective.params))
    with torch.no_grad():
        state[(0, 0)].data.reshape(-1)[0] *= 1.4
        gate[0, 0] *= .7 + .3j
    assert float(objective.loss(objective.params)) == original
    _, _, _, cached, _ = prepare(state, gate, cache=cache, boundary=boundary)
    _, _, _, fresh, _ = prepare(state, gate)
    assert float(cached.loss(cached.params)) == pytest.approx(float(fresh.loss(fresh.params)), abs=1e-10)
    assert abs(float(cached.loss(cached.params)) - original) > 1e-6


def test_pair_updates_with_cached_and_fresh_environments_match():
    state, gate, _ = fixture(shape=(3, 4))
    where = [((1, 1), (1, 2)), ((1, 1), (1, 2)),
             ((1, 2), (1, 3)), ((1, 2), (2, 2))]
    outputs = []
    cached_objectives = []
    for reuse in (False, True):
        candidate = state.copy()
        cache, boundary = StripEnvironmentCache(), None
        for index, sites in enumerate(where):
            pair, _, _, objective, handle = prepare(
                candidate, gate if index != 1 else gate.conj(), sites,
                boundary=boundary if reuse else None, cache=cache if reuse else None,
            )
            candidate, report = pair.optimize(objective, {'max_iterations': 8, 'rtol': 0.})
            boundary = handle
            if reuse:
                cached_objectives.append(objective)
            assert report['loss_history'][-1] <= report['loss_history'][0]
        outputs.append(candidate.to_dense().numpy())
        if reuse:
            assert cache.hits > 0
    assert len({id(obj.norm_expr) for obj in cached_objectives}) == 4
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-7, atol=1e-9)


def test_invalid_environment_cannot_be_optimized_as_a_good_negative_loss():
    state, gate, _ = fixture(shape=(3, 3))
    _, _, _, objective, _ = prepare(state, gate)
    objective.norm_expr = lambda *arrays: -objective.target_norm
    with pytest.raises(FloatingPointError, match='norm'):
        objective.loss(objective.params)
