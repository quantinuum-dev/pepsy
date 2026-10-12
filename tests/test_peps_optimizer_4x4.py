"""Sixteen-qubit circuit references for strip and global PEPS fitting."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _dense(state):
    return ar.to_numpy(state.to_dense(optimize="greedy")).reshape(-1)


def _fidelity(x, y):
    return float(abs(np.vdot(x, y)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real))


def _apply(vector, gate, sites):
    sites = (sites,) if isinstance(sites[0], int) else sites
    axes = [4 * x + y for x, y in sites]
    order = axes + [i for i in range(16) if i not in axes]
    data = vector.reshape((2,) * 16).transpose(order).reshape(2 ** len(axes), -1)
    data = gate @ data
    return data.reshape((2,) * 16).transpose(np.argsort(order)).reshape(-1)


@pytest.mark.parametrize("mode", ["sweep", pytest.param("global", marks=pytest.mark.optional)])
def test_4x4_circuit_fitting_improves_exact_fidelity(mode, monkeypatch):
    state = qtn.PEPS.rand(4, 4, bond_dim=2, seed=20261012, dtype="complex128")
    state.multiply_(1 / np.linalg.norm(_dense(state)))
    original = _dense(state).copy()
    zz = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    rx = np.cos(.13) * np.eye(2) - 1j * np.sin(.13) * np.array([[0., 1.], [1., 0.]])
    gates = [(rx, (x, y)) for x in range(4) for y in range(4)]
    gates += [(zz, ((x, y), (x + 1, y))) for y in range(4) for x in range(3)]
    gates += [(zz, ((x, y), (x, y + 1))) for x in range(4) for y in range(3)]
    reference = original.copy()
    for gate, sites in gates:
        reference = _apply(reference, gate, sites)

    # These caps avoid boundary truncation on this small D=2 reference.
    # Default finite-cap accuracy is a separate approximation, not an exact norm.
    options = dict(chi=2, mode=mode, contraction_opt="greedy",
                   boundary_chi=(16, 64), normalize_chi=16, evaluation_chi=(16, 64))
    if mode == "global":
        pytest.importorskip("torch")
        options["global_optimize_kwargs"] = dict(
            autodiff_backend="torch", optimizer="L-BFGS-B", n=5,
        )
    warm = PepsOptimizer(state, gates, **options).run(optimize=False, progress=False)
    driver = PepsOptimizer(state, gates, **options)
    optimize = driver._optimize_state
    visits = []

    def checked(warmstart, target, **kwargs):
        exact_target = _dense(target)
        before = 1 - _fidelity(_dense(warmstart), exact_target)
        snapshot = [(t.data, tuple(t.tags), t.inds) for t in target]
        output, reported, summary = optimize(warmstart, target, **kwargs)
        after = 1 - _fidelity(_dense(output), exact_target)
        assert after < before
        assert reported == pytest.approx(after, abs=2e-9)
        assert all(t.data is data and tuple(t.tags) == tags and t.inds == inds
                   for t, (data, tags, inds) in zip(target, snapshot))
        visits.append(after)
        return output, reported, summary

    monkeypatch.setattr(driver, "_optimize_state", checked)
    result = driver.run(progress=False, infidelity_tol=0.)
    vector = _dense(result)
    assert len(visits) == (8 if mode == "sweep" else 1)
    assert all(r["reason"] == "optimized" for r in driver.get_step_records())
    assert _fidelity(vector, reference) > _fidelity(_dense(warm), reference)
    assert np.vdot(vector, vector).real == pytest.approx(1., abs=2e-10)
    assert result.max_bond() <= 2
    assert np.isfinite(vector).all()
    np.testing.assert_array_equal(_dense(state), original)
