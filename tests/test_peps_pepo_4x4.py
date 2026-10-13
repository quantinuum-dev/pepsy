"""Full bond-2 PEPO applied to a D=3 PEPS against a 16-qubit oracle."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_pepo_4x4_d3_sufficient_caps_match_dense_fidelity_and_preserve_inputs():
    probe, gates = load_example("tfim_pepo_chi_probe"), load_example("tfim_pepo")
    peps = qtn.PEPS.rand(4, 4, bond_dim=3, seed=20261013, dtype="complex128")
    before = np.asarray(peps.to_dense(optimize="greedy")).reshape(-1)
    peps.multiply_(1 / np.linalg.norm(before))
    before /= np.linalg.norm(before)
    tensors = [(t.data.copy(), tuple(t.tags), t.inds) for t in peps]
    operator = gates.trotter_pepo(4, 4, .1)
    factors = [(t.data.copy(), tuple(t.tags), t.inds) for t in operator]
    target = probe.dense_step(before, 4, 4, .1)
    warm = operator.apply(peps, contract=True, compress=True, max_bond=3, cutoff=0.)
    warm_vector = np.asarray(warm.to_dense(optimize="greedy")).reshape(-1)
    fit = PepsOptimizer(peps, [], chi=3, mode="sweep", optimizer="als",
        boundary_chi=(81, 324), normalize_chi=81, evaluation_chi=(81, 324),
        contraction_opt="greedy", normalize_initial=False, boundary_convergence=False,
        sweep_kwargs={"local_contraction_opt": "greedy", "cutoff": 0.},
        sweep_optimize_kwargs={"axes": ("y",), "n_round_trips": 1},
        normalize_kwargs={"cutoff": 0.}, infidelity_kwargs={"cutoff": 0.})
    fit.run_pepo(operator, unitary=True, infidelity_tol=0., cutoff=0.)
    final = np.asarray(fit.state.to_dense(optimize="greedy")).reshape(-1)
    record = fit.get_step_records()[0]
    exact_loss = probe.dense_loss(final, target)
    assert exact_loss < probe.dense_loss(warm_vector, target)
    assert record["pre_infidelity"] == pytest.approx(probe.dense_loss(warm_vector, target), abs=2e-10)
    assert record["final_infidelity"] == pytest.approx(exact_loss, abs=2e-10)
    assert np.vdot(final, final).real == pytest.approx(1., abs=2e-10)
    assert np.isfinite(final).all()
    assert fit.state.max_bond() == 3
    assert operator.max_bond() == 2
    for network, saved in ((peps, tensors), (operator, factors)):
        for tensor, (array, tags, inds) in zip(network, saved, strict=True):
            np.testing.assert_array_equal(tensor.data, array)
            assert tuple(tensor.tags) == tags and tensor.inds == inds


def test_dense_strang_oracle_preserves_norm_and_matches_small_full_operator():
    probe, gates = load_example("tfim_pepo_chi_probe"), load_example("tfim_pepo")
    rng = np.random.default_rng(17)
    vector = rng.normal(size=64) + 1j * rng.normal(size=64)
    vector /= np.linalg.norm(vector)
    operator = gates.trotter_pepo(2, 3, .17, j=.7, h=1.3)
    expected = np.asarray(operator.to_dense()) @ vector
    actual = probe.dense_step(vector, 2, 3, .17, j=.7, h=1.3)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.linalg.norm(actual) == pytest.approx(1., abs=1e-14)
