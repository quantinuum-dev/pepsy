"""Importance support, branch-state dispatch and backend probability contracts."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers import noise


def _factory(bits="0", mode="direct"):
    return lambda: pepsy.MpsOptimizer(
        qtn.MPS_computational_state(bits, dtype="complex128"), chi=4, mode=mode
    )


def test_importance_support_preserves_arbitrarily_rare_positive_branches():
    target = np.array([1., 1e-40])
    policy = pepsy.ImportanceSamplingPolicy([1., 0.])
    with pytest.raises(ValueError, match="omits a target branch"):
        policy.probabilities(0, ("idle", "rare"), target)
    proposal = pepsy.ImportanceSamplingPolicy([1., 1e-40]).probabilities(
        0, ("idle", "rare"), target
    )
    assert proposal[1] == 1e-40


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_impossible_kraus_proposal_rejected_before_branch_application(monkeypatch, strategy):
    def fail_apply(*args, **kwargs):
        pytest.fail("Impossible branch reached replay instead of validation")

    monkeypatch.setattr(noise, "_run_trajectory_entries", fail_apply)
    with pytest.raises(ValueError, match="impossible target branch"):
        pepsy.run_trajectory_shots(
            _factory(),
            [pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(.2), 0)],
            shots=8, seed=6, strategy=strategy,
            importance_sampling={"jump": .5, "no_jump": .5},
            run_kwargs={"progbar": False},
        )


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_mixture_callback_observes_each_prebranch_parent(strategy):
    channel = pepsy.TrajectoryChannel.mixture(
        (("I", .5, np.eye(2)), ("X", .5, np.array([[0, 1], [1, 0]])))
    )
    seen = []

    def proposal(event_index, labels, target, optimizer):
        assert optimizer is not None
        if event_index == 0:
            return target
        bit = int(np.argmax(np.abs(optimizer.to_dense().reshape(-1))))
        seen.append(bit)
        return {"I": .8 if bit == 0 else .2, "X": .2 if bit == 0 else .8}

    result = pepsy.run_trajectory_shots(
        _factory(), [pepsy.TrajectoryEvent(channel, 0)] * 2,
        shots=64, seed=31, strategy=strategy, importance_sampling=proposal,
        parallel_workers=2 if strategy == "coalesced" else 1,
        run_kwargs={"progbar": False},
    )
    assert set(seen) == {0, 1}
    records = result.records if strategy == "independent" else [x.records for x in result.leaves]
    for weight, record in zip(result.weights, records):
        previous_bit = int(record[0].label == "X")
        idle_proposal = .8 if previous_bit == 0 else .2
        expected = idle_proposal if record[1].label == "I" else 1 - idle_proposal
        assert record[1].proposal_probability == pytest.approx(expected)
        assert weight == pytest.approx(.5 / expected)
    if strategy == "coalesced":
        assert len(seen) == 2
        assert sum(result.counts) == 64
        assert result.branches == 4
        with pytest.raises(RuntimeError, match="branch cap"):
            pepsy.run_trajectory_shots(
                _factory(), [pepsy.TrajectoryEvent(channel, 0)] * 2,
                shots=64, seed=31, strategy=strategy, importance_sampling=proposal,
                max_branches=2, run_kwargs={"progbar": False},
            )


def test_kraus_outcomes_share_one_base_norm_read(monkeypatch):
    original = noise._trajectory_norm_squared
    calls = []

    def counted(optimizer):
        calls.append(optimizer)
        return original(optimizer)

    monkeypatch.setattr(noise, "_trajectory_norm_squared", counted)
    probabilities = noise._kraus_probabilities(
        _factory("10")(), pepsy.TrajectoryChannel.amplitude_damping(.3), (0,)
    )
    np.testing.assert_allclose(probabilities, [.7, .3])
    assert len(calls) == 1


@pytest.mark.parametrize("retain", ["none", "all"])
@pytest.mark.parametrize("replay", [
    "direct",
    pytest.param("deferred", marks=[pytest.mark.optional, pytest.mark.integration]),
])
def test_diagnostics_aggregate_without_per_shot_retention(monkeypatch, retain, replay):
    original = noise._trajectory_diagnostics
    snapshots = []
    summaries = []

    def snapshot(optimizer):
        assert retain == "none"
        snapshots.append(None)
        return {
            "max_kraus_probability_residual": len(snapshots) / 100,
            "used_kraus_copy_fallback": len(snapshots) == 1,
        }

    def diagnostics(*args, **kwargs):
        summaries.extend(kwargs.get("diagnostic_infos", ()))
        return original(*args, **kwargs)

    monkeypatch.setattr(noise, "_trajectory_diagnostic_snapshot", snapshot)
    monkeypatch.setattr(noise, "_trajectory_diagnostics", diagnostics)
    if replay == "deferred":
        pytest.importorskip("stim")
        factory = lambda: pepsy.StabilizerMpsSimulator(2)
        stream = [("h", 0), ("t", 0)]
        kwargs = {"magic_strategy": "deferred", "magic_ancillas": (1,)}
    else:
        factory, stream, kwargs = _factory(), [], {}
    result = pepsy.run_trajectory_shots(
        factory, stream, shots=16, seed=6, retain=retain, **kwargs,
        run_kwargs={"progbar": False},
    )
    if retain == "none":
        assert len(snapshots) == 16
        assert len(summaries) == 1
        assert result.diagnostics.max_kraus_probability_residual == .16
        assert result.diagnostics.used_kraus_copy_fallback
        assert not result.optimizers
    else:
        assert not summaries and not snapshots
        assert len(result.optimizers) == 16


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.parametrize("backend", ["torch", "jax", "jax-nondefault", "cuda", "cupy"])
def test_one_site_exact_kraus_gram_stays_on_backend(monkeypatch, backend):
    if backend in {"torch", "cuda"}:
        torch = pytest.importorskip("torch")
        device = "cuda" if backend == "cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        convert = lambda x: torch.tensor(np.array(x), dtype=torch.complex64, device=device)
        expected_backend = "torch"
    elif backend == "cupy":
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CuPy GPU unavailable")
        convert = lambda x: cp.asarray(x, dtype=cp.complex64)
        expected_backend = backend
    else:
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        if backend == "jax-nondefault":
            if len(jax.devices()) < 2:
                pytest.skip("Requires two JAX devices")
            convert = lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.complex64), jax.devices()[1])
        else:
            convert = lambda x: jnp.asarray(x, dtype=jnp.complex64)
        expected_backend = "jax"
    state = qtn.MPS_computational_state("1", dtype="complex64")
    state.apply_to_arrays(convert)
    original_dtype = state[0].data.dtype
    original_device = state[0].data.device
    optimizer = pepsy.MpsOptimizer(state, chi=4, mode="exact")
    original = np.asarray

    def scalar_only(value, *args, **kwargs):
        if ar.infer_backend(value) == expected_backend and ar.size(value) > 1:
            pytest.fail("Kraus Gram evaluation downloaded an array")
        return original(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", scalar_only)
        probabilities = noise._kraus_probabilities(
            optimizer, pepsy.TrajectoryChannel.amplitude_damping(.3), (0,)
        )
    np.testing.assert_allclose(probabilities, [.7, .3], atol=2e-7)
    assert all(ar.infer_backend(t.data) == expected_backend for t in optimizer.p.tensors)
    assert all(t.data.dtype == original_dtype for t in optimizer.p.tensors)
    assert all(t.data.device == original_device for t in optimizer.p.tensors)
    assert not optimizer._trajectory_diagnostics.get("used_kraus_copy_fallback", False)
