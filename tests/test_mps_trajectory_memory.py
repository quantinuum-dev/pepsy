"""Memory planning preserves retention, shot counts and numerical settings."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn
from types import SimpleNamespace

from pepsy.optimizers import MpsOptimizer, noise
from pepsy.optimizers.mps import _trajectory_execution as execution
from pepsy.backends._memory import available_device_memory


def _sim(stream=None):
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    return MpsOptimizer(state, stream or [("x_error", .01, 0)] * 4, chi=2)


# Four qubits, chi=2: 24 complex128 entries at maximum bond growth = 384B.
# After 8 state workspaces and 32MiB, this leaves four parent/child slots.
_FOUR_STATES = (32 << 20) + 6144


def test_state_estimate_covers_growth_exact_storage_and_initial_large_bonds():
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    assert execution._estimated_state_bytes(state, 2, "quimb-direct") == 384
    large = qtn.MPS_rand_state(4, 4, dtype="complex128", seed=9)
    estimate = execution._estimated_state_bytes(large, 2, "quimb-direct")
    assert estimate >= sum(t.data.nbytes for t in large.tensors)
    assert execution._estimated_state_bytes(state, 2, "exact") >= 16 * 16


def test_budget_caps_branches_without_changing_seeded_results_or_state():
    sim = _sim([("x_error", .5, 0)])
    kwargs = dict(shots=32, seed=7, strategy="coalesced", progress=False)
    reference = sim.run(memory_budget=None, **kwargs)
    result = sim.run(memory_budget=_FOUR_STATES, **kwargs)
    assert result.counts == reference.counts
    assert result.records == reference.records
    assert result.diagnostics.memory_max_branches == 4
    assert result.diagnostics.estimated_state_bytes == 384
    assert result.diagnostics.memory_budget_bytes == _FOUR_STATES
    for actual, expected in zip(result.optimizers, reference.optimizers):
        np.testing.assert_array_equal(actual.to_dense(), expected.to_dense())
        assert actual.chi == expected.chi == 2
        assert actual.backend_info() == expected.backend_info()


@pytest.mark.parametrize("retain", ["all", "final"])
def test_insufficient_retained_independent_budget_fails_before_factory(monkeypatch, retain):
    sim = _sim()
    monkeypatch.setattr(sim, "_shot_factory", lambda: pytest.fail("allocated a shot"))
    with pytest.raises(MemoryError, match="requires 16"):
        sim.run(shots=16, strategy="independent", retain=retain,
                memory_budget=_FOUR_STATES, progress=False)


@pytest.mark.parametrize("workers", [1, 2])
def test_runtime_cap_fallback_respects_retention_budget(workers):
    sim = _sim()
    kwargs = dict(shots=16, seed=1134, workers=workers, progress=False,
                  memory_budget=_FOUR_STATES)
    # Occupancy estimate fits four slots, but this seed realizes five leaves.
    with pytest.raises(MemoryError, match="requires 16"):
        sim.run(**kwargs)
    streamed = sim.run(retain="none", **kwargs)
    reference = sim.run(retain="none", strategy="independent", **kwargs)
    assert streamed.coalesced
    assert streamed.shots == reference.shots == 16
    assert streamed.optimizers == ()
    assert streamed.diagnostics.fallback_reason == "continued from shared prefixes at branch cap"
    assert streamed.diagnostics.max_kraus_probability_residual == reference.diagnostics.max_kraus_probability_residual


def test_explicit_coalesced_cap_still_raises_without_pruning():
    with pytest.raises(noise._CoalescedBranchCapExceeded):
        _sim().run(shots=16, seed=1134, strategy="coalesced",
                   memory_budget=_FOUR_STATES, progress=False)


@pytest.mark.parametrize("workers", [1, 2])
def test_zero_probability_outcomes_do_not_force_retained_expansion(workers):
    result = _sim([("x_error", 0., 0)] * 4).run(
        shots=16, workers=workers, max_branches=1,
        memory_budget=_FOUR_STATES, progress=False,
    )
    assert result.counts == (16,)
    assert not result.diagnostics.continued_from_cap


def test_auto_queries_once_and_uses_half_allocator_allowance(monkeypatch):
    calls = []

    def available(array):
        calls.append(array)
        return 2 * _FOUR_STATES

    monkeypatch.setattr(execution, "available_device_memory", available)
    result = _sim().run(shots=16, seed=1, memory_budget="auto", progress=False)
    assert len(calls) == 1
    assert result.diagnostics.memory_budget_bytes == _FOUR_STATES
    assert result.diagnostics.memory_max_branches == 4


def test_unavailable_memory_and_disabled_policy_preserve_existing_replay(monkeypatch):
    monkeypatch.setattr(execution, "available_device_memory", lambda _: None)
    result = _sim().run(shots=8, memory_budget="auto", progress=False)
    assert result.diagnostics.memory_budget_bytes is None
    assert "unavailable" in result.diagnostics.memory_reason
    monkeypatch.setattr(execution, "available_device_memory", lambda _: pytest.fail("query"))
    assert _sim().run(shots=8, memory_budget=None, progress=False).shots == 8


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "bad"])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="memory_budget"):
        _sim().run(shots=4, memory_budget=budget, progress=False)


def test_empty_ensemble_needs_no_state_capacity():
    result = _sim().run(shots=0, memory_budget=1, progress=False)
    assert result.shots == 0
    assert result.optimizers == ()


def test_torch_memory_query_respects_allocator_quota_and_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    # Substitute query values without device allocation or global allocator
    # reconfiguration; quota and reusable cache must both enter the estimate.
    from pepsy.backends import _memory

    monkeypatch.setattr(_memory.ar, "infer_backend", lambda _: "torch")
    array = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (8000, 16000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 6000)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 2000)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda _: .75)
    assert available_device_memory(array) == 10000

    def unavailable(_):
        raise RuntimeError("allocator statistics unavailable")

    monkeypatch.setattr(torch.cuda, "mem_get_info", unavailable)
    assert available_device_memory(array) is None


def test_child_exact_mode_override_uses_dense_estimate():
    initial = qtn.MPS_computational_state("0" * 12, dtype="complex128")
    result = MpsOptimizer(initial, [("x_error", .1, 0)], chi=2).run(
        shots=1, run_kwargs={"mode": "exact"}, progress=False,
        memory_budget=64 << 20,
    )
    assert result.diagnostics.estimated_state_bytes == (2**12) * 16


def test_native_fermionic_memory_estimate_preserves_charge_blocks():
    pytest.importorskip("symmray")
    from pepsy.tensors import Fermion, ps_to_mps

    fermion = Fermion(spinful=False, symmetry="U1", dtype="complex128")
    state = ps_to_mps(4, fermion=fermion, occupations=(1, 0, 1, 0), dtype="complex128")
    before = state.to_dense().to_dense()
    result = MpsOptimizer(state, [], chi=4).run(
        shots=2, strategy="independent", memory_budget=64 << 20, progress=False,
    )
    assert result.diagnostics.estimated_state_bytes > 0
    for opt in result.optimizers:
        assert all(type(t.data).__name__ == "U1FermionicArray" for t in opt.p.tensors)
        np.testing.assert_allclose(opt.to_dense().to_dense(), before, atol=1e-12)


def test_memory_fallback_releases_failed_frontier_before_new_factory(monkeypatch):
    import weakref

    sim = _sim()
    factory = sim._shot_factory()
    failed = []

    def capped(factory, *args, **kwargs):
        state = factory()
        failed.append(weakref.ref(state))
        raise noise._CoalescedBranchCapExceeded("test cap")

    def checked():
        assert all(ref() is None for ref in failed)
        return factory()

    monkeypatch.setattr(noise, "run_coalesced_trajectory_shots", capped)
    monkeypatch.setattr(sim, "_shot_factory", lambda: checked)
    result = sim.run(shots=16, retain="none", memory_budget=_FOUR_STATES, progress=False)
    assert not result.coalesced
    assert result.shots == 16


@pytest.mark.parametrize("backend", ["torch", "cupy", "jax"])
def test_real_gpu_memory_query_and_replay(backend):
    state = qtn.MPS_computational_state("0000", dtype="complex64")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        convert = lambda x: torch.tensor(x, dtype=torch.complex64, device="cuda")
    elif backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.Device(0).use()
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CuPy GPU unavailable")
        convert = lambda x: cp.asarray(x, dtype=cp.complex64)
    else:
        jax = pytest.importorskip("jax")
        if jax.devices()[0].platform != "gpu":
            pytest.skip("JAX GPU unavailable")
        convert = lambda x: jax.numpy.asarray(x, dtype=jax.numpy.complex64)
    state.apply_to_arrays(convert)
    before = available_device_memory(state[0].data)
    assert before is not None and before > 0
    sim = MpsOptimizer(state, [("x_error", .1, 0)], chi=2)
    result = sim.run(shots=8, seed=32, progress=False)
    assert result.diagnostics.memory_budget_bytes > 0
    assert result.shots == 8
    assert all(ar.get_dtype_name(t.data) == "complex64"
               for opt in result.optimizers for t in opt.p.tensors)
