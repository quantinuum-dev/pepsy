"""Observable regressions for the opt-in exact factor tree sampler."""

import gc
from itertools import product
import weakref

import numpy as np
import pytest

from pepsy.operators import cnot, h
from pepsy.optimizers import TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.sampling import TreeSampler
from pepsy.sampling._tree_factor import _FactorSamplingContext, _FactorWorkspaceExceeded

pytestmark = [pytest.mark.tree, pytest.mark.integration, pytest.mark.optional]


def _backend(state, backend):
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        if backend == "torch_cuda" and not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        device = "cuda:0" if backend == "torch_cuda" else "cpu"
        state.apply_to_arrays(lambda a: torch.as_tensor(np.asfortranarray(a), device=device))
    elif backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        state.apply_to_arrays(lambda a: cp.asfortranarray(cp.asarray(a)))


@pytest.fixture
def force_grouping(monkeypatch):
    monkeypatch.setattr(_FactorSamplingContext, "min_parent", 1)
    monkeypatch.setattr(_FactorSamplingContext, "min_collapse", 0)


@pytest.mark.parametrize("backend", ["numpy", "torch", "torch_cuda", "cupy"])
@pytest.mark.parametrize("dtype", ["float64", "complex64", "complex128"])
@pytest.mark.parametrize("arity,root", [(2, None), (3, None), (4, None), (3, 2)])
@pytest.mark.parametrize("chunk", [None, 13])
def test_factor_samples_match_dense_and_standard(force_grouping, backend, dtype, arity, root, chunk):
    dims = [2, 3, 2, 3, 2, 3, 2]
    plan = TreePlan.from_order(
        [q for q in range(7) if q != root], structure="balanced", max_arity=arity,
        top_arity=arity if root is None else 2, root_qubit=root,
    )
    state = TreeTensorNetwork.rand(plan, D=5, seed=19, dtype=dtype, phys_dim=3)
    for q, dim in enumerate(dims):
        tensor = state[state.site_tag(q)]
        selection = [slice(None)] * tensor.ndim
        selection[tensor.inds.index(state.site_ind(q))] = slice(0, dim)
        tensor.modify(data=tensor.data[tuple(selection)])
    state.invalidate_canonical_form()
    vector = np.asarray(state.to_dense()).reshape(-1)
    exact = abs(vector) ** 2
    exact /= exact.sum()
    weights = np.asarray([np.prod(dims[q+1:], dtype=np.int64) for q in range(7)])
    _backend(state, backend)
    center = state.orthogonality_center
    originals = [(t.data, t.left_inds) for t in state.tensors]
    standard = TreeSampler(state, seed=11, chunk_size=chunk)
    factor = TreeSampler(
        state, seed=11, strategy="factor", chunk_size=chunk,
        cache_bytes=1024, workspace_bytes=4096,
    )
    for seed in (17, None, None):
        expected = standard.sample_batch(41, seed=seed).to_numpy()
        result = factor.sample_batch(41, seed=seed)
        assert result.backend == ("torch" if backend.startswith("torch") else backend)
        if backend != "numpy":
            assert result.configs.device == result.probs.device == originals[0][0].device
        actual = result.to_numpy()
        np.testing.assert_array_equal(actual.configs, expected.configs)
        rtol = 3e-5 if dtype == "complex64" else 2e-11
        atol = 1e-10 if dtype == "complex64" else 1e-14
        np.testing.assert_allclose(actual.probs, exact[actual.configs @ weights], rtol=rtol, atol=atol)
        assert state.orthogonality_center == center
        assert all(t.data is data and t.left_inds == left for t, (data, left) in zip(state.tensors, originals))


@pytest.mark.parametrize("backend", ["numpy", "torch", "torch_cuda", "cupy"])
@pytest.mark.parametrize("n,root", [(1, None), (1, 0), (7, None), (7, 2)])
def test_factor_default_settings_match_dense(backend, n, root):
    # The larger matrix above forces grouping with small budgets. Exercise
    # the actual defaults separately, including the single-site boundary.
    plan = TreePlan.from_order(
        [q for q in range(n) if q != root], structure="balanced",
        max_arity=3, top_arity=None if n == 1 else (3 if root is None else 2),
        root_qubit=root,
    )
    state = TreeTensorNetwork.rand(plan, D=5, seed=19, dtype="complex128")
    vector = np.asarray(state.to_dense()).reshape(-1)
    exact = abs(vector) ** 2
    exact /= exact.sum()
    weights = 2 ** np.arange(n - 1, -1, -1)
    _backend(state, backend)
    for chunk in (None, 7):
        standard = TreeSampler(state, seed=11, chunk_size=chunk, threads=1)
        factor = TreeSampler(
            state, seed=11, strategy="factor", chunk_size=chunk, threads=1,
        )
        for seed in (17, None, None):
            expected = standard.sample_batch(41, seed=seed).to_numpy()
            actual = factor.sample_batch(41, seed=seed).to_numpy()
            np.testing.assert_array_equal(actual.configs, expected.configs)
            np.testing.assert_allclose(
                actual.probs, exact[actual.configs @ weights], rtol=2e-11, atol=1e-14,
            )


@pytest.mark.parametrize("root", [None, 2])
@pytest.mark.parametrize("chunk,cache", [(None, 0), (3, 1024)])
@pytest.mark.parametrize("backend", ["torch", "torch_cuda"])
def test_factor_probabilities_preserve_gradients(force_grouping, root, chunk, cache, backend):
    torch = pytest.importorskip("torch")
    plan = TreePlan.from_order([q for q in range(5) if q != root], root_qubit=root)
    state = TreeTensorNetwork.rand(plan, D=4, seed=19, dtype="complex128")
    _backend(state, backend)
    state.apply_to_arrays(lambda a: a.requires_grad_())
    state.invalidate_canonical_form()
    inputs = tuple(t.data for t in state.tensors)
    sampler = TreeSampler(state, strategy="factor", chunk_size=chunk, cache_bytes=cache, workspace_bytes=512)
    configs, actual = sampler.sample_arrays(7, seed=17)
    expected = sampler.probabilities(configs, to_numpy=False)
    basis = np.asarray(list(product(range(2), repeat=5)))
    expected = expected / sampler.probabilities(basis, to_numpy=False).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    ag = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
    eg = torch.autograd.grad(expected.sum(), inputs)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-10)


def test_cache_exhaustion_computes_only_misses(force_grouping, monkeypatch):
    sampler = TreeSampler(TreeTensorNetwork.from_order(range(3), dtype="float64"))
    context = _FactorSamplingContext(sampler, 80, 1024)
    work = []

    def transfer(rho, ur, **kwargs):
        work.append(len(rho))
        return rho.copy()

    monkeypatch.setattr(sampler, "_sample_first_child_density", transfer)
    ur = np.ones((2, 2, 1))
    configs = np.array([[0, 0, 0], [0, 0, 1]])
    rho = np.stack([np.eye(2), np.eye(2) * 2])
    context.transfer(0, rho, ur, configs)
    actual = context.transfer(0, rho * 2, ur, np.array([[0, 0, 1], [0, 1, 0]]))
    np.testing.assert_array_equal(actual, rho * 2)
    assert work == [2, 1]
    assert context.cache_used == 80


def test_grouped_remainder_stays_compact(force_grouping, monkeypatch):
    state = TreeTensorNetwork.rand(
        TreePlan.from_order(range(7), max_arity=3, top_arity=3), D=5, seed=19,
    )
    sampler = TreeSampler(state, strategy="factor")
    batches = []
    original = sampler._tensordot

    def record(left, right, axes):
        batches.append(len(left))
        return original(left, right, axes)

    monkeypatch.setattr(sampler, "_tensordot", record)
    actual = sampler.sample_batch(128, seed=17)
    expected = TreeSampler(state).sample_batch(128, seed=17)
    np.testing.assert_array_equal(actual.configs, expected.configs)
    np.testing.assert_allclose(actual.probs, expected.probs, rtol=1e-11, atol=1e-13)
    assert max(batches) < 128


@pytest.mark.parametrize("backend", ["numpy", "torch", "torch_cuda", "cupy"])
def test_overflow_uses_exact_row_keys(force_grouping, backend):
    # Reverse the traversal relative to the configuration column order.
    # Overflow row keys then sort subtree and full-prefix rows differently.
    state = TreeTensorNetwork.rand(
        TreePlan.from_order(list(reversed(range(70))), structure="balanced"), D=2, seed=19,
    )
    _backend(state, backend)
    sampler = TreeSampler(state, strategy="factor", chunk_size=7)
    with sampler._array_device_context():
        context = _FactorSamplingContext(sampler, 1024, 4096)
        assert context.radix is None
        configs = np.zeros((4, 70), dtype=np.int64)
        configs[1:3, 0] = 1
        configs[3, 69] = 1
        _, indices, inverse = context.groups(sampler._as_backend(configs))
        indices = np.asarray(indices.get() if backend == "cupy" else indices.cpu() if backend.startswith("torch") else indices)
        inverse = np.asarray(inverse.get() if backend == "cupy" else inverse.cpu() if backend.startswith("torch") else inverse)
        assert len(indices) == 3
        np.testing.assert_array_equal(configs[indices][inverse], configs)
    actual = sampler.sample_batch(17, seed=17).to_numpy()
    expected = TreeSampler(state).sample_batch(17, seed=17).to_numpy()
    np.testing.assert_array_equal(actual.configs, expected.configs)
    np.testing.assert_allclose(actual.probs, expected.probs, rtol=1e-11, atol=1e-30)


@pytest.mark.parametrize("fail", [False, True])
def test_factor_call_releases_scratch_without_gc(force_grouping, monkeypatch, fail):
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(7)), D=5, seed=19)
    sampler = TreeSampler(state, strategy="factor", chunk_size=7)
    refs = []
    values = []
    original = _FactorSamplingContext.transfer

    def transfer(context, *args, **kwargs):
        result = original(context, *args, **kwargs)
        refs.append(weakref.ref(context))
        values.extend(weakref.ref(value) for _, value in context.cache.values())
        if fail and context.cache:
            raise RuntimeError("injected cached transfer failure")
        return result

    monkeypatch.setattr(_FactorSamplingContext, "transfer", transfer)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        if fail:
            try:
                sampler.sample_arrays(21, seed=17)
            except RuntimeError:
                pass
            else:
                pytest.fail("failure hook was not reached")
        else:
            sampler.sample_arrays(21, seed=17)
        assert refs and values
        assert all(ref() is None for ref in refs + values)
    finally:
        if was_enabled:
            gc.enable()


def test_factor_sampling_is_reentrant(force_grouping, monkeypatch):
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(7)), D=5, seed=19)
    sampler = TreeSampler(state, strategy="factor", chunk_size=7)
    original = sampler._sample_arrays
    nested = []

    def kernel(*args, **kwargs):
        if not nested:
            nested.append(None)
            nested[0] = sampler.sample_batch(11, seed=21).to_numpy()
        return original(*args, **kwargs)

    monkeypatch.setattr(sampler, "_sample_arrays", kernel)
    actual = sampler.sample_batch(17, seed=17).to_numpy()
    for result, count, seed in [(actual, 17, 17), (nested[0], 11, 21)]:
        expected = TreeSampler(state).sample_batch(count, seed=seed).to_numpy()
        np.testing.assert_array_equal(result.configs, expected.configs)
        np.testing.assert_allclose(result.probs, expected.probs, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("root", [None, 2])
@pytest.mark.parametrize("backend", ["numpy", "torch", "torch_cuda", "cupy"])
def test_factor_zero_branches_remain_finite(force_grouping, root, backend):
    plan = TreePlan.from_order(
        [q for q in range(7) if q != root], root_qubit=root, max_arity=3,
        top_arity=3 if root is None else 2,
    )
    state = TreeTensorNetwork.rand(plan, D=2, dtype="complex64", phys_dim=3, seed=19)
    # Copy tensors make a GHZ state with an unused third physical code and
    # exact zeros in the branch tensors, rather than a generic full-rank state.
    for tensor in state.tensors:
        data = np.zeros(tensor.shape, dtype="complex64")
        data[(0,) * tensor.ndim] = 1
        data[(1,) * tensor.ndim] = 1
        tensor.modify(data=data)
    state.invalidate_canonical_form()
    _backend(state, backend)
    sampler = TreeSampler(state, strategy="factor", chunk_size=7, cache_bytes=128, workspace_bytes=128)
    actual = sampler.sample_batch(31, seed=17).to_numpy()
    assert np.all(actual.configs == actual.configs[:, :1])
    assert np.all(actual.configs < 2)
    np.testing.assert_allclose(actual.probs, 0.5, rtol=2e-5, atol=1e-7)


def test_factor_workspace_reduces_chunks(force_grouping, monkeypatch):
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(7)), D=5, seed=19)
    sampler = TreeSampler(state, strategy="factor", chunk_size=31, workspace_bytes=1)
    work = []
    original = sampler._sample_arrays

    def record(count, *args, **kwargs):
        work.append(count)
        return original(count, *args, **kwargs)

    monkeypatch.setattr(sampler, "_sample_arrays", record)
    actual = sampler.sample_batch(11, seed=17)
    expected = TreeSampler(state).sample_batch(11, seed=17)
    assert work == [1] * 11
    np.testing.assert_array_equal(actual.configs, expected.configs)
    np.testing.assert_allclose(actual.probs, expected.probs, rtol=1e-11, atol=1e-13)


def test_remainder_retry_preserves_draws_and_cache(force_grouping, monkeypatch):
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(7)), D=5, seed=19)
    sampler = TreeSampler(state, strategy="factor", chunk_size=31)
    work = []
    original = sampler._sample_arrays
    checks = []
    original_check = _FactorSamplingContext.check_remainder

    def check(context, rows, elements):
        if not checks:
            checks.append(True)
            raise _FactorWorkspaceExceeded
        return original_check(context, rows, elements)

    def record(count, *args, **kwargs):
        work.append(count)
        return original(count, *args, **kwargs)

    monkeypatch.setattr(_FactorSamplingContext, "check_remainder", check)
    monkeypatch.setattr(sampler, "_sample_arrays", record)
    actual = sampler.sample_batch(31, seed=17)
    expected = TreeSampler(state).sample_batch(31, seed=17)
    assert work == [31, 15, 15, 1]
    np.testing.assert_array_equal(actual.configs, expected.configs)
    np.testing.assert_allclose(actual.probs, expected.probs, rtol=1e-11, atol=1e-13)


def test_factor_cache_does_not_survive_refresh(force_grouping):
    plan = TreePlan.from_order(range(7))
    state = TreeTensorNetwork.rand(plan, D=5, seed=19)
    sampler = TreeSampler(state, strategy="factor", chunk_size=7)
    sampler.sample_batch(31, seed=17)
    replacement = TreeTensorNetwork.rand(plan, D=5, seed=21)
    sampler.refresh(replacement)
    actual = sampler.sample_batch(31, seed=17)
    expected = TreeSampler(replacement).sample_batch(31, seed=17)
    np.testing.assert_array_equal(actual.configs, expected.configs)
    np.testing.assert_allclose(actual.probs, expected.probs, rtol=1e-11, atol=1e-13)


def test_signed_key_boundary_with_unit_physical_dimension():
    state = TreeTensorNetwork.from_order(range(64))
    tensor = state.node_tensor(state.plan.node_of_qubit[0])
    axis = tensor.inds.index(state.site_ind(0))
    selection = [slice(None)] * tensor.ndim
    selection[axis] = slice(0, 1)
    tensor.modify(data=tensor.data[tuple(selection)])
    state.invalidate_canonical_form()
    sampler = TreeSampler(state, strategy="factor")
    context = _FactorSamplingContext(sampler, 1024, 4096)
    assert context.radix is not None
    configs = np.zeros((3, 64), dtype=np.int64)
    configs[1, 1:] = 1
    configs[2, -1] = 1
    keys, indices, inverse = context.groups(configs)
    np.testing.assert_array_equal(keys, [0, 1, 2**63 - 1])
    np.testing.assert_array_equal(configs[indices][inverse], configs)


@pytest.mark.parametrize("backend", ["numpy", "torch", "torch_cuda", "cupy"])
def test_overflow_subtree_row_order_preserves_bell_pairs(force_grouping, backend):
    plan = TreePlan.from_order(range(70), structure="balanced")
    # Construction replays once. Only these four sites are populated, while
    # the 70 binary physical legs still require the overflow row-key fallback.
    optimizer = TreeOptimizer(
        [(h(), 0), (h(), 1), (cnot(), (0, 35)), (cnot(), (1, 36))],
        n=70, tree=plan, chi=4, mode="direct", cutoff=0.0,
    )
    _backend(optimizer.tn, backend)
    sampler = TreeSampler(optimizer, strategy="factor", chunk_size=7)
    actual = sampler.sample_batch(31, seed=17).to_numpy()
    np.testing.assert_array_equal(actual.configs[:, 0], actual.configs[:, 35])
    np.testing.assert_array_equal(actual.configs[:, 1], actual.configs[:, 36])
    other = [q for q in range(70) if q not in {0, 1, 35, 36}]
    assert not np.any(actual.configs[:, other])
    np.testing.assert_allclose(actual.probs, 0.25, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"strategy": "bad"}, {"cache_bytes": -1}, {"cache_bytes": True},
    {"workspace_bytes": 0}, {"workspace_bytes": 1.5},
])
def test_invalid_factor_options(kwargs):
    with pytest.raises(ValueError):
        TreeSampler(None, **kwargs)
