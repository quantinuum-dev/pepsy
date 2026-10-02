"""Native canonical-factor sampling against independent graded references."""

import gc
from itertools import product
import weakref

import autoray as ar
import numpy as np
import pytest

from pepsy.tensors import Fermion, hrs_to_ttn
from pepsy.optimizers import TreePlan, TreeTensorNetwork
from pepsy.sampling import TreeSampler
from pepsy.sampling._tree_symmray import _SymmrayFactorContext

pytestmark = [pytest.mark.tree, pytest.mark.integration, pytest.mark.optional]


def _state(kind, root=None, n=4):
    pytest.importorskip("symmray")
    plan = TreePlan.from_order(
        [q for q in range(n) if q != root], root_qubit=root,
        max_arity=3, top_arity=3 if root is None else 2,
    )
    if kind == "abelian":
        return TreeTensorNetwork.from_symmray_plan(
            plan, symmetry="U1", physical_sectors={0: 1, 1: 2},
            leaf_charges={q: q % 2 for q in range(n)},
            bond_dim=5, fermionic=False, seed=19,
        )
    fermion = Fermion(spinful=kind != "spinless", symmetry="Z2" if kind == "spinless" else kind)
    return hrs_to_ttn(
        n, tree=plan, fermion=fermion,
        occupations=fermion.half_filled_occupations(n), chi=5, seed=19,
    )


def _backend(state, backend, dtype="complex128"):
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        if backend == "torch_cuda" and not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        state.apply_to_arrays(lambda a: torch.as_tensor(
            np.asfortranarray(a), dtype=getattr(torch, dtype),
            device="cuda:0" if backend == "torch_cuda" else "cpu",
        ))
    elif backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        state.apply_to_arrays(lambda a: cp.asarray(a, dtype=dtype))


@pytest.mark.parametrize("kind", ["abelian", "spinless", "U1", "U1U1", "Z2Z2"])
@pytest.mark.parametrize("backend", ["numpy", "torch", "torch_cuda", "cupy"])
@pytest.mark.parametrize("root", [None, 2])
def test_native_factor_matches_standard_and_dense_without_densifying(monkeypatch, kind, backend, root):
    state = _state(kind, root)
    contracted = state.contract(all).transpose(*(state.site_ind(q) for q in range(4))).data
    source_indices = [state.node_tensor(state.node_of_qubit(q)).data.indices[
        state.node_tensor(state.node_of_qubit(q)).inds.index(state.site_ind(q))
    ] for q in range(4)]
    source_codes = [{metadata: code for code, metadata in enumerate(TreeSampler._symmray_code_metadata(index))}
                    for index in source_indices]
    positions = [[source_codes[q][metadata] for metadata in TreeSampler._symmray_code_metadata(index)]
                 for q, index in enumerate(contracted.indices)]
    exact = np.zeros(tuple(len(codes) for codes in source_codes))
    exact[np.ix_(*positions)] = abs(np.asarray(contracted.to_dense())) ** 2
    exact /= exact.sum()
    _backend(state, backend)
    originals = [t.data for t in state.tensors]
    original_blocks = [[np.asarray(ar.to_numpy(b)).copy() for b in a.blocks.values()] for a in originals]
    source_center = state.orthogonality_center

    def fail_dense(*args, **kwargs):
        raise AssertionError("native factor sampling must not densify Symmray arrays")

    monkeypatch.setattr(type(originals[0]), "to_dense", fail_dense)
    for chunk, cache in [(None, 0), (3, 128), (3, 128 * 1024**2)]:
        sampler = TreeSampler(state, backend="native", chunk_size=chunk, cache_bytes=cache, seed=11)
        assert sampler.resolved_backend == "symmray"
        # A fresh reference has the same persistent generator for each case.
        reference = TreeSampler(state, backend="symmray", strategy="standard", seed=11)
        for seed in (17, None):
            expected = reference.sample_batch(9, seed=seed).to_numpy()
            result = sampler.sample_batch(9, seed=seed)
            actual = result.to_numpy()
            np.testing.assert_array_equal(actual.configs, expected.configs)
            np.testing.assert_allclose(actual.probs, expected.probs, rtol=2e-10, atol=1e-12)
            np.testing.assert_allclose(actual.probs, exact[tuple(actual.configs.T)], rtol=2e-10, atol=1e-12)
            np.testing.assert_allclose(sampler.probabilities(actual.configs), actual.probs, rtol=2e-10, atol=1e-12)
            assert actual.probs.dtype == np.float64
            if backend.startswith("torch") or backend == "cupy":
                assert result.configs.device == result.probs.device == originals[0].get_any_array().device
    assert state.orthogonality_center == source_center
    for tensor, original, blocks in zip(state.tensors, originals, original_blocks):
        assert tensor.data is original
        for value, saved in zip(tensor.data.blocks.values(), blocks):
            np.testing.assert_array_equal(ar.to_numpy(value), saved)


@pytest.mark.parametrize("kind", ["abelian", "U1"])
@pytest.mark.parametrize("backend", ["torch", "torch_cuda"])
def test_native_factor_probability_and_score_gradients_match_dense(kind, backend):
    torch = pytest.importorskip("torch")
    state = _state(kind)
    _backend(state, backend)
    for tensor in state.tensors:
        data = tensor.data.copy()
        data.apply_to_arrays(lambda b: b.detach().requires_grad_())
        tensor.modify(data=data)
    state.invalidate_canonical_form()
    inputs = tuple(b for t in state.tensors for b in t.data.blocks.values())
    sampler = TreeSampler(state, backend="native", chunk_size=3)
    configs, actual = sampler.sample_arrays(7, seed=17)
    vector = state.contract(all).transpose(*(state.site_ind(q) for q in range(4))).data.to_dense()
    exact = vector.abs() ** 2
    exact = exact / exact.sum()
    expected = exact[tuple(configs.T)]
    scores = sampler.probabilities(configs, to_numpy=False)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=1e-12)
    torch.testing.assert_close(scores, expected, rtol=2e-10, atol=1e-12)
    expected_grads = torch.autograd.grad(expected.sum(), inputs, retain_graph=True, allow_unused=True)
    for values in (actual, scores):
        gradients = torch.autograd.grad(values.sum(), inputs, retain_graph=True, allow_unused=True)
        for value, reference, source in zip(gradients, expected_grads, inputs):
            torch.testing.assert_close(
                torch.zeros_like(source) if value is None else value,
                torch.zeros_like(source) if reference is None else reference,
                rtol=2e-9, atol=2e-10,
            )


@pytest.mark.parametrize("fail", [False, True])
def test_native_factor_reuses_prefixes_and_releases_cache(monkeypatch, fail):
    sampler = TreeSampler(_state("abelian"), backend="native", chunk_size=2)
    contexts, cached_factors, computations = [], [], []
    original = _SymmrayFactorContext.conditional

    def conditional(context, branch, depth, prefix):
        if prefix not in context.cache:
            computations.append(prefix)
        result = original(context, branch, depth, prefix)
        contexts.append(weakref.ref(context))
        cached_factors.extend(weakref.ref(entry[0]) for entry in context.cache.values())
        assert context.cache_used <= sampler.cache_bytes
        if fail:
            raise RuntimeError("injected native factor failure")
        return result

    monkeypatch.setattr(_SymmrayFactorContext, "conditional", conditional)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        if fail:
            with pytest.raises(RuntimeError, match="injected"):
                sampler.sample_arrays(12, seed=17)
        else:
            sampler.sample_arrays(12, seed=17)
            assert computations.count(()) == 1
        assert contexts and cached_factors
        assert all(ref() is None for ref in contexts + cached_factors)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("kind", ["abelian", "U1"])
def test_native_factor_amplitudes_preserve_graded_source_order(kind):
    state = _state(kind, root=2)
    dims = tuple(state.node_tensor(state.node_of_qubit(q)).ind_size(state.site_ind(q)) for q in range(4))
    configurations = np.asarray(list(product(*(range(dim) for dim in dims))))
    norm = TreeSampler._symmray_norm_squared(state) ** 0.5
    expected = []
    for configuration in configurations:
        branch = state.isel({state.site_ind(q): int(code) for q, code in enumerate(configuration)})
        expected.append(TreeSampler._symmray_scalar(branch.contract(all)))
    expected = np.asarray(expected).reshape(dims) / norm
    sampler = TreeSampler(state, backend="native", chunk_size=3)
    before = sampler.amplitudes(configurations).reshape(dims)
    sampler.sample_arrays(9, seed=17)
    actual = sampler.amplitudes(configurations).reshape(dims)
    np.testing.assert_array_equal(actual, before)
    # A global canonical gauge phase is immaterial; relative signs are not.
    index = np.unravel_index(np.argmax(abs(expected)), dims)
    phase = actual[index] / expected[index]
    np.testing.assert_allclose(actual, phase * expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("chunk", [None, 7, 8])
def test_native_single_chunk_does_not_retain_unusable_prefix_cache(monkeypatch, chunk):
    sampler = TreeSampler(_state("abelian"), backend="native", chunk_size=chunk)
    expected = TreeSampler(sampler._source, backend="native", strategy="standard").sample_arrays(7, seed=17)
    original = _SymmrayFactorContext.conditional
    visited = []

    def conditional(context, branch, depth, prefix):
        result = original(context, branch, depth, prefix)
        assert context.cache_bytes == context.cache_used == 0
        assert not context.cache and not context.source_blocks
        visited.append(prefix)
        return result

    monkeypatch.setattr(_SymmrayFactorContext, "conditional", conditional)
    actual = sampler.sample_arrays(7, seed=17)
    assert len(visited) == len(set(visited))
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_native_complex64_long_product_preserves_draws_and_small_probabilities(backend):
    pytest.importorskip("symmray")
    n, count, seed = 160, 3, 17
    plan = TreePlan.from_order(range(n), max_arity=2)
    state = TreeTensorNetwork.from_symmray_plan(
        plan, symmetry="U1", physical_sectors={0: 2},
        leaf_charges={q: 0 for q in range(n)}, bond_dim=1,
        fermionic=False, dtype="complex128", seed=19,
    )
    for node in plan.children:
        tensor = state.node_tensor(node)
        tensor.data.apply_to_arrays(lambda b: np.full(
            b.shape, 1 / np.sqrt(2) if node in plan.qubit_of_node else 1,
            dtype=np.complex64,
        ))
    state.invalidate_canonical_form()
    _backend(state, backend, dtype="complex64")
    sampler = TreeSampler(state, backend="native", chunk_size=2, cache_bytes=0)
    result = sampler.sample_batch(count, seed=seed).to_numpy()
    expected = np.empty((count, n), dtype=np.int64)
    expected[:, sampler._sampling_qubits] = np.random.default_rng(seed).random((count, n)) >= 0.5
    np.testing.assert_array_equal(result.configs, expected)
    assert np.all(result.probs > 0)
    np.testing.assert_allclose(result.probs, 2.0**-n, rtol=3e-5, atol=0)


@pytest.mark.parametrize("strategy", ["standard", "factor"])
@pytest.mark.parametrize("exponent", [-200.0, 200.0])
def test_native_capture_removes_global_exponent_without_mutating_source(strategy, exponent):
    state = _state("U1", root=2)
    baseline = TreeSampler(state, backend="native", strategy=strategy)
    configurations = baseline.sample_arrays(7, seed=17)[0]
    expected_amplitudes = baseline.amplitudes(configurations)
    expected_probabilities = baseline.probabilities(configurations)
    state.exponent = exponent
    sampler = TreeSampler(state, backend="native", strategy=strategy, chunk_size=2)
    np.testing.assert_allclose(sampler.amplitudes(configurations), expected_amplitudes)
    np.testing.assert_allclose(sampler.probabilities(configurations), expected_probabilities)
    actual, probabilities = sampler.sample_arrays(7, seed=17)
    np.testing.assert_array_equal(actual, configurations)
    np.testing.assert_allclose(probabilities, expected_probabilities)
    assert state.exponent == exponent
    assert sampler._symmray_state["tn"].exponent == 0.0


def test_native_single_site_zero_uniform_skips_zero_weight():
    pytest.importorskip("symmray")
    plan = TreePlan.from_order([], root_qubit=0)
    state = TreeTensorNetwork.from_symmray_plan(
        plan, symmetry="U1", physical_sectors={0: 2},
        leaf_charges={0: 0}, bond_dim=1, fermionic=False, seed=19,
    )
    state.node_tensor(state.root).data.apply_to_arrays(
        lambda block: np.asarray([0.0, 1.0], dtype=block.dtype).reshape(block.shape)
    )
    state.invalidate_canonical_form()
    sampler = TreeSampler(state, backend="native", chunk_size=2)

    class ZeroDraws:
        def random(self, shape):
            return np.zeros(shape)

    sampler._rng = ZeroDraws()
    configs, probabilities = sampler.sample_arrays(3)
    np.testing.assert_array_equal(configs, np.ones((3, 1), dtype=np.int64))
    np.testing.assert_array_equal(probabilities, np.ones(3))


def test_native_factor_reentrant_sampling_owns_separate_cache(monkeypatch):
    sampler = TreeSampler(_state("abelian"), backend="native", chunk_size=2)
    expected = sampler.sample_arrays(7, seed=17)
    original = _SymmrayFactorContext.conditional
    nested = False

    def conditional(context, branch, depth, prefix):
        nonlocal nested
        if not nested:
            nested = True
            actual = sampler.sample_arrays(7, seed=17)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])
        return original(context, branch, depth, prefix)

    monkeypatch.setattr(_SymmrayFactorContext, "conditional", conditional)
    actual = sampler.sample_arrays(7, seed=17)
    assert nested
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_native_factor_cdf_rounding_does_not_select_trailing_zero(monkeypatch):
    pytest.importorskip("symmray")
    state = TreeTensorNetwork.from_symmray_plan(
        TreePlan.from_order([], root_qubit=0), symmetry="U1",
        physical_sectors={0: 3}, leaf_charges={0: 0}, bond_dim=1,
        fermionic=False, seed=19,
    )
    sampler = TreeSampler(state, backend="native", chunk_size=2)
    original = _SymmrayFactorContext.conditional

    def conditional(context, branch, depth, prefix):
        gauged, codes, norms, probabilities, host = original(context, branch, depth, prefix)
        # Exercise the final one-ulp CDF gap with a trailing zero sector.
        # Native source codes stay valid; only the distribution fixture changes.
        host = np.asarray([0.1, 0.3, 0.0])
        host /= host.sum()
        assert np.cumsum(host)[-1] < 1.0
        return gauged, codes, norms, host, host

    class LargestDraws:
        def random(self, shape):
            return np.full(shape, np.nextafter(1.0, 0.0))

    sampler._rng = LargestDraws()
    monkeypatch.setattr(_SymmrayFactorContext, "conditional", conditional)
    configurations, probabilities = sampler.sample_arrays(3)
    np.testing.assert_array_equal(configurations, np.ones((3, 1), dtype=np.int64))
    np.testing.assert_allclose(probabilities, 0.75)


@pytest.mark.parametrize("spinful,symmetry", [
    (False, "Z2"), (False, "U1"), (True, "Z2"),
    (True, "U1"), (True, "U1U1"), (True, "Z2Z2"),
])
@pytest.mark.parametrize("root", [None, 1])
def test_native_factor_preserves_odd_global_parity_and_relative_signs(spinful, symmetry, root):
    pytest.importorskip("symmray")
    fermion = Fermion(spinful=spinful, symmetry=symmetry)
    occupied = (0, 1) if symmetry in {"U1U1", "Z2Z2"} else 1
    plan = TreePlan.from_order([q for q in range(3) if q != root], root_qubit=root)
    state = hrs_to_ttn(
        3, tree=plan, fermion=fermion,
        occupations=[occupied, fermion.zero_charge, fermion.zero_charge], chi=5, seed=19,
    )
    sampler = TreeSampler(state, backend="native", fermion=fermion, chunk_size=3)
    reference = TreeSampler(state, backend="native", fermion=fermion, strategy="standard")
    actual = sampler.sample_batch(11, seed=17).to_numpy()
    expected = reference.sample_batch(11, seed=17).to_numpy()
    np.testing.assert_array_equal(actual.configs, expected.configs)
    np.testing.assert_allclose(actual.probs, expected.probs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sampler.probabilities(actual.configs), actual.probs, rtol=1e-10, atol=1e-12)
    source_amplitudes = []
    for configuration in actual.configs:
        selected = state.isel({state.site_ind(q): int(code) for q, code in enumerate(configuration)})
        source_amplitudes.append(TreeSampler._symmray_scalar(selected.contract(all)))
    source_amplitudes = np.asarray(source_amplitudes) / TreeSampler._symmray_norm_squared(state) ** 0.5
    amplitudes = sampler.amplitudes(actual.configs)
    index = np.argmax(abs(source_amplitudes))
    phase = amplitudes[index] / source_amplitudes[index]
    np.testing.assert_allclose(amplitudes, phase * source_amplitudes, rtol=1e-10, atol=1e-12)
    occupations = actual.occupations()
    np.testing.assert_array_equal(
        occupations.sum(axis=tuple(range(1, occupations.ndim))) % 2,
        np.ones(11, dtype=int),
    )
