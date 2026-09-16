"""Tests for :class:`pepsy.TreeSampler`, the tree-tensor-network perfect sampler."""

from itertools import product

import numpy as np
import pytest

import pepsy
from pepsy.optimizers.tree import TreeOptimizer, TreePlan
from pepsy.sampling import (
    TreeBatchSampleResult,
    TreeSampleResult,
    TreeSampler,
)


# -- helpers ------------------------------------------------------------------


def _rand_unitary(k, rng):
    m = rng.standard_normal((2**k, 2**k)) + 1j * rng.standard_normal((2**k, 2**k))
    q, _ = np.linalg.qr(m)
    return q


def _random_stream(n, ngates, rng, two_qubit_frac=0.5):
    stream = []
    for _ in range(ngates):
        if n >= 2 and rng.random() < two_qubit_frac:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((_rand_unitary(2, rng), (int(a), int(b))))
        else:
            stream.append((_rand_unitary(1, rng), int(rng.integers(n))))
    return stream


def _all_configs(n):
    """All ``2**n`` basis configs; row ``idx`` has qubit 0 as the MSB."""
    return np.array(
        [[(idx >> (n - 1 - q)) & 1 for q in range(n)] for idx in range(2**n)]
    )


def _exact_probs(opt, n):
    psi = opt.to_dense()
    psi = psi / np.linalg.norm(psi)
    return psi, np.abs(psi) ** 2


# -- amplitude / probability correctness --------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 5])
@pytest.mark.parametrize("seed", [0, 1])
def test_probabilities_match_statevector(n, seed):
    rng = np.random.default_rng(seed)
    stream = _random_stream(n, 6 * n + 4, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=128)
    _, exact = _exact_probs(opt, n)

    sampler = TreeSampler(opt, seed=seed)
    cfg = _all_configs(n)
    probs = sampler.probabilities(cfg)
    assert probs.shape == (2**n,)
    assert np.allclose(probs.sum(), 1.0, atol=1e-10)
    assert np.max(np.abs(probs - exact)) < 1e-10


def test_amplitudes_match_statevector_up_to_phase():
    n = 5
    rng = np.random.default_rng(7)
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi, _ = _exact_probs(opt, n)

    sampler = TreeSampler(opt)
    amps = sampler.amplitudes(_all_configs(n))
    k = int(np.argmax(np.abs(psi)))
    phase = psi[k] / amps[k]
    assert np.max(np.abs(amps * phase - psi)) < 1e-10
    # probabilities are |amplitudes|**2.
    assert np.allclose(sampler.probabilities(_all_configs(n)), np.abs(amps) ** 2)


def test_tree_flip_ratios_match_batched_amplitude_reference():
    """Tree local-energy ratios use one message pass and remain exact."""
    n = 5
    rng = np.random.default_rng(81)
    opt = TreeOptimizer(
        _random_stream(n, 32, rng, two_qubit_frac=0.7),
        n=n,
        chi=128,
    )
    sampler = TreeSampler(opt)
    configs = _all_configs(n)[::3]
    amplitudes = sampler.amplitudes(configs, to_numpy=False)
    flipped = np.repeat(configs[:, None, :], n, axis=1)
    for qubit in range(n):
        flipped[:, qubit, qubit] ^= 1
    flipped_amplitudes = sampler.amplitudes(
        flipped.reshape(-1, n),
        to_numpy=False,
    ).reshape(-1, n)
    expected = flipped_amplitudes / amplitudes[:, None]
    actual = sampler.single_site_flip_amplitude_ratios(
        configs,
        to_numpy=False,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_tree_sampler_entropy_reuses_canonical_edge_spectra():
    n = 4
    rng = np.random.default_rng(82)
    opt = TreeOptimizer(
        _random_stream(n, 24, rng, two_qubit_frac=0.7),
        n=n,
        chi=128,
    )
    sampler = TreeSampler(opt)
    entropies, edges = sampler.tree_edge_entropies(return_edges=True)
    expected, expected_edges = opt.tree_edge_entropies(return_edges=True)
    assert edges == expected_edges
    np.testing.assert_allclose(entropies, expected, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        sampler.tree_edge_entropies(method="eig"),
        entropies,
        atol=1e-11,
        rtol=1e-11,
    )


def test_physical_root_probabilities_and_amplitudes_match_statevector():
    """Sampling treats the optional root physical leg as an ordinary site."""
    n = 5
    root_qubit = 2
    rng = np.random.default_rng(71)
    plan = TreePlan.from_order(
        [0, 1, 3, 4], structure="balanced", root_qubit=root_qubit,
    )
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, tree=plan, chi=128)
    psi, exact = _exact_probs(opt, n)
    configs = _all_configs(n)

    sampler = TreeSampler(opt, seed=0)
    amplitudes = sampler.amplitudes(configs)
    probabilities = sampler.probabilities(configs)
    pivot = int(np.argmax(np.abs(psi)))
    phase = psi[pivot] / amplitudes[pivot]

    assert np.max(np.abs(amplitudes * phase - psi)) < 1e-10
    assert np.max(np.abs(probabilities - exact)) < 1e-10
    result = sampler.sample_batch(32, seed=3)
    assert result.configs.shape == (32, n)
    assert np.allclose(
        result.probs, sampler.probabilities(result.configs), atol=1e-12
    )


# -- empirical sampling -------------------------------------------------------


def test_sample_frequencies_converge_to_born():
    n = 4
    rng = np.random.default_rng(3)
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=64)
    _, exact = _exact_probs(opt, n)

    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(200_000, seed=42)
    assert res.configs.shape == (200_000, n)
    assert res.probs.shape == (200_000,)

    idx = np.zeros(len(res), dtype=int)
    for q in range(n):
        idx = idx * 2 + res.configs[:, q]
    freq = np.bincount(idx, minlength=2**n) / len(res)
    assert np.max(np.abs(freq - exact)) < 5e-3

    # Reported per-sample probs equal probabilities() of the same configs.
    rep = sampler.probabilities(res.configs[:100])
    assert np.allclose(res.probs[:100], rep, atol=1e-12)


def test_computational_basis_state_is_deterministic():
    # |0110> product state -> a single config with probability 1.
    n = 4
    stream = [(pepsy.x(), 1), (pepsy.x(), 2)]
    opt = TreeOptimizer(stream, n=n, chi=8)
    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(64, seed=1)
    expected = np.tile(np.array([0, 1, 1, 0]), (64, 1))
    assert np.array_equal(res.configs, expected)
    assert np.allclose(res.probs, 1.0)


def test_single_qubit_superposition():
    opt = TreeOptimizer([(pepsy.h(), 0)], n=1)
    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(4000, seed=0)
    assert res.configs.shape == (4000, 1)
    assert np.allclose(res.probs, 0.5)
    assert abs(res.configs.mean() - 0.5) < 0.05


# -- reproducibility & API ----------------------------------------------------


def test_seed_reproducibility():
    n = 4
    rng = np.random.default_rng(11)
    stream = _random_stream(n, 20, rng)
    opt = TreeOptimizer(stream, n=n, chi=16)
    sampler = TreeSampler(opt)
    a1, p1 = sampler.sample_arrays(200, seed=5)
    a2, p2 = sampler.sample_arrays(200, seed=5)
    assert np.array_equal(a1, a2)
    assert np.allclose(p1, p2)
    # A fresh sampler with a persistent seed reproduces the seedless stream.
    s_a = TreeSampler(opt, seed=99).sample_arrays(50)[0]
    s_b = TreeSampler(opt, seed=99).sample_arrays(50)[0]
    assert np.array_equal(s_a, s_b)


def test_accepts_optimizer_and_tensor_network():
    n = 4
    rng = np.random.default_rng(2)
    stream = _random_stream(n, 20, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=64)
    _, exact = _exact_probs(opt, n)
    cfg = _all_configs(n)

    from_opt = TreeSampler(opt).probabilities(cfg)
    from_tn = TreeSampler(opt.tn).probabilities(cfg)
    assert np.max(np.abs(from_opt - exact)) < 1e-10
    assert np.allclose(from_opt, from_tn)


def test_batch_result_helpers():
    n = 3
    rng = np.random.default_rng(4)
    stream = _random_stream(n, 15, rng)
    opt = TreeOptimizer(stream, n=n, chi=16)
    sampler = TreeSampler(opt, seed=0)
    res = sampler.sample_batch(10, seed=0)

    assert isinstance(res, TreeBatchSampleResult)
    assert res.n_samples == 10 == len(res)
    assert res.nqubits == n

    listed = res.configs_list()
    assert listed == [[int(v) for v in row] for row in res.configs]

    mags = res.magnetizations()
    assert mags.shape == (10,)
    expected = (1 - 2 * res.configs.astype(float)).sum(axis=1) / n
    assert np.allclose(mags, expected)

    sr = res.to_sample_result()
    assert isinstance(sr, TreeSampleResult)
    assert len(sr) == 10
    assert sr.configs == listed
    assert np.allclose(sr.magnetizations(), mags)

    np_copy = res.to_numpy()
    assert np.array_equal(np_copy.configs, res.configs)
    assert np.array_equal(np_copy.probs, res.probs)


def test_tree_sampler_preserves_torch_backend_and_supports_host_copy():
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    state = TreeOptimizer(None, n=3).tn
    state.apply_to_arrays(to_backend)
    h = to_backend(
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    )
    opt = TreeOptimizer([(h, 0)], state=state, run=True, mode="direct")

    sampler = TreeSampler(opt, backend="native", seed=0)
    assert sampler.resolved_backend == "torch"
    configs = _all_configs(3)
    probs = sampler.probabilities(configs, to_numpy=False)
    assert torch.is_tensor(probs)
    assert np.allclose(probs.detach().cpu().numpy(), sampler.probabilities(configs))

    result = sampler.sample_batch(32, seed=3)
    assert result.backend == "torch"
    assert torch.is_tensor(result.configs)
    assert torch.is_tensor(result.probs)
    assert torch.is_tensor(result.magnetizations(to_numpy=False))
    host = result.to_numpy()
    assert isinstance(host.configs, np.ndarray)
    assert np.allclose(host.probs, sampler.probabilities(host.configs))

    host_sampler = TreeSampler(opt, backend="numpy")
    host_configs, host_probs = host_sampler.sample_arrays(8, seed=4)
    assert isinstance(host_configs, np.ndarray)
    assert isinstance(host_probs, np.ndarray)


def test_tree_sampler_rejects_explicit_backend_mismatch():
    torch = pytest.importorskip("torch")
    del torch
    opt = TreeOptimizer(None, n=2)
    with pytest.raises(ValueError, match="requires a live torch tree"):
        TreeSampler(opt, backend="torch")


def test_tree_sampler_preserves_cupy_backend_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    to_backend = lambda array: cupy.asarray(array, dtype=cupy.complex128)
    state = TreeOptimizer(None, n=2).tn
    state.apply_to_arrays(to_backend)
    opt = TreeOptimizer([(to_backend(pepsy.h()), 0)], state=state, mode="direct")
    sampler = TreeSampler(opt, backend="native")

    result = sampler.sample_batch(16, seed=2)
    assert sampler.resolved_backend == "cupy"
    assert isinstance(result.configs, cupy.ndarray)
    assert isinstance(result.probs, cupy.ndarray)
    assert isinstance(sampler.probabilities(result.configs, to_numpy=False), cupy.ndarray)
    assert isinstance(result.to_numpy().configs, np.ndarray)


def test_sample_returns_list_result():
    opt = TreeOptimizer([(pepsy.h(), 0), (pepsy.cnot(), (0, 1))], n=2)
    res = TreeSampler(opt, seed=0).sample(5)
    assert isinstance(res, TreeSampleResult)
    assert len(res) == 5
    assert all(len(c) == 2 for c in res.configs)


def test_refresh_recaptures_state():
    n = 3
    rng = np.random.default_rng(6)
    opt = TreeOptimizer(_random_stream(n, 12, rng), n=n, chi=16)
    sampler = TreeSampler(opt, seed=0)
    before = sampler.probabilities(_all_configs(n))

    # Mutate the optimizer state, then refresh from the same source object.
    opt.apply_gate(pepsy.x(), 0)
    sampler.refresh()
    after = sampler.probabilities(_all_configs(n))
    assert not np.allclose(before, after)
    _, exact = _exact_probs(opt, n)
    assert np.max(np.abs(after - exact)) < 1e-10


# -- validation & exports -----------------------------------------------------


def test_bad_n_samples_raises():
    opt = TreeOptimizer([(pepsy.h(), 0)], n=1)
    with pytest.raises(ValueError):
        TreeSampler(opt).sample_arrays(0)


def test_bad_configs_shape_raises():
    opt = TreeOptimizer([(pepsy.h(), 0), (pepsy.cnot(), (0, 1))], n=2)
    sampler = TreeSampler(opt)
    with pytest.raises(ValueError):
        sampler.probabilities(np.zeros((3, 5), dtype=int))


def test_bad_state_type_raises():
    with pytest.raises(TypeError):
        TreeSampler(object())


def test_public_api_exports_tree_sampler():
    assert pepsy.TreeSampler is TreeSampler
    assert pepsy.TreeSampleResult is TreeSampleResult
    assert pepsy.TreeBatchSampleResult is TreeBatchSampleResult
    for name in ("TreeSampler", "TreeSampleResult", "TreeBatchSampleResult"):
        assert name in pepsy.__all__


# -- fermionic tree sampling --------------------------------------------------


def _fermionic_tree(*, chi=64, steps=4, root_qubit=None):
    """Build a mildly entangled U1U1 spinful Fermi-Hubbard tree (L=4)."""
    from pepsy.optimizers.tree import TreeLayoutFinder

    tensors = pepsy.tensors
    Lx, Ly, L = 2, 2, 4
    t, U, dt = 1.0, 4.0, 0.05
    dtype = "complex128"

    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype=dtype)
    setup = fermion.lattice_half_filling(Lx, Ly, pattern="checkerboard", cyclic=True)
    mapper = tensors.OneDMap(Lx, Ly, mode="snake")
    _, coo2idx = mapper.build()
    edges_1d = [tuple(sorted((coo2idx[a], coo2idx[b]))) for a, b in setup.edges]
    occ_1d = {coo2idx[coo]: c for coo, c in setup.occupations.items()}
    occupations = tuple(occ_1d[p] for p in range(L))
    target = tuple(int(v) for v in np.sum(np.array(occupations), axis=0))

    plan = TreeLayoutFinder(
        [(fermion.hopping_gate(0.1, t=t, imaginary=False), e) for e in edges_1d],
        n=L, chi=8, objective="hybrid", root_qubit=root_qubit,
    ).recommend_arities((2, 3, 4), seed=0)["plan"]
    seed_ttn = pepsy.ps_to_ttn(
        L, tree=plan, fermion=fermion, occupations=occupations, dtype=dtype
    )

    half = dt / 2
    u_hop = fermion.hopping_gate(half, t=t, imaginary=False)
    onsite = [
        (fermion.onsite_gate(half, site=s, U=U, mu=0.0, imaginary=False), s)
        for s in range(L)
    ]
    layers = fermion.edge_coloring_layers(edges_1d)
    fwd = [(u_hop, e) for layer in layers for e in layer]
    rev = [(u_hop, e) for layer in reversed(layers) for e in reversed(layer)]
    gates = (onsite + fwd + rev + onsite) * steps

    engine = TreeOptimizer(
        gates, n=L, tree=plan, state=seed_ttn.copy(), chi=chi,
        cutoff=0.0, mode="mpo", run=False,
    )
    engine.run()
    return engine, fermion, target, L


def _all_base_d_configs(n, d):
    """All ``d**n`` configs with site 0 as the most significant digit."""
    return np.array(
        [[(idx // d ** (n - 1 - q)) % d for q in range(n)] for idx in range(d**n)]
    )


@pytest.mark.parametrize("root_qubit", [None, 3])
def test_fermionic_tree_probabilities_match_statevector(root_qubit):
    pytest.importorskip("symmray")
    engine, fermion, target, L = _fermionic_tree(
        chi=64, root_qubit=root_qubit,
    )
    sampler = TreeSampler(engine, fermion=fermion, seed=0)
    assert sampler._configuration_encoding is not None

    # Exact graded statevector densified into Symmray's dense basis order.
    tn = engine.p
    site_inds = [tn.site_ind(q) for q in range(L)]
    sv = np.asarray(
        tn.contract(all).transpose(*site_inds).data.to_dense()
    ).reshape(-1)
    sv = sv / np.linalg.norm(sv)
    exact = np.abs(sv) ** 2

    configs = _all_base_d_configs(L, 4)
    probs = sampler.probabilities(configs)
    assert probs.shape == (4**L,)
    assert np.allclose(probs.sum(), 1.0, atol=1e-10)
    # Dense-basis code index aligns 1:1 with the densified statevector.
    assert np.max(np.abs(probs - exact)) < 1e-10


def test_fermionic_tree_samples_conserve_charge_and_decode():
    pytest.importorskip("symmray")
    engine, fermion, target, L = _fermionic_tree(chi=64)
    sampler = TreeSampler(engine, fermion=fermion, seed=0)

    res = sampler.sample_batch(2000, seed=42)
    assert res.configs.shape == (2000, L)
    assert res.configuration_encoding is not None

    occ = res.occupations()
    assert occ.shape == (2000, L, 2)
    assert set(np.unique(occ).tolist()) <= {0, 1}
    # Every shot lives in the single (n_up, n_down) charge sector of the state.
    totals = {tuple(int(v) for v in row.sum(0)) for row in occ}
    assert totals == {target}

    # Reported per-sample probs equal probabilities() of the same configs.
    rep = sampler.probabilities(res.configs[:200])
    assert np.allclose(res.probs[:200], rep, atol=1e-12)

    # List-based result carries the same decoder.
    sr = res.to_sample_result()
    assert isinstance(sr, TreeSampleResult)
    assert np.array_equal(sr.occupations(), occ)


def test_fermionic_tree_sample_frequencies_converge():
    pytest.importorskip("symmray")
    engine, fermion, _, L = _fermionic_tree(chi=64)
    sampler = TreeSampler(engine, fermion=fermion, seed=0)

    res = sampler.sample_batch(40000, seed=7)
    uniq, inv = np.unique(res.configs, axis=0, return_inverse=True)
    freq = np.bincount(inv, minlength=len(uniq)) / len(res)
    assert np.max(np.abs(freq - sampler.probabilities(uniq))) < 1e-2


def test_non_fermionic_occupations_raises():
    opt = TreeOptimizer([(pepsy.h(), 0), (pepsy.cnot(), (0, 1))], n=2)
    res = TreeSampler(opt, seed=0).sample_batch(4, seed=0)
    assert res.configuration_encoding is None
    with pytest.raises(ValueError, match="no fermion configuration encoding"):
        res.occupations()


def test_tree_sampler_symmray_generic_abelian_stays_block_sparse(monkeypatch):
    """Generic Abelian TTNs use the native path without densifying tensors."""
    pytest.importorskip("symmray")
    plan = TreePlan.from_order(range(4), max_arity=2, top_arity=2)
    state = pepsy.TreeTensorNetwork.from_symmray_plan(
        plan,
        symmetry="U1",
        physical_sectors={0: 1, 1: 1},
        leaf_charges={0: 0, 1: 1, 2: 0, 3: 1},
        bond_dim=4,
        fermionic=False,
        seed=3,
    )

    def fail_dense(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("native TreeSampler must not densify Symmray data")

    monkeypatch.setattr(type(state.node_tensor(0).data), "to_dense", fail_dense)
    sampler = TreeSampler(state, backend="symmray", seed=7)
    configs = np.asarray(list(product(range(2), repeat=4)), dtype=np.int64)
    amplitudes = sampler.amplitudes(configs)
    probabilities = sampler.probabilities(configs)

    assert sampler.resolved_backend == "symmray"
    assert sampler.physical_code_maps == (
        {0: (0, 0), 1: (1, 0)},
    ) * 4
    np.testing.assert_allclose(probabilities, np.abs(amplitudes) ** 2, atol=1e-12)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)

    sampled_configs, sampled_probs = sampler.sample_arrays(8, seed=11)
    np.testing.assert_allclose(
        sampler.probabilities(sampled_configs), sampled_probs, atol=1e-12
    )
    entropies = sampler.tree_edge_entropies()
    assert entropies.shape == (len(plan.parent),)
    assert np.all(np.isfinite(entropies))


@pytest.mark.parametrize("symmetry", ("U1",))
def test_tree_sampler_symmray_fermionic_native_matches_statevector(symmetry):
    """Native fermionic tree probabilities agree with a dense oracle."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry=symmetry)
    state = pepsy.hrs_to_ttn(
        4,
        fermion=fermion,
        occupations=fermion.half_filled_occupations(4),
        chi=4,
        seed=17,
    )
    sampler = TreeSampler(state, backend="symmray", fermion=fermion)
    configs = np.asarray(
        list(product(*(range(len(code_map)) for code_map in sampler.physical_code_maps))),
        dtype=np.int64,
    )
    probabilities = sampler.probabilities(configs)

    site_inds = [state.site_ind(qubit) for qubit in range(4)]
    statevector = np.asarray(
        state.contract(all).transpose(*site_inds).data.to_dense()
    ).reshape(-1)
    statevector = statevector / np.linalg.norm(statevector)
    expected = np.abs(statevector) ** 2

    np.testing.assert_allclose(probabilities, expected, atol=1e-11)
    np.testing.assert_allclose(
        np.abs(sampler.amplitudes(configs)) ** 2,
        probabilities,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    ("symmetry", "occupations", "expected"),
    (
        (
            "Z2",
            (1, 1, 0, 2),
            ((0, 0), (1, 1), (1, 0), (0, 1)),
        ),
        (
            "U1",
            (1, 1, 0, 2),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        ),
        (
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        ),
        (
            "Z2Z2",
            ((1, 0), (0, 1), (1, 0), (0, 1)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        ),
    ),
)
def test_tree_sampler_symmray_fermion_code_maps(monkeypatch, symmetry, occupations, expected):
    """Fermionic code decoding follows each Symmray symmetry convention."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry=symmetry)
    state = pepsy.ps_to_ttn(4, fermion=fermion, occupations=occupations)

    def fail_dense(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("native TreeSampler must not densify Symmray data")

    monkeypatch.setattr(type(state.node_tensor(0).data), "to_dense", fail_dense)
    sampler = TreeSampler(state, backend="symmray", fermion=fermion, seed=0)
    batch = sampler.sample_batch(4, seed=5)

    assert sampler.resolved_backend == "symmray"
    assert sampler._configuration_encoding.code_to_occupations == (expected,) * 4
    assert batch.occupations().shape == (4, 4, 2)
    assert all(
        tuple(occupation) in expected
        for occupation in batch.occupations().reshape(-1, 2)
    )
