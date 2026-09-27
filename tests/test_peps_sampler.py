"""Tests for the serial direct PEPS sampler."""

from itertools import product

import numpy as np
import quimb.tensor as qtn
import pytest

import pepsy


def _scalar(value):
    data = getattr(value, "data", value)
    return np.asarray(data).reshape(()).item()


def _direct_amplitude(peps, site_order, config):
    projected = peps.copy()
    projected.isel_(
        {
            peps.site_ind(*site): int(value)
            for site, value in zip(site_order, config)
        }
    )
    return _scalar(projected.contract(all, optimize="auto-hq"))


def test_peps_sampler_conditionals_match_exact_born_probabilities():
    """The serial conditional product must equal the exact PEPS distribution."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=3,
        seed=41,
        dtype="complex128",
    )
    sampler = pepsy.PepsSampler(peps)
    norm = _scalar(sampler._norm.contract(all, optimize="auto-hq"))

    proposal = []
    born = []
    for config in product(range(3), repeat=4):
        amplitude = _direct_amplitude(peps, sampler.site_order, config)
        proposal.append(sampler.probability(config))
        born.append(abs(amplitude) ** 2 / norm)

    np.testing.assert_allclose(sum(proposal), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(proposal, born, atol=1.0e-12)


def test_peps_sampler_seed_and_source_network_are_stable():
    """Sampling is reproducible and does not mutate the input PEPS."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        seed=43,
        dtype="complex128",
    )
    original_tags = set(peps.tags)
    sampler = pepsy.PepsSampler(peps)

    first = sampler.sample(samples=3, seed=17)
    second = sampler.sample(samples=3, seed=17)

    assert first.configs == second.configs
    np.testing.assert_allclose(first.omegas[0], second.omegas[0])
    assert first.omegas[1] == second.omegas[1]
    np.testing.assert_allclose(first.ps[0], second.ps[0])
    assert first.ps[1] == second.ps[1]
    assert set(peps.tags) == original_tags


def test_peps_sampler_refresh_rebuilds_private_networks():
    """Refresh returns the sampler and keeps direct sampling available."""
    peps = qtn.PEPS.rand(Lx=1, Ly=2, bond_dim=2, seed=47)
    sampler = pepsy.PepsSampler(peps)
    old_norm = sampler._norm

    assert sampler.refresh() is sampler
    assert sampler._norm is not old_norm
    result = sampler.sample(seed=3)
    assert len(result.configs) == 1
    assert len(result.configs[0]) == 2


def test_peps_sampler_quimb_boundary_mode_uses_sample_chi_without_future():
    """The Quimb boundary path compresses the conditioned ket boundary."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        seed=53,
        dtype="complex128",
    )
    sampler = pepsy.PepsSampler(
        peps,
        sample_chi=2,
        marginal_chi=0,
        boundary_engine="quimb-mps",
        ket_compression="quimb",
    )

    probabilities = [
        sampler.probability(config)
        for config in product(range(2), repeat=4)
    ]
    np.testing.assert_allclose(sum(probabilities), 1.0, atol=1.0e-12)
    assert sampler._future_environments == {}
    sampler.sample(seed=7)
    assert sampler._last_boundary_mps.max_bond() <= 2


def test_peps_sampler_dmrg_future_and_fit_boundary_modes():
    """DMRG/FIT prepares the future boundary and compresses the ket boundary."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        seed=59,
        dtype="complex128",
    )
    sampler = pepsy.PepsSampler(
        peps,
        sample_chi=2,
        marginal_chi=2,
        boundary_engine="dmrg",
        ket_compression="fit",
        fit_n_iter=1,
    )

    result = sampler.sample(seed=11)
    assert len(result.configs) == 1
    assert set(sampler._future_environments) == {0}
    assert sampler._future_environments[0] is sampler._future_boundary.mps_b["Y0_r"]
    assert sampler._last_boundary_mps.max_bond() <= 2


@pytest.mark.parametrize("engine", ["quimb-mps", "dmrg"])
def test_peps_sampler_row_cache_matches_full_center_reference(engine):
    """The cached row transfers preserve the serial boundary proposal."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=63,
        dtype="complex128",
    )
    sampler = pepsy.PepsSampler(
        peps,
        sample_chi=2,
        marginal_chi=2,
        boundary_engine=engine,
        row_cache_max_bytes=64 * 2**20,
    )

    for config in product(range(2), repeat=4):
        cached = sampler.probability(config)
        _, reference = sampler._boundary_sample_or_probability_reference(
            config=config
        )
        reference = sampler._scaled_to_float(reference)
        np.testing.assert_allclose(cached, reference, rtol=1.0e-11, atol=1.0e-14)

    assert sampler.row_cache_stats == {
        "rows": 2,
        "suffix_cache_builds": 2,
        "site_prefix_updates": 4,
        "mode": "transfer",
        "estimated_cache_bytes": sampler._estimate_row_cache_bytes(),
        "cache_budget_bytes": 64 * 2**20,
        "cache_decision": "within-budget",
    }


def test_peps_sampler_rejects_incomplete_boundary_configuration():
    """Compressed modes require a conditioned ket bond cap."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=61)

    with pytest.raises(ValueError, match="sample_chi is required"):
        pepsy.PepsSampler(peps, boundary_engine="quimb-mps")

    with pytest.raises(ValueError, match="does not use sample_chi"):
        pepsy.PepsSampler(peps, sample_chi=2, boundary_engine="exact")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "sample_chi": 2,
            "marginal_chi": 2,
            "boundary_engine": "quimb-mps",
        },
    ],
)
def test_peps_sampler_prefix_batch_and_rho_diagnostics(kwargs):
    """Prefix grouping preserves proposals and reports local rho quality."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=67,
        dtype="complex128",
    )
    sampler = pepsy.PepsSampler(peps, **kwargs)
    first = sampler.sample_batch(samples=8, seed=13)
    second = sampler.sample_batch(samples=8, seed=13)

    assert first.configs == second.configs
    assert sampler.batch_stats["samples"] == 8
    assert sampler.batch_stats["max_prefix_groups"] >= 1
    if kwargs:
        assert sampler.batch_stats["suffix_cache_builds"] == 0
        assert sampler.batch_stats["site_prefix_updates"] == 0
        assert sampler.row_cache_stats["mode"] == "reference-prefix"
    assert set(sampler.rho_diagnostics) == set(sampler.site_order)
    for diagnostic in sampler.rho_diagnostics.values():
        assert np.isfinite(diagnostic["trace"])
        assert diagnostic["hermiticity_defect"] < 1.0e-12
        assert diagnostic["evaluation_count"] >= 1
        assert diagnostic["max_hermiticity_defect"] < 1.0e-12

    expected = [
        sampler.probability(config)
        for config in first.configs
    ]
    actual = [
        mantissa * 10.0**exponent
        for mantissa, exponent in zip(*first.omegas)
    ]
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


@pytest.fixture(params=["numpy", "torch", "jax"])
def array_backend(request):
    """Keep optional backend checks on CPU, including on GPU workstations."""
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        yield "torch", lambda x: torch.as_tensor(x, device="cpu")
    elif request.param == "jax":
        jax = pytest.importorskip("jax")
        with jax.default_device(jax.devices("cpu")[0]):
            yield "jax", jax.numpy.asarray
    else:
        yield "numpy", np.asarray


_BACKEND_PROPOSALS = [
    pytest.param({}, id="exact"),
    pytest.param(
        dict(sample_chi=4, marginal_chi=0, boundary_engine="quimb-mps"),
        id="identity-future",
    ),
    pytest.param(
        dict(sample_chi=4, marginal_chi=8, boundary_engine="quimb-mps"),
        id="quimb-future",
    ),
    pytest.param(
        dict(sample_chi=4, marginal_chi=8, boundary_engine="dmrg",
             ket_compression="fit", fit_n_iter=1),
        id="dmrg-fit",
    ),
]


@pytest.mark.parametrize("kwargs", _BACKEND_PROPOSALS)
def test_peps_sampler_infers_array_backend(array_backend, kwargs, monkeypatch):
    """Native draws preserve proposals, amplitudes, placement, and the input."""
    import autoray as ar
    from pepsy.backends import infer_backend_signature

    backend, convert = array_backend
    source = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex64", seed=83)
    reference = pepsy.PepsSampler(source, contraction_opt="greedy", **kwargs)
    dense = source.to_dense(
        [source.site_ind(*site) for site in reference.site_order]
    ).reshape(-1)
    state = source.copy()
    state.apply_to_arrays(convert)
    before = [ar.to_numpy(t.data).copy() for t in state.tensors]
    sampler = pepsy.PepsSampler(state, contraction_opt="greedy", **kwargs)
    assert sampler.backend == backend
    assert callable(sampler.to_backend)
    signature = infer_backend_signature(state.tensors[0].data)

    # Matrix/probability transfers were the original NumPy-only path. Only
    # integer choices may cross to Python during sample/sample_batch.
    to_numpy = ar.to_numpy

    def checked_to_numpy(value):
        assert "int" in ar.get_dtype_name(value)
        return to_numpy(value)

    with monkeypatch.context() as context:
        context.setattr(ar, "to_numpy", checked_to_numpy)
        first = sampler.sample_batch(4, seed=11)
        second = sampler.sample_batch(4, seed=11)
        serial = sampler.sample(2, seed=13)
        serial_again = sampler.sample(2, seed=13)
    assert first.configs == second.configs
    assert serial.configs == serial_again.configs
    for result in (first, serial):
        proposal = np.asarray(result.omegas[0]) * 10.0 ** np.asarray(result.omegas[1])
        expected = [reference.probability(config) for config in result.configs]
        np.testing.assert_allclose(proposal, expected, rtol=3e-5, atol=1e-7)
        amplitudes = np.asarray(result.ps[0]) * 10.0 ** np.asarray(result.ps[1])
        indices = [int("".join(map(str, config)), 2) for config in result.configs]
        np.testing.assert_allclose(amplitudes, dense[indices], rtol=3e-5, atol=3e-6)
    for tensor, original in zip(state.tensors, before):
        np.testing.assert_array_equal(ar.to_numpy(tensor.data), original)
    networks = [sampler._ket, sampler._norm, *sampler._future_environments.values()]
    if sampler._last_boundary_mps is not None:
        networks.append(sampler._last_boundary_mps)
    for network in networks:
        assert all(infer_backend_signature(t.data) == signature for t in network)
    rho, _ = sampler._local_rho(sampler._norm, sampler.site_order[0])
    assert infer_backend_signature(rho) == signature
    probabilities = sampler._conditional_probabilities(rho, site=(0, 0))
    assert ar.infer_backend(probabilities) == backend
    assert ar.get_dtype_name(probabilities) == "float32"
    assert all(np.isfinite(d["trace"]) for d in sampler.rho_diagnostics.values())


def test_peps_sampler_explicit_converter_and_refresh(array_backend):
    """An override converts only private tensors and persists across refresh."""
    import autoray as ar

    backend, convert = array_backend
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex64", seed=87)
    before = [t.data.copy() for t in state]
    sampler = pepsy.PepsSampler(state, to_backend=convert, contraction_opt="greedy")
    assert sampler.to_backend is convert
    assert sampler.backend == backend
    assert sampler.refresh() is sampler
    assert sampler.backend == backend
    sampler.sample(seed=2)
    for tensor, original in zip(state, before):
        assert ar.infer_backend(tensor.data) == "numpy"
        np.testing.assert_array_equal(tensor.data, original)


def test_peps_sampler_converter_cannot_mutate_source():
    """Protect source arrays even when the supplied converter acts in place."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=89)
    before = [t.data.copy() for t in state]

    def convert(data):
        data *= 2
        return data

    sampler = pepsy.PepsSampler(state, to_backend=convert)
    for tensor, original in zip(state, before):
        np.testing.assert_array_equal(tensor.data, original)
    for tensor, original in zip(sampler._ket, before):
        np.testing.assert_array_equal(tensor.data, original * 2)
    with pytest.raises(TypeError, match="to_backend must be a callable"):
        pepsy.PepsSampler(state, to_backend="torch")


def test_peps_sampler_converter_preserves_torch_gradient():
    """Private conversion must accept non-leaf tensors without detaching them."""
    torch = pytest.importorskip("torch")
    state = qtn.PEPS.rand(1, 2, bond_dim=2, dtype="complex64", seed=91)
    leaves = [torch.tensor(t.data, requires_grad=True) for t in state]
    for tensor, leaf in zip(state, leaves):
        tensor.modify(data=leaf * 2)
    sampler = pepsy.PepsSampler(state, to_backend=lambda x: x)
    amplitude = sampler._projected_amplitude([0, 0])
    amplitude.real.backward()
    assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in leaves)


def test_peps_sampler_refresh_reinfers_changed_backend():
    torch = pytest.importorskip("torch")
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex64", seed=93)
    sampler = pepsy.PepsSampler(state)
    state.tensors[0].modify(data=torch.as_tensor(state.tensors[0].data))
    with pytest.raises(ValueError, match="supply to_backend"):
        sampler.refresh()
    state.apply_to_arrays(torch.as_tensor)
    sampler.refresh()
    assert sampler.backend == "torch"
    assert all(isinstance(t.data, torch.Tensor) for t in sampler._ket)


def test_peps_sampler_float32_rho_validation(array_backend):
    """Clip dtype-scale roundoff only; reject invalid rhos and allow tiny norms."""
    import autoray as ar

    _, convert = array_backend
    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(2, 2, bond_dim=1, dtype="complex64", seed=97),
        to_backend=convert,
    )
    eps = np.finfo(np.float32).eps
    rho = convert(np.diag(np.array([1, -eps], dtype="complex64")))
    p = sampler._conditional_probabilities(rho, site=(0, 0))
    np.testing.assert_allclose(ar.to_numpy(p), [1, 0])
    assert sampler.rho_diagnostics[(0, 0)]["clipped_negative_mass"] == eps
    tiny = convert(np.diag(np.array([1e-20, 3e-20], dtype="complex64")))
    np.testing.assert_allclose(
        ar.to_numpy(sampler._conditional_probabilities(tiny, site=(0, 0))),
        [0.25, 0.75], rtol=1e-6,
    )
    for data, message in [
        ([[1, 0], [0, -0.01]], "substantially negative"),
        ([[0, 0], [0, 0]], "invalid trace"),
        ([[1, np.nan], [0, 1]], "non-finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            sampler._conditional_probabilities(
                convert(np.asarray(data, dtype="complex64")), site=(0, 0)
            )


def test_peps_sampler_reference_batch_and_duplicate_amplitudes(array_backend, monkeypatch):
    """A repeated configuration needs one amplitude contraction per batch."""
    _, convert = array_backend
    state = qtn.PEPS.product_state(
        [[np.array([1.0, 0.0], dtype="complex64")] * 2] * 2
    )
    sampler = pepsy.PepsSampler(
        state, to_backend=convert, sample_chi=2, marginal_chi=0,
        boundary_engine="quimb-mps", contraction_opt="greedy",
    )
    calls = []
    original = sampler._projected_amplitude

    def amplitude(config):
        calls.append(config)
        return original(config)

    monkeypatch.setattr(sampler, "_projected_amplitude", amplitude)
    result = sampler.sample_batch(40, seed=2)
    assert result.configs == [[0, 0, 0, 0]] * 40
    assert len(calls) == 1
    assert sampler.row_cache_stats["mode"] == "reference-prefix"
    np.testing.assert_allclose(result.omegas[0], 1)
    np.testing.assert_allclose(result.ps[0], 1)
    assert sampler.probability([1, 0, 0, 0]) == 0.0


def test_peps_sampler_jax_nondefault_device(monkeypatch):
    """Run with two CPU devices to catch default-device identity/RNG creation."""
    jax = pytest.importorskip("jax")
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires XLA_FLAGS=--xla_force_host_platform_device_count=2")
    device = devices[1]
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex64", seed=53)
    state.apply_to_arrays(lambda x: jax.device_put(x, device))
    sampler = pepsy.PepsSampler(
        state, sample_chi=2, marginal_chi=0,
        boundary_engine="quimb-mps", contraction_opt="greedy",
    )
    with jax.default_device(devices[0]):
        cap = sampler._identity_future(0)
        assert all(t.data.devices() == {device} for t in cap)
        make_rng = sampler._make_rng
        draws = []

        def checked_rng(seed):
            rng = make_rng(seed)
            random = rng.random

            def checked_random(*args, **kwargs):
                result = random(*args, **kwargs)
                assert result.devices() == {device}
                draws.append(result)
                return result

            rng.random = checked_random
            return rng

        monkeypatch.setattr(sampler, "_make_rng", checked_rng)
        first = sampler.sample_batch(4, seed=7)
        assert first.configs == sampler.sample_batch(4, seed=7).configs
        assert len(draws) == 8
        assert all(t.data.devices() == {device} for t in sampler._last_boundary_mps)


@pytest.mark.parametrize("budget", [-1, 1.5, True, None])
def test_peps_sampler_rejects_invalid_cache_budget(budget):
    with pytest.raises(ValueError, match="row_cache_max_bytes"):
        pepsy.PepsSampler(qtn.PEPS.rand(1, 1, bond_dim=1), row_cache_max_bytes=budget)


def test_peps_sampler_large_cache_uses_reference_before_allocation(monkeypatch):
    """The D=4 case previously allocated a 1 GiB column on a 3x3 PEPS."""
    state = qtn.PEPS.rand(3, 3, bond_dim=4, dtype="complex128", seed=101)
    options = dict(sample_chi=16, marginal_chi=32, boundary_engine="quimb-mps",
                   contraction_opt="greedy")
    sampler = pepsy.PepsSampler(state, row_cache_max_bytes=64 * 2**20, **options)
    reference = pepsy.PepsSampler(state, row_cache_max_bytes=0, **options)

    def forbidden_cache(*args, **kwargs):
        pytest.fail("Oversized dense row cache must not be constructed")

    monkeypatch.setattr(sampler, "_build_row_transfer_cache", forbidden_cache)
    assert sampler._estimate_row_cache_bytes() >= 2**30
    for method, shots in [("sample", 1), ("sample_batch", 8)]:
        result = getattr(sampler, method)(shots, seed=812)
        expected = getattr(reference, method)(shots, seed=812)
        assert result.configs == expected.configs
        np.testing.assert_allclose(result.omegas, expected.omegas, rtol=1e-12)
        np.testing.assert_allclose(result.ps, expected.ps, rtol=1e-12)
        assert sampler.row_cache_stats["cache_decision"] == "memory-budget"
        assert sampler.row_cache_stats["suffix_cache_builds"] == 0
    for config in result.configs[:2]:
        np.testing.assert_allclose(sampler.probability(config),
                                   reference.probability(config), rtol=1e-12)
    assert reference.row_cache_stats["cache_decision"] == "disabled"
    sampler.refresh()
    assert sampler._row_cache_estimate_per_group is None
    assert not sampler._row_bond_cache


def test_peps_sampler_local_rho_preserves_network_and_scale():
    sampler = pepsy.PepsSampler(qtn.PEPS.rand(2, 2, bond_dim=2, seed=17))
    working = sampler._norm.copy()
    working.exponent = 2.5
    snapshots = [(t.inds, tuple(t.tags), t.data.copy()) for t in working]
    site = sampler.site_order[0]
    ket_ind = sampler._site_inds[site]
    bra_ind = ket_ind + "__test_bra"
    oracle = working.copy()
    oracle.select([sampler._site_tags[site], "BRA"], "all").reindex_(
        {ket_ind: bra_ind}
    )
    expected = oracle.contract(all, output_inds=(ket_ind, bra_ind)).data
    rho, _ = sampler._local_rho(working, site)
    np.testing.assert_allclose(rho, expected, rtol=1e-12)
    for tensor, (inds, tags, data) in zip(working, snapshots):
        assert tensor.inds == inds and tuple(tensor.tags) == tags
        np.testing.assert_array_equal(tensor.data, data)
    assert working.exponent == 2.5


def test_peps_sampler_stable_large_rho_diagnostics(array_backend):
    import autoray as ar

    _, convert = array_backend
    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(1, 1, bond_dim=1, dtype="complex64", seed=19),
        to_backend=convert,
    )
    base = np.array([[1, 0.2 + 0.3j], [0.2 - 0.29j, 3]], dtype="complex64")
    for factor in (1e24, 1, 1e-24):
        data = base * np.float32(factor)
        rho64 = data.astype("complex128")
        expected = np.linalg.norm(rho64 - rho64.conj().T) / max(
            np.linalg.norm(rho64), 1
        )
        probabilities = sampler._conditional_probabilities(convert(data), site=(0, 0))
        np.testing.assert_allclose(ar.to_numpy(probabilities), [0.25, 0.75], rtol=1e-6)
        actual = sampler.rho_diagnostics[(0, 0)]["hermiticity_defect"]
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0)
    for bad, message in [
        ([[1, np.nan], [0, 1]], "non-finite"),
        ([[1, 0], [0, -0.01]], "substantially negative"),
        ([[0, 0], [0, 0]], "invalid trace"),
    ]:
        rhos = np.stack([base, np.asarray(bad, dtype="complex64"), base])
        with pytest.raises(ValueError, match=message):
            sampler._conditional_probabilities_batch(convert(rhos), site=(0, 0))


def test_peps_sampler_grouped_draws_distribution_and_zeros(array_backend):
    import autoray as ar

    _, convert = array_backend
    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(1, 1, bond_dim=1, phys_dim=3, dtype="complex64", seed=23),
        to_backend=convert,
    )
    probabilities = [np.array([0, 0.3, 0.7]), np.array([0.6, 0.4, 0])]
    groups = [dict(indices=np.arange(n), log10=sampler._zero_log_probability)
              for n in (4096, 8192)]
    rhos = [convert(np.diag(p).astype("complex64")) for p in probabilities]
    draws, logs = sampler._draw_grouped_choices(
        sampler._make_rng(29), rhos, groups, site=(0, 0)
    )
    again, _ = sampler._draw_grouped_choices(
        sampler._make_rng(29), rhos, groups, site=(0, 0)
    )
    for i, (values, repeated, p) in enumerate(zip(draws, again, probabilities)):
        np.testing.assert_array_equal(values, repeated)
        assert np.all(p[values] > 0)
        np.testing.assert_allclose(np.bincount(values, minlength=3) / len(values),
                                   p, atol=0.025, rtol=0)
        np.testing.assert_allclose(10 ** ar.to_numpy(logs[i]), p, rtol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {},
    dict(sample_chi=2, boundary_engine="quimb-mps", row_cache_max_bytes=64 * 2**20),
    dict(sample_chi=2, boundary_engine="quimb-mps"),
])
def test_peps_sampler_grouped_readbacks_per_site(kwargs, monkeypatch):
    import autoray as ar

    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(2, 2, bond_dim=2, seed=31), contraction_opt="greedy", **kwargs
    )
    validations = []
    choices = []
    scalar = sampler._scalar
    to_numpy = ar.to_numpy

    def counted_scalar(value):
        if ar.get_dtype_name(value) == "bool":
            validations.append(value)
        return scalar(value)

    def counted_to_numpy(value):
        assert "int" in ar.get_dtype_name(value)
        choices.append(value)
        return to_numpy(value)

    monkeypatch.setattr(sampler, "_scalar", counted_scalar)
    monkeypatch.setattr(ar, "to_numpy", counted_to_numpy)
    sampler.sample_batch(16, seed=37)
    assert len(validations) == len(choices) == len(sampler.site_order)
    assert sampler.batch_stats["conditional_batches"] == len(sampler.site_order)
    assert sampler.batch_stats["max_prefix_groups"] > 1


@pytest.mark.parametrize("compression", [None, "quimb"])
def test_peps_sampler_cache_estimate_bounds_represented_bonds(compression):
    """Uncompressed bonds can exceed the physical Schmidt rank bound."""
    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(2, 4, bond_dim=2, dtype="complex128", seed=41),
        sample_chi=2, ket_compression=compression, boundary_engine="quimb-mps",
        contraction_opt="greedy",
    )
    estimate = sampler._estimate_row_cache_bytes()
    phi = None
    for y in range(sampler.Ly):
        cache = sampler._build_row_transfer_cache(y, phi)
        tensors = [*cache["local"], *cache["right"]]
        actual_bytes = sum(t.data.nbytes for t in tensors if t is not None)
        assert estimate >= actual_bytes
        phi = sampler._update_conditioned_boundary(y, [0, 0], phi)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1), (2, 3), (3, 2)])
@pytest.mark.parametrize("engine", ["quimb-mps", "dmrg"])
def test_peps_sampler_simple_sweep_dimensions_and_born(shape, engine, monkeypatch):
    """Edge shapes and rectangles reduce to Born sampling without truncation."""
    state = qtn.PEPS.rand(*shape, bond_dim=2, dtype="complex128", seed=122)
    sampler = pepsy.PepsSampler(
        state, sample_chi=8, marginal_chi=8, boundary_engine=engine,
        ket_compression="fit" if engine == "dmrg" else "quimb",
        fit_n_iter=1, contraction_opt="greedy",
    )
    future_snapshots = {
        y: [(t.inds, tuple(t.tags), t.data.copy()) for t in network]
        for y, network in sampler._future_environments.items()
    }
    if sampler._future_store is not None:
        assert {side for side, _ in sampler._future_store.envs} == {"ymax"}
    if sampler._future_boundary is not None:
        assert all(key.startswith("Y") and key.endswith("_r")
                   for key in sampler._future_boundary.mps_b.keys())
    dense = state.to_dense([state.site_ind(*site) for site in sampler.site_order]).ravel()
    born = abs(dense)**2 / np.vdot(dense, dense).real

    def forbidden_cache(*args, **kwargs):
        pytest.fail("The default simple sweep must not construct dense row caches")

    monkeypatch.setattr(sampler, "_build_row_transfer_cache", forbidden_cache)
    configs = list(product(range(2), repeat=len(sampler.site_order)))
    probabilities = [sampler.probability(config) for config in configs]
    np.testing.assert_allclose(probabilities, born, atol=3e-13, rtol=3e-12)
    batch = sampler.sample_batch(4, seed=11)
    for config, mantissa, exponent in zip(batch.configs, *batch.omegas):
        index = configs.index(tuple(config))
        np.testing.assert_allclose(mantissa * 10**exponent, born[index], rtol=3e-12)
    for y, snapshot in future_snapshots.items():
        for t, (inds, tags, data) in zip(sampler._future_environments[y], snapshot):
            assert t.inds == inds and tuple(t.tags) == tags
            np.testing.assert_array_equal(t.data, data)
    assert sampler.row_cache_stats["cache_decision"] == "disabled"
    assert sampler.row_cache_stats["estimated_cache_bytes"] is None


def test_peps_sampler_missing_future_is_not_replaced_by_identity():
    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(2, 3, bond_dim=2, seed=125), sample_chi=4, marginal_chi=8,
        boundary_engine="quimb-mps", contraction_opt="greedy",
    )
    del sampler._future_environments[0]
    with pytest.raises(RuntimeError, match="Missing future boundary"):
        sampler.sample_batch(2, seed=3)


def test_peps_sampler_absorbs_only_after_complete_row_and_respects_ket_chi(monkeypatch):
    sampler = pepsy.PepsSampler(
        qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=128),
        sample_chi=1, marginal_chi=8, boundary_engine="quimb-mps",
        contraction_opt="greedy",
    )
    compressed = []
    update = sampler._update_conditioned_boundary

    def checked_update(y, row_config, phi):
        assert len(row_config) == sampler.Lx
        new_phi = update(y, row_config, phi)
        if y < sampler.Ly - 1:
            assert new_phi.max_bond() <= 1
            assert set(new_phi.outer_inds()) == set(sampler._phi_inds)
            compressed.append(y)
        return new_phi

    monkeypatch.setattr(sampler, "_update_conditioned_boundary", checked_update)
    result = sampler.sample(seed=13)
    assert compressed == [0, 1]
    amplitude = result.ps[0][0] * 10**result.ps[1][0]
    np.testing.assert_allclose(amplitude,
                               _direct_amplitude(sampler.peps, sampler.site_order, result.configs[0]),
                               rtol=1e-12)


@pytest.mark.parametrize("engine", ["exact", "quimb-mps"])
def test_peps_sampler_log_probability_survives_float_underflow(engine):
    state = qtn.PEPS.product_state([[np.array([1.0, 1e-30])] * 8])
    kwargs = {} if engine == "exact" else dict(sample_chi=1, boundary_engine=engine)
    sampler = pepsy.PepsSampler(state, contraction_opt="greedy", **kwargs)
    expected = -8 * np.log1p(1e60)
    np.testing.assert_allclose(sampler.log_probability([1] * 8), expected, atol=2e-12)
    assert sampler.probability([1] * 8) == 0.0


def test_peps_sampler_log_probability_float32_backend(array_backend):
    _, convert = array_backend
    state = qtn.PEPS.product_state([[np.array([1.0, 1e-2], dtype="complex64")] * 12])
    sampler = pepsy.PepsSampler(state, to_backend=convert, contraction_opt="greedy")
    expected = -12 * np.log1p(1e4)
    np.testing.assert_allclose(sampler.log_probability([1] * 12), expected, rtol=2e-6)
    # Python float still represents this result; a float32 product would not.
    np.testing.assert_allclose(sampler.probability([1] * 12), np.exp(expected), rtol=3e-5, atol=0)


def test_peps_sampler_validates_all_values_even_after_zero_prefix():
    state = qtn.PEPS.product_state([[np.array([1.0, 0.0])] * 2])
    sampler = pepsy.PepsSampler(state)
    assert sampler.log_probability([1, 0]) == -np.inf
    assert sampler.probability([1, 0]) == 0
    for config in ([1, 2], [0, 0.5], [0, np.nan], [0], [0, 0, 0]):
        with pytest.raises(ValueError):
            sampler.probability(config)
    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample_batch(True)


@pytest.mark.parametrize("options", [
    dict(cutoff=float("nan")), dict(cutoff=float("inf")),
    dict(sample_chi=True), dict(marginal_chi=True), dict(fit_n_iter=True),
])
def test_peps_sampler_rejects_invalid_numerical_options(options):
    with pytest.raises(ValueError):
        pepsy.PepsSampler(qtn.PEPS.rand(1, 1, bond_dim=1), **options)


def test_peps_sampler_boundary_rejects_periodic_edges():
    state = qtn.PEPS.rand(3, 3, bond_dim=1, cyclic=True, seed=132)
    with pytest.raises(ValueError, match="open boundaries"):
        pepsy.PepsSampler(state, sample_chi=2, boundary_engine="quimb-mps")
    # Exact contraction does not use the open-boundary sweep assumptions.
    assert len(pepsy.PepsSampler(state).sample(seed=2).configs[0]) == 9


def test_peps_sampler_variable_physical_dimensions_and_integer_conversion():
    arrays = [[np.ones(2), np.ones(3)], [np.ones(1), np.ones(2)]]
    sampler = pepsy.PepsSampler(qtn.PEPS.product_state(arrays), sample_chi=1,
                                boundary_engine="quimb-mps", contraction_opt="greedy")
    batch = sampler.sample_batch(8, seed=3)
    for config in batch.configs:
        assert all(0 <= value < size for value, size in zip(config, [2, 1, 3, 2]))
        np.testing.assert_allclose(sampler.probability(config), 1/12, rtol=1e-12)
    state = qtn.PEPS.product_state([[np.array([1, 0])]])
    with pytest.raises(TypeError, match="floating or complex"):
        pepsy.PepsSampler(state)
    converted = pepsy.PepsSampler(state, to_backend=lambda a: a.astype(float))
    assert converted.sample(seed=1).configs == [[0]]


@pytest.mark.parametrize("engine", ["auto", "quimb-mps", "dmrg"])
def test_peps_sampler_cutoff_aliases_preserve_sampled_distribution(engine):
    """New cutoff names select the same proposal and original amplitudes."""
    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype="complex128", seed=149)
    options = dict(boundary_engine=engine, contraction_opt="greedy")
    canonical = pepsy.PepsSampler(state, chi=4, chi_prime=2, **options)
    legacy = pepsy.PepsSampler(state, marginal_chi=4, sample_chi=2, **options)
    assert canonical.chi == legacy.marginal_chi == 4
    assert canonical.chi_prime == legacy.sample_chi == 2
    first = canonical.sample_batch(samples=8, seed=17)
    second = legacy.sample_batch(samples=8, seed=17)
    assert first.configs == second.configs
    np.testing.assert_allclose(first.log_probabilities, second.log_probabilities)
    np.testing.assert_allclose(first.ps[0], second.ps[0])
    np.testing.assert_array_equal(first.ps[1], second.ps[1])
    expected_logs = [canonical.log_probability(config) for config in first.configs]
    np.testing.assert_allclose(first.log_probabilities, expected_logs, atol=1e-12)
    amplitudes = [_direct_amplitude(state, canonical.site_order, c) for c in first.configs]
    np.testing.assert_allclose(first.log_weights, 2 * np.log(np.abs(amplitudes)) - expected_logs)


@pytest.mark.parametrize("engine", ["auto", None])
def test_peps_sampler_automatic_mode_selection(engine):
    """Omitted caps remain exact; a ket cap opts into the boundary proposal."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=151)
    assert pepsy.PepsSampler(state, boundary_engine=engine).boundary_engine == "exact"
    assert pepsy.PepsSampler(state, chi=0, boundary_engine=engine).boundary_engine == "exact"
    sampler = pepsy.PepsSampler(state, chi_prime=2, boundary_engine=engine)
    assert sampler.boundary_engine == "dmrg"
    assert sampler.chi is None
    assert sampler._future_environments == {}
    # Matching duplicate spellings are accepted, including explicit None.
    matching = pepsy.PepsSampler(state, chi=None, marginal_chi=0,
                                chi_prime=np.int64(2), sample_chi=2)
    assert matching.chi == 0
    assert matching.chi_prime == 2
    with pytest.raises(ValueError, match="chi_prime or sample_chi is required"):
        pepsy.PepsSampler(state, chi=4, boundary_engine=engine)


@pytest.mark.parametrize("options, message", [
    (dict(chi=2, marginal_chi=4, chi_prime=2), "chi and marginal_chi must agree"),
    (dict(chi_prime=2, sample_chi=4), "chi_prime and sample_chi must agree"),
    (dict(chi=0, marginal_chi=4, chi_prime=2), "chi and marginal_chi must agree"),
    (dict(chi=True, marginal_chi=1), "chi must be"),
    (dict(chi=1, marginal_chi=True), "marginal_chi must be"),
    (dict(chi_prime=2, sample_chi=True), "sample_chi must be"),
    (dict(chi_prime=np.bool_(True)), "chi_prime must be"),
    (dict(chi_prime=0), "chi_prime must be"),
    (dict(chi=-1), "chi must be"),
    (dict(chi=2.5), "chi must be"),
    (dict(chi_prime=2.5), "chi_prime must be"),
    (dict(chi=2, boundary_engine="exact"), "does not use marginal_chi"),
    (dict(chi_prime=2, boundary_engine="exact"), "does not use sample_chi"),
])
def test_peps_sampler_validates_canonical_cutoffs_and_conflicts(options, message):
    state = qtn.PEPS.rand(1, 2, bond_dim=2, seed=157)
    with pytest.raises(ValueError, match=message):
        pepsy.PepsSampler(state, **options)


@pytest.mark.parametrize("engine", ["quimb-mps", "dmrg"])
def test_peps_sampler_uncompressed_ket_needs_no_cap(engine):
    """Explicitly uncompressed boundaries recover the exact small-state Born law."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=163)
    sampler = pepsy.PepsSampler(state, chi=16, ket_compression=None,
                               boundary_engine=engine, cutoff=0.0)
    reference = pepsy.PepsSampler(state)
    assert sampler.chi_prime is None
    configs = list(product(range(2), repeat=4))
    np.testing.assert_allclose([sampler.probability(c) for c in configs],
                               [reference.probability(c) for c in configs], atol=1e-12)
    batch = sampler.sample_batch(samples=4, seed=19)
    norm = _scalar(reference._norm.contract(all))
    np.testing.assert_allclose(batch.log_weights, np.log(norm.real), atol=1e-12)


def test_peps_sample_result_logs_preserve_extreme_scales_and_zeros():
    """Log views never materialize powers of ten, even for very rare samples."""
    result = pepsy.PEPSSampleResult(
        configs=[[0], [1]], omegas=([2.0, 5.0], [-1000, -2000]),
        ps=([3.0 + 4.0j, 0.0], [1000, 0]),
    )
    np.testing.assert_allclose(result.log_probabilities,
                               np.log([2.0, 5.0]) + np.array([-1000, -2000]) * np.log(10))
    np.testing.assert_allclose(result.log_abs_amplitudes[0], np.log(5.0) + 1000 * np.log(10))
    np.testing.assert_allclose(result.log_weights[0], np.log(12.5) + 3000 * np.log(10))
    assert np.isneginf(result.log_abs_amplitudes[1])
    assert np.isneginf(result.log_weights[1])
    empty = pepsy.PEPSSampleResult([], ([], []), ([], []))
    assert empty.log_probabilities.shape == empty.log_abs_amplitudes.shape == empty.log_weights.shape == (0,)


def test_peps_sample_result_logs_accept_backend_scalars(array_backend):
    """Shared BP/direct results can contain native scalar amplitudes/exponents."""
    _, convert = array_backend
    result = pepsy.PEPSSampleResult([[0]], ([5.0], [-3]),
                                   ([convert(3.0 + 4.0j)], [convert(-2.0)]))
    np.testing.assert_allclose(result.log_weights, [np.log(0.5)], atol=1e-6)


@pytest.mark.parametrize("dtype, expected_cutoff", [("complex64", 1e-6), ("complex128", 1e-12)])
@pytest.mark.parametrize("engine, compression", [
    ("exact", "quimb"), ("quimb-mps", "quimb"), ("dmrg", "fit"),
])
def test_peps_sampler_auto_cutoffs_match_born_by_precision(
    array_backend, dtype, expected_cutoff, engine, compression,
):
    """Automatic policies preserve small Born distributions on native backends."""
    from contextlib import nullcontext
    import autoray as ar
    from pepsy.backends import infer_backend_signature

    backend, convert = array_backend
    context = nullcontext()
    if backend == "jax":
        import jax
        enable_x64 = getattr(jax, "enable_x64", None)
        if enable_x64 is None:
            from jax.experimental import enable_x64
        context = enable_x64()
    with context:
        state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype=dtype, seed=173)
        # Contract an independent higher-precision reference for the same tensors.
        reference = state.copy()
        reference.apply_to_arrays(lambda a: a.astype("complex128"))
        order = [(x, y) for y in range(2) for x in range(2)]
        dense = reference.to_dense([state.site_ind(*site) for site in order]).ravel()
        born = np.abs(dense) ** 2
        born /= born.sum()
        state.apply_to_arrays(convert)
        originals = [ar.to_numpy(t.data).copy() for t in state]
        kwargs = {} if engine == "exact" else dict(chi=8, chi_prime=4)
        sampler = pepsy.PepsSampler(
            state, boundary_engine=engine, ket_compression=compression,
            cutoff="auto", cutoff_mode="auto", contraction_opt="greedy", **kwargs,
        )
        assert sampler.cutoff == expected_cutoff
        assert sampler.cutoff_mode == "rsum2"
        tolerance = 3e-5 if dtype == "complex64" else 1e-11
        configs = list(product(range(2), repeat=4))
        probabilities = np.array([sampler.probability(c) for c in configs])
        np.testing.assert_allclose(probabilities, born, rtol=tolerance, atol=tolerance / 100)
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=tolerance)
        batch = sampler.sample_batch(samples=8, seed=29)
        np.testing.assert_allclose(batch.log_probabilities,
                                   [sampler.log_probability(c) for c in batch.configs],
                                   rtol=tolerance, atol=tolerance)
        indices = [configs.index(tuple(c)) for c in batch.configs]
        amplitudes = np.asarray(batch.ps[0]) * 10.0 ** np.asarray(batch.ps[1])
        np.testing.assert_allclose(amplitudes, dense[indices], rtol=tolerance)
        signature = infer_backend_signature(state.tensors[0].data)
        networks = [sampler._ket, sampler._norm, *sampler._future_environments.values()]
        if sampler._last_boundary_mps is not None:
            networks.append(sampler._last_boundary_mps)
        for network in networks:
            assert all(infer_backend_signature(t.data) == signature for t in network)
        for tensor, original in zip(state, originals):
            np.testing.assert_array_equal(ar.to_numpy(tensor.data), original)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("compression", ["quimb", "fit"])
def test_peps_sampler_auto_discarded_weight_and_explicit_mode(dtype, compression):
    """A known small Schmidt weight distinguishes auto rsum2 from rel cutoff."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype=dtype, seed=179)
    small = 1e-4
    # First measured row is 00. Its outgoing ket is |00> + small * |11>.
    for x, y in product(range(2), repeat=2):
        tensor = state[x, y]
        horizontal = state.bond((x, y), (1 - x, y))
        vertical = state.bond((x, y), (x, 1 - y))
        physical = state.site_ind(x, y)
        data = np.zeros(tensor.shape, dtype=dtype)
        for value in range(2):
            indices = {horizontal: value if y == 0 else 0,
                       vertical: value, physical: 0 if y == 0 else value}
            data[tuple(indices[ix] for ix in tensor.inds)] = (
                small if (x, y, value) == (0, 0, 1) else 1.0
            )
        tensor.modify(data=data)
    options = dict(chi=4, chi_prime=2, boundary_engine="quimb-mps",
                   ket_compression=compression, contraction_opt="greedy")
    automatic = pepsy.PepsSampler(state, cutoff="auto", cutoff_mode="auto", **options)
    relative = pepsy.PepsSampler(state, cutoff="auto", cutoff_mode="rel", **options)
    untruncated = pepsy.PepsSampler(state, cutoff=0.0, cutoff_mode=None, **options)
    rare = [0, 0, 1, 1]
    expected = small**2 / (1 + small**2)
    np.testing.assert_allclose(relative.probability(rare), expected, rtol=1e-5)
    np.testing.assert_allclose(untruncated.probability(rare), expected, rtol=1e-5)
    if dtype == "complex64":
        assert automatic.log_probability(rare) == -np.inf
    else:
        np.testing.assert_allclose(automatic.probability(rare), expected, rtol=1e-10)


def test_peps_sampler_auto_cutoff_resolves_after_conversion_and_refresh():
    """Resolution follows the private working dtype, and explicit values stay fixed."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=181)
    converted = pepsy.PepsSampler(state, to_backend=lambda a: a.astype("complex64"))
    assert converted.cutoff == 1e-6
    automatic = pepsy.PepsSampler(state)
    explicit = pepsy.PepsSampler(state, cutoff=2e-9, cutoff_mode="abs")
    assert automatic.cutoff == 1e-12
    state.apply_to_arrays(lambda a: a.astype("complex64"))
    automatic.refresh()
    explicit.refresh()
    assert automatic.cutoff == 1e-6
    assert explicit.cutoff == 2e-9
    assert explicit.cutoff_mode == "abs"


@pytest.mark.parametrize("options, error", [
    (dict(cutoff="invalid"), ValueError),
    (dict(cutoff=True), TypeError),
    (dict(cutoff=None), TypeError),
    (dict(cutoff=-1.0), ValueError),
    (dict(cutoff_mode="invalid"), ValueError),
    (dict(cutoff_mode=2), TypeError),
])
def test_peps_sampler_validates_automatic_truncation_options(options, error):
    with pytest.raises(error, match="cutoff"):
        pepsy.PepsSampler(qtn.PEPS.rand(1, 1, bond_dim=1), **options)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_peps_sampler_auto_cutoff_reaches_future_environment(dtype):
    """A weighted GHZ state isolates truncation in the cached future boundary."""
    state = qtn.PEPS.rand(2, 3, bond_dim=2, dtype=dtype, seed=193)
    small = 0.01
    for x, y in product(range(2), range(3)):
        tensor = state[x, y]
        data = np.zeros(tensor.shape, dtype=dtype)
        data[(0,) * tensor.ndim] = 1.0
        data[(1,) * tensor.ndim] = small if (x, y) == (0, 2) else 1.0
        tensor.modify(data=data)
    # Disable ket compression to test the future policy independently.
    options = dict(chi=16, ket_compression=None, boundary_engine="quimb-mps",
                   cutoff="auto", contraction_opt="greedy")
    automatic = pepsy.PepsSampler(state, cutoff_mode="auto", **options)
    relative = pepsy.PepsSampler(state, cutoff_mode="rel", **options)
    rare = [1] * 6
    expected = small**2 / (1.0 + small**2)
    np.testing.assert_allclose(relative.probability(rare), expected, rtol=2e-6)
    if dtype == "complex64":
        assert automatic.log_probability(rare) == -np.inf
    else:
        np.testing.assert_allclose(automatic.probability(rare), expected, rtol=1e-12)
