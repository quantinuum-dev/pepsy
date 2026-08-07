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
    }


def test_peps_sampler_rejects_incomplete_boundary_configuration():
    """Compressed modes require a conditioned ket bond cap."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=61)

    with pytest.raises(ValueError, match="sample_chi is required"):
        pepsy.PepsSampler(peps, boundary_engine="quimb-mps")

    with pytest.raises(ValueError, match="does not use sample_chi"):
        pepsy.PepsSampler(peps, sample_chi=2)


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
        assert sampler.batch_stats["suffix_cache_builds"] >= 2
        assert sampler.batch_stats["site_prefix_updates"] >= 4
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
