"""Focused tests for sampler-to-Torch-VMC proposal adaptation."""

from types import SimpleNamespace

import pytest


class _FermionBatch:
    one_d_to_two_d = {0: (1, 0), 1: (0, 0)}

    def __init__(self, occupations, probs):
        self._occupations = occupations
        self.probs = probs

    def occupations(self, *, to_numpy=False):
        del to_numpy
        return self._occupations


class _TreeBatch:
    nqubits = 2

    def __init__(self):
        self.configs = [[0, 1]]
        self.probs = [1.0]


def test_bp_sweeps_accumulate_counters_and_reject_local_move_tracking():
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import TorchBPMetropolisSampler

    class Proposal:
        def __init__(self):
            self.calls = 0

        def sample(self, *, samples, progbar=False):
            assert samples == 2
            self.calls += 1
            return SimpleNamespace(
                configs=torch.tensor([[0, 1], [1, 0]]),
                omegas=(
                    torch.full((2,), 0.25, dtype=torch.float64),
                    torch.zeros(2, dtype=torch.float64),
                ),
            )

    def make_sampler():
        return TorchBPMetropolisSampler(
            lambda configs: torch.ones(len(configs), dtype=torch.float64),
            [(0, 1)],
            Proposal(),
            configs=torch.zeros((2, 2), dtype=torch.long),
            initial_log_q=torch.full((2,), 0.25, dtype=torch.float64).log(),
            valid_config_fn=lambda configs: configs[:, 0] == 0,
            seed=17,
        )

    sampler = make_sampler()
    reference = make_sampler()
    expected = [reference.sample_sweep() for _ in range(3)]
    result = sampler.sample_sweep(n_sweeps=3)

    assert result.n_proposed == sum(step.n_proposed for step in expected) == 6
    # Each sweep rejects the second proposal through the validity mask.
    assert result.n_accepted == sum(step.n_accepted for step in expected) == 3
    assert result.acceptance_rate == pytest.approx(0.5)
    assert torch.equal(result.configs, reference.configs)
    assert torch.equal(result.amplitudes, reference.amplitudes)
    assert torch.equal(sampler.generator.get_state(), reference.generator.get_state())
    assert result.proposal_stats is None
    assert torch.equal(
        sampler.log_proposal_probabilities, reference.log_proposal_probabilities
    )
    warmed = make_sampler().burn_in(3)
    assert (warmed.n_proposed, warmed.n_accepted) == (6, 3)

    configs_before = sampler.configs.clone()
    rng_before = sampler.generator.get_state().clone()
    for advance in (
        sampler.sample_sweep,
        lambda **options: sampler.burn_in(1, **options),
        lambda **options: sampler.sample(n_samples=2, **options),
        lambda **options: sampler.warmup_proposal_mix(n_sweeps=1),
    ):
        with pytest.raises(ValueError, match="local-move statistics"):
            advance(track_proposal_stats=True)
    assert sampler.proposal_sampler.calls == 3
    assert torch.equal(sampler.configs, configs_before)
    assert torch.equal(sampler.generator.get_state(), rng_before)


def _driver(torch, *, zero_first=False):
    from pepsy.vmc import FermionSiteEncoding, TorchVMCDriver

    class Amplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, configs):
            values = torch.ones(configs.shape[0], dtype=torch.float64)
            if zero_first:
                values = torch.where(
                    configs[:, 0] == 0,
                    torch.zeros_like(values),
                    values,
                )
            return self.scale * values

    return TorchVMCDriver(
        Amplitude(),
        [],
        torch.tensor([[1, 2]], dtype=torch.long),
        terms={(0, 0): torch.eye(4, dtype=torch.float64)},
        site_order=((0, 0), (1, 0)),
        encoding=FermionSiteEncoding.vmc_torch(),
        proposal="spinful",
    )


def test_mps_batch_bridge_reorders_occupations_and_supports_observables():
    torch = pytest.importorskip("torch")
    driver = _driver(torch)
    batch = _FermionBatch(
        torch.tensor(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ],
            dtype=torch.long,
        ),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )

    samples = driver.sample_from_proposal(batch)
    assert samples.configs.tolist() == [[1, 2], [2, 1]]
    assert samples.proposal_log_probs.shape == (2,)

    result = driver.measure_samples(
        samples,
        observables={
            "energy": None,
            "eta": {(1, 0): torch.eye(4, dtype=torch.float64)},
        },
    )

    assert set(result) == {"energy", "eta"}
    assert result["energy"].configs.tolist() == [[[1, 2], [2, 1]]]
    assert all(torch.isfinite(value.energy_mean) for value in result.values())
    assert result["energy"].effective_sample_size == pytest.approx(2.0)


def test_importance_samples_refresh_target_amplitudes_after_a_peps_update():
    torch = pytest.importorskip("torch")
    driver = _driver(torch)
    batch = _FermionBatch(
        torch.tensor(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ],
            dtype=torch.long,
        ),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )

    samples = driver.sample_from_proposal(batch)
    assert torch.all(samples.amplitudes == 1.0)
    with torch.no_grad():
        driver.model.scale.add_(1.0)

    result = driver.measure_samples(samples)
    assert torch.all(result.amplitudes == 2.0)
    assert result.effective_sample_size == pytest.approx(2.0)


def test_mps_bridge_drops_zero_amplitude_nodes_before_local_energy():
    torch = pytest.importorskip("torch")
    driver = _driver(torch, zero_first=True)
    batch = _FermionBatch(
        torch.tensor(
                [
                    [[1, 0], [0, 0]],
                    [[0, 1], [1, 0]],
            ],
            dtype=torch.long,
        ),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )

    result = driver.measure_from_proposal(batch)

    assert result.n_samples == 1
    assert torch.isfinite(result.energy_mean)
    assert torch.isfinite(result.local_energies).all()


def test_tree_bridge_requires_an_explicit_occupation_map_for_spinful_states():
    torch = pytest.importorskip("torch")
    driver = _driver(torch)
    driver.metadata = SimpleNamespace(spinful=True)

    with pytest.raises(ValueError, match="occupation_map"):
        driver.measure_from_proposal(_TreeBatch())


@pytest.mark.integration
def test_real_u1u1_mps_sampler_feeds_fermionic_peps_vmc():
    """A native U1U1 MPS proposal should feed a small PEPS VMC estimate."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")

    from pepsy.sampling import MpsSampler
    from pepsy.tensors import Fermion, ps_to_mps
    from pepsy.vmc import TorchFermionVMC
    from pepsy.vmc.netket import fermionic_peps_rand

    fermion = Fermion(spinful=True, symmetry="U1U1")
    target = fermionic_peps_rand(
        "U1U1",
        2,
        3,
        2,
        n_fermions_per_spin=(3, 3),
        dtype="complex128",
        seed=4,
    )

    # The proposal is ordered column-major while the PEPS metadata is
    # row-major. The occupations alternate along the proposal chain so the
    # reordered target configuration is three ups followed by three downs.
    one_d_to_two_d = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (1, 1),
        4: (0, 2),
        5: (1, 2),
    }
    proposal_mps = ps_to_mps(
        6,
        fermion=fermion,
        occupations=[(1, 0), (0, 1)] * 3,
        dtype="complex128",
    )
    proposal = MpsSampler(
        proposal_mps,
        one_d_to_two_d,
        backend="symmray",
        fermion=fermion,
    )

    batch = proposal.sample_batch(4, seed=7, to_numpy=True)
    occupations = batch.occupations(to_numpy=True)
    assert occupations.shape == (4, 6, 2)
    assert occupations[:, :, 0].sum(axis=1).tolist() == [3] * 4
    assert occupations[:, :, 1].sum(axis=1).tolist() == [3] * 4

    sites = tuple((x, y) for x in range(target.Lx) for y in range(target.Ly))
    edges = tuple(
        ((x, y), (x + 1, y))
        for x in range(target.Lx - 1)
        for y in range(target.Ly)
    ) + tuple(
        ((x, y), (x, y + 1))
        for x in range(target.Lx)
        for y in range(target.Ly - 1)
    )
    terms = {
        edge: -0.2 * fermion.hopping_operator()
        for edge in edges
    }
    terms |= {
        site: fermion.onsite_term(site, U=1.0, mu=0.1)
        for site in sites
    }
    vmc = TorchFermionVMC(
        target,
        fermion=fermion,
        terms=terms,
        contraction="exact",
        n_walkers=1,
        seed=5,
        init_max_states=4096,
    )
    samples = vmc.sample(
        proposal=proposal,
        n_samples=4,
        seed=8,
        fermion=fermion,
        one_d_to_two_d=one_d_to_two_d,
    )
    result = vmc.measure(samples)["energy"]

    assert result.configs.shape == (1, 4, 6)
    assert result.configs[0].tolist() == [[2, 2, 2, 1, 1, 1]] * 4
    assert torch.isfinite(result.energy_mean).all()
    assert torch.isfinite(result.effective_sample_size)
    assert result.effective_sample_size.item() == pytest.approx(4.0)
