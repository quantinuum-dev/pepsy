"""Tests for Pepsy's optional torch VMC kernels."""

import pytest
import quimb.tensor as qtn
import numpy as np
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from pepsy.tensors import Fermion, ps_to_peps  # noqa: E402
from pepsy.sampling import PepsBpSampler  # noqa: E402
from pepsy.vmc.torch import (  # noqa: E402
    FermionSiteEncoding,
    TorchPEPSAmplitude,
    TorchPEPSBoundaryAmplitude,
    TorchFermionVMC,
    TorchFermionVMCMetadata,
    TorchChainDiagnostics,
    TorchMCMCSamples,
    TorchMetropolisSampler,
    TorchBPMetropolisSampler,
    TorchVMCDriver,
    TorchVMCEnergyEstimate,
    TorchVMCImportanceEstimate,
    TorchVMCStepResult,
    TorchSquareLattice,
    apply_torch_sr_update,
    count_spinful_particles,
    heisenberg_connections,
    local_energy_from_connections,
    make_torch_peps_amplitude_model,
    metropolis_local_sampler,
    metropolis_exchange_sweep,
    propose_spin_exchange,
    propose_spinful_exchange_or_hopping,
    propose_spinful_u1_exchange_or_hopping,
    propose_spinful_z2_exchange_or_hopping,
    propose_spinful_z2z2_exchange_or_hopping,
    random_spin_configs,
    random_spinful_configs,
    solve_torch_sr,
    spinful_fermi_hubbard_connections,
    torch_log_derivative_matrix,
    transverse_ising_connections,
    torch_hamiltonian_connections,
    torch_chain_diagnostics,
)


class ProductAmplitude(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.tensor([2.0, 3.0], dtype=torch.float64)
        )

    def forward(self, configs):
        return self.weights[configs].prod(dim=1)


class CountingAmplitude:
    def __init__(self):
        self.calls = []

    def __call__(self, configs):
        self.calls.append(configs.detach().clone())
        return torch.ones(configs.shape[0], dtype=torch.float64, device=configs.device)


class LogOnlyAmplitude:
    """Amplitude whose raw values underflow but whose log values are usable."""

    def __call__(self, configs):
        return torch.zeros(configs.shape[0], dtype=torch.float64)

    def forward_log(self, configs):
        high = configs[:, 0] == 1
        log_abs = torch.where(
            high,
            torch.full((configs.shape[0],), -900.0, dtype=torch.float64),
            torch.full((configs.shape[0],), -1000.0, dtype=torch.float64),
        )
        phase = torch.ones(configs.shape[0], dtype=torch.complex128)
        return phase, log_abs


def test_fermion_site_encoding_supports_symmray_and_vmc_torch_orders():
    symm = FermionSiteEncoding.symmray()
    vmct = FermionSiteEncoding.vmc_torch()

    configs = torch.tensor([[symm.empty, symm.double, symm.up, symm.down]])
    n_up, n_down = symm.decode(configs)
    assert n_up.tolist() == [[0, 1, 1, 0]]
    assert n_down.tolist() == [[0, 1, 0, 1]]
    assert symm.encode(n_up, n_down).tolist() == configs.tolist()

    configs = torch.tensor([[vmct.empty, vmct.double, vmct.up, vmct.down]])
    n_up, n_down = vmct.decode(configs)
    assert n_up.tolist() == [[0, 1, 1, 0]]
    assert n_down.tolist() == [[0, 1, 0, 1]]
    assert vmct.encode(n_up, n_down).tolist() == configs.tolist()

    with pytest.raises(ValueError, match="Unknown fermion site code"):
        symm.decode(torch.tensor([[9]]))


def test_fermion_site_encoding_derives_u1u1_peps_charge_order():
    fermion = Fermion(spinful=True, symmetry="U1U1")
    encoding = FermionSiteEncoding.from_fermion(
        fermion,
        physical_charges=((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    assert encoding == FermionSiteEncoding.vmc_torch()
    assert encoding.encode(torch.tensor([[1, 0]]), torch.tensor([[0, 1]])).tolist() == [[2, 1]]


def test_spinful_u1_proposal_preserves_total_and_can_change_spin_sector():
    encoding = FermionSiteEncoding.vmc_torch()
    configs = torch.tensor([[encoding.up, encoding.down]])
    before_up, before_down = count_spinful_particles(configs, encoding=encoding)
    proposed, changed = propose_spinful_u1_exchange_or_hopping(
        0,
        1,
        configs,
        spin_flip_rate=1.0,
        encoding=encoding,
        generator=torch.Generator().manual_seed(7),
    )
    after_up, after_down = count_spinful_particles(proposed, encoding=encoding)
    assert changed.tolist() == [True]
    assert (after_up + after_down).tolist() == (before_up + before_down).tolist()
    assert (
        not torch.equal(after_up, before_up)
        or not torch.equal(after_down, before_down)
    )


def test_spinful_z2_proposal_preserves_parity_and_can_change_number():
    encoding = FermionSiteEncoding.symmray()
    configs = torch.tensor([[encoding.empty, encoding.empty]])
    before_up, before_down = count_spinful_particles(configs, encoding=encoding)
    proposed, changed = propose_spinful_z2_exchange_or_hopping(
        0,
        1,
        configs,
        hopping_rate=0.0,
        spin_flip_rate=0.0,
        pair_toggle_rate=1.0,
        encoding=encoding,
        generator=torch.Generator().manual_seed(8),
    )
    after_up, after_down = count_spinful_particles(proposed, encoding=encoding)
    assert changed.tolist() == [True]
    assert ((after_up + after_down) % 2).tolist() == (
        (before_up + before_down) % 2
    ).tolist()
    assert (after_up + after_down).tolist() == [2]


def test_spinful_z2z2_proposal_preserves_resolved_parities():
    encoding = FermionSiteEncoding.vmc_torch()
    configs = torch.tensor([[encoding.empty, encoding.empty]])
    before_up, before_down = count_spinful_particles(configs, encoding=encoding)
    proposed, changed = propose_spinful_z2z2_exchange_or_hopping(
        0,
        1,
        configs,
        hopping_rate=0.0,
        pair_toggle_rate=1.0,
        encoding=encoding,
        generator=torch.Generator().manual_seed(8),
    )
    after_up, after_down = count_spinful_particles(proposed, encoding=encoding)
    assert changed.tolist() == [True]
    assert ((after_up % 2).tolist(), (after_down % 2).tolist()) == (
        (before_up % 2).tolist(),
        (before_down % 2).tolist(),
    )


def test_torch_square_lattice_edges_match_row_major_open_boundary():
    graph = TorchSquareLattice(2, 3)
    assert graph.row_edges == {
        0: ((0, 1), (1, 2)),
        1: ((3, 4), (4, 5)),
    }
    assert graph.col_edges == {
        0: ((0, 3),),
        1: ((1, 4),),
        2: ((2, 5),),
    }
    assert graph.edges == (
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (0, 3),
        (1, 4),
        (2, 5),
    )


def test_torch_peps_amplitude_matches_direct_quimb_contraction():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=10,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    rows = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    amps = model(rows)

    direct = []
    for row in rows:
        tnx = peps.isel({
            peps.site_ind(site): int(row[i])
            for i, site in enumerate(peps.sites)
        })
        direct.append(tnx.contract(all))

    assert torch.allclose(amps, torch.as_tensor(direct, dtype=torch.float64))
    phase, log_abs = model.forward_log(rows)
    assert torch.allclose(phase * torch.exp(log_abs), amps)
    assert model.n_sites == 4
    assert model.n_params == sum(p.numel() for p in model.parameters())


def test_torch_peps_amplitude_supports_torch_optimizer_step():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=11,
        dtype="float64",
    )
    model = make_torch_peps_amplitude_model(
        peps,
        contraction="exact",
        dtype=torch.float64,
    )
    rows = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    target = torch.tensor([0.25, -0.1], dtype=torch.float64)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    opt.zero_grad()
    before = model(rows).detach().clone()
    loss = (model(rows) - target).square().sum()
    loss.backward()
    assert all(param.grad is not None for param in model.parameters())
    opt.step()
    after = model(rows).detach()

    assert not torch.allclose(before, after)
    peps_after = model.to_peps()
    assert tuple(peps_after.sites) == tuple(peps.sites)


def test_torch_log_derivative_matrix_matches_product_model_manual_values():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 1]])

    log_derivatives = torch_log_derivative_matrix(model, configs)

    expected = torch.tensor(
        [
            [1.0 / 2.0, 1.0 / 3.0],
            [0.0, 2.0 / 3.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(log_derivatives, expected)


def test_solve_torch_sr_direct_matches_minsr():
    generator = torch.Generator().manual_seed(13)
    log_derivatives = torch.randn(
        4,
        9,
        dtype=torch.float64,
        generator=generator,
    )
    local_energies = torch.randn(4, dtype=torch.float64, generator=generator)

    direct = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="direct",
        diag_shift=1.0e-3,
    )
    minsr = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="minsr",
        diag_shift=1.0e-3,
    )
    auto = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="auto",
        diag_shift=1.0e-3,
    )

    assert direct.method == "direct"
    assert minsr.method == "minsr"
    assert auto.method == "minsr"
    assert torch.allclose(direct.direction, minsr.direction, atol=1.0e-10)
    assert torch.allclose(auto.direction, minsr.direction, atol=1.0e-10)


def test_apply_torch_sr_update_changes_model_parameters():
    model = ProductAmplitude()
    before = model.weights.detach().clone()
    direction = torch.tensor([0.5, -1.0], dtype=torch.float64)

    apply_torch_sr_update(model, direction, learning_rate=0.1)

    assert torch.allclose(model.weights, before - 0.1 * direction)


def test_torch_peps_amplitude_supports_sr_kernel():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=14,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    rows = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    log_derivatives = torch_log_derivative_matrix(model, rows)
    local_energies = torch.tensor([0.2, -0.1], dtype=torch.float64)

    result = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="auto",
        diag_shift=1.0e-2,
    )

    assert log_derivatives.shape == (2, model.n_params)
    assert result.direction.shape == (model.n_params,)
    assert torch.isfinite(result.direction).all()


def test_torch_peps_boundary_amplitude_reuses_connected_environments():
    peps = qtn.PEPS.rand(
        Lx=3,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=15,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
    )
    configs = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.long,
    )
    graph = TorchSquareLattice(3, 3)
    amplitudes = model(configs, chunk_size=1)
    connections = heisenberg_connections(configs, graph, J=1.0)

    reused = model.connected_amplitudes(
        configs,
        amplitudes,
        connections,
        chunk_size=2,
    )
    fresh = model(connections.configs, chunk_size=2)

    assert torch.allclose(reused, fresh, rtol=1.0e-7, atol=1.0e-8)
    assert model.last_connected_reuse_stats["num_diagonal"] > 0
    assert model.last_connected_reuse_stats["num_reused"] > 0
    assert model.last_connected_reuse_stats["num_fallback"] == 0


def test_local_energy_reuses_diagonal_connections_and_chunks_offdiagonal():
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    amps = torch.ones(2, dtype=torch.float64)
    conn = heisenberg_connections(configs, [(0, 1)], J=1.0)
    amplitude = CountingAmplitude()

    energy = local_energy_from_connections(
        configs,
        amps,
        conn,
        amplitude,
        chunk_size=1,
    )

    assert torch.allclose(energy, torch.tensor([0.25, 0.25], dtype=torch.float64))
    assert [tuple(call.shape) for call in amplitude.calls] == [(1, 2), (1, 2)]


def test_torch_vmc_driver_runs_sampling_and_energy_estimate():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        connection_fn="heisenberg",
        proposal="spin",
        chunk_size=1,
        generator=torch.Generator().manual_seed(3),
    )

    result = driver.step()

    assert isinstance(result, TorchVMCStepResult)
    assert result.configs.shape == configs.shape
    assert result.amplitudes.shape == (2,)
    assert result.local_energies.shape == (2,)
    assert result.n_proposed == 2
    assert result.n_accepted == 2
    assert result.acceptance_rate == 1.0
    assert torch.isfinite(result.energy_mean)
    assert torch.isfinite(result.energy_variance)


def test_torch_vmc_driver_can_apply_sr_update():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    before = model.weights.detach().clone()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        connection_fn="transverse_ising",
        connection_kwargs={"J": 1.0, "h": 1.0},
        proposal="spin",
        chunk_size=1,
        generator=torch.Generator().manual_seed(4),
    )

    result = driver.step(sr=True, learning_rate=0.05, sr_diag_shift=1.0e-2)

    assert result.sr is not None
    assert result.sr.direction.shape == (2,)
    assert not torch.allclose(model.weights.detach(), before)
    assert torch.allclose(driver.amplitudes, model(driver.configs))


def test_torch_hamiltonian_connections_accept_explicit_local_terms():
    configs = torch.tensor([[0], [1]], dtype=torch.long)
    term = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

    conn = torch_hamiltonian_connections(configs, {0: term})
    rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(
            conn.configs,
            conn.coeffs,
            conn.batch_ids,
        )
    ]

    assert ((0,), 1.0, 0) in rows
    assert ((1,), 3.0, 0) in rows
    assert ((0,), 2.0, 1) in rows
    assert ((1,), 4.0, 1) in rows


def test_torch_vmc_driver_accepts_explicit_terms_and_estimates_sampling_rate():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    term = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: term},
        proposal="spin",
        generator=torch.Generator().manual_seed(5),
    )

    result = driver.estimate_observable(
        burn_in=1,
        n_measurements=2,
        sweeps_between=1,
    )

    assert isinstance(result, TorchVMCEnergyEstimate)
    assert result.n_samples == 4
    assert result.n_measurements == 2
    assert result.elapsed_seconds > 0.0
    assert result.samples_per_second > 0.0
    assert torch.isfinite(result.energy_mean)
    assert torch.isfinite(result.energy_stderr)
    assert result.chain_diagnostics is not None
    assert result.chain_diagnostics.n_chains == 2

    legacy = driver.estimate_energy(
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
    )
    assert isinstance(legacy, TorchVMCEnergyEstimate)


def test_torch_metropolis_sampler_preserves_chains_and_accepts_netket_aliases():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sampler = TorchMetropolisSampler(
        model,
        [(0, 1)],
        configs,
        proposal="spin",
        seed=11,
    )

    result = sampler.sample(
        n_samples=5,
        n_discard=2,
        n_thin=2,
    )

    assert isinstance(result, TorchMCMCSamples)
    assert result.configs.shape == (3, 2, 2)
    assert result.amplitudes.shape == (3, 2)
    assert result.n_samples == 6
    assert result.n_samples_per_chain == 3
    assert result.n_chains == 2
    assert result.n_discard_per_chain == 2
    assert result.sweep_size == 2
    assert result.n_proposed >= result.n_accepted >= 0
    assert result.elapsed_seconds > 0.0


def test_metropolis_exchange_sweep_uses_log_amplitude_ratio_when_available():
    result = metropolis_exchange_sweep(
        torch.tensor([[0, 1]], dtype=torch.long),
        LogOnlyAmplitude(),
        [(0, 1)],
        proposal="spin",
        generator=torch.Generator().manual_seed(0),
    )

    assert result.configs.tolist() == [[1, 0]]
    assert result.log_abs_amplitudes.tolist() == [-900.0]
    assert result.nonzero_amplitudes.tolist() == [True]


def test_torch_chain_diagnostics_reports_rhat_tau_and_effective_sample_size():
    values = torch.arange(8, dtype=torch.float64).reshape(8, 1).repeat(1, 4)
    diagnostics = torch_chain_diagnostics(values)

    assert isinstance(diagnostics, TorchChainDiagnostics)
    assert diagnostics.r_hat >= 1.0
    assert diagnostics.integrated_autocorrelation_time >= 1.0
    assert 0.0 < diagnostics.effective_sample_size <= 32.0
    assert diagnostics.rhat == diagnostics.r_hat
    assert diagnostics.tau == diagnostics.integrated_autocorrelation_time


def test_metropolis_local_sampler_infers_sites_and_chain_count():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    result = metropolis_local_sampler(
        configs,
        model,
        [(0, 1)],
        n_sites=2,
        n_samples=4,
        n_chains=2,
        n_discard_per_chain=0,
        sweep_size=1,
        proposal="spin",
        seed=12,
    )

    assert result.configs.shape == (2, 2, 2)
    assert result.n_samples == 4


def test_torch_vmc_driver_exposes_chain_preserving_sampling():
    model = ProductAmplitude()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        terms={0: torch.eye(2, dtype=torch.float64)},
        proposal="spin",
    )

    result = driver.sample(
        n_samples=3,
        n_discard=1,
        n_thin=1,
        seed=13,
    )

    assert result.configs.shape == (2, 2, 2)
    assert driver.configs.shape == (2, 2)
    assert torch.allclose(driver.amplitudes, model(driver.configs))

    estimate = driver.estimate_observable(
        n_samples=3,
        n_discard=1,
        n_thin=1,
        seed=14,
    )
    assert estimate.configs.shape == (2, 2, 2)
    assert estimate.local_energies.shape == (2, 2)
    assert estimate.n_samples == 4
    assert torch.isfinite(estimate.energy_mean)


def test_torch_vmc_driver_importance_estimate_uses_proposal_weights():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        connection_fn="heisenberg",
        proposal="spin",
    )

    class Proposal:
        def sample(self, *, samples, progbar=False):
            assert samples == 2
            assert progbar is False
            return SimpleNamespace(
                configs=[[0, 1], [1, 0]],
                omegas=([1.0, 1.0], [0, 0]),
            )

    result = driver.importance_energy_estimate(Proposal(), n_samples=2)

    assert isinstance(result, TorchVMCImportanceEstimate)
    assert result.n_samples == 2
    assert result.n_valid == 2
    assert torch.allclose(
        result.effective_sample_size,
        torch.tensor(2.0, dtype=torch.float64),
    )
    assert torch.isfinite(result.energy_mean)
    assert torch.isfinite(result.energy_stderr)


@pytest.mark.parametrize(
    ("symmetry", "encoding"),
    [
        ("Z2", FermionSiteEncoding.symmray()),
        ("U1", FermionSiteEncoding.vmc_torch()),
        ("Z2Z2", FermionSiteEncoding.vmc_torch()),
        ("U1U1", FermionSiteEncoding.vmc_torch()),
    ],
)
def test_torch_bp_metropolis_filters_fermion_symmetry_sectors(symmetry, encoding):
    class Amplitude:
        def __call__(self, rows):
            return torch.ones(rows.shape[0], dtype=torch.float64)

    if symmetry == "U1":
        valid, invalid, sector = [0, 3], [0, 0], 2
    elif symmetry == "U1U1":
        valid, invalid, sector = [2, 1], [0, 0], (1, 1)
    elif symmetry == "Z2":
        valid, invalid, sector = [0, 0], [2, 0], 0
    else:
        valid, invalid, sector = [2, 1], [0, 0], (1, 1)

    class Proposal:
        def __init__(self):
            self.calls = 0

        def sample(self, *, samples, progbar=False):
            self.calls += 1
            rows = [valid] * samples if self.calls == 1 else [invalid] * samples
            return SimpleNamespace(
                configs=rows,
                omegas=([1.0] * samples, [0] * samples),
            )

    sampler = TorchBPMetropolisSampler(
        Amplitude(),
        [],
        Proposal(),
        n_chains=2,
        symmetry=symmetry,
        sector=sector,
        encoding=encoding,
        seed=7,
    )
    initial = sampler.configs.clone()
    result = sampler.sample_sweep()

    assert result.n_proposed == 2
    assert result.n_accepted == 0
    assert torch.equal(sampler.configs, initial)


def test_torch_bp_metropolis_acceptance_uses_independence_log_ratio():
    class Amplitude:
        def __call__(self, rows):
            return torch.where(
                rows[:, 0] == 0,
                torch.ones(rows.shape[0], dtype=torch.float64),
                torch.full((rows.shape[0],), 2.0, dtype=torch.float64),
            )

    class Proposal:
        def __init__(self):
            self.calls = 0

        def sample(self, *, samples, progbar=False):
            self.calls += 1
            rows = [[0]] * samples if self.calls == 1 else [[1]] * samples
            # q(0) = 3/4 and q(1) = 1/4, so the move 0 -> 1 has ratio
            # |2|^2 * (3/4) / (|1|^2 * (1/4)) = 12 and must accept.
            probs = ([0.75] * samples, [0] * samples)
            if self.calls > 1:
                probs = ([0.25] * samples, [0] * samples)
            return SimpleNamespace(configs=rows, omegas=probs)

    sampler = TorchBPMetropolisSampler(
        Amplitude(),
        [],
        Proposal(),
        n_chains=2,
        seed=3,
    )
    result = sampler.sample_sweep()
    assert result.n_accepted == 2
    assert sampler.configs.tolist() == [[1], [1]]


@pytest.mark.parametrize(
    ("symmetry", "encoding"),
    [
        ("Z2", FermionSiteEncoding.symmray()),
        ("U1", FermionSiteEncoding.vmc_torch()),
        ("Z2Z2", FermionSiteEncoding.vmc_torch()),
        ("U1U1", FermionSiteEncoding.vmc_torch()),
    ],
)
def test_peps_bp_sampler_supports_four_state_symmray_fermions(symmetry, encoding):
    peps = _symmray_fermionic_peps(symmetry)
    result = PepsBpSampler(peps, encoding=encoding).sample(
        samples=1,
        method="exact",
        seed=2,
        bp_kwargs={"max_iterations": 3},
    )

    assert len(result.configs) == 1
    assert len(result.configs[0]) == 4
    assert all(value in {0, 1, 2, 3} for value in result.configs[0])
    assert np.isfinite(np.asarray(result.omegas[0])).all()
    assert np.isfinite(np.asarray(result.omegas[1])).all()
    assert np.isfinite(np.asarray(result.ps[0])).all()
    assert np.isfinite(np.asarray(result.ps[1])).all()


def _symmray_site_charge(symmetry):
    if symmetry == "U1":
        return lambda site: 0 if (site[0] + site[1]) % 2 == 0 else 1
    if symmetry == "U1U1":
        return lambda site: (1, 0) if (site[0] + site[1]) % 2 == 0 else (0, 1)
    if symmetry == "Z2Z2":
        return lambda site: (0, 0)
    return None


def _symmray_fermionic_peps(symmetry, *, flat=False):
    sr = pytest.importorskip("symmray")
    return sr.networks.PEPS_fermionic_rand(
        symmetry,
        2,
        2,
        2,
        phys_dim=4,
        subsizes="equal",
        flat=flat,
        seed=1,
        site_charge=_symmray_site_charge(symmetry),
    )


def _direct_peps_amplitudes(peps, rows):
    values = []
    for row in rows:
        tnx = peps.isel({
            peps.site_ind(site): int(row[i])
            for i, site in enumerate(peps.sites)
        })
        value = tnx.contract(all)
        if hasattr(value, "item"):
            value = value.item()
        values.append(value)
    return torch.as_tensor(values, dtype=torch.float64)


@pytest.mark.parametrize(
    ("symmetry", "valid_row"),
    [
        ("Z2", (0, 0, 0, 0)),
        ("U1", (0, 0, 0, 3)),
        ("Z2Z2", (0, 0, 0, 0)),
        ("U1U1", (2, 0, 1, 3)),
    ],
)
def test_torch_peps_amplitude_supports_symmray_fermionic_symmetries(
    symmetry,
    valid_row,
):
    peps = _symmray_fermionic_peps(symmetry)
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    rows = torch.tensor([valid_row, (3, 3, 3, 3)], dtype=torch.long)

    amps = model(rows)
    direct = _direct_peps_amplitudes(peps, rows)

    assert model.is_symmray
    assert model.symmray_tensor_ids
    assert torch.allclose(amps, direct)

    phase, log_abs = model.forward_log(rows)
    assert phase.shape == (2,)
    assert log_abs.shape == (2,)
    assert torch.isfinite(log_abs).all()

    model.zero_grad()
    loss = model(torch.tensor([valid_row], dtype=torch.long)).square().sum()
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_torch_peps_amplitude_supports_symmray_flat_z2_vmap():
    peps = _symmray_fermionic_peps("Z2", flat=True)
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    rows = torch.tensor([(0, 0, 0, 0), (1, 1, 1, 1)], dtype=torch.long)

    loop = model(rows)
    vmapped = torch.vmap(model.amplitude)(rows)

    assert model.is_symmray
    assert torch.allclose(vmapped, loop)


def test_symmray_fermionic_peps_feeds_hubbard_local_energy_kernel():
    peps = _symmray_fermionic_peps("U1U1")
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    graph = TorchSquareLattice(2, 2)
    encoding = FermionSiteEncoding.symmray()
    configs = torch.tensor([[2, 0, 1, 3]], dtype=torch.long)

    amps = model(configs)
    conn = spinful_fermi_hubbard_connections(
        configs,
        graph,
        t=1.0,
        U=8.0,
        encoding=encoding,
    )
    energy = local_energy_from_connections(configs, amps, conn, model)

    assert energy.shape == (1,)
    assert torch.isfinite(energy).all()


def test_torch_fermion_vmc_infers_geometry_sector_encoding_and_terms():
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=1.0,
        U=2.0,
    )
    peps = ps_to_peps(
        2,
        2,
        fermion=fermion,
        occupations=[(1, 0), (0, 1), (1, 0), (0, 1)],
        seed=21,
        dtype="complex128",
    )
    vmc = TorchFermionVMC(
        peps,
        fermion,
        n_walkers=2,
        dtype=torch.complex128,
        seed=22,
    )

    assert isinstance(vmc.metadata, TorchFermionVMCMetadata)
    assert (vmc.Lx, vmc.Ly) == (2, 2)
    assert vmc.site_order == tuple(peps.sites)
    assert vmc.sector == (2, 2)
    assert vmc.encoding == FermionSiteEncoding.vmc_torch()
    assert vmc.metadata.graph_edges == ((0, 1), (2, 3), (0, 2), (1, 3))
    assert torch.isfinite(vmc.local_energies()).all()

    sample = vmc.sample_sweep()
    n_up, n_down = count_spinful_particles(sample.configs, encoding=vmc.encoding)
    assert n_up.tolist() == [2, 2]
    assert n_down.tolist() == [2, 2]


def test_torch_fermion_vmc_accepts_z2_and_explicit_observable_terms():
    fermion = Fermion(spinful=True, symmetry="Z2", t=1.0, U=2.0)
    peps = ps_to_peps(1, 2, fermion=fermion, seed=23, dtype="complex128")
    terms = {
        (0, 1): -fermion.hopping_operator(),
        0: fermion.onsite_term(0),
        1: fermion.onsite_term(1),
    }
    vmc = TorchFermionVMC(
        peps,
        terms=terms,
        configs=[[2, 3]],
        dtype=torch.complex128,
        seed=24,
    )

    assert vmc.proposal == "spinful_z2"
    assert vmc.sector == 0
    assert vmc.encoding == FermionSiteEncoding.symmray()
    assert torch.isfinite(vmc.local_energies()).all()

    sample = vmc.sample_sweep()
    n_up, n_down = count_spinful_particles(sample.configs, encoding=vmc.encoding)
    assert ((n_up + n_down) % 2).tolist() == [0]


def test_torch_fermion_vmc_accepts_z2z2_and_bp_sampling():
    fermion = Fermion(spinful=True, symmetry="Z2Z2", t=1.0, U=2.0)
    peps = ps_to_peps(2, 2, fermion=fermion, seed=25, dtype="complex128")
    vmc = TorchFermionVMC(
        peps,
        fermion,
        n_walkers=2,
        dtype=torch.complex128,
        seed=26,
    )

    assert vmc.proposal == "spinful_z2z2"
    assert vmc.sector == (0, 0)
    sampler = vmc.make_bp_sampler(
        n_chains=2,
        bp_sampler_kwargs={"max_iterations": 3},
        sample_kwargs={"method": "exact"},
        seed=27,
    )
    result = sampler.sample(n_samples=2, n_discard=1, n_thin=1)
    n_up, n_down = count_spinful_particles(
        result.configs.reshape(-1, vmc.n_sites),
        encoding=vmc.encoding,
    )
    assert ((n_up % 2) == 0).all()
    assert ((n_down % 2) == 0).all()


def test_spinful_exchange_hopping_proposal_preserves_particle_counts():
    encoding = FermionSiteEncoding.symmray()
    configs = torch.tensor([
        [encoding.empty, encoding.up, encoding.down, encoding.double],
        [encoding.up, encoding.down, encoding.empty, encoding.double],
    ])
    before = count_spinful_particles(configs, encoding=encoding)
    proposed, changed = propose_spinful_exchange_or_hopping(
        0,
        1,
        configs,
        hopping_rate=1.0,
        encoding=encoding,
        generator=torch.Generator().manual_seed(1),
    )
    after = count_spinful_particles(proposed, encoding=encoding)
    assert changed.tolist() == [True, True]
    assert after[0].tolist() == before[0].tolist()
    assert after[1].tolist() == before[1].tolist()


def test_spin_exchange_proposal_swaps_only_different_binary_spins():
    configs = torch.tensor([[0, 1], [1, 1]])
    proposed, changed = propose_spin_exchange(0, 1, configs)
    assert proposed.tolist() == [[1, 0], [1, 1]]
    assert changed.tolist() == [True, False]


def test_spinful_fermi_hubbard_connections_include_fermionic_signs():
    encoding = FermionSiteEncoding.symmray()
    graph = [(0, 1)]
    configs = torch.tensor([
        [encoding.up, encoding.down],
        [encoding.double, encoding.empty],
    ])
    conn = spinful_fermi_hubbard_connections(
        configs,
        graph,
        t=1.0,
        U=8.0,
        encoding=encoding,
    )
    rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(conn.configs, conn.coeffs, conn.batch_ids)
    ]

    assert ((encoding.empty, encoding.double), 1.0, 0) in rows
    assert ((encoding.double, encoding.empty), 1.0, 0) in rows
    assert ((encoding.up, encoding.down), 1.0, 1) in rows
    assert ((encoding.down, encoding.up), -1.0, 1) in rows
    assert ((encoding.double, encoding.empty), 8.0, 1) in rows


def test_heisenberg_and_transverse_ising_connections():
    graph = [(0, 1)]
    configs = torch.tensor([[0, 1]])

    heis = heisenberg_connections(configs, graph, J=2.0)
    heis_rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(heis.configs, heis.coeffs, heis.batch_ids)
    ]
    assert ((1, 0), 1.0, 0) in heis_rows
    assert ((0, 1), -0.5, 0) in heis_rows

    ising = transverse_ising_connections(configs, graph, J=2.0, h=3.0)
    ising_rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(ising.configs, ising.coeffs, ising.batch_ids)
    ]
    assert ((0, 1), -0.5, 0) in ising_rows
    assert ((1, 1), 1.5, 0) in ising_rows
    assert ((0, 0), 1.5, 0) in ising_rows


def test_local_energy_from_connections_matches_constant_amplitude_sum():
    configs = torch.tensor([[0, 1]])
    amps = torch.ones(1, dtype=torch.float64)
    conn = heisenberg_connections(configs, [(0, 1)], J=2.0)

    def amplitude_fn(rows):
        return torch.ones(rows.shape[0], dtype=torch.float64)

    energy = local_energy_from_connections(configs, amps, conn, amplitude_fn)
    assert energy.tolist() == [0.5]


def test_torch_peps_amplitude_feeds_local_energy_kernel():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=12,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    graph = TorchSquareLattice(2, 2)
    configs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    amps = model(configs)
    conn = heisenberg_connections(configs, graph, J=1.0)
    energy = local_energy_from_connections(configs, amps, conn, model)

    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


def test_metropolis_exchange_sweep_accepts_constant_amplitude_proposals():
    graph = TorchSquareLattice(1, 2)
    configs = torch.tensor([[0, 1], [1, 0]])

    def amplitude_fn(rows):
        return torch.ones(rows.shape[0], dtype=torch.float64)

    result = metropolis_exchange_sweep(
        configs,
        amplitude_fn,
        graph,
        proposal="spin",
        generator=torch.Generator().manual_seed(2),
    )
    assert result.configs.tolist() == [[1, 0], [0, 1]]
    assert result.n_proposed == 2
    assert result.n_accepted == 2
    assert result.acceptance_rate == 1.0


def test_random_sector_initializers_fix_particle_numbers():
    spin = random_spin_configs(
        4,
        6,
        2,
        generator=torch.Generator().manual_seed(3),
    )
    assert spin.sum(dim=1).tolist() == [2, 2, 2, 2]

    encoding = FermionSiteEncoding.symmray()
    fermion = random_spinful_configs(
        4,
        6,
        2,
        3,
        encoding=encoding,
        generator=torch.Generator().manual_seed(4),
    )
    n_up, n_down = count_spinful_particles(fermion, encoding=encoding)
    assert n_up.tolist() == [2, 2, 2, 2]
    assert n_down.tolist() == [3, 3, 3, 3]
