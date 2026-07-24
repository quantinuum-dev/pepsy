"""Tests for Pepsy's optional torch VMC kernels."""

import warnings

import pytest
import quimb.tensor as qtn
import numpy as np
from itertools import product
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from pepsy.tensors import Fermion, hrs_to_peps, ps_to_peps  # noqa: E402
from pepsy.sampling import PepsBpSampler  # noqa: E402
import pepsy.vmc.torch as torch_vmc  # noqa: E402
from pepsy.vmc.torch import (  # noqa: E402
    FermionSiteEncoding,
    SpinlessSiteEncoding,
    TorchPEPSAmplitude,
    TorchPEPSBoundaryAmplitude,
    TorchFermionVMC,
    TorchFermionVMCMetadata,
    TorchChainDiagnostics,
    TorchConnections,
    TorchMCMCSamples,
    TorchMetropolisSampler,
    TorchBPMetropolisSampler,
    TorchVMCDriver,
    TorchVMCSetup,
    TorchVMCEnergyEstimate,
    TorchVMCImportanceEstimate,
    TorchVMCStepResult,
    TorchSquareLattice,
    apply_torch_sr_update,
    build_torch_vmc,
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
    _observable_statistics,
)
from pepsy.vmc import (  # noqa: E402
    ContractionConfig,
    OptimizationConfig,
    SamplingConfig,
    VMCMeasurement,
    VMCOptimizationResult,
    VMCProblem,
)


class ProductAmplitude(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.tensor([2.0, 3.0], dtype=torch.float64)
        )

    def forward(self, configs):
        return self.weights[configs].prod(dim=1)


class ComplexProductAmplitude(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.tensor([1.3 + 0.4j, 0.8 - 0.2j], dtype=torch.complex128)
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


@pytest.mark.parametrize(
    ("Lx", "Ly", "bond_dim"),
    [
        (3, 4, 2),
        (3, 4, 4),
        (4, 4, 2),
        (4, 4, 4),
    ],
)
def test_torch_peps_amplitude_contractions_match_exact_reference(
    Lx,
    Ly,
    bond_dim,
):
    """All PEPS contraction modes agree with direct contraction at large chi."""
    peps = qtn.PEPS.rand(
        Lx=Lx,
        Ly=Ly,
        bond_dim=bond_dim,
        phys_dim=2,
        seed=1000 + 100 * Lx + 10 * Ly + bond_dim,
        dtype="float64",
    )
    n_sites = Lx * Ly
    parity = torch.arange(n_sites, dtype=torch.long) % 2
    rows = torch.stack((parity, 1 - parity, (torch.arange(n_sites) // 2) % 2))

    reference = torch.as_tensor(
        [
            peps.isel({
                peps.site_ind(site): int(row[i])
                for i, site in enumerate(peps.sites)
            }).contract(all)
            for row in rows
        ],
        dtype=torch.float64,
    )

    # Use zero cutoff and a generous chi so these are contraction-path
    # consistency checks. Production runs can deliberately choose a smaller
    # chi and should then report the resulting approximation error.
    chi = 16 if bond_dim == 2 else 64
    for contraction in ("exact", "boundary", "ctmrg", "hotrg"):
        kwargs = {"contraction": contraction, "dtype": torch.float64}
        if contraction != "exact":
            kwargs.update(chi=chi, cutoff=0.0)
        model = TorchPEPSAmplitude(peps, **kwargs)
        amplitudes = model(rows)

        assert torch.isfinite(amplitudes).all()
        assert torch.allclose(
            amplitudes,
            reference,
            rtol=2.0e-10,
            atol=1.0e-8,
        ), contraction

        phase, log_abs = model.forward_log(rows)
        assert torch.allclose(
            phase * torch.exp(log_abs),
            amplitudes,
            rtol=2.0e-10,
            atol=1.0e-8,
        ), contraction


@pytest.mark.parametrize("bond_dim", [3, 4])
def test_torch_peps_4x6_contractions_honor_production_options(bond_dim):
    """Boundary and CTMRG options remain accurate on a larger flat PEPS TN."""
    Lx, Ly = 4, 6
    peps = qtn.PEPS.rand(
        Lx=Lx,
        Ly=Ly,
        bond_dim=bond_dim,
        phys_dim=2,
        seed=1400 + bond_dim,
        dtype="float64",
    )
    n_sites = Lx * Ly
    parity = torch.arange(n_sites, dtype=torch.long) % 2
    rows = torch.stack((parity, 1 - parity, (torch.arange(n_sites) // 2) % 2))
    reference = torch.as_tensor(
        [
            peps.isel({
                peps.site_ind(site): int(row[i])
                for i, site in enumerate(peps.sites)
            }).contract(all)
            for row in rows
        ],
        dtype=torch.float64,
    )

    cutoff = 1.0e-10
    chi_contract = 32 if bond_dim == 3 else 64
    contract_opt = "auto-hq"
    boundary_opts = {
        "mode": "mps",
        "final_contract": True,
        "final_contract_opts": {"optimize": contract_opt},
        "sequence": ["xmin", "xmax", "ymin", "ymax"],
        "equalize_norms": False,
        "progbar": False,
    }
    ctmrg_opts = {
        "final_contract": False,
        "final_contract_opts": {"optimize": contract_opt},
        "max_separation": 1,
        "inplace": False,
        "equalize_norms": False,
        "progbar": False,
    }

    for contraction, contraction_opts in (
        ("boundary", boundary_opts),
        ("ctmrg", ctmrg_opts),
    ):
        model = TorchPEPSAmplitude(
            peps,
            contraction=contraction,
            chi=chi_contract,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            dtype=torch.float64,
        )
        amplitudes = model(rows)
        assert torch.isfinite(amplitudes).all()
        assert torch.allclose(
            amplitudes,
            reference,
            rtol=2.0e-10,
            atol=1.0e-6,
        ), (bond_dim, contraction)

        phase, log_abs = model.forward_log(rows)
        assert torch.allclose(
            phase * torch.exp(log_abs),
            amplitudes,
            rtol=2.0e-10,
            atol=1.0e-6,
        ), (bond_dim, contraction)


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


def test_torch_log_derivative_matrix_supports_complex_holomorphic_parameters():
    model = ComplexProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]])

    log_derivatives = torch_log_derivative_matrix(
        model,
        configs,
        complex_parameter_mode="holomorphic",
    )
    expected = model.weights.detach().reciprocal().expand(2, -1)
    assert torch.allclose(log_derivatives, expected, rtol=1.0e-10, atol=1.0e-12)

    # Check both real and imaginary parameter directions against finite
    # differences. This catches conjugation mistakes in complex autograd.
    epsilon = 1.0e-6
    amplitudes = model(configs).detach()
    weights = model.weights.detach()
    for parameter in range(weights.numel()):
        plus = weights.clone()
        minus = weights.clone()
        plus[parameter] += epsilon
        minus[parameter] -= epsilon
        finite_real = (
            plus[configs].prod(dim=1) - minus[configs].prod(dim=1)
        ) / (2.0 * epsilon) / amplitudes

        plus = weights.clone()
        minus = weights.clone()
        plus[parameter] += 1j * epsilon
        minus[parameter] -= 1j * epsilon
        finite_imag = (
            plus[configs].prod(dim=1) - minus[configs].prod(dim=1)
        ) / (2.0j * epsilon) / amplitudes

        assert torch.allclose(
            finite_real,
            log_derivatives[:, parameter],
            rtol=1.0e-8,
            atol=1.0e-9,
        )
        assert torch.allclose(
            finite_imag,
            log_derivatives[:, parameter],
            rtol=1.0e-8,
            atol=1.0e-9,
        )


def test_torch_log_derivative_matrix_supports_complex_real_imag_parameters():
    model = ComplexProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]])
    log_derivatives = torch_log_derivative_matrix(
        model,
        configs,
        complex_parameter_mode="real-imag",
    )
    weights = model.weights.detach()
    expected_row = torch.stack(
        (
            weights[0].reciprocal(),
            1j * weights[0].reciprocal(),
            weights[1].reciprocal(),
            1j * weights[1].reciprocal(),
        )
    )
    assert log_derivatives.shape == (2, 4)
    assert torch.allclose(
        log_derivatives,
        expected_row.expand(2, -1),
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    epsilon = 1.0e-6
    amplitudes = model(configs).detach()
    for parameter in range(weights.numel()):
        plus = weights.clone()
        minus = weights.clone()
        plus[parameter] += epsilon
        minus[parameter] -= epsilon
        finite_real = (
            plus[configs].prod(dim=1) - minus[configs].prod(dim=1)
        ) / (2.0 * epsilon) / amplitudes

        plus = weights.clone()
        minus = weights.clone()
        plus[parameter] += 1j * epsilon
        minus[parameter] -= 1j * epsilon
        finite_imag = (
            plus[configs].prod(dim=1) - minus[configs].prod(dim=1)
        ) / (2.0 * epsilon) / amplitudes

        assert torch.allclose(
            finite_real,
            log_derivatives[:, 2 * parameter],
            rtol=1.0e-8,
            atol=1.0e-9,
        )
        assert torch.allclose(
            finite_imag,
            log_derivatives[:, 2 * parameter + 1],
            rtol=1.0e-8,
            atol=1.0e-9,
        )


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


def test_solve_torch_sr_weighted_direct_matches_minsr():
    generator = torch.Generator().manual_seed(133)
    log_derivatives = torch.randn(
        5,
        8,
        dtype=torch.float64,
        generator=generator,
    )
    local_energies = torch.randn(5, dtype=torch.float64, generator=generator)
    weights = torch.tensor([0.05, 0.10, 0.15, 0.25, 0.45], dtype=torch.float64)

    direct = solve_torch_sr(
        log_derivatives,
        local_energies,
        sample_weights=weights,
        method="direct",
        diag_shift=1.0e-3,
    )
    minsr = solve_torch_sr(
        log_derivatives,
        local_energies,
        sample_weights=weights,
        method="minsr",
        diag_shift=1.0e-3,
    )

    assert torch.allclose(direct.direction, minsr.direction, atol=1.0e-10)
    assert direct.info["effective_sample_size"] == pytest.approx(
        float(1.0 / weights.square().sum())
    )


def test_solve_torch_sr_complex_matches_minsr_and_reports_residual():
    generator = torch.Generator().manual_seed(130)
    log_derivatives = torch.randn(
        6,
        4,
        dtype=torch.complex128,
        generator=generator,
    ) + 1j * torch.randn(6, 4, dtype=torch.complex128, generator=generator)
    local_energies = torch.randn(
        6,
        dtype=torch.complex128,
        generator=generator,
    ) + 1j * torch.randn(6, dtype=torch.complex128, generator=generator)

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

    assert torch.allclose(direct.direction, minsr.direction, atol=1.0e-10)
    assert direct.info["relative_residual"] < 1.0e-10
    assert minsr.info["relative_residual"] < 1.0e-10

    model = ComplexProductAmplitude()
    before = model.weights.detach().clone()
    direction = torch.tensor([0.2 - 0.1j, -0.3 + 0.4j], dtype=torch.complex128)
    apply_torch_sr_update(model, direction, learning_rate=0.25)
    assert torch.allclose(model.weights, before - 0.25 * direction)


def test_solve_torch_sr_real_imag_matches_minsr_and_applies_coordinates():
    generator = torch.Generator().manual_seed(131)
    log_derivatives = torch.randn(
        6,
        4,
        dtype=torch.complex128,
        generator=generator,
    ) + 1j * torch.randn(6, 4, dtype=torch.complex128, generator=generator)
    local_energies = torch.randn(
        6,
        dtype=torch.complex128,
        generator=generator,
    ) + 1j * torch.randn(6, dtype=torch.complex128, generator=generator)

    direct = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="direct",
        diag_shift=1.0e-3,
        parameter_mode="real-imag",
    )
    minsr = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="minsr",
        diag_shift=1.0e-3,
        parameter_mode="real-imag",
    )

    assert not torch.is_complex(direct.direction)
    assert torch.allclose(direct.direction, minsr.direction, atol=1.0e-10)
    assert direct.info["relative_residual"] < 1.0e-10
    assert minsr.info["relative_residual"] < 1.0e-10

    model = ComplexProductAmplitude()
    before = model.weights.detach().clone()
    direction = torch.tensor([0.2, -0.1, -0.3, 0.4], dtype=torch.float64)
    apply_torch_sr_update(
        model,
        direction,
        learning_rate=0.25,
        parameter_mode="real-imag",
    )
    expected = before - 0.25 * torch.tensor(
        [0.2 - 0.1j, -0.3 + 0.4j],
        dtype=torch.complex128,
    )
    assert torch.allclose(model.weights, expected)


def test_torch_sr_uses_scheduled_shift_and_cholesky_when_well_conditioned():
    log_derivatives = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    local_energies = torch.tensor([1.0, -1.0, 0.5, -0.5])
    calls = []

    def schedule(step):
        calls.append(step)
        return 0.25 / (step + 1)

    result = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="direct",
        diag_shift=schedule,
        step=1,
    )

    assert calls == [1]
    assert result.diag_shift == 0.125
    assert result.info["step"] == 1
    assert result.info["solver"] == "cholesky"


def test_torch_sr_falls_back_to_pseudoinverse_for_rank_deficient_metric():
    log_derivatives = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    local_energies = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)

    result = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="direct",
        diag_shift=0.0,
    )

    assert result.info["solver"] == "pinv"
    assert torch.isfinite(result.direction).all()


def test_torch_sr_spring_momentum_keeps_only_unsampled_complement():
    log_derivatives = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    local_energies = torch.tensor([1.0, -1.0, 1.0, -1.0])
    previous_direction = torch.tensor([0.0, 2.0], dtype=torch.float64)

    without_momentum = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="direct",
        diag_shift=1.0e-2,
    )
    with_momentum = solve_torch_sr(
        log_derivatives,
        local_energies,
        method="direct",
        diag_shift=1.0e-2,
        momentum=0.5,
        previous_direction=previous_direction,
    )

    assert torch.allclose(
        with_momentum.direction[0],
        without_momentum.direction[0],
    )
    assert torch.allclose(
        with_momentum.direction[1],
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert with_momentum.info["spring_complement_norm"] == pytest.approx(2.0)


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


@pytest.mark.parametrize("parameter_mode", ("holomorphic", "real-imag"))
def test_torch_peps_batched_log_derivatives_match_loop(parameter_mode):
    """The batched PEPS Jacobian must preserve the legacy SR derivatives."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=16,
        dtype="complex128",
    )
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.complex128)
    rows = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])

    batched = torch_log_derivative_matrix(
        model,
        rows,
        complex_parameter_mode=parameter_mode,
        derivative_backend="batched",
    )
    loop = torch_log_derivative_matrix(
        model,
        rows,
        complex_parameter_mode=parameter_mode,
        derivative_backend="loop",
    )

    assert batched.shape == loop.shape
    assert torch.allclose(batched, loop, rtol=1.0e-8, atol=1.0e-10)


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


def test_torch_peps_boundary_cached_closure_uses_final_contraction_options():
    """Cached local estimators must retain the caller's scalar path choice."""
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=150,
        dtype="float64",
    )
    optimizer = object()
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        contraction_opts={"final_contract_opts": {"optimize": optimizer}},
        dtype=torch.float64,
    )

    class ReusedBoundary:
        def __or__(self, other):
            return self

        def view_as_(self, *args, **kwargs):
            self.view_args = args
            self.view_kwargs = kwargs
            return self

        def contract_boundary_from_xmin_(self, **kwargs):
            self.boundary_kwargs = kwargs

        def contract(self, *args, **kwargs):
            self.contract_args = args
            self.contract_kwargs = kwargs
            return torch.tensor(1.0, dtype=torch.float64)

    reused = ReusedBoundary()
    envs = {("xmin", 0): reused, ("xmax", 0): reused}
    result = model._contract_axis_strip(
        None,
        object(),
        "x",
        (0,),
        envs,
        torch.tensor(1.0, dtype=torch.float64),
    )

    assert result.item() == 1.0
    assert reused.contract_args == (all,)
    assert reused.contract_kwargs == {"optimize": optimizer}


def test_torch_peps_final_contraction_retries_empty_reusable_tree():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=1501,
        dtype="float64",
    )
    optimizer = object()
    model = TorchPEPSAmplitude(
        peps,
        contraction="boundary",
        chi=4,
        contraction_opts={"final_contract_opts": {"optimize": optimizer}},
        dtype=torch.float64,
    )

    class EmptyReusableTree:
        def __init__(self):
            self.optimizers = []

        def contract(self, *args, **kwargs):
            self.optimizers.append(kwargs["optimize"])
            if kwargs["optimize"] is optimizer:
                raise KeyError("tree")
            return torch.tensor(1.0, dtype=torch.float64)

    closure = EmptyReusableTree()
    contract_kwargs = {}

    def approximate_contract(**kwargs):
        contract_kwargs.update(kwargs)
        return closure

    with pytest.warns(RuntimeWarning, match="produced no cotengra tree"):
        result = model._contract_approximate(
            approximate_contract,
            close_final=True,
            final_contract=True,
            final_contract_opts={"optimize": optimizer},
        )

    assert result.item() == 1.0
    assert contract_kwargs["final_contract"] is False
    assert closure.optimizers == [optimizer, "auto-hq"]
    assert model.final_optimizer_fallbacks == 1


def test_torch_peps_boundary_tries_alternative_axis_before_fallback(monkeypatch):
    """A separated PBC-style update should retain a boundary reuse route."""
    peps = qtn.PEPS.rand(
        Lx=3,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=151,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
    )
    parent = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0]])
    target = parent.clone()
    # The endpoints differ in both coordinates. The short x window is tried
    # first; y is the retained alternative boundary path.
    first = model.sites.index((0, 0))
    second = model.sites.index((1, 2))
    target[0, first] = 1 - target[0, first]
    target[0, second] = 1 - target[0, second]
    current = model(parent)
    connections = TorchConnections(
        configs=target,
        coeffs=torch.ones(1, dtype=torch.float64),
        batch_ids=torch.zeros(1, dtype=torch.long),
    )

    original = model._contract_cached_axis_window

    def reject_x(*args, **kwargs):
        if args[3] == "x":
            raise RuntimeError("exercise the alternative boundary axis")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_contract_cached_axis_window", reject_x)
    connected = model.connected_amplitudes(parent, current, connections)
    fresh = model(target)

    assert torch.allclose(connected, fresh, rtol=1.0e-7, atol=1.0e-8)
    assert model.last_connected_reuse_stats["num_alternative_axis_reused"] == 1
    assert model.last_connected_reuse_stats["num_fallback"] == 0


def test_torch_peps_boundary_groups_parent_strips_and_caches_templates(monkeypatch):
    """Several targets in one plane should share one parent strip template."""
    peps = qtn.PEPS.rand(
        Lx=3,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=153,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
    )
    parent = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0]])
    pairs = (((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 0), (0, 2)))
    targets = []
    for left, right in pairs:
        target = parent.clone()
        for site in (left, right):
            index = model.sites.index(site)
            target[0, index] = 1 - target[0, index]
        targets.append(target[0])
    connections = TorchConnections(
        configs=torch.stack(targets),
        coeffs=torch.ones(len(targets), dtype=torch.float64),
        batch_ids=torch.zeros(len(targets), dtype=torch.long),
    )
    amplitudes = model(parent)

    select_calls = []
    original_select = model._select_config

    def count_select(*args, **kwargs):
        select_calls.append(1)
        return original_select(*args, **kwargs)

    monkeypatch.setattr(model, "_select_config", count_select)
    grouped = model.connected_amplitudes(parent, amplitudes, connections)
    first_stats = model.last_connected_reuse_stats

    assert first_stats["num_groups"] == 1
    assert first_stats["num_grouped_connections"] == len(targets)
    assert first_stats["num_strip_builds"] == 1
    assert first_stats["num_fallback"] == 0
    # One parent selection serves all three local targets; the target strips
    # are cloned from the cached template and only their physical data change.
    assert len(select_calls) == 1

    fresh = model(connections.configs)
    assert torch.allclose(grouped, fresh, rtol=1.0e-7, atol=1.0e-8)

    select_calls.clear()
    repeated = model.connected_amplitudes(parent, amplitudes, connections)
    repeated_stats = model.last_connected_reuse_stats
    assert torch.allclose(repeated, fresh, rtol=1.0e-7, atol=1.0e-8)
    assert repeated_stats["num_strip_cache_hits"] == 1
    assert not select_calls


def test_torch_peps_boundary_amplitude_batches_large_connected_sets():
    """Large boundary local-energy batches use the vmapped contraction path."""
    peps = qtn.PEPS.rand(
        Lx=4,
        Ly=4,
        bond_dim=4,
        phys_dim=2,
        seed=16,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=64,
        cutoff=0.0,
        dtype=torch.float64,
    )
    configs = torch.randint(
        0,
        2,
        (8, 16),
        generator=torch.Generator().manual_seed(17),
    )
    graph = TorchSquareLattice(4, 4)
    amplitudes = model(configs)
    connections = heisenberg_connections(configs, graph, J=1.0)

    connected = model.connected_amplitudes(
        configs,
        amplitudes,
        connections,
        chunk_size=32,
    )
    reference = model(connections.configs, chunk_size=32)

    assert torch.allclose(connected, reference, rtol=2.0e-10, atol=1.0e-8)
    assert model.last_connected_reuse_stats["num_batched"] >= 64
    assert model.last_connected_reuse_stats["num_fallback"] == 0


def test_torch_peps_boundary_amplitude_can_vmap_proposal_batch():
    peps = qtn.PEPS.rand(
        Lx=3,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=167,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
        proposal_batching="vmap",
    )
    parent = torch.tensor([
        (0, 1, 0, 1, 0, 1, 0, 1, 0),
        (1, 0, 1, 0, 1, 0, 1, 0, 1),
    ])
    target = parent.clone()
    target[:, 0], target[:, 1] = parent[:, 1], parent[:, 0]
    current = model(parent)

    proposed = model.proposal_amplitudes(parent, target, current)

    assert torch.allclose(proposed, model(target), rtol=1.0e-7, atol=1.0e-8)
    assert model.last_proposal_cache_stats["num_vmapped"] == 2
    assert model.last_proposal_cache_stats["num_environment_builds"] == 0


def test_torch_peps_boundary_amplitude_vmaps_flat_symmray_proposals():
    peps = _symmray_fermionic_peps("Z2", flat=True)
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
        proposal_batching="vmap",
    )
    parent = torch.tensor([(0, 0, 0, 0), (0, 0, 0, 0)])
    target = torch.tensor([(1, 1, 0, 0), (0, 0, 1, 1)])
    current = model(parent)

    proposed = model.proposal_amplitudes(parent, target, current)

    assert model.is_symmray
    assert torch.allclose(proposed, model(target), rtol=1.0e-7, atol=1.0e-8)
    assert model.last_proposal_cache_stats["num_vmapped"] == 2
    assert model.last_proposal_cache_stats["num_environment_builds"] == 0


def test_torch_peps_boundary_factory_selects_cached_model():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=161,
        dtype="float64",
    )

    model = make_torch_peps_amplitude_model(
        peps,
        contraction="boundary",
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
    )

    assert isinstance(model, TorchPEPSBoundaryAmplitude)


def test_torch_peps_boundary_caches_local_proposals_and_invalidates():
    peps = qtn.PEPS.rand(
        Lx=3,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=162,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        chi=4,
        cutoff=0.0,
        dtype=torch.float64,
    )
    parent = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0]])
    target = parent.clone()
    target[0, 0], target[0, 1] = target[0, 1], target[0, 0]
    current = model(parent)

    cached = model.proposal_amplitudes(parent, target, current)
    fresh = model(target)
    assert torch.allclose(cached, fresh, rtol=1.0e-7, atol=1.0e-8)
    assert model.last_proposal_cache_stats["num_environment_builds"] == 1

    model.proposal_amplitudes(parent, target, current)
    assert model.last_proposal_cache_stats["num_transition_cache_hits"] == 1

    with torch.no_grad():
        model.params[0].add_(0.01)
    model.proposal_amplitudes(parent, target, current)
    assert model.last_proposal_cache_stats["num_transition_cache_hits"] == 0


@pytest.mark.parametrize(
    ("symmetry", "spinful", "shape", "occupations"),
    (
        ("U1", False, (1, 2), (1, 0)),
        (
            "U1U1",
            True,
            (2, 2),
            [(1, 0), (0, 1), (1, 0), (0, 1)],
        ),
    ),
)
def test_torch_peps_boundary_serial_amplitude_cache_deduplicates_and_invalidates(
    symmetry, spinful, shape, occupations
):
    """Serial native-symmetry amplitudes reuse values and track updates."""
    fermion = Fermion(
        symmetry=symmetry,
        spinful=spinful,
        t=1.0,
        U=2.0,
    )
    peps = ps_to_peps(
        *shape,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
        seed=163,
    )
    vmc = TorchFermionVMC(
        peps,
        fermion,
        n_walkers=2,
        dtype=torch.complex128,
        amplitude_batching="serial",
        seed=164,
    )
    model = vmc.model
    rows = torch.cat((vmc.configs, vmc.configs), dim=0)
    model.clear_boundary_cache()

    with torch.no_grad():
        first = model(rows)
        first_stats = dict(model.last_amplitude_cache_stats)
        second = model(rows)
        second_stats = dict(model.last_amplitude_cache_stats)

    assert torch.allclose(first, second)
    assert first_stats == {
        "num_requests": rows.shape[0],
        "num_unique_requests": 1,
        "num_hits": 0,
        "num_misses": 1,
    }
    assert second_stats == {
        "num_requests": rows.shape[0],
        "num_unique_requests": 1,
        "num_hits": 1,
        "num_misses": 0,
    }

    with torch.no_grad():
        model.params[0].add_(0.01)
        refreshed = model(rows)
        refreshed_stats = dict(model.last_amplitude_cache_stats)
        direct = TorchPEPSAmplitude.forward(model, rows)

    assert refreshed_stats["num_hits"] == 0
    assert refreshed_stats["num_misses"] == 1
    assert torch.allclose(refreshed, direct)


def test_metropolis_exchange_sweep_uses_cached_proposal_hook():
    class ProposalAwareAmplitude:
        def __init__(self):
            self.calls = 0

        def __call__(self, configs):
            return torch.ones(configs.shape[0], dtype=torch.float64)

        def proposal_amplitudes(
            self,
            parent_configs,
            target_configs,
            current_amplitudes,
            *,
            chunk_size=None,
        ):
            del parent_configs, target_configs, chunk_size
            self.calls += 1
            return current_amplitudes

    amplitude = ProposalAwareAmplitude()
    result = metropolis_exchange_sweep(
        torch.tensor([[0, 1], [1, 0]]),
        amplitude,
        [(0, 1)],
        proposal="spin",
        generator=torch.Generator().manual_seed(163),
    )

    assert amplitude.calls == 1
    assert result.n_accepted == result.n_proposed == 2


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


def test_local_energy_coalesces_connections_and_batches_unique_targets():
    configs = torch.tensor([[0], [0]], dtype=torch.long)
    amplitudes = torch.ones(2, dtype=torch.float64)
    connections = TorchConnections(
        configs=torch.tensor([[1], [1], [0], [1]], dtype=torch.long),
        coeffs=torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=torch.float64),
        batch_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )

    class ConnectionAwareAmplitude:
        def __init__(self):
            self.connection_sizes = []

        def __call__(self, rows):
            return torch.ones(rows.shape[0], dtype=torch.float64)

        def connected_amplitudes(
            self,
            _configs,
            _amplitudes,
            connected,
            *,
            chunk_size=None,
            reuse_diagonal=True,
        ):
            del chunk_size, reuse_diagonal
            self.connection_sizes.append(int(connected.configs.shape[0]))
            return torch.ones(connected.configs.shape[0], dtype=torch.float64)

    amplitude = ConnectionAwareAmplitude()
    energy = local_energy_from_connections(
        configs,
        amplitudes,
        connections,
        amplitude,
    )

    assert torch.allclose(energy, torch.tensor([3.0, 9.0], dtype=torch.float64))
    assert amplitude.connection_sizes == [3]

    # The two off-diagonal rows target the same configuration, so the
    # ordinary amplitude path should contract that target only once.
    counting = CountingAmplitude()
    local_energy_from_connections(
        configs,
        amplitudes,
        connections,
        counting,
    )
    assert [tuple(call.shape) for call in counting.calls] == [(1, 1)]


def test_local_energy_handles_complex_coefficients_with_real_amplitudes():
    configs = torch.tensor([[0], [1]], dtype=torch.long)
    amps = torch.ones(2, dtype=torch.float64)
    amplitude = CountingAmplitude()

    exactly_real = TorchConnections(
        configs=configs.clone(),
        coeffs=torch.tensor([1.0 + 0.0j, -2.0 + 0.0j], dtype=torch.complex128),
        batch_ids=torch.tensor([0, 1]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energy = local_energy_from_connections(
            configs,
            amps,
            exactly_real,
            amplitude,
        )
    assert energy.dtype == torch.float64
    assert torch.allclose(energy, torch.tensor([1.0, -2.0], dtype=torch.float64))

    complex_phase = TorchConnections(
        configs=configs.clone(),
        coeffs=torch.tensor([1.0 + 0.5j, -2.0 - 0.25j], dtype=torch.complex128),
        batch_ids=torch.tensor([0, 1]),
    )
    energy = local_energy_from_connections(configs, amps, complex_phase, amplitude)
    assert energy.dtype == torch.complex128
    assert torch.allclose(
        energy,
        torch.tensor([1.0 + 0.5j, -2.0 - 0.25j], dtype=torch.complex128),
    )


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


def test_torch_vmc_driver_can_disable_stable_log_sampling():
    class CountingLogProductAmplitude(ProductAmplitude):
        def __init__(self):
            super().__init__()
            self.log_calls = 0

        def forward_log(self, configs):
            self.log_calls += 1
            amplitudes = self(configs)
            return (
                torch.ones_like(amplitudes, dtype=torch.complex128),
                amplitudes.abs().log(),
            )

    model = CountingLogProductAmplitude()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        connection_fn="heisenberg",
        proposal="spin",
        log_amplitude_fn=False,
        generator=torch.Generator().manual_seed(41),
    )

    result = driver.sample_sweep()

    assert result.log_abs_amplitudes is None
    assert driver.log_amplitude_fn is None
    assert model.log_calls == 0


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


def test_torch_vmc_measures_and_optimizes_supplied_importance_batch():
    model = ProductAmplitude()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        connection_fn="transverse_ising",
        connection_kwargs={"J": 1.0, "h": 1.0},
        proposal="spin",
        generator=torch.Generator().manual_seed(46),
    )
    supplied = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=torch.long,
    )
    proposal_log_probs = torch.log(
        torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float64)
    )

    estimate = driver.measure_samples(
        supplied,
        proposal_log_probs=proposal_log_probs,
    )
    expected_weights = (
        model(supplied).abs().square() / proposal_log_probs.exp()
    )
    expected_weights = expected_weights / expected_weights.sum()
    assert torch.allclose(estimate.importance_weights.reshape(-1), expected_weights)
    assert estimate.chain_diagnostics is None
    assert torch.allclose(
        estimate.energy_mean,
        (expected_weights * estimate.local_energies.reshape(-1)).sum(),
    )

    before = model.weights.detach().clone()
    current_walkers = driver.configs.detach().clone()
    result = driver.step(
        samples=supplied,
        proposal_log_probs=proposal_log_probs,
        sr=True,
        learning_rate=0.05,
        sr_diag_shift=1.0e-2,
    )
    assert result.sample_source == "provided"
    assert result.acceptance_rate == 0.0
    assert result.sr is not None
    assert torch.allclose(result.importance_weights, expected_weights)
    assert float(result.effective_sample_size) == pytest.approx(
        float(1.0 / expected_weights.square().sum())
    )
    assert torch.equal(driver.configs, current_walkers)
    assert not torch.allclose(model.weights.detach(), before)


def test_torch_vmc_sr_optimization_updates_from_supplied_weighted_batch():
    model = ProductAmplitude()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        connection_fn="transverse_ising",
        connection_kwargs={"J": 1.0, "h": 1.0},
        proposal="spin",
    )
    before = model.weights.detach().clone()
    history = driver.optimize(
        optimization=OptimizationConfig(method="sr", n_steps=1, learning_rate=0.01),
        samples=torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.long),
        weights=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
    )

    assert history[0].sr is not None
    assert not torch.allclose(model.weights.detach(), before)


def test_torch_vmc_driver_tracks_sr_schedule_and_spring_state():
    model = ProductAmplitude()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        connection_fn="transverse_ising",
        connection_kwargs={"J": 1.0, "h": 1.0},
        proposal="spin",
        generator=torch.Generator().manual_seed(44),
    )
    steps = []

    def shift_schedule(step):
        steps.append(step)
        return 1.0e-2

    first = driver.step(
        sr=True,
        learning_rate=1.0e-3,
        sr_diag_shift=shift_schedule,
        sr_momentum=0.5,
    )
    second = driver.step(
        sr=True,
        learning_rate=1.0e-3,
        sr_diag_shift=shift_schedule,
        sr_momentum=0.5,
    )

    assert steps == [0, 1]
    assert driver.sr_step == 2
    assert first.sr.info["momentum"] == 0.5
    assert second.sr.info["momentum"] == 0.5
    driver.reset_sr_state()
    assert driver.sr_step == 0


def test_torch_vmc_compile_safe_kernels_preserve_proposals_and_estimators():
    encoding = FermionSiteEncoding.vmc_torch()
    configs = torch.tensor(
        [
            [encoding.empty, encoding.double],
            [encoding.up, encoding.down],
            [encoding.up, encoding.empty],
        ],
        dtype=torch.long,
    )
    before_up, before_down = count_spinful_particles(configs, encoding=encoding)
    proposed, changed = propose_spinful_exchange_or_hopping(
        0,
        1,
        configs,
        hopping_rate=0.5,
        encoding=encoding,
        generator=torch.Generator().manual_seed(45),
        compile_kernels=True,
    )
    after_up, after_down = count_spinful_particles(proposed, encoding=encoding)

    assert torch.equal(after_up, before_up)
    assert torch.equal(after_down, before_down)
    assert torch.equal((proposed != configs).any(dim=1), changed)

    connections = heisenberg_connections(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        [(0, 1)],
    )
    configurations = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    amplitudes = torch.ones(2, dtype=torch.float64)
    eager = local_energy_from_connections(
        configurations,
        amplitudes,
        connections,
        CountingAmplitude(),
    )
    compiled = local_energy_from_connections(
        configurations,
        amplitudes,
        connections,
        CountingAmplitude(),
        compile_kernels=True,
    )
    assert torch.allclose(compiled, eager)

    driver = TorchVMCDriver(
        ProductAmplitude(),
        [(0, 1)],
        configurations,
        connection_fn="heisenberg",
        proposal="spin",
        compile_kernels=True,
        generator=torch.Generator().manual_seed(46),
    )
    result = driver.step()
    assert driver.compile_kernels is True
    assert torch.isfinite(result.energy_mean)


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


def test_torch_hamiltonian_connections_preserve_native_fermion_grading():
    fermion = Fermion(symmetry="U1U1", spinful=True, t=1.0, U=8.0)
    sites = ((0, 0), (0, 1), (1, 0), (1, 1))
    edges = (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
    )
    configs = torch.tensor(
        list(product(range(4), repeat=4)),
        dtype=torch.long,
    )

    generic = torch_hamiltonian_connections(
        configs,
        fermion.hamiltonian(edges),
        site_order=sites,
    )
    specialized = spinful_fermi_hubbard_connections(
        configs,
        ((0, 1), (0, 2), (1, 3), (2, 3)),
        t=1.0,
        U=8.0,
        encoding=FermionSiteEncoding.vmc_torch(),
        mode_order="down-up",
    )

    def as_map(connections):
        values = {}
        for eta, coeff, batch_id in zip(
            connections.configs.tolist(),
            connections.coeffs.tolist(),
            connections.batch_ids.tolist(),
        ):
            key = (int(batch_id), tuple(eta))
            values[key] = values.get(key, 0.0) + coeff
        return values

    left = as_map(generic)
    right = as_map(specialized)
    assert set(left) == set(right)
    assert all(abs(left[key] - right[key]) < 1.0e-12 for key in left)


@pytest.mark.parametrize("spinful", [False, True])
def test_torch_hamiltonian_connections_support_flat_native_fermion_terms(spinful):
    fermion = Fermion(symmetry="Z2", spinful=spinful, t=1.0, U=8.0)
    sparse = fermion.hamiltonian(((0, 2),), flat=False).terms[(0, 2)]
    flat = fermion.hamiltonian(((0, 2),), flat=True).terms[(0, 2)]
    dimension = 4 if spinful else 2
    configs = torch.tensor(
        list(product(range(dimension), repeat=3)),
        dtype=torch.long,
    )

    def as_map(operator):
        connections = torch_hamiltonian_connections(
            configs,
            {(0, 2): operator},
            site_order=(0, 1, 2),
        )
        values = {}
        for eta, coefficient, batch_id in zip(
            connections.configs.tolist(),
            connections.coeffs.tolist(),
            connections.batch_ids.tolist(),
        ):
            key = (int(batch_id), tuple(eta))
            values[key] = values.get(key, 0.0) + coefficient
        return values

    assert as_map(flat) == pytest.approx(as_map(sparse))


def test_torch_hamiltonian_connections_cache_native_compilation(monkeypatch):
    import pepsy.vmc.torch as vmc_torch

    fermion = Fermion(symmetry="U1U1", spinful=True, t=1.0, U=8.0)
    operator = fermion.hopping_operator()
    configs = torch.tensor(
        list(product(range(4), repeat=3)),
        dtype=torch.long,
    )
    vmc_torch._FERMION_COMPILED_TERM_CACHE.clear()
    original = vmc_torch._operator_dense_numpy
    calls = []

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(vmc_torch, "_operator_dense_numpy", counted)
    first = torch_hamiltonian_connections(
        configs,
        {(0, 2): operator},
        site_order=(0, 1, 2),
    )
    second = torch_hamiltonian_connections(
        configs,
        {(0, 2): operator},
        site_order=(0, 1, 2),
    )
    assert len(calls) == 1
    assert torch.equal(first.configs, second.configs)
    assert torch.equal(first.batch_ids, second.batch_ids)
    assert torch.equal(first.coeffs, second.coeffs)

    # The cutoff changes the compiled sparsity pattern and therefore gets a
    # separate cache entry rather than reusing an incompatible table.
    torch_hamiltonian_connections(
        configs,
        {(0, 2): operator},
        site_order=(0, 1, 2),
        coefficient_cutoff=1.0e-12,
    )
    assert len(calls) == 2


def test_torch_hamiltonian_connections_reject_invalid_native_fermion_sites():
    fermion = Fermion(symmetry="U1", spinful=False)
    operator = fermion.hopping_operator()
    configs = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="distinct"):
        torch_hamiltonian_connections(
            configs,
            {(1, 1): operator},
            site_order=(0, 1, 2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        torch_hamiltonian_connections(
            torch.tensor([[-1, 0, 0]], dtype=torch.long),
            {(0, 2): operator},
            site_order=(0, 1, 2),
        )


def test_torch_hamiltonian_connections_handle_odd_local_jw_prefix():
    fermion = Fermion(symmetry="U1", spinful=False)
    create = fermion.operator("create")
    configs = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)

    connections = torch_hamiltonian_connections(
        configs,
        {1: create},
        site_order=(0, 1),
    )
    values = {
        int(batch_id): complex(coefficient)
        for coefficient, batch_id in zip(connections.coeffs, connections.batch_ids)
    }
    assert values == {0: 1.0 + 0.0j, 1: -1.0 + 0.0j}


def test_torch_hamiltonian_connections_respect_reversed_native_term_order():
    fermion = Fermion(symmetry="U1", spinful=False)
    forward = fermion.operator_term(
        [(1.0, ((0, "create"), (1, "annihilate")))],
        sites=(0, 1),
    )
    reverse = fermion.operator_term(
        [(1.0, ((2, "create"), (0, "annihilate")))],
        sites=(0, 2),
    )
    configs = torch.tensor(list(product(range(2), repeat=3)), dtype=torch.long)

    def as_map(operator, where):
        connections = torch_hamiltonian_connections(
            configs,
            {where: operator},
            site_order=(0, 1, 2),
        )
        return {
            (int(batch_id), tuple(eta)): complex(coefficient)
            for eta, coefficient, batch_id in zip(
                connections.configs.tolist(),
                connections.coeffs,
                connections.batch_ids,
            )
        }

    assert as_map(forward, (2, 0)) == as_map(reverse, (0, 2))


def test_torch_fermion_vmc_supports_spinless_terms_and_sampling():
    fermion = Fermion(symmetry="U1", spinful=False, t=1.0, V=2.0)
    peps = ps_to_peps(
        1,
        2,
        fermion=fermion,
        occupations=(1, 0),
        dtype="complex128",
    )
    vmc = TorchFermionVMC(
        peps,
        fermion,
        n_walkers=2,
        dtype=torch.complex128,
        seed=19,
    )

    assert isinstance(vmc.model, TorchPEPSBoundaryAmplitude)
    assert vmc.model.contraction == "boundary"
    assert vmc.model.chi == 4
    assert isinstance(vmc.encoding, SpinlessSiteEncoding)
    assert vmc.encoding == SpinlessSiteEncoding()
    assert vmc.sector == 1
    assert vmc.proposal == "spin"
    assert torch.isfinite(vmc.local_energies()).all()
    result = vmc.sample_sweep()
    assert torch.equal(
        vmc.encoding.decode(result.configs).sum(dim=-1),
        torch.ones(2, dtype=torch.long),
    )


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
    assert result.energy_stderr_naive is not None
    assert result.effective_sample_size is not None
    assert result.chain_diagnostics is not None
    assert result.chain_diagnostics.n_chains == 2

    legacy = driver.estimate_energy(
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
    )
    assert isinstance(legacy, TorchVMCEnergyEstimate)


def test_torch_vmc_driver_shares_connected_amplitudes_across_observables():
    model = CountingAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
    )
    model.calls.clear()

    values = driver.local_observables({
        "one": {0: flip},
        "two": {0: 2.0 * flip},
    })

    assert len(model.calls) == 1
    assert torch.allclose(values["two"], 2.0 * values["one"])
    assert torch.allclose(values["one"], torch.ones(2, dtype=torch.float64))


def test_torch_vmc_driver_estimates_observables_with_shared_samples_and_profile():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
    )

    results = driver.estimate_observables(
        {"one": {0: flip}, "two": {0: 2.0 * flip}},
        n_samples=4,
        n_discard=0,
        n_thin=1,
        seed=152,
        profile=True,
    )

    assert set(results) == {"one", "two"}
    assert results["one"].configs.shape == (2, 2, 2)
    assert torch.allclose(
        results["two"].local_energies,
        2.0 * results["one"].local_energies,
    )
    profile = results["one"].profile
    assert profile["shared_observables"] == ("one", "two")
    assert profile["sampling_seconds"] >= 0.0
    assert profile["local_estimator_seconds"] >= 0.0


def test_torch_vmc_driver_measures_saved_samples_without_sampling():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
    )
    samples = driver.sample(
        n_samples=4,
        n_chains=2,
        n_discard=0,
        n_thin=1,
        seed=153,
    )

    result = driver.measure_samples(samples, profile=True)
    external = driver.measure_samples(samples.configs, amplitudes=None)
    expected = driver.local_observables(
        {"energy": {0: flip}},
        configs=samples.configs.reshape(-1, 2),
        amplitudes=samples.amplitudes.reshape(-1),
    )["energy"].reshape(2, 2)

    assert isinstance(result, TorchVMCEnergyEstimate)
    assert result.configs.shape == (2, 2, 2)
    assert result.local_energies.shape == (2, 2)
    assert result.n_samples == 4
    assert result.n_measurements == 2
    assert result.chain_diagnostics is not None
    assert torch.allclose(result.local_energies, expected)
    assert torch.allclose(external.local_energies, result.local_energies)
    assert result.profile["samples_only"] is True
    assert result.profile["sampling_seconds"] == 0.0


def test_torch_vmc_driver_measures_multiple_saved_observables_shared_targets():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
    )
    samples = driver.sample(
        n_samples=4,
        n_chains=2,
        n_discard=0,
        n_thin=1,
        seed=154,
    )

    results = driver.measure_samples(
        samples,
        observables={"one": {0: flip}, "two": {0: 2.0 * flip}},
    )

    assert set(results) == {"one", "two"}
    assert results["one"].chain_diagnostics is not None
    assert torch.allclose(
        results["two"].local_energies,
        2.0 * results["one"].local_energies,
    )


def test_torch_vmc_driver_deduplicates_saved_parent_configurations():
    model = CountingAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
    )
    saved_configs = torch.tensor(
        [
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ],
        dtype=torch.long,
    )

    model.calls.clear()
    result = driver.measure_samples(saved_configs, profile=True)
    deduplicated_first_batch = model.calls[0]

    model.calls.clear()
    driver.measure_samples(saved_configs, deduplicate=False)
    non_deduplicated_first_batch = model.calls[0]

    assert result.profile["num_samples"] == 6
    assert result.profile["num_unique_samples"] == 2
    assert deduplicated_first_batch.shape[0] == 2
    assert non_deduplicated_first_batch.shape[0] == 6


def test_torch_fermion_vmc_boundary_measurement_dedup_matches_serial():
    fermion = Fermion(symmetry="U1", spinful=False, t=1.0, V=2.0)
    peps = ps_to_peps(
        1,
        2,
        fermion=fermion,
        occupations=(1, 0),
        dtype="complex128",
    )
    vmc = TorchFermionVMC(
        peps,
        fermion,
        n_walkers=2,
        dtype=torch.complex128,
        seed=155,
    )
    saved_configs = torch.stack((vmc.configs, vmc.configs), dim=0)

    deduplicated = vmc.measure_samples(saved_configs, deduplicate=True)
    serial = vmc.measure_samples(saved_configs, deduplicate=False)

    assert vmc.model.last_amplitude_batching == "serial"
    assert torch.allclose(
        deduplicated.local_energies,
        serial.local_energies,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_torch_vmc_step_profile_reports_phase_timings():
    model = ProductAmplitude()
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        terms={0: torch.eye(2, dtype=torch.float64)},
        proposal="spin",
    )

    result = driver.step(profile=True)

    assert isinstance(result, TorchVMCStepResult)
    assert result.profile["sampling_seconds"] >= 0.0
    assert result.profile["connection_seconds"] >= 0.0
    assert result.profile["local_estimator_seconds"] >= 0.0
    assert result.profile["total_seconds"] > 0.0


def test_torch_vmc_driver_has_progress_aware_burnin_and_optimization(monkeypatch):
    class RecordingProgress:
        def __init__(self, total, desc, unit):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.updates = 0
            self.postfixes = []
            self.closed = False

        def update(self, amount):
            self.updates += amount

        def set_postfix(self, values):
            self.postfixes.append(dict(values))

        def close(self):
            self.closed = True

    bars = []

    def make_progress(progress, *, total, desc, unit=None):
        if not progress:
            return None
        bar = RecordingProgress(total, desc, unit)
        bars.append(bar)
        return bar

    monkeypatch.setattr(torch_vmc, "_make_progress", make_progress)
    driver = TorchVMCDriver(
        ProductAmplitude(),
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        connection_fn="heisenberg",
        proposal="spin",
    )

    burn_in = driver.burn_in(
        3,
        progress=True,
        track_proposal_stats=True,
    )
    history = driver.optimize(
        2,
        progress=True,
        track_proposal_stats=True,
    )
    compatibility_history = driver.run(1)

    assert burn_in.n_proposed == burn_in.n_accepted == 6
    assert burn_in.proposal_stats["exchange"]["selected"] == 6
    assert len(history) == 2
    assert len(compatibility_history) == 1
    assert all(isinstance(result, TorchVMCStepResult) for result in history)
    assert [(bar.total, bar.unit, bar.updates, bar.closed) for bar in bars] == [
        (3, "sweep", 3, True),
        (2, "step", 2, True),
    ]
    assert bars[-1].postfixes[-1]["E/site"]
    assert bars[-1].postfixes[-1]["accept"] == "1.000"
    assert bars[-1].postfixes[-1]["no-op"] == "0.000"


def test_torch_vmc_sample_sweep_aggregates_multiple_sweeps():
    driver = TorchVMCDriver(
        ProductAmplitude(),
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        connection_fn="heisenberg",
        proposal="spin",
    )

    result = driver.sample_sweep(n_sweeps=3, track_proposal_stats=True)

    assert result.n_proposed == result.n_accepted == 6
    assert result.proposal_stats["exchange"]["selected"] == 6


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


def test_torch_metropolis_sampler_drops_failed_optional_log_cache():
    class FlakyLogAmplitude:
        def __init__(self):
            self.log_calls = 0

        def __call__(self, configs):
            return torch.ones(configs.shape[0], dtype=torch.float64)

        def forward_log(self, configs):
            self.log_calls += 1
            if self.log_calls > 1:
                raise RuntimeError("unsupported proposed sector")
            return (
                torch.ones(configs.shape[0], dtype=torch.complex128),
                torch.zeros(configs.shape[0], dtype=torch.float64),
            )

    sampler = TorchMetropolisSampler(
        FlakyLogAmplitude(),
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]]),
        proposal="spin",
        seed=164,
    )
    result = sampler.sample(n_samples=2, n_discard=0, n_thin=1)

    assert result.log_abs_amplitudes is None
    assert result.amplitudes.shape == (1, 2)


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


def test_torch_observable_statistics_uses_effective_sample_size_for_stderr():
    values = torch.arange(8, dtype=torch.float64).reshape(8, 1).repeat(1, 4)
    (
        _,
        variance,
        autocorrelation_stderr,
        naive_stderr,
        effective_sample_size,
        diagnostics,
    ) = _observable_statistics(values)

    assert diagnostics is not None
    assert effective_sample_size <= values.numel()
    assert torch.allclose(
        autocorrelation_stderr,
        torch.sqrt(variance / effective_sample_size),
    )
    assert autocorrelation_stderr >= naive_stderr


def test_torch_vmc_driver_can_stop_at_target_effective_sample_size():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
        generator=torch.Generator().manual_seed(122),
    )

    result = driver.estimate_observable(
        n_measurements=6,
        sweeps_between=1,
        target_effective_sample_size=1.0,
        min_measurements=2,
        rhat_threshold=None,
        auto_thin=True,
        profile=True,
    )

    assert result.n_measurements == 2
    assert result.effective_sample_size >= 1.0
    assert result.profile["adaptive_sampling"]["stop_reason"] == (
        "target_effective_sample_size"
    )
    assert result.profile["adaptive_sampling"]["measurements_collected"] == 2


def test_torch_vmc_shared_observables_stop_when_all_reach_target_ess():
    model = ProductAmplitude()
    configs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    driver = TorchVMCDriver(
        model,
        [(0, 1)],
        configs,
        terms={0: flip},
        proposal="spin",
        generator=torch.Generator().manual_seed(123),
    )

    results = driver.estimate_observables(
        {"one": {0: flip}, "two": {0: 2.0 * flip}},
        n_measurements=6,
        sweeps_between=1,
        target_effective_sample_size=1.0,
        min_measurements=2,
        rhat_threshold=None,
        auto_thin=True,
        profile=True,
    )

    assert {result.n_measurements for result in results.values()} == {2}
    assert all(result.effective_sample_size >= 1.0 for result in results.values())
    assert all(
        result.profile["adaptive_sampling"]["stop_reason"]
        == "target_effective_sample_size"
        for result in results.values()
    )


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

    with pytest.raises(ValueError, match="target_effective_sample_size"):
        driver.estimate_observable(
            n_samples=3,
            target_effective_sample_size=2,
        )


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


def test_torch_peps_amplitude_flat_z2_stable_logs_use_vmap():
    peps = _symmray_fermionic_peps("Z2", flat=True)
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        amplitude_batching="vmap",
    )
    rows = torch.tensor([(0, 0, 0, 0), (1, 1, 1, 1)], dtype=torch.long)

    phase, log_abs = model.forward_log(rows)
    assert model.last_amplitude_batching == "log-vmap"
    amplitudes = model(rows)
    expected_phase = torch.where(
        amplitudes.abs() > 0,
        amplitudes / amplitudes.abs(),
        torch.zeros_like(amplitudes),
    )

    assert model.last_amplitude_batching == "vmap"
    assert torch.allclose(phase, expected_phase)
    assert torch.allclose(log_abs, amplitudes.abs().log())


@pytest.mark.parametrize("batching", ["auto", "vmap", "serial"])
def test_torch_peps_amplitude_explicit_batching_modes_preserve_flat_z2_values(
    batching,
):
    peps = _symmray_fermionic_peps("Z2", flat=True)
    rows = torch.tensor([(0, 0, 0, 0), (1, 1, 1, 1)], dtype=torch.long)
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        amplitude_batching=batching,
    )

    amplitudes = model(rows)
    reference = _direct_peps_amplitudes(peps, rows)

    assert torch.allclose(amplitudes, reference)
    if batching == "serial":
        assert model.last_amplitude_batching == "serial"
    else:
        assert model.last_amplitude_batching == "vmap"


def test_torch_peps_amplitude_graded_torch_projects_u1u1_under_vmap():
    peps = _symmray_fermionic_peps("U1U1")
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        graded_torch=True,
    )
    rows = torch.tensor(
        list(product(range(4), repeat=4)),
        dtype=torch.long,
    )

    amplitudes = model(rows)
    direct = _direct_peps_amplitudes(peps, rows)

    assert torch.allclose(amplitudes, direct, atol=1.0e-10, rtol=1.0e-10)
    model.zero_grad()
    amplitudes.square().sum().backward()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def test_torch_peps_amplitude_graded_torch_can_use_serial_batches():
    peps = _symmray_fermionic_peps("U1U1")
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        graded_torch=True,
        amplitude_batching="serial",
    )
    rows = torch.tensor([(2, 0, 1, 3), (0, 2, 3, 1)], dtype=torch.long)

    amplitudes = model(rows)
    direct = _direct_peps_amplitudes(peps, rows)

    assert model.last_amplitude_batching == "graded-serial"
    assert torch.allclose(amplitudes, direct, atol=1.0e-10, rtol=1.0e-10)


def test_torch_peps_amplitude_symmray_ctmrg_uses_direct_boundary_compression():
    peps = _symmray_fermionic_peps("U1U1")
    model = TorchPEPSAmplitude(
        peps,
        contraction="ctmrg",
        chi=8,
        cutoff=1.0e-10,
        dtype=torch.float64,
    )
    rows = torch.tensor([(2, 0, 1, 3)], dtype=torch.long)

    amplitudes = model(rows)

    assert model.contraction_opts["mode"] == "direct"
    assert torch.isfinite(amplitudes).all()


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


def test_torch_fermion_vmc_keeps_hamiltonian_and_observables_separate():
    fermion = Fermion(spinful=True, symmetry="U1U1", t=1.0, U=2.0)
    peps = ps_to_peps(
        2,
        2,
        fermion=fermion,
        occupations=[(1, 0), (0, 1), (1, 0), (0, 1)],
        seed=216,
        dtype="complex128",
    )
    hamiltonian = fermion.hamiltonian(
        (
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((1, 0), (1, 1)),
        )
    )
    observables = {
        "density": {(0, 0): fermion.observable("number")},
    }
    vmc = TorchFermionVMC(
        peps,
        fermion=fermion,
        hamiltonian=hamiltonian,
        observables=observables,
        n_walkers=2,
        contraction="exact",
        dtype=torch.complex128,
        seed=217,
    )

    assert vmc.hamiltonian is hamiltonian
    assert set(vmc.observables) == {"density"}

    problem = VMCProblem(
        peps=peps,
        hamiltonian=hamiltonian,
        observables=observables,
        symmetry="U1U1",
    )
    setup = build_torch_vmc(
        problem,
        fermion=fermion,
        contraction=ContractionConfig(),
        sampling=SamplingConfig(
            n_samples_per_chain=1,
            n_chains=2,
            burn_in=0,
            seed=218,
        ),
        dtype=torch.complex128,
    )
    samples = setup.sample()
    assert isinstance(setup, TorchVMCSetup)
    assert samples.chain_shape == (1, 2)
    assert samples.native is not None

    measurement = setup.measure()
    assert isinstance(measurement, VMCMeasurement)
    assert set(measurement.observables) == {"energy", "density"}
    history = setup.optimize(
        OptimizationConfig(n_steps=1, method="sgd", learning_rate=1.0e-3),
        sample_sweeps=1,
    )
    assert isinstance(history, VMCOptimizationResult)
    assert history.energies.shape == (1,)
    assert history.native[0].sr is None


@pytest.mark.parametrize("cyclic", [False, True])
def test_torch_fermion_vmc_infers_peps_pbc_and_long_range_term_graph(cyclic):
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(3)
        for y in range(3)
    }
    peps = ps_to_peps(
        3,
        3,
        fermion=fermion,
        occupations=occupations,
        cyclic=cyclic,
        seed=29,
        dtype="complex128",
    )
    encoding = FermionSiteEncoding.vmc_torch()
    configs = torch.tensor(
        [[
            encoding.up if (x + y) % 2 == 0 else encoding.down
            for x in range(3)
            for y in range(3)
        ]],
        dtype=torch.long,
    )
    terms = {
        ((0, 0), (2, 2)): fermion.hopping_operator(),
    }

    vmc = TorchFermionVMC(
        peps,
        fermion=fermion,
        terms=terms,
        configs=configs,
        contraction="exact",
        dtype=torch.complex128,
        seed=30,
    )

    assert vmc.metadata.pbc == (cyclic, cyclic)
    positions = {site: i for i, site in enumerate(vmc.site_order)}
    graph_edges = {
        frozenset((left, right)) for left, right in vmc.metadata.graph_edges
    }
    long_range_edge = frozenset((positions[(0, 0)], positions[(2, 2)]))
    assert long_range_edge in graph_edges
    wrap_edge = frozenset((positions[(0, 0)], positions[(0, 2)]))
    assert (wrap_edge in graph_edges) is cyclic


def test_torch_fermion_vmc_4x6_d4_complex_modes_sector_and_signs():
    """Check the complex Hubbard VMC path on a production-sized PEPS.

    The random PEPS is dense so the same physical state can be evaluated by
    all three Quimb contraction modes. The native ``Fermion`` object still
    supplies the U1U1 Hubbard terms, including their graded hopping signs.
    """
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=1.0,
        U=8.0,
    )
    peps = qtn.PEPS.rand(
        Lx=4,
        Ly=6,
        bond_dim=4,
        phys_dim=4,
        seed=1404,
        dtype="complex128",
    )
    configs = torch.tensor(
        [[
            2 if (x + y) % 2 == 0 else 1
            for x in range(4)
            for y in range(6)
        ]],
        dtype=torch.long,
    )
    boundary_opts = {
        "mode": "mps",
        "final_contract": True,
        "final_contract_opts": {"optimize": "auto-hq"},
        "sequence": ["xmin", "xmax", "ymin", "ymax"],
        "equalize_norms": False,
        "progbar": False,
    }
    ctmrg_opts = {
        "final_contract": False,
        "final_contract_opts": {"optimize": "auto-hq"},
        "max_separation": 1,
        "inplace": False,
        "equalize_norms": False,
        "progbar": False,
    }
    mode_opts = {
        "exact": {"contraction": "exact"},
        "boundary": {
            "contraction": "boundary",
            "chi": 64,
            "cutoff": 1.0e-10,
            "contraction_opts": boundary_opts,
        },
        "ctmrg": {
            "contraction": "ctmrg",
            "chi": 64,
            "cutoff": 1.0e-10,
            "contraction_opts": ctmrg_opts,
        },
    }

    vmcs = {
        name: TorchFermionVMC(
            peps,
            fermion,
            configs=configs,
            n_walkers=1,
            dtype=torch.complex128,
            **options,
        )
        for name, options in mode_opts.items()
    }

    exact = vmcs["exact"]
    assert exact.metadata.physical_dim == 4
    assert exact.sector == (12, 12)
    connections = exact.make_connections()
    n_up, n_down = count_spinful_particles(
        connections.configs,
        encoding=exact.encoding,
    )
    assert torch.equal(n_up, torch.full_like(n_up, 12))
    assert torch.equal(n_down, torch.full_like(n_down, 12))

    hopping_coeffs = connections.coeffs.real
    assert torch.any(torch.isclose(hopping_coeffs, hopping_coeffs.new_tensor(1.0)))
    assert torch.any(torch.isclose(hopping_coeffs, hopping_coeffs.new_tensor(-1.0)))

    energies = {
        name: vmc.local_energies()
        for name, vmc in vmcs.items()
    }
    assert all(torch.isfinite(values).all() for values in energies.values())
    assert all(torch.is_complex(values) for values in energies.values())
    for name in ("boundary", "ctmrg"):
        assert torch.allclose(
            energies[name],
            energies["exact"],
            rtol=2.0e-10,
            atol=1.0e-8,
        )


@pytest.mark.parametrize(
    ("contraction", "contraction_opts"),
    [
        (
            "boundary",
            {
                "mode": "mps",
                "final_contract": True,
                "final_contract_opts": {"optimize": "auto-hq"},
                "equalize_norms": False,
                "progbar": False,
            },
        ),
        (
            "ctmrg",
            {
                "final_contract": False,
                "max_separation": 1,
                "inplace": False,
                "equalize_norms": False,
                "progbar": False,
            },
        ),
    ],
    ids=("boundary", "ctmrg"),
)
def test_torch_fermion_vmc_approximate_contractions_support_sr(
    contraction, contraction_opts
):
    """Approximate PEPS contractions should support one finite SR update."""
    fermion = Fermion(spinful=True, symmetry="U1U1", t=1.0, U=2.0)
    occupations = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(2)
        for y in range(2)
    }
    peps = hrs_to_peps(
        (2, 2),
        fermion=fermion,
        occupations=occupations,
        chi=2,
        seed=3,
        dtype="complex128",
    )
    vmc = TorchFermionVMC(
        peps,
        fermion=fermion,
        n_walkers=4,
        init_max_attempts=256,
        contraction=contraction,
        chi=4,
        cutoff=1.0e-10,
        contraction_opts=contraction_opts,
        dtype=torch.complex128,
        seed=32,
    )

    result = vmc.step(
        sample_sweeps=1,
        sr=True,
        learning_rate=1.0e-3,
        sr_method="minsr",
        sr_parameter_mode="real-imag",
    )

    assert result.sr is not None
    assert torch.isfinite(result.energy_mean)
    assert torch.isfinite(result.sr.direction).all()
    assert torch.isfinite(vmc.amplitudes).all()
    if contraction == "boundary":
        assert vmc.model.last_amplitude_batching == "serial"


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


def test_torch_fermion_vmc_flat_z2_vmap_covers_sampling_measurement_and_sr():
    sr = pytest.importorskip("symmray")
    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        2,
        2,
        2,
        phys_dim=4,
        subsizes="equal",
        flat=True,
        seed=1,
    )
    fermion = Fermion(spinful=True, symmetry="Z2", t=1.0, U=2.0)
    vmc = TorchFermionVMC(
        peps,
        fermion,
        configs=[
            [2, 0, 1, 3],
            [0, 2, 3, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        n_walkers=4,
        contraction="exact",
        amplitude_batching="vmap",
        dtype=torch.complex128,
        seed=2,
    )

    assert vmc.model.last_amplitude_batching == "log-vmap"
    sample = vmc.sample_sweep()
    assert sample.configs.shape == (4, 4)
    assert torch.isfinite(vmc.local_energies()).all()

    result = vmc.step(
        sr=True,
        learning_rate=1.0e-4,
        sr_method="minsr",
        sr_parameter_mode="real-imag",
    )

    assert result.sr is not None
    assert torch.isfinite(result.local_energies).all()
    assert vmc.model.last_amplitude_batching == "log-vmap"


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


def test_metropolis_proposal_stats_record_movewise_noops_and_acceptance():
    encoding = FermionSiteEncoding.vmc_torch()

    def amplitude_fn(rows):
        return torch.ones(rows.shape[0], dtype=torch.float64)

    configs = torch.tensor([
        [encoding.empty, encoding.empty],
        [encoding.empty, encoding.double],
        [encoding.up, encoding.down],
        [encoding.up, encoding.up],
    ])
    result = metropolis_exchange_sweep(
        configs,
        amplitude_fn,
        [(0, 1)],
        proposal="spinful_z2",
        hopping_rate=0.4,
        spin_flip_rate=0.3,
        pair_toggle_rate=0.5,
        encoding=encoding,
        generator=torch.Generator().manual_seed(37),
        track_proposal_stats=True,
    )

    stats = result.proposal_stats
    assert sum(move["selected"] for move in stats.values()) == len(configs)
    assert sum(move["no_op"] for move in stats.values()) + sum(
        move["proposed"] for move in stats.values()
    ) == len(configs)
    assert sum(move["proposed"] for move in stats.values()) == result.n_proposed
    assert sum(move["accepted"] for move in stats.values()) == result.n_accepted
    assert result.n_accepted == result.n_proposed


def test_torch_metropolis_warmup_adapts_rates_only_between_sweeps():
    encoding = FermionSiteEncoding.vmc_torch()

    def amplitude_fn(rows):
        return torch.ones(rows.shape[0], dtype=torch.float64)

    sampler = TorchMetropolisSampler(
        amplitude_fn,
        [(0, 1)],
        torch.full((128, 2), encoding.empty, dtype=torch.long),
        proposal="spinful_z2",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=encoding,
        seed=38,
    )
    summary = sampler.warmup_proposal_mix(
        n_sweeps=4,
        adaptation_rate=1.5,
    )

    assert summary["n_sweeps"] == 4
    assert len(summary["history"]) == 4
    assert sum(
        move["selected"] for move in summary["proposal_stats"].values()
    ) == 4 * sampler.n_chains
    assert summary["rates"]["spin_flip_rate"] < 0.25
    assert summary["rates"]["pair_toggle_rate"] > 0.25
    rates_after_warmup = dict(summary["rates"])
    sampler.sample_sweep()
    assert {
        "hopping_rate": sampler.hopping_rate,
        "spin_flip_rate": sampler.spin_flip_rate,
        "pair_toggle_rate": sampler.pair_toggle_rate,
    } == rates_after_warmup


def test_torch_vmc_step_can_return_proposal_stats():
    driver = TorchVMCDriver(
        ProductAmplitude(),
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        connection_fn="heisenberg",
        proposal="spin",
    )
    result = driver.step(track_proposal_stats=True)

    assert result.proposal_stats["exchange"]["selected"] == 2
    assert sum(
        move["accepted"] for move in result.proposal_stats.values()
    ) == result.n_accepted


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
