"""Real-time sign, gauge generators, integration and PEPO-to-tVMC handoff."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import quimb.tensor as qtn
from scipy.linalg import expm

torch = pytest.importorskip("torch")

from pepsy.vmc import (TorchPEPSAmplitude, TorchTDVPDriver, TorchVMCDriver,
                       metropolis_exchange_sweep, solve_torch_tdvp,
                       torch_log_derivative_matrix, torch_peps_gauge_vectors)


def configs(n):
    return torch.tensor(list(itertools.product((0, 1), repeat=n)), dtype=torch.long)


def test_uncentered_minsr_solves_schrodinger_rows_and_removes_gauge():
    rng = torch.Generator().manual_seed(21)
    o = torch.randn(4, 7, dtype=torch.complex128, generator=rng)
    e = torch.randn(4, dtype=torch.complex128, generator=rng)
    result = solve_torch_tdvp(o, e, method="minsr", diag_shift=0)
    torch.testing.assert_close(o @ result.velocity, -1j * e)
    torch.testing.assert_close(result.velocity, torch.linalg.pinv(o) @ (-1j * e))
    assert result.info["tdvp_relative_residual"] < 1e-12
    assert result.info["solver"] == "cholesky"
    torch.testing.assert_close(result.energy_mean, e.mean())


def test_minsr_duplicate_samples_use_pseudoinverse():
    o = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.complex128)
    result = solve_torch_tdvp(o, torch.ones(2), method="minsr", diag_shift=0)
    torch.testing.assert_close(o @ result.velocity, -1j * torch.ones(2, dtype=o.dtype))
    assert result.info["solver"] == "pinv"


@pytest.mark.parametrize("parameter_mode", ["holomorphic", "real-imag"])
def test_solved_normal_equation_can_leave_unit_tangent_residual(parameter_mode):
    # Energy is orthogonal to the entire sampled tangent span. The TDVP
    # normal equation is exactly solved by zero velocity, yet misses H|psi>.
    o = torch.tensor([[-1.], [0.], [1.]], dtype=torch.complex128)
    energy = torch.tensor([1., -2., 1.], dtype=torch.complex128)
    result = solve_torch_tdvp(o, energy, method="direct", diag_shift=0,
                              parameter_mode=parameter_mode)
    assert result.info["tdvp_relative_residual"] == 0
    assert result.info["tdvp_relative_residual_squared"] == 0
    assert result.info["tangent_relative_residual"] == pytest.approx(1.)
    assert result.info["tangent_residual_norm"] == pytest.approx(np.sqrt(2.))
    assert result.info["tangent_rhs_norm"] == pytest.approx(np.sqrt(2.))


@pytest.mark.parametrize("parameter_mode", ["holomorphic", "real-imag"])
@pytest.mark.parametrize("method", ["direct", "minsr"])
def test_tdvp_residual_diagnostics_match_weighted_wavefunction_equation(parameter_mode, method):
    rng = torch.Generator().manual_seed(36)
    o = torch.randn(4, 7, dtype=torch.complex128, generator=rng)
    energy = torch.randn(4, dtype=torch.complex128, generator=rng)
    weights = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    result = solve_torch_tdvp(o, energy, sample_weights=weights, method=method,
                              diag_shift=.01, parameter_mode=parameter_mode)
    c = o - (weights[:, None] * o).sum(dim=0)
    centered_energy = energy - (weights * energy).sum()
    tangent = c @ result.velocity.to(c.dtype) + 1j * centered_energy
    expected = (weights * tangent.abs().square()).sum().sqrt()
    denominator = (weights * centered_energy.abs().square()).sum().sqrt()
    assert result.info["tangent_residual_norm"] == pytest.approx(float(expected))
    assert result.info["tangent_relative_residual"] == pytest.approx(float(expected / denominator))
    assert result.info["tdvp_relative_residual_squared"] == pytest.approx(
        result.info["tdvp_relative_residual"] ** 2)


def test_constant_energy_has_zero_centered_residual_diagnostics():
    result = solve_torch_tdvp(torch.eye(2, dtype=torch.complex128),
                              torch.full((2,), 3., dtype=torch.complex128),
                              method="direct", diag_shift=1e-8)
    for name in ("tdvp_relative_residual", "tdvp_relative_residual_squared",
                 "tangent_relative_residual", "tangent_residual_norm", "tangent_rhs_norm"):
        assert result.info[name] == 0


def test_analytic_peps_gauges_annihilate_centered_jacobian():
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=12)
    model = TorchPEPSAmplitude(peps, dtype=torch.complex128,
                              contraction_opts={"optimize": "greedy"})
    o = torch_log_derivative_matrix(model, configs(4))
    vectors = torch_peps_gauge_vectors(model)
    torch.testing.assert_close((o - o.mean(dim=0)) @ vectors,
                              torch.zeros((16, 17), dtype=o.dtype), atol=3e-14, rtol=0)
    energies = torch.arange(16).to(o.dtype)
    projected = solve_torch_tdvp(o, energies, null_vectors=vectors, method="direct")
    plain = solve_torch_tdvp(o, energies, method="direct")
    assert projected.info["gauge_rank"] == 16
    assert projected.info["matrix_shape"] == (16, 16)
    torch.testing.assert_close(o @ projected.velocity, o @ plain.velocity,
                              atol=3e-7, rtol=1e-7)
    torch.testing.assert_close(vectors.mH @ projected.velocity,
                              torch.zeros(17, dtype=o.dtype), atol=1e-13, rtol=0)


def product_driver():
    class Amplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(torch.tensor([1, 1j], dtype=torch.complex128))

        def forward(self, x):
            return self.state[x[:, 0]]

    model = Amplitude()
    h = torch.tensor([[.3, -1.], [-1., -.3]], dtype=torch.complex128)
    vmc = TorchVMCDriver(model, [], configs(1), terms={0: h}, proposal="local")
    return model, h, vmc


@pytest.mark.parametrize("parameter_mode", ["holomorphic", "real-imag"])
def test_midpoint_real_time_direction_and_second_order_accuracy(parameter_mode):
    errors = []
    for dt in (.02, .01):
        model, h, vmc = product_driver()
        initial = model.state.detach().clone()
        driver = TorchTDVPDriver(vmc, dt=dt, method="direct", diag_shift=1e-12,
                                parameter_mode=parameter_mode)
        for _ in range(round(.1 / dt)):
            result = driver.step(samples=configs(1), proposal_log_probs=torch.zeros(2))
        assert result.time == pytest.approx(.1)
        assert len(result.stages) == 2
        exact = torch.matrix_exp(-1j * h * .1) @ initial
        a = model.state.detach() / model.state.detach().norm()
        b = exact / exact.norm()
        overlap = torch.vdot(b, a)
        errors.append(float((a - b * overlap / overlap.abs()).norm()))
    assert errors[0] < 1e-4
    assert errors[1] < errors[0] / 3.5


def test_failed_midpoint_stage_restores_parameters_and_time(monkeypatch):
    model, _, vmc = product_driver()
    before = model.state.detach().clone()
    driver = TorchTDVPDriver(vmc, dt=.01)
    original = driver._stage
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stage failure")
        return original(**kwargs)

    monkeypatch.setattr(driver, "_stage", fail_second)
    with pytest.raises(RuntimeError, match="stage failure"):
        driver.step(samples=configs(1), proposal_log_probs=torch.zeros(2))
    torch.testing.assert_close(model.state, before)
    torch.testing.assert_close(vmc.amplitudes, model(vmc.configs))
    assert driver.time == 0


def test_local_spin_proposal_explores_all_sectors_without_uniform_periodicity():
    x = torch.zeros((1024, 4), dtype=torch.long)
    result = metropolis_exchange_sweep(x, lambda x: torch.ones(x.shape[0]), [],
        proposal="local", generator=torch.Generator().manual_seed(24), track_proposal_stats=True)
    assert len(torch.unique(result.configs, dim=0)) == 16
    assert 0 < int(result.configs.sum()) < x.numel()
    assert result.n_accepted == result.n_proposed
    assert result.proposal_stats["spin_flip"]["accepted"] == result.n_accepted
    with pytest.raises(ValueError, match="binary"):
        metropolis_exchange_sweep(torch.full((1, 2), 2), lambda x: torch.ones(x.shape[0]),
                                  [], proposal="local")


@pytest.mark.parametrize("method", ["direct", "minsr"])
def test_pepo_startup_then_fixed_d_tdvp_matches_dense_evolution(method):
    from pepsy.optimizers import PepsOptimizer
    from pepsy.tensors import ps_to_peps

    spec = importlib.util.spec_from_file_location("tdvp_pepo_example",
        Path(__file__).parents[1] / "examples" / "tfim_pepo.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    fit = PepsOptimizer(ps_to_peps(2, 2, theta=np.pi / 4), [], chi=2,
        mode="sweep", optimizer="als", boundary_chi=(32, 128),
        normalize_chi=32, evaluation_chi=(32, 128), contraction_opt="greedy",
        sweep_kwargs={"local_contraction_opt": "greedy"})
    for _ in range(2):
        fit.run_pepo(example.trotter_pepo(2, 2, .05), unitary=True)
    assert fit.state.max_bond() == 2
    model = TorchPEPSAmplitude(fit.state, dtype=torch.complex128,
                               contraction_opts={"optimize": "greedy"})
    initial = model(configs(4)).detach().numpy()
    shapes = [tuple(p.shape) for p in model.parameters()]
    tags = [tuple(t.tags) for t in model.to_peps().tensors]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    vmc = TorchVMCDriver(model, edges, configs(4), connection_fn="transverse_ising",
                         connection_kwargs={"J": -4., "h": -2.}, proposal="local")
    driver = TorchTDVPDriver(vmc, dt=.002, t0=.1, method=method, diag_shift=1e-10)
    h = np.zeros((16, 16))
    for i, row in enumerate(configs(4).numpy()):
        h[i, i] = -sum((1 - 2 * row[a]) * (1 - 2 * row[b]) for a, b in edges)
        for site in range(4):
            h[i, i ^ (1 << (3 - site))] = -1
    for _ in range(3):
        result = driver.step(samples=configs(4), proposal_log_probs=torch.zeros(16))
    exact = expm(-1j * .006 * h) @ initial
    final = model(configs(4)).detach().numpy()
    fidelity = abs(np.vdot(exact, final))**2 / (np.linalg.norm(exact) * np.linalg.norm(final))**2
    assert fidelity > 1 - 1e-9
    assert result.time == pytest.approx(.106)
    assert model.to_peps().max_bond() == 2
    assert [tuple(p.shape) for p in model.parameters()] == shapes
    assert [tuple(t.tags) for t in model.to_peps().tensors] == tags


def test_duplicate_configs_share_jacobian_rows_and_preserve_weights(monkeypatch):
    from pepsy.vmc.torch import tdvp

    model, _, vmc = product_driver()
    o = torch_log_derivative_matrix(model, configs(1))
    estimate = vmc.step(samples=torch.tensor([[0], [0], [1]]),
                         weights=torch.tensor([.1, .2, .7], dtype=torch.float64))
    expected = solve_torch_tdvp(o, estimate.local_energies[[0, 2]],
                                sample_weights=torch.tensor([.3, .7]), method="direct")
    derivative = tdvp.torch_log_derivative_matrix
    sizes = []

    def counted(model, x, **kwargs):
        sizes.append(x.shape[0])
        return derivative(model, x, **kwargs)

    monkeypatch.setattr(tdvp, "torch_log_derivative_matrix", counted)
    driver = TorchTDVPDriver(vmc, dt=.01, integrator="euler", method="direct")
    step = driver.step(samples=estimate.configs, weights=estimate.importance_weights)
    torch.testing.assert_close(step.stages[0].velocity, expected.velocity,
                              atol=1e-7, rtol=1e-7)
    assert sizes == [2]
    assert step.stages[0].info["n_unique_configs"] == 2
    assert step.stages[0].info["n_samples"] == 3


@pytest.mark.parametrize("kwargs, message", [
    ({"method": "invalid"}, "method"),
    ({"diag_shift": -1}, "diag_shift"),
    ({"pinv_rtol": -1}, "pinv_rtol"),
    ({"sample_weights": [-1, 2]}, "non-negative"),
    ({"method": "minsr", "null_vectors": torch.eye(2)}, "direct"),
])
def test_tdvp_rejects_invalid_solve_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        solve_torch_tdvp(torch.eye(2), torch.ones(2), **kwargs)


def test_midpoint_rejects_fixed_weights_and_real_model():
    model, _, vmc = product_driver()
    driver = TorchTDVPDriver(vmc, dt=.01)
    with pytest.raises(ValueError, match="fixed weights"):
        driver.step(samples=configs(1), weights=torch.ones(2))
    with pytest.raises(ValueError, match="proposal_log_probs"):
        driver.step(samples=configs(1))
    model.state = torch.nn.Parameter(model.state.detach().real)
    with pytest.raises(ValueError, match="complex parameters"):
        TorchTDVPDriver(vmc, dt=.01)


@pytest.mark.parametrize("contraction", ["boundary", "hotrg", "ctmrg"])
def test_tdvp_compressed_peps_require_real_coordinates(contraction):
    model = TorchPEPSAmplitude(qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=19),
                               dtype=torch.complex128, contraction=contraction, chi=1, cutoff=0.)
    vmc = TorchVMCDriver(model, [], configs(4), terms={0: torch.eye(2)}, proposal="local")
    with pytest.raises(ValueError, match="requires parameter_mode='real-imag'"):
        TorchTDVPDriver(vmc, dt=.001)
    TorchTDVPDriver(vmc, dt=.001, parameter_mode="real-imag", gauge_projection=False)
