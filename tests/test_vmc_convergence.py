"""Regression tests for native Torch VMC convergence diagnostics."""

import pytest


def test_chain_diagnostics_can_return_split_rhat_and_max_tau():
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import torch_chain_diagnostics

    values = torch.cat(
        (
            torch.zeros((8, 2), dtype=torch.float64),
            torch.full((8, 2), 10.0, dtype=torch.float64),
        ),
        dim=0,
    )
    diagnostics = torch_chain_diagnostics(values, split_rhat=True)

    assert diagnostics.split_r_hat is not None
    assert torch.isfinite(diagnostics.max_integrated_autocorrelation_time)
    assert diagnostics.max_integrated_autocorrelation_time >= 1.0


def test_convergence_check_uses_temporary_chains_and_rng_state():
    torch = pytest.importorskip("torch")
    from pepsy.vmc import TorchVMCDriver

    class ProductAmplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(
                torch.tensor([1.0, 2.0], dtype=torch.float64)
            )

        def forward(self, configs):
            return self.weights[configs].prod(dim=1)

    generator = torch.Generator().manual_seed(23)
    driver = TorchVMCDriver(
        ProductAmplitude(),
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        terms={0: torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
        proposal="spin",
        generator=generator,
    )
    configs_before = driver.configs.clone()
    rng_before = driver.generator.get_state().clone()

    report = driver.check_mc_convergence(
        min_chain_length=4,
        max_chain_length=6,
        target_effective_samples_per_chain=1.0,
        seed=7,
    )

    assert report.n_samples_per_chain >= 4
    assert report.energy is not None
    assert report.energy.split_r_hat is not None
    assert torch.equal(driver.configs, configs_before)
    assert torch.equal(driver.generator.get_state(), rng_before)
