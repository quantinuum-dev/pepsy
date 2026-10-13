"""Compiled flat-Z2 amplitudes, functional Jacobians and real-time updates."""

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sr = pytest.importorskip("symmray")

from pepsy.backends import TorchLinalgConfig
from pepsy.vmc import (TorchPEPSAmplitude, TorchTDVPDriver, TorchVMCDriver,
                       metropolis_exchange_sweep, solve_torch_tdvp,
                       torch_log_derivative_matrix)


@pytest.fixture(autouse=True)
def native_splits():
    with TorchLinalgConfig(stabilized=True, quimb_split_drivers=True,
                           qr_rank_policy="native").activated():
        yield


def even_configs(n, count=8):
    x = torch.randint(2, (count, n), generator=torch.Generator().manual_seed(22))
    x[:, -1] = x[:, :-1].sum(dim=1) % 2
    return x


@pytest.mark.integration
@pytest.mark.parametrize("contraction", ["exact", "boundary"])
def test_flat_z2_aot_compile_preserves_truncated_amplitudes_and_coordinate_jacobian(contraction):
    peps = sr.PEPS_abelian_rand("Z2", 3, 3, bond_dim=4, flat=True,
                               dtype="complex128", seed=22)
    x = even_configs(9)
    model = TorchPEPSAmplitude(peps, contraction=contraction,
        chi=2 if contraction == "boundary" else None, cutoff=0.,
        dtype=torch.complex128, amplitude_batching="vmap")
    amplitude = model(x).detach()
    log = tuple(t.detach() for t in model.forward_log(x))
    jacobian = torch_log_derivative_matrix(model, x, derivative_backend="batched",
                                           complex_parameter_mode="real-imag")
    if contraction == "boundary":
        exact = TorchPEPSAmplitude(peps, dtype=torch.complex128)(x).detach()
        assert (amplitude - exact).abs().max() > 1e-4
    model.export_and_compile(x[:4], backend="aot_eager")
    calls = []
    compiled = model._compiled_amplitude_forward

    def record(batch, *params):
        calls.append(batch.shape[0])
        return compiled(batch, *params)

    model._compiled_amplitude_forward = record
    with torch.no_grad():
        torch.testing.assert_close(model(x[:4]), amplitude[:4])
        assert model.last_amplitude_batching == "export-vmap-compile"
        compiled_log = model.forward_log(x[:4])
    assert model.last_amplitude_batching == "export-vmap-compile-log"
    for actual, expected in zip(compiled_log, log, strict=True):
        torch.testing.assert_close(actual, expected[:4])
    n_calls = len(calls)
    actual = torch_log_derivative_matrix(model, x[:7], derivative_backend="batched",
                                         complex_parameter_mode="real-imag")
    torch.testing.assert_close(actual, jacobian[:7], atol=3e-10, rtol=3e-9)
    assert calls and all(n == 4 for n in calls)
    if contraction == "boundary":
        assert len(calls) == n_calls
        with pytest.raises(RuntimeError, match="no_grad"):
            model.compiled_forward(x[:4])
    else:
        assert len(calls) > n_calls
    # Check the actively truncated complex chart against finite differences,
    # independently of either autograd implementation.
    saved = [p.detach().clone() for p in model.params]
    rng = torch.Generator().manual_seed(34)
    directions = [torch.randn(p.shape, dtype=p.dtype, generator=rng) for p in model.params]
    norm = torch.cat([d.reshape(-1) for d in directions]).norm()
    directions = [d / norm for d in directions]
    coordinate_direction = torch.cat([torch.view_as_real(d).reshape(-1) for d in directions])
    shifted = []
    for sign in (1, -1):
        with torch.no_grad():
            for p, value, delta in zip(model.params, saved, directions, strict=True):
                p.copy_(value + sign * 1e-6 * delta)
            shifted.append(model(x[:4]).detach())
    with torch.no_grad():
        for p, value in zip(model.params, saved, strict=True):
            p.copy_(value)
    numerical = (shifted[0] - shifted[1]) / (2e-6 * amplitude[:4])
    torch.testing.assert_close(actual[:4] @ coordinate_direction.to(actual.dtype), numerical,
                              atol=1e-7, rtol=1e-6)
    # Updated explicit leaves must reach the compiled functional derivative,
    # rather than reading captured initial parameters.
    with torch.no_grad():
        model.params[0].mul_(1.01 + .02j)
    compiled_new = torch_log_derivative_matrix(model, x[:7], derivative_backend="batched",
                                               complex_parameter_mode="real-imag")
    model.clear_compiled()
    eager_new = torch_log_derivative_matrix(model, x[:7], derivative_backend="batched",
                                            complex_parameter_mode="real-imag")
    torch.testing.assert_close(compiled_new, eager_new, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("contraction", ["boundary", "ctmrg"])
def test_flat_z2_zero_cutoff_values_and_derivatives_match_exact_at_sufficient_cap(contraction):
    peps = sr.PEPS_abelian_rand("Z2", 3, 3, bond_dim=4, flat=True,
                               dtype="complex128", seed=51)
    x = even_configs(9, count=3)
    exact = TorchPEPSAmplitude(peps, dtype=torch.complex128, amplitude_batching="vmap")
    model = TorchPEPSAmplitude(peps, contraction=contraction, chi=16, cutoff=0.,
                               dtype=torch.complex128, amplitude_batching="vmap")
    torch.testing.assert_close(model(x), exact(x), atol=1e-9, rtol=1e-10)
    actual = torch_log_derivative_matrix(model, x, derivative_backend="batched",
                                         complex_parameter_mode="real-imag")
    expected = torch_log_derivative_matrix(exact, x, derivative_backend="batched",
                                           complex_parameter_mode="real-imag")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)


def test_real_coordinate_tdvp_force_projects_after_real_time_rotation():
    rng = torch.Generator().manual_seed(83)
    o = torch.randn(8, 5, dtype=torch.complex128, generator=rng)
    e = torch.randn(8, dtype=torch.complex128, generator=rng)
    c, h = o - o.mean(dim=0), e - e.mean()
    metric = (c.mH @ c).real / 8
    force = (c.mH @ h).imag / 8
    expected = torch.linalg.solve(metric + 1e-8 * torch.eye(5), force)
    for method in ("direct", "minsr"):
        result = solve_torch_tdvp(o, e, method=method, parameter_mode="real-imag")
        torch.testing.assert_close(result.velocity, expected)
        assert not torch.is_complex(result.velocity)


@pytest.mark.integration
def test_compiled_flat_z2_midpoint_keeps_charge_blocks_and_matches_eager():
    peps = sr.PEPS_abelian_rand("Z2", 2, 2, bond_dim=4, flat=True,
                               dtype="complex128", seed=27)
    x = torch.tensor([c for c in itertools.product((0, 1), repeat=4) if sum(c) % 2 == 0])
    z, pauli_x = torch.diag(torch.tensor([1., -1.])), torch.tensor([[0., 1.], [1., 0.]])
    # This is TFIM in the rotated basis: H=-sum XX-sum Z preserves Z2.
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    terms = {i: -z for i in range(4)}
    terms.update({edge: -torch.kron(pauli_x, pauli_x).reshape(2, 2, 2, 2) for edge in edges})
    models = []
    results = []
    rows = x.tolist()
    row_indices = {tuple(row): i for i, row in enumerate(rows)}
    hamiltonian = torch.zeros((8, 8), dtype=torch.complex128)
    for i, row in enumerate(rows):
        hamiltonian[i, i] = -sum(1 - 2 * bit for bit in row)
        for a, b in edges:
            flipped = row.copy()
            flipped[a], flipped[b] = 1 - flipped[a], 1 - flipped[b]
            hamiltonian[i, row_indices[tuple(flipped)]] -= 1
    for compile_graph in (False, True):
        model = TorchPEPSAmplitude(peps, contraction="boundary", chi=16, cutoff=0.,
                                   dtype=torch.complex128, amplitude_batching="vmap")
        if compile_graph:
            model.export_and_compile(x[:4], backend="aot_eager")
        vmc = TorchVMCDriver(model, edges, x, terms=terms, proposal="parity")
        tdvp = TorchTDVPDriver(vmc, dt=.001, method="minsr", parameter_mode="real-imag",
                              gauge_projection=False, derivative_backend="batched")
        initial = model(x).detach()
        results.append(tdvp.step(samples=x, proposal_log_probs=torch.zeros(8)))
        exact = torch.matrix_exp(-1j * hamiltonian * .001) @ initial
        final = model(x).detach()
        fidelity = torch.vdot(exact, final).abs().square() / (exact.norm() * final.norm()).square()
        assert float(fidelity) > 1 - 1e-10
        models.append(model)
    for a, b in zip(models[0].params, models[1].params, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    assert results[1].time == pytest.approx(.001)
    assert results[1].stages[-1].info["tdvp_relative_residual"] < 1e-6
    reconstructed = models[1].to_peps()
    assert reconstructed.max_bond() == 4
    assert all(isinstance(t.data, sr.Z2ArrayFlat) for t in reconstructed)
    odd = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 1]])
    assert torch.count_nonzero(models[1](odd)) == 0
    np.testing.assert_array_equal([t.data.charge for t in reconstructed], [t.data.charge for t in peps])


def test_flat_z2_ground_state_sr_updates_lower_exact_energy():
    peps = sr.PEPS_abelian_rand("Z2", 2, 2, bond_dim=4, flat=True,
                               dtype="complex128", seed=27)
    x = torch.tensor([c for c in itertools.product((0, 1), repeat=4) if sum(c) % 2 == 0])
    z, pauli_x = torch.diag(torch.tensor([1., -1.])), torch.tensor([[0., 1.], [1., 0.]])
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    terms = {i: -z for i in range(4)}
    terms.update({edge: -torch.kron(pauli_x, pauli_x).reshape(2, 2, 2, 2) for edge in edges})
    model = TorchPEPSAmplitude(peps, contraction="boundary", chi=16, cutoff=0.,
                               dtype=torch.complex128, amplitude_batching="vmap")
    vmc = TorchVMCDriver(model, edges, x, terms=terms, proposal="parity")
    rows = x.tolist()
    row_indices = {tuple(row): i for i, row in enumerate(rows)}
    hamiltonian = torch.zeros((8, 8), dtype=torch.complex128)
    for i, row in enumerate(rows):
        hamiltonian[i, i] = -sum(1 - 2 * bit for bit in row)
        for a, b in edges:
            flipped = row.copy()
            flipped[a], flipped[b] = 1 - flipped[a], 1 - flipped[b]
            hamiltonian[i, row_indices[tuple(flipped)]] -= 1

    def energy():
        state = model(x).detach()
        return float((torch.vdot(state, hamiltonian @ state) / torch.vdot(state, state)).real)

    energies = [energy()]
    for _ in range(5):
        vmc.step(samples=x, proposal_log_probs=torch.zeros(8), sr=True,
                 sr_parameter_mode="real-imag", sr_method="minsr", learning_rate=.01)
        energies.append(energy())
    assert all(after < before for before, after in zip(energies, energies[1:]))
    assert energies[-1] < energies[0] - .1
    assert model.to_peps().max_bond() == 4


def test_binary_pair_proposals_preserve_charge_and_explore_entire_sector():
    x = torch.zeros((2048, 4), dtype=torch.long)
    edges = [(0, 1), (1, 2), (2, 3)]
    result = metropolis_exchange_sweep(x, lambda x: torch.ones(x.shape[0]), edges,
        proposal="parity", generator=torch.Generator().manual_seed(17))
    assert (result.configs.sum(dim=1) % 2 == 0).all()
    assert torch.unique(result.configs, dim=0).shape[0] == 8


def test_export_rejects_data_dependent_truncation():
    peps = sr.PEPS_abelian_rand("Z2", 2, 2, bond_dim=4, flat=True, seed=7)
    model = TorchPEPSAmplitude(peps, contraction="boundary", chi=4, cutoff=1e-10)
    with pytest.raises(ValueError, match="cutoff=0"):
        model.export_and_compile(even_configs(4))
