"""Fixed-point FIT gradients and torch.compile compatibility."""

import pytest
import quimb.tensor as qtn

torch = pytest.importorskip("torch")

from pepsy.backends import backend_torch
from pepsy.fitting import FIT
from pepsy.optimizers.mps import MpsOptimizer


pytestmark = [pytest.mark.optional, pytest.mark.integration, pytest.mark.mps]


def _torch_mps():
    state = qtn.MPS_rand_state(3, 2, dtype="complex128", seed=23)
    state.apply_to_arrays(
        backend_torch(dtype=torch.complex128, requires_grad=True)
    )
    return state


def test_torch_implicit_fit_gradient_and_compile_match_finite_difference():
    state = _torch_mps()
    fit = FIT(state.copy(), p=state, contraction_opt="greedy").as_functional(
        backend="torch", max_iterations=30,
        tolerance=1e-11, adjoint_tolerance=1e-10,
    )
    compiled_fit = torch.compile(fit)
    base = fit.target_params

    def loss(scale, call):
        params = {key: value * scale for key, value in base.items()}
        result = fit.to_network(call(params))
        return torch.real((result.H & result).contract(all, optimize="greedy"))

    scale = torch.tensor(1.02, dtype=torch.float64, requires_grad=True)
    value = loss(scale, compiled_fit)
    derivative, = torch.autograd.grad(value, scale)
    delta = 1e-5
    upper = loss(torch.tensor(1.02 + delta, dtype=torch.float64), fit)
    lower = loss(torch.tensor(1.02 - delta, dtype=torch.float64), fit)
    finite_difference = ((upper - lower) / (2 * delta)).squeeze()
    torch.testing.assert_close(derivative, finite_difference, rtol=2e-7, atol=2e-8)
    assert fit.converged
    assert fit.relative_residual <= fit._tolerance


def _dmrg2_gate_loss(angle, *, requires_grad):
    state = qtn.MPS_computational_state("+++", dtype="complex128")
    state.apply_to_arrays(
        backend_torch(dtype=torch.complex128, requires_grad=True)
    )
    theta = torch.tensor(angle, dtype=torch.float64, requires_grad=requires_grad)
    phase = torch.exp(-0.5j * theta)
    gate = torch.diag(
        torch.stack([phase, phase.conj(), phase.conj(), phase])
    )
    optimizer = MpsOptimizer(
        state, [(gate, (0, 1))], chi=2, mode="dmrg2",
        fit_differentiation="implicit",
    )
    optimizer.run(cutoff=0.0, progbar=False)
    loss = optimizer.p.to_dense()[0].real
    return (loss, torch.autograd.grad(loss, theta)[0], optimizer) if requires_grad else loss


def test_mps_optimizer_dmrg2_implicit_gradient_matches_finite_difference():
    value, derivative, optimizer = _dmrg2_gate_loss(0.17, requires_grad=True)
    delta = 1e-5
    upper = _dmrg2_gate_loss(0.17 + delta, requires_grad=False)
    lower = _dmrg2_gate_loss(0.17 - delta, requires_grad=False)
    finite_difference = ((upper - lower) / (2 * delta)).squeeze()
    torch.testing.assert_close(derivative, finite_difference, rtol=2e-6, atol=2e-8)
    assert optimizer.get_fit_diagnostics()["convergence_reason"] == "implicit_fixed_point"


def test_local_window_implicit_vjp_includes_external_guess_dependencies():
    target = qtn.MPS_rand_state(3, 2, dtype="complex128", seed=41)
    target.apply_to_arrays(backend_torch(dtype=torch.complex128))
    guess = qtn.MPS_rand_state(3, 2, dtype="complex128", seed=42)
    guess.apply_to_arrays(backend_torch(dtype=torch.complex128))
    fit = FIT(
        target, p=guess, range_int=(1, 2), contraction_opt="greedy"
    ).as_functional(
        backend="torch", max_iterations=60, tolerance=1e-11,
        adjoint_tolerance=1e-10,
    )
    outside_key = fit._guess_site_keys[0]
    active_key = fit._guess_site_keys[1]

    def loss(scale):
        parameters = dict(fit.initial_params)
        parameters[outside_key] = parameters[outside_key] * scale
        return fit(target_params=fit.target_params, guess_params=parameters)[
            active_key
        ].real.sum()

    scale = torch.tensor(1.03, dtype=torch.float64, requires_grad=True)
    derivative, = torch.autograd.grad(loss(scale), scale)
    delta = 1e-5
    upper = loss(torch.tensor(1.03 + delta, dtype=torch.float64))
    lower = loss(torch.tensor(1.03 - delta, dtype=torch.float64))
    finite_difference = ((upper - lower) / (2 * delta)).squeeze()
    torch.testing.assert_close(derivative, finite_difference, rtol=5e-5, atol=2e-7)


def _second_order_3x3_trotter_loss(dt, *, requires_grad):
    state = qtn.MPS_computational_state("+" * 9, dtype="complex128")
    state.apply_to_arrays(
        backend_torch(dtype=torch.complex128, requires_grad=True)
    )
    time = torch.tensor(dt, dtype=torch.float64, requires_grad=requires_grad)
    identity = torch.eye(2, dtype=torch.complex128)
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    zz = torch.tensor([1, -1, -1, 1], dtype=torch.float64)
    rx_angle = 0.7 * time
    rx = (
        torch.cos(rx_angle / 2) * identity
        - 1j * torch.sin(rx_angle / 2) * pauli_x
    )
    rzz = torch.diag(
        torch.exp(-0.4j * time * zz).to(torch.complex128)
    )
    edges = [
        (3 * row + col, 3 * row + col + 1)
        for row in range(3)
        for col in range(2)
    ] + [
        (3 * row + col, 3 * (row + 1) + col)
        for row in range(2)
        for col in range(3)
    ]
    # Symmetric Strang step: exp(-i dt Hx/2) exp(-i dt Hzz) exp(-i dt Hx/2).
    gates = (
        [(rx, (site,)) for site in range(9)]
        + [(rzz, edge) for edge in edges]
        + [(rx, (site,)) for site in range(9)]
    )
    optimizer = MpsOptimizer(
        state, gates, chi=4, mode="dmrg2", fit_differentiation="implicit"
    )
    optimizer.run(cutoff=0.0, progbar=False)
    loss = optimizer.p.to_dense()[0].real
    if requires_grad:
        gradient, = torch.autograd.grad(loss, time)
        return loss, gradient
    return loss


def test_mps_optimizer_3x3_second_order_trotter_gradient_matches_finite_difference():
    value, gradient = _second_order_3x3_trotter_loss(0.13, requires_grad=True)
    delta = 2e-5
    upper = _second_order_3x3_trotter_loss(0.13 + delta, requires_grad=False)
    lower = _second_order_3x3_trotter_loss(0.13 - delta, requires_grad=False)
    finite_difference = ((upper - lower) / (2 * delta)).squeeze()
    torch.testing.assert_close(gradient, finite_difference, rtol=5e-6, atol=2e-7)
    assert torch.isfinite(value)
    assert torch.isfinite(gradient)
