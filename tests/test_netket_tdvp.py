"""PEPS coordinate derivatives and native NetKet time evolution against ED."""

from types import SimpleNamespace
from dataclasses import replace
from functools import partial
import itertools
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import quimb.tensor as qtn
from scipy.sparse.linalg import expm_multiply

from pepsy.vmc import (build_ising_vmc, make_netket_tdvp_driver,
                       make_peps_log_amplitude_model, pack_peps_ansatz)
from pepsy.vmc.netket import (_netket_variables_from_ansatz,
                              _peps_params_from_netket_variables)


def require_netket():
    nk = pytest.importorskip("netket")
    if not hasattr(nk, "experimental"):
        pytest.skip("A working NetKet installation is required, not a namespace directory")
    import jax
    # NetKet uses a named sample mesh. Each test owns its mesh because the
    # repository fixture restores the preceding mesh after optional imports.
    jax.sharding.set_mesh(jax.make_mesh((len(jax.devices()),), ("S",),
                                      axis_types=(jax.sharding.AxisType.Auto,)))
    return nk


@pytest.mark.parametrize("contraction", ["exact", "boundary", "hotrg", "ctmrg"])
@pytest.mark.parametrize("storage", ["dense", "flat-z2"])
def test_real_imag_model_preserves_values_and_directional_derivative(contraction, storage):
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    jax.config.update("jax_enable_x64", True)
    from jax.flatten_util import ravel_pytree

    if storage == "flat-z2":
        sr = pytest.importorskip("symmray")
        peps = sr.PEPS_abelian_rand("Z2", 3, 3, bond_dim=4, flat=True,
                                   dtype="complex128", seed=82)
    else:
        peps = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=82)
    ansatz = pack_peps_ansatz(peps)
    # Flat Z2 retains both charge sectors, so its smallest equal-sector cap
    # is two. A cap of one asks upstream to construct a zero-width block.
    small_chi = 2 if storage == "flat-z2" else 1
    options = dict(contraction=contraction, chi=small_chi if contraction != "exact" else None,
                   cutoff=0., contraction_opts={"optimize": "greedy"} if contraction == "exact" else {})
    real_model = make_peps_log_amplitude_model(ansatz, parameter_mode="real-imag", **options)
    complex_model = make_peps_log_amplitude_model(ansatz, **options)
    real_variables = _netket_variables_from_ansatz(ansatz, parameter_mode="real-imag")
    complex_variables = _netket_variables_from_ansatz(ansatz)
    x = np.random.default_rng(8).choice([-1, 1], size=(2, 9))
    if storage == "flat-z2":
        x[:, -1] = np.prod(x[:, :-1], axis=1)
    evaluate = jax.jit(lambda params: real_model.apply({"params": params}, x))
    actual = evaluate(real_variables["params"])
    np.testing.assert_allclose(actual, complex_model.apply(complex_variables, x), atol=1e-10)
    if contraction != "exact":
        exact_model = make_peps_log_amplitude_model(ansatz, contraction="exact")
        assert np.max(np.abs(np.exp(actual) - np.exp(exact_model.apply(complex_variables, x)))) > 1e-6
    reconstructed = _peps_params_from_netket_variables(ansatz, real_variables)
    for actual_leaf, original in zip(jax.tree_util.tree_leaves(reconstructed), ansatz.leaves, strict=True):
        np.testing.assert_array_equal(actual_leaf, original)
    theta, unpack = ravel_pytree(real_variables["params"])
    direction = np.random.default_rng(28).normal(size=theta.shape)
    direction /= np.linalg.norm(direction)
    jacobian = jax.jacrev(lambda z: jax.numpy.stack([
        evaluate(unpack(z)).real, evaluate(unpack(z)).imag]))(theta)
    prediction = (jacobian[0] + 1j * jacobian[1]) @ direction
    delta = 1e-5
    numerical = (evaluate(unpack(theta + delta * direction))
                 - evaluate(unpack(theta - delta * direction))) / (2 * delta)
    np.testing.assert_allclose(prediction, numerical, atol=1e-7, rtol=2e-6)
    assert all(not np.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(real_variables))


def make_setup(*, contraction="exact", parameter_mode="real-imag", samples=512, d=2):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    require_netket()
    peps = qtn.PEPS.rand(2, 2, bond_dim=d, dtype="complex128", seed=24)
    return build_ising_vmc(peps, Lx=2, Ly=2, h=.7, J=-1.,
        parameter_mode=parameter_mode, contraction=contraction,
        chi=4 if contraction == "boundary" else None, cutoff=0.,
        contraction_opts={"optimize": "greedy"} if contraction == "exact" else {},
        n_samples=samples, n_chains=16, n_discard_per_chain=8,
        chunk_size=32, use_sr=False, seed=44, sampler_seed=45)


@pytest.mark.parametrize("storage", ["dense", "flat-z2"])
@pytest.mark.parametrize("contraction", ["boundary", "hotrg", "ctmrg"])
@pytest.mark.parametrize("jit", [True, False])
def test_scaled_log_amplitudes_preserve_extremely_small_global_norm(storage, contraction, jit):
    from pepsy.vmc import make_peps_batched_amplitude_function
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    if storage == "flat-z2":
        sr = pytest.importorskip("symmray")
        peps = sr.PEPS_abelian_rand("Z2", 3, 3, bond_dim=4, flat=True,
                                  dtype="complex128", seed=82)
    else:
        peps = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=82)
    ansatz = pack_peps_ansatz(peps)
    kwargs = dict(contraction=contraction, chi=2 if storage == "flat-z2" else 1, cutoff=0., jit=jit)
    log_amplitude = make_peps_batched_amplitude_function(ansatz, output="log", **kwargs)
    scaled_amplitude = make_peps_batched_amplitude_function(ansatz, output="mantissa_exponent", **kwargs)
    rows = np.ones((1, 9), dtype=int)
    baseline = np.asarray(log_amplitude(rows))
    # The complete amplitude scales below floating-point range; individual
    # nonzero tensor norms are still representable and must be stripped first.
    scaled_params = jax.tree_util.tree_map(lambda z: jax.numpy.asarray(z) * 1e-40, ansatz.params)
    value = np.asarray(log_amplitude(rows, params=scaled_params))
    mantissa, exponent = scaled_amplitude(rows, params=scaled_params)
    assert np.isfinite(value).all() and np.isfinite(mantissa).all() and np.isfinite(exponent).all()
    np.testing.assert_allclose(value, baseline + 9 * np.log(1e-40), atol=1e-10)
    np.testing.assert_allclose(value, np.log(np.asarray(mantissa).astype(complex)) + np.asarray(exponent) * np.log(10.), atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("parameter_mode", ["complex", "real-imag"])
@pytest.mark.parametrize("solver", [None, "cg"])
def test_native_netket_fullsum_midpoint_matches_schrodinger_evolution(parameter_mode, solver, monkeypatch):
    nk = require_netket()
    # D=2 has a deficient tangent rank for a generic four-spin state: its
    # projected dynamics need not equal ED. D=3 gives full projective rank.
    setup = make_setup(parameter_mode=parameter_mode, d=3)
    setup = replace(setup, vstate=nk.vqs.FullSumState(setup.hilbert, setup.model,
                                                   variables=setup.vstate.variables))
    initial = np.asarray(setup.vstate.to_array())
    hamiltonian = setup.hamiltonian.to_sparse()
    reference = expm_multiply(-1j * .004 * hamiltonian, initial)
    if solver == "cg":
        from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT
        def forbidden_dense(self):
            raise AssertionError("CG must use QGT matvecs without a dense metric")
        monkeypatch.setattr(QGTJacobianDenseT, "to_dense", forbidden_dense)
    dynamics = setup.make_tdvp_driver(dt=.002, t0=.03, diag_shift=1e-10,
                                     linear_solver=solver,
                                     linear_solver_kwargs={"tol": 1e-11, "maxiter": 2000} if solver else None,
                                     holomorphic=parameter_mode == "complex")
    dynamics.advance(.004)
    final = np.asarray(setup.vstate.to_array())
    infidelity = 1 - abs(np.vdot(reference, final))**2
    assert abs(float(dynamics.t) - .034) < 1e-12
    assert dynamics.state is setup.vstate
    assert infidelity < 1e-9
    assert abs(np.vdot(final, hamiltonian @ final)
               - np.vdot(initial, hamiltonian @ initial)) < 2e-6
    # Handoff must rebuild the evolved complex tensors, not the initial leaves.
    configurations = np.asarray(setup.hilbert.all_states())
    vector = np.asarray(setup.to_peps().to_dense(optimize="greedy")).reshape(-1)
    codes = ((1 - configurations) // 2).astype(int)
    positions = codes @ (2 ** np.arange(3, -1, -1))
    np.testing.assert_allclose(abs(np.vdot(final, vector[positions] / np.linalg.norm(vector))),
                               1., atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("parameter_mode", ["complex", "real-imag"])
@pytest.mark.parametrize("propagation_type", ["real", "imag"])
@pytest.mark.parametrize("integrator, minimum_ratio", [("euler", 1.6), ("midpoint", 3.), ("rk4", 8.)])
def test_fullsum_peps_integrator_orders_and_propagation_signs(parameter_mode, propagation_type,
                                                            integrator, minimum_ratio):
    nk = require_netket()
    setup = make_setup(parameter_mode=parameter_mode, d=3)
    initial_variables = setup.vstate.variables
    initial_state = nk.vqs.FullSumState(setup.hilbert, setup.model, variables=initial_variables)
    initial = np.asarray(initial_state.to_array())
    factor = -1j if propagation_type == "real" else -1.
    reference = expm_multiply(factor * .08 * setup.hamiltonian.to_sparse(), initial)
    reference /= np.linalg.norm(reference)
    errors = []
    for dt in (.02, .01):
        state = nk.vqs.FullSumState(setup.hilbert, setup.model, variables=initial_variables)
        dynamics = make_netket_tdvp_driver(replace(setup, vstate=state), dt=dt,
            integrator=integrator, propagation_type=propagation_type,
            diag_shift=1e-10, holomorphic=parameter_mode == "complex")
        dynamics.advance(.08)
        final = np.asarray(state.to_array())
        phase = np.exp(-1j * np.angle(np.vdot(reference, final)))
        errors.append(np.linalg.norm(reference - phase * final))
        assert float(dynamics.t) == pytest.approx(.08, abs=1e-14)
    # D=3 spans the four-spin projective tangent. Test temporal order against
    # continuous evolution rather than fitting a Trotter trajectory or only
    # asserting finite parameters; imaginary time also catches a sign error.
    assert errors[1] < .03
    assert errors[0] / errors[1] > minimum_ratio


@pytest.mark.integration
def test_native_netket_sampled_boundary_tdvp_advances_shared_state():
    setup = make_setup(contraction="boundary", samples=1024)
    initial = np.asarray(setup.vstate.to_array())
    dynamics = make_netket_tdvp_driver(setup, dt=.001, t0=.02, diag_shift=1e-7)
    for index in range(2):
        dynamics.advance(.001)
        assert float(dynamics.t) == pytest.approx(.02 + (index + 1) * .001)
    final = np.asarray(setup.vstate.to_array())
    expected = expm_multiply(-.002j * setup.hamiltonian.to_sparse(), initial)
    assert 1 - abs(np.vdot(expected, final))**2 < 2e-6
    assert float(dynamics.t) == pytest.approx(.022)
    assert setup.to_peps().max_bond() == 2
    assert np.isfinite(final).all()


def _diagnostic_product_model(holomorphic):
    import flax.linen as nn
    import jax.numpy as jnp

    class ProductModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            coefficients = self.param("coefficients", lambda key: jnp.array(
                [.17 + .09j, -.23 + .12j] if holomorphic else [.17, -.23]))
            return x @ (coefficients if holomorphic else coefficients * jnp.array([1., 1j]))

    return ProductModel()


def _product_tdvp_setup(holomorphic=False, sampled=False):
    import jax
    from types import SimpleNamespace
    jax.config.update("jax_enable_x64", True)
    nk = require_netket()
    hilbert = nk.hilbert.Spin(.5, N=2)
    model = _diagnostic_product_model(holomorphic)
    if sampled:
        state = nk.vqs.MCState(nk.sampler.MetropolisLocal(hilbert, n_chains=4), model,
            seed=7, sampler_seed=8, n_samples=128, n_discard_per_chain=4, chunk_size=16)
    else:
        state = nk.vqs.FullSumState(hilbert, model, seed=7)
    operator = (-nk.operator.spin.sigmax(hilbert, 0) - nk.operator.spin.sigmax(hilbert, 1)
                + .4 * nk.operator.spin.sigmaz(hilbert, 0) @ nk.operator.spin.sigmaz(hilbert, 1))
    return SimpleNamespace(model=model, vstate=state, hamiltonian=operator)


@pytest.mark.integration
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("holomorphic", [False, True])
@pytest.mark.parametrize("qgt", ["auto", "jacobian_dense", "jacobian_pytree", "on_the_fly"])
@pytest.mark.parametrize("solver", ["pinv_smooth", "cholesky", "cg"])
def test_native_qgt_solver_choices_match_independent_tdvp(holomorphic, qgt, solver, sampled):
    setup = _product_tdvp_setup(holomorphic, sampled)
    shift = .03
    driver = make_netket_tdvp_driver(setup, dt=.001, diag_shift=shift,
                                    holomorphic=holomorphic, qgt=qgt, linear_solver=solver)
    velocity = np.asarray(driver.native.ode()["coefficients"])
    if sampled:
        import jax.numpy as jnp
        configurations = np.asarray(setup.vstate.samples).reshape(-1, 2)
        local = setup.vstate.local_estimators(setup.hamiltonian)
        energies = np.asarray(jnp.asarray(getattr(local, "data", local))).reshape(-1)
        weights = np.full(energies.size, 1 / energies.size)
    else:
        vector = np.asarray(setup.vstate.to_array())
        weights = abs(vector) ** 2
        energies = setup.hamiltonian.to_sparse() @ vector / vector
        configurations = np.asarray(setup.vstate.hilbert.all_states())
    derivative = configurations * (1 if holomorphic else np.array([1., 1j]))
    centered = derivative - weights @ derivative
    metric = centered.conj().T @ (weights[:, None] * centered)
    rhs = centered.conj().T @ (weights * (-1j * (energies - weights @ energies)))
    if not holomorphic:
        metric, rhs = metric.real, rhs.real
    expected = np.linalg.solve(metric + shift * np.eye(2), rhs)
    np.testing.assert_allclose(velocity, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.integration
def test_tdvp_tiny_and_adjacent_stops_do_not_overshoot():
    setup = _product_tdvp_setup()
    driver = make_netket_tdvp_driver(setup, dt=.01, t0=.03, diag_shift=.01,
                                    linear_solver="cholesky")
    # NetKet's first-stop np.isclose check used to consume the nearby endpoint
    # before stepping and then advance a whole .01 instead of the requested 1e-8.
    initial_parameters = np.asarray(setup.vstate.parameters["coefficients"]).copy()
    stops = [.03 + 1e-8, .03 + 2e-8]
    times = []
    for time_value in driver.iter(3e-8, tstops=[stops[1], stops[0], stops[0]]):
        times.append(float(time_value))
        np.testing.assert_array_equal(setup.vstate.parameters["coefficients"],
                                      driver.native.integrator.y["coefficients"])
    np.testing.assert_allclose(times, [*stops, .03 + 3e-8], atol=1e-15, rtol=0.)
    assert float(driver.t) == pytest.approx(.03 + 3e-8, abs=1e-15)
    assert np.linalg.norm(np.asarray(setup.vstate.parameters["coefficients"]) - initial_parameters) < 1e-6
    assert [float(t) for t in driver.iter(0.)] == [float(driver.t)]


@pytest.mark.parametrize("stops", [[np.nan], [np.inf], [-np.inf], [.19], [.31]])
def test_tdvp_rejects_invalid_stops_before_native_iteration(stops):
    from pepsy.vmc import NetKetTDVPDriver
    driver = NetKetTDVPDriver(SimpleNamespace(t=.2))
    with pytest.raises(ValueError, match="tstops"):
        list(driver.iter(.1, tstops=stops))


@pytest.mark.integration
def test_tdvp_failed_solver_cannot_silently_return_nonfinite_parameters():
    import jax
    import jax.numpy as jnp
    setup = _product_tdvp_setup()

    def failed_solver(matrix, rhs, x0=None):
        return jax.tree_util.tree_map(lambda z: jnp.full_like(z, jnp.nan), rhs), None

    driver = make_netket_tdvp_driver(setup, dt=.001, diag_shift=.01, linear_solver=failed_solver)
    with pytest.raises(RuntimeError, match="non-finite accepted parameters"):
        driver.advance(.001)


def _independent_diagnostic_reference(configurations, weights, energies, *, holomorphic,
                                      propagation_type, shift):
    # Analytic log derivatives of the product ansatz, independent of NetKet's
    # force/Jacobian code. Its tangent omits the two-spin correlation direction.
    derivative = configurations * (1 if holomorphic else np.array([1., 1j]))
    centered = derivative - weights @ derivative
    centered_energy = energies - weights @ energies
    factor = -1j if propagation_type == "real" else -1.
    metric = centered.conj().T @ (weights[:, None] * centered)
    rhs = centered.conj().T @ (weights * (factor * centered_energy))
    if not holomorphic:
        metric, rhs = metric.real, rhs.real
    expected_velocity = np.linalg.solve(metric + shift * np.eye(2), rhs)
    residual = np.linalg.norm(metric @ expected_velocity - rhs) / np.linalg.norm(rhs)
    tangent_rhs_norm = np.sqrt(weights @ abs(factor * centered_energy) ** 2)
    tangent_norm = np.sqrt(weights @ abs(centered @ expected_velocity - factor * centered_energy) ** 2)
    return residual, tangent_norm, tangent_rhs_norm


@pytest.mark.integration
@pytest.mark.parametrize("holomorphic", [False, True])
@pytest.mark.parametrize("propagation_type", ["real", "imag"])
def test_accepted_state_diagnostics_match_independent_dense_equation(holomorphic, propagation_type):
    import jax
    jax.config.update("jax_enable_x64", True)
    nk = require_netket()
    hilbert = nk.hilbert.Spin(.5, N=2)
    model = _diagnostic_product_model(holomorphic)
    state = nk.vqs.FullSumState(hilbert, model, seed=7)
    operator = (-nk.operator.spin.sigmax(hilbert, 0) - nk.operator.spin.sigmax(hilbert, 1)
                + .4 * nk.operator.spin.sigmaz(hilbert, 0) @ nk.operator.spin.sigmaz(hilbert, 1))
    setup = SimpleNamespace(model=model, vstate=state, hamiltonian=operator)
    shift = .03
    driver = make_netket_tdvp_driver(setup, dt=.01, t0=.2, diag_shift=shift,
                                    linear_solver="cholesky", holomorphic=holomorphic,
                                    propagation_type=propagation_type)
    driver.advance(.01)
    endpoint = state.parameters
    stage_velocity = driver.native._dw
    stage_forces = driver.native._loss_forces
    stage_qgt = driver.native._last_qgt
    # Simulate a stale visible RK-stage state: diagnostics must restore the
    # accepted integrator parameters rather than trust MCState's current leaves.
    state.parameters = jax.tree_util.tree_map(lambda z: z + .5, endpoint)
    actual = driver.diagnostics()
    vector = np.asarray(state.to_array())
    weights = abs(vector) ** 2
    energies = operator.to_sparse() @ vector / vector
    residual, tangent, tangent_rhs = _independent_diagnostic_reference(
        np.asarray(hilbert.all_states()), weights, energies,
        holomorphic=holomorphic, propagation_type=propagation_type, shift=shift)
    assert actual["time"] == pytest.approx(.21)
    assert actual["tdvp_relative_residual"] == pytest.approx(residual, abs=1e-12)
    assert actual["tdvp_relative_residual_squared"] == pytest.approx(residual ** 2, abs=1e-12)
    assert actual["regularized_relative_residual"] < 1e-12
    assert actual["tangent_residual_norm"] == pytest.approx(tangent, abs=1e-12)
    assert actual["tangent_rhs_norm"] == pytest.approx(tangent_rhs, abs=1e-12)
    assert actual["tangent_relative_residual"] == pytest.approx(tangent / tangent_rhs, abs=1e-12)
    assert actual["sampling"]["method"] == "full_sum"
    np.testing.assert_array_equal(state.parameters["coefficients"], endpoint["coefficients"])
    assert driver.native._dw is stage_velocity
    assert driver.native._loss_forces is stage_forces
    assert driver.native._last_qgt is stage_qgt
    # The shifted solve is exact while the physical tangent has a missing component.
    assert actual["tangent_relative_residual"] > .01


@pytest.mark.integration
def test_sampled_endpoint_diagnostics_use_one_chain_batch():
    import jax
    jax.config.update("jax_enable_x64", True)
    nk = require_netket()
    hilbert = nk.hilbert.Spin(.5, N=2)
    model = _diagnostic_product_model(False)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=4)
    state = nk.vqs.MCState(sampler, model, seed=11, sampler_seed=12,
                           n_samples=128, n_discard_per_chain=4, chunk_size=16)
    operator = (-nk.operator.spin.sigmax(hilbert, 0) - nk.operator.spin.sigmax(hilbert, 1)
                + .4 * nk.operator.spin.sigmaz(hilbert, 0) @ nk.operator.spin.sigmaz(hilbert, 1))
    driver = make_netket_tdvp_driver(SimpleNamespace(model=model, vstate=state, hamiltonian=operator),
                                    dt=.001, diag_shift=.02, linear_solver="cg")
    driver.advance(.001)
    integrator_y = driver.native.integrator.y
    actual = driver.diagnostics()
    samples = np.asarray(state.samples)
    local_energies = state.local_estimators(operator)
    energies = np.asarray(jax.numpy.asarray(getattr(local_energies, "data", local_energies))).reshape(-1)
    residual, tangent, tangent_rhs = _independent_diagnostic_reference(
        samples.reshape(-1, 2), np.full(energies.size, 1 / energies.size), energies,
        holomorphic=False, propagation_type="real", shift=.02)
    assert actual["tdvp_relative_residual"] == pytest.approx(residual, abs=1e-10)
    assert actual["tangent_relative_residual"] == pytest.approx(tangent / tangent_rhs, abs=1e-10)
    assert actual["regularized_relative_residual"] < 1e-10
    assert actual["sampling"]["n_samples"] == samples.shape[0] * samples.shape[1]
    assert actual["sampling"]["n_chains"] == 4
    assert 0 <= actual["sampling"]["acceptance_rate"] <= 1
    stats = nk.stats.statistics(energies.reshape(samples.shape[:-1]))
    assert actual["sampling"]["r_hat"] == pytest.approx(float(stats.R_hat))
    assert driver.native.integrator.y is integrator_y
    # Measurement must leave a valid integrator for the following step.
    driver.advance(.001)
    assert float(driver.t) == pytest.approx(.002)


@pytest.mark.integration
def test_endpoint_diagnostics_bound_jacobian_and_support_compressed_peps():
    setup = make_setup(contraction="boundary", samples=32)
    driver = setup.make_tdvp_driver(dt=.001, diag_shift=.02, linear_solver="cholesky")
    with pytest.raises(ValueError, match="diagnostic Jacobian"):
        driver.diagnostics(max_jacobian_bytes=1)
    actual = driver.diagnostics()
    assert actual["regularized_relative_residual"] < 1e-10
    assert np.isfinite(actual["tangent_relative_residual"])
    assert actual["sampling"]["n_samples"] == 32


@pytest.mark.integration
def test_truncated_boundary_tdvp_uses_reverse_derivatives_without_dense_metric(monkeypatch):
    import jax
    nk = require_netket()
    jax.config.update("jax_enable_x64", True)
    setup = build_ising_vmc(qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=24),
        Lx=3, Ly=3, contraction="boundary", chi=1, cutoff=0., parameter_mode="real-imag",
        n_samples=16, n_chains=2, n_discard_per_chain=4, chunk_size=8, use_sr=False,
        seed=44, sampler_seed=45)
    configurations = setup.vstate.samples.reshape(-1, 9)
    exact_model = make_peps_log_amplitude_model(setup.ansatz, contraction="exact",
                                               parameter_mode="real-imag")
    approximate = np.exp(np.asarray(setup.vstate.log_value(configurations)))
    exact = np.exp(np.asarray(exact_model.apply(setup.vstate.variables, configurations)))
    assert np.linalg.norm(approximate - exact) / np.linalg.norm(exact) > 1e-6
    before = [np.asarray(z).copy() for z in jax.tree_util.tree_leaves(setup.vstate.parameters)]

    def forbidden_dense(self):
        raise AssertionError("Boundary CG must not allocate a dense parameter metric")

    from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT
    monkeypatch.setattr(QGTJacobianDenseT, "to_dense", forbidden_dense)
    driver = setup.make_tdvp_driver(dt=.001, diag_shift=.02, linear_solver="cg", qgt="auto",
                                    max_dense_solver_bytes=1)
    driver.advance(.001)
    after = jax.tree_util.tree_leaves(setup.vstate.parameters)
    assert sum(np.linalg.norm(np.asarray(z) - old)
               for z, old in zip(after, before, strict=True)) > 1e-8
    assert setup.to_peps().max_bond() == 2
    report = driver.diagnostics()
    assert report["time"] == pytest.approx(.001)
    assert report["regularized_relative_residual"] < 1e-8
    assert np.isfinite(report["tangent_relative_residual"])
    assert isinstance(driver.native, nk.experimental.TDVP)


@pytest.mark.integration
def test_endpoint_diagnostics_distinguish_zero_normal_force_from_physical_motion():
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    nk = require_netket()

    class RealProduct(nn.Module):
        @nn.compact
        def __call__(self, x):
            coefficient = self.param("coefficient", lambda key: jnp.array(.3))
            return coefficient * x[..., 0]

    hilbert = nk.hilbert.Spin(.5, N=1)
    model = RealProduct()
    state = nk.vqs.FullSumState(hilbert, model, seed=7)
    operator = -nk.operator.spin.sigmax(hilbert, 0)
    driver = make_netket_tdvp_driver(SimpleNamespace(model=model, vstate=state, hamiltonian=operator),
                                    dt=.001, diag_shift=.01, linear_solver="cholesky")
    actual = driver.diagnostics()
    # Real amplitudes cannot follow the imaginary Schrödinger tangent here.
    # The projected force is zero, so both normal-equation checks still pass.
    assert actual["tdvp_relative_residual"] == 0.
    assert actual["regularized_relative_residual"] == 0.
    assert actual["tangent_relative_residual"] == pytest.approx(1.)
    assert actual["tangent_rhs_norm"] > .1


@pytest.mark.integration
def test_nonholomorphic_complex_parameters_require_explicit_real_coordinates():
    setup = make_setup(parameter_mode="complex")
    with pytest.raises(ValueError, match="real coordinates"):
        make_netket_tdvp_driver(setup, dt=.01)


@pytest.mark.parametrize("dt", [0., -1., np.nan, np.inf])
def test_tdvp_rejects_invalid_dt_before_optional_import(dt):
    with pytest.raises(ValueError, match="dt"):
        make_netket_tdvp_driver(SimpleNamespace(), dt=dt)


@pytest.mark.parametrize("options, exception, message", [
    ({"integrator": "bad"}, ValueError, "integrator"),
    ({"integrator": None}, TypeError, "integrator"),
    ({"propagation_type": "bad"}, ValueError, "propagation_type"),
    ({"error_norm": "bad"}, ValueError, "error_norm"),
    ({"linear_solver": "bad"}, ValueError, "linear_solver"),
    ({"linear_solver": 3}, TypeError, "linear_solver"),
])
def test_tdvp_rejects_invalid_controls_before_optional_import(monkeypatch, options, exception, message):
    import pepsy.vmc.netket as bridge

    def forbidden_import():
        raise AssertionError("Invalid controls must be rejected before importing NetKet")

    monkeypatch.setattr(bridge, "_require_netket", forbidden_import)
    with pytest.raises(exception, match=message):
        make_netket_tdvp_driver(SimpleNamespace(), dt=.01, **options)


@pytest.mark.parametrize("contraction", ["boundary", "hotrg", "ctmrg"])
def test_tdvp_rejects_holomorphic_compression_before_optional_import(monkeypatch, contraction):
    import pepsy.vmc.netket as bridge

    def forbidden_import():
        raise AssertionError("Holomorphic compression must fail before constructing NetKet TDVP")

    monkeypatch.setattr(bridge, "_require_netket", forbidden_import)
    setup = SimpleNamespace(model=SimpleNamespace(contraction_method=contraction))
    with pytest.raises(ValueError, match="holomorphic=True is unsupported for compressed PEPS"):
        make_netket_tdvp_driver(setup, dt=.01, holomorphic=True)


@pytest.mark.integration
def test_native_sampling_honors_build_thinning_and_call_time_burn_in():
    from pepsy.vmc import SamplingConfig
    setup = make_setup(samples=32)
    setup.vstate.n_discard_per_chain = 0
    request = SamplingConfig(n_samples=32, n_chains=16, n_discard_per_chain=3)
    setup.sample(request)
    # Native sampler counters include discarded and retained proposals.
    assert int(setup.vstate.sampler_state.n_steps) == (3 + 2) * 16 * setup.sampler.sweep_size
    assert setup.vstate.n_discard_per_chain == 0
    peps = setup.to_peps()
    thinned = build_ising_vmc(peps, Lx=2, Ly=2, use_sr=False,
        sampling=replace(request, thin=3), chunk_size=32)
    assert thinned.sampler.sweep_size == 3
    assert setup.sampler.sweep_size == 4
    samples = thinned.sample(replace(request, thin=3))
    assert samples.diagnostics["sweep_size"] == 3
    assert samples.configs.shape == (2, 16, 4)


@pytest.mark.integration
def test_native_sr_keeps_complex_output_derivatives_by_default():
    from pepsy.vmc import make_netket_vmc_driver
    setup = make_setup(samples=32)
    driver = make_netket_vmc_driver(setup, driver="vmc_sr", use_ntk=False, on_the_fly=False)
    assert str(driver.mode).lower() == "complex"
    real_driver = make_netket_vmc_driver(setup, driver="vmc_sr", sr_mode="real")
    assert str(real_driver.mode).lower() == "real"
    real_imag_driver = make_netket_vmc_driver(setup, driver="vmc_sr", sr_mode="real-imag")
    assert str(real_imag_driver.mode).lower() == "complex"


@pytest.mark.integration
def test_tdvp_dense_budget_and_compressed_forward_mode_fail_before_evolution():
    from pepsy.vmc import make_netket_vmc_driver
    setup = make_setup(contraction="boundary", samples=32)
    with pytest.raises(ValueError, match="dense TDVP metric"):
        setup.make_tdvp_driver(dt=.001, max_dense_solver_bytes=1)
    with pytest.raises(ValueError, match="custom VJPs"):
        setup.make_tdvp_driver(dt=.001, qgt="on_the_fly", linear_solver="cg")
    nk = require_netket()
    with pytest.raises(ValueError, match="custom VJPs"):
        setup.make_tdvp_driver(dt=.001, qgt=partial(nk.optimizer.qgt.QGTOnTheFly),
                               linear_solver="cg")
    # A matrix-free reverse-Jacobian solve does not consume the dense budget.
    setup.make_tdvp_driver(dt=.001, linear_solver="cg", max_dense_solver_bytes=1)
    setup.make_tdvp_driver(dt=.001, qgt="auto", linear_solver="cg", max_dense_solver_bytes=1)
    driver = make_netket_vmc_driver(setup, driver="vmc_sr", use_ntk=True)
    assert driver.on_the_fly is False
    with pytest.raises(ValueError, match="on_the_fly=False"):
        make_netket_vmc_driver(setup, driver="vmc_sr", on_the_fly=True)


@pytest.mark.integration
@pytest.mark.parametrize("driver_name", ["vmc", "vmc_sr"])
def test_truncated_peps_sr_updates_use_reverse_derivatives(driver_name):
    from pepsy.vmc import make_netket_vmc_driver
    jax = pytest.importorskip("jax")
    require_netket()
    setup = build_ising_vmc(qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=24),
        Lx=3, Ly=3, contraction="boundary", chi=1, cutoff=0., parameter_mode="real-imag",
        n_samples=16, n_chains=2, n_discard_per_chain=1, chunk_size=8, use_sr=True,
        seed=44, sampler_seed=45)
    before = [np.asarray(z).copy() for z in jax.tree_util.tree_leaves(setup.vstate.parameters)]
    driver = make_netket_vmc_driver(setup, driver=driver_name, use_ntk=True)
    driver.run(1, show_progress=False)
    after = jax.tree_util.tree_leaves(setup.vstate.parameters)
    assert all(np.isfinite(np.asarray(z)).all() for z in after)
    assert sum(np.linalg.norm(z - old) for z, old in zip(after, before, strict=True)) > 1e-8


@pytest.mark.integration
def test_example_dense_observables_match_pauli_matrices(monkeypatch):
    require_netket()
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    evaluate = runpy.run_path(str(examples / "tfim_netket_tdvp.py"))["dense_spin_observables"]
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([-1., 1.])
    identity = np.eye(2)
    configs = np.array(list(itertools.product((-1, 1), repeat=3)))
    rng = np.random.default_rng(51)
    vector = rng.normal(size=8) + 1j * rng.normal(size=8)
    vector /= np.linalg.norm(vector)
    actual = evaluate(vector, configs, ((0, 1), (1, 2)))
    mx = (np.kron(np.kron(x, identity), identity)
          + np.kron(np.kron(identity, x), identity)
          + np.kron(np.kron(identity, identity), x)) / 3
    zz = (np.kron(np.kron(z, z), identity) + np.kron(np.kron(identity, z), z)) / 2
    parity = np.kron(np.kron(x, x), x)
    for name, matrix in (("mx", mx), ("zz", zz), ("parity_x", parity)):
        assert actual[name] == pytest.approx(np.vdot(vector, matrix @ vector).real, abs=1e-14)


def test_example_dense_oracle_preserves_states_below_raw_float_range(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    require_netket()
    evaluate = runpy.run_path(str(examples / "tfim_netket_tdvp.py"))["normalized_peps_vector"]
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=28)
    positions = np.arange(15, -1, -1)
    reference, log_norm = evaluate(state, positions)
    for tensor in state:
        tensor.modify(data=tensor.data * 1e-100)
    actual, scaled_log_norm = evaluate(state, positions)
    np.testing.assert_allclose(actual, reference, atol=1e-13)
    assert scaled_log_norm == pytest.approx(log_norm + 8 * np.log(1e-100), abs=1e-12)
    assert np.exp(scaled_log_norm) == 0.


@pytest.mark.integration
@pytest.mark.parametrize("reference_origin", ["handoff", "initial"])
def test_example_snapshots_reconstruct_evolved_complex_parameters(monkeypatch, tmp_path, capsys, reference_origin):
    require_netket()
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    import pepsy.vmc as pvmc
    original = pvmc.build_ising_vmc
    setups = []

    def capture(*args, **kwargs):
        setup = original(*args, **kwargs)
        setups.append(setup)
        return setup

    monkeypatch.setattr(pvmc, "build_ising_vmc", capture)
    monkeypatch.setattr("sys.argv", ["tfim_netket_tdvp.py", "--reference", "--steps", "1",
                                    "--final-time", ".0215", "--reference-origin", reference_origin,
                                    "--samples", "64", "--checkpoint-dir", str(tmp_path),
                                    "--tdvp-diagnostics"])
    runpy.run_path(str(examples / "tfim_netket_tdvp.py"), run_name="__main__")
    setup = setups[0]
    data = np.load(tmp_path / "step-0002.npz")
    initial_data = np.load(tmp_path / "step-0000.npz")
    metadata = json.loads(str(data["metadata"]))
    assert metadata["step"] == 2
    assert metadata["time"] == pytest.approx(.0215)
    assert metadata["reference_origin"] == reference_origin
    for name, leaf in setup.vstate.parameters.items():
        np.testing.assert_array_equal(data[f"{name}_real"], leaf["real"])
        np.testing.assert_array_equal(data[f"{name}_imag"], leaf["imag"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    final = [row for row in rows if row["event"] == "tdvp_step"][-1]
    assert final["tdvp_diagnostics"]["time"] == pytest.approx(final["time"])
    assert final["tdvp_diagnostics"]["sampling"]["n_samples"] == 64
    assert final["tdvp_diagnostics"]["regularized_relative_residual"] < 1e-6
    assert final["observables"]["mx"] == pytest.approx(final["exact_observables"]["mx"], abs=1e-4)
    assert np.linalg.norm(data["peps_vector"]) == pytest.approx(1.)
    start_vector = initial_data["peps_vector"]
    if reference_origin == "initial":
        start_vector = np.full(start_vector.size, 1 / np.sqrt(start_vector.size))
        reference_time = .0215
    else:
        reference_time = .0015
    expected = expm_multiply(-1j * reference_time * setup.hamiltonian.to_sparse(), start_vector)
    np.testing.assert_allclose(data["reference_vector"], expected, atol=1e-12)
    phase = np.exp(-1j * np.angle(np.vdot(expected, data["peps_vector"])))
    assert final["phase_aligned_distance"] == pytest.approx(
        np.linalg.norm(expected - phase * data["peps_vector"]), abs=1e-12)
    start_variables = {name: {"real": initial_data[f"{name}_real"],
                             "imag": initial_data[f"{name}_imag"]}
                       for name in setup.vstate.parameters}
    initial_peps = setup.to_peps({"params": start_variables})
    reconstructed = np.asarray(initial_peps.to_dense(optimize="greedy")).reshape(-1)
    reconstructed = reconstructed[initial_data["basis_physical_positions"]]
    reconstructed /= np.linalg.norm(reconstructed)
    np.testing.assert_allclose(initial_data["peps_vector"], reconstructed, atol=1e-12)
