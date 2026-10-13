"""PEPS coordinate derivatives and native NetKet time evolution against ED."""

from types import SimpleNamespace
from dataclasses import replace
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


@pytest.mark.integration
def test_nonholomorphic_complex_parameters_require_explicit_real_coordinates():
    setup = make_setup(parameter_mode="complex")
    with pytest.raises(ValueError, match="real coordinates"):
        make_netket_tdvp_driver(setup, dt=.01)


@pytest.mark.parametrize("dt", [0., -1., np.nan, np.inf])
def test_tdvp_rejects_invalid_dt_before_optional_import(dt):
    with pytest.raises(ValueError, match="dt"):
        make_netket_tdvp_driver(SimpleNamespace(), dt=dt)


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
                                    "--samples", "64", "--checkpoint-dir", str(tmp_path)])
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
