"""PEPS coordinate derivatives and native NetKet time evolution against ED."""

from types import SimpleNamespace
from dataclasses import replace

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


@pytest.mark.parametrize("contraction", ["exact", "boundary"])
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
    options = dict(contraction=contraction, chi=small_chi if contraction == "boundary" else None,
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
    if contraction == "boundary":
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


@pytest.mark.integration
@pytest.mark.parametrize("parameter_mode", ["complex", "real-imag"])
def test_native_netket_fullsum_midpoint_matches_schrodinger_evolution(parameter_mode):
    nk = require_netket()
    # D=2 has a deficient tangent rank for a generic four-spin state: its
    # projected dynamics need not equal ED. D=3 gives full projective rank.
    setup = make_setup(parameter_mode=parameter_mode, d=3)
    setup = replace(setup, vstate=nk.vqs.FullSumState(setup.hilbert, setup.model,
                                                   variables=setup.vstate.variables))
    initial = np.asarray(setup.vstate.to_array())
    hamiltonian = setup.hamiltonian.to_sparse()
    reference = expm_multiply(-1j * .004 * hamiltonian, initial)
    dynamics = setup.make_tdvp_driver(dt=.002, t0=.03, diag_shift=1e-10,
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
