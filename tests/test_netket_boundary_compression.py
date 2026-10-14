"""Fixed-budget compression through the public JAX amplitude builder."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.vmc import make_peps_batched_amplitude_function, pack_peps_ansatz


@pytest.mark.parametrize("method, bsz", [("dmrg", 1), ("dmrg2", 2)])
def test_dmrg_dispatch_uses_pepsy_compressor_and_explicit_differentiation(method, bsz):
    from pepsy.fitting import compress_jax_fit
    from pepsy.vmc.netket import _boundary_contraction_options
    options = _boundary_contraction_options({"method": method}, jit=True)
    assert options["method"] is compress_jax_fit
    assert options["differentiation"] == "auto" and options["bsz"] == bsz
    unrolled = _boundary_contraction_options({"method": method, "differentiation": "unrolled"}, jit=True)
    assert unrolled["max_iterations"] == 2
    implicit = _boundary_contraction_options({"method": method, "differentiation": "implicit"}, jit=True)
    assert implicit["max_iterations"] == 100


def test_native_explicit_implicit_fit_reports_unsupported_chart():
    jax = pytest.importorskip("jax")
    sr = pytest.importorskip("symmray")
    with jax.enable_x64():
        peps = sr.PEPS_abelian_rand("Z2", 3, 3, bond_dim=4, flat=True,
                                   dtype="complex128", seed=82)
        ansatz = pack_peps_ansatz(peps)
        fn = make_peps_batched_amplitude_function(ansatz, contraction="boundary", chi=2,
            cutoff=0., contraction_opts={"method": "dmrg", "differentiation": "implicit"})
        with pytest.raises(NotImplementedError, match="Implicit native Symmray"):
            fn(np.ones((1, 9), dtype=int))


def test_implicit_dmrg_gradient_through_multiple_boundary_compressions_and_configurations():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    with jax.enable_x64():
        ansatz = pack_peps_ansatz(qtn.PEPS.rand(4, 3, bond_dim=2, dtype="complex128", seed=82))
        params = jax.tree.map(jnp.asarray, ansatz.params)
        fn = make_peps_batched_amplitude_function(ansatz, contraction="boundary", chi=1,
            cutoff=0., contraction_opts={"method": "dmrg", "sequence": ("xmin",),
                                        "max_separation": 1}, output="log")
        rows = np.asarray([[1, -1]*6, [-1, 1]*6])
        rng = np.random.default_rng(28)
        direction = jax.tree.map(lambda z: jnp.asarray(rng.normal(size=z.shape)
                                                        + 1j*rng.normal(size=z.shape)), params)
        def loss(t):
            shifted = jax.tree.map(lambda p, d: p+t*d, params, direction)
            return jnp.real(fn(rows, shifted)).sum()
        derivative = jax.jit(jax.grad(loss))(0.)
        numerical = (loss(1e-5)-loss(-1e-5))/2e-5
        np.testing.assert_allclose(derivative, numerical, atol=1e-7, rtol=3e-6)


@pytest.mark.parametrize("method", ["zipup", "src", "sdc", "dm", "dmrg", "dmrg2"])
def test_jit_compression_matches_eager_and_directional_derivative(method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from pepsy.backends import reg_rel_svd_jax

    jax.config.update("jax_enable_x64", True)
    reg_rel_svd_jax()
    ansatz = pack_peps_ansatz(qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=82))
    theta, unpack = ravel_pytree(ansatz.params)
    options = {"method": method}
    if method == "src":
        options["seed"] = 42
    configs = np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1]])
    kwargs = dict(contraction="boundary", chi=1, cutoff=0., contraction_opts=options, output="amplitude")
    compiled = make_peps_batched_amplitude_function(ansatz, **kwargs)
    eager = make_peps_batched_amplitude_function(ansatz, jit=False, **kwargs)
    jax_params = jax.tree.map(jnp.asarray, ansatz.params)
    np.testing.assert_allclose(compiled(configs, jax_params), eager(configs, jax_params), rtol=1e-11)
    exact = make_peps_batched_amplitude_function(ansatz, output="amplitude")
    assert np.max(np.abs(compiled(configs) - exact(configs))) > 1e-4
    # A real scalar path through complex tensor coordinates exercises both
    # real and imaginary derivatives without assuming holomorphic compression.
    rng = np.random.default_rng(28)
    direction = jnp.asarray(rng.normal(size=theta.shape) + 1j * rng.normal(size=theta.shape))
    direction /= jnp.linalg.norm(direction)

    def loss(t):
        return jnp.log(jnp.abs(compiled(configs, unpack(theta + t * direction))[0]))

    derivative = jax.jit(jax.grad(loss))(0.)
    delta = 1e-5
    numerical = (loss(delta) - loss(-delta)) / (2 * delta)
    np.testing.assert_allclose(derivative, numerical, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("method", ["zipup", "sdc", "dm", "dmrg", "dmrg2"])
def test_flat_z2_boundary_methods_keep_native_contractions_and_gradients(method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    sr = pytest.importorskip("symmray")
    from pepsy.backends import register_jax_linalg, reset_linalg_registrations

    jax.config.update("jax_enable_x64", True)
    peps = sr.PEPS_abelian_rand("Z2", 3, 3, bond_dim=4, flat=True,
                               dtype="complex128", seed=82)
    ansatz = pack_peps_ansatz(peps)
    params = jax.tree.map(jnp.asarray, ansatz.params)
    configs = np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1]])
    fn = make_peps_batched_amplitude_function(ansatz, contraction="boundary", chi=2,
                                              contraction_opts={"method": method}, output="amplitude")
    rng = np.random.default_rng(18)
    direction = jax.tree.map(lambda a: jnp.asarray(rng.normal(size=a.shape)
                                                  + 1j * rng.normal(size=a.shape)), params)

    def loss(t):
        p = jax.tree.map(lambda a, d: a + t * d, params, direction)
        return jnp.log(jnp.abs(fn(configs, p)[0]))

    register_jax_linalg(stabilized=True, qr_rank_policy="adaptive")
    try:
        derivative = jax.jit(jax.grad(loss))(0.)
        numerical = (loss(1e-6) - loss(-1e-6)) / 2e-6
        np.testing.assert_allclose(derivative, numerical, rtol=3e-6, atol=1e-7)
        assert all("Flat" in type(peps[site].data).__name__ for site in peps.sites)
    finally:
        reset_linalg_registrations("jax")


@pytest.mark.parametrize("options, message", [
    ({"method": "src"}, "integer seed"),
    ({"method": "dmrg", "tol": 1e-5}, "fixed sweeps"),
    ({"method": "dmrg2", "max_iterations": None}, "positive integer"),
    ({"method": "dmrg", "max_iterations": 0}, "positive integer"),
    ({"method": "dmrg", "tn_fit": None, "seed": True}, "integer seed"),
    ({"method": "dmrg2", "bsz": 1}, "requires bsz=2"),
    ({"method": "zipup", "mode": "mps"}, "Conflicting"),
    ({"method": "fit", "tn_fit": None}, "integer seed"),
    ({"method": "dmrg", "progbar": True}, "fixed sweeps"),
])
def test_invalid_jit_compression_settings_fail_before_tracing(options, message):
    pytest.importorskip("jax")
    ansatz = pack_peps_ansatz(qtn.PEPS.rand(2, 2, bond_dim=2, seed=2))
    with pytest.raises(ValueError, match=message):
        make_peps_batched_amplitude_function(ansatz, contraction="boundary", chi=2,
                                           contraction_opts=options)


def test_fit_initialization_options_are_isolated_from_caller():
    from pepsy.vmc.netket import _boundary_contraction_options

    original = {"method": "fit", "tn_fit": {"method": "zipup"}}
    resolved = _boundary_contraction_options(original, jit=True)
    resolved["tn_fit"]["max_bond"] = 4
    assert original == {"method": "fit", "tn_fit": {"method": "zipup"}}


@pytest.mark.integration
@pytest.mark.parametrize("method", ["zipup", "dmrg", "dmrg2"])
def test_native_fermionic_compression_preserves_jit_values_and_complex_derivative(method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pepsy.backends import register_jax_linalg
    from pepsy.vmc import (SpinOrbitalColumns, fermionic_peps_rand,
                           make_fermionic_peps_batched_amplitude_function,
                           pack_fermionic_peps_ansatz)

    with jax.enable_x64():
        register_jax_linalg(stabilized=True, qr_rank_policy="adaptive")
        peps = fermionic_peps_rand("Z2", 3, 3, 4, flat=True, dtype="complex128", seed=73)
        ansatz = pack_fermionic_peps_ansatz(peps)
        params = jax.tree_util.tree_map(jnp.asarray, ansatz.params)
        columns = SpinOrbitalColumns(tuple(range(0, 18, 2)), tuple(range(1, 18, 2)))
        # Eight particles preserve the native even parity sector. These are
        # occupation bits; the bridge owns physical-index and Fock-phase maps.
        rows = np.array([[1, 0] * 4 + [0, 1] * 4 + [0, 0]], dtype=np.int32)
        options = {"method": method, "sequence": ("xmin",), "max_separation": 1}
        if method.startswith("dmrg"):
            options.update(tol=0., optimize="greedy")
        kwargs = dict(contraction="boundary", chi=2, cutoff=0.,
                      contraction_opts=options, output="amplitude")
        compiled = make_fermionic_peps_batched_amplitude_function(ansatz, columns, **kwargs)
        eager = make_fermionic_peps_batched_amplitude_function(ansatz, columns, jit=False, **kwargs)
        values = np.asarray(compiled(rows, params=params))
        assert np.isfinite(values).all() and np.max(np.abs(values)) > 0
        np.testing.assert_allclose(values, eager(rows, params=params), rtol=2e-10, atol=1e-9)
        exact = make_fermionic_peps_batched_amplitude_function(ansatz, columns, output="amplitude")
        assert np.max(np.abs(values - np.asarray(exact(rows)))) > 1e-4

        leaves, tree = jax.tree_util.tree_flatten(params)
        rng = np.random.default_rng(61)
        deltas = [rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape) for z in leaves]
        norm = np.sqrt(sum(np.vdot(d, d).real for d in deltas))
        direction = jax.tree_util.tree_unflatten(tree, [jnp.asarray(d / norm) for d in deltas])

        def loss(t):
            shifted = jax.tree_util.tree_map(lambda p, d: p + t * d, params, direction)
            return jnp.log(jnp.abs(compiled(rows, params=shifted)[0]))

        derivative = jax.jit(jax.grad(loss))(0.)
        numerical = (loss(1e-5) - loss(-1e-5)) / 2e-5
        np.testing.assert_allclose(derivative, numerical, rtol=3e-6, atol=1e-7)
        assert all("Fermionic" in type(t.data).__name__ and "Flat" in type(t.data).__name__
                   for t in peps)


def test_symmray_rng_capability_error_does_not_substitute_a_dense_contraction(monkeypatch):
    from types import SimpleNamespace
    from pepsy.vmc.netket import _contract_boundary_for_vmc

    def unavailable(*args, **kwargs):
        raise ImportError("autoray couldn't find function 'random.default_rng' for backend 'symmray'.")

    monkeypatch.setattr("pepsy.vmc.netket.call_quimb_2d", unavailable)
    with pytest.raises(NotImplementedError, match="native Symmray backend"):
        _contract_boundary_for_vmc(SimpleNamespace(contract_boundary=lambda **kwargs: None),
                                  max_bond=2, cutoff=0., method_opts={"method": "src"})


@pytest.mark.integration
def test_fixed_sweep_boundary_fit_runs_native_netket_sr_and_real_coordinate_tdvp():
    jax = pytest.importorskip("jax")
    nk = pytest.importorskip("netket")
    if not hasattr(nk, "experimental"):
        pytest.skip("A working NetKet installation is required")
    from pepsy.backends import register_jax_linalg
    from pepsy.vmc import build_ising_vmc

    with jax.enable_x64():
        jax.sharding.set_mesh(jax.make_mesh((len(jax.devices()),), ("S",),
                                          axis_types=(jax.sharding.AxisType.Auto,)))
        register_jax_linalg(stabilized=True, qr_rank_policy="adaptive")
        setup = build_ising_vmc(qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=24),
            Lx=3, Ly=3, J=-1., h=.7, contraction="boundary", chi=1, cutoff=0.,
            contraction_opts={"method": "dmrg", "differentiation": "unrolled",
                              "max_iterations": 2, "optimize": "greedy"},
            parameter_mode="real-imag", n_samples=16, n_chains=2, chunk_size=8,
            n_discard_per_chain=1, seed=44, sampler_seed=45)
        shapes = [z.shape for z in jax.tree_util.tree_leaves(setup.vstate.parameters)]
        before = [np.asarray(z).copy() for z in jax.tree_util.tree_leaves(setup.vstate.parameters)]
        setup.make_driver(driver="vmc_sr", use_ntk=True, on_the_fly=False).run(1, show_progress=False)
        after = jax.tree_util.tree_leaves(setup.vstate.parameters)
        assert all(np.isfinite(np.asarray(z)).all() for z in after)
        assert sum(np.linalg.norm(np.asarray(z) - old) for z, old in zip(after, before, strict=True)) > 0
        dynamics = setup.make_tdvp_driver(dt=1e-4, diag_shift=1e-4, linear_solver="cg")
        dynamics.advance(1e-4)
        assert float(dynamics.t) == pytest.approx(1e-4)
        leaves = jax.tree_util.tree_leaves(setup.vstate.parameters)
        assert [z.shape for z in leaves] == shapes
        assert all(np.isfinite(np.asarray(z)).all() for z in leaves)
        assert setup.to_peps().max_bond() == 2
