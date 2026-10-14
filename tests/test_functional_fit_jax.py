"""Pure FIT calls, checkpointed finite sweeps, and checked implicit gradients."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.fitting import FIT, make_jax_fit

pytestmark = [pytest.mark.optional, pytest.mark.integration, pytest.mark.mps]


@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_functional_fit_matches_class_and_differentiates_target_and_guess(block_size, checkpoint):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with jax.enable_x64():
        target = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        original = guess.to_dense().copy()
        fit = FIT(target, p=guess, contraction_opt="greedy")
        wrapper = fit.as_functional(
            backend="jax", differentiation="unrolled",
            block_size=block_size, checkpoint=checkpoint,
        )
        params, initial = wrapper.target_params, wrapper.initial_params
        rng = np.random.default_rng(17)
        perturb = lambda p: jax.tree.map(lambda a: jnp.asarray(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)), p)
        dt, dg = perturb(params), perturb(initial)
        weights = jnp.asarray(rng.normal(size=16)+1j*rng.normal(size=16))
        def loss(t):
            target_arrays = jax.tree.map(lambda a, d: a+t*d, params, dt)
            guess_arrays = jax.tree.map(lambda a, d: a+t*d, initial, dg)
            fitted = wrapper(target_arrays, guess_arrays)
            return jnp.real(jnp.vdot(weights, wrapper.to_network(fitted).to_dense().reshape(-1)))
        value, derivative = jax.jit(jax.value_and_grad(loss))(.01)
        numerical = (loss(.010001)-loss(.009999))/2e-6
        np.testing.assert_allclose(derivative, numerical, atol=1e-7, rtol=3e-6)
        actual = jax.jit(wrapper)(params, initial)
        reference = FIT(qtn.unpack(params, wrapper._target_skeleton),
                        p=qtn.unpack(initial, wrapper._guess_skeleton), cutoffs=0., contraction_opt="greedy")
        reference.run_eff(n_iter=2, block_size=block_size, max_bond=2, cutoff=0.,
                          rtol=None, collect_split_diagnostics=False)
        np.testing.assert_allclose(wrapper.to_network(actual).to_dense(), reference.p.to_dense(), atol=1e-12)
        np.testing.assert_array_equal(guess.to_dense(), original)
        # A second call must use fresh tensors and environments, not stale tracers.
        assert np.isfinite(value)
        assert abs(float(loss(.02)-value)) > 1e-5


@pytest.mark.parametrize("block_size", [1, 2])
def test_implicit_fit_gradient_matches_independent_finite_difference(block_size):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with jax.enable_x64():
        target = qtn.MPS_rand_state(6, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        fit = FIT(target, p=guess, contraction_opt="greedy")
        wrapper = fit.as_functional(
            backend="jax", block_size=block_size,
            differentiation="implicit", tolerance=1e-9,
        )
        params = wrapper.target_params
        rng = np.random.default_rng(2)
        direction = jax.tree.map(lambda a: jnp.asarray(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)), params)
        weights = jnp.asarray(rng.normal(size=64)+1j*rng.normal(size=64))
        def loss(t):
            shifted = jax.tree.map(lambda a, d: a+t*d, params, direction)
            return jnp.real(jnp.vdot(weights, wrapper.to_network(wrapper(shifted)).to_dense().reshape(-1)))
        derivative = jax.jit(jax.grad(loss))(.01)
        numerical = (loss(.010005)-loss(.009995))/1e-5
        np.testing.assert_allclose(derivative, numerical, atol=2e-6, rtol=2e-5)
        fitted = jax.jit(wrapper)(params)
        assert float(wrapper.residual(params, fitted)) <= 1e-9
        warm_gradient = jax.grad(lambda g: jnp.real(wrapper.to_network(wrapper(params, g)).to_dense()).sum())(wrapper.initial_params)
        assert all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(warm_gradient))


def test_implicit_nonconvergence_raises_under_jit():
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPS_rand_state(6, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        wrapper = make_jax_fit(target, guess, differentiation="implicit", max_iterations=1, tolerance=1e-14)
        with pytest.raises(Exception, match="FIT fixed point did not converge"):
            jax.block_until_ready(jax.jit(wrapper)(wrapper.target_params))


def test_implicit_adjoint_nonconvergence_is_not_returned_as_a_gradient():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    with jax.enable_x64():
        target = qtn.MPS_rand_state(6, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        wrapper = make_jax_fit(target, guess, differentiation="implicit",
                               adjoint_maxiter=1, adjoint_tolerance=1e-18)
        loss = lambda p: jnp.real(wrapper.to_network(wrapper(p)).to_dense()).sum()
        with pytest.raises(Exception, match="FIT adjoint did not converge"):
            jax.block_until_ready(jax.jit(jax.grad(loss))(wrapper.target_params))


def test_native_flat_functional_fit_keeps_metadata_and_scale_gradient():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    sr = pytest.importorskip("symmray")
    with jax.enable_x64():
        target = sr.MPS_abelian_rand("Z2", 4, 2, flat=True, dtype="complex128", seed=71)
        wrapper = make_jax_fit(target, target.copy())
        def loss(t):
            params = jax.tree.map(lambda a: a*jnp.exp(t/4), wrapper.target_params)
            result = wrapper.to_network(wrapper(params))
            assert all("Flat" in type(t.data).__name__ for t in result)
            return jnp.real((result.H & result).contract(all, optimize="greedy"))
        value, derivative = jax.jit(jax.value_and_grad(loss))(.01)
        np.testing.assert_allclose(derivative, 2*value, rtol=2e-8, atol=1e-8)


def test_mpo_wrapper_reconstructs_both_physical_index_families():
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPO_rand(3, 2, dtype="complex128", seed=21)
        wrapper = make_jax_fit(target, target.copy())
        result = wrapper.to_network(jax.jit(wrapper)(wrapper.target_params))
        assert isinstance(result, qtn.MatrixProductOperator)
        assert result.upper_ind_id == target.upper_ind_id
        assert result.lower_ind_id == target.lower_ind_id
        np.testing.assert_allclose(result.to_dense(), target.to_dense(), atol=1e-12)


def test_native_implicit_fit_rejects_unsupported_gauge_chart():
    jax = pytest.importorskip("jax")
    sr = pytest.importorskip("symmray")
    with jax.enable_x64():
        target = sr.MPS_abelian_rand("Z2", 4, 2, flat=True, dtype="complex128", seed=71)
        with pytest.raises(NotImplementedError, match="native Symmray gauge alignment"):
            make_jax_fit(target, target.copy(), differentiation="implicit")


def test_callable_compressor_preserves_scale_tags_and_input_and_checks_convergence():
    jax = pytest.importorskip("jax")
    from pepsy.fitting import compress_jax_fit
    with jax.enable_x64():
        target = qtn.MPS_rand_state(6, 3, dtype="complex128", seed=51)
        target.exponent = 2.
        original = target.to_dense().copy()
        tags = target.tags.copy()
        target.apply_to_arrays(jax.numpy.asarray)
        arrays, skeleton = qtn.pack(target)
        def evaluate(params):
            network = qtn.unpack(params, skeleton)
            return compress_jax_fit(network, max_bond=2).to_dense()
        result = jax.jit(evaluate)(arrays)
        reference = target.copy()
        reference.compress(max_bond=2, cutoff=0.)
        fit = make_jax_fit(target, reference, differentiation="implicit")
        np.testing.assert_allclose(result, fit.to_network(fit(fit.target_params)).to_dense(), atol=1e-7)
        np.testing.assert_allclose(target.to_dense(), original, atol=1e-12)
        compressed = compress_jax_fit(target, max_bond=2)
        assert compressed.tags == tags and compressed.exponent == target.exponent
        with pytest.raises(Exception, match="FIT fixed point did not converge"):
            bad = jax.jit(lambda params: compress_jax_fit(qtn.unpack(params, skeleton), max_bond=2,
                                                        max_iterations=1, tolerance=1e-14).to_dense())
            jax.block_until_ready(bad(arrays))


@pytest.mark.parametrize("mode", ["unrolled", "implicit"])
def test_wrapper_owns_metadata_and_parameters_after_caller_reuses_networks(mode):
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        reference = FIT(target, p=guess, cutoffs=0., contraction_opt="greedy")
        reference.run_eff(n_iter=20 if mode == "implicit" else 2,
                          max_bond=2, cutoff=0., collect_split_diagnostics=False)
        expected = reference.p.to_dense().copy()
        wrapper = make_jax_fit(target, guess, differentiation=mode)
        target.site_tag_id = "Target{}"
        guess.site_tag_id = "Reused{}"
        target[0].data[...] = 0.
        guess[0].data[...] = 0.
        actual = wrapper.to_network(jax.jit(wrapper)(wrapper.target_params))
        assert actual.site_tag_id == "I{}"
        np.testing.assert_allclose(actual.to_dense(), expected, atol=2e-10)


def test_implicit_real_chart_rejects_changed_complex_parameters():
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPS_rand_state(4, 2, dtype="float64", seed=51)
        wrapper = make_jax_fit(target, target.copy(), differentiation="implicit")
        complex_params = jax.tree.map(lambda x: x*(1+.1j), wrapper.target_params)
        with pytest.raises(ValueError, match="target_params must preserve packed parameter dtypes"):
            jax.jit(wrapper)(complex_params)
        with pytest.raises(ValueError, match="guess_params must preserve packed parameter dtypes"):
            wrapper(wrapper.target_params, complex_params)


def test_implicit_guess_mixed_real_complex_chart_is_rejected():
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPS_rand_state(4, 2, dtype="float64", seed=51)
        guess = target.copy()
        guess[0].modify(data=guess[0].data*(1+.1j))
        with pytest.raises(ValueError, match="uniformly real or complex"):
            make_jax_fit(target, guess, differentiation="implicit")


def test_implicit_fit_preserves_physical_state_under_large_warm_start_gauges():
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        reference = FIT(target, p=guess, cutoffs=0., contraction_opt="greedy")
        reference.run_eff(n_iter=20, max_bond=2, cutoff=0., collect_split_diagnostics=False)
        expected = reference.p.to_dense()
        guess[1].modify(data=guess[1].data*1e12)
        guess[0].modify(data=guess[0].data/1e12)
        wrapper = make_jax_fit(target, guess, differentiation="implicit")
        actual = wrapper.to_network(jax.jit(wrapper)(wrapper.target_params))
        np.testing.assert_allclose(actual.to_dense(), expected, atol=2e-10)


@pytest.mark.parametrize("mode", ["unrolled", "implicit"])
def test_functional_fit_preserves_different_target_and_guess_exponents(mode):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    with jax.enable_x64():
        target = qtn.MPS_rand_state(3, 2, dtype="complex128", seed=51)
        guess = target.copy()
        target.exponent, guess.exponent = 2., -1.
        expected = target.to_dense()
        wrapper = make_jax_fit(target, guess, differentiation=mode)
        fitted = jax.jit(wrapper)(wrapper.target_params)
        np.testing.assert_allclose(wrapper.to_network(fitted).to_dense(), expected, rtol=2e-11, atol=2e-11)
        def loss(t):
            params = jax.tree.map(lambda x: x*jnp.exp(t/3), wrapper.target_params)
            vector = wrapper.to_network(wrapper(params)).to_dense()
            return jnp.sum(jnp.abs(vector)**2)
        value, derivative = jax.jit(jax.value_and_grad(loss))(.01)
        np.testing.assert_allclose(derivative, 2*value, rtol=2e-10)
        assert target.exponent == 2. and guess.exponent == -1.


@pytest.mark.parametrize("direction,center", [("RR", 3), ("LL", 0)])
def test_reconstruction_does_not_reuse_stale_isometry_annotations(direction, center):
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        target = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=51)
        guess = target.copy()
        guess.compress(max_bond=2, cutoff=0.)
        original_flags = [tensor.left_inds for tensor in guess]
        assert any(flags is not None for flags in original_flags)
        wrapper = make_jax_fit(target, guess, block_size=2, sweep_sequence=direction)
        state = wrapper.to_network(jax.jit(wrapper)(wrapper.target_params))
        assert all(tensor.left_inds is None for tensor in state)
        assert state.calc_current_orthog_center() == (center, center)
        assert [tensor.left_inds for tensor in guess] == original_flags
