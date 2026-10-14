"""BdyMPS direct compressors preserve native symmetry and fermionic phases."""

import numpy as np
import pytest

from pepsy.boundary import peps_norm

pytestmark = [pytest.mark.optional, pytest.mark.integration, pytest.mark.boundary]


@pytest.mark.parametrize("method", ["direct", "zipup", "dm"])
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("direction", ["x", "y"])
def test_native_flat_boundary_norm_matches_exact_and_preserves_inputs(method, fermionic, direction):
    sr = pytest.importorskip("symmray")
    from pepsy.vmc import fermionic_peps_rand

    peps = (fermionic_peps_rand("Z2", 2, 3, 2, flat=True, dtype="complex128", seed=53)
            if fermionic else sr.PEPS_abelian_rand("Z2", 2, 3, 2, flat=True,
                                                 dtype="complex128", seed=53))
    arrays = [t.data.copy() for t in peps]
    exact = (peps.H & peps).contract(all, optimize="greedy")
    holder = {}
    result = peps_norm(peps, chi=32, bdy=holder, method="dmrg", fit_mode=method,
                       cutoff=0., direction=direction, contraction_opt="greedy")
    np.testing.assert_allclose(result, exact, rtol=3e-12)
    for tensor, original in zip(peps, arrays):
        assert type(tensor.data) is type(original)
        np.testing.assert_array_equal(tensor.data.blocks, original.blocks)
    for mps in holder["bdy"].mps_b.values():
        assert all("Flat" in type(t.data).__name__ for t in mps)
        if fermionic:
            assert all("Fermionic" in type(t.data).__name__ for t in mps)


@pytest.mark.parametrize("method", ["direct", "zipup", "dm"])
def test_truncated_native_fermionic_boundary_has_correct_jitted_gradient(method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import quimb.tensor as qtn
    from pepsy.backends import register_jax_linalg, reset_linalg_registrations
    from pepsy.boundary import contract_flat
    from pepsy.vmc import fermionic_peps_rand

    with jax.enable_x64():
        register_jax_linalg(stabilized=True, qr_rank_policy="adaptive")
        try:
            peps = fermionic_peps_rand("Z2", 3, 3, 4, flat=True, dtype="complex128", seed=73)
            tn = peps.isel({peps.site_ind(site): code for site, code in
                            zip(peps.sites, [2]*4+[3]*4+[0])})
            params, skeleton = qtn.pack(tn)
            params = jax.tree.map(jnp.asarray, params)
            rng = np.random.default_rng(61)
            direction = jax.tree.map(lambda a: jnp.asarray(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)), params)
            def amplitude(p):
                return contract_flat(qtn.unpack(p, skeleton), chi=2, cutoff=0., method="dmrg",
                                     fit_mode=method, preserve_backend=True, contraction_opt="greedy")
            def loss(t):
                shifted = jax.tree.map(lambda a, d: a+t*d, params, direction)
                return jnp.log(jnp.abs(amplitude(shifted)))
            exact = contract_flat(qtn.unpack(params, skeleton), method="exact", preserve_backend=True,
                                  contraction_opt="greedy")
            assert abs(complex(amplitude(params)-exact)) > 1e-4
            derivative = jax.jit(jax.grad(loss))(0.)
            numerical = (loss(1e-6)-loss(-1e-6))/2e-6
            np.testing.assert_allclose(derivative, numerical, rtol=3e-6, atol=1e-7)
            assert all("Flat" in type(t.data).__name__ and "Fermionic" in type(t.data).__name__ for t in tn)
        finally:
            reset_linalg_registrations("jax")


@pytest.mark.parametrize("method", ["direct", "zipup", "dm"])
@pytest.mark.parametrize("symmetry,fermionic", [("U1", False), ("Z2", True)])
def test_sparse_native_boundary_compression_matches_exact_norm(method, symmetry, fermionic):
    sr = pytest.importorskip("symmray")
    peps = sr.PEPS_abelian_rand(symmetry, 2, 3, 2, flat=False, fermionic=fermionic,
                               dtype="complex128", seed=53)
    expected = (peps.H & peps).contract(all, optimize="greedy")
    result = peps_norm(peps, chi=32, fit_mode=method, cutoff=0., contraction_opt="greedy")
    np.testing.assert_allclose(result, expected, rtol=3e-12)
    assert all("Flat" not in type(t.data).__name__ for t in peps)
    if fermionic:
        assert all("Fermionic" in type(t.data).__name__ for t in peps)
