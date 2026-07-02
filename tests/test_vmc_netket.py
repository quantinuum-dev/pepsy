"""Tests for Pepsy's optional NetKet VMC bridge."""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_NO_TIPS", "1")

from pepsy.vmc.netket import (  # noqa: E402
    SpinOrbitalColumns,
    build_fermi_hubbard_vmc,
    choose_netket_chunk_size,
    make_fermionic_peps_batched_amplitude_function,
    make_fermionic_peps_log_amplitude_model,
    make_netket_autochunk_callback,
    make_netket_sr_preconditioner,
    make_netket_vmc_driver,
    netket_spin_orbital_columns,
    occupation_to_phys_indices,
    pack_fermionic_peps_ansatz,
    recommend_netket_vmc_settings,
    square_lattice_edges,
    verify_netket_spin_columns,
)


def test_square_lattice_edges_open_boundary_order():
    assert square_lattice_edges(2, 3) == (
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (4, 5),
    )


def test_occupation_to_phys_indices_spinful_ordering():
    columns = SpinOrbitalColumns(up=(4, 5, 6, 7), down=(0, 1, 2, 3))
    # sites: empty, doublon, up, down
    row = np.array([[0, 1, 0, 1, 0, 1, 1, 0]])
    phys = occupation_to_phys_indices(row, columns)
    assert phys.tolist() == [[0, 1, 2, 3]]


def test_choose_netket_chunk_size_prefers_power_of_two_divisor():
    assert choose_netket_chunk_size(96, target=64) == 32
    assert choose_netket_chunk_size(100, target=64, require_divisor=False) == 64
    assert choose_netket_chunk_size(12) == 4
    with pytest.raises(ValueError):
        choose_netket_chunk_size(0)


def test_recommend_netket_vmc_settings_large_parameter_count():
    settings = recommend_netket_vmc_settings(
        n_params=20_000,
        n_samples=1024,
        n_chains=24,
        target_chunk_size=300,
    )
    assert settings.driver == "vmc_sr"
    assert settings.use_sr is True
    assert settings.sr_mode == "real"
    assert settings.use_ntk is True
    assert settings.on_the_fly is True
    assert settings.chunks.chunk_size == 256
    assert settings.chunks.sampler_chunk_size == 8
    assert settings.chunks.chunk_size_bwd == 256


def test_netket_spin_columns_match_number_operators():
    nk = pytest.importorskip("netket")

    hi = nk.hilbert.SpinOrbitalFermions(
        4,
        s=1 / 2,
        n_fermions_per_spin=(2, 2),
    )
    columns = netket_spin_orbital_columns(hi)
    assert columns == SpinOrbitalColumns(up=(4, 5, 6, 7), down=(0, 1, 2, 3))
    assert verify_netket_spin_columns(hi, columns) == columns


def test_fermionic_peps_log_model_matches_direct_contraction():
    sr = pytest.importorskip("symmray")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        2,
        2,
        2,
        phys_dim=4,
        subsizes="equal",
        flat=True,
        seed=1,
    )
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(2, 2))
    columns = SpinOrbitalColumns(up=(4, 5, 6, 7), down=(0, 1, 2, 3))
    model = make_fermionic_peps_log_amplitude_model(
        ansatz,
        columns,
        contraction="exact",
    )

    row = jnp.asarray([[0, 1, 0, 1, 0, 1, 1, 0]], dtype=jnp.int32)
    phys = occupation_to_phys_indices(
        np.asarray(row),
        columns,
        site_to_orb=ansatz.site_to_orb,
    )[0]
    tnx = peps.isel({
        peps.site_ind(site): phys[k]
        for k, site in enumerate(ansatz.sites)
    })
    direct = tnx.contract(all)

    variables = model.init(jax.random.PRNGKey(0), row)
    log_amp = model.apply(variables, row)[0]
    amp = jax.block_until_ready(jnp.exp(log_amp))
    assert np.allclose(np.asarray(amp), np.asarray(direct))


def test_fermionic_peps_batched_amplitude_function_matches_direct_contraction():
    sr = pytest.importorskip("symmray")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        2,
        2,
        2,
        phys_dim=4,
        subsizes="equal",
        flat=True,
        seed=4,
    )
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(2, 2))
    columns = SpinOrbitalColumns(up=(4, 5, 6, 7), down=(0, 1, 2, 3))
    rows = jnp.asarray(
        [
            [0, 1, 0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 1],
        ],
        dtype=jnp.int32,
    )

    assert ansatz.site_inds == tuple(peps.site_ind(site) for site in ansatz.sites)
    batched_amp = make_fermionic_peps_batched_amplitude_function(
        ansatz,
        columns,
        contraction="exact",
        output="amplitude",
    )
    amps = jax.block_until_ready(batched_amp(rows))

    phys_rows = occupation_to_phys_indices(
        np.asarray(rows),
        columns,
        site_to_orb=ansatz.site_to_orb,
    )
    direct = []
    for phys in phys_rows:
        tnx = peps.isel({
            peps.site_ind(site): phys[k]
            for k, site in enumerate(ansatz.sites)
        })
        direct.append(tnx.contract(all))

    assert np.allclose(np.asarray(amps), np.asarray(direct))

    batched_me = make_fermionic_peps_batched_amplitude_function(
        ansatz,
        columns,
        contraction="exact",
        output="mantissa_exponent",
        jit=False,
    )
    mantissa, exponent = batched_me(rows)
    assert np.allclose(np.asarray(mantissa), np.asarray(direct))
    assert np.allclose(np.asarray(exponent), np.zeros(2))

    batched_hotrg = make_fermionic_peps_batched_amplitude_function(
        ansatz,
        columns,
        contraction="hotrg",
        chi=2,
    )
    mantissa, exponent = jax.block_until_ready(batched_hotrg(rows))
    assert np.asarray(mantissa).shape == (2,)
    assert np.asarray(exponent).shape == (2,)

    for contraction in ("ctmrg", "mps"):
        batched_boundary = make_fermionic_peps_batched_amplitude_function(
            ansatz,
            columns,
            contraction=contraction,
            chi=2,
        )
        mantissa, exponent = jax.block_until_ready(batched_boundary(rows))
        assert np.asarray(mantissa).shape == (2,)
        assert np.asarray(exponent).shape == (2,)


def test_build_fermi_hubbard_vmc_tiny_setup():
    pytest.importorskip("netket")
    pytest.importorskip("flax")
    sr = pytest.importorskip("symmray")

    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        2,
        2,
        2,
        phys_dim=4,
        subsizes="equal",
        flat=True,
        seed=2,
    )
    setup = build_fermi_hubbard_vmc(
        peps,
        Lx=2,
        Ly=2,
        n_samples=16,
        n_chains=4,
        n_discard_per_chain=0,
        chunk_size=8,
        seed=1,
        sampler_seed=2,
        use_sr=False,
    )
    assert setup.hilbert.n_states == 36
    assert setup.ansatz.n_sites == 4
    assert setup.preconditioner is None
    assert make_netket_vmc_driver(setup) is not None


def test_make_netket_sr_preconditioner_resolves_qgt():
    nk = pytest.importorskip("netket")

    sr = make_netket_sr_preconditioner(qgt="onthefly", diag_shift=0.02)
    assert sr.qgt_constructor is nk.optimizer.qgt.QGTOnTheFly
    assert sr.diag_shift == 0.02


def test_make_netket_autochunk_callback_sets_initial_values():
    pytest.importorskip("netket")

    callback = make_netket_autochunk_callback(
        sampler_chunk_size=4,
        chunk_size=8,
        chunk_size_bwd=2,
        minimum_chunk_size=1,
    )
    assert callback.sampler_chunk_size == 4
    assert callback.chunk_size == 8
    assert callback.chunk_size_bwd == 2


def test_make_netket_vmc_sr_driver_tiny_setup():
    pytest.importorskip("netket")
    pytest.importorskip("flax")
    sr = pytest.importorskip("symmray")

    peps = sr.networks.PEPS_fermionic_rand(
        "Z2",
        2,
        2,
        2,
        phys_dim=4,
        subsizes="equal",
        flat=True,
        seed=3,
    )
    setup = build_fermi_hubbard_vmc(
        peps,
        Lx=2,
        Ly=2,
        n_samples=16,
        n_chains=4,
        n_discard_per_chain=0,
        chunk_size=8,
        seed=1,
        sampler_seed=2,
        use_sr=False,
    )
    driver = make_netket_vmc_driver(
        setup,
        driver="vmc_sr",
        sr_mode="real",
        use_ntk=False,
        on_the_fly=False,
        chunk_size_bwd=8,
    )
    assert driver.mode == "real"
    assert driver.use_ntk is False
    assert driver.chunk_size_bwd == 8
