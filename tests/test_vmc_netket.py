"""Tests for Pepsy's optional NetKet VMC bridge."""

import os

import numpy as np
import pytest
import quimb.tensor as qtn

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_NO_TIPS", "1")

from pepsy.vmc.netket import (  # noqa: E402
    NetKetLocalConfigMap,
    NetKetPEPSVMC,
    PackedFermionicPEPS,
    PackedPEPS,
    SpinOrbitalColumns,
    build_fermi_hubbard_vmc,
    build_heisenberg_vmc,
    build_ising_vmc,
    choose_netket_chunk_size,
    config_to_phys_indices,
    fermionic_peps_rand,
    make_fermionic_peps_batched_amplitude_function,
    make_fermionic_peps_log_amplitude_model,
    make_netket_autochunk_callback,
    make_netket_sr_preconditioner,
    make_netket_vmc_driver,
    make_peps_batched_amplitude_function,
    make_peps_log_amplitude_model,
    netket_spin_orbital_columns,
    occupation_to_phys_indices,
    pack_fermionic_peps_ansatz,
    pack_peps_ansatz,
    recommend_netket_vmc_settings,
    square_lattice_edges,
    verify_netket_spin_columns,
)
from pepsy.vmc.netket import _spinful_phys_lookup  # noqa: E402


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


def test_config_to_phys_indices_spin_half_ordering():
    config_map = NetKetLocalConfigMap.spin_half(up=0, down=1)
    row = np.array([[1, -1, -1, 1]])
    phys = config_to_phys_indices(row, config_map)
    assert phys.tolist() == [[0, 1, 1, 0]]

    reordered = config_to_phys_indices(
        row,
        config_map,
        site_to_config=(2, 0, 3, 1),
    )
    assert reordered.tolist() == [[1, 0, 0, 1]]

    with pytest.raises(ValueError, match="not in config_map"):
        config_to_phys_indices([[0, 1]], config_map)


def test_choose_netket_chunk_size_prefers_power_of_two_divisor():
    assert choose_netket_chunk_size(96, target=64) == 32
    assert choose_netket_chunk_size(100, target=64, require_divisor=False) == 64
    assert choose_netket_chunk_size(12) == 4
    with pytest.raises(ValueError):
        choose_netket_chunk_size(0)


def test_spin_orbital_columns_validate_and_report_shape():
    columns = SpinOrbitalColumns(up=[2, 3], down=[0, 1])
    assert columns.up == (2, 3)
    assert columns.down == (0, 1)
    assert columns.n_orbitals == 2
    assert columns.n_columns == 4

    with pytest.raises(ValueError, match="same length"):
        SpinOrbitalColumns(up=(1, 2), down=(0,))
    with pytest.raises(ValueError, match="disjoint"):
        SpinOrbitalColumns(up=(0, 1), down=(1, 2))
    with pytest.raises(ValueError, match="unique"):
        SpinOrbitalColumns(up=(1, 1), down=(0, 2))


def test_netket_vmc_setup_reports_ansatz_shape_without_optional_dependencies():
    ansatz = PackedPEPS(
        params=None,
        skeleton=None,
        leaves=(),
        treedef=None,
        sites=("a", "b"),
        orbital_sites=("a", "b"),
        orb_to_site=(0, 1),
        site_to_orb=(0, 1),
        n_params=17,
    )
    setup = NetKetPEPSVMC(
        hilbert=None,
        graph=None,
        hamiltonian=None,
        sampler=None,
        vstate=None,
        model=None,
        ansatz=ansatz,
        config_map=None,
        preconditioner=None,
    )
    assert setup.n_sites == 2
    assert setup.n_params == 17


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


def test_generic_peps_log_model_matches_direct_contraction():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=5,
        dtype="complex128",
    )
    ansatz = pack_peps_ansatz(peps, lattice_shape=(2, 2))
    assert isinstance(ansatz, PackedPEPS)
    assert not isinstance(ansatz, PackedFermionicPEPS)

    config_map = NetKetLocalConfigMap.spin_half()
    model = make_peps_log_amplitude_model(ansatz, config_map, contraction="exact")
    row = jnp.asarray([[1, -1, 1, -1]], dtype=jnp.int32)
    phys = config_to_phys_indices(
        np.asarray(row),
        config_map,
        site_to_config=ansatz.site_to_config,
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


def test_generic_peps_batched_amplitude_function_matches_direct_contraction():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=6,
        dtype="complex128",
    )
    ansatz = pack_peps_ansatz(peps, lattice_shape=(2, 2))
    config_map = {1: 0, -1: 1}
    rows = jnp.asarray(
        [
            [1, 1, -1, -1],
            [1, -1, 1, -1],
        ],
        dtype=jnp.int32,
    )

    batched_amp = make_peps_batched_amplitude_function(
        ansatz,
        config_map,
        contraction="exact",
        output="amplitude",
    )
    amps = jax.block_until_ready(batched_amp(rows))

    phys_rows = config_to_phys_indices(
        np.asarray(rows),
        config_map,
        site_to_config=ansatz.site_to_config,
    )
    direct = []
    for phys in phys_rows:
        tnx = peps.isel({
            peps.site_ind(site): phys[k]
            for k, site in enumerate(ansatz.sites)
        })
        direct.append(tnx.contract(all))

    assert np.allclose(np.asarray(amps), np.asarray(direct))

    batched_me = make_peps_batched_amplitude_function(
        ansatz,
        config_map,
        contraction="exact",
        output="mantissa_exponent",
        jit=False,
    )
    mantissa, exponent = batched_me(rows)
    assert np.allclose(np.asarray(mantissa), np.asarray(direct))
    assert np.allclose(np.asarray(exponent), np.zeros(2))


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
    assert setup.n_sites == setup.ansatz.n_sites
    assert setup.n_params == setup.ansatz.n_params
    assert setup.columns.n_orbitals == setup.n_sites
    assert setup.preconditioner is None
    assert make_netket_vmc_driver(setup) is not None
    assert setup.make_driver() is not None


def test_build_ising_vmc_tiny_setup_and_expectation():
    pytest.importorskip("netket")
    pytest.importorskip("flax")

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=7,
        dtype="complex128",
    )
    setup = build_ising_vmc(
        peps,
        Lx=2,
        Ly=2,
        h=1.0,
        J=0.7,
        n_samples=16,
        n_chains=4,
        n_discard_per_chain=0,
        chunk_size=8,
        seed=1,
        sampler_seed=2,
        use_sr=False,
    )
    assert isinstance(setup, NetKetPEPSVMC)
    assert setup.hilbert.n_states == 16
    assert setup.config_map == NetKetLocalConfigMap.spin_half()
    assert setup.preconditioner is None
    assert setup.n_sites == setup.ansatz.n_sites
    assert setup.n_params == setup.ansatz.n_params
    energy = setup.expect_energy()
    assert np.isfinite(np.asarray(energy.mean).real)
    assert make_netket_vmc_driver(setup) is not None
    assert setup.make_driver() is not None


def test_build_heisenberg_vmc_tiny_setup_and_expectation():
    pytest.importorskip("netket")
    pytest.importorskip("flax")

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=8,
        dtype="complex128",
    )
    setup = build_heisenberg_vmc(
        peps,
        Lx=2,
        Ly=2,
        J=1.0,
        total_sz=0.0,
        n_samples=16,
        n_chains=4,
        n_discard_per_chain=0,
        chunk_size=8,
        seed=1,
        sampler_seed=2,
        use_sr=False,
    )
    assert isinstance(setup, NetKetPEPSVMC)
    assert setup.hilbert.n_states == 6
    assert setup.config_map == NetKetLocalConfigMap.spin_half()
    assert setup.preconditioner is None
    assert setup.n_sites == setup.ansatz.n_sites
    assert setup.n_params == setup.ansatz.n_params
    energy = setup.expect_energy()
    assert np.isfinite(np.asarray(energy.mean).real)
    assert make_netket_vmc_driver(setup) is not None
    assert setup.make_driver() is not None


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


def test_spinful_phys_lookup_u1u1_and_z2_fallback():
    # U1U1 charges resolve every (n_up, n_down) -> 2*n_up + n_down.
    lut = _spinful_phys_lookup(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert lut is not None
    assert lut[0, 0] == 0
    assert lut[0, 1] == 1
    assert lut[1, 0] == 2
    assert lut[1, 1] == 3
    # Z2 parity charges (int, 2 sectors) cannot resolve (n_up, n_down): legacy fold.
    assert _spinful_phys_lookup((0, 1)) is None
    assert _spinful_phys_lookup(()) is None


def test_occupation_fold_switches_on_phys_charges():
    columns = SpinOrbitalColumns(up=(2, 3), down=(0, 1))
    # one row per (n_up, n_down) for a single orbital pair (2 orbitals here).
    # occ layout length 2*n_orb = 4; down cols (0,1), up cols (2,3).
    # orbital 0 -> (n_up=1, n_down=1) doublon; orbital 1 -> (n_up=0, n_down=0) empty.
    occ = np.array([[1, 0, 1, 0]])  # dn0=1, dn1=0, up0=1, up1=0
    z2 = occupation_to_phys_indices(occ, columns, phys_charges=None)
    u1u1 = occupation_to_phys_indices(
        occ, columns, phys_charges=((0, 0), (0, 1), (1, 0), (1, 1))
    )
    # doublon: Z2 fold -> 1, U1U1 fold -> 3; empty -> 0 for both.
    assert z2[0].tolist() == [1, 0]
    assert u1u1[0].tolist() == [3, 0]


def test_fermionic_peps_rand_u1u1_fold_is_charge_consistent():
    pytest.importorskip("netket")
    pytest.importorskip("flax")
    pytest.importorskip("symmray")
    import netket as nk

    Lx = Ly = 2
    n_sites = Lx * Ly
    peps = fermionic_peps_rand(
        "U1U1", Lx, Ly, 4, n_fermions_per_spin=(2, 2), seed=5
    )
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    assert ansatz.phys_charges == ((0, 0), (0, 1), (1, 0), (1, 1))

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites, s=1 / 2, n_fermions_per_spin=(2, 2)
    )
    columns = netket_spin_orbital_columns(hilbert)
    states = np.asarray(hilbert.all_states()).astype(int)
    charges = np.asarray(ansatz.phys_charges)

    def charge_sums(phys_charges):
        phys = occupation_to_phys_indices(
            states, columns, site_to_orb=ansatz.site_to_orb, phys_charges=phys_charges
        )
        return charges[phys].sum(axis=1)

    # The U1U1-aware fold maps every sector config to a total charge of (2, 2).
    u1u1_sums = charge_sums(ansatz.phys_charges)
    assert np.all(u1u1_sums == np.array([2, 2]))
    # The legacy Z2 fold is not charge-consistent for U1U1 physical indices.
    legacy_sums = charge_sums(None)
    assert not np.all(legacy_sums == np.array([2, 2]))


def test_u1u1_fermionic_peps_nojit_contractions_work():
    pytest.importorskip("netket")
    pytest.importorskip("flax")
    pytest.importorskip("symmray")
    import netket as nk

    Lx = Ly = 2
    n_sites = Lx * Ly
    peps = fermionic_peps_rand(
        "U1U1", Lx, Ly, 3, n_fermions_per_spin=(2, 2), seed=11
    )
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites, s=1 / 2, n_fermions_per_spin=(2, 2)
    )
    columns = netket_spin_orbital_columns(hilbert)
    rows = np.asarray(hilbert.all_states()[:3]).astype(np.int32)

    assert ansatz.uses_flat_symmray is False
    assert ansatz.phys_charges == ((0, 0), (0, 1), (1, 0), (1, 1))

    for contraction, chi in (
        ("exact", None),
        ("hotrg", 4),
        ("ctmrg", 4),
        ("boundary", 4),
        ("mps", 4),
    ):
        kwargs = {} if chi is None else {"chi": chi}
        fn = make_fermionic_peps_batched_amplitude_function(
            ansatz,
            columns,
            contraction=contraction,
            output="mantissa_exponent",
            jit=False,
            **kwargs,
        )
        mantissa, exponent = fn(rows)

        assert np.asarray(mantissa).shape == (3,)
        assert np.asarray(exponent).shape == (3,)
        assert np.all(np.isfinite(np.asarray(mantissa)))
        assert np.all(np.isfinite(np.asarray(exponent)))

    with pytest.raises(NotImplementedError, match="flat U1U1 fermionic backend"):
        make_fermionic_peps_batched_amplitude_function(
            ansatz,
            columns,
            contraction="exact",
            output="amplitude",
            jit=True,
        )


def test_u1u1_fermionic_peps_full_netket_vmc_fails_clearly():
    pytest.importorskip("netket")
    pytest.importorskip("flax")
    pytest.importorskip("symmray")

    peps = fermionic_peps_rand(
        "U1U1", 2, 2, 3, n_fermions_per_spin=(2, 2), seed=12
    )

    with pytest.raises(NotImplementedError, match="flat U1U1 fermionic backend"):
        build_fermi_hubbard_vmc(
            peps,
            Lx=2,
            Ly=2,
            n_samples=8,
            n_chains=2,
            n_discard_per_chain=0,
            chunk_size=4,
            seed=1,
            sampler_seed=2,
            use_sr=False,
            contraction="exact",
        )


def test_fermionic_peps_rand_z2_uses_flat_backend():
    pytest.importorskip("symmray")
    peps = fermionic_peps_rand("Z2", 2, 2, 2, seed=1)
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(2, 2))
    # Z2 physical index is parity-resolved (2 sectors) -> legacy fold, flat backend.
    assert _spinful_phys_lookup(ansatz.phys_charges) is None
    assert ansatz.uses_flat_symmray is True
