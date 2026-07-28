"""Tests for MERA local-energy optimization helpers."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.energy import EnergyEstimate
from pepsy.optimizers.mera import (
    GateSpec,
    LocalTerm,
    MeraEnergyOptimizer,
    QMeraBlockSpec,
    QMeraBuilder,
    QMeraCompiledLightconeChunk,
    QMeraContractionPathCache,
    QMeraDisentanglerSpec,
    QMeraGeometry,
    QMeraIsometrySpec,
    QMeraLightconeGroup,
    QMeraLightconeTN,
    QMeraSchematicBlock,
    QMeraParametricEnergyOptimizer,
    QMeraParametricLightconeChunk,
    QMeraSymmrayFermionBackend,
    QMeraScaleSpec,
    QMeraUnitarySpec,
    UserGateFamily,
    build_lightcone_chunks,
    build_qmera_contraction_optimizer,
    build_qmera_lightcone_chunks,
    build_qmera_parametric_lightcone_chunks,
    compile_qmera_parametric_lightcones,
    contract_qmera_lightcone_tn,
    group_qmera_parametric_lightcone_chunks,
    lightcone_energy,
    default_gate_registry,
    draw_qmera_schedule,
    local_qmera_compiled_lightcone_expectation,
    local_qmera_parametric_lightcone_expectation,
    local_lightcone_expectation,
    normalize_local_terms,
    qmera_compiled_parametric_energy,
    qmera_direct_parametric_energy,
    qmera_parametric_energy,
    qmera_parametric_lightcone_state,
    qmera_parametric_lightcone_tn,
    qmera_schematic_blocks,
    qmera_symmray_fermi_hubbard_terms,
    qmera_symmray_majorana_terms,
    symmray_fermion_gate_registry,
    symmray_majorana_gate_registry,
)
from pepsy.optimizers.mera.optimizer import (
    MeraEnergyOptimizer as ModuleMeraEnergyOptimizer,
)
from pepsy.tensors import Fermion


def _zz_term():
    z_op = np.diag([1.0, -1.0])
    return np.kron(z_op, z_op).reshape(2, 2, 2, 2)


def _small_mera(seed=23):
    return qtn.MERA.rand(L=8, max_bond=2, dtype="complex128", seed=seed)


def test_qmera_contraction_optimizer_helper_delegates(monkeypatch):
    """qMERA cache helper should expose Pepsy's reusable cotengra builder."""
    calls = []

    def fake_build_optimizer(**kwargs):
        calls.append(kwargs)
        return "optimizer"

    monkeypatch.setattr(
        "pepsy.optimizers.mera.cache.build_optimizer",
        fake_build_optimizer,
    )

    opt = build_qmera_contraction_optimizer(directory="/tmp/qmera-cache", max_repeats=3)

    assert opt == "optimizer"
    assert calls[0]["directory"] == "/tmp/qmera-cache"
    assert calls[0]["max_repeats"] == 3


def test_qmera_builder_exposes_contraction_optimizer(monkeypatch):
    """QMeraBuilder should provide the same cache helper at the builder level."""
    calls = []

    def fake_build_qmera_contraction_optimizer(**kwargs):
        calls.append(kwargs)
        return "optimizer"

    monkeypatch.setattr(
        "pepsy.optimizers.mera.builders.build_qmera_contraction_optimizer",
        fake_build_qmera_contraction_optimizer,
    )

    builder = QMeraBuilder(shape=4)
    opt = builder.contraction_optimizer(directory="/tmp/qmera-cache")

    assert opt == "optimizer"
    assert calls == [{"directory": "/tmp/qmera-cache"}]


def test_qmera_builder_infers_fermion_register_convention():
    """A stored Fermion model should supply qMERA mode metadata."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    builder = QMeraBuilder(shape=(2, 2), fermion=fermion)

    assert builder.fermion is fermion
    assert builder.geometry.site_modes == ("up", "down")
    assert builder.geometry.mode_order == "mode-major"
    assert builder.geometry.num_modes == 8

    backend = QMeraSymmrayFermionBackend.from_fermion(fermion)
    assert backend.symmetry == "U1U1"
    assert backend.site_modes == ("up", "down")
    assert backend.mode_order == "mode-major"


def test_qmera_builder_preserves_explicit_geometry_override():
    """Advanced callers may select a different register order explicitly."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    geometry = QMeraGeometry(
        shape=(2, 2),
        site_modes=("up", "down"),
        mode_order="site-major",
    )
    builder = QMeraBuilder(geometry=geometry, fermion=fermion)

    assert builder.geometry is geometry
    assert builder.geometry.mode_order == "site-major"

    bad_geometry = QMeraGeometry(shape=(2, 2), site_modes=("mode",))
    with pytest.raises(ValueError, match="site_modes must match"):
        QMeraBuilder(geometry=bad_geometry, fermion=fermion)


def test_normalize_local_terms_accepts_mapping_iterable_and_local_term():
    """Hamiltonian input should normalize to explicit LocalTerm objects."""
    op = _zz_term()

    terms = normalize_local_terms({(0, 1): op})
    assert terms == (LocalTerm(where=(0, 1), operator=op),)

    weighted = normalize_local_terms([((1, 2), op, 0.5)])
    assert weighted[0].where == (1, 2)
    assert weighted[0].operator is op
    assert weighted[0].weight == pytest.approx(0.5)

    local = LocalTerm(where=((0, 0),), operator=op)
    assert normalize_local_terms([local])[0].where == ((0, 0),)


def test_normalize_local_terms_rejects_bad_supports():
    """Invalid local-term structure should fail before contraction."""
    op = _zz_term()

    with pytest.raises(ValueError, match="empty"):
        normalize_local_terms({(): op})
    with pytest.raises(ValueError, match="duplicate"):
        normalize_local_terms({(0, 0): op})
    with pytest.raises(ValueError, match="form"):
        normalize_local_terms([((0, 1), op, 1.0, "extra")])


def test_mera_lightcone_expectation_matches_quimb_exact_contraction():
    """The cached lightcone kernel should match quimb's full exact oracle."""
    mera = _small_mera()
    where = (0, 1)
    op = _zz_term()
    terms = normalize_local_terms({where: op})
    chunk = build_lightcone_chunks(mera, terms)[0]

    local_value = local_lightcone_expectation(
        mera,
        chunk,
        optimize="auto-hq",
        normalized=True,
        real=False,
    )
    direct = mera.compute_local_expectation_exact(
        {where: op},
        optimize="auto-hq",
        normalized=True,
    )

    assert complex(local_value) == pytest.approx(complex(direct))
    assert chunk.tags == ("I0", "I1")
    assert chunk.physical_width == 2


def test_mera_energy_loss_matches_quimb_exact_sum():
    """MeraEnergyOptimizer.loss() should sum local lightcone contractions."""
    mera = _small_mera(seed=24)
    op = _zz_term()
    terms = {(0, 1): op, (2, 3): op}
    direct = mera.compute_local_expectation_exact(
        terms,
        optimize="auto-hq",
        normalized=True,
    )
    opt = MeraEnergyOptimizer(
        mera,
        terms,
        energy_per_site=False,
        normalized=True,
        contraction_opt="auto-hq",
    )

    assert complex(opt.loss(real=False)) == pytest.approx(complex(direct))


def test_generic_lightcone_energy_groups_select_gate_and_contract():
    """The public fixed-state helper should match the full MERA oracle."""
    mera = _small_mera(seed=241)
    terms = {(0, 1): _zz_term(), (2, 3): _zz_term()}
    direct = mera.compute_local_expectation_exact(
        terms,
        optimize="auto-hq",
        normalized=True,
    )

    value = lightcone_energy(
        mera,
        terms,
        energy_per_site=False,
        normalized=True,
        real=False,
        group_terms=True,
    )

    assert complex(value) == pytest.approx(complex(direct))


def test_qmera_parameter_sharing_per_block_reuses_round_parameters():
    """One block can share parameters across its brickwall rounds."""
    unitary = QMeraUnitarySpec(
        gate_family="rxx",
        parameter_sharing="per-block",
    )
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler=QMeraDisentanglerSpec(
            block_shape=(2, 2),
            unitary=unitary,
            circuit_depth=2,
        ),
        isometry=QMeraIsometrySpec(
            block_shape=(2, 2),
            unitary=unitary,
            circuit_depth=2,
        ),
        max_layers=1,
    )
    layer = builder.build_schedule().layers[0]

    for placements in (layer.disentanglers, layer.isometries):
        by_block = {}
        for placement in placements:
            by_block.setdefault(placement.block, set()).add(placement.param_key)
        assert by_block
        assert all(len(keys) == 1 for keys in by_block.values())
        assert len({next(iter(keys)) for keys in by_block.values()}) == len(by_block)


def test_mera_energy_estimate_reports_lightcone_metadata():
    """energy() should return the shared EnergyEstimate dataclass."""
    mera = _small_mera(seed=25)
    opt = MeraEnergyOptimizer(
        mera,
        {(0, 1): _zz_term()},
        energy_per_site=True,
        normalized=True,
    )

    estimate = opt.energy()

    assert isinstance(estimate, EnergyEstimate)
    assert estimate.num_sites == 8
    assert estimate.boundary_mode == "lightcone-exact"
    assert estimate.energy_per_site == pytest.approx(estimate.energy / 8)
    assert estimate.metadata["num_terms"] == 1
    assert estimate.metadata["max_physical_width"] == 2
    assert estimate.metadata["max_lightcone_tensors"] >= 1


def test_mera_energy_make_tn_optimizer_and_optimize(monkeypatch):
    """TNOptimizer construction should receive loss constants and norm hook."""
    calls = []
    out = _small_mera(seed=27)

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            calls.append((state, loss_fn, kwargs))
            self.losses = [2.0, 1.0]

        def optimize(self, n=220, **kwargs):
            calls.append(("optimize", n, kwargs))
            return out

    monkeypatch.setattr(
        "pepsy.optimizers.mera.optimizer.qtn.TNOptimizer",
        _FakeTNOptimizer,
    )
    mera = _small_mera(seed=26)
    terms = {(0, 1): _zz_term()}
    opt = MeraEnergyOptimizer(mera, terms)

    tnopt = opt.make_tn_optimizer(
        optimizer="lbfgs",
        autodiff_backend="jax",
        progbar=False,
        loss_kwargs={"precompute_tags": False},
    )
    assert isinstance(tnopt, _FakeTNOptimizer)
    _, loss_fn, kwargs = calls[0]
    assert loss_fn is ModuleMeraEnergyOptimizer._tnopt_loss
    assert kwargs["loss_constants"]["terms"] == opt.terms
    assert kwargs["loss_constants"]["chunks"] is None
    assert kwargs["loss_kwargs"]["precompute_tags"] is False
    assert kwargs["optimizer"] == "L-BFGS-B"
    assert kwargs["autodiff_backend"] == "jax"
    assert callable(kwargs["norm_fn"])

    optimized, losses = opt.optimize(n=3, progbar=False, return_losses=True)
    assert optimized is out
    assert opt.state is out
    assert losses == (2.0, 1.0)
    assert calls[-1] == ("optimize", 3, {})


def test_qmera_geometry_explicit_lattice_and_mapper():
    """Geometry should keep physical labels separate from register sites."""
    geom = QMeraGeometry(shape=(2, 3), mapper="snake", boundary="periodic")

    assert geom.shape == (2, 3)
    assert geom.boundary == "periodic"
    assert geom.num_sites == 6
    assert geom.register_sites == (0, 1, 2, 3, 4, 5)
    assert geom.to_register((0, 0)) == 0
    assert geom.to_site(0) == (0, 0)
    assert geom.site_tag((0, 0)) == "I0"
    assert ((0, 2), (0, 0)) in geom.nearest_neighbor_edges()


def test_qmera_geometry_tracks_modes_and_register_order():
    """Fermionic mode labels should be explicit register objects."""
    geom = QMeraGeometry(shape=3, site_modes=("up", "down"))

    assert geom.num_sites == 3
    assert geom.num_modes == 6
    assert geom.register_sites == (0, 1, 2, 3, 4, 5)
    assert geom.register_to_mode == (
        (0, "up"),
        (0, "down"),
        (1, "up"),
        (1, "down"),
        (2, "up"),
        (2, "down"),
    )
    assert geom.mode_label(1, "down") == (1, "down")
    assert geom.mode_register(1, "down") == 3
    assert geom.to_register((1, "down")) == 3
    assert geom.to_register(3) == 3
    assert geom.to_site(3) == 1
    assert geom.to_mode(3) == (1, "down")
    assert geom.modes_on_site(1) == ((1, "up"), (1, "down"))
    assert geom.site_ind((1, "down")) == "k3"
    assert geom.onsite_mode_pairs()[1] == ((1, "up"), (1, "down"))
    assert geom.nearest_neighbor_mode_edges(modes=("up",)) == (
        ((0, "up"), (1, "up")),
        ((1, "up"), (2, "up")),
    )

    mode_major = QMeraGeometry(
        shape=3,
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    assert mode_major.to_register((1, "down")) == 4


def test_qmera_schedule_has_disentangler_and_isometry_blocks():
    """Builder schedules should expose block stages before gate tensors."""
    builder = QMeraBuilder(
        shape=8,
        disentangler={"block_size": 2, "circuit_depth": 1, "gate_family": "rxx"},
        isometry={"block_size": 2, "circuit_depth": 1, "gate_family": "rzz"},
    )

    schedule = builder.build_schedule()
    first = schedule.layers[0]

    assert isinstance(schedule.disentangler, QMeraBlockSpec)
    assert first.input_sites == tuple(range(8))
    assert first.output_sites == (0, 2, 4, 6)
    assert first.isometry_blocks == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert first.disentangler_blocks == ((1, 2), (3, 4), (5, 6))
    assert first.disentanglers
    assert first.isometries
    assert first.disentanglers[0].stage == "disentangler"
    assert first.isometries[0].stage == "isometry"
    assert first.disentanglers[0].where == (1, 2)
    assert first.isometries[0].where == (0, 1)
    assert first.disentanglers[0].gate_family == "rxx"
    assert first.isometries[0].gate_family == "rzz"
    assert "DISENTANGLER" in first.disentanglers[0].tags
    assert "ISOMETRY" in first.isometries[0].tags
    assert schedule.top_sites == (0,)
    assert schedule.num_scales == 3


def test_qmera_periodic_schedule_wraps_boundary_disentangler():
    """Periodic 1D schedules should connect the last and first isometry blocks."""
    builder = QMeraBuilder(
        shape=8,
        boundary="periodic",
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": 2, "circuit_depth": 1},
    )

    first = builder.build_schedule().layers[0]

    assert first.disentangler_blocks[-1] == (7, 0)
    assert first.disentanglers[-1].where == (7, 0)


def test_qmera_2d_schedule_uses_rg_blocks_and_face_disentanglers():
    """2D qMERA should coarse-grain coordinate blocks bottom-to-top."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1, "gate_family": "rxx"},
        isometry={"block_size": (2, 2), "circuit_depth": 2, "gate_family": "rzz"},
    )

    schedule = builder.build_schedule()
    first = schedule.layers[0]

    assert schedule.num_scales == 2
    assert first.isometry_blocks[0] == (0, 1, 4, 5)
    assert first.output_sites == (0, 2, 8, 10)
    assert (4, 5, 8, 9) in first.disentangler_blocks
    assert (1, 5, 2, 6) in first.disentangler_blocks
    assert any(
        placement.axis == "x" and placement.where == (4, 8)
        for placement in first.disentanglers
    )
    assert any(
        placement.axis == "y" and placement.where == (1, 2)
        for placement in first.disentanglers
    )
    assert any("AXIS_X" in placement.tags for placement in first.disentanglers)
    assert any("AXIS_Y" in placement.tags for placement in first.disentanglers)
    assert any(placement.axis == "x" for placement in first.isometries)
    assert any(placement.axis == "y" for placement in first.isometries)


def test_qmera_explicit_specs_build_4x4_periodic_hubbard_schedule():
    """Explicit square layers should include all 4x4 PBC interfaces."""
    backend = QMeraSymmrayFermionBackend(
        symmetry="U1U1",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    unitary = QMeraUnitarySpec(
        gate_family="symmray-hubbard",
        family="fermion",
        arity_kind="mode",
        symmetry="U1U1",
        preserves_parity=True,
        parameter_sharing="per-axis",
        metadata={"model": "fermi-hubbard", "term": "hopping"},
    )
    builder = QMeraBuilder(
        shape=(4, 4),
        boundary="periodic",
        site_modes=backend.site_modes,
        mode_order="mode-major",
        gate_registry=symmray_fermion_gate_registry(backend=backend),
        disentangler=QMeraDisentanglerSpec(
            block_shape=(2, 2),
            unitary=unitary,
            placement="boundary-square",
            circuit_depth=2,
        ),
        isometry=QMeraIsometrySpec(
            block_shape=(2, 2),
            unitary=unitary,
            circuit_depth=2,
        ),
        max_layers=2,
    )

    schedule = builder.build_schedule()
    first, second = schedule.layers

    assert schedule.num_scales == 2
    assert [len(layer.input_sites) for layer in schedule.layers] == [32, 8]
    assert [len(layer.output_sites) for layer in schedule.layers] == [8, 2]
    assert [len(layer.isometry_blocks) for layer in schedule.layers] == [4, 1]
    assert len(first.disentangler_blocks) == 8
    assert not second.disentanglers
    assert schedule.disentangler.placement == "boundary-square"
    assert schedule.disentangler.unitary_spec is unitary
    assert schedule.isometry.implementation == "unitary-completion"
    assert {placement.axis for placement in first.disentanglers} == {"x", "y"}
    assert len(builder.initialize_parameters(schedule)) == len(
        set(schedule.param_keys)
    )

    physical_supports = [
        {geometry_site for geometry_site in map(schedule.geometry.to_site, block)}
        for block in first.disentangler_blocks
    ]
    assert all(
        len({site[0] for site in support}) == 2
        and len({site[1] for site in support}) == 2
        for support in physical_supports
    )
    assert any(
        {site[0] for site in support} == {0, 3}
        for support in physical_supports
    )
    assert any(
        {site[1] for site in support} == {0, 3}
        for support in physical_supports
    )


def test_qmera_explicit_layer_spec_rejects_true_isometry_until_supported():
    """The public API should not silently call a unitary a rectangular isometry."""
    with pytest.raises(NotImplementedError, match="true-isometry"):
        QMeraIsometrySpec(implementation="true-isometry").to_block_spec()


def test_qmera_scale_plan_supports_heterogeneous_6x6_periodic_layers():
    """A scale plan should express 2x2 then 3x3 RG blocks and vertical strips."""
    scale_plan = (
        QMeraScaleSpec(
            name="6x6-to-3x3",
            disentangler=QMeraDisentanglerSpec(
                block_shape=(2, 2),
                placement="boundary-square",
            ),
            isometry=QMeraIsometrySpec(block_shape=(2, 2)),
        ),
        QMeraScaleSpec(
            name="3x3-to-1",
            disentangler=QMeraDisentanglerSpec(
                block_shape=3,
                orientation="vertical",
                placement="within-block",
                circuit_depth=3,
            ),
            isometry=QMeraIsometrySpec(block_shape=(3, 3)),
        ),
    )
    schedule = QMeraBuilder(
        shape=(6, 6),
        boundary="periodic",
        scales=scale_plan,
    ).build_schedule()

    first, second = schedule.layers
    assert [len(layer.input_sites) for layer in schedule.layers] == [36, 9]
    assert [len(layer.output_sites) for layer in schedule.layers] == [9, 1]
    assert [len(layer.isometry_blocks) for layer in schedule.layers] == [9, 1]
    assert len(first.disentangler_blocks) == 18
    assert len(second.disentangler_blocks) == 3
    assert second.disentangler_spec.orientation == "y"
    assert second.disentangler_spec.placement == "within-block"
    assert {placement.axis for placement in second.disentanglers} == {"y"}
    assert any(
        {schedule.geometry.to_site(site)[1] for site in placement.where} == {0, 4}
        for placement in second.disentanglers
    )
    assert [scale.name for scale in schedule.scale_specs] == [
        "6x6-to-3x3",
        "3x3-to-1",
    ]


def test_qmera_1d_mode_geometry_schedules_register_modes():
    """1D qMERA schedules should operate on mode/register positions."""
    builder = QMeraBuilder(
        shape=2,
        site_modes=("up", "down"),
        gate_family="fsim",
        isometry_gate_family="fsim",
        max_layers=1,
    )

    schedule = builder.build_schedule()
    first = schedule.layers[0]
    chunks = builder.parametric_lightcone_chunks(
        {((0, "up"), (0, "down")): _zz_term()},
        schedule,
    )
    ansatz = builder.build()

    assert schedule.geometry.num_sites == 2
    assert schedule.geometry.num_modes == 4
    assert first.input_sites == (0, 1, 2, 3)
    assert first.isometry_blocks == ((0, 1), (2, 3))
    assert schedule.geometry.to_register_where(((0, "up"), (0, "down"))) == (0, 1)
    assert chunks[0].term.where == (0, 1)
    assert chunks[0].term.metadata["original_where"] == ((0, "up"), (0, "down"))
    assert ansatz.metadata["num_sites"] == 2
    assert ansatz.metadata["num_modes"] == 4
    assert ansatz.metadata["site_modes"] == ("up", "down")
    assert ansatz.state.num_tensors >= schedule.geometry.num_modes
    assert "I3" in ansatz.state.tags


def test_qmera_1d_mode_schedule_never_pairs_different_fermion_modes():
    """1D brickwall layers should preserve each explicit mode flavor."""
    builder = QMeraBuilder(
        shape=4,
        site_modes=("up", "down"),
        mode_order="mode-major",
        gate_family="fsim",
        isometry_gate_family="fsim",
        max_layers=2,
    )
    schedule = builder.build_schedule()

    for placement in schedule.placements:
        modes = {
            schedule.geometry.to_mode(register_site)[1]
            for register_site in placement.where
        }
        assert len(modes) == 1


def test_qmera_symmray_fermion_backend_builds_native_hubbard_terms():
    """qMERA Hubbard terms should be native Symmray fermionic mode arrays."""
    pytest.importorskip("symmray")
    geometry = QMeraGeometry(shape=3, site_modes=("up", "down"))
    backend = QMeraSymmrayFermionBackend()

    terms = qmera_symmray_fermi_hubbard_terms(
        geometry,
        backend=backend,
        t=0.5,
        U=4.0,
        mu=0.0,
    )
    kinds = [term.metadata["kind"] for term in terms]

    assert kinds.count("hubbard-onsite") == 3
    assert kinds.count("hubbard-hopping") == 4
    assert all(term.metadata["backend"] == "symmray" for term in terms)
    assert all(term.metadata["fermionic"] is True for term in terms)
    assert all("FermionicArray" in type(term.operator).__name__ for term in terms)
    assert terms[0].where == ((0, "up"), (0, "down"))
    assert terms[-1].where == ((1, "down"), (2, "down"))
    with pytest.raises(ValueError, match="spin-changing"):
        backend.hopping_operator((0, "up"), (0, "down"))


def test_unified_fermion_helper_adapts_to_qmera_mode_terms():
    """One Fermion model should feed both site and qMERA energy layouts."""
    pytest.importorskip("symmray")
    geometry = QMeraGeometry(shape=3, site_modes=("up", "down"))
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=0.5,
        U=4.0,
        mu=0.1,
    )

    site_terms = fermion.local_terms(((0, 1), (1, 2)))
    mode_terms = fermion.local_terms(geometry, layout="qmera")
    direct_terms = qmera_symmray_fermi_hubbard_terms(
        geometry,
        fermion=fermion,
    )

    assert set(site_terms) == {(0, 1), (1, 2)}
    assert len(mode_terms) == len(direct_terms) == 13
    assert [term.metadata["kind"] for term in mode_terms] == [
        term.metadata["kind"] for term in direct_terms
    ]
    assert all("FermionicArray" in type(term.operator).__name__ for term in mode_terms)
    assert mode_terms[0].where == ((0, "up"), (0, "down"))
    assert all(term.metadata["backend"] == "symmray" for term in mode_terms)


def test_unified_fermion_qmera_optimizer_runs_torch_autodiff():
    """The builder convenience should run a native qMERA energy step."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")
    from pepsy.backends import backend_torch

    array_backend = backend_torch(dtype=torch.complex128)
    backend = QMeraSymmrayFermionBackend(to_backend=array_backend)
    registry = symmray_fermion_gate_registry(backend=backend)
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=0.2,
        U=0.5,
        mu=0.1,
    )

    def product_state_factory(schedule, sites, **kwargs):
        return backend.product_state(
            schedule,
            sites,
            occupations={0: 1, 1: 0, 2: 0, 3: 1},
            **kwargs,
        )

    builder = QMeraBuilder(
        shape=2,
        fermion=fermion,
        gate_registry=registry,
        array_backend=array_backend,
        disentangler={"block_size": 2, "circuit_depth": 0},
        isometry={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=44,
        param_scale=0.01,
        product_state_factory=product_state_factory,
    )
    terms = builder.fermion_terms()
    optimizer = builder.fermion_parametric_optimizer(
        energy_per_site=False,
    )
    initial = optimizer.loss(energy_per_site=False)
    result = optimizer.run(
        solver="torch-adam",
        n_steps=1,
        log_every=1,
        options={"lr": 0.01},
    )

    assert len(terms) == 8
    assert np.isfinite(float(initial))
    assert len(result.history) == 1
    assert np.isfinite(float(result.history[-1]))


def test_qmera_symmray_fsim_runs_full_fermionic_lightcone():
    """Symmray qMERA gates, state, and terms should share one fermion path."""
    pytest.importorskip("symmray")
    backend = QMeraSymmrayFermionBackend()
    registry = symmray_fermion_gate_registry(backend=backend)
    spec = registry.get("symmray-fsim")

    assert spec.is_fermionic
    assert spec.arity_kind == "mode"
    assert spec.preserves_parity is True
    with pytest.raises(ValueError, match="placement context"):
        spec.matrix([0.0, 0.0])

    builder = QMeraBuilder(
        shape=2,
        site_modes=backend.site_modes,
        mode_order="mode-major",
        disentangler={"block_size": 2, "circuit_depth": 0},
        isometry={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        gate_registry=registry,
        max_layers=1,
        seed=43,
        param_scale=0.05,
        product_state_factory=backend.product_state,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    gate_tensors = builder.gate_tensors(params, schedule)
    ansatz = builder.build(params)
    terms = backend.fermi_hubbard_terms(
        schedule.geometry,
        t=0.5,
        U=0.0,
        mu=0.0,
    )
    chunks = build_qmera_parametric_lightcone_chunks(schedule, terms[:1])
    lightcone = builder.parametric_lightcone_tn(
        chunks[0],
        params,
        schedule=schedule,
    )
    value = contract_qmera_lightcone_tn(
        lightcone,
        optimize="auto-hq",
        real=False,
    )

    assert schedule.geometry.register_to_mode == (
        (0, "up"),
        (1, "up"),
        (0, "down"),
        (1, "down"),
    )
    assert schedule.layers[0].isometries[0].where == (0, 1)
    assert all(
        "FermionicArray" in type(gate).__name__
        for gate in gate_tensors.values()
    )
    assert all(
        "FermionicArray" in type(tensor.data).__name__
        for tensor in ansatz.state
    )
    assert chunks[0].term.where == (0, 1)
    assert all(
        "FermionicArray" in type(tensor.data).__name__
        for tensor in lightcone.ket
    )
    assert complex(value) == pytest.approx(0.0)

    single_site_builder = QMeraBuilder(
        shape=1,
        site_modes=backend.site_modes,
        gate_registry=registry,
        isometry={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
    )
    single_site_schedule = single_site_builder.build_schedule()
    assert not single_site_schedule.layers
    assert not single_site_schedule.placements
    assert single_site_builder.gate_tensors(
        single_site_builder.initialize_parameters(single_site_schedule),
        single_site_schedule,
    ) == {}


def test_qmera_symmray_fermion_lightcone_contracts_native_term():
    """Scheduled qMERA lightcones should consume Symmray fermionic TNs."""
    pytest.importorskip("symmray")
    backend = QMeraSymmrayFermionBackend()
    builder = QMeraBuilder(
        shape=2,
        site_modes=backend.site_modes,
        max_layers=0,
        product_state_factory=backend.product_state,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    terms = backend.fermi_hubbard_terms(
        schedule.geometry,
        t=0.0,
        U=4.0,
        mu=0.0,
    )
    chunks = build_qmera_parametric_lightcone_chunks(schedule, terms[:1])

    lightcone = builder.parametric_lightcone_tn(
        chunks[0],
        params,
        schedule=schedule,
    )
    value = contract_qmera_lightcone_tn(
        lightcone,
        optimize="auto-hq",
        real=False,
    )
    builder_value = builder.parametric_loss(
        params,
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        real=False,
    )
    fixed_state_value = lightcone_energy(
        builder.build(params),
        terms[:1],
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
        real=False,
    )

    assert isinstance(lightcone, QMeraLightconeTN)
    assert lightcone.ket.num_tensors == 2
    assert all(
        "FermionicArray" in type(tensor.data).__name__
        for tensor in lightcone.ket
    )
    assert complex(value) == pytest.approx(0.0)
    assert complex(builder_value) == pytest.approx(complex(value))
    assert complex(fixed_state_value) == pytest.approx(complex(value))


def test_qmera_2d_multi_mode_schedule_retains_modes_and_pairs_like_modes():
    """2D RG blocks should retain modes and never pair different flavors."""
    builder = QMeraBuilder(
        shape=(2, 2),
        site_modes=("up", "down"),
        mode_order="mode-major",
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 1},
        max_layers=1,
    )

    schedule = builder.build_schedule()
    first = schedule.layers[0]

    assert first.isometry_blocks[0] == (0, 4, 1, 5, 2, 6, 3, 7)
    assert first.output_sites == (0, 4)
    for placement in (*first.disentanglers, *first.isometries):
        modes = [schedule.geometry.to_mode(site)[1] for site in placement.where]
        assert len(set(modes)) == 1


def test_qmera_2d_multimode_rg_keeps_populated_axis_after_coarse_graining():
    """An anisotropic coarse grid should still receive its final isometry."""
    builder = QMeraBuilder(
        shape=(2, 4),
        site_modes=("up", "down"),
        mode_order="mode-major",
        isometry={"block_size": (2, 2), "circuit_depth": 1},
        max_layers=2,
    )

    schedule = builder.build_schedule()

    assert len(schedule.layers) == 2
    assert schedule.layers[-1].isometries
    assert schedule.layers[-1].isometries[0].axis == "y"


def test_fermion_majorana_convention_is_z2_and_parity_preserving():
    """Majoranas are odd Z2 operators; bilinears and gates are neutral."""
    pytest.importorskip("symmray")
    majorana = Fermion(spinful=False, symmetry="Z2")

    gamma_x = majorana.majorana_operator("x", site=0)
    gamma_y = majorana.majorana_operator("y", site=0)
    bilinear = majorana.majorana_bilinear_operator(
        (0, 1),
        left_component="y",
        right_component="x",
    )
    pairing = majorana.pairing_operator((0, 1), phase=0.25)
    gates = (
        majorana.majorana_gate(0.1, edge=(0, 1)),
        majorana.pairing_gate(0.1, edge=(0, 1), phase=0.25),
    )

    assert gamma_x.charge == 1
    assert gamma_y.charge == 1
    assert bilinear.charge == 0
    assert pairing.charge == 0
    gamma_x_dense = np.asarray(gamma_x.to_dense())
    gamma_y_dense = np.asarray(gamma_y.to_dense())
    np.testing.assert_allclose(gamma_x_dense @ gamma_x_dense, np.eye(2))
    np.testing.assert_allclose(gamma_y_dense @ gamma_y_dense, np.eye(2))
    np.testing.assert_allclose(
        gamma_x_dense @ gamma_y_dense + gamma_y_dense @ gamma_x_dense,
        np.zeros((2, 2)),
    )
    for operator in (bilinear, pairing):
        dense = np.asarray(operator.to_dense())
        matrix = dense.reshape((4, 4))
        np.testing.assert_allclose(matrix, matrix.conj().T)
    for gate in gates:
        dense = np.asarray(gate.to_dense()).reshape((4, 4))
        np.testing.assert_allclose(dense.conj().T @ dense, np.eye(4), atol=1.0e-12)
    assert all("Z2FermionicArray" in type(value).__name__ for value in (
        gamma_x,
        gamma_y,
        bilinear,
        pairing,
        *gates,
    ))
    with pytest.raises(ValueError, match="requires symmetry='Z2'"):
        Fermion(spinful=False, symmetry="U1").majorana_operator()


def test_qmera_2d_fermion_and_majorana_direct_oracles_match_lightcones():
    """Native graded 2D Hubbard and Majorana paths agree with direct TNs."""
    pytest.importorskip("symmray")

    def run_case(geometry, backend, registry, fermion, terms, gate_family):
        def product_state_factory(schedule, sites, **kwargs):
            occupations = {
                site: int(
                    (sum(schedule.geometry.to_site(site)) % 2 == 0)
                    == (schedule.geometry.to_mode(site)[-1] == "up")
                )
                for site in sites
            }
            return backend.product_state(
                schedule,
                sites,
                occupations=occupations,
                **kwargs,
            )

        builder = QMeraBuilder(
            geometry=geometry,
            gate_registry=registry,
            gate_family=gate_family,
            disentangler={
                "block_size": 2,
                "circuit_depth": 1,
                "gate_family": gate_family,
            },
            isometry={
                "block_size": (2, 2),
                "circuit_depth": 1,
                "gate_family": gate_family,
            },
            max_layers=1,
            seed=91,
            param_scale=0.01,
            product_state_factory=product_state_factory,
        )
        schedule = builder.build_schedule()
        parameters = builder.initialize_parameters(schedule)
        lightcone = builder.parametric_loss(
            parameters,
            terms,
            schedule=schedule,
            convert_terms=False,
            energy_per_site=False,
            real=False,
        )
        direct = builder.direct_parametric_loss(
            parameters,
            terms,
            schedule=schedule,
            convert_terms=False,
            energy_per_site=False,
            real=False,
        )
        return lightcone, direct

    hubbard_geometry = QMeraGeometry(
        shape=(2, 2),
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    hubbard_backend = QMeraSymmrayFermionBackend()
    hubbard_registry = symmray_fermion_gate_registry(backend=hubbard_backend)
    hubbard = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=0.2,
        U=0.5,
        mu=0.1,
    )
    hubbard_terms = qmera_symmray_fermi_hubbard_terms(
        hubbard_geometry,
        fermion=hubbard,
    )
    hubbard_lightcone, hubbard_direct = run_case(
        hubbard_geometry,
        hubbard_backend,
        hubbard_registry,
        hubbard,
        hubbard_terms,
        "symmray-fsim",
    )

    majorana_geometry = QMeraGeometry(
        shape=(2, 2),
        site_modes=("mode",),
        mode_order="mode-major",
    )
    majorana_backend = QMeraSymmrayFermionBackend(
        symmetry="Z2",
        site_modes=("mode",),
    )
    majorana_registry = symmray_majorana_gate_registry(backend=majorana_backend)
    majorana = Fermion(spinful=False, symmetry="Z2")
    majorana_terms = qmera_symmray_majorana_terms(
        majorana_geometry,
        fermion=majorana,
        coupling=0.4,
        pairing=0.2,
    )
    majorana_lightcone, majorana_direct = run_case(
        majorana_geometry,
        majorana_backend,
        majorana_registry,
        majorana,
        majorana_terms,
        "symmray-majorana",
    )

    assert complex(hubbard_lightcone) == pytest.approx(complex(hubbard_direct))
    assert complex(majorana_lightcone) == pytest.approx(complex(majorana_direct))


def test_qmera_grouped_and_direct_energy_match_schedule_lightcones():
    """Grouping and the full direct-gate oracle must preserve local energy."""
    builder = QMeraBuilder(shape=8, seed=12, param_scale=0.02)
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    terms = {(0, 1): _zz_term(), (2, 3): _zz_term(), (4, 5): _zz_term()}
    chunks = builder.parametric_lightcone_chunks(terms, schedule)
    groups = group_qmera_parametric_lightcone_chunks(chunks)

    assert all(isinstance(group, QMeraLightconeGroup) for group in groups)
    assert sum(group.num_terms for group in groups) == len(chunks)
    local = builder.parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        energy_per_site=False,
    )
    direct = qmera_direct_parametric_energy(
        schedule,
        parameters,
        terms,
        energy_per_site=False,
    )
    assert complex(local) == pytest.approx(complex(direct))


def test_qmera_contraction_path_cache_reuses_topology_optimizer(monkeypatch):
    """Path caches should lazily create one reusable optimizer per topology."""
    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "pepsy.optimizers.mera.cache.build_qmera_contraction_optimizer",
        fake_builder,
    )
    cache = QMeraContractionPathCache({"directory": False})
    first = cache.optimizer_for(("cone",))
    second = cache.optimizer_for(("cone",))
    other = cache.resolve("auto-hq", key=("other",))

    assert first is second
    assert other is not first
    assert cache.num_cached_paths == 2
    assert calls == [{"directory": False}, {"directory": False}]


def test_qmera_gate_registry_generates_parametrized_two_qubit_gates():
    """The registry should produce backend-ready two-qubit gate tensors."""
    registry = default_gate_registry()
    spec = registry.get("rxx")
    gate = spec.matrix([0.25])

    assert isinstance(spec, GateSpec)
    assert spec.arity == 2
    assert spec.arity_kind == "qubit"
    assert spec.family == "spin"
    assert not spec.is_fermionic
    assert spec.num_params == 1
    assert gate.shape == (2, 2, 2, 2)
    assert "su4" in registry.names(arity=2)

    fsim_spec = registry.get("fsim")
    assert fsim_spec.is_fermionic
    assert fsim_spec.arity_kind == "mode"
    assert fsim_spec.preserves_parity is True
    assert fsim_spec.mode_order == "register"
    assert "fsim" in registry.names(family="fermion", arity_kind="mode")

    custom = UserGateFamily(
        name="diag-phase",
        arity=2,
        num_params=1,
        generator=lambda params: np.diag(
            [1.0, 1.0, 1.0, np.exp(1.0j * params[0])]
        ).reshape(2, 2, 2, 2),
        family="fermion",
        arity_kind="mode",
        preserves_parity=True,
        mode_order="site-major",
    )
    registry.register(custom)
    custom_spec = registry.get("diag-phase")
    custom_gate = custom_spec.matrix([0.3])
    assert custom_spec.family == "fermion"
    assert custom_spec.arity_kind == "mode"
    assert custom_spec.mode_order == "site-major"
    assert custom_gate.shape == (2, 2, 2, 2)
    assert custom_gate.reshape(4, 4)[-1, -1] == pytest.approx(np.exp(0.3j))


def test_qmera_builder_outputs_parameters_gates_state_and_lightcone_tags():
    """QMeraBuilder should assemble a deterministic direct-gate TN payload."""
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=7,
        param_scale=0.0,
    )

    ansatz = builder.build()

    assert ansatz.state is not None
    assert ansatz.metadata["state_kind"] == "direct-gate-tn"
    assert ansatz.metadata["num_layers"] == 2
    assert tuple(ansatz.parameters) == ansatz.schedule.param_keys
    assert set(ansatz.gate_tensors) == {
        placement.gate_id for placement in ansatz.schedule.placements
    }
    assert all(value.shape[0] in {0, 1} for value in ansatz.parameters.values())

    tags = ansatz.reverse_lightcone_tags((0, 1))
    assert "I0" in tags
    assert "I1" in tags
    assert any(tag.startswith("GATE_L0_DIS") for tag in tags)
    assert "DISENTANGLER" in ansatz.state.tags
    assert "ISOMETRY" in ansatz.state.tags

    opt = MeraEnergyOptimizer(
        ansatz.state,
        {(0, 1): _zz_term()},
        energy_per_site=False,
        contraction_opt="auto-hq",
    )
    assert np.isfinite(float(opt.loss()))


def test_qmera_schedule_lightcone_chunks_follow_placements():
    """Schedule chunks should select the RG reverse cone, not only site tags."""
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=19,
        param_scale=0.1,
    )
    ansatz = builder.build()
    op = _zz_term()

    chunks = build_qmera_lightcone_chunks(
        ansatz.state,
        ansatz.schedule,
        normalize_local_terms({(0, 1): op}),
    )
    opt = MeraEnergyOptimizer(
        ansatz,
        {(0, 1): op},
        energy_per_site=False,
        contraction_opt="auto-hq",
    )
    direct = ansatz.state.compute_local_expectation_exact(
        {(0, 1): op},
        optimize="auto-hq",
        normalized=True,
    )
    expected_ids = tuple(
        placement.gate_id
        for placement in ansatz.schedule.reverse_lightcone_placements((0, 1))
    )

    assert opt.schedule is ansatz.schedule
    assert opt.lightcones[0].tags == chunks[0].tags
    assert opt.lightcones[0].schedule_placement_ids == chunks[0].schedule_placement_ids
    assert chunks[0].source == "schedule"
    assert chunks[0].schedule_placement_ids == expected_ids
    assert any(tag.startswith("GATE_L0_DIS") for tag in chunks[0].tags)
    assert chunks[0].schedule_width >= chunks[0].support_size
    assert complex(opt.loss(real=False)) == pytest.approx(complex(direct))
    assert opt.energy().metadata["lightcone_sources"] == ("schedule",)


def test_qmera_schedule_lightcone_chunks_map_coordinate_terms():
    """Coordinate-lattice terms should compile to qMERA register supports."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 2},
        seed=20,
        param_scale=0.03,
    )
    ansatz = builder.build()
    op = _zz_term()

    opt = MeraEnergyOptimizer(
        ansatz,
        {((0, 0), (0, 1)): op},
        energy_per_site=False,
        contraction_opt="auto-hq",
    )
    chunk = opt.lightcones[0]
    direct = ansatz.state.compute_local_expectation_exact(
        {(0, 1): op},
        optimize="auto-hq",
        normalized=True,
    )

    assert chunk.source == "schedule"
    assert chunk.term.where == (0, 1)
    assert chunk.term.metadata["original_where"] == ((0, 0), (0, 1))
    assert chunk.term.metadata["register_where"] == (0, 1)
    assert chunk.schedule_placement_ids
    assert chunk.schedule_width_by_scale
    assert complex(opt.loss(real=False)) == pytest.approx(complex(direct))


def test_qmera_parametric_lightcone_loss_rebuilds_local_cone_from_params():
    """Parameterized qMERA loss should not need a prebuilt global state."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1, "gate_family": "rxx"},
        isometry={"block_size": (2, 2), "circuit_depth": 2, "gate_family": "rzz"},
        seed=31,
        param_scale=0.04,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    hamiltonian = {((0, 0), (0, 1)): _zz_term()}

    chunks = builder.parametric_lightcone_chunks(hamiltonian, schedule)
    value = builder.parametric_loss(
        params,
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        real=False,
        contraction_opt="auto-hq",
    )
    state, _ = builder.build_state(params, schedule)
    direct = state.compute_local_expectation_exact(
        {(0, 1): _zz_term()},
        optimize="auto-hq",
        normalized=True,
    )

    assert isinstance(chunks[0], QMeraParametricLightconeChunk)
    assert chunks[0].source == "parametric-schedule"
    assert chunks[0].term.where == (0, 1)
    assert chunks[0].num_gates < schedule.num_gates
    assert complex(value) == pytest.approx(complex(direct))


def test_qmera_parametric_energy_helpers_match_builder_loss():
    """Low-level parametric helpers should compose with schedule/chunk metadata."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 2},
        seed=32,
        param_scale=0.02,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    terms = normalize_local_terms({((0, 0), (0, 1)): _zz_term()})
    chunks = build_qmera_parametric_lightcone_chunks(schedule, terms)

    local_state = qmera_parametric_lightcone_state(
        schedule,
        chunks[0],
        params,
        gate_registry=builder.gate_registry,
    )
    lightcone_tn = qmera_parametric_lightcone_tn(
        schedule,
        chunks[0],
        params,
        gate_registry=builder.gate_registry,
    )
    builder_lightcone_tn = builder.parametric_lightcone_tn(
        chunks[0],
        params,
        schedule=schedule,
    )
    tn_value = contract_qmera_lightcone_tn(
        lightcone_tn,
        optimize="auto-hq",
        real=False,
    )
    local_value = local_qmera_parametric_lightcone_expectation(
        schedule,
        chunks[0],
        params,
        gate_registry=builder.gate_registry,
        optimize="auto-hq",
        real=False,
    )
    energy_value = qmera_parametric_energy(
        schedule,
        params,
        chunks=chunks,
        gate_registry=builder.gate_registry,
        energy_per_site=False,
        optimize="auto-hq",
        real=False,
    )
    builder_value = builder.parametric_loss(
        params,
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        real=False,
        contraction_opt="auto-hq",
    )

    assert local_state.num_tensors == len(chunks[0].input_sites) + chunks[0].num_gates
    assert isinstance(lightcone_tn, QMeraLightconeTN)
    assert isinstance(builder_lightcone_tn, QMeraLightconeTN)
    assert lightcone_tn.num_gates == chunks[0].num_gates
    assert builder_lightcone_tn.num_gates == lightcone_tn.num_gates
    assert lightcone_tn.num_numerator_tensors > lightcone_tn.ket.num_tensors
    assert lightcone_tn.num_denominator_tensors == 2 * lightcone_tn.ket.num_tensors
    assert complex(tn_value) == pytest.approx(complex(local_value))
    assert complex(local_value) == pytest.approx(complex(energy_value))
    assert complex(energy_value) == pytest.approx(complex(builder_value))


def test_qmera_compiled_parametric_loss_matches_rebuilt_cone():
    """Compiled qMERA expressions should match the local TN rebuild path."""
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=34,
        param_scale=0.07,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    terms = normalize_local_terms({(0, 1): _zz_term()})
    chunks = build_qmera_parametric_lightcone_chunks(schedule, terms)
    compiled = compile_qmera_parametric_lightcones(
        schedule,
        chunks,
        gate_registry=builder.gate_registry,
    )

    local_value = local_qmera_compiled_lightcone_expectation(
        schedule,
        compiled[0],
        params,
        gate_registry=builder.gate_registry,
        real=False,
    )
    compiled_value = qmera_compiled_parametric_energy(
        schedule,
        params,
        compiled_chunks=compiled,
        gate_registry=builder.gate_registry,
        energy_per_site=False,
        real=False,
    )
    builder_value = builder.compiled_parametric_loss(
        params,
        schedule=schedule,
        compiled_chunks=compiled,
        energy_per_site=False,
        real=False,
    )
    rebuilt_value = builder.parametric_loss(
        params,
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        real=False,
    )

    assert isinstance(compiled[0], QMeraCompiledLightconeChunk)
    assert compiled[0].num_gates == chunks[0].num_gates
    assert compiled[0].num_numerator_tensors >= compiled[0].num_denominator_tensors
    assert complex(local_value) == pytest.approx(complex(rebuilt_value))
    assert complex(compiled_value) == pytest.approx(complex(rebuilt_value))
    assert complex(builder_value) == pytest.approx(complex(rebuilt_value))


def test_qmera_parametric_loss_reports_missing_gate_parameter():
    """A missing parameter key should fail at the scheduled gate that needs it."""
    builder = QMeraBuilder(shape=4, gate_family="rxx", isometry_gate_family="rzz")
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    missing_key = schedule.placements[0].param_key
    params.pop(missing_key)

    with pytest.raises(KeyError, match=missing_key):
        builder.parametric_loss(
            params,
            {(0, 1): _zz_term()},
            schedule=schedule,
            energy_per_site=False,
        )


def test_qmera_builder_cast_params_to_torch_backend():
    """qMERA parameter dicts should cast through Pepsy backend helpers."""
    torch = pytest.importorskip("torch")
    builder = QMeraBuilder(shape=4)
    params = {"theta": np.array([0.25], dtype=np.float64)}

    trainable = builder.cast_params(
        params,
        backend="torch",
        trainable=True,
        dtype=torch.float64,
    )
    frozen = builder.cast_params(
        params,
        backend="torch",
        trainable=False,
        dtype=torch.float64,
    )

    assert isinstance(trainable["theta"], torch.Tensor)
    assert trainable["theta"].requires_grad
    assert trainable["theta"].dtype == torch.float64
    assert not frozen["theta"].requires_grad


def test_qmera_parametric_optimizer_runs_compiled_torch_solver():
    """The parametric optimizer shell should expose compiled loss to solvers."""
    pytest.importorskip("torch")
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=33,
        param_scale=0.1,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    opt = builder.parametric_optimizer(
        {(0, 1): _zz_term()},
        schedule=schedule,
        parameters=params,
        energy_per_site=False,
        real=True,
        contraction_opt="auto-hq",
    )

    initial = opt.loss_fn()(params)
    result = opt.run(
        solver="torch-adam",
        n_steps=2,
        log_every=1,
        options={"lr": 0.05},
        compiled=True,
    )

    assert isinstance(opt, QMeraParametricEnergyOptimizer)
    assert np.isfinite(float(initial))
    assert result.solver == "torch-adam"
    assert len(result.history) == 2
    assert result.history[-1] <= result.history[0]
    assert opt.compiled_chunks
    assert opt.losses == result.history
    assert set(result.params) == set(params)


def test_qmera_compiled_parametric_loss_jax_jit_smoke():
    """Compiled local-cone loss should be JAX-jittable over params."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from pepsy.backends import backend_jax

    jax.config.update("jax_enable_x64", True)
    to_array = backend_jax(dtype=jnp.complex128)
    to_param = backend_jax(dtype=jnp.float64)
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=35,
        param_scale=0.05,
        parameter_backend=to_param,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    compiled = builder.compile_parametric_lightcones(
        {(0, 1): _zz_term()},
        schedule,
        array_backend=to_array,
    )
    loss_fn = builder.compiled_parametric_loss_fn(
        schedule=schedule,
        compiled_chunks=compiled,
        energy_per_site=False,
        real=True,
    )

    value = jax.jit(loss_fn)(params)
    grads = jax.grad(loss_fn)(params)

    assert np.isfinite(float(value))
    assert set(grads) == set(params)


def test_qmera_2d_builder_outputs_direct_gate_tensor_network():
    """A 2D RG schedule should feed the direct-gate tensor-network builder."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 2},
        param_scale=0.0,
    )

    ansatz = builder.build()

    assert ansatz.metadata["state_kind"] == "direct-gate-tn"
    assert ansatz.metadata["num_layers"] == 2
    assert ansatz.metadata["num_gates"] == 28
    assert "DISENTANGLER" in ansatz.state.tags
    assert "ISOMETRY" in ansatz.state.tags
    assert "AXIS_X" in ansatz.state.tags
    assert "AXIS_Y" in ansatz.state.tags


def test_qmera_schematic_blocks_group_disentanglers_and_isometries():
    """Schematic data should expose visible block groupings per layer."""
    builder = QMeraBuilder(
        shape=8,
        disentangler={"block_size": 4, "circuit_depth": 2, "gate_family": "rxx"},
        isometry={"block_size": 2, "circuit_depth": 1, "gate_family": "rzz"},
    )
    schedule = builder.build_schedule()

    blocks = schedule.schematic_blocks(layer=0)
    direct_blocks = qmera_schematic_blocks(schedule, layer=0)

    assert blocks == direct_blocks
    assert all(isinstance(block, QMeraSchematicBlock) for block in blocks)
    assert {block.stage for block in blocks} == {"disentangler", "isometry"}
    assert {block.stage_label for block in blocks} == {"D", "W"}
    assert any(block.register_sites == (0, 1, 2, 3) for block in blocks)
    assert any(
        block.stage == "disentangler" and block.register_sites == (2, 3, 4, 5)
        for block in blocks
    )
    assert any(block.round == 1 for block in blocks if block.stage == "disentangler")


def test_qmera_2d_schematic_blocks_expose_coordinate_rg_windows():
    """2D schematic blocks should show full isometry and boundary windows."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 2},
    )
    schedule = builder.build_schedule()

    blocks = schedule.schematic_blocks(layer=0)

    assert any(
        block.stage == "isometry"
        and block.round == 0
        and block.register_sites == (0, 1, 4, 5)
        and block.sites == ((0, 0), (0, 1), (1, 0), (1, 1))
        and block.axis == "x"
        for block in blocks
    )
    assert any(
        block.stage == "isometry"
        and block.round == 1
        and block.register_sites == (0, 1, 4, 5)
        and block.axis == "y"
        for block in blocks
    )
    assert any(
        block.stage == "disentangler"
        and block.register_sites == (4, 5, 8, 9)
        and block.axis == "x"
        for block in blocks
    )
    assert any(
        block.stage == "disentangler"
        and block.register_sites == (1, 5, 2, 6)
        and block.axis == "y"
        for block in blocks
    )


def test_qmera_draw_schematic_builds_quimb_drawing():
    """The quimb schematic wrapper should draw layer blocking without a Circuit."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from quimb import schematic

    builder = QMeraBuilder(shape=4, gate_family="rxx", isometry_gate_family="rzz")
    schedule = builder.build_schedule()

    drawing = draw_qmera_schedule(
        schedule,
        layer=0,
        label_sites=False,
        scale_figsize=False,
    )

    assert isinstance(drawing, schematic.Drawing)
    assert isinstance(builder.draw_schematic(layer=0, scale_figsize=False), schematic.Drawing)
    assert isinstance(
        builder.build(build_state=False).draw_schematic(layer=0, scale_figsize=False),
        schematic.Drawing,
    )


def test_qmera_2d_draw_schematic_builds_quimb_drawing():
    """The schematic wrapper should handle 2D RG blocking."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from quimb import schematic

    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 2},
    )

    drawing = builder.draw_schematic(
        layer=0,
        label_sites=False,
        scale_figsize=False,
    )

    assert isinstance(drawing, schematic.Drawing)
    clean_drawing = builder.draw_schematic(
        layer=0,
        style="clean",
        label_sites=False,
        scale_figsize=False,
    )
    register_drawing = builder.draw_schematic(
        layer=0,
        style="register",
        label_sites=False,
        scale_figsize=False,
    )
    assert isinstance(clean_drawing, schematic.Drawing)
    assert isinstance(register_drawing, schematic.Drawing)
    with pytest.raises(ValueError, match="style"):
        builder.draw_schematic(layer=0, style="unknown")
