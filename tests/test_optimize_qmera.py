"""Tests for qMERA local-energy optimization helpers."""

import numpy as np
import pytest

from pepsy.optimizers.qmera import (
    GateSpec,
    LocalTerm,
    QMeraBlockSpec,
    QMeraBuilder,
    QMeraCompiledLightconeChunk,
    QMeraContractionPathCache,
    QMeraDisentanglerSpec,
    QMeraEnergyOptimizer,
    QMeraGeometry,
    QMeraIsometrySpec,
    QMeraLayoutFinder,
    QMeraLightconeGroup,
    QMeraLightconeTN,
    QMeraSchematicBlock,
    QMeraParametricEnergyOptimizer,
    QMeraParametricLightconeChunk,
    QMeraPrototypeLayout,
    QMeraSymmrayFermionBackend,
    QMeraScaleSpec,
    QMeraUnitarySpec,
    UserGateFamily,
    build_qmera_contraction_optimizer,
    build_qmera_parametric_lightcone_chunks,
    compile_qmera_parametric_lightcones,
    contract_qmera_lightcone_tn,
    group_qmera_parametric_lightcone_chunks,
    default_gate_registry,
    draw_qmera_schedule,
    local_qmera_compiled_lightcone_expectation,
    local_qmera_parametric_lightcone_expectation,
    load_qmera_prototype_layout,
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
from pepsy.tensors import Fermion


def _zz_term():
    z_op = np.diag([1.0, -1.0])
    return np.kron(z_op, z_op).reshape(2, 2, 2, 2)


def test_qmera_contraction_optimizer_helper_delegates(monkeypatch):
    """qMERA cache helper should expose Pepsy's reusable cotengra builder."""
    calls = []

    def fake_build_optimizer(**kwargs):
        calls.append(kwargs)
        return "optimizer"

    monkeypatch.setattr(
        "pepsy.optimizers.qmera.cache.build_optimizer",
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
        "pepsy.optimizers.qmera.builders.build_qmera_contraction_optimizer",
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


def test_qmera_layout_finder_returns_valid_ranked_candidates():
    """Layout search should rank immutable schedule candidates."""
    geometry = QMeraGeometry(shape=8)
    finder = QMeraLayoutFinder(
        geometry,
        isometry_block_shapes=(2, 4),
        disentangler_block_shapes=(2,),
        disentangler_depths=(1, 2),
        isometry_depths=(1,),
        max_layers=4,
    )

    report = finder.search({(0, 1): _zz_term()})
    repeat = finder.search({(0, 1): _zz_term()})

    assert report.candidates
    assert report.scores
    assert report.best is not None
    assert report.best_score is not None
    assert report.pareto_front
    assert tuple(candidate.candidate_id for candidate in report.candidates) == tuple(
        candidate.candidate_id for candidate in repeat.candidates
    )
    assert all(score.valid for score in report.scores)
    assert all("max_lightcone_width" in score.components for score in report.scores)


def test_qmera_prototype_layout_loader_and_structural_score():
    """Prototype U-streams should load without being mistaken for schedules."""
    layout = load_qmera_prototype_layout(
        "/tmp/U_q3_l1",
        loader=lambda _path: ((0, 1), (2, 3), (1, 2)),
        num_sites=4,
    )

    assert isinstance(layout, QMeraPrototypeLayout)
    assert layout.level == 1
    assert layout.gate_count == 3
    assert layout.round_depth == 2
    assert layout.unique_sites == (0, 1, 2, 3)

    finder = QMeraLayoutFinder(QMeraGeometry(shape=4))
    score = finder.score_prototype_layout(layout, {(0, 1): _zz_term()})
    assert score.valid
    assert score.components["interaction_coverage"] == pytest.approx(1.0)
    assert score.components["gate_count"] == 3


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

    # Boundary disentangler blocks are assigned disjoint executable rounds.
    # Their supports intentionally overlap the neighboring isometry blocks,
    # which is the MERA boundary-coupling pattern.
    d_round_by_block = {}
    for placement in first.disentanglers:
        d_round_by_block.setdefault(placement.block, placement.round)
        assert d_round_by_block[placement.block] == placement.round
    for round_index in set(d_round_by_block.values()):
        blocks = [
            set(first.disentangler_blocks[block_index])
            for block_index, block_round in d_round_by_block.items()
            if block_round == round_index
        ]
        assert all(
            not (left & right)
            for index, left in enumerate(blocks)
            for right in blocks[:index]
        )
    assert all(
        not (left & right)
        for index, left in enumerate(map(set, first.isometry_blocks))
        for right in map(set, first.isometry_blocks[:index])
    )
    assert any(
        set(disentangler) & set(isometry)
        for disentangler in first.disentangler_blocks
        for isometry in first.isometry_blocks
    )
    assert all(
        sum(
            bool(set(disentangler) & set(isometry))
            for isometry in first.isometry_blocks
        )
        >= 2
        for disentangler in first.disentangler_blocks
    )


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
    fermion = Fermion(spinful=True, symmetry="U1U1")

    site_terms = fermion.local_terms(((0, 1), (1, 2)), t=0.5, U=4.0, mu=0.1)
    mode_terms = fermion.local_terms(geometry, layout="qmera", t=0.5, U=4.0, mu=0.1)
    direct_terms = qmera_symmray_fermi_hubbard_terms(
        geometry,
        fermion=fermion, t=0.5, U=4.0, mu=0.1,
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
    fermion = Fermion(spinful=True, symmetry="U1U1")

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
    terms = builder.fermion_terms(t=0.2, U=0.5, mu=0.1)
    optimizer = builder.fermion_parametric_optimizer(
        energy_per_site=False,
        term_params={"t": 0.2, "U": 0.5, "mu": 0.1},
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
    direct_value = builder.direct_parametric_loss(
        params,
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
    assert complex(direct_value) == pytest.approx(complex(value))


def test_qmera_native_symmray_compilation_preserves_graded_metadata():
    """Compiled native cones keep Symmray grading instead of densifying."""
    pytest.importorskip("symmray")
    backend = QMeraSymmrayFermionBackend()
    registry = symmray_fermion_gate_registry(backend=backend)
    builder = QMeraBuilder(
        shape=2,
        site_modes=backend.site_modes,
        gate_registry=registry,
        gate_family="symmray-fsim",
        isometry={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        product_state_factory=backend.product_state,
    )
    terms = backend.fermi_hubbard_terms(
        builder.geometry,
        t=0.2,
        U=0.0,
        mu=0.0,
    )

    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    chunks = builder.parametric_lightcone_chunks(
        terms[:1], schedule, convert_terms=False,
    )
    compiled = builder.compile_parametric_lightcones(
        chunks=chunks,
        schedule=schedule,
        convert_terms=False,
    )
    value = builder.compiled_parametric_loss(
        parameters,
        schedule=schedule,
        compiled_chunks=compiled,
        energy_per_site=False,
        real=False,
    )
    direct = builder.parametric_loss(
        parameters,
        schedule=schedule,
        chunks=chunks,
        convert_terms=False,
        energy_per_site=False,
        real=False,
    )

    assert compiled[0].is_graded
    assert compiled[0].contraction_backend == "symmray"
    assert compiled[0].symmetry == "U1U1"
    assert complex(value) == pytest.approx(complex(direct))


def test_qmera_compiled_native_pbc_terms_match_each_explicit_cone():
    """A compiled periodic Hubbard cone agrees term-by-term with Symmray."""
    pytest.importorskip("symmray")
    geometry = QMeraGeometry(
        shape=(2, 2),
        boundary="periodic",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="U1U1",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    registry = symmray_fermion_gate_registry(backend=backend)

    def product_state_factory(schedule, sites, **kwargs):
        occupations = {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}
        return backend.product_state(
            schedule,
            sites,
            occupations=occupations,
            **kwargs,
        )

    builder = QMeraBuilder(
        geometry=geometry,
        gate_registry=registry,
        gate_family="symmray-fsim",
        disentangler={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        isometry={
            "block_size": (2, 2),
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=12,
        param_scale=0.07,
        product_state_factory=product_state_factory,
    )
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    terms = qmera_symmray_fermi_hubbard_terms(
        geometry,
        backend=backend,
        t=0.2,
        U=0.5,
        mu=0.1,
        peierls_angle=np.pi / 3,
    )
    chunks = builder.parametric_lightcone_chunks(
        terms,
        schedule,
        convert_terms=False,
    )

    class RecordingPathCache:
        def __init__(self):
            self.keys = []

        def resolve(self, optimize, *, key=None):
            self.keys.append(key)
            return "greedy" if str(optimize).startswith("auto") else optimize

    path_cache = RecordingPathCache()
    compiled = builder.compile_parametric_lightcones(
        chunks=chunks,
        schedule=schedule,
        convert_terms=False,
        path_cache=path_cache,
    )
    compiled_values = [
        local_qmera_compiled_lightcone_expectation(
            schedule,
            item,
            parameters,
            gate_registry=registry,
            normalized=False,
            real=False,
        )
        for item in compiled
    ]
    explicit_values = [
        contract_qmera_lightcone_tn(
            builder.parametric_lightcone_tn(
                chunk,
                parameters,
                schedule=schedule,
                gate_array_backend=None,
            ),
            optimize="greedy",
            normalized=False,
            real=False,
        )
        for chunk in chunks
    ]

    assert len(compiled) == len(terms)
    assert len(set(path_cache.keys)) < len(path_cache.keys)
    for compiled_value, explicit_value in zip(compiled_values, explicit_values):
        assert complex(compiled_value) == pytest.approx(complex(explicit_value))


def test_qmera_compiled_native_z2_majorana_matches_explicit():
    """The graded compiler also supports the parity-only Majorana symmetry."""
    pytest.importorskip("symmray")
    geometry = QMeraGeometry(
        shape=(2, 2),
        boundary="periodic",
        site_modes=("mode",),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="Z2",
        site_modes=("mode",),
    )
    registry = symmray_majorana_gate_registry(backend=backend)
    builder = QMeraBuilder(
        geometry=geometry,
        gate_registry=registry,
        gate_family="symmray-majorana",
        disentangler={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-majorana",
        },
        isometry={
            "block_size": (2, 2),
            "circuit_depth": 1,
            "gate_family": "symmray-majorana",
        },
        max_layers=1,
        seed=15,
        param_scale=0.03,
        product_state_factory=backend.product_state,
    )
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    terms = qmera_symmray_majorana_terms(
        geometry,
        fermion=Fermion(spinful=False, symmetry="Z2"),
        coupling=0.4,
        pairing=0.2,
    )
    chunks = builder.parametric_lightcone_chunks(
        terms,
        schedule,
        convert_terms=False,
    )
    compiled = builder.compile_parametric_lightcones(
        chunks=chunks,
        schedule=schedule,
        convert_terms=False,
        contraction_opt="greedy",
    )
    compiled_value = builder.compiled_parametric_loss(
        parameters,
        schedule=schedule,
        compiled_chunks=compiled,
        energy_per_site=False,
        real=False,
    )
    explicit_value = builder.parametric_loss(
        parameters,
        schedule=schedule,
        chunks=chunks,
        convert_terms=False,
        energy_per_site=False,
        real=False,
    )

    assert compiled
    assert all(item.is_graded and item.symmetry == "Z2" for item in compiled)
    assert complex(compiled_value) == pytest.approx(complex(explicit_value))


def test_qmera_compiled_native_symmray_torch_gradients_match_explicit():
    """Graded compiled contractions keep the qMERA Torch parameter graph."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")
    from pepsy.backends import backend_torch

    gate_backend = backend_torch(dtype=torch.complex128)
    backend = QMeraSymmrayFermionBackend(to_backend=gate_backend)
    registry = symmray_fermion_gate_registry(backend=backend)

    def product_state_factory(schedule, sites, **kwargs):
        return backend.product_state(
            schedule,
            sites,
            occupations={0: 1, 1: 0, 2: 0, 3: 1},
            **kwargs,
        )

    builder = QMeraBuilder(
        shape=2,
        site_modes=backend.site_modes,
        mode_order="mode-major",
        gate_registry=registry,
        gate_family="symmray-fsim",
        isometry={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=7,
        param_scale=0.04,
        product_state_factory=product_state_factory,
    )
    schedule = builder.build_schedule()
    parameters = builder.cast_params(
        builder.initialize_parameters(schedule),
        backend="torch",
        trainable=True,
        dtype=torch.float64,
    )
    terms = backend.fermi_hubbard_terms(
        builder.geometry,
        t=0.2,
        U=0.3,
        mu=0.1,
    )
    chunks = builder.parametric_lightcone_chunks(
        terms,
        schedule,
        convert_terms=False,
    )
    compiled = builder.compile_parametric_lightcones(
        chunks=chunks,
        schedule=schedule,
        convert_terms=False,
        contraction_opt="greedy",
    )
    compiled_value = builder.compiled_parametric_loss(
        parameters,
        schedule=schedule,
        compiled_chunks=compiled,
        energy_per_site=False,
        real=True,
    )
    explicit_value = builder.parametric_loss(
        parameters,
        schedule=schedule,
        chunks=chunks,
        convert_terms=False,
        energy_per_site=False,
        real=True,
    )
    compiled_gradients = torch.autograd.grad(
        compiled_value,
        tuple(parameters.values()),
        retain_graph=True,
    )
    explicit_gradients = torch.autograd.grad(
        explicit_value,
        tuple(parameters.values()),
    )

    assert compiled_value.requires_grad
    assert explicit_value.requires_grad
    assert float(compiled_value) == pytest.approx(float(explicit_value))
    for compiled_gradient, explicit_gradient in zip(
        compiled_gradients,
        explicit_gradients,
    ):
        np.testing.assert_allclose(
            compiled_gradient.detach().numpy(),
            explicit_gradient.detach().numpy(),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


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
    hubbard = Fermion(spinful=True, symmetry="U1U1")
    hubbard_terms = qmera_symmray_fermi_hubbard_terms(
        hubbard_geometry,
        fermion=hubbard, t=0.2, U=0.5, mu=0.1,
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


def test_qmera_fermion_every_term_and_grouping_match_direct_oracle():
    """Every native Hubbard term should agree before grouping is enabled."""
    pytest.importorskip("symmray")
    geometry = QMeraGeometry(
        shape=(2, 2),
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="U1U1",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    registry = symmray_fermion_gate_registry(backend=backend)

    def product_state_factory(schedule, sites, **kwargs):
        occupations = {
            site: int(
                (sum(schedule.geometry.to_site(site)) % 2 == 0)
                == (schedule.geometry.to_mode(site)[1] == "up")
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
        gate_family="symmray-fsim",
        disentangler={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        isometry={
            "block_size": (2, 2),
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=97,
        param_scale=0.01,
        product_state_factory=product_state_factory,
    )
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    fermion = Fermion(spinful=True, symmetry="U1U1")
    terms = builder.fermion_terms(fermion, t=0.2, U=0.7, mu=0.1)

    for term in terms:
        chunk = build_qmera_parametric_lightcone_chunks(schedule, (term,))
        local = builder.parametric_loss(
            parameters,
            (term,),
            schedule=schedule,
            chunks=chunk,
            convert_terms=False,
            energy_per_site=False,
            group_terms=False,
            real=False,
        )
        direct = builder.direct_parametric_loss(
            parameters,
            (term,),
            schedule=schedule,
            convert_terms=False,
            energy_per_site=False,
            group_terms=False,
            real=False,
        )
        assert complex(local) == pytest.approx(complex(direct))

    grouped = builder.parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
        group_terms=True,
        real=False,
    )
    ungrouped = builder.parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
        group_terms=False,
        real=False,
    )
    assert complex(grouped) == pytest.approx(complex(ungrouped))


def test_qmera_periodic_fermion_terms_match_jordan_wigner_fock_oracle():
    """PBC native qMERA terms must match an independent JW Fock oracle."""
    pytest.importorskip("symmray")

    def jw_annihilate(num_modes, mode):
        eye = np.eye(2)
        zed = np.diag([1.0, -1.0])
        lower = np.array([[0.0, 1.0], [0.0, 0.0]])
        mats = [zed] * mode + [lower] + [eye] * (num_modes - mode - 1)
        out = mats[0]
        for matrix in mats[1:]:
            out = np.kron(out, matrix)
        return out

    def fock_vector(state, geometry, backend):
        full = state.contract(all)
        labels = tuple(f"k{site}" for site in geometry.register_sites)
        permutation = tuple(full.inds.index(label) for label in labels)
        # The native contraction already carries the graded swap phases. The
        # phase-aware reorder converts its output-index order to the canonical
        # qMERA register order without applying a second bosonization gauge.
        data = full.data.transpose(permutation, phase=True)
        dense = np.asarray(data.to_dense())
        occupation_positions = []
        for axis, register_site in enumerate(geometry.register_sites):
            mode = geometry.to_mode(register_site)
            occupied_charge = backend.mode_index_map(mode)[1]
            charges = []
            for charge, size in data.indices[axis].chargemap.items():
                charges.extend([charge] * int(size))
            occupation_positions.append(
                tuple(int(charge == occupied_charge) for charge in charges)
            )

        vector = np.zeros(2 ** geometry.num_modes, dtype=complex)
        for index in np.ndindex(dense.shape):
            flat = 0
            for axis, local_index in enumerate(index):
                flat = (flat << 1) | occupation_positions[axis][local_index]
            vector[flat] = dense[index]
        return vector

    geometry = QMeraGeometry(
        shape=(2, 2),
        boundary="periodic",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="U1U1",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    registry = symmray_fermion_gate_registry(backend=backend)

    def product_state_factory(schedule, sites, **kwargs):
        # The qMERA isometry pairs are (0, 2) and (1, 3) in this register
        # ordering, so both native hopping directions are populated.
        occupations = {
            0: 1,
            1: 0,
            2: 0,
            3: 1,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
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
        gate_family="symmray-fsim",
        disentangler={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        isometry={
            "block_size": (2, 2),
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=12,
        param_scale=0.2,
        product_state_factory=product_state_factory,
    )
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    state, _ = builder.build_state(parameters, schedule)
    vector = fock_vector(state, geometry, backend)
    norm = np.vdot(vector, vector)
    assert norm == pytest.approx(1.0)

    t, U, mu, angle = 0.2, 0.5, 0.1, np.pi / 3
    terms = qmera_symmray_fermi_hubbard_terms(
        geometry,
        backend=backend,
        t=t,
        U=U,
        mu=mu,
        peierls_angle=angle,
    )
    annihilators = [
        jw_annihilate(geometry.num_modes, mode)
        for mode in range(geometry.num_modes)
    ]
    number_operators = [
        operator.conj().T @ operator for operator in annihilators
    ]

    native_total = 0.0j
    oracle_total = 0.0j
    for term in terms:
        kind = term.metadata["kind"]
        if kind == "hubbard-onsite":
            site = term.metadata["site"]
            up = geometry.mode_register(site, "up")
            down = geometry.mode_register(site, "down")
            oracle_operator = U * number_operators[up] @ number_operators[down]
        elif kind == "hubbard-chemical":
            site = term.metadata["site"]
            mode = geometry.mode_register(site, term.metadata["mode"])
            oracle_operator = -mu * number_operators[mode]
        else:
            left, right = term.metadata["edge"]
            mode = term.metadata["mode"]
            left_mode = geometry.mode_register(left, mode)
            right_mode = geometry.mode_register(right, mode)
            oracle_operator = -t * (
                np.exp(1.0j * angle)
                * annihilators[left_mode].conj().T
                @ annihilators[right_mode]
                + np.exp(-1.0j * angle)
                * annihilators[right_mode].conj().T
                @ annihilators[left_mode]
            )

        native_term_state = state.gate_inds(
            term.operator,
            inds=tuple(geometry.site_ind(site) for site in term.where),
            contract=False,
            inplace=False,
        )
        native_value = (state.H & native_term_state).contract(all) / norm
        oracle_value = np.vdot(vector, oracle_operator @ vector) / norm
        assert complex(native_value) == pytest.approx(complex(oracle_value))
        native_total += native_value
        oracle_total += oracle_value

    assert complex(native_total) == pytest.approx(complex(oracle_total))


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
        "pepsy.optimizers.qmera.cache.build_qmera_contraction_optimizer",
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

    value = builder.parametric_loss(
        ansatz.parameters,
        {(0, 1): _zz_term()},
        schedule=ansatz.schedule,
        energy_per_site=False,
    )
    assert np.isfinite(float(value))


def test_qmera_schedule_lightcone_chunks_follow_placements():
    """Schedule chunks should select the RG reverse cone, not only site tags."""
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=19,
        param_scale=0.1,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    op = _zz_term()

    chunks = builder.parametric_lightcone_chunks(
        {(0, 1): op},
        schedule,
    )
    value = builder.parametric_loss(
        params,
        {(0, 1): op},
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        real=False,
    )
    direct = builder.direct_parametric_loss(
        params,
        {(0, 1): op},
        schedule=schedule,
        energy_per_site=False,
        real=False,
    )
    expected_ids = tuple(
        placement.gate_id
        for placement in schedule.reverse_lightcone_placements((0, 1))
    )

    assert chunks[0].source == "parametric-schedule"
    assert chunks[0].schedule_placement_ids == expected_ids
    assert any(tag.startswith("GATE_L0_DIS") for tag in chunks[0].tags)
    assert chunks[0].schedule_width >= chunks[0].support_size
    assert complex(value) == pytest.approx(complex(direct))


def test_qmera_schedule_lightcone_chunks_map_coordinate_terms():
    """Coordinate-lattice terms should compile to qMERA register supports."""
    builder = QMeraBuilder(
        shape=(4, 4),
        disentangler={"block_size": 2, "circuit_depth": 1},
        isometry={"block_size": (2, 2), "circuit_depth": 2},
        seed=20,
        param_scale=0.03,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    op = _zz_term()

    hamiltonian = {((0, 0), (0, 1)): op}
    chunks = builder.parametric_lightcone_chunks(
        hamiltonian,
        schedule,
    )
    value = builder.parametric_loss(
        params,
        hamiltonian,
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        real=False,
    )
    direct = builder.direct_parametric_loss(
        params,
        hamiltonian,
        schedule=schedule,
        energy_per_site=False,
        real=False,
    )
    chunk = chunks[0]

    assert chunk.source == "parametric-schedule"
    assert chunk.term.where == (0, 1)
    assert chunk.term.metadata["original_where"] == ((0, 0), (0, 1))
    assert chunk.term.metadata["register_where"] == (0, 1)
    assert chunk.schedule_placement_ids
    assert chunk.schedule_width_by_scale
    assert complex(value) == pytest.approx(complex(direct))


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
    assert isinstance(opt, QMeraEnergyOptimizer)
    assert QMeraEnergyOptimizer is QMeraParametricEnergyOptimizer
    assert opt.loss_kwargs["normalized"] is False
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

    # Each RG scale should start from the same left edge. This catches the
    # easy-to-miss cursor drift that makes later 2D scales look detached.
    all_layers = builder.draw_schematic(
        style="clean",
        label_sites=False,
        label_blocks=False,
        scale_figsize=False,
    )
    input_labels = [
        text
        for text in all_layers.ax.texts
        if text.get_text() == "input"
    ]
    assert len(input_labels) == 2
    assert input_labels[0].get_position()[0] == pytest.approx(
        input_labels[1].get_position()[0]
    )
    assert input_labels[0].get_position()[1] != pytest.approx(
        input_labels[1].get_position()[1]
    )

    first_step = builder.draw_schematic(
        rg_step=0,
        style="clean",
        label_sites=False,
        label_blocks=True,
        scale_figsize=False,
    )
    assert sum(text.get_text() == "input" for text in first_step.ax.texts) == 1
    assert sum(text.get_text() == "coarse" for text in first_step.ax.texts) == 1
    assert any(text.get_text().startswith("D[") for text in first_step.ax.texts)
    assert any(text.get_text().startswith("W[") for text in first_step.ax.texts)
    with pytest.raises(TypeError, match="only one of layer= or rg_step="):
        builder.draw_schematic(layer=0, rg_step=0)
    with pytest.raises(ValueError, match="style"):
        builder.draw_schematic(layer=0, style="unknown")
