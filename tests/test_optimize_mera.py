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
    QMeraGeometry,
    QMeraLightconeTN,
    QMeraSchematicBlock,
    QMeraParametricEnergyOptimizer,
    QMeraParametricLightconeChunk,
    UserGateFamily,
    build_lightcone_chunks,
    build_qmera_lightcone_chunks,
    build_qmera_parametric_lightcone_chunks,
    compile_qmera_parametric_lightcones,
    contract_qmera_lightcone_tn,
    default_gate_registry,
    draw_qmera_schedule,
    local_qmera_compiled_lightcone_expectation,
    local_qmera_parametric_lightcone_expectation,
    local_lightcone_expectation,
    normalize_local_terms,
    qmera_compiled_parametric_energy,
    qmera_parametric_energy,
    qmera_parametric_lightcone_state,
    qmera_parametric_lightcone_tn,
    qmera_schematic_blocks,
)
from pepsy.optimizers.mera.optimizer import (
    MeraEnergyOptimizer as ModuleMeraEnergyOptimizer,
)


def _zz_term():
    z_op = np.diag([1.0, -1.0])
    return np.kron(z_op, z_op).reshape(2, 2, 2, 2)


def _small_mera(seed=23):
    return qtn.MERA.rand(L=8, max_bond=2, dtype="complex128", seed=seed)


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


def test_qmera_gate_registry_generates_parametrized_two_qubit_gates():
    """The registry should produce backend-ready two-qubit gate tensors."""
    registry = default_gate_registry()
    spec = registry.get("rxx")
    gate = spec.matrix([0.25])

    assert isinstance(spec, GateSpec)
    assert spec.arity == 2
    assert spec.num_params == 1
    assert gate.shape == (2, 2, 2, 2)
    assert "su4" in registry.names(arity=2)

    custom = UserGateFamily(
        name="diag-phase",
        arity=2,
        num_params=1,
        generator=lambda params: np.diag(
            [1.0, 1.0, 1.0, np.exp(1.0j * params[0])]
        ).reshape(2, 2, 2, 2),
    )
    registry.register(custom)
    custom_gate = registry.get("diag-phase").matrix([0.3])
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
