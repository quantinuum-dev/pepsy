"""qMERA ansatz builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import quimb.tensor as qtn

from ...backends import (
    backend_jax,
    backend_numpy,
    backend_torch,
    get_default_array_backend,
    get_default_grad_backend,
)
from .compiled import (
    compile_qmera_parametric_lightcones,
    qmera_compiled_parametric_energy,
)
from .cache import build_qmera_contraction_optimizer
from .gates import GateRegistry, default_gate_registry, resolve_gate_spec
from .geometry import QMeraGeometry
from .lightcones import (
    build_qmera_parametric_lightcone_chunks,
    qmera_parametric_energy,
    qmera_parametric_lightcone_tn,
)
from .schedules import QMeraBlockSpec, QMeraSchedule, build_qmera_schedule
from .terms import convert_local_terms, normalize_local_terms

__all__ = ["QMeraAnsatz", "QMeraBuilder"]


def _backend_from_name(backend, *, trainable=True, dtype=None, device="cpu"):
    if backend is None or callable(backend):
        return backend
    key = str(backend).strip().lower().replace("_", "-")
    if key in {"torch", "pytorch"}:
        return backend_torch(device=device, dtype=dtype, requires_grad=bool(trainable))
    if key == "jax":
        return backend_jax(device=device, dtype=dtype)
    if key in {"numpy", "np"}:
        return backend_numpy(dtype=np.float64 if dtype is None else dtype)
    raise ValueError(
        "backend must be a callable, None, or one of 'torch', 'jax', or 'numpy'."
    )


@dataclass(frozen=True)
class QMeraAnsatz:
    """Built qMERA ansatz payload."""

    state: Any
    schedule: QMeraSchedule
    parameters: Mapping[str, Any]
    gate_tensors: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def geometry(self):
        """Return the ansatz geometry."""
        return self.schedule.geometry

    def reverse_lightcone_tags(self, where):
        """Return schedule tags in the reverse lightcone of ``where``."""
        return self.schedule.reverse_lightcone_tags(where)

    def schematic_blocks(self, *, layer=None):
        """Return display-oriented disentangler/isometry blocks."""
        return self.schedule.schematic_blocks(layer=layer)

    def draw_schematic(self, *, layer=None, **kwargs):
        """Draw qMERA blocking for this ansatz."""
        return self.schedule.draw_schematic(layer=layer, **kwargs)


def _coerce_geometry(
    geometry=None,
    *,
    shape=None,
    boundary="open",
    mapper=None,
    site_modes=None,
    mode_order="site-major",
):
    if isinstance(geometry, QMeraGeometry):
        return geometry
    if geometry is not None:
        if isinstance(geometry, Mapping):
            opts = dict(geometry)
            if site_modes is not None:
                opts.setdefault("site_modes", site_modes)
            if mode_order != "site-major":
                opts.setdefault("mode_order", mode_order)
            return QMeraGeometry(**opts)
        if shape is not None:
            raise TypeError("Provide either geometry or shape, not both.")
        return QMeraGeometry(
            geometry,
            boundary=boundary,
            mapper=mapper,
            site_modes=site_modes,
            mode_order=mode_order,
        )
    if shape is None:
        raise TypeError("QMeraBuilder requires geometry or shape.")
    return QMeraGeometry(
        shape,
        boundary=boundary,
        mapper=mapper,
        site_modes=site_modes,
        mode_order=mode_order,
    )


def _coerce_block_spec(value, *, kind, gate_family):
    if isinstance(value, QMeraBlockSpec):
        if value.kind != kind:
            raise ValueError(f"{kind} block spec has kind={value.kind!r}.")
        return value
    opts = dict(value or {})
    opts.setdefault("gate_family", gate_family)
    return QMeraBlockSpec(kind=kind, **opts)


class QMeraBuilder:
    """Build a schedule-first qMERA ansatz from explicit Pepsy objects."""

    def __init__(
        self,
        *,
        geometry=None,
        shape=None,
        boundary="open",
        mapper=None,
        site_modes=None,
        mode_order="site-major",
        physical_dim: int = 2,
        disentangler=None,
        isometry=None,
        gate_family: str = "rxx",
        isometry_gate_family: str | None = None,
        gate_registry: GateRegistry | None = None,
        max_layers: int | None = None,
        top_size: int = 1,
        seed: int | None = None,
        param_scale: float = 0.0,
        array_backend=None,
        parameter_backend=None,
        product_state_factory=None,
    ):
        self.geometry = _coerce_geometry(
            geometry,
            shape=shape,
            boundary=boundary,
            mapper=mapper,
            site_modes=site_modes,
            mode_order=mode_order,
        )
        self.physical_dim = int(physical_dim)
        if self.physical_dim != 2:
            raise NotImplementedError("QMeraBuilder currently supports qubits only.")
        self.gate_registry = (
            default_gate_registry()
            if gate_registry is None
            else gate_registry.copy()
        )
        self.disentangler = _coerce_block_spec(
            disentangler,
            kind="disentangler",
            gate_family=gate_family,
        )
        self.isometry = _coerce_block_spec(
            isometry,
            kind="isometry",
            gate_family=isometry_gate_family or gate_family,
        )
        self.max_layers = max_layers
        self.top_size = top_size
        self.seed = seed
        self.param_scale = float(param_scale)
        self.array_backend = array_backend
        self.parameter_backend = parameter_backend
        self.product_state_factory = product_state_factory

    def build_schedule(self):
        """Build the static qMERA schedule."""
        return build_qmera_schedule(
            self.geometry,
            disentangler=self.disentangler,
            isometry=self.isometry,
            max_layers=self.max_layers,
            top_size=self.top_size,
        )

    def schematic_blocks(self, *, layer=None):
        """Return display-oriented disentangler/isometry blocks."""
        return self.build_schedule().schematic_blocks(layer=layer)

    def draw_schematic(self, *, layer=None, **kwargs):
        """Draw the qMERA blocking implied by this builder."""
        return self.build_schedule().draw_schematic(layer=layer, **kwargs)

    def contraction_optimizer(self, **kwargs):
        """Build a reusable contraction optimizer for repeated local cones."""
        return build_qmera_contraction_optimizer(**kwargs)

    def parametric_lightcone_chunks(
        self,
        hamiltonian,
        schedule=None,
        *,
        array_backend=None,
        convert_terms=True,
    ):
        """Compile schedule-only qMERA lightcone chunks for local terms."""
        schedule = self.build_schedule() if schedule is None else schedule
        terms = normalize_local_terms(hamiltonian)
        if convert_terms:
            backend = self.array_backend if array_backend is None else array_backend
            if backend is None:
                backend = get_default_array_backend()
            terms = convert_local_terms(terms, backend)
        return build_qmera_parametric_lightcone_chunks(schedule, terms)

    def fermion_terms(self, fermion, **params):
        """Return qMERA mode terms from a unified :class:`Fermion` helper.

        qMERA's fermionic path represents each physical site as the explicit
        ``("up", "down")`` pair of two-state modes. Keeping this conversion
        on the builder makes the representation choice visible while allowing
        the regular local-term and optimizer machinery to handle the result.
        """
        if not callable(getattr(fermion, "local_terms", None)):
            raise TypeError("fermion must provide local_terms(...).")
        if not getattr(fermion, "spinful", False):
            raise ValueError("qMERA Hubbard terms require a spinful Fermion helper.")
        if tuple(self.geometry.site_modes or ()) != ("up", "down"):
            raise ValueError(
                "Fermion qMERA integration requires "
                "site_modes=('up', 'down')."
            )
        return tuple(
            fermion.local_terms(
                self.geometry,
                layout="qmera",
                **params,
            )
        )

    def fermion_parametric_loss(
        self,
        fermion,
        parameters,
        schedule=None,
        *,
        term_params=None,
        **loss_kwargs,
    ):
        """Evaluate qMERA energy directly from a unified ``Fermion`` model."""
        schedule = self.build_schedule() if schedule is None else schedule
        terms = self.fermion_terms(fermion, **dict(term_params or {}))
        return self.parametric_loss(
            parameters,
            terms,
            schedule=schedule,
            **loss_kwargs,
        )

    def fermion_parametric_optimizer(
        self,
        fermion,
        *,
        schedule=None,
        term_params=None,
        **optimizer_kwargs,
    ):
        """Create a qMERA optimizer directly from a unified ``Fermion`` model."""
        schedule = self.build_schedule() if schedule is None else schedule
        terms = self.fermion_terms(fermion, **dict(term_params or {}))
        return self.parametric_optimizer(
            terms,
            schedule=schedule,
            **optimizer_kwargs,
        )

    def parametric_loss(
        self,
        parameters,
        hamiltonian=None,
        schedule=None,
        *,
        chunks=None,
        array_backend=None,
        gate_array_backend=None,
        convert_terms=True,
        normalized=True,
        energy_per_site=True,
        real=True,
        contraction_opt="auto-hq",
        simplify=False,
        gate_contract=True,
        contract_opts=None,
    ):
        """Evaluate qMERA energy from params by rebuilding local cones only."""
        schedule = self.build_schedule() if schedule is None else schedule
        backend = self.array_backend if array_backend is None else array_backend
        return qmera_parametric_energy(
            schedule,
            parameters,
            hamiltonian,
            chunks=chunks,
            gate_registry=self.gate_registry,
            array_backend=backend,
            gate_array_backend=gate_array_backend,
            convert_terms=convert_terms,
            physical_dim=self.physical_dim,
            optimize=contraction_opt,
            normalized=normalized,
            energy_per_site=energy_per_site,
            real=real,
            simplify=simplify,
            gate_contract=gate_contract,
            contract_opts=contract_opts,
            product_state_factory=self.product_state_factory,
        )

    def _parameter_converter(self):
        if self.parameter_backend is not None:
            return self.parameter_backend
        default_grad = get_default_grad_backend()
        if default_grad is not None:
            return default_grad
        if self.array_backend is not None:
            return self.array_backend
        return get_default_array_backend()

    def initialize_parameters(self, schedule=None, *, seed=None, scale=None):
        """Initialize one parameter vector per scheduled gate."""
        schedule = self.build_schedule() if schedule is None else schedule
        rng = np.random.default_rng(self.seed if seed is None else seed)
        scale = self.param_scale if scale is None else float(scale)
        converter = self._parameter_converter()
        params = {}
        for placement in schedule.placements:
            spec = resolve_gate_spec(placement.gate_family, self.gate_registry)
            if spec.num_params == 0:
                values = np.empty((0,), dtype=np.float64)
            elif scale == 0.0:
                values = np.zeros((spec.num_params,), dtype=np.float64)
            else:
                values = rng.normal(scale=scale, size=(spec.num_params,))
            params[placement.param_key] = values if converter is None else converter(values)
        return params

    def cast_params(
        self,
        values,
        *,
        trainable: bool = True,
        backend=None,
        dtype=None,
        device="cpu",
        stop_grad_fn=None,
    ):
        """Cast a parameter dictionary to a Pepsy-supported backend."""
        if backend is None:
            converter = self._parameter_converter() if trainable else self.array_backend
        else:
            converter = _backend_from_name(
                backend,
                trainable=trainable,
                dtype=dtype,
                device=device,
            )
        if converter is None:
            return dict(values)
        out = {}
        for key, value in values.items():
            if stop_grad_fn is not None:
                value = stop_grad_fn(value)
            out[key] = converter(value)
        return out

    cast_parameters = cast_params

    def parametric_loss_fn(
        self,
        hamiltonian=None,
        schedule=None,
        *,
        chunks=None,
        **loss_kwargs,
    ):
        """Return a pure ``loss(params)`` callable for qMERA parameters."""
        schedule = self.build_schedule() if schedule is None else schedule

        def _loss(parameters):
            return self.parametric_loss(
                parameters,
                hamiltonian=hamiltonian,
                schedule=schedule,
                chunks=chunks,
                **loss_kwargs,
            )

        return _loss

    def parametric_lightcone_tn(
        self,
        chunk,
        parameters,
        schedule=None,
        *,
        array_backend=None,
        gate_array_backend=None,
        simplify=False,
        gate_contract=True,
    ):
        """Build explicit TNs for one scheduled qMERA local-energy chunk."""
        schedule = self.build_schedule() if schedule is None else schedule
        backend = self.array_backend if array_backend is None else array_backend
        return qmera_parametric_lightcone_tn(
            schedule,
            chunk,
            parameters,
            gate_registry=self.gate_registry,
            array_backend=backend,
            gate_array_backend=gate_array_backend,
            physical_dim=self.physical_dim,
            simplify=simplify,
            gate_contract=gate_contract,
            product_state_factory=self.product_state_factory,
        )

    def compile_parametric_lightcones(
        self,
        hamiltonian=None,
        schedule=None,
        *,
        chunks=None,
        array_backend=None,
        convert_terms=True,
        contraction_opt="auto-hq",
        expression_opts=None,
    ):
        """Compile static contraction expressions for qMERA local cones."""
        schedule = self.build_schedule() if schedule is None else schedule
        backend = self.array_backend if array_backend is None else array_backend
        if chunks is None:
            if hamiltonian is None:
                raise ValueError(
                    "compile_parametric_lightcones requires hamiltonian or chunks."
                )
            chunks = self.parametric_lightcone_chunks(
                hamiltonian,
                schedule,
                array_backend=backend,
                convert_terms=convert_terms,
            )
        return compile_qmera_parametric_lightcones(
            schedule,
            chunks,
            gate_registry=self.gate_registry,
            array_backend=backend,
            physical_dim=self.physical_dim,
            optimize=contraction_opt,
            expression_opts=expression_opts,
        )

    def compiled_parametric_loss(
        self,
        parameters,
        hamiltonian=None,
        schedule=None,
        *,
        chunks=None,
        compiled_chunks=None,
        array_backend=None,
        gate_array_backend=None,
        convert_terms=True,
        normalized=True,
        energy_per_site=True,
        real=True,
        contraction_opt="auto-hq",
        expression_opts=None,
    ):
        """Evaluate qMERA energy with precompiled local-cone contractions."""
        schedule = self.build_schedule() if schedule is None else schedule
        backend = self.array_backend if array_backend is None else array_backend
        return qmera_compiled_parametric_energy(
            schedule,
            parameters,
            hamiltonian,
            compiled_chunks=compiled_chunks,
            chunks=chunks,
            gate_registry=self.gate_registry,
            array_backend=backend,
            gate_array_backend=gate_array_backend,
            convert_terms=convert_terms,
            physical_dim=self.physical_dim,
            optimize=contraction_opt,
            normalized=normalized,
            energy_per_site=energy_per_site,
            real=real,
            expression_opts=expression_opts,
        )

    def compiled_parametric_loss_fn(
        self,
        hamiltonian=None,
        schedule=None,
        *,
        chunks=None,
        compiled_chunks=None,
        **loss_kwargs,
    ):
        """Return ``loss(params)`` using precompiled qMERA local cones."""
        schedule = self.build_schedule() if schedule is None else schedule

        def _loss(parameters):
            return self.compiled_parametric_loss(
                parameters,
                hamiltonian=hamiltonian,
                schedule=schedule,
                chunks=chunks,
                compiled_chunks=compiled_chunks,
                **loss_kwargs,
            )

        return _loss

    def parametric_optimizer(
        self,
        hamiltonian,
        *,
        schedule=None,
        chunks=None,
        parameters=None,
        **loss_kwargs,
    ):
        """Create a parameter-dict qMERA energy optimizer shell."""
        from .parametric import QMeraParametricEnergyOptimizer

        schedule = self.build_schedule() if schedule is None else schedule
        if chunks is None:
            chunks = self.parametric_lightcone_chunks(
                hamiltonian,
                schedule,
                array_backend=loss_kwargs.get("array_backend"),
                convert_terms=loss_kwargs.get("convert_terms", True),
            )
        if parameters is None:
            parameters = self.initialize_parameters(schedule)
        return QMeraParametricEnergyOptimizer(
            builder=self,
            schedule=schedule,
            hamiltonian=hamiltonian,
            chunks=chunks,
            parameters=parameters,
            loss_kwargs=loss_kwargs,
        )

    def gate_tensors(self, parameters, schedule=None):
        """Generate gate tensors for every scheduled placement."""
        schedule = self.build_schedule() if schedule is None else schedule
        tensors = {}
        for placement in schedule.placements:
            spec = resolve_gate_spec(placement.gate_family, self.gate_registry)
            if spec.arity != placement.arity:
                raise ValueError(
                    f"Gate family {spec.name!r} has arity {spec.arity}, "
                    f"but placement {placement.gate_id} acts on {placement.arity} sites."
                )
            tensors[placement.gate_id] = spec.matrix_for_placement(
                parameters[placement.param_key],
                placement=placement,
                schedule=schedule,
                array_backend=self.array_backend,
            )
        return tensors

    def _initial_state(self):
        binary = "0" * self.geometry.num_modes
        return qtn.MPS_computational_state(
            binary,
            site_ind_id="k{}",
            site_tag_id="I{}",
        )

    def build_state(self, parameters, schedule=None, *, contract=False):
        """Build a quimb tensor network by directly applying scheduled gates."""
        schedule = self.build_schedule() if schedule is None else schedule
        tensors = self.gate_tensors(parameters, schedule)
        if self.product_state_factory is None:
            state = self._initial_state()
        else:
            state = self.product_state_factory(
                schedule,
                schedule.geometry.register_sites,
                physical_dim=self.physical_dim,
                array_backend=self.array_backend,
            )
        for placement in schedule.placements:
            if callable(getattr(state, "gate", None)):
                state = state.gate(
                    tensors[placement.gate_id],
                    placement.where,
                    contract=contract,
                    tags=placement.tags,
                    propagate_tags="sites",
                    inplace=False,
                )
            else:
                state = state.gate_inds(
                    tensors[placement.gate_id],
                    inds=tuple(f"k{site}" for site in placement.where),
                    contract=contract,
                    tags=placement.tags,
                    inplace=False,
                )
        return state, tensors

    def build(self, parameters=None, *, build_state=True, contract=False):
        """Build qMERA ansatz metadata, parameters, gates, and optional state."""
        schedule = self.build_schedule()
        if parameters is None:
            parameters = self.initialize_parameters(schedule)
        else:
            parameters = dict(parameters)

        if build_state:
            state, tensors = self.build_state(parameters, schedule, contract=contract)
        else:
            state = None
            tensors = self.gate_tensors(parameters, schedule)

        metadata = {
            "shape": self.geometry.shape,
            "boundary": self.geometry.boundary,
            "physical_dim": self.physical_dim,
            "num_sites": self.geometry.num_sites,
            "num_modes": self.geometry.num_modes,
            "site_modes": self.geometry.site_modes,
            "mode_order": self.geometry.mode_order,
            "num_gates": schedule.num_gates,
            "num_layers": len(schedule.layers),
            "top_sites": schedule.top_sites,
            "gate_families": tuple(
                sorted({placement.gate_family for placement in schedule.placements})
            ),
            "state_kind": "direct-gate-tn" if build_state else "schedule-only",
        }
        return QMeraAnsatz(
            state=state,
            schedule=schedule,
            parameters=parameters,
            gate_tensors=tensors,
            metadata=metadata,
        )
