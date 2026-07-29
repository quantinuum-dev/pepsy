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
    group_qmera_parametric_lightcone_chunks,
    qmera_direct_parametric_energy,
    qmera_parametric_energy,
    qmera_parametric_lightcone_tn,
)
from .schedules import (
    QMeraBlockSpec,
    QMeraDisentanglerSpec,
    QMeraIsometrySpec,
    QMeraScaleSpec,
    QMeraSchedule,
    QMeraUnitarySpec,
    build_qmera_schedule,
)
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
    mode_order=None,
):
    if isinstance(geometry, QMeraGeometry):
        # A model-aware builder may need to add the explicit local modes to a
        # geometry that was created as a plain spatial lattice. Preserve an
        # already explicit geometry unless the caller supplied an override.
        requested_modes = geometry.site_modes if site_modes is None else site_modes
        requested_order = geometry.mode_order if mode_order is None else mode_order
        if (
            tuple(requested_modes or ()) == tuple(geometry.site_modes or ())
            and requested_order == geometry.mode_order
        ):
            return geometry
        return QMeraGeometry(
            geometry.shape,
            boundary=geometry.boundary,
            site_labels=geometry.site_labels,
            site_modes=requested_modes,
            mode_order=requested_order,
        )
    if geometry is not None:
        if isinstance(geometry, Mapping):
            opts = dict(geometry)
            if site_modes is not None:
                opts.setdefault("site_modes", site_modes)
            if mode_order is not None:
                opts.setdefault("mode_order", mode_order)
            return QMeraGeometry(**opts)
        if shape is not None:
            raise TypeError("Provide either geometry or shape, not both.")
        return QMeraGeometry(
            geometry,
            boundary=boundary,
            mapper=mapper,
            site_modes=site_modes,
            mode_order="site-major" if mode_order is None else mode_order,
        )
    if shape is None:
        raise TypeError("QMeraBuilder requires geometry or shape.")
    return QMeraGeometry(
        shape,
        boundary=boundary,
        mapper=mapper,
        site_modes=site_modes,
        mode_order="site-major" if mode_order is None else mode_order,
    )


def _infer_fermion_site_modes(fermion):
    """Infer qMERA's explicit mode labels from a ``Fermion`` helper."""
    if getattr(fermion, "spinful", False):
        return ("up", "down")
    return ("mode",)


def _coerce_block_spec(value, *, kind, gate_family):
    if kind == "disentangler" and isinstance(value, QMeraDisentanglerSpec):
        return value.to_block_spec(default_gate_family=gate_family)
    if kind == "isometry" and isinstance(value, QMeraIsometrySpec):
        return value.to_block_spec(default_gate_family=gate_family)
    if isinstance(value, QMeraBlockSpec):
        if value.kind != kind:
            raise ValueError(f"{kind} block spec has kind={value.kind!r}.")
        return value
    opts = dict(value or {})
    unitary = opts.pop("unitary", None)
    if unitary is not None:
        unitary = QMeraUnitarySpec.coerce(
            unitary,
            default_gate_family=gate_family,
        )
        opts.setdefault("gate_family", unitary.gate_family)
        opts.setdefault("unitary_spec", unitary)
    elif "unitary_spec" in opts and opts["unitary_spec"] is not None:
        opts["unitary_spec"] = QMeraUnitarySpec.coerce(
            opts["unitary_spec"],
            default_gate_family=opts.get("gate_family", gate_family),
        )
    opts.setdefault("gate_family", gate_family)
    return QMeraBlockSpec(kind=kind, **opts)


def _normalize_gate_token(value):
    key = str(value).strip().lower().replace("_", "-")
    return {"fermionic": "fermion", "qubit": "spin"}.get(key, key)


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
        mode_order=None,
        fermion=None,
        physical_dim: int = 2,
        disentangler=None,
        isometry=None,
        scales=None,
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
        self.fermion = fermion
        if fermion is not None and not callable(
            getattr(fermion, "local_terms", None)
        ):
            raise TypeError("fermion must provide local_terms(...).")

        inferred_modes = (
            _infer_fermion_site_modes(fermion)
            if fermion is not None
            else site_modes
        )
        requested_modes = (
            site_modes
            if site_modes is not None
            else (
                geometry.site_modes
                if isinstance(geometry, QMeraGeometry)
                and geometry.site_modes is not None
                else inferred_modes
            )
        )
        # A mode-major register is the natural default for a spinful qMERA:
        # each spatial layer contains one complete up/down mode register. The
        # generic builder retains its historical site-major default.
        inferred_order = (
            mode_order
            if mode_order is not None
            else (
                geometry.mode_order
                if isinstance(geometry, QMeraGeometry)
                else ("mode-major" if fermion is not None else None)
            )
        )
        self.geometry = _coerce_geometry(
            geometry,
            shape=shape,
            boundary=boundary,
            mapper=mapper,
            site_modes=requested_modes,
            mode_order=inferred_order,
        )
        if fermion is not None:
            expected_modes = tuple(_infer_fermion_site_modes(fermion))
            actual_modes = tuple(self.geometry.site_modes or ())
            if actual_modes != expected_modes:
                raise ValueError(
                    "The qMERA geometry's site_modes must match the Fermion "
                    f"helper: expected {expected_modes!r}, got {actual_modes!r}."
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
        self._validate_unitary_spec(self.disentangler)
        self._validate_unitary_spec(self.isometry)
        if scales is not None:
            self.scales = tuple(scales)
            for scale in self.scales:
                if not isinstance(scale, (QMeraScaleSpec, Mapping)):
                    raise TypeError(
                        "scales must contain QMeraScaleSpec or mapping objects."
                    )
        else:
            self.scales = None
        self.max_layers = max_layers
        self.top_size = top_size
        self.seed = seed
        self.param_scale = float(param_scale)
        self.array_backend = array_backend
        self.parameter_backend = parameter_backend
        self.product_state_factory = product_state_factory

    def _validate_unitary_spec(self, block_spec):
        """Validate explicit unitary metadata against the selected registry."""
        unitary = block_spec.unitary_spec
        if unitary is None:
            return
        if _normalize_gate_token(unitary.gate_family) != _normalize_gate_token(
            block_spec.gate_family
        ):
            raise ValueError(
                f"{block_spec.kind} unitary gate_family={unitary.gate_family!r} "
                f"does not match block gate_family={block_spec.gate_family!r}."
            )
        spec = resolve_gate_spec(block_spec.gate_family, self.gate_registry)
        checks = (
            ("family", unitary.family, spec.family),
            ("arity_kind", unitary.arity_kind, spec.arity_kind),
            ("symmetry", unitary.symmetry, getattr(spec, "symmetry", None)),
            ("preserves_parity", unitary.preserves_parity, spec.preserves_parity),
        )
        for name, expected, actual in checks:
            if expected is None:
                continue
            if name in {"family", "arity_kind"}:
                expected = _normalize_gate_token(expected)
                actual = _normalize_gate_token(actual)
            elif name == "symmetry":
                expected = str(expected).upper()
                actual = None if actual is None else str(actual).upper()
            if actual != expected:
                raise ValueError(
                    f"{block_spec.kind} unitary requires {name}={expected!r}, "
                    f"but gate family {spec.name!r} provides {actual!r}."
                )

    def build_schedule(self):
        """Build the static qMERA schedule."""
        schedule = build_qmera_schedule(
            self.geometry,
            disentangler=self.disentangler,
            isometry=self.isometry,
            scales=self.scales,
            max_layers=self.max_layers,
            top_size=self.top_size,
        )
        for scale in schedule.scale_specs:
            self._validate_unitary_spec(scale.disentangler)
            self._validate_unitary_spec(scale.isometry)
        return schedule

    def schematic_blocks(self, *, layer=None):
        """Return display-oriented disentangler/isometry blocks."""
        return self.build_schedule().schematic_blocks(layer=layer)

    def draw_schematic(self, *, layer=None, **kwargs):
        """Draw the qMERA blocking implied by this builder."""
        return self.build_schedule().draw_schematic(layer=layer, **kwargs)

    def contraction_optimizer(self, **kwargs):
        """Build a reusable contraction optimizer for repeated local cones."""
        return build_qmera_contraction_optimizer(**kwargs)

    def contraction_path_cache(self, **kwargs):
        """Create a lazy topology-aware contraction-path cache."""
        from .cache import QMeraContractionPathCache

        return QMeraContractionPathCache(optimizer_options=kwargs)

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

    def parametric_lightcone_groups(self, hamiltonian, schedule=None, **kwargs):
        """Group local terms sharing one reverse-lightcone topology."""
        chunks = self.parametric_lightcone_chunks(
            hamiltonian,
            schedule=schedule,
            **kwargs,
        )
        return group_qmera_parametric_lightcone_chunks(chunks)

    def fermion_terms(self, fermion=None, **params):
        """Return qMERA mode terms from a unified :class:`Fermion` helper.

        qMERA's fermionic path represents each physical site as the explicit
        ``("up", "down")`` pair of two-state modes. Keeping this conversion
        on the builder keeps the representation choice in one place while
        allowing the regular local-term and optimizer machinery to handle the
        result. If the builder was created with ``fermion=...``, the argument
        can be omitted. Pass physical couplings (for example ``t=`` and
        ``U=``) explicitly; ``Fermion`` deliberately does not store them.
        """
        fermion = self.fermion if fermion is None else fermion
        if fermion is None:
            raise TypeError(
                "Provide fermion=... to QMeraBuilder or pass a Fermion helper "
                "to fermion_terms(...)."
            )
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

    def majorana_terms(self, fermion=None, **params):
        """Return native parity-preserving Majorana terms for this geometry."""
        fermion = self.fermion if fermion is None else fermion
        if fermion is None:
            raise TypeError(
                "Provide fermion=... to QMeraBuilder or pass a Fermion helper "
                "to majorana_terms(...)."
            )
        if not callable(getattr(fermion, "majorana_terms", None)):
            raise TypeError("fermion must provide majorana_terms(...).")
        return tuple(fermion.majorana_terms(self.geometry, **params))

    def fermion_parametric_loss(
        self,
        fermion=None,
        parameters=None,
        schedule=None,
        *,
        term_params=None,
        **loss_kwargs,
    ):
        """Evaluate qMERA energy directly from a unified ``Fermion`` model."""
        if parameters is None:
            raise TypeError("parameters must be supplied for fermion_parametric_loss.")
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
        fermion=None,
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

    def majorana_parametric_loss(
        self,
        fermion=None,
        parameters=None,
        schedule=None,
        *,
        term_params=None,
        **loss_kwargs,
    ):
        """Evaluate a native ``Z2`` Majorana qMERA energy."""
        if parameters is None:
            raise TypeError("parameters must be supplied for majorana_parametric_loss.")
        schedule = self.build_schedule() if schedule is None else schedule
        terms = self.majorana_terms(fermion, **dict(term_params or {}))
        return self.parametric_loss(
            parameters,
            terms,
            schedule=schedule,
            **loss_kwargs,
        )

    def majorana_parametric_optimizer(
        self,
        fermion=None,
        *,
        schedule=None,
        term_params=None,
        **optimizer_kwargs,
    ):
        """Create a qMERA optimizer for native ``Z2`` Majorana terms."""
        schedule = self.build_schedule() if schedule is None else schedule
        terms = self.majorana_terms(fermion, **dict(term_params or {}))
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
        group_terms=True,
        path_cache=None,
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
            group_terms=group_terms,
            path_cache=path_cache,
        )

    def direct_parametric_loss(
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
        contract_opts=None,
        group_terms=True,
        path_cache=None,
    ):
        """Evaluate the full direct-gate TN as a validation oracle."""
        schedule = self.build_schedule() if schedule is None else schedule
        backend = self.array_backend if array_backend is None else array_backend
        return qmera_direct_parametric_energy(
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
            contract_opts=contract_opts,
            group_terms=group_terms,
            path_cache=path_cache,
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
        """Initialize one parameter vector per unique sharing key."""
        schedule = self.build_schedule() if schedule is None else schedule
        rng = np.random.default_rng(self.seed if seed is None else seed)
        scale = self.param_scale if scale is None else float(scale)
        converter = self._parameter_converter()
        params = {}
        for placement in schedule.placements:
            if placement.param_key in params:
                continue
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
