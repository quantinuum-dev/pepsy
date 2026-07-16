"""Compiled qMERA local-cone contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import autoray as ar
import cotengra as ctg

from .gates import default_gate_registry
from .lightcones import (
    QMeraParametricLightconeChunk,
    _gate_for_placement,
    _maybe_real,
    _product_state_on_sites,
    _site_ind,
    build_qmera_parametric_lightcone_chunks,
)
from .terms import convert_local_terms, normalize_local_terms

__all__ = [
    "QMeraCompiledLightconeChunk",
    "compile_qmera_parametric_lightcone",
    "compile_qmera_parametric_lightcones",
    "local_qmera_compiled_lightcone_expectation",
    "qmera_compiled_parametric_energy",
]

_BRA_TAG = "_QMERA_BRA_COPY"
_KET_TAG = "_QMERA_KET_COPY"


@dataclass(frozen=True)
class QMeraCompiledLightconeChunk:
    """Static contraction expressions for one qMERA local cone."""

    chunk: QMeraParametricLightconeChunk
    numerator_expr: Any
    denominator_expr: Any
    numerator_slots: tuple[tuple[str, bool], ...]
    denominator_slots: tuple[tuple[str, bool], ...]
    num_numerator_tensors: int
    num_denominator_tensors: int
    optimize: Any = "auto-hq"

    @property
    def schedule_placement_ids(self):
        """Gate ids selected by this compiled local cone."""
        return self.chunk.schedule_placement_ids

    @property
    def num_gates(self):
        """Number of parametrized gates used by this compiled local cone."""
        return self.chunk.num_gates


def _gate_tag_to_id(tag):
    text = str(tag)
    return text[5:] if text.startswith("GATE_") else None


def _gate_id_for_tensor(tensor):
    gate_ids = []
    for tag in tensor.tags:
        gate_id = _gate_tag_to_id(tag)
        if gate_id is not None:
            gate_ids.append(gate_id)
    if not gate_ids:
        return None
    if len(gate_ids) != 1:
        raise ValueError(
            "compiled qMERA lightcones require uncontracted single-gate tensors; "
            f"found gate tags {gate_ids!r}."
        )
    return gate_ids[0]


def _copy_with_tag(tn, tag):
    out = tn.copy()
    out.add_tag(tag)
    return out


def _dummy_params(schedule, chunk, gate_registry):
    placements = schedule.placements_by_id()
    params = {}
    for gate_id in chunk.schedule_placement_ids:
        placement = placements[gate_id]
        spec = gate_registry.get(placement.gate_family)
        params[placement.param_key] = [0.0] * spec.num_params
    return params


def _static_lightcone_state(
    schedule,
    chunk,
    *,
    gate_registry,
    array_backend,
    physical_dim,
):
    placements = schedule.placements_by_id()
    state = _product_state_on_sites(
        chunk.input_sites,
        physical_dim=physical_dim,
        array_backend=array_backend,
    )
    params = _dummy_params(schedule, chunk, gate_registry)
    for gate_id in chunk.schedule_placement_ids:
        placement = placements[gate_id]
        gate = _gate_for_placement(
            placement,
            params,
            gate_registry=gate_registry,
            array_backend=None,
        )
        state = state.gate_inds(
            gate,
            inds=tuple(_site_ind(site) for site in placement.where),
            contract=False,
            tags=placement.tags,
            inplace=False,
        )
    return state


def _expression_from_tn(tn, *, optimize, expression_opts=None):
    inputs = []
    shapes = []
    constants = {}
    slots = []
    tensors = tuple(tn.tensor_map.values())
    for pos, tensor in enumerate(tensors):
        inputs.append(tuple(tensor.inds))
        shapes.append(tuple(tensor.shape))
        gate_id = _gate_id_for_tensor(tensor)
        if gate_id is None:
            constants[pos] = tensor.data
            continue
        tags = set(tensor.tags)
        if _BRA_TAG in tags:
            slots.append((gate_id, True))
        elif _KET_TAG in tags:
            slots.append((gate_id, False))
        else:
            raise ValueError(
                f"Could not identify bra/ket copy for qMERA gate {gate_id!r}."
            )
    opts = {} if expression_opts is None else dict(expression_opts)
    expr = ctg.array_contract_expression(
        inputs,
        output=(),
        shapes=shapes,
        optimize=optimize,
        constants=constants,
        **opts,
    )
    return expr, tuple(slots), len(tensors)


def _compiled_tns(schedule, chunk, *, gate_registry, array_backend, physical_dim):
    ket = _static_lightcone_state(
        schedule,
        chunk,
        gate_registry=gate_registry,
        array_backend=array_backend,
        physical_dim=physical_dim,
    )
    bra = _copy_with_tag(ket.H, _BRA_TAG)
    ket_side = _copy_with_tag(ket, _KET_TAG)
    numerator = bra & ket_side.gate_inds(
        chunk.term.operator,
        inds=tuple(_site_ind(site) for site in chunk.term.where),
        contract=False,
        inplace=False,
    )
    denominator = bra & ket_side
    return numerator, denominator


def compile_qmera_parametric_lightcone(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    *,
    gate_registry=None,
    array_backend=None,
    physical_dim=2,
    optimize="auto-hq",
    expression_opts=None,
):
    """Compile static numerator and denominator contractions for ``chunk``."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    numerator, denominator = _compiled_tns(
        schedule,
        chunk,
        gate_registry=gate_registry,
        array_backend=array_backend,
        physical_dim=physical_dim,
    )
    numerator_expr, numerator_slots, num_num_tensors = _expression_from_tn(
        numerator,
        optimize=optimize,
        expression_opts=expression_opts,
    )
    denominator_expr, denominator_slots, num_den_tensors = _expression_from_tn(
        denominator,
        optimize=optimize,
        expression_opts=expression_opts,
    )
    return QMeraCompiledLightconeChunk(
        chunk=chunk,
        numerator_expr=numerator_expr,
        denominator_expr=denominator_expr,
        numerator_slots=numerator_slots,
        denominator_slots=denominator_slots,
        num_numerator_tensors=num_num_tensors,
        num_denominator_tensors=num_den_tensors,
        optimize=optimize,
    )


def compile_qmera_parametric_lightcones(
    schedule,
    chunks,
    *,
    gate_registry=None,
    array_backend=None,
    physical_dim=2,
    optimize="auto-hq",
    expression_opts=None,
):
    """Compile every qMERA local cone in ``chunks``."""
    return tuple(
        compile_qmera_parametric_lightcone(
            schedule,
            chunk,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            optimize=optimize,
            expression_opts=expression_opts,
        )
        for chunk in chunks
    )


def _placement_gates(schedule, compiled, parameters, *, gate_registry, gate_array_backend):
    placements = schedule.placements_by_id()
    gates = {}
    for gate_id in compiled.schedule_placement_ids:
        placement = placements[gate_id]
        gates[gate_id] = _gate_for_placement(
            placement,
            parameters,
            gate_registry=gate_registry,
            array_backend=gate_array_backend,
        )
    return gates


def _arrays_for_slots(slots, gates):
    arrays = []
    for gate_id, conjugate in slots:
        gate = gates[gate_id]
        arrays.append(ar.do("conj", gate) if conjugate else gate)
    return tuple(arrays)


def local_qmera_compiled_lightcone_expectation(
    schedule,
    compiled: QMeraCompiledLightconeChunk,
    parameters,
    *,
    gate_registry=None,
    gate_array_backend=None,
    normalized=True,
    real=True,
):
    """Evaluate one compiled qMERA local-cone expectation from ``parameters``."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    gates = _placement_gates(
        schedule,
        compiled,
        parameters,
        gate_registry=gate_registry,
        gate_array_backend=gate_array_backend,
    )
    numerator = compiled.numerator_expr(
        *_arrays_for_slots(compiled.numerator_slots, gates)
    )
    if normalized:
        denominator = compiled.denominator_expr(
            *_arrays_for_slots(compiled.denominator_slots, gates)
        )
        numerator = numerator / denominator
    term = compiled.chunk.term
    if term.weight != 1.0:
        numerator = numerator * term.weight
    if real:
        numerator = _maybe_real(numerator)
    return numerator


def qmera_compiled_parametric_energy(
    schedule,
    parameters,
    hamiltonian=None,
    *,
    compiled_chunks=None,
    chunks=None,
    gate_registry=None,
    array_backend=None,
    gate_array_backend=None,
    convert_terms=True,
    physical_dim=2,
    optimize="auto-hq",
    normalized=True,
    energy_per_site=True,
    real=True,
    expression_opts=None,
):
    """Evaluate qMERA energy from precompiled local-cone expressions."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    if compiled_chunks is None:
        if chunks is None:
            if hamiltonian is None:
                raise ValueError(
                    "qmera_compiled_parametric_energy requires hamiltonian, "
                    "chunks, or compiled_chunks."
                )
            terms = normalize_local_terms(hamiltonian)
            if convert_terms:
                terms = convert_local_terms(terms, array_backend)
            chunks = build_qmera_parametric_lightcone_chunks(schedule, terms)
        compiled_chunks = compile_qmera_parametric_lightcones(
            schedule,
            chunks,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            optimize=optimize,
            expression_opts=expression_opts,
        )
    value = None
    for compiled in compiled_chunks:
        term_value = local_qmera_compiled_lightcone_expectation(
            schedule,
            compiled,
            parameters,
            gate_registry=gate_registry,
            gate_array_backend=gate_array_backend,
            normalized=normalized,
            real=False,
        )
        value = term_value if value is None else value + term_value
    if value is None:
        raise ValueError("hamiltonian contains no local terms.")
    if energy_per_site:
        value = value / schedule.geometry.num_sites
    if real:
        value = _maybe_real(value)
    return value
