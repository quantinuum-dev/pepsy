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
    _product_state_for_schedule,
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


def _is_native_symmray_spec(spec):
    """Return whether ``spec`` builds graded Symmray tensors."""
    return (
        str(getattr(spec, "name", "")).lower().startswith("symmray-")
        or str(getattr(spec, "convention", "")).lower().startswith("symmray-")
    )


def _is_native_fermionic_operator(operator):
    """Return whether an operator payload is a Symmray graded array."""
    return bool(getattr(operator, "fermionic", False)) or (
        "fermionicarray" in type(operator).__name__.lower()
    )


def _is_native_symmray_array(value):
    """Return whether ``value`` is a native graded Symmray array.

    The check intentionally uses the public Symmray surface rather than an
    exact class name so that both block-sparse and flat fermionic arrays are
    accepted across Symmray versions.
    """
    return bool(
        getattr(value, "fermionic", False)
        and callable(getattr(value, "tensordot", None))
        and callable(getattr(value, "transpose", None))
        and hasattr(value, "duals")
        and hasattr(value, "indices")
    )


def _native_symmray_requested(schedule, chunks, gate_registry):
    """Return whether ``chunks`` require the graded Symmray route."""
    placements = schedule.placements_by_id()
    return any(
        _is_native_symmray_spec(
            gate_registry.get(placements[gate_id].gate_family)
        )
        for chunk in chunks
        for gate_id in chunk.schedule_placement_ids
    ) or any(
        _is_native_fermionic_operator(chunk.term.operator)
        for chunk in chunks
    )


def _validate_native_symmray_compile(
    schedule,
    chunks,
    gate_registry,
    *,
    product_state_factory=None,
):
    """Validate that native compilation has a graded product-state source."""
    native_requested = _native_symmray_requested(schedule, chunks, gate_registry)
    if native_requested and product_state_factory is None:
        raise ValueError(
            "Native Symmray qMERA compilation requires a graded "
            "product_state_factory (for example, "
            "QMeraSymmrayFermionBackend.product_state) so that the frozen "
            "contraction constants retain charge maps, duals, and fermionic "
            "ordering."
        )
    return native_requested


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
    contraction_backend: str = "array"
    symmetry: str | None = None
    fermionic: bool = False

    @property
    def schedule_placement_ids(self):
        """Gate ids selected by this compiled local cone."""
        return self.chunk.schedule_placement_ids

    @property
    def num_gates(self):
        """Number of parametrized gates used by this compiled local cone."""
        return self.chunk.num_gates

    @property
    def is_graded(self):
        """Whether this expression uses native graded Symmray arrays."""
        return self.fermionic


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


def _expression_topology_key(inputs, output, shapes):
    """Canonical, hashable topology key for a local contraction expression."""
    labels = {}

    def canonical(label):
        if label not in labels:
            labels[label] = len(labels)
        return labels[label]

    canonical_inputs = tuple(
        tuple(canonical(label) for label in tensor_inputs)
        for tensor_inputs in inputs
    )
    canonical_output = tuple(canonical(label) for label in output)
    return canonical_inputs, canonical_output, tuple(tuple(shape) for shape in shapes)


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
    product_state_factory=None,
):
    placements = schedule.placements_by_id()
    state = _product_state_for_schedule(
        schedule,
        chunk.input_sites,
        physical_dim=physical_dim,
        array_backend=array_backend,
        product_state_factory=product_state_factory,
    )
    params = _dummy_params(schedule, chunk, gate_registry)
    for gate_id in chunk.schedule_placement_ids:
        placement = placements[gate_id]
        gate = _gate_for_placement(
            placement,
            params,
            gate_registry=gate_registry,
            array_backend=None,
            schedule=schedule,
        )
        state = state.gate_inds(
            gate,
            inds=tuple(_site_ind(site) for site in placement.where),
            contract=False,
            tags=placement.tags,
            inplace=False,
        )
    return state


def _expression_from_tn(tn, *, optimize, expression_opts=None, path_cache=None):
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
    if path_cache is not None:
        optimize = path_cache.resolve(
            optimize,
            key=_expression_topology_key(inputs, (), shapes),
        )
    if any(_is_native_symmray_array(tensor.data) for tensor in tensors):
        if opts.get("implementation") is not None:
            raise ValueError(
                "Native Symmray qMERA compilation must leave the cotengra "
                "implementation unset so Symmray's graded autoray dispatch "
                "is selected from the runtime arrays."
            )
    # Leave the implementation options unset. Cotengra's default expression
    # builder traces the static expression and dispatches each runtime
    # pairwise contraction through the backend inferred from the native
    # Symmray operands. Forcing an implementation at trace time would feed
    # NumPy lazy placeholders into the constants folder and lose the graded
    # array object before evaluation.
    if not slots:
        # A local term can have an empty reverse lightcone (for example an
        # onsite operator on a schedule with no active gate). Cotengra's
        # constants-folding helper expects at least one lazy input, so
        # evaluate this immutable scalar once and expose the same zero-arg
        # callable interface as a dynamic expression.
        static_value = ctg.array_contract(
            tuple(tensor.data for tensor in tensors),
            inputs,
            output=(),
            optimize=optimize,
            **opts,
        )

        def static_expression(*_arrays):
            return static_value

        return static_expression, (), len(tensors)
    expr = ctg.array_contract_expression(
        inputs,
        output=(),
        shapes=shapes,
        optimize=optimize,
        constants=constants,
        **opts,
    )
    return expr, tuple(slots), len(tensors)


def _compiled_tns(
    schedule,
    chunk,
    *,
    gate_registry,
    array_backend,
    physical_dim,
    product_state_factory=None,
):
    ket = _static_lightcone_state(
        schedule,
        chunk,
        gate_registry=gate_registry,
        array_backend=array_backend,
        physical_dim=physical_dim,
        product_state_factory=product_state_factory,
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


def _native_tn_metadata(tn):
    """Return ``(fermionic, symmetry)`` metadata for a compiled TN."""
    arrays = tuple(tensor.data for tensor in tn)
    native = tuple(value for value in arrays if _is_native_symmray_array(value))
    if not native:
        return False, None
    if len(native) != len(arrays):
        raise TypeError(
            "Native Symmray qMERA compilation requires every tensor in the "
            "frozen local cone to be a graded Symmray array; a dense tensor "
            "would drop fermionic signs or charge-sector metadata."
        )
    symmetries = {str(getattr(value, "symmetry", "")) for value in native}
    if len(symmetries) != 1:
        raise ValueError(
            "Native Symmray qMERA lightcones must use one compatible symmetry; "
            f"found {sorted(symmetries)!r}."
        )
    return True, next(iter(symmetries))


def compile_qmera_parametric_lightcone(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    *,
    gate_registry=None,
    array_backend=None,
    physical_dim=2,
    optimize="auto-hq",
    expression_opts=None,
    product_state_factory=None,
    path_cache=None,
):
    """Compile static numerator and denominator contractions for ``chunk``.

    Native Symmray chunks are compiled as graded-array expressions. Their
    static product state and local operator remain Symmray objects, while
    cotengra freezes only the contraction topology.
    """
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    native_requested = _validate_native_symmray_compile(
        schedule,
        (chunk,),
        gate_registry,
        product_state_factory=product_state_factory,
    )
    numerator, denominator = _compiled_tns(
        schedule,
        chunk,
        gate_registry=gate_registry,
        array_backend=array_backend,
        physical_dim=physical_dim,
        product_state_factory=product_state_factory,
    )
    fermionic, symmetry = _native_tn_metadata(numerator)
    if native_requested and not fermionic:
        raise TypeError(
            "Native Symmray qMERA compilation did not produce a fully graded "
            "frozen local cone. Check product_state_factory and gate registry."
        )
    numerator_expr, numerator_slots, num_num_tensors = _expression_from_tn(
        numerator,
        optimize=optimize,
        expression_opts=expression_opts,
        path_cache=path_cache,
    )
    denominator_expr, denominator_slots, num_den_tensors = _expression_from_tn(
        denominator,
        optimize=optimize,
        expression_opts=expression_opts,
        path_cache=path_cache,
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
        contraction_backend="symmray" if fermionic else "array",
        symmetry=symmetry,
        fermionic=fermionic,
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
    product_state_factory=None,
    path_cache=None,
):
    """Compile every qMERA local cone in ``chunks``.

    The compiled expressions are static and can be evaluated repeatedly with
    new native Symmray gate arrays without rebuilding the qMERA lightcones.
    """
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    chunks = tuple(chunks)
    _validate_native_symmray_compile(
        schedule,
        chunks,
        gate_registry,
        product_state_factory=product_state_factory,
    )
    return tuple(
        compile_qmera_parametric_lightcone(
            schedule,
            chunk,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            optimize=optimize,
            expression_opts=expression_opts,
            product_state_factory=product_state_factory,
            path_cache=path_cache,
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
            schedule=schedule,
        )
    return gates


def _arrays_for_slots(slots, gates):
    arrays = []
    for gate_id, conjugate in slots:
        gate = gates[gate_id]
        arrays.append(ar.do("conj", gate) if conjugate else gate)
    return tuple(arrays)


def _validate_compiled_gate_arrays(compiled, gates):
    """Ensure a graded expression is evaluated with graded gate arrays."""
    if not compiled.fermionic:
        return
    if not all(_is_native_symmray_array(gate) for gate in gates.values()):
        raise TypeError(
            "This compiled qMERA lightcone was built for native Symmray "
            "fermionic gates, but evaluation received a dense gate array. "
            "Use the same Symmray gate registry used during compilation."
        )
    if compiled.symmetry is not None and gates:
        found = {str(getattr(gate, "symmetry", "")) for gate in gates.values()}
        if found != {compiled.symmetry}:
            raise ValueError(
                "Compiled Symmray qMERA gate symmetry does not match the "
                f"frozen lightcone: expected {compiled.symmetry!r}, found "
                f"{sorted(found)!r}."
            )


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
    _validate_compiled_gate_arrays(compiled, gates)
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
    product_state_factory=None,
    path_cache=None,
):
    """Evaluate qMERA energy from precompiled local-cone expressions."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    if compiled_chunks is not None:
        chunks_for_guard = tuple(chunks or ())
        if chunks_for_guard:
            _validate_native_symmray_compile(
                schedule,
                chunks_for_guard,
                gate_registry,
                product_state_factory=product_state_factory,
            )
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
        chunks = tuple(chunks)
        _validate_native_symmray_compile(
            schedule,
            chunks,
            gate_registry,
            product_state_factory=product_state_factory,
        )
        compiled_chunks = compile_qmera_parametric_lightcones(
            schedule,
            chunks,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            optimize=optimize,
            expression_opts=expression_opts,
            product_state_factory=product_state_factory,
            path_cache=path_cache,
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
