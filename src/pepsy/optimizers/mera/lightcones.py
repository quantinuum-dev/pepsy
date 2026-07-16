"""Reverse-lightcone selection and local expectation kernels for MERA states."""

from __future__ import annotations

from dataclasses import dataclass, replace

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import get_default_array_backend
from .gates import default_gate_registry, resolve_gate_spec
from .terms import LocalTerm, convert_local_terms, normalize_local_terms

__all__ = [
    "LightconeChunk",
    "QMeraParametricLightconeChunk",
    "build_lightcone_chunks",
    "build_qmera_lightcone_chunks",
    "build_qmera_parametric_lightcone_chunks",
    "local_qmera_parametric_lightcone_expectation",
    "local_lightcone_expectation",
    "qmera_parametric_energy",
    "qmera_parametric_lightcone_state",
    "select_lightcone",
    "site_tags_for_where",
]


@dataclass(frozen=True)
class LightconeChunk:
    """Cached selector metadata for one local Hamiltonian term."""

    term: LocalTerm
    tags: tuple[str, ...]
    num_tensors: int
    num_indices: int
    outer_inds: tuple[str, ...]
    physical_outer_inds: tuple[str, ...]
    source: str = "tags"
    schedule_placement_ids: tuple[str, ...] = ()
    schedule_width_by_scale: tuple[tuple[int, int], ...] = ()

    @property
    def support_size(self):
        """Number of physical sites in the Hamiltonian support."""
        return len(self.term.where)

    @property
    def physical_width(self):
        """Number of physical outer legs left by the selected lightcone."""
        return len(self.physical_outer_inds)

    @property
    def schedule_width(self):
        """Largest register support touched by the schedule lightcone."""
        if not self.schedule_width_by_scale:
            return self.support_size
        return max(width for _, width in self.schedule_width_by_scale)


@dataclass(frozen=True)
class QMeraParametricLightconeChunk:
    """Schedule-only lightcone metadata for rebuilding a local qMERA cone."""

    term: LocalTerm
    tags: tuple[str, ...]
    input_sites: tuple[int, ...]
    schedule_placement_ids: tuple[str, ...]
    schedule_width_by_scale: tuple[tuple[int, int], ...]
    source: str = "parametric-schedule"

    @property
    def support_size(self):
        """Number of physical sites in the Hamiltonian support."""
        return len(self.term.where)

    @property
    def physical_width(self):
        """Number of product-state sites needed by this local cone."""
        return len(self.input_sites)

    @property
    def schedule_width(self):
        """Largest register support touched by the schedule lightcone."""
        if not self.schedule_width_by_scale:
            return self.support_size
        return max(width for _, width in self.schedule_width_by_scale)

    @property
    def num_gates(self):
        """Number of parametrized gates rebuilt for this local cone."""
        return len(self.schedule_placement_ids)


def _default_site_tag(site):
    return f"I{site}"


def _default_site_ind(site):
    return f"k{site}"


def _call_site_method(state, method_name, site, fallback):
    method = getattr(state, method_name, None)
    if callable(method):
        return method(site)
    return fallback(site)


def site_tags_for_where(state, where, *, validate=True):
    """Return stable physical site tags for a local support."""
    tags = tuple(
        _call_site_method(state, "site_tag", site, _default_site_tag)
        for site in tuple(where)
    )
    if validate:
        state_tags = getattr(state, "tags", None)
        if state_tags is not None:
            missing = [tag for tag in tags if tag not in state_tags]
            if missing:
                raise ValueError(
                    "state does not contain site tag(s) for local support: "
                    f"{missing!r}."
                )
    return tags


def _site_inds_for_where(state, where):
    return tuple(
        _call_site_method(state, "site_ind", site, _default_site_ind)
        for site in tuple(where)
    )


def select_lightcone(state, *, where=None, tags=None, which="any", validate=True):
    """Select the reverse lightcone subnetwork for ``where`` or ``tags``."""
    if tags is None:
        if where is None:
            raise ValueError("select_lightcone requires either where or tags.")
        tags = site_tags_for_where(state, where, validate=validate)
    tags = tuple(tags)
    selected = state.select(tags, which=which)
    if getattr(selected, "num_tensors", 0) == 0:
        raise ValueError(f"lightcone selection for tags {tags!r} is empty.")
    return selected


def _chunk_for_term(state, term, *, validate=True):
    tags = term.tags
    if tags is None:
        tags = site_tags_for_where(state, term.where, validate=validate)
        term = term.with_tags(tags)
    else:
        tags = tuple(tags)

    selected = select_lightcone(state, tags=tags, validate=validate)
    outer_inds = tuple(selected.outer_inds())
    physical_inds = set(_site_inds_for_where(state, term.where))
    physical_outer_inds = tuple(ind for ind in outer_inds if ind in physical_inds)
    return LightconeChunk(
        term=term,
        tags=tags,
        num_tensors=int(selected.num_tensors),
        num_indices=int(selected.num_indices),
        outer_inds=outer_inds,
        physical_outer_inds=physical_outer_inds,
    )


def build_lightcone_chunks(state, terms, *, validate=True):
    """Precompute one lightcone chunk per normalized local term."""
    return tuple(_chunk_for_term(state, term, validate=validate) for term in terms)


def _term_on_schedule_registers(schedule, term):
    register_where = schedule.geometry.to_register_where(term.where)
    metadata = dict(term.metadata or {})
    metadata.setdefault("original_where", tuple(term.where))
    metadata["register_where"] = register_where
    return replace(term, where=register_where, metadata=metadata)


def _schedule_width_trace(schedule, where, placements):
    selected_ids = {placement.gate_id for placement in placements}
    support = set(schedule.geometry.to_register_where(where))
    width_by_scale = {}
    for placement in reversed(schedule.placements):
        if placement.gate_id not in selected_ids:
            continue
        support.update(placement.where)
        width_by_scale[placement.scale] = max(
            width_by_scale.get(placement.scale, 0),
            len(support),
        )
    return tuple(
        (scale, width_by_scale[scale])
        for scale in sorted(width_by_scale)
    )


def _chunk_for_schedule_term(state, schedule, term, *, validate=True):
    term = _term_on_schedule_registers(schedule, term)
    placements = schedule.reverse_lightcone_placements(term.where)
    tags = schedule.reverse_lightcone_tags(term.where)
    chunk = _chunk_for_term(state, term.with_tags(tags), validate=validate)
    return replace(
        chunk,
        source="schedule",
        schedule_placement_ids=tuple(placement.gate_id for placement in placements),
        schedule_width_by_scale=_schedule_width_trace(
            schedule,
            term.where,
            placements,
        ),
    )


def _schedule_placements_for_term(schedule, term):
    term = _term_on_schedule_registers(schedule, term)
    placements = schedule.reverse_lightcone_placements(term.where)
    return term, placements


def _input_sites_for_placements(schedule, term, placements):
    support = set(term.where)
    for placement in placements:
        support.update(placement.where)
    return tuple(site for site in schedule.geometry.register_sites if site in support)


def build_qmera_lightcone_chunks(state, schedule, terms, *, validate=True):
    """Precompute lightcone chunks from an explicit qMERA schedule.

    Unlike :func:`build_lightcone_chunks`, this follows
    :class:`~pepsy.optimizers.mera.QMeraSchedule` placements first and only then
    turns the selected sites/gates into tensor-network tags. This keeps local
    energy chunks tied to the designed RG blocks rather than to a generic tag
    query on an already-built network.
    """
    return tuple(
        _chunk_for_schedule_term(state, schedule, term, validate=validate)
        for term in terms
    )


def _parametric_chunk_for_schedule_term(schedule, term):
    term, placements = _schedule_placements_for_term(schedule, term)
    return QMeraParametricLightconeChunk(
        term=term.with_tags(schedule.reverse_lightcone_tags(term.where)),
        tags=schedule.reverse_lightcone_tags(term.where),
        input_sites=_input_sites_for_placements(schedule, term, placements),
        schedule_placement_ids=tuple(placement.gate_id for placement in placements),
        schedule_width_by_scale=_schedule_width_trace(
            schedule,
            term.where,
            placements,
        ),
    )


def build_qmera_parametric_lightcone_chunks(schedule, terms):
    """Precompute qMERA local cones without requiring a built global state."""
    return tuple(
        _parametric_chunk_for_schedule_term(schedule, term)
        for term in terms
    )


def _maybe_real(value):
    try:
        return ar.do("real", value)
    except Exception:  # pragma: no cover - defensive for unusual scalar types
        return value.real


def _maybe_simplify(tn, simplify):
    if not simplify:
        return tn
    seq = "R" if simplify is True else simplify
    return tn.full_simplify(seq=seq, inplace=False)


def _contract(tn, *, optimize, contract_opts):
    return tn.contract(all, optimize=optimize, **dict(contract_opts or {}))


def _site_ind(site):
    return f"k{site}"


def _product_state_on_sites(sites, *, physical_dim=2, array_backend=None):
    tensors = []
    base = np.zeros((int(physical_dim),), dtype=np.complex128)
    base[0] = 1.0
    for site in tuple(sites):
        data = base if array_backend is None else array_backend(base)
        tensors.append(qtn.Tensor(data, inds=(_site_ind(site),), tags=(f"I{site}",)))
    return qtn.TensorNetwork(tensors)


def _resolve_array_backend(array_backend):
    return get_default_array_backend() if array_backend is None else array_backend


def _placements_by_id(schedule):
    by_id = schedule.placements_by_id()
    return by_id if isinstance(by_id, dict) else dict(by_id)


def _gate_for_placement(placement, parameters, *, gate_registry, array_backend=None):
    try:
        params = parameters[placement.param_key]
    except KeyError as exc:
        raise KeyError(
            f"Missing parameter {placement.param_key!r} for qMERA gate "
            f"{placement.gate_id!r}."
        ) from exc
    spec = resolve_gate_spec(placement.gate_family, gate_registry)
    if spec.arity != placement.arity:
        raise ValueError(
            f"Gate family {spec.name!r} has arity {spec.arity}, "
            f"but placement {placement.gate_id} acts on {placement.arity} sites."
        )
    return spec.matrix(params, array_backend=array_backend)


def qmera_parametric_lightcone_state(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    physical_dim=2,
    contract=False,
):
    """Build only the local qMERA cone selected by ``chunk`` from parameters."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    array_backend = _resolve_array_backend(array_backend)
    placements = _placements_by_id(schedule)
    state = _product_state_on_sites(
        chunk.input_sites,
        physical_dim=physical_dim,
        array_backend=array_backend,
    )
    for gate_id in chunk.schedule_placement_ids:
        placement = placements[gate_id]
        gate = _gate_for_placement(
            placement,
            parameters,
            gate_registry=gate_registry,
            array_backend=array_backend,
        )
        state = state.gate_inds(
            gate,
            inds=tuple(_site_ind(site) for site in placement.where),
            contract=contract,
            tags=placement.tags,
            inplace=False,
        )
    return state


def local_qmera_parametric_lightcone_expectation(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    physical_dim=2,
    optimize="auto-hq",
    normalized=True,
    real=True,
    simplify=False,
    gate_contract=True,
    contract_opts=None,
):
    """Contract one qMERA local term by rebuilding only its scheduled cone."""
    ket = qmera_parametric_lightcone_state(
        schedule,
        chunk,
        parameters,
        gate_registry=gate_registry,
        array_backend=array_backend,
        physical_dim=physical_dim,
        contract=False,
    )
    term = chunk.term
    ket_g = ket.gate_inds(
        term.operator,
        inds=tuple(_site_ind(site) for site in term.where),
        contract=gate_contract,
        inplace=False,
    )
    expec = _maybe_simplify(ket.H & ket_g, simplify)
    value = _contract(expec, optimize=optimize, contract_opts=contract_opts)

    if normalized:
        norm = _contract(
            _maybe_simplify(ket.H & ket, simplify),
            optimize=optimize,
            contract_opts=contract_opts,
        )
        value = value / norm

    if term.weight != 1.0:
        value = value * term.weight

    if real:
        value = _maybe_real(value)
    return value


def qmera_parametric_energy(
    schedule,
    parameters,
    hamiltonian=None,
    *,
    chunks=None,
    gate_registry=None,
    array_backend=None,
    convert_terms=True,
    physical_dim=2,
    optimize="auto-hq",
    normalized=True,
    energy_per_site=True,
    real=True,
    simplify=False,
    gate_contract=True,
    contract_opts=None,
):
    """Evaluate a qMERA energy by rebuilding each scheduled lightcone only."""
    array_backend = _resolve_array_backend(array_backend)
    if chunks is None:
        if hamiltonian is None:
            raise ValueError("qmera_parametric_energy requires hamiltonian or chunks.")
        terms = normalize_local_terms(hamiltonian)
        if convert_terms:
            terms = convert_local_terms(terms, array_backend)
        chunks = build_qmera_parametric_lightcone_chunks(schedule, terms)
    value = None
    for chunk in chunks:
        term_value = local_qmera_parametric_lightcone_expectation(
            schedule,
            chunk,
            parameters,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            optimize=optimize,
            normalized=normalized,
            real=False,
            simplify=simplify,
            gate_contract=gate_contract,
            contract_opts=contract_opts,
        )
        value = term_value if value is None else value + term_value
    if value is None:
        raise ValueError("hamiltonian contains no local terms.")
    if energy_per_site:
        value = value / schedule.geometry.num_sites
    if real:
        value = _maybe_real(value)
    return value


def local_lightcone_expectation(
    state,
    chunk: LightconeChunk,
    *,
    optimize="auto-hq",
    normalized=True,
    real=True,
    simplify=False,
    gate_contract=True,
    contract_opts=None,
):
    """Contract one local expectation value over a MERA reverse lightcone."""
    term = chunk.term
    ket = select_lightcone(state, tags=chunk.tags, validate=False)
    ket_g = ket.gate(
        term.operator,
        term.where,
        contract=gate_contract,
        inplace=False,
    )
    expec = _maybe_simplify(ket.H & ket_g, simplify)
    value = _contract(expec, optimize=optimize, contract_opts=contract_opts)

    if normalized:
        norm = _contract(
            _maybe_simplify(ket.H & ket, simplify),
            optimize=optimize,
            contract_opts=contract_opts,
        )
        value = value / norm

    if term.weight != 1.0:
        value = value * term.weight

    if real:
        value = _maybe_real(value)
    return value
