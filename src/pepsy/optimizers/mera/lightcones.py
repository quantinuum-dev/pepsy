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
    "QMeraLightconeGroup",
    "QMeraLightconeTN",
    "QMeraParametricLightconeChunk",
    "build_lightcone_chunks",
    "build_qmera_lightcone_chunks",
    "build_qmera_parametric_lightcone_chunks",
    "group_qmera_parametric_lightcone_chunks",
    "contract_qmera_lightcone_tn",
    "contract_qmera_lightcone_group",
    "lightcone_energy",
    "qmera_direct_parametric_energy",
    "qmera_parametric_state",
    "qmera_parametric_lightcone_group_state",
    "local_qmera_parametric_lightcone_expectation",
    "local_lightcone_expectation",
    "qmera_parametric_energy",
    "qmera_parametric_lightcone_state",
    "qmera_parametric_lightcone_tn",
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


@dataclass(frozen=True)
class QMeraLightconeGroup:
    """Local terms sharing one qMERA reverse-lightcone topology."""

    key: tuple
    chunks: tuple[QMeraParametricLightconeChunk, ...]
    input_sites: tuple[int, ...]
    schedule_placement_ids: tuple[str, ...]

    @property
    def num_terms(self):
        """Number of local Hamiltonian terms in this group."""
        return len(self.chunks)

    @property
    def num_gates(self):
        """Number of gates in the shared local cone."""
        return len(self.schedule_placement_ids)


@dataclass(frozen=True)
class QMeraLightconeTN:
    """Explicit tensor networks for one scheduled qMERA local term."""

    chunk: QMeraParametricLightconeChunk
    ket: qtn.TensorNetwork
    numerator: qtn.TensorNetwork
    denominator: qtn.TensorNetwork

    @property
    def term(self):
        """Local Hamiltonian term represented by this lightcone."""
        return self.chunk.term

    @property
    def num_gates(self):
        """Number of qMERA gates rebuilt inside this lightcone."""
        return self.chunk.num_gates

    @property
    def num_numerator_tensors(self):
        """Number of tensors in the expectation-value network."""
        return int(self.numerator.num_tensors)

    @property
    def num_denominator_tensors(self):
        """Number of tensors in the norm network."""
        return int(self.denominator.num_tensors)


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


def group_qmera_parametric_lightcone_chunks(chunks):
    """Group qMERA chunks that share input support and gate topology."""
    groups = {}
    for chunk in tuple(chunks):
        key = (tuple(chunk.input_sites), tuple(chunk.schedule_placement_ids))
        groups.setdefault(key, []).append(chunk)
    return tuple(
        QMeraLightconeGroup(
            key=key,
            chunks=tuple(group_chunks),
            input_sites=key[0],
            schedule_placement_ids=key[1],
        )
        for key, group_chunks in groups.items()
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


def _apply_local_gate(state, operator, where, *, contract, inplace):
    """Apply a local operator through the state's native TN gate API."""
    gate = getattr(state, "gate", None)
    if callable(gate):
        return gate(
            operator,
            where,
            contract=contract,
            inplace=inplace,
        )
    gate_inds = getattr(state, "gate_inds", None)
    if callable(gate_inds):
        return gate_inds(
            operator,
            inds=_site_inds_for_where(state, where),
            contract=contract,
            inplace=inplace,
        )
    raise TypeError("state must provide gate() or gate_inds().")


def _product_state_on_sites(sites, *, physical_dim=2, array_backend=None):
    tensors = []
    base = np.zeros((int(physical_dim),), dtype=np.complex128)
    base[0] = 1.0
    for site in tuple(sites):
        data = base if array_backend is None else array_backend(base)
        tensors.append(qtn.Tensor(data, inds=(_site_ind(site),), tags=(f"I{site}",)))
    return qtn.TensorNetwork(tensors)


def _product_state_for_schedule(
    schedule,
    sites,
    *,
    physical_dim=2,
    array_backend=None,
    product_state_factory=None,
):
    if product_state_factory is None:
        return _product_state_on_sites(
            sites,
            physical_dim=physical_dim,
            array_backend=array_backend,
        )
    return product_state_factory(
        schedule,
        tuple(sites),
        physical_dim=physical_dim,
        array_backend=array_backend,
    )


def _resolve_array_backend(array_backend):
    return get_default_array_backend() if array_backend is None else array_backend


def _placements_by_id(schedule):
    by_id = schedule.placements_by_id()
    return by_id if isinstance(by_id, dict) else dict(by_id)


def _resolve_contraction_opt(optimize, path_cache, *, key):
    if path_cache is None:
        return optimize
    resolver = getattr(path_cache, "resolve", None)
    if not callable(resolver):
        raise TypeError("path_cache must provide resolve(optimize, key=...).")
    return resolver(optimize, key=key)


def _gate_for_placement(
    placement,
    parameters,
    *,
    gate_registry,
    array_backend=None,
    schedule=None,
):
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
    return spec.matrix_for_placement(
        params,
        placement=placement,
        schedule=schedule,
        array_backend=array_backend,
    )


def qmera_parametric_state(
    schedule,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    gate_array_backend=None,
    physical_dim=2,
    contract=False,
    product_state_factory=None,
):
    """Build the complete direct-gate qMERA state from a parameter map."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    array_backend = _resolve_array_backend(array_backend)
    placements = tuple(schedule.placements)
    state = _product_state_for_schedule(
        schedule,
        schedule.geometry.register_sites,
        physical_dim=physical_dim,
        array_backend=array_backend,
        product_state_factory=product_state_factory,
    )
    for placement in placements:
        gate = _gate_for_placement(
            placement,
            parameters,
            gate_registry=gate_registry,
            array_backend=gate_array_backend,
            schedule=schedule,
        )
        state = state.gate_inds(
            gate,
            inds=tuple(_site_ind(site) for site in placement.where),
            contract=contract,
            tags=placement.tags,
            inplace=False,
        )
    return state


def qmera_parametric_lightcone_state(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    gate_array_backend=None,
    physical_dim=2,
    contract=False,
    product_state_factory=None,
):
    """Build only the local qMERA cone selected by ``chunk`` from parameters."""
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    array_backend = _resolve_array_backend(array_backend)
    placements = _placements_by_id(schedule)
    state = _product_state_for_schedule(
        schedule,
        chunk.input_sites,
        physical_dim=physical_dim,
        array_backend=array_backend,
        product_state_factory=product_state_factory,
    )
    for gate_id in chunk.schedule_placement_ids:
        placement = placements[gate_id]
        gate = _gate_for_placement(
            placement,
            parameters,
            gate_registry=gate_registry,
            array_backend=gate_array_backend,
            schedule=schedule,
        )
        state = state.gate_inds(
            gate,
            inds=tuple(_site_ind(site) for site in placement.where),
            contract=contract,
            tags=placement.tags,
            inplace=False,
        )
    return state


def qmera_parametric_lightcone_group_state(
    schedule,
    group: QMeraLightconeGroup,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    gate_array_backend=None,
    physical_dim=2,
    contract=False,
    product_state_factory=None,
):
    """Build one shared ket for all terms in a lightcone group."""
    representative = QMeraParametricLightconeChunk(
        term=group.chunks[0].term,
        tags=group.chunks[0].tags,
        input_sites=group.input_sites,
        schedule_placement_ids=group.schedule_placement_ids,
        schedule_width_by_scale=group.chunks[0].schedule_width_by_scale,
    )
    return qmera_parametric_lightcone_state(
        schedule,
        representative,
        parameters,
        gate_registry=gate_registry,
        array_backend=array_backend,
        gate_array_backend=gate_array_backend,
        physical_dim=physical_dim,
        contract=contract,
        product_state_factory=product_state_factory,
    )


def qmera_parametric_lightcone_tn(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    gate_array_backend=None,
    physical_dim=2,
    simplify=False,
    gate_contract=True,
    product_state_factory=None,
):
    """Build explicit numerator and norm TNs for one scheduled qMERA cone."""
    ket = qmera_parametric_lightcone_state(
        schedule,
        chunk,
        parameters,
        gate_registry=gate_registry,
        array_backend=array_backend,
        gate_array_backend=gate_array_backend,
        physical_dim=physical_dim,
        contract=False,
        product_state_factory=product_state_factory,
    )
    term = chunk.term
    ket_g = ket.gate_inds(
        term.operator,
        inds=tuple(_site_ind(site) for site in term.where),
        contract=gate_contract,
        inplace=False,
    )
    numerator = _maybe_simplify(ket.H & ket_g, simplify)
    denominator = _maybe_simplify(ket.H & ket, simplify)
    return QMeraLightconeTN(
        chunk=chunk,
        ket=ket,
        numerator=numerator,
        denominator=denominator,
    )


def contract_qmera_lightcone_tn(
    lightcone: QMeraLightconeTN,
    *,
    optimize="auto-hq",
    normalized=True,
    real=True,
    contract_opts=None,
    path_cache=None,
):
    """Contract an explicit qMERA local-cone TN with a cotengra optimizer."""
    optimize = _resolve_contraction_opt(
        optimize,
        path_cache,
        key=(lightcone.chunk.input_sites, lightcone.chunk.schedule_placement_ids),
    )
    value = _contract(
        lightcone.numerator,
        optimize=optimize,
        contract_opts=contract_opts,
    )

    if normalized:
        norm = _contract(
            lightcone.denominator,
            optimize=optimize,
            contract_opts=contract_opts,
        )
        value = value / norm

    term = lightcone.term
    if term.weight != 1.0:
        value = value * term.weight

    if real:
        value = _maybe_real(value)
    return value


def contract_qmera_lightcone_group(
    group: QMeraLightconeGroup,
    ket,
    *,
    optimize="auto-hq",
    normalized=True,
    real=True,
    contract_opts=None,
    path_cache=None,
    gate_contract=True,
):
    """Contract all terms in ``group`` while reusing its ket and norm."""
    optimize = _resolve_contraction_opt(optimize, path_cache, key=group.key)
    denominator = None
    if normalized:
        denominator = _contract(ket.H & ket, optimize=optimize, contract_opts=contract_opts)
    values = []
    for chunk in group.chunks:
        ket_g = ket.gate_inds(
            chunk.term.operator,
            inds=tuple(_site_ind(site) for site in chunk.term.where),
            contract=gate_contract,
            inplace=False,
        )
        value = _contract(ket.H & ket_g, optimize=optimize, contract_opts=contract_opts)
        if normalized:
            value = value / denominator
        if chunk.term.weight != 1.0:
            value = value * chunk.term.weight
        values.append(value)
    value = sum(values[1:], values[0]) if values else 0.0
    return _maybe_real(value) if real else value


def local_qmera_parametric_lightcone_expectation(
    schedule,
    chunk: QMeraParametricLightconeChunk,
    parameters,
    *,
    gate_registry=None,
    array_backend=None,
    gate_array_backend=None,
    physical_dim=2,
    optimize="auto-hq",
    normalized=True,
    real=True,
    simplify=False,
    gate_contract=True,
    contract_opts=None,
    product_state_factory=None,
    path_cache=None,
):
    """Contract one qMERA local term by rebuilding only its scheduled cone."""
    lightcone = qmera_parametric_lightcone_tn(
        schedule,
        chunk,
        parameters,
        gate_registry=gate_registry,
        array_backend=array_backend,
        gate_array_backend=gate_array_backend,
        physical_dim=physical_dim,
        simplify=simplify,
        gate_contract=gate_contract,
        product_state_factory=product_state_factory,
    )
    return contract_qmera_lightcone_tn(
        lightcone,
        optimize=optimize,
        normalized=normalized,
        real=real,
        contract_opts=contract_opts,
        path_cache=path_cache,
    )


def qmera_parametric_energy(
    schedule,
    parameters,
    hamiltonian=None,
    *,
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
    simplify=False,
    gate_contract=True,
    contract_opts=None,
    product_state_factory=None,
    group_terms=True,
    path_cache=None,
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
    # ``simplify`` is implemented by the per-chunk TN builder. Fall back to
    # that path when requested so grouping never changes this public option.
    if group_terms and simplify:
        group_terms = False
    if group_terms:
        for group in group_qmera_parametric_lightcone_chunks(chunks):
            ket = qmera_parametric_lightcone_group_state(
                schedule,
                group,
                parameters,
                gate_registry=gate_registry,
                array_backend=array_backend,
                gate_array_backend=gate_array_backend,
                physical_dim=physical_dim,
                product_state_factory=product_state_factory,
            )
            group_value = contract_qmera_lightcone_group(
                group,
                ket,
                optimize=optimize,
                normalized=normalized,
                real=False,
                contract_opts=contract_opts,
                path_cache=path_cache,
                gate_contract=gate_contract,
            )
            value = group_value if value is None else value + group_value
    else:
        for chunk in chunks:
            term_value = local_qmera_parametric_lightcone_expectation(
                schedule,
                chunk,
                parameters,
                gate_registry=gate_registry,
                array_backend=array_backend,
                gate_array_backend=gate_array_backend,
                physical_dim=physical_dim,
                optimize=optimize,
                normalized=normalized,
                real=False,
                simplify=simplify,
                gate_contract=gate_contract,
                contract_opts=contract_opts,
                product_state_factory=product_state_factory,
                path_cache=path_cache,
            )
            value = term_value if value is None else value + term_value
    if value is None:
        raise ValueError("hamiltonian contains no local terms.")
    if energy_per_site:
        value = value / schedule.geometry.num_sites
    if real:
        value = _maybe_real(value)
    return value


def qmera_direct_parametric_energy(
    schedule,
    parameters,
    hamiltonian=None,
    *,
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
    contract_opts=None,
    group_terms=True,
    path_cache=None,
    product_state_factory=None,
):
    """Evaluate qMERA energy from the complete direct-gate tensor network.

    This is intentionally a validation/debugging oracle for the
    schedule-first local-cone path, not the primary optimization route.
    """
    array_backend = _resolve_array_backend(array_backend)
    if chunks is None:
        if hamiltonian is None:
            raise ValueError("qmera_direct_parametric_energy requires hamiltonian or chunks.")
        terms = normalize_local_terms(hamiltonian)
        if convert_terms:
            terms = convert_local_terms(terms, array_backend)
        chunks = build_qmera_parametric_lightcone_chunks(schedule, terms)
    state = qmera_parametric_state(
        schedule,
        parameters,
        gate_registry=gate_registry,
        array_backend=array_backend,
        gate_array_backend=gate_array_backend,
        physical_dim=physical_dim,
        product_state_factory=product_state_factory,
    )
    groups = group_qmera_parametric_lightcone_chunks(chunks) if group_terms else (
        QMeraLightconeGroup(
            key=(chunk.input_sites, chunk.schedule_placement_ids),
            chunks=(chunk,),
            input_sites=chunk.input_sites,
            schedule_placement_ids=chunk.schedule_placement_ids,
        )
        for chunk in chunks
    )
    value = None
    for group in groups:
        # Direct-state validation uses the full state, while the grouping still
        # shares the norm/path topology for terms with the same local support.
        denominator = None
        group_optimize = _resolve_contraction_opt(
            optimize,
            path_cache,
            key=group.key,
        )
        if normalized:
            denominator = _contract(
                state.H & state,
                optimize=group_optimize,
                contract_opts=contract_opts,
            )
        group_value = 0.0
        for chunk in group.chunks:
            state_g = state.gate_inds(
                chunk.term.operator,
                inds=tuple(_site_ind(site) for site in chunk.term.where),
                contract=True,
                inplace=False,
            )
            term_value = _contract(
                state.H & state_g,
                optimize=group_optimize,
                contract_opts=contract_opts,
            )
            if normalized:
                term_value = term_value / denominator
            if chunk.term.weight != 1.0:
                term_value = term_value * chunk.term.weight
            group_value = group_value + term_value
        value = group_value if value is None else value + group_value
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
    path_cache=None,
):
    """Contract one local expectation value over a MERA reverse lightcone."""
    term = chunk.term
    ket = select_lightcone(state, tags=chunk.tags, validate=False)
    optimize = _resolve_contraction_opt(
        optimize,
        path_cache,
        key=(tuple(chunk.tags), tuple(chunk.physical_outer_inds)),
    )
    ket_g = _apply_local_gate(
        ket,
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


def _lightcone_state_and_schedule(state, schedule):
    """Unwrap an ansatz payload while retaining an optional qMERA schedule."""
    if schedule is None:
        schedule = getattr(state, "schedule", None)
    candidate = getattr(state, "state", state)
    if hasattr(candidate, "select") and (
        hasattr(candidate, "gate") or hasattr(candidate, "gate_inds")
    ):
        return candidate, schedule
    raise TypeError(
        "state must be a MERA-like TensorNetwork with select() and gate(), "
        "or an ansatz object exposing .state."
    )


def _lightcone_num_sites(state, schedule=None):
    """Infer the physical-site count used for an energy-per-site result."""
    if schedule is not None:
        geometry = getattr(schedule, "geometry", None)
        num_sites = getattr(geometry, "num_sites", None)
        if num_sites is not None:
            return int(num_sites() if callable(num_sites) else num_sites)
    for name in ("num_sites", "sites", "L"):
        value = getattr(state, name, None)
        if value is None:
            continue
        return int(len(tuple(value)) if name == "sites" else value() if callable(value) else value)
    raise ValueError("Could not infer the number of MERA physical sites.")


def lightcone_energy(
    state,
    hamiltonian=None,
    *,
    chunks=None,
    schedule=None,
    array_backend=None,
    convert_terms=True,
    optimize="auto-hq",
    normalized=True,
    energy_per_site=True,
    real=True,
    simplify=False,
    gate_contract=True,
    contract_opts=None,
    group_terms=True,
    path_cache=None,
):
    """Evaluate a local energy by contracting only reverse-lightcone TNs.

    This is the generic, fixed-state counterpart to
    :func:`qmera_parametric_energy`. Each term selects its local cone, applies
    the operator with ``TensorNetwork.gate`` (or the native indexed equivalent
    for graded networks), and contracts the numerator and norm with a reusable
    topology-specific optimizer when ``path_cache`` is supplied. ``schedule``
    may be supplied for a qMERA state so the selector follows schedule-derived
    reverse-lightcone tags.

    Native Symmray operators and tensors are passed through unchanged when
    ``convert_terms=False``. Consequently the graded contraction and
    fermionic signs remain owned by Symmray; this function does not introduce
    Jordan--Wigner strings or dense sign corrections.
    """
    state, schedule = _lightcone_state_and_schedule(state, schedule)
    if chunks is None:
        if hamiltonian is None:
            raise ValueError("lightcone_energy requires hamiltonian or chunks.")
        terms = normalize_local_terms(hamiltonian)
        if convert_terms:
            terms = convert_local_terms(terms, array_backend)
        if schedule is None:
            chunks = build_lightcone_chunks(state, terms)
        else:
            chunks = build_qmera_lightcone_chunks(state, schedule, terms)
    else:
        chunks = tuple(chunks)
        if not chunks:
            raise ValueError("lightcone chunks cannot be empty.")

    value = None
    if group_terms:
        groups = {}
        for chunk in chunks:
            key = (tuple(chunk.tags), tuple(chunk.physical_outer_inds))
            groups.setdefault(key, []).append(chunk)
        grouped_chunks = tuple(groups.items())
    else:
        grouped_chunks = tuple(
            (((tuple(chunk.tags), tuple(chunk.physical_outer_inds))), [chunk])
            for chunk in chunks
        )

    for key, group in grouped_chunks:
        ket = select_lightcone(state, tags=group[0].tags, validate=False)
        group_optimize = _resolve_contraction_opt(
            optimize,
            path_cache,
            key=("lightcone", key),
        )
        denominator = None
        if normalized:
            denominator = _contract(
                _maybe_simplify(ket.H & ket, simplify),
                optimize=group_optimize,
                contract_opts=contract_opts,
            )
        for chunk in group:
            term = chunk.term
            ket_g = _apply_local_gate(
                ket,
                term.operator,
                term.where,
                contract=gate_contract,
                inplace=False,
            )
            term_value = _contract(
                _maybe_simplify(ket.H & ket_g, simplify),
                optimize=group_optimize,
                contract_opts=contract_opts,
            )
            if normalized:
                term_value = term_value / denominator
            if term.weight != 1.0:
                term_value = term_value * term.weight
            value = term_value if value is None else value + term_value

    if value is None:
        raise ValueError("hamiltonian contains no local terms.")
    if energy_per_site:
        value = value / _lightcone_num_sites(state, schedule)
    if real:
        value = _maybe_real(value)
    return value
