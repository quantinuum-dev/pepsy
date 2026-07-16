"""Reverse-lightcone selection and local expectation kernels for MERA states."""

from __future__ import annotations

from dataclasses import dataclass

import autoray as ar

from .terms import LocalTerm

__all__ = [
    "LightconeChunk",
    "build_lightcone_chunks",
    "local_lightcone_expectation",
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

    @property
    def support_size(self):
        """Number of physical sites in the Hamiltonian support."""
        return len(self.term.where)

    @property
    def physical_width(self):
        """Number of physical outer legs left by the selected lightcone."""
        return len(self.physical_outer_inds)


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
