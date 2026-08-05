"""Local-term normalization for qMERA energy objectives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["LocalTerm", "convert_local_terms", "normalize_local_terms"]


@dataclass(frozen=True)
class LocalTerm:
    """A local Hamiltonian term acting on one or more site labels."""

    where: tuple[Any, ...]
    operator: Any
    weight: Any = 1.0
    tags: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] | None = None

    def with_operator(self, operator):
        """Return a copy with a replaced operator payload."""
        return replace(self, operator=operator)

    def with_tags(self, tags):
        """Return a copy with cached site tags."""
        return replace(self, tags=tuple(tags))


def _is_local_term(value):
    return isinstance(value, LocalTerm)


def _normalize_where(where):
    if isinstance(where, str):
        items = (where,)
    else:
        try:
            items = tuple(where)
        except TypeError:
            items = (where,)

    if not items:
        raise ValueError("local term support cannot be empty.")

    try:
        unique_items = set(items)
    except TypeError as exc:
        raise TypeError("local term site labels must be hashable.") from exc

    if len(unique_items) != len(items):
        raise ValueError(f"local term support contains duplicate sites: {items!r}.")

    return items


def _normalize_term_entry(entry):
    if _is_local_term(entry):
        where = _normalize_where(entry.where)
        if entry.operator is None:
            raise ValueError(f"local term on {where!r} has no operator.")
        return replace(entry, where=where)

    if not isinstance(entry, tuple):
        try:
            entry = tuple(entry)
        except TypeError as exc:
            raise TypeError(
                "local term entries must be LocalTerm objects or "
                "(where, operator[, weight]) tuples."
            ) from exc

    if len(entry) not in {2, 3}:
        raise ValueError(
            "local term entries must have form (where, operator) or "
            "(where, operator, weight)."
        )

    where = _normalize_where(entry[0])
    operator = entry[1]
    if operator is None:
        raise ValueError(f"local term on {where!r} has no operator.")
    weight = 1.0 if len(entry) == 2 else entry[2]
    return LocalTerm(where=where, operator=operator, weight=weight)


def _term_entries_from_hamiltonian(hamiltonian):
    if hamiltonian is None:
        raise ValueError("hamiltonian is required.")

    if isinstance(hamiltonian, Mapping):
        if "local_terms" in hamiltonian:
            return _term_entries_from_hamiltonian(hamiltonian["local_terms"])
        if "local_ham" in hamiltonian:
            return _term_entries_from_hamiltonian(hamiltonian["local_ham"])
        return tuple(hamiltonian.items())

    terms = getattr(hamiltonian, "terms", None)
    if terms is not None:
        return _term_entries_from_hamiltonian(terms)

    if isinstance(hamiltonian, Iterable):
        return tuple(hamiltonian)

    raise TypeError(
        "hamiltonian must be a local-term mapping, a LocalHam-like object "
        "with .terms, or an iterable of local term entries."
    )


def normalize_local_terms(hamiltonian):
    """Normalize local Hamiltonian terms.

    Supported first inputs are mappings such as ``{(i, j): H2}``, iterable
    entries ``[((i, j), H2)]``, and :class:`LocalTerm` instances.
    """
    entries = _term_entries_from_hamiltonian(hamiltonian)
    terms = tuple(_normalize_term_entry(entry) for entry in entries)
    if not terms:
        raise ValueError("hamiltonian contains no local terms.")
    return terms


def _copy_array_like(value):
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    return value


def _convert_operator(operator, array_backend):
    if array_backend is None:
        return operator
    operator = _copy_array_like(operator)
    apply_to_arrays = getattr(operator, "apply_to_arrays", None)
    if callable(apply_to_arrays):
        apply_to_arrays(array_backend)
        return operator
    return array_backend(operator)


def convert_local_terms(terms, array_backend):
    """Return ``terms`` with operators converted by ``array_backend``."""
    if array_backend is None:
        return tuple(terms)
    converted = []
    for term in terms:
        try:
            operator = _convert_operator(term.operator, array_backend)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            raise TypeError(
                f"Could not convert local operator on {term.where!r} "
                "to the requested Pepsy backend."
            ) from exc
        converted.append(term.with_operator(operator))
    return tuple(converted)
