"""Adapters for comparing qMERA schedules with the research prototype streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

__all__ = [
    "QMeraPrototypeLayout",
    "load_qmera_prototype_layout",
]


_LEVEL_RE = re.compile(r"(?:^|[_-])(?:l|level)[_-]?(\d+)(?:$|[_-])", re.I)


def _normalize_pair(pair, index):
    try:
        left, right = pair
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"prototype gate entry {index} must contain exactly two sites."
        ) from exc
    if isinstance(left, bool) or isinstance(right, bool):
        raise TypeError(f"prototype gate entry {index} contains a boolean site.")
    try:
        left, right = int(left), int(right)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"prototype gate entry {index} must contain integer site labels."
        ) from exc
    if left < 0 or right < 0:
        raise ValueError(f"prototype gate entry {index} contains a negative site.")
    if left == right:
        raise ValueError(f"prototype gate entry {index} is a self-gate.")
    return left, right


@dataclass(frozen=True)
class QMeraPrototypeLayout:
    """A normalized two-site gate stream loaded from the qMERA prototype.

    The prototype ``U_q3_l*`` files are serialized placement streams rather
    than Pepsy RG schedules. This object keeps them available for structural
    comparison without pretending that their flat stream is a qMERA
    ``QMeraScaleSpec``.
    """

    name: str
    path: str
    pairs: tuple[tuple[int, int], ...]
    num_sites: int
    level: int | None = None
    source: str = "mera-prototype"

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "path", str(self.path))
        pairs = tuple((int(left), int(right)) for left, right in self.pairs)
        object.__setattr__(self, "pairs", pairs)
        num_sites = int(self.num_sites)
        if num_sites < 1:
            raise ValueError("num_sites must be >= 1.")
        if pairs and max(max(pair) for pair in pairs) >= num_sites:
            raise ValueError("num_sites must cover every prototype gate site.")
        object.__setattr__(self, "num_sites", num_sites)
        object.__setattr__(self, "level", None if self.level is None else int(self.level))

    @property
    def gate_count(self):
        """Return the number of two-site placements in the stream."""
        return len(self.pairs)

    @property
    def unique_sites(self):
        """Return the sites touched by at least one prototype placement."""
        return tuple(sorted({site for pair in self.pairs for site in pair}))

    @property
    def max_support(self):
        """Return the largest placement arity, currently always two."""
        return max((len(pair) for pair in self.pairs), default=0)

    def greedy_rounds(self):
        """Partition the stream into deterministic non-overlapping rounds."""
        rounds = []
        for pair in self.pairs:
            support = set(pair)
            for current in rounds:
                if all(not support.intersection(other) for other in current):
                    current.append(pair)
                    break
            else:
                rounds.append([pair])
        return tuple(tuple(current) for current in rounds)

    @property
    def round_depth(self):
        """Return the greedy non-overlapping stream depth."""
        return len(self.greedy_rounds())


def load_qmera_prototype_layout(
    path,
    *,
    loader: Callable[[str], Any] | None = None,
    num_sites: int | None = None,
    level: int | None = None,
):
    """Load a serialized ``U_q3_l*`` prototype placement stream.

    ``quimb.load_from_disk`` is imported only when no loader is supplied, so
    callers can test or adapt the format without making quimb serialization a
    new hard dependency. The returned stream is a diagnostic adapter; use
    :class:`QMeraLayoutFinder` for Pepsy-native RG schedules.
    """
    path = Path(path)
    if loader is None:
        try:
            import quimb as qu  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Loading qMERA prototype streams requires `quimb` or a custom loader."
            ) from exc
        loader = qu.load_from_disk

    raw = loader(str(path))
    try:
        entries = tuple(raw)
    except TypeError as exc:
        raise TypeError("prototype layout data must be an iterable of site pairs.") from exc
    pairs = tuple(_normalize_pair(pair, index) for index, pair in enumerate(entries))
    inferred_sites = max((max(pair) for pair in pairs), default=-1) + 1
    if num_sites is None:
        if inferred_sites < 1:
            raise ValueError("num_sites is required when the prototype stream is empty.")
        num_sites = inferred_sites
    if int(num_sites) < max(1, inferred_sites):
        raise ValueError(
            f"num_sites={num_sites} does not cover prototype site {inferred_sites - 1}."
        )
    if level is None:
        match = _LEVEL_RE.search(path.name)
        level = None if match is None else int(match.group(1))
    return QMeraPrototypeLayout(
        name=path.name,
        path=str(path),
        pairs=pairs,
        num_sites=int(num_sites),
        level=level,
    )
