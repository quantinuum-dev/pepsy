"""Architecture search for schedule-first qMERA RG layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from itertools import product
from typing import Any

from .builders import QMeraBuilder
from .schedules import QMeraScaleSpec
from .terms import normalize_local_terms

__all__ = [
    "QMeraLayoutCandidate",
    "QMeraLayoutFinder",
    "QMeraLayoutReport",
    "QMeraLayoutScore",
]


def _freeze_mapping(value):
    return dict(value or {})


def _stable_id(config):
    payload = repr(tuple(sorted(config.items(), key=lambda item: item[0])))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"qmera-layout-{digest}"


def _as_options(value, default):
    if value is None:
        return (default,)
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, int):
        return (value,)
    if (
        isinstance(default, tuple)
        and isinstance(value, (tuple, list))
        and len(value) == len(default)
        and all(isinstance(item, (int, float)) for item in value)
    ):
        # In 2D, ``block_shapes=(2, 2)`` is the natural spelling for one
        # rectangular shape. Multiple shapes remain explicit as
        # ``((2, 2), (3, 3))``.
        return (tuple(value),)
    try:
        values = tuple(value)
    except TypeError:
        return (value,)
    return values or (default,)


def _shape_key(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    return int(value)


def _candidate_config_repr(config):
    return {
        "disentangler_block_shape": _shape_key(config["disentangler_block_shape"]),
        "isometry_block_shape": _shape_key(config["isometry_block_shape"]),
        "disentangler_depth": int(config["disentangler_depth"]),
        "isometry_depth": int(config["isometry_depth"]),
        "num_scales": int(config["num_scales"]),
    }


@dataclass(frozen=True)
class QMeraLayoutCandidate:
    """One immutable qMERA RG architecture candidate."""

    candidate_id: str
    scales: tuple[QMeraScaleSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "scales", tuple(self.scales))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class QMeraLayoutScore:
    """Comparable structural score for one qMERA layout candidate."""

    candidate_id: str
    total: float
    components: Mapping[str, float]
    valid: bool = True
    error: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(
            self,
            "components",
            {str(key): float(value) for key, value in dict(self.components).items()},
        )
        object.__setattr__(self, "total", float(self.total))


@dataclass(frozen=True)
class QMeraLayoutReport:
    """Search result containing scores and the non-dominated front."""

    candidates: tuple[QMeraLayoutCandidate, ...]
    scores: tuple[QMeraLayoutScore, ...]
    pareto_front: tuple[str, ...]

    @property
    def best(self):
        """Return the lowest-total valid candidate, or ``None``."""
        valid = [score for score in self.scores if score.valid]
        if not valid:
            return None
        best_id = min(valid, key=lambda score: score.total).candidate_id
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == best_id
        )

    @property
    def best_score(self):
        """Return the score associated with :attr:`best`, or ``None``."""
        best = self.best
        if best is None:
            return None
        return next(
            score for score in self.scores if score.candidate_id == best.candidate_id
        )


class QMeraLayoutFinder:
    """Generate and rank valid qMERA RG architectures.

    The finder searches immutable scale plans. It never mutates a builder or
    changes geometry/mode order while scoring a candidate. Scores are cheap
    structural/contraction proxies intended for pre-ranking; a caller can use
    the returned candidate with :class:`QMeraBuilder` for a measured pilot.
    """

    def __init__(
        self,
        geometry,
        *,
        gate_family="rxx",
        isometry_gate_family=None,
        gate_registry=None,
        isometry_block_shapes=None,
        disentangler_block_shapes=None,
        disentangler_depths=None,
        isometry_depths=None,
        max_layers=None,
        top_size=1,
        weights=None,
        builder_options=None,
    ):
        self.geometry = geometry
        self.gate_family = gate_family
        self.isometry_gate_family = isometry_gate_family or gate_family
        self.gate_registry = gate_registry
        default_shape = (2, 2) if getattr(geometry, "ndim", 1) == 2 else 2
        self.isometry_block_shapes = _as_options(
            isometry_block_shapes,
            default_shape,
        )
        self.disentangler_block_shapes = _as_options(
            disentangler_block_shapes,
            default_shape,
        )
        self.disentangler_depths = tuple(
            int(value) for value in _as_options(disentangler_depths, 1)
        )
        self.isometry_depths = tuple(
            int(value) for value in _as_options(isometry_depths, 1)
        )
        self.max_layers = None if max_layers is None else int(max_layers)
        self.top_size = int(top_size)
        if self.top_size < 1:
            raise ValueError("top_size must be >= 1.")
        defaults = {
            "structural": 1.0,
            "contraction": 0.1,
            "coverage": 1.0,
        }
        if weights:
            defaults.update({str(key): float(value) for key, value in weights.items()})
        self.weights = defaults
        self.builder_options = dict(builder_options or {})

    def _candidate_configs(self):
        for values in product(
            self.disentangler_block_shapes,
            self.isometry_block_shapes,
            self.disentangler_depths,
            self.isometry_depths,
        ):
            yield {
                "disentangler_block_shape": values[0],
                "isometry_block_shape": values[1],
                "disentangler_depth": values[2],
                "isometry_depth": values[3],
            }

    def _scales_for_config(self, config):
        """Find the shortest repeated scale plan that reaches ``top_size``."""
        limit = self.max_layers
        if limit is None:
            limit = max(1, sum(int(dim).bit_length() for dim in self.geometry.shape))
        for num_scales in range(1, limit + 1):
            scales = tuple(
                QMeraScaleSpec(
                    name=f"candidate-scale-{scale}",
                    disentangler={
                        "block_size": config["disentangler_block_shape"],
                        "circuit_depth": config["disentangler_depth"],
                        "gate_family": self.gate_family,
                    },
                    isometry={
                        "block_size": config["isometry_block_shape"],
                        "circuit_depth": config["isometry_depth"],
                        "gate_family": self.isometry_gate_family,
                    },
                )
                for scale in range(num_scales)
            )
            try:
                self._build_schedule(scales)
            except (ValueError, IndexError, NotImplementedError):
                continue
            return scales
        return None

    def _builder(self, scales):
        options = dict(self.builder_options)
        options.update(
            geometry=self.geometry,
            gate_family=self.gate_family,
            isometry_gate_family=self.isometry_gate_family,
            gate_registry=self.gate_registry,
            scales=scales,
            top_size=self.top_size,
        )
        return QMeraBuilder(**options)

    def _build_schedule(self, scales):
        return self._builder(scales).build_schedule()

    def generate_candidates(self):
        """Generate valid, deterministically ordered layout candidates."""
        candidates = []
        for config in self._candidate_configs():
            scales = self._scales_for_config(config)
            if scales is None:
                continue
            metadata = _candidate_config_repr(
                {**config, "num_scales": len(scales)}
            )
            candidates.append(
                QMeraLayoutCandidate(
                    candidate_id=_stable_id(metadata),
                    scales=scales,
                    metadata=metadata,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _lightcone_metrics(schedule, terms):
        widths = []
        total_weight = 0.0
        covered_weight = 0.0
        for term in terms:
            weight = abs(float(term.weight))
            total_weight += weight
            support = set(schedule.geometry.to_register_where(term.where))
            initial_size = len(support)
            selected = schedule.reverse_lightcone_placements(term.where)
            for placement in selected:
                support.update(placement.where)
            widths.append(len(support))
            disentangler_support = {
                site
                for placement in selected
                if placement.stage == "disentangler"
                for site in placement.where
            }
            fraction = (
                len(disentangler_support.intersection(set(
                    schedule.geometry.to_register_where(term.where)
                )))
                / max(1, initial_size)
            )
            covered_weight += weight * fraction
        if not widths:
            widths = [0]
        coverage = (
            covered_weight / total_weight
            if total_weight
            else 0.0
        )
        return {
            "max_lightcone_width": float(max(widths)),
            "mean_lightcone_width": float(sum(widths) / len(widths)),
            "interaction_coverage": float(coverage),
        }

    def score(self, candidate, hamiltonian=None):
        """Score one candidate, returning an invalid score instead of raising."""
        try:
            schedule = self._build_schedule(candidate.scales)
            terms = () if hamiltonian is None else normalize_local_terms(hamiltonian)
            metrics = self._lightcone_metrics(schedule, terms)
            gate_count = float(schedule.num_gates)
            circuit_depth = float(
                sum(
                    max(
                        [placement.round for placement in layer.placements],
                        default=-1,
                    )
                    + 1
                    for layer in schedule.layers
                )
            )
            structural = gate_count + circuit_depth
            contraction = metrics["max_lightcone_width"] ** 3
            total = (
                self.weights["structural"] * structural
                + self.weights["contraction"] * contraction
                + self.weights["coverage"]
                * (1.0 - metrics["interaction_coverage"])
            )
            components = {
                "gate_count": gate_count,
                "circuit_depth": circuit_depth,
                "structural_cost": structural,
                "contraction_cost_proxy": contraction,
                **metrics,
            }
            return QMeraLayoutScore(candidate.candidate_id, total, components)
        except (TypeError, ValueError, IndexError, KeyError, NotImplementedError) as exc:
            return QMeraLayoutScore(
                candidate.candidate_id,
                float("inf"),
                {},
                valid=False,
                error=str(exc),
            )

    def score_prototype_layout(self, prototype, hamiltonian=None):
        """Score a loaded prototype gate stream for structural comparison.

        Prototype streams are not converted into Pepsy schedules. This method
        reports a separate stream-level score so users can compare placement
        count, greedy parallel depth, and Hamiltonian-support coverage against
        native qMERA candidates without mixing their contraction semantics.
        """
        candidate_id = f"prototype:{prototype.name}"
        try:
            if int(prototype.num_sites) != int(self.geometry.num_modes):
                raise ValueError(
                    "prototype num_sites must equal geometry.num_modes for a "
                    "direct register-order comparison."
                )
            terms = () if hamiltonian is None else normalize_local_terms(hamiltonian)
            pair_supports = {frozenset(pair) for pair in prototype.pairs}
            total_weight = 0.0
            covered_weight = 0.0
            for term in terms:
                weight = abs(float(term.weight))
                total_weight += weight
                support = frozenset(self.geometry.to_register_where(term.where))
                if len(support) == 1:
                    covered = any(next(iter(support)) in pair for pair in pair_supports)
                else:
                    covered = support in pair_supports
                if covered:
                    covered_weight += weight
            coverage = covered_weight / total_weight if total_weight else 0.0
            unique_fraction = len(prototype.unique_sites) / max(1, prototype.num_sites)
            structural = float(prototype.gate_count + prototype.round_depth)
            contraction = float(max(1, prototype.max_support) ** 3)
            total = (
                self.weights["structural"] * structural
                + self.weights["contraction"] * contraction
                + self.weights["coverage"] * (1.0 - coverage)
            )
            return QMeraLayoutScore(
                candidate_id,
                total,
                {
                    "gate_count": float(prototype.gate_count),
                    "round_depth": float(prototype.round_depth),
                    "unique_site_fraction": float(unique_fraction),
                    "interaction_coverage": float(coverage),
                    "structural_cost": structural,
                    "contraction_cost_proxy": contraction,
                },
            )
        except (TypeError, ValueError, KeyError) as exc:
            return QMeraLayoutScore(
                candidate_id,
                float("inf"),
                {},
                valid=False,
                error=str(exc),
            )

    @staticmethod
    def _pareto_front(scores):
        valid = [score for score in scores if score.valid]
        front = []
        for score in valid:
            dominated = False
            for other in valid:
                if other is score:
                    continue
                other_components = other.components
                components = score.components
                no_worse = (
                    other.total <= score.total
                    and other_components.get("interaction_coverage", 0.0)
                    >= components.get("interaction_coverage", 0.0)
                )
                strictly_better = (
                    other.total < score.total
                    or other_components.get("interaction_coverage", 0.0)
                    > components.get("interaction_coverage", 0.0)
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(score.candidate_id)
        return tuple(front)

    def search(self, hamiltonian=None):
        """Generate candidates, score them, and return a Pareto report."""
        candidates = self.generate_candidates()
        scores = tuple(self.score(candidate, hamiltonian) for candidate in candidates)
        return QMeraLayoutReport(
            candidates=candidates,
            scores=scores,
            pareto_front=self._pareto_front(scores),
        )
