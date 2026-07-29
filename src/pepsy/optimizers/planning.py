"""Static cross-simulator advice for circuit gate streams.

The planner compares Pepsy's ordinary MPS/tree simulators with their
stabilizer-frame counterparts.  It never executes the circuit or constructs a
live ordinary tensor-network state.  Its scores are transparent operation-count
proxies at a caller-supplied bond dimension, not wall-clock predictions.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Optional

import numpy as np

from .mps import MpsGateStreamLayoutFinder
from .stabilizer_tn import MpsStabOptimizer, StreamAnalysisRecord
from .tree import TreeLayoutFinder, TreePlan

__all__ = [
    "SimulatorCandidate",
    "SimulatorPlan",
    "SimulatorPlanner",
    "recommend_simulator",
]


_ROTATION_NAMES = frozenset(
    {"rot", "rx", "ry", "rz", "rxx", "ryy", "rzz", "t", "tdg"}
)
_CANDIDATE_ORDER = {
    "MpsOptimizer": 0,
    "TreeOptimizer": 1,
    "MpsStabOptimizer": 2,
    "TreeStabOptimizer": 3,
}


class _PlannerRecord(MutableMapping):
    """Mapping-compatible facade shared by the typed planner records."""

    @classmethod
    def _field_names(cls):
        return tuple(field.name for field in fields(cls))

    def __getitem__(self, key):
        if key in self._field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key not in self._field_names():
            raise KeyError(key)
        setattr(self, key, value)

    def __delitem__(self, key):  # pragma: no cover - fixed-shape diagnostics
        raise TypeError(f"{type(self).__name__} fields cannot be deleted.")

    def __iter__(self):
        return iter(self._field_names())

    def __len__(self):
        return len(self._field_names())

    def as_dict(self) -> dict:
        """Return a shallow plain-``dict`` snapshot."""
        return {name: getattr(self, name) for name in self._field_names()}


@dataclass
class SimulatorCandidate(_PlannerRecord):
    """One ranked simulator choice and its auditable cost components."""

    optimizer: str
    geometry: str
    stabilizer_frame: bool
    applicable: bool
    score: Optional[float]
    relative_score: Optional[float]
    tensor_work: Optional[float]
    tableau_work: Optional[float]
    event_count: int
    one_site_events: int
    multi_site_events: int
    weighted_geometry: float
    max_geometry: int
    layout: object
    layout_report: dict
    settings: dict
    rationale: str
    warnings: tuple[str, ...] = ()


@dataclass
class SimulatorPlan(_PlannerRecord):
    """Ranked, non-executing advice for one circuit stream."""

    recommended: str
    candidates: tuple[SimulatorCandidate, ...]
    analysis: StreamAnalysisRecord
    n_qubits: int
    chi: int
    physical_events: tuple[dict, ...]
    frame_events: tuple[dict, ...]
    weight_mode: str
    cost_model: str
    warnings: tuple[str, ...]

    @property
    def best(self) -> SimulatorCandidate:
        """Return the first (lowest-score) applicable candidate."""
        return self.candidates[0]

    def candidate(self, optimizer: str) -> SimulatorCandidate:
        """Return advice for ``optimizer`` by public class name."""
        for candidate in self.candidates:
            if candidate.optimizer == optimizer:
                return candidate
        choices = ", ".join(item.optimizer for item in self.candidates)
        raise KeyError(f"Unknown simulator {optimizer!r}; choices are: {choices}.")


def _unique_warnings(warnings):
    return tuple(dict.fromkeys(str(warning) for warning in warnings if warning))


def _normalize_weight_mode(weight_mode):
    name = str(weight_mode).replace("-", "_").strip().lower()
    aliases = {"unit": "count", "uniform": "count", "none": "count"}
    name = aliases.get(name, name)
    if name not in {"count", "angle", "auto"}:
        raise ValueError(
            "weight_mode must be 'count', 'angle', or 'auto', "
            f"got {weight_mode!r}."
        )
    return name


def _event_weight(entry, *, weight_mode):
    """Return the physical-stream weight used by the static layout search."""
    if weight_mode == "count":
        return 1.0
    if (
        isinstance(entry, (tuple, list))
        and len(entry) >= 2
        and isinstance(entry[0], str)
        and str(entry[0]).replace("-", "_").strip().lower() in _ROTATION_NAMES
    ):
        try:
            angle = abs(float(entry[1]))
        except (TypeError, ValueError):
            return 1.0
        if np.isfinite(angle):
            return min(1.0, max(0.0, angle))
    return 1.0


def _layout_stream(records):
    """Encode weighted supports through the public layout-stream protocol."""
    return [
        (
            "submpo",
            {"angle": float(record["weight"])},
            tuple(record["support"]),
        )
        for record in records
    ]


def _layout_weight(_payload, _support, _event_type):
    return float(_payload.get("angle", 1.0))


def _mps_geometry(records, plan):
    position = {int(site): int(pos) for site, pos in plan["site_map"].items()}
    geometry = []
    for record in records:
        support = tuple(dict.fromkeys(int(site) for site in record["support"]))
        if len(support) <= 1:
            geometry.append(1)
            continue
        positions = [position[site] for site in support]
        geometry.append(max(positions) - min(positions) + 1)
    return tuple(geometry)


def _tree_steiner_size(plan: TreePlan, support):
    support = tuple(dict.fromkeys(int(site) for site in support))
    if len(support) <= 1:
        return 1
    nodes = [plan.node_of_qubit[site] for site in support]
    steiner = {nodes[0]}
    for node in nodes[1:]:
        steiner.update(plan.node_path(nodes[0], node))
    return len(steiner)


def _tree_geometry(records, plan):
    return tuple(
        _tree_steiner_size(plan, record["support"]) for record in records
    )


def _work_components(
    records,
    geometry,
    *,
    chi,
    operator_factor,
    tableau_work=0.0,
):
    one_site_events = 0
    multi_site_events = 0
    weighted_geometry = 0.0
    local_weight = 0.0
    routed_weight = 0.0
    max_geometry = 0

    for record, extent in zip(records, geometry):
        support = tuple(dict.fromkeys(record["support"]))
        weight = float(record["weight"])
        extent = int(extent)
        max_geometry = max(max_geometry, extent)
        if len(support) <= 1:
            one_site_events += 1
            local_weight += weight
        else:
            multi_site_events += 1
            weighted_geometry += weight * extent
            routed_weight += weight * extent

    chi_float = float(chi)
    tensor_work = float(operator_factor) * (
        local_weight * chi_float**2 + routed_weight * chi_float**3
    )
    score = tensor_work + float(tableau_work)
    return {
        "score": float(score),
        "tensor_work": float(tensor_work),
        "tableau_work": float(tableau_work),
        "event_count": len(records),
        "one_site_events": int(one_site_events),
        "multi_site_events": int(multi_site_events),
        "weighted_geometry": float(weighted_geometry),
        "max_geometry": int(max_geometry),
    }


class SimulatorPlanner:
    """Rank MPS, tree, MPS-STN, and tree-STN strategies without executing.

    Parameters
    ----------
    gates
        A Pepsy-native gate stream accepted by
        :meth:`MpsStabOptimizer.analyze_stream`.
    n_qubits
        Circuit width. It is inferred from known supports when omitted.
    chi
        Target tensor-network bond dimension used by the work proxy.
    weight_mode
        ``"count"`` weights every priced event equally. ``"angle"`` and
        ``"auto"`` down-weight named rotations with small absolute angles.
    mps_order
        Deterministic MPS layout candidate used for physical and dressed
        supports. The default avoids optional offline search backends.
    tree_layout_kwargs
        Optional keyword overrides for :class:`TreeLayoutFinder`. ``n``,
        ``gates``, ``supports``, ``chi``, and ``weight_mode`` are planner-owned.
    frame_mpo_factor
        Relative contraction constant for a dressed Pauli MPO. The default 16
        follows the bond-dimension-two HSMPO operation-count estimate.
    tableau_factor
        Relative cost assigned to each qubit touched by an incremental
        Clifford-frame update or dressed-Pauli query.
    """

    def __init__(
        self,
        gates,
        *,
        n_qubits=None,
        chi=64,
        weight_mode="count",
        mps_order="recursive_refined",
        tree_layout_kwargs=None,
        frame_mpo_factor=16.0,
        tableau_factor=1.0,
    ):
        if isinstance(chi, bool) or not isinstance(chi, Integral):
            raise TypeError("chi must be a positive integer.")
        chi = int(chi)
        if chi <= 0:
            raise ValueError("chi must be a positive integer.")
        if not np.isfinite(float(chi) ** 3):
            raise ValueError("chi is too large for the planner work proxy.")

        frame_mpo_factor = float(frame_mpo_factor)
        tableau_factor = float(tableau_factor)
        if not np.isfinite(frame_mpo_factor) or frame_mpo_factor <= 0.0:
            raise ValueError("frame_mpo_factor must be finite and positive.")
        if not np.isfinite(tableau_factor) or tableau_factor < 0.0:
            raise ValueError("tableau_factor must be finite and nonnegative.")

        if gates is None or (
            isinstance(gates, (tuple, list)) and len(gates) == 0
        ):
            self.entries = ()
        else:
            self.entries = tuple(MpsStabOptimizer._as_entries(gates))
        self.analysis = MpsStabOptimizer.analyze_stream(
            None if not self.entries else self.entries,
            n_qubits=n_qubits,
        )
        inferred = self.analysis.estimated_qubits
        if inferred is None or inferred <= 0:
            raise ValueError(
                "Could not infer a positive circuit width; pass n_qubits explicitly."
            )

        tree_layout_kwargs = (
            {} if tree_layout_kwargs is None else dict(tree_layout_kwargs)
        )
        forbidden = {
            "chi",
            "gates",
            "n",
            "supports",
            "weight_mode",
        }.intersection(tree_layout_kwargs)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(
                f"tree_layout_kwargs contains planner-owned option(s): {names}."
            )

        self.n_qubits = int(inferred)
        self.chi = chi
        self.weight_mode = _normalize_weight_mode(weight_mode)
        self.mps_order = str(mps_order)
        self.tree_layout_kwargs = tree_layout_kwargs
        self.frame_mpo_factor = frame_mpo_factor
        self.tableau_factor = tableau_factor

    def _physical_records(self):
        records = []
        warnings = []
        for index, entry in enumerate(self.entries):
            kind = MpsStabOptimizer._analysis_entry_kind(entry)
            if kind == "control":
                continue
            try:
                sites = MpsStabOptimizer._analysis_entry_sites(
                    entry,
                    self.n_qubits,
                )
            except (IndexError, TypeError, ValueError):
                sites = None
            estimated = sites is None
            if sites is None:
                sites = set(range(self.n_qubits))
                warnings.append(
                    f"Entry {index} has unknown support and was conservatively "
                    "priced across all qubits."
                )
            support = tuple(sorted(int(site) for site in sites))
            if not support:
                continue
            records.append(
                {
                    "index": int(index),
                    "kind": str(kind),
                    "support": support,
                    "weight": _event_weight(
                        entry,
                        weight_mode=self.weight_mode,
                    ),
                    "estimated_support": bool(estimated),
                }
            )
        return tuple(records), tuple(warnings)

    def _frame_records(self):
        if not self.entries:
            return ()
        simulator = MpsStabOptimizer(
            self.n_qubits,
            gates=self.entries,
            chi=self.chi,
            exact_cooling=False,
        )
        return simulator._frame_layout_records(
            self.entries,
            weight_mode=self.weight_mode,
        )

    def _mps_layout(self, records):
        finder = MpsGateStreamLayoutFinder(
            _layout_stream(records),
            L=self.n_qubits,
        )
        order = self.mps_order
        if not any(len(record["support"]) >= 2 for record in records):
            order = "input"
        return finder.run(
            order=order,
            weight_fn=_layout_weight,
            weight_mode="count",
        )

    def _tree_layout(self, records):
        kwargs = {
            "structure": "quality",
            "max_arity": (2, 3, 4),
            "objective": "path",
            **self.tree_layout_kwargs,
        }
        finder = TreeLayoutFinder(
            _layout_stream(records),
            n=self.n_qubits,
            chi=self.chi,
            weight_mode="angle",
            **kwargs,
        )
        plan = finder.run()
        return plan, finder.report(plan)

    def _candidate(
        self,
        *,
        optimizer,
        geometry_name,
        stabilizer_frame,
        records,
        geometry,
        layout,
        layout_report,
        operator_factor,
        tableau_work,
        warnings=(),
    ):
        parts = _work_components(
            records,
            geometry,
            chi=self.chi,
            operator_factor=operator_factor,
            tableau_work=tableau_work,
        )
        basis = "dressed frame" if stabilizer_frame else "physical"
        geometry_label = (
            "MPS window width" if geometry_name == "mps" else "tree Steiner size"
        )
        rationale = (
            f"Prices {parts['event_count']} {basis} event(s) using "
            f"{geometry_label}; {parts['multi_site_events']} require routed "
            f"multi-site work."
        )
        settings = {"chi": self.chi, "layout": layout}
        if stabilizer_frame:
            settings.update({"exact_cooling": True, "track_infidelity": True})
        return SimulatorCandidate(
            optimizer=optimizer,
            geometry=geometry_name,
            stabilizer_frame=stabilizer_frame,
            applicable=True,
            score=parts["score"],
            relative_score=None,
            tensor_work=parts["tensor_work"],
            tableau_work=parts["tableau_work"],
            event_count=parts["event_count"],
            one_site_events=parts["one_site_events"],
            multi_site_events=parts["multi_site_events"],
            weighted_geometry=parts["weighted_geometry"],
            max_geometry=parts["max_geometry"],
            layout=layout,
            layout_report=layout_report,
            settings=settings,
            rationale=rationale,
            warnings=_unique_warnings(warnings),
        )

    def _unavailable_stabilizer_candidate(self, optimizer, geometry, reason):
        return SimulatorCandidate(
            optimizer=optimizer,
            geometry=geometry,
            stabilizer_frame=True,
            applicable=False,
            score=None,
            relative_score=None,
            tensor_work=None,
            tableau_work=None,
            event_count=0,
            one_site_events=0,
            multi_site_events=0,
            weighted_geometry=0.0,
            max_geometry=0,
            layout=None,
            layout_report={},
            settings={"chi": self.chi},
            rationale="The stabilizer-frame dry run could not price this stream.",
            warnings=(str(reason),),
        )

    def plan(self) -> SimulatorPlan:
        """Return ranked advice while leaving the circuit and simulators untouched."""
        physical_records, physical_warnings = self._physical_records()
        physical_mps_layout = self._mps_layout(physical_records)
        physical_tree_layout, physical_tree_report = self._tree_layout(
            physical_records
        )

        candidates = [
            self._candidate(
                optimizer="MpsOptimizer",
                geometry_name="mps",
                stabilizer_frame=False,
                records=physical_records,
                geometry=_mps_geometry(physical_records, physical_mps_layout),
                layout=physical_mps_layout,
                layout_report={"stats": physical_mps_layout["stats"]},
                operator_factor=1.0,
                tableau_work=0.0,
                warnings=physical_warnings,
            ),
            self._candidate(
                optimizer="TreeOptimizer",
                geometry_name="tree",
                stabilizer_frame=False,
                records=physical_records,
                geometry=_tree_geometry(physical_records, physical_tree_layout),
                layout=physical_tree_layout,
                layout_report=physical_tree_report,
                operator_factor=1.0,
                tableau_work=0.0,
                warnings=physical_warnings,
            ),
        ]

        frame_records = ()
        frame_failure = None
        try:
            frame_records = self._frame_records()
        except (ImportError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            frame_failure = (
                "Stabilizer-frame prepass failed: "
                f"{type(exc).__name__}: {exc}"
            )

        if frame_failure is None:
            frame_mps_layout = self._mps_layout(frame_records)
            frame_tree_layout, frame_tree_report = self._tree_layout(frame_records)
            tableau_work = self.tableau_factor * self.n_qubits * (
                self.analysis.clifford_entries + len(frame_records)
            )
            candidates.extend(
                [
                    self._candidate(
                        optimizer="MpsStabOptimizer",
                        geometry_name="mps",
                        stabilizer_frame=True,
                        records=frame_records,
                        geometry=_mps_geometry(frame_records, frame_mps_layout),
                        layout=frame_mps_layout,
                        layout_report={"stats": frame_mps_layout["stats"]},
                        operator_factor=self.frame_mpo_factor,
                        tableau_work=tableau_work,
                    ),
                    self._candidate(
                        optimizer="TreeStabOptimizer",
                        geometry_name="tree",
                        stabilizer_frame=True,
                        records=frame_records,
                        geometry=_tree_geometry(frame_records, frame_tree_layout),
                        layout=frame_tree_layout,
                        layout_report=frame_tree_report,
                        operator_factor=self.frame_mpo_factor,
                        tableau_work=tableau_work,
                    ),
                ]
            )
        else:
            candidates.extend(
                [
                    self._unavailable_stabilizer_candidate(
                        "MpsStabOptimizer",
                        "mps",
                        frame_failure,
                    ),
                    self._unavailable_stabilizer_candidate(
                        "TreeStabOptimizer",
                        "tree",
                        frame_failure,
                    ),
                ]
            )

        candidates.sort(
            key=lambda candidate: (
                not candidate.applicable,
                float("inf") if candidate.score is None else candidate.score,
                _CANDIDATE_ORDER[candidate.optimizer],
            )
        )
        applicable_scores = [
            candidate.score
            for candidate in candidates
            if candidate.applicable and candidate.score is not None
        ]
        minimum = min(applicable_scores)
        for candidate in candidates:
            if not candidate.applicable or candidate.score is None:
                continue
            if minimum == 0.0:
                candidate.relative_score = (
                    1.0 if candidate.score == 0.0 else float("inf")
                )
            else:
                candidate.relative_score = float(candidate.score / minimum)

        warnings = [
            *self.analysis.warnings,
            *physical_warnings,
            (
                "Scores are static chi-scaled work proxies, not measured runtime "
                "or accuracy guarantees; benchmark close candidates."
            ),
            (
                "Stabilizer scores model direct dressed-operator replay. Exact "
                "cooling and immediate/deferred magic injection can change peak "
                "bond dimension and runtime."
            ),
        ]
        if frame_failure is not None:
            warnings.append(frame_failure)
        if self.analysis.is_clifford_only:
            warnings.append(
                "For a Clifford-only circuit, a tableau simulator such as Stim "
                "is normally preferable when full tensor-network state access "
                "is unnecessary."
            )

        return SimulatorPlan(
            recommended=candidates[0].optimizer,
            candidates=tuple(candidates),
            analysis=self.analysis,
            n_qubits=self.n_qubits,
            chi=self.chi,
            physical_events=physical_records,
            frame_events=tuple(frame_records),
            weight_mode=self.weight_mode,
            cost_model=(
                "local: weight*chi^2; routed: weight*geometry*chi^3; "
                f"dressed-MPO factor: {self.frame_mpo_factor:g}; "
                f"tableau factor: {self.tableau_factor:g}"
            ),
            warnings=_unique_warnings(warnings),
        )

    recommend = plan


def recommend_simulator(gates, **kwargs) -> SimulatorPlan:
    """Convenience wrapper for :class:`SimulatorPlanner`."""
    return SimulatorPlanner(gates, **kwargs).plan()
