"""Stabilizer coefficient-frame layout tracing and installation.

Operations receive the live owner explicitly and retain dispatch through its hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mps_stab_optimizer import StabilizerMpsSimulator

import math
from collections.abc import Mapping
import autoray as ar
import quimb.tensor as qtn
from ..mps.layout import MpsGateStreamLayoutFinder
from .._stream_events import conditional_event_parts, submpo_event_parts
from .operators import pauli_decomposition
from .dense import _as_gate_matrix, _is_unitary, _tableau_from_exact_unitary
from .paulis import hermitian_pauli_terms, pauli_string
from ._stream_helpers import (
    _CLIFFORD_NAMES,
    _ROTATION_AXES,
    _ROTATION_AXES_2Q,
    _RESET_AXIS_ALIASES,
    _MR_ALIASES,
    _MR_AXIS_ALIASES,
    _normalize_event_name,
    _normalize_sites,
    _unique_ordered,
    _layout_angle_weight,
    _operator_schmidt_tail_weight,
    _dense_operator_schmidt_layout_weight,
    _submpo_operator_layout_weight,
    _parse_reset_args,
    _parse_measure_reset_args,
    _localizing_clifford,
    _zyz_angles,
)


def _refresh_layout_map(self) -> None:
    """Refresh the logical-coefficient-site -> MPS-position map."""
    self._logical_to_mps = {
        int(logical): int(pos)
        for pos, logical in enumerate(self.logical_order)
    }


def _layout_is_identity(self) -> bool:
    """Return whether the coefficient MPS is in logical site order."""
    return tuple(self.logical_order) == tuple(range(self.n))


def _mps_site(self, logical_site: int) -> int:
    """Map a logical coefficient qubit to its current MPS site position."""
    try:
        return self._logical_to_mps[int(logical_site)]
    except KeyError as exc:
        raise ValueError(
            f"coefficient site {logical_site!r} is not present in the "
            f"current STN layout {self.logical_order!r}."
        ) from exc


def _mps_sites(self, logical_sites) -> tuple[int, ...]:
    """Map logical coefficient support sites to MPS positions."""
    return tuple(self._mps_site(site) for site in logical_sites)


def _mps_terms(self, logical_terms) -> dict[int, str]:
    """Map a logical coefficient-frame Pauli support to MPS positions."""
    return {
        self._mps_site(site): axis
        for site, axis in logical_terms.items()
    }


def current_frame_layout(
    self,
    *,
    order="auto",
    refine_passes=8,
    refine_numba=True,
    spectral_dense_max=512,
    recursive_dense_max=1024,
    nevergrad_budget=64,
    nevergrad_seed=0,
    nevergrad_optimizer="OnePlusOne",
    kahypar_config_path=None,
    kahypar_seed=0,
    weight_mode="operator_schmidt",
):
    """Find a static MPS order from the queued STN frame supports.

    The pre-pass replays only tableau-changing events on a temporary copy.
    Each expensive coefficient-frame event contributes the support of its
    current ``C^dagger O C`` image.  By default, multi-site events are
    weighted by a baseline locality cost plus an operator-Schmidt
    entanglement proxy, so stronger two-branch rotations and wider
    coefficient-frame operators receive more layout priority.  The
    returned plan is a Pepsy-style layout plan whose ``site_order`` maps
    MPS positions to logical coefficient qubits.  It does not mutate the
    simulator.

    ``weight_mode`` accepts ``"operator_schmidt"`` (the default),
    ``"count"`` for the historical uniform weighting, and ``"angle"`` /
    ``"auto"`` for angle-based weighting.
    """
    records = self._frame_layout_records(
        self._queue,
        weight_mode=weight_mode,
    )
    stream = [
        ("submpo", {"weight": record["weight"]}, record["support"])
        for record in records
    ]
    finder = MpsGateStreamLayoutFinder(stream, L=self.n)

    def weight_fn(payload, _support, _event_type):
        if isinstance(payload, Mapping):
            return float(payload.get("weight", 1.0))
        return 1.0

    plan = finder.run(
        order=order,
        refine_passes=refine_passes,
        refine_numba=refine_numba,
        spectral_dense_max=spectral_dense_max,
        recursive_dense_max=recursive_dense_max,
        nevergrad_budget=nevergrad_budget,
        nevergrad_seed=nevergrad_seed,
        nevergrad_optimizer=nevergrad_optimizer,
        kahypar_config_path=kahypar_config_path,
        kahypar_seed=kahypar_seed,
        weight_fn=weight_fn,
        weight_mode="count",
    )
    plan = dict(plan)
    plan["kind"] = "stn_frame_layout"
    plan["source"] = "queued_frame_supports"
    plan["frame_events"] = tuple(records)
    plan["frame_weight_mode"] = weight_mode
    return plan


def _frame_layout_records(self, entries, *, weight_mode="operator_schmidt"):
    """Return weighted logical frame-support records for a stream."""
    mode = str(weight_mode).replace("-", "_").strip().lower()
    if mode in ("unit", "uniform", "none"):
        mode = "count"
    if mode not in ("count", "angle", "auto", "operator_schmidt"):
        raise ValueError(
            "STN frame layout weight_mode must be 'operator_schmidt', "
            "'count', 'angle', or 'auto'."
        )
    dry = self.copy()
    dry._queue = []
    records = []
    for entry in self._as_entries(entries):
        dry._frame_layout_trace_entry(entry, records, weight_mode=mode)
    return tuple(records)


def _frame_layout_weight(
    self,
    *,
    weight_mode,
    theta=None,
    coeff=None,
    support_size=None,
    operator_weight=None,
):
    """Return the scalar weight used for one frame-layout record."""
    if weight_mode == "operator_schmidt":
        if operator_weight is not None:
            amplitude = 1.0
            if coeff is not None:
                try:
                    amplitude = max(abs(complex(coeff)), 1.0e-12)
                except (TypeError, ValueError):
                    amplitude = 1.0
            return max(1.0e-12, float(operator_weight) * amplitude)
        if theta is not None:
            if support_size is not None and int(support_size) < 2:
                return 1.0
            # Keep a unit locality baseline and add the non-leading
            # operator-Schmidt weight of the I/P rotation branches.
            return 1.0 + _operator_schmidt_tail_weight(theta)
        if coeff is not None:
            try:
                return max(abs(complex(coeff)), 1.0e-12)
            except (TypeError, ValueError):
                return 1.0
        return 1.0
    if coeff is not None:
        try:
            return float(abs(complex(coeff)))
        except (TypeError, ValueError):
            return 1.0
    if weight_mode in ("angle", "auto") and theta is not None:
        return _layout_angle_weight(theta)
    return 1.0


def _frame_layout_add_pauli(
    self,
    pauli,
    where,
    records,
    *,
    kind,
    entry,
    weight_mode,
    theta=None,
    weight=None,
    absorb_basis=False,
):
    """Record one current frame image and optionally dry-run its basis update."""
    m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
    terms, _sign = hermitian_pauli_terms(m_pauli)
    support = tuple(sorted(terms))
    if support:
        if weight is None:
            weight = self._frame_layout_weight(
                weight_mode=weight_mode,
                theta=theta,
                support_size=len(support),
            )
        records.append({
            "kind": kind,
            "entry": entry,
            "support": support,
            "weight": float(weight),
            "operator_weight": float(weight),
            "absorbs_basis": bool(absorb_basis),
        })
    if absorb_basis and support:
        _ops, v_tableau, _k = _localizing_clifford(
            terms,
            self.n,
            site_position=self._mps_site,
        )
        self.state.absorb_basis_clifford(v_tableau)


def _frame_layout_trace_rotation(self, name, params, records, *, entry, weight_mode):
    """Trace a rotation entry for layout without changing ``|p>``."""
    theta, where, axes = self._rotation_spec(name, params)
    phys = pauli_string(axes, where, self.n)
    if self._is_clifford_angle(theta):
        self._apply_clifford_rotation(theta, where, axes)
        return
    m_pauli = self.state.frame_pauli(phys)
    terms, _sign = hermitian_pauli_terms(m_pauli)
    support = tuple(sorted(terms))
    if support:
        weight = self._frame_layout_weight(
            weight_mode=weight_mode,
            theta=theta,
            support_size=len(support),
        )
        records.append({
            "kind": "rotation",
            "entry": entry,
            "support": support,
            "weight": float(weight),
            "operator_weight": float(weight),
            "absorbs_basis": False,
        })


def _frame_layout_trace_matrix(self, gate, where, records, *, entry, weight_mode):
    """Trace an explicit physical matrix entry for layout."""
    where = _normalize_sites(where)
    gate = _as_gate_matrix(gate, len(where))
    dim = gate.shape[0]
    nq = int(round(math.log2(dim)))
    if 2 ** nq != dim or gate.shape != (dim, dim):
        raise ValueError(f"Gate matrix must be square 2^k x 2^k, got {gate.shape}.")
    if len(where) != nq:
        raise ValueError(f"Gate on {nq} qubit(s) but where={where!r}.")

    dense_operator_weight = (
        _dense_operator_schmidt_layout_weight(gate, nq)
        if weight_mode == "operator_schmidt"
        else None
    )

    tableau = _tableau_from_exact_unitary(gate)
    gate_is_unitary = _is_unitary(gate)
    if tableau is not None:
        self.state.do_tableau(tableau, where)
        return

    if nq == 1 and gate_is_unitary:
        alpha, theta, beta = _zyz_angles(gate)
        q = where[0]
        self._frame_layout_trace_rotation(
            "rz", (beta, q), records, entry=entry, weight_mode=weight_mode
        )
        self._frame_layout_trace_rotation(
            "ry", (theta, q), records, entry=entry, weight_mode=weight_mode
        )
        self._frame_layout_trace_rotation(
            "rz", (alpha, q), records, entry=entry, weight_mode=weight_mode
        )
        return

    limit = self.max_pauli_decomposition_qubits
    if limit is not None and nq > limit:
        raise ValueError(
            f"Pauli decomposition of a {nq}-qubit dense gate would enumerate "
            f"{4**nq} candidate terms, exceeding "
            f"max_pauli_decomposition_qubits={limit}."
        )
    for term_index, (labels, coeff) in enumerate(
        pauli_decomposition(gate, nq, tol=self.operator_tol), start=1
    ):
        if (
            self.max_pauli_terms is not None
            and term_index > self.max_pauli_terms
        ):
            raise ValueError(
                f"dense gate retained more than max_pauli_terms="
                f"{self.max_pauli_terms} during layout analysis."
            )
        phys = pauli_string(labels, where, self.n)
        frame_terms, _sign = hermitian_pauli_terms(self.state.frame_pauli(phys))
        support = tuple(sorted(frame_terms))
        if support:
            weight = self._frame_layout_weight(
                weight_mode=weight_mode,
                coeff=coeff,
                operator_weight=dense_operator_weight,
            )
            records.append({
                "kind": "matrix_branch",
                "entry": entry,
                "support": support,
                "weight": float(weight),
                "operator_weight": (
                    float(dense_operator_weight)
                    if dense_operator_weight is not None
                    else float(weight)
                ),
                "absorbs_basis": False,
            })


def _frame_layout_trace_entry(self, entry, records, *, weight_mode):
    """Trace one queued entry into weighted frame-support records."""
    conditional = conditional_event_parts(entry)
    if conditional is not None:
        raise ValueError(
            "static STN frame_layout='auto' cannot safely prepass a "
            "branch-dependent feed-forward action; provide an explicit "
            "layout or use the ordinary interaction layout."
        )
    parts = submpo_event_parts(entry, normalize_where=True)
    if parts is not None:
        mpo, where = parts
        support = tuple(sorted(_unique_ordered(where)))
        if support:
            weight = (
                _submpo_operator_layout_weight(mpo)
                if weight_mode == "operator_schmidt"
                else 1.0
            )
            records.append({
                "kind": "submpo",
                "entry": entry,
                "support": support,
                "weight": float(weight),
                "operator_weight": float(weight),
                "absorbs_basis": False,
            })
        return

    if not (isinstance(entry, (list, tuple)) and len(entry) >= 1):
        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    head = entry[0]
    if not isinstance(head, str):
        if len(entry) != 2:
            raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
        gate, where = entry
        self._frame_layout_trace_matrix(
            self._gate_to_numpy(gate),
            where,
            records,
            entry=entry,
            weight_mode=weight_mode,
        )
        return

    name = _normalize_event_name(head)
    if name == "disentangle":
        return
    if name in _CLIFFORD_NAMES:
        self.state.apply_clifford(name, *entry[1:])
        return
    if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in (
        "rot", "t", "tdg",
    ):
        self._frame_layout_trace_rotation(
            name,
            entry[1:],
            records,
            entry=entry,
            weight_mode=weight_mode,
        )
        return
    if name == "measure":
        pauli, where = entry[1], entry[2]
        absorb = bool(entry[4]) if len(entry) > 4 else False
        self._frame_layout_add_pauli(
            pauli,
            where,
            records,
            kind="measure",
            entry=entry,
            weight_mode=weight_mode,
            absorb_basis=absorb,
        )
        return
    if name == "reset" or name in _RESET_AXIS_ALIASES:
        axes, where = _parse_reset_args(
            entry[1:],
            default_axis=_RESET_AXIS_ALIASES.get(name),
        )
        for axis, q in zip(axes, where):
            self._frame_layout_add_pauli(
                axis,
                q,
                records,
                kind="reset",
                entry=entry,
                weight_mode=weight_mode,
                absorb_basis=True,
            )
        return
    if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
        axes, where, _outcomes, absorb = _parse_measure_reset_args(
            entry[1:],
            default_axis=_MR_AXIS_ALIASES.get(name),
        )
        for axis, q in zip(axes, where):
            self._frame_layout_add_pauli(
                axis,
                q,
                records,
                kind="measure_reset",
                entry=entry,
                weight_mode=weight_mode,
                absorb_basis=absorb,
            )
        return
    if name == "cap":
        raise ValueError(
            "static STN auto-layout is not supported with cap events, "
            "because cap changes the qubit/MPS length."
        )
    raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")


def _validate_layout_plan_for_stn(self, plan) -> None:
    """Validate that a layout plan is a full permutation of STN qubits."""
    site_order = tuple(int(site) for site in plan.get("site_order", plan.get("order", ())))
    if len(site_order) != self.n:
        raise ValueError(
            f"layout site_order length must match n={self.n}, got {len(site_order)}."
        )
    if sorted(site_order) != list(range(self.n)):
        raise ValueError(
            f"layout site_order must be a permutation of range({self.n})."
        )
    site_map = plan.get("site_map", plan.get("layout"))
    if site_map is None:
        raise ValueError("layout plan must contain a site_map/layout mapping.")
    expected = {site: pos for pos, site in enumerate(site_order)}
    if {int(k): int(v) for k, v in dict(site_map).items()} != expected:
        raise ValueError(
            "layout site_map must map each logical coefficient site to its "
            "position in site_order."
        )


def _explicit_layout_plan(self, site_order):
    """Build a minimal STN frame-layout plan from an explicit site order."""
    site_order = tuple(int(site) for site in site_order)
    site_map = {site: position for position, site in enumerate(site_order)}
    return {
        "kind": "stn_frame_layout",
        "selected_order": "explicit",
        "qubit_inds": site_order,
        "site_order": site_order,
        "order": site_order,
        "layout": site_map,
        "site_map": site_map,
        "inverse_site_map": {
            position: site for site, position in site_map.items()
        },
        "stats": {},
        "input_stats": {},
    }


def _resolve_layout_plan_argument(self, plan_or_order, layout_kwargs=None):
    """Resolve a static STN layout request without mutating the simulator."""
    if isinstance(plan_or_order, Mapping):
        plan = dict(plan_or_order)
    elif isinstance(plan_or_order, str):
        kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
        plan = self.current_frame_layout(order=plan_or_order, **kwargs)
    else:
        try:
            plan = self._explicit_layout_plan(plan_or_order)
        except TypeError as exc:
            raise TypeError(
                "plan_or_order must be a layout mapping, an order name, "
                "or a permutation of logical coefficient sites."
            ) from exc
    self._validate_layout_plan_for_stn(plan)
    return plan


def _product_site_vector(p, physical_site):
    """Extract a local vector from an isolated coefficient-MPS site."""
    tensor = p[p.site_tag(int(physical_site))]
    physical_ind = p.site_ind(int(physical_site))
    try:
        physical_axis = tensor.inds.index(physical_ind)
    except ValueError as exc:  # pragma: no cover - defensive quimb guard
        raise ValueError(
            "product-state relabeling could not locate a physical site index."
        ) from exc
    if any(
        int(size) != 1
        for axis, size in enumerate(tensor.shape)
        if axis != physical_axis
    ):
        raise ValueError(
            "extracting a local product vector requires every virtual dimension "
            "to be one."
        )
    axes = [axis for axis in range(tensor.ndim) if axis != physical_axis]
    axes.append(physical_axis)
    data = ar.do("transpose", tensor.data, tuple(axes))
    return data.reshape(-1)


def _relabel_product_mps(self, target_order, *, current_order):
    """Rebuild a bond-one coefficient MPS in a new logical site order."""
    p = self.state.p
    if getattr(p, "cyclic", False):
        raise ValueError(
            "STN static layout relabeling currently requires an open-boundary MPS."
        )
    vectors = {
        logical_site: self._product_site_vector(p, physical_site)
        for physical_site, logical_site in enumerate(current_order)
    }
    arrays = [vectors[logical_site] for logical_site in target_order]
    new_p = qtn.MPS_product_state(
        arrays,
        site_ind_id=p.site_ind_id,
        site_tag_id=p.site_tag_id,
    )
    if hasattr(p, "exponent") and hasattr(new_p, "exponent"):
        new_p.exponent = p.exponent
    self.state.p = new_p
    self.state.info = {"cur_orthog": None}


def _format_layout_value(value):
    """Format one layout diagnostic value compactly."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3g}"


def _format_layout_reduction(cls, before, after):
    """Format a before/after layout diagnostic compactly."""
    text = f"{cls._format_layout_value(before)} -> {cls._format_layout_value(after)}"
    try:
        before = float(before)
        after = float(after)
    except (TypeError, ValueError):
        return text
    if before > 0.0:
        text += f" ({100.0 * (before - after) / before:.1f}% lower)"
    return text


def _layout_report_text(cls, plan):
    """Return a concise human-readable STN layout report."""
    stats = plan.get("stats", {})
    input_stats = plan.get("input_stats", {})
    if not input_stats:
        return None
    selected = plan.get("selected_order", "<unknown>")
    site_order = plan.get("site_order", ())
    lines = [
        (
            "StabilizerMpsSimulator frame layout: "
            f"order={selected}, sites={len(site_order)}, "
            f"events={stats.get('num_events', input_stats.get('num_events', 0))}"
        ),
        (
            "  frame event span max/mean: "
            + cls._format_layout_value(input_stats.get("max_event_span", 0))
            + "/"
            + cls._format_layout_value(input_stats.get("weighted_mean_event_span", 0.0))
            + " -> "
            + cls._format_layout_value(stats.get("max_event_span", 0))
            + "/"
            + cls._format_layout_value(stats.get("weighted_mean_event_span", 0.0))
        ),
        (
            "  score: "
            + cls._format_layout_reduction(
                input_stats.get("loss", input_stats.get("score", 0.0)),
                stats.get("loss", stats.get("score", 0.0)),
            )
            + " | cut L2: "
            + cls._format_layout_reduction(
                input_stats.get("weighted_cut_congestion_l2", 0.0),
                stats.get("weighted_cut_congestion_l2", 0.0),
            )
        ),
    ]
    return "\n".join(lines)


def apply_layout(
    self,
    plan_or_order="auto",
    *,
    layout_kwargs=None,
    layout_report: bool = True,
) -> "StabilizerMpsSimulator":
    """Install a static STN frame layout while ``|p>`` is still product.

    The tableau/physical qubit labels stay unchanged.  Only the coefficient
    MPS tensor order changes, and every future coefficient-frame support is
    mapped through the installed logical-order map.  This keeps the operation
    safe and exact for any state whose coefficient MPS has ``max_bond()==1``
    (including Clifford-entangled stabilizer states), and rejects entangled
    coefficient states before mutation.
    """
    for entry in self._queue:
        if isinstance(entry, (list, tuple)) and entry:
            head = entry[0]
            if isinstance(head, str) and _normalize_event_name(head) == "cap":
                raise ValueError(
                    "static STN layout cannot be installed for streams with "
                    "cap events, because cap changes the qubit/MPS length."
                )
    plan = self._resolve_layout_plan_argument(plan_or_order, layout_kwargs)
    target_order = tuple(int(site) for site in plan["site_order"])
    current_order = tuple(self.logical_order)
    if target_order != current_order:
        if int(self.state.max_bond()) != 1:
            raise ValueError(
                "static STN layout requires a product coefficient MPS "
                "(state.max_bond() == 1); got max_bond={} . Apply the "
            "layout before non-Clifford evolution entangles |p>.".format(
                    self.state.max_bond()
                )
            )
        self._relabel_product_mps(target_order, current_order=current_order)
    self.logical_order = list(target_order)
    self._refresh_layout_map()
    self._localizer_cache.clear()
    self.layout_plan = plan
    self.last_layout_plan = plan
    if layout_report:
        report = self._layout_report_text(plan)
        if report:
            print(report)
    return self


def _apply_layout_from_entries(
    self,
    entries,
    layout,
    *,
    layout_kwargs=None,
    layout_report: bool = True,
) -> None:
    """Install a static layout found from ``entries`` without queuing them."""
    if layout is None or layout is False:
        return
    old_queue = self._queue
    self._queue = list(entries)
    try:
        self.apply_layout(
            layout,
            layout_kwargs=layout_kwargs,
            layout_report=layout_report,
        )
    finally:
        self._queue = old_queue
