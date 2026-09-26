"""Layout execution for the MPS optimizer.

Functions receive the live optimizer explicitly; there is no second state
container. They own layout installation, site mapping, reordering, schedule
installation, and logical readout. Canonicalization, norm accounting, native
swaps, stream validation, and replay remain optimizer hooks. Calls through
``self`` deliberately preserve subclass overrides at those boundaries.

Geometry search and schedule construction belong to ``layout.py``. This
module does not import ``optimizer.py`` and is not a public optimizer API.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import time
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .._stream_events import _control_event_contains_cap
from .layout import _normalize_layout_support, _unique_ordered


def current_gate_stream_schedule(
    self,
    *,
    sites=None,
    L=None,
    lattice_shape=None,
    lattice_site=None,
    qubit_roles=None,
    site_coords=None,
    layout_order="quality",
    schedule_order="mountain",
    layout_kwargs=None,
):
    """Implement the public ``MpsOptimizer.current_gate_stream_schedule`` operation."""
    unsupported = set(self.event_types) - {"gate", "submpo", "cap"}
    if unsupported:
        raise ValueError(
            "current_gate_stream_schedule accepts ordinary gate, sub-MPO, "
            "and direct cap events; measurement/reset/conditional events "
            f"must remain in the stateful run path ({sorted(unsupported)!r})."
        )
    finder_kwargs, run_kwargs = self._split_layout_finder_kwargs(
        layout_kwargs
    )
    finder_options = dict(finder_kwargs)
    for name, value in (
        ("lattice_shape", lattice_shape),
        ("lattice_site", lattice_site),
        ("qubit_roles", qubit_roles),
        ("site_coords", site_coords),
    ):
        if value is not None:
            finder_options.setdefault(name, value)
    finder = self.layout_finder(sites=sites, L=L, **finder_options)
    plan = finder.run(order=layout_order, **run_kwargs)
    return finder.compile_schedule(plan, strategy=schedule_order)


def select_layout_for_compression(
    self,
    *,
    sites=None,
    L=None,
    layout_kwargs=None,
    pilot_candidates=4,
    pilot_steps=None,
    cutoff=1e-12,
    cutoff_mode="rsum2",
    run_kwargs=None,
):
    """Implement the public ``MpsOptimizer.select_layout_for_compression`` operation."""
    if self.mode == "exact":
        raise ValueError(
            "compression layout pilots require an MPS compression mode, "
            "not mode='exact'."
        )
    if self._persistent_layout_plan is not None:
        raise ValueError(
            "compression layout pilots require an optimizer without a "
            "persistent layout; create the pilot before apply_layout()."
        )
    if self.mode == "perm":
        raise ValueError(
            "compression layout pilots require a fixed-layout compression "
            "mode; mode='perm' changes the order during replay."
        )
    if any(
        event_type == "conditional"
        and _control_event_contains_cap(event_type, payload)
        for payload, event_type in zip(self.G, self.event_types)
    ):
        raise ValueError(
            "compression layout pilots do not support conditional cap "
            "events; the active branch is needed to update the shrinking "
            "layout."
        )
    try:
        pilot_candidates = int(pilot_candidates)
    except (TypeError, ValueError) as exc:
        raise ValueError("pilot_candidates must be a positive integer.") from exc
    if pilot_candidates < 1:
        raise ValueError("pilot_candidates must be a positive integer.")
    if pilot_steps is not None:
        try:
            pilot_steps = int(pilot_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("pilot_steps must be a positive integer or None.") from exc
        if pilot_steps < 1:
            raise ValueError("pilot_steps must be a positive integer or None.")

    base_run_kwargs = dict(run_kwargs or {})
    conflicting = sorted(
        {"layout", "use_layout_finder"}.intersection(base_run_kwargs)
    )
    if conflicting:
        raise ValueError(
            "run_kwargs for compression layout pilots must not contain "
            f"{', '.join(conflicting)}; the pilot supplies its own layout."
        )
    pilot_mode = base_run_kwargs.get("mode", self.mode)
    if pilot_mode is None:
        pilot_mode = self.mode
    if self._normalize_mode(pilot_mode) == "perm":
        raise ValueError(
            "compression layout pilots require a fixed-layout compression "
            "mode; mode='perm' changes the order during replay."
        )
    if self._normalize_mode(pilot_mode) == "exact":
        raise ValueError(
            "compression layout pilots require an MPS compression mode, "
            "not mode='exact'."
        )

    kwargs = dict(layout_kwargs or {})
    finder_kwargs, kwargs = self._split_layout_finder_kwargs(kwargs)
    finder = self.layout_finder(
        sites=sites,
        L=L,
        **finder_kwargs,
    )
    kwargs["objective"] = "compression"
    static_plan = finder.run(**kwargs)
    candidates = dict(static_plan.get("candidate_plans", {}))
    if not candidates:
        candidates = {static_plan["selected_order"]: static_plan}
    ranked_names = sorted(
        candidates,
        key=lambda name: candidates[name]["stats"].get(
            "compression_score", candidates[name]["stats"].get("score", 0.0)
        ),
    )[:pilot_candidates]

    base_run_kwargs.setdefault("progbar", False)
    base_run_kwargs.setdefault("layout_report", False)
    base_run_kwargs.setdefault("cutoff", cutoff)
    base_run_kwargs.setdefault("cutoff_mode", cutoff_mode)
    pilot_reports = {}
    successful = []
    for name in ranked_names:
        trial = self.copy()
        if pilot_steps is not None:
            trial.G = trial.G[:pilot_steps]
            trial.where = trial.where[:pilot_steps]
            trial.event_types = trial.event_types[:pilot_steps]
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                trial.run(layout=candidates[name], **base_run_kwargs)
            elapsed = time.perf_counter() - started
            final_bond = int(trial.p.max_bond())
            report = {
                "status": "ok",
                "elapsed_seconds": float(elapsed),
                "final_bond": final_bond,
                "pilot_steps": len(trial.G),
            }
            successful.append((final_bond, elapsed, name))
        except Exception as exc:  # pragma: no cover - backend-specific
            report = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - started),
                "pilot_steps": len(trial.G),
            }
        pilot_reports[name] = report

    if not successful:
        raise RuntimeError(
            "All MPS compression layout pilot candidates failed. "
            f"Diagnostics: {pilot_reports!r}"
        )
    selected_name = min(successful)[-1]
    selected = dict(candidates[selected_name])
    selected["selected_order"] = selected_name
    selected["pilot"] = {
        "objective": "compression",
        "pilot_candidates": tuple(ranked_names),
        "selected_order": selected_name,
        "reports": pilot_reports,
    }
    selected["candidate_plans"] = candidates
    return selected


def _restore_permutation(self):
    """Restore logical site order after a lazy-permutation replay."""
    if self._persistent_layout_plan is not None:
        raise ValueError(
            "persistent layouts are intentionally not restored; use "
            "to_dense(logical_order=True) or remap_sample(...) for readout."
        )
    target = tuple(range(int(getattr(self.p, "L", 0))))
    current = tuple(self.qubits)
    if current != target:
        self._reorder_mps_to_logical_order(target, current_order=current)
    self._set_site_order(target)


def _set_site_order(self, order):
    """Set the physical-position to logical-site mapping consistently.

    ``qubits`` is the name used by Quimb's permutation MPS helpers,
    whereas ``logical_order`` is Pepsy's public readout/layout name. They
    intentionally expose the same mapping while a lazy permutation is
    active, so all state transitions go through this helper.
    """
    order = [int(site) for site in order]
    self.qubits = list(order)
    self.logical_order = list(order)


def _logical_to_physical_where(self, where):
    """Map logical site locations to current physical MPS positions."""
    if self._persistent_layout_plan is None and self.mode != "perm":
        return tuple(int(site) for site in where)
    order = self.logical_order if self._persistent_layout_plan is not None else self.qubits
    try:
        return tuple(order.index(int(site)) for site in where)
    except ValueError as exc:
        raise ValueError(
            f"logical site in {where!r} is not present in the current "
            f"permutation {order!r}."
        ) from exc


def _record_permutation_move(self, where):
    """Record the no-swap-back movement made by a two-site gate."""
    i, j = sorted(map(int, where))
    # The routed right endpoint is left immediately to the right of the
    # left endpoint, exactly as in Quimb's no-swap-back permutation MPS.
    order = list(self.qubits)
    moved = order.pop(j)
    order.insert(i + 1, moved)
    self._set_site_order(order)


def _update_permutation_after_cap(self, logical_site, physical_site):
    """Remove a capped logical site and renumber the shortened chain."""
    logical_site = int(logical_site)
    physical_site = int(physical_site)
    if self.qubits[physical_site] != logical_site:
        raise ValueError(
            "cap permutation bookkeeping lost the logical site mapping."
        )
    remaining = [
        logical
        for physical, logical in enumerate(self.qubits)
        if physical != physical_site
    ]
    self._set_site_order(
        logical if logical < logical_site else logical - 1
        for logical in remaining
    )


def remap_sample(self, config):
    """Implement the public ``MpsOptimizer.remap_sample`` operation."""
    if isinstance(config, Mapping):
        return {
            self.logical_site(position): value
            for position, value in config.items()
        }
    config = np.asarray(ar.to_numpy(config))
    if config.ndim == 0 or config.shape[-1] != len(self.logical_order):
        raise ValueError(
            "sample configuration must have MPS length as its final "
            f"dimension, got shape {config.shape}."
        )
    logical = np.empty_like(config)
    logical[..., np.asarray(self.logical_order, dtype=int)] = config
    return logical


def to_dense(self, logical_order=True, **kwargs):
    """Implement the public ``MpsOptimizer.to_dense`` operation."""
    if not hasattr(self.p, "L"):
        # Exact mode stores a contracted TensorNetwork rather than an MPS,
        # so its output indices must be supplied explicitly to Quimb.
        inds = (
            [self._format_ind(site) for site in range(len(self.logical_order))]
            if logical_order
            else list(self.p.outer_inds())
        )
        return self.p.to_dense(inds, **kwargs)
    if not logical_order or self.logical_order == list(range(self.p.L)):
        return self.p.to_dense(**kwargs)
    logical_inds = [self.p.site_ind(self.position(site)) for site in range(self.p.L)]
    return self.p.to_dense(logical_inds, **kwargs)


def set_gate_schedule(self, schedule, *, reorder_product_state=True):
    """Implement the public ``MpsOptimizer.set_gate_schedule`` operation."""
    if self.mode == "perm":
        raise ValueError(
            "scheduled streams use fixed physical positions after caps; "
            "mode='perm' cannot apply its own lazy permutation on top."
        )
    if self._persistent_layout_plan is not None:
        raise ValueError(
            "install a scheduled stream on a fresh optimizer; persistent "
            "layouts and dynamic cap positions are separate operations."
        )
    stream = getattr(schedule, "stream", None)
    if stream is None:
        raise TypeError("schedule must provide a compiled 'stream'.")
    site_order = tuple(
        getattr(schedule, "site_order", tuple(range(int(self.p.L))))
    )
    if len(site_order) != int(self.p.L) or set(site_order) != set(range(int(self.p.L))):
        raise ValueError(
            "schedule.site_order must be a permutation of the current MPS sites."
        )
    identity = tuple(range(int(self.p.L)))
    if site_order != identity:
        if not reorder_product_state:
            raise ValueError(
                "schedule has a non-identity site_order; set "
                "reorder_product_state=True or provide the state in that order."
            )
        if self._effective_max_bond(self.p) != 1:
            raise ValueError(
                "installing a scheduled non-identity layout requires a "
                "product input MPS; reorder it explicitly before replay."
            )
        self._relabel_product_mps(site_order, current_order=identity)
        # Shot replay must start from the reordered product state, not the
        # pre-schedule order captured by the constructor.
        self._initial_p = self.p.copy()

    self.set_gates(stream)
    self.scheduled_site_order = site_order
    layout_plan = getattr(schedule, "layout_plan", None)
    self.scheduled_layout_plan = deepcopy(layout_plan)
    return self


def _resolve_run_layout(self, layout, layout_order, layout_kwargs):
    """Return ``(finder, plan)`` for a run-time layout request."""
    self.last_layout_plan = None
    if not self._layout_request_enabled(layout):
        return None, None
    if self.mode == "exact":
        raise ValueError("layout-aware replay requires an MPS mode, not exact.")

    if isinstance(layout, Mapping):
        plan = dict(layout)
        finder = self.layout_finder()
    else:
        order = layout_order
        if isinstance(layout, str):
            order = layout
        finder_kwargs, kwargs = self._split_layout_finder_kwargs(layout_kwargs)
        finder = self.layout_finder(**finder_kwargs)
        plan = finder.run(order=order, **kwargs)

    self._validate_layout_plan_for_mps(plan)
    self.last_layout_plan = plan
    return finder, plan


def _validate_layout_plan_for_mps(self, plan):
    """Validate that a layout plan can be used by this MPS."""
    L = int(getattr(self.p, "L", 0))
    original_order = tuple(range(L))
    site_order = tuple(plan.get("site_order", plan.get("qubit_inds", ())))
    if set(site_order) != set(original_order):
        raise ValueError(
            "layout-aware MpsOptimizer replay currently requires a "
            "permutation of integer MPS sites range(L)."
        )
    if len(site_order) != L:
        raise ValueError("layout site_order length must match p.L.")
    site_map = plan.get("site_map", plan.get("layout"))
    if not isinstance(site_map, Mapping):
        raise ValueError("layout plan must contain a site_map/layout mapping.")
    if set(site_map) != set(original_order):
        raise ValueError("layout site_map keys must match range(p.L).")
    if set(site_map.values()) != set(original_order):
        raise ValueError("layout site_map values must be a permutation of range(p.L).")
    expected_map = {site: position for position, site in enumerate(site_order)}
    if dict(site_map) != expected_map:
        raise ValueError(
            "layout site_map must map each logical site to its position in "
            "site_order."
        )


def _explicit_layout_plan(self, site_order):
    """Build the standard layout-plan mapping from an explicit site order."""
    site_order = tuple(int(site) for site in site_order)
    site_map = {site: position for position, site in enumerate(site_order)}
    return {
        "kind": "mps_gate_stream_layout",
        "selected_order": "explicit",
        "qubit_inds": site_order,
        "site_order": site_order,
        "order": site_order,
        "layout": site_map,
        "site_map": site_map,
        "inverse_site_map": {
            position: site for site, position in site_map.items()
        },
    }


def _resolve_layout_plan_argument(self, plan_or_order, layout_kwargs=None):
    """Resolve a persistent-layout argument without touching the MPS."""
    if isinstance(plan_or_order, Mapping):
        plan = dict(plan_or_order)
    elif isinstance(plan_or_order, str):
        finder_kwargs, kwargs = self._split_layout_finder_kwargs(layout_kwargs)
        plan = self.layout_finder(**finder_kwargs).run(
            order=plan_or_order,
            **kwargs,
        )
    else:
        try:
            plan = self._explicit_layout_plan(plan_or_order)
        except TypeError as exc:
            raise TypeError(
                "plan_or_order must be a layout mapping, an order name, "
                "or a permutation of logical sites."
            ) from exc
    self._validate_layout_plan_for_mps(plan)
    return plan


def _product_site_vector(p, physical_site):
    """Extract one local vector from a bond-one MPS tensor."""
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
            "product-state relabeling requires every virtual dimension to "
            "be one."
        )
    axes = [axis for axis in range(tensor.ndim) if axis != physical_axis]
    axes.append(physical_axis)
    data = ar.do("transpose", tensor.data, tuple(axes))
    return data.reshape(-1)


def _relabel_product_mps(self, target_order, *, current_order):
    """Rebuild a bond-one MPS in a new site order without SVD swaps."""
    p = self.p
    if getattr(p, "cyclic", False):
        raise ValueError(
            "persistent layout relabeling currently requires an open-boundary MPS."
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
    self.p = self._install_represented_norm(new_p)
    self._invalidate_replay_metadata()
    self.info_c = {}
    self._init_canonicalization()


def apply_layout(
    self,
    plan_or_order="quality",
    *,
    cutoff=None,
    cutoff_mode="rsum2",
    allow_lossy_reorder=False,
    layout_kwargs=None,
    layout_report=True,
):
    """Implement the public ``MpsOptimizer.apply_layout`` operation."""
    if self.mode == "exact":
        raise ValueError("persistent layouts require an MPS execution mode, not exact.")
    if self.mode == "perm":
        raise ValueError(
            "persistent layouts cannot be combined with mode='perm'; choose one."
        )
    if any(
        event_type == "conditional"
        and _control_event_contains_cap(event_type, payload)
        for payload, event_type in zip(self.G, self.event_types)
    ):
        raise ValueError(
            "persistent layouts do not support conditional cap events; "
            "the active branch is needed to update the shrinking layout."
        )

    plan = self._resolve_layout_plan_argument(plan_or_order, layout_kwargs)
    target_order = tuple(plan["site_order"])
    current_order = tuple(self.logical_order)

    if self._persistent_layout_plan is not None:
        if target_order != current_order:
            raise ValueError(
                "a persistent layout is already installed; use the existing "
                "logical_order or create a new optimizer for another layout."
            )
        return self

    identity = tuple(range(int(getattr(self.p, "L", 0))))
    if current_order != identity:
        raise ValueError(
            "cannot install a persistent layout while the MPS already has "
            "a lazy permutation; restore it or create a new optimizer."
        )

    if target_order != current_order:
        if self._effective_max_bond(self.p) == 1:
            self._relabel_product_mps(target_order, current_order=current_order)
        elif not allow_lossy_reorder:
            raise ValueError(
                "persistent layout requires an initially product MPS "
                "(p.max_bond() == 1); got max_bond={} . Set "
                "allow_lossy_reorder=True to pay a one-time reorder cost, "
                "or apply the layout before entangling the state.".format(
                    self.p.max_bond()
                )
            )
        else:
            reorder_cutoff = 1e-12 if cutoff is None else float(cutoff)
            if reorder_cutoff < 0.0:
                raise ValueError("cutoff must be non-negative.")
            self._reorder_mps_to_logical_order(
                target_order,
                current_order=current_order,
                cutoff=reorder_cutoff,
                cutoff_mode=cutoff_mode,
            )

    self._set_site_order(target_order)
    self._persistent_layout_plan = plan
    self.layout_plan = plan
    self.last_layout_plan = plan
    # Shot replay starts from the configured template. Once a persistent
    # layout is installed, that template must include the one-time reorder
    # so every fresh child can reuse the frozen physical arrangement.
    self._initial_p = self.p.copy()
    if target_order != current_order:
        # Exact product relabeling preserves the norm, while an explicitly
        # lossy entangled reorder can change it. Re-establish the raw
        # unitary baseline in either case instead of trusting metadata from
        # the pre-layout tensor representation.
        self._invalidate_unitary_norm_baseline()
    if layout_report:
        report = self._layout_report_text(plan)
        if report:
            print(report)
    return self


def _reorder_mps_to_logical_order(
    self,
    target_order,
    *,
    current_order=None,
    cutoff=0.0,
    cutoff_mode="abs",
):
    """Physically permute MPS site contents into ``target_order``."""
    target = list(target_order)
    current = (
        list(range(int(getattr(self.p, "L", 0))))
        if current_order is None
        else list(current_order)
    )
    if set(target) != set(current) or len(target) != len(current):
        raise ValueError("target_order must be a permutation of current_order.")

    self._invalidate_replay_metadata()
    for target_pos, logical_site in enumerate(target):
        current_pos = current.index(logical_site)
        if current_pos == target_pos:
            continue
        if self._replay_has_symmray_data(self.p) and self._native_needs_safe_qr(self.p):
            self._native_swap_site_to(
                self.p,
                current_pos,
                target_pos,
                info=self.info_c,
                compress_opts={
                    "method": "svd",
                    "cutoff": cutoff,
                    "cutoff_mode": cutoff_mode,
                },
            )
        else:
            self.p.swap_site_to_(
                current_pos,
                target_pos,
                info=self.info_c,
                method="svd",
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        moved = current.pop(current_pos)
        current.insert(target_pos, moved)

    self._current_orthog(self.p)
    return tuple(current)


def _normalize_visible_mps_order(self):
    """Make cached visible MPS order match canonical site order."""
    L = int(getattr(self.p, "L", 0))
    site_inds = [self.p.site_ind(site) for site in range(L)]
    outer_inds = getattr(self.p, "_outer_inds", None)
    if outer_inds is not None:
        outer_set = set(outer_inds)
        ordered_outer = [ind for ind in site_inds if ind in outer_set]
        ordered_outer.extend(ind for ind in outer_inds if ind not in site_inds)
        self.p._outer_inds = type(outer_inds)(ordered_outer)

    tid_to_site = self.p._get_tid_to_site_map()
    if tid_to_site:
        ordered_tensors = {}
        for site in range(L):
            for tid, mapped_site in tid_to_site.items():
                if mapped_site == site:
                    ordered_tensors[tid] = self.p.tensor_map[tid]
        for tid, tensor in self.p.tensor_map.items():
            ordered_tensors.setdefault(tid, tensor)
        self.p.tensor_map.clear()
        self.p.tensor_map.update(ordered_tensors)


def _copy_submpo_for_layout(submpo, site_map, support):
    """Return a copied sub-MPO with site labels remapped by ``site_map``."""
    support = _unique_ordered(support)
    if not support:
        return submpo

    mpo = submpo.copy()
    token = f"_pepsy_layout_{id(mpo)}"
    reindex_to_temp = {}
    reindex_to_final = {}
    retag_to_temp = {}
    retag_to_final = {}

    for count, old_site in enumerate(support):
        new_site = site_map[old_site]
        if old_site == new_site:
            continue

        for kind in ("upper_ind", "lower_ind"):
            ind_fn = getattr(mpo, kind, None)
            if ind_fn is None:
                continue
            old_ind = ind_fn(old_site)
            new_ind = ind_fn(new_site)
            tmp_ind = f"{token}_{count}_{kind}"
            reindex_to_temp[old_ind] = tmp_ind
            reindex_to_final[tmp_ind] = new_ind

        site_tag = getattr(mpo, "site_tag", None)
        if site_tag is not None:
            old_tag = site_tag(old_site)
            new_tag = site_tag(new_site)
            tmp_tag = f"{token}_{count}_tag"
            retag_to_temp[old_tag] = tmp_tag
            retag_to_final[tmp_tag] = new_tag

    if reindex_to_temp:
        mpo.reindex_(reindex_to_temp)
        mpo.reindex_(reindex_to_final)
    if retag_to_temp:
        mpo.retag_(retag_to_temp)
        mpo.retag_(retag_to_final)
    return mpo


def _layout_run_sequences(self, G_seq, where_seq, event_seq, plan):
    """Return run-local payloads and mapped locations for ``plan``.

    Direct cap events are handled as compile-time lifetime boundaries:
    after mapping a cap at the current physical position, the removed
    logical label is dropped and higher labels are compacted. This keeps
    later gate/control locations aligned with the live shortened MPS. A
    conditional cap cannot be compiled safely without knowing the branch,
    so callers reject that form before entering this path.
    """
    site_map = plan.get("site_map", plan.get("layout"))
    if not isinstance(site_map, Mapping):
        raise ValueError("layout plan must contain a site_map/layout mapping.")
    if self._persistent_layout_plan is not None:
        runtime_order = list(self.logical_order)
    else:
        runtime_order = list(plan.get("site_order", ()))
    if not runtime_order:
        raise ValueError("layout plan must contain a non-empty site_order.")
    mapped_G = []
    mapped_where = []
    for payload, where, event_type in zip(G_seq, where_seq, event_seq):
        support = _normalize_layout_support(where)
        try:
            mapped = tuple(runtime_order.index(site) for site in support)
        except ValueError as exc:
            raise ValueError(
                "layout stream references a logical site that is no longer "
                f"live: {support!r}; current order is {runtime_order!r}."
            ) from exc
        if event_type == "submpo":
            payload = self._copy_submpo_for_layout(
                payload,
                dict(zip(support, mapped)),
                support,
            )
        mapped_G.append(payload)
        mapped_where.append(mapped)
        if event_type == "cap":
            if len(support) != 1:
                raise ValueError(
                    "layout cap events must target exactly one logical site."
                )
            logical_site = int(support[0])
            physical_site = int(mapped[0])
            if runtime_order[physical_site] != logical_site:
                raise ValueError(
                    "layout cap lifetime mapping lost its logical site "
                    f"at physical position {physical_site}."
                )
            runtime_order.pop(physical_site)
            runtime_order = [
                site if site < logical_site else site - 1
                for site in runtime_order
            ]
        elif event_type == "conditional" and _control_event_contains_cap(
            event_type, payload
        ):
            raise ValueError(
                "layout streams do not support conditional cap events; "
                "the active branch is needed to update the shrinking layout."
            )
    return mapped_G, mapped_where
