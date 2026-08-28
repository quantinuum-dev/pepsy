"""Tree-embedded PEPS operator replay and compression."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np

from .operators import TreePepo, TreeSubPepo, plan_signature
from .plan import TreePepsPlan
from .state import TreePeps, _normalize_compression_mode

__all__ = ["TreePepsOptimizer"]

_UNSET = object()


class TreePepsOptimizer:
    """Apply gates and tree-structured PEPO fragments to a ``TreePeps``.

    Two update modes are supported:

    ``"direct"``
        Build a small dense operator on the supplied support, factorize it on
        the support's unique tree span, and compress only that span.  For a
        two-site gate the span is exactly the tree geodesic between its sites.

    ``"sub_treepepo"``
        Apply an already factorized :class:`TreeSubPepo`.  The complete
        operator span is fused first and its internal edges are compressed in
        one leaf-to-center sweep.

    In both modes, intermediate routing is lossless.  The optimizer owns an
    independent state copy by default and mutates that live state when
    :meth:`apply` or :meth:`apply_gate` is called.
    """

    _MODE_ALIASES = {
        "subtreepepo": "sub_treepepo",
        "sub_tree_pepo": "sub_treepepo",
        "subtree_pepo": "sub_treepepo",
        "subtree": "sub_treepepo",
    }

    def __init__(
        self,
        state: TreePeps,
        *,
        plan: TreePepsPlan | None = None,
        mode="direct",
        compression_mode="direct",
        chi=64,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        reduced=True,
        inplace=False,
        info_c=None,
        max_operator_sites=12,
        max_subtree_nodes=None,
        gates=None,
        run=True,
        record_history=True,
    ):
        if not isinstance(state, TreePeps):
            raise TypeError("state must be a TreePeps")
        if plan is not None:
            if not isinstance(plan, TreePepsPlan):
                raise TypeError("plan must be a TreePepsPlan")
            if plan_signature(plan) != plan_signature(state.plan):
                raise ValueError("plan and state must use the same tree plan")
        compression_mode = _normalize_compression_mode(compression_mode)
        raw_mode = str(mode).strip().lower().replace("-", "_")
        if raw_mode == "dm":
            if compression_mode not in {"direct", "dm"}:
                raise ValueError(
                    "mode='dm' cannot be combined with a different "
                    "compression_mode."
                )
            compression_mode = "dm"
            raw_mode = "direct"
        self.mode = self._normalize_mode(raw_mode)
        self.compression_mode = compression_mode
        if max_bond is not None:
            chi = max_bond
        self.chi = self._normalize_max_bond(chi)
        self.cutoff = self._normalize_cutoff(cutoff)
        self.cutoff_mode = cutoff_mode
        self.reduced = bool(reduced)
        self.info_c = info_c
        self.max_operator_sites = self._normalize_limit(max_operator_sites, "max_operator_sites")
        self.max_subtree_nodes = self._normalize_limit(max_subtree_nodes, "max_subtree_nodes")
        self.record_history = bool(record_history)
        self.history = []
        self.state = state if inplace else state.copy()
        self.state.validate()
        self._sync_info()

        if gates is not None and run:
            self.run(gates)

    @classmethod
    def _normalize_mode(cls, mode):
        mode = str(mode).strip().lower()
        mode = cls._MODE_ALIASES.get(mode, mode)
        if mode not in {"direct", "sub_treepepo", "auto"}:
            raise ValueError("mode must be 'direct', 'sub_treepepo', or 'auto'")
        return mode

    @staticmethod
    def _resolve_modes(mode, compression_mode):
        """Resolve operator routing and compression modes independently."""

        raw_mode = str(mode).strip().lower().replace("-", "_")
        compression_mode = _normalize_compression_mode(compression_mode)
        if raw_mode == "dm":
            if compression_mode not in {"direct", "dm"}:
                raise ValueError(
                    "mode='dm' cannot be combined with a different "
                    "compression_mode."
                )
            return "direct", "dm"
        return (
            TreePepsOptimizer._normalize_mode(raw_mode),
            compression_mode,
        )

    @staticmethod
    def _normalize_max_bond(max_bond):
        if max_bond is None:
            return None
        if isinstance(max_bond, bool) or not isinstance(max_bond, Integral):
            raise TypeError("chi/max_bond must be a positive integer or None")
        max_bond = int(max_bond)
        if max_bond < 1:
            raise ValueError("chi/max_bond must be a positive integer or None")
        return max_bond

    @staticmethod
    def _normalize_limit(value, name):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer or None")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer or None")
        return value

    @staticmethod
    def _normalize_cutoff(cutoff):
        if cutoff is None:
            return 1e-10
        if isinstance(cutoff, str):
            if cutoff.strip().lower() == "auto":
                return 1e-10
            raise ValueError("cutoff must be a non-negative number or 'auto'")
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be a non-negative number or 'auto'")
        return cutoff

    @property
    def tn(self):
        """Compatibility alias for the live ``TreePeps`` state."""

        return self.state

    @tn.setter
    def tn(self, state):
        if not isinstance(state, TreePeps):
            raise TypeError("tn must be a TreePeps")
        if plan_signature(state.plan) != plan_signature(self.plan):
            raise ValueError("tn and optimizer must use the same tree plan")
        self.state = state
        self._sync_info()

    @property
    def plan(self):
        return self.state.plan

    @property
    def center(self):
        return self.state.orthogonality_center

    @property
    def canonical_region(self):
        return self.state.canonical_region

    def _sync_info(self):
        if self.info_c is not None:
            self.state._sync_info_c(self.info_c)

    def validate(self, *, check_canonical=False):
        self.state.validate(check_canonical=check_canonical)
        return self

    def is_canonical_form(self, center=None):
        return self.state.is_canonical_form(center)

    def is_subtree_canonical_form(self, sites=None, *, span=False):
        return self.state.is_subtree_canonical_form(sites, span=span)

    def _normalize_support(self, support):
        if isinstance(support, Integral):
            support = (support,)
        try:
            support = tuple(self.plan.resolve_site(site) for site in support)
        except TypeError as exc:
            raise TypeError("support must be a site or iterable of sites") from exc
        if not support:
            raise ValueError("support cannot be empty")
        if len(set(support)) != len(support):
            raise ValueError("support must contain distinct sites")
        if self.max_operator_sites is not None and len(support) > self.max_operator_sites:
            raise ValueError(
                f"operator support has {len(support)} sites, exceeding "
                f"max_operator_sites={self.max_operator_sites}"
            )
        return support

    def _normalize_span(self, span):
        span = frozenset(self.plan.resolve_site(site) for site in span)
        if not span or not self.plan.is_connected(span):
            raise ValueError("operator span must be a non-empty connected subtree")
        if self.max_subtree_nodes is not None and len(span) > self.max_subtree_nodes:
            raise ValueError(
                f"operator span has {len(span)} nodes, exceeding "
                f"max_subtree_nodes={self.max_subtree_nodes}"
            )
        return span

    def _physical_dims(self):
        return {
            q: int(self.state.node_tensor(q).ind_size(self.state.site_ind_1d(q)))
            for q in self.state.sites
        }

    def _state_dtype(self):
        return np.asarray(self.state.node_tensor(self.plan.root).data).dtype

    def _gate_dtype(self, gate):
        candidate = gate.to_dense() if hasattr(gate, "to_dense") else gate
        candidate = getattr(candidate, "data", candidate)
        return np.result_type(self._state_dtype(), np.asarray(candidate).dtype)

    def _region_center(self, region, preferred=None):
        region = frozenset(region)
        if preferred in region:
            return preferred
        return min(
            region,
            key=lambda q: (
                max(len(self.plan.path(q, other)) for other in region),
                sum(len(self.plan.path(q, other)) for other in region),
                q,
            ),
        )

    def _region_edges(self, region):
        region = frozenset(region)
        return tuple(
            edge for edge in self.plan.tree_edges if edge[0] in region and edge[1] in region
        )

    @staticmethod
    def _bond_sizes(state, edges):
        return {
            tuple(edge): int(state.node_tensor(edge[0]).ind_size(state.bond(*edge)))
            for edge in edges
        }

    def _prepare_span(self, span):
        span = frozenset(span)
        if self.state.canonical_region != span or not self.state.is_subtree_canonical_form(span):
            self.state.canonize_subtree(span, inplace=True)

    def _apply_operator(
        self,
        operator,
        support,
        span,
        *,
        mode,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
    ):
        if not isinstance(operator, TreePepo):
            raise TypeError("operator must be a TreePepo")
        if plan_signature(operator.plan) != plan_signature(self.plan):
            raise ValueError("operator and optimizer must use the same tree plan")
        support = self._normalize_support(support)
        span = self._normalize_span(span)
        if operator.operator_span is not None and not operator.operator_span.issubset(span):
            span = self._normalize_span(operator.operator_span)
        operator.validate()

        max_bond = self.chi if max_bond is _UNSET else self._normalize_max_bond(max_bond)
        cutoff = self.cutoff if cutoff is _UNSET else self._normalize_cutoff(cutoff)
        if cutoff_mode is None:
            cutoff_mode = self.cutoff_mode
        if compression_mode is None:
            compression_mode = self.compression_mode
        compression_mode = _normalize_compression_mode(compression_mode)
        center_before = self.center
        edges = self._region_edges(span)
        before_bonds = self._bond_sizes(self.state, edges)

        # The state is canonical around the active region before the complete
        # PEPO is fused.  This is a fast metadata-aware move when possible.
        self._prepare_span(span)
        result = operator.apply_to(
            self.state,
            compress=False,
            _active_sites=span,
        )
        result._canonical_region = frozenset(span)
        result._set_isometry_metadata_from_region(span)
        result.validate(check_canonical=True)

        if compress:
            center = self._region_center(span, preferred=center)
            result.compress_subtree(
                span,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                reduced=self.reduced,
                inplace=True,
                info_c=self.info_c,
            )
        self.state = result
        self._sync_info()

        after_bonds = self._bond_sizes(self.state, edges)
        report = {
            "mode": mode,
            "compression_mode": compression_mode,
            "support": tuple(support),
            "span": tuple(sorted(span)),
            "path": (self.plan.path(support[0], support[1]) if len(support) == 2 else None),
            "center_before": center_before,
            "center_after": self.center,
            "before_bonds": before_bonds,
            "after_bonds": after_bonds,
            "max_bond": max_bond,
            "cutoff": cutoff,
            "compressed": bool(compress),
            "truncated": any(after_bonds[edge] < before_bonds[edge] for edge in before_bonds),
        }
        if self.record_history:
            self.history.append(report)
        return self

    def apply_gate(
        self,
        gate,
        where,
        *,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
        _mode=None,
    ):
        """Apply a dense one- or multi-site gate in direct tree mode."""

        route_mode = self.mode if _mode is None else self._normalize_mode(_mode)
        if route_mode == "sub_treepepo":
            raise ValueError("mode='sub_treepepo' requires a TreeSubPepo operator")
        support = self._normalize_support(where)
        operator = TreePepo.from_operator(
            self.plan,
            gate,
            support,
            dims=self._physical_dims(),
            dtype=self._gate_dtype(gate),
            max_operator_sites=self.max_operator_sites,
        )
        return self._apply_operator(
            operator,
            support,
            operator.operator_span,
            mode="direct",
            compress=compress,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_mode=(
                self.compression_mode
                if compression_mode is None else compression_mode
            ),
        )

    def apply_sub_treepepo(
        self,
        operator,
        *,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
    ):
        """Apply a complete ``TreeSubPepo`` without intermediate truncation."""

        if not isinstance(operator, TreeSubPepo):
            raise TypeError("apply_sub_treepepo requires a TreeSubPepo")
        if operator.plan_signature != plan_signature(self.plan):
            raise ValueError("operator and optimizer must use the same tree plan")
        return self._apply_operator(
            operator.operator,
            operator.support,
            operator.span,
            mode="sub_treepepo",
            compress=compress,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_mode=(
                self.compression_mode
                if compression_mode is None else compression_mode
            ),
        )

    apply_subtree_pepo = apply_sub_treepepo
    apply_sub_tree_pepo = apply_sub_treepepo
    apply_subtree_operator = apply_sub_treepepo

    def apply(
        self,
        operator,
        where=None,
        *,
        mode=None,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
    ):
        """Dispatch a raw dense gate or explicit tree PEPO fragment."""

        selected_mode, selected_compression = self._resolve_modes(
            self.mode if mode is None else mode,
            self.compression_mode if compression_mode is None else compression_mode,
        )
        if isinstance(operator, TreeSubPepo):
            if where is not None:
                raise TypeError("where cannot be supplied with a TreeSubPepo")
            return self.apply_sub_treepepo(
                operator,
                compress=compress,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=selected_compression,
            )
        if isinstance(operator, TreePepo):
            if where is None:
                where = operator.operator_support
                if not where:
                    where = operator.sites
            return self._apply_operator(
                operator,
                where,
                operator.operator_span or operator.sites,
                mode="sub_treepepo",
                compress=compress,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=selected_compression,
            )
        if selected_mode == "sub_treepepo":
            raise TypeError("mode='sub_treepepo' requires a TreeSubPepo operator")
        if where is None:
            raise TypeError("where is required for a dense direct gate")
        return self.apply_gate(
            operator,
            where,
            compress=compress,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_mode=selected_compression,
            _mode=selected_mode,
        )

    def run(self, gates: Iterable):
        """Replay ``(gate, support)`` entries or explicit ``TreeSubPepo``s."""

        for entry in gates:
            if isinstance(entry, TreeSubPepo):
                self.apply_sub_treepepo(entry)
                continue
            if isinstance(entry, TreePepo):
                self.apply(entry)
                continue
            try:
                gate, where = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "gate entries must be (gate, support), TreePepo, or TreeSubPepo"
                ) from exc
            self.apply_gate(gate, where)
        return self

    @property
    def last_report(self):
        return None if not self.history else self.history[-1]

    def copy(self):
        """Copy the optimizer and its live state without replaying gates."""

        copied = type(self)(
            self.state,
            mode=self.mode,
            compression_mode=self.compression_mode,
            chi=self.chi,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            reduced=self.reduced,
            inplace=False,
            info_c=None,
            max_operator_sites=self.max_operator_sites,
            max_subtree_nodes=self.max_subtree_nodes,
            run=False,
            record_history=self.record_history,
        )
        copied.history = [dict(report) for report in self.history]
        return copied

    def __repr__(self):
        return (
            f"TreePepsOptimizer(mode={self.mode!r}, "
            f"compression_mode={self.compression_mode!r}, "
            f"shape={self.plan.shape!r}, "
            f"center={self.center!r}, chi={self.chi!r})"
        )
