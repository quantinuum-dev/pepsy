"""Symmray-native fermionic helpers for qMERA."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import quimb.tensor as qtn

from ...operators import fsim as dense_fsim
from .gates import GateSpec, default_gate_registry
from .terms import LocalTerm

__all__ = [
    "QMeraSymmrayFermionBackend",
    "qmera_symmray_fermi_hubbard_terms",
    "symmray_fermion_gate_registry",
]


def _require_symmray():
    try:
        import symmray as sr  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Symmray qMERA fermion helpers require `symmray`.") from exc
    return sr


def _require_fermionic_local_operators():
    _require_symmray()
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    return flo


def _apply_to_array_blocks(value, to_backend):
    if to_backend is None:
        return value
    out = value.copy() if callable(getattr(value, "copy", None)) else value
    apply = getattr(out, "apply_to_arrays", None)
    if callable(apply):
        apply(to_backend)
        return out
    return to_backend(out)


def _as_spin_pair(value, *, name):
    try:
        left, right = value
    except TypeError:
        return value, value
    except ValueError as exc:
        raise ValueError(f"{name} must be a scalar or a length-2 sequence.") from exc
    return left, right


def _node_parameter(value, site):
    if callable(value):
        return value(site)
    if isinstance(value, Mapping):
        return value[site]
    return value


def _edge_parameter(value, left, right):
    if callable(value):
        return value(left, right)
    if isinstance(value, Mapping):
        try:
            return value[(left, right)]
        except KeyError:
            return value[(right, left)]
    return value


def _edge_angle_parameter(value, left, right):
    if callable(value):
        return value(left, right)
    if isinstance(value, Mapping):
        if (left, right) in value:
            return value[(left, right)]
        return -value[(right, left)]
    return value


@dataclass(frozen=True)
class QMeraSymmrayFermionBackend:
    """Build Symmray fermionic qMERA states and local mode operators.

    The backend is deliberately mode-native: each qMERA register position is a
    two-state fermionic mode, e.g. ``(site, "up")`` or ``(site, "down")``.
    Symmray's graded array algebra carries fermionic signs during contraction;
    no Jordan-Wigner strings are inserted here.
    """

    symmetry: str = "U1U1"
    site_modes: tuple[Any, ...] = ("up", "down")
    mode_charges: Mapping[Any, Any] | None = None
    dtype: Any = "complex128"
    to_backend: Any = None
    flat: bool = False
    mode_order: str = "site-major"
    zero_charge: Any = (0, 0)
    _mode_charges: dict[Any, tuple[Any, Any]] = field(init=False, repr=False)

    def __post_init__(self):
        modes = tuple(self.site_modes)
        if not modes:
            raise ValueError("site_modes must not be empty.")
        object.__setattr__(self, "site_modes", modes)
        charges = self._default_mode_charges(modes)
        if self.mode_charges is not None:
            charges.update(dict(self.mode_charges))
        missing = [mode for mode in modes if mode not in charges]
        if missing:
            raise ValueError(f"mode_charges is missing mode labels {missing!r}.")
        object.__setattr__(
            self,
            "_mode_charges",
            {mode: tuple(charges[mode]) for mode in modes},
        )
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @staticmethod
    def _default_mode_charges(modes):
        if tuple(modes) == ("up", "down"):
            return {
                "up": ((0, 0), (1, 0)),
                "down": ((0, 0), (0, 1)),
            }
        if len(modes) == 1:
            return {modes[0]: (0, 1)}
        return {mode: (0, idx + 1) for idx, mode in enumerate(modes)}

    def mode_from_label(self, mode_label):
        """Return the local mode kind from a canonical qMERA mode label."""
        if mode_label in self.site_modes:
            return mode_label
        if isinstance(mode_label, tuple) and len(mode_label) >= 2:
            candidate = mode_label[-1]
            if candidate in self.site_modes:
                return candidate
        raise KeyError(f"Cannot infer qMERA fermion mode from {mode_label!r}.")

    def mode_index_map(self, mode_label):
        """Return the Symmray charge index map for one two-state mode."""
        mode = self.mode_from_label(mode_label)
        return list(self._mode_charges[mode])

    def vacuum_vector(self, mode_label):
        """Return the Symmray fermionic ``|0>`` vector for one mode."""
        sr = _require_symmray()
        data = np.array([1.0, 0.0], dtype=self.dtype)
        vector = sr.utils.from_dense(
            data,
            symmetry=self.symmetry,
            index_maps=[self.mode_index_map(mode_label)],
            duals=[False],
            fermionic=True,
            charge=self.zero_charge,
            flat=self.flat,
        )
        return _apply_to_array_blocks(vector, self.to_backend)

    def product_state(self, schedule, sites, *, physical_dim=2, array_backend=None):
        """Return a quimb TN vacuum state on scheduled qMERA register sites."""
        if int(physical_dim) != 2:
            raise NotImplementedError("Symmray qMERA fermion modes are two-state.")
        _ = array_backend
        tensors = []
        for register_site in tuple(sites):
            mode = schedule.geometry.to_mode(register_site)
            data = self.vacuum_vector(mode)
            tensors.append(
                qtn.Tensor(
                    data,
                    inds=(f"k{register_site}",),
                    tags=(f"I{register_site}",),
                )
            )
        return qtn.TensorNetwork(tensors)

    def _fermionic_operator(self, terms, mode_labels):
        sr = _require_symmray()
        flo = _require_fermionic_local_operators()
        operators = [
            flo.FermionicOperator(chr(ord("a") + idx))
            for idx, _ in enumerate(mode_labels)
        ]
        bases = tuple(((), (op.dag,)) for op in operators)
        index_maps = [self.mode_index_map(mode) for mode in mode_labels]

        parsed_terms = []
        for coeff, op_refs in terms:
            ops = []
            for which, dag in op_refs:
                op = operators[int(which)]
                ops.append(op.dag if dag else op)
            parsed_terms.append((coeff, tuple(ops)))

        dense = np.zeros((2,) * (2 * len(mode_labels)), dtype=self.dtype)
        for idx, value in flo.build_local_fermionic_elements(
            tuple(parsed_terms),
            bases,
        ).items():
            dense[idx] += value

        array = sr.utils.from_dense(
            dense,
            self.symmetry,
            index_maps=index_maps * 2,
            duals=[False] * len(mode_labels) + [True] * len(mode_labels),
            fermionic=True,
            charge=self.zero_charge,
            flat=self.flat,
        )
        return _apply_to_array_blocks(array, self.to_backend)

    def operator_from_dense(self, array, mode_labels):
        """Convert a dense local mode operator to a Symmray fermionic array."""
        sr = _require_symmray()
        mode_labels = tuple(mode_labels)
        if not mode_labels:
            raise ValueError("mode_labels must not be empty.")
        arr = np.asarray(array, dtype=self.dtype)
        if arr.ndim == 2:
            dim = 2 ** len(mode_labels)
            if arr.shape != (dim, dim):
                raise ValueError(
                    f"dense operator shape {arr.shape} does not match "
                    f"{len(mode_labels)} two-state modes."
                )
            arr = arr.reshape((2,) * (2 * len(mode_labels)))
        expected = (2,) * (2 * len(mode_labels))
        if tuple(arr.shape) != expected:
            raise ValueError(
                f"dense operator shape {arr.shape} does not match expected {expected}."
            )
        index_maps = [self.mode_index_map(mode) for mode in mode_labels]
        array = sr.utils.from_dense(
            arr,
            self.symmetry,
            index_maps=index_maps * 2,
            duals=[False] * len(mode_labels) + [True] * len(mode_labels),
            fermionic=True,
            charge=self.zero_charge,
            flat=self.flat,
        )
        return _apply_to_array_blocks(array, self.to_backend)

    def number_operator(self, mode_label, *, coefficient=1.0):
        """Return ``coefficient * n`` for one fermionic mode."""
        return self._fermionic_operator(
            ((coefficient, ((0, True), (0, False))),),
            (mode_label,),
        )

    def onsite_interaction_operator(self, up_mode, down_mode, *, U=1.0):
        """Return ``U * n_up * n_down`` on two onsite modes."""
        return self._fermionic_operator(
            ((U, ((0, True), (0, False), (1, True), (1, False))),),
            (up_mode, down_mode),
        )

    def hopping_operator(self, left_mode, right_mode, *, t=1.0, peierls_angle=0.0):
        """Return native fermionic ``-t c_l^dag c_r + h.c.`` on two modes."""
        self._require_matching_mode_maps(left_mode, right_mode, name="hopping_operator")
        phase = np.exp(1.0j * peierls_angle)
        phase_conj = np.conjugate(phase)
        return self._fermionic_operator(
            (
                (-t * phase, ((0, True), (1, False))),
                (-t * phase_conj, ((1, True), (0, False))),
            ),
            (left_mode, right_mode),
        )

    def fsim_gate(self, left_mode, right_mode, *, theta=0.0, phi=0.0):
        """Return a native Symmray fermionic fSim gate for two like modes."""
        self._require_matching_mode_maps(left_mode, right_mode, name="fsim_gate")
        return self.operator_from_dense(dense_fsim((theta, phi)), (left_mode, right_mode))

    def _require_matching_mode_maps(self, left_mode, right_mode, *, name):
        if self.mode_index_map(left_mode) != self.mode_index_map(right_mode):
            raise ValueError(
                f"{name} requires two modes with matching charge maps; "
                "spin-changing operations are not neutral in this backend."
            )

    def fermi_hubbard_terms(
        self,
        geometry,
        *,
        t=1.0,
        U=8.0,
        mu=0.0,
        peierls_angle=0.0,
        include_hopping=True,
        include_onsite=True,
        include_chemical=True,
    ):
        """Return native Symmray local terms for a qMERA Fermi-Hubbard model."""
        if tuple(geometry.site_modes or ()) != tuple(self.site_modes):
            raise ValueError(
                "geometry.site_modes must match the Symmray fermion backend "
                f"site_modes={self.site_modes!r}."
            )
        if len(self.site_modes) != 2:
            raise NotImplementedError(
                "fermi_hubbard_terms currently expects spinful two-mode sites."
            )

        up, down = self.site_modes
        terms = []
        if include_onsite:
            for site in geometry.site_labels:
                up_mode = geometry.mode_label(site, up)
                down_mode = geometry.mode_label(site, down)
                U_site = _node_parameter(U, site)
                if U_site != 0:
                    terms.append(
                        LocalTerm(
                            where=(up_mode, down_mode),
                            operator=self.onsite_interaction_operator(
                                up_mode,
                                down_mode,
                                U=U_site,
                            ),
                            metadata={
                                "kind": "hubbard-onsite",
                                "site": site,
                                "fermionic": True,
                                "backend": "symmray",
                            },
                        )
                    )
        if include_chemical:
            mu_up, mu_down = _as_spin_pair(mu, name="mu")
            for site in geometry.site_labels:
                for mode, mu_mode in ((up, mu_up), (down, mu_down)):
                    mu_site = _node_parameter(mu_mode, site)
                    if mu_site == 0:
                        continue
                    mode_label = geometry.mode_label(site, mode)
                    terms.append(
                        LocalTerm(
                            where=(mode_label,),
                            operator=self.number_operator(
                                mode_label,
                                coefficient=-mu_site,
                            ),
                            metadata={
                                "kind": "hubbard-chemical",
                                "site": site,
                                "mode": mode,
                                "fermionic": True,
                                "backend": "symmray",
                            },
                        )
                    )
        if include_hopping:
            for left, right in geometry.nearest_neighbor_edges():
                for mode in self.site_modes:
                    left_mode = geometry.mode_label(left, mode)
                    right_mode = geometry.mode_label(right, mode)
                    t_edge = _edge_parameter(t, left, right)
                    if t_edge == 0:
                        continue
                    angle = _edge_angle_parameter(peierls_angle, left, right)
                    terms.append(
                        LocalTerm(
                            where=(left_mode, right_mode),
                            operator=self.hopping_operator(
                                left_mode,
                                right_mode,
                                t=t_edge,
                                peierls_angle=angle,
                            ),
                            metadata={
                                "kind": "hubbard-hopping",
                                "edge": (left, right),
                                "mode": mode,
                                "fermionic": True,
                                "backend": "symmray",
                            },
                        )
                    )
        return tuple(terms)


def qmera_symmray_fermi_hubbard_terms(geometry, **kwargs):
    """Return native Symmray fermionic Hubbard terms for a qMERA geometry."""
    backend = kwargs.pop("backend", None)
    if backend is None:
        backend = QMeraSymmrayFermionBackend(
            site_modes=tuple(geometry.site_modes or ("up", "down"))
        )
    return backend.fermi_hubbard_terms(geometry, **kwargs)


def _null_context_gate(_params):
    raise ValueError("This qMERA gate family requires placement context.")


def symmray_fermion_gate_registry(backend=None, *, base_registry=None):
    """Return a qMERA gate registry with Symmray-native fermionic gates."""
    backend = QMeraSymmrayFermionBackend() if backend is None else backend
    registry = (
        default_gate_registry()
        if base_registry is None
        else base_registry.copy()
    )

    def fsim_context(params, *, placement=None, schedule=None, array_backend=None):
        _ = array_backend
        if placement is None or schedule is None:
            raise ValueError("symmray-fsim requires qMERA placement and schedule.")
        left, right = (
            schedule.geometry.to_mode(register_site)
            for register_site in placement.where
        )
        return backend.fsim_gate(left, right, theta=params[0], phi=params[1])

    registry.register(
        GateSpec(
            "symmray-fsim",
            2,
            2,
            _null_context_gate,
            family="fermion",
            convention="symmray-fermionic-mode",
            default_tags=("SYMMRAY_FSIM", "FSIM"),
            arity_kind="mode",
            preserves_parity=True,
            mode_order="register",
            contextual_generator=fsim_context,
        )
    )
    return registry
