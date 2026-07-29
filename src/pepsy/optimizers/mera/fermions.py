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
    "qmera_symmray_majorana_terms",
    "symmray_fermion_gate_registry",
    "symmray_majorana_gate_registry",
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
    zero_charge: Any = None
    _mode_charges: dict[Any, tuple[Any, Any]] = field(init=False, repr=False)

    @classmethod
    def from_fermion(cls, fermion, **kwargs):
        """Construct the mode backend from a unified :class:`Fermion` model.

        Spinful models use the canonical ``("up", "down")`` pair while a
        spinless model uses one ``"mode"`` register. Callers can still
        override backend-specific options, including ``mode_order``.
        """
        spinful = bool(getattr(fermion, "spinful", False))
        site_modes = ("up", "down") if spinful else ("mode",)
        kwargs.setdefault("symmetry", getattr(fermion, "symmetry", "U1U1"))
        kwargs.setdefault("site_modes", site_modes)
        kwargs.setdefault("mode_order", "mode-major")
        if "dtype" not in kwargs and hasattr(fermion, "dtype"):
            kwargs["dtype"] = fermion.dtype
        if "to_backend" not in kwargs and hasattr(fermion, "to_backend"):
            kwargs["to_backend"] = fermion.to_backend
        return cls(**kwargs)

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
        if self.zero_charge is None:
            zero_charge = (0, 0) if self.symmetry in {"U1U1", "Z2Z2"} else 0
            object.__setattr__(self, "zero_charge", zero_charge)
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

    def vacuum_vector(self, mode_label, *, occupied=False):
        """Return the Symmray fermionic ``|0>`` or ``|1>`` mode vector."""
        sr = _require_symmray()
        data = np.array(
            [0.0, 1.0] if occupied else [1.0, 0.0],
            dtype=self.dtype,
        )
        vector = sr.utils.from_dense(
            data,
            symmetry=self.symmetry,
            index_maps=[self.mode_index_map(mode_label)],
            duals=[False],
            fermionic=True,
            charge=(
                self.zero_charge
                if not occupied
                else self._mode_charges[self.mode_from_label(mode_label)][1]
            ),
            label=None if not occupied else f"pepsy_mode_{mode_label!r}",
            flat=self.flat,
        )
        return _apply_to_array_blocks(vector, self.to_backend)

    def product_state(
        self,
        schedule,
        sites,
        *,
        physical_dim=2,
        array_backend=None,
        occupations=None,
    ):
        """Return a product state on scheduled qMERA register sites.

        ``occupations`` can be a sequence aligned with ``sites`` or a mapping
        keyed by register-site or mode labels. The default is the vacuum.
        """
        if int(physical_dim) != 2:
            raise NotImplementedError("Symmray qMERA fermion modes are two-state.")
        _ = array_backend
        sites = tuple(sites)
        if occupations is None:
            occupation_values = (False,) * len(sites)
        elif isinstance(occupations, Mapping):
            occupation_values = tuple(
                bool(
                    occupations.get(
                        register_site,
                        occupations.get(schedule.geometry.to_mode(register_site), False),
                    )
                )
                for register_site in sites
            )
        else:
            occupation_values = tuple(bool(value) for value in occupations)
            if len(occupation_values) != len(sites):
                raise ValueError("occupations must align with the register sites.")
        tensors = []
        for register_site, occupied in zip(sites, occupation_values):
            mode = schedule.geometry.to_mode(register_site)
            data = self.vacuum_vector(mode, occupied=occupied)
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
        # Keep Torch/JAX/CuPy values on their original backend. In particular,
        # a trainable Torch fSim parameter cannot be converted through NumPy.
        if isinstance(array, np.ndarray) or not hasattr(array, "shape"):
            arr = np.asarray(array, dtype=self.dtype)
        else:
            arr = array
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
        # A backend converter configured for non-trainable state data must not
        # detach a computed gate carrying a Torch autodiff graph.
        to_backend = None if getattr(arr, "requires_grad", False) else self.to_backend
        return _apply_to_array_blocks(array, to_backend)

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


def qmera_symmray_fermi_hubbard_terms(geometry, *, fermion=None, **kwargs):
    """Return native Symmray fermionic Hubbard terms for a qMERA geometry.

    Parameters
    ----------
    geometry : QMeraGeometry
        Geometry whose physical sites are expanded into two-state modes.
    fermion : pepsy.Fermion, optional
        Unified model helper supplying only the local symmetry convention.
        Pass ``t=``, ``U=``, and optional ``mu=`` explicitly.
        qMERA deliberately remains mode-native, so this adapter accepts only
        a spinful ``U1U1`` helper and does not turn a four-state site tensor
        into a qMERA register tensor.
    """
    if fermion is not None:
        if not getattr(fermion, "spinful", False):
            raise ValueError("qMERA Hubbard terms require a spinful Fermion helper.")
        if str(getattr(fermion, "symmetry", "")) != "U1U1":
            raise ValueError(
                "qMERA Hubbard terms currently require Fermion(symmetry='U1U1')."
            )
        missing = [name for name in ("t", "U") if name not in kwargs]
        if missing:
            raise TypeError(
                "qMERA Fermi-Hubbard terms require explicit "
                + ", ".join(f"{name}=" for name in missing)
                + ". Fermion does not store couplings."
            )
    backend = kwargs.pop("backend", None)
    if backend is None:
        backend = QMeraSymmrayFermionBackend(
            symmetry=getattr(fermion, "symmetry", "U1U1"),
            site_modes=tuple(geometry.site_modes or ("up", "down")),
        )
    elif fermion is not None and str(backend.symmetry) != str(fermion.symmetry):
        raise ValueError("The qMERA backend symmetry must match the Fermion helper.")
    return backend.fermi_hubbard_terms(geometry, **kwargs)


def qmera_symmray_majorana_terms(geometry, *, fermion=None, **kwargs):
    """Return native parity-preserving Majorana terms for qMERA.

    The first qMERA Majorana convention is one spinless complex mode per
    lattice site, represented with ``site_modes=("mode",)`` and conserved
    ``Z2`` fermion parity. The physical Majoranas are the two quadratures of
    each complex mode; they are not separate Hilbert-space sites.

    Parameters
    ----------
    geometry : QMeraGeometry
        A 1D or 2D geometry with one explicit ``"mode"`` per site.
    fermion : pepsy.Fermion, optional
        A spinless ``Fermion(symmetry="Z2")`` helper.
    coupling : scalar, mapping, or callable, optional
        Coefficient of ``i gamma_{j,y} gamma_{k,x}`` on each nearest-neighbor
        edge.
    pairing : scalar, mapping, or callable, optional
        Coefficient of the Hermitian ``c_j^dag c_k^dag + h.c.`` term.
    """
    if tuple(geometry.site_modes or ()) != ("mode",):
        raise ValueError(
            "Majorana qMERA requires geometry site_modes=('mode',), one "
            "complex fermion mode per physical site."
        )
    if fermion is None:
        from ...tensors import Fermion  # pylint: disable=import-outside-toplevel

        fermion = Fermion(spinful=False, symmetry="Z2")
    if getattr(fermion, "spinful", True):
        raise ValueError("Majorana qMERA requires a spinless Fermion helper.")
    if str(getattr(fermion, "symmetry", "")) != "Z2":
        raise ValueError(
            "Majorana qMERA currently uses the native Z2 parity convention."
        )

    coupling = kwargs.pop("coupling", kwargs.pop("t", 1.0))
    pairing = kwargs.pop("pairing", 0.0)
    left_component = kwargs.pop("left_component", 1)
    right_component = kwargs.pop("right_component", 0)
    phase = kwargs.pop("phase", 0.0)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown Majorana qMERA term option(s): {unknown}.")

    terms = []
    for left, right in geometry.nearest_neighbor_edges():
        left_mode = geometry.mode_label(left, "mode")
        right_mode = geometry.mode_label(right, "mode")
        coupling_edge = _edge_parameter(coupling, left, right)
        if coupling_edge != 0:
            terms.append(
                LocalTerm(
                    where=(left_mode, right_mode),
                    operator=fermion.majorana_bilinear_operator(
                        (left_mode, right_mode),
                        left_component=left_component,
                        right_component=right_component,
                        coefficient=coupling_edge,
                    ),
                    metadata={
                        "kind": "majorana-bilinear",
                        "edge": (left, right),
                        "fermionic": True,
                        "symmetry": "Z2",
                        "convention": "i-gamma-y-gamma-x",
                    },
                )
            )
        pairing_edge = _edge_parameter(pairing, left, right)
        if pairing_edge != 0:
            terms.append(
                LocalTerm(
                    where=(left_mode, right_mode),
                    operator=fermion.pairing_operator(
                        (left_mode, right_mode),
                        coefficient=pairing_edge,
                        phase=_edge_angle_parameter(phase, left, right),
                    ),
                    metadata={
                        "kind": "majorana-pairing",
                        "edge": (left, right),
                        "fermionic": True,
                        "symmetry": "Z2",
                        "convention": "creation-pair-plus-hc",
                    },
                )
            )
    return tuple(terms)


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
            raise ValueError(
                "native Symmray fermion gates require qMERA placement and schedule."
            )
        left, right = (
            schedule.geometry.to_mode(register_site)
            for register_site in placement.where
        )
        return backend.fsim_gate(left, right, theta=params[0], phi=params[1])

    common = dict(
        arity=2,
        num_params=2,
        generator=_null_context_gate,
        family="fermion",
        arity_kind="mode",
        preserves_parity=True,
        mode_order="register",
        symmetry=backend.symmetry,
        contextual_generator=fsim_context,
    )
    registry.register(
        GateSpec(
            "symmray-fsim",
            convention="symmray-fermionic-mode",
            default_tags=("SYMMRAY_FSIM", "FSIM"),
            **common,
        )
    )
    registry.register(
        GateSpec(
            "symmray-hubbard",
            convention="symmray-fermionic-hubbard-hopping",
            default_tags=("SYMMRAY_HUBBARD", "HOPPING"),
            **common,
        )
    )
    return registry


def symmray_majorana_gate_registry(backend=None, *, base_registry=None):
    """Return a native ``Z2`` parity-preserving Majorana gate registry."""
    from ...tensors import Fermion  # pylint: disable=import-outside-toplevel

    backend = (
        QMeraSymmrayFermionBackend(symmetry="Z2", site_modes=("mode",))
        if backend is None
        else backend
    )
    if str(backend.symmetry) != "Z2" or tuple(backend.site_modes) != ("mode",):
        raise ValueError(
            "The Majorana gate registry requires a Z2 backend with "
            "site_modes=('mode',)."
        )
    fermion = Fermion(
        spinful=False,
        symmetry="Z2",
        dtype=backend.dtype,
        to_backend=backend.to_backend,
    )
    registry = default_gate_registry() if base_registry is None else base_registry.copy()

    def _mode_pair(placement, schedule):
        return tuple(schedule.geometry.to_mode(site) for site in placement.where)

    def majorana_context(params, *, placement=None, schedule=None, array_backend=None):
        _ = array_backend
        if placement is None or schedule is None:
            raise ValueError("symmray-majorana requires qMERA placement and schedule.")
        left, right = _mode_pair(placement, schedule)
        return fermion.majorana_gate(
            params[0],
            edge=(left, right),
            left_component=1,
            right_component=0,
        )

    def pairing_context(params, *, placement=None, schedule=None, array_backend=None):
        _ = array_backend
        if placement is None or schedule is None:
            raise ValueError("symmray-pairing requires qMERA placement and schedule.")
        left, right = _mode_pair(placement, schedule)
        return fermion.pairing_gate(params[0], edge=(left, right))

    common = dict(
        family="fermion",
        convention="symmray-z2-majorana",
        arity_kind="mode",
        preserves_parity=True,
        mode_order="register",
    )
    registry.register(
        GateSpec(
            "symmray-majorana",
            2,
            1,
            _null_context_gate,
            default_tags=("MAJORANA", "PARITY"),
            contextual_generator=majorana_context,
            **common,
        )
    )
    registry.register(
        GateSpec(
            "symmray-pairing",
            2,
            1,
            _null_context_gate,
            default_tags=("PAIRING", "PARITY"),
            contextual_generator=pairing_context,
            **common,
        )
    )
    return registry
