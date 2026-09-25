"""Parametrized gate registry for qMERA builders."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import autoray as ar
import numpy as np

from ...backends import get_default_array_backend
from ...operators import cphase, crx, cry, crz, fsim, fsimg, rxx, ryy, rzz, su4

__all__ = [
    "GateRegistry",
    "GateSpec",
    "QMeraPairSpec",
    "available_qmera_pair_ansatzes",
    "available_qmera_pair_ansatze",
    "get_qmera_pair_ansatz",
    "qmera_pair_gate_spec",
    "UserGateFamily",
    "default_gate_registry",
    "resolve_gate_spec",
]


def _params_tuple(params, num_params):
    if num_params == 0:
        return ()
    if params is None:
        return tuple(0.0 for _ in range(num_params))
    if num_params == 1 and not hasattr(params, "__len__"):
        return (params,)
    values = tuple(params)
    if len(values) != num_params:
        raise ValueError(f"expected {num_params} gate parameters, got {len(values)}.")
    return values


def _convert_array(array, array_backend):
    if array_backend is None:
        array_backend = get_default_array_backend()
    return array if array_backend is None else array_backend(array)


def _normalize_family(family):
    key = str(family).strip().lower().replace("_", "-")
    if key in {"fermion", "fermionic"}:
        return "fermion"
    if key in {"spin", "qubit"}:
        return "spin"
    if key in {"user", "custom"}:
        return "user"
    return key


def _normalize_arity_kind(arity_kind):
    key = str(arity_kind).strip().lower().replace("_", "-")
    if key in {"qubit", "qubits", "register", "register-qubit"}:
        return "qubit"
    if key in {"mode", "modes", "fermion-mode", "fermionic-mode"}:
        return "mode"
    if key in {"site", "sites", "lattice-site"}:
        return "site"
    raise ValueError("arity_kind must be 'qubit', 'mode', or 'site'.")


def _normalize_mode_order(mode_order):
    if mode_order is None:
        return None
    key = str(mode_order).strip().lower().replace("_", "-")
    if key in {"register", "register-order"}:
        return "register"
    if key in {"site-major", "site", "interleaved"}:
        return "site-major"
    if key in {"mode-major", "mode", "spin-major"}:
        return "mode-major"
    raise ValueError("mode_order must be 'register', 'site-major', or 'mode-major'.")


@dataclass(frozen=True)
class GateSpec:
    """Parametrized local gate family."""

    name: str
    arity: int
    num_params: int
    generator: Callable[[tuple[Any, ...]], Any]
    family: str = "spin"
    supports_backend: tuple[str, ...] = ("numpy", "torch", "jax", "cupy")
    convention: str = "spin"
    default_tags: tuple[str, ...] = ()
    arity_kind: str = "qubit"
    preserves_parity: bool | None = None
    mode_order: str | None = None
    contextual_generator: Callable[..., Any] | None = None
    symmetry: str | None = None

    def __post_init__(self):
        arity = int(self.arity)
        num_params = int(self.num_params)
        if arity < 1:
            raise ValueError("gate arity must be >= 1.")
        if num_params < 0:
            raise ValueError("num_params must be >= 0.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "num_params", num_params)
        object.__setattr__(self, "family", _normalize_family(self.family))
        object.__setattr__(self, "supports_backend", tuple(self.supports_backend))
        object.__setattr__(self, "convention", str(self.convention))
        object.__setattr__(self, "default_tags", tuple(self.default_tags))
        object.__setattr__(self, "arity_kind", _normalize_arity_kind(self.arity_kind))
        object.__setattr__(
            self,
            "preserves_parity",
            None if self.preserves_parity is None else bool(self.preserves_parity),
        )
        object.__setattr__(self, "mode_order", _normalize_mode_order(self.mode_order))
        object.__setattr__(
            self,
            "symmetry",
            None if self.symmetry is None else str(self.symmetry),
        )

    @property
    def is_fermionic(self):
        """Whether this gate family represents a fermionic-mode operation."""
        return self.family == "fermion"

    def parameters(self, params=None):
        """Validate and return parameters as a tuple."""
        return _params_tuple(params, self.num_params)

    def matrix(self, params=None, *, array_backend=None):
        """Generate a backend-compatible gate tensor."""
        if self.contextual_generator is not None:
            raise ValueError(
                f"Gate family {self.name!r} requires placement context; use "
                "matrix_for_placement(..., placement=..., schedule=...)."
            )
        gate = self.generator(self.parameters(params))
        return _convert_array(gate, array_backend)

    def matrix_for_placement(
        self,
        params=None,
        *,
        placement=None,
        schedule=None,
        array_backend=None,
    ):
        """Generate a gate tensor, optionally using placement context."""
        params = self.parameters(params)
        if self.contextual_generator is None:
            return _convert_array(self.generator(params), array_backend)
        return self.contextual_generator(
            params,
            placement=placement,
            schedule=schedule,
            array_backend=array_backend,
        )


@dataclass(frozen=True)
class UserGateFamily:
    """User-provided parametrized gate family."""

    name: str
    arity: int
    num_params: int
    generator: Callable[[tuple[Any, ...]], Any]
    family: str = "user"
    convention: str = "user"
    default_tags: tuple[str, ...] = ()
    arity_kind: str = "qubit"
    preserves_parity: bool | None = None
    mode_order: str | None = None
    contextual_generator: Callable[..., Any] | None = None
    symmetry: str | None = None

    def to_gate_spec(self):
        """Convert to a registry-ready :class:`GateSpec`."""
        return GateSpec(
            name=self.name,
            arity=self.arity,
            num_params=self.num_params,
            generator=self.generator,
            family=self.family,
            convention=self.convention,
            default_tags=self.default_tags,
            arity_kind=self.arity_kind,
            preserves_parity=self.preserves_parity,
            mode_order=self.mode_order,
            contextual_generator=self.contextual_generator,
            symmetry=self.symmetry,
        )


class GateRegistry:
    """Small registry for parametrized qMERA gate families."""

    def __init__(self, specs=()):
        self._specs = {}
        for spec in specs:
            self.register(spec)

    @staticmethod
    def _key(name):
        return str(name).strip().lower().replace("_", "-")

    def register(self, spec):
        """Register a :class:`GateSpec` or :class:`UserGateFamily`."""
        if isinstance(spec, UserGateFamily):
            spec = spec.to_gate_spec()
        if not isinstance(spec, GateSpec):
            raise TypeError("register expects a GateSpec or UserGateFamily.")
        self._specs[self._key(spec.name)] = spec
        return self

    def get(self, name):
        """Return a registered gate spec by name."""
        key = self._key(name)
        try:
            return self._specs[key]
        except KeyError as exc:
            known = ", ".join(sorted(spec.name for spec in self._specs.values()))
            raise KeyError(f"Unknown qMERA gate family {name!r}. Known: {known}.") from exc

    def names(self, *, arity=None, family=None, arity_kind=None):
        """Return registered gate names, optionally filtered by arity."""
        specs = self._specs.values()
        if arity is not None:
            specs = [spec for spec in specs if spec.arity == arity]
        if family is not None:
            family = _normalize_family(family)
            specs = [spec for spec in specs if spec.family == family]
        if arity_kind is not None:
            arity_kind = _normalize_arity_kind(arity_kind)
            specs = [spec for spec in specs if spec.arity_kind == arity_kind]
        return tuple(sorted(spec.name for spec in specs))

    def copy(self):
        """Return a shallow copy of the registry."""
        return GateRegistry(self._specs.values())


def _one_param(fn):
    return lambda params: fn(params[0])


def _multi_param(fn):
    return lambda params: fn(params)


def _identity_2q(_params):
    return np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)


_QMERA_PAULIS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}
_QMERA_Z2_WORDS = frozenset(("XI", "IX", "XX", "YY", "YZ", "ZY", "ZZ"))
_QMERA_ALL_WORDS = frozenset(
    first + second
    for first in "IXYZ"
    for second in "IXYZ"
    if first + second != "II"
)
_QMERA_PAIR_PRESETS = {
    "z2_zz_yy_rx": ("ZZ", "YY", "XI", "IX"),
    "z2_rx_zz_yy_xx_rx": ("XI", "IX", "ZZ", "YY", "XX", "XI", "IX"),
    "z2_zz_rx": ("ZZ", "XI", "IX"),
    "z2_yz_zy": ("YZ", "ZY"),
    "z2_all_paulis": ("XI", "IX", "XX", "YY", "YZ", "ZY", "ZZ"),
    "all_paulis": (
        "XI", "IX", "YI", "IY", "ZI", "IZ", "XX", "XY", "XZ",
        "YX", "YY", "YZ", "ZX", "ZY", "ZZ",
    ),
}
_QMERA_PAIR_ALIASES = {
    "minimal": "z2_zz_yy_rx",
    "extended": "z2_rx_zz_yy_xx_rx",
    "ising": "z2_zz_rx",
    "real_z2": "z2_yz_zy",
    "pauli_z2": "z2_all_paulis",
    "unrestricted": "all_paulis",
}


@dataclass(frozen=True)
class QMeraPairSpec:
    """Ordered two-qubit Pauli rotations with an explicit symmetry contract.

    Each listed word has its own angle, including every repetition. The first
    letter acts on the first scheduled wire. ``global-X-Z2`` rejects words
    that do not commute with X tensor X; ``unrestricted`` allows all 15
    nonidentity Pauli words. Native fermion modes use their own registry.
    """

    generators: tuple[str, ...]
    repetitions: int = 1
    name: str = "custom"
    symmetry: str = "global-X-Z2"
    initialization: str = "independent"

    def __post_init__(self):
        if isinstance(self.generators, str):
            raise ValueError("generators must be a sequence of two-qubit Pauli words.")
        try:
            words = tuple(self.generators)
        except TypeError as exc:
            raise ValueError("generators must be a nonempty sequence of Pauli words.") from exc
        symmetry = str(self.symmetry).strip().lower().replace("_", "-")
        if symmetry == "z2":
            symmetry = "global-x-z2"
        if symmetry not in {"global-x-z2", "unrestricted"}:
            raise ValueError("symmetry must be 'global-X-Z2' or 'unrestricted'.")
        allowed = _QMERA_Z2_WORDS if symmetry == "global-x-z2" else _QMERA_ALL_WORDS
        if not words or any(
            not isinstance(word, str) or word not in allowed
            for word in words
        ):
            if symmetry == "global-x-z2":
                raise ValueError("qMERA pair generators must preserve global-X parity.")
            raise ValueError("Unrestricted qMERA generators must be nonidentity Pauli words.")
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions < 1
        ):
            raise ValueError("repetitions must be a positive integer.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string.")
        if self.initialization not in {"independent", "shared"}:
            raise ValueError("initialization must be 'independent' or 'shared'.")
        object.__setattr__(self, "generators", words)
        object.__setattr__(
            self, "symmetry",
            "global-X-Z2" if symmetry == "global-x-z2" else "unrestricted",
        )

    @property
    def preserves_global_x(self):
        """Whether every allowed angle commutes with global X parity."""
        return self.symmetry == "global-X-Z2"

    @property
    def num_params(self):
        """Number of independent angles per scheduled pair placement."""
        return len(self.generators) * self.repetitions

    @property
    def rotation_sequence(self):
        """Chronological logical rotations as ``(gate, pair-wire-indices)``.

        Indices 0 and 1 refer to the first and second scheduled pair wires.
        Each entry consumes one independent angle. Pepsy contracts the whole
        sequence into one differentiable pair tensor for each placement.
        """
        rotations = []
        for word in self.generators * self.repetitions:
            if word[1] == "I":
                rotations.append(("R" + word[0], (0,)))
            elif word[0] == "I":
                rotations.append(("R" + word[1], (1,)))
            else:
                rotations.append(("R" + word, (0, 1)))
        return tuple(rotations)


def available_qmera_pair_ansatzes(*, include_aliases=False):
    """Return descriptive built-in names, optionally including old aliases."""
    names = tuple(_QMERA_PAIR_PRESETS)
    return names + tuple(_QMERA_PAIR_ALIASES) if include_aliases else names


def available_qmera_pair_ansatze(*, include_aliases=False):
    """Return the old family names; use ``ansatzes`` for preferred names."""
    aliases = tuple(_QMERA_PAIR_ALIASES)
    return aliases + tuple(_QMERA_PAIR_PRESETS) if include_aliases else aliases


def get_qmera_pair_ansatz(ansatz="minimal", *, repetitions=None):
    """Resolve a named or custom pair ansatz, optionally repeating it."""
    if isinstance(ansatz, str):
        name = ansatz.strip().lower().replace("-", "_")
        canonical = _QMERA_PAIR_ALIASES.get(name, name)
        try:
            words = _QMERA_PAIR_PRESETS[canonical]
        except KeyError as exc:
            choices = ", ".join(available_qmera_pair_ansatzes())
            raise ValueError(
                f"Unknown qMERA pair ansatz {ansatz!r}; choose {choices} "
                "or supply a QMeraPairSpec. Old names remain aliases."
            ) from exc
        resolved = QMeraPairSpec(
            words,
            name=name.replace("_", "-") if name in _QMERA_PAIR_ALIASES else canonical,
            symmetry="unrestricted" if canonical == "all_paulis" else "global-X-Z2",
            initialization="shared" if canonical == "z2_zz_yy_rx" else "independent",
        )
    elif isinstance(ansatz, QMeraPairSpec):
        resolved = ansatz
    else:
        raise TypeError("ansatz must be a preset name or a QMeraPairSpec.")
    return resolved if repetitions is None else replace(resolved, repetitions=repetitions)


def _qmera_pair_matrix(spec, params):
    """Compose chronological Pauli rotations on the parameter backend."""
    params = tuple(params)
    if len(params) != spec.num_params:
        raise ValueError(f"expected {spec.num_params} qMERA pair angles.")
    identity = ar.do("array", np.eye(4, dtype=np.complex128), like=params[0])
    result = identity
    for word, angle in zip(spec.generators * spec.repetitions, params):
        pauli = ar.do(
            "array",
            np.kron(_QMERA_PAULIS[word[0]], _QMERA_PAULIS[word[1]]),
            like=angle,
        )
        rotation = (
            ar.do("cos", angle / 2) * identity
            - 1j * ar.do("sin", angle / 2) * pauli
        )
        result = ar.do("matmul", rotation, result)
    return ar.do("reshape", result, (2, 2, 2, 2))


def qmera_pair_gate_spec(spec, *, name=None):
    """Create a backend-differentiable gate family for one qMERA pair spec."""
    if not isinstance(spec, QMeraPairSpec):
        raise TypeError("spec must be a QMeraPairSpec.")
    gate_name = f"qmera-{spec.name}" if name is None else str(name)
    return GateSpec(
        gate_name,
        2,
        spec.num_params,
        lambda params: _qmera_pair_matrix(spec, params),
        family="spin",
        convention=f"{spec.symmetry.lower()}-pauli",
        default_tags=("QMERA_PAIR",),
        preserves_parity=spec.preserves_global_x,
        symmetry=spec.symmetry if spec.preserves_global_x else None,
    )


def default_gate_registry():
    """Return the default parametrized spin-gate registry."""
    return GateRegistry(
        (
            *(
                qmera_pair_gate_spec(get_qmera_pair_ansatz(name))
                for name in (*_QMERA_PAIR_PRESETS, *_QMERA_PAIR_ALIASES)
            ),
            GateSpec("rxx", 2, 1, _one_param(rxx), default_tags=("RXX",)),
            GateSpec("ryy", 2, 1, _one_param(ryy), default_tags=("RYY",)),
            GateSpec("rzz", 2, 1, _one_param(rzz), default_tags=("RZZ",)),
            GateSpec("cphase", 2, 1, _one_param(cphase), default_tags=("CPHASE",)),
            GateSpec("crx", 2, 1, _one_param(crx), default_tags=("CRX",)),
            GateSpec("cry", 2, 1, _one_param(cry), default_tags=("CRY",)),
            GateSpec("crz", 2, 1, _one_param(crz), default_tags=("CRZ",)),
            GateSpec(
                "fsim",
                2,
                2,
                _multi_param(fsim),
                family="fermion",
                convention="fermionic-mode",
                default_tags=("FSIM",),
                arity_kind="mode",
                preserves_parity=True,
                mode_order="register",
            ),
            GateSpec(
                "fsimg",
                2,
                5,
                _multi_param(fsimg),
                family="fermion",
                convention="fermionic-mode",
                default_tags=("FSIMG",),
                arity_kind="mode",
                preserves_parity=True,
                mode_order="register",
            ),
            GateSpec("su4", 2, 15, _multi_param(su4), default_tags=("SU4",)),
            GateSpec("identity-2q", 2, 0, _identity_2q, default_tags=("ID2",)),
        )
    )


def resolve_gate_spec(gate_family, registry=None):
    """Resolve a gate-family name or object to :class:`GateSpec`."""
    if isinstance(gate_family, GateSpec):
        return gate_family
    if isinstance(gate_family, UserGateFamily):
        return gate_family.to_gate_spec()
    registry = default_gate_registry() if registry is None else registry
    return registry.get(gate_family)
