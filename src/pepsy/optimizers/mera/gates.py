"""Parametrized gate registry for qMERA builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ...backends import get_default_array_backend
from ...operators import cphase, crx, cry, crz, fsim, fsimg, rxx, ryy, rzz, su4

__all__ = [
    "GateRegistry",
    "GateSpec",
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
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "convention", str(self.convention))
        object.__setattr__(self, "default_tags", tuple(self.default_tags))

    def parameters(self, params=None):
        """Validate and return parameters as a tuple."""
        return _params_tuple(params, self.num_params)

    def matrix(self, params=None, *, array_backend=None):
        """Generate a backend-compatible gate tensor."""
        gate = self.generator(self.parameters(params))
        return _convert_array(gate, array_backend)


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

    def names(self, *, arity=None):
        """Return registered gate names, optionally filtered by arity."""
        specs = self._specs.values()
        if arity is not None:
            specs = [spec for spec in specs if spec.arity == arity]
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


def default_gate_registry():
    """Return the default parametrized spin-gate registry."""
    return GateRegistry(
        (
            GateSpec("rxx", 2, 1, _one_param(rxx), default_tags=("RXX",)),
            GateSpec("ryy", 2, 1, _one_param(ryy), default_tags=("RYY",)),
            GateSpec("rzz", 2, 1, _one_param(rzz), default_tags=("RZZ",)),
            GateSpec("cphase", 2, 1, _one_param(cphase), default_tags=("CPHASE",)),
            GateSpec("crx", 2, 1, _one_param(crx), default_tags=("CRX",)),
            GateSpec("cry", 2, 1, _one_param(cry), default_tags=("CRY",)),
            GateSpec("crz", 2, 1, _one_param(crz), default_tags=("CRZ",)),
            GateSpec("fsim", 2, 2, _multi_param(fsim), default_tags=("FSIM",)),
            GateSpec("fsimg", 2, 5, _multi_param(fsimg), default_tags=("FSIMG",)),
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
