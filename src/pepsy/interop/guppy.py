"""Convert straight-line Guppy programs into Pepsy gate streams.

Guppy programs compile to HUGR, which is a data-flow IR rather than a flat
list of gates.  This adapter deliberately accepts only a straight-line HUGR
region.  A branch, loop, call, dynamic qubit lifetime, or unsupported quantum
operation cannot be represented by one deterministic Pepsy gate stream and is
rejected with :class:`GuppyConversionError`.

The optional ``guppylang`` dependency is imported by the caller when compiling
the program, not by this module.  Therefore importing :mod:`pepsy.interop`
does not make Guppy a mandatory Pepsy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..operators.primitives import (
    crz,
    cx,
    cy,
    cz,
    h,
    rx,
    ry,
    rz,
    s,
    sdg,
    swap,
    t,
    tdg,
    x,
    y,
    z,
)

__all__ = [
    "GuppyConversionError",
    "GuppyGateStream",
    "GuppyMeasurement",
    "guppy_gate_stream",
]


class GuppyConversionError(ValueError):
    """Raised when a Guppy/HUGR program cannot be represented statically."""


@dataclass
class GuppyMeasurement:
    """A Guppy measurement and the Pepsy site that represents it."""

    site: int
    result: str | None = None


class GuppyGateStream(list):
    """List-compatible Pepsy stream with Guppy circuit metadata.

    The object is intentionally a ``list`` subclass, so it can be passed
    directly to ``MpsOptimizer``, ``MpoOptimizer``, ``PepsOptimizer``, or
    ``MpsStabOptimizer``.  ``n_qubits`` is the number of statically allocated
    qubits (all start in ``|0>`` for Pepsy's product-state constructors).
    """

    def __init__(
        self,
        entries=(),
        *,
        n_qubits: int,
        measurements=(),
        initial_bits=None,
        format: str = "matrix",
    ):
        super().__init__(entries)
        self.n_qubits = int(n_qubits)
        self.measurements = tuple(measurements)
        self.initial_bits = tuple(
            int(bit) for bit in (initial_bits or (0,) * self.n_qubits)
        )
        self.format = str(format)

    @property
    def gate_stream(self):
        """Return the ordinary list view used by Pepsy optimizers."""

        return self


_CONTROL_FLOW_OPS = {
    "CFG",
    "DFG",
    "Conditional",
    "TailLoop",
    "Call",
    "FuncDefn",
    "Function",
    "Tag",
}

_SINGLE_MATRICES = {
    "H": h,
    "S": s,
    "Sdg": sdg,
    "T": t,
    "Tdg": tdg,
    "X": x,
    "Y": y,
    "Z": z,
}

_NAMED_GATES = {
    "H": "h",
    "S": "s",
    "Sdg": "sdg",
    "T": "t",
    "Tdg": "tdg",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "CX": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SWAP": "swap",
}


def _op_name(op: Any) -> str:
    """Return the unqualified operation name across HUGR API versions."""

    op_def = getattr(op, "_op_def", None)
    name = getattr(op_def, "name", None)
    if name:
        return str(name).split("<", 1)[0]
    method = getattr(op, "name", None)
    name = method() if callable(method) else type(op).__name__
    return str(name).rsplit(".", 1)[-1].split("<", 1)[0]


def _op_kind(op: Any) -> str:
    return type(op).__name__


def _sources(module: Any, node: Any) -> dict[int, Any]:
    """Return value-input port offsets and their unique source ports."""

    result = {}
    for port, linked in module.incoming_links(node):
        if port.offset < 0:
            continue
        if len(linked) != 1:
            raise GuppyConversionError(
                f"HUGR input {node} port {port.offset} has {len(linked)} sources; "
                "fan-in is not a static gate stream."
            )
        result[int(port.offset)] = linked[0]
    return result


def _value_half_turns(value: Any) -> float:
    """Read HUGR's public ``ConstRotation`` value representation."""

    if getattr(value, "name", None) != "ConstRotation":
        raise GuppyConversionError(
            "Only compile-time Guppy angles are supported; a dynamic angle "
            f"has value {value!r}."
        )
    payload = getattr(value, "val", None)
    if not isinstance(payload, dict) or "half_turns" not in payload:
        raise GuppyConversionError(f"Malformed HUGR rotation constant {value!r}.")
    try:
        half_turns = float(payload["half_turns"])
    except (TypeError, ValueError) as exc:
        raise GuppyConversionError(
            f"Malformed HUGR rotation constant {value!r}."
        ) from exc
    if not math.isfinite(half_turns):
        raise GuppyConversionError("Guppy rotation angles must be finite.")
    # Guppy angle(1) is pi radians. Pepsy rotation constructors use radians.
    return half_turns * math.pi


def _toffoli() -> np.ndarray:
    matrix = np.eye(8, dtype=complex)
    matrix[6, 6] = 0.0
    matrix[7, 7] = 0.0
    matrix[6, 7] = 1.0
    matrix[7, 6] = 1.0
    return matrix


def _sqrt_x(sign: int = 1) -> np.ndarray:
    """Return Guppy's V/Vdg matrix (the global phase is intentional)."""

    return np.array([[1.0, -1j * sign], [-1j * sign, 1.0]], dtype=complex) / math.sqrt(2)


def _matrix_gate(name: str, theta: float | None = None) -> np.ndarray:
    if name in _SINGLE_MATRICES:
        return np.asarray(_SINGLE_MATRICES[name](), dtype=complex)
    if name == "V":
        return _sqrt_x(+1)
    if name == "Vdg":
        return _sqrt_x(-1)
    if name == "CX":
        return np.asarray(cx(), dtype=complex)
    if name == "CY":
        return np.asarray(cy(), dtype=complex)
    if name == "CZ":
        return np.asarray(cz(), dtype=complex)
    if name == "SWAP":
        return np.asarray(swap(), dtype=complex)
    if name == "Toffoli":
        return _toffoli()
    if name in {"Rx", "Ry", "Rz"}:
        if theta is None:
            raise GuppyConversionError(f"{name} is missing its angle.")
        constructor = {"Rx": rx, "Ry": ry, "Rz": rz}[name]
        return np.asarray(constructor(theta), dtype=complex)
    if name == "CRz":
        if theta is None:
            raise GuppyConversionError("CRz is missing its angle.")
        return np.asarray(crz(theta), dtype=complex)
    raise GuppyConversionError(
        f"Unsupported Guppy quantum operation {name!r}; decompose it before "
        "converting to a Pepsy stream."
    )


def _named_entry(name: str, sites: tuple[int, ...], theta: float | None):
    if name in _NAMED_GATES:
        return (_NAMED_GATES[name], *sites)
    if name in {"Rx", "Ry", "Rz"}:
        return (name.lower(), float(theta), sites[0])
    if name == "V":
        return ("sqrt_x", sites[0])
    if name == "Vdg":
        return ("sqrt_x_dag", sites[0])
    # MpsStabOptimizer accepts dense matrices as well, which is the safest
    # representation for Toffoli and controlled rotations.
    return (_matrix_gate(name, theta), sites)


def _compile_package(program: Any) -> Any:
    if hasattr(program, "modules"):
        return program
    compile_method = getattr(program, "compile", None)
    if callable(compile_method):
        return compile_method()
    raise TypeError(
        "program must be a @guppy definition with .compile(), or a HUGR "
        "Package returned by .compile()."
    )


def guppy_gate_stream(
    program: Any,
    *,
    format: str = "matrix",
) -> GuppyGateStream:
    """Compile a straight-line Guppy program into a Pepsy gate stream.

    Parameters
    ----------
    program
        A Guppy definition (normally decorated with ``@guppy``) or its HUGR
        ``Package``.  Guppy itself is an optional dependency; install it with
        ``pip install 'pepsy[guppy]'``.
    format : {"matrix", "named"}, default="matrix"
        ``"matrix"`` is accepted by dense MPS/MPO/PEPS optimizers and by
        ``MpsStabOptimizer``.  ``"named"`` uses Pepsy's compact named entries
        where possible and is convenient for stabilizer replay; operations
        without an exact named equivalent remain dense matrices.

    Returns
    -------
    GuppyGateStream
        A list-compatible stream.  Use ``stream.n_qubits`` as the optimizer's
        initial qubit count.  Guppy ``MeasureFree`` becomes a Pepsy Z-measure
        event; Pepsy retains the fixed site because it cannot delete a site
        from the middle of a live MPS without an explicit cap policy.

    Notes
    -----
    This is intentionally a static adapter.  Guppy's branches, loops,
    mid-circuit feed-forward, calls, and dynamic qubit allocation need a
    trajectory executor rather than one gate stream and are rejected.
    """

    normalized_format = str(format).strip().lower()
    if normalized_format not in {"matrix", "named"}:
        raise ValueError("format must be 'matrix' or 'named'.")

    package = _compile_package(program)
    modules = getattr(package, "modules", None)
    if not modules:
        raise GuppyConversionError("The compiled Guppy package contains no HUGR module.")
    if len(modules) != 1:
        raise GuppyConversionError(
            "The static Guppy adapter requires one HUGR module; linked helper "
            "functions or calls need a circuit-level lowering first."
        )
    module = modules[0]
    # ``module_root`` contains the function definition itself.  The compiled
    # entrypoint is the region whose direct children are the executable body.
    root = getattr(module, "entrypoint", None) or getattr(module, "module_root", None)
    try:
        # HUGR stores the body children in source/data-flow construction order.
        # A topological sort is not sufficient here: an independent QAlloc may
        # legally be scheduled after a gate even though Guppy allocated it
        # before that gate.
        nodes = list(module.children(root)) if root is not None else list(module)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GuppyConversionError("Could not read the HUGR entrypoint region.") from exc

    stream = []
    wire_sites = {}
    value_ports = {}
    measurement_ports = {}
    measurements = []
    n_qubits = 0
    saw_quantum_operation = False
    saw_nonallocation_quantum_operation = False

    for node in nodes:
        op = module[node].op
        kind = _op_kind(op)
        name = _op_name(op)
        sources = _sources(module, node)

        if kind == "Const":
            value_ports[node.out(0)] = getattr(op, "val", None)
            continue
        if kind == "LoadConst":
            source = sources.get(0)
            if source not in value_ports:
                raise GuppyConversionError(
                    f"HUGR constant loader {node} does not reference a static value."
                )
            value_ports[node.out(0)] = value_ports[source]
            continue
        if name in {"Input", "Output"}:
            # Input/Output nodes carry the function boundary and no stream event.
            if name == "Input":
                signature = getattr(op, "types", None)
                for offset, type_ in enumerate(signature or ()):
                    if "qubit" in str(type_).lower():
                        wire_sites[node.out(offset)] = n_qubits
                        n_qubits += 1
            continue
        if name in _CONTROL_FLOW_OPS or kind in _CONTROL_FLOW_OPS:
            raise GuppyConversionError(
                f"HUGR node {node} is {name!r}; branches/loops/calls cannot be "
                "flattened into one deterministic Pepsy gate stream."
            )
        if name in {"Read", "result_bool"}:
            if name == "Read":
                source = sources.get(0)
                if source not in measurement_ports:
                    raise GuppyConversionError(
                        "A Guppy measurement read is not connected to MeasureFree."
                    )
                measurement_ports[node.out(0)] = measurement_ports[source]
            else:
                source = sources.get(0)
                measurement = measurement_ports.get(source)
                if measurement is not None:
                    args = getattr(op, "args", ())
                    label = getattr(args[0], "value", None) if args else None
                    measurement.result = str(label) if label is not None else None
            continue

        if name == "QAlloc":
            if saw_nonallocation_quantum_operation:
                raise GuppyConversionError(
                    "QAlloc after another quantum operation is dynamic qubit "
                    "allocation and cannot be represented by a fixed Pepsy MPS."
                )
            wire_sites[node.out(0)] = n_qubits
            n_qubits += 1
            saw_quantum_operation = True
            continue

        if name in {"QFree", "MeasureFree", "Reset"} or name in {
            "H", "CX", "CY", "CZ", "SWAP", "T", "S", "V", "X", "Y", "Z",
            "Tdg", "Sdg", "Vdg", "Rx", "Ry", "Rz", "CRz", "Toffoli",
        }:
            saw_quantum_operation = True
            saw_nonallocation_quantum_operation = True

        if name == "QFree":
            if sources.get(0) not in wire_sites:
                raise GuppyConversionError(f"QFree at {node} references an unknown qubit.")
            continue

        if name == "MeasureFree":
            source = sources.get(0)
            if source not in wire_sites:
                raise GuppyConversionError(
                    f"MeasureFree at {node} references an unknown qubit."
                )
            measurement = GuppyMeasurement(wire_sites[source])
            measurements.append(measurement)
            measurement_ports[node.out(0)] = measurement
            stream.append(("measure", "Z", measurement.site))
            continue

        if name == "Reset":
            source = sources.get(0)
            if source not in wire_sites:
                raise GuppyConversionError(f"Reset at {node} references an unknown qubit.")
            site = wire_sites[source]
            stream.append(("reset", site, "Z"))
            wire_sites[node.out(0)] = site
            continue

        if name not in {
            "H", "CX", "CY", "CZ", "SWAP", "T", "S", "V", "X", "Y", "Z",
            "Tdg", "Sdg", "Vdg", "Rx", "Ry", "Rz", "CRz", "Toffoli",
        }:
            raise GuppyConversionError(
                f"Unsupported HUGR operation {name!r} at {node}. Decompose the "
                "Guppy circuit into supported tket.quantum operations first."
            )

        arity = {
            "H": 1, "T": 1, "S": 1, "V": 1, "X": 1, "Y": 1, "Z": 1,
            "Tdg": 1, "Sdg": 1, "Vdg": 1, "Rx": 1, "Ry": 1, "Rz": 1,
            "CX": 2, "CY": 2, "CZ": 2, "SWAP": 2, "CRz": 2, "Toffoli": 3,
        }[name]
        try:
            sites = tuple(wire_sites[sources[offset]] for offset in range(arity))
        except (KeyError, TypeError) as exc:
            raise GuppyConversionError(
                f"HUGR operation {name!r} at {node} has an unknown qubit wire."
            ) from exc

        theta = None
        if name in {"Rx", "Ry", "Rz", "CRz"}:
            angle_source = sources.get(arity)
            if angle_source not in value_ports:
                raise GuppyConversionError(
                    f"{name} at {node} has a dynamic angle; only static Guppy "
                    "angles can be converted to a gate stream."
                )
            theta = _value_half_turns(value_ports[angle_source])

        if normalized_format == "matrix":
            entry = (_matrix_gate(name, theta), sites)
        else:
            entry = _named_entry(name, sites, theta)
        stream.append(entry)
        for offset, site in enumerate(sites):
            wire_sites[node.out(offset)] = site

    if not saw_quantum_operation and n_qubits == 0:
        raise GuppyConversionError("The Guppy program contains no qubits.")

    return GuppyGateStream(
        stream,
        n_qubits=n_qubits,
        measurements=measurements,
        format=normalized_format,
    )
