"""Translate Stim syntax into reusable Pepsy trajectory plans.

Compilation owns instruction parsing and Clifford matrix caching. Replay,
random draws, and public record classes remain in ``noise``. Stim itself is
loaded only when a circuit needs compiling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .noise import StimCircuitPlan

_STIM_NOISE_NAMES = frozenset(
    {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "II_ERROR",
        "I_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    }
)


_STIM_IGNORED_NAMES = frozenset(
    {
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    }
)


_STIM_SINGLE_MEASUREMENTS = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
}


_STIM_SINGLE_MEASURE_RESETS = {
    "MR": "Z",
    "MRX": "X",
    "MRY": "Y",
}


_STIM_PAIR_MEASUREMENTS = {
    "MXX": "XX",
    "MYY": "YY",
    "MZZ": "ZZ",
}


_STIM_RESETS = {"R": "Z", "RX": "X", "RY": "Y"}


_STIM_UNITARY_CACHE: dict[str, np.ndarray] = {}


def _require_stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - only without optional stim
        raise ImportError(
            "Stim circuit noise support requires the optional 'stim' package. "
            "Install it with `python -m pip install stim`."
        ) from exc
    return stim


def _coerce_stim_circuit(circuit):
    stim = _require_stim()
    if isinstance(circuit, stim.Circuit):
        return circuit
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    raise TypeError("circuit must be a stim.Circuit, Stim source string, or StimCircuitPlan.")


def _stim_qubit_targets(instruction, *, allow_inverted_result=False) -> tuple[int, ...]:
    targets = []
    for target in instruction.targets_copy():
        if not target.is_qubit_target:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} has a non-qubit target that "
                "cannot be replayed as an MPS gate stream."
            )
        if target.is_inverted_result_target and not allow_inverted_result:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} has a classical-record "
                "controlled/inverted target that cannot be replayed as an MPS gate stream."
            )
        targets.append(int(target.value))
    return tuple(targets)


def _stim_pauli_targets(instruction) -> tuple[tuple[str, int], ...]:
    terms = []
    for target in instruction.targets_copy():
        if target.is_combiner:
            raise NotImplementedError(
                f"Stim noise instruction {instruction!s} unexpectedly contains a combiner."
            )
        if target.is_x_target:
            axis = "X"
        elif target.is_y_target:
            axis = "Y"
        elif target.is_z_target:
            axis = "Z"
        else:
            raise NotImplementedError(
                f"Stim noise instruction {instruction!s} has a non-Pauli target."
            )
        terms.append((axis, int(target.value)))
    return tuple(terms)


def _stim_record_targets(instruction) -> tuple[tuple[int, bool], ...]:
    """Normalize Stim ``rec[k]`` targets to ``(offset, inverted)`` pairs."""
    targets = []
    for target in instruction.targets_copy():
        if not target.is_measurement_record_target:
            raise NotImplementedError(
                f"Stim annotation {instruction!s} has a non-record target."
            )
        targets.append((int(target.value), bool(target.is_inverted_result_target)))
    return tuple(targets)


def _stim_unitary_matrix(name: str) -> np.ndarray:
    """Return a cached small Clifford matrix from Stim's public tableau API."""
    try:
        return _STIM_UNITARY_CACHE[name]
    except KeyError:
        stim = _require_stim()
        matrix = np.asarray(
            stim.Tableau.from_named_gate(name).to_unitary_matrix(endian="big"),
            dtype=np.complex128,
        )
        _STIM_UNITARY_CACHE[name] = matrix
        return matrix


def _compile_stim_measurement(instruction, name: str) -> tuple[object, ...]:
    # These change reported classical bits, not the measured physical state.
    # Until the stream has readout-error/inversion events, reject them rather
    # than silently deleting their effect on feed-forward and annotations.
    if any(instruction.gate_args_copy()):
        raise NotImplementedError(
            f"Stim measurement readout noise is not supported: {instruction!s}. "
            "Use explicit Pauli noise before an ideal measurement when that "
            "physical noise model is intended."
        )
    if any(target.is_inverted_result_target for target in instruction.targets_copy()):
        raise NotImplementedError(
            f"Inverted Stim measurement results are not supported: {instruction!s}."
        )
    if name in _STIM_SINGLE_MEASUREMENTS:
        axis = _STIM_SINGLE_MEASUREMENTS[name]
        return tuple(
            ("measure", axis, site)
            for site in _stim_qubit_targets(instruction, allow_inverted_result=True)
        )
    if name in _STIM_SINGLE_MEASURE_RESETS:
        axis = _STIM_SINGLE_MEASURE_RESETS[name]
        return tuple(
            ("measure_reset", axis, site)
            for site in _stim_qubit_targets(instruction, allow_inverted_result=True)
        )
    if name in _STIM_PAIR_MEASUREMENTS:
        targets = _stim_qubit_targets(instruction, allow_inverted_result=True)
        if len(targets) % 2:
            raise ValueError(f"Stim instruction {instruction!s} needs target pairs.")
        axis = _STIM_PAIR_MEASUREMENTS[name]
        return tuple(
            ("measure", axis, targets[offset : offset + 2])
            for offset in range(0, len(targets), 2)
        )
    if name != "MPP":
        raise NotImplementedError(f"Unsupported Stim measurement instruction {instruction!s}.")

    groups = []
    targets = instruction.targets_copy()
    offset = 0
    while offset < len(targets):
        axes = []
        sites = []
        while True:
            target = targets[offset]
            if target.is_x_target:
                axis = "X"
            elif target.is_y_target:
                axis = "Y"
            elif target.is_z_target:
                axis = "Z"
            else:
                raise ValueError(f"Malformed Stim MPP instruction {instruction!s}.")
            axes.append(axis)
            sites.append(int(target.value))
            offset += 1
            if offset == len(targets) or not targets[offset].is_combiner:
                break
            offset += 1
            if offset == len(targets):
                raise ValueError(f"Malformed Stim MPP instruction {instruction!s}.")
        groups.append(("measure", "".join(axes), tuple(sites)))
    return tuple(groups)


def _compile_stim_unitary(instruction, name: str) -> tuple[object, ...]:
    stim = _require_stim()
    if name in {"I", "II"}:
        return ()
    gate_data = stim.gate_data(name)
    if gate_data.is_single_qubit_gate:
        targets = _stim_qubit_targets(instruction)
        matrix = _stim_unitary_matrix(name)
        return tuple((matrix, site) for site in targets)
    if gate_data.is_two_qubit_gate:
        targets = _stim_qubit_targets(instruction)
        if len(targets) % 2:
            raise ValueError(f"Stim instruction {instruction!s} needs target pairs.")
        matrix = _stim_unitary_matrix(name)
        return tuple(
            (matrix, targets[offset : offset + 2])
            for offset in range(0, len(targets), 2)
        )
    raise NotImplementedError(
        f"Stim instruction {instruction!s} is not a one- or two-qubit unitary "
        "supported by both MPS optimizers."
    )


def _compile_stim_classical_control(instruction, name: str):
    """Lower grouped Stim measurement-record-controlled Pauli gates.

    The supported form is CX/CY/CZ rec[k] q, including an inverted record
    target. It becomes the backend-independent ("if", k, bit, action) event.
    More elaborate classical arithmetic remains intentionally outside this
    compiler so a record cannot be mistaken for a quantum wire.
    """
    targets = instruction.targets_copy()
    if len(targets) % 2:
        raise NotImplementedError(
            f"Stim instruction {instruction!s} needs control/target pairs."
        )
    if name not in {"CX", "CY", "CZ"}:
        raise NotImplementedError(
            f"Stim classical-record-controlled gate {name} is not supported."
        )
    axis = {"CX": "X", "CY": "Y", "CZ": "Z"}[name]
    entries = []
    for offset in range(0, len(targets), 2):
        pair = targets[offset:offset + 2]
        records = [t for t in pair if t.is_measurement_record_target]
        qubits = [t for t in pair if t.is_qubit_target]
        if not records and len(qubits) == 2:
            entries.append((_stim_unitary_matrix(name), tuple(int(t.value) for t in qubits)))
        elif len(records) == len(qubits) == 1:
            record = records[0]
            # An inverted record means 'apply when the recorded bit is 0'.
            expected_bit = 0 if record.is_inverted_result_target else 1
            entries.append((
                "if", int(record.value), expected_bit,
                (_stim_unitary_matrix(axis), int(qubits[0].value)),
            ))
        else:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} needs a qubit target for "
                "each measurement-record control."
            )
    return tuple(entries)


def compile_stim_circuit(circuit) -> StimCircuitPlan:
    """Compile a Stim circuit into reusable physical MPS stream operations.

    All native stochastic error instructions are retained for trajectory
    sampling. Clifford one- and two-qubit gates, single/product Pauli
    measurements, and Pauli-basis resets are also translated. Detector and
    observable annotations are retained in the plan and resolved after replay;
    coordinate and tick instructions have no quantum effect. Measurement-record-
    controlled Pauli gates ``CX/CY/CZ rec[k] q`` are lowered to explicit
    feed-forward events. General classical arithmetic and record-to-record
    operations remain rejected.
    """
    # Resolve record constructors at call time to keep module imports acyclic.
    # Their public definitions stay in noise, preserving serialized class paths.
    from .noise import (
        StimCircuitPlan, StimDetector, StimObservable, _StimPlanOperation,
    )

    if isinstance(circuit, StimCircuitPlan):
        return circuit
    stim_circuit = _coerce_stim_circuit(circuit)
    stim = _require_stim()
    operations = []
    detectors = []
    observables = []
    measurement_count = 0
    for instruction_index, instruction in enumerate(stim_circuit.flattened()):
        name = instruction.name.upper()
        args = tuple(float(value) for value in instruction.gate_args_copy())
        if name == "DETECTOR":
            detectors.append(
                StimDetector(
                    instruction_index=instruction_index,
                    detector_index=len(detectors),
                    rec_targets=_stim_record_targets(instruction),
                    coordinates=args,
                    measurement_count=measurement_count,
                )
            )
            continue
        if name == "OBSERVABLE_INCLUDE":
            if len(args) != 1 or int(args[0]) != args[0] or args[0] < 0:
                raise ValueError(
                    f"Stim OBSERVABLE_INCLUDE needs one nonnegative integer id: {instruction!s}."
                )
            observables.append(
                StimObservable(
                    instruction_index=instruction_index,
                    observable_index=int(args[0]),
                    rec_targets=_stim_record_targets(instruction),
                    measurement_count=measurement_count,
                )
            )
            continue
        if name in _STIM_NOISE_NAMES:
            targets = (
                _stim_pauli_targets(instruction)
                if name in {"E", "ELSE_CORRELATED_ERROR"}
                else tuple(("I", site) for site in _stim_qubit_targets(instruction))
            )
            operations.append(
                _StimPlanOperation(
                    instruction_index, name, args, targets, is_noise=True
                )
            )
            continue
        if name in _STIM_IGNORED_NAMES:
            continue
        if any(
            target.is_measurement_record_target
            for target in instruction.targets_copy()
        ):
            entries = _compile_stim_classical_control(instruction, name)
            operations.append(
                _StimPlanOperation(instruction_index, name, args, (), entries)
            )
            continue
        if name in _STIM_SINGLE_MEASUREMENTS or name in _STIM_SINGLE_MEASURE_RESETS:
            entries = _compile_stim_measurement(instruction, name)
        elif name in _STIM_PAIR_MEASUREMENTS or name == "MPP":
            entries = _compile_stim_measurement(instruction, name)
        elif name in _STIM_RESETS:
            entries = tuple(
                ("reset", site, _STIM_RESETS[name])
                for site in _stim_qubit_targets(instruction)
            )
        else:
            gate_data = stim.gate_data(name)
            if not gate_data.is_unitary:
                raise NotImplementedError(
                    f"Stim instruction {instruction!s} is not a supported quantum "
                    "operation for MPS replay."
                )
            entries = _compile_stim_unitary(instruction, name)
        if name in _STIM_SINGLE_MEASUREMENTS or name in _STIM_SINGLE_MEASURE_RESETS:
            measurement_count += len(entries)
        elif name in _STIM_PAIR_MEASUREMENTS or name == "MPP":
            measurement_count += len(entries)
        operations.append(_StimPlanOperation(instruction_index, name, args, (), entries))
    return StimCircuitPlan(
        int(stim_circuit.num_qubits),
        tuple(operations),
        tuple(detectors),
        tuple(observables),
    )


def stim_plan_to_gate_stream(plan) -> tuple[object, ...]:
    """Lower independent Stim Pauli channels without sampling any faults."""
    from .noise import StimCircuitPlan

    if not isinstance(plan, StimCircuitPlan):
        raise TypeError("plan must be a StimCircuitPlan from compile_stim_circuit(...).")
    single = {
        "X_ERROR": "x_error", "Y_ERROR": "y_error", "Z_ERROR": "z_error",
        "DEPOLARIZE1": "depolarize1", "PAULI_CHANNEL_1": "pauli_channel1",
    }
    double = {"DEPOLARIZE2": "depolarize2", "PAULI_CHANNEL_2": "pauli_channel2"}
    stream = []
    for operation in plan.operations:
        if not operation.is_noise:
            stream.extend(operation.entries)
            continue
        name = operation.name
        sites = tuple(site for _axis, site in operation.targets)
        if name in {"I_ERROR", "II_ERROR"}:
            continue
        if name in single:
            probabilities = operation.args if name == "PAULI_CHANNEL_1" else operation.args[0]
            stream.extend((single[name], probabilities, site) for site in sites)
        elif name in double:
            if len(sites) % 2:
                raise ValueError(f"Stim {name} needs target pairs.")
            probabilities = operation.args if name == "PAULI_CHANNEL_2" else operation.args[0]
            stream.extend(
                (double[name], probabilities, sites[index], sites[index + 1])
                for index in range(0, len(sites), 2)
            )
        else:
            raise NotImplementedError(
                f"Stim {name} at instruction {operation.instruction_index} cannot "
                "be lowered to independent gate-stream noise. Use run_stim_shots "
                "or run_coalesced_stim_shots for correlated/heralded channels."
            )
    return tuple(stream)


def stim_readout_parities(plan, measurement_records):
    """Resolve raw detector and indexed observable parities from shot records."""
    from .noise import StimCircuitPlan

    if not isinstance(plan, StimCircuitPlan):
        raise TypeError("plan must be a StimCircuitPlan from compile_stim_circuit(...).")
    bits = tuple(
        tuple(int(record.outcome < 0) for record in records
              if not getattr(record, "reset", False))
        for records in measurement_records
    )

    def resolve(annotations):
        values = np.zeros((len(bits), len(annotations)), dtype=np.uint8)
        for row, measurements in enumerate(bits):
            for column, annotation in enumerate(annotations):
                for offset, inverted in annotation.rec_targets:
                    index = annotation.measurement_count + offset
                    if not 0 <= index < len(measurements):
                        raise ValueError("Stim annotation references an unavailable measurement.")
                    values[row, column] ^= measurements[index] ^ int(inverted)
        return values

    detectors = resolve(plan.detectors)
    annotations = resolve(plan.observables)
    n_observables = max((a.observable_index for a in plan.observables), default=-1) + 1
    observables = np.zeros((len(bits), n_observables), dtype=np.uint8)
    for column, annotation in enumerate(plan.observables):
        observables[:, annotation.observable_index] ^= annotations[:, column]
    return detectors, observables
