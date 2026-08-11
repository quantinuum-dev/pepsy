"""Optional Guppy-to-Pepsy gate-stream integration checks."""

import math

import pytest


guppylang = pytest.importorskip("guppylang")

from guppylang import guppy  # noqa: E402
from guppylang.std.angles import angle  # noqa: E402
from guppylang.std.builtins import result  # noqa: E402
from guppylang.std.quantum import cx, measure, qubit, rz  # noqa: E402

import pepsy  # noqa: E402


@guppy
def _static_program() -> None:
    q0 = qubit()
    q1 = qubit()
    cx(q0, q1)
    rz(q1, angle(1 / 2))
    result("readout", measure(q0).read())
    q1.discard()


@pytest.mark.smoke
def test_guppy_program_becomes_list_compatible_stream():
    """A compiled Guppy stream can be consumed by stabilizer replay."""
    stream = pepsy.guppy_gate_stream(_static_program)

    assert isinstance(stream, list)
    assert stream.n_qubits == 2
    assert stream.measurements[0].site == 0
    assert stream.measurements[0].result == "readout"
    assert stream[0][1] == (0, 1)
    assert math.isclose(stream[1][0][0, 0].real, math.cos(math.pi / 4))

    result = pepsy.MpsStabOptimizer(stream.n_qubits, stream).run()
    assert len(result.measurements) == 1


@pytest.mark.smoke
def test_guppy_named_format_uses_pepsy_rotation_convention():
    """Guppy half-turn angles become Pepsy radian rotation entries."""
    stream = pepsy.guppy_gate_stream(_static_program, format="named")

    assert stream[0] == ("cx", 0, 1)
    assert stream[1] == ("rz", math.pi / 2, 1)
