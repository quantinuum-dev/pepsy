"""Gate-application utilities for 1D/2D/3D tensor networks."""

from __future__ import annotations

from heapq import heappop, heappush
from numbers import Integral
import random
import re
from string import Formatter
import warnings
from itertools import count

import autoray as ar
import numpy as np
import quimb as qu
import quimb.tensor as qtn

from ..backends.convert import (
    infer_backend_converter_from_sample,
    resolve_backend_sample_data_from_tn,
)
from ..tensors.core import add_cycle, id_to_mpo, id_to_pepo

__all__ = [
    "gate",
    "gate_loop_cluster",
    "gate_simple",
    "renorm_gauge",
    "build_pepo_from_gates",
    "build_mpo_from_gates",
    "pauli",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "h",
    "hadamard",
    "cnot",
    "cx",
    "cy",
    "cz",
    "swap",
    "iswap",
    "phase",
    "u1",
    "u2",
    "cphase",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cu2",
    "cu3",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "u3",
    "su4",
    "fsim",
    "fsimg",
]


def _stop_gradient(x):
    """Best-effort backend-agnostic stop-gradient helper."""
    try:
        return ar.do("stop_gradient", x)
    except Exception:
        return x


def rx(theta):
    """Return a one-qubit RX gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rx_gate_param_gen([theta])


def ry(theta):
    """Return a one-qubit RY gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.ry_gate_param_gen([theta])


def rz(theta):
    """Return a one-qubit RZ gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rz_gate_param_gen([theta])


def pauli(which, dtype=None):
    """Return a one-qubit Pauli matrix by label, e.g. ``'X'`` or ``'Z'``."""
    label = str(which).upper()
    if dtype is None:
        return qu.pauli(label)
    return qu.pauli(label, dtype=dtype)


def x():
    """Return the one-qubit Pauli-X gate."""
    return pauli("X")


def y():
    """Return the one-qubit Pauli-Y gate."""
    return pauli("Y")


def z():
    """Return the one-qubit Pauli-Z gate."""
    return pauli("Z")


def s():
    """Return the one-qubit S gate."""
    return qu.S_gate()


def sdg():
    """Return the one-qubit S-dagger gate."""
    return s().H


def t():
    """Return the one-qubit T gate."""
    return qu.T_gate()


def tdg():
    """Return the one-qubit T-dagger gate."""
    return t().H


def hadamard():
    """Return the one-qubit Hadamard gate."""
    return qu.hadamard()


def h():
    """Alias for :func:`hadamard`."""
    return hadamard()


def cnot():
    """Return the two-qubit controlled-X (CNOT) gate."""
    return qu.CNOT()


def cx():
    """Alias for :func:`cnot`."""
    return cnot()


def cy():
    """Return the two-qubit controlled-Y gate."""
    return qu.cY()


def cz():
    """Return the two-qubit controlled-Z gate."""
    return qu.cZ()


def swap():
    """Return the two-qubit SWAP gate."""
    return qu.swap()


def iswap():
    """Return the two-qubit iSWAP gate."""
    return qu.iswap()


def phase(theta):
    """Alias for :func:`u1`."""
    return u1(theta)


def u1(theta):
    """Return a one-qubit U1 gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.u1_gate_param_gen([theta])


def u2(params):
    """Return a one-qubit U2 gate from 2 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 2 parameters.
    """
    if len(params) != 2:
        raise ValueError("u2 expects exactly 2 parameters.")
    return qtn.circuit.u2_gate_param_gen(params)


def cphase(theta):
    """Return a two-qubit controlled-phase gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.cu1_param_gen([theta])


def cu1(theta):
    """Alias for :func:`cphase`."""
    return cphase(theta)


def crx(theta):
    """Return a two-qubit controlled-RX gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.crx_param_gen([theta])


def cry(theta):
    """Return a two-qubit controlled-RY gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.cry_param_gen([theta])


def crz(theta):
    """Return a two-qubit controlled-RZ gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.crz_param_gen([theta])


def cu2(params):
    """Return a two-qubit controlled-U2 gate from 2 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 2 parameters.
    """
    if len(params) != 2:
        raise ValueError("cu2 expects exactly 2 parameters.")
    return qtn.circuit.cu2_param_gen(params)


def cu3(params):
    """Return a two-qubit controlled-U3 gate from 3 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 3 parameters.
    """
    if len(params) != 3:
        raise ValueError("cu3 expects exactly 3 parameters.")
    return qtn.circuit.cu3_param_gen(params)


def rzz(theta):
    """Return a two-qubit RZZ gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rzz_param_gen([theta])


def rxx(theta):
    """Return a two-qubit RXX gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.rxx_param_gen([theta])


def ryy(theta):
    """Return a two-qubit RYY gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.ryy_param_gen([theta])


def su4(params):
    """Return a two-qubit SU(4) gate from 15 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 15 parameters.
    """
    if len(params) != 15:
        raise ValueError("su4 expects exactly 15 parameters.")
    return qtn.circuit.su4_gate_param_gen(params)


def u3(params):
    """Return a one-qubit U3 gate from 3 parameters.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 3 parameters.
    """
    if len(params) != 3:
        raise ValueError("u3 expects exactly 3 parameters.")
    return qtn.circuit.u3_gate_param_gen(params)


def fsim(params):
    """Return a two-qubit fSim gate from 2 parameters.

    The fSim gate is defined as::

        [[1,           0,           0, 0          ],
         [0,  cos(theta), -i*sin(theta), 0          ],
         [0, -i*sin(theta),  cos(theta), 0          ],
         [0,           0,           0, exp(-i*phi)]]

    Parameters
    ----------
    params : sequence
        Sequence of exactly 2 parameters ``(theta, phi)``.
    """
    if len(params) != 2:
        raise ValueError("fsim expects exactly 2 parameters (theta, phi).")
    return qtn.circuit.fsim_param_gen(params)


def fsimg(params):
    """Return a two-qubit generalized fSim gate from 5 parameters.

    The most general number-conserving two-qubit gate parametrized by
    ``(theta, zeta, chi, gamma, phi)``.

    Parameters
    ----------
    params : sequence
        Sequence of exactly 5 parameters ``(theta, zeta, chi, gamma, phi)``.
    """
    if len(params) != 5:
        raise ValueError("fsimg expects exactly 5 parameters (theta, zeta, chi, gamma, phi).")
    return qtn.circuit.fsimg_param_gen(params)


def gen_long_range_swap_path_2d(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    ij_a, ij_b, sequence=None, *, cyclic=False, Lx=None, Ly=None
):
    """Generate a SWAP path that brings two lattice sites together."""
    ia, ja = ij_a
    ib, jb = ij_b

    if cyclic:
        if Lx is None or Ly is None:
            raise ValueError("When cyclic=True, Lx and Ly must be provided.")
        Lx = int(Lx)
        Ly = int(Ly)
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Lx and Ly must be positive integers when cyclic=True.")

    def _wrapped_delta(delta, size):
        if not cyclic:
            return delta
        wrapped = delta % size
        half = size / 2
        if wrapped > half:
            wrapped -= size
        elif (size % 2 == 0) and (wrapped == half) and (delta < 0):
            wrapped -= size
        return int(wrapped)

    di = _wrapped_delta(ib - ia, Lx)
    dj = _wrapped_delta(jb - ja, Ly)

    if (di == 0) and (dj == 0):
        return

    if abs(di) + abs(dj) == 1:
        yield (ij_a, ij_b)
        return

    allowed_moves = {"av", "bv", "ah", "bh"}
    if isinstance(sequence, str) and sequence in {"auto", "smart", "min_bond"}:
        sequence = None

    if isinstance(sequence, str) and sequence not in {
        "random",
        "x_then_y",
        "xy",
        "y_then_x",
        "yx",
    }:
        warnings.warn(
            f"Unknown string sequence='{sequence}'. Falling back to default cycle.",
            RuntimeWarning,
            stacklevel=2,
        )
        sequence = None
    elif (
        (sequence is not None)
        and (sequence != "random")
        and not (isinstance(sequence, str) and (sequence in {"x_then_y", "xy", "y_then_x", "yx"}))
    ):
        sequence = tuple(sequence)
        invalid_moves = tuple(move for move in sequence if move not in allowed_moves)
        if invalid_moves:
            warnings.warn(
                f"Ignoring invalid move tokens in sequence: {invalid_moves}.",
                RuntimeWarning,
                stacklevel=2,
            )
            sequence = tuple(move for move in sequence if move in allowed_moves)
        if not sequence:
            sequence = None

    def _wrap_x(i):
        return i % Lx if cyclic else i

    def _wrap_y(j):
        return j % Ly if cyclic else j

    def apply_move(move):
        nonlocal ij_a, ij_b, ia, ja, ib, jb, di, dj
        if (move == "av") and (di != 0):
            istep = min(max(di, -1), +1)
            new_ij_a = (_wrap_x(ia + istep), ja)
            pair = (ij_a, new_ij_a)
            ij_a = new_ij_a
            ia = new_ij_a[0]
            di -= istep
            return pair

        if (move == "bv") and (di != 0):
            istep = min(max(di, -1), +1)
            new_ij_b = (_wrap_x(ib - istep), jb)
            pair = (ij_a, ij_b) if (new_ij_b == ij_a) else (ij_b, new_ij_b)
            ij_b = new_ij_b
            ib = new_ij_b[0]
            di -= istep
            return pair

        if (move == "ah") and (dj != 0):
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, _wrap_y(ja + jstep))
            pair = (ij_a, new_ij_a)
            ij_a = new_ij_a
            ja = new_ij_a[1]
            dj -= jstep
            return pair

        if (move == "bh") and (dj != 0):
            jstep = min(max(dj, -1), +1)
            new_ij_b = (ib, _wrap_y(jb - jstep))
            pair = (ij_a, ij_b) if (new_ij_b == ij_a) else (ij_b, new_ij_b)
            ij_b = new_ij_b
            jb = new_ij_b[1]
            dj -= jstep
            return pair

        return None

    if sequence in {"x_then_y", "xy", "y_then_x", "yx"}:
        axis_order = ("x", "y") if sequence in {"x_then_y", "xy"} else ("y", "x")

        for axis in axis_order:
            if axis == "x":
                while di != 0:
                    istep = min(max(di, -1), +1)
                    new_ij_a = (_wrap_x(ia + istep), ja)
                    yield (ij_a, new_ij_a)
                    ij_a = new_ij_a
                    ia = new_ij_a[0]
                    di -= istep
            else:
                while dj != 0:
                    jstep = min(max(dj, -1), +1)
                    new_ij_a = (ia, _wrap_y(ja + jstep))
                    yield (ij_a, new_ij_a)
                    ij_a = new_ij_a
                    ja = new_ij_a[1]
                    dj -= jstep
        return

    if sequence is None:
        move_order = ("av", "bv", "ah", "bh")
    elif sequence == "random":
        move_order = ("av", "bv", "ah", "bh")
    else:
        move_order = sequence

    if sequence == "random":
        poss_moves = (random.choice(move_order) for _ in count())
        for move in poss_moves:
            pair = apply_move(move)
            if pair is not None:
                yield pair
            if di == dj == 0:
                return
    else:
        while True:
            progress_made = False
            for move in move_order:
                pair = apply_move(move)
                if pair is not None:
                    progress_made = True
                    yield pair
                if di == dj == 0:
                    return
            if not progress_made:
                raise ValueError(
                    "Stalled swap-path generation: sequence cannot reduce current site separation."
                )


def gen_long_range_swap_path_1d(x, y):
    """Generate adjacent 1D pairs to route a long-range two-site gate."""
    x = int(x)
    y = int(y)

    if x == y:
        raise ValueError("Two-site gate requires distinct site indices.")

    if abs(y - x) == 1:
        yield (x, y)
        return

    step = 1 if (y > x) else -1
    current = x
    while abs(y - current) > 1:
        nxt = current + step
        yield (current, nxt)
        current = nxt

    yield (current, y)


def gen_long_range_swap_path_3d(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    ijk_a, ijk_b, sequence=None, *, cyclic=False, Lx=None, Ly=None, Lz=None
):
    """Generate a SWAP path that brings two 3D lattice sites together."""
    ia, ja, ka = ijk_a
    ib, jb, kb = ijk_b

    if cyclic:
        if (Lx is None) or (Ly is None) or (Lz is None):
            raise ValueError("When cyclic=True, Lx, Ly, and Lz must be provided.")
        Lx = int(Lx)
        Ly = int(Ly)
        Lz = int(Lz)
        if (Lx <= 0) or (Ly <= 0) or (Lz <= 0):
            raise ValueError("Lx, Ly, and Lz must be positive integers when cyclic=True.")

    def _wrapped_delta(delta, size):
        if not cyclic:
            return delta
        wrapped = delta % size
        half = size / 2
        if wrapped > half:
            wrapped -= size
        elif (size % 2 == 0) and (wrapped == half) and (delta < 0):
            wrapped -= size
        return int(wrapped)

    dx = _wrapped_delta(ib - ia, Lx)
    dy = _wrapped_delta(jb - ja, Ly)
    dz = _wrapped_delta(kb - ka, Lz)

    if (dx == 0) and (dy == 0) and (dz == 0):
        return

    if abs(dx) + abs(dy) + abs(dz) == 1:
        yield (ijk_a, ijk_b)
        return

    allowed_moves = {"ax", "bx", "ay", "by", "az", "bz"}
    named_axis_orders = {
        "x_then_y_then_z": ("x", "y", "z"),
        "xyz": ("x", "y", "z"),
        "x_then_z_then_y": ("x", "z", "y"),
        "xzy": ("x", "z", "y"),
        "y_then_x_then_z": ("y", "x", "z"),
        "yxz": ("y", "x", "z"),
        "y_then_z_then_x": ("y", "z", "x"),
        "yzx": ("y", "z", "x"),
        "z_then_x_then_y": ("z", "x", "y"),
        "zxy": ("z", "x", "y"),
        "z_then_y_then_x": ("z", "y", "x"),
        "zyx": ("z", "y", "x"),
    }

    if isinstance(sequence, str) and sequence in {"auto", "smart", "min_bond"}:
        sequence = None

    if isinstance(sequence, str) and (sequence not in {"random"} | set(named_axis_orders)):
        warnings.warn(
            f"Unknown string sequence='{sequence}'. Falling back to default cycle.",
            RuntimeWarning,
            stacklevel=2,
        )
        sequence = None
    elif (
        (sequence is not None)
        and (sequence != "random")
        and not (isinstance(sequence, str) and (sequence in named_axis_orders))
    ):
        sequence = tuple(sequence)
        invalid_moves = tuple(move for move in sequence if move not in allowed_moves)
        if invalid_moves:
            warnings.warn(
                f"Ignoring invalid move tokens in sequence: {invalid_moves}.",
                RuntimeWarning,
                stacklevel=2,
            )
            sequence = tuple(move for move in sequence if move in allowed_moves)
        if not sequence:
            sequence = None

    def _wrap_x(i):
        return i % Lx if cyclic else i

    def _wrap_y(j):
        return j % Ly if cyclic else j

    def _wrap_z(k):
        return k % Lz if cyclic else k

    def apply_move(move):
        nonlocal ijk_a, ijk_b, ia, ja, ka, ib, jb, kb, dx, dy, dz

        if (move == "ax") and (dx != 0):
            istep = min(max(dx, -1), +1)
            new_a = (_wrap_x(ia + istep), ja, ka)
            pair = (ijk_a, new_a)
            ijk_a = new_a
            ia = new_a[0]
            dx -= istep
            return pair

        if (move == "bx") and (dx != 0):
            istep = min(max(dx, -1), +1)
            new_b = (_wrap_x(ib - istep), jb, kb)
            pair = (ijk_a, ijk_b) if (new_b == ijk_a) else (ijk_b, new_b)
            ijk_b = new_b
            ib = new_b[0]
            dx -= istep
            return pair

        if (move == "ay") and (dy != 0):
            jstep = min(max(dy, -1), +1)
            new_a = (ia, _wrap_y(ja + jstep), ka)
            pair = (ijk_a, new_a)
            ijk_a = new_a
            ja = new_a[1]
            dy -= jstep
            return pair

        if (move == "by") and (dy != 0):
            jstep = min(max(dy, -1), +1)
            new_b = (ib, _wrap_y(jb - jstep), kb)
            pair = (ijk_a, ijk_b) if (new_b == ijk_a) else (ijk_b, new_b)
            ijk_b = new_b
            jb = new_b[1]
            dy -= jstep
            return pair

        if (move == "az") and (dz != 0):
            kstep = min(max(dz, -1), +1)
            new_a = (ia, ja, _wrap_z(ka + kstep))
            pair = (ijk_a, new_a)
            ijk_a = new_a
            ka = new_a[2]
            dz -= kstep
            return pair

        if (move == "bz") and (dz != 0):
            kstep = min(max(dz, -1), +1)
            new_b = (ib, jb, _wrap_z(kb - kstep))
            pair = (ijk_a, ijk_b) if (new_b == ijk_a) else (ijk_b, new_b)
            ijk_b = new_b
            kb = new_b[2]
            dz -= kstep
            return pair

        return None

    if isinstance(sequence, str) and (sequence in named_axis_orders):
        for axis in named_axis_orders[sequence]:
            if axis == "x":
                while dx != 0:
                    istep = min(max(dx, -1), +1)
                    new_a = (_wrap_x(ia + istep), ja, ka)
                    yield (ijk_a, new_a)
                    ijk_a = new_a
                    ia = new_a[0]
                    dx -= istep
            elif axis == "y":
                while dy != 0:
                    jstep = min(max(dy, -1), +1)
                    new_a = (ia, _wrap_y(ja + jstep), ka)
                    yield (ijk_a, new_a)
                    ijk_a = new_a
                    ja = new_a[1]
                    dy -= jstep
            elif axis == "z":
                while dz != 0:
                    kstep = min(max(dz, -1), +1)
                    new_a = (ia, ja, _wrap_z(ka + kstep))
                    yield (ijk_a, new_a)
                    ijk_a = new_a
                    ka = new_a[2]
                    dz -= kstep
        return

    if sequence is None:
        move_order = ("ax", "bx", "ay", "by", "az", "bz")
    elif sequence == "random":
        move_order = ("ax", "bx", "ay", "by", "az", "bz")
    else:
        move_order = sequence

    if sequence == "random":
        poss_moves = (random.choice(move_order) for _ in count())
        for move in poss_moves:
            pair = apply_move(move)
            if pair is not None:
                yield pair
            if dx == dy == dz == 0:
                return
    else:
        while True:
            progress_made = False
            for move in move_order:
                pair = apply_move(move)
                if pair is not None:
                    progress_made = True
                    yield pair
                if dx == dy == dz == 0:
                    return
            if not progress_made:
                raise ValueError(
                    "Stalled swap-path generation: sequence cannot reduce current site separation."
                )


def _is_auto_sequence(sequence):
    return isinstance(sequence, str) and (sequence.lower() in {"auto", "smart", "min_bond"})


def _wrapped_axis_delta(delta, size):
    """Return the shortest signed wrapped displacement along one axis."""
    if size is None:
        return int(delta)
    size = int(size)
    wrapped = int(delta) % size
    half = size / 2
    if wrapped > half:
        wrapped -= size
    elif (size % 2 == 0) and (wrapped == half) and (delta < 0):
        wrapped -= size
    return int(wrapped)


def _site_tensor_for_coord(tn, coord):
    """Return the tensor carrying a lattice site tag, if available."""
    site_tag = getattr(tn, "site_tag", None)
    if not callable(site_tag):
        return None
    try:
        return tn[site_tag(coord)]
    except (KeyError, TypeError, AttributeError):
        return None


def _tensor_index_size(tensor, ix):
    ind_size = getattr(tensor, "ind_size", None)
    if callable(ind_size):
        return int(ind_size(ix))

    inds = getattr(tensor, "inds", ())
    shape = getattr(tensor, "shape", None)
    if shape is not None and ix in inds:
        return int(shape[inds.index(ix)])

    data_shape = getattr(getattr(tensor, "data", None), "shape", None)
    if data_shape is not None and ix in inds:
        return int(data_shape[inds.index(ix)])

    return 1


def _bond_dimension_between_sites(tn, coord_a, coord_b):
    """Estimate the virtual bond dimension between two adjacent lattice sites."""
    tensor_a = _site_tensor_for_coord(tn, coord_a)
    tensor_b = _site_tensor_for_coord(tn, coord_b)
    if tensor_a is None or tensor_b is None:
        return 1

    bonds = qtn.bonds(tensor_a, tensor_b)
    if not bonds:
        return 10**12
    return max(_tensor_index_size(tensor_a, ix) for ix in bonds)


def _smart_long_range_swap_path(tn, start, target, *, cyclic=False, sizes=None):
    """Choose a shortest SWAP path with the smallest current bond bottleneck."""
    start = tuple(int(x) for x in start)
    target = tuple(int(x) for x in target)
    ndim = len(start)
    sizes = tuple(sizes or (None,) * ndim)

    if start == target:
        return []

    axis_delta_cache = {}
    distance_cache = {}
    bond_dim_cache = {}

    def _axis_delta(coord, axis):
        key = (coord, axis)
        if key not in axis_delta_cache:
            axis_delta_cache[key] = _wrapped_axis_delta(
                target[axis] - coord[axis],
                sizes[axis] if cyclic else None,
            )
        return axis_delta_cache[key]

    def _coord_distance(coord):
        if coord not in distance_cache:
            distance_cache[coord] = sum(
                abs(_axis_delta(coord, axis)) for axis in range(ndim)
            )
        return distance_cache[coord]

    def _edge_bond_dim(coord_a, coord_b):
        key = frozenset((coord_a, coord_b))
        if key not in bond_dim_cache:
            bond_dim_cache[key] = _bond_dimension_between_sites(
                tn, coord_a, coord_b
            )
        return bond_dim_cache[key]

    def _step_coord(coord, axis, step):
        coord = list(coord)
        coord[axis] += step
        if cyclic:
            coord[axis] %= int(sizes[axis])
        return tuple(coord)

    def _reconstruct_path(coord, parents):
        path = []
        while coord != start:
            prev, pair = parents[coord]
            path.append(pair)
            coord = prev
        path.reverse()
        return path

    queue = []
    counter = count()
    start_score = (0, 0, 0)
    heappush(queue, (*start_score, next(counter), start))
    best = {start: start_score}
    parents = {start: None}

    while queue:
        steps, bottleneck, total, _, coord = heappop(queue)
        if best.get(coord) != (steps, bottleneck, total):
            continue
        if coord == target:
            return _reconstruct_path(coord, parents)

        coord_distance = _coord_distance(coord)
        for axis in range(ndim):
            delta = _axis_delta(coord, axis)
            if delta == 0:
                continue
            step = 1 if delta > 0 else -1
            nxt = _step_coord(coord, axis, step)

            if _coord_distance(nxt) >= coord_distance:
                continue

            bond_dim = _edge_bond_dim(coord, nxt)
            new_score = (steps + 1, max(bottleneck, bond_dim), total + bond_dim)
            if new_score < best.get(nxt, (10**18, 10**18, 10**18)):
                best[nxt] = new_score
                parents[nxt] = (coord, (coord, nxt))
                heappush(queue, (*new_score, next(counter), nxt))

    return []


def _select_long_range_swap_path_2d(tn, start, target, *, sequence=None, cyclic=False, Lx=None, Ly=None):
    if _is_auto_sequence(sequence) and (not cyclic or (Lx is not None and Ly is not None)):
        path = _smart_long_range_swap_path(
            tn, start, target, cyclic=cyclic, sizes=(Lx, Ly)
        )
        if path:
            return path
    return list(
        gen_long_range_swap_path_2d(
            start,
            target,
            sequence=sequence,
            cyclic=cyclic,
            Lx=Lx,
            Ly=Ly,
        )
    )


def _select_long_range_swap_path_3d(
    tn, start, target, *, sequence=None, cyclic=False, Lx=None, Ly=None, Lz=None
):
    if _is_auto_sequence(sequence) and (
        not cyclic or (Lx is not None and Ly is not None and Lz is not None)
    ):
        path = _smart_long_range_swap_path(
            tn, start, target, cyclic=cyclic, sizes=(Lx, Ly, Lz)
        )
        if path:
            return path
    return list(
        gen_long_range_swap_path_3d(
            start,
            target,
            sequence=sequence,
            cyclic=cyclic,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
        )
    )


def _path_site_tags(tn, path_pairs):
    """Return unique lattice site tags touched by a path."""
    site_tag = getattr(tn, "site_tag", None)
    if not callable(site_tag):
        return []

    tags = []
    seen = set()
    for pair in path_pairs:
        for site in pair:
            try:
                tag = site_tag(site)
            except (TypeError, ValueError):
                continue
            if tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def _unique_path_pairs(path_pairs):
    """Return unique undirected path pairs, preserving first-seen order."""
    pairs = []
    seen = set()
    for a, b in path_pairs:
        key = frozenset((a, b))
        if key in seen:
            continue
        pairs.append((a, b))
        seen.add(key)
    return pairs


def _maybe_canonize_path(
    tn,
    path_pairs,
    *,
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
):
    """Canonize once around the full path touched by a routed gate."""
    if not path_canonize:
        return

    canonize_around = getattr(tn, "canonize_around_", None)
    if not callable(canonize_around):
        return

    tags = _path_site_tags(tn, path_pairs)
    if not tags:
        return

    max_distance = path_canonize_distance
    if isinstance(path_canonize, Integral) and not isinstance(path_canonize, bool):
        max_distance = int(path_canonize)

    opts = dict(path_canonize_opts or {})
    opts.setdefault("which", "any")
    opts.setdefault("max_distance", max_distance)
    canonize_around(tags, **opts)


def _maybe_compress_path(
    tn,
    path_pairs,
    *,
    path_compress=False,
    max_bond=None,
    cutoff=1.0e-12,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
):
    """Optionally clean up only the bonds traversed by a routed gate path."""
    if not path_compress:
        return

    compress_between = getattr(tn, "compress_between", None)
    site_tag = getattr(tn, "site_tag", None)
    if not callable(compress_between) or not callable(site_tag):
        return

    opts = dict(path_compress_opts or {})
    opts.setdefault("canonize_distance", path_compress_canonize_distance)

    for site_a, site_b in _unique_path_pairs(path_pairs):
        try:
            compress_between(
                site_tag(site_a),
                site_tag(site_b),
                max_bond=max_bond,
                cutoff=cutoff,
                **opts,
            )
        except (KeyError, TypeError, ValueError):
            continue


def _local_split_opts(max_bond):
    """Return split/compression kwargs for quimb gate calls."""
    if max_bond is None:
        return {}
    return {"max_bond": int(max_bond)}


def _normalize_where_arg_1d(where):
    """Normalize one-site and two-site where specs for 1D gate application."""
    if isinstance(where, Integral):
        return (int(where),)

    if isinstance(where, (tuple, list)) and where and all(
        isinstance(v, Integral) for v in where
    ):
        if len(where) not in (1, 2):
            raise ValueError("1D where must contain one or two integer site indices.")
        return tuple(int(v) for v in where)

    raise ValueError("Invalid 1D where specification.")


def _is_explicit_index_where(where):
    """Return True when ``where`` is already explicit TN index name(s)."""
    return isinstance(where, str) or (
        isinstance(where, (tuple, list))
        and where
        and all(isinstance(site, str) for site in where)
    )


def _ind_id_arity(ind_id):
    """Return how many replacement fields appear in ``ind_id``."""
    return sum(1 for _, field_name, _, _ in Formatter().parse(ind_id) if field_name is not None)


def _format_ind_id(ind_id, site):
    """Format one index name from ``ind_id`` and site coordinate(s)."""
    site_arity = len(site) if isinstance(site, (tuple, list)) else 1
    id_arity = _ind_id_arity(ind_id)
    if id_arity != site_arity:
        if isinstance(site, (tuple, list)):
            site_str = "(" + ", ".join(str(int(v)) for v in site) + ")"
        else:
            site_str = str(int(site))
        raise ValueError(
            f"ind_id={ind_id!r} is incompatible with site coordinate {site_str}. "
            "Use one placeholder for 1D (e.g. 'b{}'), two for 2D (e.g. 'b{},{}'), "
            "and three for 3D (e.g. 'b{},{},{}')."
        )

    try:
        if isinstance(site, (tuple, list)):
            return ind_id.format(*[int(v) for v in site])
        return ind_id.format(int(site))
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        if isinstance(site, (tuple, list)):
            site_str = "(" + ", ".join(str(int(v)) for v in site) + ")"
        else:
            site_str = str(int(site))
        raise ValueError(
            f"ind_id={ind_id!r} is incompatible with site coordinate {site_str}. "
            "Use one placeholder for 1D (e.g. 'b{}'), two for 2D (e.g. 'b{},{}'), "
            "and three for 3D (e.g. 'b{},{},{}')."
        ) from exc


def _physical_index_for_site(tn, site, ind_id=None):
    """Return the current physical index name for a lattice site."""
    if ind_id is not None:
        return _format_ind_id(ind_id, site)

    site_ind = getattr(tn, "site_ind", None)
    if callable(site_ind):
        try:
            if isinstance(site, (tuple, list)):
                return site_ind(*site)
            return site_ind(site)
        except TypeError:
            return site_ind(site)

    if isinstance(site, (tuple, list)):
        if len(site) == 2:
            return _format_ind_id("k{},{}", site)
        if len(site) == 3:
            return _format_ind_id("k{},{},{}", site)
    elif isinstance(site, Integral):
        return _format_ind_id("k{}", site)

    raise ValueError(
        "Cannot infer physical index for routed SWAP. Pass ind_id explicitly "
        "or use a tensor network with a site_ind method."
    )


def _physical_index_size_for_site(tn, site, ind_id=None):
    """Return the live physical-index dimension for one lattice site."""
    ix = _physical_index_for_site(tn, site, ind_id)

    ind_size = getattr(tn, "ind_size", None)
    if callable(ind_size):
        try:
            return int(ind_size(ix))
        except (KeyError, TypeError, ValueError):
            pass

    tensor = _site_tensor_for_coord(tn, site)
    if tensor is not None and ix in getattr(tensor, "inds", ()):
        return _tensor_index_size(tensor, ix)

    # Keep compatibility with generic or mocked TNs that cannot report physical
    # index sizes. This matches the old routed-SWAP assumption while real PEPS
    # and MPS objects take the dimension-aware path above.
    return 2


def _rectangular_swap_gate(dim_a, dim_b, *, dtype="complex128"):
    """Build the exact SWAP from d_a x d_b to d_b x d_a."""
    dim_a = int(dim_a)
    dim_b = int(dim_b)
    if dim_a <= 0 or dim_b <= 0:
        raise ValueError("SWAP dimensions must be positive integers.")

    swap_gate = np.zeros((dim_b, dim_a, dim_a, dim_b), dtype=dtype)
    for ia in range(dim_a):
        for ib in range(dim_b):
            swap_gate[ib, ia, ia, ib] = 1
    return swap_gate


def _is_symmray_array(value):
    """Return whether ``value`` looks like a Symmray block-sparse array."""
    return hasattr(value, "blocks") and hasattr(value, "indices")


def _symmray_sample_data_from_tn(tn):
    """Return representative Symmray array data from a tensor network."""
    tensor_map = getattr(tn, "tensor_map", None)
    if tensor_map:
        tensors = tensor_map.values()
    else:
        try:
            tensors = tuple(tn)
        except TypeError:
            tensors = ()

    for tensor in tensors:
        data = getattr(tensor, "data", None)
        if _is_symmray_array(data):
            return data
    return None


def _symmray_index_map_for_tn_ind(tn, ind):
    """Return the Symmray charge map for a live tensor-network index."""
    tensor_ids = getattr(tn, "ind_map", {}).get(ind, ())
    tensors = []
    tensor_map = getattr(tn, "tensor_map", {})
    for tid in tensor_ids:
        try:
            tensors.append(tensor_map[tid])
        except KeyError:
            continue
    if not tensors:
        try:
            tensors = tuple(tn)
        except TypeError:
            tensors = ()

    for tensor in tensors:
        data = getattr(tensor, "data", None)
        if not _is_symmray_array(data) or ind not in getattr(tensor, "inds", ()):
            continue
        axis = tensor.inds.index(ind)
        return dict(data.indices[axis].chargemap)

    raise ValueError(f"Could not infer Symmray charge map for index {ind!r}.")


def _symmray_dense_index_map_from_chargemap(chargemap):
    """Expand ``{charge: size}`` into Symmray's ``{dense_index: charge}``."""
    index_map = {}
    dense_index = 0
    for charge, size in dict(chargemap).items():
        for _ in range(int(size)):
            index_map[dense_index] = charge
            dense_index += 1
    return index_map


def _symmray_dense_gate_from_site_maps(tn, gate, output_sites, input_sites, ind_id):
    """Convert a dense site gate to a Symmray array using live site sectors."""
    sample = _symmray_sample_data_from_tn(tn)
    if sample is None:
        return None

    index_maps = []
    for site in tuple(output_sites) + tuple(input_sites):
        ind = _physical_index_for_site(tn, site, ind_id)
        index_maps.append(
            _symmray_dense_index_map_from_chargemap(
                _symmray_index_map_for_tn_ind(tn, ind)
            )
        )

    nout = len(tuple(output_sites))
    nin = len(tuple(input_sites))
    array_cls = type(sample)
    kwargs = {}
    if array_cls.__name__ in {"AbelianArray", "FermionicArray"}:
        symmetry = getattr(sample, "symmetry", None)
        if symmetry is not None:
            kwargs["symmetry"] = symmetry

    return array_cls.from_dense(
        np.asarray(gate),
        index_maps=tuple(index_maps),
        duals=(False,) * nout + (True,) * nin,
        charge=0,
        **kwargs,
    )


def _symmray_swap_gate_for_site_pair(tn, site_a, site_b, *, ind_id=None, dtype="complex128"):
    """Return a Symmray block-sparse SWAP for two live physical site legs."""
    sample = _symmray_sample_data_from_tn(tn)
    if sample is None:
        return None

    dim_a = _physical_index_size_for_site(tn, site_a, ind_id)
    dim_b = _physical_index_size_for_site(tn, site_b, ind_id)
    dense_swap = _rectangular_swap_gate(dim_a, dim_b, dtype=dtype)
    return _symmray_dense_gate_from_site_maps(
        tn,
        dense_swap,
        output_sites=(site_b, site_a),
        input_sites=(site_a, site_b),
        ind_id=ind_id,
    )


def _convert_internal_gate_to_backend(gate, inferred_converter):
    """Best-effort conversion for internally generated exact gates."""
    if inferred_converter is None:
        return gate
    try:
        return inferred_converter(gate)
    except (TypeError, ValueError):
        return gate


def _swap_gate_for_site_pair(
    tn,
    site_a,
    site_b,
    *,
    ind_id=None,
    dtype="complex128",
    inferred_converter=None,
):
    """Return a SWAP tensor matching the sites' current physical dimensions."""
    symmray_swap = _symmray_swap_gate_for_site_pair(
        tn,
        site_a,
        site_b,
        ind_id=ind_id,
        dtype=dtype,
    )
    if symmray_swap is not None:
        return symmray_swap

    dim_a = _physical_index_size_for_site(tn, site_a, ind_id)
    dim_b = _physical_index_size_for_site(tn, site_b, ind_id)
    swap_gate = _rectangular_swap_gate(dim_a, dim_b, dtype=dtype)
    return _convert_internal_gate_to_backend(swap_gate, inferred_converter)


def _normalize_gate_which(which):
    """Normalize an upper/lower layer selector."""
    if which is None:
        return None

    which_norm = str(which).strip().lower()
    if which_norm not in ("upper", "lower"):
        raise ValueError("which must be 'upper' or 'lower' (case-insensitive).")
    return which_norm


def _ind_id_from_which(which, arity):
    """Return the physical-index format corresponding to *which* and arity."""
    which_norm = _normalize_gate_which(which)
    if which_norm is None:
        return None

    prefix = "k" if which_norm == "upper" else "b"
    if arity == 1:
        return f"{prefix}{{}}"
    if arity == 2:
        return f"{prefix}{{}},{{}}"
    if arity == 3:
        return f"{prefix}{{}},{{}},{{}}"
    raise ValueError("which can only be mapped for 1D, 2D, or 3D coordinates.")


def _validate_gate_target_inds_exist(tn, where_norm, ind_id):
    """Raise a clear error when derived physical indices are missing."""
    outer_inds_fn = getattr(tn, "outer_inds", None)
    if not callable(outer_inds_fn):
        return

    outer_inds = set(outer_inds_fn())
    if not outer_inds:
        return

    if where_norm and all(isinstance(v, Integral) for v in where_norm):
        target_inds = [_format_ind_id(ind_id, site) for site in where_norm]
    else:
        target_inds = [_format_ind_id(ind_id, site) for site in where_norm]

    missing = [ind for ind in target_inds if ind not in outer_inds]
    if missing:
        missing_str = ", ".join(sorted(set(missing)))
        raise ValueError(
            "Could not find target physical indices in tn.outer_inds(): "
            f"{missing_str}. If your TN uses non-'k' physical index names, "
            "pass ind_id explicitly (for example ind_id='b{}')."
        )


def _tn_lattice_arity_hint(tn):
    """Infer likely lattice-coordinate arity from TN outer index names."""
    outer_inds_fn = getattr(tn, "outer_inds", None)
    if not callable(outer_inds_fn):
        return None

    candidates = []
    for ind in outer_inds_fn():
        match = re.match(r"^.*?(-?\d+(?:,-?\d+)*)$", str(ind))
        if match is not None:
            arity = len(match.group(1).split(","))
            candidates.append(arity)

    if not candidates:
        return None

    counts = {}
    for arity in candidates:
        counts[arity] = counts.get(arity, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _infer_where_arity_for_gate(tn, where):
    """Infer if ``where`` is intended for 1D, 2D, or 3D gate application."""
    outer_hint = None

    def _get_outer_hint():
        nonlocal outer_hint
        if outer_hint is None:
            outer_hint = _tn_lattice_arity_hint(tn)
        return outer_hint

    if isinstance(where, Integral):
        if hasattr(tn, "Lz"):
            return None
        if hasattr(tn, "Lx") and hasattr(tn, "Ly"):
            return None
        if (_get_outer_hint() or 0) >= 2:
            return None
        return 1

    if not isinstance(where, (tuple, list)):
        return None
    if not where:
        return None

    if len(where) == 1 and _is_lattice_coord(where[0], 2):
        return 2

    if len(where) == 1 and _is_lattice_coord(where[0], 3):
        return 3

    if len(where) == 2 and _is_lattice_coord(where[0], 2) and _is_lattice_coord(where[1], 2):
        return 2

    if len(where) == 2 and _is_lattice_coord(where[0], 3) and _is_lattice_coord(where[1], 3):
        return 3

    if all(isinstance(v, Integral) for v in where):
        if len(where) == 1:
            if hasattr(tn, "Lz"):
                return None
            if hasattr(tn, "Lx") and hasattr(tn, "Ly"):
                return None
            if (_get_outer_hint() or 0) >= 2:
                return None
            return 1
        if len(where) == 2:
            if hasattr(tn, "Lz"):
                return None
            if hasattr(tn, "Lx") and hasattr(tn, "Ly"):
                return 2
            outer_hint = _get_outer_hint()
            if (outer_hint is not None) and (outer_hint >= 2):
                return 2
            return 1
        if len(where) == 3:
            if hasattr(tn, "Lz"):
                return 3
            outer_hint = _get_outer_hint()
            if (outer_hint is not None) and (outer_hint >= 3):
                return 3
            return None
        return len(where)

    if isinstance(where[0], (tuple, list)) and where[0] and all(
        isinstance(v, Integral) for v in where[0]
    ):
        if len(where[0]) == 1:
            if hasattr(tn, "Lz"):
                return None
            if hasattr(tn, "Lx") and hasattr(tn, "Ly"):
                return None
            if (_get_outer_hint() or 0) >= 2:
                return None
        return len(where[0])

    return None


def _looks_like_where_payload(where):
    """Return True when ``where`` matches valid 1D/2D/3D location shapes."""
    if where is None:
        return False

    if isinstance(where, Integral):
        return True

    if _is_explicit_index_where(where):
        return True

    if not isinstance(where, (tuple, list)) or len(where) == 0:
        return False

    # 1D site tuples like (i,) or (i, j)
    if all(isinstance(v, Integral) for v in where):
        return len(where) in (1, 2, 3)

    # Coordinate tuples like ((i, j),), ((i0, j0), (i1, j1)), etc.
    if all(
        isinstance(site, (tuple, list))
        and len(site) > 0
        and all(isinstance(v, Integral) for v in site)
        for site in where
    ):
        return len(where[0]) in (1, 2, 3)

    return False


def _looks_like_gate_where_pair_entry(item, *, allow_which=False):
    """Return True when ``item`` has canonical ``(gate, where[, which])`` shape."""
    if (
        allow_which
        and isinstance(item, tuple)
        and len(item) == 3
        and not isinstance(item[0], (Integral, float, complex, str, bytes, bool))
        and _looks_like_where_payload(item[1])
    ):
        try:
            _normalize_gate_which(item[2])
        except ValueError:
            return False
        return True

    return (
        isinstance(item, tuple)
        and len(item) == 2
        and not isinstance(item[0], (Integral, float, complex, str, bytes, bool))
        and _looks_like_where_payload(item[1])
    )


def _looks_like_where_stream(where):
    """Return True when ``where`` resembles a stream of location payloads."""
    return (
        isinstance(where, (tuple, list))
        and len(where) > 0
        and any(not isinstance(w, Integral) for w in where)
        and all(_looks_like_where_payload(w) for w in where)
    )


def _looks_like_parallel_gate_where_alias(gates, where):
    """Detect legacy alias ``(G_list, where_list)`` for bundled streams."""
    return (
        isinstance(gates, (tuple, list))
        and len(gates) > 0
        and isinstance(where, (tuple, list))
        and len(where) > 0
        and (len(gates) == len(where))
        and _looks_like_where_stream(where)
    )


def _normalize_gate_entries(
    gates, where=None, *, allow_empty=True, allow_which=False
):
    """Normalize gate input into canonical entries.

    With ``allow_which=False`` entries have shape ``(gate, where)``. With
    ``allow_which=True`` entries have shape ``(gate, where, which_or_None)``
    and bundled stream entries may use ``(gate, where, "upper"|"lower")``.
    """
    if gates is None:
        if allow_empty and where is None:
            return []
        raise ValueError("gates must not be None.")

    if where is not None:
        if _looks_like_gate_where_pair_entry(gates, allow_which=allow_which):
            raise ValueError(
                "Do not pass a separate where when gates is already a bundled entry."
            )
        if isinstance(gates, (tuple, list)) and len(gates) > 0:
            if all(
                _looks_like_gate_where_pair_entry(item, allow_which=allow_which)
                for item in gates
            ):
                raise ValueError(
                    "Do not pass a separate where when gates is already a bundled stream. "
                    "Use ((gate, where), ...) or ((gate, where, which), ...)."
                )
            if (
                len(gates) > 1
                and isinstance(where, (tuple, list))
                and not _looks_like_gate_where_pair_entry(
                    gates, allow_which=allow_which
                )
            ):
                raise ValueError(
                    "For multiple gates, use a bundled stream "
                    "instead of parallel gates/where lists."
                )
            if _looks_like_parallel_gate_where_alias(gates, where):
                raise ValueError(
                    "For multiple gates, use a bundled stream "
                    "instead of parallel gates/where lists."
                )
        if allow_which:
            return [(gates, where, None)]
        return [(gates, where)]

    if _looks_like_gate_where_pair_entry(gates, allow_which=allow_which):
        gate_i, where_i = gates[0], gates[1]
        if _looks_like_parallel_gate_where_alias(gate_i, where_i):
            raise ValueError(
                "Bundled gate streams must use exact entry shape; "
                "the alias (gates, wheres) is not allowed."
            )
        if allow_which and len(gates) == 3:
            raise ValueError(
                "Single bundled triple alias (gate, where, which) is not allowed. "
                "Pass a single gate as gate(..., gate, where=where, which=which), "
                "or use bundled stream ((gate, where, which), ...)."
            )
        raise ValueError(
            "Single bundled pair alias (gate, where) is not allowed. "
            "Pass a single gate as gate(..., gate, where=where), or use "
            "bundled stream ((gate, where), ...)."
        )

    if isinstance(gates, (tuple, list)):
        if len(gates) == 0:
            if allow_empty:
                return []
            raise ValueError("gates must not be empty.")

        if all(
            _looks_like_gate_where_pair_entry(item, allow_which=allow_which)
            for item in gates
        ):
            if allow_which:
                return [
                    (item[0], item[1], _normalize_gate_which(item[2]) if len(item) == 3 else None)
                    for item in gates
                ]
            return [(item[0], item[1]) for item in gates]
        if any(
            _looks_like_gate_where_pair_entry(item, allow_which=allow_which)
            for item in gates
        ):
            raise ValueError(
                "Bundled gate stream entries must all have exact shape "
                "(gate, where) or (gate, where, which)."
            )
        if (
            isinstance(gates, list)
            and len(gates) == 2
            and _looks_like_where_payload(gates[1])
        ):
            raise ValueError(
                "Single bundled pair alias (gate, where) is not allowed. "
                "Pass a single gate as gate(..., gate, where=where), or use "
                "bundled stream ((gate, where), ...)."
            )
        if (
            allow_which
            and isinstance(gates, list)
            and len(gates) == 3
            and _looks_like_where_payload(gates[1])
        ):
            raise ValueError(
                "Single bundled triple alias (gate, where, which) is not allowed. "
                "Pass a single gate as gate(..., gate, where=where, which=which), "
                "or use bundled stream ((gate, where, which), ...)."
            )
        if len(gates) > 1:
            raise ValueError(
                "For multiple gates, use bundled stream ((gate, where), ...)."
            )

    raise ValueError(
        "where must be provided for a single gate, or use bundled stream "
        "((gate, where), ...)."
    )


def gate(tn, gates, where=None, which=None, **kwargs):
    """Apply one or many gates with automatic 1D/2D/3D dispatch.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Input tensor network.
    gates : array_like | tuple | sequence
        Gate payload. Use ``gate(tn, G, where=...)`` for a single gate, or
        ``gate(tn, ((G1, where1), (G2, where2), ...))`` for bundled gates.
        The single bundled alias ``(gate, where)`` is rejected to avoid
        ambiguity with a plain rank-2 gate tensor.
    where : object, optional
        Target location for single-gate form.
        - 1D: ``1``, ``(1,)``, ``(1, 2)``
        - 2D: ``(i, j)``, ``((i, j),)``, ``((i0, j0), (i1, j1))``
        - 3D: ``(i, j, k)``, ``((i, j, k),)``, ``((...), (...))``
        - explicit physical index names: ``"k3"`` or ``("b0", "b1")``
    which : {"upper", "lower"} | None, optional
        Convenience selector for operator-like TNs. ``"upper"`` maps numeric
        coordinates to ``k...`` physical indices and ``"lower"`` maps them to
        ``b...`` indices. Bundled stream entries may override this with
        ``(gate, where, which)``. Explicit ``ind_id`` still wins for entries
        without their own ``which`` value.
    **kwargs
        Forwarded to the selected implementation. Common options include
        ``ind_id``, ``contract``, ``cutoff``, ``max_bond``, ``inplace``,
        optional final compression controls ``chi`` / ``chi_cutoff``, and
        route-local controls ``path_canonize`` / ``path_compress``. Use
        ``max_bond`` for the per-SWAP/per-gate local split truncation;
        ``chi`` is a final whole-network compression pass after the stream.

    Returns
    -------
    qtn.TensorNetwork
        Updated tensor network. If ``inplace=False``, a copy is returned.

    Notes
    -----
    User-provided gates are applied as-is; backend coercion is not performed for
    gate tensors. Provide TN and gate tensors on compatible backends explicitly.
    For one-site gates, ``contract`` is normalized to a boolean mode:
    non-boolean values are treated as ``True``.
    Internal SWAP tensors used for long-range routing infer the current
    physical dimensions of each adjacent pair and are backend-aligned from the
    TN sample data when available.
    For nonlocal two-site gates, long-range SWAP routing is used in 1D/2D/3D
    when ``contract`` is ``"split"`` or ``"reduce-split"``. For other contract
    modes, the gate is applied directly to the requested endpoints.
    By default, 2D/3D routing uses ``sequence="auto"`` to choose a shortest
    route with the smallest current virtual-bond bottleneck, and
    ``contract="reduce-split"`` for quimb's reduced two-site split path. Pass
    an explicit deterministic sequence or contract mode to force a particular
    route or split strategy.
    The efficient routed pattern is::

        gate(
            peps,
            G,
            where,
            max_bond=chi,
            cutoff=cutoff,
            path_canonize=True,
            path_compress=False,
        )

    Here ``path_canonize=True`` performs one local canonization around the full
    route before the SWAP/gate/SWAP-back sequence. Each local split already
    receives ``max_bond``; ``path_compress=True`` is only an extra cleanup pass
    over the route bonds after the sequence.
    """
    entries = _normalize_gate_entries(
        gates, where=where, allow_empty=True, allow_which=True
    )
    opts = dict(kwargs)
    opts.setdefault("cutoff_mode", "rsum2")
    inplace = opts.pop("inplace", True)
    chi = opts.pop("chi", None)
    chi_cutoff = float(opts.pop("chi_cutoff", 1.0e-12))
    which_default = _normalize_gate_which(which)
    if "gauges" in opts or "renorm" in opts or "smudge" in opts:
        raise TypeError(
            "gate() no longer accepts 'gauges'/'renorm'/'smudge'. "
            "Use pepsy.gate_simple(tn, G, where, gauges, ...) instead."
        )

    tn_work = tn if inplace else (tn.copy() if hasattr(tn, "copy") else tn)
    if not entries:
        _apply_chi_compression(tn_work, chi=chi, chi_cutoff=chi_cutoff)
        return tn_work

    for gate_payload, where_payload, which_payload in entries:
        if _is_explicit_index_where(where_payload):
            inds = [where_payload] if isinstance(where_payload, str) else list(where_payload)
            opts_local = dict(opts)
            opts_local.pop("ind_id", None)
            if len(inds) == 1:
                contract_mode = opts_local.get("contract", True)
                if not isinstance(contract_mode, bool):
                    contract_mode = True
                opts_local["contract"] = contract_mode
            tn_work = qtn.tensor_network_gate_inds(
                tn_work,
                gate_payload,
                inds,
                inplace=True,
                **opts_local,
            )
            continue

        arity = _infer_where_arity_for_gate(tn_work, where_payload)
        if arity == 1:
            opts_local = dict(opts)
            opts_local.setdefault("contract", "split-gate")
            if which_payload is not None:
                opts_local["ind_id"] = _ind_id_from_which(which_payload, 1)
            elif (which_default is not None) and ("ind_id" not in opts_local):
                opts_local["ind_id"] = _ind_id_from_which(which_default, 1)
            where_norm = _normalize_where_arg_1d(where_payload)
            if len(where_norm) == 1 and not isinstance(opts_local.get("contract"), bool):
                opts_local["contract"] = True
            ind_id = opts_local.get("ind_id", "k{}")
            _validate_gate_target_inds_exist(tn_work, where_norm, ind_id)

            # Generic TNs without 1D lattice metadata should use direct index application.
            if not hasattr(tn_work, "L"):
                inds = [_format_ind_id(ind_id, site) for site in where_norm]
                opts_local.pop("ind_id", None)
                opts_local.pop("site_tags", None)
                opts_local.pop("dtype", None)
                tn_work = qtn.tensor_network_gate_inds(
                    tn_work,
                    gate_payload,
                    inds,
                    inplace=True,
                    **opts_local,
                )
            else:
                tn_work = _apply_gate_1d(
                    tn_work,
                    gate_payload,
                    where_norm,
                    inplace=True,
                    **opts_local,
                )
            continue

        if arity == 2:
            opts_local = dict(opts)
            opts_local.setdefault("contract", "reduce-split")
            if which_payload is not None:
                opts_local["ind_id"] = _ind_id_from_which(which_payload, 2)
            elif (which_default is not None) and ("ind_id" not in opts_local):
                opts_local["ind_id"] = _ind_id_from_which(which_default, 2)
            where_norm = _normalize_where_arg_2d(where_payload)
            if len(where_norm) == 1 and not isinstance(opts_local.get("contract"), bool):
                opts_local["contract"] = True
            _validate_gate_target_inds_exist(
                tn_work,
                where_norm,
                opts_local.get("ind_id", "k{},{}"),
            )
            tn_work = _apply_gate_2d(tn_work, gate_payload, where_norm, **opts_local)
            continue

        if arity == 3:
            opts_local = dict(opts)
            opts_local.setdefault("contract", "reduce-split")
            if which_payload is not None:
                opts_local["ind_id"] = _ind_id_from_which(which_payload, 3)
            elif (which_default is not None) and ("ind_id" not in opts_local):
                opts_local["ind_id"] = _ind_id_from_which(which_default, 3)
            where_norm = _normalize_where_arg_3d(where_payload)
            if len(where_norm) == 1 and not isinstance(opts_local.get("contract"), bool):
                opts_local["contract"] = True
            _validate_gate_target_inds_exist(
                tn_work,
                where_norm,
                opts_local.get("ind_id", "k{},{},{}"),
            )
            tn_work = _apply_gate_3d(tn_work, gate_payload, where_norm, **opts_local)
            continue

        if arity is not None and arity > 3:
            raise NotImplementedError(
                "gate currently supports only 1D, 2D, and 3D coordinates."
            )
        raise ValueError("Could not infer gate dimensionality from where.")

    _apply_chi_compression(tn_work, chi=chi, chi_cutoff=chi_cutoff)
    return tn_work


def gate_simple(
    tn,
    G,
    where=None,
    gauges=None,
    *,
    which=None,
    ind_id=None,
    renorm=True,
    smudge=1e-12,
    max_bond=None,
    cutoff=1e-12,
    cutoff_mode="rsum2",
    sequence="auto",
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
    path_compress=False,
    path_compress_max_bond=None,
    path_compress_cutoff=None,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
    inplace=True,
):
    """Apply one or many gates using simple-update gauges.

    Thin pepsy wrapper around quimb's ``tn.gate_simple_()`` that adds:

    * Canonical bundled gate streams ``((G1, where1), (G2, where2), ...)``.
      Entries may use ``(G, where, "upper"|"lower")`` to override ``which``.
    * Automatic long-range SWAP routing when the two sites are not adjacent
      (works for 1D / 2D / 3D ``where`` coordinates).
    * ``which``/``ind_id`` selection for vector-like networks whose physical
      site-index family is not the default ``k...`` family.
    * Dimension-aware, backend-aligned internal SWAP tensors for long-range
      routing through mixed physical dimensions.
    * Optional out-of-place semantics via ``inplace=False``.

    The ``gauges`` dictionary is mutated in place by ``gate_simple_`` and is
    the single source of truth for the simple-update bond environment.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Tensor network supporting ``gate_simple_()`` (1D MPS, 2D PEPS, 3D PEPS).
    G : array_like
        Gate tensor, or bundled gate stream.
    where : tuple, optional
        Site coordinates for single-gate form. For a one-site gate: ``(site,)``
        or ``site``.
        For a two-site gate: ``(site_a, site_b)``, where each ``site`` is an
        ``int`` (1D) or a tuple ``(i, j)`` / ``(i, j, k)`` (2D / 3D).
    gauges : dict
        Simple-update gauge dictionary keyed by bond index (mutated in place).
    which : {"upper", "lower"} | None, optional
        Convenience selector for the physical index family. ``"upper"`` maps
        to ``k...`` indices and ``"lower"`` maps to ``b...`` indices.
    ind_id : str | None, optional
        Explicit physical index format, e.g. ``"k{}"``, ``"b{},{}"``.
    renorm : bool, optional
        Whether to renormalize the singular values after the gate. Default True.
        Ignored for one-site gates.
    smudge : float, optional
        Small numerical-safety value. Default 1e-12.
    max_bond : int or None, optional
        Maximum bond dimension for each local SWAP/gate simple-update split.
        Default None.
    cutoff : float, optional
        Truncation cutoff. Default 1e-12.
    cutoff_mode : str, optional
        Cutoff mode passed to ``gate_simple_`` (e.g. ``'rsum2'``, ``'rel'``).
        Default ``'rsum2'``.
    sequence : object, optional
        Long-range routing preference. Defaults to ``"auto"``, which chooses a
        shortest 2D/3D path with the smallest current bond-dimension bottleneck.
    path_canonize, path_compress : bool, optional
        If enabled, ``path_canonize`` uses one local ``canonize_around_`` call
        before the full route, while ``path_compress`` runs an extra
        ``compress_between`` cleanup on route bonds after the gate/SWAP
        sequence. The local gate calls already receive ``max_bond``.
    inplace : bool, optional
        If False, work on a copy of ``tn``. Default True.

    Returns
    -------
    qtn.TensorNetwork
        The updated tensor network (same object as ``tn`` when ``inplace=True``).
    """
    if gauges is None and isinstance(where, dict):
        gauges = where
        where = None
    if gauges is None:
        raise TypeError("gate_simple() requires a gauges dictionary.")

    tn_work = tn if inplace else (tn.copy() if hasattr(tn, "copy") else tn)
    entries = _normalize_gate_entries(
        G, where=where, allow_empty=True, allow_which=True
    )
    which_default = _normalize_gate_which(which)

    gate_opts = {
        "cutoff": cutoff,
        "cutoff_mode": cutoff_mode,
    }
    if max_bond is not None:
        gate_opts["max_bond"] = int(max_bond)

    for gate_payload, where_payload, which_payload in entries:
        if _is_explicit_index_where(where_payload):
            raise ValueError(
                "gate_simple() requires lattice site coordinates, not explicit "
                "index-name selectors."
            )

        arity = _infer_where_arity_for_gate(tn_work, where_payload)
        if arity == 1:
            where_norm = _normalize_where_arg_1d(where_payload)
        elif arity == 2:
            where_norm = _normalize_where_arg_2d(where_payload)
        elif arity == 3:
            where_norm = _normalize_where_arg_3d(where_payload)
        elif arity is not None and arity > 3:
            raise NotImplementedError(
                "gate_simple currently supports only 1D, 2D, and 3D coordinates."
            )
        else:
            raise ValueError("Could not infer gate dimensionality from where.")

        ind_id_local = ind_id
        if which_payload is not None:
            ind_id_local = _ind_id_from_which(which_payload, arity)
        elif which_default is not None:
            ind_id_local = _ind_id_from_which(which_default, arity)

        if ind_id_local is not None:
            _validate_gate_target_inds_exist(tn_work, where_norm, ind_id_local)

        _gate_simple_one(
            tn_work,
            gate_payload,
            where_norm,
            gauges,
            renorm=renorm,
            smudge=smudge,
            gate_opts=gate_opts,
            ind_id=ind_id_local,
            sequence=sequence,
            path_canonize=path_canonize,
            path_canonize_distance=path_canonize_distance,
            path_canonize_opts=path_canonize_opts,
            path_compress=path_compress,
            path_compress_max_bond=path_compress_max_bond,
            path_compress_cutoff=path_compress_cutoff,
            path_compress_canonize_distance=path_compress_canonize_distance,
            path_compress_opts=path_compress_opts,
        )

    return tn_work


def gate_loop_cluster(
    tn,
    G,
    where=None,
    gauges=None,
    *,
    which=None,
    ind_id=None,
    max_bond=None,
    max_loop_size: int = 0,
    base_radius: int = 0,
    include_full_system: bool | None = None,
    autocomplete: bool = True,
    psd_project: bool = True,
    psd_floor: float = 0.0,
    smudge: float = 0.0,
    als_opts=None,
    regauge_opts=None,
    inplace=True,
    return_results: bool = False,
):
    """Apply a PEPS gate stream with SU-gauged reduced loop-cluster updates.

    This is the gate-stream bridge for
    :func:`pepsy.bp.apply_reduced_loop_cluster_gate`. Adjacent two-site gates
    are updated by the open-leg loop-cluster metric and re-gauged into the
    supplied SU ``gauges`` dictionary. One-site gates are applied with quimb's
    simple-update path and do not change the gauges. Long-range routing is not
    part of this first nearest-neighbour TEBD path. The current reduced solver
    is dense and rejects symmray block-sparse tensor arrays.
    """
    if gauges is None and isinstance(where, dict):
        gauges = where
        where = None
    if gauges is None:
        raise TypeError("gate_loop_cluster() requires a gauges dictionary.")

    tn_work = tn if inplace else (tn.copy() if hasattr(tn, "copy") else tn)
    entries = _normalize_gate_entries(
        G, where=where, allow_empty=True, allow_which=True
    )
    which_default = _normalize_gate_which(which)
    results = []
    regauge_opts_use = {} if regauge_opts is None else dict(regauge_opts)

    if any(hasattr(tensor.data, "blocks") for tensor in tn_work.tensor_map.values()):
        raise NotImplementedError(
            "gate_loop_cluster currently requires dense PEPS tensor arrays; "
            "symmetry-block-aware reduced loop-cluster updates are not "
            "implemented yet."
        )

    if entries and any(index not in gauges for index in tn_work.inner_inds()):
        from ..bp import gauge_all_simple

        initial_gauge_opts = dict(regauge_opts_use)
        initial_gauge_opts.setdefault("max_iterations", 20)
        initial_gauge_opts.setdefault("tol", 0.0)
        gauge_all_simple(
            tn_work,
            gauges=gauges,
            inplace=True,
            **initial_gauge_opts,
        )

    if any(hasattr(tensor.data, "to_dense") for tensor in tn_work.tensor_map.values()):
        for tensor in tn_work.tensor_map.values():
            if hasattr(tensor.data, "to_dense"):
                tensor.modify(data=np.asarray(tensor.data.to_dense()))
        for index, gauge in tuple(gauges.items()):
            if hasattr(gauge, "to_dense"):
                gauges[index] = np.asarray(gauge.to_dense())

    for gate_payload, where_payload, which_payload in entries:
        if hasattr(gate_payload, "to_dense"):
            gate_payload = np.asarray(gate_payload.to_dense())

        if _is_explicit_index_where(where_payload):
            raise ValueError(
                "gate_loop_cluster() requires lattice site coordinates, not "
                "explicit index-name selectors."
            )

        arity = _infer_where_arity_for_gate(tn_work, where_payload)
        if arity == 1:
            where_norm = _normalize_where_arg_1d(where_payload)
        elif arity == 2:
            where_norm = _normalize_where_arg_2d(where_payload)
        elif arity == 3:
            where_norm = _normalize_where_arg_3d(where_payload)
        elif arity is not None and arity > 3:
            raise NotImplementedError(
                "gate_loop_cluster currently supports only 1D, 2D, and 3D "
                "coordinates."
            )
        else:
            raise ValueError("Could not infer gate dimensionality from where.")

        ind_id_local = ind_id
        if which_payload is not None:
            ind_id_local = _ind_id_from_which(which_payload, arity)
        elif which_default is not None:
            ind_id_local = _ind_id_from_which(which_default, arity)
        if ind_id_local is not None:
            _validate_gate_target_inds_exist(tn_work, where_norm, ind_id_local)

        has_site_ind_id = hasattr(tn_work, "site_ind_id")
        old_site_ind_id = getattr(tn_work, "site_ind_id", None)
        if ind_id_local is not None:
            if not has_site_ind_id:
                raise ValueError(
                    "gate_loop_cluster() can only use ind_id/which on tensor "
                    "networks with a site_ind_id attribute."
                )
            tn_work.site_ind_id = ind_id_local

        try:
            if len(where_norm) == 1:
                tn_work.gate_simple_(
                    gate_payload,
                    where=where_norm,
                    gauges=gauges,
                    renorm=False,
                    smudge=smudge,
                    inplace=True,
                )
                continue

            if len(where_norm) != 2:
                raise ValueError(
                    "gate_loop_cluster() supports only one-site and two-site "
                    "gates."
                )

            site_a, site_b = where_norm
            if site_a == site_b:
                raise ValueError("Two-site gate requires distinct coordinates.")
            if not qtn.bonds(
                tn_work[tn_work.site_tag(site_a)],
                tn_work[tn_work.site_tag(site_b)],
            ):
                raise ValueError(
                    "gate_loop_cluster() currently requires adjacent two-site "
                    "gates; route long-range gates before this update."
                )

            from ..bp import apply_reduced_loop_cluster_gate

            results.append(
                apply_reduced_loop_cluster_gate(
                    tn_work,
                    gauges,
                    gate_payload,
                    where=where_norm,
                    max_bond=max_bond,
                    max_loop_size=max_loop_size,
                    base_radius=base_radius,
                    include_full_system=include_full_system,
                    autocomplete=autocomplete,
                    psd_project=psd_project,
                    psd_floor=psd_floor,
                    smudge=smudge,
                    als_opts=als_opts,
                    regauge_opts=regauge_opts_use,
                    inplace=True,
                )
            )
        finally:
            if ind_id_local is not None and has_site_ind_id:
                tn_work.site_ind_id = old_site_ind_id

    if return_results:
        return tn_work, tuple(results)
    return tn_work


def _gate_simple_one(
    tn_work,
    G,
    where,
    gauges,
    *,
    renorm,
    smudge,
    gate_opts,
    ind_id=None,
    sequence=None,
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
    path_compress=False,
    path_compress_max_bond=None,
    path_compress_cutoff=None,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
):
    """Apply a single normalized gate through quimb simple update."""
    if not hasattr(tn_work, "gate_simple_"):
        raise TypeError(
            "gate_simple() requires a vector-like tensor network supporting "
            "gate_simple_(); use gate(..., which=...) or gate_nonlocal_opt(...) "
            "for MPO/PEPO operator layers."
        )

    has_site_ind_id = hasattr(tn_work, "site_ind_id")
    old_site_ind_id = getattr(tn_work, "site_ind_id", None)
    if ind_id is not None:
        if not has_site_ind_id:
            raise ValueError(
                "gate_simple() can only use ind_id/which on tensor networks "
                "with a site_ind_id attribute."
            )
        tn_work.site_ind_id = ind_id

    try:
        return _gate_simple_one_with_current_site_ind_id(
            tn_work,
            G,
            where,
            gauges,
            renorm=renorm,
            smudge=smudge,
            gate_opts=gate_opts,
            sequence=sequence,
            path_canonize=path_canonize,
            path_canonize_distance=path_canonize_distance,
            path_canonize_opts=path_canonize_opts,
            path_compress=path_compress,
            path_compress_max_bond=path_compress_max_bond,
            path_compress_cutoff=path_compress_cutoff,
            path_compress_canonize_distance=path_compress_canonize_distance,
            path_compress_opts=path_compress_opts,
        )
    finally:
        if ind_id is not None and has_site_ind_id:
            tn_work.site_ind_id = old_site_ind_id


def _gate_simple_one_with_current_site_ind_id(
    tn_work,
    G,
    where,
    gauges,
    *,
    renorm,
    smudge,
    gate_opts,
    sequence=None,
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
    path_compress=False,
    path_compress_max_bond=None,
    path_compress_cutoff=None,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
):
    """Apply a single gate assuming ``site_ind_id`` has already been selected."""
    # One-site gate — no gauge update needed.
    if len(where) == 1:
        tn_work.gate_simple_(
            G, where=where, gauges=gauges,
            renorm=False, smudge=smudge, inplace=True,
        )
        return tn_work

    # Two-site gate — check if the sites share a bond.
    site_a, site_b = where
    if site_a == site_b:
        raise ValueError("Two-site gate requires distinct coordinates.")

    tag_a = tn_work.site_tag(site_a)
    tag_b = tn_work.site_tag(site_b)
    adjacent = bool(qtn.bonds(tn_work[tag_a], tn_work[tag_b]))

    if adjacent:
        path_pairs = [(site_a, site_b)]
        _maybe_canonize_path(
            tn_work,
            path_pairs,
            path_canonize=path_canonize,
            path_canonize_distance=path_canonize_distance,
            path_canonize_opts=path_canonize_opts,
        )
        tn_work.gate_simple_(
            G, where=where, gauges=gauges,
            renorm=renorm, smudge=smudge, inplace=True,
            **gate_opts,
        )
        _maybe_compress_path(
            tn_work,
            path_pairs,
            path_compress=path_compress,
            max_bond=path_compress_max_bond if path_compress_max_bond is not None else gate_opts.get("max_bond"),
            cutoff=path_compress_cutoff if path_compress_cutoff is not None else gate_opts.get("cutoff", 1.0e-12),
            path_compress_canonize_distance=path_compress_canonize_distance,
            path_compress_opts=path_compress_opts,
        )
        return tn_work

    # Non-adjacent: route through a SWAP chain. Each SWAP is built from the
    # live physical dimensions because routed mixed-dimensional sites exchange
    # their physical index sizes as they move along the path.
    backend_sample = resolve_backend_sample_data_from_tn(tn_work)
    inferred_converter = infer_backend_converter_from_sample(
        backend_sample,
        cast_complex_to_real=True,
    )
    swap_ind_id = getattr(tn_work, "site_ind_id", None)

    ndim = len(site_a) if isinstance(site_a, (tuple, list)) else 1
    if ndim == 1:
        path_pairs = list(gen_long_range_swap_path_1d(site_a, site_b))
    elif ndim == 2:
        cyclic = bool(getattr(tn_work, "_cyclic", False))
        path_pairs = _select_long_range_swap_path_2d(
            tn_work,
            site_a, site_b, cyclic=cyclic,
            Lx=getattr(tn_work, "Lx", None),
            Ly=getattr(tn_work, "Ly", None),
            sequence=sequence,
        )
    elif ndim == 3:
        cyclic = bool(getattr(tn_work, "_cyclic", False))
        path_pairs = _select_long_range_swap_path_3d(
            tn_work,
            site_a, site_b, cyclic=cyclic,
            Lx=getattr(tn_work, "Lx", None),
            Ly=getattr(tn_work, "Ly", None),
            Lz=getattr(tn_work, "Lz", None),
            sequence=sequence,
        )
    else:
        raise ValueError(f"gate_simple: unsupported dimensionality: {ndim}")

    *swaps, final = path_pairs
    _maybe_canonize_path(
        tn_work,
        path_pairs,
        path_canonize=path_canonize,
        path_canonize_distance=path_canonize_distance,
        path_canonize_opts=path_canonize_opts,
    )

    # Forward SWAPs.
    for pair in swaps:
        swap_gate = _swap_gate_for_site_pair(
            tn_work,
            pair[0],
            pair[1],
            ind_id=swap_ind_id,
            dtype="complex128",
            inferred_converter=inferred_converter,
        )
        tn_work.gate_simple_(
            swap_gate, where=pair, gauges=gauges,
            renorm=renorm, smudge=smudge, inplace=True,
            **gate_opts,
        )

    # Apply the actual gate on the final (now adjacent) pair.
    tn_work.gate_simple_(
        G, where=final, gauges=gauges,
        renorm=renorm, smudge=smudge, inplace=True,
        **gate_opts,
    )

    # Reverse SWAPs.
    for pair in reversed(swaps):
        swap_gate = _swap_gate_for_site_pair(
            tn_work,
            pair[0],
            pair[1],
            ind_id=swap_ind_id,
            dtype="complex128",
            inferred_converter=inferred_converter,
        )
        tn_work.gate_simple_(
            swap_gate, where=pair, gauges=gauges,
            renorm=renorm, smudge=smudge, inplace=True,
            **gate_opts,
        )

    _maybe_compress_path(
        tn_work,
        path_pairs,
        path_compress=path_compress,
        max_bond=path_compress_max_bond if path_compress_max_bond is not None else gate_opts.get("max_bond"),
        cutoff=path_compress_cutoff if path_compress_cutoff is not None else gate_opts.get("cutoff", 1.0e-12),
        path_compress_canonize_distance=path_compress_canonize_distance,
        path_compress_opts=path_compress_opts,
    )

    return tn_work


def renorm_gauge(tn, gauges, where, smudge=1e-12):
    """Renormalize the simple-update gauge on the bond between sites in *where*.

    Divides the gauge vector by its RMS norm (with *smudge* for safety)
    and accumulates the extracted scale into ``tn.exponent`` (if present).

    Works for 1D/2D/3D — finds the bond index between the two site tensors.

    Parameters
    ----------
    tn : TensorNetwork
        Must have ``site_tag()`` method and optionally an ``exponent`` attribute.
    gauges : dict[str, array_like]
        Gauge dictionary (mutated in-place).
    where : tuple
        Pair of site coordinates, e.g. ``(3, 4)`` for 1D,
        ``((0,1), (1,1))`` for 2D, ``((0,0,0), (0,0,1))`` for 3D.
    smudge : float
        Small value for numerical safety.
    """
    site_a, site_b = where
    tag_a = tn.site_tag(site_a)
    tag_b = tn.site_tag(site_b)
    tensor_a = tn[tag_a]
    tensor_b = tn[tag_b]
    bond_ix_set = qtn.bonds(tensor_a, tensor_b)
    if not bond_ix_set:
        raise ValueError(
            f"renorm_gauge: sites {site_a} and {site_b} are not adjacent "
            "(no shared bond)."
        )
    ix = next(iter(bond_ix_set))
    s = gauges[ix]
    norm_s = ar.do("sqrt", ar.do("mean", ar.do("abs", s) ** 2))
    # Stop gradient on the scalar norm — only tensor data carries AD info.
    norm_s = _stop_gradient(norm_s)
    if hasattr(tn, "exponent"):
        tn.exponent = tn.exponent + ar.do("log10", norm_s)
    gauges[ix] = s / (norm_s + smudge)


def _apply_gate_2d(
    peps,
    gate,
    where,
    *,
    bond_dim=None,
    max_bond=None,
    bra=False,
    contract="reduce-split",
    tags=None,
    dtype="complex128",
    cutoff=1.0e-12,
    canonize_distance=2,
    sequence="auto",
    cyclic=False,
    Lx=None,
    Ly=None,
    ind_id="k{},{}",
    cutoff_mode="rsum2",
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
    path_compress=False,
    path_compress_max_bond=None,
    path_compress_cutoff=None,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
):
    """Apply a single normalized 2D gate, routing long-range pairs with SWAPs."""

    if bra:
        warnings.warn(
            "Unused option in gate(...): 'bra' for 2D gates.",
            RuntimeWarning,
            stacklevel=2,
        )

    if tags is None:
        tags = ["G"]

    G_apply = gate
    split_max_bond = max_bond if max_bond is not None else bond_dim
    split_opts = _local_split_opts(split_max_bond)

    if len(where) == 1:
        ((i, j),) = where
        qtn.tensor_network_gate_inds(
            peps,
            G_apply,
            [_format_ind_id(ind_id, (i, j))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return peps

    if len(where) != 2:
        raise ValueError("where must contain one or two site coordinates")

    x, y = where
    if x == y:
        raise ValueError("Two-site gate requires distinct coordinates.")

    # Match 1D behavior:
    # - split / reduce-split: route long-range gates via SWAP chains
    # - split-gate (or others): apply directly to the requested endpoints
    if contract not in ("split", "reduce-split"):
        i, j = x
        m, n = y
        path_pairs = [(x, y)]
        _maybe_canonize_path(
            peps,
            path_pairs,
            path_canonize=path_canonize,
            path_canonize_distance=path_canonize_distance,
            path_canonize_opts=path_canonize_opts,
        )
        qtn.tensor_network_gate_inds(
            peps,
            G_apply,
            [_format_ind_id(ind_id, (i, j)), _format_ind_id(ind_id, (m, n))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **split_opts,
        )
        _maybe_compress_path(
            peps,
            path_pairs,
            path_compress=path_compress,
            max_bond=path_compress_max_bond if path_compress_max_bond is not None else split_max_bond,
            cutoff=path_compress_cutoff if path_compress_cutoff is not None else cutoff,
            path_compress_canonize_distance=path_compress_canonize_distance,
            path_compress_opts=path_compress_opts,
        )
        return peps

    backend_sample = resolve_backend_sample_data_from_tn(peps)
    inferred_converter = infer_backend_converter_from_sample(
        backend_sample,
        cast_complex_to_real=True,
    )
    lx_use = Lx
    ly_use = Ly
    if cyclic and (lx_use is None or ly_use is None):
        lx_use = getattr(peps, "Lx", lx_use)
        ly_use = getattr(peps, "Ly", ly_use)

    path_pairs = _select_long_range_swap_path_2d(
        peps,
        x,
        y,
        sequence=sequence,
        cyclic=cyclic,
        Lx=lx_use,
        Ly=ly_use,
    )
    *swaps, final = path_pairs
    _maybe_canonize_path(
        peps,
        path_pairs,
        path_canonize=path_canonize,
        path_canonize_distance=path_canonize_distance,
        path_canonize_opts=path_canonize_opts,
    )

    for pair in swaps:
        x_, y_ = pair
        i_, j_ = x_
        m_, n_ = y_
        swap = _swap_gate_for_site_pair(
            peps,
            x_,
            y_,
            ind_id=ind_id,
            dtype=dtype,
            inferred_converter=inferred_converter,
        )
        qtn.tensor_network_gate_inds(
            peps,
            swap,
            [_format_ind_id(ind_id, (i_, j_)), _format_ind_id(ind_id, (m_, n_))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **split_opts,
        )

    x_, y_ = final
    i_, j_ = x_
    m_, n_ = y_
    qtn.tensor_network_gate_inds(
        peps,
        G_apply,
        [_format_ind_id(ind_id, (i_, j_)), _format_ind_id(ind_id, (m_, n_))],
        contract=contract,
        tags=tags,
        info=None,
        inplace=True,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        **split_opts,
    )

    for pair in reversed(swaps):
        x_, y_ = pair
        i_, j_ = x_
        m_, n_ = y_
        swap = _swap_gate_for_site_pair(
            peps,
            x_,
            y_,
            ind_id=ind_id,
            dtype=dtype,
            inferred_converter=inferred_converter,
        )
        qtn.tensor_network_gate_inds(
            peps,
            swap,
            [_format_ind_id(ind_id, (i_, j_)), _format_ind_id(ind_id, (m_, n_))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **split_opts,
        )

    _maybe_compress_path(
        peps,
        path_pairs,
        path_compress=path_compress,
        max_bond=path_compress_max_bond if path_compress_max_bond is not None else split_max_bond,
        cutoff=path_compress_cutoff if path_compress_cutoff is not None else cutoff,
        path_compress_canonize_distance=path_compress_canonize_distance,
        path_compress_opts=path_compress_opts,
    )

    return peps


def _is_lattice_coord(value, ndim):
    return (
        isinstance(value, (tuple, list))
        and (len(value) == ndim)
        and all(isinstance(v, Integral) for v in value)
    )


def _normalize_where_arg_2d(where):
    """Normalize one-site and two-site where specs for 2D gate application."""
    if _is_lattice_coord(where, 2):
        i, j = where
        return ((int(i), int(j)),)

    if isinstance(where, (tuple, list)):
        if len(where) == 1 and _is_lattice_coord(where[0], 2):
            i, j = where[0]
            return ((int(i), int(j)),)
        if len(where) == 2 and _is_lattice_coord(where[0], 2) and _is_lattice_coord(where[1], 2):
            i0, j0 = where[0]
            i1, j1 = where[1]
            return ((int(i0), int(j0)), (int(i1), int(j1)))

    raise ValueError(
        "Invalid where specification. Expected (i, j), ((i, j),), or ((i0, j0), (i1, j1))."
    )


def _normalize_where_arg_3d(where):
    """Normalize one-site and two-site where specs for 3D gate application."""
    if _is_lattice_coord(where, 3):
        i, j, k = where
        return ((int(i), int(j), int(k)),)

    if isinstance(where, (tuple, list)):
        if len(where) == 1 and _is_lattice_coord(where[0], 3):
            i, j, k = where[0]
            return ((int(i), int(j), int(k)),)
        if len(where) == 2 and _is_lattice_coord(where[0], 3) and _is_lattice_coord(where[1], 3):
            i0, j0, k0 = where[0]
            i1, j1, k1 = where[1]
            return ((int(i0), int(j0), int(k0)), (int(i1), int(j1), int(k1)))

    raise ValueError(
        "Invalid where specification. Expected (i, j, k), ((i, j, k),), or "
        "((i0, j0, k0), (i1, j1, k1))."
    )


def _apply_chi_compression(tn, *, chi=None, chi_cutoff=1.0e-12):
    """Apply optional final bond compression used by gate streams."""
    if chi is None:
        return

    chi_value = int(chi)
    if chi_value <= 0:
        raise ValueError("chi must be a positive integer when provided.")

    if hasattr(tn, "compress"):
        tn.compress(form="left", max_bond=chi_value, cutoff=chi_cutoff)
    elif hasattr(tn, "compress_all_"):
        tn.compress_all_(max_bond=chi_value, cutoff=chi_cutoff)


def _apply_gate_3d(
    tn,
    gate,
    where,
    *,
    bond_dim=None,
    max_bond=None,
    bra=False,
    contract="reduce-split",
    tags=None,
    dtype="complex128",
    cutoff=1.0e-12,
    canonize_distance=2,
    sequence="auto",
    cyclic=False,
    Lx=None,
    Ly=None,
    Lz=None,
    ind_id="k{},{},{}",
    cutoff_mode="rsum2",
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
    path_compress=False,
    path_compress_max_bond=None,
    path_compress_cutoff=None,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
):
    """Apply a single normalized 3D gate, routing long-range pairs with SWAPs."""

    if bra:
        warnings.warn(
            "Unused option in gate(...): 'bra' for 3D gates.",
            RuntimeWarning,
            stacklevel=2,
        )

    if tags is None:
        tags = ["G"]

    G_apply = gate
    split_max_bond = max_bond if max_bond is not None else bond_dim
    split_opts = _local_split_opts(split_max_bond)

    if len(where) == 1:
        ((i, j, k),) = where
        qtn.tensor_network_gate_inds(
            tn,
            G_apply,
            [_format_ind_id(ind_id, (i, j, k))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return tn

    if len(where) != 2:
        raise ValueError("where must contain one or two site coordinates")

    x, y = where
    if x == y:
        raise ValueError("Two-site gate requires distinct coordinates.")

    if contract not in ("split", "reduce-split"):
        i, j, k = x
        m, n, p = y
        path_pairs = [(x, y)]
        _maybe_canonize_path(
            tn,
            path_pairs,
            path_canonize=path_canonize,
            path_canonize_distance=path_canonize_distance,
            path_canonize_opts=path_canonize_opts,
        )
        qtn.tensor_network_gate_inds(
            tn,
            G_apply,
            [_format_ind_id(ind_id, (i, j, k)), _format_ind_id(ind_id, (m, n, p))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **split_opts,
        )
        _maybe_compress_path(
            tn,
            path_pairs,
            path_compress=path_compress,
            max_bond=path_compress_max_bond if path_compress_max_bond is not None else split_max_bond,
            cutoff=path_compress_cutoff if path_compress_cutoff is not None else cutoff,
            path_compress_canonize_distance=path_compress_canonize_distance,
            path_compress_opts=path_compress_opts,
        )
        return tn

    backend_sample = resolve_backend_sample_data_from_tn(tn)
    inferred_converter = infer_backend_converter_from_sample(
        backend_sample,
        cast_complex_to_real=True,
    )
    lx_use = Lx
    ly_use = Ly
    lz_use = Lz
    if cyclic and ((lx_use is None) or (ly_use is None) or (lz_use is None)):
        lx_use = getattr(tn, "Lx", lx_use)
        ly_use = getattr(tn, "Ly", ly_use)
        lz_use = getattr(tn, "Lz", lz_use)

    path_pairs = _select_long_range_swap_path_3d(
        tn,
        x,
        y,
        sequence=sequence,
        cyclic=cyclic,
        Lx=lx_use,
        Ly=ly_use,
        Lz=lz_use,
    )
    *swaps, final = path_pairs
    _maybe_canonize_path(
        tn,
        path_pairs,
        path_canonize=path_canonize,
        path_canonize_distance=path_canonize_distance,
        path_canonize_opts=path_canonize_opts,
    )

    for pair in swaps:
        x_, y_ = pair
        i_, j_, k_ = x_
        m_, n_, p_ = y_
        swap = _swap_gate_for_site_pair(
            tn,
            x_,
            y_,
            ind_id=ind_id,
            dtype=dtype,
            inferred_converter=inferred_converter,
        )
        qtn.tensor_network_gate_inds(
            tn,
            swap,
            [_format_ind_id(ind_id, (i_, j_, k_)), _format_ind_id(ind_id, (m_, n_, p_))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **split_opts,
        )

    x_, y_ = final
    i_, j_, k_ = x_
    m_, n_, p_ = y_
    qtn.tensor_network_gate_inds(
        tn,
        G_apply,
        [_format_ind_id(ind_id, (i_, j_, k_)), _format_ind_id(ind_id, (m_, n_, p_))],
        contract=contract,
        tags=tags,
        info=None,
        inplace=True,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        **split_opts,
    )

    for pair in reversed(swaps):
        x_, y_ = pair
        i_, j_, k_ = x_
        m_, n_, p_ = y_
        swap = _swap_gate_for_site_pair(
            tn,
            x_,
            y_,
            ind_id=ind_id,
            dtype=dtype,
            inferred_converter=inferred_converter,
        )
        qtn.tensor_network_gate_inds(
            tn,
            swap,
            [_format_ind_id(ind_id, (i_, j_, k_)), _format_ind_id(ind_id, (m_, n_, p_))],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **split_opts,
        )

    _maybe_compress_path(
        tn,
        path_pairs,
        path_compress=path_compress,
        max_bond=path_compress_max_bond if path_compress_max_bond is not None else split_max_bond,
        cutoff=path_compress_cutoff if path_compress_cutoff is not None else cutoff,
        path_compress_canonize_distance=path_compress_canonize_distance,
        path_compress_opts=path_compress_opts,
    )

    return tn


def build_pepo_from_gates(
    gates,
    wheres=None,
    where=None,
    cyclic=False,
    cutoff=1.0e-12,
    pepo_=None,
    dtype="complex128",
    max_bond=16,
    sequence="auto",
    contract="reduce-split",
    ind_id="k{},{}",
):
    """Build a PEPO from gate-style input on top of a PEPO identity.

    Parameters
    ----------
    gates : array_like | sequence | tuple
        Gate payload. Accepted forms are:
        ``build_pepo_from_gates(G, where=...)`` for a single gate,
        ``build_pepo_from_gates(((G1, where1), (G2, where2), ...))`` for the
        canonical bundled stream, and
        ``build_pepo_from_gates([G1, G2], [where1, where2])`` for the legacy
        parallel ``wheres`` form.
    wheres : sequence[tuple] | None, optional
        Legacy parallel where stream aligned with ``gates``.
    where : object, optional
        Single-gate location (gate-style input).
    cyclic : bool, default=False
        If ``True`` and ``pepo_`` is not supplied, periodic bonds are added.
    cutoff : float, default=1e-12
        Truncation cutoff passed to gate application.
    pepo_ : qtn.TensorNetwork | None, optional
        Optional initial PEPO. If omitted, a fresh identity PEPO is created.
    dtype : str, default="complex128"
        Dtype used when creating an identity PEPO.
    max_bond : int, default=16
        Per-gate local split truncation cap and fallback construction
        compression cap.
    sequence : object, optional
        2D SWAP-path preference for long-range two-site gates. Defaults to
        ``"auto"`` for the same lower-bond smart routing used by
        :func:`gate`.
    contract : str, default="reduce-split"
        Gate contraction mode. The default uses quimb's reduced two-site split
        path, which is usually cheaper than ``"split"`` for PEPO/PEPS tensors.
    ind_id : str, default="k{},{}"
        Physical index format used for PEPO ket-family indices.

    Returns
    -------
    qtn.TensorNetwork
        Constructed PEPO operator.
    """
    entries = _normalize_builder_entries(gates, wheres=wheres, where=where, ndim=2)
    gate_list = [g for g, _ in entries]
    where_list = [w for _, w in entries]

    coords = [c for w in where_list for c in w]
    Lx = max(i for i, _ in coords) + 1
    Ly = max(j for _, j in coords) + 1

    pepo = pepo_.copy() if pepo_ is not None else id_to_pepo(Lx, Ly, dtype=dtype)
    if pepo_ is None and cyclic:
        pepo = add_cycle(pepo, 1)

    for tensor in pepo:
        tensor.modify(data=ar.do("array", tensor.data, like=gate_list[0]))

    for gate_op, where_norm in zip(gate_list, where_list):
        gate_use = _to_ket_gate_layout(gate_op, len(where_norm))

        gate(
            pepo, gate_use, where_norm,
            max_bond=max_bond, bra=False, contract=contract,
            tags=[], dtype=dtype, cutoff=cutoff,
            sequence=sequence, cyclic=cyclic, Lx=Lx, Ly=Ly, ind_id=ind_id,
            inplace=True,
        )

        if pepo.max_bond() > max_bond:
            pepo.compress_all(
                inplace=True,
                max_bond=max_bond,
                canonize_distance=4,
                cutoff=1e-14,
            )

    return pepo


def _to_ket_gate_layout(gate, n_sites):
    """Map input gate to ket-side index ordering used by PEPO/MPO builders."""
    if n_sites == 1:
        return ar.do("transpose", gate, (1, 0))

    if n_sites == 2:
        shape = getattr(gate, "shape", ())
        if len(shape) == 2:
            din, dout = shape
            if int(din) != int(dout):
                raise ValueError(
                    "Two-site gate matrix must be square with shape (d**2, d**2)."
                )
            return ar.do("transpose", gate, (1, 0))

        if len(shape) == 4:
            return ar.do("transpose", gate, (2, 3, 0, 1))

        raise ValueError(
            "Two-site gate must have shape (d**2, d**2) or (d, d, d, d)."
        )

    raise ValueError("Each gate location must have one or two sites.")


def build_mpo_from_gates(
    gates,
    wheres=None,
    where=None,
    cyclic=False,
    cutoff=1.0e-12,
    mpo_=None,
    dtype="complex128",
    max_bond=16,
    contract="reduce-split",
    ind_id="k{}",
):
    """Build an MPO from gate-style input on top of an MPO identity.

    Parameters
    ----------
    gates : array_like | sequence | tuple
        Gate payload. Accepted forms are:
        ``build_mpo_from_gates(G, where=...)`` for a single gate,
        ``build_mpo_from_gates(((G1, where1), (G2, where2), ...))`` for the
        canonical bundled stream, and
        ``build_mpo_from_gates([G1, G2], [where1, where2])`` for the legacy
        parallel ``wheres`` form.
    wheres : sequence[tuple[int, ...]] | None, optional
        Legacy parallel where stream aligned with ``gates``.
    where : object, optional
        Single-gate location (gate-style input).
    cyclic : bool, default=False
        Whether to create/use cyclic MPO boundary conditions.
    cutoff : float, default=1e-12
        Truncation cutoff passed to gate application.
    mpo_ : qtn.TensorNetwork | None, optional
        Optional initial MPO. If omitted, a fresh identity MPO is created.
    dtype : str, default="complex128"
        Dtype used when creating an identity MPO.
    max_bond : int, default=16
        Per-gate local split truncation cap and fallback construction
        compression cap.
    contract : str, default="reduce-split"
        Gate contraction mode. The default uses quimb's reduced two-site split
        path, which is usually cheaper than ``"split"`` for MPO tensors.
    ind_id : str, default="k{}"
        Physical index format used for MPO ket-family indices.

    Returns
    -------
    qtn.TensorNetwork
        Constructed MPO operator.
    """
    entries = _normalize_builder_entries(gates, wheres=wheres, where=where, ndim=1)
    gate_list = [g for g, _ in entries]
    where_list = [w for _, w in entries]

    coords = [int(i) for w in where_list for i in w]
    L = max(coords) + 1

    mpo = mpo_.copy() if mpo_ is not None else id_to_mpo(
        L, phys_dim=2, dtype=dtype, cyclic=cyclic
    )

    for tensor in mpo:
        tensor.modify(data=ar.do("array", tensor.data, like=gate_list[0]))

    for gate_op, where_norm in zip(gate_list, where_list):
        gate_use = _to_ket_gate_layout(gate_op, len(where_norm))

        gate(
            mpo,
            gate_use,
            where_norm,
            ind_id=ind_id,
            cutoff=cutoff,
            contract=contract,
            max_bond=max_bond,
            inplace=True,
            dtype=dtype,
        )

        if mpo.max_bond() > max_bond:
            mpo.compress(
                form="left",
                max_bond=max_bond,
                cutoff=1e-14,
            )

    return mpo


def _normalize_builder_entries(gates, *, wheres=None, where=None, ndim):
    """Normalize builder input to ``[(gate, normalized_where), ...]``."""
    if (where is not None) and (wheres is not None):
        raise ValueError("Pass either 'where' or 'wheres', not both.")

    if where is not None:
        entries = _normalize_gate_entries(gates, where=where, allow_empty=False)
    elif wheres is None:
        entries = _normalize_gate_entries(gates, where=None, allow_empty=False)
    else:
        gate_list = list(gates)
        where_list = list(wheres)
        if not gate_list:
            raise ValueError("gates must not be empty.")
        if len(gate_list) != len(where_list):
            raise ValueError("gates and wheres must have the same length.")
        entries = list(zip(gate_list, where_list))

    normalized = []
    for gate_i, where_i in entries:
        if _is_explicit_index_where(where_i):
            raise ValueError(
                "Explicit index-name where selectors are not supported by "
                "build_mpo_from_gates/build_pepo_from_gates. "
                "Use integer site coordinates."
            )
        if ndim == 1:
            where_norm = _normalize_where_arg_1d(where_i)
        elif ndim == 2:
            where_norm = _normalize_where_arg_2d(where_i)
        else:
            raise ValueError("ndim must be 1 or 2.")
        normalized.append((gate_i, where_norm))

    return normalized


def _apply_gate_1d(
    tn,
    gate,
    where,
    ind_id="k{}",
    site_tags="I{}",
    cutoff=1.e-12,
    contract="split-gate",
    inplace=True,
    *,
    dtype="complex128",
    cutoff_mode="rsum2",
    max_bond=None,
    path_canonize=False,
    path_canonize_distance=1,
    path_canonize_opts=None,
    path_compress=False,
    path_compress_max_bond=None,
    path_compress_cutoff=None,
    path_compress_canonize_distance=0,
    path_compress_opts=None,
):
    """Apply a single normalized 1D gate on one or two sites."""

    G_apply = gate
    split_opts = _local_split_opts(max_bond)

    if len(where) == 2:
        x, y = where
        x = int(x)
        y = int(y)

        if x == y:
            raise ValueError("where must contain distinct site indices for two-site gates.")

        route_with_swaps = isinstance(contract, str) and (contract in {"split", "reduce-split"})

        if route_with_swaps:
            backend_sample = resolve_backend_sample_data_from_tn(tn)
            inferred_converter = infer_backend_converter_from_sample(
                backend_sample,
                cast_complex_to_real=True,
            )
            path_pairs = list(gen_long_range_swap_path_1d(x, y))
            *swaps, final = path_pairs
            _maybe_canonize_path(
                tn,
                path_pairs,
                path_canonize=path_canonize,
                path_canonize_distance=path_canonize_distance,
                path_canonize_opts=path_canonize_opts,
            )

            for i_, j_ in swaps:
                swap = _swap_gate_for_site_pair(
                    tn,
                    i_,
                    j_,
                    ind_id=ind_id,
                    dtype=dtype,
                    inferred_converter=inferred_converter,
                )
                tn = qtn.tensor_network_gate_inds(
                    tn,
                    swap,
                    [_format_ind_id(ind_id, i_), _format_ind_id(ind_id, j_)],
                    contract=contract,
                    inplace=inplace,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    **split_opts,
                )

            i_, j_ = final
            tn = qtn.tensor_network_gate_inds(
                tn,
                G_apply,
                [_format_ind_id(ind_id, i_), _format_ind_id(ind_id, j_)],
                contract=contract,
                inplace=inplace,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                **split_opts,
            )

            for i_, j_ in reversed(swaps):
                swap = _swap_gate_for_site_pair(
                    tn,
                    i_,
                    j_,
                    ind_id=ind_id,
                    dtype=dtype,
                    inferred_converter=inferred_converter,
                )
                tn = qtn.tensor_network_gate_inds(
                    tn,
                    swap,
                    [_format_ind_id(ind_id, i_), _format_ind_id(ind_id, j_)],
                    contract=contract,
                    inplace=inplace,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    **split_opts,
                )
            _maybe_compress_path(
                tn,
                path_pairs,
                path_compress=path_compress,
                max_bond=path_compress_max_bond if path_compress_max_bond is not None else max_bond,
                cutoff=path_compress_cutoff if path_compress_cutoff is not None else cutoff,
                path_compress_canonize_distance=path_compress_canonize_distance,
                path_compress_opts=path_compress_opts,
            )
        else:
            path_pairs = [(x, y)]
            _maybe_canonize_path(
                tn,
                path_pairs,
                path_canonize=path_canonize,
                path_canonize_distance=path_canonize_distance,
                path_canonize_opts=path_canonize_opts,
            )
            tn = qtn.tensor_network_gate_inds(
                tn,
                G_apply,
                [_format_ind_id(ind_id, x), _format_ind_id(ind_id, y)],
                contract=contract,
                inplace=inplace,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                **split_opts,
            )
            _maybe_compress_path(
                tn,
                path_pairs,
                path_compress=path_compress,
                max_bond=path_compress_max_bond if path_compress_max_bond is not None else max_bond,
                cutoff=path_compress_cutoff if path_compress_cutoff is not None else cutoff,
                path_compress_canonize_distance=path_compress_canonize_distance,
                path_compress_opts=path_compress_opts,
            )

            # Add site tags after the gate has been applied.
            tensor_x = [tn.tensor_map[i] for i in tn.ind_map[_format_ind_id(ind_id, x)]][0]
            tensor_x.add_tag(site_tags.format(x))
            tensor_y = [tn.tensor_map[i] for i in tn.ind_map[_format_ind_id(ind_id, y)]][0]
            tensor_y.add_tag(site_tags.format(y))

    elif len(where) == 1:
        (x,) = where
        tn = qtn.tensor_network_gate_inds(
            tn,
            G_apply,
            [_format_ind_id(ind_id, x)],
            contract=contract,
            inplace=inplace,
            **split_opts,
        )

    else:
        raise ValueError("where must contain one or two site indices.")

    return tn



def gate_with_submpo(
    p,
    submpo,
    where=None,
    which="upper",
    method="direct",
    transpose=False,
    tags=None,
    info=None,
    inplace=False,
    inplace_mpo=True,
    ind_id_k="k{}",
    ind_id_b="b{}",
    **compress_opts,
):
    """Apply a sub-MPO to the upper (ket) or lower (bra) layer of an MPS/MPO.

    The gate region is canonicalized, the sub-MPO is lazily absorbed, the
    affected region is compressed via ``tensor_network_1d_compress``, and the
    physical indices are re-mapped back to their canonical names.

    Parameters
    ----------
    p : qtn.MatrixProductState or qtn.MatrixProductOperator
        The target MPS/MPO to apply the sub-MPO to.
    submpo : qtn.MatrixProductOperator
        The sub-MPO to apply.
    where : tuple[int], optional
        The site indices where the sub-MPO acts.
    which : {"upper", "lower"}, default="upper"
        Whether to absorb into the upper (ket) or lower (bra) layer.
    method : str, default="direct"
        Compression method passed to ``tensor_network_1d_compress``.
    transpose : bool, default=False
        Whether to transpose the sub-MPO before application.
    tags : sequence[str] | None, optional
        Tags to add to the gate tensors (currently unused, reserved for
        future alignment with quimb's ``gate_upper``/``gate_lower`` API).
    info : dict | None, optional
        If provided, ``cur_orthog`` is written back after compression.
    inplace : bool, default=False
        Whether to modify ``p`` in place.
    inplace_mpo : bool, default=True
        Whether to modify ``submpo`` in place during absorption.
    ind_id_k : str, default="k{}"
        Format string for ket-family physical index names.
    ind_id_b : str, default="b{}"
        Format string for bra-family physical index names.
    **compress_opts :
        Additional options forwarded to ``tensor_network_1d_compress``.

    Returns
    -------
    p : qtn.MatrixProductState or qtn.MatrixProductOperator
        The updated MPS/MPO after sub-MPO application and compression.
    """
    which_norm = str(which).strip().lower()
    if which_norm not in ("upper", "lower"):
        raise ValueError("which must be 'upper' or 'lower' (case-insensitive).")

    p = p if inplace else p.copy()
    si, sf = min(where), max(where)

    # make the region canonical
    p.canonicalize_((si, sf), info=info)

    # lazily absorb the sub-MPO into the selected layer
    if which_norm == "upper":
        p.gate_upper_with_op_lazy_(submpo, transpose=transpose, inplace=inplace_mpo)
        ind_id = ind_id_k
        other_prefix = ind_id_b.replace("{}", "").rstrip("{}")
    else:
        p.gate_lower_with_op_lazy_(submpo, transpose=transpose, inplace=inplace_mpo)
        ind_id = ind_id_b
        other_prefix = ind_id_k.replace("{}", "").rstrip("{}")

    # split off and compress the affected region
    sub_site_tags = [p.site_tag(s) for s in range(si, sf + 1)]
    _, subpsi = p.partition(sub_site_tags, which="any", inplace=True)
    qtn.tensor_network_1d_compress(
        subpsi,
        site_tags=sub_site_tags,
        method=method,
        # the sub TN can't be automatically permuted when missing sites
        permute_arrays=False,
        inplace=True,
        **compress_opts,
    )

    if info is not None:
        if compress_opts.get("sweep_reverse", False):
            info["cur_orthog"] = (sf, sf)
        else:
            info["cur_orthog"] = (si, si)

    # recombine and remap physical indices back to canonical names
    p |= subpsi
    outer_inds = set(p.outer_inds())
    mapping = {
        ind: _format_ind_id(ind_id, i)
        for i in range(p.L)
        for ind in p[f"I{i}"].inds
        if ind in outer_inds and not ind.startswith(other_prefix)
    }
    p.reindex_(mapping)
    return p


# Thin backward-compatible wrappers kept for external callers.
def gate_with_submpo_upper(p, submpo, where=None, method="direct", transpose=False,
                           info=None, inplace=False, inplace_mpo=True, **compress_opts):
    """Apply a sub-MPO to the upper (ket) layer. Delegates to :func:`gate_with_submpo`."""
    return gate_with_submpo(
        p, submpo, where=where, which="upper", method=method, transpose=transpose,
        info=info, inplace=inplace, inplace_mpo=inplace_mpo, **compress_opts,
    )


def gate_with_submpo_lower(p, submpo, where=None, method="direct", transpose=False,
                           info=None, inplace=False, inplace_mpo=True, **compress_opts):
    """Apply a sub-MPO to the lower (bra) layer. Delegates to :func:`gate_with_submpo`."""
    return gate_with_submpo(
        p, submpo, where=where, which="lower", method=method, transpose=transpose,
        info=info, inplace=inplace, inplace_mpo=inplace_mpo, **compress_opts,
    )


def gate_nonlocal_opt(
    p,
    G,
    where,
    dims=None,
    which="upper",
    method="direct",
    tags=None,
    info=None,
    inplace=False,
    ind_id_k="k{}",
    ind_id_b="b{}",
    **compress_opts,
):
    """Apply a nonlocal gate (dense operator) to an MPO layer via sub-MPO compression.

    The gate ``G`` is converted to a :class:`~quimb.tensor.MatrixProductOperator`,
    absorbed lazily into the requested layer of ``p``, and the affected region is
    compressed using ``tensor_network_1d_compress``.  This keeps bond dimension
    controlled without the unbounded growth that quimb's native ``gate_upper`` /
    ``gate_lower`` can produce.

    Parameters
    ----------
    p : qtn.MatrixProductOperator
        The target MPO to apply the gate to.
    G : array_like
        Dense gate operator with shape ``(d**n, d**n)`` or ``(d,)*2n``.
    where : tuple[int]
        Site indices the gate acts on.
    dims : tuple[int] | None, optional
        Physical dimensions for each site in ``where``.  Inferred from ``p``
        when ``None``.
    which : {"upper", "lower"}, default="upper"
        Which layer to apply the gate to — consistent with quimb's
        ``gate_upper`` / ``gate_lower`` naming.
    method : str, default="direct"
        Compression method passed to ``tensor_network_1d_compress``.
    tags : sequence[str] | None, optional
        Tags forwarded to the sub-MPO tensors (reserved for future use).
    info : dict | None, optional
        If provided, ``cur_orthog`` is written back after compression.
    inplace : bool, default=False
        Whether to modify ``p`` in place.
    ind_id_k : str, default="k{}"
        Format string for ket-family physical index names.
    ind_id_b : str, default="b{}"
        Format string for bra-family physical index names.
    **compress_opts :
        Additional options forwarded to ``tensor_network_1d_compress``
        (e.g. ``max_bond``, ``cutoff``, ``cutoff_mode``).

    Returns
    -------
    p : qtn.MatrixProductOperator
        The updated MPO after gate application and compression.
    """
    if dims is None:
        dims = tuple(p.phys_dim(i) for i in where)
    submpo = qtn.MatrixProductOperator.from_dense(G, dims=dims, sites=where, L=p.L)
    return gate_with_submpo(
        p,
        submpo,
        where=where,
        which=which,
        method=method,
        tags=tags,
        info=info,
        inplace=inplace,
        inplace_mpo=True,
        ind_id_k=ind_id_k,
        ind_id_b=ind_id_b,
        **compress_opts,
    )
