"""Gate-application utilities for 2D PEPS networks."""

from __future__ import annotations

from numbers import Integral
import random
import warnings
from itertools import count

import autoray as ar
import numpy as np
import quimb as qu
import quimb.tensor as qtn

from ._backend_utils import (
    infer_backend_converter_from_sample,
    resolve_backend_sample_data,
    resolve_backend_sample_data_from_tn,
)
from .core import add_cycle, pepo_identity

__all__ = [
    "gate_tn_2d",
    "gates_tn_2d",
    "build_pepo_from_gates",
    "gate_tn_1d",
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
]


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


def gen_long_range_swap_path(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
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


def gate_tn_2d(
    peps,
    G,
    where,
    *,
    bond_dim=None,
    bra=False,
    contract="split",
    tags=None,
    dtype="complex128",
    cutoff=1.0e-12,
    canonize_distance=2,
    sequence=("av", "bh", "ah", "bv"),
    cyclic=False,
    Lx=None,
    Ly=None,
    ind_id="k{},{}",
):
    """Apply a local gate to a PEPS, routing long-range gates with SWAPs."""

    if bra or (canonize_distance != 2):
        warnings.warn(
            "Unused options in gate_tn_2d: 'bra' and/or 'canonize_distance'.",
            RuntimeWarning,
            stacklevel=2,
        )

    if tags is None:
        tags = ["G"]

    backend_sample = resolve_backend_sample_data_from_tn(peps)
    if backend_sample is None:
        backend_sample = resolve_backend_sample_data(G)
    inferred_converter = infer_backend_converter_from_sample(backend_sample)

    G_apply = G
    if inferred_converter is not None:
        try:
            G_apply = inferred_converter(G)
        except (TypeError, ValueError):
            G_apply = G

    swap = qu.swap(dim=2, dtype=dtype).reshape(2, 2, 2, 2)
    if inferred_converter is not None:
        swap = inferred_converter(swap)

    if len(where) == 1:
        ((i, j),) = where
        qtn.tensor_network_gate_inds(
            peps,
            G_apply,
            [ind_id.format(i, j)],
            contract=True,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
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
        qtn.tensor_network_gate_inds(
            peps,
            G_apply,
            [ind_id.format(i, j), ind_id.format(m, n)],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
        )
        return peps

    lx_use = Lx
    ly_use = Ly
    if cyclic and (lx_use is None or ly_use is None):
        lx_use = getattr(peps, "Lx", lx_use)
        ly_use = getattr(peps, "Ly", ly_use)

    *swaps, final = gen_long_range_swap_path(
        x,
        y,
        sequence=sequence,
        cyclic=cyclic,
        Lx=lx_use,
        Ly=ly_use,
    )

    for pair in swaps:
        x_, y_ = pair
        i_, j_ = x_
        m_, n_ = y_
        qtn.tensor_network_gate_inds(
            peps,
            swap,
            [ind_id.format(i_, j_), ind_id.format(m_, n_)],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
            max_bond=bond_dim,
        )

    x_, y_ = final
    i_, j_ = x_
    m_, n_ = y_
    qtn.tensor_network_gate_inds(
        peps,
        G_apply,
        [ind_id.format(i_, j_), ind_id.format(m_, n_)],
        contract=contract,
        tags=tags,
        info=None,
        inplace=True,
        cutoff=cutoff,
    )

    for pair in reversed(swaps):
        x_, y_ = pair
        i_, j_ = x_
        m_, n_ = y_
        qtn.tensor_network_gate_inds(
            peps,
            swap,
            [ind_id.format(i_, j_), ind_id.format(m_, n_)],
            contract=contract,
            tags=tags,
            info=None,
            inplace=True,
            cutoff=cutoff,
        )

    return peps


def _is_lattice_coord(value):
    return (
        isinstance(value, (tuple, list))
        and (len(value) == 2)
        and all(isinstance(v, Integral) for v in value)
    )


def _normalize_where_arg(where):
    """Normalize one-site and two-site where specs for :func:`gates_tn_2d`."""
    if _is_lattice_coord(where):
        i, j = where
        return ((int(i), int(j)),)

    if isinstance(where, (tuple, list)):
        if len(where) == 1 and _is_lattice_coord(where[0]):
            i, j = where[0]
            return ((int(i), int(j)),)
        if len(where) == 2 and _is_lattice_coord(where[0]) and _is_lattice_coord(where[1]):
            i0, j0 = where[0]
            i1, j1 = where[1]
            return ((int(i0), int(j0)), (int(i1), int(j1)))

    raise ValueError(
        "Invalid where specification. Expected (i, j), ((i, j),), or ((i0, j0), (i1, j1))."
    )


def gates_tn_2d(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    peps,
    gates,
    *,
    bond_dim=None,
    bra=False,
    contract="split",
    tags=None,
    dtype="complex128",
    cutoff=1.0e-12,
    canonize_distance=2,
    sequence=("av", "bh", "ah", "bv"),
    cyclic=False,
    Lx=None,
    Ly=None,
    chi=None,
    chi_cutoff=1.0e-12,
):
    """Apply an iterable of gates to a PEPS in order.

    Parameters
    ----------
    peps : qtn.TensorNetwork
        PEPS state modified in place.
    gates : iterable
        Each item can be:
        ``(where, gate)`` or ``(where, gate, kwargs_dict)``.

        - ``where`` accepts ``(i, j)``, ``((i, j),)``, or
          ``((i0, j0), (i1, j1))``.
        - ``kwargs_dict`` optionally overrides per-gate apply options.
    chi : int | None, default=None
        If provided, perform a final ``peps.compress_all_`` with
        ``max_bond=chi`` after all gates are applied.
    chi_cutoff : float, default=1e-12
        Cutoff forwarded to final ``compress_all_`` when ``chi`` is set.

    Returns
    -------
    qtn.TensorNetwork
        The same ``peps`` object, updated in place.
    """
    base_opts = {
        "bond_dim": bond_dim,
        "bra": bra,
        "contract": contract,
        "tags": tags,
        "dtype": dtype,
        "cutoff": cutoff,
        "canonize_distance": canonize_distance,
        "sequence": sequence,
        "cyclic": cyclic,
        "Lx": Lx,
        "Ly": Ly,
    }

    for idx, item in enumerate(gates):
        if isinstance(item, dict):
            where = item.get("where")
            gate = item.get("gate", item.get("G"))
            gate_opts = dict(item.get("kwargs", {}))
        else:
            if not isinstance(item, (tuple, list)):
                raise TypeError(
                    f"Gate spec at position {idx} must be tuple/list/dict; got {type(item).__name__}."
                )
            if len(item) == 2:
                where, gate = item
                gate_opts = {}
            elif len(item) == 3:
                where, gate, gate_opts = item
                gate_opts = dict(gate_opts or {})
            else:
                raise ValueError(
                    f"Gate spec at position {idx} must have length 2 or 3; got {len(item)}."
                )

        if gate is None:
            raise ValueError(f"Gate spec at position {idx} has no gate tensor.")

        where_norm = _normalize_where_arg(where)
        opts = dict(base_opts)
        opts.update(gate_opts)
        peps = gate_tn_2d(peps, gate, where_norm, **opts)

    if chi is not None:
        chi_value = int(chi)
        if chi_value <= 0:
            raise ValueError("chi must be a positive integer when provided.")
        peps.compress_all_(max_bond=chi_value, cutoff=chi_cutoff)

    return peps

def build_pepo_from_gates(
    gates,
    cyclic=False,
    cutoff=1.0e-12,
    pepo_=None,
    dtype="complex128",
    bnd=32,
    sequence=("av", "bh", "ah", "bv"),
    contract="split",
    compress_threshold=16,
    ind_id="k{},{}",
):
    """Build a PEPO by applying ``(where, G)`` gate pairs onto a PEPO identity.

    ``gates`` is a list of ``(where, G)`` where ``where`` is already in
    normalized form: ``((i, j),)`` for single-site or
    ``((i0, j0), (i1, j1))`` for two-site gates.
    Lx / Ly are inferred from the gate coordinates.
    """
    gates = list(gates)
    if not gates:
        raise ValueError("gates must not be empty.")

    where_list = [where for where, _ in gates]
    gate_list  = [G    for _,     G in gates]

    coords = [c for w in where_list for c in w]
    Lx = max(i for i, _ in coords) + 1
    Ly = max(j for _, j in coords) + 1

    pepo = pepo_.copy() if pepo_ is not None else pepo_identity(Lx, Ly, dtype=dtype)
    if pepo_ is None and cyclic:
        pepo = add_cycle(pepo, 1)

    for tensor in pepo:
        tensor.modify(data=ar.do("array", tensor.data, like=gate_list[0]))

    for G, where_norm in zip(gate_list, where_list):
        gate_use = _to_ket_gate_layout(G, len(where_norm))

        gate_tn_2d(
            pepo, gate_use, where_norm,
            bond_dim=bnd, bra=False, contract=contract,
            tags=[], dtype=dtype, cutoff=cutoff,
            sequence=sequence, cyclic=cyclic, Lx=Lx, Ly=Ly, ind_id=ind_id,
        )

        if pepo.max_bond() > compress_threshold:
            pepo.compress_all(
                inplace=True,
                max_bond=compress_threshold,
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
    cyclic=False,
    cutoff=1.0e-12,
    mpo_=None,
    dtype="complex128",
    bnd=32,
    contract="split",
    compress_threshold=16,
    ind_id="k{}",
):
    """Build an MPO by applying ``(where, G)`` gates onto an MPO identity.

    ``gates`` is a list of ``(where, G)`` where ``where`` is
    ``(i,)`` for one-qubit or ``(i, j)`` for two-qubit gates.
    """
    gates = list(gates)
    if not gates:
        raise ValueError("gates must not be empty.")

    where_list = [tuple(where) for where, _ in gates]
    gate_list = [G for _, G in gates]

    coords = [int(i) for w in where_list for i in w]
    L = max(coords) + 1

    mpo = mpo_.copy() if mpo_ is not None else qtn.MPO_identity(
        L, phys_dim=2, dtype=dtype, cyclic=cyclic
    )

    for tensor in mpo:
        tensor.modify(data=ar.do("array", tensor.data, like=gate_list[0]))

    for G, where_norm in zip(gate_list, where_list):
        gate_use = _to_ket_gate_layout(G, len(where_norm))

        gate_tn_1d(
            mpo,
            where_norm,
            gate_use,
            ind_id=ind_id,
            cutoff=cutoff,
            contract=contract,
            inplace=True,
            dtype=dtype,
        )

        if mpo.max_bond() > compress_threshold:
            mpo.compress(
                form="left",
                max_bond=compress_threshold,
                cutoff=1e-14,
            )

    if bnd is not None:
        mpo.compress(form="left", max_bond=int(bnd), cutoff=cutoff)

    return mpo


def gate_tn_1d(
    tn,
    where,
    G,
    ind_id="k{}",
    site_tags="I{}",
    cutoff=1.e-12,
    contract="split-gate",
    inplace=False,
    *,
    dtype="complex128",
):

    """Apply a 1D gate to a tensor network at one or two sites.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Tensor network to apply the gate to.
    where : sequence[int]
        Site indices; length 1 (single-qubit) or 2 (two-qubit).
    G : array_like
        Gate tensor (or matrix).
    ind_id : str, default="k{}"
        Format string for site indices (e.g., ``"k{}"`` produces ``"k3"``).
    site_tags : str, default="I{}"
        Format string for site tags (e.g., ``"I{}"`` produces ``"I3"``).
    cutoff : float, default=1e-12
        SVD cutoff used by split contraction paths.
    contract : str | bool, default="split-gate"
        Contraction mode. String modes include ``"split-gate"`` and
        ``"reduce-split"``; ``True`` contracts single-qubit gates directly.
    inplace : bool, default=False
        If ``True``, modify ``tn`` in place; otherwise return a new TN.
    dtype : str, default="complex128"
        Dtype used to build SWAP gates for split-routing modes.

    Returns
    -------
    qtn.TensorNetwork
        Tensor network with the gate applied and site tags added.
    """
    backend_sample = resolve_backend_sample_data_from_tn(tn)
    if backend_sample is None:
        backend_sample = resolve_backend_sample_data(G)
    inferred_converter = infer_backend_converter_from_sample(backend_sample)

    G_apply = G
    if inferred_converter is not None:
        try:
            G_apply = inferred_converter(G)
        except (TypeError, ValueError):
            G_apply = G

    if len(where) == 2:
        x, y = where
        x = int(x)
        y = int(y)

        if x == y:
            raise ValueError("where must contain distinct site indices for two-site gates.")

        route_with_swaps = isinstance(contract, str) and (contract in {"split", "reduce-split"})

        if route_with_swaps:
            swap = qu.swap(dim=2, dtype=dtype).reshape(2, 2, 2, 2)
            if inferred_converter is not None:
                swap = inferred_converter(swap)

            *swaps, final = gen_long_range_swap_path_1d(x, y)

            for i_, j_ in swaps:
                tn = qtn.tensor_network_gate_inds(
                    tn,
                    swap,
                    [ind_id.format(i_), ind_id.format(j_)],
                    contract=contract,
                    inplace=inplace,
                    cutoff=cutoff,
                )

            i_, j_ = final
            tn = qtn.tensor_network_gate_inds(
                tn,
                G_apply,
                [ind_id.format(i_), ind_id.format(j_)],
                contract=contract,
                inplace=inplace,
                cutoff=cutoff,
            )

            for i_, j_ in reversed(swaps):
                tn = qtn.tensor_network_gate_inds(
                    tn,
                    swap,
                    [ind_id.format(i_), ind_id.format(j_)],
                    contract=contract,
                    inplace=inplace,
                    cutoff=cutoff,
                )
        else:
            tn = qtn.tensor_network_gate_inds(
                tn,
                G_apply,
                [ind_id.format(x), ind_id.format(y)],
                contract=contract,
                inplace=inplace,
                cutoff=cutoff,
            )

        # Add site tags after the gate has been applied.
        tensor_x = [tn.tensor_map[i] for i in tn.ind_map[ind_id.format(x)]][0]
        tensor_x.add_tag(site_tags.format(x))
        tensor_y = [tn.tensor_map[i] for i in tn.ind_map[ind_id.format(y)]][0]
        tensor_y.add_tag(site_tags.format(y))

    elif len(where) == 1:
        (x,) = where
        tn = qtn.tensor_network_gate_inds(
            tn,
            G_apply,
            [ind_id.format(x)],
            contract=True,
            inplace=inplace,
        )

    else:
        raise ValueError("where must contain one or two site indices.")

    return tn
