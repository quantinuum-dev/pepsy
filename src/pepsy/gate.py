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

__all__ = [
    "gate_2d",
    "gate_to_pepo",
    "gate_1d",
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


def _resolve_backend_sample_data(gate):
    """Return representative array data from a gate-like object."""
    if hasattr(gate, "shape") and hasattr(gate, "dtype"):
        return gate
    data = getattr(gate, "data", None)
    if hasattr(data, "shape") and hasattr(data, "dtype"):
        return data
    return None


def _infer_backend_converter(sample_data):
    """Infer array converter callable from sample tensor data."""
    if sample_data is None:
        return None

    backend = ar.infer_backend(sample_data)

    if backend == "numpy":
        dtype_name = ar.get_dtype_name(sample_data)
        try:
            dtype_np = np.dtype(dtype_name)
        except TypeError:
            return np.asarray
        return lambda x, dtype=dtype_np: np.asarray(x, dtype=dtype)

    if backend == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        target_dtype = getattr(sample_data, "dtype", None)
        target_device = getattr(sample_data, "device", None)

        def _to_torch(x, dtype=target_dtype, device=target_device):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if device is not None:
                kwargs["device"] = device
            return torch.as_tensor(np.array(x, copy=True), **kwargs)

        return _to_torch

    return None


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
    """Return a one-qubit phase gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.u1_gate_param_gen([theta])


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


def cu1(theta):
    """Return a two-qubit controlled-U1 gate for angle ``theta``.

    Parameters
    ----------
    theta : float
        Rotation angle.
    """
    return qtn.circuit.cu1_param_gen([theta])


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
    ij_a, ij_b, sequence=None
):
    """Generate a SWAP path that brings two lattice sites together."""
    ia, ja = ij_a
    ib, jb = ij_b
    di = ib - ia
    dj = jb - ja

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

    def apply_move(move):
        nonlocal ij_a, ij_b, ia, ja, ib, jb, di, dj
        if (move == "av") and (di != 0):
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            pair = (ij_a, new_ij_a)
            ij_a = new_ij_a
            ia += istep
            di -= istep
            return pair

        if (move == "bv") and (di != 0):
            istep = min(max(di, -1), +1)
            new_ij_b = (ib - istep, jb)
            pair = (ij_a, ij_b) if (new_ij_b == ij_a) else (ij_b, new_ij_b)
            ij_b = new_ij_b
            ib -= istep
            di -= istep
            return pair

        if (move == "ah") and (dj != 0):
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, ja + jstep)
            pair = (ij_a, new_ij_a)
            ij_a = new_ij_a
            ja += jstep
            dj -= jstep
            return pair

        if (move == "bh") and (dj != 0):
            jstep = min(max(dj, -1), +1)
            new_ij_b = (ib, jb - jstep)
            pair = (ij_a, ij_b) if (new_ij_b == ij_a) else (ij_b, new_ij_b)
            ij_b = new_ij_b
            jb -= jstep
            dj -= jstep
            return pair

        return None

    if sequence in {"x_then_y", "xy", "y_then_x", "yx"}:
        axis_order = ("x", "y") if sequence in {"x_then_y", "xy"} else ("y", "x")

        for axis in axis_order:
            if axis == "x":
                while di != 0:
                    istep = min(max(di, -1), +1)
                    new_ij_a = (ia + istep, ja)
                    yield (ij_a, new_ij_a)
                    ij_a = new_ij_a
                    ia += istep
                    di -= istep
            else:
                while dj != 0:
                    jstep = min(max(dj, -1), +1)
                    new_ij_a = (ia, ja + jstep)
                    yield (ij_a, new_ij_a)
                    ij_a = new_ij_a
                    ja += jstep
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


def apply_2dtn_(
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
    to_backend=None,
    sequence=("av", "bh", "ah", "bv"),
    ind_id="k{},{}",
):
    """Apply a local gate to a PEPS, routing long-range gates with SWAPs."""

    if bra or (canonize_distance != 2):
        warnings.warn(
            "Unused options in apply_2dtn_: 'bra' and/or 'canonize_distance'.",
            RuntimeWarning,
            stacklevel=2,
        )

    if tags is None:
        tags = ["G"]

    swap = qu.swap(dim=2, dtype=dtype).reshape(2, 2, 2, 2)
    if to_backend:
        swap = to_backend(swap)
    else:
        inferred_converter = _infer_backend_converter(_resolve_backend_sample_data(G))
        if inferred_converter is not None:
            swap = inferred_converter(swap)

    if len(where) == 1:
        ((i, j),) = where
        qtn.tensor_network_gate_inds(
            peps,
            G,
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

    *swaps, final = gen_long_range_swap_path(x, y, sequence=sequence)

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
        G,
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
    """Normalize one-site and two-site where specs for :func:`gate_2d`."""
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


def gate_2d(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
    to_backend=None,
    sequence=("av", "bh", "ah", "bv"),
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
        "to_backend": to_backend,
        "sequence": sequence,
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
        peps = apply_2dtn_(peps, gate, where_norm, **opts)

    if chi is not None:
        chi_value = int(chi)
        if chi_value <= 0:
            raise ValueError("chi must be a positive integer when provided.")
        peps.compress_all_(max_bond=chi_value, cutoff=chi_cutoff)

    return peps



def pepo_identity(lx, ly, dtype="complex128"):
    """Create bond-dimension-1 PEPO identity on an ``lx x ly`` lattice."""
    pepo = qtn.PEPO.rand(Lx=lx, Ly=ly, bond_dim=1, seed=666, dtype=dtype)
    eye = qu.pauli("I", dtype=dtype)

    for tensor in pepo:
        ndim = len(tensor.data.shape)
        if ndim == 4:
            data = np.zeros([1, 1, 2, 2], dtype=dtype)
            data[0, 0, :, :] = eye
            tensor.modify(data=data)
        elif ndim == 5:
            data = np.zeros([1, 1, 1, 2, 2], dtype=dtype)
            data[0, 0, 0, :, :] = eye
            tensor.modify(data=data)
        elif ndim == 6:
            data = np.zeros([1, 1, 1, 1, 2, 2], dtype=dtype)
            data[0, 0, 0, 0, :, :] = eye
            tensor.modify(data=data)

    return pepo


def peps_cycle(peps, bond_dim, cylinder=False):
    """Add periodic bonds; wrapper around :func:`pepsy.core.add_cycle`."""
    from .core import add_cycle  # pylint: disable=import-outside-toplevel

    return add_cycle(peps, bond_dim=bond_dim, cylinder=cylinder)


def gate_to_pepo(
    gates,
    cyclic=False,
    cutoff=1.0e-12,
    pepo_=None,
    dtype="complex128",
    bnd=32,
    sequence=("av", "bh", "ah", "bv"),
    contract="split",
    compress_threshold=16,
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
        pepo = peps_cycle(pepo, 1)

    for tensor in pepo:
        tensor.modify(data=ar.do("array", tensor.data, like=gate_list[0]))

    for G, where_norm in zip(gate_list, where_list):
        n = len(where_norm)
        if n == 1:
            gate_use = ar.do("transpose", G, (1, 0))
        elif n == 2:
            gate_use = ar.do("transpose", G, (2, 3, 0, 1))
        else:
            raise ValueError("where must contain one or two site coordinates.")

        apply_2dtn_(
            pepo, gate_use, where_norm,
            bond_dim=bnd, bra=False, contract=contract,
            tags=[], dtype=dtype, cutoff=cutoff,
            sequence=sequence, ind_id="b{},{}",
        )

        if pepo.max_bond() > compress_threshold:
            pepo.compress_all(
                inplace=True,
                max_bond=compress_threshold,
                canonize_distance=4,
                cutoff=1e-14,
            )

    return pepo


def gate_1d(
    tn,
    where,
    G,
    ind_id="k{}",
    site_tags="I{}",
    cutoff=1.e-12,
    contract="split-gate",
    inplace=False,
):

    """
    Apply a 1D gate to a tensor network at one or two sites.

    Args:
        tn:      Tensor network (quimb/qtn TensorNetwork).
        where:   Iterable of site indices; length 1 (single-qubit) or 2 (two-qubit).
        G:       Gate tensor (or matrix).
        ind_id: Format string for site indices (e.g., "k{}" -> "k3").
        site_tags: Format string for site tags   (e.g., "I{}" -> "I3").
        cutoff:  SVD cutoff (used for split contraction paths).
        contract: Contraction mode (e.g., "split-gate") or bool for single-qubit.
        inplace: Modify tn in place if True; otherwise return a new TN.

    Returns:
        TensorNetwork with the gate applied and site tags added.
    """
    if len(where) == 2:
        x, y = where
        tn = qtn.tensor_network_gate_inds(
            tn,
            G,
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
            G,
            [ind_id.format(x)],
            contract=True,
            inplace=inplace,
        )

    else:
        raise ValueError("where must contain one or two site indices.")

    return tn


def energy_global(MPO_origin, mps_a, opt="auto-hq"):  # pylint: disable=invalid-name
    """Compute global energy ``<mps_a|MPO_origin|mps_a>`` with normalization."""

    mps_a_ = mps_a.copy()
    mps_a_.normalize()
    p_h = mps_a_.H
    p_h.reindex_({f"k{i}": f"b{i}" for i in range(mps_a.L)})
    mpo_t = MPO_origin * 1.0

    energy_dmrg = (p_h | mpo_t | mps_a_).contract(all, optimize=opt)
    return energy_dmrg
