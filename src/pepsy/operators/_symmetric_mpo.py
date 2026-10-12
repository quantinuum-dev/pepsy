"""Symmetric MPO assembly for Hamiltonians and native fermion operators.

Shared charge, basis, and mapping helpers live in ``tensors.symmetric``.
Native graded assembly and Jordan-Wigner compatibility paths retain their
separate construction rules. Symmray is imported only when building arrays.
"""

from __future__ import annotations

import warnings
from itertools import product
from numbers import Integral

import numpy as np
import quimb.tensor as qtn

from ..tensors.symmetric import (
    _apply_to_tensor_network_arrays,
    _as_edges,
    _as_term_where,
    _charge_neg,
    _charge_particle_number,
    _charge_sort_key,
    _charge_sub,
    _dense_numpy,
    _dtype_from_hamiltonian_terms,
    _edge_parameter,
    _expanded_index_charges,
    _fh_spinless_dense_local_ops,
    _is_fermionic_symmray_array,
    _map_edges_to_mpo_indices,
    _map_site_to_mpo_index,
    _node_parameter,
    _normalize_group_charge,
    _require_symmray,
    _resolve_mpo_mapping,
    _term_mapping_uses_coordinate_sites,
    _zero_like_charge,
)


def _term_dense_and_phys_maps(term, *, dtype, reverse=False):
    dense = _dense_numpy(term, dtype=dtype)
    if dense.ndim != 4 or dense.shape[0] != dense.shape[2] or dense.shape[1] != dense.shape[3]:
        raise ValueError(
            "SymHamiltonian.to_mpo requires two-site terms with shape "
            "(da, db, da, db)."
        )

    indices = getattr(term, "indices", None)
    if indices is None or len(indices) != 4:
        raise TypeError("SymHamiltonian.to_mpo requires Symmray rank-4 terms.")

    left_out = _expanded_index_charges(indices[0])
    right_out = _expanded_index_charges(indices[1])
    left_in = _expanded_index_charges(indices[2])
    right_in = _expanded_index_charges(indices[3])

    if reverse:
        dense = dense.transpose(1, 0, 3, 2)
        left_out, right_out = right_out, left_out
        left_in, right_in = right_in, left_in

    if left_out != left_in or right_out != right_in:
        raise ValueError("MPO terms must use matching upper/lower physical charge maps.")

    return dense, left_out, right_out


def _term_dense_and_phys_map(term, *, dtype):
    """Return dense data and the physical charge map for a one-site term."""
    dense = _dense_numpy(term, dtype=dtype)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(
            "SymHamiltonian.to_mpo requires one-site terms with shape (d, d)."
        )

    indices = getattr(term, "indices", None)
    if indices is None or len(indices) != 2:
        raise TypeError("SymHamiltonian.to_mpo requires Symmray rank-2 terms.")
    output = _expanded_index_charges(indices[0])
    input_ = _expanded_index_charges(indices[1])
    if output != input_:
        raise ValueError("One-site terms must use matching upper/lower physical charge maps.")
    return dense, output


def _svd_rank_cutoff(singular_values, shape):
    if singular_values.size == 0:
        return 0.0
    dtype = singular_values.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    eps = np.finfo(dtype).eps
    return eps * max(shape) * float(singular_values[0])


def _decompose_neutral_two_site_term(
    term,
    *,
    symmetry,
    dtype,
    reverse=False,
    fermionic=False,
):
    """Split a neutral two-site operator into charged one-site channels."""
    dense, left_phys, right_phys = _term_dense_and_phys_maps(
        term,
        dtype=dtype,
        reverse=reverse,
    )
    if fermionic:
        # The raw native tensor has its ket axes in fermionic tensor-product
        # order. Converting those two axes to an ordinary site-major MPO
        # inserts the crossing phase (-1) whenever both endpoint ket states
        # have odd particle parity. The Jordan-Wigner string between the
        # endpoints is added separately by the MPO channel below.
        left_odd = np.array(
            [
                _charge_particle_number(charge) % 2 != 0
                for charge in left_phys
            ],
            dtype=bool,
        )
        right_odd = np.array(
            [
                _charge_particle_number(charge) % 2 != 0
                for charge in right_phys
            ],
            dtype=bool,
        )
        crossing = np.ones((len(left_phys), len(right_phys)), dtype=dtype)
        crossing[np.ix_(left_odd, right_odd)] = -1.0
        dense = dense * crossing[None, None, :, :]
    dl, dr, _, _ = dense.shape
    matrix = dense.transpose(0, 2, 1, 3).reshape(dl * dl, dr * dr)

    left_entries = []
    for out_i, out_charge in enumerate(left_phys):
        for in_i, in_charge in enumerate(left_phys):
            charge = _charge_sub(out_charge, in_charge, symmetry)
            left_entries.append((out_i, in_i, charge))

    right_entries = []
    for out_i, out_charge in enumerate(right_phys):
        for in_i, in_charge in enumerate(right_phys):
            charge = _charge_sub(out_charge, in_charge, symmetry)
            right_entries.append((out_i, in_i, charge))

    left_by_charge = {}
    for pos, (_, _, charge) in enumerate(left_entries):
        left_by_charge.setdefault(charge, []).append(pos)
    right_by_charge = {}
    for pos, (_, _, charge) in enumerate(right_entries):
        right_by_charge.setdefault(charge, []).append(pos)

    channels = []
    for left_charge in sorted(left_by_charge, key=_charge_sort_key):
        right_charge = _charge_neg(left_charge, symmetry)
        rows = left_by_charge[left_charge]
        cols = right_by_charge.get(right_charge, ())
        if not cols:
            continue
        block = matrix[np.ix_(rows, cols)]
        if not np.any(block):
            continue

        u, s, vh = np.linalg.svd(block, full_matrices=False)
        rank_cutoff = _svd_rank_cutoff(s, block.shape)
        for rank, singular_value in enumerate(s):
            if float(singular_value) <= rank_cutoff:
                continue
            root = np.sqrt(singular_value)
            left_vec = u[:, rank] * root
            right_vec = root * vh[rank, :]

            left_op = np.zeros((dl, dl), dtype=dtype)
            for entry_pos, value in zip(rows, left_vec):
                out_i, in_i, _ = left_entries[entry_pos]
                left_op[out_i, in_i] = value

            right_op = np.zeros((dr, dr), dtype=dtype)
            for entry_pos, value in zip(cols, right_vec):
                out_i, in_i, _ = right_entries[entry_pos]
                right_op[out_i, in_i] = value

            channels.append((left_charge, left_op, right_op))

    return channels, left_phys, right_phys


def _fermion_parity_operator(phys_map, dtype):
    diag = []
    for charge in phys_map:
        particle_number = _charge_particle_number(charge)
        if particle_number is None:
            raise ValueError("Cannot infer fermionic parity from physical charges.")
        diag.append(-1.0 if particle_number % 2 else 1.0)
    return np.diag(diag).astype(dtype)


def _charged_op_needs_fermion_string(charge):
    particle_number = _charge_particle_number(charge)
    return particle_number is not None and particle_number % 2 != 0


def _add_local_transition(tensor, site, L, left_pos, right_pos, op):
    if L == 1:
        tensor[:, :] += op
    elif site == 0:
        tensor[right_pos, :, :] += op
    elif site == L - 1:
        tensor[left_pos, :, :] += op
    else:
        tensor[left_pos, right_pos, :, :] += op


def _assemble_symmray_mpo(
    *,
    L,
    channels,
    transitions,
    phys_map,
    symmetry,
    zero,
    dtype,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    fermionic=False,
    operator_charge=None,
):
    operator_charge = (
        zero
        if operator_charge is None
        else _normalize_group_charge(operator_charge, symmetry)
    )
    channel_pos = [
        {channel_id: pos for pos, (channel_id, _) in enumerate(cut_channels)}
        for cut_channels in channels
    ]

    arrays = []
    _require_symmray()
    from symmray import utils as sr_utils  # pylint: disable=import-outside-toplevel

    identity = np.eye(len(phys_map), dtype=dtype)
    for site in range(L):
        if L == 1:
            data = np.zeros((len(phys_map), len(phys_map)), dtype=dtype)
            index_maps = [phys_map, phys_map]
            duals = [False, True]
        elif site == 0:
            right_map = [charge for _, charge in channels[site]]
            data = np.zeros((len(right_map), len(phys_map), len(phys_map)), dtype=dtype)
            index_maps = [right_map, phys_map, phys_map]
            duals = [False, False, True]
            _add_local_transition(data, site, L, None, channel_pos[site][("start",)], identity)
        elif site == L - 1:
            left_map = [charge for _, charge in channels[site - 1]]
            data = np.zeros((len(left_map), len(phys_map), len(phys_map)), dtype=dtype)
            index_maps = [left_map, phys_map, phys_map]
            duals = [True, False, True]
            _add_local_transition(data, site, L, channel_pos[site - 1][("done",)], None, identity)
        else:
            left_map = [charge for _, charge in channels[site - 1]]
            right_map = [charge for _, charge in channels[site]]
            data = np.zeros(
                (len(left_map), len(right_map), len(phys_map), len(phys_map)),
                dtype=dtype,
            )
            index_maps = [left_map, right_map, phys_map, phys_map]
            duals = [True, False, False, True]
            _add_local_transition(
                data,
                site,
                L,
                channel_pos[site - 1][("start",)],
                channel_pos[site][("start",)],
                identity,
            )
            _add_local_transition(
                data,
                site,
                L,
                channel_pos[site - 1][("done",)],
                channel_pos[site][("done",)],
                identity,
            )

        for left_id, right_id, op in transitions[site]:
            left_pos = None if site == 0 else channel_pos[site - 1][left_id]
            right_pos = None if site == L - 1 else channel_pos[site][right_id]
            _add_local_transition(data, site, L, left_pos, right_pos, op)

        arrays.append(
            sr_utils.from_dense(
                data,
                symmetry=symmetry,
                index_maps=index_maps,
                duals=duals,
                fermionic=bool(fermionic),
                charge=operator_charge if site == L - 1 else zero,
                label=(site if fermionic and site == L - 1 and
                       _charged_op_needs_fermion_string(operator_charge)
                       else None),
            )
        )

    mpo = qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
    )
    raw_bond = mpo.max_bond()
    raw_max_bond = 1 if raw_bond is None else int(raw_bond)
    did_compress = bool(compress and L > 1)
    if compress and L > 1:
        compress_opts = {"cutoff": cutoff}
        if max_bond is not None:
            compress_opts["max_bond"] = int(max_bond)
        mpo.compress(**compress_opts)
    if to_backend is not None:
        # Cast after compression so the SVD-based bond truncation runs in the
        # stable build precision (e.g. complex128). Converting first and then
        # compressing runs the SVD in the target precision, which for a
        # near-singular Hamiltonian MPO in complex64 can hit non-finite values.
        _apply_to_tensor_network_arrays(mpo, to_backend)

    requested_max_bond = None if max_bond is None else int(max_bond)
    final_bond = mpo.max_bond()
    final_max_bond = 1 if final_bond is None else int(final_bond)
    report = {
        "compressed": did_compress,
        "cutoff": cutoff,
        "requested_max_bond": requested_max_bond,
        "raw_max_bond": raw_max_bond,
        "final_max_bond": final_max_bond,
        "rank_reduced": final_max_bond < raw_max_bond,
        "cap_bound": (
            did_compress
            and requested_max_bond is not None
            and raw_max_bond > requested_max_bond
        ),
        "max_bond_exceeded": (
            did_compress
            and requested_max_bond is not None
            and final_max_bond > requested_max_bond
        ),
    }
    # This record describes MPO construction only; it is not used during
    # contraction and can safely travel with the returned MPO as user-facing
    # build metadata.
    mpo.pepsy_compression_report = report
    if report["max_bond_exceeded"]:
        warnings.warn(
            "SymHamiltonian.to_mpo requested "
            f"max_bond={requested_max_bond}, but Symmray compression returned "
            f"max bond {final_max_bond}. Tied singular values at the "
            "truncation threshold can make this a soft cap; inspect "
            "mpo.pepsy_compression_report before relying on a hard memory "
            "limit.",
            RuntimeWarning,
            stacklevel=3,
        )
    return mpo


def _build_factorized_pair_mpo(
    *,
    L,
    pair_create,
    pair_annihilate,
    onsite,
    signs,
    normalization,
    symmetry,
    max_bond=None,
    cutoff=1e-12,
    compress=False,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    dtype=None,
):
    """Build a compact native MPO for a staggered pair structure factor.

    The operator is assembled from the finite-state identity

    ``normalization * ((sum_i s_i D_i^dagger) (sum_j s_j D_j)
    - sum_i D_i^dagger D_i)``.

    The explicit two-site expansion has ``O(L**2)`` terms, but this automaton
    has only two open pair channels plus start/done.  In particular, it does
    not first create one MPO channel per pair and then try to compress the
    result.  The local operators are native Symmray fermionic arrays; the
    graded evaluator therefore remains responsible for the MPO--MPS signs.
    """
    if not isinstance(L, Integral) or int(L) < 1:
        raise ValueError("L must be a positive integer.")
    L = int(L)
    signs = tuple(signs)
    if len(signs) != L:
        raise ValueError(f"signs must contain exactly L={L} values.")
    normalization = np.asarray(normalization).reshape(())
    if not np.isfinite(normalization):
        raise ValueError("normalization must be finite.")
    dtype = np.dtype(dtype or np.result_type(
        _dense_numpy(pair_create).dtype,
        _dense_numpy(pair_annihilate).dtype,
        _dense_numpy(onsite).dtype,
        np.asarray(signs).dtype,
        normalization.dtype,
    ))

    create_dense, phys_map = _term_dense_and_phys_map(
        pair_create,
        dtype=dtype,
    )
    annihilate_dense, annihilate_phys_map = _term_dense_and_phys_map(
        pair_annihilate,
        dtype=dtype,
    )
    _, onsite_phys_map = _term_dense_and_phys_map(
        onsite,
        dtype=dtype,
    )
    if phys_map != annihilate_phys_map or phys_map != onsite_phys_map:
        raise ValueError(
            "factorized pair MPO operators must share one physical charge map."
        )

    # Preserve the charge rank for product symmetries (e.g. ``U1U1``); a
    # scalar ``0`` is not equal to the neutral tuple ``(0, 0)``.
    zero = _normalize_group_charge(
        _zero_like_charge(getattr(pair_create, "charge", 0)),
        symmetry,
    )
    create_charge = _normalize_group_charge(
        getattr(pair_create, "charge", zero), symmetry
    )
    annihilate_charge = _normalize_group_charge(
        getattr(pair_annihilate, "charge", zero), symmetry
    )
    onsite_charge = _normalize_group_charge(
        getattr(onsite, "charge", zero), symmetry
    )
    if create_charge == zero or annihilate_charge == zero:
        raise ValueError("pair creation and annihilation operators must be charged.")
    if create_charge != _charge_neg(annihilate_charge, symmetry):
        raise ValueError(
            "pair creation and annihilation charges must be negatives of one "
            "another."
        )
    if onsite_charge != zero:
        raise ValueError("the onsite pair subtraction must be charge neutral.")

    # ``_assemble_symmray_mpo`` reserves these boundary labels for the
    # identity path shared by all of its finite-state MPO builders.
    start = ("start",)
    create_state = ("factorized_pair_create",)
    annihilate_state = ("factorized_pair_annihilate",)
    done = ("done",)
    channels = [
        [
            (start, zero),
            (create_state, _charge_neg(create_charge, symmetry)),
            (annihilate_state, _charge_neg(annihilate_charge, symmetry)),
            (done, zero),
        ]
        for _ in range(max(L - 1, 0))
    ]
    transitions = [[] for _ in range(L)]
    identity = np.eye(len(phys_map), dtype=dtype)
    scaled_normalization = dtype.type(normalization)

    for site in range(L):
        if site < L - 1:
            # Put the overall normalization on the closing endpoint.  The
            # open-channel transition is the first factor in F^dagger F, so
            # applying normalization at both ends would incorrectly produce
            # normalization**2 for every pair.
            coefficient = dtype.type(signs[site])
            transitions[site].extend(
                (
                    (start, create_state, coefficient * create_dense),
                    (start, annihilate_state, coefficient * annihilate_dense),
                )
            )

        if 0 < site < L - 1:
            transitions[site].extend(
                (
                    (create_state, create_state, identity),
                    (annihilate_state, annihilate_state, identity),
                )
            )

        if site > 0:
            coefficient = scaled_normalization * dtype.type(signs[site])
            transitions[site].extend(
                (
                    (create_state, done, coefficient * annihilate_dense),
                    (annihilate_state, done, coefficient * create_dense),
                )
            )

        # The start->done onsite transition is intentionally omitted.  The
        # finite-state paths above contain only i < j and i > j, which is
        # exactly the off-diagonal structure factor.  Equivalently, this is
        # F^dagger F - sum_i Delta_i^dagger Delta_i without constructing the
        # diagonal term and subtracting it afterward.

    mpo = _assemble_symmray_mpo(
        L=L,
        channels=channels,
        transitions=transitions,
        phys_map=phys_map,
        symmetry=symmetry,
        zero=zero,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        to_backend=to_backend,
        fermionic=True,
        operator_charge=zero,
    )
    mpo.pepsy_compression_report.update(
        {
            "factorized_pair": True,
            "factorized_channels": 4,
            "pair_term_count": L * (L - 1) // 2,
            "normalization": complex(normalization),
        }
    )
    return mpo


def _build_fermionic_model_mpo(
    hamiltonian,
    L,
    *,
    mapper=None,
    idx2coo=None,
    coo2idx=None,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    dtype=None,
):
    _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
        mapper=mapper,
        idx2coo=idx2coo,
        coo2idx=coo2idx,
    )
    raw_edges = _as_edges(hamiltonian.edges)
    edges = _map_edges_to_mpo_indices(raw_edges, coo2idx_use)

    if L is None:
        L = (
            mapped_L
            if mapped_L is not None
            else max(max(int(i), int(j)) for i, j in edges) + 1
        )
    L = int(L)
    if L < 1:
        raise ValueError("L must be a positive integer.")
    if mapped_L is not None and L != mapped_L:
        raise ValueError(f"L={L} does not match MPO mapping length {mapped_L}.")

    dtype = (
        _dtype_from_hamiltonian_terms(hamiltonian.terms)
        if dtype is None
        else np.dtype(dtype)
    )
    zero = _normalize_group_charge(
        getattr(next(iter(hamiltonian.terms.values())), "charge", 0),
        hamiltonian.symmetry,
    )
    start = ("start",)
    done = ("done",)
    channels = [[(start, zero), (done, zero)] for _ in range(max(L - 1, 0))]
    transitions = [[] for _ in range(L)]

    coordinations = {}
    for left, right in raw_edges:
        coordinations[left] = coordinations.setdefault(left, 0) + 1
        coordinations[right] = coordinations.setdefault(right, 0) + 1

    if hamiltonian.model != "fermi_hubbard_spinless":  # pragma: no cover - guarded by caller
        raise NotImplementedError(f"Unsupported fermionic model {hamiltonian.model!r}.")
    delta = hamiltonian.parameters.get("delta", 0.0)
    if delta != 0:
        raise NotImplementedError(
            "SymHamiltonian.to_mpo does not yet support spinless "
            "Fermi-Hubbard pairing terms with delta != 0."
        )
    ops = _fh_spinless_dense_local_ops(hamiltonian.symmetry, dtype)
    phys_map = list(ops["index_map"])
    mode_terms = (("spinless", "t", ops["create"], ops["annihilate"], 1),)

    def add_channel(edge_pos, i, j, label, left_charge, left_op, right_op):
        channel_id = ("fermion", edge_pos, label, left_charge)
        channel_charge = _charge_neg(left_charge, hamiltonian.symmetry)
        for cut in range(i, j):
            channels[cut].append((channel_id, channel_charge))
        transitions[i].append((start, channel_id, left_op))
        string_op = (
            ops["parity"]
            if _charged_op_needs_fermion_string(left_charge)
            else ops["identity"]
        )
        for site in range(i + 1, j):
            transitions[site].append((channel_id, channel_id, string_op))
        transitions[j].append((channel_id, done, right_op))

    for edge_pos, (raw_edge, edge) in enumerate(zip(raw_edges, edges)):
        raw_left, raw_right = raw_edge
        left_site, right_site = int(edge[0]), int(edge[1])
        if left_site == right_site:
            raise ValueError("Hamiltonian edges must connect distinct sites.")
        if not (0 <= left_site < L and 0 <= right_site < L):
            raise ValueError(f"edge {edge!r} is outside MPO length L={L}.")

        t_values = {"t": _edge_parameter(hamiltonian.parameters.get("t", 1.0), raw_left, raw_right)}
        V_edge = _edge_parameter(hamiltonian.parameters.get("V", 0.0), raw_left, raw_right)
        if V_edge != 0:
            i, j = sorted((left_site, right_site))
            add_channel(edge_pos, i, j, "V", zero, V_edge * ops["number"], ops["number"])

        for raw_site, site in ((raw_left, left_site), (raw_right, right_site)):
            coordination = coordinations[raw_site]
            mu_site = _node_parameter(hamiltonian.parameters.get("mu", 0.0), raw_site)
            onsite = -(mu_site / coordination) * ops["number"]
            if np.any(onsite != 0):
                transitions[site].append((start, done, onsite))

        i, j = sorted((left_site, right_site))
        for spin, t_key, create, annihilate, create_charge in mode_terms:
            t_sigma = t_values[t_key]
            if t_sigma == 0:
                continue
            for direction, first, second, first_charge in (
                ("forward", create, annihilate, create_charge),
                ("backward", annihilate, create, _charge_neg(create_charge, hamiltonian.symmetry)),
            ):
                endpoint = (
                    first @ ops["parity"]
                    if direction == "forward"
                    else ops["parity"] @ first
                )
                add_channel(
                    edge_pos,
                    i,
                    j,
                    (spin, direction),
                    first_charge,
                    -t_sigma * endpoint,
                    second,
                )

    return _assemble_symmray_mpo(
        L=L,
        channels=channels,
        transitions=transitions,
        phys_map=phys_map,
        symmetry=hamiltonian.symmetry,
        zero=zero,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        to_backend=to_backend,
    )


def _add_native_term_to_mpo(
    term,
    sites,
    *,
    term_pos,
    channels,
    transitions,
    symmetry,
    dtype,
    zero,
    operator_charge=None,
):
    """Add one homogeneous native fermion term as graded MPO transitions.

    The term is split by operator Schmidt decompositions over the ordered
    support sites. This preserves Symmray's fermionic bond phases while
    allowing arbitrary term rank and non-contiguous support.
    """
    sites = tuple(int(site) for site in sites)
    if len(set(sites)) != len(sites):
        raise ValueError("Hamiltonian term supports must contain unique sites.")
    if any(site < 0 or site >= len(transitions) for site in sites):
        raise ValueError(f"Hamiltonian term support {sites!r} is outside MPO bounds.")
    start = ("start",)
    done = ("done",)

    order = tuple(sorted(range(len(sites)), key=sites.__getitem__))
    sites = tuple(sorted(sites))
    indices = getattr(term, "indices", None)
    if indices is None or len(indices) == 0 or len(indices) % 2:
        raise TypeError(
            "SymHamiltonian.to_mpo requires an even-rank Symmray operator."
        )
    n_sites = len(sites)
    if len(indices) != 2 * n_sites:
        raise ValueError("Term metadata and support size disagree.")

    output_maps = [_expanded_index_charges(index) for index in indices[:n_sites]]
    input_maps = [_expanded_index_charges(index) for index in indices[n_sites:]]
    if output_maps != input_maps:
        raise ValueError(
            "Hamiltonian terms must use matching upper/lower physical charge "
            "maps at every site."
        )
    if any(site_map != output_maps[0] for site_map in output_maps[1:]):
        raise ValueError(
            "SymHamiltonian.to_mpo currently requires one physical charge "
            "map shared by all sites."
        )
    physical_maps = [output_maps[pos] for pos in order]

    term_charge = _normalize_group_charge(
        getattr(term, "charge", zero),
        symmetry,
    )
    expected_charge = (
        zero
        if operator_charge is None
        else _normalize_group_charge(operator_charge, symmetry)
    )
    if term_charge != expected_charge:
        raise ValueError(
            "Native MPO terms must share one homogeneous operator charge "
            f"{expected_charge!r}; term {term_pos} has charge {term_charge!r}."
        )

    if n_sites == 1:
        dense = _dense_numpy(term, dtype=dtype)
        for out_i, in_i in product(range(len(physical_maps[0])), repeat=2):
            coefficient = dense[out_i, in_i]
            if not np.any(coefficient):
                continue
            local = np.zeros((len(physical_maps[0]), len(physical_maps[0])), dtype=dtype)
            local[out_i, in_i] = coefficient
            transitions[sites[0]].append((start, done, local))
        return list(physical_maps[0])

    axes = order + tuple(n_sites + pos for pos in order)
    ordered_term = term if order == tuple(range(n_sites)) else term.transpose(axes)
    local_term = ordered_term.fuse(
        *((pos, n_sites + pos) for pos in range(n_sites))
    )

    factors = []
    current = local_term
    for pos in range(n_sites - 1):
        ndim = current.ndim
        if pos == 0:
            left_group = (0,)
            right_group = tuple(range(1, ndim))
        else:
            left_group = (0, 1)
            right_group = tuple(range(2, ndim))
        # Absorb the singular values into the left factor for a charged
        # operator. This leaves the total operator charge on the final site
        # factor, where the open MPO boundary can carry it, while all
        # preceding tensors remain neutral and can propagate identity paths.
        absorb = "left" if expected_charge != zero else "right"
        left, _, right = current.fuse(left_group, right_group).svd(
            absorb=absorb
        )
        if pos == 0:
            factors.append(left.unfuse(0).transpose((2, 0, 1)))
        else:
            factors.append(
                left.unfuse(0).unfuse(1).transpose((0, 3, 1, 2))
            )
        current = right.unfuse(1)
    factors.append(current)

    # Each operator-Schmidt bond becomes a family of MPO channels. The
    # channel charge is taken directly from the native factor bond index,
    # rather than reconstructed from dense matrix elements.
    interval_channel_ids = []
    interval_channel_charges = []
    for interval in range(n_sites - 1):
        factor = factors[interval]
        bond_axis = 0 if interval == 0 else 1
        bond_map = _expanded_index_charges(factor.indices[bond_axis])
        channel_ids = []
        for bond_pos, bond_charge in enumerate(bond_map):
            if expected_charge != zero:
                # With absorb="left", the factor bond is dual on the side
                # that becomes the MPO's outgoing bond. Reverse its charge
                # when installing the common MPO bond orientation.
                bond_charge = _charge_neg(bond_charge, symmetry)
            channel_id = ("native", term_pos, interval, bond_pos)
            channel_ids.append(channel_id)
            for cut in range(sites[interval], sites[interval + 1]):
                channels[cut].append((channel_id, bond_charge))
        interval_channel_ids.append(channel_ids)
        interval_channel_charges.append(tuple(bond_map))

    first_data = _dense_numpy(factors[0], dtype=dtype)
    for bond_pos, channel_id in enumerate(interval_channel_ids[0]):
        op = first_data[bond_pos]
        if np.any(op):
            transitions[sites[0]].append((start, channel_id, op))

    for interval in range(n_sites - 2):
        factor_data = _dense_numpy(factors[interval + 1], dtype=dtype)
        left_ids = interval_channel_ids[interval]
        right_ids = interval_channel_ids[interval + 1]
        for left_pos, left_id in enumerate(left_ids):
            for right_pos, right_id in enumerate(right_ids):
                op = factor_data[left_pos, right_pos]
                if np.any(op):
                    transitions[sites[interval + 1]].append(
                        (left_id, right_id, op)
                    )

    last_data = _dense_numpy(factors[-1], dtype=dtype)
    for bond_pos, channel_id in enumerate(interval_channel_ids[-1]):
        op = last_data[bond_pos]
        if np.any(op):
            transitions[sites[-1]].append((channel_id, done, op))

    identity = np.eye(len(physical_maps[0]), dtype=dtype)
    for interval, channel_ids in enumerate(interval_channel_ids):
        for site in range(sites[interval] + 1, sites[interval + 1]):
            for channel_pos, channel_id in enumerate(channel_ids):
                # A skipped site still participates in the graded ordering.
                # For an odd operator-Schmidt channel, moving that channel
                # past one omitted physical site contributes a scalar -1.
                # This is a graded factorization phase, not a JW parity
                # operator: inserting the latter would change the native
                # operator by making the result state-dependent.
                bond_charge = interval_channel_charges[interval][channel_pos]
                phase_op = (
                    -identity
                    if _charged_op_needs_fermion_string(bond_charge)
                    else identity
                )
                transitions[site].append((channel_id, channel_id, phase_op))

    return list(physical_maps[0])


def _native_local_term_mpo(
    term,
    support,
    L,
    *,
    symmetry,
    dtype,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
):
    """Build an exact local-term MPO without start/done channel inflation.

    The generic native MPO assembler is designed for a collection of terms,
    so it carries explicit start and done paths at every chain cut. For one
    one-site or two-site term those paths are unnecessary. Factorizing the
    native local array directly and propagating its operator-Schmidt bond
    through identity tensors leaves only the non-zero local Schmidt sectors.
    """
    support = tuple(int(site) for site in support)
    if len(support) not in {1, 2} or len(set(support)) != len(support):
        raise ValueError("direct local PEPO terms must act on one or two sites.")
    if any(site < 0 or site >= int(L) for site in support):
        raise ValueError(f"term support {support!r} is outside MPO length L={L}.")

    _require_symmray()
    from symmray import utils as sr_utils  # pylint: disable=import-outside-toplevel

    zero = _zero_like_charge(0 if symmetry in {"U1", "Z2"} else (0, 0))
    term_charge = _normalize_group_charge(
        getattr(term, "charge", zero), symmetry
    )
    indices = getattr(term, "indices", None)
    if indices is None or len(indices) != 2 * len(support):
        raise TypeError("direct local PEPO terms require matching native rank.")

    physical_maps = [
        _expanded_index_charges(index) for index in indices[:len(support)]
    ]
    input_maps = [
        _expanded_index_charges(index) for index in indices[len(support):]
    ]
    if physical_maps != input_maps or any(
        physical_maps[site] != physical_maps[0]
        for site in range(len(support))
    ):
        raise ValueError(
            "direct local PEPO terms require one matching physical charge map."
        )
    phys_map = physical_maps[0]
    phys_dim = len(phys_map)
    zero_map = [zero]

    def make_array(data, index_maps, duals, *, charge=zero, label=None):
        return sr_utils.from_dense(
            data,
            symmetry=symmetry,
            index_maps=index_maps,
            duals=duals,
            fermionic=True,
            charge=charge,
            label=label,
        )

    def identity_tensor(site):
        identity = np.eye(phys_dim, dtype=dtype)
        if L == 1:
            return make_array(identity, [phys_map, phys_map], [False, True])
        if site == 0:
            return make_array(
                identity.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [False, False, True],
            )
        if site == L - 1:
            return make_array(
                identity.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [True, False, True],
            )
        data = np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
        data[0, 0] = identity
        return make_array(
            data,
            [zero_map, zero_map, phys_map, phys_map],
            [True, False, False, True],
        )

    arrays = [identity_tensor(site) for site in range(int(L))]
    local_schmidt_bond = 1

    if len(support) == 1:
        site = support[0]
        dense = _dense_numpy(term, dtype=dtype)
        label = site if _charged_op_needs_fermion_string(term_charge) else None
        if L == 1:
            arrays[site] = make_array(
                dense, [phys_map, phys_map], [False, True],
                charge=term_charge, label=label,
            )
        elif site == 0:
            arrays[site] = make_array(
                dense.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [False, False, True],
                charge=term_charge, label=label,
            )
        elif site == L - 1:
            arrays[site] = make_array(
                dense.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [True, False, True],
                charge=term_charge, label=label,
            )
        else:
            arrays[site] = make_array(
                dense.reshape(1, 1, phys_dim, phys_dim),
                [zero_map, zero_map, phys_map, phys_map],
                [True, False, False, True],
                charge=term_charge, label=label,
            )
    else:
        # Order the support by the MPO chain. A native operator's upper and
        # lower legs are reordered together so its graded local signs survive.
        ordered = tuple(sorted(enumerate(support), key=lambda item: item[1]))
        if tuple(item[0] for item in ordered) == (0, 1):
            ordered_term = term
        else:
            ordered_term = term.transpose((1, 0, 3, 2))
        fused = ordered_term.fuse((0, 2), (1, 3))
        # The only cutoff here removes exact numerical zero singular blocks
        # left by Symmray's block SVD. It is not the user-requested PEPO
        # compression cutoff and does not cap the resulting local bond.
        structural_cutoff = 64.0 * np.finfo(float).eps
        left, _, right = fused.svd(
            absorb="right",
            cutoff=structural_cutoff,
        )
        left = left.unfuse(0).transpose((2, 0, 1))
        right = right.unfuse(1)
        bond_map = _expanded_index_charges(left.indices[0])
        bond_dim = len(bond_map)
        local_schmidt_bond = bond_dim
        left_dense = _dense_numpy(left, dtype=dtype)
        right_dense = _dense_numpy(right, dtype=dtype)
        left_charge = _normalize_group_charge(
            getattr(left, "charge", zero), symmetry
        )
        right_charge = _normalize_group_charge(
            getattr(right, "charge", zero), symmetry
        )
        left_site, right_site = (item[1] for item in ordered)

        if left_site == 0:
            arrays[left_site] = left
        else:
            arrays[left_site] = make_array(
                left_dense.reshape(1, bond_dim, phys_dim, phys_dim),
                [zero_map, bond_map, phys_map, phys_map],
                [True, False, False, True],
                charge=left_charge,
            )
        if right_site == L - 1:
            arrays[right_site] = right
        else:
            arrays[right_site] = make_array(
                right_dense.reshape(bond_dim, 1, phys_dim, phys_dim),
                [bond_map, zero_map, phys_map, phys_map],
                [True, False, False, True],
                charge=right_charge,
            )
        identity = np.eye(phys_dim, dtype=dtype)
        for site in range(left_site + 1, right_site):
            data = np.zeros(
                (bond_dim, bond_dim, phys_dim, phys_dim),
                dtype=dtype,
            )
            for bond_pos in range(bond_dim):
                data[bond_pos, bond_pos] = identity
            arrays[site] = make_array(
                data,
                [bond_map, bond_map, phys_map, phys_map],
                [True, False, False, True],
            )

    mpo = qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
    )
    if to_backend is not None:
        _apply_to_tensor_network_arrays(mpo, to_backend)
    raw_bond = mpo.max_bond()
    raw_max_bond = 1 if raw_bond is None else int(raw_bond)
    did_compress = bool(compress and L > 1)
    if did_compress:
        compress_opts = {"cutoff": cutoff}
        if max_bond is not None:
            compress_opts["max_bond"] = int(max_bond)
        mpo.compress(**compress_opts)
    final_bond = mpo.max_bond()
    final_max_bond = 1 if final_bond is None else int(final_bond)
    mpo.pepsy_compression_report = {
        "direct_local": True,
        "compressed": did_compress,
        "cutoff": cutoff,
        "requested_max_bond": None if max_bond is None else int(max_bond),
        "operator_schmidt_bond": local_schmidt_bond,
        "raw_max_bond": raw_max_bond,
        "final_max_bond": final_max_bond,
        "rank_reduced": final_max_bond < raw_max_bond,
        "max_bond_exceeded": (
            did_compress
            and max_bond is not None
            and final_max_bond > int(max_bond)
        ),
    }
    return mpo


def _generic_symhamiltonian_to_mpo(
    hamiltonian,
    L,
    *,
    mapper=None,
    idx2coo=None,
    coo2idx=None,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    dtype=None,
    fermionic=False,
):
    """Build a Symmray MPO from explicit local terms."""
    _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
        mapper=mapper,
        idx2coo=idx2coo,
        coo2idx=coo2idx,
    )
    raw_wheres = tuple(hamiltonian.terms)
    if not raw_wheres:
        raise ValueError("At least one Hamiltonian term is required to build an MPO.")
    coordinate_sites = _term_mapping_uses_coordinate_sites(raw_wheres)
    wheres = tuple(
        _as_term_where(where, coordinate_sites=coordinate_sites)
        for where in raw_wheres
    )
    mapped_wheres = tuple(
        tuple(_map_site_to_mpo_index(site, coo2idx_use) for site in where)
        for where in wheres
    )

    if L is None:
        L = (
            mapped_L
            if mapped_L is not None
            else max(max(int(site) for site in where) for where in mapped_wheres) + 1
        )
    L = int(L)
    if L < 1:
        raise ValueError("L must be a positive integer.")
    if mapped_L is not None and L != mapped_L:
        raise ValueError(f"L={L} does not match MPO mapping length {mapped_L}.")

    dtype = (
        _dtype_from_hamiltonian_terms(hamiltonian.terms)
        if dtype is None
        else np.dtype(dtype)
    )
    first_term = next(iter(hamiltonian.terms.values()))
    first_charge = _normalize_group_charge(
        getattr(first_term, "charge", 0),
        hamiltonian.symmetry,
    )
    zero = _zero_like_charge(first_charge)
    operator_charge = first_charge if fermionic else zero
    if not fermionic and first_charge != zero:
        raise ValueError(
            "Charged native operator terms require fermionic=True; the "
            "Jordan-Wigner compatibility MPO is neutral-only."
        )
    start = ("start",)
    done = ("done",)
    channels = [
        [(start, zero), (done, _charge_neg(operator_charge, hamiltonian.symmetry))]
        for _ in range(max(L - 1, 0))
    ]
    transitions = [[] for _ in range(L)]
    phys_map = None

    for term_pos, (raw_where, where) in enumerate(zip(raw_wheres, mapped_wheres)):
        term = hamiltonian.terms[raw_where]
        term_is_fermionic = _is_fermionic_symmray_array(term)
        term_charge = _normalize_group_charge(
            getattr(term, "charge", zero),
            hamiltonian.symmetry,
        )
        if not fermionic and term_charge != zero:
            raise ValueError(
                "Charged native operator terms require fermionic=True; the "
                "Jordan-Wigner compatibility MPO is neutral-only."
            )
        if fermionic and not term_is_fermionic:
            raise TypeError(
                "Native fermionic MPO construction requires every Hamiltonian "
                "term to be a Symmray FermionicArray."
            )

        if fermionic:
            term_phys = _add_native_term_to_mpo(
                term,
                where,
                term_pos=term_pos,
                channels=channels,
                transitions=transitions,
                symmetry=hamiltonian.symmetry,
                dtype=dtype,
                zero=zero,
                operator_charge=operator_charge,
            )
            if phys_map is None:
                phys_map = term_phys
            elif phys_map != term_phys:
                raise ValueError(
                    "SymHamiltonian.to_mpo requires one physical charge map "
                    "shared by all sites."
                )
            continue

        if len(where) == 1:
            site = int(where[0])
            if not 0 <= site < L:
                raise ValueError(f"site {site!r} is outside MPO length L={L}.")
            dense, term_phys = _term_dense_and_phys_map(term, dtype=dtype)
            if phys_map is None:
                phys_map = list(term_phys)
            elif phys_map != list(term_phys):
                raise ValueError(
                    "SymHamiltonian.to_mpo requires one physical charge map "
                    "shared by all sites."
                )
            transitions[site].append((start, done, dense))
            continue

        if len(where) > 2:
            raise NotImplementedError(
                "Jordan-Wigner compatibility MPO conversion currently supports "
                "one- and two-site terms; use fermionic=True for native "
                "multi-site terms."
            )

        i, j = (int(where[0]), int(where[1]))
        if i == j:
            raise ValueError("Hamiltonian edges must connect distinct sites.")
        if not (0 <= i < L and 0 <= j < L):
            raise ValueError(f"edge {where!r} is outside MPO length L={L}.")
        reverse = i > j
        if reverse:
            i, j = j, i

        term_channels, left_phys, right_phys = _decompose_neutral_two_site_term(
            term,
            symmetry=hamiltonian.symmetry,
            dtype=dtype,
            reverse=reverse,
            # ``_decompose_neutral_two_site_term(..., fermionic=True)`` is the
            # legacy conversion from native local data to a bosonic/JW
            # site-major matrix. A native graded MPO keeps the raw fermionic
            # tensor ordering and lets Symmray supply the Koszul signs.
            fermionic=term_is_fermionic and not fermionic,
        )
        if left_phys != right_phys:
            raise ValueError(
                "SymHamiltonian.to_mpo currently requires a uniform physical "
                "charge map on both sites of each term."
            )
        if phys_map is None:
            phys_map = list(left_phys)
        elif phys_map != list(left_phys):
            raise ValueError(
                "SymHamiltonian.to_mpo currently requires one physical charge "
                "map shared by all sites."
            )

        for rank, (left_charge, left_op, right_op) in enumerate(term_channels):
            channel_id = ("term", term_pos, rank, left_charge)
            channel_charge = _charge_neg(left_charge, hamiltonian.symmetry)
            for cut in range(i, j):
                channels[cut].append((channel_id, channel_charge))
            transitions[i].append((start, channel_id, left_op))

            if (
                term_is_fermionic
                and not fermionic
                and _charged_op_needs_fermion_string(left_charge)
            ):
                string_op = _fermion_parity_operator(phys_map, dtype)
            else:
                string_op = np.eye(len(phys_map), dtype=dtype)
            for site in range(i + 1, j):
                transitions[site].append((channel_id, channel_id, string_op))

            transitions[j].append((channel_id, done, right_op))

    if phys_map is None:
        raise ValueError("At least one Hamiltonian term is required to build an MPO.")

    return _assemble_symmray_mpo(
        L=L,
        channels=channels,
        transitions=transitions,
        phys_map=phys_map,
        symmetry=hamiltonian.symmetry,
        zero=zero,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        to_backend=to_backend,
        fermionic=fermionic,
        operator_charge=operator_charge,
    )


def _group_symhamiltonian_terms_by_charge(hamiltonian):
    """Group native Hamiltonian terms into homogeneous charge sectors."""
    sectors = {}
    for where, term in hamiltonian.terms.items():
        charge = _normalize_group_charge(
            getattr(term, "charge", 0),
            hamiltonian.symmetry,
        )
        sectors.setdefault(charge, {})[where] = term
    return sectors
