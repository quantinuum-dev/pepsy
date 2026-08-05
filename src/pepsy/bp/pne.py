"""Partitioned network expansions (PNE) for D1BP and D2BP contractions.

This is the construction of Evenbly, Gray, and Chan,
*Partitioned Expansions for Approximate Tensor Network Contractions*,
arXiv:2512.10910.  It is a third expansion family, separate from both
``loop_series_expand`` and ``loop_cluster_expand``.

The user chooses a set of tensor-network indices and a projector ``P`` on each
one.  The complementary projector is ``Q = I - P``.  The linear form uses
``Q ... Q P`` terms, while the combinatorial form uses alternating sums of
``P`` insertions.  The all-``Q`` residue can be retained for an exact identity
or dropped for the approximation proposed in the paper.

For the BP-backed default, rank-one projectors are formed from D1BP messages
or D2BP density messages.  Explicit matrix projectors are also accepted,
which allows the PNE to be used without a converged BP fixed point and with
higher-rank dominant subspaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, ClassVar

import autoray as ar
import numpy as np

from .series import _build_bp
from ._symmray import (
    align_d2bp_messages as _align_symmray_d2bp_messages,
    d2_operator as _symmray_d2_operator,
    rank_one_d2_projector as _symmray_rank_one_d2_projector,
    to_dense as _symmray_to_dense,
    uses_symmray as _uses_symmray,
)

__all__ = [
    "PNEExpansionTerm",
    "PNEExpansionResult",
    "PNEPartitionScore",
    "PNEPartitionSelection",
    "RecursivePNEExpansionResult",
    "pne_projector_diagnostics",
    "pne_projectors",
    "select_pne_partitions",
    "recursive_partitioned_expand",
    "pne_expand",
    "partitioned_expand",
]


@dataclass(frozen=True)
class PNEExpansionTerm:
    """One non-residue or residue term in a partitioned expansion."""

    partition_ids: tuple[int, ...]
    p_inds: tuple[Any, ...]
    q_inds: tuple[Any, ...]
    coefficient: int = 1
    residue: bool = False


@dataclass(frozen=True)
class PNEPartitionScore:
    """Estimated neglected-residue score for one candidate partition index."""

    index: Any
    residue_norm: float
    relative_residue: float
    result: Any = field(repr=False)


@dataclass(frozen=True)
class PNEPartitionSelection:
    """Partition indices ranked by estimated single-index residue."""

    indices: tuple[Any, ...]
    scores: tuple[PNEPartitionScore, ...]


@dataclass
class RecursivePNEExpansionResult:
    """Result of a fixed recursive PNE partition schedule."""

    expansion: ClassVar[str] = "pne-recursive"
    cutoff_kind: ClassVar[str] = "recursive-partition-schedule"

    estimate: Any
    norm: str
    form: str
    levels: tuple[tuple[Any, ...], ...]
    terms: tuple[PNEExpansionTerm, ...]
    term_values: dict[PNEExpansionTerm, Any]
    residue_included: bool
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None
    bp: Any

    @property
    def messages(self):
        return self.bp.messages


@dataclass
class PNEExpansionResult:
    """Result of a D1BP/D2BP partitioned network expansion."""

    expansion: ClassVar[str] = "pne"
    cutoff_kind: ClassVar[str] = "selected-partition-indices"

    estimate: Any
    norm: str
    form: str
    partitions: tuple[tuple[Any, ...], ...]
    terms: tuple[PNEExpansionTerm, ...]
    term_values: dict[PNEExpansionTerm, Any]
    residue_estimate: Any | None
    residue_included: bool
    open_inds: tuple[Any, ...]
    bp_converged: bool | None
    bp_iterations: int | None
    bp_max_mdiff: float | None
    bp: Any
    _projectors: dict[Any, Any] | None = field(default=None, repr=False)
    _contract_defaults: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def messages(self):
        """Return the BP messages used for default projectors."""
        return self.bp.messages

    @property
    def partition_inds(self) -> tuple[Any, ...]:
        """Return the flattened selected index set."""
        return tuple(index for group in self.partitions for index in group)

    @property
    def residue_norm(self) -> float | None:
        """Return the absolute residue norm when it was evaluated."""
        if self.residue_estimate is None:
            return None
        return _value_norm(self.residue_estimate)

    def expand(
        self,
        partition_inds=None,
        *,
        partitions=None,
        form: str | None = None,
        include_residue: bool | None = None,
        max_terms: int = 4096,
        strip_exponent: bool = False,
        optimize: str = "auto-hq",
        **contract_opts,
    ):
        """Evaluate another partition choice using the same normalized BP state."""
        if form is None:
            form = self.form
        if include_residue is None:
            include_residue = self.residue_included
        if partition_inds is None and partitions is None:
            partitions = self.partitions
        return _partitioned_contract(
            self.bp,
            _normalize_partitions(partition_inds, partitions),
            form=form,
            projectors=self._projectors,
            include_residue=include_residue,
            max_terms=max_terms,
            strip_exponent=strip_exponent,
            optimize=optimize,
            contract_opts=contract_opts,
            open_inds=self.open_inds,
            normalize=False,
        )[0]


def _normalize_partitions(partition_inds, partitions):
    if partition_inds is not None and partitions is not None:
        raise ValueError("pass either partition_inds or partitions, not both")
    if partitions is None:
        if partition_inds is None:
            raise ValueError("partition_inds or partitions must be supplied")
        selected = tuple(partition_inds)
        if not selected:
            raise ValueError("at least one partition index must be supplied")
        return tuple((index,) for index in selected)

    groups = []
    for group in partitions:
        group = tuple(group)
        if not group:
            raise ValueError("partition groups cannot be empty")
        groups.append(group)
    if not groups:
        raise ValueError("at least one partition group must be supplied")
    return tuple(groups)


def _validate_partitions(tn, partitions, *, norm):
    internal = {
        index
        for index, tids in tn.ind_map.items()
        if len(tids) == 2
    }
    flattened = [index for group in partitions for index in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("partition indices must be unique")
    unknown = set(flattened).difference(internal)
    if unknown:
        kind = "virtual pairwise" if norm == "2norm" else "pairwise"
        raise ValueError(
            f"PNE currently requires selected {kind} indices; invalid "
            f"partition indices: {unknown!r}"
        )
    return partitions


def _operator_shape(operator):
    return tuple(ar.do("shape", operator))


def _value_norm(value) -> float:
    array = np.asarray(ar.to_numpy(value))
    return float(np.linalg.norm(array.reshape(-1)))


def pne_projectors(source, indices=None):
    """Return BP-derived dominant projectors for selected indices.

    ``source`` can be a D1BP/D2BP object or any result object exposing a
    ``.bp`` attribute. The BP message pairs are normalized before forming the
    projectors. The returned D1 matrices have shape ``(d, d)``; D2 matrices
    act on the vectorized double-layer bond and have shape ``(d*d, d*d)``.
    """
    bp = getattr(source, "bp", source)
    if indices is None:
        indices = tuple(bp.tn.inner_inds())
    indices = tuple(indices)
    if bp.__class__.__name__ == "D2BP":
        _align_symmray_d2bp_messages(bp)
    bp.normalize_message_pairs()
    return _projector_map(
        bp,
        tuple((index,) for index in indices),
        projectors=None,
    )


def pne_projector_diagnostics(projectors):
    """Return rank, idempotency, and Hermiticity diagnostics for ``P`` maps."""
    diagnostics = {}
    for index, operator in projectors.items():
        matrix = _symmray_to_dense(operator)
        if matrix.ndim > 2:
            half = int(np.sqrt(matrix.size))
            matrix = matrix.reshape(half, half)
        identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
        scale = max(1.0, float(np.linalg.norm(matrix)))
        diagnostics[index] = {
            "shape": _operator_shape(operator),
            "rank": int(np.linalg.matrix_rank(matrix)),
            "idempotency_error": float(
                np.linalg.norm(matrix @ matrix - matrix) / scale
            ),
            "hermiticity_error": float(
                np.linalg.norm(matrix - matrix.conj().T) / scale
            ),
            "trace": np.trace(matrix),
            "complement_trace": np.trace(identity - matrix),
        }
    return diagnostics


def _rank_one_d1_projector(bp, index):
    left, right = tuple(bp.tn.ind_map[index])
    return ar.do(
        "einsum",
        "i,j->ij",
        bp.messages[index, left],
        bp.messages[index, right],
    )


def _rank_one_d2_projector(bp, index):
    left, right = tuple(bp.tn.ind_map[index])
    if _uses_symmray(bp.tn):
        return _symmray_rank_one_d2_projector(
            bp.tn,
            index,
            bp.messages[index, left],
            bp.messages[index, right],
        )
    ml = bp.messages[index, left].reshape(-1)
    mr = bp.messages[index, right].reshape(-1)
    return ar.do("einsum", "i,j->ij", ml, mr)


def _projector_map(bp, partitions, projectors):
    explicit = {} if projectors is None else dict(projectors)
    selected = {index for group in partitions for index in group}
    unknown = set(explicit).difference(bp.tn.ind_map)
    if unknown:
        raise ValueError(
            "projectors supplied for unknown network indices: "
            f"{unknown!r}"
        )

    result = {}
    for index in selected:
        result[index] = explicit.get(
            index,
            _rank_one_d1_projector(bp, index)
            if bp.__class__.__name__ == "D1BP"
            else _rank_one_d2_projector(bp, index),
        )
    return result


def _operator_for(bp, index, projectors, *, complement):
    operator = projectors[index]
    if bp.__class__.__name__ == "D1BP":
        expected = bp.tn.tensor_map[next(iter(bp.tn.ind_map[index]))].ind_size(index)
        shape = _operator_shape(operator)
        if shape != (expected, expected):
            raise ValueError(
                f"projector for {index!r} must have shape "
                f"{(expected, expected)}, got {shape}"
            )
        if complement:
            operator = ar.do("eye", expected) - operator
        return operator

    if _uses_symmray(bp.tn):
        return _symmray_d2_operator(
            bp.tn,
            index,
            operator,
            complement=complement,
        )

    left = next(iter(bp.tn.ind_map[index]))
    dimension = bp.tn.tensor_map[left].ind_size(index)
    flat_dimension = dimension * dimension
    shape = _operator_shape(operator)
    if shape == (dimension, dimension, dimension, dimension):
        operator = ar.do("reshape", operator, (flat_dimension, flat_dimension))
    elif shape != (flat_dimension, flat_dimension):
        raise ValueError(
            f"D2BP projector for {index!r} must have shape "
            f"{(flat_dimension, flat_dimension)} or "
            f"{(dimension, dimension, dimension, dimension)}, got {shape}"
        )
    if complement:
        operator = ar.do("eye", flat_dimension) - operator
    return ar.do(
        "reshape",
        operator,
        (dimension, dimension, dimension, dimension),
    )


def _term_network_d1(bp, p_inds, q_inds, projectors):
    network = bp.tn.copy()
    for index in q_inds:
        _, right = tuple(network.ind_map[index])
        network.tensor_map[right].gate_(
            _operator_for(bp, index, projectors, complement=True), index
        )
    for index in p_inds:
        _, right = tuple(network.ind_map[index])
        network.tensor_map[right].gate_(
            _operator_for(bp, index, projectors, complement=False), index
        )
    return network


def _term_network_d2(bp, p_inds, q_inds, projectors, open_inds=()):
    import quimb.tensor as qtn

    stn = bp.tn
    selected = set(p_inds) | set(q_inds)
    dual_inds = {index: qtn.rand_uuid() for index in stn._inner_inds}
    ket_maps = {tid: {} for tid in stn.tensor_map}
    bra_maps = {tid: {} for tid in stn.tensor_map}
    selected_maps = {}

    open_inds = tuple(open_inds)
    open_set = set(open_inds)
    bra_open_labels = {
        index: f"__pne_bra_{repr(index)}" for index in open_inds
    }

    for index, tids in stn.ind_map.items():
        if index in bp.output_inds:
            if index in open_set:
                (tid,) = tuple(tids)
                bra_maps[tid][index] = bra_open_labels[index]
            continue
        if index in stn._inner_inds and index in selected:
            selected_maps[index] = {}
            for tid in tids:
                selected_maps[index][tid] = (
                    qtn.rand_uuid(),
                    qtn.rand_uuid(),
                )
                ket_maps[tid][index] = selected_maps[index][tid][0]
                bra_maps[tid][index] = selected_maps[index][tid][1]
        elif index in stn._inner_inds:
            for tid in tids:
                ket_maps[tid][index] = index
                bra_maps[tid][index] = dual_inds[index]

    local = qtn.TensorNetwork()
    for tid, tensor in stn.tensor_map.items():
        local |= tensor.reindex(ket_maps[tid])
        local |= tensor.conj().reindex(bra_maps[tid])

    for index, tids in selected_maps.items():
        tid_left, tid_right = tuple(stn.ind_map[index])
        left_bra, left_ket = selected_maps[index][tid_left]
        right_bra, right_ket = selected_maps[index][tid_right]
        projector = _operator_for(
            bp,
            index,
            projectors,
            complement=index in q_inds,
        )
        local |= qtn.Tensor(
            projector,
            inds=(left_ket, left_bra, right_ket, right_bra),
        )

    return local


def _term_network(bp, term, projectors, open_inds=()):
    if bp.__class__.__name__ == "D1BP":
        return _term_network_d1(bp, term.p_inds, term.q_inds, projectors)
    return _term_network_d2(
        bp, term.p_inds, term.q_inds, projectors, open_inds=open_inds
    )


def _make_terms(partitions, *, form, include_residue, max_terms):
    form = str(form).lower()
    if form not in {"linear", "combinatorial"}:
        raise ValueError("form must be either 'linear' or 'combinatorial'")
    if len(partitions) > 62:
        raise ValueError("too many partitions for explicit PNE enumeration")

    if form == "linear":
        count = len(partitions) + int(include_residue)
    else:
        count = (1 << len(partitions)) - 1 + int(include_residue)
    if count > max_terms:
        raise ValueError(
            f"PNE would contain {count} terms, exceeding max_terms={max_terms}"
        )

    def flatten(ids):
        return tuple(index for ident in ids for index in partitions[ident])

    terms = []
    if form == "linear":
        for current in range(len(partitions)):
            q_ids = tuple(range(current))
            p_ids = (current,)
            terms.append(
                PNEExpansionTerm(
                    partition_ids=p_ids,
                    p_inds=flatten(p_ids),
                    q_inds=flatten(q_ids),
                )
            )
    else:
        for size in range(1, len(partitions) + 1):
            for p_ids in combinations(range(len(partitions)), size):
                terms.append(
                    PNEExpansionTerm(
                        partition_ids=p_ids,
                        p_inds=flatten(p_ids),
                        q_inds=(),
                        coefficient=1 if size % 2 else -1,
                    )
                )

    if include_residue:
        all_ids = tuple(range(len(partitions)))
        terms.append(
            PNEExpansionTerm(
                partition_ids=all_ids,
                p_inds=(),
                q_inds=flatten(all_ids),
                residue=True,
            )
        )
    return tuple(terms)


def _partitioned_contract(
    bp,
    partitions,
    *,
    form,
    projectors,
    include_residue,
    max_terms,
    strip_exponent,
    optimize,
    contract_opts,
    open_inds,
    normalize,
):
    if any(len(group) > 1 for group in partitions) and form == "linear":
        raise ValueError(
            "multi-index PNE partitions currently require form='combinatorial'"
        )
    if normalize:
        if bp.__class__.__name__ == "D2BP":
            _align_symmray_d2bp_messages(bp)
        bp.normalize_message_pairs()
        if not (bp.__class__.__name__ == "D1BP" and open_inds):
            bp.normalize_tensors()

    projectors = _projector_map(bp, partitions, projectors)
    open_inds = tuple(open_inds)
    if bp.__class__.__name__ == "D2BP":
        invalid_open = set(open_inds).difference(bp.output_inds)
        if invalid_open:
            raise ValueError(
                "D2BP open_inds must be physical/output indices; invalid "
                f"indices: {invalid_open!r}"
            )
        output_inds = tuple(open_inds) + tuple(
            f"__pne_bra_{repr(index)}" for index in open_inds
        )
    else:
        unknown_open = set(open_inds).difference(bp.tn.ind_map)
        if unknown_open:
            raise ValueError(
                f"open_inds contains unknown tensor-network indices: "
                f"{unknown_open!r}"
            )
        output_inds = open_inds
    if output_inds and "output_inds" in contract_opts:
        raise TypeError(
            "pass open_inds to partitioned_expand rather than output_inds "
            "inside contract_opts"
        )

    def contract_term(term):
        network = _term_network(bp, term, projectors, open_inds=open_inds)
        return network.contract(
            output_inds=output_inds if output_inds else None,
            optimize=optimize,
            **contract_opts,
        )

    terms = _make_terms(
        partitions,
        form=form,
        include_residue=include_residue,
        max_terms=max_terms,
    )
    values = {}
    total = 0
    residue_value = None
    for term in terms:
        if term.residue and any(len(group) > 1 for group in partitions):
            # For a factorized multi-index partition G, Q_G is
            # I - product_{i in G} P_i, not product_i (I - P_i). Expand the
            # product of these complements into P-only contractions. This is
            # exact, though retaining such a residue can be more expensive.
            value = 0
            for size in range(len(partitions) + 1):
                for p_ids in combinations(range(len(partitions)), size):
                    p_inds = tuple(
                        index
                        for ident in p_ids
                        for index in partitions[ident]
                    )
                    temporary = PNEExpansionTerm(
                        partition_ids=p_ids,
                        p_inds=p_inds,
                        q_inds=(),
                    )
                    p_value = contract_term(temporary)
                    value = value + (-1) ** size * p_value
        else:
            value = contract_term(term)
        values[term] = value
        if term.residue:
            residue_value = value
        else:
            total = total + term.coefficient * value
    if include_residue:
        total = total + residue_value

    mantissa = bp.sign * total
    if strip_exponent:
        estimate = (mantissa, bp.exponent)
    else:
        estimate = mantissa * 10**bp.exponent
    return estimate, terms, values, residue_value, projectors


def partitioned_expand(
    tn,
    partition_inds=None,
    *,
    norm: str = "2norm",
    partitions=None,
    form: str = "linear",
    projectors=None,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = False,
    include_residue: bool = False,
    max_terms: int = 4096,
    optimize: str = "auto-hq",
    strip_exponent: bool = False,
    progbar: bool = False,
    contract_opts: dict[str, Any] | None = None,
    open_inds=None,
    allow_open: bool = False,
    **bp_opts,
) -> PNEExpansionResult:
    """Approximate a norm-1 or norm-2 contraction using PNE.

    Parameters
    ----------
    partition_inds : iterable, optional
        Flat list of pairwise virtual indices to partition independently.
    partitions : iterable of iterables, optional
        Factorized multi-index partitions. Each group is treated as one
        partition whose dominant projector is the product of its single-index
        projectors. Multi-index groups currently use the combinatorial form.
    form : {"linear", "combinatorial"}
        The paper's Eq. (4) linear form or Eq. (7) combinatorial form.
    projectors : dict, optional
        Explicit dominant projectors ``P[index]``. D1BP projectors are
        ``(d, d)`` matrices. D2BP projectors are ``(d*d, d*d)`` matrices or
        ``(d, d, d, d)`` tensors. Missing entries use rank-one BP projectors.
    include_residue : bool, optional
        Retain the all-``Q`` residue. This makes the expansion an exact
        identity for single-index partitions, but is normally ``False`` for
        the approximation.
    open_inds : iterable, optional
        Return an open contraction instead of a scalar. For D1BP these may be
        any network indices. For D2BP these are physical/output indices and
        the result carries both ket and bra output indices.
    allow_open : bool, optional
        Allow open scalar D1BP networks. This is needed for transfer tensors;
        selected PNE partitions must still be pairwise internal indices.

    Notes
    -----
    PNE does not require a converged BP fixed point. Accordingly,
    ``require_fixed_point`` defaults to ``False``. BP messages are still a
    convenient source of rank-one projectors; explicit ``projectors`` can be
    supplied when BP fails or when a higher-rank dominant subspace is known.
    """
    if run_bp and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer when run_bp=True")
    if allow_open and str(norm).lower() == "1norm" and run_bp:
        raise ValueError(
            "open D1BP PNE currently requires run_bp=False; supply explicit "
            "projectors or messages for the internal partition indices"
        )
    if not isinstance(max_terms, (int, np.integer)) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    if contract_opts is None:
        contract_opts = {}
    else:
        contract_opts = dict(contract_opts)

    bp, info = _build_bp(
        tn,
        norm=norm,
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=progbar,
        validate_graph=not allow_open,
    )
    if require_fixed_point and run_bp and not info.get("converged", False):
        raise RuntimeError(
            "partitioned_expand requires converged BP messages; pass "
            "require_fixed_point=False for the PNE's non-fixed-point mode"
        )

    partitions = _normalize_partitions(partition_inds, partitions)
    _validate_partitions(bp.tn, partitions, norm=str(norm).lower())
    estimate, terms, values, residue_value, projector_map = _partitioned_contract(
        bp,
        partitions,
        form=form,
        projectors=projectors,
        include_residue=include_residue,
        max_terms=max_terms,
        strip_exponent=strip_exponent,
        optimize=optimize,
        contract_opts=contract_opts,
        open_inds=() if open_inds is None else tuple(open_inds),
        normalize=not (allow_open and str(norm).lower() == "1norm"),
    )
    return PNEExpansionResult(
        estimate=estimate,
        norm=str(norm).lower(),
        form=str(form).lower(),
        partitions=partitions,
        terms=terms,
        term_values=values,
        residue_estimate=residue_value,
        residue_included=include_residue,
        open_inds=() if open_inds is None else tuple(open_inds),
        bp_converged=info.get("converged"),
        bp_iterations=info.get("iterations"),
        bp_max_mdiff=info.get("max_mdiff"),
        bp=bp,
        _projectors=projector_map,
        _contract_defaults={
            "optimize": optimize,
            "contract_opts": contract_opts,
        },
    )


def select_pne_partitions(
    tn,
    *,
    norm: str = "2norm",
    max_partitions: int = 1,
    candidate_inds=None,
    partition_opts: dict[str, Any] | None = None,
) -> PNEPartitionSelection:
    """Rank candidate indices by a one-index PNE residue estimate.

    Each candidate is tested with a single-index expansion retaining its
    residue. The candidate with the smallest relative residue is preferred
    for a subsequent approximate PNE because dropping that residue is then
    expected to be least damaging. BP is run only for the first candidate and
    its message snapshot is reused for the remaining candidates.

    This is a diagnostic/selection heuristic, not a rigorous error bound.
    """
    if not isinstance(max_partitions, (int, np.integer)) or max_partitions < 1:
        raise ValueError("max_partitions must be a positive integer")
    if candidate_inds is None:
        candidate_inds = tuple(sorted(tn.inner_inds(), key=repr))
    else:
        candidate_inds = tuple(candidate_inds)
    if not candidate_inds:
        raise ValueError("candidate_inds must contain at least one index")

    options = {} if partition_opts is None else dict(partition_opts)
    options.pop("partition_inds", None)
    options.pop("partitions", None)
    options.pop("norm", None)
    options["include_residue"] = True
    options["strip_exponent"] = False
    scores = []
    first = None
    for index in candidate_inds:
        call_options = dict(options)
        if first is not None:
            call_options["messages"] = first.messages
            call_options["run_bp"] = False
        result = partitioned_expand(
            tn,
            partition_inds=(index,),
            norm=norm,
            **call_options,
        )
        if first is None:
            first = result
        residue_norm = result.residue_norm
        if residue_norm is None:
            raise RuntimeError("single-index PNE did not return a residue")
        denominator = max(_value_norm(result.estimate), 1e-300)
        scores.append(
            PNEPartitionScore(
                index=index,
                residue_norm=residue_norm,
                relative_residue=residue_norm / denominator,
                result=result,
            )
        )

    scores = tuple(
        sorted(scores, key=lambda score: (score.relative_residue, repr(score.index)))
    )
    return PNEPartitionSelection(
        indices=tuple(score.index for score in scores[:max_partitions]),
        scores=scores,
    )


def recursive_partitioned_expand(
    tn,
    partition_levels,
    *,
    norm: str = "2norm",
    form: str = "linear",
    projectors=None,
    messages=None,
    gauges=None,
    run_bp: bool = True,
    bp_runner: str = "plain",
    relay_opts: dict[str, Any] | None = None,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    diis: bool | dict[str, Any] = False,
    damping: float = 0.0,
    update: str = "sequential",
    require_fixed_point: bool = False,
    include_residue: bool = False,
    max_terms: int = 4096,
    open_inds=None,
    allow_open: bool = False,
    optimize: str = "auto-hq",
    strip_exponent: bool = False,
    progbar: bool = False,
    contract_opts: dict[str, Any] | None = None,
    **bp_opts,
) -> RecursivePNEExpansionResult:
    """Apply a fixed sequence of disjoint PNE partition levels.

    Each level is expanded independently and the resulting terms are
    distributed across the previous level's terms. This implements the useful
    fixed-schedule form of recursive partitioning from Sec. IV A of the PNE
    paper: start with a coarse partition and refine the surviving terms with
    later levels. The caller controls the schedule; automatic contraction-cost
    driven repartition selection remains a policy layer above this primitive.

    ``partition_levels`` is an iterable such as
    ``(("e0", "e1"), ("e2", "e3"))``. Levels must be disjoint and currently
    contain single-index partitions. Use ``include_residue=True`` for an
    exact recursive identity; the usual approximation drops each level's
    residue.
    """
    levels = tuple(tuple(level) for level in partition_levels)
    if not levels or any(not level for level in levels):
        raise ValueError("partition_levels must contain non-empty levels")
    flat = tuple(index for level in levels for index in level)
    if len(set(flat)) != len(flat):
        raise ValueError("recursive partition levels must be disjoint")
    if not isinstance(max_terms, (int, np.integer)) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    if allow_open and str(norm).lower() == "1norm" and run_bp:
        raise ValueError(
            "open D1BP recursive PNE currently requires run_bp=False"
        )
    contract_opts = {} if contract_opts is None else dict(contract_opts)

    bp, info = _build_bp(
        tn,
        norm=norm,
        messages=messages,
        gauges=gauges,
        run_bp=run_bp,
        bp_runner=bp_runner,
        relay_opts=relay_opts,
        max_iterations=max_iterations,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        diis=diis,
        damping=damping,
        update=update,
        optimize=optimize,
        bp_opts=bp_opts,
        progbar=progbar,
        validate_graph=not allow_open,
    )
    if require_fixed_point and run_bp and not info.get("converged", False):
        raise RuntimeError(
            "recursive_partitioned_expand requires converged BP messages; "
            "pass require_fixed_point=False for PNE mode"
        )

    partitions = tuple((index,) for index in flat)
    _validate_partitions(bp.tn, partitions, norm=str(norm).lower())
    projectors = _projector_map(bp, partitions, projectors)
    form = str(form).lower()
    if form not in {"linear", "combinatorial"}:
        raise ValueError("form must be either 'linear' or 'combinatorial'")

    leaves = (
        PNEExpansionTerm(partition_ids=(), p_inds=(), q_inds=()),
    )
    offset = 0
    for level in levels:
        level_parts = tuple((index,) for index in level)
        level_terms = _make_terms(
            level_parts,
            form=form,
            include_residue=include_residue,
            max_terms=max_terms,
        )
        next_leaves = []
        for parent in leaves:
            for child in level_terms:
                if set(parent.p_inds + parent.q_inds).intersection(
                    child.p_inds + child.q_inds
                ):
                    raise ValueError("recursive partition levels must be disjoint")
                next_leaves.append(
                    PNEExpansionTerm(
                        partition_ids=parent.partition_ids
                        + tuple(offset + ident for ident in child.partition_ids),
                        p_inds=parent.p_inds + child.p_inds,
                        q_inds=parent.q_inds + child.q_inds,
                        coefficient=parent.coefficient * child.coefficient,
                        residue=parent.residue or child.residue,
                    )
                )
        leaves = tuple(next_leaves)
        offset += len(level_parts)
        if len(leaves) > max_terms:
            raise ValueError(
                f"recursive PNE would contain {len(leaves)} terms, "
                f"exceeding max_terms={max_terms}"
            )

    open_inds = () if open_inds is None else tuple(open_inds)
    if str(norm).lower() == "2norm":
        invalid_open = set(open_inds).difference(bp.output_inds)
        if invalid_open:
            raise ValueError(
                "D2BP open_inds must be physical/output indices; invalid "
                f"indices: {invalid_open!r}"
            )
        output_inds = tuple(open_inds) + tuple(
            f"__pne_bra_{repr(index)}" for index in open_inds
        )
    else:
        output_inds = open_inds
    if output_inds and "output_inds" in contract_opts:
        raise TypeError("use open_inds rather than output_inds in contract_opts")
    if not (allow_open and str(norm).lower() == "1norm" and open_inds):
        bp.normalize_message_pairs()
        bp.normalize_tensors()
    else:
        bp.normalize_message_pairs()

    values = {}
    total = 0
    for term in leaves:
        network = _term_network(bp, term, projectors, open_inds=open_inds)
        value = network.contract(
            output_inds=output_inds if output_inds else None,
            optimize=optimize,
            **contract_opts,
        )
        values[term] = value
        total = total + term.coefficient * value

    mantissa = bp.sign * total
    estimate = (
        (mantissa, bp.exponent)
        if strip_exponent
        else mantissa * 10**bp.exponent
    )
    return RecursivePNEExpansionResult(
        estimate=estimate,
        norm=str(norm).lower(),
        form=form,
        levels=levels,
        terms=leaves,
        term_values=values,
        residue_included=include_residue,
        bp_converged=info.get("converged"),
        bp_iterations=info.get("iterations"),
        bp_max_mdiff=info.get("max_mdiff"),
        bp=bp,
    )


# Short name matching the paper's abbreviation, while the descriptive name
# remains the primary public API.
pne_expand = partitioned_expand
