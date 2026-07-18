"""Compare BP, loop-series, loop-cluster, and PNE contractions.

This benchmark helper is intentionally library-like: callers construct the
norm-1 or norm-2 tensor network, then pass it to
:func:`benchmark_bp_expansions`. The returned records are JSON-ready through
``record.as_dict()`` and can be used from notebooks or a command-line driver.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from pepsy.bp import (
    loop_cluster_expand,
    loop_series_expand,
    partitioned_expand,
    recursive_partitioned_expand,
    select_pne_partitions,
    weight_pass,
)


def _numpy_value(value):
    if hasattr(value, "data") and hasattr(value, "inds"):
        value = value.data
    return np.asarray(value)


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return [_json_value(value[0]), _json_value(value[1])]
    array = _numpy_value(value)
    if array.size != 1:
        return {
            "shape": list(array.shape),
            "norm": float(np.linalg.norm(array.reshape(-1))),
        }
    scalar = array.reshape(-1)[0]
    if np.iscomplexobj(scalar):
        scalar = complex(scalar)
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    return float(scalar)


def _relative_error(estimate, exact):
    estimate = _numpy_value(estimate)
    exact = _numpy_value(exact)
    if estimate.shape != exact.shape:
        return float("nan")
    denominator = max(float(np.linalg.norm(exact.reshape(-1))), 1e-300)
    return float(np.linalg.norm((estimate - exact).reshape(-1)) / denominator)


@dataclass(frozen=True)
class ExpansionBenchmarkRecord:
    """One JSON-ready expansion benchmark result."""

    method: str
    estimate: Any
    exact: Any
    relative_error: float
    wall_seconds: float
    num_terms: int | None
    residue_norm: float | None
    bp_converged: bool | None
    error: str | None = None

    def as_dict(self):
        return {
            "method": self.method,
            "estimate": _json_value(self.estimate),
            "exact": _json_value(self.exact),
            "relative_error": self.relative_error,
            "wall_seconds": self.wall_seconds,
            "num_terms": self.num_terms,
            "residue_norm": self.residue_norm,
            "bp_converged": self.bp_converged,
            "error": self.error,
        }


def _run_record(method, function, exact):
    start = time.perf_counter()
    try:
        result = function()
    except Exception as exc:  # benchmark tables should retain failed methods
        return ExpansionBenchmarkRecord(
            method=method,
            estimate=None,
            exact=exact,
            relative_error=float("nan"),
            wall_seconds=time.perf_counter() - start,
            num_terms=None,
            residue_norm=None,
            bp_converged=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    if method == "loop_cluster":
        num_terms = (
            len(result.region_counts)
            if result.region_counts is not None
            else None
        )
    else:
        num_terms = len(result.terms)
    return ExpansionBenchmarkRecord(
        method=method,
        estimate=result.estimate,
        exact=exact,
        relative_error=_relative_error(result.estimate, exact),
        wall_seconds=time.perf_counter() - start,
        num_terms=num_terms,
        residue_norm=getattr(result, "residue_norm", None),
        bp_converged=result.bp_converged,
    )


def benchmark_bp_expansions(
    tn,
    *,
    norm: str = "2norm",
    exact=None,
    loop_cutoff: int = 4,
    cluster_cutoff: int = 4,
    partition_inds=None,
    recursive_levels=None,
    partition_form: str = "linear",
    include_residue: bool = False,
    max_iterations: int = 1000,
    tol: float = 5e-6,
    optimize: str = "auto-hq",
    contract_opts: dict[str, Any] | None = None,
    auto_select: bool = False,
    max_partitions: int = 1,
    candidate_inds=None,
    partition_opts: dict[str, Any] | None = None,
    weight_passing_rank: int | None = None,
    weight_passing_opts: dict[str, Any] | None = None,
):
    """Benchmark the available BP corrections on one scalar contraction.

    Parameters
    ----------
    tn
        A closed scalar network for ``norm="1norm"`` or a PEPS-like network
        for ``norm="2norm"``.
    exact
        Optional exact reference. If omitted, ``tn.contract()`` is used.
    partition_inds
        PNE partition indices. If omitted, the first internal index is used,
        unless ``auto_select=True``.
    auto_select
        Rank candidate indices by the one-index PNE residue heuristic.
    max_partitions, candidate_inds, partition_opts
        Controls for ``auto_select``.
    recursive_levels
        Optional fixed recursive PNE schedule, e.g.
        ``(("e0", "e1"), ("e2",))``.
    weight_passing_rank
        If set, add a higher-rank PNE record produced by Appendix-C weight
        passing. This route currently targets closed pairwise norm-1 inputs.
    weight_passing_opts
        Options forwarded to :func:`pepsy.bp.weight_pass`.
    """
    if exact is None:
        exact = tn.contract(optimize=optimize, **(contract_opts or {}))
    contract_opts = {} if contract_opts is None else dict(contract_opts)
    common = {
        "norm": norm,
        "max_iterations": max_iterations,
        "tol": tol,
        "optimize": optimize,
        "contract_opts": contract_opts,
    }
    internal = tuple(sorted(tn.inner_inds(), key=repr))
    if partition_inds is None and auto_select:
        selection_options = {
            "max_iterations": max_iterations,
            "tol": tol,
        }
        selection_options.update(partition_opts or {})
        selection = select_pne_partitions(
            tn,
            norm=norm,
            max_partitions=max_partitions,
            candidate_inds=candidate_inds,
            partition_opts=selection_options,
        )
        partition_inds = selection.indices
    if partition_inds is None:
        if not internal:
            raise ValueError("no internal indices available for the PNE benchmark")
        partition_inds = (internal[0],)

    records = []
    records.append(
        _run_record(
            "bp",
            lambda: loop_series_expand(
                tn.copy(),
                gloops=0,
                multi_excitation_correct=False,
                **common,
            ),
            exact,
        )
    )
    records.append(
        _run_record(
            "loop_series",
            lambda: loop_series_expand(
                tn.copy(), gloops=loop_cutoff, **common
            ),
            exact,
        )
    )
    records.append(
        _run_record(
            "loop_cluster",
            lambda: loop_cluster_expand(
                tn.copy(), gloops=cluster_cutoff, **common
            ),
            exact,
        )
    )
    records.append(
        _run_record(
            "pne",
            lambda: partitioned_expand(
                tn.copy(),
                partition_inds=partition_inds,
                form=partition_form,
                include_residue=include_residue,
                **common,
            ),
            exact,
        )
    )
    if recursive_levels is not None:
        records.append(
            _run_record(
                "pne_recursive",
                lambda: recursive_partitioned_expand(
                    tn.copy(),
                    recursive_levels,
                    form=partition_form,
                    include_residue=include_residue,
                    **common,
                ),
                exact,
            )
        )
    if weight_passing_rank is not None:
        weight_options = {} if weight_passing_opts is None else dict(weight_passing_opts)
        records.append(
            _run_record(
                "pne_weight_pass",
                lambda: _weight_passing_expansion(
                    tn,
                    norm=norm,
                    partition_inds=partition_inds,
                    rank=weight_passing_rank,
                    include_residue=include_residue,
                    optimize=optimize,
                    contract_opts=contract_opts,
                    weight_options=weight_options,
                ),
                exact,
            )
        )
    return tuple(records)


def _weight_passing_expansion(
    tn,
    *,
    norm,
    partition_inds,
    rank,
    include_residue,
    optimize,
    contract_opts,
    weight_options,
):
    weight_result = weight_pass(tn.copy(), **weight_options)
    return partitioned_expand(
        weight_result.network,
        partition_inds=partition_inds,
        norm=norm,
        projectors=weight_result.projectors(rank=rank),
        run_bp=False,
        include_residue=include_residue,
        optimize=optimize,
        contract_opts=contract_opts,
    )
