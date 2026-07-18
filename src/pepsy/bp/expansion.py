"""Common entry point for BP loop and partitioned corrections.

The two implementations intentionally remain separate:

``expansion="series"``
    Edge-resolved ``P + Q`` loop series. Integer ``gloops`` means maximum
    number of excited bonds, and explicit terms are :class:`LoopSeriesTerm`.

``expansion="cluster"``
    Region/inclusion--exclusion loop-cluster expansion. Integer ``gloops``
    means maximum number of tensors in a generalized-loop region, and
    explicit terms are tensor-id regions.

``expansion="pne"``
    Partitioned network expansion. It uses ``partition_inds`` or
    ``partitions`` rather than ``gloops``.

This dispatcher supplies a consistent selection surface without hiding that
the two methods have different combinatorics and different convergence
behavior.
"""

from __future__ import annotations

from typing import Any

from .cluster import loop_cluster_expand
from .pne import partitioned_expand, recursive_partitioned_expand
from .series import loop_series_expand

__all__ = ["loop_expand"]

_EXPANSION_ALIASES = {
    "series": "series",
    "loop_series": "series",
    "cluster": "cluster",
    "loop_cluster": "cluster",
    "pne": "pne",
    "partitioned": "pne",
    "partitioned_network": "pne",
    "pne_recursive": "pne_recursive",
    "recursive_pne": "pne_recursive",
}


def loop_expand(
    tn,
    gloops=None,
    *,
    expansion: str = "series",
    norm: str = "2norm",
    **options: Any,
):
    """Choose the edge loop series, loop-cluster, or PNE expansion.

    Parameters
    ----------
    tn : TensorNetwork
        The network to contract. ``norm="1norm"`` expects a closed scalar
        pairwise network; ``norm="2norm"`` expects a PEPS-like network.
    gloops : int or iterable, optional
        The cutoff or explicit loop specification. Its meaning depends on
        ``expansion`` and is intentionally not silently converted:

        * ``"series"``: integer = maximum excited-bond degree; iterable =
          edge-resolved :class:`LoopSeriesTerm` objects (or legacy tensor
          regions).
        * ``"cluster"``: integer = maximum tensor-region size; iterable =
          generalized-loop tensor regions.
        * ``"pne"``: pass ``partition_inds`` or factorized ``partitions``
          through ``options``; ``gloops`` is not used.
    expansion : {"series", "cluster", "pne", "pne_recursive"}, optional
        Which correction to use. ``"loop_series"``, ``"loop_cluster"``, and
        ``"recursive_pne"`` are accepted aliases. The default is the
        edge-resolved loop series.
    norm : {"2norm", "1norm"}, optional
        BP family shared by both direct APIs.
    options
        Options for the selected direct API. Series-only options include
        ``multi_excitation_correct``, ``tol_correction``, and
        ``maxiter_correction``. Cluster-only options include ``combine``,
        ``autocomplete``, and ``autoreduce``. Shared BP options such as
        ``messages``, ``gauges``, ``run_bp``, ``bp_runner``, convergence
        controls, ``cache``, ``optimize``, and ``contract_opts`` are passed
        through unchanged. PNE-specific options include ``partition_inds``,
        ``partitions``, ``form``, ``projectors``, and ``include_residue``.

    Returns
    -------
    LoopSeriesResult, LoopClusterResult, or PNEExpansionResult
        The selected result type. Inspect ``result.expansion`` and
        ``result.cutoff_kind`` when writing code that accepts either method.

    Notes
    -----
    This is a selector, not an attempt to make the expansions
    mathematically interchangeable. Use ``loop_series_expand`` directly when
    you need individual ``Q``-bond terms, chord subsets, or multi-excitation
    suppression factors. Use ``loop_cluster_expand`` directly when you need
    region counting numbers, product/sum formulas, or cluster autoreduction.
    """
    try:
        key = _EXPANSION_ALIASES[str(expansion).lower()]
    except KeyError as exc:
        choices = "'series', 'cluster', or 'pne'"
        raise ValueError(
            f"expansion must be {choices}; got {expansion!r}. "
            "Choose 'series' for edge-resolved P/Q terms, 'cluster' for "
            "tensor-region inclusion-exclusion, or 'pne' for partitioned "
            "network expansion."
        ) from exc

    if key == "series":
        unsupported = {
            "combine",
            "autocomplete",
            "autoreduce",
        }.intersection(options)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(
                f"{names} is only a loop-cluster option; choose "
                "expansion='cluster' for tensor-region inclusion-exclusion"
            )
        return loop_series_expand(tn, gloops, norm=norm, **options)
    if key == "cluster":
        unsupported = {
        "multi_excitation_correct",
        "tol_correction",
        "maxiter_correction",
        }.intersection(options)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(
                f"{names} is only a loop-series option; choose "
                "expansion='series' for edge-resolved P/Q terms"
            )
        return loop_cluster_expand(tn, gloops, norm=norm, **options)
    if gloops is not None:
        raise TypeError(
            "PNE uses partition_inds or partitions, not gloops; choose "
            "partitioned_expand(..., partition_inds=...)"
        )
    if key == "pne_recursive":
        return recursive_partitioned_expand(tn, norm=norm, **options)
    return partitioned_expand(tn, norm=norm, **options)
