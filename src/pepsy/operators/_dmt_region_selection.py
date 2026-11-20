"""State-dependent choice among complete physical-string covers."""

from itertools import product
from dataclasses import dataclass

from ._pauli_dmt import (
    _basis, _cat, _dag, _epsilon, _float, _norm, _region_cover, _region_pattern, _region_vectors,
    _svd, _unit,
)
from .pauli_dmt_geometry import _maximal_regions


@dataclass(frozen=True)
class _RankAwarePlan:
    patterns: tuple
    candidates: tuple


def rank_aware_plan(regions, bond, dim):
    """Compile immutable string/index candidates; contains no state arrays."""
    regions = _maximal_regions(regions)
    all_left, all_right = {(): None}, {(): None}
    for region in regions:
        ls, rs = tuple(s for s in region if s <= bond), tuple(s for s in region if s > bond)
        if not ls or not rs:
            continue
        for sites, inventory in [(ls, all_left), (rs, all_right)]:
            for labels in product(range(dim), repeat=len(sites)):
                inventory[tuple((s,p) for s,p in zip(sites,labels) if p)] = None
    all_patterns = tuple(all_left), tuple(all_right)
    candidates = [
        _region_pattern(regions, bond, dim, selection="min_cover"),
        _region_pattern(regions, bond, dim),
        _region_cover(regions, bond, dim, 0, reverse=True),
        (all_patterns[0], ((),)),
        (((),), all_patterns[1]),
    ]
    indices = [dict(zip(patterns, range(len(patterns)))) for patterns in all_patterns]
    compiled = tuple(dict.fromkeys(
        tuple(tuple(index[p] for p in side) for index, side in zip(indices, patterns))
        for patterns in candidates
    ))
    return _RankAwarePlan(all_patterns, compiled)


def rank_aware_vectors(cores, bond, regions, eye, environments, matrix, left_map, right_map,
                       *, plan=None):
    """Score complete candidate covers using the current numerical ranks.

    Candidate geometry may be reused; projected vectors and their ranks may
    not. This bounded search is not a globally minimum-rank cover algorithm.
    """
    if plan is None:
        plan = rank_aware_plan(regions, bond, cores[0].shape[2])
    lv, rv = _region_vectors(cores, bond, plan.patterns, eye, environments)
    if left_map is not None:
        lv, rv = _dag(left_map) @ lv, right_map @ rv
    rtol = max(matrix.shape) * _epsilon(matrix)
    support = None
    if matrix.shape[0] != matrix.shape[1]:
        u, _, vh, _ = _svd(_unit(matrix))
        support = u, vh
    best, best_score = None, None
    lower_bound = sum(_float(_norm(v[:, :1])) > 0 for v in (lv, rv))
    for candidate in plan.candidates:
        selected = tuple(_cat([vectors[:, i:i+1] for i in side], axis=1)
                         for vectors, side in zip((lv,rv), candidate))
        scoring = selected if support is None else (_dag(support[0]) @ selected[0], support[1] @ selected[1])
        p, q = (_basis(v, rtol).shape[1] for v in scoring)
        score = p+q, max(p,q), sum(map(len,candidate))
        if best_score is None or score < best_score:
            best, best_score = selected, score
        if best_score[0] == lower_bound:
            break
    return best
