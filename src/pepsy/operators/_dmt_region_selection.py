"""State-dependent choice among complete physical-string covers."""

from itertools import product

from ._pauli_dmt import (
    _basis, _cat, _dag, _epsilon, _float, _norm, _region_cover, _region_pattern, _region_vectors,
    _svd, _unit,
)
from .pauli_dmt_geometry import _maximal_regions


def rank_aware_vectors(cores, bond, regions, eye, environments, matrix, left_map, right_map):
    """Compare actual protected ranks without dropping any requested string.

    Candidate covers include the old smaller-half rule, both minimum-cover
    orientations and both complete one-sided covers. This is a bounded
    search, not a globally minimum-rank cover algorithm. Native contractions
    are shared over the union of their physical patterns.
    """
    regions = _maximal_regions(regions)
    dim = cores[0].shape[2]
    all_left, all_right = {(): None}, {(): None}
    for region in regions:
        ls, rs = tuple(s for s in region if s <= bond), tuple(s for s in region if s > bond)
        if not ls or not rs:
            continue
        for sites, inventory in [(ls, all_left), (rs, all_right)]:
            for labels in product(range(dim), repeat=len(sites)):
                inventory[tuple((s,p) for s,p in zip(sites,labels) if p)] = None
    all_patterns = tuple(all_left), tuple(all_right)
    lv, rv = _region_vectors(cores, bond, all_patterns, eye, environments)
    if left_map is not None:
        lv, rv = _dag(left_map) @ lv, right_map @ rv
    candidates = [
        _region_pattern(regions, bond, dim, selection="min_cover"),
        _region_pattern(regions, bond, dim),
        _region_cover(regions, bond, dim, 0, reverse=True),
        (all_patterns[0], ((),)),
        (((),), all_patterns[1]),
    ]
    indices = [dict(zip(patterns, range(len(patterns)))) for patterns in all_patterns]
    rtol = max(matrix.shape) * _epsilon(matrix)
    support = None
    if matrix.shape[0] != matrix.shape[1]:
        u, _, vh, _ = _svd(_unit(matrix))
        support = u, vh
    best, best_score, seen = None, None, set()
    lower_bound = sum(_float(_norm(v[:, :1])) > 0 for v in (lv, rv))
    for patterns in candidates:
        if patterns in seen:
            continue
        seen.add(patterns)
        selected = tuple(_cat([vectors[:, index[p]:index[p]+1] for p in side], axis=1)
                         for vectors, index, side in zip((lv,rv), indices, patterns))
        scoring = selected if support is None else (_dag(support[0]) @ selected[0], support[1] @ selected[1])
        p, q = (_basis(v, rtol).shape[1] for v in scoring)
        score = p+q, max(p,q), sum(map(len,patterns))
        if best_score is None or score < best_score:
            best, best_score = selected, score
        if best_score[0] == lower_bound:
            break
    return best
