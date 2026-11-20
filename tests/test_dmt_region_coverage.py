"""Physical-support coverage does not infer unrequested joint correlations."""

import pytest

from pepsy.operators import complete_dmt_regions, square_lattice_dmt_regions
from pepsy.tensors import OneDMap


@pytest.mark.parametrize('mode', ['snake-row-major', 'row-major', 'snake'])
def test_plaquettes_inherit_l_shapes_but_require_straight_triples(mode):
    mapper = OneDMap(4, 4, mode=mode)
    plaquettes = square_lattice_dmt_regions(4, 4, mapper=mapper)
    l_shapes = square_lattice_dmt_regions(4, 4, mapper=mapper, kind='l_shapes')
    triples = square_lattice_dmt_regions(4, 4, mapper=mapper, kind='triples')
    assert complete_dmt_regions(16, plaquettes, required=l_shapes) == plaquettes
    combined = complete_dmt_regions(16, plaquettes, required=triples)
    assert len(combined) == 25  # nine plaquettes and sixteen straight triples
    assert all(any(set(s) <= set(r) for r in combined) for s in plaquettes + triples)


def test_joint_six_spin_family_already_covers_every_triple_and_pair_on_16_sites():
    regions = tuple((i, i+1, i+2, i+5, i+6, i+7) for i in range(9))
    required = [tuple(range(i, i+3)) for i in range(14)]
    required += [(i, i+1) for i in range(15)]
    assert complete_dmt_regions(16, regions, required=required) == regions


def test_overlapping_parents_do_not_cover_their_joint_union():
    parents = ((0, 1, 3, 4), (1, 2, 4, 5))
    joint = (0, 1, 2, 3, 4, 5)
    assert complete_dmt_regions(6, parents, required=[joint]) == (joint,)
    # Two separate triples likewise do not imply the joint six-spin marginal.
    assert complete_dmt_regions(8, [(0, 1, 2), (5, 6, 7)],
                                required=[(0, 1, 2, 5, 6, 7)]) == ((0, 1, 2, 5, 6, 7),)


@pytest.mark.parametrize('nsites,regions,required', [
    (True, [(0,)], []), (0, [(0,)], []), (4, [(0,)], [(4,)]),
    (4, [(0,)], [(1, 1)]), (4, [(0,)], [(True,)]), (4, [], []),
    (4, None, [(1,)]),
])
def test_coverage_validates_even_contained_requests(nsites, regions, required):
    with pytest.raises(ValueError):
        complete_dmt_regions(nsites, regions, required=required)
