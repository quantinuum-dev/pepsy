"""Geometry-only reuse and exact-update shortcuts retain physical targets."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.operators import _pauli_dmt as dmt
from pepsy.operators import _dmt_region_selection as selection
from test_pauli_dmt import _density
from test_pauli_dmt_large import product_mixture


@pytest.mark.parametrize('region_selection', ['smallest', 'min_cover', 'rank_aware'])
def test_exact_gate_skips_region_vectors_but_preserves_the_complete_target(monkeypatch, region_selection):
    rho = _density()
    before = rho.to_dense()
    workspace = dmt._PauliDMTWorkspace(rho, max_bond=64, regions=[(0, 1, 3, 4)],
                                      region_selection=region_selection)

    def forbidden(*args, **kwargs):
        raise AssertionError('an exact factorization constructed unused region maps')

    monkeypatch.setattr(dmt, '_region_vectors', forbidden)
    monkeypatch.setattr(selection, 'rank_aware_vectors', forbidden)
    u = expm(-.23j * np.kron(np.diag([1., -1.]), [[0., 1.], [1., 0.]]))
    transfer = dmt._gate_transfer(u, workspace.cores[0], 'conjugate')
    workspace.split_pair(2, transfer)
    full = np.kron(np.eye(4), np.kron(u, np.eye(4)))
    np.testing.assert_allclose(PauliMPO.from_pauli_cores(workspace.cores).to_dense(),
                               full @ before @ full.conj().T, atol=2e-13)
    # The shortcut still rejects an unresolved density trace.
    with pytest.raises(ValueError, match='nonzero trace'):
        workspace.split_pair(2, np.zeros_like(transfer))


def test_candidate_cache_reuses_only_geometry_and_is_bounded_under_remapping(monkeypatch):
    rho, _, _ = product_mixture(9, rank=48, strength=.5)
    regions = square_lattice_dmt_regions(3, 3)
    workspace = dmt._PauliDMTWorkspace(rho, max_bond=30, regions=regions,
                                      region_selection='rank_aware', diagnostics=True)
    builds, contractions = [], []
    plan, vectors = selection.rank_aware_plan, selection._region_vectors

    def planned(*args):
        builds.append(args)
        return plan(*args)

    def projected(*args):
        contractions.append(args[1])
        return vectors(*args)

    monkeypatch.setattr(selection, 'rank_aware_plan', planned)
    monkeypatch.setattr(selection, '_region_vectors', projected)
    workspace.split_pair(4)
    first = workspace.region_candidate_plans[4][1]
    workspace.move_center(2)
    workspace.split_pair(4)
    assert len(builds) == 1 and len(contractions) == 2
    assert workspace.region_candidate_plans[4][1] is first
    clone = workspace.copy()
    assert clone.region_candidate_plans is not workspace.region_candidate_plans
    assert clone.region_candidate_plans[4][1] is first
    clone.regions = ((0, 4, 8), (1, 5, 7))
    clone.split_pair(4)
    assert len(builds) == 2 and len(contractions) == 3
    assert len(clone.region_candidate_plans) == 1
    assert workspace.region_candidate_plans[4][1] is first


def test_planned_and_fresh_rank_selection_return_identical_directions():
    rho = _density()
    workspace = dmt._PauliDMTWorkspace(rho, max_bond=20, regions=[(0, 1, 3, 4)],
                                      region_selection='rank_aware')
    workspace.move_center(2)
    a, b = workspace.cores[2:4]
    matrix = np.einsum('abp,bcq->apqc', a, b).reshape(a.shape[0]*4, b.shape[1]*4)
    args = (workspace.cores, 2, workspace.regions, workspace.eye,
            workspace.environments, matrix, None, None)
    plan = selection.rank_aware_plan(workspace.regions, 2, 4)
    for theta in (1., .9):
        # Changing a tensor changes projected directions even with the same plan.
        workspace.cores[1] = workspace.cores[1] * theta
        workspace.environments.invalidate(1, 2)
        fresh = selection.rank_aware_vectors(*args)
        cached = selection.rank_aware_vectors(*args, plan=plan)
        for a, b in zip(fresh, cached):
            np.testing.assert_array_equal(a, b)
