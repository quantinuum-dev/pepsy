"""Matched endpoints and independent exact references in the comparison."""

import pytest

from examples import dmt_convergence as study


def test_mps_comparison_agrees_with_exact_untruncated_circuit():
    model = study.bench.problem(2, 2, .1)
    references = study.bench.exact_references(model, .1, 2, 1)
    row = study.mps_trajectory(model, references, chi=4, steps=2, dt=.1)
    assert row['completed_steps'] == 2
    assert row['samples'][-1]['hs_relative_trotter'] < 2e-12
    assert row['samples'][-1]['min_eigenvalue_hermitian_part'] > 0


def test_study_compares_equal_final_times_and_keeps_rank_meaning(tmp_path):
    result = study.study((2, 2), chis=[16], dts=[.1, .05], final_time=.2,
                         methods=['dmt', 'mps'], output=tmp_path/'results.json')
    assert len(result['rows']) == 4
    for row in result['rows']:
        assert row['status'] == 'completed'
        assert row['samples'][-1]['time'] == pytest.approx(.2)
        assert row['samples'][-1]['hs_relative_trotter'] < 2e-12
    with pytest.raises(ValueError, match='divide'):
        study.study((2, 2), chis=[16], dts=[.07], final_time=.2)
