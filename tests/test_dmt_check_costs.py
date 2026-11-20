"""Fast replay defaults, one final sweep check, and trace coverage."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import PauliMPO, compress_mpo_dmt
from pepsy.operators import _pauli_dmt as kernel
from pepsy.optimizers import MpoOptimizer, PauliDMT
from test_mpo_dmt_replay import gate, mixture


def _watch_panels(monkeypatch, name, replacement):
    import pepsy.operators.mpo_dmt as ordinary
    monkeypatch.setattr(kernel, name, replacement)
    monkeypatch.setattr(ordinary, name, replacement)


@pytest.mark.parametrize("method", ["one-gate", "persistent", "mpo"])
@pytest.mark.parametrize("regions", [None, [(0, 1, 2, 3), (2, 3, 4, 5)]])
def test_replay_defaults_skip_full_scans_and_keep_density_trace_checks(monkeypatch, method, regions):
    def forbidden(*args, **kwargs):
        raise AssertionError("default replay contracted full-region verification panels")

    _watch_panels(monkeypatch, "_local_trace_data", forbidden)
    _watch_panels(monkeypatch, "_region_trace_data", forbidden)
    if method == "mpo":
        initial, _, _ = mixture(6, rank=12)
        opt = MpoOptimizer(initial, [(gate(3), (1, 2)), (gate(4), (2, 3))],
                           chi=24, mode="dmt", contraction_opt="greedy")
        opt.run(dmt_regions=regions, dmt_diagnostics=True, fidelity_samples=0)
        reports = opt.get_dmt_reports()
    else:
        initial = PauliMPO.identity(6, coefficient=1 / 64)
        if method == "one-gate":
            _, report = initial.apply_gate_dmt(gate(), (2, 3), max_bond=24,
                                               regions=regions, return_report=True)
        else:
            opt = PauliDMT(initial, max_bond=24, regions=regions)
            opt.apply_gate(gate(), (2, 3))
            report = opt.report
        reports = (report,)
    assert reports and all(not report.check_regions for report in reports)
    assert all(report.max_relative_trace_error is not None for report in reports)


@pytest.mark.parametrize("ordinary", [False, True])
def test_standalone_strict_sweep_checks_regions_only_once_against_original(monkeypatch, ordinary):
    from examples.pauli_dmt_benchmark import _mixture

    initial, _, _ = _mixture(9, 48, "complex128")
    regions = [tuple(range(j, j + 4)) for j in range(6)]
    original, calls = kernel._region_trace_data, []

    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    _watch_panels(monkeypatch, "_region_trace_data", observe)
    if ordinary:
        out, report = compress_mpo_dmt(initial.to_mpo(), max_bond=24,
                                       regions=regions, return_report=True)
    else:
        out, report = initial.compress_dmt(max_bond=24, regions=regions, return_report=True)
    assert len(calls) == 2
    assert report.check_regions and not report.preservation_errors
    assert report.max_relative_trace_error is not None
    assert any(b.input_rank > b.final_bond for b in report.bond_reports)
    assert out is not initial


@pytest.mark.parametrize("ordinary", [False, True])
def test_final_sweep_check_still_rejects_corruption_transactionally(monkeypatch, ordinary):
    original = kernel._PauliDMTWorkspace.split_pair

    def corrupt_last(self, bond, *args, **kwargs):
        report = original(self, bond, *args, **kwargs)
        if bond == len(self.cores) - 2:
            self.cores[-1] = self.cores[-1] * (1 + 1e-5)
        return report

    monkeypatch.setattr(kernel._PauliDMTWorkspace, "split_pair", corrupt_last)
    if ordinary:
        initial = qtn.MPO_identity(6) / 64
        call = lambda: compress_mpo_dmt(initial, max_bond=8, inplace=True)
    else:
        initial = PauliMPO.identity(6, coefficient=1 / 64)
        call = lambda: initial.compress_dmt(max_bond=8, inplace=True)
    before = initial.to_dense().copy()
    with pytest.raises(FloatingPointError, match="local marginals"):
        call()
    np.testing.assert_array_equal(initial.to_dense(), before)


@pytest.mark.parametrize("method", ["persistent", "mpo"])
def test_explicit_full_replay_checks_also_cover_default_three_site_windows(monkeypatch, method):
    original, calls = kernel._local_trace_data, []

    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    _watch_panels(monkeypatch, "_local_trace_data", observe)
    if method == "persistent":
        opt = PauliDMT(PauliMPO.identity(6, coefficient=1 / 64), max_bond=16,
                       check_regions=True)
        opt.apply_gate(gate(), (2, 3))
        report = opt.report
        assert len(calls) == 2
    else:
        initial, _, _ = mixture(6, rank=8)
        opt = MpoOptimizer(initial, [(gate(), (1, 2)), (gate(31), (2, 3))],
                           chi=24, mode="dmt", contraction_opt="greedy")
        opt.run(dmt_check_regions=True, dmt_diagnostics=True, fidelity_samples=0)
        report = opt.get_dmt_reports()[-1]
        assert len(calls) == 4
    assert report.check_regions


def test_traceless_operator_replay_does_not_enable_density_trace_guard():
    initial = PauliMPO.from_terms(5, [(1., "ZIIII")]).to_mpo()
    opt = MpoOptimizer(initial, [(gate(), (1, 2)), (gate(31), (2, 3))],
                       chi=24, mode="dmt", contraction_opt="greedy")
    opt.run(dmt_variant="operator", dmt_diagnostics=True, fidelity_samples=0)
    assert all(r.max_relative_trace_error is None for r in opt.get_dmt_reports())


@pytest.mark.parametrize("ordinary", [False, True])
def test_off_report_resolves_an_explicit_full_request_to_disabled(monkeypatch, ordinary):
    def forbidden(*args, **kwargs):
        raise AssertionError("off mode ran a full-region check")

    _watch_panels(monkeypatch, "_local_trace_data", forbidden)
    _watch_panels(monkeypatch, "_region_trace_data", forbidden)
    if ordinary:
        opt = MpoOptimizer(qtn.MPO_identity(5) / 32, [(gate(), (1, 2))],
                           chi=16, mode="dmt", contraction_opt="greedy")
        opt.run(dmt_check_regions=True, dmt_preservation_check="off",
                dmt_diagnostics=True, fidelity_samples=0)
        report = opt.get_dmt_reports()[-1]
    else:
        opt = PauliDMT(PauliMPO.identity(5, coefficient=1 / 32), max_bond=16,
                       check_regions=True, preservation_check="off")
        opt.apply_gate(gate(), (1, 2))
        report = opt.report
    assert not report.check_regions
    assert report.max_relative_trace_error is None
