"""Larger protected windows against physical oracles and unchanged plans."""

import numpy as np
import pytest

from pepsy.operators import PauliMPO
from pepsy.operators import _pauli_dmt as dmt
from examples.dmt_window_scaling import (
    compression_case, dynamics_case, large_dynamics_case, mixture_window, windows,
)
from examples.pauli_dmt_benchmark import _mixture


def legacy_patterns(regions, nsites, physical_dim=4, *, identity_label=0, selection="smallest"):
    regions = dmt._maximal_regions(regions)
    return tuple(dmt._region_pattern(regions, bond, physical_dim,
                                     identity_label=identity_label, selection=selection)
                 for bond in range(nsites - 1))


@pytest.mark.parametrize("selection", ["smallest", "min_cover", "rank_aware"])
@pytest.mark.parametrize("kind", ["contiguous", "gapped", "nested"])
@pytest.mark.parametrize("physical_dim,identity", [(2, 0), (4, 0), (3, 1)])
def test_crossing_region_index_retains_every_protected_string_and_order(selection, kind, physical_dim, identity):
    regions = {"contiguous": windows(12, 5),
               "gapped": ((0, 2, 7), (1, 4, 8), (3, 6, 11)),
               "nested": ((0,), (0, 1), (0, 1, 2), (3, 4, 5, 6), (4, 5), (9, 10, 11))}[kind]
    assert dmt._region_patterns(regions, 12, physical_dim, selection=selection,
                                identity_label=identity) == legacy_patterns(
        regions, 12, physical_dim, selection=selection, identity_label=identity)


@pytest.mark.parametrize("width,chi", [(4, 24), (5, 40)])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("method", ["pauli", "mpo"])
def test_larger_windows_are_actually_truncated_and_match_product_mixture_oracle(width, chi, backend, method):
    if backend == "torch":
        pytest.importorskip("torch").set_num_threads(1)
    row = compression_case(9, chi, width, backend=backend, method=method, repeats=1,
                           check_regions=True)
    assert row["final_max_bond"] <= chi
    assert row["rank_reduced_bonds"] > 0
    assert row["checked_windows"] == 10 - width
    assert row["max_window_frobenius_error"] < 2e-12
    assert row["trace_defect"] < 2e-12
    assert not row["preservation_errors"]


def test_mixture_window_oracle_matches_independent_dense_partial_trace():
    from test_pauli_dmt_lattice import dense_marginal

    rho, weights, local = _mixture(6, 8, "complex128")
    region = (0, 2, 3, 5)
    np.testing.assert_allclose(mixture_window(weights, local, region),
                               dense_marginal(rho.to_dense(), 6, region), atol=1e-14)


@pytest.mark.parametrize("width", [4, 5])
def test_expanded_protection_preserves_gated_targets_and_has_lossy_dynamics(width):
    row = dynamics_case(8, 32, width, backend="numpy", steps=6, dt=.3)
    assert row["lossy_splits"] > 0
    assert row["max_audited_protected_window_error"] < 2e-12
    assert row["final_max_bond"] <= 32
    assert max(s["trace_defect"] for s in row["samples"]) < 2e-12
    assert row["samples"][-1]["rdm5_rmse"] > 1e-6


def test_untruncated_dynamics_reference_agrees_for_mps_and_dmt():
    for method in ("mps", "pauli"):
        row = dynamics_case(5, 16, 5 if method == "pauli" else None,
                            backend="numpy", method=method, steps=2, dt=.1)
        assert row["samples"][-1]["rdm5_rmse"] < 2e-12


def test_large_chain_gate_benchmark_audits_current_targets_after_actual_loss():
    row = large_dynamics_case(9, 24, 4, backend="numpy", steps=2)
    assert row["rank_reduced_gates"] > 0
    assert row["audited_gates"] == 8
    assert row["max_audited_protected_window_error"] < 2e-12
    assert row["trace_defect"] < 2e-12
    assert row["final_max_bond"] <= 24


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_indexed_planning_leaves_compressed_arrays_bitwise_unchanged(monkeypatch, backend):
    rho, _, _ = _mixture(9, 48, "complex128")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        rho = PauliMPO.from_pauli_cores([torch.tensor(a, dtype=torch.complex128)
                                       for a in rho.to_pauli_cores(copy=False)])
    regions = windows(9, 5)
    indexed = rho.compress_dmt(max_bond=40, regions=regions)
    monkeypatch.setattr(dmt, "_region_patterns", legacy_patterns)
    legacy = rho.compress_dmt(max_bond=40, regions=regions)
    assert isinstance(indexed, PauliMPO)
    for a, b in zip(indexed.to_pauli_cores(), legacy.to_pauli_cores()):
        np.testing.assert_array_equal(a, b)


def test_benchmark_records_numerical_failure_and_continues_other_cases(monkeypatch, tmp_path):
    import json
    from examples import dmt_window_scaling as scaling

    original = scaling.compression_case

    def fail_first(length, chi, width, **kwargs):
        if chi == 1:
            raise np.linalg.LinAlgError("injected SVD convergence failure")
        return original(length, chi, width, **kwargs)

    monkeypatch.setattr(scaling, "compression_case", fail_first)
    output = tmp_path / "results.json"
    monkeypatch.setattr("sys.argv", ["dmt_window_scaling.py", "--backend", "numpy",
                                    "--lengths", "9", "--chis", "1", "24", "--widths", "4",
                                    "--repeats", "1", "--output", str(output)])
    with pytest.warns(RuntimeWarning, match="SVD convergence failure"):
        scaling.main()
    rows = json.loads(output.read_text())["rows"]
    assert rows[0]["status"] == "failed" and rows[0]["chi"] == 1
    assert rows[1]["rank_reduced_bonds"] > 0
    assert rows[1]["max_window_frobenius_error"] < 2e-12
