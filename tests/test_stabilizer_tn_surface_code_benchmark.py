"""Small end-to-end equivalence checks for the STN surface-code benchmark."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("stim")

_BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "stabilizer_tn_surface_code.py"
_SPEC = importlib.util.spec_from_file_location("pepsy_surface_code_benchmark", _BENCHMARK_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
run_benchmark = _MODULE.run_benchmark


def test_surface_code_style_benchmark_matches_mps_and_tree_at_exact_chi():
    report = run_benchmark(
        distance=2,
        rounds=2,
        depolarize2=0.0005,
        shots=2,
        chi_values=(1, None),
        seed=41,
        crosstalk_theta=0.01,
    )

    assert report["circuit"]["num_qubits"] == 8
    assert report["circuit"]["detectors"] == 8
    assert report["exact_equivalence"] == {
        "measurements_equal": True,
        "syndromes_equal": True,
        "observables_equal": True,
        "minimum_state_fidelity": pytest.approx(1.0),
        "dense_statevector_comparison": True,
    }
    assert len(report["chi_convergence"]) == 4
    assert {row["chi"] for row in report["chi_convergence"]} == {1, None}
    assert len(report["coherent_crosstalk"]) == 2
    assert all(row["theta"] == pytest.approx(0.01) for row in report["coherent_crosstalk"])
    assert report["coherent_crosstalk_equivalence"]["measurements_equal"] is True
    assert report["coherent_crosstalk_equivalence"]["syndromes_equal"] is True
    assert report["coherent_crosstalk_equivalence"]["minimum_state_fidelity"] == pytest.approx(1.0)


def test_surface_code_style_benchmark_scales_without_dense_statevectors():
    report = run_benchmark(
        distance=3,
        rounds=2,
        depolarize1=0.0001,
        depolarize2=0.0005,
        shots=1,
        chi_values=(None,),
        seed=43,
    )

    assert report["circuit"]["num_qubits"] == 13
    assert report["circuit"]["distance"] == 3
    assert report["exact_equivalence"]["dense_statevector_comparison"] is False
    assert report["exact_equivalence"]["measurements_equal"] is True
    assert report["exact_equivalence"]["syndromes_equal"] is True
    assert report["exact_equivalence"]["minimum_state_fidelity"] is None
