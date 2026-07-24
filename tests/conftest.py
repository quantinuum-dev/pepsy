"""Test configuration for local src-layout imports."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Keep the default developer/CI loop focused on the package's core contracts.
# These suites remain available with ``pytest -o addopts=''`` when an extended
# validation pass is needed.
_INTEGRATION_MODULES = frozenset(
    {
        "test_bp_cluster.py",
        "test_bp_pne.py",
        "test_bp_reduced_update.py",
        "test_bp_relay.py",
        "test_bp_series.py",
        "test_fh_jw_gates.py",
        "test_optimize_energy_mps.py",
        "test_optimize_energy_peps.py",
        "test_optimize_global.py",
        "test_optimize_mera.py",
        "test_optimize_peps.py",
        "test_optimize_sweep_plot.py",
        "test_optimize_tree.py",
        "test_optimize_tree_stabilizer.py",
        "test_simple_update_gen.py",
        "test_stabilizer_tn.py",
        "test_stim_noise.py",
        "test_sweep_symmray_backend.py",
        "test_sym_dmrg.py",
        "test_symmetric_tensors.py",
        "test_tree_sampler.py",
        "test_tree_symmetric.py",
        "test_vmc_api.py",
        "test_vmc_netket.py",
        "test_vmc_torch.py",
    }
)
_SLOW_MODULES = frozenset({"test_stabilizer_tn_stress.py"})


def pytest_collection_modifyitems(config, items):
    """Tag extended suites so the default test command stays lightweight."""
    del config
    import pytest

    for item in items:
        module_name = Path(str(item.fspath)).name
        if module_name in _INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        if module_name in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
