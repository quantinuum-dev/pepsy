"""Test configuration for local src-layout imports."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Keep the default developer/CI loop focused on a small set of deterministic
# package contracts. The complete suite remains available with
# ``pytest -o addopts=''`` when an extended validation pass is needed.
_SMOKE_MODULES = frozenset(
    {
        "test_backends.py",
        "test_package_layout.py",
        "test_public_api.py",
        "test_tensor_constructors.py",
    }
)

_INTEGRATION_MODULES = frozenset(
    {
        "test_bp_relay.py",
        "test_fh_jw_gates.py",
        "test_optimize_global.py",
        "test_optimize_qmera.py",
        "test_optimize_peps.py",
        "test_optimize_tree.py",
        "test_optimize_tree_stabilizer.py",
        "test_simple_update_gen.py",
        "test_stabilizer_tn.py",
        "test_symmetric_tensors.py",
        "test_tree_sampler.py",
        "test_vmc_api.py",
    }
)


def pytest_collection_modifyitems(config, items):
    """Tag extended suites so the default test command stays lightweight."""
    del config
    import pytest

    for item in items:
        module_name = Path(str(item.fspath)).name
        if module_name in _SMOKE_MODULES:
            item.add_marker(pytest.mark.smoke)
        if module_name in _INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
