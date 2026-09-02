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
        "test_contraction_dependencies.py",
        "test_import_boundaries.py",
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

# Explicit test tiers. A module can belong to both ``core`` and ``optional``
# when it exercises a stable API through an optional backend. The intersection
# is intentional: ``-m 'core and not optional'`` is the dependency-light core
# profile, while ``-m core`` is the complete core API profile.
_CORE_MODULES = frozenset(
    {
        *_SMOKE_MODULES,
        "test_energy_tree.py",
        "test_factorized_pair_mpo.py",
        "test_gate.py",
        "test_ham.py",
        "test_mpo.py",
        "test_mpo_cluster.py",
        "test_mpo_trotter.py",
        "test_mpo_automaton.py",
        "test_optimize_mpo.py",
        "test_optimize_mps.py",
        "test_optimize_peps.py",
        "test_peps_sampler.py",
        "test_prepare_boundary_inputs.py",
        "test_sampler.py",
        "test_simple_update_gen.py",
    }
)

_OPTIONAL_MODULES = frozenset(
    {
        "test_backends.py",
        "test_bp_compression.py",
        "test_bp_open_series.py",
        "test_bp_reduced_update.py",
        "test_bp_relay.py",
        "test_bp_symmray.py",
        "test_energy_tree.py",
        "test_factorized_pair_mpo.py",
        "test_fermion_gate_cache.py",
        "test_fermionic_boundary.py",
        "test_fh_jw_gates.py",
        "test_gradient_solver.py",
        "test_guppy_interop.py",
        "test_mpo_benchmarks.py",
        "test_native_fermion_pepo_2x3.py",
        "test_netket_flat_z2.py",
        "test_optimize_global.py",
        "test_optimize_qmera.py",
        "test_optimize_tree.py",
        "test_optimize_tree_stabilizer.py",
        "test_stabilizer_tn.py",
        "test_symmetric_tensors.py",
        "test_trajectory_noise.py",
        "test_tree_mpo.py",
        "test_tree_sampler.py",
        "test_vmc_api.py",
        "test_vmc_convergence.py",
        "test_vmc_distributed.py",
        "test_vmc_importance.py",
        "test_vmc_local_energy.py",
        "test_vmc_transition_plan.py",
    }
)

_DOMAIN_BY_MODULE = {
    "test_backends.py": "backends",
    "test_contraction_dependencies.py": "contractions",
    "test_energy_tree.py": "tree",
    "test_factorized_pair_mpo.py": "mpo",
    "test_fermion_gate_cache.py": "fermions",
    "test_fermionic_boundary.py": "boundary",
    "test_fh_jw_gates.py": "fermions",
    "test_gate.py": "operators",
    "test_gradient_solver.py": "solvers",
    "test_guppy_interop.py": "interop",
    "test_ham.py": "operators",
    "test_import_boundaries.py": "package",
    "test_mpo.py": "mpo",
    "test_mpo_cluster.py": "mpo",
    "test_mpo_trotter.py": "mpo",
    "test_mpo_automaton.py": "mpo",
    "test_mpo_benchmarks.py": "benchmarks",
    "test_native_fermion_pepo_2x3.py": "fermions",
    "test_native_identity_pepo.py": "tensors",
    "test_netket_flat_z2.py": "vmc",
    "test_optimize_global.py": "optimizers",
    "test_optimize_mpo.py": "mpo",
    "test_optimize_mps.py": "mps",
    "test_optimize_peps.py": "peps",
    "test_optimize_qmera.py": "qmera",
    "test_optimize_tree.py": "tree",
    "test_optimize_tree_stabilizer.py": "tree_stabilizer",
    "test_package_layout.py": "package",
    "test_peps_sampler.py": "sampling",
    "test_prepare_boundary_inputs.py": "boundary",
    "test_public_api.py": "package",
    "test_sampler.py": "sampling",
    "test_simple_update_gen.py": "peps",
    "test_simulator_planner.py": "optimizers",
    "test_stabilizer_tn.py": "stabilizer",
    "test_symmetric_tensors.py": "symmetry",
    "test_tensor_constructors.py": "tensors",
    "test_trajectory_noise.py": "noise",
    "test_tree_mpo.py": "tree",
    "test_tree_sampler.py": "tree",
    "test_vmc_api.py": "vmc",
    "test_vmc_convergence.py": "vmc",
    "test_vmc_distributed.py": "vmc",
    "test_vmc_importance.py": "vmc",
    "test_vmc_local_energy.py": "vmc",
    "test_vmc_transition_plan.py": "vmc",
}


def pytest_ignore_collect(collection_path, config):
    """Avoid importing extended modules during the default smoke run.

    Pytest normally imports every test module before applying ``-m smoke``.
    That defeats the lightweight default when an extended module imports an
    optional backend at module scope. Keep explicit file or directory runs
    unchanged, and let ``pytest -o addopts=''`` collect the complete suite.
    """
    if config.getoption("markexpr") != "smoke":
        return False
    path = Path(collection_path)
    return (
        path.is_file()
        and path.suffix == ".py"
        and path.name.startswith("test_")
        and path.name not in _SMOKE_MODULES
    )


def pytest_collection_modifyitems(config, items):
    """Tag tests with their tier and responsibility-based domain."""
    del config
    import pytest

    for item in items:
        module_name = Path(str(item.fspath)).name
        if module_name in _SMOKE_MODULES:
            item.add_marker(pytest.mark.smoke)
        if module_name in _INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        if module_name in _CORE_MODULES:
            item.add_marker(pytest.mark.core)
        if module_name in _OPTIONAL_MODULES or module_name.startswith("test_bp_"):
            item.add_marker(pytest.mark.optional)

        domain = _DOMAIN_BY_MODULE.get(module_name)
        if domain is None and module_name.startswith("test_bp_"):
            domain = "bp"
        if domain is not None:
            item.add_marker(getattr(pytest.mark, domain))
