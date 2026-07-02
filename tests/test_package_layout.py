"""Tests for the responsibility-based package namespaces."""

import importlib

import pytest

from pepsy.boundary import (
    BdyMPS,
    contract_boundary,
    contract_flat,
    peps_fidelity,
    peps_infidelity,
    peps_norm,
    peps_normalize,
)
from pepsy.operators import gate, rx
from pepsy.optimizers import MpsOptimizer, PepsEnergyOptimizer, PepsOptimizer, SimpleUpdateGen, SweepOptimizer
from pepsy.sampling import MpsSampler, PepsBpSampler
from pepsy.solvers import FDSolver
from pepsy.tensors import (
    OneDMap,
    SymGateStream,
    SymMPS,
    SymPEPS,
    backend_torch,
    default_physical_sectors,
    haar_random_state,
    ps_to_3dpeps,
    ps_to_peps,
    reg_complex_qr_torch,
    reg_complex_svd_jax,
    reg_complex_svd_torch,
    reg_real_qr_torch,
    reg_real_svd_jax,
    reg_real_svd_torch,
    reg_rel_svd_jax,
    reg_rel_svd_torch,
    site_charge_from_occupations,
)
from pepsy.vmc import (
    FermionSiteEncoding,
    TorchPEPSAmplitude,
    TorchSquareLattice,
    apply_torch_sr_update,
    build_heisenberg_vmc,
    build_ising_vmc,
    heisenberg_connections,
    make_fermionic_peps_batched_amplitude_function,
    make_peps_batched_amplitude_function,
    make_torch_peps_amplitude_model,
    pack_peps_ansatz,
    solve_torch_sr,
    spinful_fermi_hubbard_connections,
    square_lattice_edges,
    torch_log_derivative_matrix,
)


def test_new_namespace_imports_resolve():
    """Common new namespace imports should resolve to usable objects."""
    assert BdyMPS is not None
    assert callable(contract_boundary)
    assert callable(contract_flat)
    assert callable(peps_norm)
    assert callable(peps_normalize)
    assert callable(peps_infidelity)
    assert callable(peps_fidelity)
    assert callable(gate)
    assert callable(rx)
    assert MpsOptimizer is not None
    assert PepsEnergyOptimizer is not None
    assert PepsOptimizer is not None
    assert SimpleUpdateGen is not None
    assert SweepOptimizer is not None
    assert MpsSampler is not None
    assert PepsBpSampler is not None
    assert FDSolver is not None
    assert OneDMap is not None
    assert SymGateStream is not None
    assert SymMPS is not None
    assert SymPEPS is not None
    assert callable(default_physical_sectors)
    assert callable(backend_torch)
    assert callable(haar_random_state)
    assert callable(ps_to_peps)
    assert callable(ps_to_3dpeps)
    assert callable(reg_rel_svd_torch)
    assert callable(reg_real_svd_torch)
    assert callable(reg_complex_svd_torch)
    assert callable(reg_real_qr_torch)
    assert callable(reg_complex_qr_torch)
    assert callable(reg_rel_svd_jax)
    assert callable(reg_real_svd_jax)
    assert callable(reg_complex_svd_jax)
    assert callable(site_charge_from_occupations)
    assert FermionSiteEncoding is not None
    assert TorchPEPSAmplitude is not None
    assert TorchSquareLattice is not None
    assert callable(apply_torch_sr_update)
    assert callable(build_heisenberg_vmc)
    assert callable(build_ising_vmc)
    assert callable(heisenberg_connections)
    assert callable(make_fermionic_peps_batched_amplitude_function)
    assert callable(make_peps_batched_amplitude_function)
    assert callable(make_torch_peps_amplitude_model)
    assert callable(pack_peps_ansatz)
    assert callable(solve_torch_sr)
    assert callable(spinful_fermi_hubbard_connections)
    assert callable(square_lattice_edges)
    assert callable(torch_log_derivative_matrix)


@pytest.mark.parametrize(
    "old_module",
    [
        "pepsy.boundary_metrics",
        "pepsy.boundary_states",
        "pepsy.boundary_sweeps",
        "pepsy.core",
        "pepsy.fit",
        "pepsy.ft_solver",
        "pepsy.gates",
        "pepsy.gradient_solver",
        "pepsy.ham",
        "pepsy.optimize_energy",
        "pepsy.optimize_global",
        "pepsy.optimize_mpo",
        "pepsy.optimize_mps",
        "pepsy.optimize_sweep",
        "pepsy.sampler",
    ],
)
def test_old_layout_modules_are_removed(old_module):
    """Old flat module paths should no longer be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_module)
