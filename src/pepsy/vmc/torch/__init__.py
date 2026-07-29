"""PyTorch VMC kernels organized by responsibility.

The package-level imports are the compatibility surface for both
``pepsy.vmc.torch.<name>`` and the lazy re-exports from :mod:`pepsy.vmc`.
Small cross-leaf helpers live in :mod:`pepsy.vmc.torch._common`; compatibility
aliases and dispatch that still span workflows remain in
:mod:`pepsy.vmc.torch._core`.
"""

from ..torch_types import FermionSiteEncoding, SpinlessSiteEncoding, TorchSquareLattice
from ._common import _count_spinful_particles as count_spinful_particles
from ._common import _require_torch as _require_torch
from .amplitude import (
    TorchPEPSAmplitude,
    TorchPEPSBoundaryAmplitude,
    make_torch_peps_amplitude_model,
)
from .benchmark import (
    TorchAmplitudeBenchmark,
    TorchAmplitudeBenchmarkRun,
    benchmark_torch_amplitudes,
)
from .connections import TorchConnections, compile_operator_sum_torch, torch_hamiltonian_connections
from .driver import TorchVMCDriver
from .fermion import (
    TorchFermionVMC,
    TorchVMCSetup,
    build_torch_vmc,
    random_spin_configs,
    random_spinful_configs,
)
from .local_energy import (
    heisenberg_connections,
    local_energy_from_connections,
    spinful_fermi_hubbard_connections,
    torch_chain_diagnostics,
    transverse_ising_connections,
)
from .metadata import TorchFermionVMCMetadata
from .proposals import (
    metropolis_exchange_sweep,
    propose_spin_exchange,
    propose_spinful_exchange_or_hopping,
    propose_spinful_u1_exchange_or_hopping,
    propose_spinful_z2_exchange_or_hopping,
    propose_spinful_z2z2_exchange_or_hopping,
)
from .results import (
    TorchMetropolisResult,
    TorchImportanceSamples,
    TorchMCMCSamples,
    TorchDistributedMetadata,
    TorchSampleProvenance,
    TorchChainDiagnostics,
    TorchVMCConvergenceEstimate,
    TorchVMCConvergenceReport,
    TorchVMCEnergyEstimate,
    TorchVMCImportanceEstimate,
    TorchVMCMeasurementRun,
    TorchVMCStepResult,
    TorchVMCWarmupResult,
)
from .sampler import TorchBPMetropolisSampler, TorchMetropolisSampler, metropolis_local_sampler
from .sr import TorchSRResult, apply_torch_sr_update, solve_torch_sr, torch_log_derivative_matrix

__all__ = [
    "FermionSiteEncoding",
    "SpinlessSiteEncoding",
    "TorchFermionVMCMetadata",
    "TorchPEPSAmplitude",
    "TorchPEPSBoundaryAmplitude",
    "TorchAmplitudeBenchmark",
    "TorchAmplitudeBenchmarkRun",
    "TorchConnections",
    "TorchMetropolisResult",
    "TorchImportanceSamples",
    "TorchMCMCSamples",
    "TorchDistributedMetadata",
    "TorchSampleProvenance",
    "TorchChainDiagnostics",
    "TorchVMCConvergenceEstimate",
    "TorchVMCConvergenceReport",
    "TorchMetropolisSampler",
    "TorchBPMetropolisSampler",
    "TorchVMCDriver",
    "TorchFermionVMC",
    "TorchVMCSetup",
    "TorchVMCEnergyEstimate",
    "TorchVMCImportanceEstimate",
    "TorchVMCMeasurementRun",
    "TorchVMCStepResult",
    "TorchVMCWarmupResult",
    "TorchSRResult",
    "TorchSquareLattice",
    "apply_torch_sr_update",
    "benchmark_torch_amplitudes",
    "count_spinful_particles",
    "heisenberg_connections",
    "local_energy_from_connections",
    "torch_chain_diagnostics",
    "metropolis_local_sampler",
    "metropolis_exchange_sweep",
    "propose_spin_exchange",
    "propose_spinful_exchange_or_hopping",
    "propose_spinful_u1_exchange_or_hopping",
    "propose_spinful_z2_exchange_or_hopping",
    "propose_spinful_z2z2_exchange_or_hopping",
    "random_spin_configs",
    "random_spinful_configs",
    "make_torch_peps_amplitude_model",
    "compile_operator_sum_torch",
    "build_torch_vmc",
    "solve_torch_sr",
    "spinful_fermi_hubbard_connections",
    "torch_log_derivative_matrix",
    "transverse_ising_connections",
    "torch_hamiltonian_connections",
]
