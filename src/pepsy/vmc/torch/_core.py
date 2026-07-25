"""Shared PyTorch VMC kernels and compatibility imports.

Responsibility-specific implementations live in the sibling modules. This
module retains small shared tensor helpers and the historical private import
surface.
"""

from __future__ import annotations

from ..torch_types import (
    FermionSiteEncoding,
    SpinlessSiteEncoding,
    TorchSquareLattice,
)
from ._common import (
    _COMPILED_CHEAP_TORCH_KERNELS,  # noqa: F401
    _FAILED_CHEAP_TORCH_KERNELS,  # noqa: F401
    _as_contraction_options,  # noqa: F401
    _as_long_matrix,  # noqa: F401
    _count_spinful_particles as count_spinful_particles,  # noqa: F401
    _edge_value,  # noqa: F401
    _iter_edges,  # noqa: F401
    _make_torch_generator,  # noqa: F401
    _proposal_log_probabilities,  # noqa: F401
    _require_torch,  # noqa: F401
    _run_cheap_torch_kernel,  # noqa: F401
    _site_value,  # noqa: F401
    _torch_finfo_tiny,  # noqa: F401
    _validate_contraction,  # noqa: F401
)
from .results import (
    TorchChainDiagnostics,
    TorchMCMCSamples,
    TorchMetropolisResult,
    TorchVMCImportanceEstimate,
    TorchVMCEnergyEstimate,
    TorchVMCStepResult,
)
from .metadata import (
    TorchFermionVMCMetadata,
)
from .connections import (
    TorchConnections,
    compile_operator_sum_torch,
    torch_hamiltonian_connections,
)
from .local_energy import (
    heisenberg_connections,
    local_energy_from_connections,
    spinful_fermi_hubbard_connections,
    torch_chain_diagnostics,
    transverse_ising_connections,
)
from .amplitude import (
    TorchPEPSAmplitude,
    TorchPEPSBoundaryAmplitude,
    make_torch_peps_amplitude_model,
)
from ._graded import (
    _GradedTorchPair,  # noqa: F401
    _GradedTorchProjector,  # noqa: F401
    _find_symmray_tensors,  # noqa: F401
    _graded_torch_compile_pair,  # noqa: F401
    _graded_torch_contraction_mask,  # noqa: F401
    _graded_torch_dense,  # noqa: F401
    _graded_torch_embed_dense,  # noqa: F401
    _graded_torch_index_map,  # noqa: F401
    _graded_torch_pad,  # noqa: F401
    _graded_torch_prepare_pair,  # noqa: F401
    _graded_torch_sign_mask,  # noqa: F401
    _graded_torch_unit_probe,  # noqa: F401
    _is_symmray_data,  # noqa: F401
)
from .sr import (
    TorchSRResult,  # noqa: F401
    _batched_model_log_derivatives,  # noqa: F401
    _flatten_torch_tensors,  # noqa: F401
    _log_derivative_denominator,  # noqa: F401
    _promote_sr_tensors,  # noqa: F401
    _resolve_sr_diag_shift,  # noqa: F401
    _spring_complement,  # noqa: F401
    _torch_log_derivative_matrix_loop,  # noqa: F401
    _torch_model_parameters,  # noqa: F401
    _torch_solve_linear,  # noqa: F401
    apply_torch_sr_update,  # noqa: F401
    solve_torch_sr,  # noqa: F401
    torch_log_derivative_matrix,  # noqa: F401
)
from .proposals import (
    metropolis_exchange_sweep,
    propose_spin_exchange,
    propose_spinful_exchange_or_hopping,
    propose_spinful_u1_exchange_or_hopping,
    propose_spinful_z2_exchange_or_hopping,
    propose_spinful_z2z2_exchange_or_hopping,
)
from .sampler import (
    TorchBPMetropolisSampler,
    TorchMetropolisSampler,
    metropolis_local_sampler,
)
from .driver import TorchVMCDriver
from .fermion import (
    TorchFermionVMC,
    TorchVMCSetup,
    build_torch_vmc,
    random_spin_configs,
    random_spinful_configs,
)

__all__ = [
    "FermionSiteEncoding",
    "SpinlessSiteEncoding",
    "TorchFermionVMCMetadata",
    "TorchPEPSAmplitude",
    "TorchPEPSBoundaryAmplitude",
    "TorchConnections",
    "TorchMetropolisResult",
    "TorchMCMCSamples",
    "TorchChainDiagnostics",
    "TorchMetropolisSampler",
    "TorchBPMetropolisSampler",
    "TorchVMCDriver",
    "TorchFermionVMC",
    "TorchVMCSetup",
    "TorchVMCEnergyEstimate",
    "TorchVMCImportanceEstimate",
    "TorchVMCStepResult",
    "TorchSRResult",
    "TorchSquareLattice",
    "apply_torch_sr_update",
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


_PROPOSAL_BATCHING_MODES = {"auto", "cache", "vmap"}

# ``amplitude_batching`` controls the independent-configuration path.  This
# is intentionally separate from ``proposal_batching``: local Metropolis
# proposals can use boundary-environment reuse even when a PEPS's ordinary
# amplitude path must remain serial (for example, native U1/U1U1 Symmray).
_AMPLITUDE_BATCHING_MODES = {"auto", "serial", "vmap"}

# Boundary-environment reuse is useful for small connected sets, while the
# vmapped full-boundary path is substantially faster once many off-diagonal
# configurations are measured together.
_BOUNDARY_VMAP_CONNECTION_THRESHOLD = 64

def _resolve_connection_fn(connection_fn):
    if callable(connection_fn):
        return None, connection_fn
    key = str(connection_fn).replace("-", "_").lower()
    aliases = {
        "fermi_hubbard": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "fh": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "hubbard": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "spinful": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "spinful_fermi_hubbard": (
            "spinful_fermi_hubbard",
            spinful_fermi_hubbard_connections,
        ),
        "heisenberg": ("heisenberg", heisenberg_connections),
        "heis": ("heisenberg", heisenberg_connections),
        "transverse_ising": ("transverse_ising", transverse_ising_connections),
        "tfim": ("transverse_ising", transverse_ising_connections),
        "ising": ("transverse_ising", transverse_ising_connections),
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown torch VMC connection_fn {connection_fn!r}. "
            f"Expected a callable or one of: {allowed}."
        ) from exc
