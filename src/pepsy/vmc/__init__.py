"""Variational Monte Carlo helpers.

The VMC namespace is optional-dependency friendly. Import concrete integrations
from leaf modules, for example ``pepsy.vmc.netket``.
"""

from importlib import import_module

_SYMBOL_MODULES = {
    "FermionSiteEncoding": ".torch",
    "NetKetLocalConfigMap": ".netket",
    "NetKetChunkSettings": ".netket",
    "NetKetPEPSVMC": ".netket",
    "NetKetFermiHubbardVMC": ".netket",
    "NetKetVMCSettings": ".netket",
    "PackedPEPS": ".netket",
    "PackedFermionicPEPS": ".netket",
    "SpinOrbitalColumns": ".netket",
    "TorchConnections": ".torch",
    "TorchMetropolisResult": ".torch",
    "TorchPEPSAmplitude": ".torch",
    "TorchSRResult": ".torch",
    "TorchSquareLattice": ".torch",
    "apply_torch_sr_update": ".torch",
    "build_heisenberg_vmc": ".netket",
    "build_ising_vmc": ".netket",
    "build_fermi_hubbard_vmc": ".netket",
    "fermionic_peps_rand": ".netket",
    "choose_netket_chunk_size": ".netket",
    "configure_jax_for_vmc": ".netket",
    "config_to_phys_indices": ".netket",
    "count_spinful_particles": ".torch",
    "heisenberg_connections": ".torch",
    "local_energy_from_connections": ".torch",
    "make_peps_batched_amplitude_function": ".netket",
    "make_peps_log_amplitude_model": ".netket",
    "make_fermionic_peps_batched_amplitude_function": ".netket",
    "make_fermionic_peps_log_amplitude_model": ".netket",
    "make_netket_autochunk_callback": ".netket",
    "make_netket_sr_preconditioner": ".netket",
    "make_netket_vmc_driver": ".netket",
    "make_torch_peps_amplitude_model": ".torch",
    "netket_spin_orbital_columns": ".netket",
    "occupation_to_phys_indices": ".netket",
    "pack_peps_ansatz": ".netket",
    "pack_fermionic_peps_ansatz": ".netket",
    "propose_spin_exchange": ".torch",
    "propose_spinful_exchange_or_hopping": ".torch",
    "random_spin_configs": ".torch",
    "random_spinful_configs": ".torch",
    "recommend_netket_vmc_settings": ".netket",
    "solve_torch_sr": ".torch",
    "square_lattice_edges": ".netket",
    "metropolis_exchange_sweep": ".torch",
    "spinful_fermi_hubbard_connections": ".torch",
    "torch_log_derivative_matrix": ".torch",
    "transverse_ising_connections": ".torch",
    "verify_netket_spin_columns": ".netket",
}

__all__ = tuple(_SYMBOL_MODULES)


def __getattr__(name):
    """Lazily import optional VMC integrations."""
    if name in _SYMBOL_MODULES:
        module = import_module(_SYMBOL_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'pepsy.vmc' has no attribute {name!r}")
