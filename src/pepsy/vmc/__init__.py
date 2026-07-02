"""Variational Monte Carlo helpers.

The VMC namespace is optional-dependency friendly. Import concrete integrations
from leaf modules, for example ``pepsy.vmc.netket``.
"""

from importlib import import_module

_SYMBOL_MODULES = {
    "NetKetChunkSettings": ".netket",
    "NetKetFermiHubbardVMC": ".netket",
    "NetKetVMCSettings": ".netket",
    "PackedFermionicPEPS": ".netket",
    "SpinOrbitalColumns": ".netket",
    "build_fermi_hubbard_vmc": ".netket",
    "choose_netket_chunk_size": ".netket",
    "configure_jax_for_vmc": ".netket",
    "make_fermionic_peps_batched_amplitude_function": ".netket",
    "make_fermionic_peps_log_amplitude_model": ".netket",
    "make_netket_autochunk_callback": ".netket",
    "make_netket_sr_preconditioner": ".netket",
    "make_netket_vmc_driver": ".netket",
    "netket_spin_orbital_columns": ".netket",
    "occupation_to_phys_indices": ".netket",
    "pack_fermionic_peps_ansatz": ".netket",
    "recommend_netket_vmc_settings": ".netket",
    "square_lattice_edges": ".netket",
    "verify_netket_spin_columns": ".netket",
}

__all__ = tuple(_SYMBOL_MODULES)


def __getattr__(name):
    """Lazily import optional VMC integrations."""
    if name in _SYMBOL_MODULES:
        module = import_module(_SYMBOL_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'pepsy.vmc' has no attribute {name!r}")
