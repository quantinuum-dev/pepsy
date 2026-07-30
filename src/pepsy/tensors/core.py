"""Compatibility aggregator for tensor-network implementations.

New code should import from the modules pepsy.tensors.maps,
pepsy.tensors.constructors, pepsy.tensors.contractions, or
pepsy.tensors.observables. This module remains as a stable compatibility
surface for existing pepsy.tensors.core imports.
"""

from .maps import OneDMap
from . import contractions as _contractions
from . import observables as _observables
from .contractions import (
    build_compressed_optimizer as _build_compressed_optimizer,
    build_contraction as _build_contraction,
    build_optimizer as _build_optimizer,
    contract_hypercompressed_tn as _contract_hypercompressed_tn,
    contract_hypercompressed_tn_batch,
    tn_norm,
)
from .observables import measure_obs, tn_fidelity as _tn_fidelity
from .constructors import (
    add_cycle,
    expec_mpo,
    haar_random_state,
    hrs_to_mps,
    hrs_to_peps,
    hrs_to_ttn,
    hrps_to_mps,
    hrps_to_peps,
    hrps_to_ttn,
    id_to_mpo,
    id_to_pepo,
    ps_to_3dpeps,
    ps_to_mpo,
    ps_to_mps,
    ps_to_pepo,
    ps_to_peps,
    ps_to_ttn,
    random_haar_qubit,
    tns_align,
)
from ..backends.config import (
    backend_cupy,
    backend_jax,
    backend_numpy,
    backend_torch,
    build_backend,
    get_default_array_backend,
    get_default_grad_backend,
    register_jax_linalg,
    reg_native_svd_jax,
    reg_native_svd_torch,
    reg_complex_qr_torch,
    reg_complex_svd_jax,
    reg_complex_svd_torch,
    reg_real_qr_torch,
    reg_real_svd_jax,
    reg_real_svd_torch,
    reg_rel_svd_jax,
    reg_rel_svd_torch,
    reg_stop_gradient_torch,
    reset_linalg_registrations,
    register_torch_linalg,
    reset_default_backends,
    set_default_array_backend,
    set_default_grad_backend,
    stop_grad,
)

# A few long-standing tests and downstream integrations patch these names on
# ``tensors.core``. Keep the compatibility hooks while the implementations
# live in focused modules.
ctg = _contractions.ctg
_ensure_cotengrust = _contractions._ensure_cotengrust


def build_optimizer(*args, **kwargs):
    original_ensure = _contractions._ensure_cotengrust
    _contractions._ensure_cotengrust = _ensure_cotengrust
    try:
        return _build_optimizer(*args, **kwargs)
    finally:
        _contractions._ensure_cotengrust = original_ensure


def build_contraction(*args, **kwargs):
    """Compatibility wrapper for :func:`pepsy.tensors.build_contraction`."""

    original_ensure = _contractions._ensure_cotengrust
    _contractions._ensure_cotengrust = _ensure_cotengrust
    try:
        return _build_contraction(*args, **kwargs)
    finally:
        _contractions._ensure_cotengrust = original_ensure


def build_compressed_optimizer(*args, **kwargs):
    original_ensure = _contractions._ensure_cotengrust
    _contractions._ensure_cotengrust = _ensure_cotengrust
    try:
        return _build_compressed_optimizer(*args, **kwargs)
    finally:
        _contractions._ensure_cotengrust = original_ensure


def tn_fidelity(*args, **kwargs):
    original_build_optimizer = _observables.build_optimizer
    _observables.build_optimizer = build_optimizer
    try:
        return _tn_fidelity(*args, **kwargs)
    finally:
        _observables.build_optimizer = original_build_optimizer


def contract_hypercompressed_tn(*args, **kwargs):
    original_build_compressed_optimizer = _contractions.build_compressed_optimizer
    _contractions.build_compressed_optimizer = build_compressed_optimizer
    try:
        return _contract_hypercompressed_tn(*args, **kwargs)
    finally:
        _contractions.build_compressed_optimizer = original_build_compressed_optimizer

__all__ = [
    "OneDMap", "build_backend", "backend_torch", "backend_numpy", "backend_cupy", "backend_jax",
    "register_torch_linalg", "register_jax_linalg", "reg_native_svd_torch",
    "reg_native_svd_jax", "reg_rel_svd_torch", "reg_real_svd_torch",
    "reg_complex_svd_torch", "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
    "reset_linalg_registrations",
    "reg_stop_gradient_torch", "stop_grad", "set_default_array_backend",
    "get_default_array_backend", "set_default_grad_backend", "get_default_grad_backend",
    "reset_default_backends", "build_contraction", "build_optimizer", "build_compressed_optimizer",
    "contract_hypercompressed_tn", "contract_hypercompressed_tn_batch", "tn_fidelity",
    "tn_norm", "measure_obs", "tns_align", "expec_mpo", "id_to_mpo", "id_to_pepo",
    "ps_to_peps", "ps_to_3dpeps", "ps_to_mps", "ps_to_ttn", "ps_to_pepo", "ps_to_mpo",
    "haar_random_state", "random_haar_qubit", "hrs_to_peps", "hrs_to_mps", "hrs_to_ttn",
    "hrps_to_peps", "hrps_to_mps", "hrps_to_ttn", "add_cycle",
]
