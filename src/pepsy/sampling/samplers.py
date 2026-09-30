"""Compatibility imports for samplers now owned by their sampling family.

Use ``pepsy.sampling`` for public imports. Legacy classes and serialized
references resolve to the same objects. Public namespace imports select the
owning engine directly.
"""

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np  # noqa: F401 -- historical array-module attribute

from .mps import MpsSampler
from .vector import VecSampler
from .peps import PepsSampler
from .bp import PepsBpSampler
from .results import (
    FermionConfigurationEncoding,
    MpsBatchSampleResult,
    MpsDiagonalEstimate,
    MpsSampleResult,
    PEPSSampleResult,
)

__all__ = [
    'FermionConfigurationEncoding',
    'MpsDiagonalEstimate',
    'MpsBatchSampleResult',
    'MpsSampleResult',
    'MpsSampler',
    'PEPSSampleResult',
    'PepsSampler',
    'PepsBpSampler',
    'VecSampler',
]

_SYMBOL_MODULES = {
    '_normalize_dense_memory_limit': '.mps',
    '_normalize_symmray_prefix_strategy': '.mps',
    'MpsSampler': '.mps',
    '_normalize_mps_sampler_backend': '.mps',
    '_fermion_symmray_occupations': '.mps',
    '_validate_sample_chunk_size': '.vector',
    '_validate_sample_count': '.vector',
    '_measurement_rotation': '.vector',
    '_resolve_measurement_basis': '.vector',
    '_TORCH_MULTINOMIAL_MAX_CATEGORIES': '.vector',
    'VecSampler': '.vector',
    '_DEFAULT_VEC_SAMPLE_CHUNK_SIZE': '.vector',
    '_basis_selection_probability': '.vector',
    '_MEASUREMENT_BASIS_LABELS': '.vector',
    'PepsSampler': '.peps',
    '_prepare_bp_binary_network': '.bp',
    'sample_d2bp': '.bp',
    '_to_dense_numpy': '.bp',
    'PepsBpSampler': '.bp',
    'build_optimizer': '.bp',
    'MpsSampleResult': '.results',
    'MpsDiagonalEstimate': '.results',
    'PEPSSampleResult': '.results',
    '_configs_to_sample_result': '.results',
    'FermionConfigurationEncoding': '.results',
    'MpsBatchSampleResult': '.results',
    '_validate_one_d_to_two_d': '._common',
    '_fermion_code_order': '._common',
    '_backend_array_to_numpy': '._common',
    '_mps_array_backend': '._common',
    '_infer_fermion_code_order': '._common',
}

if TYPE_CHECKING:
    from .mps import (  # noqa: F401 -- historical import aliases
        _fermion_symmray_occupations,
        _normalize_dense_memory_limit,
        _normalize_mps_sampler_backend,
        _normalize_symmray_prefix_strategy,
    )
    from .vector import (  # noqa: F401 -- historical import aliases
        _DEFAULT_VEC_SAMPLE_CHUNK_SIZE,
        _MEASUREMENT_BASIS_LABELS,
        _TORCH_MULTINOMIAL_MAX_CATEGORIES,
        _basis_selection_probability,
        _measurement_rotation,
        _resolve_measurement_basis,
        _validate_sample_chunk_size,
        _validate_sample_count,
    )
    from .bp import (  # noqa: F401 -- historical import aliases
        _prepare_bp_binary_network,
        _to_dense_numpy,
        build_optimizer,
        sample_d2bp,
    )
    from .results import (  # noqa: F401 -- historical import aliases
        _configs_to_sample_result,
    )
    from ._common import (  # noqa: F401 -- historical import aliases
        _backend_array_to_numpy,
        _fermion_code_order,
        _infer_fermion_code_order,
        _mps_array_backend,
        _validate_one_d_to_two_d,
    )


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __package__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_SYMBOL_MODULES))
