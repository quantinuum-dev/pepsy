"""Lazy compatibility proxy for the former qMERA package name.

Use :mod:`pepsy.optimizers.qmera` for new code. The old namespace remains
available during the compatibility window without importing the qMERA
implementation until a symbol or child module is actually requested.
"""

from importlib import import_module
from types import ModuleType
import sys
import warnings


_SUBMODULES = (
    "builders",
    "cache",
    "compiled",
    "fermions",
    "gates",
    "geometry",
    "layout",
    "lightcones",
    "parametric",
    "prototype",
    "schedules",
    "schematics",
    "terms",
)

# Keep the compatibility namespace's star-import contract without importing
# qMERA merely to read its ``__all__``.
__all__ = (
    "GateRegistry",
    "GateSpec",
    "LocalTerm",
    "QMeraAnsatz",
    "QMeraBlockSpec",
    "QMeraBuilder",
    "QMeraCompiledLightconeChunk",
    "QMeraContractionPathCache",
    "QMeraDisentanglerSpec",
    "QMeraEnergyOptimizer",
    "QMeraGatePlacement",
    "QMeraGeometry",
    "QMeraIsometrySpec",
    "QMeraLayerSpec",
    "QMeraLayoutCandidate",
    "QMeraLayoutFinder",
    "QMeraLayoutReport",
    "QMeraLayoutScore",
    "QMeraLightconeTN",
    "QMeraLightconeGroup",
    "QMeraParametricLightconeChunk",
    "QMeraParametricEnergyOptimizer",
    "QMeraPrototypeLayout",
    "QMeraSchedule",
    "QMeraScaleSpec",
    "QMeraSchematicBlock",
    "QMeraSymmrayFermionBackend",
    "QMeraUnitarySpec",
    "UserGateFamily",
    "build_qmera_contraction_optimizer",
    "build_qmera_lightcone_chunks",
    "build_qmera_parametric_lightcone_chunks",
    "build_qmera_schedule",
    "compile_qmera_parametric_lightcone",
    "compile_qmera_parametric_lightcones",
    "contract_qmera_lightcone_tn",
    "contract_qmera_lightcone_group",
    "default_gate_registry",
    "draw_qmera_schedule",
    "group_qmera_parametric_lightcone_chunks",
    "local_qmera_compiled_lightcone_expectation",
    "local_qmera_parametric_lightcone_expectation",
    "load_qmera_prototype_layout",
    "normalize_local_terms",
    "qmera_compiled_parametric_energy",
    "qmera_direct_parametric_energy",
    "qmera_parametric_energy",
    "qmera_parametric_lightcone_group_state",
    "qmera_parametric_lightcone_state",
    "qmera_parametric_state",
    "qmera_parametric_lightcone_tn",
    "qmera_schematic_blocks",
    "qmera_symmray_fermi_hubbard_terms",
    "qmera_symmray_majorana_terms",
    "resolve_gate_spec",
    "select_lightcone",
    "site_tags_for_where",
    "symmray_fermion_gate_registry",
    "symmray_majorana_gate_registry",
)

_WARNING = "pepsy.optimizers.mera is a compatibility alias; use pepsy.optimizers.qmera instead."

warnings.warn(_WARNING, DeprecationWarning, stacklevel=2)


def _load_submodule(name):
    """Load and cache the canonical qMERA child module for ``name``."""
    target_name = f"{__name__}.{name}"
    target = import_module(f"..qmera.{name}", __name__)
    sys.modules[target_name] = target
    globals()[name] = target
    return target


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module("..qmera", __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return _load_submodule(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))


def _make_submodule_proxy(name):
    """Create a module proxy so direct legacy child imports stay lazy."""
    fullname = f"{__name__}.{name}"
    proxy = ModuleType(fullname, f"Lazy compatibility proxy for {fullname}.")
    proxy.__package__ = __name__

    def proxy_getattr(attribute, *, _name=name):
        target = _load_submodule(_name)
        return getattr(target, attribute)

    proxy.__getattr__ = proxy_getattr
    return proxy


for _module_name in _SUBMODULES:
    _legacy_module = sys.modules.setdefault(
        f"{__name__}.{_module_name}",
        _make_submodule_proxy(_module_name),
    )
    globals()[_module_name] = _legacy_module

del _legacy_module, _module_name
