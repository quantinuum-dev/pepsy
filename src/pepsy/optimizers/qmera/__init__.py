"""Canonical qMERA energy optimization helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

_SYMBOL_MODULES = {
    "QMeraAnsatz": ".builders",
    "QMeraBuilder": ".builders",
    "QMeraContractionPathCache": ".cache",
    "build_qmera_contraction_optimizer": ".cache",
    "QMeraCompiledLightconeChunk": ".compiled",
    "compile_qmera_parametric_lightcone": ".compiled",
    "compile_qmera_parametric_lightcones": ".compiled",
    "local_qmera_compiled_lightcone_expectation": ".compiled",
    "qmera_compiled_parametric_energy": ".compiled",
    "QMeraSymmrayFermionBackend": ".fermions",
    "qmera_symmray_fermi_hubbard_terms": ".fermions",
    "qmera_symmray_majorana_terms": ".fermions",
    "symmray_fermion_gate_registry": ".fermions",
    "symmray_majorana_gate_registry": ".fermions",
    "GateRegistry": ".gates",
    "GateSpec": ".gates",
    "QMeraPairSpec": ".gates",
    "available_qmera_pair_ansatzes": ".gates",
    "available_qmera_pair_ansatze": ".gates",
    "get_qmera_pair_ansatz": ".gates",
    "qmera_pair_gate_spec": ".gates",
    "UserGateFamily": ".gates",
    "default_gate_registry": ".gates",
    "resolve_gate_spec": ".gates",
    "QMeraGeometry": ".geometry",
    "QMeraLayoutCandidate": ".layout",
    "QMeraLayoutFinder": ".layout",
    "QMeraLayoutReport": ".layout",
    "QMeraLayoutScore": ".layout",
    "QMeraLightconeTN": ".lightcones",
    "QMeraLightconeGroup": ".lightcones",
    "QMeraParametricLightconeChunk": ".lightcones",
    "build_qmera_lightcone_chunks": ".lightcones",
    "build_qmera_parametric_lightcone_chunks": ".lightcones",
    "contract_qmera_lightcone_group": ".lightcones",
    "contract_qmera_lightcone_tn": ".lightcones",
    "group_qmera_parametric_lightcone_chunks": ".lightcones",
    "local_qmera_parametric_lightcone_expectation": ".lightcones",
    "qmera_parametric_energy": ".lightcones",
    "qmera_direct_parametric_energy": ".lightcones",
    "qmera_parametric_lightcone_group_state": ".lightcones",
    "qmera_parametric_state": ".lightcones",
    "qmera_parametric_lightcone_state": ".lightcones",
    "qmera_parametric_lightcone_tn": ".lightcones",
    "select_lightcone": ".lightcones",
    "site_tags_for_where": ".lightcones",
    "QMeraEnergyOptimizer": ".parametric",
    "QMeraParametricEnergyOptimizer": ".parametric",
    "QMeraPrototypeLayout": ".prototype",
    "load_qmera_prototype_layout": ".prototype",
    "QMeraBlockSpec": ".schedules",
    "QMeraDisentanglerSpec": ".schedules",
    "QMeraGatePlacement": ".schedules",
    "QMeraIsometrySpec": ".schedules",
    "QMeraLayerSpec": ".schedules",
    "QMeraScaleSpec": ".schedules",
    "QMeraSchedule": ".schedules",
    "QMeraUnitarySpec": ".schedules",
    "build_qmera_schedule": ".schedules",
    "QMeraSchematicBlock": ".schematics",
    "draw_qmera_schedule": ".schematics",
    "qmera_schematic_blocks": ".schematics",
    "LocalTerm": ".terms",
    "normalize_local_terms": ".terms",
}

# Preserve explicit and historically accessible child-module paths.
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

__all__ = [
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
    "QMeraPairSpec",
    "QMeraPrototypeLayout",
    "QMeraSchedule",
    "QMeraScaleSpec",
    "QMeraSchematicBlock",
    "QMeraSymmrayFermionBackend",
    "QMeraUnitarySpec",
    "UserGateFamily",
    "available_qmera_pair_ansatzes",
    "available_qmera_pair_ansatze",
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
    "get_qmera_pair_ansatz",
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
    "qmera_pair_gate_spec",
    "qmera_schematic_blocks",
    "qmera_symmray_fermi_hubbard_terms",
    "qmera_symmray_majorana_terms",
    "resolve_gate_spec",
    "select_lightcone",
    "site_tags_for_where",
    "symmray_fermion_gate_registry",
    "symmray_majorana_gate_registry",
]


def __dir__():
    """List exports and child modules without loading implementations."""
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))


def __getattr__(name):
    """Load the owning implementation only when requested."""
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .builders import QMeraAnsatz, QMeraBuilder  # noqa: F401
    from .cache import QMeraContractionPathCache, build_qmera_contraction_optimizer  # noqa: F401
    from .compiled import (  # noqa: F401
        QMeraCompiledLightconeChunk,
        compile_qmera_parametric_lightcone,
        compile_qmera_parametric_lightcones,
        local_qmera_compiled_lightcone_expectation,
        qmera_compiled_parametric_energy,
    )
    from .fermions import (  # noqa: F401
        QMeraSymmrayFermionBackend,
        qmera_symmray_fermi_hubbard_terms,
        qmera_symmray_majorana_terms,
        symmray_fermion_gate_registry,
        symmray_majorana_gate_registry,
    )
    from .gates import (  # noqa: F401
        GateRegistry,
        GateSpec,
        QMeraPairSpec,
        UserGateFamily,
        available_qmera_pair_ansatzes,
        available_qmera_pair_ansatze,
        default_gate_registry,
        get_qmera_pair_ansatz,
        qmera_pair_gate_spec,
        resolve_gate_spec,
    )
    from .geometry import QMeraGeometry  # noqa: F401
    from .layout import (  # noqa: F401
        QMeraLayoutCandidate,
        QMeraLayoutFinder,
        QMeraLayoutReport,
        QMeraLayoutScore,
    )
    from .lightcones import (  # noqa: F401
        QMeraLightconeTN,
        QMeraLightconeGroup,
        QMeraParametricLightconeChunk,
        build_qmera_lightcone_chunks,
        build_qmera_parametric_lightcone_chunks,
        contract_qmera_lightcone_group,
        contract_qmera_lightcone_tn,
        group_qmera_parametric_lightcone_chunks,
        local_qmera_parametric_lightcone_expectation,
        qmera_parametric_energy,
        qmera_direct_parametric_energy,
        qmera_parametric_lightcone_group_state,
        qmera_parametric_state,
        qmera_parametric_lightcone_state,
        qmera_parametric_lightcone_tn,
        select_lightcone,
        site_tags_for_where,
    )
    from .parametric import QMeraEnergyOptimizer, QMeraParametricEnergyOptimizer  # noqa: F401
    from .prototype import QMeraPrototypeLayout, load_qmera_prototype_layout  # noqa: F401
    from .schedules import (  # noqa: F401
        QMeraBlockSpec,
        QMeraDisentanglerSpec,
        QMeraGatePlacement,
        QMeraIsometrySpec,
        QMeraLayerSpec,
        QMeraScaleSpec,
        QMeraSchedule,
        QMeraUnitarySpec,
        build_qmera_schedule,
    )
    from .schematics import (  # noqa: F401
        QMeraSchematicBlock,
        draw_qmera_schedule,
        qmera_schematic_blocks,
    )
    from .terms import LocalTerm, normalize_local_terms  # noqa: F401
