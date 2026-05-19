"""Pepsy boundary-contraction library package."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING

try:
    __version__ = _pkg_version("pepsy")
except PackageNotFoundError:
    __version__ = "0+unknown"

if TYPE_CHECKING:
    from . import (
        boundary_metrics,
        boundary_states,
        boundary_sweeps,
        core,
        fit,
        gate,
        ham,
        gradient_solver,
        optimize_global,
        optimize_sweep,
        optimize_energy,
        optimize_mpo,
    )
    from .boundary_metrics import (
        BoundaryContractResult,
        contract_boundary,
        build_bra_ket,
        infidelity,
        normalize,
    )
    from .boundary_states import BdyMPS, make_numpy_array_caster
    from .boundary_sweeps import CompBdy
    from .core import (
        expec_mpo,
        get_default_array_backend,
        get_default_grad_backend,
        measure_obs,
        id_to_mpo,
        id_to_pepo,
        ps_to_peps,
        ps_to_mps,
        ps_to_pepo,
        ps_to_mpo,
        random_haar_qubit,
        hrps_to_peps,
        hrps_to_mps,
        register_torch_linalg,
        reset_default_backends,
        set_default_array_backend,
        set_default_grad_backend,
        tns_align,
    )
    from .fit import FIT
    from .gate import (
        gate,
        build_pepo_from_gates,
        build_mpo_from_gates,
        pauli,
        x,
        y,
        z,
        s,
        sdg,
        t,
        tdg,
        h,
        hadamard,
        cnot,
        cx,
        cy,
        cz,
        swap,
        iswap,
        phase,
        u1,
        u2,
        cphase,
        crx,
        cry,
        crz,
        cu1,
        cu2,
        cu3,
        rx,
        ry,
        rz,
        rxx,
        ryy,
        rzz,
        u3,
        su4,
        fsim,
        fsimg,
    )
    from .ham import (
        ham_tn,
    )
    from .optimize_global import GlobalOptimizer
    from .optimize_sweep import SweepOptimizer
    from .optimize_energy import EnergyOptimizer
    from .optimize_mps import MpsOptimizer
    from .optimize_mpo import MpoOptimizer

__all__ = [
    "__version__",
    "BdyMPS",
    "CompBdy",
    "BoundaryContractResult",
    "contract_boundary",
    "build_bra_ket",
    "normalize",
    "infidelity",
    "GlobalOptimizer",
    "FIT",
    "make_numpy_array_caster",
    "set_default_array_backend",
    "get_default_array_backend",
    "set_default_grad_backend",
    "get_default_grad_backend",
    "register_torch_linalg",
    "reset_default_backends",
    "SweepOptimizer",
    "EnergyOptimizer",
    "tns_align",
    "measure_obs",
    "gate",
    "build_pepo_from_gates",
    "build_mpo_from_gates",
    "pauli",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "h",
    "hadamard",
    "cnot",
    "cx",
    "cy",
    "cz",
    "swap",
    "iswap",
    "phase",
    "u1",
    "u2",
    "cphase",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cu2",
    "cu3",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "u3",
    "su4",
    "fsim",
    "fsimg",
    "ham_tn",
    "expec_mpo",
    "id_to_mpo",
    "id_to_pepo",
    "ps_to_peps",
    "ps_to_mps",
    "ps_to_pepo",
    "ps_to_mpo",
    "random_haar_qubit",
    "hrps_to_peps",
    "hrps_to_mps",
    "optimize_global",
    "optimize_sweep",
    "optimize_energy",
    "optimize_mps",
    "optimize_mpo",
    "gradient_solver",
    "ham",
    "boundary_metrics",
    "boundary_states",
    "boundary_sweeps",
    "core",
    "fit",
    "MpsOptimizer",
    "MpoOptimizer",
]


def __getattr__(name):
    """Lazily import public API symbols and common submodules."""
    if name in (
        "boundary_metrics",
        "boundary_states",
        "boundary_sweeps",
        "ham",
        "gradient_solver",
        "optimize_mps",
        "optimize_global",
        "optimize_sweep",
        "optimize_energy",
        "optimize_mpo",
        "core",
        "fit",
    ):
        return import_module(f".{name}", __name__)

    if name in (
        "contract_boundary",
        "build_bra_ket",
        "BoundaryContractResult",
        "normalize",
        "infidelity",
    ):
        from .boundary_metrics import (  # pylint: disable=import-outside-toplevel
            BoundaryContractResult,
            contract_boundary,
            build_bra_ket,
            infidelity,
            normalize,
        )

        return {
            "BoundaryContractResult": BoundaryContractResult,
            "contract_boundary": contract_boundary,
            "build_bra_ket": build_bra_ket,
            "infidelity": infidelity,
            "normalize": normalize,
        }[name]

    if name == "GlobalOptimizer":
        from .optimize_global import GlobalOptimizer  # pylint: disable=import-outside-toplevel

        return GlobalOptimizer

    if name == "FIT":
        from .fit import FIT  # pylint: disable=import-outside-toplevel

        return FIT

    if name == "gate":
        # Handle "gate" separately: importing from .gate causes Python to bind
        # the submodule to pepsy.__dict__['gate'], overwriting our function.
        # We must explicitly re-set the attribute to the function afterward.
        import sys  # pylint: disable=import-outside-toplevel
        from .gate import gate as _gate_fn  # pylint: disable=import-outside-toplevel
        sys.modules[__name__].__dict__["gate"] = _gate_fn
        return _gate_fn

    if name in (
        "build_pepo_from_gates",
        "build_mpo_from_gates",
        "pauli",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "h",
        "hadamard",
        "cnot",
        "cx",
        "cy",
        "cz",
        "swap",
        "iswap",
        "phase",
        "u1",
        "u2",
        "cphase",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cu2",
        "cu3",
        "rx",
        "ry",
        "rz",
        "rxx",
        "ryy",
        "rzz",
        "u3",
        "su4",
        "fsim",
        "fsimg",
    ):
        import sys as _sys  # pylint: disable=import-outside-toplevel
        from .gate import (  # pylint: disable=import-outside-toplevel
            gate as _gate_fn,
            build_pepo_from_gates,
            build_mpo_from_gates,
            pauli,
            x,
            y,
            z,
            s,
            sdg,
            t,
            tdg,
            h,
            hadamard,
            cnot,
            cx,
            cy,
            cz,
            swap,
            iswap,
            phase,
            u1,
            u2,
            cphase,
            crx,
            cry,
            crz,
            cu1,
            cu2,
            cu3,
            rx,
            ry,
            rz,
            rxx,
            ryy,
            rzz,
            u3,
            su4,
            fsim,
            fsimg,
        )
        # Re-bind "gate" to the function; `from .gate import ...` causes
        # Python to set __dict__["gate"] to the submodule.
        _sys.modules[__name__].__dict__["gate"] = _gate_fn

        return {
            "build_pepo_from_gates": build_pepo_from_gates,
            "build_mpo_from_gates": build_mpo_from_gates,
            "pauli": pauli,
            "x": x,
            "y": y,
            "z": z,
            "s": s,
            "sdg": sdg,
            "t": t,
            "tdg": tdg,
            "h": h,
            "hadamard": hadamard,
            "cnot": cnot,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "swap": swap,
            "iswap": iswap,
            "phase": phase,
            "u1": u1,
            "u2": u2,
            "cphase": cphase,
            "crx": crx,
            "cry": cry,
            "crz": crz,
            "cu1": cu1,
            "cu2": cu2,
            "cu3": cu3,
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "rxx": rxx,
            "ryy": ryy,
            "rzz": rzz,
            "u3": u3,
            "su4": su4,
            "fsim": fsim,
            "fsimg": fsimg,
        }[name]

    if name == "ham_tn":
        from .ham import ham_tn  # pylint: disable=import-outside-toplevel

        return ham_tn

    if name in (
        "tns_align",
        "measure_obs",
        "expec_mpo",
        "id_to_mpo",
        "id_to_pepo",
        "ps_to_peps",
        "ps_to_mps",
        "ps_to_pepo",
        "ps_to_mpo",
        "random_haar_qubit",
        "hrps_to_peps",
        "hrps_to_mps",
    ):
        from .core import (  # pylint: disable=import-outside-toplevel
            expec_mpo,
            measure_obs,
            id_to_mpo,
            id_to_pepo,
            ps_to_peps,
            ps_to_mps,
            ps_to_pepo,
            ps_to_mpo,
            random_haar_qubit,
            hrps_to_peps,
            hrps_to_mps,
            tns_align,
        )

        return {
            "tns_align": tns_align,
            "measure_obs": measure_obs,
            "expec_mpo": expec_mpo,
            "id_to_mpo": id_to_mpo,
            "id_to_pepo": id_to_pepo,
            "ps_to_peps": ps_to_peps,
            "ps_to_mps": ps_to_mps,
            "ps_to_pepo": ps_to_pepo,
            "ps_to_mpo": ps_to_mpo,
            "random_haar_qubit": random_haar_qubit,
            "hrps_to_peps": hrps_to_peps,
            "hrps_to_mps": hrps_to_mps,
        }[name]

    if name in ("BdyMPS", "make_numpy_array_caster"):
        from .boundary_states import (  # pylint: disable=import-outside-toplevel
            BdyMPS,
            make_numpy_array_caster,
        )

        return {
            "BdyMPS": BdyMPS,
            "make_numpy_array_caster": make_numpy_array_caster,
        }[name]

    if name in (
        "set_default_array_backend",
        "get_default_array_backend",
        "set_default_grad_backend",
        "get_default_grad_backend",
        "register_torch_linalg",
        "reset_default_backends",
    ):
        from .core import (  # pylint: disable=import-outside-toplevel
            get_default_array_backend,
            get_default_grad_backend,
            register_torch_linalg,
            reset_default_backends,
            set_default_array_backend,
            set_default_grad_backend,
        )

        return {
            "set_default_array_backend": set_default_array_backend,
            "get_default_array_backend": get_default_array_backend,
            "set_default_grad_backend": set_default_grad_backend,
            "get_default_grad_backend": get_default_grad_backend,
            "register_torch_linalg": register_torch_linalg,
            "reset_default_backends": reset_default_backends,
        }[name]

    if name == "CompBdy":
        from .boundary_sweeps import CompBdy  # pylint: disable=import-outside-toplevel

        return CompBdy

    if name == "SweepOptimizer":
        from .optimize_sweep import SweepOptimizer  # pylint: disable=import-outside-toplevel

        return SweepOptimizer

    if name == "EnergyOptimizer":
        from .optimize_energy import EnergyOptimizer  # pylint: disable=import-outside-toplevel

        return EnergyOptimizer

    if name == "MpsOptimizer":
        from .optimize_mps import MpsOptimizer  # pylint: disable=import-outside-toplevel

        return MpsOptimizer

    if name == "MpoOptimizer":
        from .optimize_mpo import MpoOptimizer  # pylint: disable=import-outside-toplevel

        return MpoOptimizer

    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")
