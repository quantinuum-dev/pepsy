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
        get_default_array_backend,
        get_default_grad_backend,
        ps_to_peps,
        register_torch_linalg,
        reset_default_backends,
        set_default_array_backend,
        set_default_grad_backend,
        tns_align,
    )
    from .linalg_registrations import reg_complex_svd_jax, reg_complex_svd_torch
    from .fit import FIT
    from .gate import (
        apply_2d_gate,
        apply_2d_gates,
        gates_to_pepo,
        gate_1d,
        gen_long_range_swap_path,
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
    )
    from .ham import (
        ham_tn,
    )
    from .optimize_global import GlobalOptimizer
    from .optimize_sweep import SweepOptimizer
    from .optimize_energy import EnergyOptimizer
    from .optimize_mps import MpsOptimizer

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
    "reg_complex_svd_torch",
    "reg_complex_svd_jax",
    "reset_default_backends",
    "SweepOptimizer",
    "EnergyOptimizer",
    "tns_align",
    "gen_long_range_swap_path",
    "apply_2d_gate",
    "apply_2d_gates",
    "gates_to_pepo",
    "gate_1d",
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
    "ham_tn",
    "ps_to_peps",
    "optimize_global",
    "optimize_sweep",
    "optimize_energy",
    "gradient_solver",
    "gate",
    "ham",
    "boundary_metrics",
    "boundary_states",
    "boundary_sweeps",
    "core",
    "fit",
    "MpsOptimizer",
]


def __getattr__(name):
    """Lazily import public API symbols and common submodules."""
    if name in (
        "boundary_metrics",
        "boundary_states",
        "boundary_sweeps",
        "gate",
        "ham",
        "gradient_solver",
        "optimize_mps",
        "optimize_global",
        "optimize_sweep",
        "optimize_energy",
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

    if name in (
        "gen_long_range_swap_path",
        "apply_2d_gate",
        "apply_2d_gates",
        "gates_to_pepo",
        "gate_1d",
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
    ):
        from .gate import (  # pylint: disable=import-outside-toplevel
            apply_2d_gate,
            apply_2d_gates,
            gates_to_pepo,
            gate_1d,
            gen_long_range_swap_path,
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
        )

        return {
            "gen_long_range_swap_path": gen_long_range_swap_path,
            "apply_2d_gate": apply_2d_gate,
            "apply_2d_gates": apply_2d_gates,
            "gates_to_pepo": gates_to_pepo,
            "gate_1d": gate_1d,
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
        }[name]

    if name == "ham_tn":
        from .ham import ham_tn  # pylint: disable=import-outside-toplevel

        return ham_tn

    if name in ("tns_align", "ps_to_peps"):
        from .core import (  # pylint: disable=import-outside-toplevel
            ps_to_peps,
            tns_align,
        )

        return {
            "tns_align": tns_align,
            "ps_to_peps": ps_to_peps,
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

    if name in ("reg_complex_svd_torch", "reg_complex_svd_jax"):
        from .linalg_registrations import (  # pylint: disable=import-outside-toplevel
            reg_complex_svd_jax,
            reg_complex_svd_torch,
        )

        return {
            "reg_complex_svd_torch": reg_complex_svd_torch,
            "reg_complex_svd_jax": reg_complex_svd_jax,
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

    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")
