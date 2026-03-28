"""Pepsy boundary-contraction library package."""

from importlib import import_module
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

__version__ = _pkg_version("pepsy")

if TYPE_CHECKING:
    from . import (
        boundary_norm,
        boundary_states,
        boundary_sweeps,
        core,
        debug,
        dmrg_fit,
        gate,
        gradient_solver,
        optimize_global,
        optimize_sweep,
    )
    from .boundary_norm import (
        BoundaryContractResult,
        ContractBoundary,
        infidelity,
        normalize,
        prepare_boundary_inputs,
    )
    from .boundary_states import BdyMPS, make_numpy_array_caster
    from .boundary_sweeps import CompBdy
    from .core import (
        get_default_array_backend,
        get_default_grad_backend,
        product_state_peps,
        register_torch_linalg,
        reset_default_backends,
        set_default_array_backend,
        set_default_grad_backend,
        tn_applied,
    )
    from .linalg_registrations import reg_complex_svd_jax, reg_complex_svd_torch
    from .debug import plot_sweep_diagnostics, plot_inner_loss, plot_global_loss_trajectory
    from .dmrg_fit import FIT
    from .gate import (
        apply_2dtn_,
        apply_gates,
        canonize_mps,
        gate_1d,
        gen_long_range_swap_path,
    )
    from .optimize_global import GlobalOptimizer
    from .optimize_sweep import PEPSSweepOptimizer, SweepResult

__all__ = [
    "__version__",
    "BdyMPS",
    "CompBdy",
    "BoundaryContractResult",
    "ContractBoundary",
    "prepare_boundary_inputs",
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
    "PEPSSweepOptimizer",
    "SweepResult",
    "plot_sweep_diagnostics",
    "plot_inner_loss",
    "plot_global_loss_trajectory",
    "tn_applied",
    "gen_long_range_swap_path",
    "apply_2dtn_",
    "apply_gates",
    "gate_1d",
    "canonize_mps",
    "product_state_peps",
    "optimize_global",
    "optimize_sweep",
    "gradient_solver",
    "gate",
    "boundary_norm",
    "boundary_states",
    "boundary_sweeps",
    "core",
    "dmrg_fit",
    "debug",
]


def __getattr__(name):
    """Lazily import public API symbols and common submodules."""
    if name in (
        "boundary_norm",
        "boundary_states",
        "boundary_sweeps",
        "debug",
        "gate",
        "gradient_solver",
        "optimize_global",
        "optimize_sweep",
        "core",
        "dmrg_fit",
    ):
        return import_module(f".{name}", __name__)

    if name in (
        "ContractBoundary",
        "prepare_boundary_inputs",
        "BoundaryContractResult",
        "normalize",
        "infidelity",
    ):
        from .boundary_norm import (  # pylint: disable=import-outside-toplevel
            BoundaryContractResult,
            ContractBoundary,
            infidelity,
            normalize,
            prepare_boundary_inputs,
        )

        return {
            "BoundaryContractResult": BoundaryContractResult,
            "ContractBoundary": ContractBoundary,
            "infidelity": infidelity,
            "prepare_boundary_inputs": prepare_boundary_inputs,
            "normalize": normalize,
        }[name]

    if name == "GlobalOptimizer":
        from .optimize_global import GlobalOptimizer  # pylint: disable=import-outside-toplevel

        return GlobalOptimizer

    if name == "FIT":
        from .dmrg_fit import FIT  # pylint: disable=import-outside-toplevel

        return FIT

    if name in (
        "gen_long_range_swap_path",
        "apply_2dtn_",
        "apply_gates",
        "gate_1d",
        "canonize_mps",
    ):
        from .gate import (  # pylint: disable=import-outside-toplevel
            apply_2dtn_,
            apply_gates,
            canonize_mps,
            gate_1d,
            gen_long_range_swap_path,
        )

        return {
            "gen_long_range_swap_path": gen_long_range_swap_path,
            "apply_2dtn_": apply_2dtn_,
            "apply_gates": apply_gates,
            "gate_1d": gate_1d,
            "canonize_mps": canonize_mps,
        }[name]

    if name in ("tn_applied", "product_state_peps"):
        from .core import (  # pylint: disable=import-outside-toplevel
            product_state_peps,
            tn_applied,
        )

        return {
            "tn_applied": tn_applied,
            "product_state_peps": product_state_peps,
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

    if name in ("plot_sweep_diagnostics", "plot_inner_loss", "plot_global_loss_trajectory"):
        from .debug import (  # pylint: disable=import-outside-toplevel
            plot_global_loss_trajectory,
            plot_inner_loss,
            plot_sweep_diagnostics,
        )

        return {
            "plot_sweep_diagnostics": plot_sweep_diagnostics,
            "plot_inner_loss": plot_inner_loss,
            "plot_global_loss_trajectory": plot_global_loss_trajectory,
        }[name]

    if name in ("PEPSSweepOptimizer", "SweepResult"):
        from .optimize_sweep import (  # pylint: disable=import-outside-toplevel
            PEPSSweepOptimizer,
            SweepResult,
        )

        return {
            "PEPSSweepOptimizer": PEPSSweepOptimizer,
            "SweepResult": SweepResult,
        }[name]

    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")
