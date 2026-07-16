"""Reusable contraction optimizers for MERA and qMERA local cones."""

from __future__ import annotations

from typing import Any

from ...tensors import build_optimizer

__all__ = ["build_qmera_contraction_optimizer"]


def build_qmera_contraction_optimizer(
    *,
    progbar: bool = False,
    alpha: int = 64,
    max_time: Any = "rate:7e8",
    max_repeats: int = 2**8,
    parallel: Any = "auto",
    optlib: str = "cmaes",
    directory: Any = False,
    hash_method: str = "b",
    overwrite: Any = False,
    on_trial_error: str = "warn",
    slicing_opts: dict | None = None,
    reconf_opts: dict | None = None,
    slicing_reconf_opts: dict | None = None,
):
    """Build a reusable cotengra optimizer for MERA/qMERA local cones.

    The helper intentionally mirrors :func:`pepsy.tensors.build_optimizer` while
    giving MERA examples and call sites a named place to configure path reuse.
    Pass ``directory`` as a path to persist contraction trees across repeated
    local-cone evaluations.
    """
    return build_optimizer(
        progbar=progbar,
        alpha=alpha,
        max_time=max_time,
        max_repeats=max_repeats,
        parallel=parallel,
        optlib=optlib,
        directory=directory,
        hash_method=hash_method,
        overwrite=overwrite,
        on_trial_error=on_trial_error,
        slicing_opts=slicing_opts,
        reconf_opts=reconf_opts,
        slicing_reconf_opts=slicing_reconf_opts,
    )
