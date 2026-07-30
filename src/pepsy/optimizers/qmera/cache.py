"""Reusable contraction optimizers for qMERA local cones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...tensors import build_optimizer

__all__ = [
    "QMeraContractionPathCache",
    "build_qmera_contraction_optimizer",
]


@dataclass
class QMeraContractionPathCache:
    """Lazily build reusable cotengra optimizers per local-cone topology.

    A qMERA schedule produces a small number of repeated cone topologies. The
    cache keeps one :class:`cotengra.ReusableHyperOptimizer` per topology key,
    so numerator and norm contractions for repeated terms reuse the same
    searched paths. Passing a directory in ``optimizer_options`` additionally
    persists cotengra's reusable paths across processes.
    """

    optimizer_options: Mapping[str, Any] = field(default_factory=dict)
    _optimizers: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)

    def optimizer_for(self, key=None):
        """Return the reusable optimizer associated with ``key``."""
        key = "default" if key is None else key
        try:
            return self._optimizers[key]
        except KeyError:
            optimizer = build_qmera_contraction_optimizer(**dict(self.optimizer_options))
            self._optimizers[key] = optimizer
            return optimizer

    @property
    def num_cached_paths(self):
        """Number of topology-specific reusable optimizers created so far."""
        return len(self._optimizers)

    def resolve(self, optimize, *, key=None):
        """Resolve an ``optimize`` setting, reusing paths for auto settings."""
        if optimize is None or str(optimize).lower() in {"auto", "auto-hq"}:
            return self.optimizer_for(key)
        return optimize


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
