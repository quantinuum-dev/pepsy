"""MPS, vector, and PEPS samplers."""

from importlib import import_module

from .samplers import (
    MpsBatchSampleResult,
    MpsSampleResult,
    MpsSampler,
    PEPSSampleResult,
    PepsBpSampler,
    VecSampler,
)

__all__ = [
    "MpsBatchSampleResult",
    "MpsSampleResult",
    "MpsSampler",
    "PEPSSampleResult",
    "PepsBpSampler",
    "VecSampler",
    "mps",
    "peps",
    "vector",
]


def __getattr__(name):
    if name in {"mps", "peps", "vector"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
