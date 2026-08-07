"""MPS, vector, and PEPS samplers."""

from importlib import import_module

from .samplers import (
    FermionConfigurationEncoding,
    MpsDiagonalEstimate,
    MpsBatchSampleResult,
    MpsSampleResult,
    MpsSampler,
    PEPSSampleResult,
    PepsSampler,
    PepsBpSampler,
    VecSampler,
)
from .tree import TreeBatchSampleResult, TreeSampleResult, TreeSampler

__all__ = [
    "FermionConfigurationEncoding",
    "MpsDiagonalEstimate",
    "MpsBatchSampleResult",
    "MpsSampleResult",
    "MpsSampler",
    "PEPSSampleResult",
    "PepsSampler",
    "PepsBpSampler",
    "TreeBatchSampleResult",
    "TreeSampleResult",
    "TreeSampler",
    "VecSampler",
    "tree",
]


def __getattr__(name):
    if name == "tree":
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
