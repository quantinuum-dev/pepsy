"""Tensor-network contraction and optimizer facade."""

from .core import (
    build_compressed_optimizer,
    build_optimizer,
    contract_hypercompressed_tn,
    tn_fidelity,
    tn_norm,
    tns_align,
)

__all__ = [
    "build_compressed_optimizer",
    "build_optimizer",
    "contract_hypercompressed_tn",
    "tn_fidelity",
    "tn_norm",
    "tns_align",
]
