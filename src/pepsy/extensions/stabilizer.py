"""Stabilizer tensor-network extension, backed by the STN optimizer package."""

from ._proxy import make_proxy

__all__ = []
__getattr__, __dir__ = make_proxy("pepsy.optimizers.stabilizer_tn")

