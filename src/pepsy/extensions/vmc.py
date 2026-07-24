"""Variational Monte Carlo extension.

The implementation remains in :mod:`pepsy.vmc`; this namespace is a clearly
marked, lazy entry point for the optional Torch/NetKet integrations.
"""

from ._proxy import make_proxy

__all__ = []
__getattr__, __dir__ = make_proxy("pepsy.vmc")

