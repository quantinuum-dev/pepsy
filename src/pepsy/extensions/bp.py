"""Belief-propagation and loop-expansion extension, backed by :mod:`pepsy.bp`."""

from ._proxy import make_proxy

__all__ = []
__getattr__, __dir__ = make_proxy("pepsy.bp")

