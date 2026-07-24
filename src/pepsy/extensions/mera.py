"""MERA/qMERA extension, backed by :mod:`pepsy.optimizers.mera`."""

from ._proxy import make_proxy

__all__ = []
__getattr__, __dir__ = make_proxy("pepsy.optimizers.mera")

