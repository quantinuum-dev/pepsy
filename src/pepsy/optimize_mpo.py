"""Deprecated compatibility facade; use :mod:`pepsy.optimizers.mpo`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.optimizers.mpo", "pepsy.optimizers.mpo")
