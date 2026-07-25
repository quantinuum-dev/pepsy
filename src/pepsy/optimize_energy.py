"""Deprecated compatibility facade; use :mod:`pepsy.optimizers.energy`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.optimizers.energy", "pepsy.optimizers.energy")
