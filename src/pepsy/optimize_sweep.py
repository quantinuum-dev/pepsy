"""Deprecated compatibility facade; use :mod:`pepsy.optimizers.sweep`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.optimizers.sweep", "pepsy.optimizers.sweep")
