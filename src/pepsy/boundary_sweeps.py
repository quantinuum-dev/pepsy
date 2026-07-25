"""Deprecated compatibility facade; use :mod:`pepsy.boundary.sweeps`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.boundary.sweeps", "pepsy.boundary.sweeps")
