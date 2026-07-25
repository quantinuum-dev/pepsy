"""Deprecated compatibility facade; use :mod:`pepsy.boundary.states`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.boundary.states", "pepsy.boundary.states")
