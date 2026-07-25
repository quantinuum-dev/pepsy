"""Deprecated compatibility facade; use :mod:`pepsy.solvers.gradient`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.solvers.gradient", "pepsy.solvers.gradient")
