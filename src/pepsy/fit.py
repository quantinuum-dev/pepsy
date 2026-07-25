"""Deprecated compatibility facade; use :mod:`pepsy.fitting.local`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.fitting.local", "pepsy.fitting.local")
