"""Deprecated compatibility facade; use :mod:`pepsy.operators.gates`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.operators.gates", "pepsy.operators.gates")
