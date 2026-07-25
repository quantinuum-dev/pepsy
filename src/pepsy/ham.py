"""Deprecated compatibility facade; use :mod:`pepsy.operators.hamiltonians`."""

from ._compat import install_deprecated_module

install_deprecated_module(
    globals(), "pepsy.operators.hamiltonians", "pepsy.operators.hamiltonians"
)
