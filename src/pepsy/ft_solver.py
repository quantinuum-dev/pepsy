"""Deprecated compatibility facade; use :mod:`pepsy.solvers.finite_difference`."""

from ._compat import install_deprecated_module

install_deprecated_module(
    globals(), "pepsy.solvers.finite_difference", "pepsy.solvers.finite_difference"
)
