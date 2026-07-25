"""Deprecated compatibility facade; use :mod:`pepsy.optimizers.global_opt`."""

from ._compat import install_deprecated_module

install_deprecated_module(
    globals(), "pepsy.optimizers.global_opt", "pepsy.optimizers.global_opt"
)
