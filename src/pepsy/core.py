"""Deprecated compatibility facade; use :mod:`pepsy.tensors.core`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.tensors.core", "pepsy.tensors.core")
