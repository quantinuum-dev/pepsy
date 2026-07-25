"""Deprecated compatibility facade; use :mod:`pepsy.sampling.samplers`."""

from ._compat import install_deprecated_module

install_deprecated_module(globals(), "pepsy.sampling.samplers", "pepsy.sampling.samplers")
