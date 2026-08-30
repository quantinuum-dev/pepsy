"""Shared PEPO geometry/planner boundary.

The dense PEPO implementation currently supplies the planner algorithms, but
fixed-channel bases resolve them through this small module rather than
reaching through the legacy ``cluster`` facade.  This keeps geometry and
planner ownership explicit while allowing the dense implementation to be
split further without changing the basis API.
"""

from . import pepo_dense as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
