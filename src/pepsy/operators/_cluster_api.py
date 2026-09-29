"""Shared argument conventions for spatial cluster expansions."""

from functools import wraps
from inspect import Parameter, signature
from numbers import Integral


def resolve_cluster_size(order, cluster_size, *, default):
    """Resolve the spatial cutoff, retaining the PEPO ``order`` spelling."""
    for name, value in (("order", order), ("cluster_size", cluster_size)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral)):
            raise TypeError(f"{name} must be an integer.")
    if order is not None and cluster_size is not None and order != cluster_size:
        raise ValueError("order and cluster_size must agree when both are supplied.")
    value = cluster_size if cluster_size is not None else order
    return default if value is None else int(value)


def cluster_size_constructor_alias(cls):
    """Add a keyword alias without duplicating a dataclass's ``order`` field.

    Keeping only ``order`` as stored state preserves ``dataclasses.replace``.
    Bind the original signature to distinguish omitted and explicit order,
    including the legacy positional argument, before checking the alias.
    """
    init = cls.__init__
    original = signature(init)
    default = original.parameters["order"].default

    @wraps(init)
    def initialize(self, *args, cluster_size=None, **kwargs):
        bound = original.bind(self, *args, **kwargs)
        bound.arguments["order"] = resolve_cluster_size(
            bound.arguments.get("order"), cluster_size, default=default,
        )
        init(*bound.args, **bound.kwargs)

    initialize.__signature__ = original.replace(parameters=(
        *original.parameters.values(),
        Parameter("cluster_size", Parameter.KEYWORD_ONLY, default=None),
    ))
    cls.__init__ = initialize
    return cls
