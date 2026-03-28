"""Shared tensor-network tag and physical-index validation helpers."""

from __future__ import annotations

import re

_TAG_X = re.compile(r"^X\d+$")
_TAG_Y = re.compile(r"^Y\d+$")
_TAG_I = re.compile(r"^I\d+(?:,\d+)*$")
_PHYS_OUTER = re.compile(r"^[kb]\d+(?:,\d+)*$")

__all__ = ["_PHYS_OUTER", "validate_tensor_network_tags"]


def validate_tensor_network_tags(tn):
    """Ensure PEPS lattice/site tags are present for shape inference."""
    tags = set(getattr(tn, "tags", ()))
    has_x = any(isinstance(tag, str) and _TAG_X.fullmatch(tag) for tag in tags)
    has_y = any(isinstance(tag, str) and _TAG_Y.fullmatch(tag) for tag in tags)
    has_i = any(isinstance(tag, str) and _TAG_I.fullmatch(tag) for tag in tags)

    if not (has_x and has_y and has_i):
        raise ValueError(
            "Input network must contain X*, Y*, and I* tags "
            "(I<int>[,<int>...])."
        )
