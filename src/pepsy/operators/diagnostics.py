"""Shared diagnostics vocabulary for operator construction reports.

Concrete report classes remain owned by their algorithm modules because their
detailed fields are representation-specific. ``OperatorReportInfo`` is the
small stable view shared by MPO and PEPO reports: it identifies the family,
algorithm, result representation, and the meaning of the main order/count
fields without pretending to be a global approximation-error bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["OperatorReportInfo"]


@dataclass(frozen=True)
class OperatorReportInfo:
    """Common summary metadata exposed by operator construction reports."""

    family: str
    algorithm: str
    representation: str
    order: int | None = None
    factor_count: int | None = None
    truncated: bool | None = None
    differentiable: bool | None = None

    def as_dict(self):
        """Return a plain serializable summary for logs and dashboards."""

        return asdict(self)
