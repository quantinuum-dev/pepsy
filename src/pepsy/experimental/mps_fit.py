"""Research-stage MPS compression backend declarations.

The stable implementation is :class:`pepsy.FIT` and
``MpsOptimizer(mode="fit")`` (the clear alias of ``mode="dmrg"``). This
module names newer algorithms without pretending that a sequential fallback
is their published parallel method.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentalMpsFitBackend:
    """Status record for a research-stage MPS compression backend."""

    name: str
    status: str
    reference: str
    available: bool
    reason: str


_BACKENDS = {
    "ptebd-ipmc": ExperimentalMpsFitBackend(
        name="ptebd-ipmc",
        status="design",
        reference="https://doi.org/10.1103/PhysRevB.110.085149",
        available=False,
        reason=(
            "The published method requires parallel independent MPS "
            "compression scheduling and explicit norm stabilization. Pepsy's "
            "current sequential FIT loop is not labeled pTEBD/IPMC."
        ),
    ),
    "local-tdvp-circuit": ExperimentalMpsFitBackend(
        name="local-tdvp-circuit",
        status="research-preprint",
        reference="https://arxiv.org/abs/2508.10096",
        available=False,
        reason=(
            "The local-TDVP circuit method is still a research preprint and "
            "needs an independently validated tangent-space integrator."
        ),
    ),
}

__all__ = [
    "ExperimentalMpsFitBackend",
    "experimental_mps_fit_backends",
    "require_experimental_mps_fit_backend",
]


def experimental_mps_fit_backends():
    """Return copy-safe status metadata for proposed MPS FIT backends."""
    return {name: asdict(record) for name, record in _BACKENDS.items()}


def require_experimental_mps_fit_backend(name):
    """Return an available backend record or fail with its research status.

    This explicit gate prevents selecting an algorithm name that Pepsy has not
    implemented faithfully. It will become the dispatch boundary when a
    backend has correctness, symmetry, and performance regressions.
    """
    name = str(name).strip().lower()
    try:
        record = _BACKENDS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unknown experimental MPS FIT backend {name!r}; choose {choices}."
        ) from exc
    if not record.available:
        raise NotImplementedError(
            f"Experimental backend {name!r} is {record.status}: "
            f"{record.reason} Reference: {record.reference}"
        )
    return record
