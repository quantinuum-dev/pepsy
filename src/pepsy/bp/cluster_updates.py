"""Scalar diagnostics retained across ordered local cluster gate updates."""

from dataclasses import dataclass, field
import math


@dataclass
class LoopClusterUpdateTracker:
    """Track per-pair local fidelities and their product heuristic.

    No tensor networks or device arrays are retained. Reuse one tracker across
    stream chunks; call reset() explicitly for an independent trajectory.
    The product is not a global fidelity estimate or an error bound.
    """

    keep_records: bool = True
    records: list[dict] = field(default_factory=list, init=False)
    pair_updates: int = field(default=0, init=False)
    missing_fidelities: int = field(default=0, init=False)
    _log_product: float = field(default=0., init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.keep_records, bool):
            raise TypeError('keep_records must be a bool')

    @property
    def accumulated_local_fidelity(self):
        """Product of accepted local metrics, absent if any are unavailable."""
        return None if self.missing_fidelities else math.exp(self._log_product)

    @property
    def accumulated_local_infidelity(self):
        value = self.accumulated_local_fidelity
        return None if value is None else -math.expm1(self._log_product)

    def reset(self):
        """Clear scalar records and start a new independent accumulation."""
        self.records.clear()
        self.pair_updates = self.missing_fidelities = 0
        self._log_product = 0.

    def record(self, result):
        """Append one completed pair fit, without retaining its tensor state."""
        solution = result.solution
        fidelity = solution.local_fidelity
        self.pair_updates += 1
        if fidelity is None or not math.isfinite(fidelity) or not 0 <= fidelity <= 1:
            self.missing_fidelities += 1
        elif fidelity == 0:
            self._log_product = -math.inf
        else:
            self._log_product += (math.log1p(fidelity-1) if fidelity > .5
                                  else math.log(fidelity))
        row = dict(pair_update=self.pair_updates, where=tuple(result.pair.where),
                   which=result.problem.which, solver=solution.solver,
                   iterations=solution.iterations, converged=solution.converged,
                   termination_reason=solution.termination_reason,
                   bond=int(solution.left.shape[-1]), local_fidelity=fidelity,
                   local_infidelity=solution.local_infidelity,
                   fidelity_status=solution.fidelity_status,
                   accumulated_local_fidelity=self.accumulated_local_fidelity,
                   accumulated_local_infidelity=self.accumulated_local_infidelity)
        if self.keep_records:
            self.records.append(row)
        return row

    def summary(self):
        """Return JSON-compatible scalar totals for checkpoints and reports."""
        return dict(pair_updates=self.pair_updates, missing_fidelities=self.missing_fidelities,
                    accumulated_local_fidelity=self.accumulated_local_fidelity,
                    accumulated_local_infidelity=self.accumulated_local_infidelity,
                    convention='product of local environment fidelities; not global fidelity')
