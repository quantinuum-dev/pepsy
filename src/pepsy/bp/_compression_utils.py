"""Shared policies for local BP/SU compression routes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings
from typing import Any

import numpy as np

from ._backend import cast_like as _cast_like


class CompressionBudgetError(RuntimeError):
    """Raised when a requested local contraction exceeds its cost budget."""


@dataclass(frozen=True)
class ContractionCost:
    """Cotengra log-cost estimate for one local contraction."""

    flops_log10: float
    peak_memory_log2: float

    def as_dict(self) -> dict[str, float]:
        """Return the stable dictionary form used by public results."""
        return {
            "flops_log10": self.flops_log10,
            "peak_memory_log2": self.peak_memory_log2,
        }


@dataclass
class ContractionPlanCache:
    """Reuse Cotengra trees for topology-identical local contractions."""

    plans: dict[Any, Any] = field(default_factory=dict)
    costs: dict[Any, dict[str, float]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def _topology_key(network, output_inds, optimize):
        tensors = tuple(
            sorted(
                (
                    tuple(tensor.inds),
                    tuple(tensor.shape),
                    tuple(sorted(map(repr, tensor.tags))),
                )
                for tensor in network.tensors
            )
        )
        return (tensors, tuple(output_inds), repr(optimize))

    def clear(self) -> None:
        """Discard plans and reset statistics."""
        self.plans.clear()
        self.costs.clear()
        self.hits = 0
        self.misses = 0

    def snapshot(self) -> dict[str, int]:
        """Return stable cache statistics."""
        return {
            "plans": len(self.plans),
            "hits": int(self.hits),
            "misses": int(self.misses),
        }

    def plan_for(self, network, *, output_inds, optimize):
        """Return ``(tree, cost, cache_hit)`` for one network."""
        key = self._topology_key(network, output_inds, optimize)
        if key in self.plans:
            self.hits += 1
            return self.plans[key], self.costs[key], True
        self.misses += 1
        tree = network.contract(
            get="tree",
            output_inds=output_inds,
            optimize=optimize,
        )
        cost = estimate_cost(tree).as_dict()
        self.plans[key] = tree
        self.costs[key] = cost
        return tree, cost, False


def validate_input_mode(input_mode: str, gauges) -> str:
    """Resolve the common physical-versus-SU-core input convention."""
    if input_mode not in {"auto", "physical", "su_core"}:
        raise ValueError("input_mode must be 'auto', 'physical', or 'su_core'")
    if input_mode == "auto":
        # A non-empty gauge mapping is the only unambiguous signal that the
        # caller supplied an SU core. With no gauges, the network is already
        # physical and a fresh BP solve is the safe default.
        return "su_core" if gauges else "physical"
    if input_mode == "su_core" and not gauges:
        raise ValueError(
            "input_mode='su_core' requires a non-empty gauges mapping"
        )
    return input_mode


def validate_bp_convergence_policy(policy: str) -> str:
    """Validate the policy for an unfinished fresh BP solve."""
    if policy not in {"ignore", "warn", "raise"}:
        raise ValueError("bp_convergence must be 'ignore', 'warn', or 'raise'")
    return policy


def enforce_bp_convergence_policy(info: dict[str, Any], policy: str) -> None:
    """Warn or raise when an internally run BP solve did not converge."""
    policy = validate_bp_convergence_policy(policy)
    if info.get("source") != "fresh_bp" or info.get("converged", True):
        return
    message = (
        "fresh D2BP did not converge before reduced loop-cluster construction: "
        f"iterations={info.get('iterations')!r}, "
        f"max_mdiff={info.get('max_mdiff')!r}"
    )
    if policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    elif policy == "raise":
        raise RuntimeError(message)


def prepare_working_network(tn, gauges, *, input_mode: str, smudge: float = 0.0):
    """Copy ``tn`` and optionally insert SU gauges into its tensors.

    The returned gauge values are cast to the endpoint backend and dtype. The
    input network is never modified by this helper.
    """
    if gauges is not None and not hasattr(gauges, "items"):
        raise TypeError("gauges must be a mapping")
    gauges = {} if gauges is None else gauges
    mode = validate_input_mode(input_mode, gauges)
    work = tn.copy()
    gauge_inputs = {}
    for index, gauge in gauges.items():
        if index not in work.ind_map:
            raise ValueError(f"SU gauge refers to unknown bond {index!r}")
        endpoints = tuple(work.ind_map[index])
        if len(endpoints) != 2:
            raise ValueError(
                f"SU gauge bond {index!r} must have two endpoints, got {endpoints!r}"
            )
        endpoint = endpoints[0]
        gauge_inputs[index] = _cast_like(gauge, work.tensor_map[endpoint].data)

    if mode == "su_core":
        work.gauge_simple_insert(gauge_inputs, smudge=smudge)
    return work, gauge_inputs, mode


def resolve_d2bp_boundaries(
    work,
    boundary_messages,
    gauges,
    *,
    run_bp: bool,
    bp_runner: str,
    bp_opts: dict[str, Any] | None,
    bp_convergence: str = "ignore",
):
    """Use explicit closures, SU closures, or run a fresh D2BP solve."""
    bp_convergence = validate_bp_convergence_policy(bp_convergence)
    # Preserve the caller's strongest information first: directed messages
    # describe the actual working network and can contain non-diagonal
    # correlations that diagonal SU vectors cannot represent.
    if boundary_messages is not None:
        return boundary_messages, {
            "source": "boundary_messages",
            "bp_convergence": bp_convergence,
            "converged": None,
            "iterations": None,
            "max_mdiff": None,
        }
    if gauges:
        return None, {
            "source": "su_gauges",
            "bp_convergence": bp_convergence,
            "converged": None,
            "iterations": None,
            "max_mdiff": None,
        }
    if not run_bp:
        return None, {
            "source": "none",
            "bp_convergence": bp_convergence,
            "converged": None,
            "iterations": None,
            "max_mdiff": None,
        }

    # With no explicit closure, the public API chooses plain D2BP by default.
    # Relay BP is opt-in because it has different convergence dynamics and
    # should not silently change a reproducible default calculation.
    options = {} if bp_opts is None else dict(bp_opts)
    if bp_runner == "plain":
        from .relay import two_norm_bp

        result = two_norm_bp(work, **options)
    elif bp_runner == "relay":
        from .relay import relay_bp

        result = relay_bp(work, method="d2bp", **options)
    else:
        raise ValueError("bp_runner must be 'plain' or 'relay'")

    info = {
        "source": "fresh_bp",
        "bp_convergence": bp_convergence,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "max_mdiff": float(result.max_mdiff),
    }
    enforce_bp_convergence_policy(info, bp_convergence)
    return result.messages, info


def validate_cost_options(
    max_flops_log10,
    max_peak_memory_log2,
    on_budget: str,
) -> tuple[float | None, float | None, str]:
    """Validate local contraction cost controls."""
    for name, value in (
        ("max_flops_log10", max_flops_log10),
        ("max_peak_memory_log2", max_peak_memory_log2),
    ):
        if value is not None and (
            not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite or None")
    if on_budget not in {"ignore", "warn", "raise"}:
        raise ValueError("on_budget must be 'ignore', 'warn', or 'raise'")
    return (
        None if max_flops_log10 is None else float(max_flops_log10),
        None if max_peak_memory_log2 is None else float(max_peak_memory_log2),
        on_budget,
    )


def cost_check_requested(
    cost_check: bool,
    max_flops_log10: float | None,
    max_peak_memory_log2: float | None,
) -> bool:
    """Return whether a contraction tree should be built before measuring."""
    if not isinstance(cost_check, bool):
        raise TypeError("cost_check must be a bool")
    # A limit necessarily requires a preflight even when the caller omitted
    # the convenience flag; otherwise an over-budget contraction could start
    # before Pepsy has a chance to reject it.
    return bool(
        cost_check
        or max_flops_log10 is not None
        or max_peak_memory_log2 is not None
    )


def estimate_cost(tree) -> ContractionCost:
    """Extract standard Cotengra log-cost diagnostics from a tree."""
    return ContractionCost(
        flops_log10=float(tree.total_flops(log=10)),
        peak_memory_log2=float(tree.peak_size(log=2)),
    )


def _over_budget(cost: ContractionCost, max_flops_log10, max_peak_memory_log2):
    return (
        (
            max_flops_log10 is not None
            and cost.flops_log10 > max_flops_log10
        )
        or (
            max_peak_memory_log2 is not None
            and cost.peak_memory_log2 > max_peak_memory_log2
        )
    )


def enforce_cost_budget(
    cost: ContractionCost,
    *,
    max_flops_log10: float | None,
    max_peak_memory_log2: float | None,
    on_budget: str,
    label: str,
) -> None:
    """Reject an over-budget contraction before tensor data is contracted."""
    if not _over_budget(cost, max_flops_log10, max_peak_memory_log2):
        return
    message = (
        f"{label} contraction exceeds the requested budget: "
        f"flops_log10={cost.flops_log10:.3f}, "
        f"peak_memory_log2={cost.peak_memory_log2:.3f}; "
        f"limits=({max_flops_log10!r}, {max_peak_memory_log2!r})"
    )
    if on_budget == "ignore":
        return
    if on_budget == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    raise CompressionBudgetError(message)


def contract_with_preflight(
    network,
    *,
    output_inds,
    optimize,
    cost_check: bool,
    max_flops_log10: float | None,
    max_peak_memory_log2: float | None,
    on_budget: str,
    label: str,
):
    """Estimate then contract one network, reusing the estimated tree."""
    if cost_check:
        tree = network.contract(
            get="tree",
            output_inds=output_inds,
            optimize=optimize,
        )
        cost = estimate_cost(tree)
        enforce_cost_budget(
            cost,
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
            on_budget=on_budget,
            label=label,
        )
        return (
            network.contract(output_inds=output_inds, optimize=tree),
            cost.as_dict(),
        )
    return (
        network.contract(output_inds=output_inds, optimize=optimize),
        None,
    )


def contract_many_with_preflight(
    networks,
    *,
    output_inds,
    optimize,
    cost_check: bool,
    max_flops_log10: float | None,
    max_peak_memory_log2: float | None,
    on_budget: str,
    label: str,
    plan_cache: ContractionPlanCache | None = None,
):
    """Preflight every network before contracting any of them."""
    if not cost_check and plan_cache is None:
        return [
            network.contract(output_inds=output_inds, optimize=optimize)
            for network in networks
        ], []

    plans = []
    costs = []
    for network in networks:
        if plan_cache is None:
            tree = network.contract(
                get="tree",
                output_inds=output_inds,
                optimize=optimize,
            )
            cost = estimate_cost(tree).as_dict()
        else:
            tree, cost, _ = plan_cache.plan_for(
                network,
                output_inds=output_inds,
                optimize=optimize,
            )
        plans.append(tree)
        costs.append(cost)
        enforce_cost_budget(
            ContractionCost(**cost),
            max_flops_log10=max_flops_log10,
            max_peak_memory_log2=max_peak_memory_log2,
            on_budget=on_budget,
            label=label,
        )

    values = [
        network.contract(output_inds=output_inds, optimize=tree)
        for network, tree in zip(networks, plans)
    ]
    return values, costs


def aggregate_costs(costs) -> dict[str, float] | None:
    """Combine independent contraction costs conservatively."""
    costs = tuple(costs)
    if not costs:
        return None
    return {
        "flops_log10": math.log10(
            sum(10.0 ** float(cost["flops_log10"]) for cost in costs)
        ),
        "peak_memory_log2": max(
            float(cost["peak_memory_log2"]) for cost in costs
        ),
    }


__all__ = [
    "CompressionBudgetError",
    "ContractionCost",
    "ContractionPlanCache",
    "aggregate_costs",
    "contract_many_with_preflight",
    "contract_with_preflight",
    "cost_check_requested",
    "enforce_bp_convergence_policy",
    "estimate_cost",
    "prepare_working_network",
    "resolve_d2bp_boundaries",
    "validate_cost_options",
    "validate_bp_convergence_policy",
    "validate_input_mode",
]
