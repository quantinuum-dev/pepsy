"""Pre-run contraction cost estimates for qMERA local cones."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any

import cotengra as ctg

from .cache import QMeraContractionPathCache
from .compiled import (
    _compiled_tns,
    _expression_topology_key,
    _native_tn_metadata,
    _validate_native_symmray_compile,
)
from .gates import default_gate_registry
from .lightcones import build_qmera_parametric_lightcone_chunks
from .terms import convert_local_terms, normalize_local_terms

__all__ = [
    "QMeraConeCost",
    "QMeraContractionReport",
    "estimate_qmera_contraction_cost",
]


@dataclass(frozen=True)
class QMeraConeCost:
    """Forward cost of one local Hamiltonian term, excluding autodiff."""

    where: tuple[Any, ...]
    num_gates: int
    numerator_flops: int
    denominator_flops: int
    numerator_peak_elements: int
    denominator_peak_elements: int


@dataclass(frozen=True)
class QMeraContractionReport:
    """Dense forward path estimates and paths reusable during compilation."""

    cones: tuple[QMeraConeCost, ...]
    normalized: bool
    element_bytes: int
    unique_topologies: int
    planning_seconds: float
    path_cache: QMeraContractionPathCache = field(repr=False, compare=False)
    contraction_opt: Any = field(repr=False, compare=False)

    @property
    def total_flops(self):
        """Sum of estimated complex-arithmetic FLOPs over all local terms."""
        return sum(
            cone.numerator_flops + (cone.denominator_flops if self.normalized else 0)
            for cone in self.cones
        )

    @property
    def peak_elements(self):
        """Largest estimated live tensor-element count in one evaluated cone."""
        return max(
            max(
                cone.numerator_peak_elements,
                cone.denominator_peak_elements if self.normalized else 0,
            )
            for cone in self.cones
        )

    @property
    def log10_flops(self):
        return math.log10(max(1, self.total_flops))

    @property
    def log2_peak_bytes(self):
        return math.log2(max(1, self.peak_elements * self.element_bytes))

    def summary(self):
        """Short user-facing estimate for this forward loss evaluation."""
        return (
            f"qMERA paths: terms={len(self.cones)}, "
            f"topologies={self.unique_topologies}, "
            f"primed={self.path_cache.num_primed_paths}\n"
            f"Forward: log10(FLOPs)={self.log10_flops:.2f}; "
            f"log2(peak bytes)={self.log2_peak_bytes:.2f} "
            f"({self.element_bytes} bytes/element)\n"
            f"Path planning: {self.planning_seconds:.2f}s; "
            "peak excludes autodiff and optimizer storage"
        )


def _estimate_tree(tn, *, optimize, path_cache, estimates):
    tensors = tuple(tn.tensor_map.values())
    inputs = tuple(tuple(tensor.inds) for tensor in tensors)
    shapes = tuple(tuple(tensor.shape) for tensor in tensors)
    key = _expression_topology_key(inputs, (), shapes)
    if key in estimates:
        return estimates[key]
    tree = ctg.array_contract_tree(
        inputs,
        output=(),
        shapes=shapes,
        optimize=path_cache.resolve(optimize, key=key),
    )
    # A raw path cannot represent slicing; retain the reusable optimizer for
    # sliced trees so compilation keeps the same slicing semantics.
    if not tree.sliced_inds:
        path_cache.prime_path(key, tree.get_path(), optimize=optimize)
    estimates[key] = int(tree.total_flops(dtype="complex")), int(tree.peak_size())
    return estimates[key]


def estimate_qmera_contraction_cost(
    schedule,
    hamiltonian=None,
    *,
    chunks=None,
    gate_registry=None,
    array_backend=None,
    convert_terms=True,
    physical_dim=2,
    optimize="auto-hq",
    path_cache=None,
    normalized=False,
    element_bytes=16,
    product_state_factory=None,
):
    """Plan local cones and estimate forward FLOPs and peak resident bytes.

    The memory estimate applies to one dense contraction at a time, excluding
    autodiff tape, parameter storage, and framework overhead. Pass the returned
    ``path_cache`` with the same ``optimize`` setting to reuse unsliced paths.
    """
    if not isinstance(element_bytes, int) or element_bytes < 1:
        raise ValueError("element_bytes must be a positive integer.")
    if path_cache is None:
        path_cache = QMeraContractionPathCache()
    if not isinstance(path_cache, QMeraContractionPathCache):
        raise TypeError("path_cache must be a QMeraContractionPathCache.")
    gate_registry = default_gate_registry() if gate_registry is None else gate_registry
    if chunks is None:
        terms = normalize_local_terms(hamiltonian)
        if convert_terms:
            terms = convert_local_terms(terms, array_backend)
        chunks = build_qmera_parametric_lightcone_chunks(schedule, terms)
    chunks = tuple(chunks)
    if not chunks:
        raise ValueError("hamiltonian contains no local terms.")
    _validate_native_symmray_compile(
        schedule, chunks, gate_registry, product_state_factory=product_state_factory
    )
    started = perf_counter()
    cones = []
    estimates = {}
    for chunk in chunks:
        numerator, denominator = _compiled_tns(
            schedule,
            chunk,
            gate_registry=gate_registry,
            array_backend=array_backend,
            physical_dim=physical_dim,
            product_state_factory=product_state_factory,
        )
        if _native_tn_metadata(numerator)[0]:
            raise NotImplementedError(
                "Dense-shape qMERA cost estimates do not model native Symmray blocks."
            )
        numerator_flops, numerator_peak = _estimate_tree(
            numerator,
            optimize=optimize,
            path_cache=path_cache,
            estimates=estimates,
        )
        denominator_flops = denominator_peak = 0
        if normalized:
            denominator_flops, denominator_peak = _estimate_tree(
                denominator,
                optimize=optimize,
                path_cache=path_cache,
                estimates=estimates,
            )
        cones.append(
            QMeraConeCost(
                where=chunk.term.where,
                num_gates=chunk.num_gates,
                numerator_flops=numerator_flops,
                denominator_flops=denominator_flops,
                numerator_peak_elements=numerator_peak,
                denominator_peak_elements=denominator_peak,
            )
        )
    return QMeraContractionReport(
        cones=tuple(cones),
        normalized=bool(normalized),
        element_bytes=element_bytes,
        unique_topologies=len(estimates),
        planning_seconds=perf_counter() - started,
        path_cache=path_cache,
        contraction_opt=optimize,
    )
