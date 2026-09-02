"""Native Quimb Trotter-to-MPO construction.

This module provides the gate-product counterpart to the semantic
``exp_mpo`` builder.  Quimb owns the commuting-layer and Suzuki product
formula, while :class:`pepsy.optimizers.mpo.MpoOptimizer` owns the numerical
MPO replay and compression boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..optimizers.mpo import MpoOptimizer
from .mpo_basis import MPOBasis, _apply_to_backend
from .mpo_semantic import (
    MPOLocalOperatorTerm,
    MPOProductTerm,
    _resolve_compression_cutoff_mode,
    _resolve_exp_step,
)

__all__ = ["TrotterMPOReport", "exp_trotter"]


def _backend_name(value):
    """Return an Autoray backend name without coercing an array to NumPy."""

    try:
        return ar.infer_backend(value)
    except Exception:  # pragma: no cover - defensive backend boundary
        return None


def _validate_order(value):
    """Validate a Quimb-supported Suzuki order."""

    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) not in (1, 2, 4)
    ):
        raise ValueError("order must be one of 1, 2, or 4.")
    return int(value)


def _validate_steps(value):
    """Validate the number of Trotter substeps."""

    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 1
    ):
        raise ValueError("steps must be a positive integer.")
    return int(value)


def _validate_bool(value, *, name):
    """Validate a public boolean scheduler option."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _combine_term(target, site, value):
    """Add one backend-native local contribution into a site/pair mapping."""

    if site in target:
        target[site] = ar.do("add", target[site], value)
    else:
        target[site] = value


def _term_operator(term):
    """Return the local dense matrix represented by a one/two-site term."""

    if isinstance(term, MPOLocalOperatorTerm):
        return term.operator
    operator = term.operators[0]
    for factor in term.operators[1:]:
        operator = ar.do("kron", operator, factor)
    return operator


def _build_local_hamiltonian(basis, coefficients, *, to_backend):
    """Build ``LocalHamGen`` terms and exact isolated-site contributions."""

    pair_terms = {}
    site_terms = {}
    for index, (term, coefficient) in enumerate(zip(basis.terms, coefficients)):
        sites = tuple(int(site) for site in term.sites)
        if len(sites) not in (1, 2):
            raise NotImplementedError(
                "exp_trotter currently supports only one- and two-site terms; "
                f"term {index} acts on {len(sites)} sites."
            )
        if (
            isinstance(term, MPOProductTerm)
            and term.string_operators is not None
            and any(
                right > left + 1
                for left, right in zip(sites, sites[1:])
            )
        ):
            raise NotImplementedError(
                "exp_trotter does not yet encode string operators across a "
                "gap; use a plain two-site term or exp_mpo."
            )
        if isinstance(term, MPOProductTerm) and (
            term.charge is not None or term.braiding.fermionic
        ):
            raise NotImplementedError(
                "exp_trotter currently supports ordinary bosonic MPO terms; "
                "charged or fermionic term metadata requires a native graded "
                "Trotter gate path."
            )
        contribution = ar.do(
            "multiply",
            _term_operator(term),
            coefficient,
        )
        key = tuple(sorted(sites))
        if len(key) == 1:
            _combine_term(site_terms, key[0], contribution)
        else:
            _combine_term(pair_terms, key, contribution)

    if not pair_terms:
        return None, site_terms

    covering = {}
    for pair in pair_terms:
        for site in pair:
            covering.setdefault(site, []).append(pair)
    h1 = {}
    isolated = {}
    for site, operator in site_terms.items():
        pairs = covering.get(site, ())
        if pairs:
            h1[site] = operator
        else:
            isolated[site] = operator

    ham = qtn.LocalHamGen(pair_terms, h1 or None)
    if to_backend is not None:
        def convert(array):
            if _backend_name(array) in {"builtins", "numpy"}:
                return to_backend(array)
            return array

        ham.apply_to_arrays(convert)
    return ham, isolated


def _isolated_gates(isolated, *, step, phys_dim):
    """Exponentiate site terms disconnected from every pair interaction."""

    gates = []
    for site in sorted(isolated):
        operator = ar.do("linalg.expm", ar.do("multiply", isolated[site], step))
        operator = ar.do("reshape", operator, (phys_dim, phys_dim))
        gates.append((operator, (int(site),)))
    return tuple(gates)


def _native_trotter_gates(ham, step, *, order, steps, ordering, fuse_adjacent, alternate):
    """Generate Quimb gates, including the unhashable-backend fallback."""

    kwargs = {
        "order": order,
        "steps": steps,
        "ordering": ordering,
        "fuse_adjacent": fuse_adjacent,
        "alternate": alternate,
    }
    try:
        hash(step)
    except TypeError:
        # Quimb's local exponential cache keys the exponent itself. JAX
        # scalars are intentionally unhashable, so obtain the schedule and
        # metadata with a neutral scalar, then rebuild only the payloads using
        # the real backend-native exponent.
        scheduled = tuple(ham.get_trotter_gates(0.0, **kwargs))
        return tuple(
            type(item)(
                ar.do(
                    "linalg.expm",
                    ar.do(
                        "multiply",
                        ham.get_gate(item.where),
                        ar.do("multiply", step, item.frac),
                    ),
                ),
                item.where,
                item.frac,
                item.layer,
                item.step,
            )
            for item in scheduled
        )
    return tuple(ham.get_trotter_gates(step, **kwargs))


def _normalize_ordering(ham, ordering):
    """Return stable layer metadata and the scheduler input."""

    if isinstance(ordering, str) or ordering is None:
        layers = tuple(ham.get_auto_ordering(ordering, group=True))
        return ordering, layers
    try:
        layers = tuple(
            tuple(tuple(int(site) for site in where) for where in layer)
            for layer in ordering
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "ordering must be a string, None, or a sequence of commuting "
            "layers of site tuples."
        ) from exc
    expected = set(ham.terms)
    actual = [where for layer in layers for where in layer]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "explicit ordering must contain every two-site Hamiltonian term "
            "exactly once."
        )
    for layer in layers:
        sites = [site for where in layer for site in where]
        if len(sites) != len(set(sites)):
            raise ValueError(
                "each explicit Trotter ordering layer must contain only "
                "non-overlapping terms."
            )
    return layers, layers


@dataclass(frozen=True)
class TrotterMPOReport:
    """Diagnostics for a native Trotter gate-product MPO."""

    order: int
    steps: int
    ordering: object
    layers: tuple
    gate_count: int
    isolated_sites: tuple[int, ...]
    mode: str
    chi: int
    cutoff: object
    cutoff_mode: str
    final_bond_dimensions: tuple[int, ...]
    optimizer_timing: dict | None = None


def exp_trotter(
    terms,
    step=None,
    *,
    shape=None,
    mapper=None,
    map_mode="snake",
    parameters=None,
    coefficients=None,
    dt=None,
    phys_dim=None,
    order=2,
    steps=1,
    ordering="sort",
    fuse_adjacent=True,
    alternate=True,
    chi=64,
    mode="mpo",
    contraction_opt="auto-hq",
    cutoff=1.0e-12,
    cutoff_mode="rsum2",
    fidelity_samples=0,
    n_iter=6,
    progress=False,
    optimizer_kwargs=None,
    run_kwargs=None,
    to_backend=None,
    return_report=False,
):
    r"""Build ``exp(step * H)`` as a Quimb-Trotter gate-product MPO.

    Terms use the same ``shape``/``mapper``/``map_mode`` and coefficient
    handling as :func:`exp_mpo`. Quimb's :class:`LocalHamGen` groups the
    normalized one- and two-site terms into commuting layers and generates a
    first-, second-, or fourth-order Suzuki schedule. The overall ``step`` is
    divided across ``steps`` Trotter substeps, so the function approximates
    ``exp(step * H)`` rather than ``exp(steps * step * H)``.

    ``MpoOptimizer`` replays the ordered gates as ket-only updates on an MPO
    identity. Thus ``mode`` selects its numerical backend: ``"mpo"`` is the
    direct multi-site gate path, while ``"svd"`` and ``"dmrg"`` (including
    ``"dmrg1"``/``"dmrg2"``/``"dmrg3"``) select its corresponding local
    compression algorithms. Additional optimizer constructor and run options
    can be supplied through ``optimizer_kwargs`` and ``run_kwargs``.

    This is deliberately a numerical gate-product builder, not a semantic
    history or cluster expansion. ``chi`` is therefore required in practice
    and defaults to 64. The returned object is always a Quimb MPO; use
    ``return_report=True`` or the attached ``pepsy_trotter_optimizer`` for
    diagnostics.
    """

    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean.")
    order = _validate_order(order)
    steps = _validate_steps(steps)
    fuse_adjacent = _validate_bool(fuse_adjacent, name="fuse_adjacent")
    alternate = _validate_bool(alternate, name="alternate")
    if isinstance(cutoff, str) and cutoff.strip().lower() == "auto":
        cutoff = "auto"
    cutoff_mode = _resolve_compression_cutoff_mode(cutoff_mode)
    if (
        not isinstance(chi, Integral)
        or isinstance(chi, bool)
        or int(chi) < 1
    ):
        raise ValueError("chi must be a positive integer.")
    chi = int(chi)
    if to_backend is not None and not callable(to_backend):
        raise TypeError("to_backend must be callable or None.")

    step = _resolve_exp_step(step, dt)
    if to_backend is not None and _backend_name(step) in {"builtins", "numpy"}:
        step = to_backend(step)

    basis = MPOBasis.from_terms(
        terms,
        shape=shape,
        mapper=mapper,
        map_mode=map_mode,
        phys_dim=phys_dim,
        to_backend=to_backend,
    )
    coefficient_values = basis._coefficient_values(  # pylint: disable=protected-access
        parameters,
        coefficients,
    )
    ham, isolated = _build_local_hamiltonian(
        basis,
        coefficient_values,
        to_backend=to_backend,
    )
    if ham is None:
        native_gates = ()
        layers = ()
        scheduler_ordering = ordering
    else:
        scheduler_ordering, layers = _normalize_ordering(ham, ordering)
        substep = ar.do("divide", step, steps)
        native_gates = _native_trotter_gates(
            ham,
            substep,
            order=order,
            steps=steps,
            ordering=scheduler_ordering,
            fuse_adjacent=fuse_adjacent,
            alternate=alternate,
        )

    gate_entries = [
        (
            (
                ar.do("transpose", trotter_gate.U, (1, 0)),
                None,
            ),
            tuple(int(site) for site in trotter_gate.where),
        )
        for trotter_gate in native_gates
    ]
    isolated_entries = _isolated_gates(
        isolated,
        step=step,
        phys_dim=basis.phys_dim,
    )
    gate_entries.extend(
        (
            ((ar.do("transpose", gate, (1, 0)), None), where)
            for gate, where in isolated_entries
        )
    )

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    elif not isinstance(optimizer_kwargs, Mapping):
        raise TypeError("optimizer_kwargs must be a mapping or None.")
    else:
        optimizer_kwargs = dict(optimizer_kwargs)
    reserved_constructor = {"mpo", "gates", "chi", "mode", "contraction_opt"}
    overlap = reserved_constructor.intersection(optimizer_kwargs)
    if overlap:
        raise TypeError(
            "optimizer_kwargs cannot override exp_trotter-owned options: "
            + ", ".join(sorted(overlap))
        )

    upper_ind_id = str(optimizer_kwargs.get("ind_id_k", "k{}"))
    lower_ind_id = str(optimizer_kwargs.get("ind_id_b", "b{}"))
    if basis.L == 1:
        mpo = qtn.MatrixProductOperator(
            [np.eye(basis.phys_dim, dtype="complex128")],
            shape="ud",
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
        )
    else:
        mpo = qtn.MPO_identity(
            basis.L,
            phys_dim=basis.phys_dim,
            dtype="complex128",
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
        )
    _apply_to_backend(mpo, to_backend)

    optimizer = MpoOptimizer(
        mpo,
        gates=gate_entries,
        chi=chi,
        mode=mode,
        contraction_opt=contraction_opt,
        **optimizer_kwargs,
    )
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")
    else:
        run_kwargs = dict(run_kwargs)
    reserved_run = {
        "n_iter",
        "mode",
        "progbar",
        "cutoff",
        "cutoff_mode",
        "fidelity_samples",
    }
    overlap = reserved_run.intersection(run_kwargs)
    if overlap:
        raise TypeError(
            "run_kwargs cannot override exp_trotter-owned options: "
            + ", ".join(sorted(overlap))
        )
    run_kwargs.update(
        {
            "progbar": progress,
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
            "fidelity_samples": fidelity_samples,
        }
    )
    output = optimizer.run(n_iter=n_iter, **run_kwargs)
    final_bond_dimensions = tuple(int(size) for size in output.bond_sizes())
    report = TrotterMPOReport(
        order=order,
        steps=steps,
        ordering=scheduler_ordering,
        layers=tuple(layers),
        gate_count=len(gate_entries),
        isolated_sites=tuple(sorted(isolated)),
        mode=optimizer.mode,
        chi=chi,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        final_bond_dimensions=final_bond_dimensions,
        optimizer_timing=optimizer.last_run_timing,
    )
    metadata = {
        "operation": "exp_trotter",
        "order": order,
        "steps": steps,
        "ordering": scheduler_ordering,
        "layers": tuple(layers),
        "gate_count": len(gate_entries),
        "isolated_sites": tuple(sorted(isolated)),
        "mode": optimizer.mode,
        "chi": chi,
        "cutoff": cutoff,
        "cutoff_mode": cutoff_mode,
        "report": report,
        "optimizer_timing": optimizer.last_run_timing,
    }
    if isinstance(getattr(output, "metadata", None), dict):
        output.metadata.update(metadata)
    output.pepsy_trotter_report = report
    output.pepsy_trotter_metadata = metadata
    output.pepsy_trotter_gates = tuple(native_gates)
    output.pepsy_trotter_optimizer = optimizer
    _apply_to_backend(output, to_backend)
    return (output, report) if return_report else output
