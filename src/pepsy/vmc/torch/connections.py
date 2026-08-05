"""Operator-to-connection compilation for Torch VMC.

This module owns the static translation from dense or native graded operator
terms to batched configuration transitions. The driver and local-energy code
consume :class:`TorchConnections` without needing to know how an operator was
represented.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import autoray as ar
import numpy as np

from ..torch_types import _require_torch
from ._common import _as_long_matrix


@dataclass(frozen=True)
class TorchConnections:
    """Batched Hamiltonian connections.

    ``configs[k]`` is connected to source sample ``batch_ids[k]`` with
    coefficient ``coeffs[k]``.
    """

    configs: Any
    coeffs: Any
    batch_ids: Any

    def to(self, device):
        """Return this connection table on ``device``."""
        return TorchConnections(
            configs=self.configs.to(device=device),
            coeffs=self.coeffs.to(device=device),
            batch_ids=self.batch_ids.to(device=device),
        )

    def slice(self, start, stop):
        """Slice rows belonging to parent samples ``[start, stop)``."""
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start:
            raise ValueError("connection slice must satisfy 0 <= start <= stop")
        mask = (self.batch_ids >= start) & (self.batch_ids < stop)
        return TorchConnections(
            configs=self.configs[mask],
            coeffs=self.coeffs[mask],
            batch_ids=self.batch_ids[mask] - start,
        )


@dataclass(frozen=True)
class TorchFockTransitionPlan:
    """Reusable parent-to-connected-configuration tables.

    A plan is tied to one ordered parent configuration stream and one named
    observable map. It contains no PEPS parameters, so the same plan can be
    reused for every PEPS bond dimension or dtype that measures the same
    stored proposal stream. :meth:`slice` makes it safe to process a persisted
    full-stream plan in sequential chunks.
    """

    configs: Any
    connection_map: Mapping[str, TorchConnections]

    def __post_init__(self):
        configs = _as_long_matrix(self.configs)
        if not isinstance(self.connection_map, Mapping):
            raise TypeError("connection_map must be a mapping of observable names.")
        normalized = {}
        for name, connections in self.connection_map.items():
            if not isinstance(name, str):
                raise TypeError("transition-plan observable names must be strings.")
            if not isinstance(connections, TorchConnections):
                raise TypeError("connection_map values must be TorchConnections.")
            normalized[name] = connections
        object.__setattr__(self, "configs", configs)
        object.__setattr__(self, "connection_map", normalized)

    @property
    def observable_names(self):
        return tuple(self.connection_map)

    @property
    def n_samples(self):
        return int(self.configs.shape[0])

    @property
    def n_connections(self):
        return sum(
            int(connections.configs.shape[0])
            for connections in self.connection_map.values()
        )

    def to(self, device):
        """Return the plan on ``device`` without changing its parent order."""
        return TorchFockTransitionPlan(
            configs=self.configs.to(device=device),
            connection_map={
                name: connections.to(device)
                for name, connections in self.connection_map.items()
            },
        )

    def slice(self, start, stop):
        """Return the plan for parent samples ``[start, stop)``."""
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.n_samples:
            raise ValueError(
                f"invalid transition-plan slice [{start}, {stop}) for "
                f"{self.n_samples} samples"
            )
        return TorchFockTransitionPlan(
            configs=self.configs[start:stop],
            connection_map={
                name: connections.slice(start, stop)
                for name, connections in self.connection_map.items()
            },
        )


def _term_items(terms):
    """Return ``(where, operator)`` pairs from common Hamiltonian containers."""
    if hasattr(terms, "terms"):
        terms = terms.terms
    if hasattr(terms, "items"):
        return tuple(terms.items())
    try:
        return tuple(terms)
    except TypeError as exc:
        raise TypeError(
            "terms must be a mapping, a SymHamiltonian-like object with "
            "`.terms`, or an iterable of (where, operator) pairs."
        ) from exc


def _term_dense_array(operator):
    """Convert a local operator to a dense array without changing its backend."""
    if hasattr(operator, "to_dense"):
        operator = operator.to_dense()
    elif hasattr(operator, "data") and not hasattr(operator, "shape"):
        operator = operator.data
    return operator


def _is_fermionic_operator(operator):
    """Return whether ``operator`` carries Symmray fermionic grading."""
    return bool(getattr(operator, "fermionic", False)) or (
        "FermionicArray" in type(operator).__name__
    )


def _expanded_operator_charges(index, *, symmetry=None):
    """Expand a Symmray index charge map into linear-index order."""
    chargemap = getattr(index, "chargemap", None)
    if chargemap is not None:
        # A spinful Z2 index has two states in each sector. Block sizes alone
        # cannot recover the canonical ``empty, down, up, double`` ordering.
        if str(symmetry) == "Z2" and sum(int(size) for size in chargemap.values()) == 4:
            import symmray.fermionic_local_operators as flo  # noqa: PLC0415

            return tuple(flo.get_spinful_charge_indexmap("Z2"))
        charges = []
        for charge, size in chargemap.items():
            charges.extend([charge] * int(size))
        return tuple(charges)

    linearmap = getattr(index, "_linearmap", None)
    if linearmap is not None:
        return tuple(entry[0] for entry in linearmap)

    size = getattr(index, "size_total", None)
    if size is None or symmetry is None:
        raise TypeError(
            "Fermionic VMC terms require Symmray indices with charge maps "
            "or recognizable flat fermion physical indices."
        )
    import symmray.fermionic_local_operators as flo  # noqa: PLC0415

    if int(size) == 2:
        charges = flo.get_spinless_charge_indexmap(str(symmetry))
    elif int(size) == 4:
        charges = flo.get_spinful_charge_indexmap(str(symmetry))
    else:
        raise TypeError(
            "Cannot infer fermionic charges for a flat physical index of "
            f"dimension {size}; provide a sparse Symmray index instead."
        )
    return tuple(charges)


def _charge_parity(charge):
    """Return the fermion parity of an Abelian charge."""
    if isinstance(charge, tuple):
        return sum(int(value) for value in charge) % 2
    return int(charge) % 2


def _operator_dense_numpy(operator):
    """Get a detached CPU view of a fixed native operator tensor."""
    dense = _term_dense_array(operator)
    return np.asarray(ar.to_numpy(dense))


@dataclass(frozen=True)
class _CompiledFermionicTerm:
    """Static sparse data for one native fermionic operator term."""

    operator: Any
    rank: int
    local_sites: tuple[int, ...]
    local_dims: tuple[int, ...]
    transitions: tuple[tuple[Any, ...], ...]
    input_parity: tuple[np.ndarray, ...]
    between: tuple[int, ...]
    parity_sites: tuple[int, ...]


# Native operators are fixed observables during VMC optimization. Keep a
# strong reference in each value so an id-based key cannot become stale.
_FERMION_COMPILED_TERM_CACHE = {}
_FERMION_COMPILED_TERM_CACHE_MAXSIZE = 1024


def _fermionic_operator_shape(operator):
    """Return a native operator shape without materializing dense data."""
    shape = getattr(operator, "shape", None)
    if shape is None:
        indices = getattr(operator, "indices", None)
        if indices is None:
            return tuple(_operator_dense_numpy(operator).shape)
        shape = tuple(getattr(index, "size_total", 0) for index in indices)
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError("Fermionic operator shape must be a finite sequence.") from exc


def _compile_fermionic_operator(operator, local_sites, *, coefficient_cutoff):
    """Compile a native fermionic operator into configuration transitions."""
    shape = _fermionic_operator_shape(operator)
    rank = len(shape)
    if rank not in (2, 4):
        raise ValueError(
            f"Hamiltonian operators must have rank 2 or 4, got rank {rank}."
        )
    split = rank // 2
    if shape[:split] != shape[split:]:
        raise ValueError(
            "Fermionic Hamiltonian terms must have square input/output "
            f"dimensions, got shape {shape}."
        )

    indices = getattr(operator, "indices", None)
    if indices is None or len(indices) != rank:
        raise TypeError(
            "Fermionic VMC terms require native Symmray operator indices."
        )
    symmetry = getattr(operator, "symmetry", None)
    output_charges = tuple(
        _expanded_operator_charges(index, symmetry=symmetry)
        for index in indices[:split]
    )
    input_charges = tuple(
        _expanded_operator_charges(index, symmetry=symmetry)
        for index in indices[split:]
    )
    if any(len(values) != shape[axis] for axis, values in enumerate(output_charges)):
        raise ValueError("Fermionic operator output charge maps do not match its shape.")
    if any(
        len(values) != shape[split + axis]
        for axis, values in enumerate(input_charges)
    ):
        raise ValueError("Fermionic operator input charge maps do not match its shape.")

    dense = _operator_dense_numpy(operator)
    if dense.shape != shape:
        raise ValueError(
            "Fermionic operator dense data shape does not match its native "
            f"shape: {dense.shape} != {shape}."
        )

    if rank == 4 and local_sites[0] > local_sites[1]:
        dense = dense.transpose(1, 0, 3, 2)
        local_sites = (local_sites[1], local_sites[0])
        output_charges = (output_charges[1], output_charges[0])
        input_charges = (input_charges[1], input_charges[0])

    transitions = []
    if rank == 2:
        nonzero = np.argwhere(np.abs(dense) > float(coefficient_cutoff))
        output_parity = np.asarray(
            [_charge_parity(charge) for charge in output_charges[0]],
            dtype=np.int64,
        )
        input_parity = np.asarray(
            [_charge_parity(charge) for charge in input_charges[0]],
            dtype=np.int64,
        )
        for output, input_ in nonzero:
            output = int(output)
            input_ = int(input_)
            transitions.append(
                (
                    output,
                    input_,
                    dense[output, input_],
                    int(output_parity[output] ^ input_parity[input_]),
                )
            )
        return _CompiledFermionicTerm(
            operator=operator,
            rank=rank,
            local_sites=tuple(local_sites),
            local_dims=(shape[0],),
            transitions=tuple(transitions),
            input_parity=(input_parity,),
            between=(),
            parity_sites=(
                tuple(range(local_sites[0]))
                if any(transition[3] for transition in transitions)
                else ()
            ),
        )

    input_parity = tuple(
        np.asarray(
            [_charge_parity(charge) for charge in charges],
            dtype=np.int64,
        )
        for charges in input_charges
    )
    crossing = np.ones((shape[2], shape[3]), dtype=np.int8)
    crossing[np.ix_(input_parity[0].astype(bool), input_parity[1].astype(bool))] = -1
    dense = dense * crossing[None, None, :, :]

    output_left_parity = np.asarray(
        [_charge_parity(charge) for charge in output_charges[0]],
        dtype=np.int64,
    )
    input_left_parity = input_parity[0]
    nonzero = np.argwhere(np.abs(dense) > float(coefficient_cutoff))
    for output_left, output_right, input_left, input_right in nonzero:
        output_left = int(output_left)
        output_right = int(output_right)
        input_left = int(input_left)
        input_right = int(input_right)
        transitions.append(
            (
                output_left,
                output_right,
                input_left,
                input_right,
                dense[output_left, output_right, input_left, input_right],
                int(output_left_parity[output_left] ^ input_left_parity[input_left]),
            )
        )

    left, right = local_sites
    return _CompiledFermionicTerm(
        operator=operator,
        rank=rank,
        local_sites=tuple(local_sites),
        local_dims=(shape[0], shape[1]),
        transitions=tuple(transitions),
        input_parity=input_parity,
        between=tuple(range(left + 1, right)),
        parity_sites=tuple(range(left + 1, right)),
    )


def _term_site_indices(where, rank, *, site_order, n_sites):
    """Resolve a one- or two-site term location to config-column indices."""
    n_local_sites = rank // 2
    if n_local_sites not in (1, 2):
        raise ValueError(
            "Hamiltonian operators must have rank 2 or 4, with output axes "
            "followed by input axes."
        )
    if n_local_sites == 1:
        where = (where,)
    elif isinstance(where, (str, bytes)):
        raise ValueError("A two-site operator location must contain two sites.")
    else:
        try:
            where = tuple(where)
        except TypeError as exc:
            raise ValueError(
                "A two-site operator location must contain two sites."
            ) from exc
        if len(where) != 2:
            raise ValueError("A two-site operator location must contain two sites.")

    if site_order is None:
        site_order = tuple(range(n_sites))
    position = {site: i for i, site in enumerate(site_order)}
    missing = [site for site in where if site not in position]
    if missing:
        raise ValueError(
            f"Hamiltonian term site(s) {missing!r} are not in site_order. "
            "Pass site_order matching the PEPS physical-site order."
        )
    return tuple(position[site] for site in where)


def _get_compiled_fermionic_operator(
    operator,
    where,
    *,
    site_order,
    n_sites,
    coefficient_cutoff,
):
    """Get or create the static compilation for one native term."""
    rank = len(_fermionic_operator_shape(operator))
    raw_sites = _term_site_indices(
        where,
        rank,
        site_order=site_order,
        n_sites=n_sites,
    )
    if any(site < 0 or site >= n_sites for site in raw_sites):
        raise ValueError(
            f"Hamiltonian term at {where!r} resolves outside the supplied "
            f"configuration width {n_sites}."
        )
    if rank == 4 and raw_sites[0] == raw_sites[1]:
        raise ValueError(
            "A native two-site fermionic operator must act on two distinct "
            "configuration sites."
        )
    key = (id(operator), tuple(raw_sites), int(n_sites), float(coefficient_cutoff))
    compiled = _FERMION_COMPILED_TERM_CACHE.get(key)
    if compiled is not None and compiled.operator is operator:
        return compiled

    compiled = _compile_fermionic_operator(
        operator,
        raw_sites,
        coefficient_cutoff=coefficient_cutoff,
    )
    if len(_FERMION_COMPILED_TERM_CACHE) >= _FERMION_COMPILED_TERM_CACHE_MAXSIZE:
        _FERMION_COMPILED_TERM_CACHE.pop(next(iter(_FERMION_COMPILED_TERM_CACHE)))
    _FERMION_COMPILED_TERM_CACHE[key] = compiled
    return compiled


def _empty_connections(configs):
    """Return an empty connection table matching ``configs``."""
    torch = _require_torch()
    return TorchConnections(
        configs=configs.new_empty((0, configs.shape[1])),
        coeffs=torch.empty(0, dtype=torch.float64, device=configs.device),
        batch_ids=torch.empty(0, dtype=torch.long, device=configs.device),
    )


def _fermionic_operator_connections(
    configs,
    where,
    operator,
    *,
    site_order,
    coefficient_cutoff=0.0,
):
    """Build connections for one native graded fermionic operator."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    compiled = _get_compiled_fermionic_operator(
        operator,
        where,
        site_order=site_order,
        n_sites=configs.shape[1],
        coefficient_cutoff=coefficient_cutoff,
    )
    rank = compiled.rank
    local_sites = compiled.local_sites
    if configs.numel() and int(configs.min()) < 0:
        raise ValueError("Fermionic configurations must use non-negative local codes.")
    if configs.numel() and any(
        int(configs[:, site].max()) >= local_dim
        for site, local_dim in zip(local_sites, compiled.local_dims)
    ):
        raise ValueError(
            "Fermionic Hamiltonian term has a local dimension too small for "
            "the supplied configurations."
        )
    all_etas = []
    all_coeffs = []
    all_bids = []

    if rank == 2:
        site = local_sites[0]
        if compiled.parity_sites:
            config_parity = torch.as_tensor(
                compiled.input_parity[0],
                dtype=torch.long,
                device=configs.device,
            )
            prefix_parity = (
                config_parity[configs[:, compiled.parity_sites]].sum(dim=-1) % 2
            )
        else:
            prefix_parity = None
        for output, input_, coefficient, transfer_parity in compiled.transitions:
            mask = configs[:, site] == input_
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            eta[:, site] = output
            all_etas.append(eta)
            coefficient = torch.as_tensor(coefficient, device=configs.device)
            if prefix_parity is not None and transfer_parity:
                phase = torch.where(
                    prefix_parity[batch_ids] == 1,
                    torch.as_tensor(-1.0, device=configs.device),
                    torch.as_tensor(1.0, device=configs.device),
                )
                coefficient = coefficient * phase
            all_coeffs.append(coefficient.expand(batch_ids.numel()))
            all_bids.append(batch_ids)
    else:
        left, right = local_sites
        if compiled.between:
            config_parity = torch.as_tensor(
                compiled.input_parity[0],
                dtype=torch.long,
                device=configs.device,
            )
            between_parity = config_parity[configs[:, compiled.between]].sum(dim=-1) % 2
        else:
            between_parity = None

        for (
            output_left,
            output_right,
            input_left,
            input_right,
            coefficient,
            transfer_parity,
        ) in compiled.transitions:
            mask = (configs[:, left] == input_left) & (
                configs[:, right] == input_right
            )
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            eta[:, left] = output_left
            eta[:, right] = output_right
            if between_parity is None:
                phase = 1.0
            elif transfer_parity:
                phase = torch.where(
                    between_parity[batch_ids] == 1,
                    torch.as_tensor(-1.0, device=configs.device),
                    torch.as_tensor(1.0, device=configs.device),
                )
            else:
                phase = 1.0
            coefficient = torch.as_tensor(coefficient, device=configs.device)
            if not isinstance(phase, float):
                coefficient = coefficient * phase
            all_etas.append(eta)
            all_coeffs.append(coefficient.expand(batch_ids.numel()))
            all_bids.append(batch_ids)

    if not all_etas:
        return _empty_connections(configs)
    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def torch_hamiltonian_connections(
    configs,
    terms,
    *,
    site_order=None,
    coefficient_cutoff=0.0,
    constant=0.0,
):
    """Build connected configurations from explicit local Hamiltonian terms."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    all_etas = []
    all_coeffs = []
    all_bids = []

    for where, operator in _term_items(terms):
        if _is_fermionic_operator(operator):
            term_connections = _fermionic_operator_connections(
                configs,
                where,
                operator,
                site_order=site_order,
                coefficient_cutoff=coefficient_cutoff,
            )
            if term_connections.configs.shape[0]:
                all_etas.append(term_connections.configs)
                all_coeffs.append(term_connections.coeffs)
                all_bids.append(term_connections.batch_ids)
            continue

        dense = torch.as_tensor(_term_dense_array(operator), device=device)
        if dense.ndim not in (2, 4):
            raise ValueError(
                f"Hamiltonian term at {where!r} has rank {dense.ndim}; "
                "only one- and two-site terms are supported."
            )
        if dense.shape[: dense.ndim // 2] != dense.shape[dense.ndim // 2 :]:
            raise ValueError(
                f"Hamiltonian term at {where!r} must have square input/output "
                f"dimensions, got shape {tuple(dense.shape)}."
            )
        local_sites = _term_site_indices(
            where,
            dense.ndim,
            site_order=site_order,
            n_sites=n_sites,
        )
        local_dim = int(dense.shape[0])
        if any(int(configs[:, site].max()) >= local_dim for site in local_sites):
            raise ValueError(
                f"Hamiltonian term at {where!r} has local dimension {local_dim}, "
                "which is too small for the supplied configurations."
            )

        nonzero = torch.nonzero(
            dense.abs() > float(coefficient_cutoff),
            as_tuple=False,
        )
        n_local_sites = len(local_sites)
        for entry in nonzero:
            entry = tuple(int(x) for x in entry.tolist())
            outputs = entry[:n_local_sites]
            inputs = entry[n_local_sites:]
            mask = torch.ones(batch, dtype=torch.bool, device=device)
            for site, value in zip(local_sites, inputs):
                mask &= configs[:, site] == value
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            for site, value in zip(local_sites, outputs):
                eta[:, site] = value
            all_etas.append(eta)
            all_coeffs.append(
                dense[entry].expand(batch_ids.numel()).to(device=device)
            )
            all_bids.append(batch_ids)

    if constant:
        identity = configs.clone()
        all_etas.append(identity)
        all_coeffs.append(torch.as_tensor(constant, device=device).expand(batch))
        all_bids.append(torch.arange(batch, device=device, dtype=torch.long))
    if not all_etas:
        return _empty_connections(configs)
    coefficient_dtype = all_coeffs[0].dtype
    for values in all_coeffs[1:]:
        coefficient_dtype = torch.promote_types(coefficient_dtype, values.dtype)
    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(
            [values.to(dtype=coefficient_dtype) for values in all_coeffs],
            dim=0,
        ),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def _driver_terms_connections(configs, graph, *, terms, site_order=None, **kwargs):
    """Adapt explicit-term connections to the driver's connection signature."""
    del graph
    return torch_hamiltonian_connections(
        configs,
        terms,
        site_order=site_order,
        constant=kwargs.get("constant", 0.0),
    )


def compile_operator_sum_torch(terms, *, fermion=None, site_order=None):
    """Compile a backend-neutral operator sum for Torch connections."""
    from ..api import (
        CompiledOperatorSum,
        LocalMatrixTerm,
        ProductTerm,
        _expand_fermion_factor,
        normalize_operator_sum,
    )

    operator_sum = normalize_operator_sum(terms)
    site_order = None if site_order is None else tuple(site_order)

    def map_site(site):
        if site_order is None:
            return site
        if site in site_order:
            return site
        if isinstance(site, Integral) and 0 <= int(site) < len(site_order):
            return site_order[int(site)]
        raise ValueError(f"Term site {site!r} is not present in site_order.")

    def native_fermion_name(name):
        return {
            "create_u": "create_d",
            "annihilate_u": "annihilate_d",
            "create_d": "create_u",
            "annihilate_d": "annihilate_u",
        }.get(name, name)

    compiled = []
    for term in operator_sum:
        if isinstance(term, LocalMatrixTerm):
            operator = term.matrix
            if term.coefficient != 1:
                operator = operator * term.coefficient
            mapped_support = tuple(map_site(site) for site in term.support)
            where = mapped_support[0] if len(mapped_support) == 1 else mapped_support
            compiled.append((where, operator))
            continue
        if not isinstance(term, ProductTerm):  # pragma: no cover - guarded by IR
            raise TypeError(f"Unsupported operator term {type(term).__name__}.")
        if fermion is None:
            raise ValueError(
                "Symbolic ProductTerm entries require fermion=... when compiling "
                "for Torch."
            )
        references = []
        for factor in term.factors:
            references.extend(
                (map_site(site), native_fermion_name(name))
                for site, name in _expand_fermion_factor(factor)
            )
        mapped_support = tuple(dict.fromkeys(site for site, _ in references))
        if site_order is not None:
            positions = {site: position for position, site in enumerate(site_order)}
            mapped_support = tuple(
                sorted(mapped_support, key=lambda site: positions[site])
            )
        elif all(isinstance(site, Integral) for site in mapped_support):
            mapped_support = tuple(sorted(mapped_support))
        operator = fermion.operator_term(
            [(term.coefficient, tuple(references))],
            sites=mapped_support,
        )
        where = mapped_support[0] if len(mapped_support) == 1 else mapped_support
        compiled.append((where, operator))
    return CompiledOperatorSum(
        backend="torch",
        terms=tuple(compiled),
        constant=operator_sum.constant,
        metadata=operator_sum.metadata,
    )


def _normalize_terms_site_labels(terms, site_order):
    """Map positional integer term labels onto PEPS site labels when needed."""
    site_order = tuple(site_order)
    positions = {site: i for i, site in enumerate(site_order)}

    def map_site(site):
        if site in positions:
            return site
        if (
            isinstance(site, Integral)
            and not isinstance(site, bool)
            and 0 <= int(site) < len(site_order)
        ):
            return site_order[int(site)]
        return site

    normalized = {}
    for where, operator in _term_items(terms):
        dense = _term_dense_array(operator)
        rank = getattr(dense, "ndim", None)
        if rank is None:
            rank = len(getattr(dense, "shape", ()))
        n_local_sites = int(rank) // 2
        if n_local_sites == 1:
            normalized[map_site(where)] = operator
        elif n_local_sites == 2:
            try:
                left, right = tuple(where)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "A two-site observable location must contain two sites."
                ) from exc
            normalized[(map_site(left), map_site(right))] = operator
        else:
            normalized[where] = operator
    return normalized


__all__ = [
    "TorchConnections",
    "TorchFockTransitionPlan",
    "compile_operator_sum_torch",
    "torch_hamiltonian_connections",
    "_driver_terms_connections",
]
