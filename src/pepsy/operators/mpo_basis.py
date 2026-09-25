"""Parameterized higher-order MPO bases and compiled evaluators.

This module owns coefficient-slot topology and repeated value-only
evaluation. The semantic history implementation lives in
:mod:`operators.mpo_semantic`, while this module provides the basis/compiled
lifecycle boundary.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

import autoray as ar
import numpy as np

from .mpo_automaton import (
    MPOAutomaton,
    _as_backend,
    _backend_name,
    _backend_reference,
    _multiply_scalar,
)

__all__ = ["CompiledMPOExp", "CompiledMPOEvolution", "MPOBasis", "exp_mpo"]


def _convert_term_to_backend(term, to_backend):
    """Convert only operator payloads, leaving coefficient graphs untouched."""
    if isinstance(term, MPOProductTerm):
        return replace(
            term,
            operators=tuple(to_backend(operator) for operator in term.operators),
            string_operators=(
                None
                if term.string_operators is None
                else tuple(to_backend(operator) for operator in term.string_operators)
            ),
        )
    if isinstance(term, MPOLocalOperatorTerm):
        return replace(term, operator=to_backend(term.operator))
    return term


def _convert_automaton_to_backend(automaton, to_backend):
    """Convert all automaton transition blocks to the requested backend."""
    transitions = tuple(
        tuple(
            type(transition)(
                transition.left_state,
                transition.right_state,
                to_backend(transition.operator),
            )
            for transition in site_transitions
        )
        for site_transitions in automaton.transitions
    )
    return MPOAutomaton(
        automaton.L,
        channels=automaton.channels,
        transitions=transitions,
        start_state=automaton.start_state,
        done_state=automaton.done_state,
        phys_dim=automaton.phys_dim,
    )


def _apply_to_backend(tn, to_backend):
    """Apply a converter to host arrays without detaching backend arrays."""
    if to_backend is None or not hasattr(tn, "apply_to_arrays"):
        return tn

    def convert(array):
        if _backend_name(array) in {"builtins", "numpy"}:
            return to_backend(array)
        return array

    # Quimb can expose read-only NumPy views for some boundary tensors.
    # Make those writable before its in-place ``apply_to_arrays`` traversal.
    for tensor in tn:
        data = tensor.data
        if (
            isinstance(data, np.ndarray)
            and not data.flags.writeable
            and _backend_name(data) in {"builtins", "numpy"}
        ):
            tensor.modify(data=np.array(data, copy=True))
    tn.apply_to_arrays(convert)
    return tn


def _canonical_history_storage(history_storage):
    """Normalize the compatibility spelling for persistent block storage."""
    if history_storage == "blocks":
        return "block_sparse"
    return history_storage


class CompiledMPOExp:
    """Value-only higher-order exponential evaluator for an :class:`MPOBasis`.

    The object owns the structural pieces of a parameterized exponential step:
    the unit-coefficient MPO tensors, coefficient-slot indices, static local
    operator banks, and the history execution plans. Calls assemble fresh
    backend tensors and then execute the same Algorithms 1--4 as
    :meth:`FirstDegreeMPO.extensive_exponential`, so coefficients and the
    exponential step remain in the current Torch/JAX autodiff graph.

    ``CompiledMPOExp`` is intentionally a numerical boundary.  Its
    primary methods return raw ``(left, right, up, down)`` tensor tuples;
    :meth:`evaluate` is available when a semantic :class:`FirstDegreeMPO`
    wrapper is needed outside a compiled optimizer loop.
    """

    _MODE_ALIASES = {
        "base": (False, False, "base"),
        "exact": (True, False, "exact"),
        "folded": (False, True, "folded"),
        "hybrid": (True, True, "hybrid"),
        "auto": (False, False, "auto"),
        "algorithm4": (False, True, "folded"),
        "paper_algorithm4": (False, True, "folded"),
        "optimal": (True, False, "exact"),
        "paper_optimal": (True, False, "exact"),
        "approximate": (True, True, "hybrid"),
        "paper_approximate": (True, True, "hybrid"),
    }

    def __init__(
        self,
        basis,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        history_storage="auto",
        extension_budget=None,
    ):
        if not isinstance(basis, MPOBasis):
            raise TypeError("basis must be an MPOBasis.")
        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError("mode must be a string or None.")
            if mode == "auto":
                if extend or approximate:
                    raise ValueError(
                        "mode='auto' cannot be combined with extend or "
                        "approximate flags."
                    )
                canonical_mode = "auto"
            else:
                try:
                    mode_extend, mode_approximate, canonical_mode = (
                        self._MODE_ALIASES[mode]
                    )
                except KeyError as exc:
                    allowed = ", ".join(
                        ["base", "exact", "folded", "hybrid", "auto"]
                        + sorted(
                            name for name in self._MODE_ALIASES
                            if name not in {"base", "exact", "folded", "hybrid", "auto"}
                        )
                    )
                    raise ValueError(
                        f"unknown mode {mode!r}; expected one of {allowed}."
                    ) from exc
                if extend or approximate:
                    raise ValueError(
                        "mode cannot be combined with extend or approximate flags."
                    )
                extend = mode_extend
                approximate = mode_approximate
        else:
            canonical_mode = (
                "hybrid" if approximate and extend
                else "exact" if extend
                else "folded" if approximate
                else "base"
            )

        history_storage = _canonical_history_storage(history_storage)
        if history_storage == "streaming":
            raise ValueError(
                "compiled evolution requires cached history; use "
                "history_storage='auto', 'sparse', 'block_sparse', or "
                "'reduced'."
            )

        self.basis = basis
        self.order = order
        self.mode = canonical_mode
        self.extend = bool(extend)
        self.approximate = bool(approximate)
        self.max_bond = max_bond
        self.on_exceed = on_exceed
        self.history_storage = history_storage
        self.extension_budget = extension_budget

        # This validates the complete option set and fills every symbolic
        # history/tensor plan once.  The unit-coefficient numerical result is
        # discarded; only its structural caches are retained.
        basis._template.extensive_exponential(  # pylint: disable=protected-access
            0.0,
            order=order,
            mode=canonical_mode,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=True,
            history_storage=history_storage,
            extension_budget=extension_budget,
        )

        self._base_arrays = tuple(basis._template.arrays)  # pylint: disable=protected-access
        self._site_records = self._compile_slot_records(basis)
        self._fused_site_banks = self._compile_fused_site_banks(basis)

    @staticmethod
    def _compile_slot_records(basis):
        """Resolve coefficient slots to dense virtual tensor positions."""
        automaton = basis._automaton  # pylint: disable=protected-access
        records_by_site = [[] for _ in range(basis.L)]

        def position(site, state, *, left):
            if basis.L == 1:
                return 0
            if left and site == 0:
                return 0
            if not left and site == basis.L - 1:
                return 0
            cut = site - 1 if left else site
            return {
                channel.state: index
                for index, channel in enumerate(automaton.channels[cut])
            }[state]

        for site, transition_index, contributions in basis._slot_groups:
            transition = automaton.transitions[site][transition_index]
            row = position(site, transition.left_state, left=True)
            column = position(site, transition.right_state, left=False)
            term_indices = np.asarray(
                [term_index for term_index, _operator in contributions],
                dtype=int,
            )
            operators = tuple(operator for _term_index, operator in contributions)
            if all(
                _backend_name(operator) in {"builtins", "numpy"}
                for operator in operators
            ):
                operator_bank = np.stack(
                    [np.asarray(operator) for operator in operators],
                    axis=0,
                )
            else:
                operator_bank = None
            records_by_site[site].append(
                (
                    term_indices,
                    row,
                    column,
                    operators,
                    transition.operator,
                    operator_bank,
                ),
            )

        compiled = []
        for records in records_by_site:
            if not records:
                compiled.append(None)
                continue
            compiled.append(tuple(records))
        return tuple(compiled)

    def _compile_fused_site_banks(self, basis):
        """Build optional dense affine coefficient banks per MPO site.

        A unit-coefficient template contains the structural rails and one
        copy of every slot edge.  Subtracting those unit edge blocks from a
        static bias leaves an affine representation

        ``local_tensor = bias + sum(term_coefficient * operator_bank[term])``.

        The bank turns coefficient assembly into one backend contraction per
        site.  It is used only for ordinary host-backed static operators and
        only below a memory bound; backend-native operator arrays and large
        sparse layouts use the grouped autodiff-safe fallback.
        """
        fused = []
        for base, records in zip(self._base_arrays, self._site_records):
            if records is None or _backend_name(base) not in {"builtins", "numpy"}:
                fused.append(None)
                continue
            if any(
                _backend_name(operator) not in {"builtins", "numpy"}
                for record in records
                for operator in (*record[3], record[4])
            ):
                fused.append(None)
                continue
            try:
                bank_elements = basis.num_terms * int(np.prod(base.shape))
                if bank_elements > _MAX_FUSED_SLOT_BANK_ELEMENTS:
                    fused.append(None)
                    continue
                operators = [
                    np.asarray(operator)
                    for record in records
                    for operator in record[3]
                ]
                unit_operators = [
                    np.asarray(record[4])
                    for record in records
                ]
                dtype = np.result_type(
                    np.asarray(base).dtype,
                    *(operator.dtype for operator in operators),
                    *(operator.dtype for operator in unit_operators),
                    np.float32,
                )
                bias = np.asarray(base, dtype=dtype).copy()
                bank = np.zeros(
                    (basis.num_terms, *tuple(int(size) for size in base.shape)),
                    dtype=dtype,
                )
                for record in records:
                    term_indices, row, column, operators, unit_operator, _bank = (
                        record
                    )
                    bias[row, column] -= np.asarray(unit_operator, dtype=dtype)
                    for term_index, operator in zip(term_indices, operators):
                        bank[int(term_index), row, column] += np.asarray(
                            operator,
                            dtype=dtype,
                        )
            except (TypeError, ValueError):
                fused.append(None)
                continue
            fused.append((bias, bank))
        return tuple(fused)

    @property
    def cache_info(self):
        """Return the shared structural cache diagnostics."""
        info = dict(self.basis.cache_info)
        info.update({
            "fused_slot_sites": sum(
                bank is not None for bank in self._fused_site_banks
            ),
            "fused_slot_bank_elements": sum(
                int(np.prod(bank[1].shape))
                for bank in self._fused_site_banks
                if bank is not None
            ),
        })
        return info

    def _assemble_arrays(self, dt, parameters, coefficients):
        coefficient_values = self.basis._coefficient_values(  # pylint: disable=protected-access
            parameters,
            coefficients,
        )
        coefficient_batch = _stack(coefficient_values, axis=0)
        reference = _backend_reference((dt, *coefficient_values, *self._base_arrays))
        step = _as_backend(dt, like=reference)
        arrays = []

        for base, records, fused in zip(
            self._base_arrays,
            self._site_records,
            self._fused_site_banks,
        ):
            if fused is not None:
                bias, bank = fused
                array = _as_backend(bias, like=reference)
                if _complex_dtype(getattr(step, "dtype", None)) and not _complex_dtype(
                    getattr(array, "dtype", None)
                ):
                    array = ar.do(
                        "multiply",
                        array,
                        _as_backend(1.0 + 0.0j, like=step),
                    )
                bank = _as_backend(bank, like=coefficient_batch)
                coefficients_backend, bank = _align_tensordot_dtypes(
                    coefficient_batch,
                    bank,
                )
                weighted = ar.do(
                    "tensordot",
                    coefficients_backend,
                    bank,
                    axes=([0], [0]),
                )
                array, weighted = _align_tensordot_dtypes(array, weighted)
                arrays.append(ar.do("add", array, weighted))
                continue
            array = _as_backend(base, like=reference)
            if _complex_dtype(getattr(step, "dtype", None)) and not _complex_dtype(
                getattr(array, "dtype", None)
            ):
                # Higher-order prefactors can be complex even when H and its
                # coefficients are real. Promote the fresh local view before
                # any history transfer so no imaginary component is lost.
                array = ar.do(
                    "multiply",
                    array,
                    _as_backend(1.0 + 0.0j, like=step),
                )
            if records is not None:
                corrections = []
                rows = []
                columns = []
                for (
                    term_indices,
                    row,
                    column,
                    operators,
                    unit_operator,
                    operator_bank,
                ) in records:
                    if operator_bank is not None:
                        indices = _as_backend(
                            term_indices,
                            like=coefficient_batch,
                        )
                        selected = coefficient_batch[indices]
                        operator_values = _as_backend(
                            operator_bank,
                            like=selected,
                        )
                        selected, operator_values = _align_tensordot_dtypes(
                            selected,
                            operator_values,
                        )
                        weighted = ar.do(
                            "tensordot",
                            selected,
                            operator_values,
                            axes=([0], [0]),
                        )
                    else:
                        weighted = None
                        for term_index, operator in zip(
                            term_indices,
                            operators,
                        ):
                            contribution = _multiply_scalar(
                                coefficient_values[term_index],
                                operator,
                            )
                            if weighted is None:
                                weighted = contribution
                                continue
                            reference = _backend_reference(
                                (weighted, contribution),
                            )
                            weighted = ar.do(
                                "add",
                                _as_backend(weighted, like=reference),
                                _as_backend(contribution, like=reference),
                            )
                    unit_operator = _as_backend(unit_operator, like=weighted)
                    weighted, unit_operator = _align_tensordot_dtypes(
                        weighted,
                        unit_operator,
                    )
                    corrections.append(
                        ar.do("subtract", weighted, unit_operator),
                    )
                    rows.append(row)
                    columns.append(column)
                values = _stack(corrections, axis=0)
                array, values = _align_tensordot_dtypes(array, values)
                delta = _zeros(array.shape, like=array)
                delta = _scatter_add_into_2d(
                    delta,
                    rows,
                    columns,
                    values,
                )
                array = ar.do("add", array, delta)
            arrays.append(array)
        return tuple(arrays)

    def evaluate_arrays(self, dt, parameters=None, *, coefficients=None):
        """Compatibility wrapper for :meth:`exp_arrays`."""
        return self.exp_arrays(dt, parameters, coefficients=coefficients)

    def exp_arrays(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
    ):
        """Evaluate ``exp(step * H)`` as fresh backend-native tensor tuples.

        ``dt`` is accepted as a compatibility keyword for ``step``. The
        returned tuple is suitable for a backend contraction kernel and does
        not create a semantic MPO wrapper.
        """
        step = _resolve_exp_step(step, dt)
        arrays = self._assemble_arrays(step, parameters, coefficients)
        bound = self.basis._template._bind_arrays(arrays)  # pylint: disable=protected-access
        return bound.extensive_exponential(
            step,
            order=self.order,
            mode=self.mode,
            max_bond=self.max_bond,
            on_exceed=self.on_exceed,
            cache_history=True,
            history_storage=self.history_storage,
            extension_budget=self.extension_budget,
        ).arrays

    def evaluate(self, dt, parameters=None, *, coefficients=None, **kwargs):
        """Compatibility wrapper for :meth:`exp`."""
        return self.exp(dt, parameters, coefficients=coefficients, **kwargs)

    def exp(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
        progress=False,
    ):
        """Evaluate ``exp(step * H)`` with optional final compression.

        Use this form when downstream code needs MPO metadata or methods such
        as ``to_mpo()``. Use :meth:`exp_arrays` when it only needs raw tensors.
        ``chi`` is applied after the higher-order construction; additional
        Quimb compression keywords are supplied with ``compress_opts``.
        """
        step = _resolve_exp_step(step, dt)
        arrays = self._assemble_arrays(step, parameters, coefficients)
        bound = self.basis._template._bind_arrays(arrays)  # pylint: disable=protected-access
        result = bound.exp(
            step,
            order=self.order,
            mode=self.mode,
            max_bond=self.max_bond,
            on_exceed=self.on_exceed,
            cache_history=True,
            history_storage=self.history_storage,
            extension_budget=self.extension_budget,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
            progress=progress,
        )
        if return_report:
            semantic_result, report = result
        else:
            semantic_result, report = result, None
        if isinstance(semantic_result, FirstDegreeMPO):
            semantic_result.metadata["compiled_exp"] = True
            # Retain the historical metadata key for callers that inspect it.
            semantic_result.metadata["compiled_evolution"] = True
        else:
            semantic_result.pepsy_exp_metadata["compiled_exp"] = True
            semantic_result.pepsy_exp_metadata["compiled_evolution"] = True
        return (semantic_result, report) if return_report else semantic_result

    def time_evolution_arrays(self, dt, parameters=None, *, coefficients=None):
        """Evaluate ``exp(-1j * dt * H)`` as backend-native tensors."""
        return self.evaluate_arrays(
            -1j * dt,
            parameters,
            coefficients=coefficients,
        )

    def time_evolution(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
        progress=False,
    ):
        """Evaluate real-time evolution and return a semantic MPO."""
        return self.exp(
            -1j * dt,
            parameters,
            coefficients=coefficients,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
            progress=progress,
        )

    __call__ = exp_arrays


# Compatibility name for callers of the original evolution-oriented API.
# Keep it as an exact alias so existing isinstance checks remain valid while
# ``CompiledMPOExp`` remains the canonical class name.
CompiledMPOEvolution = CompiledMPOExp


class MPOBasis:
    """Reusable coefficient basis for parameterized first-degree MPOs.

    ``MPOBasis`` separates the topology of a Hamiltonian from the scalar
    coefficients that are changed by an optimization loop.  It builds one
    shared automaton and records coefficient contributions for term-specific
    path transition slots.  :meth:`build` then only copies the small
    transition table and assembles those slots; it does not rebuild the term
    graph or infer channel topology.

    Coefficients can be ordinary scalars, :class:`MPOParameter` references,
    or callables receiving the parameter container.  The latter two forms are
    useful when the values are Torch or JAX scalars.  The output is a normal
    :class:`FirstDegreeMPO`, so it can immediately be passed to
    :meth:`FirstDegreeMPO.extensive_exponential`.

    The cache is deliberately topology-only.  Caching a completed MPO by a
    backend tensor's object identity would return stale values after an
    optimizer updates that tensor and could also retain an obsolete autodiff
    graph.  Every call to :meth:`build` therefore creates fresh local blocks
    while reusing the compiled channel layout.

    Parameters
    ----------
    L : int
        Number of sites.
    terms : iterable
        ``MPOProductTerm`` values or the same term forms accepted by
        :meth:`FirstDegreeMPO.from_local_terms`.
    phys_dim : int, optional
        Local physical dimension.  It is inferred from the first operator
        when omitted.
    symmetry : {"U1", "Z2", "U1U1", "Z2Z2"}, optional
        Abelian symmetry used for native Symmray MPO compilation.
    physical_charges : sequence or mapping, optional
        Charge of each local dense basis state. A mapping from charge to
        positive sector multiplicity is also accepted; its insertion order
        defines the dense basis sector order. Equal charges must form
        contiguous sectors.
    fermionic : bool, default=False
        Reserved for a future sign-preserving graded history backend. The
        current higher-order block-sparse compiler rejects ``True``.
    to_backend : callable, optional
        Array converter used for compiled local operator blocks and
        coefficient assembly, for example ``pepsy.backend_torch(...)`` or
        ``pepsy.backend_jax(...)``. This is applied before higher-order
        contractions so final Quimb compression uses the same backend.
    """

    def __init__(
        self,
        L,
        terms,
        *,
        phys_dim=None,
        symmetry=None,
        physical_charges=None,
        fermionic=False,
        physical_space=None,
        to_backend=None,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
    ):
        if not isinstance(L, Integral):
            raise TypeError("L must be an integer.")
        L = int(L)
        if L < 1:
            raise ValueError("L must be >= 1.")
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        if to_backend is not None and symmetry is not None:
            raise ValueError(
                "to_backend cannot be combined with symmetry; native Symmray "
                "MPO blocks currently require NumPy arrays."
            )
        terms = tuple(_term_from_input(term) for term in terms)
        if not terms:
            raise ValueError("terms must contain at least one product term.")
        if phys_dim is None:
            first_term = terms[0]
            if isinstance(first_term, MPOLocalOperatorTerm):
                phys_dim = first_term.phys_dim
            else:
                shape = tuple(getattr(first_term.operators[0], "shape", ()))
                if len(shape) != 2 or shape[0] != shape[1]:
                    raise ValueError("cannot infer phys_dim from the first operator.")
                phys_dim = int(shape[0])

        for term in terms:
            term_phys_dim = (
                term.phys_dim
                if isinstance(term, MPOLocalOperatorTerm)
                else int(term.operators[0].shape[0])
            )
            if term_phys_dim != int(phys_dim):
                raise ValueError(
                    f"term physical dimension {term_phys_dim} does not match "
                    f"phys_dim={phys_dim}."
                )

        # Compile the topology with unit coefficients once.  The shared
        # builder returns independent path coefficient slots, so common
        # prefixes and suffixes remain compressed without coupling terms'
        # autodiff values.
        unit_terms = tuple(replace(term, coefficient=1.0) for term in terms)
        has_local_operators = any(
            isinstance(term, MPOLocalOperatorTerm) for term in unit_terms
        )
        if has_local_operators:
            automaton, term_slots = _mixed_term_automaton(
                L,
                unit_terms,
                phys_dim=phys_dim,
                unit_coefficients=True,
            )
        else:
            automaton, slots = MPOAutomaton.from_product_terms(
                L,
                unit_terms,
                share_channels=True,
                return_slots=True,
                phys_dim=int(phys_dim),
            )
            term_slots = tuple(
                ((site, transition_index, self._local_operator(term, site)),)
                for term, (site, transition_index) in zip(terms, slots)
            )

        if to_backend is not None:
            # Keep structural sharing and fingerprinting on host arrays, then
            # move the completed automaton and all coefficient-slot operators
            # to the requested backend before any higher-order contractions.
            terms = tuple(
                _convert_term_to_backend(term, to_backend)
                for term in terms
            )
            automaton = _convert_automaton_to_backend(automaton, to_backend)
            term_slots = tuple(
                tuple(
                    (site, transition_index, to_backend(operator))
                    for site, transition_index, operator in slots
                )
                for slots in term_slots
            )

        self.L = L
        self.phys_dim = int(phys_dim)
        self._terms = terms
        self.to_backend = to_backend
        self._slots = tuple(
            tuple((site, transition_index) for site, transition_index, _ in slots)
            for slots in term_slots
        )
        slot_groups = {}
        for term_index, slots in enumerate(term_slots):
            for site, transition_index, operator in slots:
                slot_groups.setdefault((site, transition_index), []).append(
                    (term_index, operator)
                )
        self._slot_groups = tuple(
            (site, transition_index, tuple(contributions))
            for (site, transition_index), contributions in slot_groups.items()
        )
        self._vectorized_slot_groups = tuple(
            (
                site,
                transition_index,
                np.asarray([term_index for term_index, _ in contributions], dtype=int),
                np.stack(
                    [np.asarray(operator) for _, operator in contributions],
                    axis=0,
                ),
            )
            for site, transition_index, contributions in self._slot_groups
            if all(
                _backend_name(operator) in {"builtins", "numpy"}
                for _, operator in contributions
            )
        )
        self._vectorized_slot_keys = frozenset(
            (site, transition_index)
            for site, transition_index, _indices, _operators
            in self._vectorized_slot_groups
        )
        self._automaton = automaton
        self._template = FirstDegreeMPO.from_automaton(
            automaton,
            symmetry=symmetry,
            physical_charges=physical_charges,
            fermionic=fermionic,
            physical_space=physical_space,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
        )
        self._lattice_shape = None
        self._lattice_mapper = None
        self._lattice_to_chain = None
        self._chain_to_lattice = None
        self._location_mode = "chain"
        self._shape_inferred = False
        self._build_count = 0
        self._compiled_evolution_cache = {}
        self._cluster_expansion_cache = {}
        self._graph_cluster_expansion_cache = {}

    @classmethod
    def from_terms(
        cls,
        terms,
        *,
        shape=None,
        mapper=None,
        map_mode="snake",
        **kwargs,
    ):
        """Create a basis directly from operator/location/coefficient terms.

        This is the term-centric entry point for chain and regular-lattice
        inputs. A term may be written as
        ``{"operator": "ZZ", "location": (0, 1), "coefficient": value}``
        or with the existing plural ``operators``/``locations`` aliases.
        Tuple terms may use ``(location, paulis, coefficient)`` or
        ``((paulis, coefficient), location)`` in addition to the explicit
        ``(operators, locations[, coefficient])`` form.
        Pepsy's compact Pauli mapping is also accepted: ``{"XX": (2, 3)}``
        or ``{"XX": ((2, 3), coefficient)}``. A word key with a nested
        coordinate support follows the same convention, for example
        ``{"xyz": (((0, 0), (1, 0), (0, 1)), coefficient)}``.
        ``shape`` may be an integer chain length or a 2D/3D lattice shape. If
        it is omitted, the smallest shape containing all term locations is
        inferred. Integer locations are already-mapped chain positions and
        need no mapper. Coordinate locations are mapped with ``mapper`` or,
        when it is omitted, an internally constructed ``OneDMap`` using
        ``map_mode``. A single 1D site should be written as a bare integer;
        a tuple such as ``(x, y)`` is a coordinate when one local operator is
        supplied. Do not mix chain indices and coordinates in one call.
        Common supports are canonicalized before the shared MPO automaton is
        built, while each coefficient remains an independent slot for
        autodiff.
        """
        length, normalized_terms, metadata = _compile_generic_terms(
            terms,
            shape=shape,
            mapper=mapper,
            map_mode=map_mode,
        )
        basis = cls(length, normalized_terms, **kwargs)
        if len(metadata["shape"]) > 1:
            basis._lattice_shape = metadata["shape"]
            basis._lattice_mapper = metadata["mapper"]
            basis._lattice_to_chain = metadata["lattice_to_chain"]
            basis._chain_to_lattice = metadata["chain_to_lattice"]
        basis._location_mode = metadata["location_mode"]
        basis._shape_inferred = metadata["shape_inferred"]
        return basis

    @classmethod
    def from_local_terms(
        cls,
        L,
        terms,
        *,
        phys_dim=None,
        **kwargs,
    ):
        """Create a reusable basis from factorized local product terms."""
        return cls(L, terms, phys_dim=phys_dim, **kwargs)

    @classmethod
    def from_pauli_terms(
        cls,
        L,
        terms,
        **kwargs,
    ):
        """Create a reusable qubit basis from compact Pauli term labels."""
        return cls(L, terms, phys_dim=2, **kwargs)

    @classmethod
    def from_square_lattice(
        cls,
        lx,
        ly,
        terms,
        *,
        mapper=None,
        map_mode="snake",
        **kwargs,
    ):
        """Compile coordinate-based Pauli terms into a reusable MPO basis.

        ``terms`` accepts mappings such as
        ``{"locations": ((0, 0), (1, 0)), "paulis": "ZZ", ...}``, where
        ``...`` can contain ``coefficient`` or ``parameter``. Tuple forms may
        be written as either ``(locations, paulis[, coefficient])`` or
        ``(paulis, locations[, coefficient])``. Locations are mapped to a
        one-dimensional MPO chain using :class:`pepsy.tensors.OneDMap`.

        Reversed location/Pauli descriptions are canonicalized before the
        shared automaton is built, so equivalent terms reuse one structural
        channel while their backend-native coefficients remain differentiable.
        """
        from pepsy.tensors import OneDMap  # pylint: disable=import-outside-toplevel

        if not _is_integral_value(lx) or not _is_integral_value(ly):
            raise TypeError("lx and ly must be positive integer dimensions.")
        lx, ly = int(lx), int(ly)
        if mapper is None:
            mapper = OneDMap(lx, ly, mode=map_mode)
        elif not isinstance(mapper, OneDMap):
            raise TypeError("mapper must be a pepsy.tensors.OneDMap or None.")
        if mapper.shape != (lx, ly):
            raise ValueError(
                f"mapper shape {mapper.shape} does not match lattice shape "
                f"{(lx, ly)}."
            )
        chain_to_lattice, lattice_to_chain = mapper.build()
        normalized_terms = tuple(
            _square_lattice_pauli_term(term, lattice_to_chain)
            for term in terms
        )
        if not normalized_terms:
            raise ValueError("terms must contain at least one Pauli term.")

        basis = cls.from_pauli_terms(
            lx * ly,
            normalized_terms,
            **kwargs,
        )
        basis._lattice_shape = (lx, ly)
        basis._lattice_mapper = mapper
        basis._lattice_to_chain = dict(lattice_to_chain)
        basis._chain_to_lattice = dict(chain_to_lattice)
        basis._location_mode = "lattice"
        basis._shape_inferred = False
        return basis

    @property
    def terms(self):
        """Read-only term specifications, including coefficient references."""
        return self._terms

    @property
    def lattice_shape(self):
        """Return the compiled ``(lx, ly)`` shape, or ``None`` for chain input."""
        return self._lattice_shape

    @property
    def location_mode(self):
        """Return ``"chain"`` or ``"lattice"`` for the parsed locations."""
        return self._location_mode

    @property
    def lattice_to_chain(self):
        """Return a copy of the coordinate-to-MPO-site map, when configured."""
        return None if self._lattice_to_chain is None else dict(self._lattice_to_chain)

    @property
    def chain_to_lattice(self):
        """Return a copy of the MPO-site-to-coordinate map, when configured."""
        return None if self._chain_to_lattice is None else dict(self._chain_to_lattice)

    @property
    def num_terms(self):
        """Number of coefficient slots in the basis."""
        return len(self._terms)

    @property
    def template(self):
        """The unit-coefficient first-degree MPO defining this basis."""
        return self._template.copy()

    @property
    def bond_dimensions(self):
        """Internal bond dimensions of the cached common automaton."""
        return self._template.bond_dimensions

    @property
    def cache_info(self):
        """Small immutable cache diagnostic for optimization bookkeeping."""
        return {
            "compiled": True,
            "compiled_terms": self.num_terms,
            "builds": self._build_count,
            "compiled_exp_variants": len(self._compiled_evolution_cache),
            # Compatibility diagnostic name.
            "compiled_evolution_variants": len(self._compiled_evolution_cache),
            "compiled_cluster_variants": len(self._cluster_expansion_cache),
            "compiled_graph_cluster_variants": len(self._graph_cluster_expansion_cache),
            "topology_bond_dimensions": self.bond_dimensions,
            "vectorized_slot_groups": len(self._vectorized_slot_groups),
            "lattice_shape": self._lattice_shape,
            "location_mode": self._location_mode,
            "shape_inferred": getattr(self, "_shape_inferred", False),
            "lattice_mode": (
                None if self._lattice_mapper is None else self._lattice_mapper.mode
            ),
            "history": self._template.history_cache_info,
        }

    def clear_history_cache(self):
        """Release cached higher-order plans while retaining the basis graph."""
        self._template.clear_history_cache()
        self._compiled_evolution_cache.clear()
        return self

    def clear_cluster_expansion_cache(self):
        """Release cached cluster-expansion topologies while retaining the basis."""

        self._cluster_expansion_cache.clear()
        self._graph_cluster_expansion_cache.clear()
        return self

    def cluster_expansion(
        self,
        step=1.0,
        parameters=None,
        *,
        cluster_size=2,
        cutoff=1.0e-12,
        max_bond=None,
        symmetry=None,
        physical_charges=None,
        fermionic=False,
        physical_space=None,
    ):
        """Build a local exact cluster expansion from this term basis.

        This is the cluster-basis counterpart to :meth:`exp`: local
        exponentials are evaluated exactly on intervals through
        ``cluster_size`` and assembled as disjoint connected MPO channels.
        ``parameters`` resolves the same :class:`MPOParameter` references
        used by ``build``.  Use :class:`MPOClusterProductExpansion` directly
        when composing several ordered exponential factors.
        """
        from .mpo_product import (  # pylint: disable=import-outside-toplevel
            MPOClusterProductExpansion,
        )

        cache_key = (
            int(cluster_size),
            None if cutoff is None else float(cutoff),
            None if max_bond is None else int(max_bond),
            symmetry,
            repr(physical_charges),
            bool(fermionic),
            repr(physical_space),
        )
        expansion = self._cluster_expansion_cache.get(cache_key)
        if expansion is None:
            expansion = MPOClusterProductExpansion.from_mpo_basis(
                self,
                cluster_size=cluster_size,
                cutoff=cutoff,
                max_bond=max_bond,
                symmetry=symmetry,
                physical_charges=physical_charges,
                fermionic=fermionic,
                physical_space=physical_space,
            )
            self._cluster_expansion_cache[cache_key] = expansion
        return expansion.exp(step, parameters=parameters)

    def compile_cluster_expansion(
        self,
        *,
        cluster_size=2,
        cutoff=1.0e-12,
        max_bond=None,
        symmetry=None,
        physical_charges=None,
        fermionic=False,
        physical_space=None,
    ):
        """Return a reusable compiled evaluator for local cluster products."""

        from .mpo_product import (  # pylint: disable=import-outside-toplevel
            MPOClusterProductExpansion,
        )

        cache_key = (
            int(cluster_size),
            None if cutoff is None else float(cutoff),
            None if max_bond is None else int(max_bond),
            symmetry,
            repr(physical_charges),
            bool(fermionic),
            repr(physical_space),
        )
        expansion = self._cluster_expansion_cache.get(cache_key)
        if expansion is None:
            expansion = MPOClusterProductExpansion.from_mpo_basis(
                self,
                cluster_size=cluster_size,
                cutoff=cutoff,
                max_bond=max_bond,
                symmetry=symmetry,
                physical_charges=physical_charges,
                fermionic=fermionic,
                physical_space=physical_space,
            )
            self._cluster_expansion_cache[cache_key] = expansion
        return expansion.compile_exp()

    def graph_cluster_expansion(
        self,
        step=1.0,
        parameters=None,
        *,
        graph=None,
        cluster_size=2,
        cutoff=1.0e-12,
        max_bond=None,
        graph_assembly="auto",
        max_collection_order=None,
        collection_budget=128,
        assembly="direct",
        assembly_chi=None,
        assembly_batch_size="auto",
        assembly_cutoff=None,
        assembly_cutoff_mode="auto",
        assembly_form="left",
        symmetry=None,
        physical_charges=None,
        fermionic=False,
        physical_space=None,
    ):
        """Build a graph-aware cluster expansion with MPO output.

        ``graph`` may be a :class:`ClusterLattice`, a ``(sites, edges)``
        pair, or a mapping. Coordinate-labelled graphs are mapped through
        this basis' square-lattice map. When omitted, a square-lattice basis
        uses its physical nearest-neighbour graph; a chain basis infers edges
        from its term supports.

        ``cluster_size`` counts graph sites, not the span in MPO chain
        positions. A long-range two-site graph edge is therefore a genuine
        two-site cluster even when its MPO representation crosses many chain
        sites. ``graph_assembly`` controls products of disjoint graph
        residuals whose chain spans cross or nest; ``"auto"`` first uses a
        cutwidth-aware frontier planner and falls back to a bounded
        one-cluster approximation when its finite budget or planner work
        limit is exceeded. ``assembly="streaming"`` builds local residual
        cores, inserts bounded batches directly into the accumulator, and
        applies a semantic fixed-rank SVD after each batch;
        ``assembly_chi`` and ``assembly_batch_size`` control that working
        boundary.
        """
        compiled = self.compile_graph_cluster_expansion(
            graph=graph,
            cluster_size=cluster_size,
            cutoff=cutoff,
            max_bond=max_bond,
            graph_assembly=graph_assembly,
            max_collection_order=max_collection_order,
            collection_budget=collection_budget,
            assembly=assembly,
            assembly_chi=assembly_chi,
            assembly_batch_size=assembly_batch_size,
            assembly_cutoff=assembly_cutoff,
            assembly_cutoff_mode=assembly_cutoff_mode,
            assembly_form=assembly_form,
            symmetry=symmetry,
            physical_charges=physical_charges,
            fermionic=fermionic,
            physical_space=physical_space,
        )
        return compiled.exp(step, parameters=parameters)

    def compile_graph_cluster_expansion(
        self,
        *,
        graph=None,
        cluster_size=2,
        cutoff=1.0e-12,
        max_bond=None,
        graph_assembly="auto",
        max_collection_order=None,
        collection_budget=128,
        assembly="direct",
        assembly_chi=None,
        assembly_batch_size="auto",
        assembly_cutoff=None,
        assembly_cutoff_mode="auto",
        assembly_form="left",
        symmetry=None,
        physical_charges=None,
        fermionic=False,
        physical_space=None,
    ):
        """Compile a reusable graph-aware ordered cluster evaluator.

        ``graph_assembly="exact"`` retains every compatible graph-cluster
        collection up to ``collection_budget``. Use
        ``graph_assembly="bounded"`` and ``max_collection_order`` for an
        explicit approximation on wide MPO orderings. ``assembly="streaming"``
        is a separate working-memory control that inserts graph-path cores
        directly into the accumulator in batches, without temporary path or
        batch MPOs.
        """
        from .mpo_product import (  # pylint: disable=import-outside-toplevel
            MPOGraphClusterProductExpansion,
            _graph_lattice_for_basis,
            _normalize_graph_assembly,
            _normalize_mpo_assembly,
            _validate_assembly_batch_size,
            _validate_assembly_chi,
            _normalize_assembly_cutoff_mode,
            _normalize_assembly_form,
            _validate_assembly_cutoff,
            _validate_graph_collection_budget,
            _validate_graph_collection_order,
        )

        graph_assembly = _normalize_graph_assembly(graph_assembly)
        assembly = _normalize_mpo_assembly(assembly)
        assembly_chi = _validate_assembly_chi(assembly_chi)
        assembly_batch_size = _validate_assembly_batch_size(
            assembly_batch_size
        )
        assembly_cutoff = _validate_assembly_cutoff(assembly_cutoff)
        assembly_cutoff_mode = _normalize_assembly_cutoff_mode(
            assembly_cutoff_mode
        )
        assembly_form = _normalize_assembly_form(assembly_form)
        max_collection_order = _validate_graph_collection_order(
            max_collection_order
        )
        collection_budget = _validate_graph_collection_budget(collection_budget)
        normalized_graph = _graph_lattice_for_basis(graph, self)
        cache_key = (
            tuple(normalized_graph.sites),
            tuple(normalized_graph.edges),
            int(cluster_size),
            None if cutoff is None else float(cutoff),
            None if max_bond is None else int(max_bond),
            graph_assembly,
            None if max_collection_order is None else int(max_collection_order),
            None if collection_budget is None else int(collection_budget),
            assembly,
            None if assembly_chi is None else int(assembly_chi),
            assembly_batch_size,
            assembly_cutoff,
            assembly_cutoff_mode,
            assembly_form,
            symmetry,
            repr(physical_charges),
            bool(fermionic),
            repr(physical_space),
        )
        expansion = self._graph_cluster_expansion_cache.get(cache_key)
        if expansion is None:
            expansion = MPOGraphClusterProductExpansion.from_mpo_basis(
                self,
                graph=normalized_graph,
                cluster_size=cluster_size,
                cutoff=cutoff,
                max_bond=max_bond,
                graph_assembly=graph_assembly,
                max_collection_order=max_collection_order,
                collection_budget=collection_budget,
                assembly=assembly,
                assembly_chi=assembly_chi,
                assembly_batch_size=assembly_batch_size,
                assembly_cutoff=assembly_cutoff,
                assembly_cutoff_mode=assembly_cutoff_mode,
                assembly_form=assembly_form,
                symmetry=symmetry,
                physical_charges=physical_charges,
                fermionic=fermionic,
                physical_space=physical_space,
            )
            self._graph_cluster_expansion_cache[cache_key] = expansion
        return expansion.compile_exp()

    def compile_evolution(
        self,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        history_storage="auto",
        extension_budget=None,
    ):
        """Compatibility wrapper for :meth:`compile_exp`.

        New code should call ``compile_exp``. This historical name remains
        available for existing programs and returns the same cached
        :class:`CompiledMPOExp` object.
        """
        history_storage = _canonical_history_storage(history_storage)
        key = (
            order,
            mode,
            extend,
            approximate,
            max_bond,
            on_exceed,
            history_storage,
            extension_budget,
        )
        try:
            compiled = self._compiled_evolution_cache.get(key)
        except TypeError:
            compiled = None
        if compiled is None:
            compiled = CompiledMPOExp(
                self,
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
                extension_budget=extension_budget,
            )
            try:
                self._compiled_evolution_cache[key] = compiled
            except TypeError:
                pass
        return compiled

    def compile_exp(self, **kwargs):
        """Compile a reusable value-only exponential evaluator.

        The returned :class:`CompiledMPOExp` stores only reusable structure:
        history topology, virtual transfer plans, coefficient-slot indices,
        and static operator banks. Each ``exp`` or ``exp_arrays`` call creates
        fresh backend values, keeping Torch/JAX autodiff graphs current.

        Parameters are the same as :meth:`MPOBasis.exp`, except that the
        ``step`` and coefficient values are supplied when the compiled object
        is called. Use ``compiled.exp`` for a semantic MPO and
        ``compiled.exp_arrays`` for raw backend tensor tuples.
        """
        return self.compile_evolution(**kwargs)

    def _resolve_coefficient(self, coefficient, parameters):
        if isinstance(coefficient, MPOParameter):
            return coefficient.resolve(parameters)
        if callable(coefficient):
            if parameters is None:
                raise KeyError(
                    "callable MPO coefficients require a parameter container."
                )
            coefficient = coefficient(parameters)
        _check_scalar(coefficient, name="MPO coefficient")
        return coefficient

    def _convert_value_to_backend(self, value):
        """Convert host scalar values while preserving existing graph values."""
        if self.to_backend is not None and _backend_name(value) in {
            "builtins",
            "numpy",
        }:
            return self.to_backend(value)
        return value

    @staticmethod
    def _local_operator(term, site):
        """Return the local factor carried by ``term`` at ``site``."""
        for position, support_site in enumerate(term.sites):
            if support_site == site:
                return term.operators[position]
        string_position = 0
        for left, right in zip(term.sites, term.sites[1:]):
            for gap_site in range(left + 1, right):
                if gap_site == site:
                    if term.string_operators is None:
                        return ar.do("eye", term.operators[0].shape[0], like=term.operators[0])
                    return term.string_operators[string_position]
                string_position += 1
        raise ValueError(
            f"coefficient slot site {site} is outside term support {term.sites}."
        )

    def coefficients(self, parameters=None):
        """Evaluate all term coefficients as one backend-native batch."""
        values = tuple(
            self._resolve_coefficient(term.coefficient, parameters)
            for term in self._terms
        )
        values = tuple(self._convert_value_to_backend(value) for value in values)
        reference = _backend_reference(values)
        values = tuple(_as_backend(value, like=reference) for value in values)
        return ar.do("stack", values, axis=0)

    def _coefficient_values(self, parameters, coefficients):
        if coefficients is None:
            batch = self.coefficients(parameters)
            return tuple(batch[index] for index in range(self.num_terms))
        if parameters is not None:
            raise ValueError(
                "parameters and coefficients are mutually exclusive."
            )
        shape = getattr(coefficients, "shape", None)
        if shape is not None:
            shape = tuple(shape)
            if not shape:
                if self.num_terms != 1:
                    raise ValueError(
                        "a scalar coefficient batch is valid only for one term."
                    )
                values = (coefficients,)
            elif len(shape) == 1:
                if int(shape[0]) != self.num_terms:
                    raise ValueError(
                        f"coefficients must have length {self.num_terms}, "
                        f"got {shape[0]}."
                    )
                values = tuple(coefficients[index] for index in range(self.num_terms))
            else:
                raise ValueError("coefficients must be a one-dimensional batch.")
        else:
            try:
                values = tuple(coefficients)
            except TypeError as exc:
                raise TypeError(
                    "coefficients must be a one-dimensional batch."
                ) from exc
            if len(values) != self.num_terms:
                raise ValueError(
                    f"coefficients must have length {self.num_terms}, "
                    f"got {len(values)}."
                )
        for index, value in enumerate(values):
            _check_scalar(value, name=f"coefficients[{index}]")
        values = tuple(self._convert_value_to_backend(value) for value in values)
        reference = _backend_reference(values)
        return tuple(_as_backend(value, like=reference) for value in values)

    def build(self, parameters=None, *, coefficients=None):
        """Bind current coefficients and return ``H(parameters)``.

        Parameters
        ----------
        parameters : mapping or sequence, optional
            Values used by :class:`MPOParameter` references and coefficient
            callables.  A backend scalar is inserted directly into the local
            transition, preserving its autodiff graph.
        coefficients : one-dimensional array-like, optional
            Already-evaluated coefficient batch. This is useful when an
            optimizer evaluates all coefficients in one backend operation.
            It is mutually exclusive with ``parameters``.
        """
        transitions = [list(site_transitions) for site_transitions in self._automaton.transitions]
        coefficient_values = self._coefficient_values(parameters, coefficients)
        coefficient_batch = _stack(coefficient_values, axis=0)
        vectorized_keys = self._vectorized_slot_keys
        for site, transition_index, term_indices, operators in (
            self._vectorized_slot_groups
        ):
            index = _as_backend(term_indices, like=coefficient_batch)
            selected = coefficient_batch[index]
            operator_batch = _as_backend(operators, like=selected)
            selected, operator_batch = _align_tensordot_dtypes(
                selected,
                operator_batch,
            )
            weighted = ar.do(
                "tensordot",
                selected,
                operator_batch,
                axes=([0], [0]),
            )
            transition = transitions[site][transition_index]
            transitions[site][transition_index] = type(transition)(
                transition.left_state,
                transition.right_state,
                weighted,
            )
        for site, transition_index, contributions in self._slot_groups:
            if (site, transition_index) in vectorized_keys:
                continue
            weighted = None
            for term_index, operator in contributions:
                contribution = _multiply_scalar(
                    coefficient_values[term_index],
                    operator,
                )
                if weighted is None:
                    weighted = contribution
                    continue
                reference = _backend_reference((weighted, contribution))
                weighted = ar.do(
                    "add",
                    _as_backend(weighted, like=reference),
                    _as_backend(contribution, like=reference),
                )
            transition = transitions[site][transition_index]
            transitions[site][transition_index] = type(transition)(
                transition.left_state,
                transition.right_state,
                weighted,
            )

        automaton = MPOAutomaton(
            self.L,
            channels=self._automaton.channels,
            transitions=transitions,
            start_state=self._automaton.start_state,
            done_state=self._automaton.done_state,
            phys_dim=self.phys_dim,
        )
        result = FirstDegreeMPO.from_automaton(
            automaton,
            **self._template._symmetry_options(),
            upper_ind_id=self._template.upper_ind_id,
            lower_ind_id=self._template.lower_ind_id,
            site_tag_id=self._template.site_tag_id,
            degree=1,
        )
        result._history_topology_cache = self._template._history_topology_cache
        result._history_symbolic_cache = self._template._history_symbolic_cache
        result._history_extension_plan_cache = (
            self._template._history_extension_plan_cache
        )
        result._history_compression_plan_cache = (
            self._template._history_compression_plan_cache
        )
        result._history_approximation_plan_cache = (
            self._template._history_approximation_plan_cache
        )
        result._history_tensor_plan_cache = self._template._history_tensor_plan_cache
        result._history_reduced_plan_cache = self._template._history_reduced_plan_cache
        result.metadata.update({
            "operation": "mpo_basis_build",
            "basis_terms": self.num_terms,
            "basis_build": self._build_count + 1,
            "coefficient_batch": True,
        })
        self._build_count += 1
        return result

    __call__ = build

    def extensive_exponential(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
    ):
        """Build ``exp(dt * H(parameters))`` with the higher-order MPO path."""
        return self.build(
            parameters,
            coefficients=coefficients,
        ).extensive_exponential(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
            progress=progress,
        )

    def exp(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
    ):
        """Build ``exp(step * H(parameters))`` with optional compression.

        ``step`` is the actual scalar multiplying the Hamiltonian. For
        real-time evolution, pass ``step=-1j * tau``; ``dt=...`` remains a
        compatibility keyword. ``chi`` is the final MPO bond cap, while
        ``max_bond`` only guards the temporary higher-order history.
        ``progress=True`` displays stage timings and current/final bond sizes.
        """
        step = _resolve_exp_step(step, dt)
        return self.build(parameters, coefficients=coefficients).exp(
            step,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
            progress=progress,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
        )

    def time_evolution(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
    ):
        """Build the real-time MPO ``exp(-1j * dt * H(parameters))``."""
        return self.build(parameters, coefficients=coefficients).time_evolution(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
            progress=progress,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
        )

    def exp_arrays(
        self,
        step=None,
        parameters=None,
        *,
        coefficients=None,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        extension_budget=None,
    ):
        """Evaluate ``exp(step * H)`` as backend-native tensor tuples.

        This method is intended for JAX/Torch compiled numerical kernels:
        the reusable basis remains a Python-side structural object, while
        the returned tensors are the only values captured by the optimizer's
        autodiff graph.
        """
        step = _resolve_exp_step(step, dt)
        if cache_history:
            return self.compile_exp(
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
                extension_budget=extension_budget,
            ).exp_arrays(
                step,
                parameters,
                coefficients=coefficients,
            )
        return self.build(
            parameters,
            coefficients=coefficients,
        ).exp_arrays(
            step,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
        )

    def time_evolution_arrays(
        self,
        dt,
        parameters=None,
        *,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        extension_budget=None,
    ):
        """Evaluate real-time evolution as backend-native tensor tuples."""
        if cache_history:
            return self.compile_evolution(
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
                extension_budget=extension_budget,
            ).time_evolution_arrays(
                dt,
                parameters,
                coefficients=coefficients,
            )
        return self.build(
            parameters,
            coefficients=coefficients,
        ).time_evolution_arrays(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
        )

    def exp_batch(
        self,
        step=None,
        coefficients=None,
        *,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        extension_budget=None,
    ):
        """Evaluate ``exp(step * H)`` for a batch of coefficient vectors.

        The structural plan is compiled once and every row is evaluated with
        fresh backend tensors, preserving gradients with respect to both the
        coefficient batch and ``step``. JAX and current Torch releases use
        native ``vmap``; NumPy and unsupported backend operations use a small
        autodiff-safe assembly loop.
        """
        step = _resolve_exp_step(step, dt)
        if coefficients is None:
            raise TypeError("exp_batch requires a coefficient batch.")
        shape = getattr(coefficients, "shape", None)
        if shape is None or len(tuple(shape)) != 2:
            raise ValueError(
                "coefficients must have shape (batch, number_of_terms)."
            )
        if int(shape[1]) != self.num_terms:
            raise ValueError(
                f"coefficients must have {self.num_terms} columns, got {shape[1]}."
            )
        options = {
            "order": order,
            "mode": mode,
            "extend": extend,
            "approximate": approximate,
            "max_bond": max_bond,
            "on_exceed": on_exceed,
            "cache_history": cache_history,
            "history_storage": history_storage,
            "extension_budget": extension_budget,
        }
        if cache_history:
            compiled = self.compile_exp(
                order=order,
                mode=mode,
                extend=extend,
                approximate=approximate,
                max_bond=max_bond,
                on_exceed=on_exceed,
                history_storage=history_storage,
                extension_budget=extension_budget,
            )
            if _backend_name(coefficients) == "jax":
                import jax  # pylint: disable=import-outside-toplevel

                return jax.vmap(
                    lambda row: compiled.exp_arrays(
                        step,
                        coefficients=row,
                    )
                )(coefficients)
            if _backend_name(coefficients) == "torch":
                try:
                    import torch  # pylint: disable=import-outside-toplevel

                    vmap = getattr(torch, "vmap", None)
                    if vmap is None:
                        from torch.func import vmap  # pylint: disable=import-outside-toplevel
                    return vmap(
                        lambda row: compiled.exp_arrays(
                            step,
                            coefficients=row,
                        )
                    )(coefficients)
                except (ImportError, RuntimeError, TypeError):
                    pass
            rows = [
                compiled.exp_arrays(
                    step,
                    coefficients=coefficients[index],
                )
                for index in range(int(shape[0]))
            ]
            if not rows:
                return tuple()
            return tuple(
                ar.do("stack", tuple(row[site] for row in rows), axis=0)
                for site in range(self.L)
            )
        backend = _backend_name(coefficients)
        if backend == "jax":
            import jax  # pylint: disable=import-outside-toplevel

            return jax.vmap(
                lambda row: self.exp_arrays(
                    step,
                    coefficients=row,
                    **options,
                )
            )(coefficients)
        if backend == "torch":
            try:
                import torch  # pylint: disable=import-outside-toplevel

                vmap = getattr(torch, "vmap", None)
                if vmap is None:
                    from torch.func import vmap  # pylint: disable=import-outside-toplevel
                return vmap(
                    lambda row: self.exp_arrays(
                        step,
                        coefficients=row,
                        **options,
                    )
                )(coefficients)
            except (ImportError, RuntimeError, TypeError):
                # Some older Torch releases cannot batch one of the backend
                # scatter primitives. The ordinary loop remains fully
                # differentiable and is a safe compatibility fallback.
                pass
        rows = [
            self.exp_arrays(
                step,
                coefficients=coefficients[index],
                **options,
            )
            for index in range(int(shape[0]))
        ]
        if not rows:
            return tuple()
        return tuple(
            ar.do("stack", tuple(row[site] for row in rows), axis=0)
            for site in range(self.L)
        )

    def evolution_mpo(
        self,
        parameters=None,
        *,
        dt,
        coefficients=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
    ):
        """Build ``exp(-1j * dt * H(parameters))`` with optional compression.

        ``chi`` is the final MPO bond cap. It is separate from the
        ``max_bond`` keyword accepted by the higher-order builder, which only
        protects the temporary history representation. With ``chi=None`` the
        method returns the existing semantic :class:`FirstDegreeMPO`. When
        ``chi`` is set, the default numerical path returns a Quimb MPO; set
        ``differentiable=True`` for a fixed-rank autodiff-compatible semantic
        MPO instead.
        """
        return self.time_evolution(
            dt,
            parameters,
            coefficients=coefficients,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
            progress=progress,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
        )


def exp_mpo(
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
    order=1,
    mode=None,
    extend=False,
    approximate=False,
    max_bond=None,
    on_exceed="raise",
    cache_history=True,
    history_storage="auto",
    extension_budget=None,
    chi=None,
    cutoff=1.0e-10,
    cutoff_mode="rel",
    compression=None,
    differentiable=False,
    sector_aware="auto",
    symmetry=None,
    physical_charges=None,
    fermionic=False,
    physical_space=None,
    to_backend=None,
    return_semantic=False,
    return_report=False,
    form=None,
    create_bond=False,
    compress_opts=None,
    progress=False,
):
    """Build an exponential MPO directly from local operator terms.

    The usual term form is ``{"operator": "ZZ", "location": (0, 1),
    "coefficient": value}``. The compact Pepsy mapping form
    ``{"XX": (2, 3)}`` or ``{"XX": ((2, 3), coefficient)}`` is equivalent.
    Tuple terms may also use ``(location, paulis, coefficient)`` or
    ``((paulis, coefficient), location)``. The latter forms are shared with
    :meth:`ham_tn.build_mpo`.
    ``operator`` may also be a local matrix or a sequence of local matrices,
    and locations may be integer chain sites or 2D/3D coordinates. The lattice
    shape is inferred when possible, or can be supplied as
    ``shape=(lx, ly[, lz])``. Terms with common supports are
    compiled through one shared automaton, so onsite contributions are
    combined at their transition while coefficient slots remain independent.

    ``to_backend`` converts the compiled local operator blocks and coefficient
    values before the higher-order contraction path runs. The final Quimb MPO
    is also checked at the boundary with ``apply_to_arrays`` so contractions
    and numerical compression remain on the requested backend.

    By default this convenience function returns a compiled Quimb MPO. Set
    ``return_semantic=True`` when the higher-order history object and its
    ``to_mpo()`` boundary are needed. When ``chi`` is supplied, the terms are
    first compiled into the higher-order MPO and the resulting final MPO is
    then compressed to that bond cap. Pass ``form``/``create_bond`` for direct
    Quimb controls and use ``compress_opts`` for additional compression
    keywords such as ``method``, ``absorb``, ``renorm``, or ``info``. Pass
    ``symmetry`` and
    ``physical_charges`` to select the native bosonic block-sparse compiler.
    Pass ``progress=True`` to display stage timings, current bond sizes, and
    the final ``chi`` compression.
    """
    if (
        return_semantic
        and chi is not None
        and not differentiable
        and compression != "fixed_rank"
    ):
        raise ValueError(
            "return_semantic=True with chi requires compression='fixed_rank' "
            "or differentiable=True; Quimb compression returns an ordinary MPO."
        )
    basis = MPOBasis.from_terms(
        terms,
        shape=shape,
        mapper=mapper,
        map_mode=map_mode,
        phys_dim=phys_dim,
        symmetry=symmetry,
        physical_charges=physical_charges,
        fermionic=fermionic,
        physical_space=physical_space,
        to_backend=to_backend,
    )
    result = basis.exp(
        step,
        parameters,
        coefficients=coefficients,
        dt=dt,
        order=order,
        mode=mode,
        extend=extend,
        approximate=approximate,
        max_bond=max_bond,
        on_exceed=on_exceed,
        cache_history=cache_history,
        history_storage=history_storage,
        extension_budget=extension_budget,
        chi=chi,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        compression=compression,
        differentiable=differentiable,
        sector_aware=sector_aware,
        return_report=return_report,
        form=form,
        create_bond=create_bond,
        compress_opts=compress_opts,
        progress=progress,
    )
    if return_report:
        semantic_result, report = result
    else:
        semantic_result, report = result, None

    if return_semantic:
        output = semantic_result
    elif hasattr(semantic_result, "to_mpo"):
        output = semantic_result.to_mpo()
    else:
        output = semantic_result
    if not return_semantic:
        _apply_to_backend(output, to_backend)
        if progress and not hasattr(output, "pepsy_exp_metadata"):
            semantic = getattr(output, "pepsy_first_degree", None)
            if semantic is not None:
                output.pepsy_exp_metadata = dict(semantic.metadata)
    return (output, report) if return_report else output


# Resolve the semantic implementation only after this module's classes and
# convenience function have been defined. This keeps direct imports safe while
# ``mpo.py`` re-exports the same objects at its extraction boundary.
from .mpo_semantic import (  # noqa: E402
    FirstDegreeMPO,
    MPOLocalOperatorTerm,
    MPOParameter,
    MPOProductTerm,
    _MAX_FUSED_SLOT_BANK_ELEMENTS,
    _align_tensordot_dtypes,
    _check_scalar,
    _complex_dtype,
    _compile_generic_terms,
    _is_integral_value,
    _mixed_term_automaton,
    _resolve_exp_step,
    _scatter_add_into_2d,
    _square_lattice_pauli_term,
    _stack,
    _term_from_input,
    _zeros,
)
