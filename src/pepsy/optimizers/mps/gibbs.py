"""Finite-temperature Gibbs states represented by purified MPSs.

The first Gibbs-MPS implementation keeps the physical and ancilla legs as an
interleaved open MPS::

    (physical_0, ancilla_0, physical_1, ancilla_1, ...).

The initial state is a product of Bell pairs.  Imaginary-time evolution acts
only on the even sites, and :class:`MpsOptimizer` supplies the ordinary MPS
gate replay and compression machinery.  Tracing the odd sites returns the
thermal operator as an MPO.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends.convert import (
    infer_backend_converter_from_sample,
    resolve_backend_sample_data,
)
from ...operators.mpo_higher_order import (
    MPOLocalOperatorTerm,
    MPOBasis,
    MPOProductTerm,
)
from ...tensors.constructors import bell_to_mps
from .optimizer import MpsOptimizer

__all__ = ["GibbsMps"]


def _backend_name(value):
    """Return an Autoray backend name without forcing host conversion."""
    try:
        return ar.infer_backend(value)
    except Exception:  # pragma: no cover - defensive duck-typing boundary
        return None


def _find_backend_sample(value):
    """Find the first non-NumPy array nested in a term specification."""
    sample = resolve_backend_sample_data(value)
    if sample is not None:
        if _backend_name(sample) not in {"builtins", "numpy"}:
            return sample
        return None
    if isinstance(value, Mapping):
        for item in value.values():
            sample = _find_backend_sample(item)
            if sample is not None:
                return sample
        return None
    if isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(value)
    except TypeError:
        return None
    for item in values:
        sample = _find_backend_sample(item)
        if sample is not None:
            return sample
    return None


def _scalar_float(value, *, name):
    """Convert a scalar-like value to a host float for validation only."""
    try:
        array = np.asarray(ar.to_numpy(value))
    except Exception as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc
    if array.shape != ():
        raise TypeError(f"{name} must be a real scalar.")
    result = array.item()
    if np.iscomplexobj(result) and abs(np.imag(result)) > 1.0e-14:
        raise ValueError(f"{name} must be real.")
    return float(np.real(result))


def _dtype_name(value):
    """Return a backend-independent dtype name when one is available."""
    try:
        return str(ar.get_dtype_name(value))
    except Exception:
        return None


def _host_dtype(values):
    """Choose a host dtype compatible with the generated gate stream."""
    names = tuple(
        name
        for value in values
        if (name := _dtype_name(value)) is not None
    )
    if not names:
        return np.dtype("complex128")
    try:
        dtype = np.result_type(*(np.dtype(name) for name in names))
    except TypeError:
        return np.dtype("complex128")
    if dtype.kind in "biu":
        return np.dtype("float64")
    return dtype


class GibbsMps:
    r"""Prepare a finite-temperature Gibbs state as a purified MPS.

    The Hamiltonian is supplied in the same term-centric forms as
    :class:`pepsy.operators.MPOBasis`.  One-dimensional terms use integer
    sites directly.  A ``shape=(Lx, Ly[, Lz])`` together with ``map_mode`` or
    ``mapper`` uses :class:`pepsy.tensors.OneDMap` before gates are generated.

    The internal MPS has ``2 * L`` sites, with physical site ``i`` at MPS
    position ``2 * i`` and ancilla site ``i`` at ``2 * i + 1``.  The initial
    state is

    .. math:: |I\rangle = \bigotimes_i |\Phi_i\rangle,

    and :meth:`prepare` applies a second-order imaginary-time Trotterization
    of ``exp(-beta * H / 2)`` to the physical sites.  Consequently, tracing
    out the ancillas gives a positive operator proportional to
    ``exp(-beta * H)``.

    This first step supports one-site terms and two-site terms, including
    long-range two-site couplings.  Product terms with explicit string
    operators across a gap and general operators on more than two sites are
    rejected until a multi-site gate route is added.

    Parameters
    ----------
    terms : iterable or mapping
        Hamiltonian terms accepted by :meth:`MPOBasis.from_terms`, including
        compact entries such as ``(('ZZ', J), (i, j))`` and mappings such as
        ``{"operator": "X", "location": i, "coefficient": h}``.
    shape : int or tuple, optional
        Chain length or regular lattice shape. If omitted, the location
        structure in ``terms`` is used to infer one-dimensional, 2D, or 3D
        input and the smallest enclosing shape is selected.
    mapper : OneDMap, optional
        Explicit lattice-to-chain mapper.  It is mutually checked against
        ``shape`` by ``MPOBasis``.
    map_mode : str, default="snake"
        Traversal mode used when ``shape`` is a lattice and ``mapper`` is not
        supplied.
    normalized : bool, default=True
        Whether each initial Bell pair is normalized.  Normalized pairs make
        the raw reduced operator ``exp(-beta H) / d**L``; ``to_mpo()`` removes
        this overall scale when returning the normalized Gibbs state.
    to_backend : callable, optional
        Converter applied to all compiled local operators, Bell-pair tensors,
        and generated gates.  If omitted, a converter is inferred when the
        term data already contain Torch, JAX, or CuPy arrays.
    """

    def __init__(
        self,
        terms,
        *,
        shape=None,
        mapper=None,
        map_mode="snake",
        normalized=True,
        to_backend=None,
    ):
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        self.normalized = bool(normalized)
        if isinstance(terms, Mapping):
            basis_terms = terms
        else:
            try:
                basis_terms = tuple(terms)
            except TypeError as exc:
                raise TypeError("terms must be an iterable or mapping.") from exc
        self._raw_terms = basis_terms
        inferred_sample = (
            None if to_backend is not None else _find_backend_sample(basis_terms)
        )
        if inferred_sample is not None:
            to_backend = infer_backend_converter_from_sample(inferred_sample)
        self.to_backend = to_backend
        self.basis = MPOBasis.from_terms(
            basis_terms,
            shape=shape,
            mapper=mapper,
            map_mode=map_mode,
            to_backend=to_backend,
        )
        self._validate_terms()
        self.length = int(self.basis.L)
        self.phys_dim = int(self.basis.phys_dim)
        self.shape = (
            self.basis.lattice_shape
            if self.basis.lattice_shape is not None
            else (self.length,)
        )
        self.mapper = getattr(self.basis, "_lattice_mapper", None)
        self.map_mode = None if self.mapper is None else self.mapper.mode
        self.physical_sites = tuple(2 * site for site in range(self.length))
        self.ancilla_sites = tuple(site + 1 for site in self.physical_sites)
        self._identity_mps = None
        self.optimizer = None
        self.gates = ()
        self.trotter_gates = ()
        self.trotter_layers = ()
        self._trotter_ham = None
        self.trotter_order = None
        self.trotter_ordering = None
        self.trotter_fuse_adjacent = None
        self.trotter_alternate = None
        # ``LocalHamGen.get_auto_ordering`` depends only on the interaction
        # graph. Keep deterministic orderings so repeated preparations do not
        # redo the graph coloring work. Random ordering is deliberately not
        # cached: each request should still produce a fresh randomized path.
        self._trotter_layers_cache = {}
        self.beta = None
        self.trotter_step = None
        self.n_steps = 0
        self._last_parameters = None

    def _validate_terms(self):
        """Validate the gate-supported subset of the Hamiltonian terms."""
        for index, term in enumerate(self.basis.terms):
            sites = tuple(int(site) for site in term.sites)
            if len(sites) not in (1, 2):
                raise NotImplementedError(
                    "GibbsMps currently supports only one- and two-site terms; "
                    f"term {index} acts on {len(sites)} sites."
                )
            if isinstance(term, MPOLocalOperatorTerm) and sites != tuple(sorted(sites)):
                raise ValueError(
                    "general local-operator terms must list sites in increasing order."
                )
            if (
                isinstance(term, MPOProductTerm)
                and term.string_operators is not None
                and len(sites) == 2
                and sites[1] > sites[0] + 1
            ):
                raise NotImplementedError(
                    "GibbsMps does not yet apply explicit string operators across "
                    "a gap; use a plain two-site coupling or a contiguous term."
                )

    def _term_operator(self, term):
        """Return the dense physical operator represented by one term."""
        if isinstance(term, MPOLocalOperatorTerm):
            return term.operator
        if len(term.operators) == 1:
            return term.operators[0]
        return ar.do("kron", term.operators[0], term.operators[1])

    def _scaled_term_operator(self, term, coefficient):
        """Return one term's local Hamiltonian contribution."""
        return ar.do(
            "multiply",
            self._term_operator(term),
            coefficient,
        )

    def _build_identity_mps(self, coefficients=(), beta=None):
        """Build the interleaved Bell-pair MPS on the active backend."""
        values = [
            operator
            for term in self.basis.terms
            for operator in (
                (term.operator,)
                if isinstance(term, MPOLocalOperatorTerm)
                else term.operators
            )
        ]
        values.extend(coefficients)
        if beta is not None:
            values.append(beta)
        dtype = _host_dtype(values)
        return bell_to_mps(
            self.length,
            phys_dim=self.phys_dim,
            dtype=dtype,
            normalized=self.normalized,
            to_backend=self.to_backend,
            site_ind_id="k{}",
            site_tag_id="I{}",
        )

    def _convert_gate_stream(self, gates):
        """Place generated gates on the same backend as the purification."""
        if self.to_backend is None:
            return gates
        return tuple(
            (
                self.to_backend(gate)
                if _backend_name(gate) in {"builtins", "numpy"}
                else gate,
                where,
            )
            for gate, where in gates
        )

    def _build_local_ham(self, coefficients):
        """Compile normalized terms into Quimb's graph Hamiltonian form.

        ``LocalHamGen`` stores pair interactions and can absorb one-site terms
        into incident pairs. Doing that lifting here, rather than passing
        ``H1`` directly, keeps the identity Kronecker products on the selected
        Autoray backend. This preserves differentiable Torch/JAX coefficient
        graphs while retaining Quimb's local-Hamiltonian representation.

        Sites carrying only one-site terms are returned separately. They are
        disconnected from every pair interaction and can therefore be applied
        as exact one-site exponentials without changing the product formula.
        """
        pair_terms = {}
        site_terms = {}

        for term, coefficient in zip(self.basis.terms, coefficients):
            operator = self._scaled_term_operator(term, coefficient)
            sites = tuple(int(site) for site in term.sites)
            if len(sites) == 2:
                where = tuple(sorted(sites))
                if where in pair_terms:
                    pair_terms[where] = ar.do(
                        "add",
                        pair_terms[where],
                        operator,
                    )
                else:
                    pair_terms[where] = operator
            else:
                site = sites[0]
                if site in site_terms:
                    site_terms[site] = ar.do(
                        "add",
                        site_terms[site],
                        operator,
                    )
                else:
                    site_terms[site] = operator

        if not pair_terms:
            return None, site_terms

        covering = {}
        for where in pair_terms:
            for site in where:
                covering.setdefault(site, []).append(where)

        isolated = {}
        for site, operator in site_terms.items():
            pairs = covering.get(site, ())
            if not pairs:
                isolated[site] = operator
                continue

            identity = ar.do("eye", self.phys_dim, like=operator)
            for where in pairs:
                if where[0] == site:
                    lifted = ar.do("kron", operator, identity)
                else:
                    lifted = ar.do("kron", identity, operator)
                lifted = ar.do("divide", lifted, len(pairs))
                pair_terms[where] = ar.do(
                    "add",
                    pair_terms[where],
                    lifted,
                )

        ham = qtn.LocalHamGen(pair_terms)
        if self.to_backend is not None:
            def convert(array):
                if _backend_name(array) in {"builtins", "numpy"}:
                    return self.to_backend(array)
                return array

            # Keep the explicit backend boundary visible on the native
            # Quimb Hamiltonian as well as on the final gate stream.
            ham.apply_to_arrays(convert)
        return ham, isolated

    def _build_isolated_gates(self, isolated, total_exponent):
        """Build exact one-site gates for sites with no pair interactions."""
        gates = []
        for site in sorted(isolated):
            generator = ar.do(
                "multiply",
                isolated[site],
                total_exponent,
            )
            gate = ar.do("linalg.expm", generator)
            gate = ar.do("reshape", gate, (self.phys_dim, self.phys_dim))
            gates.append((gate, (2 * int(site),)))
        return tuple(gates)

    def _build_trotter_stream(
        self,
        coefficients,
        step,
        n_steps,
        *,
        order,
        ordering,
        fuse_adjacent,
        alternate,
    ):
        """Build a Quimb Trotter stream and map it to purification sites."""
        if n_steps == 0:
            return (), (), None, ()

        ham, isolated = self._build_local_ham(coefficients)
        if ham is None:
            total_exponent = ar.do("multiply", step, -int(n_steps))
            return (
                self._build_isolated_gates(isolated, total_exponent),
                (),
                None,
                (),
            )

        get_trotter_gates = getattr(ham, "get_trotter_gates", None)
        if get_trotter_gates is None:
            raise RuntimeError(
                "GibbsMps requires a Quimb LocalHamGen with "
                "get_trotter_gates; upgrade Quimb to use the native "
                "TrotterGate scheduler."
            )

        if isinstance(ordering, str) or ordering is None:
            cache_key = ordering
            if ordering in {"random", "random-ungrouped"}:
                layers = tuple(ham.get_auto_ordering(ordering, group=True))
            else:
                try:
                    layers = self._trotter_layers_cache[cache_key]
                except KeyError:
                    layers = tuple(ham.get_auto_ordering(ordering, group=True))
                    self._trotter_layers_cache[cache_key] = layers
            # Pass the already resolved layers back to Quimb. Besides avoiding
            # a second graph-coloring pass, this makes the metadata and the
            # executable schedule identical for randomized orderings.
            native_ordering = layers
        else:
            native_ordering = tuple(tuple(layer) for layer in ordering)
            layers = native_ordering

        exponent = ar.do("multiply", step, -1.0)
        schedule_kwargs = {
            "order": order,
            "steps": int(n_steps),
            "ordering": native_ordering,
            "fuse_adjacent": bool(fuse_adjacent),
            "alternate": bool(alternate),
        }
        try:
            hash(exponent)
        except TypeError:
            # JAX arrays are intentionally unhashable, while the installed
            # Quimb cache uses ``(id(term), x)`` as its key. Ask Quimb for the
            # exact schedule with a neutral scalar, then regenerate only the
            # payloads through Autoray so the real exponent stays in the JAX
            # graph. The TrotterGate metadata remains Quimb-native.
            scheduled_gates = tuple(get_trotter_gates(0.0, **schedule_kwargs))
            native_gates = tuple(
                type(scheduled_gate)(
                    ar.do(
                        "linalg.expm",
                        ar.do(
                            "multiply",
                            ham.get_gate(scheduled_gate.where),
                            ar.do("multiply", exponent, scheduled_gate.frac),
                        ),
                    ),
                    scheduled_gate.where,
                    scheduled_gate.frac,
                    scheduled_gate.layer,
                    scheduled_gate.step,
                )
                for scheduled_gate in scheduled_gates
            )
        else:
            native_gates = tuple(get_trotter_gates(exponent, **schedule_kwargs))
        gates = tuple(
            (
                ar.do(
                    "reshape",
                    trotter_gate.U,
                    (self.phys_dim,) * (2 * len(trotter_gate.where)),
                ),
                tuple(2 * int(site) for site in trotter_gate.where),
            )
            for trotter_gate in native_gates
        )

        total_exponent = ar.do("multiply", step, -int(n_steps))
        isolated_gates = self._build_isolated_gates(isolated, total_exponent)
        return gates + isolated_gates, native_gates, ham, layers

    def prepare(
        self,
        beta,
        *,
        dt=None,
        n_steps=None,
        chi=64,
        mode="mpo",
        contraction_opt="auto-hq",
        cutoff="auto",
        cutoff_mode="auto",
        progress=False,
        n_iter=8,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        trotter_order=2,
        trotter_ordering="sort",
        trotter_fuse_adjacent=True,
        trotter_alternate=True,
        parameters=None,
        optimizer_kwargs=None,
        run_kwargs=None,
    ):
        r"""Prepare ``(exp(-beta * H / 2) \otimes I) |I>``.

        ``n_steps`` selects the number of Trotter steps. By default these are
        symmetric second-order steps, generated by Quimb's
        :meth:`quimb.tensor.tnag.tebd.LocalHamGen.get_trotter_gates`. If
        ``dt`` is supplied, it is interpreted as a requested maximum imaginary
        time step and the number of steps is chosen by ceiling. With neither
        argument supplied, one Trotter step is used. ``trotter_order`` may be
        ``1``, ``2``, or ``4``; ``trotter_ordering`` controls graph edge
        ordering, while ``trotter_fuse_adjacent`` and ``trotter_alternate``
        are passed to Quimb's scheduler.

        The returned object is ``self``.  Inspect the purification through
        :attr:`mps`, or call :meth:`to_mpo` to trace out the ancillas.
        ``MpsOptimizer`` is always run with ``non_unitary=True`` and without
        unitary stabilization or overlap-fidelity diagnostics. ``n_iter``,
        ``contraction_opt``, and the normalization options are the common
        direct ``MpsOptimizer`` controls; all other constructor and run
        options can be forwarded through ``optimizer_kwargs`` and
        ``run_kwargs``.
        """
        beta_float = _scalar_float(beta, name="beta")
        if beta_float < 0.0:
            raise ValueError("beta must be non-negative.")
        if (
            not isinstance(trotter_order, Integral)
            or isinstance(trotter_order, bool)
            or int(trotter_order) not in (1, 2, 4)
        ):
            raise ValueError("trotter_order must be one of 1, 2, or 4.")
        trotter_order = int(trotter_order)
        if (
            trotter_ordering is not None
            and not isinstance(trotter_ordering, str)
        ):
            try:
                trotter_ordering = tuple(
                    tuple(layer) for layer in trotter_ordering
                )
            except TypeError as exc:
                raise TypeError(
                    "trotter_ordering must be a string, None, or a sequence "
                    "of commuting edge layers."
                ) from exc
        if n_steps is not None:
            if not isinstance(n_steps, Integral) or isinstance(n_steps, bool):
                raise TypeError("n_steps must be a positive integer or None.")
            n_steps = int(n_steps)
            if n_steps < 1 and beta_float > 0.0:
                raise ValueError("n_steps must be positive when beta is nonzero.")
        elif dt is None:
            n_steps = 1 if beta_float > 0.0 else 0
        else:
            dt_float = _scalar_float(dt, name="dt")
            if dt_float <= 0.0:
                raise ValueError("dt must be positive.")
            n_steps = int(np.ceil((beta_float / 2.0) / dt_float))

        if beta_float == 0.0:
            n_steps = 0
        if mode is None:
            mode = "mpo"
        mode_name = str(mode).strip().lower().replace("_", "-")
        if mode_name in {"exact", "perm", "su", "mix"}:
            raise ValueError(
                "GibbsMps requires an ordinary open MPS replay mode; "
                f"mode={mode!r} cannot preserve the physical/ancilla layout."
            )
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        elif not isinstance(optimizer_kwargs, Mapping):
            raise TypeError("optimizer_kwargs must be a mapping or None.")
        else:
            optimizer_kwargs = dict(optimizer_kwargs)
        if run_kwargs is None:
            run_kwargs = {}
        elif not isinstance(run_kwargs, Mapping):
            raise TypeError("run_kwargs must be a mapping or None.")
        else:
            run_kwargs = dict(run_kwargs)

        if run_kwargs.get("stabilize_unitary", False):
            raise ValueError(
                "GibbsMps uses non-unitary evolution; stabilize_unitary is not applicable."
            )
        reserved_constructor = {
            "p",
            "gates",
            "chi",
            "mode",
            "to_backend",
            "contraction_opt",
        }
        overlap = reserved_constructor.intersection(optimizer_kwargs)
        if overlap:
            raise TypeError(
                "optimizer_kwargs cannot override GibbsMps-owned options: "
                + ", ".join(sorted(overlap))
            )
        reserved_run = {
            "progbar",
            "cutoff",
            "cutoff_mode",
            "non_unitary",
            "n_iter",
            "normalize_every",
            "normalize_final",
            "normalize_eps",
            "use_layout_finder",
            "layout_order",
            "layout_kwargs",
            "layout",
            "layout_report",
        }
        overlap = reserved_run.intersection(run_kwargs)
        if "non_unitary" in overlap and run_kwargs["non_unitary"] is not True:
            raise ValueError("GibbsMps requires run_kwargs['non_unitary']=True.")
        overlap.discard("non_unitary")
        if overlap:
            raise TypeError(
                "run_kwargs cannot override GibbsMps-owned options: "
                + ", ".join(sorted(overlap))
            )

        coefficient_batch = self.basis.coefficients(parameters)
        coefficients = tuple(coefficient_batch[index] for index in range(self.basis.num_terms))
        step = 0.0 if n_steps == 0 else ar.do("divide", beta, 2 * n_steps)
        if self.to_backend is not None and _backend_name(step) in {
            "builtins",
            "numpy",
        }:
            step = self.to_backend(step)
        (
            gates,
            trotter_gates,
            trotter_ham,
            trotter_layers,
        ) = self._build_trotter_stream(
            coefficients,
            step,
            n_steps,
            order=trotter_order,
            ordering=trotter_ordering,
            fuse_adjacent=trotter_fuse_adjacent,
            alternate=trotter_alternate,
        )
        gates = self._convert_gate_stream(gates)
        identity = self._build_identity_mps(coefficients, beta=beta)
        identity_snapshot = identity.copy()

        constructor_options = dict(optimizer_kwargs)
        constructor_options.setdefault("ind_id", "k{}")
        constructor_options.setdefault("contraction_opt", contraction_opt)
        self.optimizer = MpsOptimizer(
            identity,
            gates=gates,
            chi=chi,
            mode=mode,
            to_backend=self.to_backend,
            **constructor_options,
        )
        options = dict(run_kwargs)
        options.update(
            {
                "progbar": bool(progress),
                "n_iter": n_iter,
                "cutoff": cutoff,
                "cutoff_mode": cutoff_mode,
                "non_unitary": True,
                "normalize_every": normalize_every,
                "normalize_final": normalize_final,
                "normalize_eps": normalize_eps,
                "stabilize_unitary": False,
            }
        )
        self.optimizer.run(**options)
        self._identity_mps = identity_snapshot
        self.gates = gates
        self.trotter_gates = trotter_gates
        self.trotter_layers = trotter_layers
        self._trotter_ham = trotter_ham
        self.trotter_order = trotter_order
        self.trotter_ordering = trotter_ordering
        self.trotter_fuse_adjacent = bool(trotter_fuse_adjacent)
        self.trotter_alternate = bool(trotter_alternate)
        self.beta = beta
        self.trotter_step = step
        self.n_steps = int(n_steps)
        self._last_parameters = parameters
        return self

    run = prepare

    @property
    def mps(self):
        """Return the live purification MPS, or the identity before prepare."""
        if self.optimizer is not None:
            return self.optimizer.p
        if self._identity_mps is None:
            self._identity_mps = self._build_identity_mps()
        return self._identity_mps

    @property
    def identity_mps(self):
        """Return a copy of the un evolved Bell-pair purification."""
        if self._identity_mps is None:
            self._identity_mps = self._build_identity_mps()
        return self._identity_mps.copy()

    @property
    def raw_mpo(self):
        """Return the ancilla-traced, unnormalized thermal operator."""
        return self._partial_trace_to_mpo()

    def _partial_trace_to_mpo(self, contract_opts=None):
        """Trace ancillas through Quimb, preserving Pepsy scale metadata.

        Quimb's native ``partial_trace_to_mpo`` implementation propagates the
        MPS global exponent into the returned MPO, so it is the default for
        both scaled and unscaled states. Explicit contraction options use the
        equivalent tag-wise route below because the installed public Quimb
        method does not accept those options.
        """
        if contract_opts is None:
            return self.mps.partial_trace_to_mpo(
                keep=self.physical_sites,
                upper_ind_id="b{}",
                rescale_sites=True,
            )
        elif not isinstance(contract_opts, Mapping):
            raise TypeError("contract_opts must be a mapping or None.")
        contract_opts = dict(contract_opts)
        forbidden = {
            "tags",
            "which",
            "output_inds",
            "get",
            "backend",
            "strip_exponent",
            "equalize_norms",
            "preserve_tensor",
            "inplace",
        }
        overlap = forbidden.intersection(contract_opts)
        if overlap:
            raise TypeError(
                "contract_opts cannot override the internal tag-trace contract: "
                + ", ".join(sorted(overlap))
            )

        mps = self.mps
        keep = tuple(sorted(self.physical_sites))
        p_bra = mps.copy()
        p_bra.reindex_sites_("b{}", where=keep)
        rho = mps.H & p_bra

        for site in mps.gen_sites_present():
            tag = mps.site_tag(site)
            if site in keep:
                rho.contract_tags(
                    tag,
                    inplace=True,
                    strip_exponent=True,
                    **contract_opts,
                )
            else:
                next_site = site + 1 if site < mps.L - 1 else max(keep)
                rho.contract_cumulative(
                    (tag, mps.site_tag(next_site)),
                    inplace=True,
                    strip_exponent=True,
                    **contract_opts,
                )
                rho.drop_tags(tag)

        retag = {}
        reind = {}
        for new, old in enumerate(keep):
            retag[mps.site_tag(old)] = mps.site_tag(new)
            reind[mps.site_ind(old)] = mps.site_ind(new)
            reind[f"b{old}"] = f"b{new}"
        rho.retag_(retag)
        rho.reindex_(reind)
        rho.view_as_(
            qtn.MatrixProductOperator,
            cyclic=mps.cyclic,
            L=len(keep),
            site_tag_id=mps.site_tag_id,
            lower_ind_id="b{}",
            upper_ind_id=mps.site_ind_id,
        )
        rho.fuse_multibonds_()
        return rho

    def to_mpo(self, *, normalized=True, contract_opts=None):
        """Trace out ancillas and return the thermal operator as an MPO.

        By default this uses Quimb's native ``partial_trace_to_mpo`` path,
        including when Pepsy has stored a global MPS exponent. If
        ``contract_opts`` is supplied, those options are forwarded to native
        ``contract_tags`` and ``contract_cumulative`` operations with
        ``strip_exponent=True``. Neither route densifies the full
        purification.
        """
        rho = self._partial_trace_to_mpo(contract_opts)
        if not normalized:
            return rho
        trace = ar.do("real", rho.trace())
        trace_float = _scalar_float(trace, name="thermal trace")
        if not np.isfinite(trace_float) or trace_float <= 1.0e-15:
            raise FloatingPointError("cannot normalize a zero or non-finite thermal trace.")

        # ``TensorNetwork.exponent`` stores the scale removed from the working
        # purification. The traced MPO carries twice the MPS amplitude
        # exponent, but ``MPO.trace()`` contracts only the rescaled tensor
        # data. Normalize the data trace and clear the metadata so the returned
        # MPO is represented at unit trace without materializing a potentially
        # enormous physical scale.
        if hasattr(rho, "exponent"):
            rho.exponent = 0.0
        scale = ar.do("divide", 1.0, trace)
        return rho.multiply(scale, inplace=False)

    density_mpo = to_mpo

    def to_dense(self, *, normalized=True, trace_contract_opts=None, **contract_opts):
        """Return the ancilla-traced thermal operator as a dense matrix.

        ``trace_contract_opts`` controls the preceding MPO trace; the other
        keyword arguments are passed to the final MPO-to-dense contraction.
        """
        return self.to_mpo(
            normalized=normalized,
            contract_opts=trace_contract_opts,
        ).to_dense(**contract_opts)

    def trace(self, *, contract_opts=None):
        """Return the trace of the raw ancilla-traced operator."""
        rho = self._partial_trace_to_mpo(contract_opts)
        trace = rho.trace()
        exponent = getattr(rho, "exponent", 0.0)
        if exponent == 0:
            return trace
        scale = ar.do("power", 10.0, exponent)
        return ar.do("multiply", trace, scale)

    @staticmethod
    def _log_mpo_trace(rho):
        """Return the natural log of a positive MPO trace in log-space."""
        trace = ar.do("real", rho.trace())
        trace_float = _scalar_float(trace, name="thermal trace")
        if not np.isfinite(trace_float) or trace_float <= 1.0e-15:
            raise FloatingPointError(
                "cannot take the log of a zero, negative, or non-finite "
                "thermal trace."
            )
        log_trace = ar.do("log", trace)
        exponent = getattr(rho, "exponent", 0.0)
        if exponent != 0:
            log_trace = ar.do(
                "add",
                log_trace,
                float(exponent) * np.log(10.0),
            )
        return log_trace

    def log_partition_function(self, *, contract_opts=None):
        """Return ``log(Z)`` in natural-log units without exponentiating it.

        The scale removed from a normalized working purification is combined
        with the rescaled MPO trace in log-space. For normalized Bell pairs,
        the physical partition function also includes the ``phys_dim**length``
        factor.
        """
        rho = self._partial_trace_to_mpo(contract_opts)
        log_z = self._log_mpo_trace(rho)
        if self.normalized:
            log_z = ar.do(
                "add",
                log_z,
                self.length * np.log(self.phys_dim),
            )
        return log_z

    def partition_function(self, *, contract_opts=None):
        """Return ``Z = Tr(exp(-beta H))`` represented by the purification."""
        return ar.do(
            "exp",
            self.log_partition_function(contract_opts=contract_opts),
        )

    def __repr__(self):
        return (
            f"GibbsMps(length={self.length}, phys_dim={self.phys_dim}, "
            f"beta={self.beta!r}, n_steps={self.n_steps})"
        )
