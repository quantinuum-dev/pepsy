"""Symmray-backed MPS and PEPS construction, evolution, and measurement.

Model terms, array conversions, and diagnostics are owned by ``symmetric``.
That module resolves its historical state-class aliases lazily, so either
module can be imported first without an eager circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import quimb.tensor as qtn

from pepsy.boundary._measurements import compute_peps_local_expectation
from pepsy._internal.quimb import call_quimb_2d
from .symmetric_diagnostics import (
    symmray_mps_summary,
    symmray_peps_summary,
)
from .symmetric import (
    SymHamiltonian,
    _MODEL_DEFAULTS,
    _apply_to_tensor_network_arrays,
    _as_edges,
    _as_scalar,
    _default_site_charge,
    _format_site_ind,
    _is_symmray_array,
    _normalize_model,
    _open_chain_edges,
    _random_charge_preserving_two_site_dense,
    _require_symmray,
    _resolve_phys_sectors,
    _right_canonize_mps,
    _sites_from_gate_where,
    _zero_like_charge,
    symm_operator_from_dense,
)

__all__ = ["SymMPS", "SymPEPS"]


@dataclass(init=False)
class _SymState:
    """Shared implementation for symmetric tensor-network states."""

    psi: qtn.TensorNetwork
    symmetry: str
    edges: tuple
    fermionic: bool = False
    model: str | None = None
    hamiltonian: SymHamiltonian | None = None
    contraction_opt: object = "auto-hq"
    site_ind_id: str = "k{}"
    gauges: dict | None = None
    phys_sectors: dict | None = None
    site_charge: object = None

    def __init__(
        self,
        psi=None,
        symmetry=None,
        edges=None,
        *,
        network=None,
        mps=None,
        peps=None,
        fermionic=False,
        model=None,
        hamiltonian=None,
        contraction_opt="auto-hq",
        site_ind_id="k{}",
        gauges=None,
        phys_sectors=None,
        site_charge=None,
    ):
        supplied_states = [
            (name, value)
            for name, value in (
                ("psi", psi),
                ("network", network),
                ("mps", mps),
                ("peps", peps),
            )
            if value is not None
        ]
        if len(supplied_states) != 1:
            raise TypeError("Pass exactly one of `psi`, `network`, `mps`, or `peps`.")

        state_name, state = supplied_states[0]
        class_name = type(self).__name__
        if state_name == "mps" and class_name != "SymMPS":
            raise TypeError("`mps=` is only valid when constructing `SymMPS`.")
        if state_name == "peps" and class_name != "SymPEPS":
            raise TypeError("`peps=` is only valid when constructing `SymPEPS`.")
        if symmetry is None:
            raise TypeError("`symmetry` is required.")
        if edges is None:
            raise TypeError("`edges` is required.")

        self.psi = state
        self.symmetry = symmetry
        self.edges = edges
        self.fermionic = bool(fermionic)
        self.model = model
        self.hamiltonian = hamiltonian
        self.contraction_opt = contraction_opt
        self.site_ind_id = site_ind_id
        self.gauges = gauges
        self.phys_sectors = phys_sectors
        self.site_charge = site_charge

    def apply_to_arrays(self, fn, *, inplace=True):
        """Apply ``fn`` to each dense array/block in the wrapped state."""
        target = self if inplace else self.copy()
        _apply_to_tensor_network_arrays(target.psi, fn)
        return target

    def to_backend(self, to_backend, *, inplace=True):
        """Convert wrapped state arrays with a backend mapper callable."""
        return self.apply_to_arrays(to_backend, inplace=inplace)

    @property
    def tn(self):
        """The wrapped quimb tensor network."""
        return self.psi

    @property
    def network(self):
        """Compatibility alias for the wrapped quimb tensor network."""
        return self.psi

    @network.setter
    def network(self, value):
        self.psi = value

    def copy(self):
        """Return a shallow configuration copy with a copied tensor network."""
        return type(self)(
            psi=self.psi.copy(),
            symmetry=self.symmetry,
            edges=self.edges,
            fermionic=self.fermionic,
            model=self.model,
            hamiltonian=self.hamiltonian,
            contraction_opt=self.contraction_opt,
            site_ind_id=self.site_ind_id,
            gauges=None if self.gauges is None else dict(self.gauges),
            phys_sectors=None if self.phys_sectors is None else dict(self.phys_sectors),
            site_charge=self.site_charge,
        )

    @property
    def sites(self):
        """Return the sites in the wrapped state."""
        if hasattr(self.psi, "gen_site_coos"):
            return tuple(self.psi.gen_site_coos())
        return tuple(range(self.num_sites))

    def charge_at(self, site):
        """Return the configured local tensor charge for ``site``."""
        if callable(self.site_charge):
            return self.site_charge(site)
        if self.site_charge is None:
            return None
        if isinstance(self.site_charge, dict):
            return self.site_charge[site]
        return self.site_charge

    def site_charges(self):
        """Return ``{site: charge}`` for all sites when charges are configured."""
        return {site: self.charge_at(site) for site in self.sites}

    @staticmethod
    def _add_charges(a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            a_t = a if isinstance(a, tuple) else (a,) * len(b)
            b_t = b if isinstance(b, tuple) else (b,) * len(a)
            return tuple(x + y for x, y in zip(a_t, b_t))
        return a + b

    def overall_charge(self, *, mod=None):
        """Return the sum of configured local tensor charges.

        For U(1), this is the fixed total charge sector represented by the
        local charge pattern. For Z2 parity, use ``overall_parity()`` or pass
        ``mod=2``.
        """
        charges = [charge for charge in self.site_charges().values() if charge is not None]
        if not charges:
            return None
        total = charges[0]
        for charge in charges[1:]:
            total = self._add_charges(total, charge)
        if mod is not None:
            if isinstance(total, tuple):
                return tuple(x % mod for x in total)
            return total % mod
        return total

    def overall_parity(self):
        """Return the configured total Z2 parity, i.e. charge sum modulo 2."""
        return self.overall_charge(mod=2)

    def fermionic_ordering(self):
        """Return site and edge ordering metadata for this Symmray state.

        The metadata records the site order, stored edge order, local bond index
        directions, and the direct-fermion tensor-network reference used by the
        Symmray summaries. For fermionic states, this is the package-level
        record of the graph/order data that Symmray uses for parity-aware
        contractions.
        """
        if self.site_ind_id == "k{}":
            return symmray_mps_summary(self)["fermionic_ordering"]
        if self.site_ind_id == "k{},{}":
            return symmray_peps_summary(self)["fermionic_ordering"]
        raise ValueError("Unsupported symmetric state site index convention.")

    def operator_from_dense(self, array, *, charge=0, sectors=None, sites=None):
        """Convert a dense local observable/operator to this state's symmetry."""
        sectors_use = self.phys_sectors if sectors is None else sectors
        if sectors_use is None:
            raise ValueError("Physical sectors are not known; pass sectors explicitly.")
        return symm_operator_from_dense(
            array,
            sectors_use,
            symmetry=self.symmetry,
            charge=charge,
            fermionic=self.fermionic,
            sites=sites,
        )

    def _site_count_for_where(self, where):
        if self.site_ind_id == "k{}":
            if isinstance(where, Integral):
                return 1
            if isinstance(where, (tuple, list)):
                if len(where) == 1:
                    return 1
                return len(where)
        if self.site_ind_id == "k{},{}":
            if isinstance(where, tuple) and len(where) == 2 and all(isinstance(x, Integral) for x in where):
                return 1
            if isinstance(where, (tuple, list)) and len(where) == 1:
                return 1
            if isinstance(where, (tuple, list)):
                return len(where)
        return 1

    @staticmethod
    def _is_symmray_array(value):
        return _is_symmray_array(value)

    def _coerce_observable(self, obs, where, charge=0):
        if self._is_symmray_array(obs):
            return obs
        return self.operator_from_dense(
            obs,
            charge=charge,
            sites=self._site_count_for_where(where),
        )

    def measure(
        self,
        obs,
        where,
        *,
        charge=0,
        bra=None,
        normalize=True,
        contraction_opt=None,
    ):
        """Measure a generic local observable on this symmetric state.

        Dense observables are automatically converted to Symmray arrays using
        the state's physical sectors. For operators that change charge, pass
        the operator charge explicitly, e.g. ``charge=1`` for a Z2 parity-flip
        operator or ``charge=-1`` for a U(1) lowering operator.
        """
        from .observables import measure_obs  # pylint: disable=import-outside-toplevel

        if isinstance(obs, (list, tuple)):
            if not isinstance(where, (list, tuple)) or len(obs) != len(where):
                raise ValueError("When obs is a sequence, where must be a matching sequence.")
            if isinstance(charge, (list, tuple)):
                if len(charge) != len(obs):
                    raise ValueError("When charge is a sequence, it must match obs length.")
                charges = charge
            else:
                charges = [charge] * len(obs)
            obs_use = [
                self._coerce_observable(obs_i, where_i, charge=charge_i)
                for obs_i, where_i, charge_i in zip(obs, where, charges)
            ]
        else:
            obs_use = self._coerce_observable(obs, where, charge=charge)

        return measure_obs(
            self.psi,
            obs_use,
            where=where,
            ind_id=self.site_ind_id,
            bra=bra,
            normalize=normalize,
            contraction_opt=self.contraction_opt if contraction_opt is None else contraction_opt,
        )

    expectation = measure

    def build_hamiltonian(self, model=None, **params):
        """Build and store a Symmray Hamiltonian for this state's edge set."""
        model_use = _normalize_model(model or self.model or "heisenberg")
        self.hamiltonian = SymHamiltonian.from_edges(
            model_use,
            self.symmetry,
            self.edges,
            **params,
        )
        self.model = model_use
        return self.hamiltonian

    def require_hamiltonian(self, model=None, hamiltonian=None, **params):
        """Resolve an explicit, cached, or newly built Hamiltonian."""
        if hamiltonian is None:
            if model is None and params == {} and self.hamiltonian is not None:
                return self._coerce_hamiltonian(self.hamiltonian)
            return self.build_hamiltonian(model=model, **params)
        if isinstance(hamiltonian, SymHamiltonian):
            return self._coerce_hamiltonian(hamiltonian)
        model_use = _normalize_model(model or self.model or "heisenberg")
        return self._coerce_hamiltonian(
            SymHamiltonian.from_terms(
                model=model_use,
                symmetry=self.symmetry,
                terms=hamiltonian,
                parameters=params,
            )
        )

    def _coerce_hamiltonian(self, hamiltonian):
        """Return ``hamiltonian`` with dense local terms converted to Symmray."""
        terms = {}
        changed = False
        for edge, term in hamiltonian.terms.items():
            if self._is_symmray_array(term):
                terms[edge] = term
                continue
            terms[edge] = self._coerce_observable(term, edge, charge=0)
            changed = True

        if not changed:
            return hamiltonian

        return SymHamiltonian(
            model=hamiltonian.model,
            symmetry=self.symmetry,
            edges=hamiltonian.edges,
            terms=terms,
            parameters=dict(hamiltonian.parameters),
            explicit_terms=hamiltonian.explicit_terms,
        )

    def norm(self, *, contraction_opt=None):
        """Return ``<psi|psi>`` using the configured contraction optimizer."""
        opt = self.contraction_opt if contraction_opt is None else contraction_opt
        return _as_scalar((self.psi.H & self.psi).contract(all, optimize=opt))

    def normalize(self):
        """Normalize the wrapped tensor network in place."""
        normalized = self.psi.normalize()
        # MPS normalization is in-place and returns the old scalar norm,
        # whereas quimb's PEPS normalization returns a new network by
        # default. Keep the wrapper's state synchronized with both APIs.
        if hasattr(normalized, "tensors"):
            self.psi = normalized
        return self

    def trotter_gates(self, dt, *, model=None, hamiltonian=None, imaginary=False, order=1, **params):
        """Return one step of local Trotter gates for this state."""
        ham = self.require_hamiltonian(model=model, hamiltonian=hamiltonian, **params)
        return ham.trotter_gates(dt, imaginary=imaginary, order=order)

    gate_stream = trotter_gates

    def apply_gates(
        self,
        gates,
        *,
        contract="auto",
        max_bond=None,
        cutoff=1e-10,
        normalize=False,
        inplace=True,
        method="direct",
        gauges=None,
        gate_kwargs=None,
        **compress_opts,
    ):
        """Apply a bundled local gate stream to this state."""
        target = self if inplace else self.copy()
        method = str(method).strip().lower()
        contract_auto = contract is None or str(contract).strip().lower() == "auto"
        if max_bond is not None:
            compress_opts.setdefault("max_bond", max_bond)
        if cutoff is not None:
            compress_opts.setdefault("cutoff", cutoff)

        if method == "gate":
            from ..operators import gate as pepsy_gate

            opts = dict(compress_opts)
            if not contract_auto:
                opts.setdefault("contract", contract)
            opts.update({} if gate_kwargs is None else dict(gate_kwargs))
            target.psi = pepsy_gate(
                target.psi,
                tuple(gates),
                inplace=True,
                **opts,
            )
            if normalize:
                target.normalize()
            return target

        if method in {"simple", "gate_simple", "simple_gate"}:
            from ..operators import gate_simple

            gauges_use = gauges
            if gauges_use is None:
                gauges_use = target.gauges if target.gauges is not None else {}
            opts = dict(compress_opts)
            opts.update({} if gate_kwargs is None else dict(gate_kwargs))
            target.psi = gate_simple(
                target.psi,
                tuple(gates),
                gauges=gauges_use,
                inplace=True,
                **opts,
            )
            target.gauges = gauges_use
            if normalize:
                target.normalize()
            return target

        if method in {"loop_cluster", "gate_loop_cluster", "su_loop_cluster"}:
            if type(target).__name__ != "SymPEPS":
                raise ValueError("method='loop_cluster' is only supported for SymPEPS.")
            from ..operators import gate_loop_cluster

            gauges_use = gauges
            if gauges_use is None:
                gauges_use = target.gauges if target.gauges is not None else {}
            opts = dict(compress_opts)
            opts.pop("cutoff", None)
            opts.update({} if gate_kwargs is None else dict(gate_kwargs))
            target.psi = gate_loop_cluster(
                target.psi,
                tuple(gates),
                gauges=gauges_use,
                inplace=True,
                **opts,
            )
            target.gauges = gauges_use
            if normalize:
                target.normalize()
            return target

        if method not in {"direct", "qtn", "tensor_network_gate_inds"}:
            raise ValueError(
                "method must be 'direct', 'gate', 'simple', or 'loop_cluster'."
            )

        for gate, where in gates:
            sites = _sites_from_gate_where(where, target.site_ind_id)
            inds = [_format_site_ind(site, target.site_ind_id) for site in sites]
            qtn.tensor_network_gate_inds(
                target.psi,
                gate,
                inds,
                contract="split" if contract_auto else contract,
                tags=[],
                info=None,
                inplace=True,
                **compress_opts,
            )

        if normalize:
            target.normalize()
        return target

    def time_evolve(
        self,
        dt,
        *,
        steps=1,
        model=None,
        hamiltonian=None,
        imaginary=False,
        order=1,
        max_bond=None,
        cutoff=1e-10,
        normalize=None,
        contract="auto",
        inplace=True,
        method="direct",
        gauges=None,
        gate_kwargs=None,
        **params,
    ):
        """Apply local Trotter time evolution.

        ``imaginary=False`` applies ``exp(-i dt H)``. ``imaginary=True`` applies
        ``exp(-dt H)`` and normalizes after each step by default.
        """
        if not isinstance(steps, Integral) or int(steps) < 1:
            raise ValueError("steps must be a positive integer.")
        target = self if inplace else self.copy()
        normalize_each = bool(imaginary) if normalize is None else bool(normalize)
        ham = target.require_hamiltonian(model=model, hamiltonian=hamiltonian, **params)
        gates = ham.trotter_gates(dt, imaginary=imaginary, order=order)
        for _ in range(int(steps)):
            target.apply_gates(
                gates,
                contract=contract,
                max_bond=max_bond,
                cutoff=cutoff,
                normalize=normalize_each,
                inplace=True,
                method=method,
                gauges=gauges,
                gate_kwargs=gate_kwargs,
            )
        return target

    def ground_state(
        self,
        dt=0.05,
        *,
        steps=20,
        model=None,
        hamiltonian=None,
        order=2,
        max_bond=None,
        cutoff=1e-10,
        inplace=True,
        method="direct",
        gauges=None,
        gate_kwargs=None,
        **params,
    ):
        """Run a simple imaginary-time projection toward a ground state."""
        return self.time_evolve(
            dt,
            steps=steps,
            model=model,
            hamiltonian=hamiltonian,
            imaginary=True,
            order=order,
            max_bond=max_bond,
            cutoff=cutoff,
            normalize=True,
            inplace=inplace,
            method=method,
            gauges=gauges,
            gate_kwargs=gate_kwargs,
            **params,
        )

    def energy(
        self,
        hamiltonian=None,
        *,
        model=None,
        normalized=True,
        contraction_opt=None,
        chi=None,
        measure_kwargs=None,
        boundary_kwargs=None,
        **params,
    ):
        """Estimate ``<psi|H|psi>`` from local Symmray terms.

        For :class:`SymPEPS`, passing ``chi`` or boundary options evaluates
        each local term through :meth:`SymPEPS.measure`, so finite-boundary
        estimates can be reused and controlled by the same boundary API as
        direct observables. Without those options the exact doubled-network
        contraction remains the default. MPS and qMERA callers keep the
        existing exact local-term path.
        """
        ham = self.require_hamiltonian(model=model, hamiltonian=hamiltonian, **params)
        opt = self.contraction_opt if contraction_opt is None else contraction_opt

        if self.site_ind_id == "k{},{}" and (chi is not None or measure_kwargs or boundary_kwargs):
            options = {}
            if boundary_kwargs is not None:
                options.update(dict(boundary_kwargs))
            if measure_kwargs is not None:
                options.update(dict(measure_kwargs))
            if chi is not None:
                options.setdefault("chi", chi)
            options.pop("normalize", None)
            options.setdefault("contraction_opt", opt)
            return _as_scalar(
                sum(
                    self.measure(
                        term,
                        edge,
                        normalize=normalized,
                        **options,
                    )
                    for edge, term in ham.terms.items()
                )
            )

        bra = self.psi.H
        total = 0
        for edge, term in ham.terms.items():
            inds = [_format_site_ind(site, self.site_ind_id) for site in edge]
            gated = qtn.tensor_network_gate_inds(
                self.psi,
                term,
                inds,
                contract="split",
                tags=[],
                info=None,
                inplace=False,
            )
            total = total + (bra | gated).contract(all, optimize=opt)
        total = _as_scalar(total)
        if normalized:
            total = total / self.norm(contraction_opt=opt)
        return _as_scalar(total)

    def energy_density(
        self,
        hamiltonian=None,
        *,
        model=None,
        normalized=True,
        contraction_opt=None,
        chi=None,
        measure_kwargs=None,
        boundary_kwargs=None,
        **params,
    ):
        """Return local-term energy divided by the number of sites."""
        return self.energy(
            hamiltonian=hamiltonian,
            model=model,
            normalized=normalized,
            contraction_opt=contraction_opt,
            chi=chi,
            measure_kwargs=measure_kwargs,
            boundary_kwargs=boundary_kwargs,
            **params,
        ) / self.num_sites



class SymMPS(_SymState):
    """Symmray-backed finite open-chain MPS wrapper."""

    @classmethod
    def random(
        cls,
        L,
        *,
        symmetry="U1",
        bond_dim=4,
        phys_dim=2,
        seed=None,
        dtype="float64",
        fermionic=False,
        site_charge=None,
        subsizes="maximal",
        contraction_opt="auto-hq",
        to_backend=None,
        **kwargs,
    ):
        """Create a raw block-filled random symmetric open-chain MPS."""
        edges = _open_chain_edges(L)
        site_charge_use = _default_site_charge(symmetry) if site_charge is None else site_charge
        phys_sectors = _resolve_phys_sectors(symmetry, phys_dim)
        sr = _require_symmray()
        constructor = sr.TN_fermionic_from_edges_rand if fermionic else sr.TN_abelian_from_edges_rand
        mps = constructor(
            symmetry,
            edges,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            seed=seed,
            dtype=dtype,
            site_tag_id="I{}",
            site_ind_id="k{}",
            site_charge=site_charge_use,
            subsizes=subsizes,
            **kwargs,
        )
        _apply_to_tensor_network_arrays(mps, to_backend)
        mps.view_as_(
            qtn.MatrixProductState,
            L=int(L),
            site_tag_id="I{}",
            site_ind_id="k{}",
            cyclic=False,
        )
        return cls(
            mps=mps,
            symmetry=str(symmetry),
            edges=edges,
            fermionic=bool(fermionic),
            contraction_opt=contraction_opt,
            site_ind_id="k{}",
            phys_sectors=phys_sectors,
            site_charge=site_charge_use,
        )

    @classmethod
    def random_unitary_evolution(
        cls,
        L,
        *,
        symmetry="U1",
        bond_dim=4,
        phys_dim=2,
        seed=None,
        dtype="float64",
        fermionic=False,
        site_charge=None,
        rounds=1000,
        stall_rounds=8,
        cutoff=1e-12,
        contraction_opt="auto-hq",
        to_backend=None,
        **kwargs,
    ):
        """Create a canonical random MPS by growing a product state.

        This mirrors TeNPy's robust random-initial-state construction more
        closely than raw block filling: start from a same-charge product MPS,
        apply random charge-preserving two-site unitaries on alternating
        nearest-neighbor layers, truncate to ``bond_dim``, and canonicalize.
        ``stall_rounds`` stops early when symmetry constraints prevent further
        bond growth.
        """
        bond_dim = int(bond_dim)
        if bond_dim < 1:
            raise ValueError("bond_dim must be a positive integer.")
        rounds = int(rounds)
        if rounds < 1:
            raise ValueError("rounds must be a positive integer.")
        if stall_rounds is not None:
            stall_rounds = int(stall_rounds)
            if stall_rounds < 1:
                raise ValueError("stall_rounds must be positive or None.")

        site_charge_use = (
            _default_site_charge(symmetry) if site_charge is None else site_charge
        )
        phys_sectors = _resolve_phys_sectors(symmetry, phys_dim)
        state = cls.random(
            L,
            symmetry=symmetry,
            bond_dim=1,
            phys_dim=phys_dim,
            seed=seed,
            dtype=dtype,
            fermionic=fermionic,
            site_charge=site_charge_use,
            subsizes="maximal",
            contraction_opt=contraction_opt,
            **kwargs,
        )
        if int(L) < 2 or bond_dim <= 1:
            state.psi = _right_canonize_mps(state.psi)
            state.normalize()
            _apply_to_tensor_network_arrays(state.psi, to_backend)
            return state

        rng = np.random.default_rng(seed)
        best_bond = int(state.psi.max_bond())
        stalled = 0
        for _ in range(rounds):
            for parity in (0, 1):
                gates = []
                for site in range(parity, int(L) - 1, 2):
                    gate_dense = _random_charge_preserving_two_site_dense(
                        phys_sectors,
                        symmetry,
                        rng,
                        dtype,
                    )
                    gates.append(
                        (
                            symm_operator_from_dense(
                                gate_dense,
                                phys_sectors,
                                symmetry=symmetry,
                                charge=_zero_like_charge(next(iter(phys_sectors))),
                                fermionic=fermionic,
                                sites=2,
                            ),
                            (site, site + 1),
                        )
                    )
                if gates:
                    state.apply_gates(
                        gates,
                        method="direct",
                        contract="split",
                        max_bond=bond_dim,
                        cutoff=cutoff,
                        normalize=True,
                        inplace=True,
                    )
            state.psi = _right_canonize_mps(state.psi)
            current_bond = int(state.psi.max_bond())
            if current_bond >= bond_dim:
                break
            if current_bond > best_bond:
                best_bond = current_bond
                stalled = 0
            else:
                stalled += 1
                if stall_rounds is not None and stalled >= stall_rounds:
                    break
        state.psi = _right_canonize_mps(state.psi)
        state.normalize()
        _apply_to_tensor_network_arrays(state.psi, to_backend)
        return state

    @classmethod
    def random_unitary_for_model(
        cls, model, L, *, symmetry=None, fermionic=None, phys_dim=None, **kwargs
    ):
        """Create a random-unitary MPS with defaults suitable for a model."""
        model_norm = _normalize_model(model)
        defaults = _MODEL_DEFAULTS[model_norm]
        state = cls.random_unitary_evolution(
            L,
            symmetry=defaults["symmetry"] if symmetry is None else symmetry,
            fermionic=defaults["fermionic"] if fermionic is None else fermionic,
            phys_dim=defaults["phys_dim"] if phys_dim is None else phys_dim,
            **kwargs,
        )
        state.model = model_norm
        return state

    @classmethod
    def for_model(cls, model, L, *, symmetry=None, fermionic=None, phys_dim=None, **kwargs):
        """Create a raw random MPS with defaults suitable for a named model."""
        model_norm = _normalize_model(model)
        defaults = _MODEL_DEFAULTS[model_norm]
        state = cls.random(
            L,
            symmetry=defaults["symmetry"] if symmetry is None else symmetry,
            fermionic=defaults["fermionic"] if fermionic is None else fermionic,
            phys_dim=defaults["phys_dim"] if phys_dim is None else phys_dim,
            **kwargs,
        )
        state.model = model_norm
        return state

    @property
    def mps(self):
        """The wrapped quimb matrix-product state."""
        return self.psi

    @mps.setter
    def mps(self, value):
        self.psi = value

    def time_evolve_mps_optimizer(
        self,
        dt,
        *,
        steps=1,
        model=None,
        hamiltonian=None,
        imaginary=False,
        order=1,
        chi=None,
        mode="mpo",
        cutoff=1e-10,
        inplace=True,
        optimizer_kwargs=None,
        run_kwargs=None,
        **params,
    ):
        """Apply a Symmray gate stream through :class:`pepsy.MpsOptimizer`.

        This is useful for checking that a symmetry-preserving local gate stream
        can drive the existing MPS optimizer backends such as ``mode="mpo"``.
        """
        if not isinstance(steps, Integral) or int(steps) < 1:
            raise ValueError("steps must be a positive integer.")
        from ..optimizers import MpsOptimizer

        target = self if inplace else self.copy()
        ham = target.require_hamiltonian(model=model, hamiltonian=hamiltonian, **params)
        stream = ham.gate_stream(dt, imaginary=imaginary, order=order).repeat(int(steps))
        chi_use = target.psi.max_bond() if chi is None else int(chi)
        opt_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        opt = MpsOptimizer(
            target.psi,
            stream,
            chi=chi_use,
            mode=mode,
            inplace=True,
            **opt_kwargs,
        )
        run_opts = {
            "progbar": False,
            "cutoff": cutoff,
        }
        if imaginary:
            run_opts.update(
                {
                    "non_unitary": True,
                    "normalize_every": True,
                    "normalize_final": True,
                }
            )
        if run_kwargs is not None:
            run_opts.update(dict(run_kwargs))
        target.psi = opt.run(**run_opts)
        return target

    @property
    def num_sites(self):
        """Number of MPS sites."""
        return int(self.psi.L)

    @property
    def L(self):
        """Number of MPS sites."""
        return self.num_sites



class SymPEPS(_SymState):
    """Symmray-backed finite 2D PEPS wrapper."""

    @classmethod
    def random(
        cls,
        Lx,
        Ly,
        *,
        symmetry="U1",
        bond_dim=2,
        phys_dim=2,
        cyclic=False,
        seed=None,
        dtype="float64",
        fermionic=False,
        site_charge=None,
        subsizes="maximal",
        contraction_opt="auto-hq",
        to_backend=None,
        **kwargs,
    ):
        """Create a random symmetric 2D PEPS."""
        site_charge_use = _default_site_charge(symmetry) if site_charge is None else site_charge
        phys_sectors = _resolve_phys_sectors(symmetry, phys_dim)
        sr = _require_symmray()
        constructor = sr.PEPS_fermionic_rand if fermionic else sr.PEPS_abelian_rand
        peps = constructor(
            symmetry,
            Lx=int(Lx),
            Ly=int(Ly),
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            cyclic=cyclic,
            seed=seed,
            dtype=dtype,
            site_tag_id="I{},{}",
            site_ind_id="k{},{}",
            x_tag_id="X{}",
            y_tag_id="Y{}",
            site_charge=site_charge_use,
            subsizes=subsizes,
            **kwargs,
        )
        _apply_to_tensor_network_arrays(peps, to_backend)
        edges = _as_edges(qtn.edges_2d_square(int(Lx), int(Ly), cyclic=cyclic))
        return cls(
            peps=peps,
            symmetry=str(symmetry),
            edges=edges,
            fermionic=bool(fermionic),
            contraction_opt=contraction_opt,
            site_ind_id="k{},{}",
            phys_sectors=phys_sectors,
            site_charge=site_charge_use,
        )

    @classmethod
    def for_model(cls, model, Lx, Ly, *, symmetry=None, fermionic=None, phys_dim=None, **kwargs):
        """Create a random PEPS with defaults suitable for a named model."""
        model_norm = _normalize_model(model)
        defaults = _MODEL_DEFAULTS[model_norm]
        state = cls.random(
            Lx,
            Ly,
            symmetry=defaults["symmetry"] if symmetry is None else symmetry,
            fermionic=defaults["fermionic"] if fermionic is None else fermionic,
            phys_dim=defaults["phys_dim"] if phys_dim is None else phys_dim,
            **kwargs,
        )
        state.model = model_norm
        return state

    @property
    def peps(self):
        """The wrapped quimb projected-entangled pair state."""
        return self.psi

    @peps.setter
    def peps(self, value):
        self.psi = value

    @staticmethod
    def _is_site_coordinate(site):
        return (
            isinstance(site, tuple)
            and len(site) == 2
            and all(isinstance(x, Integral) for x in site)
        )

    def _sites_from_where(self, where):
        """Normalize PEPS one-/two-site selectors to coordinate tuples."""
        if self._is_site_coordinate(where):
            return (tuple(int(x) for x in where),)
        if not isinstance(where, (list, tuple)):
            raise TypeError("PEPS where must be a coordinate or a sequence of coordinates.")
        if len(where) == 0:
            raise ValueError("PEPS where must select at least one site.")
        if len(where) == 1 and self._is_site_coordinate(where[0]):
            return (tuple(int(x) for x in where[0]),)

        sites = tuple(tuple(int(x) for x in site) for site in where)
        if not all(self._is_site_coordinate(site) for site in sites):
            raise TypeError("PEPS where entries must be two-integer coordinates.")
        if len(sites) > 2:
            raise ValueError("SymPEPS.measure currently supports one- and two-site observables.")
        return sites

    @staticmethod
    def _validate_boundary_chi(chi):
        if chi is None:
            return None
        if not isinstance(chi, Integral):
            raise TypeError("chi must be an integer when provided.")
        chi = int(chi)
        if chi < 1:
            raise ValueError("chi must be >= 1 when provided.")
        return chi

    @staticmethod
    def _where_key_from_sites(sites):
        return sites[0] if len(sites) == 1 else tuple(sites)

    def _single_quimb_term(self, obs, where, charge):
        sites = self._sites_from_where(where)
        obs_use = self._coerce_observable(obs, where, charge=charge)
        return {self._where_key_from_sites(sites): obs_use}

    def _quimb_plaquette_env_options(
        self,
        *,
        progress,
        equalize_norms,
        first_contract,
        second_dense,
        compress_opts,
    ):
        _ = progress
        opts = {}
        if equalize_norms is not False:
            opts["equalize_norms"] = equalize_norms
        if first_contract is not None:
            opts["first_contract"] = first_contract
        if second_dense is not None:
            opts["second_dense"] = second_dense
        if compress_opts is not None:
            opts["compress_opts"] = compress_opts
        return opts

    def _resolve_quimb_plaquette_envs(
        self,
        terms,
        *,
        chi,
        bdy,
        plaquette_envs,
        plaquette_map,
        cutoff,
        canonize,
        mode,
        layer_tags,
        autogroup,
        progress,
        equalize_norms,
        first_contract,
        second_dense,
        compress_opts,
    ):
        try:
            from quimb.tensor.tn2d.core import (  # pylint: disable=import-outside-toplevel
                calc_plaquette_map,
                calc_plaquette_sizes,
            )
        except ModuleNotFoundError:
            # Older quimb releases expose the PEPS boundary contraction
            # methods but not the plaquette-environment helper module. Let
            # ``compute_local_expectation`` perform its supported fallback.
            if chi is None and plaquette_envs is None:
                raise ValueError(
                    "Provide chi when quimb plaquette environments are not supplied."
                )
            if isinstance(bdy, dict):
                bdy.setdefault("plaquette_envs", {})
                bdy.setdefault("plaquette_map", {})
                bdy.setdefault("chi", chi)
            return None, None

        holder = bdy if isinstance(bdy, dict) else None
        if bdy is not None and holder is None:
            raise TypeError("bdy must be a dict holder for quimb plaquette environments.")

        if holder is not None:
            if plaquette_envs is None:
                plaquette_envs = holder.get("plaquette_envs")
            if plaquette_map is None:
                plaquette_map = holder.get("plaquette_map")

        if plaquette_envs is None:
            if chi is None:
                raise ValueError("Provide chi when quimb plaquette environments are not supplied.")
            env_options = self._quimb_plaquette_env_options(
                progress=progress,
                equalize_norms=equalize_norms,
                first_contract=first_contract,
                second_dense=second_dense,
                compress_opts=compress_opts,
            )
            norm_tn = self.psi.make_norm(layer_tags=layer_tags)
            plaquette_envs = {}
            for x_bsz, y_bsz in calc_plaquette_sizes(terms.keys(), autogroup):
                plaquette_envs.update(
                    call_quimb_2d(
                        norm_tn.compute_plaquette_environments,
                        x_bsz=x_bsz,
                        y_bsz=y_bsz,
                        max_bond=chi,
                        cutoff=cutoff,
                        canonize=canonize,
                        mode=mode,
                        layer_tags=layer_tags,
                        **env_options,
                    )
                )
            plaquette_map = calc_plaquette_map(plaquette_envs)
            if holder is not None:
                holder["plaquette_envs"] = plaquette_envs
                holder["plaquette_map"] = plaquette_map
                holder["chi"] = chi
                holder["mode"] = mode
        elif plaquette_map is None:
            plaquette_map = calc_plaquette_map(plaquette_envs)
            if holder is not None:
                holder["plaquette_map"] = plaquette_map

        return plaquette_envs, plaquette_map

    def _contract_quimb_double_layer(
        self,
        double_layer,
        *,
        chi,
        cutoff,
        canonize,
        mode,
        layer_tags,
        contraction_opt,
        max_separation,
        progress,
        equalize_norms,
    ):
        if chi is None:
            raise ValueError("Provide chi for quimb boundary contraction.")
        final_contract_opts = {"optimize": contraction_opt}
        if mode == "ctmrg":
            return _as_scalar(
                call_quimb_2d(
                    double_layer.contract_ctmrg,
                    max_bond=chi,
                    cutoff=cutoff,
                    canonize=canonize,
                    mode="projector",
                    max_separation=max_separation,
                    equalize_norms=equalize_norms,
                    final_contract=True,
                    final_contract_opts=final_contract_opts,
                    progbar=progress,
                )
            )
        return _as_scalar(
            call_quimb_2d(
                double_layer.contract_boundary,
                max_bond=chi,
                cutoff=cutoff,
                canonize=canonize,
                mode=mode,
                layer_tags=layer_tags,
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                final_contract=True,
                final_contract_opts=final_contract_opts,
                progbar=progress,
            )
        )

    def _measure_quimb_overlap(
        self,
        measurement_terms,
        *,
        bra,
        normalize,
        norm,
        contraction_opt,
        chi,
        mode,
        layer_tags,
        cutoff,
        cutoff_mode,
        canonize,
        max_separation,
        progress,
        equalize_norms,
    ):
        ket_obs = self.psi.copy()
        for obs_i, where_i, charge_i in measurement_terms:
            sites = self._sites_from_where(where_i)
            obs_use = self._coerce_observable(obs_i, where_i, charge=charge_i)
            inds = [_format_site_ind(site, self.site_ind_id) for site in sites]
            qtn.tensor_network_gate_inds(
                ket_obs,
                obs_use,
                inds,
                contract=True if len(sites) == 1 else "split",
                tags=[],
                info=None,
                inplace=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )

        if bra is None:
            bra_network = self.psi
        elif isinstance(bra, _SymState):
            bra_network = bra.psi
        else:
            bra_network = bra

        numer_tn = ket_obs.make_overlap(bra_network, layer_tags=layer_tags)
        numerator = self._contract_quimb_double_layer(
            numer_tn,
            chi=chi,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            contraction_opt=contraction_opt,
            max_separation=max_separation,
            progress=progress,
            equalize_norms=equalize_norms,
        )
        if bra is not None or not normalize:
            return numerator

        if norm is None:
            denom_tn = self.psi.make_norm(layer_tags=layer_tags)
            norm = self._contract_quimb_double_layer(
                denom_tn,
                chi=chi,
                cutoff=cutoff,
                canonize=canonize,
                mode=mode,
                layer_tags=layer_tags,
                contraction_opt=contraction_opt,
                max_separation=max_separation,
                progress=progress,
                equalize_norms=equalize_norms,
            )
        if norm == 0.0:
            raise ValueError("Cannot compute normalized observable for a zero-norm state.")
        return _as_scalar(numerator / norm)

    def _measurement_terms(self, obs, where, charge):
        if isinstance(obs, (list, tuple)):
            if not isinstance(where, (list, tuple)) or len(obs) != len(where):
                raise ValueError("When obs is a sequence, where must be a matching sequence.")
            if isinstance(charge, (list, tuple)):
                if len(charge) != len(obs):
                    raise ValueError("When charge is a sequence, it must match obs length.")
                charges = charge
            else:
                charges = [charge] * len(obs)
            return tuple(zip(obs, where, charges))
        return ((obs, where, charge),)

    def measure(
        self,
        obs,
        where,
        *,
        charge=0,
        bra=None,
        normalize=True,
        norm=None,
        contraction_opt=None,
        chi=None,
        bdy=None,
        bdy_norm=None,
        n_iter=10,
        direction="y",
        max_separation=1,
        progress=False,
        track_boundary_fidelity=False,
        fit_mode="eff",
        single_layer=False,
        visualize=False,
        equalize_norms=False,
        cutoff=1.0e-12,
        cutoff_mode="rsum2",
        mode="mps",
        route="boundary",
        canonize=True,
        autogroup=True,
        layer_tags=("KET", "BRA"),
        plaquette_envs=None,
        plaquette_map=None,
        first_contract=None,
        second_dense=None,
        compress_opts=None,
    ):
        """Measure local PEPS observables via quimb PEPS boundary contraction.

        Dense observables are first converted to Symmray arrays, then quimb's
        PEPS plaquette-environment machinery measures one local term with
        ``compute_local_expectation(..., max_bond=chi)``. For cross-bra
        overlaps or multiple observable insertions, the observable is applied
        explicitly and the resulting double layer is contracted with quimb's
        boundary methods.
        """
        opt = self.contraction_opt if contraction_opt is None else contraction_opt
        chi = self._validate_boundary_chi(chi)
        layer_tags_use = None if single_layer else layer_tags
        measurement_terms = self._measurement_terms(obs, where, charge)
        mode_local = "projector" if mode == "ctmrg" else mode

        # These arguments belonged to the older PEPSY BdyMPS path. Keep them
        # accepted for compatibility, but let quimb choose its sweep details.
        _ = (bdy_norm, n_iter, direction, track_boundary_fidelity, fit_mode, visualize)

        if bra is not None or len(measurement_terms) != 1:
            if route != "boundary":
                raise ValueError("The measurement route option applies to single local terms.")
            return self._measure_quimb_overlap(
                measurement_terms,
                bra=bra,
                normalize=normalize,
                norm=norm,
                contraction_opt=opt,
                chi=chi,
                mode=mode,
                layer_tags=layer_tags_use,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                canonize=canonize,
                max_separation=max_separation,
                progress=progress,
                equalize_norms=equalize_norms,
            )

        obs_i, where_i, charge_i = measurement_terms[0]
        terms = self._single_quimb_term(obs_i, where_i, charge_i)
        single_line = min(self.psi.Lx, self.psi.Ly) == 1
        if single_line and not bdy:
            bdy = None
        if not single_line and chi is None and plaquette_envs is None and not (
            isinstance(bdy, dict) and bdy.get("plaquette_envs") is not None
        ):
            raise ValueError("Provide chi when quimb plaquette environments are not supplied.")

        if route != "boundary" and (bdy is not None or plaquette_envs is not None):
            raise ValueError("Precomputed boundary plaquettes require route='boundary'.")
        if bdy is not None or plaquette_envs is not None:
            plaquette_envs, plaquette_map = self._resolve_quimb_plaquette_envs(
                terms,
                chi=chi,
                bdy=bdy,
                plaquette_envs=plaquette_envs,
                plaquette_map=plaquette_map,
                cutoff=cutoff,
                canonize=canonize,
                mode=mode_local,
                layer_tags=layer_tags_use,
                autogroup=autogroup,
                progress=progress,
                equalize_norms=equalize_norms,
                first_contract=first_contract,
                second_dense=second_dense,
                compress_opts=compress_opts,
            )
        else:
            plaquette_env_options = self._quimb_plaquette_env_options(
                progress=progress,
                equalize_norms=equalize_norms,
                first_contract=first_contract,
                second_dense=second_dense,
                compress_opts=compress_opts,
            )

        # Some supported quimb/symmray combinations route projector/CTMRG
        # boundary compression through a dense reshape that Symmray's
        # BlockVector does not implement. The MPS boundary path computes the
        # same local expectation while retaining the native operator blocks.
        if mode_local == "projector" and self._is_symmray_array(next(iter(terms.values()))):
            mode_local = "mps"

        value = compute_peps_local_expectation(
            self.psi,
            terms,
            max_bond=chi,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode_local,
            route=route,
            layer_tags=layer_tags_use,
            normalized=bool(normalize and norm is None),
            autogroup=autogroup,
            contract_optimize=opt,
            plaquette_envs=plaquette_envs,
            plaquette_map=plaquette_map,
            **({} if bdy is not None or plaquette_envs is not None else plaquette_env_options),
        )
        if normalize and norm is not None:
            if norm == 0.0:
                raise ValueError("Cannot compute normalized observable for a zero-norm state.")
            value = value / norm
        return _as_scalar(value)

    expectation = measure

    @property
    def num_sites(self):
        """Number of PEPS sites."""
        return int(self.psi.Lx) * int(self.psi.Ly)

    @property
    def Lx(self):
        """PEPS x dimension."""
        return int(self.psi.Lx)

    @property
    def Ly(self):
        """PEPS y dimension."""
        return int(self.psi.Ly)
