"""Symmray-backed symmetric MPS and PEPS convenience wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

__all__ = ["SymGateStream", "SymHamiltonian", "SymMPS", "SymPEPS"]
__all__ += [
    "default_physical_sectors",
    "sector_index_map",
    "site_charge_alternating",
    "site_charge_from_map",
    "site_charge_from_occupations",
    "site_charge_uniform",
    "symm_operator_from_dense",
]

_SYMMRAY_AUTORAY_REGISTERED = False


def _require_symmray():
    """Import symmray with a clear optional-dependency message."""
    try:
        import symmray as sr
    except ImportError as exc:  # pragma: no cover - exercised without symmray
        raise ImportError(
            "SymMPS and SymPEPS require the optional dependency `symmray`. "
            "Install it with `pip install symmray`."
        ) from exc
    _register_symmray_autoray_compat()
    return sr


def _to_dense(value):
    return value.to_dense() if hasattr(value, "to_dense") else value


def _is_symmray_array(value):
    return hasattr(value, "blocks") and hasattr(value, "indices")


def _register_symmray_autoray_compat():
    """Register tiny creation/comparison shims used by quimb canonicalization."""
    global _SYMMRAY_AUTORAY_REGISTERED  # pylint: disable=global-statement
    if _SYMMRAY_AUTORAY_REGISTERED:
        return

    def _eye(n, m=None, k=0, dtype=None, **_):
        return np.eye(n, n if m is None else m, k=k, dtype=dtype)

    def _allclose(a, b, rtol=1e-5, atol=1e-8, **kwargs):
        return np.allclose(_to_dense(a), _to_dense(b), rtol=rtol, atol=atol, **kwargs)

    ar.register_function("symmray", "eye", _eye)
    ar.register_function("symmray", "allclose", _allclose)
    _SYMMRAY_AUTORAY_REGISTERED = True


_MODEL_ALIASES = {
    "tfim": "tfim",
    "itf": "tfim",
    "ising": "tfim",
    "transverse_field_ising": "tfim",
    "transverse-field-ising": "tfim",
    "heis": "heisenberg",
    "heisenberg": "heisenberg",
    "fermi_hubbard": "fermi_hubbard",
    "fermi-hubbard": "fermi_hubbard",
    "hubbard": "fermi_hubbard",
    "fh": "fermi_hubbard",
    "spinless_fermi_hubbard": "fermi_hubbard_spinless",
    "spinless-fermi-hubbard": "fermi_hubbard_spinless",
    "fermi_hubbard_spinless": "fermi_hubbard_spinless",
    "fermi-hubbard-spinless": "fermi_hubbard_spinless",
    "tv": "fermi_hubbard_spinless",
    "t-v": "fermi_hubbard_spinless",
}

_MODEL_DEFAULTS = {
    "tfim": {"symmetry": "Z2", "fermionic": False, "phys_dim": 2},
    "heisenberg": {"symmetry": "U1", "fermionic": False, "phys_dim": 2},
    "fermi_hubbard": {"symmetry": "U1", "fermionic": True, "phys_dim": 4},
    "fermi_hubbard_spinless": {"symmetry": "U1", "fermionic": True, "phys_dim": 2},
}

_DEFAULT_PHYS_SECTORS = {
    ("Z2", 2): {0: 1, 1: 1},
    ("U1", 2): {0: 1, 1: 1},
    ("Z2", 4): {0: 2, 1: 2},
    ("U1", 4): {0: 1, 1: 2, 2: 1},
    ("Z2Z2", 4): {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    ("U1U1", 4): {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
}


def default_physical_sectors(symmetry=None, phys_dim=None, *, model=None):
    """Return the default physical charge-sector map.

    Examples
    --------
    ``default_physical_sectors("U1", 2)`` returns ``{0: 1, 1: 1}``.
    ``default_physical_sectors("U1", 4)`` returns the spinful fermion sectors
    ``{0: 1, 1: 2, 2: 1}``.
    """
    if model is not None:
        defaults = _MODEL_DEFAULTS[_normalize_model(model)]
        if symmetry is None:
            symmetry = defaults["symmetry"]
        if phys_dim is None:
            phys_dim = defaults["phys_dim"]
    key = (str(symmetry), int(phys_dim))
    try:
        return dict(_DEFAULT_PHYS_SECTORS[key])
    except KeyError as exc:
        raise ValueError(f"No default physical sectors for symmetry/phys_dim {key!r}.") from exc


def sector_index_map(sectors):
    """Expand ``{charge: size}`` sectors to ``{dense_index: charge}``."""
    out = {}
    dense_index = 0
    for charge, size in dict(sectors).items():
        if int(size) < 1:
            raise ValueError("Sector sizes must be positive integers.")
        for _ in range(int(size)):
            out[dense_index] = charge
            dense_index += 1
    return out


def _site_parity(site):
    if isinstance(site, (tuple, list)):
        return sum(int(x) for x in site) % 2
    return int(site) % 2


def site_charge_uniform(charge=0):
    """Return a site-charge function with the same charge on every site."""

    def _site_charge(_site):
        return charge

    return _site_charge


def site_charge_alternating(even=0, odd=1):
    """Return a checkerboard/alternating site-charge function.

    For 1D sites, even/odd means ``site % 2``. For PEPS coordinates it means
    ``sum(site_coordinate) % 2``.
    """

    def _site_charge(site):
        return odd if _site_parity(site) else even

    return _site_charge


def site_charge_from_map(mapping, *, default=None):
    """Return a site-charge function backed by an explicit ``{site: charge}`` map."""
    charges = dict(mapping)

    def _site_charge(site):
        if site in charges:
            return charges[site]
        if default is not None:
            return default
        raise KeyError(f"No site charge supplied for site {site!r}.")

    return _site_charge


def site_charge_from_occupations(occupations, *, default=None):
    """Return a site-charge function from occupation/charge labels.

    ``occupations`` can be a 1D sequence such as ``[1, 0, 1, 0]`` or an
    explicit mapping such as ``{(0, 0): 1, (0, 1): 0}``. The total U(1) charge
    or Z2 parity is the sum of these values, with Z2 understood modulo 2.
    """
    if isinstance(occupations, dict):
        return site_charge_from_map(occupations, default=default)
    return site_charge_from_map({i: charge for i, charge in enumerate(occupations)}, default=default)


def _array_class_for_symmetry(symmetry, *, fermionic=False):
    sr = _require_symmray()
    name = str(symmetry)
    if name == "U1":
        return sr.U1FermionicArray if fermionic else sr.U1Array
    if name == "Z2":
        return sr.Z2FermionicArray if fermionic else sr.Z2Array
    if name == "U1U1":
        return sr.U1U1FermionicArray if fermionic else sr.U1U1Array
    if name == "Z2Z2":
        return sr.Z2Z2FermionicArray if fermionic else sr.Z2Z2Array
    return sr.FermionicArray if fermionic else sr.AbelianArray


def symm_operator_from_dense(
    array,
    sectors,
    *,
    symmetry="U1",
    charge=0,
    fermionic=False,
    sites=None,
):
    """Convert a dense local operator to a Symmray block-sparse array.

    Parameters
    ----------
    array : array_like
        Dense one- or two-site operator. Rank-2 arrays are treated as one-site
        operators unless ``sites=2`` is supplied, in which case they are
        reshaped from ``(d**2, d**2)`` to ``(d, d, d, d)``.
    sectors : dict
        Physical charge-sector map, for example ``{0: 1, 1: 1}``.
    symmetry, charge, fermionic
        Symmray array metadata. Use ``charge=0`` for number/diagonal
        observables, ``charge=1`` for Z2 parity-flipping operators, and
        ``charge=+/-1`` for U(1) raising/lowering-style operators.
    sites : int | None
        Number of local sites acted on. Inferred from rank when omitted.
    """
    arr = np.asarray(array)
    sectors = dict(sectors)
    phys_dim = sum(int(size) for size in sectors.values())
    if sites is None:
        if arr.ndim == 2:
            sites = 1
        elif arr.ndim == 4:
            sites = 2
        else:
            raise ValueError("sites must be supplied for dense operators not rank 2 or 4.")
    sites = int(sites)
    if sites < 1:
        raise ValueError("sites must be a positive integer.")
    if arr.ndim == 2 and sites > 1:
        arr = arr.reshape((phys_dim,) * sites * 2)
    expected_shape = (phys_dim,) * sites * 2
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"Operator shape {arr.shape} does not match expected {expected_shape}.")

    index_map = sector_index_map(sectors)
    index_maps = tuple(dict(index_map) for _ in range(2 * sites))
    duals = (False,) * sites + (True,) * sites
    array_cls = _array_class_for_symmetry(symmetry, fermionic=fermionic)
    kwargs = {}
    if array_cls.__name__ in {"AbelianArray", "FermionicArray"}:
        kwargs["symmetry"] = symmetry
    return array_cls.from_dense(
        arr,
        index_maps=index_maps,
        duals=duals,
        charge=charge,
        **kwargs,
    )


class SymGateStream(tuple):
    """Tuple-like bundled stream of Symmray local gates."""

    def __new__(
        cls,
        entries=(),
        *,
        hamiltonian=None,
        dt=None,
        imaginary=False,
        order=1,
    ):
        obj = super().__new__(cls, tuple(entries))
        obj.hamiltonian = hamiltonian
        obj.dt = dt
        obj.imaginary = bool(imaginary)
        obj.order = int(order)
        return obj

    def repeat(self, steps):
        """Return a stream with this step repeated ``steps`` times."""
        if not isinstance(steps, Integral) or int(steps) < 1:
            raise ValueError("steps must be a positive integer.")
        return type(self)(
            tuple(self) * int(steps),
            hamiltonian=self.hamiltonian,
            dt=self.dt,
            imaginary=self.imaginary,
            order=self.order,
        )


def _normalize_model(model):
    key = str(model).strip().lower().replace(" ", "_")
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_MODEL_ALIASES)))
        raise ValueError(f"Unknown symmetric model {model!r}. Expected one of: {allowed}.") from exc


def _default_site_charge(symmetry):
    symmetry = str(symmetry)
    if symmetry == "U1":
        return site_charge_alternating(0, 1)
    if symmetry.startswith("Z"):
        return site_charge_uniform(0)
    return None


def _resolve_phys_sectors(symmetry, phys_dim):
    if phys_dim is None:
        return None
    if isinstance(phys_dim, dict):
        return dict(phys_dim)
    if isinstance(phys_dim, Integral):
        return default_physical_sectors(symmetry, int(phys_dim))
    return None


def _open_chain_edges(length):
    if not isinstance(length, Integral):
        raise TypeError("length must be an integer.")
    length = int(length)
    if length < 2:
        raise ValueError("length must be >= 2.")
    return tuple((i, i + 1) for i in range(length - 1))


def _as_edges(edges):
    out = tuple(tuple(edge) for edge in edges)
    if not out:
        raise ValueError("At least one edge is required.")
    if any(len(edge) != 2 for edge in out):
        raise ValueError("Each edge must connect exactly two sites.")
    return out


def _format_site_ind(site, site_ind_id):
    if isinstance(site, tuple):
        return site_ind_id.format(*site)
    return site_ind_id.format(site)


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _hamiltonian_from_edges(model, symmetry, edges, *, flat=False, **params):
    sr = _require_symmray()
    model = _normalize_model(model)
    if model == "tfim":
        return sr.ham_tfim_from_edges(symmetry, edges, flat=flat, **params)
    if model == "heisenberg":
        return sr.ham_heisenberg_from_edges(symmetry, edges, flat=flat, **params)
    if model == "fermi_hubbard":
        return sr.ham_fermi_hubbard_from_edges(symmetry, edges, flat=flat, **params)
    if model == "fermi_hubbard_spinless":
        return sr.ham_fermi_hubbard_spinless_from_edges(symmetry, edges, flat=flat, **params)
    raise AssertionError(f"Unhandled model {model!r}.")


def _gate_from_term(term, dt, *, imaginary=False):
    """Exponentiate a two-site local Hamiltonian term."""
    shape = tuple(int(d) for d in term.shape)
    if len(shape) != 4 or shape[0] != shape[2] or shape[1] != shape[3]:
        raise ValueError("Only two-site Hamiltonian terms with shape (da, db, da, db) are supported.")
    matrix_shape = (shape[0] * shape[1], shape[2] * shape[3])
    scale = -dt if imaginary else -1j * dt
    return ar.do("linalg.expm", scale * term.reshape(matrix_shape)).reshape(shape)


@dataclass(frozen=True)
class SymHamiltonian:
    """Container for Symmray local two-site Hamiltonian terms."""

    model: str
    symmetry: str
    edges: tuple
    terms: dict
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_edges(cls, model, symmetry, edges, *, flat=False, **params):
        """Build a Symmray Hamiltonian dictionary from lattice edges."""
        model_norm = _normalize_model(model)
        edges = _as_edges(edges)
        terms = _hamiltonian_from_edges(model_norm, symmetry, edges, flat=flat, **params)
        return cls(
            model=model_norm,
            symmetry=str(symmetry),
            edges=edges,
            terms=dict(terms),
            parameters=dict(params),
        )

    def trotter_gates(self, dt, *, imaginary=False, order=1):
        """Return local gate entries ``[(gate, edge), ...]`` for one Trotter step."""
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2.")
        entries = list(self.terms.items())
        if order == 1:
            gates = [(_gate_from_term(term, dt, imaginary=imaginary), edge) for edge, term in entries]
            return SymGateStream(
                gates,
                hamiltonian=self,
                dt=dt,
                imaginary=imaginary,
                order=order,
            )

        half = dt / 2
        forward = [(_gate_from_term(term, half, imaginary=imaginary), edge) for edge, term in entries]
        backward = [(_gate_from_term(term, half, imaginary=imaginary), edge) for edge, term in reversed(entries)]
        return SymGateStream(
            forward + backward,
            hamiltonian=self,
            dt=dt,
            imaginary=imaginary,
            order=order,
        )

    gate_stream = trotter_gates


@dataclass
class _SymState:
    """Shared implementation for symmetric tensor-network states."""

    network: qtn.TensorNetwork
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

    @property
    def tn(self):
        """The wrapped quimb tensor network."""
        return self.network

    @property
    def psi(self):
        """Alias for the wrapped state."""
        return self.network

    def copy(self):
        """Return a shallow configuration copy with a copied tensor network."""
        return type(self)(
            network=self.network.copy(),
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
        if hasattr(self.network, "gen_site_coos"):
            return tuple(self.network.gen_site_coos())
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
        from .core import measure_obs  # pylint: disable=import-outside-toplevel

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
            self.network,
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
            SymHamiltonian(
                model=model_use,
                symmetry=self.symmetry,
                edges=_as_edges(hamiltonian.keys()),
                terms=dict(hamiltonian),
                parameters=dict(params),
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
        )

    def norm(self, *, contraction_opt=None):
        """Return ``<psi|psi>`` using the configured contraction optimizer."""
        opt = self.contraction_opt if contraction_opt is None else contraction_opt
        return _as_scalar((self.network.H & self.network).contract(all, optimize=opt))

    def normalize(self):
        """Normalize the wrapped tensor network in place."""
        self.network.normalize()
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
        contract="split",
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
        if max_bond is not None:
            compress_opts.setdefault("max_bond", max_bond)
        if cutoff is not None:
            compress_opts.setdefault("cutoff", cutoff)

        if method == "gate":
            from ..operators import gate as pepsy_gate

            opts = dict(compress_opts)
            opts.setdefault("contract", contract)
            opts.update({} if gate_kwargs is None else dict(gate_kwargs))
            target.network = pepsy_gate(
                target.network,
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
            target.network = gate_simple(
                target.network,
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
            raise ValueError("method must be 'direct', 'gate', or 'simple'.")

        for gate, where in gates:
            inds = [_format_site_ind(site, target.site_ind_id) for site in where]
            qtn.tensor_network_gate_inds(
                target.network,
                gate,
                inds,
                contract=contract,
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
        contract="split",
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

    def energy(self, hamiltonian=None, *, model=None, normalized=True, contraction_opt=None, **params):
        """Estimate ``<psi|H|psi>`` from local two-site Symmray terms."""
        ham = self.require_hamiltonian(model=model, hamiltonian=hamiltonian, **params)
        opt = self.contraction_opt if contraction_opt is None else contraction_opt
        bra = self.network.H
        total = 0
        for edge, term in ham.terms.items():
            inds = [_format_site_ind(site, self.site_ind_id) for site in edge]
            gated = qtn.tensor_network_gate_inds(
                self.network,
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

    def energy_density(self, hamiltonian=None, *, model=None, normalized=True, contraction_opt=None, **params):
        """Return local-term energy divided by the number of sites."""
        return self.energy(
            hamiltonian=hamiltonian,
            model=model,
            normalized=normalized,
            contraction_opt=contraction_opt,
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
        **kwargs,
    ):
        """Create a random symmetric open-chain MPS."""
        edges = _open_chain_edges(L)
        site_charge_use = _default_site_charge(symmetry) if site_charge is None else site_charge
        phys_sectors = _resolve_phys_sectors(symmetry, phys_dim)
        sr = _require_symmray()
        constructor = sr.TN_fermionic_from_edges_rand if fermionic else sr.TN_abelian_from_edges_rand
        network = constructor(
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
        network.view_as_(
            qtn.MatrixProductState,
            L=int(L),
            site_tag_id="I{}",
            site_ind_id="k{}",
            cyclic=False,
        )
        return cls(
            network=network,
            symmetry=str(symmetry),
            edges=edges,
            fermionic=bool(fermionic),
            contraction_opt=contraction_opt,
            site_ind_id="k{}",
            phys_sectors=phys_sectors,
            site_charge=site_charge_use,
        )

    @classmethod
    def for_model(cls, model, L, *, symmetry=None, fermionic=None, phys_dim=None, **kwargs):
        """Create a random MPS with defaults suitable for a named model."""
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
        chi_use = target.network.max_bond() if chi is None else int(chi)
        opt_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        opt = MpsOptimizer(
            target.network,
            stream,
            chi=chi_use,
            mode=mode,
            inplace=True,
            **opt_kwargs,
        )
        run_opts = {
            "progbar": False,
            "cutoff": cutoff,
            "fidelity_samples": 0,
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
        target.network = opt.run(**run_opts)
        return target

    @property
    def num_sites(self):
        """Number of MPS sites."""
        return int(self.network.L)

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
        **kwargs,
    ):
        """Create a random symmetric 2D PEPS."""
        site_charge_use = _default_site_charge(symmetry) if site_charge is None else site_charge
        phys_sectors = _resolve_phys_sectors(symmetry, phys_dim)
        sr = _require_symmray()
        constructor = sr.PEPS_fermionic_rand if fermionic else sr.PEPS_abelian_rand
        network = constructor(
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
        edges = _as_edges(qtn.edges_2d_square(int(Lx), int(Ly), cyclic=cyclic))
        return cls(
            network=network,
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
        from quimb.tensor.tn2d.core import (  # pylint: disable=import-outside-toplevel
            calc_plaquette_map,
            calc_plaquette_sizes,
        )

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
            norm_tn = self.network.make_norm(layer_tags=layer_tags)
            plaquette_envs = {}
            for x_bsz, y_bsz in calc_plaquette_sizes(terms.keys(), autogroup):
                plaquette_envs.update(
                    norm_tn.compute_plaquette_environments(
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
                double_layer.contract_ctmrg(
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
            double_layer.contract_boundary(
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
        ket_obs = self.network.copy()
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
            bra_network = self.network
        elif isinstance(bra, _SymState):
            bra_network = bra.network
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
            denom_tn = self.network.make_norm(layer_tags=layer_tags)
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
        cutoff_mode="rel",
        mode="mps",
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
        if chi is None and plaquette_envs is None and not (
            isinstance(bdy, dict) and bdy.get("plaquette_envs") is not None
        ):
            raise ValueError("Provide chi when quimb plaquette environments are not supplied.")

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

        value = self.network.compute_local_expectation(
            terms,
            max_bond=chi,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode_local,
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
        return int(self.network.Lx) * int(self.network.Ly)

    @property
    def Lx(self):
        """PEPS x dimension."""
        return int(self.network.Lx)

    @property
    def Ly(self):
        """PEPS y dimension."""
        return int(self.network.Ly)
