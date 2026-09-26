"""Native fermion models, observables, parameterized gates, and factory helpers.

``Fermion`` owns local-space and model behavior. Shared charge conversion,
Hamiltonian, and MPO construction live in :mod:`pepsy.tensors.symmetric`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .symmetric import (
    SymGateStream,
    SymHamiltonian,
    _apply_to_array_blocks,
    _apply_to_hamiltonian_terms,
    _as_edges,
    _as_spin_pair,
    _as_term_where,
    _build_factorized_pair_mpo,
    _charge_add,
    _coupling_is_active,
    _edge_angle_parameter,
    _edge_coloring_layers,
    _edge_parameter,
    _fh_spinful_dense_local_ops,
    _fh_spinless_dense_local_ops,
    _gate_from_term,
    _is_fermionic_symmray_array,
    _is_single_site_identity_hamiltonian,
    _neg_charge,
    _node_parameter,
    _normalize_group_charge,
    _operator_content_fingerprint,
    _site_parity,
    _sites_from_edges,
    _term_mapping_uses_coordinate_sites,
    _yoshida4_stream,
    _zero_like_charge,
    default_physical_sectors,
    site_charge_from_occupations,
    symm_operator_from_dense,
)

__all__ = [
    "Fermion",
    "FermionLatticeSetup",
    "SpinfulFermion",
    "SpinfulFermionHubbard",
    "fermion_density_param_gen",
    "fermion_hopping_param_gen",
    "fermion_interaction_param_gen",
    "SymmFermions",
]


@dataclass(frozen=True)
class FermionLatticeSetup:
    """Metadata for a spinful half-filled rectangular fermion lattice.

    This container deliberately does not build a PEPS, Hamiltonian, or gate
    stream. It only centralizes the lattice sites, edges, symmetry-compatible
    occupations, and conserved charge needed by an explicit workflow.
    """

    Lx: int
    Ly: int
    pattern: str
    cyclic: bool
    sites: tuple
    edges: tuple
    occupations: Mapping
    spin_occupations: Mapping
    target_charge: object
    target_particles: int
    site_charge: object


def _fermion_backend_anchor(*values):
    """Return an array-like value suitable for an Autoray backend context."""
    for value in values:
        if hasattr(value, "shape") and not isinstance(value, (str, bytes)):
            return value
    return np.asarray(0.0j)


def _fermion_scalar_anchor(value, dtype):
    """Represent a plain scalar in the requested native fermion dtype."""
    if hasattr(value, "shape") and not isinstance(value, (str, bytes)):
        return value
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype("complex64"), np.dtype("float32")}:
        return value
    return np.asarray(value, dtype=dtype)


def _fermion_complex_like(value):
    """Return a complex scalar on ``value``'s backend for local builders."""
    if isinstance(value, np.ndarray):
        return np.asarray(0.0j, dtype=np.result_type(value.dtype, np.complex64))
    if hasattr(value, "shape"):
        with ar.backend_like(value):
            zero = 0.0 * value
            return ar.do("complex", zero, zero)
    return np.asarray(0.0j)


def _fermion_complex_phase(angle, *, like):
    """Build ``exp(i * angle)`` without coercing an autodiff scalar."""
    with ar.backend_like(like):
        zero = 0.0 * like
        return ar.do("exp", ar.do("complex", zero, angle))


def _fermion_diagonal_gate_param_gen(
    params,
    diagonal,
    sectors,
    *,
    symmetry,
    imaginary=False,
    sites=1,
):
    """Build a native diagonal fermion gate from one backend-native angle."""
    theta = params[0]
    like = _fermion_backend_anchor(theta, *diagonal)
    with ar.backend_like(like):
        zero = 0.0 * like
        one = zero + 1.0
        scale = (
            -theta
            if imaginary
            else ar.do("complex", zero, -theta)
        )
        values = ar.do(
            "stack",
            tuple(ar.do("exp", scale * value) for value in diagonal),
        )
        # ``one`` ensures that a scalar backend value is still represented by
        # the selected backend when the diagonal contains only zero entries.
        values = values + 0.0 * one
        dense = ar.do("diag", values)
    return symm_operator_from_dense(
        dense,
        sectors,
        symmetry=symmetry,
        charge=_zero_like_charge(next(iter(sectors))),
        fermionic=True,
        sites=sites,
    )


def fermion_interaction_param_gen(params, *, symmetry="U1U1", imaginary=False):
    """Build ``exp(-i theta n_up n_down)`` as a native Symmray gate.

    This follows the parameter-generator convention used by Quimb gate
    registries: ``params[0]`` is the differentiable angle. For imaginary
    time, the phase becomes ``exp(-theta)``.
    """
    if str(symmetry) not in {"U1", "Z2", "U1U1", "Z2Z2"}:
        raise ValueError(
            "Spinful interaction gates require symmetry 'U1', 'Z2', "
            "'U1U1', or 'Z2Z2'."
        )
    return _fermion_diagonal_gate_param_gen(
        params,
        (0.0, 0.0, 0.0, 1.0),
        default_physical_sectors(str(symmetry), 4),
        symmetry=str(symmetry),
        imaginary=imaginary,
        sites=1,
    )


def fermion_density_param_gen(params, *, symmetry="U1", imaginary=False):
    """Build ``exp(-i theta n_i n_j)`` as a native two-site gate."""
    if str(symmetry) not in {"U1", "Z2"}:
        raise ValueError("Spinless density gates require symmetry 'U1' or 'Z2'.")
    return _fermion_diagonal_gate_param_gen(
        params,
        (0.0, 0.0, 0.0, 1.0),
        default_physical_sectors(str(symmetry), 2),
        symmetry=str(symmetry),
        imaginary=imaginary,
        sites=2,
    )


def _fermion_spinful_density_param_gen(params, *, symmetry="U1U1", imaginary=False):
    """Build a spinful total-density interaction gate on two sites."""
    if str(symmetry) not in {"U1", "Z2", "U1U1", "Z2Z2"}:
        raise ValueError(
            "Spinful density gates require symmetry 'U1', 'Z2', 'U1U1', "
            "or 'Z2Z2'."
        )
    occupations = (0.0, 1.0, 1.0, 2.0)
    diagonal = tuple(
        left * right
        for left in occupations
        for right in occupations
    )
    return _fermion_diagonal_gate_param_gen(
        params,
        diagonal,
        default_physical_sectors(str(symmetry), 4),
        symmetry=str(symmetry),
        imaginary=imaginary,
        sites=2,
    )


def _fermion_terms_exponential_gate(
    terms,
    bases,
    sectors,
    *,
    symmetry,
    dt,
    imaginary=False,
    like=None,
):
    """Exponentiate raw local fermion terms while preserving their backend."""
    raw = _fermion_terms_dense(terms, bases, like=like, dt=dt)
    rank = len(bases)
    # Construct the native operator before exponentiating. Flattening the raw
    # fermionic tensor directly treats it as an ordinary site-major matrix and
    # loses the graded reshape convention used by ``FermionicArray``. The
    # Hamiltonian-term exponentiator preserves that convention and also keeps
    # the supplied backend/autodiff values intact.
    operator = symm_operator_from_dense(
        raw,
        sectors,
        symmetry=symmetry,
        charge=_zero_like_charge(next(iter(sectors))),
        fermionic=True,
        sites=rank,
    )
    return _gate_from_term(operator, dt, imaginary=imaginary)


def _fermion_terms_dense(terms, bases, *, like=None, dt=None):
    """Build raw dense data for local fermion terms.

    ``symmray`` uses indexed updates while assembling fermionic terms. Keep
    the JAX path functional and use the supplied coefficient/backend anchor
    for Torch and other Autoray backends.
    """
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    coefficients = tuple(coefficient for coefficient, _ in terms)
    like = _fermion_complex_like(
        _fermion_backend_anchor(dt, like, *coefficients)
    )
    backend = ar.infer_backend(like)
    if backend == "jax":
        # Symmray's dense helper uses in-place indexed updates, while JAX
        # arrays are immutable. Build the same raw tensor with ``.at`` so
        # parameter gradients remain traceable.
        raw = ar.do("zeros", tuple(len(basis) for basis in bases) * 2, like=like)
        for index, value in flo.build_local_fermionic_elements(terms, bases).items():
            raw = raw.at[index].add(value)
    else:
        raw = flo.build_local_fermionic_dense(terms, bases, like=like)
    return raw


def _fermion_generic_local_modes(spinful, sites):
    """Build local bases and named monomials for generic fermion terms."""
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    bases = []
    operators = {}
    for position, site in enumerate(sites):
        label = f"pepsy_site_{position}"
        if spinful:
            up = flo.FermionicOperator(f"{label}_up")
            down = flo.FermionicOperator(f"{label}_down")
            bases.append(((), (up.dag,), (down.dag,), (down.dag, up.dag)))
            operators[site] = {
                "annihilate_u": (up,),
                "create_u": (up.dag,),
                "number_u": (up.dag, up),
                "annihilate_d": (down,),
                "create_d": (down.dag,),
                "number_d": (down.dag, down),
                "double": (up.dag, up, down.dag, down),
                "s_plus": (up.dag, down),
                "s_minus": (down.dag, up),
                "pair_create": (up.dag, down.dag),
                "pair_annihilate": (down, up),
            }
        else:
            mode = flo.FermionicOperator(label)
            bases.append(((), (mode.dag,)))
            operators[site] = {
                "annihilate": (mode,),
                "create": (mode.dag,),
                "number": (mode.dag, mode),
            }
    return tuple(bases), operators


def _spinless_hopping_gate(
    symmetry,
    dt,
    *,
    t=1.0,
    peierls_angle=0.0,
    imaginary=False,
):
    """Build a backend-native spinless hopping gate from graded terms."""
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    a = flo.FermionicOperator("a")
    b = flo.FermionicOperator("b")
    like = _fermion_backend_anchor(dt, t, peierls_angle)
    phase = _fermion_complex_phase(peierls_angle, like=like)
    terms = (
        (-t * phase, (a.dag, b)),
        (-t * ar.do("conj", phase), (b.dag, a)),
    )
    bases = (((), (a.dag,)), ((), (b.dag,)))
    return _fermion_terms_exponential_gate(
        terms,
        bases,
        default_physical_sectors(str(symmetry), 2),
        symmetry=str(symmetry),
        dt=dt,
        imaginary=imaginary,
        like=like,
    )


def _spinful_hopping_gate(
    symmetry,
    dt,
    *,
    t=1.0,
    peierls_angle=0.0,
    imaginary=False,
):
    """Build a backend-native spinful hopping gate from graded terms."""
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    au = flo.FermionicOperator("a↑")
    ad = flo.FermionicOperator("a↓")
    bu = flo.FermionicOperator("b↑")
    bd = flo.FermionicOperator("b↓")
    tu, td = _as_spin_pair(t, name="t")
    like = _fermion_backend_anchor(dt, tu, td, peierls_angle)
    phase = _fermion_complex_phase(peierls_angle, like=like)
    phase_conj = ar.do("conj", phase)
    terms = (
        (-tu * phase, (au.dag, bu)),
        (-tu * phase_conj, (bu.dag, au)),
        (-td * phase, (ad.dag, bd)),
        (-td * phase_conj, (bd.dag, ad)),
    )
    bases = (
        ((), (au.dag,), (ad.dag,), (ad.dag, au.dag)),
        ((), (bu.dag,), (bd.dag,), (bd.dag, bu.dag)),
    )
    return _fermion_terms_exponential_gate(
        terms,
        bases,
        default_physical_sectors(str(symmetry), 4),
        symmetry=str(symmetry),
        dt=dt,
        imaginary=imaginary,
        like=like,
    )


def fermion_hopping_param_gen(
    params,
    *,
    spinful=False,
    symmetry="U1",
    imaginary=False,
    peierls_angle=0.0,
):
    """Build a native hopping gate from a Quimb-style angle parameter."""
    theta = params[0]
    if spinful:
        if str(symmetry) not in {"U1", "Z2", "U1U1", "Z2Z2"}:
            raise ValueError(
                "Spinful hopping gates require symmetry 'U1', 'Z2', "
                "'U1U1', or 'Z2Z2'."
            )
        return _spinful_hopping_gate(
            str(symmetry),
            theta,
            t=1.0,
            peierls_angle=peierls_angle,
            imaginary=imaginary,
        )
    if str(symmetry) not in {"U1", "Z2"}:
        raise ValueError("Spinless hopping gates require symmetry 'U1' or 'Z2'.")
    return _spinless_hopping_gate(
        str(symmetry),
        theta,
        t=1.0,
        peierls_angle=peierls_angle,
        imaginary=imaginary,
    )


@dataclass
class Fermion:
    """Native spinless or spinful fermion observables, gates, and streams.

    The helper owns only the local fermionic space, symmetry convention, and
    optional backend conversion. Hamiltonian couplings are deliberately not
    stored here: construct them as explicit native terms, then validate and
    bundle them with :meth:`hamiltonian`. This prevents a native Hamiltonian,
    a gate stream, and a VMC adapter from silently using different couplings.
    It is intended for direct Symmray-backed fermionic MPS or PEPS workflows;
    it does not introduce a qubit or Jordan-Wigner circuit representation.

    ``strang_gate_stream`` uses a deterministic edge colouring and a
    forward/reverse half-step sequence.  Consequently its hopping layers are
    vertex-disjoint and the complete interaction-plus-hopping product formula
    is second order even when hopping terms on neighbouring edges do not
    commute.
    """

    symmetry: str | None = None
    dtype: object = "complex128"
    to_backend: object = None
    spinful: bool = True
    _dense_ops: dict = field(default_factory=dict, init=False, repr=False)
    _observable_cache: dict = field(default_factory=dict, init=False, repr=False)
    _gate_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.spinful = bool(self.spinful)
        if self.symmetry is None:
            self.symmetry = "U1U1" if self.spinful else "U1"
        self.symmetry = str(self.symmetry)
        allowed = (
            {"U1", "Z2", "U1U1", "Z2Z2"}
            if self.spinful
            else {"U1", "Z2"}
        )
        if self.symmetry not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            kind = "spinful" if self.spinful else "spinless"
            raise ValueError(f"{kind} fermions require symmetry in {{{allowed_text}}}.")
        self.dtype = np.dtype(self.dtype)

    @property
    def model(self):
        """The matching :class:`SymHamiltonian` model name."""
        if not self.spinful:
            return "fermi_hubbard_spinless"
        return (
            "fermi_hubbard"
            if self.symmetry in {"U1", "Z2"}
            else "fermi_hubbard_u1u1"
        )

    @property
    def physical_sectors(self):
        """Return the local charge sectors for this fermion space."""
        return default_physical_sectors(
            self.symmetry,
            4 if self.spinful else 2,
        )

    @property
    def zero_charge(self):
        """Return the neutral local operator charge."""
        return 0 if self.symmetry in {"U1", "Z2"} else (0, 0)

    @property
    def pair_charge(self):
        """Return the charge of a local pair-creation operator."""
        if not self.spinful:
            raise AttributeError("Spinless fermions do not have spin-pair charge.")
        charge = 2 if self.symmetry in {"U1", "Z2"} else (1, 1)
        return _normalize_group_charge(charge, self.symmetry)

    @property
    def pair_annihilation_charge(self):
        """Return the charge of a local pair-annihilation operator."""
        if not self.spinful:
            raise AttributeError("Spinless fermions do not have spin-pair charge.")
        return _normalize_group_charge(
            _neg_charge(self.pair_charge),
            self.symmetry,
        )

    def lattice_half_filling(
        self,
        Lx,
        Ly=None,
        *,
        pattern="checkerboard",
        cyclic=False,
    ):
        """Prepare metadata for an explicit half-filled spinful lattice workflow.

        The returned :class:`FermionLatticeSetup` contains the coordinate
        sites, nearest-neighbor lattice edges, spin-resolved occupations, and
        occupations expressed in this fermion's symmetry. It intentionally
        does not construct a PEPS, Hamiltonian, or gate stream, so callers can
        keep those steps explicit.

        Parameters
        ----------
        Lx, Ly : int
            Lattice dimensions. If ``Ly`` is omitted, use a square lattice.
        pattern : {"checkerboard", "neel", "neel_like"}
            Spin pattern for the one-particle-per-site initial state.
        cyclic : bool, optional
            Whether to include periodic physical lattice edges. This metadata
            flag does not determine the boundary conditions of a PEPS or MPS
            state built from the returned setup.
        """
        if not self.spinful:
            raise ValueError(
                "lattice_half_filling is defined for spinful fermions."
            )
        if Ly is None:
            Ly = Lx
        if not isinstance(Lx, Integral) or int(Lx) < 1:
            raise ValueError("Lx must be a positive integer.")
        if not isinstance(Ly, Integral) or int(Ly) < 1:
            raise ValueError("Ly must be a positive integer.")

        pattern_name = str(pattern).lower().replace("-", "_")
        if pattern_name not in {"checkerboard", "neel", "neel_like"}:
            raise ValueError(
                "pattern must be 'checkerboard', 'neel', or 'neel_like'."
            )

        Lx = int(Lx)
        Ly = int(Ly)
        sites = tuple((x, y) for x in range(Lx) for y in range(Ly))
        spin_occupations = {
            site: (1, 0) if (site[0] + site[1]) % 2 == 0 else (0, 1)
            for site in sites
        }
        if self.symmetry in {"U1U1", "Z2Z2"}:
            occupations = dict(spin_occupations)
        else:
            occupations = {
                site: n_up + n_down
                for site, (n_up, n_down) in spin_occupations.items()
            }

        edges = tuple(
            tuple(edge)
            for edge in qtn.edges_2d_square(Lx, Ly, cyclic=cyclic)
        )
        target_particles = sum(
            n_up + n_down for n_up, n_down in spin_occupations.values()
        )
        return FermionLatticeSetup(
            Lx=Lx,
            Ly=Ly,
            pattern=pattern_name,
            cyclic=bool(cyclic),
            sites=sites,
            edges=edges,
            occupations=occupations,
            spin_occupations=spin_occupations,
            target_charge=self.total_charge(occupations.values()),
            target_particles=target_particles,
            site_charge=site_charge_from_occupations(occupations),
        )

    def half_filled_occupations(self, L):
        """Return a half-filled product-state charge pattern of length ``L``."""
        if not isinstance(L, Integral) or int(L) < 1:
            raise ValueError("L must be a positive integer.")
        L = int(L)
        if not self.spinful:
            return (1,) * L
        if self.symmetry in {"U1", "Z2"}:
            return (1,) * L
        return tuple((1, 0) if site % 2 == 0 else (0, 1) for site in range(L))

    def local_fock_state(self, occupation, *, site=0):
        """Return ``(physical_charge, sector_index)`` for one local Fock state.

        The physical basis is ``|0>, |up>, |down>, |up down>`` for spinful
        fermions (and ``|0>, |1>`` for spinless fermions).  A spin-resolved
        ``(n_up, n_down)`` occupation always chooses that basis state exactly.
        For spinful ``U1`` and ``Z2`` models, a scalar charge label ``1`` does
        not resolve the two one-particle states; it therefore denotes the
        deterministic checkerboard representative: ``|up>`` at even sites and
        ``|down>`` at odd sites.  Pass a pair to select either spin explicitly.

        This distinction matters for product-state constructors: fixing only a
        degenerate symmetry sector leaves an arbitrary vector in that sector,
        whereas a product state must select a definite Fock basis vector.
        """
        if self.spinful:
            if isinstance(occupation, (tuple, list, np.ndarray)):
                if len(occupation) != 2:
                    raise ValueError(
                        "a spinful local occupation must be a scalar or "
                        "a length-2 (n_up, n_down) pair."
                    )
                n_up, n_down = (int(value) for value in occupation)
                if (n_up, n_down) not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
                    raise ValueError(
                        "spinful local occupations must have n_up, n_down in {0, 1}."
                    )
            else:
                number = int(occupation)
                if number not in {0, 1, 2}:
                    raise ValueError(
                        "a scalar spinful local occupation must be 0, 1, or 2."
                    )
                if number == 0:
                    n_up, n_down = 0, 0
                elif number == 2:
                    n_up, n_down = 1, 1
                elif _site_parity(site):
                    n_up, n_down = 0, 1
                else:
                    n_up, n_down = 1, 0

            charge = (
                n_up + n_down
                if self.symmetry in {"U1", "Z2"}
                else (n_up, n_down)
            )
            charge = _normalize_group_charge(charge, self.symmetry)
            if self.symmetry in {"U1", "Z2"}:
                fock_charges = (0, 1, 1, 2)
            else:
                fock_charges = ((0, 0), (1, 0), (0, 1), (1, 1))
            fock_charges = tuple(
                _normalize_group_charge(candidate, self.symmetry)
                for candidate in fock_charges
            )
            fock_index = n_up + 2 * n_down
            return charge, sum(
                candidate == charge for candidate in fock_charges[:fock_index]
            )

        if isinstance(occupation, (tuple, list, np.ndarray)):
            raise ValueError("a spinless local occupation must be the scalar 0 or 1.")
        number = int(occupation)
        if number not in {0, 1}:
            raise ValueError("a spinless local occupation must be 0 or 1.")
        return _normalize_group_charge(number, self.symmetry), 0

    def total_charge(self, occupations):
        """Return the charge sum for an occupation/charge sequence."""
        occupations = tuple(occupations)
        if not occupations:
            return self.zero_charge
        if self.symmetry in {"U1", "Z2"}:
            total = sum(int(charge) for charge in occupations)
        else:
            total = tuple(
            sum(int(charge[axis]) for charge in occupations)
            for axis in range(2)
            )
        return _normalize_group_charge(total, self.symmetry)

    def half_filled_site_charge(self, L):
        """Return the site-charge callable for ``half_filled_occupations(L)``."""
        return site_charge_from_occupations(self.half_filled_occupations(L))

    def _local_ops(self):
        if not self._dense_ops:
            if self.spinful:
                ops = _fh_spinful_dense_local_ops(self.symmetry, self.dtype)
                ops["charge"] = ops["number_u"] + ops["number_d"]
                ops["number"] = ops["charge"]
                ops["sz"] = 0.5 * (ops["number_u"] - ops["number_d"])
                ops["s_plus"] = ops["create_u"] @ ops["annihilate_d"]
                ops["s_minus"] = ops["create_d"] @ ops["annihilate_u"]
                ops["sx"] = 0.5 * (ops["s_plus"] + ops["s_minus"])
                ops["sy"] = (-0.5j * ops["s_plus"]) + (0.5j * ops["s_minus"])
                ops["pair_create"] = ops["create_u"] @ ops["create_d"]
                ops["pair_annihilate"] = ops["annihilate_d"] @ ops["annihilate_u"]
            else:
                ops = _fh_spinless_dense_local_ops(self.symmetry, self.dtype)
                ops["charge"] = ops["number"]
            self._dense_ops = ops
        return self._dense_ops

    @staticmethod
    def _operator_name(name):
        aliases = {
            "n": "number",
            "occupation": "number",
            "create_up": "create_u",
            "n_up": "number_u",
            "number_up": "number_u",
            "annihilate_up": "annihilate_u",
            "n_down": "number_d",
            "number_down": "number_d",
            "create_down": "create_d",
            "annihilate_down": "annihilate_d",
            "doublon": "double",
            "pair_annihilation": "pair_annihilate",
            "spin_plus": "s_plus",
            "s_plus": "s_plus",
            "spin_minus": "s_minus",
            "s_minus": "s_minus",
            "spin_x": "sx",
            "spin_y": "sy",
            "spin_z": "sz",
        }
        return aliases.get(str(name), str(name))

    @classmethod
    def _adjoint_operator_name(cls, name):
        """Return the local fermion-operator name for its adjoint."""
        name = cls._operator_name(name)
        adjoints = {
            "create": "annihilate",
            "annihilate": "create",
            "create_u": "annihilate_u",
            "annihilate_u": "create_u",
            "create_d": "annihilate_d",
            "annihilate_d": "create_d",
            "s_plus": "s_minus",
            "s_minus": "s_plus",
            "pair_create": "pair_annihilate",
            "pair_annihilate": "pair_create",
        }
        return adjoints.get(name, name)

    def dense_operator(self, name):
        """Return a dense one-site operator in the native basis order.

        Spinless names include ``create``, ``annihilate``, ``number``, and
        ``parity``. Spinful names additionally include the spin-resolved
        number, spin, doublon, and pair operators.
        """
        name = self._operator_name(name)
        try:
            return self._local_ops()[name]
        except KeyError as exc:
            allowed = ", ".join(sorted(self._local_ops()))
            raise ValueError(
                "Unknown fermion operator "
                f"{name!r}; expected one of {allowed}."
            ) from exc

    def operator_charge(self, name):
        """Return the Abelian charge carried by ``dense_operator(name)``."""
        name = self._operator_name(name)
        if name in {"s_plus", "s_minus", "sx", "sy", "sz"} and not self.spinful:
            raise ValueError("Spin operators require spinful fermions.")
        if not self.spinful:
            if name in {"create", "annihilate"}:
                charge = 1 if name == "create" else -1
                return _normalize_group_charge(charge, self.symmetry)
            return self.zero_charge
        if name in {"create_u", "annihilate_u", "create_d", "annihilate_d"}:
            if self.symmetry in {"U1", "Z2"}:
                charge = 1
            elif name.endswith("_u"):
                charge = (0, 1)
            else:
                charge = (1, 0)
            if not name.startswith("create"):
                charge = _neg_charge(charge)
            return _normalize_group_charge(charge, self.symmetry)
        if name in {"s_plus", "s_minus"}:
            if name == "s_plus":
                charge = _charge_add(
                    self.operator_charge("create_u"),
                    self.operator_charge("annihilate_d"),
                    self.symmetry,
                )
            else:
                charge = _charge_add(
                    self.operator_charge("create_d"),
                    self.operator_charge("annihilate_u"),
                    self.symmetry,
                )
            return _normalize_group_charge(charge, self.symmetry)
        if name in {"sx", "sy"}:
            if self.symmetry not in {"U1", "Z2"}:
                raise ValueError(
                    f"{name} is not a homogeneous operator under symmetry "
                    f"{self.symmetry!r}; use symmetry='U1' or 'Z2'."
                )
            return self.zero_charge
        if name == "pair_create":
            return self.pair_charge
        if name == "pair_annihilate":
            return self.pair_annihilation_charge
        return self.zero_charge

    def operator(self, name):
        """Return the cached native Symmray operator for ``name``."""
        return self.observable(name)

    def _require_spinful(self, feature):
        if not self.spinful:
            raise ValueError(f"{feature} requires spinful fermions.")

    def _require_spin_flip_symmetry(self, feature):
        self._require_spinful(feature)
        if self.symmetry not in {"U1", "Z2"}:
            raise ValueError(
                f"{feature} requires symmetry='U1' or 'Z2'; "
                f"symmetry={self.symmetry!r} keeps up/down charges separate."
            )

    @staticmethod
    def _resolve_operator_parameter(value, *, site=None, edge=None):
        if site is not None and edge is not None:
            raise ValueError("Specify either site= or edge=, not both.")
        if edge is not None:
            try:
                left, right = tuple(edge)
            except (TypeError, ValueError) as exc:
                raise ValueError("edge must contain exactly two site labels.") from exc
            if callable(value) or isinstance(value, Mapping):
                return _edge_parameter(value, left, right)
            return value
        if site is not None:
            if callable(value) or isinstance(value, Mapping):
                return _node_parameter(value, site)
            return value
        if callable(value) or isinstance(value, Mapping):
            raise ValueError("site= or edge= is required for a site/edge parameter.")
        return value

    def _spin_flip_operator(self, name):
        self._require_spin_flip_symmetry(f"{name.upper()} operator")
        if name == "sx":
            terms = [
                (0.5, ((0, "s_plus"),)),
            ]
        elif name == "sy":
            terms = [
                (-0.5j, ((0, "s_plus"),)),
            ]
        else:  # pragma: no cover - private callers pass canonical names.
            raise ValueError(f"Unknown spin-flip operator {name!r}.")
        return self.operator_term(terms, sites=(0,), add_hc=True)

    def spin_x_operator(self):
        """Return the native one-site ``Sx`` operator."""
        return self.observable("sx")

    def spin_y_operator(self):
        """Return the native one-site ``Sy`` operator."""
        return self.observable("sy")

    def spin_z_operator(self):
        """Return the native one-site ``Sz`` operator."""
        self._require_spinful("Sz operator")
        return self.observable("sz")

    def sx_operator(self):
        """Alias for :meth:`spin_x_operator`."""
        return self.spin_x_operator()

    def sy_operator(self):
        """Alias for :meth:`spin_y_operator`."""
        return self.spin_y_operator()

    def sz_operator(self):
        """Alias for :meth:`spin_z_operator`."""
        return self.spin_z_operator()

    def spin_x_term(self, site, *, field):
        """Return ``field * Sx`` on one spinful physical site."""
        self._require_spin_flip_symmetry("Sx terms")
        if field is None:
            raise TypeError("spin_x_term requires explicit field=... .")
        field = _node_parameter(field, site)
        return self.operator_term(
            [(0.5 * field, ((site, "s_plus"),))],
            sites=(site,),
            add_hc=True,
        )

    def spin_y_term(self, site, *, field):
        """Return ``field * Sy`` on one spinful physical site."""
        self._require_spin_flip_symmetry("Sy terms")
        if field is None:
            raise TypeError("spin_y_term requires explicit field=... .")
        field = _node_parameter(field, site)
        return self.operator_term(
            [(-0.5j * field, ((site, "s_plus"),))],
            sites=(site,),
            add_hc=True,
        )

    def spin_z_term(self, site, *, field):
        """Return ``field * Sz`` on one spinful physical site."""
        self._require_spinful("Sz terms")
        if field is None:
            raise TypeError("spin_z_term requires explicit field=... .")
        field = _node_parameter(field, site)
        return self.operator_term(
            [
                (0.5 * field, ((site, "number_up"),)),
                (-0.5 * field, ((site, "number_down"),)),
            ],
            sites=(site,),
        )

    def spin_z_correlator(self):
        """Return the bare native two-site ``Sz_i Sz_j`` operator."""
        self._require_spinful("Sz-Sz correlators")
        return self.operator_term(
            [
                (0.25, ((0, "number_u"), (1, "number_u"))),
                (-0.25, ((0, "number_u"), (1, "number_d"))),
                (-0.25, ((0, "number_d"), (1, "number_u"))),
                (0.25, ((0, "number_d"), (1, "number_d"))),
            ],
            sites=(0, 1),
            charge=self.zero_charge,
        )

    def spin_x_correlator(self):
        """Return the native two-site ``Sx_i Sx_j`` operator."""
        self._require_spin_flip_symmetry("Sx-Sx correlators")
        return self.operator_term(
            [
                (0.25, ((0, "s_plus"), (1, "s_plus"))),
                (0.25, ((0, "s_plus"), (1, "s_minus"))),
                (0.25, ((0, "s_minus"), (1, "s_plus"))),
                (0.25, ((0, "s_minus"), (1, "s_minus"))),
            ],
            sites=(0, 1),
            charge=self.zero_charge,
        )

    def spin_y_correlator(self):
        """Return the native two-site ``Sy_i Sy_j`` operator."""
        self._require_spin_flip_symmetry("Sy-Sy correlators")
        return self.operator_term(
            [
                (-0.25, ((0, "s_plus"), (1, "s_plus"))),
                (0.25, ((0, "s_plus"), (1, "s_minus"))),
                (0.25, ((0, "s_minus"), (1, "s_plus"))),
                (-0.25, ((0, "s_minus"), (1, "s_minus"))),
            ],
            sites=(0, 1),
            charge=self.zero_charge,
        )

    def xy_exchange_operator(self):
        """Return the native ``Sx_i Sx_j + Sy_i Sy_j`` operator."""
        self._require_spinful("XY exchange operators")
        return self.operator_term(
            [(0.5, ((0, "s_plus"), (1, "s_minus")))],
            sites=(0, 1),
            charge=self.zero_charge,
            add_hc=True,
        )

    def heisenberg_operator(self):
        """Return the native two-site ``S_i dot S_j`` operator."""
        self._require_spinful("Heisenberg operators")
        return self.operator_term(
            [
                (0.25, ((0, "number_u"), (1, "number_u"))),
                (-0.25, ((0, "number_u"), (1, "number_d"))),
                (-0.25, ((0, "number_d"), (1, "number_u"))),
                (0.25, ((0, "number_d"), (1, "number_d"))),
                (0.5, ((0, "s_plus"), (1, "s_minus"))),
                (0.5, ((1, "s_plus"), (0, "s_minus"))),
            ],
            sites=(0, 1),
            charge=self.zero_charge,
        )

    def operator_term(
        self,
        terms,
        *,
        sites=None,
        charge=None,
        like=None,
        add_hc=False,
        label=None,
    ):
        """Return a native operator made from explicit fermion monomials.

        Parameters
        ----------
        terms : sequence of ``(coefficient, operators)``
            ``operators`` is a sequence of ``(site, name)`` pairs. Names are
            the same local names accepted by :meth:`operator`, for example
            ``create_up``, ``annihilate_down``, ``number_up``, ``double``,
            ``create`, and ``annihilate`` for spinless fermions.
        sites : sequence, optional
            Ordered site labels for the returned operator. If omitted, sites
            are inferred from their first appearance in ``terms``. The order
            is also the order expected by ``state.measure(operator, where)``.
        charge : optional
            Total Abelian operator charge. By default it is inferred and all
            monomials must have the same charge.
        add_hc : bool, optional
            Append the Hermitian conjugate of every supplied monomial. The
            input must then define a self-conjugate charge sector, as required
            for one homogeneous Symmray operator. Fermionic factor order is
            reversed when taking the adjoint.


        This returns the operator itself, not ``exp(-i dt H)``. For example,
        the spin-up hopping term is constructed with::

            fermion.operator_term([
                (-t, ((i, "create_up"), (j, "annihilate_up"))),
                (-t, ((j, "create_up"), (i, "annihilate_up"))),
            ])
        """
        entries = tuple(terms)
        if not entries:
            raise ValueError("terms must contain at least one local term.")

        if like is None:
            like = _fermion_scalar_anchor(0.0, self.dtype)

        inferred_sites = []
        normalized = []
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("terms must have form (coefficient, operators).")
            coefficient, references = entry
            expanded = []
            term_charge = self.zero_charge
            for reference in tuple(references):
                if not isinstance(reference, (tuple, list)) or len(reference) != 2:
                    raise ValueError(
                        "operator references must have form (site, name)."
                    )
                site, name = reference
                if site not in inferred_sites:
                    inferred_sites.append(site)
                name = self._operator_name(name)
                expanded.append((site, name))
            normalized.append((coefficient, tuple(expanded)))

        if add_hc:
            normalized.extend(
                (
                    ar.do("conj", coefficient),
                    tuple(
                        (site, self._adjoint_operator_name(name))
                        for site, name in reversed(references)
                    ),
                )
                for coefficient, references in tuple(normalized)
            )

        term_charges = []
        for _, references in normalized:
            term_charge = self.zero_charge
            for _, name in references:
                term_charge = _charge_add(
                    term_charge,
                    self.operator_charge(name),
                    self.symmetry,
                )
            term_charges.append(term_charge)

        if sites is None:
            sites = tuple(inferred_sites)
        elif isinstance(sites, (str, bytes)):
            sites = (sites,)
        else:
            sites = tuple(sites)
        if not sites:
            raise ValueError("sites must contain at least one local site.")
        if len(set(sites)) != len(sites):
            raise ValueError("sites must contain unique site labels.")
        missing = [site for site in inferred_sites if site not in sites]
        if missing:
            raise ValueError(f"sites is missing referenced labels: {missing!r}.")

        inferred_charge = term_charges[0]
        if any(term_charge != inferred_charge for term_charge in term_charges[1:]):
            if add_hc:
                raise ValueError(
                    "add_hc requires a self-conjugate operator charge; "
                    "charged and Hermitian-conjugate monomials cannot be "
                    "combined into one homogeneous Symmray operator."
                )
            raise ValueError(
                "All monomials in one operator_term must carry the same charge."
            )
        if charge is None:
            charge = inferred_charge

        bases, local_operators = _fermion_generic_local_modes(self.spinful, sites)
        site_positions = {site: position for position, site in enumerate(sites)}
        raw_terms = []
        for coefficient, references in normalized:
            expanded = []
            for site, name in references:
                if site not in site_positions:
                    raise ValueError(
                        f"operator reference uses site {site!r}, which is not in sites."
                    )
                try:
                    expanded.extend(local_operators[site][name])
                except KeyError as exc:
                    allowed = ", ".join(sorted(local_operators[site]))
                    raise ValueError(
                        f"Unknown generic fermion monomial {name!r}; "
                        f"expected one of {allowed}."
                    ) from exc
            raw_terms.append((coefficient, tuple(expanded)))

        dense = _fermion_terms_dense(raw_terms, bases, like=like)
        operator = symm_operator_from_dense(
            dense,
            self.physical_sectors,
            symmetry=self.symmetry,
            charge=charge,
            fermionic=True,
            sites=len(sites),
            index_maps=tuple(
                {
                    index: value
                    for index, value in enumerate(self._local_ops()["index_map"])
                }
                for _ in range(2 * len(sites))
            ),
            label=label,
        )
        return _apply_to_array_blocks(operator, self.to_backend)

    @staticmethod
    def _normalize_majorana_component(component):
        key = str(component).strip().lower()
        if component in {0, "0"} or key in {"x", "real", "gamma_x", "gamma0"}:
            return 0
        if component in {1, "1"} or key in {"y", "imag", "gamma_y", "gamma1"}:
            return 1
        raise ValueError("Majorana component must be 0/'x' or 1/'y'.")

    def _require_majorana(self, feature):
        if self.spinful:
            raise NotImplementedError(
                f"{feature} currently targets one complex mode per site; "
                "use Fermion(spinful=False) or provide an explicit flavor map."
            )
        if self.symmetry != "Z2":
            raise ValueError(
                f"{feature} uses the native parity convention and requires "
                "symmetry='Z2'; U1/U1U1 does not make a single Majorana "
                "operator homogeneous."
            )

    def _majorana_charge(self):
        self._require_majorana("Majorana operators")
        return _normalize_group_charge(1, self.symmetry)

    def _majorana_mode_terms(self, site, component):
        component = self._normalize_majorana_component(component)
        if component == 0:
            return ((1.0, ((site, "create"),)), (1.0, ((site, "annihilate"),)))
        return (
            (1.0j, ((site, "create"),)),
            (-1.0j, ((site, "annihilate"),)),
        )

    def majorana_operator(self, component=0, *, site=0):
        """Return a native parity-odd Majorana operator.

        The convention is ``gamma_x = c + c^†`` and
        ``gamma_y = -i (c - c^†)``. It is intentionally a ``Z2`` path:
        individual Majoranas are not homogeneous under particle-number ``U1``.
        """
        charge = self._majorana_charge()
        return self.operator_term(
            self._majorana_mode_terms(site, component),
            sites=(site,),
            charge=charge,
            label=f"majorana_{site!r}",
        )

    def _majorana_bilinear_terms(
        self,
        left,
        right,
        *,
        left_component=0,
        right_component=0,
        coefficient=1.0,
        canonical=True,
    ):
        if left == right:
            raise ValueError("Majorana bilinears require distinct mode sites.")
        terms = []
        for left_coeff, left_ops in self._majorana_mode_terms(left, left_component):
            for right_coeff, right_ops in self._majorana_mode_terms(right, right_component):
                coefficient_term = 1.0j * coefficient * left_coeff * right_coeff
                # Symmray's graded local-element builder canonicalizes the
                # all-annihilator monomial in site order. Compensate that
                # reversal sign so ``i * gamma_left * gamma_right`` is
                # Hermitian in the native fermionic representation.
                if canonical and (
                    left_ops[0][1] == "annihilate"
                    and right_ops[0][1] == "annihilate"
                ):
                    coefficient_term = -coefficient_term
                terms.append(
                    (
                        coefficient_term,
                        (*left_ops, *right_ops),
                    )
                )
        return tuple(terms)

    def majorana_bilinear_operator(
        self,
        edge,
        *,
        left_component=0,
        right_component=0,
        coefficient=1.0,
    ):
        """Return ``coefficient * i gamma_left gamma_right``."""
        self._require_majorana("Majorana bilinears")
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge must contain exactly two mode sites.") from exc
        return self.operator_term(
            self._majorana_bilinear_terms(
                left,
                right,
                left_component=left_component,
                right_component=right_component,
                coefficient=coefficient,
                canonical=True,
            ),
            sites=(left, right),
            charge=self.zero_charge,
        )

    def pairing_operator(self, edge, *, coefficient=1.0, phase=0.0):
        """Return a Hermitian spinless pairing operator on ``edge``."""
        self._require_majorana("Pairing operators")
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge must contain exactly two mode sites.") from exc
        amplitude = coefficient * _fermion_complex_phase(phase, like=coefficient)
        return self.operator_term(
            (
                (amplitude, ((left, "create"), (right, "create"))),
                (
                    ar.do("conj", amplitude),
                    ((left, "annihilate"), (right, "annihilate")),
                ),
            ),
            sites=(left, right),
            charge=self.zero_charge,
        )

    def majorana_gate(
        self,
        dt,
        *,
        edge,
        left_component=0,
        right_component=0,
        coefficient=1.0,
        imaginary=False,
    ):
        """Return ``exp(-i dt * i gamma_left gamma_right)``."""
        self._require_majorana("Majorana gates")
        left, right = tuple(edge)
        return self.exponential(
            self._majorana_bilinear_terms(
                left,
                right,
                left_component=left_component,
                right_component=right_component,
                coefficient=coefficient,
                canonical=False,
            ),
            dt,
            sites=(left, right),
            imaginary=imaginary,
        )

    def pairing_gate(self, dt, *, edge, coefficient=1.0, phase=0.0, imaginary=False):
        """Return ``exp(-i dt H_pair)`` for a parity-preserving pairing term."""
        self._require_majorana("Pairing gates")
        left, right = tuple(edge)
        amplitude = coefficient * _fermion_complex_phase(phase, like=coefficient)
        return self.exponential(
            (
                (amplitude, ((left, "create"), (right, "create"))),
                (
                    -ar.do("conj", amplitude),
                    ((left, "annihilate"), (right, "annihilate")),
                ),
            ),
            dt,
            sites=(left, right),
            imaginary=imaginary,
        )

    def eta_pair_operator(self, *, coefficient=1.0):
        """Return ``coefficient * Delta_0^dag Delta_1 + h.c.``.

        The returned operator uses canonical two-site locations ``(0, 1)``;
        place it on physical sites through ``state.measure(..., where=...)``
        or an explicit VMC term mapping. It is neutral under the selected
        spinful symmetry and preserves native fermionic grading.
        """
        if not self.spinful:
            raise ValueError(
                "Eta-pair operators require spinful fermions with up and down "
                "modes."
            )
        return self.operator_term(
            [
                (
                    coefficient,
                    ((0, "pair_create"), (1, "pair_annihilate")),
                )
            ],
            sites=(0, 1),
            charge=self.zero_charge,
            add_hc=True,
        )

    def eta_pair_structure_factor_mpo(
        self,
        L,
        *,
        signs,
        normalization=None,
        max_bond=None,
        cutoff=1e-12,
        compress=False,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        dtype=None,
        to_backend=None,
    ):
        """Build a compact native MPO for the staggered eta structure factor.

        ``signs`` gives the staggered sign in the one-dimensional MPS chain
        order.  The returned neutral MPO represents

        ``normalization * (F^dagger F - sum_i Delta_i^dagger Delta_i)``

        with ``F = sum_i signs[i] * Delta_i``.  For the common convention
        ``P_eta = (2 / L) * sum_{i < j} signs[i] * signs[j] *
        Re(<Delta_i^dagger Delta_j>)``, use ``normalization=1 / L``.

        This uses a constant-size native graded automaton rather than the
        explicit ``O(L**2)`` two-site term expansion.  The internal pair
        channels carry the nonzero Symmray charges while the MPO boundary is
        neutral, so no charge-changing MPO sector or Jordan--Wigner copy is
        introduced.
        """
        self._require_spinful("eta-pair structure-factor MPOs")
        if not isinstance(L, Integral) or int(L) < 1:
            raise ValueError("L must be a positive integer.")
        L = int(L)
        if normalization is None:
            normalization = 1.0 / L
        to_backend = self.to_backend if to_backend is None else to_backend
        pair_create = self.observable("pair_create")
        pair_annihilate = self.observable("pair_annihilate")
        onsite = self.observable("doublon")
        return _build_factorized_pair_mpo(
            L=L,
            pair_create=pair_create,
            pair_annihilate=pair_annihilate,
            onsite=onsite,
            signs=signs,
            normalization=normalization,
            symmetry=self.symmetry,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
            to_backend=to_backend,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _hopping_operator_on_sites(
        self,
        left,
        right,
        *,
        t=1.0,
        spin=None,
        peierls_angle=0.0,
        include_minus=False,
    ):
        if self.spinful:
            t_up, t_down = _as_spin_pair(t, name="t")
            phase = _fermion_complex_phase(
                peierls_angle,
                like=_fermion_backend_anchor(t_up, t_down, peierls_angle),
            )
            phase_conj = ar.do("conj", phase)
            channels = {
                "up": (t_up, "create_up", "annihilate_up"),
                "down": (t_down, "create_down", "annihilate_down"),
            }
            selected = ("up", "down") if spin is None else (str(spin).lower(),)
        else:
            if spin is not None:
                raise ValueError("Spinless fermions do not have spin channels.")
            phase = _fermion_complex_phase(
                peierls_angle,
                like=_fermion_backend_anchor(t, peierls_angle),
            )
            phase_conj = ar.do("conj", phase)
            channels = {"spinless": (t, "create", "annihilate")}
            selected = ("spinless",)

        aliases = {"u": "up", "↑": "up", "d": "down", "↓": "down"}
        selected = tuple(aliases.get(channel, channel) for channel in selected)
        unknown = [channel for channel in selected if channel not in channels]
        if unknown:
            raise ValueError("spin must be 'up', 'down', or None.")

        terms = []
        for channel in selected:
            coefficient, create, annihilate = channels[channel]
            if include_minus:
                coefficient = -coefficient
            terms.extend(
                (
                    (coefficient * phase, ((left, create), (right, annihilate))),
                    (coefficient * phase_conj, ((right, create), (left, annihilate))),
                )
            )
        return self.operator_term(terms, sites=(left, right))

    def hopping_operator(self, *, spin=None, peierls_angle=0.0):
        """Return the bare two-site hopping operator.

        The returned operator is ``c_a^dag c_b + c_b^dag c_a`` for each
        selected spin channel. It carries no ``-t`` coefficient and has
        canonical two-site locations; a Hamiltonian term mapping supplies the
        physical edge location and coefficient.
        """
        return self._hopping_operator_on_sites(
            0,
            1,
            spin=spin,
            peierls_angle=peierls_angle,
        )

    def hopping_term(self, edge, *, spin=None, t, peierls_angle=0.0):
        """Return ``-t`` times the hopping operator on ``edge``."""
        if t is None:
            raise TypeError("hopping_term requires explicit t=... .")
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge must contain exactly two site labels.") from exc
        t = _edge_parameter(t, left, right)
        return self._hopping_operator_on_sites(
            left,
            right,
            t=t,
            spin=spin,
            peierls_angle=peierls_angle,
            include_minus=True,
        )

    def interaction_operator(self):
        """Return the bare onsite doublon operator ``n_up n_down``."""
        if not self.spinful:
            raise ValueError(
                "Spinless fermions have no onsite doublon interaction; use "
                "density_operator() for the nearest-neighbor density term."
            )
        return self.operator_term(
            [(1.0, ((0, "double"),))],
            sites=(0,),
        )

    def interaction_term(self, site, *, U):
        """Return ``U n_up n_down`` on one physical site."""
        if not self.spinful:
            raise ValueError(
                "Spinless fermions have no onsite doublon interaction; use "
                "density_term(...) for the nearest-neighbor V interaction."
            )
        if U is None:
            raise TypeError("interaction_term requires explicit U=... .")
        U = _node_parameter(U, site)
        return self.operator_term(
            [(U, ((site, "double"),))],
            sites=(site,),
        )

    def chemical_potential_operator(self):
        """Return the bare onsite total-number operator."""
        if self.spinful:
            terms = [
                (1.0, ((0, "number_up"),)),
                (1.0, ((0, "number_down"),)),
            ]
        else:
            terms = [(1.0, ((0, "number"),))]
        return self.operator_term(terms, sites=(0,))

    def chemical_potential_term(self, site, *, mu):
        """Return ``-mu n`` on one physical site."""
        if mu is None:
            raise TypeError("chemical_potential_term requires explicit mu=... .")
        if self.spinful:
            mu = _node_parameter(mu, site)
            mu_up, mu_down = _as_spin_pair(mu, name="mu")
            terms = [
                (-mu_up, ((site, "number_up"),)),
                (-mu_down, ((site, "number_down"),)),
            ]
        else:
            terms = [(-_node_parameter(mu, site), ((site, "number"),))]
        return self.operator_term(terms, sites=(site,))

    def onsite_term(self, site, *, U=None, mu=0.0):
        """Return ``U n_up n_down - mu n`` on one site."""
        terms = []
        if self.spinful:
            if U is None:
                raise TypeError("onsite_term requires explicit U=... for spinful fermions.")
            terms.append((_node_parameter(U, site), ((site, "double"),)))
            mu = _node_parameter(mu, site)
            mu_up, mu_down = _as_spin_pair(mu, name="mu")
            terms.extend(
                (
                    (-mu_up, ((site, "number_up"),)),
                    (-mu_down, ((site, "number_down"),)),
                )
            )
        else:
            if U is not None:
                raise TypeError(
                    "onsite_term does not accept U=... for spinless fermions; "
                    "use V=... for nearest-neighbor density interactions."
                )
            terms.append((-_node_parameter(mu, site), ((site, "number"),)))
        return self.operator_term(terms, sites=(site,))

    def density_operator(self):
        """Return the bare nearest-neighbor density product operator."""
        if self.spinful:
            names = ("number_up", "number_down")
        else:
            names = ("number",)
        terms = [
            (1.0, ((0, left_name), (1, right_name)))
            for left_name in names
            for right_name in names
        ]
        return self.operator_term(terms, sites=(0, 1))

    def density_term(self, edge, *, V):
        """Return ``V n_i n_j`` on a physical edge."""
        if V is None:
            raise TypeError("density_term requires explicit V=... .")
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge must contain exactly two site labels.") from exc
        V = _edge_parameter(V, left, right)
        if self.spinful:
            names = ("number_up", "number_down")
        else:
            names = ("number",)
        terms = [
            (V, ((left, left_name), (right, right_name)))
            for left_name in names
            for right_name in names
        ]
        return self.operator_term(terms, sites=(left, right))

    def observable(self, name):
        """Return a cached one-site fermionic Symmray operator for ``name``."""
        name = self._operator_name(name)
        if name not in self._observable_cache:
            if name in {"sx", "sy"}:
                operator = self._spin_flip_operator(name)
            elif name in {"s_plus", "s_minus"}:
                self._require_spinful(f"{name} operators")
                operator = self.operator_term(
                    [(1.0, ((0, name),))],
                    sites=(0,),
                    charge=self.operator_charge(name),
                )
            else:
                operator = symm_operator_from_dense(
                    self.dense_operator(name),
                    self.physical_sectors,
                    symmetry=self.symmetry,
                    charge=self.operator_charge(name),
                    fermionic=True,
                    sites=1,
                )
            self._observable_cache[name] = _apply_to_array_blocks(operator, self.to_backend)
        return self._observable_cache[name]

    def _cached_gate(self, key, build):
        # A cached Symmray object may retain an autodiff graph. Do not reuse
        # such a gate across optimizer evaluations or repeated backward calls.
        dynamic = any(getattr(value, "requires_grad", False) for value in key)
        if dynamic:
            return build()
        key = tuple(repr(value) for value in key)
        if key not in self._gate_cache:
            self._gate_cache[key] = build()
        return self._gate_cache[key]

    def operator_gate(self, operator, theta, *, imaginary=False):
        """Exponentiate a native operator as ``exp(-i theta operator)``.

        ``operator`` can be a named one-site observable, one of the built-in
        two-site names ``sxx``, ``syy``, ``szz``, ``xy``, or ``heisenberg``,
        or an already-built native Symmray operator from :meth:`operator_term`.
        Gates require a charge-neutral operator because exponentiation adds the
        identity term and must remain in one homogeneous symmetry sector.
        """
        if isinstance(operator, str):
            name = self._operator_name(operator)
            factories = {
                "sxx": self.spin_x_correlator,
                "syy": self.spin_y_correlator,
                "szz": self.spin_z_correlator,
                "xy": self.xy_exchange_operator,
                "heisenberg": self.heisenberg_operator,
            }
            if name in factories:
                factory = factories[name]
            else:
                factory = lambda: self.observable(name)
            operator_key = ("name", name)
        else:
            factory = lambda: operator
            operator_key = ("term", _operator_content_fingerprint(operator))

        def build():
            term = factory()
            charge = getattr(term, "charge", None)
            if charge is not None and charge != self.zero_charge:
                raise ValueError(
                    "operator_gate requires a charge-neutral operator; "
                    f"got charge {charge!r} for symmetry {self.symmetry!r}."
                )
            gate = _gate_from_term(
                term,
                _fermion_scalar_anchor(theta, self.dtype)
                if self.to_backend is None
                else theta,
                imaginary=imaginary,
            )
            return _apply_to_array_blocks(gate, self.to_backend)

        if operator_key[1] is None:
            # No stable content fingerprint (an unrecognised operator type or
            # autodiff tensors): build without caching so that a recycled
            # Python ``id`` can never alias an unrelated operator's gate.
            return build()

        return self._cached_gate(
            ("operator", operator_key, theta, imaginary),
            build,
        )

    def spin_x_gate(self, theta, *, site=None, imaginary=False):
        """Return the native one-site ``exp(-i theta Sx)`` gate."""
        theta = self._resolve_operator_parameter(theta, site=site)
        return self.operator_gate("sx", theta, imaginary=imaginary)

    def spin_y_gate(self, theta, *, site=None, imaginary=False):
        """Return the native one-site ``exp(-i theta Sy)`` gate."""
        theta = self._resolve_operator_parameter(theta, site=site)
        return self.operator_gate("sy", theta, imaginary=imaginary)

    def spin_z_gate(self, theta, *, site=None, imaginary=False):
        """Return the native one-site ``exp(-i theta Sz)`` gate."""
        theta = self._resolve_operator_parameter(theta, site=site)
        return self.operator_gate("sz", theta, imaginary=imaginary)

    def spin_x_correlator_gate(self, theta, *, edge=None, imaginary=False):
        """Return the native two-site ``exp(-i theta Sx Sx)`` gate."""
        theta = self._resolve_operator_parameter(theta, edge=edge)
        return self.operator_gate("sxx", theta, imaginary=imaginary)

    def spin_y_correlator_gate(self, theta, *, edge=None, imaginary=False):
        """Return the native two-site ``exp(-i theta Sy Sy)`` gate."""
        theta = self._resolve_operator_parameter(theta, edge=edge)
        return self.operator_gate("syy", theta, imaginary=imaginary)

    def spin_z_correlator_gate(self, theta, *, edge=None, imaginary=False):
        """Return the native two-site ``exp(-i theta Sz Sz)`` gate."""
        theta = self._resolve_operator_parameter(theta, edge=edge)
        return self.operator_gate("szz", theta, imaginary=imaginary)

    def xy_exchange_gate(self, theta, *, edge=None, imaginary=False):
        """Return the native two-site XY-exchange gate."""
        theta = self._resolve_operator_parameter(theta, edge=edge)
        return self.operator_gate("xy", theta, imaginary=imaginary)

    def heisenberg_gate(self, theta, *, edge=None, imaginary=False):
        """Return the native two-site Heisenberg gate."""
        theta = self._resolve_operator_parameter(theta, edge=edge)
        return self.operator_gate("heisenberg", theta, imaginary=imaginary)

    # Short gate spellings match the operator aliases and are convenient in
    # small native gate streams.
    sx_gate = spin_x_gate
    sy_gate = spin_y_gate
    sz_gate = spin_z_gate
    sxx_gate = spin_x_correlator_gate
    syy_gate = spin_y_correlator_gate
    szz_gate = spin_z_correlator_gate
    xy_gate = xy_exchange_gate

    def interaction_gate(self, dt, *, site=None, U, imaginary=False):
        """Return the exact onsite interaction gate.

        With a site-dependent ``U`` mapping or callable, pass ``site`` so the
        corresponding local value can be selected. This gate is defined for
        spinful fermions as ``exp(-i dt U n_up n_down)``.
        """
        if not self.spinful:
            raise ValueError(
                "Spinless fermions have no onsite doublon interaction; use "
                "density_gate(...) for the nearest-neighbor V interaction."
            )
        if U is None:
            raise TypeError("interaction_gate requires explicit U=... .")
        U = U if site is None else _node_parameter(U, site)
        theta = _fermion_scalar_anchor(dt * U, self.dtype)

        def build():
            gate = fermion_interaction_param_gen(
                (theta,),
                symmetry=self.symmetry,
                imaginary=imaginary,
            )
            return _apply_to_array_blocks(gate, self.to_backend)

        return self._cached_gate(("interaction", dt, site, U, imaginary), build)

    def onsite_gate(self, dt, *, site=None, U=None, mu=0.0, imaginary=False):
        """Return the complete one-site Hubbard gate.

        The generated gate represents ``U n_up n_down - mu n`` for spinful
        fermions and ``-mu n`` for spinless fermions. ``U`` and ``mu`` may be
        site-dependent mappings or callables when ``site`` is supplied.
        """
        if not self.spinful and U is not None:
            raise TypeError(
                "onsite_gate does not accept U=... for spinless fermions; "
                "use V=... for nearest-neighbor density interactions."
            )

        if site is not None:
            U = _node_parameter(U, site)
            mu = _node_parameter(mu, site)

        dt = _fermion_scalar_anchor(dt, self.dtype)

        if self.spinful:
            if U is None:
                raise TypeError("onsite_gate requires explicit U=... for spinful fermions.")
            mu_up, mu_down = _as_spin_pair(mu, name="mu")
            U_site = U
            diagonal = (
                0.0,
                -mu_up,
                -mu_down,
                U_site - mu_up - mu_down,
            )
        else:
            diagonal = (0.0, -mu)

        def build():
            gate = _fermion_diagonal_gate_param_gen(
                (dt,),
                diagonal,
                self.physical_sectors,
                symmetry=self.symmetry,
                imaginary=imaginary,
                sites=1,
            )
            return _apply_to_array_blocks(gate, self.to_backend)

        return self._cached_gate(("onsite", dt, site, U, mu, imaginary), build)

    def hopping_gate(self, dt, *, t, peierls_angle=0.0, imaginary=False):
        """Return a two-site native fermionic hopping gate with Peierls phase."""
        if t is None:
            raise TypeError("hopping_gate requires explicit t=... .")

        dt = _fermion_scalar_anchor(dt, self.dtype)

        def build():
            if not self.spinful:
                gate = _spinless_hopping_gate(
                    self.symmetry,
                    dt,
                    t=t,
                    peierls_angle=peierls_angle,
                    imaginary=imaginary,
                )
            else:
                gate = _spinful_hopping_gate(
                    self.symmetry,
                    dt,
                    t=t,
                    peierls_angle=peierls_angle,
                    imaginary=imaginary,
                )
            return _apply_to_array_blocks(gate, self.to_backend)

        return self._cached_gate(("hopping", dt, t, peierls_angle, imaginary), build)

    def density_gate(self, dt, *, V, imaginary=False):
        """Return the nearest-neighbor density interaction gate.

        For spinless fermions this is ``V n_i n_j``. For spinful fermions it
        is ``V (n_up + n_down)_i (n_up + n_down)_j``.
        """
        if V is None:
            raise TypeError("density_gate requires explicit V=... .")
        theta = _fermion_scalar_anchor(dt * V, self.dtype)

        def build():
            if self.spinful:
                gate = _fermion_spinful_density_param_gen(
                    (theta,), symmetry=self.symmetry, imaginary=imaginary
                )
            else:
                gate = fermion_density_param_gen(
                    (theta,), symmetry=self.symmetry, imaginary=imaginary
                )
            return _apply_to_array_blocks(gate, self.to_backend)

        return self._cached_gate(("density", dt, V, imaginary), build)

    def chemical_potential_gate(self, dt, *, mu, site=None, imaginary=False):
        """Return the chemical-potential part of an onsite gate."""
        if mu is None:
            raise TypeError("chemical_potential_gate requires explicit mu=... .")
        mu = mu if site is None else _node_parameter(mu, site)
        if self.spinful:
            mu_up, mu_down = _as_spin_pair(mu, name="mu")
            diagonal = (0.0, -mu_up, -mu_down, -mu_up - mu_down)
        else:
            diagonal = (0.0, -mu)

        dt = _fermion_scalar_anchor(dt, self.dtype)

        def build():
            gate = _fermion_diagonal_gate_param_gen(
                (dt,),
                diagonal,
                self.physical_sectors,
                symmetry=self.symmetry,
                imaginary=imaginary,
                sites=1,
            )
            return _apply_to_array_blocks(gate, self.to_backend)

        return self._cached_gate(("chemical", dt, site, mu, imaginary), build)

    def gate(self, name, dt, *, site=None, where=None, imaginary=False, **params):
        """Build a named native gate using the local fermionic conventions."""
        if "edge" in params and where is not None:
            raise TypeError(
                "Fermion.gate accepts at most one of where=... and edge=... ."
            )
        edge = params.pop("edge", where)
        del where  # Gate locations belong to the stream entry, not the tensor.
        name = str(name).lower().replace("-", "_")

        def require(parameter):
            try:
                return params.pop(parameter)
            except KeyError as exc:
                raise TypeError(
                    f"Fermion.gate({name!r}, ...) requires explicit "
                    f"{parameter}=... ."
                ) from exc

        def finish(gate, *, accepts_site=False, accepts_edge=False):
            if site is not None and not accepts_site:
                raise TypeError(
                    f"Fermion.gate({name!r}, ...) does not accept site=... ."
                )
            if edge is not None and not accepts_edge:
                raise TypeError(
                    f"Fermion.gate({name!r}, ...) does not accept edge=... ."
                )
            if params:
                names = ", ".join(sorted(params))
                raise TypeError(
                    f"Unexpected Fermion.gate parameter(s) for {name!r}: {names}."
                )
            return gate

        if name in {"sx", "spin_x"}:
            return finish(
                self.spin_x_gate(dt, site=site, imaginary=imaginary),
                accepts_site=True,
            )
        if name in {"sy", "spin_y"}:
            return finish(
                self.spin_y_gate(dt, site=site, imaginary=imaginary),
                accepts_site=True,
            )
        if name in {"sz", "spin_z"}:
            return finish(
                self.spin_z_gate(dt, site=site, imaginary=imaginary),
                accepts_site=True,
            )
        if name in {"sxx", "spin_x_x", "sx_sx"}:
            return finish(
                self.spin_x_correlator_gate(dt, edge=edge, imaginary=imaginary),
                accepts_edge=True,
            )
        if name in {"syy", "spin_y_y", "sy_sy"}:
            return finish(
                self.spin_y_correlator_gate(dt, edge=edge, imaginary=imaginary),
                accepts_edge=True,
            )
        if name in {"szz", "spin_z_z", "sz_sz"}:
            return finish(
                self.spin_z_correlator_gate(dt, edge=edge, imaginary=imaginary),
                accepts_edge=True,
            )
        if name in {"xy", "xy_exchange"}:
            return finish(
                self.xy_exchange_gate(dt, edge=edge, imaginary=imaginary),
                accepts_edge=True,
            )
        if name in {"heisenberg", "heis"}:
            return finish(
                self.heisenberg_gate(dt, edge=edge, imaginary=imaginary),
                accepts_edge=True,
            )
        if name in {"onsite", "hubbard_onsite"}:
            return finish(
                self.onsite_gate(
                    dt,
                    site=site,
                    U=params.pop("U", None),
                    mu=params.pop("mu", 0.0),
                    imaginary=imaginary,
                ),
                accepts_site=True,
            )
        if name in {"interaction", "onsite_interaction", "doublon"}:
            return finish(
                self.interaction_gate(
                    dt,
                    site=site,
                    U=require("U"),
                    imaginary=imaginary,
                ),
                accepts_site=True,
            )
        if name in {"hopping", "hop"}:
            return finish(
                self.hopping_gate(
                    dt,
                    t=require("t"),
                    peierls_angle=params.pop("peierls_angle", 0.0),
                    imaginary=imaginary,
                ),
            )
        if name in {"density", "density_interaction", "nn"}:
            return finish(
                self.density_gate(
                    dt,
                    V=require("V"),
                    imaginary=imaginary,
                ),
            )
        if name in {"chemical", "chemical_potential", "mu"}:
            return finish(
                self.chemical_potential_gate(
                    dt,
                    mu=require("mu"),
                    site=site,
                    imaginary=imaginary,
                ),
                accepts_site=True,
            )
        raise ValueError(f"Unknown fermion gate {name!r}.")

    def param_gate(self, name, params, *, imaginary=False, **kwargs):
        """Build a gate from a Quimb-style parameter sequence."""
        name = str(name).lower().replace("-", "_")
        params = tuple(params)
        if params:
            params = (
                _fermion_scalar_anchor(params[0], self.dtype),
                *params[1:],
            )

        def finish(gate):
            if kwargs:
                names = ", ".join(sorted(kwargs))
                raise TypeError(
                    f"Unexpected Fermion.param_gate parameter(s) for {name!r}: "
                    f"{names}."
                )
            return gate

        if name in {"interaction", "onsite_interaction", "doublon"}:
            if not self.spinful:
                raise ValueError("Spinless fermions do not have doublon gates.")
            return finish(
                fermion_interaction_param_gen(
                    params,
                    symmetry=self.symmetry,
                    imaginary=imaginary,
                )
            )
        if name in {"density", "density_interaction", "nn"}:
            if self.spinful:
                raise ValueError("Spinful density gates are not the onsite interaction gate.")
            return finish(
                fermion_density_param_gen(
                    params,
                    symmetry=self.symmetry,
                    imaginary=imaginary,
                )
            )
        if name in {"hopping", "hop"}:
            return finish(
                fermion_hopping_param_gen(
                    params,
                    spinful=self.spinful,
                    symmetry=self.symmetry,
                    imaginary=imaginary,
                    peierls_angle=kwargs.pop("peierls_angle", 0.0),
                )
            )
        raise ValueError(f"Unknown parameterized fermion gate {name!r}.")

    def exponential(
        self,
        terms,
        dt,
        *,
        sites=None,
        bases=None,
        imaginary=False,
        like=None,
    ):
        """Build ``exp(-i dt H)`` for a neutral local fermion Hamiltonian.

        The convenient term format is ``(coefficient, operators)`` where each
        operator is ``(site, name)``. Names are local fermionic monomials such
        as ``create_up``, ``annihilate_down``, ``number_up``, ``double``, or
        ``create``/``annihilate`` for spinless fermions. ``sites`` fixes the
        local basis order; when omitted it is inferred from the term order.

        Advanced callers can pass Symmray ``FermionicOperator`` terms together
        with explicit ``bases``. The exponential must be a neutral operator so
        it can be represented in one conserved Symmray charge sector.
        """
        entries = tuple(terms)
        if not entries:
            raise ValueError("terms must contain at least one local term.")

        if like is None:
            like = _fermion_scalar_anchor(0.0, self.dtype)

        if bases is not None:
            bases = tuple(tuple(basis) for basis in bases)
            if not bases:
                raise ValueError("bases must contain at least one local basis.")
            raw_terms = entries
        else:
            if sites is None:
                inferred_sites = []
                for entry in entries:
                    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                        raise ValueError(
                            "terms must have form (coefficient, operators)."
                        )
                    for reference in tuple(entry[1]):
                        if not isinstance(reference, (tuple, list)) or len(reference) != 2:
                            raise ValueError(
                                "operator references must have form (site, name)."
                            )
                        site = reference[0]
                        if site not in inferred_sites:
                            inferred_sites.append(site)
                sites = tuple(inferred_sites)
            elif isinstance(sites, (str, bytes)):
                sites = (sites,)
            else:
                try:
                    sites = tuple(sites)
                except TypeError:
                    sites = (sites,)
            if not sites:
                raise ValueError("sites must contain at least one local site.")
            try:
                site_positions = {site: position for position, site in enumerate(sites)}
            except TypeError as exc:
                raise TypeError("sites must contain hashable labels.") from exc
            bases, local_operators = _fermion_generic_local_modes(self.spinful, sites)
            raw_terms = []
            for entry in entries:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ValueError("terms must have form (coefficient, operators).")
                coefficient, references = entry
                expanded = []
                for reference in tuple(references):
                    if not isinstance(reference, (tuple, list)) or len(reference) != 2:
                        raise ValueError(
                            "operator references must have form (site, name)."
                        )
                    site, name = reference
                    if site not in site_positions:
                        raise ValueError(
                            f"operator reference uses site {site!r}, which is not in sites."
                        )
                    name = self._operator_name(name)
                    try:
                        expanded.extend(local_operators[site][name])
                    except KeyError as exc:
                        allowed = ", ".join(sorted(local_operators[site]))
                        raise ValueError(
                            f"Unknown generic fermion monomial {name!r}; "
                            f"expected one of {allowed}."
                        ) from exc
                raw_terms.append((coefficient, tuple(expanded)))
            raw_terms = tuple(raw_terms)

        gate = _fermion_terms_exponential_gate(
            raw_terms,
            bases,
            self.physical_sectors,
            symmetry=self.symmetry,
            dt=(
                _fermion_scalar_anchor(dt, self.dtype)
                if like is None
                else dt
            ),
            imaginary=imaginary,
            like=like,
        )
        return _apply_to_array_blocks(gate, self.to_backend)

    local_exponential = exponential

    @staticmethod
    def edge_coloring_layers(edges):
        """Partition edges into deterministic vertex-disjoint layers."""
        return _edge_coloring_layers(edges)

    def gate_stream(
        self,
        edges,
        dt,
        *,
        sites=None,
        order=2,
        peierls_angle=0.0,
        imaginary=False,
        t=None,
        U=None,
        V=0.0,
        mu=0.0,
        field_x=0.0,
        field_y=0.0,
        field_z=0.0,
        pairing=0.0,
        pairing_phase=0.0,
    ):
        """Return a canonical fermion gate stream with explicit couplings."""
        if order not in {1, 2, 4}:
            raise ValueError("order must be 1, 2, or 4.")
        if t is None:
            raise TypeError("gate_stream requires explicit t=... .")
        if self.spinful and U is None:
            raise TypeError("gate_stream requires explicit U=... for spinful fermions.")
        if not self.spinful and U is not None:
            raise TypeError(
                "gate_stream does not accept U=... for spinless fermions; "
                "use V=... for nearest-neighbor density interactions."
            )
        edges = _as_edges(edges)
        sites = _sites_from_edges(edges, sites)

        if order == 4:
            return _yoshida4_stream(
                lambda sub_dt: self.strang_gate_stream(
                    edges,
                    sub_dt,
                    sites=sites,
                    peierls_angle=peierls_angle,
                    imaginary=imaginary,
                    t=t,
                    U=U,
                    V=V,
                    mu=mu,
                    field_x=field_x,
                    field_y=field_y,
                    field_z=field_z,
                    pairing=pairing,
                    pairing_phase=pairing_phase,
                ),
                dt,
                imaginary=imaginary,
            )

        if order == 2:
            return self.strang_gate_stream(
                edges,
                dt,
                sites=sites,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                t=t,
                U=U,
                V=V,
                mu=mu,
                field_x=field_x,
                field_y=field_y,
                field_z=field_z,
                pairing=pairing,
                pairing_phase=pairing_phase,
            )

        entries = []
        entries.extend(
            (
                self.onsite_gate(
                    dt,
                    site=site,
                    U=U,
                    mu=mu,
                    imaginary=imaginary,
                ),
                site,
            )
            for site in sites
        )
        for coupling, gate in (
            (field_x, self.spin_x_gate),
            (field_y, self.spin_y_gate),
            (field_z, self.spin_z_gate),
        ):
            if _coupling_is_active(coupling):
                entries.extend(
                    (
                        gate(
                            dt * _node_parameter(coupling, site),
                            imaginary=imaginary,
                        ),
                        site,
                    )
                    for site in sites
                )
        if _coupling_is_active(V):
            entries.extend(
                (
                    self.density_gate(
                        dt,
                        V=_edge_parameter(V, left, right),
                        imaginary=imaginary,
                    ),
                    (left, right),
                )
                for left, right in edges
            )
        if _coupling_is_active(pairing):
            entries.extend(
                (
                    self.pairing_gate(
                        dt,
                        edge=(left, right),
                        coefficient=_edge_parameter(pairing, left, right),
                        phase=_edge_angle_parameter(pairing_phase, left, right),
                        imaginary=imaginary,
                    ),
                    (left, right),
                )
                for left, right in edges
            )
        entries.extend(
            (
                self.hopping_gate(
                    dt,
                    t=_edge_parameter(t, left, right),
                    peierls_angle=_edge_angle_parameter(peierls_angle, left, right),
                    imaginary=imaginary,
                ),
                (left, right),
            )
            for left, right in edges
        )
        return SymGateStream(
            entries,
            hamiltonian=self.hamiltonian(
                edges,
                sites=sites,
                t=t,
                U=U,
                V=V,
                mu=mu,
                field_x=field_x,
                field_y=field_y,
                field_z=field_z,
                pairing=pairing,
                pairing_phase=pairing_phase,
            ),
            dt=dt,
            imaginary=imaginary,
            order=1,
        )

    def strang_gate_stream(
        self,
        edges,
        dt,
        *,
        sites=None,
        peierls_angle=0.0,
        imaginary=False,
        t=None,
        U=None,
        V=0.0,
        mu=0.0,
        field_x=0.0,
        field_y=0.0,
        field_z=0.0,
        pairing=0.0,
        pairing_phase=0.0,
    ):
        """Return an edge-coloured second-order stream with explicit couplings."""
        if t is None:
            raise TypeError("strang_gate_stream requires explicit t=... .")
        if self.spinful and U is None:
            raise TypeError(
                "strang_gate_stream requires explicit U=... for spinful fermions."
            )
        if not self.spinful and U is not None:
            raise TypeError(
                "strang_gate_stream does not accept U=... for spinless fermions; "
                "use V=... for nearest-neighbor density interactions."
            )
        edges = _as_edges(edges)
        sites = _sites_from_edges(edges, sites)
        half_dt = dt / 2
        layers = self.edge_coloring_layers(edges)
        entries = [
            (
                self.onsite_gate(
                    half_dt,
                    site=site,
                    U=U,
                    mu=mu,
                    imaginary=imaginary,
                ),
                site,
            )
            for site in sites
        ]
        fields = (
            (field_x, self.spin_x_gate),
            (field_y, self.spin_y_gate),
            (field_z, self.spin_z_gate),
        )
        for coupling, gate in fields:
            if _coupling_is_active(coupling):
                entries.extend(
                    (
                        gate(
                            half_dt * _node_parameter(coupling, site),
                            imaginary=imaginary,
                        ),
                        site,
                    )
                    for site in sites
                )
        if _coupling_is_active(V):
            entries.extend(
                (
                    self.density_gate(
                        half_dt,
                        V=_edge_parameter(V, left, right),
                        imaginary=imaginary,
                    ),
                    (left, right),
                )
                for left, right in edges
            )
        if _coupling_is_active(pairing):
            quarter_dt = dt / 4
            for layer in layers:
                entries.extend(
                    (
                        self.pairing_gate(
                            quarter_dt,
                            edge=(left, right),
                            coefficient=_edge_parameter(pairing, left, right),
                            phase=_edge_angle_parameter(
                                pairing_phase, left, right
                            ),
                            imaginary=imaginary,
                        ),
                        (left, right),
                    )
                    for left, right in layer
                )
            for layer in reversed(layers):
                entries.extend(
                    (
                        self.pairing_gate(
                            quarter_dt,
                            edge=(left, right),
                            coefficient=_edge_parameter(pairing, left, right),
                            phase=_edge_angle_parameter(
                                pairing_phase, left, right
                            ),
                            imaginary=imaginary,
                        ),
                        (left, right),
                    )
                    for left, right in reversed(layer)
                )
        for layer in layers:
            entries.extend(
                (
                    self.hopping_gate(
                        half_dt,
                        t=_edge_parameter(t, left, right),
                        peierls_angle=_edge_angle_parameter(peierls_angle, left, right),
                        imaginary=imaginary,
                    ),
                    (left, right),
                )
                for left, right in layer
            )
        for layer in reversed(layers):
            entries.extend(
                (
                    self.hopping_gate(
                        half_dt,
                        t=_edge_parameter(t, left, right),
                        peierls_angle=_edge_angle_parameter(peierls_angle, left, right),
                        imaginary=imaginary,
                    ),
                    (left, right),
                )
                for left, right in layer
            )
        if _coupling_is_active(pairing):
            quarter_dt = dt / 4
            for layer in reversed(layers):
                entries.extend(
                    (
                        self.pairing_gate(
                            quarter_dt,
                            edge=(left, right),
                            coefficient=_edge_parameter(pairing, left, right),
                            phase=_edge_angle_parameter(
                                pairing_phase, left, right
                            ),
                            imaginary=imaginary,
                        ),
                        (left, right),
                    )
                    for left, right in reversed(layer)
                )
            for layer in layers:
                entries.extend(
                    (
                        self.pairing_gate(
                            quarter_dt,
                            edge=(left, right),
                            coefficient=_edge_parameter(pairing, left, right),
                            phase=_edge_angle_parameter(
                                pairing_phase, left, right
                            ),
                            imaginary=imaginary,
                        ),
                        (left, right),
                    )
                    for left, right in layer
                )
        if _coupling_is_active(V):
            entries.extend(
                (
                    self.density_gate(
                        half_dt,
                        V=_edge_parameter(V, left, right),
                        imaginary=imaginary,
                    ),
                    (left, right),
                )
                for left, right in edges
            )
        for coupling, gate in reversed(fields):
            if _coupling_is_active(coupling):
                entries.extend(
                    (
                        gate(
                            half_dt * _node_parameter(coupling, site),
                            imaginary=imaginary,
                        ),
                        site,
                    )
                    for site in sites
                )
        entries.extend(
            (
                self.onsite_gate(
                    half_dt,
                    site=site,
                    U=U,
                    mu=mu,
                    imaginary=imaginary,
                ),
                site,
            )
            for site in sites
        )
        return SymGateStream(
            entries,
            hamiltonian=self.hamiltonian(
                edges,
                sites=sites,
                t=t,
                U=U,
                V=V,
                mu=mu,
                field_x=field_x,
                field_y=field_y,
                field_z=field_z,
                pairing=pairing,
                pairing_phase=pairing_phase,
            ),
            dt=dt,
            imaginary=imaginary,
            order=2,
        )

    def _validate_hamiltonian_terms(self, terms):
        """Validate native local terms against this Fermion's local space."""
        terms = dict(terms)
        coordinate_sites = _term_mapping_uses_coordinate_sites(terms)
        expected_physical = {
            charge: int(size)
            for charge, size in self.physical_sectors.items()
        }
        backends = set()

        for where, term in terms.items():
            support = _as_term_where(
                where,
                coordinate_sites=coordinate_sites,
            )
            if not _is_fermionic_symmray_array(term):
                raise TypeError(
                    "Fermion.hamiltonian requires native fermionic Symmray "
                    f"arrays; term at {where!r} is {type(term).__name__}."
                )
            if str(getattr(term, "symmetry", None)) != self.symmetry:
                raise ValueError(
                    f"Term at {where!r} has symmetry "
                    f"{getattr(term, 'symmetry', None)!r}, expected "
                    f"{self.symmetry!r}."
                )
            indices = tuple(getattr(term, "indices", ()))
            expected_rank = 2 * len(support)
            if len(indices) != expected_rank:
                raise ValueError(
                    f"Term at {where!r} has rank {len(indices)}, but its "
                    f"{len(support)}-site key requires rank {expected_rank}."
                )
            for axis, index in enumerate(indices):
                actual_physical = {
                    charge: int(size)
                    for charge, size in dict(getattr(index, "chargemap", {})).items()
                }
                if actual_physical != expected_physical:
                    raise ValueError(
                        f"Term at {where!r} axis {axis} has physical sectors "
                        f"{actual_physical!r}, expected {expected_physical!r}."
                    )
            for block in getattr(term, "blocks", {}).values():
                backends.add(ar.infer_backend(block))

        if len(backends) > 1:
            raise TypeError(
                "Fermion.hamiltonian terms use mixed array backends "
                f"{sorted(backends)!r}. Supply to_backend=... so every native "
                "block is converted consistently."
            )

    def hamiltonian(
        self,
        terms_or_edges,
        *,
        sites=None,
        t=None,
        U=None,
        V=0.0,
        mu=0.0,
        field_x=0.0,
        field_y=0.0,
        field_z=0.0,
        pairing=0.0,
        pairing_phase=0.0,
        flat=False,
        to_backend=None,
    ):
        """Validate explicit terms or build a model only from explicit couplings.

        The canonical form is a mapping from one-site or two-site locations to
        native fermionic Symmray arrays. It is checked for symmetry, physical
        sectors, support rank, and backend consistency before being bundled in
        a :class:`SymHamiltonian`. Passing lattice edges remains a compact
        convenience, but requires its couplings explicitly; no coupling is
        stored on :class:`Fermion`.
        """
        to_backend = self.to_backend if to_backend is None else to_backend
        extra_couplings = (field_x, field_y, field_z, pairing)
        if isinstance(terms_or_edges, Mapping):
            if (
                any(value is not None for value in (t, U))
                or _coupling_is_active(V)
                or _coupling_is_active(mu)
                or any(_coupling_is_active(value) for value in extra_couplings)
            ):
                raise TypeError(
                    "When passing explicit terms, put every coupling in the "
                    "native arrays rather than passing model couplings again."
                )
            terms = _apply_to_hamiltonian_terms(terms_or_edges, to_backend)
            self._validate_hamiltonian_terms(terms)
            return SymHamiltonian.from_terms(
                self.model,
                self.symmetry,
                terms,
                parameters={},
            )

        if t is None:
            raise TypeError("hamiltonian(edges, ...) requires explicit t=... .")
        if self.spinful and U is None:
            raise TypeError(
                "hamiltonian(edges, ...) requires explicit U=... for spinful fermions."
            )
        if not self.spinful and U is not None:
            raise TypeError(
                "hamiltonian(edges, ...) does not accept U=... for spinless "
                "fermions; use V=... for nearest-neighbor density interactions."
            )
        edges = _as_edges(terms_or_edges)
        params = {"t": t, "V": V, "mu": mu}
        if self.spinful:
            params["U"] = U
        hamiltonian = SymHamiltonian.from_edges(
            self.model,
            self.symmetry,
            edges,
            flat=flat,
            to_backend=to_backend,
            **params,
        )
        if not any(_coupling_is_active(value) for value in extra_couplings):
            self._validate_hamiltonian_terms(hamiltonian.terms)
            return hamiltonian

        sites = _sites_from_edges(edges, sites)
        terms = dict(hamiltonian.terms)
        for coupling, build_term in (
            (field_x, self.spin_x_term),
            (field_y, self.spin_y_term),
            (field_z, self.spin_z_term),
        ):
            if _coupling_is_active(coupling):
                for site in sites:
                    where = (site,)
                    term = build_term(site, field=coupling)
                    terms[where] = terms[where] + term if where in terms else term
        if _coupling_is_active(pairing):
            for left, right in edges:
                edge = (left, right)
                terms[edge] = terms[edge] + self.pairing_operator(
                    edge,
                    coefficient=_edge_parameter(pairing, left, right),
                    phase=_edge_angle_parameter(pairing_phase, left, right),
                )
        parameters = {
            **params,
            "field_x": field_x,
            "field_y": field_y,
            "field_z": field_z,
            "pairing": pairing,
            "pairing_phase": pairing_phase,
        }
        hamiltonian = SymHamiltonian.from_terms(
            self.model,
            self.symmetry,
            terms,
            parameters=parameters,
        )
        self._validate_hamiltonian_terms(hamiltonian.terms)
        return hamiltonian

    def build_mpo(
        self,
        terms_or_edges=None,
        *,
        hamiltonian=None,
        L=None,
        mapper=None,
        idx2coo=None,
        coo2idx=None,
        max_bond=None,
        cutoff=1e-12,
        compress=True,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        dtype=None,
        fermionic=True,
        charge_sectors=False,
        to_backend=None,
        **params,
    ):
        """Build the model-facing one-dimensional MPO.

        This is the canonical ``Fermion`` entry point for a chain MPO. Pass
        either model terms or edges in ``terms_or_edges`` or an existing
        :class:`SymHamiltonian` with ``hamiltonian=``. Native graded
        ``FermionicArray`` tensors are built by default; pass
        ``fermionic=False`` for the explicit Jordan--Wigner compatibility
        MPO.

        ``to_mpo`` is retained as a compatibility alias of this method.
        ``t``, ``U``/``V``, and ``mu`` remain explicit build parameters and
        are forwarded to :meth:`hamiltonian`.
        """
        to_backend = self.to_backend if to_backend is None else to_backend
        if hamiltonian is not None:
            if terms_or_edges is not None:
                raise TypeError(
                    "Pass either terms_or_edges or hamiltonian, not both."
                )
            if not isinstance(hamiltonian, SymHamiltonian):
                raise TypeError("hamiltonian must be a SymHamiltonian instance.")
            target = hamiltonian
        elif isinstance(terms_or_edges, SymHamiltonian):
            target = terms_or_edges
        else:
            if terms_or_edges is None:
                raise TypeError("build_mpo requires terms_or_edges or hamiltonian.")
            target = self.hamiltonian(
                terms_or_edges,
                to_backend=to_backend,
                **params,
            )
            params = {}

        if params:
            names = ", ".join(sorted(params))
            raise TypeError(
                "Model parameters cannot be supplied with an existing "
                f"SymHamiltonian: {names}."
            )
        return target.to_mpo(
            L=L,
            mapper=mapper,
            idx2coo=idx2coo,
            coo2idx=coo2idx,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
            dtype=dtype,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            to_backend=to_backend,
        )

    # ``to_mpo`` was the original model-facing name. Keep one implementation
    # so the two spellings cannot drift in defaults or supported arguments.
    to_mpo = build_mpo

    def build_pepo(
        self,
        terms_or_edges=None,
        *,
        hamiltonian=None,
        Lx=None,
        Ly=None,
        mapper=None,
        max_bond=None,
        cutoff=1e-12,
        compress=True,
        cyclic=False,
        cycle_bond_dim=1,
        dtype=None,
        fermionic=True,
        charge_sectors=False,
        to_backend=None,
        **params,
    ):
        """Build a 2D PEPO from this fermion model.

        This is the model-facing shorthand for :meth:`to_pepo`. Native
        graded construction is selected by default; pass ``fermionic=False``
        for the compatibility Jordan--Wigner MPO before PEPO embedding.
        Coordinate-keyed explicit terms can be supplied with a
        ``mapper=OneDMap(...)``.
        """
        return self.to_pepo(
            terms_or_edges,
            hamiltonian=hamiltonian,
            Lx=Lx,
            Ly=Ly,
            mapper=mapper,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            cyclic=cyclic,
            cycle_bond_dim=cycle_bond_dim,
            dtype=dtype,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            to_backend=to_backend,
            **params,
        )

    def build_tree_operator(
        self,
        terms_or_edges=None,
        *,
        hamiltonian=None,
        tree=None,
        plan=None,
        max_bond=None,
        cutoff=1e-12,
        compress=True,
        dtype=None,
        fermionic=True,
        charge_sectors=False,
        to_backend=None,
        **params,
    ):
        """Build the native :class:`pepsy.TreeMPO` for a selected plan.

        ``tree`` and ``plan`` are aliases.  The returned object is the native
        ``TreeMPO``; its TreePlan representation is exposed through
        ``.tree_networks`` and ``.expectation``. A linear chain MPO, if
        needed, is built separately with ``Fermion.to_mpo``.
        Native ``fermionic=True`` keeps Symmray's graded tensors intact for
        U1, U1U1, and other supported symmetries.

        Mixed operator charges are exposed as one public ``TreeMPO`` whose
        internal ``tree_networks`` keep one homogeneous native network per
        charge. Pass ``charge_sectors=True`` only when separate sector
        objects are specifically desired.

        This is the canonical ``Fermion`` tree-operator entry point.
        ``to_tree_mpo`` and ``build_tree_mpo`` remain compatibility aliases.
        """
        if tree is not None and plan is not None:
            raise TypeError("pass only one of tree= or plan=")
        plan = tree if tree is not None else plan
        if plan is None:
            raise TypeError("build_tree_operator requires tree= or plan=.")
        if hamiltonian is not None:
            if terms_or_edges is not None:
                raise TypeError(
                    "Pass either terms_or_edges or hamiltonian, not both."
                )
            if not isinstance(hamiltonian, SymHamiltonian):
                raise TypeError("hamiltonian must be a SymHamiltonian instance.")
            target = hamiltonian
        elif isinstance(terms_or_edges, SymHamiltonian):
            target = terms_or_edges
        else:
            if terms_or_edges is None:
                raise TypeError(
                    "build_tree_operator requires terms_or_edges or hamiltonian."
                )
            target = self.hamiltonian(
                terms_or_edges,
                to_backend=to_backend,
                **params,
            )
            params = {}
        if params:
            names = ", ".join(sorted(params))
            raise TypeError(
                "Model parameters cannot be supplied with an existing "
                f"SymHamiltonian: {names}."
            )
        from ..optimizers.tree import build_tree_operator

        return build_tree_operator(
            plan,
            target,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            dtype=dtype,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            to_backend=to_backend,
        )

    # Keep the historical spellings as aliases of the one canonical builder.
    to_tree_mpo = build_tree_operator
    build_tree_mpo = build_tree_operator

    def to_pepo(
        self,
        terms_or_edges=None,
        *,
        hamiltonian=None,
        Lx=None,
        Ly=None,
        mapper=None,
        max_bond=None,
        cutoff=1e-12,
        compress=True,
        cyclic=False,
        cycle_bond_dim=1,
        dtype=None,
        fermionic=True,
        charge_sectors=False,
        to_backend=None,
        **params,
    ):
        """Build a native fermionic PEPO on a 2D lattice.

        The operator terms can be keyed by lattice coordinates, for example
        ``{((0, 1), (2, 2)): term}``, where ``term`` is a native
        :class:`symmray.FermionicArray` returned by :meth:`operator_term`.
        Use ``{((0, 1),): term}`` for a one-site coordinate term so it is not
        confused with a one-dimensional ``(i, j)`` edge.
        The native graded MPO assembler is used internally with the supplied
        one-dimensional map, and the result is embedded as a snake-style
        PEPO. This keeps the fermionic charge and grading metadata intact,
        including homogeneous nonzero operator charge and odd-parity dummy
        modes; it does not pass through a dense or Jordan--Wigner
        representation when ``fermionic=True``.

        Parameters
        ----------
        terms_or_edges : mapping, sequence, or SymHamiltonian
            Explicit coordinate-keyed native terms, built-in model edges, or
            an already assembled Hamiltonian.
        hamiltonian : SymHamiltonian, optional
            Existing Hamiltonian. Pass either this or ``terms_or_edges``.
        Lx, Ly : int
            Dimensions of the 2D PEPO lattice.
        mapper : OneDMap, optional
            One-dimensional ordering used for the native fermionic channels.
            The PEPO embedding currently requires ``snake`` or
            ``snake-row-major`` ordering.
        max_bond, cutoff, compress
            Forwarded to native MPO construction before PEPO embedding.
        cyclic : bool, optional
            Add dimension-``cycle_bond_dim`` PEPO bonds around both lattice
            directions after embedding.
        dtype, fermionic, to_backend
            Forwarded to :meth:`to_mpo`. Keep ``fermionic=True`` for the
            native graded path; ``False`` explicitly selects the compatibility
            MPO path.
        charge_sectors : bool, optional
            Return ``{charge: PEPO}`` for mixed-charge term collections.

        Returns
        -------
        qtn.PEPO
            A PEPO with coordinate tags ``I{x},{y}``, input indices
            ``k{x},{y}``, and output indices ``b{x},{y}``.

        Notes
        -----
        Native terms must be homogeneous: all terms in one operator
        collection must carry the same Abelian charge, unless
        ``charge_sectors=True`` is requested. Neutral and nonzero charges are
        both supported with ``fermionic=True``. Odd-parity terms
        should be created with an explicit ``label=`` in
        :meth:`operator_term` so their dummy-mode phase metadata is retained.
        The Jordan--Wigner compatibility path remains neutral-only.
        The current implementation uses the MPO ordering as the fermionic
        ordering. Thus arbitrary two-site and non-contiguous terms are
        supported, but the PEPO's nontrivial operator bonds follow the
        selected snake-style chain; the added transverse lattice bonds have
        dimension one unless ``cyclic=True``.
        """
        if Lx is None or Ly is None:
            raise TypeError("to_pepo requires both Lx and Ly.")
        if hamiltonian is not None:
            if terms_or_edges is not None:
                raise TypeError(
                    "Pass either terms_or_edges or hamiltonian, not both."
                )
            if not isinstance(hamiltonian, SymHamiltonian):
                raise TypeError("hamiltonian must be a SymHamiltonian instance.")
            target = hamiltonian
        elif isinstance(terms_or_edges, SymHamiltonian):
            target = terms_or_edges
        else:
            if terms_or_edges is None:
                raise TypeError("to_pepo requires terms_or_edges or hamiltonian.")
            target = self.hamiltonian(
                terms_or_edges,
                to_backend=to_backend,
                **params,
            )
            params = {}

        if params:
            names = ", ".join(sorted(params))
            raise TypeError(
                "Model parameters cannot be supplied with an existing "
                f"SymHamiltonian: {names}."
            )
        if (
            fermionic
            and not charge_sectors
            and _is_single_site_identity_hamiltonian(
                target,
                sum(int(size) for size in self.physical_sectors.values()),
                self.zero_charge,
            )
        ):
            from .constructors import (  # pylint: disable=import-outside-toplevel
                _native_fermion_identity_pepo,
            )

            return _native_fermion_identity_pepo(
                self,
                Lx,
                Ly,
                cyclic=cyclic,
                cycle_bond_dim=cycle_bond_dim,
                mapper=mapper,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                dtype=dtype,
                to_backend=to_backend,
            )
        return target.to_pepo(
            Lx=Lx,
            Ly=Ly,
            mapper=mapper,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            cyclic=cyclic,
            cycle_bond_dim=cycle_bond_dim,
            dtype=dtype,
            fermionic=fermionic,
            to_backend=to_backend,
            charge_sectors=charge_sectors,
        )

    def local_terms(self, edges, *, layout="site", **params):
        """Return native local terms for site or qMERA energy workflows.

        The default ``layout="site"`` returns the existing
        ``{edge: SymmrayArray}`` dictionary. ``layout="qmera"`` treats
        ``edges`` as a :class:`QMeraGeometry` and returns mode-native
        :class:`LocalTerm` objects for qMERA. For the site layout, onsite
        Hubbard and chemical-potential terms are distributed across the
        incident edge terms using each site's coordination, so summing the
        edge dictionary counts every onsite contribution exactly once.
        """
        layout = str(layout).lower().replace("-", "_")
        if layout in {"site", "sites", "native"}:
            return self.hamiltonian(edges, **params).terms
        if layout in {"qmera", "qmera_modes", "modes"}:
            from ..optimizers.qmera import (  # pylint: disable=import-outside-toplevel
                qmera_symmray_fermi_hubbard_terms,
            )

            return qmera_symmray_fermi_hubbard_terms(
                edges,
                fermion=self,
                **params,
            )
        if layout in {"majorana", "qmera_majorana"}:
            from ..optimizers.qmera import (  # pylint: disable=import-outside-toplevel
                qmera_symmray_majorana_terms,
            )

            return qmera_symmray_majorana_terms(
                edges,
                fermion=self,
                **params,
            )
        raise ValueError(
            "layout must be 'site' for native site terms or 'qmera' for "
            "two-state qMERA mode terms, or 'majorana'."
        )

    def qmera_terms(self, geometry, **params):
        """Return the explicit two-state qMERA terms for ``geometry``."""
        return self.local_terms(geometry, layout="qmera", **params)

    def majorana_terms(self, geometry, **params):
        """Return parity-preserving Majorana terms for a qMERA geometry."""
        return self.local_terms(geometry, layout="majorana", **params)


class SpinfulFermion(Fermion):
    """Compatibility constructor that always selects the spinful local space."""

    def __init__(self, *args, spinful=True, **kwargs):
        if len(args) > 3:
            raise TypeError(
                "SpinfulFermion fixes spinful=True; pass at most symmetry, "
                "dtype, and to_backend positionally."
            )
        if not spinful:
            raise TypeError("SpinfulFermion always uses spinful=True.")
        super().__init__(*args, spinful=True, **kwargs)

SpinfulFermionHubbard = SpinfulFermion


class SymmFermions:
    """Namespace for direct Symmray-backed fermion helper factories."""

    @staticmethod
    def fermion(*args, **kwargs):
        """Build the unified :class:`Fermion` helper."""
        return Fermion(*args, **kwargs)

    @staticmethod
    def spinful(*args, **kwargs):
        """Build a spinful :class:`Fermion` helper.

        This namespace makes it possible to add other local fermion spaces
        later while keeping the direct ``SpinfulFermion(...)`` form concise.
        """
        return SpinfulFermion(*args, **kwargs)

    @staticmethod
    def spinless(*args, **kwargs):
        """Build a spinless :class:`Fermion` helper."""
        kwargs.setdefault("spinful", False)
        kwargs.setdefault("symmetry", "U1")
        return Fermion(*args, **kwargs)
