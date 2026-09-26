"""Shared symmetry, Hamiltonians, MPO construction, and compatible imports."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from importlib import import_module
from typing import TYPE_CHECKING
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

__all__ = [
    "Fermion",
    "FermionLatticeSetup",
    "SpinfulFermion",
    "fermion_density_param_gen",
    "fermion_hopping_param_gen",
    "fermion_interaction_param_gen",
    "SymGateStream",
    "SymHamiltonian",
    "SymMPS",
    "SymPEPS",
]
__all__ += [
    "default_physical_sectors",
    "draw_symmray_blocks",
    "draw_symmray_mps",
    "draw_symmray_mpo",
    "draw_symmray_peps",
    "fermi_hubbard_u1u1_gate_stream",
    "fermi_hubbard_u1u1_hopping_gate_stream",
    "fermi_hubbard_u1u1_interaction_gate_stream",
    "fermi_hubbard_u1u1_light_pulse_gate_stream",
    "fermi_hubbard_u1u1_jw_gate_stream",
    "fermi_hubbard_u1u1_jw_hopping_gate_stream",
    "fermi_hubbard_u1u1_jw_interaction_gate_stream",
    "sector_index_map",
    "site_charge_alternating",
    "site_charge_from_map",
    "site_charge_from_occupations",
    "site_charge_uniform",
    "symmray_block_summary",
    "symmray_mps_summary",
    "symmray_mpo_summary",
    "symmray_peps_summary",
    "symm_operator_from_dense",
]

# Historical attributes stay lazy so importing Hamiltonians does not load
# models, state wrappers, or plotting implementation. This also resolves old
# pickle globals to the owning classes and functions.
_LEGACY_MODULES = {}
for _name in (
    'SymMPS',
    'SymPEPS',
    '_SymState',
):
    _LEGACY_MODULES[_name] = '.symmetric_states'
for _name in (
    '_FERMIONIC_TN_METHODS_REFERENCE',
    '_add_charges',
    '_as_mpo_tensor_network',
    '_as_mps_tensor_network',
    '_as_peps_tensor_network',
    '_block_size',
    '_bond_endpoint_directions',
    '_charge_spin_label_lines',
    '_charge_summary_text',
    '_directions_are_complementary',
    '_draw_mapped_chain_diagnostics',
    '_draw_mapped_chain_grid',
    '_draw_symmray_mpo_mapped',
    '_draw_symmray_mps_mapped',
    '_edge_lookup',
    '_explicit_source_edges',
    '_fermionic_edge_record',
    '_fermionic_ordering_summary',
    '_flow_math',
    '_flow_text',
    '_format_charge',
    '_format_compact_mapping',
    '_format_half_integer',
    '_format_sector',
    '_format_shape',
    '_format_signed_half_integer',
    '_infer_fermionic',
    '_infer_symmetry',
    '_is_fermionic_array_data',
    '_is_mpo_like',
    '_is_mps_like_not_peps',
    '_lighten_rgba',
    '_mapped_bond_label',
    '_mapped_chain_limits',
    '_mapped_charge_spin_lines',
    '_mapped_contrast_text_color',
    '_mapped_label_offset',
    '_mapped_physical_label',
    '_mapped_site_color',
    '_mapped_tensor_label_lines',
    '_mapping_items',
    '_mod_charge',
    '_mps_site_tensor',
    '_mps_sites',
    '_node_charge_label_lines',
    '_opposite_direction',
    '_peps_display_bonds',
    '_peps_primary_bond_key',
    '_peps_relative_direction',
    '_peps_site_distance',
    '_peps_site_ind',
    '_peps_site_tag',
    '_peps_site_tensor',
    '_peps_sites',
    '_require_symmray_array',
    '_resolve_chain_mapper',
    '_resolve_mps_position',
    '_resolve_peps_center',
    '_resolve_q_total',
    '_resolve_total_charge',
    '_shape_size',
    '_shared_virtual_ind',
    '_shared_virtual_inds',
    '_source_edges',
    '_sum_charges',
    '_summarize_symmray_index',
    'draw_symmray_blocks',
    'draw_symmray_mpo',
    'draw_symmray_mps',
    'draw_symmray_peps',
    'symmray_block_summary',
    'symmray_mpo_summary',
    'symmray_mps_summary',
    'symmray_peps_summary',
):
    _LEGACY_MODULES[_name] = '.symmetric_diagnostics'
for _name in (
    'Fermion',
    'FermionLatticeSetup',
    'SpinfulFermion',
    'SpinfulFermionHubbard',
    '_fermion_backend_anchor',
    '_fermion_complex_like',
    '_fermion_complex_phase',
    '_fermion_diagonal_gate_param_gen',
    '_fermion_generic_local_modes',
    '_fermion_scalar_anchor',
    '_fermion_spinful_density_param_gen',
    '_fermion_terms_dense',
    '_fermion_terms_exponential_gate',
    '_spinful_hopping_gate',
    '_spinless_hopping_gate',
    'fermion_density_param_gen',
    'fermion_hopping_param_gen',
    'fermion_interaction_param_gen',
):
    _LEGACY_MODULES[_name] = '.symm_fermions'

if TYPE_CHECKING:
    from .symmetric_states import (  # noqa: F401 -- historical import aliases
        SymMPS,
        SymPEPS,
        _SymState,
    )
    from .symmetric_diagnostics import (  # noqa: F401 -- historical import aliases
        draw_symmray_blocks,
        draw_symmray_mpo,
        draw_symmray_mps,
        draw_symmray_peps,
        symmray_block_summary,
        symmray_mpo_summary,
        symmray_mps_summary,
        symmray_peps_summary,
    )
    from .symm_fermions import (  # noqa: F401 -- historical import aliases
        Fermion,
        FermionLatticeSetup,
        SpinfulFermion,
        SpinfulFermionHubbard,
        fermion_density_param_gen,
        fermion_hopping_param_gen,
        fermion_interaction_param_gen,
    )


def __getattr__(name):
    """Resolve historical imports and serialized globals to their owner."""
    module = _LEGACY_MODULES.get(name)
    if module is not None:
        value = getattr(import_module(module, __package__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List compatibility names without loading their implementations."""
    return sorted(set(globals()) | set(__all__) | set(_LEGACY_MODULES))


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
    # Symmetric states can be handed directly to Quimb boundary, PEPS, and
    # projector-compression routines, so install the narrow compatibility
    # hook as soon as the optional backend is actually in use. This keeps the
    # dense Quimb path untouched while covering callers that do not pass
    # through one of Pepsy's BP wrappers first.
    from ..bp._symmray import install_quimb_symmray_compat

    install_quimb_symmray_compat()
    _register_symmray_autoray_compat()
    return sr


def _to_dense(value):
    return value.to_dense() if hasattr(value, "to_dense") else value


def _as_python_bool(value):
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return bool(value)


def _to_host_numpy(value):
    value = _to_dense(value)
    return np.asarray(ar.to_numpy(value))


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
        a_dense = _to_dense(a)
        b_dense = _to_dense(b)
        try:
            b_dense = ar.do("array", b_dense, like=a_dense)
            return _as_python_bool(
                ar.do("allclose", a_dense, b_dense, rtol=rtol, atol=atol, **kwargs)
            )
        except Exception:
            return np.allclose(
                _to_host_numpy(a_dense),
                _to_host_numpy(b_dense),
                rtol=rtol,
                atol=atol,
                **kwargs,
            )

    def _pad(a, pad_width, mode="constant", **kwargs):
        if mode != "constant":
            raise NotImplementedError("Symmray autoray pad only supports constant mode.")
        constant_values = kwargs.get("constant_values", 0)
        if np.any(np.asarray(constant_values) != 0):
            raise NotImplementedError("Symmray autoray pad only supports zero padding.")
        if not _is_symmray_array(a):
            return np.pad(a, pad_width, mode=mode, **kwargs)

        pad_width = tuple((int(lo), int(hi)) for lo, hi in pad_width)
        if len(pad_width) != len(a.shape):
            raise ValueError("pad_width rank does not match Symmray array rank.")

        for axis, (lo, hi) in enumerate(pad_width):
            if lo == 0 and hi == 0:
                continue
            chargemap = getattr(a.indices[axis], "chargemap", None)
            if chargemap is None or len(chargemap) != 1:
                raise NotImplementedError(
                    "Symmray autoray pad currently supports only single-sector "
                    "padded axes."
                )

        blocks = {
            sector: ar.do("pad", block, pad_width, mode="constant")
            for sector, block in a.blocks.items()
        }
        return type(a).from_blocks(
            blocks,
            duals=a.duals,
            charge=getattr(a, "charge", None),
            symmetry=getattr(a, "symmetry", None),
            phases=getattr(a, "phases", None),
            label=getattr(a, "label", None),
            dummy_modes=getattr(a, "dummy_modes", None),
        )

    ar.register_function("symmray", "eye", _eye)
    ar.register_function("symmray", "allclose", _allclose)
    ar.register_function("symmray", "pad", _pad)
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
    "fermi_hubbard_u1u1": "fermi_hubbard_u1u1",
    "fermi-hubbard-u1u1": "fermi_hubbard_u1u1",
    "hubbard_u1u1": "fermi_hubbard_u1u1",
    "fh_u1u1": "fermi_hubbard_u1u1",
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
    "fermi_hubbard_u1u1": {"symmetry": "U1U1", "fermionic": True, "phys_dim": 4},
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


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _charge_particle_number(charge):
    """Total particle number carried by a tensor charge (sum of components)."""
    if charge is None:
        return None
    if isinstance(charge, tuple):
        return sum(int(x) for x in charge)
    try:
        return int(charge)
    except (TypeError, ValueError):
        return None


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
    index_maps=None,
    label=None,
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
    index_maps : sequence of mappings, optional
        Explicit ordered charge maps for the row and column indices. When
        omitted for a native fermionic one- or two-site operator, Symmray's
        canonical fermion basis ordering is used.
    """
    # Keep user supplied backend arrays intact. In particular, converting a
    # torch or jax value through ``np.asarray`` either errors for a value that
    # requires gradients or silently leaves the autodiff backend. Shape
    # validation only needs the public array protocol here.
    arr = array
    try:
        arr_shape = tuple(int(dim) for dim in ar.shape(arr))
    except (AttributeError, TypeError):
        arr_shape = tuple(int(dim) for dim in getattr(arr, "shape", ()))
    if not arr_shape:
        raise ValueError("array must be a rank-2 or rank-4 dense operator.")
    sectors = dict(sectors)
    phys_dim = sum(int(size) for size in sectors.values())
    if sites is None:
        if len(arr_shape) == 2:
            sites = 1
        elif len(arr_shape) == 4:
            sites = 2
        else:
            raise ValueError("sites must be supplied for dense operators not rank 2 or 4.")
    sites = int(sites)
    if sites < 1:
        raise ValueError("sites must be a positive integer.")
    if len(arr_shape) == 2 and sites > 1:
        arr = ar.do("reshape", arr, (phys_dim,) * sites * 2)
    expected_shape = (phys_dim,) * sites * 2
    if tuple(int(dim) for dim in ar.shape(arr)) != expected_shape:
        raise ValueError(f"Operator shape {ar.shape(arr)} does not match expected {expected_shape}.")

    if index_maps is None:
        index_map = sector_index_map(sectors)
        if fermionic and phys_dim in {2, 4}:
            import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

            if phys_dim == 2:
                charges = flo.get_spinless_charge_indexmap(str(symmetry))
            else:
                charges = flo.get_spinful_charge_indexmap(str(symmetry))
            if len(charges) == phys_dim:
                index_map = dict(enumerate(charges))
        index_maps = tuple(dict(index_map) for _ in range(2 * sites))
    else:
        index_maps = tuple(dict(index_map) for index_map in index_maps)
        if len(index_maps) != 2 * sites:
            raise ValueError(
                "index_maps must contain one map for each row and column index."
            )
    duals = (False,) * sites + (True,) * sites
    array_cls = _array_class_for_symmetry(symmetry, fermionic=fermionic)
    kwargs = {}
    if array_cls.__name__ in {"AbelianArray", "FermionicArray"}:
        kwargs["symmetry"] = symmetry
    if label is not None:
        kwargs["label"] = label
    return array_cls.from_dense(
        arr,
        index_maps=index_maps,
        duals=duals,
        charge=charge,
        **kwargs,
    )


def _random_orthogonal_or_unitary(rng, size, dtype):
    dtype = np.dtype(dtype)
    matrix = rng.standard_normal((size, size))
    if np.issubdtype(dtype, np.complexfloating):
        matrix = matrix + 1.0j * rng.standard_normal((size, size))
    q, r = np.linalg.qr(matrix)
    diag = np.diag(r)
    phase = np.ones_like(diag)
    nonzero = np.abs(diag) > 0.0
    phase[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
    return np.asarray(q * phase.conj(), dtype=dtype)


def _random_charge_preserving_two_site_dense(sectors, symmetry, rng, dtype):
    sectors = dict(sectors)
    charges = [
        charge
        for charge, size in sectors.items()
        for _ in range(int(size))
    ]
    phys_dim = len(charges)
    out = np.eye(phys_dim * phys_dim, dtype=np.dtype(dtype))
    by_total_charge = {}
    for dense_index, (left_charge, right_charge) in enumerate(
        product(charges, repeat=2)
    ):
        total = _charge_add(left_charge, right_charge, symmetry)
        by_total_charge.setdefault(total, []).append(dense_index)

    for positions in by_total_charge.values():
        block = _random_orthogonal_or_unitary(rng, len(positions), dtype)
        out[np.ix_(positions, positions)] = block
    return out


def _right_canonize_mps(mps):
    method = getattr(mps, "right_canonize", None)
    if callable(method):
        try:
            result = method(bra=None)
        except TypeError:
            result = method()
        if result is not None:
            return result
    return mps


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


_YOSHIDA4_A = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA4_B = -2.0 ** (1.0 / 3.0) * _YOSHIDA4_A
_YOSHIDA4_COEFFICIENTS = (_YOSHIDA4_A, _YOSHIDA4_B, _YOSHIDA4_A)


def _yoshida4_stream(second_order_factory, dt, *, imaginary=False, hamiltonian=None):
    """Compose three symmetric second-order streams into a fourth-order step.

    The middle coefficient is negative, as required by the Suzuki-Yoshida
    triple jump. This helper only combines already validated local gates; it
    does not alter the existing first- or second-order stream construction.
    """
    streams = tuple(
        second_order_factory(coefficient * dt)
        for coefficient in _YOSHIDA4_COEFFICIENTS
    )
    if hamiltonian is None:
        hamiltonian = getattr(streams[0], "hamiltonian", None)
    entries = tuple(entry for stream in streams for entry in stream)
    return SymGateStream(
        entries,
        hamiltonian=hamiltonian,
        dt=dt,
        imaginary=imaginary,
        order=4,
    )


def _sites_from_edges(edges, sites):
    if sites is not None:
        out = tuple(sites)
        if not out:
            raise ValueError("sites must not be empty.")
        return out

    out = []
    seen = set()
    for left, right in edges:
        for site in (left, right):
            if site not in seen:
                seen.add(site)
                out.append(site)
    return tuple(out)


def _edge_coloring_layers(edges):
    """Partition edges into deterministic vertex-disjoint layers."""
    layers = []
    occupied_sites = []
    for edge in _as_edges(edges):
        if edge[0] == edge[1]:
            raise ValueError(
                "A hopping edge must connect distinct sites; "
                f"got {edge!r}."
            )
        endpoints = frozenset(edge)
        for layer, occupied in zip(layers, occupied_sites):
            if endpoints.isdisjoint(occupied):
                layer.append(edge)
                occupied.update(endpoints)
                break
        else:
            layers.append([edge])
            occupied_sites.append(set(endpoints))
    return tuple(tuple(layer) for layer in layers)


def _edge_angle_parameter(value, left, right):
    """Return an oriented edge angle, negating reversed mapping lookups."""
    if callable(value):
        return value(left, right)
    if isinstance(value, Mapping):
        if (left, right) in value:
            return value[(left, right)]
        return -value[(right, left)]
    return value


def _fh_spinful_peierls_hopping_array(
    symmetry,
    *,
    t=1.0,
    peierls_angle=0.0,
    dtype="complex128",
):
    """Return a spinful Fermi-Hubbard hopping term with Peierls phases."""
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    au = flo.FermionicOperator("a↑")
    ad = flo.FermionicOperator("a↓")
    bu = flo.FermionicOperator("b↑")
    bd = flo.FermionicOperator("b↓")

    tu, td = _as_spin_pair(t, name="t")
    phase = np.exp(1j * peierls_angle)
    phase_conj = np.conjugate(phase)
    terms = (
        (-tu * phase, (au.dag, bu)),
        (-tu * phase_conj, (bu.dag, au)),
        (-td * phase, (ad.dag, bd)),
        (-td * phase_conj, (bd.dag, ad)),
    )
    basis_a = ((), (au.dag,), (ad.dag,), (ad.dag, au.dag))
    basis_b = ((), (bu.dag,), (bd.dag,), (bd.dag, bu.dag))
    bases = (basis_a, basis_b)
    dense = np.zeros((4, 4, 4, 4), dtype=np.dtype(dtype))
    # Symmray's dense helper currently initializes real zeros, so complex
    # Peierls coefficients need an explicitly complex accumulation buffer.
    for idx, val in flo.build_local_fermionic_elements(terms, bases).items():
        dense[idx] += val

    sectors = default_physical_sectors(symmetry, 4)
    return symm_operator_from_dense(
        dense,
        sectors,
        symmetry=symmetry,
        charge=_zero_like_charge(next(iter(sectors))),
        fermionic=True,
        sites=2,
    )


def _fh_u1u1_onsite_interaction_gate(
    site,
    dt,
    *,
    U=8.0,
    imaginary=False,
    dtype="complex128",
    to_backend=None,
):
    dtype = np.dtype(dtype)
    scale = -dt if imaginary else -1j * dt
    double = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    U_site = _node_parameter(U, site)
    gate_dense = np.diag(np.exp(scale * U_site * double)).astype(dtype, copy=False)
    gate = symm_operator_from_dense(
        gate_dense,
        default_physical_sectors(model="fermi_hubbard_u1u1"),
        symmetry="U1U1",
        charge=(0, 0),
        fermionic=True,
        sites=1,
    )
    return _apply_to_array_blocks(gate, to_backend)


def fermi_hubbard_u1u1_interaction_gate_stream(
    sites,
    dt,
    *,
    U=8.0,
    imaginary=False,
    dtype="complex128",
    to_backend=None,
):
    """Return onsite ``U n_up n_down`` gates for spinful ``U1U1`` Hubbard.

    The returned stream contains one charge-preserving fermionic one-site gate
    per supplied site. It can be mixed with two-site hopping streams and passed
    to :class:`pepsy.MpsOptimizer`, :func:`pepsy.gate`, or
    :func:`pepsy.gate_simple`.
    """
    sites = tuple(sites)
    if not sites:
        raise ValueError("sites must not be empty.")
    entries = [
        (
            _fh_u1u1_onsite_interaction_gate(
                site,
                dt,
                U=U,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            ),
            site,
        )
        for site in sites
    ]
    return SymGateStream(entries, dt=dt, imaginary=imaginary, order=1)


def fermi_hubbard_u1u1_hopping_gate_stream(
    edges,
    dt,
    *,
    t=1.0,
    peierls_angle=0.0,
    imaginary=False,
    dtype="complex128",
    to_backend=None,
):
    """Return spin-preserving ``U1U1`` Fermi-Hubbard hopping gates.

    ``peierls_angle`` is the oriented left-to-right bond angle ``A``: hopping
    from the first edge site to the second receives ``exp(+i A)`` and the
    reverse direction receives ``exp(-i A)``. It may be a scalar, edge mapping,
    or ``callable(left, right)``.
    """
    edges = _as_edges(edges)
    entries = []
    for left, right in edges:
        term = _fh_spinful_peierls_hopping_array(
            "U1U1",
            t=_edge_parameter(t, left, right),
            peierls_angle=_edge_angle_parameter(peierls_angle, left, right),
            dtype=dtype,
        )
        gate = _gate_from_term(term, dt, imaginary=imaginary)
        gate = _apply_to_array_blocks(gate, to_backend)
        entries.append((gate, (left, right)))
    return SymGateStream(entries, dt=dt, imaginary=imaginary, order=1)


def fermi_hubbard_u1u1_gate_stream(
    edges,
    dt,
    *,
    sites=None,
    t=1.0,
    U=8.0,
    peierls_angle=0.0,
    imaginary=False,
    order=2,
    dtype="complex128",
    to_backend=None,
):
    """Return a native fermionic ``U1U1`` Fermi-Hubbard Trotter stream.

    ``order=1`` returns a Lie step ``U_int(dt) U_hop(dt)``. ``order=2``
    returns the Strang step ``U_int(dt/2) U_hop(dt) U_int(dt/2)``. ``order=4``
    applies the Suzuki-Yoshida triple jump to the second-order stream. The
    onsite and hopping gates are exact fermionic Symmray arrays, so no
    spin/qubit mapping is introduced.
    """
    if order not in {1, 2, 4}:
        raise ValueError("order must be 1, 2, or 4.")
    edges = _as_edges(edges)
    sites = _sites_from_edges(edges, sites)
    if order == 4:
        return _yoshida4_stream(
            lambda sub_dt: fermi_hubbard_u1u1_gate_stream(
                edges,
                sub_dt,
                sites=sites,
                t=t,
                U=U,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                order=2,
                dtype=dtype,
                to_backend=to_backend,
            ),
            dt,
            imaginary=imaginary,
        )
    if order == 1:
        entries = (
            tuple(
                fermi_hubbard_u1u1_interaction_gate_stream(
                    sites,
                    dt,
                    U=U,
                    imaginary=imaginary,
                    dtype=dtype,
                    to_backend=to_backend,
                )
            )
            + tuple(
                fermi_hubbard_u1u1_hopping_gate_stream(
                    edges,
                    dt,
                    t=t,
                    peierls_angle=peierls_angle,
                    imaginary=imaginary,
                    dtype=dtype,
                    to_backend=to_backend,
                )
            )
        )
        return SymGateStream(entries, dt=dt, imaginary=imaginary, order=order)

    half = dt / 2
    layers = _edge_coloring_layers(edges)
    entries = (
        tuple(
            fermi_hubbard_u1u1_interaction_gate_stream(
                sites,
                half,
                U=U,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        + tuple(
            gate_entry
            for layer in layers
            for gate_entry in fermi_hubbard_u1u1_hopping_gate_stream(
                layer,
                half,
                t=t,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        + tuple(
            gate_entry
            for layer in reversed(layers)
            for gate_entry in fermi_hubbard_u1u1_hopping_gate_stream(
                tuple(reversed(layer)),
                half,
                t=t,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        + tuple(
            fermi_hubbard_u1u1_interaction_gate_stream(
                sites,
                half,
                U=U,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
    )
    return SymGateStream(entries, dt=dt, imaginary=imaginary, order=order)


def _require_jw_adjacent_edge(left, right):
    """Validate that a Jordan-Wigner hopping bond is nearest-neighbour.

    A bosonic Jordan-Wigner hop is only a two-site gate when the sites are
    adjacent in the chain; otherwise the parity string spans the intervening
    sites and cannot be written as a single two-site operator.
    """
    try:
        li, ri = int(left), int(right)
    except (TypeError, ValueError):
        raise ValueError(
            "Jordan-Wigner hopping gate streams require integer nearest-neighbour "
            f"chain sites; got edge ({left!r}, {right!r})."
        ) from None
    if abs(li - ri) != 1:
        raise ValueError(
            f"Jordan-Wigner hopping on non-adjacent bond ({left}, {right}) is not "
            "a two-site gate: the parity string spans the sites between them. "
            "Order the sites so hopping is nearest-neighbour, or use "
            "SymHamiltonian.to_mpo(model='fermi_hubbard_u1u1') for the long-range "
            "Jordan-Wigner path."
        )


def _fh_u1u1_jw_hopping_term(*, t=1.0, peierls_angle=0.0, dtype="complex128"):
    """Return the bosonic Jordan-Wigner two-site spinful FH hopping term.

    This is the nearest-neighbour hopping term the ``fermi_hubbard_u1u1``
    :meth:`SymHamiltonian.to_mpo` path places on an adjacent bond, written as a
    *bosonic* Symmray array. The on-site Jordan-Wigner parity string is absorbed
    into the lower-site endpoint (``create @ parity`` and ``parity @
    annihilate``), so no fermionic swap phases are introduced and the term acts
    on a bosonic (Jordan-Wigner) MPS. ``peierls_angle`` is the ascending-site
    bond angle ``A``: forward hopping receives ``exp(+i A)`` and its Hermitian
    conjugate ``exp(-i A)``.
    """
    ops = _fh_u1u1_jw_local_ops(dtype)
    out_dtype = np.dtype(dtype)
    parity = ops["parity"]
    t_u, t_d = _as_spin_pair(t, name="t")
    phase = np.exp(1j * float(peierls_angle))
    phase_conj = np.conjugate(phase)
    # Peierls phases are complex, so accumulate in a complex buffer and cast to
    # the requested dtype once (lossless when peierls_angle == 0).
    dense = np.zeros((16, 16), dtype=np.complex128)
    for t_sigma, create, annihilate in (
        (t_u, ops["create_u"], ops["annihilate_u"]),
        (t_d, ops["create_d"], ops["annihilate_d"]),
    ):
        if t_sigma == 0:
            continue
        # Site-major JW: the parity string endpoint lives on the lower site.
        forward = np.kron(create @ parity, annihilate)   # c_i^dag P_i (x) c_j
        backward = np.kron(parity @ annihilate, create)  # P_i c_i     (x) c_j^dag
        dense += (-t_sigma * phase) * forward
        dense += (-t_sigma * phase_conj) * backward
    return symm_operator_from_dense(
        dense.astype(out_dtype, copy=False),
        default_physical_sectors(model="fermi_hubbard_u1u1"),
        symmetry="U1U1",
        charge=(0, 0),
        fermionic=False,
        sites=2,
    )


def _fh_u1u1_jw_onsite_term(*, U=8.0, mu=0.0, dtype="complex128"):
    """Return the bosonic Jordan-Wigner one-site Fermi-Hubbard onsite term.

    ``U n_up n_down - mu_up n_up - mu_down n_down`` as a diagonal bosonic U1U1
    array, for measuring the onsite energy of a Jordan-Wigner state.
    """
    out_dtype = np.dtype(dtype)
    mu_u, mu_d = _as_spin_pair(mu, name="mu")
    number_u = np.array([0.0, 1.0, 0.0, 1.0])
    number_d = np.array([0.0, 0.0, 1.0, 1.0])
    double = np.array([0.0, 0.0, 0.0, 1.0])
    diag = U * double - mu_u * number_u - mu_d * number_d
    return symm_operator_from_dense(
        np.diag(diag).astype(out_dtype),
        default_physical_sectors(model="fermi_hubbard_u1u1"),
        symmetry="U1U1",
        charge=(0, 0),
        fermionic=False,
        sites=1,
    )


def _fh_u1u1_jw_onsite_interaction_gate(
    site,
    dt,
    *,
    U=8.0,
    mu=0.0,
    imaginary=False,
    dtype="complex128",
    to_backend=None,
):
    """Return a bosonic Jordan-Wigner onsite ``U n_up n_down - mu n`` gate."""
    dtype = np.dtype(dtype)
    scale = -dt if imaginary else -1j * dt
    U_site = _node_parameter(U, site)
    mu_u, mu_d = _as_spin_pair(_node_parameter(mu, site), name="mu")
    number_u = np.array([0.0, 1.0, 0.0, 1.0], dtype=dtype)
    number_d = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
    double = np.array([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    onsite = U_site * double - mu_u * number_u - mu_d * number_d
    gate_dense = np.diag(np.exp(scale * onsite)).astype(dtype, copy=False)
    gate = symm_operator_from_dense(
        gate_dense,
        default_physical_sectors(model="fermi_hubbard_u1u1"),
        symmetry="U1U1",
        charge=(0, 0),
        fermionic=False,
        sites=1,
    )
    return _apply_to_array_blocks(gate, to_backend)


def fermi_hubbard_u1u1_jw_hopping_gate_stream(
    edges,
    dt,
    *,
    t=1.0,
    peierls_angle=0.0,
    imaginary=False,
    dtype="complex128",
    to_backend=None,
):
    """Return bosonic Jordan-Wigner ``U1U1`` Fermi-Hubbard hopping gates.

    Bosonic (Jordan-Wigner spin-picture) counterpart of
    :func:`fermi_hubbard_u1u1_hopping_gate_stream`. The Jordan-Wigner parity
    string is written explicitly into the two-site operator, so these gates act
    on a bosonic Jordan-Wigner MPS -- the representation used by
    ``SymHamiltonian.to_mpo(model="fermi_hubbard_u1u1")`` and
    :class:`pepsy.SymDMRG2` -- without introducing fermionic swap phases.

    Only **nearest-neighbour** chain bonds are supported: a long-range
    Jordan-Wigner hop is not a two-site gate because its parity string spans the
    intervening sites. Order the sites so hopping is nearest-neighbour, or use
    ``SymHamiltonian.to_mpo(model="fermi_hubbard_u1u1")`` for the long-range
    Jordan-Wigner path. ``peierls_angle`` is the ascending-site bond angle and
    may be a scalar, edge mapping, or ``callable(left, right)``.
    """
    edges = _as_edges(edges)
    entries = []
    for left, right in edges:
        _require_jw_adjacent_edge(left, right)
        lo, hi = (left, right) if int(left) < int(right) else (right, left)
        term = _fh_u1u1_jw_hopping_term(
            t=_edge_parameter(t, lo, hi),
            peierls_angle=_edge_angle_parameter(peierls_angle, lo, hi),
            dtype=dtype,
        )
        gate = _gate_from_term(term, dt, imaginary=imaginary)
        gate = _apply_to_array_blocks(gate, to_backend)
        entries.append((gate, (lo, hi)))
    return SymGateStream(entries, dt=dt, imaginary=imaginary, order=1)


def fermi_hubbard_u1u1_jw_interaction_gate_stream(
    sites,
    dt,
    *,
    U=8.0,
    mu=0.0,
    imaginary=False,
    dtype="complex128",
    to_backend=None,
):
    """Return bosonic Jordan-Wigner onsite ``U n_up n_down`` gates.

    Bosonic (Jordan-Wigner spin-picture) counterpart of
    :func:`fermi_hubbard_u1u1_interaction_gate_stream`. The onsite term
    ``U n_up n_down - mu_up n_up - mu_down n_down`` is diagonal, so each gate is a
    charge-preserving bosonic one-site gate that mixes with the bosonic hopping
    stream. ``mu`` may be a scalar or a ``(mu_up, mu_down)`` pair.
    """
    sites = tuple(sites)
    if not sites:
        raise ValueError("sites must not be empty.")
    entries = [
        (
            _fh_u1u1_jw_onsite_interaction_gate(
                site,
                dt,
                U=U,
                mu=mu,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            ),
            site,
        )
        for site in sites
    ]
    return SymGateStream(entries, dt=dt, imaginary=imaginary, order=1)


def fermi_hubbard_u1u1_jw_gate_stream(
    edges,
    dt,
    *,
    sites=None,
    t=1.0,
    U=8.0,
    mu=0.0,
    peierls_angle=0.0,
    imaginary=False,
    order=2,
    dtype="complex128",
    to_backend=None,
):
    """Return a bosonic Jordan-Wigner ``U1U1`` Fermi-Hubbard Trotter stream.

    Bosonic (Jordan-Wigner spin-picture) counterpart of
    :func:`fermi_hubbard_u1u1_gate_stream`. ``order=1`` returns a Lie step
    ``U_int(dt) U_hop(dt)``; ``order=2`` returns the Strang step
    ``U_int(dt/2) U_hop(dt) U_int(dt/2)``. ``order=4`` applies the
    Suzuki-Yoshida triple jump to the second-order stream. All gates are
    bosonic Symmray arrays that act on a Jordan-Wigner MPS; hopping bonds must
    be nearest-neighbour.
    """
    if order not in {1, 2, 4}:
        raise ValueError("order must be 1, 2, or 4.")
    edges = _as_edges(edges)
    sites = _sites_from_edges(edges, sites)
    if order == 4:
        return _yoshida4_stream(
            lambda sub_dt: fermi_hubbard_u1u1_jw_gate_stream(
                edges,
                sub_dt,
                sites=sites,
                t=t,
                U=U,
                mu=mu,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                order=2,
                dtype=dtype,
                to_backend=to_backend,
            ),
            dt,
            imaginary=imaginary,
        )
    if order == 1:
        entries = (
            tuple(
                fermi_hubbard_u1u1_jw_interaction_gate_stream(
                    sites,
                    dt,
                    U=U,
                    mu=mu,
                    imaginary=imaginary,
                    dtype=dtype,
                    to_backend=to_backend,
                )
            )
            + tuple(
                fermi_hubbard_u1u1_jw_hopping_gate_stream(
                    edges,
                    dt,
                    t=t,
                    peierls_angle=peierls_angle,
                    imaginary=imaginary,
                    dtype=dtype,
                    to_backend=to_backend,
                )
            )
        )
        return SymGateStream(entries, dt=dt, imaginary=imaginary, order=order)

    half = dt / 2
    layers = _edge_coloring_layers(edges)
    entries = (
        tuple(
            fermi_hubbard_u1u1_jw_interaction_gate_stream(
                sites,
                half,
                U=U,
                mu=mu,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        + tuple(
            gate_entry
            for layer in layers
            for gate_entry in fermi_hubbard_u1u1_jw_hopping_gate_stream(
                layer,
                half,
                t=t,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        + tuple(
            gate_entry
            for layer in reversed(layers)
            for gate_entry in fermi_hubbard_u1u1_jw_hopping_gate_stream(
                tuple(reversed(layer)),
                half,
                t=t,
                peierls_angle=peierls_angle,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        + tuple(
            fermi_hubbard_u1u1_jw_interaction_gate_stream(
                sites,
                half,
                U=U,
                mu=mu,
                imaginary=imaginary,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
    )
    return SymGateStream(entries, dt=dt, imaginary=imaginary, order=order)


def _default_light_pulse_angle(time, omega):
    return np.pi * (1.0 - np.cos(omega * time)) / 2.0


def _pulse_angle_for_step(peierls_angles, step, time, omega):
    if peierls_angles is None:
        return _default_light_pulse_angle(time, omega)
    if callable(peierls_angles):
        return peierls_angles(step, time)
    if isinstance(peierls_angles, Mapping):
        return peierls_angles
    try:
        return peierls_angles[step]
    except TypeError:
        return peierls_angles


def fermi_hubbard_u1u1_light_pulse_gate_stream(
    edges,
    *,
    sites=None,
    t=1.0,
    U=8.0,
    omega=4 * np.pi / 3,
    tau=None,
    pulse_steps=2,
    relaxation_steps=0,
    peierls_angles=None,
    dtype="complex128",
    to_backend=None,
):
    """Return the paper-style real-time Hubbard light-pulse gate stream.

    The default pulse uses midpoint Peierls angles from
    ``A(s) = pi * (1 - cos(omega * s)) / 2`` with
    ``tau = pi / (pulse_steps * omega)``. For the trapped-ion paper settings,
    ``pulse_steps=2`` gives ``tau=0.375`` when ``omega=4*pi/3``. After the
    pulsed hopping layers, optional field-off relaxation steps are appended.

    Adjacent Strang half-interaction layers are merged, so the default two-step
    pulse has the structure ``U_int(tau/2), U_hop(A_0), U_int(tau),
    U_hop(A_1), U_int(tau/2)``.
    """
    if not isinstance(pulse_steps, Integral) or int(pulse_steps) < 1:
        raise ValueError("pulse_steps must be a positive integer.")
    if not isinstance(relaxation_steps, Integral) or int(relaxation_steps) < 0:
        raise ValueError("relaxation_steps must be a non-negative integer.")

    pulse_steps = int(pulse_steps)
    relaxation_steps = int(relaxation_steps)
    tau = np.pi / (pulse_steps * omega) if tau is None else tau
    edges = _as_edges(edges)
    sites = _sites_from_edges(edges, sites)

    hop_angles = [
        _pulse_angle_for_step(peierls_angles, step, (step + 0.5) * tau, omega)
        for step in range(pulse_steps)
    ]
    hop_angles.extend(0.0 for _ in range(relaxation_steps))

    entries = list(
        fermi_hubbard_u1u1_interaction_gate_stream(
            sites,
            tau / 2,
            U=U,
            imaginary=False,
            dtype=dtype,
            to_backend=to_backend,
        )
    )
    for step, angle in enumerate(hop_angles):
        entries.extend(
            fermi_hubbard_u1u1_hopping_gate_stream(
                edges,
                tau,
                t=t,
                peierls_angle=angle,
                imaginary=False,
                dtype=dtype,
                to_backend=to_backend,
            )
        )
        interaction_dt = tau / 2 if step == len(hop_angles) - 1 else tau
        entries.extend(
            fermi_hubbard_u1u1_interaction_gate_stream(
                sites,
                interaction_dt,
                U=U,
                imaginary=False,
                dtype=dtype,
                to_backend=to_backend,
            )
        )

    return SymGateStream(entries, dt=tau, imaginary=False, order=2)


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


def _is_lattice_coordinate(value):
    """Return whether ``value`` looks like a two-dimensional site label."""
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(part, Integral) for part in value)
    )


def _term_mapping_uses_coordinate_sites(terms):
    """Detect the explicit PEPS ``{edge: ..., site: ...}`` convention.

    A flat pair of integers is ambiguous on its own: it can be an MPS edge
    ``(i, j)`` or a PEPS site ``(x, y)``. The nested coordinate edge used by
    PEPS, ``((x0, y0), (x1, y1))``, resolves that ambiguity for the complete
    mapping.
    """
    return any(
        isinstance(where, (tuple, list))
        and len(where) == 2
        and all(_is_lattice_coordinate(site) for site in where)
        for where in terms
    )


def _as_term_where(where, *, coordinate_sites=False):
    """Normalize a local-term location to a one-or-more-site tuple."""
    if coordinate_sites and _is_lattice_coordinate(where):
        # Preserve the coordinate as one site rather than interpreting (x, y)
        # as an MPS edge. The public mapping retains the flat coordinate key
        # for PEPS consumers such as ``compute_local_expectation``.
        return (tuple(where),)
    if isinstance(where, (tuple, list)):
        out = tuple(where)
    else:
        out = (where,)
    if not out:
        raise ValueError(
            "Hamiltonian term locations must contain at least one site."
        )
    return out


def _normalize_term_mapping(terms):
    """Normalize ``{site_or_edge: operator}`` Hamiltonian mappings."""
    terms = dict(terms)
    if not terms:
        raise ValueError("At least one Hamiltonian term is required.")
    coordinate_sites = _term_mapping_uses_coordinate_sites(terms)
    normalized = {}
    seen_wheres = set()
    for where, term in terms.items():
        normalized_where = _as_term_where(
            where,
            coordinate_sites=coordinate_sites,
        )
        # Keep coordinate-site keys in their PEPS-native ``(x, y)`` form.
        # Other keys retain the historical normalized MPS representation.
        output_where = (
            tuple(where)
            if coordinate_sites and _is_lattice_coordinate(where)
            else normalized_where
        )
        if normalized_where in seen_wheres:
            raise ValueError(
                f"Duplicate Hamiltonian term location {normalized_where!r}."
            )
        seen_wheres.add(normalized_where)
        normalized[output_where] = term
    return normalized


def _normalize_mpo_coord(site):
    try:
        coord = tuple(site)
    except TypeError as exc:
        raise TypeError(
            "MPO lattice coordinates must be non-empty tuples/lists of integers."
        ) from exc
    if not coord:
        raise ValueError("MPO lattice coordinates cannot be empty.")
    if not all(isinstance(part, Integral) for part in coord):
        raise TypeError("MPO lattice coordinate components must be integers.")
    return tuple(int(part) for part in coord)


def _validate_mpo_mapping_indices(indices, *, name):
    indices = sorted(int(idx) for idx in indices)
    if indices != list(range(len(indices))):
        raise ValueError(f"{name} must use contiguous integer indices 0..L-1.")


def _normalize_idx2coo(idx2coo):
    out = {}
    for idx, coord in dict(idx2coo).items():
        if not isinstance(idx, Integral):
            raise TypeError("idx2coo keys must be integer chain indices.")
        idx = int(idx)
        if idx in out:
            raise ValueError("idx2coo contains duplicate chain indices.")
        out[idx] = _normalize_mpo_coord(coord)
    if not out:
        raise ValueError("idx2coo cannot be empty.")
    _validate_mpo_mapping_indices(out, name="idx2coo")
    if len(set(out.values())) != len(out):
        raise ValueError("idx2coo contains duplicate lattice coordinates.")
    return out


def _normalize_coo2idx(coo2idx):
    out = {}
    for coord, idx in dict(coo2idx).items():
        if not isinstance(idx, Integral):
            raise TypeError("coo2idx values must be integer chain indices.")
        coord = _normalize_mpo_coord(coord)
        if coord in out:
            raise ValueError("coo2idx contains duplicate lattice coordinates.")
        out[coord] = int(idx)
    if not out:
        raise ValueError("coo2idx cannot be empty.")
    _validate_mpo_mapping_indices(out.values(), name="coo2idx")
    if len(set(out.values())) != len(out):
        raise ValueError("coo2idx contains duplicate chain indices.")
    return out


def _resolve_mpo_mapping(*, mapper=None, idx2coo=None, coo2idx=None):
    if mapper is not None:
        if idx2coo is not None or coo2idx is not None:
            raise TypeError("Pass either mapper or idx2coo/coo2idx, not both.")
        from .maps import OneDMap  # pylint: disable=import-outside-toplevel

        if not isinstance(mapper, OneDMap):
            raise TypeError("mapper must be a pepsy.tensors.OneDMap instance.")
        idx2coo, coo2idx = mapper.build()

    if idx2coo is None and coo2idx is None:
        return None, None, None

    idx2coo_norm = None if idx2coo is None else _normalize_idx2coo(idx2coo)
    coo2idx_norm = None if coo2idx is None else _normalize_coo2idx(coo2idx)

    if idx2coo_norm is None:
        idx2coo_norm = {idx: coord for coord, idx in coo2idx_norm.items()}
    elif coo2idx_norm is None:
        coo2idx_norm = {coord: idx for idx, coord in idx2coo_norm.items()}
    else:
        expected = {coord: idx for idx, coord in idx2coo_norm.items()}
        if coo2idx_norm != expected:
            raise ValueError("idx2coo and coo2idx describe different mappings.")

    return idx2coo_norm, coo2idx_norm, len(idx2coo_norm)


def _map_edges_to_mpo_indices(edges, coo2idx):
    mapped_edges = []
    for edge in edges:
        mapped_edge = []
        for site in edge:
            if isinstance(site, Integral):
                mapped_edge.append(int(site))
                continue
            if coo2idx is None:
                raise ValueError(
                    "SymHamiltonian.to_mpo requires mapper=OneDMap(...) or "
                    "coo2idx=... when Hamiltonian edges use lattice coordinates."
                )
            coord = _normalize_mpo_coord(site)
            try:
                mapped_edge.append(coo2idx[coord])
            except KeyError as exc:
                raise ValueError(
                    f"Hamiltonian site {coord!r} is not present in the MPO mapping."
                ) from exc
        mapped_edges.append(tuple(mapped_edge))
    return tuple(mapped_edges)


def _map_site_to_mpo_index(site, coo2idx):
    """Map one integer or lattice-coordinate site to an MPO index."""
    if isinstance(site, Integral):
        return int(site)
    if coo2idx is None:
        raise ValueError(
            "SymHamiltonian.to_mpo requires mapper=OneDMap(...) or "
            "coo2idx=... when Hamiltonian terms use lattice coordinates."
        )
    coord = _normalize_mpo_coord(site)
    try:
        return int(coo2idx[coord])
    except KeyError as exc:
        raise ValueError(
            f"Hamiltonian site {coord!r} is not present in the MPO mapping."
        ) from exc


def _format_site_ind(site, site_ind_id):
    if isinstance(site, tuple):
        return site_ind_id.format(*site)
    return site_ind_id.format(site)


def _sites_from_gate_where(where, site_ind_id):
    """Normalize one-/two-site gate locations for local index formatting."""
    if site_ind_id == "k{}" and isinstance(where, Integral):
        return (int(where),)
    if (
        site_ind_id == "k{},{}"
        and isinstance(where, tuple)
        and len(where) == 2
        and all(isinstance(x, Integral) for x in where)
    ):
        return (tuple(int(x) for x in where),)
    if isinstance(where, (tuple, list)):
        if (
            site_ind_id == "k{},{}"
            and len(where) == 1
            and isinstance(where[0], (tuple, list))
            and len(where[0]) == 2
            and all(isinstance(x, Integral) for x in where[0])
        ):
            return (tuple(int(x) for x in where[0]),)
        if (
            site_ind_id == "k{}"
            and len(where) == 1
            and isinstance(where[0], Integral)
        ):
            return (int(where[0]),)
        return tuple(where)
    return (where,)


def _as_scalar(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        shape = tuple(shape)
        if shape != ():
            return value
        if _is_symmray_array(value):
            # NumPy can wrap a Symmray scalar in an object array, whose item()
            # simply returns the original wrapper. Use Symmray's phase-aware
            # scalar API instead (fixed upstream in the required 0.4 release).
            return value.item()
        try:
            return ar.to_numpy(value).item()
        except Exception:
            # Preserve the small duck-typed scalar contract for backend
            # wrappers that expose ``item`` but are not Autoray-registered.
            item = getattr(value, "item", None)
            if callable(item):
                return item()
            raise
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _validate_backend_mapper(to_backend):
    if to_backend is not None and not callable(to_backend):
        raise TypeError("to_backend must be callable or None.")


def _copy_array_like(value):
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    return value


def _apply_to_array_blocks(value, to_backend):
    """Apply a backend mapper while preserving Symmray block structure."""
    if to_backend is None:
        return value
    _validate_backend_mapper(to_backend)
    if _is_symmray_array(value):
        value.apply_to_arrays(to_backend)
        return value
    return to_backend(value)


def _operator_content_fingerprint(operator):
    """Return a stable content digest for a native Symmray operator.

    ``Fermion.operator_gate`` caches gate exponentials, so the cache key must
    depend on the operator's *contents* rather than its Python ``id``.  A
    freshly built operator can be garbage collected and have its memory
    address reused, which would otherwise let the cache return a stale gate
    for an unrelated operator.  Returns ``None`` when a stable fingerprint
    cannot be formed -- an unrecognised operator type or blocks that carry an
    autodiff graph -- signalling that the resulting gate must not be cached.
    """
    blocks = getattr(operator, "blocks", None)
    if blocks is None:
        return None
    hasher = hashlib.blake2b(digest_size=16)
    header = (
        getattr(operator, "symmetry", None),
        getattr(operator, "charge", None),
        tuple(getattr(operator, "duals", ()) or ()),
        tuple(getattr(operator, "shape", ()) or ()),
    )
    hasher.update(repr(header).encode())
    for sector in sorted(blocks, key=repr):
        block = blocks[sector]
        if getattr(block, "requires_grad", False):
            return None
        try:
            array = np.ascontiguousarray(ar.to_numpy(block))
        except Exception:  # pragma: no cover - defensive backend guard
            return None
        hasher.update(repr((sector, array.shape, array.dtype.str)).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _apply_to_tensor_network_arrays(tn, to_backend):
    if to_backend is None:
        return tn
    _validate_backend_mapper(to_backend)
    tn.apply_to_arrays(lambda array: _apply_to_array_blocks(array, to_backend))
    return tn


def _apply_to_hamiltonian_terms(terms, to_backend):
    if to_backend is None:
        return dict(terms)
    _validate_backend_mapper(to_backend)
    return {
        edge: _apply_to_array_blocks(term, to_backend)
        for edge, term in dict(terms).items()
    }


def _fh_spinful_density_edge_array(symmetry, *, V, like="numpy", flat=False):
    """Return a spinful edge density-density term in Symmray form."""
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    au = flo.FermionicOperator("a↑")
    ad = flo.FermionicOperator("a↓")
    bu = flo.FermionicOperator("b↑")
    bd = flo.FermionicOperator("b↓")

    terms = (
        (V, (au.dag, au, bu.dag, bu)),
        (V, (au.dag, au, bd.dag, bd)),
        (V, (ad.dag, ad, bu.dag, bu)),
        (V, (ad.dag, ad, bd.dag, bd)),
    )
    basis_a = ((), (au.dag,), (ad.dag,), (ad.dag, au.dag))
    basis_b = ((), (bu.dag,), (bd.dag,), (bd.dag, bu.dag))
    indexmap = flo.get_spinful_charge_indexmap(symmetry)

    return flo.build_local_fermionic_array(
        terms,
        (basis_a, basis_b),
        symmetry,
        index_maps=[indexmap, indexmap],
        like=like,
        flat=flat,
    )


def _fh_spinful_local_array_with_v(
    symmetry,
    *,
    t=1.0,
    U=8.0,
    mu=0.0,
    V=0.0,
    coordinations=(1, 1),
    like="numpy",
    flat=False,
):
    """Return a Symmray spinful Fermi-Hubbard local array with optional V."""
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    base = flo.fermi_hubbard_local_array(
        symmetry,
        t=t,
        U=U,
        mu=mu,
        coordinations=coordinations,
        like=like,
        flat=flat,
    )
    if V == 0:
        return base
    return base + _fh_spinful_density_edge_array(
        symmetry,
        V=V,
        like=like,
        flat=flat,
    )


def _ham_fermi_hubbard_spinful_from_edges_with_v(
    symmetry,
    edges,
    *,
    t=1.0,
    U=8.0,
    mu=0.0,
    V=0.0,
    like="numpy",
    flat=False,
):
    """Return spinful Fermi-Hubbard terms, including edge density V."""
    edges = _as_edges(edges)
    coordinations = {}
    for left, right in edges:
        coordinations[left] = coordinations.setdefault(left, 0) + 1
        coordinations[right] = coordinations.setdefault(right, 0) + 1

    return {
        (left, right): _fh_spinful_local_array_with_v(
            symmetry,
            t=_edge_parameter(t, left, right),
            U=(_node_parameter(U, left), _node_parameter(U, right)),
            mu=(_node_parameter(mu, left), _node_parameter(mu, right)),
            V=_edge_parameter(V, left, right),
            coordinations=(coordinations[left], coordinations[right]),
            like=like,
            flat=flat,
        )
        for left, right in edges
    }


def _hamiltonian_from_edges(model, symmetry, edges, *, flat=False, **params):
    sr = _require_symmray()
    model = _normalize_model(model)
    if model == "tfim":
        return sr.ham_tfim_from_edges(symmetry, edges, flat=flat, **params)
    if model == "heisenberg":
        return sr.ham_heisenberg_from_edges(symmetry, edges, flat=flat, **params)
    if model in {"fermi_hubbard", "fermi_hubbard_u1u1"}:
        if "V" in params:
            return _ham_fermi_hubbard_spinful_from_edges_with_v(
                symmetry,
                edges,
                flat=flat,
                **params,
            )
        return sr.ham_fermi_hubbard_from_edges(symmetry, edges, flat=flat, **params)
    if model == "fermi_hubbard_spinless":
        return sr.ham_fermi_hubbard_spinless_from_edges(symmetry, edges, flat=flat, **params)
    raise AssertionError(f"Unhandled model {model!r}.")


def _gate_from_term(term, dt, *, imaginary=False):
    """Exponentiate a one- or two-site local Hamiltonian term."""
    shape = tuple(int(d) for d in term.shape)
    if len(shape) == 2 and shape[0] == shape[1]:
        matrix_shape = shape
    elif len(shape) == 4 and shape[0] == shape[2] and shape[1] == shape[3]:
        matrix_shape = (shape[0] * shape[1], shape[2] * shape[3])
    else:
        raise ValueError(
            "Hamiltonian terms must have one-site shape (d, d) or two-site "
            "shape (da, db, da, db)."
        )
    scale = -dt if imaginary else -1j * dt
    return ar.do("linalg.expm", scale * term.reshape(matrix_shape)).reshape(shape)


def _zero_like_charge(charge):
    if isinstance(charge, tuple):
        return tuple(0 for _ in charge)
    return 0


def _neg_charge(charge):
    if isinstance(charge, tuple):
        return tuple(-int(x) for x in charge)
    return -int(charge)


def _normalize_group_charge(charge, symmetry):
    symmetry = str(symmetry)
    if isinstance(charge, tuple):
        charge = tuple(int(x) for x in charge)
        if symmetry == "Z2Z2":
            return tuple(x % 2 for x in charge)
        return charge
    charge = int(charge)
    if symmetry == "Z2":
        return charge % 2
    return charge


def _charge_add(a, b, symmetry):
    if isinstance(a, tuple) or isinstance(b, tuple):
        a = _as_tuple(a)
        b = _as_tuple(b)
        if len(a) != len(b):
            raise ValueError("Cannot add charges with different ranks.")
        charge = tuple(int(x) + int(y) for x, y in zip(a, b))
    else:
        charge = int(a) + int(b)
    return _normalize_group_charge(charge, symmetry)


def _charge_sub(a, b, symmetry):
    return _charge_add(a, _charge_neg(b, symmetry), symmetry)


def _charge_neg(charge, symmetry):
    if isinstance(charge, tuple):
        charge = tuple(-int(x) for x in charge)
    else:
        charge = -int(charge)
    return _normalize_group_charge(charge, symmetry)


def _charge_sort_key(charge):
    return repr(charge)


def _is_fermionic_symmray_array(value):
    return "FermionicArray" in type(value).__name__


def _dtype_from_hamiltonian_terms(terms, default="complex128"):
    dtypes = []
    for term in dict(terms).values():
        blocks = getattr(term, "blocks", None)
        values = blocks.values() if blocks else (term,)
        for value in values:
            dtype = getattr(value, "dtype", None)
            if dtype is None:
                continue
            try:
                dtypes.append(np.dtype(dtype))
            except TypeError:
                continue
    return np.result_type(*dtypes) if dtypes else np.dtype(default)


def _as_spin_pair(value, *, name):
    try:
        left, right = value
    except TypeError:
        return value, value
    except ValueError as exc:
        raise ValueError(f"{name} must be a scalar or a length-2 sequence.") from exc
    return left, right


def _edge_parameter(value, left, right):
    if callable(value):
        return value(left, right)
    if isinstance(value, Mapping):
        try:
            return value[(left, right)]
        except KeyError:
            return value[(right, left)]
    return value


def _node_parameter(value, site):
    if callable(value):
        return value(site)
    if isinstance(value, Mapping):
        return value[site]
    return value


def _coupling_is_active(value):
    """Whether a scalar, site map, or edge map contributes to a model."""
    if callable(value) or isinstance(value, Mapping):
        return True
    return value != 0


def _dense_numpy(value, *, dtype=None):
    value = _to_dense(value)
    return np.asarray(ar.to_numpy(value), dtype=dtype)


def _is_single_site_identity_hamiltonian(target, local_dim, zero_charge):
    """Return whether ``target`` is exactly one full local identity term."""
    if len(target.terms) != 1:
        return False
    term = next(iter(target.terms.values()))
    if getattr(term, "charge", None) != zero_charge:
        return False
    dense = _dense_numpy(term)
    return dense.shape == (local_dim, local_dim) and np.array_equal(
        dense,
        np.eye(local_dim, dtype=dense.dtype),
    )


def _expanded_index_charges(index):
    chargemap = getattr(index, "chargemap", None)
    if chargemap is None:
        raise TypeError("SymHamiltonian.to_mpo requires Symmray index charge maps.")
    out = []
    for charge, size in chargemap.items():
        out.extend([charge] * int(size))
    return out


def _term_dense_and_phys_maps(term, *, dtype, reverse=False):
    dense = _dense_numpy(term, dtype=dtype)
    if dense.ndim != 4 or dense.shape[0] != dense.shape[2] or dense.shape[1] != dense.shape[3]:
        raise ValueError(
            "SymHamiltonian.to_mpo requires two-site terms with shape "
            "(da, db, da, db)."
        )

    indices = getattr(term, "indices", None)
    if indices is None or len(indices) != 4:
        raise TypeError("SymHamiltonian.to_mpo requires Symmray rank-4 terms.")

    left_out = _expanded_index_charges(indices[0])
    right_out = _expanded_index_charges(indices[1])
    left_in = _expanded_index_charges(indices[2])
    right_in = _expanded_index_charges(indices[3])

    if reverse:
        dense = dense.transpose(1, 0, 3, 2)
        left_out, right_out = right_out, left_out
        left_in, right_in = right_in, left_in

    if left_out != left_in or right_out != right_in:
        raise ValueError("MPO terms must use matching upper/lower physical charge maps.")

    return dense, left_out, right_out


def _term_dense_and_phys_map(term, *, dtype):
    """Return dense data and the physical charge map for a one-site term."""
    dense = _dense_numpy(term, dtype=dtype)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(
            "SymHamiltonian.to_mpo requires one-site terms with shape (d, d)."
        )

    indices = getattr(term, "indices", None)
    if indices is None or len(indices) != 2:
        raise TypeError("SymHamiltonian.to_mpo requires Symmray rank-2 terms.")
    output = _expanded_index_charges(indices[0])
    input_ = _expanded_index_charges(indices[1])
    if output != input_:
        raise ValueError("One-site terms must use matching upper/lower physical charge maps.")
    return dense, output


def _svd_rank_cutoff(singular_values, shape):
    if singular_values.size == 0:
        return 0.0
    dtype = singular_values.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    eps = np.finfo(dtype).eps
    return eps * max(shape) * float(singular_values[0])


def _decompose_neutral_two_site_term(
    term,
    *,
    symmetry,
    dtype,
    reverse=False,
    fermionic=False,
):
    """Split a neutral two-site operator into charged one-site channels."""
    dense, left_phys, right_phys = _term_dense_and_phys_maps(
        term,
        dtype=dtype,
        reverse=reverse,
    )
    if fermionic:
        # The raw native tensor has its ket axes in fermionic tensor-product
        # order. Converting those two axes to an ordinary site-major MPO
        # inserts the crossing phase (-1) whenever both endpoint ket states
        # have odd particle parity. The Jordan-Wigner string between the
        # endpoints is added separately by the MPO channel below.
        left_odd = np.array(
            [
                _charge_particle_number(charge) % 2 != 0
                for charge in left_phys
            ],
            dtype=bool,
        )
        right_odd = np.array(
            [
                _charge_particle_number(charge) % 2 != 0
                for charge in right_phys
            ],
            dtype=bool,
        )
        crossing = np.ones((len(left_phys), len(right_phys)), dtype=dtype)
        crossing[np.ix_(left_odd, right_odd)] = -1.0
        dense = dense * crossing[None, None, :, :]
    dl, dr, _, _ = dense.shape
    matrix = dense.transpose(0, 2, 1, 3).reshape(dl * dl, dr * dr)

    left_entries = []
    for out_i, out_charge in enumerate(left_phys):
        for in_i, in_charge in enumerate(left_phys):
            charge = _charge_sub(out_charge, in_charge, symmetry)
            left_entries.append((out_i, in_i, charge))

    right_entries = []
    for out_i, out_charge in enumerate(right_phys):
        for in_i, in_charge in enumerate(right_phys):
            charge = _charge_sub(out_charge, in_charge, symmetry)
            right_entries.append((out_i, in_i, charge))

    left_by_charge = {}
    for pos, (_, _, charge) in enumerate(left_entries):
        left_by_charge.setdefault(charge, []).append(pos)
    right_by_charge = {}
    for pos, (_, _, charge) in enumerate(right_entries):
        right_by_charge.setdefault(charge, []).append(pos)

    channels = []
    for left_charge in sorted(left_by_charge, key=_charge_sort_key):
        right_charge = _charge_neg(left_charge, symmetry)
        rows = left_by_charge[left_charge]
        cols = right_by_charge.get(right_charge, ())
        if not cols:
            continue
        block = matrix[np.ix_(rows, cols)]
        if not np.any(block):
            continue

        u, s, vh = np.linalg.svd(block, full_matrices=False)
        rank_cutoff = _svd_rank_cutoff(s, block.shape)
        for rank, singular_value in enumerate(s):
            if float(singular_value) <= rank_cutoff:
                continue
            root = np.sqrt(singular_value)
            left_vec = u[:, rank] * root
            right_vec = root * vh[rank, :]

            left_op = np.zeros((dl, dl), dtype=dtype)
            for entry_pos, value in zip(rows, left_vec):
                out_i, in_i, _ = left_entries[entry_pos]
                left_op[out_i, in_i] = value

            right_op = np.zeros((dr, dr), dtype=dtype)
            for entry_pos, value in zip(cols, right_vec):
                out_i, in_i, _ = right_entries[entry_pos]
                right_op[out_i, in_i] = value

            channels.append((left_charge, left_op, right_op))

    return channels, left_phys, right_phys


def _fermion_parity_operator(phys_map, dtype):
    diag = []
    for charge in phys_map:
        particle_number = _charge_particle_number(charge)
        if particle_number is None:
            raise ValueError("Cannot infer fermionic parity from physical charges.")
        diag.append(-1.0 if particle_number % 2 else 1.0)
    return np.diag(diag).astype(dtype)


def _charged_op_needs_fermion_string(charge):
    particle_number = _charge_particle_number(charge)
    return particle_number is not None and particle_number % 2 != 0


def _fh_u1u1_dense_local_ops(dtype):
    """Return dense one-site spinful FH operators in Symmray's basis order."""
    return _fh_spinful_dense_local_ops("U1U1", dtype)


def _fh_spinful_jw_local_ops(symmetry, dtype):
    """Return ITensor/JW one-site spinful FH operator matrices.

    Symmray's fermionic local dense helper returns raw tensor data whose signs
    are completed by fermionic contraction. A two-site bosonic MPO needs the
    explicit one-site Jordan-Wigner matrices instead.
    """
    ops = _fh_spinful_dense_local_ops(symmetry, dtype)
    dtype = np.dtype(dtype)

    annihilate_u = np.zeros((4, 4), dtype=dtype)
    annihilate_u[0, 1] = 1
    annihilate_u[2, 3] = 1

    annihilate_d = np.zeros((4, 4), dtype=dtype)
    annihilate_d[0, 2] = 1
    annihilate_d[1, 3] = -1

    ops.update(
        {
            "annihilate_u": annihilate_u,
            "create_u": annihilate_u.conj().T,
            "number_u": np.diag([0, 1, 0, 1]).astype(dtype),
            "annihilate_d": annihilate_d,
            "create_d": annihilate_d.conj().T,
            "number_d": np.diag([0, 0, 1, 1]).astype(dtype),
            "double": np.diag([0, 0, 0, 1]).astype(dtype),
        }
    )
    return ops


def _fh_u1u1_jw_local_ops(dtype):
    """Return U1U1 ITensor/JW one-site spinful FH operator matrices."""
    return _fh_spinful_jw_local_ops("U1U1", dtype)


def _fh_spinful_dense_local_ops(symmetry, dtype):
    """Return dense one-site spinful FH operators in Symmray's basis order."""
    sr = _require_symmray()
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    au = sr.FermionicOperator("a↑")
    ad = sr.FermionicOperator("a↓")
    basis = ((), (au.dag,), (ad.dag,), (ad.dag, au.dag))

    def dense(term_ops, coeff=1.0):
        arr = flo.build_local_fermionic_dense(
            [(coeff, tuple(term_ops))],
            [basis],
        )
        return np.asarray(arr, dtype=dtype)

    return {
        "index_map": flo.get_spinful_charge_indexmap(symmetry),
        "identity": np.eye(4, dtype=dtype),
        "parity": np.diag([1.0, -1.0, -1.0, 1.0]).astype(dtype),
        "annihilate_u": dense((au,)),
        "create_u": dense((au.dag,)),
        "number_u": dense((au.dag, au)),
        "annihilate_d": dense((ad,)),
        "create_d": dense((ad.dag,)),
        "number_d": dense((ad.dag, ad)),
        "double": dense((au.dag, au, ad.dag, ad)),
    }


def _fh_spinless_dense_local_ops(symmetry, dtype):
    """Return dense one-site spinless FH operators in Symmray's basis order."""
    sr = _require_symmray()
    import symmray.fermionic_local_operators as flo  # pylint: disable=import-outside-toplevel

    a = sr.FermionicOperator("a")
    basis = ((), (a.dag,))

    def dense(term_ops, coeff=1.0):
        arr = flo.build_local_fermionic_dense(
            [(coeff, tuple(term_ops))],
            [basis],
        )
        return np.asarray(arr, dtype=dtype)

    return {
        "index_map": flo.get_spinless_charge_indexmap(symmetry),
        "identity": np.eye(2, dtype=dtype),
        "parity": np.diag([1.0, -1.0]).astype(dtype),
        "annihilate": dense((a,)),
        "create": dense((a.dag,)),
        "number": dense((a.dag, a)),
    }


def _add_local_transition(tensor, site, L, left_pos, right_pos, op):
    if L == 1:
        tensor[:, :] += op
    elif site == 0:
        tensor[right_pos, :, :] += op
    elif site == L - 1:
        tensor[left_pos, :, :] += op
    else:
        tensor[left_pos, right_pos, :, :] += op


def _assemble_symmray_mpo(
    *,
    L,
    channels,
    transitions,
    phys_map,
    symmetry,
    zero,
    dtype,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    fermionic=False,
    operator_charge=None,
):
    operator_charge = (
        zero
        if operator_charge is None
        else _normalize_group_charge(operator_charge, symmetry)
    )
    channel_pos = [
        {channel_id: pos for pos, (channel_id, _) in enumerate(cut_channels)}
        for cut_channels in channels
    ]

    arrays = []
    _require_symmray()
    from symmray import utils as sr_utils  # pylint: disable=import-outside-toplevel

    identity = np.eye(len(phys_map), dtype=dtype)
    for site in range(L):
        if L == 1:
            data = np.zeros((len(phys_map), len(phys_map)), dtype=dtype)
            index_maps = [phys_map, phys_map]
            duals = [False, True]
        elif site == 0:
            right_map = [charge for _, charge in channels[site]]
            data = np.zeros((len(right_map), len(phys_map), len(phys_map)), dtype=dtype)
            index_maps = [right_map, phys_map, phys_map]
            duals = [False, False, True]
            _add_local_transition(data, site, L, None, channel_pos[site][("start",)], identity)
        elif site == L - 1:
            left_map = [charge for _, charge in channels[site - 1]]
            data = np.zeros((len(left_map), len(phys_map), len(phys_map)), dtype=dtype)
            index_maps = [left_map, phys_map, phys_map]
            duals = [True, False, True]
            _add_local_transition(data, site, L, channel_pos[site - 1][("done",)], None, identity)
        else:
            left_map = [charge for _, charge in channels[site - 1]]
            right_map = [charge for _, charge in channels[site]]
            data = np.zeros(
                (len(left_map), len(right_map), len(phys_map), len(phys_map)),
                dtype=dtype,
            )
            index_maps = [left_map, right_map, phys_map, phys_map]
            duals = [True, False, False, True]
            _add_local_transition(
                data,
                site,
                L,
                channel_pos[site - 1][("start",)],
                channel_pos[site][("start",)],
                identity,
            )
            _add_local_transition(
                data,
                site,
                L,
                channel_pos[site - 1][("done",)],
                channel_pos[site][("done",)],
                identity,
            )

        for left_id, right_id, op in transitions[site]:
            left_pos = None if site == 0 else channel_pos[site - 1][left_id]
            right_pos = None if site == L - 1 else channel_pos[site][right_id]
            _add_local_transition(data, site, L, left_pos, right_pos, op)

        arrays.append(
            sr_utils.from_dense(
                data,
                symmetry=symmetry,
                index_maps=index_maps,
                duals=duals,
                fermionic=bool(fermionic),
                charge=operator_charge if site == L - 1 else zero,
                label=(site if fermionic and site == L - 1 and
                       _charged_op_needs_fermion_string(operator_charge)
                       else None),
            )
        )

    mpo = qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
    )
    raw_bond = mpo.max_bond()
    raw_max_bond = 1 if raw_bond is None else int(raw_bond)
    did_compress = bool(compress and L > 1)
    if compress and L > 1:
        compress_opts = {"cutoff": cutoff}
        if max_bond is not None:
            compress_opts["max_bond"] = int(max_bond)
        mpo.compress(**compress_opts)
    if to_backend is not None:
        # Cast after compression so the SVD-based bond truncation runs in the
        # stable build precision (e.g. complex128). Converting first and then
        # compressing runs the SVD in the target precision, which for a
        # near-singular Hamiltonian MPO in complex64 can hit non-finite values.
        _apply_to_tensor_network_arrays(mpo, to_backend)

    requested_max_bond = None if max_bond is None else int(max_bond)
    final_bond = mpo.max_bond()
    final_max_bond = 1 if final_bond is None else int(final_bond)
    report = {
        "compressed": did_compress,
        "cutoff": cutoff,
        "requested_max_bond": requested_max_bond,
        "raw_max_bond": raw_max_bond,
        "final_max_bond": final_max_bond,
        "rank_reduced": final_max_bond < raw_max_bond,
        "cap_bound": (
            did_compress
            and requested_max_bond is not None
            and raw_max_bond > requested_max_bond
        ),
        "max_bond_exceeded": (
            did_compress
            and requested_max_bond is not None
            and final_max_bond > requested_max_bond
        ),
    }
    # This record describes MPO construction only; it is not used during
    # contraction and can safely travel with the returned MPO as user-facing
    # build metadata.
    mpo.pepsy_compression_report = report
    if report["max_bond_exceeded"]:
        warnings.warn(
            "SymHamiltonian.to_mpo requested "
            f"max_bond={requested_max_bond}, but Symmray compression returned "
            f"max bond {final_max_bond}. Tied singular values at the "
            "truncation threshold can make this a soft cap; inspect "
            "mpo.pepsy_compression_report before relying on a hard memory "
            "limit.",
            RuntimeWarning,
            stacklevel=3,
        )
    return mpo


def _build_factorized_pair_mpo(
    *,
    L,
    pair_create,
    pair_annihilate,
    onsite,
    signs,
    normalization,
    symmetry,
    max_bond=None,
    cutoff=1e-12,
    compress=False,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    dtype=None,
):
    """Build a compact native MPO for a staggered pair structure factor.

    The operator is assembled from the finite-state identity

    ``normalization * ((sum_i s_i D_i^dagger) (sum_j s_j D_j)
    - sum_i D_i^dagger D_i)``.

    The explicit two-site expansion has ``O(L**2)`` terms, but this automaton
    has only two open pair channels plus start/done.  In particular, it does
    not first create one MPO channel per pair and then try to compress the
    result.  The local operators are native Symmray fermionic arrays; the
    graded evaluator therefore remains responsible for the MPO--MPS signs.
    """
    if not isinstance(L, Integral) or int(L) < 1:
        raise ValueError("L must be a positive integer.")
    L = int(L)
    signs = tuple(signs)
    if len(signs) != L:
        raise ValueError(f"signs must contain exactly L={L} values.")
    normalization = np.asarray(normalization).reshape(())
    if not np.isfinite(normalization):
        raise ValueError("normalization must be finite.")
    dtype = np.dtype(dtype or np.result_type(
        _dense_numpy(pair_create).dtype,
        _dense_numpy(pair_annihilate).dtype,
        _dense_numpy(onsite).dtype,
        np.asarray(signs).dtype,
        normalization.dtype,
    ))

    create_dense, phys_map = _term_dense_and_phys_map(
        pair_create,
        dtype=dtype,
    )
    annihilate_dense, annihilate_phys_map = _term_dense_and_phys_map(
        pair_annihilate,
        dtype=dtype,
    )
    _, onsite_phys_map = _term_dense_and_phys_map(
        onsite,
        dtype=dtype,
    )
    if phys_map != annihilate_phys_map or phys_map != onsite_phys_map:
        raise ValueError(
            "factorized pair MPO operators must share one physical charge map."
        )

    # Preserve the charge rank for product symmetries (e.g. ``U1U1``); a
    # scalar ``0`` is not equal to the neutral tuple ``(0, 0)``.
    zero = _normalize_group_charge(
        _zero_like_charge(getattr(pair_create, "charge", 0)),
        symmetry,
    )
    create_charge = _normalize_group_charge(
        getattr(pair_create, "charge", zero), symmetry
    )
    annihilate_charge = _normalize_group_charge(
        getattr(pair_annihilate, "charge", zero), symmetry
    )
    onsite_charge = _normalize_group_charge(
        getattr(onsite, "charge", zero), symmetry
    )
    if create_charge == zero or annihilate_charge == zero:
        raise ValueError("pair creation and annihilation operators must be charged.")
    if create_charge != _charge_neg(annihilate_charge, symmetry):
        raise ValueError(
            "pair creation and annihilation charges must be negatives of one "
            "another."
        )
    if onsite_charge != zero:
        raise ValueError("the onsite pair subtraction must be charge neutral.")

    # ``_assemble_symmray_mpo`` reserves these boundary labels for the
    # identity path shared by all of its finite-state MPO builders.
    start = ("start",)
    create_state = ("factorized_pair_create",)
    annihilate_state = ("factorized_pair_annihilate",)
    done = ("done",)
    channels = [
        [
            (start, zero),
            (create_state, _charge_neg(create_charge, symmetry)),
            (annihilate_state, _charge_neg(annihilate_charge, symmetry)),
            (done, zero),
        ]
        for _ in range(max(L - 1, 0))
    ]
    transitions = [[] for _ in range(L)]
    identity = np.eye(len(phys_map), dtype=dtype)
    scaled_normalization = dtype.type(normalization)

    for site in range(L):
        if site < L - 1:
            # Put the overall normalization on the closing endpoint.  The
            # open-channel transition is the first factor in F^dagger F, so
            # applying normalization at both ends would incorrectly produce
            # normalization**2 for every pair.
            coefficient = dtype.type(signs[site])
            transitions[site].extend(
                (
                    (start, create_state, coefficient * create_dense),
                    (start, annihilate_state, coefficient * annihilate_dense),
                )
            )

        if 0 < site < L - 1:
            transitions[site].extend(
                (
                    (create_state, create_state, identity),
                    (annihilate_state, annihilate_state, identity),
                )
            )

        if site > 0:
            coefficient = scaled_normalization * dtype.type(signs[site])
            transitions[site].extend(
                (
                    (create_state, done, coefficient * annihilate_dense),
                    (annihilate_state, done, coefficient * create_dense),
                )
            )

        # The start->done onsite transition is intentionally omitted.  The
        # finite-state paths above contain only i < j and i > j, which is
        # exactly the off-diagonal structure factor.  Equivalently, this is
        # F^dagger F - sum_i Delta_i^dagger Delta_i without constructing the
        # diagonal term and subtracting it afterward.

    mpo = _assemble_symmray_mpo(
        L=L,
        channels=channels,
        transitions=transitions,
        phys_map=phys_map,
        symmetry=symmetry,
        zero=zero,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        to_backend=to_backend,
        fermionic=True,
        operator_charge=zero,
    )
    mpo.pepsy_compression_report.update(
        {
            "factorized_pair": True,
            "factorized_channels": 4,
            "pair_term_count": L * (L - 1) // 2,
            "normalization": complex(normalization),
        }
    )
    return mpo


def _build_fermionic_model_mpo(
    hamiltonian,
    L,
    *,
    mapper=None,
    idx2coo=None,
    coo2idx=None,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    dtype=None,
):
    _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
        mapper=mapper,
        idx2coo=idx2coo,
        coo2idx=coo2idx,
    )
    raw_edges = _as_edges(hamiltonian.edges)
    edges = _map_edges_to_mpo_indices(raw_edges, coo2idx_use)

    if L is None:
        L = (
            mapped_L
            if mapped_L is not None
            else max(max(int(i), int(j)) for i, j in edges) + 1
        )
    L = int(L)
    if L < 1:
        raise ValueError("L must be a positive integer.")
    if mapped_L is not None and L != mapped_L:
        raise ValueError(f"L={L} does not match MPO mapping length {mapped_L}.")

    dtype = (
        _dtype_from_hamiltonian_terms(hamiltonian.terms)
        if dtype is None
        else np.dtype(dtype)
    )
    zero = _normalize_group_charge(
        getattr(next(iter(hamiltonian.terms.values())), "charge", 0),
        hamiltonian.symmetry,
    )
    start = ("start",)
    done = ("done",)
    channels = [[(start, zero), (done, zero)] for _ in range(max(L - 1, 0))]
    transitions = [[] for _ in range(L)]

    coordinations = {}
    for left, right in raw_edges:
        coordinations[left] = coordinations.setdefault(left, 0) + 1
        coordinations[right] = coordinations.setdefault(right, 0) + 1

    if hamiltonian.model != "fermi_hubbard_spinless":  # pragma: no cover - guarded by caller
        raise NotImplementedError(f"Unsupported fermionic model {hamiltonian.model!r}.")
    delta = hamiltonian.parameters.get("delta", 0.0)
    if delta != 0:
        raise NotImplementedError(
            "SymHamiltonian.to_mpo does not yet support spinless "
            "Fermi-Hubbard pairing terms with delta != 0."
        )
    ops = _fh_spinless_dense_local_ops(hamiltonian.symmetry, dtype)
    phys_map = list(ops["index_map"])
    mode_terms = (("spinless", "t", ops["create"], ops["annihilate"], 1),)

    def add_channel(edge_pos, i, j, label, left_charge, left_op, right_op):
        channel_id = ("fermion", edge_pos, label, left_charge)
        channel_charge = _charge_neg(left_charge, hamiltonian.symmetry)
        for cut in range(i, j):
            channels[cut].append((channel_id, channel_charge))
        transitions[i].append((start, channel_id, left_op))
        string_op = (
            ops["parity"]
            if _charged_op_needs_fermion_string(left_charge)
            else ops["identity"]
        )
        for site in range(i + 1, j):
            transitions[site].append((channel_id, channel_id, string_op))
        transitions[j].append((channel_id, done, right_op))

    for edge_pos, (raw_edge, edge) in enumerate(zip(raw_edges, edges)):
        raw_left, raw_right = raw_edge
        left_site, right_site = int(edge[0]), int(edge[1])
        if left_site == right_site:
            raise ValueError("Hamiltonian edges must connect distinct sites.")
        if not (0 <= left_site < L and 0 <= right_site < L):
            raise ValueError(f"edge {edge!r} is outside MPO length L={L}.")

        t_values = {"t": _edge_parameter(hamiltonian.parameters.get("t", 1.0), raw_left, raw_right)}
        V_edge = _edge_parameter(hamiltonian.parameters.get("V", 0.0), raw_left, raw_right)
        if V_edge != 0:
            i, j = sorted((left_site, right_site))
            add_channel(edge_pos, i, j, "V", zero, V_edge * ops["number"], ops["number"])

        for raw_site, site in ((raw_left, left_site), (raw_right, right_site)):
            coordination = coordinations[raw_site]
            mu_site = _node_parameter(hamiltonian.parameters.get("mu", 0.0), raw_site)
            onsite = -(mu_site / coordination) * ops["number"]
            if np.any(onsite != 0):
                transitions[site].append((start, done, onsite))

        i, j = sorted((left_site, right_site))
        for spin, t_key, create, annihilate, create_charge in mode_terms:
            t_sigma = t_values[t_key]
            if t_sigma == 0:
                continue
            for direction, first, second, first_charge in (
                ("forward", create, annihilate, create_charge),
                ("backward", annihilate, create, _charge_neg(create_charge, hamiltonian.symmetry)),
            ):
                endpoint = (
                    first @ ops["parity"]
                    if direction == "forward"
                    else ops["parity"] @ first
                )
                add_channel(
                    edge_pos,
                    i,
                    j,
                    (spin, direction),
                    first_charge,
                    -t_sigma * endpoint,
                    second,
                )

    return _assemble_symmray_mpo(
        L=L,
        channels=channels,
        transitions=transitions,
        phys_map=phys_map,
        symmetry=hamiltonian.symmetry,
        zero=zero,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        to_backend=to_backend,
    )


def _add_native_term_to_mpo(
    term,
    sites,
    *,
    term_pos,
    channels,
    transitions,
    symmetry,
    dtype,
    zero,
    operator_charge=None,
):
    """Add one homogeneous native fermion term as graded MPO transitions.

    The term is split by operator Schmidt decompositions over the ordered
    support sites. This preserves Symmray's fermionic bond phases while
    allowing arbitrary term rank and non-contiguous support.
    """
    sites = tuple(int(site) for site in sites)
    if len(set(sites)) != len(sites):
        raise ValueError("Hamiltonian term supports must contain unique sites.")
    if any(site < 0 or site >= len(transitions) for site in sites):
        raise ValueError(f"Hamiltonian term support {sites!r} is outside MPO bounds.")
    start = ("start",)
    done = ("done",)

    order = tuple(sorted(range(len(sites)), key=sites.__getitem__))
    sites = tuple(sorted(sites))
    indices = getattr(term, "indices", None)
    if indices is None or len(indices) == 0 or len(indices) % 2:
        raise TypeError(
            "SymHamiltonian.to_mpo requires an even-rank Symmray operator."
        )
    n_sites = len(sites)
    if len(indices) != 2 * n_sites:
        raise ValueError("Term metadata and support size disagree.")

    output_maps = [_expanded_index_charges(index) for index in indices[:n_sites]]
    input_maps = [_expanded_index_charges(index) for index in indices[n_sites:]]
    if output_maps != input_maps:
        raise ValueError(
            "Hamiltonian terms must use matching upper/lower physical charge "
            "maps at every site."
        )
    if any(site_map != output_maps[0] for site_map in output_maps[1:]):
        raise ValueError(
            "SymHamiltonian.to_mpo currently requires one physical charge "
            "map shared by all sites."
        )
    physical_maps = [output_maps[pos] for pos in order]

    term_charge = _normalize_group_charge(
        getattr(term, "charge", zero),
        symmetry,
    )
    expected_charge = (
        zero
        if operator_charge is None
        else _normalize_group_charge(operator_charge, symmetry)
    )
    if term_charge != expected_charge:
        raise ValueError(
            "Native MPO terms must share one homogeneous operator charge "
            f"{expected_charge!r}; term {term_pos} has charge {term_charge!r}."
        )

    if n_sites == 1:
        dense = _dense_numpy(term, dtype=dtype)
        for out_i, in_i in product(range(len(physical_maps[0])), repeat=2):
            coefficient = dense[out_i, in_i]
            if not np.any(coefficient):
                continue
            local = np.zeros((len(physical_maps[0]), len(physical_maps[0])), dtype=dtype)
            local[out_i, in_i] = coefficient
            transitions[sites[0]].append((start, done, local))
        return list(physical_maps[0])

    axes = order + tuple(n_sites + pos for pos in order)
    ordered_term = term if order == tuple(range(n_sites)) else term.transpose(axes)
    local_term = ordered_term.fuse(
        *((pos, n_sites + pos) for pos in range(n_sites))
    )

    factors = []
    current = local_term
    for pos in range(n_sites - 1):
        ndim = current.ndim
        if pos == 0:
            left_group = (0,)
            right_group = tuple(range(1, ndim))
        else:
            left_group = (0, 1)
            right_group = tuple(range(2, ndim))
        # Absorb the singular values into the left factor for a charged
        # operator. This leaves the total operator charge on the final site
        # factor, where the open MPO boundary can carry it, while all
        # preceding tensors remain neutral and can propagate identity paths.
        absorb = "left" if expected_charge != zero else "right"
        left, _, right = current.fuse(left_group, right_group).svd(
            absorb=absorb
        )
        if pos == 0:
            factors.append(left.unfuse(0).transpose((2, 0, 1)))
        else:
            factors.append(
                left.unfuse(0).unfuse(1).transpose((0, 3, 1, 2))
            )
        current = right.unfuse(1)
    factors.append(current)

    # Each operator-Schmidt bond becomes a family of MPO channels. The
    # channel charge is taken directly from the native factor bond index,
    # rather than reconstructed from dense matrix elements.
    interval_channel_ids = []
    interval_channel_charges = []
    for interval in range(n_sites - 1):
        factor = factors[interval]
        bond_axis = 0 if interval == 0 else 1
        bond_map = _expanded_index_charges(factor.indices[bond_axis])
        channel_ids = []
        for bond_pos, bond_charge in enumerate(bond_map):
            if expected_charge != zero:
                # With absorb="left", the factor bond is dual on the side
                # that becomes the MPO's outgoing bond. Reverse its charge
                # when installing the common MPO bond orientation.
                bond_charge = _charge_neg(bond_charge, symmetry)
            channel_id = ("native", term_pos, interval, bond_pos)
            channel_ids.append(channel_id)
            for cut in range(sites[interval], sites[interval + 1]):
                channels[cut].append((channel_id, bond_charge))
        interval_channel_ids.append(channel_ids)
        interval_channel_charges.append(tuple(bond_map))

    first_data = _dense_numpy(factors[0], dtype=dtype)
    for bond_pos, channel_id in enumerate(interval_channel_ids[0]):
        op = first_data[bond_pos]
        if np.any(op):
            transitions[sites[0]].append((start, channel_id, op))

    for interval in range(n_sites - 2):
        factor_data = _dense_numpy(factors[interval + 1], dtype=dtype)
        left_ids = interval_channel_ids[interval]
        right_ids = interval_channel_ids[interval + 1]
        for left_pos, left_id in enumerate(left_ids):
            for right_pos, right_id in enumerate(right_ids):
                op = factor_data[left_pos, right_pos]
                if np.any(op):
                    transitions[sites[interval + 1]].append(
                        (left_id, right_id, op)
                    )

    last_data = _dense_numpy(factors[-1], dtype=dtype)
    for bond_pos, channel_id in enumerate(interval_channel_ids[-1]):
        op = last_data[bond_pos]
        if np.any(op):
            transitions[sites[-1]].append((channel_id, done, op))

    identity = np.eye(len(physical_maps[0]), dtype=dtype)
    for interval, channel_ids in enumerate(interval_channel_ids):
        for site in range(sites[interval] + 1, sites[interval + 1]):
            for channel_pos, channel_id in enumerate(channel_ids):
                # A skipped site still participates in the graded ordering.
                # For an odd operator-Schmidt channel, moving that channel
                # past one omitted physical site contributes a scalar -1.
                # This is a graded factorization phase, not a JW parity
                # operator: inserting the latter would change the native
                # operator by making the result state-dependent.
                bond_charge = interval_channel_charges[interval][channel_pos]
                phase_op = (
                    -identity
                    if _charged_op_needs_fermion_string(bond_charge)
                    else identity
                )
                transitions[site].append((channel_id, channel_id, phase_op))

    return list(physical_maps[0])


def _native_local_term_mpo(
    term,
    support,
    L,
    *,
    symmetry,
    dtype,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
):
    """Build an exact local-term MPO without start/done channel inflation.

    The generic native MPO assembler is designed for a collection of terms,
    so it carries explicit start and done paths at every chain cut. For one
    one-site or two-site term those paths are unnecessary. Factorizing the
    native local array directly and propagating its operator-Schmidt bond
    through identity tensors leaves only the non-zero local Schmidt sectors.
    """
    support = tuple(int(site) for site in support)
    if len(support) not in {1, 2} or len(set(support)) != len(support):
        raise ValueError("direct local PEPO terms must act on one or two sites.")
    if any(site < 0 or site >= int(L) for site in support):
        raise ValueError(f"term support {support!r} is outside MPO length L={L}.")

    _require_symmray()
    from symmray import utils as sr_utils  # pylint: disable=import-outside-toplevel

    zero = _zero_like_charge(0 if symmetry in {"U1", "Z2"} else (0, 0))
    term_charge = _normalize_group_charge(
        getattr(term, "charge", zero), symmetry
    )
    indices = getattr(term, "indices", None)
    if indices is None or len(indices) != 2 * len(support):
        raise TypeError("direct local PEPO terms require matching native rank.")

    physical_maps = [
        _expanded_index_charges(index) for index in indices[:len(support)]
    ]
    input_maps = [
        _expanded_index_charges(index) for index in indices[len(support):]
    ]
    if physical_maps != input_maps or any(
        physical_maps[site] != physical_maps[0]
        for site in range(len(support))
    ):
        raise ValueError(
            "direct local PEPO terms require one matching physical charge map."
        )
    phys_map = physical_maps[0]
    phys_dim = len(phys_map)
    zero_map = [zero]

    def make_array(data, index_maps, duals, *, charge=zero, label=None):
        return sr_utils.from_dense(
            data,
            symmetry=symmetry,
            index_maps=index_maps,
            duals=duals,
            fermionic=True,
            charge=charge,
            label=label,
        )

    def identity_tensor(site):
        identity = np.eye(phys_dim, dtype=dtype)
        if L == 1:
            return make_array(identity, [phys_map, phys_map], [False, True])
        if site == 0:
            return make_array(
                identity.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [False, False, True],
            )
        if site == L - 1:
            return make_array(
                identity.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [True, False, True],
            )
        data = np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
        data[0, 0] = identity
        return make_array(
            data,
            [zero_map, zero_map, phys_map, phys_map],
            [True, False, False, True],
        )

    arrays = [identity_tensor(site) for site in range(int(L))]
    local_schmidt_bond = 1

    if len(support) == 1:
        site = support[0]
        dense = _dense_numpy(term, dtype=dtype)
        label = site if _charged_op_needs_fermion_string(term_charge) else None
        if L == 1:
            arrays[site] = make_array(
                dense, [phys_map, phys_map], [False, True],
                charge=term_charge, label=label,
            )
        elif site == 0:
            arrays[site] = make_array(
                dense.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [False, False, True],
                charge=term_charge, label=label,
            )
        elif site == L - 1:
            arrays[site] = make_array(
                dense.reshape(1, phys_dim, phys_dim),
                [zero_map, phys_map, phys_map],
                [True, False, True],
                charge=term_charge, label=label,
            )
        else:
            arrays[site] = make_array(
                dense.reshape(1, 1, phys_dim, phys_dim),
                [zero_map, zero_map, phys_map, phys_map],
                [True, False, False, True],
                charge=term_charge, label=label,
            )
    else:
        # Order the support by the MPO chain. A native operator's upper and
        # lower legs are reordered together so its graded local signs survive.
        ordered = tuple(sorted(enumerate(support), key=lambda item: item[1]))
        if tuple(item[0] for item in ordered) == (0, 1):
            ordered_term = term
        else:
            ordered_term = term.transpose((1, 0, 3, 2))
        fused = ordered_term.fuse((0, 2), (1, 3))
        # The only cutoff here removes exact numerical zero singular blocks
        # left by Symmray's block SVD. It is not the user-requested PEPO
        # compression cutoff and does not cap the resulting local bond.
        structural_cutoff = 64.0 * np.finfo(float).eps
        left, _, right = fused.svd(
            absorb="right",
            cutoff=structural_cutoff,
        )
        left = left.unfuse(0).transpose((2, 0, 1))
        right = right.unfuse(1)
        bond_map = _expanded_index_charges(left.indices[0])
        bond_dim = len(bond_map)
        local_schmidt_bond = bond_dim
        left_dense = _dense_numpy(left, dtype=dtype)
        right_dense = _dense_numpy(right, dtype=dtype)
        left_charge = _normalize_group_charge(
            getattr(left, "charge", zero), symmetry
        )
        right_charge = _normalize_group_charge(
            getattr(right, "charge", zero), symmetry
        )
        left_site, right_site = (item[1] for item in ordered)

        if left_site == 0:
            arrays[left_site] = left
        else:
            arrays[left_site] = make_array(
                left_dense.reshape(1, bond_dim, phys_dim, phys_dim),
                [zero_map, bond_map, phys_map, phys_map],
                [True, False, False, True],
                charge=left_charge,
            )
        if right_site == L - 1:
            arrays[right_site] = right
        else:
            arrays[right_site] = make_array(
                right_dense.reshape(bond_dim, 1, phys_dim, phys_dim),
                [bond_map, zero_map, phys_map, phys_map],
                [True, False, False, True],
                charge=right_charge,
            )
        identity = np.eye(phys_dim, dtype=dtype)
        for site in range(left_site + 1, right_site):
            data = np.zeros(
                (bond_dim, bond_dim, phys_dim, phys_dim),
                dtype=dtype,
            )
            for bond_pos in range(bond_dim):
                data[bond_pos, bond_pos] = identity
            arrays[site] = make_array(
                data,
                [bond_map, bond_map, phys_map, phys_map],
                [True, False, False, True],
            )

    mpo = qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
    )
    if to_backend is not None:
        _apply_to_tensor_network_arrays(mpo, to_backend)
    raw_bond = mpo.max_bond()
    raw_max_bond = 1 if raw_bond is None else int(raw_bond)
    did_compress = bool(compress and L > 1)
    if did_compress:
        compress_opts = {"cutoff": cutoff}
        if max_bond is not None:
            compress_opts["max_bond"] = int(max_bond)
        mpo.compress(**compress_opts)
    final_bond = mpo.max_bond()
    final_max_bond = 1 if final_bond is None else int(final_bond)
    mpo.pepsy_compression_report = {
        "direct_local": True,
        "compressed": did_compress,
        "cutoff": cutoff,
        "requested_max_bond": None if max_bond is None else int(max_bond),
        "operator_schmidt_bond": local_schmidt_bond,
        "raw_max_bond": raw_max_bond,
        "final_max_bond": final_max_bond,
        "rank_reduced": final_max_bond < raw_max_bond,
        "max_bond_exceeded": (
            did_compress
            and max_bond is not None
            and final_max_bond > int(max_bond)
        ),
    }
    return mpo


def _generic_symhamiltonian_to_mpo(
    hamiltonian,
    L,
    *,
    mapper=None,
    idx2coo=None,
    coo2idx=None,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    to_backend=None,
    dtype=None,
    fermionic=False,
):
    """Build a Symmray MPO from explicit local terms."""
    _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
        mapper=mapper,
        idx2coo=idx2coo,
        coo2idx=coo2idx,
    )
    raw_wheres = tuple(hamiltonian.terms)
    if not raw_wheres:
        raise ValueError("At least one Hamiltonian term is required to build an MPO.")
    coordinate_sites = _term_mapping_uses_coordinate_sites(raw_wheres)
    wheres = tuple(
        _as_term_where(where, coordinate_sites=coordinate_sites)
        for where in raw_wheres
    )
    mapped_wheres = tuple(
        tuple(_map_site_to_mpo_index(site, coo2idx_use) for site in where)
        for where in wheres
    )

    if L is None:
        L = (
            mapped_L
            if mapped_L is not None
            else max(max(int(site) for site in where) for where in mapped_wheres) + 1
        )
    L = int(L)
    if L < 1:
        raise ValueError("L must be a positive integer.")
    if mapped_L is not None and L != mapped_L:
        raise ValueError(f"L={L} does not match MPO mapping length {mapped_L}.")

    dtype = (
        _dtype_from_hamiltonian_terms(hamiltonian.terms)
        if dtype is None
        else np.dtype(dtype)
    )
    first_term = next(iter(hamiltonian.terms.values()))
    first_charge = _normalize_group_charge(
        getattr(first_term, "charge", 0),
        hamiltonian.symmetry,
    )
    zero = _zero_like_charge(first_charge)
    operator_charge = first_charge if fermionic else zero
    if not fermionic and first_charge != zero:
        raise ValueError(
            "Charged native operator terms require fermionic=True; the "
            "Jordan-Wigner compatibility MPO is neutral-only."
        )
    start = ("start",)
    done = ("done",)
    channels = [
        [(start, zero), (done, _charge_neg(operator_charge, hamiltonian.symmetry))]
        for _ in range(max(L - 1, 0))
    ]
    transitions = [[] for _ in range(L)]
    phys_map = None

    for term_pos, (raw_where, where) in enumerate(zip(raw_wheres, mapped_wheres)):
        term = hamiltonian.terms[raw_where]
        term_is_fermionic = _is_fermionic_symmray_array(term)
        term_charge = _normalize_group_charge(
            getattr(term, "charge", zero),
            hamiltonian.symmetry,
        )
        if not fermionic and term_charge != zero:
            raise ValueError(
                "Charged native operator terms require fermionic=True; the "
                "Jordan-Wigner compatibility MPO is neutral-only."
            )
        if fermionic and not term_is_fermionic:
            raise TypeError(
                "Native fermionic MPO construction requires every Hamiltonian "
                "term to be a Symmray FermionicArray."
            )

        if fermionic:
            term_phys = _add_native_term_to_mpo(
                term,
                where,
                term_pos=term_pos,
                channels=channels,
                transitions=transitions,
                symmetry=hamiltonian.symmetry,
                dtype=dtype,
                zero=zero,
                operator_charge=operator_charge,
            )
            if phys_map is None:
                phys_map = term_phys
            elif phys_map != term_phys:
                raise ValueError(
                    "SymHamiltonian.to_mpo requires one physical charge map "
                    "shared by all sites."
                )
            continue

        if len(where) == 1:
            site = int(where[0])
            if not 0 <= site < L:
                raise ValueError(f"site {site!r} is outside MPO length L={L}.")
            dense, term_phys = _term_dense_and_phys_map(term, dtype=dtype)
            if phys_map is None:
                phys_map = list(term_phys)
            elif phys_map != list(term_phys):
                raise ValueError(
                    "SymHamiltonian.to_mpo requires one physical charge map "
                    "shared by all sites."
                )
            transitions[site].append((start, done, dense))
            continue

        if len(where) > 2:
            raise NotImplementedError(
                "Jordan-Wigner compatibility MPO conversion currently supports "
                "one- and two-site terms; use fermionic=True for native "
                "multi-site terms."
            )

        i, j = (int(where[0]), int(where[1]))
        if i == j:
            raise ValueError("Hamiltonian edges must connect distinct sites.")
        if not (0 <= i < L and 0 <= j < L):
            raise ValueError(f"edge {where!r} is outside MPO length L={L}.")
        reverse = i > j
        if reverse:
            i, j = j, i

        term_channels, left_phys, right_phys = _decompose_neutral_two_site_term(
            term,
            symmetry=hamiltonian.symmetry,
            dtype=dtype,
            reverse=reverse,
            # ``_decompose_neutral_two_site_term(..., fermionic=True)`` is the
            # legacy conversion from native local data to a bosonic/JW
            # site-major matrix. A native graded MPO keeps the raw fermionic
            # tensor ordering and lets Symmray supply the Koszul signs.
            fermionic=term_is_fermionic and not fermionic,
        )
        if left_phys != right_phys:
            raise ValueError(
                "SymHamiltonian.to_mpo currently requires a uniform physical "
                "charge map on both sites of each term."
            )
        if phys_map is None:
            phys_map = list(left_phys)
        elif phys_map != list(left_phys):
            raise ValueError(
                "SymHamiltonian.to_mpo currently requires one physical charge "
                "map shared by all sites."
            )

        for rank, (left_charge, left_op, right_op) in enumerate(term_channels):
            channel_id = ("term", term_pos, rank, left_charge)
            channel_charge = _charge_neg(left_charge, hamiltonian.symmetry)
            for cut in range(i, j):
                channels[cut].append((channel_id, channel_charge))
            transitions[i].append((start, channel_id, left_op))

            if (
                term_is_fermionic
                and not fermionic
                and _charged_op_needs_fermion_string(left_charge)
            ):
                string_op = _fermion_parity_operator(phys_map, dtype)
            else:
                string_op = np.eye(len(phys_map), dtype=dtype)
            for site in range(i + 1, j):
                transitions[site].append((channel_id, channel_id, string_op))

            transitions[j].append((channel_id, done, right_op))

    if phys_map is None:
        raise ValueError("At least one Hamiltonian term is required to build an MPO.")

    return _assemble_symmray_mpo(
        L=L,
        channels=channels,
        transitions=transitions,
        phys_map=phys_map,
        symmetry=hamiltonian.symmetry,
        zero=zero,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        compress=compress,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id=site_tag_id,
        to_backend=to_backend,
        fermionic=fermionic,
        operator_charge=operator_charge,
    )


def _group_symhamiltonian_terms_by_charge(hamiltonian):
    """Group native Hamiltonian terms into homogeneous charge sectors."""
    sectors = {}
    for where, term in hamiltonian.terms.items():
        charge = _normalize_group_charge(
            getattr(term, "charge", 0),
            hamiltonian.symmetry,
        )
        sectors.setdefault(charge, {})[where] = term
    return sectors


@dataclass(frozen=True)
class SymHamiltonian:
    """Container for Symmray local Hamiltonian terms."""

    model: str
    symmetry: str
    edges: tuple
    terms: dict
    parameters: dict = field(default_factory=dict)
    explicit_terms: bool = False

    @classmethod
    def from_edges(cls, model, symmetry, edges, *, flat=False, to_backend=None, **params):
        """Build a Symmray Hamiltonian dictionary from lattice edges."""
        model_norm = _normalize_model(model)
        edges = _as_edges(edges)
        terms = _hamiltonian_from_edges(model_norm, symmetry, edges, flat=flat, **params)
        terms = _apply_to_hamiltonian_terms(terms, to_backend)
        return cls(
            model=model_norm,
            symmetry=str(symmetry),
            edges=edges,
            terms=terms,
            parameters=dict(params),
        )

    @classmethod
    def from_terms(
        cls,
        model,
        symmetry,
        terms,
        *,
        to_backend=None,
        parameters=None,
    ):
        """Build a Hamiltonian container from explicit local operators.

        ``terms`` maps a site label or support tuple to a native local
        operator. This preserves the operator locations while
        retaining the model and symmetry metadata required for fermionic MPO
        conversion.
        """
        model_norm = _normalize_model(model)
        terms = _normalize_term_mapping(terms)
        terms = _apply_to_hamiltonian_terms(terms, to_backend)
        return cls(
            model=model_norm,
            symmetry=str(symmetry),
            edges=tuple(terms),
            terms=terms,
            parameters=dict(parameters or {}),
            explicit_terms=True,
        )

    def apply_to_arrays(self, fn, *, inplace=True):
        """Apply ``fn`` to each dense block of each Hamiltonian term."""
        _validate_backend_mapper(fn)
        target = self if inplace else type(self)(
            model=self.model,
            symmetry=self.symmetry,
            edges=self.edges,
            terms={
                edge: _copy_array_like(term)
                for edge, term in self.terms.items()
            },
            parameters=dict(self.parameters),
            explicit_terms=self.explicit_terms,
        )
        for edge, term in list(target.terms.items()):
            target.terms[edge] = _apply_to_array_blocks(term, fn)
        return target

    def to_backend(self, to_backend, *, inplace=True):
        """Convert Hamiltonian term blocks with a backend mapper callable."""
        return self.apply_to_arrays(to_backend, inplace=inplace)

    def to_mpo(
        self,
        L=None,
        *,
        mapper=None,
        idx2coo=None,
        coo2idx=None,
        max_bond=None,
        cutoff=1e-12,
        compress=True,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        to_backend=None,
        dtype=None,
        fermionic=False,
        charge_sectors=False,
    ):
        """Build a symmetry-preserving MPS-chain MPO for this Hamiltonian.

        Coordinate-lattice edges can be mapped with ``mapper=OneDMap(...)`` or
        the ``idx2coo, coo2idx`` dictionaries from ``OneDMap(...).build()``.
        Fermionic compatibility paths include parity strings along
        non-adjacent mapped hopping channels. With ``fermionic=True``, native
        Symmray ``FermionicArray`` tensors are built directly from arbitrary
        homogeneous-charge one- or multi-site terms. The open MPO boundary
        carries a nonzero operator charge when required.

        With ``charge_sectors=True``, return a mapping from each operator
        charge to its own homogeneous native MPO. This is the explicit way to
        represent a mixed-charge operator such as ``I + c^\u2020`` without
        converting it to a dense or non-symmetric tensor.
        """
        if charge_sectors:
            if not fermionic:
                raise ValueError("charge_sectors=True requires fermionic=True.")
            sectors = _group_symhamiltonian_terms_by_charge(self)
            if not sectors:
                raise ValueError(
                    "At least one Hamiltonian term is required to build an MPO."
                )
            return {
                charge: type(self).from_terms(
                    self.model,
                    self.symmetry,
                    terms,
                    parameters=self.parameters,
                ).to_mpo(
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
                    to_backend=to_backend,
                    dtype=dtype,
                    fermionic=True,
                    charge_sectors=False,
                )
                for charge, terms in sectors.items()
            }
        if self.explicit_terms or fermionic:
            return _generic_symhamiltonian_to_mpo(
                self,
                L,
                mapper=mapper,
                idx2coo=idx2coo,
                coo2idx=coo2idx,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                upper_ind_id=upper_ind_id,
                lower_ind_id=lower_ind_id,
                site_tag_id=site_tag_id,
                to_backend=to_backend,
                dtype=dtype,
                fermionic=fermionic,
            )

        if self.model == "fermi_hubbard_spinless":
            return _build_fermionic_model_mpo(
                self,
                L,
                mapper=mapper,
                idx2coo=idx2coo,
                coo2idx=coo2idx,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                upper_ind_id=upper_ind_id,
                lower_ind_id=lower_ind_id,
                site_tag_id=site_tag_id,
                to_backend=to_backend,
                dtype=dtype,
            )

        is_spinful_fh_mpo = (
            (self.model == "fermi_hubbard" and self.symmetry == "U1")
            or (self.model == "fermi_hubbard_u1u1" and self.symmetry == "U1U1")
        )
        if not is_spinful_fh_mpo:
            return _generic_symhamiltonian_to_mpo(
                self,
                L,
                mapper=mapper,
                idx2coo=idx2coo,
                coo2idx=coo2idx,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                upper_ind_id=upper_ind_id,
                lower_ind_id=lower_ind_id,
                site_tag_id=site_tag_id,
                to_backend=to_backend,
                dtype=dtype,
            )

        _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
            mapper=mapper,
            idx2coo=idx2coo,
            coo2idx=coo2idx,
        )
        raw_edges = _as_edges(self.edges)
        edges = _map_edges_to_mpo_indices(raw_edges, coo2idx_use)

        if L is None:
            L = (
                mapped_L
                if mapped_L is not None
                else max(max(int(i), int(j)) for i, j in edges) + 1
            )
        L = int(L)
        if L < 1:
            raise ValueError("L must be a positive integer.")
        if mapped_L is not None and L != mapped_L:
            raise ValueError(f"L={L} does not match MPO mapping length {mapped_L}.")

        dtype = _dtype_from_hamiltonian_terms(self.terms) if dtype is None else np.dtype(dtype)
        ops = _fh_spinful_jw_local_ops(self.symmetry, dtype)
        phys_map = list(ops["index_map"])
        zero = _zero_like_charge(next(iter(phys_map)))
        start = ("start",)
        done = ("done",)

        t_u, t_d = _as_spin_pair(self.parameters.get("t", 1.0), name="t")
        mu_u, mu_d = _as_spin_pair(self.parameters.get("mu", 0.0), name="mu")
        U = self.parameters.get("U", 8.0)
        V = self.parameters.get("V", 0.0)

        channels = [[(start, zero), (done, zero)] for _ in range(max(L - 1, 0))]
        transitions = [[] for _ in range(L)]

        onsite = (
            U * ops["double"]
            - mu_u * ops["number_u"]
            - mu_d * ops["number_d"]
        )
        if np.any(onsite != 0):
            for site in range(L):
                transitions[site].append((start, done, onsite))

        parity = ops["parity"]
        if self.symmetry == "U1U1":
            create_u_charge = (0, 1)
            create_d_charge = (1, 0)
        else:
            create_u_charge = 1
            create_d_charge = 1
        mode_terms = (
            (t_u, ops["create_u"], ops["annihilate_u"], create_u_charge, "u"),
            (t_d, ops["create_d"], ops["annihilate_d"], create_d_charge, "d"),
        )
        number = ops["number_u"] + ops["number_d"]
        for edge_pos, (raw_edge, edge) in enumerate(zip(raw_edges, edges)):
            i, j = (int(edge[0]), int(edge[1]))
            if i == j:
                raise ValueError("Hamiltonian edges must connect distinct sites.")
            if not (0 <= i < L and 0 <= j < L):
                raise ValueError(f"edge {edge!r} is outside MPO length L={L}.")
            if i > j:
                i, j = j, i
            V_edge = _edge_parameter(V, raw_edge[0], raw_edge[1])
            if V_edge != 0:
                channel_id = ("density", edge_pos)
                for cut in range(i, j):
                    channels[cut].append((channel_id, zero))
                transitions[i].append((start, channel_id, V_edge * number))
                for site in range(i + 1, j):
                    transitions[site].append(
                        (channel_id, channel_id, ops["identity"])
                    )
                transitions[j].append((channel_id, done, number))
            for t_sigma, create, annihilate, create_charge, spin in mode_terms:
                if t_sigma == 0:
                    continue
                for direction, first, second, first_charge in (
                    ("forward", create, annihilate, create_charge),
                    ("backward", annihilate, create, _neg_charge(create_charge)),
                ):
                    channel_id = ("hop", edge_pos, spin, direction)
                    channel_charge = _neg_charge(first_charge)
                    # Site-major JW convention: for i < j the string spans
                    # i <= l < j. Thus c_i^dag c_j has c_i^dag P_i on the
                    # left endpoint, while its Hermitian conjugate has P_i c_i.
                    endpoint = first @ parity if direction == "forward" else parity @ first
                    first_op = -t_sigma * endpoint
                    for cut in range(i, j):
                        channels[cut].append((channel_id, channel_charge))
                    transitions[i].append((start, channel_id, first_op))
                    for site in range(i + 1, j):
                        transitions[site].append(
                            (channel_id, channel_id, parity)
                        )
                    transitions[j].append((channel_id, done, second))

        return _assemble_symmray_mpo(
            L=L,
            channels=channels,
            transitions=transitions,
            phys_map=phys_map,
            symmetry=self.symmetry,
            zero=zero,
            dtype=dtype,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            site_tag_id=site_tag_id,
            to_backend=to_backend,
        )

    def to_pepo(
        self,
        Lx=None,
        Ly=None,
        *,
        mapper=None,
        max_bond=None,
        cutoff=1e-12,
        compress=True,
        cyclic=False,
        cycle_bond_dim=1,
        dtype=None,
        fermionic=True,
        to_backend=None,
        charge_sectors=False,
    ):
        """Build a 2D PEPO from this Hamiltonian's native local terms.

        The Hamiltonian is first assembled as an MPO using ``mapper`` and is
        then embedded with the same snake-style ordering into a PEPO. Native
        ``fermionic=True`` construction preserves homogeneous neutral or
        nonzero operator charge and Symmray grading metadata. Set
        ``fermionic=False`` to request the compatibility MPO path.
        With ``charge_sectors=True``, return ``{charge: PEPO}`` for mixed
        charge collections.
        """
        if Lx is None or Ly is None:
            raise TypeError("to_pepo requires both Lx and Ly.")

        from ..operators.hamiltonians import ham_tn

        builder = ham_tn(
            Lx=Lx,
            Ly=Ly,
            mapper=mapper,
            max_bond=256 if max_bond is None else max_bond,
            cutoff=cutoff,
            data_type=(
                _dtype_from_hamiltonian_terms(self.terms)
                if dtype is None
                else dtype
            ),
        )

        # A single local term does not need the generic start/done channel
        # construction used to combine a Hamiltonian.  Build its native MPO
        # directly from the local operator Schmidt factorization instead.  In
        # particular, a hopping term then has its physical rank (D=4 for the
        # spinful U1 hopping operator) rather than the inflated collection
        # channel count of the multi-term assembler. Charged terms keep the
        # generic native route because their open boundary must carry the
        # operator charge through the remaining chain.
        if fermionic and len(self.terms) == 1:
            raw_where, term = next(iter(self.terms.items()))
            coordinate_sites = _term_mapping_uses_coordinate_sites(self.terms)
            where = _as_term_where(
                raw_where,
                coordinate_sites=coordinate_sites,
            )
            zero = _zero_like_charge(
                0 if self.symmetry in {"U1", "Z2"} else (0, 0)
            )
            term_charge = _normalize_group_charge(
                getattr(term, "charge", zero),
                self.symmetry,
            )
            if len(where) in {1, 2} and term_charge == zero:
                _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
                    mapper=builder.mapper,
                )
                if mapped_L != builder.L:
                    raise ValueError(
                        f"MPO mapping length {mapped_L} does not match PEPO length "
                        f"{builder.L}."
                    )
                mapped_where = tuple(
                    _map_site_to_mpo_index(site, coo2idx_use)
                    for site in where
                )
                dtype_use = (
                    _dtype_from_hamiltonian_terms(self.terms)
                    if dtype is None
                    else np.dtype(dtype)
                )
                mpo = _native_local_term_mpo(
                    term,
                    mapped_where,
                    builder.L,
                    symmetry=self.symmetry,
                    dtype=dtype_use,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    compress=compress,
                    to_backend=to_backend,
                )
                pepo = builder.mpo_to_pepo(
                    mpo,
                    cycle_peps=cyclic,
                    cycle_bond_dim=cycle_bond_dim,
                    inplace=True,
                )
                # Keep the diagnostic on the returned PEPO after the MPO is
                # relabelled and viewed as a PEPO.
                pepo.pepsy_compression_report = dict(
                    mpo.pepsy_compression_report
                )
                if charge_sectors:
                    charge = _normalize_group_charge(
                        getattr(term, "charge", 0),
                        self.symmetry,
                    )
                    return {charge: pepo}
                return pepo

        mpo = self.to_mpo(
            L=builder.L,
            mapper=builder.mapper,
            max_bond=max_bond,
            cutoff=cutoff,
            compress=compress,
            dtype=dtype,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            to_backend=to_backend,
        )
        if charge_sectors:
            return {
                charge: builder.mpo_to_pepo(
                    sector_mpo,
                    cycle_peps=cyclic,
                    cycle_bond_dim=cycle_bond_dim,
                    inplace=True,
                )
                for charge, sector_mpo in mpo.items()
            }
        return builder.mpo_to_pepo(
            mpo,
            cycle_peps=cyclic,
            cycle_bond_dim=cycle_bond_dim,
            inplace=True,
        )

    def jw_trotter_gates(
        self,
        dt,
        *,
        mapper=None,
        idx2coo=None,
        coo2idx=None,
        order=2,
        imaginary=False,
        peierls_angle=0.0,
        dtype=None,
        to_backend=None,
    ):
        """Return a bosonic Jordan-Wigner Trotter gate stream consistent with ``to_mpo``.

        This is the gate-based counterpart of :meth:`to_mpo` for the
        ``model="fermi_hubbard_u1u1"`` Jordan-Wigner (bosonic) picture. It reads
        the *same* Jordan-Wigner conversion the MPO path uses -- the site
        ordering (from ``mapper``/``idx2coo``/``coo2idx``), the one-site
        operators, and the parity-string convention -- and the model parameters
        (``t``, ``U``, ``mu``) from :attr:`parameters`. The returned gate stream
        therefore agrees with the MPO by construction, so an energy computed from
        :meth:`to_mpo` and a time evolution driven by these gates use one and the
        same Jordan-Wigner conversion.

        Only bonds that map to **nearest-neighbour** chain sites are supported: a
        bond whose mapped endpoints are non-adjacent has a Jordan-Wigner string
        that spans the intervening sites and is not a two-site gate. Such a bond
        raises; reorder the sites (choose a ``mapper``) so every bond is
        nearest-neighbour, or use :meth:`to_mpo` for the long-range path.

        The static :meth:`to_mpo` Hamiltonian carries no Peierls phase, so
        ``peierls_angle`` (for real-time driven hopping) defaults to ``0``.
        """
        edges, sites, params = self._resolve_jw_fermi_hubbard(
            mapper=mapper, idx2coo=idx2coo, coo2idx=coo2idx
        )
        if order not in {1, 2, 4}:
            raise ValueError("order must be 1, 2, or 4.")
        if order == 4:
            return _yoshida4_stream(
                lambda sub_dt: self.jw_trotter_gates(
                    sub_dt,
                    mapper=mapper,
                    idx2coo=idx2coo,
                    coo2idx=coo2idx,
                    order=2,
                    imaginary=imaginary,
                    peierls_angle=peierls_angle,
                    dtype=dtype,
                    to_backend=to_backend,
                ),
                dt,
                imaginary=imaginary,
                hamiltonian=self,
            )
        dtype = "complex128" if dtype is None else np.dtype(dtype)
        return fermi_hubbard_u1u1_jw_gate_stream(
            edges,
            dt,
            sites=sites,
            t=params["t"],
            U=params["U"],
            mu=params["mu"],
            peierls_angle=peierls_angle,
            imaginary=imaginary,
            order=order,
            dtype=dtype,
            to_backend=to_backend,
        )

    def _resolve_jw_fermi_hubbard(self, *, mapper=None, idx2coo=None, coo2idx=None):
        """Resolve the shared U1U1 Fermi-Hubbard Jordan-Wigner conversion.

        Returns ``(edges, sites, params)`` where ``edges`` are the mapped
        nearest-neighbour bonds, ``sites`` covers every chain site, and
        ``params`` is ``{"t", "U", "mu"}``. Both :meth:`jw_trotter_gates` and
        :meth:`jw_energy` use this, so the evolution gates and the measured
        energy share one conversion. Validates the model/symmetry, rejects the
        unsupported ``V`` term, and rejects bonds that map to non-adjacent chain
        sites.
        """
        if self.model != "fermi_hubbard_u1u1" or self.symmetry != "U1U1":
            raise NotImplementedError(
                "Jordan-Wigner gate/energy paths require "
                "model='fermi_hubbard_u1u1' with U1U1 symmetry; got "
                f"model={self.model!r}, symmetry={self.symmetry!r}. Use to_mpo "
                "for the MPO path or trotter_gates for native fermionic gates."
            )
        V = self.parameters.get("V", 0.0)
        if (
            callable(V)
            or isinstance(V, Mapping)
            or not np.all(np.asarray(V, dtype=complex) == 0)
        ):
            raise NotImplementedError(
                "Jordan-Wigner gate/energy paths do not yet support the "
                "density-density 'V' term; use "
                "to_mpo(model='fermi_hubbard_u1u1'), or set V=0."
            )
        _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
            mapper=mapper, idx2coo=idx2coo, coo2idx=coo2idx
        )
        raw_edges = _as_edges(self.edges)
        edges = _map_edges_to_mpo_indices(raw_edges, coo2idx_use)
        long_range = [
            (int(i), int(j)) for i, j in edges if abs(int(i) - int(j)) != 1
        ]
        if long_range:
            raise ValueError(
                "Jordan-Wigner two-site gates/terms need nearest-neighbour "
                f"bonds; {len(long_range)} bond(s) map to non-adjacent chain "
                f"sites under this ordering (e.g. {long_range[:3]}). Reorder "
                "sites via mapper=OneDMap(...) so every bond is "
                "nearest-neighbour, or use to_mpo(model='fermi_hubbard_u1u1') "
                "for the long-range Jordan-Wigner path."
            )
        if mapped_L is not None:
            sites = tuple(range(int(mapped_L)))
        else:
            sites = tuple(sorted({int(s) for edge in edges for s in edge}))
        params = {
            "t": self.parameters.get("t", 1.0),
            "U": self.parameters.get("U", 8.0),
            "mu": self.parameters.get("mu", 0.0),
        }
        return edges, sites, params

    def jw_energy(
        self,
        state,
        *,
        mapper=None,
        idx2coo=None,
        coo2idx=None,
        normalize=True,
        dtype=None,
    ):
        """Return the Jordan-Wigner energy of a bosonic state.

        Sums the local Jordan-Wigner term expectations -- onsite
        ``U n_up n_down - mu n`` on every site and nearest-neighbour hopping on
        every bond -- built from the *same* conversion as :meth:`to_mpo` and
        :meth:`jw_trotter_gates`. It therefore reads out the energy of a bosonic
        (``fermionic=False``) state evolved by :meth:`jw_trotter_gates`, using
        the state's own symmetry-aware :meth:`SymMPS.measure` contraction.

        With ``normalize=True`` (default) the returned value is
        ``<psi|H|psi> / <psi|psi>``. Nearest-neighbour bonds only.
        """
        from .symmetric_states import _SymState

        if not isinstance(state, _SymState):
            raise TypeError(
                "jw_energy expects a bosonic SymMPS/SymPEPS with a "
                "symmetry-aware .measure; got "
                f"{type(state).__name__}. Wrap a raw MPS via "
                "SymMPS(mps=..., symmetry='U1U1', edges=..., fermionic=False), "
                "or read a DMRG energy from SymDMRG2.energy."
            )
        edges, sites, params = self._resolve_jw_fermi_hubbard(
            mapper=mapper, idx2coo=idx2coo, coo2idx=coo2idx
        )
        dtype = "complex128" if dtype is None else np.dtype(dtype)
        onsite = _fh_u1u1_jw_onsite_term(
            U=params["U"], mu=params["mu"], dtype=dtype
        )
        hopping = _fh_u1u1_jw_hopping_term(t=params["t"], dtype=dtype)
        total = 0.0 + 0.0j
        for site in sites:
            total += complex(
                state.measure(onsite, int(site), normalize=normalize)
            )
        for i, j in edges:
            lo, hi = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            total += complex(
                state.measure(hopping, (lo, hi), normalize=normalize)
            )
        return total.real if abs(total.imag) < 1e-9 else total

    def jw_bond_layout(self, *, mapper=None, idx2coo=None, coo2idx=None):
        """Classify Fermi-Hubbard bonds by Jordan-Wigner locality under an ordering.

        Returns ``{"adjacent": [...], "long_range": [...], "sites": [...]}``: the
        mapped bonds that are nearest-neighbour (usable as two-site gates by
        :meth:`jw_trotter_gates`) versus those whose Jordan-Wigner string spans
        intervening sites (currently reachable only through :meth:`to_mpo`). Use
        it to choose a site ordering (``mapper``) that maximizes the number of
        nearest-neighbour bonds. Unlike :meth:`jw_trotter_gates`, this does not
        raise on long-range bonds -- it reports them.
        """
        if self.model != "fermi_hubbard_u1u1" or self.symmetry != "U1U1":
            raise NotImplementedError(
                "jw_bond_layout requires model='fermi_hubbard_u1u1' with U1U1 "
                f"symmetry; got model={self.model!r}, symmetry={self.symmetry!r}."
            )
        _, coo2idx_use, mapped_L = _resolve_mpo_mapping(
            mapper=mapper, idx2coo=idx2coo, coo2idx=coo2idx
        )
        raw_edges = _as_edges(self.edges)
        edges = _map_edges_to_mpo_indices(raw_edges, coo2idx_use)
        adjacent = []
        long_range = []
        for i, j in edges:
            lo, hi = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            (adjacent if hi - lo == 1 else long_range).append((lo, hi))
        if mapped_L is not None:
            sites = list(range(int(mapped_L)))
        else:
            sites = sorted({int(s) for edge in edges for s in edge})
        return {"adjacent": adjacent, "long_range": long_range, "sites": sites}

    def trotter_gates(self, dt, *, imaginary=False, order=1):
        """Return local gate entries ``[(gate, edge), ...]`` for one Trotter step."""
        if order not in {1, 2, 4}:
            raise ValueError("order must be 1, 2, or 4.")
        if order == 4:
            return _yoshida4_stream(
                lambda sub_dt: self.trotter_gates(
                    sub_dt,
                    imaginary=imaginary,
                    order=2,
                ),
                dt,
                imaginary=imaginary,
                hamiltonian=self,
            )
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


# Compatibility spelling for the initial, overly model-specific public name.
