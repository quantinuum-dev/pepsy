"""Symmray-backed symmetric MPS and PEPS convenience wrappers."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
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

_SYMMRAY_AUTORAY_REGISTERED = False

_FERMIONIC_TN_METHODS_REFERENCE = {
    "title": "Fermionic tensor network contraction for arbitrary geometries",
    "doi": "10.1103/PhysRevResearch.7.023193",
    "url": "https://doi.org/10.1103/PhysRevResearch.7.023193",
}


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


def _as_python_bool(value):
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return bool(value)


def _to_host_numpy(value):
    value = _to_dense(value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


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


def _require_symmray_array(value, *, name="value"):
    if _is_symmray_array(getattr(value, "data", None)):
        return value.data
    if not _is_symmray_array(value):
        raise TypeError(f"{name} must be a Symmray block-sparse array.")
    return value


def _mapping_items(mapping):
    return sorted(mapping.items(), key=lambda item: repr(item[0]))


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _format_compact_mapping(mapping, *, max_items=4):
    items = _mapping_items(mapping)
    pieces = [f"{_format_charge(key)}:{val}" for key, val in items[:max_items]]
    if len(items) > max_items:
        pieces.append("...")
    return "{" + ", ".join(pieces) + "}"


def _format_charge(charge):
    if isinstance(charge, tuple):
        return "(" + ", ".join(str(x) for x in charge) + ")"
    return str(charge)


def _format_sector(sector):
    sector = _as_tuple(sector)
    return "(" + ", ".join(_format_charge(x) for x in sector) + ")"


def _format_shape(shape):
    return "x".join(str(int(dim)) for dim in shape) or "scalar"


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


def _format_signed_half_integer(twice_value):
    twice_value = int(twice_value)
    if twice_value == 0:
        return "0"
    if twice_value % 2 == 0:
        value = twice_value // 2
        return f"+{value}" if value > 0 else str(value)
    sign = "+" if twice_value > 0 else "-"
    return f"{sign}{abs(twice_value)}/2"


def _charge_spin_label_lines(charge):
    """Return compact in-node charge/spin labels for spin-resolved charges."""
    if not (isinstance(charge, tuple) and len(charge) == 2):
        return None
    try:
        n_up, n_down = (int(charge[0]), int(charge[1]))
    except (TypeError, ValueError):
        return None
    charge_total = n_up + n_down
    spin_twice = n_up - n_down
    return [
        rf"$Q={charge_total}$",
        rf"$S_z={_format_signed_half_integer(spin_twice)}$",
    ]


def _node_charge_label_lines(charge):
    if charge is None:
        return []
    spin_lines = _charge_spin_label_lines(charge)
    if spin_lines is not None:
        return spin_lines
    node_lines = [rf"$q={_format_charge(charge)}$"]
    particle = _charge_particle_number(charge)
    if particle is not None:
        node_lines.append(rf"$N={particle}$")
    return node_lines


def _add_charges(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        a_t = a if isinstance(a, tuple) else (a,) * len(b)
        b_t = b if isinstance(b, tuple) else (b,) * len(a)
        return tuple(x + y for x, y in zip(a_t, b_t))
    return a + b


def _sum_charges(charges):
    charges = [charge for charge in charges if charge is not None]
    if not charges:
        return None
    total = charges[0]
    for charge in charges[1:]:
        total = _add_charges(total, charge)
    return total


def _mod_charge(charge, mod):
    if charge is None:
        return None
    if isinstance(charge, tuple):
        return tuple(int(x) % mod for x in charge)
    return int(charge) % mod


def _resolve_total_charge(source, tensors):
    if hasattr(source, "overall_charge"):
        try:
            return source.overall_charge()
        except Exception:  # pragma: no cover - summary should stay best-effort
            pass
    return _sum_charges(tensor.get("charge") for tensor in tensors)


def _resolve_q_total(symmetry, total_charge):
    if total_charge is None:
        return None
    if str(symmetry).upper() == "Z2":
        return _mod_charge(total_charge, 2)
    return total_charge


def _charge_summary_text(summary):
    pieces = []
    symmetry = summary.get("symmetry")
    if symmetry is not None:
        pieces.append(f"sym={symmetry}")
    if summary.get("fermionic"):
        pieces.append("fermionic")
    total_charge = summary.get("charge_total", summary.get("total_charge"))
    if total_charge is not None:
        pieces.append(f"charge_total={_format_charge(total_charge)}")
        q_total = summary.get("Q_total")
        if q_total is None:
            q_total = _resolve_q_total(symmetry, total_charge)
        if str(symmetry).upper() == "Z2":
            pieces.append(f"Q_total={_format_charge(q_total)} mod 2")
        else:
            pieces.append(f"Q_total={_format_charge(q_total)}")
    return " | ".join(pieces)


def _flow_text(*directions):
    return "->".join(str(direction) for direction in directions if direction is not None)


def _flow_math(*directions):
    parts = [str(direction) for direction in directions if direction is not None]
    if not parts:
        return ""
    return r"\to".join(rf"\mathrm{{{part}}}" for part in parts)


def _shape_size(shape):
    return int(np.prod(tuple(shape), dtype=int)) if shape else 1


def _block_size(block, shape):
    if hasattr(block, "numel"):
        return int(block.numel())
    size = getattr(block, "size", None)
    if callable(size):
        try:
            size = size()
        except TypeError:
            size = None
    if size is not None and not isinstance(size, tuple):
        try:
            return int(size)
        except TypeError:
            pass
    return _shape_size(shape)


def _summarize_symmray_index(index, axis):
    chargemap = {
        charge: int(size)
        for charge, size in _mapping_items(dict(getattr(index, "chargemap", {})))
    }
    dual = bool(getattr(index, "dual", False))
    return {
        "axis": int(axis),
        "dual": dual,
        "direction": "in" if dual else "out",
        "chargemap": chargemap,
        "sectors": [
            {"charge": charge, "size": size}
            for charge, size in _mapping_items(chargemap)
        ],
        "dim": int(sum(chargemap.values())),
        "num_sectors": len(chargemap),
    }


def symmray_block_summary(array):
    """Return leg and block metadata for a Symmray block-sparse array.

    Parameters
    ----------
    array : symmray array or quimb.Tensor
        Block-sparse array exposing ``indices`` and ``get_sector_block_pairs``.

    Returns
    -------
    dict
        Summary with ``shape``, optional ``charge``, per-leg ``indices``,
        present block-sector records under ``blocks``, and dense/stored size
        counts useful for diagnostics.
    """
    array = _require_symmray_array(array, name="array")

    indices = [
        _summarize_symmray_index(index, axis)
        for axis, index in enumerate(getattr(array, "indices", ()))
    ]

    blocks = []
    for sector, block in array.get_sector_block_pairs():
        shape = tuple(int(dim) for dim in getattr(block, "shape", ()))
        size = _block_size(block, shape)
        blocks.append(
            {
                "sector": _as_tuple(sector),
                "shape": shape,
                "size": size,
                "dtype": str(getattr(block, "dtype", "")),
            }
        )

    shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
    dense_size = _shape_size(shape)
    stored_size = int(sum(block["size"] for block in blocks))
    return {
        "shape": shape,
        "charge": getattr(array, "charge", None),
        "indices": indices,
        "blocks": blocks,
        "num_blocks": len(blocks),
        "dense_size": dense_size,
        "stored_size": stored_size,
        "density": stored_size / dense_size if dense_size else 0.0,
    }


def _as_mps_tensor_network(value):
    if hasattr(value, "tn"):
        value = value.tn
    elif hasattr(value, "p"):
        value = value.p
    if not hasattr(value, "sites") or not hasattr(value, "site_ind"):
        raise TypeError("mps must be a SymMPS, MpsOptimizer, or quimb MPS object.")
    return value


def _as_mpo_tensor_network(value):
    if hasattr(value, "mpo"):
        value = value.mpo
    elif hasattr(value, "H_mpo"):
        value = value.H_mpo
    if not hasattr(value, "sites"):
        raise TypeError("mpo must be a quimb MatrixProductOperator object.")
    if not hasattr(value, "upper_ind") or not hasattr(value, "lower_ind"):
        raise TypeError("mpo must expose upper_ind(site) and lower_ind(site).")
    return value


def _is_mpo_like(value):
    try:
        _as_mpo_tensor_network(value)
    except TypeError:
        return False
    return True


def _is_mps_like_not_peps(value):
    try:
        tn = _as_mps_tensor_network(value)
    except TypeError:
        return False
    if hasattr(tn, "upper_ind") and hasattr(tn, "lower_ind"):
        return False
    return not (hasattr(tn, "Lx") and hasattr(tn, "Ly"))


def _mps_sites(tn):
    sites = tuple(getattr(tn, "sites", ()))
    if not sites and hasattr(tn, "gen_sites_present"):
        sites = tuple(tn.gen_sites_present())
    if not sites:
        raise ValueError("mps does not expose any site labels.")
    return sites


def _mps_site_tensor(tn, site):
    try:
        return tn[site]
    except Exception as exc:  # pragma: no cover - defensive for quimb variants
        if hasattr(tn, "site_tag"):
            try:
                return tn[tn.site_tag(site)]
            except Exception:
                pass
        raise ValueError(f"Could not resolve MPS tensor for site {site!r}.") from exc


def _shared_virtual_ind(tensor_a, tensor_b, physical_inds):
    physical_inds = set(physical_inds)
    for ind in tensor_a.inds:
        if ind in tensor_b.inds and ind not in physical_inds:
            return ind
    return None


def _shared_virtual_inds(tensor_a, tensor_b, physical_inds):
    physical_inds = set(physical_inds)
    return tuple(
        ind
        for ind in tensor_a.inds
        if ind in tensor_b.inds and ind not in physical_inds
    )


def _is_fermionic_array_data(data):
    if data is None:
        return False
    if bool(getattr(data, "fermionic", False)):
        return True
    return "fermionic" in type(data).__name__.lower()


def _infer_fermionic(source, tensors):
    fermionic = getattr(source, "fermionic", None)
    if fermionic is not None:
        return bool(fermionic)
    return any(_is_fermionic_array_data(getattr(tensor, "data", None)) for tensor in tensors)


def _infer_symmetry(source, tensors):
    symmetry = getattr(source, "symmetry", None)
    if symmetry is not None:
        return symmetry
    for tensor in tensors:
        symmetry = getattr(getattr(tensor, "data", None), "symmetry", None)
        if symmetry is not None:
            return symmetry
    return None


def _source_edges(source, bonds):
    edges = getattr(source, "edges", None)
    if edges is not None:
        return _as_edges(edges)
    return tuple(tuple(bond["between"]) for bond in bonds)


def _explicit_source_edges(source):
    edges = getattr(source, "edges", None)
    if edges is None:
        return None
    return _as_edges(edges)


def _edge_lookup(edges):
    order_by_pair = {}
    edge_by_pair = {}
    for position, edge in enumerate(edges):
        edge = tuple(edge)
        reverse_edge = tuple(reversed(edge))
        order_by_pair.setdefault(edge, int(position))
        order_by_pair.setdefault(reverse_edge, int(position))
        edge_by_pair.setdefault(edge, edge)
        edge_by_pair.setdefault(reverse_edge, edge)
    return order_by_pair, edge_by_pair


def _bond_endpoint_directions(bond):
    if "left_site" in bond:
        return {
            bond["left_site"]: bond.get("left_direction"),
            bond["right_site"]: bond.get("right_direction"),
        }
    return {
        bond["site_a"]: bond.get("site_a_direction"),
        bond["site_b"]: bond.get("site_b_direction"),
    }


def _fermionic_edge_record(bond, order_by_pair, edge_by_pair):
    between = tuple(bond["between"])
    edge = edge_by_pair.get(between, between)
    directions = _bond_endpoint_directions(bond)
    return {
        "position": int(bond["position"]),
        "edge_order": order_by_pair.get(between),
        "edge": edge,
        "between": between,
        "ind": bond["ind"],
        "lattice_direction": bond.get("direction"),
        "index_directions": tuple(
            {"site": site, "direction": directions.get(site)}
            for site in edge
        ),
    }


def _fermionic_ordering_summary(source, *, network_kind, sites, bonds, fermionic):
    edges = _source_edges(source, bonds)
    order_by_pair, edge_by_pair = _edge_lookup(edges)
    return {
        "enabled": bool(fermionic),
        "network_kind": network_kind,
        "methods_reference": dict(_FERMIONIC_TN_METHODS_REFERENCE),
        "site_order": tuple(sites),
        "edge_order": edges,
        "edges": tuple(
            _fermionic_edge_record(bond, order_by_pair, edge_by_pair)
            for bond in bonds
        ),
    }


def _directions_are_complementary(direction_a, direction_b):
    return {direction_a, direction_b} == {"in", "out"}


def _resolve_mps_position(tensors, value, *, name="position"):
    if value is None:
        return None
    if value == "middle":
        return tensors[len(tensors) // 2]["position"] if tensors else None
    for tensor in tensors:
        if value == tensor["position"] or value == tensor["site"]:
            return tensor["position"]
    raise ValueError(f"{name}={value!r} does not identify a shown MPS site.")


def _resolve_chain_mapper(mapper, summary, *, name="mapper"):
    if mapper is None:
        return None

    from .core import OneDMap  # pylint: disable=import-outside-toplevel

    if not isinstance(mapper, OneDMap):
        raise TypeError(f"{name} must be a pepsy.tensors.OneDMap instance.")
    if mapper.Lz is not None:
        raise NotImplementedError(
            f"{name} plotting is currently only available for 2D OneDMap instances."
        )

    idx2coo, _ = mapper.build()
    if len(idx2coo) != summary["num_sites"]:
        raise ValueError(
            f"{name} length {len(idx2coo)} does not match network length "
            f"{summary['num_sites']}."
        )

    coords = {}
    for position in range(summary["num_sites"]):
        try:
            coord = idx2coo[position]
        except KeyError as exc:
            raise ValueError(f"{name} is missing chain position {position}.") from exc
        if len(coord) != 2:
            raise NotImplementedError(
                f"{name} plotting is currently only available for 2D coordinates."
            )
        coords[int(position)] = (int(coord[0]), int(coord[1]))
    return coords


def _mapped_site_color(colormaps, cmap_name, position, num_sites):
    cmap_key = str(cmap_name).strip().lower()
    if cmap_key in {"auto", "quimb", "quimb-green", "green"}:
        from quimb import schematic  # pylint: disable=import-outside-toplevel

        return _lighten_rgba(schematic.get_color("green"), amount=0.52)
    if cmap_key == "hash":
        from quimb import schematic  # pylint: disable=import-outside-toplevel

        return _lighten_rgba(schematic.hash_to_color(f"I{int(position)}"), amount=0.46)
    cmap = colormaps.get_cmap(cmap_name)
    colors = getattr(cmap, "colors", None)
    if colors is not None and len(colors) > 0:
        return cmap(int(position) % len(colors))
    return cmap(int(position) / max(1, int(num_sites) - 1))


def _lighten_rgba(color, *, amount=0.45):
    from matplotlib.colors import to_rgba  # pylint: disable=import-outside-toplevel

    red, green, blue, alpha = to_rgba(color)
    amount = min(max(float(amount), 0.0), 1.0)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
        alpha,
    )


def _mapped_contrast_text_color(color):
    from matplotlib.colors import to_rgba  # pylint: disable=import-outside-toplevel

    red, green, blue, _ = to_rgba(color)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    if luminance < 0.54:
        return (1.0, 1.0, 1.0, 0.96)
    return (0.08, 0.10, 0.13, 0.96)


def _format_half_integer(twice_value):
    try:
        twice_value = int(twice_value)
    except (TypeError, ValueError):
        return str(twice_value)
    if twice_value == 0:
        return "0"
    sign = "+" if twice_value > 0 else "-"
    numerator = abs(twice_value)
    if numerator % 2 == 0:
        return f"{sign}{numerator // 2}"
    return f"{sign}{numerator}/2"


def _mapped_charge_spin_lines(charge):
    if charge is None:
        return []
    charge_tuple = _as_tuple(charge)
    if len(charge_tuple) == 2:
        try:
            n_up, n_down = (int(x) for x in charge_tuple)
        except (TypeError, ValueError):
            return [rf"$q={_format_charge(charge)}$"]
        return [
            rf"$N={n_up + n_down}$",
            rf"$S_z={_format_half_integer(n_up - n_down)}$",
        ]
    return [rf"$q={_format_charge(charge)}$"]


def _mapped_tensor_label_lines(tensor, *, kind):
    _ = kind
    return _mapped_charge_spin_lines(tensor.get("charge"))


def _mapped_bond_label(bond, *, show_leg_chargemaps):
    pieces = [rf"$e_{{{bond['position']}}}$", rf"$\chi={bond['dim']}$"]
    if show_leg_chargemaps:
        pieces.append(rf"$Q={len(bond['chargemap'])}$")
    return " | ".join(pieces)


def _mapped_physical_label(prefix, tensor, physical, *, show_leg_chargemaps):
    if show_leg_chargemaps:
        return (
            rf"${prefix}_{{{tensor['site']}}}$ | "
            rf"$d/Q={physical['dim']}/{len(physical['chargemap'])}$"
        )
    return rf"${prefix}_{{{tensor['site']}}}$ | $d={physical['dim']}$"


def _mapped_label_offset(x0, y0, x1, y1, amount=0.17):
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    norm = float(np.hypot(dx, dy))
    if norm == 0.0:
        return (0.0, amount)
    return (-amount * dy / norm, amount * dx / norm)


def _draw_mapped_chain_grid(drawing, mapper, *, spacing):
    Lx = int(mapper.Lx)
    Ly = int(mapper.Ly)
    for x in range(Lx):
        for y in range(Ly):
            xy = (x * spacing, y * spacing)
            if x + 1 < Lx:
                drawing.line(xy, ((x + 1) * spacing, y * spacing), preset="lattice", zorder=0)
            if y + 1 < Ly:
                drawing.line(xy, (x * spacing, (y + 1) * spacing), preset="lattice", zorder=0)
            if mapper.mode == "diag" and x + 1 < Lx and y + 1 < Ly:
                drawing.line(
                    xy,
                    ((x + 1) * spacing, (y + 1) * spacing),
                    preset="lattice",
                    zorder=0,
                )


def _mapped_chain_limits(mapper, xy_by_position, *, spacing, right_pad, y_pad, left_pad=0.82):
    xs = [xy[0] for xy in xy_by_position.values()] or [0.0]
    ys = [xy[1] for xy in xy_by_position.values()] or [0.0]
    grid_xs = [0.0, (int(mapper.Lx) - 1) * spacing]
    grid_ys = [0.0, (int(mapper.Ly) - 1) * spacing]
    x_min = min(xs + grid_xs) - left_pad
    x_max = max(xs + grid_xs) + right_pad
    y_min = min(ys + grid_ys) - y_pad
    y_max = max(ys + grid_ys) + y_pad
    return x_min, x_max, y_min, y_max


def _draw_mapped_chain_diagnostics(
    drawing,
    summary,
    shown_tensors,
    *,
    show_blocks,
    show_arrows,
    x,
    y,
):
    charge_line = _charge_summary_text(summary)
    diagnostic_lines = [f"sites {summary['num_sites']}"]
    if charge_line:
        diagnostic_lines.append(charge_line)
    diagnostic_lines += [
        f"max bond {summary['max_bond_dim']}",
        f"bond sectors {summary['max_bond_sectors']}",
        f"stored {summary['total_stored_size']}/{summary['total_dense_size']}",
        f"density {summary['density']:.3f}",
    ]
    diagnostic = "\n".join(diagnostic_lines)
    if show_blocks:
        diagnostic += "\ncolored tiles: stored blocks"
    if show_arrows:
        diagnostic += "\narrows/labels: charge in/out flow"
    if len(shown_tensors) < summary["num_sites"]:
        diagnostic += f"\n+{summary['num_sites'] - len(shown_tensors)} sites hidden"
    drawing.ax.text(
        x,
        y,
        diagnostic,
        fontsize=8,
        ha="left",
        va="top",
        color=(0.15, 0.17, 0.20, 1.0),
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": (1.0, 1.0, 1.0, 0.94),
            "edgecolor": (0.68, 0.70, 0.74, 1.0),
            "linewidth": 0.8,
        },
        zorder=10,
    )


def symmray_mps_summary(mps):
    """Return site, bond, and block metadata for a Symmray-backed MPS.

    Parameters
    ----------
    mps : SymMPS, MpsOptimizer, or quimb MatrixProductState
        Object whose site tensors store Symmray block-sparse arrays.

    Returns
    -------
    dict
        Summary with per-site tensor block counts, physical-sector maps,
        nearest-neighbor bond-sector maps, and aggregate storage diagnostics.
    """
    source = mps
    tn = _as_mps_tensor_network(mps)
    sites = _mps_sites(tn)

    tensors = []
    site_tensors = []
    index_maps = []
    physical_inds = []

    for position, site in enumerate(sites):
        tensor = _mps_site_tensor(tn, site)
        array_summary = symmray_block_summary(tensor)
        site_ind = tn.site_ind(site)

        index_by_ind = {}
        indices = []
        for ind, index_summary in zip(tensor.inds, array_summary["indices"]):
            entry = dict(index_summary)
            entry["ind"] = ind
            index_by_ind[ind] = entry
            indices.append(entry)

        physical = index_by_ind.get(site_ind)
        if physical is None:
            raise ValueError(
                f"MPS tensor for site {site!r} does not expose physical index {site_ind!r}."
            )

        tensors.append(
            {
                "position": int(position),
                "site": site,
                "site_tag": tn.site_tag(site) if hasattr(tn, "site_tag") else None,
                "site_ind": site_ind,
                "inds": tuple(tensor.inds),
                "shape": array_summary["shape"],
                "charge": array_summary["charge"],
                "indices": indices,
                "physical": physical,
                "left_bond": None,
                "right_bond": None,
                "blocks": array_summary["blocks"],
                "num_blocks": array_summary["num_blocks"],
                "dense_size": array_summary["dense_size"],
                "stored_size": array_summary["stored_size"],
                "density": array_summary["density"],
            }
        )
        site_tensors.append(tensor)
        index_maps.append(index_by_ind)
        physical_inds.append(site_ind)

    bonds = []
    for position, (site_l, site_r) in enumerate(zip(sites[:-1], sites[1:])):
        ind = _shared_virtual_ind(
            site_tensors[position],
            site_tensors[position + 1],
            (physical_inds[position], physical_inds[position + 1]),
        )
        if ind is None:
            continue
        left_index = index_maps[position].get(ind)
        right_index = index_maps[position + 1].get(ind)
        index_summary = left_index or right_index
        bond = {
            "position": int(position),
            "left_position": int(position),
            "right_position": int(position + 1),
            "left_site": site_l,
            "right_site": site_r,
            "between": (site_l, site_r),
            "ind": ind,
            "left_direction": left_index["direction"] if left_index is not None else None,
            "right_direction": right_index["direction"] if right_index is not None else None,
            "dim": index_summary["dim"],
            "num_sectors": index_summary["num_sectors"],
            "chargemap": index_summary["chargemap"],
            "sectors": index_summary["sectors"],
        }
        bonds.append(bond)
        tensors[position]["right_bond"] = bond
        tensors[position + 1]["left_bond"] = bond

    total_dense_size = int(sum(tensor["dense_size"] for tensor in tensors))
    total_stored_size = int(sum(tensor["stored_size"] for tensor in tensors))
    total_charge = _resolve_total_charge(source, tensors)
    symmetry = _infer_symmetry(source, site_tensors)
    fermionic = _infer_fermionic(source, site_tensors)
    q_total = _resolve_q_total(symmetry, total_charge)
    return {
        "num_sites": len(sites),
        "sites": sites,
        "tensors": tensors,
        "bonds": bonds,
        "symmetry": symmetry,
        "fermionic": fermionic,
        "fermionic_ordering": _fermionic_ordering_summary(
            source,
            network_kind="mps",
            sites=sites,
            bonds=bonds,
            fermionic=fermionic,
        ),
        "total_charge": total_charge,
        "charge_total": total_charge,
        "Q_total": q_total,
        "total_parity": _mod_charge(total_charge, 2),
        "max_bond_dim": max((bond["dim"] for bond in bonds), default=1),
        "max_bond_sectors": max((bond["num_sectors"] for bond in bonds), default=0),
        "total_dense_size": total_dense_size,
        "total_stored_size": total_stored_size,
        "density": total_stored_size / total_dense_size if total_dense_size else 0.0,
    }


def symmray_mpo_summary(mpo):
    """Return site, bond, and block metadata for a Symmray-backed MPO.

    Parameters
    ----------
    mpo : quimb MatrixProductOperator
        Operator whose site tensors store Symmray block-sparse arrays.

    Returns
    -------
    dict
        Summary with per-site tensor block counts, upper/lower physical-sector
        maps, nearest-neighbor bond-sector maps, and aggregate storage
        diagnostics.
    """
    source = mpo
    tn = _as_mpo_tensor_network(mpo)
    sites = _mps_sites(tn)

    tensors = []
    site_tensors = []
    index_maps = []
    physical_inds = []

    for position, site in enumerate(sites):
        tensor = _mps_site_tensor(tn, site)
        array_summary = symmray_block_summary(tensor)
        upper_ind = tn.upper_ind(site)
        lower_ind = tn.lower_ind(site)

        index_by_ind = {}
        indices = []
        for ind, index_summary in zip(tensor.inds, array_summary["indices"]):
            entry = dict(index_summary)
            entry["ind"] = ind
            index_by_ind[ind] = entry
            indices.append(entry)

        upper_physical = index_by_ind.get(upper_ind)
        lower_physical = index_by_ind.get(lower_ind)
        if upper_physical is None or lower_physical is None:
            raise ValueError(
                f"MPO tensor for site {site!r} does not expose physical "
                f"indices {upper_ind!r} and {lower_ind!r}."
            )

        tensors.append(
            {
                "position": int(position),
                "site": site,
                "site_tag": tn.site_tag(site) if hasattr(tn, "site_tag") else None,
                "upper_ind": upper_ind,
                "lower_ind": lower_ind,
                "inds": tuple(tensor.inds),
                "shape": array_summary["shape"],
                "charge": array_summary["charge"],
                "indices": indices,
                "upper_physical": upper_physical,
                "lower_physical": lower_physical,
                "left_bond": None,
                "right_bond": None,
                "blocks": array_summary["blocks"],
                "num_blocks": array_summary["num_blocks"],
                "dense_size": array_summary["dense_size"],
                "stored_size": array_summary["stored_size"],
                "density": array_summary["density"],
            }
        )
        site_tensors.append(tensor)
        index_maps.append(index_by_ind)
        physical_inds.append((upper_ind, lower_ind))

    bonds = []
    for position, (site_l, site_r) in enumerate(zip(sites[:-1], sites[1:])):
        excluded = (
            physical_inds[position][0],
            physical_inds[position][1],
            physical_inds[position + 1][0],
            physical_inds[position + 1][1],
        )
        ind = _shared_virtual_ind(
            site_tensors[position],
            site_tensors[position + 1],
            excluded,
        )
        if ind is None:
            continue
        left_index = index_maps[position].get(ind)
        right_index = index_maps[position + 1].get(ind)
        index_summary = left_index or right_index
        bond = {
            "position": int(position),
            "left_position": int(position),
            "right_position": int(position + 1),
            "left_site": site_l,
            "right_site": site_r,
            "between": (site_l, site_r),
            "ind": ind,
            "left_direction": left_index["direction"] if left_index is not None else None,
            "right_direction": right_index["direction"] if right_index is not None else None,
            "dim": index_summary["dim"],
            "num_sectors": index_summary["num_sectors"],
            "chargemap": index_summary["chargemap"],
            "sectors": index_summary["sectors"],
        }
        bonds.append(bond)
        tensors[position]["right_bond"] = bond
        tensors[position + 1]["left_bond"] = bond

    total_dense_size = int(sum(tensor["dense_size"] for tensor in tensors))
    total_stored_size = int(sum(tensor["stored_size"] for tensor in tensors))
    total_charge = _resolve_total_charge(source, tensors)
    symmetry = _infer_symmetry(source, site_tensors)
    fermionic = _infer_fermionic(source, site_tensors)
    q_total = _resolve_q_total(symmetry, total_charge)
    return {
        "num_sites": len(sites),
        "sites": sites,
        "tensors": tensors,
        "bonds": bonds,
        "symmetry": symmetry,
        "fermionic": fermionic,
        "fermionic_ordering": _fermionic_ordering_summary(
            source,
            network_kind="mpo",
            sites=sites,
            bonds=bonds,
            fermionic=fermionic,
        ),
        "total_charge": total_charge,
        "charge_total": total_charge,
        "Q_total": q_total,
        "total_parity": _mod_charge(total_charge, 2),
        "max_bond_dim": max((bond["dim"] for bond in bonds), default=1),
        "max_bond_sectors": max((bond["num_sectors"] for bond in bonds), default=0),
        "total_dense_size": total_dense_size,
        "total_stored_size": total_stored_size,
        "density": total_stored_size / total_dense_size if total_dense_size else 0.0,
    }


def draw_symmray_blocks(
    array,
    *,
    ax=None,
    title=None,
    max_blocks=12,
    show_leg_chargemaps=True,
    figsize=None,
    return_summary=False,
):
    """Draw a lightweight sector schematic for a Symmray array.

    The diagram uses :mod:`quimb.schematic` to show array legs, their charge
    maps, and the present block sectors with block shapes.
    """
    summary = symmray_block_summary(array)
    blocks = summary["blocks"]
    max_blocks = int(max_blocks)
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1.")
    shown_blocks = blocks[:max_blocks]

    try:
        from quimb import schematic  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - quimb is a declared dep
        raise ImportError("draw_symmray_blocks requires quimb.schematic.") from exc

    try:
        from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - plotting optionality
        raise ImportError("draw_symmray_blocks requires matplotlib.") from exc

    rank = max(1, len(summary["indices"]))
    block_count = max(1, len(shown_blocks))
    if figsize is None:
        figsize = (
            max(5.5, 1.15 * max(rank, block_count) + 2.2),
            4.2,
        )

    presets = {
        "array": {
            "color": schematic.get_color("blue"),
            "alpha": 0.28,
            "linewidth": 1.6,
        },
        "leg": {
            "color": (0.32, 0.35, 0.38, 1.0),
            "linewidth": 1.5,
        },
        "block": {
            "alpha": 0.62,
            "linewidth": 1.2,
        },
    }
    drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

    drawing.ax.add_patch(
        Rectangle(
            (-1.05, -0.38),
            2.10,
            0.76,
            facecolor=schematic.get_color("blue"),
            edgecolor=(0.12, 0.25, 0.34, 1.0),
            alpha=0.70,
            linewidth=1.3,
            zorder=3,
        )
    )
    center_label = (
        f"shape {_format_shape(summary['shape'])}"
        f"\n{summary['num_blocks']} blocks"
        f"\n{summary['stored_size']}/{summary['dense_size']} stored"
    )
    if summary["charge"] is not None:
        center_label += f"\ncharge {summary['charge']}"
    drawing.text(
        (0.0, 0.0),
        center_label,
        fontsize=10,
        ha="center",
        va="center",
        color="white",
        zorder=4,
    )
    if title is not None:
        drawing.ax.set_title(str(title))

    if summary["indices"]:
        if len(summary["indices"]) == 1:
            xs = [0.0]
        else:
            xs = np.linspace(-1.8, 1.8, len(summary["indices"]))
        for x_pos, index in zip(xs, summary["indices"]):
            drawing.line((0.0, 0.35), (float(x_pos), 0.95), preset="leg")
            label = f"axis {index['axis']} ({index['direction']})\ndim {index['dim']}"
            if show_leg_chargemaps:
                label += "\n" + _format_compact_mapping(index["chargemap"])
            drawing.text((float(x_pos), 1.08), label, fontsize=8, ha="center", va="bottom")

    if shown_blocks:
        if len(shown_blocks) == 1:
            block_xs = [0.0]
        else:
            block_xs = np.linspace(-1.9, 1.9, len(shown_blocks))
        for x_pos, block in zip(block_xs, shown_blocks):
            color = schematic.hash_to_color(str(block["sector"]))
            drawing.ax.add_patch(
                Rectangle(
                    (float(x_pos) - 0.42, -1.28),
                    0.84,
                    0.56,
                    facecolor=color,
                    edgecolor=(0.16, 0.18, 0.21, 0.75),
                    alpha=0.78,
                    linewidth=0.6,
                    zorder=3,
                )
            )
            drawing.text(
                (float(x_pos), -0.91),
                _format_sector(block["sector"]),
                fontsize=8,
                ha="center",
            )
            drawing.text(
                (float(x_pos), -1.43),
                _format_shape(block["shape"]),
                fontsize=8,
                ha="center",
            )
        drawing.text((0.0, -0.55), "present blocks", fontsize=9, ha="center")
        if len(blocks) > len(shown_blocks):
            drawing.text(
                (2.45, -1.0),
                f"+{len(blocks) - len(shown_blocks)} more",
                fontsize=8,
                ha="left",
            )
    else:
        drawing.text((0.0, -1.0), "no stored blocks", fontsize=9, ha="center")

    drawing.ax.set_xlim(-2.55, 2.55)
    drawing.ax.set_ylim(-1.62, 1.58)
    drawing.ax.axis("off")
    if return_summary:
        return drawing, summary
    return drawing


def _draw_symmray_mps_mapped(
    summary,
    shown_tensors,
    shown_bonds,
    *,
    mapper,
    ax,
    title,
    center_position,
    pair_right_position,
    show_arrows,
    show_leg_chargemaps,
    show_bond_labels,
    show_phys_labels,
    show_tensor_labels,
    show_diagnostics,
    show_blocks,
    show_block_labels,
    max_blocks_per_site,
    node_shape,
    node_radius,
    figsize,
    site_cmap,
    return_summary,
):
    coords_by_position = _resolve_chain_mapper(mapper, summary)
    spacing = 1.28
    node_radius = float(node_radius)
    xy_by_position = {
        tensor["position"]: (
            coords_by_position[tensor["position"]][0] * spacing,
            coords_by_position[tensor["position"]][1] * spacing,
        )
        for tensor in shown_tensors
    }

    try:
        from matplotlib import colormaps  # pylint: disable=import-outside-toplevel
        from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - plotting optionality
        raise ImportError("draw_symmray_mps requires matplotlib.") from exc

    try:
        from quimb import schematic  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - quimb is a declared dep
        raise ImportError("draw_symmray_mps requires quimb.schematic.") from exc

    detailed_labels = bool(show_leg_chargemaps and (show_bond_labels or show_phys_labels))
    if figsize is None:
        width = max(5.6, spacing * int(mapper.Lx) + 2.35)
        height = max(4.4, spacing * int(mapper.Ly) + 2.10)
        if show_bond_labels or show_phys_labels or show_blocks:
            width += 0.55 if detailed_labels else 0.35
            height += 0.55 if detailed_labels else 0.35
        if show_diagnostics:
            width += 1.45
        figsize = (width, height)

    presets = {
        "lattice": {
            "color": (0.84, 0.86, 0.89, 1.0),
            "linewidth": 0.95,
            "alpha": 0.78,
        },
        "bond": {
            "color": (0.34, 0.37, 0.41, 0.96),
            "linewidth": 2.0,
            "solid_capstyle": "round",
        },
        "phys": {
            "color": (0.40, 0.43, 0.48, 0.88),
            "linewidth": 1.05,
            "solid_capstyle": "round",
        },
    }
    drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)
    _draw_mapped_chain_grid(drawing, mapper, spacing=spacing)

    max_dim = max((bond["dim"] for bond in shown_bonds), default=1)
    for bond in shown_bonds:
        xy_l = xy_by_position[bond["left_position"]]
        xy_r = xy_by_position[bond["right_position"]]
        width = 1.25 + 1.05 * np.sqrt(bond["dim"] / max_dim)
        drawing.line(
            xy_l,
            xy_r,
            preset="bond",
            linewidth=width,
            shorten=node_radius + 0.035,
            zorder=1,
        )

        if show_arrows:
            start, stop = xy_l, xy_r
            if center_position is not None:
                if bond["right_position"] <= center_position:
                    start, stop = xy_l, xy_r
                elif (
                    pair_right_position is not None
                    and bond["left_position"] >= pair_right_position
                ) or (
                    pair_right_position is None
                    and bond["left_position"] >= center_position
                ):
                    start, stop = xy_r, xy_l
                else:
                    start, stop = None, None
            if start is not None:
                drawing.arrowhead(
                    start,
                    stop,
                    preset="bond",
                    center=0.58,
                    width=0.040,
                    length=0.085,
                    zorder=2,
                )

        if show_bond_labels:
            x0, y0 = xy_l
            x1, y1 = xy_r
            mid = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
            off = _mapped_label_offset(x0, y0, x1, y1, amount=0.15)
            label = _mapped_bond_label(
                bond,
                show_leg_chargemaps=show_leg_chargemaps,
            )
            drawing.text(
                (mid[0] + off[0], mid[1] + off[1]),
                label,
                fontsize=5.7,
                ha="center",
                va="center",
                color=(0.18, 0.20, 0.23, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.05",
                    "facecolor": (1.0, 1.0, 1.0, 0.78),
                    "edgecolor": (1.0, 1.0, 1.0, 0.0),
                    "linewidth": 0.0,
                },
                zorder=8,
            )

    show_block_labels = bool(show_block_labels)
    for tensor in shown_tensors:
        position = tensor["position"]
        x_pos, y_pos = xy_by_position[position]
        facecolor = _mapped_site_color(
            colormaps,
            site_cmap,
            position,
            summary["num_sites"],
        )
        edgecolor = (
            schematic.get_color("orange")
            if position == center_position
            else (0.18, 0.20, 0.23, 0.72)
        )
        linewidth = 2.35 if position == center_position else 0.95

        if node_shape == "circle":
            drawing.circle(
                (x_pos, y_pos),
                radius=node_radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=4,
            )
        elif node_shape == "cube":
            drawing.cube(
                (x_pos, y_pos, 0.0),
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=4,
            )
        else:
            raise ValueError("node_shape must be 'circle' or 'cube'.")

        phys_xy = (x_pos - 0.22, y_pos - 0.30)
        drawing.line(
            (x_pos, y_pos),
            phys_xy,
            preset="phys",
            shorten=(node_radius * 0.55, 0.0),
            zorder=1,
        )
        if show_arrows and position != center_position:
            drawing.arrowhead(
                phys_xy,
                (x_pos, y_pos),
                preset="phys",
                center=0.57,
                width=0.034,
                length=0.070,
                zorder=2,
            )

        if show_tensor_labels:
            label_lines = _mapped_tensor_label_lines(tensor, kind="T")
            if label_lines:
                drawing.text(
                    (x_pos, y_pos),
                    "\n".join(label_lines),
                    fontsize=5.2 if len(label_lines) > 1 else 5.8,
                    ha="center",
                    va="center",
                    color=_mapped_contrast_text_color(facecolor),
                    fontweight="bold",
                    linespacing=0.84,
                    zorder=7,
                )

        physical = tensor["physical"]
        if show_phys_labels:
            phys_label = (
                _mapped_physical_label(
                    "p",
                    tensor,
                    physical,
                    show_leg_chargemaps=show_leg_chargemaps,
                )
            )
            drawing.text(
                (phys_xy[0] - 0.04, phys_xy[1] - 0.08),
                phys_label,
                fontsize=5.5,
                ha="right",
                va="top",
                color=(0.18, 0.20, 0.23, 1.0),
                zorder=7,
            )

        if show_blocks and tensor["blocks"]:
            blocks = tensor["blocks"][:max_blocks_per_site]
            block_width = 0.15
            block_height = 0.12
            gap = 0.032
            total_width = len(blocks) * block_width + (len(blocks) - 1) * gap
            start = x_pos - 0.5 * total_width
            block_y = y_pos - node_radius - 0.17
            for block_pos, block in enumerate(blocks):
                bx = start + block_pos * (block_width + gap)
                color = schematic.hash_to_color(str(block["sector"]))
                drawing.ax.add_patch(
                    Rectangle(
                        (bx, block_y),
                        block_width,
                        block_height,
                        facecolor=color,
                        edgecolor=(0.16, 0.18, 0.21, 0.75),
                        alpha=0.86,
                        linewidth=0.60,
                        zorder=5,
                    )
                )
                if show_block_labels:
                    drawing.ax.text(
                        bx + 0.5 * block_width,
                        block_y - 0.04,
                        _format_sector(block["sector"]),
                        fontsize=4.8,
                        ha="center",
                        va="top",
                        rotation=45,
                        color=(0.15, 0.17, 0.20, 1.0),
                    )

    xs = [xy[0] for xy in xy_by_position.values()] or [0.0]
    ys = [xy[1] for xy in xy_by_position.values()] or [0.0]
    right_pad = 0.85
    if show_diagnostics:
        _draw_mapped_chain_diagnostics(
            drawing,
            summary,
            shown_tensors,
            show_blocks=show_blocks,
            show_arrows=show_arrows,
            x=max(xs + [(int(mapper.Lx) - 1) * spacing]) + 0.82,
            y=max(ys + [(int(mapper.Ly) - 1) * spacing]) + 0.62,
        )
        right_pad = 2.05
    elif len(shown_tensors) < summary["num_sites"]:
        drawing.text(
            (max(xs) + 0.55, min(ys)),
            f"+{summary['num_sites'] - len(shown_tensors)}",
            fontsize=9,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
        right_pad = 1.20

    if title is None:
        title = "Symmray MPS block structure"
    drawing.ax.set_title(title)
    charge_text = _charge_summary_text(summary)
    if charge_text and not show_diagnostics:
        drawing.text(
            (-0.62, (int(mapper.Ly) - 1) * spacing + 0.62),
            charge_text,
            fontsize=8,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
    x_min, x_max, y_min, y_max = _mapped_chain_limits(
        mapper,
        xy_by_position,
        spacing=spacing,
        right_pad=right_pad,
        y_pad=0.95 if show_phys_labels else 0.78,
        left_pad=1.14 if show_phys_labels else 0.82,
    )
    drawing.ax.set_xlim(x_min, x_max)
    drawing.ax.set_ylim(y_min, y_max)
    drawing.ax.set_aspect("equal")
    drawing.ax.axis("off")
    if return_summary:
        return drawing, summary
    return drawing


def draw_symmray_mps(
    mps,
    *,
    ax=None,
    title=None,
    max_sites=None,
    mapper=None,
    center="middle",
    highlight_pair=True,
    show_regions=True,
    show_arrows=True,
    show_leg_chargemaps=True,
    show_bond_labels=False,
    show_phys_labels=False,
    show_tensor_labels=True,
    show_diagnostics=False,
    show_blocks=False,
    show_block_labels=False,
    max_blocks_per_site=4,
    node_shape="circle",
    node_radius=0.24,
    site_cmap="quimb",
    figsize=None,
    return_summary=False,
):
    """Draw a block-aware MPS schematic for a Symmray-backed MPS.

    The diagram uses :mod:`quimb.schematic` in the same style as quimb's manual
    tensor-network schematics: tensor nodes, virtual bonds, physical legs,
    optional canonical-flow arrows, and optional left/center/right region
    highlighting. Pass ``mapper=OneDMap(...)`` to draw the 1D chain on the
    corresponding 2D lattice path; the mapped view uses site-colored nodes and
    quieter gray bonds while preserving the Symmray charge labels.
    """
    summary = symmray_mps_summary(mps)
    max_blocks_per_site = int(max_blocks_per_site)
    if max_blocks_per_site < 1:
        raise ValueError("max_blocks_per_site must be >= 1.")
    if max_sites is None:
        shown_tensors = list(summary["tensors"])
    else:
        max_sites = int(max_sites)
        if max_sites < 1:
            raise ValueError("max_sites must be >= 1.")
        shown_tensors = list(summary["tensors"][:max_sites])

    shown_positions = {tensor["position"] for tensor in shown_tensors}
    center_position = _resolve_mps_position(shown_tensors, center, name="center")
    pair_right_position = None
    if highlight_pair and center_position is not None:
        candidate = center_position + 1
        if candidate in shown_positions:
            pair_right_position = candidate
    shown_bonds = [
        bond
        for bond in summary["bonds"]
        if bond["left_position"] in shown_positions
        and bond["right_position"] in shown_positions
    ]

    if mapper is not None:
        return _draw_symmray_mps_mapped(
            summary,
            shown_tensors,
            shown_bonds,
            mapper=mapper,
            ax=ax,
            title=title,
            center_position=center_position,
            pair_right_position=pair_right_position,
            show_arrows=show_arrows,
            show_leg_chargemaps=show_leg_chargemaps,
            show_bond_labels=show_bond_labels,
            show_phys_labels=show_phys_labels,
            show_tensor_labels=show_tensor_labels,
            show_diagnostics=show_diagnostics,
            show_blocks=show_blocks,
            show_block_labels=show_block_labels,
            max_blocks_per_site=max_blocks_per_site,
            node_shape=node_shape,
            node_radius=node_radius,
            figsize=figsize,
            site_cmap=site_cmap,
            return_summary=return_summary,
        )

    try:
        from quimb import schematic  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - quimb is a declared dep
        raise ImportError("draw_symmray_mps requires quimb.schematic.") from exc

    try:
        from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - plotting optionality
        raise ImportError("draw_symmray_mps requires matplotlib.") from exc

    n_show = len(shown_tensors)
    detailed_labels = bool(show_leg_chargemaps and (show_bond_labels or show_phys_labels))
    if detailed_labels:
        spacing = 1.72 if n_show <= 12 else 1.32
    else:
        spacing = 1.0 if n_show > 12 else 1.12
    if figsize is None:
        if show_bond_labels or show_phys_labels or show_blocks:
            height = 5.25 if detailed_labels else 3.9
            if show_block_labels:
                height += 0.35
            if show_diagnostics:
                height += 0.15
            figsize = (max(8.0, spacing * max(1, n_show - 1) + 4.0), height)
        else:
            figsize = (max(6.5, 0.95 * n_show + 1.5), 2.9)

    presets = {
        "bond": {
            "color": (0.12, 0.14, 0.16, 1.0),
            "linewidth": 3.0,
        },
        "phys": {
            "color": (0.12, 0.14, 0.16, 1.0),
            "linewidth": 1.55,
        },
        "center": {
            "color": schematic.get_color("orange"),
            "hatch": "/////",
            "linewidth": 1.3,
            "edgecolor": (0.55, 0.38, 0.00, 1.0),
        },
        "left": {
            "color": schematic.get_color("bluedark"),
            "linewidth": 1.3,
            "edgecolor": (0.02, 0.22, 0.34, 1.0),
        },
        "right": {
            "color": schematic.get_color("blue"),
            "linewidth": 1.3,
            "edgecolor": (0.05, 0.34, 0.50, 1.0),
        },
        "pair": {
            "facecolor": (0.20, 0.80, 0.50, 0.34),
            "edgecolor": (0.18, 0.50, 0.30, 0.72),
            "linestyle": ":",
            "linewidth": 1.5,
        },
        "region": {
            "facecolor": (0.83, 0.83, 0.83, 0.42),
            "edgecolor": (0.42, 0.42, 0.42, 0.72),
            "linestyle": ":",
            "linewidth": 1.4,
        },
    }
    drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

    x_by_position = {
        tensor["position"]: i * spacing
        for i, tensor in enumerate(shown_tensors)
    }
    y0 = 0.0
    phys_y = -1.05 if (show_phys_labels and show_blocks) else -0.68
    block_y = -0.50 if (show_phys_labels and show_blocks) else -0.46
    max_dim = max((bond["dim"] for bond in shown_bonds), default=1)

    left_region = []
    right_region = []
    if center_position is not None:
        right_region_start = (
            pair_right_position + 1
            if pair_right_position is not None
            else center_position + 1
        )
        left_region = [
            (x_by_position[tensor["position"]], y0)
            for tensor in shown_tensors
            if tensor["position"] < center_position
        ]
        right_region = [
            (x_by_position[tensor["position"]], y0)
            for tensor in shown_tensors
            if tensor["position"] >= right_region_start
        ]
        if show_regions and left_region:
            drawing.patch_around(left_region, radius=0.46, preset="region", zorder=0)
            drawing.text(
                (left_region[-1][0] - 0.25 * spacing, 1.36 if detailed_labels else 0.90),
                "LEFT",
                fontsize=12,
                color=(0.18, 0.19, 0.21, 1.0),
                ha="center",
            )
        if show_regions and right_region:
            drawing.patch_around(right_region, radius=0.46, preset="region", zorder=0)
            drawing.text(
                (right_region[0][0] + 0.35 * spacing, 1.36 if detailed_labels else 0.90),
                "RIGHT",
                fontsize=12,
                color=(0.18, 0.19, 0.21, 1.0),
                ha="center",
            )
        if show_regions and pair_right_position is not None:
            drawing.patch_around_circles(
                (x_by_position[center_position], y0),
                node_radius + 0.04,
                (x_by_position[pair_right_position], y0),
                node_radius + 0.04,
                padding=0.22,
                preset="pair",
                zorder=0,
            )

    for bond in shown_bonds:
        x0 = x_by_position[bond["left_position"]]
        x1 = x_by_position[bond["right_position"]]
        width = 2.4 + 1.2 * (bond["dim"] / max_dim)
        drawing.line((x0, y0), (x1, y0), preset="bond", linewidth=width, zorder=1)

        if show_arrows:
            if center_position is None:
                drawing.arrowhead((x0, y0), (x1, y0), preset="bond", center=0.58, width=0.055, length=0.11)
            elif bond["right_position"] <= center_position:
                drawing.arrowhead((x0, y0), (x1, y0), preset="bond", center=0.58, width=0.055, length=0.11)
            elif (
                pair_right_position is not None
                and bond["left_position"] >= pair_right_position
            ) or (
                pair_right_position is None
                and bond["left_position"] >= center_position
            ):
                drawing.arrowhead((x1, y0), (x0, y0), preset="bond", center=0.58, width=0.055, length=0.11)

        mid = 0.5 * (x0 + x1)
        if show_bond_labels:
            flow = _flow_math(bond.get("left_direction"), bond.get("right_direction"))
            if flow:
                label = rf"$e_{{{bond['position']}}}: {flow}, \chi={bond['dim']}$"
            else:
                label = rf"$e_{{{bond['position']}}}: \chi={bond['dim']}$"
            if show_leg_chargemaps:
                label += "\n" + rf"$q_e:$ {_format_compact_mapping(bond['chargemap'], max_items=4)}"
            label_y = 0.28 + (0.18 if detailed_labels and bond["position"] % 2 else 0.0)
            drawing.text(
                (mid, label_y),
                label,
                fontsize=6.6,
                ha="center",
                va="bottom",
                color=(0.18, 0.20, 0.23, 1.0),
                zorder=5,
            )

    show_block_labels = bool(show_block_labels)
    for tensor in shown_tensors:
        x_pos = x_by_position[tensor["position"]]
        position = tensor["position"]
        if position == center_position:
            preset = "center"
        elif pair_right_position is not None and position == pair_right_position:
            preset = "right"
        elif center_position is not None and position < center_position:
            preset = "left"
        else:
            preset = "right"

        if node_shape == "circle":
            drawing.circle((x_pos, y0), radius=node_radius, preset=preset, zorder=3)
        elif node_shape == "cube":
            drawing.cube((x_pos, y0, 0.0), preset=preset, zorder=3)
        else:
            raise ValueError("node_shape must be 'circle' or 'cube'.")

        drawing.line(
            (x_pos, y0),
            (x_pos, phys_y),
            preset="phys",
            zorder=1,
        )
        if show_arrows and position != center_position:
            drawing.arrowhead(
                (x_pos, phys_y),
                (x_pos, y0),
                preset="phys",
                center=0.55,
                width=0.040,
                length=0.080,
            )

        if show_tensor_labels:
            label_lines = [rf"$T_{{{tensor['site']}}}$"]
            if tensor["charge"] is not None:
                label_lines.append(rf"$q={_format_charge(tensor['charge'])}$")
            if not show_blocks:
                label_lines.append(rf"$B={tensor['num_blocks']}$")
            label = "\n".join(label_lines)
            label_color = (
                schematic.get_color("orange")
                if position == center_position
                else (0.06, 0.20, 0.30, 1.0)
            )
            tensor_label_y = 0.76 if detailed_labels else 0.54
            drawing.text(
                (x_pos, tensor_label_y),
                label,
                fontsize=8.5,
                ha="center",
                va="bottom",
                color=label_color,
                zorder=5,
            )

        physical = tensor["physical"]
        if show_phys_labels:
            phys_label = (
                rf"$p_{{{tensor['site']}}}: \mathrm{{{physical['direction']}}}, d={physical['dim']}$"
            )
            if show_leg_chargemaps:
                phys_label += "\n" + rf"$q_p:$ {_format_compact_mapping(physical['chargemap'], max_items=4)}"
            drawing.text(
                (x_pos, phys_y - 0.12),
                phys_label,
                fontsize=6.4,
                ha="center",
                va="top",
                color=(0.18, 0.20, 0.23, 1.0),
            )

        if show_blocks and tensor["blocks"]:
            blocks = tensor["blocks"][:max_blocks_per_site]
            block_width = 0.18
            block_height = 0.14
            gap = 0.035
            total_width = len(blocks) * block_width + (len(blocks) - 1) * gap
            start = x_pos - 0.5 * total_width
            for block_pos, block in enumerate(blocks):
                bx = start + block_pos * (block_width + gap)
                color = schematic.hash_to_color(str(block["sector"]))
                drawing.ax.add_patch(
                    Rectangle(
                        (bx, block_y),
                        block_width,
                        block_height,
                        facecolor=color,
                        edgecolor=(0.16, 0.18, 0.21, 0.75),
                        alpha=0.86,
                        linewidth=0.65,
                        zorder=4,
                    )
                )
                if show_block_labels:
                    drawing.ax.text(
                        bx + 0.5 * block_width,
                        block_y - 0.045,
                        _format_sector(block["sector"]),
                        fontsize=5.2,
                        ha="center",
                        va="top",
                        rotation=45,
                        color=(0.15, 0.17, 0.20, 1.0),
                    )
            drawing.ax.text(
                x_pos,
                block_y + block_height + 0.025,
                rf"$B={tensor['num_blocks']}$",
                fontsize=6.5,
                ha="center",
                va="bottom",
                color=(0.15, 0.17, 0.20, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.06",
                    "facecolor": (1.0, 1.0, 1.0, 0.78),
                    "edgecolor": (1.0, 1.0, 1.0, 0.0),
                    "linewidth": 0.0,
                },
                zorder=5,
            )

    last_x = x_by_position[shown_tensors[-1]["position"]] if shown_tensors else 0.0
    right_pad = 0.75
    if show_diagnostics:
        charge_line = _charge_summary_text(summary)
        diagnostic_lines = [f"sites {summary['num_sites']}"]
        if charge_line:
            diagnostic_lines.append(charge_line)
        diagnostic_lines += [
            f"max bond {summary['max_bond_dim']}",
            f"bond sectors {summary['max_bond_sectors']}",
            f"stored {summary['total_stored_size']}/{summary['total_dense_size']}",
            f"density {summary['density']:.3f}",
        ]
        diagnostic = "\n".join(diagnostic_lines)
        if show_blocks:
            diagnostic += "\ncolored tiles: stored blocks"
        if show_arrows:
            diagnostic += "\narrows/labels: charge in/out flow"
        if len(shown_tensors) < summary["num_sites"]:
            diagnostic += f"\n+{summary['num_sites'] - len(shown_tensors)} sites hidden"
        summary_x = last_x + 0.82
        drawing.ax.text(
            summary_x,
            0.52,
            diagnostic,
            fontsize=8,
            ha="left",
            va="top",
            color=(0.15, 0.17, 0.20, 1.0),
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": (1.0, 1.0, 1.0, 0.92),
                "edgecolor": (0.68, 0.70, 0.74, 1.0),
                "linewidth": 0.8,
            },
        )
        right_pad = 1.80
    elif len(shown_tensors) < summary["num_sites"]:
        drawing.text(
            (last_x + 0.55, y0),
            f"+{summary['num_sites'] - len(shown_tensors)}",
            fontsize=9,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
        right_pad = 1.10

    if title is None:
        title = "Symmray MPS block structure"
    drawing.ax.set_title(title)
    charge_text = _charge_summary_text(summary)
    if charge_text and not show_diagnostics:
        drawing.text(
            (-0.55, 1.12),
            charge_text,
            fontsize=8,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
    drawing.ax.set_xlim(-0.65, last_x + right_pad)
    y_min = -1.55 if (show_phys_labels and show_blocks) else (-1.18 if (show_phys_labels or show_block_labels) else -0.96)
    y_max = 1.78 if detailed_labels else (1.35 if (show_regions or show_leg_chargemaps) else 0.92)
    drawing.ax.set_ylim(y_min, y_max)
    drawing.ax.axis("off")
    if return_summary:
        return drawing, summary
    return drawing


def _draw_symmray_mpo_mapped(
    summary,
    shown_tensors,
    shown_bonds,
    *,
    mapper,
    ax,
    title,
    center_position,
    pair_right_position,
    show_arrows,
    show_leg_chargemaps,
    show_bond_labels,
    show_phys_labels,
    show_tensor_labels,
    show_diagnostics,
    show_blocks,
    show_block_labels,
    max_blocks_per_site,
    node_shape,
    node_radius,
    figsize,
    site_cmap,
    return_summary,
):
    coords_by_position = _resolve_chain_mapper(mapper, summary)
    spacing = 1.28
    node_radius = float(node_radius)
    xy_by_position = {
        tensor["position"]: (
            coords_by_position[tensor["position"]][0] * spacing,
            coords_by_position[tensor["position"]][1] * spacing,
        )
        for tensor in shown_tensors
    }

    try:
        from matplotlib import colormaps  # pylint: disable=import-outside-toplevel
        from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - plotting optionality
        raise ImportError("draw_symmray_mpo requires matplotlib.") from exc

    try:
        from quimb import schematic  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - quimb is a declared dep
        raise ImportError("draw_symmray_mpo requires quimb.schematic.") from exc

    detailed_labels = bool(show_leg_chargemaps and (show_bond_labels or show_phys_labels))
    if figsize is None:
        width = max(5.6, spacing * int(mapper.Lx) + 2.35)
        height = max(4.6, spacing * int(mapper.Ly) + 2.35)
        if show_bond_labels or show_phys_labels or show_blocks:
            width += 0.65 if detailed_labels else 0.40
            height += 0.70 if detailed_labels else 0.45
        if show_diagnostics:
            width += 1.45
        figsize = (width, height)

    presets = {
        "lattice": {
            "color": (0.84, 0.86, 0.89, 1.0),
            "linewidth": 0.95,
            "alpha": 0.78,
        },
        "bond": {
            "color": (0.34, 0.37, 0.41, 0.96),
            "linewidth": 2.0,
            "solid_capstyle": "round",
        },
        "phys": {
            "color": (0.40, 0.43, 0.48, 0.88),
            "linewidth": 1.05,
            "solid_capstyle": "round",
        },
    }
    drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)
    _draw_mapped_chain_grid(drawing, mapper, spacing=spacing)

    max_dim = max((bond["dim"] for bond in shown_bonds), default=1)
    for bond in shown_bonds:
        xy_l = xy_by_position[bond["left_position"]]
        xy_r = xy_by_position[bond["right_position"]]
        width = 1.25 + 1.05 * np.sqrt(bond["dim"] / max_dim)
        drawing.line(
            xy_l,
            xy_r,
            preset="bond",
            linewidth=width,
            shorten=node_radius + 0.035,
            zorder=1,
        )

        if show_arrows:
            start, stop = xy_l, xy_r
            if center_position is not None:
                if bond["right_position"] <= center_position:
                    start, stop = xy_l, xy_r
                elif (
                    pair_right_position is not None
                    and bond["left_position"] >= pair_right_position
                ) or (
                    pair_right_position is None
                    and bond["left_position"] >= center_position
                ):
                    start, stop = xy_r, xy_l
                else:
                    start, stop = None, None
            if start is not None:
                drawing.arrowhead(
                    start,
                    stop,
                    preset="bond",
                    center=0.58,
                    width=0.040,
                    length=0.085,
                    zorder=2,
                )

        if show_bond_labels:
            x0, y0 = xy_l
            x1, y1 = xy_r
            mid = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
            off = _mapped_label_offset(x0, y0, x1, y1, amount=0.15)
            label = _mapped_bond_label(
                bond,
                show_leg_chargemaps=show_leg_chargemaps,
            )
            drawing.text(
                (mid[0] + off[0], mid[1] + off[1]),
                label,
                fontsize=5.7,
                ha="center",
                va="center",
                color=(0.18, 0.20, 0.23, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.05",
                    "facecolor": (1.0, 1.0, 1.0, 0.78),
                    "edgecolor": (1.0, 1.0, 1.0, 0.0),
                    "linewidth": 0.0,
                },
                zorder=8,
            )

    show_block_labels = bool(show_block_labels)
    for tensor in shown_tensors:
        position = tensor["position"]
        x_pos, y_pos = xy_by_position[position]
        facecolor = _mapped_site_color(
            colormaps,
            site_cmap,
            position,
            summary["num_sites"],
        )
        edgecolor = (
            schematic.get_color("orange")
            if position == center_position
            else (0.18, 0.20, 0.23, 0.72)
        )
        linewidth = 2.35 if position == center_position else 0.95

        if node_shape == "circle":
            drawing.circle(
                (x_pos, y_pos),
                radius=node_radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=4,
            )
        elif node_shape == "cube":
            drawing.cube(
                (x_pos, y_pos, 0.0),
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=4,
            )
        else:
            raise ValueError("node_shape must be 'circle' or 'cube'.")

        upper_xy = (x_pos - 0.22, y_pos + 0.30)
        lower_xy = (x_pos + 0.22, y_pos - 0.30)
        drawing.line(
            (x_pos, y_pos),
            upper_xy,
            preset="phys",
            shorten=(node_radius * 0.55, 0.0),
            zorder=1,
        )
        drawing.line(
            (x_pos, y_pos),
            lower_xy,
            preset="phys",
            shorten=(node_radius * 0.55, 0.0),
            zorder=1,
        )
        if show_arrows:
            drawing.arrowhead(
                (x_pos, y_pos),
                upper_xy,
                preset="phys",
                center=0.57,
                width=0.034,
                length=0.070,
                zorder=2,
            )
            drawing.arrowhead(
                lower_xy,
                (x_pos, y_pos),
                preset="phys",
                center=0.57,
                width=0.034,
                length=0.070,
                zorder=2,
            )

        if show_tensor_labels:
            label_lines = _mapped_tensor_label_lines(tensor, kind="W")
            if label_lines:
                drawing.text(
                    (x_pos, y_pos),
                    "\n".join(label_lines),
                    fontsize=5.2 if len(label_lines) > 1 else 5.8,
                    ha="center",
                    va="center",
                    color=_mapped_contrast_text_color(facecolor),
                    fontweight="bold",
                    linespacing=0.84,
                    zorder=7,
                )

        if show_phys_labels:
            upper = tensor["upper_physical"]
            lower = tensor["lower_physical"]
            upper_label = _mapped_physical_label(
                "u",
                tensor,
                upper,
                show_leg_chargemaps=show_leg_chargemaps,
            )
            lower_label = _mapped_physical_label(
                "l",
                tensor,
                lower,
                show_leg_chargemaps=show_leg_chargemaps,
            )
            drawing.text(
                (upper_xy[0] - 0.04, upper_xy[1] + 0.08),
                upper_label,
                fontsize=5.5,
                ha="right",
                va="bottom",
                color=(0.18, 0.20, 0.23, 1.0),
                zorder=7,
            )
            drawing.text(
                (lower_xy[0] + 0.04, lower_xy[1] - 0.08),
                lower_label,
                fontsize=5.5,
                ha="left",
                va="top",
                color=(0.18, 0.20, 0.23, 1.0),
                zorder=7,
            )

        if show_blocks and tensor["blocks"]:
            blocks = tensor["blocks"][:max_blocks_per_site]
            block_width = 0.15
            block_height = 0.12
            gap = 0.032
            total_width = len(blocks) * block_width + (len(blocks) - 1) * gap
            start = x_pos - 0.5 * total_width
            block_y = y_pos - node_radius - 0.17
            for block_pos, block in enumerate(blocks):
                bx = start + block_pos * (block_width + gap)
                color = schematic.hash_to_color(str(block["sector"]))
                drawing.ax.add_patch(
                    Rectangle(
                        (bx, block_y),
                        block_width,
                        block_height,
                        facecolor=color,
                        edgecolor=(0.16, 0.18, 0.21, 0.75),
                        alpha=0.86,
                        linewidth=0.60,
                        zorder=5,
                    )
                )
                if show_block_labels:
                    drawing.ax.text(
                        bx + 0.5 * block_width,
                        block_y - 0.04,
                        _format_sector(block["sector"]),
                        fontsize=4.8,
                        ha="center",
                        va="top",
                        rotation=45,
                        color=(0.15, 0.17, 0.20, 1.0),
                    )

    xs = [xy[0] for xy in xy_by_position.values()] or [0.0]
    ys = [xy[1] for xy in xy_by_position.values()] or [0.0]
    right_pad = 0.85
    if show_diagnostics:
        _draw_mapped_chain_diagnostics(
            drawing,
            summary,
            shown_tensors,
            show_blocks=show_blocks,
            show_arrows=show_arrows,
            x=max(xs + [(int(mapper.Lx) - 1) * spacing]) + 0.82,
            y=max(ys + [(int(mapper.Ly) - 1) * spacing]) + 0.62,
        )
        right_pad = 2.05
    elif len(shown_tensors) < summary["num_sites"]:
        drawing.text(
            (max(xs) + 0.55, min(ys)),
            f"+{summary['num_sites'] - len(shown_tensors)}",
            fontsize=9,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
        right_pad = 1.20

    if title is None:
        title = "Symmray MPO block structure"
    drawing.ax.set_title(title)
    charge_text = _charge_summary_text(summary)
    if charge_text and not show_diagnostics:
        drawing.text(
            (-0.62, (int(mapper.Ly) - 1) * spacing + 0.62),
            charge_text,
            fontsize=8,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
    x_min, x_max, y_min, y_max = _mapped_chain_limits(
        mapper,
        xy_by_position,
        spacing=spacing,
        right_pad=right_pad,
        y_pad=1.08 if show_phys_labels else 0.88,
        left_pad=1.22 if show_phys_labels else 0.82,
    )
    drawing.ax.set_xlim(x_min, x_max)
    drawing.ax.set_ylim(y_min, y_max)
    drawing.ax.set_aspect("equal")
    drawing.ax.axis("off")
    if return_summary:
        return drawing, summary
    return drawing


def draw_symmray_mpo(
    mpo,
    *,
    ax=None,
    title=None,
    max_sites=None,
    mapper=None,
    center="middle",
    highlight_pair=False,
    show_regions=False,
    show_arrows=True,
    show_leg_chargemaps=True,
    show_bond_labels=False,
    show_phys_labels=False,
    show_tensor_labels=True,
    show_diagnostics=False,
    show_blocks=False,
    show_block_labels=False,
    max_blocks_per_site=4,
    node_shape="circle",
    node_radius=0.24,
    site_cmap="quimb",
    figsize=None,
    return_summary=False,
):
    """Draw a block-aware MPO schematic for a Symmray-backed MPO.

    The diagram uses the same compact 1D style as ``draw_symmray_mps`` but
    shows both operator physical legs at each site: upper/output and
    lower/input. Pass ``mapper=OneDMap(...)`` to draw the MPS-chain MPO on the
    corresponding 2D lattice path.
    """
    summary = symmray_mpo_summary(mpo)
    max_blocks_per_site = int(max_blocks_per_site)
    if max_blocks_per_site < 1:
        raise ValueError("max_blocks_per_site must be >= 1.")
    if max_sites is None:
        shown_tensors = list(summary["tensors"])
    else:
        max_sites = int(max_sites)
        if max_sites < 1:
            raise ValueError("max_sites must be >= 1.")
        shown_tensors = list(summary["tensors"][:max_sites])

    shown_positions = {tensor["position"] for tensor in shown_tensors}
    center_position = _resolve_mps_position(shown_tensors, center, name="center")
    pair_right_position = None
    if highlight_pair and center_position is not None:
        candidate = center_position + 1
        if candidate in shown_positions:
            pair_right_position = candidate
    shown_bonds = [
        bond
        for bond in summary["bonds"]
        if bond["left_position"] in shown_positions
        and bond["right_position"] in shown_positions
    ]

    if mapper is not None:
        return _draw_symmray_mpo_mapped(
            summary,
            shown_tensors,
            shown_bonds,
            mapper=mapper,
            ax=ax,
            title=title,
            center_position=center_position,
            pair_right_position=pair_right_position,
            show_arrows=show_arrows,
            show_leg_chargemaps=show_leg_chargemaps,
            show_bond_labels=show_bond_labels,
            show_phys_labels=show_phys_labels,
            show_tensor_labels=show_tensor_labels,
            show_diagnostics=show_diagnostics,
            show_blocks=show_blocks,
            show_block_labels=show_block_labels,
            max_blocks_per_site=max_blocks_per_site,
            node_shape=node_shape,
            node_radius=node_radius,
            figsize=figsize,
            site_cmap=site_cmap,
            return_summary=return_summary,
        )

    try:
        from quimb import schematic  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - quimb is a declared dep
        raise ImportError("draw_symmray_mpo requires quimb.schematic.") from exc

    try:
        from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - plotting optionality
        raise ImportError("draw_symmray_mpo requires matplotlib.") from exc

    n_show = len(shown_tensors)
    detailed_labels = bool(show_leg_chargemaps and (show_bond_labels or show_phys_labels))
    spacing = 1.72 if detailed_labels and n_show <= 12 else (1.32 if detailed_labels else 1.08)
    if figsize is None:
        height = 4.25
        if show_bond_labels or show_phys_labels or show_blocks:
            height = 5.45 if detailed_labels else 4.65
        if show_block_labels:
            height += 0.35
        if show_diagnostics:
            height += 0.15
        figsize = (max(7.0, spacing * max(1, n_show - 1) + 3.0), height)

    presets = {
        "bond": {
            "color": (0.12, 0.14, 0.16, 1.0),
            "linewidth": 3.0,
        },
        "phys": {
            "color": (0.12, 0.14, 0.16, 1.0),
            "linewidth": 1.55,
        },
        "center": {
            "color": schematic.get_color("orange"),
            "hatch": "/////",
            "linewidth": 1.3,
            "edgecolor": (0.55, 0.38, 0.00, 1.0),
        },
        "left": {
            "color": schematic.get_color("bluedark"),
            "linewidth": 1.3,
            "edgecolor": (0.02, 0.22, 0.34, 1.0),
        },
        "right": {
            "color": schematic.get_color("blue"),
            "linewidth": 1.3,
            "edgecolor": (0.05, 0.34, 0.50, 1.0),
        },
        "pair": {
            "facecolor": (0.20, 0.80, 0.50, 0.34),
            "edgecolor": (0.18, 0.50, 0.30, 0.72),
            "linestyle": ":",
            "linewidth": 1.5,
        },
        "region": {
            "facecolor": (0.83, 0.83, 0.83, 0.42),
            "edgecolor": (0.42, 0.42, 0.42, 0.72),
            "linestyle": ":",
            "linewidth": 1.4,
        },
    }
    drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

    x_by_position = {
        tensor["position"]: i * spacing
        for i, tensor in enumerate(shown_tensors)
    }
    y0 = 0.0
    upper_y = 0.72
    lower_y = -0.72
    block_y = -1.25 if show_phys_labels else -1.05
    max_dim = max((bond["dim"] for bond in shown_bonds), default=1)

    if center_position is not None:
        right_region_start = (
            pair_right_position + 1
            if pair_right_position is not None
            else center_position + 1
        )
        left_region = [
            (x_by_position[tensor["position"]], y0)
            for tensor in shown_tensors
            if tensor["position"] < center_position
        ]
        right_region = [
            (x_by_position[tensor["position"]], y0)
            for tensor in shown_tensors
            if tensor["position"] >= right_region_start
        ]
        if show_regions and left_region:
            drawing.patch_around(left_region, radius=0.46, preset="region", zorder=0)
            drawing.text(
                (left_region[-1][0] - 0.25 * spacing, 1.42),
                "LEFT",
                fontsize=12,
                color=(0.18, 0.19, 0.21, 1.0),
                ha="center",
            )
        if show_regions and right_region:
            drawing.patch_around(right_region, radius=0.46, preset="region", zorder=0)
            drawing.text(
                (right_region[0][0] + 0.35 * spacing, 1.42),
                "RIGHT",
                fontsize=12,
                color=(0.18, 0.19, 0.21, 1.0),
                ha="center",
            )
        if show_regions and pair_right_position is not None:
            drawing.patch_around_circles(
                (x_by_position[center_position], y0),
                node_radius + 0.04,
                (x_by_position[pair_right_position], y0),
                node_radius + 0.04,
                padding=0.22,
                preset="pair",
                zorder=0,
            )

    for bond in shown_bonds:
        x0 = x_by_position[bond["left_position"]]
        x1 = x_by_position[bond["right_position"]]
        width = 2.4 + 1.2 * (bond["dim"] / max_dim)
        drawing.line((x0, y0), (x1, y0), preset="bond", linewidth=width, zorder=1)

        if show_arrows:
            if center_position is None:
                drawing.arrowhead((x0, y0), (x1, y0), preset="bond", center=0.58, width=0.055, length=0.11)
            elif bond["right_position"] <= center_position:
                drawing.arrowhead((x0, y0), (x1, y0), preset="bond", center=0.58, width=0.055, length=0.11)
            elif (
                pair_right_position is not None
                and bond["left_position"] >= pair_right_position
            ) or (
                pair_right_position is None
                and bond["left_position"] >= center_position
            ):
                drawing.arrowhead((x1, y0), (x0, y0), preset="bond", center=0.58, width=0.055, length=0.11)

        if show_bond_labels:
            mid = 0.5 * (x0 + x1)
            flow = _flow_math(bond.get("left_direction"), bond.get("right_direction"))
            if flow:
                label = rf"$e_{{{bond['position']}}}: {flow}, \chi={bond['dim']}$"
            else:
                label = rf"$e_{{{bond['position']}}}: \chi={bond['dim']}$"
            if show_leg_chargemaps:
                label += "\n" + rf"$q_e:$ {_format_compact_mapping(bond['chargemap'], max_items=4)}"
            drawing.text(
                (mid, 0.30),
                label,
                fontsize=6.6,
                ha="center",
                va="bottom",
                color=(0.18, 0.20, 0.23, 1.0),
                zorder=5,
            )

    show_block_labels = bool(show_block_labels)
    for tensor in shown_tensors:
        x_pos = x_by_position[tensor["position"]]
        position = tensor["position"]
        if position == center_position:
            preset = "center"
        elif pair_right_position is not None and position == pair_right_position:
            preset = "right"
        elif center_position is not None and position < center_position:
            preset = "left"
        else:
            preset = "right"

        if node_shape == "circle":
            drawing.circle((x_pos, y0), radius=node_radius, preset=preset, zorder=3)
        elif node_shape == "cube":
            drawing.cube((x_pos, y0, 0.0), preset=preset, zorder=3)
        else:
            raise ValueError("node_shape must be 'circle' or 'cube'.")

        drawing.line((x_pos, y0), (x_pos, upper_y), preset="phys", zorder=1)
        drawing.line((x_pos, y0), (x_pos, lower_y), preset="phys", zorder=1)
        if show_arrows:
            drawing.arrowhead(
                (x_pos, y0),
                (x_pos, upper_y),
                preset="phys",
                center=0.56,
                width=0.040,
                length=0.080,
            )
            drawing.arrowhead(
                (x_pos, lower_y),
                (x_pos, y0),
                preset="phys",
                center=0.56,
                width=0.040,
                length=0.080,
            )
        if show_tensor_labels:
            label_lines = [rf"$W_{{{tensor['site']}}}$"]
            if tensor["charge"] is not None:
                label_lines.append(rf"$q={_format_charge(tensor['charge'])}$")
            if not show_blocks:
                label_lines.append(rf"$B={tensor['num_blocks']}$")
            label = "\n".join(label_lines)
            label_color = (
                schematic.get_color("orange")
                if position == center_position
                else (0.06, 0.20, 0.30, 1.0)
            )
            drawing.text(
                (x_pos, 1.02),
                label,
                fontsize=8.5,
                ha="center",
                va="bottom",
                color=label_color,
                zorder=5,
            )

        if show_phys_labels:
            upper = tensor["upper_physical"]
            lower = tensor["lower_physical"]
            upper_label = (
                rf"$u_{{{tensor['site']}}}: \mathrm{{{upper['direction']}}}, d={upper['dim']}$"
            )
            lower_label = (
                rf"$l_{{{tensor['site']}}}: \mathrm{{{lower['direction']}}}, d={lower['dim']}$"
            )
            if show_leg_chargemaps:
                upper_label += "\n" + rf"$q_u:$ {_format_compact_mapping(upper['chargemap'], max_items=4)}"
                lower_label += "\n" + rf"$q_l:$ {_format_compact_mapping(lower['chargemap'], max_items=4)}"
            drawing.text(
                (x_pos, upper_y + 0.14),
                upper_label,
                fontsize=6.4,
                ha="center",
                va="bottom",
                color=(0.18, 0.20, 0.23, 1.0),
            )
            drawing.text(
                (x_pos, lower_y - 0.14),
                lower_label,
                fontsize=6.4,
                ha="center",
                va="top",
                color=(0.18, 0.20, 0.23, 1.0),
            )

        if show_blocks and tensor["blocks"]:
            blocks = tensor["blocks"][:max_blocks_per_site]
            block_width = 0.18
            block_height = 0.14
            gap = 0.035
            total_width = len(blocks) * block_width + (len(blocks) - 1) * gap
            start = x_pos - 0.5 * total_width
            for block_pos, block in enumerate(blocks):
                bx = start + block_pos * (block_width + gap)
                color = schematic.hash_to_color(str(block["sector"]))
                drawing.ax.add_patch(
                    Rectangle(
                        (bx, block_y),
                        block_width,
                        block_height,
                        facecolor=color,
                        edgecolor=(0.16, 0.18, 0.21, 0.75),
                        alpha=0.86,
                        linewidth=0.65,
                        zorder=4,
                    )
                )
                if show_block_labels:
                    drawing.ax.text(
                        bx + 0.5 * block_width,
                        block_y - 0.045,
                        _format_sector(block["sector"]),
                        fontsize=5.2,
                        ha="center",
                        va="top",
                        rotation=45,
                        color=(0.15, 0.17, 0.20, 1.0),
                    )
            drawing.ax.text(
                x_pos,
                block_y + block_height + 0.025,
                rf"$B={tensor['num_blocks']}$",
                fontsize=6.5,
                ha="center",
                va="bottom",
                color=(0.15, 0.17, 0.20, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.06",
                    "facecolor": (1.0, 1.0, 1.0, 0.78),
                    "edgecolor": (1.0, 1.0, 1.0, 0.0),
                    "linewidth": 0.0,
                },
                zorder=5,
            )

    last_x = x_by_position[shown_tensors[-1]["position"]] if shown_tensors else 0.0
    right_pad = 0.75
    if show_diagnostics:
        charge_line = _charge_summary_text(summary)
        diagnostic_lines = [f"sites {summary['num_sites']}"]
        if charge_line:
            diagnostic_lines.append(charge_line)
        diagnostic_lines += [
            f"max bond {summary['max_bond_dim']}",
            f"bond sectors {summary['max_bond_sectors']}",
            f"stored {summary['total_stored_size']}/{summary['total_dense_size']}",
            f"density {summary['density']:.3f}",
        ]
        diagnostic = "\n".join(diagnostic_lines)
        if show_blocks:
            diagnostic += "\ncolored tiles: stored blocks"
        if show_arrows:
            diagnostic += "\narrows/labels: charge in/out flow"
        if len(shown_tensors) < summary["num_sites"]:
            diagnostic += f"\n+{summary['num_sites'] - len(shown_tensors)} sites hidden"
        drawing.ax.text(
            last_x + 0.82,
            0.62,
            diagnostic,
            fontsize=8,
            ha="left",
            va="top",
            color=(0.15, 0.17, 0.20, 1.0),
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": (1.0, 1.0, 1.0, 0.92),
                "edgecolor": (0.68, 0.70, 0.74, 1.0),
                "linewidth": 0.8,
            },
        )
        right_pad = 1.80
    elif len(shown_tensors) < summary["num_sites"]:
        drawing.text(
            (last_x + 0.55, y0),
            f"+{summary['num_sites'] - len(shown_tensors)}",
            fontsize=9,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
        right_pad = 1.10

    if title is None:
        title = "Symmray MPO block structure"
    drawing.ax.set_title(title)
    charge_text = _charge_summary_text(summary)
    if charge_text and not show_diagnostics:
        drawing.text(
            (-0.55, 1.38),
            charge_text,
            fontsize=8,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
    drawing.ax.set_xlim(-0.65, last_x + right_pad)
    y_min = -1.72 if (show_phys_labels or show_blocks or show_block_labels) else -1.02
    y_max = 1.92 if (show_phys_labels or detailed_labels) else 1.42
    drawing.ax.set_ylim(y_min, y_max)
    drawing.ax.axis("off")
    if return_summary:
        return drawing, summary
    return drawing


def _as_peps_tensor_network(value):
    if hasattr(value, "tn"):
        value = value.tn
    elif hasattr(value, "state"):
        value = value.state
    if not hasattr(value, "sites") or not hasattr(value, "site_ind"):
        raise TypeError("peps must be a SymPEPS, PepsOptimizer, or quimb PEPS object.")
    if not hasattr(value, "Lx") or not hasattr(value, "Ly"):
        raise TypeError("peps must expose PEPS lattice dimensions Lx and Ly.")
    return value


def _peps_sites(tn):
    sites = tuple(getattr(tn, "sites", ()))
    if not sites and hasattr(tn, "gen_site_coos"):
        sites = tuple(tn.gen_site_coos())
    if not sites:
        sites = tuple((i, j) for i in range(int(tn.Lx)) for j in range(int(tn.Ly)))
    if not all(isinstance(site, tuple) and len(site) == 2 for site in sites):
        raise ValueError("peps sites must be two-dimensional coordinate tuples.")
    return tuple(sorted(tuple(int(x) for x in site) for site in sites))


def _peps_site_tensor(tn, site):
    try:
        return tn[site]
    except Exception as exc:  # pragma: no cover - defensive for quimb variants
        if hasattr(tn, "site_tag"):
            try:
                return tn[tn.site_tag(site)]
            except Exception:
                pass
        raise ValueError(f"Could not resolve PEPS tensor for site {site!r}.") from exc


def _peps_site_tag(tn, site):
    if hasattr(tn, "site_tag"):
        return tn.site_tag(site)
    return None


def _peps_site_ind(tn, site):
    if hasattr(tn, "site_ind"):
        return tn.site_ind(site)
    return _format_site_ind(site, getattr(tn, "site_ind_id", "k{},{}"))


def _peps_relative_direction(site_a, site_b):
    dx = int(site_b[0]) - int(site_a[0])
    dy = int(site_b[1]) - int(site_a[1])
    if dx == 0 and dy == 1:
        return "right"
    if dx == 0 and dy == -1:
        return "left"
    if dx == 1 and dy == 0:
        return "down"
    if dx == -1 and dy == 0:
        return "up"
    return "other"


def _opposite_direction(direction):
    return {
        "right": "left",
        "left": "right",
        "down": "up",
        "up": "down",
    }.get(direction, "other")


def symmray_peps_summary(peps):
    """Return site, bond, and block metadata for a Symmray-backed PEPS.

    Parameters
    ----------
    peps : SymPEPS, PepsOptimizer, or quimb PEPS
        Object whose site tensors store Symmray block-sparse arrays.

    Returns
    -------
    dict
        Summary with per-site tensor block counts, physical-sector maps,
        virtual-bond sector maps, lattice dimensions, and aggregate storage
        diagnostics.
    """
    source = peps
    tn = _as_peps_tensor_network(peps)
    sites = _peps_sites(tn)
    site_set = set(sites)

    tensors = []
    tensors_by_site = {}
    site_tensors = {}
    index_maps = {}
    physical_inds = {}
    source_edges = _explicit_source_edges(source)
    order_by_pair, edge_by_pair = (
        _edge_lookup(source_edges)
        if source_edges is not None
        else ({}, {})
    )

    for position, site in enumerate(sites):
        tensor = _peps_site_tensor(tn, site)
        array_summary = symmray_block_summary(tensor)
        site_ind = _peps_site_ind(tn, site)

        index_by_ind = {}
        indices = []
        for ind, index_summary in zip(tensor.inds, array_summary["indices"]):
            entry = dict(index_summary)
            entry["ind"] = ind
            index_by_ind[ind] = entry
            indices.append(entry)

        physical = index_by_ind.get(site_ind)
        if physical is None:
            raise ValueError(
                f"PEPS tensor for site {site!r} does not expose physical index {site_ind!r}."
            )

        entry = {
            "position": int(position),
            "site": site,
            "site_tag": _peps_site_tag(tn, site),
            "site_ind": site_ind,
            "inds": tuple(tensor.inds),
            "shape": array_summary["shape"],
            "charge": array_summary["charge"],
            "indices": indices,
            "physical": physical,
            "bonds": {},
            "blocks": array_summary["blocks"],
            "num_blocks": array_summary["num_blocks"],
            "dense_size": array_summary["dense_size"],
            "stored_size": array_summary["stored_size"],
            "density": array_summary["density"],
        }
        tensors.append(entry)
        tensors_by_site[site] = entry
        site_tensors[site] = tensor
        index_maps[site] = index_by_ind
        physical_inds[site] = site_ind

    bonds = []
    for left_pos, site_a in enumerate(sites):
        for site_b in sites[left_pos + 1 :]:
            shared_inds = _shared_virtual_inds(
                site_tensors[site_a],
                site_tensors[site_b],
                (physical_inds[site_a], physical_inds[site_b]),
            )
            for ind in shared_inds:
                index_a = index_maps[site_a].get(ind)
                index_b = index_maps[site_b].get(ind)
                index_summary = index_a or index_b
                direction = _peps_relative_direction(site_a, site_b)
                between = (site_a, site_b)
                edge_order = order_by_pair.get(between)
                bond = {
                    "position": len(bonds),
                    "site_a": site_a,
                    "site_b": site_b,
                    "between": between,
                    "edge": edge_by_pair.get(between),
                    "edge_order": edge_order,
                    "is_source_edge": edge_order is not None,
                    "ind": ind,
                    "direction": direction,
                    "site_a_direction": index_a["direction"] if index_a is not None else None,
                    "site_b_direction": index_b["direction"] if index_b is not None else None,
                    "has_complementary_directions": _directions_are_complementary(
                        index_a["direction"] if index_a is not None else None,
                        index_b["direction"] if index_b is not None else None,
                    ),
                    "dim": index_summary["dim"],
                    "num_sectors": index_summary["num_sectors"],
                    "chargemap": index_summary["chargemap"],
                    "sectors": index_summary["sectors"],
                }
                bonds.append(bond)
                tensors_by_site[site_a]["bonds"][direction] = bond
                tensors_by_site[site_b]["bonds"][_opposite_direction(direction)] = bond

    total_dense_size = int(sum(tensor["dense_size"] for tensor in tensors))
    total_stored_size = int(sum(tensor["stored_size"] for tensor in tensors))
    Lx = int(getattr(tn, "Lx", max(site[0] for site in site_set) + 1))
    Ly = int(getattr(tn, "Ly", max(site[1] for site in site_set) + 1))
    total_charge = _resolve_total_charge(source, tensors)
    symmetry = getattr(source, "symmetry", None)
    fermionic = _infer_fermionic(source, site_tensors.values())
    q_total = _resolve_q_total(symmetry, total_charge)
    return {
        "Lx": Lx,
        "Ly": Ly,
        "num_sites": len(sites),
        "sites": sites,
        "tensors": tensors,
        "bonds": bonds,
        "source_edges": (
            source_edges
            if source_edges is not None
            else tuple(tuple(bond["between"]) for bond in bonds)
        ),
        "has_source_edges": source_edges is not None,
        "num_extra_bonds": (
            sum(1 for bond in bonds if bond["edge_order"] is None)
            if source_edges is not None
            else 0
        ),
        "symmetry": symmetry,
        "fermionic": fermionic,
        "fermionic_ordering": _fermionic_ordering_summary(
            source,
            network_kind="peps",
            sites=sites,
            bonds=bonds,
            fermionic=fermionic,
        ),
        "total_charge": total_charge,
        "charge_total": total_charge,
        "Q_total": q_total,
        "total_parity": _mod_charge(total_charge, 2),
        "max_bond_dim": max((bond["dim"] for bond in bonds), default=1),
        "max_bond_sectors": max((bond["num_sectors"] for bond in bonds), default=0),
        "total_dense_size": total_dense_size,
        "total_stored_size": total_stored_size,
        "density": total_stored_size / total_dense_size if total_dense_size else 0.0,
    }


def _resolve_peps_center(tensors, value, *, Lx, Ly, name="center"):
    if value is None:
        return None
    sites = {tensor["site"] for tensor in tensors}
    if value == "middle":
        target_x = (int(Lx) - 1) / 2
        target_y = (int(Ly) - 1) / 2
        return min(
            sites,
            key=lambda site: (
                abs(site[0] - target_x) + abs(site[1] - target_y),
                site[0],
                site[1],
            ),
        )
    if value in sites:
        return tuple(value)
    raise ValueError(f"{name}={value!r} does not identify a shown PEPS site.")


def _peps_site_distance(site, center):
    return abs(int(site[0]) - int(center[0])) + abs(int(site[1]) - int(center[1]))


def _peps_primary_bond_key(bond):
    return (
        0 if bond.get("has_complementary_directions") else 1,
        0 if bond.get("direction") != "other" else 1,
        -int(bond.get("dim", 0)),
        int(bond["position"]),
    )


def _peps_display_bonds(summary, shown_sites, *, show_extra_bonds):
    bonds = [
        bond
        for bond in summary["bonds"]
        if bond["site_a"] in shown_sites and bond["site_b"] in shown_sites
    ]
    if show_extra_bonds or not summary.get("has_source_edges"):
        return bonds

    grouped = {}
    for bond in bonds:
        edge_order = bond.get("edge_order")
        if edge_order is None:
            continue
        grouped.setdefault(edge_order, []).append(bond)

    return [
        min(group, key=_peps_primary_bond_key)
        for _edge_order, group in sorted(grouped.items())
    ]


def draw_symmray_peps(
    peps,
    *,
    ax=None,
    title=None,
    max_sites=None,
    mapper=None,
    center="middle",
    show_region=True,
    show_arrows=True,
    show_leg_chargemaps=True,
    show_bond_labels=False,
    show_bond_sectors=False,
    show_extra_bonds=False,
    show_phys_labels=False,
    show_tensor_labels=True,
    show_diagnostics=False,
    show_blocks=False,
    show_block_labels=False,
    charge_in_node=True,
    max_blocks_per_site=4,
    node_shape="circle",
    node_radius=0.22,
    site_cmap="tab20",
    figsize=None,
    return_summary=False,
):
    """Draw a block-aware PEPS schematic for a Symmray-backed PEPS.

    The schematic follows the compact :mod:`quimb.schematic` style with a PEPS
    lattice, virtual-bond arrows, physical legs, and optional Symmray block and
    dimension labels.

    By default each node circle contains a compact white charge label. For
    spin-resolved two-component charges this is the total charge ``Q`` and
    spin projection ``S_z``; for other charges it shows the tensor charge ``q``
    and total particle number ``N`` where available. The node is enlarged
    automatically so the text fits. Set ``charge_in_node=False`` to keep the
    charge outside the node with the tensor label.

    ``show_bond_labels=True`` annotates only the edge id, charge-flow
    directions, and bond dimension. Set ``show_bond_sectors=True`` to add the
    compact bond charge-sector maps. When the input is a ``SymPEPS`` wrapper,
    only one primary shared index per configured ``edges`` entry is drawn by
    default; set ``show_extra_bonds=True`` to debug all shared virtual indices,
    including non-lattice and multibond indices introduced by routing/gauges.
    """
    if _is_mpo_like(peps):
        return draw_symmray_mpo(
            peps,
            ax=ax,
            title=title,
            max_sites=max_sites,
            mapper=mapper,
            center=center,
            show_arrows=show_arrows,
            show_leg_chargemaps=show_leg_chargemaps,
            show_bond_labels=show_bond_labels,
            show_phys_labels=show_phys_labels,
            show_tensor_labels=show_tensor_labels,
            show_diagnostics=show_diagnostics,
            show_blocks=show_blocks,
            show_block_labels=show_block_labels,
            max_blocks_per_site=max_blocks_per_site,
            node_shape=node_shape,
            node_radius=node_radius,
            site_cmap=site_cmap,
            figsize=figsize,
            return_summary=return_summary,
        )
    if _is_mps_like_not_peps(peps):
        return draw_symmray_mps(
            peps,
            ax=ax,
            title=title,
            max_sites=max_sites,
            mapper=mapper,
            center=center,
            show_regions=show_region,
            show_arrows=show_arrows,
            show_leg_chargemaps=show_leg_chargemaps,
            show_bond_labels=show_bond_labels,
            show_phys_labels=show_phys_labels,
            show_tensor_labels=show_tensor_labels,
            show_diagnostics=show_diagnostics,
            show_blocks=show_blocks,
            show_block_labels=show_block_labels,
            max_blocks_per_site=max_blocks_per_site,
            node_shape=node_shape,
            node_radius=node_radius,
            site_cmap=site_cmap,
            figsize=figsize,
            return_summary=return_summary,
        )

    if mapper is not None:
        raise TypeError(
            "mapper is only supported for MPS/MPO inputs; PEPS inputs already "
            "carry two-dimensional lattice coordinates."
        )

    summary = symmray_peps_summary(peps)
    max_blocks_per_site = int(max_blocks_per_site)
    if max_blocks_per_site < 1:
        raise ValueError("max_blocks_per_site must be >= 1.")
    if max_sites is None:
        shown_tensors = list(summary["tensors"])
    else:
        max_sites = int(max_sites)
        if max_sites < 1:
            raise ValueError("max_sites must be >= 1.")
        shown_tensors = list(summary["tensors"][:max_sites])

    shown_sites = {tensor["site"] for tensor in shown_tensors}
    center_site = _resolve_peps_center(
        shown_tensors,
        center,
        Lx=summary["Lx"],
        Ly=summary["Ly"],
        name="center",
    )
    show_bond_sectors = bool(show_bond_sectors)
    show_extra_bonds = bool(show_extra_bonds)
    shown_bonds = _peps_display_bonds(
        summary,
        shown_sites,
        show_extra_bonds=show_extra_bonds,
    )
    hidden_bond_count = sum(
        1
        for bond in summary["bonds"]
        if bond["site_a"] in shown_sites and bond["site_b"] in shown_sites
    ) - len(shown_bonds)

    try:
        from quimb import schematic  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - quimb is a declared dep
        raise ImportError("draw_symmray_peps requires quimb.schematic.") from exc

    try:
        from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - plotting optionality
        raise ImportError("draw_symmray_peps requires matplotlib.") from exc

    spacing = 1.16
    if figsize is None:
        width = max(4.8, 1.15 * max(1, summary["Ly"]) + 1.8)
        height = max(4.4, 1.20 * max(1, summary["Lx"]) + 1.9)
        if show_bond_labels or show_bond_sectors or show_phys_labels or show_blocks:
            detailed_leg_maps = bool(
                show_bond_sectors or (show_phys_labels and show_leg_chargemaps)
            )
            width += 0.85 if detailed_leg_maps else 0.55
            height += 0.55 if detailed_leg_maps else 0.35
        if show_diagnostics:
            width += 1.35
        figsize = (width, height)

    presets = {
        "bond": {
            "color": (0.12, 0.14, 0.16, 1.0),
            "linewidth": 3.0,
        },
        "phys": {
            "color": (0.12, 0.14, 0.16, 1.0),
            "linewidth": 1.45,
        },
        "center": {
            "color": schematic.get_color("orange"),
            "hatch": "/////",
            "linewidth": 1.2,
            "edgecolor": (0.55, 0.38, 0.00, 1.0),
        },
        "site_a": {
            "color": schematic.get_color("bluedark"),
            "linewidth": 1.2,
            "edgecolor": (0.02, 0.22, 0.34, 1.0),
        },
        "site_b": {
            "color": schematic.get_color("blue"),
            "linewidth": 1.2,
            "edgecolor": (0.05, 0.34, 0.50, 1.0),
        },
        "region": {
            "facecolor": (0.83, 0.83, 0.83, 0.38),
            "edgecolor": (0.42, 0.42, 0.42, 0.72),
            "linestyle": ":",
            "linewidth": 1.35,
        },
    }
    drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

    xy_by_site = {
        tensor["site"]: (tensor["site"][1] * spacing, -tensor["site"][0] * spacing)
        for tensor in shown_tensors
    }
    max_dim = max((bond["dim"] for bond in shown_bonds), default=1)

    if show_region and shown_tensors:
        drawing.patch_around(list(xy_by_site.values()), radius=0.42, preset="region", zorder=0)

    for bond in shown_bonds:
        xy_a = xy_by_site[bond["site_a"]]
        xy_b = xy_by_site[bond["site_b"]]
        width = 2.25 + 1.05 * (bond["dim"] / max_dim)
        drawing.line(xy_a, xy_b, preset="bond", linewidth=width, zorder=1)

        if show_arrows:
            start, stop = xy_a, xy_b
            if center_site is not None:
                dist_a = _peps_site_distance(bond["site_a"], center_site)
                dist_b = _peps_site_distance(bond["site_b"], center_site)
                if dist_a < dist_b:
                    start, stop = xy_b, xy_a
            drawing.arrowhead(start, stop, preset="bond", center=0.58, width=0.055, length=0.11)

        if show_bond_labels or show_bond_sectors:
            mid = (0.5 * (xy_a[0] + xy_b[0]), 0.5 * (xy_a[1] + xy_b[1]))
            offset = (0.0, 0.16) if bond["direction"] in {"left", "right"} else (0.17, 0.0)
            flow = _flow_math(bond.get("site_a_direction"), bond.get("site_b_direction"))
            label_lines = []
            if show_bond_labels:
                if flow:
                    label_lines.append(
                        rf"$e_{{{bond['position']}}}: {flow}, \chi={bond['dim']}$"
                    )
                else:
                    label_lines.append(rf"$e_{{{bond['position']}}}: \chi={bond['dim']}$")
            if show_bond_sectors:
                label_lines.append(
                    rf"$q_e:$ {_format_compact_mapping(bond['chargemap'], max_items=4)}"
                )
            drawing.text(
                (mid[0] + offset[0], mid[1] + offset[1]),
                "\n".join(label_lines),
                fontsize=6.2,
                ha="center" if offset[0] == 0.0 else "left",
                va="bottom" if offset[1] > 0.0 else "center",
                color=(0.18, 0.20, 0.23, 1.0),
                bbox=(
                    {
                        "boxstyle": "round,pad=0.05",
                        "facecolor": (1.0, 1.0, 1.0, 0.72),
                        "edgecolor": (1.0, 1.0, 1.0, 0.0),
                        "linewidth": 0.0,
                    }
                    if show_bond_sectors
                    else None
                ),
                zorder=5,
            )

    show_block_labels = bool(show_block_labels)
    charge_in_node = bool(charge_in_node)
    node_text_radius = node_radius
    if charge_in_node and node_shape == "circle":
        node_text_radius = max(node_radius, 0.32)
    for tensor in shown_tensors:
        site = tensor["site"]
        x_pos, y_pos = xy_by_site[site]
        preset = "center" if site == center_site else ("site_a" if sum(site) % 2 == 0 else "site_b")

        if node_shape == "circle":
            drawing.circle((x_pos, y_pos), radius=node_text_radius, preset=preset, zorder=3)
        elif node_shape == "cube":
            drawing.cube((x_pos, y_pos, 0.0), preset=preset, zorder=3)
        else:
            raise ValueError("node_shape must be 'circle' or 'cube'.")

        if charge_in_node and tensor["charge"] is not None:
            node_lines = _node_charge_label_lines(tensor["charge"])
            drawing.text(
                (x_pos, y_pos),
                "\n".join(node_lines),
                fontsize=6.4,
                ha="center",
                va="center",
                color=(1.0, 1.0, 1.0, 1.0),
                zorder=4,
            )

        phys_xy = (x_pos - 0.24, y_pos - 0.34)
        drawing.line((x_pos, y_pos), phys_xy, preset="phys", zorder=1)
        if show_arrows:
            drawing.arrowhead(
                phys_xy,
                (x_pos, y_pos),
                preset="phys",
                center=0.56,
                width=0.038,
                length=0.075,
            )

        if show_tensor_labels:
            label_lines = [rf"$T_{{({site[0]},{site[1]})}}$"]
            if tensor["charge"] is not None and not charge_in_node:
                label_lines.append(rf"$q={_format_charge(tensor['charge'])}$")
            if not show_blocks:
                label_lines.append(rf"$B={tensor['num_blocks']}$")
            label = "\n".join(label_lines)
            label_color = (
                schematic.get_color("orange")
                if site == center_site
                else (0.06, 0.20, 0.30, 1.0)
            )
            drawing.text(
                (x_pos, y_pos + node_text_radius + 0.12),
                label,
                fontsize=7.5,
                ha="center",
                va="bottom",
                color=label_color,
                zorder=5,
            )

        physical = tensor["physical"]
        if show_phys_labels:
            phys_label = (
                rf"$p_{{({site[0]},{site[1]})}}: \mathrm{{{physical['direction']}}}, d={physical['dim']}$"
            )
            if show_leg_chargemaps:
                phys_label += "\n" + rf"$q_p:$ {_format_compact_mapping(physical['chargemap'], max_items=4)}"
            drawing.text(
                (phys_xy[0] - 0.05, phys_xy[1] - 0.10),
                phys_label,
                fontsize=5.8,
                ha="right",
                va="top",
                color=(0.18, 0.20, 0.23, 1.0),
            )

        if show_blocks and tensor["blocks"]:
            blocks = tensor["blocks"][:max_blocks_per_site]
            block_width = 0.15
            block_height = 0.12
            gap = 0.032
            total_width = len(blocks) * block_width + (len(blocks) - 1) * gap
            start = x_pos - 0.5 * total_width
            block_y = y_pos - node_text_radius - 0.12
            for block_pos, block in enumerate(blocks):
                bx = start + block_pos * (block_width + gap)
                color = schematic.hash_to_color(str(block["sector"]))
                drawing.ax.add_patch(
                    Rectangle(
                        (bx, block_y),
                        block_width,
                        block_height,
                        facecolor=color,
                        edgecolor=(0.16, 0.18, 0.21, 0.75),
                        alpha=0.86,
                        linewidth=0.60,
                        zorder=4,
                    )
                )
                if show_block_labels:
                    drawing.ax.text(
                        bx + 0.5 * block_width,
                        block_y - 0.04,
                        _format_sector(block["sector"]),
                        fontsize=4.8,
                        ha="center",
                        va="top",
                        rotation=45,
                        color=(0.15, 0.17, 0.20, 1.0),
                    )
            drawing.ax.text(
                x_pos,
                block_y + block_height + 0.022,
                rf"$B={tensor['num_blocks']}$",
                fontsize=5.8,
                ha="center",
                va="bottom",
                color=(0.15, 0.17, 0.20, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.05",
                    "facecolor": (1.0, 1.0, 1.0, 0.78),
                    "edgecolor": (1.0, 1.0, 1.0, 0.0),
                    "linewidth": 0.0,
                },
                zorder=5,
            )

    xs = [xy[0] for xy in xy_by_site.values()] or [0.0]
    ys = [xy[1] for xy in xy_by_site.values()] or [0.0]
    right_pad = 0.78
    if show_diagnostics:
        charge_line = _charge_summary_text(summary)
        diagnostic_lines = [f"sites {summary['num_sites']}"]
        if charge_line:
            diagnostic_lines.append(charge_line)
        diagnostic_lines += [
            f"max bond {summary['max_bond_dim']}",
            f"bond sectors {summary['max_bond_sectors']}",
            f"stored {summary['total_stored_size']}/{summary['total_dense_size']}",
            f"density {summary['density']:.3f}",
        ]
        diagnostic = "\n".join(diagnostic_lines)
        if show_blocks:
            diagnostic += "\ncolored tiles: stored blocks"
        if show_arrows:
            diagnostic += "\narrows/labels: charge in/out flow"
        if hidden_bond_count:
            diagnostic += f"\n{hidden_bond_count} extra bonds hidden"
        if len(shown_tensors) < summary["num_sites"]:
            diagnostic += f"\n+{summary['num_sites'] - len(shown_tensors)} sites hidden"
        drawing.ax.text(
            max(xs) + 0.82,
            max(ys) + 0.46,
            diagnostic,
            fontsize=8,
            ha="left",
            va="top",
            color=(0.15, 0.17, 0.20, 1.0),
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": (1.0, 1.0, 1.0, 0.92),
                "edgecolor": (0.68, 0.70, 0.74, 1.0),
                "linewidth": 0.8,
            },
        )
        right_pad = 1.85
    elif len(shown_tensors) < summary["num_sites"]:
        drawing.text(
            (max(xs) + 0.55, min(ys)),
            f"+{summary['num_sites'] - len(shown_tensors)}",
            fontsize=9,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
        right_pad = 1.10

    if title is None:
        title = "Symmray PEPS block structure"
    drawing.ax.set_title(title)
    charge_text = _charge_summary_text(summary)
    if charge_text and not show_diagnostics:
        drawing.text(
            (min(xs) - 0.68, max(ys) + 0.58),
            charge_text,
            fontsize=8,
            ha="left",
            va="center",
            color=(0.18, 0.20, 0.23, 1.0),
        )
    drawing.ax.set_xlim(min(xs) - 0.82, max(xs) + right_pad)
    y_min = min(ys) - (0.92 if show_phys_labels else 0.76)
    y_max = max(ys) + 0.76
    drawing.ax.set_ylim(y_min, y_max)
    drawing.ax.axis("off")
    if return_summary:
        return drawing, summary
    return drawing


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
    returns the Strang step ``U_int(dt/2) U_hop(dt) U_int(dt/2)``. The onsite
    and hopping gates are exact fermionic Symmray arrays, so no spin/qubit
    mapping is introduced.
    """
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2.")
    edges = _as_edges(edges)
    sites = _sites_from_edges(edges, sites)
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
    ``U_int(dt/2) U_hop(dt) U_int(dt/2)``. All gates are bosonic Symmray arrays
    that act on a Jordan-Wigner MPS; hopping bonds must be nearest-neighbour.
    """
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2.")
    edges = _as_edges(edges)
    sites = _sites_from_edges(edges, sites)
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
        from .core import OneDMap  # pylint: disable=import-outside-toplevel

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
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        item = getattr(value, "item", None)
        if callable(item):
            return item()
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
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    # CuPy arrays reject implicit host conversion; move to host explicitly.
    if type(value).__module__.split(".", 1)[0] == "cupy":
        value = value.get()
    return np.asarray(value, dtype=dtype)


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
            for channel_id in channel_ids:
                transitions[site].append((channel_id, channel_id, identity))

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
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2.")
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


def _fermion_backend_anchor(*values):
    """Return an array-like value suitable for an Autoray backend context."""
    for value in values:
        if hasattr(value, "shape") and not isinstance(value, (str, bytes)):
            return value
    return np.asarray(0.0j)


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
            gate = _gate_from_term(term, theta, imaginary=imaginary)
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
        theta = dt * U

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
        theta = dt * V

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
            dt=dt,
            imaginary=imaginary,
            like=like,
        )
        return _apply_to_array_blocks(gate, self.to_backend)

    local_exponential = exponential

    @staticmethod
    def edge_coloring_layers(edges):
        """Partition edges into deterministic vertex-disjoint layers."""
        edges = _as_edges(edges)
        layers = []
        occupied_sites = []
        for edge in edges:
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
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2.")
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

        ``tree`` and ``plan`` are aliases.  The returned object exposes the
        optional linear representation as ``.chain_mpo`` and the TreePlan
        representation through ``.tree_networks`` and ``.expectation``.
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


# Compatibility spelling for the initial, overly model-specific public name.
SpinfulFermionHubbard = SpinfulFermion


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

        # Some supported quimb/symmray combinations route projector/CTMRG
        # boundary compression through a dense reshape that Symmray's
        # BlockVector does not implement. The MPS boundary path computes the
        # same local expectation while retaining the native operator blocks.
        if mode_local == "projector" and self._is_symmray_array(next(iter(terms.values()))):
            mode_local = "mps"

        value = self.psi.compute_local_expectation(
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
        return int(self.psi.Lx) * int(self.psi.Ly)

    @property
    def Lx(self):
        """PEPS x dimension."""
        return int(self.psi.Lx)

    @property
    def Ly(self):
        """PEPS y dimension."""
        return int(self.psi.Ly)
