"""Block, charge, ordering, and drawing diagnostics for native symmetric tensors.

This module inspects arrays and tensor networks without owning model or state
construction. Plotting dependencies are loaded only by the drawing functions.
"""

from __future__ import annotations

import numpy as np

from .symmetric import (
    _as_edges,
    _as_tuple,
    _charge_particle_number,
    _format_site_ind,
    _is_symmray_array,
)

__all__ = [
    "draw_symmray_blocks",
    "draw_symmray_mpo",
    "draw_symmray_mps",
    "draw_symmray_peps",
    "symmray_block_summary",
    "symmray_mpo_summary",
    "symmray_mps_summary",
    "symmray_peps_summary",
]


_FERMIONIC_TN_METHODS_REFERENCE = {
    "title": "Fermionic tensor network contraction for arbitrary geometries",
    "doi": "10.1103/PhysRevResearch.7.023193",
    "url": "https://doi.org/10.1103/PhysRevResearch.7.023193",
}


def _require_symmray_array(value, *, name="value"):
    if _is_symmray_array(getattr(value, "data", None)):
        return value.data
    if not _is_symmray_array(value):
        raise TypeError(f"{name} must be a Symmray block-sparse array.")
    return value


def _mapping_items(mapping):
    return sorted(mapping.items(), key=lambda item: repr(item[0]))


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

    from .maps import OneDMap  # pylint: disable=import-outside-toplevel

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
