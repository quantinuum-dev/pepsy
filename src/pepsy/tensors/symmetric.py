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
    "draw_symmray_blocks",
    "draw_symmray_mps",
    "draw_symmray_peps",
    "sector_index_map",
    "site_charge_alternating",
    "site_charge_from_map",
    "site_charge_from_occupations",
    "site_charge_uniform",
    "symmray_block_summary",
    "symmray_mps_summary",
    "symmray_peps_summary",
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
        size = int(getattr(block, "size", _shape_size(shape)))
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


def _resolve_mps_position(tensors, value, *, name="position"):
    if value is None:
        return None
    if value == "middle":
        return tensors[len(tensors) // 2]["position"] if tensors else None
    for tensor in tensors:
        if value == tensor["position"] or value == tensor["site"]:
            return tensor["position"]
    raise ValueError(f"{name}={value!r} does not identify a shown MPS site.")


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
    symmetry = getattr(source, "symmetry", None)
    q_total = _resolve_q_total(symmetry, total_charge)
    return {
        "num_sites": len(sites),
        "sites": sites,
        "tensors": tensors,
        "bonds": bonds,
        "symmetry": symmetry,
        "fermionic": getattr(source, "fermionic", None),
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


def draw_symmray_mps(
    mps,
    *,
    ax=None,
    title=None,
    max_sites=None,
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
    figsize=None,
    return_summary=False,
):
    """Draw a block-aware MPS schematic for a Symmray-backed MPS.

    The diagram uses :mod:`quimb.schematic` in the same style as quimb's manual
    tensor-network schematics: tensor nodes, virtual bonds, physical legs,
    optional canonical-flow arrows, and optional left/center/right region
    highlighting. Symmray-specific dimensions remain available through
    ``return_summary=True`` and can be overlaid with the label options.
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
                drawing.arrowhead((x0, y0), (x1, y0), preset="bond", center=0.58, width=0.10)
            elif bond["right_position"] <= center_position:
                drawing.arrowhead((x0, y0), (x1, y0), preset="bond", center=0.58, width=0.10)
            elif (
                pair_right_position is not None
                and bond["left_position"] >= pair_right_position
            ) or (
                pair_right_position is None
                and bond["left_position"] >= center_position
            ):
                drawing.arrowhead((x1, y0), (x0, y0), preset="bond", center=0.58, width=0.10)

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
                width=0.09,
            )
        drawing.dot(
            (x_pos, phys_y),
            color=(0.12, 0.14, 0.16, 1.0),
            radius=0.028,
            zorder=2,
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
                bond = {
                    "position": len(bonds),
                    "site_a": site_a,
                    "site_b": site_b,
                    "between": (site_a, site_b),
                    "ind": ind,
                    "direction": direction,
                    "site_a_direction": index_a["direction"] if index_a is not None else None,
                    "site_b_direction": index_b["direction"] if index_b is not None else None,
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
    q_total = _resolve_q_total(symmetry, total_charge)
    return {
        "Lx": Lx,
        "Ly": Ly,
        "num_sites": len(sites),
        "sites": sites,
        "tensors": tensors,
        "bonds": bonds,
        "symmetry": symmetry,
        "fermionic": getattr(source, "fermionic", None),
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


def draw_symmray_peps(
    peps,
    *,
    ax=None,
    title=None,
    max_sites=None,
    center="middle",
    show_region=True,
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
    node_radius=0.22,
    figsize=None,
    return_summary=False,
):
    """Draw a block-aware PEPS schematic for a Symmray-backed PEPS.

    The schematic follows the compact :mod:`quimb.schematic` style with a PEPS
    lattice, virtual-bond arrows, physical legs, and optional Symmray block and
    dimension labels.
    """
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
    shown_bonds = [
        bond
        for bond in summary["bonds"]
        if bond["site_a"] in shown_sites and bond["site_b"] in shown_sites
    ]

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
        if show_bond_labels or show_phys_labels or show_blocks:
            width += 0.85 if show_leg_chargemaps else 0.55
            height += 0.55 if show_leg_chargemaps else 0.35
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
            drawing.arrowhead(start, stop, preset="bond", center=0.58, width=0.10)

        if show_bond_labels:
            mid = (0.5 * (xy_a[0] + xy_b[0]), 0.5 * (xy_a[1] + xy_b[1]))
            offset = (0.0, 0.16) if bond["direction"] in {"left", "right"} else (0.17, 0.0)
            flow = _flow_math(bond.get("site_a_direction"), bond.get("site_b_direction"))
            if flow:
                label = rf"$e_{{{bond['position']}}}: {flow}, \chi={bond['dim']}$"
            else:
                label = rf"$e_{{{bond['position']}}}: \chi={bond['dim']}$"
            if show_leg_chargemaps:
                label += "\n" + rf"$q_e:$ {_format_compact_mapping(bond['chargemap'], max_items=4)}"
            drawing.text(
                (mid[0] + offset[0], mid[1] + offset[1]),
                label,
                fontsize=6.2,
                ha="center" if offset[0] == 0.0 else "left",
                va="bottom" if offset[1] > 0.0 else "center",
                color=(0.18, 0.20, 0.23, 1.0),
                zorder=5,
            )

    show_block_labels = bool(show_block_labels)
    for tensor in shown_tensors:
        site = tensor["site"]
        x_pos, y_pos = xy_by_site[site]
        preset = "center" if site == center_site else ("site_a" if sum(site) % 2 == 0 else "site_b")

        if node_shape == "circle":
            drawing.circle((x_pos, y_pos), radius=node_radius, preset=preset, zorder=3)
        elif node_shape == "cube":
            drawing.cube((x_pos, y_pos, 0.0), preset=preset, zorder=3)
        else:
            raise ValueError("node_shape must be 'circle' or 'cube'.")

        phys_xy = (x_pos - 0.30, y_pos - 0.42)
        drawing.line((x_pos, y_pos), phys_xy, preset="phys", zorder=1)
        if show_arrows:
            drawing.arrowhead(phys_xy, (x_pos, y_pos), preset="phys", center=0.56, width=0.08)
        drawing.dot(
            phys_xy,
            color=(0.12, 0.14, 0.16, 1.0),
            radius=0.025,
            zorder=2,
        )

        if show_tensor_labels:
            label_lines = [rf"$T_{{({site[0]},{site[1]})}}$"]
            if tensor["charge"] is not None:
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
                (x_pos, y_pos + 0.34),
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
            block_y = y_pos - 0.34
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
    if model in {"fermi_hubbard", "fermi_hubbard_u1u1"}:
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
