"""Shared DMRG backend, optimizer, and fidelity helpers."""

import math
import warnings
from numbers import Integral
from string import Formatter
from typing import Any
import autoray as ar
import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from .validation import validate_tensor_network_tags

__all__ = [
    "OneDMap",
    "backend_torch",
    "backend_numpy",
    "backend_cupy",
    "backend_jax",
    "register_torch_linalg",
    "reg_rel_svd_torch",
    "reg_real_svd_torch",
    "reg_complex_svd_torch",
    "reg_real_qr_torch",
    "reg_complex_qr_torch",
    "reg_rel_svd_jax",
    "reg_real_svd_jax",
    "reg_complex_svd_jax",
    "reg_stop_gradient_torch",
    "stop_grad",
    "set_default_array_backend",
    "get_default_array_backend",
    "set_default_grad_backend",
    "get_default_grad_backend",
    "reset_default_backends",
    "build_optimizer",
    "build_compressed_optimizer",
    "contract_hypercompressed_tn",
    "contract_hypercompressed_tn_batch",
    "tn_fidelity",
    "tn_norm",
    "measure_obs",
    "tns_align",
    "expec_mpo",
    "id_to_mpo",
    "ps_to_peps",
    "ps_to_3dpeps",
    "ps_to_mps",
    "ps_to_pepo",
    "ps_to_mpo",
    "haar_random_state",
    "random_haar_qubit",
    "hrps_to_peps",
    "hrps_to_mps",
    "id_to_pepo",
    "add_cycle",
]

_DEFAULT_ARRAY_BACKEND = None
_DEFAULT_GRAD_BACKEND = None

_ONE_D_MAP_UNSET = object()


class _dualmethod:
    """Descriptor that supports both class-style and instance-style calls."""

    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = getattr(fn, "__doc__", None)

    def __get__(self, obj, cls):
        def _bound(*args, **kwargs):
            target = cls if obj is None else obj
            return self.fn(target, *args, **kwargs)

        _bound.__doc__ = self.__doc__
        _bound.__name__ = getattr(self.fn, "__name__", "dualmethod")
        return _bound



class OneDMap:
    """Build 1D chain-index maps for 2D or 3D regular lattices."""

    _KNOWN_MODES = (
        "snake",
        "snake-row-major",
        "folded-snake",
        "folded-snake-row-major",
        "row-major",
        "col-major",
        "hilbert",
        "hilbert-row-major",
        "diag",
    )
    _MODE_ALIASES = {
        "snake": "snake",
        "snake-col": "snake",
        "snake-column": "snake",
        "snake-col-major": "snake",
        "snake-column-major": "snake",
        "snake-row": "snake-row-major",
        "snake-row-major": "snake-row-major",
        "row-snake": "snake-row-major",
        "folded-snake": "folded-snake",
        "folded-snake-col": "folded-snake",
        "folded-snake-column": "folded-snake",
        "folded-snake-col-major": "folded-snake",
        "folded-snake-column-major": "folded-snake",
        "periodic-snake": "folded-snake",
        "torus-snake": "folded-snake",
        "folded-snake-row": "folded-snake-row-major",
        "folded-snake-row-major": "folded-snake-row-major",
        "periodic-snake-row": "folded-snake-row-major",
        "periodic-snake-row-major": "folded-snake-row-major",
        "torus-snake-row": "folded-snake-row-major",
        "torus-snake-row-major": "folded-snake-row-major",
        "row-major": "row-major",
        "col-major": "col-major",
        "hilbert": "hilbert",
        "hilbert-curve": "hilbert",
        "hilbert-col": "hilbert",
        "hilbert-column": "hilbert",
        "hilbert-col-major": "hilbert",
        "hilbert-column-major": "hilbert",
        "hilbert-row": "hilbert-row-major",
        "hilbert-row-major": "hilbert-row-major",
        "diag": "diag",
        "diagonal": "diag",
        "diag-snake": "diag",
    }

    @classmethod
    def _coalesce_dim_names(
        cls,
        *,
        Lx=None,
        Ly=None,
        Lz=_ONE_D_MAP_UNSET,
        L_x=None,
        L_y=None,
        L_z=_ONE_D_MAP_UNSET,
    ):
        if Lx is None:
            Lx = L_x
        elif L_x is not None and L_x != Lx:
            raise TypeError("Got both Lx and L_x with different values.")

        if Ly is None:
            Ly = L_y
        elif L_y is not None and L_y != Ly:
            raise TypeError("Got both Ly and L_y with different values.")

        if Lz is _ONE_D_MAP_UNSET:
            Lz = L_z
        elif L_z is not _ONE_D_MAP_UNSET and L_z != Lz:
            raise TypeError("Got both Lz and L_z with different values.")

        return Lx, Ly, Lz

    def __init__(self, Lx=None, Ly=None, Lz=None, mode="snake", *, L_x=None, L_y=None, L_z=_ONE_D_MAP_UNSET):
        Lx, Ly, Lz = self._coalesce_dim_names(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            L_x=L_x,
            L_y=L_y,
            L_z=L_z,
        )
        self.Lx, self.Ly, self.Lz = self._normalize_dims(Lx, Ly, Lz=Lz)
        self.L_x, self.L_y, self.L_z = self.Lx, self.Ly, self.Lz
        self.mode = self._normalize_mode(mode)

    def __repr__(self):
        shape = (self.Lx, self.Ly) if self.Lz is None else (self.Lx, self.Ly, self.Lz)
        return f"OneDMap(shape={shape}, mode={self.mode!r})"

    @property
    def shape(self):
        return (self.Lx, self.Ly) if self.Lz is None else (self.Lx, self.Ly, self.Lz)

    @classmethod
    def _resolve_call_params(
        cls,
        target,
        Lx=None,
        Ly=None,
        Lz=_ONE_D_MAP_UNSET,
        mode=None,
        *,
        L_x=None,
        L_y=None,
        L_z=_ONE_D_MAP_UNSET,
    ):
        if isinstance(target, cls):
            Lx, Ly, Lz = cls._coalesce_dim_names(
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                L_x=L_x,
                L_y=L_y,
                L_z=L_z,
            )
            if Lx is None:
                Lx = target.Lx
            if Ly is None:
                Ly = target.Ly
            if Lz is _ONE_D_MAP_UNSET:
                Lz = target.Lz
            mode = target.mode if mode is None else cls._normalize_mode(mode)
        else:
            Lx, Ly, Lz = cls._coalesce_dim_names(
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                L_x=L_x,
                L_y=L_y,
                L_z=L_z,
            )
            if Lx is None or Ly is None:
                raise TypeError(
                    "OneDMap.build/show called on the class requires Lx and Ly. "
                    "Use OneDMap(Lx, Ly, ...).build()/show() for instance-style access."
                )
            if Lz is _ONE_D_MAP_UNSET:
                Lz = None
            mode = "snake" if mode is None else cls._normalize_mode(mode)

        Lx, Ly, Lz = cls._normalize_dims(Lx, Ly, Lz=Lz)
        return Lx, Ly, Lz, mode

    @staticmethod
    def _normalize_dims(Lx, Ly, Lz=None):
        for name, value in (("Lx", Lx), ("Ly", Ly)):
            if not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1.")

        if Lz is None:
            return int(Lx), int(Ly), None
        if not isinstance(Lz, Integral):
            raise TypeError("Lz must be an integer or None.")
        if int(Lz) < 1:
            raise ValueError("Lz must be >= 1 when provided.")
        return int(Lx), int(Ly), int(Lz)

    @staticmethod
    def _coords_diag_2d(L_x, L_y):
        """Diagonal (anti-diagonal) traversal: sweep lines of constant
        ``x + y`` from 0 to ``L_x + L_y - 2``, snaking direction each
        stripe.  Natural 1-D ordering for triangular / square-plus-diagonal
        lattices."""
        coords = []
        for s in range(L_x + L_y - 1):
            diag = []
            for x in range(max(0, s - L_y + 1), min(s + 1, L_x)):
                y = s - x
                diag.append((x, y))
            if s % 2 == 1:
                diag.reverse()
            coords.extend(diag)
        return coords

    @staticmethod
    def _coords_row_major_2d(L_x, L_y):
        return [(x, y) for x in range(L_x) for y in range(L_y)]

    @staticmethod
    def _coords_row_major_3d(L_x, L_y, L_z):
        return [(x, y, z) for z in range(L_z) for x in range(L_x) for y in range(L_y)]

    @staticmethod
    def _coords_col_major_2d(L_x, L_y):
        return [(x, y) for y in range(L_y) for x in range(L_x)]

    @staticmethod
    def _coords_col_major_3d(L_x, L_y, L_z):
        return [(x, y, z) for z in range(L_z) for y in range(L_y) for x in range(L_x)]

    @staticmethod
    def _coords_snake_2d(L_x, L_y, *, major="col"):
        coords = []
        if major == "col":
            for x in range(L_x):
                y_iter = range(L_y) if (x % 2 == 0) else range(L_y - 1, -1, -1)
                for y in y_iter:
                    coords.append((x, y))
            return coords
        if major == "row":
            for y in range(L_y):
                x_iter = range(L_x) if (y % 2 == 0) else range(L_x - 1, -1, -1)
                for x in x_iter:
                    coords.append((x, y))
            return coords
        raise ValueError(f"Unknown snake major axis: {major!r}.")

    @staticmethod
    def _folded_axis_order(length):
        """Return ``0, L-1, 1, L-2, ...`` for periodic boundary mappings."""
        left = 0
        right = int(length) - 1
        order = []
        while left <= right:
            order.append(left)
            if left != right:
                order.append(right)
            left += 1
            right -= 1
        return order

    @classmethod
    def _coords_folded_snake_2d(cls, L_x, L_y, *, major="col"):
        coords = []
        if major == "col":
            for step, x in enumerate(cls._folded_axis_order(L_x)):
                y_iter = range(L_y) if (step % 2 == 0) else range(L_y - 1, -1, -1)
                for y in y_iter:
                    coords.append((x, y))
            return coords
        if major == "row":
            for step, y in enumerate(cls._folded_axis_order(L_y)):
                x_iter = range(L_x) if (step % 2 == 0) else range(L_x - 1, -1, -1)
                for x in x_iter:
                    coords.append((x, y))
            return coords
        raise ValueError(f"Unknown folded snake major axis: {major!r}.")

    @staticmethod
    def _is_power_of_two(value):
        return value > 0 and (value & (value - 1)) == 0

    @staticmethod
    def _next_power_of_two(value):
        value = int(value)
        if value < 1:
            raise ValueError("value must be >= 1.")
        return 1 << (value - 1).bit_length()

    @staticmethod
    def _hilbert_rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    @classmethod
    def _coords_hilbert_2d_base(cls, L_x, L_y):
        # For rectangles, traverse the smallest enclosing power-of-two square
        # in Hilbert order and keep only points inside the requested bounds.
        side = cls._next_power_of_two(max(L_x, L_y))
        coords = []
        for distance in range(side * side):
            x = 0
            y = 0
            t = distance
            scale = 1
            while scale < side:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                x, y = cls._hilbert_rot(scale, x, y, rx, ry)
                x += scale * rx
                y += scale * ry
                t //= 4
                scale *= 2
            if x < L_x and y < L_y:
                coords.append((x, y))
                if len(coords) == L_x * L_y:
                    break
        return coords

    @classmethod
    def _coords_hilbert_2d(cls, L_x, L_y, *, major="col"):
        if major == "col":
            return cls._coords_hilbert_2d_base(L_x, L_y)
        if major == "row":
            return [(y, x) for x, y in cls._coords_hilbert_2d_base(L_y, L_x)]
        raise ValueError(f"Unknown hilbert major axis: {major!r}.")

    @classmethod
    def _coords_snake_3d(cls, L_x, L_y, L_z, *, major="col"):
        coords = []
        for z in range(L_z):
            layer = cls._coords_snake_2d(L_x, L_y, major=major)
            if z % 2 == 1:
                layer.reverse()
            for x, y in layer:
                coords.append((x, y, z))
        return coords

    @staticmethod
    def _coords_to_maps(coords):
        one_d_to_lattice = {idx: coord for idx, coord in enumerate(coords)}
        lattice_to_one_d = {coord: idx for idx, coord in one_d_to_lattice.items()}
        return one_d_to_lattice, lattice_to_one_d

    @classmethod
    def _normalize_mode(cls, mode):
        mode_norm = str(mode).strip().lower().replace("_", "-")
        return cls._MODE_ALIASES.get(mode_norm, mode_norm)

    @_dualmethod
    def build(target, Lx=None, Ly=None, Lz=_ONE_D_MAP_UNSET, mode=None, *, L_x=None, L_y=None, L_z=_ONE_D_MAP_UNSET):
        """Build ``(one_d_to_lattice, lattice_to_one_d)`` for a traversal mode.

        This can be called either as ``OneDMap.build(Lx, Ly, ...)`` or on an
        instance, e.g. ``OneDMap(Lx, Ly, mode="row-major").build()``.
        Instance calls can override options per use, for example
        ``mapper.build(mode="snake")``.
        """
        cls = target if isinstance(target, type) else type(target)
        Lx, Ly, Lz, mode_norm = cls._resolve_call_params(
            target,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            mode=mode,
            L_x=L_x,
            L_y=L_y,
            L_z=L_z,
        )

        if mode_norm == "snake":
            coords = (
                cls._coords_snake_2d(Lx, Ly, major="col")
                if Lz is None
                else cls._coords_snake_3d(Lx, Ly, Lz, major="col")
            )
        elif mode_norm == "snake-row-major":
            coords = (
                cls._coords_snake_2d(Lx, Ly, major="row")
                if Lz is None
                else cls._coords_snake_3d(Lx, Ly, Lz, major="row")
            )
        elif mode_norm == "folded-snake":
            if Lz is not None:
                raise NotImplementedError("folded-snake mode is currently implemented only for 2D lattices.")
            coords = cls._coords_folded_snake_2d(Lx, Ly, major="col")
        elif mode_norm == "folded-snake-row-major":
            if Lz is not None:
                raise NotImplementedError("folded-snake mode is currently implemented only for 2D lattices.")
            coords = cls._coords_folded_snake_2d(Lx, Ly, major="row")
        elif mode_norm == "row-major":
            coords = (
                cls._coords_row_major_2d(Lx, Ly)
                if Lz is None
                else cls._coords_row_major_3d(Lx, Ly, Lz)
            )
        elif mode_norm == "col-major":
            coords = (
                cls._coords_col_major_2d(Lx, Ly)
                if Lz is None
                else cls._coords_col_major_3d(Lx, Ly, Lz)
            )
        elif mode_norm == "hilbert":
            if Lz is not None:
                raise NotImplementedError("hilbert mode is currently implemented only for 2D lattices.")
            coords = cls._coords_hilbert_2d(Lx, Ly, major="col")
        elif mode_norm == "hilbert-row-major":
            if Lz is not None:
                raise NotImplementedError("hilbert mode is currently implemented only for 2D lattices.")
            coords = cls._coords_hilbert_2d(Lx, Ly, major="row")
        elif mode_norm == "diag":
            if Lz is not None:
                raise NotImplementedError("diag mode is currently implemented only for 2D lattices.")
            coords = cls._coords_diag_2d(Lx, Ly)
        else:
            supported = ", ".join(cls._KNOWN_MODES)
            raise ValueError(f"Unknown lattice mapping mode: {mode}. Supported modes: {supported}")

        return cls._coords_to_maps(coords)

    @staticmethod
    def _path_directions_2d(one_d_to_lattice):
        directions = {}
        for idx in range(len(one_d_to_lattice) - 1):
            c0 = one_d_to_lattice[idx]
            c1 = one_d_to_lattice[idx + 1]
            if (len(c0) != 2) or (len(c1) != 2):
                continue
            if abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) != 1:
                continue
            directions[frozenset((c0, c1))] = (c0, c1)
        return directions

    @classmethod
    def _show_grid_2d_lines(cls, one_d_to_lattice, lattice_to_one_d, L_x, L_y, *, title):
        lines = [title, "    " + "    ".join(f"X{x}" for x in range(L_x))]
        path_dirs = cls._path_directions_2d(one_d_to_lattice)

        for y in range(L_y - 1, -1, -1):
            row_parts = []
            for x in range(L_x):
                idx = lattice_to_one_d[(x, y)]
                row_parts.append(f"o{idx:02d}")
                if x < L_x - 1:
                    edge = frozenset(((x, y), (x + 1, y)))
                    step = path_dirs.get(edge)
                    if step == ((x, y), (x + 1, y)):
                        row_parts.append(">>")
                    elif step == ((x + 1, y), (x, y)):
                        row_parts.append("<<")
                    else:
                        row_parts.append("--")
            lines.append(f"Y{y}  " + "".join(row_parts))

            if y > 0:
                conn_parts = []
                for x in range(L_x):
                    edge = frozenset(((x, y), (x, y - 1)))
                    step = path_dirs.get(edge)
                    if step == ((x, y), (x, y - 1)):
                        token = "v"
                    elif step == ((x, y - 1), (x, y)):
                        token = "^"
                    else:
                        token = "|"
                    conn_parts.append(f" {token} ")
                    if x < L_x - 1:
                        conn_parts.append("  ")
                lines.append("    " + "".join(conn_parts).rstrip())

        return lines

    @classmethod
    def _show_schematic_2d(
        cls,
        one_d_to_lattice,
        lattice_to_one_d,
        L_x,
        L_y,
        *,
        mode_norm,
        ax=None,
        title=None,
        show_order=True,
        path_cmap="plasma",
        node_radius=0.16,
        figsize=None,
    ):
        try:
            from matplotlib import colormaps
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Schematic plotting requires matplotlib to be available."
            ) from exc

        try:
            from quimb import schematic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Schematic plotting requires quimb.schematic to be available."
            ) from exc

        if title is None:
            title = f"OneDMap {mode_norm} ({L_x}x{L_y})"

        if figsize is None:
            figsize = (max(4.5, 1.35 * L_x), max(3.8, 1.35 * L_y))

        presets = {
            "lattice": {
                "color": (0.78, 0.82, 0.84, 1.0),
                "linewidth": 1.35,
            },
            "node": {
                "facecolor": (0.67, 0.86, 0.72, 1.0),
                "edgecolor": "white",
                "linewidth": 1.2,
                "radius": node_radius,
            },
            "label": {
                "color": (0.08, 0.16, 0.12, 1.0),
                "fontsize": 8,
                "ha": "center",
                "va": "center",
            },
            "coord": {
                "color": (0.20, 0.23, 0.28, 0.95),
                "fontsize": 7,
                "ha": "center",
                "va": "top",
            },
        }
        drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

        for x in range(L_x):
            for y in range(L_y):
                if x + 1 < L_x:
                    drawing.line((x, y), (x + 1, y), preset="lattice")
                if y + 1 < L_y:
                    drawing.line((x, y), (x, y + 1), preset="lattice")
                if mode_norm == "diag" and x + 1 < L_x and y + 1 < L_y:
                    drawing.line((x, y), (x + 1, y + 1), preset="lattice")

        coords = [one_d_to_lattice[idx] for idx in range(len(one_d_to_lattice))]
        cmap = colormaps.get_cmap(path_cmap)
        for idx, (coord0, coord1) in enumerate(zip(coords[:-1], coords[1:])):
            color = cmap(idx / max(1, len(coords) - 2))
            drawing.line(coord0, coord1, color=color, linewidth=2.35)
            drawing.arrowhead(
                coord0,
                coord1,
                color=color,
                center=0.56,
                width=0.052,
                length=0.105,
            )

        for coord, idx in lattice_to_one_d.items():
            drawing.circle(coord, preset="node")
            if show_order:
                drawing.text(coord, str(idx), preset="label")
            drawing.text(
                (coord[0], coord[1] - (node_radius + 0.15)),
                f"({coord[0]},{coord[1]})",
                preset="coord",
            )

        drawing.ax.set_title(title)
        drawing.ax.set_aspect("equal")
        drawing.ax.set_xticks(range(L_x))
        drawing.ax.set_yticks(range(L_y))
        drawing.ax.set_xticklabels([f"x{x}" for x in range(L_x)])
        drawing.ax.set_yticklabels([f"y{y}" for y in range(L_y)])
        drawing.ax.set_xlim(-0.55, L_x - 0.45)
        drawing.ax.set_ylim(-0.70, L_y - 0.35)
        drawing.ax.grid(False)

        return drawing

    @_dualmethod
    def show(
        target,
        Lx=None,
        Ly=None,
        Lz=_ONE_D_MAP_UNSET,
        mode=None,
        *,
        L_x=None,
        L_y=None,
        L_z=_ONE_D_MAP_UNSET,
        print_out=False,
        ax=None,
        title=None,
        show_order=True,
        path_cmap="plasma",
        node_radius=0.16,
        figsize=None,
    ):
        """Render a schematic illustration of 1D<->lattice mapping.

        This can be called either as ``OneDMap.show(Lx, Ly, ...)`` or on an
        instance, e.g. ``mapper.show()``. Instance calls can override layout
        options per use, for example ``mapper.show(mode="snake-row-major")``.
        """
        cls = target if isinstance(target, type) else type(target)
        Lx, Ly, Lz, mode_norm = cls._resolve_call_params(
            target,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            mode=mode,
            L_x=L_x,
            L_y=L_y,
            L_z=L_z,
        )
        one_d_to_lattice, lattice_to_one_d = cls.build(Lx, Ly, Lz=Lz, mode=mode_norm)
        if Lz is not None:
            raise NotImplementedError(
                "OneDMap.show() is currently only available for 2D lattices."
            )
        drawing = cls._show_schematic_2d(
            one_d_to_lattice,
            lattice_to_one_d,
            Lx,
            Ly,
            mode_norm=mode_norm,
            ax=ax,
            title=title,
            show_order=show_order,
            path_cmap=path_cmap,
            node_radius=node_radius,
            figsize=figsize,
        )
        if print_out:
            drawing.fig.show()
        return drawing


def _patch_unhashable_device_namespace_key():
    """Patch autoray namespace cache keys for unhashable backend device objects."""
    try:
        import autoray  # pylint: disable=import-outside-toplevel
        import autoray.autoray as ar_core  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return

    if getattr(ar_core, "_pepsy_unhashable_device_patch", False):
        return

    original_get_namespace = ar_core.get_namespace

    def _safe_get_namespace(like=None, device=None, dtype=None, submodule=None):
        if (device is None) and (like is not None) and (not isinstance(like, str)):
            try:
                device = like.device
            except AttributeError:
                device = None

        if device is not None:
            try:
                hash(device)
            except TypeError:
                dev_id = getattr(device, "id", None)
                device = f"device:{dev_id}" if dev_id is not None else str(device)

        return original_get_namespace(
            like=like,
            device=device,
            dtype=dtype,
            submodule=submodule,
        )

    ar_core.get_namespace = _safe_get_namespace
    autoray.get_namespace = _safe_get_namespace

    try:
        import quimb.tensor.decomp as qtn_decomp  # pylint: disable=import-outside-toplevel

        qtn_decomp.get_namespace = _safe_get_namespace
    except Exception:  # pragma: no cover
        pass

    ar_core._pepsy_unhashable_device_patch = True


def _validate_backend_callable(name, fn):
    if fn is not None and not callable(fn):
        raise TypeError(f"{name} must be callable or None")


def set_default_array_backend(array_backend):
    """Set package-wide default array backend caster.

    Parameters
    ----------
    array_backend : callable | None
        Function mapping arrays to a target backend. ``None`` clears default.
    """
    _validate_backend_callable("array_backend", array_backend)
    global _DEFAULT_ARRAY_BACKEND  # pylint: disable=global-statement
    _DEFAULT_ARRAY_BACKEND = array_backend


def get_default_array_backend():
    """Return package-wide default array backend caster, or ``None``."""
    return _DEFAULT_ARRAY_BACKEND


def set_default_grad_backend(grad_backend):
    """Set package-wide default gradient backend caster.

    Parameters
    ----------
    grad_backend : callable | None
        Function mapping arrays to trainable backend tensors.
    """
    _validate_backend_callable("grad_backend", grad_backend)
    global _DEFAULT_GRAD_BACKEND  # pylint: disable=global-statement
    _DEFAULT_GRAD_BACKEND = grad_backend


def get_default_grad_backend():
    """Return package-wide default gradient backend caster, or ``None``."""
    return _DEFAULT_GRAD_BACKEND


def reset_default_backends():
    """Clear package-wide backend defaults."""
    global _DEFAULT_ARRAY_BACKEND  # pylint: disable=global-statement
    global _DEFAULT_GRAD_BACKEND  # pylint: disable=global-statement
    _DEFAULT_ARRAY_BACKEND = None
    _DEFAULT_GRAD_BACKEND = None


def backend_torch(device="cpu", dtype=None, requires_grad=False):
    """Return a converter that materializes arrays as torch tensors."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "backend_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )

    def cast_array(x, device=device, dtype=dtype, requires_grad=requires_grad):

        if isinstance(x, torch.Tensor):
            out = x.detach() if requires_grad else x

            if dtype is None:
                out = out.to(device=device)
            else:
                out = out.to(device=device, dtype=dtype)

        else:
            if dtype is None:
                out = torch.as_tensor(x, device=device)
            else:
                out = torch.as_tensor(x, dtype=dtype, device=device)

        # Trainable tensors must be floating or complex
        if requires_grad and not (out.is_floating_point() or out.is_complex()):
            out = out.to(dtype=torch.float64)

        if requires_grad:
            out.requires_grad_(True)
        else:
            out.requires_grad_(False)

        return out

    return cast_array


def backend_numpy(dtype=np.float64):
    """Return a converter that materializes arrays as NumPy arrays."""

    def cast_array(x, dtype=dtype):
        return np.array(x, dtype=dtype)

    return cast_array


def backend_cupy(device=None, dtype=None):
    """Return a converter that materializes arrays as CuPy arrays.

    Parameters
    ----------
    device : int | cupy.cuda.Device | None, optional
        Target CUDA device. If ``None``, use CuPy's current device.
    dtype : dtype-like | torch.dtype | None, optional
        Target CuPy dtype. If ``None``, infer from input. Torch dtypes are
        accepted and internally mapped to CuPy-compatible dtypes.
    """
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend_cupy requires optional dependency 'cupy'. "
            "Install it with: pip install cupy-cuda12x (or your CUDA variant)."
        ) from exc

    _patch_unhashable_device_namespace_key()

    target_device = device
    if isinstance(target_device, int):
        target_device = cp.cuda.Device(target_device)

    if torch is not None and isinstance(dtype, torch.dtype):
        torch_to_cupy = {
            torch.complex128: cp.complex128,
            torch.complex64: cp.complex64,
            torch.float64: cp.float64,
            torch.float32: cp.float32,
            torch.float16: cp.float16,
            torch.int64: cp.int64,
            torch.int32: cp.int32,
            torch.int16: cp.int16,
            torch.int8: cp.int8,
            torch.uint8: cp.uint8,
            torch.bool: cp.bool_,
        }
        if dtype not in torch_to_cupy:
            raise ValueError(
                f"backend_cupy does not support torch dtype {dtype!r}."
            )
        dtype = torch_to_cupy[dtype]

    def cast_array(x, device=target_device, dtype=dtype):
        if device is None:
            return cp.asarray(x, dtype=dtype)
        with device:
            return cp.asarray(x, dtype=dtype)

    return cast_array


def backend_jax(device="cpu", dtype=None):
    """Return a converter that materializes arrays as JAX arrays.

    Parameters
    ----------
    device : str | jax.Device | None, optional
        Target device. Strings ``"cpu"``, ``"cuda"``/``"gpu"`` (optionally with
        an index, e.g. ``"cuda:1"``) are resolved against ``jax.devices``. A
        ``jax.Device`` instance is used as-is. ``None`` leaves placement to
        JAX's default.
    dtype : str | jax.numpy.dtype | None, optional
        Target dtype, e.g. ``"float64"`` or ``jnp.complex128``. ``None`` infers
        from the input.

    Notes
    -----
    JAX arrays are immutable and have no ``requires_grad`` flag; gradients in
    JAX flow via tracing (``jax.grad`` / ``jax.value_and_grad``). This
    converter therefore does not expose a ``requires_grad`` argument.
    """
    try:
        import jax  # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax (or jax[cuda12])."
        ) from exc

    def _resolve_device(dev):
        if dev is None or not isinstance(dev, str):
            return dev
        s = dev.lower()
        if ":" in s:
            kind, idx_s = s.split(":", 1)
            idx = int(idx_s)
        else:
            kind, idx = s, 0
        if kind == "cuda":
            kind = "gpu"
        try:
            return jax.devices(kind)[idx]
        except (RuntimeError, IndexError) as err:
            raise ValueError(
                f"backend_jax: device {dev!r} not available; "
                f"jax.devices() = {jax.devices()}"
            ) from err

    target_device = _resolve_device(device)
    if dtype is None:
        target_dtype = None
    else:
        # Canonicalize dtypes under current JAX x64 policy so requests like
        # float64/complex128 do not emit truncation warnings when x64 is off.
        target_dtype = jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))

    def cast_array(x, device=target_device, dtype=target_dtype):
        # Coerce non-JAX inputs (incl. torch tensors) to a numpy-compatible
        # form first so jnp.asarray accepts them on any backend.
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        arr = jnp.asarray(x, dtype=dtype)
        if device is not None:
            arr = jax.device_put(arr, device)
        return arr

    return cast_array


def register_torch_linalg(mode="complex"):
    """Register custom torch linalg gradients in autoray.

    Parameters
    ----------
    mode : {"complex", "real"}, default="complex"
        Which SVD/QR registrations to install.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "register_torch_linalg requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    if mode == "complex":
        lr.reg_rel_svd_torch()
        lr.reg_complex_qr_torch()
        return
    if mode == "real":
        lr.reg_real_svd_torch()
        lr.reg_real_qr_torch()
        return
    raise ValueError("mode must be 'complex' or 'real'")


def reg_rel_svd_torch():
    """Register torch SVD with a stable relative-regularized backward rule.

    The registered autoray ``torch`` SVD uses Townsend's rectangular SVD
    reverse-mode update, Lorentzian broadening of singular-value denominators
    from differentiable tensor-network practice, and the complex phase/gauge
    correction for complex-valued SVDs.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_rel_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_torch()


def reg_complex_svd_torch():
    """Register complex torch SVD autograd rule in autoray.

    Compatibility wrapper for :func:`reg_rel_svd_torch`.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_complex_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_torch()


def reg_real_svd_torch():
    """Register the real-only torch SVD autograd rule in autoray.

    This is the real counterpart of :func:`reg_rel_svd_torch`. It shares the
    robust forward path (``gesvd`` driver on CUDA plus a batched SciPy ``gesvd``
    fallback), the same Townsend rectangular reverse-mode update, and the
    scale-aware Lorentzian broadening of the singular-value denominators, while
    dropping the complex phase/gauge correction. It supports rectangular and
    batched real inputs.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_real_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_svd_torch()


def reg_complex_qr_torch():
    """Register complex torch QR autograd rule in autoray."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_complex_qr_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_complex_qr_torch()


def reg_real_qr_torch():
    """Register real torch QR autograd rule in autoray."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_real_qr_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_qr_torch()


def reg_complex_svd_jax():
    """Register complex JAX SVD custom-VJP rule in autoray."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - exercised in no-jax CI
        raise ImportError(
            "reg_complex_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc

    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_complex_svd_jax()


def reg_rel_svd_jax():
    """Register JAX SVD with Pepsy's custom VJP rule in autoray."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - exercised in no-jax CI
        raise ImportError(
            "reg_rel_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc

    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_jax()


def reg_real_svd_jax():
    """Register JAX SVD custom-VJP rule for real-valued SVD workloads."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - exercised in no-jax CI
        raise ImportError(
            "reg_real_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc

    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_svd_jax()


def reg_stop_gradient_torch():
    """Register torch stop-gradient helper in autoray."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_stop_gradient_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_stop_gradient_torch()


def stop_grad(x):
    """Return ``x`` detached from autograd when the backend supports it.

    This is the public convenience wrapper for backend-agnostic code that
    otherwise would need to repeat ``ar.do("stop_gradient", x)`` boilerplate.
    """
    try:
        from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

        return lr.stop_grad(x)
    except Exception:
        try:
            return ar.do("stop_gradient", x)
        except Exception:
            return x


def _ensure_cotengrust():
    """Import cotengrust so cotengra can use accelerated pathfinders."""
    try:
        import cotengrust as ctgr  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - exercised by packaging/env failures
        raise ImportError(
            "Pepsy requires 'cotengrust' for accelerated cotengra path search. "
            "Install it with: pip install cotengrust"
        ) from exc
    return ctgr


def build_optimizer(
    progbar: bool = False,
    alpha: int = 64,
    max_time="rate:7e8",
    max_repeats: int = 2**8,
    parallel="auto",
    optlib: str = "cmaes",
    directory=False,
    hash_method: str = "b",
    overwrite=False,
    on_trial_error: str = "warn",
    slicing_opts: dict | None = None,
    reconf_opts: dict | None = None,
    slicing_reconf_opts: dict | None = None,
):
    """Build a reusable cotengra contraction optimizer.

    ``cotengrust`` is imported up front so cotengra's ``accel="auto"``
    pathfinders can use the Rust implementations when constructing greedy,
    random-greedy, optimal, and reconfiguration paths.

    Parameters
    ----------
    progbar : bool, optional
        Whether to show optimizer progress.
    alpha : int, optional
        Weight for the combo objective.
    max_time : str | float | None, optional
        Search budget for the hyper-optimizer.
    max_repeats : int, optional
        Maximum number of optimization trials.
    parallel : bool | str, optional
        Parallel search setting passed to cotengra.
    optlib : str, optional
        Backend optimizer library.
    directory : None | bool | str, optional
        Cache directory for reusable contraction trees.
    hash_method : str, optional
        Hashing method for reusable contraction lookup.
    overwrite : bool | str, optional
        Cache overwrite behavior.
    on_trial_error : str, optional
        How to handle individual trial failures.
    slicing_opts : dict | None, optional
        Options passed to cotengra slicing heuristics.
    reconf_opts : dict | None, optional
        Options for subtree reconfiguration.
    slicing_reconf_opts : dict | None, optional
        Options for interleaved slicing and reconfiguration.
    """
    _ensure_cotengrust()

    # cotengra expects directory to be str, True, or None — not False.
    if directory is False:
        directory = None

    kwargs = dict(
        minimize=f"combo-{int(alpha)}",
        max_time=max_time,
        max_repeats=max_repeats,
        parallel=parallel,
        optlib=optlib,
        directory=directory,
        hash_method=hash_method,
        overwrite=overwrite,
        progbar=progbar,
        on_trial_error=on_trial_error,
    )

    if reconf_opts is not None:
        kwargs["reconf_opts"] = reconf_opts

    if slicing_opts is not None:
        kwargs["slicing_opts"] = slicing_opts

    if slicing_reconf_opts is not None:
        kwargs["slicing_reconf_opts"] = slicing_reconf_opts

    return ctg.ReusableHyperOptimizer(**kwargs)


def build_compressed_optimizer(
    progbar=True,
    chi=4,
    directory=None,
    max_repeats=2**8,
    max_time="rate:1e7",
):
    """Build and return a reusable cotengra compressed optimizer.

    ``cotengrust`` is imported up front so cotengra can use accelerated
    contraction-ordering primitives in any supported compressed path searches.

    Parameters
    ----------
    directory : None, True, or str, optional
        Passed directly to cotengra. ``None`` disables caching; ``True``
        auto-generates a directory in the current working directory.
    """
    _ensure_cotengrust()

    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time=max_time,
        directory=directory,
    )
    return copt


def contract_hypercompressed_tn(
    tn,
    copt=None,
    max_bond=None,
    *,
    chi=None,
    output_inds=None,
    tree_gauge_distance=4,
    progbar=False,
    cutoff=1.0e-12,
    equalize_norms=False,
    inplace=False,
    do_full_simplify=True,
    seq="R",
):
    """Contract a generic tensor network with compressed hyper-optimization.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Tensor network to compress-contract.
    copt : object, optional
        Reusable compressed cotengra optimizer. If ``None``, one is built
        with :func:`build_compressed_optimizer` using ``chi``.
    max_bond : int | None, optional
        Maximum retained bond dimension during compressed contraction.
        If ``None``, defaults to ``chi``.
    chi : int | None, optional
        Bond dimension used to build ``copt`` when ``copt`` is ``None``.
        Required if both ``copt`` and ``max_bond`` are missing.
    output_inds : sequence[str] | None, optional
        Output indices to preserve during contraction.
    tree_gauge_distance : int, optional
        Gauge distance passed to ``contract_compressed_``.
    progbar : bool, optional
        Whether to show progress during compressed contraction.
    cutoff : float, optional
        Truncation cutoff passed to ``contract_compressed_``.
    equalize_norms : bool | float, optional
        Norm equalization option passed to ``contract_compressed_``.
    inplace : bool, optional
        If ``True``, mutate ``tn`` directly. Otherwise, contract a copy.
    do_full_simplify : bool, optional
        Whether to run ``tn_out.full_simplify_(seq="R", split_method="svd")``
        before building the contraction tree. Enabled by default.
    seq : str, optional
        Simplification sequence passed to ``full_simplify_`` when
        ``do_full_simplify=True``.

    Returns
    -------
    qtn.TensorNetwork
        The compressed-contracted tensor network.
    """
    if max_bond is None:
        max_bond = chi

    if copt is None:
        if chi is None:
            raise ValueError(
                "When `copt` is not provided, please provide `chi` "
                "to build a compressed optimizer."
            )
        copt = build_compressed_optimizer(progbar=progbar, chi=chi)

    if max_bond is None:
        raise ValueError("Please provide `max_bond` (or `chi`) for compressed contraction.")

    tn_out = tn if inplace else tn.copy()
    if do_full_simplify:
        tn_out.full_simplify_(seq=seq, split_method="svd", inplace=True)
    tree = tn_out.contraction_tree(copt)
    tn_out.contract_compressed_(
        optimize=tree,
        output_inds=output_inds,
        max_bond=max_bond,
        tree_gauge_distance=tree_gauge_distance,
        equalize_norms=equalize_norms,
        cutoff=cutoff,
        progbar=progbar,
    )
    return tn_out


def contract_hypercompressed_tn_batch(
    tn,
    samples,
    *,
    copt=None,
    chi=None,
    max_bond=None,
    site_inds=None,
    tree=None,
    cutoff=0.0,
    tree_gauge_distance=6,
    equalize_norms=1.0,
    output_inds=(),
    vmap=True,
    strip_exponent=True,
    chunk_size=None,
    report_timing=True,
    progbar=False,
    mem_warn_gb=None,
    return_tree=False,
):
    """Batch-contract amplitudes ``<x|psi>`` for many configs with ONE fixed tree.

    Torch-only batched analogue of :func:`contract_hypercompressed_tn`, following
    the symmray *batch gpu amplitudes* pattern.  The tensor network ``tn`` (arrays
    on the torch backend) is packed once with ``quimb.tensor.pack``; each sample's
    physical index is selected by a **one-hot contraction** (a ``torch.vmap``-safe
    replacement for the dense ``isel``, which would call ``.item()`` on a traced
    tensor); and the compressed contraction tree is built **once** -- the cotengra
    hyper-optimization search is a single warm-up -- then reused for every sample,
    either fused with ``torch.vmap`` (``vmap=True``) or in a Python loop.

    The search runs only once and the resulting tree lives in host memory, so the
    per-sample cost is just the compressed contraction (no path re-search).  Pass
    ``return_tree=True`` once and feed the returned ``tree`` back in via ``tree=``
    to reuse the warm-up across separate calls.

    Parameters
    ----------
    tn : quimb.tensor.TensorNetwork
        State network with per-site physical indices (e.g. a PEPS), arrays on the
        torch backend.  Must expose ``sites``/``site_ind`` unless ``site_inds`` is
        given.
    samples : torch.Tensor
        Integer (``int64``) configs.  Shape ``(num_samples, L)`` (batch-major) or
        ``(L, num_samples)``; the ``L`` axis is matched to ``len(site_inds)`` and
        transposed to batch-major automatically.  Column ``i`` selects the
        physical value of ``site_inds[i]``.
    copt : object, optional
        Reusable compressed cotengra optimizer used to build the tree.  If
        ``None`` (and ``tree`` is not given) one is built from ``chi``.
    chi, max_bond : int, optional
        ``chi`` sizes ``copt``; ``max_bond`` caps the retained bond during
        contraction (defaults to ``chi``).
    site_inds : sequence[str], optional
        Physical index order matching the columns of ``samples``.  Defaults to
        ``[tn.site_ind(s) for s in tn.sites]``.
    tree : cotengra.ContractionTree, optional
        Pre-built contraction tree (from a previous ``return_tree=True`` call) to
        reuse, skipping the warm-up search.
    cutoff : float, optional
        Singular-value cutoff.  **Must be ``0.0`` when ``vmap=True``**: a positive
        cutoff makes the SVD truncation rank data-dependent, which ``torch.vmap``
        cannot trace.  Fixed-rank truncation to ``max_bond`` is used instead.
    tree_gauge_distance, equalize_norms, output_inds : optional
        Passed through to ``contract_compressed`` (see
        :func:`contract_hypercompressed_tn`).
    vmap : bool, optional
        If ``True`` (default) fuse the batch with ``torch.vmap``; otherwise loop.
    strip_exponent : bool, optional
        If ``True`` (default) return ``(mantissa, exponent)`` base-10 pairs
        (``amplitude = mantissa * 10 ** exponent``) for over/underflow stability.
    chunk_size : int | None, optional
        If given, split the batch into chunks of this many samples and process one
        chunk at a time (each chunk still fused by ``torch.vmap`` when
        ``vmap=True``).  Bounds peak memory for large batches; ``None`` (default)
        processes the whole batch at once.
    report_timing : bool, optional
        If ``True`` (default) print a one-line timing summary: the warm-up
        (tree-build) time and the cost of a single-sample contraction.  Set
        ``False`` to silence.
    progbar : bool, optional
        If ``True`` show a ``tqdm`` progress bar over the batch chunks (one
        update per chunk; set ``chunk_size`` for finer granularity).
    mem_warn_gb : float | None, optional
        Estimated peak-memory threshold (GB) above which a warning is emitted
        (the contraction still proceeds).  ``None`` (default) uses half of
        physical RAM.  The estimate is the compressed tree's peak intermediate
        size times the (chunked) batch size times the element byte-width.
    return_tree : bool, optional
        If ``True`` also return the (possibly newly built) contraction tree.

    Returns
    -------
    (mantissa, exponent) : tuple[torch.Tensor, torch.Tensor]
        Length-``num_samples`` tensors when ``strip_exponent=True``.
    amplitudes : torch.Tensor
        Length-``num_samples`` complex tensor when ``strip_exponent=False``.
    tree : optional
        Returned alongside the above when ``return_tree=True``.
    """
    import torch  # local import: the batch path is torch-only
    import time
    import os
    import warnings
    import math

    if not isinstance(samples, torch.Tensor):
        raise TypeError(
            "contract_hypercompressed_tn_batch expects `samples` as a torch "
            "int64 tensor (this batch path is torch-only)."
        )
    if vmap and float(cutoff) != 0.0:
        raise ValueError(
            "vmap=True requires cutoff=0.0: a positive cutoff makes the SVD "
            "truncation rank data-dependent (n_chi = count_nonzero(s > cutoff)), "
            "so the retained shape varies per sample, which torch.vmap cannot "
            "trace. Use cutoff=0.0 -- for a fixed max_bond it keeps exactly "
            "max_bond singular values (the most accurate choice for that bond) -- "
            "or pass vmap=False to loop (which supports an adaptive cutoff)."
        )

    if site_inds is None:
        site_inds = [tn.site_ind(s) for s in tn.sites]
    site_inds = list(site_inds)
    n_sites = len(site_inds)
    phys_dims = [int(tn.ind_size(ind)) for ind in site_inds]

    samples = samples.to(torch.int64)
    if samples.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got shape {tuple(samples.shape)}.")
    # Accept (num_samples, L) or (L, num_samples): orient to batch-major.
    if samples.shape[1] != n_sites:
        if samples.shape[0] == n_sites:
            samples = samples.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"`samples` shape {tuple(samples.shape)} does not match "
                f"L={n_sites} on either axis."
            )

    if max_bond is None:
        max_bond = chi
    if max_bond is None and tree is None:
        raise ValueError("Please provide `max_bond` (or `chi`) for compressed contraction.")

    params, skeleton = qtn.pack(tn)
    # Reference dtype from the packed leaves (all torch arrays share it).
    ref = next(iter(params.values())) if isinstance(params, dict) else params[0]
    dtype = ref.dtype
    ref_is_cuda = bool(getattr(ref, "is_cuda", False))

    def _sync():
        if ref_is_cuda:
            torch.cuda.synchronize()

    def _selected_tn(params, x):
        """Unpack and select each site's physical value via a one-hot contraction."""
        tnx = qtn.unpack(params, skeleton)
        for i, ind in enumerate(site_inds):
            one_hot = torch.nn.functional.one_hot(x[i], phys_dims[i]).to(dtype)
            tnx |= qtn.Tensor(one_hot, (ind,))
        return tnx

    # Warm-up: build the compressed contraction tree exactly once.  The one-hot
    # values do not change the network topology, so a tree built on a
    # representative config is valid for every sample.
    if tree is None:
        if copt is None:
            if chi is None:
                raise ValueError(
                    "When neither `copt` nor `tree` is provided, pass `chi` to "
                    "build a compressed optimizer."
                )
            copt = build_compressed_optimizer(progbar=False, chi=chi)
        x0 = torch.zeros(n_sites, dtype=torch.int64)
        _t0 = time.perf_counter()
        tree = _selected_tn(params, x0).contraction_tree(copt, output_inds=output_inds)
        warmup_time = time.perf_counter() - _t0
    else:
        warmup_time = None  # reused an existing tree

    def _amplitude(params, x):
        return _selected_tn(params, x).contract_compressed(
            optimize=tree,
            output_inds=output_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            strip_exponent=strip_exponent,
        )

    n_samples = int(samples.shape[0])
    if n_samples == 0:
        empty = samples.new_zeros(0, dtype=dtype)
        out = (empty, empty.real.clone()) if strip_exponent else empty
        return (out, tree) if return_tree else out

    # Diagnostic: time a single-sample contraction (one extra call) so the
    # per-sample cost and the one-off warm-up are always visible.
    _sync()
    _t0 = time.perf_counter()
    _ = _amplitude(params, samples[0])
    _sync()
    one_sample_time = time.perf_counter() - _t0
    if report_timing:
        warm = "reused" if warmup_time is None else f"{warmup_time:.3f}s"
        print(
            f"[contract_hypercompressed_tn_batch] warm-up(tree)={warm} | "
            f"one-sample={one_sample_time * 1e3:.1f}ms | batch={n_samples}"
            + (f" | chunk_size={int(chunk_size)}" if chunk_size else "")
        )

    # Cost/memory estimate from the compressed contraction tree.  The vmapped
    # batch adds a leading dimension of size `chunk_eff` to every intermediate,
    # so peak memory ~= (per-sample peak intermediate elements) * chunk_eff *
    # bytes/element.  FLOPs are reported as log10 and memory as log2; a warning
    # (but not an abort) fires if the estimate exceeds `mem_warn_gb`.
    chunk_eff = n_samples if not chunk_size else min(int(chunk_size), n_samples)
    try:
        log10_flops = float(tree.total_flops(chi=max_bond, log=10))  # per sample
        peak_elems = float(tree.peak_size(chi=max_bond))
    except Exception:  # pragma: no cover - defensive: tree API drift
        log10_flops = float("nan")
        peak_elems = float("nan")
    bytes_per = 8 if dtype in (torch.complex64, torch.complex32) else 16
    peak_mem_bytes = peak_elems * chunk_eff * bytes_per
    log10_total_flops = log10_flops + (math.log10(n_samples) if n_samples > 0 else 0.0)
    log2_peak_mem = math.log2(peak_mem_bytes) if peak_mem_bytes > 0 else float("-inf")
    if mem_warn_gb is not None:
        mem_warn_bytes = float(mem_warn_gb) * 1e9
    else:
        try:
            mem_warn_bytes = 0.5 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            mem_warn_bytes = float("inf")
    if report_timing:
        print(
            f"[contract_hypercompressed_tn_batch] log10(flops/sample)={log10_flops:.2f} "
            f"| log10(total_flops)={log10_total_flops:.2f} "
            f"| log2(peak_mem_bytes, chunk={chunk_eff})={log2_peak_mem:.2f} (chi={max_bond})"
        )
    if peak_mem_bytes > mem_warn_bytes:
        warnings.warn(
            "contract_hypercompressed_tn_batch estimated peak memory "
            f"log2(bytes)={log2_peak_mem:.1f} (~{peak_mem_bytes / 1e9:.1f} GB, "
            f"chunk={chunk_eff}, chi={max_bond}) exceeds "
            f"log2={math.log2(mem_warn_bytes):.1f} (~{mem_warn_bytes / 1e9:.1f} GB); "
            "reduce chunk_size or max_bond. Proceeding anyway.",
            stacklevel=2,
        )

    # Batch execution, optionally chunked (each chunk still vmap-fused) to bound
    # peak memory on large batches.
    cs = n_samples if not chunk_size else int(chunk_size)

    def _run_chunk(chunk):
        if vmap:
            return torch.vmap(_amplitude, in_dims=(None, 0))(params, chunk)
        if strip_exponent:
            pairs = [_amplitude(params, chunk[b]) for b in range(chunk.shape[0])]
            return (
                torch.stack([p[0] for p in pairs]),
                torch.stack([p[1] for p in pairs]),
            )
        return torch.stack([_amplitude(params, chunk[b]) for b in range(chunk.shape[0])])

    chunk_starts = range(0, n_samples, cs)
    if progbar:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

        chunk_starts = tqdm(chunk_starts, desc="amp batch", unit="chunk")
    chunk_outs = [_run_chunk(samples[i:i + cs]) for i in chunk_starts]
    if strip_exponent:
        out = (
            torch.cat([o[0] for o in chunk_outs]),
            torch.cat([o[1] for o in chunk_outs]),
        )
    else:
        out = torch.cat(chunk_outs)

    if return_tree:
        return out, tree
    return out


def tn_norm(
    psi,
    *,
    contraction_opt: Any | None = None,
    strip_exponent: bool = False,
    simplify: bool = False,
    simplify_seq: str = "R",
):
    """Compute the norm of a tensor network state.

    Parameters
    ----------
    psi : qtn.TensorNetwork
        State whose norm is computed.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.
    strip_exponent : bool, default=False
        If ``True``, pass ``strip_exponent=True`` to the contraction, which
        returns ``(mantissa, exponent)`` instead of the scalar result.
    simplify : bool, default=False
        Whether to simplify the closed norm network before contraction.
    simplify_seq : str, optional
        Simplification sequence passed to ``full_simplify_`` when
        ``simplify=True``.

    Returns
    -------
    float | tuple[float, float]
        ``|<psi|psi>|`` when ``strip_exponent=False``, or
        ``(mantissa, exponent)`` when ``strip_exponent=True``.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    norm_tn = psi.H & psi
    if simplify:
        norm_tn.full_simplify_(seq=simplify_seq, output_inds=())
    if not strip_exponent:
        return ar.do(
            "abs",
            norm_tn.contract(all, optimize=contraction_opt, output_inds=()),
        )

    return norm_tn.contract(
        all,
        optimize=contraction_opt,
        output_inds=(),
        strip_exponent=strip_exponent,
    )


def _count_format_fields(fmt):
    return sum(field is not None for _, field, _, _ in Formatter().parse(fmt))


def _build_ind_id(prefix, arity):
    return prefix + ",".join("{}" for _ in range(int(arity)))


def _infer_where_coord_arity(where):
    """Infer coordinate arity from ``where`` when unambiguous."""
    if isinstance(where, str):
        return None

    if isinstance(where, Integral):
        return 1

    if not isinstance(where, (list, tuple)):
        return None
    if not where:
        return None

    if all(isinstance(v, Integral) for v in where):
        if len(where) == 1:
            return 1
        return None

    for site in where:
        if isinstance(site, (list, tuple)) and site and all(
            isinstance(v, Integral) for v in site
        ):
            return len(site)

    return None


def _infer_phys_ind_id(tn, where):
    """Infer default ``k``-prefixed physical index format for ``where``."""
    arity_hint = _infer_where_coord_arity(where)
    if arity_hint is None:
        if hasattr(tn, "Lz"):
            arity_hint = 3
        elif hasattr(tn, "Lx") and hasattr(tn, "Ly"):
            arity_hint = 2
        else:
            arity_hint = 1

    return _build_ind_id("k", arity_hint)


def _where_to_phys_inds(where, *, ind_id="k{}"):
    """Convert user-provided site selector(s) to physical index names."""
    n_fields = _count_format_fields(ind_id)
    if n_fields < 1:
        raise ValueError("ind_id must include at least one format field, e.g. 'k{}'.")

    if isinstance(where, str):
        return [where]

    if isinstance(where, Integral):
        if n_fields != 1:
            raise TypeError(
                "Scalar integer sites require a 1-field ind_id like 'k{}'."
            )
        return [ind_id.format(int(where))]

    if not isinstance(where, (list, tuple)):
        raise TypeError("where must be a site or a sequence of sites.")
    if not where:
        raise ValueError("where must not be empty.")

    if n_fields > 1 and len(where) == n_fields and all(
        isinstance(v, Integral) for v in where
    ):
        return [ind_id.format(*[int(v) for v in where])]

    inds = []
    for site in where:
        if isinstance(site, str):
            inds.append(site)
            continue

        if isinstance(site, Integral):
            if n_fields != 1:
                raise TypeError(
                    "Integer site entries require a 1-field ind_id like 'k{}'."
                )
            inds.append(ind_id.format(int(site)))
            continue

        if isinstance(site, (list, tuple)):
            if n_fields == 1:
                if not site or not all(isinstance(v, Integral) for v in site):
                    raise TypeError(
                        "For ind_id='k{}', nested where entries must contain integer sites."
                    )
                inds.extend(ind_id.format(int(v)) for v in site)
                continue

            if len(site) != n_fields or not all(isinstance(v, Integral) for v in site):
                raise TypeError(
                    "Each site tuple/list must match the number of ind_id fields."
                )
            inds.append(ind_id.format(*[int(v) for v in site]))
            continue

        raise TypeError(
            "where entries must be index strings, integers, or tuples/lists "
            "matching ind_id."
        )

    return inds


def measure_obs(
    tn,
    obs,
    where,
    *,
    ind_id=None,
    bra=None,
    normalize=True,
    contraction_opt: Any | None = None,
):
    """Measure local observable(s) on a tensor network ket.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Ket tensor network.
    obs : array_like | sequence[array_like]
        Observable tensor(s) to apply. This can be a single observable or a
        sequence matched with ``where``.
    where : site selector | sequence[site selector]
        Site selector(s) matching ``obs``. For batched use, provide one entry
        per observable. Site formatting follows ``ind_id`` and mirrors
        :func:`pepsy.operators.gates.gate` single-gate ``where`` usage.
    ind_id : str | None, optional
        Site-index format. If ``None`` (default), assume ``k``-prefixed
        indices based on ``where`` and TN dimensionality
        (``"k{}"``, ``"k{},{}"``, or ``"k{},{},{}"``).
        Networks using other prefixes (for example ``"b{}"``) must set
        ``ind_id`` explicitly.
    bra : qtn.TensorNetwork | None, optional
        If provided, compute ``<bra|obs|tn>`` directly without normalization.
        If ``None``, normalization is controlled by ``normalize``.
    normalize : bool, default=True
        If ``True`` and ``bra`` is ``None``, compute
        ``<tn|obs|tn> / <tn|tn>``. If ``False``, return raw ``<tn|obs|tn>``
        without computing ``tn_norm``.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.

    Returns
    -------
    scalar
        Measured observable value.

    Notes
    -----
    This function applies observables using :func:`pepsy.operators.gates.gate` with
    ``contract=False`` on a copy of ``tn`` before contraction.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    # Local import avoids circular import at module load time.
    from ..operators.gates import gate  # pylint: disable=import-outside-toplevel

    if isinstance(obs, (list, tuple)):
        if not isinstance(where, (list, tuple)):
            raise ValueError(
                "When obs is a sequence, where must be a matching sequence with "
                "the same length."
            )
        if len(obs) != len(where):
            raise ValueError(
                "When obs is a sequence, where must be a matching sequence with "
                "the same length."
            )
        obs_where_pairs = zip(obs, where)
    else:
        obs_where_pairs = ((obs, where),)

    tn_obs = tn.copy()
    infer_ind_id = ind_id is None
    for obs_i, where_i in obs_where_pairs:
        ind_id_i = _infer_phys_ind_id(tn, where_i) if infer_ind_id else ind_id
        target_inds = _where_to_phys_inds(where_i, ind_id=ind_id_i)
        outer_inds = set(tn_obs.outer_inds())
        missing = [ind for ind in target_inds if ind not in outer_inds]
        if missing:
            missing_str = ", ".join(sorted(set(missing)))
            raise ValueError(
                "Could not find target physical indices in tn.outer_inds(): "
                f"{missing_str}. If your TN uses non-'k' physical index names, "
                "pass ind_id explicitly (for example ind_id='b{}')."
            )
        tn_obs = gate(
            tn_obs,
            obs_i,
            where=where_i,
            ind_id=ind_id_i,
            contract=False,
            inplace=False,
        )

    if bra is not None:
        return (bra & tn_obs).contract(all, optimize=contraction_opt)

    numer = (tn.H & tn_obs).contract(all, optimize=contraction_opt)
    if not normalize:
        return numer

    norm_ = tn_norm(tn, contraction_opt=contraction_opt)
    if norm_ == 0.0:
        raise ValueError("Cannot compute normalized observable for a zero-norm state.")
    return numer / norm_


def tn_fidelity(
    psi,
    psi_fix,
    *,
    contraction_opt: Any | None = None,
    simplify: bool = False,
    simplify_seq: str = "R",
):
    """Compute normalized overlap fidelity.

    Parameters
    ----------
    psi : qtn.TensorNetwork
        Trial state.
    psi_fix : qtn.TensorNetwork
        Reference state.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.
    simplify : bool, default=False
        Whether to simplify each closed norm/overlap network before
        contraction.
    simplify_seq : str, optional
        Simplification sequence passed to ``full_simplify_`` when
        ``simplify=True``.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)

    def closed_overlap(left, right):
        tn = left.H & right
        if simplify:
            tn.full_simplify_(seq=simplify_seq, output_inds=())
        return abs(tn.contract(all, optimize=contraction_opt, output_inds=()))

    val_0 = closed_overlap(psi, psi)
    val_1 = closed_overlap(psi, psi_fix)
    val_ref = closed_overlap(psi_fix, psi_fix)

    val_1 = val_1**2
    fidelity = ar.do("abs", val_1) / (val_0 * val_ref)
    return ar.do("abs", fidelity)


def add_cycle(peps, bond_dim, cylinder=False):
    """Add periodic bonds to a PEPS network in x (and optional y) directions."""
    Ly = peps.Ly
    Lx = peps.Lx
    for j in range(Ly):
        T1 = peps[f"I{Lx-1},{j}"]
        T2 = peps[f"I{0},{j}"]
        qtn.new_bond(T1, T2, size=bond_dim, name=None, axis1=0, axis2=0)

    if not cylinder:
        for i in range(Lx):
            T1 = peps[f"I{i},{Ly-1}"]
            T2 = peps[f"I{i},{0}"]
            qtn.new_bond(T1, T2, size=bond_dim, name=None, axis1=0, axis2=0)
    return peps


def id_to_pepo(lx, ly, phys_dim=2, dtype="complex128", chi=1, rand_strength=0.0):
    """Create a PEPO identity on an ``lx x ly`` lattice.

    Parameters
    ----------
    lx : int
        Lattice size in x direction.
    ly : int
        Lattice size in y direction.
    phys_dim : int, optional
        Physical dimension per site.
    dtype : str, optional
        Tensor dtype.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.PEPO
        Identity PEPO with bond dimension ``chi``.
    """
    pepo = qtn.PEPO.rand(Lx=lx, Ly=ly, bond_dim=1, seed=666, dtype=dtype)
    eye = np.eye(phys_dim, dtype=dtype)

    for tensor in pepo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [phys_dim, phys_dim], dtype=dtype)
        data[tuple([0] * n_virt)] = eye
        tensor.modify(data=data)

    if chi > 1:
        pepo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return pepo


def id_to_mpo(L, phys_dim=2, dtype="complex128", cyclic=False, chi=1, rand_strength=0.0):
    """Create a 1D MPO identity.

    Parameters
    ----------
    L : int
        Number of sites.
    phys_dim : int, optional
        Physical dimension per site.
    dtype : str, optional
        Tensor dtype.
    cyclic : bool, optional
        Whether to create a periodic MPO.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.MatrixProductOperator
        Identity MPO with bond dimension ``chi``.
    """
    mpo = qtn.MPO_rand(L, bond_dim=1, phys_dim=phys_dim, cyclic=cyclic, seed=666, dtype=dtype)
    eye = np.eye(phys_dim, dtype=dtype)

    for tensor in mpo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [phys_dim, phys_dim], dtype=dtype)
        data[tuple([0] * n_virt)] = eye
        tensor.modify(data=data)

    if chi > 1:
        mpo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mpo


def tns_align(p, pepo):
    r"""Apply a PEPO operator to a PEPS ket: :math:`\hat{O}|\psi\rangle`.

    The PEPO ``k``-indices contract with the PEPS ``k``-indices on join.
    The PEPO ``b``-indices (output legs) are renamed to ``k``-indices so
    the result has the same physical index convention as a standard PEPS.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state :math:`|\psi\rangle`.  Outer indices must follow
        the ``k<int>[,<int>...]`` convention.
    pepo : qtn.TensorNetwork
        PEPO operator :math:`\hat{O}`.  Outer indices must follow the
        ``k<int>[,<int>...]`` and ``b<int>[,<int>...]`` convention.
        This matches :func:`pepsy.operators.gates.build_pepo_from_gates` output.

    Returns
    -------
    qtn.TensorNetwork
        The resulting network :math:`\hat{O}|\psi\rangle` with ``k``-type
        physical indices.
    """
    # Validate lattice tags
    validate_tensor_network_tags(p)
    validate_tensor_network_tags(pepo)

    tn = p & pepo
    # Only randomize the physical k-indices (shared between p and pepo).
    # Virtual bond indices must NOT be renamed — they must stay stable so
    # the Y-cut outer indices of the double-layer TN match the stored
    # boundary MPS across repeated calls to _prepare_current_double_layers.
    # Use non-mutating reindex to avoid modifying the original p/pepo tensors
    # (quimb's & shares tensor objects, so reindex_ would mutate the originals).
    contracted_k = {
        idx: qtn.rand_uuid()
        for idx in tn.inner_inds()
        if isinstance(idx, str) and idx.startswith("k")
    }
    if contracted_k:
        tn.reindex_(contracted_k)
    # Rename PEPO output b-indices -> k-indices (physical convention)
    b_to_k = {
        idx: f"k{idx[1:]}"
        for idx in tn.outer_inds()
        if idx.startswith("b")
    }
    if b_to_k:
        tn.reindex_(b_to_k)
    return tn



def expec_mpo(mpo, mps, *, contraction_opt=None):
    """Compute normalized 1D expectation value ``<mps|mpo|mps> / <mps|mps>``.

    Parameters
    ----------
    mpo : qtn.TensorNetwork
        1D MPO using ``k{i}``/``b{i}`` physical index families.
    mps : qtn.MatrixProductState | qtn.TensorNetwork
        1D state network with physical indices ``k{i}``.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    if isinstance(mps, qtn.MatrixProductState):
        mps_n = mps.copy()
        norm_ = mps_n.normalize()
        L = mps.L
        divisor = 1.0
    else:
        mps_n = mps.copy()
        L = len(mps.outer_inds())
        norm_ = tn_norm(mps_n, contraction_opt=contraction_opt)
        divisor = norm_

    if norm_ == 0.0:
        raise ValueError("Cannot compute normalized expectation for a zero-norm state.")

    mps_h = mps_n.H
    mps_h.reindex_({f"k{i}": f"b{i}" for i in range(L)})
    return (mps_h | mpo | mps_n).contract(all, optimize=contraction_opt) / divisor


def ps_to_peps(Lx: int, Ly: int, dtype: str = "complex128", theta: float = 0.0, cyclic: bool = False, chi: int = 1, rand_strength: float = 0.0):
    """Create a product-state PEPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool, optional
        If True, add periodic bonds (bond dimension 1) via :func:`add_cycle`.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.PEPS
        Initialized PEPS with bond dimension ``chi``.
    """
    peps = qtn.PEPS.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for x in range(Lx):
        for y in range(Ly):
            tensor = peps[x, y]
            phys_ind = peps.site_ind(x, y)
            phys_axis = tensor.inds.index(phys_ind)
            data = np.zeros_like(tensor.data, dtype=dtype)

            slicer = [0] * data.ndim
            slicer[phys_axis] = slice(None)
            data[tuple(slicer)] = local_vec
            tensor.modify(data=data)
    peps.astype_(dtype)
    if cyclic:
        peps = add_cycle(peps, bond_dim=1)
    if chi > 1:
        peps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return peps


def ps_to_3dpeps(
    Lx: int,
    Ly: int,
    Lz: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create a product-state 3D PEPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    Lz : int
        Lattice size in z direction.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool or tuple[bool, bool, bool], optional
        If True, create periodic bonds with bond dimension 1. A three-tuple
        can set periodicity independently for x, y, and z.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.PEPS3D
        Initialized 3D PEPS with bond dimension ``chi``.
    """
    peps = qtn.PEPS3D.rand(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        bond_dim=1,
        seed=666,
        dtype=dtype,
        cyclic=cyclic,
    )
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                tensor = peps[x, y, z]
                phys_ind = peps.site_ind(x, y, z)
                phys_axis = tensor.inds.index(phys_ind)
                data = np.zeros_like(tensor.data, dtype=dtype)

                slicer = [0] * data.ndim
                slicer[phys_axis] = slice(None)
                data[tuple(slicer)] = local_vec
                tensor.modify(data=data)
    peps.astype_(dtype)
    if chi > 1:
        peps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return peps


def ps_to_mps(L: int, dtype: str = "complex128", theta: float = 0.0, cyclic: bool = False, chi: int = 1, rand_strength: float = 0.0):
    """Create a product-state MPS parameterized by ``theta``.

    Each site tensor is set so the physical vector is
    ``[cos(theta), sin(theta)]`` with trivial virtual bonds.

    Parameters
    ----------
    L : int
        Number of sites.
    dtype : str, optional
        Tensor dtype passed to numpy/quimb.
    theta : float, optional
        Product-state angle controlling local amplitudes.
    cyclic : bool, optional
        If True, create a periodic MPS with bond dimension 1.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.MatrixProductState
        Initialized MPS with bond dimension ``chi``.
    """
    mps = qtn.MPS_rand_state(L=L, bond_dim=1, phys_dim=2, cyclic=cyclic, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)

    for i in range(L):
        tensor = mps[i]
        phys_ind = mps.site_ind(i)
        phys_axis = tensor.inds.index(phys_ind)
        data = np.zeros_like(tensor.data, dtype=dtype)

        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    mps.astype_(dtype)
    if chi > 1:
        mps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mps


def ps_to_pepo(
    Lx: int,
    Ly: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create a PEPO of local projectors ``|v><v|`` parameterized by ``theta``.

    Each site tensor is the rank-1 operator
    ``|v><v|`` where ``v = [cos(theta), sin(theta)]``.

    Parameters
    ----------
    Lx : int
        Lattice size in x direction.
    Ly : int
        Lattice size in y direction.
    dtype : str, optional
        Tensor dtype.
    theta : float, optional
        Product-state angle controlling the local vector.
    cyclic : bool, optional
        If True, add periodic bonds (bond dimension 1) via :func:`add_cycle`.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.PEPO
        PEPO with local projectors and bond dimension ``chi``.
    """
    pepo = qtn.PEPO.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    proj = np.outer(local_vec, np.conj(local_vec))

    for tensor in pepo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [2, 2], dtype=dtype)
        data[tuple([0] * n_virt)] = proj
        tensor.modify(data=data)

    pepo.astype_(dtype)
    if cyclic:
        pepo = add_cycle(pepo, bond_dim=1)
    if chi > 1:
        pepo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return pepo


def ps_to_mpo(
    L: int,
    dtype: str = "complex128",
    theta: float = 0.0,
    cyclic: bool = False,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create an MPO of local projectors ``|v><v|`` parameterized by ``theta``.

    Each site tensor is the rank-1 operator
    ``|v><v|`` where ``v = [cos(theta), sin(theta)]``.

    Parameters
    ----------
    L : int
        Number of sites.
    dtype : str, optional
        Tensor dtype.
    theta : float, optional
        Product-state angle controlling the local vector.
    cyclic : bool, optional
        Whether to create a periodic MPO.
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.

    Returns
    -------
    quimb.tensor.MatrixProductOperator
        MPO with local projectors and bond dimension ``chi``.
    """
    mpo = qtn.MPO_rand(L, bond_dim=1, phys_dim=2, cyclic=cyclic, seed=666, dtype=dtype)
    local_vec = np.array([math.cos(theta), math.sin(theta)], dtype=dtype)
    proj = np.outer(local_vec, np.conj(local_vec))

    for tensor in mpo:
        ndim = len(tensor.data.shape)
        n_virt = ndim - 2
        virt_shape = [1] * n_virt
        data = np.zeros(virt_shape + [2, 2], dtype=dtype)
        data[tuple([0] * n_virt)] = proj
        tensor.modify(data=data)

    mpo.astype_(dtype)
    if chi > 1:
        mpo.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mpo


def random_haar_qubit(seed=None, perturb=0.0):
    """Generate one random single-qubit Haar sample as ``(theta, phi)``.

    Parameters
    ----------
    seed : int | None, optional
        If set, produce a deterministic sample.
    perturb : float, optional
        Additive offset applied to both sampled parameters.

    Returns
    -------
    tuple[float, float]
        ``(theta, phi)`` Bloch angles.
    """
    rng = np.random.default_rng(seed)
    phi = 2 * np.pi * rng.random() + perturb
    z = 2 * rng.random() - 1 + perturb
    z = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z)
    return float(theta), float(phi)


def haar_random_state(
    L: int,
    dtype: str = "complex128",
    seed=None,
    L_max: int = 20,
    as_tensor: bool = False,
):
    """Create a dense Haar-random ``L``-qubit state.

    This samples a full Hilbert-space state, so the result is generally
    entangled. Unlike :func:`hrps_to_mps`, this is not a product-state tensor
    network: it returns dense amplitudes with ``2**L`` entries.

    Parameters
    ----------
    L : int
        Number of qubits.
    dtype : str, optional
        Complex numpy dtype for the returned amplitudes.
    seed : int | None, optional
        Seed for deterministic samples.
    L_max : int, optional
        Maximum allowed number of qubits. Values above 20 are capped to 20
        with a warning because this helper constructs a dense state.
    as_tensor : bool, optional
        If True, return the amplitudes reshaped as a dense tensor with shape
        ``(2,) * L``. Otherwise return a dense vector with shape ``(2**L,)``.

    Returns
    -------
    numpy.ndarray
        Normalized dense Haar-random state amplitudes.
    """
    if not isinstance(L, Integral) or L < 0:
        raise ValueError("L must be a non-negative integer.")
    if not isinstance(L_max, Integral) or L_max < 0:
        raise ValueError("L_max must be a non-negative integer.")
    if L_max > 20:
        warnings.warn(
            "haar_random_state constructs dense entangled states and is "
            "intended for L <= 20; capping L_max to 20.",
            UserWarning,
            stacklevel=2,
        )
        L_max = 20
    if L > L_max:
        raise ValueError(
            "haar_random_state constructs a dense entangled state and only "
            f"supports L <= L_max (got L={L}, L_max={L_max})."
        )

    dtype = np.dtype(dtype)
    if dtype.kind != "c":
        raise TypeError("dtype must be a complex numpy dtype.")

    real_dtype = np.float32 if dtype == np.dtype("complex64") else np.float64
    rng = np.random.default_rng(seed)
    dim = 2 ** int(L)
    state = rng.normal(size=dim).astype(real_dtype)
    state = state + 1j * rng.normal(size=dim).astype(real_dtype)
    state = state.astype(dtype, copy=False)
    state /= np.linalg.norm(state)

    if as_tensor:
        return state.reshape((2,) * int(L))
    return state


def hrps_to_peps(
    Lx: int,
    Ly: int,
    dtype: str = "complex128",
    cyclic: bool = False,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create a PEPS with per-site single-qubit Haar states.

    If ``haar_params`` is omitted, each site uses :func:`random_haar_qubit`.
    With ``seed`` set, site ``k`` uses ``seed + k`` for reproducible but
    distinct samples.

    Parameters
    ----------
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.
    """
    peps = ps_to_peps(Lx=Lx, Ly=Ly, dtype=dtype, theta=0.0, cyclic=cyclic)

    n_sites = Lx * Ly
    if haar_params is not None:
        if len(haar_params) != n_sites:
            raise ValueError(f"haar_params must have length {n_sites}.")
        params = list(haar_params)
    else:
        params = [
            random_haar_qubit(None if seed is None else int(seed) + k, perturb=perturb)
            for k in range(n_sites)
        ]

    for x in range(Lx):
        for y in range(Ly):
            idx = x * Ly + y
            theta, phi = params[idx]
            local_vec = np.array(
                [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
                dtype=dtype,
            )
            tensor = peps[x, y]
            phys_ind = peps.site_ind(x, y)
            phys_axis = tensor.inds.index(phys_ind)
            data = np.zeros_like(tensor.data, dtype=dtype)

            slicer = [0] * data.ndim
            slicer[phys_axis] = slice(None)
            data[tuple(slicer)] = local_vec
            tensor.modify(data=data)

    peps.astype_(dtype)
    if chi > 1:
        peps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return peps


def hrps_to_mps(
    L: int,
    dtype: str = "complex128",
    cyclic: bool = False,
    seed=None,
    perturb: float = 0.0,
    haar_params=None,
    chi: int = 1,
    rand_strength: float = 0.0,
):
    """Create a MPS with per-site single-qubit Haar states.

    If ``haar_params`` is omitted, each site uses :func:`random_haar_qubit`.
    With ``seed`` set, site ``k`` uses ``seed + k`` for reproducible but
    distinct samples.

    Parameters
    ----------
    chi : int, optional
        Target bond dimension. If greater than 1, the bond dimension is
        expanded via ``expand_bond_dimension`` after initialization.
    rand_strength : float, optional
        Random noise strength passed to ``expand_bond_dimension``.
    """
    mps = ps_to_mps(L=L, dtype=dtype, theta=0.0, cyclic=cyclic)

    if haar_params is not None:
        if len(haar_params) != L:
            raise ValueError(f"haar_params must have length {L}.")
        params = list(haar_params)
    else:
        params = [
            random_haar_qubit(None if seed is None else int(seed) + k, perturb=perturb)
            for k in range(L)
        ]

    for i in range(L):
        theta, phi = params[i]
        local_vec = np.array(
            [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
            dtype=dtype,
        )
        tensor = mps[i]
        phys_ind = mps.site_ind(i)
        phys_axis = tensor.inds.index(phys_ind)
        data = np.zeros_like(tensor.data, dtype=dtype)

        slicer = [0] * data.ndim
        slicer[phys_axis] = slice(None)
        data[tuple(slicer)] = local_vec
        tensor.modify(data=data)

    mps.astype_(dtype)
    if chi > 1:
        mps.expand_bond_dimension_(chi, rand_strength=rand_strength)
    return mps
