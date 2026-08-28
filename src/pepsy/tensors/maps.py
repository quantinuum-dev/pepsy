"""Lattice-to-chain mapping implementation."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

__all__ = ["OneDMap"]

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
        "inside-out",
        "diag",
        "finder",
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
        "inside-out": "inside-out",
        "insideout": "inside-out",
        "center-out": "inside-out",
        "centerout": "inside-out",
        "center": "inside-out",
        "outward": "inside-out",
        "diag": "diag",
        "diagonal": "diag",
        "diag-snake": "diag",
        "mps-finder": "finder",
        "layout-finder": "finder",
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

    def __init__(
        self,
        Lx=None,
        Ly=None,
        Lz=None,
        mode="snake",
        *,
        L_x=None,
        L_y=None,
        L_z=_ONE_D_MAP_UNSET,
        finder=None,
        gate_stream=None,
        gates=None,
        layout_kwargs=None,
        finder_base_mode="snake",
    ):
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
        if gate_stream is not None and gates is not None:
            raise TypeError("pass only one of gate_stream= or gates=.")
        if gates is not None:
            gate_stream = gates
        if finder is not None and gate_stream is not None:
            raise TypeError(
                "pass either finder= or gate_stream=/gates=, not both."
            )
        if hasattr(gate_stream, "__next__"):
            gate_stream = list(gate_stream)
        self.finder = finder
        self.gate_stream = gate_stream
        self.layout_kwargs = (
            {} if layout_kwargs is None else dict(layout_kwargs)
        )
        self.finder_base_mode = self._normalize_mode(finder_base_mode)
        if self.finder_base_mode == "finder":
            raise ValueError("finder_base_mode cannot itself be 'finder'.")

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
    def _coords_inside_out_2d(L_x, L_y):
        """Return a deterministic center-to-boundary site ordering.

        This is a *growth ordering*, rather than a Hamiltonian path. Sites
        are grouped by Manhattan shells around the geometric center and use
        row-major order to break shell ties. The ordering therefore remains
        useful for rectangular lattices and for tree builders that attach
        each new site through an available nearest-neighbor edge.
        """
        center_x = L_x - 1
        center_y = L_y - 1
        coords = [(x, y) for x in range(L_x) for y in range(L_y)]
        return sorted(
            coords,
            key=lambda coord: (
                abs(2 * coord[0] - center_x) + abs(2 * coord[1] - center_y),
                coord[0],
                coord[1],
            ),
        )

    @staticmethod
    def _coords_inside_out_3d(L_x, L_y, L_z):
        """Return a deterministic center-to-boundary ordering in 3D."""
        center_x = L_x - 1
        center_y = L_y - 1
        center_z = L_z - 1
        coords = [
            (x, y, z)
            for x in range(L_x)
            for y in range(L_y)
            for z in range(L_z)
        ]
        return sorted(
            coords,
            key=lambda coord: (
                abs(2 * coord[0] - center_x)
                + abs(2 * coord[1] - center_y)
                + abs(2 * coord[2] - center_z),
                coord[0],
                coord[1],
                coord[2],
            ),
        )

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
    def _hilbert_sign(value):
        """Return the sign of an integer Hilbert basis component."""
        return -1 if value < 0 else (1 if value > 0 else 0)

    @classmethod
    def _generate_hilbert_2d(cls, x, y, ax, ay, bx, by):
        """Recursively generate one oriented rectangular Hilbert region.

        This is the generalized rectangular Hilbert (often called Gilbert)
        construction. Unlike cropping a power-of-two Hilbert square, each
        recursive region is part of the requested rectangle, so arbitrary
        rectangular sizes remain complete and deterministic. The classical
        Hilbert traversal is recovered when both dimensions are powers of two.

        The construction is adapted from Jakub Cerveny's BSD-2-Clause
        reference implementation (copyright 2018) at
        https://github.com/jakubcerveny/gilbert.
        """
        width = abs(ax + ay)
        height = abs(bx + by)
        dax, day = cls._hilbert_sign(ax), cls._hilbert_sign(ay)
        dbx, dby = cls._hilbert_sign(bx), cls._hilbert_sign(by)

        if height == 1:
            for _ in range(width):
                yield x, y
                x += dax
                y += day
            return
        if width == 1:
            for _ in range(height):
                yield x, y
                x += dbx
                y += dby
            return

        ax2, ay2 = ax // 2, ay // 2
        bx2, by2 = bx // 2, by // 2
        width2 = abs(ax2 + ay2)
        height2 = abs(bx2 + by2)

        if 2 * width > 3 * height:
            if width2 % 2 and width > 2:
                ax2 += dax
                ay2 += day
            yield from cls._generate_hilbert_2d(
                x, y, ax2, ay2, bx, by,
            )
            yield from cls._generate_hilbert_2d(
                x + ax2,
                y + ay2,
                ax - ax2,
                ay - ay2,
                bx,
                by,
            )
            return

        if height2 % 2 and height > 2:
            bx2 += dbx
            by2 += dby
        yield from cls._generate_hilbert_2d(
            x, y, bx2, by2, ax2, ay2,
        )
        yield from cls._generate_hilbert_2d(
            x + bx2,
            y + by2,
            ax,
            ay,
            bx - bx2,
            by - by2,
        )
        yield from cls._generate_hilbert_2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )

    @classmethod
    def _coords_hilbert_2d_base(cls, L_x, L_y):
        """Return a complete x-oriented Hilbert traversal of a rectangle."""
        # Keep the basis orientation explicit instead of rotating based on
        # aspect ratio.  ``hilbert-row-major`` is the transposed orientation,
        # and both remain useful distinct baselines on rectangular lattices.
        coords = list(
            cls._generate_hilbert_2d(0, 0, L_x, 0, 0, L_y)
        )

        expected = L_x * L_y
        if len(coords) != expected or len(set(coords)) != expected:
            raise RuntimeError(
                "generalized Hilbert traversal did not cover the requested "
                f"rectangle {L_x}x{L_y} exactly."
            )
        if any(
            x < 0 or x >= L_x or y < 0 or y >= L_y
            for x, y in coords
        ):
            raise RuntimeError(
                "generalized Hilbert traversal escaped the requested "
                f"rectangle {L_x}x{L_y}."
            )
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
    def _coords_from_mps_finder(
        cls,
        Lx,
        Ly,
        Lz,
        *,
        finder=None,
        gate_stream=None,
        layout_kwargs=None,
        finder_base_mode="snake",
    ):
        """Compose an MPS layout permutation with a regular lattice map.

        The MPS finder works on logical integer site labels and returns a
        position-to-logical-site permutation. This method applies that
        permutation to the coordinates of ``finder_base_mode``. It performs
        no tensor-network construction or state replay.
        """
        if finder is not None and gate_stream is not None:
            raise TypeError(
                "pass either finder= or gate_stream=/gates=, not both."
            )
        if finder is None:
            if gate_stream is None:
                raise ValueError(
                    "mode='finder' requires gate_stream=, gates=, or finder=."
                )
            from ..optimizers.mps.layout import MpsGateStreamLayoutFinder

            nsites = Lx * Ly if Lz is None else Lx * Ly * Lz
            finder = MpsGateStreamLayoutFinder(gate_stream, L=nsites)

        layout_kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
        if isinstance(finder, Mapping):
            plan = finder
        else:
            run = getattr(finder, "run", None)
            if not callable(run):
                raise TypeError(
                    "finder must be an MpsGateStreamLayoutFinder or a layout "
                    "plan mapping returned by its run() method."
                )
            plan = run(**layout_kwargs)

        site_order = plan.get(
            "site_order",
            plan.get("qubit_inds", plan.get("order")),
        )
        if site_order is None:
            raise ValueError(
                "MPS finder plan must contain site_order, qubit_inds, or order."
            )
        try:
            site_order = tuple(int(site) for site in site_order)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MPS finder site_order must contain integer lattice labels."
            ) from exc
        nsites = Lx * Ly if Lz is None else Lx * Ly * Lz
        if len(site_order) != nsites or set(site_order) != set(range(nsites)):
            raise ValueError(
                "MPS finder site_order must be a permutation of the lattice "
                f"labels 0..{nsites - 1}."
            )

        base_idx2coo, _ = cls.build(
            Lx,
            Ly,
            Lz=Lz,
            mode=finder_base_mode,
        )
        return [base_idx2coo[site] for site in site_order]

    @classmethod
    def _normalize_mode(cls, mode):
        mode_norm = str(mode).strip().lower().replace("_", "-")
        return cls._MODE_ALIASES.get(mode_norm, mode_norm)

    @_dualmethod
    def build(
        target,
        Lx=None,
        Ly=None,
        Lz=_ONE_D_MAP_UNSET,
        mode=None,
        *,
        L_x=None,
        L_y=None,
        L_z=_ONE_D_MAP_UNSET,
        finder=None,
        gate_stream=None,
        gates=None,
        layout_kwargs=None,
        finder_base_mode=None,
    ):
        """Build ``(one_d_to_lattice, lattice_to_one_d)`` for a traversal mode.

        This can be called either as ``OneDMap.build(Lx, Ly, ...)`` or on an
        instance, e.g. ``OneDMap(Lx, Ly, mode="row-major").build()``.
        Instance calls can override options per use, for example
        ``mapper.build(mode="snake")``.

        ``mode="finder"`` composes an MPS gate-stream layout with the base
        lattice traversal. Supply ``gate_stream=``/``gates=`` or a previously
        constructed MPS layout ``finder=``. The finder only analyzes supports
        and returns a site permutation; it never allocates or truncates an MPS.
        """
        cls = target if isinstance(target, type) else type(target)
        if gate_stream is not None and gates is not None:
            raise TypeError("pass only one of gate_stream= or gates=.")
        if gates is not None:
            gate_stream = gates
        if not isinstance(target, type):
            if finder is None:
                finder = target.finder
            if gate_stream is None:
                gate_stream = target.gate_stream
            if layout_kwargs is None:
                layout_kwargs = target.layout_kwargs
            if finder_base_mode is None:
                finder_base_mode = target.finder_base_mode
        if finder_base_mode is None:
            finder_base_mode = "snake"
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

        if mode_norm == "finder":
            coords = cls._coords_from_mps_finder(
                Lx,
                Ly,
                Lz,
                finder=finder,
                gate_stream=gate_stream,
                layout_kwargs=layout_kwargs,
                finder_base_mode=finder_base_mode,
            )
            return cls._coords_to_maps(coords)

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
        elif mode_norm == "inside-out":
            coords = (
                cls._coords_inside_out_2d(Lx, Ly)
                if Lz is None
                else cls._coords_inside_out_3d(Lx, Ly, Lz)
            )
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
                # Keep arrowheads fixed-size. Quimb's default relative
                # scaling makes non-local periodic jumps produce enormous
                # arrowheads that obscure the lattice schematic.
                relative=False,
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
        finder=None,
        gate_stream=None,
        gates=None,
        layout_kwargs=None,
        finder_base_mode=None,
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
        if not isinstance(target, type):
            if finder is None:
                finder = target.finder
            if gate_stream is None:
                gate_stream = target.gate_stream
            if layout_kwargs is None:
                layout_kwargs = target.layout_kwargs
            if finder_base_mode is None:
                finder_base_mode = target.finder_base_mode
        one_d_to_lattice, lattice_to_one_d = cls.build(
            Lx,
            Ly,
            Lz=Lz,
            mode=mode_norm,
            finder=finder,
            gate_stream=gate_stream,
            gates=gates,
            layout_kwargs=layout_kwargs,
            finder_base_mode=finder_base_mode,
        )
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
