"""Hamiltonian builders for dense operators and MPOs."""

from __future__ import annotations

import warnings

from numbers import Integral

import numpy as np
import quimb
import quimb.tensor as qtn

__all__ = [
    "ham_tn",
]


class ham_tn:
    """Build MPO Hamiltonians from local terms on a snake-ordered 2D lattice.

    Parameters
    ----------
    L_x : int
        Number of lattice sites along x.
    L_y : int
        Number of lattice sites along y.
    max_bond : int, default=300
        Compression cap used after each term addition.
    cutoff : float, default=1e-12
        Compression cutoff used after each term addition.
    data_type : str | numpy.dtype, default="float64"
        Default dtype used for identity MPO tensors and operators.
    build_snake_maps : callable | None, default=None
        Optional mapping builder that replaces the default snake layout.
        It must accept ``(L_x, L_y)`` and return ``(map, map_inv)`` where:
        - ``map`` is either ``dict[int, tuple[int, int]]`` or a length-``L``
          sequence of coordinates.
        - ``map_inv`` is either ``dict[tuple[int, int], int]`` or ``None``
          (in which case it is inferred from ``map``).

    Attributes
    ----------
    map : dict[int, tuple[int, int]]
        Snake mapping from 1D chain index to 2D coordinate ``(x, y)``.
    map_inv : dict[tuple[int, int], int]
        Inverse snake mapping from coordinate ``(x, y)`` to 1D index.
    """

    def __init__(
        self,
        L_x,
        L_y,
        *,
        max_bond=256,
        cutoff=1e-12,
        data_type="float64",
        build_snake_maps=None,
    ):
        if not isinstance(L_x, Integral) or not isinstance(L_y, Integral):
            raise TypeError("L_x and L_y must be integers.")
        if L_x < 1 or L_y < 1:
            raise ValueError("L_x and L_y must be >= 1.")

        self.L_x = int(L_x)
        self.L_y = int(L_y)
        self.L = self.L_x * self.L_y
        if self.L < 2:
            raise ValueError("MPO construction requires L_x * L_y >= 2.")

        if not isinstance(max_bond, Integral):
            raise TypeError("max_bond must be an integer.")
        if int(max_bond) < 1:
            raise ValueError("max_bond must be >= 1.")
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be >= 0.")

        self.max_bond = int(max_bond)
        self.cutoff = cutoff
        self.data_type = np.dtype(data_type)

        self.map, self.map_inv = self._resolve_snake_maps(build_snake_maps)

    @classmethod
    def build_itf_lattice(
        cls,
        *,
        L_x,
        L_y,
        lattice="square",
        edges=None,
        cyclic=False,
        J=1.0,
        field=1.0,
        max_bond=256,
        cutoff=1e-12,
        data_type="float64",
        build_snake_maps=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        edge_kwargs=None,
        plot_geometry=False,
        plot_kwargs=None,
        return_plot=False,
        return_edges=True,
        return_builder=True,
    ):
        """Construct a builder and ITF Hamiltonian in one call.

        This is a convenience wrapper around ``ham_tn(...).build_itf(...)`` so
        callers can pass lattice size and model parameters directly.

        Parameters
        ----------
        L_x, L_y : int
            Lattice dimensions used to build the internal ``ham_tn`` builder.
        lattice, edges, cyclic, J, field, compress_each, cycle_peps, cycle_bond_dim, \
        edge_kwargs, plot_geometry, plot_kwargs, return_plot, return_edges
            Forwarded directly to :meth:`build_itf`.
        max_bond, cutoff, data_type, build_snake_maps
            Used to construct the internal builder instance.
        return_builder : bool, default=False
            Deprecated compatibility argument. Output is always a dict and
            always includes the constructed builder.

        Returns
        -------
        dict
            Dictionary with named outputs and mappings:
            ``mpo``, ``pepo``, optional ``edges``/``ax``, optional
            ``edges_1d`` (when ``edges`` available), ``builder``,
            ``one_d_to_two_d``, and ``two_d_to_one_d``.
        """
        builder = cls(
            L_x=L_x,
            L_y=L_y,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            build_snake_maps=build_snake_maps,
        )
        out = builder.build_itf(
            lattice=lattice,
            edges=edges,
            cyclic=cyclic,
            J=J,
            field=field,
            compress_each=compress_each,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            edge_kwargs=edge_kwargs,
            plot_geometry=plot_geometry,
            plot_kwargs=plot_kwargs,
            return_plot=return_plot,
            return_edges=return_edges,
        )
        _ = return_builder  # accepted for backward compatibility
        payload = {
            "mpo": out[0],
            "pepo": out[1],
            "edges": None,
            "edges_1d": None,
            "ax": None,
            "builder": builder,
            "one_d_to_two_d": dict(builder.map),
            "two_d_to_one_d": dict(builder.map_inv),
        }
        if return_edges and return_plot:
            payload["edges"] = out[2]
            payload["ax"] = out[3]
        elif return_edges:
            payload["edges"] = out[2]
        elif return_plot:
            payload["ax"] = out[2]

        if payload["edges"] is not None:
            map_inv = payload["two_d_to_one_d"]
            payload["edges_1d"] = tuple(
                (map_inv[tuple(site0)], map_inv[tuple(site1)])
                for site0, site1 in payload["edges"]
            )
        return payload

    @staticmethod
    def _coerce_coord(site):
        if isinstance(site, tuple) and len(site) == 2 and all(
            isinstance(v, Integral) for v in site
        ):
            return (int(site[0]), int(site[1]))
        if isinstance(site, list) and len(site) == 2 and all(
            isinstance(v, Integral) for v in site
        ):
            return (int(site[0]), int(site[1]))
        raise TypeError(f"Invalid coordinate: {site!r}")

    def _normalize_forward_map(self, mapping):
        if isinstance(mapping, dict):
            out = {}
            for key, coord in mapping.items():
                if not isinstance(key, Integral):
                    raise TypeError("map keys must be integers.")
                out[int(key)] = self._coerce_coord(coord)
            return out

        sequence = tuple(mapping)
        if len(sequence) != self.L:
            raise ValueError(
                f"Forward map sequence must have length {self.L}, got {len(sequence)}."
            )
        return {idx: self._coerce_coord(coord) for idx, coord in enumerate(sequence)}

    def _normalize_inverse_map(self, mapping):
        if mapping is None:
            return None
        if not isinstance(mapping, dict):
            raise TypeError("map_inv must be a dict[(x, y) -> int] or None.")
        out = {}
        for coord, idx in mapping.items():
            if not isinstance(idx, Integral):
                raise TypeError("map_inv values must be integers.")
            out[self._coerce_coord(coord)] = int(idx)
        return out

    def _validate_maps(self, map_, map_inv):
        expected_indices = set(range(self.L))
        if set(map_) != expected_indices:
            raise ValueError(
                "map keys must exactly cover 0..L-1 with no gaps."
            )

        coords = list(map_.values())
        for x, y in coords:
            if x < 0 or x >= self.L_x or y < 0 or y >= self.L_y:
                raise ValueError(f"Coordinate {(x, y)} is outside lattice bounds.")
        if len(set(coords)) != self.L:
            raise ValueError("map coordinates must be unique.")

        expected_inv = {coord: idx for idx, coord in map_.items()}
        if map_inv is None:
            return expected_inv
        if map_inv != expected_inv:
            raise ValueError("map_inv is inconsistent with map.")
        return map_inv

    def _resolve_snake_maps(self, build_snake_maps):
        builder = self._build_snake_maps if build_snake_maps is None else build_snake_maps
        if not callable(builder):
            raise TypeError("build_snake_maps must be callable or None.")

        raw_maps = builder(self.L_x, self.L_y)
        if not isinstance(raw_maps, (tuple, list)) or len(raw_maps) != 2:
            raise ValueError(
                "build_snake_maps must return (map, map_inv)."
            )

        map_raw, map_inv_raw = raw_maps
        map_ = self._normalize_forward_map(map_raw)
        map_inv = self._normalize_inverse_map(map_inv_raw)
        map_inv = self._validate_maps(map_, map_inv)
        return map_, map_inv

    @staticmethod
    def _build_snake_maps(L_x, L_y):
        dic = {}
        for i in range(L_x):
            if i % 2 == 0:
                chain_sites = [i * L_y + j for j in range(L_y)]
                coords = [(i, j) for j in range(L_y)]
                dic = dic | dict(zip(chain_sites, coords))
            else:
                chain_sites = [i * L_y + j for j in range(L_y)]
                chain_sites.reverse()
                coords = [(i, j) for j in range(L_y)]
                dic = dic | dict(zip(chain_sites, coords))

        dic_ = {coord: idx for idx, coord in dic.items()}
        return dic, dic_

    def map_site(self, site):
        """Map site spec to 1D chain index.

        ``site`` can be either an integer chain index or a coordinate tuple
        ``(x, y)``.
        """
        if isinstance(site, Integral):
            index = int(site)
            if index < 0 or index >= self.L:
                raise ValueError(f"Site index {index} is outside [0, {self.L - 1}].")
            return index

        if isinstance(site, tuple) and len(site) == 2 and all(
            isinstance(v, Integral) for v in site
        ):
            coord = (int(site[0]), int(site[1]))
            if coord not in self.map_inv:
                raise ValueError(f"Coordinate {coord} is outside lattice bounds.")
            return self.map_inv[coord]

        raise TypeError("Site must be int or (x, y) tuple.")

    @staticmethod
    def _site_tensor(op, site, L):
        if site == 0 or site == L - 1:
            return op[None, :, :]
        return op[None, None, :, :]

    @staticmethod
    def _as_matrix(op):
        data = getattr(op, "data", op)
        return np.asarray(data)

    def _coerce_op(self, op, *, phys_dim, dtype):
        if callable(op) and not hasattr(op, "shape"):
            op = op()
        arr = self._as_matrix(op)
        if arr.shape != (phys_dim, phys_dim):
            raise ValueError(
                f"Operator must have shape ({phys_dim}, {phys_dim}), got {arr.shape}."
            )
        if np.iscomplexobj(arr) and not np.issubdtype(np.dtype(dtype), np.complexfloating):
            if np.allclose(arr.imag, 0.0):
                arr = arr.real
            else:
                raise ValueError(
                    "Complex-valued operator requires complex data_type "
                    f"(got {np.dtype(dtype)})."
                )
        return np.asarray(arr, dtype=dtype)

    @staticmethod
    def _is_coord_site(site):
        return (
            isinstance(site, tuple)
            and len(site) == 2
            and all(isinstance(v, Integral) for v in site)
        )

    @staticmethod
    def _parse_term(term):
        if not isinstance(term, (tuple, list)):
            raise TypeError("Each term must be tuple/list: (sites, ops) or (sites, ops, coeff).")
        if len(term) not in (2, 3):
            raise ValueError("Each term must be (sites, ops) or (sites, ops, coeff).")

        sites, ops = term[0], term[1]
        coeff = term[2] if len(term) == 3 else 1.0
        if not np.isscalar(coeff):
            raise TypeError("coeff must be a scalar.")

        if isinstance(ops, np.ndarray):
            ops = (ops,)
        elif isinstance(ops, tuple):
            pass
        else:
            ops = tuple(ops)

        if not isinstance(sites, (tuple, list)):
            raise TypeError(
                "sites must be a tuple/list of 2D coordinates, e.g. "
                "((x, y),) or ((x1, y1), (x2, y2))."
            )
        sites = tuple(sites)

        if len(sites) != len(ops):
            raise ValueError("sites and ops lengths must match.")
        if len(sites) not in (1, 2):
            raise ValueError("Only 1-site and 2-site terms are supported.")
        if not all(ham_tn._is_coord_site(site) for site in sites):
            raise TypeError(
                "Only 2D coordinate layout is supported in ints. "
                "Use terms like ((x, y),) or ((x1, y1), (x2, y2))."
            )

        return sites, ops, coeff

    def _term_to_mpo(self, term, *, phys_dim, dtype):
        sites, ops, coeff = self._parse_term(term)
        chain_sites = tuple(self.map_site(site) for site in sites)
        if len(set(chain_sites)) != len(chain_sites):
            raise ValueError("Duplicate sites in one term are not supported.")

        mpo_term = qtn.MPO_identity(
            self.L,
            phys_dim=phys_dim,
            dtype=dtype,
        )

        for n, (site, op) in enumerate(zip(chain_sites, ops)):
            op_arr = self._coerce_op(op, phys_dim=phys_dim, dtype=dtype)
            if n == 0:
                op_arr = coeff * op_arr
            mpo_term[site].modify(data=self._site_tensor(op_arr, site, self.L))

        return mpo_term

    def _zero_mpo(self, *, phys_dim, dtype):
        mpo = qtn.MPO_identity(
            self.L,
            phys_dim=phys_dim,
            dtype=dtype,
        )
        for tensor in mpo:
            tensor.modify(data=np.zeros_like(tensor.data, dtype=dtype))
        return mpo

    def build_mpo(
        self,
        ints,
        *,
        phys_dim=2,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
    ):
        """Build MPO from user interactions.

        Parameters
        ----------
        ints : sequence
            Sequence of terms. Supported term formats:
            - ``(((x, y),), (op,))``
            - ``(((x1, y1), (x2, y2)), (op1, op2))``
            - ``(((x, y),), (op,), coeff)``
            - ``(((x1, y1), (x2, y2)), (op1, op2), coeff)``
            Only 2D coordinate layout is accepted.
        phys_dim : int, default=2
            On-site physical dimension.
        max_bond : int | None, default=None
            MPO compression max bond. Uses instance default when None.
        cutoff : float | None, default=None
            MPO compression cutoff. Uses instance default when None.
        data_type : str | numpy.dtype | None, default=None
            Operator/MPO dtype. Uses instance default when None.
        compress_each : bool, default=True
            Compress after each term addition. If False, only compress once at end.

        Returns
        -------
        qtn.MatrixProductOperator
            Built Hamiltonian MPO.
        """
        if ints is None:
            raise ValueError("ints must be provided.")
        if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
            raise ValueError("phys_dim must be an integer >= 1.")

        dtype = self.data_type if data_type is None else np.dtype(data_type)
        max_bond = self.max_bond if max_bond is None else int(max_bond)
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        if max_bond < 1:
            raise ValueError("max_bond must be >= 1.")
        if cutoff < 0.0:
            raise ValueError("cutoff must be >= 0.")

        mpo_total = self._zero_mpo(phys_dim=phys_dim, dtype=dtype)
        for term in ints:
            mpo_term = self._term_to_mpo(term, phys_dim=phys_dim, dtype=dtype)
            mpo_total = mpo_total + mpo_term
            if compress_each:
                mpo_total.compress(max_bond=max_bond, cutoff=cutoff)

        if not compress_each:
            mpo_total.compress(max_bond=max_bond, cutoff=cutoff)
        return mpo_total

    def _add_snake_column_bonds_(self, mpo):
        """Add rank-1 vertical bonds that recover 2D connectivity in snake order."""
        for x in range(self.L_x - 1):
            for y in range(self.L_y):
                if x % 2 == 0:
                    if y < self.L_y - 1:
                        mpo[f"I{x},{y}"].new_bond(mpo[f"I{x + 1},{y}"], size=1)
                else:
                    if y > 0:
                        mpo[f"I{x},{y}"].new_bond(mpo[f"I{x + 1},{y}"], size=1)
        return mpo

    def _add_cycle_bonds_(self, pepo, *, bond_dim=1):
        """Optionally add periodic bonds in x and y directions."""
        if not isinstance(bond_dim, Integral) or bond_dim < 1:
            raise ValueError("bond_dim must be an integer >= 1.")

        if self.L_x > 1:
            for y in range(self.L_y):
                pepo[f"I{self.L_x - 1},{y}"].new_bond(pepo[f"I0,{y}"], size=int(bond_dim))

        if self.L_y > 1:
            for x in range(self.L_x):
                pepo[f"I{x},{self.L_y - 1}"].new_bond(pepo[f"I{x},0"], size=int(bond_dim))

        return pepo

    def mpo_to_pepo(
        self,
        mpo,
        *,
        cycle_peps=False,
        cycle_bond_dim=1,
        inplace=False,
    ):
        """Convert snake-ordered MPO into a 2D PEPO with lattice tags/indices.

        Parameters
        ----------
        mpo : qtn.MatrixProductOperator
            Input MPO with chain length ``L_x * L_y``.
        cycle_peps : bool, default=False
            If True, add periodic bonds along x and y boundaries.
        cycle_bond_dim : int, default=1
            Bond dimension used when ``cycle_peps=True``.
        inplace : bool, default=False
            If True, modify ``mpo`` in place.

        Returns
        -------
        qtn.PEPO
            Converted PEPO object with site tags ``I{x},{y}`` and physical
            index ids ``k{x},{y}``, ``b{x},{y}``.
        """
        if getattr(mpo, "L", None) != self.L:
            raise ValueError(
                f"MPO length mismatch: expected {self.L}, got {getattr(mpo, 'L', None)}."
            )

        pepo = mpo if inplace else mpo.copy()

        for chain_idx, tensor in enumerate(pepo):
            x, y = self.map[chain_idx]
            tensor.modify(tags=[f"I{x},{y}", f"X{x}", f"Y{y}"])
            upper_ind = pepo.upper_ind(chain_idx)
            lower_ind = pepo.lower_ind(chain_idx)
            tensor.reindex_(
                {
                    upper_ind: f"k{x},{y}",
                    lower_ind: f"b{x},{y}",
                }
            )

        self._add_snake_column_bonds_(pepo)

        pepo.view_as_(
            qtn.PEPO,
            Lx=self.L_x,
            Ly=self.L_y,
            site_tag_id="I{},{}",
            x_tag_id="X{}",
            y_tag_id="Y{}",
            upper_ind_id="k{},{}",
            lower_ind_id="b{},{}",
        )

        if cycle_peps:
            self._add_cycle_bonds_(pepo, bond_dim=cycle_bond_dim)

        return pepo

    def build_pepo(
        self,
        ints,
        *,
        phys_dim=2,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
    ):
        """Build PEPO directly from interaction terms."""
        mpo = self.build_mpo(
            ints,
            phys_dim=phys_dim,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            compress_each=compress_each,
        )
        return self.mpo_to_pepo(
            mpo,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            inplace=False,
        )

    def mpo_itf_2d(
        self,
        J=1.0,
        field=1.0,
        *,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        as_pepo=False,
        cycle_peps=False,
        cycle_bond_dim=1,
    ):
        """Build 2D transverse-field Ising MPO using snake mapping.

        Hamiltonian:
        ``H = J * sum_<ij> Z_i Z_j + field * sum_i X_i``

        Returns
        -------
        tuple
            ``(op, coord_to_chain_map)`` where ``op`` is MPO by default and
            PEPO when ``as_pepo=True``.
        """
        square_edges = tuple(qtn.edges_2d_square(self.L_x, self.L_y, cyclic=False))
        mpo, pepo = self.build_itf_from_edges(
            square_edges,
            J=J,
            field=field,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            compress_each=compress_each,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
        )
        if as_pepo:
            return pepo, dict(self.map_inv)
        return mpo, dict(self.map_inv)

    def _itf_ints_from_edges(
        self,
        edges,
        *,
        J,
        field,
        dtype,
    ):
        z_op = np.asarray(quimb.pauli("Z", dtype=dtype), dtype=dtype)
        x_op = np.asarray(quimb.pauli("X", dtype=dtype), dtype=dtype)

        sites = sorted({site for edge in edges for site in edge})
        ints = [(edge, (z_op, z_op), J) for edge in edges]
        ints.extend((((site,), (x_op,), field) for site in sites))
        return ints

    def build_itf_from_edges(
        self,
        edges,
        J=1.0,
        field=1.0,
        *,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
    ):
        """Build ITF Hamiltonian MPO + PEPO from an arbitrary edge list.

        Accepts any quimb geometry edge list, e.g.::

            qtn.edges_2d_square(Lx, Ly, cyclic=False)
            qtn.edges_2d_triangular(Lx, Ly, cyclic=False)

        Each edge must be ``((x0, y0), (x1, y1))``.  Sites are inferred as
        the union of all edge endpoints.

        Hamiltonian:
        ``H = J * sum_{edges} Z_i Z_j + field * sum_{sites} X_i``

        Parameters
        ----------
        edges : iterable of ((int, int), (int, int))
            Nearest-neighbour edge list from a quimb geometry function.
        J : float, default=1.0
            ZZ coupling strength.
        field : float, default=1.0
            Transverse-field (X) strength.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)``
        """
        dtype = self.data_type if data_type is None else np.dtype(data_type)
        edges = [tuple(edge) for edge in edges]
        if not edges:
            raise ValueError("edges must not be empty.")

        ints = self._itf_ints_from_edges(edges, J=J, field=field, dtype=dtype)

        mpo = self.build_mpo(
            ints,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=dtype,
            compress_each=compress_each,
        )
        pepo = self.mpo_to_pepo(
            mpo,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            inplace=False,
        )
        return mpo, pepo

    def plot_lattice_snake(
        self,
        edges,
        *,
        ax=None,
        title=None,
        show_chain_index=True,
        edge_color="0.72",
        snake_color="tab:red",
        node_color="tab:blue",
        edge_alpha=0.9,
        snake_alpha=0.95,
        node_size=165,
        edge_linewidth=1.6,
        snake_linewidth=2.6,
        node_edge_color="white",
        node_edge_width=1.2,
        snake_arrows=True,
        snake_cmap="plasma",
        show_legend=True,
        site_positions=None,
        invert_y=None,
    ):
        """Plot lattice geometry and snake traversal with richer styling.

        Parameters
        ----------
        edges : iterable of ((int, int), (int, int))
            Lattice edge list.
        ax : matplotlib.axes.Axes | None, default=None
            Optional axis. When None, creates a new figure + axis.
        title : str | None, default=None
            Plot title.
        show_chain_index : bool, default=True
            If True, annotate each lattice site with its snake-chain index.
        snake_arrows : bool, default=True
            Draw directional arrows along the snake traversal.
        snake_cmap : str | None, default="plasma"
            Colormap for progression along the snake path. If None, use
            ``snake_color`` uniformly.
        show_legend : bool, default=True
            If True, add a compact legend.
        site_positions : Mapping[(int, int), (float, float)] | None, default=None
            Optional plotting coordinates for sites. Useful for lattices whose
            build coordinates are remapped internally (e.g. hexagonal/kagome).
        invert_y : bool | None, default=None
            Whether to invert the y-axis. Defaults to True for plain integer
            grids and False when ``site_positions`` is supplied.
        """
        try:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
            from matplotlib import colors as mcolors  # pylint: disable=import-outside-toplevel
            from matplotlib.collections import LineCollection  # pylint: disable=import-outside-toplevel
            from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel
            from matplotlib.patches import FancyArrowPatch  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_lattice_snake requires matplotlib. "
                "Install with: pip install matplotlib"
            ) from exc

        def _coerce_plot_point(point):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise TypeError(f"plot position must be (x, y), got {point!r}")
            x_val, y_val = point
            if not (np.isscalar(x_val) and np.isscalar(y_val)):
                raise TypeError(f"plot position values must be numeric, got {point!r}")
            return float(x_val), float(y_val)

        edges = [tuple(edge) for edge in edges]
        if not edges:
            raise ValueError("edges must not be empty for plotting.")

        edges_norm = []
        for idx, edge in enumerate(edges):
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise TypeError(
                    f"Edge at position {idx} must be ((x0, y0), (x1, y1)), got {edge!r}."
                )
            site0 = self._coerce_coord(edge[0])
            site1 = self._coerce_coord(edge[1])
            edges_norm.append((site0, site1))
        edges = edges_norm

        positions = {}
        if site_positions is not None:
            if not hasattr(site_positions, "items"):
                raise TypeError("site_positions must be a mapping from site to (x, y).")
            for site, point in site_positions.items():
                positions[self._coerce_coord(site)] = _coerce_plot_point(point)

        def _pos(site):
            if site in positions:
                return positions[site]
            return float(site[0]), float(site[1])

        sites = sorted({self._coerce_coord(site) for edge in edges for site in edge})
        node_plot_points = [_pos(site) for site in sites]

        if ax is None:
            _, ax = plt.subplots(figsize=(7.6, 5.8))

        edge_segments = [
            (_pos(site0), _pos(site1))
            for site0, site1 in edges
        ]
        edge_collection = LineCollection(
            edge_segments,
            colors=edge_color,
            linewidths=edge_linewidth,
            alpha=edge_alpha,
            capstyle="round",
            zorder=1,
        )
        ax.add_collection(edge_collection)

        snake_sites = [self.map[idx] for idx in range(self.L)]
        snake_plot_sites = [_pos(site) for site in snake_sites]
        snake_segments = [
            (site0, site1)
            for site0, site1 in zip(snake_plot_sites[:-1], snake_plot_sites[1:])
        ]

        cmap = None
        snake_colors = snake_color
        if snake_segments and snake_cmap is not None:
            cmap = plt.get_cmap(snake_cmap)
            snake_colors = cmap(np.linspace(0.08, 0.92, len(snake_segments)))
            snake_colors[:, 3] = snake_alpha

        snake_collection = LineCollection(
            snake_segments,
            colors=snake_colors,
            linewidths=snake_linewidth,
            alpha=snake_alpha if snake_cmap is None else 1.0,
            capstyle="round",
            zorder=3,
        )
        ax.add_collection(snake_collection)

        if snake_arrows and snake_segments:
            for seg_idx, (site0, site1) in enumerate(zip(snake_plot_sites[:-1], snake_plot_sites[1:])):
                x0, y0 = site0
                x1, y1 = site1
                dx = x1 - x0
                dy = y1 - y0
                arrow_start = (x0 + 0.20 * dx, y0 + 0.20 * dy)
                arrow_end = (x0 + 0.80 * dx, y0 + 0.80 * dy)
                if snake_cmap is not None and cmap is not None:
                    ratio = seg_idx / max(len(snake_segments) - 1, 1)
                    arrow_color = cmap(0.08 + 0.84 * ratio)
                else:
                    arrow_color = snake_color
                arrow = FancyArrowPatch(
                    arrow_start,
                    arrow_end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=0.0,
                    color=arrow_color,
                    alpha=snake_alpha,
                    zorder=4,
                )
                ax.add_patch(arrow)

        if snake_cmap is not None and cmap is not None:
            norm = mcolors.Normalize(vmin=0, vmax=max(self.L - 1, 1))
            node_colors = [
                cmap(norm(self.map_inv[(x, y)]))
                for x, y in sites
            ]
        else:
            node_colors = node_color

        x_nodes = [pt[0] for pt in node_plot_points]
        y_nodes = [pt[1] for pt in node_plot_points]
        ax.scatter(
            x_nodes,
            y_nodes,
            s=node_size,
            c=node_colors,
            edgecolors=node_edge_color,
            linewidths=node_edge_width,
            zorder=5,
        )

        start_x, start_y = snake_plot_sites[0]
        end_x, end_y = snake_plot_sites[-1]
        ax.scatter(
            [start_x],
            [start_y],
            s=node_size * 1.35,
            c="#2ca02c",
            marker="s",
            edgecolors="white",
            linewidths=1.3,
            zorder=6,
        )
        ax.scatter(
            [end_x],
            [end_y],
            s=node_size * 1.45,
            c="#d62728",
            marker="X",
            edgecolors="white",
            linewidths=1.3,
            zorder=6,
        )

        if show_chain_index:
            for chain_idx, (x_val, y_val) in enumerate(snake_plot_sites):
                ax.text(
                    x_val,
                    y_val,
                    str(chain_idx),
                    fontsize=7.5,
                    color="white",
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=7,
                )

        if show_legend:
            snake_legend_color = snake_color
            if snake_cmap is not None and cmap is not None:
                snake_legend_color = cmap(0.5)
            legend_handles = [
                Line2D([0], [0], color=edge_color, lw=edge_linewidth, label="Lattice edges"),
                Line2D([0], [0], color=snake_legend_color, lw=snake_linewidth, label="Snake path"),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markerfacecolor="#2ca02c",
                    markeredgecolor="white",
                    markersize=8,
                    label="Snake start",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    linestyle="",
                    markerfacecolor="#d62728",
                    markeredgecolor="white",
                    markersize=8,
                    label="Snake end",
                ),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                framealpha=0.92,
                fontsize=8,
            )

        if title is None:
            title = f"Lattice Geometry + Snake MPS Path ({self.L_x}x{self.L_y})"

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

        if positions:
            x_min = min(x_nodes)
            x_max = max(x_nodes)
            y_min = min(y_nodes)
            y_max = max(y_nodes)
            span = max(x_max - x_min, y_max - y_min, 1.0)
            pad = 0.10 * span
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks(range(self.L_x))
            ax.set_yticks(range(self.L_y))
            ax.set_xlim(-0.6, self.L_x - 0.4)
            ax.set_ylim(-0.6, self.L_y - 0.4)

        if invert_y is None:
            invert_y = not positions
        if invert_y:
            ax.invert_yaxis()

        ax.set_facecolor("#f7f8fb")
        ax.grid(alpha=0.22, linewidth=0.8)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return ax

    def build_itf(
        self,
        lattice="square",
        *,
        edges=None,
        cyclic=False,
        J=1.0,
        field=1.0,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        edge_kwargs=None,
        plot_geometry=False,
        plot_kwargs=None,
        return_plot=False,
        return_edges=False,
    ):
        """Build ITF Hamiltonian from a named quimb 2D lattice generator.

        This wraps :meth:`build_itf_from_edges` by generating edges from
        ``qtn.edges_2d_<lattice>``. For example:

        - ``lattice="square"`` -> ``qtn.edges_2d_square``
        - ``lattice="triangular"`` -> ``qtn.edges_2d_triangular``
        - ``lattice="hexagonal"`` -> ``qtn.edges_2d_hexagonal``

        You can also pass a callable as ``lattice`` or provide ``edges``
        directly to bypass name-based generation.

        Notes
        -----
        Quimb lattices such as ``hexagonal`` and ``kagome`` use site labels
        of form ``(x, y, sublattice)``. These are remapped internally to an
        expanded rectangular grid ``(x, y * n_sub + offset[sublattice])`` so
        that MPO/PEPO construction can proceed on a 2D snake layout. For
        plotting, a geometric embedding is used so these lattices look like
        their expected physical connectivity.

        Parameters
        ----------
        lattice : str | callable, default="square"
            Lattice name suffix or edge-builder callable.
        edges : iterable | None, default=None
            Optional explicit edge list. If provided, ``lattice`` is ignored.
        cyclic : bool, default=False
            Passed to quimb edge generators when available.
        edge_kwargs : dict | None, default=None
            Extra kwargs forwarded to the edge generator.
        plot_geometry : bool, default=False
            If True, plot lattice geometry and snake traversal path.
        plot_kwargs : dict | None, default=None
            Extra kwargs forwarded to :meth:`plot_lattice_snake`.
        return_plot : bool, default=False
            If True, include plot axis in the return tuple.
        return_edges : bool, default=False
            If True, return ``(H_mpo, H_pepo, edges)``.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)`` by default, or
            ``(H_mpo, H_pepo, edges)`` when ``return_edges=True``, or
            includes ``ax`` when ``return_plot=True``.
        """

        def _is_xy_site(site):
            return (
                isinstance(site, tuple)
                and len(site) == 2
                and all(isinstance(v, Integral) for v in site)
            )

        def _is_sublattice_site(site):
            return (
                isinstance(site, tuple)
                and len(site) == 3
                and isinstance(site[0], Integral)
                and isinstance(site[1], Integral)
            )

        def _sublattice_display_positions(site_map_):
            labels_local = sorted({site[2] for site in site_map_}, key=repr)
            n_sub_local = len(labels_local)
            sqrt3 = float(np.sqrt(3.0))

            if n_sub_local == 2:
                offset_seq = [
                    (0.00, 0.00),
                    (0.50, sqrt3 / 6.0),
                ]
            elif n_sub_local == 3:
                offset_seq = [
                    (0.00, 0.00),
                    (0.50, 0.00),
                    (0.25, sqrt3 / 4.0),
                ]
            else:
                offset_seq = [
                    (
                        0.35 * np.cos(2.0 * np.pi * k / max(n_sub_local, 1)),
                        0.35 * np.sin(2.0 * np.pi * k / max(n_sub_local, 1)),
                    )
                    for k in range(n_sub_local)
                ]

            label_to_off = {
                lab: offset_seq[idx]
                for idx, lab in enumerate(labels_local)
            }

            out = {}
            for site_raw, site_rect in site_map_.items():
                x_raw, y_raw, lab = site_raw
                base_x = float(x_raw) + 0.5 * float(y_raw)
                base_y = (sqrt3 / 2.0) * float(y_raw)
                off_x, off_y = label_to_off[lab]
                out[site_rect] = (base_x + off_x, base_y + off_y)
            return out

        if edge_kwargs is None:
            edge_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        lattice_for_title = "custom"
        if edges is None:
            if callable(lattice):
                edge_builder = lattice
                lattice_for_title = "custom"
            elif isinstance(lattice, str):
                lattice_name = lattice.strip().lower()
                if lattice_name.startswith("edges_2d_"):
                    lattice_name = lattice_name.replace("edges_2d_", "", 1)
                builder_name = f"edges_2d_{lattice_name}"
                edge_builder = getattr(qtn, builder_name, None)
                if edge_builder is None or not callable(edge_builder):
                    available = sorted(
                        name.replace("edges_2d_", "", 1)
                        for name in dir(qtn)
                        if name.startswith("edges_2d_") and callable(getattr(qtn, name))
                    )
                    raise ValueError(
                        f"Unknown lattice '{lattice}'. Available 2D generators: {available}"
                    )
                lattice_for_title = lattice_name
            else:
                raise TypeError("lattice must be a string, callable, or None when edges given.")

            try:
                edges_raw = edge_builder(self.L_x, self.L_y, cyclic=cyclic, **edge_kwargs)
            except TypeError:
                edges_raw = edge_builder(self.L_x, self.L_y, **edge_kwargs)
            edges_use = list(edges_raw)
        else:
            edges_use = list(edges)

        if not edges_use:
            raise ValueError("edges must not be empty.")

        edges_use = [tuple(edge) for edge in edges_use]
        raw_sites = {
            tuple(site) if isinstance(site, (tuple, list)) else site
            for edge in edges_use
            for site in edge
        }
        raw_sites = sorted(raw_sites, key=repr)

        builder_use = self
        plot_site_positions = None
        if all(_is_xy_site(site) for site in raw_sites):
            site_map = {site: (int(site[0]), int(site[1])) for site in raw_sites}
        elif all(_is_sublattice_site(site) for site in raw_sites):
            labels = sorted({site[2] for site in raw_sites}, key=repr)
            n_sub = len(labels)
            label_to_off = {lab: off for off, lab in enumerate(labels)}
            site_map = {
                site: (
                    int(site[0]),
                    int(site[1]) * n_sub + label_to_off[site[2]],
                )
                for site in raw_sites
            }
            plot_site_positions = _sublattice_display_positions(site_map)
            if n_sub > 1:
                builder_use = ham_tn(
                    L_x=self.L_x,
                    L_y=self.L_y * n_sub,
                    max_bond=self.max_bond,
                    cutoff=self.cutoff,
                    data_type=self.data_type,
                )
                warnings.warn(
                    "Detected sublattice-labelled sites (e.g. kagome/hexagonal). "
                    f"Remapping to rectangular grid with shape "
                    f"({builder_use.L_x}, {builder_use.L_y}) for MPO/PEPO build.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            sample = raw_sites[0]
            raise TypeError(
                "Unsupported site format from edges. Expected (x, y) or "
                f"(x, y, sublattice), got sample site {sample!r}."
            )

        edges_norm = []
        for idx, edge in enumerate(edges_use):
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise TypeError(
                    f"Edge at position {idx} must be ((x0, y0), (x1, y1)), got {edge!r}."
                )
            site0_raw = tuple(edge[0]) if isinstance(edge[0], (tuple, list)) else edge[0]
            site1_raw = tuple(edge[1]) if isinstance(edge[1], (tuple, list)) else edge[1]
            if site0_raw not in site_map or site1_raw not in site_map:
                raise ValueError(
                    f"Edge at position {idx} references site not present in normalized map."
                )
            site0 = site_map[site0_raw]
            site1 = site_map[site1_raw]

            if site0 == site1:
                raise ValueError(f"Edge at position {idx} has identical endpoints {site0}.")

            for site in (site0, site1):
                x_val, y_val = site
                if not (0 <= x_val < builder_use.L_x and 0 <= y_val < builder_use.L_y):
                    raise ValueError(
                        f"Edge at position {idx} has out-of-bounds site {site} "
                        f"for shape ({builder_use.L_x}, {builder_use.L_y})."
                    )
            edges_norm.append((site0, site1))

        h_mpo, h_pepo = builder_use.build_itf_from_edges(
            edges_norm,
            J=J,
            field=field,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            compress_each=compress_each,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
        )

        ax = None
        if plot_geometry or return_plot:
            plot_opts = dict(plot_kwargs)
            if "title" not in plot_opts:
                plot_opts["title"] = (
                    f"{lattice_for_title.capitalize()} Lattice + Snake MPS Path"
                )
            if plot_site_positions is not None and "site_positions" not in plot_opts:
                plot_opts["site_positions"] = plot_site_positions
            if plot_site_positions is not None and "invert_y" not in plot_opts:
                plot_opts["invert_y"] = False
            ax = builder_use.plot_lattice_snake(edges_norm, **plot_opts)

        if return_edges and return_plot:
            return h_mpo, h_pepo, tuple(edges_norm), ax
        if return_edges:
            return h_mpo, h_pepo, tuple(edges_norm)
        if return_plot:
            return h_mpo, h_pepo, ax
        return h_mpo, h_pepo
