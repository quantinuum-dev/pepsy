"""Hamiltonian builders for dense operators and MPOs."""

from __future__ import annotations

import warnings

from numbers import Integral

import numpy as np
import quimb
import quimb.tensor as qtn
from .__utils import ansi_wrap, resolve_color_mode

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
            ``mpo``, ``pepo``, optional ``edges``/``plot_text`` (and legacy
            alias ``ax``), optional
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
            "plot_text": None,
            "ax": None,
            "builder": builder,
            "one_d_to_two_d": dict(builder.map),
            "two_d_to_one_d": dict(builder.map_inv),
        }
        if return_edges and return_plot:
            payload["edges"] = out[2]
            payload["plot_text"] = out[3]
            payload["ax"] = out[3]
        elif return_edges:
            payload["edges"] = out[2]
        elif return_plot:
            payload["plot_text"] = out[2]
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

        mpo_term = self._identity_mpo_with_swapped_phys_inds(
            phys_dim=phys_dim,
            dtype=dtype,
        )

        for n, (site, op) in enumerate(zip(chain_sites, ops)):
            op_arr = self._coerce_op(op, phys_dim=phys_dim, dtype=dtype)
            if n == 0:
                op_arr = coeff * op_arr
            mpo_term[site].modify(data=self._site_tensor(op_arr, site, self.L))

        return mpo_term

    def _swap_mpo_phys_inds_(self, mpo):
        """Swap MPO physical index families from ``(k, b)`` to ``(b, k)``."""
        mpo.reindex_({f"k{i}": f"l{i}" for i in range(self.L)})
        mpo.reindex_({f"b{i}": f"k{i}" for i in range(self.L)})
        mpo.reindex_({f"l{i}": f"b{i}" for i in range(self.L)})
        return mpo

    def _identity_mpo_with_swapped_phys_inds(self, *, phys_dim, dtype):
        """Build identity MPO and immediately swap physical index families."""
        mpo = qtn.MPO_identity(
            self.L,
            phys_dim=phys_dim,
            dtype=dtype,
        )
        self._swap_mpo_phys_inds_(mpo)
        return mpo

    def _zero_mpo(self, *, phys_dim, dtype):
        mpo = self._identity_mpo_with_swapped_phys_inds(
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
        print_output=True,
        color="auto",
    ):
        """Render lattice geometry and snake traversal as ASCII text.

        Parameters
        ----------
        edges : iterable of ((int, int), (int, int))
            Lattice edge list.
        ax : object | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        title : str | None, default=None
            Header title shown above the ASCII preview.
        show_chain_index : bool, default=True
            If True, append the snake chain-index listing.
        snake_arrows : bool, default=True
            If True, mark snake-path direction on traversed links.
        show_legend : bool, default=True
            If True, append a one-line symbol legend.
        site_positions : Mapping[(int, int), (float, float)] | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        invert_y : bool | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        print_output : bool, default=True
            If True, print the rendered text.
        color : bool | {"auto"}, default="auto"
            Enable ANSI color styling. ``"auto"`` enables colors when stdout
            is a TTY.

        Returns
        -------
        str
            Multiline ASCII rendering of lattice edges and snake traversal.
        """
        _ = (
            edge_color,
            snake_color,
            node_color,
            edge_alpha,
            snake_alpha,
            node_size,
            edge_linewidth,
            snake_linewidth,
            node_edge_color,
            node_edge_width,
            snake_cmap,
            site_positions,
            invert_y,
        )

        color_enabled = resolve_color_mode(color)

        if ax is not None:
            warnings.warn(
                "plot_lattice_snake now returns ASCII text; argument 'ax' is ignored.",
                UserWarning,
                stacklevel=2,
            )

        edges = [tuple(edge) for edge in edges]
        if not edges:
            raise ValueError("edges must not be empty for plotting.")

        edge_keys = set()
        for idx, edge in enumerate(edges):
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise TypeError(
                    f"Edge at position {idx} must be ((x0, y0), (x1, y1)), got {edge!r}."
                )
            site0 = self._coerce_coord(edge[0])
            site1 = self._coerce_coord(edge[1])
            edge_keys.add(frozenset((site0, site1)))

        snake_sites = [self.map[idx] for idx in range(self.L)]
        snake_dir = {}
        for site0, site1 in zip(snake_sites[:-1], snake_sites[1:]):
            dx = site1[0] - site0[0]
            dy = site1[1] - site0[1]
            if dx == 1 and dy == 0:
                token = "r"
            elif dx == -1 and dy == 0:
                token = "l"
            elif dx == 0 and dy == 1:
                token = "u"
            elif dx == 0 and dy == -1:
                token = "d"
            elif dx == 1 and dy == -1:
                token = "dr"
            elif dx == -1 and dy == 1:
                token = "ul"
            elif dx == 1 and dy == 1:
                token = "ur"
            elif dx == -1 and dy == -1:
                token = "dl"
            else:
                token = "."
            snake_dir[frozenset((site0, site1))] = token

        if title is None:
            title = f"Lattice Geometry + Snake MPS Path ({self.L_x}x{self.L_y})"

        row_count = max(1, 2 * self.L_y - 1)
        row_width = 4 * self.L_x - 3
        canvas = [[" "] * row_width for _ in range(row_count)]
        layer = [["empty"] * row_width for _ in range(row_count)]

        node_to_rc = {
            (x, y): (2 * (self.L_y - 1 - y), 4 * x)
            for x in range(self.L_x)
            for y in range(self.L_y)
        }
        for (x, y), (row, col) in node_to_rc.items():
            if 0 <= row < row_count and 0 <= col < row_width:
                _ = (x, y)
                canvas[row][col] = "●"
                layer[row][col] = "node"

        def _put(row, col, char, layer_name):
            if not (0 <= row < row_count and 0 <= col < row_width):
                return
            current = canvas[row][col]
            if current == "●":
                return
            char_priority = {
                " ": 0,
                "-": 1,
                "_": 1,
                "|": 1,
                "/": 1,
                "\\": 1,
                "*": 1,
                ">": 2,
                "<": 2,
                "^": 2,
                "v": 2,
            }
            layer_priority = {
                "empty": 0,
                "lattice": 1,
                "snake": 2,
                "node": 3,
            }
            current_layer = layer[row][col]
            new_lp = layer_priority.get(layer_name, 1)
            old_lp = layer_priority.get(current_layer, 0)
            new_cp = char_priority.get(char, 1)
            old_cp = char_priority.get(current, 0)
            if (new_lp > old_lp) or (new_lp == old_lp and new_cp >= old_cp):
                canvas[row][col] = char
                layer[row][col] = layer_name

        periodic_edges_summary = set()
        for edge_key in edge_keys:
            site0, site1 = tuple(edge_key)
            if site0 not in node_to_rc or site1 not in node_to_rc:
                continue
            row0, col0 = node_to_rc[site0]
            row1, col1 = node_to_rc[site1]
            token = snake_dir.get(edge_key)
            x0, y0 = site0
            x1, y1 = site1

            wrap_x = self.L_x > 2 and {x0, x1} == {0, self.L_x - 1}
            wrap_y = self.L_y > 2 and {y0, y1} == {0, self.L_y - 1}
            far_jump = abs(x1 - x0) > 1 or abs(y1 - y0) > 1
            is_cyclic_edge = bool(wrap_x or wrap_y or far_jump)
            if is_cyclic_edge:
                periodic_edges_summary.add(tuple(sorted((site0, site1))))

            if row0 == row1 and abs(col0 - col1) == 4:
                col_left = min(col0, col1)
                if token in {"r", "l"}:
                    if snake_arrows:
                        connector = "__>" if token == "r" else "<__"
                    else:
                        connector = "___"
                    conn_layer = "snake"
                else:
                    connector = "___"
                    conn_layer = "cyclic" if is_cyclic_edge else "lattice"
                for i, ch in enumerate(connector):
                    _put(row0, col_left + 1 + i, ch, conn_layer)
                continue

            if col0 == col1 and abs(row0 - row1) == 2:
                row_mid = (row0 + row1) // 2
                if token in {"u", "d"}:
                    if snake_arrows:
                        symbol = "^" if token == "u" else "v"
                    else:
                        symbol = "|"
                    conn_layer = "snake"
                else:
                    symbol = "|"
                    conn_layer = "cyclic" if is_cyclic_edge else "lattice"
                _put(row_mid, col0, symbol, conn_layer)
                continue

            if abs(row0 - row1) == 2 and abs(col0 - col1) == 4:
                row_mid = (row0 + row1) // 2
                col_mid = (col0 + col1) // 2
                slope = (row1 - row0) * (col1 - col0)
                base = "\\" if slope > 0 else "/"
                if token in {"dr", "ur", "dl", "ul"}:
                    if snake_arrows and token in {"dr", "ur"}:
                        symbol = ">"
                    elif snake_arrows and token in {"dl", "ul"}:
                        symbol = "<"
                    else:
                        symbol = base
                    conn_layer = "snake"
                else:
                    symbol = base
                    conn_layer = "cyclic" if is_cyclic_edge else "lattice"
                _put(row_mid, col_mid, symbol, conn_layer)
                continue

            # Non-local periodic connections are summarized separately below.
            if is_cyclic_edge:
                continue

            row_mid = (row0 + row1) // 2
            col_mid = (col0 + col1) // 2
            _put(row_mid, col_mid, "_", "snake" if token is not None else "lattice")

        def _style_row(chars, layers):
            if not color_enabled:
                return "".join(chars)
            codes = {
                "node": "1;37",
                "lattice": "36",
                "cyclic": "1;35",
                "snake": "90",
            }
            out = []
            for ch, layer_name in zip(chars, layers):
                if ch == " ":
                    out.append(ch)
                    continue
                if layer_name == "snake" and ch in {">", "<", "^", "v"}:
                    code = "1;31"
                else:
                    code = codes.get(layer_name)
                out.append(ansi_wrap(ch, code, True) if code else ch)
            return "".join(out)

        lines = [title]
        for row in range(row_count):
            if row % 2 == 0:
                y = self.L_y - 1 - (row // 2)
                prefix = f"Y{y:<2} "
            else:
                prefix = "    "
            if color_enabled and row % 2 == 0:
                prefix = ansi_wrap(prefix, "1;33", True)
            lines.append(prefix + _style_row(canvas[row], layer[row]))

        x_line = "    " + "   ".join(f"X{x}" for x in range(self.L_x))
        if color_enabled:
            x_line = ansi_wrap(x_line, "1;33", True)
        lines.append(x_line)

        if show_legend:
            legend = "legend: ● node  _|/\\ edge  > < ^ v snake direction"
            if color_enabled:
                legend = (
                    "legend: "
                    + ansi_wrap("●", "1;37", True)
                    + " node  "
                    + ansi_wrap("___|/\\", "36", True)
                    + " lattice  "
                    + ansi_wrap("___|", "90", True)
                    + " snake  "
                    + ansi_wrap("> < ^ v", "1;31", True)
                    + " direction  "
                    + ansi_wrap("~~~", "1;35", True)
                    + " cyclic"
                )
            lines.append(legend)

        if periodic_edges_summary:
            header = "cyclic edges (wrap connections):"
            if color_enabled:
                header = ansi_wrap(header, "1;35", True)
            lines.append(header)
            for site0, site1 in sorted(periodic_edges_summary):
                edge_line = f"  {site0} <-> {site1}"
                if color_enabled:
                    edge_line = ansi_wrap(edge_line, "1;35", True)
                lines.append(edge_line)

        if show_chain_index:
            lines.append(
                "snake: "
                + ", ".join(f"{idx}:{coord}" for idx, coord in enumerate(snake_sites))
            )

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text

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
            If True, render a text preview of lattice geometry + snake path.
        plot_kwargs : dict | None, default=None
            Extra kwargs forwarded to :meth:`plot_lattice_snake`.
        return_plot : bool, default=False
            If True, include rendered ASCII text in the return tuple.
        return_edges : bool, default=False
            If True, return ``(H_mpo, H_pepo, edges)``.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)`` by default, or
            ``(H_mpo, H_pepo, edges)`` when ``return_edges=True``, or
            includes rendered text when ``return_plot=True``.
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

        plot_text = None
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
            if "print_output" not in plot_opts:
                plot_opts["print_output"] = bool(plot_geometry)
            plot_text = builder_use.plot_lattice_snake(edges_norm, **plot_opts)

        if return_edges and return_plot:
            return h_mpo, h_pepo, tuple(edges_norm), plot_text
        if return_edges:
            return h_mpo, h_pepo, tuple(edges_norm)
        if return_plot:
            return h_mpo, h_pepo, plot_text
        return h_mpo, h_pepo
