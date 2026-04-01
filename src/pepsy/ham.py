"""Hamiltonian builders for dense operators and MPOs."""

from __future__ import annotations

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
        # Compatibility aliases.
        self.chain_to_coord = tuple(self.map[i] for i in range(self.L))
        self.coord_to_chain = dict(self.map_inv)

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

    def MPO_to_PEPO(
        self,
        x_mpo,
        *,
        cycle_peps=False,
        cycle_bond_dim=1,
        inplace=False,
    ):
        """Compatibility wrapper for :meth:`mpo_to_pepo`."""
        return self.mpo_to_pepo(
            x_mpo,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            inplace=inplace,
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
        dtype = self.data_type if data_type is None else np.dtype(data_type)
        Z = np.asarray(quimb.pauli("Z", dtype=dtype), dtype=dtype)
        X = np.asarray(quimb.pauli("X", dtype=dtype), dtype=dtype)

        ints = []
        for x in range(self.L_x - 1):
            for y in range(self.L_y):
                ints.append((((x, y), (x + 1, y)), (Z, Z), J))
        for x in range(self.L_x):
            for y in range(self.L_y - 1):
                ints.append((((x, y), (x, y + 1)), (Z, Z), J))
        for x in range(self.L_x):
            for y in range(self.L_y):
                ints.append((((x, y),), (X,), field))

        mpo = self.build_mpo(
            ints,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=dtype,
            compress_each=compress_each,
        )
        if as_pepo:
            op = self.mpo_to_pepo(
                mpo,
                cycle_peps=cycle_peps,
                cycle_bond_dim=cycle_bond_dim,
                inplace=False,
            )
            return op, dict(self.map_inv)
        return mpo, dict(self.map_inv)
