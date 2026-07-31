"""Hamiltonian builders for dense operators and MPOs."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

from numbers import Integral

import numpy as np
import quimb
import quimb.tensor as qtn
from .._internal.formatting import (
    ansi_wrap,
    coerce_integral_tuple,
    is_integral_tuple,
    is_xy_site,
    is_xy_sublattice_site,
    resolve_color_mode,
)
from ..tensors.core import OneDMap

__all__ = [
    "ham_tn",
]


class ham_tn:
    """Build MPO Hamiltonians from local terms on a mapped lattice.

    Parameters
    ----------
    Lx : int
        Number of lattice sites along x.
    Ly : int
        Number of lattice sites along y.
    Lz : int | None, default=None
        Optional number of lattice sites along z. When provided, terms can
        use 3D coordinates ``(x, y, z)`` and the 1D mapping is built in 3D.
    max_bond : int, default=300
        Compression cap used after each term addition.
    cutoff : float, default=1e-12
        Compression cutoff used after each term addition.
    data_type : str | numpy.dtype, default="float64"
        Default dtype used for identity MPO tensors and operators.
    mapper : pepsy.tensors.core.OneDMap | None, default=None
        Optional preconfigured lattice mapper. When omitted, a default
        ``OneDMap(Lx, Ly, Lz=Lz, mode="snake")`` is constructed.

    Attributes
    ----------
    map : dict[int, tuple[int, int] | tuple[int, int, int]]
        Mapping from 1D chain index to lattice coordinate.
    map_inv : dict[tuple[int, int] | tuple[int, int, int], int]
        Inverse mapping from lattice coordinate to 1D index.
    mapper : pepsy.tensors.core.OneDMap
        Stored mapping helper instance used to build ``map`` and ``map_inv``.
    """

    @staticmethod
    def _coalesce_dim_names(*, Lx=None, Ly=None, Lz=None, L_x=None, L_y=None, L_z=None):
        if Lx is None:
            Lx = L_x
        elif L_x is not None and L_x != Lx:
            raise TypeError("Got both Lx and L_x with different values.")

        if Ly is None:
            Ly = L_y
        elif L_y is not None and L_y != Ly:
            raise TypeError("Got both Ly and L_y with different values.")

        if Lz is None:
            Lz = L_z
        elif L_z is not None and L_z != Lz:
            raise TypeError("Got both Lz and L_z with different values.")

        return Lx, Ly, Lz

    def __init__(
        self,
        Lx=None,
        Ly=None,
        Lz=None,
        *,
        L_x=None,
        L_y=None,
        L_z=None,
        max_bond=256,
        cutoff=1e-12,
        data_type="float64",
        mapper=None,
    ):
        Lx, Ly, Lz = self._coalesce_dim_names(Lx=Lx, Ly=Ly, Lz=Lz, L_x=L_x, L_y=L_y, L_z=L_z)
        if not isinstance(Lx, Integral) or not isinstance(Ly, Integral):
            raise TypeError("Lx and Ly must be integers.")
        if Lx < 1 or Ly < 1:
            raise ValueError("Lx and Ly must be >= 1.")
        if Lz is not None:
            if not isinstance(Lz, Integral):
                raise TypeError("Lz must be an integer or None.")
            if int(Lz) < 1:
                raise ValueError("Lz must be >= 1 when provided.")

        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.Lz = None if Lz is None else int(Lz)
        self.L_x, self.L_y, self.L_z = self.Lx, self.Ly, self.Lz
        self.ndim = 2 if self.L_z is None else 3
        self.L = self.L_x * self.L_y if self.L_z is None else self.L_x * self.L_y * self.L_z
        if self.L < 2:
            dims_str = "Lx * Ly" if self.L_z is None else "Lx * Ly * Lz"
            raise ValueError(f"MPO construction requires {dims_str} >= 2.")

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
        if mapper is None:
            mapper = OneDMap(
                self.Lx,
                self.Ly,
                Lz=self.Lz,
                mode="snake",
            )
        elif not isinstance(mapper, OneDMap):
            raise TypeError("mapper must be a pepsy.tensors.core.OneDMap instance or None.")

        if mapper.shape != ((self.L_x, self.L_y) if self.L_z is None else (self.L_x, self.L_y, self.L_z)):
            raise ValueError(
                f"mapper shape {mapper.shape} does not match builder shape "
                f"{(self.L_x, self.L_y) if self.L_z is None else (self.L_x, self.L_y, self.L_z)}."
            )

        self.mapper = mapper
        self.map_mode = self.mapper.mode
        self.map, self.map_inv = self.mapper.build()

    @classmethod
    def build_itf_lattice(
        cls,
        *,
        Lx=None,
        Ly=None,
        Lz=None,
        L_x=None,
        L_y=None,
        L_z=None,
        lattice="square",
        edges=None,
        cyclic=False,
        J=1.0,
        field=1.0,
        max_bond=256,
        cutoff=1e-12,
        data_type="float64",
        mapper=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        edge_kwargs=None,
        show=False,
        return_edges=True,
        return_mpo=True,
        return_pepo=False,
        return_builder=True,
    ):
        """Construct a builder and ITF Hamiltonian in one call.

        This is a convenience wrapper around ``ham_tn(...).build_itf(...)`` so
        callers can pass lattice size and model parameters directly.

        Parameters
        ----------
        Lx, Ly : int
            Lattice dimensions used to build the internal ``ham_tn`` builder.
        Lz : int | None, default=None
            Optional z dimension. When provided, the builder accepts 3D site
            coordinates ``(x, y, z)`` in direct MPO term construction.
        lattice, edges, cyclic, J, field, compress_each, cycle_peps, cycle_bond_dim, \
        edge_kwargs, show, return_edges, return_mpo, return_pepo
            Forwarded directly to :meth:`build_itf`.
        max_bond, cutoff, data_type
            Used to construct the internal builder instance.
        mapper : pepsy.tensors.core.OneDMap | None, default=None
            Optional mapper forwarded to the internal builder. When omitted,
            the default snake-style mapper is used.
        return_mpo : bool, default=True
            If True, include the constructed MPO in the returned payload.
        return_pepo : bool, default=False
            If True, include the constructed PEPO in the returned payload.
            PEPO construction is opt-in and remains restricted to snake-style
            2D mappings.
        return_builder : bool, default=True
            Deprecated compatibility argument. Output is always a dict and
            always includes the constructed builder.

        Returns
        -------
        dict
            Dictionary with named outputs and mappings:
            optional ``mpo``, optional ``pepo``, optional ``edges``/``drawing``,
            optional ``edges_1d`` (when ``edges`` available),
            ``builder``, ``one_d_to_lattice``, and ``lattice_to_one_d``.
            Legacy aliases ``one_d_to_two_d`` and ``two_d_to_one_d`` are also
            provided for compatibility.
        """
        Lx, Ly, Lz = cls._coalesce_dim_names(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            L_x=L_x,
            L_y=L_y,
            L_z=L_z,
        )
        builder = cls(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            mapper=mapper,
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
            show=show,
            return_edges=return_edges,
            return_mpo=return_mpo,
            return_pepo=return_pepo,
        )
        _ = return_builder  # accepted for backward compatibility
        payload = {
            "mpo": out[0],
            "pepo": out[1],
            "edges": None,
            "edges_1d": None,
            "drawing": None,
            "builder": builder,
            "one_d_to_lattice": dict(builder.map),
            "lattice_to_one_d": dict(builder.map_inv),
            "one_d_to_two_d": dict(builder.map),
            "two_d_to_one_d": dict(builder.map_inv),
        }
        if return_edges and show:
            payload["edges"] = out[2]
            payload["drawing"] = out[3]
        elif return_edges:
            payload["edges"] = out[2]
        elif show:
            payload["drawing"] = out[2]

        if payload["edges"] is not None:
            map_inv = payload["lattice_to_one_d"]
            payload["edges_1d"] = tuple(
                (map_inv[tuple(site0)], map_inv[tuple(site1)])
                for site0, site1 in payload["edges"]
            )
        return payload

    def _coord_dims(self):
        return self.ndim

    def _coord_label(self):
        return "(x, y)" if self.ndim == 2 else "(x, y, z)"

    def _coord_bounds_label(self):
        if self.ndim == 2:
            return f"(Lx={self.Lx}, Ly={self.Ly})"
        return f"(Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz})"

    def _coerce_coord(self, site):
        try:
            return coerce_integral_tuple(site, length=self._coord_dims(), name="coordinate")
        except TypeError as exc:
            raise TypeError(
                f"Invalid coordinate: {site!r}. Expected {self._coord_label()} for this builder."
            ) from exc

    def map_site(self, site):
        """Map site spec to 1D chain index.

        ``site`` can be either an integer chain index or a coordinate tuple
        ``(x, y)`` (2D) or ``(x, y, z)`` (3D).
        """
        if isinstance(site, Integral):
            index = int(site)
            if index < 0 or index >= self.L:
                raise ValueError(f"Site index {index} is outside [0, {self.L - 1}].")
            return index

        coord = self._coerce_coord(site)
        if coord not in self.map_inv:
            raise ValueError(
                f"Coordinate {coord} is outside lattice bounds {self._coord_bounds_label()}."
            )
        return self.map_inv[coord]

    def _mapped_chain_edges_2d(self, *, require_local=False):
        self._require_2d("_mapped_chain_edges_2d")
        chain_edges = set()
        for idx in range(self.L - 1):
            site0 = self.map[idx]
            site1 = self.map[idx + 1]
            if abs(site0[0] - site1[0]) + abs(site0[1] - site1[1]) != 1:
                if require_local:
                    raise NotImplementedError(
                        "PEPO conversion requires a 2D mapping whose consecutive chain "
                        "sites remain nearest neighbours on the lattice. "
                        f"mapper.mode={self.map_mode!r} introduces non-local chain steps."
                    )
                continue
            chain_edges.add(frozenset((site0, site1)))
        return chain_edges

    def _require_snake_style_map(self, method_name):
        """Restrict PEPO-style lattice wiring to serpentine 2D traversals."""
        self._require_2d(method_name)
        mode_norm = OneDMap._normalize_mode(self.map_mode)
        if mode_norm not in {"snake", "snake-row-major"}:
            raise NotImplementedError(
                f"{method_name} requires a snake-style 2D mapping. "
                f"Supported PEPO mapper modes are 'snake' and 'snake-row-major'; "
                f"got mapper.mode={self.map_mode!r}."
            )

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
    def _is_coord_site(site, *, n_dims):
        return is_integral_tuple(site, length=n_dims)

    def _parse_term(self, term):
        if not isinstance(term, (tuple, list)):
            raise TypeError(
                "Each term must be tuple/list: (ops, sites) or (ops, sites, coeff)."
            )
        if len(term) not in (2, 3):
            raise ValueError(
                "Each term must be (ops, sites) or (ops, sites, coeff)."
            )

        ops, sites = term[0], term[1]

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
                f"sites must be a tuple/list of {self._coord_label()} coordinates."
            )
        sites = tuple(sites)

        if len(sites) != len(ops):
            raise ValueError("sites and ops lengths must match.")
        if len(sites) not in (1, 2):
            raise ValueError("Only 1-site and 2-site terms are supported.")
        if not all(ham_tn._is_coord_site(site, n_dims=self._coord_dims()) for site in sites):
            raise TypeError(
                f"Only {self._coord_dims()}D coordinates are supported for this builder. "
                f"Use terms like (({self._coord_label()}),) or a two-site pair."
            )

        return sites, ops, coeff

    def _require_2d(self, method_name):
        if self.ndim != 2:
            raise NotImplementedError(
                f"{method_name} is currently only available for 2D builders "
                f"(initialize ham_tn with Lz=None)."
            )

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
        ints=None,
        *,
        phys_dim=2,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        mapper=None,
        fermion=None,
        edges=None,
        fermionic=None,
        charge_sectors=False,
        **model_params,
    ):
        """Build MPO from user interactions.

        Parameters
        ----------
        ints : sequence | Mapping | pepsy.Fermion | None
            Sequence of terms. Supported term formats:
            - ``((op,), (coord,))``
            - ``((op1, op2), (coord1, coord2))``
            - ``((op,), (coord,), coeff)``
            - ``((op1, op2), (coord1, coord2), coeff)``
            This canonical order is ``(ops, coords, coeff)``. Each coordinate
            is ``(x, y)`` for 2D builders and ``(x, y, z)`` when ``Lz`` is
            provided.
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
        mapper : pepsy.tensors.core.OneDMap | None, default=None
            Optional mapper override used only for this MPO build. When
            omitted, the builder's configured mapper is used.
        fermion : pepsy.Fermion | None, default=None
            Optional native fermion model. When supplied, ``ints`` (or the
            explicit ``edges`` alias) is passed to ``fermion.build_mpo`` and
            the returned Symmray MPO keeps the model's U1/U1U1 symmetry.
        edges : sequence | None, default=None
            Explicit edge alias for the ``fermion=...`` form. For example,
            ``builder.build_mpo(fermion=f, edges=edges, t=..., U=...)``.
        fermionic : bool | None, default=None
            Native graded encoding flag for the fermion-model form. ``None``
            and ``False`` select the Jordan-Wigner-compatible MPO builder;
            ``True`` selects ``Fermion.to_mpo(...)``.
        charge_sectors : bool, default=False
            When native construction is enabled, return one MPO per operator
            charge as ``{charge: mpo}`` instead of requiring one homogeneous
            charge for the whole collection.
        **model_params
            Explicit fermion couplings such as ``t``, ``U``/``V``, and ``mu``.

        Returns
        -------
        qtn.MatrixProductOperator
            Built Hamiltonian MPO.
        """
        if (
            fermion is None
            and hasattr(ints, "build_mpo")
            and hasattr(ints, "hamiltonian")
        ):
            fermion = ints
            ints = None

        if fermion is not None:
            if edges is not None:
                if ints is not None:
                    raise TypeError(
                        "Pass fermion terms through either ints or edges, not both."
                    )
                ints = edges
            if ints is None:
                raise ValueError(
                    "Fermion MPO construction requires terms or an edge sequence."
                )
            if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
                raise ValueError("phys_dim must be an integer >= 1.")
            if not hasattr(fermion, "build_mpo"):
                raise TypeError(
                    "fermion must provide the Fermion.build_mpo interface."
                )
            dtype = self.data_type if data_type is None else np.dtype(data_type)
            max_bond_use = self.max_bond if max_bond is None else int(max_bond)
            cutoff_use = self.cutoff if cutoff is None else float(cutoff)
            mapper_use = self.mapper if mapper is None else mapper
            fermionic_use = False if fermionic is None else bool(fermionic)
            if charge_sectors and not fermionic_use:
                raise ValueError("charge_sectors=True requires fermionic=True.")
            mpo_builder = fermion.to_mpo if fermionic_use else fermion.build_mpo
            return mpo_builder(
                ints,
                L=self.L,
                mapper=mapper_use,
                max_bond=max_bond_use,
                cutoff=cutoff_use,
                compress=bool(compress_each),
                dtype=dtype,
                fermionic=fermionic_use,
                charge_sectors=charge_sectors,
                **model_params,
            )

        if edges is not None or model_params or fermionic is not None or charge_sectors:
            raise TypeError(
                "edges, fermion model parameters, and fermionic encoding are "
                "only valid with fermion=... ."
            )
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

        builder = self
        if mapper is not None:
            builder = ham_tn(
                Lx=self.Lx,
                Ly=self.Ly,
                Lz=self.Lz,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                data_type=self.data_type,
                mapper=mapper,
            )

        mpo_total = builder._zero_mpo(phys_dim=phys_dim, dtype=dtype)
        for term in ints:
            mpo_term = builder._term_to_mpo(term, phys_dim=phys_dim, dtype=dtype)
            mpo_total = mpo_total + mpo_term
            if compress_each:
                mpo_total.compress(max_bond=max_bond, cutoff=cutoff)

        if not compress_each:
            mpo_total.compress(max_bond=max_bond, cutoff=cutoff)
        return mpo_total

    def _add_missing_lattice_bonds_(self, pepo):
        """Add rank-1 bonds for lattice neighbours not already used by the 1D path."""
        self._require_2d("_add_missing_lattice_bonds_")
        chain_edges = self._mapped_chain_edges_2d(require_local=True)

        for x in range(self.L_x):
            for y in range(self.L_y):
                if x + 1 < self.L_x:
                    edge = frozenset(((x, y), (x + 1, y)))
                    if edge not in chain_edges:
                        pepo[f"I{x},{y}"].new_bond(pepo[f"I{x + 1},{y}"], size=1)
                if y + 1 < self.L_y:
                    edge = frozenset(((x, y), (x, y + 1)))
                    if edge not in chain_edges:
                        pepo[f"I{x},{y}"].new_bond(pepo[f"I{x},{y + 1}"], size=1)
        return pepo

    def _add_cycle_bonds_(self, pepo, *, bond_dim=1):
        """Optionally add periodic bonds in x and y directions."""
        self._require_2d("_add_cycle_bonds_")
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
        """Convert a snake-style ordered MPO into a 2D PEPO with lattice tags/indices.

        Parameters
        ----------
        mpo : qtn.MatrixProductOperator
            Input MPO with chain length ``L_x * L_y``.
            PEPO conversion is currently restricted to snake-style 2D maps:
            ``"snake"`` and ``"snake-row-major"``.
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
        self._require_snake_style_map("mpo_to_pepo")
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

        self._add_missing_lattice_bonds_(pepo)

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
        ints=None,
        *,
        phys_dim=2,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        mapper=None,
        fermion=None,
        edges=None,
        fermionic=None,
        charge_sectors=False,
        **model_params,
    ):
        """Build a PEPO from interactions or a native fermion model.

        The ``fermion=...``/``edges=...`` form mirrors :meth:`build_mpo` and
        forwards ``mapper=OneDMap(...)`` and ``fermionic=True`` to the native
        fermion MPO builder before converting the result to a PEPO.
        With ``charge_sectors=True``, return ``{charge: pepo}`` for a mixed
        native operator.
        """
        self._require_2d("build_pepo")
        mpo = self.build_mpo(
            ints,
            phys_dim=phys_dim,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            compress_each=compress_each,
            mapper=mapper,
            fermion=fermion,
            edges=edges,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            **model_params,
        )
        if isinstance(mpo, Mapping):
            return {
                charge: self.mpo_to_pepo(
                    sector_mpo,
                    cycle_peps=cycle_peps,
                    cycle_bond_dim=cycle_bond_dim,
                    inplace=False,
                )
                for charge, sector_mpo in mpo.items()
            }
        return self.mpo_to_pepo(
            mpo,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            inplace=False,
        )

    def mpo_itf(
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
        """Build transverse-field Ising MPO on the builder lattice.

        Hamiltonian:
        ``H = J * sum_<ij> Z_i Z_j + field * sum_i X_i``

        For 2D builders this uses square-lattice nearest-neighbour edges.
        For 3D builders (``L_z`` provided) this uses cubic-lattice nearest-
        neighbour edges.

        Returns
        -------
        tuple
            ``(op, coord_to_chain_map)`` where ``op`` is MPO by default and
            PEPO when ``as_pepo=True`` (2D only).
        """
        dtype = self.data_type if data_type is None else np.dtype(data_type)

        if self.ndim == 2:
            edges = tuple(qtn.edges_2d_square(self.L_x, self.L_y, cyclic=False))
        else:
            edges = tuple(qtn.edges_3d_cubic(self.L_x, self.L_y, self.L_z, cyclic=False))

        ints = self._itf_ints_from_edges(edges, J=J, field=field, dtype=dtype)
        mpo = self.build_mpo(
            ints,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=dtype,
            compress_each=compress_each,
        )

        if as_pepo:
            self._require_2d("mpo_itf(as_pepo=True)")
            pepo = self.mpo_to_pepo(
                mpo,
                cycle_peps=cycle_peps,
                cycle_bond_dim=cycle_bond_dim,
                inplace=False,
            )
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
        ints = [((z_op, z_op), edge, J) for edge in edges]
        ints.extend((((x_op,), (site,), field) for site in sites))
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
        return_mpo=True,
        return_pepo=False,
    ):
        """Build ITF Hamiltonian MPO and optionally PEPO from an edge list.

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
        return_mpo : bool, default=True
            If True, build and return the MPO.
        return_pepo : bool, default=False
            If True, also convert the MPO to a PEPO. This requires a
            snake-style 2D mapping.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)`` where either entry can be ``None`` when not
            requested.
        """
        dtype = self.data_type if data_type is None else np.dtype(data_type)
        edges = [tuple(edge) for edge in edges]
        if not edges:
            raise ValueError("edges must not be empty.")

        if not return_mpo and not return_pepo:
            return None, None

        ints = self._itf_ints_from_edges(edges, J=J, field=field, dtype=dtype)

        mpo = self.build_mpo(
            ints,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=dtype,
            compress_each=compress_each,
        )
        pepo = None
        if return_pepo:
            pepo = self.mpo_to_pepo(
                mpo,
                cycle_peps=cycle_peps,
                cycle_bond_dim=cycle_bond_dim,
                inplace=False,
            )
        if not return_mpo:
            mpo = None
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
        """Render lattice geometry and mapped traversal as ASCII text.

        Parameters
        ----------
        edges : iterable of ((int, int), (int, int))
            Lattice edge list.
        ax : object | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        title : str | None, default=None
            Header title shown above the ASCII preview.
        show_chain_index : bool, default=True
            If True, append the chain-index listing for the active map.
        snake_arrows : bool, default=True
            If True, mark mapped-path direction on traversed links.
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
            Multiline ASCII rendering of lattice edges and mapped traversal.
        """
        self._require_2d("plot_lattice_snake")
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
            title = f"Lattice Geometry + {self.map_mode} Path ({self.L_x}x{self.L_y})"

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

    @staticmethod
    def _mpo_chain_bond_dims(mpo):
        """Return MPO bond dimensions between consecutive chain tensors."""
        dims = []
        for idx in range(mpo.L - 1):
            left = mpo[idx]
            right = mpo[idx + 1]
            shared = tuple(set(left.inds) & set(right.inds))
            if not shared:
                dims.append(1)
                continue
            dims.append(max(int(left.ind_size(ix)) for ix in shared))
        return dims

    def _show_mpo_schematic_2d(
        self,
        mpo,
        edges,
        *,
        title=None,
        site_positions=None,
        ax=None,
        figsize=None,
    ):
        """Render a schematic MPO-on-lattice view with chain bond dimensions."""
        self._require_2d("_show_mpo_schematic_2d")
        if mpo is None:
            raise ValueError("An MPO is required to render the ITF schematic.")
        if getattr(mpo, "L", None) != self.L:
            raise ValueError(
                f"MPO length mismatch: expected {self.L}, got {getattr(mpo, 'L', None)}."
            )

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

        positions = (
            {tuple(site): (float(xy[0]), float(xy[1])) for site, xy in site_positions.items()}
            if site_positions is not None
            else {
                (x, y): (float(x), float(y))
                for x in range(self.L_x)
                for y in range(self.L_y)
            }
        )
        if title is None:
            title = f"ITF MPO ({self.map_mode}, max bond={int(mpo.max_bond())})"
        if figsize is None:
            xs = [xy[0] for xy in positions.values()]
            ys = [xy[1] for xy in positions.values()]
            figsize = (
                max(5.0, 1.35 * (max(xs) - min(xs) + 1.0)),
                max(4.2, 1.35 * (max(ys) - min(ys) + 1.0)),
            )

        presets = {
            "lattice": {
                "color": (0.80, 0.82, 0.86, 1.0),
                "linewidth": 1.8,
            },
            "node": {
                "facecolor": schematic.get_color("blue"),
                "edgecolor": "white",
                "linewidth": 1.2,
                "radius": 0.18,
            },
        }
        drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

        for site0, site1 in edges:
            drawing.line(positions[tuple(site0)], positions[tuple(site1)], preset="lattice")

        coords = [self.map[idx] for idx in range(self.L)]
        bond_dims = self._mpo_chain_bond_dims(mpo)
        cmap = colormaps.get_cmap("plasma")
        dim_min = min(bond_dims, default=1)
        dim_max = max(bond_dims, default=1)
        dim_span = max(1, dim_max - dim_min)

        for idx, (site0, site1) in enumerate(zip(coords[:-1], coords[1:])):
            dim = bond_dims[idx]
            frac = (dim - dim_min) / dim_span
            color = cmap(frac)
            width = 2.2 + 2.4 * frac
            pos0 = positions[tuple(site0)]
            pos1 = positions[tuple(site1)]
            drawing.line(pos0, pos1, color=color, linewidth=width, zorder=3)
            drawing.arrowhead(pos0, pos1, color=color, center=0.58, width=0.10)

            mid_x = 0.5 * (pos0[0] + pos1[0])
            mid_y = 0.5 * (pos0[1] + pos1[1])
            dx = pos1[0] - pos0[0]
            dy = pos1[1] - pos0[1]
            norm = max((dx * dx + dy * dy) ** 0.5, 1e-12)
            off_x = -0.08 * dy / norm
            off_y = 0.08 * dx / norm
            drawing.ax.text(
                mid_x + off_x,
                mid_y + off_y,
                str(dim),
                fontsize=8,
                ha="center",
                va="center",
                color=(0.18, 0.20, 0.24, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": (1.0, 1.0, 1.0, 0.88),
                    "edgecolor": color,
                    "linewidth": 0.8,
                },
                zorder=5,
            )

        for coord, chain_idx in self.map_inv.items():
            pos = positions[tuple(coord)]
            drawing.circle(pos, preset="node", zorder=4)
            drawing.ax.text(
                pos[0],
                pos[1],
                str(chain_idx),
                fontsize=9,
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )

        xs = [xy[0] for xy in positions.values()]
        ys = [xy[1] for xy in positions.values()]
        pad_x = max(0.35, 0.08 * (max(xs) - min(xs) + 1.0))
        pad_y = max(0.35, 0.08 * (max(ys) - min(ys) + 1.0))
        drawing.ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        drawing.ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        drawing.ax.set_aspect("equal")
        drawing.ax.set_title(title)
        drawing.ax.axis("off")
        return drawing

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
        show=False,
        return_edges=False,
        return_mpo=True,
        return_pepo=False,
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
        that MPO/PEPO construction can proceed on a supported 2D mapped layout. For
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
        show : bool, default=False
            If True, include a schematic drawing of the MPO path on the lattice.
        return_edges : bool, default=False
            If True, include ``edges`` in the return tuple.
        return_mpo : bool, default=True
            If True, build and return the MPO.
        return_pepo : bool, default=False
            If True, also build and return the PEPO.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)`` by default, or
            ``(H_mpo, H_pepo, edges)`` when ``return_edges=True``, or
            includes the schematic drawing when ``show=True``.
        """
        self._require_2d("build_itf")

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

        # Some quimb 2D periodic generators emit degenerate singleton/self-loop
        # edges when a lattice dimension is 1 (e.g. Ly=1 with cyclic=True).
        # Drop those generated artifacts so 1D reductions still build cleanly.
        if edges is None:
            filtered_edges = []
            dropped = 0
            for edge in edges_use:
                if isinstance(edge, (tuple, list)) and len(edge) == 1:
                    dropped += 1
                    continue
                if (
                    isinstance(edge, (tuple, list))
                    and len(edge) == 2
                    and edge[0] == edge[1]
                ):
                    dropped += 1
                    continue
                filtered_edges.append(edge)
            if dropped:
                warnings.warn(
                    f"Dropped {dropped} degenerate generated edge(s) for "
                    f"shape ({self.L_x}, {self.L_y}) with cyclic={cyclic}.",
                    UserWarning,
                    stacklevel=2,
                )
            edges_use = filtered_edges

            if not edges_use:
                raise ValueError(
                    "Generated edges are empty after filtering degenerate periodic edges."
                )

        edges_use = [tuple(edge) for edge in edges_use]
        raw_sites = {
            tuple(site) if isinstance(site, (tuple, list)) else site
            for edge in edges_use
            for site in edge
        }
        raw_sites = sorted(raw_sites, key=repr)

        builder_use = self
        plot_site_positions = None
        if all(is_xy_site(site) for site in raw_sites):
            site_map = {site: (int(site[0]), int(site[1])) for site in raw_sites}
        elif all(is_xy_sublattice_site(site) for site in raw_sites):
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
                    mapper=OneDMap(self.L_x, self.L_y * n_sub, mode=self.map_mode),
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
            return_mpo=(return_mpo or show),
            return_pepo=return_pepo,
        )

        drawing = None
        if show:
            drawing = builder_use._show_mpo_schematic_2d(
                h_mpo,
                edges_norm,
                title=f"{lattice_for_title.capitalize()} ITF MPO ({builder_use.map_mode})",
                site_positions=plot_site_positions,
            )

        if not return_mpo:
            h_mpo = None

        if return_edges and show:
            return h_mpo, h_pepo, tuple(edges_norm), drawing
        if return_edges:
            return h_mpo, h_pepo, tuple(edges_norm)
        if show:
            return h_mpo, h_pepo, drawing
        return h_mpo, h_pepo
