"""Direct PEPS sampling with conditioned boundary states."""

from __future__ import annotations

import math
import autoray as ar
import numpy as np
from .results import (
    PEPSSampleResult,
)

__all__ = ['PepsSampler']


class PepsSampler:
    """Directly sample a finite PEPS with exact or boundary-MPS proposals.

    ``sample_chi=None`` with ``boundary_engine="exact"`` selects the
    correctness-first reference implementation. Supplying ``sample_chi``
    selects the direct boundary-MPS proposal: a conditioned single-layer ket
    boundary is compressed after every projected row, while an optional
    future double-layer boundary is controlled by ``marginal_chi``.

    The two boundary objects are deliberately separate. ``sample_chi`` is the
    bond cap for the conditioned ket boundary. ``marginal_chi`` is the bond cap
    for the unmeasured-row future environment. The latter can be disabled with
    ``None`` or ``0``.

    Parameters
    ----------
    peps : TensorNetwork2D
        Finite open-boundary PEPS with ``site_tag`` and ``site_ind`` methods.
    sample_chi : int, optional
        Maximum bond dimension of the conditioned single-layer ket boundary.
        ``None`` selects exact serial mode and requires
        ``boundary_engine="exact"``.
    marginal_chi : int, optional
        Maximum bond dimension of the future double-layer environment. ``None``
        or ``0`` disables the future environment and uses identity caps.
    boundary_engine : {"exact", "quimb-mps", "dmrg", "auto"}, default="exact"
        Future-environment backend. ``"dmrg"`` uses Pepsy's
        :class:`BdyMPS` and :class:`CompBdy`; ``"quimb-mps"`` uses Quimb's
        cached MPS environments. ``"auto"`` is a compatibility alias for
        the dense DMRG/FIT future-environment path.
    ket_compression : {"quimb", "fit", None}, default="quimb"
        Compression backend for the conditioned ket boundary. ``None`` leaves
        it uncompressed. This option is ignored in exact mode.
    cutoff : float, default=1e-12
        Singular-value or fit cutoff used by boundary compression.
    fit_n_iter : int, default=2
        Number of local FIT sweeps used by the ``"fit"`` ket compressor and
        by the ``"dmrg"`` future-environment preparation.
    contraction_opt : object, optional
        Quimb contraction optimizer passed to ``TensorNetwork.contract``.
        The default ``"auto-hq"`` uses the high-quality Cotengra path when
        available.

    Notes
    -----
    The input PEPS is never projected or otherwise modified. The tagged ket
    and double-layer norm network are private copies owned by the sampler. The
    exact mode contracts the full conditioned norm network at every site. The
    boundary mode contracts only the current row with the conditioned lower
    boundary and optional future environment.
    """

    def __init__(
        self,
        peps,
        *,
        sample_chi=None,
        marginal_chi=None,
        boundary_engine="exact",
        ket_compression="quimb",
        cutoff=1.0e-12,
        fit_n_iter=2,
        contraction_opt="auto-hq",
    ):
        self.peps = getattr(peps, "tn", peps)
        self.sample_chi = self._validate_optional_chi(sample_chi, "sample_chi")
        self.marginal_chi = self._validate_marginal_chi(marginal_chi)
        self.boundary_engine = self._normalize_boundary_engine(boundary_engine)
        self.ket_compression = self._normalize_ket_compression(ket_compression)
        if not isinstance(cutoff, (int, float, np.integer, np.floating)):
            raise TypeError("cutoff must be a non-negative real number.")
        if float(cutoff) < 0:
            raise ValueError("cutoff must be a non-negative real number.")
        self.cutoff = float(cutoff)
        if not isinstance(fit_n_iter, (int, np.integer)) or int(fit_n_iter) < 1:
            raise ValueError("fit_n_iter must be a positive integer.")
        self.fit_n_iter = int(fit_n_iter)
        self.contraction_opt = contraction_opt
        self._source_peps = self.peps
        self._future_environments = {}
        self._future_boundary = None
        self._future_store = None
        self._last_boundary_mps = None
        self._last_rho_diagnostics = {}
        self._last_batch_stats = {}
        self._last_row_cache_stats = {}
        self._phi_inds = tuple(f"__pepsy_phi_{x}" for x in range(
            int(getattr(self.peps, "Lx", 0))
        ))
        self._phi_input_inds = tuple(f"__pepsy_phi_in_{x}" for x in range(
            int(getattr(self.peps, "Lx", 0))
        ))
        if self.boundary_engine == "exact":
            if self.sample_chi is not None:
                raise ValueError(
                    "boundary_engine='exact' does not use sample_chi; choose "
                    "'quimb-mps' or 'dmrg' for a conditioned boundary MPS."
                )
            if self.marginal_chi not in (None, 0):
                raise ValueError(
                    "boundary_engine='exact' does not use marginal_chi; choose "
                    "'quimb-mps' or 'dmrg' for a future environment."
                )
        elif self.sample_chi is None:
            raise ValueError(
                "sample_chi is required when boundary_engine is not 'exact'."
            )
        self.refresh()

    @staticmethod
    def _validate_optional_chi(value, name):
        if value is None:
            return None
        if not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer or None.")
        return int(value)

    @staticmethod
    def _validate_marginal_chi(value):
        if value is None:
            return None
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise ValueError("marginal_chi must be a non-negative integer or None.")
        return int(value)

    @staticmethod
    def _normalize_boundary_engine(engine):
        key = "exact" if engine is None else str(engine).strip().lower()
        aliases = {
            "exact": "exact",
            "quimb": "quimb-mps",
            "mps": "quimb-mps",
            "quimb-mps": "quimb-mps",
            "dmrg": "dmrg",
            "fit": "dmrg",
            "auto": "dmrg",
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise ValueError(
                "boundary_engine must be 'exact', 'quimb-mps', or 'dmrg'."
            ) from exc

    @staticmethod
    def _normalize_ket_compression(compression):
        if compression is None:
            return None
        key = str(compression).strip().lower().replace("_", "-")
        aliases = {"quimb": "quimb", "mps": "quimb", "fit": "fit"}
        try:
            return aliases[key]
        except KeyError as exc:
            raise ValueError(
                "ket_compression must be 'quimb', 'fit', or None."
            ) from exc

    def refresh(self):
        """Rebuild the private ket and norm networks from the source PEPS."""
        from ..boundary.metrics import build_bra_ket  # noqa: PLC0415

        if not all(hasattr(self.peps, name) for name in ("site_tag", "site_ind")):
            raise TypeError(
                "PepsSampler requires a PEPS-like object exposing site_tag "
                "and site_ind."
            )
        try:
            self.Lx = int(self.peps.Lx)
            self.Ly = int(self.peps.Ly)
        except AttributeError as exc:
            raise TypeError("PepsSampler requires a finite 2D PEPS.") from exc
        if self.Lx < 1 or self.Ly < 1:
            raise ValueError("PepsSampler requires a non-empty PEPS.")

        ket = self.peps.copy()
        self._ket, self._norm = build_bra_ket(ket=ket)
        # Cache immutable row selections and site metadata. Sampling still
        # copies these templates before projection, so source networks and
        # reusable caches remain unchanged.
        self._site_tags = {
            (x, y): self._ket.site_tag(x, y)
            for y in range(self.Ly)
            for x in range(self.Lx)
        }
        self._site_inds = {
            (x, y): self._ket.site_ind(x, y)
            for y in range(self.Ly)
            for x in range(self.Lx)
        }
        self._ket_row_templates = tuple(
            self._ket.select(f"Y{y}", "any").copy()
            for y in range(self.Ly)
        )
        self._norm_row_templates = tuple(
            self._norm.select(f"Y{y}", "any").copy()
            for y in range(self.Ly)
        )
        self.site_order = tuple(
            (x, y) for y in range(self.Ly) for x in range(self.Lx)
        )
        self._phi_inds = tuple(f"__pepsy_phi_{x}" for x in range(self.Lx))
        self._phi_input_inds = tuple(
            f"__pepsy_phi_in_{x}" for x in range(self.Lx)
        )
        self._future_environments = {}
        self._future_boundary = None
        self._future_store = None
        self._last_boundary_mps = None
        self._last_rho_diagnostics = {}
        self._last_batch_stats = {}
        self._last_row_cache_stats = {}
        if self.boundary_engine != "exact" and self.marginal_chi not in (None, 0):
            self._prepare_future_environments()
        return self

    def _prepare_future_environments(self):
        """Prepare compressed double-layer environments for unmeasured rows."""
        # A one-row PEPS has no unmeasured future boundary to prepare.
        if self.Ly < 2:
            return
        if self.boundary_engine == "quimb-mps":
            from ..optimizers.sweep.environments import (  # noqa: PLC0415
                QuimbMpsBoundaryStore,
            )

            store = QuimbMpsBoundaryStore(
                chi=self.marginal_chi,
                cutoff=self.cutoff,
                canonize=True,
                mode="mps",
                layer_tags=("KET", "BRA"),
            )
            # Quimb owns this boundary sweep and returns the native ``ymax``
            # MPS objects. Keep the cache separate from the shot-conditioned
            # ket boundary because this network still sums over future rows.
            store.update_axis(self._norm, "y")
            self._future_store = store
            self._future_environments = {
                y: store.envs[("ymax", y)]
                for y in range(self.Ly - 1)
                if ("ymax", y) in store.envs
            }
            return

        from ..boundary.states import BdyMPS  # noqa: PLC0415
        from ..boundary.sweeps import CompBdy  # noqa: PLC0415

        boundary = BdyMPS(
            tn_flat=self._ket,
            tn_double=self._norm,
            chi=self.marginal_chi,
            single_layer=False,
        )
        compressor = CompBdy(
            self._norm,
            boundary.mps_b,
            contraction_opt=self.contraction_opt,
            fit_contraction_opt=self.contraction_opt,
            fit_mode="eff",
        )
        # Rows above the current row are the right-hand side in BdyMPS's
        # ``Y*_r`` convention, hence ``y_right`` rather than ``y_left``.
        compressor.move_bdy(
            direction="y_right",
            n_iter=self.fit_n_iter,
            equalize_norms=True,
            progress=False,
        )
        self._future_boundary = boundary
        # Right-side boundary indices are counted from the top: ``Y0_r`` is
        # the last row, so row ``y`` needs ``Y(Ly - 2 - y)_r``.
        self._future_environments = {
            y: boundary.mps_b[f"Y{self.Ly - 2 - y}_r"]
            for y in range(self.Ly - 1)
            if f"Y{self.Ly - 2 - y}_r" in boundary.mps_b
        }

    @staticmethod
    def _scalar(value):
        """Extract a Python scalar from a Quimb tensor or array scalar."""
        data = getattr(value, "data", value)
        return np.asarray(ar.to_numpy(data)).reshape(()).item()

    @property
    def rho_diagnostics(self):
        """Return diagnostics for the most recently evaluated local rhos."""
        return {
            site: dict(values)
            for site, values in self._last_rho_diagnostics.items()
        }

    @property
    def batch_stats(self):
        """Return statistics from the most recent prefix-grouped batch."""
        return dict(self._last_batch_stats)

    @property
    def row_cache_stats(self):
        """Return statistics from the most recent boundary row-cache run."""
        return dict(self._last_row_cache_stats)

    def _reset_rho_diagnostics(self):
        """Start a fresh local-rho diagnostic trace."""
        self._last_rho_diagnostics = {}

    @staticmethod
    def _scaled(value):
        """Represent a real or complex value as mantissa times 10**exponent."""
        value = value.item() if isinstance(value, np.generic) else value
        magnitude = abs(value)
        if magnitude == 0:
            return 0.0 if not np.iscomplexobj(value) else 0.0j, 0
        exponent = math.floor(math.log10(magnitude))
        mantissa = value / (10.0 ** exponent)
        return mantissa, exponent

    def _local_rho(self, working, site):
        """Contract the current site's ket/bra physical density matrix."""
        site_tag = self._site_tags[site]
        ket_ind = self._site_inds[site]
        bra_ind = f"{ket_ind}__pepsy_bra"

        # The norm network initially shares each ket/bra physical index. The
        # current bra index is split only in this temporary network, leaving
        # ``working`` suitable for fixing the selected value on both layers.
        local = working.copy()
        local.select([site_tag, "BRA"], which="all").reindex_(
            {ket_ind: bra_ind}
        )
        # ``contraction_opt`` can be a reusable ``pepsy.build_contraction``
        # optimizer, allowing all local rho paths to share Cotengra searches.
        rho = local.contract(
            all,
            output_inds=(ket_ind, bra_ind),
            optimize=self.contraction_opt,
        )
        rho = np.asarray(ar.to_numpy(rho.data))
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(
                f"Local density matrix at site {site!r} has invalid shape "
                f"{rho.shape!r}."
            )
        return rho, ket_ind

    def _conditional_probabilities(self, rho, *, site):
        """Convert a local density matrix diagonal into a probability vector."""
        diagonal = np.real(np.diag(rho)).astype(float, copy=False)
        trace = float(np.real(np.trace(rho)))
        scale = max(1.0, abs(trace), float(np.max(np.abs(diagonal))))
        tolerance = 256.0 * np.finfo(float).eps * scale
        rho_norm = float(np.linalg.norm(rho))
        hermiticity_defect = float(
            np.linalg.norm(rho - rho.conj().T) / max(rho_norm, 1.0)
        )
        negative = diagonal[diagonal < 0.0]
        negative_mass = float(-negative.sum()) if negative.size else 0.0
        previous = self._last_rho_diagnostics.get(site)
        diagnostics = {
            "trace": trace,
            "hermiticity_defect": hermiticity_defect,
            "negative_diagonal_mass": negative_mass,
            "negative_diagonal_max": (
                float(-negative.min()) if negative.size else 0.0
            ),
            "clip_tolerance": tolerance,
            "clipped_negative_mass": 0.0,
            "evaluation_count": 1,
        }
        if previous is not None:
            diagnostics["evaluation_count"] = previous["evaluation_count"] + 1
            diagnostics["max_hermiticity_defect"] = max(
                previous.get("max_hermiticity_defect", 0.0),
                hermiticity_defect,
            )
            diagnostics["max_negative_diagonal_mass"] = max(
                previous.get("max_negative_diagonal_mass", 0.0),
                negative_mass,
            )
        else:
            diagnostics["max_hermiticity_defect"] = hermiticity_defect
            diagnostics["max_negative_diagonal_mass"] = negative_mass
        self._last_rho_diagnostics[site] = diagnostics
        if not np.isfinite(trace) or not np.all(np.isfinite(diagonal)):
            raise ValueError(
                f"Conditional density matrix at site {site!r} contains "
                "non-finite values."
            )
        if trace <= tolerance:
            raise ValueError(
                f"Conditional density matrix at site {site!r} has invalid "
                f"trace {trace!r}."
            )
        if np.any(diagonal < -tolerance):
            raise ValueError(
                f"Conditional density matrix at site {site!r} has a "
                "substantially negative diagonal."
            )
        # Only roundoff-scale negative diagonal mass is clipped. A larger
        # violation indicates an invalid/truncated environment and is raised.
        diagonal = np.maximum(diagonal, 0.0)
        self._last_rho_diagnostics[site]["clipped_negative_mass"] = negative_mass
        probabilities = diagonal / diagonal.sum()
        probabilities /= probabilities.sum()
        return probabilities

    def _row_bonds(self, y, direction):
        """Return the vertical PEPS bonds adjacent to row ``y``."""
        if direction == "bottom":
            if y == 0:
                return ()
            return tuple(
                self._ket.bond((x, y), (x, y - 1)) for x in range(self.Lx)
            )
        if direction == "top":
            if y == self.Ly - 1:
                return ()
            return tuple(
                self._ket.bond((x, y), (x, y + 1)) for x in range(self.Lx)
            )
        raise ValueError("direction must be 'bottom' or 'top'.")

    def _identity_future(self, y):
        """Build the ``marginal_chi=0`` identity future cap for row ``y``."""
        import quimb.tensor as qtn  # noqa: PLC0415

        # With no future compression requested, identity tensors preserve the
        # current row's top virtual legs without summing over future rows.
        tensors = []
        for x, top_ind in enumerate(self._row_bonds(y, "top")):
            size = self._ket.ind_size(top_ind)
            tensors.append(
                qtn.Tensor(
                    np.eye(size, dtype=complex),
                    inds=(top_ind, f"{top_ind}_*"),
                    # Give the identity cap the same column tag as the
                    # current row. This lets the row transfer cache retain
                    # it with the column instead of leaving it dangling.
                    tags=(f"PEPSY_FUTURE_{x}", f"X{x}"),
                )
            )
        return qtn.TensorNetwork(tensors)

    def _conditioned_boundary_norm(self, phi, y):
        """Build the double-layer norm of the conditioned lower ket boundary."""
        from ..boundary.metrics import build_bra_ket  # noqa: PLC0415

        # ``phi`` is a single-layer ket conditioned on this shot. We build its
        # norm only for attaching it to the local double-layer rho network.
        bottom_inds = self._row_bonds(y, "bottom")
        phi_ket = phi.copy()
        phi_ket.reindex_({
            self._phi_inds[x]: f"k{x}"
            for x in range(self.Lx)
        })
        _, phi_norm = build_bra_ket(ket=phi_ket)

        phi_norm.reindex_({
            f"k{x}": bottom_inds[x]
            for x in range(self.Lx)
        })

        # ``build_bra_ket`` shares physical outer indices between its ket and
        # bra. Connect the bra copy to the row's bra-side vertical bonds.
        phi_norm.select("BRA", "all").reindex_({
            bottom_ind: f"{bottom_ind}_*"
            for bottom_ind in bottom_inds
        })
        return phi_norm

    def _boundary_center(self, y, phi):
        """Attach lower boundary, current row, and future boundary."""
        # The lower object is shot-dependent; only the future object may be
        # reused because its rows have not been sampled yet.
        center = self._norm_row_templates[y].copy()
        if phi is not None:
            center |= self._conditioned_boundary_norm(phi, y)

        future = self._future_environments.get(y)
        center |= future.copy() if future is not None else self._identity_future(y)
        return center

    @staticmethod
    def _network_inds(network):
        """Return all indices occurring in a tensor network in order."""
        seen = set()
        ordered = []
        for tensor in network.tensors:
            for ind in tensor.inds:
                if ind not in seen:
                    seen.add(ind)
                    ordered.append(ind)
        return tuple(ordered)

    def _contract_row_tensors(self, tensors, output_inds):
        """Contract a small row transfer network with the shared optimizer."""
        import quimb.tensor as qtn  # noqa: PLC0415

        tensors = [tensor for tensor in tensors if tensor is not None]
        if not tensors:
            return None
        # Keep this as a regular Quimb network contraction rather than
        # materializing an einsum ourselves. It supports the reusable
        # ``build_contraction`` optimizer and all Quimb array backends.
        return qtn.TensorNetwork(tensors).contract(
            all,
            output_inds=tuple(output_inds),
            optimize=self.contraction_opt,
        )

    def _build_row_transfer_cache(self, y, phi):
        """Build local row transfers and all right-to-left suffixes.

        The current proposal network is a one-dimensional chain in ``x``
        once each column's internal tensors are contracted. ``local`` keeps
        the current ket/bra physical indices open for a rho calculation;
        ``trace`` closes them and is reused by every later site in the row.
        The suffix transfers are deliberately contracted without another
        chi cutoff: ``marginal_chi`` has already controlled the future
        double-layer boundary attached to ``center``.
        """
        center = self._boundary_center(y, phi)
        columns = [
            center.select(f"X{x}", "any").copy()
            for x in range(self.Lx)
        ]
        local = []
        trace = []
        interfaces = []
        physical = []

        for x, column in enumerate(columns):
            site = (x, y)
            ket_ind = self._site_inds[site]
            bra_ind = f"{ket_ind}__pepsy_row_bra"
            site_tag = self._site_tags[site]
            column_inds = set(self._network_inds(column))
            left_inds = (
                column_inds & set(self._network_inds(columns[x - 1]))
                if x
                else set()
            )
            right_inds = (
                column_inds & set(self._network_inds(columns[x + 1]))
                if x < self.Lx - 1
                else set()
            )
            # Preserve the tensor's index order, which makes the transfer
            # shapes deterministic and keeps left/right interfaces distinct.
            left_inds = tuple(
                ind for ind in self._network_inds(column) if ind in left_inds
            )
            right_inds = tuple(
                ind for ind in self._network_inds(column) if ind in right_inds
            )
            interfaces.append((left_inds, right_inds))
            physical.append((ket_ind, bra_ind))

            split = column.copy()
            split.select([site_tag, "BRA"], which="all").reindex_(
                {ket_ind: bra_ind}
            )
            local.append(
                split.contract(
                    all,
                    output_inds=(ket_ind, bra_ind, *left_inds, *right_inds),
                    optimize=self.contraction_opt,
                )
            )
            # The un-split physical index is shared by ket and bra, so
            # excluding it from output_inds performs the local trace.
            trace.append(
                column.contract(
                    all,
                    output_inds=(*left_inds, *right_inds),
                    optimize=self.contraction_opt,
                )
            )

        right = [None] * self.Lx
        if self.Lx > 1:
            suffix = trace[-1]
            right[-2] = suffix
            for x in range(self.Lx - 2, 0, -1):
                suffix = self._contract_row_tensors(
                    (trace[x], suffix), interfaces[x][0]
                )
                right[x - 1] = suffix

        return {
            "center": center,
            "local": tuple(local),
            "right": tuple(right),
            "interfaces": tuple(interfaces),
            "physical": tuple(physical),
        }

    def _row_local_rho(self, row_cache, x, y, left):
        """Contract one cached row transfer with its prefix and suffix."""
        site = (x, y)
        ket_ind = self._site_inds[site]
        bra_ind = f"{ket_ind}__pepsy_row_bra"
        rho = self._contract_row_tensors(
            (
                left,
                row_cache["local"][x],
                row_cache["right"][x],
            ),
            (ket_ind, bra_ind),
        )
        rho = np.asarray(ar.to_numpy(rho.data))
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(
                f"Local density matrix at site {site!r} has invalid shape "
                f"{rho.shape!r}."
            )
        return rho, ket_ind, bra_ind

    def _advance_row_prefix(self, row_cache, x, value, left):
        """Fix one local transfer and return the conditioned left prefix."""
        import quimb.tensor as qtn  # noqa: PLC0415

        local = row_cache["local"][x].copy()
        ket_ind, bra_ind = row_cache["physical"][x]
        local.isel_({ket_ind: int(value), bra_ind: int(value)})
        if left is None:
            return local
        return qtn.TensorNetwork((left, local)).contract(
            all,
            output_inds=row_cache["interfaces"][x][1],
            optimize=self.contraction_opt,
        )

    def _projected_row_network(self, y, row_config):
        """Build a projected row as an MPS or MPO for boundary updates."""
        import quimb.tensor as qtn  # noqa: PLC0415

        row = self._ket_row_templates[y].copy()
        # The row is projected from the private ket, not from ``center``:
        # ``center`` is only the local proposal network.
        row.isel_({
            self._ket.site_ind(x, y): int(row_config[x])
            for x in range(self.Lx)
        })

        top_inds = self._row_bonds(y, "top")
        row.reindex_({
            top_inds[x]: self._phi_inds[x]
            for x in range(self.Lx)
        })
        if y == 0:
            # The first projected row creates the initial conditioned MPS.
            row.view_as_(
                qtn.MatrixProductState,
                L=self.Lx,
                site_tag_id="X{}",
                site_ind_id="__pepsy_phi_{}",
                cyclic=False,
            )
            return row

        bottom_inds = self._row_bonds(y, "bottom")
        # Later rows act as MPOs: bottom virtual legs consume ``phi`` and top
        # virtual legs become the next conditioned boundary.
        row.reindex_({
            bottom_inds[x]: self._phi_input_inds[x]
            for x in range(self.Lx)
        })
        row.view_as_(
            qtn.MatrixProductOperator,
            L=self.Lx,
            site_tag_id="X{}",
            upper_ind_id="__pepsy_phi_{}",
            lower_ind_id="__pepsy_phi_in_{}",
            cyclic=False,
        )
        return row

    def _apply_projected_row(self, mpo, phi):
        """Apply a projected row MPO with Quimb's native MPS operation."""
        # Native ``MPO.apply`` contracts each site and preserves MPS topology;
        # compression is deliberately a separate policy choice below.
        return mpo.apply(phi, contract=True, inplace=False)

    def _compress_conditioned_boundary(self, phi):
        """Compress a conditioned boundary with the selected backend."""
        if self.ket_compression is None:
            return phi

        if self.ket_compression == "quimb":
            # Quimb is the direct, deterministic SVD/truncation path.
            maybe_compressed = phi.compress(
                max_bond=self.sample_chi,
                cutoff=self.cutoff,
            )
            result = phi if maybe_compressed is None else maybe_compressed
        else:
            from ..fitting.local import FIT  # noqa: PLC0415

            # FIT starts from a bounded MPS guess, then optimizes the full
            # projected boundary when a variational approximation is desired.
            guess = phi.copy()
            maybe_compressed = guess.compress(
                max_bond=self.sample_chi,
                cutoff=self.cutoff,
            )
            guess = guess if maybe_compressed is None else maybe_compressed
            fit = FIT(
                phi,
                p=guess,
                cutoffs=self.cutoff,
                site_tag_id="X{}",
                contraction_opt=self.contraction_opt,
                inplace=True,
            )
            if fit.p.L == 1:
                fit.run(n_iter=self.fit_n_iter)
            else:
                fit.run_eff(n_iter=self.fit_n_iter)
            result = fit.p

        result.normalize()
        return result

    def _update_conditioned_boundary(self, y, row_config, phi):
        """Project one row into the conditioned boundary and compress it."""
        if y == self.Ly - 1:
            return phi
        row_network = self._projected_row_network(y, row_config)
        if phi is None:
            return self._compress_conditioned_boundary(row_network)
        return self._compress_conditioned_boundary(
            self._apply_projected_row(row_network, phi)
        )

    def _boundary_sample_or_probability(self, rng=None, config=None):
        """Sample or evaluate one boundary proposal with cached row suffixes."""
        # A collapsed future MPS can make each dense column transfer much
        # larger than the original local center. For larger PEPS, retain the
        # reference contraction in that case: it is both faster and avoids
        # turning the existing marginal approximation into a second dense
        # transfer bottleneck.
        if self.marginal_chi not in (None, 0) and self.Lx * self.Ly > 9:
            result = self._boundary_sample_or_probability_reference(
                rng=rng,
                config=config,
            )
            self._last_row_cache_stats = {
                "rows": self.Ly,
                "suffix_cache_builds": 0,
                "site_prefix_updates": 0,
                "mode": "reference-center",
            }
            return result

        self._reset_rho_diagnostics()
        if config is not None:
            config = tuple(int(value) for value in config)
            if len(config) != len(self.site_order):
                raise ValueError(
                    f"Expected {len(self.site_order)} physical values, got "
                    f"{len(config)}."
                )

        sampled = []
        log10_probability = 0.0
        zero_probability = False
        phi = None
        config_pos = 0
        cache_builds = 0

        for y in range(self.Ly):
            row_cache = self._build_row_transfer_cache(y, phi)
            cache_builds += 1
            left = None
            row_config = []
            for x in range(self.Lx):
                site = (x, y)
                rho, _, _ = self._row_local_rho(row_cache, x, y, left)
                probabilities = self._conditional_probabilities(rho, site=site)
                if config is None:
                    value = int(rng.choice(len(probabilities), p=probabilities))
                else:
                    value = config[config_pos]
                    config_pos += 1
                    if value < 0 or value >= len(probabilities):
                        raise ValueError(
                            f"Physical value {value} at site {site!r} is "
                            f"outside range(0, {len(probabilities)})."
                        )
                selected_probability = float(probabilities[value])
                if selected_probability <= 0:
                    zero_probability = True
                elif not zero_probability:
                    log10_probability += math.log10(selected_probability)
                sampled.append(value)
                row_config.append(value)
                # Fix immediately, then carry only the contracted prefix into
                # the next x-site. The cached suffix is never mutated.
                left = self._advance_row_prefix(row_cache, x, value, left)
            phi = self._update_conditioned_boundary(y, row_config, phi)

        self._last_boundary_mps = None if phi is None else phi.copy()
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": cache_builds,
            "site_prefix_updates": len(self.site_order),
            "mode": "transfer",
        }
        if zero_probability:
            omega = (0.0, 0)
        else:
            exponent = math.floor(log10_probability)
            omega = (
                10.0 ** (log10_probability - exponent),
                exponent,
            )
        return sampled, omega

    def _boundary_sample_or_probability_reference(self, rng=None, config=None):
        """Reference proposal using a full center contraction at each site."""
        self._reset_rho_diagnostics()
        if config is not None:
            config = tuple(int(value) for value in config)
            if len(config) != len(self.site_order):
                raise ValueError(
                    f"Expected {len(self.site_order)} physical values, got "
                    f"{len(config)}."
                )

        sampled = []
        log10_probability = 0.0
        zero_probability = False
        phi = None
        config_pos = 0

        for y in range(self.Ly):
            center = self._boundary_center(y, phi)
            row_config = []
            for x in range(self.Lx):
                site = (x, y)
                rho, ket_ind = self._local_rho(center, site)
                probabilities = self._conditional_probabilities(rho, site=site)
                if config is None:
                    value = int(rng.choice(len(probabilities), p=probabilities))
                else:
                    value = config[config_pos]
                    config_pos += 1
                    if value < 0 or value >= len(probabilities):
                        raise ValueError(
                            f"Physical value {value} at site {site!r} is "
                            f"outside range(0, {len(probabilities)})."
                        )
                selected_probability = float(probabilities[value])
                if selected_probability <= 0:
                    zero_probability = True
                elif not zero_probability:
                    log10_probability += math.log10(selected_probability)
                sampled.append(value)
                row_config.append(value)
                # Fix immediately: the next x-site must condition on this
                # sampled prefix rather than on the unconditioned row.
                center.isel_({ket_ind: value})
            phi = self._update_conditioned_boundary(y, row_config, phi)

        self._last_boundary_mps = None if phi is None else phi.copy()
        if zero_probability:
            omega = (0.0, 0)
        else:
            exponent = math.floor(log10_probability)
            omega = (
                10.0 ** (log10_probability - exponent),
                exponent,
            )
        return sampled, omega

    def _projected_amplitude(self, config):
        """Contract the original ket after fixing a complete configuration."""
        # Keep this contraction independent of the proposal network: boundary
        # truncation changes q(S), but the importance estimator still needs
        # the amplitude of the unmodified PEPS for the sampled configuration.
        projected = self._ket.copy()
        projected.isel_(
            {
                self._site_inds[site]: int(value)
                for site, value in zip(self.site_order, config)
            }
        )
        return self._scalar(
            projected.contract(all, optimize=self.contraction_opt)
        )

    def _exact_proposal_probability(self, config):
        """Evaluate the exact sequential proposal probability of ``config``."""
        self._reset_rho_diagnostics()
        config = tuple(int(value) for value in config)
        if len(config) != len(self.site_order):
            raise ValueError(
                f"Expected {len(self.site_order)} physical values, got "
                f"{len(config)}."
            )

        working = self._norm.copy()
        probability = 1.0
        for site, value in zip(self.site_order, config):
            rho, ket_ind = self._local_rho(working, site)
            probabilities = self._conditional_probabilities(rho, site=site)
            if value < 0 or value >= len(probabilities):
                raise ValueError(
                    f"Physical value {value} at site {site!r} is outside "
                    f"range(0, {len(probabilities)})."
                )
            probability *= float(probabilities[value])
            working.isel_({ket_ind: value})
        return float(probability)

    def _sample_one_exact(self, rng):
        """Draw one configuration from the exact serial proposal."""
        self._reset_rho_diagnostics()
        working = self._norm.copy()
        config = []
        log10_proposal = 0.0

        for site in self.site_order:
            rho, ket_ind = self._local_rho(working, site)
            probabilities = self._conditional_probabilities(rho, site=site)
            value = int(rng.choice(len(probabilities), p=probabilities))
            config.append(value)
            log10_proposal += math.log10(float(probabilities[value]))
            working.isel_({ket_ind: value})

        proposal_mantissa = 10.0 ** (log10_proposal - math.floor(log10_proposal))
        proposal_exponent = math.floor(log10_proposal)
        amplitude = self._projected_amplitude(config)
        return config, (proposal_mantissa, proposal_exponent), self._scaled(amplitude)

    def _sample_one(self, rng):
        """Draw one configuration from the selected proposal backend."""
        if self.boundary_engine == "exact":
            return self._sample_one_exact(rng)

        config, omega = self._boundary_sample_or_probability(rng=rng)
        amplitude = self._projected_amplitude(config)
        return config, omega, self._scaled(amplitude)

    @staticmethod
    def _log10_to_scaled(log10_probability):
        """Convert a base-10 log probability to mantissa/exponent form."""
        exponent = math.floor(log10_probability)
        return 10.0 ** (log10_probability - exponent), exponent

    def _sample_batch_exact(self, rng, samples):
        """Sample exact proposals while sharing identical prefixes."""
        groups = [
            {
                "indices": np.arange(samples, dtype=int),
                "config": [],
                "working": self._norm.copy(),
                "log10": 0.0,
            }
        ]
        max_groups = 1

        for site in self.site_order:
            next_groups = []
            for group in groups:
                rho, ket_ind = self._local_rho(group["working"], site)
                probabilities = self._conditional_probabilities(rho, site=site)
                values = np.asarray(
                    rng.choice(
                        len(probabilities),
                        size=len(group["indices"]),
                        p=probabilities,
                    ),
                    dtype=int,
                )
                for value in np.unique(values):
                    shot_indices = group["indices"][values == value]
                    working = group["working"].copy()
                    working.isel_({ket_ind: int(value)})
                    next_groups.append(
                        {
                            "indices": shot_indices,
                            "config": group["config"] + [int(value)],
                            "working": working,
                            "log10": group["log10"]
                            + math.log10(float(probabilities[value])),
                        }
                    )
            groups = next_groups
            max_groups = max(max_groups, len(groups))

        return groups, max_groups

    def _sample_batch_boundary(self, rng, samples):
        """Sample boundary proposals with one row cache per prefix group."""
        # Once almost every shot has a different post-row prefix, constructing
        # a dense transfer cache per group costs more than the original local
        # center contractions. Keep large production batches on that stable
        # path; small batches still exercise and benefit from row-cache reuse.
        group_limit = 32 if self.Lx * self.Ly <= 9 else 4
        if samples > group_limit:
            return self._sample_batch_boundary_reference(rng, samples)

        groups = [
            {
                "indices": np.arange(samples, dtype=int),
                "config": [],
                "log10": 0.0,
                "phi": None,
                "row_cache": None,
                "left": None,
                "row_config": [],
            }
        ]
        max_groups = 1
        cache_builds = 0
        prefix_updates = 0

        for y in range(self.Ly):
            for group in groups:
                # A group represents one shared prefix, hence it has one
                # conditioned lower boundary and one reusable row suffix.
                group["row_cache"] = self._build_row_transfer_cache(
                    y,
                    group["phi"],
                )
                cache_builds += 1
                group["left"] = None
                group["row_config"] = []

            for x in range(self.Lx):
                site = (x, y)
                next_groups = []
                for group in groups:
                    rho, _, _ = self._row_local_rho(
                        group["row_cache"],
                        x,
                        y,
                        group["left"],
                    )
                    probabilities = self._conditional_probabilities(
                        rho,
                        site=site,
                    )
                    values = np.asarray(
                        rng.choice(
                            len(probabilities),
                            size=len(group["indices"]),
                            p=probabilities,
                        ),
                        dtype=int,
                    )
                    for value in np.unique(values):
                        shot_indices = group["indices"][values == value]
                        left = self._advance_row_prefix(
                            group["row_cache"],
                            x,
                            int(value),
                            group["left"],
                        )
                        prefix_updates += 1
                        next_groups.append(
                            {
                                "indices": shot_indices,
                                "config": group["config"] + [int(value)],
                                "log10": group["log10"]
                                + math.log10(float(probabilities[value])),
                                "phi": group["phi"],
                                "row_cache": group["row_cache"],
                                "left": left,
                                "row_config": group["row_config"] + [int(value)],
                            }
                        )
                groups = next_groups
                max_groups = max(max_groups, len(groups))

            for group in groups:
                group["phi"] = self._update_conditioned_boundary(
                    y,
                    group["row_config"],
                    group["phi"],
                )
                group["row_cache"] = None
                group["left"] = None

        if groups:
            last_phi = groups[-1]["phi"]
            self._last_boundary_mps = (
                None if last_phi is None else last_phi.copy()
            )
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": cache_builds,
            "site_prefix_updates": prefix_updates,
            "mode": "transfer",
        }
        return groups, max_groups

    def _sample_batch_boundary_reference(self, rng, samples):
        """Reference prefix batch path used when groups are highly fragmented."""
        groups = [
            {
                "indices": np.arange(samples, dtype=int),
                "config": [],
                "log10": 0.0,
                "phi": None,
                "center": None,
                "row_config": [],
            }
        ]
        max_groups = 1

        for y in range(self.Ly):
            for group in groups:
                group["center"] = self._boundary_center(y, group["phi"])
                group["row_config"] = []

            for x in range(self.Lx):
                site = (x, y)
                next_groups = []
                for group in groups:
                    rho, ket_ind = self._local_rho(group["center"], site)
                    probabilities = self._conditional_probabilities(
                        rho,
                        site=site,
                    )
                    values = np.asarray(
                        rng.choice(
                            len(probabilities),
                            size=len(group["indices"]),
                            p=probabilities,
                        ),
                        dtype=int,
                    )
                    for value in np.unique(values):
                        shot_indices = group["indices"][values == value]
                        center = group["center"].copy()
                        center.isel_({ket_ind: int(value)})
                        next_groups.append(
                            {
                                "indices": shot_indices,
                                "config": group["config"] + [int(value)],
                                "log10": group["log10"]
                                + math.log10(float(probabilities[value])),
                                "phi": group["phi"],
                                "center": center,
                                "row_config": group["row_config"] + [
                                    int(value)
                                ],
                            }
                        )
                groups = next_groups
                max_groups = max(max_groups, len(groups))

            for group in groups:
                group["phi"] = self._update_conditioned_boundary(
                    y,
                    group["row_config"],
                    group["phi"],
                )
                group["center"] = None

        if groups:
            last_phi = groups[-1]["phi"]
            self._last_boundary_mps = (
                None if last_phi is None else last_phi.copy()
            )
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": 0,
            "site_prefix_updates": 0,
            "mode": "reference-prefix",
        }
        return groups, max_groups

    def sample_batch(
        self,
        samples: int = 1,
        seed: int | None = None,
    ) -> PEPSSampleResult:
        """Draw a prefix-grouped batch of independent PEPS samples.

        Groups share a local conditional network until their sampled prefixes
        differ. This is the safe PEPS analogue of a batch axis: Quimb does not
        receive per-shot isel values on one shared network.

        A group therefore represents identical history, not merely equal
        tensor shapes. Once two shots choose different physical values, their
        conditioned boundary states and subsequent conditional networks are
        different and must split.
        """
        if not isinstance(samples, (int, np.integer)) or int(samples) < 1:
            raise ValueError("samples must be a positive integer.")
        samples = int(samples)
        self._reset_rho_diagnostics()
        rng = np.random.default_rng(seed)

        if self.boundary_engine == "exact":
            groups, max_groups = self._sample_batch_exact(rng, samples)
        else:
            groups, max_groups = self._sample_batch_boundary(rng, samples)

        configs = [None] * samples
        omegas = [None] * samples
        amplitudes = [None] * samples
        for group in groups:
            omega = self._log10_to_scaled(group["log10"])
            for index in group["indices"]:
                index = int(index)
                configs[index] = group["config"]
                omegas[index] = omega
                amplitudes[index] = self._scaled(
                    self._projected_amplitude(group["config"])
                )

        self._last_batch_stats = {
            "samples": samples,
            "final_prefix_groups": len(groups),
            "max_prefix_groups": max_groups,
            "boundary_engine": self.boundary_engine,
        }
        if self.boundary_engine != "exact":
            self._last_batch_stats.update(
                {
                    "suffix_cache_builds": self._last_row_cache_stats.get(
                        "suffix_cache_builds", 0
                    ),
                    "site_prefix_updates": self._last_row_cache_stats.get(
                        "site_prefix_updates", 0
                    ),
                }
            )
        return PEPSSampleResult(
            configs=configs,
            omegas=(
                [value[0] for value in omegas],
                [value[1] for value in omegas],
            ),
            ps=(
                [value[0] for value in amplitudes],
                [value[1] for value in amplitudes],
            ),
        )

    @staticmethod
    def _scaled_to_float(value):
        """Convert a real mantissa/exponent pair to a float when possible."""
        mantissa, exponent = value
        if mantissa == 0:
            return 0.0
        return float(mantissa * (10.0 ** exponent))

    def probability(self, config):
        """Return the sequential proposal probability of ``config``.

        In exact mode, the proposal is the exact Born probability of the PEPS
        configuration. In boundary mode, it is the normalized approximate
        proposal generated by the selected boundary cutoffs. It is deliberately
        the proposal ``q(S)``, not a second amplitude contraction: importance
        sampling needs this quantity separately from ``|Psi(S)|**2``.
        """
        if self.boundary_engine == "exact":
            return self._exact_proposal_probability(config)
        _, omega = self._boundary_sample_or_probability(config=config)
        return self._scaled_to_float(omega)

    def sample(self, samples: int = 1, seed: int | None = None) -> PEPSSampleResult:
        """Draw independent configurations from the selected proposal."""
        if not isinstance(samples, (int, np.integer)) or int(samples) < 1:
            raise ValueError("samples must be a positive integer.")
        rng = np.random.default_rng(seed)
        configs = []
        omegas_mantissa = []
        omegas_exponent = []
        ps_mantissa = []
        ps_exponent = []

        for _ in range(int(samples)):
            config, omega, amplitude = self._sample_one(rng)
            configs.append(config)
            omega_mantissa, omega_exponent = omega
            amplitude_mantissa, amplitude_exponent = amplitude
            omegas_mantissa.append(omega_mantissa)
            omegas_exponent.append(omega_exponent)
            ps_mantissa.append(amplitude_mantissa)
            ps_exponent.append(amplitude_exponent)

        return PEPSSampleResult(
            configs=configs,
            omegas=(omegas_mantissa, omegas_exponent),
            ps=(ps_mantissa, ps_exponent),
        )
