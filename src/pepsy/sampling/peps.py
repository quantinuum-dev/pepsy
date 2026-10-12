"""Direct PEPS sampling with conditioned boundary states."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
import math

import autoray as ar
import numpy as np

from .._internal.cutoff import dtype_auto_cutoff
from .results import PEPSSampleResult

__all__ = ["PepsSampler"]


class PepsSampler:
    """Directly sample a finite PEPS with exact or boundary-MPS proposals.

    Without bond caps, ``boundary_engine="auto"`` selects exact contraction.
    Supplying ``chi_prime`` selects boundary sampling with a conditioned
    single-layer ket boundary compressed after every projected row. ``chi``
    independently controls the cached future double-layer environment.

    Parameters
    ----------
    peps : TensorNetwork2D
        Finite dense PEPS with ``site_tag`` and ``site_ind`` methods.
        Boundary sampling requires open edges; exact mode can contract cycles.
    chi : int, optional
        Maximum bond dimension of the future double-layer environment (χ).
        ``None`` or ``0`` uses identity future caps in boundary mode.
    chi_prime : int, optional
        Maximum bond dimension of the conditioned single-layer ket boundary
        (χ′). Required in boundary mode when ket compression is enabled.
    to_backend : callable, optional
        Convert each tensor array on a private copy of the PEPS. If omitted,
        infer the array backend, dtype, and device from the input PEPS.
        Local density matrices, probabilities, and draws use that backend.
    boundary_engine : {"auto", "exact", "quimb-mps", "dmrg"}, default="auto"
        Future-environment algorithm. ``"auto"`` (also ``None``) selects
        ``"dmrg"`` when a positive cap is supplied, otherwise ``"exact"``.
        ``"dmrg"`` uses Pepsy's :class:`BdyMPS` and :class:`CompBdy`;
        ``"quimb-mps"`` uses Quimb's cached MPS environments.
        Explicit ``"exact"`` rejects positive caps.
    ket_compression : {"quimb", "fit", None}, default="quimb"
        Compression backend for the conditioned ket boundary. ``None`` leaves
        it uncompressed and allows ``chi_prime=None`` in boundary mode.
        This option is ignored in exact mode.
    cutoff : float or {"auto"}, default="auto"
        Singular-value cutoff for boundary compression. ``"auto"`` uses
        Pepsy's shared dtype policy: 1e-6 for complex64/float32 and 1e-12 for
        complex128/float64, resolved after ``to_backend`` and on refresh.
    cutoff_mode : str or None, default="auto"
        ``"auto"`` and ``None`` use relative discarded squared weight
        (``"rsum2"``), as in ordinary MpsOptimizer compression. Explicit Quimb
        modes (``"rel"``, ``"abs"``, ``"sum1"``, ``"sum2"``, ``"rsum1"``,
        ``"rsum2"``) override it. Fixed-rank one-site FIT does not truncate
        singular values; the cutoff controls its compressed ket guess.
        Relative modes rescale private Quimb future and conditioned ket
        boundaries before compression to protect squared singular values.
        Absolute modes preserve the original compression scale.
    rho_positivity : {None, "clip", "absolute"}, default=None
        Sampling uses the Hermitian part of each local rho. ``None`` keeps
        its diagonal, clipping only negative roundoff. ``"clip"`` replaces
        negative Hermitian eigenvalues with zero; ``"absolute"`` takes their
        absolute values, equivalent to sqrt(H.H @ H) for the Hermitian part H.
        Both repairs change the proposal and are included in returned log
        probabilities and importance weights. Non-finite rhos still raise.
    fit_n_iter : int, default=2
        Number of local FIT sweeps used by the ``"fit"`` ket compressor and
        by the ``"dmrg"`` future-environment preparation.
    contraction_opt : object, optional
        Full-network optimizer. None constructs one reusable Cotengra optimizer
        with ``pepsy.tensors.build_optimizer(parallel=False)`` (formerly named
        ``build_contraction``). Explicit optimizer objects/strings are retained.
    row_contraction_opt : object, optional
        Optimizer for cached-row prefix/suffix/local contractions, default
        ``"auto-hq"`` independently of the full-network optimizer. Explicit
        None inherits ``contraction_opt``.
    amplitude_max_intermediate_bytes : int, optional
        Reject an exact amplitude plan before execution if its largest tensor
        exceeds this byte count. None disables the check. Not a peak-memory cap.
    amplitude_max_cost : float, optional
        Optional maximum Cotengra estimated exact contraction cost. Exceeding
        it raises before execution; no approximate amplitude is substituted.
    amplitude_mode : {"proposal", "boundary", "exact", "none"}, default="proposal"
        Evaluate amplitudes with a full exact contraction or a truncated
        single-layer boundary-MPS sweep. Boundary amplitudes and the resulting
        importance weights are approximate; proposal probabilities remain those
        of the actual sampling procedure. "proposal" skips amplitude evaluation
        and returns proposal probabilities with equal averaging weights.
        "none" is a compatibility alias; result metadata retains the supplied
        spelling. The default is "proposal"; amplitude corrections are opt-in.
    amplitude_chi : int, optional
        Boundary amplitude bond cap. Defaults to chi_prime in boundary mode.
        If both are None, the boundary sweep has no bond cap; cutoff still applies.
    row_cache_max_bytes : int, default=67108864
        Budget for estimated row-environment storage and workspace across live
        prefix groups, including the initial row retained until ``refresh()``.
        The default is 64 MiB. Zero explicitly selects the reference sweep;
        oversized caches also fall back to it. This is not a total process-
        memory cap. Use chunked sampling to bound simultaneous prefix groups.
    row_cache_mode : {"dense", "factored"}, default="factored"
        Boundary sampling defaults to FIT-style numerical environment reuse:
        build right suffixes once per conditioned row and update the left
        prefix after each draw. Local factors stay separate, avoiding dense
        two-interface transfers. ``"dense"`` retains the legacy alternative.
        Neither cache adds truncation or changes the selected proposal.
    sample_chi, marginal_chi : int, optional
        Compatible aliases for ``chi_prime`` and ``chi``, respectively.
        If both spellings are non-None, their values must agree. ``None``
        leaves the value to the other spelling.

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
        chi=None,
        chi_prime=None,
        to_backend=None,
        boundary_engine="auto",
        ket_compression="quimb",
        cutoff="auto",
        cutoff_mode="auto",
        rho_positivity=None,
        fit_n_iter=2,
        contraction_opt=None,
        row_contraction_opt="auto-hq",
        amplitude_max_intermediate_bytes=None,
        amplitude_max_cost=None,
        amplitude_mode="proposal",
        amplitude_chi=None,
        row_cache_max_bytes=64 * 2**20,
        row_cache_mode="factored",
        sample_chi=None,
        marginal_chi=None,
    ):
        self.peps = getattr(peps, "tn", peps)
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be a callable or None.")
        self._to_backend_override = to_backend
        self.sample_chi = self._resolve_chi_alias(
            self._validate_optional_chi(chi_prime, "chi_prime"),
            self._validate_optional_chi(sample_chi, "sample_chi"),
            "chi_prime", "sample_chi",
        )
        self.marginal_chi = self._resolve_chi_alias(
            self._validate_marginal_chi(chi, "chi"),
            self._validate_marginal_chi(marginal_chi, "marginal_chi"),
            "chi", "marginal_chi",
        )
        self.boundary_engine = self._normalize_boundary_engine(boundary_engine)
        if self.boundary_engine == "auto":
            has_cap = self.sample_chi is not None or self.marginal_chi not in (None, 0)
            self.boundary_engine = "dmrg" if has_cap else "exact"
        self.ket_compression = self._normalize_ket_compression(ket_compression)
        if amplitude_mode not in {"exact", "boundary", "proposal", "none"}:
            raise ValueError("amplitude_mode must be 'boundary', 'proposal', or 'exact' "
                             "('none' is an alias for 'proposal').")
        self.amplitude_mode = amplitude_mode
        self.amplitude_chi = self._validate_optional_chi(amplitude_chi, "amplitude_chi")
        if amplitude_mode == "boundary":
            if self.amplitude_chi is None:
                self.amplitude_chi = self.sample_chi
        from ..optimizers.sweep.environments import (  # noqa: PLC0415
            _canonical_cutoff_mode,
        )

        if isinstance(cutoff, str):
            if cutoff.strip().lower() != "auto":
                raise ValueError("cutoff must be 'auto' or a finite non-negative real number.")
            self._cutoff_requested = "auto"
        else:
            if isinstance(cutoff, (bool, np.bool_)) or not isinstance(
                cutoff, (int, float, np.integer, np.floating)
            ):
                raise TypeError("cutoff must be 'auto' or a finite non-negative real number.")
            if not math.isfinite(float(cutoff)) or float(cutoff) < 0:
                raise ValueError("cutoff must be 'auto' or a finite non-negative real number.")
            self._cutoff_requested = float(cutoff)
        self.cutoff_mode = _canonical_cutoff_mode(
            "auto" if cutoff_mode is None else cutoff_mode
        )
        if rho_positivity not in (None, "clip", "absolute"):
            raise ValueError("rho_positivity must be None, 'clip', or 'absolute'.")
        self.rho_positivity = rho_positivity
        if (
            isinstance(fit_n_iter, (bool, np.bool_))
            or not isinstance(fit_n_iter, (int, np.integer))
            or int(fit_n_iter) < 1
        ):
            raise ValueError("fit_n_iter must be a positive integer.")
        self.fit_n_iter = int(fit_n_iter)
        if contraction_opt is None:
            from ..tensors import build_optimizer  # noqa: PLC0415

            # Own one structural plan cache per sampler. Parallel path search
            # is opt-in through a caller-supplied optimizer, not hidden workers.
            contraction_opt = build_optimizer(parallel=False)
        self.contraction_opt = contraction_opt
        self.row_contraction_opt = (
            contraction_opt if row_contraction_opt is None else row_contraction_opt
        )
        if amplitude_max_intermediate_bytes is not None:
            amplitude_max_intermediate_bytes = self._positive_sample_count(
                amplitude_max_intermediate_bytes, "amplitude_max_intermediate_bytes"
            )
        if amplitude_max_cost is not None:
            if (isinstance(amplitude_max_cost, (bool, np.bool_))
                    or not isinstance(amplitude_max_cost, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(amplitude_max_cost))
                    or amplitude_max_cost <= 0):
                raise ValueError("amplitude_max_cost must be positive and finite, or None.")
            amplitude_max_cost = float(amplitude_max_cost)
        self.amplitude_max_intermediate_bytes = amplitude_max_intermediate_bytes
        self.amplitude_max_cost = amplitude_max_cost
        if (
            isinstance(row_cache_max_bytes, bool)
            or not isinstance(row_cache_max_bytes, (int, np.integer))
            or row_cache_max_bytes < 0
        ):
            raise ValueError("row_cache_max_bytes must be a non-negative integer.")
        self.row_cache_max_bytes = int(row_cache_max_bytes)
        if row_cache_mode not in {"dense", "factored"}:
            raise ValueError("row_cache_mode must be 'dense' or 'factored'.")
        self.row_cache_mode = row_cache_mode
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
                    "boundary_engine='exact' does not use sample_chi (chi_prime); choose "
                    "'quimb-mps' or 'dmrg' for a conditioned boundary MPS."
                )
            if self.marginal_chi not in (None, 0):
                raise ValueError(
                    "boundary_engine='exact' does not use marginal_chi (chi); choose "
                    "'quimb-mps' or 'dmrg' for a future environment."
                )
        elif self.sample_chi is None and self.ket_compression is not None:
            raise ValueError(
                "chi_prime or sample_chi is required for boundary sampling with "
                "ket compression; set ket_compression=None to leave it uncompressed."
            )
        self.refresh()

    @property
    def chi(self):
        """Resolved future double-layer bond cap (legacy ``marginal_chi``)."""
        return self.marginal_chi

    @property
    def chi_prime(self):
        """Resolved conditioned ket bond cap (legacy ``sample_chi``)."""
        return self.sample_chi

    @staticmethod
    def _resolve_chi_alias(value, alias, name, alias_name):
        if value is not None and alias is not None and value != alias:
            raise ValueError(f"{name} and {alias_name} must agree when both are supplied.")
        return alias if value is None else value

    @staticmethod
    def _validate_optional_chi(value, name):
        if value is None:
            return None
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None.")
        return int(value)

    @staticmethod
    def _validate_marginal_chi(value, name="marginal_chi"):
        if value is None:
            return None
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None.")
        return int(value)

    @staticmethod
    def _normalize_boundary_engine(engine):
        key = "auto" if engine is None else str(engine).strip().lower()
        aliases = {
            "exact": "exact",
            "quimb": "quimb-mps",
            "mps": "quimb-mps",
            "quimb-mps": "quimb-mps",
            "dmrg": "dmrg",
            "fit": "dmrg",
            "auto": "auto",
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise ValueError(
                "boundary_engine must be 'auto', 'exact', 'quimb-mps', or 'dmrg'."
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
        from ..backends import (  # noqa: PLC0415
            backend_signatures_compatible,
            infer_backend_converter_from_sample,
            infer_backend_signature,
            resolve_backend_sample_data_from_tn,
        )
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
        if (self.boundary_engine != "exact" or self.amplitude_mode == "boundary") and (
            any(self.peps.is_cyclic_x(y) for y in range(self.Ly))
            or any(self.peps.is_cyclic_y(x) for x in range(self.Lx))
        ):
            raise ValueError("Boundary PEPS sampling requires open boundaries.")

        # A user converter may mutate its argument: isolate the source arrays
        # before calling it. Without conversion Quimb can share immutable data.
        ket = self.peps.copy()
        if self._to_backend_override is not None:
            # Multiplication copies dense arrays without detaching Torch
            # gradients (unlike its generic copy/deepcopy implementations).
            ket.apply_to_arrays(lambda data: self._to_backend_override(data * 1))
        template = resolve_backend_sample_data_from_tn(ket)
        signature = infer_backend_signature(template)
        if signature[0] == "symmray":
            raise TypeError("PepsSampler currently requires dense PEPS arrays.")
        if any(
            not backend_signatures_compatible(
                infer_backend_signature(tensor.data), signature
            )
            for tensor in ket.tensors
        ):
            raise ValueError(
                "PEPS arrays must have compatible backends, dtypes, and devices; "
                "supply to_backend to convert them consistently."
            )
        self.to_backend = (
            self._to_backend_override
            if self._to_backend_override is not None
            else infer_backend_converter_from_sample(template)
        )
        self.backend = signature[0]
        if np.dtype(ar.get_dtype_name(template)).kind not in "fc":
            raise TypeError(
                "PepsSampler requires floating or complex arrays; supply "
                "to_backend to convert integer arrays explicitly."
            )
        self.cutoff = (
            dtype_auto_cutoff(ar.get_dtype_name(template))
            if self._cutoff_requested == "auto"
            else self._cutoff_requested
        )
        self._xp = ar.get_namespace(template)
        self._real_xp = ar.get_namespace(self._xp.real(template))
        self._ket, self._norm = build_bra_ket(ket=ket)
        self._amplitude_tree = None
        self._amplitude_plan_info = None
        self._amplitude_specs = None
        self._amplitude_leaves = None
        self._amplitude_stats = {"plan_builds": 0, "contractions": 0}
        self._row_bond_cache = {}
        self._identity_future_cache = {}
        self._initial_row_cache = None
        self._initial_row_cache_estimate_bytes = 0
        self._row_cache_estimate_per_group = None
        self._last_cache_decision = {}
        self._grouped_draw_count = 0
        self._array_itemsize = np.dtype(ar.get_dtype_name(template)).itemsize
        with self._array_device_context():
            self._zero_log_probability = self._real_xp.zeros(())
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
            missing = set(range(self.Ly - 1)) - self._future_environments.keys()
            if missing:
                raise RuntimeError(f"Missing future boundaries for rows {sorted(missing)}.")
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
                cutoff_mode=self.cutoff_mode,
                canonize=True,
                mode="mps",
                layer_tags=("KET", "BRA"),
                equalize_norms=self.cutoff_mode in {"rel", "rsum1", "rsum2"},
            )
            norm = self._norm.copy()
            self._rescale_proposal_boundary_(norm)
            # Quimb owns this boundary sweep and returns the native ``ymax``
            # MPS objects. Keep the cache separate from the shot-conditioned
            # ket boundary because this network still sums over future rows.
            store.start_sweep(norm, "y", update_side="left")
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
            lazy=True,
        )
        compressor = CompBdy(
            self._norm,
            boundary.mps_b,
            contraction_opt=self.contraction_opt,
            fit_contraction_opt=self.contraction_opt,
            fit_mode="eff",
            fit_cutoff=self._cutoff_requested,
            fit_cutoff_mode=self.cutoff_mode,
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
        if hasattr(value, "item"):
            return value.item()
        data = getattr(value, "data", value)
        if hasattr(data, "item"):
            return data.item()
        return np.asarray(ar.to_numpy(data)).reshape(()).item()

    @property
    def rho_diagnostics(self):
        """Return diagnostics for the most recently evaluated local rhos."""
        result = {}
        for site, values in self._last_rho_diagnostics.items():
            keys = [key for key in values if key != "evaluation_count"]
            scalars = ar.to_numpy(
                self._real_xp.stack([values[key] for key in keys])
            )
            result[site] = dict(zip(keys, map(float, scalars)))
            result[site]["evaluation_count"] = values["evaluation_count"]
        return result

    @property
    def batch_stats(self):
        """Return statistics from the most recent prefix-grouped batch."""
        return dict(self._last_batch_stats)

    @property
    def row_cache_stats(self):
        """Return statistics from the most recent boundary row-cache run."""
        return {**self._last_row_cache_stats, **self._last_cache_decision}

    def _reset_rho_diagnostics(self):
        """Start a fresh local-rho diagnostic trace."""
        self._last_rho_diagnostics = {}

    @property
    def amplitude_stats(self):
        """Exact amplitude plan builds and calls since construction/refresh."""
        return dict(self._amplitude_stats)

    @property
    def amplitude_plan_info(self):
        """Last exact plan estimate, including one rejected by a limit.

        Largest intermediate bytes and Cotengra contraction cost are estimates,
        not a bound on process/device memory or a wall-time prediction.
        Cleared by refresh; inspecting this property does not plan a contraction.
        """
        return None if self._amplitude_plan_info is None else dict(self._amplitude_plan_info)

    def _check_amplitude_plan(self, tree):
        info = {
            "largest_intermediate_bytes": int(tree.max_size()) * self._array_itemsize,
            "contraction_cost": float(tree.contraction_cost()),
        }
        self._amplitude_plan_info = info
        if (self.amplitude_max_intermediate_bytes is not None
                and info["largest_intermediate_bytes"] > self.amplitude_max_intermediate_bytes):
            raise MemoryError(
                f"Exact amplitude plan needs a largest intermediate of "
                f"{info['largest_intermediate_bytes']} bytes, exceeding "
                f"amplitude_max_intermediate_bytes={self.amplitude_max_intermediate_bytes}. "
                "Choose a better or exactly sliced contraction plan, or raise the explicit limit."
            )
        if (self.amplitude_max_cost is not None
                and info["contraction_cost"] > self.amplitude_max_cost):
            raise RuntimeError(
                f"Exact amplitude plan cost {info['contraction_cost']} exceeds "
                f"amplitude_max_cost={self.amplitude_max_cost}."
            )

    @property
    def diagnostics(self):
        """Host summary of rho, batch, and cumulative amplitude diagnostics.

        Rho values describe the latest query, serial draw, or complete batch.
        Batch counts describe the latest batch; amplitude counts last until
        refresh. Chunked collected batches aggregate rho counts and maxima.
        Access synchronizes backend scalars. Neither small corrections nor
        a high effective sample size establish that proposal support is intact.
        """
        values = tuple(self.rho_diagnostics.values())
        return {
            "boundary_engine": self.boundary_engine,
            "rho_positivity": self.rho_positivity,
            "amplitude_mode": self.amplitude_mode,
            "amplitude_chi": self.amplitude_chi,
            "conditional_evaluations": sum(d["evaluation_count"] for d in values),
            "max_hermiticity_defect": max(
                (d["max_hermiticity_defect"] for d in values), default=0.0
            ),
            "max_relative_positivity_correction": max(
                (d["max_relative_positivity_correction"] for d in values), default=0.0
            ),
            "max_negative_diagonal_mass": max(
                (d["max_negative_diagonal_mass"] for d in values), default=0.0
            ),
            "batch": self.batch_stats,
            "amplitudes": self.amplitude_stats,
            "amplitude_plan": self.amplitude_plan_info,
        }

    def _local_rho(self, working, site, *, strip_exponent=False):
        """Contract a local rho, copying only tensors whose bra index changes."""
        import quimb.tensor as qtn  # noqa: PLC0415

        site_tag = self._site_tags[site]
        ket_ind = self._site_inds[site]
        bra_ind = f"{ket_ind}__pepsy_bra"
        # tensor_contract reads its inputs. Relabel the local bra tensor(s)
        # without copying every tensor and rebuilding network index maps.
        tensors = (
            tensor.reindex({ket_ind: bra_ind})
            if site_tag in tensor.tags and "BRA" in tensor.tags
            else tensor
            for tensor in working.tensors
        )
        rho = qtn.tensor_contract(
            *tensors,
            output_inds=(ket_ind, bra_ind),
            optimize=self.contraction_opt,
            exponent=working.exponent,
            strip_exponent=strip_exponent,
        )
        if strip_exponent:
            # A common positive scale cancels from every conditional. This
            # path keeps rare explicit likelihood queries representable.
            rho = rho[0]
        rho = rho.data
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(
                f"Local density matrix at site {site!r} has invalid shape "
                f"{rho.shape!r}."
            )
        return rho, ket_ind

    def _conditional_probabilities(self, rho, *, site):
        """Normalize one local rho using the same kernel as grouped draws."""
        return self._conditional_probabilities_batch(rho[None, ...], site=site)[0]

    def _proposal_rho_diagonal(self, scaled, finite):
        """Return the Hermitian/positive proposal diagonal and repair size."""
        xp = self._xp
        real_xp = self._real_xp
        # Scale first and halve before adding, including at large finite rho.
        hermitian = 0.5 * scaled + 0.5 * xp.transpose(xp.conj(scaled), (0, 2, 1))
        if self.rho_positivity is None:
            diagonal = xp.real(xp.diagonal(hermitian, axis1=-2, axis2=-1))
            return diagonal, real_xp.zeros_like(diagonal[:, 0])

        # Do not send NaN/Inf to an eigensolver. Invalid groups are masked here
        # and rejected by the common validation below, with no extra host sync.
        hermitian = xp.where(finite[:, None, None], hermitian, xp.zeros_like(hermitian))
        diagonal_h = xp.real(xp.diagonal(hermitian, axis1=-2, axis2=-1))
        if hermitian.shape[-1] == 2:
            # Qubit spectral projectors directly: avoid a tiny eigensolver,
            # including its CUDA synchronization. All operations are batched.
            mean = 0.5 * (diagonal_h[:, 0] + diagonal_h[:, 1])
            delta = 0.5 * (diagonal_h[:, 0] - diagonal_h[:, 1])
            offdiag2 = xp.abs(hermitian[:, 0, 1])**2
            radius2 = delta**2 + offdiag2
            safe_radius = real_xp.sqrt(real_xp.where(
                radius2 > 0, radius2, real_xp.ones_like(radius2)
            ))
            radius = real_xp.where(radius2 > 0, safe_radius, real_xp.zeros_like(radius2))
            # Obtain the small root from the determinant, avoiding mean-radius
            # cancellation for e.g. diag(-1e-20, 1) in complex64.
            large = mean + real_xp.where(mean >= 0, radius, -radius)
            determinant = diagonal_h[:, 0] * diagonal_h[:, 1] - offdiag2
            small = determinant / real_xp.where(large != 0, large, real_xp.ones_like(large))
            eigenvalues = real_xp.stack((real_xp.where(mean >= 0, small, large),
                                        real_xp.where(mean >= 0, large, small)), axis=-1)
            # The minor projector weight also needs a cancellation-free form.
            magnitude_delta = real_xp.abs(delta)
            major = 0.5 * (1.0 + real_xp.clip(magnitude_delta / safe_radius, 0.0, 1.0))
            minor = 0.5 * (offdiag2 / safe_radius) / (safe_radius + magnitude_delta)
            minus = real_xp.where(delta >= 0, minor, major)
            plus = real_xp.where(delta >= 0, major, minor)
            weights = real_xp.stack((real_xp.stack((minus, plus), axis=-1),
                                    real_xp.stack((plus, minus), axis=-1)), axis=-2)
        else:
            eigenvalues, vectors = xp.linalg.eigh(hermitian)
            weights = xp.abs(vectors)**2
        if self.rho_positivity == "clip":
            positive = real_xp.maximum(eigenvalues, real_xp.zeros_like(eigenvalues))
        else:
            positive = real_xp.abs(eigenvalues)
        # diag(V f(Lambda) V.H) only: no full reconstruction or rho.H @ rho.
        diagonal = real_xp.sum(weights * positive[:, None, :], axis=-1)
        if hermitian.shape[-1] == 2:
            # Keep the exact linear map (and its gradient) at repeated positive
            # or negative eigenvalues, where individual projectors are undefined.
            diagonal = real_xp.where((eigenvalues[:, 0] >= 0)[:, None], diagonal_h, diagonal)
            if self.rho_positivity == "absolute":
                diagonal = real_xp.where((eigenvalues[:, 1] <= 0)[:, None], -diagonal_h, diagonal)
        norm = real_xp.linalg.norm(eigenvalues, axis=-1)
        correction = real_xp.linalg.norm(positive - eigenvalues, axis=-1)
        correction = correction / real_xp.where(norm > 0, norm, real_xp.ones_like(norm))
        return diagonal, correction

    def _conditional_probabilities_batch(self, rhos, *, site):
        """Validate all active prefix rhos with one scalar synchronization."""
        xp = self._xp
        real_xp = self._real_xp
        diagonal = xp.real(xp.diagonal(rhos, axis1=-2, axis2=-1))
        trace = real_xp.sum(diagonal, axis=-1)
        scale = real_xp.maximum(
            real_xp.abs(trace), real_xp.max(real_xp.abs(diagonal), axis=-1)
        )
        tolerance = 256.0 * np.finfo(ar.get_dtype_name(diagonal)).eps * scale

        # Scale before squaring or subtracting. This evaluates the existing
        # ||rho-rho.H|| / max(||rho||, 1) convention without float32 overflow,
        # including rho=0 and norms smaller than one.
        magnitude = real_xp.max(xp.abs(rhos), axis=(-2, -1))
        safe_magnitude = real_xp.where(
            magnitude > 0, magnitude, real_xp.ones_like(magnitude)
        )
        scaled = rhos / safe_magnitude[:, None, None]
        finite = xp.all(xp.isfinite(rhos), axis=(-2, -1))
        proposal_diagonal, correction = self._proposal_rho_diagonal(scaled, finite)
        norm = xp.linalg.norm(scaled, axis=(-2, -1))
        defect = xp.linalg.norm(
            scaled - xp.transpose(xp.conj(scaled), (0, 2, 1)),
            axis=(-2, -1),
        )
        factor = real_xp.minimum(magnitude, real_xp.ones_like(magnitude))
        hermiticity = (defect * factor) / real_xp.maximum(
            norm * factor, real_xp.ones_like(norm)
        )
        negative = real_xp.minimum(diagonal, real_xp.zeros_like(diagonal))
        negative_mass = -real_xp.sum(negative, axis=-1)
        negative_max = -real_xp.min(negative, axis=-1)
        previous = self._last_rho_diagnostics.get(site)
        max_hermiticity = real_xp.max(hermiticity)
        max_negative_mass = real_xp.max(negative_mass)
        max_correction = real_xp.max(correction)
        if previous is not None:
            max_hermiticity = real_xp.maximum(
                previous["max_hermiticity_defect"], max_hermiticity
            )
            max_negative_mass = real_xp.maximum(
                previous["max_negative_diagonal_mass"], max_negative_mass
            )
            max_correction = real_xp.maximum(
                previous["max_relative_positivity_correction"], max_correction
            )
        diagnostics = {
            "trace": trace[-1],
            "hermiticity_defect": hermiticity[-1],
            "negative_diagonal_mass": negative_mass[-1],
            "negative_diagonal_max": negative_max[-1],
            "clip_tolerance": tolerance[-1],
            "clipped_negative_mass": real_xp.zeros_like(trace[-1]),
            "evaluation_count": len(rhos) + (
                0 if previous is None else previous["evaluation_count"]
            ),
            "max_hermiticity_defect": max_hermiticity,
            "max_negative_diagonal_mass": max_negative_mass,
            "relative_positivity_correction": correction[-1],
            "max_relative_positivity_correction": max_correction,
        }
        self._last_rho_diagnostics[site] = diagnostics

        # Validate and normalize the scale-free proposal diagonal. Raw rho
        # diagnostics above retain their original scale and anti-Hermitian part.
        proposal_trace = real_xp.sum(proposal_diagonal, axis=-1)
        proposal_scale = real_xp.maximum(
            real_xp.abs(proposal_trace),
            real_xp.max(real_xp.abs(proposal_diagonal), axis=-1),
        )
        proposal_tolerance = 256.0 * np.finfo(ar.get_dtype_name(diagonal)).eps * proposal_scale
        finite = finite & xp.all(xp.isfinite(proposal_diagonal), axis=-1)
        valid_trace = proposal_trace > proposal_tolerance
        valid_negative = real_xp.min(proposal_diagonal, axis=-1) >= -proposal_tolerance
        if not self._scalar(xp.all(finite & valid_trace & valid_negative)):
            if not self._scalar(xp.all(finite)):
                raise ValueError(
                    f"Conditional density matrix at site {site!r} contains "
                    "non-finite values."
                )
            if not self._scalar(xp.all(valid_trace)):
                raise ValueError(
                    f"Conditional density matrix at site {site!r} has invalid "
                    "trace in a prefix group."
                )
            raise ValueError(
                f"Conditional density matrix at site {site!r} has a "
                "substantially negative diagonal."
            )
        proposal_diagonal = real_xp.maximum(
            proposal_diagonal, real_xp.zeros_like(proposal_diagonal)
        )
        if self.rho_positivity is None:
            diagnostics["clipped_negative_mass"] = negative_mass[-1]
        probabilities = proposal_diagonal / real_xp.sum(proposal_diagonal, axis=-1, keepdims=True)
        return probabilities / real_xp.sum(probabilities, axis=-1, keepdims=True)

    def _draw_grouped_choices(self, rng, rhos, groups, *, site):
        """Draw all groups at a site, with one host transfer of chosen indices."""
        xp = self._real_xp
        probabilities = self._conditional_probabilities_batch(
            self._xp.stack(rhos), site=site
        )
        counts = [len(group["indices"]) for group in groups]
        # Only integer grouping metadata moves to the array device.
        with self._array_device_context():
            group_ids = xp.asarray(
                np.repeat(np.arange(len(groups), dtype=np.int32), counts)
            )
            uniforms = rng.random(size=sum(counts))
        cdf = xp.cumsum(probabilities, axis=-1)
        cdf = cdf / cdf[:, -1:]
        # Ignore the final unit CDF entry. Repeated entries skip zero-mass
        # categories, including leading zeros when a uniform draw is zero.
        choices = xp.sum(uniforms[:, None] >= cdf[group_ids, :-1], axis=-1)
        choices = np.asarray(ar.to_numpy(choices), dtype=int)
        offsets = np.cumsum(counts)[:-1]
        values = np.split(choices, offsets)
        safe_probabilities = xp.where(
            probabilities > 0, probabilities, xp.ones_like(probabilities)
        )
        log_probabilities = xp.where(
            probabilities > 0,
            xp.log10(safe_probabilities),
            xp.full_like(probabilities, -float("inf")),
        )
        logs = xp.stack([group["log10"] for group in groups])[:, None]
        self._grouped_draw_count += 1
        return values, logs + log_probabilities

    def _array_device_context(self):
        """Place JAX array creation and RNG operations on the PEPS device."""
        if self.backend == "jax":
            import jax  # noqa: PLC0415

            # Autoray forwards device for Torch, but JAX's eye and RNG key
            # creation use the active default device in supported versions.
            device = next(iter(self._ket.tensors[0].data.devices()))
            return jax.default_device(device)
        return nullcontext()

    def _make_rng(self, seed):
        with self._array_device_context():
            return self._real_xp.random.default_rng(seed)

    def _draw_choices(self, rng, probabilities, *, size=None):
        """Draw on the array backend, then return indices for Quimb/Python."""
        with self._array_device_context():
            choices = rng.choice(len(probabilities), size=size, p=probabilities)
        if size is None:
            return int(self._scalar(choices))
        return np.asarray(ar.to_numpy(choices), dtype=int)

    def _row_bonds(self, y, direction):
        """Return the vertical PEPS bonds adjacent to row ``y``."""
        key = (y, direction)
        if key not in self._row_bond_cache:
            if direction == "bottom":
                neighbor = y - 1
            elif direction == "top":
                neighbor = y + 1
            else:
                raise ValueError("direction must be 'bottom' or 'top'.")
            self._row_bond_cache[key] = (
                tuple(
                    self._ket.bond((x, y), (x, neighbor))
                    for x in range(self.Lx)
                )
                if 0 <= neighbor < self.Ly else ()
            )
        return self._row_bond_cache[key]

    def _identity_future(self, y):
        """Build the ``marginal_chi=0`` identity future cap for row ``y``."""
        import quimb.tensor as qtn  # noqa: PLC0415

        if y in self._identity_future_cache:
            return self._identity_future_cache[y].copy()
        # With no future compression requested, identity tensors preserve the
        # current row's top virtual legs without summing over future rows.
        tensors = []
        for x, top_ind in enumerate(self._row_bonds(y, "top")):
            size = self._ket.ind_size(top_ind)
            with self._array_device_context():
                identity = self._xp.eye(size)
            tensors.append(
                qtn.Tensor(
                    identity,
                    inds=(top_ind, f"{top_ind}_*"),
                    # Give the identity cap the same column tag as the
                    # current row. This lets the row transfer cache retain
                    # it with the column instead of leaving it dangling.
                    tags=(f"PEPSY_FUTURE_{x}", f"X{x}"),
                )
            )
        result = qtn.TensorNetwork(tensors)
        self._identity_future_cache[y] = result.copy()
        return result

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
        if (
            future is None and y < self.Ly - 1
            and self.marginal_chi not in (None, 0)
        ):
            raise RuntimeError(f"Missing future boundary for row {y}; call refresh().")
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

    def _unit_row_array(self, data):
        """Remove a positive scale from a proposal-only cached array."""
        xp = self._xp
        scale = xp.max(xp.abs(data))
        scale = xp.where(scale > 0, scale, xp.ones_like(scale))
        if ar.get_dtype_name(data).startswith("complex"):
            # Component-wise real division also handles subnormal scales:
            # some complex division kernels overflow their reciprocal first.
            return xp.real(data) / scale + 1j * (xp.imag(data) / scale)
        return data / scale

    def _unit_row_tensor(self, tensor):
        """Normalize a newly created cache tensor without changing its phase."""
        tensor.modify(data=self._unit_row_array(tensor.data))
        return tensor

    def _contract_row_tensors(self, tensors, output_inds):
        """Contract a small row transfer network with the shared optimizer."""
        import quimb.tensor as qtn  # noqa: PLC0415

        tensors = [tensor for tensor in tensors if tensor is not None]
        if not tensors:
            return None
        # Use Quimb's public tensor contraction and shared optimizer without
        # constructing another network and its index/tag maps.
        return qtn.tensor_contract(
            *tensors,
            output_inds=tuple(output_inds),
            preserve_tensor=True,
            optimize=self.row_contraction_opt,
        )

    def _estimate_row_cache_bytes(self):
        """Estimate retained row factors/environments before building a cache."""
        if self._row_cache_estimate_per_group is not None:
            return self._row_cache_estimate_per_group
        largest_row = 0
        for y in range(self.Ly):
            center = self._boundary_center(y, None)
            columns = [center.select(f"X{x}", "any") for x in range(self.Lx)]
            column_inds = [set(self._network_inds(column)) for column in columns]
            bottom_sizes = [
                self._ket.ind_size(ind) for ind in self._row_bonds(y, "bottom")
            ]
            interfaces = []
            for x in range(self.Lx - 1):
                shared = column_inds[x] & column_inds[x + 1]
                size = math.prod(center.ind_size(ind) for ind in shared)
                if bottom_sizes:
                    # Without compression, the represented bond can exceed
                    # the Schmidt rank: every applied row multiplies it.
                    rank = math.prod(
                        self._ket.ind_size(self._ket.bond((x, row), (x + 1, row)))
                        for row in range(y)
                    )
                    if self.ket_compression is not None:
                        rank = min(
                            rank, self.sample_chi,
                            math.prod(bottom_sizes[:x + 1]),
                            math.prod(bottom_sizes[x + 1:]),
                        )
                    size *= rank**2
                interfaces.append(size)
            row_elements = 0
            if self.row_cache_mode == "factored":
                row_elements = sum(t.data.size if self.backend == "numpy" else
                                   math.prod(t.shape) for t in center.tensors)
                if bottom_sizes:
                    # Two copies of the conditioned ket, bounded by the
                    # represented row bond and physical dimensions.
                    ranks = [1]
                    for x in range(self.Lx - 1):
                        rank = math.prod(
                            self._ket.ind_size(self._ket.bond((x, row), (x + 1, row)))
                            for row in range(y)
                        )
                        if self.ket_compression is not None:
                            rank = min(rank, self.sample_chi,
                                       math.prod(bottom_sizes[:x + 1]),
                                       math.prod(bottom_sizes[x + 1:]))
                        ranks.append(rank)
                    ranks.append(1)
                    row_elements += 2 * sum(
                        d * ranks[x] * ranks[x + 1] for x, d in enumerate(bottom_sizes)
                    )
            for x in range(self.Lx):
                left = interfaces[x - 1] if x else 1
                right = interfaces[x] if x < self.Lx - 1 else 1
                physical = self._ket.ind_size(self._site_inds[x, y])
                # Local transfer, traced transfer, and suffix/prefix storage.
                if self.row_cache_mode == "factored":
                    row_elements += left + right + physical**2
                else:
                    row_elements += (physical**2 + 1) * left * right + left + right
            if y == 0:
                # Retained between calls, in addition to the current row's
                # live groups. Include it even before the first cache build.
                self._initial_row_cache_estimate_bytes = row_elements * self._array_itemsize
            # Allow extra simultaneous intermediates and contraction workspace.
            largest_row = max(largest_row, 4 * row_elements * self._array_itemsize)
        self._row_cache_estimate_per_group = largest_row
        return largest_row

    def _use_row_cache(self, samples):
        """Select dense transfers only when their estimated work/storage fits."""
        estimated = (
            self._estimate_row_cache_bytes() * samples + self._initial_row_cache_estimate_bytes
            if self.row_cache_max_bytes else None
        )
        group_limit = 32 if self.Lx * self.Ly <= 9 else 4
        if self.row_cache_max_bytes == 0:
            reason = "disabled"
        elif estimated > self.row_cache_max_bytes:
            reason = "memory-budget"
        elif self.row_cache_mode == "dense" and samples > group_limit:
            reason = "prefix-count"
        elif (self.row_cache_mode == "dense" and self.marginal_chi not in (None, 0)
              and self.Lx * self.Ly > 9):
            reason = "large-future"
        else:
            reason = "within-budget"
        self._last_cache_decision = {
            "estimated_cache_bytes": estimated,
            "cache_budget_bytes": self.row_cache_max_bytes,
            "cache_representation": self.row_cache_mode,
            "cache_decision": reason,
        }
        return reason == "within-budget"

    def _get_row_transfer_cache(self, y, phi):
        """Reuse only the row with no sampled predecessor; return cache, built."""
        if y == 0 and phi is None and self._initial_row_cache is not None:
            return self._initial_row_cache, False
        cache = self._build_row_transfer_cache(y, phi)
        if y == 0 and phi is None:
            self._initial_row_cache = cache
        return cache, True

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
        if self.row_cache_mode == "factored":
            return self._build_factored_row_cache(y, phi)
        center = self._boundary_center(y, phi)
        columns = [
            center.select(f"X{x}", "any")
            for x in range(self.Lx)
        ]
        local = []
        trace = []
        interfaces = []
        physical = []
        ordered_inds = [self._network_inds(column) for column in columns]
        index_sets = [set(inds) for inds in ordered_inds]

        for x, column in enumerate(columns):
            site = (x, y)
            ket_ind = self._site_inds[site]
            bra_ind = f"{ket_ind}__pepsy_row_bra"
            site_tag = self._site_tags[site]
            column_inds = index_sets[x]
            left_inds = (
                column_inds & index_sets[x - 1]
                if x
                else set()
            )
            right_inds = (
                column_inds & index_sets[x + 1]
                if x < self.Lx - 1
                else set()
            )
            # Preserve the tensor's index order, which makes the transfer
            # shapes deterministic and keeps left/right interfaces distinct.
            left_inds = tuple(
                ind for ind in ordered_inds[x] if ind in left_inds
            )
            right_inds = tuple(
                ind for ind in ordered_inds[x] if ind in right_inds
            )
            interfaces.append((left_inds, right_inds))
            physical.append((ket_ind, bra_ind))

            split = column.copy()
            # Each factor contributes only a common positive scalar to rho.
            # Scale before contracting so a cached column cannot overflow just
            # because the original PEPS carries a large overall normalization.
            split.apply_to_arrays(self._unit_row_array)
            split.select([site_tag, "BRA"], which="all").reindex_(
                {ket_ind: bra_ind}
            )
            local.append(
                self._unit_row_tensor(split.contract(
                    all,
                    output_inds=(ket_ind, bra_ind, *left_inds, *right_inds),
                    optimize=self.contraction_opt,
                ))
            )
            # Trace the already-contracted local tensor instead of
            # contracting the whole column a second time.
            trace.append(self._unit_row_tensor(
                local[-1].trace(ket_ind, bra_ind, preserve_tensor=True)
            ))

        right = [None] * self.Lx
        if self.Lx > 1:
            suffix = trace[-1]
            right[-2] = suffix
            for x in range(self.Lx - 2, 0, -1):
                suffix = self._unit_row_tensor(self._contract_row_tensors(
                    (trace[x], suffix), interfaces[x][0]
                ))
                right[x - 1] = suffix

        return {
            "local": tuple(local),
            "right": tuple(right),
            "interfaces": tuple(interfaces),
            "physical": tuple(physical),
        }

    def _build_factored_row_cache(self, y, phi):
        """Cache suffixes while keeping each column's local factors separate.

        No approximation is introduced here: only the configured future and
        conditioned-ket compression truncate. A dense transfer with both
        horizontal interfaces open is never materialized.
        """
        center = self._boundary_center(y, phi)
        columns = [tuple(center.select(f"X{x}", "any").tensors) for x in range(self.Lx)]
        ordered = [tuple(dict.fromkeys(i for t in ts for i in t.inds)) for ts in columns]
        index_sets = [set(inds) for inds in ordered]
        local, interfaces, physical, traced = [], [], [], []
        for x, tensors in enumerate(columns):
            left = tuple(i for i in ordered[x] if x and i in index_sets[x - 1])
            right = tuple(i for i in ordered[x] if x + 1 < self.Lx and i in index_sets[x + 1])
            interfaces.append((left, right))
            ki = self._site_inds[x, y]
            bi = f"{ki}__pepsy_row_bra"
            tag = self._site_tags[x, y]
            physical.append((ki, bi))
            factors = tuple(self._unit_row_tensor(t.copy()) for t in tensors)
            traced.append(factors)
            local.append(tuple(t.reindex({ki: bi}) if tag in t.tags and "BRA" in t.tags
                               else t for t in factors))
        right = [None] * self.Lx
        suffix = None
        for x in range(self.Lx - 1, 0, -1):
            suffix = self._unit_row_tensor(self._contract_row_tensors(
                (*traced[x], suffix), interfaces[x][0]
            ))
            right[x - 1] = suffix
        return {"local": tuple(local), "right": tuple(right),
                "interfaces": tuple(interfaces), "physical": tuple(physical),
                "factored": True}

    def _row_local_rho(self, row_cache, x, y, left):
        """Contract one cached row transfer with its prefix and suffix."""
        site = (x, y)
        ket_ind = self._site_inds[site]
        bra_ind = f"{ket_ind}__pepsy_row_bra"
        rho = self._contract_row_tensors(
            (
                left,
                *(row_cache["local"][x] if row_cache.get("factored")
                  else (row_cache["local"][x],)),
                row_cache["right"][x],
            ),
            (ket_ind, bra_ind),
        )
        rho = rho.data
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(
                f"Local density matrix at site {site!r} has invalid shape "
                f"{rho.shape!r}."
            )
        return rho, ket_ind, bra_ind

    def _advance_row_prefix(self, row_cache, x, value, left):
        """Fix one local transfer and return the conditioned left prefix."""
        # No later site consumes the completed row's scalar prefix.
        if x == self.Lx - 1:
            return None
        if row_cache.get("factored"):
            physical = dict.fromkeys(row_cache["physical"][x], int(value))
            projected = tuple(t.isel({i: physical[i] for i in t.inds if i in physical})
                              for t in row_cache["local"][x])
            return self._unit_row_tensor(self._contract_row_tensors(
                (left, *projected), row_cache["interfaces"][x][1]
            ))
        local = row_cache["local"][x].copy()
        ket_ind, bra_ind = row_cache["physical"][x]
        local.isel_({ket_ind: int(value), bra_ind: int(value)})
        if left is None:
            return self._unit_row_tensor(local)
        return self._unit_row_tensor(self._contract_row_tensors(
            (left, local), row_cache["interfaces"][x][1]
        ))

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

    def _rescale_proposal_boundary_(self, network):
        """Bound private proposal tensors before scale-invariant compression.

        Relative cutoffs are unchanged by a positive overall network scalar.
        Remove it before Quimb squares singular values (which can overflow in
        complex64). Absolute cutoffs must see their original tensor scales.
        This helper must never be applied to the amplitude ket or norm oracle.
        """
        if self.cutoff_mode not in {"rel", "rsum1", "rsum2"}:
            return

        def rescale(data):
            # Divide before taking a Euclidean norm: even the tensor norm can
            # overflow on finite complex64 data. Keep zero arrays unchanged.
            largest = self._xp.max(self._xp.abs(data))
            data = data / self._xp.where(largest == 0, 1.0, largest)
            norm = self._xp.linalg.norm(self._xp.reshape(data, (-1,)))
            return data / self._xp.where(norm == 0, 1.0, norm)

        network.apply_to_arrays(rescale)
        # Only a normalized conditional uses this private network; its scalar
        # cancels. Physical amplitudes are contracted separately from _ket.
        network.exponent = 0.0

    def _compress_conditioned_boundary(self, phi):
        """Compress a conditioned boundary with the selected backend."""
        if self.ket_compression is None:
            return phi

        self._rescale_proposal_boundary_(phi)
        if self.ket_compression == "quimb":
            # Quimb is the direct, deterministic SVD/truncation path.
            maybe_compressed = phi.compress(
                max_bond=self.sample_chi,
                cutoff=self.cutoff,
                cutoff_mode=self.cutoff_mode,
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
                cutoff_mode=self.cutoff_mode,
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
                fit.run_eff(
                    n_iter=self.fit_n_iter,
                    cutoff=self.cutoff,
                    cutoff_mode=self.cutoff_mode,
                )
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
        if not self._use_row_cache(samples=1):
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
            config = self._validate_config(config)

        sampled = []
        log10_probability = 0.0
        phi = None
        config_pos = 0
        cache_builds = 0
        cache_hits = 0
        self._last_row_cache_stats = {
            "rows": self.Ly, "suffix_cache_builds": 0,
            "site_prefix_updates": 0, "initial_row_cache_hits": 0,
            "mode": "factored" if self.row_cache_mode == "factored" else "transfer",
        }

        for y in range(self.Ly):
            row_cache, built = self._get_row_transfer_cache(y, phi)
            cache_builds += int(built)
            cache_hits += int(not built)
            self._last_row_cache_stats.update(
                suffix_cache_builds=cache_builds, initial_row_cache_hits=cache_hits,
            )
            left = None
            row_config = []
            for x in range(self.Lx):
                site = (x, y)
                rho, _, _ = self._row_local_rho(row_cache, x, y, left)
                probabilities = self._conditional_probabilities(rho, site=site)
                if config is None:
                    value = self._draw_choices(rng, probabilities)
                else:
                    value = config[config_pos]
                    config_pos += 1
                    if value < 0 or value >= len(probabilities):
                        raise ValueError(
                            f"Physical value {value} at site {site!r} is "
                            f"outside range(0, {len(probabilities)})."
                        )
                selected_probability = probabilities[value]
                if config is not None and self._scalar(selected_probability) <= 0:
                    return list(config), (0.0, 0)
                log10_probability = (
                    log10_probability + self._real_xp.log10(selected_probability)
                )
                sampled.append(value)
                row_config.append(value)
                # Fix immediately, then carry only the contracted prefix into
                # the next x-site. The cached suffix is never mutated.
                left = self._advance_row_prefix(row_cache, x, value, left)
                self._last_row_cache_stats["site_prefix_updates"] += int(x < self.Lx - 1)
            phi = self._update_conditioned_boundary(y, row_config, phi)

        self._last_boundary_mps = None if phi is None else phi.copy()
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": cache_builds,
            "site_prefix_updates": self.Ly * (self.Lx - 1),
            "initial_row_cache_hits": cache_hits,
            "mode": "factored" if self.row_cache_mode == "factored" else "transfer",
        }
        omega = self._log10_to_scaled(log10_probability)
        return sampled, omega

    def _boundary_sample_or_probability_reference(self, rng=None, config=None):
        """Reference proposal using a full center contraction at each site."""
        self._reset_rho_diagnostics()
        if config is not None:
            config = self._validate_config(config)

        sampled = []
        log10_probability = 0.0
        phi = None
        config_pos = 0

        for y in range(self.Ly):
            center = self._boundary_center(y, phi)
            row_config = []
            for x in range(self.Lx):
                site = (x, y)
                rho, ket_ind = self._local_rho(
                    center, site, strip_exponent=config is not None
                )
                probabilities = self._conditional_probabilities(rho, site=site)
                if config is None:
                    value = self._draw_choices(rng, probabilities)
                else:
                    value = config[config_pos]
                    config_pos += 1
                    if value < 0 or value >= len(probabilities):
                        raise ValueError(
                            f"Physical value {value} at site {site!r} is "
                            f"outside range(0, {len(probabilities)})."
                        )
                selected_probability = probabilities[value]
                if config is not None and self._scalar(selected_probability) <= 0:
                    return list(config), (0.0, 0)
                log10_probability = (
                    log10_probability + self._real_xp.log10(selected_probability)
                )
                sampled.append(value)
                row_config.append(value)
                # Fix immediately: the next x-site must condition on this
                # sampled prefix rather than on the unconditioned row.
                center.isel_({ket_ind: value})
            phi = self._update_conditioned_boundary(y, row_config, phi)

        self._last_boundary_mps = None if phi is None else phi.copy()
        omega = self._log10_to_scaled(log10_probability)
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
        return projected.contract(all, optimize=self.contraction_opt)

    def _prepare_amplitude_leaves(self, specs):
        """Cache normalized physical slices and their scale for either method."""
        leaves = []
        for tensor, spec in zip(self._ket.tensors, specs):
            axes = tuple(i for i, pos in enumerate(spec) if pos is None)
            magnitude = self._xp.abs(tensor.data)
            largest = self._real_xp.max(magnitude, axis=axes, keepdims=True) if axes else magnitude
            scale = self._real_xp.where(largest > 0, largest, self._real_xp.ones_like(largest))
            if ar.get_dtype_name(tensor.data).startswith("complex"):
                data = self._xp.real(tensor.data) / scale + 1j * (self._xp.imag(tensor.data) / scale)
            else:
                data = tensor.data / scale
            power = self._real_xp.where(largest > 0, self._real_xp.log10(scale), -math.inf)
            leaves.append((data, power))
        # Publish only complete caches so failed preparation can be retried.
        self._amplitude_specs = specs
        self._amplitude_leaves = tuple(leaves)

    def _projected_amplitude_scaled(self, config):
        """Evaluate a scaled amplitude using the selected contraction method.

        Projected leaves and intermediate contractions are rescaled before
        magnitudes leave the dtype range. Removed positive factors are added
        to the returned base-10 exponent; phase and the physical scale remain.
        Normalized physical slices are cached until refresh(); evaluated
        amplitudes and sampled configurations are not retained between calls.
        Boundary mode instead compresses projected rows before contracting
        the final one-dimensional boundary, retaining its physical scale.
        """
        if self.amplitude_mode == "boundary":
            if self._amplitude_leaves is None:
                positions = {self._site_inds[site]: i for i, site in enumerate(self.site_order)}
                self._prepare_amplitude_leaves(tuple(
                    tuple(positions.get(ind) for ind in tensor.inds) for tensor in self._ket.tensors
                ))
            projected = self._ket.isel({
                self._site_inds[site]: int(value)
                for site, value in zip(self.site_order, config)
            })
            powers = []
            for tensor, (data, scale), spec in zip(
                projected.tensors, self._amplitude_leaves, self._amplitude_specs
            ):
                selectors = tuple(slice(None) if pos is None else int(config[pos]) for pos in spec)
                tensor.modify(data=data[selectors])
                powers.append(self._real_xp.reshape(scale[selectors], ()))
            physical_power = float(self._scalar(self._real_xp.sum(self._real_xp.stack(powers))))
            self._amplitude_stats["contractions"] += 1
            if physical_power == -math.inf:
                return 0.0j, 0
            # Keep large Python exponent metadata out of backend float32 math.
            projected.exponent = 0.0
            # Collapse every row into a bounded single-layer MPS before the
            # remaining one-dimensional scalar contraction. Proposal-only
            # rescaled boundaries cannot supply physical amplitude scales.
            boundary = projected.contract_boundary(
                max_bond=self.amplitude_chi, cutoff=self.cutoff,
                compress_opts={"cutoff_mode": self.cutoff_mode},
                sequence=("ymin",), max_separation=0,
                equalize_norms=True, final_contract=False,
            )
            # This tree is only the remaining one-dimensional boundary, not
            # a full-network amplitude plan. Preserve zeros and extreme scales.
            tree = boundary.contraction_tree(optimize=self.row_contraction_opt, output_inds=())
            mantissa, exponent = tree.contract(boundary.arrays, strip_exponent=True,
                                               check_zero=True, backend=self.backend, autojit=False)
            if self._scalar(mantissa) == 0:
                return 0.0j, 0
            return self._scalar(mantissa), float(
                self._scalar(exponent) + boundary.exponent + physical_power + self._ket.exponent
            )
        if self._amplitude_tree is None:
            site_positions = {self._site_inds[site]: i for i, site in enumerate(self.site_order)}
            specs = tuple(
                tuple(site_positions.get(ind) for ind in tensor.inds)
                for tensor in self._ket.tensors
            )
            projected = self._ket.isel({ind: int(config[pos])
                                        for ind, pos in site_positions.items()})
            tree = projected.contraction_tree(
                optimize=self.contraction_opt, output_inds=()
            )
            # Refuse oversized work before allocating scaled leaves or running
            # the tree. The estimator does not include all live input/workspace.
            self._check_amplitude_plan(tree)
            # Every site's physical slices are immutable until refresh. Scale
            # them together once, retaining at most one extra ket's array data,
            # instead of launching max/divide/log kernels for every shot.
            self._prepare_amplitude_leaves(specs)
            self._amplitude_tree = tree
            self._amplitude_stats["plan_builds"] += 1
        # Recheck cached plans too, including limits adjusted between batches.
        self._check_amplitude_plan(self._amplitude_tree)
        arrays, powers = [], []
        for (data, scale), spec in zip(self._amplitude_leaves, self._amplitude_specs):
            selectors = tuple(slice(None) if pos is None else int(config[pos]) for pos in spec)
            arrays.append(data[selectors])
            powers.append(self._real_xp.reshape(scale[selectors], ()))
        exponent = self._real_xp.sum(self._real_xp.stack(powers))
        mantissa, power = self._amplitude_tree.contract(
            arrays, strip_exponent=True, check_zero=True, backend=self.backend,
            autojit=False,
        )
        self._amplitude_stats["contractions"] += 1
        mantissa = self._scalar(mantissa)
        if mantissa == 0:
            return 0.0j, 0
        # Return fractional exponents directly: the public pair/log convention
        # supports them and never needs to reconstruct an unscaled amplitude.
        # Network exponent metadata is a Python float: do not round it through
        # float32 array arithmetic before adding a small slice exponent.
        return mantissa, float(self._scalar(exponent) + self._scalar(power) + self._ket.exponent)

    def _validate_config(self, config):
        """Validate the whole configuration before a zero branch can exit early."""
        config = tuple(config)
        if len(config) != len(self.site_order):
            raise ValueError(
                f"Expected {len(self.site_order)} physical values, got {len(config)}."
            )
        for site, value in zip(self.site_order, config):
            if not isinstance(value, (int, np.integer, np.bool_)):
                raise ValueError(
                    f"Physical value {value!r} at site {site!r} must be an integer."
                )
            size = self._ket.ind_size(self._site_inds[site])
            if not 0 <= value < size:
                raise ValueError(
                    f"Physical value {value} at site {site!r} is outside range(0, {size})."
                )
        return tuple(int(value) for value in config)

    def _exact_log10_probability(self, config):
        """Accumulate an exact likelihood without multiplying small probabilities."""
        self._reset_rho_diagnostics()
        config = self._validate_config(config)
        working = self._norm.copy()
        log10_probability = self._zero_log_probability
        for site, value in zip(self.site_order, config):
            rho, ket_ind = self._local_rho(working, site, strip_exponent=True)
            probabilities = self._conditional_probabilities(rho, site=site)
            selected_probability = probabilities[value]
            if self._scalar(selected_probability) <= 0:
                return -math.inf
            log10_probability = (
                log10_probability + self._real_xp.log10(selected_probability)
            )
            working.isel_({ket_ind: value})
        return float(self._scalar(log10_probability))

    def _sampled_amplitude(self, config):
        """Keep proposal-only draws independent of every amplitude evaluator."""
        if self.amplitude_mode in {"proposal", "none"}:
            return None
        return self._projected_amplitude_scaled(config)

    def _sample_one_exact(self, rng):
        """Draw one configuration from the exact serial proposal."""
        self._reset_rho_diagnostics()
        working = self._norm.copy()
        config = []
        log10_proposal = 0.0

        for site in self.site_order:
            rho, ket_ind = self._local_rho(working, site)
            probabilities = self._conditional_probabilities(rho, site=site)
            value = self._draw_choices(rng, probabilities)
            config.append(value)
            log10_proposal = (
                log10_proposal + self._real_xp.log10(probabilities[value])
            )
            working.isel_({ket_ind: value})

        amplitude = self._sampled_amplitude(config)
        return config, self._log10_to_scaled(log10_proposal), amplitude

    def _sample_one(self, rng):
        """Draw one configuration from the selected proposal backend."""
        if self.boundary_engine == "exact":
            return self._sample_one_exact(rng)

        config, omega = self._boundary_sample_or_probability(rng=rng)
        return config, omega, self._sampled_amplitude(config)

    def _log10_to_scaled(self, log10_probability):
        """Convert a base-10 log probability to mantissa/exponent form."""
        log10_probability = self._scalar(log10_probability)
        exponent = math.floor(log10_probability)
        return 10.0 ** (log10_probability - exponent), exponent

    def _sample_batch_exact(self, rng, samples):
        """Sample exact proposals while sharing identical prefixes."""
        groups = [
            {
                "indices": np.arange(samples, dtype=int),
                "config": [],
                "working": self._norm.copy(),
                "log10": self._zero_log_probability,
            }
        ]
        max_groups = 1

        for site in self.site_order:
            next_groups = []
            rhos = [self._local_rho(group["working"], site)[0] for group in groups]
            draws, logs = self._draw_grouped_choices(rng, rhos, groups, site=site)
            ket_ind = self._site_inds[site]
            for gi, (group, values) in enumerate(zip(groups, draws)):
                outcomes = np.unique(values)
                for value in outcomes:
                    shot_indices = group["indices"][values == value]
                    # A group owns its network. If it does not split, move
                    # that network forward instead of copying it at every site.
                    working = (group.pop("working") if len(outcomes) == 1
                               else group["working"].copy())
                    working.isel_({ket_ind: int(value)})
                    next_groups.append(
                        {
                            "indices": shot_indices,
                            "config": group["config"] + [int(value)],
                            "working": working,
                            "log10": logs[gi, value],
                        }
                    )
            groups = next_groups
            max_groups = max(max_groups, len(groups))

        # Completed conditionals are not needed by amplitude contractions.
        for group in groups:
            group.pop("working")
        return groups, max_groups

    def _sample_batch_boundary(self, rng, samples):
        """Sample boundary proposals with one row cache per prefix group."""
        # Default factored environments are numerical caches tied to each
        # incoming prefix. Bound their estimated live storage before building;
        # the explicit legacy dense mode has additional fallback heuristics.
        if not self._use_row_cache(samples):
            return self._sample_batch_boundary_reference(rng, samples)

        groups = [
            {
                "indices": np.arange(samples, dtype=int),
                "config": [],
                "log10": self._zero_log_probability,
                "phi": None,
                "row_cache": None,
                "left": None,
                "row_config": [],
            }
        ]
        max_groups = 1
        cache_builds = 0
        cache_hits = 0
        prefix_updates = 0

        for y in range(self.Ly):
            for group in groups:
                # A group represents one shared prefix, hence it has one
                # conditioned lower boundary and one reusable row suffix.
                group["row_cache"], built = self._get_row_transfer_cache(
                    y, group["phi"],
                )
                cache_builds += int(built)
                cache_hits += int(not built)
                group["left"] = None
                group["row_config"] = []

            for x in range(self.Lx):
                site = (x, y)
                next_groups = []
                rhos = [
                    self._row_local_rho(
                        group["row_cache"], x, y, group["left"]
                    )[0]
                    for group in groups
                ]
                draws, logs = self._draw_grouped_choices(rng, rhos, groups, site=site)
                for gi, (group, values) in enumerate(zip(groups, draws)):
                    for value in np.unique(values):
                        shot_indices = group["indices"][values == value]
                        left = self._advance_row_prefix(
                            group["row_cache"],
                            x,
                            int(value),
                            group["left"],
                        )
                        prefix_updates += int(x < self.Lx - 1)
                        next_groups.append(
                            {
                                "indices": shot_indices,
                                "config": group["config"] + [int(value)],
                                "log10": logs[gi, value],
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
        for group in groups:
            group["phi"] = None
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": cache_builds,
            "site_prefix_updates": prefix_updates,
            "initial_row_cache_hits": cache_hits,
            "mode": "factored" if self.row_cache_mode == "factored" else "transfer",
        }
        return groups, max_groups

    def _sample_batch_boundary_reference(self, rng, samples):
        """Reference prefix batch path used when groups are highly fragmented."""
        groups = [
            {
                "indices": np.arange(samples, dtype=int),
                "config": [],
                "log10": self._zero_log_probability,
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
                rhos = [
                    self._local_rho(group["center"], site)[0] for group in groups
                ]
                draws, logs = self._draw_grouped_choices(rng, rhos, groups, site=site)
                ket_ind = self._site_inds[site]
                for gi, (group, values) in enumerate(zip(groups, draws)):
                    outcomes = np.unique(values)
                    for value in outcomes:
                        shot_indices = group["indices"][values == value]
                        center = (group.pop("center") if len(outcomes) == 1
                                  else group["center"].copy())
                        center.isel_({ket_ind: int(value)})
                        next_groups.append(
                            {
                                "indices": shot_indices,
                                "config": group["config"] + [int(value)],
                                "log10": logs[gi, value],
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
        for group in groups:
            group["phi"] = None
        self._last_row_cache_stats = {
            "rows": self.Ly,
            "suffix_cache_builds": 0,
            "site_prefix_updates": 0,
            "mode": "reference-prefix",
        }
        return groups, max_groups

    def _sample_batch(self, rng, samples):
        """Draw a prefix-grouped batch of independent PEPS samples.

        Groups share a local conditional network until their sampled prefixes
        differ. This is the safe PEPS analogue of a batch axis: Quimb does not
        receive per-shot isel values on one shared network.

        A group therefore represents identical history, not merely equal
        tensor shapes. Once two shots choose different physical values, their
        conditioned boundary states and subsequent conditional networks are
        different and must split.
        """
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer))
            or int(samples) < 1
        ):
            raise ValueError("samples must be a positive integer.")
        samples = int(samples)
        self._reset_rho_diagnostics()
        self._grouped_draw_count = 0
        if self.boundary_engine == "exact":
            groups, max_groups = self._sample_batch_exact(rng, samples)
        else:
            groups, max_groups = self._sample_batch_boundary(rng, samples)

        configs = [None] * samples
        omegas = [None] * samples
        amplitudes = [None] * samples
        for group in groups:
            omega = self._log10_to_scaled(group["log10"])
            amplitude = self._sampled_amplitude(group["config"])
            for index in group["indices"]:
                index = int(index)
                configs[index] = list(group["config"])
                omegas[index] = omega
                amplitudes[index] = amplitude

        self._last_batch_stats = {
            "samples": samples,
            "final_prefix_groups": len(groups),
            "max_prefix_groups": max_groups,
            "conditional_batches": self._grouped_draw_count,
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
            amplitude_mode=self.amplitude_mode,
            omegas=(
                [value[0] for value in omegas],
                [value[1] for value in omegas],
            ),
            ps=None if self.amplitude_mode in {"proposal", "none"} else (
                [value[0] for value in amplitudes],
                [value[1] for value in amplitudes],
            ),
        )

    @staticmethod
    def _positive_sample_count(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _resolve_chunk_size(self, chunk_size, samples):
        """Bound automatic prefix groups while retaining useful row caches."""
        if not isinstance(chunk_size, str) or chunk_size != "auto":
            return self._positive_sample_count(chunk_size, "chunk_size")
        # Exact groups hold full conditioned networks. Without a usable row
        # cache, prefer one history at a time over many independent centers.
        if self.boundary_engine == "exact" or not self.row_cache_max_bytes:
            return 1
        per_group = self._estimate_row_cache_bytes()
        available = self.row_cache_max_bytes - self._initial_row_cache_estimate_bytes
        count = max(1, min(samples, 32, available // max(1, per_group)))
        if self.row_cache_mode == "dense":
            count = min(count, 32 if self.Lx * self.Ly <= 9 else 4)
            if self.marginal_chi not in (None, 0) and self.Lx * self.Ly > 9:
                return 1
        return int(count)

    def iter_samples(
        self, samples: int, *, chunk_size: int | str = 128, seed: int | None = None,
    ) -> Iterator[PEPSSampleResult]:
        """Yield bounded prefix batches with one continuous backend RNG.

        Reproducible for a fixed seed and chunk size; changing chunk size can
        change draw order. Each yielded result and diagnostics describe that
        chunk. Consume/discard results to bound memory; retained results still
        require output storage. Boundary caches are reused until refresh().
        ``"auto"`` selects at most 32 shots using the row-cache estimate, or
        one shot for exact proposals or disabled caches. It is not a total
        memory limit. The resolved size is reported in ``batch_stats``.
        """
        samples = self._positive_sample_count(samples, "samples")
        requested_chunk_size = chunk_size
        chunk_size = self._resolve_chunk_size(chunk_size, samples)
        rng = self._make_rng(seed)
        for start in range(0, samples, chunk_size):
            batch = self._sample_batch(rng, min(chunk_size, samples - start))
            self._last_batch_stats.update(
                requested_chunk_size=requested_chunk_size, chunk_size=chunk_size,
            )
            yield batch
            del batch  # Do not retain discarded output while building the next chunk.

    def sample_batch(
        self, samples: int = 1, seed: int | None = None, *, chunk_size: int | str | None = None,
    ) -> PEPSSampleResult:
        """Sample with shared prefixes and optional bounded working batches.

        ``chunk_size`` bounds live prefix states, not the returned output or
        exact-contraction workspace. Omit it to preserve the original single
        batch draw order. Use :meth:`iter_samples` to stream the output too.
        ``"auto"`` uses the cache-aware bounded policy of :meth:`iter_samples`.
        """
        samples = self._positive_sample_count(samples, "samples")
        if chunk_size is None:
            return self._sample_batch(self._make_rng(seed), samples)
        requested_chunk_size = chunk_size
        chunk_size = self._resolve_chunk_size(chunk_size, samples)
        result = PEPSSampleResult([], ([], []),
                                  None if self.amplitude_mode in {"proposal", "none"} else ([], []),
                                  amplitude_mode=self.amplitude_mode)
        stats = {"samples": samples, "chunks": 0, "chunk_size": chunk_size,
                 "requested_chunk_size": requested_chunk_size,
                 "max_prefix_groups": 0, "conditional_batches": 0,
                 "final_prefix_groups": 0, "boundary_engine": self.boundary_engine}
        diagnostics = {}
        for batch in self.iter_samples(samples, chunk_size=chunk_size, seed=seed):
            result.configs.extend(batch.configs)
            for target, source in ((result.omegas, batch.omegas), (result.ps, batch.ps)):
                if target is None:
                    continue
                target[0].extend(source[0])
                target[1].extend(source[1])
            stats["chunks"] += 1
            current = self.batch_stats
            stats["max_prefix_groups"] = max(stats["max_prefix_groups"], current["max_prefix_groups"])
            for key in ("conditional_batches", "final_prefix_groups", "suffix_cache_builds",
                        "site_prefix_updates"):
                stats[key] = stats.get(key, 0) + current.get(key, 0)
            for site, values in self._last_rho_diagnostics.items():
                # Public diagnostics are host scalars, not differentiable
                # outputs. Do not retain every chunk's Torch autograd graph
                # through a chain of maxima when the source requires grad.
                merged = {key: value.detach() if hasattr(value, "detach") else value
                          for key, value in values.items()}
                if site in diagnostics:
                    previous = diagnostics[site]
                    merged["evaluation_count"] += previous["evaluation_count"]
                    for key in ("max_hermiticity_defect", "max_negative_diagonal_mass",
                                "max_relative_positivity_correction"):
                        merged[key] = self._real_xp.maximum(previous[key], merged[key])
                diagnostics[site] = merged
        self._last_batch_stats = stats
        self._last_rho_diagnostics = diagnostics
        return result

    @staticmethod
    def _scaled_to_float(value):
        """Convert a real mantissa/exponent pair to a float when possible."""
        mantissa, exponent = value
        if mantissa == 0:
            return 0.0
        return float(mantissa * (10.0 ** exponent))

    def log_probability(self, config):
        """Return natural-log proposal probability, or ``-inf`` for a zero branch.

        Exact and default boundary likelihood evaluation use scaled conditional
        contractions and log accumulation. Use this method when the probability is
        too small for a Python float. Boundary mode evaluates the selected
        approximate proposal, while exact mode evaluates the Born probability.
        """
        config = self._validate_config(config)
        if self.boundary_engine == "exact":
            log10_probability = self._exact_log10_probability(config)
        else:
            _, omega = self._boundary_sample_or_probability(config=config)
            if omega[0] == 0:
                return -math.inf
            log10_probability = math.log10(omega[0]) + omega[1]
        return log10_probability * math.log(10.0)

    def probability(self, config):
        """Return the selected sequential proposal probability of ``config``.

        This is the exact Born probability in exact mode and the normalized
        approximate proposal in boundary mode. Use ``log_probability`` for
        rare configurations whose probability underflows a Python float.
        """
        return math.exp(self.log_probability(config))

    def sample(self, samples: int = 1, seed: int | None = None) -> PEPSSampleResult:
        """Draw independent configurations from the selected proposal."""
        samples = self._positive_sample_count(samples, "samples")
        rng = self._make_rng(seed)
        configs = []
        omegas_mantissa = []
        omegas_exponent = []
        ps_mantissa = []
        ps_exponent = []

        for _ in range(int(samples)):
            config, omega, amplitude = self._sample_one(rng)
            configs.append(config)
            omega_mantissa, omega_exponent = omega
            omegas_mantissa.append(omega_mantissa)
            omegas_exponent.append(omega_exponent)
            if amplitude is not None:
                amplitude_mantissa, amplitude_exponent = amplitude
                ps_mantissa.append(amplitude_mantissa)
                ps_exponent.append(amplitude_exponent)

        return PEPSSampleResult(
            configs=configs,
            amplitude_mode=self.amplitude_mode,
            omegas=(omegas_mantissa, omegas_exponent),
            ps=None if self.amplitude_mode in {"proposal", "none"} else (ps_mantissa, ps_exponent),
        )
