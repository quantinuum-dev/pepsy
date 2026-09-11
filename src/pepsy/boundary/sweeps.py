"""Boundary-MPS sweep utilities for approximate 2D tensor-network contraction."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
import time
import warnings

import numpy as np
import quimb.tensor as qtn
from tqdm.auto import tqdm

from .._internal.cutoff import dtype_auto_cutoff
from ..tensors.core import tn_fidelity
from ..fitting.local import FIT
from ._lattice import has_numbered_axis_tag, infer_lattice_shape

__all__ = ["BoundaryFitDiagnostic", "CompBdy"]


_FIT_MODE_ALIASES = {
    "direct": "direct",
    "src": "src",
    "zipup": "zipup",
    "sdc": "sdc",
    "dm": "dm",
    "eff": "eff",
    "one-site": "eff",
    "dmrg": "eff",
    "dmrg1": "eff",
    "two-site": "two-site",
    "dmrg2": "dmrg2",
    "global": "global",
}

_FIT_INIT_STRATEGY_ALIASES = {
    "auto": "direct",
    "direct": "direct",
    "guess-direct": "guess-direct",
    "guess-src": "guess-src",
    "guess-sdc": "guess-sdc",
}

_FIT_QUIMB_MODES = frozenset({"direct", "src", "zipup", "sdc", "dm"})
_FIT_CUTOFF_MODES = frozenset(
    {"rel", "rsum2", "rsum1", "abs", "sum2", "sum1"}
)
_FIT_LAYER_MODES = frozenset({"joint", "sequential"})


def _canonical_fit_mode_selector(fit_mode):
    """Return the canonical boundary-FIT mode or fail with a useful error.

    Underscores are accepted as spelling aliases so configuration-file values
    such as ``"two_site"`` behave like the documented ``"two-site"`` form.
    ``"one-site"`` is also accepted as a descriptive alias for the historical
    ``"eff"`` mode; the canonical values stored on runtime objects remain
    stable for downstream comparisons and serialization.
    """
    key = str(fit_mode).strip().lower().replace("_", "-")
    if key not in _FIT_MODE_ALIASES:
        raise ValueError(
            f"Unknown fit_mode={fit_mode!r}. Expected 'direct', 'src', "
            "'zipup', 'sdc', 'dm', 'eff', 'two-site', 'dmrg', 'dmrg2', "
            "or 'global'."
        )
    return _FIT_MODE_ALIASES[key]


def _canonical_fit_init_strategy(strategy):
    """Return the supported disposable boundary-guess strategy."""
    key = str(strategy).strip().lower().replace("_", "-")
    try:
        return _FIT_INIT_STRATEGY_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown fit_init_strategy={strategy!r}. Expected 'direct', "
            "'auto', 'guess-direct', 'guess-src', or 'guess-sdc'."
        ) from exc


def _canonical_fit_cutoff_mode(mode):
    """Resolve the boundary cutoff-mode policy before calling Quimb."""
    if mode is None:
        return "rsum2"
    key = str(mode).strip().lower()
    if key == "auto":
        return "rsum2"
    if key not in _FIT_CUTOFF_MODES:
        allowed = ", ".join(sorted(_FIT_CUTOFF_MODES))
        raise ValueError(
            f"Unknown fit_cutoff_mode={mode!r}. Expected 'auto' or one of "
            f"{allowed}."
        )
    return key


def _canonical_fit_layer_mode(mode):
    """Normalize how direct compressors combine tagged tensor layers."""
    key = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "joint": "joint",
        "combined": "joint",
        "all": "joint",
        "sequential": "sequential",
        "layerwise": "sequential",
        "layer-by-layer": "sequential",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_FIT_LAYER_MODES))
        raise ValueError(
            f"Unknown fit_layer_mode={mode!r}. Expected one of {allowed}."
        ) from exc


@dataclass(frozen=True)
class DirectionSpec:
    """Direction-dependent tags and sweep extents."""

    cut_tag_id: str
    site_tag_id: str
    n_steps: int
    left_steps: int
    right_steps: int
    left_index: int


@dataclass(frozen=True)
class BoundaryFitDiagnostic:
    """One boundary-MPS FIT attempt and its convergence/profile metadata.

    ``sweep_timings`` is populated only when ``fit_timing=True``. Cheap
    convergence fields are always retained, so callers can verify adaptive
    stopping without enabling per-site timers or accelerator barriers.
    """

    boundary_key: str
    fit_mode: str
    status: str
    iterations: int
    converged: bool
    convergence_reason: str
    relative_change: float | None
    center_site: int | None
    direction: str | None
    max_bond: int | None
    elapsed_seconds: float | None = None
    sweep_timings: tuple[dict[str, object], ...] = ()
    error: str | None = None
    fit_init_strategy: str = "direct"
    adaptive_sweeps: int = 0
    one_site_refinement_sweeps: int = 0


class CompBdy:  # pylint: disable=too-many-instance-attributes
    """Approximate 2D tensor-network contraction via boundary-MPS sweeps.

    The class fits boundary MPS tensors slice-by-slice on a tagged 2D
    tensor network (``norm``), then contracts the resulting boundary pair to
    a scalar. ``flat=True`` is a shortcut for a single already-flattened
    effective layer: it uses the first slice directly and skips fitting that
    initial slice. It does not mean that a ``BRA``/``PEPO``/``KET`` stack has
    been automatically flattened. The class also supports boundary-only
    updates (full side or single-step) without final contraction.

    Parameters
    ----------
    norm : qtn.TensorNetwork
        Tagged double-layer network. Must include ``X*`` and ``Y*`` tags so
        lattice shape can be inferred.
    mps_boundaries : dict[str, qtn.MatrixProductState]
        Boundary dictionary, typically from ``BdyMPS(...).mps_b``.
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer used for final contraction and fidelity calls.
    fit_contraction_opt : str | object, default="auto-hq"
        Contraction optimizer used by the local :class:`~pepsy.fitting.local.FIT`
        boundary fits. Kept separate from ``contraction_opt`` so the local
        fitting path can be tuned independently of the final contraction.
    fit_mode : {"direct", "src", "zipup", "sdc", "dm", "eff",
        "one-site", "dmrg", "dmrg1", "two-site", "dmrg2", "global"},
        default="eff"
        Boundary compression backend. The Quimb modes ``"direct"``,
        ``"src"``, ``"zipup"``, ``"sdc"``, and ``"dm"`` directly compress
        each boundary target. ``"dmrg"`` (also ``"eff"``/``"one-site"``)
        uses one-site FIT. ``"dmrg2"`` uses two-site FIT for the configured
        warm-up sweeps, then one-site refinement. ``"two-site"`` remains the
        legacy all-two-site FIT mode and ``"global"`` uses ``FIT.run``.
    fit_layer_mode : {"joint", "sequential"}, default="joint"
        For direct Quimb compression modes, combine all tagged layers in one
        local boundary target (``"joint"``), or compress them one at a time
        in the order given by ``layer_tags`` (``"sequential"``). Sequential
        compression is intended for explicitly layered non-flat targets and
        is unavailable for variational FIT modes.
    layer_tags : sequence[str] | None, default=None
        Layer tags and, for ``fit_layer_mode="sequential"``, their absorption
        order. The standard two-layer order is ``("KET", "BRA")``. Supply
        tags explicitly for a three-layer BRA--PEPO--KET network.
    fit_init_strategy : {"direct", "guess-direct", "guess-src", "guess-sdc", "auto"}, default="direct"
        Disposable initial guess for each boundary FIT solve. ``"direct"``
        reuses the existing boundary guess. ``"guess-src"`` first applies
        Quimb's successive randomized compression to a copy of the exact
        boundary target, then starts FIT from that copy. ``"guess-direct"``
        and ``"guess-sdc"`` use the corresponding Quimb compressors.
        ``"auto"`` is an alias for the compatibility-preserving direct
        strategy. This option is ignored by direct Quimb compression modes.
    fit_init_seed : int | None, default=0
        Seed forwarded to randomized disposable boundary guesses.
    fit_block_size : {1, 2, 3}, default=1
        Block size passed to ``FIT.run_eff`` when ``fit_mode="eff"``.
        Block sizes 2 and 3 enable native SVD growth for full-boundary fits.
    fit_adaptive_sweeps : int | None, default=None
        For ``fit_mode="eff"`` with ``fit_block_size`` 2 or 3, use the block
        update for this many initial sweeps and then use one-site refinement.
        ``None`` keeps the selected block size for all fixed sweeps.
    fit_max_bond : int | None, default=None
        Two-site SVD bond cap. When omitted, the current boundary-MPS bond is
        used as a safe cap. PEPS metric helpers pass their requested ``chi``
        explicitly so a lower-rank warm start can grow up to that target.
    fit_sweep_sequence : str, default="RL"
        Repeating local-fit sweep directions. ``"RL"`` runs each boundary
        left-to-right and then right-to-left; alternating directions normally
        converges more evenly than repeatedly sweeping from one side.
    fit_cutoff : float | {"auto"}, default="auto"
        Boundary compression cutoff. ``"auto"`` selects the shared
        dtype-aware cutoff policy.
    fit_cutoff_mode : str | None | {"auto"}, default="auto"
        Quimb cutoff convention. ``"auto"`` and ``None`` resolve to
        ``"rsum2"`` before compression or native FIT splits.
    fit_min_iter : int | None, default=None
        Minimum completed sweeps before adaptive convergence can stop. For
        ``FIT.run_eff``, adaptive stopping requires at least two sweeps.
    fit_rtol : float | None, default=None
        Relative final-center-norm tolerance. ``None`` preserves fixed-sweep
        behavior.
    fit_patience : int, default=1
        Consecutive converged sweeps required when ``fit_rtol`` is enabled.
    fit_timing : bool, default=False
        Record coarse elapsed time for every boundary fit and detailed
        two-site sweep/site timings in :class:`BoundaryFitDiagnostic`.
    fit_timing_sync_device : bool, default=False
        Synchronize supported accelerators at FIT timing boundaries. This is
        intended for profiling and deliberately adds device barriers.

    Notes
    -----
    - ``run(...)`` performs a full two-sided sweep and returns a scalar.
    - ``move_bdy(...)`` updates one side (or both sides) and returns nothing.
    - ``move_step_bdy(...)`` updates exactly one boundary position.
    - ``self.fidel`` is reset at the start of each public call and populated
      only when ``track_boundary_fidelity=True``.
    - ``self.mps_boundaries`` is updated in place when ``write_back=True``.
    """

    def __init__(
        self,
        norm,
        mps_boundaries,
        *,
        contraction_opt="auto-hq",
        fit_contraction_opt="auto-hq",
        fit_mode="eff",
        fit_layer_mode="joint",
        layer_tags=None,
        fit_init_strategy="direct",
        fit_init_seed=0,
        fit_block_size=1,
        fit_adaptive_sweeps=None,
        fit_max_bond=None,
        fit_sweep_sequence="RL",
        fit_cutoff="auto",
        fit_cutoff_mode="auto",
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=1,
        fit_timing=False,
        fit_timing_sync_device=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        if not isinstance(mps_boundaries, dict):
            raise TypeError("mps_boundaries must be a dictionary of boundary states.")

        self.norm = norm
        self.mps_boundaries = mps_boundaries
        self.contraction_opt = contraction_opt
        self.fit_contraction_opt = fit_contraction_opt
        # Validate at construction time rather than after an expensive PEPS
        # boundary has already reached its first local fit.
        self.fit_mode = _canonical_fit_mode_selector(fit_mode)
        self.fit_layer_mode = _canonical_fit_layer_mode(fit_layer_mode)
        if (
            self.fit_layer_mode == "sequential"
            and self.fit_mode not in _FIT_QUIMB_MODES
        ):
            raise ValueError(
                "fit_layer_mode='sequential' is only supported with direct "
                "Quimb fit modes: 'direct', 'src', 'zipup', 'sdc', or 'dm'."
            )
        if layer_tags is None:
            self.layer_tags = None
        elif isinstance(layer_tags, str):
            self.layer_tags = (layer_tags,)
        else:
            try:
                self.layer_tags = tuple(str(tag) for tag in layer_tags)
            except TypeError as exc:
                raise TypeError(
                    "layer_tags must be a string or sequence of strings."
                ) from exc
        if self.layer_tags is not None:
            if not self.layer_tags or any(not tag for tag in self.layer_tags):
                raise ValueError("layer_tags must contain at least one non-empty tag.")
            if len(set(self.layer_tags)) != len(self.layer_tags):
                raise ValueError("layer_tags must not contain duplicates.")
        self.fit_init_strategy = _canonical_fit_init_strategy(fit_init_strategy)
        self.fit_init_seed = fit_init_seed
        if not isinstance(fit_block_size, Integral) or int(fit_block_size) not in {
            1,
            2,
            3,
        }:
            raise ValueError("fit_block_size must be 1, 2, or 3.")
        fit_block_size = int(fit_block_size)
        if self.fit_mode not in {"eff"} and fit_block_size != 1:
            raise ValueError(
                "fit_block_size is only configurable with fit_mode='dmrg'/'eff'."
            )
        if fit_adaptive_sweeps is not None:
            if self.fit_mode == "dmrg2":
                if not isinstance(fit_adaptive_sweeps, Integral) or int(
                    fit_adaptive_sweeps
                ) < 1:
                    raise ValueError(
                        "fit_adaptive_sweeps must be a positive integer or None."
                    )
            elif fit_block_size not in {2, 3}:
                raise ValueError(
                    "fit_adaptive_sweeps requires fit_block_size=2 or 3."
                )
            if (
                not isinstance(fit_adaptive_sweeps, Integral)
                or int(fit_adaptive_sweeps) < 1
            ):
                raise ValueError(
                    "fit_adaptive_sweeps must be a positive integer or None."
                )
            fit_adaptive_sweeps = int(fit_adaptive_sweeps)
        self.fit_block_size = fit_block_size
        self.fit_adaptive_sweeps = fit_adaptive_sweeps
        self.fit_max_bond = fit_max_bond
        self.fit_sweep_sequence = fit_sweep_sequence
        self.fit_cutoff = fit_cutoff
        self.fit_cutoff_mode = _canonical_fit_cutoff_mode(fit_cutoff_mode)
        self.fit_min_iter = fit_min_iter
        self.fit_rtol = fit_rtol
        self.fit_patience = fit_patience
        self.fit_timing = bool(fit_timing)
        self.fit_timing_sync_device = bool(
            self.fit_timing and fit_timing_sync_device
        )

        # Runtime sweep options are configured per call in run/move methods.
        self.equalize_norms = False
        self.n_iter = 10
        self.flat = False
        self.write_back = True
        self.retag = True
        self.visualize = False
        self.track_boundary_fidelity = False
        self.fidel = []
        self.fit_diagnostics = []
        self._fit_init_strategy_used = "direct"
        self._fit_init_warned = False
        self.progress = False
        self.max_separation = 0
        self.direction = "y"

        self.y_left = 0
        self.y_right = 0
        self.x_left = 0
        self.x_right = 0

        # Use the same Lx/Ly-or-tag inference as BdyMPS, while still requiring
        # numbered axis tags because sweep slices are selected by those tags.
        self.Lx, self.Ly = infer_lattice_shape(norm)
        if not (
            has_numbered_axis_tag(norm, "X")
            and has_numbered_axis_tag(norm, "Y")
        ):
            raise ValueError(
                "norm must include numbered X*/Y* tags so boundary slices "
                "can be selected."
            )
        self._update_separation()

    def _reset_fidelity_history(self):
        """Reset stored fidelity values for a fresh public call."""
        self.fidel = []

    def _reset_fit_diagnostics(self):
        """Reset run-local convergence and timing diagnostics."""
        self.fit_diagnostics = []

    @staticmethod
    def _direction_base(direction):
        """Return canonical base direction: ``y`` or ``x``."""
        if direction.startswith("y"):
            return "y"
        if direction.startswith("x"):
            return "x"
        raise ValueError(f"Unsupported direction: {direction}")

    def _axis_length_from_cut_tag(self, cut_tag_id):
        """Return number of cut positions addressed by ``cut_tag_id``."""
        if cut_tag_id == "Y{}":
            return self.Ly
        if cut_tag_id == "X{}":
            return self.Lx
        raise ValueError(f"Unsupported cut_tag_id: {cut_tag_id}")

    def _direction_tags(self, direction):
        """Return ``(cut_tag_id, site_tag_id, n_steps)`` for a direction."""
        base = self._direction_base(direction)
        if base == "y":
            return "Y{}", "X{}", self.Ly - 1
        if base == "x":
            return "X{}", "Y{}", self.Lx - 1
        raise ValueError(f"Unsupported direction: {direction}")

    def _run_direction_spec(self, direction):
        """Return run-time sweep tags and left/right extents."""
        base = self._direction_base(direction)
        if base == "y":
            return DirectionSpec(
                cut_tag_id="Y{}",
                site_tag_id="X{}",
                n_steps=self.y_left + self.y_right,
                left_steps=self.y_left,
                right_steps=self.y_right,
                left_index=self.y_left,
            )
        if base == "x":
            return DirectionSpec(
                cut_tag_id="X{}",
                site_tag_id="Y{}",
                n_steps=self.x_left + self.x_right,
                left_steps=self.x_left,
                right_steps=self.x_right,
                left_index=self.x_left,
            )
        raise ValueError(f"Unsupported direction: {direction}")

    def _apply_runtime_overrides(
        self,
        *,
        mps_boundaries=None,
        retag=True,
        visualize=False,
        flat=False,
        track_boundary_fidelity=False,
        progress=False,
        n_iter=10,
        equalize_norms=False,
        write_back=True,
    ):  # pylint: disable=too-many-arguments
        """Apply run-time options explicitly for a single public call."""
        if mps_boundaries is not None:
            if not isinstance(mps_boundaries, dict):
                raise TypeError("mps_boundaries must be a dictionary of boundary states.")
            self.mps_boundaries = mps_boundaries

        if flat and self.fit_layer_mode != "joint":
            raise ValueError(
                "flat=True handles one already-flattened effective layer and "
                "requires fit_layer_mode='joint'; use an explicitly layered "
                "non-flat target for sequential compression."
            )

        self.retag = retag
        self.visualize = visualize
        self.flat = flat
        self.track_boundary_fidelity = track_boundary_fidelity
        self.progress = progress
        self.n_iter = n_iter
        self.equalize_norms = equalize_norms
        self.write_back = write_back

    def _update_separation(self):
        """Update left/right sweep extents from ``max_separation``."""
        if self.max_separation == 0:
            self.y_left = self.Ly // 2
            self.y_right = self.Ly - (self.Ly // 2)

            self.x_left = self.Lx // 2
            self.x_right = self.Lx - (self.Lx // 2)
        elif self.max_separation == 1:
            # y dir
            self.y_left = (self.Ly // 2) - 1
            self.y_right = self.Ly - (self.Ly // 2)

            # x dir
            self.x_left = (self.Lx // 2) - 1
            self.x_right = self.Lx - (self.Lx // 2)
        else:
            raise ValueError("max_separation must be 0 or 1.")

    def _effective_fit_steps(self, steps):
        """Return number of fitting updates after accounting for ``flat`` skip."""
        if not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if self.flat and steps > 0:
            return steps - 1
        return steps

    def _is_skipped_flat_step(self, step_idx):
        """Return whether a step is skipped due to ``flat=True`` first-slice rule."""
        return self.flat and step_idx == 0

    def _cut_idx_and_key(self, side, step_idx, cut_tag_id):
        """Return ``(cut_idx, boundary_key, axis_len)`` for sweep side/step."""
        axis_len = self._axis_length_from_cut_tag(cut_tag_id)
        if side == "right":
            cut_idx = axis_len - step_idx - 1
            boundary_key = f"{cut_tag_id.format(step_idx)}_r"
        else:
            cut_idx = step_idx
            boundary_key = f"{cut_tag_id.format(step_idx)}_l"
        return cut_idx, boundary_key, axis_len

    @staticmethod
    def _previous_boundary_key(side, step_idx, cut_tag_id):
        """Return previous boundary key for a given side and step."""
        suffix = "_r" if side == "right" else "_l"
        return f"{cut_tag_id.format(step_idx - 1)}{suffix}"

    def _run_fit_solver(self, fit, boundary_mps):
        """Run selected fitting backend with explicit validation."""
        # Fidelity is tracked externally by CompBdy via self.fidel.
        verbose = False
        if boundary_mps.L == 1:
            fit.run(n_iter=self.n_iter, verbose=verbose)
            return
        if self.fit_mode == "eff":
            if (
                self.fit_block_size == 1
                and self.fit_adaptive_sweeps is None
                and self.fit_min_iter is None
                and self.fit_rtol is None
                and self.fit_sweep_sequence == "RL"
                and not (
                    isinstance(self.fit_cutoff, str)
                    and self.fit_cutoff.strip().lower() == "auto"
                )
            ):
                # Keep the historical call shape for lightweight compatible
                # FIT doubles when the caller supplied an explicit numeric
                # cutoff. Automatic cutoffs need the expanded call below.
                fit.run_eff(n_iter=self.n_iter, verbose=verbose)
                return
            max_bond = self.fit_max_bond
            if max_bond is None and self.fit_block_size in {2, 3}:
                max_bond = int(boundary_mps.max_bond())
            fit.run_eff(
                n_iter=self.n_iter,
                verbose=verbose,
                block_size=self.fit_block_size,
                sweep_sequence=self.fit_sweep_sequence,
                max_bond=max_bond,
                cutoff=self.fit_cutoff,
                cutoff_mode=self.fit_cutoff_mode,
                adaptive_block_sweeps=self.fit_adaptive_sweeps,
                min_iter=self.fit_min_iter,
                rtol=self.fit_rtol,
                patience=self.fit_patience,
            )
            return
        if self.fit_mode in {"two-site", "dmrg2"}:
            # The complete boundary is the active DMRG interval. ``run_gate``
            # builds one fixed environment per sweep and updates the opposite
            # environment incrementally, so pair updates remain O(L) rather
            # than rebuilding an O(L) contraction at every bond.
            fit.range_int = (0, int(boundary_mps.L) - 1)
            max_bond = self.fit_max_bond
            if max_bond is None:
                # An uncapped two-site split can grow exponentially. Direct
                # CompBdy callers therefore inherit the current boundary cap;
                # PEPS metric APIs pass the requested chi explicitly.
                max_bond = int(boundary_mps.max_bond())
            run_kwargs = dict(
                n_iter=self.n_iter,
                verbose=verbose,
                block_size=2,
                sweep_sequence=self.fit_sweep_sequence,
                max_bond=max_bond,
                cutoff=self.fit_cutoff,
                cutoff_mode=self.fit_cutoff_mode,
                # Legacy ``two-site`` owns a fixed pair-update schedule.
                # ``dmrg2`` follows MpsOptimizer: two-site warm-up followed
                # by one-site refinement for the remaining sweeps.
                adaptive_block_sweeps=(
                    None
                    if self.fit_mode == "two-site"
                    or self.n_iter < 2
                    else (
                        2
                        if self.fit_adaptive_sweeps is None
                        else self.fit_adaptive_sweeps
                    )
                ),
                min_iter=self.fit_min_iter,
                rtol=self.fit_rtol,
                patience=self.fit_patience,
                collect_split_diagnostics=False,
            )
            if getattr(self, "fit_timing", False):
                run_kwargs["timing"] = True
                run_kwargs["timing_sync_device"] = bool(
                    getattr(self, "fit_timing_sync_device", False)
                )
            fit.run_gate(**run_kwargs)
            return
        if self.fit_mode == "global":
            fit.run(n_iter=self.n_iter, verbose=verbose)
            return
        # Constructor canonicalization makes this defensive rather than a
        # normal user-input path.
        raise RuntimeError(f"Unhandled canonical fit_mode: {self.fit_mode}")

    def _resolve_fit_cutoff(self, tn):
        """Resolve the boundary cutoff for direct Quimb compression modes."""
        if isinstance(self.fit_cutoff, str) and self.fit_cutoff.strip().lower() == "auto":
            for tensor in tn.tensors:
                dtype = getattr(tensor.data, "dtype", None)
                if dtype is not None:
                    return dtype_auto_cutoff(dtype)
            return 1.0e-12
        try:
            cutoff = float(self.fit_cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fit_cutoff must be 'auto' or a non-negative number."
            ) from exc
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("fit_cutoff must be 'auto' or a non-negative number.")
        return cutoff

    def _record_compression_diagnostic(
        self,
        boundary_mps,
        boundary_key,
        *,
        elapsed_seconds,
        fit_mode,
    ):
        """Record one completed non-variational boundary compression."""
        current_bond = boundary_mps.max_bond()
        self.fit_diagnostics.append(
            BoundaryFitDiagnostic(
                boundary_key=str(boundary_key),
                fit_mode=str(fit_mode),
                status="complete",
                iterations=1,
                converged=False,
                convergence_reason="fixed_compression",
                relative_change=None,
                center_site=None,
                direction=None,
                max_bond=(
                    None if current_bond is None else int(current_bond)
                ),
                elapsed_seconds=elapsed_seconds,
                sweep_timings=(),
                error=None,
                fit_init_strategy="not-applicable",
                adaptive_sweeps=0,
                one_site_refinement_sweeps=0,
            )
        )

    def _resolve_fit_layer_tags(self, tn):
        """Return layer tags in the requested sequential absorption order."""
        if self.layer_tags is None:
            layer_tags = tuple(
                tag for tag in ("KET", "BRA") if tag in getattr(tn, "tags", ())
            )
        else:
            layer_tags = self.layer_tags

        missing = [tag for tag in layer_tags if tag not in getattr(tn, "tags", ())]
        if missing:
            missing_text = ", ".join(repr(tag) for tag in missing)
            raise ValueError(
                "fit_layer_mode='sequential' could not find layer tag(s) "
                f"{missing_text} in the current boundary target. Supply "
                "layer_tags in the desired absorption order."
            )
        if not layer_tags:
            raise ValueError(
                "fit_layer_mode='sequential' requires tagged layers. Supply "
                "layer_tags, for example ('BRA', 'PEPO', 'KET')."
            )
        return layer_tags

    def _compress_direct_target(self, tn, *, max_bond, cutoff, site_tags):
        """Compress one local 1D target with the selected Quimb method."""
        if len(site_tags) <= 1:
            return tn.copy()

        method = self.fit_mode
        if method == "src":
            # SRC is rank-controlled; keep the explicit zero cutoff to avoid
            # Quimb's advisory warning and match MPS semantics.
            cutoff = 0.0
        compress_kwargs = {
            "max_bond": int(max_bond),
            "cutoff": cutoff,
            "method": method,
            "site_tags": site_tags,
            "permute_arrays": False,
            "inplace": False,
        }
        if method == "src":
            compress_kwargs["seed"] = self.fit_init_seed
        else:
            compress_kwargs["cutoff_mode"] = self.fit_cutoff_mode
        return qtn.tensor_network_1d_compress(tn.copy(), **compress_kwargs)

    def _compress_boundary(  # pylint: disable=too-many-locals
        self,
        tn,
        boundary_mps,
        boundary_key,
        site_tag_id,
        previous=None,
    ):
        """Compress one boundary target with a direct Quimb method."""
        started = time.perf_counter() if self.fit_timing else None
        if self._uses_symmray_arrays(tn):
            raise NotImplementedError(
                f"fit_mode={self.fit_mode!r} is currently dense-only for "
                "native Symmray boundary targets; use fit_mode='dmrg'."
            )

        max_bond = self.fit_max_bond
        if max_bond is None:
            max_bond = int(boundary_mps.max_bond())
        cutoff = self._resolve_fit_cutoff(tn)
        site_tags = tuple(
            site_tag_id.format(site) for site in range(int(boundary_mps.L))
        )

        if self.fit_layer_mode == "joint":
            compressed = self._compress_direct_target(
                tn,
                max_bond=max_bond,
                cutoff=cutoff,
                site_tags=site_tags,
            )
        else:
            compressed = previous
            layer_tags = self._resolve_fit_layer_tags(tn)
            for layer_tag in layer_tags:
                layer_tn = tn.select(layer_tag, "any")
                target = layer_tn if compressed is None else layer_tn | compressed
                compressed = self._compress_direct_target(
                    target,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    site_tags=site_tags,
                )
                # The result is now a boundary carrying open indices for the
                # remaining layers. Do not let its source-layer tags leak into
                # the next spatial step, where ``select(layer_tag)`` should
                # see only the newly absorbed physical layer.
                present_layer_tags = tuple(
                    tag for tag in layer_tags if tag in compressed.tags
                )
                if present_layer_tags:
                    compressed.drop_tags(present_layer_tags)

        compressed.view_as_(
            qtn.MatrixProductState,
            L=boundary_mps.L,
            site_tag_id=site_tag_id,
            site_ind_id=None,
            cyclic=False,
        )

        if self.equalize_norms:
            compressed.equalize_norms_(value=self.equalize_norms)
        if self.fit_timing_sync_device:
            FIT.synchronize_backend(compressed)
        elapsed = (
            None if started is None else float(time.perf_counter() - started)
        )
        self._record_compression_diagnostic(
            compressed,
            boundary_key,
            elapsed_seconds=elapsed,
            fit_mode=self.fit_mode,
        )
        if self.track_boundary_fidelity:
            fidelity = tn_fidelity(
                tn,
                compressed,
                contraction_opt=self.contraction_opt,
            )
            self.fidel.append(fidelity)
        if self.write_back:
            self.mps_boundaries[boundary_key] = compressed
        return compressed

    def _record_fit_diagnostic(
        self,
        fit,
        boundary_mps,
        boundary_key,
        *,
        status,
        elapsed_seconds,
        error=None,
    ):
        """Append one typed, copy-safe boundary FIT diagnostic."""
        gate_solver_ran = (
            self.fit_mode in {"two-site", "dmrg2"}
            and boundary_mps.L > 1
        )
        adaptive_eff_ran = self.fit_mode == "eff" and self.fit_rtol is not None
        if status == "failed":
            # Failure reporting must not assume run_gate reached its normal
            # epilogue. FIT initializes these fields, but getattr keeps this
            # path safe for compatible solver doubles and future backends.
            iterations = int(getattr(fit, "iterations_run", 0) or 0)
            converged = False
            reason = "failed"
            relative_change = getattr(fit, "last_relative_change", None)
            center_site = getattr(fit, "final_center_site", None)
            direction = getattr(fit, "final_direction", None)
        elif gate_solver_ran or adaptive_eff_ran:
            iterations = int(getattr(fit, "iterations_run", self.n_iter))
            converged = bool(fit.converged)
            reason = str(fit.convergence_reason)
            relative_change = getattr(fit, "last_relative_change", None)
            center_site = getattr(fit, "final_center_site", None)
            direction = getattr(fit, "final_direction", None)
        elif status == "complete":
            # ``run`` and ``run_eff`` are fixed-sweep compatibility solvers and
            # predate FIT's adaptive diagnostic fields. Report what actually
            # ran instead of exposing their constructor's ``not_run`` state.
            iterations = int(self.n_iter)
            converged = False
            reason = "fixed_sweeps"
            relative_change = None
            center_site = None
            direction = None
        if relative_change is not None:
            relative_change = float(relative_change)
        adaptive_sweeps = int(getattr(fit, "adaptive_sweeps_run", 0) or 0)
        one_site_refinement_sweeps = int(
            getattr(fit, "one_site_sweeps_run", 0) or 0
        )
        sweep_timings = (
            tuple(deepcopy(fit.get_timing())) if self.fit_timing else ()
        )
        self.fit_diagnostics.append(
            BoundaryFitDiagnostic(
                boundary_key=str(boundary_key),
                fit_mode=str(self.fit_mode),
                status=str(status),
                iterations=iterations,
                converged=converged,
                convergence_reason=reason,
                relative_change=relative_change,
                center_site=(
                    None if center_site is None else int(center_site)
                ),
                direction=None if direction is None else str(direction),
                max_bond=(
                    None
                    if fit.p.max_bond() is None
                    else int(fit.p.max_bond())
                ),
                elapsed_seconds=elapsed_seconds,
                sweep_timings=sweep_timings,
                error=error,
                fit_init_strategy=str(self._fit_init_strategy_used),
                adaptive_sweeps=adaptive_sweeps,
                one_site_refinement_sweeps=one_site_refinement_sweeps,
            )
        )

    def _maybe_visualize_fit(
        self,
        tn,
        boundary_mps,
        fit,
        site_tag_id,
        axis_len,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Draw intermediate networks when ``visualize`` is enabled."""
        if not self.visualize:
            return

        draw_tags = [site_tag_id.format(i) for i in range(axis_len)]
        draw_kwargs = {
            "legend": False,
            "show_inds": False,
            "show_tags": True,
            "figsize": (8, 8),
            "node_outline_darkness": 0.1,
            "node_outline_size": None,
            "highlight_inds_color": "darkred",
            "edge_scale": 2.0,
            "layout": "neato",
            "refine_layout": "auto",
            "highlight_inds": tn.outer_inds(),
        }
        tn.draw(draw_tags, **draw_kwargs)
        (tn & boundary_mps).draw(draw_tags, **draw_kwargs)
        fit.visual(figsize=(8, 8), show_inds="bond-size", tags_=[])

    def _initial_boundary_mps(self, boundary_key, previous):
        """Return an owned boundary guess with exponent representation aligned."""
        boundary_mps = self.mps_boundaries[boundary_key].copy()

        if self.equalize_norms and previous is not None:
            boundary_mps.exponent = complex(getattr(previous, "exponent", 0.0)).real
        else:
            boundary_mps.exponent = 0.0

        return boundary_mps

    @staticmethod
    def _uses_symmray_arrays(network):
        """Return whether *network* contains native Symmray arrays."""
        return any(
            type(tensor.data).__module__.split(".", maxsplit=1)[0] == "symmray"
            for tensor in network.tensors
        )

    def _build_fit_initial_guess(self, tn, boundary_mps, site_tag_id):
        """Build the disposable initial guess for one boundary FIT solve."""
        self._fit_init_strategy_used = "direct"
        if self.fit_init_strategy == "direct":
            return boundary_mps

        # There is no compression problem for a one-site MPS. Quimb's SRC
        # implementation assumes at least one internal bond, so preserve the
        # direct guess for this valid degenerate boundary shape.
        if int(boundary_mps.L) <= 1:
            return boundary_mps

        guess_method = self.fit_init_strategy.removeprefix("guess-")

        # Quimb's dense compressors cannot preserve Symmray block sectors or
        # fermionic metadata. Keep the exact native path safe and visible
        # rather than coercing the boundary to dense arrays.
        if self._uses_symmray_arrays(tn) or self._uses_symmray_arrays(boundary_mps):
            if not self._fit_init_warned:
                warnings.warn(
                    f"fit_init_strategy={self.fit_init_strategy!r} is not "
                    "available for native Symmray PEPS boundaries; falling "
                    "back to 'direct'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._fit_init_warned = True
            return boundary_mps

        site_tags = tuple(
            site_tag_id.format(site) for site in range(int(boundary_mps.L))
        )
        max_bond = self.fit_max_bond
        if max_bond is None:
            max_bond = int(boundary_mps.max_bond())
        cutoff = self._resolve_fit_cutoff(tn)
        if guess_method == "src":
            # SRC is rank-controlled; keep the explicit zero cutoff to avoid
            # Quimb's advisory warning and match MPS initialization semantics.
            cutoff = 0.0
        guess_kwargs = {
            "max_bond": int(max_bond),
            "cutoff": cutoff,
            "method": guess_method,
            "site_tags": site_tags,
            "permute_arrays": False,
            "inplace": False,
        }
        if guess_method == "src":
            guess_kwargs["seed"] = self.fit_init_seed
        else:
            guess_kwargs["cutoff_mode"] = self.fit_cutoff_mode
        guess = qtn.tensor_network_1d_compress(tn.copy(), **guess_kwargs)
        guess.view_as_(
            qtn.MatrixProductState,
            L=boundary_mps.L,
            site_tag_id=site_tag_id,
            site_ind_id=None,
            cyclic=False,
        )
        # Preserve the boundary's scale convention. The source target is a
        # disposable guess only; FIT still owns the exact target ``tn``.
        guess.exponent = complex(getattr(boundary_mps, "exponent", 0.0)).real
        self._fit_init_strategy_used = self.fit_init_strategy
        return guess

    def _fit_boundary(
        self,
        tn,
        boundary_key,
        previous,
        site_tag_id,
        axis_len,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Fit one boundary MPS against ``tn`` and return the owned result."""
        boundary_mps = self._initial_boundary_mps(boundary_key, previous)

        if self.fit_mode in _FIT_QUIMB_MODES:
            return self._compress_boundary(
                tn,
                boundary_mps,
                boundary_key,
                site_tag_id,
                previous=previous,
            )

        fit_guess = self._build_fit_initial_guess(
            tn,
            boundary_mps,
            site_tag_id,
        )

        fit = FIT(
            tn,
            p=fit_guess,
            inplace=True,
            site_tag_id=site_tag_id,
            contraction_opt=self.fit_contraction_opt,
            retag=self.retag,
        )
        self._maybe_visualize_fit(tn, boundary_mps, fit, site_tag_id, axis_len)
        if self.fit_timing_sync_device:
            FIT.synchronize_backend(fit.p)
        started = time.perf_counter() if self.fit_timing else None
        try:
            self._run_fit_solver(fit, boundary_mps)
        except Exception as exc:
            if self.fit_timing_sync_device:
                FIT.synchronize_backend(fit.p)
            elapsed = (
                None if started is None else float(time.perf_counter() - started)
            )
            try:
                self._record_fit_diagnostic(
                    fit,
                    boundary_mps,
                    boundary_key,
                    status="failed",
                    elapsed_seconds=elapsed,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                # Diagnostics are secondary: never replace the solver error
                # with a profiling or metadata-extraction failure.
                pass
            raise
        else:
            if self.fit_timing_sync_device:
                FIT.synchronize_backend(fit.p)
            elapsed = (
                None if started is None else float(time.perf_counter() - started)
            )
            self._record_fit_diagnostic(
                fit,
                boundary_mps,
                boundary_key,
                status="complete",
                elapsed_seconds=elapsed,
            )

        if self.equalize_norms:
            fit.p.equalize_norms_(value=self.equalize_norms)
        if self.track_boundary_fidelity:
            fidelity = tn_fidelity(tn, fit.p, contraction_opt=self.contraction_opt)
            self.fidel.append(fidelity)

        if self.write_back:
            self.mps_boundaries[boundary_key] = fit.p

        return fit.p

    def _fit_one_side(
        self,
        side,
        steps,
        progress_bar,
        cut_tag_id,
        site_tag_id,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Sweep one side boundary for ``steps`` and fit at each position."""
        previous = None

        for step_idx in range(steps):
            cut_idx, boundary_key, axis_len = self._cut_idx_and_key(
                side,
                step_idx,
                cut_tag_id,
            )
            tn_slice = self.norm.select(cut_tag_id.format(cut_idx), "any")

            if step_idx == 0 and self.flat:
                previous = tn_slice
                continue

            if step_idx == 0:
                tn = tn_slice
            else:
                if previous is None:
                    raise ValueError("Missing previous boundary MPS during fitting.")
                tn = tn_slice | previous

            previous = self._fit_boundary(
                tn,
                boundary_key,
                previous if step_idx > 0 else None,
                site_tag_id=site_tag_id,
                axis_len=axis_len,
            )

            if progress_bar is not None:
                current_bond = previous.max_bond()
                postfix = {
                    "chi": None if current_bond is None else int(current_bond)
                }
                if self.track_boundary_fidelity:
                    prod_fidelity = np.prod(self.fidel)
                    postfix["F"] = complex(prod_fidelity).real
                progress_bar.set_postfix(postfix)
                progress_bar.refresh()
                progress_bar.update(1)

        return previous

    def _fit_one_step(
        self,
        side,
        step_,
        cut_tag_id,
        site_tag_id,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Fit a single boundary step for one side and return updated MPS."""
        previous = None

        cut_idx, boundary_key, axis_len = self._cut_idx_and_key(side, step_, cut_tag_id)
        tn_slice = self.norm.select(cut_tag_id.format(cut_idx), "any")

        if step_ > 0:
            previous = self.mps_boundaries.get(
                self._previous_boundary_key(side, step_, cut_tag_id)
            )

        if step_ == 0 and self.flat:
            return tn_slice

        if step_ == 0:
            tn = tn_slice
        else:
            if previous is None:
                raise ValueError("Missing previous boundary MPS during fitting.")
            tn = tn_slice | previous

        previous = self._fit_boundary(
            tn,
            boundary_key,
            previous if step_ > 0 else None,
            site_tag_id=site_tag_id,
            axis_len=axis_len,
        )

        return previous


    @staticmethod
    def _direction_sides(direction):
        """Return ``(move_left, move_right)`` from direction selector."""
        return "left" in direction, "right" in direction

    def _build_final_boundary_network(self, spec, p_previous_l, p_previous_r):
        """Build final TN by combining left/right fitted boundaries."""
        if p_previous_r is None:
            raise ValueError("Boundary contraction failed: missing right boundary MPS.")
        if self.max_separation == 0:
            return p_previous_r if p_previous_l is None else (p_previous_r | p_previous_l)

        center = self.norm.select(spec.cut_tag_id.format(spec.left_index), "any")
        if p_previous_l is None:
            return p_previous_r | center
        return p_previous_r | center | p_previous_l

    @staticmethod
    def _network_exponent(tn):
        """Return the real base-10 exponent carried by ``tn``."""
        return complex(getattr(tn, "exponent", 0.0)).real

    def run(
        self,
        *,
        write_back=True,
        max_separation=0,
        mps_boundaries=None,
        retag=True,
        visualize=False,
        flat=False,
        track_boundary_fidelity=False,
        progress=False,
        n_iter=10,
        equalize_norms=True,
        direction="y",
        strip_exponent=False,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run two-sided boundary sweeps and contract the final network.

        Parameters
        ----------
        write_back : bool, default=True
            Write fitted boundary MPS values back to ``self.mps_boundaries``.
        max_separation : int, default=0
            Separation mode (currently ``0`` or ``1``).
        mps_boundaries : dict | None, default=None
            Optional replacement boundary dictionary for this call.
        retag : bool, default=True
            Forwarded to fit backend.
        visualize : bool, default=False
            Enable intermediate tensor-network drawings.
        flat : bool, default=False
            Skip first-step fitting and use raw slice directly.
        track_boundary_fidelity : bool, default=False
            If ``True``, compute and store per-step fidelity values in
            ``self.fidel``.
        progress : bool, default=False
            Show progress bar.
        n_iter : int, default=10
            Number of local fit iterations for each step.
        equalize_norms : bool, default=True
            Forwarded normalization option for fitted MPS tensors.
        direction : str, default="y"
            Sweep selector.
        strip_exponent : bool, default=False
            If ``True``, return ``(mantissa, exponent)`` from the final
            contraction instead of reconstructing the scalar.

        Returns
        -------
        complex | float | tuple[complex | float, float]
            Final contracted scalar, or ``(mantissa, exponent)`` when
            ``strip_exponent=True``.
        """
        # Fidelity history is run-local and resets for each run() call.
        self._reset_fidelity_history()
        self._reset_fit_diagnostics()
        self.max_separation = max_separation
        self._update_separation()
        self._apply_runtime_overrides(
            mps_boundaries=mps_boundaries,
            retag=retag,
            visualize=visualize,
            flat=flat,
            track_boundary_fidelity=track_boundary_fidelity,
            progress=progress,
            n_iter=n_iter,
            equalize_norms=equalize_norms,
            write_back=write_back,
        )

        self.direction = direction
        spec = self._run_direction_spec(direction)
        total_updates = (
            self._effective_fit_steps(spec.left_steps)
            + self._effective_fit_steps(spec.right_steps)
        )
        with tqdm(
            total=total_updates,
            desc="bdy_dmrg:",
            leave=True,
            position=0,
            colour="CYAN",
            disable=not self.progress,
        ) as progress_bar:
            p_previous_l = self._fit_one_side(
                "left",
                spec.left_steps,
                progress_bar,
                spec.cut_tag_id,
                spec.site_tag_id,
            )
            p_previous_r = self._fit_one_side(
                "right",
                spec.right_steps,
                progress_bar,
                spec.cut_tag_id,
                spec.site_tag_id,
            )

        tn_f = self._build_final_boundary_network(spec, p_previous_l, p_previous_r)
        main, exp = tn_f.contract(all, optimize=self.contraction_opt, strip_exponent=True)
        exp = exp + self._network_exponent(self.norm)
        if strip_exponent:
            return main, exp
        return main * 10**exp

    def move_bdy(
        self,
        *,
        mps_boundaries=None,
        retag=True,
        visualize=False,
        flat=False,
        track_boundary_fidelity=False,
        progress=False,
        n_iter=10,
        equalize_norms=False,
        direction="y_left",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Sweep one or both sides and write updated boundary MPS tensors.

        Parameters mirror :meth:`run`, except no final scalar contraction is
        performed. This method is useful for environment preconditioning.
        """
        self._reset_fidelity_history()
        self._reset_fit_diagnostics()
        self._apply_runtime_overrides(
            mps_boundaries=mps_boundaries,
            retag=retag,
            visualize=visualize,
            flat=flat,
            track_boundary_fidelity=track_boundary_fidelity,
            progress=progress,
            n_iter=n_iter,
            equalize_norms=equalize_norms,
            write_back=True,
        )

        self.direction = direction
        cut_tag_id, site_tag_id, n_steps = self._direction_tags(direction)
        move_left, move_right = self._direction_sides(direction)
        if not (move_left or move_right):
            raise ValueError(f"direction must include 'left' or 'right', got: {direction}")
        total_updates = (
            (self._effective_fit_steps(n_steps) if move_left else 0)
            + (self._effective_fit_steps(n_steps) if move_right else 0)
        )

        with tqdm(
            total=total_updates,
            desc="move:",
            leave=True,
            position=0,
            colour="CYAN",
            disable=not self.progress,
        ) as progress_bar:
            if move_left:
                self._fit_one_side(
                    "left",
                    n_steps,
                    progress_bar,
                    cut_tag_id,
                    site_tag_id,
                )
            if move_right:
                self._fit_one_side(
                    "right",
                    n_steps,
                    progress_bar,
                    cut_tag_id,
                    site_tag_id,
                )

    def move_step_bdy(
        self,
        *,
        pos=0,
        mps_boundaries=None,
        retag=True,
        visualize=False,
        flat=False,
        track_boundary_fidelity=False,
        progress=False,
        n_iter=10,
        equalize_norms=False,
        direction="y_left",
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Fit and update a single boundary step at ``pos``.

        The update is applied to the side(s) encoded in ``direction``:
        ``*_left``, ``*_right``, or both.
        """
        self._reset_fidelity_history()
        self._reset_fit_diagnostics()
        self._apply_runtime_overrides(
            mps_boundaries=mps_boundaries,
            retag=retag,
            visualize=visualize,
            flat=flat,
            track_boundary_fidelity=track_boundary_fidelity,
            progress=progress,
            n_iter=n_iter,
            equalize_norms=equalize_norms,
            write_back=True,
        )

        self.direction = direction
        cut_tag_id, site_tag_id, n_steps = self._direction_tags(direction)
        if not isinstance(pos, int):
            raise TypeError("pos must be an integer")
        if pos < 0 or pos >= n_steps:
            raise ValueError(f"pos must be in [0, {n_steps - 1}] for direction={direction}")

        move_left, move_right = self._direction_sides(direction)
        if not (move_left or move_right):
            raise ValueError(f"direction must include 'left' or 'right', got: {direction}")

        sides = []
        if move_left:
            sides.append("left")
        if move_right:
            sides.append("right")

        step_does_fit = not self._is_skipped_flat_step(pos)
        total_updates = len(sides) if step_does_fit else 0

        with tqdm(
            total=total_updates,
            desc="move_step:",
            leave=True,
            position=0,
            colour="CYAN",
            disable=not self.progress,
        ) as progress_bar:
            for side in sides:
                updated = self._fit_one_step(
                    side,
                    pos,
                    cut_tag_id=cut_tag_id,
                    site_tag_id=site_tag_id,
                )

                if progress_bar is not None and step_does_fit:
                    postfix = {"pos": int(pos)}
                    if hasattr(updated, "max_bond"):
                        current_bond = updated.max_bond()
                        postfix["chi"] = (
                            None if current_bond is None else int(current_bond)
                        )
                    if self.track_boundary_fidelity and self.fidel:
                        postfix["F"] = complex(self.fidel[-1]).real
                    if postfix:
                        progress_bar.set_postfix(postfix)
                        progress_bar.refresh()
                    progress_bar.update(1)
