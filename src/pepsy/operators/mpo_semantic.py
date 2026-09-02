"""Semantic finite-chain MPOs for higher-order operator construction.

This module is the first layer above ordinary Quimb MPO tensors needed by the
higher-order exponential construction of Van Damme et al. It deliberately
keeps the virtual-level history separate from the tensor data.  Ordinary
Quimb MPOs remain the compiled interchange format, while :class:`FirstDegreeMPO`
retains enough structure for exact algebra and history compression.

The implementation is finite-chain and exact at this stage.  The extensive
Taylor construction is assembled from local MPO blocks and virtual channels;
it never forms a global operator matrix. Numerical bond truncation remains an
explicit separate layer, while optional Abelian metadata compiles native
Symmray blocks at the Quimb boundary.

Design contract
---------------
``FirstDegreeMPO`` is the semantic construction object.  Its virtual-bond
histories are part of the data model because the paper's Algorithms 1--4 act
on those histories, not just on the numerical MPO entries.  ``to_mpo()`` is
the compatibility boundary: it produces an ordinary Quimb MPO for existing
contraction and MPS-application code, while retaining a copy of the semantic
object on the compiled MPO.

The exact paths only use local tensor operations and exact equality checks.
``mode="folded"`` is deliberately separate because Algorithm 4 changes the
analytical history representation even though it does not use an SVD cutoff.
This module targets ordinary NumPy/Autoray-compatible tensors and finite open
chains. Native bosonic Abelian Symmray output accepts NumPy local blocks;
graded fermionic histories remain a future sign-preserving backend layer.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from math import factorial
from numbers import Integral
import re
import time
import warnings

import autoray as ar
import numpy as np

from .._internal.validation import is_strict_integer, normalize_integer_tuple
from .mpo_automaton import (
    MPOAutomaton,
    _as_backend,
    _backend_name,
    _backend_reference,  # noqa: F401 - legacy helper imported by mpo_cluster
    _matmul,
    _multiply_scalar,
)
from .mpo_block_plan import (
    MPOBlock,
    MPOBlockPlan,
    MPOChargeValidationReport,
)
from ._mpo_sparse import (
    SparseVirtualTensor,
    normalize_charge,
    symmray_arrays_from_sparse,
)
from .diagnostics import OperatorReportInfo
from .mpo_space import MPOBraiding, MPOPhysicalSpace

__all__ = [
    "MPOParameter",
    "MPOLevelToken",
    "MPOLevel",
    "MPOProductTerm",
    "MPOLocalOperatorTerm",
    "MPOBraiding",
    "MPOPhysicalSpace",
    "MPOCompressionReport",
    "MPONumericalCompressionReport",
    "MPODifferentiableCompressionReport",
    "MPOAdaptiveCompressionReport",
    "MPOBlock",
    "MPOBlockPlan",
    "MPOChargeValidationReport",
    "FirstDegreeMPO",
]


# Dense virtual transfer maps are much cheaper than repeated backend scatter
# updates for the small-to-medium history bonds normally produced by Taylor
# orders two through five.  Keep a conservative escape hatch for very large
# bonds, where a dense ``new_bond x old_bond`` map would use more memory than
# the original gather/scatter implementation.
_MAX_HISTORY_TRANSFER_ELEMENTS = 4_000_000

# Keep the fused coefficient bank bounded.  Very long-range or high-rank
# bases can have many terms relative to their small sparse slot count; those
# cases retain the exact grouped scatter fallback rather than allocating a
# mostly-zero ``num_terms x left x right x phys x phys`` bank.
_MAX_FUSED_SLOT_BANK_ELEMENTS = 4_000_000

# ``auto`` is intentionally a conservative work-budget policy: Algorithm 3 is
# valuable at low order, while its selected next-order replay can dominate
# high-order builds. Keep the budget explicit so the policy is deterministic
# and easy to tune independently of Taylor order.
# ``mode="auto"`` uses a structural Algorithm-3 work budget rather than a
# Taylor-order threshold.  The budget is deliberately expressed in selected
# extension terms: it is backend independent and can be evaluated before the
# numerical left/right pair plan is materialized.
_DEFAULT_AUTO_EXTENSION_BUDGET = 1_024


class _MPOProgress:
    """Optional progress display for higher-order MPO construction."""

    _STAGE_COLORS = {
        "history": "cyan",
        "a3": "yellow",
        "a1": "yellow",
        "a2": "yellow",
        "a4": "red",
        "boundary": "green",
        "chi": "magenta",
    }

    def __init__(self, enabled, *, order=None, chi=None):
        if not isinstance(enabled, bool):
            raise TypeError("progress must be a boolean.")
        self.enabled = enabled
        self.order = order
        self.chi = chi
        self.timings = {}
        self._bar = None
        self._stage = None
        self._stage_start = None
        self._stage_count = 0
        self._last_refresh = None
        self._detail_interval = 0.2
        self._construction_start = None
        self.total_seconds = None
        self._title = "exp" if order is None else f"exp(order={order})"
        if enabled:
            from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

            self._bar = tqdm(
                total=1,
                desc=self._title,
                unit="stage",
                leave=True,
                dynamic_ncols=True,
                colour="cyan",
            )

    @classmethod
    def _stage_color(cls, label):
        """Return a stable color for a construction stage category."""
        lowered = str(label).lower()
        for marker, color in cls._STAGE_COLORS.items():
            if marker in lowered:
                return color
        return "blue"

    def _set_description(self, label, *, refresh=False):
        """Set the stage description and progress-bar color."""
        color = self._stage_color(label)
        # ``colour`` is supported by the tqdm versions Pepsy depends on, but
        # assigning it separately also keeps this compatible with lightweight
        # test doubles and older tqdm adapters.
        self._bar.colour = color
        self._bar.set_description(f"{self._title} | {label}", refresh=False)
        if refresh:
            self._bar.refresh()

    @staticmethod
    def _timing_key(label):
        return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")

    @staticmethod
    def _bond_details(bond_dimensions):
        if bond_dimensions is None:
            return {}
        dimensions = tuple(int(value) for value in bond_dimensions)
        return {
            "maxchi": max(dimensions, default=1),
            "nbonds": len(dimensions),
        }

    def start(self, label):
        """Start a displayed and timed construction stage."""
        if not self.enabled:
            return
        if self._stage is not None:
            raise RuntimeError(f"progress stage {self._stage!r} is still active.")
        self._stage = str(label)
        self._stage_start = time.perf_counter()
        if self._construction_start is None:
            self._construction_start = self._stage_start
        self._last_refresh = self._stage_start
        self._stage_count += 1
        self._bar.total = self._stage_count
        self._set_description(self._stage, refresh=True)

    def detail(self, label=None, *, bond_dimensions=None, **details):
        """Refresh the display while a stage is still running."""
        if not self.enabled:
            return
        now = time.perf_counter()
        postfix = {
            "s": f"{now - self._stage_start:.2f}",
            **self._bond_details(bond_dimensions),
            **details,
        }
        if label is not None:
            # Keep the main description stable. Changing its width on every
            # site or bond makes some terminals redraw the bar on new lines.
            postfix["step"] = str(label)
        self._bar.set_postfix(postfix, refresh=False)
        if now - self._last_refresh >= self._detail_interval:
            self._bar.refresh()
            self._last_refresh = now

    def finish(self, label=None, *, bond_dimensions=None, **details):
        """Finish the current stage and record its elapsed time."""
        if not self.enabled:
            return None
        if self._stage is None:
            return None
        stage = self._stage if label is None else str(label)
        elapsed = time.perf_counter() - self._stage_start
        self.total_seconds = time.perf_counter() - self._construction_start
        self.timings[self._timing_key(stage)] = float(elapsed)
        postfix = {
            "s": f"{elapsed:.2f}",
            "order_s": f"{self.total_seconds:.2f}",
            **self._bond_details(bond_dimensions),
            **details,
        }
        self._set_description(stage)
        self._bar.set_postfix(postfix, refresh=False)
        self._bar.update(1)
        self._bar.refresh()
        self._stage = None
        self._stage_start = None
        self._last_refresh = None
        return elapsed

    def close(self):
        """Close the optional progress display."""
        if self._construction_start is not None and self.total_seconds is None:
            self.total_seconds = time.perf_counter() - self._construction_start
        if self._bar is not None:
            self._bar.close()


def _make_mpo_progress(progress, *, order=None, chi=None):
    """Normalize a public progress flag and report ownership."""
    if isinstance(progress, _MPOProgress):
        return progress, False
    return _MPOProgress(progress, order=order, chi=chi), True


@dataclass(frozen=True)
class MPOLevelToken:
    """One symbolic level in a virtual-state history.

    ``level`` follows the paper's first-degree convention: ``1`` and ``3``
    denote the two identity rails and ``2`` denotes an active operator
    channel.  ``payload`` distinguishes independent operator channels that
    happen to have the same level number.  The level number is intentionally
    small and symbolic; backend charge information belongs on ``MPOLevel``.
    """

    level: int
    payload: Hashable = None

    def __post_init__(self):
        if self.level not in (1, 2, 3):
            raise ValueError("MPO level tokens must have level 1, 2, or 3.")
        if self.payload is not None and not isinstance(self.payload, Hashable):
            raise TypeError("MPO level token payload must be hashable or None.")


@dataclass(frozen=True)
class MPOLevel:
    """A virtual state together with its symbolic history."""

    label: Hashable
    history: tuple[MPOLevelToken, ...]
    charge: object = None

    def __post_init__(self):
        if not isinstance(self.label, Hashable):
            raise TypeError("MPO level labels must be hashable.")
        object.__setattr__(self, "history", tuple(self.history))
        if not self.history:
            raise ValueError("MPO level history must contain at least one token.")
        if not all(isinstance(token, MPOLevelToken) for token in self.history):
            raise TypeError("MPO level history must contain MPOLevelToken values.")


@dataclass(frozen=True)
class MPOProductTerm:
    """A factorized local product term used to build a first-degree MPO.

    ``sites`` and ``operators`` describe only the non-identity factors.  A
    string operator can be supplied for fermion-compatible automaton routes,
    but the higher-order block-sparse backend is currently bosonic. With
    symmetry metadata configured on :class:`FirstDegreeMPO`, ``charge`` is the
    virtual channel charge after the first non-identity factor. For example,
    a U1 raising/lowering product has the negative of the raising operator's
    charge because the left MPO virtual leg is dualized.
    """

    sites: tuple[int, ...]
    operators: tuple[object, ...]
    coefficient: object = 1.0
    string_operators: tuple[object, ...] | None = None
    charge: object = None
    parities: tuple[int, ...] | None = None
    braiding: MPOBraiding | str | None = None

    @classmethod
    def from_pauli(
        cls,
        sites,
        paulis,
        *,
        coefficient=1.0,
        string_paulis=None,
        charge=None,
        parities=None,
        braiding=None,
    ):
        """Construct a product term from labels such as ``"ZXY"``.

        ``sites`` lists the non-identity support positions. Gaps receive
        identities unless ``string_paulis`` supplies one label per gap.
        """
        return cls(
            sites=sites,
            operators=paulis,
            coefficient=coefficient,
            string_operators=string_paulis,
            charge=charge,
            parities=parities,
            braiding=braiding,
        )

    def __post_init__(self):
        sites = normalize_integer_tuple(self.sites, name="sites", allow_scalar=False)
        operators = _normalize_operator_sequence(self.operators, name="operators")
        if not sites or len(sites) != len(operators):
            raise ValueError("sites and operators must be non-empty and aligned.")
        braiding = MPOBraiding.resolve(self.braiding)
        order, ordered_parities, phase = braiding.canonical_order(
            sites,
            self.parities,
        )
        ordered = tuple((sites[index], operators[index]) for index in order)
        if (
            (self.charge is not None or self.string_operators is not None)
            and tuple(site for site, _operator in ordered) != sites
        ):
            raise ValueError(
                "terms with charge or string_operators metadata must list "
                "sites in increasing order."
            )
        combined_sites = []
        combined_operators = []
        combined_parities = []
        for (site, operator), parity in zip(ordered, ordered_parities):
            if combined_sites and site == combined_sites[-1]:
                if self.charge is not None or self.string_operators is not None:
                    raise ValueError(
                        "terms with charge or string_operators metadata cannot "
                        "repeat a site."
                    )
                combined_operators[-1] = _matmul(combined_operators[-1], operator)
                combined_parities[-1] = (combined_parities[-1] + parity) % 2
            else:
                combined_sites.append(site)
                combined_operators.append(operator)
                combined_parities.append(parity)
        sites = tuple(combined_sites)
        operators = tuple(combined_operators)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "coefficient", _scale_term_coefficient(self.coefficient, phase))
        object.__setattr__(self, "parities", tuple(combined_parities))
        object.__setattr__(self, "braiding", braiding)
        if self.string_operators is not None:
            object.__setattr__(
                self,
                "string_operators",
                _normalize_operator_sequence(
                    self.string_operators,
                    name="string_operators",
                ),
            )


@dataclass(frozen=True)
class MPOLocalOperatorTerm:
    """A general dense operator acting on an ordered collection of sites.

    Unlike :class:`MPOProductTerm`, ``operator`` need not factor across its
    support. Pepsy performs an exact fixed-rank operator-Schmidt decomposition
    and inserts the resulting local MPO segment directly into the automaton.
    """

    sites: tuple[int, ...]
    operator: object
    coefficient: object = 1.0
    phys_dim: int | None = None
    charge: object = None

    def __post_init__(self):
        sites = normalize_integer_tuple(self.sites, name="sites", allow_scalar=False)
        if not sites:
            raise ValueError("sites must not be empty.")
        if len(set(sites)) != len(sites):
            raise ValueError("a general local operator cannot repeat a site.")
        if self.charge is not None and len(sites) > 1:
            raise ValueError(
                "charged general local operators require a sector-aware "
                "Schmidt decomposition and are not yet accepted."
            )

        operator = self.operator
        shape = tuple(getattr(operator, "shape", ()))
        if not shape:
            try:
                operator = np.asarray(operator)
                shape = tuple(operator.shape)
            except (TypeError, ValueError) as exc:
                raise TypeError("operator must be an array-like square matrix.") from exc
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("operator must be a square matrix.")

        if self.phys_dim is None:
            dimension = int(shape[0])
            phys_dim = int(round(dimension ** (1.0 / len(sites))))
            if phys_dim ** len(sites) != dimension:
                raise ValueError(
                    f"operator dimension {dimension} is not d**{len(sites)} for "
                    "an integer local physical dimension d."
                )
        else:
            if not is_strict_integer(self.phys_dim) or int(self.phys_dim) < 1:
                raise TypeError("phys_dim must be a positive integer.")
            phys_dim = int(self.phys_dim)
            expected = phys_dim ** len(sites)
            if shape != (expected, expected):
                raise ValueError(
                    f"operator has shape {shape}, expected ({expected}, {expected})."
                )

        order = tuple(sorted(range(len(sites)), key=sites.__getitem__))
        if order != tuple(range(len(sites))):
            if self.charge is not None:
                raise ValueError(
                    "charged general operators must list sites in increasing order."
                )
            tensor = ar.do(
                "reshape",
                operator,
                (phys_dim,) * (2 * len(sites)),
            )
            axes = order + tuple(len(sites) + index for index in order)
            operator = ar.do(
                "reshape",
                ar.do("transpose", tensor, axes),
                shape,
            )
            sites = tuple(sites[index] for index in order)

        _check_scalar(self.coefficient, name="MPO coefficient")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "phys_dim", phys_dim)


@dataclass(frozen=True)
class MPOCompressionReport:
    """Diagnostics returned by exact or analytical history compression.

    ``exact`` distinguishes scalar gauge eliminations from Algorithm 4's
    order-controlled analytical approximation.  ``merges`` contains stable,
    human-readable provenance records rather than backend tensor objects, so
    reports can be logged or serialized by callers.  The report describes the
    semantic history stage; it does not describe a later Quimb SVD/truncation.
    """

    method: str
    exact: bool
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    merged_channels: int
    merges: tuple[Mapping[str, object], ...] = ()
    skipped_candidates: int = 0

    @property
    def api_info(self):
        """Return the stable cross-family report summary."""
        return OperatorReportInfo(
            family="mpo",
            algorithm="history_compression",
            representation="semantic_mpo",
            truncated=not self.exact,
        )


@dataclass(frozen=True)
class MPONumericalCompressionReport:
    """Report the numerical compression boundary delegated to Quimb.

    The semantic history representation cannot survive a numerical bond
    truncation, so this report deliberately describes the compiled Quimb MPO
    only.  When requested, the operator error is measured as a tensor-network
    Frobenius norm of the difference between the pre- and post-compression
    MPOs.  This avoids forming a global dense matrix, but remains an optional
    contraction because it can be expensive for a large MPO.
    """

    method: str
    form: object
    max_bond: int | None
    cutoff: float
    cutoff_mode: str
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    truncated: bool
    truncation_error: object = None
    operator_frobenius_error: object = None
    operator_frobenius_relative_error: object = None
    error_estimator: str | None = None
    sector_aware: bool = False
    initial_sector_dimensions: tuple = ()
    final_sector_dimensions: tuple = ()
    initial_sector_block_counts: tuple[int, ...] = ()
    final_sector_block_counts: tuple[int, ...] = ()

    @property
    def api_info(self):
        """Return the stable cross-family report summary."""
        return OperatorReportInfo(
            family="mpo",
            algorithm="numerical_compression",
            representation="materialized_mpo",
            truncated=self.truncated,
        )


@dataclass(frozen=True)
class MPODifferentiableCompressionReport:
    """Report a fixed-rank, autodiff-friendly MPO compression.

    Unlike cutoff-based compression, fixed-rank TT-SVD never changes its
    selected rank as a function of singular values.  The resulting map is
    therefore differentiable through the backend SVD away from the usual
    repeated-singular-value singularities.
    """

    method: str
    max_bond: int
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    truncated: bool
    differentiable: bool = True

    @property
    def api_info(self):
        """Return the stable cross-family report summary."""
        return OperatorReportInfo(
            family="mpo",
            algorithm="fixed_rank_compression",
            representation="semantic_mpo",
            truncated=self.truncated,
            differentiable=self.differentiable,
        )


@dataclass(frozen=True)
class MPOAdaptiveCompressionReport:
    """Report a cutoff-aware backend TT-SVD compression.

    This path is intended for bounded intermediate assembly, where the
    semantic MPO must remain available for another addition. Unlike the
    fixed-rank path, ranks depend on the singular values, so rank selection is
    a discrete control decision and cannot be safely staged inside a compiled
    trace.
    """

    method: str
    form: str
    max_bond: int
    cutoff: float | None
    cutoff_mode: str
    initial_bond_dimensions: tuple[int, ...]
    final_bond_dimensions: tuple[int, ...]
    truncated: bool
    discarded_weights: tuple[float, ...] = ()
    differentiable: bool = False

    @property
    def api_info(self):
        """Return the stable cross-family report summary."""
        return OperatorReportInfo(
            family="mpo",
            algorithm="adaptive_compression",
            representation="semantic_mpo",
            truncated=self.truncated,
            differentiable=self.differentiable,
        )


def _check_scalar(value, *, name):
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        ndim = np.ndim(value)
    if ndim != 0:
        raise TypeError(f"{name} must be scalar, got ndim={ndim}.")


def _resolve_compression_cutoff(cutoff, reference):
    """Resolve a numeric or dtype-aware numerical compression cutoff."""
    if isinstance(cutoff, str):
        if cutoff.strip().lower() != "auto":
            raise ValueError("cutoff must be 'auto' or a non-negative number.")
        dtype = getattr(reference, "dtype", None)
        dtype_name = str(dtype).lower()
        if "float16" in dtype_name or "bfloat16" in dtype_name:
            return 1.0e-3
        if "float32" in dtype_name or "complex64" in dtype_name:
            return 1.0e-6
        return 1.0e-12
    _check_scalar(cutoff, name="cutoff")
    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be 'auto' or a non-negative number.") from exc
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be 'auto' or a non-negative number.")
    return cutoff


def _resolve_compression_cutoff_mode(cutoff_mode):
    """Resolve ``cutoff_mode='auto'`` to Pepsy's default convention."""
    if not isinstance(cutoff_mode, str):
        raise TypeError("cutoff_mode must be a string.")
    if cutoff_mode.strip().lower() == "auto":
        return "rsum2"
    return cutoff_mode


def _normalize_sector_aware_request(value):
    """Normalize the explicit native sector-compression policy."""

    if value is None:
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    raise TypeError("sector_aware must be True, False, or 'auto'.")


def _native_sector_summary(mpo):
    """Inspect native Symmray sectors without converting them to dense arrays."""

    tensors = getattr(mpo, "tensors", ())
    if not tensors:
        return None
    data_values = [getattr(tensor, "data", None) for tensor in tensors]
    if not all(
        hasattr(data, "blocks") and hasattr(data, "indices")
        for data in data_values
    ):
        return None

    def normalize_sector_charge(charge):
        if isinstance(charge, tuple):
            return tuple(normalize_sector_charge(value) for value in charge)
        if isinstance(charge, Integral):
            return int(charge)
        return charge

    bond_sector_dimensions = []
    for bond in range(1, int(mpo.L)):
        data = data_values[bond - 1]
        virtual_position = 0 if bond - 1 == 0 else 1
        try:
            index = data.indices[virtual_position]
            chargemap = index.chargemap
        except (AttributeError, IndexError, TypeError):
            return None
        if not isinstance(chargemap, Mapping):
            return None
        bond_sector_dimensions.append(tuple(
            (normalize_sector_charge(charge), int(dimension))
            for charge, dimension in chargemap.items()
        ))

    site_block_counts = tuple(
        len(data.blocks) if isinstance(data.blocks, Mapping) else None
        for data in data_values
    )
    if any(count is None for count in site_block_counts):
        return None
    return {
        "native": True,
        "bond_sector_dimensions": tuple(bond_sector_dimensions),
        "site_block_counts": tuple(int(count) for count in site_block_counts),
        "total_blocks": sum(site_block_counts),
    }


def _resolve_sector_aware(value, native_summary):
    """Resolve ``sector_aware`` against the materialized tensor boundary."""

    requested = _normalize_sector_aware_request(value)
    if requested == "auto":
        return native_summary is not None
    if requested and native_summary is None:
        raise ValueError(
            "sector_aware=True requires native Symmray MPO tensors; "
            "dense MPOs use the separate ordinary Quimb compression path."
        )
    return requested


def _normalize_exp_compress_opts(
    compress_opts,
    *,
    form=None,
    create_bond=False,
):
    """Normalize final numerical-compression options for exponential APIs."""
    if compress_opts is None:
        options = {}
    elif not isinstance(compress_opts, Mapping):
        raise TypeError("compress_opts must be a mapping or None.")
    else:
        options = dict(compress_opts)

    reserved = {
        "chi",
        "max_bond",
        "cutoff",
        "cutoff_mode",
        "return_report",
        "compression",
        "differentiable",
        "sector_aware",
    }
    duplicated = sorted(reserved.intersection(options))
    if duplicated:
        names = ", ".join(duplicated)
        raise TypeError(
            f"{names} must be supplied as explicit exponential compression "
            "arguments, not in compress_opts."
        )

    if form is not None:
        if "form" in options and options["form"] != form:
            raise TypeError("conflicting form values supplied.")
        options["form"] = form
    if create_bond:
        if "create_bond" in options and options["create_bond"] is not True:
            raise TypeError("conflicting create_bond values supplied.")
        options["create_bond"] = True
    return options


_SUPPORTED_MPO_SYMMETRIES = frozenset({"U1", "Z2", "U1U1", "Z2Z2"})


def _normalize_mpo_symmetry(value):
    """Normalize a public MPO symmetry name to Symmray's spelling."""
    name = str(value).upper().replace("-", "")
    if name not in _SUPPORTED_MPO_SYMMETRIES:
        allowed = ", ".join(sorted(_SUPPORTED_MPO_SYMMETRIES))
        raise ValueError(
            f"block-sparse MPO symmetry must be one of {allowed}; got {value!r}."
        )
    return name


def _normalize_mpo_physical_charges(charges, phys_dim, symmetry):
    """Expand and normalize per-state or per-sector physical charges."""
    if isinstance(charges, Mapping):
        expanded = []
        for charge, multiplicity in charges.items():
            if (
                isinstance(multiplicity, bool)
                or not isinstance(multiplicity, Integral)
                or int(multiplicity) < 1
            ):
                raise ValueError(
                    "physical charge-sector multiplicities must be positive "
                    f"integers, got {multiplicity!r} for charge {charge!r}."
                )
            expanded.extend([charge] * int(multiplicity))
        charges = tuple(expanded)
    else:
        try:
            charges = tuple(charges)
        except TypeError as exc:
            raise TypeError(
                "physical_charges must be a sequence or a mapping from "
                "charge to positive sector multiplicity."
            ) from exc

    if len(charges) != phys_dim:
        raise ValueError(
            "physical_charges must contain one charge per local basis state "
            f"(or sector multiplicities summing to {phys_dim}), got {len(charges)}."
        )
    return tuple(normalize_charge(charge, symmetry) for charge in charges)


def _is_integral_value(value):
    """Compatibility spelling for strict integer coordinate validation."""
    return is_strict_integer(value)


def _resolve_exp_step(step, dt):
    """Resolve the canonical ``step`` and legacy ``dt`` spellings.

    Public exponential methods use ``step`` in their documentation.  The
    optional ``dt`` keyword is accepted only as a compatibility spelling so
    existing callers can migrate without changing numerical semantics.
    """
    if step is not None and dt is not None:
        raise TypeError("pass either step or dt, not both.")
    if step is None:
        if dt is None:
            raise TypeError("exp requires a scalar step.")
        step = dt
    _check_scalar(step, name="step")
    return step


def _fixed_rank_svd(matrix):
    """Run the configured thin SVD, enabling the safe Torch VJP when needed."""
    if _backend_name(matrix) == "jax":
        # Pepsy's custom JAX SVD registration intentionally exposes only the
        # matrix positional argument, whereas native JAX accepts
        # ``full_matrices``.  Both policies return a thin decomposition; trim
        # defensively so a third-party Autoray registration returning a full
        # decomposition cannot change the MPO structural rank.
        left, singular_values, right = ar.do("linalg.svd", matrix)
        rank = min(int(matrix.shape[-2]), int(matrix.shape[-1]))
        return (
            left[..., :rank],
            singular_values[..., :rank],
            right[..., :rank, :],
        )
    if _backend_name(matrix) != "torch":
        return ar.do("linalg.svd", matrix, full_matrices=False)

    # Native Torch's singular-vector VJP is undefined at repeated or zero
    # singular values, which are common in MPO blocks. Reuse Pepsy's one
    # public Torch policy for this path and scope the registration so an MPO
    # compression does not silently change the caller's global backend mode.
    from pepsy.backends import (  # pylint: disable=import-outside-toplevel
        TorchLinalgConfig,
        get_torch_linalg_config,
    )

    mode = "complex" if getattr(matrix.dtype, "is_complex", False) else "real"
    current = get_torch_linalg_config()
    if current is not None and current.stabilized and current.mode == mode:
        return ar.do("linalg.svd", matrix)
    config = (
        replace(current, mode=mode, stabilized=True)
        if current is not None
        else TorchLinalgConfig(stabilized=True, mode=mode)
    )
    with config.activated():
        return ar.do("linalg.svd", matrix)


_TT_SVD_CUTOFF_MODES = frozenset(
    {"rel", "abs", "sum1", "sum2", "rsum1", "rsum2"}
)


def _normalize_tt_svd_form(form):
    """Normalize the directional form used by semantic TT-SVD."""
    if form is None:
        return "left"
    if not isinstance(form, str):
        raise TypeError("form must be 'left' or 'right'.")
    form = form.strip().lower().replace("-", "_")
    if form not in {"left", "right"}:
        raise ValueError("form must be 'left' or 'right'.")
    return form


def _normalize_tt_svd_cutoff_mode(cutoff_mode):
    """Validate the cutoff convention shared with Quimb's SVD split."""
    cutoff_mode = _resolve_compression_cutoff_mode(cutoff_mode)
    if cutoff_mode not in _TT_SVD_CUTOFF_MODES:
        allowed = ", ".join(sorted(_TT_SVD_CUTOFF_MODES))
        raise ValueError(
            f"cutoff_mode must be one of {allowed} or 'auto'; got "
            f"{cutoff_mode!r}."
        )
    return cutoff_mode


def _host_singular_values(singular_values):
    """Read a singular spectrum for rank selection without moving tensors."""
    values = singular_values
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif _backend_name(values) == "cupy":
        values = values.get()
    try:
        return np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "adaptive semantic TT-SVD requires concrete singular values; "
            "use fixed-rank compression inside a JAX or other compiled trace."
        ) from exc


def _select_tt_svd_rank(singular_values, cutoff, cutoff_mode):
    """Select a cutoff rank using the same conventions as Quimb."""
    values = _host_singular_values(singular_values)
    size = int(values.shape[0])
    if cutoff is None or cutoff == 0.0 or size <= 1:
        return size
    values = np.abs(values.astype(float, copy=False))
    if cutoff_mode == "rel":
        return max(1, int(np.count_nonzero(values > cutoff * values[0])))
    if cutoff_mode == "abs":
        return max(1, int(np.count_nonzero(values > cutoff)))

    squared = values * values
    linear = values
    if cutoff_mode in {"sum2", "rsum2"}:
        tail = np.concatenate((np.cumsum(squared[::-1])[::-1], [0.0]))
        threshold = float(cutoff)
        if cutoff_mode == "rsum2":
            threshold *= float(np.sum(squared))
    else:
        tail = np.concatenate((np.cumsum(linear[::-1])[::-1], [0.0]))
        threshold = float(cutoff)
        if cutoff_mode == "rsum1":
            threshold *= float(np.sum(linear))

    for rank in range(1, size + 1):
        if tail[rank] <= threshold:
            return rank
    return size


def _tt_svd_arrays(
    source_arrays,
    max_bond,
    *,
    cutoff=None,
    cutoff_mode="rel",
    form="left",
    adaptive=False,
):
    """Compress dense MPO cores with a directional TT-SVD sweep."""
    source_arrays = tuple(source_arrays)
    length = len(source_arrays)
    if length == 1:
        return source_arrays, (), ()
    form = _normalize_tt_svd_form(form)
    arrays = [None] * length
    discarded_weights = []

    if form == "left":
        carry = None
        for site, array in enumerate(source_arrays[:-1]):
            combined = array if carry is None else ar.do(
                "tensordot", carry, array, axes=([1], [0])
            )
            left_dim, right_dim, phys_up, phys_down = combined.shape
            matrix_data = ar.do(
                "transpose", combined, (0, 2, 3, 1)
            )
            matrix = ar.do(
                "reshape",
                matrix_data,
                (int(left_dim) * int(phys_up) * int(phys_down), int(right_dim)),
            )
            u, singular_values, vh = _fixed_rank_svd(matrix)
            rank = min(max_bond, int(singular_values.shape[0]))
            if adaptive:
                rank = min(
                    rank,
                    _select_tt_svd_rank(
                        singular_values,
                        cutoff,
                        cutoff_mode,
                    ),
                )
                discarded_weights.append(
                    float(
                        np.linalg.norm(
                            _host_singular_values(singular_values)[rank:]
                        )
                    )
                )
            u = u[:, :rank]
            singular_values = singular_values[:rank]
            vh = vh[:rank, :]
            local = ar.do(
                "reshape",
                u,
                (int(left_dim), int(phys_up), int(phys_down), rank),
            )
            arrays[site] = ar.do("transpose", local, (0, 3, 1, 2))
            carry = ar.do(
                "multiply",
                ar.do("reshape", singular_values, (rank, 1)),
                vh,
            )
        arrays[-1] = ar.do(
            "tensordot", carry, source_arrays[-1], axes=([1], [0])
        )
    else:
        carry = None
        for site in range(length - 1, 0, -1):
            combined = source_arrays[site] if carry is None else ar.do(
                "transpose",
                ar.do(
                    "tensordot",
                    source_arrays[site],
                    carry,
                    axes=([1], [0]),
                ),
                (0, 3, 1, 2),
            )
            left_dim, right_dim, phys_up, phys_down = combined.shape
            matrix_data = ar.do(
                "transpose", combined, (0, 2, 3, 1)
            )
            matrix = ar.do(
                "reshape",
                matrix_data,
                (int(left_dim), int(phys_up) * int(phys_down) * int(right_dim)),
            )
            u, singular_values, vh = _fixed_rank_svd(matrix)
            rank = min(max_bond, int(singular_values.shape[0]))
            if adaptive:
                rank = min(
                    rank,
                    _select_tt_svd_rank(
                        singular_values,
                        cutoff,
                        cutoff_mode,
                    ),
                )
                discarded_weights.append(
                    float(
                        np.linalg.norm(
                            _host_singular_values(singular_values)[rank:]
                        )
                    )
                )
            u = u[:, :rank]
            singular_values = singular_values[:rank]
            vh = vh[:rank, :]
            right_factor = ar.do(
                "multiply",
                ar.do("reshape", singular_values, (rank, 1)),
                vh,
            )
            right_factor = ar.do(
                "reshape",
                right_factor,
                (rank, int(phys_up), int(phys_down), int(right_dim)),
            )
            arrays[site] = ar.do(
                "transpose", right_factor, (0, 3, 1, 2)
            )
            carry = u
        first = ar.do(
            "tensordot", source_arrays[0], carry, axes=([1], [0])
        )
        arrays[0] = ar.do("transpose", first, (0, 3, 1, 2))

    return tuple(arrays), tuple(
        int(array.shape[1]) for array in arrays[:-1]
    ), tuple(discarded_weights)


_UNSET = object()


@dataclass(frozen=True)
class MPOParameter:
    """A named or positional scalar coefficient resolved by :class:`MPOBasis`.

    The object is intentionally only a reference to a coefficient.  The
    current backend scalar is supplied to :meth:`MPOBasis.build`, which means
    Torch/JAX values are never copied into a topology cache and remain in the
    autodiff graph.

    Parameters
    ----------
    name : hashable
        Mapping key, or integer positional index for sequence-like parameter
        containers.
    default : scalar, optional
        Value used when ``build`` receives no parameter container.  Omitting
        it makes the parameter required.
    """

    name: Hashable
    default: object = _UNSET

    def __post_init__(self):
        if not isinstance(self.name, Hashable):
            raise TypeError("MPO parameter names must be hashable.")

    def resolve(self, parameters):
        """Resolve this reference against a mapping or positional container."""
        if parameters is None:
            if self.default is _UNSET:
                raise KeyError(
                    f"missing MPO parameter {self.name!r}; pass it to build()."
                )
            value = self.default
        elif isinstance(parameters, Mapping):
            try:
                value = parameters[self.name]
            except KeyError as exc:
                raise KeyError(
                    f"missing MPO parameter {self.name!r}; pass it to build()."
                ) from exc
        else:
            if not isinstance(self.name, Integral):
                raise TypeError(
                    "non-integer MPO parameter names require a mapping; got "
                    f"{type(parameters).__name__}."
                )
            try:
                value = parameters[int(self.name)]
            except (IndexError, KeyError, TypeError) as exc:
                raise KeyError(
                    f"missing positional MPO parameter {self.name!r}."
                ) from exc
        _check_scalar(value, name=f"parameter {self.name!r}")
        return value


def _scale_term_coefficient(coefficient, phase):
    """Apply a canonicalization phase without disconnecting parameter values."""

    if phase == 1:
        return coefficient
    if isinstance(coefficient, MPOParameter):
        return lambda parameters, reference=coefficient: _multiply_scalar(
            phase,
            reference.resolve(parameters),
        )
    if callable(coefficient):
        return lambda parameters, reference=coefficient: _multiply_scalar(
            phase,
            reference(parameters),
        )
    return _multiply_scalar(phase, coefficient)


def _as_4d(data, *, site, length):
    """Normalize a Quimb MPO tensor to ``(left, right, up, down)``."""
    shape = tuple(getattr(data, "shape", ()))
    if len(shape) == 4:
        out = data
    elif len(shape) == 3:
        if length == 1:
            raise ValueError("a one-site MPO tensor must have rank 2 or 4.")
        if site == 0:
            out = ar.do("reshape", data, (1, shape[0], shape[1], shape[2]))
        else:
            out = ar.do("reshape", data, (shape[0], 1, shape[1], shape[2]))
    elif len(shape) == 2 and length == 1:
        out = ar.do("reshape", data, (1, 1, shape[0], shape[1]))
    else:
        raise ValueError(
            "MPO tensors must have rank 4, rank 3 at an open boundary, "
            "or rank 2 for a one-site MPO."
        )
    if out.shape[-1] != out.shape[-2]:
        raise ValueError("MPO physical output and input dimensions must match.")
    return out


def _sparse_virtual_to_dense(tensor):
    """Materialize one sparse virtual tensor on its local block backend."""
    if not tensor.blocks:
        reference = getattr(tensor, "_like", None)
        if reference is None:
            raise ValueError(
                "cannot materialize an empty sparse MPO tensor without a "
                "backend reference."
            )
        return _zeros(tensor.shape, like=reference)
    rows = np.fromiter((key[0] for key in tensor.blocks), dtype=int)
    columns = np.fromiter((key[1] for key in tensor.blocks), dtype=int)
    values = _stack(tuple(tensor.blocks.values()), axis=0)
    result = _zeros(tensor.shape, like=values)
    return _scatter_add_2d(result, rows, columns, values)


def _dense_virtual_to_sparse(array):
    """Convert a NumPy virtual tensor to retained operator-valued blocks."""
    if _backend_name(array) not in {"builtins", "numpy"}:
        raise TypeError(
            "native block-sparse Symmray MPO compilation currently requires "
            "NumPy local tensors."
        )
    blocks = {
        (left, right): array[left, right]
        for left in range(array.shape[0])
        for right in range(array.shape[1])
        if np.any(array[left, right])
    }
    if not blocks:
        blocks[(0, 0)] = array[0, 0]
    return SparseVirtualTensor(array.shape, blocks)


def _native_mpo_dense_index_maps(mpo, semantic, groups, value):
    """Build physical-basis charge maps for a fused native MPO result.

    Symmray stores a fused index in contiguous sector order.  That order is
    useful for block operations, but it is not necessarily the caller's
    computational-basis order.  ``AbelianArray.to_dense(index_maps=...)``
    accepts the original basis-to-charge map and restores that order without
    expanding the MPO's virtual or sector blocks first.
    """
    if not hasattr(value, "to_dense") or not hasattr(value, "indices"):
        return None
    if semantic is not None:
        if semantic.symmetry is None:
            return None
        physical_charges = tuple(semantic.physical_charges)
        physical_dimension = int(semantic.phys_dim)
    else:
        if getattr(mpo, "pepsy_mpo_symmetry", None) is None:
            return None
        physical_charges = tuple(
            getattr(mpo, "pepsy_mpo_physical_charges", ())
        )
        physical_dimension = int(
            getattr(mpo, "pepsy_mpo_physical_dimension", 0)
        )
        if not physical_charges or physical_dimension < 1:
            return None
    symmetry = getattr(value, "symmetry", None)
    if symmetry is None or not hasattr(symmetry, "combine"):
        return None

    try:
        upper_by_ind = {
            mpo.upper_ind(site): int(site)
            for site in mpo.sites
        }
        lower_by_ind = {
            mpo.lower_ind(site): int(site)
            for site in mpo.sites
        }
    except (AttributeError, TypeError, ValueError):
        return None

    index_maps = []
    for axis, group in enumerate(groups):
        charges_by_site = []
        for ind in group:
            site = upper_by_ind.get(ind, lower_by_ind.get(ind))
            if site is None or site < 0:
                return None
            charges_by_site.append(physical_charges)

        index_map = tuple(
            symmetry.combine(*(
                charges_by_site[charge_position][state]
                for charge_position, state in enumerate(states)
            ))
            for states in product(
                range(physical_dimension),
                repeat=len(charges_by_site),
            )
        )
        if (
            axis >= len(value.indices)
            or len(index_map) != value.indices[axis].size_total
        ):
            return None
        index_maps.append(index_map)

    return tuple(index_maps)


class _PhysicalBasisDenseMPO:
    """Mixin restoring computational-basis order for native Symmray MPOs."""

    @staticmethod
    def _maybe_qarray(value, to_qarray):
        if to_qarray and ar.infer_backend(value) == "numpy":
            import quimb as qu  # pylint: disable=import-outside-toplevel

            return qu.qarray(value)
        return value

    def to_dense(self, *inds_seq, to_qarray=False, **contract_opts):
        """Convert an MPO to dense order, including native Symmray sectors."""
        value = super().to_dense(
            *inds_seq,
            to_qarray=False,
            **contract_opts,
        )
        semantic = getattr(self, "pepsy_first_degree", None)
        has_physical_metadata = (
            semantic is not None
            or getattr(self, "pepsy_mpo_symmetry", None) is not None
        )
        if not hasattr(value, "indices") or not has_physical_metadata:
            return self._maybe_qarray(value, to_qarray)

        groups = (
            tuple(tuple(group) for group in inds_seq)
            if inds_seq
            else (
                tuple(self.upper_inds_present),
                tuple(self.lower_inds_present),
            )
        )
        index_maps = _native_mpo_dense_index_maps(
            self,
            semantic,
            groups,
            value,
        )
        if index_maps is not None:
            value = value.to_dense(index_maps=index_maps)
        elif hasattr(value, "unfuse_all"):
            value = value.unfuse_all()
            if hasattr(value, "to_dense"):
                value = value.to_dense()

        return self._maybe_qarray(value, to_qarray)


@lru_cache(maxsize=1)
def _pepsy_mpo_class(base_class):
    """Return the cached Quimb MPO subclass used at Pepsy's boundary."""
    return type(
        "PepsyMatrixProductOperator",
        (_PhysicalBasisDenseMPO, base_class),
        {"__module__": __name__},
    )


def _ensure_pepsy_mpo_boundary(mpo):
    """Install Pepsy's dense-order boundary on a compatible MPO in-place.

    FIT and some Quimb compressors return a fresh base
    ``MatrixProductOperator`` even when their input was a Pepsy semantic MPO.
    The tensor data is still the desired result, but the base class does not
    know how to restore computational-basis order for fused Symmray physical
    indices.  Changing the Python class is a narrow, in-memory compatibility
    shim: it does not copy tensor data or alter the tensor network topology.
    """
    if isinstance(mpo, _PhysicalBasisDenseMPO):
        return mpo
    try:
        mpo.__class__ = _pepsy_mpo_class(type(mpo))
    except TypeError as exc:
        raise TypeError(
            "could not install Pepsy's MPO boundary wrapper on the "
            f"compression result of type {type(mpo).__name__}."
        ) from exc
    return mpo


def _pauli_matrix(label):
    """Return a dense one-qubit Pauli matrix for a single-character label."""
    label = str(label).upper()
    matrices = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    try:
        return matrices[label]
    except KeyError as exc:
        raise ValueError(
            f"Pauli labels must be one of 'I', 'X', 'Y', or 'Z', got {label!r}."
        ) from exc


def _normalize_operator_sequence(operators, *, name):
    """Normalize a matrix sequence or a compact Pauli label string."""
    if isinstance(operators, str):
        operators = tuple(operators)
    else:
        operators = tuple(operators)
    normalized = []
    for operator in operators:
        if isinstance(operator, str):
            if len(operator) != 1:
                raise ValueError(
                    f"{name} string entries must be single Pauli labels, "
                    f"got {operator!r}."
                )
            normalized.append(_pauli_matrix(operator))
        else:
            normalized.append(operator)
    return tuple(normalized)


def _as_square_operator(value):
    """Return a square matrix payload without coercing backend arrays."""

    if isinstance(value, str):
        return None
    shape = tuple(getattr(value, "shape", ()))
    if shape:
        return value if len(shape) == 2 and shape[0] == shape[1] else None
    array = _as_array_like_matrix(value)
    if array is None or array.shape[0] != array.shape[1]:
        return None
    return array


def _local_operator_mpo_cores(term):
    """Decompose one general local operator into exact fixed-rank MPO cores."""

    if not isinstance(term, MPOLocalOperatorTerm):
        raise TypeError("term must be an MPOLocalOperatorTerm.")
    nsites = len(term.sites)
    phys_dim = term.phys_dim
    if nsites == 1:
        return (ar.do("reshape", term.operator, (1, 1, phys_dim, phys_dim)),)

    tensor = ar.do(
        "reshape",
        term.operator,
        (phys_dim,) * (2 * nsites),
    )
    interleaved = tuple(
        axis
        for site in range(nsites)
        for axis in (site, nsites + site)
    )
    remainder = ar.do("transpose", tensor, interleaved)
    left_rank = 1
    cores = []
    local_size = phys_dim * phys_dim
    for site in range(nsites - 1):
        matrix = ar.do(
            "reshape",
            remainder,
            (left_rank * local_size, -1),
        )
        u, singular_values, vh = _fixed_rank_svd(matrix)
        rank = min(int(matrix.shape[0]), int(matrix.shape[1]))
        u = u[:, :rank]
        singular_values = singular_values[:rank]
        vh = vh[:rank]
        core = ar.do(
            "reshape",
            u,
            (left_rank, phys_dim, phys_dim, rank),
        )
        cores.append(ar.do("transpose", core, (0, 3, 1, 2)))
        remainder = ar.do(
            "multiply",
            singular_values[:, None],
            vh,
        )
        left_rank = rank
        remaining_sites = nsites - site - 1
        remainder = ar.do(
            "reshape",
            remainder,
            (left_rank,) + (local_size,) * remaining_sites,
        )

    last = ar.do(
        "reshape",
        remainder,
        (left_rank, phys_dim, phys_dim, 1),
    )
    cores.append(ar.do("transpose", last, (0, 3, 1, 2)))
    return tuple(cores)


def _zeros(shape, *, like):
    if like is None:
        raise ValueError("backend-native zeros require a reference array.")
    try:
        return ar.do("zeros", tuple(shape), like=like)
    except Exception as exc:  # pragma: no cover - backend compatibility guard
        backend = _backend_name(like)
        raise TypeError(
            f"cannot construct zeros for backend {backend!r}; register the "
            "backend with Autoray or provide a compatible to_backend "
            "converter."
        ) from exc


def _stack(blocks, *, axis):
    if len(blocks) == 1:
        return ar.do("expand_dims", blocks[0], axis=axis)
    return ar.do("stack", tuple(blocks), axis=axis)


def _concat(blocks, *, axis):
    return ar.do("concatenate", tuple(blocks), axis=axis)


def _drop_axis(array, axis, position):
    """Remove one virtual channel for the sequential reference primitive."""
    if isinstance(array, SparseVirtualTensor):
        groups = [
            {source: 1.0}
            for source in range(array.shape[axis])
            if source != int(position)
        ]
        return array.apply_axis_groups(groups, axis=axis)
    size = int(array.shape[axis])
    if size <= 1:
        raise ValueError("cannot remove the last virtual channel.")
    pieces = []
    if position:
        index = [slice(None)] * len(array.shape)
        index[axis] = slice(0, position)
        pieces.append(array[tuple(index)])
    if position + 1 < size:
        index = [slice(None)] * len(array.shape)
        index[axis] = slice(position + 1, None)
        pieces.append(array[tuple(index)])
    if len(pieces) == 1:
        return pieces[0]
    return _concat(pieces, axis=axis)


def _backend_index_2d(array, rows, columns):
    """Gather a virtual submatrix on the active Autoray backend."""
    rows = _as_backend(np.asarray(rows, dtype=int), like=array)
    columns = _as_backend(np.asarray(columns, dtype=int), like=array)
    return array[rows[:, None], columns[None, :]]


def _backend_index_pairs(array, rows, columns):
    """Gather paired virtual entries on the active Autoray backend."""
    rows = _as_backend(np.asarray(rows, dtype=int), like=array)
    columns = _as_backend(np.asarray(columns, dtype=int), like=array)
    return array[rows, columns]


def _scatter_add_2d(array, rows, columns, values):
    """Scatter-add paired virtual blocks without leaving the backend graph."""
    if isinstance(array, SparseVirtualTensor):
        return array.scatter_add(rows, columns, values)
    backend = ar.infer_backend(array)
    value_dtype = getattr(values, "dtype", None)
    if value_dtype is not None and getattr(array, "dtype", None) != value_dtype:
        array = ar.do("astype", array, value_dtype)
    rows_np = np.asarray(rows, dtype=int)
    columns_np = np.asarray(columns, dtype=int)

    if backend == "numpy":
        delta = np.zeros_like(array)
        np.add.at(delta, (rows_np, columns_np), values)
        return array + delta
    if backend == "torch":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        delta = ar.do("zeros_like", array)
        delta = delta.index_put(
            (rows_backend, columns_backend),
            values,
            accumulate=True,
        )
        return array + delta
    if backend == "jax":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        delta = ar.do("zeros_like", array)
        delta = delta.at[(rows_backend, columns_backend)].add(values)
        return array + delta
    if backend == "cupy":
        import cupy as cp  # pylint: disable=import-outside-toplevel

        delta = cp.zeros_like(array)
        cp.add.at(delta, (rows_np, columns_np), values)
        return array + delta

    # Keep an Autoray-compatible fallback for backends without a native
    # scatter-add rule. The common NumPy/Torch/JAX/CuPy paths above are the
    # performance-critical implementations.
    delta = ar.do("zeros_like", array)
    for row, column, value in zip(rows_np, columns_np, values):
        current = delta[row, column]
        delta = _setitem(delta, (row, column), current + value)
    return array + delta


def _scatter_set_2d(array, rows, columns, values):
    """Scatter unique paired virtual blocks without allocating a delta tensor."""
    backend = ar.infer_backend(array)
    rows_np = np.asarray(rows, dtype=int)
    columns_np = np.asarray(columns, dtype=int)
    if backend == "numpy":
        array[rows_np, columns_np] = values
        return array
    if backend == "torch":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.index_put(
            (rows_backend, columns_backend),
            values,
            accumulate=False,
        )
    if backend == "jax":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.at[(rows_backend, columns_backend)].set(values)
    if backend == "cupy":
        array[rows_np, columns_np] = values
        return array

    updated = array
    for row, column, value in zip(rows_np, columns_np, values):
        updated = _setitem(updated, (row, column), value)
    return updated


def _scatter_add_into_2d(array, rows, columns, values):
    """Scatter-add into a fresh array without creating a second full tensor."""
    backend = ar.infer_backend(array)
    rows_np = np.asarray(rows, dtype=int)
    columns_np = np.asarray(columns, dtype=int)
    if backend == "numpy":
        np.add.at(array, (rows_np, columns_np), values)
        return array
    if backend == "torch":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.index_put(
            (rows_backend, columns_backend),
            values,
            accumulate=True,
        )
    if backend == "jax":
        rows_backend = _as_backend(rows_np, like=array)
        columns_backend = _as_backend(columns_np, like=array)
        return array.at[(rows_backend, columns_backend)].add(values)
    if backend == "cupy":
        import cupy as cp  # pylint: disable=import-outside-toplevel

        cp.add.at(array, (rows_np, columns_np), values)
        return array

    updated = array
    for row, column, value in zip(rows_np, columns_np, values):
        updated = _setitem(updated, (row, column), updated[row, column] + value)
    return updated


def _history_transfer_matrix(groups, old_size):
    """Compile ordered virtual-channel groups into a dense transfer map.

    ``groups`` is the symbolic representation used by the history
    compression planners: each output channel maps to a weighted collection
    of original input channels.  The returned matrix is structural NumPy
    data, so it is safe to cache and later move to Torch/JAX/CuPy alongside
    the numerical tensor being transformed.
    """
    new_size = len(groups)
    if new_size * old_size > _MAX_HISTORY_TRANSFER_ELEMENTS:
        return None
    transfer = np.zeros((new_size, old_size), dtype=float)
    for target, group in enumerate(groups):
        for source, weight in group.items():
            transfer[target, int(source)] += float(weight)
    return transfer


def _apply_history_transfer(array, transfer, *, axis):
    """Apply a cached virtual transfer map with one backend contraction."""
    transfer = _as_backend(transfer, like=array)
    array, transfer = _align_tensordot_dtypes(array, transfer)
    mapped = ar.do(
        "tensordot",
        transfer,
        array,
        axes=([1], [axis]),
    )
    if axis == 1:
        # tensordot puts the new right-channel axis first.
        mapped = ar.do("transpose", mapped, (1, 0, 2, 3))
    return mapped


def _complex_dtype(dtype):
    """Return whether a backend dtype is complex without importing it."""
    if bool(getattr(dtype, "is_complex", False)):
        return True
    try:
        return bool(np.issubdtype(np.dtype(dtype), np.complexfloating))
    except (TypeError, ValueError):
        return False


def _align_tensordot_dtypes(left, right):
    """Align contraction operands where Torch requires identical dtypes."""
    left_dtype = getattr(left, "dtype", None)
    right_dtype = getattr(right, "dtype", None)
    if left_dtype == right_dtype:
        return left, right
    if _complex_dtype(left_dtype) and not _complex_dtype(right_dtype):
        right = ar.do("astype", right, left_dtype)
    else:
        left = ar.do("astype", left, right_dtype)
    return left, right


def _array_equal(left, right):
    """Check exact equality without introducing a numerical cutoff."""
    # Prefer backend dispatch for non-host arrays.  In particular, attempting
    # ``np.asarray`` first can synchronize or fail for CUDA arrays and can
    # sever the intended backend boundary for JAX/Torch values.
    if _backend_name(left) not in {"builtins", "numpy"} or _backend_name(right) not in {
        "builtins",
        "numpy",
    }:
        try:
            equal = ar.do("equal", left, right)
            result = ar.do("all", equal)
            return bool(result.item() if hasattr(result, "item") else result)
        except Exception:  # pragma: no cover - defensive backend guard
            return False
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception:
        try:
            equal = ar.do("equal", left, right)
            result = ar.do("all", equal)
            return bool(result.item() if hasattr(result, "item") else result)
        except Exception:  # pragma: no cover - defensive backend guard
            return False


def _setitem(array, index, value):
    """Set one tensor entry on mutable and functional array backends."""
    backend = ar.infer_backend(array)
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is None and backend == "numpy":
        value_dtype = np.asarray(value).dtype
    if value_dtype is not None and getattr(array, "dtype", None) != value_dtype:
        # Real Hamiltonian tensors must be promoted before Algorithm 3 inserts
        # complex real-time terms. Otherwise NumPy/Torch assignment or JAX
        # scatter updates silently discard the imaginary component.
        array = ar.do("astype", array, value_dtype)
    if backend == "jax":
        return array.at[index].set(value)
    if backend == "torch":
        # Do not mutate a view participating in an autodiff graph.  A clone
        # retains the graph while turning the indexed update into a tracked
        # CopySlices operation, avoiding Torch's version-counter error during
        # the later virtual-channel contractions.
        updated = array.clone()
        if isinstance(index, tuple):
            index = tuple(
                _as_backend(item, like=array)
                if isinstance(item, np.ndarray)
                else item
                for item in index
            )
        elif isinstance(index, np.ndarray):
            index = _as_backend(index, like=array)
        updated[index] = value
        return updated
    array[index] = value
    return array


def _level_number(token):
    return token.level if isinstance(token, MPOLevelToken) else int(token)


def _move_level_front(history, level):
    selected = tuple(token for token in history if _level_number(token) == level)
    remaining = tuple(token for token in history if _level_number(token) != level)
    return selected + remaining


def _history_signature(history):
    """Return the level-only signature used by the paper algorithms."""
    return tuple(_level_number(token) for token in history)


def _sort_history_front(history, level):
    """Move all tokens with ``level`` to the front, preserving the rest."""
    return _move_level_front(history, level)


def _term_from_input(term):
    if isinstance(term, (MPOProductTerm, MPOLocalOperatorTerm)):
        return term
    if isinstance(term, Mapping):
        sites = term.get("sites", term.get("locations"))
        operators = term.get(
            "operators",
            term.get("paulis", term.get("operator")),
        )
        if sites is None or operators is None:
            raise ValueError("each product term needs sites and operators.")
        coefficient = term.get("coefficient", _UNSET)
        if coefficient is _UNSET:
            if "parameter" in term:
                coefficient = MPOParameter(
                    term["parameter"],
                    term.get("default", _UNSET),
                )
            else:
                coefficient = 1.0
        coefficient = coefficient
        matrix = _as_square_operator(operators)
        normalized_sites = normalize_integer_tuple(
            sites,
            name="sites",
            allow_scalar=True,
        )
        if matrix is not None and len(normalized_sites) > 1:
            if term.get("string_operators", term.get("string_paulis")) is not None:
                raise ValueError(
                    "general local operators cannot use string_operators."
                )
            return MPOLocalOperatorTerm(
                sites=normalized_sites,
                operator=matrix,
                coefficient=coefficient,
                phys_dim=term.get("phys_dim"),
                charge=term.get("charge"),
            )
        return MPOProductTerm(
            sites=sites,
            operators=operators,
            coefficient=coefficient,
            string_operators=term.get(
                "string_operators",
                term.get("string_paulis"),
            ),
            charge=term.get("charge"),
            parities=term.get("parities"),
            braiding=term.get("braiding"),
        )
    if (
        isinstance(term, (tuple, list))
        and len(term) == 2
        and isinstance(term[0], (tuple, list))
        and len(term[0]) == 2
        and _looks_like_pauli_labels(term[0][0])
        and _looks_like_scalar(term[0][1])
    ):
        # Compact Hamiltonian form: (('ZZ', J), (site0, site1)).
        return MPOProductTerm(
            sites=term[1],
            operators=term[0][0],
            coefficient=term[0][1],
        )
    if isinstance(term, (tuple, list)) and len(term) in (2, 3):
        matrix = _as_square_operator(term[1])
        sites = normalize_integer_tuple(term[0], name="sites", allow_scalar=True)
        if matrix is not None and len(sites) > 1:
            return MPOLocalOperatorTerm(
                sites=sites,
                operator=matrix,
                coefficient=term[2] if len(term) == 3 else 1.0,
            )
        return MPOProductTerm(
            sites=term[0],
            operators=term[1],
            coefficient=term[2] if len(term) == 3 else 1.0,
        )
    raise TypeError(
        "terms must contain MPOProductTerm values, mappings, or "
        "(sites, operators[, coefficient]) / "
        "((paulis, coefficient), sites) pairs."
    )


def _looks_like_pauli_labels(value):
    """Return whether ``value`` is a compact Pauli-label sequence."""
    if isinstance(value, str):
        labels = tuple(value)
    else:
        try:
            labels = tuple(value)
        except TypeError:
            return False
    return bool(labels) and all(
        isinstance(label, str)
        and len(label) == 1
        and label.upper() in "IXYZ"
        for label in labels
    )


def _looks_like_scalar(value):
    """Return whether ``value`` is a scalar-like backend value."""
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        try:
            ndim = np.ndim(value)
        except (TypeError, ValueError):
            return False
    return ndim == 0


def _square_lattice_pauli_term(term, lattice_to_chain):
    """Normalize one coordinate-based Pauli term to chain sites."""
    if isinstance(term, MPOProductTerm):
        if not all(isinstance(site, Integral) for site in term.sites):
            raise TypeError(
                "MPOProductTerm inputs to from_square_lattice must already "
                "use integer chain sites; use a mapping or tuple for coordinates."
            )
        return term

    charge = None
    if isinstance(term, Mapping):
        locations = term.get("locations", term.get("sites"))
        paulis = term.get("paulis", term.get("operators"))
        coefficient = term.get("coefficient", _UNSET)
        if coefficient is _UNSET:
            if "parameter" in term:
                coefficient = MPOParameter(
                    term["parameter"],
                    term.get("default", _UNSET),
                )
            else:
                coefficient = 1.0
        charge = term.get("charge")
    elif isinstance(term, (tuple, list)) and len(term) in (2, 3):
        if (
            len(term) == 2
            and isinstance(term[0], (tuple, list))
            and len(term[0]) == 2
            and _looks_like_pauli_labels(term[0][0])
            and _looks_like_scalar(term[0][1])
        ):
            # Compact Hamiltonian form: (('ZZ', J), (site0, site1)).
            paulis, coefficient = term[0]
            locations = term[1]
        else:
            first, second = term[:2]
            if _looks_like_pauli_labels(first) and not _looks_like_pauli_labels(second):
                paulis, locations = first, second
            else:
                locations, paulis = first, second
            coefficient = term[2] if len(term) == 3 else 1.0
    else:
        raise TypeError(
            "square-lattice Pauli terms must be mappings with 'locations' "
            "and 'paulis', or (locations, paulis[, coefficient]) pairs."
        )

    if locations is None or paulis is None:
        raise ValueError("each square-lattice Pauli term needs locations and paulis.")
    try:
        locations = tuple(locations)
    except TypeError as exc:
        raise TypeError("term locations must be a sequence of sites.") from exc
    labels = tuple(paulis) if isinstance(paulis, str) else tuple(paulis)
    if not locations or len(locations) != len(labels):
        raise ValueError("term locations and Pauli labels must be non-empty and aligned.")

    mapped = []
    for location in locations:
        if _is_integral_value(location):
            chain_site = int(location)
            if not 0 <= chain_site < len(lattice_to_chain):
                raise ValueError(
                    f"chain site {chain_site} is outside the square lattice chain."
                )
        else:
            try:
                coordinate_values = tuple(location)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"lattice locations must be 2D coordinates, got {location!r}."
                ) from exc
            if not all(_is_integral_value(value) for value in coordinate_values):
                raise TypeError(
                    f"lattice locations must be 2D integer coordinates, got {location!r}."
                )
            coordinate = tuple(int(value) for value in coordinate_values)
            if len(coordinate) != 2:
                raise ValueError(
                    f"lattice locations must be 2D coordinates, got {location!r}."
                )
            try:
                chain_site = lattice_to_chain[coordinate]
            except KeyError as exc:
                raise ValueError(
                    f"lattice location {coordinate!r} is outside the configured lattice."
                ) from exc
        mapped.append(chain_site)

    if len(set(mapped)) != len(mapped):
        raise ValueError("a Pauli term cannot contain the same lattice site twice.")
    if charge is not None and tuple(mapped) != tuple(sorted(mapped)):
        raise ValueError(
            "square-lattice terms with charge metadata must list locations "
            "in increasing chain order."
        )

    # Operators on distinct sites commute, so sorting location/Pauli pairs
    # gives one canonical word for reversed coordinate descriptions. This
    # lets the shared automaton merge equivalent terms before evaluation.
    ordered = sorted(zip(mapped, labels), key=lambda item: item[0])
    sites, labels = zip(*ordered)
    return MPOProductTerm.from_pauli(
        sites,
        labels,
        coefficient=coefficient,
        charge=charge,
    )


def _looks_like_operator_payload(value):
    """Return whether ``value`` looks like one or more local operators."""
    if isinstance(value, str) or hasattr(value, "shape"):
        return True
    try:
        entries = tuple(value)
    except TypeError:
        return False
    return bool(entries) and all(
        isinstance(entry, str) or hasattr(entry, "shape")
        for entry in entries
    )


def _as_array_like_matrix(value):
    """Return a NumPy view for a 2D array-like value, or ``None``."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    return array if array.ndim == 2 else None


def _generic_term_parts(term):
    """Split a user-facing term into operator, locations, coefficient, metadata."""
    if isinstance(term, MPOProductTerm):
        return {
            "operator": term.operators,
            "location": term.sites,
            "coefficient": term.coefficient,
            "string_operators": term.string_operators,
            "charge": term.charge,
            "parities": term.parities,
            "braiding": term.braiding,
        }
    if isinstance(term, MPOLocalOperatorTerm):
        return {
            "operator": term.operator,
            "location": term.sites,
            "coefficient": term.coefficient,
            "string_operators": None,
            "charge": term.charge,
            "parities": None,
            "braiding": None,
        }

    if isinstance(term, Mapping):
        operator = term.get(
            "operator",
            term.get("operators", term.get("paulis", term.get("word"))),
        )
        location = term.get(
            "location",
            term.get(
                "locations",
                term.get("sites", term.get("where")),
            ),
        )
        if operator is None or location is None:
            raise ValueError(
                "each term needs an operator and location (or the plural "
                "operators/locations aliases)."
            )
        coefficient = term.get("coefficient", term.get("weight", _UNSET))
        if coefficient is _UNSET:
            if "parameter" in term:
                coefficient = MPOParameter(
                    term["parameter"],
                    term.get("default", _UNSET),
                )
            else:
                coefficient = 1.0
        return {
            "operator": operator,
            "location": location,
            "coefficient": coefficient,
            "string_operators": term.get(
                "string_operators",
                term.get("string_paulis"),
            ),
            "charge": term.get("charge"),
            "parities": term.get("parities"),
            "braiding": term.get("braiding"),
        }

    if isinstance(term, (tuple, list)) and len(term) in (2, 3):
        if (
            len(term) == 2
            and isinstance(term[0], (tuple, list))
            and len(term[0]) == 2
            and _looks_like_pauli_labels(term[0][0])
            and _looks_like_scalar(term[0][1])
        ):
            # Compact Hamiltonian form: (('ZZ', J), (site0, site1)).
            return {
                "operator": term[0][0],
                "location": term[1],
                "coefficient": term[0][1],
                "string_operators": None,
                "charge": None,
                "parities": None,
                "braiding": None,
            }
        first, second = term[:2]
        first_is_operator = _looks_like_operator_payload(first)
        second_is_operator = _looks_like_operator_payload(second)
        # Nested Python lists are useful array-like matrices but are
        # structurally indistinguishable from a 2D coordinate list in the
        # fully nested case. Resolve the common scalar/flat-site forms here;
        # callers with two nested numeric payloads can use mapping syntax.
        if (
            not first_is_operator
            and _as_array_like_matrix(first) is not None
            and _looks_like_location_sequence(second)
            and _as_array_like_matrix(second) is None
        ):
            first_is_operator = True
        if (
            not second_is_operator
            and _as_array_like_matrix(second) is not None
            and _looks_like_location_sequence(first)
            and _as_array_like_matrix(first) is None
        ):
            second_is_operator = True
        if first_is_operator and not second_is_operator:
            operator, location = first, second
        elif second_is_operator and not first_is_operator:
            location, operator = first, second
        else:
            location, operator = first, second
        return {
            "operator": operator,
            "location": location,
            "coefficient": term[2] if len(term) == 3 else 1.0,
            "string_operators": None,
            "charge": None,
            "parities": None,
            "braiding": None,
        }

    raise TypeError(
        "terms must contain MPOProductTerm values, mappings, or "
        "(location, operator[, coefficient]) / "
        "(operator, location[, coefficient]) / "
        "((paulis, coefficient), location) pairs."
    )


def _operator_factors(operator):
    """Return a tuple of local operator factors."""
    if isinstance(operator, str):
        return tuple(operator)
    if hasattr(operator, "shape"):
        return (operator,)
    array = _as_array_like_matrix(operator)
    if array is not None and array.ndim == 2:
        return (array,)
    try:
        factors = tuple(operator)
    except TypeError as exc:
        raise TypeError("operator must be a local matrix or operator sequence.") from exc
    if not factors:
        raise ValueError("operator sequences must be non-empty.")
    return factors


def _looks_like_location_sequence(value):
    """Return whether a shorthand value is a site or coordinate sequence."""
    if _is_integral_value(value):
        return True
    try:
        values = tuple(value)
    except TypeError:
        return False
    if not values:
        return False
    if all(_is_integral_value(item) for item in values):
        return True
    return all(
        not isinstance(item, (str, bytes, Integral))
        and _looks_like_location_sequence(item)
        for item in values
    )


def _expand_term_collection(terms):
    """Expand Pepsy-style ``{operator: location[, coefficient]}`` mappings."""
    if not isinstance(terms, Mapping):
        return tuple(terms)

    canonical_keys = {
        "operator",
        "operators",
        "paulis",
        "word",
        "location",
        "locations",
        "sites",
        "where",
        "coefficient",
        "weight",
        "parameter",
    }
    if any(key in canonical_keys for key in terms):
        return (terms,)

    expanded = []
    for operator, value in terms.items():
        if isinstance(value, Mapping):
            record = dict(value)
            record.setdefault("operator", operator)
            expanded.append(record)
            continue

        # ``{"XX": (2, 3)}`` is a two-site term. To attach a coefficient,
        # use ``{"XX": ((2, 3), coefficient)}``; nested coordinate supports
        # such as ``{"XX": ((0, 0), (1, 0))}`` remain unambiguous.
        location = value
        coefficient = 1.0
        if isinstance(value, (tuple, list)) and len(value) == 2:
            first, second = value
            if not _looks_like_location_sequence(value):
                location, coefficient = first, second
            elif _looks_like_location_sequence(first) and not _looks_like_location_sequence(second):
                location, coefficient = first, second
        expanded.append(
            {
                "operator": operator,
                "location": location,
                "coefficient": coefficient,
            }
        )
    return tuple(expanded)


def _location_dimensions(
    location,
    num_factors,
    expected_ndim=None,
    *,
    mapper=None,
):
    """Normalize locations and identify whether they are chain or lattice sites."""
    if _is_integral_value(location):
        return 1, (int(location),)

    try:
        values = tuple(location)
    except TypeError as exc:
        raise TypeError("location must be an integer or a coordinate sequence.") from exc
    if not values:
        raise ValueError("term locations must be non-empty.")

    if all(_is_integral_value(value) for value in values):
        # A single local operator at (x, y) or (x, y, z) is one lattice site.
        # Multiple factors at flat integer locations are the conventional 1D form.
        if (
            num_factors == 1
            and expected_ndim in (2, 3)
            and not (mapper is not None and len(values) != expected_ndim)
        ):
            return expected_ndim, (tuple(int(value) for value in values),)
        if num_factors == 1 and expected_ndim is None and len(values) in (2, 3):
            return len(values), (tuple(int(value) for value in values),)
        return 1, tuple(int(value) for value in values)

    coordinates = []
    for value in values:
        try:
            coordinate_values = tuple(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "lattice locations must be integer coordinates."
            ) from exc
        if not all(_is_integral_value(item) for item in coordinate_values):
            raise TypeError("lattice locations must be integer coordinates.")
        coordinate = tuple(int(item) for item in coordinate_values)
        if not coordinate:
            raise ValueError("lattice coordinates must be non-empty.")
        coordinates.append(coordinate)
    ndim = len(coordinates[0])
    if any(len(coordinate) != ndim for coordinate in coordinates):
        raise ValueError("all lattice coordinates in a term must have the same dimension.")
    if ndim == 1:
        return 1, tuple(coordinate[0] for coordinate in coordinates)
    return ndim, tuple(coordinates)


def _normalize_term_shape(shape):
    """Normalize a public ``shape`` argument to a tuple."""
    if _is_integral_value(shape):
        shape = (int(shape),)
    else:
        try:
            values = tuple(shape)
        except (TypeError, ValueError) as exc:
            raise TypeError("shape must be an integer or a 2D/3D shape tuple.") from exc
        if not all(_is_integral_value(value) for value in values):
            raise TypeError("shape must contain only integer dimensions.")
        shape = tuple(int(value) for value in values)
    if len(shape) not in (1, 2, 3) or any(value < 1 for value in shape):
        raise ValueError("shape must contain one, two, or three positive dimensions.")
    return shape


def _compile_generic_terms(terms, *, shape=None, mapper=None, map_mode="snake"):
    """Compile coordinate-aware term records into chain product terms."""
    raw_terms = tuple(
        _generic_term_parts(term)
        for term in _expand_term_collection(terms)
    )
    if not raw_terms:
        raise ValueError("terms must contain at least one product term.")

    if mapper is not None:
        from pepsy.tensors import OneDMap  # pylint: disable=import-outside-toplevel

        if not isinstance(mapper, OneDMap):
            raise TypeError("mapper must be a pepsy.tensors.OneDMap or None.")
        mapped_shape = tuple(mapper.shape)
        if shape is not None and _normalize_term_shape(shape) != mapped_shape:
            raise ValueError(
                f"shape {_normalize_term_shape(shape)} does not match mapper shape "
                f"{mapped_shape}."
            )
        shape_tuple = mapped_shape
    elif shape is not None:
        shape_tuple = _normalize_term_shape(shape)
    else:
        shape_tuple = None

    expected_ndim = None if shape_tuple is None else len(shape_tuple)
    located_terms = []
    inferred_shape = []
    for raw in raw_terms:
        local_operator = _as_square_operator(raw["operator"])
        if local_operator is None:
            factors = _operator_factors(raw["operator"])
            num_factors = len(factors)
        else:
            try:
                raw_locations = tuple(raw["location"])
            except TypeError:
                raw_locations = None
            if raw_locations is None or (
                expected_ndim in (2, 3)
                and all(_is_integral_value(value) for value in raw_locations)
            ):
                num_factors = 1
            else:
                num_factors = len(raw_locations)
            factors = None
        ndim, locations = _location_dimensions(
            raw["location"],
            num_factors,
            expected_ndim=expected_ndim,
            mapper=mapper,
        )
        if expected_ndim is not None and ndim != expected_ndim and not (
            mapper is not None and ndim == 1
        ):
            raise ValueError(
                f"locations have dimension {ndim}, but configured shape is "
                f"{shape_tuple}."
            )
        if ndim not in (1, 2, 3):
            raise ValueError("locations must be one-dimensional, 2D, or 3D.")
        if located_terms and ndim != located_terms[0][0]:
            raise ValueError("all terms must use the same location dimension.")
        if factors is not None and len(locations) != len(factors):
            raise ValueError(
                "each term must provide one local operator per location; "
                f"got {len(factors)} operators for {len(locations)} locations."
            )
        if local_operator is not None and len(locations) == 1:
            factors = (local_operator,)
            local_operator = None
        located_terms.append((ndim, locations, factors, local_operator, raw))
        if shape_tuple is None:
            if not inferred_shape:
                inferred_shape = [0] * ndim
            for location in locations:
                if ndim == 1:
                    inferred_shape[0] = max(inferred_shape[0], int(location) + 1)
                else:
                    for axis, value in enumerate(location):
                        inferred_shape[axis] = max(inferred_shape[axis], int(value) + 1)

    if shape_tuple is None:
        shape_tuple = tuple(inferred_shape)
        if not shape_tuple or any(value < 1 for value in shape_tuple):
            raise ValueError("could not infer a positive lattice shape from terms.")
    if len(shape_tuple) == 1:
        chain_length = shape_tuple[0]
        chain_maps = None
    else:
        from pepsy.tensors import OneDMap  # pylint: disable=import-outside-toplevel

        if mapper is None:
            mapper = OneDMap(*shape_tuple, mode=map_mode)
        elif tuple(mapper.shape) != shape_tuple:
            raise ValueError(
                f"mapper shape {mapper.shape} does not match configured shape {shape_tuple}."
            )
        chain_maps = mapper.build()
        chain_length = int(np.prod(shape_tuple))

    compiled = []
    for ndim, locations, factors, local_operator, raw in located_terms:
        if ndim == 1:
            mapped = tuple(int(location) for location in locations)
            if any(location < 0 or location >= chain_length for location in mapped):
                raise ValueError("a term location is outside the configured chain shape.")
        else:
            _chain_to_lattice, lattice_to_chain = chain_maps
            try:
                mapped = tuple(lattice_to_chain[tuple(location)] for location in locations)
            except KeyError as exc:
                raise ValueError(
                    f"lattice location {exc.args[0]!r} is outside shape {shape_tuple}."
                ) from exc
        if local_operator is not None:
            if raw["parities"] is not None or raw["braiding"] is not None:
                raise ValueError(
                    "general local operators carry their complete ordering; "
                    "factor parities/braiding apply only to product terms."
                )
            if raw["string_operators"] is not None:
                raise ValueError(
                    "general local operators cannot use string_operators."
                )
            compiled.append(
                MPOLocalOperatorTerm(
                    sites=mapped,
                    operator=local_operator,
                    coefficient=raw["coefficient"],
                    charge=raw["charge"],
                )
            )
            continue
        if (
            (raw["charge"] is not None or raw["string_operators"] is not None)
            and tuple(mapped) != tuple(sorted(mapped))
        ):
            raise ValueError(
                "terms with charge or string_operators metadata must list "
                "locations in increasing order."
            )
        ordered = sorted(zip(mapped, factors), key=lambda item: item[0])
        sites, ordered_factors = zip(*ordered)
        compiled.append(
            MPOProductTerm(
                sites=sites,
                operators=ordered_factors,
                coefficient=raw["coefficient"],
                string_operators=raw["string_operators"],
                charge=raw["charge"],
                parities=raw["parities"],
                braiding=raw["braiding"],
            )
        )

    metadata = {
        "shape": shape_tuple,
        "mapper": mapper,
        "chain_to_lattice": None if chain_maps is None else dict(chain_maps[0]),
        "lattice_to_chain": None if chain_maps is None else dict(chain_maps[1]),
        "location_mode": "chain" if located_terms[0][0] == 1 else "lattice",
        "shape_inferred": shape is None and mapper is None,
    }
    return chain_length, tuple(compiled), metadata


def _mixed_term_automaton(L, terms, *, phys_dim, unit_coefficients):
    """Compile product and general local terms into one exact automaton."""

    automaton = MPOAutomaton(L, phys_dim=int(phys_dim))
    term_slots = []
    for term_index, term in enumerate(terms):
        coefficient = 1.0 if unit_coefficients else term.coefficient
        if isinstance(term, MPOProductTerm):
            first_site = term.sites[0]
            transition_index = len(automaton.transitions[first_site])
            automaton.add_product_term(
                term.sites,
                term.operators,
                coefficient=coefficient,
                string_operators=term.string_operators,
                channel_id=("basis-product", term_index),
                charge=term.charge,
            )
            term_slots.append(((first_site, transition_index, term.operators[0]),))
            continue

        cores = _local_operator_mpo_cores(term)
        slots = automaton.add_local_mpo_term(
            term.sites,
            cores,
            coefficient=coefficient,
            channel_id=("basis-local-mpo", term_index),
            charge=term.charge,
            return_slots=True,
        )
        term_slots.append(tuple(slots))
    return automaton, tuple(term_slots)


class FirstDegreeMPO:
    """A finite-chain MPO with explicit virtual-level histories.

    The object is intentionally usable for intermediate products as well as
    first-degree Hamiltonians.  ``degree`` records the algebraic degree of
    the current expression; it is ``1`` for a Hamiltonian built from local
    terms and increases under products.

    The public algebraic methods return new semantic objects.  The one
    exception is ``compress_exact(inplace=True)``, which is explicit because
    it mutates virtual-bond tensors and histories.  ``arrays`` is a read-only
    tuple view, but the backend arrays inside it are not copied; callers that
    need ownership should call :meth:`copy` before mutating backend data.

    This is a semantic MPO for the paper's construction, not a drop-in
    replacement for every arbitrary Quimb MPO.  Higher-order history methods
    require the first-degree level-1/2/3 rail structure created by
    :meth:`from_automaton` or :meth:`from_local_terms`.

    Parameters
    ----------
    arrays : sequence[array_like]
        MPO tensors in ``lrud`` order.  Open-boundary rank-3 tensors and a
        rank-2 one-site tensor are accepted.
    levels : sequence[sequence[MPOLevel]], optional
        Labels for the ``L + 1`` virtual bonds, including the two singleton
        boundary bonds.  When omitted, neutral level metadata is generated.
    degree : int, default=1
        Algebraic degree represented by the expression.
    symmetry : {"U1", "Z2", "U1U1", "Z2Z2"}, optional
        Abelian symmetry used when compiling to native Symmray tensors.
    physical_charges : sequence or mapping, optional
        Charge of each local dense basis state, in physical-index order. A
        mapping such as ``{0: 1, 1: 2, 2: 1}`` is also accepted and expands
        charge sectors by multiplicity in mapping order. Equal charges must
        form contiguous sectors.
    fermionic : bool, default=False
        Request graded Symmray tensors. The higher-order block-sparse backend
        currently rejects this option until native fermionic history signs are
        represented semantically.
    """

    def __init__(
        self,
        arrays,
        *,
        levels=None,
        degree=1,
        symmetry=None,
        physical_charges=None,
        fermionic=False,
        physical_space=None,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        metadata=None,
    ):
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError("arrays must contain at least one MPO tensor.")
        if not isinstance(degree, Integral) or int(degree) < 0:
            raise ValueError("degree must be a non-negative integer.")
        self.L = len(arrays)
        self._arrays = tuple(
            _as_4d(array, site=site, length=self.L)
            for site, array in enumerate(arrays)
        )
        self.degree = int(degree)
        self.upper_ind_id = upper_ind_id
        self.lower_ind_id = lower_ind_id
        self.site_tag_id = site_tag_id
        self.metadata = dict(metadata or {})
        # The plan is intentionally separate from numerical tensor storage.
        # Constructors that know the source automaton install an exact plan;
        # direct array construction builds a conservative plan lazily.
        self._block_plan = None
        if physical_space is not None and not isinstance(
            physical_space, MPOPhysicalSpace
        ):
            raise TypeError("physical_space must be an MPOPhysicalSpace or None.")
        if physical_space is not None and (
            symmetry is not None or physical_charges is not None or fermionic
        ):
            raise ValueError(
                "physical_space cannot be combined with symmetry, "
                "physical_charges, or fermionic metadata."
            )
        self.physical_space = physical_space
        selected_symmetry = (
            physical_space.symmetry if physical_space is not None else symmetry
        )
        self.symmetry = (
            None
            if selected_symmetry is None
            else _normalize_mpo_symmetry(selected_symmetry)
        )
        self.physical_charges = (
            physical_space.physical_charges
            if physical_space is not None
            else physical_charges
        )
        self.fermionic = (
            physical_space.fermionic if physical_space is not None else bool(fermionic)
        )
        # Keep this attribute present on every instance so callers do not
        # need to probe for it after an optional compression stage. It is
        # populated by ``compress_exact`` or ``extensive_exponential``.
        self.compression_report = self.metadata.get("compression_report")
        # Populated by ``from_automaton`` when the source graph is available.
        # Keeping this private means direct array construction remains a
        # supported fallback, while local-term constructors can avoid
        # materializing unreachable Cartesian history states.
        self._structural_transitions = None
        # Raw history topology depends only on the structural MPO and the
        # Taylor order. Keep it separate from value-dependent exact merges so
        # parameterized builds can reuse the expensive graph walk safely.
        self._history_topology_cache = {}
        # The following caches contain symbolic indices and histories only.
        # Numerical tensors are deliberately never retained here: a bound
        # Torch/JAX MPO must build a new autodiff graph for every call.
        self._history_symbolic_cache = None
        self._history_extension_plan_cache = {}
        self._history_compression_plan_cache = {}
        self._history_approximation_plan_cache = {}
        self._history_tensor_plan_cache = {}
        self._history_reduced_plan_cache = {}
        self._base_level_position_cache = None
        self._levels = self._normalize_levels(levels)
        self._validate()
        self._validate_symmetry()

    def _validate_symmetry(self):
        """Validate optional native block-sparse compilation metadata."""
        if self.physical_space is not None and self.physical_space.phys_dim != self.phys_dim:
            raise ValueError(
                f"physical_space has phys_dim={self.physical_space.phys_dim}, "
                f"but MPO tensors use phys_dim={self.phys_dim}."
            )
        if self.symmetry is None:
            if self.physical_charges is not None:
                raise ValueError("physical_charges requires symmetry metadata.")
            if self.fermionic:
                raise ValueError("fermionic=True requires symmetry metadata.")
            self.physical_space = MPOPhysicalSpace(self.phys_dim)
            return
        if self.physical_charges is None:
            raise ValueError("symmetry requires physical_charges.")
        self.physical_charges = _normalize_mpo_physical_charges(
            self.physical_charges,
            self.phys_dim,
            self.symmetry,
        )
        closed_sectors = set()
        sentinel = object()
        previous = sentinel
        for charge in self.physical_charges:
            if charge != previous:
                if charge in closed_sectors:
                    raise ValueError(
                        "physical_charges must group equal charge sectors "
                        "contiguously in the local dense basis."
                    )
                if previous is not sentinel:
                    closed_sectors.add(previous)
                previous = charge
        self.physical_space = MPOPhysicalSpace(
            self.phys_dim,
            symmetry=self.symmetry,
            physical_charges=tuple(self.physical_charges),
            fermionic=self.fermionic,
            braiding=(
                None
                if self.physical_space is None
                else self.physical_space.braiding
            ),
        )

    def _symmetry_options(self):
        """Return constructor options that preserve native symmetry metadata."""
        return {
            "physical_space": self.physical_space,
        }

    def _normalize_levels(self, levels):
        if levels is None:
            out = [[MPOLevel(
                ("left",),
                (MPOLevelToken(1),),
            )]]
            for bond, array in enumerate(self._arrays[:-1], start=1):
                out.append([
                    MPOLevel(
                        ("bond", bond, pos),
                        (MPOLevelToken(2, ("bond", bond, pos)),),
                    )
                    for pos in range(array.shape[1])
                ])
            out.append([
                MPOLevel(("right",), (MPOLevelToken(3),))
            ])
            return out

        if len(levels) != self.L + 1:
            raise ValueError(f"levels must have length L + 1 = {self.L + 1}.")
        normalized = []
        for bond, (bond_levels, array) in enumerate(zip(levels, [None, *self._arrays])):
            values = []
            for pos, level in enumerate(bond_levels):
                if isinstance(level, MPOLevel):
                    values.append(level)
                else:
                    values.append(
                        MPOLevel(
                            level,
                            (MPOLevelToken(2, level),),
                        )
                    )
            normalized.append(values)
        return normalized

    def _validate(self):
        if len(self._levels) != self.L + 1:
            raise ValueError("there must be one level list per virtual bond.")
        for site, array in enumerate(self._arrays):
            if len(self._levels[site]) != array.shape[0]:
                raise ValueError(
                    f"bond {site} has {len(self._levels[site])} levels but "
                    f"tensor {site} has left dimension {array.shape[0]}.")
            if len(self._levels[site + 1]) != array.shape[1]:
                raise ValueError(
                    f"bond {site + 1} has {len(self._levels[site + 1])} levels but "
                    f"tensor {site} has right dimension {array.shape[1]}.")
            if array.shape[-1] != array.shape[-2]:
                raise ValueError("all local MPO physical dimensions must be square.")
        phys_dim = self._arrays[0].shape[-1]
        if any(array.shape[-1] != phys_dim for array in self._arrays):
            raise ValueError("all MPO sites must have the same physical dimension.")
        if any(len(levels) != 1 for levels in (self._levels[0], self._levels[-1])):
            raise ValueError("open-boundary first-degree MPOs need singleton boundary bonds.")

    @property
    def arrays(self):
        """Read-only tuple of normalized ``(left, right, up, down)`` tensors.

        The tuple itself is immutable, and dense backend tensors are returned
        by reference to preserve Autoray dtype/device/backend behavior. A
        block-sparse history result is materialized only when this dense-array
        compatibility property is explicitly read; :meth:`to_mpo` compiles it
        directly into Symmray sectors when symmetry metadata is present.
        """
        return tuple(
            _sparse_virtual_to_dense(array)
            if isinstance(array, SparseVirtualTensor)
            else array
            for array in self._arrays
        )

    @property
    def is_block_sparse(self):
        """Whether local tensors retain sparse operator-valued virtual blocks."""
        return all(isinstance(array, SparseVirtualTensor) for array in self._arrays)

    @property
    def block_plan(self):
        """Return the backend-neutral structural MPO block plan.

        The plan contains virtual-state labels, local block recipes, and
        optional charge metadata, but never owns local numerical arrays. It
        can therefore be inspected or cached independently of backend and
        coefficient rebinding. Dense tensors without a source automaton are
        represented conservatively by all virtual pairs.
        """
        if self._block_plan is None:
            self._block_plan = MPOBlockPlan.from_semantic(
                self,
                kind="history" if self.degree > 1 else "compiled",
                metadata=(
                    {"symmetry": self.symmetry}
                    if self.symmetry is not None
                    else None
                ),
            )
            self.metadata.setdefault("block_plan", self._block_plan.summary())
        return self._block_plan

    def validate_charge_flow(self):
        """Validate symbolic virtual charges and native physical flow.

        The structural plan is checked first without touching numerical values.
        When Abelian symmetry metadata is configured, ``to_mpo`` then performs
        the value-level local block check at the native Symmray boundary. No
        dense global operator is formed by this method.
        """

        report = self.block_plan.validate_charges(self.symmetry)
        if self.symmetry is None:
            self.metadata["charge_validation"] = report.as_dict()
            return report

        mpo = self.to_mpo()
        native_summary = _native_sector_summary(mpo)
        report = replace(
            report,
            native=native_summary is not None,
            native_blocks=(
                0
                if native_summary is None
                else native_summary["total_blocks"]
            ),
            native_sectors=(
                0
                if native_summary is None
                else sum(
                    len(tensor.data.blocks)
                    for tensor in mpo.tensors
                )
            ),
            message="structural and native physical charge flow are valid",
        )
        self.metadata["charge_validation"] = report.as_dict()
        return report

    @property
    def sparse_block_counts(self):
        """Stored virtual block count per site, or ``None`` for dense tensors."""
        return tuple(
            array.stored_blocks
            if isinstance(array, SparseVirtualTensor)
            else None
            for array in self._arrays
        )

    @property
    def levels(self):
        """Read-only level metadata grouped by virtual bond."""
        return tuple(tuple(levels) for levels in self._levels)

    @property
    def bond_dimensions(self):
        return tuple(len(levels) for levels in self._levels[1:-1])

    @property
    def phys_dim(self):
        return int(self._arrays[0].shape[-1])

    @property
    def is_first_degree(self):
        return self.degree == 1

    def copy(self):
        """Return a semantic copy sharing backend tensor storage.

        This is intentionally a structural copy, not a deep array copy.  It
        is sufficient for the non-mutating algebraic API and avoids an
        unnecessary device transfer; use backend-specific copying before
        editing tensor values in place.
        """
        out = type(self)(
            self._arrays,
            levels=self.levels,
            degree=self.degree,
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=self.metadata,
        )
        out._structural_transitions = self._structural_transitions
        out._history_topology_cache = self._history_topology_cache
        out._history_symbolic_cache = self._history_symbolic_cache
        out._history_extension_plan_cache = self._history_extension_plan_cache
        out._history_compression_plan_cache = self._history_compression_plan_cache
        out._history_approximation_plan_cache = (
            self._history_approximation_plan_cache
        )
        out._history_tensor_plan_cache = self._history_tensor_plan_cache
        out._history_reduced_plan_cache = self._history_reduced_plan_cache
        out._base_level_position_cache = self._base_level_position_cache
        out._block_plan = self._block_plan
        return out

    def _bind_arrays(self, arrays):
        """Make a lightweight view with new local tensors and shared plans.

        ``MPOBasis`` uses this private primitive for its value-only compiled
        evaluator.  Reconstructing a semantic MPO through ``__init__`` would
        repeat validation and level normalization on every optimizer step;
        the structural metadata and history plans are immutable for this
        purpose, while the backend arrays must remain fresh so their current
        autodiff graph is never cached.
        """
        arrays = tuple(arrays)
        if len(arrays) != self.L:
            raise ValueError(f"arrays must have length {self.L}.")
        out = object.__new__(type(self))
        out.L = self.L
        out._arrays = tuple(
            _as_4d(array, site=site, length=self.L)
            for site, array in enumerate(arrays)
        )
        out.degree = self.degree
        out.upper_ind_id = self.upper_ind_id
        out.lower_ind_id = self.lower_ind_id
        out.site_tag_id = self.site_tag_id
        out.symmetry = self.symmetry
        out.physical_charges = self.physical_charges
        out.fermionic = self.fermionic
        out.physical_space = self.physical_space
        out.metadata = {}
        out.compression_report = None
        out._structural_transitions = self._structural_transitions
        out._history_topology_cache = self._history_topology_cache
        out._history_symbolic_cache = self._history_symbolic_cache
        out._history_extension_plan_cache = self._history_extension_plan_cache
        out._history_compression_plan_cache = self._history_compression_plan_cache
        out._history_approximation_plan_cache = (
            self._history_approximation_plan_cache
        )
        out._history_tensor_plan_cache = self._history_tensor_plan_cache
        out._history_reduced_plan_cache = self._history_reduced_plan_cache
        out._base_level_position_cache = self._base_level_position_cache
        out._block_plan = self._block_plan
        out._levels = self._levels
        return out

    @staticmethod
    def _history_for_state(state, *, start_state, done_state):
        if state == start_state:
            return (MPOLevelToken(1),)
        if state == done_state:
            return (MPOLevelToken(3),)
        return (MPOLevelToken(2, state),)

    @classmethod
    def from_automaton(cls, automaton, *, degree=1, **kwargs):
        """Compile an :class:`MPOAutomaton` into a semantic MPO.

        The automaton remains the source of numerical local blocks, while the
        returned object adds the symbolic level histories needed by the
        higher-order algorithms.  No dense operator is constructed.
        """
        if not isinstance(automaton, MPOAutomaton):
            raise TypeError("automaton must be an MPOAutomaton.")
        arrays = automaton.to_arrays()
        levels = [[
            MPOLevel(
                automaton.start_state,
                cls._history_for_state(
                    automaton.start_state,
                    start_state=automaton.start_state,
                    done_state=automaton.done_state,
                ),
            )
        ]]
        for cut_channels in automaton.channels:
            levels.append([
                MPOLevel(
                    channel.state,
                    cls._history_for_state(
                        channel.state,
                        start_state=automaton.start_state,
                        done_state=automaton.done_state,
                    ),
                    charge=channel.charge,
                )
                for channel in cut_channels
            ])
        levels.append([MPOLevel(
            automaton.done_state,
            cls._history_for_state(
                automaton.done_state,
                start_state=automaton.start_state,
                done_state=automaton.done_state,
            ),
        )])
        result = cls(arrays, levels=levels, degree=degree, **kwargs)
        if result.L > 1:
            result._structural_transitions = (
                result._structural_transitions_from_automaton(automaton)
            )
            result._history_symbolic_cache = None
        result._block_plan = MPOBlockPlan.from_automaton(
            automaton,
            metadata=(
                {"symmetry": result.symmetry}
                if result.symmetry is not None
                else None
            ),
        )
        result.metadata["block_plan"] = result._block_plan.summary()
        return result

    @classmethod
    def from_local_terms(
        cls,
        L,
        terms,
        *,
        phys_dim=None,
        degree=1,
        share_channels=True,
        **kwargs,
    ):
        """Build a first-degree MPO from factorized local product terms.

        This is the preferred public constructor for Hamiltonian-like sums.
        The input terms are compiled through :class:`MPOAutomaton`, which
        keeps the identity rails and active channels explicit.
        """
        terms = tuple(_term_from_input(term) for term in terms)
        if not terms:
            raise ValueError("terms must contain at least one product term.")
        if phys_dim is None:
            first_term = terms[0]
            phys_dim = (
                first_term.phys_dim
                if isinstance(first_term, MPOLocalOperatorTerm)
                else int(first_term.operators[0].shape[0])
            )
        if any(isinstance(term, MPOLocalOperatorTerm) for term in terms):
            automaton, _term_slots = _mixed_term_automaton(
                L,
                terms,
                phys_dim=phys_dim,
                unit_coefficients=False,
            )
            return cls.from_automaton(automaton, degree=degree, **kwargs)
        automaton = MPOAutomaton.from_product_terms(
            L,
            terms,
            phys_dim=phys_dim,
            share_channels=share_channels,
        )
        return cls.from_automaton(automaton, degree=degree, **kwargs)

    @classmethod
    def from_pauli_terms(
        cls,
        L,
        terms,
        *,
        degree=1,
        share_channels=True,
        **kwargs,
    ):
        """Build a first-degree MPO from compact Pauli product terms.

        Each term may be an :class:`MPOProductTerm`, a mapping with
        ``sites`` (or ``locations``), ``paulis`` (or ``operators``), and an
        optional ``coefficient``, or a ``(sites, paulis)``/
        ``(sites, paulis, coefficient)`` tuple. The compact Hamiltonian form
        ``((paulis, coefficient), sites)`` is accepted as well. Pauli strings
        such as ``"ZXY"`` and label sequences such as ``("Z", "X", "Y")``
        are accepted. Sites are zero-based and list the non-identity support.
        """
        return cls.from_local_terms(
            L,
            terms,
            phys_dim=2,
            degree=degree,
            share_channels=share_channels,
            **kwargs,
        )

    @classmethod
    def identity(cls, L, phys_dim, *, like=None, **kwargs):
        """Construct an exact identity MPO with degree zero.

        ``like`` optionally supplies the backend and dtype for the local
        identity blocks; it is useful when the identity is used as the
        neutral element of a backend-native algebraic operation.
        """
        return cls.from_automaton(
            MPOAutomaton.identity(L, phys_dim, like=like),
            degree=0,
            **kwargs,
        )

    def to_mpo(self):
        """Compile to a Quimb ``MatrixProductOperator`` without compression.

        This method is the deliberate interop boundary.  It preserves local
        tensor backend/dtype information, performs no SVD or bond truncation,
        and attaches a semantic copy as ``pepsy_first_degree`` so callers can
        move between Quimb execution and Pepsy history inspection.
        """
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        # Quimb is the stable tensor-network interchange boundary. Native
        # symmetry metadata selects direct Symmray block compilation; this
        # avoids ever allocating the full dense virtual tensors retained only
        # for compatibility by ``arrays``.
        if self.symmetry is not None:
            sparse_arrays = tuple(
                array
                if isinstance(array, SparseVirtualTensor)
                else _dense_virtual_to_sparse(array)
                for array in self._arrays
            )
            compiled_arrays = symmray_arrays_from_sparse(
                sparse_arrays,
                self._levels,
                symmetry=self.symmetry,
                physical_charges=self.physical_charges,
                fermionic=self.fermionic,
            )
        else:
            dense_arrays = tuple(
                _sparse_virtual_to_dense(array)
                if isinstance(array, SparseVirtualTensor)
                else array
                for array in self._arrays
            )
            if self.L == 1:
                compiled_arrays = (dense_arrays[0][0, 0],)
            else:
                compiled_arrays = (
                    dense_arrays[0][0],
                    *dense_arrays[1:-1],
                    dense_arrays[-1][:, 0],
                )
        mpo = _pepsy_mpo_class(qtn.MatrixProductOperator)(
            compiled_arrays,
            shape="lrud",
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
        )
        mpo.pepsy_first_degree = self.copy()
        mpo.pepsy_block_plan = self.block_plan
        mpo.pepsy_mpo_symmetry = self.symmetry
        mpo.pepsy_mpo_physical_charges = self.physical_charges
        mpo.pepsy_mpo_physical_dimension = self.phys_dim
        return mpo

    def apply_to_mps(
        self, mps, *, method="direct", inplace=False, **compress_opts,
    ):
        """Apply this MPO to a Quimb MPS using tensor-network compression.

        The semantic object is compiled at the Quimb boundary and delegated
        to ``MatrixProductState.gate_with_mpo``.  No dense state or operator
        is formed by this method.
        """
        if not hasattr(mps, "gate_with_mpo"):
            raise TypeError("mps must provide Quimb's gate_with_mpo method.")
        return mps.gate_with_mpo(
            self.to_mpo(),
            method=method,
            inplace=inplace,
            **compress_opts,
        )

    def compress_numerical(
        self,
        *,
        form=None,
        max_bond=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        create_bond=False,
        estimate_error=False,
        return_report=False,
        sector_aware="auto",
        **compress_opts,
    ):
        """Numerically compress the compiled MPO through Quimb.

        This is intentionally separate from :meth:`compress_exact` and the
        paper's analytical Algorithm 4.  The returned object is an ordinary
        Quimb ``MatrixProductOperator`` because a numerical truncation can no
        longer be represented faithfully by the original semantic histories.
        The report records the requested policy and the bond dimensions before
        and after the Quimb sweep.  Set ``estimate_error=True`` to additionally
        contract the Frobenius norm of ``MPO_before - MPO_after`` without
        densifying either operator. ``sector_aware='auto'`` uses the native
        Symmray sector-wise SVD when the compiled MPO has Abelian blocks;
        ``sector_aware=True`` rejects a dense fallback.
        """
        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        sector_aware_request = _normalize_sector_aware_request(sector_aware)
        mpo = self.to_mpo()
        initial_sector_summary = _native_sector_summary(mpo)
        sector_aware = _resolve_sector_aware(
            sector_aware_request,
            initial_sector_summary,
        )
        cutoff = _resolve_compression_cutoff(cutoff, mpo[0].data)
        cutoff_mode = _resolve_compression_cutoff_mode(cutoff_mode)
        if not isinstance(estimate_error, bool):
            raise TypeError("estimate_error must be a boolean.")
        if "max_bond" in compress_opts or "cutoff" in compress_opts:
            raise TypeError(
                "max_bond and cutoff must be supplied as explicit compression "
                "arguments, not duplicated in compress_opts."
            )

        reference = self.to_mpo() if estimate_error else None
        initial_bond_dimensions = tuple(int(size) for size in mpo.bond_sizes())
        mpo.compress(
            form=form,
            create_bond=create_bond,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            **compress_opts,
        )
        final_bond_dimensions = tuple(int(size) for size in mpo.bond_sizes())
        final_sector_summary = _native_sector_summary(mpo)
        operator_frobenius_error = None
        operator_frobenius_relative_error = None
        error_estimator = None
        if estimate_error:
            # ``TensorNetwork.norm`` contracts the doubled network and hence
            # computes the operator Frobenius norm here. Keeping this behind
            # an explicit flag leaves the normal compression path unchanged.
            difference = reference - mpo
            operator_frobenius_error = difference.norm()
            reference_norm = reference.norm()
            try:
                has_reference_norm = bool(reference_norm != 0)
            except (TypeError, ValueError):  # pragma: no cover - tracer guard
                has_reference_norm = True
            if has_reference_norm:
                operator_frobenius_relative_error = (
                    operator_frobenius_error / reference_norm
                )
            error_estimator = "tensor-network-frobenius"
        report = MPONumericalCompressionReport(
            method="quimb",
            form=form,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            truncated=final_bond_dimensions != initial_bond_dimensions,
            truncation_error=operator_frobenius_error,
            operator_frobenius_error=operator_frobenius_error,
            operator_frobenius_relative_error=operator_frobenius_relative_error,
            error_estimator=error_estimator,
            sector_aware=sector_aware,
            initial_sector_dimensions=(
                ()
                if initial_sector_summary is None
                else initial_sector_summary["bond_sector_dimensions"]
            ),
            final_sector_dimensions=(
                ()
                if final_sector_summary is None
                else final_sector_summary["bond_sector_dimensions"]
            ),
            initial_sector_block_counts=(
                ()
                if initial_sector_summary is None
                else initial_sector_summary["site_block_counts"]
            ),
            final_sector_block_counts=(
                ()
                if final_sector_summary is None
                else final_sector_summary["site_block_counts"]
            ),
        )
        # Numerical truncation invalidates the semantic history attachment.
        mpo.pepsy_first_degree = None
        mpo.pepsy_numerical_compression_report = report
        mpo.pepsy_sector_compression_metadata = {
            "requested": sector_aware_request,
            "sector_aware": sector_aware,
            "native": initial_sector_summary is not None,
            "initial": initial_sector_summary,
            "final": final_sector_summary,
        }
        if return_report:
            return mpo, report
        return mpo

    def compress_fixed_rank(
        self,
        max_bond,
        *,
        form="left",
        return_report=False,
    ):
        """Compress with a fixed-rank, backend-differentiable TT-SVD sweep.

        This is the autodiff-oriented numerical path. It selects at most
        ``max_bond`` singular vectors at every cut, with the rank determined
        only by the requested cap and matrix dimensions. Unlike a cutoff
        sweep, it never branches on singular values, so gradients can flow
        through the backend SVD away from repeated-singular-value points.

        The returned object is a ``FirstDegreeMPO`` with neutral virtual
        history metadata: a numerical rank reduction changes the original
        paper-history representation. Use :meth:`to_mpo` for contraction or
        use :meth:`compress_exact` before this method if analytical history
        compression is also desired.
        """
        if not isinstance(max_bond, Integral) or int(max_bond) < 1:
            raise ValueError("max_bond must be a positive integer.")
        max_bond = int(max_bond)
        form = _normalize_tt_svd_form(form)
        initial_bond_dimensions = tuple(self.bond_dimensions)

        if self.L == 1:
            output = self.copy()
            output.metadata.update({
                "operation": "fixed_rank_compression",
                "history_valid": False,
                "max_bond": max_bond,
                "form": form,
            })
            report = MPODifferentiableCompressionReport(
                method="fixed-rank-tt-svd",
                max_bond=max_bond,
                initial_bond_dimensions=initial_bond_dimensions,
                final_bond_dimensions=initial_bond_dimensions,
                truncated=False,
            )
            output.metadata["compression_report"] = report
            output.compression_report = report
            return (output, report) if return_report else output

        source_arrays = self.arrays
        arrays, final_bond_dimensions, _discarded_weights = _tt_svd_arrays(
            source_arrays,
            max_bond,
            form=form,
        )
        output = type(self)(
            arrays,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                "operation": "fixed_rank_compression",
                "history_valid": False,
                "max_bond": max_bond,
                "form": form,
            },
        )
        report = MPODifferentiableCompressionReport(
            method="fixed-rank-tt-svd",
            max_bond=max_bond,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            truncated=final_bond_dimensions != initial_bond_dimensions,
        )
        output.metadata["compression_report"] = report
        output.compression_report = report
        return (output, report) if return_report else output

    def compress_adaptive(
        self,
        max_bond,
        *,
        cutoff=1.0e-10,
        cutoff_mode="rsum2",
        form="left",
        return_report=False,
    ):
        """Compress with a cutoff-aware semantic TT-SVD sweep.

        This is the bounded-assembly counterpart to Quimb's numerical MPO
        compression: it returns a new semantic MPO so another path batch can
        be inserted without first materializing a temporary Quimb MPO. Rank
        selection is a discrete control-flow decision. Singular values are
        read only for that decision and tensor arithmetic remains on the
        source backend, but compiled/JIT traces should use
        :meth:`compress_fixed_rank` instead.
        """
        if not isinstance(max_bond, Integral) or int(max_bond) < 1:
            raise ValueError("max_bond must be a positive integer.")
        max_bond = int(max_bond)
        if self.symmetry is not None:
            raise ValueError(
                "adaptive semantic TT-SVD cannot preserve native symmetry; "
                "use sector-aware Quimb compression instead."
            )
        form = _normalize_tt_svd_form(form)
        cutoff_mode = _normalize_tt_svd_cutoff_mode(cutoff_mode)
        if cutoff is None:
            resolved_cutoff = None
        else:
            resolved_cutoff = _resolve_compression_cutoff(
                cutoff,
                self.arrays[0],
            )
        source_arrays = self.arrays
        initial_bond_dimensions = tuple(self.bond_dimensions)
        if self.L == 1:
            arrays = source_arrays
            final_bond_dimensions = initial_bond_dimensions
            discarded_weights = ()
        else:
            arrays, final_bond_dimensions, discarded_weights = _tt_svd_arrays(
                source_arrays,
                max_bond,
                cutoff=resolved_cutoff,
                cutoff_mode=cutoff_mode,
                form=form,
                adaptive=True,
            )
        output = type(self)(
            arrays,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                "operation": "adaptive_compression",
                "history_valid": False,
                "max_bond": max_bond,
                "cutoff": resolved_cutoff,
                "cutoff_mode": cutoff_mode,
                "form": form,
            },
        )
        report = MPOAdaptiveCompressionReport(
            method="adaptive-tt-svd",
            form=form,
            max_bond=max_bond,
            cutoff=resolved_cutoff,
            cutoff_mode=cutoff_mode,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            truncated=final_bond_dimensions != initial_bond_dimensions,
            discarded_weights=discarded_weights,
        )
        output.metadata["compression_report"] = report
        output.compression_report = report
        return (output, report) if return_report else output

    def compress_to_bond(
        self,
        chi,
        *,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        **compress_opts,
    ):
        """Compress to a requested MPO bond dimension.

        ``chi`` is the final MPO bond cap. It is deliberately separate from
        ``extensive_exponential(max_bond=...)``, whose guard applies only to
        the temporary paper-history representation. The default compression
        is Quimb's cutoff-based sweep for ordinary numerical execution and a
        fixed-rank TT-SVD sweep when ``differentiable=True``.

        The fixed-rank path returns a :class:`FirstDegreeMPO` with invalidated
        analytical histories, while the Quimb path returns an ordinary
        ``MatrixProductOperator``. Numerical compression cannot preserve the
        pre-compression history table. ``sector_aware='auto'`` selects the
        native sector-wise Quimb/Symmray path when available. Fixed-rank
        TT-SVD is intentionally not a sector-aware fallback.
        """
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer.")
        chi = int(chi)
        sector_aware = _normalize_sector_aware_request(sector_aware)
        if compression is None:
            compression = "fixed_rank" if differentiable else "quimb"
        if compression not in {"quimb", "fixed_rank"}:
            raise ValueError(
                "compression must be 'quimb' or 'fixed_rank'."
            )
        if differentiable and compression != "fixed_rank":
            raise ValueError(
                "differentiable=True requires compression='fixed_rank'."
            )
        if compression == "fixed_rank":
            if sector_aware is True:
                raise ValueError(
                    "sector_aware=True is incompatible with "
                    "compression='fixed_rank'; use compression='quimb'."
                )
            if compress_opts:
                unexpected = ", ".join(sorted(compress_opts))
                raise TypeError(
                    "fixed-rank compression does not accept Quimb options: "
                    f"{unexpected}."
                )
            return self.compress_fixed_rank(chi, return_report=return_report)
        return self.compress_numerical(
            max_bond=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            sector_aware=sector_aware,
            return_report=return_report,
            **compress_opts,
        )

    def _exp_with_compression(
        self,
        step,
        *,
        metadata_dt,
        metadata_operation,
        order=1,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
        progress=False,
        **kwargs,
    ):
        """Build ``exp(step * self)`` and optionally compress the result."""
        compress_opts = _normalize_exp_compress_opts(
            compress_opts,
            form=form,
            create_bond=create_bond,
        )
        progress, owns_progress = _make_mpo_progress(
            progress,
            order=order,
            chi=chi,
        )
        if chi is None:
            if compression is not None or compress_opts or sector_aware is True:
                raise ValueError(
                    "compression options require chi; omit them for an "
                    "uncompressed MPO."
                )
            if return_report:
                raise ValueError("return_report requires chi compression.")
            result = self.extensive_exponential(
                step,
                order=order,
                progress=progress,
                **kwargs,
            )
            result.metadata.update({
                "operation": metadata_operation,
                "dt": metadata_dt,
                "exponent": step,
                "numerical_compression": "none",
            })
            if owns_progress:
                progress.close()
            return result

        output = self.extensive_exponential(
            step,
            order=order,
            progress=progress,
            **kwargs,
        )
        numerical_compression = (
            "fixed_rank" if differentiable and compression is None
            else compression or "quimb"
        )
        chi_stage = f"chi-compress (chi={int(chi)})"
        progress.start(chi_stage)
        compressed = output.compress_to_bond(
            chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            **compress_opts,
        )
        if return_report:
            result, report = compressed
        else:
            result, report = compressed, None
        final_bond_dimensions = (
            tuple(int(size) for size in result.bond_sizes())
            if hasattr(result, "bond_sizes")
            else tuple(result.bond_dimensions)
        )
        progress.finish(
            chi_stage,
            bond_dimensions=final_bond_dimensions,
            chi=int(chi),
        )
        exponential_metadata = {
            "operation": metadata_operation,
            "dt": metadata_dt,
            "exponent": step,
            "order": order,
            "chi": int(chi),
            "compression": (
                "fixed_rank" if differentiable and compression is None
                else compression or "quimb"
            ),
            "numerical_compression": numerical_compression,
            "chi_compression": int(chi),
            "differentiable": bool(differentiable),
            "sector_aware": (
                getattr(report, "sector_aware", False)
                if report is not None
                else getattr(
                    result,
                    "pepsy_sector_compression_metadata",
                    {},
                ).get("sector_aware", False)
            ),
        }
        if progress.enabled:
            exponential_metadata["progress"] = True
            exponential_metadata["timings"] = dict(progress.timings)
            exponential_metadata["timing_history"] = {
                int(order): dict(progress.timings),
            }
            exponential_metadata["order_seconds"] = progress.total_seconds
        for key in (
            "mode",
            "history_storage",
            "history_cache_hit",
            "tensor_plan_cache_hit",
            "compression_plan_cache_hit",
            "extension_plan_cache_hit",
            "approximation_plan_cache_hit",
            "analytical_compression",
            "requested_mode",
            "estimated_extension_terms",
            "extension_budget",
            "mode_reason",
        ):
            if key in output.metadata:
                exponential_metadata[key] = output.metadata[key]
        if output.compression_report is not None:
            exponential_metadata["analytical_compression_report"] = (
                output.compression_report
            )
        if isinstance(result, FirstDegreeMPO):
            result.metadata.update(exponential_metadata)
        else:
            setattr(result, "pepsy_exp_metadata", exponential_metadata)
            if metadata_operation == "time_evolution":
                # Keep the attribute used by the original real-time API.
                result.pepsy_evolution_metadata = exponential_metadata
        if owns_progress:
            progress.close()
        return (result, report) if return_report else result

    def exp(
        self,
        step=None,
        *,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
    ):
        """Build ``exp(step * self)`` with optional final compression.

        Parameters
        ----------
        step : scalar
            Actual scalar in the exponential. Use ``-1j * tau`` for
            real-time evolution and ``-beta`` for imaginary time.
        dt : scalar, optional
            Compatibility spelling for ``step``. Supplying both raises.
        order, mode, max_bond, on_exceed, cache_history, history_storage
            Controls for the analytical higher-order history construction.
        chi : int, optional
            Separate final numerical MPO bond cap. It is not the temporary
            history-bond guard ``max_bond``.
        differentiable : bool, optional
            With ``chi``, select fixed-rank autodiff compression instead of a
            value-dependent numerical cutoff.
        form, create_bond : optional
            Quimb numerical-compression controls used after the analytical
            higher-order MPO is built.
        compress_opts : mapping, optional
            Additional Quimb compression keywords such as ``method``,
            ``absorb``, ``renorm``, or ``info``.
        sector_aware : {True, False, "auto"}, default="auto"
            Use and report native Abelian sector-wise compression when the
            compiled MPO is backed by Symmray. This never densifies a native
            symmetric MPO; ``True`` errors if that boundary is unavailable.
        progress : bool, default=False
            Show construction stages with elapsed seconds and current MPO
            bond dimensions.
        """
        step = _resolve_exp_step(step, dt)
        return self._exp_with_compression(
            step,
            metadata_dt=step,
            metadata_operation="exp",
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
            progress=progress,
        )

    def time_evolution(
        self,
        dt,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
        chi=None,
        cutoff=1.0e-10,
        cutoff_mode="rel",
        compression=None,
        differentiable=False,
        return_report=False,
        sector_aware="auto",
        form=None,
        create_bond=False,
        compress_opts=None,
    ):
        """Build real-time ``exp(-1j * dt * self)``.

        This is a convenience wrapper around :meth:`exp`; use :meth:`exp`
        directly when the exponential step should be explicit.
        """
        _check_scalar(dt, name="dt")
        return self._exp_with_compression(
            -1j * dt,
            metadata_dt=dt,
            metadata_operation="time_evolution",
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
            chi=chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression=compression,
            differentiable=differentiable,
            return_report=return_report,
            sector_aware=sector_aware,
            form=form,
            create_bond=create_bond,
            compress_opts=compress_opts,
            progress=progress,
        )

    def exp_arrays(
        self,
        step=None,
        *,
        dt=None,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        extension_budget=None,
    ):
        """Return ``exp(step * self)`` tensors without a Quimb wrapper.

        This is the low-level numerical interface for compiled optimization
        loops. It returns the normalized ``(left, right, up, down)`` tensor
        tuple directly. The backend values remain connected to ``step`` and
        the Hamiltonian coefficients for autodiff.
        """
        step = _resolve_exp_step(step, dt)
        return self.exp(
            step,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
        ).arrays

    def time_evolution_arrays(
        self,
        dt,
        *,
        order=1,
        mode=None,
        extend=False,
        approximate=False,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        extension_budget=None,
    ):
        """Return real-time evolution tensors without a semantic wrapper."""
        return self.time_evolution(
            dt,
            order=order,
            mode=mode,
            extend=extend,
            approximate=approximate,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            history_storage=history_storage,
            extension_budget=extension_budget,
        ).arrays

    def expectation(self, mps, *, contraction_opt=None):
        """Evaluate ``<mps|self|mps>`` through Pepsy's MPS contraction API."""
        from pepsy.tensors import expec_mpo  # pylint: disable=import-outside-toplevel

        # Quimb's contraction backend requires every operand to use one
        # array backend.  A parameterized observable has a Torch/JAX MPO while
        # a convenient product-state constructor often returns NumPy tensors;
        # align that fixed state data to the observable backend without
        # touching the observable's differentiable tensors.
        local_values = (
            value
            for array in self._arrays
            for value in (
                array.blocks.values()
                if isinstance(array, SparseVirtualTensor)
                else (array,)
            )
        )
        reference = next(
            (
                value
                for value in local_values
                if _backend_name(value) not in {"builtins", "numpy"}
            ),
            None,
        )
        if reference is not None and hasattr(mps, "copy"):
            mps = mps.copy()
            for tensor in getattr(mps, "tensors", ()):
                data = _as_backend(
                    tensor.data,
                    like=reference,
                    dtype=getattr(reference, "dtype", None),
                )
                if data is not tensor.data:
                    tensor.modify(data=data)

        return expec_mpo(
            self.to_mpo(),
            mps,
            contraction_opt=contraction_opt,
        )

    def scale(self, coefficient):
        """Return ``coefficient * self`` by scaling one boundary tensor."""
        _check_scalar(coefficient, name="coefficient")
        arrays = list(self.arrays)
        arrays[0] = _multiply_scalar(coefficient, arrays[0])
        out = type(self)(
            arrays,
            levels=self.levels,
            degree=self.degree,
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={**self.metadata, "scale": coefficient},
        )
        out._structural_transitions = self._structural_transitions
        return out

    def add(self, other):
        """Return the exact direct sum ``self + other``."""
        self._check_compatible(other)
        self_arrays = self.arrays
        other_arrays = other.arrays
        if self.L == 1:
            arrays = (self_arrays[0] + other_arrays[0],)
            levels = [[self._levels[0][0]], [self._levels[1][0]]]
        else:
            arrays = []
            levels = [[self._levels[0][0]]]
            first = _concat((self_arrays[0], other_arrays[0]), axis=1)
            arrays.append(first)
            for site in range(1, self.L - 1):
                left, right = self_arrays[site], other_arrays[site]
                top = _concat(
                    (
                        left,
                        _zeros((left.shape[0], right.shape[1], *left.shape[2:]), like=left),
                    ),
                    axis=1,
                )
                bottom = _concat(
                    (
                        _zeros((right.shape[0], left.shape[1], *right.shape[2:]), like=right),
                        right,
                    ),
                    axis=1,
                )
                arrays.append(_concat((top, bottom), axis=0))
            arrays.append(_concat((self_arrays[-1], other_arrays[-1]), axis=0))
            for bond in range(1, self.L):
                levels.append([
                    *self._levels[bond],
                    *other._levels[bond],
                ])
            levels.append([self._levels[-1][0]])

        return type(self)(
            arrays,
            levels=levels,
            degree=max(self.degree, other.degree),
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": "add"},
        )

    def _add_path_cores(self, path_cores):
        """Add one open-boundary path without constructing a path MPO.

        ``path_cores`` contains full-chain ``(left, right, up, down)`` cores
        with singleton boundary bonds. This is the memory-bounded companion
        to :meth:`add`: the path is inserted directly into the accumulator's
        virtual direct sum, so no temporary :class:`FirstDegreeMPO` and no
        batch block-diagonal tensor are required.
        """
        return self._add_path_cores_batch((path_cores,))

    def _add_path_cores_batch(self, paths):
        """Insert several open-boundary paths in one accumulator allocation.

        The result is the direct sum of ``self`` and all supplied paths, but
        unlike repeated :meth:`add` calls it allocates the expanded
        accumulator only once. Each path remains a tuple of local cores; no
        temporary batch ``FirstDegreeMPO`` or batch block-diagonal MPO is
        constructed.
        """
        normalized_paths = []
        for path in paths:
            path = tuple(path)
            if len(path) != self.L:
                raise ValueError(f"path_cores must have length {self.L}.")
            path = tuple(
                _as_4d(array, site=site, length=self.L)
                for site, array in enumerate(path)
            )
            if path[0].shape[0] != 1 or path[-1].shape[1] != 1:
                raise ValueError("path_cores must have singleton boundary bonds.")
            if any(
                core.shape[-2:] != (self.phys_dim, self.phys_dim)
                for core in path
            ):
                raise ValueError(
                    "path physical dimensions must match the accumulator."
                )
            normalized_paths.append(path)
        normalized_paths = tuple(normalized_paths)
        if not normalized_paths:
            return self.copy()

        self_arrays = self.arrays
        if self.L == 1:
            result = self_arrays[0]
            for path in normalized_paths:
                result = result + path[0]
            arrays = (result,)
        else:
            arrays = [
                _concat(
                    (self_arrays[0], *(path[0] for path in normalized_paths)),
                    axis=1,
                ),
            ]
            for site in range(1, self.L - 1):
                left = self_arrays[site]
                path_cores = tuple(path[site] for path in normalized_paths)
                path_left_offsets = []
                path_right_offsets = []
                left_offset = 0
                right_offset = 0
                for path in path_cores:
                    path_left_offsets.append(left_offset)
                    path_right_offsets.append(right_offset)
                    left_offset += int(path.shape[0])
                    right_offset += int(path.shape[1])
                path_block = _zeros(
                    (left_offset, right_offset, *left.shape[2:]),
                    like=left,
                )
                rows = []
                columns = []
                values = []
                for path, left_start, right_start in zip(
                    path_cores,
                    path_left_offsets,
                    path_right_offsets,
                ):
                    path_left, path_right = int(path.shape[0]), int(path.shape[1])
                    rows.append(
                        np.repeat(
                            np.arange(path_left, dtype=int) + left_start,
                            path_right,
                        )
                    )
                    columns.append(
                        np.tile(
                            np.arange(path_right, dtype=int) + right_start,
                            path_left,
                        )
                    )
                    values.append(
                        ar.do(
                            "reshape",
                            path,
                            (path_left * path_right, self.phys_dim, self.phys_dim),
                        )
                    )
                path_block = _scatter_add_2d(
                    path_block,
                    np.concatenate(rows),
                    np.concatenate(columns),
                    ar.do("concatenate", tuple(values), axis=0),
                )
                top = _concat(
                    (
                        left,
                        _zeros((left.shape[0], right_offset, *left.shape[2:]), like=left),
                    ),
                    axis=1,
                )
                bottom = _concat(
                    (
                        _zeros((left_offset, left.shape[1], *left.shape[2:]), like=left),
                        path_block,
                    ),
                    axis=1,
                )
                arrays.append(_concat((top, bottom), axis=0))
            arrays.append(
                _concat(
                    (self_arrays[-1], *(path[-1] for path in normalized_paths)),
                    axis=0,
                )
            )

        return type(self)(
            arrays,
            degree=self.degree,
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": "add_path", "history_valid": False},
        )

    def product(self, other, *, kind="ordinary"):
        """Return an exact virtual-space product of two MPO expressions.

        ``kind`` is provenance metadata only; it does not change the
        multiplication.  The tensor product is exact and keeps all paths.
        The paper-specific extensive-path filtering is intentionally applied
        later by the Taylor builder so this foundational algebra remains
        predictable.

        The explicit local loops are a deliberate tensor-network choice: they
        multiply physical blocks site by site and retain histories, rather
        than converting either MPO to a global matrix.  A future optimized
        implementation can replace the loops with a backend-aware batched
        kernel without changing this semantic contract.
        """
        self._check_compatible(other)
        arrays = []
        levels = []
        for site, (left, right) in enumerate(zip(self.arrays, other.arrays)):
            # Pairing virtual states explicitly is more verbose than calling
            # Quimb's generic MPO product, but it preserves the symbolic
            # history needed by the paper's later exact rewiring steps.
            Dl1, Dr1, d, _ = left.shape
            Dl2, Dr2, _, _ = right.shape
            rows = []
            for left_pos in range(Dl1):
                for right_pos in range(Dl2):
                    blocks = []
                    for left_next in range(Dr1):
                        for right_next in range(Dr2):
                            blocks.append(
                                ar.do(
                                    "matmul",
                                    left[left_pos, left_next],
                                    right[right_pos, right_next],
                                )
                            )
                    rows.append(_stack(blocks, axis=0).reshape(Dr1 * Dr2, d, d))
            arrays.append(_stack(rows, axis=0).reshape(Dl1 * Dl2, Dr1 * Dr2, d, d))
            levels.append([
                MPOLevel(
                    ("product", a.label, b.label),
                    a.history + b.history,
                    charge=(a.charge, b.charge),
                )
                for a in self._levels[site]
                for b in other._levels[site]
            ])
        levels.append([
            MPOLevel(
                ("product", a.label, b.label),
                a.history + b.history,
                charge=(a.charge, b.charge),
            )
            for a in self._levels[-1]
            for b in other._levels[-1]
        ])
        return type(self)(
            arrays,
            levels=levels,
            degree=self.degree + other.degree,
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={"operation": kind},
        )

    def non_disjoint_product(self, other):
        """Return the exact raw product used as the algebraic input.

        The result retains the level histories and all connected and
        disconnected paths.  The later Taylor construction will apply the
        paper's level rewiring and exact-compression rules; keeping this
        operation explicit prevents accidental use of generic Quimb MPO
        multiplication in that implementation.  The name records that no
        support analysis or connected-term filtering has happened yet.
        """
        return self.product(other, kind="non_disjoint")

    def disjoint_product(self, other):
        """Return an explicitly labelled exact product of two expressions.

        ``disjoint`` is currently a provenance label, not an assertion or an
        automatic support decomposition.  Future extensive builders can use
        this hook for overlap-aware channel pruning once that analysis is
        implemented.
        """
        return self.product(other, kind="disjoint")

    def commutator(self, other):
        """Return the exact commutator ``self @ other - other @ self``."""
        return self.non_disjoint_product(other).add(
            other.non_disjoint_product(self).scale(-1)
        )

    def power(self, exponent):
        """Return an exact non-negative integer power."""
        if not isinstance(exponent, Integral) or int(exponent) < 0:
            raise ValueError("exponent must be a non-negative integer.")
        exponent = int(exponent)
        if exponent == 0:
            return type(self).identity(
                self.L,
                self.phys_dim,
                like=self.arrays[0],
                **self._symmetry_options(),
                upper_ind_id=self.upper_ind_id,
                lower_ind_id=self.lower_ind_id,
                site_tag_id=self.site_tag_id,
            )
        result = self.copy()
        for _ in range(1, exponent):
            result = result.non_disjoint_product(self)
        return result

    def power_raw(self, exponent):
        """Return an exact MPO power without forming a global dense operator.

        The virtual indices are paired at every site and the physical blocks
        are multiplied locally.  The resulting histories therefore retain the
        factor order required by the higher-order MPO construction.

        This compatibility spelling currently delegates to :meth:`power`.
        It remains explicit because callers working from the paper often need
        to distinguish a raw power from a later history-rewired power.
        """
        return self.power(exponent)

    def _history_schemas(self):
        """Return the base-level schema used at every raw-history cut."""
        return tuple(
            self._levels[1]
            if bond == 0
            else self._levels[-2]
            if bond == self.L
            else self._levels[bond]
            for bond in range(self.L + 1)
        )

    def _history_symbolic_data(self):
        """Build structural history lookup tables once per MPO topology.

        The history algorithms used to resolve a token by scanning a whole
        level list and to test reachability by walking all earlier sites for
        every candidate.  Both operations are structural.  This table turns
        them into dictionary lookups and one precomputed forward reachability
        walk.  It is safe to retain across parameter rebinding because it
        contains no numerical tensor or backend scalar.
        """
        if self._history_symbolic_cache is not None:
            return self._history_symbolic_cache

        schemas = self._history_schemas()
        token_levels = tuple(
            {
                level.history: level
                for level in schema
                if len(level.history) == 1
            }
            for schema in schemas
        )
        reachable_labels = None
        if self._structural_transitions is not None:
            start_labels = frozenset(
                level.label
                for level in schemas[0]
                if _level_number(level.history[0]) == 1
            )
            reachable = [start_labels]
            for edges in self._structural_transitions:
                reachable.append(frozenset(
                    right
                    for left, right in edges
                    if left in reachable[-1]
                ))
            reachable_labels = tuple(reachable)

        self._history_symbolic_cache = {
            "schemas": schemas,
            "token_levels": token_levels,
            "reachable_labels": reachable_labels,
            "inserted_states": {},
        }
        return self._history_symbolic_cache

    def _history_inserted_states(self, bond, history, token):
        """Return reachable factor states after inserting one token.

        Equal insertions (for example inserting a level-1 token next to an
        existing level-1 token) are grouped with a multiplicity.  This is the
        same sum as the paper's insertion loop, but avoids evaluating the
        same local product repeatedly.
        """
        symbolic = self._history_symbolic_data()
        key = (bond, tuple(history), token)
        cached = symbolic["inserted_states"].get(key)
        if cached is not None:
            return cached

        grouped = {}
        for position in range(len(history) + 1):
            extended = (
                history[:position]
                + (token,)
                + history[position:]
            )
            state = self._history_tokens_reachable(
                symbolic["schemas"],
                bond,
                extended,
                {},
            )
            if state is None:
                continue
            if extended in grouped:
                old_state, multiplicity = grouped[extended]
                grouped[extended] = (old_state, multiplicity + 1)
            else:
                grouped[extended] = (state, 1)

        result = tuple(grouped.values())
        symbolic["inserted_states"][key] = result
        return result

    @staticmethod
    def _history_level_positions(levels):
        """Return the current virtual positions indexed by history."""
        return {
            level.history: position
            for position, level in enumerate(levels)
        }

    def _base_level_positions(self):
        """Cache label/history positions for the original first-degree MPO."""
        if self._base_level_position_cache is None:
            self._base_level_position_cache = tuple(
                {
                    "label": {
                        level.label: position
                        for position, level in enumerate(levels)
                    },
                    "history": {
                        level.history: position
                        for position, level in enumerate(levels)
                    },
                    "number": {
                        number: position
                        for position, level in enumerate(levels)
                        for number in [_level_number(level.history[0])]
                        if number in (1, 3)
                    },
                }
                for levels in self._levels
            )
        return self._base_level_position_cache

    def _structural_transitions_from_automaton(self, automaton):
        """Map automaton edges onto the raw-history boundary schemas."""
        transitions = []
        schemas = self._history_schemas()
        for site in range(self.L):
            left_schema = schemas[site]
            right_schema = schemas[site + 1]
            left_labels = {level.label: level for level in left_schema}
            right_labels = {level.label: level for level in right_schema}
            edges = set()
            for transition, _implicit in automaton._materialized_transitions(site):
                if (
                    transition.left_state in left_labels
                    and transition.right_state in right_labels
                ):
                    edges.add((transition.left_state, transition.right_state))

            # The finite Hamiltonian's last tensor is right-boundary
            # contracted onto level 3, while the transformed evolution MPO
            # uses an all-one right boundary. The batched local-product
            # executor supplies this synthetic identity edge, so include it
            # in the structural graph as well.
            if site == self.L - 1:
                start_level = next(
                    level
                    for level in left_schema
                    if _level_number(level.history[0]) == 1
                )
                if start_level.label in right_labels:
                    edges.add((start_level.label, start_level.label))
            transitions.append(frozenset(edges))
        return tuple(transitions)

    def _reachable_history_states(
        self,
        schemas,
        exponent,
        *,
        max_bond=None,
        on_exceed="raise",
        progress=None,
    ):
        """Generate only raw histories reachable from the left boundary.

        The previous reference implementation materialized every Cartesian
        product at every cut, including channels that cannot be reached from
        the finite-chain all-one left boundary.  This forward graph walk keeps
        the same factor ordering and level metadata while avoiding those dead
        states.  Direct array construction, which has no source automaton,
        falls back to the complete local schema for compatibility.
        """
        start_levels = tuple(
            level
            for level in schemas[0]
            if _level_number(level.history[0]) == 1
        )
        if len(start_levels) != 1:
            raise ValueError(
                "history construction requires one level-1 starting channel."
            )
        warned = False

        def check_bond_limit(count, site):
            nonlocal warned
            if max_bond is None or count <= max_bond or on_exceed == "ignore":
                return
            message = (
                "extensive_exponential history bond dimension "
                f"{count} exceeds max_bond={max_bond} after site {site}."
            )
            if on_exceed == "raise":
                raise MemoryError(message)
            if not warned:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                warned = True

        state_lists = [
            (tuple(start_levels[0] for _ in range(exponent)),)
        ]

        for site in range(self.L):
            right_schema = schemas[site + 1]
            edges = (
                None
                if self._structural_transitions is None
                else self._structural_transitions[site]
            )
            next_states = []
            seen = set()
            for left_state in state_lists[-1]:
                options = []
                for left_level in left_state:
                    if edges is None:
                        level_options = right_schema
                    else:
                        level_options = tuple(
                            level
                            for level in right_schema
                            if (left_level.label, level.label) in edges
                        )
                    if not level_options:
                        break
                    options.append(level_options)
                else:
                    for right_state in product(*options):
                        key = tuple(level.label for level in right_state)
                        if key not in seen:
                            seen.add(key)
                            next_states.append(right_state)
                            check_bond_limit(len(next_states), site)
            if not next_states:
                raise ValueError(
                    f"raw-history construction found no reachable states after "
                    f"site {site}."
                )
            state_lists.append(tuple(next_states))
            if progress is not None:
                progress.detail(
                    f"history topology {site + 1}/{self.L}",
                    bond_dimensions=tuple(
                        len(states) for states in state_lists[1:-1]
                    ),
                )
        return state_lists

    def _history_topology(
        self,
        exponent,
        *,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        progress=None,
    ):
        """Return cached raw history states and whether the cache was used.

        The state graph is structural: it depends on the base MPO channels,
        not on ``dt`` or on the numerical values in the local operator
        blocks.  Generate it without a limit on the first request, then
        apply the caller's per-request safety policy to the cached result.
        This keeps a failed small ``max_bond`` request from poisoning the
        reusable topology cache.
        """
        exponent = int(exponent)
        state_lists = (
            self._history_topology_cache.get(exponent)
            if cache_history
            else None
        )
        cache_hit = state_lists is not None
        guarded_generation = False
        if state_lists is None:
            guarded_generation = max_bond is not None and on_exceed != "ignore"
            state_lists = self._reachable_history_states(
                self._history_schemas(),
                exponent,
                max_bond=max_bond if guarded_generation else None,
                on_exceed=on_exceed if guarded_generation else "ignore",
                progress=progress,
            )
        if cache_history and not cache_hit:
            self._history_topology_cache[exponent] = state_lists

        # A first-time guarded generation already enforced the policy while
        # walking the graph. Rechecking it would duplicate warnings; cached
        # topologies still need the caller's per-request policy applied here.
        if guarded_generation:
            return state_lists, cache_hit

        warned = False
        if max_bond is not None and on_exceed != "ignore":
            for site, states in enumerate(state_lists[1:]):
                count = len(states)
                if count <= max_bond:
                    continue
                message = (
                    "extensive_exponential history bond dimension "
                    f"{count} exceeds max_bond={max_bond} after site {site}."
                )
                if on_exceed == "raise":
                    raise MemoryError(message)
                if not warned:
                    warnings.warn(message, RuntimeWarning, stacklevel=3)
                    warned = True
        return state_lists, cache_hit

    def _history_levels_for_states(self, state_lists, exponent):
        """Convert raw state tuples into the public level metadata."""
        return [
            [
                MPOLevel(
                    ("raw-history", exponent, bond, pos),
                    tuple(
                        token
                        for factor in state
                        for token in factor.history
                    ),
                    charge=tuple(factor.charge for factor in state),
                )
                for pos, state in enumerate(states)
            ]
            for bond, states in enumerate(state_lists)
        ]

    def _history_state_step(
        self,
        schemas,
        left_states,
        site,
        *,
        max_bond,
        on_exceed,
        warned,
    ):
        """Advance one raw-history cut without retaining earlier cuts."""
        right_schema = schemas[site + 1]
        edges = (
            None
            if self._structural_transitions is None
            else self._structural_transitions[site]
        )
        next_states = []
        seen = set()
        for left_state in left_states:
            options = []
            for left_level in left_state:
                if edges is None:
                    level_options = right_schema
                else:
                    level_options = tuple(
                        level
                        for level in right_schema
                        if (left_level.label, level.label) in edges
                    )
                if not level_options:
                    break
                options.append(level_options)
            else:
                for right_state in product(*options):
                    key = tuple(level.label for level in right_state)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_states.append(right_state)
                    if max_bond is not None and len(next_states) > max_bond:
                        message = (
                            "extensive_exponential history bond dimension "
                            f"{len(next_states)} exceeds max_bond={max_bond} "
                            f"after site {site}."
                        )
                        if on_exceed == "raise":
                            raise MemoryError(message)
                        if on_exceed == "warn" and not warned[0]:
                            warnings.warn(message, RuntimeWarning, stacklevel=3)
                            warned[0] = True
        if not next_states:
            raise ValueError(
                f"raw-history construction found no reachable states after "
                f"site {site}."
            )
        return tuple(next_states)

    def _history_transition_allowed(self, site, left_state, right_state):
        """Return whether a compound history has a structural local edge."""
        if self._structural_transitions is None:
            return True
        edges = self._structural_transitions[site]
        return all(
            (left_level.label, right_level.label) in edges
            for left_level, right_level in zip(left_state, right_state)
        )

    def _history_local_position_arrays(self, site, left_states, right_states):
        """Resolve all base virtual positions for one local history batch.

        The old history builder resolved a base block independently for every
        pair of compound histories.  The lookup is structural, so resolve it
        once per factor and gather all physical blocks in one backend call.
        ``-1`` denotes the edge padding handled by
        :meth:`_history_local_product_batch_values`.
        """
        base_positions = self._base_level_positions()

        def resolve(bond, level):
            positions = base_positions[bond]
            position = positions["label"].get(level.label)
            if position is None:
                position = positions["history"].get(level.history)
            return -1 if position is None else position

        order = len(left_states[0])
        left_positions = tuple(
            np.asarray(
                [resolve(site, state[factor]) for state in left_states],
                dtype=int,
            )
            for factor in range(order)
        )
        right_positions = tuple(
            np.asarray(
                [resolve(site + 1, state[factor]) for state in right_states],
                dtype=int,
            )
            for factor in range(order)
        )
        left_numbers = tuple(
            np.asarray(
                [
                    _level_number(state[factor].history[0])
                    for state in left_states
                ],
                dtype=int,
            )
            for factor in range(order)
        )
        right_numbers = tuple(
            np.asarray(
                [
                    _level_number(state[factor].history[0])
                    for state in right_states
                ],
                dtype=int,
            )
            for factor in range(order)
        )
        return (
            left_positions,
            right_positions,
            left_numbers,
            right_numbers,
        )

    def _history_allowed_pairs(self, site, left_states, right_states, *, sparse):
        """Return virtual pairs with a nonzero structural local product."""
        if not sparse or self._structural_transitions is None:
            left = np.repeat(
                np.arange(len(left_states), dtype=int),
                len(right_states),
            )
            right = np.tile(
                np.arange(len(right_states), dtype=int),
                len(left_states),
            )
            return left, right

        left = []
        right = []
        for left_pos, left_state in enumerate(left_states):
            for right_pos, right_state in enumerate(right_states):
                if self._history_transition_allowed(
                    site, left_state, right_state,
                ):
                    left.append(left_pos)
                    right.append(right_pos)
        return np.asarray(left, dtype=int), np.asarray(right, dtype=int)

    def _history_tensor_execution_plan(
        self,
        exponent,
        state_lists,
        *,
        sparse,
        cache_history,
    ):
        """Compile local gather metadata for one raw-history order.

        History topology caching used to leave one structural cost on every
        numerical evaluation: each site recomputed allowed pairs, base-level
        positions, and identity-rail metadata before doing the actual local
        products.  This plan contains only NumPy integer/boolean arrays and
        can therefore be shared safely by parameterized Torch and JAX calls.
        """
        key = (int(exponent), bool(sparse))
        if cache_history:
            cached = self._history_tensor_plan_cache.get(key)
            if cached is not None:
                return cached, True

        sites = []
        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            left_indices, right_indices = self._history_allowed_pairs(
                site,
                left_states,
                right_states,
                sparse=sparse,
            )
            sites.append({
                "left_indices": left_indices,
                "right_indices": right_indices,
                "positions": self._history_local_position_arrays(
                    site,
                    left_states,
                    right_states,
                ),
                "total_blocks": len(left_states) * len(right_states),
            })

        plan = tuple(sites)
        if cache_history:
            self._history_tensor_plan_cache[key] = plan
        return plan, False

    def _history_local_product_batch_values(
        self,
        site,
        positions,
        left_indices,
        right_indices,
    ):
        """Evaluate many history products with batched physical matmuls."""
        array = self._arrays[site]
        left_positions, right_positions, left_numbers, right_numbers = positions
        reference = array[0, 0]
        product_block = None
        for factor in range(len(left_positions)):
            local_left = left_positions[factor][left_indices]
            local_right = right_positions[factor][right_indices]
            left_valid = local_left >= 0
            right_valid = local_right >= 0
            safe_left = np.where(left_valid, local_left, 0)
            safe_right = np.where(right_valid, local_right, 0)
            local = _backend_index_pairs(array, safe_left, safe_right)
            valid = _as_backend(left_valid & right_valid, like=local)
            local = ar.do(
                "where",
                valid[..., None, None],
                local,
                ar.do("zeros_like", local),
            )

            # The finite Hamiltonian's final tensor is contracted onto the
            # level-3 rail.  History products use the all-one boundary, so the
            # missing (1 -> 1) edge is the identity rather than zero.
            synthetic = (
                left_numbers[factor][left_indices] == 1
            ) & (
                right_numbers[factor][right_indices] == 1
            ) & ~right_valid if site == self.L - 1 else np.zeros(
                len(left_indices),
                dtype=bool,
            )
            if np.any(synthetic):
                synthetic = _as_backend(synthetic, like=local)
                local = ar.do(
                    "where",
                    synthetic[..., None, None],
                    ar.do("eye", self.phys_dim, like=reference),
                    local,
                )
            product_block = (
                local
                if product_block is None
                else ar.do("matmul", product_block, local)
            )
        return product_block

    def _history_tensor_batch(
        self,
        site,
        left_states,
        right_states,
        *,
        sparse,
        chunk_size=65536,
        execution_plan=None,
    ):
        """Build one history tensor without dispatching per virtual pair."""
        if execution_plan is None:
            left_indices, right_indices = self._history_allowed_pairs(
                site,
                left_states,
                right_states,
                sparse=sparse,
            )
            positions = self._history_local_position_arrays(
                site,
                left_states,
                right_states,
            )
        else:
            left_indices = execution_plan["left_indices"]
            right_indices = execution_plan["right_indices"]
            positions = execution_plan["positions"]
        reference = self._arrays[site]
        result = _zeros(
            (
                len(left_states),
                len(right_states),
                *reference.shape[-2:],
            ),
            like=reference,
        )
        if not len(left_indices):
            return result, 0, len(left_states) * len(right_states)

        for start in range(0, len(left_indices), chunk_size):
            stop = start + chunk_size
            values = self._history_local_product_batch_values(
                site,
                positions,
                left_indices[start:stop],
                right_indices[start:stop],
            )
            result = _scatter_set_2d(
                result,
                left_indices[start:stop],
                right_indices[start:stop],
                values,
            )
        return result, len(left_indices), len(left_states) * len(right_states)

    def _batched_history_power_data(
        self,
        exponent,
        *,
        state_lists,
        storage_mode,
        cache_hit,
        execution_plan=None,
        tensor_plan_cache_hit=False,
        progress=None,
    ):
        """Build raw history tensors using batched local block products."""
        levels = self._history_levels_for_states(state_lists, exponent)
        arrays = []
        total_blocks = 0
        stored_blocks = 0
        sparse = storage_mode == "sparse"
        for site in range(self.L):
            array, stored, total = self._history_tensor_batch(
                site,
                state_lists[site],
                state_lists[site + 1],
                sparse=sparse,
                execution_plan=(
                    None
                    if execution_plan is None
                    else execution_plan[site]
                ),
            )
            arrays.append(array)
            stored_blocks += stored
            total_blocks += total
            if progress is not None:
                progress.detail(
                    f"history tensors {site + 1}/{self.L}",
                    bond_dimensions=tuple(
                        len(states) for states in state_lists[1:-1]
                    ),
                    blocks=stored_blocks,
                )
        storage_info = {
            "mode": storage_mode,
            "stored_blocks": stored_blocks,
            "total_blocks": total_blocks,
            "tensor_plan_cache_hit": bool(tensor_plan_cache_hit),
        }
        return arrays, levels, cache_hit, storage_info

    def _block_sparse_history_power_data(
        self,
        exponent,
        *,
        state_lists,
        cache_hit,
        execution_plan=None,
        tensor_plan_cache_hit=False,
        chunk_size=65536,
        progress=None,
    ):
        """Build raw histories as sparse matrices of local operators.

        Only structurally allowed virtual transitions are stored. Unlike the
        older ``history_storage="sparse"`` execution policy, this path keeps
        that sparsity through Algorithms 1--4 and into the returned semantic
        MPO rather than scattering into a dense virtual array at every site.
        """
        levels = self._history_levels_for_states(state_lists, exponent)
        arrays = []
        total_blocks = 0
        stored_blocks = 0
        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            if execution_plan is None:
                left_indices, right_indices = self._history_allowed_pairs(
                    site,
                    left_states,
                    right_states,
                    sparse=True,
                )
                positions = self._history_local_position_arrays(
                    site,
                    left_states,
                    right_states,
                )
            else:
                site_plan = execution_plan[site]
                left_indices = site_plan["left_indices"]
                right_indices = site_plan["right_indices"]
                positions = site_plan["positions"]

            source = self._arrays[site]
            reference = (
                source._like
                if isinstance(source, SparseVirtualTensor)
                else source[0, 0]
            )
            tensor = SparseVirtualTensor((
                len(left_states),
                len(right_states),
                self.phys_dim,
                self.phys_dim,
            ), like=reference)
            for start in range(0, len(left_indices), chunk_size):
                stop = start + chunk_size
                values = self._history_local_product_batch_values(
                    site,
                    positions,
                    left_indices[start:stop],
                    right_indices[start:stop],
                )
                tensor = tensor.scatter_add(
                    left_indices[start:stop],
                    right_indices[start:stop],
                    values,
                )
            arrays.append(tensor)
            stored_blocks += tensor.stored_blocks
            total_blocks += len(left_states) * len(right_states)
            if progress is not None:
                progress.detail(
                    f"history blocks {site + 1}/{self.L}",
                    bond_dimensions=tuple(
                        len(states) for states in state_lists[1:-1]
                    ),
                    blocks=stored_blocks,
                )

        storage_info = {
            "mode": "block_sparse",
            "stored_blocks": stored_blocks,
            "total_blocks": total_blocks,
            "tensor_plan_cache_hit": bool(tensor_plan_cache_hit),
            "materialized_dense_virtual_tensors": False,
        }
        return arrays, levels, cache_hit, storage_info

    def _stream_history_power_data(
        self,
        exponent,
        *,
        schemas,
        max_bond,
        on_exceed,
        sparse,
        progress=None,
    ):
        """Build non-cached history tensors with batched local products.

        The final MPO tensors still have to be materialized for Algorithms
        1--4, but this path does not retain the topology cache. Its sparse
        variant also avoids structurally impossible local block products.
        The resulting virtual arrays are still dense because Algorithms 1--4
        operate on those arrays.
        """
        state_lists = self._reachable_history_states(
            schemas,
            exponent,
            max_bond=max_bond,
            on_exceed=on_exceed,
            progress=progress,
        )
        return self._batched_history_power_data(
            exponent,
            state_lists=state_lists,
            storage_mode="sparse" if sparse else "streaming",
            cache_hit=False,
            progress=progress,
        )

    @property
    def history_cache_info(self):
        """Describe cached raw history topologies without exposing arrays."""
        return {
            "orders": tuple(sorted(self._history_topology_cache)),
            "bond_dimensions": {
                order: tuple(len(states) for states in state_lists[1:-1])
                for order, state_lists in self._history_topology_cache.items()
            },
            "compression_plan_orders": tuple(
                sorted(self._history_compression_plan_cache),
            ),
            "approximation_plan_orders": tuple(
                sorted(self._history_approximation_plan_cache),
            ),
            "tensor_plan_orders": tuple(
                sorted({order for order, _sparse in self._history_tensor_plan_cache}),
            ),
            "extension_plan_orders": tuple(
                sorted(self._history_extension_plan_cache),
            ),
            "reduced_plan_orders": tuple(
                sorted(self._history_reduced_plan_cache),
            ),
            "extension_plan_batches": {
                order: len(plan["batches"])
                for order, plan in self._history_extension_plan_cache.items()
            },
        }

    def clear_history_cache(self):
        """Release cached history topologies and symbolic execution plans.

        The first-degree MPO itself remains usable.  This only clears the
        reusable structural data used by higher-order exponential calls; it
        never changes the current local tensors or their autodiff graph.
        Shared :class:`MPOBasis` templates observe the same clear operation.
        """
        self._history_topology_cache.clear()
        self._history_extension_plan_cache.clear()
        self._history_compression_plan_cache.clear()
        self._history_approximation_plan_cache.clear()
        self._history_tensor_plan_cache.clear()
        self._history_reduced_plan_cache.clear()
        self._history_symbolic_cache = None
        return self

    def _history_power_data(
        self,
        exponent,
        *,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="dense",
        progress=None,
    ):
        """Build the full virtual-history representation of ``H**exponent``.

        The ordinary :meth:`power` method intentionally keeps singleton open
        boundaries.  Algorithms 1--4 in the paper are most naturally applied
        before those boundary vectors are contracted, so this private helper
        also includes the virtual histories at both boundary cuts.  Boundary
        histories that are unreachable from a finite-chain boundary have zero
        tensor entries and are removed when the final boundary vectors are
        imposed.
        """
        if not isinstance(exponent, Integral) or int(exponent) < 1:
            raise ValueError("exponent must be a positive integer.")
        exponent = int(exponent)
        if self.metadata.get("history_valid", True) is False:
            raise ValueError(
                "history construction is unavailable after numerical "
                "fixed-rank compression."
            )
        if not self.is_first_degree:
            raise ValueError("history powers require a first-degree MPO.")
        self._first_degree_structure()

        if history_storage == "blocks":
            history_storage = "block_sparse"
        if history_storage not in {
            "auto", "dense", "sparse", "streaming", "block_sparse",
        }:
            raise ValueError(
                "history_storage must be one of 'auto', 'dense', 'sparse', "
                "'streaming', or 'block_sparse'."
            )
        if history_storage == "streaming" and cache_history:
            raise ValueError(
                "history_storage='streaming' requires cache_history=False; "
                "use history_storage='auto' for cached construction."
            )
        if history_storage == "auto":
            # A local-term automaton supplies exact structural transitions.
            # Preserve only those nonzero history blocks by default; MPSKit's
            # corresponding path is sparse as well.  Directly constructed
            # MPOs have no safe structural filter and retain the dense path.
            if self.symmetry is not None:
                history_storage = "block_sparse"
            elif self._structural_transitions is not None and cache_history:
                history_storage = "sparse"
            else:
                history_storage = "streaming" if not cache_history else "dense"
        schemas = self._history_schemas()
        if history_storage == "block_sparse":
            state_lists, cache_hit = self._history_topology(
                exponent,
                max_bond=max_bond,
                on_exceed=on_exceed,
                cache_history=cache_history,
                progress=progress,
            )
            tensor_plan, tensor_plan_cache_hit = (
                self._history_tensor_execution_plan(
                    exponent,
                    state_lists,
                    sparse=True,
                    cache_history=cache_history,
                )
            )
            return self._block_sparse_history_power_data(
                exponent,
                state_lists=state_lists,
                cache_hit=cache_hit,
                execution_plan=tensor_plan,
                tensor_plan_cache_hit=tensor_plan_cache_hit,
                progress=progress,
            )
        if history_storage in {"sparse", "streaming"}:
            if cache_history:
                state_lists, cache_hit = self._history_topology(
                    exponent,
                    max_bond=max_bond,
                    on_exceed=on_exceed,
                    cache_history=True,
                    progress=progress,
                )
                tensor_plan, tensor_plan_cache_hit = (
                    self._history_tensor_execution_plan(
                        exponent,
                        state_lists,
                        sparse=history_storage == "sparse",
                        cache_history=True,
                    )
                )
                return self._batched_history_power_data(
                    exponent,
                    state_lists=state_lists,
                    storage_mode=history_storage,
                    cache_hit=cache_hit,
                    execution_plan=tensor_plan,
                    tensor_plan_cache_hit=tensor_plan_cache_hit,
                    progress=progress,
                )
            return self._stream_history_power_data(
                exponent,
                schemas=schemas,
                max_bond=max_bond,
                on_exceed=on_exceed,
                sparse=history_storage == "sparse",
                progress=progress,
            )

        state_lists, cache_hit = self._history_topology(
            exponent,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            progress=progress,
        )
        tensor_plan, tensor_plan_cache_hit = (
            self._history_tensor_execution_plan(
                exponent,
                state_lists,
                sparse=history_storage == "sparse",
                cache_history=cache_history,
            )
        )
        return self._batched_history_power_data(
            exponent,
            state_lists=state_lists,
            storage_mode=history_storage,
            cache_hit=cache_hit,
            execution_plan=tensor_plan,
            tensor_plan_cache_hit=tensor_plan_cache_hit,
            progress=progress,
        )

    def _history_levels_from_tokens(self, schemas, bond, history):
        """Resolve flattened history tokens to base levels at one cut."""
        token_levels = self._history_symbolic_data()["token_levels"][bond]
        levels = []
        for token in history:
            level = token_levels.get((token,))
            if level is None:
                return None
            levels.append(level)
        return tuple(levels)

    def _history_tokens_reachable(
        self,
        schemas,
        bond,
        history,
        cache,
    ):
        """Check one raw-history state without building its order table.

        Raw product histories are reachable exactly when each factor follows
        a valid base-MPO transition from the finite all-one left boundary.
        This factor-wise check is equivalent to the forward product walk in
        ``_reachable_history_states`` but only resolves the candidate state
        requested by Algorithm 3.
        """
        key = (bond, tuple(history))
        if key in cache:
            return cache[key]
        factor_levels = self._history_levels_from_tokens(
            schemas,
            bond,
            history,
        )
        if factor_levels is None:
            cache[key] = None
            return None
        reachable_labels = self._history_symbolic_data()["reachable_labels"]
        if reachable_labels is not None and any(
            factor_level.label not in reachable_labels[bond]
            for factor_level in factor_levels
        ):
            cache[key] = None
            return None
        cache[key] = factor_levels
        return factor_levels

    @staticmethod
    def _remove_history_column(arrays, levels, bond, source, target, coefficient):
        """Apply one sequential Algorithm-1/4 column elimination.

        The production executor uses fused transfer maps. This small
        primitive remains available for deterministic reference tests and for
        debugging a compiled elimination schedule.
        """
        left = arrays[bond - 1]
        left = _setitem(
            left,
            (slice(None), target),
            left[:, target]
            + _multiply_scalar(coefficient, left[:, source]),
        )
        arrays[bond - 1] = _drop_axis(left, axis=1, position=source)
        if bond < len(arrays):
            arrays[bond] = _drop_axis(arrays[bond], axis=0, position=source)
        levels[bond].pop(source)

    @staticmethod
    def _history_axis_groups(size, operations):
        """Compile ordered channel merges into original-index contributions."""
        groups = [{position: 1.0} for position in range(size)]
        for source, target, merge in operations:
            if merge:
                for original, weight in groups[source].items():
                    groups[target][original] = (
                        groups[target].get(original, 0.0) + weight
                    )
            groups.pop(source)
        return groups

    @staticmethod
    def _apply_history_axis_groups(array, groups, *, axis):
        """Apply all channel deletions on one virtual axis in one scatter."""
        if isinstance(array, SparseVirtualTensor):
            return array.apply_axis_groups(groups, axis=axis)
        transfer = _history_transfer_matrix(groups, int(array.shape[axis]))
        if transfer is not None:
            return _apply_history_transfer(array, transfer, axis=axis)

        sources = []
        targets = []
        weights = []
        for target, group in enumerate(groups):
            for source, weight in group.items():
                sources.append(source)
                targets.append(target)
                weights.append(weight)

        output_shape = list(array.shape)
        output_shape[axis] = len(groups)
        result = _zeros(output_shape, like=array)
        if not sources:
            return result

        source_indices = np.asarray(sources, dtype=int)
        target_indices = np.asarray(targets, dtype=int)
        if axis == 1:
            source_backend = _as_backend(source_indices, like=array)
            values = array[:, source_backend]
            rows = np.repeat(
                np.arange(array.shape[0], dtype=int),
                len(source_indices),
            )
            columns = np.tile(target_indices, array.shape[0])
        else:
            source_backend = _as_backend(source_indices, like=array)
            values = array[source_backend, :]
            rows = np.repeat(target_indices, array.shape[1])
            columns = np.tile(
                np.arange(array.shape[1], dtype=int),
                len(source_indices),
            )
        coefficient = _as_backend(np.asarray(weights), like=values)
        weight_shape = (
            (1, len(coefficient), 1, 1)
            if axis == 1
            else (len(coefficient), 1, 1, 1)
        )
        values = ar.do(
            "multiply",
            values,
            ar.do("reshape", coefficient, weight_shape),
        )
        return _scatter_add_2d(
            result,
            rows,
            columns,
            values.reshape((-1, *array.shape[-2:])),
        )

    @staticmethod
    def _history_axis_polynomial_groups(size, operations):
        """Compile channel merges whose weights are polynomials in ``dt``."""
        groups = [
            {position: [(0, 1.0)]}
            for position in range(size)
        ]
        for source, target, merge, power, coefficient in operations:
            if merge and coefficient != 0.0:
                for original, terms in groups[source].items():
                    shifted = [
                        (old_power + power, old_coefficient * coefficient)
                        for old_power, old_coefficient in terms
                    ]
                    groups[target].setdefault(original, []).extend(shifted)
            groups.pop(source)
        return groups

    @staticmethod
    def _apply_history_polynomial_groups(array, groups, dt, *, axis):
        """Apply a parameterized channel schedule in one backend scatter."""
        if isinstance(array, SparseVirtualTensor):
            return array.apply_polynomial_axis_groups(groups, dt, axis=axis)
        old_size = int(array.shape[axis])
        power_maps = {}
        for target, group in enumerate(groups):
            for source, terms in group.items():
                for power, coefficient in terms:
                    power_maps.setdefault(int(power), []).append(
                        (target, source, coefficient),
                    )
        if (
            len(groups) * old_size <= _MAX_HISTORY_TRANSFER_ELEMENTS
            and power_maps
        ):
            transfer = None
            for power, entries in power_maps.items():
                structural = np.zeros((len(groups), old_size), dtype=float)
                for target, source, coefficient in entries:
                    structural[target, source] += float(coefficient)
                structural = _as_backend(structural, like=array)
                weighted = _multiply_scalar(dt ** power, structural)
                transfer = weighted if transfer is None else ar.do(
                    "add",
                    transfer,
                    weighted,
                )
            return _apply_history_transfer(array, transfer, axis=axis)

        sources = []
        targets = []
        powers = []
        coefficients = []
        for target, group in enumerate(groups):
            for source, terms in group.items():
                for power, coefficient in terms:
                    sources.append(source)
                    targets.append(target)
                    powers.append(power)
                    coefficients.append(coefficient)

        output_shape = list(array.shape)
        output_shape[axis] = len(groups)
        result = _zeros(output_shape, like=array)
        if not sources:
            return result

        source_indices = np.asarray(sources, dtype=int)
        target_indices = np.asarray(targets, dtype=int)
        if axis == 1:
            source_backend = _as_backend(source_indices, like=array)
            values = array[:, source_backend]
            rows = np.repeat(
                np.arange(array.shape[0], dtype=int),
                len(source_indices),
            )
            columns = np.tile(target_indices, array.shape[0])
        else:
            source_backend = _as_backend(source_indices, like=array)
            values = array[source_backend, :]
            rows = np.repeat(target_indices, array.shape[1])
            columns = np.tile(
                np.arange(array.shape[1], dtype=int),
                len(source_indices),
            )

        powers = tuple(int(power) for power in powers)
        power_values = {
            power: dt ** power
            for power in set(powers)
        }
        weights = tuple(
            _multiply_scalar(power_values[power], coefficient)
            for power, coefficient in zip(powers, coefficients)
        )
        if any(
            _backend_name(value) not in {"builtins", "numpy"}
            for value in weights
        ):
            weights = ar.do("stack", weights)
        else:
            weights = np.asarray(weights)
            weights = _as_backend(weights, like=values)
        weight_shape = (
            (1, len(weights), 1, 1)
            if axis == 1
            else (len(weights), 1, 1, 1)
        )
        values = ar.do(
            "multiply",
            values,
            ar.do("reshape", weights, weight_shape),
        )
        return _scatter_add_into_2d(
            result,
            rows,
            columns,
            values.reshape((-1, *array.shape[-2:])),
        )

    def _algorithm_two_bond(self, arrays, levels, bond, actions):
        """Apply one bond's Algorithm-2 schedule as two fused transforms."""
        current = list(levels[bond])
        row_operations = []
        column_operations = []
        merges = []
        for (
            source_history,
            canonical,
            mode,
            source_label,
            source,
            target,
        ) in actions:
            if source >= len(current) or target >= len(current) or target == source:
                continue
            if mode == "row":
                if bond >= len(arrays):
                    continue
                row_operations.append((source, target, True))
                column_operations.append((source, target, False))
            else:
                column_operations.append((source, target, True))
                row_operations.append((source, target, False))
            current.pop(source)
            merges.append({
                "bond": bond - 1,
                "source": source_label,
                "target": canonical,
                "mode": mode,
                "history": source_history,
            })

        if not merges:
            return merges
        if column_operations:
            arrays[bond - 1] = self._apply_history_axis_groups(
                arrays[bond - 1],
                self._history_axis_groups(
                    arrays[bond - 1].shape[1],
                    column_operations,
                ),
                axis=1,
            )
        if row_operations and bond < len(arrays):
            arrays[bond] = self._apply_history_axis_groups(
                arrays[bond],
                self._history_axis_groups(
                    arrays[bond].shape[0],
                    row_operations,
                ),
                axis=0,
            )
        levels[bond] = current
        return merges

    def _history_compression_plan(self, levels, order, *, cache_history):
        """Compile Algorithms 1--2 as a topology-only elimination plan.

        The plan is generated by simulating only virtual histories.  Applying
        it later performs the same ordered eliminations on the current
        numerical arrays, with ``dt`` supplied at execution time.  This is
        the key separation needed by parameter optimization: the expensive
        history decisions are cached, while all values remain differentiable
        and are recomputed for each call.
        """
        if cache_history:
            cached = self._history_compression_plan_cache.get(order)
            if cached is not None:
                return cached, True

        working = [list(bond_levels) for bond_levels in levels]
        algorithm_one = []
        target_history = tuple(MPOLevelToken(1) for _ in range(order))
        for bond in range(1, self.L + 1):
            for number_of_threes in range(1, order + 1):
                for level in tuple(working[bond]):
                    history = level.history
                    if not (
                        all(
                            _level_number(token) in (1, 3)
                            for token in history
                        )
                        and sum(
                            _level_number(token) == 3 for token in history
                        ) == number_of_threes
                    ):
                        continue
                    positions = self._history_level_positions(working[bond])
                    source = positions.get(history)
                    target = positions.get(target_history)
                    if source is None or target is None or source == target:
                        raise ValueError(
                            "history power lost its all-one Algorithm-1 target."
                        )
                    algorithm_one.append((
                        bond,
                        history,
                        target_history,
                        number_of_threes,
                        source,
                        target,
                    ))
                    working[bond].pop(source)

        algorithm_two = []
        for bond in range(1, self.L + 1):
            changed = True
            while changed:
                changed = False
                for level in tuple(working[bond]):
                    history = level.history
                    number_of_ones = sum(
                        _level_number(token) == 1 for token in history
                    )
                    number_of_threes = sum(
                        _level_number(token) == 3 for token in history
                    )
                    if number_of_threes <= number_of_ones:
                        canonical = _sort_history_front(history, 1)
                        mode = "row"
                    else:
                        canonical = _sort_history_front(history, 3)
                        mode = "column"
                    if canonical == history:
                        continue
                    positions = self._history_level_positions(working[bond])
                    source = positions.get(history)
                    target = positions.get(canonical)
                    if source is None or target is None or source == target:
                        continue
                    algorithm_two.append((
                        bond,
                        history,
                        canonical,
                        mode,
                        level.label,
                        source,
                        target,
                    ))
                    working[bond].pop(source)
                    changed = True
                    break

        plan = {
            "algorithm_one": tuple(algorithm_one),
            "algorithm_two": tuple(algorithm_two),
        }
        if cache_history:
            self._history_compression_plan_cache[order] = plan
        return plan, False

    @staticmethod
    def _history_add_polynomial_vector(target, source, *, power=0, coefficient=1.0):
        """Add one sparse raw-axis polynomial vector into another."""

        output = {
            raw: dict(polynomial)
            for raw, polynomial in target.items()
        }
        for raw, polynomial in source.items():
            destination = output.setdefault(raw, {})
            for source_power, source_coefficient in polynomial.items():
                total_power = int(source_power) + int(power)
                destination[total_power] = (
                    destination.get(total_power, 0.0)
                    + coefficient * source_coefficient
                )
        return output

    def _history_reduction_map(self, levels, order):
        """Compile Algorithms 1--2 into raw-to-reduced axis maps.

        The ordinary executor first creates every raw operator-valued tensor
        block and then applies row/column eliminations. This symbolic variant
        applies those eliminations to sparse basis vectors instead. Every raw
        axis entry consequently knows its final reduced target and the small
        polynomial weight it carries, allowing numerical blocks to be
        scattered directly into the reduced MPO.
        """

        current = list(levels)
        row_vectors = [{index: {0: 1.0}} for index in range(len(current))]
        column_vectors = [{index: {0: 1.0}} for index in range(len(current))]
        target_history = tuple(MPOLevelToken(1) for _ in range(order))
        coefficient_denominator = factorial(order)

        algorithm_one = 0
        for number_of_threes in range(1, order + 1):
            for level in tuple(current):
                history = level.history
                if not (
                    all(_level_number(token) in (1, 3) for token in history)
                    and sum(_level_number(token) == 3 for token in history)
                    == number_of_threes
                ):
                    continue
                positions = self._history_level_positions(current)
                source = positions.get(history)
                target = positions.get(target_history)
                if source is None or target is None or source == target:
                    raise ValueError(
                        "history power lost its all-one Algorithm-1 target."
                    )
                column_vectors[target] = self._history_add_polynomial_vector(
                    column_vectors[target],
                    column_vectors[source],
                    power=number_of_threes,
                    coefficient=(
                        factorial(order - number_of_threes)
                        / coefficient_denominator
                    ),
                )
                current.pop(source)
                row_vectors.pop(source)
                column_vectors.pop(source)
                algorithm_one += 1

        algorithm_two = 0
        changed = True
        while changed:
            changed = False
            for level in tuple(current):
                history = level.history
                number_of_ones = sum(
                    _level_number(token) == 1 for token in history
                )
                number_of_threes = sum(
                    _level_number(token) == 3 for token in history
                )
                if number_of_threes <= number_of_ones:
                    canonical = _sort_history_front(history, 1)
                    mode = "row"
                else:
                    canonical = _sort_history_front(history, 3)
                    mode = "column"
                if canonical == history:
                    continue
                positions = self._history_level_positions(current)
                source = positions.get(history)
                target = positions.get(canonical)
                if source is None or target is None or source == target:
                    continue
                if mode == "row":
                    row_vectors[target] = self._history_add_polynomial_vector(
                        row_vectors[target],
                        row_vectors[source],
                    )
                else:
                    column_vectors[target] = self._history_add_polynomial_vector(
                        column_vectors[target],
                        column_vectors[source],
                    )
                current.pop(source)
                row_vectors.pop(source)
                column_vectors.pop(source)
                algorithm_two += 1
                changed = True
                break

        def invert(vectors):
            targets = np.full(len(levels), -1, dtype=int)
            powers = np.zeros(len(levels), dtype=int)
            coefficients = np.zeros(len(levels), dtype=float)
            for target, vector in enumerate(vectors):
                for raw, polynomial in vector.items():
                    if len(polynomial) != 1:
                        raise ValueError(
                            "history reduction produced a non-monomial raw-axis map."
                        )
                    ((power, coefficient),) = polynomial.items()
                    targets[raw] = target
                    powers[raw] = int(power)
                    coefficients[raw] = coefficient
            return targets, powers, coefficients

        row_targets, row_powers, row_coefficients = invert(row_vectors)
        column_targets, column_powers, column_coefficients = invert(column_vectors)
        return {
            "levels": tuple(current),
            "row_targets": row_targets,
            "row_powers": row_powers,
            "row_coefficients": row_coefficients,
            "column_targets": column_targets,
            "column_powers": column_powers,
            "column_coefficients": column_coefficients,
            "algorithm_one": algorithm_one,
            "algorithm_two": algorithm_two,
        }

    def _history_reduced_plan(
        self,
        order,
        *,
        max_bond,
        on_exceed,
        cache_history,
        include_extension=False,
    ):
        """Stream raw histories into a reusable direct-reduced execution plan."""

        if cache_history:
            cached = self._history_reduced_plan_cache.get(order)
            if cached is not None and (
                not include_extension
                or cached.get("includes_extension", False)
            ):
                self._check_history_bond_dimensions(
                    cached["raw_bond_dimensions"],
                    max_bond=max_bond,
                    on_exceed=on_exceed,
                )
                return cached, True

        schemas = self._history_schemas()
        start_levels = tuple(
            level
            for level in schemas[0]
            if _level_number(level.history[0]) == 1
        )
        if len(start_levels) != 1:
            raise ValueError(
                "history construction requires one level-1 starting channel."
            )
        raw_left = (tuple(start_levels[0] for _ in range(order)),)
        warned = [False]

        def levels_for(states, bond):
            return tuple(
                MPOLevel(
                    ("raw-history", order, bond, position),
                    tuple(
                        token
                        for factor in state
                        for token in factor.history
                    ),
                    charge=tuple(factor.charge for factor in state),
                )
                for position, state in enumerate(states)
            )

        raw_levels_left = levels_for(raw_left, 0)
        reduction_left = self._history_reduction_map(raw_levels_left, order)
        reduced_levels = [reduction_left["levels"]]
        raw_dimensions = [len(raw_left)]
        site_plans = []
        extension_terms = 0
        algorithm_one = reduction_left["algorithm_one"]
        algorithm_two = reduction_left["algorithm_two"]

        for site in range(self.L):
            raw_right = self._history_state_step(
                schemas,
                raw_left,
                site,
                max_bond=max_bond,
                on_exceed=on_exceed,
                warned=warned,
            )
            raw_levels_right = levels_for(raw_right, site + 1)
            reduction_right = self._history_reduction_map(
                raw_levels_right,
                order,
            )

            left_indices, right_indices = self._history_allowed_pairs(
                site,
                raw_left,
                raw_right,
                sparse=True,
            )
            positions = self._history_local_position_arrays(
                site,
                raw_left,
                raw_right,
            )
            keep = (
                reduction_left["row_targets"][left_indices] >= 0
            ) & (
                reduction_right["column_targets"][right_indices] >= 0
            )
            left_indices = left_indices[keep]
            right_indices = right_indices[keep]
            output_left = reduction_left["row_targets"][left_indices]
            output_right = reduction_right["column_targets"][right_indices]
            powers = (
                reduction_left["row_powers"][left_indices]
                + reduction_right["column_powers"][right_indices]
            )
            coefficients = (
                reduction_left["row_coefficients"][left_indices]
                * reduction_right["column_coefficients"][right_indices]
            )

            raw_extension = None
            if include_extension:
                fake_levels = [()] * (self.L + 1)
                fake_levels[site] = raw_levels_left
                fake_levels[site + 1] = raw_levels_right
                extension_plan, _ = self._history_extension_plan(
                    fake_levels,
                    order,
                    cache_history=False,
                    materialize=True,
                )
                raw_extension = extension_plan["site_plans"][site]
                extension_terms += extension_plan["selected_terms"]
            reduced_extension = None
            if raw_extension is not None:
                extension_keep = (
                    reduction_left["row_targets"][raw_extension["left_targets"]] >= 0
                ) & (
                    reduction_right["column_targets"][raw_extension["right_targets"]]
                    >= 0
                )
                if np.any(extension_keep):
                    extension_left = raw_extension["left_targets"][extension_keep]
                    extension_right = raw_extension["right_targets"][extension_keep]
                    reduced_extension = {
                        "site": site,
                        "left_targets": raw_extension["left_targets"][extension_keep],
                        "right_targets": raw_extension["right_targets"][extension_keep],
                        "left_positions": tuple(
                            values[extension_keep]
                            for values in raw_extension["left_positions"]
                        ),
                        "right_positions": tuple(
                            values[extension_keep]
                            for values in raw_extension["right_positions"]
                        ),
                        "left_identity": tuple(
                            values[extension_keep]
                            for values in raw_extension["left_identity"]
                        ),
                        "right_identity": tuple(
                            values[extension_keep]
                            for values in raw_extension["right_identity"]
                        ),
                        "weights": raw_extension["weights"][extension_keep],
                        "output_left": reduction_left["row_targets"][extension_left],
                        "output_right": reduction_right["column_targets"][extension_right],
                        "powers": (
                            reduction_left["row_powers"][extension_left]
                            + reduction_right["column_powers"][extension_right]
                            + 1
                        ),
                        "coefficients": (
                            reduction_left["row_coefficients"][extension_left]
                            * reduction_right["column_coefficients"][extension_right]
                        ),
                    }

            site_plans.append({
                "site": site,
                "positions": positions,
                "left_indices": left_indices,
                "right_indices": right_indices,
                "output_left": output_left,
                "output_right": output_right,
                "powers": powers,
                "coefficients": coefficients,
                "extension": reduced_extension,
                "raw_total_blocks": len(raw_left) * len(raw_right),
            })
            raw_dimensions.append(len(raw_right))
            reduced_levels.append(reduction_right["levels"])
            algorithm_one += reduction_right["algorithm_one"]
            algorithm_two += reduction_right["algorithm_two"]
            raw_left = raw_right
            raw_levels_left = raw_levels_right
            reduction_left = reduction_right

        plan = {
            "sites": tuple(site_plans),
            "levels": tuple(reduced_levels),
            "raw_bond_dimensions": tuple(raw_dimensions),
            "reduced_bond_dimensions": tuple(len(levels) for levels in reduced_levels),
            "extension_terms": extension_terms,
            "algorithm_one": algorithm_one,
            "algorithm_two": algorithm_two,
            "includes_extension": bool(include_extension),
        }
        if cache_history:
            self._history_reduced_plan_cache[order] = plan
        return plan, False

    @staticmethod
    def _check_history_bond_dimensions(dimensions, *, max_bond, on_exceed):
        """Apply the public temporary-bond policy to known dimensions."""

        if max_bond is None or on_exceed == "ignore":
            return
        warned = False
        for site, dimension in enumerate(dimensions[1:]):
            if dimension <= max_bond:
                continue
            message = (
                "extensive_exponential history bond dimension "
                f"{dimension} exceeds max_bond={max_bond} after site {site}."
            )
            if on_exceed == "raise":
                raise MemoryError(message)
            if not warned:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                warned = True

    @staticmethod
    def _history_polynomial_weights(powers, coefficients, dt, *, like):
        """Evaluate monomial execution weights on the active backend."""

        values = tuple(
            _multiply_scalar(coefficient, dt ** int(power))
            for power, coefficient in zip(powers, coefficients)
        )
        if not values:
            return None
        if any(_backend_name(value) not in {"builtins", "numpy"} for value in values):
            return ar.do("stack", values)
        return _as_backend(np.asarray(values), like=like)

    def _reduced_history_power_data(
        self,
        dt,
        *,
        order,
        extend,
        max_bond,
        on_exceed,
        cache_history,
        chunk_size=65536,
        progress=None,
    ):
        """Build final Algorithms-1/2 tensors without raw virtual tensors."""

        plan, cache_hit = self._history_reduced_plan(
            order,
            max_bond=max_bond,
            on_exceed=on_exceed,
            cache_history=cache_history,
            include_extension=extend,
        )
        arrays = []
        stored_blocks = 0
        raw_stored_blocks = 0
        for site_plan in plan["sites"]:
            site = site_plan["site"]
            source = self._arrays[site]
            reference = (
                source._like
                if isinstance(source, SparseVirtualTensor)
                else source[0, 0]
            )
            tensor = SparseVirtualTensor((
                len(plan["levels"][site]),
                len(plan["levels"][site + 1]),
                self.phys_dim,
                self.phys_dim,
            ), like=reference)
            left_indices = site_plan["left_indices"]
            right_indices = site_plan["right_indices"]
            raw_stored_blocks += len(left_indices)
            for start in range(0, len(left_indices), chunk_size):
                stop = start + chunk_size
                values = self._history_local_product_batch_values(
                    site,
                    site_plan["positions"],
                    left_indices[start:stop],
                    right_indices[start:stop],
                )
                weights = self._history_polynomial_weights(
                    site_plan["powers"][start:stop],
                    site_plan["coefficients"][start:stop],
                    dt,
                    like=values,
                )
                if weights is not None:
                    values = ar.do(
                        "multiply",
                        values,
                        weights[..., None, None],
                    )
                tensor = tensor.scatter_add(
                    site_plan["output_left"][start:stop],
                    site_plan["output_right"][start:stop],
                    values,
                )

            extension = site_plan["extension"] if extend else None
            if extension is not None:
                values = self._history_local_product_site_batch(extension, order)
                combinatorial = _as_backend(extension["weights"], like=values)
                reduction = self._history_polynomial_weights(
                    extension["powers"],
                    extension["coefficients"],
                    dt,
                    like=values,
                )
                weights = ar.do("multiply", combinatorial, reduction)
                values = ar.do("multiply", values, weights[..., None, None])
                tensor = tensor.scatter_add(
                    extension["output_left"],
                    extension["output_right"],
                    values,
                )
            arrays.append(tensor)
            stored_blocks += tensor.stored_blocks
            if progress is not None:
                progress.detail(
                    f"reduced history {site + 1}/{self.L}",
                    bond_dimensions=tuple(
                        len(levels) for levels in plan["levels"][1:-1]
                    ),
                    blocks=stored_blocks,
                )

        storage_info = {
            "mode": "reduced",
            "stored_blocks": stored_blocks,
            "total_blocks": sum(
                len(plan["levels"][site]) * len(plan["levels"][site + 1])
                for site in range(self.L)
            ),
            "raw_stored_blocks": raw_stored_blocks,
            "raw_total_blocks": sum(
                site_plan["raw_total_blocks"] for site_plan in plan["sites"]
            ),
            "materialized_raw_virtual_tensors": False,
            "tensor_plan_cache_hit": cache_hit,
            "initial_bond_dimensions": tuple(plan["raw_bond_dimensions"][1:-1]),
            "exact_history_merges": plan["algorithm_two"],
            "algorithm_one_eliminations": plan["algorithm_one"],
            "extension_terms": plan["extension_terms"] if extend else 0,
        }
        return (
            arrays,
            [list(levels) for levels in plan["levels"]],
            cache_hit,
            storage_info,
        )

    def _algorithm_one(self, arrays, levels, order, dt, *, plan=None):
        """Apply the paper's extensive prefactor transformation.

        This pass removes all-identity/all-level-3 histories into the all-one
        target with the paper's factorial coefficient.  It is intentionally
        separate from Algorithm 2 so a report can distinguish coefficient
        rewiring from exact equal-history compression.
        """
        if plan is None:
            plan, _ = self._history_compression_plan(
                levels, order, cache_history=False,
            )
        coefficient_denominator = factorial(order)
        actions_by_bond = [[] for _ in range(self.L + 1)]
        for action in plan["algorithm_one"]:
            actions_by_bond[action[0]].append(action[1:])

        # Compile one polynomial virtual transfer per bond.  This is
        # algebraically identical to the ordered column eliminations, but it
        # replaces one full tensor copy per history channel by one backend
        # contraction per side of the cut.
        for bond, actions in enumerate(actions_by_bond):
            if not actions:
                continue
            current = list(levels[bond])
            polynomial_operations = []
            removal_operations = []
            for (
                source_history,
                target_history,
                number_of_threes,
                source,
                target,
            ) in actions:
                if source >= len(current) or target >= len(current) or target == source:
                    raise ValueError(
                        "history power lost its all-one Algorithm-1 target."
                    )
                polynomial_operations.append((
                    source,
                    target,
                    True,
                    number_of_threes,
                    factorial(order - number_of_threes)
                    / coefficient_denominator,
                ))
                removal_operations.append((source, target, False))
                current.pop(source)

            if not polynomial_operations:
                continue
            arrays[bond - 1] = self._apply_history_polynomial_groups(
                arrays[bond - 1],
                self._history_axis_polynomial_groups(
                    arrays[bond - 1].shape[1],
                    polynomial_operations,
                ),
                dt,
                axis=1,
            )
            if bond < len(arrays):
                arrays[bond] = self._apply_history_axis_groups(
                    arrays[bond],
                    self._history_axis_groups(
                        arrays[bond].shape[0],
                        removal_operations,
                    ),
                    axis=0,
                )
            levels[bond] = current

    def _algorithm_two(self, arrays, levels, *, plan=None):
        """Apply the paper's exact history-only compression transformations.

        The implementation chooses the row or column orientation from the
        number of level-1 and level-3 tokens.  This is a structural rule from
        the paper, not a numerical rank heuristic; no backend conversion or
        tolerance is involved.
        """
        if plan is None:
            plan, _ = self._history_compression_plan(
                levels, len(levels[0][0].history), cache_history=False,
            )
        actions_by_bond = [[] for _ in range(self.L + 1)]
        for (
            bond,
            source_history,
            canonical,
            mode,
            source_label,
            source,
            target,
        ) in plan[
            "algorithm_two"
        ]:
            actions_by_bond[bond].append(
                (source_history, canonical, mode, source_label, source, target),
            )

        merges = []
        for bond, actions in enumerate(actions_by_bond):
            if actions:
                merges.extend(
                    self._algorithm_two_bond(arrays, levels, bond, actions),
                )
        return merges

    def _base_level_position(self, bond, level):
        """Resolve a symbolic base level to its original virtual position."""
        positions = self._base_level_positions()[bond]
        position = positions["label"].get(level.label)
        if position is not None:
            return position
        return positions["history"].get(level.history)

    def _estimate_history_extension_terms(self, state_lists, order):
        """Estimate Algorithm-3 terms without building pair arrays.

        Algorithm 3 inserts one level-1 token on the left history and one
        level-3 token on the right history.  Its numerical executor later
        forms the Cartesian product of the reachable insertion lists.  This
        helper performs only that symbolic reachability/counting work, so it
        can select ``mode="auto"`` before the potentially large pair plan is
        materialized.  The count uses the same candidate and insertion rules
        as :meth:`_history_extension_plan`, and is therefore also a useful
        exact upper-level work estimate for the selected extension batches.
        """
        order = int(order)
        symbolic = self._history_symbolic_data()
        schemas = symbolic["schemas"]
        one = MPOLevelToken(1)
        three = MPOLevelToken(3)
        reachability_cache = {}
        estimated_terms = 0

        def flattened_history(state):
            return tuple(
                token
                for factor in state
                for token in factor.history
            )

        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            left_candidates = []
            for left_pos, state in enumerate(left_states):
                history = flattened_history(state)
                numbers = _history_signature(history)
                if all(number in (1, 3) for number in numbers) and 3 in numbers:
                    continue
                left_candidates.append((left_pos, history, numbers))
            right_candidates = []
            for right_pos, state in enumerate(right_states):
                history = flattened_history(state)
                numbers = _history_signature(history)
                if all(number > 1 for number in numbers):
                    right_candidates.append((right_pos, history, numbers))
            if not left_candidates or not right_candidates:
                continue

            left_counts = []
            for insert_position in range(order + 1):
                count = 0
                for _left_pos, history, _numbers in left_candidates:
                    extended = (
                        history[:insert_position]
                        + (one,)
                        + history[insert_position:]
                    )
                    if self._history_tokens_reachable(
                        schemas,
                        site,
                        extended,
                        reachability_cache,
                    ) is not None:
                        count += 1
                left_counts.append(count)

            right_counts = []
            for insert_position in range(order + 1):
                count = 0
                for _right_pos, history, _numbers in right_candidates:
                    extended = (
                        history[:insert_position]
                        + (three,)
                        + history[insert_position:]
                    )
                    if self._history_tokens_reachable(
                        schemas,
                        site + 1,
                        extended,
                        reachability_cache,
                    ) is not None:
                        count += 1
                right_counts.append(count)

            estimated_terms += sum(
                left_count * right_count
                for left_count in left_counts
                for right_count in right_counts
            )

        return int(estimated_terms)

    def _history_extension_plan(
        self,
        levels,
        order,
        *,
        cache_history,
        materialize=False,
    ):
        """Compile Algorithm 3 as batched local insertion transitions.

        The naive pseudocode has a pair of history loops inside every site.
        The selected contribution is separable at the virtual level: for a
        fixed insertion position, one only needs the valid left state list,
        the valid right state list, and the local base-block positions for
        each factor. Store those short lists and let execution stream their
        Cartesian products in backend-sized chunks. The default plan therefore
        retains no repeated left/right pair arrays. ``materialize=True`` is a
        private compatibility path used by the reduced quotient, which needs
        a flat index list to apply its symbolic axis maps.
        """
        if cache_history:
            cached = self._history_extension_plan_cache.get(order)
            if cached is not None:
                return cached, True

        symbolic = self._history_symbolic_data()
        schemas = symbolic["schemas"]
        snapshot = [tuple(bond_levels) for bond_levels in levels]
        one = MPOLevelToken(1)
        three = MPOLevelToken(3)
        batches = []
        batches_by_site = [[] for _ in range(self.L)]
        selected_terms = 0

        for site in range(self.L):
            left_levels = snapshot[site]
            right_levels = snapshot[site + 1]
            left_candidates = []
            for left_pos, left_level in enumerate(left_levels):
                history = left_level.history
                numbers = _history_signature(history)
                if all(number in (1, 3) for number in numbers) and 3 in numbers:
                    continue
                left_candidates.append((left_pos, history, numbers))
            right_candidates = [
                (right_pos, right_level.history, _history_signature(right_level.history))
                for right_pos, right_level in enumerate(right_levels)
                if all(number > 1 for number in _history_signature(right_level.history))
            ]
            if not left_candidates or not right_candidates:
                continue

            left_insertions = []
            for insert_position in range(order + 1):
                entries = []
                for left_pos, history, numbers in left_candidates:
                    extended = (
                        history[:insert_position]
                        + (one,)
                        + history[insert_position:]
                    )
                    state = self._history_tokens_reachable(
                        schemas,
                        site,
                        extended,
                        {},
                    )
                    if state is not None:
                        entries.append((left_pos, numbers, state))
                if entries:
                    left_insertions.append(tuple(entries))
                else:
                    left_insertions.append(())

            right_insertions = []
            for insert_position in range(order + 1):
                entries = []
                for right_pos, history, numbers in right_candidates:
                    extended = (
                        history[:insert_position]
                        + (three,)
                        + history[insert_position:]
                    )
                    state = self._history_tokens_reachable(
                        schemas,
                        site + 1,
                        extended,
                        {},
                    )
                    if state is not None:
                        entries.append((right_pos, numbers, state))
                if entries:
                    right_insertions.append(tuple(entries))
                else:
                    right_insertions.append(())

            for left_entries in left_insertions:
                if not left_entries:
                    continue
                for right_entries in right_insertions:
                    if not right_entries:
                        continue
                    left_targets = np.asarray(
                        [entry[0] for entry in left_entries],
                        dtype=int,
                    )
                    right_targets = np.asarray(
                        [entry[0] for entry in right_entries],
                        dtype=int,
                    )
                    left_weights = np.asarray(
                        [
                            1.0 / ((order + 1) * (entry[1].count(1) + 1))
                            for entry in left_entries
                        ],
                        dtype=float,
                    )
                    right_weights = np.asarray(
                        [1.0 / (entry[1].count(3) + 1) for entry in right_entries],
                        dtype=float,
                    )
                    left_states = tuple(entry[2] for entry in left_entries)
                    right_states = tuple(entry[2] for entry in right_entries)
                    left_positions = tuple(
                        np.asarray(
                            [
                                self._base_level_position(site, state[factor])
                                if self._base_level_position(site, state[factor]) is not None
                                else -1
                                for state in left_states
                            ],
                            dtype=int,
                        )
                        for factor in range(order + 1)
                    )
                    right_positions = tuple(
                        np.asarray(
                            [
                                self._base_level_position(site + 1, state[factor])
                                if self._base_level_position(site + 1, state[factor]) is not None
                                else -1
                                for state in right_states
                            ],
                            dtype=int,
                        )
                        for factor in range(order + 1)
                    )
                    left_identity = tuple(
                        np.asarray(
                            [
                                _level_number(state[factor].history[0]) == 1
                                for state in left_states
                            ],
                            dtype=bool,
                        )
                        for factor in range(order + 1)
                    )
                    right_identity = tuple(
                        np.asarray(
                            [
                                _level_number(state[factor].history[0]) == 1
                                for state in right_states
                            ],
                            dtype=bool,
                        )
                        for factor in range(order + 1)
                    )
                    batch = {
                        "site": site,
                        "left_targets": left_targets,
                        "right_targets": right_targets,
                        "left_positions": left_positions,
                        "right_positions": right_positions,
                        "left_identity": left_identity,
                        "right_identity": right_identity,
                        "left_weights": left_weights,
                        "right_weights": right_weights,
                    }
                    batches.append(batch)
                    batches_by_site[site].append(batch)
                    selected_terms += int(left_targets.size * right_targets.size)

        site_plans = []
        for site, site_batches in enumerate(batches_by_site):
            if not site_batches:
                site_plans.append(None)
                continue

            if not materialize:
                # Keep one site-local view of the symbolic batches. The
                # compatibility arrays below are one-dimensional candidate
                # lists, never Cartesian products.
                site_plans.append({
                    "site": site,
                    "batches": tuple(site_batches),
                    "left_targets": np.concatenate(
                        [batch["left_targets"] for batch in site_batches],
                    ),
                    "right_targets": np.concatenate(
                        [batch["right_targets"] for batch in site_batches],
                    ),
                })
                continue

            left_targets = []
            right_targets = []
            weights = []
            left_positions = [[] for _ in range(order + 1)]
            right_positions = [[] for _ in range(order + 1)]
            left_identity = [[] for _ in range(order + 1)]
            right_identity = [[] for _ in range(order + 1)]
            for batch in site_batches:
                left_size = batch["left_targets"].size
                right_size = batch["right_targets"].size
                left_targets.append(
                    np.repeat(batch["left_targets"], right_size),
                )
                right_targets.append(
                    np.tile(batch["right_targets"], left_size),
                )
                weights.append(
                    (
                        batch["left_weights"][:, None]
                        * batch["right_weights"][None, :]
                    ).reshape(-1),
                )
                for factor in range(order + 1):
                    left_positions[factor].append(
                        np.repeat(
                            batch["left_positions"][factor],
                            right_size,
                        ),
                    )
                    right_positions[factor].append(
                        np.tile(
                            batch["right_positions"][factor],
                            left_size,
                        ),
                    )
                    left_identity[factor].append(
                        np.repeat(
                            batch["left_identity"][factor],
                            right_size,
                        ),
                    )
                    right_identity[factor].append(
                        np.tile(
                            batch["right_identity"][factor],
                            left_size,
                        ),
                    )

            site_plans.append({
                "site": site,
                "left_targets": np.concatenate(left_targets),
                "right_targets": np.concatenate(right_targets),
                "left_positions": tuple(
                    np.concatenate(values) for values in left_positions
                ),
                "right_positions": tuple(
                    np.concatenate(values) for values in right_positions
                ),
                "left_identity": tuple(
                    np.concatenate(values) for values in left_identity
                ),
                "right_identity": tuple(
                    np.concatenate(values) for values in right_identity
                ),
                "weights": np.concatenate(weights),
            })

        plan = {
            # ``batches`` and the site-local views refer to the same symbolic
            # batch dictionaries. No Cartesian arrays are duplicated here.
            "batches": tuple(batches),
            "site_plans": tuple(site_plans),
            "selected_terms": selected_terms,
            "materialized": bool(materialize),
        }
        if cache_history:
            self._history_extension_plan_cache[order] = plan
        return plan, False

    def _history_local_product_batch(self, batch, order):
        """Evaluate one Algorithm-3 batch as a backend-native matrix product."""
        site = batch["site"]
        reference = self._arrays[site][0, 0]
        product_block = None
        for factor in range(order + 1):
            left_positions = batch["left_positions"][factor]
            right_positions = batch["right_positions"][factor]
            left_valid = left_positions >= 0
            right_valid = right_positions >= 0
            safe_left = np.where(left_valid, left_positions, 0)
            safe_right = np.where(right_valid, right_positions, 0)
            local = _backend_index_2d(
                self._arrays[site],
                safe_left,
                safe_right,
            )
            valid = _as_backend(
                left_valid[:, None] & right_valid[None, :],
                like=local,
            )
            local = ar.do(
                "where",
                valid[..., None, None],
                local,
                ar.do("zeros_like", local),
            )
            synthetic = _as_backend(
                batch["left_identity"][factor][:, None]
                & (~right_valid[None, :])
                & batch["right_identity"][factor][None, :],
                like=local,
            )
            if bool(np.any(batch["right_identity"][factor] & ~right_valid)):
                identity = ar.do("eye", self.phys_dim, like=reference)
                local = ar.do(
                    "where",
                    synthetic[..., None, None],
                    identity,
                    local,
                )
            product_block = (
                local
                if product_block is None
                else ar.do("matmul", product_block, local)
            )
        return product_block

    def _history_local_product_site_batch(self, site_plan, order):
        """Evaluate every selected Algorithm-3 product at one site together."""
        site = site_plan["site"]
        reference = self._arrays[site][0, 0]
        product_block = None
        for factor in range(order + 1):
            left_positions = site_plan["left_positions"][factor]
            right_positions = site_plan["right_positions"][factor]
            left_valid = left_positions >= 0
            right_valid = right_positions >= 0
            safe_left = np.where(left_valid, left_positions, 0)
            safe_right = np.where(right_valid, right_positions, 0)
            local = _backend_index_pairs(
                self._arrays[site],
                safe_left,
                safe_right,
            )
            valid = _as_backend(
                left_valid & right_valid,
                like=local,
            )
            local = ar.do(
                "where",
                valid[..., None, None],
                local,
                ar.do("zeros_like", local),
            )
            synthetic = _as_backend(
                site_plan["left_identity"][factor]
                & (~right_valid)
                & site_plan["right_identity"][factor],
                like=local,
            )
            if bool(
                np.any(
                    site_plan["right_identity"][factor] & ~right_valid,
                )
            ):
                identity = ar.do("eye", self.phys_dim, like=reference)
                local = ar.do(
                    "where",
                    synthetic[..., None, None],
                    identity,
                    local,
                )
            product_block = (
                local
                if product_block is None
                else ar.do("matmul", product_block, local)
            )
        return product_block

    def _stream_history_extension_site(
        self,
        arrays,
        site_plan,
        order,
        dt,
        *,
        chunk_size=65536,
    ):
        """Evaluate and scatter one site's Algorithm-3 batches in chunks."""
        if not isinstance(chunk_size, Integral) or int(chunk_size) < 1:
            raise ValueError("chunk_size must be a positive integer.")
        chunk_size = int(chunk_size)
        # Use a roughly square tile so neither side of the Cartesian product
        # becomes a long-lived array when one candidate list is much larger
        # than the other.
        left_chunk_size = max(1, int(chunk_size**0.5))
        for batch in site_plan["batches"]:
            left_size = int(batch["left_targets"].size)
            right_size = int(batch["right_targets"].size)
            right_chunk_size = max(
                1,
                chunk_size // min(left_chunk_size, max(1, left_size)),
            )
            for left_start in range(0, left_size, left_chunk_size):
                left_stop = min(left_start + left_chunk_size, left_size)
                left_slice = slice(left_start, left_stop)
                for right_start in range(0, right_size, right_chunk_size):
                    right_stop = min(right_start + right_chunk_size, right_size)
                    right_slice = slice(right_start, right_stop)
                    pair_batch = {
                        "site": batch["site"],
                        "left_targets": batch["left_targets"][left_slice],
                        "right_targets": batch["right_targets"][right_slice],
                        "left_positions": tuple(
                            values[left_slice]
                            for values in batch["left_positions"]
                        ),
                        "right_positions": tuple(
                            values[right_slice]
                            for values in batch["right_positions"]
                        ),
                        "left_identity": tuple(
                            values[left_slice]
                            for values in batch["left_identity"]
                        ),
                        "right_identity": tuple(
                            values[right_slice]
                            for values in batch["right_identity"]
                        ),
                    }
                    values = self._history_local_product_batch(
                        pair_batch,
                        order,
                    )
                    weights = _as_backend(
                        batch["left_weights"][left_slice, None]
                        * batch["right_weights"][None, right_slice],
                        like=values,
                    )
                    values = ar.do(
                        "multiply",
                        values,
                        weights[..., None, None],
                    )
                    arrays[batch["site"]] = _scatter_add_2d(
                        arrays[batch["site"]],
                        np.repeat(
                            pair_batch["left_targets"],
                            right_stop - right_start,
                        ),
                        np.tile(
                            pair_batch["right_targets"],
                            left_stop - left_start,
                        ),
                        _multiply_scalar(dt, values).reshape(
                            (-1, *values.shape[-2:])
                        ),
                    )

    def _algorithm_three_extension(
        self,
        arrays,
        levels,
        order,
        dt,
        *,
        cache_history=True,
        progress=None,
    ):
        """Add Algorithm 3's selected ``N + 1`` local history transitions.

        The topology-only insertion plan is cached; this numerical pass only
        evaluates selected local products and scatters them into the existing
        order-``N`` tensor.  Thus parameter rebinding does not reuse a stale
        backend value or autodiff graph.
        """
        plan, cache_hit = self._history_extension_plan(
            levels,
            order,
            cache_history=cache_history,
        )
        if progress is not None:
            progress.detail(
                f"A3 extension plan (order={order})",
                terms=plan["selected_terms"],
            )
        for site_plan in plan["site_plans"]:
            if site_plan is None:
                continue
            site = site_plan["site"]
            if plan.get("materialized", False):
                extension = self._history_local_product_site_batch(
                    site_plan,
                    order,
                )
                weights = _as_backend(site_plan["weights"], like=extension)
                extension = ar.do(
                    "multiply",
                    extension,
                    weights[..., None, None],
                )
                arrays[site] = _scatter_add_2d(
                    arrays[site],
                    site_plan["left_targets"],
                    site_plan["right_targets"],
                    _multiply_scalar(dt, extension),
                )
            else:
                self._stream_history_extension_site(
                    arrays,
                    site_plan,
                    order,
                    dt,
                )
            if progress is not None:
                progress.detail(
                    f"A3 extension site {site + 1}/{self.L}",
                    bond_dimensions=tuple(
                        len(bond_levels) for bond_levels in levels[1:-1]
                    ),
                    terms=plan["selected_terms"],
                )
        return plan["selected_terms"], cache_hit

    def _history_approximation_plan(self, levels, order, *, cache_history):
        """Compile Algorithm 4's ordered structural merge actions."""
        if cache_history:
            cached = self._history_approximation_plan_cache.get(order)
            if cached is not None:
                return cached, True

        working = [list(bond_levels) for bond_levels in levels]
        actions = []
        for bond in range(1, self.L + 1):
            while True:
                positions = self._history_level_positions(working[bond])
                source_history = None
                canonical = None
                number_of_threes = None
                for level in tuple(working[bond]):
                    history = level.history
                    if any(_level_number(token) == 1 for token in history):
                        continue
                    count = sum(
                        _level_number(token) == 3 for token in history
                    )
                    if count == 0:
                        continue
                    candidate = tuple(
                        MPOLevelToken(
                            1 if _level_number(token) == 3 else token.level,
                            token.payload,
                        )
                        for token in history
                    )
                    source = positions.get(history)
                    target = positions.get(candidate)
                    if source is None or target is None or source == target:
                        continue
                    source_history = history
                    canonical = candidate
                    number_of_threes = count
                    break
                if source_history is None:
                    break
                actions.append((
                    bond,
                    positions[source_history],
                    positions[canonical],
                    number_of_threes,
                ))
                working[bond].pop(positions[source_history])

        plan = {"actions": tuple(actions)}
        if cache_history:
            self._history_approximation_plan_cache[order] = plan
        return plan, False

    def _algorithm_four(
        self,
        arrays,
        levels,
        order,
        dt,
        *,
        cache_history=True,
        progress=None,
    ):
        """Apply the paper's order-controlled approximate compression.

        The merge schedule is structural and compiled once per Taylor order.
        Coefficients are evaluated during every numerical pass so ``dt`` can
        remain a backend value connected to an autodiff graph.
        """
        plan, cache_hit = self._history_approximation_plan(
            levels,
            order,
            cache_history=cache_history,
        )
        if progress is not None:
            progress.detail(
                f"A4 analytical plan (order={order})",
                actions=len(plan["actions"]),
            )
        actions_by_bond = [[] for _ in range(self.L + 1)]
        for bond, source, target, number_of_threes in plan["actions"]:
            actions_by_bond[bond].append(
                (source, target, number_of_threes),
            )

        removed = 0
        for bond, actions in enumerate(actions_by_bond):
            current = list(levels[bond])
            operations = []
            for source, target, number_of_threes in actions:
                # The approximation plan stores positions in the working
                # level list at the point where each action was compiled.
                # They therefore remain valid as long as we replay the
                # actions in their original order, just like the sequential
                # implementation did.
                if (
                    source >= len(current)
                    or target >= len(current)
                    or source == target
                ):
                    continue
                if number_of_threes <= order:
                    coefficient = (
                        factorial(order - number_of_threes)
                        / factorial(order)
                    )
                    power = number_of_threes
                else:
                    coefficient = 0.0
                    power = 0
                operations.append((source, target, True, power, coefficient))
                current.pop(source)

            if not operations:
                continue
            arrays[bond - 1] = self._apply_history_polynomial_groups(
                arrays[bond - 1],
                self._history_axis_polynomial_groups(
                    arrays[bond - 1].shape[1],
                    operations,
                ),
                dt,
                axis=1,
            )
            if bond < len(arrays):
                # A column elimination also removes the matching row from
                # the tensor on the right of the cut.  Keep this as a fused
                # gather so Algorithm 4 does not fall back to one full-tensor
                # copy per merge.
                arrays[bond] = self._apply_history_axis_groups(
                    arrays[bond],
                    self._history_axis_groups(
                        arrays[bond].shape[0],
                        [
                            (source, target, False)
                            for source, target, _merge, _power, _coefficient
                            in operations
                        ],
                    ),
                    axis=0,
                )
            levels[bond] = current
            removed += len(operations)
            if progress is not None:
                progress.detail(
                    f"A4 analytical replay bond {bond}/{self.L}",
                    bond_dimensions=tuple(
                        len(bond_levels) for bond_levels in levels[1:-1]
                    ),
                    merges=removed,
                )
        return removed, cache_hit

    def _contract_history_boundaries(self, arrays, levels, order):
        """Impose the finite-chain all-one boundary vectors.

        Boundary contraction is delayed until after Algorithms 1--4 because
        those algorithms need the histories at both open cuts.  This ordering
        is essential to the finite-chain implementation and avoids treating
        unreachable edge states as physical channels.
        """
        boundary_history = tuple(MPOLevelToken(1) for _ in range(order))
        left_target = self._history_level_positions(levels[0]).get(
            boundary_history,
        )
        right_target = self._history_level_positions(levels[-1]).get(
            boundary_history,
        )
        if left_target is None or right_target is None:
            raise ValueError("history construction lost a finite boundary state.")

        first = arrays[0]
        arrays[0] = (
            first.select_axis(0, left_target)
            if isinstance(first, SparseVirtualTensor)
            else _stack([first[left_target]], axis=0)
        )
        levels[0] = [MPOLevel(("boundary", "left", order), boundary_history)]

        last = arrays[-1]
        arrays[-1] = (
            last.select_axis(1, right_target)
            if isinstance(last, SparseVirtualTensor)
            else _stack([last[:, right_target]], axis=1)
        )
        levels[-1] = [MPOLevel(("boundary", "right", order), boundary_history)]

    def _extensive_history_exponential(
        self,
        dt,
        *,
        order,
        extend=False,
        approximate=False,
        mode="base",
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=None,
        extension_budget=None,
        estimated_extension_terms=None,
        mode_reason=None,
    ):
        """Construct an arbitrary-order MPO using Algorithms 1--4.

        This is the single multi-site execution path.  The order of passes is
        part of the paper implementation contract: optional Algorithm 3 first
        adds selected next-order terms, Algorithm 1 rewires extensive
        prefactors, Algorithm 2 performs exact compression, and optional
        Algorithm 4 applies the analytical approximation.
        """
        direct_reduced = history_storage == "reduced"
        if direct_reduced:
            reduced_stage = (
                f"history + A1-A{3 if extend else 2} (reduced)"
            )
            if progress is not None:
                progress.start(reduced_stage)
            arrays, levels, history_cache_hit, storage_info = (
                self._reduced_history_power_data(
                    dt,
                    order=order,
                    extend=extend,
                    max_bond=max_bond,
                    on_exceed=on_exceed,
                    cache_history=cache_history,
                    progress=progress,
                )
            )
            if progress is not None:
                progress.finish(
                    reduced_stage,
                    bond_dimensions=tuple(
                        len(bond_levels) for bond_levels in levels[1:-1]
                    ),
                )
            initial_bond_dimensions = storage_info["initial_bond_dimensions"]
            compression_plan_cache_hit = history_cache_hit
            extension_plan_cache_hit = history_cache_hit if extend else False
            extension_terms = storage_info["extension_terms"]
            exact_merges = tuple(
                {"mode": "direct-reduced"}
                for _ in range(storage_info["exact_history_merges"])
            )
        else:
            if progress is not None:
                progress.start("history")
            arrays, levels, history_cache_hit, storage_info = self._history_power_data(
                order,
                max_bond=max_bond,
                on_exceed=on_exceed,
                cache_history=cache_history,
                history_storage=history_storage,
                progress=progress,
            )
            initial_bond_dimensions = tuple(
                len(bond_levels) for bond_levels in levels[1:-1]
            )
            if progress is not None:
                progress.finish(
                    "history",
                    bond_dimensions=initial_bond_dimensions,
                )
            compression_plan, compression_plan_cache_hit = (
                self._history_compression_plan(
                    levels,
                    order,
                    cache_history=cache_history,
                )
            )
            extension_terms = 0
            extension_plan_cache_hit = False
            if extend:
                if progress is not None:
                    progress.start("A3 extension")
                extension_terms, extension_plan_cache_hit = (
                    self._algorithm_three_extension(
                        arrays,
                        levels,
                        order,
                        dt,
                        cache_history=cache_history,
                        progress=progress,
                    )
                )
                if progress is not None:
                    progress.finish(
                        "A3 extension",
                        bond_dimensions=tuple(
                            len(bond_levels) for bond_levels in levels[1:-1]
                        ),
                    )
            if progress is not None:
                progress.start("A1 prefactor")
            self._algorithm_one(
                arrays,
                levels,
                order,
                dt,
                plan=compression_plan,
            )
            if progress is not None:
                progress.finish(
                    "A1 prefactor",
                    bond_dimensions=tuple(
                        len(bond_levels) for bond_levels in levels[1:-1]
                    ),
                )
                progress.start("A2 exact analytical-compress")
            exact_merges = self._algorithm_two(
                arrays,
                levels,
                plan=compression_plan,
            )
            if progress is not None:
                progress.finish(
                    "A2 exact analytical-compress",
                    bond_dimensions=tuple(
                        len(bond_levels) for bond_levels in levels[1:-1]
                    ),
                )
        approximate_merges = 0
        approximation_plan_cache_hit = False
        if approximate:
            if progress is not None:
                progress.start("A4 analytical-compress")
            approximate_merges, approximation_plan_cache_hit = (
                self._algorithm_four(
                    arrays,
                    levels,
                    order,
                    dt,
                    cache_history=cache_history,
                    progress=progress,
                )
            )
            if progress is not None:
                progress.finish(
                    "A4 analytical-compress",
                    bond_dimensions=tuple(
                        len(bond_levels) for bond_levels in levels[1:-1]
                    ),
                )
        if progress is not None:
            progress.start("boundary")
        self._contract_history_boundaries(arrays, levels, order)
        final_bond_dimensions = tuple(
            len(bond_levels) for bond_levels in levels[1:-1]
        )
        if progress is not None:
            progress.finish(
                "boundary",
                bond_dimensions=final_bond_dimensions,
            )
        if all(isinstance(array, SparseVirtualTensor) for array in arrays):
            storage_info = {
                **storage_info,
                "final_stored_blocks": sum(
                    array.stored_blocks for array in arrays
                ),
                "final_dense_virtual_blocks": sum(
                    array.shape[0] * array.shape[1] for array in arrays
                ),
            }
        metadata = {
            "operation": "extensive_exponential",
            "dt": dt,
            "order": order,
            "mode": mode,
            "max_bond": max_bond,
            "on_exceed": on_exceed,
            "cache_history": bool(cache_history),
            "history_storage": storage_info["mode"],
            "history_storage_requested": history_storage,
            "history_storage_blocks": storage_info,
            "tensor_plan_cache_hit": storage_info.get(
                "tensor_plan_cache_hit",
                False,
            ),
            "initial_bond_dimensions": initial_bond_dimensions,
            "history_generation": (
                "reachable" if self._structural_transitions is not None else "cartesian"
            ),
            "history_cache_hit": history_cache_hit,
            "compression_plan_cache_hit": compression_plan_cache_hit,
            "extension_plan_cache_hit": extension_plan_cache_hit,
            "approximation_plan_cache_hit": approximation_plan_cache_hit,
            "algorithms": (1, 2) + ((3,) if extend else ()) + ((4,) if approximate else ()),
            "exact_history_merges": len(exact_merges),
            "approximate_history_merges": approximate_merges,
            "extension_terms": extension_terms,
            "estimated_extension_terms": estimated_extension_terms,
            "extension_budget": extension_budget,
            "mode_reason": mode_reason,
            "approximate": bool(approximate),
            "analytical_compression": (
                "folded" if approximate else "algorithms1-2"
            ),
            "numerical_compression": "none",
        }
        output = type(self)(
            arrays,
            levels=levels,
            degree=order,
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=metadata,
        )
        output._block_plan = MPOBlockPlan.from_semantic(
            output,
            kind="history",
            metadata={
                "order": order,
                "mode": mode,
                "history_storage": storage_info["mode"],
                "structural_sparsity": (
                    "block" if output.is_block_sparse else "conservative"
                ),
            },
        )
        output.metadata["block_plan"] = output._block_plan.summary()
        report = MPOCompressionReport(
            method="paper-history",
            exact=not approximate,
            initial_bond_dimensions=initial_bond_dimensions,
            final_bond_dimensions=final_bond_dimensions,
            merged_channels=len(exact_merges) + approximate_merges,
            merges=tuple(exact_merges),
        )
        output.compression_report = report
        output.metadata["compression_report"] = report
        if not cache_history:
            # One-off builds should not leave the raw topology, insertion
            # states, or execution plans attached to the source object.
            self._history_topology_cache.pop(order, None)
            self._history_symbolic_cache = None
            self._history_extension_plan_cache.pop(order, None)
            self._history_compression_plan_cache.pop(order, None)
            self._history_approximation_plan_cache.pop(order, None)
            self._history_tensor_plan_cache.pop((order, True), None)
            self._history_tensor_plan_cache.pop((order, False), None)
            self._history_reduced_plan_cache.pop(order, None)
        return output

    def _first_degree_structure(self):
        """Validate and return active channel counts on each virtual bond."""
        if self.L == 1:
            return (0,)
        if _level_number(self._levels[0][0].history[0]) != 1:
            raise ValueError("the left MPO boundary must be level 1.")
        if _level_number(self._levels[-1][0].history[0]) != 3:
            raise ValueError("the Hamiltonian MPO right boundary must be level 3.")
        active_counts = []
        for bond, levels in enumerate(self._levels[1:-1], start=1):
            numbers = [_level_number(level.history[0]) for level in levels]
            if numbers.count(1) != 1 or numbers.count(3) != 1:
                raise ValueError(
                    "extensive_exponential requires one level-1 and one level-3 "
                    f"rail on internal bond {bond}, got {numbers!r}."
                )
            if any(number not in (1, 2, 3) for number in numbers):
                raise ValueError("first-degree virtual levels must be 1, 2, or 3.")
            active_counts.append(numbers.count(2))
        return (0, *active_counts, 0)

    def extensive_exponential(
        self,
        dt,
        *,
        order=1,
        extend=False,
        approximate=False,
        mode=None,
        max_bond=None,
        on_exceed="raise",
        cache_history=True,
        history_storage="auto",
        progress=False,
        extension_budget=None,
    ):
        """Build the paper's size-extensive higher-order MPO.

        The construction is local in the MPO tensors.  It contracts only
        physical operator blocks at each site and assembles the new virtual
        channels with ``stack``; it never forms ``H`` or ``exp(dt * H)`` as a
        global dense matrix.

        Parameters
        ----------
        dt : scalar
            Time-step or imaginary-time parameter ``tau``.
        order : int, default=1
            Taylor order. Multi-site chains use the generic history engine for
            every positive order. One-site chains use a direct local Taylor
            polynomial at arbitrary order.
        extend : bool, default=False
            Include Algorithm 3's selected order ``N + 1`` terms without
            increasing the analytical history bond dimension.
        approximate : bool, default=False
            Apply Algorithm 4's order-controlled analytical compression after
            exact history compression. This is not a numerical cutoff.
        mode : {None, "base", "exact", "folded", "hybrid", "auto"}, optional
            Named construction policy. ``"base"`` selects Algorithms 1--2,
            ``"exact"`` selects the paper's exact extended construction
            (Algorithms 1--3), ``"folded"`` selects Algorithms 1, 2, and 4,
            and ``"hybrid"`` selects Algorithms 1--4. ``"auto"`` uses the
            exact policy when the symbolic Algorithm-3 work estimate is within
            ``extension_budget`` and the folded policy otherwise. When omitted,
            the legacy ``extend`` and
            ``approximate`` flags are used unchanged. The compatibility names
            ``"algorithm4"``, ``"optimal"``, ``"approximate"`` and their
            ``"paper_*"`` spellings are accepted and normalized to the
            canonical names in metadata.
        max_bond : int, optional
            Maximum temporary history bond dimension allowed during
            construction. This guard applies before exact history compression
            and therefore protects the allocation stage. ``None`` disables
            the guard.
        on_exceed : {"raise", "warn", "ignore"}, default="raise"
            Action when ``max_bond`` is exceeded. ``"raise"`` stops before
            completing the oversized history table, ``"warn"`` continues,
            and ``"ignore"`` disables the action while retaining the value in
            metadata.
        cache_history : bool, default=True
            Retain the raw reachable history topology for later calls. Set
            this to ``False`` for large-order or one-off constructions so the
            topology is released after the current MPO is assembled. With the
            default ``history_storage="auto"``, this also selects the
            streaming local-history builder.
        history_storage : {"auto", "dense", "sparse", "streaming", "block_sparse", "reduced"}, default="auto"
            Storage policy for temporary raw-history tensors. ``"dense"``
            retains all structural local pairs. ``"sparse"`` skips
            structurally impossible local transition products and batches the
            remaining physical block products. ``"streaming"`` keeps the
            topology ephemeral when ``cache_history=False``. For an
            automaton-built MPO, ``"auto"`` selects the structural sparse path
            for cached builds and the compatibility streaming path otherwise;
            direct MPO construction retains the dense/streaming policy. The
            ``"block_sparse"`` retains structurally present operator-valued
            virtual blocks through Algorithms 1--4. With configured symmetry
            metadata, ``"auto"`` selects this path and :meth:`to_mpo` compiles
            the result directly into native Symmray charge blocks.
            ``"reduced"`` streams raw history products directly into the
            Algorithms-1/2 quotient and never allocates raw virtual tensors.
        progress : bool, default=False
            Show a ``tqdm`` stage bar with elapsed seconds and the current MPO
            bond dimensions. This reports the requested order as one history
            stage; it does not redundantly rebuild orders one through
            ``order``.
        extension_budget : int, optional
            Maximum number of symbolically selected Algorithm-3 extension
            terms allowed by ``mode="auto"``. The estimate is made before
            the numerical left/right pair plan is materialized. ``None``
            uses the deterministic default budget of 1,024 terms. Auto mode
            selects Algorithms 1--3 when the estimate is less than or equal
            to this budget, and Algorithms 1, 2, and 4 otherwise.

        Notes
        -----
        Native Symmray compilation currently supports neutral bosonic Abelian
        ``U1``, ``Z2``, ``U1U1``, and ``Z2Z2`` operators with NumPy local
        blocks. Graded fermionic histories remain a separate sign-preserving
        backend. The one-site path is exact through its requested local Taylor
        order; Algorithm 3/4 have no virtual history to extend or merge there.
        """
        _check_scalar(dt, name="dt")
        if not isinstance(order, Integral) or int(order) < 1:
            raise ValueError("order must be a positive integer.")
        order = int(order)
        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        if on_exceed not in {"raise", "warn", "ignore"}:
            raise ValueError(
                "on_exceed must be one of 'raise', 'warn', or 'ignore'."
            )
        if not isinstance(cache_history, bool):
            raise TypeError("cache_history must be a boolean.")
        if extension_budget is None:
            extension_budget = _DEFAULT_AUTO_EXTENSION_BUDGET
        elif (
            not isinstance(extension_budget, Integral)
            or int(extension_budget) < 1
        ):
            raise ValueError(
                "extension_budget must be a positive integer or None."
            )
        else:
            extension_budget = int(extension_budget)
        if history_storage == "blocks":
            history_storage = "block_sparse"
        if history_storage not in {
            "auto", "dense", "sparse", "streaming", "block_sparse", "reduced",
        }:
            raise ValueError(
                "history_storage must be one of 'auto', 'dense', 'sparse', "
                "'streaming', 'block_sparse', or 'reduced'."
            )
        if history_storage == "streaming" and cache_history:
            raise ValueError(
                "history_storage='streaming' requires cache_history=False; "
                "use history_storage='auto' for cached construction."
            )
        estimated_extension_terms = None
        mode_reason = None
        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError("mode must be a string or None.")
            mode_aliases = {
                "base": (False, False, "base"),
                "exact": (True, False, "exact"),
                "folded": (False, True, "folded"),
                "hybrid": (True, True, "hybrid"),
                "algorithm4": (False, True, "folded"),
                "paper_algorithm4": (False, True, "folded"),
                "optimal": (True, False, "exact"),
                "paper_optimal": (True, False, "exact"),
                "approximate": (True, True, "hybrid"),
                "paper_approximate": (True, True, "hybrid"),
            }
            requested_mode = mode
            if mode == "auto":
                if extend or approximate:
                    raise ValueError(
                        "mode='auto' cannot be combined with extend or "
                        "approximate flags."
                    )
                if self.L > 1:
                    state_lists, _ = self._history_topology(
                        order,
                        max_bond=max_bond,
                        on_exceed=on_exceed,
                        cache_history=cache_history,
                        progress=None,
                    )
                    estimated_extension_terms = (
                        self._estimate_history_extension_terms(
                            state_lists,
                            order,
                        )
                    )
                else:
                    # With no virtual bonds Algorithm 3 is represented by
                    # one additional local Taylor term.
                    estimated_extension_terms = 1
                extend = estimated_extension_terms <= extension_budget
                approximate = not extend
                canonical_mode = "exact" if extend else "folded"
                mode_reason = (
                    "within extension budget"
                    if extend
                    else "extension budget exceeded"
                )
            else:
                try:
                    mode_extend, mode_approximate, canonical_mode = mode_aliases[mode]
                except KeyError as exc:
                    allowed = ", ".join(
                        ["base", "exact", "folded", "hybrid", "auto"]
                        + sorted(
                            name for name in mode_aliases
                            if name not in {"base", "exact", "folded", "hybrid"}
                        )
                    )
                    raise ValueError(
                        f"unknown mode {mode!r}; expected one of {allowed}."
                    ) from exc
                if extend or approximate:
                    raise ValueError(
                        "mode cannot be combined with extend or approximate flags."
                    )
                extend = mode_extend
                approximate = mode_approximate
        else:
            canonical_mode = (
                "hybrid" if approximate and extend
                else "exact" if extend
                else "folded" if approximate
                else "base"
            )
            requested_mode = None
        progress, owns_progress = _make_mpo_progress(
            progress,
            order=order,
        )
        if self.L > 1:
            output = self._extensive_history_exponential(
                dt,
                order=order,
                extend=extend,
                approximate=approximate,
                mode=canonical_mode,
                max_bond=max_bond,
                on_exceed=on_exceed,
                cache_history=cache_history,
                history_storage=history_storage,
                progress=progress,
                extension_budget=(
                    extension_budget if requested_mode == "auto" else None
                ),
                estimated_extension_terms=estimated_extension_terms,
                mode_reason=mode_reason,
            )
            if progress.enabled:
                output.metadata["progress"] = True
                output.metadata["timings"] = dict(progress.timings)
                output.metadata["timing_history"] = {
                    int(order): dict(progress.timings),
                }
                output.metadata["order_seconds"] = progress.total_seconds
            if requested_mode == "auto":
                output.metadata["requested_mode"] = requested_mode
            if owns_progress:
                progress.close()
            return output
        # A one-site operator has no non-trivial virtual history. Evaluate its
        # requested Taylor polynomial directly, with Algorithm 3's extension
        # represented by one additional local Taylor term.
        effective_order = order + int(extend)
        progress.start(f"local Taylor order={effective_order}")
        reference = self._arrays[0][0, 0]
        identity = ar.do("eye", self.phys_dim, like=reference)
        h = reference
        data = identity
        power = identity
        for power_order in range(1, effective_order + 1):
            power = ar.do("matmul", power, h)
            data = data + _multiply_scalar(
                dt ** power_order / factorial(power_order),
                power,
            )
        output = type(self)(
            (data,),
            levels=[[
                MPOLevel(
                    ("extensive", effective_order, 0, ("11", None, None)),
                    tuple(
                        self._levels[0][0].history[0]
                        for _ in range(effective_order)
                    ),
                )
            ]] * 2,
            degree=effective_order,
            **self._symmetry_options(),
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                "operation": "extensive_exponential",
                "dt": dt,
                "order": effective_order,
                "requested_order": order,
                "mode": canonical_mode,
                "max_bond": max_bond,
                "on_exceed": on_exceed,
                "cache_history": bool(cache_history),
                "history_storage": "one-site-local",
                "history_storage_requested": history_storage,
                "algorithms": ("one-site-taylor",),
                "approximate": False,
                "approximation_requested": bool(approximate),
                "extension_requested": bool(extend),
                "estimated_extension_terms": estimated_extension_terms,
                "extension_budget": (
                    extension_budget if requested_mode == "auto" else None
                ),
                "mode_reason": mode_reason,
            },
        )
        output._block_plan = MPOBlockPlan.from_semantic(
            output,
            kind="history",
            metadata={
                "order": effective_order,
                "mode": canonical_mode,
                "history_storage": "one-site-local",
                "structural_sparsity": "conservative",
            },
        )
        output.metadata["block_plan"] = output._block_plan.summary()
        if requested_mode == "auto":
            output.metadata["requested_mode"] = requested_mode
        progress.finish(
            f"local Taylor order={effective_order}",
            bond_dimensions=(),
        )
        if progress.enabled:
            output.metadata["progress"] = True
            output.metadata["timings"] = dict(progress.timings)
            output.metadata["timing_history"] = {
                int(effective_order): dict(progress.timings),
            }
            output.metadata["order_seconds"] = progress.total_seconds
        if owns_progress:
            progress.close()
        return output

    def compress_exact(self, *, inplace=False):
        """Apply exact history/column compression without a numerical cutoff.

        The candidate histories follow the paper: move level-1 tokens to the
        front for column equivalence and level-3 tokens to the front for row
        equivalence.  A candidate is accepted only when the corresponding
        operator-valued rows or columns are exactly equal, so this method is
        conservative and cannot introduce a truncation error.
        """
        target = self if inplace else self.copy()
        # Exact compression changes the virtual state set. Do not let a
        # copied plan describe the pre-compression bond structure.
        target._block_plan = None
        if target.is_block_sparse:
            # The standalone exact compressor predates the sparse history
            # executor and performs row/column equality checks by slicing.
            # Materialize only for this explicit compatibility operation;
            # Algorithms 1--4 themselves remain sparse.
            target._arrays = target.arrays
        initial = target.bond_dimensions
        merges = []
        skipped = 0

        for bond in range(1, target.L):
            changed = True
            while changed:
                changed = False
                levels = target._levels[bond]
                for source_pos, source in enumerate(levels):
                    history = source.history
                    n1 = sum(_level_number(token) == 1 for token in history)
                    n3 = sum(_level_number(token) == 3 for token in history)
                    if n3 <= n1:
                        canonical = _move_level_front(history, 1)
                        mode = "column"
                    else:
                        canonical = _move_level_front(history, 3)
                        mode = "row"
                    if canonical == history:
                        continue
                    target_pos = next(
                        (
                            pos
                            for pos, candidate in enumerate(levels)
                            if pos != source_pos and candidate.history == canonical
                        ),
                        None,
                    )
                    if target_pos is None:
                        skipped += 1
                        continue
                    # A history match alone is not enough: the corresponding
                    # operator-valued row or column must also be identical.
                    # This conservative check is why this stage is exact and
                    # has no numerical cutoff or hidden approximation.
                    if target._try_merge(bond, source_pos, target_pos, mode):
                        merges.append({
                            "bond": bond - 1,
                            "source": source.label,
                            "target": levels[target_pos].label,
                            "mode": mode,
                            "history": history,
                        })
                        changed = True
                        break
                    skipped += 1

        report = MPOCompressionReport(
            method="exact-history",
            exact=True,
            initial_bond_dimensions=tuple(initial),
            final_bond_dimensions=target.bond_dimensions,
            merged_channels=len(merges),
            merges=tuple(merges),
            skipped_candidates=skipped,
        )
        target.metadata["compression_report"] = report
        target.compression_report = report
        return target

    def _try_merge(self, bond, source, target, mode):
        """Try one exact scalar gauge elimination on a virtual bond."""
        if source == target:
            return False
        left = self._arrays[bond - 1]
        right = self._arrays[bond]
        if mode == "column":
            source_block = left[:, source]
            target_block = left[:, target]
            if not _array_equal(source_block, target_block):
                return False
            left_blocks = [left[:, pos] for pos in range(left.shape[1])]
            left_blocks[source] = left_blocks[source] - left_blocks[target]
            right_blocks = [right[pos] for pos in range(right.shape[0])]
            right_blocks[target] = right_blocks[target] + right_blocks[source]
        else:
            source_block = right[source]
            target_block = right[target]
            if not _array_equal(source_block, target_block):
                return False
            left_blocks = [left[:, pos] for pos in range(left.shape[1])]
            left_blocks[target] = left_blocks[target] + left_blocks[source]
            right_blocks = [right[pos] for pos in range(right.shape[0])]
            right_blocks[source] = right_blocks[source] - right_blocks[target]

        left_blocks = [block for pos, block in enumerate(left_blocks) if pos != source]
        right_blocks = [block for pos, block in enumerate(right_blocks) if pos != source]
        self._arrays = (
            *self._arrays[: bond - 1],
            _stack(left_blocks, axis=1),
            _stack(right_blocks, axis=0),
            *self._arrays[bond + 1 :],
        )
        self._levels[bond].pop(source)
        self._base_level_position_cache = None
        self._block_plan = None
        self._validate()
        return True

    def _check_compatible(self, other):
        if not isinstance(other, FirstDegreeMPO):
            raise TypeError("other must be a FirstDegreeMPO.")
        if self.L != other.L:
            raise ValueError(f"MPO lengths differ: {self.L} and {other.L}.")
        if self.phys_dim != other.phys_dim:
            raise ValueError(
                f"MPO physical dimensions differ: {self.phys_dim} and {other.phys_dim}."
            )
        if self._symmetry_options() != other._symmetry_options():
            raise ValueError("MPO symmetry metadata must match.")

    def __add__(self, other):
        return self.add(other)

    def __matmul__(self, other):
        return self.non_disjoint_product(other)

    def __mul__(self, coefficient):
        return self.scale(coefficient)

    def __rmul__(self, coefficient):
        return self.scale(coefficient)
